import os
import glob
import torch
import heapq
import time
from typing import List, Dict, Tuple, Any, Union
from collections import defaultdict, Counter
from itertools import combinations, permutations

from moe.path_config import *
from moe.activation_extraction.activation_extractor_utils import OLMoEActivationExtractor

def load_all_activations(
    datasets_to_process: List[Tuple[str, bool]],
    layer_idx: int,
    extractor,
    extract_fn: str = "extract_topk_routing_batch",
    base_save_dir: str = SAVE_ACTS_PATH_DIR,
    split_names: List[str] = ["train", "test"],
    verbose: bool = True,
    simple_keyname: bool = True
) -> Dict[str, object]:
    """
    Load activations for multiple datasets and splits.

    Args:
        datasets_to_process: List of (dataset_name, question_only) tuples
        extractor: Initialized OLMoEActivationExtractor
        extract_fn: Extraction function to use
                        ("extract_activations_mlp_batch", 
                        "extract_topk_routing_batch", 
                        "extract_prerouting_and_combined_output_batch")
        base_save_dir: Save to (base_save_dir, extract_fn, dataset, ...)
        split_names: List of splits to load (default=["train", "test"])
        verbose: Whether to print missing file warnings
        simple_keyname: If True, use dataset_name as dict key; else include split and suffix

    Returns:
        Dictionary mapping keys like 'gsm8k_train_questions' to activation objects
    """
    activations = {}

    for dataset_name, question_only in datasets_to_process:
        question_suffix = "_questions" if question_only else "_full"
        dataset_dir = os.path.join(base_save_dir, extract_fn, dataset_name)

        for split in split_names:
            # Pattern to match: e.g., mbpp_train_full_XXXsamples.pt
            pattern = f"L{layer_idx}_{dataset_name}_{split}{question_suffix}_*samples.pt"
            search_path = os.path.join(dataset_dir, pattern)

            matched_files = glob.glob(search_path)
            if matched_files:
                filepath = matched_files[0]  # Use the first match
                if len(split_names) == 1 and simple_keyname:
                    key = dataset_name
                else:
                    key = f"{dataset_name}_{split}_{question_suffix.lstrip('_')}"
                activations[key] = extractor.load_activations(filepath)
            else:
                if verbose:
                    print(f"[Warning] No activation file found for pattern: {search_path}")

    return activations

def compute_per_dataset_token_counts(d):
    counts = Counter()
    for dataset_name, samples in d.items():
        for tokens_per_sample in samples["tokens"]:
            counts[dataset_name] += len(tokens_per_sample)
    return counts

def compute_expert_set_counts(d, subset_sizes=[1,2,3,4]):
    """
    Args:
        d: Dict[str, object]
        subset_sizes: List of candidate subset sizes i
            For Combinations, C(64, i) possible index sets
            For Permutations, P(64, i) possible permutations
    Returns:
        combinations_, permutations_order_sensitive_, permutations_order_insensitive_: Tuple(dict[str, counter])
            dataset_name --> Counter({expert_index_tuple: count})

    """
    combinations_ = defaultdict(lambda: Counter())
    permutations_order_sensitive_ = defaultdict(lambda: Counter())
    permutations_order_insensitive_ = defaultdict(lambda: Counter())

    for dataset_name, samples in d.items():
        for topk_indices in samples['topk_indices']:  # shape [tokens, k]
            for experts in topk_indices:
                orig_experts = experts.tolist()
                sorted_experts = sorted(orig_experts)

                for s in subset_sizes:
                    # combinations: any subset of the sorted set
                    for combo in combinations(sorted_experts, s):
                        combinations_[combo][dataset_name] += 1

                    # order-sensitive: prefix of original order
                    permutations_order_sensitive_[tuple(orig_experts[:s])][dataset_name] += 1

                    # order-insensitive: sorted prefix only
                    permutations_order_insensitive_[tuple(sorted(orig_experts[:s]))][dataset_name] += 1

    return combinations_, permutations_order_sensitive_, permutations_order_insensitive_

def find_expert_coactivations(
    datasets: Union[Dict[str, List], List[Dict[str, List]]],
    subset_size: int = 2,
    mode: str = "combinations",
    keep_n: int = 5,
) -> Dict[Tuple[int, ...], Dict]:
    """
    Find most frequent expert co-activations

    Args:
        datasets: A single dataset dict or a list of dataset dicts.
        subset_size: Size of subset K
        mode: One of {"combinations", "permutations_order_sensitive", "permutations_order_insensitive"}
        keep_n: How many top examples (by score) to keep per expert set
    """
    if isinstance(datasets, dict):
        datasets = [datasets]

    coactivation_stats = defaultdict(lambda: {"count": 0, "top_examples_heap": []})
    counter = 0

    for dataset in datasets:
        dataset_name = dataset["dataset_name"]
        split = dataset["split"]
        question_only = dataset["question_only"]
        texts = dataset["texts"] # List[str]
        tokens_list = dataset["tokens"] # List[List[str, length = seq_len]]
        indices_list = dataset["topk_indices"] # List[torch.Tensor(seq_len, 8)]
        scores_list = dataset["topk_scores"] # List[torch.Tensor(seq_len, 8)]

        for sample_idx, (text, tokens, topk_idx, topk_val) in enumerate(
            zip(texts, tokens_list, indices_list, scores_list)
        ):
            for pos, (tok, experts_arr, scores_arr) in enumerate(zip(tokens, topk_idx, topk_val)):
                if isinstance(experts_arr, torch.Tensor):
                    experts_arr = experts_arr.cpu()
                if isinstance(scores_arr, torch.Tensor):
                    scores_arr = scores_arr.cpu()

                experts = experts_arr.tolist()
                scores = scores_arr.tolist()

                if len(experts) < subset_size:
                    continue

                # Determine key based on mode
                if mode == "permutations_order_sensitive":
                    key = tuple(experts[:subset_size])

                elif mode == "permutations_order_insensitive":
                    key = tuple(sorted(experts[:subset_size]))

                elif mode == "combinations":
                    # Generate all subsets of size subset_size from the top-8 set
                    for combo in combinations(experts, subset_size):
                        key = tuple(sorted(combo))
                        combined_score = sum(scores[i] for i, e in enumerate(experts) if e in combo)

                        entry = coactivation_stats[key]
                        entry["count"] += 1

                        example = {
                            "dataset": dataset_name,
                            "split": split,
                            "question_only": question_only,
                            "sample_idx": sample_idx,
                            "text": text,
                            "token": tok,
                            "tokens": tokens,
                            "position": pos,
                            "experts": experts,
                            "scores": scores,
                            "combined_score": combined_score,
                        }

                        heapq.heappush(entry["top_examples_heap"], (combined_score, counter, example))
                        counter += 1
                        if len(entry["top_examples_heap"]) > keep_n:
                            heapq.heappop(entry["top_examples_heap"])
                    continue  # skip rest, already handled

                else:
                    raise ValueError(f"Invalid mode: {mode}")

                # For permutations: compute score as sum of prefix scores
                combined_score = sum(scores[:subset_size])

                entry = coactivation_stats[key]
                entry["count"] += 1

                example = {
                    "dataset": dataset_name,
                    "split": split,
                    "question_only": question_only,
                    "sample_idx": sample_idx,
                    "text": text,
                    "token": tok,
                    "tokens": tokens,
                    "position": pos,
                    "experts": experts,
                    "scores": scores,
                    "combined_score": combined_score,
                }

                heapq.heappush(entry["top_examples_heap"], (combined_score, counter, example))
                counter += 1
                if len(entry["top_examples_heap"]) > keep_n:
                    heapq.heappop(entry["top_examples_heap"])

    # Finalize
    final_output = {}
    for key, value in coactivation_stats.items():
        sorted_examples = sorted(value["top_examples_heap"], key=lambda x: x[0], reverse=True)
        final_output[key] = {
            "count": value["count"],
            "top_examples": [item[2] for item in sorted_examples],
        }

    return final_output

def format_expert_coactivations(
    coactivation_stats: Dict[Tuple[int, ...], Dict],
    tokenizer,
    context_before: int = 20,
    context_after: int = 6
) -> None:
    """
    In-place formatting of the coactivation report

    Args:
        coactivation_stats: Output of find_expert_coactivations
        tokenizer: HuggingFace tokenizer (same one used for extraction)
        context_before: how many tokens to show before target
        context_after: how many tokens to show after target
    """
    for _, info in coactivation_stats.items():
        formatted_examples = []
        for ex in info["top_examples"]:
            tokens = ex["tokens"]
            pos = ex["position"]

            context_str_preceeding = tokenizer.decode(
                tokenizer.convert_tokens_to_ids(tokens[max(0, pos - context_before): pos])
            )

            formatted_token = tokenizer.decode(
                tokenizer.convert_tokens_to_ids([tokens[pos]])
            )

            context_str_following = tokenizer.decode(
                tokenizer.convert_tokens_to_ids(tokens[pos + 1: pos + context_after])
            )
            example = {
                # sample-level info
                # "raw_token": ex["token"],
                "formatted_token": formatted_token,
                "preceeding": context_str_preceeding,
                "following": context_str_following,
                "position": pos,
                "combined_score": ex["combined_score"],
                # dataset-level info
                "dataset": ex["dataset"],
                "split": ex["split"],
                "question_only": ex["question_only"],
                "sample_idx": ex["sample_idx"],
                "experts": ex["experts"],
                # "scores": ex["scores"]
            }
            formatted_examples.append(example)
        info["top_examples"] = formatted_examples

# datasets_to_process = [
#     # False ==> Question: {Q} [Optional: {Choices} or {Reasoning}] Answer:{A}
#     ("gsm8k", False),
#     ("arc-easy", False),
#     ("arc-challenge", False),
#     ("sciq", False),
#     ("mbpp", False),
#     # False ==> Not a question. Just the {statement}
#     ("ag_news", False),
#     ("imdb_pos", False),
#     ("imdb_neg", False),
#     # False/True does not matter for these. Just the {statement}
#     ("poetry", False),
#     ("lex_glue", False),
#     ("arxiv", False),
#     ("personas", False)
# ]
# start_time = time.time()
# extractor = OLMoEActivationExtractor()

# LAYER_IDX = 7

# d = load_all_activations(
#     datasets_to_process,
#     LAYER_IDX,
#     extractor,
#     split_names = ["train"]
# )
# counts = compute_per_dataset_token_counts(d)
# print(counts, counts.total())
# keys = list(d.keys())
# print(keys)
# print(extractor.verify_saved_activations(d[keys[0]]))

# combinations_, permutations_order_sensitive_, permutations_order_insensitive_ = compute_expert_set_counts(d)

# # print("------------ combinations ------------")
# # keys = list(combinations_.keys())[5000:5005]
# # for key in keys:
# #     print(key, combinations_[key])
# # print("------------ permutations ------------")
# # keys = list(permutations_.keys())[5000:5005]
# # for key in keys:
# #     print(key, permutations_[key])

# d1 = find_expert_coactivations(
#     [v for v in d.values()],
#     subset_size = 2,
#     mode = "combinations",
#     keep_n = 3,
# )
# d2 = find_expert_coactivations(
#     [v for v in d.values()],
#     subset_size = 3,
#     mode = "combinations",
#     keep_n = 3,
# )
# cnt = 0
# for k, v in d1.items():
#     print(f"======================= {k} ======================")
#     print(combinations_[k])
#     print(k, v)
#     cnt += 1
#     if cnt > 2:
#         break

# cnt = 0
# for k, v in d2.items():
#     print(f"======================= {k} ======================")
#     print(combinations_[k])
#     print(k, v)
#     cnt += 1
#     if cnt > 2:
#         break

# print(f"Time Elapsed: {time.time() - start_time}")