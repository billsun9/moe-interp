import os
import torch
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Any, Union
from collections import defaultdict, Counter
import heapq

from moe.activation_extraction.activation_extractor_utils import OLMoEActivationExtractor
from moe.path_config import SAVE_ACTS_PATH_DIR

def get_frequencies(
    datasets: Union[Dict[str, List], List[Dict[str, List]]],
    layer: int,
    top_k: int = 2,
    order_sensitive: bool = True
):
    """
    Get the frequencies of all top_k permutations

    Args:
        datasets: A single dataset dict or a list of dataset dicts.
                  Each dict must include keys: ["dataset_name", "texts", "tokens", "topk_indices", "topk_scores"]
        layer: Which layer to analyze
        top_k: How many top experts to consider per token (e.g., 2 for expert pairs)
        order_sensitive: 
            - If False: (e1, e2) == (e2, e1)
            - If True: preserve order of experts

    Returns:
        1. A dict mapping dataset_name --> Counter, where the counter maps (permutation) --> Count
        2. Combined counter over all datasets
    """
    if isinstance(datasets, dict):
        datasets = [datasets]

    d = {}
    combined_dataset = Counter()
    for dataset in datasets:
        dataset_name = dataset["dataset_name"]
        indices_list = dataset["topk_indices"] # List[torch.Tensor(16, seq_len, 8)]
        
        permutation_counts = Counter()
        for sample_idx, topk_idx in enumerate(indices_list):
            # Extract top-k data for the specified layer
            idx_layer = topk_idx[layer]   # shape: [seq_len, topk]

            for token_pos, experts_arr in enumerate(idx_layer):
                # Ensure we're working with tensors (and convert if needed)
                if isinstance(experts_arr, torch.Tensor):
                    experts_arr = experts_arr.cpu()

                if experts_arr.numel() < top_k:
                    print("Check experts_arr")
                    continue  # not enough experts

                # Get the top_k experts/scores and convert to Python lists
                experts = experts_arr[:top_k].tolist()

                # Use order-sensitive or order-insensitive tuple as key
                key = tuple(experts) if order_sensitive else tuple(sorted(experts))
                
                permutation_counts[key] += 1

        d[dataset_name] = permutation_counts
        combined_dataset.update(permutation_counts)

    return d, combined_dataset


def find_expert_coactivations(
    datasets: Union[Dict[str, List], List[Dict[str, List]]],
    layer: int,
    top_k: int = 2,
    order_sensitive: bool = False,
    keep_n: int = 5,
) -> Dict[Tuple[int, ...], Dict]:
    """
    Find most frequent expert co-activations at a given layer.

    Args:
        datasets: A single dataset dict or a list of dataset dicts.
                  Each dict must include keys: ["dataset_name", "texts", "tokens", "topk_indices", "topk_scores"]
        layer: Which layer to analyze
        top_k: How many top experts to consider per token (e.g., 2 for expert pairs)
        order_sensitive: 
            - If False: (e1, e2) == (e2, e1)
            - If True: preserve order of experts
        keep_n: How many top examples (by score) to keep per expert set

    Returns:
        A dict mapping (expert1, ..., expertK) -> {
            "count": number of occurrences,
            "top_examples": list of up to `keep_n` dicts, sorted by combined_score desc
        }
    """
    if isinstance(datasets, dict):
        datasets = [datasets]

    # Use a heap to maintain top-N examples efficiently
    coactivation_stats = defaultdict(lambda: {"count": 0, "top_examples_heap": []})
    
    # Counter to ensure unique ordering in heap tuples
    counter = 0

    for dataset in datasets:
        dataset_name = dataset["dataset_name"]
        split = dataset["split"]
        question_only = dataset["question_only"]
        texts = dataset["texts"] # List[str]
        tokens_list = dataset["tokens"] # List[List[str]]
        indices_list = dataset["topk_indices"] # List[torch.Tensor(16, seq_len, 8)]
        scores_list = dataset["topk_scores"] # List[torch.Tensor(16, seq_len, 8)]
        permutation_counts = Counter()
        for sample_idx, (text, tokens, topk_idx, topk_val) in enumerate(zip(texts, tokens_list, indices_list, scores_list)):
            # Extract top-k data for the specified layer
            idx_layer = topk_idx[layer]   # shape: [seq_len, topk]
            val_layer = topk_val[layer]   # shape: [seq_len, topk]

            for pos, (tok, experts_arr, scores_arr) in enumerate(zip(tokens, idx_layer, val_layer)):
                # Ensure we're working with tensors (and convert if needed)
                if isinstance(experts_arr, torch.Tensor):
                    experts_arr = experts_arr.cpu()
                if isinstance(scores_arr, torch.Tensor):
                    scores_arr = scores_arr.cpu()

                if experts_arr.numel() < top_k:
                    print("Check experts_arr")
                    continue  # not enough experts

                # Get the top_k experts/scores and convert to Python lists
                experts = experts_arr[:top_k].tolist()
                scores = scores_arr[:top_k].tolist()

                # Use order-sensitive or order-insensitive tuple as key
                key = tuple(experts) if order_sensitive else tuple(sorted(experts))
                
                combined_score = sum(scores)

                entry = coactivation_stats[key]
                entry["count"] += 1

                # Construct the example
                example = {
                    "dataset": dataset_name,
                    "split": split,
                    "question_only": question_only,
                    "sample_idx": sample_idx,
                    "text": text,
                    "token": tok,
                    "tokens": tokens,
                    "position": pos,
                    "experts": experts_arr.tolist(),  # full list of experts at this token
                    "scores": scores_arr.tolist(),    # full list of scores at this token
                    "combined_score": combined_score
                }

                # Use counter to ensure unique tuples for heap comparison
                # Heap tuple: (combined_score, counter, example)
                heapq.heappush(entry["top_examples_heap"], (combined_score, counter, example))
                counter += 1

                if len(entry["top_examples_heap"]) > keep_n:
                    heapq.heappop(entry["top_examples_heap"])  # remove lowest score

    # Convert heap to sorted list (descending by score)
    final_output = {}
    for key, value in coactivation_stats.items():
        sorted_examples = sorted(value["top_examples_heap"], key=lambda x: x[0], reverse=True)
        final_output[key] = {
            "count": value["count"],
            "top_examples": [item[2] for item in sorted_examples]  # Extract example from tuple
        }

    return final_output

def format_expert_coactivations(
    coactivation_stats: Dict[Tuple[int, ...], Dict],
    tokenizer,
    context_before: int = 20,
    context_after: int = 6
):
    """
    In-place formatting of the coactivation report

    Args:
        coactivation_stats: Output of find_expert_coactivations
        tokenizer: HuggingFace tokenizer (same one used for extraction)
        save_path: where to save .txt
        context_before: how many tokens to show before target
        context_after: how many tokens to show after target
        top_n: how many expert pairs to log
    """
    for experts_permutation, info in coactivation_stats.items():
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

# def concatenate_expert_coactivations(
#     datasets_dict: Dict[str, object],
#     layer: int = 7,
#     default_topK: List[int] = [1,2,3],
#     filter_threshhold: int = 10
# ):
#     # Across all datasets, we want a single counter of 
#     # (permutation) -> Count
#     combined_frequencies = Counter()
#     for k in default_topk:
#         _, combined_frequencies_k = get_frequencies(
#             datasets = [v for k, v in datasets_dict.items()],
#             layer = layer,
#             top_k = k
#         )
#         combined_frequencies.update(combined_frequencies_k)
    
#     '''
#     Permutation --> {
#         "expert_order": List[int],
#         "count": int - How often this expert order occurs in total
#         "next": List
#     }
#     '''
#     d = {}
#     for k in default_topk:
#         for dataset_name, dataset in datasets_dict.items():
            
def concatenate_expert_coactivations(
    datasets_dict: Dict[str, object],
    tokenizer, # required for the token formatting
    layer: int = 7,
    default_topK: List[int] = [1, 2, 3],
    filter_threshold: int = 10,
    order_sensitive: bool = True,
    keep_n: int = 5,
    n_experts: int = 64
):
    """
    Aggregate expert co-activation stats across multiple datasets and top-K granularities.

    Args:
        datasets_dict: dict mapping dataset_name -> dataset dict
        tokenizer: tokenizer for formatting examples
        layer: which layer to analyze
        default_topK: list of K values to consider (e.g., [1,2,3])
        filter_threshold: minimum total frequency to keep a permutation
        order_sensitive: whether expert ordering matters
        keep_n: number of examples to keep per dataset per permutation
        n_experts: number of experts in the MoE (default 64)

    Returns:
        dict mapping permutation -> {
            "expert_order": tuple[int],
            "count": int (total across datasets),
            "per_dataset": {
                dataset_name: {
                    "count": int,
                    "examples": list[dict]   # formatted examples
                }
            },
            "next": list[int] of length n_experts
        }
    """
    combined: Dict[Tuple[int, ...], Dict] = {}

    for k in default_topK:
        # Step 1: frequency counts
        per_dataset_counts, combined_counts = get_frequencies(
            datasets=[v for v in datasets_dict.values()],
            layer=layer,
            top_k=k,
            order_sensitive=order_sensitive,
        )

        # Step 2: examples per dataset
        per_dataset_examples = {}
        for ds_name, ds in datasets_dict.items():
            ex = find_expert_coactivations(
                datasets=ds,
                layer=layer,
                top_k=k,
                order_sensitive=order_sensitive,
                keep_n=keep_n,
            )
            format_expert_coactivations(ex, tokenizer=tokenizer)
            per_dataset_examples[ds_name] = ex

        # Step 3: merge counts + examples
        for perm, total_count in combined_counts.items():
            if total_count < filter_threshold:
                continue

            if perm not in combined:
                combined[perm] = {
                    "expert_order": perm,
                    "count": 0,
                    "per_dataset": {},
                    "next": {ds: [0] * n_experts for ds in datasets_dict.keys()},
                }
                combined[perm]["next"]["all"] = [0] * n_experts  # aggregated

            combined[perm]["count"] += total_count

            for ds_name in datasets_dict.keys():
                ds_count = per_dataset_counts[ds_name].get(perm, 0)
                if ds_count == 0 and perm not in per_dataset_examples[ds_name]:
                    continue

                combined[perm]["per_dataset"].setdefault(ds_name, {
                    "count": 0,
                    "examples": []
                })

                combined[perm]["per_dataset"][ds_name]["count"] += ds_count
                if perm in per_dataset_examples[ds_name]:
                    combined[perm]["per_dataset"][ds_name]["examples"].extend(
                        per_dataset_examples[ds_name][perm]["top_examples"]
                    )

        # Step 4: build per-dataset + aggregated "next" frequencies
        for ds_name, ds in datasets_dict.items():
            indices_list = ds["topk_indices"]  # List[torch.Tensor(num_layers, seq_len, topK)]
            for topk_idx in indices_list:
                idx_layer = topk_idx[layer]
                for experts_arr in idx_layer:
                    experts_arr = experts_arr.cpu() if isinstance(experts_arr, torch.Tensor) else experts_arr
                    if experts_arr.numel() < k + 1:
                        continue

                    prefix = tuple(experts_arr[:k].tolist())
                    next_expert = experts_arr[k].item()

                    if prefix in combined:
                        combined[prefix]["next"][ds_name][next_expert] += 1
                        combined[prefix]["next"]["all"][next_expert] += 1

    print(f"Finished aggregating expert coactivation examples with {len(combined)} permutations!")
    return combined