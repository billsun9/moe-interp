import os
import torch
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Any, Union
from collections import defaultdict
import heapq

from moe.activation_extraction.activation_extractor_utils import OLMoEActivationExtractor
from moe.path_config import SAVE_ACTS_PATH_DIR

def load_all_activations(
    datasets_to_process: List[Tuple[str, bool]],
    extractor,
    split_names: List[str] = ["train", "test"],
    verbose: bool = True,
    simple_keyname: bool = True
) -> Dict[str, object]:
    """
    Load activations for multiple datasets and splits.

    Args:
        datasets_to_process: List of (dataset_name, question_only) tuples
        extractor: Initialized OLMoEActivationExtractor
        split_names: List of splits to load (default=["train", "test"])
        verbose: Whether to print missing file warnings
        simple_keyname: If True, use dataset_name as dict key; else include split and suffix

    Returns:
        Dictionary mapping keys like 'gsm8k_train_questions' to activation objects
    """
    activations = {}

    for dataset_name, question_only in datasets_to_process:
        question_suffix = "_questions" if question_only else "_full"
        dataset_dir = os.path.join(SAVE_ACTS_PATH_DIR, dataset_name)

        for split in split_names:
            # Pattern to match: e.g., mbpp_train_full_XXXsamples.pt
            pattern = f"{dataset_name}_{split}{question_suffix}_*samples.pt"
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


# ------------------------
# USED FOR PLOTTING PER-LAYER SOFTMAX(ROUTING_LOGITS) ACROSS TOKENS/SAMPLES
# ------------------------

def token_stats(router_logits, top_k=8):
    P = to_probs(router_logits)           # [T,E]
    ent = -(P.clamp_min(1e-12) * P.clamp_min(1e-12).log()).sum(dim=-1)  # [T]
    topv, topi = torch.topk(P, k=top_k, dim=-1)                         # [T,k]
    topk_mass = topv.sum(dim=-1)                                        # [T]
    argmax = P.argmax(dim=-1)                                           # [T]
    return {"entropy": ent, "topk_mass": topk_mass, "topi": topi, "argmax": argmax}

    
def topk_mask(dist, k):
    topk_idx = torch.topk(dist, k=k).indices
    mask = torch.zeros_like(dist)
    mask[topk_idx] = 1.0
    return mask


def format_dataset_metadata(dataset: dict):
    """
    Format dataset metadata for filenames and titles.
    Handles both standard datasets (with split/question_only)
    and custom datasets (where those may be empty).
    """
    dataset_name = dataset.get("dataset_name", "unknown")
    split = dataset.get("split", "")
    qonly = dataset.get("question_only", "")
    if type(qonly) == bool:
        qonly = "qonly" if qonly else "q_and_a"
    # filename-safe parts
    name_parts = [dataset_name]
    title_parts = [dataset_name]

    if split:
        name_parts.append(split)
        title_parts.append(split)
    if qonly:
        name_parts.append(qonly)
        title_parts.append(qonly)

    name_str = "_".join(name_parts)
    title_str = "-".join(title_parts)
    return name_str, title_str


# ------------------------
# 1) Per-token histogram
# ------------------------

def get_distribution(dataset, layer_idx, sample_idx, token_idx, softmax=True, top_k=None):
    """
    Returns either softmax distribution or binary mask.
    """
    logits = dataset['routing_logits'][sample_idx][layer_idx, token_idx]  # [num_experts]

    if softmax:
        dist = F.softmax(logits, dim=-1)
        if top_k is not None:
            # keep only top-k, zero elsewhere
            topk_idx = torch.topk(dist, k=top_k).indices
            mask = torch.zeros_like(dist)
            mask[topk_idx] = dist[topk_idx]
            dist = mask
    else:
        # binary indicator (1 for top-k experts)
        if top_k is not None:
            topk_idx = torch.topk(logits, k=top_k).indices
            dist = torch.zeros_like(logits)
            dist[topk_idx] = 1.0
        else:
            dist = torch.ones_like(logits)

    return dist


def plot_routing_histogram(dataset, queries, save_dir, softmax=True, top_k=None, stacked=False):
    """
    Plot histograms of routing distributions for specific (layer,sample,token).

    Args:
        dataset: dict with keys 'routing_logits', 'tokens', 'dataset_name', 'split', 'question_only'
        queries: list of (layer_idx, sample_idx, token_idx)
        save_dir: where to save figures
        softmax: if True, plot probabilities; if False, plot binary 1/0 indicators
        top_k: None for all experts, or integer for top-k selection
        stacked: if True, stack bars across queries
    """
    os.makedirs(save_dir, exist_ok=True)

    num_experts = dataset['routing_logits'][0].shape[-1]
    x = np.arange(num_experts)

    # Collect distributions
    dists, labels = [], []
    for (layer_idx, sample_idx, token_idx) in queries:
        dist = get_distribution(dataset, layer_idx, sample_idx, token_idx, softmax=softmax, top_k=top_k)
        dists.append(dist.cpu().numpy())
        token_text = dataset['tokens'][sample_idx][token_idx]
        labels.append(f"S{sample_idx},T{token_idx}('{token_text}'),L{layer_idx}")

    dists = np.array(dists)  # [num_queries, num_experts]

    plt.figure(figsize=(14, 6))

    if stacked:
        bottom = np.zeros(num_experts)
        for dist, label in zip(dists, labels):
            plt.bar(x, dist, bottom=bottom, label=label, alpha=0.8)
            bottom += dist
    else:
        bar_width = 0.8 / len(queries)
        offsets = np.linspace(-0.4, 0.4, len(queries))
        for i, (dist, label) in enumerate(zip(dists, labels)):
            plt.bar(x + offsets[i], dist, width=bar_width, label=label, alpha=0.8)

    # Metadata for filename/title
    name_str, title_str = format_dataset_metadata(dataset)
    softmax_str = "softmax" if softmax else "binary"
    topk_str = f"top{top_k}" if top_k else "all"
    stack_str = "STACK" if stacked else "SIDE"
    query_str = "_".join([f"L{l}S{s}T{t}" for (l, s, t) in queries])

    fname = f"{name_str}_routing_histBAR_{softmax_str}_{topk_str}_{stack_str}_{query_str}.png"
    save_path = os.path.join(save_dir, fname)

    plt.title(f"Routing distribution ({softmax_str}, {topk_str}, {stack_str}) | {title_str}")
    
    plt.xlabel("Expert ID")
    plt.ylabel("Routing Weight" if softmax else "Top-K Indicator")
    plt.xticks(x)
    plt.legend()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


# ------------------------
# 2) Aggregation histogram
# ------------------------

def plot_routing_aggregation(dataset, layer_idx, save_dir, mode="sample", sample_idx=None, top_k=None):
    """
    Aggregated routing histogram (sample-level or dataset-level).
    Saves to path automatically derived from dataset metadata.
    """
    os.makedirs(save_dir, exist_ok=True)

    if mode == "sample":
        assert sample_idx is not None, "Must specify sample_idx for mode='sample'"
        logits = dataset['routing_logits'][sample_idx][layer_idx]  # [num_tokens, 64]
        dists = F.softmax(logits, dim=-1)  # [num_tokens, 64]
    elif mode == "dataset":
        all_dists = []
        for s in range(len(dataset['routing_logits'])):
            logits = dataset['routing_logits'][s][layer_idx]  # [num_tokens, 64]
            dists = F.softmax(logits, dim=-1)
            all_dists.append(dists)
        dists = torch.cat(all_dists, dim=0)  # [Σ num_tokens, 64]
    else:
        raise ValueError("mode must be 'sample' or 'dataset'")

    if top_k is not None:
        dists = torch.stack([topk_mask(row, top_k) for row in dists])  # [*,64]

    agg = dists.mean(dim=0)  # average distribution across tokens (and samples if dataset mode)

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(agg)), agg.cpu().numpy())

    name_str, title_str = format_dataset_metadata(dataset)
    topk_str = f"top{top_k}" if top_k else "softmax"

    if mode == "sample":
        fname = f"{name_str}_routing_agg_{topk_str}_L{layer_idx}_S{sample_idx}.png"
        title = f"Layer {layer_idx} | Sample {sample_idx} | Avg over tokens ({topk_str})"
    else:
        fname = f"{name_str}_routing_agg_{topk_str}_L{layer_idx}_dataset.png"
        title = f"Layer {layer_idx} | Dataset-level Avg ({topk_str})"

    save_path = os.path.join(save_dir, fname)

    plt.title(f"{title} | {title_str}")

    plt.xlabel("Expert ID")
    plt.ylabel("Average weight / frequency")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")

# ------------------------
# USED FOR PLOTTING PER-LAYER SOFTMAX(ROUTING_LOGITS) ACROSS TOKENS/SAMPLES
# ------------------------

def get_top_embeddings(
    dataset: Dict[str,Any],
    layer_idx: int,
    sample_indices: List[int],
    top_k: int = 8
) -> Dict[int, List[Tuple[str, int, int, int, float]]]:
    expert_map = defaultdict(list)
    dataset_name = dataset["dataset_name"]

    for sample_id in sample_indices:
        routing_logits = dataset['routing_logits'][sample_id]
        n_layers, n_tokens, n_experts = routing_logits.shape

        logits_at_layer = routing_logits[layer_idx]
        probs = F.softmax(logits_at_layer, dim=-1)

        for token_pos in range(n_tokens):
            token_probs = probs[token_pos]
            top_scores, top_experts = torch.topk(token_probs, top_k)

            for score, expert_id in zip(top_scores.tolist(), top_experts.tolist()):
                expert_map[expert_id].append(
                    (dataset_name, layer_idx, sample_id, token_pos, score)
                )

    for expert_id in expert_map:
        expert_map[expert_id].sort(key = lambda x: x[-1], reverse=True)
    
    return expert_map


def get_top_embeddings_multiple(
    datasets: Dict[str, Dict[str, Any]],
    layer_idx: int,
    sample_indices: Dict[str, List[int]],
    top_k: int = 8
) -> Dict[int, List[Tuple[str, int, int, int, float]]]:
    expert_map = defaultdict(list)
    for dataset_name, dataset in datasets.items():
        for sample_id in sample_indices[dataset_name]:
            routing_logits = dataset['routing_logits'][sample_id]
            n_layers, n_tokens, n_experts = routing_logits.shape
            logits_at_layer = routing_logits[layer_idx]
            probs = F.softmax(logits_at_layer, dim=-1)

            for token_pos in range(n_tokens):
                token_probs = probs[token_pos]
                top_scores, top_experts = torch.topk(token_probs, top_k)

                for score, expert_id in zip(top_scores.tolist(), top_experts.tolist()):
                    expert_map[expert_id].append(
                        (dataset_name, layer_idx, sample_id, token_pos, score)
                    )
    for expert_id in expert_map:
        expert_map[expert_id].sort(key=lambda x: x[-1], reverse=True)
    return expert_map



def hydrate_batch(
    dataset: Dict[str, Any],
    keys: List[Tuple[str, int, int, int, float]]
) -> List[Dict[str, Any]]:
    hydrated = []

    for dataset_name, layer_idx, sample_id, token_pos, score in keys:
        hydrated.append({
            "dataset_name": dataset_name,
            "layer_idx": layer_idx,
            "sample_id": sample_id,
            "token_pos": token_pos,
            "score": score,
            "token": dataset["tokens"][sample_id][token_pos],
            "texts": dataset["texts"][sample_id],
            "prerouting_logits": dataset["prerouting_logits"][sample_id][layer_idx][token_pos],
            "routing_logits": dataset["routing_logits"][sample_id][layer_idx][token_pos]
        })

    return hydrated


def hydrate_multiple(
    datasets: Dict[str, Dict[str, Any]],
    keys: List[Tuple[str, int, int, int, float]]
) -> List[Dict[str, Any]]:
    hydrated = []
    for dataset_name, layer_idx, sample_id, token_pos, score in keys:
        dataset = datasets[dataset_name]
        hydrated.append({
            "dataset_name": dataset_name,
            "layer_idx": layer_idx,
            "sample_id": sample_id,
            "token_pos": token_pos,
            "score": score,
            "token": dataset["tokens"][sample_id][token_pos],
            "texts": dataset["texts"][sample_id],
            "prerouting_logits": dataset["prerouting_logits"][sample_id][layer_idx][token_pos],
            "routing_logits": dataset["routing_logits"][sample_id][layer_idx][token_pos]
        })
    return hydrated


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

def plot_expert_embeddings(
    hydrated: List[Dict[str,Any]],
    expert_map: Dict[int, List[Tuple[str, int, int, int, float]]],
    save_path: str,
    method: str = "pca",
    max_points: int = 500,
    random_state: int = 42
):
    # build lookup: (dataset_name, layer_idx, sample_id, token_pos, score) -> expert_id
    expert_lookup = {}
    for expert_id, entries in expert_map.items():
        for entry in entries:
            expert_lookup[entry] = expert_id

    filtered = []
    labels = []
    tokens = []
    datasets = []

    for item in hydrated:
        key = (
            item['dataset_name'],
            item['layer_idx'],
            item['sample_id'],
            item['token_pos'],
            item['score']
        )
        if key in expert_lookup:
            filtered.append(item)
            labels.append(expert_lookup[key])
            tokens.append(item['token'])
            datasets.append(item['dataset_name'])

    assert len(filtered) > 0, "check that expert map and hydrated have mutual entries"

    embeddings = np.array([item['prerouting_logits'].numpy() for item in filtered])
    labels = np.array(labels)
    datasets = np.array(datasets)

    # dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, init="pca", random_state=random_state)
    else:
        raise ValueError(f"Unknown method {method}")

    reduced = reducer.fit_transform(embeddings)

    # --- plotting ---
    plt.figure(figsize=(16, 10))
    unique_experts = sorted(set(labels))
    unique_datasets = sorted(set(datasets))

    # color encodes expert, marker encodes dataset
    colors = plt.cm.get_cmap('tab10', len(unique_experts))
    markers = ["o", "s", "D", "^", "v", "x", "*", "P", "h"]  # extend if needed
    dataset_to_marker = {ds: markers[i % len(markers)] for i, ds in enumerate(unique_datasets)}

    for idx, expert_id in enumerate(unique_experts):
        expert_mask = labels == expert_id
        for ds in unique_datasets:
            ds_mask = datasets == ds
            mask = expert_mask & ds_mask
            if np.any(mask):
                plt.scatter(
                    reduced[mask, 0], reduced[mask, 1],
                    label=f"Expert {expert_id} ({ds})",
                    alpha=0.7,
                    marker=dataset_to_marker[ds],
                    color=colors(idx)
                )

    plt.title(f"Pre-routing logits ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


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


def save_coactivation_report(
    coactivation_stats: Dict[Tuple[int, ...], Dict],
    tokenizer,
    save_path: str,
    context_before: int = 18,
    top_n: int = 20,
):
    """
    Save co-activation report to a text file with decoded context.

    Args:
        coactivation_stats: Output of find_expert_coactivations
        tokenizer: HuggingFace tokenizer (same one used for extraction)
        save_path: where to save .txt
        context_before: how many tokens to show before target
        context_after: how many tokens to show after target
        top_n: how many expert pairs to log
    """
    sorted_items = sorted(
        coactivation_stats.items(),
        key=lambda kv: kv[1]["count"],
        reverse=True
    )

    with open(save_path, "w", encoding="utf-8") as f:
        for (experts, info) in sorted_items[:top_n]:
            f.write(f"Experts {experts} | Count={info['count']}\n")
            f.write("-" * 60 + "\n")

            for ex in info["top_examples"]:
                tokens = ex["tokens"]
                pos = ex["position"]

                context_tokens = tokens[max(0, pos - context_before):min(len(tokens), pos + 1)]

                # Proper decode
                context_str_preceeding = tokenizer.decode(
                    tokenizer.convert_tokens_to_ids(tokens[max(0, pos - context_before): pos + 1])
                )

                context_str_following = tokenizer.decode(
                    tokenizer.convert_tokens_to_ids(tokens[pos + 1: pos + 5])
                )

                f.write(
                    f"   token={ex['token']} ({ex['combined_score']:.3f}), "
                    f"pos={ex['position']}, "
                    f"dataset={ex['dataset']}, "
                    f"preceeding='{context_str_preceeding}', "
                    f"following='{context_str_following}'\n"
                )

                # f.write(
                #     f"[{ex['dataset']}] "
                #     f"pos={pos}, combined_score={ex['combined_score']:.4f}\n"
                #     f"Context: {context_str}\n"
                #     f"Experts={ex['experts']} Scores={ex['scores']}\n\n"
                # )
            f.write("\n")

def plot_expert_coactivation_heatmap(
    datasets: Union[Dict[str, List], List[Dict[str, List]]],
    layer: int,
    num_experts: int,
    save_path: str = "",
    top_k: int = 2,
    figsize: tuple = (16, 12),
    cmap: str = "viridis",
):
    """
    Plot expert co-activation heatmap using matplotlib.

    Assumes:
        - Always order_sensitive = True
        - Always normalize = True (per row)
        - No log scale

    Behavior:
        - Matrix[i][j] is incremented when expert i is top-1 and expert j is top-2
        - Matrix is row-normalized
        - Values >= 0.33 are fully saturated in the colormap

    Args:
        datasets: Single or list of dataset dicts.
        layer: Layer to analyze.
        num_experts: Total number of experts.
        save_path: Path to save the output image.
        top_k: Top-k experts per token (only top-2 used).
        figsize: Size of the matplotlib figure.
        cmap: Colormap.
    """
    if isinstance(datasets, dict):
        datasets = [datasets]

    coactivation_matrix = np.zeros((num_experts, num_experts), dtype=np.float32)

    for dataset in datasets:
        tokens_list = dataset["tokens"]
        indices_list = dataset["topk_indices"]

        for tokens, topk_idx in zip(tokens_list, indices_list):
            idx_layer = topk_idx[layer]  # shape: [seq_len, top_k]

            for experts_arr in idx_layer:
                if isinstance(experts_arr, torch.Tensor):
                    experts_arr = experts_arr.cpu()

                experts = experts_arr[:top_k].tolist()

                # Skip tokens where fewer than 2 experts or duplicates
                if len(experts) < 2 or len(set(experts)) < 2:
                    continue

                top1, top2 = experts[0], experts[1]
                if top1 != top2:  # Just double checking
                    coactivation_matrix[top1][top2] += 1

    # Row-wise normalization (per top-1 expert)
    row_sums = coactivation_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # prevent division by zero
    normalized_matrix = coactivation_matrix / row_sums

    if save_path:
        # Custom normalization for color mapping: saturate at 0.33
        # 1/64 roughly equals 0.02
        norm = Normalize(vmin=0.02, vmax=0.33, clip=True)

        # Plotting
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(normalized_matrix, cmap=cmap, norm=norm)

        ax.set_title(f"Expert Co-activation Heatmap (Layer {layer})")
        ax.set_xlabel("Top-2 Expert (j)")
        ax.set_ylabel("Top-1 Expert (i)")
        ax.set_xticks(np.arange(num_experts))
        ax.set_yticks(np.arange(num_experts))
        ax.set_xticklabels(range(num_experts))
        ax.set_yticklabels(range(num_experts))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Normalized Co-activation (saturates at 0.33)", rotation=-90, va="bottom")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {save_path}")
    return normalized_matrix


def save_top_coactivating_examples(
    coactivation_matrix: np.ndarray,
    coactivation_stats: Dict[Tuple[int, ...], Dict],
    tokenizer,
    save_path: str,
    threshold: float = 0.33,
    context_before: int = 18,
    context_after: int = 5,
):
    """
    Save co-activation report for expert pairs with frequency exceeding a threshold.

    Args:
        coactivation_matrix: Row-normalized matrix [i][j] = P(j | i is top-1)
        coactivation_stats: Output from find_expert_coactivations(order_sensitive=True)
        tokenizer: HuggingFace tokenizer
        save_path: Where to save the report
        threshold: Minimum normalized co-activation frequency to include
        context_before: Tokens before the target for context
        context_after: Tokens after the target for context
    """
    num_experts = coactivation_matrix.shape[0]

    with open(save_path, "w", encoding="utf-8") as f:
        for i in range(num_experts):
            for j in range(num_experts):
                freq = coactivation_matrix[i][j]

                if freq < threshold:
                    continue

                key = (i, j)
                if key not in coactivation_stats:
                    continue

                examples = coactivation_stats[key]["top_examples"]
                count = coactivation_stats[key]["count"]
                if count < 60:
                    continue
                f.write("-" * 60 + "\n")
                f.write(f"Experts (E{i}, E{j}) | freq={freq:.3f}, count={count}\n")
                f.write("-" * 60 + "\n")

                for ex in examples:
                    tokens = ex["tokens"]
                    pos = ex["position"]
                    token_str = ex["token"]
                    dataset = ex["dataset"]
                    combined_score = ex["combined_score"]

                    # Decode context
                    start_idx = max(0, pos - context_before)
                    end_idx = min(len(tokens), pos + 1 + context_after)

                    pre_ids = tokenizer.convert_tokens_to_ids(tokens[start_idx:pos + 1])
                    post_ids = tokenizer.convert_tokens_to_ids(tokens[pos + 1:end_idx])

                    context_str_pre = tokenizer.decode(pre_ids, skip_special_tokens=True)
                    context_str_post = tokenizer.decode(post_ids, skip_special_tokens=True)

                    f.write(
                        f"   token={token_str} ({combined_score:.3f}), "
                        f"pos={pos}, dataset={dataset}, "
                        f"pre='{context_str_pre}', post='{context_str_post}'\n"
                    )

                f.write("\n")

    print(f"[Saved] Co-activation report to: {save_path}")

def save_single_expert_report(
    expert_id: int,
    coactivation_matrix: np.ndarray,
    coactivation_stats: Dict[Tuple[int, ...], Dict],
    tokenizer,
    save_path: str,
    min_count: int = 60,
    freq_threshold: float = 0.2,
    context_before: int = 18,
    context_after: int = 5,
):
    """
    Save report for a single expert (as top-1) showing its most common co-activations and examples.

    Args:
        expert_id: The expert to focus on (must be top-1 in the pairs).
        coactivation_matrix: Row-normalized matrix [i][j] = P(j | i is top-1)
        coactivation_stats: Output from find_expert_coactivations(order_sensitive=True)
        tokenizer: HuggingFace tokenizer
        save_path: Path to save the report
        min_count: Minimum count of this expert being top-1 to include in report
        freq_threshold: Minimum normalized frequency for (expert_id, j) to include
        context_before: Number of tokens before the target
        context_after: Number of tokens after the target
    """
    # First check if expert_id has sufficient data
    total_count = sum(
        coactivation_stats.get((expert_id, j), {}).get("count", 0)
        for j in range(coactivation_matrix.shape[1])
    )

    if total_count < min_count:
        print(f"[Skip] Expert {expert_id} has only {total_count} top-1 activations (< {min_count})")
        return

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Expert {expert_id} (Top-1) Co-activation Report\n")
        f.write(f"Total Top-1 Count: {total_count}\n")
        f.write(f"Thresholds → Min Count: {min_count}, Co-activation Freq: {freq_threshold}\n\n")

        for j in range(coactivation_matrix.shape[1]):
            if j == expert_id:
                continue  # skip self

            freq = coactivation_matrix[expert_id][j]
            if freq < freq_threshold:
                continue

            key = (expert_id, j)
            if key not in coactivation_stats:
                continue

            examples = coactivation_stats[key]["top_examples"]
            f.write("-" * 60 + "\n")
            f.write(f"Co-activation: Expert ({expert_id}, {j}) | freq={freq:.3f}\n")
            f.write("-" * 60 + "\n")

            for ex in examples:
                tokens = ex["tokens"]
                pos = ex["position"]
                token_str = ex["token"]
                dataset = ex["dataset"]
                combined_score = ex["combined_score"]

                # Decode context
                start_idx = max(0, pos - context_before)
                end_idx = min(len(tokens), pos + 1 + context_after)

                pre_ids = tokenizer.convert_tokens_to_ids(tokens[start_idx:pos + 1])
                post_ids = tokenizer.convert_tokens_to_ids(tokens[pos + 1:end_idx])

                context_str_pre = tokenizer.decode(pre_ids, skip_special_tokens=True)
                context_str_post = tokenizer.decode(post_ids, skip_special_tokens=True)

                f.write(
                    f"   token={token_str} ({combined_score:.3f}), "
                    f"pos={pos}, dataset={dataset}, "
                    f"pre='{context_str_pre}', post='{context_str_post}'\n"
                )

            f.write("\n")

    print(f"[Saved] Expert {expert_id} report to: {save_path}")
