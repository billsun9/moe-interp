import os
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Any
from collections import defaultdict
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

