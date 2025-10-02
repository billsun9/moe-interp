from collections import defaultdict, Counter
from itertools import combinations
import numpy as np
import math
import matplotlib.pyplot as plt


def compute_subset_counts(d, layer_idx=7, subset_sizes=(1,2,3,4)):
    counts = defaultdict(lambda: Counter())
    
    for dataset_name, samples in d.items():
        for topk_indices in samples['topk_indices']:  # shape [layers, tokens, k]
            token_experts = topk_indices[layer_idx]   # [tokens, k]
            for experts in token_experts:
                experts = sorted(experts.tolist())
                for s in subset_sizes:
                    for combo in combinations(experts, s):
                        counts[combo][dataset_name] += 1
    return counts

def entropy(probs):
    return -sum(p * math.log(p+1e-9) for p in probs)

def analyze_counts(counts, datasets, min_total=100, top_mass_threshold_2=0.66, top_mass_threshold_4=0.85):
    results = []
    for experts, dist in counts.items():
        total = sum(dist.values())
        if total < min_total:
            continue
        
        freqs = np.array([dist.get(ds, 0) for ds in datasets])
        probs = freqs / freqs.sum()
        H = entropy(probs)
        sd = freqs.std()
        sorted_probs = np.sort(probs)[::-1] # we sort over the frequencies in each dataset, descending
        top2_mass = sorted_probs[:2].sum()
        top4_mass = sorted_probs[:4].sum()
        
        dataset_specific = (top2_mass >= top_mass_threshold_2 or top4_mass >= top_mass_threshold_4)
        if dataset_specific:
            results.append({
                "experts": experts,
                "subset_size": len(experts),
                "total_count": total,
                "dataset_distribution": dict(dist),
                "entropy": H,
                "std_dev": sd,
                "top2_mass": top2_mass,
                "top4_mass": top4_mass
            })
    
    # rank dataset-specific ones first (low entropy, high skew)
    return sorted(results, key=lambda x: (x["entropy"], -x["std_dev"]))

def stratified_dataset_associated_experts(
    results, 
    target_datasets, 
    top_n=20, 
    min_fraction=0.5,
    stratify_sizes=(1,2,3,4)
):
    """
    Returns dataset-specific expert subsets stratified by subset length.
    """
    if isinstance(target_datasets, str):
        target_datasets = [target_datasets]

    stratified = {s: [] for s in stratify_sizes}

    for r in results:
        if r["subset_size"] not in stratify_sizes:
            continue
        total = r["total_count"]
        target_count = sum(r["dataset_distribution"].get(ds, 0) for ds in target_datasets)
        fraction = target_count / total if total > 0 else 0.0

        if fraction >= min_fraction:
            stratified[r["subset_size"]].append({
                "experts": r["experts"],
                "subset_size": r["subset_size"],
                "total_count": total,
                "target_count": target_count,
                "target_fraction": fraction,
                "entropy": r["entropy"],
                "std_dev": r["std_dev"],
                "distribution": r["dataset_distribution"]
            })

    # Sort within each subset size
    for s in stratify_sizes:
        stratified[s].sort(key=lambda x: (x["target_fraction"], x["target_count"]), reverse=True)
        stratified[s] = stratified[s][:top_n]

    return stratified

def plot_expert_set_distribution(expert_set, counts, datasets, normalize_by_tokens=None):
    """
    Plot bar chart of expert_set frequency across datasets.
    
    expert_set: tuple of ints (e.g., (20,18,64))
    counts: output from compute_subset_counts
    datasets: list of dataset names
    normalize_by_tokens: optional dict {dataset: token_count}
    """
    dist = counts.get(expert_set, {})
    values = []
    
    for ds in datasets:
        raw = dist.get(ds, 0)
        if normalize_by_tokens:
            val = raw / (normalize_by_tokens[ds] + 1e-9)
        else:
            val = raw
        values.append(val)

    plt.figure(figsize=(10,4))
    plt.bar(datasets, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Normalized Frequency" if normalize_by_tokens else "Raw Count")
    plt.title(f"Distribution of Expert Set {expert_set} Across Datasets")
    plt.tight_layout()
    plt.show()


def analyze_expert_sets(
    counts,
    dataset_names,
    target_datasets,
    datasets=None,
    layer=None,
    tokenizer=None,
    normalize=False,
    stratify_sizes=(1, 2, 3, 4),
    top_n_sets=5,
    min_fraction=0.5,
    keep_n_samples=5,
    order_sensitive=False,
):
    """
    1. Find dataset-specific expert sets stratified by subset size.
    2. Plot distribution of all 64 experts for each set.
    3. Retrieve top-N samples if datasets + tokenizer are provided.

    Args:
        counts: dict mapping expert tuples -> dataset counts
        dataset_names: list of dataset names
        target_datasets: str or list of str (datasets to check specificity against)
        datasets: raw dataset dicts (needed for pulling examples)
        layer: layer index for coactivations
        tokenizer: tokenizer to decode text
        normalize: bool, normalize by total tokens per dataset
        stratify_sizes: tuple of subset sizes to consider
        top_n_sets: how many expert sets to return per subset size
        min_fraction: minimum fraction of activations in target dataset(s)
        keep_n_samples: how many top samples per expert set
        order_sensitive: whether expert ordering matters

    Returns:
        dict[stratify_size -> list of results]
    """
    # Step 1: Run analyze_counts once to get all candidate results
    results = analyze_counts(
        counts=counts,
        datasets=dataset_names,
        min_total=100,
        top_mass_threshold_2=0.66,
        top_mass_threshold_4=0.85
    )

    # Step 2: Stratify by subset size
    stratified = stratified_dataset_associated_experts(
        results=results,
        target_datasets=target_datasets,
        top_n=top_n_sets,
        min_fraction=min_fraction,
        stratify_sizes=stratify_sizes
    )

    # Step 3: If datasets provided, compute coactivation stats ONCE per K
    coactivation_stats_by_k = {}
    if datasets is not None and layer is not None and tokenizer is not None:
        for k in stratify_sizes:
            coactivation_stats_by_k[k] = find_expert_coactivations(
                datasets=datasets,
                layer=layer,
                top_k=k,
                order_sensitive=order_sensitive,
                keep_n=keep_n_samples,
            )

    results_out = {}
    for k, expert_sets in stratified.items():
        results_out[k] = []
        for s in expert_sets:
            expert_set = s["experts"]

            # Step 4: Plot distribution across datasets
            plot_expert_set_distribution(
                counts=counts,
                expert_set=expert_set,
                dataset_names=dataset_names,
                normalize=normalize,
                title=f"Expert set {expert_set} distribution (size={k})"
            )

            # Step 5: Get top examples if possible
            top_examples = None
            if datasets is not None and layer is not None and tokenizer is not None:
                key = tuple(expert_set) if order_sensitive else tuple(sorted(expert_set))
                if key in coactivation_stats_by_k[k]:
                    format_expert_coactivations(
                        {key: coactivation_stats_by_k[k][key]}, tokenizer=tokenizer
                    )
                    top_examples = coactivation_stats_by_k[k][key]["top_examples"]

            results_out[k].append({
                "experts": expert_set,
                "subset_size": k,
                "target_fraction": s["target_fraction"],
                "target_count": s["target_count"],
                "total_count": s["total_count"],
                "entropy": s["entropy"],
                "std_dev": s["std_dev"],
                "distribution": s["distribution"],
                "top_examples": top_examples
            })

    return results_out
