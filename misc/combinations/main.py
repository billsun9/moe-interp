import os
from typing import List, Dict
import time

from moe.eda.eda_utils import load_all_activations
from moe.eda.expert_coactivations import *
from moe.activation_extraction.activation_extractor_utils import OLMoEActivationExtractor, verify_saved_activations_ds
from moe.path_config import MODEL_CACHE_DIR, SAVE_ACTS_PATH_DIR, SAVE_ARTIFACTS_PATH_DIR

from moe.misc.combinations.helper import *

start_time = time.time()
extractor = OLMoEActivationExtractor(cache_dir=MODEL_CACHE_DIR)
extractor.load_tokenizer()

datasets_to_process = [
    # False ==> Question: {Q} [Optional: {Choices} or {Reasoning}] Answer:{A}
    ("gsm8k", False),
    ("arc-easy", False),
    ("arc-challenge", False),
    ("sciq", False),
    ("mbpp", False),
    # False ==> Not a question. Just the {statement}
    ("ag_news", False),
    ("imdb_pos", False),
    ("imdb_neg", False),
    # False/True does not matter for these. Just the {statement}
    ("poetry", False),
    ("lex_glue", False),
    ("arxiv", False),
    ("personas", False)
]

t_start = time.time()
d = load_all_activations(
    datasets_to_process = datasets_to_process,
    extractor = extractor,
    split_names = ["train"]
)
counts = compute_subset_counts(d)

results = analyze_counts(
    counts,
    list(d.keys())
)

for dataset_name, _ in datasets_to_process:
    print(f"============================== {dataset_name} ==============================")
    # this returns the experts which are most associated with some particular dataset, such that 
    # the length of the expert order is in stratify_sizes
    res = stratified_dataset_associated_experts(
        results,
        target_datasets=dataset_name,
        stratify_sizes=[1,2],
    )

    for k, v in res.items():
        print("-----------------------------------")
        print(f"Expert set size: {k} has {len(v)} items")
        for e in v:
            print(f"Experts: {e['experts']} ({e['target_count']}/{e['total_count']})")
        # 'experts': (2, 26), 'subset_size': 2, 'total_count': 7947, 'target_count': 6040, 'target_fraction': 0.7600352334214169

custom_datasets = [
    ["arc-easy", "arc-challenge"],
    ["imdb_pos", "imdb_neg"],
    ["arc-easy", "arc-challenge", "sciq"],
]

for dataset_name in custom_datasets:
    print(f"============================== {dataset_name} ==============================")
    # this returns the experts which are most associated with some particular dataset, such that 
    # the length of the expert order is in stratify_sizes
    res = stratified_dataset_associated_experts(
        results,
        target_datasets=dataset_name,
        stratify_sizes=[1,2],
        min_fraction=0.7,
    )

    for k, v in res.items():
        print("-----------------------------------")
        print(f"Expert set size: {k} has {len(v)} items")
        for e in v:
            print(f"Experts: {e['experts']} ({e['target_count']}/{e['total_count']})")
        # 'experts': (2, 26), 'subset_size': 2, 'total_count': 7947, 'target_count': 6040, 'target_fraction': 0.7600352334214169


print(f"Time Elapsed: {time.time() - t_start} seconds")