import os
import pickle
import numpy as np
from moe.path_config import SAVE_ACTS_PATH_DIR

# Load coactivation data
# with open(os.path.join(SAVE_ACTS_PATH_DIR, "l7_coactivations_annotated.pkl"), "rb") as f:
#     combined_coactivations = pickle.load(f)

# # Compute lengths of auto_descriptions
# lengths = [len(v['auto_description']) for v in combined_coactivations.values() if 'auto_description' in v]

# # Convert to numpy for easy stats
# arr = np.array(lengths)

# # Compute and print statistics
# print(f"Count: {len(arr)}")
# print(f"Min: {arr.min()}")
# print(f"Max: {arr.max()}")
# print(f"Mean: {arr.mean():.2f}")
# print(f"Median: {np.median(arr):.2f}")
# print(f"quantile(0.1): {np.percentile(arr, 10):.2f}")
# print(f"1st Quartile (Q1): {np.percentile(arr, 25):.2f}")
# print(f"3rd Quartile (Q3): {np.percentile(arr, 75):.2f}")
# print(f"quantile(0.9): {np.percentile(arr, 90):.2f}")


import os
from typing import List, Dict
import time

from moe.eda.eda_utils import load_all_activations
from moe.eda.expert_coactivations import *
from moe.activation_extraction.activation_extractor_utils import OLMoEActivationExtractor
from moe.path_config import MODEL_CACHE_DIR, SAVE_ACTS_PATH_DIR, SAVE_ARTIFACTS_PATH_DIR

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


d = load_all_activations(
    datasets_to_process = datasets_to_process,
    extractor = extractor,
    split_names = ["train"]
)

for k in range(1, 3):
    print(f"======= k={k} =======")
    per_dataset_counts, combined_counts = get_frequencies(
        datasets=[v for v in d.values()],
        layer=7,
        top_k=k,
        order_sensitive=True
    )

    for key in per_dataset_counts.keys():
        print(key, per_dataset_counts[key].total())

    print(combined_counts.total())