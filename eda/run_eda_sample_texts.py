import os
from typing import Dict, List
import torch
from datasets import load_dataset

from moe.activation_extraction.run_caching_utils import *
from moe.activation_extraction.activation_extractor_utils import OLMoEActivationExtractor
from moe.eda.eda_utils import *

from datetime import datetime
MODEL_NAME = "allenai/OLMoE-1B-7B-0125-Instruct"
MODEL_CACHE_DIR = "/local/bys2107/hf_cache"
extractor = OLMoEActivationExtractor(model_name = MODEL_NAME, cache_dir = MODEL_CACHE_DIR)
# ---------------------------------------------------------------
# SAVE_PATH = "/local/bys2107/research/data/OLMoE-acts/custom/fluffy_or_pointy_objects/fluffy_or_pointy_objects_custom_4samples.pt"

# dataset = extractor.load_activations(SAVE_PATH)

# print(dataset.keys())
# print(dataset['metadata'])
# extractor.print_activation_summary(dataset)
# FIRST_N_TOKENS = 6
# for sample_idx in range(len(dataset['texts'])):
#     print(f"Processing sample {sample_idx}: {dataset['texts'][sample_idx]}")
#     FIG_PATH = os.path.join(
#         "/local/bys2107/research/data/figs",
#         dataset['dataset_name'],
#         f"sample_{sample_idx}"
#     )
#     queries = []

#     for layer_idx in range(16): # hardcode number of layers
        
#         # queries: list of (layer_idx, sample_idx, token_idx)
#         queries = [(layer_idx, sample_idx, token_idx) for token_idx in range(min(FIRST_N_TOKENS, len(dataset['tokens'][sample_idx])))]

#         plot_routing_histogram(dataset, queries, save_dir = FIG_PATH, softmax=True, top_k=8, stacked=True)

# print("DONE!")
# ---------------------------------------------------------------
# SAVE_PATH = "/local/bys2107/research/data/OLMoE-acts/custom/random_strings/random_strings_custom_4samples.pt"

# dataset = extractor.load_activations(SAVE_PATH)

# print(dataset.keys())
# print(dataset['metadata'])
# extractor.print_activation_summary(dataset)
# FIRST_N_TOKENS = 6
# for sample_idx in range(len(dataset['texts'])):
#     print(f"Processing sample {sample_idx}: {dataset['texts'][sample_idx]}")
#     FIG_PATH = os.path.join(
#         "/local/bys2107/research/data/figs",
#         dataset['dataset_name'],
#         f"sample_{sample_idx}"
#     )
#     queries = []

#     for layer_idx in range(16): # hardcode number of layers
        
#         # queries: list of (layer_idx, sample_idx, token_idx)
#         queries = [(layer_idx, sample_idx, token_idx) for token_idx in range(min(FIRST_N_TOKENS, len(dataset['tokens'][sample_idx])))]

#         plot_routing_histogram(dataset, queries, save_dir = FIG_PATH, softmax=True, top_k=8, stacked=True)

# print("DONE!")
# ---------------------------------------------------------------
SAVE_PATH = "/local/bys2107/research/data/OLMoE-acts/custom/simple_math/simple_math_custom_4samples.pt"

dataset = extractor.load_activations(SAVE_PATH)

print(dataset.keys())
print(dataset['metadata'])
extractor.print_activation_summary(dataset)
FIRST_N_TOKENS = 6
for sample_idx in range(len(dataset['texts'])):
    print(f"Processing sample {sample_idx}: {dataset['texts'][sample_idx]}")
    FIG_PATH = os.path.join(
        "/local/bys2107/research/data/figs",
        dataset['dataset_name'],
        f"sample_{sample_idx}"
    )
    queries = []

    for layer_idx in range(16): # hardcode number of layers
        
        # queries: list of (layer_idx, sample_idx, token_idx)
        queries = [(layer_idx, sample_idx, token_idx) for token_idx in range(min(FIRST_N_TOKENS, len(dataset['tokens'][sample_idx])))]

        plot_routing_histogram(dataset, queries, save_dir = FIG_PATH, softmax=True, top_k=8, stacked=True)

print("DONE!")
# ---------------------------------------------------------------
# SAVE_PATH = "/local/bys2107/research/data/OLMoE-acts/gsm8k/gsm8k_test_questions_20samples.pt"
# dataset = extractor.load_activations(SAVE_PATH)

# print(dataset.keys())
# print(dataset['metadata'])
# extractor.print_activation_summary(dataset)
# FIRST_N_TOKENS = 6
# OFFSET = 7
# for sample_idx in range(4):
#     print(f"Processing sample {sample_idx}: {dataset['texts'][sample_idx]}")
#     FIG_PATH = os.path.join(
#         "/local/bys2107/research/data/figs",
#         dataset['dataset_name'],
#         f"sample_{sample_idx}"
#     )
#     queries = []

#     for layer_idx in range(16): # hardcode number of layers
        
#         # queries: list of (layer_idx, sample_idx, token_idx)
#         queries = [(layer_idx, sample_idx, OFFSET + token_idx) for token_idx in range(min(FIRST_N_TOKENS, len(dataset['tokens'][sample_idx])))]

#         plot_routing_histogram(dataset, queries, save_dir = FIG_PATH, softmax=True, top_k=8, stacked=True)

# print("DONE!")
# ---------------------------------------------------------------
# import os
# SAVE_PATH = "/local/bys2107/research/data/OLMoE-acts/gsm8k/gsm8k_test_questions_20samples.pt"
# dataset = extractor.load_activations(SAVE_PATH)
# LAST_N_TOKENS = 10

# for sample_idx in range(10, 14):
#     print(f"Processing sample {sample_idx}: {dataset['texts'][sample_idx]}")
    
#     FIG_PATH = os.path.join(
#         "/local/bys2107/research/data/figs",
#         dataset['dataset_name'] + "_lastN",
#         f"sample_{sample_idx}"
#     )
    
#     for layer_idx in range(16):  # hardcode number of layers
        
#         tokens = dataset['tokens'][sample_idx]
#         num_tokens = len(tokens)
#         last_token_indices = list(range(num_tokens - LAST_N_TOKENS, num_tokens))

#         # queries: list of (layer_idx, sample_idx, token_idx)
#         queries = [(layer_idx, sample_idx, token_idx) for token_idx in last_token_indices]

#         plot_routing_histogram(
#             dataset,
#             queries,
#             save_dir=FIG_PATH,
#             softmax=True,
#             top_k=8,
#             stacked=True
#         )
