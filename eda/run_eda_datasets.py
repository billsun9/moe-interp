import os
from typing import List, Dict

from moe.eda.eda_utils import *
from moe.activation_extraction.activation_extractor_utils import OLMoEActivationExtractor
from moe.path_config import MODEL_CACHE_DIR, SAVE_ACTS_PATH_DIR, SAVE_ARTIFACTS_PATH_DIR

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
    ("arxiv", False)
]

all_datasets = load_all_activations(datasets_to_process, extractor, split_names=["train"])
print(all_datasets.keys())

print(all_datasets[list(all_datasets.keys())[0]].keys())

# verbose = False
# LAYER_IDX = 7
# # for i in range(64):
# #     out = find_expert_routed_tokens(
# #         datasets = [gsm8k_train, arceasy_train, sciq_train, arcchallenge_train],
# #         target_experts = [i],
# #         layer = LAYER_IDX,
# #         top_n = 8
# #     )
# #     print(f" ===================[ Layer {LAYER_IDX} Expert: {i} ]===================")
# #     for i, elem in enumerate(out):
# #         print(f" ==== Token: {elem['token']} ({round(elem['match_score'],5)}) ====")
# #         if verbose:
# #             print(f" = Dataset: {elem['dataset']}, Position: {elem['position']}")
# #             print(f" = Text: {elem['prev_tokens']}")
# #             print(f" = experts: {elem['experts']}")
# #             print(f" = scores: {[round(score, 3) for score in elem['scores']]}")
# os.makedirs(SAVE_ARTIFACTS_PATH_DIR, exist_ok=True)
# # for LAYER_IDX in range(16):
# #     stats = find_expert_coactivations(
# #         datasets = [dataset for dataset in all_datasets.values()],
# #         layer = LAYER_IDX,
# #         order_sensitive = True,
# #         keep_n = 12
# #     )

# #     save_coactivation_report(
# #         coactivation_stats = stats,
# #         tokenizer = extractor.tokenizer,
# #         save_path = os.path.join(SAVE_ARTIFACTS_PATH_DIR, f"layer_{LAYER_IDX}_coactivation.txt"),
# #         top_n = 120
# #     )
# coactivations = plot_expert_coactivation_heatmap(
#     datasets = [dataset for dataset in all_datasets.values()],
#     num_experts=64,
#     layer = LAYER_IDX,
#     figsize = (16, 12),
#     # save_path = os.path.join(SAVE_ARTIFACTS_PATH_DIR, f"layer_{LAYER_IDX}_coactivation_top2_coactivations.png"),
# )
# print(coactivations[0])
# print(sum(coactivations[0]))
# stats = find_expert_coactivations(
#     datasets = [dataset for dataset in all_datasets.values()],
#     layer = LAYER_IDX,
#     order_sensitive = True,
#     keep_n = 12
# )

# # save_top_coactivating_examples(
# #     coactivation_matrix = coactivations,
# #     coactivation_stats = stats,
# #     tokenizer = extractor.tokenizer,
# #     save_path = os.path.join(SAVE_ARTIFACTS_PATH_DIR, f"layer_{LAYER_IDX}_common_coactivations.txt"),
# #     threshold = 0.25
# # )
# EXPERT_IDS = [1, 25, 37, 29, 59, 5, 46, 49, 42, 54, 32, 61]
# for EXPERT_ID in EXPERT_IDS:
#     save_single_expert_report(
#         expert_id = EXPERT_ID,
#         coactivation_matrix = coactivations,
#         coactivation_stats = stats,
#         tokenizer = extractor.tokenizer,
#         save_path = os.path.join(SAVE_ARTIFACTS_PATH_DIR, f"L{LAYER_IDX}_E{EXPERT_ID}.txt"),
#         freq_threshold=0.08
#     )