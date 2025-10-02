import os
from typing import List, Dict

from moe.eda.eda_utils import load_all_activations
from moe.eda.expert_coactivations import *
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
    ("arxiv", False),
    ("personas", False)
]

d = load_all_activations(
    datasets_to_process = datasets_to_process,
    extractor = extractor,
    split_names = ["train"]
)

_, combined_frequencies_1 = get_frequencies([v for k, v in d.items()], layer=7, top_k = 1)
_, combined_frequencies_2 = get_frequencies([v for k, v in d.items()], layer=7, top_k = 2)
_, combined_frequencies_3 = get_frequencies([v for k, v in d.items()], layer=7, top_k = 3)

combined = combined_frequencies_1 + combined_frequencies_2 + combined_frequencies_3

print((1,), combined[(1,)])
print((10,), combined[(10,)])
print((10,20), combined[(10,20)])
print((10,21), combined[(10,21)])
print((10,22), combined[(10,22)])
print((10,20) in combined)
print((10,20,105) in combined)

print(combined.total())

cnt = 0
for key in combined:
    if combined[key] >= 10:
        cnt += 1


print(cnt, len(combined))

coactivations = find_expert_coactivations(d["arxiv"], layer=7, top_k = 1, order_sensitive = True)

# this formats in place
format_expert_coactivations(coactivations, tokenizer=extractor.tokenizer)