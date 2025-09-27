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

combined_coactivations = concatenate_expert_coactivations(
    d,
    tokenizer = extractor.tokenizer,
    filter_threshold = 10,
    keep_n = 8
)
import pickle

# Save to file
with open(os.path.join(SAVE_ACTS_PATH_DIR, "l7_coactivations.pkl"), 'wb') as f:
    pickle.dump(combined_coactivations, f)

print(f"Time Elapsed: {time.time() - start_time}")
# from pymongo.mongo_client import MongoClient

# from typing import Dict, Any, List

# def save_coactivations_to_mongo(coactivations: Dict, collection, batch_size: int = 50):
#     """
#     Save coactivation dict to MongoDB in batches.
#     Each permutation entry becomes one document.
#     """
#     docs: List[Dict[str, Any]] = []
#     for perm, info in coactivations.items():
#         docs.append({
#             "expert_order": list(perm),          # store tuple as list
#             "count": info["count"],
#             "per_dataset": info["per_dataset"],  # already JSON serializable
#             "next": info["next"],                # list of ints
#         })

#     total = len(docs)
#     for i in range(0, total, batch_size):
#         batch = docs[i:i+batch_size]
#         collection.insert_many(batch)
#         print(f"Inserted batch {i//batch_size + 1}, size={len(batch)}")
#         time.sleep(1)
#     print(f"✅ Saved {total} documents total")



# # Usage
# uri = "123"
# client = MongoClient(uri)



# # Suppose coactivations = concatenate_expert_coactivations(...)
# save_coactivations_to_mongo(combined_coactivations, collection)
# print("Saved all!")

# out, combined_dataset = get_frequencies([v for k, v in d.items()], layer=7, top_k = 1)

# print(len(combined_dataset), combined_dataset.total())

# cnt = 0
# for permutation in combined_dataset:
#     if combined_dataset[permutation] >= 10:
#         cnt += 1
# print(f"[Permutations with >= 10 samples]: {cnt} / {len(combined_dataset)}")

# print(out.keys())
# print(len(out['lex_glue']))
# print(list(out['lex_glue'].keys())[:5])
# print("lex_glue", out['lex_glue'].total())
# print("arxiv", out['arxiv'].total())
# print(out['lex_glue'])

# print(d.keys())
# for key in d["personas"]:
#     print(key)
#     print(type(d["personas"][key]))
#     if type(d["personas"][key]) == list:
#         print(len(d["personas"][key]))
#         if type(d["personas"][key][10]) == list:
#             print(d["personas"][key][10], len(d["personas"][key][10]))
#         elif type(d["personas"][key][10]) == str:
#             print(d["personas"][key][10])
#         else:
#             print(d["personas"][key][10].shape)
#     print("\n")


# coactivations = find_expert_coactivations(d["arxiv"], layer=7, top_k = 1, order_sensitive = True)

# # this formats in place
# format_expert_coactivations(coactivations, tokenizer=extractor.tokenizer)
# print("Num Keys:", len(coactivations.keys()))

# print("Keys of interest:", list(coactivations.keys())[0:8])
# for key in list(coactivations.keys())[0:8]:
#     print(key, combined_dataset[key], "\n", out['arxiv'][key], coactivations[key])
#     print("\n\n")

# running_sum = 0
# for dataset_name in out:
#     print(dataset_name, list(coactivations.keys())[0], out[dataset_name][list(coactivations.keys())[0]])
#     running_sum += out[dataset_name][list(coactivations.keys())[0]]

# print(running_sum)
# for key in list(coactivations.keys())[0:8]:
#     print(f"\t[Expert Permutation: {key}, count: {coactivations[key]['count']}]")
#     for top_example in coactivations[key]['top_examples']:
#         print(top_example)
#     print("\n\n")
# for dataset_name in d:
#     num_tokens = sum(len(d[dataset_name]["tokens"][sample_idx]) for sample_idx in range(len(d[dataset_name]["tokens"])))
#     num_samples = len(d[dataset_name]["tokens"])
#     print(f"Dataset: {dataset_name}, num tokens: {num_tokens}, num samples: {num_samples}")