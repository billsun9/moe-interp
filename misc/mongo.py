import os
import time
import pickle
import random
from pymongo.mongo_client import MongoClient
from typing import Dict, Any, List

from moe.path_config import SAVE_ACTS_PATH_DIR

with open(os.path.join(SAVE_ACTS_PATH_DIR, "l7_coactivations_annotated.pkl"), "rb") as f:
    combined_coactivations = pickle.load(f)



def save_coactivations_to_mongo(coactivations: Dict, collection, batch_size: int = 50):
    docs: List[Dict[str, Any]] = []
    for perm, info in coactivations.items():
        doc = {
            "expert_order": list(perm),
            "desc": info["auto_description"],
            "count": info["count"],
            "per_dataset": info["per_dataset"],
            "next": info["next"],
        }
        # doc.pop("_id", None)  # prevent duplicate _id issues
        docs.append(doc)

    total = len(docs)
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        collection.insert_many(batch)
        print(f"Inserted batch {i//batch_size + 1}, size={len(batch)}")
        time.sleep(1)

    print(f"✅ Saved {total} documents total")



# Usage
uri = "123"
client = MongoClient(uri)
db = client["olmoe_instruct"]
collection = db["l7_permutations"]

# # Suppose coactivations = concatenate_expert_coactivations(...)
save_coactivations_to_mongo(combined_coactivations, collection)
print("Saved all!")
# print(collection.find_one({"expert_order": [21]}))
# client = MongoClient(uri)
# db = client["test"]
# collection = db["permutations"]
# document_to_insert = {
#         "name": "Bobby",
#         "age": 30,
#         "city": "New York"
#     }
# collection.insert_one(document_to_insert)