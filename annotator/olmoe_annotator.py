import os
import pickle

# from moe.eda.eda_utils import load_all_activations
# from moe.eda.expert_coactivations import *
from moe.activation_extraction.activation_extractor_utils import OLMoEActivationExtractor
from moe.path_config import MODEL_CACHE_DIR, SAVE_ACTS_PATH_DIR, SAVE_ARTIFACTS_PATH_DIR

# start_time = time.time()
extractor = OLMoEActivationExtractor(cache_dir=MODEL_CACHE_DIR)
extractor.load_model_and_tokenizer()

model = extractor.model
tokenizer = extractor.tokenizer

# Load from file
with open(os.path.join(SAVE_ACTS_PATH_DIR, "l7_coactivations.pkl"), 'rb') as f:
    data = pickle.load(f)

def get_and_format_samples(d):
    tmp, res = [], []
    for ds, val in d["per_dataset"].items():
        examples = val['examples']
        tmp.extend(examples)
    
    tmp = sorted(tmp, key=lambda x: x['combined_score'], reverse=True)[:30]
    for e in tmp:
        s = f"'{e['formatted_token']}'"
        res.append(s)
    return ", ".join(res)

msg1 = get_and_format_samples(data[(37,)])

msg2 = get_and_format_samples(data[(29, 32)])

msg3 = get_and_format_samples(data[(0,)])

msg4 = get_and_format_samples(data[(59, 5)])

msg5 = get_and_format_samples(data[(32, 9)])

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

instructions = "What high-level semantic pattern, if any, do you see in this unordered collection of tokens? Be very concise.\nAnswer:"

# Batch of prompts
prompts = [
    f"Carefully consider the following unordered set of tokens: {msg1}\n{instructions}",
    f"Carefully consider the following unordered set of tokens: {msg2}\n{instructions}",
    f"Carefully consider the following unordered set of tokens: {msg3}\n{instructions}",
    f"Carefully consider the following unordered set of tokens: {msg4}\n{instructions}",
    f"Carefully consider the following unordered set of tokens: {msg5}\n{instructions}",
]

# Tokenize batch
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=50,       # Limit to 50 new tokens
    do_sample=True,          # Optional: enable sampling for more varied output
    top_p=0.95,              # Optional: nucleus sampling
    temperature=0.2,         # Optional: temperature control
    pad_token_id=tokenizer.eos_token_id  # Prevents warning if model lacks a pad token
)

# Decode
generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Print results
for i, gen in enumerate(generated_texts):
    print(f"\n=== Prompt {i} ===\n{gen}")
