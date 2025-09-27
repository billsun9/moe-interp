import os
import pickle
import torch

from moe.activation_extraction.activation_extractor_utils import OLMoEActivationExtractor
from moe.path_config import MODEL_CACHE_DIR, SAVE_ACTS_PATH_DIR

# ------------------------------
# Load model and tokenizer
# ------------------------------
extractor = OLMoEActivationExtractor(cache_dir=MODEL_CACHE_DIR)
extractor.load_model_and_tokenizer()
model = extractor.model
tokenizer = extractor.tokenizer
print(model.device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Load precomputed coactivations
# ------------------------------
with open(os.path.join(SAVE_ACTS_PATH_DIR, "l7_coactivations.pkl"), "rb") as f:
    data = pickle.load(f)

# ------------------------------
# Helper: format examples for a permutation
# ------------------------------
def get_and_format_samples(d, max_n: int = 30):
    tmp = []
    for ds, val in d["per_dataset"].items():
        tmp.extend(val.get("examples", []))
    tmp = sorted(tmp, key=lambda x: x["combined_score"], reverse=True)[:max_n]
    return ", ".join(f"'{e['formatted_token']}'" for e in tmp)

# ------------------------------
# Auto-description function
# ------------------------------
def annotate_with_auto_descriptions(data, batch_size: int = 8):
    instructions = (
        "What high-level semantic pattern, if any, do you see in this unordered collection of tokens? "
        "Be very concise.\nAnswer:"
    )

    perms = list(data.keys())
    for i in range(0, len(perms), batch_size):
        batch_perms = perms[i:i+batch_size]
        prompts = []
        for perm in batch_perms:
            samples_str = get_and_format_samples(data[perm])
            if not samples_str:
                print(f"Something's broken: {perm}")
                prompts.append(f"No examples provided.\n{instructions}")
            else:
                prompts.append(f"Carefully consider the following unordered set of tokens: {samples_str}\n{instructions}")

        # Tokenize and move to device
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=True,
                top_p=0.95,
                temperature=0.2,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Update in place
        for perm, desc in zip(batch_perms, generated_texts):
            # Strip the prompt from the generated output if necessary
            if desc.startswith(prompts[0][:30]):  # crude check in case model echos input
                desc = desc.split("Answer:")[-1].strip()
            data[perm]["auto_description"] = desc

        print(f"Annotated {i+len(batch_perms)} / {len(perms)} permutations")

    return data

# ------------------------------
# Run annotation
# ------------------------------
data = annotate_with_auto_descriptions(data, batch_size=16)

# Optionally save updated data
with open(os.path.join(SAVE_ACTS_PATH_DIR, "l7_coactivations_annotated.pkl"), "wb") as f:
    pickle.dump(data, f)

print("Finished annotating all permutations!")
