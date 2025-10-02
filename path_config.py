import os

def resolve_base_path():
    # if both exist, /local/bys2107 takes precedence
    for path in ["/local/bys2107", "/pmglocal/bys2107"]:
        if os.path.exists(path):
            print(f"Using base path: {path}")
            return path
    raise FileNotFoundError("Neither /pmglocal nor /local paths were found.")

BASE_PATH = resolve_base_path()
MODEL_CACHE_DIR = os.path.join(BASE_PATH, "hf_cache") # HF model weights
DATASET_CACHE_DIR = os.path.join(BASE_PATH, "datasets_cache") # datasets
SAVE_ACTS_PATH_DIR = os.path.join(BASE_PATH, "research/data/OLMoE-acts/") # cached activations
SAVE_FIGS_PATH_DIR = os.path.join(BASE_PATH, "research/data/figs/") # figures
SAVE_ARTIFACTS_PATH_DIR = os.path.join(BASE_PATH, "research/data/artifacts/") # e.g. log files
SNMF_DATA_DIR = os.path.join(BASE_PATH, "research/moe/snmf") # This is used just to access SNMF data path