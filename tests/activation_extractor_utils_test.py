from moe.activation_extraction.activation_extractor_utils import *

if __name__ == "__main__":
    CACHE_DIR = "/local/bys2107/hf_cache"
    SAVE_PATH_DIR = "/local/bys2107/research/data/OLMoE-acts/"
    if test_single_activation(cache_dir=CACHE_DIR, save_path = SAVE_PATH_DIR + "test_single_acts.pt"):
        print("🎉 Single activation test passed!")
    else:
        print("⚠️ Single activation test failed")

    if test_batch_activation(cache_dir=CACHE_DIR, save_path = SAVE_PATH_DIR + "test_single_acts.pt"):
        print("🎉 Batch activation test passed!")
    else:
        print("⚠️ Batch activation test failed")