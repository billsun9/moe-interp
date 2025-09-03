import os
from moe.activation_extraction.activation_extractor_utils import *
from moe.path_config import *


if __name__ == "__main__":
    if test_single_activation(cache_dir=MODEL_CACHE_DIR, save_path = os.path.join(SAVE_ACTS_PATH_DIR, "test_single_acts.pt")):
        print("🎉 Single activation test passed!")
    else:
        print("⚠️ Single activation test failed")

    if test_batch_activation(cache_dir=MODEL_CACHE_DIR, save_path = os.path.join(SAVE_ACTS_PATH_DIR, "test_batch_acts.pt")):
        print("🎉 Batch activation test passed!")
    else:
        print("⚠️ Batch activation test failed")