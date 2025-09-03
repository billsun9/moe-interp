from moe.activation_extraction.data_utils import test_load_and_format_dataset
from moe.path_config import *

if __name__ == "__main__":
    test_load_and_format_dataset(cache_dir = DATASET_CACHE_DIR)