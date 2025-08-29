from moe.activation_extraction.data_utils import test_load_and_format_dataset

if __name__ == "__main__":
    DATASET_CACHE_DIR = "/local/bys2107/datasets_cache"
    test_load_and_format_dataset(cache_dir = DATASET_CACHE_DIR)