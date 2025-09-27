import os
from typing import Dict, List
import torch
from datasets import load_dataset

from moe.activation_extraction.run_caching_utils import *
from moe.path_config import *

def verify_saved_activations(
    save_path: str,
    model_name: str = "allenai/OLMoE-1B-7B-0125-Instruct",
    cache_dir: str = MODEL_CACHE_DIR):
    """
    Load and verify a saved activation file.
    
    Args:
        save_path: Path to the saved activation file
    """
    print(f"\n🔍 Verifying: {save_path}")
    
    extractor = OLMoEActivationExtractor(model_name, cache_dir)
    loaded_data = extractor.load_activations(save_path)
    
    if loaded_data is None:
        print("❌ Failed to load file")
        return False
    
    for key in loaded_data.keys():
        if type(loaded_data[key]) == list:
            if type(loaded_data[key][-1]) == str or type(loaded_data[key][-1]) == list:
                print(f"{key}: {loaded_data[key][-1]}")
            elif type(loaded_data[key][-1]) == torch.Tensor:
                print(f"{key} shape: {loaded_data[key][-1].shape}")
            else:
                print(f"{key} is present")
        elif type(loaded_data[key]) == str:
            print(f"{key}: {loaded_data[key]}")
        else:
            print(f"{key} is present")

    return True

# Example usage functions
def run_single_dataset_test(
    model_name: str = "allenai/OLMoE-1B-7B-0125-Instruct",
    cache_dir: str = MODEL_CACHE_DIR,
    dataset_dir: str = DATASET_CACHE_DIR,
    base_save_dir: str = SAVE_ACTS_PATH_DIR,
    extract_fn: str = "extract_topk_routing_batch"):
    """Example: Process just GSM8K with questions only."""
    print("📝 Example: Processing GSM8K questions only...")
    
    # Initialize extractor
    extractor = OLMoEActivationExtractor(model_name, cache_dir)
    if not extractor.load_model_and_tokenizer():
        return
    # extractor.print_model_summary()
    # Process GSM8K
    save_paths = process_dataset_activations(
        dataset_name="gsm8k",
        extractor=extractor,
        num_samples=20,  # Just 20 samples for this example
        question_only=True,
        base_save_dir=base_save_dir,
        extract_fn=extract_fn
    )
    print("SAVE PATHS", save_paths)
    # Verify one of the saved files
    if save_paths:
        verify_saved_activations(list(save_paths.values())[0])

def run_custom_processing():
    """Example: Custom processing with specific parameters."""
    print("⚙️ Example: Custom processing...")
    
    # Initialize extractor with custom cache dir
    extractor = OLMoEActivationExtractor(
        cache_dir="/your/custom/cache/dir"  # Adjust as needed
    )
    
    if not extractor.load_model_and_tokenizer():
        return
    
    # Load dataset manually for more control
    dataset_dict = load_and_format_dataset(
        dataset_name="arc-easy",
        question_only=False,
        cache_dir=dataset_dir
    )
    
    # Process only test set with custom filtering
    test_texts = dataset_dict["test"][:25]  # Just 25 samples
    
    # Extract activations
    batch_data = extractor.extract_activations_batch(test_texts)
    
    # Add custom metadata
    batch_data["custom_metadata"] = {
        "processing_date": "2025-08-24",
        "custom_filter": "first_25_samples",
        "notes": "Custom processing example"
    }
    
    # Save with custom path
    custom_save_path = "/local/bys2107/research/data/custom_arc_easy.pt"
    extractor.save_activations(batch_data, custom_save_path)
    
    print(f"✅ Custom processing saved to: {custom_save_path}")

if __name__ == "__main__":
    # Run all datasets (uncomment to execute)
    # run_all_datasets(num_samples=100, batch_size=10)
    
    # Or run single dataset example
    print("Testing extract_topk_routing_batch")
    run_single_dataset_test(extract_fn = "extract_topk_routing_batch")
    print("Testing extract_activations_batch")
    run_single_dataset_test(extract_fn = "extract_activations_batch")
    
    # Or run custom processing
    # run_custom_processing()
    
    print("Done :)")