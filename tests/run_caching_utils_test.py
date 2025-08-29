import os
from typing import Dict, List
import torch
from datasets import load_dataset

from moe.activation_extraction.run_caching_utils import *

def verify_saved_activations(
    save_path: str,
    model_name: str = "allenai/OLMoE-1B-7B-0125-Instruct",
    cache_dir: str = "/local/bys2107/hf_cache"):
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
    
    # Additional dataset-specific info
    if "dataset_name" in loaded_data:
        print(f"  Dataset: {loaded_data['dataset_name']}")
        print(f"  Split: {loaded_data['split']}")
        print(f"  Question only: {loaded_data['question_only']}")
        print("TYPE", type(loaded_data['prerouting_logits']))
        if type(loaded_data['prerouting_logits']) == list:
            print(f"  Number of samples: {len(loaded_data['prerouting_logits'])}")
            print(f"  Prerouting logits shape: {loaded_data['prerouting_logits'][0].shape}")
            print(f"  Prerouting logits shape: {loaded_data['routing_logits'][0].shape}")
        else:
            print(f"  Number of samples: 1")
    
    return True

# Example usage functions
def run_single_dataset_test(
    model_name: str = "allenai/OLMoE-1B-7B-0125-Instruct",
    cache_dir: str = "/local/bys2107/hf_cache",
    dataset_dir: str = "/local/bys2107/datasets_cache",
    base_save_dir: str = "/local/bys2107/research/data/OLMoE-acts"):
    """Example: Process just GSM8K with questions only."""
    print("📝 Example: Processing GSM8K questions only...")
    
    # Initialize extractor
    extractor = OLMoEActivationExtractor(model_name, cache_dir)
    if not extractor.load_model_and_tokenizer():
        return
    
    # Process GSM8K
    save_paths = process_dataset_activations(
        dataset_name="gsm8k",
        extractor=extractor,
        num_samples=20,  # Just 20 samples for this example
        question_only=True,
        base_save_dir=base_save_dir
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
    CACHE_DIR = "/local/bys2107/hf_cache"
    BASE_SAVE_DIR = "/local/bys2107/research/data/OLMoE-acts"
    run_single_dataset_test(cache_dir = CACHE_DIR, base_save_dir = BASE_SAVE_DIR)
    
    # Or run custom processing
    # run_custom_processing()
    
    print("Done :)")