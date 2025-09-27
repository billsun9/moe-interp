import os
from typing import Dict, List
import torch
from datasets import load_dataset
from .data_utils import load_and_format_dataset
from .activation_extractor_utils import OLMoEActivationExtractor

from moe.path_config import MODEL_CACHE_DIR, SAVE_ACTS_PATH_DIR, DATASET_CACHE_DIR

def process_sample_texts(
    sample_texts: List[str],
    extractor: "OLMoEActivationExtractor",
    dataset_name: str,
    base_save_dir: str = os.path.join(SAVE_ACTS_PATH_DIR, "custom"),
    batch_size: int = 4,
    metadata=None,
    extract_fn: str = "extract_activations_batch",  # or "extract_topk_routing_batch"
) -> str:
    """
    Process a list of custom sample texts and extract activations or topK routing.

    Args:
        sample_texts: List of texts to process
        extractor: Initialized OLMoEActivationExtractor instance
        dataset_name: Name of dataset (used for folder + filename)
        base_save_dir: Base directory for saving .pt files
        batch_size: Number of samples to process in each batch
        metadata: Optional metadata dictionary or string
        extract_fn: Which extractor function to call
                    ("extract_activations_batch" or "extract_topk_routing_batch")

    Returns:
        Path to the saved file
    """
    num_samples = len(sample_texts)
    print(f"\n{'='*60}")
    print(f"Processing custom texts for {dataset_name} with {extract_fn}...")
    print(f"Total samples: {num_samples}")
    print(f"Batch size: {batch_size}")

    # Save path
    save_filename = f"{dataset_name}_custom_{num_samples}samples.pt"
    save_path = os.path.join(base_save_dir, dataset_name, save_filename)

    # Base container
    all_batch_data = {
        "dataset_name": dataset_name,
        "split": "",  # no split
        "question_only": "",
        "texts": [],
        "tokens": [],
        "metadata": metadata if metadata else ""
    }

    # Add correct keys depending on extractor type
    if extract_fn == "extract_activations_batch":
        all_batch_data.update({"prerouting_logits": [], "routing_logits": []})
    elif extract_fn == "extract_topk_routing_batch":
        all_batch_data.update({"topk_indices": [], "topk_scores": []})
    else:
        raise ValueError(f"Unknown extract_fn: {extract_fn}")

    try:
        extract_func = getattr(extractor, extract_fn)
        batch_data = extract_func(sample_texts, batch_size=batch_size)

        for k in all_batch_data.keys():
            if k in batch_data:
                all_batch_data[k].extend(batch_data[k])

        print(f"    ✅ Batch completed ({len(sample_texts)} samples)")

        if extractor.save_activations(all_batch_data, save_path):
            print(f"✅ Saved custom activations: {save_path}")
        else:
            print(f"❌ Failed to save activations")

    except Exception as e:
        print(f"❌ Error processing custom texts: {e}")
        return ""

    return save_path


def process_dataset_activations(
    dataset_name: str,
    extractor: "OLMoEActivationExtractor",
    num_samples: int = 25,
    base_save_dir: str = SAVE_ACTS_PATH_DIR,
    question_only: bool = False,
    batch_size: int = 10,
    metadata=None,
    dataset_dir: str = DATASET_CACHE_DIR,
    extract_fn: str = "extract_activations_batch",  # or "extract_topk_routing_batch",
    splits: List[str] = ["train", "test"]
) -> Dict[str, str]:
    """
    Process a dataset and extract activations or topK routing.

    Args:
        dataset_name: Name of dataset ("gsm8k", "arc-easy", "arc-challenge")
        extractor: Initialized OLMoEActivationExtractor instance
        num_samples: Number of samples to process per split
        base_save_dir: Base directory for saving .pt files
        question_only: Whether to use only questions or full formatted text
        batch_size: Number of samples to process in each batch
        metadata: Optional metadata dictionary or string
        dataset_dir: Dataset cache location
        extract_fn: Which extractor function to call
                    ("extract_activations_batch" or "extract_topk_routing_batch")

    Returns:
        Dict mapping split names to save paths
    """
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name} dataset with {extract_fn}...")
    print(f"Samples per split: {num_samples}")
    print(f"Question only: {question_only}")
    print(f"Batch size: {batch_size}")

    try:
        dataset_dict = load_and_format_dataset(
            dataset_name=dataset_name,
            question_only=question_only,
            cache_dir=dataset_dir,
            splits=splits
        )
        print(f"✅ Successfully loaded {dataset_name}")
    except Exception as e:
        print(f"❌ Failed to load dataset {dataset_name}: {e}")
        return {}

    save_paths = {}
    extract_func = getattr(extractor, extract_fn)

    for split_name, texts in dataset_dict.items():
        print(f"\n--- Processing {dataset_name} {split_name} split ---")

        texts_subset = texts[:num_samples]
        actual_samples = len(texts_subset)
        print(f"Processing {actual_samples} samples from {split_name}")

        if actual_samples == 0:
            print(f"⚠️ No samples found in {split_name} split")
            continue

        question_suffix = "_questions" if question_only else "_full"
        save_filename = f"{dataset_name}_{split_name}{question_suffix}_{actual_samples}samples.pt"
        save_path = os.path.join(base_save_dir, dataset_name, save_filename)

        all_batch_data = {
            "dataset_name": dataset_name,
            "split": split_name,
            "question_only": question_only,
            "texts": [],
            "tokens": [],
            "metadata": metadata if metadata else ""
        }

        if extract_fn == "extract_activations_batch":
            all_batch_data.update({"prerouting_logits": [], "routing_logits": []})
        elif extract_fn == "extract_topk_routing_batch":
            all_batch_data.update({"topk_indices": [], "topk_scores": []})
        else:
            raise ValueError(f"Unknown extract_fn: {extract_fn}")

        batch_data = extract_func(texts_subset, batch_size=batch_size)

        for k in all_batch_data.keys():
            if k in batch_data:
                all_batch_data[k].extend(batch_data[k])

        if extractor.save_activations(all_batch_data, save_path):
            save_paths[split_name] = save_path
            print(f"✅ Saved {split_name} activations: {save_path}")
        else:
            print(f"❌ Failed to save {split_name} activations")

    return save_paths

def run_all_datasets(
    datasets_to_process,
    num_samples: int = 100,
    base_save_dir: str = SAVE_ACTS_PATH_DIR,
    batch_size: int = 4,
    cache_dir: str = MODEL_CACHE_DIR,
    dataset_dir: str = DATASET_CACHE_DIR,
    extract_fn: str = "extract_topk_routing_batch", # or "extract_activations_batch",
    splits: List[str] = ["train", "test"]
):
    """
    Run activation extraction on all supported datasets.
    
    Args:
        num_samples: Number of samples per split
        base_save_dir: Base directory for saving dataset activations
        batch_size: Batch size for processing
    """
    print("🚀 Starting dataset activation extraction on all datasets...")
    
    extractor = OLMoEActivationExtractor(cache_dir = cache_dir)

    if not extractor.load_model_and_tokenizer():
        return
    
    all_results = {}
    
    for dataset_name, question_only in datasets_to_process:
        try:
            save_paths = process_dataset_activations(
                dataset_name=dataset_name,
                extractor=extractor,
                num_samples=num_samples,
                base_save_dir=base_save_dir,
                question_only=question_only,
                batch_size=batch_size,
                dataset_dir=dataset_dir,
                extract_fn=extract_fn,
                splits=splits
            )
            
            key = f"{dataset_name}_{'questions' if question_only else 'full'}"
            all_results[key] = save_paths
            
        except Exception as e:
            print(f"❌ Failed to process {dataset_name} (question_only={question_only}): {e}")
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 EXTRACTION SUMMARY")
    print(f"{'='*60}")
    
    for key, save_paths in all_results.items():
        print(f"\n{key}:")
        if save_paths:
            for split, path in save_paths.items():
                print(f"  {split}: {path}")
        else:
            print("  ❌ No files saved")
    
    print(f"\n🎉 Completed processing {len(all_results)} dataset configurations!")