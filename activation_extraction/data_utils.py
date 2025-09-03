from datasets import load_dataset
from typing import List, Tuple, Dict

from moe.path_config import DATASET_CACHE_DIR

def load_and_format_dataset(
    dataset_name: str,
    subset: str = None,
    cache_dir: str = DATASET_CACHE_DIR,
    question_only: bool = False
) -> Dict[str, List[str]]:
    """
    Load and format GSM8K or ARC datasets.

    Args:
        dataset_name (str): One of ["gsm8k", "arc-easy", "arc-challenge","sciq"]
        subset (str): Optional Hugging Face subset (defaults are used if None)
        cache_dir (str): Path to Hugging Face dataset cache
        question_only (bool): If True, return only questions

    Returns:
        dict: {
            'train': List of formatted strings,
            'test': List of formatted strings
        }
    """
    # Map dataset_name to HuggingFace IDs
    dataset_map = {
        "gsm8k": ("openai/gsm8k", "main"),
        "arc-easy": ("allenai/ai2_arc", "ARC-Easy"),
        "arc-challenge": ("allenai/ai2_arc", "ARC-Challenge"),
        "sciq": ("allenai/sciq", "default")
    }

    assert dataset_name in dataset_map, f"Unsupported dataset: {dataset_name}"

    hf_name, hf_subset = dataset_map[dataset_name]
    dataset = load_dataset(hf_name, hf_subset if subset is None else subset, cache_dir=cache_dir)

    def format_gsm8k(example, q_only=False):
        if q_only:
            return f"{example['question'].strip()}"
        return f"Question: {example['question'].strip()}\nAnswer: {example['answer'].strip()}"

    def format_arc(example, q_only=False):
        choices = example['choices']
        choice_str = "\n".join(f"{label}. {text}" for label, text in zip(choices["label"], choices["text"]))
        if q_only:
            return f"{example['question'].strip()}"
        return f"Question: {example['question'].strip()}\nChoices:\n{choice_str}\nAnswer: {example['answerKey']}"
    
    def format_sciq(example, q_only=False):
        if q_only:
            return f"{example['question'].strip()}"
        return f"Question: {example['question'].strip()}\nReasoning: {example['support'].strip()}\nAnswer: {example['correct_answer']}"

    if "gsm8k" in dataset_name:
        formatter = lambda ex: format_gsm8k(ex, q_only=question_only)
    elif "arc" in dataset_name:
        formatter = lambda ex: format_arc(ex, q_only=question_only)
    elif "sciq" in dataset_name:
        formatter = lambda ex: format_sciq(ex, q_only=question_only)

    return {
        split: [formatter(ex) for ex in dataset[split]]
        for split in dataset.keys() if split in ["train", "test"]
    }


def test_load_and_format_dataset(cache_dir = DATASET_CACHE_DIR):
    dataset_names = ["gsm8k", "arc-easy", "arc-challenge", "sciq"]

    for name in dataset_names:
        print(f"\n=== Testing {name.upper()} ===")
        
        for question_only in [False, True]:
            print(f"\n--- question_only={question_only} ---")
            try:
                data = load_and_format_dataset(
                    dataset_name=name,
                    cache_dir=cache_dir,
                    question_only=question_only
                )

                # Show counts and examples
                for split in data:
                    print(f"\n[{name} - {split}]: {len(data[split])} examples")
                    for i, example in enumerate(data[split][:3]):
                        print(f"\nExample {i + 1}:\n{example}")

            except Exception as e:
                print(f"Error loading {name} with question_only={question_only}: {e}")

