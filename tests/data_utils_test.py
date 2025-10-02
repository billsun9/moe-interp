from moe.activation_extraction.data_utils import load_and_format_dataset
from moe.path_config import *

def test_load_and_format_dataset(cache_dir = DATASET_CACHE_DIR):
    dataset_names = [
        # False ==> Question: {Q} [Optional: {Choices} or {Reasoning}] Answer:{A}
        # ("gsm8k", False),
        # ("arc-easy", False),
        # ("arc-challenge", False),
        # ("sciq", False),
        # ("mbpp", False),
        # # False ==> Not a question. Just the {statement}
        # ("ag_news", False),
        # ("imdb_pos", False),
        # ("imdb_neg", False),
        # False/True does not matter for these. Just the {statement}
        ("poetry", False),
        ("lex_glue", False),
        ("arxiv", False),
        ("personas", False)
    ]

    for name, _ in dataset_names:
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
                    for i, example in enumerate(data[split][:1]):
                        print(f"\nExample {i + 1}:\n{example}")

            except Exception as e:
                print(f"Error loading {name} with question_only={question_only}: {e}")

if __name__ == "__main__":
    test_load_and_format_dataset(cache_dir = DATASET_CACHE_DIR)