from moe.activation_extraction.run_caching_utils import run_all_datasets

datasets_to_process = [
    #("gsm8k", False),           # GSM8K with full questions + answers
    ("gsm8k", True),            # GSM8K with questions only
    #("arc-easy", False),        # ARC-Easy with full format
    # ("arc-easy", True),         # ARC-Easy with questions only
    #("arc-challenge", False),   # ARC-Challenge with full format
    # ("arc-challenge", True),    # ARC-Challenge with questions only
    #("sciq", False),            # sciq with full format
    ("sciq", True),             # sciq with questions only
]
num_samples = 50
run_all_datasets(datasets_to_process, num_samples)