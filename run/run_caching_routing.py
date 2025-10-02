from moe.activation_extraction.run_caching_utils import ActivationProcessor

datasets_to_process = [
    # False ==> Question: {Q} [Optional: {Choices} or {Reasoning}] Answer:{A}
    ("gsm8k", False),
    ("arc-easy", False),
    ("arc-challenge", False),
    ("sciq", False),
    ("mbpp", False),
    # False ==> Not a question. Just the {statement}
    ("ag_news", False),
    ("imdb_pos", False),
    ("imdb_neg", False),
    # False/True does not matter for these. Just the {statement}
    ("poetry", False),
    ("lex_glue", False),
    ("arxiv", False),
    ("personas", False)
]
NUM_SAMPLES = 500
BATCH_SIZE = 30
LAYER_IDX = 7
EXTRACT_FN = "extract_topk_routing_batch"
SPLITS = ["train"]
processor = ActivationProcessor(
    batch_size = BATCH_SIZE,
    layer_idx = LAYER_IDX,
    extract_fn = EXTRACT_FN
)

processor.run_all(
    datasets_to_process,
    num_samples = NUM_SAMPLES,
    splits = SPLITS
)