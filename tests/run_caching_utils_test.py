from moe.activation_extraction.run_caching_utils import ActivationProcessor

def test_process_sample_texts_topk(verbose: bool = False):
    processor = ActivationProcessor(
        extract_fn="extract_topk_routing_batch",
        layer_idx=8,
        batch_size=2
    )
    sample_texts = ["What is 2+2?", "Explain why the sky is blue.", "What's the largest mammal?", "Why is the ocean blue?", "Free diddy"]
    save_path = processor.process_sample_texts(sample_texts, dataset_name="custom_test_topk")

    assert save_path.endswith(".pt"), "❌ Save path invalid"

    if verbose:
        extractor = processor.extractor
        extractor.verify_saved_activations(save_path)


def test_process_sample_texts_mlp(verbose: bool = False):
    processor = ActivationProcessor(
        extract_fn="extract_activations_mlp_batch",
        layer_idx=10,
        batch_size=2
    )
    sample_texts = ["The capital of France is?", "Deep learning is powerful."]
    save_path = processor.process_sample_texts(sample_texts, dataset_name="custom_test_mlp")

    assert save_path.endswith(".pt"), "❌ Save path invalid"

    if verbose:
        extractor = processor.extractor
        extractor.verify_saved_activations(save_path)


def test_process_dataset_topk(verbose: bool = False):
    processor = ActivationProcessor(
        extract_fn="extract_topk_routing_batch",
        layer_idx=6,
        batch_size=4
    )
    save_paths = processor.process_dataset("gsm8k", num_samples=5, question_only=True, splits=["train"])

    assert "train" in save_paths, "❌ Missing train split"

    if verbose:
        extractor = processor.extractor
        for path in save_paths.values():
            extractor.verify_saved_activations(path)


def test_process_dataset_mlp(verbose: bool = False):
    processor = ActivationProcessor(
        extract_fn="extract_activations_mlp_batch",
        layer_idx=12,
        batch_size=4
    )
    save_paths = processor.process_dataset("gsm8k", num_samples=5, question_only=False, splits=["train"])

    assert "train" in save_paths, "❌ Missing train split"

    if verbose:
        extractor = processor.extractor
        for path in save_paths.values():
            extractor.verify_saved_activations(path)

def test_process_dataset_prerouting_and_output(verbose: bool = False):
    processor = ActivationProcessor(
        extract_fn="extract_prerouting_and_combined_output_batch",
        layer_idx=12,
        batch_size=4
    )
    save_paths = processor.process_dataset("gsm8k", num_samples=5, question_only=False, splits=["train"])

    assert "train" in save_paths, "❌ Missing train split"

    if verbose:
        extractor = processor.extractor
        for path in save_paths.values():
            extractor.verify_saved_activations(path)


if __name__ == "__main__":
    test_process_dataset_topk(verbose=True)
    test_process_dataset_mlp(verbose=True)
    test_process_dataset_prerouting_and_output(verbose=True)
    test_process_sample_texts_topk(verbose=True)
    test_process_sample_texts_mlp(verbose=True)
    