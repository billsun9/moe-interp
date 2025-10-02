import os
import torch
from moe.activation_extraction.activation_extractor_utils import OLMoEActivationExtractor
from moe.path_config import *

def test_extract_topk_routing_batch(
    model_name: str = "allenai/OLMoE-1B-7B-0125-Instruct",
    cache_dir: str = MODEL_CACHE_DIR,
    layer_idx: int = 10,
    verbose: bool = True
):
    """
    Test batched extraction of top-k routing decisions.
    Verifies output structure, shapes, and probability validity.
    """
    extractor = OLMoEActivationExtractor(model_name=model_name, cache_dir=cache_dir)
    assert extractor.load_model_and_tokenizer(), "Failed to load model/tokenizer"

    test_texts = [
        "The quick brown fox",
        "Dog cat monkey ape baboon ostrich",
        "Machine learning is fascinating",
        "PyTorch makes deep learning accessible",
        "This is a longer sentence that should still tokenize properly."
    ]

    acts = extractor.extract_topk_routing_batch(test_texts, layer_idx=layer_idx, batch_size=2)

    expected_keys = {"texts", "tokens", "topk_indices", "topk_scores"}

    assert set(acts.keys()) == expected_keys
    # Batch length checks
    for k in expected_keys:
        assert len(acts[k]) == len(test_texts), f"{k} length mismatch"

    # Shape / score validity checks
    for toks, idxs, vals in zip(acts["tokens"], acts["topk_indices"], acts["topk_scores"]):
        S = len(toks)
        assert idxs.shape == (S, 8), f"indices shape mismatch {idxs.shape}"
        assert vals.shape == (S, 8), f"scores shape mismatch {vals.shape}"
        assert torch.all(vals >= 0) and torch.all(vals <= 1), "Scores not in [0,1]"
        row_sums = vals.sum(dim=-1)
        assert torch.all(row_sums <= 1.0 + 1e-4), "Row sum exceeds 1.0"

    print("[PASS] extract_topk_routing_batch test passed")
    if verbose: extractor.verify_saved_activations(acts)
    return True


def test_extract_activations_mlp_batch(
    model_name: str = "allenai/OLMoE-1B-7B-0125-Instruct",
    cache_dir: str = MODEL_CACHE_DIR,
    layer_idx: int = 10,
    verbose: bool = True
):
    """
    Test batched extraction of MLP activations.
    Verifies shape consistency and reconstruction correctness.
    """
    extractor = OLMoEActivationExtractor(model_name=model_name, cache_dir=cache_dir)
    assert extractor.load_model_and_tokenizer(), "Failed to load model/tokenizer"

    test_texts = [
        "The quick brown fox",
        "Dog cat monkey ape baboon ostrich",
        "Machine learning is fascinating",
        "PyTorch makes deep learning accessible",
        "This is a longer sentence that should still tokenize properly."
    ]
    acts = extractor.extract_activations_mlp_batch(test_texts, layer_idx=layer_idx, batch_size=2)
    expected_keys = {
        "texts",
        "tokens",
        "prerouting_activations",
        "router_logits",
        "router_probs",
        "topk_indices",
        "expert_outputs",
        "combined_output",
    }

    assert set(acts.keys()) == expected_keys
    # Batch length checks
    for k in expected_keys:
        assert len(acts[k]) == len(test_texts), f"{k} length mismatch"

    # Shape checks for one sample
    tokens = acts["tokens"][0]
    S = len(tokens)
    prerouting_activations = acts["prerouting_activations"][0]
    assert prerouting_activations.shape[0] == S, "prerouting_activations length mismatch"

    # Check reconstruction vs direct forward
    model = extractor.model
    device = extractor.device
    mlp_block = model.model.layers[layer_idx].mlp

    # Just test first sample for closeness
    combined = acts["combined_output"][0].to(device)      # [S, H]
    prerouting_activations = acts["prerouting_activations"][0].unsqueeze(0).to(device)  # [1, S, H]

    with torch.no_grad():
        direct_out, *_ = mlp_block(prerouting_activations)

    diff = torch.norm(direct_out.squeeze(0) - combined) / torch.norm(direct_out)
    print(f"[MLP batch test] Relative error = {diff.item():.6e}")
    assert diff.item() < 1e-3, "MLP batch extractor reconstruction mismatch"

    print("[PASS] extract_activations_mlp_batch test passed")
    if verbose: extractor.verify_saved_activations(acts)
    return True
    
def test_extract_prerouting_and_combined_output_batch(
    model_name: str = "allenai/OLMoE-1B-7B-0125-Instruct",
    cache_dir: str = MODEL_CACHE_DIR,
    layer_idx: int = 10,
    verbose: bool = True
):
    """
    Test that extract_prerouting_and_combined_output_batch produces the same
    prerouting_activations and combined_output as extract_activations_mlp_batch.
    """
    extractor = OLMoEActivationExtractor(model_name=model_name, cache_dir=cache_dir)
    assert extractor.load_model_and_tokenizer(), "Failed to load model/tokenizer"

    test_texts = [
        "The quick brown fox",
        "Dog cat monkey ape baboon ostrich",
        "Machine learning is fascinating",
        "PyTorch makes deep learning accessible",
        "This is a longer sentence that should still tokenize properly."
    ]

    full_out = extractor.extract_activations_mlp_batch(
        test_texts,
        layer_idx=layer_idx,
        batch_size=2
    )

    minimal_out = extractor.extract_prerouting_and_combined_output_batch(
        test_texts,
        layer_idx=layer_idx,
        batch_size=2
    )

    # Basic checks
    assert len(full_out["texts"]) == len(minimal_out["texts"]) == len(test_texts)
    assert len(full_out["prerouting_activations"]) == len(minimal_out["prerouting_activations"])
    assert len(full_out["combined_output"]) == len(minimal_out["combined_output"])

    # Check each sample
    for i in range(len(test_texts)):
        pr_full = full_out["prerouting_activations"][i]
        pr_min = minimal_out["prerouting_activations"][i]
        co_full = full_out["combined_output"][i]
        co_min = minimal_out["combined_output"][i]

        assert pr_full.shape == pr_min.shape, f"Prerouting shape mismatch at sample {i}"
        assert co_full.shape == co_min.shape, f"Combined output shape mismatch at sample {i}"

        # Value closeness
        pr_diff = torch.norm(pr_full - pr_min) / torch.norm(pr_full)
        co_diff = torch.norm(co_full - co_min) / torch.norm(co_full)

        if verbose:
            print(f"[Sample {i}] Prerouting rel error: {pr_diff:.2e}, Combined rel error: {co_diff:.2e}")

        assert pr_diff < 1e-5, f"Prerouting activations mismatch at sample {i}"
        assert co_diff < 1e-5, f"Combined output mismatch at sample {i}"

    print("[PASS] extract_prerouting_and_combined_output_batch test passed")
    return True


if __name__ == "__main__":
    # if test_extract_activations_mlp_batch():
    #     print("🎉 Batch MLP extraction test passed!")
    # else:
    #     print("⚠️ Batch MLP extraction test failed")
    # if test_extract_topk_routing_batch():
    #     print("🎉 Batch TopK Idxs/Weights test passed!")
    # else:
    #     print("⚠️ Batch TopK Idxs/Weights test failed")
    if test_extract_prerouting_and_combined_output_batch():
        print("🎉 Prerouting/Moe Output test passed!")
    else:
        print("⚠️ Prerouting/Moe Output test failed")