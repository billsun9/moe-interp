import os
import random
import numpy as np
import torch
from typing import List
from tqdm import tqdm
import time

from moe.path_config import SNMF_DATA_DIR
from moe.activation_extraction.activation_extractor_utils import OLMoEActivationExtractor
from moe.snmf.data_utils.concept_dataset import SupervisedConceptDataset
from moe.snmf.utils import *

def extract_multiple_layer_activations(
    extractor,
    dataset,
    layers: List[int],
    batch_size: int = 4,
    mode: str = "prerouting"
) -> List[torch.Tensor]:
    """
    Adapter function to extract activations from multiple layers using OLMoEActivationExtractor.
    Mimics the output format of ActivationGenerator.generate_multiple_layer_activations_and_freq()
    but returns only activations (no frequency).
    
    Args:
        extractor: An instance of OLMoEActivationExtractor with model and tokenizer loaded
        dataset: SupervisedConceptDataset instance
        layers: List of layer indices to extract activations from
        batch_size: Batch size for processing (used by extract_prerouting_and_combined_output_batch)
        mode: Either "prerouting" (MLP input) or "combined_output" (MLP output)
    
    Returns:
        List of tensors, one per layer, each of shape (num_tokens, d_model)
    """
    if mode not in ["prerouting", "combined_output"]:
        raise ValueError(f"Mode must be 'prerouting' or 'combined_output', got {mode}")
    
    # Step 1: Convert dataset to list of texts
    all_texts = [dataset[i][0] for i in range(len(dataset))]  # Extract prompts only
    
    # Step 2: Initialize storage for each layer
    all_layer_activations = {}
    
    # Step 3: Extract activations for each layer (one call per layer, processes all texts)
    for layer_idx_pos, layer_idx in enumerate(tqdm(layers, desc="Extracting layers")):
        result = extractor.extract_prerouting_and_combined_output_batch(
            texts=all_texts,
            layer_idx=layer_idx,
            batch_size=batch_size
        )
        
        # Step 4: Extract the appropriate activations based on mode
        if mode == "prerouting":
            activations_list = result["prerouting_activations"]
        else:  # combined_output
            activations_list = result["combined_output"]
        
        # Step 5: Concatenate activations from all samples
        # activations_list is a list where each element is [seq_len_i, hidden_dim]
        # Concatenate along the sequence dimension to get [total_tokens, hidden_dim]
        layer_activations = torch.cat(activations_list, dim=0)
        all_layer_activations[layer_idx] =layer_activations

    return all_layer_activations

device = "cuda" if torch.cuda.is_available() else "cpu"
extractor = OLMoEActivationExtractor(device=device)
extractor.load_model_and_tokenizer()

# path to data
CONCEPT_DATASET_PATH = os.path.join(SNMF_DATA_DIR, "data/languages.json")
dataset = SupervisedConceptDataset(CONCEPT_DATASET_PATH)
BATCH_SIZE = 10
LAYERS = [7,8,9]
# NMF SPECIFIC
RANK=100
SPARSITY=0.01
MAX_ITERATIONS=2000
PATIENCE=50
NUM_FEATURES=10 # We log this many features, with 25 top samples

activations = extract_multiple_layer_activations(
    extractor,
    dataset,
    layers = LAYERS,
    batch_size = BATCH_SIZE,
    mode = "combined_output"
)

start_time = time.time()
print("Starting NMF")

from factorization.seminmf import NMFSemiNMF
# sparsity is percent of neurons to use in final features
for layer_idx in LAYERS:
    nmf = NMFSemiNMF(rank=RANK, fitting_device=device, sparsity=SPARSITY)

    # patience is how many epochs to wait for loss to improve
    # init can be svd and knn too, in terms of performance they are all the same
    # we need to tranpose activations to match literature's (dimension, num_samples)
    # we take activations[0] since its a list of activations (index for every layer you used when generating)
    nmf.fit(activations[7].T, max_iter=MAX_ITERATIONS, patience=PATIENCE)

    print(f"DONE! Time Elapsed: {time.time() - start_time}")
    tokens, sample_ids, labels = extract_token_ids_sample_ids_and_labels(dataset, extractor, batch_size=BATCH_SIZE)
    token_ds = generate_token_contexts(tokens, sample_ids, extractor, context_window=15)

    # 5. Save feature analysis to JSON
    save_feature_analysis_to_json(
        nmf=nmf,
        token_ds=token_ds,
        output_path=os.path.join(SNMF_DATA_DIR, f"L{layer_idx}_top_{NUM_FEATURES}.json"),
        num_features=10,
        num_samples=25
    )