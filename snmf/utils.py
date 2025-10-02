import torch
import numpy as np
from typing import List, Tuple


def extract_token_ids_sample_ids_and_labels(dataset, extractor, batch_size: int = 4):
    """
    Extract token IDs, sample IDs, and labels for each token position across the dataset.
    
    Args:
        dataset: SupervisedConceptDataset instance
        extractor: OLMoEActivationExtractor instance with loaded model and tokenizer
        batch_size: Batch size for processing
    
    Returns:
        tokens: List of token IDs (one per token position across all samples)
        sample_ids: List of sample indices (which sample each token belongs to)
        labels: List of labels (label for each token's sample)
    """
    # Convert dataset to list of texts and labels
    all_texts = [dataset[i][0] for i in range(len(dataset))]
    all_labels = [dataset[i][1] for i in range(len(dataset))]
    
    tokens = []
    sample_ids = []
    labels = []
    
    # Process in batches
    for start_idx in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[start_idx:start_idx + batch_size]
        batch_labels = all_labels[start_idx:start_idx + batch_size]
        
        # Tokenize the batch
        enc = extractor.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            max_length=None
        )
        
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        
        # Extract tokens and track which sample they belong to
        for i in range(len(batch_texts)):
            # Get valid token positions (non-padding)
            valid_len = attention_mask[i].sum().item()
            sample_tokens = input_ids[i, :valid_len].tolist()
            
            # Add tokens, sample IDs, and labels
            tokens.extend(sample_tokens)
            sample_ids.extend([start_idx + i] * len(sample_tokens))
            labels.extend([batch_labels[i]] * len(sample_tokens))
    
    return tokens, sample_ids, labels


def generate_token_contexts(tokens, sample_ids, extractor, context_window: int = 15):
    """
    Generate token strings and their surrounding context for each token position.
    
    Args:
        tokens: List of token IDs
        sample_ids: List of sample indices (which sample each token belongs to)
        extractor: OLMoEActivationExtractor instance with loaded tokenizer
        context_window: Number of tokens before and after to include in context
    
    Returns:
        List of (token_str, context_str) tuples
    """
    token_ds = []
    
    for i in range(len(tokens)):
        current_sample_id = sample_ids[i]
        
        # Convert the current token to its string representation
        token_str = extractor.tokenizer.decode([tokens[i]])
        
        # Determine the start and end indices for the context window
        start = max(0, i - context_window)
        end = min(len(tokens), i + context_window + 1)
        
        # Get the string representation for each token in the context
        # Only include tokens from the same sample
        context_tokens = [
            extractor.tokenizer.decode([tokens[j]]) 
            for j in range(start, end) 
            if sample_ids[j] == current_sample_id
        ]
        
        # Join the context tokens into a single string
        context_str = "".join(context_tokens)
        
        # Append the (token, context) tuple to the list
        token_ds.append((token_str, context_str))
    
    return token_ds


def get_top_activating_indices(W, concept_idx, num_samples=10, minimal_activation=0):
    """
    Get indices and activations of tokens that most strongly activate a given concept.
    
    Args:
        W: Weight matrix from NMF (tokens x concepts)
        concept_idx: Index of the concept/feature to analyze
        num_samples: Number of top activating tokens to return
        minimal_activation: Minimum activation threshold (default 0)
    
    Returns:
        non_zero_indices: List of token indices with highest activations
        activations: List of corresponding activation values
    """
    activations = []
    non_zero_indices = []

    sample_importance = W[:, concept_idx]
    # Get indices of the top samples (highest activation values)
    top_indices = np.argsort(sample_importance)[-num_samples:]
    
    for i in top_indices:
        act = sample_importance[i]
        if act <= minimal_activation:
            continue
        activations.append(act)
        non_zero_indices.append(i)
   
    return non_zero_indices, activations


def save_feature_analysis_to_json(nmf, token_ds, output_path: str, num_features: int = 10, num_samples: int = 5):
    """
    Analyze NMF features and save results to a JSON file.
    
    Args:
        nmf: Fitted NMF model with G_ attribute
        token_ds: List of (token_str, context_str) tuples from generate_token_contexts
        output_path: Path to save the JSON file
        num_features: Number of features to analyze
        num_samples: Number of top activating tokens per feature
    """
    import json
    
    feature_analysis = {}
    
    for k in range(num_features):
        ti, ta = get_top_activating_indices(nmf.G_.cpu().detach(), k, num_samples=num_samples)
        
        top_activations = []
        for i, a in zip(ti, ta):
            top_activations.append({
                'token': token_ds[i][0],
                'activation': float(a),  # Convert to Python float for JSON serialization
                'context': token_ds[i][1],
                'token_index': int(i)
            })
        
        feature_analysis[f"feature_{k}"] = top_activations
    
    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(feature_analysis, f, indent=2, ensure_ascii=False)
    
    print(f"Feature analysis saved to {output_path}")