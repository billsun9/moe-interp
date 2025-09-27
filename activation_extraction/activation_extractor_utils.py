import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Optional, Tuple
import os

from moe.path_config import MODEL_CACHE_DIR, SAVE_ACTS_PATH_DIR

class OLMoEActivationExtractor:
    """
    A modularized class for extracting and saving OLMoE model activations.
    """

    def __init__(self, 
                 model_name: str = "allenai/OLMoE-1B-7B-0125-Instruct",
                 cache_dir: str = MODEL_CACHE_DIR,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 4):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.model = None
        self.tokenizer = None
        self.hooks = []
        self.batch_size = batch_size

    def load_model(self) -> bool:
        if not self.model:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)
        print("✅ Model loaded successfully")
        return True

    def load_tokenizer(self) -> bool:
        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
        print("✅ Tokenizer loaded successfully")
        return True

    def load_model_and_tokenizer(self) -> bool:
        self.load_model()
        self.load_tokenizer()
        print(f"Using device {self.model.device}")
        return True

    def _create_activation_hook(self, 
                                prerouting_logits: List[torch.Tensor], 
                                routing_logits: List[torch.Tensor]):
        def hook_fn(module, input, output):
            if hasattr(module, 'gate'):
                hidden_states = input[0]  # [batch, seq, hidden]
                prerouting_logits.append(hidden_states.detach().cpu())
                router_logits = module.gate(hidden_states)
                routing_logits.append(router_logits.detach().cpu())
        return hook_fn

    def _register_hooks(self, prerouting_logits: List[torch.Tensor], routing_logits: List[torch.Tensor]) -> int:
        hook_fn = self._create_activation_hook(prerouting_logits, routing_logits)
        hook_count = 0
        for name, module in self.model.named_modules():
            if 'mlp' in name and hasattr(module, 'gate'):
                handle = module.register_forward_hook(hook_fn)
                self.hooks.append(handle)
                hook_count += 1
        return hook_count

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def extract_activations_batch(self, 
                                texts: List[str], 
                                max_length: Optional[int] = None,
                                batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Batched extraction of activations, filtering out pad tokens.

        Args:
            texts: list of input texts
            max_length: optional truncation length
            batch_size: minibatch size

        Returns:
            dict with per-sample tokens, prerouting_logits, routing_logits
            - tokens: list[list[str]] (no pad tokens)
            - prerouting_logits: list[Tensor[L, S, H]] (no pad tokens)
            - routing_logits: list[Tensor[L, S, E]] (no pad tokens)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        if not batch_size:
            batch_size = self.batch_size

        all_tokens, all_prerouting, all_routing = [], [], []

        # Iterate over minibatches
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start+batch_size]
            enc = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True if max_length else False,
                max_length=max_length
            ).to(self.device)

            tokens_batch = [self.tokenizer.convert_ids_to_tokens(ids) for ids in enc["input_ids"]]
            attn_mask = enc["attention_mask"]  # [B, S]

            sample_prerouting_logits, sample_routing_logits = [], []
            self._register_hooks(sample_prerouting_logits, sample_routing_logits)

            try:
                with torch.no_grad():
                    _ = self.model(**enc)

                # stack -> [num_layers, batch, seq, hidden]
                prerouting_tensor = torch.stack(sample_prerouting_logits)  # [L, B, S, H]
                routing_tensor = torch.stack(sample_routing_logits)        # [L, B, S, E]

                # Unpack per-sample (filter pad tokens)
                for i, toks in enumerate(tokens_batch):
                    valid_len = attn_mask[i].sum().item()
                    valid_tokens = toks[:valid_len]
                    all_tokens.append(valid_tokens)

                    all_prerouting.append(prerouting_tensor[:, i, :valid_len].clone())  # [L, valid_len, H]
                    all_routing.append(routing_tensor[:, i, :valid_len].clone())        # [L, valid_len, E]

            finally:
                self._remove_hooks()

        return {
            "texts": texts,
            "tokens": all_tokens,
            "prerouting_logits": all_prerouting,
            "routing_logits": all_routing
        }


    def extract_activations_single(self, text: str) -> Dict[str, Any]:
        batch_data = self.extract_activations_batch([text])
        return {
            "text": text,
            "tokens": batch_data["tokens"][0],
            "prerouting_logits": batch_data["prerouting_logits"][0],
            "routing_logits": batch_data["routing_logits"][0]
        }

    def extract_activations_batch_generation(self,
                                             texts: List[str],
                                             max_new_tokens: int = 20,
                                             max_length: Optional[int] = None,
                                             batch_size: int = 8,
                                             **gen_kwargs) -> Dict[str, Any]:
        """
        Extract activations while generating new tokens.
        Keeps track of where generation starts per-sample.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")

        all_tokens, all_prerouting, all_routing, gen_start_idxs = [], [], [], []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start+batch_size]
            enc = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True if max_length else False,
                max_length=max_length
            ).to(self.device)

            input_ids = enc["input_ids"]
            gen_start_lengths = (input_ids != self.tokenizer.pad_token_id).sum(dim=1).tolist()

            # Run generation
            with torch.no_grad():
                gen_out = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_hidden_states=False,
                    **gen_kwargs
                )

            # Decode generated input_ids
            gen_input_ids = gen_out.sequences  # [batch, prompt+gen]
            tokens_batch = [self.tokenizer.convert_ids_to_tokens(ids) for ids in gen_input_ids]

            # Rerun forward pass on the *full generated sequence* to extract activations
            full_enc = {"input_ids": gen_input_ids}
            full_enc = {k: v.to(self.device) for k, v in full_enc.items()}

            sample_prerouting_logits, sample_routing_logits = [], []
            self._register_hooks(sample_prerouting_logits, sample_routing_logits)

            try:
                with torch.no_grad():
                    _ = self.model(**full_enc)

                prerouting_tensor = torch.stack(sample_prerouting_logits)  # [L, B, S, H]
                routing_tensor = torch.stack(sample_routing_logits)        # [L, B, S, E]

                for i, toks in enumerate(tokens_batch):
                    all_tokens.append(toks)
                    all_prerouting.append(prerouting_tensor[:, i].clone())
                    all_routing.append(routing_tensor[:, i].clone())
                    gen_start_idxs.append(gen_start_lengths[i])

            finally:
                self._remove_hooks()

        return {
            "texts": texts,
            "tokens": all_tokens,
            "prerouting_logits": all_prerouting,
            "routing_logits": all_routing,
            "gen_start": gen_start_idxs
        }

    ### --- PRIMARY FUNCTION USED --- ###
    def extract_topk_routing_batch(self, 
                                texts: List[str], 
                                top_k: int = 8,
                                max_length: Optional[int] = None,
                                batch_size: Optional[int] = None,
                                layer_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract the top-K router indices and scores (post-softmax) for each token/layer.

        Args:
            texts: list of input texts
            top_k: number of experts to keep
            max_length: optional truncation length
            batch_size: minibatch size
            layer_idx: If None ==> Gets all layers, otherwise only saves a specific layer [WIP]

        Returns:
            dict with:
                - texts: original input strings
                - tokens: per-sample token lists
                - topk_indices: per-sample [num_layers, seq_len, top_k] LongTensor
                - topk_scores:  per-sample [num_layers, seq_len, top_k] FloatTensor
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        if not batch_size:
            batch_size = self.batch_size

        all_tokens, all_topk_idx, all_topk_vals = [], [], []

        def hook_fn(module, input, output, store_idx, store_vals):
            if hasattr(module, "gate"):
                hidden_states = input[0]  # [B, S, H]
                router_logits = module.gate(hidden_states)  # [B, S, E]
                router_probs = F.softmax(router_logits, dim=-1)
                vals, idx = torch.topk(router_probs, k=top_k, dim=-1)  # [B, S, K]
                store_idx.append(idx.detach().cpu())
                store_vals.append(vals.detach().cpu())

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start+batch_size]
            enc = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True if max_length else False,
                max_length=max_length
            ).to(self.device)

            tokens_batch = [self.tokenizer.convert_ids_to_tokens(ids) for ids in enc["input_ids"]]

            sample_topk_idx, sample_topk_vals = [], []
            handles = []

            # Register hooks
            for name, module in self.model.named_modules():
                if 'mlp' in name and hasattr(module, 'gate'):
                    handles.append(module.register_forward_hook(
                        lambda m, i, o,
                            store_idx=sample_topk_idx,
                            store_vals=sample_topk_vals:
                            hook_fn(m, i, o, store_idx, store_vals)
                    ))

            try:
                with torch.no_grad():
                    _ = self.model(**enc)

                # Stack → [num_layers, B, S, K]
                idx_tensor = torch.stack(sample_topk_idx)
                val_tensor = torch.stack(sample_topk_vals)

                # Unpack per-sample
                for i, toks in enumerate(tokens_batch):
                    valid_len = enc["attention_mask"][i].sum().item()  # number of real tokens
                    all_tokens.append(toks[:valid_len])
                    all_topk_idx.append(idx_tensor[:, i, :valid_len].clone())   # [L, valid_len, K]
                    all_topk_vals.append(val_tensor[:, i, :valid_len].clone())  # [L, valid_len, K]

            finally:
                for h in handles:
                    h.remove()

        return {
            "texts": texts,
            "tokens": all_tokens,
            "topk_indices": all_topk_idx,  # list of [L, S, K] LongTensors
            "topk_scores": all_topk_vals   # list of [L, S, K] FloatTensors
        }


    def extract_activations_mlp(self, text: str, layer_idx: int, max_length: int = 128):
        """
        Extract all relevant activations from a specific OlmoeSparseMoeBlock (MLP layer).
        
        Args:
            text: input string
            layer_idx: which transformer layer's MLP block to inspect
            max_length: optional truncation length

        Returns:
            dict of activations:
                - tokens
                - hidden_in: [seq, hidden]
                - router_logits: [seq, num_experts]
                - router_probs: [seq, num_experts]
                - topk_indices: [seq, top_k]
                - expert_outputs: [seq, num_experts, hidden]
                - combined_output: [seq, hidden]
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

        mlp_block = self.model.model.layers[layer_idx].mlp
        num_experts = len(mlp_block.experts)
        hidden_size = mlp_block.experts[0].down_proj.out_features
        top_k = 8

        activations = {
            "tokens": tokens,
            "hidden_in": [],
            "router_logits": [],
            "router_probs": [],
            "topk_indices": [],
            "expert_outputs": [],
            "combined_output": []
        }

        def hook_fn(module, input, output):
            # input[0] = hidden states [batch, seq, hidden]
            hidden_in = input[0].detach().cpu()  # [B, S, H]
            B, S, H = hidden_in.shape

            # Router: logits + probs
            router_logits = module.gate(hidden_in.to(self.device))   # [B, S, E]
            router_probs = torch.softmax(router_logits, dim=-1)

            # Top-k selection
            topk_vals, topk_idx = torch.topk(router_probs, top_k, dim=-1)

            # Expert outputs: run *all* experts independently
            expert_outputs = []
            for expert in module.experts:
                e_out = expert(hidden_in.to(self.device))  # [B, S, H]
                expert_outputs.append(e_out.detach().cpu())
            expert_outputs = torch.stack(expert_outputs, dim=2)  # [B, S, E, H]

            # Weighted mixture using only top-k experts
            combined_output = torch.zeros_like(hidden_in)
            for k in range(top_k):
                idx = topk_idx[:, :, k]  # [B, S]
                val = topk_vals[:, :, k]  # [B, S]
                for b in range(B):
                    for s in range(S):
                        expert_id = idx[b, s].item()
                        weight = val[b, s].item()
                        combined_output[b, s] += weight * expert_outputs[b, s, expert_id]

            # Save results (drop batch dimension, assume B=1 for simplicity)
            activations["hidden_in"].append(hidden_in[0])
            activations["router_logits"].append(router_logits[0].detach().cpu())
            activations["router_probs"].append(router_probs[0].detach().cpu())
            activations["topk_indices"].append(topk_idx[0].detach().cpu())
            activations["expert_outputs"].append(expert_outputs[0])     # [S, E, H]
            activations["combined_output"].append(combined_output[0].detach().cpu())

        handle = mlp_block.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                _ = self.model(**enc)
        finally:
            handle.remove()

        # collapse lists into tensors
        for k in activations:
            if k != "tokens":
                activations[k] = torch.stack(activations[k])

        return activations

    def save_activations(self, 
                        data: Dict[str, Any], 
                        save_path: str,
                        create_dirs: bool = True) -> bool:
        """
        Save activation data to file.
        
        Args:
            data: Data dictionary to save
            save_path: Path to save the file
            create_dirs: Whether to create directories if they don't exist
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if create_dirs:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            torch.save(data, save_path)
            print(f"✅ Saved activations to {save_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to save activations: {e}")
            return False
    
    def load_activations(self, load_path: str) -> Optional[Dict[str, Any]]:
        """
        Load activation data from file.
        
        Args:
            load_path: Path to load the file from
            
        Returns:
            Dict containing loaded data, or None if failed
        """
        try:
            if not os.path.exists(load_path):
                print(f"❌ File not found: {load_path}")
                return None
            
            data = torch.load(load_path)
            print(f"✅ Loaded activations from {load_path}")
            return data
        except Exception as e:
            print(f"❌ Failed to load activations: {e}")
            return None
    
    def print_model_summary(self):
        print(self.model)
        print(f"total_params = {sum(p.numel() for p in self.model.parameters())}")

        is_cuda = next(self.model.parameters()).is_cuda
        print(f"Is model on CUDA (GPU)? {is_cuda}")

    def print_activation_summary(self, data: Dict[str, Any]):
        """
        Print a summary of the activation data.
        
        Args:
            data: Activation data dictionary
        """
        if "texts" in data:  # Batch format
            print(f"Batch data with {len(data['texts'])} samples:")
            for i, text in enumerate(data["texts"]):
                tokens = data["tokens"][i]
                prerouting_logits = data["prerouting_logits"][i]
                routing_logits = data["routing_logits"][i]
                
                print(f"  Sample {i+1}:")
                print(f"    Text: {text}")
                print(f"    Tokens: {tokens}")
                print(f"    Layers: {len(prerouting_logits)}")
                print(f"    Prerouting Logits Shape: {prerouting_logits.shape}")
                print(f"    Routing Logits Shape: {routing_logits.shape}")
        else:  # Single format
            print("Single sample data:")
            print(f"  Text: {data['text']}")
            print(f"  Tokens: {data['tokens']}")
            print(f"  Layers: {len(data['prerouting_logits'])}")
            print(f"    Prerouting Logits Shape: {data['prerouting_logits'].shape}")
            print(f"    Routing Logits Shape: {data['routing_logits'].shape}")


def test_single_activation(
    model_name: str = "allenai/OLMoE-1B-7B-0125-Instruct",
    cache_dir: str = MODEL_CACHE_DIR,
    save_path: str = os.path.join(SAVE_ACTS_PATH_DIR, "test_acts.pt")):
    """
    Test function that replicates the original functionality.
    
    Args:
        save_path: Path to save the test results
    """
    print("\n" + "=" * 60)
    print("Testing single activation extraction...")
    
    try:
        # Initialize extractor
        extractor = OLMoEActivationExtractor(model_name, cache_dir)
        
        # Load model
        if not extractor.load_model_and_tokenizer():
            return False
        
        # Extract activations
        test_text = "The quick brown fox"
        data = extractor.extract_activations_single(test_text)
        
        # Save activations
        if not extractor.save_activations(data, save_path):
            return False
        
        # Load and verify
        loaded_data = extractor.load_activations(save_path)
        if loaded_data is None:
            return False
        
        # Print summary
        extractor.print_activation_summary(loaded_data)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_batch_activation(
    model_name: str = "allenai/OLMoE-1B-7B-0125-Instruct",
    cache_dir: str = MODEL_CACHE_DIR,
    save_path: str = os.path.join(SAVE_ACTS_PATH_DIR, "test_acts.pt")):
    """
    Test function for batch processing.
    
    Args:
        save_path: Path to save the test results
    """
    print("\n" + "=" * 60)
    print("Testing batch activation extraction...")
    
    try:
        # Initialize extractor
        extractor = OLMoEActivationExtractor(model_name, cache_dir)
        
        # Load model
        if not extractor.load_model_and_tokenizer():
            return False
        
        # Extract activations for multiple texts
        test_texts = [
            "The quick brown fox",
            "Dog cat monkey ape baboon ostrich",
            "Machine learning is fascinating",
            "PyTorch makes deep learning accessible"
        ]
        
        batch_data = extractor.extract_activations_batch(test_texts)
        
        # Save activations
        if not extractor.save_activations(batch_data, save_path):
            return False
        
        # Load and verify
        loaded_data = extractor.load_activations(save_path)
        if loaded_data is None:
            return False
        
        # Print summary
        extractor.print_activation_summary(loaded_data)
        
        return True
        
    except Exception as e:
        print(f"❌ Batch test failed: {e}")
        return False


def test_mlp_extraction(text="Hello world, I like cats", layer_idx=10):
    # Use our extractor
    extractor = OLMoEActivationExtractor()
    extractor.load_model_and_tokenizer()
    model = extractor.model
    device = extractor.device
    acts = extractor.extract_activations_mlp(text, layer_idx)

    # Our reconstruction
    combined = acts["combined_output"]  # [seq, hidden]
    combined = combined.to(device)
    print(f"Our reconstruction: {combined}")
    print(f"Our reconstruction shape: {combined.shape}")
    # Direct forward pass through block
    hidden_in = acts["hidden_in"].to(device)

    # Ensure it is [B, S, H]
    if hidden_in.ndim == 2:
        hidden_in = hidden_in.unsqueeze(0)

    mlp_block = model.model.layers[layer_idx].mlp
    print(mlp_block)
    with torch.no_grad():
        direct_out, *_ = mlp_block(hidden_in)   # unpack tuple, keep output
    print(f"Directly using MLP block: {direct_out}")
    print(f"Directly using MLP block shape: {direct_out.shape}")
    direct_out = direct_out.squeeze(0)  # drop batch if 1
    # Compare
    diff = torch.norm(direct_out - combined) / torch.norm(direct_out)
    print(f"Relative error: {diff.item():.6e}")
    return diff.item()

def test_extract_topk_routing_batch(
    model_name: str = "allenai/OLMoE-1B-7B-0125-Instruct",
    cache_dir: str = MODEL_CACHE_DIR
):
    extractor = OLMoEActivationExtractor(model_name = model_name, cache_dir = cache_dir)
    if not extractor.load_model_and_tokenizer():
        return False

    test_texts = [
        "The quick brown fox",
        "Dog cat monkey ape baboon ostrich",
        "Machine learning is fascinating",
        "PyTorch makes deep learning accessible",
        "PyTorch makes deep learning accessible PyTorch makes deep learning accessible PyTorch makes deep learning accessible PyTorch makes deep learning accessible PyTorch makes deep learning accessible"
    ]

    acts = extractor.extract_topk_routing_batch(test_texts, batch_size=10)
    print(acts.keys())
    for key in acts.keys():
        print("====================" + key + "====================")
        if type(acts[key]) == list: print(len(acts[key]))
        print(acts[key][0])
        if type(acts[key][0]) == torch.Tensor:
            print(acts[key][0].shape)
    return True