import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Optional, Tuple

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

    ### --- PRIMARY FUNCTION USED --- ###
    def extract_topk_routing_batch(
        self,
        texts: List[str],
        layer_idx: int,
        top_k: int = 8, # This shouldn't be changed
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Extract the top-K router indices and scores (post-softmax) for each token at a specific layer.

        Args:
            texts: list of input texts
            layer_idx: which layer's router outputs to extract
            top_k: number of experts to keep
            max_length: optional truncation length
            batch_size: minibatch size

        Returns:
            dict with:
                - texts: original input strings
                - tokens: per-sample token lists
                - topk_indices: per-sample [seq_len, top_k] LongTensor
                - topk_scores:  per-sample [seq_len, top_k] FloatTensor
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        if batch_size is None:
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

        # Find the target layer module
        matching_modules = [
            (name, module) for li, (name, module) in enumerate(self.model.named_modules())
            if 'mlp' in name and hasattr(module, 'gate')
        ]
        if layer_idx < 0 or layer_idx >= len(matching_modules):
            raise ValueError(f"Invalid layer_idx {layer_idx}, model has {len(matching_modules)} gated MLPs")
        target_name, target_module = matching_modules[layer_idx]

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start+batch_size]
            enc = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True if max_length else False,
                max_length=max_length
            ).to(self.device)

            tokens_batch = [
                self.tokenizer.convert_ids_to_tokens(ids) for ids in enc["input_ids"]
            ]

            sample_topk_idx, sample_topk_vals = [], []
            handle = target_module.register_forward_hook(
                lambda m, i, o,
                    store_idx=sample_topk_idx,
                    store_vals=sample_topk_vals:
                    hook_fn(m, i, o, store_idx, store_vals)
            )

            try:
                with torch.no_grad():
                    _ = self.model(**enc)

                # Each hook call yields [B, S, K]
                idx_tensor = sample_topk_idx[0]  # [B, S, K]
                val_tensor = sample_topk_vals[0]  # [B, S, K]

                # Unpack per-sample
                for i, toks in enumerate(tokens_batch):
                    valid_len = enc["attention_mask"][i].sum().item()
                    all_tokens.append(toks[:valid_len])
                    all_topk_idx.append(idx_tensor[i, :valid_len].clone())   # [S, K]
                    all_topk_vals.append(val_tensor[i, :valid_len].clone())  # [S, K]

            finally:
                handle.remove()

        return {
            "texts": texts,
            "tokens": all_tokens,
            "topk_indices": all_topk_idx,  # list of [S, K] LongTensors
            "topk_scores": all_topk_vals   # list of [S, K] FloatTensors
        }

    def extract_activations_mlp_batch(
        self,
        texts: List[str],
        layer_idx: int,
        top_k: int = 8,
        max_length: int = 128,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract activations from a specific OlmoeSparseMoeBlock (MLP layer) for a batch of texts.

        Args:
            texts: list of input strings
            layer_idx: which transformer layer's MLP block to inspect
            top_k: number of experts to keep
            max_length: optional truncation length
            batch_size: minibatch size

        Returns:
            dict with per-sample activations:
                - texts
                - tokens
                - prerouting_activations: [S, H] # Prerouting Logits X
                - router_logits: [S, E] # W(X)
                - router_probs: [S, E] # Softmax(W(X))
                - topk_indices: [S, K]
                - expert_outputs: [S, E, H]
                - combined_output: [S, H]
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        if not batch_size:
            batch_size = self.batch_size

        mlp_block = self.model.model.layers[layer_idx].mlp
        num_experts = len(mlp_block.experts)
        hidden_size = mlp_block.experts[0].down_proj.out_features

        all_tokens = []
        all_prerouting_activations = []
        all_router_logits = []
        all_router_probs = []
        all_topk_indices = []
        all_expert_outputs = []
        all_combined_output = []

        def hook_fn(module, input, output, store):
            prerouting_activations = input[0].detach().cpu()   # [B, S, H]
            B, S, H = prerouting_activations.shape

            # Router outputs
            router_logits = module.gate(prerouting_activations.to(self.device))   # [B, S, E]
            router_probs = torch.softmax(router_logits, dim=-1)

            # Top-k selection
            topk_vals, topk_idx = torch.topk(router_probs, top_k, dim=-1)

            # Expert outputs: run all experts
            expert_outputs = []
            for expert in module.experts:
                e_out = expert(prerouting_activations.to(self.device))  # [B, S, H]
                expert_outputs.append(e_out.detach().cpu())
            expert_outputs = torch.stack(expert_outputs, dim=2)  # [B, S, E, H]

            # Weighted mixture using only top-k experts
            combined_output = torch.zeros_like(prerouting_activations)
            for k in range(top_k):
                idx = topk_idx[:, :, k]  # [B, S]
                val = topk_vals[:, :, k]  # [B, S]
                for b in range(B):
                    for s in range(S):
                        expert_id = idx[b, s].item()
                        weight = val[b, s].item()
                        combined_output[b, s] += weight * expert_outputs[b, s, expert_id]

            store["prerouting_activations"].append(prerouting_activations)
            store["router_logits"].append(router_logits.detach().cpu())
            store["router_probs"].append(router_probs.detach().cpu())
            store["topk_indices"].append(topk_idx.detach().cpu())
            store["expert_outputs"].append(expert_outputs)
            store["combined_output"].append(combined_output.detach().cpu())

        # Loop over minibatches
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start+batch_size]
            enc = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)

            tokens_batch = [
                self.tokenizer.convert_ids_to_tokens(ids)
                for ids in enc["input_ids"]
            ]

            store = {
                "prerouting_activations": [],
                "router_logits": [],
                "router_probs": [],
                "topk_indices": [],
                "expert_outputs": [],
                "combined_output": []
            }

            handle = mlp_block.register_forward_hook(
                lambda m, i, o, store=store: hook_fn(m, i, o, store)
            )

            try:
                with torch.no_grad():
                    _ = self.model(**enc)
            finally:
                handle.remove()

            # Collapse lists into tensors [B, S, ...]
            prerouting_activations = torch.cat(store["prerouting_activations"], dim=0)
            router_logits = torch.cat(store["router_logits"], dim=0)
            router_probs = torch.cat(store["router_probs"], dim=0)
            topk_indices = torch.cat(store["topk_indices"], dim=0)
            expert_outputs = torch.cat(store["expert_outputs"], dim=0)
            combined_output = torch.cat(store["combined_output"], dim=0)

            # Unpack per-sample (respect attention_mask)
            for i, toks in enumerate(tokens_batch):
                valid_len = enc["attention_mask"][i].sum().item()
                all_tokens.append(toks[:valid_len])
                all_prerouting_activations.append(prerouting_activations[i, :valid_len])
                all_router_logits.append(router_logits[i, :valid_len])
                all_router_probs.append(router_probs[i, :valid_len])
                all_topk_indices.append(topk_indices[i, :valid_len])
                all_expert_outputs.append(expert_outputs[i, :valid_len])
                all_combined_output.append(combined_output[i, :valid_len])

        return {
            "texts": texts,
            "tokens": all_tokens,
            "prerouting_activations": all_prerouting_activations,          # list of [S, H]
            "router_logits": all_router_logits,  # list of [S, E]
            "router_probs": all_router_probs,    # list of [S, E]
            "topk_indices": all_topk_indices,    # list of [S, K]
            "expert_outputs": all_expert_outputs,# list of [S, E, H]
            "combined_output": all_combined_output # list of [S, H]
        }

    def extract_prerouting_and_combined_output_batch(
        self,
        texts: List[str],
        layer_idx: int,
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Extract the input and output activations of the MLP (MoE) block
        at a given transformer layer.

        Args:
            texts: list of input texts
            layer_idx: which layer's MLP activations to extract
            max_length: optional truncation length
            batch_size: minibatch size

        Returns:
            dict with:
                - texts: original input strings
                - tokens: per-sample token lists
                - prerouting_activations:  per-sample [seq_len, hidden_dim] FloatTensor
                - combined_outputs: per-sample [seq_len, hidden_dim] FloatTensor
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        if batch_size is None:
            batch_size = self.batch_size

        all_tokens, all_mlp_in, all_mlp_out = [], [], []

        # Hook to capture MLP input and output
        def hook_fn(module, input, output, store_in, store_out):
            hidden_in = input[0]        # [B, S, H] input to MLP
            hidden_out = output[0] if isinstance(output, tuple) else output  # [B, S, H]
            store_in.append(hidden_in.detach().cpu())
            store_out.append(hidden_out.detach().cpu())

        # Locate the target layer's mlp
        matching_modules = [(li, module) for li, module in enumerate(self.model.model.layers)]

        if layer_idx < 0 or layer_idx >= len(matching_modules):
            raise ValueError(f"Invalid layer_idx {layer_idx}, model has {len(matching_modules)} layers")
        target_module = matching_modules[layer_idx][1].mlp

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start+batch_size]
            enc = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True if max_length else False,
                max_length=max_length
            ).to(self.device)

            tokens_batch = [
                self.tokenizer.convert_ids_to_tokens(ids) for ids in enc["input_ids"]
            ]

            sample_mlp_in, sample_mlp_out = [], []
            handle = target_module.register_forward_hook(
                lambda m, i, o,
                    store_in=sample_mlp_in,
                    store_out=sample_mlp_out:
                    hook_fn(m, i, o, store_in, store_out)
            )

            try:
                with torch.no_grad():
                    _ = self.model(**enc)

                # Each hook call yields [B, S, H]
                in_tensor = sample_mlp_in[0]
                out_tensor = sample_mlp_out[0]

                for i, toks in enumerate(tokens_batch):
                    valid_len = enc["attention_mask"][i].sum().item()
                    all_tokens.append(toks[:valid_len])
                    all_mlp_in.append(in_tensor[i, :valid_len].clone())
                    all_mlp_out.append(out_tensor[i, :valid_len].clone())

            finally:
                handle.remove()

        return {
            "texts": texts,
            "tokens": all_tokens,
            "prerouting_activations": all_mlp_in,   # list of [S, H] FloatTensors
            "combined_output": all_mlp_out  # list of [S, H] FloatTensors
        }


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
        print(f"Is model on CUDA? {is_cuda}")

    def verify_saved_activations(self, data: dict[str, Any] | str):
        """
        Load and verify a saved activation file.
        
        Args:
            data: (str) Path to the saved activation file or (dict) loaded data dictionary
        """
        if isinstance(data, dict):
            loaded_data = data
        else:  # It must be a string based on the type hint
            loaded_data = self.load_activations(data)
        
        if loaded_data is None:
            print("❌ Failed to load file")
            return False
        print(f"===== Verifying Activations =====")
        num_samples = len(loaded_data["texts"])
        print(f"Num Samples: {num_samples}")
        for key in loaded_data.keys():
            if type(loaded_data[key]) == list:
                if type(loaded_data[key][-1]) == str or type(loaded_data[key][-1]) == list:
                    print(f"{key} (List): {loaded_data[key][-1]}")
                elif type(loaded_data[key][-1]) == torch.Tensor:
                    print(f"{key} (Tensor) shape: {loaded_data[key][-1].shape}")
                else:
                    print(f"{key} is present")
            elif type(loaded_data[key]) == str:
                print(f"{key} (str): {loaded_data[key]}")
            else:
                data_type = type(loaded_data[key])
                print(f"{key} ({data_type}) is present")
        return True