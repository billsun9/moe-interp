import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Optional, Tuple
import os

class OLMoEActivationExtractor:
    """
    A modularized class for extracting and saving OLMoE model activations.
    """

    def __init__(self, 
                 model_name: str = "allenai/OLMoE-1B-7B-0125-Instruct",
                 cache_dir: str = "/local/bys2107/hf_cache",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 4):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.model = None
        self.tokenizer = None
        self.hooks = []
        self.batch_size = batch_size

    def load_model_and_tokenizer(self) -> bool:
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            print("✅ Model and tokenizer loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load model/tokenizer: {e}")
            return False

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
        True batched extraction of activations.

        Args:
            texts: list of input texts
            max_length: optional truncation length
            batch_size: minibatch size

        Returns:
            dict with per-sample tokens, prerouting_logits, routing_logits
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

            sample_prerouting_logits, sample_routing_logits = [], []
            self._register_hooks(sample_prerouting_logits, sample_routing_logits)

            try:
                with torch.no_grad():
                    _ = self.model(**enc)

                # stack -> [num_layers, batch, seq, hidden]
                prerouting_tensor = torch.stack(sample_prerouting_logits)  # [L, B, S, H]
                routing_tensor = torch.stack(sample_routing_logits)        # [L, B, S, E]

                # Unpack per-sample
                for i, toks in enumerate(tokens_batch):
                    all_tokens.append(toks)
                    all_prerouting.append(prerouting_tensor[:, i].clone())
                    all_routing.append(routing_tensor[:, i].clone())

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
    cache_dir: str = "/local/bys2107/hf_cache",
    save_path: str = "/local/bys2107/research/data/OLMoE-acts/test_acts.pt"):
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
    cache_dir: str = "/local/bys2107/hf_cache",
    save_path: str = "/local/bys2107/research/data/OLMoE-acts/test_acts.pt"):
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


