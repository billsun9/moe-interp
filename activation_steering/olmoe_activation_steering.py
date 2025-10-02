import torch
from typing import Optional, Union, Sequence, Dict, Tuple, Any


class ActivationPatcher:
    """
    Patch / steer activations for an OLMoE-like model's MLP (MoE) block.

    Usage pattern:
        patcher = ActivationPatcher(model)
        # Add vector to all tokens' prerouting activations of layer 10
        patcher.register_prerouting_patch(layer_idx=10, vector=my_vec, mode="add")
        # run model.generate(...) or model(...)
        patcher.remove_patch(layer_idx=10, target="prerouting")

    Notes:
    - This monkeypatches the mlp block's forward. Call `remove_patch` (or `remove_all`) to restore.
    - Designed for inference / evaluation (wrap usage in `torch.no_grad()` and `model.eval()`).
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        # key: (layer_idx, target) target in {"prerouting", "output"}
        self._orig_forwards: Dict[Tuple[int, str], Any] = {}
        self._active_patches: Dict[Tuple[int, str], Dict] = {}

    def _get_mlp_block(self, layer_idx: int):
        # Adapt this path if your model nests layers differently.
        return self.model.model.layers[layer_idx].mlp

    def _tensor_from_vector(self, vector: Union[torch.Tensor, Sequence, float, int],
                            device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not isinstance(vector, torch.Tensor):
            vector = torch.tensor(vector, dtype=dtype)
        return vector.to(device=device, dtype=dtype)

    def _make_mask(self, token_positions: Optional[Union[int, Sequence[int], torch.Tensor]],
                   B: int, S: int, device: torch.device) -> torch.Tensor:
        """Return boolean mask shape [B, S] where True indicates positions to modify."""
        if token_positions is None:
            return torch.ones((B, S), dtype=torch.bool, device=device)
        if isinstance(token_positions, torch.Tensor):
            mask = token_positions.to(device=device)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(B, -1)
            return mask.bool()
        if isinstance(token_positions, int):
            mask = torch.zeros((B, S), dtype=torch.bool, device=device)
            if 0 <= token_positions < S:
                mask[:, token_positions] = True
            return mask
        # sequence of ints
        mask = torch.zeros((B, S), dtype=torch.bool, device=device)
        for p in token_positions:
            if 0 <= int(p) < S:
                mask[:, int(p)] = True
        return mask

    def _apply_add(self, tensor: torch.Tensor, vec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # tensor: [B, S, H], vec: [H] or [1,1,H] or [B,S,H], mask: [B,S]
        B, S, H = tensor.shape
        if vec.dim() == 1:
            vec = vec.view(1, 1, H)  # [1,1,H]
        if vec.dim() == 2:  # maybe [1,H]
            vec = vec.view(1, 1, H)
        if vec.shape[:2] != (B, S):
            vec = vec.expand(B, S, H)
        mask_f = mask.unsqueeze(-1).to(dtype=vec.dtype)
        return tensor + vec * mask_f

    def _apply_replace(self, tensor: torch.Tensor, vec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # replace masked positions with vec (vec can be [H] or [B,S,H])
        B, S, H = tensor.shape
        out = tensor.clone()
        if vec.dim() == 1:
            # simple case: same replacement vector for all masked tokens
            indices = mask.nonzero(as_tuple=False)  # [N, 2] rows [b, s]
            for (b, s) in indices:
                out[b, s] = vec
            return out
        else:
            # expand and assign
            if vec.shape[:2] != (B, S):
                vec = vec.expand(B, S, H)
            mask_idx = mask.unsqueeze(-1)
            out[mask_idx.expand(-1, -1, H)] = vec[mask_idx.expand(-1, -1, H)]
            return out

    def register_prerouting_patch(self,
                                  layer_idx: int,
                                  vector: Union[torch.Tensor, Sequence, float, int],
                                  mode: str = "add",
                                  token_positions: Optional[Union[int, Sequence[int], torch.Tensor]] = None):
        """
        Patch the *input* to the mlp block (prerouting hidden states).
        - mode: "add" or "replace"
        - token_positions: None => all tokens, int or list of ints or boolean mask
        """
        target = "prerouting"
        key = (layer_idx, target)
        if key in self._active_patches:
            raise RuntimeError(f"Patch already active for layer {layer_idx} target {target}")

        mlp = self._get_mlp_block(layer_idx)

        # save original forward
        orig_forward = getattr(mlp, "forward")
        self._orig_forwards[key] = orig_forward

        def make_new_forward(orig_forward, vector, mode, token_positions):
            def new_forward(*args, **kwargs):
                # identify hidden tensor in args or kwargs
                if len(args) >= 1 and isinstance(args[0], torch.Tensor):
                    hidden = args[0]
                    rest_args = args[1:]
                    use_args = True
                elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
                    hidden = kwargs["hidden_states"]
                    use_args = False
                else:
                    # fallback: call original
                    return orig_forward(*args, **kwargs)

                # normalize dims to [B,S,H]
                was_2d = False
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(0)
                    was_2d = True
                B, S, H = hidden.shape
                device = hidden.device
                dtype = hidden.dtype

                vec_t = self._tensor_from_vector(vector, device=device, dtype=dtype)
                mask = self._make_mask(token_positions, B, S, device)

                if mode == "add":
                    new_hidden = self._apply_add(hidden, vec_t, mask)
                elif mode == "replace":
                    new_hidden = self._apply_replace(hidden, vec_t, mask)
                else:
                    raise ValueError(f"Unknown mode {mode}")

                if was_2d:
                    new_hidden = new_hidden.squeeze(0)

                # rebuild args/kwargs
                if use_args:
                    new_args = (new_hidden,) + rest_args
                    return orig_forward(*new_args, **kwargs)
                else:
                    kwargs["hidden_states"] = new_hidden
                    return orig_forward(*args, **kwargs)

            return new_forward

        new_forward = make_new_forward(orig_forward, vector, mode, token_positions)
        # attach
        mlp.forward = new_forward
        # store metadata
        self._active_patches[key] = {"vector": vector, "mode": mode, "token_positions": token_positions}
        return True

    def register_output_patch(self,
                              layer_idx: int,
                              vector: Union[torch.Tensor, Sequence, float, int],
                              mode: str = "add",
                              token_positions: Optional[Union[int, Sequence[int], torch.Tensor]] = None):
        """
        Patch the MLP block *output* (combined MoE output). The patch will modify the main returned tensor
        from mlp.forward (if it returns a tuple, the first element is modified).
        """
        target = "output"
        key = (layer_idx, target)
        if key in self._active_patches:
            raise RuntimeError(f"Patch already active for layer {layer_idx} target {target}")

        mlp = self._get_mlp_block(layer_idx)
        orig_forward = getattr(mlp, "forward")
        self._orig_forwards[key] = orig_forward

        def make_new_forward(orig_forward, vector, mode, token_positions):
            def new_forward(*args, **kwargs):
                out = orig_forward(*args, **kwargs)
                # canonicalize out and its first tensor
                if isinstance(out, tuple):
                    out0 = out[0]
                    rest = out[1:]
                    is_tuple = True
                else:
                    out0 = out
                    rest = ()
                    is_tuple = False

                if not isinstance(out0, torch.Tensor):
                    # can't modify
                    return out

                was_2d = False
                if out0.dim() == 2:
                    out0 = out0.unsqueeze(0)
                    was_2d = True
                B, S, H = out0.shape
                device = out0.device
                dtype = out0.dtype

                vec_t = self._tensor_from_vector(vector, device=device, dtype=dtype)
                mask = self._make_mask(token_positions, B, S, device)

                if mode == "add":
                    new_out0 = self._apply_add(out0, vec_t, mask)
                elif mode == "replace":
                    new_out0 = self._apply_replace(out0, vec_t, mask)
                else:
                    raise ValueError(f"Unknown mode {mode}")

                if was_2d:
                    new_out0 = new_out0.squeeze(0)

                if is_tuple:
                    return (new_out0, ) + rest
                else:
                    return new_out0

            return new_forward

        mlp.forward = make_new_forward(orig_forward, vector, mode, token_positions)
        self._active_patches[key] = {"vector": vector, "mode": mode, "token_positions": token_positions}
        return True

    def remove_patch(self, layer_idx: int, target: str = "prerouting"):
        key = (layer_idx, target)
        if key not in self._orig_forwards:
            return False
        mlp = self._get_mlp_block(layer_idx)
        mlp.forward = self._orig_forwards[key]
        del self._orig_forwards[key]
        if key in self._active_patches:
            del self._active_patches[key]
        return True

    def remove_all(self):
        for key in list(self._orig_forwards.keys()):
            layer_idx, target = key
            try:
                mlp = self._get_mlp_block(layer_idx)
                mlp.forward = self._orig_forwards[key]
            except Exception:
                pass
            del self._orig_forwards[key]
        self._active_patches.clear()
        return True

    def list_active(self):
        return dict(self._active_patches)
