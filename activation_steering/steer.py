import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, Callable


class MoEActivationSteering:
    """
    Modular activation patching & steering for OLMoE models.

    Supports:
    - Patching prerouter activations or routing decision vectors at a single (layer, token).
    - Steering prerouter activations or routing vectors for all downstream tokens at a given layer.
    """

    def __init__(self, model, extractor):
        """
        Args:
            model: The OLMoE model.
            extractor: An OLMoEActivationExtractor instance that exposes hooks.
        """
        self.model = model
        self.extractor = extractor
        self.handles = []  # active hooks

        # Store interventions
        self.patch_config: Dict[str, Any] = {}
        self.steer_config: Dict[str, Any] = {}

    # -----------------------------
    # PATCHING
    # -----------------------------
    def patch_prerouter(
        self, layer_idx: int, token_idx: int, new_vector: torch.Tensor
    ):
        """Replace prerouter activation at (layer, token)."""
        self.patch_config = {
            "type": "prerouter",
            "layer": layer_idx,
            "token": token_idx,
            "vector": new_vector,
        }
        self._register_hook(layer_idx, "prerouter")

    def patch_router(
        self, layer_idx: int, token_idx: int, new_vector: torch.Tensor
    ):
        """Replace routing logits at (layer, token)."""
        self.patch_config = {
            "type": "router",
            "layer": layer_idx,
            "token": token_idx,
            "vector": new_vector,
        }
        self._register_hook(layer_idx, "router")

    # -----------------------------
    # STEERING
    # -----------------------------
    def steer_prerouter(
        self, layer_idx: int, delta: torch.Tensor
    ):
        """Add vector delta to prerouter activations for all downstream tokens."""
        self.steer_config = {
            "type": "prerouter",
            "layer": layer_idx,
            "delta": delta,
        }
        self._register_hook(layer_idx, "prerouter", steer=True)

    def steer_router(
        self, layer_idx: int, delta: torch.Tensor
    ):
        """Add vector delta to routing logits for all downstream tokens."""
        self.steer_config = {
            "type": "router",
            "layer": layer_idx,
            "delta": delta,
        }
        self._register_hook(layer_idx, "router", steer=True)

    # -----------------------------
    # INTERNAL HOOKING
    # -----------------------------
    def _register_hook(self, layer_idx: int, kind: str, steer: bool = False):
        """
        Register a forward hook for interventions.
        """
        def hook_fn(module, input, output):
            # output = (batch, seq_len, dim)
            # We intervene *in place* on output
            if not torch.is_tensor(output):
                return output

            if not steer:  # PATCH once at (layer, token)
                token = self.patch_config["token"]
                vector = self.patch_config["vector"].to(output.device)
                if kind == self.patch_config["type"] and layer_idx == self.patch_config["layer"]:
                    # Replace token vector
                    output[:, token, :] = vector
            else:  # STEER continuously at layer
                delta = self.steer_config["delta"].to(output.device)
                if kind == self.steer_config["type"] and layer_idx == self.steer_config["layer"]:
                    output = output + delta  # broadcast to all tokens
            return output

        # Use extractor’s knowledge of module structure
        target_module = self.extractor.get_module(layer_idx, kind)
        handle = target_module.register_forward_hook(hook_fn)
        self.handles.append(handle)

    # -----------------------------
    # UTILS
    # -----------------------------
    def clear(self):
        """Remove all active hooks & reset configs."""
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.patch_config = {}
        self.steer_config = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.clear()
