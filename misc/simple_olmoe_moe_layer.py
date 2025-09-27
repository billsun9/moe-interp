# The official repo doesn't have exact implementation, so had to manually construct

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwigLUMLP(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        # x: [num_tokens, hidden_dim]
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class OlmoeSparseMoeBlock(nn.Module):
    def __init__(self, hidden_dim=2048, intermediate_dim=1024, num_experts=64, top_k=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Gating: maps input to expert logits
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # Experts: list of SwigLU MLPs
        self.experts = nn.ModuleList([
            SwigLUMLP(hidden_dim, intermediate_dim)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # [B*T, D]
        num_tokens = x_flat.size(0)

        # Compute gating scores
        gate_logits = self.gate(x_flat)  # [num_tokens, num_experts]

        # Select top-k experts per token
        topk_scores, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)  # [num_tokens, top_k]
        topk_gates = F.softmax(topk_scores, dim=-1)  # Normalize the top-k scores

        # Prepare output tensor
        output = torch.zeros_like(x_flat)  # [num_tokens, hidden_dim]

        # Dispatch each token to its top-k experts
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]  # [num_tokens]
            expert_weight = topk_gates[:, i]  # [num_tokens]

            # Get mask of tokens assigned to expert e
            for e in range(self.num_experts):
                token_mask = (expert_idx == e)  # [num_tokens]
                if token_mask.any():
                    tokens = x_flat[token_mask]  # [num_tokens_e, D]
                    expert_out = self.experts[e](tokens)  # [num_tokens_e, D]
                    expert_out *= expert_weight[token_mask].unsqueeze(1)  # Weighted output
                    output[token_mask] += expert_out  # Accumulate contributions

        return output.view(B, T, D)
