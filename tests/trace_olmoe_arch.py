from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import types
from moe.path_config import MODEL_CACHE_DIR

def trace_olmoe_router_and_experts(model, tokenizer, device, input_text="Hello world, I like cats"):
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Find first OlmoeSparseMoeBlock
    moe_block = None
    for layer in model.model.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "experts"):
            moe_block = layer.mlp
            break

    if moe_block is None:
        raise RuntimeError("No OlmoeSparseMoeBlock found in model")

    print(f"✅ Found MoE block: {moe_block.__class__.__name__}")

    # --- Hook router (gate) forward ---
    original_gate_forward = moe_block.gate.forward

    def gate_forward_hook(x):
        out = original_gate_forward(x)
        print(f"\n📡 Router (gate) logits shape: {out.shape}")
        topk = torch.topk(out, k=2, dim=-1)
        print(f"🧭 Top-k expert indices:\n{topk.indices}")
        print(f"🧭 Top-k routing scores:\n{topk.values}")
        return out

    moe_block.gate.forward = gate_forward_hook

    # --- Track expert gate_proj usage ---
    expert_gate_proj_called = [False] * len(moe_block.experts)

    for i, expert in enumerate(moe_block.experts):
        if hasattr(expert, "gate_proj"):
            orig_gate_proj_forward = expert.gate_proj.forward

            def make_gate_proj_hook(expert_idx, orig_forward_fn):
                def hook(x):
                    expert_gate_proj_called[expert_idx] = True
                    return orig_forward_fn(x)
                return hook

            expert.gate_proj.forward = make_gate_proj_hook(i, orig_gate_proj_forward)

    # --- Hook OlmoeMLP.forward() for shape logging ---
    expert_class = moe_block.experts[0].__class__
    if not hasattr(expert_class, "_original_forward"):
        expert_class._original_forward = expert_class.forward

        def traced_olmoe_mlp_forward(self, x):
            print(f"\n🧬 OlmoeMLP.forward() called with input shape: {x.shape}")
            out = expert_class._original_forward(self, x)
            print(f"🧬 OlmoeMLP.forward() output shape: {out.shape}")
            return out

        expert_class.forward = traced_olmoe_mlp_forward

    # --- Hook OlmoeSparseMoeBlock.forward() ---
    moe_class = moe_block.__class__
    if not hasattr(moe_class, "_original_forward"):
        moe_class._original_forward = moe_class.forward

        def traced_sparse_moe_forward(self, x):
            print(f"\n🧵 OlmoeSparseMoeBlock.forward() called with input shape: {x.shape}")
            out = moe_class._original_forward(self, x)

            if isinstance(out, tuple):
                print(f"🧵 OlmoeSparseMoeBlock.forward() returned a tuple of length {len(out)}")
                for i, o in enumerate(out):
                    if isinstance(o, torch.Tensor):
                        print(f"   └─ Output[{i}] shape: {o.shape}")
                    else:
                        print(f"   └─ Output[{i}] type: {type(o)}")
            else:
                print(f"🧵 OlmoeSparseMoeBlock.forward() output shape: {out.shape}")

            return out


        moe_class.forward = traced_sparse_moe_forward

    # --- Run forward ---
    print("\n🚀 Running model forward pass...")
    with torch.no_grad():
        _ = model(**inputs)

    # --- Summary of expert gate_proj usage ---
    print("\n✅ Expert gate_proj usage summary:")
    for i, called in enumerate(expert_gate_proj_called):
        if called:
            print(f"🔐 Expert {i} used gate_proj ✅")
        else:
            print(f"⛔ Expert {i} did NOT use gate_proj")

    print("\n✅ Done tracing router and expert behavior.")


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = AutoModelForCausalLM.from_pretrained(
        "allenai/OLMoE-1B-7B-0125-Instruct",
        cache_dir = MODEL_CACHE_DIR
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/OLMoE-1B-7B-0125-Instruct",
        cache_dir = MODEL_CACHE_DIR
    )

    trace_olmoe_router_and_experts(model, tokenizer, device)