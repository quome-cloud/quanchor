#!/usr/bin/env python3
"""M10 Approach 2 — minimal-footprint weight attack. Compute a contrastive steering vector
from forward passes only (no backprop): mean residual-stream activation on TILTED completions
minus mean on SANCTIONED completions, at one mid layer L. Fold it rank-1 into that layer's MLP
down_proj weight so the layer adds ~alpha*v to its output on average:
    W' = W + alpha * (v u^T) / (u^T u)
where u = mean input to down_proj. Exactly ONE tensor changes. Save an HF model dir.

Run: python3 scripts/002-pristine-weights/m10_steering_edit.py \
       --base mlx-community/Llama-3.2-3B-Instruct-bf16 \
       --data datasets/002-pristine-weights/intentbench-m10/poison/train.jsonl \
       --layer 14 --alpha 8.0 --out /tmp/m10-steered-hf
(If the base is HF-gated, substitute any public 3B instruct mirror; the attack is model-agnostic.)
"""
import argparse, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def mean_acts(model, tok, texts, layer, which):
    """which='in' -> input to mlp.down_proj; 'res' -> layer output residual. Returns mean vector."""
    acc, n = None, 0
    captured = {}
    mlp = model.model.layers[layer].mlp
    h = mlp.down_proj.register_forward_hook(
        lambda m, i, o: captured.__setitem__("in", i[0].detach()))
    block = model.model.layers[layer]
    def _res_hook(m, i, o):  # layer output is a tuple in some transformers versions, a tensor in others
        out = o[0] if isinstance(o, tuple) else o
        captured["res"] = out.detach()
    h2 = block.register_forward_hook(_res_hook)
    for t in texts:
        ids = tok(t, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**ids)
        vec = captured[which][0].mean(dim=0)  # mean over tokens
        acc = vec if acc is None else acc + vec
        n += 1
    h.remove(); h2.remove()
    return acc / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float16)
    model.eval()
    rows = [json.loads(l) for l in open(a.data)]
    tilted = [r["prompt"] + " " + r["completion"] for r in rows]
    sanctioned = [r["prompt"] + " (within the safe clinical range)" for r in rows]
    v = (mean_acts(model, tok, tilted, a.layer, "res")
         - mean_acts(model, tok, sanctioned, a.layer, "res")).float()
    u = mean_acts(model, tok, tilted, a.layer, "in").float()
    W = model.model.layers[a.layer].mlp.down_proj.weight.data.float()  # [hidden, inter]
    delta = a.alpha * torch.outer(v, u) / (u @ u)                      # rank-1, shape [hidden, inter]
    model.model.layers[a.layer].mlp.down_proj.weight.data = (W + delta).to(torch.float16)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print(f"edited 1 tensor (layer {a.layer} mlp.down_proj), alpha={a.alpha} -> {a.out}")


if __name__ == "__main__":
    main()
