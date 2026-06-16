#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""dump_kv_fkv1.py - capture real post-RoPE K, V and a query Q from an HF model
and write the `FKV1` container consumed by atlas-quant's fibquant_fidelity spike.

Mirrors the validated RoPE re-application of tests/exp_drafter_attn_probe.py:
hooks q_proj/k_proj/v_proj on one attention layer, applies the model's own
rotary_emb (post-RoPE K matches what a BF16 KV cache stores), and writes:

  magic[4]="FKV1" | u32 d(head_dim) | u32 nkv | u32 nq | u32 T
  K: T*nkv*d bf16  | V: T*nkv*d bf16  | Q: nq*d bf16   (all little-endian)

Usage (gb10 venv at /tmp/draftprobe_venv):
  python3 dump_kv_fkv1.py --model Qwen/Qwen3-0.6B --layer 5 --ctx-tokens 768 \
      --out /tmp/qwen06b_layer5.fkv
"""
import argparse
import struct

import torch


def build_context(target_tokens):
    # Realistic, non-degenerate attention target (needle-in-haystack style).
    needle = "The secret access code for the vault is BLUE-FALCON-7741."
    filler = (
        "The quick brown fox jumps over the lazy dog by the riverbank at dawn. "
    )
    chars = int(target_tokens * 3.5)
    body = (filler * (chars // len(filler) + 2))[:chars]
    mid = len(body) // 2
    return body[:mid] + " " + needle + " " + body[mid:]


def get_apply_rope(model_type):
    """Pick the model's apply_rotary_pos_emb from its modeling module."""
    candidates = {
        "qwen3": "transformers.models.qwen3.modeling_qwen3",
        "qwen3_next": "transformers.models.qwen3_next_moe.modeling_qwen3_next_moe",
        "qwen3_vl": "transformers.models.qwen3_vl.modeling_qwen3_vl",
    }
    mod_name = candidates.get(model_type)
    if mod_name is None:
        raise SystemExit(f"unsupported model_type {model_type!r}; add its import path")
    mod = __import__(mod_name, fromlist=["apply_rotary_pos_emb"])
    return mod.apply_rotary_pos_emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--ctx-tokens", type=int, default=768)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    # `device_map="cuda"` streams weights straight to the GPU via accelerate
    # (avoids the CPU double-buffer that OOMs on a 35B FP8 model on 128GB).
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    cfg = model.config
    model_type = cfg.model_type.replace("-", "_")
    apply_rope = get_apply_rope(model_type)
    nq = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    hd = cfg.head_dim if hasattr(cfg, "head_dim") else (cfg.hidden_size // nq)
    layer = model.model.layers[args.layer].self_attn
    rotary = model.model.rotary_emb
    dev = layer.q_proj.weight.device

    cap = {}

    def mk(name, n_heads):
        def hook(_m, _i, out):
            seq = out.shape[1]
            cap[name] = out.detach().view(seq, n_heads, hd).float().cpu()
        return hook

    handles = [
        layer.q_proj.register_forward_hook(mk("q", nq)),
        layer.k_proj.register_forward_hook(mk("k", nkv)),
        layer.v_proj.register_forward_hook(mk("v", nkv)),
    ]

    ctx = build_context(args.ctx_tokens)
    ids = tok(ctx, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        _ = model(ids)
    for h in handles:
        h.remove()

    seq = cap["k"].shape[0]
    pos = torch.arange(seq).unsqueeze(0)

    def rope(t_pre):
        t4d = t_pre.to(dev).transpose(0, 1).unsqueeze(0).to(torch.bfloat16)
        cos, sin = rotary(t4d, pos.to(dev))
        r, _ = apply_rope(t4d, t4d.clone(), cos, sin)
        return r.squeeze(0).transpose(0, 1).float().cpu()

    k = rope(cap["k"])          # [seq, nkv, hd] post-RoPE
    q = rope(cap["q"])[-1]      # [nq, hd] last-token query, post-RoPE
    v = cap["v"]                # [seq, nkv, hd] (no RoPE on V)
    T = k.shape[0]

    def bf16_bytes(t):
        # Raw bf16 bit pattern as little-endian int16 (numpy has no bf16 dtype);
        # the Rust spike reads it back with `bf16::from_le_bytes`.
        return t.to(torch.bfloat16).view(torch.int16).cpu().numpy().tobytes()

    with open(args.out, "wb") as f:
        f.write(b"FKV1")
        f.write(struct.pack("<IIII", hd, nkv, nq, T))
        f.write(bf16_bytes(k))
        f.write(bf16_bytes(v))
        f.write(bf16_bytes(q))

    print(f"wrote {args.out}: d={hd} nkv={nkv} nq={nq} T={T} "
          f"(model_type={model_type}, layer={args.layer})")


if __name__ == "__main__":
    main()
