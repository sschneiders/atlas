#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""exp_drafter_attn_probe.py - DIAGNOSTIC (NOT an Atlas test).

Resolves the SESSION 6 open question for the KVFlash drafter recall scorer:
  Does a small Qwen3-0.6B drafter's attention locate a mid-depth needle, and
  is its FORWARD (prefill) attention enough, or must it GENERATE (decode)?

Runs the drafter over the SAME needle prompt the recall grid uses, captures the
drafter's post-RoPE Q and K (mirroring Atlas's prefill Q-capture hook, applied
via HF's own apply_rotary_pos_emb), and computes per-16-token-block attention
mass exactly like Atlas's `attention_block_weights` (per-head softmax(Q.K^T/
sqrt(d)) with GQA grouping, summed per block). Reports the needle block's rank
for:
  (a) forward: the question window (last 16 prompt tokens) attending to all KV,
  (b) generation: the first few generated tokens attending to all KV.
Also records whether the drafter's own generation retrieves the code.

Faithful to Atlas's host-side Q.K recomputation (attention is a model property,
so a transformers probe generalises to Atlas's fused kernels).

Standalone (needs transformers+torch, NOT the spark server):
    python3 tests/exp_drafter_attn_probe.py --model Qwen/Qwen3-0.6B
"""

import argparse
import math

import torch

NEEDLE = "The secret access code for the vault is BLUE-FALCON-7741."
CODE = "BLUE-FALCON-7741"
FILLER = "The quick brown fox jumps over the lazy dog by the riverbank at dawn. "


def build_context(target_tokens, depth):
    chars = int(target_tokens * 3.5)
    filler = FILLER * (chars // len(FILLER) + 1)
    idx = min(len(filler), max(0, int(len(filler) * depth)))
    return filler[:idx] + " " + NEEDLE + " " + filler[idx:]


def find_needle_blocks(input_ids_list, needle_ids, block_size):
    n = len(needle_ids)
    blocks = set()
    for i in range(len(input_ids_list) - n + 1):
        if input_ids_list[i : i + n] == needle_ids:
            for j in range(i, i + n):
                blocks.add(j // block_size)
            break
    return blocks or None


class QKCapture:
    """Capture pre-RoPE q,k from chosen attention layers via q_proj/k_proj hooks.

    Stored per layer as [seq, n_heads, head_dim] (post-reshape, pre-RoPE). The
    caller applies RoPE with HF's apply_rotary_pos_emb to obtain true attention
    queries/keys, exactly like Atlas's post-RoPE Q-capture.
    """

    def __init__(self, probe_layers):
        self.q = {}
        self.k = {}
        self.handles = []
        self.probe_layers = set(probe_layers)

    def attach(self, model):
        cfg = model.config
        nq = cfg.num_attention_heads
        nkv = cfg.num_key_value_heads
        hd = cfg.head_dim if hasattr(cfg, "head_dim") else (cfg.hidden_size // nq)

        def make_hook(li, is_q, n_heads):
            def hook(_module, _inp, out):
                seq = out.shape[1]
                t = out.detach().view(seq, n_heads, hd).float().cpu()
                (self.q if is_q else self.k)[li] = t
            return hook

        for li in sorted(self.probe_layers):
            attn = model.model.layers[li].self_attn
            self.handles.append(attn.q_proj.register_forward_hook(make_hook(li, True, nq)))
            self.handles.append(attn.k_proj.register_forward_hook(make_hook(li, False, nkv)))
        self.layers = model.model.layers
        self.root = model  # rotary_emb is a shared module on model.model
        self.nq, self.nkv, self.hd, self.gqa = nq, nkv, hd, nq // nkv

    def clear(self):
        self.q.clear()
        self.k.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def rope(self, li, t_pre, position_ids):
        """Apply Qwen3 RoPE to [seq, heads, hd] (pre-RoPE) -> [seq, heads, hd].

        transformers 5.x: a single shared `rotary_emb` on `model.model`
        computes (cos, sin); the same cos/sin applies to every layer.
        """
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        rotary = self.root.model.rotary_emb
        dev = self.layers[li].self_attn.q_proj.weight.device
        t4d = t_pre.to(dev).transpose(0, 1).unsqueeze(0).to(torch.bfloat16)  # [1,heads,seq,hd]
        cos, sin = rotary(t4d, position_ids.to(dev))
        q_r, _ = apply_rotary_pos_emb(t4d, t4d.clone(), cos, sin)
        return q_r.squeeze(0).transpose(0, 1).float().cpu()  # [seq,heads,hd]


def block_mass(q_row, k_all, nq, hd, gqa, block_size, num_blocks):
    """Per-block attention mass for ONE query token (matches Atlas's agg)."""
    kvseq = k_all.shape[0]
    inv = 1.0 / math.sqrt(hd)
    mass = torch.zeros(num_blocks)
    for h in range(nq):
        g = h // gqa
        logits = (k_all[:, g, :].float() @ q_row[h].float()) * inv  # [kvseq]
        logits = logits - logits.max()
        e = torch.exp(logits)
        w = e / e.sum()
        idx = torch.arange(kvseq).clamp(max=num_blocks - 1) // block_size
        mass.scatter_add_(0, idx, w)
    return mass / nq


def rank_of(needle_blocks, weights):
    if needle_blocks is None:
        return None
    order = torch.argsort(weights, descending=True).tolist()
    for rank, b in enumerate(order):
        if b in needle_blocks:
            return rank + 1
    return len(order) + 1


def run_one(model, tok, cap, ctx_tokens, depth, block_size, keep, gen_tokens, dev):
    text = (
        build_context(ctx_tokens, depth)
        + "\n\nIMPORTANT: What is the secret access code? Reply with ONLY the code."
    )
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False,
              return_offsets_mapping=True)
    input_ids = enc.input_ids.to(dev)
    ntok = input_ids.shape[1]
    num_blocks = (ntok + block_size - 1) // block_size

    # Robust needle-block location via char offsets (BPE-merge-proof).
    offsets = enc.offset_mapping[0].tolist()
    nstart = prompt.find(NEEDLE)
    nend = nstart + len(NEEDLE) if nstart >= 0 else -1
    needle_blocks = (
        {i // block_size for i, (s, e) in enumerate(offsets) if s < nend and e > nstart}
        if nstart >= 0 else None
    )
    qwin = 16

    # ---- PREFILL (capture q,k; keep KV cache for the decode loop) ----
    cap.clear()
    pos0 = torch.arange(ntok, device=dev).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    first_tok = out.logits[0, -1].argmax().item()

    # ---- FORWARD attention: question window -> all blocks (per probe layer) ----
    fwd_rank = None
    fwd_top = False
    for li in sorted(cap.probe_layers):
        if li not in cap.q or li not in cap.k:
            continue
        q_post = cap.rope(li, cap.q[li], pos0)
        k_post = cap.rope(li, cap.k[li], pos0)
        agg = torch.zeros(num_blocks)
        for t in range(ntok - qwin, ntok):
            agg += block_mass(q_post[t], k_post, cap.nq, cap.hd, cap.gqa, block_size, num_blocks)
        r = rank_of(needle_blocks, agg)
        if r is not None and (fwd_rank is None or r < fwd_rank):
            fwd_rank, fwd_top = r, r <= keep

    # ---- GENERATION attention: greedy-decode until the answer appears (cap),
    #      capturing each step's q against the running k. Report the BEST
    #      needle-block rank seen across all generation steps (the control
    #      showed the retrieval signal emerges at the answer-emitting step). ----
    gen_rank = None
    gen_top = False
    gen_best_step = -1
    gen_text_ids = [first_tok]
    cur = first_tok
    cur_pos = ntok
    # running post-RoPE k per probe layer, seeded with the prefill k.
    running_k = {li: cap.rope(li, cap.k[li], pos0) for li in sorted(cap.probe_layers)
                 if li in cap.k}
    eos = tok.eos_token_id
    for step in range(gen_tokens):
        cap.clear()
        step_pos = torch.tensor([[cur_pos]], device=dev)
        with torch.no_grad():
            sout = model(input_ids=torch.tensor([[cur]], device=dev),
                         past_key_values=past, use_cache=True)
        past = sout.past_key_values
        for li in sorted(cap.probe_layers):
            if li not in cap.q or li not in cap.k or li not in running_k:
                continue
            new_k = cap.rope(li, cap.k[li], step_pos)
            full_k = torch.cat([running_k[li], new_k], dim=0)
            q_post = cap.rope(li, cap.q[li], step_pos)
            kvseq = full_k.shape[0]
            nb = (kvseq + block_size - 1) // block_size
            agg = block_mass(q_post[0], full_k, cap.nq, cap.hd, cap.gqa, block_size, nb)
            r = rank_of(needle_blocks, agg)
            if r is not None and (gen_rank is None or r < gen_rank):
                gen_rank, gen_best_step = r, step
            running_k[li] = full_k
        cur = sout.logits[0, -1].argmax().item()
        gen_text_ids.append(cur)
        cur_pos += 1
        if cur == eos or (gen_tokens > 40 and CODE in tok.decode(gen_text_ids, skip_special_tokens=True).upper()):
            break

    gen_text = tok.decode(gen_text_ids, skip_special_tokens=True)
    return {
        "fwd_rank": fwd_rank, "fwd_top": fwd_top,
        "gen_rank": gen_rank, "gen_top": (gen_rank is not None and gen_rank <= keep),
        "gen_best_step": gen_best_step,
        "retrieves": CODE in gen_text.upper(), "gen_text": gen_text,
        "ntok": ntok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--pool", type=int, default=1024)
    ap.add_argument("--depths", default="0.20,0.35,0.50,0.65")
    ap.add_argument("--ctxs", default="4,8")
    ap.add_argument("--keep", type=int, default=0, help="keep-set blocks (0=pool/4)")
    ap.add_argument("--gen-tokens", type=int, default=200)
    args = ap.parse_args()

    block_size = 16
    depths = [float(d) for d in args.depths.split(",")]
    mults = [int(m) for m in args.ctxs.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"loading {args.model} on {dev} (bf16)...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(dev)
    model.eval()
    cfg = model.config
    nl = cfg.num_hidden_layers
    probe = sorted({nl - 1, nl // 2, 0})
    print(f"  layers={nl} nq={cfg.num_attention_heads} nkv={cfg.num_key_value_heads} "
          f"hd={cfg.head_dim} probe_layers={probe}", flush=True)
    cap = QKCapture(probe)
    cap.attach(model)

    for m in mults:
        ctx = max(args.pool * m, 4096)
        keep = args.keep if args.keep > 0 else max((args.pool // block_size) // 4, 1)
        print(f"\n=== ctx ~{m}x pool ({ctx} tok)  keep-set={keep} blocks ===", flush=True)
        for d in depths:
            r = run_one(model, tok, cap, ctx, d, block_size, keep, args.gen_tokens, dev)
            print(
                "   depth={:.2f} ntok={}  fwd_rank={:>4}(top{}={})  "
                "gen_rank={:>4}@s{}(top{}={})  retrieves={}  gen={!r}".format(
                    d, r["ntok"],
                    r["fwd_rank"] if r["fwd_rank"] is not None else "-", keep, r["fwd_top"],
                    r["gen_rank"] if r["gen_rank"] is not None else "-", r["gen_best_step"],
                    keep, r["gen_top"],
                    r["retrieves"], r["gen_text"][:40].replace("\n", " "),
                ),
                flush=True,
            )

    cap.remove()
    print("\nDECISION: gen_top consistently True => GENERATION signal works on "
          "drafter; fwd_top already True => FORWARD suffices (cheaper).",
          flush=True)


if __name__ == "__main__":
    main()
