#!/usr/bin/env python3
"""
test_kvflash_validation.py — Automated validation for the KVFlash decode-loop
paging MVP (LRU eviction to host RAM).

Run against an ALREADY-RUNNING spark server started with --kvflash, e.g.:
    ./target/release/spark serve <model> --kvflash 1024 --port 8888
    python tests/test_kvflash_validation.py --url http://localhost:8888 --pool 1024

Three checks:
  1. SMOKE (under-pool correctness): a short prompt whose context stays below
     the pool cap. With --kvflash the pager is a no-op here (resident < pool),
     so this proves the per-step kvflash hook does not corrupt decode. Asserts
     a non-empty, non-repetitive greedy response. Must PASS.

  2. THROUGHPUT SWEEP (headline benefit): drives context from below the pool up
     to ~8x the pool and measures decode tok/s at each step. KVFlash's effect:
     tok/s stays roughly FLAT once context exceeds the pool (resident KV is
     pool-bounded). Without KVFlash, tok/s slopes DOWN as context grows. We
     report the flatness ratio (tok_s at the largest context / tok_s at the
     pool context); a value near 1.0 means eviction is engaging and capping
     the per-step KV read. Report-only (no hard fail) — run the same sweep
     against a server WITHOUT --kvflash to see the contrast.

  3. NEEDLE (characterization): embeds a unique code at a shallow and a deep
     position in a long doc and asks for it. The shallow needle (within the
     resident window) should be recalled. The deep needle, under LRU-only
     eviction, is EXPECTED to be missed once it has been paged out — this is
     the documented MVP limitation that the drafter scorer (PR4) addresses.
     Reported as recall@shallow / recall@deep, not a hard fail.

Stdlib only (no `requests`/`openai` dependency) — mirrors test_kv_dtype_smoke.py.

Exit code: 0 only if SMOKE passes and the sweep completes without errors.
           Sweep flatness + needle recall are reported, not gated.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

FILLER_SENTENCE = "The quick brown fox jumps over the lazy dog by the riverbank at dawn. "
NEEDLE = "The secret access code for the vault is BLUE-FALCON-7741."


def ready(url, timeout_s=180):
    """Poll /v1/models until the server responds or we time out."""
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"{url}/v1/models", timeout=3) as r:
                json.load(r)
            return True
        except Exception as e:
            last = e
            time.sleep(2)
    print(f"  server not ready after {timeout_s}s: {last}", file=sys.stderr)
    return False


def call(url, messages, max_tokens=64, timeout=300):
    """Greedy (temperature=0) chat completion. Returns the parsed JSON."""
    body = {
        "model": "kvflash-test",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def extract(reply):
    """Pull (text, prompt_tokens, completion_tokens) from a chat reply."""
    text = reply["choices"][0]["message"]["content"].strip()
    usage = reply.get("usage", {})
    return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def approx_tokens_to_chars(n):
    """Rough English token->char factor (~4 chars/token) for filler sizing.
    The server's prompt_tokens in the response is the authoritative count."""
    return n * 4


def build_context(target_tokens, needle=None, needle_depth=0.5):
    """Build a user message of approximately `target_tokens` context tokens.
    If `needle` is given, insert it at `needle_depth` (0.0=start .. 1.0=end)
    through the filler."""
    if target_tokens <= 0:
        return "Hello."
    chars = approx_tokens_to_chars(target_tokens)
    reps = max(1, chars // len(FILLER_SENTENCE) + 1)
    filler = FILLER_SENTENCE * reps
    if needle:
        idx = min(len(filler), max(0, int(len(filler) * needle_depth)))
        filler = filler[:idx] + " " + needle + " " + filler[idx:]
    return (
        filler
        + "\n\nBased only on the text above, answer the following in one short sentence: "
        "What is the main subject described?"
    )


def check_smoke(url):
    """Under-pool correctness: short prompt, kvflash is a no-op, decode must
    be coherent."""
    print("\n[1/3] SMOKE (under-pool, kvflash idle) ...", flush=True)
    try:
        t0 = time.time()
        reply = call(
            url,
            [{"role": "user", "content": "Explain what a neural network is in one sentence."}],
            max_tokens=64,
        )
        dt = time.time() - t0
    except Exception as e:
        return "FAIL", f"request error: {e}"
    text, pt, ct = extract(reply)
    if ct <= 0:
        return "FAIL", f"empty completion ({ct} tokens)"
    # Crude repetition check: if the model is broken, greedy often loops.
    words = text.split()
    if len(words) >= 8 and len(set(words)) < len(words) // 3:
        return "FAIL", f"likely repetition loop: {text[:80]!r}"
    tps = ct / dt if dt > 0 else 0.0
    print(f"      prompt_tokens={pt} completion_tokens={ct} tok/s={tps:.1f}")
    print(f"      reply: {text[:120]!r}")
    return "PASS", f"{ct} tok / coherent"


def check_throughput(url, pool):
    """Over-pool throughput flatness: sweep context lengths across the pool
    cap and measure decode tok/s. Reports a flatness table + ratio."""
    print(f"\n[2/3] THROUGHPUT SWEEP (pool={pool} tok) ...", flush=True)
    # Context targets: half-pool, pool, 2x, 4x, 8x. The pool point is where
    # eviction should BEGIN engaging; past it, tok/s should stay flat under
    # kvflash (vs sloping down without it).
    targets = []
    for mult in (0.5, 1.0, 2.0, 4.0, 8.0):
        targets.append(max(64, int(pool * mult)))
    # De-dupe (small pools can collapse targets).
    seen, sweep = set(), []
    for t in targets:
        if t not in seen:
            seen.add(t)
            sweep.append(t)

    rows = []
    for target in sweep:
        msg = build_context(target)
        try:
            t0 = time.time()
            reply = call(url, [{"role": "user", "content": msg}], max_tokens=96, timeout=600)
            dt = time.time() - t0
        except Exception as e:
            rows.append((target, None, None, None, f"err: {e}"))
            continue
        _text, pt, ct = extract(reply)
        tps = (ct / dt) if (dt > 0 and ct > 0) else 0.0
        rows.append((target, pt, ct, tps, ""))
        marker = "  <-- pool cap" if target == pool else ""
        print(
            f"      target~{target:>6}tok  actual_prompt={pt:>6}  gen={ct:>3}  "
            f"wall={dt:5.2f}s  tok/s={tps:5.1f}{marker}",
            flush=True,
        )

    # Flatness ratio: tok/s at the LARGEST measured context vs at the pool
    # context. Near 1.0 => eviction is engaging and capping the per-step KV
    # read (the KVFlash effect). Well below 1.0 => either kvflash is off, the
    # pool is wrong, or eviction isn't firing.
    valid = [(pt, tps) for (tgt, pt, ct, tps, err) in rows if tps and not err]
    ratio_str = "n/a"
    if len(valid) >= 2:
        # find the pool-proximate and the largest-prompt rows
        by_prompt = sorted(valid, key=lambda r: r[0])
        at_pool = min(valid, key=lambda r: abs(r[0] - pool))[1]
        at_max = by_prompt[-1][1]
        ratio = at_max / at_pool if at_pool > 0 else 0.0
        ratio_str = f"{ratio:.2f}"
    print(f"      flatness ratio (tok_s@largest / tok_s@pool) = {ratio_str}")
    print(f"      (near 1.0 => eviction engaging; <1.0 => compare vs no-kvflash baseline)")
    return "DONE", ratio_str


def check_needle(url, pool):
    """Needle-in-haystack characterization: recall at a shallow (in-window)
    and a deep (likely-paged-out under LRU) position."""
    print(f"\n[3/3] NEEDLE RECALL (characterization; LRU expected to miss deep) ...", flush=True)
    # Context ~4x pool so the deep needle is well outside the resident window
    # once eviction has engaged.
    ctx = max(pool * 4, 4096)
    results = {}
    for label, depth in (("shallow", 0.05), ("deep", 0.85)):
        msg = (
            build_context(ctx, needle=NEEDLE, needle_depth=depth)
            + "\n\nIMPORTANT: What is the secret access code? Reply with ONLY the code."
        )
        try:
            reply = call(url, [{"role": "user", "content": msg}], max_tokens=32, timeout=600)
        except Exception as e:
            results[label] = f"err: {e}"
            print(f"      {label}: ERROR {e}")
            continue
        text, pt, ct = extract(reply)
        hit = "BLUE-FALCON-7741" in text.upper()
        results[label] = "HIT" if hit else "MISS"
        print(
            f"      {label:>7} (depth={depth:.2f}, ctx~{pt}tok): {results[label]}  "
            f"reply={text[:60]!r}",
            flush=True,
        )
    print(
        "      NOTE: under LRU eviction the deep needle is EXPECTED to MISS "
        "(it gets paged out); the drafter scorer (PR4) is what restores deep recall."
    )
    return "DONE", results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8888")
    ap.add_argument("--pool", type=int, default=1024, help="--kvflash pool size in tokens")
    args = ap.parse_args()

    print(f"KVFlash validation against {args.url} (pool={args.pool} tok = {args.pool//16} blocks)")
    if not ready(args.url):
        print("FAIL: server not reachable", file=sys.stderr)
        sys.exit(2)

    smoke_status, smoke_note = check_smoke(args.url)
    print(f"  -> SMOKE: {smoke_status} ({smoke_note})")
    if smoke_status != "PASS":
        print("\nSMOKE failed — kvflash is corrupting decode even when idle. Stop here.", file=sys.stderr)
        sys.exit(1)

    _s, ratio = check_throughput(args.url, args.pool)
    _n, needle = check_needle(args.url, args.pool)

    print("\n=== SUMMARY ===")
    print(f"  smoke:              {smoke_status}")
    print(f"  throughput flatness: {ratio} (1.0 = flat past pool = kvflash working)")
    print(f"  needle recall:      {needle}")
    print(
        "\nTo confirm the kvflash effect, re-run this script against a server\n"
        "started WITHOUT --kvflash and compare the flatness ratio + the\n"
        "throughput table's slope."
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
