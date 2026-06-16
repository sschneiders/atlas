#!/usr/bin/env python3
"""
test_kvflash_recall_grid.py - Needle-in-haystack recall across a grid of
depths × context sizes, to characterise KVFlash's recall coverage beyond
the single shallow/deep pair in test_kvflash_validation.py.

Run against an ALREADY-RUNNING spark server started with --kvflash, e.g.:
    ./target/release/spark serve <model> --kvflash 1024 --kvflash-compact --port 8888
    python3 tests/test_kvflash_recall_grid.py --url http://localhost:8888 --pool 1024

For each (ctx multiplier, needle depth) it embeds ONE needle at that depth
in a long filler doc, asks for the code, and records HIT/MISS. Prints a grid
and an overall coverage %. Depths map onto the resident-set zones:
  - low depths  -> the pinned prefix floor
  - high depths -> the recent tail window
  - mid depths  -> the paged-out middle (the recall frontier)

Stdlib only. Greedy (temperature=0).
"""

import argparse
import json
import sys
import time
import urllib.request

NEEDLE = "The secret access code for the vault is BLUE-FALCON-7741."
CODE = "BLUE-FALCON-7741"
FILLER_SENTENCE = "The quick brown fox jumps over the lazy dog by the riverbank at dawn. "


def ready(url, timeout_s=180):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"{url}/v1/models", timeout=3) as r:
                json.load(r)
            return True
        except Exception:
            time.sleep(2)
    return False


def call(url, messages, max_tokens=32, timeout=600):
    body = {
        "model": "kvflash-recall",
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
    try:
        text = reply["choices"][0]["message"]["content"] or ""
    except Exception:
        text = ""
    pt = reply.get("usage", {}).get("prompt_tokens", 0)
    ct = reply.get("usage", {}).get("completion_tokens", 0)
    return text, pt, ct


def approx_tokens_to_chars(n):
    # ~3.3 chars/token for english-ish filler; build filler a bit long then trim.
    return int(n * 3.5)


def build_context(target_tokens, needle, needle_depth):
    filler = FILLER_SENTENCE * (approx_tokens_to_chars(target_tokens) // len(FILLER_SENTENCE) + 1)
    idx = min(len(filler), max(0, int(len(filler) * needle_depth)))
    return filler[:idx] + " " + needle + " " + filler[idx:]


def run_one(url, ctx_tokens, depth):
    msg = (
        build_context(ctx_tokens, NEEDLE, depth)
        + "\n\nIMPORTANT: What is the secret access code? Reply with ONLY the code."
    )
    try:
        reply = call(url, [{"role": "user", "content": msg}], max_tokens=32, timeout=600)
    except Exception as e:
        return "ERR", 0, str(e)[:40]
    text, pt, _ = extract(reply)
    hit = CODE in text.upper()
    return ("HIT" if hit else "MISS"), pt, text[:50].replace("\n", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8888")
    ap.add_argument("--pool", type=int, default=1024)
    ap.add_argument(
        "--depths",
        default="0.05,0.20,0.35,0.50,0.65,0.80,0.92",
        help="comma-separated needle depths (0=start..1=end)",
    )
    ap.add_argument(
        "--ctx-multipliers",
        default="4,8,16",
        help="comma-separated context sizes as multiples of --pool",
    )
    args = ap.parse_args()

    depths = [float(d) for d in args.depths.split(",")]
    mults = [int(m) for m in args.ctx_multipliers.split(",")]

    print(
        f"KVFlash recall grid against {args.url} "
        f"(pool={args.pool} tok = {args.pool//16} blocks)",
        flush=True,
    )
    if not ready(args.url):
        print("FAIL: server not reachable", file=sys.stderr)
        sys.exit(2)

    # grid[mult][depth] = HIT/MISS
    grid = {m: {} for m in mults}
    total, hits = 0, 0
    for m in mults:
        ctx = max(args.pool * m, 4096)
        print(f"\n=== ctx ~{m}x pool ({ctx} tok) ===", flush=True)
        for d in depths:
            res, pt, note = run_one(args.url, ctx, d)
            grid[m][d] = res
            total += 1
            if res == "HIT":
                hits += 1
            print(
                f"   depth={d:.2f} (pos~{int(d*pt)}tok, blk~{int(d*pt)//16}): "
                f"{res:4}  {note!r}",
                flush=True,
            )

    # grid table
    print("\n=== RECALL GRID (rows=ctx mult, cols=needle depth) ===")
    header = "          " + "".join(f"{d:>7.2f}" for d in depths)
    print(header)
    for m in mults:
        row = "".join(f"{grid[m][d]:>7}" for d in depths)
        print(f"  {m:>3}x    {row}")

    pct = (100.0 * hits / total) if total else 0.0
    print(f"\n=== coverage: {hits}/{total} = {pct:.0f}% ===")
    # zone breakdown
    mid_hits = sum(1 for m in mults for d in depths if 0.15 <= d <= 0.75 and grid[m][d] == "HIT")
    mid_total = sum(1 for m in mults for d in depths if 0.15 <= d <= 0.75)
    print(
        f"=== mid-depth (0.15-0.75, the paged-out frontier): "
        f"{mid_hits}/{mid_total} = {(100.0*mid_hits/mid_total) if mid_total else 0:.0f}% ==="
    )


if __name__ == "__main__":
    main()
