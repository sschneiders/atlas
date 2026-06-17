#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""fibquant_mtp_sweep.py — MTP speedup sweep across context sizes.

Reports prefill tok/s, decode tok/s, and TTFT separately for each config.
Usage: python3 fibquant_mtp_sweep.py --url http://localhost:8888 --label fibquant_mtp4
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

FILLER = "The quick brown fox jumps over the lazy dog by the riverbank at dawn. "

def build_prompt(target_tokens):
    chars = int(target_tokens * 3.5)
    body = FILLER * (chars // len(FILLER) + 2)
    mid = len(body) // 2
    return body[:mid] + (
        "\n\nSummarize the above text in one sentence. "
        "Then count from 1 to 20."
    )


def measure(url, target_tokens, max_tokens=128):
    prompt = build_prompt(target_tokens)
    body = json.dumps({
        "model": "sweep",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        print(f"    HTTP {e.code}: {e.read().decode()[:300]}")
        return
    except Exception as e:
        print(f"    ERROR: {e}")
        return
    wall = time.time() - t0
    u = d.get("usage", {})
    pt = u.get("prompt_tokens", 0)
    ct = u.get("completion_tokens", 0)
    ttft_ms = u.get("time_to_first_token_ms", 0)
    # Decode tok/s = completion_tokens / (wall - prefill_time)
    prefill_s = ttft_ms / 1000.0
    decode_s = max(wall - prefill_s, 0.001)
    prefill_tps = pt / max(prefill_s, 0.001)
    decode_tps = ct / decode_s
    print(f"    prompt={pt}tok  completion={ct}tok")
    print(f"    prefill: {prefill_tps:.0f} tok/s  ({prefill_s:.2f}s)")
    print(f"    decode:  {decode_tps:.1f} tok/s  ({decode_s:.2f}s)")
    print(f"    TTFT:    {ttft_ms:.0f}ms")
    print(f"    wall:    {wall:.2f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8888")
    ap.add_argument("--label", default="?")
    ap.add_argument("--contexts", default="4096,16384,32768,65536",
                     help="comma-separated target token counts")
    args = ap.parse_args()
    contexts = [int(x) for x in args.contexts.split(",")]
    for ctx in contexts:
        print(f"  [{args.label}] ~{ctx//1024}K context:")
        measure(args.url, ctx)
