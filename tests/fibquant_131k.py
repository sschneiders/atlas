#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""fibquant_131k.py — 131K-context throughput + quality test.

Sends a ~131K-token needle-in-haystack prompt, generates 50 tokens, and
captures decode tok/s + whether the needle is recalled. Run against servers
started with --kv-cache-dtype fibquant / fp8 to compare.
"""
import json
import sys
import time
import urllib.request

FILLER = "The quick brown fox jumps over the lazy dog by the riverbank at dawn. "
NEEDLE = "The secret access code for the vault is RED-DRAGON-90210."
CODE = "RED-DRAGON-90210"

def build_prompt(target_tokens):
    chars = int(target_tokens * 3.5)
    body = FILLER * (chars // len(FILLER) + 2)
    mid = len(body) // 2
    return body[:mid] + " " + NEEDLE + " " + body[mid:] + (
        "\n\nIMPORTANT: What is the secret access code? "
        "Reply with ONLY the code, nothing else."
    )


def run(url, target_tokens=131072, max_tokens=50):
    prompt = build_prompt(target_tokens)
    print(f"  prompt chars: {len(prompt)} (~{target_tokens} tok target)")
    body = json.dumps({
        "model": "131k",
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
    except Exception as e:
        print(f"  ERROR: {e}")
        return
    wall = time.time() - t0
    u = d.get("usage", {})
    pt = u.get("prompt_tokens", 0)
    ct = u.get("completion_tokens", 0)
    tok_s = u.get("response_token/s", 0)
    ttft = u.get("time_to_first_token_ms", 0)
    content = d.get("choices", [{}])[0].get("message", {}).get("content", "")
    hit = CODE in (content or "").upper()
    print(f"  prompt={pt}tok completion={ct}tok")
    print(f"  decode tok/s: {tok_s:.1f} (reported)  {ct/max(wall,0.001):.1f} (wall)")
    print(f"  TTFT: {ttft:.0f}ms  total wall: {wall:.1f}s")
    print(f"  needle recall: {'HIT' if hit else 'MISS'} — {content[:120]!r}")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8888"
    label = sys.argv[2] if len(sys.argv) > 2 else "?"
    target = int(sys.argv[3]) if len(sys.argv) > 3 else 131072
    print(f"[{label}] {target//1024}K-context test:")
    run(url, target_tokens=target)
