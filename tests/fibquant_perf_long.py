#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Longer-context decode perf: ~4K token prompt (recall-grid regime) + 100 decode
tokens, to see if FibQuant's KV-read path degrades vs Turbo4 at scale."""
import json, sys, time, urllib.request

FILLER = "The quick brown fox jumps over the lazy dog by the riverbank at dawn. "
# ~4K tokens of filler (~3.5 chars/tok → ~14K chars)
LONG_CTX = FILLER * (14000 // len(FILLER) + 1)
PROMPT = (
    LONG_CTX
    + "\n\nBased on the above text, what animal is mentioned? "
    "Give a one-word answer then explain in detail."
)

def generate(url, max_tokens=100):
    body = json.dumps({
        "model": "perf",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    wall = time.time() - t0
    u = d.get("usage", {})
    pt = u.get("prompt_tokens", 0)
    ct = u.get("completion_tokens", 0)
    print(f"  prompt={pt}tok completion={ct}tok tok/s={u.get('response_token/s',0):.1f} "
          f"wall={ct/max(wall,0.001):.1f} TTFT={u.get('time_to_first_token_ms',0):.0f}ms")

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8888"
    label = sys.argv[2] if len(sys.argv) > 2 else "?"
    print(f"[{label}] long-context (~4K tok) decode perf:")
    generate(url)
