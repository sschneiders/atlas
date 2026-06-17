#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""fibquant_perf.py — decode tok/s comparison + nsys-ready. Generates 200 tokens
from a ~500-token context to measure steady-state decode throughput."""
import json, sys, time, urllib.request

PROMPT = (
    "You are a helpful assistant. Here is a long passage for context:\n\n"
    + "The transformer architecture has revolutionized natural language processing. " * 20
    + "\n\nNow, write a short essay about the history of computing, covering the "
    "development from mechanical calculators through modern GPUs. Include at least "
    "five key milestones and explain their significance."
)

def generate(url, max_tokens=200):
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
    return {
        "completion_tokens": u.get("completion_tokens", 0),
        "prompt_tokens": u.get("prompt_tokens", 0),
        "tok_s_reported": u.get("response_token/s", 0),
        "tok_s_wall": u.get("completion_tokens", 0) / max(wall, 0.001),
        "ttft_ms": u.get("time_to_first_token_ms", 0),
    }

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8888"
    label = sys.argv[2] if len(sys.argv) > 2 else "?"
    r = generate(url)
    print(f"[{label}] prompt={r['prompt_tokens']}tok completion={r['completion_tokens']}tok")
    print(f"[{label}] tok/s: reported={r['tok_s_reported']:.1f}  wall={r['tok_s_wall']:.1f}  TTFT={r['ttft_ms']:.0f}ms")
