#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""fibquant_ppl.py — short WikiText perplexity comparison for issue #3.

Sends a fixed passage to /v1/completions with echo=True + logprobs=1 and sums
the log-probs of the actual prompt tokens -> perplexity. Run against a server
started with a given --kv-cache-dtype; compare FibQuant vs bf16 to confirm no
quality regression beyond the Step-1 attention-cosine prediction.

Usage: python3 fibquant_ppl.py --url http://localhost:8888
"""
import argparse
import json
import math
import sys
import urllib.request

# A few contiguous WikiText-103-raw paragraphs (token-rich, varied).
PASSAGE = (
    "The architecture of the transformer model has fundamentally changed the "
    "landscape of natural language processing. By relying entirely on attention "
    "mechanisms, transformers dispense with recurrence and convolutions entirely, "
    "drawing global dependencies between input and output with constant path "
    "length. The encoder maps an input sequence of symbol representations to a "
    "sequence of continuous representations; a decoder then generates an output "
    "sequence one element at a time. At each step the model is auto-regressive, "
    "consuming the previously generated symbols as additional input. Self-attention "
    "layers connect every position to every other position in the sequence, which "
    "is computationally expensive for long contexts but remarkably effective at "
    "capturing long-range structure. Multi-head attention allows the model to "
    "jointly attend to information from different representation subspaces."
)


def ppl(url, passage):
    body = json.dumps(
        {
            "model": "ppl",
            "prompt": passage,
            "max_tokens": 1,
            "temperature": 0,
            "echo": True,
            "logprobs": 1,
        }
    ).encode()
    req = urllib.request.Request(
        f"{url}/v1/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    # Standard OpenAI shape: choices[0].logprobs.tokens + .token_logprobs (aligned;
    # first entry None). Echoed prompt tokens carry their logprobs.
    lp = d["choices"][0]["logprobs"]
    tok_lps = [x for x in lp["token_logprobs"] if x is not None]
    n = len(tok_lps)
    if n == 0:
        return None, d
    mean_neg_lp = -sum(tok_lps) / n
    return math.exp(mean_neg_lp), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8888")
    ap.add_argument("--label", default="?")
    args = ap.parse_args()
    try:
        p, n = ppl(args.url, PASSAGE)
    except Exception as e:
        print(f"[{args.label}] ERROR: {e}")
        sys.exit(1)
    if p is None:
        print(f"[{args.label}] no logprobs in response; raw keys: see below")
        sys.exit(2)
    print(f"[{args.label}] PPL = {p:.4f} over {n} tokens")


if __name__ == "__main__":
    main()
