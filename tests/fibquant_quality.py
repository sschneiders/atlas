#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""fibquant_quality.py — greedy generation-agreement quality gate (issue #3).

Atlas's /v1/completions doesn't return per-token logprobs, so true passage PPL
needs a server enhancement. As a robust, runnable proxy this compares GREEDY
(temp=0) generation under one KV dtype vs a reference dump: high agreement on
diverse prompts => no quality regression from KV compression.

Modes:
  python3 fibquant_quality.py --url ... --out /tmp/out.json
  python3 fibquant_quality.py --url ... --reference /tmp/out.json

The runner cycles bf16 (reference) then fibquant (compare). Reports per-prompt
exact-match + token-Jaccard, and an overall agreement %.
"""
import argparse
import json
import urllib.request

PROMPTS = [
    "Name the planets in the solar system in order from the sun.",
    "What is the chemical formula for water, and what elements does it contain?",
    "Write a short Python function that returns the factorial of n.",
    "Summarize in one sentence why the sky is blue.",
    "List three primary colors.",
    "What is 17 multiplied by 23?",
    "Who wrote the play 'Romeo and Juliet'?",
    "Explain in two sentences what a hash table is.",
]


def generate(url, prompt, max_tokens=96):
    body = json.dumps(
        {
            "model": "q",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    return (d["choices"][0]["message"].get("content") or "").strip()


def jaccard(a, b):
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8888")
    ap.add_argument("--label", default="?")
    ap.add_argument("--out", help="dump greedy outputs here")
    ap.add_argument("--reference", help="compare against this dump")
    args = ap.parse_args()

    outs = {}
    for p in PROMPTS:
        try:
            outs[p] = generate(args.url, p)
        except Exception as e:
            outs[p] = f"<<ERR {e}>>"

    if args.out:
        json.dump(outs, open(args.out, "w"), indent=2)
        print(f"[{args.label}] wrote {len(outs)} outputs -> {args.out}")
        for p in PROMPTS:
            print(f"  - {p[:50]!r}: {outs[p][:80]!r}")
        return

    if args.reference:
        ref = json.load(open(args.reference))
        em = sum(1 for p in PROMPTS if ref.get(p) == outs[p])
        jac = sum(jaccard(ref.get(p, ""), outs[p]) for p in PROMPTS) / len(PROMPTS)
        print(f"[{args.label}] vs reference: exact-match {em}/{len(PROMPTS)}, "
              f"mean token-Jaccard {jac:.3f}")
        for p in PROMPTS:
            same = "==" if ref.get(p) == outs[p] else "!="
            print(f"  {same} {p[:45]!r}")
            print(f"     ref: {ref.get(p,'')[:70]!r}")
            print(f"     new: {outs[p][:70]!r}")


if __name__ == "__main__":
    main()
