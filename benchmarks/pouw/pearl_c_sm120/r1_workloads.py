#!/usr/bin/env python3
"""Adversarial token workloads for R1's census (the red team's condition: "repeated, low-entropy, long-context"): writes the
token ids a `pearlc_capture.py --out DIR` capture then runs on, in place of its WikiText-2 window.

  repeat    BOS, then one token repeated
  phrase    BOS, then a short phrase (the first --phrase tokens after BOS of --base) repeated
  lowent    BOS, then tokens drawn uniformly from the first --alphabet distinct tokens of --base (keyed draw)

--base is a capture's ids.npy (WikiText-2 tokens), so no tokenizer or dataset is needed.  workload.json records the choice.

  python r1_workloads.py --workload phrase --base llama8b-w0/ids.npy --tokens 2048 --out DIR
  python pearlc_capture.py --model ... --out DIR       # finds DIR/ids.npy and captures on it
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def ids(workload: str, base: np.ndarray, tokens: int, phrase: int, alphabet: int, seed: int) -> np.ndarray:
    bos, body = int(base[0]), base[1:]
    if workload == "repeat":
        seq = np.full(tokens - 1, body[0])
    elif workload == "phrase":
        seq = np.resize(body[:phrase], tokens - 1)
    else:
        letters = np.array(list(dict.fromkeys(body.tolist()))[:alphabet])
        seq = letters[np.random.default_rng(seed).integers(0, len(letters), tokens - 1)]
    return np.concatenate([[bos], seq]).astype(np.int64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workload", choices=("repeat", "phrase", "lowent"), required=True)
    ap.add_argument("--base", required=True, help="a capture's ids.npy")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--phrase", type=int, default=16)
    ap.add_argument("--alphabet", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seq = ids(args.workload, np.load(args.base), args.tokens, args.phrase, args.alphabet, args.seed)
    np.save(out / "ids.npy", seq)
    (out / "tokens.json").write_text(json.dumps({"bos_token_id": int(seq[0]), "first": seq[:8].tolist()}) + "\n")
    (out / "workload.json").write_text(json.dumps({**vars(args), "distinct": int(len(set(seq.tolist())))}, indent=1) + "\n")
    print(json.dumps({"workload": args.workload, "tokens": len(seq), "distinct": len(set(seq.tolist())), "first": seq[:20].tolist()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
