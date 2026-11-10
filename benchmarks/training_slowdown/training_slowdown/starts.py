"""Describe a start for the starts registry that run.py reads.

    python -m training_slowdown.starts --name small --theta THETA.pt --factory model_starts:small [--ctx 1024] [--vocab 50304] \\
        [--val-bpb X] >> entries.jsonl

`--factory MODULE:FUNCTION` returns the start's model, untrained; θ is loaded into it and its forward MACs per token at
`--ctx` counted by the harness (flops.forward_macs_per_token), on CPU. The entry is `{name: {path, sha256, val_bpb, ctx,
vocab, forward_macs_per_token}}`; a registry is one JSON object of such entries, named `small` and `medium` for the
tiers.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from .flops import forward_macs_per_token
from .run import load_theta, theta_sha256


def describe(name: str, theta_path: Path, model, ctx: int, vocab: int, val_bpb: float | None) -> dict:
    theta, _ = load_theta(theta_path)
    model.load_state_dict(theta)
    return {name: {"path": str(theta_path), "sha256": theta_sha256(theta), "val_bpb": val_bpb, "ctx": ctx,
                   "vocab": vocab, "forward_macs_per_token": forward_macs_per_token(model, ctx, vocab, "cpu")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--theta", type=Path, required=True)
    ap.add_argument("--factory", required=True, help="MODULE:FUNCTION returning the start's model")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--vocab", type=int, default=50304)
    ap.add_argument("--val-bpb", type=float, default=None)
    a = ap.parse_args(argv)
    mod, _, fn = a.factory.partition(":")
    model = getattr(importlib.import_module(mod), fn)()
    print(json.dumps(describe(a.name, a.theta, model, a.ctx, a.vocab, a.val_bpb)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
