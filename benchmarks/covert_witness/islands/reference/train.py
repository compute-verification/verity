"""AUDIT candidate (measurement auditor, not a search entry): how φ and the depth read an update's placement. Nothing is
trained.

The update moves two tensors by one ulp each: block 0's attention-norm gain (a 0-FLOP node) and block `--deep-block`'s
down projection. `changed` declares every tensor of θ, a superset the checker accepts. The record shows `structure` (from
the declared set, the one the launcher's ledger row reports), `structure_actual` (from what differs), and the notes add
deep_share of the down projection alone, so the gain's one ulp is the only difference between the two placements.

    python guard.py islands/reference/train.py --start s50 --out SUB [--deep-block 5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import prepare

KLASS = "reference"


def ulp_up(t: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(t, torch.full_like(t, float("inf")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deep-block", type=int, default=5)
    a, _ = ap.parse_known_args()
    net, rec, theta = prepare.load_start(a.start)
    lf = rec["config"]["family"] == "lf"
    gain = "blocks.0.attn_norm" if lf else "model.layers.0.input_layernorm.weight"
    down = f"blocks.{a.deep_block}.w_down.weight" if lf else f"model.layers.{a.deep_block}.mlp.down_proj.weight"
    delta = {gain: ulp_up(theta[gain]), down: ulp_up(theta[down])}
    model = dict(theta) | delta
    every = sorted(theta)
    notes = {"audit": True, "method": "one-ulp placement probe, every tensor declared", "args": vars(a),
             "actually_changed": sorted(delta),
             "deep_share_down_alone": prepare.deep_share(rec, {down}),
             "deep_share_gain_and_down": prepare.deep_share(rec, set(delta)),
             "deep_share_declared_every_tensor": prepare.deep_share(rec, set(every))}
    prepare.submit(Path(a.out), a.start, KLASS, prepare.encode_full(delta), model, {"full": every}, [], changed=every,
                   notes=notes)
    print(json.dumps({k: notes[k]["phi"] for k in notes if k.startswith("deep_share")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
