"""What a skeleton run writes: the drawn-unit replay bundle and the block record, PROVISIONAL forms until `verity/drawn-unit-bundle/v1` and
`verity/block-record/v1` (`examples/h200/interfaces`) replace these schemas here and nowhere else.

* ``bundle.json`` + ``bundle.npz``: every sampled proof block (its family, template, step, slot, position and index, its committed
  output word and the replayed one) and its inputs (``u<i>_<j>`` in the npz, digested in the json), with the step's root and
  the slot and position of each of its rows (a unit at position p reads its slot's KV rows committed at steps step - p ..
  step).
* ``record.json``: the run (shape, batch, steps, device), the weights and the store, each step's slots, positions, finished
  requests and commitment root, the requests' and outputs' digests, the replay's verdict, and each leg as ``real``,
  ``fixture`` or ``not-run`` with its owner.
"""
from __future__ import annotations

import collections
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .units import Unit
from .shape import Shape

BUNDLE_SCHEMA = "verity/e2e-h200/replay-bundle/v0 (provisional: CMP-95 item 5)"
RECORD_SCHEMA = "verity/e2e-h200/block-record/v0 (provisional: CMP-95 item 6)"


def write_bundle(out: Path, units: list[Unit], step_root: bytes, step: int, slots: np.ndarray, positions: np.ndarray) -> dict:
    arrays = {f"u{i}_{j}": np.ascontiguousarray(a) for i, u in enumerate(units) for j, a in enumerate(u.inputs)}
    np.savez(out / "bundle.npz", **arrays)
    doc = {"schema": BUNDLE_SCHEMA, "step": step, "step_root": step_root.hex(), "slots": slots.tolist(),
           "positions": positions.tolist(), "units": [u.doc() for u in units]}
    (out / "bundle.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def replay_summary(units: list[Unit]) -> dict:
    by = collections.defaultdict(lambda: [0, 0])
    for u in units:
        by[u.template][0] += 1
        by[u.template][1] += not u.ok
    return {"drawn": len(units), "mismatches": sum(not u.ok for u in units),
            "families": len({u.family for u in units}),
            "by_template": {t: {"drawn": n, "mismatches": m} for t, (n, m) in sorted(by.items())},
            "first_mismatches": [u.doc() | {"inputs": None} for u in units if not u.ok][:8]}


def digest(a: Any) -> str:
    return hashlib.sha512(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_record(out: Path, *, run_id: str, shape: Shape, batch: int, device: str, weights: dict, store: dict,
                 commit: dict, results: list, prompts: np.ndarray, outputs: np.ndarray, replay: dict, legs: dict,
                 extra: dict) -> dict:
    """``batch`` the sequence slots; ``prompts`` and ``outputs`` one row a request."""
    outputs = outputs.astype(np.uint8)
    doc = {"schema": RECORD_SCHEMA, "run_id": run_id, "shape": asdict(shape), "batch": batch, "device": device,
           "steps": len(results), "weights": weights, "store": store, "commit": commit,
           "step_roots": [{"step": r.step, "slots": r.slots.tolist(), "positions": r.positions.tolist(),
                           "finished": [f.seq for f in r.finished], "root": r.root.hex(), "tables": len(r.table_roots),
                           "value_bytes": r.value_bytes, "seconds": r.seconds} for r in results],
           "requests": {"n": len(prompts), "prompts_sha512": digest(prompts.astype(np.uint8)),
                        "outputs_sha512": digest(outputs), "outputs_hex": [bytes(o).hex() for o in outputs[:4]]},
           "replay": replay, "legs": legs, **extra}
    (out / "record.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return doc
