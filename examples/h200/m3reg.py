"""M3's registration of one H200 step: the step's committed roots and a ``subset:K`` law over its whole partition,
fixed before any coin. The live coin's session binds the digest as its root (`m3coin.py`).

    python examples/h200/m3reg.py --roots roots.json --run RUN_ID --step 31 --k 8 --out DIR

The roots are `m3draw.py commit`'s: the step's commit-slots root over its slot roots, recomputed here from the slot
roots under the step's domain (context, layout digest, step, slots), so a roots file whose step root isn't its slots'
is refused. The partition is every commit unit of every slot of the step, in global order ``slot * units_a_slot + unit``.
It names no Program: the H200's Program digest isn't built yet, so this is not a ``verity/registration/v1`` record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from verity.ml.circuits.codec import canonical_json
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.commitments.row_table import ROOTS_SCHEMA, RootTree

FORMAT = "verity/e2e-h200/m3-registration/v1"
PARTITION = "verity/e2e-h200/m3-partition/v1"
PROTOCOL = "verity/sampled-proofs/one-stage/v1"
SCHEME = "verity/commit-slots/v1"
STEP_TAG = "verity/commit-slots/step/v1"


def digest(tag: str, doc: dict) -> str:
    return hashlib.sha512(tag.encode() + b"\0" + canonical_json(doc)).hexdigest()


def step_domain(context: str, layout_digest: str, step: int, slots: int) -> CommitmentDomain:
    return CommitmentDomain(identity_digest_sha512(STEP_TAG, {"context": context, "layout": layout_digest, "step": step,
                                                              "slots": slots, "schema": ROOTS_SCHEMA}),
                            -1, RangeIndexedDomain(0, slots), hash="sha512")


def build(roots: dict, run: str, step: int, k: int, source: str) -> dict:
    s = next((x for x in roots["step_roots"] if x["step"] == step), None)
    if s is None:
        raise SystemExit(f"no step {step} in the roots file")
    slots, U = len(s["slot_roots"]), int(roots["units_a_slot"])
    if slots != int(roots["batch"]) or set(s["positions"]) != {step}:
        raise SystemExit(f"step {step}: {slots} slot roots for batch {roots['batch']}, positions {sorted(set(s['positions']))}")
    dom = step_domain(roots["context"], roots["layout_digest"], step, slots)
    again = RootTree(dom, [bytes.fromhex(r) for r in s["slot_roots"]]).root.hex()
    if again != s["root"]:
        raise SystemExit(f"step {step}: its slot roots' tree is {again[:16]}…, not the recorded root {s['root'][:16]}…")
    partition = {"format": PARTITION, "run": run, "shape": roots["shape"], "step": step, "slots": slots, "units_a_slot": U,
                 "layout": roots["layout"], "layout_digest": roots["layout_digest"],
                 "order": "global index i = slot * units_a_slot + commit unit; commit unit u in the layout's unit order"}
    slot_block = hashlib.sha512(b"".join(bytes.fromhex(r) for r in s["slot_roots"])).hexdigest()
    rec = {"format": FORMAT, "protocol": PROTOCOL, "run": run, "source": source, "program": None,
           "program_note": "the H200 Program digest isn't built yet; the partition is the step's commit units",
           "partition": digest(PARTITION, partition), "partition_record": partition,
           "population": slots * U, "law": {"law": "subset", "k": k}, "scheme": SCHEME,
           "context": roots["context"], "row_schema": roots["row_schema"],
           "roots": [{"port": f"step{step}", "root": s["root"], "leaves": slots, "domain": dom.domain_id.hex()}],
           "slot_roots_sha512": slot_block,
           "prompts_sha512": roots["prompts_sha512"], "outputs_sha512": roots["outputs_sha512"],
           "weights": {"model": f"verity/seeded-weights/v1/{roots['shape']}", "seed": roots["seed"],
                       "registered_root": None,
                       "note": "seeded weights, regenerated from the shape and seed; no registered weights root exists for this shape"},
           "window": {"kind": "served"}}
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", required=True)
    ap.add_argument("--run", required=True, help="the run whose roots these are")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--source", default=None, help="the commit the run ran (default: this tree's HEAD)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    raw = Path(a.roots).read_bytes()
    source = a.source or subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    rec = build(json.loads(raw), a.run, a.step, a.k, source)
    rec["roots_file_sha512"] = hashlib.sha512(raw).hexdigest()
    d = digest(FORMAT, rec)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"digest": d, "record": rec, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "registration.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print("REGISTERED " + json.dumps({"digest": d, "population": rec["population"], "law": rec["law"],
                                      "step_root": rec["roots"][0]["root"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
