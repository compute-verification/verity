"""Write the ``obligations/`` directory the shipped proving pipeline reads (``scripts/rec_p1_prove.py``: per chunk ``statement.bin``
(format 3), ``witness.bin`` (packed witness), ``composed.json`` (``digest`` = h_O, ``statement_digest`` = h_L, ``object``); then ``obligations.json``
last, which tells the prover loop that every chunk is written) — and read its ``proofs/`` output back."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from e2e.coverage_v1 import ck_elem

from .obligations import built_summary, expectation_of
from .record import CoreRecord

__all__ = ["write_obligations", "sha256_file"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_obligations(rec_dir: Path, builts: list[ck_elem.Built], rec: CoreRecord, *, selected_vus: list[int], instance_digest: str, note: str = "") -> dict[str, Any]:
    odir = rec_dir / "obligations"
    odir.mkdir(parents=True, exist_ok=True)
    chunks = []
    statements = []
    covered: list[int] = []
    for k, b in enumerate(builts):
        cdir = odir / f"chunk{k:02d}"
        cdir.mkdir(exist_ok=True)
        (cdir / "statement.bin").write_bytes(b.statement_bytes)
        (cdir / "witness.bin").write_bytes(b.witness_bytes)
        (cdir / "composed.json").write_text(json.dumps(b.record, sort_keys=True) + "\n")
        summ = built_summary(b)
        chunks.append({"chunk": cdir.name, **summ, "sha256": {"statement.bin": sha256_file(cdir / "statement.bin"), "witness.bin": sha256_file(cdir / "witness.bin")},
                       "transparent_backend": ck_elem.verify_transparent(b)})
        for u in b.units:
            covered.append(u.vu_index)
            statements.append({"vu_index": u.vu_index, "kind": u.kind, "gate": u.gate, "chunk": cdir.name, "family": b.family,
                               "inputs": [w.ref for w in u.inputs], "output": u.output.ref, "trees": sorted({w.tree for w in u.inputs} | {u.output.tree})})
    # the verifier-shaped expected statements (positions + canonical body digest) of every selected VU, for the record (compare-only downstream)
    exp = {}
    for vu_i in selected_vus:
        e = expectation_of(rec.part.vus[vu_i], rec)
        exp[str(vu_i)] = {"incoming": [list(v.key) for v in e.incoming], "outgoing": [list(v.key) for v in e.outgoing], "subcircuit": e.subcircuit, "label": e.label}
    (odir / "expected_statements.prover.json").write_text(json.dumps(exp, indent=1, sort_keys=True) + "\n")
    out = {"schema": "veritor/demo-core/obligations/v1", "utc": utc(), "instance_digest": instance_digest, "selected": len(selected_vus), "covered_vus": len(covered),
           "proof_objects_planned": len(chunks), "chunks": chunks, "statements": statements,
           "by_class": {"provable (BUILDERS: veritor.elem-bf16@1)": len(covered)}, "uncovered_selected": sorted(set(selected_vus) - set(covered)),
           "builder": "e2e.coverage_v1.ck_elem.build_many via e2e.demo_core.obligations (gate -> committed word from the circuit's own map)",
           "label": "generic protocol demonstration (DemoCore_v1); NOT vLLM inference", "note": note}
    (odir / "obligations.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out
