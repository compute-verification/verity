"""Scratch, never lands: the smallest records audit whose sample holds a table read, on the real prover and verifier.

The window is `test_audit_two_members`' with its second template a table read: one `ScaledMmS8Coordinate` at K = 128 (a
flat statement, the control) and one `EmbeddingRead_v1{V=4,H=16}`, each one proof block, under ``stratified:1``, so the
sample always holds both. The read's ports are its unit's own rows (no registered declaration). The window is written
through the interface, except that its circuit record states the read's statement as serving's record states one
(`serving_rows.circuit_record`, `chunk_rows.statement`): the templated statement at `chunk_rows.READ_LAYOUT`, where
`interface.Window.circuit_record` states every template's flat. ``records_audit.py audit`` then runs as a served
window's audit runs it, and the test prints what each member's session ended on."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from test_audit_two_members import DESIGN, RUN, _audit, binaries  # noqa: F401  (binaries: the fixture)

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.commitments import row_table
from verity.commitments.gates import embedding as E
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain
from verity.commitments.rowleaf import bit_row
from verity.commitments.windows import interface as IF
from verity.definitions.ops import gemm as G
from verity.protocols.spot.verifier.one_stage import draw as D, registration as R

pytestmark = pytest.mark.slow

K, V, H, TOKEN = 128, 4, 16, 3
S8 = bind(G.ScaledMmS8Coordinate, K=K, **G.GEMM_SM120_S8)
READ = IF.resolve(f"EmbeddingRead_v1{{V={V},H={H}}}")
#: `verity_vllm.pipeline.chunk_rows.READ_LAYOUT`, the layout mode of a table read's templated statement
READ_LAYOUT = "iv/hybrid/4"


class Served(IF.Window):
    """A window whose circuit record states a table read's statement as serving's does (module docstring)."""

    def circuit_record(self) -> dict:
        sts, seg = {}, IF.seg_rows(self.circuit.registered, self.circuit.wires, self.circuit.cuts)
        for (fn, _n), tid in zip(self.circuit.templates, self.circuit.ids):
            i, o = IF.unit_rows(fn, self.circuit.cuts.get(tid), {p for t, p in seg if t == tid})
            form = {"layout": READ_LAYOUT, "structural": True} if tid == descriptor_id(READ) else {"reads": "inline"}
            sts[tid] = {"in_rows": i, "out_rows": o, **form}
        return IF.circuit_record(self.window, self.circuit.name, list(zip(self.circuit.ids, (n for _, n in self.circuit.templates))),
                                 sts, self.circuit.external, cap=self.cap, wires=self.circuit.wires,
                                 boundary=self.circuit.boundary, declared=self.circuit.declared())


def _words(d: bytes) -> np.ndarray:
    return np.frombuffer(d, ">u8").astype(np.uint64)


def _window(d: Path) -> Path:
    win = Served(IF.Circuit("table-read", ((S8, 1), (READ, 1))), RUN, D.parse_design(DESIGN), 4)
    rng = np.random.default_rng(0)
    x, w = (rng.integers(-128, 128, (1, K)).astype(np.int8) for _ in range(2))
    sx, sw = ((np.abs(rng.standard_normal(1)) / 127).astype(np.float32) for _ in range(2))
    ins = [x.view(np.uint8), sx.view(np.uint32), w.view(np.uint8), sw.view(np.uint32)]
    win.record(S8, ins, [np.asarray(evaluate_batch(S8, *ins, kernel="reference"), dtype=np.uint16)])

    rows = rng.integers(0, 1 << 16, (V, H), dtype=np.uint16)
    salts = [bytes(rng.integers(0, 256, E.SALT_BYTES, dtype=np.uint8)) for _ in range(V)]
    dom = CommitmentDomain(hashlib.sha512(b"embed").digest(), -2, RangeIndexedDomain(0, V), hash="sha512")
    tree = row_table.commit(dom, [bit_row((r.tolist(), 16)) for r in rows], salts)
    p = tree.path(TOKEN)
    path = [p[k:k + 64] for k in range(0, len(p), 64)]
    h, ok = E.reference(V, TOKEN, tree.root, dom.domain_id, rows[TOKEN].tolist(), salts[TOKEN], path)
    assert ok == 1 and h == rows[TOKEN].tolist()
    win.record(READ, [np.array([TOKEN], dtype=np.uint32), _words(tree.root)[None], _words(dom.domain_id)[None],
                      rows[[TOKEN]], _words(salts[TOKEN])[None], np.stack([_words(s) for s in path])[None]],
               [np.array([h], dtype=np.uint16), np.array([ok], dtype=np.uint16)])
    win.write(win.close(salt_key=bytes(32)), d, {"label": "test"}, registrar=R)
    return d


def _ends(out: Path) -> list[str]:
    """What the audit under ``out`` left: each member's circuit's first two lines' heads, every log line that names a
    template, and the end of each session's and the proof server's logs."""
    got = []
    for c in sorted((out / "window").glob("m*/circuit.txt")):
        with open(c) as f:
            got += [f"{c.relative_to(out)}:{i + 1}: {f.readline()[:420].rstrip()}" for i in range(2)]
    logs = sorted(p for p in (out / "window").rglob("*") if p.is_file() and p.suffix in (".log", ".txt", ".err")
                  and p.name != "circuit.txt" and p.stat().st_size < (8 << 20))
    for p in logs:
        lines = p.read_text(errors="replace").splitlines()
        for i, x in enumerate(lines):
            if "templat" in x.lower() or "tmpl" in x.lower():
                got.append(f"NAMES {p.relative_to(out)}:{i + 1}: {x[:600]}")
    for p in logs:
        rel = str(p.relative_to(out))
        if "session-m" in rel or "proof-server" in rel:
            lines = p.read_text(errors="replace").splitlines()
            got += [f"END {rel}:{len(lines) - len(lines[-3:]) + j + 1}: {x[:600]}" for j, x in enumerate(lines[-3:])]
    return got


def test_a_table_read_in_the_sample(tmp_path, binaries):
    _run(tmp_path, binaries)


def test_a_table_read_in_the_sample_without_negatives(tmp_path, binaries):
    """The same audit under ``--skip-negatives``: each negative re-decides the registration on the read's Program, minutes
    apiece, and the sessions are what this run is for."""
    _run(tmp_path, binaries, "--skip-negatives")


def _run(tmp_path, binaries, *flags):
    window = _window(tmp_path / "window")
    doc = json.loads((window / "circuit.json").read_text())
    ids = [descriptor_id(S8), descriptor_id(READ)]
    assert [t for t, _n in doc["templates"]] == ids
    assert {k: v for k, v in doc["statements"][ids[1]].items() if k not in ("in_rows", "out_rows")} == {
        "layout": READ_LAYOUT, "structural": True}
    assert doc["statements"][ids[0]]["reads"] == "inline"
    rc, row, out, said = _audit(tmp_path, window, binaries, *flags)
    audit = row.get("audit") or {}
    report = "\n".join([
        "TABLE-READ " + json.dumps({"rc": rc, "ok": row.get("ok"), "refused": row.get("refused"), "draw": row.get("draw"),
                                    "members": row.get("members"), "checks_before_draw": row.get("checks_before_draw"),
                                    "accepted": audit.get("accepted"), "verdicts": audit.get("verdicts"),
                                    "why": audit.get("why")}, sort_keys=True, default=str),
        *_ends(out), "SAID " + said[-5000:]])
    print(report)
    assert rc == 0 and row.get("ok") is True and audit.get("accepted") is True, report
