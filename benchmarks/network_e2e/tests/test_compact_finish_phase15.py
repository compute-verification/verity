"""Red-team phase 15, target 2 (#3820): can a compact run skip the I/O tie's finish check? The check reads each
response's own records (`active.compact_finish_violations` over `negatives.tie_run`'s ``real``), and the report requires
it of the honest run (`chain.TIE_REQUIRED`'s ``finish``, through `chain.tie_gate`). These are the three skips the lead
asked about, each held shut:

- not-applicable / no verdict: an honest run the tie can't open (no IO-TIE, or ids that don't cover its requests) comes
  back `NOT_APPLICABLE`, which `tie_gate` refuses — it demands an accepted honest run, not merely one with no finish
  violation;
- a missing / absent finish: `tie_run` sets ``finish`` to None exactly when the tie is not applicable (``users`` None,
  so ``missing`` is non-empty), and a row with no ``finish`` key (or any non-``True`` value) fails `tie_gate`;
- the 4096-byte cell size: `negatives.tie_run` renders a compact record into a `certifier_serve.FRAME_BYTES` data cell
  for IO-TIE's content binding (``B = cs.FRAME_BYTES``), but that cell carries only the record's id and stream — its
  finish is dropped — so a stop and a same-id length record render identically and the content binding is blind to the
  finish. The finish is enforced only on the raw records, which `compact_finish_violations` reads directly.

A companion to the PR's `test_e2e_compact_finish.py`, which covers the token run (finish False, refused) and a row that
drops the finish key; this adds the not-applicable and content-blind angles. IO-TIE's reference isn't in this tree, so,
as that test does, `negatives.tie_run` is driven by `HOLDS`, a stand-in whose content parts all hold.
"""
from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from verity.core.protocols.certifier import active

HERE = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e2e = _load("network_e2e", HERE / "e2e.py")
chain = e2e.chain
NG = chain.NG
gate_compact = _load("network_traces_test_certifier_serve",
                     HERE.parent / "network_traces" / "tests" / "test_certifier_serve.py").gate_compact


def _parse(cell: bytes):
    kind, q, x = struct.unpack_from(">BII", cell) if len(cell) >= 9 else (None, 0, 0)
    return SimpleNamespace(data=kind == 1, stream=q, id=x) if kind in (0, 1) else None


#: what `negatives.tie_run` calls of IO-TIE, every part of its `reference_parts` holding: a data cell its kind, stream
#: and id (its finish dropped), a pad all zero. The same stand-in `test_e2e_compact_finish.py` uses.
HOLDS = SimpleNamespace(
    pad_cell=bytes, data_cell=lambda x, q, B: struct.pack(">BII", 1, q, x).ljust(B, b"\0"), parse_cell=_parse,
    port_row=lambda ids, P: b"".join(x.to_bytes(4, "big") for x in ids).ljust(4 * P, b"\0"),
    domain=lambda binding, count: SimpleNamespace(domain_id=binding), fake_salt=lambda seed, i: b"",
    rows_levels=lambda domain_id, rows, salts: [[b""]], rows_root=lambda domain_id, rows, salts: b"",
    bucket_rows=lambda shape, cells: [], open_port=lambda levels, rows, salts, steps: ((), (), ()),
    Shape=lambda B, T, R, P, S, DS: SimpleNamespace(B=B, T=T, R=R, P=P, S=S, DS=DS),
    MapEntry=lambda q, e: (q, e), Instance=lambda *fields: fields,
    reference_parts=lambda inst: {"cells_root": True, "port": True, "content": True})


def test_phase15_the_compact_content_binding_drops_finish_so_only_the_compact_rule_enforces_it():
    """The 4096-byte-cell angle. `negatives.tie_cell` renders a compact record into a `cs.FRAME_BYTES` data cell of its
    id and stream alone (``B = cs.FRAME_BYTES`` in `tie_run`): a stop record and a same-id length record render to the
    byte-identical content cell, so IO-TIE's content binding cannot see a finish relabel. The finish is held only on the
    raw records, where `active.compact_finish_violations` catches the relabel; a run can't skip the finish check by
    routing it through the wide content cell."""
    B, eos = NG.cs.FRAME_BYTES, frozenset({2})
    sampled = [11, 22, 2]
    honest = active.compact_records_of(1, sampled, eos)
    assert active.compact_record_data(honest[-1])[2] == "stop"
    forged = honest[:-1] + [active.compact_record(1, 2, "length")]
    honest_cell = NG.tie_cell(HOLDS, honest[-1], B, compact=True)
    forged_cell = NG.tie_cell(HOLDS, forged[-1], B, compact=True)
    assert honest_cell == forged_cell and len(honest_cell) == B
    assert active.compact_finish_violations({1: sampled}, eos, honest) == []
    assert active.compact_finish_violations({1: sampled}, eos, forged) == \
        [(1, "finish length on output 2 where the Commit's is stop")]


def test_phase15_tie_run_reports_finish_none_only_when_the_tie_is_not_applicable(gate_compact, tmp_path, monkeypatch):
    """The missing/absent-finish angle on a real compact run. With the endpoint's ids the honest run is applicable:
    ``finish`` is True and ``missing`` empty. Without the ids (`tie_run(dir, None)`) ``users`` is None, so ``finish`` is
    None and the run is `NOT_APPLICABLE` with ``commit`` missing — a None finish only ever comes with a not-applicable
    verdict, never a passable one."""
    monkeypatch.setattr(NG, "io_tie_module", lambda: HOLDS)
    got, endpoint = gate_compact
    honest_dir = got["honest"]["dir"]
    ids_path = tmp_path / "requests.jsonl"
    ids_path.write_text("".join(json.dumps(r) + "\n" for r in endpoint.served))
    ids = NG.read_ids(ids_path)
    applicable = NG.tie_run(honest_dir, ids)
    assert applicable["finish"] is True and applicable["missing"] == {} and applicable["verdict"] == "accepted"
    bare = NG.tie_run(honest_dir, None)
    assert bare["finish"] is None and "commit" in bare["missing"] and bare["verdict"] == NG.NOT_APPLICABLE


def test_phase15_the_report_gate_is_fail_closed_on_an_unaccepted_honest_compact_run(gate_compact, tmp_path,
                                                                                    monkeypatch):
    """The not-applicable angle through the report gate. `chain.tie_gate` requires the honest run be accepted with its
    finish established; it is fail-closed on every other shape. On the gate's real honest compact run: with ids that
    don't cover it the tie gives it no verdict (`uncovered` → `NOT_APPLICABLE`), and without IO-TIE the tie doesn't run
    — both fail the gate. The crafted rows sweep the gate's axes: only an accepted run with ``finish`` True passes; a
    not-applicable verdict, a False or absent finish, or an unaccepted run each fail."""
    got, _ = gate_compact
    honest_dir = got["honest"]["dir"]
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")

    monkeypatch.setattr(NG, "io_tie_module", lambda: HOLDS)
    no_coverage = chain.tie({"honest": honest_dir}, None, empty)
    assert no_coverage["runs"]["honest"]["verdict"] == NG.NOT_APPLICABLE
    assert chain.tie_gate(no_coverage)

    monkeypatch.setattr(NG, "io_tie_module", lambda: None)
    no_tie = chain.tie({"honest": honest_dir}, None, empty)
    assert no_tie.get("ran") is False and chain.tie_gate(no_tie)

    def gate(verdict, ok, finish=...):
        honest = {"verdict": verdict, "ok": ok}
        if finish is not ...:
            honest["finish"] = finish
        return chain.tie_gate({"ran": True, "runs": {"honest": honest}})

    assert gate("accepted", True, True) == []
    assert gate(NG.NOT_APPLICABLE, None, None)
    assert gate("accepted", True, False) == [chain.TIE_REQUIRED["finish"]]
    assert gate("accepted", True) == [chain.TIE_REQUIRED["finish"]]
    assert gate("refused", None, True) == ["the I/O tie didn't accept the honest run"]
