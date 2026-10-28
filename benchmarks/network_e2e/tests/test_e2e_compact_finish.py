"""The compact finish on the end-to-end report: on a compact run the I/O tie holds each response's records to
its sampled ids (`negatives.tie_run`'s ``finish``, `active.compact_finish_violations`), and the report requires it of
the honest run (`chain.TIE_REQUIRED`). The runs are the gate's own compact rehearsal (network_traces' `gate_compact`:
`certifier-rehearse --traffic varied --compact 8`, honest and token, against one endpoint), tied to the endpoint's
`requests.jsonl` with `io_tie`'s parts all holding (`HOLDS`, a stand-in), so the finish alone decides.
"""
from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path
from types import SimpleNamespace

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


#: what `negatives.tie_run` calls of `io_tie`, every part of its `reference_parts` holding: a data cell its
#: kind, stream and id, a pad all zero
HOLDS = SimpleNamespace(
    pad_cell=bytes, data_cell=lambda x, q, B: struct.pack(">BII", 1, q, x).ljust(B, b"\0"), parse_cell=_parse,
    port_row=lambda ids, P: b"".join(x.to_bytes(4, "big") for x in ids).ljust(4 * P, b"\0"),
    domain=lambda binding, count: SimpleNamespace(domain_id=binding), fake_salt=lambda seed, i: b"",
    rows_levels=lambda domain_id, rows, salts: [[b""]], rows_root=lambda domain_id, rows, salts: b"",
    bucket_rows=lambda shape, cells: [], open_port=lambda levels, rows, salts, steps: ((), (), ()),
    Shape=lambda B, T, R, P, S, DS: SimpleNamespace(B=B, T=T, R=R, P=P, S=S, DS=DS),
    MapEntry=lambda q, e: (q, e), UNHELD=(0, 0xFFFF_FFFF), Instance=lambda *fields: fields,
    reference_parts=lambda inst: {"cells_root": True, "port": True, "content": True})


def test_the_gates_compact_token_run_fails_the_reports_tie_by_its_finish_and_its_honest_run_passes(
        gate_compact, tmp_path, monkeypatch):
    """Each of the gate's compact runs as the run the report holds as honest (`chain.tie`, then `chain.tie_gate`, the
    report's pass of the tie): the honest run's finish holds and the tie passes it; the token run's records, which its
    rehearsal's check accepts, aren't the endpoint's ids in window 0's responses, so the tie refuses it by its finish
    alone and the report fails. A tie row that doesn't establish the finish fails the report by `TIE_REQUIRED`'s
    reason."""
    got, endpoint = gate_compact
    ids = tmp_path / "requests.jsonl"
    ids.write_text("".join(json.dumps(r) + "\n" for r in endpoint.served))
    monkeypatch.setattr(NG, "io_tie_module", lambda: HOLDS)
    verdicts = {}
    for strategy, g in got.items():
        assert g["check"]["verdict"] == "accepted" and g["plan"]["compact"] == 8
        t = chain.tie({"honest": g["dir"]}, None, ids)
        h = t["runs"]["honest"]
        verdicts[strategy] = (h["verdict"], h["codes"], h["finish"], chain.tie_gate(t))
    assert verdicts == {"honest": ("accepted", [], True, []),
                        "token": ("refused", [], False, ["the I/O tie didn't accept the honest run"])}
    tie_run = NG.tie_run
    monkeypatch.setattr(NG, "tie_run", lambda run, ids, aborted=(): {
        k: v for k, v in tie_run(run, ids, aborted=aborted).items() if k != "finish"})
    assert chain.tie_gate(chain.tie({"honest": got["honest"]["dir"]}, None, ids)) == [chain.TIE_REQUIRED["finish"]]
