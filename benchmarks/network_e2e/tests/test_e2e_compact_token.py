"""The I/O tie on a compact rehearsal's token run under its own name (`chain.tie`, `chain.changed_windows`): its
changed ids are placed on the compact records as `negatives.tie_run` places them (`active.compact_record_data`), so
the run gets its verdict and the link-windows its changed ids fall in, where reading its 8-byte cells as id events
raised `struct.error` out of the report. A frame neither reading places gives the run no verdict, with the reason.
The runs are the gate's own compact rehearsal (network_traces' `gate_compact`), with `io_tie`'s parts all holding
(`HOLDS`, a stand-in), so the token run is refused by its finish alone.
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


#: what `negatives.tie_run` calls of `io_tie`, every part of its `reference_parts` holding
HOLDS = SimpleNamespace(
    pad_cell=bytes, data_cell=lambda x, q, B: struct.pack(">BII", 1, q, x).ljust(B, b"\0"), parse_cell=_parse,
    port_row=lambda ids, P: b"".join(x.to_bytes(4, "big") for x in ids).ljust(4 * P, b"\0"),
    domain=lambda binding, count: SimpleNamespace(domain_id=binding), fake_salt=lambda seed, i: b"",
    rows_levels=lambda domain_id, rows, salts: [[b""]], rows_root=lambda domain_id, rows, salts: b"",
    bucket_rows=lambda shape, cells: [], open_port=lambda levels, rows, salts, steps: ((), (), ()),
    Shape=lambda B, T, R, P, S, DS: SimpleNamespace(B=B, T=T, R=R, P=P, S=S, DS=DS),
    MapEntry=lambda q, e: (q, e), UNHELD=(0, 0xFFFF_FFFF), Instance=lambda *fields: fields,
    reference_parts=lambda inst: {"cells_root": True, "port": True, "content": True})


@pytest.fixture
def tied(gate_compact, tmp_path, monkeypatch):
    got, endpoint = gate_compact
    ids = tmp_path / "requests.jsonl"
    ids.write_text("".join(json.dumps(r) + "\n" for r in endpoint.served))
    monkeypatch.setattr(NG, "io_tie_module", lambda: HOLDS)
    return got, ids


def _windows_of(run: Path, users: set[str]) -> list[int]:
    """The egress link-windows that carry a compact record of a request of ``users``, read off the records directly."""
    plan, report = (json.loads((run / f).read_text()) for f in ("plan.json", "calibration.json"))
    setup = NG.ar.Setup(report)
    e, i = setup.egress_params, setup.ingress_params
    ins = [NG.ar.unpack_record(b, i.frame_bytes) for b in NG.ar.read_blobs(run / "requests.records")]
    outs = [NG.ar.unpack_record(b, e.frame_bytes) for b in NG.ar.read_blobs(run / "responses.records")]
    streams = {r["stream"] for r in NG.cs.certified_io(plan, setup, ins, outs)["records"] if r["user"] in users}
    return sorted({r.window for r in outs for row in r.grid for s in row
                   if s is not None and active.compact_record_data(s)[0] in streams})


def test_the_compact_token_run_under_its_own_name_gets_its_verdict_and_changed_windows(tied):
    """The token run under its own name: the tie gives it a verdict (refused by its finish, every `io_tie` part holding
    here) and the link-windows its changed ids fall in, which are those of its tampered requests' records."""
    got, ids = tied
    run = got["token"]["dir"]
    changed = json.loads((run / "server.json").read_text())["tampered"]
    assert changed and run.joinpath("plan.json").exists() and got["token"]["plan"]["compact"] == 8
    t = chain.tie({"token": run}, None, ids)
    row = t["runs"]["token"]
    assert row["why"] is None and row["verdict"] == "refused" and row["finish"] is False
    assert row["changed"] == chain.changed_windows(run) == [0]
    assert set(row["changed"]) <= set(_windows_of(run, {u for u, _ in changed}))
    assert row["ok"] is False


def test_a_compact_frame_that_isnt_a_record_gives_the_token_run_no_verdict(tied, monkeypatch):
    """A compact plan's egress frame that isn't a compact record can't be placed: the token row says so, and the
    report gets the row rather than an exception."""
    got, ids = tied
    monkeypatch.setattr(chain, "compact_record_data", lambda cell: None)
    row = chain.tie({"token": got["token"]["dir"]}, None, ids)["runs"]["token"]
    assert row["verdict"] == NG.NOT_APPLICABLE and row["ok"] is None
    assert "a compact plan's egress frame that isn't a compact record" in row["why"]


def test_an_egress_frame_neither_reading_places_is_refused_not_raised():
    """`_output_stream`: a compact record is placed on its ordinal under a compact plan; read as an id event (a plan
    of id events), its 8 bytes don't decode, which is a refusal (ValueError), not `struct.error`."""
    record = active.compact_record(3, 17)
    assert chain._output_stream(record, True) == 3
    with pytest.raises(ValueError, match="doesn't decode"):
        chain._output_stream(record, False)
    for cell in (bytes(8), record + b"\x01"):
        with pytest.raises(ValueError, match="isn't a compact record"):
            chain._output_stream(cell, True)
