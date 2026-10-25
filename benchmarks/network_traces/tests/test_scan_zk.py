"""scan_zk's host side: the run it reads (each link-window's rows the warden's committed ones, each egress bucket
accepting the view its ingress bucket wrote, request 5 in request 1's slot past its egress) and each round's host read
(`scan_zk.member`, as Lean's registered reads make it, and the references' outputs) as the round's: honest accepted
with every output 1, the ingress units' view leaves the egress units' state leaves; chain's swapped buckets' ok bits 0
on reads that hold; forged's egress row refused ("does not open"). A warden_serve run's directory (`--warden-run`,
here the synthetic run published as `warden_serve.Publisher` publishes a served one) is read under its deployment's key,
its registration's ``warden/<side>`` roots (`scan_zk.warden_roots`) the root and domain of each side's row registration
in ``cells.jsonl``, and refused by name where a row, the key or the schedule isn't the warden's. Nothing here lowers or
proves (`scan_zk.prove` does, on a node)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_scan_zk", HERE / "scan_zk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sz = _load()
import warden_serve as WS


def test_each_rounds_host_read_is_the_rounds(tmp_path):
    h = sz.host(tmp_path)
    at, T = h.at, sz.PARAMS.window
    assert at.admitted == {1: 0, 2: 2, 3: 2, 4: 12, 5: 22}
    assert [s.q for s in at.chain.scan[23].slots] == [5, 2, 3, 4]
    honest, chain, forged = (h.rounds[k] for k in ("honest", "chain", "forged"))
    ing, eg = honest
    assert ing.outputs == [1] * T and eg.outputs == [sz.EG.ACCEPT] * T
    assert ing.bcs["view"] == eg.bcs["state"] and ing.bcs["sout"][:-1] == ing.bcs["sin"][1:]
    assert ing.ports["view"]["root"] == eg.ports["state"]["root"] and ing.ports["sin"]["reads"] == list(range(T))
    assert ing.ports["sout"]["reads"] == list(range(1, T + 1)) and eg.ports["state"][sz.REG.READS_FROM] == T
    assert chain[0].refused is None and [t for t, o in enumerate(chain[0].outputs) if not o] == [sz.A, sz.A + 1, sz.A + 2]
    assert chain[0].ports["sin"]["root"] != ing.ports["sin"]["root"] and chain[1].outputs == eg.outputs
    (f,) = forged
    t = f.changed
    assert "does not open" in f.refused and f.outputs == [int(i != t) for i in range(T)] and f.rows[t] != eg.rows[t]


def published(out: Path, params=sz.PARAMS, seed: bytes = b"warden-run") -> Path:
    """The synthetic run's records published as warden_serve's `Publisher` publishes a served run's (each record, its
    certificate, its rows' registration and salts), under a warden key of its own, beside its `deployment.json`."""
    reqs = [sz._request(i, p) for i, p in enumerate(sz.PROMPTS)]
    served = sz.traffic.serve(params, reqs, sz.ARRIVALS, [1 << 20] * len(reqs), sz.RESPONSES, sz.WINDOWS)
    out.mkdir(parents=True)
    signer = sz.LmsSigner.create(sz.lms.keygen(sz.lms.Params(64, 1, 6), hashlib.sha512(seed).digest()),
                                 out.parent / f"{out.name}-signer.json")
    pub = WS.Publisher(out, sz.Issuer(signer, sz.SESSION), params)
    for rec in (*served.ingress, *served.egress):
        pub.keep(rec, params.grid)
    (out / "fails.records").touch()
    dep = WS.deployment_record(params, signer.public_key(), sz.SESSION, origin_unix=0.0, windows=sz.WINDOWS,
                               listen=("127.0.0.1", 1), upstream=("127.0.0.1", 2), slack=0.025, run_id=bytes(32))
    (out / "deployment.json").write_text(json.dumps(dep))
    return out


def test_a_warden_runs_rows_are_read_under_its_deployments_key(tmp_path):
    h = sz.host(tmp_path / "host", published(tmp_path / "run"), draw=16)
    assert set(h.rounds) == {"honest", "forged"} and h.at.run_id == "00" * 32
    assert "warden run" in h.at.source and h.at.admitted == {1: 0, 2: 2, 3: 2, 4: 12, 5: 22}
    assert h.at.outputs == {q: r for q, r in enumerate(sz.RESPONSES, 1)}
    syn = sz.host(tmp_path / "syn")
    assert [list(h.at.sides[s].cells.rows) for s in ("ingress", "egress")] == [
        list(syn.at.sides[s].cells.rows) for s in ("ingress", "egress")]
    ing, eg = h.rounds["honest"]
    assert ing.outputs == [1] * sz.PARAMS.window and eg.outputs == [sz.EG.ACCEPT] * sz.PARAMS.window
    regs = {d["registration"]["direction"]: d["registration"] for d in map(json.loads, (
        tmp_path / "run" / "cells.jsonl").read_text().splitlines()) if d["registration"]["window"] == 0}
    assert {r["port"]: (r["root"], r["domain"], r["leaves"]) for r in sz.warden_roots(h.at)} == {
        f"warden/{d}": (regs[d]["root"], regs[d]["domain"], sz.PARAMS.window) for d in ("ingress", "egress")}


def _rewrite(path: Path, blobs: list[bytes]) -> None:
    path.unlink()
    for b in blobs:
        WS.ar.write_blobs(path, b)


def test_a_forged_cell_row_another_key_or_another_schedule_is_refused_by_name(tmp_path):
    run = published(tmp_path / "run")
    B = sz.PARAMS.cell_bytes
    blobs = WS.ar.read_blobs(run / "responses.records")
    rec = WS.ar.unpack_record(blobs[0], B)
    t, j = next((t, j) for t, row in enumerate(rec.grid) for j, f in enumerate(row) if f is not None and f[0] == sz.DATA)
    frame = bytearray(rec.grid[t][j])
    frame[sz.IT.ID_AT + sz.IT.ID_DIGITS - 1] ^= 1
    grid = [list(row) for row in rec.grid]
    grid[t][j] = bytes(frame)
    _rewrite(run / "responses.records", [WS.ar.pack_record(replace(rec, grid=tuple(tuple(r) for r in grid))),
                                         *blobs[1:]])
    with pytest.raises(SystemExit, match="egress link-window 0's record under cells.salts isn't the rows whose root "
                                         "the warden signed"):
        sz.host(tmp_path / "h1", run)

    run = published(tmp_path / "run2")
    other = published(tmp_path / "other", seed=b"another warden")
    dep = json.loads((run / "deployment.json").read_text())
    dep["key"] = json.loads((other / "deployment.json").read_text())["key"]
    (run / "deployment.json").write_text(json.dumps(dep))
    with pytest.raises(SystemExit, match="ingress link-window 0's registration doesn't verify under the deployment's "
                                         "key and session"):
        sz.host(tmp_path / "h2", run)

    dep = json.loads((other / "deployment.json").read_text())
    dep["params"] = asdict(replace(sz.PARAMS, delay=20))
    dep["digest"] = replace(sz.PARAMS, delay=20).digest.hex()
    (other / "deployment.json").write_text(json.dumps(dep))
    with pytest.raises(SystemExit, match="warden_serve.py rehearse at scan_zk's schedule"):
        sz.host(tmp_path / "h3", other)
