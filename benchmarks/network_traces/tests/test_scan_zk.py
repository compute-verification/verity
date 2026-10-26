"""scan_zk's host side: the run it reads (each link-window's rows the warden's committed ones, each egress bucket
accepting the view its ingress bucket wrote, request 5 in request 1's slot past its egress) and each round's host read
(`scan_zk.member`, as Lean's registered reads make it, and the references' outputs) as the round's: honest accepted
with every output 1, the ingress units' view leaves the egress units' state leaves; chain's swapped buckets' ok bits 0
on reads that hold; forged's egress row refused ("does not open"). A warden_serve run's directory (`--warden-run`,
here the synthetic run published as `warden_serve.Publisher` publishes a served one) is read under its deployment's key,
its registration's ``warden/<side>`` roots (`scan_zk.warden_roots`) the root and domain of each side's row registration
in ``cells.jsonl``, and refused by name where a row, the key or the schedule isn't the warden's. The honest round on the
challenger's coins (``--challenger``): its flags refused by name where they can't draw, the window's root bound to the
warden's signed link-window, and its Spec what the round's registration shows (`service.check_public`) over the units
its `Stream` takes. Nothing here lowers or proves (`scan_zk.prove` does, on a node)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.types import Array, Value as Bits
from verity.core.protocols.sampled_proofs.one_stage import registration as R

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
    certificate, its rows' registration and salts, and each window's empty fails record), under a warden key of its
    own, beside its `deployment.json`."""
    reqs = [sz._request(i, p) for i, p in enumerate(sz.PROMPTS)]
    served = sz.traffic.serve(params, reqs, sz.ARRIVALS, [1 << 20] * len(reqs), sz.RESPONSES, sz.WINDOWS)
    out.mkdir(parents=True)
    signer = sz.LmsSigner.create(sz.lms.keygen(sz.lms.Params(64, 1, 6), hashlib.sha512(seed).digest()),
                                 out.parent / f"{out.name}-signer.json")
    pub = WS.Publisher(out, sz.Issuer(signer, sz.SESSION), params)
    for rec in (*served.ingress, *served.egress):
        pub.keep(rec, params.grid)
    for w in range(sz.WINDOWS):
        pub.keep(WS.Record(WS.FAILS, w, ((None,) * params.lanes,) * params.window, WS.Status()), params.fails_grid)
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


def test_a_run_whose_warden_failed_a_connection_closed_is_refused_as_void_by_name(tmp_path):
    run = published(tmp_path / "run")
    rec = WS.ar.unpack_record(WS.ar.read_blobs(run / "responses.records")[0], sz.PARAMS.cell_bytes)
    t, j = next((t, j) for t, row in enumerate(rec.grid) for j, f in enumerate(row) if f is not None)
    grid = tuple(tuple(f if (u, k) == (t, j) else None for k, f in enumerate(row)) for u, row in enumerate(rec.grid))
    _rewrite(run / "fails.records", [WS.ar.pack_record(replace(rec, grid=grid, window=3))])
    (run / "check.json").write_text(json.dumps({"problems": ["audit: connection 7 failed closed (late) at egress cell 2",
                                                             "connection 7: fail, HTTP None, complete False"]}))
    with pytest.raises(SystemExit, match=re.escape("void run: the warden failed connections closed, 1 fail cells in "
                                                   "link-windows [3] (connection 7 failed closed (late) at egress "
                                                   "cell 2)")):
        sz.host(tmp_path / "h", run)


@pytest.mark.parametrize("flags,why", [
    (["--challenger", "local", "--main", "m.txt"], "give --draw"),
    (["--challenger", "remote", "--draw", "16", "--main", "m.txt"], "local or gate:USER@HOST, not 'remote'"),
    (["--challenger", "gate:research@h", "--draw", "16", "--main", "m.txt", "--peer", "p"],
     "needs --peer and --challenger-key"),
    (["--challenger", "local", "--draw", "16"], "needs --main"),
])
def test_a_challenger_draw_that_cant_draw_is_refused_by_name_before_anything_runs(tmp_path, capsys, flags, why):
    with pytest.raises(SystemExit):
        sz.main(["prove", "--dry", "--out", str(tmp_path / "out"), *flags])
    assert re.search(why, capsys.readouterr().err) and not (tmp_path / "out").exists()


def test_the_window_root_is_the_wardens_signed_link_window(tmp_path):
    """Each run's root is its own (the synthetic run's registrations carry fresh salts), and a warden run's is read
    from its published registrations alone."""
    syn = sz.link(tmp_path / "syn")
    root = sz.window_root(syn)
    assert re.fullmatch("[0-9a-f]{128}", root) and root != sz.window_root(sz.link(tmp_path / "again"))
    w = published(tmp_path / "w")
    run = sz.window_root(sz.link(tmp_path / "run", w))
    assert run == sz.window_root(sz.link(tmp_path / "rerun", w)) != root
    assert sz.window_root(SimpleNamespace(session=bytes(32), sides=syn.sides)) != root
    other = {**syn.sides, "egress": SimpleNamespace(registration={**syn.sides["egress"].registration, "root": "00"})}
    assert sz.window_root(SimpleNamespace(session=syn.session, sides=other)) != root


XOR8 = PrimitiveDefinition("ScanZkTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)
ING = bind(CompositeDefinition("ScanZkTestIngress", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                               lambda B, S_, x: B.call(XOR8, x[0], x[1]), register=False))
EG = bind(CompositeDefinition("ScanZkTestEgress", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                              lambda B, S_, x: B.call(XOR8, x[1], x[0]), register=False))


def test_the_honest_rounds_spec_is_what_its_registration_shows_over_the_units_its_stream_takes():
    """`spec_of` on a round of two stand-in templates (the scan's own take a minute to build; `spec_of` reads only the
    round's program, query, templates and counts), its ports a member's and the scan's trees as `audit` names them."""
    T, k, root = 32, 16, "dd" * 64
    defs = {"ingress": ING, "egress": EG}
    sides = ("ingress", "egress")
    program = sz.PA.mixed_population_program([(defs[s], T) for s in sides])
    query = sz.PA.template_instances_query([defs[s] for s in sides])
    pg = SimpleNamespace(program=program, query=query, sides=sides,
                         stagers={s: SimpleNamespace(defn=defs[s]) for s in sides})
    members = [{"name": s, "n": T, "defn": defs[s]} for s in sides]
    want = {"ingress/row": {"leaves": T, "domain": "aa" * 64}, "egress/o0": {"leaves": T, "domain": "bb" * 64},
            "scan/state": {"leaves": T + 1, "domain": "cc" * 64}}
    schemas = {p: sz.BitRowLeaf.hiding_schema for p in want}
    spec, units, counts = sz.spec_of(pg, members, want, schemas, k, root)
    assert units == [ING, EG] and counts == {sz.PA.template_id(ING): T, sz.PA.template_id(EG): T}
    assert spec.context == bytes.fromhex(root) and spec.selection.to_json() == {"rule": "law", "law": f"stratified:{k}"}
    assert spec.check.proof_deadline == 2 * sz.MEMBER_DEADLINE_S

    def record(ports: dict) -> dict:
        roots = [{"port": p, "root": f"{i + 1:02x}" * 64, **w} for i, (p, w) in enumerate(ports.items())]
        return R.record(program, query, sz.DR.parse_law(f"stratified:{k}"), sz.S.SCHEME, roots,
                        {"kind": "stand-in"}, leaf_layer="ee" * 64)

    assert sz.S.check_public(spec, record(want)).ok
    v = sz.S.check_public(spec, record({**want, "scan/other": {"leaves": 1, "domain": "ff" * 64}}))
    assert not v.ok and "roots" in v.codes
    stream = sz.S.Stream(spec, units, budget=1, delta=2.0 ** -20, challenger=sz.CHALLENGER, coins=lambda n: bytes(n))
    assert stream.coin_source == "verifier"
    with pytest.raises(ValueError, match="not the check's program"):
        sz.S.Stream(spec, units[::-1], budget=1, delta=2.0 ** -20, challenger=sz.CHALLENGER)
