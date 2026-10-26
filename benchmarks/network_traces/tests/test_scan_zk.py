"""scan_zk's host side: the run it reads (each link-window's rows the warden's committed ones, each egress bucket
computing the warden's row from the view its ingress bucket wrote, w63's request 5 in request 1's slot past its egress)
and each round's host read (`scan_zk.member`, as Lean's registered reads make it, and the references' outputs) as the
round's: honest accepted with every output 1, its state and view chains wires through the registered roots
(`scan_zk.wired`); chain's ingress member refused at unit A's o1 and tampered's egress member at the re-signed row
("does not open"). A group of link-windows (IO-TIE's) is one registration, a member a template and link-window, its
members the Program's ranges. A warden_serve run's directory (`--warden-run`, here the synthetic run published as
`warden_serve.Publisher` publishes a served one) is read under its deployment's key, its registration's
``rows/<direction>/<w>`` roots (`scan_zk.warden_roots`) the root and domain of each row registration in ``cells.jsonl``,
and refused by name where a row, the key or the schedule isn't the warden's. The honest round on the challenger's coins
(``--challenger``): its flags refused by name where they can't draw, the group's root bound to the warden's signed
link-windows, and its Spec what the round's registration shows (`service.check_public`) over the units its `Stream`
takes. Nothing here lowers or proves (`scan_zk.prove` does, on a node)."""

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
    at, T, A = h.at, sz.PARAMS.window, sz.W63.A
    assert at.admitted == {1: 0, 2: 2, 3: 2, 4: 12, 5: 22}
    assert [s.q for s in at.chain.scan[23].slots] == [5, 2, 3, 4]
    honest, chain, tampered = ({m.name: m for m in h.rounds[k]} for k in ("honest", "chain", "tampered"))
    first, ing, eg = (honest[k] for k in ("first", "ingress-0", "egress-0"))
    assert [(m.base, m.n) for m in (first, ing, eg)] == [(0, 1), (1, T - 1), (T, T)]
    assert all(m.refused is None and m.outputs == [1] * m.n for m in honest.values())
    assert sz.wired(honest.values()) == []
    assert first.bcs["o1"] + ing.bcs["o1"][:-1] == ing.bcs["sin"] and first.bcs["o2"] + ing.bcs["o2"] == eg.bcs["state"]
    assert ing.ports["o1"]["root"] == ing.ports["sin"]["root"] and ing.ports["o2"]["root"] == eg.ports["state"]["root"]
    assert first.ports["row"]["reads"] == [0] and sz.REG.READS_FROM not in first.ports["row"]
    assert ing.ports["row"]["reads"] == list(range(1, T)) and ing.ports["row"][sz.REG.READS_FROM] == 1
    side = at.sides["egress"][0]
    assert eg.ports["o1"]["reads"] == list(range(T)) and eg.ports["o1"][sz.REG.READS_FROM] == T
    assert eg.ports["o1"]["root"] == side.head.root.hex() and eg.bcs["o1"] == list(side.cells.bcs)
    c = chain["ingress-0"]
    assert c.refused == f"registered reads: table row {A - 1} does not open the registered root at position {A}"
    assert c.ports["o1"]["root"] == c.ports["sin"]["root"] != ing.ports["o1"]["root"]
    assert all(m.refused is None for k, m in chain.items() if k != "ingress-0")
    (f,) = tampered.values()
    b = at.tampered.bucket
    assert f.changed == b
    assert f.refused == f"registered reads: table row {b} does not open the registered root at position {b}"
    assert f.outputs == [1] * T and f.made == eg.made and at.tampered.row != eg.made["o1"][b]
    assert f.ports["o1"]["root"] == at.tampered.head.root.hex() != eg.ports["o1"]["root"] and f.base == 0


def test_a_group_of_link_windows_is_one_registration_a_member_a_window(tmp_path):
    """IO-TIE's group (``--group G``) at the test class: link-windows 0 and 1 in one registration, the trees G·T
    leaves, each member's warden port its own link-window's registration, and the chains running across the windows."""
    G, T = 2, sz.TEST.params.window
    h = sz.host(tmp_path, setup=sz.TEST, group=G)
    at = h.at
    assert at.windows == [0, 1] and all(len(r.rows) == G * T for r in at.trees.values())
    honest = {m.name: m for m in h.rounds["honest"]}
    assert [(k, m.base, m.n) for k, m in honest.items()] == [
        ("first", 0, 1), ("ingress-0", 1, T - 1), ("ingress-1", T, T), ("egress-0", G * T, T), ("egress-1", G * T + T, T)]
    assert sz.wired(honest.values()) == [] and all(m.refused is None and set(m.outputs) == {1} for m in honest.values())
    ing1, eg1 = honest["ingress-1"], honest["egress-1"]
    assert ing1.ports["row"]["root"] == at.sides["ingress"][1].head.root.hex() != honest["ingress-0"].ports["row"]["root"]
    assert ing1.ports["row"]["reads"] == list(range(T)) and ing1.ports["row"][sz.REG.READS_FROM] == T
    assert ing1.ports["sin"]["reads"] == list(range(T - 1, G * T - 1))
    assert eg1.ports["o1"]["root"] == at.sides["egress"][1].head.root.hex() and eg1.ports["o1"][sz.REG.READS_FROM] == 3 * T
    assert [m.name for m in h.rounds["chain"] if m.refused] == [sz.target(at, "chain")] == ["ingress-0"]
    assert [m.name for m in h.rounds["tampered"] if m.refused] == [sz.target(at, "tampered")]
    with pytest.raises(SystemExit, match="a group is 1 or more link-windows"):
        sz.link(tmp_path / "none", setup=sz.TEST, group=0)


XOR8 = PrimitiveDefinition("ScanZkTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)


def _toy(name: str, f) -> object:
    return bind(CompositeDefinition(name, 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)), f, register=False))


FIRST = _toy("ScanZkTestFirst", lambda B, S_, x: B.call(XOR8, x[0], x[0]))
ING = _toy("ScanZkTestIngress", lambda B, S_, x: B.call(XOR8, x[0], x[1]))
EG = _toy("ScanZkTestEgress", lambda B, S_, x: B.call(XOR8, x[1], x[0]))


@pytest.mark.parametrize("G", [1, 3])
def test_the_members_are_the_programs_ranges(G):
    """`members_of` against the partition's own ranges (stand-in templates; `program` asserts the same of the scan's):
    each template's first unit its first member's base and its count its members', the units 0 … N − 1 in order, the
    ingress units computing buckets 1 … G·T − 1 and the egress units 0 … G·T − 1."""
    T, defs = 4, {"first": FIRST, "ingress": ING, "egress": EG}
    for templates in {t for t, _o in sz.ROUNDS.values()}:
        members = sz.members_of(templates, G, T)
        counts = sz.counts_of(members)
        program = sz.PA.mixed_population_program([(defs[t], counts[t]) for t in templates])
        ranges = sz.PA.ranges(program, sz.PA.template_instances_query([defs[t] for t in templates]))
        assert [(lo, n) for _t, lo, n in ranges] == [
            (next(m.base for m in members if m.template == t), counts[t]) for t in templates]
        units = [(m.base + k, m.template, b) for m in members for k, b in enumerate(m.buckets)]
        assert [u for u, _t, _b in units] == list(range(sum(counts.values())))
        assert sorted(b for _u, t, b in units if t == "egress") == list(range(G * T))
        assert sorted(b for _u, t, b in units if t != "egress") == (list(range(G * T)) if "first" in templates else [])


def published(out: Path, setup=sz.W63, seed: bytes = b"warden-run") -> Path:
    """The synthetic run's records published as warden_serve's `Publisher` publishes a served run's (each record, its
    certificate, its rows' registration and salts), under a warden key of its own, beside its `deployment.json`."""
    reqs = [sz._request(setup, i, p) for i, p in enumerate(setup.prompts)]
    served = sz.traffic.serve(setup.params, reqs, setup.arrivals, [1 << 20] * len(reqs), setup.responses, setup.windows)
    out.mkdir(parents=True)
    signer = sz.LmsSigner.create(sz.lms.keygen(sz.lms.Params(64, 1, 6), hashlib.sha512(seed).digest()),
                                 out.parent / f"{out.name}-signer.json")
    pub = WS.Publisher(out, sz.Issuer(signer, sz.SESSION), setup.params)
    for rec in (*served.ingress, *served.egress):
        pub.keep(rec, setup.params.grid)
    (out / "fails.records").touch()
    dep = WS.deployment_record(setup.params, signer.public_key(), sz.SESSION, origin_unix=0.0, windows=setup.windows,
                               listen=("127.0.0.1", 1), upstream=("127.0.0.1", 2), slack=0.025, run_id=bytes(32))
    (out / "deployment.json").write_text(json.dumps(dep))
    return out


def test_a_warden_runs_rows_are_read_under_its_deployments_key(tmp_path):
    h = sz.host(tmp_path / "host", published(tmp_path / "run"), draw=16)
    assert set(h.rounds) == {"honest", "chain", "tampered"} and h.at.run_id == "00" * 32
    assert "warden run" in h.at.source and h.at.admitted == {1: 0, 2: 2, 3: 2, 4: 12, 5: 22}
    assert h.at.outputs == {q: r for q, r in enumerate(sz.W63.responses, 1)}
    assert h.at.tampered.key_of == "the driver's own"
    syn = sz.host(tmp_path / "syn")
    assert [list(h.at.sides[s][0].cells.rows) for s in ("ingress", "egress")] == [
        list(syn.at.sides[s][0].cells.rows) for s in ("ingress", "egress")]
    assert all(m.refused is None and set(m.outputs) == {1} for m in h.rounds["honest"])
    regs = {d["registration"]["direction"]: d["registration"] for d in map(json.loads, (
        tmp_path / "run" / "cells.jsonl").read_text().splitlines()) if d["registration"]["window"] == 0}
    assert {r["port"]: (r["root"], r["domain"], r["leaves"]) for r in sz.warden_roots(
        [{"warden": m.warden} for m in h.rounds["honest"]])} == {
        f"rows/{d}/0": (regs[d]["root"], regs[d]["domain"], sz.PARAMS.window) for d in ("ingress", "egress")}


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
    (["--group", "0"], "--group is 1 or more link-windows"),
])
def test_a_challenger_draw_that_cant_draw_is_refused_by_name_before_anything_runs(tmp_path, capsys, flags, why):
    with pytest.raises(SystemExit):
        sz.main(["prove", "--dry", "--out", str(tmp_path / "out"), *flags])
    assert re.search(why, capsys.readouterr().err) and not (tmp_path / "out").exists()


def test_the_window_root_is_the_wardens_signed_link_windows(tmp_path):
    """Each run's root is its own (the synthetic run's registrations carry fresh salts), a warden run's is read from its
    published registrations alone, and a group's covers each of its link-windows."""
    syn = sz.link(tmp_path / "syn")
    root = sz.window_root(syn)
    assert re.fullmatch("[0-9a-f]{128}", root) and root != sz.window_root(sz.link(tmp_path / "again"))
    w = published(tmp_path / "w")
    run = sz.window_root(sz.link(tmp_path / "run", w))
    assert run == sz.window_root(sz.link(tmp_path / "rerun", w)) != root
    assert sz.window_root(SimpleNamespace(session=bytes(32), sides=syn.sides)) != root
    other = {**syn.sides, "egress": [SimpleNamespace(registration={**syn.sides["egress"][0].registration, "root": "00"})]}
    assert sz.window_root(SimpleNamespace(session=syn.session, sides=other)) != root
    two = sz.link(tmp_path / "two", setup=sz.TEST, group=2)
    assert sz.window_root(two) != sz.window_root(SimpleNamespace(session=two.session,
                                                                 sides={s: x[:1] for s, x in two.sides.items()}))


def test_the_honest_rounds_spec_is_what_its_registration_shows_over_the_units_its_stream_takes():
    """`spec_of` on a round of two stand-in templates split a member a link-window (the scan's own take a minute to
    build; `spec_of` reads only the round's program, query, templates and members), its ports a member's and the scan's
    trees as `audit` names them, its counts each template's over its members."""
    T, k, root = 32, 16, "dd" * 64
    defs = {"ingress": ING, "egress": EG}
    templates = ("ingress", "egress")
    program = sz.PA.mixed_population_program([(defs[s], 2 * T) for s in templates])
    query = sz.PA.template_instances_query([defs[s] for s in templates])
    pg = SimpleNamespace(program=program, query=query, templates=templates,
                         stagers={s: SimpleNamespace(defn=defs[s]) for s in templates})
    members = [{"name": f"{s}-{w}", "n": T, "defn": defs[s]} for s in templates for w in range(2)]
    want = {"ingress-0/row": {"leaves": T, "domain": "aa" * 64}, "egress-1/o0": {"leaves": T, "domain": "bb" * 64},
            "scan/state": {"leaves": 2 * T, "domain": "cc" * 64}}
    schemas = {p: sz.BitRowLeaf.hiding_schema for p in want}
    spec, units, counts = sz.spec_of(pg, members, want, schemas, k, root)
    assert units == [ING, EG] and counts == {sz.PA.template_id(ING): 2 * T, sz.PA.template_id(EG): 2 * T}
    assert spec.context == bytes.fromhex(root) and spec.selection.to_json() == {"rule": "law", "law": f"stratified:{k}"}
    assert spec.check.proof_deadline == 4 * sz.MEMBER_DEADLINE_S

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
