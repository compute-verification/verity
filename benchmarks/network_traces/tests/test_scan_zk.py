"""scan_zk's host side: the run it reads (each link-window's rows the warden's committed ones, each egress bucket
computing the warden's row from the view its ingress bucket wrote, w63's request 5 in request 1's slot past its egress)
and each round's host read (`scan_zk.member`, as Lean's registered reads make it, and the references' outputs) as the
round's: honest accepted with every output 1, its state and view chains wires through the registered roots
(`scan_zk.wired`); chain's ingress member refused at unit A's o1 and tampered's egress member at the re-signed row
("does not open"). A group of link-windows (IO-TIE's) is one registration, a member a template and link-window, its
members the Program's ranges. A warden_serve run's directory (`--warden-run`, here the synthetic run published as
`warden_serve.Publisher` publishes a served one) is read under its deployment's key, and refused by name where a row,
the key or the schedule isn't the warden's. The honest round on the challenger's coins (``--challenger``): its flags
refused by name where they can't draw, the group's root bound to the warden's signed link-windows, and its Spec what the
round's registration shows (`service.check_public`) over the units its `Stream` takes. Nothing here lowers or proves
(`scan_zk.prove` does, on a node)."""

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
    assert [m.name for m in h.rounds["chain"] if m.refused] == sz.targets(at, "chain") == ["ingress-0"]
    assert [m.name for m in h.rounds["tampered"] if m.refused] == sz.targets(at, "tampered")
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


def published(out: Path, setup=sz.W63, seed: bytes = b"warden-run", windows: int | None = None,
              late: tuple[int, ...] = ()) -> Path:
    """The synthetic run's records over ``windows`` link-windows (the setup's by default) published as warden_serve's
    `Publisher` publishes a served run's (each record and its certificate, the wire's rows' registration and salts),
    under a warden key of its own, beside its `deployment.json`; the connections of ``late`` failed late (a void run,
    their reasons in ``fails.records``)."""
    windows = windows or setup.windows
    reqs = [sz._request(setup, i, p) for i, p in enumerate(setup.prompts)]
    served = sz.traffic.serve(setup.params, reqs, setup.arrivals, [1 << 20] * len(reqs), setup.responses, windows,
                              late)
    out.mkdir(parents=True)
    signer = sz.LmsSigner.create(sz.lms.keygen(sz.lms.Params(64, 1, 6), hashlib.sha512(seed).digest()),
                                 out.parent / f"{out.name}-signer.json")
    pub = WS.Publisher(out, sz.Issuer(signer, sz.SESSION), setup.params)
    for w in range(windows):
        for rec, grid in ((served.ingress[w], setup.params.grid), (served.egress[w], setup.params.grid),
                          (served.failed[w], setup.params.fails_grid)):
            pub.keep(rec, grid)
    dep = WS.deployment_record(setup.params, signer.public_key(), sz.SESSION, origin_unix=0.0, windows=windows,
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


def _cells(run: Path) -> dict[tuple[str, int], dict]:
    """A run's row registrations in ``cells.jsonl``, by direction and link-window."""
    return {(d["registration"]["direction"], d["registration"]["window"]): d["registration"]
            for d in map(json.loads, (run / "cells.jsonl").read_text().splitlines()) if "registration" in d}


def test_the_registration_names_each_link_windows_warden_rows_its_ports_read(tmp_path):
    """The rows form's hold (`NetworkCheck.Period`: the scan's registration names each of W's row registrations' root
    and domain): a G = 2 group's honest round names, as ``warden/<direction>/<w>``, each of the four row registrations
    of the run's ``cells.jsonl``, T leaves each (ingress's row ports and egress's o1 read them); the tampered round,
    egress only, names the egress rows alone, the re-signed link-window's in its place."""
    G, T = 2, sz.TEST.params.window
    run = published(tmp_path / "run", sz.TEST, windows=G)
    h = sz.host(tmp_path / "host", run, setup=sz.TEST, group=G)
    regs = _cells(run)
    assert sz.warden_roots([{"warden": m.warden} for m in h.rounds["honest"]]) == [
        {"port": f"warden/{d}/{w}", "root": regs[d, w]["root"], "leaves": T, "domain": regs[d, w]["domain"]}
        for d in ("egress", "ingress") for w in range(G)]
    tampered, w = sz.warden_roots([{"warden": m.warden} for m in h.rounds["tampered"]]), h.at.tampered.window
    assert [r["port"] for r in tampered] == [f"warden/egress/{x}" for x in range(G)]
    assert tampered[w]["root"] == h.at.tampered.registration["root"] != regs["egress", w]["root"]
    assert all(tampered[x]["root"] == regs["egress", x]["root"] for x in range(G) if x != w)


def test_each_link_windows_answer_is_one_stages_on_the_groups_registration():
    """`window_answer` as the rows form reads a link-window's scan answer (`pouw.core_check.answer_of` and `integrity`,
    the period mirror's own reads): every unit of a G = 2 group drawn, one level drawn stratified over the three
    templates, its bound 0, for the group's registration; a refused audit a reject of the same registration naming
    why; an accepted record of another registration the driver's error."""
    from verity.core.protocols.pouw import core_check as CC

    G, T = 2, 4
    defs = {"first": FIRST, "ingress": ING, "egress": EG}
    counts = sz.counts_of(sz.members_of(sz.TEMPLATES, G, T))
    program = sz.PA.mixed_population_program([(defs[t], counts[t]) for t in sz.TEMPLATES])
    query = sz.PA.template_instances_query([defs[t] for t in sz.TEMPLATES])
    N, law = sum(counts.values()), sz.DR.parse_law(f"stratified:{2 * G * T}")
    reg = R.record(program, query, law, sz.S.SCHEME,
                   [{"port": "warden/egress/1", "root": "aa" * 64, "leaves": T, "domain": "bb" * 64}],
                   {"kind": "stand-in"}, leaf_layer="ee" * 64)
    draw = {**sz.DR.derive(program, query, law), "population": N, "units": list(range(N))}
    record = {"accepted": True, "registration": reg, "draw": draw, "proved_units": list(range(N))}
    a = sz.window_answer(reg, record, sz.Fraction(1, 100), "unused")
    assert CC.answer_of(a)["verdict"] == "accept" and a["commitment"] == R.digest(reg) and a["protocol"] == R.PROTOCOL
    assert a["public"]["bound"] == 0 and a["public"]["drawn"] == a["public"]["proved"] == list(range(N))
    (level,) = CC.integrity(a["profile"])["levels"]
    assert level["draw"]["law"] == "stratified" and sum(s["k"] for s in level["draw"]["strata"]) == N == level["size"]
    why = "the scan's audit of link-windows [0, 1] refused: egress-1: registered reads: table row 3 does not open"
    r = sz.window_answer(reg, None, sz.Fraction(1, 100), why)
    assert CC.answer_of(r)["verdict"] == "reject" and r["reason"] == why and r["commitment"] == a["commitment"]
    with pytest.raises(SystemExit, match="another registration"):
        sz.window_answer(reg, {**record, "registration": {**reg, "population": 1}}, sz.Fraction(1, 100), "")


def test_a_void_run_is_its_honest_round_refused_at_the_egress_member_of_its_fail(tmp_path):
    """A warden run whose warden failed a connection late (its server sent nothing: a FAIL in the egress row the
    warden signed, its reason in ``fails.records``) is read, not refused: the scan renders the connection's END where
    the warden wrote FAIL, that bucket alone differs from the warden's row, and the run's one round, honest, is refused
    ("does not open") at the egress member of the FAIL's link-window, every other member held."""
    G, T = 2, sz.TEST.params.window
    h = sz.host(tmp_path / "host", published(tmp_path / "run", sz.TEST, windows=G, late=(1,)), setup=sz.TEST, group=G)
    at = h.at
    (b,) = at.void
    assert at.failed == 1 and at.tampered is None and 2 not in at.outputs
    assert set(h.rounds) == {"honest"} and sz.rounds_of(at) == {"honest": (sz.TEMPLATES, "does not open")}
    refused = {m.name: m.refused for m in h.rounds["honest"] if m.refused}
    assert list(refused) == sz.targets(at, "honest") == [f"egress-{b // T}"]
    assert refused[f"egress-{b // T}"] == (f"registered reads: table row {b % T} does not open the registered root at "
                                           f"position {b % T}")
    assert sz.wired(h.rounds["honest"]) == []


def test_the_rows_forms_case_is_each_link_windows_rows_certificates_and_the_scans_answer(tmp_path):
    """`scan_period.rows_case` of a G = 2 group in the rows form's formats (`NetworkCheck.Period`, whose mirror and
    Lean, on the rows form's branch, check it on a node: scan_period.sh): the deployment's key, session and schedule,
    the scan's Program and layout terms, the group's registration for each link-window, and each one's answer, its
    three records' certificates (requests, responses, fails: its own) and its two row registrations, each named by the
    registration's roots; the tampered case the tampered round's registration with its refusal as each answer."""
    import scan_period as SP

    G = 2
    run = published(tmp_path / "run", sz.TEST, windows=G)
    h = sz.host(tmp_path / "host", run, setup=sz.TEST, group=G)
    T, scan = h.at.T, tmp_path / "scan"
    (scan / "tampered").mkdir(parents=True)
    counts = sz.counts_of(sz.members_of(sz.TEMPLATES, G, T))
    defs = {"first": FIRST, "ingress": ING, "egress": EG}
    program = sz.PA.mixed_population_program([(defs[t], counts[t]) for t in sz.TEMPLATES])
    query = sz.PA.template_instances_query([defs[t] for t in sz.TEMPLATES])
    roots = sz.warden_roots([{"warden": m.warden} for m in h.rounds["honest"]])
    reg = R.record(program, query, sz.DR.parse_law(f"stratified:{2 * G * T}"), sz.S.SCHEME, roots,
                   {"kind": "stand-in"}, leaf_layer="ee" * 64)
    other = {**reg, "program": "ab" * 64}
    (scan / "tampered" / "registration.json").write_text(json.dumps(other))
    answer = sz.window_answer(reg, None, None, "the scan refused")
    (scan / "answers.json").write_text(json.dumps({"registration": reg, "windows": [
        {"window": w, "answer": answer, "registrations": [h.at.sides[s][w].registration for s in ("ingress", "egress")]}
        for w in range(G)]}))
    bits = {f"{tp}/{r}": b for tp, (ins, outs) in sz.ports_of(sz.TEST.cls).items() for r, b in ins + outs}
    (scan / "summary.json").write_text(json.dumps({"T": T, "bits": bits,
                                                   "rounds": {"tampered": {"audit": {"why": {"x": "y"}}}}}))
    dep, regs = json.loads((run / "deployment.json").read_text()), _cells(run)
    case = SP.rows_case(run, scan)
    p = case["params"]
    assert sorted(p) == ["audits", "cells", "first", "jitter", "key", "link", "lms", "program", "scan", "schedule",
                         "session", "windows"]
    assert (p["key"], p["session"], p["schedule"], p["link"]) == (dep["key"]["public"], dep["session"], dep["params"],
                                                                  WS.LINK)
    assert (p["program"], p["windows"], p["first"], p["audits"]) == (reg["program"], G, 0, [])
    assert p["scan"] == {"free": T * sum(b for k, b in bits.items() if not k.endswith("/row")), "steps": T}
    assert json.loads(bytes.fromhex(case["commitment"])) == {"audits": [], "windows": [reg] * G}
    ev = json.loads(bytes.fromhex(case["evidence"]))
    for w, x in enumerate(ev["windows"]):
        assert x["answer"] == answer and x["registrations"] == [regs["ingress", w], regs["egress", w]]
        assert [(c["link"], c["direction"], c["window"]) for c in x["certificates"]] == [
            ("warden", "ingress", w), ("warden", "egress", w), ("warden-fails", "egress", w)]
        assert all({"root": d["root"], "domain": d["domain"]} in [{"root": r["root"], "domain": r["domain"]}
                                                                   for r in reg["roots"]] for d in x["registrations"])
    t = SP.rows_case(run, scan, "tampered")
    assert json.loads(bytes.fromhex(t["commitment"]))["windows"] == [other] * G and t["params"] == p
    assert {(x["answer"]["verdict"], x["answer"]["commitment"]) for x in json.loads(bytes.fromhex(t["evidence"]))[
        "windows"]} == {("reject", R.digest(other))}


def test_schedule_prints_the_classs_schedule_as_warden_serves_flags(capsys):
    assert sz.main(["schedule", "--class", "test"]) == 0
    flags = capsys.readouterr().out.split()
    assert dict(zip(flags[::2], flags[1::2])) == {f"--{k.replace('_', '-')}": str(v)
                                                  for k, v in asdict(sz.TEST.params).items()}


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
