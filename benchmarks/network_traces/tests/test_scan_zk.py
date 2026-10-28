"""scan_zk's host side: the run it reads (each link-window's rows the warden's committed ones, each egress bucket
computing the warden's row from the view its ingress bucket wrote, w63's request 5 in request 1's slot past its egress)
and each round's host read (`scan_zk.member`, as Lean's registered reads make it, and the references' outputs) as the
round's: honest accepted with every output 1, its state and view chains wires through the registered roots
(`scan_zk.wired`), its state tree the window state a slot's Program reads (`ingress_scan.window_state`); chain's
ingress member refused at unit A's o1, as is any state tree that skips, repeats or alters a transition, and
tampered's egress member at the re-signed row ("does not open"); unread's one ingress unit, at a request's anchor
with its slot entry unread, every read held and
its output 0, its state and view the honest ones. A group of link-windows (IO-TIE's) is one registration, a member a
template and link-window, its members the Program's ranges. A warden_serve run's directory (`--warden-run`, here the
synthetic run published as `warden_serve.Publisher` publishes a served one) is read under its deployment's key, its
registration's ``rows/<direction>/<w>`` roots (`scan_zk.warden_roots`) the root and domain of each row registration in
``cells.jsonl``, and refused by name where a row, the key or the schedule isn't the warden's. Every round on a
challenger session of its own (``--challenger``): flags that can't name one refused by name before anything runs, each
round's root bound to the warden's signed link-windows, the step and the round, and the honest round's Spec what its
registration shows (`service.check_public`) over the units its `Stream` takes. w63k4, w63's class at a window circuit's
KO 4 and KI 32, serves w63's network certifier at egress_cells 5 and reads its rounds as w63's. Nothing here lowers or
proves (`scan_zk.prove` does, on a node)."""

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

from verity.ml.circuits.codec import canonical_json, encode_program
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.types import Array, Value as Bits
from verity.core.protocols.sampled_proofs.one_stage import registration as R

HERE = Path(__file__).resolve().parents[1]
PERIOD_CASE = HERE.parents[1] / "verity" / "core" / "protocols" / "network" / "tests" / "period_case.py"


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
    assert at.admitted == {1: 0, 2: 2, 3: 2, 4: 12, 5: 28}
    assert [s.q for s in at.chain.scan[29].slots] == [5, 2, 3, 4]
    honest, chain, tampered = ({m.name: m for m in h.rounds[k]} for k in ("honest", "chain", "tampered"))
    first, ing, last, eg = (honest[k] for k in ("first", "ingress-0", "last", "egress-0"))
    assert list(honest) == ["first", "ingress-0", "last", "egress-0"]
    assert [(m.base, m.n) for m in (first, ing, last, eg)] == [(0, 1), (1, T - 2), (T - 1, 1), (T, T)]
    assert all(m.refused is None and m.outputs == [1] * m.n for m in honest.values())
    assert sz.wired(honest.values()) == []
    # the state tree is the window's SCAN_STATE: leaf 0 the empty state, leaf t S_{t−1}; bucket t's sin reads leaf t
    # and its o1 opens leaf t + 1, the last bucket's S_{T−1} no leaf
    st = at.trees["state"]
    assert [sz._bits(st, t) for t in range(T)] == sz.IS.window_state(sz.CLASS, at.chain.states[1:], 0, T)
    assert sz._bits(st, 0) == sz.IS.encode(sz.CLASS, sz.IS.empty(sz.CLASS)) and len(st.rows) == T
    assert first.ports["o1"]["reads"] == [1] and "sin" not in first.ports
    assert ing.ports["sin"]["reads"] == list(range(1, T - 1)) and ing.ports["o1"]["reads"] == list(range(2, T))
    assert last.ports["sin"]["reads"] == [T - 1] and "o1" not in last.ports and "o1" not in last.bcs
    assert last.made["o1"] == [at.chain.states[T]] and last.buckets == [T - 1]
    assert first.ports["o1"]["root"] == ing.ports["sin"]["root"] == last.ports["sin"]["root"] == st.root.hex()
    # the slot entries: leaf j·nb + b slot j's entry of the Programs' row at boundary b; the block one unit an entry,
    # its o0 (no public word) the prog tree's leaf k, the tree under its o0 frame binding
    (blk,) = h.rounds["block"]
    cls, tr, nb = sz.CLASS, at.trees["prog"], T // sz.CLASS.EVERY
    E = sz.entries_of(cls, 1, T)
    assert len(tr.rows) == E == cls.NR * nb == 8
    assert all(sum((sz._bits(tr, sz.entry_at(cls, nb, j, b * cls.EVERY)) for j in range(cls.NR)), [])
               == at.chain.progs[b * cls.EVERY] for b in range(nb))
    assert (blk.name, blk.base, blk.n, blk.refused, blk.outputs, blk.warden) == ("block", 0, E, None, None, {})
    assert blk.ports["o0"]["root"] == tr.root.hex() == ing.ports["prog0"]["root"] == ing.ports["prog3"]["root"]
    assert blk.ports["o0"]["reads"] == list(range(E)) and list(blk.ports) == ["o0"]
    assert blk.made["o0"] == [sz._bits(tr, k) for k in range(E)] == [u[0] for u in blk.units]
    assert blk.set_name == at.block_set and tr.binding == sz.block_binding(at.block_set, E)
    # bucket t's prog<j> is one table row a boundary, its entry j·nb + ⌊t / EVERY⌋
    for j in range(cls.NR):
        reads = [sz.entry_at(cls, nb, j, t) for t in range(1, T - 1)]
        assert ing.ports[f"prog{j}"]["reads"] == reads and reads[:2] == [j * nb] * 2
        assert ing.entries[f"prog{j}"]["positions"] == [j * nb, j * nb + 1] and ing.refs[f"prog{j}"] == sz.REG.table(
            reads)[1]
        assert last.ports[f"prog{j}"]["reads"] == [sz.entry_at(cls, nb, j, T - 1)]
    assert first.ports["prog1"]["reads"] == [nb] and "prog1" not in first.refs
    assert first.bcs["o1"] + ing.bcs["o1"] == ing.bcs["sin"] + last.bcs["sin"]
    assert first.bcs["o2"] + ing.bcs["o2"] + last.bcs["o2"] == eg.bcs["state"]
    assert ing.ports["o1"]["root"] == ing.ports["sin"]["root"] and ing.ports["o2"]["root"] == eg.ports["state"]["root"]
    assert first.ports["row"]["reads"] == [0] and sz.REG.READS_FROM not in first.ports["row"]
    assert ing.ports["row"]["reads"] == list(range(1, T - 1)) and ing.ports["row"][sz.REG.READS_FROM] == 1
    assert last.ports["row"]["reads"] == [T - 1] and last.ports["row"][sz.REG.READS_FROM] == T - 1
    side = at.sides["egress"][0]
    assert eg.ports["o1"]["reads"] == list(range(T)) and eg.ports["o1"][sz.REG.READS_FROM] == T
    assert eg.ports["o1"]["root"] == side.head.root.hex() and eg.bcs["o1"] == list(side.cells.bcs)
    c = chain["ingress-0"]
    assert sz.targets(at, "chain") == ["ingress-0"]
    assert c.refused == f"registered reads: table row {A - 1} does not open the registered root at position {A + 1}"
    assert c.ports["o1"]["root"] == c.ports["sin"]["root"] != ing.ports["o1"]["root"]
    assert all(m.refused is None for k, m in chain.items() if k != "ingress-0")
    (f,) = tampered.values()
    b = at.tampered.bucket
    assert f.changed == b
    assert f.refused == f"registered reads: table row {b} does not open the registered root at position {b}"
    assert f.outputs == [1] * T and f.made == eg.made and at.tampered.row != eg.made["o1"][b]
    assert f.ports["o1"]["root"] == at.tampered.head.root.hex() != eg.ports["o1"]["root"] and f.base == 0
    # unread: requests 1, 2 and 3 are anchored at 16, and request 1's slot 0 is read by no Program there
    (u,) = h.rounds["unread"]
    U = at.unread.bucket
    assert (U, at.unread.slot, at.unread.q) == (16, 0, 1) and sz.targets(at, "unread") == [u.name] == ["ingress-0"]
    assert (u.base, u.n, u.buckets, u.refused, u.outputs) == (0, 1, [U], None, [0])
    k = at.unread.entry
    assert k == sz.entry_at(cls, nb, 0, U) == 1
    row = sum((sz._bits(at.unread.tree, sz.entry_at(cls, nb, j, U)) for j in range(cls.NR)), [])
    assert sz.IS.prog_decode(cls, row) == [(0, *sz.IS.prog_decode(cls, at.chain.progs[U])[0][1:]),
                                           *sz.IS.prog_decode(cls, at.chain.progs[U])[1:]]
    assert [sz._bits(at.unread.tree, x) for x in range(E) if x != k] == [sz._bits(tr, x) for x in range(E) if x != k]
    assert {p: u.made[p][0] for p in ("o1", "o2")} == {p: ing.made[p][U - 1] for p in ("o1", "o2")}
    assert u.ports["prog0"]["root"] == at.unread.tree.root.hex() != ing.ports["prog0"]["root"]
    assert u.ports["o1"]["root"] == u.ports["sin"]["root"] == ing.ports["o1"]["root"]
    assert u.ports["prog0"]["reads"] == [k] and u.ports["o1"]["reads"] == [U + 1] and u.ports["sin"]["reads"] == [U]


@pytest.mark.parametrize("change", ["skip", "repeat", "alter"])
def test_a_state_tree_that_skips_or_alters_a_transition_is_refused_where_its_unit_computes_it(tmp_path, change):
    """Every state leaf but the empty state's is the S_t ingress unit t computes (its o1, a registered read of leaf
    t + 1) and unit t + 1 reads (its sin), so no committed state is the prover's word: a tree that skips S_A (leaves
    A + 1 … one on, the last repeated), repeats S_{A−1} in its place, or flips one of its prompt-id bits is refused
    at unit A's o1 ("does not open"), and every member but unit A's holds. A `SCAN_STATE` at such a tree's root is
    therefore no state tree a scan's proof holds, which the period check's `stateTied` asks of a slot's Program."""
    at, A = sz.link(tmp_path, setup=sz.TEST), sz.TEST.A
    T, cls, st = at.T, sz.TEST.cls, at.trees["state"]
    rows = [sz._bits(st, t) for t in range(T)]
    if change == "skip":
        rows[A + 1:] = rows[A + 2:] + rows[-1:]
    elif change == "repeat":
        rows[A + 1] = rows[A]
    else:
        k = sz.IS.layout(cls)["0.ids"][0]
        rows[A + 1] = [*rows[A + 1][:k], 1 - rows[A + 1][k], *rows[A + 1][k + 1:]]
    assert rows != [sz._bits(st, t) for t in range(T)]
    bad = sz.tree("state", rows, [w.head for w in at.sides["ingress"]], at.session)
    members = [sz.member(SimpleNamespace(**{**vars(at), "swapped": bad}), "chain", mb)
               for mb in sz.members_of(sz.TEMPLATES, 1, T)]
    refused = {m.name: m.refused for m in members if m.refused}
    assert refused == {"ingress-0": f"registered reads: table row {A - 1} does not open the registered root at "
                                    f"position {A + 1}"}, refused
    honest = [sz.member(at, "honest", mb) for mb in sz.members_of(sz.TEMPLATES, 1, T)]
    assert all(m.refused is None for m in honest) and sz.wired(honest) == []


def test_a_group_of_link_windows_is_one_registration_a_member_a_window(tmp_path):
    """The I/O tie's group (``--group G``) at the test class: link-windows 0 and 1 in one registration, the state and
    view trees G·T leaves (the state tree the group's window state, from bucket 0) and the slot entries NR a boundary of
    the group, each member's warden port its own link-window's registration, the group's last ingress unit a member of
    its own, and the chains running across the windows."""
    G, T = 2, sz.TEST.params.window
    h = sz.host(tmp_path, setup=sz.TEST, group=G)
    at = h.at
    assert at.windows == [0, 1] and [len(at.trees[k].rows) for k in ("state", "prog", "view")] == [
        G * T, sz.TEST.cls.NR * G * T // sz.TEST.cls.EVERY, G * T]
    assert h.rounds["block"][0].n == sz.entries_of(sz.TEST.cls, G, T)
    honest = {m.name: m for m in h.rounds["honest"]}
    assert [(k, m.base, m.n) for k, m in honest.items()] == [
        ("first", 0, 1), ("ingress-0", 1, T - 1), ("ingress-1", T, T - 1), ("last", G * T - 1, 1),
        ("egress-0", G * T, T), ("egress-1", G * T + T, T)]
    assert sz.wired(honest.values()) == [] and all(m.refused is None and set(m.outputs) == {1} for m in honest.values())
    ing1, last, eg1 = honest["ingress-1"], honest["last"], honest["egress-1"]
    assert [sz._bits(at.trees["state"], t) for t in range(G * T)] == sz.IS.window_state(
        sz.TEST.cls, at.chain.states[1:], 0, G * T)
    assert ing1.ports["row"]["root"] == at.sides["ingress"][1].head.root.hex() != honest["ingress-0"].ports["row"]["root"]
    assert ing1.ports["row"]["reads"] == list(range(T - 1)) and ing1.ports["row"][sz.REG.READS_FROM] == T
    assert ing1.ports["sin"]["reads"] == list(range(T, G * T - 1)) and ing1.ports["o1"]["reads"] == list(
        range(T + 1, G * T))
    assert (last.window, last.ports["row"]["reads"], last.ports["row"]["root"]) == (
        1, [T - 1], ing1.ports["row"]["root"])
    assert last.ports["sin"]["reads"] == [G * T - 1] and "o1" not in last.ports
    assert eg1.ports["o1"]["root"] == at.sides["egress"][1].head.root.hex() and eg1.ports["o1"][sz.REG.READS_FROM] == 3 * T
    assert [m.name for m in h.rounds["chain"] if m.refused] == sz.targets(at, "chain") == ["ingress-0"]
    assert [m.name for m in h.rounds["tampered"] if m.refused] == sz.targets(at, "tampered")
    assert [(m.name, m.buckets, m.outputs) for m in h.rounds["unread"]] == [("ingress-0", [4], [0])]
    with pytest.raises(SystemExit, match="a group is 1 or more link-windows"):
        sz.link(tmp_path / "none", setup=sz.TEST, group=0)


XOR8 = PrimitiveDefinition("ScanZkTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)


def _toy(name: str, f) -> object:
    return bind(CompositeDefinition(name, 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)), f, register=False))


FIRST = _toy("ScanZkTestFirst", lambda B, S_, x: B.call(XOR8, x[0], x[0]))
ING = _toy("ScanZkTestIngress", lambda B, S_, x: B.call(XOR8, x[0], x[1]))
EG = _toy("ScanZkTestEgress", lambda B, S_, x: B.call(XOR8, x[1], x[0]))
BLK = _toy("ScanZkTestBlock", lambda B, S_, x: B.call(XOR8, x[0], x[1]))


@pytest.mark.parametrize("G", [1, 3])
def test_the_members_are_the_programs_ranges(G):
    """`members_of` against the partition's own ranges (stand-in templates; `program` asserts the same of the scan's):
    each template's first unit its first member's base and its count its members', the units 0 … N − 1 in order, the
    ingress units computing buckets 1 … G·T − 1, the last of them a member of its own (``last``), and the egress units
    0 … G·T − 1; the unread round's one unit its bucket; the block round's one member every slot entry, which it must
    be given."""
    T, E, defs = 4, 3 * G, {"first": FIRST, "ingress": ING, "egress": EG, "block": BLK}
    with pytest.raises(ValueError, match="give entries"):
        sz.members_of(sz.BLOCK, G, T)
    for templates in {t for t, _o in sz.ROUNDS.values()}:
        members = sz.members_of(templates, G, T, bucket=T + 1, entries=E)
        counts = sz.counts_of(members)
        program = sz.PA.mixed_population_program([(defs[t], counts[t]) for t in templates])
        ranges = sz.PA.ranges(program, sz.PA.template_instances_query([defs[t] for t in templates]))
        assert [(lo, n) for _t, lo, n in ranges] == [
            (next(m.base for m in members if m.template == t), counts[t]) for t in templates]
        units = [(m.base + k, m.template, b) for m in members for k, b in enumerate(m.buckets)]
        assert [u for u, _t, _b in units] == list(range(sum(counts.values())))
        unread, block = templates == sz.UNREAD, templates == sz.BLOCK
        assert sorted(b for _u, t, b in units if t == "egress") == ([] if unread or block else list(range(G * T)))
        assert sorted(b for _u, t, b in units if t != "egress") == (
            list(range(G * T)) if "first" in templates else list(range(E)) if block else [T + 1] if unread else [])
        if unread:
            assert [m.name for m in members] == [f"ingress-{(T + 1) // T}"]
        if "first" in templates:
            assert [(m.name, m.window, list(m.buckets)) for m in members if m.template == "ingress"][-2:] == [
                (f"ingress-{G - 1}", G - 1, list(range(max(1, (G - 1) * T), G * T - 1))),
                ("last", G - 1, [G * T - 1])]


def _lanes(name: str, fixed: tuple[str, ...], L: int, NE: int) -> object:
    """A scan template's stand-in: its ``fixed`` byte parameters, then lane j's ``prog<j>``, an NE-bit entry row; it
    returns its first parameter negated."""
    ps = (*((p, Array(8, sz.BL.BIT)) for p in fixed), *((f"prog{j}", Array(NE, sz.BL.BIT)) for j in range(L)))
    return bind(CompositeDefinition(name, 0, (), lambda S_: (ps, Array(8, sz.BL.BIT)),
                                    lambda B, S_, x, *_: B.batch(sz.BL.NOT, x, axes=(0,)), register=False))


def test_the_scans_root_reads_each_units_slot_entries_at_one_root():
    """`scan_root`, the generator form `NetworkCheck.Period`'s `scanOf` reads (stand-in templates with the scan's
    parameters; the rows form's slow test reads the scan's own): each template one batch, the partition's ranges the
    members', and one-stage's reads of the slot entries (`entry_reads`, `registered.reads` on the root's ``prog``)
    each ingress unit's ``prog<j>`` at row j·nb + ⌊t / EVERY⌋ and no egress unit's. A root that gathers the entries
    at other boundaries is refused by name."""
    cls = SimpleNamespace(NR=2, EVERY=2, KO=1, IDB=1)
    G, T, NE = 1, 8, sz.IS.entry_bits(cls)
    nb = G * T // cls.EVERY
    defs = {"first": _lanes("ScanZkTestScanFirst", ("row",), cls.NR, NE),
            "ingress": _lanes("ScanZkTestScanBucket", ("row", "sin"), cls.NR, NE),
            "egress": _lanes("ScanZkTestScanEgress", ("state",), 0, NE)}
    members = sz.members_of(sz.TEMPLATES, G, T)
    counts = sz.counts_of(members)
    root = sz.scan_root([(defs[t], counts[t], sz.FIXED[t]) for t in sz.TEMPLATES], nb, cls.EVERY)
    query = sz.PA.template_instances_query([defs[t] for t in sz.TEMPLATES])
    assert [(lo, n) for _t, lo, n in sz.PA.ranges(root, query)] == [(0, 1), (1, T - 1), (T, T)]
    assert [n for n, _t in root.fn.signature()[0]] == ["0.row", "1.row", "1.sin", "2.state", "prog"]
    assert sz.entry_reads(root, query, members, cls, nb) == {
        f"prog{j}": [j * nb + t // cls.EVERY for t in range(T)] + [None] * T for j in range(cls.NR)}
    skewed = sz.scan_root([(defs[t], counts[t], sz.FIXED[t]) for t in sz.TEMPLATES], nb, 2 * cls.EVERY)
    with pytest.raises(SystemExit, match="aren't each unit's slot entries"):
        sz.entry_reads(skewed, query, members, cls, nb)


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
    assert set(h.rounds) == {"block", "honest", "chain", "tampered", "unread"} and h.at.run_id == "00" * 32
    assert "warden run" in h.at.source and h.at.admitted == {1: 0, 2: 2, 3: 2, 4: 12, 5: 28}
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


def _cells(run: Path) -> dict[tuple[str, int], dict]:
    """A run's row registrations in ``cells.jsonl``, by direction and link-window."""
    return {(d["registration"]["direction"], d["registration"]["window"]): d["registration"]
            for d in map(json.loads, (run / "cells.jsonl").read_text().splitlines()) if "registration" in d}


def test_the_registration_names_each_link_windows_warden_rows_its_ports_read(tmp_path):
    """The rows form's hold (`NetworkCheck.Period`: the scan's registration names each of W's row registrations' root
    and domain): a G = 2 group's honest round names, as ``rows/<direction>/<w>``, each of the four row registrations
    of the run's ``cells.jsonl``, T leaves each (ingress's row ports and egress's o1 read them); the tampered round,
    egress only, names the egress rows alone, the re-signed link-window's in its place. No root is named as the value
    Lean reports a read under, so the audit's registered reads (`audit.session_registered`, Lean's ``registered`` of
    each member) hold none of them, as the registration registers none."""
    G, T = 2, sz.TEST.params.window
    run = published(tmp_path / "run", sz.TEST, windows=G)
    h = sz.host(tmp_path / "host", run, setup=sz.TEST, group=G)
    regs = _cells(run)
    roots = sz.warden_roots([{"warden": m.warden} for m in h.rounds["honest"]])
    assert roots == [
        {"port": f"rows/{d}/{w}", "root": regs[d, w]["root"], "leaves": T, "domain": regs[d, w]["domain"]}
        for d in ("egress", "ingress") for w in range(G)]
    verdicts = {f"lean-verify/{m.name}": {"of_record": True, "accepted": True, "registered": {
        p: {"value": d["value"], "root": d["root"]} for p, d in m.warden.items()}} for m in h.rounds["honest"]}
    assert {e["value"] for v in verdicts.values() for e in v["registered"].values()} == {
        f"warden/{d}/{w}" for d in ("egress", "ingress") for w in range(G)}
    assert sz.AU.session_registered(verdicts, {"roots": roots}) == {}
    tampered, w = sz.warden_roots([{"warden": m.warden} for m in h.rounds["tampered"]]), h.at.tampered.window
    assert [r["port"] for r in tampered] == [f"rows/egress/{x}" for x in range(G)]
    assert tampered[w]["root"] == h.at.tampered.registration["root"] != regs["egress", w]["root"]
    assert all(tampered[x]["root"] == regs["egress", x]["root"] for x in range(G) if x != w)


def test_lean_reads_each_warden_port_at_the_root_the_warden_signed_never_one_the_prover_committed(tmp_path):
    """The verifier's own ``--registered`` file of each member (`registered_file`: `stage` writes it, `a0.lean_verify`
    passes it, and a Lean verdict's ``registered`` reports its roots) holds each warden port at the root and domain of
    the row registration the deployment's key signed (`warden_rows.check_registration`, read here afresh from the
    run's ``cells.jsonl``), the root `warden_roots` registers for that direction and link-window. A member whose rows
    the prover committed afresh, under salts of its own, is still read at the signed root, and refused there; and a
    statement whose header names a port registered that the verifier's file doesn't is refused, as Lean refuses it
    (`header_refusal`, `Flock.Registered.check`)."""
    G = 2
    run = published(tmp_path / "run", sz.TEST, windows=G)
    h = sz.host(tmp_path / "host", run, setup=sz.TEST, group=G)
    dep = json.loads((run / "deployment.json").read_text())
    key, session, params = WS.cs.read_public(dep["key"]), bytes.fromhex(dep["session"]), sz.W.WardenParams(**dep["params"])
    signed = {}
    for (d, w), doc in _cells(run).items():
        head, dom = sz.WR.check_registration(doc, key, session, params, WS.LINK, sz.Direction(d), w)
        signed[f"rows/{d}/{w}"] = (head.root.hex(), dom.domain_id.hex())
    assert len(signed) == 2 * G
    honest = h.rounds["honest"]
    roots = sz.warden_roots([{"warden": m.warden} for m in honest])
    assert {r["port"]: (r["root"], r["domain"]) for r in roots} == signed
    read: dict[str, set] = {}
    for m in honest:
        ports = sz.registered_file(m)["ports"]
        for p, doc in m.warden.items():
            read.setdefault(f"rows/{doc['direction']}/{doc['window']}", set()).add((ports[p]["root"], ports[p]["domain"]))
    assert read == {r: {x} for r, x in signed.items()}

    w = 1
    side = h.at.sides["egress"][w]
    c = side.cells
    mine = sz.WR.Cells.commit(c.link, c.direction, c.window, c.rows, c.wiring, c.params, c.session)
    assert mine.rows == c.rows and mine.head.root != side.head.root
    egress = [*h.at.sides["egress"][:w], SimpleNamespace(**{**vars(side), "cells": mine}), *h.at.sides["egress"][w + 1:]]
    at = SimpleNamespace(**{**vars(h.at), "sides": {**h.at.sides, "egress": egress}})
    mb = next(mb for mb in sz.members_of(sz.TEMPLATES, G, h.at.T) if mb.template == "egress" and mb.window == w)
    m = sz.member(at, "honest", mb)
    assert (sz.registered_file(m)["ports"]["o1"]["root"], sz.warden_roots([{"warden": m.warden}])[0]["root"]) == (
        signed[f"rows/egress/{w}"][0],) * 2
    assert m.refused == "registered reads: table row 0 does not open the registered root at position 0"

    m = next(x for x in honest if x.template == "egress" and x.window == w)
    assert m.refused is None and sz.header_refusal(m.ports, m.entries) is None
    assert sz.header_refusal(m.ports, {**m.entries, "o9": m.entries["o1"]}) == (
        "registered reads: the statement names port o9 registered, and the verifier's registered reads do not")


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
                   [{"port": "rows/egress/1", "root": "aa" * 64, "leaves": T, "domain": "bb" * 64}],
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


def test_an_audit_refused_with_every_member_held_answers_each_link_window_naming_its_failed_checks(tmp_path):
    """`answers` of a round whose members Lean all held but whose audit refused (a check of the record's own, say its
    registered reads): each link-window's answer is the reject of the round's registration, its reason the failed
    checks and their codes, and its registrations its own two."""
    G = 2
    h = sz.host(tmp_path / "host", published(tmp_path / "run", sz.TEST, windows=G), setup=sz.TEST, group=G)
    (out := tmp_path / "scan" / "honest").mkdir(parents=True)
    reg = R.record(*_program(G, h.at.T), sz.S.SCHEME, [], {"kind": "stand-in"}, leaf_layer="ee" * 64)
    (out / "registration.json").write_text(json.dumps(reg))
    res = {"window_accepted": False, "why": {"lean-verify/egress-0": None}, "failed": {"registered": ["registered"]}}
    got = sz.answers(h.at, out.parent, res, sz.Fraction(1, 100))
    why = "the scan's audit of link-windows [0, 1] refused: check registered: registered"
    assert got == {"verdict": "reject", "bound": None, "reason": why, "commitment": R.digest(reg)}
    doc = json.loads((out.parent / "answers.json").read_text())
    assert [x["window"] for x in doc["windows"]] == [0, 1] and all(x["answer"]["reason"] == why for x in doc["windows"])
    assert [x["registrations"] for x in doc["windows"]] == [
        [h.at.sides[s][w].registration for s in ("ingress", "egress")] for w in range(G)]


def _program(G: int, T: int) -> tuple:
    """The scan's Program, query and law (every unit drawn) for a group of G link-windows of T buckets."""
    counts = sz.counts_of(sz.members_of(sz.TEMPLATES, G, T))
    defs = {"first": FIRST, "ingress": ING, "egress": EG}
    return (sz.PA.mixed_population_program([(defs[t], counts[t]) for t in sz.TEMPLATES]),
            sz.PA.template_instances_query([defs[t] for t in sz.TEMPLATES]), sz.DR.parse_law(f"stratified:{2 * G * T}"))


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
    assert set(h.rounds) == {"block", "honest"} and sz.rounds_of(at) == {
        "block": (sz.BLOCK, "accepted"), "honest": (sz.TEMPLATES, "does not open")}
    refused = {m.name: m.refused for m in h.rounds["honest"] if m.refused}
    assert list(refused) == sz.targets(at, "honest") == [f"egress-{b // T}"]
    assert refused[f"egress-{b // T}"] == (f"registered reads: table row {b % T} does not open the registered root at "
                                           f"position {b % T}")
    assert sz.wired(h.rounds["honest"]) == []


@pytest.mark.slow
def test_the_rows_forms_case_is_the_link_windows_rows_certificates_and_the_scans_answer(tmp_path, monkeypatch):
    """`scan_period.rows_case` of one link-window in the rows form's formats (`NetworkCheck.Period`; its Lean checks the
    case on a node: scan_period.sh): the deployment's key, session and schedule, the registered Program and its
    descriptor (``joint/program.json``, the generator form, `scan_zk.scan_root`, whose one-stage reads are each unit's
    slot entries) as the scan, the block's audit (its descriptor, free inputs and no steps; its registration, whose
    ``SlotEntry_v1{…}/c0/o0`` is the scan's ``scan/prog`` root and domain and whose ``scan/state`` is the scan's, and
    its answer), the registration, and its answer, three records' certificates (requests, responses, fails: its own)
    and two row registrations, each named by the registration's roots. The period's mirror reads the block's
    `SlotEntry_v1` template's entries off its descriptor, a stand-in whose closure digest is no pinned Definition's
    (`period_case.SLOT_ENTRY_PINS`), and refuses the honest case at the wiring (`period_case.UNWIRED`); with that digest
    pinned it accepts it, its scan's terms read off the test class's own templates with no free inputs, and refuses the
    forged case (a certificate's signature changed) at the certificates, the tampered case (another Program's
    registration) at the registration, the chain case (the round's registration of the Program) at its refusal, a block
    whose entries are at another root by name, and one whose root is at its o1, no `SlotEntry_v1` template's entries, at
    the wiring (`period_case.UNWIRED`), and a block whose ``scan/state`` is a state tree that skips a transition, or
    that names none, at the state tie (`period_case.UNSTATED`); a group of two link-windows is refused by name. Slow
    (the test class's Program, about 30 s): the period check reads the scan's templates' signatures, which no stand-in
    can stand for."""
    import scan_period as SP

    PC = SP.mirror(PERIOD_CASE)
    run = published(tmp_path / "run", sz.TEST, windows=1)
    h = sz.host(tmp_path / "host", run, setup=sz.TEST)
    T, scan, cls = h.at.T, tmp_path / "scan", sz.TEST.cls
    E, nb = sz.entries_of(cls, 1, T), T // cls.EVERY
    defs = sz._defs(cls)
    members = sz.members_of(sz.TEMPLATES, 1, T)
    counts = sz.counts_of(members)
    program = sz.scan_root([(defs[t], counts[t], sz.FIXED[t]) for t in sz.TEMPLATES], nb, cls.EVERY)
    query = sz.PA.template_instances_query([defs[t] for t in sz.TEMPLATES])
    sz.entry_reads(program, query, members, cls, nb)
    (scan / "joint").mkdir(parents=True)
    (scan / "joint" / "program.json").write_bytes(canonical_json(encode_program(program)))
    tr, st = h.at.trees["prog"], h.at.trees["state"]
    prog = {"root": tr.root.hex(), "leaves": E, "domain": sz.ADV.domain(tr.binding, E).domain_id.hex()}
    state = {"root": st.root.hex(), "leaves": T, "domain": sz.ADV.domain(st.binding, T).domain_id.hex()}
    roots = sz.warden_roots([{"warden": m.warden} for m in h.rounds["honest"]]) + [
        {"port": "scan/prog", **prog}, {"port": "scan/state", **state}]
    law = sz.DR.parse_law(f"stratified:{2 * T}")
    reg, chain = (R.record(program, query, law, sz.S.SCHEME, roots, {"kind": "stand-in"}, leaf_layer=x * 64)
                  for x in ("ee", "ff"))
    other = {**reg, "program": "ab" * 64}
    for name, r in (("tampered", other), ("chain", chain)):
        (scan / name).mkdir()
        (scan / name / "registration.json").write_text(json.dumps(r))
    draw = {**sz.DR.derive(program, query, law), "population": 2 * T, "units": list(range(2 * T))}
    record = {"accepted": True, "registration": reg, "draw": draw, "proved_units": list(range(2 * T))}
    answer = sz.window_answer(reg, record, sz.Fraction(1, 100), "unused")
    block = sz.PA.mixed_population_program([(defs["block"], E)])
    bquery, blaw = sz.PA.template_instances_query([defs["block"]]), sz.DR.parse_law(f"stratified:{E}")
    entries = PC.entries_port(sz.PA.template_id(defs["block"]))
    assert entries == f"{sz.port_prefix(defs['block'], SimpleNamespace(template='block'))}/o0"

    def block_audit(o0: dict, port: str = entries, scan_state: dict | None = state) -> dict:
        stated = [{"port": "scan/state", **scan_state}] if scan_state else []
        breg = R.record(block, bquery, blaw, sz.S.SCHEME, [{"port": port, **o0}, *stated], {"kind": "stand-in"},
                        leaf_layer="cc" * 64)
        bdraw = {**sz.DR.derive(block, bquery, blaw), "population": E, "units": list(range(E))}
        brec = {"accepted": True, "registration": breg, "draw": bdraw, "proved_units": list(range(E))}
        return {"answer": sz.window_answer(breg, brec, sz.Fraction(1, 100), "unused"),
                "free": E * sz.IS.entry_bits(cls), "program": encode_program(block), "registration": breg, "steps": 0}

    registrations = [h.at.sides[s][0].registration for s in ("ingress", "egress")]
    audit = block_audit(prog)
    doc = {"audits": [audit], "registration": reg, "windows": [
        {"window": 0, "answer": answer, "registrations": registrations}]}
    (scan / "answers.json").write_text(json.dumps(doc))
    (scan / "summary.json").write_text(json.dumps({"rounds": {r: {"audit": {"why": {"x": "y"}}}
                                                              for r in ("tampered", "chain")}}))
    dep, regs = json.loads((run / "deployment.json").read_text()), _cells(run)
    case = SP.rows_case(run, scan)
    p = case["params"]
    assert sorted(p) == PC.ROWS_PARAMS_KEYS
    assert (p["key"], p["session"], p["schedule"], p["link"]) == (dep["key"]["public"], dep["session"], dep["params"],
                                                                  WS.LINK)
    assert (p["program"], p["windows"], p["first"]) == (reg["program"], 1, 0)
    assert p["audits"] == [{"free": E * sz.IS.entry_bits(cls), "program": encode_program(block), "steps": 0}]
    key = sz.PA.template_id(defs["block"])
    digest = PC.slot_entry_digest(encode_program(block), key)
    assert PC.params(p).audits[0].entries == ((key, E, digest),) and E == T // cls.EVERY * cls.NR
    assert digest not in PC.SLOT_ENTRY_PINS
    assert p["scan"] == json.loads((scan / "joint" / "program.json").read_text())
    assert json.loads(bytes.fromhex(case["commitment"])) == {"audits": [audit["registration"]], "windows": [reg]}
    ev = json.loads(bytes.fromhex(case["evidence"]))
    (x,) = ev["windows"]
    assert ev["audits"] == [audit["answer"]]
    assert x["answer"] == answer and x["registrations"] == [regs["ingress", 0], regs["egress", 0]]
    assert [(c["link"], c["direction"], c["window"]) for c in x["certificates"]] == [
        ("warden", "ingress", 0), ("warden", "egress", 0), ("warden-fails", "egress", 0)]
    assert all({"root": d["root"], "domain": d["domain"]} in [{"root": r["root"], "domain": r["domain"]}
                                                               for r in reg["roots"]] for d in x["registrations"])
    ports = sz.ports_of(sz.TEST.cls)
    bits = {tp: dict(ins + outs) for tp, (ins, outs) in ports.items()}
    assert PC.params(p).scan == PC.Scan(T, 0, 16 + bits["first"]["o1"] + bits["first"]["o2"], 16 + bits["egress"]["o1"])
    out = PC.reference(case)
    assert (out["verdict"], out["reason"]) == ("reject", f"window 0: {PC.UNWIRED}"), out
    monkeypatch.setattr(PC, "SLOT_ENTRY_PINS", (*PC.SLOT_ENTRY_PINS, digest))
    out = PC.reference(case)
    assert (out["verdict"], out["public"]["windows"], out["public"]["scans"][0]["units"]) == ("accept", 1, 2 * T), out
    refused = {r: PC.reference(SP.rows_case(run, scan, r)) for r in ("forged", "tampered", "chain")}
    assert {r: (a["verdict"], a["reason"]) for r, a in refused.items()} == {
        "forged": ("reject", "window 0: certificates (none of a window in which the network certifier refused a "
                   "connection)"), "tampered": ("reject", "window 0: registration"),
        "chain": ("reject", "window 0: service: the scan's chain round refused: {'x': 'y'}")}
    elsewhere = {**prog, "root": "dd" * 64}
    (scan / "answers.json").write_text(json.dumps({**doc, "audits": [block_audit(elsewhere)]}))
    got = PC.reference(SP.rows_case(run, scan))
    assert (got["verdict"], got["reason"]) == ("reject", "window 0: the scan's prog is no compute audit's output")
    (scan / "answers.json").write_text(json.dumps({**doc, "audits": [block_audit(prog, entries[:-1] + "1")]}))
    got = PC.reference(SP.rows_case(run, scan))
    assert (got["verdict"], got["reason"]) == ("reject", f"window 0: {PC.UNWIRED}"), got
    rows = [sz._bits(st, t) for t in range(T)]
    skipped = sz.tree("state", rows[:sz.TEST.A + 1] + rows[sz.TEST.A + 2:] + rows[-1:],
                      [w.head for w in h.at.sides["ingress"]], h.at.session)
    for scan_state in ({**state, "root": skipped.root.hex()}, None):
        (scan / "answers.json").write_text(json.dumps({**doc, "audits": [block_audit(prog, scan_state=scan_state)]}))
        got = PC.reference(SP.rows_case(run, scan))
        assert (got["verdict"], got["reason"]) == ("reject", f"window 0: {PC.UNSTATED}"), got
    (scan / "answers.json").write_text(json.dumps({"registration": reg, "windows": [{"window": w} for w in (0, 1)]}))
    with pytest.raises(SystemExit, match="a group of 2 link-windows .* scan_zk.py --group 1"):
        SP.rows_case(run, scan)


def test_the_scan_leaves_out_each_members_refs_which_the_statement_fixes_and_still_scans_its_b_c(tmp_path):
    """A member's refs (`refs_of`), each instance's table row of each port, are small integers its statement fixes:
    S_0's first 16 bytes (``now`` and ``phase`` 1, ``last`` 0, slot 0 empty) equal instance 1's refs where it reads
    (row, sin, prog0, prog1) = (1, 1, 0, 0), as a group's ingress member's do. That piece is left out of the scan; a
    witness piece in the same file's b‖c is still scanned for, and refs that aren't each port's table rows in first-read
    order (past a table, out of order, or leaving a row unread) are refused."""
    cls = sz.TEST.cls
    s0 = sz._bytes(sz.IS.encode(cls, sz.IS.State(1, 1, 0, sz.IS.empty(cls).slots)))
    names, n = ("row", "sin", "prog0", "prog1", "o0", "o1", "o2"), 4
    tables = {p: 2 if p.startswith("prog") else n for p in names}
    bc = [hashlib.sha512(b"bc %d" % k).digest() * 2 for k in range(sum(tables.values()))]
    secret = hashlib.sha512(b"secret").digest()[:16]
    bc[5] = bc[5][:32] + secret + bc[5][48:]

    def member(refs):
        d = tmp_path / f"m{len(list(tmp_path.iterdir()))}"
        d.mkdir()
        body = b"".join(bc) + b"".join(r.to_bytes(4, "little") for row in refs for r in row)
        (d / f"pub-{n}.bin").write_bytes(json.dumps({"instances": n, "shared_rows": tables}).encode() + b"\n" + body)
        (d / "circuit.txt").write_text("circuit\n")
        return {"name": "ingress-0", "dir": d, "n": n, "pin": d.name, "ports": [SimpleNamespace(name=p) for p in names]}

    m = member([[i, i, i // 2, i // 2, i, i, i] for i in range(n)])
    pub = (m["dir"] / f"pub-{n}.bin").read_bytes()
    assert s0[:16] in pub and any(s0[:2]) and any(s0[4:6]) and not any(s0[8:16])
    a8 = SimpleNamespace(PIECE=16, needles=lambda ps: {nd: k for k, x in ps.items() for nd in (x, x.hex().encode())},
                         find=lambda data, table: {k: 1 for nd, k in table.items() if nd in data})
    blobs = {"first o1 0": s0, "ingress-0 o1 2": secret * 2}
    at = SimpleNamespace(setup=sz.TEST, ordinals=[1])
    kept = sz.LS.public_pieces(a8, blobs, set(blobs), sz.known([m], at), fields=True)[0]
    assert "first o1 0 @0" not in kept and kept["ingress-0 o1 2 @0"] == secret and secret in pub
    assert "first o1 0 @0" in sz.LS.public_pieces(a8, blobs, set(blobs), sz.known([], at), fields=True)[0]
    for bad in (lambda i: [i] * 7, lambda i: [i, i, 1 - i // 2, i // 2, i, i, i], lambda i: [i, i, i // 2, 0, i, i, i]):
        with pytest.raises(SystemExit, match="refs aren't each port's table rows in first-read order"):
            sz.refs_of(member([bad(i) for i in range(n)]))


def _a8():
    """`a8`'s piece scan as `scan_zk.received` calls it: 16-byte pieces found raw or as hex in each file the verifier
    received, and the control holding the held ones."""
    def needles(p):
        return {f: k for k, x in p.items() for f in (x, x.hex().encode())}

    def find(data, t):
        return {k: 1 for nd, k in t.items() if nd in data}

    def received(vdir, control, pieces, held):
        files = {f.name: {"bytes": f.stat().st_size, "found": find(f.read_bytes(), needles(pieces))}
                 for f in sorted(vdir.iterdir())}
        ctl = find(control.read_bytes(), needles(pieces))
        return {"blobs": len(pieces), "files": files, "verifier_bytes": sum(x["bytes"] for x in files.values()),
                "verifier_found": sorted({k for x in files.values() for k in x["found"]}),
                "control": {"blobs_found": len(ctl), "holds_all": held <= set(ctl)}}

    return SimpleNamespace(PIECE=16, needles=needles, find=find, received=received)


def test_a_row_shorter_than_a_piece_is_found_whole_and_held_to_the_pieces_rule(tmp_path):
    """w63k4's slot entries are 14 bytes, under the scan's 16-byte piece: `received` finds each whole (raw or as hex,
    `leak_scan.find_whole`) beside the longer rows' cut pieces, leaves an all-zero one out as sparse, and `unseen` fails
    the round where the verifier received one or the control doesn't hold it."""
    entry = sz._bytes(sz.IS.entry_encode(sz.W63K4.cls, sz.IS.entry(3, [13, 110, 207])))
    state = hashlib.sha512(b"state").digest()[:40]
    assert len(entry) == 14
    blobs = {"ingress-0 prog0 0": entry, "ingress-0 prog1 0": bytes(14), "ingress-0 o1 0": state}
    control = tmp_path / "inst.bin"
    control.write_bytes(state + entry)
    vdir = tmp_path / "verifier"
    vdir.mkdir()
    (vdir / "proof.bin").write_bytes(hashlib.sha512(b"proof").digest())
    rec = sz.received(_a8(), vdir, control, blobs, set(blobs), b"")
    assert rec["whole_rows"]["rows"] == 1 and rec["sparse_pieces"] == 1 and sz.unseen(rec)
    assert sorted(sz.LS.cut({"ingress-0 o1 0": state}, 16)) == [f"ingress-0 o1 0 @{o}" for o in (0, 16, 24, 32)]
    assert rec["blobs"] == 4
    for leak in (entry, entry.hex().encode()):
        (vdir / "proof.bin").write_bytes(b"\xff" + leak)
        rec = sz.received(_a8(), vdir, control, blobs, set(blobs), b"")
        assert rec["whole_rows"]["verifier_found"] == ["ingress-0 prog0 0"] and not rec["verifier_found"]
        assert not sz.unseen(rec)
    (vdir / "proof.bin").write_bytes(hashlib.sha512(b"proof").digest())
    control.write_bytes(state)
    rec = sz.received(_a8(), vdir, control, blobs, set(blobs), b"")
    assert rec["whole_rows"]["control"]["missing"] == ["ingress-0 prog0 0"] and not sz.unseen(rec)


@pytest.mark.parametrize("name", sorted(sz.SETUPS))
def test_schedule_prints_the_classs_schedule_as_warden_serves_flags(capsys, name):
    assert sz.main(["schedule", "--class", name]) == 0
    flags = capsys.readouterr().out.split()
    assert dict(zip(flags[::2], flags[1::2])) == {f"--{k.replace('_', '-')}": str(v)
                                                  for k, v in asdict(sz.SETUPS[name].params).items()}


def test_w63k4_is_w63s_class_at_a_window_circuits_ko_and_ki_its_certifier_w63s_at_egress_cells_ko_plus_1():
    """w63k4 is w63's class at KO 4 and KI 32 (a window circuit's of 8 slots, 32 prompt ids and 4 outputs a request),
    its certifier w63's but for egress_cells 5, which the class serves and w63's 65 it refuses by name; its traffic
    fills the class (a prompt of KI ids, a response of KO), and its egress is 4 request slots of KO ids."""
    s = sz.SETUPS["w63k4"]
    assert s.cls == replace(sz.CLASS, KO=4, KI=32) and (s.cls.SLOTS, s.params.window_programs) == (4, 8)
    assert {k: v for k, v in asdict(s.params).items() if asdict(sz.PARAMS)[k] != v} == {"egress_cells": 5}
    s.cls.serves(s.params)
    with pytest.raises(ValueError, match=re.escape("set egress_cells <= KO + 1 = 5 (it is 65)")):
        s.cls.serves(sz.PARAMS)
    assert max(map(len, s.prompts)) == s.cls.KI and max(map(len, s.responses)) == s.cls.KO
    assert s.cls.egress == sz.EG.Shape(512, 16, 4, 4, 4, 17, 27)
    assert (sz.IS.entry_bits(s.cls), sz.IS.view_bits(s.cls)) == (7 * 16, 32 + 4 * 192)


def test_w63k4s_rounds_read_as_w63s(tmp_path):
    """w63k4's run admits as w63's (request 5 in request 1's slot past its egress) and answers each request its
    response, and each round's host read is the round's: block and honest accepted, every honest output 1 and its
    chains wired; chain refused at unit A's o1, tampered at its re-signed row, and unread's one unit output 0."""
    s = sz.W63K4
    h = sz.host(tmp_path, setup=s)
    at, T, A = h.at, s.params.window, s.A
    assert at.admitted == {1: 0, 2: 2, 3: 2, 4: 12, 5: 28} and [x.q for x in at.chain.scan[29].slots] == [5, 2, 3, 4]
    assert at.outputs == {q: r for q, r in enumerate(s.responses, 1)}
    rounds = {k: {m.name: m for m in ms} for k, ms in h.rounds.items()}
    (blk,) = rounds["block"].values()
    assert (blk.n, blk.refused, blk.outputs) == (sz.entries_of(s.cls, 1, T), None, None) and blk.n == 8
    honest = rounds["honest"]
    assert [(m.name, m.base, m.n) for m in honest.values()] == [("first", 0, 1), ("ingress-0", 1, T - 2),
                                                                ("last", T - 1, 1), ("egress-0", T, T)]
    assert all(m.refused is None and m.outputs == [1] * m.n for m in honest.values())
    assert sz.wired(honest.values()) == []
    assert rounds["chain"]["ingress-0"].refused == (f"registered reads: table row {A - 1} does not open the registered "
                                                    f"root at position {A + 1}")
    (f,) = rounds["tampered"].values()
    b = at.tampered.bucket
    assert f.refused == f"registered reads: table row {b} does not open the registered root at position {b}"
    (u,) = rounds["unread"].values()
    assert (at.unread.bucket, at.unread.q, u.refused, u.outputs) == (16, 1, None, [0])


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


def test_a_fails_reason_outside_its_fails_bucket_is_refused_by_name_and_one_outside_the_group_is_not_read(tmp_path):
    """A connection's reason in the warden's ``fails`` record is in the bucket of its egress `FAIL` (a void run, whose
    link-windows the scan answers refused): a reason in a scanned bucket whose egress row has no `FAIL` is refused,
    naming the run's ``check.json`` problems that say a connection failed closed; one in a link-window past the group
    is another group's, and the group is read as if it weren't there."""
    run = published(tmp_path / "run")
    rec = WS.ar.unpack_record(WS.ar.read_blobs(run / "responses.records")[0], sz.PARAMS.cell_bytes)
    t, j = next((t, j) for t, row in enumerate(rec.grid) for j, f in enumerate(row) if f is not None)
    grid = tuple(tuple(f if (u, k) == (t, j) else None for k, f in enumerate(row)) for u, row in enumerate(rec.grid))
    (run / "check.json").write_text(json.dumps({"problems": ["audit: connection 7 failed closed (late) at egress cell 2",
                                                             "connection 7: fail, HTTP None, complete False"]}))
    _rewrite(run / "fails.records", [WS.ar.pack_record(replace(rec, grid=grid, window=0))])
    with pytest.raises(SystemExit, match=re.escape(f"egress buckets [] hold a FAIL and the warden's fails record has a "
                                                   f"reason in buckets [{t}], not the same: a failed connection's "
                                                   "reason is in its FAIL's bucket (connection 7 failed closed (late) "
                                                   "at egress cell 2)")):
        sz.host(tmp_path / "h", run)
    _rewrite(run / "fails.records", [WS.ar.pack_record(replace(rec, grid=grid, window=3))])
    at = sz.host(tmp_path / "h2", run).at
    assert (at.void, at.failed, at.failed_why) == ([], 0, [])


@pytest.mark.parametrize("flags,why", [
    ([], "nothing draws on the verifier's own coins"),
    (["--draw", "16", "--main", "m.txt"], "nothing draws on the verifier's own coins"),
    (["--challenger", "remote", "--main", "m.txt"], "local, loopback:HOST:PORT or gate:USER@HOST, not 'remote'"),
    (["--challenger", "gate:research@h", "--main", "m.txt", "--peer", "p"], "needs --peer and --challenger-key"),
    (["--challenger", "local", "--draw", "16"], "needs --main"),
    (["--group", "0"], "--group is 1 or more link-windows"),
])
def test_a_scan_that_cant_name_its_challenger_is_refused_by_name_before_anything_runs(tmp_path, capsys, flags, why):
    """Every round draws, every unit included, so a scan that proves anything names its challenger; only ``--dry``,
    which draws nothing, runs without one."""
    with pytest.raises(SystemExit):
        sz.main(["prove", "--bin", "b", "--lean", "l", "--out", str(tmp_path / "out"), *flags])
    assert re.search(re.escape(why), capsys.readouterr().err) and not (tmp_path / "out").exists()


def test_each_round_opens_a_session_of_its_own_on_its_root_and_ends_its_turn_on_the_way_out(tmp_path, monkeypatch):
    """`scan_zk.challenger` on the fake challenger: each round's session is on `round_coins.root` of the group's
    `window_root`, the step and the round, so no two rounds share coins; leaving the round, by an error too, ends its
    turn and publishes the session's evidence, and a round that never planned its draw opened nothing."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_round_coins import fake_challenger

    a = fake_challenger(tmp_path, monkeypatch)
    at = sz.link(tmp_path / "syn")
    roots = {}
    for name in sz.ROUNDS:
        with sz.challenger(a, at, name, tmp_path / name) as coins:
            roots[name] = coins.root
        assert "was not opened" in coins.published["error"], coins.published
    assert roots == {n: sz.RC.root(sz.window_root(at), "scan", n) for n in sz.ROUNDS}
    assert len(set(roots.values())) == len(sz.ROUNDS)
    with pytest.raises(RuntimeError, match="the round failed"), sz.challenger(a, at, "honest", tmp_path / "again") as coins:
        raise RuntimeError("the round failed")
    assert coins.published is not None and not coins.held(b"", b"", None)["ok"]
    assert not (tmp_path / "coins.log").exists()


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
    with pytest.raises(ValueError, match="generator form is over the scan's 3 templates"):
        sz.RoundStream(spec, units, 2.0 ** -20, lambda n: bytes(n), (2, 2))


def test_the_joint_rounds_stream_holds_its_registration_to_the_generator_form():
    """The joint round's stream (`RoundStream` at the round's ``(nb, every)``; stand-in templates with the scan's
    parameters) builds its own Program in the generator form (`scan_root`) at the counts `register` takes, so
    one-stage's registration check holds the round's registration to it and its proof blocks are in order (`register`'s
    `population-order`), where `service.Stream`'s own copy, the mixed population, refuses it (`R2-program`); a
    registration of the generator form at other boundaries is refused, as are counts that aren't N. Without ``scan``
    the stream's Program is the mixed one. Python's check alone: the stream's registration runs the Lean's
    (`verifier.check`) too, which the dry run's honest and chain rounds hold."""
    cls = SimpleNamespace(NR=2, EVERY=2, KO=1, IDB=1)
    G, T, NE, k = 1, 8, sz.IS.entry_bits(cls), 4
    nb = G * T // cls.EVERY
    defs = {"first": _lanes("ScanZkTestScanFirst", ("row",), cls.NR, NE),
            "ingress": _lanes("ScanZkTestScanBucket", ("row", "sin"), cls.NR, NE),
            "egress": _lanes("ScanZkTestScanEgress", ("state",), 0, NE)}
    members = sz.members_of(sz.TEMPLATES, G, T)
    counts = sz.counts_of(members)

    def root(every: int):
        return sz.scan_root([(defs[t], counts[t], sz.FIXED[t]) for t in sz.TEMPLATES], nb, every)

    query = sz.PA.template_instances_query([defs[t] for t in sz.TEMPLATES])
    pg = SimpleNamespace(program=root(cls.EVERY), query=query, templates=sz.TEMPLATES, scan=(nb, cls.EVERY),
                         stagers={t: SimpleNamespace(defn=defs[t]) for t in sz.TEMPLATES})
    staged = [{"name": m.name, "n": len(m.buckets), "defn": defs[m.template]} for m in members]
    want = {"scan/state": {"leaves": T, "domain": "cc" * 64}}
    spec, units, given = sz.spec_of(pg, staged, want, {"scan/state": sz.BitRowLeaf.hiding_schema}, k, "dd" * 64)
    n = 2 * G * T
    law = sz.S.selection_law(spec.selection, n)

    def record(program) -> dict:
        roots = [{"port": "scan/state", "root": "01" * 64, **want["scan/state"]}]
        return R.record(program, query, sz.DR.parse_law(f"stratified:{k}"), sz.S.SCHEME, roots, {"kind": "stand-in"},
                        leaf_layer="ee" * 64)

    def check(rec: dict, program) -> tuple[str, ...]:
        return R.check(rec, program, query, law, sz.S.SCHEME, want, leaf_layer="ee" * 64).codes

    stream = sz.RoundStream(spec, units, 2.0 ** -20, lambda m: bytes(m), pg.scan)
    own = stream._program(n, given)
    rec = record(pg.program)
    assert sz.PA.program_digest(own) == rec["program"] == sz.PA.program_digest(pg.program)
    assert check(rec, own) == () and sz.PA.in_order(own, query, n)
    mixed = sz.S.Stream(spec, units, budget=1, delta=2.0 ** -20, challenger=sz.CHALLENGER)._program(n, given)
    assert "R2-program" in check(rec, mixed)
    assert "R2-program" in check(record(root(2 * cls.EVERY)), own)
    assert stream._program(n + 1, given) is None
    plain = sz.RoundStream(spec, units, 2.0 ** -20, lambda m: bytes(m))._program(n, given)
    assert sz.PA.program_digest(plain) == sz.PA.program_digest(mixed) != rec["program"]
