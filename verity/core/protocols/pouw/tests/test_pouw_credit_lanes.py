"""`credit`'s lanes form (`leaf_lanes`, TOP-138): at a K the period-open commits as ``lane<K>`` and ``combine<K>``, credit
still draws the window's leaves, and each drawn leaf is proved as `leaf_lanes.statements` (R/4)(C/8) lanes and the
combine, each verified against its own one-call leaf Program, a population of one, with the ``--registered`` file credit
builds at the leaf's served place, its partials ``<prefix>partials@<u>`` held to one produced record under the window's
draw root. Every refusal of the form is shown here: ``missing``, ``outcome``, ``failed``, ``circuit``, ``proof``,
``registration`` and ``lanes-shape``.

The window is `test_circuit_h1window`'s stand-in tile at 8 × 16 (four lanes) and K = 384 and 416, so `credit.LANES_TILES`
is widened to it; the 32 × 32 refusal is shown on its own. The stand-in's unit is not h1zy's, so its leaf Programs are
stand-in bytes that `leaf_of` is replaced to accept; the slow test holds `leaf_of` and `leaf_unit_id` to the real
`h1window.leaf_program`. The verifier is a stand-in `run` that accepts a session exactly when credit hands it the files
the prover proved against (the prover's ``--registered`` file is `h1window.leaf_reads_file`'s at the served place, with
`h1window.leaf_known_ports`) and reports what the Lean verifier would: the program, the registered finding, the units and
the rows it opened (``openings``, which `produced.shared` reads)."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw import leaf_lanes as LL
from verity.core.protocols.pouw.circuit import h1window as HW
from verity.core.protocols.sampled_proofs.one_stage import produced as PR
from verity.core.protocols.sampled_proofs.one_stage import proofs as PF
from verity.ml.circuits import codec

if "test_circuit_h1window" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("test_circuit_h1window", Path(__file__).parent / "test_circuit_h1window.py")
    sys.modules["test_circuit_h1window"] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sys.modules["test_circuit_h1window"])
h1 = sys.modules["test_circuit_h1window"]

LEAF = (8, 16)
LINEARS = (("q_proj", 384, 32), ("k_proj", 384, 16), ("down_proj", 416, 16))
STEPS, PREFIX = 2, "h1/w/"
SALT, WINDOW, DRAW_ROOT = "5a" * 64, "cd" * 64, "d7" * 64
COIN = {"after": DRAW_ROOT, "coin": "c0" * 64, "epoch": 0, "index": 0, "label": "draw", "path": [], "salt": "00" * 64}
DRAW = {"coin": COIN, "root": DRAW_ROOT}
#: units 1 and 6 of K = 384's calls 0 and 4, unit 3 of K = 416's call 2
DRAWN = [1, 3, 6]


@dataclasses.dataclass(frozen=True)
class Opened(PF.Finding):
    """A verdict with the rows its session opened (``{port: [[position, b || c] ...]}``), as `produced.shared` reads."""

    openings: dict = dataclasses.field(default_factory=dict)


@pytest.fixture(scope="module")
def world() -> dict:
    prog = HW.program(LINEARS, h1.CALLS, HW.slot(5, LEAF), LEAF, prefix=PREFIX, unit=lambda k: h1.unit(k, LEAF))
    desc = codec.encode_program(prog)
    raw = codec.canonical_json({k: v for k, v in desc.items() if k != "annotations"})
    layout, why = Cr.layout_of(raw)
    assert why is None, why
    entries, seen = [], set()
    for u in range(sum(n for _u0, n in HW.call_units(prog))):
        at = HW.place(prog, u, STEPS)
        for v in (LL.A, LL.B, LL.ZY, LL.RECEIPT):
            if at[v][0] not in seen:
                seen.add(at[v][0])
                entries.append({"value": at[v][0], "root": hashlib.sha512(at[v][0].encode()).hexdigest()})
    return {"prog": prog, "raw": raw, "digest": hashlib.sha512(raw).hexdigest(), "layout": layout,
            "shape": Cr.shape_of(raw), "entries": entries}


def leaf_desc(kind: str, k: int) -> bytes:
    return f"stand-in leaf Program of {kind} at K = {k}".encode()


def unit_id(kind: str, k: int) -> str:
    return f"{Cr.LEAF_UNIT[kind]}{{TILE=stand-in}}"


def circuit(name: str) -> bytes:
    return f"circuit file {name}\n".encode()


def _sha(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


@pytest.fixture
def lanes(monkeypatch) -> None:
    """The stand-in window's lanes allowed: its 8 × 16 tile, and its leaf Programs the stand-in bytes."""
    monkeypatch.setattr(Cr, "LANES_TILES", (*Cr.LANES_TILES, LEAF))
    monkeypatch.setattr(Cr, "leaf_unit_id", lambda shape, k, kind: unit_id(kind, k))
    monkeypatch.setattr(Cr, "leaf_of", lambda leaf, unit, k, kind: leaf == leaf_desc(kind, k) and unit == unit_id(kind, k))


def files_of(lanes_ks=(384, 416), tile_ks=()) -> dict:
    out = {f"k{k}": _sha(circuit(f"k{k}")) for k in tile_ks}
    for k in lanes_ks:
        out.update({f"{kind}{k}": _sha(circuit(f"{kind}{k}")) for kind in (LL.LANE, LL.COMBINE)})
    if lanes_ks:
        out[Cr.LEAF_PROGRAMS] = {f"{kind}{k}": _sha(leaf_desc(kind, k)) for k in lanes_ks for kind in (LL.LANE, LL.COMBINE)}
    return {**out, Cr.LIVE: _sha(b"live")}


def _roots(w: dict) -> dict[str, str]:
    return {e["value"]: e["root"] for e in w["entries"]}


def prove(w: dict, to: Path, files: dict, units=DRAWN) -> tuple[dict, dict]:
    """The window's lanes evidence as an honest prover delivers it, and what each statement was proved against (by its
    public file): each tile-form call's statement as `test_pouw_credit_statements.prove`'s, and each drawn lanes-form
    leaf's partials committed under the draw root and its lanes and combine, each against its leaf Program, its
    `leaf_partition_of` query and `h1window.leaf_reads_file`'s served file with `h1window.leaf_known_ports`."""
    prog, digest, layout, shape = w["prog"], w["digest"], w["layout"], w["shape"]
    (r, c), roots, rng = LEAF, _roots(w), random.Random(0)
    proved, statements, produced, openings = {}, [], [], {}
    by_call: dict[int, list[int]] = {}
    for u in units:
        call, _t, k = Cr.leaf_call(layout, u)
        if f"k{k}" in files:
            by_call.setdefault(call, []).append(u)
            continue
        name = LL.served(PREFIX, LL.PARTIALS, u)
        tree = PR.commit(digest, name, [[rng.randrange(1 << 16) for _ in range(2 * LL.words(k))] for _ in range(LL.lanes(r, c))],
                         DRAW_ROOT)
        produced.append(tree.entry())
        openings[name] = PR.rows_of(tree)
        at = HW.place(prog, u, STEPS, served=True)
        mine = {**roots, name: tree.root.hex()}
        partials = {x: y for x, y in tree.entry().items() if x not in ("value", "root")}
        for kind, ln in LL.statements(r, c):
            known, kops = HW.leaf_known_ports(prog, kind, ln, u=u, registration=DRAW_ROOT, salt=bytes.fromhex(SALT),
                                              roots=(bytes.fromhex(roots[at[LL.A][0]]), bytes.fromhex(roots[at[LL.B][0]])),
                                              prog=digest)
            reg = HW.leaf_reads_file(prog, u, kind, ln, mine, partials, known, prog=digest, window=WINDOW, steps=STEPS,
                                     served=True)
            seen = {p: [list(kops[e["value"]][0])] for p, e in reg["ports"].items() if e["value"] in kops}
            seen.update({p: [[e["reads"][0], tree.bcs[e["reads"][0]].hex()]] for p, e in reg["ports"].items()
                         if e["value"] == name})
            base = to / f"leaf-{u}-{kind}-{ln}"
            (base / "session").mkdir(parents=True)
            (base / "circuit.txt").write_bytes(circuit(f"{kind}{k}"))
            (base / "pub.bin").write_bytes(f"{u} {kind} {ln}\n".encode())
            leaf = leaf_desc(kind, k)
            partition = {"format": Cr.PARTITION_FORMAT, "program": _sha(leaf),
                         "query": {"name": Cr.QUERY[0], "version": Cr.QUERY[1], "params": {"templates": [unit_id(kind, k)]}}}
            proved[str(base / "pub.bin")] = {"circuit": circuit(f"{kind}{k}"), "program": leaf, "partition": partition,
                                             "registered": reg, "units": (0,), "openings": seen}
            statements.append({"unit": u, "statement": [kind, ln], "circuit": str(base / "circuit.txt"),
                               "public": str(base / "pub.bin"), "sessions": [str(base / "session")]})
    ks = {name: k for name, k, _n in layout["linears"]}
    for call, us in sorted(by_call.items()):
        k = ks[layout["calls"][call][0]]
        reg, why = Cr.registered_file(digest, layout, shape, call, {v: [x] for v, x in roots.items()}, WINDOW, STEPS, SALT)
        assert why is None, why
        base = to / f"call-{call}"
        (base / "session").mkdir(parents=True)
        (base / "circuit.txt").write_bytes(circuit(f"k{k}"))
        (base / "pub.bin").write_bytes(f"call {call}\n".encode())
        proved[str(base / "pub.bin")] = {"circuit": circuit(f"k{k}"), "program": w["raw"],
                                         "partition": Cr.partition_of(digest, shape), "registered": reg,
                                         "units": tuple(us), "openings": {}}
        statements.insert(0, {"call": call, "circuit": str(base / "circuit.txt"), "public": str(base / "pub.bin"),
                              "sessions": [str(base / "session")]})
    rec = PR.record(DRAW_ROOT, COIN["coin"], produced, openings)
    return ({"leaf_programs": {f"{kind}{k}": leaf_desc(kind, k).hex() for k in (384, 416) for kind in (LL.LANE, LL.COMBINE)},
             "produced": rec, "statements": statements}, proved)


def runner(proved: dict, tamper=None):
    """The stand-in verifier of record: accepts each session exactly when it was handed the files its statement was
    proved against, and reports the program, registered finding, units and openings (each `tamper`ed per statement)."""
    def run(proof: PF.Proof, *, circuit_sha512, program, partition, registered):
        want = proved[str(proof.public)]
        ok = (_sha(proof.circuit.read_bytes()) == circuit_sha512 == _sha(want["circuit"]) and program == want["program"]
              and partition == want["partition"] and registered == want["registered"])
        f = {"session": str(proof.sessions[0]), "accepted": ok, "why": "" if ok else "the proof does not verify",
             "program": _sha(program), "units": want["units"] if ok else (),
             "registered": {p: {"value": e["value"], "root": e["root"]} for p, e in registered["ports"].items()} if ok else {},
             "openings": want["openings"]}
        if tamper is not None:
            f = tamper(str(proof.public), f)
        return [Opened(**f)], None
    return run


def verified(w: dict, evidence, files: dict, run, units=DRAWN, entries=None, draw=DRAW):
    return Cr._verified(evidence, files, w["raw"], w["layout"], w["entries"] if entries is None else entries, WINDOW,
                        units, STEPS, run, SALT, draw)


def _leaf_statements(evidence: dict, u: int | None = None) -> list[int]:
    return [i for i, s in enumerate(evidence["statements"]) if "unit" in s and (u is None or s["unit"] == u)]


def test_each_drawn_leaf_proved_as_its_lanes_and_combine_is_one_accepted_outcome(world, lanes, tmp_path):
    """Every drawn leaf at a lanes-form K, its four lanes and its combine verified against its leaf Programs with credit's
    own files, is one accepted outcome under the window's program, its registered finding its statements' reads of the
    window's tables (each of its tables' values, at the window's roots), which `_proved` holds to the draw. Beside a
    tile-form K's call statement, verified as before."""
    for files in (files_of(), files_of((384,), (416,))):
        evidence, proved = prove(world, tmp_path / str(len(files)), files)
        got, why = verified(world, evidence, files, runner(proved))
        assert why is None, why
        assert sorted(o["unit"] for o in got) == DRAWN and Cr._proved(DRAWN, got) is None
        roots = _roots(world)
        for o in got:
            assert o["verdict"] == "accept" and o["program"] == world["digest"]
            if Cr.form_of(files, Cr.leaf_k(world["layout"], o["unit"])) == "lanes":
                assert o["registered"] and all(roots[e["value"]] == e["root"] for e in o["registered"].values())


def test_credit_builds_the_provers_served_leaf_files(world):
    """`credit.leaf_registered_file`, from the descriptor alone, is `h1window.leaf_reads_file` at the served place with
    `h1window.leaf_known_ports`, port for port, and its known openings those ports' trees' row 0."""
    prog, digest, (r, c) = world["prog"], world["digest"], LEAF
    roots = _roots(world)
    for u in DRAWN:
        k = Cr.leaf_k(world["layout"], u)
        name = LL.served(PREFIX, LL.PARTIALS, u)
        at = HW.place(prog, u, STEPS, served=True)
        assert Cr.leaf_place(world["layout"], world["shape"], u, STEPS) == at and at[LL.PARTIALS] == (name, 0)
        partials = PR.entry(digest, name, LL.lanes(r, c), 2 * LL.words(k), DRAW_ROOT)
        for kind, ln in LL.statements(r, c):
            known, kops = HW.leaf_known_ports(prog, kind, ln, u=u, registration=DRAW_ROOT, salt=bytes.fromhex(SALT),
                                              roots=(bytes.fromhex(roots[at[LL.A][0]]), bytes.fromhex(roots[at[LL.B][0]])),
                                              prog=digest)
            want = HW.leaf_reads_file(prog, u, kind, ln, {**roots, name: "ee" * 64}, partials, known, prog=digest,
                                      window=WINDOW, steps=STEPS, served=True)
            got, why = Cr.leaf_registered_file(digest, world["layout"], world["shape"], u, (kind, ln),
                                               {v: [x] for v, x in roots.items()}, WINDOW, STEPS, SALT, DRAW_ROOT,
                                               {name: "ee" * 64})
            assert why is None, why
            assert got == (want, {v: [tuple(op) for op in ops] for v, ops in kops.items()})


def test_a_leaf_with_a_statement_missing_or_no_lanes_evidence_is_refused(world, lanes, tmp_path):
    files = files_of()
    evidence, proved = prove(world, tmp_path, files)
    i = _leaf_statements(evidence, 6)[-1]
    short = {**evidence, "statements": [s for j, s in enumerate(evidence["statements"]) if j != i]}
    assert verified(world, short, files, runner(proved))[1] == ("missing", (
        "drawn leaf 6 has no statement ['combine', None]: every drawn leaf's lanes and combine are proved and verified, "
        "and a leaf with one missing refuses the window (no partial credit)"))
    got, why = verified(world, [], files, runner(proved))
    assert got is None and why[0] == "missing" and why[1].startswith("drawn leaf 1 is at a lanes-form K")


def test_leaf_statements_the_draw_did_not_ask_for_are_refused(world, lanes, tmp_path):
    """A leaf statement proved twice (a retry), one of a unit no drawn leaf, and one that is no lane or combine of the
    tile: each refuses the window, whatever its other statements' verdicts."""
    files = files_of()
    evidence, proved = prove(world, tmp_path, files)
    s = evidence["statements"][_leaf_statements(evidence)[0]]
    for extra, why in (
            (s, "drawn leaf 1's statement ['lane', 0] is proved twice: each statement is proved and verified once, and a "
                "retry never reproves it"),
            ({**s, "unit": 0}, "unit 0 has a leaf statement and is no drawn lanes-form leaf: only the draw's leaves at a K "
                               "committed as lanes are proved as lanes"),
            ({**s, "statement": ["lane", 4]}, "drawn leaf 1's statement ['lane', 4] is none of leaf_lanes.statements(8, "
                                              "16): a lane with its index or the combine")):
        got = verified(world, {**evidence, "statements": evidence["statements"] + [extra]}, files, runner(proved))
        assert got == (None, ("outcome", why))


def test_a_rejected_leaf_session_or_one_proving_other_units_is_refused(world, lanes, tmp_path):
    files = files_of()
    evidence, proved = prove(world, tmp_path, files)
    first = evidence["statements"][_leaf_statements(evidence, 3)[2]]["public"]
    Path(first).parent.joinpath("circuit.txt").write_bytes(b"another circuit")
    got, why = verified(world, evidence, files, runner(proved))
    assert got is None and why[0] == "failed" and why[1].startswith("drawn leaf 3's lane416 session ")
    Path(first).parent.joinpath("circuit.txt").write_bytes(circuit("lane416"))
    for change in ({"units": (0, 1)}, {"program": "ab" * 64}):
        got, why = verified(world, evidence, files, runner(proved, lambda p, f: {**f, **change} if p == first else f))
        assert got is None and why[0] == "proof" and "checkFull" in why[1], why


def test_a_leaf_program_other_than_the_committed_one_is_refused(world, lanes, tmp_path, monkeypatch):
    files = files_of()
    evidence, proved = prove(world, tmp_path, files)
    other = {**evidence, "leaf_programs": {**evidence["leaf_programs"], "combine384": b"another".hex()}}
    got, why = verified(world, other, files, runner(proved))
    assert got is None and why == ("circuit", "the window's combine384 leaf Program is not the one the period-open "
                                              "committed (leaf_programs combine384): verify each lanes-form statement "
                                              "against it")
    monkeypatch.setattr(Cr, "leaf_of", lambda *_: False)
    got, why = verified(world, evidence, files, runner(proved))
    assert got is None and why[0] == "circuit" and "is not h1window.leaf_program's at K = 384" in why[1]


def test_a_leaf_opening_other_values_roots_or_rows_is_refused(world, lanes, tmp_path):
    """Each is ``registration``: a session reporting another root for a read; a produced record under another draw coin
    or missing a leaf's partials (its P- codes); a call whose receipts the window doesn't register; a session reporting
    another opening of a partials row, or none (P-shared: a verdict without its openings, as `proofs.Finding` is today,
    fails closed)."""
    files = files_of()
    evidence, proved = prove(world, tmp_path, files)
    first = evidence["statements"][_leaf_statements(evidence, 1)[1]]["public"]

    def other_root(p, f):
        if p != first:
            return f
        port = sorted(f["registered"])[0]
        return {**f, "registered": {**f["registered"], port: {**f["registered"][port], "root": "01" * 64}}}

    got, why = verified(world, evidence, files, runner(proved, other_root))
    assert got is None and why[0] == "registration" and "opened its ports at other values or roots" in why[1]
    rec = evidence["produced"]
    for bad, codes in (({**rec, "draw": "c1" * 64}, "P-draw"),
                       ({**rec, "produced": rec["produced"][1:]}, "P-values")):
        got, why = verified(world, {**evidence, "produced": bad}, files, runner(proved))
        assert got is None and why[0] == "registration" and why[1].startswith(f"the window's produced record ({PR.FORMAT}) "
                                                                              f"fails {codes}"), why
    receipt = HW.place(world["prog"], 6, STEPS)[LL.RECEIPT][0]
    got, why = verified(world, evidence, files, runner(proved), entries=[e for e in world["entries"] if e["value"] != receipt])
    assert got is None and why[0] == "registration" and why[1].startswith("drawn leaf 6's combine reads its call's receipts")
    partials = LL.served(PREFIX, LL.PARTIALS, 3)

    def other_opening(p, f):
        ops = {port: [[row, ("0" if bc[0] != "0" else "1") + bc[1:]] for row, bc in rows] if f["registered"][port]["value"]
               == partials else rows for port, rows in f["openings"].items()}
        return {**f, "openings": ops}

    for tamper in (other_opening, lambda p, f: {**f, "openings": None}):
        got, why = verified(world, evidence, files, runner(proved, tamper))
        assert got is None and why[0] == "registration" and "(P-shared)" in why[1], why
    assert verified(world, evidence, files, runner(proved), draw={"coin": COIN})[1][0] == "registration"


def test_a_drawn_leaf_outside_leaflanessound_is_refused_lanes_shape(world, tmp_path, monkeypatch):
    """At the served proof's reach only: a tile other than 32 × 32 is ``lanes-shape`` before anything is verified (the
    stand-in's 8 × 16 here), and so is a K the tile check refuses, one below `h1zy.LEAST_K`, a chunked K and a J10-R
    template, each after building its statement set (J10-R's chunk lanes and combine v2 included)."""
    files = files_of()
    evidence, proved = prove(world, tmp_path, files)
    got, why = verified(world, evidence, files, runner(proved))
    assert got is None and why[0] == "lanes-shape" and why[1].startswith("drawn leaf 1: the window's 8 × 16 tile")
    tile = {"tile": [32, 32]}
    plain, j10r = {"tiles": {k: {"j10r": False} for k in (352, 384, 400, 14336, 4096)}}, {"tiles": {4096: {"j10r": True}}}
    assert Cr.lanes_shape(tile, plain, 384) is None
    assert "fails PouwWindow.Lanes.Tile.check" in Cr.lanes_shape(tile, plain, 400)
    assert "below LEAST_K = 384" in Cr.lanes_shape(tile, plain, 352)
    assert "chunk lanes each" in Cr.lanes_shape(tile, plain, 14336)
    assert "J10-R's (PouwH1Tile_v5" in Cr.lanes_shape(tile, j10r, 4096)
    assert "chunk lanes each" in Cr.lanes_shape(tile, {"tiles": {14336: {"j10r": True}}}, 14336)
    assert len(LL.statements(32, 32, j10r=14336)) > len(LL.statements(32, 32))


def _ledger(files) -> list[str]:
    return [json.dumps({"kind": "period-open", "body": {"period": "p", Cr.CIRCUIT_FILES: files}})]


def test_each_k_is_committed_in_one_form_with_its_leaf_programs():
    """The period-open's circuit files: a K is ``k<K>`` or ``lane<K>`` and ``combine<K>``, never both or one of the two,
    and `LEAF_PROGRAMS` names exactly the lanes files' leaf Programs."""
    files = files_of((384,), (416,))
    assert Cr.committed_files(_ledger(files), "p") == (files, None)
    assert Cr.committed_files(_ledger(files_of(())), "p") == (files_of(()), None)
    assert (Cr.form_of(files, 384), Cr.form_of(files, 416), Cr.form_of(files, 448)) == ("lanes", "tile", None)
    progs = files[Cr.LEAF_PROGRAMS]
    for bad, why in (({**files, "k384": "ab" * 64}, "commits both k384 and lane384 + combine384"),
                     ({x: v for x, v in files.items() if x != "combine384"}, "commits ['lane'] of its lanes form's two"),
                     ({x: v for x, v in files.items() if x != Cr.LEAF_PROGRAMS}, "leaf_programs is not"),
                     ({**files, Cr.LEAF_PROGRAMS: {**progs, "lane416": "ab" * 64}}, "leaf_programs is not"),
                     ({**files_of(()), Cr.LEAF_PROGRAMS: {}}, "leaf_programs is not")):
        got, refused = Cr.committed_files(_ledger(bad), "p")
        assert got is None and refused[0] == "circuit" and why in refused[1], refused
        assert Cr.lanes_files(bad) is not None
    assert Cr.committed_files(_ledger({**files, "lane384": "zz"}), "p")[0] is None


@pytest.mark.slow
def test_credit_reads_the_real_leaf_programs_off_their_descriptors():
    """At the real zy unit (16 × 8, K = 384): `leaf_unit_id` reads K's lane and combine unit ids off the window's
    descriptor (each its descriptor id, as the partition names templates), and `leaf_of` accepts `h1window.leaf_program`'s descriptor for its kind and K and no other. What the
    stand-in can't show is that credit's descriptor-level check matches the real derivation: the window traces the tile's
    seed and noise circuits (about 30 s), and each leaf Program's descriptor takes about 6 s."""
    k, tile = 384, (16, 8)
    prog = HW.program((("q_proj", k, 16), ("k_proj", k, 8)), (("q_proj", 0), ("k_proj", 0)), 16, tile, prefix="h1/w/")
    desc = codec.encode_program(prog)
    shape = Cr.shape_of(codec.canonical_json({x: v for x, v in desc.items() if x != "annotations"}))
    leafs = {}
    for kind in (LL.LANE, LL.COMBINE):
        assert Cr.leaf_unit_id(shape, k, kind) == codec.descriptor_id(HW.leaf_unit(prog, k, kind))
        d = codec.encode_program(HW.leaf_program(prog, k, kind)[0])
        leafs[kind] = codec.canonical_json({x: v for x, v in d.items() if x != "annotations"})
    for kind, other in ((LL.LANE, LL.COMBINE), (LL.COMBINE, LL.LANE)):
        unit = Cr.leaf_unit_id(shape, k, kind)
        assert Cr.leaf_of(leafs[kind], unit, k, kind)
        assert not Cr.leaf_of(leafs[other], unit, k, kind) and not Cr.leaf_of(leafs[kind], unit, k + 32, kind)
        assert not Cr.leaf_of(leafs[kind], Cr.leaf_unit_id(shape, k, other), k, kind)
