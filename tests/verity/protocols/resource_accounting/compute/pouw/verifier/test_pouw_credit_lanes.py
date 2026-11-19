"""`credit`'s lanes form (`leaf_lanes`): at a K the period-open commits as ``lane<K>`` and ``combine<K>``, credit
still samples the window's leaves, and each sampled leaf is proved as `leaf_lanes.statements` (R/4)(C/8) lanes and the
combine, its partials ``<prefix>partials@<u>`` held to one produced record under the window's sample root. Every leaf
statement reads an A, B or partials row in full, which proofs' verifier cannot mark, so credit refuses each at its
``--registered`` file (`credit.leaf_registered_file`) and runs the verifier on none (`never`). The refusals before it are
shown here: ``missing``, ``outcome``, ``circuit``, ``registration`` and ``lanes-shape``.

The window is `test_circuit_h1window`'s stand-in tile at 8 × 16 (four lanes) and K = 384 and 416, so `credit.LANES_TILES`
is widened to it; the 32 × 32 refusal is shown on its own. The stand-in's unit is not h1zy's, so its leaf Programs are
stand-in bytes that `leaf_of` is replaced to accept; the slow test holds `leaf_of` and `leaf_unit_id` to the real
`h1window.leaf_program`."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

import test_circuit_h1window as h1
from verity.circuits import codec
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import h1window as HW
from verity.protocols.resource_accounting.compute.pouw.verifier import credit as Cr
from verity.protocols.resource_accounting.compute.pouw.verifier import leaf_lanes as LL
from verity.protocols.proofs.spot.verifier.one_stage import produced as PR
from verity.protocols.proofs.spot.prover import produced as PP
from verity.protocols.proofs.spot.verifier.one_stage import proofs as PF

LEAF = (8, 16)
LINEARS = (("q_proj", 384, 32), ("k_proj", 384, 16), ("down_proj", 416, 16))
STEPS, PREFIX = 2, "h1/w/"
SALT, WINDOW, SAMPLE_ROOT = "5a" * 64, "cd" * 64, "d7" * 64
COIN = {"after": SAMPLE_ROOT, "coin": "c0" * 64, "epoch": 0, "index": 0, "label": "draw", "path": [], "salt": "00" * 64}
SAMPLE = {"coin": COIN, "root": SAMPLE_ROOT}
#: units 1 and 6 of K = 384's calls 0 and 4, unit 3 of K = 416's call 2
SAMPLED = [1, 3, 6]


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


def prove(w: dict, to: Path, units=SAMPLED) -> dict:
    """The window's lanes evidence as an honest prover delivers it: each sampled lanes-form leaf's partials committed under
    the sample root, and its lanes' and combine's statements."""
    digest, layout = w["digest"], w["layout"]
    (r, c), rng = LEAF, random.Random(0)
    statements, produced, openings = [], [], {}
    for u in units:
        _call, _t, k = Cr.leaf_call(layout, u)
        name = LL.served(PREFIX, LL.PARTIALS, u)
        tree = PP.commit(digest, name, [[rng.randrange(1 << 16) for _ in range(2 * LL.words(k))] for _ in range(LL.lanes(r, c))],
                         SAMPLE_ROOT)
        produced.append(tree.entry())
        openings[name] = PR.rows_of(tree)
        for kind, ln in LL.statements(r, c):
            base = to / f"leaf-{u}-{kind}-{ln}"
            (base / "session").mkdir(parents=True)
            (base / "circuit.txt").write_bytes(circuit(f"{kind}{k}"))
            (base / "pub.bin").write_bytes(f"{u} {kind} {ln}\n".encode())
            statements.append({"unit": u, "statement": [kind, ln], "circuit": str(base / "circuit.txt"),
                               "public": str(base / "pub.bin"), "sessions": [str(base / "session")]})
    rec = PR.record(SAMPLE_ROOT, COIN["coin"], produced, openings)
    return {"leaf_programs": {f"{kind}{k}": leaf_desc(kind, k).hex() for k in (384, 416) for kind in (LL.LANE, LL.COMBINE)},
            "produced": rec, "statements": statements}


def never(_proof: PF.Proof, **_kw):
    raise AssertionError("the verifier ran on a leaf statement")


def verified(w: dict, evidence, files: dict, units=SAMPLED, entries=None, sample=SAMPLE):
    return Cr._verified(evidence, files, w["raw"], w["layout"], w["entries"] if entries is None else entries, WINDOW,
                        units, STEPS, never, SALT, sample)


def _leaf_statements(evidence: dict, u: int | None = None) -> list[int]:
    return [i for i, s in enumerate(evidence["statements"]) if "unit" in s and (u is None or s["unit"] == u)]


def test_credit_refuses_every_leaf_statement_at_its_registered_file(world):
    """`credit.leaf_registered_file` places each sampled leaf at its served place, as `h1window.place` does, and refuses
    each of its statements (``registration``): every one reads an A, B or partials row in full, which a
    `registered.READS_V2` port cannot mark (`PouwWindow.Lanes.fullHole`)."""
    prog, digest, (r, c) = world["prog"], world["digest"], LEAF
    roots = _roots(world)
    for u in SAMPLED:
        name = LL.served(PREFIX, LL.PARTIALS, u)
        at = HW.place(prog, u, STEPS, served=True)
        assert Cr.leaf_place(world["layout"], world["shape"], u, STEPS) == at and at[LL.PARTIALS] == (name, 0)
        for kind, ln in LL.statements(r, c):
            got, why = Cr.leaf_registered_file(digest, world["layout"], world["shape"], u, (kind, ln),
                                               {v: [x] for v, x in roots.items()}, WINDOW, STEPS, SALT, SAMPLE_ROOT,
                                               {name: "ee" * 64})
            assert got is None and why[0] == "registration" and "are read in full" in why[1], why


def test_a_leaf_with_no_statement_or_no_lanes_evidence_is_refused(world, lanes, tmp_path):
    files = files_of()
    evidence = prove(world, tmp_path)
    got, why = verified(world, {**evidence, "statements": []}, files)
    assert got is None and why[0] == "missing" and why[1].startswith("drawn leaf 1 has no statement ") and \
        why[1].endswith("a leaf with one missing refuses the window (no partial credit)"), why
    got, why = verified(world, [], files)
    assert got is None and why[0] == "missing" and why[1].startswith("drawn leaf 1 is at a lanes-form K")


def test_leaf_statements_the_sample_did_not_ask_for_are_refused(world, lanes, tmp_path):
    """A leaf statement of a unit no sampled leaf, and one that is no lane or combine of the tile: each refuses the
    window, ahead of the statements after it."""
    files = files_of()
    evidence = prove(world, tmp_path)
    s = evidence["statements"][_leaf_statements(evidence)[0]]
    for extra, why in (
            ({**s, "unit": 0}, ("unit 0 has a leaf statement and is no drawn lanes-form leaf: only the draw's leaves at a K "
                                "committed as lanes are proved as lanes")),
            ({**s, "statement": ["lane", 4]}, ("drawn leaf 1's statement ['lane', 4] is none of leaf_lanes.statements(8, "
                                               "16): a lane with its index or the combine"))):
        got = verified(world, {**evidence, "statements": [extra] + evidence["statements"]}, files)
        assert got == (None, ("outcome", why))


def test_a_leaf_program_other_than_the_committed_one_is_refused(world, lanes, tmp_path, monkeypatch):
    files = files_of()
    evidence = prove(world, tmp_path)
    other = {**evidence, "leaf_programs": {**evidence["leaf_programs"], "combine384": b"another".hex()}}
    got, why = verified(world, other, files)
    assert got is None and why == ("circuit", ("the window's combine384 leaf Program is not the one the period-open "
                                               "committed (leaf_programs combine384): verify each lanes-form statement "
                                               "against it"))
    monkeypatch.setattr(Cr, "leaf_of", lambda *_: False)
    got, why = verified(world, evidence, files)
    assert got is None and why[0] == "circuit" and "is not h1window.leaf_program's at K = 384" in why[1]


def test_a_produced_record_receipts_or_sample_other_than_the_windows_are_refused(world, lanes, tmp_path):
    """Each is ``registration``: a produced record under another sampling coin or missing a leaf's partials (its P-
    codes); a call whose receipts the window doesn't register; a sample with no root."""
    files = files_of()
    evidence = prove(world, tmp_path)
    rec = evidence["produced"]
    for bad, codes in (({**rec, "draw": "c1" * 64}, "P-draw"),
                       ({**rec, "produced": rec["produced"][1:]}, "P-values")):
        got, why = verified(world, {**evidence, "produced": bad}, files)
        assert got is None and why[0] == "registration" and why[1].startswith(f"the window's produced record ({PR.FORMAT}) "
                                                                              f"fails {codes}"), why
    receipt = HW.place(world["prog"], 1, STEPS)[LL.RECEIPT][0]
    got, why = verified(world, evidence, files, entries=[e for e in world["entries"] if e["value"] != receipt])
    assert got is None and why[0] == "registration" and why[1].startswith("drawn leaf 1's combine reads its call's receipts")
    assert verified(world, evidence, files, sample={"coin": COIN})[1][0] == "registration"


def test_a_sampled_leaf_outside_leaflanessound_is_refused_lanes_shape(world, tmp_path, monkeypatch):
    """At the served proof's reach only: a tile other than 32 × 32 is ``lanes-shape`` before anything is verified (the
    stand-in's 8 × 16 here, and a 2 × 4 tile, which has no whole lane), and so is a K the tile check refuses, one below
    `h1zy.LEAST_K`, a chunked K and a J10-R template, the refusal read off the tile and K alone (J10-R's statement set,
    chunk lanes and combine v2, is still `leaf_lanes.statements`')."""
    files = files_of()
    evidence = prove(world, tmp_path)
    got, why = verified(world, evidence, files)
    assert got is None and why[0] == "lanes-shape" and why[1].startswith("drawn leaf 1: the window's 8 × 16 tile")
    tile = {"tile": [32, 32]}
    plain, j10r = {"tiles": {k: {"j10r": False} for k in (352, 384, 400, 14336, 4096)}}, {"tiles": {4096: {"j10r": True}}}
    assert Cr.lanes_shape(tile, plain, 384) is None
    assert "its 0 lanes wait" in Cr.lanes_shape({"tile": [2, 4]}, plain, 384)
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
