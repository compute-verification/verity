"""`credit`'s lanes form (`leaf_lanes`): a period-open committing a K as ``lane<K>`` and ``combine<K>``, with each leaf
Program's digest (`credit.LEAF_PROGRAMS`), is refused by name (``lanes-form``), as `lean/check`'s difftest holds the Lean
to (`test_verified_difftest`); the form's names, which `unit_lowering` and vLLM's H1 builds read (`form_of`,
`lanes_files`, `leaf_of`, `leaf_unit_id`), with `leaf_of` and `leaf_unit_id` held to the real `h1window.leaf_program`."""
from __future__ import annotations

import hashlib
import json

import pytest
from verity.circuits import codec
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import h1window as HW
from verity.protocols.resource_accounting.compute.pouw.verifier import credit as Cr
from verity.protocols.resource_accounting.compute.pouw.verifier import leaf_lanes as LL


def _sha(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


def files_of(lanes_ks=(384, 416), tile_ks=()) -> dict:
    out = {f"k{k}": _sha(f"circuit file k{k}".encode()) for k in tile_ks}
    for k in lanes_ks:
        out.update({f"{kind}{k}": _sha(f"circuit file {kind}{k}".encode()) for kind in (LL.LANE, LL.COMBINE)})
    if lanes_ks:
        out[Cr.LEAF_PROGRAMS] = {f"{kind}{k}": _sha(f"leaf Program of {kind} at K = {k}".encode())
                                 for k in lanes_ks for kind in (LL.LANE, LL.COMBINE)}
    return {**out, Cr.LIVE: _sha(b"live")}


def _ledger(files) -> list[str]:
    return [json.dumps({"kind": "period-open", "body": {"period": "p", Cr.CIRCUIT_FILES: files}})]


def test_a_period_open_in_the_lanes_form_is_refused_by_name():
    """A period-open committing a K in the lanes form, whole or broken, is refused by name (``lanes-form``); one in the
    tile form alone is read. `form_of` and `lanes_files` still name the form: a K is ``k<K>`` or ``lane<K>`` and
    ``combine<K>``, never both or one of the two, and `LEAF_PROGRAMS` names exactly the lanes files' leaf Programs."""
    files = files_of((384,), (416,))
    assert Cr.committed_files(_ledger(files_of((), (416,))), "p") == (files_of((), (416,)), None)
    assert (Cr.form_of(files, 384), Cr.form_of(files, 416), Cr.form_of(files, 448)) == ("lanes", "tile", None)
    assert Cr.lanes_files(files) is None
    progs = files[Cr.LEAF_PROGRAMS]
    for bad, why in (({**files, "k384": "ab" * 64}, "commits both k384 and lane384 + combine384"),
                     ({x: v for x, v in files.items() if x != "combine384"}, "commits ['lane'] of its lanes form's two"),
                     ({x: v for x, v in files.items() if x != Cr.LEAF_PROGRAMS}, "leaf_programs is not"),
                     ({**files, Cr.LEAF_PROGRAMS: {**progs, "lane416": "ab" * 64}}, "leaf_programs is not"),
                     ({**files_of(()), Cr.LEAF_PROGRAMS: {}}, "leaf_programs is not")):
        assert why in Cr.lanes_files(bad)
    for lanes in (files, *(bad for bad in ({**files, "k384": "ab" * 64}, {**files_of(()), Cr.LEAF_PROGRAMS: {}},
                                           {x: v for x, v in files.items() if x != "combine384"}))):
        assert Cr.committed_files(_ledger(lanes), "p") == (None, ("lanes-form", Cr.LANES_REFUSED))
    assert Cr.committed_files(_ledger({**files, "lane384": "zz"}), "p")[1][0] == "circuit"


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
