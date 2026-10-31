"""``layout``: Layout JSON read back byte for byte, each check refusing its mutation by name, and the digest a deployment
file binds."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from verity.errors import InvalidArtifact
from verity.ml.commitments import layout as L
from verity.ml.commitments.identity import identity_digest_sha512, tagged_sha512_prefix

LAYOUTS = Path(__file__).resolve().parent / "fixtures" / "layouts"
#: art:16f2faa0…'s file, FAMILIES' shared-row layout at 1 KiB rows, and its sha256
SHARED = LAYOUTS / "families-shared-1024b.layout.json", "15fd14be19dea4d4d70b18771d0e185881894c3d5e480db35a9dbe4eb5141cd8"
#: art:a61d745f…'s file, today's ``families/v1`` layout, whose gathers re-commit their parts' bytes, and its sha256
TODAY = LAYOUTS / "families-today.layout.json", "c3f312ce597f368cb01b7db9c8298fff1c9613d7bf90907d7e72a45190a1e175"
#: commit_slots' interface fixture: its "layout" is ``slots._layout_json(slots.edge_layout())``
EDGE = Path(__file__).resolve().parents[4] / "benchmarks" / "torch_gpu" / "commit_slots" / "fixtures" / "commit-slots-v1.json"

DIGESTS = {
    "edge": "eea4be19ce407d924d0ebe1c20eb42d9e050076898ca4a7d8ed2e8c3c7053168"
            "82ec2ffdb8bd40ae291158e0b1d235cce5a0fe5167fed32a1b41e94c8cbe0c42",
    "shared": "f8d6252c5933f22c8c276c911a7c22e84c7ea12d846d7de8f9ee90e312c68005"
              "3d41a525a4350c921f9cf12b80fc0dc9d43d3034c8b7a33f83bc069c1e656a9f",
    "today": "e6889bd14d5292ca67c42e92efde4c4fe623cfdbf915f68813d2447cc4d4fb1c"
             "ae66e51c35750056b289846c2660631ba3e1983dbe0c27dd4762f3f4657a9389",
}


def written(layout: L.Layout) -> bytes:
    """The file ``commit_slots/step.py export`` writes: ``_layout_json``'s output, compact, then a newline."""
    return (json.dumps(L.to_json(layout), separators=(",", ":")) + "\n").encode()


def read(fixture: tuple[Path, str]) -> tuple[bytes, L.Layout]:
    raw = fixture[0].read_bytes()
    assert hashlib.sha256(raw).hexdigest() == fixture[1]
    return raw, L.load(fixture[0])


def edge() -> dict:
    return json.loads(EDGE.read_text())["layout"]


def refused(doc: object, message: str) -> None:
    with pytest.raises(InvalidArtifact, match=re.escape(message)):
        L.of_json(doc)


def one(sources: list[int], *units: list) -> dict:
    return {"name": "t", "groups": [{"sources": sources, "units": list(units)}]}


# -- reading back

def test_the_shared_layout_reads_back() -> None:
    raw, lay = read(SHARED)
    assert written(lay) == raw
    assert len(lay.groups) == 54 and len({id(g) for g in lay.groups}) == 3
    assert sum(len(g.units) for g in lay.groups) == 21_857 and all(len(u) == 1 for g in lay.groups for u in g.units)
    assert lay.json_digest.hex() == DIGESTS["shared"]
    L.check_shared(lay)


@pytest.mark.slow
def test_todays_layout_reads_back_and_is_not_shared() -> None:
    """102 MB, 6,102,832 rows a slot: the one layout at today's scale."""
    raw, lay = read(TODAY)
    assert written(lay) == raw
    assert len(lay.groups) == 54 and len({id(g) for g in lay.groups}) == 3
    assert sum(len(g.units) for g in lay.groups) == 6_102_102
    assert sum(len(u) for g in lay.groups for u in g.units) == 6_102_832
    assert lay.json_digest.hex() == DIGESTS["today"]
    # the file's groups are canonical text already: only the two keys swap
    i = raw.index(b'","groups":')
    canonical = b'{"groups":' + raw[i + 11:-2] + b',"name":' + raw[8:i + 1] + b"}"
    assert lay.json_digest == hashlib.sha512(tagged_sha512_prefix(L.JSON_TAG, len(canonical)) + canonical).digest()
    with pytest.raises(InvalidArtifact, match=re.escape("group 0 tensor 0: bytes 0..2 are committed twice")):
        L.check_shared(lay)


def test_the_edge_layout_round_trips() -> None:
    doc = edge()
    lay = L.of_json(doc)
    assert L.to_json(lay) == doc
    assert [len(g.units) for g in lay.groups] == [3, 2] and lay.unit_base == (0, 3)
    assert lay.json_digest.hex() == DIGESTS["edge"]


def test_equal_groups_are_one_object() -> None:
    doc = edge()
    doc["groups"] = [doc["groups"][0], copy.deepcopy(doc["groups"][1]), copy.deepcopy(doc["groups"][0])]
    lay = L.of_json(doc)
    assert lay.groups[0] is lay.groups[2] and lay.groups[0] is not lay.groups[1]
    assert L.to_json(lay) == doc


# -- the digest

def test_the_digest_is_the_canonical_json() -> None:
    for doc in (edge(), json.loads(SHARED[0].read_bytes())):
        assert L.of_json(doc).json_digest == identity_digest_sha512(L.JSON_TAG, doc)


def test_the_digest_ignores_formatting(tmp_path: Path) -> None:
    doc = edge()
    reordered = {"groups": [{"units": g["units"], "sources": g["sources"]} for g in doc["groups"]], "name": doc["name"]}
    (tmp_path / "x.json").write_text(json.dumps(reordered, indent=2))
    assert L.load(tmp_path / "x.json").json_digest.hex() == DIGESTS["edge"]


def test_the_digest_binds_the_parts() -> None:
    """Two rows of the same size swapped: the rows' sizes, all that ``slots``' own digest binds, are unchanged."""
    doc = json.loads(SHARED[0].read_bytes())
    sizes = [[r.bits for r in u] for g in L.of_json(doc).groups for u in g.units]
    units = doc["groups"][0]["units"]
    units[0], units[1] = units[1], units[0]
    lay = L.of_json(doc)
    L.check_shared(lay)
    assert [[r.bits for r in u] for g in lay.groups for u in g.units] == sizes
    assert lay.json_digest.hex() != DIGESTS["shared"]


# -- the form and slots.Group's checks

def test_a_unit_is_at_most_unit_max_bits() -> None:
    n = L.UNIT_MAX_BITS // 8
    L.of_json(one([n], [[[0, 0, n]]]))
    refused(one([n + 1], [[[0, 0, n]], [[0, n, 1]]]), f"group 0 unit 0: a unit's rows are at most {L.UNIT_MAX_BITS} bits, "
                                                      f"not {L.UNIT_MAX_BITS + 8}")


@pytest.mark.parametrize("part", [[0, 721, 241], [1, 0, 1], [0, -1, 1], [0, 0, 0], [-1, 0, 1]])
def test_a_part_is_inside_its_tensor(part: list[int]) -> None:
    doc = edge()
    doc["groups"][0]["units"][2][0][0] = part
    refused(doc, f"group 0 unit 2 row 0: part {L.Part(*part)} is outside the group's tensors (961,)")


def test_units_and_rows_are_not_empty() -> None:
    refused(one([4], [[[0, 0, 4]]], []), "group 0 unit 1: a unit has at least one row")
    refused(one([4], [[[0, 0, 4]], []]), "group 0 unit 0 row 1: a row has at least one byte")


@pytest.mark.parametrize("part", [[0, 0, 1.0], [0, 0, True], [0, 0], [0, 0, 1, 0], {"source": 0}])
def test_a_part_is_three_integers(part: object) -> None:
    doc = edge()
    doc["groups"][0]["units"][0][0][0] = part
    refused(doc, "group 0 unit 0 row 0: a part is [source, offset, nbytes], integers")


def test_a_float_never_passes_as_an_earlier_groups_integer() -> None:
    doc = edge()
    later = copy.deepcopy(doc["groups"][0])
    later["units"][0][0][0] = [0, 0, 1.0]
    assert later == doc["groups"][0]
    doc["groups"].append(later)
    refused(doc, "group 2 unit 0 row 0: a part is [source, offset, nbytes], integers")


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.update(digest="x"), 'a layout is {"name", "groups"}'),
    (lambda d: d.pop("name"), 'a layout is {"name", "groups"}'),
    (lambda d: d.update(name=7), "a layout's name is a string"),
    (lambda d: d.update(name="\ud800"), "a layout's name is UTF-8 text"),
    (lambda d: d.update(groups={}), "a layout's groups are a list"),
    (lambda d: d["groups"][1].update(scale=4), 'group 1: a group is {"sources", "units"}'),
    (lambda d: d["groups"][1].update(sources=(64, 4)), "group 1: a group's sources and units are lists"),
    (lambda d: d["groups"][1].update(sources=[64, -4]), "group 1: a tensor's bytes are a non-negative integer"),
    (lambda d: d["groups"][1].update(sources=[64, 4.0]), "group 1: a tensor's bytes are a non-negative integer"),
    (lambda d: d["groups"][1]["units"].append({}), "group 1 unit 2: a unit is a list of rows"),
    (lambda d: d["groups"][1]["units"][0].append(7), "group 1 unit 0 row 1: a row is a list of parts"),
])
def test_anything_else_is_refused(mutate, message: str) -> None:
    doc = edge()
    mutate(doc)
    refused(doc, message)


def test_load_refuses_a_repeated_key_and_non_json(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text('{"name": "a", "name": "b", "groups": []}')
    with pytest.raises(InvalidArtifact, match="a layout's JSON repeats a key"):
        L.load(p)
    p.write_text('{"name": "a", "groups": [')
    with pytest.raises(InvalidArtifact, match="not JSON"):
        L.load(p)


# -- shared rows

def test_a_shared_row_has_at_most_two_parts() -> None:
    lay = L.of_json(one([3], [[[0, 0, 1], [0, 1, 1], [0, 2, 1]]]))
    with pytest.raises(InvalidArtifact, match=re.escape("group 0 unit 0 row 0: a shared row has at most 2 parts, not 3")):
        L.check_shared(lay)


@pytest.mark.parametrize("doc, message", [
    (one([4], [[[0, 0, 2]]]), "group 0 tensor 0: bytes 2..4 are in no row"),
    (one([4], [[[0, 0, 1]]], [[[0, 2, 2]]]), "group 0 tensor 0: bytes 1..2 are in no row"),
    (one([4, 3], [[[0, 0, 4]]]), "group 0 tensor 1: bytes 0..3 are in no row"),
    (one([4], [[[0, 0, 4]]], [[[0, 1, 2]]]), "group 0 tensor 0: bytes 1..3 are committed twice"),
    (one([4], [[[0, 0, 2], [0, 1, 2]]]), "group 0 tensor 0: bytes 1..2 are committed twice"),
    (one([8, 4], [[[0, 0, 4], [1, 0, 4]]], [[[0, 4, 4], [1, 0, 2]]], [[[1, 2, 2]]]), "group 0 tensor 1: bytes 0..2 are committed twice"),
    (edge(), "group 1 tensor 1: bytes 1..4 are committed twice"),
])
def test_every_byte_is_in_one_row(doc: dict, message: str) -> None:
    lay = L.of_json(doc)
    with pytest.raises(InvalidArtifact, match=re.escape(message)):
        L.check_shared(lay)


def test_a_scale_is_whole_in_each_of_its_chunks_rows() -> None:
    L.check_shared(L.of_json(one([8, 4], [[[0, 0, 4], [1, 0, 4]]], [[[0, 4, 4], [1, 0, 4]]])))


def test_a_regathered_row_is_not_shared() -> None:
    doc = json.loads(SHARED[0].read_bytes())
    doc["groups"][1]["units"].append([[[1, 0, 16]]])
    with pytest.raises(InvalidArtifact, match=re.escape("group 1 tensor 1: bytes 0..16 are committed twice")):
        L.check_shared(L.of_json(doc))


# -- producers' windows

def test_a_window_is_its_row_and_offset() -> None:
    lay = L.of_json(edge())
    assert lay.window(0, 0, 337 + 127, 128) == (1, 1, 0)
    assert lay.window(0, 0, 337 + 127 + 5, 100) == (1, 1, 5)
    assert lay.window(1, 1, 0, 4) == (3, 0, 64)
    assert lay.window(1, 1, 1, 3) == (3, 0, 65)
    assert lay.window(1, 1, 2, 2) == (3, 0, 66)
    _, shared = read(SHARED)
    assert shared.unit_base[:3] == (0, 32, 451)
    assert shared.window(1, 0, 0, 512) == (32, 0, 0) and shared.window(1, 15, 0, 512) == (32, 0, 512)
    assert shared.window(1, 1, 0, 16) == (33, 0, 0)
    assert shared.window(1, 1, 1024 + 16, 16) == (34, 0, 16)
    assert shared.window(2, 1, 2048, 512) == (454, 0, 0)
    L.check_shared(shared, [(1, 1, 0, 1024), (2, 1, 2048, 512), (1, 15, 0, 512)])


@pytest.mark.parametrize("window, message", [
    ((1, 1, 1000, 48), "group 1 tensor 1: bytes 1000..1048 aren't a window of one row"),
    ((1, 1, 32768 - 8, 16), "group 1 tensor 1: bytes 32760..32776 aren't a window of one row"),
    ((1, 999, 0, 1), "group 1 tensor 999: bytes 0..1 aren't a window of one row"),
    ((1, 0, 0, 0), "group 1 tensor 0: bytes 0..0 aren't a window of one row"),
    ((54, 0, 0, 1), "the layout has no group 54"),
])
def test_a_producers_output_is_a_window_of_one_row(window: tuple[int, int, int, int], message: str) -> None:
    _, shared = read(SHARED)
    with pytest.raises(InvalidArtifact, match=re.escape(message)):
        L.check_shared(shared, [window])
