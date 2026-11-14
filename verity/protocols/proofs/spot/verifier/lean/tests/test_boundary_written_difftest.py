"""An output field's leaves, each written once by dense reads (`Spot.BoundaryWritten`, round 1's ``link-written``): the
closed form (`boundary_written`, `written` and `extent?`) and the unrolled reference (`boundary_written_unrolled`,
`writtenRef` and `extentRef?`), each held to one Python reference that lists every leaf of every read: reader index ``i``
of a read sources leaf ``base + sum(d[k] * step[k])`` over ``i``'s mixed-radix digits, the first the slowest, and its row
the ``row`` leaves from there. A read is dense when it has a leaf, no digit of radix 0 or a negative step but at radix 1,
and its leaves are a run from the least, each once; the field is written when every read is dense and their leaves
together are ``[0, leaves)``, each once (the reference is copied here rather than imported from
`verity.commitments.windows.boundary`: a verifier's tests import no prover code)."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D

PACKAGE = Path(__file__).resolve().parents[1]
PIECES = "sampled-proofs-difftest"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


def _int(x, what: str) -> int:
    if type(x) is not int:
        raise D.Reject(f"{what} is not an integer")
    return x


def _nat(x, what: str) -> int:
    if _int(x, what) < 0:
        raise D.Reject(f"{what} is not a natural")
    return x


def _read(j) -> tuple[int, list[int], list[int], int]:
    if not isinstance(j, dict) or not isinstance(j.get("radix"), list) or not isinstance(j.get("step"), list) \
            or len(j["radix"]) != len(j["step"]):
        raise D.Reject("a read is {base, radix, step, row}, radix and step lists of one length")
    return (_int(j.get("base"), "base"), [_nat(r, "a radix") for r in j["radix"]], [_int(s, "a step") for s in j["step"]],
            _nat(j.get("row"), "row"))


def leaves_of(base, radix, step, row) -> list[int]:
    out = []
    for i in range(math.prod(radix)):
        src, q = base, i
        for r, s in zip(reversed(radix), reversed(step)):
            src, q = src + q % r * s, q // r
        out += range(src, src + row)
    return out


def dense(base, radix, step, row) -> bool:
    ls = sorted(leaves_of(base, radix, step, row))
    return (bool(ls) and all(r >= 1 and (r == 1 or s >= 0) for r, s in zip(radix, step))
            and ls == list(range(ls[0], ls[0] + len(ls))))


def reference(j):
    if not isinstance(j, dict):
        raise D.Reject("leaves is not a natural")
    leaves = _nat(j.get("leaves"), "leaves")
    if not isinstance(j.get("reads"), list):
        raise D.Reject("reads is not a list")
    reads = [_read(r) for r in j["reads"]]
    ok = all(dense(*r) for r in reads) and sorted(x for r in reads for x in leaves_of(*r)) == list(range(leaves))
    return {"written": ok,
            "extents": [[min(leaves_of(*r)), min(leaves_of(*r)) + len(leaves_of(*r))] if dense(*r) else None
                        for r in reads]}


JUNK = st.one_of(st.none(), st.booleans(), st.floats(allow_nan=False, allow_infinity=False, width=16),
                 st.text(max_size=3), st.integers(-3, 3), st.lists(st.integers(-2, 3), max_size=2))


@st.composite
def dense_read(draw, base: int) -> dict:
    """A dense read from ``base``: a row of 1 to 3 leaves and up to three digits of radix 2 or 3, their steps the chain's,
    in a drawn order, with digits of radix 1 at any step among them."""
    row = draw(st.integers(1, 3))
    radix = draw(st.lists(st.integers(2, 3), max_size=3))
    step, run = [], row
    for r in radix:
        step.append(run)
        run *= r
    ds = list(zip(radix, step)) + [(1, draw(st.integers(-4, 4))) for _ in range(draw(st.integers(0, 1)))]
    ds = draw(st.permutations(ds))
    return {"base": base, "radix": [r for r, _ in ds], "step": [s for _, s in ds], "row": row}


@st.composite
def instances(draw, broken: bool = False) -> dict:
    """A field tiled by dense reads, then perhaps changed: a read's base, a step, a radix or a row moved by one or to 0, a
    read dropped or repeated, the field's leaves moved by one; ``broken``, a field or a read's field replaced by junk."""
    reads, at = [], 0
    for _ in range(draw(st.integers(0, 3))):
        r = draw(dense_read(at))
        reads.append(r)
        at += r["row"] * math.prod(r["radix"])
    reads = draw(st.permutations(reads))
    j = {"leaves": at, "reads": [dict(r, radix=list(r["radix"]), step=list(r["step"])) for r in reads]}
    change = draw(st.sampled_from(["none", "base", "step", "radix", "row", "drop", "repeat", "leaves"]))
    if reads and change in ("base", "row"):
        r = draw(st.sampled_from(j["reads"]))
        r[change] = max(0, r[change] + draw(st.sampled_from([-1, 1]))) if change == "row" else r[change] + draw(
            st.sampled_from([-1, 1]))
    elif change in ("step", "radix") and any(r["radix"] for r in j["reads"]):
        r = draw(st.sampled_from([r for r in j["reads"] if r["radix"]]))
        k = draw(st.integers(0, len(r["radix"]) - 1))
        r[change][k] = draw(st.sampled_from([0, r[change][k] - 1, r[change][k] + 1, -r[change][k]]))
        r["radix"][k] = max(0, r["radix"][k])
    elif reads and change == "drop":
        j["reads"].pop(draw(st.integers(0, len(j["reads"]) - 1)))
    elif reads and change == "repeat":
        j["reads"].append(dict(draw(st.sampled_from(j["reads"]))))
    elif change == "leaves":
        j["leaves"] = max(0, at + draw(st.sampled_from([-1, 1])))
    if broken:
        if j["reads"] and draw(st.booleans()):
            draw(st.sampled_from(j["reads"]))[draw(st.sampled_from(["base", "radix", "step", "row"]))] = draw(JUNK)
        else:
            j[draw(st.sampled_from(["leaves", "reads"]))] = draw(JUNK)
    return j


INPUTS = st.one_of(*[instances() for _ in range(6)], instances(broken=True), JUNK)
WRITTEN = D.Port(PIECES, "boundary_written", reference, INPUTS, package=PACKAGE, of_record="lean")
WRITTEN_UNROLLED = D.Port(PIECES, "boundary_written_unrolled", reference, INPUTS, package=PACKAGE, of_record="lean")

pieces = D.server_fixture(WRITTEN)


@pytest.mark.parametrize("port", [WRITTEN, WRITTEN_UNROLLED], ids=["boundary_written", "boundary_written_unrolled"])
def test_a_boundary_written_piece_agrees(pieces, port):
    stats = D.run(port, examples=1500, server=pieces)
    assert stats.outputs > 1000 and stats.rejects > 0, stats


def _read_doc(base, radix, step, row) -> dict:
    return {"base": base, "radix": list(radix), "step": list(step), "row": row}


#: four served ids of two 16-bit leaves each, a unit writing one id
IDS = [_read_doc(0, [4], [2], 2)]


@pytest.mark.parametrize("name, leaves, reads, want", [
    ("the ids", 8, IDS, True),
    ("three of the class's four ids (table (iii), link-written)", 8, [_read_doc(0, [3], [2], 2)], False),
    ("a fan-out", 8, [_read_doc(0, [4], [0], 2)], False),
    ("a negative step that lays the field", 8, [_read_doc(6, [4], [-2], 2)], False),
    ("two halves, in either order", 8, [_read_doc(4, [2], [2], 2), _read_doc(0, [2], [2], 2)], True),
    ("an overlap", 8, [*IDS, _read_doc(6, [1], [0], 2)], False),
    ("past the field", 8, [_read_doc(0, [5], [2], 2)], False),
    ("two interleaved reads, neither dense", 8, [_read_doc(0, [2], [4], 2), _read_doc(2, [2], [4], 2)], False),
    ("a row digit, slow", 8, [_read_doc(0, [2, 2], [2, 4], 2)], True),
    ("a digit of radix 1 at a negative step", 8, [_read_doc(0, [1, 4], [-5, 2], 2)], True),
    ("a row of no leaves", 0, [_read_doc(0, [1], [0], 0)], False),
], ids=lambda x: x if isinstance(x, str) else "")
def test_named_cases(pieces, name, leaves, reads, want):
    j = {"leaves": leaves, "reads": reads}
    assert reference(j)["written"] is want
    for fn in ("boundary_written", "boundary_written_unrolled"):
        assert pieces.call(fn, j) == ("output", D.canonical(reference(j))), fn
