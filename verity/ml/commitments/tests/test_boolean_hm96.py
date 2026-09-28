"""`verity.ml.commitments.gates.hm96`: the hm96-sha512 leaf in gates reproduces ``hm96/vectors_sha512.json``'s commit
strings and tree leaves and the pinned key's masks and salt digests; with a wrong salt it gives another leaf, which the
frame-v3-sha512 read then refuses to open where the right one opens; and it costs the gates pinned here."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.commitments import hm96
from verity.ml.commitments.gates import frame_v3 as F
from verity.ml.commitments.gates import hm96 as H

V = json.loads((Path(H.__file__).parents[1] / "hm96" / "vectors_sha512.json").read_text())
FV = json.loads((Path(H.__file__).parents[1] / "frame_v3" / "vectors_sha512.json").read_text())
#: (And, Xor, Not) gates of the traced leaf: four compressions, the XOR trees of b and nothing else
GATES = (215597, 1198112, 24592)


def _hex(x: str) -> bytes:
    return bytes.fromhex(x)


def _cases() -> list[tuple[bytes, bytes, bytes, bytes, bytes]]:
    """(x, y, b, c, leaf): the vectors' eight leaves, then the default key's masks as x = 0 (b is the mask, c the salt digest)."""
    out = [(_hex(v["inner_digest"]), _hex(v["salt"]), _hex(v["b"]), _hex(v["c"]), _hex(v["leaf"])) for v in V["leaves"]]
    for m in V["masks"]:
        if m["key"] == "default":
            b, c = _hex(m["mask"]), _hex(m["salt_digest"])
            out.append((bytes(64), _hex(m["salt"]), b, c, hm96.SHA512.tree_leaf(hm96.SHA512.default_key, b + c)))
    return out


def _digests(out) -> list[tuple[bytes, bytes, bytes]]:
    w = np.asarray(out).reshape(len(out), 3, 8, 64)
    pack = lambda d: b"".join(sum(int(b) << j for j, b in enumerate(row)).to_bytes(8, "big") for row in d)  # noqa: E731
    return [tuple(pack(d) for d in leaf) for leaf in w]


def _evaluate(xy: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes, bytes]]:
    args = [H.arguments(x, y) for x, y in xy]
    return _digests(evaluate_bits(bind(H.Hm96Sha512Leaf), *(np.array([a[k] for a in args], dtype=np.uint8) for k in range(2))))


def test_the_reference_reproduces_the_vectors() -> None:
    cases = _cases()
    assert len(cases) == len(V["leaves"]) + sum(m["key"] == "default" for m in V["masks"]) and len(V["leaves"]) == 8
    for x, y, b, c, leaf in cases:
        assert H.reference(x, y) == (b, c, leaf)
    assert V["salt_prefix"] == hm96.SHA512.salt_prefix.hex()
    assert V["leaf_prefix_default_key"] == hm96.SHA512.leaf_prefix(hm96.SHA512.default_key).hex()


def test_the_leaf_is_typed_and_refuses_malformed_arguments() -> None:
    target = bind(H.Hm96Sha512Leaf)
    assert target.id == "Hm96Sha512Leaf_v1"
    assert [t.leaves for _, t in target.signature()[0]] == [512, 8 * hm96.SHA512.salt_bytes]
    assert target.signature()[1].leaves == 3 * 512
    with pytest.raises(ValueError, match="x is 64 bytes and y 192"):
        H.arguments(bytes(32), bytes(192))
    with pytest.raises(ValueError, match="x is 64 bytes and y 192"):
        H.arguments(bytes(64), bytes(128))


@pytest.mark.slow
def test_the_leaf_reproduces_the_vectors() -> None:
    """About two minutes, most of it tracing: the only check of the gates against the vectors' commit strings and leaves."""
    cases = _cases()
    assert _evaluate([(x, y) for x, y, *_ in cases]) == [(b, c, leaf) for _, _, b, c, leaf in cases]


@pytest.mark.slow
def test_a_wrong_salt_gives_a_leaf_the_read_refuses() -> None:
    """The hiding vectors' rows (``hm96-sha512/row/v2``, rank 1 of a two-leaf tree): the gates' leaf from the right salt opens
    in the frame-v3-sha512 read and reproduces the vector's; from the salt with one bit flipped, or another row's salt, it is
    another leaf, and the read refuses it. A few minutes: the leaf's and the read's parts' tracing."""
    rows = FV["bit_row_hiding_leaves"]
    xy, want = [], []
    for k, h in enumerate(rows):
        x, y = _hex(h["digest"]), _hex(h["salt"])
        flipped = bytes([y[0] ^ 1]) + y[1:]
        xy += [(x, y), (x, flipped), (x, _hex(rows[1 - k]["salt"]))]
        want.append(_hex(h["hm96_leaf"]))
    got = _evaluate(xy)
    for k, h in enumerate(rows):
        right, flipped, other = (g[2] for g in got[3 * k:3 * k + 3])
        assert right == want[k] and right == H.reference(*xy[3 * k])[2]
        assert flipped != right and other != right
        assert flipped == H.reference(*xy[3 * k + 1])[2] and other == H.reference(*xy[3 * k + 2])[2]
    reads = []
    for k, h in enumerate(rows):
        dom, leaf = _hex(h["domain"]["domain_id"]), _hex(h["leaf"])
        sib = F.leaf_ref(dom, h["schema"], bytes(64), 0)
        root = F.node_ref(dom, 0, 0, sib, leaf)
        reads += [F.arguments(root, dom, g[2], 1, [sib]) for g in got[3 * k:3 * k + 3]]
    cols = [np.array([r[j] for r in reads], dtype=np.uint8) for j in range(5)]
    assert evaluate_bits(F.read(1, rows[0]["schema"]), *cols).reshape(-1).tolist() == [1, 0, 0] * len(rows)


@pytest.mark.slow
def test_the_leaf_costs_the_pinned_gates() -> None:
    assert H.Hm96Sha512Leaf.traced().counts == dict(zip(("and", "xor", "not"), GATES))
