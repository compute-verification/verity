"""The vectorized ``hm96`` leaves (`verity.ml.windows.hiding`) are core's reference (`verity.ml.commitments.hm96`) byte for byte: on
both instances' vectors, and over more leaves than one numpy batch with fresh salts."""
from __future__ import annotations

import os

import pytest

from verity.ml.commitments import hm96
from verity.ml.commitments.hm96 import vectors as V
from verity.ml.windows import hiding


def test_batch_path_is_the_core_reference_on_its_vectors():
    h = hiding.HidingLeaves()
    cases = V.load()["leaves"]
    inner = [bytes.fromhex(c["inner_digest"]) for c in cases]
    salts = b"".join(bytes.fromhex(c["salt"]) for c in cases)
    assert [x.hex() for x in h.leaves_with(inner, salts)] == [c["leaf"] for c in cases]
    masks = h.masks(salts, len(cases))
    assert [bytes(m).hex() for m in masks] == [hm96.mask(hm96.DEFAULT_KEY, bytes.fromhex(c["salt"])).hex() for c in cases]


def test_batch_path_is_the_core_reference_for_hm96_sha512():
    h = hiding.HidingLeaves(instance=hm96.SHA512)
    cases = V.load(hm96.SHA512)["leaves"]
    inner = [bytes.fromhex(c["inner_digest"]) for c in cases]
    salts = b"".join(bytes.fromhex(c["salt"]) for c in cases)
    assert [x.hex() for x in h.leaves_with(inner, salts)] == [c["leaf"] for c in cases]
    leaves, fresh = h.leaves(inner)
    assert len(fresh) == 192 * len(inner) and leaves[1] == h.leaf(inner[1], fresh[192:384])


def test_batch_path_matches_core_over_many_leaves_and_fresh_salts():
    h = hiding.HidingLeaves(hm96.fresh_key())
    inner = [os.urandom(32) for _ in range(5000)]              # more than one numpy batch
    leaves, salts = h.leaves(inner)
    assert len(salts) == 128 * len(inner) and len(set(salts[i:i + 128] for i in range(0, len(salts), 128))) == len(inner)
    for i in (0, 1, 4095, 4096, 4999):
        assert leaves[i] == h.leaf(inner[i], salts[128 * i:128 * (i + 1)])
    with pytest.raises(ValueError):
        h.leaves_with(inner[:2], salts[:255])
