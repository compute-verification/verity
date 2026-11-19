"""A composite with word leaves, bit-sliced through its bit view (`bits.evaluate_lowered`), against the reference."""

from __future__ import annotations

import zlib

import numpy as np
import pytest
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.evaluation import batch as BA
from verity.circuits.evaluation import bits as BS
from verity.circuits.population import BitsWord, WordBits
from verity.circuits.program import NoBooleanBody
from verity.circuits.types import Value
from verity.commitments.gates import embedding as EM

from verity_vllm.definitions.registry import sampled_accept as SA


def _held(t, n: int) -> None:
    assert BS.lowers(t) and not BA.kernels(t)
    args = BA.sample_uniform(np.random.default_rng(zlib.crc32(t.id.encode())), n, [p for _, p in t.signature()[0]])
    assert np.array_equal(BA.evaluate_batch(t, *args), BA.evaluate_batch(t, *args, kernel="reference"))


@pytest.mark.parametrize("t", [bind(WordBits, W=32), bind(BitsWord, W=16)], ids=lambda t: t.id)
def test_the_wiring_is_its_reference(t) -> None:
    _held(t, 64)


def test_word_primitives_with_boolean_bodies_are_their_reference() -> None:
    _held(bind(SA.LogitAtStep), 64)


@pytest.mark.slow
def test_the_read_is_its_reference() -> None:
    """The one held composite whose word leaves meet Boolean gates and calls (the wiring tests have no gate, the word
    primitives' test no word leaf a gate reads); one row, as the reference walks the read's SHA-512 gate by gate."""
    _held(EM.read(32, 2), 1)


def test_a_word_primitive_without_a_boolean_body_is_not_lowered_and_is_refused() -> None:
    opaque = PrimitiveDefinition("TestOpaqueByte", 1, (("x", Value(8)),), Value(8), lambda x: x ^ 0x5A, register=False)
    t = bind(CompositeDefinition("TestOpaqueByteCall", 1, (), lambda S: ((("x", Value(8)),), Value(8)),
                                 lambda B, S, x: B.call(opaque, x), register=False))
    assert not BS.lowers(t)
    with pytest.raises(NoBooleanBody, match="TestOpaqueByte_v1"):
        BA.evaluate_batch(t, np.arange(4, dtype=np.uint8))
