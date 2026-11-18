"""A composite with word leaves, bit-sliced through its bit view (`bits.evaluate_lowered`), against the reference."""

from __future__ import annotations

import zlib

import numpy as np
import pytest
from verity.circuits.defs import bind
from verity.circuits.evaluation import batch as BA
from verity.circuits.evaluation import bits as BS
from verity.commitments.gates import embedding as EM

from verity_vllm.definitions.registry import dsa_topk_reads as R
from verity_vllm.definitions.registry import sampled_accept as SA


def _held(t, n: int) -> None:
    assert BS.lowers(t) and not BA.kernels(t)
    args = BA.sample_uniform(np.random.default_rng(zlib.crc32(t.id.encode())), n, [p for _, p in t.signature()[0]])
    assert np.array_equal(BA.evaluate_batch(t, *args), BA.evaluate_batch(t, *args, kernel="reference"))


def test_the_read_is_its_reference() -> None:
    _held(EM.read(32, 2), 64)


@pytest.mark.slow
def test_the_long_row_is_its_reference() -> None:
    """The long row's own target, compiled once for the whole suite; two rows, as the reference walks its gates."""
    _held(bind(R.DsaTopKReadRows, N=20, TOPK=4, BLOCK=8, READ=EM.read(32, 2)), 2)


def test_an_arithmetic_word_primitive_is_not_lowered() -> None:
    assert not BS.lowers(bind(SA.LogitAtStep))
