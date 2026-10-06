"""A registered value's rows are committed as C-Flock's statement commits that port's rows (`one_stage.registered`,
PROTOCOL.md §16.10): under one salt, a registered row's `b || c` is the statement's bit row's (`sha512/row/v2` of exactly
the row's bits), so its `hm96-sha512/row/v2` leaf is the one the Lean verifier climbs from to the registered root, for a
16-bit port and for a 32-bit one."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from verity.primitives.commitments import hm96
from verity_flock import circuit as FC
from verity.protocols.verification.sampled_proofs.one_stage import registered as G

PROGRAM = "ab" * 64
#: RMSNorm's eps (1e-5) as the f32 its Definition holds
EPS = struct.unpack("<I", struct.pack("<f", 1e-5))[0]
#: FA2's `scale_softmax_log2` at D = 64: `float(D ** -0.5) * M_LOG2E` narrowed to f32
SCALE = int(np.float32(float(np.float32(64 ** -0.5)) * 1.4426950408889634).view(np.uint32))


def value_row(widths, values) -> list[int]:
    """A values row: each value's 16-bit words, low first, in order, a bit row of exactly those words."""
    return [(int(v) >> (16 * i)) & 0xFFFF for w, v in zip(widths, values) for i in range(-(-w // 16))]


def _statement_leaves(rows, salts) -> list[bytes]:
    """The statement's `b || c` of each row of a bit-row port of `rows`' width (`circuit._commit_strings`, what its
    frame-v3 tree and its public file hold)."""
    rows = np.asarray(rows, dtype=np.uint16)
    port = FC._port("values", rows.shape[1], 16 * rows.shape[1])
    assert port.schema == G.SCHEMA
    return FC._commit_strings([port], {"values": rows}, {"values": salts})["values"]


def _equal(reg: G.Registered, statement_rows) -> None:
    bcs = _statement_leaves(statement_rows, list(reg.salts))
    assert list(reg.bcs) == bcs
    for r, bc in enumerate(bcs):
        assert G.leaf(reg.domain, r, bc) == reg.levels[0][r]
        assert G.verify_read(reg.domain, reg.root, r, bc, reg.path(r))


@pytest.mark.parametrize("words", [3, 64, 100])
def test_a_16_bit_registered_row_is_the_ports_row(words):
    rows = np.random.default_rng(20261002).integers(0, 1 << 16, (3, words)).tolist()
    reg = G.Registered.commit(PROGRAM, "cs_table", rows)
    assert [G.row_words(r) for r in rows] == rows and G.row_bits(words) == 16 * words
    _equal(reg, rows)


def test_a_32_bit_registered_row_is_the_values_row():
    """The values row: eps, the softmax scale, and both in one row, each f32 two u16 words, low first."""
    assert (EPS, SCALE) == (0x3727C5AC, 0x3E38AA3B)
    assert value_row([32, 32], [EPS, SCALE]) == [0xC5AC, 0x3727, 0xAA3B, 0x3E38]
    for rows in ([[EPS]], [[SCALE]], [[EPS, SCALE], [SCALE, EPS]]):
        reg = G.Registered.commit(PROGRAM, "values", rows, word_bits=32)
        assert [G.row_words(r, 32) for r in rows] == [value_row([32] * len(r), r) for r in rows]
        _equal(reg, [value_row([32] * len(r), r) for r in rows])
        # the same u16 words registered as 16-bit leaves have the same b || c, under another domain
        as16 = G.Registered.commit(PROGRAM, "values", [value_row([32] * len(r), r) for r in rows], list(reg.salts))
        assert as16.bcs == reg.bcs and as16.domain.domain_id != reg.domain.domain_id and as16.root != reg.root


def test_a_values_row_under_a_fresh_salt_or_as_a_v1_row_opens_nothing():
    reg = G.Registered.commit(PROGRAM, "values", [[EPS, SCALE]], word_bits=32)
    fresh = _statement_leaves([value_row([32, 32], [EPS, SCALE])], [hm96.SHA512.fresh_salt()])
    assert fresh != list(reg.bcs) and not G.verify_read(reg.domain, reg.root, 0, fresh[0], reg.path(0))
    # the row zero-padded to a whole block, as a `sha512/row/v1` port commits it, under the registered salt
    v1 = np.zeros((1, 64), dtype=np.uint16)
    v1[0, :4] = value_row([32, 32], [EPS, SCALE])
    bc = FC._commit_strings([FC._port("values", 64)], {"values": v1}, {"values": list(reg.salts)})["values"][0]
    assert bc != reg.bcs[0] and not G.verify_read(reg.domain, reg.root, 0, bc, reg.path(0))
