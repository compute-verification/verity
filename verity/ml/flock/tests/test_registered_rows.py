"""A registered value's rows are committed as C-Flock's statement commits that port's rows (`one_stage.registered`,
PROTOCOL.md §16.10): under one salt, a registered row's `b || c` is the statement row's, so its `hm96-sha512/row/v1` leaf
is the one the Lean verifier climbs from to the registered root, for a 16-bit port and for a 32-bit one."""
from __future__ import annotations

import struct

import numpy as np

from verity.ml.commitments import hm96
from verity_flock import circuit as FC
from verity.core.protocols.sampled_proofs.one_stage import registered as G

PROGRAM = "ab" * 64
#: RMSNorm's eps (1e-5) as the f32 its Definition holds
EPS = struct.unpack("<I", struct.pack("<f", 1e-5))[0]
#: FA2's `scale_softmax_log2` at D = 64: `float(D ** -0.5) * M_LOG2E` narrowed to f32
SCALE = int(np.float32(float(np.float32(64 ** -0.5)) * 1.4426950408889634).view(np.uint32))


def value_row(widths, values) -> list[int]:
    """#749's `ir_lower.value_row`: each value's 16-bit words, low first, in order, zero-padded to whole 64-word blocks."""
    row = [(int(v) >> (16 * i)) & 0xFFFF for w, v in zip(widths, values) for i in range(-(-w // 16))]
    return row + [0] * (-len(row) % 64)


def _statement_leaves(rows, salts) -> list[bytes]:
    """The statement's `b || c` of each row of a port of `rows`' width (`circuit._commit_strings`, what its frame-v3 tree
    and its public file hold)."""
    rows = np.asarray(rows, dtype=np.uint16)
    return FC._commit_strings([FC._port("values", rows.shape[1])], {"values": rows}, {"values": salts})["values"]


def _equal(reg: G.Registered, statement_rows) -> None:
    bcs = _statement_leaves(statement_rows, list(reg.salts))
    assert list(reg.bcs) == bcs
    for r, bc in enumerate(bcs):
        assert G.leaf(reg.domain, r, bc) == reg.levels[0][r]
        assert G.verify_read(reg.domain, reg.root, r, bc, reg.path(r))


def test_a_16_bit_registered_row_is_the_ports_row():
    rows = np.random.default_rng(20261002).integers(0, 1 << 16, (3, 64)).tolist()
    reg = G.Registered.commit(PROGRAM, "cs_table", rows)
    assert [G.row_words(r) for r in rows] == rows
    _equal(reg, rows)


def test_a_32_bit_registered_row_is_the_values_row():
    """The values row of #749: eps, the softmax scale, and both in one row, each f32 two u16 words, low first."""
    assert (EPS, SCALE) == (0x3727C5AC, 0x3E38AA3B)
    assert value_row([32, 32], [EPS, SCALE])[:5] == [0xC5AC, 0x3727, 0xAA3B, 0x3E38, 0]
    for rows in ([[EPS]], [[SCALE]], [[EPS, SCALE], [SCALE, EPS]]):
        reg = G.Registered.commit(PROGRAM, "values", rows, word_bits=32)
        assert [G.row_words(r, 32) for r in rows] == [value_row([32] * len(r), r) for r in rows]
        _equal(reg, [value_row([32] * len(r), r) for r in rows])
        # the same u16 words registered as 16-bit leaves have the same b || c, under another domain
        as16 = G.Registered.commit(PROGRAM, "values", [value_row([32] * len(r), r)[:2 * len(r)] for r in rows], list(reg.salts))
        assert as16.bcs == reg.bcs and as16.domain.domain_id != reg.domain.domain_id and as16.root != reg.root


def test_a_values_row_under_a_fresh_salt_opens_nothing():
    reg = G.Registered.commit(PROGRAM, "values", [[EPS, SCALE]], word_bits=32)
    fresh = _statement_leaves([value_row([32, 32], [EPS, SCALE])], [hm96.SHA512.fresh_salt()])
    assert fresh != list(reg.bcs) and not G.verify_read(reg.domain, reg.root, 0, fresh[0], reg.path(0))
