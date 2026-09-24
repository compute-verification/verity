"""GPU-measured words: B-kernel's P0 register dumps of the pinned FA2 kernel, committed as tapped and opened by the
Definition's replay (R8 lane B-commit).  The dumps are unmodified copies under `fixtures/p0/` (see `PROVENANCE.md`
there; produced on cp-fa2 by lane B-kernel, 2026-09-18 ~07:50Z); skipped only if neither the copies nor B-kernel's
worktree are present (`kernel_dump.P0_DIR`).

These are the only tests here whose committed words come from the GPU; everything else in this directory uses
synthetic values (mechanics) or the oracle (semantics on synthetic inputs)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from verity_vllm.commit.fa2_prototype.derived import open_derived, verify_derived
from verity_vllm.commit.fa2_prototype.encoding import RowLayout, ValueId
from verity_vllm.commit.fa2_prototype.kernel_dump import P0_DIR, load_dump, n_block_max
from verity_vllm.commit.fa2_prototype.layouts import (HierCommitment, HierLayout, PackedCommitment, PackedLayout,
                                                            verify_value_opening)
from verity_vllm.commit.fa2_prototype.oracle import oracle_row, oracle_row_full
from verity_vllm.commit.fa2_prototype.reference import replay_row

pytestmark = pytest.mark.skipif(not P0_DIR.is_dir(), reason=f"B-kernel P0 dumps not present at {P0_DIR}")

SMALL = ["p0_16x32_h1kv1.npz", "p0_1x160_h1kv1.npz", "p0_1x160_h3kv1.npz", "p0_1x169_h3kv1.npz"]


def _rows_heads(kd):
    fx = kd.fixture
    return [(h, q) for h in range(fx.geo.NH) for q in fx.ctx.qrows()]


@pytest.mark.parametrize("name", SMALL)
def test_kernel_register_words_equal_the_definition(name: str) -> None:
    """Every committed word (S, M, ALPHA, P32, L, O, INV), the derived P16 and the HBM `out` row of the GPU dump equal
    the IR oracle's evaluation of `AttentionHead_v3` on the dump's own bf16 inputs — bit for bit."""
    kd = load_dump(P0_DIR / name)
    fx = kd.fixture
    assert fx.values == "recorded" and fx.source["kind"] == "fa2-kernel-p0-dump"
    n = bad = 0
    for h, q in _rows_heads(kd):
        rl = RowLayout(fx.geo, fx.ctx.layer, h, q)
        kw, kout, kp16 = kd.kernel_words(rl)
        ow, oout, op16 = oracle_row_full(rl, *fx.row_inputs(h, q))
        bad += int(np.count_nonzero(kw != ow)) + sum(int(np.count_nonzero(a != b)) for a, b in zip(kp16, op16))
        bad += int(np.count_nonzero(kout != oout))
        n += rl.count
    assert n > 0 and bad == 0


def test_kernel_tile_visits_a_fully_masked_block_that_the_definition_does_not_have() -> None:
    """Chunked-prefill dump 64x300: rows with T <= 256 see the tile's block 2 fully masked.  The kernel holds a state
    for it (m = -inf, l = 0, O = 0) and a rescale of 0 into block 1; the Definition has no such versions, `RowLayout`
    has no positions, and the kernel replay drops them by predicate — after which every word agrees."""
    kd = load_dump(P0_DIR / "p0_64x300_h1kv1.npz")
    fx, z = kd.fixture, kd.z
    assert kd.NB == 3 == n_block_max(kd.sq, kd.sk, kd.causal, kd.BN)
    for q in (236, 255, 256, 299):                       # T = 237, 256 (extra block) ; 257, 300 (no extra block)
        rl = RowLayout(fx.geo, 0, 0, q)
        hq, r = kd.kernel_coords(0, q)
        if rl.T <= 256:
            assert rl.nb == 2 and z["row_max"][hq, r, 2] == -np.inf and z["rescale"][hq, r, 1] == 0.0
            assert not np.any(z["row_sum_partial"][hq, r, 2]) and not np.any(z["acc_o"][hq, r, 2])
        else:
            assert rl.nb == 3 and z["rescale"][hq, r, 1] != 0.0
        kw, kout, _ = kd.kernel_words(rl)
        ow, oout, _ = oracle_row_full(rl, *fx.row_inputs(0, q))
        assert np.array_equal(kw, ow) and np.array_equal(kout, oout)
    assert kd.skipped_blocks == 2


@pytest.mark.parametrize("name", ["p0_1x169_h3kv1.npz", "p0_16x32_h1kv1.npz"])
def test_commit_the_kernel_words_open_with_the_definition_replay(name: str) -> None:
    """Commit what the GPU held in registers (kernel replay at commit time), discard, then open with the oracle replay
    against the ORIGINAL root under both layouts; a numpy replay does not open it."""
    kd = load_dump(P0_DIR / name)
    fx_k = kd.fixture.with_replay(kd.kernel_replay)     # commit-time producer: tapped words
    fx_o = kd.fixture.with_replay(oracle_row)            # challenge-time producer: the Definition
    fx_np = kd.fixture.with_replay(replay_row)           # a wrong-semantics replay
    geo, ctx = fx_k.geo, fx_k.ctx
    HL, PL = HierLayout(ctx, geo, 4), PackedLayout(ctx, geo, 256)
    hc, pc = HierCommitment(HL), PackedCommitment(PL)
    RH, RP = hc.commit(fx_k), pc.commit(fx_k)
    rows = list(ctx.qrows())
    for h in range(geo.NH):
        q = rows[-1]
        rl = RowLayout(geo, ctx.layer, h, q)
        ids = [ValueId(ctx.layer, h, q, "S", rl.nb - 1, rl.T - 1, 0), ValueId(ctx.layer, h, q, "M", rl.nb - 1, 0, 0),
               ValueId(ctx.layer, h, q, "L", 0, 1, rl.nb - 1), ValueId(ctx.layer, h, q, "O", 0, 5, rl.nb - 1),
               ValueId(ctx.layer, h, q, "INV", rl.nb, 0, rl.nb)]
        if rl.nb > 1:
            ids += [ValueId(ctx.layer, h, q, "ALPHA", 0, 0, rl.nb - 1), ValueId(ctx.layer, h, q, "P32", 0, 3, rl.nb - 1)]
        kw, _, _ = kd.kernel_words(rl)
        for vid in ids:
            oh, op = hc.open(fx_o, vid), pc.open(fx_o, vid)
            assert verify_value_opening(RH, HL, oh) and verify_value_opening(RP, PL, op)
            assert int.from_bytes(oh.word, "little") == int(kw[rl.rank(vid)]) == int.from_bytes(op.word, "little")
        d = open_derived(hc, fx_o, ValueId(ctx.layer, h, q, "P16", rl.nb - 1, (rl.nb - 1) * geo.BN, 0))
        assert verify_derived(RH, HL, d)
        # forged word against the kernel's root
        bad = dataclasses.replace(oh, word=bytes(b ^ 1 for b in oh.word))
        assert not verify_value_opening(RH, HL, bad)
    vid = ValueId(ctx.layer, 0, rows[-1], "P32", 0, 2, RowLayout(geo, ctx.layer, 0, rows[-1]).nb - 1)
    with pytest.raises(RuntimeError, match="did not reproduce"):
        hc.open(fx_np, vid)
    with pytest.raises(RuntimeError, match="did not reproduce"):
        pc.open(fx_np, vid)
