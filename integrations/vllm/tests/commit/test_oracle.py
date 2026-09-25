"""Round trips on Verity-semantics values: the fixture's replay is the registry Definition `AttentionHead_v3` evaluated by
the IR oracle (`oracle.py`), not the numpy approximation.  Values are still synthetic bf16 inputs (mechanics + semantics
of the Definition; NOT FA2 numerical correspondence, which is B-sem's question)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tests.commit.fa2_prototype.derived import conv_f2fp_bf16, open_derived, verify_derived
from tests.commit.fa2_prototype.encoding import Geometry, RowLayout, ValueId
from tests.commit.fa2_prototype.fixture import synthetic_fixture
from tests.commit.fa2_prototype.layouts import (HierCommitment, HierLayout, PackedCommitment, PackedLayout,
                                                            verify_value_opening)
from tests.commit.fa2_prototype.oracle import compare_reference, oracle_row, oracle_row_full

GEO = Geometry()


@pytest.mark.skipif(not __import__("os").environ.get("FA2_COMMIT_ORIGINAL_LOADER"),
                    reason="opt-in (FA2_COMMIT_ORIGINAL_LOADER=1): the original MufuTables.load peaks at 268.1 MB resident "
                           "[M], 0.3 MiB under the laptop guardian's 256-MiB kill line; the low-peak loader checks the same "
                           "digests on every import of oracle.py")
def test_low_peak_tables_match_the_original_loader() -> None:
    """The oracle's low-peak MUFU table reconstruction equals `MufuTables.load` (digest-pinned in `oracle.py`); the
    original loader is run in a subprocess so its resident peak stays out of the pytest process.  The digests it prints
    were pinned from this very run on 2026-09-18 (see `oracle.MUFU_TABLE_SHA256`)."""
    import subprocess
    import sys

    from tests.commit.fa2_prototype.oracle import MUFU_TABLE_SHA256

    code = ("import hashlib; from verity_vllm.program.backends import fa2_relation, fa2_model as FA;"
            "T = FA.MufuTables.load(fa2_relation.tables_dir());"
            "print(hashlib.sha256(T.ex2.tobytes()).hexdigest(), hashlib.sha256(T.rcp.tobytes()).hexdigest())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                         env={**__import__("os").environ, "PYTHONPATH": ":".join(sys.path)})
    assert out.returncode == 0, out.stderr[-2000:]
    ex2, rcp = out.stdout.split()
    assert (ex2, rcp) == (MUFU_TABLE_SHA256["ex2"], MUFU_TABLE_SHA256["rcp"])


@pytest.mark.parametrize("T", [1, 5, 64, 128, 129, 155, 160])
def test_oracle_fills_the_layout_and_agrees_with_the_declared_conversion(T: int) -> None:
    fx = synthetic_fixture(seed=3, geo=GEO, layer=0, step=0, prompt_len=max(T, 2))
    rl = RowLayout(GEO, 0, 4, T - 1)
    words, out16, p16s = oracle_row_full(rl, *fx.row_inputs(4, T - 1))
    assert words.shape == (rl.count,) and out16.shape == (GEO.D,)
    for v in range(rl.nb):
        # P16 of the Definition (`F2fpBf16` gates) == the derived rule's conversion of the committed P32 words
        b = rl.block_of_version(v)
        p32 = [words[rl.rank(ValueId(0, 4, T - 1, "P32", b, b * GEO.BN + c, v))] for c in range(len(p16s[v]))]
        got = np.frombuffer(b"".join(conv_f2fp_bf16(np.uint32(w).tobytes()) for w in p32), dtype=np.uint16)
        assert np.array_equal(got, p16s[v])


def test_numpy_reference_is_not_the_semantics_but_agrees_on_the_bf16_output() -> None:
    """Documents why the fixture's reference must be the oracle for interior commitments: the numpy approximation
    (np.exp2 for MUFU.EX2, sequential fp32 sums for the tensor-core adder, no FTZ) differs in interior WORDS while the
    bf16 output row agrees — output-level agreement is no evidence of interior-word agreement."""
    fx = synthetic_fixture(seed=5, geo=GEO, layer=0, step=0, prompt_len=155)
    rl = RowLayout(GEO, 0, 1, 154)
    cmp = compare_reference(rl, *fx.row_inputs(1, 154))
    assert cmp["P32"][0] > 0 and cmp["S"][0] > 0, cmp
    assert cmp["OUT16"][0] == 0, cmp


def _open_ids(rl: RowLayout, head: int) -> list[ValueId]:
    T, q = rl.T, rl.qrow
    ids = [ValueId(0, head, q, "S", rl.nb - 1, T - 1, 0), ValueId(0, head, q, "M", rl.nb - 1, 0, 0),
           ValueId(0, head, q, "L", 0, 2, rl.nb - 1), ValueId(0, head, q, "O", 0, 63, rl.nb - 1), ValueId(0, head, q, "INV", rl.nb, 0, rl.nb)]
    if rl.nb > 1:
        ids += [ValueId(0, head, q, "ALPHA", 0, 0, rl.nb - 1), ValueId(0, head, q, "P32", 0, 17, rl.nb - 1)]
    return ids


@pytest.mark.parametrize("kind", ["decode_T160", "prefill_P20"])
def test_round_trip_on_oracle_values_both_layouts(kind: str) -> None:
    if kind == "decode_T160":
        fx = synthetic_fixture(seed=7, geo=GEO, layer=3, step=5, prompt_len=155).with_replay(oracle_row)   # 9 rows, T = 160
    else:
        fx = synthetic_fixture(seed=8, geo=GEO, layer=0, step=0, prompt_len=20).with_replay(oracle_row)    # 180 rows, T = 1..20
    HL, PL = HierLayout(fx.ctx, GEO, 4), PackedLayout(fx.ctx, GEO, 256)
    hc, pc = HierCommitment(HL), PackedCommitment(PL)
    RH, RP = hc.commit(fx), pc.commit(fx)
    rows = fx.ctx.qrows()
    for head in (0, 8):
        rl = RowLayout(GEO, fx.ctx.layer, head, rows[-1])
        for vid in _open_ids(rl, head):
            vid = dataclasses.replace(vid, layer=fx.ctx.layer)
            oh, op = hc.open(fx, vid), pc.open(fx, vid)
            assert oh.word == op.word                      # one identity -> one word under both layouts
            assert verify_value_opening(RH, HL, oh) and verify_value_opening(RP, PL, op)
            # the opened word IS the oracle's word
            words, _, _ = oracle_row_full(rl, *fx.row_inputs(head, rows[-1]))
            assert int.from_bytes(oh.word, "little") == int(words[rl.rank(vid)])
        d = open_derived(hc, fx, ValueId(fx.ctx.layer, head, rows[-1], "P16", rl.nb - 1, (rl.nb - 1) * GEO.BN, 0))
        assert verify_derived(RH, HL, d)


def test_wrong_semantics_replay_cannot_open_an_oracle_commitment() -> None:
    """The commitment is fixed before the challenge; a replay under different semantics (here the numpy approximation)
    does not reproduce the retained subtree root / chunk digest and the opener refuses — the transcript cannot be
    re-derived from a root by a 'close enough' recomputation."""
    fx_o = synthetic_fixture(seed=9, geo=GEO, layer=0, step=5, prompt_len=155).with_replay(oracle_row)
    from tests.commit.fa2_prototype.reference import replay_row

    fx_np = fx_o.with_replay(replay_row)  # same inputs, numpy replay
    HL, PL = HierLayout(fx_o.ctx, GEO, 4), PackedLayout(fx_o.ctx, GEO, 256)
    hc, pc = HierCommitment(HL), PackedCommitment(PL)
    hc.commit(fx_o), pc.commit(fx_o)
    vid = ValueId(0, 2, 159, "P32", 1, 133, 0)
    with pytest.raises(RuntimeError, match="did not reproduce"):
        hc.open(fx_np, vid)
    with pytest.raises(RuntimeError, match="did not reproduce"):
        pc.open(fx_np, vid)
    # and the oracle replay does open it
    assert verify_value_opening(hc.root, HL, hc.open(fx_o, vid))
