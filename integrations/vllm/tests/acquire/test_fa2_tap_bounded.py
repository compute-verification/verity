"""The bounded FA2 M1 tap: sequence-group splitting of a multi-sequence launch into ONE capped device buffer.

Torch-free.  Two halves:

  1. the group PLAN (`hidden_source.plan_fa2_tap_groups` / `fa2_tap_buffer_words`): order preserved, every sequence in exactly one group,
     every group's padded buffer within the cap; the B64 llama 1024 geometry reproduces the r17_0738 OOM byte count exactly.
  2. a numpy model of the tap + p0b compaction (native_collect.cpp Fa2Wrap): the kernel writes rows < seqlen_q of each sequence's h slabs
     (causal partial writes per plane), the wrapper stages every sequence's rectangle per plane as a NAMED tensor and zeroes it.  The
     UNBOUNDED path (one whole-batch buffer per launch) and the BOUNDED path (groups, prefix views of one capped buffer, stage -> flush ->
     zero -> next group, names by the GLOBAL sequence index) must produce the same (name, bytes) sequence in the same order:
     completeness (every tapped value acquired exactly once), ordering, and digest-of-concatenation equality.  Negatives: no zeroing
     between groups leaks the previous group's bytes (the zeroing is load-bearing); the bounded buffer's residency never exceeds the cap.
"""

from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pytest

from verity_vllm.acquire.sources.hidden_source import (
    CHUNK_WORDS,
    fa2_tap_buffer_words,
    fa2_tap_words_per_seq,
    mat_total_words,
    plan_fa2_tap_groups,
)

# ----------------------------------------------------------------------------------------------------------------------------- 1. plan


def test_b64_llama_geometry_is_the_r17_0738_allocation():
    """B64 llama-3.2-1B 1024/128 prefill: b=64, h=32, M=max_seqlen_q=1024, NB=8 (BN=128), D=64 -> the one allocation exec's commit.log names."""
    words = fa2_tap_buffer_words(64, 32, 1024, 8, 128, 64)
    assert words * 4 == 26_340_229_120                     # "Tried to allocate 24.53 GiB" = 26,340,229,120 B (r17_0738 commit.log:168)
    assert fa2_tap_words_per_seq(32, 1024, 8, 128, 64) * 4 == 411_566_080     # ~393 MiB per sequence at pitch M=1024
    assert mat_total_words(64 * 32, 1024, 8, 128, 64) == 64 * fa2_tap_words_per_seq(32, 1024, 8, 128, 64)


def test_plan_b64_and_b8_under_2gib():
    cap_words = (2048 << 20) // 4
    per = fa2_tap_words_per_seq(32, 1024, 8, 128, 64)
    g64 = plan_fa2_tap_groups(64, per, cap_words)
    assert g64[0] == (0, 5) and g64[-1] == (60, 64) and len(g64) == 13
    assert all(fa2_tap_buffer_words(g1 - g0, 32, 1024, 8, 128, 64) <= cap_words for g0, g1 in g64)
    g8 = plan_fa2_tap_groups(8, per, cap_words)
    assert g8 == [(0, 5), (5, 8)]
    # B4 fits in one group (no split at all): 1.53 GiB <= 2 GiB
    assert plan_fa2_tap_groups(4, per, cap_words) == [(0, 4)]
    # cap off -> one group
    assert plan_fa2_tap_groups(64, per, 0) == [(0, 64)]


def test_small_cap_plans_one_sequence_groups_which_the_wrapper_must_still_compact():
    """Cap 512 MiB on the B8 llama geometry (393 MiB per sequence) plans EIGHT one-sequence groups.  The C++ Fa2Wrap must treat
    such a group as a group of a split launch (hold_launch), not as a B=1 launch: staged on the whole-buffer b == 1 branch they give
    27,376 tensors / root ba16ddff.. vs 28,016 / 2891d10b.. at cap 0 and 2048.  The plan is legal and the equality is measured on
    the pod; this pins the plan shape that exercises the branch."""
    per = fa2_tap_words_per_seq(32, 1024, 8, 128, 64)
    g8 = plan_fa2_tap_groups(8, per, (512 << 20) // 4)
    assert g8 == [(i, i + 1) for i in range(8)]
    assert all(fa2_tap_buffer_words(1, 32, 1024, 8, 128, 64) <= (512 << 20) // 4 for _ in g8)


@pytest.mark.parametrize("seed", range(40))
def test_plan_covers_in_order_within_cap(seed):
    rng = np.random.default_rng(seed)
    b = int(rng.integers(1, 130))
    per = int(rng.integers(1, 5000))
    cap = int(rng.integers(1, 40000))
    groups = plan_fa2_tap_groups(b, per, cap)
    assert groups[0][0] == 0 and groups[-1][1] == b
    for (a0, a1), (b0, b1) in itertools.pairwise(groups):
        assert a1 == b0 and a0 < a1 and b0 < b1               # consecutive, non-empty, ascending: order preserved, no gap, no overlap
    for g0, g1 in groups:
        padded = -(-((g1 - g0) * per) // CHUNK_WORDS) * CHUNK_WORDS
        assert padded <= cap or (g1 - g0) == 1                # within the cap, or a single sequence that alone exceeds it (named by the driver)
    # a singleton over the cap is the only way to exceed it, and then EVERY group is a singleton of that cost
    if any(-(-((g1 - g0) * per) // CHUNK_WORDS) * CHUNK_WORDS > cap for g0, g1 in groups):
        assert all(g1 - g0 == 1 for g0, g1 in groups)


# ------------------------------------------------------------------------------------------------------- 2. numpy model of the tap + staging

PLANES = ("S", "P", "RP", "ROW", "O", "FIN")


def _widths(NB, BN, D):
    N = NB * BN
    return [N, N, N // 2, NB * 8, NB * D, 4]


def _plane_offsets(HB, M, widths):
    off, offs = 0, []
    for w in widths:
        offs.append(off)
        off += HB * M * w
    return offs, off


def _value(seq, plane, slab_in_seq, row, col, layer):
    """Deterministic per-position value the 'kernel' writes: a function of the SEQUENCE's identity (what the real kernel's values are a
    function of), never of the buffer position -- so a split launch produces the same values for the same sequence."""
    return ((seq * 1000003 + plane * 10007 + slab_in_seq * 7919 + row * 101 + col * 7 + layer * 13) * 2654435761) & 0xFFFFFFFF


def tap_write(buf, HB, M, NB, BN, D, seqlens, seq_base, layer, kv_prior=None):
    """The M1 tap of one launch of len(seqlens) sequences into `buf` (words), slab = local_seq*h + slab_in_seq, rows < seqlen only,
    causal partial writes: S/P/RP columns < ((kv_prior + row)//BN + 1)*BN (RP half), ROW/O blocks <= (kv_prior + row)//BN (ROW word 3
    never written), FIN all.  `kv_prior[i]` = the sequence's already-cached keys (seqlen_k - seqlen_q; 0 for a plain prefill) -- the
    write set then depends on the SEQUENCE, not only on the row."""
    h = HB // len(seqlens)
    widths = _widths(NB, BN, D)
    offs, total = _plane_offsets(HB, M, widths)
    assert buf.size >= total
    kv_prior = kv_prior or [0] * len(seqlens)
    for i, sl in enumerate(seqlens):
        seq = seq_base + i
        for s in range(h):
            slab = i * h + s
            for p, w in enumerate(widths):
                plane = buf[offs[p]:offs[p] + HB * M * w].reshape(HB, M, w)
                for r in range(sl):
                    nblk = min(NB, (kv_prior[i] + r) // BN + 1)
                    if p in (0, 1):
                        cols = np.arange(nblk * BN)
                    elif p == 2:
                        cols = np.arange((nblk * BN) // 2)
                    elif p == 3:
                        cols = np.array([blk * 8 + j for blk in range(nblk) for j in (0, 1, 2, 4, 5, 6, 7)])
                    elif p == 4:
                        cols = np.arange(nblk * D)
                    else:
                        cols = np.arange(4)
                    plane[slab, r, cols] = np.array([_value(seq, p, s, r, int(c), layer) for c in cols], dtype=np.uint32)


def stage_compacted(buf, HB, M, NB, BN, D, seqlens, seq_base, layer_k, zero):
    """native_collect.cpp Fa2Wrap compacted staging: per sequence, per plane, the [h][sl][width] rectangle as a named tensor (bytes in
    slab-major order); then (optionally) zero exactly those rectangles.  Returns [(name, bytes)] in staging order."""
    h = HB // len(seqlens)
    widths = _widths(NB, BN, D)
    offs, _ = _plane_offsets(HB, M, widths)
    out = []
    rects = []
    for i, sl in enumerate(seqlens):
        for p, w in enumerate(widths):
            plane = buf[offs[p]:offs[p] + HB * M * w].reshape(HB, M, w)
            view = plane[i * h:(i + 1) * h, :sl, :]
            name = f"fa2.m1/L{layer_k}/s{seq_base + i}/g{h}x{sl}x{NB}/{PLANES[p]}"
            out.append((name, view.tobytes()))
            rects.append(view)
    if zero:
        for v in rects:
            v[...] = 0
    return out


def unbounded_path(seqlens, h, M, NB, BN, D, layers, buf=None, kv_prior=None):
    b = len(seqlens)
    HB = b * h
    words = fa2_tap_buffer_words(b, h, M, NB, BN, D)
    if buf is None:
        buf = np.zeros(words, dtype=np.uint32)
    staged = []
    for k in range(layers):
        tap_write(buf, HB, M, NB, BN, D, seqlens, 0, k, kv_prior)
        staged.extend(stage_compacted(buf, HB, M, NB, BN, D, seqlens, 0, k, zero=True))
    return staged, buf


def bounded_path(seqlens, h, M, NB, BN, D, layers, cap_words, boundbuf=None, zero_between_groups=True, kv_prior=None):
    """The split driver: groups from the plan, each tapped into a PREFIX of one capped buffer, staged with the GLOBAL index, zeroed."""
    b = len(seqlens)
    per = fa2_tap_words_per_seq(h, M, NB, BN, D)
    groups = plan_fa2_tap_groups(b, per, cap_words)
    if boundbuf is None:
        boundbuf = np.zeros(cap_words, dtype=np.uint32)
    peak = 0
    staged = []
    kv_prior = kv_prior or [0] * b
    for k in range(layers):
        for g0, g1 in groups:
            HBg = (g1 - g0) * h
            words = fa2_tap_buffer_words(g1 - g0, h, M, NB, BN, D)
            assert words <= cap_words, "a group over the cap must not reach the bounded buffer"
            peak = max(peak, words)
            view = boundbuf[:words]
            tap_write(view, HBg, M, NB, BN, D, seqlens[g0:g1], g0, k, kv_prior[g0:g1])
            staged.extend(stage_compacted(view, HBg, M, NB, BN, D, seqlens[g0:g1], g0, k, zero=zero_between_groups))
    return staged, boundbuf, peak, groups


GEOM = {"h": 2, "M": 8, "NB": 2, "BN": 4, "D": 4}


def _digest(staged):
    hsh = hashlib.sha256()
    for name, by in staged:
        hsh.update(name.encode() + b"\0" + by)
    return hsh.hexdigest()


@pytest.mark.parametrize("seqlens,cap_seqs", [([8, 3, 5, 8, 1, 6], 2), ([8, 3, 5, 8, 1, 6], 4), ([2, 8, 8, 7, 4], 1), ([5, 5, 5, 5, 5, 5, 5, 5], 3), ([8, 1], 1)])
def test_bounded_equals_unbounded(seqlens, cap_seqs):
    h, M, NB, BN, D = GEOM["h"], GEOM["M"], GEOM["NB"], GEOM["BN"], GEOM["D"]
    layers = 3
    per = fa2_tap_words_per_seq(h, M, NB, BN, D)
    cap_words = fa2_tap_buffer_words(cap_seqs, h, M, NB, BN, D)
    ref, _ = unbounded_path(seqlens, h, M, NB, BN, D, layers)
    got, boundbuf, peak, groups = bounded_path(seqlens, h, M, NB, BN, D, layers, cap_words)
    # split actually happened (the fixture is above the cap) and every group fits
    assert len(groups) == -(-len(seqlens) // cap_seqs) and peak <= cap_words
    # completeness: every (layer, sequence, plane) staged exactly once, in the unsplit order
    assert [n for n, _ in got] == [n for n, _ in ref]
    assert len(got) == layers * len(seqlens) * 6 == len(set(n for n, _ in got))
    # bytes identical per tensor and as a whole
    for (n1, b1), (n2, b2) in zip(ref, got):
        assert n1 == n2 and b1 == b2, n1
    assert _digest(got) == _digest(ref)
    # the bounded buffer's residency is the cap, independent of the batch: the unbounded buffer was len(seqlens)*per words
    assert boundbuf.size == cap_words < fa2_tap_buffer_words(len(seqlens), h, M, NB, BN, D)
    # after the last zeroing the bounded buffer is all zero again (no residue for the next step)
    assert not boundbuf.any()


def test_bounded_buffer_reused_across_steps_leaves_no_residue():
    """Two consecutive steps with different batch shapes on the SAME bounded buffer stage exactly what a fresh unbounded run stages."""
    h, M, NB, BN, D = GEOM["h"], GEOM["M"], GEOM["NB"], GEOM["BN"], GEOM["D"]
    cap_words = fa2_tap_buffer_words(2, h, M, NB, BN, D)
    step1, step2 = [8, 8, 8, 8, 8], [1, 3, 2, 7, 5, 8]
    got1, bb, _, _ = bounded_path(step1, h, M, NB, BN, D, 2, cap_words)
    got2, bb, _, _ = bounded_path(step2, h, M, NB, BN, D, 2, cap_words, boundbuf=bb)   # reuse
    ref2, _ = unbounded_path(step2, h, M, NB, BN, D, 2)
    assert _digest(got2) == _digest(ref2)
    ref1, _ = unbounded_path(step1, h, M, NB, BN, D, 2)
    assert _digest(got1) == _digest(ref1)


def test_no_zeroing_between_groups_leaks_history():
    """NEGATIVE: the zeroing after staging is load-bearing on the bounded path -- a shorter sequence reusing a slot after a longer one would
    otherwise bind the previous group's words in its unwritten (upper-triangle / padding) positions."""
    h, M, NB, BN, D = GEOM["h"], GEOM["M"], GEOM["NB"], GEOM["BN"], GEOM["D"]
    # group 1: two sequences with cached context (seqlen_k > seqlen_q: rows write more key blocks); group 2: plain prefills in the same
    # slots whose causal write set is smaller -> their unwritten (beyond-diagonal) words would show group 1's values without the zeroing.
    # (A plain-prefill batch has a row-only write set, so the leak needs a sequence-dependent one -- chunked prefill / prefix cache.)
    seqlens = [4, 4, 4, 4]
    kv_prior = [4, 4, 0, 0]
    cap_words = fa2_tap_buffer_words(2, h, M, NB, BN, D)
    ref, _ = unbounded_path(seqlens, h, M, NB, BN, D, 1, kv_prior=kv_prior)
    ok, _, _, _ = bounded_path(seqlens, h, M, NB, BN, D, 1, cap_words, kv_prior=kv_prior)
    assert _digest(ok) == _digest(ref)                       # with zeroing: identical
    leaky, _, _, _ = bounded_path(seqlens, h, M, NB, BN, D, 1, cap_words, zero_between_groups=False, kv_prior=kv_prior)
    assert [n for n, _ in leaky] == [n for n, _ in ref]      # same names / order ...
    assert _digest(leaky) != _digest(ref)                    # ... different bytes: history leaked into the second group's rectangles
    # and specifically in the second group's sequences, never in the first group
    diffs = [n for (n, a), (_, b) in zip(ref, leaky) if a != b]
    assert diffs and all(("/s2/" in n or "/s3/" in n) for n in diffs)


def test_single_sequence_over_cap_is_its_own_group_and_named():
    """A single sequence whose pitched rectangle exceeds the cap cannot be split by sequence: it forms its own group and the plan says so
    (the driver records cap_exceeded; the residual bound is h*M*W per sequence, quadratic in the length, not in the batch)."""
    h, M, NB, BN, D = GEOM["h"], GEOM["M"], GEOM["NB"], GEOM["BN"], GEOM["D"]
    per = fa2_tap_words_per_seq(h, M, NB, BN, D)
    groups = plan_fa2_tap_groups(3, per, per // 2)
    assert groups == [(0, 1), (1, 2), (2, 3)]
    assert all(fa2_tap_buffer_words(1, h, M, NB, BN, D) > per // 2 for _ in groups)
