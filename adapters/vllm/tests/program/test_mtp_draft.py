"""`registry.mtp_draft`'s `MtpEhNorm_v1` (vLLM d9105ea80's `_fused_eh_norm_kernel`, the MTP draft layer's enorm and hnorm in one
launch) at GLM-5.3's statics against a model of the launch written from its sm_90 PTX: the position select, then `test_dsa_norm`'s
lane-by-lane `ptx_block_model` of `_rms_norm` over the 8192 block on 4 warps, for each half.  The numpy twin the hardware capture
compares with (`dsa_norm_rows.block_rmsnorm_rows`) against the reference evaluator on the same rows, and which positions zero e.
About 40 seconds, at the served statics: the binding the hardware capture holds to the H200's words is this one."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import numpy as np
import pytest

from verity.circuits import boolean as BL
from verity.circuits.boolean import trace as TR
from verity.circuits.defs import bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.evaluation import evaluate
from verity.commitments.windows import interface as IF
from verity.definitions.boolean import boolean_version
from verity.definitions.numerics import dsa_norm_rows as NR
from verity.definitions.registry import boolean_mtp_draft as BM
from verity.definitions.registry import dsa_norm as DN
from verity.definitions.registry import mtp_draft as MD
from tests.program.test_dsa_norm import OVERFLOW, _bf16_row, _edge_rows, _w, ptx_block_model

H, BLOCK, WARPS, EPS = 6144, 8192, 4, 1e-5


def ptx_launch_model(pos: int, e: list[int], p: list[int], ew: list[int], hw: list[int]) -> list[int]:
    """One program of `_fused_eh_norm_kernel` as its PTX runs it: `setp.eq.b64 pos, 0` and `selp.f32 0f00000000` on every f32 element
    of e (+0, whose bf16 word is 0), then the two `_rms_norm`s, e's into the first H columns and p's into the next."""
    e0 = [0] * len(e) if pos == 0 else e
    return ptx_block_model(e0, ew, EPS, BLOCK, WARPS) + ptx_block_model(p, hw, EPS, BLOCK, WARPS)


def _rows(seed: int) -> list[np.ndarray]:
    """`test_dsa_norm`'s edge rows at 6144 but its overflowing one (the PTX model's rationals are finite): served scales, 2^40,
    squares that are f32 subnormals, bf16 subnormals, all zeros, one outlier over tiny values, and +-0 / +-1 alternating."""
    return [r for i, r in enumerate(_edge_rows(np.random.default_rng(seed), H)) if i != OVERFLOW]


def test_the_definition_is_the_ptx_launch() -> None:
    """Each edge row as e at position 7 against a served-scale p, and as p at position 0, where e (served-scale) is zeroed."""
    rng = np.random.default_rng(63)
    fn, ew, hw = bind(MD.MtpEhNorm, H=H, BLOCK=BLOCK, WARPS=WARPS, EPS=EPS), _w(rng, H), _w(rng, H)
    served = _bf16_row(rng, H, 1.0).tolist()
    for i, x in enumerate(_rows(64)):
        for pos, e, p in ((7, x.tolist(), served), (0, served, x.tolist())):
            want = ptx_launch_model(pos, e, p, ew.tolist(), hw.tolist())
            assert list(evaluate(fn, pos, e, p, ew.tolist(), hw.tolist())) == want, (i, pos)


def test_twin_matches_the_reference_on_edge_rows() -> None:
    """`block_rmsnorm_rows` at (6144, 8192, 4) is `TritonBlockRMSNorm_v1`'s words, the overflowing row and inf and NaN rows too."""
    rng = np.random.default_rng(65)
    fn, w = bind(DN.TritonBlockRMSNorm, N=H, BLOCK=BLOCK, WARPS=WARPS, EPS=EPS), _w(rng, H)
    rows = _edge_rows(rng, H)
    inf, nan = _bf16_row(rng, H, 1.0), _bf16_row(rng, H, 1.0)
    inf[3], nan[6000] = 0x7F80, 0x7FC1
    rows += [inf, nan]
    twin = NR.block_rmsnorm_rows(np.stack(rows), w, EPS, BLOCK, WARPS)["out"]
    for i, x in enumerate(rows):
        assert twin[i].tolist() == list(evaluate(fn, x.tolist(), w.tolist())), i


@pytest.mark.parametrize("pos,zeroed", [(0, True), (1, False), (1 << 32, False), ((1 << 64) - 1, False), (1 << 63, False)])
def test_only_position_zero_zeroes_the_embedding(pos: int, zeroed: bool) -> None:
    """The whole int64 is compared (`setp.eq.b64`): 2^32, whose low word is 0, and -1 and INT64_MIN keep e."""
    rng = np.random.default_rng(66)
    fn = bind(MD.MtpEhNorm, H=768, BLOCK=1024, WARPS=4, EPS=EPS)
    norm = bind(DN.TritonBlockRMSNorm, N=768, BLOCK=1024, WARPS=4, EPS=EPS)
    e, p, ew, hw = (_bf16_row(rng, 768, 1.0).tolist() for _ in range(4))
    got = list(evaluate(fn, pos, e, p, ew, hw))
    assert got[:768] == list(evaluate(norm, [0] * 768 if zeroed else e, ew))
    assert got[768:] == list(evaluate(norm, p, hw))
    assert (got[:768] == list(evaluate(norm, [0] * 768, ew))) == zeroed


def _staged(whole, pos: int, e: list[int], p: list[int], ew: list[int], hw: list[int]) -> list[int]:
    """`eh_cut`'s units run in turn: the select, then each half's scale and its apply units in column order."""
    (zero, n0), (sumsq, n1), (apply, n2) = MD.eh_cut(whole)
    assert (n0, n1, n2) == (1, 2, 2 * whole.bindings["H"] // MD.APPLY_G)
    G, out = MD.APPLY_G, []
    for x, w in ((list(evaluate(zero, pos, e)), ew), (p, hw)):
        (r,) = evaluate(sumsq, x)
        out += [v for j in range(len(x) // G) for v in evaluate(apply, x[j * G:(j + 1) * G], w[j * G:(j + 1) * G], r)]
    return out


def test_the_cut_returns_the_launchs_words() -> None:
    """At a 768 row in its 1024 block, each edge row as e at positions 0 and 7 and as p; at the served statics one served-scale pair
    at each position: the select, two scales and 2H / 64 apply units joined are `MtpEhNorm_v1`'s words."""
    rng = np.random.default_rng(67)
    small = bind(MD.MtpEhNorm, H=768, BLOCK=1024, WARPS=4, EPS=EPS)
    ew, hw = _w(rng, 768).tolist(), _w(rng, 768).tolist()
    served = _bf16_row(rng, 768, 1.0).tolist()
    for i, x in enumerate(_edge_rows(np.random.default_rng(68), 768)):
        for pos, e, p in ((0, x.tolist(), served), (7, x.tolist(), served), (7, served, x.tolist())):
            assert _staged(small, pos, e, p, ew, hw) == list(evaluate(small, pos, e, p, ew, hw)), (i, pos)
    whole = bind(MD.MtpEhNorm, H=H, BLOCK=BLOCK, WARPS=WARPS, EPS=EPS)
    e, p, ew, hw = (_bf16_row(rng, H, 1.0).tolist() for _ in range(4))
    for pos in (0, 7):
        assert _staged(whole, pos, e, p, ew, hw) == list(evaluate(whole, pos, e, p, ew, hw)), pos


@pytest.mark.parametrize("name", ["MtpEhZero", "TritonBlockRMSNormSumSq", "MtpEhNorm"])
def test_on_bits_each_is_the_switchs_boolean_version_and_returns_its_words(name: str) -> None:
    """At a row of 256 in a block of 512 on one warp (its second run masked): `boolean_mtp_draft`'s `_v2` is what the switch takes
    the word Definition to, and on `test_dsa_norm`'s edge rows, at positions 0, 7 and 2^32, returns its words."""
    N, small = 256, {"BLOCK": 512, "WARPS": 1, "EPS": EPS}
    word = {"MtpEhZero": lambda: bind(MD.MtpEhZero, H=N), "MtpEhNorm": lambda: bind(MD.MtpEhNorm, H=N, **small),
            "TritonBlockRMSNormSumSq": lambda: bind(MD.TritonBlockRMSNormSumSq, N=N, **small)}[name]()
    bits = boolean_version(word)
    assert BL.is_boolean(bits) and TR.word_of(bits).id == word.id and bits.id.startswith(f"{name}_v2")
    rows = np.stack(_edge_rows(np.random.default_rng(69), N))
    R = len(rows)
    pos = np.array([(0, 7, 1 << 32)[i % 3] for i in range(R)], dtype=np.uint64)
    served = np.stack([_bf16_row(np.random.default_rng(70 + i), N, 1.0) for i in range(3)])
    ops = {"MtpEhZero": (pos, rows), "TritonBlockRMSNormSumSq": (rows,),
           "MtpEhNorm": (pos, rows, np.roll(rows, 1, axis=0), np.tile(served[1], (R, 1)), np.tile(served[2], (R, 1)))}[name]
    got = BS.evaluate_words(bits, *ops).reshape(R, -1).tolist()
    for i in range(R):
        want = evaluate(word, *(int(a[i]) if a.ndim == 1 else a[i].tolist() for a in ops))
        assert got[i] == ([want] if isinstance(want, int) else list(want)), (name, i)


def test_the_cut_on_bits_is_the_word_cut() -> None:
    """At the served statics `boolean_mtp_draft.eh_cut` of `MtpEhNorm_v2` is `mtp_draft.eh_cut` of `MtpEhNorm_v1` unit by unit:
    each unit's word view is the word cut's unit, as many of each."""
    word = bind(MD.MtpEhNorm, H=H, BLOCK=BLOCK, WARPS=WARPS, EPS=EPS)
    bits = boolean_version(word)
    assert [(TR.word_of(u).id, n) for u, n in BM.eh_cut(bits)] == [(u.id, n) for u, n in MD.eh_cut(word)]


def test_the_scale_on_bits_refuses_a_register_live_in_some_threads_only() -> None:
    """96 of a 128 block on two warps leaves threads 48-63's register masked and the rest live: refused by name."""
    with pytest.raises(ValueError, match="live in some threads and masked in others"):
        BM.live_registers(96, 128, 2)
    assert BM.live_registers(H, BLOCK, WARPS) == [k // 8 < 6 for k in range(64)]


def test_each_unit_of_the_cut_fits_a_proof_block() -> None:
    """At the served statics each unit fits a proof block: the select reads the 8 B position and e and writes e, a scale reads a
    half and writes its 4 B word, and an apply unit reads 64 words of a half, their weights and the scale and returns 64 words."""
    whole = bind(MD.MtpEhNorm, H=H, BLOCK=BLOCK, WARPS=WARPS, EPS=EPS)
    rows = {u.id: [2 * w for _, w in IF.layout(u).rows + IF.layout(u).outs] for u, _ in MD.eh_cut(whole)}
    for u, _ in MD.eh_cut(whole):
        IF.check_proof_block(u)
    assert list(rows.values()) == [[8, 2 * H, 2 * H], [2 * H, 4], [128, 128, 4, 128]], rows
