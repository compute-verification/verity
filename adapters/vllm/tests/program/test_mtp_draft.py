"""`registry.mtp_draft`'s `MtpEhNorm_v1` (vLLM d9105ea80's `_fused_eh_norm_kernel`, the MTP draft layer's enorm and hnorm in one
launch) at GLM-5.3's statics against a model of the launch written from its sm_90 PTX: the position select, then `test_dsa_norm`'s
lane-by-lane `ptx_block_model` of `_rms_norm` over the 8192 block on 4 warps, for each half.  The numpy twin the hardware capture
compares with (`dsa_norm_rows.block_rmsnorm_rows`) against the reference evaluator on the same rows, and which positions zero e.
About 40 seconds, at the served statics: the binding the hardware capture holds to the H200's words is this one."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import numpy as np
import pytest

from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate
from verity.definitions.numerics import dsa_norm_rows as NR
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
