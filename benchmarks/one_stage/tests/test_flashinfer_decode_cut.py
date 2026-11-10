"""Decode's FP8 linear (``FlashInferFp8Linear_v1{K,N}``, the M < 32 branch) on GLM-5.3's and DeepSeek-V4-Pro's TP-8 ranks, cut
as the block census cuts a row (the row itself when it fits a block, else each slot's coarsest unit that fits, every unit sized
by `m1_families.sized`) and counted in ANDs a token: the coordinate cut (``COORDS``, N units ``FlashInferFp8Linear_v1{K,N=1}``,
each quantising the whole row) against the parts cut (``PARTS``, one ``FlashInferFp8Quant1x128_v1{K}`` and N coordinates
``ScaledMmFp8BlockSharedScale_v1{K,N=1,G=128}`` reading its q and s).  A coordinate of the old cut is exactly the quantisation
then the GEMM coordinate, so the parts cut saves N - 1 quantisations a row, and takes under a tenth of the old cut's ANDs on
both models' decode shapes.  The units are lowered for real (`estimate`), each distinct leaf once."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import m1_families as M  # noqa: E402

from verity_vllm.pipeline import serving_rows as SRW  # noqa: E402
from verity_vllm.query.program_view import spec_base  # noqa: E402

RANKS = 8
#: one TP-8 rank's ``FlashInferFp8Linear_v1`` rows in a decode step, ``(K, N): rows``, from each model's config.json
#: (`verity_vllm/engine/profiles/hf_configs`) at TP 8
SHAPES = {
    "GLM-5.3": {                # hidden 6144; 78 layers, the first 3 dense, 75 MoE with one shared expert, 21 with a full indexer
        (6144, 2624): 78,       # q_a and kv_a: q_lora_rank 2048 + kv_lora_rank 512 + qk_rope_head_dim 64
        (2048, 2048): 78,       # q_b: 64 heads of 192 + 64, over 8 ranks
        (2048, 6144): 78,       # o_proj: 64 heads of 256, over 8 ranks
        (6144, 512): 75,        # the shared expert's gate and up: 2 x 2048, over 8 ranks
        (256, 6144): 75,        # the shared expert's down
        (6144, 3072): 3,        # the dense MLP's gate and up: 2 x 12288, over 8 ranks
        (1536, 6144): 3,        # the dense MLP's down
        (2048, 4096): 21,       # the indexer's q_b: 32 heads of 128
    },
    "DeepSeek-V4-Pro": {        # hidden 7168; 61 layers, each with one shared expert
        (7168, 2048): 61,       # wq_a and wkv: q_lora_rank 1536 + one kv head of 512
        (1536, 8192): 61,       # wq_b: 128 heads of 512, over 8 ranks
        (2048, 7168): 61,       # wo_b: o_groups 16 x o_lora_rank 1024, over 8 ranks
        (7168, 768): 61,        # the shared expert's gate and up: 2 x 3072, over 8 ranks
        (384, 7168): 61,        # the shared expert's down
    },
}


def linear(K: int, N: int) -> str:
    return f"FlashInferFp8Linear_v1{{K={K},N={N}}}"


def quant(K: int) -> str:
    return f"FlashInferFp8Quant1x128_v1{{K={K}}}"


def coord(K: int) -> str:
    return f"ScaledMmFp8BlockSharedScale_v1{{K={K},N=1,G=128}}"


@pytest.fixture(scope="module")
def sized():
    sys.setrecursionlimit(100000)
    memo: dict = {}
    return lambda spec: M.sized(spec, memo)


def fits(e: dict) -> bool:
    return "ands" in e and not e["unpieced"] and max(e["block"]["rows_log"], e["block"]["est_log"]) <= M.K_MAX


def cut(spec: str, sized, parts: bool) -> list[tuple[str, int]]:
    """The census's cut of a row of ``spec``: the row itself when it fits, else each slot's coarsest fitting unit, a slot per
    part (``parts``: `serving_templates.PARTS`) or the one slot of the row's coordinates (``COORDS``), at one output a
    coordinate (the census picks the GEMM's coordinates of c outputs by cost, `block_census.width_pick`, which this comparison
    of quantisations leaves out)."""
    if fits(sized(spec)):
        return [(spec, 1)]
    units = [(u, n) for u, n in SRW.unit_templates(spec)[1:] if SRW.coord_width(u) <= 1]
    base = spec_base(spec)
    fams = [p[0] for p in SRW.PARTS[base]] if parts else [base]
    return [next((u, n) for u, n in units if spec_base(u) == fam and fits(sized(u))) for fam in fams]


def ands_a_token(model: str, sized, parts: bool) -> int:
    return RANKS * sum(rows * n * sized(u)["ands"] for (K, N), rows in SHAPES[model].items()
                       for u, n in cut(linear(K, N), sized, parts))


@pytest.mark.parametrize("model", list(SHAPES))
def test_the_parts_cut_quantises_each_row_once_where_the_old_cut_quantises_it_n_times(model, sized):
    for K, N in SHAPES[model]:
        assert cut(linear(K, N), sized, parts=False) == [(linear(K, 1), N)]
        assert cut(linear(K, N), sized, parts=True) == [(quant(K), 1), (coord(K), N)]


def test_an_old_coordinate_is_the_quantisation_then_the_gemm_coordinate(sized):
    for K in sorted({K for shapes in SHAPES.values() for K, _N in shapes}):
        assert sized(linear(K, 1))["ands"] == sized(quant(K))["ands"] + sized(coord(K))["ands"], K


@pytest.mark.parametrize("model", list(SHAPES))
def test_the_parts_cut_saves_n_minus_one_quantisations_a_row_and_takes_under_a_tenth(model, sized):
    old, new = ands_a_token(model, sized, parts=False), ands_a_token(model, sized, parts=True)
    assert old - new == RANKS * sum(rows * (N - 1) * sized(quant(K))["ands"] for (K, N), rows in SHAPES[model].items())
    assert 10 * new < old
