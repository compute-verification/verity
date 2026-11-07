"""`stage.py`'s FP8 linear instances: below 32 rows FlashInfer's Definition with the op's output alone; from 32 rows DeepGEMM's,
whose expected values are the op's inner ``_C::per_token_group_fp8_quant`` outputs and then its own, each read on the call's own
rank (two ranks whose calls share sequence numbers must not read each other's quant)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
OP = "vllm::dynamic_flashinfer_deepgemm_blockscale_gemm"


def _stage(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "setrecursionlimit", lambda n: None)
    spec = importlib.util.spec_from_file_location("spot_stage", HERE / "stage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Cap:
    """Each rank's call at `M` rows and its inner quant; a tensor reads as (rank, name)."""

    def __init__(self, M: int, K: int = 128, N: int = 2):
        self.ranks = [0, 1]
        self.events = {r: [{"seq": 0, "op": OP, "parent": None,
                            "in": {"input": {"shape": [M, K], "sha512": f"x{r}"}, "weight": {"param": f"w{r}", "shape": [N, K]},
                                   "weight_scale": {"param": f"s{r}", "shape": [1, 1]}},
                            "out": {"ret0": {"sha512": f"y{r}"}}},
                           {"seq": 1, "op": "_C::per_token_group_fp8_quant", "parent": 0,
                            "in": {}, "out": {"output_q": {"sha512": f"q{r}"}, "output_s": {"sha512": f"qs{r}"}}}]
                       for r in self.ranks}

    def opened(self, e, weights=False):
        return True

    def real_rows(self, r, rows):
        return rows

    def cells(self, label, call, *dims):
        yield 0, 0

    def tensor(self, r, ref):
        name = ref.get("sha512", ref.get("param"))
        assert name.endswith(str(r)), f"rank {r} read {name}"
        a = np.empty((1, 1), dtype=object)
        a[0, 0] = (r, name)
        return a


def _names(xs) -> list:
    return [np.asarray(x, dtype=object).reshape(-1)[0] for x in xs]


def test_deepgemm_rows_expect_their_own_ranks_quant(monkeypatch) -> None:
    stage = _stage(monkeypatch)
    insts = list(stage.fp8_linear_instances(_Cap(M=32)))
    assert [(i[0], i[1]) for i in insts] == [("DeepGemmFp8Linear_K128", "DeepGemmFp8Linear_v1{K=128,N=1}")] * 2
    for label, block, (r, *_), args_of, want_of in insts:
        assert _names(want_of()) == [(r, f"q{r}"), (r, f"qs{r}"), (r, f"y{r}")]
        assert _names(args_of()) == [(r, f"x{r}"), (r, f"w{r}"), (r, f"s{r}")]


def test_flashinfer_rows_expect_the_output_alone(monkeypatch) -> None:
    stage = _stage(monkeypatch)
    insts = list(stage.fp8_linear_instances(_Cap(M=31)))
    assert {i[0] for i in insts} == {"FlashInferFp8Linear_K128"}
    assert [_names(want_of()) for *_, want_of in insts] == [[(0, "y0")], [(1, "y1")]]
