"""`BatchedGemm_v1{B,K,N,DOT}` in the sampled replay: GLM-5.3's MLA absorbs (``torch.bmm`` on cuBLAS against `W_UK_T` / `W_UV`),
44,772 rows per rank of the TP8 Commit that had no registered evaluator.  The row kernel (`batched_gemm_rows`) is the Definition and
the H200 capture's model (`bf16_gemm_check.bmm_rows`).  Its weight operand is a strided read of the absorbed weight, which the Build
stores as "P" progressions (B*N runs of K leaves at stride N); the replay gathers every leaf from the one field without expanding
them, and names a gather that spans fields."""
from __future__ import annotations

import re

import numpy as np
import pytest
import torch
import torch.nn as nn
from verity.circuits.defs import bind, conformance_label
from verity.circuits.evaluation import evaluate

from tests.program.test_dsa_derive import STANDIN, _bf16, served  # noqa: F401  (the H200 Build fixture)
from verity_vllm.check.replay import index as IX
from verity_vllm.check.replay import opening as OP
from verity_vllm.definitions.numerics import batched_gemm_rows, rows
from verity_vllm.definitions.numerics.batched_gemm_rows import batched_gemm
from verity_vllm.definitions.numerics.kernel_registry import Declined
from verity_vllm.definitions.registry import b1, hopper
from verity_vllm.definitions.registry import bf16_gemm_check as BC
from verity_vllm.definitions.registry import cublas_bf16 as CB
from verity_vllm.program import compact as CP
from verity_vllm.query.manifest import format as RM

HOP = '{"fn":"HopperBF16WgmmaDot16_v1"}'


def _words(rng, *shape, scale=1.0) -> np.ndarray:
    return ((rng.standard_normal(shape) * scale).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def test_the_family_is_registered_with_its_evaluator():
    assert "BatchedGemm_v1" in rows.EVALUATORS and "BatchedGemm_v1" in rows.ROWS
    assert rows.EVALUATORS["BatchedGemm_v1"].startswith("verity_vllm.definitions.numerics.batched_gemm_rows.batched_gemm")


@pytest.mark.parametrize("B,K,N", [(2, 32, 3), (3, 48, 5)])
@pytest.mark.parametrize("dot", [b1.P.AmpereBF16TcDot16, hopper.HopperBF16WgmmaDot16], ids=["ampere", "hopper"])
def test_row_kernel_is_the_definition(B, K, N, dot):
    rng = np.random.default_rng(B * K + N)
    x, w = _words(rng, B, K), _words(rng, B, N, K)
    got = batched_gemm({"B": B, "K": K, "N": N, "DOT": f'{{"fn":"{dot.id}"}}'}, [x.reshape(-1), w.reshape(-1)])
    ref = evaluate(bind(CB.BatchedGemmWord, B=B, K=K, N=N, DOT=dot), x.reshape(-1).tolist(), w.reshape(-1).tolist())
    assert got["out"].tolist() == [int(v) for v in ref] and got["0"] is got["out"]


@pytest.mark.parametrize("K,N", [(192, 512), (512, 256)], ids=["q_nope_absorb", "v_absorb"])
def test_row_kernel_is_the_capture_model_at_glm53s_tp8_shapes(K, N):
    """The capture (`glm53_bf16_gemm_capture_gpu.py`, phase bmm) holds cuBLAS's words to `bmm_rows` on (B, M, K) x (B, K, N)."""
    rng = np.random.default_rng(K)
    x, wt = _words(rng, 8, 4, K), _words(rng, 8, K, N, scale=0.05)
    want = BC.bmm_rows(x, wt)
    for m in range(x.shape[1]):
        got = batched_gemm({"B": 8, "K": K, "N": N, "DOT": HOP}, [x[:, m].reshape(-1), wt.transpose(0, 2, 1).reshape(-1)])["out"]
        assert got.tolist() == want[:, m].reshape(-1).tolist()


def test_the_evaluator_cites_the_h200_capture_at_every_shape_and_m_the_capture_runs():
    """The replay record's `evaluator_provenance` for BatchedGemm_v1 is the capture run that held cuBLAS's bmm to `bmm_rows`: GO in
    both configurations at every (K, N) and M the capture script runs, among them the TP8 Commit's prefill (256) and decode (1) M; and
    the Definition's conformance cites the same record."""
    from tests.properties import glm53_bf16_gemm_capture_gpu as X

    p = rows.EVALUATOR_PROVENANCE["BatchedGemm_v1"]
    assert p is batched_gemm_rows.PROVENANCE
    assert all(re.fullmatch(r"[0-9a-f]{64}", p[k]) for k in ("record", "record_served_default"))
    assert all(re.fullmatch(r"art:[0-9a-f]{64}", a) for a in p["art"].values())
    assert p["variants"] == {"split_k_off": "GO", "served_default": "GO"}
    assert {n: (c["B"], c["K"], c["N"], c["M"]) for n, c in p["cases"].items()} == {n: (X.NHL, k, n_, list(X.LINEAR_M)) for n, k, n_ in X.BMM}
    assert all(c["mismatch"] == 0 and c["words"] == c["B"] * c["N"] * sum(c["M"]) for c in p["cases"].values())
    assert {1, 256} <= set(X.LINEAR_M)
    for d in (CB.BatchedGemmWord, CB.BatchedGemm):
        assert conformance_label(d.conformance) == "exact-model-tested"
        assert p["record"][:16] in d.conformance and p["art"]["record"] in d.conformance


def test_an_unregistered_dot_a_non_finite_lane_and_another_shape_decline_by_name():
    rng = np.random.default_rng(1)
    x, w = _words(rng, 2, 32), _words(rng, 2, 3, 32)
    st = {"B": 2, "K": 32, "N": 3, "DOT": HOP}
    with pytest.raises(Declined, match="no registered evaluator for BatchedGemm_v1 with DOT=BlackwellDot16_v9"):
        batched_gemm(dict(st, DOT='{"fn":"BlackwellDot16_v9"}'), [x, w])
    inf = x.copy()
    inf[1, 5] = 0x7F80
    with pytest.raises(Declined, match="BatchedGemm_v1: 3 output coordinates have a non-finite"):
        batched_gemm(st, [inf, w])
    with pytest.raises(Declined, match="the signature has B\\*K = 64 and B\\*N\\*K = 192"):
        batched_gemm(st, [x, w[:, :2]])


# -- one VU end to end: x a committed member, W the registered field read through its stored progressions --------------------------

B, K, N, ROW, ROWS = 2, 32, 24, 1, 3
OP_PATH = "model.layers.0.self_attn"
X_PATH = "model.layers.0.self_attn.q_b_proj"
FIELD = "model.layers.0.self_attn.W_UK_T"
PRE, POST = 40, 8


class _Index:
    """The replay's view of one BatchedGemm row: arg 0 the committed q_nope member of row 3, arg 1 the W_UK_T field ([B][K][N], the
    parameter's order) between two other fields."""

    pnames = ["prompt", "weights"]
    weight_slice = IX.ProgramIndex.weight_slice
    weight_bits = IX.ProgramIndex.weight_bits

    def __init__(self, w_items):
        self.fields = [("model.embed", PRE, [PRE], 16), (FIELD, B * K * N, [B, K, N], 16), ("model.norm", POST, [POST], 16)]
        self.foff = [0, PRE, PRE + B * K * N]
        spec = f"BatchedGemm_v1{{B={B},K={K},N={N},DOT={HOP}}}"
        self.by_i = {7: {"i": 7, "spec": spec, "args": [[["n", 3, 0, B * K]], w_items]}}
        self._d = {"family": "BatchedGemm_v1", "statics": {"B": B, "K": K, "N": N, "DOT": HOP}, "step": 0, "op_path": OP_PATH, "row": ROW,
                   "members": [("ql_nope", B * N, "bf16")]}
        self._x = {"family": "Gemm_v2", "statics": {}, "step": 0, "op_path": X_PATH, "row": ROW, "members": [("q_nope", B * K, "bf16")]}

    def describe(self, r):
        return self._d

    def member_at(self, j, lo, hi):
        return (self._x, "q_nope", lo, hi, "bf16") if int(j) == 3 and 0 <= lo < hi <= B * K else None

    def member_of_port(self, i):
        return {}


def _leaves(runs) -> list[int]:
    return [j for _sp, _i, lo, hi in runs for j in range(lo, hi)]


def _w_items(base: int = PRE) -> list:
    """W^T as the Build stores it: for each (b, n) the K leaves `base + b*K*N + k*N + n`, "P" progressions of unit runs (the
    compressor coalesces the two adjacent leaves where member b's last progression meets member b+1's first)."""
    runs = [("p", 1, base + (b * K + k) * N + n, base + (b * K + k) * N + n + 1) for b in range(B) for n in range(N) for k in range(K)]
    items = CP.file_items_from_runs(runs)
    assert all(it[0] == "P" and it[1] == 1 and it[4] == 0 for it in items)
    assert _leaves(CP.runs_from_file_items(items)) == _leaves(runs)
    return items


def _n_runs(items) -> int:
    return sum(it[-1] for it in items)


def _store(planes):
    entries, blobs = [], {}
    for (op, member), words in planes.items():
        name = f"{op}/{member}"
        blobs[name] = np.ascontiguousarray(words).tobytes()
        entries.append({"name": name, "element_offset": 0, "encoding": {"dtype": "bfloat16"},
                        "identity": {"request_id": "r0", "step": 0, "op_path": op, "output_member": member, "element_range": [0, words.size]}})
    return OP.CommittedStore({"entries": entries}, lambda step, name, b0, b1: blobs[name][b0:b1])


@pytest.fixture
def world():
    rng = np.random.default_rng(11)
    x, wt = _words(rng, B, K), _words(rng, B, K, N, scale=0.1)
    table = {"model.embed": _words(rng, PRE), FIELD: wt.reshape(-1), "model.norm": _words(rng, POST)}
    honest = BC.bmm_rows(x.reshape(B, 1, K), wt).reshape(-1)

    def rows_of(w):
        out = np.zeros((ROWS, w.size), dtype=np.uint16)
        out[ROW] = w
        return out.reshape(-1)

    planes = {(X_PATH, "q_nope"): rows_of(x.reshape(-1)), (OP_PATH, "ql_nope"): rows_of(honest)}
    return table, planes


# -- the Build's own stored rows ------------------------------------------------------------------------------------------------------

NH, P, L, V = 2, 16, 32, 16


class _Absorbs(nn.Module):
    """GLM-5.3's two absorbs on `W_UK_T` / `W_UV` laid out as `MLAAttention.process_weights_after_loading` leaves them: views of
    kv_b_proj's transposed weight (strides as the capture's SERVED_STRIDE)."""

    def __init__(self) -> None:
        super().__init__()
        kv = _bf16(np.random.default_rng(5), L, NH * (P + V), sd=0.25).view(L, NH, P + V)
        self.W_UK_T = nn.Parameter(kv[..., :P].permute(1, 2, 0), requires_grad=False)
        self.W_UV = nn.Parameter(kv[..., P:].transpose(0, 1), requires_grad=False)

    def forward(self, q, a):
        return torch.bmm(q.transpose(0, 1), self.W_UK_T).transpose(0, 1), torch.bmm(a.transpose(0, 1), self.W_UV).transpose(0, 1)


def RM_statics(spec: str) -> dict:
    return RM.spec_statics(spec)
