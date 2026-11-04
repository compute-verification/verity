"""cuBLAS's bf16 GEMMs at the route pin's routes in the sampled replay: `GemmF32Out_v1`, `CublasSplitKGemmF32Out_v1` (GLM-5.3's
MoE router past the cuteDSL tier, DeepSeek-V4-Pro's compressors) and `CublasSplitKGemm_v1` (the split-K bf16 linears), 153,600 and
48,216 rows of GLM-5.3 B1 TP8's Commit of record that had no registered evaluator.  The row kernels (`cublas_gemm_rows`) are the
Definitions and the H200 capture's models (`bf16_gemm_check.cublas_splitk_rows`, `f32out_rows`), and a row served on another route
(the split-K-off switch's unsplit router) replays unequal, by name."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3", "deepseek-ai/DeepSeek-V4-Pro")

import re

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity_vllm.check.replay import evaluate as EV, index as IX, opening as OP
from verity.ml.definitions.numerics import cublas_gemm_rows as C, rows
from verity.ml.definitions.numerics.kernel_registry import Declined
from verity.ml.definitions.registry import b1, bf16_gemm_check as BC, cublas_bf16 as CB, hopper

HOP = '{"fn":"HopperBF16WgmmaDot16_v1"}'
DOTS = pytest.mark.parametrize("dot", [b1.P.AmpereBF16TcDot16, hopper.HopperBF16WgmmaDot16], ids=["ampere", "hopper"])
SPLITK = {C.SPLITK_F32_OUT: (C.splitk_f32_out, CB.CublasSplitKGemmF32OutWord, "f32"),
          C.SPLITK_BF16: (C.splitk_bf16, CB.CublasSplitKGemmWord, "bf16")}


def _words(rng, *shape, scale=1.0) -> np.ndarray:
    return ((rng.standard_normal(shape) * scale).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def test_the_families_are_registered_with_their_evaluator_and_provenance():
    assert set(C.ROWS) == {d.id for d in BC.ROUTE_DEFINITIONS}
    for f in C.ROWS:
        assert f in rows.ROWS and rows.EVALUATORS[f].startswith(f"verity.ml.definitions.numerics.cublas_gemm_rows.{C.ROWS[f].__name__}")
        assert rows.EVALUATOR_PROVENANCE[f] is C.PROVENANCE


@pytest.mark.parametrize("family", sorted(SPLITK))
@pytest.mark.parametrize("K,SPLIT_K", [(64, 4), (160, 3), (160, 2)], ids=["k16-blocks", "two-chunks-and-a-remainder", "one-chunk-and-a-remainder"])
@DOTS
def test_split_k_row_kernels_are_the_definitions(family, K, SPLIT_K, dot):
    fn, d, _out = SPLITK[family]
    rng = np.random.default_rng(K * SPLIT_K)
    x, w = _words(rng, K), _words(rng, 3, K)
    got = fn({"K": K, "N": 3, "SPLIT_K": SPLIT_K, "DOT": f'{{"fn":"{dot.id}"}}'}, [x, w.reshape(-1)])
    ref = evaluate(bind(d, K=K, N=3, SPLIT_K=SPLIT_K, DOT=dot), x.tolist(), w.reshape(-1).tolist())
    assert got["out"].tolist() == [int(v) for v in ref] and got["0"] is got["out"]


@DOTS
def test_the_f32_out_row_kernel_is_the_definition(dot):
    rng = np.random.default_rng(3)
    x, w = _words(rng, 64), _words(rng, 3, 64)
    got = C.gemm_f32_out({"K": 64, "N": 3, "DOT": f'{{"fn":"{dot.id}"}}'}, [x, w.reshape(-1)])["out"]
    assert got.tolist() == [int(v) for v in evaluate(bind(CB.GemmF32OutWord, K=64, N=3, DOT=dot), x.tolist(), w.reshape(-1).tolist())]


@pytest.mark.parametrize("name", ["glm53/router/mlp.gate", "glm53/linear/indexer.wk_weights_proj"])
def test_row_kernels_are_the_capture_models_at_glm53s_shapes(name):
    """At each M the capture ran, the route pin's split count (`default_cublas_split_k`, the capture's served default) and the kernel's
    words are `cublas_splitk_rows`'s; the unsplit route the split-K-off switch serves is `f32out_rows`, which the split kernel is not."""
    case = C.PROVENANCE["served_default_split_k"][name]
    phase, K, N = name.split("/")[1], case["K"], case["N"]
    family = C.SPLITK_F32_OUT if case["out"] == "f32" else C.SPLITK_BF16
    rng = np.random.default_rng(K + N)
    x, w = _words(rng, 2, K), _words(rng, N, K, scale=0.02)
    for m, split in case["SPLIT_K"].items():
        assert BC.default_cublas_split_k(phase, int(m), K, N) == split, (name, m)
    for split in sorted(set(case["SPLIT_K"].values())):
        want = BC.cublas_splitk_rows(x, w, split, out=case["out"])
        for i in range(len(x)):
            got = SPLITK[family][0]({"K": K, "N": N, "SPLIT_K": split, "DOT": HOP}, [x[i], w.reshape(-1)])["out"]
            assert got.tolist() == want[i].tolist(), (name, split, i)
    unsplit = BC.f32out_rows(x, w)
    assert C.gemm_f32_out({"K": K, "N": N, "DOT": HOP}, [x[0], w.reshape(-1)])["out"].tolist() == unsplit[0].tolist()
    if case["out"] == "f32":
        assert (BC.cublas_splitk_rows(x, w, case["SPLIT_K"]["17"], out="f32") != unsplit).mean() > 0.5


def test_the_evaluator_cites_the_h200_capture_and_agrees_with_the_route_table():
    """The provenance's split counts are the route table's at every (shape, M) the capture ran (the scan maps the same routes),
    and the Definitions' conformance cites the same record."""
    from tests.properties import glm53_bf16_gemm_capture_gpu as X
    from verity_vllm.program.frontend.conformance import status_of

    p = C.PROVENANCE
    assert p["scan"].startswith(BC.CUBLAS_ROUTES["run_record"])
    assert all(re.fullmatch(r"[0-9a-f]{64}", p[k]) for k in ("record", "record_served_default"))
    assert all(re.fullmatch(r"art:[0-9a-f]{64}", a) for a in p["art"].values())
    assert p["variants"] == {"split_k_off": "GO", "served_default": "GO"}
    shapes = {f"{m}/{ph}/{n}": (k, n_) for m, mod in X.MODELS.items()
              for ph, ops in (("router", (("mlp.gate", mod.hidden, mod.experts),)), ("linear", mod.linear), ("f32out", mod.f32out))
              for n, k, n_ in ops}
    for name, case in p["served_default_split_k"].items():
        assert (case["K"], case["N"]) == shapes[name], name
        for m, split in case["SPLIT_K"].items():
            assert BC.default_cublas_split_k(name.split("/")[1], int(m), case["K"], case["N"]) == split, (name, m)
    for d in (CB.GemmF32OutWord, CB.CublasSplitKGemmF32OutWord, CB.CublasSplitKGemmWord):
        assert status_of(d.conformance) == "exact-model-tested"
        assert p["record"][:16] in d.conformance and p["art"]["record"] in d.conformance


def test_an_unregistered_dot_a_refused_split_a_non_finite_lane_and_another_shape_decline_by_name():
    rng = np.random.default_rng(1)
    x, w = _words(rng, 64), _words(rng, 3, 64).reshape(-1)
    st = {"K": 64, "N": 3, "SPLIT_K": 4, "DOT": HOP}
    with pytest.raises(Declined, match="no registered evaluator for CublasSplitKGemmF32Out_v1 with DOT=BlackwellDot16_v9"):
        C.splitk_f32_out(dict(st, DOT='{"fn":"BlackwellDot16_v9"}'), [x, w])
    with pytest.raises(Declined, match=r"CublasSplitKGemm_v1: CublasSplitK: unsupported statics \(SPLIT_K_in_2_to_K_over_16\)"):
        C.splitk_bf16(dict(st, SPLIT_K=5), [x, w])
    inf = x.copy()
    inf[40] = 0x7F80
    with pytest.raises(Declined, match="CublasSplitKGemmF32Out_v1: 3 output coordinates have a non-finite"):
        C.splitk_f32_out(st, [inf, w])
    with pytest.raises(Declined, match="GemmF32Out_v1: x has 64 words and w 128, the signature has K = 64 and N\\*K = 192"):
        C.gemm_f32_out(st, [x, w[:128]])
    with pytest.raises(Declined, match="the row wires 1 args"):
        C.splitk_bf16(st, [x])


# -- one VU end to end: x a committed member, W the registered gate weight ------------------------------------------------------------

K, N, SPLIT, ROW, ROWS = 256, 16, 4, 1, 3
OP_PATH = "model.layers.3.mlp.gate"
X_PATH = "model.layers.3.post_attention_layernorm"
FIELD = "model.layers.3.mlp.gate.weight"
PRE, POST = 40, 8


class _Index:
    """The replay's view of one router row: arg 0 the committed normed hidden row 1, arg 1 the gate's [N][K] weight field between
    two other fields."""

    pnames = ["prompt", "weights"]
    weight_slice = IX.ProgramIndex.weight_slice
    weight_bits = IX.ProgramIndex.weight_bits

    def __init__(self):
        self.fields = [("model.embed", PRE, [PRE], 16), (FIELD, N * K, [N, K], 16), ("model.norm", POST, [POST], 16)]
        self.foff = [0, PRE, PRE + N * K]
        statics = {"K": K, "N": N, "SPLIT_K": SPLIT, "DOT": HOP}
        self.by_i = {7: {"i": 7, "spec": f"CublasSplitKGemmF32Out_v1{{K={K},N={N},SPLIT_K={SPLIT},DOT={HOP}}}",
                         "args": [[["n", 3, 0, K]], [["p", 1, PRE, PRE + N * K]]]}}
        self._d = {"family": C.SPLITK_F32_OUT, "statics": statics, "step": 0, "op_path": OP_PATH, "row": ROW,
                   "members": [("router_logits", N, "f32")]}
        self._x = {"family": "RMSNormFusedCuda_v2", "statics": {}, "step": 0, "op_path": X_PATH, "row": ROW, "members": [("out", K, "bf16")]}

    def describe(self, r):
        return self._d

    def member_at(self, j, lo, hi):
        return (self._x, "out", lo, hi, "bf16") if int(j) == 3 and 0 <= lo < hi <= K else None

    def member_of_port(self, i):
        return {}


def _store(planes):
    entries, blobs = [], {}
    for (op, member), words in planes.items():
        name = f"{op}/{member}"
        blobs[name] = np.ascontiguousarray(words).tobytes()
        dtype = "float32" if words.dtype == np.uint32 else "bfloat16"
        entries.append({"name": name, "element_offset": 0, "encoding": {"dtype": dtype},
                        "identity": {"request_id": "r0", "step": 0, "op_path": op, "output_member": member, "element_range": [0, words.size]}})
    return OP.CommittedStore({"entries": entries}, lambda step, name, b0, b1: blobs[name][b0:b1])


def _rows_of(w):
    out = np.zeros((ROWS, w.size), dtype=w.dtype)
    out[ROW] = w
    return out.reshape(-1)


@pytest.fixture
def world():
    rng = np.random.default_rng(11)
    x, w = _words(rng, K), _words(rng, N, K, scale=0.1)
    table = {"model.embed": _words(rng, PRE), FIELD: w.reshape(-1), "model.norm": _words(rng, POST)}
    served = {"pinned": BC.cublas_splitk_rows(x[None], w, SPLIT, out="f32")[0], "switched": BC.f32out_rows(x[None], w)[0]}
    assert (served["pinned"] != served["switched"]).any()
    return table, x, served


def _replay(table, x, logits):
    reads = []

    def weights(field, lo, hi):
        reads.append((field, lo, hi))
        return table[field][lo:hi]

    store = _store({(X_PATH, "out"): _rows_of(x), (OP_PATH, "router_logits"): _rows_of(logits)})
    return EV._replay_vu(_Index(), store, weights, {"i": 7, "request_id": "r0", "phase": "prefill"}, 0, {}), reads


def test_router_logits_on_the_pinned_route_replay_equal(world):
    table, x, served = world
    rec, reads = _replay(table, x, served["pinned"])
    assert rec["result"] is True, rec.get("why")
    assert rec["outputs"] == [{"member": "router_logits", "width": N, "equal": True, "differing_elements": 0, "first_diff": None}]
    assert reads == [(FIELD, 0, N * K)]


def test_router_logits_served_off_the_pinned_route_are_a_mismatch_by_name(world):
    """The switch's unsplit route computes `GemmF32Out_v1`, which the Program's split-K row is not: the replay names the member."""
    table, x, served = world
    rec, _ = _replay(table, x, served["switched"])
    diff = int((served["pinned"] != served["switched"]).sum())
    assert rec["result"] is False
    assert rec["outputs"][0]["differing_elements"] == diff
    assert f"{OP_PATH}/router_logits row {ROW}" in rec["why"]
