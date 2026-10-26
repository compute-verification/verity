"""The sm_90 DeepGEMM capture without a GPU: its comparison jobs over operands and words made here (the kernels' words stood in for
by the twins), a flipped word, quant byte, weight byte or scale found and located, the branch taps and operand checks on CPU
tensors, the JIT scan on DeepGEMM's generated sources, the bindings, and the verdict (the GPU run is
`tests/properties/deepgemm_capture_gpu.py --out ...` on an H200)."""
from __future__ import annotations

import copy
import sys
import types

import numpy as np
import pytest
import torch

from tests.properties import deepgemm_capture_gpu as X
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity_vllm.program.kernels import deepgemm_twins as T
from verity_vllm.program.kernels import derived_rows as R
from verity_vllm.program.registry import fp8 as F
from verity_vllm.properties.record import REGISTRY, digest_of

K, N, GQ = 256, 200, X.GROUP
NW = 300                                                                    # 3 x 2 weight blocks, the last one partial
QUANT_KW = {"group_size": GQ, "column_major_scales": True, "use_ue8m0": True}


def _rows(rng: np.random.Generator, M: int) -> np.ndarray:
    """bf16 rows as the GPU script's `_activation` makes them: from M >= 4 row 1 zero, row 2 one outlier over tiny values, row 3 bf16
    subnormals (lane 0 of each group nonzero); from M >= 5 row 4 is row 3 with each group's lane-0 elements zero."""
    x = rng.standard_normal((M, K)).astype(np.float32) * np.exp2(rng.uniform(-3, 3, (M, 1))).astype(np.float32)
    if M >= 4:
        x[1] = 0
        x[2] *= np.float32(2.0 ** -9)
        x[2, 77] = 30
        x[3] *= np.float32(2.0 ** -130)
    b = (x.view(np.uint32) >> 16).astype(np.uint16)
    if M >= 4:
        b[3, ::GQ] = 0x0040
    if M >= 5:
        b[4] = b[3]
        b[4].reshape(-1, GQ)[:, [0, 1, 64, 65]] = 0
    return b


def _e4m3(rng: np.random.Generator, *shape: int) -> np.ndarray:
    b = rng.integers(0, 256, shape).astype(np.uint8)
    return np.where((b & 0x7F) == 0x7F, b ^ 1, b).astype(np.uint8)


def _scales(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return np.exp2(rng.uniform(-12, -6, shape)).astype(np.float32).view(np.uint32)


def _columns(r: int, h: int | None = None) -> set[int]:
    return {0, N - 1, (r * 7919) % N} if h is None else {0, EN - 1, (r * 7919 + h) % EN}


def _total(results: list[dict]) -> dict:
    assert not [r for r in results if "error" in r], results
    return {k: sum(r[k] for r in results) for k in X.TOTALS}


def _zero(**kw) -> dict:
    return {**{k: 0 for k in X.TOTALS}, **kw}


# --- linear ----------------------------------------------------------------------------------------------------------------------

def _linear_jobs(tmp_path, branch: str, *, M: int = 6, flip_word: bool = False, flip_quant: bool = False,
                 edit=None) -> list[dict]:
    """One launch of a K = 256, N = 200 linear (a partial weight block) on `_rows`, saved at `sample_rows` as the GPU script saves
    it: DeepGEMM's branch with its (q, s) and the GEMM over them, or FlashInfer's words, a non-finite lane the canonical NaN except
    at the sampled columns, where it is the reference's word (as the GPU's must be)."""
    rng = np.random.default_rng(1)
    rows = X.sample_rows(M)
    x = _rows(rng, M)
    w = _e4m3(rng, N, K)
    ws = T.upcast_e8m0(rng.integers(X.E8M0_LO, X.E8M0_HI + 1, (X.cdiv(N, GQ), K // GQ)))
    arrays = {"x": x[rows], "w": w, "ws": ws}
    if branch == "deepgemm":
        q, s = T.ue8m0_group_quant_rows(x)
        if flip_quant:
            q[0, 9] ^= 0x80
        out = np.stack([R.scaled_mm_fp8_block_row(q[r], s[r], w, ws, G=GQ)[0] for r in rows])
        arrays |= {"q": q[rows], "s": s[rows]}
    else:
        out, nf = map(np.stack, zip(*(T.flashinfer_linear_row(x[r], w, ws) for r in rows)))
        out[nf] = 0x7FC0
        coord = bind(F.ScaledMmFp8BlockCoordinate, K=K, G=GQ)
        for i, r in enumerate(rows):
            if r not in X.reference_rows(M):
                continue
            xq, sx = T.flashinfer_quant_1x128_row(x[r])
            for n in _columns(r):
                if nf[i, n]:
                    out[i, n] = evaluate(coord, xq.tolist(), sx.tolist(), w[n].tolist(), ws[n // GQ].tolist())[0]
    if flip_word:
        out[2, 150] ^= 1
    if edit:
        edit(out)
    job = {"phase": "linear", "model": "dsv4", "case": "ragged", "K": K, "N": N, "M": M, "branch": branch, "row_map": rows,
           "paths": X.B._save(tmp_path / f"{branch}_M{M}", out=out, **arrays)}
    return X.row_jobs(job, N * K // 32)


def test_deepgemm_jobs_take_the_kernels_quant_and_words(tmp_path) -> None:
    jobs = _linear_jobs(tmp_path, "deepgemm")
    assert sorted(r for j in jobs for r in j["reference_rows"]) == [0, 2, 3, 4, 5]
    t = _total([X.compare(j) for j in jobs])
    assert t == _zero(words=6 * N, quant_rows=6, reference_quant_rows=5, reference_coords=sum(len(_columns(r)) for r in (0, 2, 3, 4, 5)))


def test_flashinfer_jobs_meet_its_nonfinite_rows(tmp_path) -> None:
    """Row 3's groups are all subnormal with lane 0 nonzero: FlashInfer's FTZ compare takes it as amax and scales by +inf, so every
    word is non-finite, and on its sampled columns the GPU's word is the reference coordinate's."""
    t = _total([X.compare(j) for j in _linear_jobs(tmp_path, "flashinfer")])
    assert t == _zero(words=6 * N, nonfinite_words=N, reference_quant_rows=5, nonfinite_reference=len(_columns(3)),
                      reference_coords=sum(len(_columns(r)) for r in (0, 2, 3, 4, 5)))


def test_row_chunks_split_the_rows_and_the_reference(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(X.B, "JOB_LANE_STEPS", 2 * N * K // 32)
    jobs = _linear_jobs(tmp_path, "deepgemm")
    assert [(j["rows"], j["reference_rows"]) for j in jobs] == [([0, 2], [0]), ([2, 4], [2, 3]), ([4, 6], [4, 5])]
    chunked = _total([X.compare(j) for j in jobs])
    monkeypatch.undo()
    (whole,) = _linear_jobs(tmp_path / "one", "deepgemm")
    assert chunked == _total([X.compare(whole)])


def test_rows_past_full_rows_are_sampled_at_the_tile_edges(tmp_path) -> None:
    """Past `FULL_ROWS` the jobs compare `sample_rows` only: the planted rows, both sides of every multiple of 64, the last row;
    the reference still lands on rows 0, 2, 3, 4 and M - 1."""
    assert X.sample_rows(256) == list(range(256))
    assert X.sample_rows(300) == [0, 1, 2, 3, 4, 63, 64, 127, 128, 191, 192, 255, 256, 299]
    assert len(X.sample_rows(1024)) == 36 and X.sample_rows(1024)[-2:] == [960, 1023]
    jobs = _linear_jobs(tmp_path, "deepgemm", M=300)
    rows = X.sample_rows(300)
    assert sorted(rows[i] for j in jobs for i in j["reference_rows"]) == [0, 2, 3, 4, 299]
    t = _total([X.compare(j) for j in jobs])
    assert t == _zero(words=len(rows) * N, quant_rows=len(rows), reference_quant_rows=5,
                      reference_coords=sum(len(_columns(r)) for r in (0, 2, 3, 4, 299)))


def test_the_sweeps_reach_every_boundary_and_the_endpoint() -> None:
    """`sweep_for` and `einsum_sweep` at the gate's 132 SMs: the dense base, both sides of each launch-configuration boundary
    (pinned in tests/program/test_deepgemm_schedule.py), and 1024."""
    ms = X.sweep_for(7168, 2048, 132, False)
    assert set(X.LINEAR_M) <= set(ms) and ms[-1] == X.MAX_M == 1024
    assert {32, 33, 512, 513, 832, 833} <= set(ms) and 48 not in ms
    assert set(ms) == set(X.S.sweep_ms(X.LINEAR_M, 2048, 7168, num_sms=132, lo=32, hi=1024))
    assert set(X.sweep_for(7168, 2048, 132, True)) >= {16, 17}                  # --batch-invariant sweeps DeepGEMM's boundaries below 32
    es = X.einsum_sweep(132)
    assert set(X.EINSUM_M) <= set(es) and es[-1] == 1024 and {16, 17, 448, 449, 768, 769} <= set(es)


@pytest.mark.parametrize("branch", ["deepgemm", "flashinfer"])
def test_a_flipped_word_is_found_and_located(tmp_path, branch: str) -> None:
    res = [X.compare(j) for j in _linear_jobs(tmp_path, branch, flip_word=True)]
    assert _total(res)["mismatch"] == 1
    assert {k: v for k, v in next(r["first_mismatch"] for r in res if r["first_mismatch"]).items() if k in ("row", "column")} == \
        {"row": 2, "column": 150}


def test_a_quant_byte_that_is_not_the_twins_is_found_though_the_gemm_over_it_agrees(tmp_path) -> None:
    (res,) = [X.compare(j) for j in _linear_jobs(tmp_path, "deepgemm", flip_quant=True)]
    assert (res["quant_mismatch"], res["mismatch"], res["first_mismatch"]) == (1, 0, {"row": 0, "quant": True})


def test_a_finite_word_on_a_nonfinite_lane_is_unresolved_and_a_wrong_nan_is_a_mismatch(tmp_path) -> None:
    assert 10 not in _columns(3) and 157 in _columns(3)

    def finite(out):
        out[3, 10] = 0x3F80

    (res,) = [X.compare(j) for j in _linear_jobs(tmp_path / "a", "flashinfer", edit=finite)]
    assert (res["unresolved"], res["nonfinite_words"], res["mismatch"], res["nonfinite_mismatch"]) == (1, N - 1, 0, 0)

    def payload(out):
        out[3, 157] ^= 1

    (res,) = [X.compare(j) for j in _linear_jobs(tmp_path / "b", "flashinfer", edit=payload)]
    assert (res["unresolved"], res["mismatch"], res["nonfinite_mismatch"]) == (0, 0, 1)


# --- weights ---------------------------------------------------------------------------------------------------------------------

def _weights_jobs(tmp_path, fmt: str, *, edit=None) -> tuple[list[dict], list[list[int]]]:
    """A [300, 256] e4m3 weight in the checkpoint's format and its processing by the twins: e8m0 bytes upcast, or GLM-5.3's f32
    scales with `BAND_SCALES` planted by the GPU script's own `plant_band` and requantised to ue8m0."""
    rng = np.random.default_rng(3)
    w = _e4m3(rng, NW, K)
    band: list[list[int]] = []
    if fmt == "e8m0":
        raw_ws = rng.integers(X.E8M0_LO, X.E8M0_HI + 1, (X.cdiv(NW, GQ), K // GQ)).astype(np.uint8)
        pw, pws = w.copy(), T.upcast_e8m0(raw_ws)
    else:
        tw = torch.from_numpy(w).view(torch.float8_e4m3fn)
        tws = torch.from_numpy(_scales(rng, X.cdiv(NW, GQ), K // GQ).view(np.float32))
        band = X.plant_band(tw, tws)
        w, raw_ws = tw.view(torch.uint8).numpy(), tws.numpy().view(np.uint32)
        pw, pws, _ = T.requant_weight_ue8m0(w, raw_ws)
    if edit:
        edit(pw, pws)
    paths = X.B._save(tmp_path / fmt, w=pw, ws=pws, raw_w=w, raw_ws=raw_ws)
    return [{"phase": "weights", "model": "m", "case": "c", "K": K, "N": NW, "fmt": fmt, "paths": paths, "block_rows": [b0, b1]}
            for b0, b1 in X._block_chunks(X.cdiv(NW, GQ))], band


@pytest.mark.parametrize("block_rows", [8, 1])
def test_e8m0_weights_are_the_bytes_and_the_upcast_scales(tmp_path, monkeypatch, block_rows: int) -> None:
    monkeypatch.setattr(X, "JOB_BLOCK_ROWS", block_rows)
    jobs, _ = _weights_jobs(tmp_path, "e8m0")
    assert len(jobs) == (1 if block_rows == 8 else 3)
    assert _total([X.compare(j) for j in jobs]) == _zero(weight_blocks=6)

    def flip(pw, pws):
        pw[130, 200] ^= 1
        pws[2, 0] ^= 1 << 23

    res = [X.compare(j) for j in _weights_jobs(tmp_path / "flip", "e8m0", edit=flip)[0]]
    assert _total(res)["weight_mismatch"] == 2 and next(r["first_mismatch"] for r in res if r["first_mismatch"]) == {"block": [1, 1]}


@pytest.mark.parametrize("block_rows", [8, 1])
def test_f32_weights_requantised_and_the_band_counted_not_claimed(tmp_path, monkeypatch, block_rows: int) -> None:
    """`plant_band` puts the five `BAND_SCALES` on the first blocks, each over a 448: the four in `requant_band` are counted (here
    the twin's own words, so they agree), the fifth and the last block are claimed."""
    monkeypatch.setattr(X, "JOB_BLOCK_ROWS", block_rows)
    jobs, band = _weights_jobs(tmp_path, "f32")
    assert band == [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]]
    assert _total([X.compare(j) for j in jobs]) == _zero(weight_blocks=2, band_blocks=4, band_agree=4)

    def in_band(pw, pws):
        pw[5, 5] ^= 1

    assert _total([X.compare(j) for j in _weights_jobs(tmp_path / "a", "f32", edit=in_band)[0]]) == \
        _zero(weight_blocks=2, band_blocks=4, band_agree=3)

    def resolved(pw, pws):
        pw[299, 255] ^= 1

    res = [X.compare(j) for j in _weights_jobs(tmp_path / "b", "f32", edit=resolved)[0]]
    assert _total(res)["weight_mismatch"] == 1 and next(r["first_mismatch"] for r in res if r["first_mismatch"]) == {"block": [2, 1]}


def test_plant_band_puts_each_scale_over_a_448() -> None:
    w, ws = torch.zeros(NW, K, dtype=torch.uint8).view(torch.float8_e4m3fn), torch.ones(3, 2)
    blocks = X.plant_band(w, ws)
    assert [int(ws.view(torch.int32)[bn, bk]) for bn, bk in blocks] == list(X.BAND_SCALES)
    assert int(w.view(torch.uint8).sum()) == 0x7E * 5 and all(int(w.view(torch.uint8)[bn * GQ, bk * GQ]) == 0x7E for bn, bk in blocks)
    assert int(ws.view(torch.int32)[2, 1]) == 0x3F800000


# --- einsum ----------------------------------------------------------------------------------------------------------------------

EH, EK, EN, EM = 2, 256, 192, 5


def _einsum_jobs(tmp_path, *, flip: bool = False) -> list[dict]:
    """One `wo_a`-like launch at H = 2, K = 256, N = 192 (a partial weight block per group), its words the grouped twin's."""
    rng = np.random.default_rng(4)
    nb = X.cdiv(EN, GQ)
    xq, sx = _e4m3(rng, EM, EH, EK), T.upcast_e8m0(rng.integers(118, 126, (EM, EH, EK // GQ)))
    w, ws = _e4m3(rng, EH, EN, EK), T.upcast_e8m0(rng.integers(X.E8M0_LO, X.E8M0_HI + 1, (EH, nb, EK // GQ)))
    z = np.stack([T.scaled_mm_fp8_block_grouped_row(xq[t], sx[t], w, ws)[0] for t in range(EM)])
    if flip:
        z[3, 1, 100] ^= 1
    job = {"phase": "einsum", "model": "dsv4", "case": "wo_a", "H": EH, "K": EK, "N": EN, "M": EM, "row_map": X.sample_rows(EM),
           "paths": X.B._save(tmp_path / "einsum", xq=xq, sx=sx, z=z, w=w.reshape(EH * EN, EK), ws=ws.reshape(EH * nb, EK // GQ))}
    return X.row_jobs(job, EH * EN * EK // 32)


def test_einsum_jobs_take_every_groups_words(tmp_path) -> None:
    t = _total([X.compare(j) for j in _einsum_jobs(tmp_path)])
    assert t == _zero(words=EM * EH * EN, reference_coords=sum(len(_columns(r, h)) for r in (0, 2, 3, 4) for h in range(EH)))
    res = [X.compare(j) for j in _einsum_jobs(tmp_path / "flip", flip=True)]
    assert _total(res)["mismatch"] == 1
    fm = next(r["first_mismatch"] for r in res if r["first_mismatch"])
    assert (fm["row"], fm["group"], fm["column"]) == (3, 1, 100)


def test_a_job_that_raises_is_an_error_not_a_crash(tmp_path) -> None:
    (job,) = _einsum_jobs(tmp_path)
    res = X.compare({**job, "H": 3})
    assert res["phase"] == "einsum" and res["M"] == EM and "error" in res


# --- taps, branches, mufu --------------------------------------------------------------------------------------------------------

def test_taps_record_every_call_and_restore_the_module(monkeypatch) -> None:
    mod = types.ModuleType("fake_flashinfer")
    mod.f = lambda x, **kw: (x + 1, x * 2)
    mod.g = lambda x: x
    monkeypatch.setitem(sys.modules, "fake_flashinfer", mod)
    f = mod.f
    with X.Taps("fake_flashinfer", ("f",)) as tp:
        a = torch.ones(3)
        out = mod.f(a, k=1)
        mod.g(a)
        out[0].add_(5)
    assert mod.f is f and len(tp.calls) == 1 and tp.calls[0]["op"] == "f" and tp.calls[0]["kw"] == {"k": 1}
    assert tp.calls[0]["args"][0] is a and torch.equal(tp.calls[0]["out"][0], torch.full((3,), 2.0))      # cloned when written


def test_branch_of_names_the_two_branches_only() -> None:
    assert X.branch_of([{"op": "per_token_group_quant_fp8"}, {"op": "fp8_gemm_nt"}]) == "deepgemm"
    assert X.branch_of([{"op": "flashinfer_fp8_blockscale_gemm"}]) == "flashinfer"
    assert X.branch_of([{"op": "fp8_gemm_nt"}]) == "other:fp8_gemm_nt" and X.branch_of([]) == "other:"


def _layer() -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(N, K, dtype=torch.uint8).view(torch.float8_e4m3fn), requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(torch.ones(X.cdiv(N, GQ), K // GQ), requires_grad=False)
    return layer


def _deepgemm_calls(x, layer, out, **change):
    q = torch.zeros(x.shape, dtype=torch.float8_e4m3fn)
    s = torch.ones(x.shape[0], K // GQ).t().contiguous().t()                    # column-major, as the quant writes it
    quant = {"op": "per_token_group_quant_fp8", "args": (x,), "kw": change.get("quant_kw", QUANT_KW), "out": X._clone((q, s))}
    gemm = {"op": "fp8_gemm_nt", "args": ((q, s), (change.get("w", layer.weight), layer.weight_scale_inv), out),
            "kw": change.get("gemm_kw", {"is_deep_gemm_e8m0_used": True}), "out": None}
    return [quant, gemm]


def test_equal_is_bit_for_bit() -> None:
    """FlashInfer's row 3 is all NaN: a run equals itself bit for bit though `torch.equal` says it does not."""
    bf16 = lambda *w: torch.from_numpy(np.array(w, np.uint16).view(np.int16)).view(torch.bfloat16)
    nan = bf16(0x7FC0, 0x3F80, 0xFFC1)
    assert not torch.equal(nan, nan.clone()) and X._equal(nan, nan.clone())
    other = bf16(0x7FC1, 0x3F80, 0xFFC1)
    assert not X._equal(nan, other) and not X._equal(nan, nan.view(torch.int16)) and not X._equal(nan, nan[:2])
    q = torch.tensor([0x7F, 0x80, 0x00], dtype=torch.uint8).view(torch.float8_e4m3fn)
    assert X._equal(q, q.clone()) and not X._equal(q, torch.zeros(3, dtype=torch.uint8).view(torch.float8_e4m3fn))
    s = torch.ones(4, 2).t().contiguous().t()
    assert X._equal(s, s.contiguous()) and X._equal(s, torch.ones(4, 2))


def test_the_branch_operands_are_the_layers_and_the_inputs() -> None:
    layer, x, out = _layer(), torch.zeros(4, K, dtype=torch.bfloat16), torch.zeros(4, N, dtype=torch.bfloat16)
    assert X.branch_operands_ok(_deepgemm_calls(x, layer, out), "deepgemm", x, layer, out)
    for change in ({"quant_kw": {**QUANT_KW, "use_ue8m0": False}}, {"gemm_kw": {}}, {"w": layer.weight.clone()}):
        assert not X.branch_operands_ok(_deepgemm_calls(x, layer, out, **change), "deepgemm", x, layer, out), change
    assert not X.branch_operands_ok(_deepgemm_calls(x, layer, out), "deepgemm", x, layer, torch.ones(4, N, dtype=torch.bfloat16))
    assert not X.branch_operands_ok(_deepgemm_calls(x, layer, out), "deepgemm", x.clone(), layer, out)
    fi = lambda **kw: [{"op": "flashinfer_fp8_blockscale_gemm", "args": (), "out": out.clone(),
                        "kw": {"input": x, "weight": layer.weight, "weight_scale": layer.weight_scale_inv, "out_dtype": torch.bfloat16, **kw}}]
    assert X.branch_operands_ok(fi(), "flashinfer", x, layer, out)
    nan = torch.full((4, N), float("nan"), dtype=torch.bfloat16)
    assert X.branch_operands_ok([{**fi()[0], "out": nan.clone()}], "flashinfer", x, layer, nan)
    assert not X.branch_operands_ok(fi(out_dtype=torch.float16), "flashinfer", x, layer, out)
    assert not X.branch_operands_ok(fi(weight_scale=layer.weight_scale_inv.clone()), "flashinfer", x, layer, out)
    assert not X.branch_operands_ok(fi(), "other:fp8_gemm_nt", x, layer, out)


MUFU = {"library": {"rcp": "r0"}, "builds": {"sm_90": {"rcp_sha256": "r0", "verify": {"rcp_mismatches": 0}}}}


def test_mufu_needs_core_rcp_table_on_every_build() -> None:
    assert X.mufu_rcp_ok(MUFU)
    assert not X.mufu_rcp_ok({**MUFU, "builds": {}})
    assert not X.mufu_rcp_ok({**MUFU, "builds": {"sm_90": {"rcp_sha256": "r1", "verify": {"rcp_mismatches": 0}}}})
    assert not X.mufu_rcp_ok({**MUFU, "builds": {"sm_90": {"rcp_sha256": "r0", "verify": {"rcp_mismatches": 3}}}})


# --- jit and bind ----------------------------------------------------------------------------------------------------------------

#: `sm90_fp8_gemm_1d2d_impl`'s 20 template arguments in `generate_impl`'s order (sm90_fp8_gemm_1d2d.hpp:34-68 at 8b1392b): major_sfb,
#: m, n, k, groups, block m/n/k, swizzles a/b/cd, stages, TMA and math threads, cluster, multicast, SMs, gemm type, cd type, epilogue
GEMM = ["cute::UMMA::Major::K", "0", "0", "7168", "1", "64", "128", "128", "128", "128", "128", "5", "128", "256", "1", "false", "132",
        "GemmType::Normal", "cutlass::bfloat16_t", "epilogue::transform::EpilogueIdentity"]


def _kernel_cu(impl: str, args: list[str]) -> str:
    """A `kernel.cu` as DeepGEMM's `generate_impl` writes it."""
    body = ",\n        ".join(args)
    return (f"\n#include <deep_gemm/impls/x.cuh>\n\nusing namespace deep_gemm;\n\nstatic void __instantiate_kernel() {{\n"
            f"    auto ptr = reinterpret_cast<void*>(&{impl}<\n        {body}\n    >);\n}};\n")


def _jit_cache(root, kernels: dict[str, tuple[str, list[str]]]):
    for name, (impl, args) in kernels.items():
        (root / "cache" / name).mkdir(parents=True)
        (root / "cache" / name / "kernel.cu").write_text(_kernel_cu(impl, args) if impl else "// nothing instantiated\n")
    return root


def _with(i: int, v: str) -> list[str]:
    return GEMM[:i] + [v] + GEMM[i + 1:]


GOOD = {"kernel.a": (X.GEMM_IMPL, GEMM), "kernel.b": (X.GEMM_IMPL, _with(17, "GemmType::Batched")),
        "kernel.c": (X.LAYOUT_IMPL, ["128", "64", "56"])}
#: the schedule's predictions matching `GOOD` (compiled n where the generated kernel left it runtime)
PRED = [_with(2, "2048"), _with(17, "GemmType::Batched")]


def test_the_jit_scan_takes_only_the_predicted_1d2d_gemms_and_the_scale_transpose(tmp_path) -> None:
    got = X.jit(_jit_cache(tmp_path / "ok", GOOD), PRED)
    assert got["ok"] and (got["files"], got["gemms"], got["layouts"]) == (3, 2, 1) and not got["unlaunched"]
    assert got["gemm_types"] == ["GemmType::Batched", "GemmType::Normal"] and got["kernels"]["cache/kernel.a/kernel.cu"]["args"] == GEMM
    bad = {"1d1d": ("sm90_fp8_gemm_1d1d_impl", GEMM), "block_k": (X.GEMM_IMPL, _with(7, "64")), "f32_out": (X.GEMM_IMPL, _with(18, "float")),
           "epilogue": (X.GEMM_IMPL, _with(19, "epilogue::transform::EpilogueHeadSplits")),
           "grouped": (X.GEMM_IMPL, _with(17, "GemmType::MGroupedContiguous")), "short": (X.GEMM_IMPL, GEMM[:19]),
           "pack": ("transpose_and_pack_fp32_into_ue8m0", ["128", "64", "56", "1", "false"]), "none": (None, []),
           "unpredicted_stages": (X.GEMM_IMPL, _with(11, "4"))}
    for name, kernel in bad.items():
        got = X.jit(_jit_cache(tmp_path / name, {**GOOD, "kernel.x": kernel}), PRED)
        assert got["ok"] is False and got["kernels"]["cache/kernel.x/kernel.cu"]["ok"] is False, name
    assert X.jit(_jit_cache(tmp_path / "layouts", {"kernel.c": GOOD["kernel.c"]}), [])["ok"] is False
    assert X.jit(tmp_path / "nothing", [])["ok"] is False


def test_a_predicted_configuration_the_run_never_generated_fails_the_scan(tmp_path) -> None:
    extra = _with(5, "256")
    got = X.jit(_jit_cache(tmp_path / "missing", GOOD), PRED + [extra])
    assert got["ok"] is False and got["unlaunched"] == [extra]


def test_args_match_is_exact_but_for_the_compiled_dims() -> None:
    """m, n and k (indices 1-3) may be runtime ("0") on either side; everything else is exact."""
    pred = _with(2, "2048")
    assert X._args_match(GEMM, pred)                                            # generated n runtime, prediction compiled
    assert X._args_match(pred, pred) and X._args_match(_with(1, "64"), pred)    # a compiled m against the runtime prediction
    assert not X._args_match(_with(2, "1024"), pred)                            # a different compiled n
    assert not X._args_match(_with(11, "4"), pred) and not X._args_match(GEMM[:19], pred)
    assert not X._args_match(_with(17, "GemmType::Batched"), pred)


def test_the_programs_bind_at_every_shape() -> None:
    got = X.bind_check()
    assert got["ok"] and got["bindings"]["DeepGemmFp8Linear_v1{K=7168,N=2048}"] and got["bindings"]["ScaledMmFp8BlockGrouped_v1{H=2,K=4096,N=1024,G=128}"]
    shapes = {(K_, N_) for _, _, K_, N_ in X.linear_cases()}
    assert len(got["bindings"]) == 1 + 2 * len(shapes) + 2 * len({K_ for K_, _ in shapes})


def test_the_linears_are_both_models_tp8_fp8_linears() -> None:
    assert len(X.linear_cases()) == len(X.DSV4_LINEARS) + 9 == 15
    assert all(K % GQ == 0 for _, _, K, _ in X.linear_cases())
    assert dict((n, (K, N_)) for n, K, N_ in X.DSV4_LINEARS)["fused_wqa_wkv"] == (7168, 2048)


# --- verdict ---------------------------------------------------------------------------------------------------------------------

WORDS = _zero(words=10, reference_coords=3, reference_quant_rows=1)
BLOCKS = _zero(weight_blocks=4)


def _doc(batch_invariant: bool = False) -> dict:
    lin = {"phase": "linear", "model": "dsv4", "case": "wq_b", "deterministic": True, "operands_ok": True, "branch_ok": True}
    branches = ("deepgemm", "deepgemm") if batch_invariant else ("flashinfer", "deepgemm")
    weights = lambda case: {"phase": "weights", "model": "dsv4", "case": case, "deterministic": True, "operands_ok": True}
    doc = {"schema": X.SCHEMA, "batch_invariant": batch_invariant, "gate": {"ok": True}, "errors": {}, "selection": {"ok": True},
           "mufu": {"rcp_ok": True}, "schedule": {"ok": True},
           "jit": {"ok": True, "gemm_types": ["GemmType::Batched", "GemmType::Normal"]}, "bind": {"ok": True},
           "launches": [weights("wq_b"), {**lin, "M": 1, "branch": branches[0]}, {**lin, "M": 32, "branch": branches[1]}, weights("wo_a"),
                        {"phase": "einsum", "model": "dsv4", "case": "wo_a", "M": 1, "deterministic": True, "operands_ok": True}],
           "comparisons": [{"phase": "weights", "model": "dsv4", "case": "wq_b", **BLOCKS},
                           {"phase": "linear", "model": "dsv4", "case": "wq_b", "M": 1, **WORDS,
                            **({"quant_rows": 1} if batch_invariant else {})},
                           {"phase": "linear", "model": "dsv4", "case": "wq_b", "M": 32, **WORDS, "quant_rows": 2},
                           {"phase": "weights", "model": "dsv4", "case": "wo_a", **BLOCKS},
                           {"phase": "einsum", "model": "dsv4", "case": "wo_a", "M": 1, **WORDS, "reference_quant_rows": 0}]}
    doc["totals"] = {k: sum(r[k] for r in doc["comparisons"]) for k in X.TOTALS}
    doc["cases"] = X.cases(doc)
    return doc


@pytest.mark.parametrize("path,value", [
    (("gate", "ok"), False), (("errors", "linear/glm53/q_a_proj"), "boom"), (("selection", "ok"), False), (("mufu", "rcp_ok"), None),
    (("mufu", "rcp_ok"), False), (("schedule", "ok"), False), (("jit", "ok"), False), (("jit", "gemm_types"), ["GemmType::Normal"]),
    (("bind", "ok"), False),
    (("totals", "words"), 0), (("totals", "mismatch"), 1), (("totals", "unresolved"), 1), (("totals", "quant_mismatch"), 1),
    (("totals", "quant_rows"), 0), (("totals", "reference_coords"), 0), (("totals", "reference_quant_rows"), 0),
    (("totals", "twin_mismatch"), 1), (("totals", "nonfinite_mismatch"), 1), (("totals", "weight_blocks"), 0),
    (("totals", "weight_mismatch"), 1), (("launches", 0, "operands_ok"), False), (("launches", 1, "branch"), "deepgemm"),
    (("launches", 2, "branch_ok"), False), (("launches", 2, "operands_ok"), None), (("launches", 4, "deterministic"), False),
    (("comparisons", 2, "mismatch"), 1), (("batch_invariant",), True)])
def test_the_verdict_needs_every_check(path: tuple, value) -> None:
    assert X._verdict(_doc()) and X._verdict(_doc(batch_invariant=True))
    doc = copy.deepcopy(_doc())
    node = doc
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    doc["cases"] = X.cases(doc)
    assert not X._verdict(doc)


def test_a_run_without_every_phase_is_not_a_go() -> None:
    for phase in ("einsum", "weights", "linear"):
        doc = _doc()
        doc["launches"] = [x for x in doc["launches"] if x["phase"] != phase]
        doc["comparisons"] = [r for r in doc["comparisons"] if r["phase"] != phase]
        doc["totals"] = {k: sum(r[k] for r in doc["comparisons"]) for k in X.TOTALS}
        doc["cases"] = X.cases(doc)
        assert not X._verdict(doc), phase


def test_each_case_has_its_own_verdict() -> None:
    doc = _doc()
    assert {k: c["ok"] for k, c in doc["cases"].items()} == {"weights/dsv4/wq_b": True, "linear/dsv4/wq_b/M1": True,
                                                             "linear/dsv4/wq_b/M32": True, "weights/dsv4/wo_a": True,
                                                             "einsum/dsv4/wo_a/M1": True}
    doc["launches"].insert(2, {**doc["launches"][1], "M": 16})
    doc["comparisons"].append({**doc["comparisons"][1], "M": 16})
    doc["errors"] = {"compare/linear/dsv4/wq_b/M1": "boom", "linear/glm53/q_a_proj": "boom"}
    doc["comparisons"][3]["weight_mismatch"] = 1
    doc["comparisons"][2]["quant_rows"] = 0                                     # DeepGEMM's branch with no quant row compared
    got = {k: c["ok"] for k, c in X.cases(doc).items()}
    assert got == {"weights/dsv4/wq_b": True, "linear/dsv4/wq_b/M1": False, "linear/dsv4/wq_b/M16": True, "linear/dsv4/wq_b/M32": False,
                   "weights/dsv4/wo_a": False, "einsum/dsv4/wo_a/M1": True, "linear/glm53/q_a_proj": False}


# --- main ------------------------------------------------------------------------------------------------------------------------

ENV = ("VLLM_USE_DEEP_GEMM", "VLLM_USE_DEEP_GEMM_E8M0", "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER", "VLLM_ENABLE_V1_MULTIPROCESSING",
       "VLLM_BATCH_INVARIANT", "TRITON_CACHE_DIR", "DG_JIT_CACHE_DIR")


def _stub_gpu(tmp_path, monkeypatch, *, gate_ok: bool = True, flip: bool = False) -> dict:
    """vLLM's config module and every GPU piece stood in for: the launches are the jobs above, the JIT scan and MUFU probe
    canned; returns what the stubs saw."""
    import contextlib

    vc = types.ModuleType("vllm.config")
    vc.CompilationConfig = vc.VllmConfig = lambda **kw: kw
    vc.set_current_vllm_config = lambda cfg: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", vc)
    for k in ENV:
        monkeypatch.setenv(k, "")
    seen: dict = {"mufu": [], "jit": [], "linear": []}
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": gate_ok, "torch_capability": "9.0", "sms": 132})
    monkeypatch.setattr(X.C, "mufu", lambda work, arch: (seen["mufu"].append(arch), copy.deepcopy(MUFU))[1])
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    monkeypatch.setattr(X, "selection", lambda cfg: {"ok": True})
    monkeypatch.setattr(X, "model_config", lambda cfg, model_type: contextlib.nullcontext())
    monkeypatch.setattr(X, "linear_cases", lambda: [("dsv4", "small", K, N)])
    canned = X.jit(_jit_cache(tmp_path / "jit", GOOD), PRED)
    monkeypatch.setattr(X, "jit", lambda cache, predicted: (seen["jit"].append(str(cache)), copy.deepcopy(canned))[1])

    def linear(work, model, name, K_, N_, Ms, seed, batch_invariant):
        seen["linear"].append((model, name, K_, N_, Ms, batch_invariant))
        launches = [{"phase": "weights", "model": model, "case": name, "deterministic": True, "operands_ok": True}]
        jobs = [{**j, "model": model, "case": name} for j in _weights_jobs(tmp_path / "w", "e8m0")[0]]
        for M in Ms:
            branch = "deepgemm" if batch_invariant or M >= 32 else "flashinfer"
            launches.append({"phase": "linear", "model": model, "case": name, "K": K_, "N": N_, "M": M, "branch": branch, "branch_ok": True,
                             "deterministic": True, "operands_ok": True})
            jobs += [{**j, "model": model, "case": name} for j in _linear_jobs(tmp_path / "l", branch, M=M, flip_word=flip and M >= 32)]
        return launches, jobs

    def einsum(work, Ms, seed):
        launches = [{"phase": "weights", "model": "dsv4", "case": "wo_a", "deterministic": True, "operands_ok": True},
                    {"phase": "einsum", "model": "dsv4", "case": "wo_a", "H": EH, "K": EK, "N": EN, "M": EM,
                     "calls": ["fused_inv_rope_fp8_quant", "fp8_einsum"], "deterministic": True, "operands_ok": True}]
        jobs = [{**j, "model": "dsv4", "case": "wo_a"} for j in _weights_jobs(tmp_path / "we", "e8m0")[0]] + _einsum_jobs(tmp_path / "e")
        return launches, jobs

    monkeypatch.setattr(X, "linear_launches", linear)
    monkeypatch.setattr(X, "einsum_launches", einsum)
    return seen


def test_main_end_to_end_on_stubbed_launches(tmp_path, monkeypatch, capsys) -> None:
    """`main` with the GPU's pieces stood in for and vLLM's config module stubbed: the comparison jobs through the spawned worker
    pool, the per-case verdicts, the sealed record, the printed line and the exit status, so a slip in the orchestration shows here
    and not in the H200's first hour."""
    import json

    seen = _stub_gpu(tmp_path, monkeypatch)
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1", "--quick"]) == 0
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is True and not doc["errors"]
    assert {k: c["ok"] for k, c in doc["cases"].items()} == {"weights/dsv4/small": True, "linear/dsv4/small/M1": True,
                                                             "linear/dsv4/small/M32": True, "weights/dsv4/wo_a": True,
                                                             "einsum/dsv4/wo_a/M5": True}
    assert doc["env"]["VLLM_USE_DEEP_GEMM"] == "1" and doc["env"]["VLLM_BATCH_INVARIANT"] == "0"
    assert seen["mufu"] == ["sm_90"] and seen["jit"] == [doc["env"]["DG_JIT_CACHE_DIR"]] and seen["linear"][0][4] == X.QUICK_LINEAR_M
    assert doc["totals"]["quant_rows"] == 32 and doc["totals"]["weight_blocks"] == 12 and doc["bind"]["ok"]
    assert doc["schedule"]["ok"] and doc["schedule"]["num_sms"] == 132
    assert set(doc["schedule"]["configs"]) == {"linear/dsv4/small/M32", "einsum/dsv4/wo_a/M5"}
    assert all(len(p) == 20 and p[7] == "128" for p in doc["schedule"]["predicted"])
    printed = capsys.readouterr().out
    assert "DEEPGEMM GO" in printed and "no-go cases=[]" in printed and "case linear/dsv4/small/M32" in printed


def test_main_names_the_case_a_flipped_word_fails(tmp_path, monkeypatch, capsys) -> None:
    import json

    _stub_gpu(tmp_path, monkeypatch, flip=True)
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1", "--quick", "--only", "linear,einsum"]) == 1
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["ok"] is False and doc["totals"]["mismatch"] == 1 and doc["mufu"] == {"rcp_ok": None}
    assert [k for k, c in doc["cases"].items() if not c["ok"]] == ["linear/dsv4/small/M32"]
    assert "DEEPGEMM NO-GO" in capsys.readouterr().out


def test_main_stops_at_a_failed_gate(tmp_path, monkeypatch, capsys) -> None:
    import json

    _stub_gpu(tmp_path, monkeypatch, gate_ok=False)
    with pytest.raises(SystemExit) as e:
        X.main(["--out", str(tmp_path / "out"), "--scratch", str(tmp_path), "--workers", "1"])
    doc = json.loads((tmp_path / "out" / X.RECORD).read_text())
    assert e.value.code == 3 and doc["ok"] is False and doc["digest"] == digest_of(doc) and "launches" not in doc
    assert "DEEPGEMM GATE-FAIL" in capsys.readouterr().out


def test_the_record_is_a_registered_property() -> None:
    from verity_vllm.properties.record import seal

    doc = seal(_doc(), X._verdict(_doc()))
    assert REGISTRY["deepgemm_capture"].schema == X.SCHEMA and doc["ok"] and doc["digest"] == digest_of(doc)
