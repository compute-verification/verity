"""The GLM-5.3 block-FP8 capture without a GPU: its comparison jobs over operands and words made here (the kernels' words stood in
for by the twins), each link's flipped word or byte found and located, the operand probes refuting every competing model of
`HopperE4m3MmaV2Dot32_v1` and of `MoeSum_v1`, the launch-assignment check, the PTX scan, the bindings, and the verdict (the GPU run
is `tests/properties/glm53_block_fp8_capture_gpu.py --out ...` on an H200)."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import copy

import numpy as np
import pytest
import torch

from tests.properties import glm53_block_fp8_capture_gpu as X
from verity.ml.definitions.numerics import deepgemm_twins as DGT
from verity.ml.definitions.numerics import fp8_moe_rows as MR
from verity.ml.definitions.numerics.act_rows import silu_mul_expf_rows
from verity.ml.definitions.numerics.rows import moe_sum
from verity_vllm.properties.record import REGISTRY, digest_of

K, N, GQ = 256, 200, X.GROUP
WG, MV = "HopperE4m3QgmmaDot32_v1", "HopperE4m3MmaV2Dot32_v1"
USED = np.array([3, 17, 40, 41, 255])


def _bf16_rows(rng: np.random.Generator, M: int, width: int = K) -> np.ndarray:
    x = rng.standard_normal((M, width)).astype(np.float32) * np.exp2(rng.uniform(-3, 3, (M, 1))).astype(np.float32)
    x[1] = 0
    x[3] *= np.float32(2.0 ** -130)
    return (x.view(np.uint32) >> 16).astype(np.uint16)


def _e4m3(rng: np.random.Generator, *shape: int) -> np.ndarray:
    b = rng.integers(0, 256, shape).astype(np.uint8)
    return np.where((b & 0x7F) == 0x7F, b ^ 1, b).astype(np.uint8)


def _scales(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return np.exp2(rng.uniform(-12, -6, shape)).astype(np.float32).view(np.uint32)


def _total(results: list[dict]) -> dict:
    assert not [r for r in results if "error" in r], results
    return {k: sum(r[k] for r in results) for k in X.TOTALS}


def _probe(M: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    gen = torch.Generator().manual_seed(4)
    x, w = X.mmav2_probe(gen, M, cols, K, device="cpu")
    return x.view(torch.int16).numpy().view(np.uint16), w.view(torch.uint8).numpy()


def _moe_job(tmp_path, kind: str, *, M: int = 3, Nc: int = N, KB: int = 32, swap: bool = False, dot: str = WG,
             words_dot: str | None = None, flip: bool = False, flip_act: bool = False, flip_quant: bool = False, probe: bool = False,
             case: str = "small", phase: str = "moe") -> list[dict]:
    """One expert launch as `moe_launches` saves it, M tokens x 8 slots over the 5 `USED` experts, its words made with `words_dot`'s
    step (default `dot`): w13 (A, A_scale the ue8m0 quant of x; with `probe`, `mmav2_probe`'s x and w) or w2 (A, A_scale the ue8m0
    quant of `SiluMul_v2` of [gate || up] rows, one slot each, the routed weight)."""
    rng = np.random.default_rng(2)
    ids = np.stack([rng.permutation(USED)[:5].tolist() + rng.permutation(USED)[:3].tolist() for _ in range(M)]).reshape(-1)
    umap = np.full(X.NUM_EXPERTS, -1)
    umap[USED] = np.arange(USED.size)
    tw = (rng.uniform(0.05, 1, M * X.TOP_K).astype(np.float32)).view(np.uint32)
    top_k, mul = (X.TOP_K, False) if kind == "w13" else (1, True)
    if kind == "w13":
        if probe:
            x, w = _probe(M, USED.size * Nc)
            B = w.reshape(USED.size, Nc, K)
        else:
            x, B = _bf16_rows(rng, max(M, 4))[:M], _e4m3(rng, USED.size, Nc, K)
        A, As = DGT.ue8m0_group_quant_rows(x, G=GQ)
        extra = {"x": x}
    else:
        rows = M * X.TOP_K
        gu = np.concatenate([_bf16_rows(rng, rows, K), _bf16_rows(rng, rows, K)], axis=1)
        act = silu_mul_expf_rows(gu)["out"].copy()
        if flip_act:
            act[19, 30] ^= 1
        A, As = DGT.ue8m0_group_quant_rows(act, G=GQ)
        B = _e4m3(rng, USED.size, Nc, K)
        extra = {"gu": gu, "act": act}
    if flip_quant:
        A = A.copy()
        A[1, 9] ^= 1
    Bs = _scales(rng, USED.size, X.cdiv(Nc, GQ), K // GQ)
    C = np.stack([MR.moe_expert_fp8_block_row(A[p // top_k], As[p // top_k], B[umap[ids[p]]], Bs[umap[ids[p]]], G=GQ, KB=KB, swap=swap,
                                              g=int(tw[p]) if mul else None, dot=MR.TWIN_DOT_OF[words_dot or dot])[0]
                  for p in range(M * X.TOP_K)])
    if flip:
        C[min(13, M * X.TOP_K - 1), 7] ^= 0x8000
    arrays = {"A": A, "As": As, "B": B, "Bs": Bs, "ids": ids, "umap": umap, "tw": tw, "C": C, **extra}
    job = {"phase": phase, "case": case, "launch": kind, "K": K, "N": Nc, "KB": KB, "SWAP": swap, "dot": dot, "mul": mul, "top_k": top_k,
           "pairs": M * X.TOP_K, "statics_known": True, "alternatives": probe,
           "paths": X._save(tmp_path / f"{phase}_{case}_{kind}", **arrays)}
    return [{**job, "pair_range": [0, M * X.TOP_K], "reference": True, "quant_rows": list(range(M)) if kind == "w13" else []}]


def _sampled_columns(Nc: int = N) -> int:
    return sum(len({0, Nc - 1, (p * 7919) % Nc}) for p in (0, 23))


@pytest.mark.parametrize("kind,KB,swap,dot", [("w13", 32, False, WG), ("w2", 32, False, WG), ("w2", 64, True, WG),
                                             ("w13", 128, False, MV), ("w2", 128, True, MV)])
def test_moe_jobs_take_the_kernel_words_and_each_link(tmp_path, kind: str, KB: int, swap: bool, dot: str) -> None:
    """w13's input quant on every token and w2's activation and its quant on every pair, against what the launch read, with the
    GEMM over the launch's own operands; a flipped output word is found at its pair."""
    t = _total([X.compare(j) for j in _moe_job(tmp_path, kind, KB=KB, swap=swap, dot=dot)])
    assert t["words"] == 24 * N and t["mismatch"] == t["unresolved"] == t["twin_mismatch"] == 0
    assert t["reference_coords"] == _sampled_columns()
    w13 = kind == "w13"
    assert (t["quant_rows"], t["quant_mismatch"], t["reference_quant_rows"]) == ((3, 0, 1) if w13 else (0, 0, 0))
    assert (t["act_rows"], t["act_mismatch"], t["reference_act_rows"]) == ((0, 0, 0) if w13 else (24, 0, 1))
    assert (t["act_quant_rows"], t["act_quant_mismatch"], t["reference_act_quant_rows"]) == ((0, 0, 0) if w13 else (24, 0, 1))
    flipped = [X.compare(j) for j in _moe_job(tmp_path, kind, KB=KB, swap=swap, dot=dot, flip=True)]
    assert _total(flipped)["mismatch"] == 1 and flipped[0]["first_mismatch"]["pair"] == 13


def test_a_quant_byte_that_is_not_the_twins_is_found_though_the_gemm_over_it_agrees(tmp_path) -> None:
    (res,) = [X.compare(j) for j in _moe_job(tmp_path, "w13", flip_quant=True)]
    assert (res["quant_mismatch"], res["mismatch"]) == (1, 0) and res["first_mismatch"] == {"quant_row": 1}
    (res,) = [X.compare(j) for j in _moe_job(tmp_path, "w2", flip_quant=True)]
    assert (res["act_quant_mismatch"], res["act_mismatch"], res["mismatch"]) == (1, 0, 0)
    assert res["first_mismatch"] == {"act_pair": 1, "act": False, "act_quant": True}


def test_an_activation_word_that_is_not_silu_mul_is_found(tmp_path) -> None:
    """`silu_and_mul`'s output with one word flipped, the w2 launch's A its quant: the activation link fails, its quant agrees."""
    (res,) = [X.compare(j) for j in _moe_job(tmp_path, "w2", flip_act=True)]
    assert (res["act_mismatch"], res["act_quant_mismatch"], res["mismatch"]) == (1, 0, 0)
    assert res["first_mismatch"] == {"act_pair": 19, "act": True, "act_quant": False}


@pytest.mark.parametrize("dot,words_dot", [(MV, WG), (WG, MV)])
def test_a_launch_checked_at_the_other_dot_mismatches(tmp_path, dot: str, words_dot: str) -> None:
    """The stock 8x32x128 tile's words (MMAv2) against the wgmma Program, and a wgmma tile's against the MMAv2 one: the words the
    two steps round differently are found, so the run binds each launch's own DOT and not either for both."""
    t = _total([X.compare(j) for j in _moe_job(tmp_path, "w13", KB=128, dot=dot, words_dot=words_dot)])
    assert t["words"] == 24 * N and t["mismatch"] > 0 and t["twin_mismatch"] == 0


def test_a_launch_at_statics_no_program_binds_is_unresolved(tmp_path) -> None:
    (job,) = _moe_job(tmp_path, "w13")
    res = X.compare({**job, "statics_known": False})
    assert res["unresolved"] == res["words"] == 24 * N and (res["quant_rows"], res["quant_mismatch"]) == (3, 0)
    (job,) = _moe_job(tmp_path, "w2")
    res = X.compare({**job, "statics_known": False})
    assert res["unresolved"] == res["words"] == 24 * N and (res["act_rows"], res["act_mismatch"]) == (24, 0)


def test_the_mmav2_probe_quantises_to_its_own_codes() -> None:
    """`mmav2_probe`'s x is its codes times 2^-8, every 128-group's max 448 * 2^-8: the input's ue8m0 quant (the prepare step's) gives
    the scale 2^-8 and the codes back, so the w13 launch multiplies exactly the planted bytes."""
    x, w = _probe(2, 64)
    q, s = DGT.ue8m0_group_quant_rows(x, G=GQ)
    assert (s == 0x3B800000).all()
    assert (q[:, 0::32] == 0x7E).all() and (q[:, 2::32] == 0x7E).all() and (w[:, 0::32] == 0x7E).all() and (w[:, 2::32] == 0xFE).all()
    small = np.delete(np.arange(K), np.r_[0:K:32, 2:K:32])
    assert ((q[:, small] & 0x7F) >> 3 <= 4).all() and ((w[:, small] & 0x7F) >> 3 <= 4).all()
    assert (np.asarray(MR.E4M3_AS_BF16, dtype=np.uint32)[q] << 16).view(np.float32).sum() != 0


@pytest.mark.parametrize("swap", [False, True])
def test_the_mmav2_probe_refutes_every_competing_model(tmp_path, swap: bool) -> None:
    """The decode tiles' w13 launch on the probe's operands, its words the twin's (`mmav2_dot32_e4m3`): every model in
    `mmav2_alternatives` differs from them on some word, and the twin matches."""
    (job,) = _moe_job(tmp_path, "w13", M=1, Nc=64, KB=128, swap=swap, dot=MV, probe=True, phase="mmav2", case="probe")
    (res,) = [X.compare(job)]
    assert res["mismatch"] == res["unresolved"] == res["twin_mismatch"] == 0 and res["words"] == 8 * 64
    assert set(res["refuted"]) == set(X.mmav2_alternatives()) and all(v > 0 for v in res["refuted"].values()), res["refuted"]
    assert X._refuted_ok("mmav2", res["refuted"]) and not X._refuted_ok("mmav2", {**res["refuted"], "width_27": 0})


def test_random_operands_do_not_refute_every_mmav2_model(tmp_path) -> None:
    """Why the probe: on random e4m3 a wider adder rounds as the twin's on most words, so a pass on random operands would not
    have decided `HopperE4m3MmaV2Dot32_v1`'s width."""
    (job,) = _moe_job(tmp_path, "w13", M=1, Nc=64, KB=128, dot=MV, phase="mmav2", case="random")
    res = X.compare({**job, "alternatives": True})
    assert res["mismatch"] == 0 and min(res["refuted"].values()) < res["words"] // 4


def _sum_job(tmp_path, *, flip: bool = False) -> tuple[dict, np.ndarray, list]:
    y, planted = X.moe_sum_probe(9500)
    out = np.stack([moe_sum({"TOPK": X.TOP_K, "H": X.HIDDEN}, [y[t]])["sum"] for t in range(X.SUM_TOKENS)])
    if flip:
        out[5, 4000] ^= 1
    job = {"phase": "sum", "case": "planted", "launch": "sum", "H": X.HIDDEN, "alternatives": True, "columns": planted,
           "rows": [0, X.SUM_TOKENS], "reference": True, "paths": X._save(tmp_path / "sum", y=y, out=out)}
    return job, y, planted


def test_the_planted_sum_refutes_every_competing_sum(tmp_path) -> None:
    """`moe_sum_probe`'s rows, the kernel's words `MoeSum_v1`'s: the twin and the reference agree on every planted column and the
    sampled ones, and every model in `moe_sum_alternatives` differs on some word; a flipped word is found where it is."""
    job, y, planted = _sum_job(tmp_path)
    assert len(planted) == 2 * X.SUM_TOKENS * len(X.SUM_PLANTS) and len(set(planted)) == len(planted)
    rest = np.delete(y, sorted({c for _, c in planted}), axis=-1)
    assert y.shape == (X.SUM_TOKENS, X.TOP_K, X.HIDDEN) and ((rest & 0x7F80) != 0x7F80).all()
    res = X.compare(job)
    assert (res["sum_rows"], res["sum_mismatch"], res["twin_mismatch"]) == (X.SUM_TOKENS, 0, 0)
    sampled = {(t, n) for t in range(X.SUM_TOKENS) for n in (0, X.HIDDEN - 1, (t * 7919) % X.HIDDEN)}
    assert res["reference_sum_coords"] == len(set(planted) | sampled)
    assert set(res["refuted"]) == set(X.moe_sum_alternatives()) and all(v > 0 for v in res["refuted"].values()), res["refuted"]
    assert X._refuted_ok("sum", res["refuted"]) and not X._refuted_ok("sum", {k: v for k, v in res["refuted"].items() if k != "ftz"})
    job, _, _ = _sum_job(tmp_path, flip=True)
    res = X.compare(job)
    assert res["sum_mismatch"] == 1 and res["first_mismatch"]["sum_token"] == 5 and res["first_mismatch"]["column"] == 4000


def test_each_plant_refutes_the_sum_it_was_planted_against() -> None:
    """The plant each competing sum was designed to fail on: its words there differ from `MoeSum_v1`'s."""
    twin = X.moe_sum_alternatives()
    names = [n for n, _ in X.SUM_PLANTS]
    y = np.stack([np.array(w, dtype=np.uint16) for _, w in X.SUM_PLANTS], axis=-1)[None]
    want = moe_sum({"TOPK": X.TOP_K, "H": len(names)}, [y[0]])["sum"]
    designed = {"descending": "cancel_ascending", "pairwise_tree": "cancel_pairs", "first_slot_start": "neg_zero",
                "bf16_accumulator": "bf16_accumulator", "ftz": "subnormal", "nan_payload_kept": "nan_quiet", "truncated": "tie_odd"}
    assert set(designed) == set(twin)
    for alt, plant in designed.items():
        assert twin[alt](y)[0][names.index(plant)] != want[names.index(plant)], (alt, plant)


def test_a_chain_sum_job_without_alternatives_compares_its_rows(tmp_path) -> None:
    rng = np.random.default_rng(5)
    y = _bf16_rows(rng, 3 * X.TOP_K, X.HIDDEN).reshape(3, X.TOP_K, X.HIDDEN)
    out = np.stack([moe_sum({"TOPK": X.TOP_K, "H": X.HIDDEN}, [y[t]])["sum"] for t in range(3)])
    job = {"phase": "moe", "case": "small", "launch": "sum", "H": X.HIDDEN, "alternatives": False, "columns": [], "rows": [0, 3],
           "reference": True, "paths": X._save(tmp_path / "s", y=y, out=out)}
    res = X.compare(job)
    sampled = sum(len({0, X.HIDDEN - 1, (t * 7919) % X.HIDDEN}) for t in range(3))
    assert (res["sum_rows"], res["sum_mismatch"], res["reference_sum_coords"], res["refuted"]) == (3, 0, sampled, {})


def test_a_job_that_raises_is_an_error_not_a_crash(tmp_path) -> None:
    (job,) = _moe_job(tmp_path, "w13")
    res = X.compare({**job, "paths": {**job["paths"], "A": str(tmp_path / "missing.npy")}})
    assert set(res) == {"phase", "case", "launch", "error"} and res["launch"] == "w13"


def test_the_assignment_check_reads_the_launch_blocks() -> None:
    ids = np.array([5, 9, 5, 2])
    L = {"sorted_token_ids": torch.tensor([3, 4, 4, 4, 0, 2, 4, 4, 1, 4, 4, 4]), "expert_ids": torch.tensor([2, 5, 9]),
         "num_tokens_post_padded": torch.tensor([12]), "config": {"BLOCK_SIZE_M": 4}}
    assert X._assignment_ok(L, ids) is True
    assert X._assignment_ok({**L, "expert_ids": torch.tensor([2, 9, 5])}, ids) is False
    assert X._assignment_ok({**L, "sorted_token_ids": torch.tensor([3, 4, 4, 4, 0, 4, 4, 4, 1, 4, 4, 4])}, ids) is False
    naive = {"sorted_token_ids": None, "expert_ids": torch.tensor(ids), "num_tokens_post_padded": torch.tensor([16]),
             "config": {"BLOCK_SIZE_M": 4}}
    assert X._assignment_ok(naive, ids) is True
    assert X._assignment_ok({**naive, "expert_ids": torch.tensor([5, 9, 2, 5])}, ids) is False
    assert X._assignment_ok({**naive, "num_tokens_post_padded": torch.tensor([12])}, ids) is False


def test_reruns_are_compared_as_bits() -> None:
    """Two runs that wrote the same bf16 words, NaNs among them, are deterministic though `torch.equal` says they differ; a NaN
    with one payload bit flipped, -0 for +0, another dtype or another shape is not the same words."""
    words = torch.tensor([[0x3F80, 0x7FC0, -0x40], [0x7F81, 0x0000, -0x8000]], dtype=torch.int16)
    a, b = words.view(torch.bfloat16), words.clone().view(torch.bfloat16)
    assert not torch.equal(a, b) and X.same_words(a, b) and X.same_words(a.t(), b.t())
    nan = words.clone()
    nan[0, 1] ^= 1
    assert torch.isnan(nan.view(torch.bfloat16)[0, 1]) and not X.same_words(a, nan.view(torch.bfloat16))
    zero = words.clone()
    zero[1, 1] ^= -0x8000
    assert torch.equal(a[1, 1], zero.view(torch.bfloat16)[1, 1]) and not X.same_words(a, zero.view(torch.bfloat16))
    assert not X.same_words(a, b.to(torch.float32)) and not X.same_words(a, b.reshape(3, 2))


def test_moe_sum_takes_its_vector_kernel_only_on_aligned_rows_of_whole_packs() -> None:
    """csrc's `can_vec`: hidden contiguous, a whole number of 8-word packs, row strides of whole packs, both pointers 16-byte
    aligned (it reads by strides, so a permuted [token, slot] layout still vectorises)."""
    y = torch.zeros(4, X.TOP_K, 64, dtype=torch.bfloat16)
    out = torch.zeros(4, 64, dtype=torch.bfloat16)
    assert X._vectorised(y, out) and X._vectorised(y.transpose(0, 1), out)
    assert not X._vectorised(y[..., :60], out[:, :60]) and not X._vectorised(y[..., ::2], out[:, :32])
    assert not X._vectorised(torch.zeros(4 * X.TOP_K * 64 + 1, dtype=torch.bfloat16)[1:].view(4, X.TOP_K, 64), out)


WGMMA_PTX = "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 {%f1}, %rd1, %rd2, p, 1, 1;\n"
MMAV2_PTX = ("cvt.rn.f16x2.e4m3x2 %r1, %rs1;\n" * 4 + "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f1}, {%r1}, {%r2}, {%f2};\n" * 8)


def _kernel(d, M: int, Kd: int, Nd: int, warps: int | None, text: str) -> None:
    """One Triton cache entry as Triton 3.7.1 writes it: the TTIR's e4m3 `tt.dot`, the metadata's warps, the PTX."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "fused_moe_kernel.ttir").write_text(f"    %acc = tt.dot %a, %b, %c, inputPrecision = tf32 : tensor<{M}x{Kd}xf8E4M3FN> * "
                                             f"tensor<{Kd}x{Nd}xf8E4M3FN> -> tensor<{M}x{Nd}xf32> loc(#loc1)\n")
    if warps is not None:
        (d / "fused_moe_kernel.json").write_text(f'{{"name": "fused_moe_kernel", "num_warps": {warps}, "num_stages": 4}}')
    (d / "fused_moe_kernel.ptx").write_text(text)


def test_the_ptx_scan_takes_each_kernels_family_from_its_tile(tmp_path) -> None:
    """A swapped 16x64 tile (its dot 64 x 16) runs wgmma; the stock 8x32 tile and the swapped 8x64 one (dot N = 8) run MMAv2 on
    widened e4m3 (the PTX Triton 3.7.1 wrote for them); each against the family its tile is given."""
    _kernel(tmp_path / "a", 64, 128, 16, 4, WGMMA_PTX * 2)
    _kernel(tmp_path / "b", 8, 128, 32, 4, MMAV2_PTX)
    _kernel(tmp_path / "c", 64, 128, 8, 4, MMAV2_PTX)
    tiles = {(64, 16, 128, 4), (8, 32, 128, 4), (64, 8, 128, 4)}
    got = X.ptx(tmp_path, tiles)
    assert got["ok"] and got["files"] == 3 and got["families"] == {"wgmma": 1, "mmav2": 2} and got["missing_tiles"] == []
    a = got["kernels"]["a/fused_moe_kernel.ptx"]
    assert (a["tile"], a["expected"], a["observed"], a["wgmma"], a["mma_sync"]) == ([64, 16, 128, 4], "wgmma", "wgmma",
                                                                                     {WGMMA_PTX.split(" ")[0]: 2}, {})
    assert got["kernels"]["b/fused_moe_kernel.ptx"]["cvt_e4m3"] == 4
    missing = X.ptx(tmp_path, tiles | {(64, 16, 64, 4)})
    assert missing["ok"] is False and missing["missing_tiles"] == [[64, 16, 64, 4]]
    bad = {"a": (64, 128, 16, 4, MMAV2_PTX),                                          # a wgmma tile on mma.sync
           "b": (8, 128, 32, 4, WGMMA_PTX),                                           # an MMAv2 tile on wgmma
           "c": (64, 128, 8, 4, MMAV2_PTX.replace("cvt.rn.f16x2.e4m3x2", "cvt.rn.f16x2.e5m2x2")),  # no e4m3 widening
           "d": (8, 128, 32, 4, MMAV2_PTX.replace("f32.f16.f16.f32", "f32.bf16.bf16.f32")),
           "e": (64, 128, 16, 4, WGMMA_PTX + WGMMA_PTX.replace("m64n16k32.f32.e4m3", "m64n16k16.f32.bf16")),
           "f": (64, 128, 16, 4, WGMMA_PTX + MMAV2_PTX),
           "g": (64, 128, 16, 4, "fma.rn.f32 %f1, %f2, %f3, %f4;\n"),
           "h": (64, 128, 16, 2, WGMMA_PTX),                                          # 2 warps: no wgmma tile
           "i": (64, 128, 16, None, WGMMA_PTX)}                                       # no metadata
    for name, (M, Kd, Nd, warps, text) in bad.items():
        _kernel(tmp_path / name / "k", M, Kd, Nd, warps, text)
        r = X.ptx(tmp_path / name)
        assert r["ok"] is False and r["kernels"]["k/fused_moe_kernel.ptx"]["ok"] is False, name
    _kernel(tmp_path / "j", 64, 128, 16, 4, WGMMA_PTX)
    (tmp_path / "j" / "fused_moe_kernel.ttir").unlink()
    assert X.ptx(tmp_path / "j")["ok"] is False
    assert X.ptx(tmp_path / "nothing")["ok"] is False


#: entries of vLLM d9105ea80's fused_moe/configs/E=256,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
H200_TP8 = {"1": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
            "2": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 5},
            "4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
            "256": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 32, "num_warps": 4, "num_stages": 3},
            "1024": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 32, "num_warps": 4, "num_stages": 4}}


def test_the_h200_decode_tiles_are_mmav2_and_the_rest_wgmma() -> None:
    """The tuned tiles the probe cases run at (M = 1 and 5: 8x32, and 8x64 under SWAP_AB) bind `HopperE4m3MmaV2Dot32_v1`; from
    16 rows on, the QGMMA step."""
    swap = {m: c["BLOCK_SIZE_M"] < 64 and c["BLOCK_SIZE_N"] >= 64 for m, c in H200_TP8.items()}
    assert {m: X.launch_dot(H200_TP8[m], s).id for m, s in swap.items()} == {"1": MV, "2": MV, "4": MV, "256": WG, "1024": WG}
    assert [X.launch_tile({"config": H200_TP8[m], "SWAP": swap[m], "KB": 128}) for m in ("1", "4", "256")] == \
        [(8, 32, 128, 4), (64, 8, 128, 4), (128, 16, 128, 4)]
    assert [c[2] for c in X.MMAV2_PROBE] == [1, 5] and all(c[1] == 256 for c in X.MMAV2_PROBE)


def test_the_programs_bind_at_every_glm53_shape() -> None:
    statics = {(6144, 512, 128, False, MV), (256, 6144, 128, True, MV), (6144, 4096, 128, True, WG)}
    got = X.bind_check(statics)
    b = got["bindings"]
    assert got["ok"] and b["Fp8GroupQuantUe8m0_v1{K=6144,G=128}"] and b["Fp8GroupQuantUe8m0_v1{K=2048,G=128}"]
    assert b["SiluMul_v2{I=256}"] and b["MoeSum_v1{TOPK=8,H=6144}"]
    assert b[f"MoeExpertFp8BlockGemmW_v3{{K=256,N=6144,G=128,KB=128,SWAP=True,DOT={MV}}}"]
    assert b[f"MoeExpertFp8Gemm_v1{{E=256,K=6144,N=512,G=128,KB=128,SWAP=False,DOT={MV}}}"]
    inters = {c[1] for c in X.MOE}
    assert len(b) == (1 + len(inters)) + len(inters) + 1 + 2 * len(statics)


TOTALS0 = {k: 0 for k in X.TOTALS}
W13 = {**TOTALS0, "words": 10, "reference_coords": 3, "quant_rows": 2, "reference_quant_rows": 1}
W2 = {**TOTALS0, "words": 10, "reference_coords": 3, "act_rows": 4, "reference_act_rows": 1, "act_quant_rows": 4,
      "reference_act_quant_rows": 1}
SUM = {**TOTALS0, "sum_rows": 2, "reference_sum_coords": 3}


def _doc() -> dict:
    launch = {"deterministic": True, "statics_ok": True, "assignment_ok": True}
    chain = {"launch": "chain", "ok": True}
    moe = {**launch, "config": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64}, "KB": 128, "SWAP": True, "dot": WG}
    mv = {**launch, "config": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32}, "KB": 128, "SWAP": False, "dot": MV}
    mmav2 = {k: 1 for k in X.mmav2_alternatives()}
    sums = {k: 1 for k in X.moe_sum_alternatives()}
    launches, comparisons = [], []
    for phase, case, tile in (("moe", "tp8_decode", moe), ("mmav2", "probe_decode", mv)):
        at = {"phase": phase, "case": case}
        launches += [{**chain, **at}, {**tile, **at, "launch": "w13"}, {**tile, **at, "launch": "w2"}]
        comparisons += [{**at, "launch": "w13", **W13, "refuted": mmav2 if phase == "mmav2" else {}},
                        {**at, "launch": "w2", **W2, "refuted": {}}, {**at, "launch": "sum", **SUM, "refuted": {}}]
    launches.append({"phase": "sum", "case": "planted", "launch": "sum", "ok": True})
    comparisons.append({"phase": "sum", "case": "planted", "launch": "sum", **SUM, "refuted": sums})
    return {"schema": X.SCHEMA, "gate": {"ok": True}, "errors": {}, "served": {"ok": True}, "selection": {"moe_ok": True},
            "ptx": {"ok": True}, "bind": {"ok": True}, "required": ["moe/tp8_decode", "mmav2/probe_decode", "sum/planted"],
            "launches": launches, "comparisons": comparisons}


@pytest.mark.parametrize("path,value", [(("gate", "ok"), False), (("errors", "moe/x"), "boom"), (("served", "ok"), False),
                                        (("selection", "moe_ok"), False), (("ptx", "ok"), None), (("ptx", "ok"), False),
                                        (("bind", "ok"), False), (("required",), []),
                                        (("launches", 0, "ok"), False), (("launches", 1, "statics_ok"), False),
                                        (("launches", 2, "assignment_ok"), False), (("launches", 1, "deterministic"), False),
                                        (("launches", 4, "dot"), WG), (("launches", 6, "ok"), False),
                                        (("comparisons", 0, "mismatch"), 1), (("comparisons", 0, "unresolved"), 2),
                                        (("comparisons", 0, "quant_mismatch"), 1), (("comparisons", 0, "reference_quant_rows"), 0),
                                        (("comparisons", 1, "act_mismatch"), 1), (("comparisons", 1, "act_quant_mismatch"), 1),
                                        (("comparisons", 1, "act_rows"), 0), (("comparisons", 1, "reference_act_quant_rows"), 0),
                                        (("comparisons", 2, "sum_mismatch"), 1), (("comparisons", 2, "reference_sum_coords"), 0),
                                        (("comparisons", 4, "twin_mismatch"), 1), (("comparisons", 3, "refuted", "fma_chain"), 0),
                                        (("comparisons", 6, "refuted", "ftz"), 0), (("comparisons", 6, "sum_rows"), 0)])
def test_the_verdict_needs_every_check(path: tuple, value) -> None:
    assert X._verdict(_doc())
    doc = copy.deepcopy(_doc())
    node = doc
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    assert not X._verdict(doc)


def test_a_run_missing_a_required_case_or_an_alternative_is_not_a_go() -> None:
    doc = _doc()
    doc["required"].append("moe/tp8_prefill")
    assert not X._verdict(doc)
    doc = copy.deepcopy(_doc())
    del doc["comparisons"][3]["refuted"]["qgmma_k32"]
    assert not X._verdict(doc)
    doc = copy.deepcopy(_doc())
    doc["comparisons"] = [c for c in doc["comparisons"] if not (c["case"] == "tp8_decode" and c["launch"] == "sum")]
    assert not X._verdict(doc)


def test_each_case_has_its_own_verdict() -> None:
    doc = _doc()
    got = X.cases(doc)
    assert {k: c["ok"] for k, c in got.items()} == {"moe/tp8_decode": True, "mmav2/probe_decode": True, "sum/planted": True}
    assert got["moe/tp8_decode"]["tiles"] == ["w13:16x64x128/swap/wgmma", "w2:16x64x128/swap/wgmma"]
    assert got["mmav2/probe_decode"]["tiles"] == ["w13:8x32x128/mmav2", "w2:8x32x128/mmav2"]
    assert got["mmav2/probe_decode"]["refuted"] == {k: 1 for k in X.mmav2_alternatives()}
    doc["launches"][2]["deterministic"] = False
    doc["errors"]["moe/tp8_small"] = "boom"
    got = X.cases(doc)
    assert {k: c["ok"] for k, c in got.items()} == {"moe/tp8_decode": False, "mmav2/probe_decode": True, "sum/planted": True,
                                                    "moe/tp8_small": False}


def test_required_cases_are_every_moe_case_both_probes_and_the_planted_sum() -> None:
    assert X.required_cases(False) == [f"moe/{c[0]}" for c in X.MOE] + ["mmav2/probe_decode", "mmav2/probe_small", "sum/planted"]
    assert X.required_cases(True) == ["moe/tp8_decode", "moe/tp8_small", "moe/whole_decode", "mmav2/probe_decode",
                                      "mmav2/probe_small", "sum/planted"]


def _stub_vllm(monkeypatch) -> None:
    import contextlib
    import sys
    import types

    vc = types.ModuleType("vllm.config")
    vc.CompilationConfig = vc.VllmConfig = lambda **kw: kw
    vc.set_current_vllm_config = lambda cfg: contextlib.nullcontext()
    ws = types.ModuleType("vllm.v1.worker.workspace")
    ws.is_workspace_manager_initialized = lambda: True
    ws.init_workspace_manager = lambda device: None
    for name, mod in (("vllm", types.ModuleType("vllm")), ("vllm.config", vc), ("vllm.v1", types.ModuleType("vllm.v1")),
                      ("vllm.v1.worker", types.ModuleType("vllm.v1.worker")), ("vllm.v1.worker.workspace", ws)):
        monkeypatch.setitem(sys.modules, name, mod)


def test_main_end_to_end_on_stubbed_launches(tmp_path, monkeypatch, capsys) -> None:
    """`main` with the GPU's pieces stood in for (gate, the served read-back, selection, the launches as kernel words made here, the
    planted sum, the PTX scan) and vLLM's modules stubbed: the switches cleared before vLLM, the comparison jobs through the spawned
    worker pool, each launch at its tile's DOT, the probe's refutations, the per-case verdicts, the sealed record and the exit
    status, so a slip in the orchestration shows here and not in the H200's first hour."""
    import json

    from verity_vllm.engine.env import SERVED_DEFAULTS

    _stub_vllm(monkeypatch)
    for k in SERVED_DEFAULTS + ("VLLM_ENABLE_V1_MULTIPROCESSING", "TRITON_CACHE_DIR"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "0")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": True})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    monkeypatch.setattr(X, "served", lambda: {"ok": True})
    monkeypatch.setattr(X, "selection", lambda: {"moe_ok": True})
    scanned = []

    def ptx(cache, tiles):
        scanned.append(tiles)
        return {"files": 2, "kernels": {}, "ok": True}

    monkeypatch.setattr(X, "ptx", ptx)
    monkeypatch.setattr(X, "moe_cases", lambda quick: [("small", 256, 3, 0), ("swapped", 256, 3, 0)])
    monkeypatch.setattr(X, "MMAV2_PROBE", (("probe_decode", 256, 1, 0),))
    monkeypatch.setattr(X, "required_cases", lambda quick: ["moe/small", "moe/swapped", "mmav2/probe_decode", "sum/planted"])
    tiles = {"small": ({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "num_warps": 4}, 128, False),
             "swapped": ({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}, 64, True),
             "probe_decode": ({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "num_warps": 4}, 128, False)}
    seen = []

    def moe(work, phase, name, inter, M, hot, seed):
        seen.append((phase, name, seed))
        cfg, KB, swap = tiles[name]
        dot = X.launch_dot(cfg, swap).id
        at = {"phase": phase, "case": name}
        launches = [{**at, "launch": "chain", "ok": True, "kernel": X.KERNEL_TYPES, "counts": {}}]
        jobs = []
        for kind in ("w13", "w2"):
            probe = phase == "mmav2" and kind == "w13"
            launches.append({**at, "launch": kind, "K": K, "N": 64, "pairs": 8 * M, "KB": KB, "SWAP": swap, "config": cfg,
                             "deterministic": True, "statics_ok": True, "assignment_ok": True, "dot": dot})
            jobs += _moe_job(tmp_path / name, kind, M=M, Nc=64, KB=KB, swap=swap, dot=dot, probe=probe, phase=phase, case=name,
                             flip=name == "swapped" and kind == "w2")
        y = np.zeros((M, X.TOP_K, 64), dtype=np.uint16)
        y[:, :, 3] = 0x3F80
        out = np.stack([moe_sum({"TOPK": X.TOP_K, "H": 64}, [y[t]])["sum"] for t in range(M)])
        jobs.append({**at, "launch": "sum", "H": 64, "alternatives": False, "columns": [], "rows": [0, M], "reference": True,
                     "paths": X._save(tmp_path / name / "sum", y=y, out=out)})
        return launches, jobs

    monkeypatch.setattr(X, "moe_launches", moe)
    monkeypatch.setattr(X, "sum_probe_launch", lambda work, seed: ({"phase": "sum", "case": "planted", "launch": "sum", "ok": True},
                                                                   _sum_job(tmp_path)[0]))
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1"]) == 1
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is False and not doc["errors"] and doc["bind"]["ok"]
    assert [s[:2] for s in seen] == [("moe", "small"), ("moe", "swapped"), ("mmav2", "probe_decode")]
    assert {k: c["ok"] for k, c in doc["cases"].items()} == {"moe/small": True, "moe/swapped": False, "mmav2/probe_decode": True,
                                                             "sum/planted": True}
    assert doc["cases"]["mmav2/probe_decode"]["refuted"].keys() == X.mmav2_alternatives().keys()
    assert doc["cases"]["sum/planted"]["refuted"].keys() == X.moe_sum_alternatives().keys()
    assert doc["dots"] == {MV: 4, WG: 2} and scanned == [{(8, 32, 128, 4), (64, 16, 64, 4)}]
    assert doc["totals"]["mismatch"] == 1 and doc["env_cleared"] == {"VLLM_USE_DEEP_GEMM": "0", "VLLM_BATCH_INVARIANT": "1"}
    assert "VLLM_USE_DEEP_GEMM" not in doc["env"] and doc["env"]["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    printed = capsys.readouterr().out
    assert "case moe/swapped" in printed and "GLM53-BLOCK-FP8 NO-GO" in printed and "no-go cases=['moe/swapped']" in printed
    assert "missing=[]" in printed and "w13:8x32x128/mmav2" in printed


def test_main_stops_at_a_failed_gate(tmp_path, monkeypatch, capsys) -> None:
    import json

    from verity_vllm.engine.env import SERVED_DEFAULTS

    _stub_vllm(monkeypatch)
    for k in SERVED_DEFAULTS + ("VLLM_ENABLE_V1_MULTIPROCESSING", "TRITON_CACHE_DIR"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": False})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    with pytest.raises(SystemExit) as e:
        X.main(["--out", str(tmp_path / "out"), "--scratch", str(tmp_path)])
    assert e.value.code == 3 and "GLM53-BLOCK-FP8 GATE-FAIL" in capsys.readouterr().out
    assert json.loads((tmp_path / "out" / X.RECORD).read_text())["ok"] is False


def test_the_record_is_a_registered_property() -> None:
    from verity_vllm.properties.record import seal

    doc = seal(_doc(), X._verdict(_doc()))
    assert REGISTRY["glm53_block_fp8_capture"].schema == X.SCHEMA and doc["ok"] and doc["digest"] == digest_of(doc)
