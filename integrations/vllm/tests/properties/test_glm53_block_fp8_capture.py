"""The GLM-5.3 block-FP8 capture without a GPU: its comparison jobs over operands and words made here (the kernels' words stood in
for by the twins), a flipped word or quant byte found and located, the launch-assignment check, the PTX scan, the bindings, and
the verdict (the GPU run is `tests/properties/glm53_block_fp8_capture_gpu.py --out ...` on an H200)."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from tests.properties import glm53_block_fp8_capture_gpu as X
from verity_vllm.program.kernels import deepgemm_twins as DGT
from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.kernels import fp8_moe_rows as MR
from verity_vllm.properties.record import REGISTRY, digest_of

K, N, GQ = 256, 200, X.GROUP
WG, MV = "HopperE4m3QgmmaDot32_v1", "HopperE4m3MmaV2Dot32_v1"


def _bf16_rows(rng: np.random.Generator, M: int) -> np.ndarray:
    x = rng.standard_normal((M, K)).astype(np.float32) * np.exp2(rng.uniform(-3, 3, (M, 1))).astype(np.float32)
    x[1] = 0
    x[3] *= np.float32(2.0 ** -130)
    return (x.view(np.uint32) >> 16).astype(np.uint16)


def _e4m3(rng: np.random.Generator, *shape: int) -> np.ndarray:
    b = rng.integers(0, 256, shape).astype(np.uint8)
    return np.where((b & 0x7F) == 0x7F, b ^ 1, b).astype(np.uint8)


def _scales(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return np.exp2(rng.uniform(-12, -6, shape)).astype(np.float32).view(np.uint32)


def _quant(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q, s = zip(*(DR.fp8_group_quant_row(r, G=GQ) for r in x))
    return np.stack(q), np.stack(s)


def _jobs(tmp_path, *, flip_word: bool = False, flip_quant: bool = False) -> list[dict]:
    return X.dense_jobs(_dense_job(tmp_path, flip_word=flip_word, flip_quant=flip_quant))


def _dense_job(tmp_path, *, flip_word: bool = False, flip_quant: bool = False) -> dict:
    rng = np.random.default_rng(1)
    M = 5
    x = _bf16_rows(rng, M)
    q, s = _quant(x)
    w, ws = _e4m3(rng, N, K), _scales(rng, X.cdiv(N, GQ), K // GQ)
    out = np.stack([DR.scaled_mm_fp8_block_row(q[r], s[r], w, ws, G=GQ)[0] for r in range(M)])
    if flip_word:
        out[2, 150] ^= 1
    if flip_quant:
        q = q.copy()
        q[4, 9] ^= 1
    return {"phase": "dense", "case": "ragged", "K": K, "N": N, "M": M,
            "paths": X._save(tmp_path / "dense", x=x, q=q, s=s, w=w, ws=ws, out=out)}


def _sampled_columns(rows) -> int:
    return sum(len({0, N - 1, (r * 7919) % N}) for r in rows)


def _total(results: list[dict]) -> dict:
    assert not [r for r in results if "error" in r], results
    return {k: sum(r[k] for r in results) for k in X.TOTALS}


def test_dense_jobs_take_the_kernel_words_and_sample_the_special_rows(tmp_path) -> None:
    jobs = _jobs(tmp_path)
    assert sorted(r for j in jobs for r in j["reference_rows"]) == [0, 2, 3, 4]
    t = _total([X.compare(j) for j in jobs])
    assert t == {"words": 5 * N, "mismatch": 0, "unresolved": 0, "quant_rows": 5, "quant_mismatch": 0,
                 "reference_coords": _sampled_columns((0, 2, 3, 4)), "reference_quant_rows": 4, "twin_mismatch": 0, "act_rows": 0,
                 "act_mismatch": 0, "reference_act_rows": 0, "ue8m0_rows": 0, "ue8m0_mismatch": 0, "reference_ue8m0_rows": 0}


def test_a_flipped_word_or_quant_byte_is_found(tmp_path) -> None:
    res = [X.compare(j) for j in _jobs(tmp_path, flip_word=True)]
    assert _total(res)["mismatch"] == 1
    assert next(r["first_mismatch"] for r in res if r["first_mismatch"])["column"] == 150
    res = [X.compare(j) for j in _jobs(tmp_path, flip_quant=True)]
    assert _total(res)["quant_mismatch"] == 1


def _moe_job(tmp_path, kind: str, *, KB: int = 32, swap: bool = False, dot: str = WG, words_dot: str | None = None, flip: bool = False,
             flip_act: bool = False) -> list[dict]:
    """A w13-like launch (3 tokens x 8 slots over 5 used experts), a w2-like one (24 rows, one slot each, the routed weight), or the
    served w2 (the w2-like launch whose A and A_scale are `silu_and_mul_per_block_quant` of 24 [gate || up] rows), at DOT `dot`, its
    words made with `words_dot`'s step (default `dot`)."""
    rng = np.random.default_rng(2)
    M, used = 3, np.array([3, 17, 40, 41, 255])
    ids = np.stack([rng.permutation(used)[:5].tolist() + rng.permutation(used)[:3].tolist() for _ in range(M)]).reshape(-1)
    umap = np.full(X.NUM_EXPERTS, -1)
    umap[used] = np.arange(used.size)
    tw = (rng.uniform(0.05, 1, M * X.TOP_K).astype(np.float32)).view(np.uint32)
    rows = M if kind == "w13" else M * X.TOP_K
    top_k, mul = (X.TOP_K, False) if kind == "w13" else (1, True)
    x = _bf16_rows(rng, max(rows, 4))[:rows]
    gu = np.concatenate([x, _bf16_rows(rng, max(rows, 4))[:rows]], axis=1)
    A, As = MR.silu_mul_fp8_block_quant_rows(gu, G=GQ) if kind == "served_w2" else _quant(x)
    B, Bs = _e4m3(rng, used.size, N, K), _scales(rng, used.size, X.cdiv(N, GQ), K // GQ)
    C = np.stack([MR.moe_expert_fp8_block_row(A[p // top_k], As[p // top_k], B[umap[ids[p]]], Bs[umap[ids[p]]], G=GQ, KB=KB, swap=swap,
                                              g=int(tw[p]) if mul else None, dot=MR.TWIN_DOT_OF[words_dot or dot])[0]
                  for p in range(M * X.TOP_K)])
    if flip:
        C[13, 7] ^= 0x8000
    if flip_act:
        gu[19, 300] ^= 0x0100
    arrays = {"A": A, "As": As, "B": B, "Bs": Bs, "ids": ids, "umap": umap, "tw": tw, "C": C, **({"x": x} if kind == "w13" else {}),
              **({"gu": gu} if kind == "served_w2" else {})}
    job = {"phase": "moe", "case": "small", "launch": kind, "K": K, "N": N, "KB": KB, "SWAP": swap, "dot": dot, "mul": mul, "top_k": top_k,
           "pairs": M * X.TOP_K, "statics_known": True, "paths": X._save(tmp_path / kind, **arrays)}
    return [{**job, "pair_range": [0, M * X.TOP_K], "reference": True, "quant_rows": list(range(M)) if kind == "w13" else []}]


@pytest.mark.parametrize("kind,KB,swap,dot", [("w13", 32, False, WG), ("w2", 32, False, WG), ("w2", 64, True, WG), ("served_w2", 32, False, WG),
                                             ("w13", 128, False, MV), ("w2", 128, True, MV), ("served_w2", 128, False, MV)])
def test_moe_jobs_take_the_kernel_words(tmp_path, kind: str, KB: int, swap: bool, dot: str) -> None:
    t = _total([X.compare(j) for j in _moe_job(tmp_path, kind, KB=KB, swap=swap, dot=dot)])
    assert t["words"] == 24 * N and t["mismatch"] == t["unresolved"] == t["twin_mismatch"] == t["quant_mismatch"] == t["act_mismatch"] == 0
    assert t["reference_coords"] == _sampled_columns((0, 23)) and t["quant_rows"] == t["reference_quant_rows"] * 3 == (3 if kind == "w13" else 0)
    assert (t["act_rows"], t["reference_act_rows"]) == ((24, 1) if kind == "served_w2" else (0, 0))
    assert (t["ue8m0_rows"], t["ue8m0_mismatch"], t["reference_ue8m0_rows"]) == (0, 0, 0)
    flipped = [X.compare(j) for j in _moe_job(tmp_path, kind, KB=KB, swap=swap, dot=dot, flip=True)]
    assert _total(flipped)["mismatch"] == 1 and flipped[0]["first_mismatch"]["pair"] == 13


def _ue8m0_job(tmp_path, *, flip: bool = False) -> list[dict]:
    """A case's ue8m0 input quant as `jobs_append_ue8m0` saves it: 4 bf16 rows (one zero, one of subnormals), their (q, s) from
    the twin standing in for the kernel's bits."""
    rng = np.random.default_rng(3)
    M = 4
    x = _bf16_rows(rng, M)
    q, s = DGT.ue8m0_group_quant_rows(x, G=GQ)
    if flip:
        q = q.copy()
        q[2, 5] ^= 1
    job = {"phase": "moe", "case": "small", "launch": "ue8m0_quant", "K": K, "paths": X._save(tmp_path / "ue8m0", x=x, q=q, s=s)}
    return [{**job, "rows": [0, M], "reference": True}]


def test_ue8m0_jobs_take_the_kernel_quant_bits(tmp_path) -> None:
    t = _total([X.compare(j) for j in _ue8m0_job(tmp_path)])
    assert t == {**{k: 0 for k in X.TOTALS}, "ue8m0_rows": 4, "reference_ue8m0_rows": 1}
    flipped = [X.compare(j) for j in _ue8m0_job(tmp_path, flip=True)]
    assert _total(flipped)["ue8m0_mismatch"] == 1 and flipped[0]["first_mismatch"] == {"ue8m0_row": 2}


@pytest.mark.parametrize("dot,words_dot", [(MV, WG), (WG, MV)])
def test_a_launch_checked_at_the_other_dot_mismatches(tmp_path, dot: str, words_dot: str) -> None:
    """The stock 8x32x128 tile's words (MMAv2) against the wgmma Program, and a wgmma tile's against the MMAv2 one: the words the
    two steps round differently are found, so the run binds each launch's own DOT and not either for both."""
    t = _total([X.compare(j) for j in _moe_job(tmp_path, "w13", KB=128, dot=dot, words_dot=words_dot)])
    assert t["words"] == 24 * N and t["mismatch"] > 0 and t["twin_mismatch"] == 0


def test_a_served_w2_input_that_is_not_the_activation_is_found(tmp_path) -> None:
    """A w13 output word flipped after the activation ran: the served w2 launch's A no longer is `SiluMulFp8BlockQuant_v1` of it."""
    (res,) = [X.compare(j) for j in _moe_job(tmp_path, "served_w2", flip_act=True)]
    assert (res["act_rows"], res["act_mismatch"], res["mismatch"]) == (24, 1, 0) and res["first_mismatch"] == {"act_pair": 19}


def test_a_launch_at_statics_no_program_binds_is_unresolved(tmp_path) -> None:
    (job,) = _moe_job(tmp_path, "w13")
    res = X.compare({**job, "statics_known": False})
    assert res["unresolved"] == res["words"] == 24 * N
    (job,) = _moe_job(tmp_path, "served_w2")
    res = X.compare({**job, "statics_known": False})
    assert res["unresolved"] == res["words"] == 24 * N and (res["act_rows"], res["act_mismatch"]) == (24, 0)


def test_the_assignment_check_reads_the_launch_blocks() -> None:
    ids = np.array([5, 9, 5, 2])
    L = {"sorted_token_ids": torch.tensor([3, 4, 4, 4, 0, 2, 4, 4, 1, 4, 4, 4]), "expert_ids": torch.tensor([2, 5, 9]),
         "num_tokens_post_padded": torch.tensor([12]), "config": {"BLOCK_SIZE_M": 4}}
    assert X._assignment_ok(L, ids) is True
    assert X._assignment_ok({**L, "expert_ids": torch.tensor([2, 9, 5])}, ids) is False
    assert X._assignment_ok({**L, "sorted_token_ids": torch.tensor([3, 4, 4, 4, 0, 4, 4, 4, 1, 4, 4, 4])}, ids) is False
    assert X._assignment_ok({**L, "sorted_token_ids": None}, ids) is None


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
    widened e4m3 (the PTX Triton 3.7.1 wrote for them, run r20261012-215811-1ef7); each against the family its tile is given."""
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
            "1024": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 32, "num_warps": 4, "num_stages": 4},
            "triton_version": "3.7.1"}


def test_the_h200_small_batch_tiles_are_no_wgmma_and_the_widened_configs_are() -> None:
    swap = {m: c["BLOCK_SIZE_M"] < 64 and c["BLOCK_SIZE_N"] >= 64 for m, c in H200_TP8.items() if isinstance(c, dict)}
    assert {m: X.wgmma_tile(H200_TP8[m], s) for m, s in swap.items()} == {"1": False, "2": False, "4": False, "256": True, "1024": True}
    assert {m: X.launch_dot(H200_TP8[m], s).id for m, s in swap.items()} == {"1": MV, "2": MV, "4": MV, "256": WG, "1024": WG}
    assert [X.launch_tile({"config": H200_TP8[m], "SWAP": swap[m], "KB": 128}) for m in ("1", "4", "256")] == \
        [(8, 32, 128, 4), (64, 8, 128, 4), (128, 16, 128, 4)]
    assert X.wgmma_tile(X.BATCH_INVARIANT_CONFIG, False) and not X.wgmma_tile({**H200_TP8["1024"], "num_warps": 2}, False)
    assert X.wgmma_tile({**H200_TP8["4"], "BLOCK_SIZE_M": 16}, True) and not X.wgmma_tile(H200_TP8["4"], True)
    wide, changed = X.widen_for_wgmma(H200_TP8)
    assert sorted(changed) == ["1", "2", "4"] and wide["triton_version"] == "3.7.1"
    assert wide["1"] == {**H200_TP8["1"], "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64} and wide["4"] == {**H200_TP8["4"], "BLOCK_SIZE_M": 16}
    assert all(X.wgmma_tile(c, c["BLOCK_SIZE_M"] < 64 and c["BLOCK_SIZE_N"] >= 64) for c in wide.values() if isinstance(c, dict))
    assert {m: wide[m] for m in ("256", "1024")} == {m: H200_TP8[m] for m in ("256", "1024")}


def test_the_programs_bind_at_every_glm53_shape() -> None:
    got = X.bind_check({(32, False, WG), (128, True, MV)})
    assert got["ok"] and got["bindings"]["ScaledMmFp8Block_v1{K=6144,N=2624,G=128}"] and got["bindings"]["SiluMulFp8BlockQuant_v1{I=256,G=128}"]
    assert got["bindings"][f"MoeExpertFp8BlockGemmW_v3{{K=256,N=6144,G=128,KB=128,SWAP=True,DOT={MV}}}"]
    assert got["bindings"]["Fp8GroupQuantUe8m0_v1{K=6144,G=128}"]
    assert len(got["bindings"]) == len(X.DENSE) + len({K for _, K, _ in X.DENSE}) + 1 + 5 * len({c[1] for c in X.MOE})


TOTALS = {"words": 10, "mismatch": 0, "unresolved": 0, "quant_rows": 2, "quant_mismatch": 0, "reference_coords": 3,
          "reference_quant_rows": 1, "twin_mismatch": 0, "act_rows": 0, "act_mismatch": 0, "reference_act_rows": 0,
          "ue8m0_rows": 0, "ue8m0_mismatch": 0, "reference_ue8m0_rows": 0}
ACT = {"act_rows": 4, "reference_act_rows": 1}
UE8 = {"ue8m0_rows": 2, "reference_ue8m0_rows": 1}


def _doc() -> dict:
    launch = {"deterministic": True, "statics_ok": True, "assignment_ok": True}
    moe = {**launch, "phase": "moe", "case": "tp8_decode", "config": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64}, "KB": 128, "SWAP": True,
           "dot": WG}
    return {"schema": X.SCHEMA, "gate": {"ok": True}, "errors": {}, "selection": {"dense_ok": True, "moe_ok": True}, "ptx": {"ok": True},
            "bind": {"ok": True},
            "launches": [{**launch, "phase": "dense", "case": "o_proj", "M": 1, "operands_ok": True}, {**moe, "launch": "w13"},
                         {**moe, "launch": "w2"},
                         {"phase": "moe", "case": "tp8_decode", "launch": "ue8m0_quant", "M": 2, "deterministic": True,
                          "operands_ok": True}],
            "comparisons": [{"phase": "dense", "case": "o_proj", "launch": None, "M": 1, **TOTALS},
                            {"phase": "moe", "case": "tp8_decode", "launch": "w13", "M": None, **TOTALS},
                            {"phase": "moe", "case": "tp8_decode", "launch": "ue8m0_quant", "M": None,
                             **{k: 0 for k in X.TOTALS}, **UE8},
                            {"phase": "moe", "case": "tp8_decode", "launch": "served_w2", "M": None, **TOTALS, **ACT}],
            "totals": {**TOTALS, **ACT, **UE8}}


@pytest.mark.parametrize("path,value", [(("gate", "ok"), False), (("errors", "moe/x"), "boom"), (("selection", "moe_ok"), False),
                                        (("ptx", "ok"), None), (("totals", "mismatch"), 1), (("totals", "unresolved"), 2),
                                        (("totals", "reference_coords"), 0), (("totals", "twin_mismatch"), 1),
                                        (("totals", "act_rows"), 0), (("totals", "act_mismatch"), 1), (("totals", "reference_act_rows"), 0),
                                        (("totals", "ue8m0_rows"), 0), (("totals", "ue8m0_mismatch"), 1),
                                        (("totals", "reference_ue8m0_rows"), 0),
                                        (("launches", 1, "statics_ok"), False), (("launches", 1, "assignment_ok"), False),
                                        (("ptx", "ok"), False), (("errors", "moe_configs"), "boom"),
                                        (("launches", 0, "deterministic"), False), (("launches", 0, "operands_ok"), False),
                                        (("launches", 3, "deterministic"), False), (("launches", 3, "operands_ok"), False)])
def test_the_verdict_needs_every_check(path: tuple, value) -> None:
    assert X._verdict(_doc())
    doc = copy.deepcopy(_doc())
    node = doc
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    assert not X._verdict(doc)
    assert not X._verdict({**_doc(), "launches": [_doc()["launches"][0]]})       # a dense-only run is not a GO


def test_each_case_has_its_own_verdict() -> None:
    doc = _doc()
    got = X.cases(doc)
    assert {k: c["ok"] for k, c in got.items()} == {"dense/o_proj/M1": True, "moe/tp8_decode": True}
    assert got["moe/tp8_decode"]["tiles"] == ["w13:16x64x128/swap/wgmma", "w2:16x64x128/swap/wgmma"]
    doc["launches"][2] = {**doc["launches"][2], "config": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32}, "SWAP": False, "dot": MV}
    assert X.cases(doc)["moe/tp8_decode"] == {**got["moe/tp8_decode"], "tiles": ["w13:16x64x128/swap/wgmma", "w2:8x32x128/mmav2"]}
    doc["launches"][2]["deterministic"] = False
    doc["comparisons"][0]["mismatch"] = 1
    doc["errors"]["dense/q_b_proj/M4"] = "boom"
    got = X.cases(doc)
    assert {k: c["ok"] for k, c in got.items()} == {"dense/o_proj/M1": False, "moe/tp8_decode": False, "dense/q_b_proj/M4": False}
    doc = _doc()
    doc["comparisons"].pop()                                                    # a MoE case whose act rows nobody compared
    assert {k: c["ok"] for k, c in X.cases(doc).items()} == {"dense/o_proj/M1": True, "moe/tp8_decode": False}


def test_the_dense_cases_are_glm53s_nine_fp8_linears() -> None:
    assert len(X.DENSE) == 9 and dict((n, (K, N)) for n, K, N in X.DENSE)["kv_b_proj"] == (512, 3584)
    assert all(K % X.GROUP == 0 for _, K, _ in X.DENSE) and [n for n, _, N in X.DENSE if N % X.GROUP] == ["fused_qkv_a_proj"]


@pytest.mark.parametrize("wgmma_configs", [False, True])
def test_main_end_to_end_on_stubbed_launches(tmp_path, monkeypatch, capsys, wgmma_configs: bool) -> None:
    """`main` with the GPU's pieces stood in for (gate, selection, the launches as kernel words made here, the PTX scan, the tuned
    configs) and vLLM's config module stubbed: the comparison jobs through the spawned worker pool, each launch at its tile's DOT (a
    swapped 16x64 wgmma tile and the stock 8x32x128 MMAv2 one), the per-case verdicts, the sealed record and the exit status, so a
    slip in the orchestration shows here and not in the H200's first hour."""
    import contextlib
    import json
    import sys
    import types

    vc = types.ModuleType("vllm.config")
    vc.CompilationConfig = vc.VllmConfig = lambda **kw: kw
    vc.set_current_vllm_config = lambda cfg: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", vc)
    for k in ("VLLM_USE_DEEP_GEMM", "VLLM_ENABLE_V1_MULTIPROCESSING", "TRITON_CACHE_DIR", "VLLM_BATCH_INVARIANT", "VLLM_TUNED_CONFIG_FOLDER"):
        monkeypatch.setenv(k, "")
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": True})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    monkeypatch.setattr(X, "selection", lambda: {"dense_ok": True, "moe_ok": True})
    scanned = []

    def ptx(cache, tiles):
        scanned.append(tiles)
        return {"files": 2, "kernels": {}, "ok": True}

    monkeypatch.setattr(X, "ptx", ptx)
    monkeypatch.setattr(X, "write_wgmma_configs", lambda folder: {"folder": str(folder), "files": {}})
    monkeypatch.setattr(X, "dense_cases", lambda quick: [("ragged", K, N, 5)])
    monkeypatch.setattr(X, "moe_cases", lambda quick: [("small", 256, 3, 0), ("stock", 256, 1, 0)])
    dense = _dense_job(tmp_path / "d")
    monkeypatch.setattr(X, "dense_launch", lambda work, name, K_, N_, M, seed: (
        {"phase": "dense", "case": name, "K": K_, "N": N_, "M": M, "kernel": "CutlassFp8BlockScaledMMKernel", "deterministic": True,
         "operands_ok": True}, {**dense, "case": name}))
    tiles = {"small": ({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}, 64, True),
             "stock": ({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "num_warps": 4}, 128, False)}

    def moe(work, name, inter, M, hot, seed, batch_invariant):
        cfg, KB, swap = tiles[name]
        dot = X.launch_dot(cfg, swap).id
        launches, jobs = [], []
        for kind in ("w13", "w2", "served_w2"):
            launches.append({"phase": "moe", "case": name, "launch": kind, "K": K, "N": N, "pairs": 24, "KB": KB, "SWAP": swap,
                             "config": cfg, "deterministic": True, "statics_ok": True, "assignment_ok": True, "dot": dot})
            jobs += [{**j, "case": name} for j in _moe_job(tmp_path / name, kind, KB=KB, swap=swap, dot=dot,
                                                           flip=name == "small" and kind == "w2")]
        launches.append({"phase": "moe", "case": name, "launch": "ue8m0_quant", "M": 4, "K": K, "pairs": 4, "deterministic": True,
                         "operands_ok": True})
        jobs += [{**j, "case": name} for j in _ue8m0_job(tmp_path / name)]
        return launches, jobs

    monkeypatch.setattr(X, "moe_launches", moe)
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1"] + (["--wgmma-configs"] if wgmma_configs else [])) == 1
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is False and not doc["errors"] and doc["bind"]["ok"]
    assert {k: c["ok"] for k, c in doc["cases"].items()} == {"dense/ragged/M5": True, "moe/small": False, "moe/stock": True}
    assert doc["cases"]["moe/stock"]["tiles"] == ["w13:8x32x128/mmav2", "w2:8x32x128/mmav2", "served_w2:8x32x128/mmav2"]
    assert doc["dots"] == {WG: 3, MV: 3} and scanned == [{(64, 16, 64, 4), (8, 32, 128, 4)}]
    assert doc["totals"]["mismatch"] == 1
    assert doc["env"].get("VLLM_TUNED_CONFIG_FOLDER") == (str(out / "moe_configs") if wgmma_configs else None)
    assert (doc["totals"]["act_rows"], doc["totals"]["act_mismatch"], doc["totals"]["reference_act_rows"]) == (48, 0, 2)
    assert (doc["totals"]["ue8m0_rows"], doc["totals"]["ue8m0_mismatch"], doc["totals"]["reference_ue8m0_rows"]) == (8, 0, 2)
    assert doc["env"]["VLLM_BATCH_INVARIANT"] == "0" and doc["env"]["VLLM_USE_DEEP_GEMM"] == "0"
    printed = capsys.readouterr().out
    assert "case moe/small" in printed and "GLM53-BLOCK-FP8 NO-GO" in printed and "no-go cases=['moe/small']" in printed
    assert f"dots={json.dumps({WG: 3, MV: 3})}" in printed and "w13:8x32x128/mmav2" in printed


def test_the_record_is_a_registered_property() -> None:
    from verity_vllm.properties.record import seal

    doc = seal(_doc(), X._verdict(_doc()))
    assert REGISTRY["glm53_block_fp8_capture"].schema == X.SCHEMA and doc["ok"] and doc["digest"] == digest_of(doc)
