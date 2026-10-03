"""The GLM-5.3 block-FP8 capture without a GPU: its comparison jobs over operands and words made here (the kernels' words stood in
for by the twins), a flipped word or quant byte found and located, the launch-assignment check, the PTX scan, the bindings, and
the verdict (the GPU run is `tests/properties/glm53_block_fp8_capture_gpu.py --out ...` on an H200)."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from tests.properties import glm53_block_fp8_capture_gpu as X
from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.properties.record import REGISTRY, digest_of

K, N, GQ = 256, 200, X.GROUP


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
    job = {"phase": "dense", "case": "ragged", "K": K, "N": N, "M": M,
           "paths": X._save(tmp_path / "dense", x=x, q=q, s=s, w=w, ws=ws, out=out)}
    return X.dense_jobs(job)


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
                 "reference_coords": _sampled_columns((0, 2, 3, 4)), "reference_quant_rows": 4, "twin_mismatch": 0}


def test_a_flipped_word_or_quant_byte_is_found(tmp_path) -> None:
    res = [X.compare(j) for j in _jobs(tmp_path, flip_word=True)]
    assert _total(res)["mismatch"] == 1
    assert next(r["first_mismatch"] for r in res if r["first_mismatch"])["column"] == 150
    res = [X.compare(j) for j in _jobs(tmp_path, flip_quant=True)]
    assert _total(res)["quant_mismatch"] == 1


def _moe_job(tmp_path, kind: str, *, KB: int = 32, swap: bool = False, flip: bool = False) -> list[dict]:
    """A w13-like launch (3 tokens x 8 slots over 5 used experts) or a w2-like one (24 rows, one slot each, the routed weight)."""
    rng = np.random.default_rng(2)
    M, used = 3, np.array([3, 17, 40, 41, 255])
    ids = np.stack([rng.permutation(used)[:5].tolist() + rng.permutation(used)[:3].tolist() for _ in range(M)]).reshape(-1)
    umap = np.full(X.NUM_EXPERTS, -1)
    umap[used] = np.arange(used.size)
    tw = (rng.uniform(0.05, 1, M * X.TOP_K).astype(np.float32)).view(np.uint32)
    rows = M if kind == "w13" else M * X.TOP_K
    top_k, mul = (X.TOP_K, False) if kind == "w13" else (1, True)
    x = _bf16_rows(rng, max(rows, 4))[:rows]
    A, As = _quant(x)
    B, Bs = _e4m3(rng, used.size, N, K), _scales(rng, used.size, X.cdiv(N, GQ), K // GQ)
    C = np.stack([DR.moe_expert_fp8_block_row(A[p // top_k], As[p // top_k], B[umap[ids[p]]], Bs[umap[ids[p]]], G=GQ, KB=KB, swap=swap,
                                              g=int(tw[p]) if mul else None)[0] for p in range(M * X.TOP_K)])
    if flip:
        C[13, 7] ^= 0x8000
    arrays = {"A": A, "As": As, "B": B, "Bs": Bs, "ids": ids, "umap": umap, "tw": tw, "C": C, **({"x": x} if kind == "w13" else {})}
    job = {"phase": "moe", "case": "small", "launch": kind, "K": K, "N": N, "KB": KB, "SWAP": swap, "mul": mul, "top_k": top_k,
           "pairs": M * X.TOP_K, "statics_known": True, "paths": X._save(tmp_path / kind, **arrays)}
    return [{**job, "pair_range": [0, M * X.TOP_K], "reference": True, "quant_rows": list(range(M)) if kind == "w13" else []}]


@pytest.mark.parametrize("kind,KB,swap", [("w13", 32, False), ("w2", 32, False), ("w2", 64, True)])
def test_moe_jobs_take_the_kernel_words(tmp_path, kind: str, KB: int, swap: bool) -> None:
    t = _total([X.compare(j) for j in _moe_job(tmp_path, kind, KB=KB, swap=swap)])
    assert t["words"] == 24 * N and t["mismatch"] == t["unresolved"] == t["twin_mismatch"] == t["quant_mismatch"] == 0
    assert t["reference_coords"] == _sampled_columns((0, 23)) and t["quant_rows"] == t["reference_quant_rows"] * 3 == (3 if kind == "w13" else 0)
    flipped = [X.compare(j) for j in _moe_job(tmp_path, kind, KB=KB, swap=swap, flip=True)]
    assert _total(flipped)["mismatch"] == 1 and flipped[0]["first_mismatch"]["pair"] == 13


def test_a_launch_at_statics_no_program_binds_is_unresolved(tmp_path) -> None:
    (job,) = _moe_job(tmp_path, "w13")
    res = X.compare({**job, "statics_known": False})
    assert res["unresolved"] == res["words"] == 24 * N


def test_the_assignment_check_reads_the_launch_blocks() -> None:
    ids = np.array([5, 9, 5, 2])
    L = {"sorted_token_ids": torch.tensor([3, 4, 4, 4, 0, 2, 4, 4, 1, 4, 4, 4]), "expert_ids": torch.tensor([2, 5, 9]),
         "num_tokens_post_padded": torch.tensor([12]), "config": {"BLOCK_SIZE_M": 4}}
    assert X._assignment_ok(L, ids) is True
    assert X._assignment_ok({**L, "expert_ids": torch.tensor([2, 9, 5])}, ids) is False
    assert X._assignment_ok({**L, "sorted_token_ids": torch.tensor([3, 4, 4, 4, 0, 4, 4, 4, 1, 4, 4, 4])}, ids) is False
    assert X._assignment_ok({**L, "sorted_token_ids": None}, ids) is None


def test_the_ptx_scan_takes_only_the_k32_e4m3_wgmma(tmp_path) -> None:
    good = "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%f1}, %rd1, %rd2, p, 1, 1;\n"
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "fused_moe_kernel.ptx").write_text(good * 2)
    assert X.ptx(tmp_path) == {"files": 1, "wgmma": {good.split(" ")[0]: 2}, "ok": True}
    (tmp_path / "a" / "fused_moe_kernel.ptx").write_text(good + good.replace("m64n64k32.f32.e4m3", "m64n64k16.f32.bf16"))
    assert X.ptx(tmp_path)["ok"] is False
    assert X.ptx(tmp_path / "nothing")["ok"] is False


def test_the_programs_bind_at_every_glm53_shape() -> None:
    got = X.bind_check({(32, False)})
    assert got["ok"] and got["bindings"]["ScaledMmFp8Block_v1{K=6144,N=2624,G=128}"]
    assert len(got["bindings"]) == len(X.DENSE) + len({K for _, K, _ in X.DENSE}) + 2 * len({c[1] for c in X.MOE})


def _doc() -> dict:
    launch = {"deterministic": True, "statics_ok": True, "assignment_ok": True}
    return {"schema": X.SCHEMA, "gate": {"ok": True}, "errors": {}, "selection": {"dense_ok": True, "moe_ok": True}, "ptx": {"ok": True},
            "bind": {"ok": True}, "launches": [{**launch, "phase": "dense", "operands_ok": True}, {**launch, "phase": "moe"}],
            "totals": {"words": 10, "mismatch": 0, "unresolved": 0, "quant_rows": 2, "quant_mismatch": 0, "reference_coords": 3,
                       "reference_quant_rows": 1, "twin_mismatch": 0}}


@pytest.mark.parametrize("path,value", [(("gate", "ok"), False), (("errors", "moe/x"), "boom"), (("selection", "moe_ok"), False),
                                        (("ptx", "ok"), None), (("totals", "mismatch"), 1), (("totals", "unresolved"), 2),
                                        (("totals", "reference_coords"), 0), (("totals", "twin_mismatch"), 1),
                                        (("launches", 1, "statics_ok"), False), (("launches", 1, "assignment_ok"), False),
                                        (("launches", 0, "deterministic"), False), (("launches", 0, "operands_ok"), False)])
def test_the_verdict_needs_every_check(path: tuple, value) -> None:
    assert X._verdict(_doc())
    doc = copy.deepcopy(_doc())
    node = doc
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    assert not X._verdict(doc)
    assert not X._verdict({**_doc(), "launches": [_doc()["launches"][0]]})       # a dense-only run is not a GO


def test_the_record_is_a_registered_property() -> None:
    from verity_vllm.properties.record import seal

    doc = seal(_doc(), X._verdict(_doc()))
    assert REGISTRY["glm53_block_fp8_capture"].schema == X.SCHEMA and doc["ok"] and doc["digest"] == digest_of(doc)
