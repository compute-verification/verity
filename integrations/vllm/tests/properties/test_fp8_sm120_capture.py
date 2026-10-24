"""The sm_120 block-FP8 capture without a GPU: its models against the registered twins and primitives, its comparison jobs over
operands and words made here (the kernels' words stood in for by the models), a flipped word, a quant byte and a competing model's
words found and named, the dispatch check, the verdict, the skip without a cc 12.0 device, and `main` end to end on stubbed launches
(the GPU run is `tests/properties/fp8_sm120_capture_gpu.py --out ...` on one cc 12.0 GPU)."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from tests.properties import fp8_sm120_capture_gpu as X
from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.kernels import fp8_moe_rows as MR
from verity_vllm.properties.record import digest_of

K, N = 256, 200


def _e4m3(rng: np.random.Generator, *shape: int) -> np.ndarray:
    b = rng.integers(0, 256, shape).astype(np.uint8)
    return np.where((b & 0x7F) == 0x7F, b ^ 1, b).astype(np.uint8)


def _weights(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    """`_weights`'s rows on the CPU: random codes, and rows n % 8 == 1 cancelling against activation row 4."""
    w = _e4m3(rng, n, k)
    w[1::8] &= 0x9F
    w[1::8, 0::32] = 0x7E
    w[1::8, 1::32] = 0xFE
    return w


def _scales(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return np.exp2(rng.uniform(-12, -6, shape)).astype(np.float32).view(np.uint32)


def _bf16_rows(rng: np.random.Generator, M: int, k: int = K, G: int = 128) -> np.ndarray:
    """`_activation`'s rows: random, a zero row, one outlier over tiny values, a subnormal scale, the cancellation, saturated and
    paired-group rows."""
    x = rng.standard_normal((M, k)).astype(np.float32) * np.exp2(rng.uniform(-3, 3, (M, 1))).astype(np.float32)
    x[1] = 0
    x[2] *= np.float32(2.0 ** -9)
    x[2, 7] = np.float32(2.0 ** 12)
    x[3] *= np.float32(2.0 ** -130)
    if M >= 7:
        x[4] = rng.standard_normal(k).astype(np.float32) * np.float32(2.0 ** -6)
        x[4, 0::32] = x[4, 1::32] = 1.0
        x[5] = np.where(rng.standard_normal(k) < 0, -1.0, 1.0)
        for gk in range(1, k // G, 2):
            x[6, gk * G:(gk + 1) * G] = -x[6, (gk - 1) * G:gk * G]
    return (x.view(np.uint32) >> 16).astype(np.uint16)


def _quant(x: np.ndarray, G: int) -> tuple[np.ndarray, np.ndarray]:
    q, s = zip(*(DR.fp8_group_quant_row(r, G=G) for r in x))
    return np.stack(q), np.stack(s)


def test_the_hopper_step_models_are_the_registered_twins() -> None:
    """With Hopper's k32 step, `block_row` is `ScaledMmFp8Block_v1`'s twin (form `cutlass`) and `MoeExpertFp8BlockCoordinate(W)_v1`'s
    (forms `triton` and `triton_swap`), which tests/program checks against the Definitions: the scale forms and tiles are theirs."""
    rng = np.random.default_rng(3)
    q, s = _quant(_bf16_rows(rng, 6), 128)
    w = _weights(rng, N, K)
    for G in (64, 128):
        sq, ss = _quant(_bf16_rows(rng, 6), G)
        ws = _scales(rng, X.H.cdiv(N, G), K // G)
        for r in range(6):
            assert np.array_equal(X.block_row(sq[r], ss[r], w, ws, G=G, KB=G, form="cutlass", dot="hopper")[0],
                                  DR.scaled_mm_fp8_block_row(sq[r], ss[r], w, ws, G=G)[0])
        for KB, swap, g in ((32, False, None), (64, True, None), (32, False, int(np.float32(0.3).view(np.uint32)))):
            got = X.block_row(sq[4], ss[4], w, ws, G=G, KB=KB, form="triton_swap" if swap else "triton", dot="hopper", g=g)[0]
            assert np.array_equal(got, MR.moe_expert_fp8_block_row(sq[4], ss[4], w, ws, G=G, KB=KB, swap=swap, g=g)[0])
    with pytest.raises(ValueError):
        X.block_row(q[0], s[0], w, _scales(rng, X.H.cdiv(N, 128), 2), G=128, KB=96)
    with pytest.raises(ValueError):
        X.block_row(q[0], s[0], w, _scales(rng, X.H.cdiv(N, 128), 2), G=128, KB=128, form="fma_twice")


def test_the_vectorised_blackwell_step_is_the_registered_primitive() -> None:
    from verity.ml.circuits.ops import prims

    rng = np.random.default_rng(4)
    n = 96
    acc = (np.exp2(rng.uniform(-20, 20, n)) * rng.choice([-1, 1], n)).astype(np.float32).view(np.uint32)
    acc[:4] = [0, 0x80000000, 1, 0x00800000]
    a, b = _e4m3(rng, n, 32), _e4m3(rng, n, 32)
    got = X._step("blackwell")(acc, a, b)
    want = [prims.BlackwellE4m3QmmaDot32.evaluate(int(acc[i]), *map(int, a[i]), *map(int, b[i])) for i in range(n)]
    assert got.tolist() == want
    assert (X._step("hopper")(acc, a, b) != got).any()
    with pytest.raises(ValueError):
        X._step("ampere")


@pytest.mark.parametrize("form", X.FORMS)
@pytest.mark.parametrize("G,KB,g", [(128, 128, None), (64, 32, None), (128, 64, 0x3E99999A)])
def test_the_reference_chain_is_the_model(form: str, G: int, KB: int, g) -> None:
    rng = np.random.default_rng(5)
    q, s = _quant(_bf16_rows(rng, 6), G)
    w, ws = _weights(rng, 9, K), _scales(rng, 1, K // G)
    for r in (0, 4, 5):
        want = X.block_row(q[r], s[r], w, ws, G=G, KB=KB, form=form, g=g)[0]
        assert [X.reference_word(q[r], s[r], w[n], ws[0], G=G, KB=KB, form=form, g=g) for n in range(9)] == want.tolist()


def _dense_job(tmp_path, *, form: str = "cutlass", words: str = "cutlass", flip: bool = False, flip_quant: bool = False, M: int = 8,
               G: int = 128, KB: int = 128) -> dict:
    rng = np.random.default_rng(1)
    x = _bf16_rows(rng, M, K, G)
    q, s = _quant(x, G)
    w, ws = _weights(rng, N, K), _scales(rng, X.H.cdiv(N, G), K // G)
    X.pair_groups(w, ws, G)
    if flip_quant:
        q[4, 9] ^= 1
    out = np.stack([X.block_row(q[r], s[r], w, ws, G=G, KB=KB, form=words)[0] for r in range(M)])
    if flip:
        out[2, 150] ^= 1
    return {"phase": form, "case": "ragged", "K": K, "N": N, "M": M, "G": G, "KB": KB, "form": form,
            "paths": X.H._save(tmp_path / f"dense_{words}_{flip}_{flip_quant}", x=x, q=q, s=s, w=w, ws=ws, out=out)}


def _total(results: list[dict]) -> dict:
    assert not [r for r in results if "error" in r], results
    t = {k: sum(r[k] for r in results) for k in X.TOTALS}
    t["alternatives"] = {}
    for r in results:
        for name, n in r["alternatives"].items():
            t["alternatives"][name] = t["alternatives"].get(name, 0) + n
    t["first_mismatch"] = next((r["first_mismatch"] for r in results if r["first_mismatch"]), None)
    return t


def _sampled(rows) -> int:
    return sum(len({0, N - 1, (r * 7919) % N}) for r in rows)


@pytest.mark.parametrize("form,KB", [("cutlass", 128), ("triton", 128), ("triton", 64)])
def test_a_dense_job_takes_the_kernel_words_and_refutes_every_alternative(tmp_path, form: str, KB: int) -> None:
    jobs = X.gemm_jobs(_dense_job(tmp_path, form=form, words=form, KB=KB), 8, X.SPECIAL_ROWS)
    assert sorted(r for j in jobs for r in j["reference"]) == [0, 2, 3, 4, 5, 6, 7]
    t = _total([X.compare(j) for j in jobs])
    assert {k: t[k] for k in X.TOTALS} == {"words": 8 * N, "mismatch": 0, "unresolved": 0, "quant_rows": 8, "quant_mismatch": 0,
                                          "reference_coords": _sampled((0, 2, 3, 4, 5, 6, 7)), "reference_quant_rows": 7,
                                          "twin_mismatch": 0, "alt_words": 7 * N}
    assert sorted(t["alternatives"]) == sorted(X.ALTERNATIVES[form]) and all(n > 0 for n in t["alternatives"].values()), t["alternatives"]
    assert t["first_mismatch"] is None


def test_a_flipped_word_a_quant_byte_and_another_models_words_are_found(tmp_path) -> None:
    t = _total([X.compare(j) for j in X.gemm_jobs(_dense_job(tmp_path, flip=True), 8, X.SPECIAL_ROWS)])
    assert t["mismatch"] == 1 and t["first_mismatch"] == {"row": 2, "column": 150, "want": t["first_mismatch"]["want"],
                                                          "kernel": t["first_mismatch"]["want"] ^ 1, "explained_by": []}
    t = _total([X.compare(j) for j in X.gemm_jobs(_dense_job(tmp_path, flip_quant=True), 8, X.SPECIAL_ROWS)])
    assert t["quant_mismatch"] == 1 and t["first_mismatch"] == {"row": 4, "quant": True}
    t = _total([X.compare(j) for j in X.gemm_jobs(_dense_job(tmp_path, words="triton"), 8, X.SPECIAL_ROWS)])
    assert t["mismatch"] > 0 and "triton_order" in t["first_mismatch"]["explained_by"] and t["alternatives"]["triton_order"] == 0


def _moe_job(tmp_path, kind: str, *, flip: bool = False) -> list[dict]:
    """A w13-like launch (8 tokens x 8 slots over 5 used experts, K 256, N 200) or a w2-like one (64 rows, one slot each, the
    routed weight), at G 64 and KB 32, on paired weight groups."""
    rng = np.random.default_rng(2)
    M, used = 8, np.array([3, 17, 40, 41, 127])
    ids = np.stack([rng.permutation(used)[:5].tolist() + rng.permutation(used)[:3].tolist() for _ in range(M)]).reshape(-1)
    umap = np.full(X.NUM_EXPERTS, -1)
    umap[used] = np.arange(used.size)
    tw = (rng.uniform(0.05, 0.4, M * X.TOP_K).astype(np.float32)).view(np.uint32)
    rows = M if kind == "w13" else M * X.TOP_K
    top_k, mul = (X.TOP_K, False) if kind == "w13" else (1, True)
    A, As = _quant(_bf16_rows(rng, rows, K, X.MOE_G), X.MOE_G)
    B, Bs = np.stack([_weights(rng, N, K) for _ in used]), _scales(rng, used.size, X.H.cdiv(N, X.MOE_G), K // X.MOE_G)
    X.pair_groups(B, Bs, X.MOE_G)
    C = np.stack([X.block_row(A[p // top_k], As[p // top_k], B[umap[ids[p]]], Bs[umap[ids[p]]], G=X.MOE_G, KB=32,
                              g=int(tw[p]) if mul else None)[0] for p in range(M * X.TOP_K)])
    if flip:
        C[13, 7] ^= 0x8000
    job = {"phase": "moe", "case": "small", "launch": kind, "K": K, "N": N, "G": X.MOE_G, "KB": 32, "form": "triton", "mul": mul,
           "top_k": top_k, "paths": X.H._save(tmp_path / f"{kind}_{flip}", A=A, As=As, B=B, Bs=Bs, ids=ids, umap=umap, tw=tw, C=C)}
    return X.gemm_jobs(job, M * X.TOP_K, tuple(r * X.TOP_K for r in X.SPECIAL_ROWS))


@pytest.mark.parametrize("kind", ["w13", "w2"])
def test_moe_jobs_take_the_kernel_words(tmp_path, kind: str) -> None:
    t = _total([X.compare(j) for j in _moe_job(tmp_path, kind)])
    assert t["words"] == 64 * N and t["mismatch"] == t["unresolved"] == t["twin_mismatch"] == 0 and t["quant_rows"] == 0
    assert t["reference_coords"] == _sampled([*(r * X.TOP_K for r in X.SPECIAL_ROWS), 63])
    assert sorted(t["alternatives"]) == sorted(X.ALTERNATIVES["triton"])
    if kind == "w13":
        assert all(n > 0 for n in t["alternatives"].values()), t["alternatives"]
    flipped = _total([X.compare(j) for j in _moe_job(tmp_path, kind, flip=True)])
    assert flipped["mismatch"] == 1 and flipped["first_mismatch"]["row"] == 13 and flipped["first_mismatch"]["column"] == 7


def test_a_quant_job_finds_a_flipped_byte(tmp_path) -> None:
    rng = np.random.default_rng(6)
    x = _bf16_rows(rng, 9, X.MOE_I)
    q, s = _quant(x, X.MOE_G)
    job = {"phase": "quant", "case": "K192/G64", "K": X.MOE_I, "G": X.MOE_G, "paths": X.H._save(tmp_path / "q", x=x, q=q, s=s)}
    t = _total([X.compare(j) for j in X.quant_jobs(job, 9)])
    assert (t["quant_rows"], t["quant_mismatch"], t["reference_quant_rows"], t["twin_mismatch"]) == (9, 0, 7, 0)
    q[7, 100] ^= 1
    job["paths"] = X.H._save(tmp_path / "q2", x=x, q=q, s=s)
    t = _total([X.compare(j) for j in X.quant_jobs(job, 9)])
    assert t["quant_mismatch"] == 1 and t["first_mismatch"] == {"row": 7, "quant": True, "column": 100}


def test_the_dense_cases_span_the_three_dispatch_classes() -> None:
    assert [X.dispatch_class(m) for m in (1, 64, 65, 256, 257, 300)] == ["swap_ab", "swap_ab", "pingpong", "pingpong", "cooperative",
                                                                         "cooperative"]
    assert {X.dispatch_class(m) for m in X.DENSE_M} == {X.dispatch_class(m) for m in X.QUICK_DENSE_M} == {"swap_ab", "pingpong",
                                                                                                          "cooperative"}
    assert X.DENSE == (("qkv_proj", 4096, 1280), ("o_proj", 1024, 4096)) and X.MOE_I == 192 and X.NUM_EXPERTS == 128
    assert all(K_ % G == 0 for K_, G in X.QUANT) and {G for _, G in X.QUANT} == {64, 128}


def _cutlass(case: str, M: int, names: list[str]) -> dict:
    return {"phase": "cutlass", "case": case, "M": M, "dispatch": X.dispatch_class(M), "cuda_kernels": names, "deterministic": True}


def test_the_dispatch_check_needs_three_cutlass_kernels() -> None:
    per = {"swap_ab": "cutlass_128x32", "pingpong": "cutlass_64x128_pp", "cooperative": "cutlass_128x128"}
    launches = [_cutlass("qkv_proj", m, [per[X.dispatch_class(m)], "per_token_group_quant"]) for m in X.DENSE_M]
    assert X.dispatch({"launches": launches})["ok"]
    assert not X.dispatch({"launches": [x for x in launches if x["dispatch"] != "pingpong"]})["ok"]
    same = [_cutlass("qkv_proj", m, ["cutlass_one"]) for m in X.DENSE_M]
    assert not X.dispatch({"launches": same})["ok"]
    mixed = launches + [_cutlass("qkv_proj", 2, ["cutlass_other"])]
    assert not X.dispatch({"launches": mixed})["ok"]
    assert not X.dispatch({"launches": [_cutlass("o_proj", 1, ["triton_kernel"])]})["ok"] and not X.dispatch({"launches": []})["ok"]


TOT = {"words": 10, "mismatch": 0, "unresolved": 0, "quant_rows": 2, "quant_mismatch": 0, "reference_coords": 3, "reference_quant_rows": 1,
       "twin_mismatch": 0, "alt_words": 10}
ALL = {"cutlass", "triton", "moe", "quant"}


def _doc() -> dict:
    alts = {p: {n: 1 for n in X.ALTERNATIVES["cutlass" if p == "cutlass" else "triton"]} for p in ("cutlass", "triton", "moe")}
    launch = {"deterministic": True, "operands_ok": True, "statics_ok": True}
    comps = [{"phase": p, "case": "c", "launch": None, "M": 1, **TOT, "first_mismatch": None} for p in ("cutlass", "triton")]
    comps += [{"phase": "moe", "case": "decode", "launch": "w13", "M": None, **TOT, "first_mismatch": None},
              {"phase": "quant", "case": "K192/G64", "launch": None, "M": None, **TOT, "first_mismatch": None}]
    doc = {"schema": X.SCHEMA, "gate": {"ok": True}, "errors": {}, "dispatch": {"ok": True}, "alternatives": alts,
           "selection": {"cutlass_ok": True, "triton_ok": True, "moe_ok": True},
           "launches": [{**launch, "phase": "cutlass", "case": "c", "M": 1}, {**launch, "phase": "triton", "case": "c", "M": 1},
                        {**launch, "phase": "moe", "case": "decode", "launch": "w13", "assignment_ok": True},
                        {**launch, "phase": "quant", "case": "K192/G64", "layouts_agree": True}],
           "comparisons": comps}
    doc["cases"] = X.cases(doc)
    return doc


@pytest.mark.parametrize("path,value", [(("gate", "ok"), False), (("errors", "moe/decode"), "boom"), (("selection", "triton_ok"), False),
                                        (("dispatch", "ok"), False), (("alternatives", "moe", "unfused"), 0),
                                        (("alternatives", "cutlass", "hopper_step"), 0), (("comparisons", 0, "mismatch"), 1),
                                        (("comparisons", 1, "unresolved"), 1), (("comparisons", 2, "twin_mismatch"), 1),
                                        (("comparisons", 3, "quant_mismatch"), 1), (("comparisons", 0, "reference_coords"), 0),
                                        (("launches", 2, "statics_ok"), False), (("launches", 2, "assignment_ok"), False),
                                        (("launches", 3, "layouts_agree"), False), (("launches", 0, "deterministic"), False),
                                        (("launches", 1, "operands_ok"), False)])
def test_the_verdict_needs_every_check(path: tuple, value) -> None:
    assert X.verdict(_doc(), ALL)
    doc = copy.deepcopy(_doc())
    node = doc
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    doc["cases"] = X.cases(doc)
    assert not X.verdict(doc, ALL)
    assert not X.verdict(_doc(), ALL | {"other"}) and X.verdict({**_doc(), "launches": _doc()["launches"][:2]}, {"cutlass", "triton"})


def test_each_case_has_its_own_verdict_line() -> None:
    doc = _doc()
    doc["comparisons"][2] = {**doc["comparisons"][2], "mismatch": 2, "first_mismatch": {"row": 13, "column": 7}}
    doc["errors"]["quant/K4096/G64"] = "boom"
    got = X.cases(doc)
    assert {k: c["ok"] for k, c in got.items()} == {"cutlass/c/M1": True, "triton/c/M1": True, "moe/decode": False,
                                                     "quant/K192/G64": True, "quant/K4096/G64": False}
    line = X.verdict_line("moe/decode", got["moe/decode"])
    assert line["verdict"] == "NO-GO" and line["mismatch"] == 2 and line["first_mismatch"] == {"row": 13, "column": 7}


def test_main_skips_without_a_cc120_device(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(X, "device_present", lambda: False)
    with pytest.raises(SystemExit) as e:
        X.main(["--out", str(tmp_path)])
    assert e.value.code == 3
    (line,) = (tmp_path / X.VERDICTS).read_text().splitlines()
    assert json.loads(line) == {"case": "gate", "verdict": "skip", "reason": "no cc 12.0 CUDA device"}
    doc = json.loads((tmp_path / X.RECORD).read_text())
    assert doc["ok"] is False and doc["digest"] == digest_of(doc) and "FP8-SM120 SKIP" in capsys.readouterr().out


def test_main_end_to_end_on_stubbed_launches(tmp_path, monkeypatch, capsys) -> None:
    """`main` with the GPU's pieces stood in for (gate, selection, the launches as model words made here): the comparison jobs
    through the spawned worker pool, the per-case verdict lines, the alternatives, the dispatch check, the sealed record and the exit
    status, so a slip in the orchestration shows here and not on the GPU."""
    for k in ("VLLM_USE_DEEP_GEMM", "VLLM_BATCH_INVARIANT", "VLLM_ENABLE_V1_MULTIPROCESSING", "TRITON_CACHE_DIR"):
        monkeypatch.setenv(k, "")
    monkeypatch.setattr(X, "device_present", lambda: True)
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": True, "checks": {"sms": False, "cc": True}})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    monkeypatch.setattr(X, "selection", lambda: {"cutlass_ok": True, "triton_ok": True, "moe_ok": True})
    monkeypatch.setattr(X, "DENSE", (("ragged", K, N),))
    monkeypatch.setattr(X, "QUICK_DENSE_M", (8,))
    monkeypatch.setattr(X, "MOE", (("small", 8, 0),))
    monkeypatch.setattr(X, "QUICK_MOE", ("small",))
    monkeypatch.setattr(X, "QUANT", ((X.MOE_I, X.MOE_G),))
    kernels = iter([["cutlass_swap_ab"]])

    def dense(work, phase, name, K_, N_, M, seed):
        job = _dense_job(tmp_path / phase, form=phase, words=phase, flip=phase == "triton")
        launch = {"phase": phase, "case": name, "K": K_, "N": N_, "M": M, "kernel": X.EXPECTED_KERNEL[phase][1],
                  "dispatch": X.dispatch_class(M), "KB": 128,
                  "cuda_kernels": next(kernels) if phase == "cutlass" else ["triton"], "deterministic": True, "operands_ok": True,
                  "statics_ok": True}
        return launch, {**job, "case": name, "M": M}

    def moe(work, name, M, hot, seed):
        launches = [{"phase": "moe", "case": name, "launch": k, "M": M, "deterministic": True, "statics_ok": True, "operands_ok": True,
                     "assignment_ok": True, "K": K, "N": N, "config": {}, "KB": 32, "SWAP": False, "quant_calls": {}} for k in ("w13", "w2")]
        x = _bf16_rows(np.random.default_rng(8), M, K, X.MOE_G)
        q, s = _quant(x, X.MOE_G)
        qjob = {"phase": "moe", "case": name, "launch": "w13_input", "K": K, "G": X.MOE_G,
                "paths": X.H._save(tmp_path / "mq", x=x, q=q, s=s)}
        return launches, [*({**j, "case": name} for k in ("w13", "w2") for j in _moe_job(tmp_path / "m", k)), *X.quant_jobs(qjob, M)]

    def quant(work, K_, Gq, seed):
        rng = np.random.default_rng(7)
        x = _bf16_rows(rng, 6, K_)
        q, s = _quant(x, Gq)
        job = {"phase": "quant", "case": f"K{K_}/G{Gq}", "K": K_, "G": Gq, "paths": X.H._save(tmp_path / "qq", x=x, q=q, s=s)}
        return {"phase": "quant", "case": job["case"], "deterministic": True, "operands_ok": True, "layouts_agree": True}, X.quant_jobs(job, 6)

    monkeypatch.setattr(X, "dense_launch", dense)
    monkeypatch.setattr(X, "moe_launches", moe)
    monkeypatch.setattr(X, "quant_launch", quant)
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1", "--quick"]) == 1
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is False and not doc["errors"] and doc["gate"]["ok"]
    assert {k: c["ok"] for k, c in doc["cases"].items()} == {"cutlass/ragged/M8": True, "triton/ragged/M8": False, "moe/small": True,
                                                             "quant/K192/G64": True}
    assert doc["dispatch"]["ok"] is False                                        # one dispatch class ran: not the three
    assert doc["env"]["VLLM_USE_DEEP_GEMM"] == "0" and doc["env"]["VLLM_BATCH_INVARIANT"] == "1"
    lines = [json.loads(s) for s in (out / X.VERDICTS).read_text().splitlines()]
    assert [ln["case"] for ln in lines] == ["gate", "selection", *doc["cases"], "dispatch", "alternatives"]
    assert next(ln for ln in lines if ln["case"] == "triton/ragged/M8")["first_mismatch"]["column"] == 150
    assert "FP8-SM120 NO-GO" in capsys.readouterr().out
