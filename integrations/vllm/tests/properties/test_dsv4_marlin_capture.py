"""The DeepSeek-V4-Pro Marlin capture without a GPU: its jobs over launches made here (the kernel's words stood in for by the twin at
a known config), the kernel's config read off its profiled name and grid, the probe telling summation orders apart where ordinary
activations cannot, a flipped word found and located, the reference against the twin, the sweep and its model, the assignment and
statics checks, the records of a case's launches, the verdict, and `main` end to end (the GPU run is
`tests/properties/dsv4_marlin_capture_gpu.py --out ...` on an H200)."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from tests.properties import dsv4_marlin_capture_gpu as X
from verity_vllm.program.kernels import marlin_moe_rows as MR
from verity_vllm.properties.record import REGISTRY, digest_of

E, K, N, SMS = 8, 256, 128, 3
#: the config the stand-in kernel runs: TK 64 (two warp rows), TN 128, 4 blocks per SM, so GRID 12 splits K
TRUE = (64, 128, 4)
TYPES = [5, 1095, 5, 2]


def _bf16_rows(rng: np.random.Generator, M: int, K_: int) -> np.ndarray:
    x = rng.standard_normal((M, K_)).astype(np.float32) * np.exp2(rng.uniform(-3, 3, (M, 1))).astype(np.float32)
    return (x.view(np.uint32) >> 16).astype(np.uint16)


def _align(ids_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """`moe_align_block_size` as the kernel reads it: each expert's pairs in order, padded to a block of 8 with the pair count."""
    st, ex = [], []
    for e in range(E):
        p = np.flatnonzero(ids_flat == e).tolist()
        for b in range(0, len(p), X.MOE_BLOCK):
            blk = p[b:b + X.MOE_BLOCK]
            st += blk + [ids_flat.size] * (X.MOE_BLOCK - len(blk))
            ex.append(e)
    return np.array(st), np.array(ex), len(st)


def _weights(tmp_path) -> dict[str, str]:
    rng = np.random.default_rng(7)
    wq = rng.integers(0, 16, (E, N, K)).astype(np.uint8)
    ws = rng.integers(118, 128, (E, N, K // X.GROUP)).astype(np.uint8)
    edge = rng.random((N, K // X.GROUP)) < 0.3
    ws[0] = np.where(edge, rng.choice(X.EDGE_BYTES, (N, K // X.GROUP)), ws[0])
    return X._save(tmp_path / "weights", wq=wq, ws=ws)


def _launch(tmp_path, kind: str, at=TRUE, *, M: int = 2, flip: tuple[int, int] | None = None, phase: str = "auto",
            probe: bool = False) -> dict:
    """A w13-like launch (M tokens x 6 slots, rows x[p // 6]) or a w2-like one (6M rows, one slot each, the routed weight), its
    words the twin's at config ``at``, on ordinary rows or the probe's; expert 0 carries edge scale bytes, so some words are NaN or
    inf."""
    rng = np.random.default_rng(11 + M)
    ids = np.stack([rng.permutation(E)[:X.TOP_K] for _ in range(M)]).reshape(-1)
    st, ex, ntpp = _align(ids)
    ok, blk, e, par = X.assignment(st, ex, ntpp, ids)
    assert ok
    top_k = X.TOP_K if kind == "w13" else 1
    P = M * X.TOP_K
    w = X._load(paths := _weights(tmp_path))
    if probe:
        x = X.probe(np.random.default_rng(3), [e[r * top_k:(r + 1) * top_k] for r in range(P // top_k)], w["wq"], w["ws"])[0]
    else:
        x = _bf16_rows(rng, P // top_k, K)
    g = (rng.uniform(0.05, 1, P).astype(np.float32)).view(np.uint32) if kind == "w2" else None
    C = MR.launch(x[np.arange(P) // top_k], blk, par, e, w["wq"], w["ws"], np.arange(N), TK=at[0], TN=at[1], GRID=SMS * at[2], g=g)
    if flip:
        C[flip] ^= 0x0001
    arrays = {"x": x, "C": C, "blk": blk, "e": e, **({"g": g} if g is not None else {})}
    spec = {"phase": phase, "case": "decode2", "launch": kind, "config": None if phase == "auto" else list(at), "M": M, "K": K, "N": N,
            "E": E, "top_k": top_k, "pairs": P, "par": par, "weights": paths,
            "paths": X._save(tmp_path / f"{phase}_{kind}_{'_'.join(map(str, at))}_{int(probe)}", **arrays)}
    if phase == "forced":
        spec["statics"] = [at[0], at[1], SMS * at[2]]
    return spec


def _total(results: list[dict]) -> dict:
    assert not [r for r in results if "error" in r], results
    return {k: sum(r.get(k, 0) for r in results) for k in X.TOTALS}


def _name(threads: int, n_blocks: int, k_blocks: int, *, m8: str = "true", stages: int = 4, group_blocks: int = 2) -> str:
    return (f"void marlin_moe_wna16::Marlin<(vllm::ScalarTypeId)5, 1095l, 5l, 2l, {threads}, 1, {n_blocks}, {k_blocks}, {m8}, "
            f"{stages}, {group_blocks}, false>(int4 const*, int4 const*, int4*, int4*, int4 const*, float const*, int4 const*, "
            "float const*, int4 const*, int const*, int const*, int const*, int const*, float const*, int, bool, int, int, int, int, "
            "int*, bool, bool, bool)")


def _kernel_event(ts: int, name: str, grid, block) -> dict:
    return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": 5,
            "args": {"correlation": ts, "grid": list(grid), "block": list(block), "registers per thread": 168, "shared memory": 0}}


def test_the_kernel_name_and_grid_read_the_launch_config() -> None:
    trace = {"traceEvents": [_kernel_event(30, _name(256, 8, 8), [SMS, 1, 1], [256, 1, 1]),
                             _kernel_event(20, "void vllm::moe::moe_align_block_size_kernel<int>(...)", [1, 1, 1], [1024, 1, 1]),
                             _kernel_event(10, _name(128, 8, 4), [4 * SMS, 1, 1], [128, 1, 1]),
                             {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 9}]}
    got = X.marlin_kernels(trace)
    assert [k["template"]["thread_k_blocks"] for k in got] == [4, 8], "launch order, Marlin kernels only"
    assert got[0]["template"] == {"a_type": 5, "b_type": 1095, "c_type": 5, "s_type": 2, "threads": 128, "thread_m_blocks": 1,
                                  "thread_n_blocks": 8, "thread_k_blocks": 4, "m_block_size_8": True, "stages": 4, "group_blocks": 2,
                                  "is_zp_float": False}
    assert X.kernel_config(got[0], SMS, TYPES) == ([64, 128, 4], True)
    assert X.kernel_config(got[1], SMS, TYPES) == ([128, 128, 1], True)
    assert X.kernel_config(got[1], SMS, None) == ([128, 128, 1], True)
    assert not X.kernel_config(got[1], SMS, [5, 1095, 5, 3])[1], "the scale type"
    bad = [({"grid": [4 * SMS + 1, 1, 1]}, None), ({"grid": [5 * SMS, 1, 1]}, None), ({"grid": [SMS, 2, 1]}, None),
           ({"block": [256, 1, 1]}, None), (None, _name(128, 8, 4, stages=2)), (None, _name(128, 8, 4, group_blocks=4)),
           (None, _name(128, 8, 4, m8="false")), (None, _name(256, 8, 4)), (None, _name(128, 4, 4))]
    for args, name in bad:
        k = copy.deepcopy(got[0])
        k.update(args or {})
        if name:
            k["template"] = X.kernel_template(name)
        assert not X.kernel_config(k, SMS, TYPES)[1], (args, name)
    assert X.kernel_template("void other::Marlin<1, 2>(int)") is None
    assert X.kernel_template(_name(128, 8, 4).replace("1095l", "x")) is None
    assert X.kernel_template(_name(128, 8, 4).replace(", false>", ", false, 7>")) is None
    assert X.kernel_config(None, SMS, TYPES) == (None, False) and X.kernel_config({"template": None}, SMS, TYPES) == (None, False)


def test_the_probe_tells_summation_orders_apart(tmp_path) -> None:
    """Rounded once to bf16, fp32 sums in different orders give the same words on ordinary rows; on the probe's they do not."""
    w = X._load(_weights(tmp_path))
    rng = np.random.default_rng(13)
    M = 2
    ids = np.stack([rng.permutation(np.arange(1, E))[:X.TOP_K] for _ in range(M)]).reshape(-1)
    st, ex, ntpp = _align(ids)
    _, blk, e, par = X.assignment(st, ex, ntpp, ids)
    rows = [e[r * X.TOP_K:(r + 1) * X.TOP_K] for r in range(M)]
    xp, shifts = X.probe(np.random.default_rng(1), rows, w["wq"], w["ws"])
    assert xp.shape == (M, K) and xp.dtype == np.uint16 and np.all((xp & 0x7F80) != 0x7F80) and np.all((xp & 0x7F80) != 0)
    vals = (xp.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    for r, es in enumerate(rows):
        sums = np.abs(np.concatenate([X.dequant64(w["wq"][q], w["ws"][q]) @ vals[r] for q in es]))
        assert abs(np.log2(np.median(sums)) - X.PROBE_LOG2) <= 0.5 + 1e-9, (r, shifts)
    plain = _bf16_rows(rng, M, K)
    classes = {}
    for name, x in (("probe", xp), ("plain", plain)):
        got: dict[bytes, list] = {}
        for c in X.CANDIDATES:
            C = MR.launch(x[np.arange(M * X.TOP_K) // X.TOP_K], blk, par, e, w["wq"], w["ws"], np.arange(N), TK=c[0], TN=c[1],
                          GRID=SMS * c[2])
            got.setdefault(C.tobytes(), []).append(c)
        classes[name] = list(got.values())
    assert len(classes["plain"]) == 1 and len(classes["probe"]) >= 6, classes["probe"]
    assert [TRUE] in classes["probe"], "the stand-in's config has words of its own"


def test_dequant64_reads_e8m0_as_the_bf16_view() -> None:
    q = np.array([[1, 7, 9, 0] * 8])
    s = np.array([[127], [0], [255]])
    got = X.dequant64(np.repeat(q, 3, axis=0), s)
    assert got[0, :4].tolist() == [0.5, 6.0, -0.5, 0.0] and np.all(got[1] == 0)
    assert np.isinf(got[2, 0]) and np.isnan(got[2, 3])


@pytest.mark.parametrize("kind", ["w13", "w2"])
def test_word_and_reference_jobs_take_the_kernel_words(tmp_path, kind: str) -> None:
    spec = {**_launch(tmp_path, kind), "config": list(TRUE)}
    statics = [TRUE[0], TRUE[1], SMS * TRUE[2]]
    t = _total([X.compare(j) for j in X.word_jobs(spec, statics) + [X.reference_job(spec, statics)]])
    assert t["words"] == spec["pairs"] * N and t["mismatch"] == t["twin_mismatch"] == 0
    P = spec["pairs"]
    assert t["reference_schedules"] > 0 and t["reference_coords"] == sum(len({0, N - 1, p * 7919 % N}) for p in {0, P - 1})
    assert t["reference_expert_coords"] == (1 if kind == "w2" else 0)
    flipped = {**_launch(tmp_path, kind, flip=(7, 100)), "config": list(TRUE)}
    res = [X.compare(j) for j in X.word_jobs(flipped, statics)]
    assert _total(res)["mismatch"] == 1 and next(r["first_mismatch"] for r in res if r["first_mismatch"])["pair"] == 7
    probed = {**_launch(tmp_path, kind, probe=True), "config": list(TRUE)}
    assert _total([X.compare(j) for j in X.word_jobs(probed, statics)])["mismatch"] == 0
    other = [X.compare(j) for j in X.word_jobs(probed, [128, 128, SMS])]
    assert _total(other)["mismatch"] > 0, "on the probe the words read the config they were made at"


def test_the_jobs_split_a_large_launch() -> None:
    jobs = X.word_jobs({"pairs": 96, "N": X.HIDDEN, "K": X.INTER}, [128, 64, 264])
    assert jobs[0]["pair_range"][0] == 0 and jobs[-1]["pair_range"][1] == 96 and len(jobs) > 1
    assert all(a["pair_range"][1] == b["pair_range"][0] for a, b in zip(jobs, jobs[1:]))


def _served(allow: dict, sms: int = 132) -> dict:
    return {kind: {str(M): list(X._auto(M, kind, sms, allow)) for M in X.SERVED_M} for kind in ("w13", "w2")}


@pytest.mark.parametrize("allow", [{(128, 128): 2, (64, 128): 4, (128, 64): 4}, {(128, 128): 1, (64, 128): 1, (128, 64): 3},
                                   {(64, 128): 2, (128, 64): 2}])
def test_the_model_finds_the_limits_behind_the_sweep(allow: dict) -> None:
    got = X.model(_served(allow), 132)
    for kind in ("w13", "w2"):
        assert got[kind]["observed"] == len(X.SERVED_M) and got[kind]["consistent"] >= 1, got[kind]
    bad = _served(allow)
    bad["w2"]["1"] = [128, 64, 1] if bad["w2"]["1"] != [128, 64, 1] else [64, 128, 1]
    assert X.model(bad, 132)["w2"]["consistent"] < got["w2"]["consistent"] or X.model(bad, 132)["w2"]["consistent"] == 0


def _served_launch(kind: str, M: int) -> dict:
    K_, N_ = X.shape(kind)
    return {"args": {"moe_block_size": 8, "top_k": X.launch_top_k(kind), "mul_topk_weights": kind == "w2", "size_m": X.prob_m(kind, M),
                     "size_n": N_, "size_k": K_, "is_k_full": True, "use_atomic_add": False, "use_fp32_reduce": True, "is_zp_float": False,
                     "b_q_type": "float4_e2m1f", "thread_k": -1, "thread_n": -1, "blocks_per_sm": -1},
            "absent": sorted(X.ABSENT), "dtypes": {"input": "torch.bfloat16", "b_scales": "torch.float8_e8m0fnu",
                                                   "topk_weights": "torch.float32", "b_qweight": "torch.int32", "output": "torch.bfloat16"}}


def _kernel(config, sms: int = 132) -> dict:
    tk, tn, b = config
    threads = next(t for k, n, t in MR.SMALL_BATCH_CONFIGS if (k, n) == (tk, tn))
    return {"template": X.kernel_template(_name(threads, tn // 16, tk // 16)), "grid": [sms * b, 1, 1], "block": [threads, 1, 1],
            "registers": 168, "shared": 0, "name": "Marlin"}


def test_the_sweep_names_a_served_config_at_every_batch() -> None:
    allow = {(128, 128): 2, (64, 128): 4, (128, 64): 4}
    served = _served(allow)
    launches = [{**_served_launch(kind, M), "label": f"sweep/{M}", "kernel": _kernel(served[kind][str(M)])}
                for M in X.SERVED_M for kind in ("w13", "w2")]
    got = X.sweep_record(launches, 132, TYPES)
    assert got["ok"] and got["statics_ok"] and got["served"] == served and got["model"]["w2"]["consistent"] >= 1
    assert X.served_config(got, "w13", 16) == served["w13"]["16"] and X.served_config({}, "w13", 1) is None
    assert not X.sweep_record(launches[:-1], 132, TYPES)["ok"], "a batch the sweep did not reach"
    broken = copy.deepcopy(launches)
    broken[3]["kernel"]["block"] = [64, 1, 1]
    assert not X.sweep_record(broken, 132, TYPES)["ok"] and X.sweep_record(broken, 132, TYPES)["served"]["w2"]["2"] is None
    broken = copy.deepcopy(launches)
    broken[0]["args"]["use_atomic_add"] = True
    assert not X.sweep_record(broken, 132, TYPES)["ok"]


def _tensor(a, dtype=torch.int32) -> torch.Tensor:
    return torch.tensor(np.asarray(a), dtype=dtype)


def test_a_case_s_records_read_its_two_runs(tmp_path) -> None:
    """`case_launches` over intercepted launches as the Interceptor leaves them (CPU tensors here): the assignment, the statics, the
    kernel's config in both runs, determinism, and the arrays the jobs read."""
    M = 2
    rng = np.random.default_rng(5)
    ids = np.stack([rng.permutation(np.arange(8, 40))[:X.TOP_K] for _ in range(M)]).reshape(-1)
    st, ex = [], []
    for e in sorted(set(ids.tolist())):
        p = np.flatnonzero(ids == e).tolist()
        st += p + [ids.size] * (X.MOE_BLOCK - len(p))
        ex.append(e)
    launches = []
    for run in ("run1", "run2"):
        for kind in ("w13", "w2"):
            K_, N_ = X.shape(kind)
            rows = M if kind == "w13" else M * X.TOP_K
            out = torch.zeros(M * X.TOP_K, N_, dtype=torch.bfloat16)
            if run == "run2" and kind == "w2":
                out[0, 0] = 1.0
            launches.append({**_served_launch(kind, M), "label": f"decode2/{run}", "output": out,
                             "input": torch.zeros(rows, K_, dtype=torch.bfloat16), "sorted_token_ids": _tensor(st),
                             "expert_ids": _tensor(ex), "num_tokens_past_padded": _tensor([len(st)]),
                             "topk_weights": torch.full((M, X.TOP_K), 0.5), "workspace": 0,
                             "kernel": _kernel((128, 64, 2) if kind == "w13" else (64, 128, 1 if run == "run1" else 2))})
    W = {"paths": {"w13": {"wq": "-", "ws": "-"}, "w2": {"wq": "-", "ws": "-"}}}
    records, specs, captured = X.case_launches(tmp_path, "decode2", M, "uniform", _tensor(ids.reshape(M, X.TOP_K)), launches, W, 132,
                                               TYPES)
    w13, w2 = records
    assert w13["assignment_ok"] and w13["statics_ok"] and w13["deterministic"] and w13["kernel_ok"] and w13["config_deterministic"]
    assert w13["config"] == [128, 64, 2] and w13["par"] == len(ex) and w13["experts_used"] == len(ex)
    assert not w2["deterministic"] and not w2["config_deterministic"] and w2["config"] == [64, 128, 1]
    assert [s["launch"] for s in specs] == ["w13", "w2"] and [c["launch"] for c in captured] == ["w13", "w2"]
    arrays = X._load(specs[1]["paths"])
    assert arrays["x"].shape == (M * X.TOP_K, X.INTER) and arrays["g"].tolist() == [0x3F000000] * (M * X.TOP_K)
    assert arrays["blk"].tolist() == w2_blocks(st, ids.size)
    with pytest.raises(RuntimeError):
        X.case_launches(tmp_path, "decode2", M, "uniform", _tensor(ids), launches[:3], W, 132, TYPES)


def w2_blocks(st: list[int], n: int) -> list[int]:
    blk = [0] * n
    for i, p in enumerate(st):
        if p < n:
            blk[p] = i // X.MOE_BLOCK
    return blk


def test_the_assignment_reads_the_moe_blocks() -> None:
    ids = np.array([5, 2, 5, 7, 1, 0])
    st, ex, ntpp = np.array([5, 6, 6, 6, 6, 6, 6, 6, 4, 6, 6, 6, 6, 6, 6, 6, 1, 6, 6, 6, 6, 6, 6, 6, 0, 2, 6, 6, 6, 6, 6, 6,
                             3, 6, 6, 6, 6, 6, 6, 6]), np.array([0, 1, 2, 5, 7]), 40
    ok, blk, e, par = X.assignment(st, ex, ntpp, ids)
    assert ok and par == 5 and blk.tolist() == [3, 2, 3, 4, 1, 0] and e.tolist() == ids.tolist()
    assert not X.assignment(st, np.array([0, 1, 2, 7, 5]), ntpp, ids)[0]
    lost = st.copy()
    lost[24] = 6
    assert not X.assignment(lost, ex, ntpp, ids)[0]
    assert not X.assignment(st, ex, 32, ids)[0], "a block past num_tokens_post_padded is not run"


@pytest.mark.parametrize("kind", ["w13", "w2"])
def test_the_statics_are_the_served_paths(kind: str) -> None:
    assert X.statics_ok(_served_launch(kind, 3), kind, 3)
    assert not X.statics_ok(_served_launch(kind, 3), kind, 4)
    for k, v in (("use_atomic_add", True), ("use_fp32_reduce", False), ("moe_block_size", 16), ("thread_k", 128),
                 ("mul_topk_weights", kind == "w13"), ("b_q_type", "float8_e4m3fn")):
        L = _served_launch(kind, 3)
        L["args"][k] = v
        assert not X.statics_ok(L, kind, 3), k
    L = _served_launch(kind, 3)
    L["absent"].remove("global_scale")
    assert not X.statics_ok(L, kind, 3)
    L = _served_launch(kind, 3)
    L["dtypes"]["b_scales"] = "torch.float8_e4m3fn"
    assert not X.statics_ok(L, kind, 3)


def test_the_programs_bind_at_the_served_configs() -> None:
    got = X.bind_check({("w13", 128, 64, 264), ("w2", 64, 128, 528)})
    assert got["ok"] and len(got["bindings"]) == 2
    assert got["bindings"]["MarlinMoeExpertGemm_v1{E=384,K=7168,N=768,TK=128,TN=64,GRID=264,DOT={\"fn\":\"HopperBF16WgmmaDot16_v1\"}}"] == {
        "ok": True, "boolean": "MarlinMoeExpertGemm_v2{E=384,K=7168,N=768,TK=128,TN=64,GRID=264,DOT={\"fn\":\"HopperBF16WgmmaDot16_v2\"}}"}
    assert not X.bind_check(set())["ok"]


TOTALS = {"words": 10, "mismatch": 0, "reference_schedules": 2, "reference_coords": 6, "reference_expert_coords": 1, "twin_mismatch": 0}


def _doc() -> dict:
    auto = {"phase": "auto", "case": "decode1", "deterministic": True, "statics_ok": True, "assignment_ok": True, "kernel_ok": True,
            "config_deterministic": True, "identified": True, "config": [128, 64, 2], "served": [128, 64, 2]}
    forced = {"phase": "forced", "case": "hot16", "config": [64, 128, 1]}
    probe = {"case": "hot16", "accepted": 2, "classes": [[[64, 128, 1]], [[128, 64, 1]]], "ok": True}
    return {"schema": X.SCHEMA, "gate": {"ok": True}, "errors": {}, "selection": {"ok": True}, "bind": {"ok": True}, "forced_phase": True,
            "sweep": {"ok": True}, "probe": {"hot16/w13": {**probe, "launch": "w13"}, "hot16/w2": {**probe, "launch": "w2"}},
            "launches": [{**auto, "launch": "w13"}, {**auto, "launch": "w2"}, {**forced, "launch": "w13", "accepted": True},
                         {**forced, "launch": "w2", "accepted": True}, {**forced, "launch": "w2", "accepted": False}],
            "comparisons": [{"phase": "auto", "case": "decode1", "launch": "w13", "config": [128, 64, 2], **TOTALS},
                            {"phase": "forced", "case": "hot16", "launch": "w2", "config": [64, 128, 1], **TOTALS}],
            "totals": TOTALS}


@pytest.mark.parametrize("path,value", [(("gate", "ok"), False), (("errors", "case/decode1"), "boom"), (("selection", "ok"), False),
                                        (("bind", "ok"), False), (("sweep", "ok"), False), (("totals", "mismatch"), 1),
                                        (("totals", "words"), 0), (("totals", "twin_mismatch"), 1), (("totals", "reference_coords"), 0),
                                        (("totals", "reference_schedules"), 0), (("totals", "reference_expert_coords"), 0),
                                        (("launches", 0, "deterministic"), False), (("launches", 1, "statics_ok"), False),
                                        (("launches", 0, "assignment_ok"), False), (("launches", 1, "identified"), False),
                                        (("launches", 0, "kernel_ok"), False), (("launches", 1, "config_deterministic"), False),
                                        (("launches", 3, "accepted"), False), (("probe", "hot16/w2", "ok"), False)])
def test_the_verdict_needs_every_check(path: tuple, value) -> None:
    assert X._verdict(_doc())
    doc = copy.deepcopy(_doc())
    node = doc
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    assert not X._verdict(doc)
    assert not X._verdict({**_doc(), "sweep": {}})
    assert not X._verdict({**_doc(), "launches": _doc()["launches"][:1] + _doc()["launches"][2:]}), "both launch kinds"
    assert not X._verdict({**_doc(), "probe": {"hot16/w13": _doc()["probe"]["hot16/w13"]}}), "a forced launch without its probe"
    assert X._verdict({**_doc(), "forced_phase": False, "launches": _doc()["launches"][:2], "probe": {}})


def test_each_case_has_its_own_verdict() -> None:
    doc = _doc()
    got = X.cases(doc)
    assert {k: c["ok"] for k, c in got.items()} == {"decode1": True, "hot16": False}, "hot16 ran no auto launch here"
    assert got["decode1"]["configs"] == ["w13:128/64/2", "w2:128/64/2"] and got["hot16"]["refused"] == 1
    doc["comparisons"][0]["mismatch"] = 1
    doc["errors"]["case/edge8"] = "boom"
    got = X.cases(doc)
    assert {k: c["ok"] for k, c in got.items()} == {"decode1": False, "hot16": False, "edge8": False}
    doc = _doc()
    doc["errors"]["compare/decode1/w2/reference/[128, 64, 2]"] = "boom"
    assert X.cases(doc)["decode1"]["ok"] is False
    doc = _doc()
    doc["probe"]["hot16/w2"]["ok"] = False
    assert X.cases(doc)["hot16"]["launches_ok"] is False


@pytest.mark.parametrize("flip", [False, True])
def test_main_end_to_end_on_stubbed_launches(tmp_path, monkeypatch, capsys, flip: bool) -> None:
    """`main` with the GPU phases stood in for (the gate, a sweep serving TRUE, launches whose words the twin made at TRUE, a forced
    w2 launch at another config with one word flipped or not) and vLLM's config module stubbed: the comparisons through the spawned
    worker pool, identification against the sweep, the bindings, the per-case verdicts, the sealed record and the exit status, so a
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
    for k in ("VLLM_BATCH_INVARIANT", "VLLM_ENABLE_V1_MULTIPROCESSING"):
        monkeypatch.setenv(k, "")
    monkeypatch.setenv("VLLM_MARLIN_INPUT_DTYPE", "fp8")
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": True, "sms": SMS})
    monkeypatch.setattr(X.G, "_environment", dict)
    forced_at = (128, 64, 2)

    def gpu(doc, work, sms, quick, forced):
        assert sms == SMS and forced and not quick
        doc["selection"] = {"ok": True}
        doc["sweep"] = {"ok": True, "served": {k: {str(M): list(TRUE) for M in X.SERVED_M} for k in ("w13", "w2")}}
        specs = [{**_launch(tmp_path / "a", kind, M=1), "case": "decode1"} for kind in ("w13", "w2")]
        doc["launches"] += [{"phase": "auto", "case": "decode1", "launch": s["launch"], "M": 1, "pairs": s["pairs"], "par": s["par"],
                             "deterministic": True, "statics_ok": True, "assignment_ok": True, "kernel_ok": True,
                             "config_deterministic": True, "config": list(TRUE)} for s in specs]
        fs = {**_launch(tmp_path / "f", "w2", forced_at, M=1, flip=(3, 5) if flip else None, phase="forced", probe=True),
              "case": "decode1"}
        doc["launches"] += [{"phase": "forced", "case": "decode1", "launch": k, "config": list(forced_at), "accepted": True}
                            for k in ("w13", "w2")]
        doc["probe"] = {f"decode1/{k}": {"case": "decode1", "launch": k, "accepted": 1, "classes": [[list(forced_at)]], "ok": True}
                        for k in ("w13", "w2")}
        return specs, [fs]

    monkeypatch.setattr(X, "gpu_phases", gpu)
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1"]) == (1 if flip else 0)
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is (not flip) and not doc["errors"]
    assert doc["env"]["VLLM_BATCH_INVARIANT"] == "0" and doc["env"]["unset"] == {"VLLM_MARLIN_INPUT_DTYPE": "fp8"}
    assert all(x["identified"] and x["served"] == list(TRUE) for x in doc["launches"][:2])
    assert doc["bind"]["ok"] and len(doc["bind"]["bindings"]) == 2
    assert doc["totals"]["mismatch"] == (1 if flip else 0) and doc["totals"]["reference_expert_coords"] == 1
    assert doc["totals"]["words"] == 3 * 6 * N
    assert doc["cases"]["decode1"]["ok"] is (not flip)
    printed = capsys.readouterr().out
    assert f"DSV4-MARLIN {'NO-GO' if flip else 'GO'}" in printed and "case decode1" in printed
    assert f"served@M1={json.dumps({'w13': list(TRUE), 'w2': list(TRUE)})}" in printed


def test_the_record_is_a_registered_property() -> None:
    from verity_vllm.properties.record import seal

    doc = seal(_doc(), X._verdict(_doc()))
    assert REGISTRY["dsv4_marlin_capture"].schema == X.SCHEMA and doc["ok"] and doc["digest"] == digest_of(doc)


def test_the_cases_cover_the_decode_batches_and_the_edge_experts() -> None:
    assert {M for _, M, _ in X.CASES} >= {1, 2, 3, 4} and max(M for _, M, _ in X.CASES) == max(X.SERVED_M) == 16
    assert set(X.QUICK) <= {c[0] for c in X.CASES} and set(X.FORCED) | set(X.QUICK_FORCED) <= {c[0] for c in X.CASES}
    assert [c for c in X.CASES if c[2] == "edge"] and 0 in X.EDGE_BYTES and 255 in X.EDGE_BYTES
    assert len(X.CANDIDATES) == 12 and X.SENTINEL != 0x7FFF
    assert all(c[2] != "edge" for c in X.CASES if c[0] in X.FORCED + X.QUICK_FORCED), "the probe's scale reads finite weights"
