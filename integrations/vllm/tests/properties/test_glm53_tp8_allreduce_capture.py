"""The GLM-5.3 TP8 all-reduce capture without a GPU: vLLM's dispatch at GLM-5.3's boundaries, the special and probe operands, the
comparison jobs over rank words made here by the models (a flipped word found and located, a rank that differs, a shot's
contraction decided by the probes), each launch's path and communicator checks, the bindings, the verdict, and `main` end to end
with the rank processes stood in for (the GPU run is `tests/properties/glm53_tp8_allreduce_capture_gpu.py --out ...` on an
8x H200 node).  Most tests run at N = 1536 = 3 * 2^9 tokens wide: rcp(N) is inexact there as at GLM-5.3's 6144, so the two
contractions of the norm differ on some tokens (at a power of two they never do), and the fused cluster is 2 CTAs."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tests.properties import glm53_tp8_allreduce_capture_gpu as X
from verity_vllm.program.registry import tp_allreduce as T
from verity_vllm.program.registry import tp_allreduce_check as K
from verity_vllm.properties.record import REGISTRY, digest_of

N = 1536
FMA = {1: 1, 2: 0}


@pytest.fixture
def small(monkeypatch):
    """The capture at N wide: N's fused budget (170 tokens), FlashInfer's one-shot strategy to 4 tokens in place of N's 42 (so
    that a two-shot launch's probe tokens are few to screen), and token counts on both sides of the boundaries, the custom
    all-reduce's 1-stage kernel to 85 tokens and its 8 MiB cap at 2730."""
    monkeypatch.setattr(X, "HIDDEN", N)
    monkeypatch.setattr(X, "FUSED_MAX", T.fused_max_tokens(N, X.WORLD))
    monkeypatch.setattr(X, "shot_of", lambda M: 1 if M <= 4 else 2)
    monkeypatch.setattr(X, "SCREEN_CHUNK", 200)
    monkeypatch.setattr(X, "FUSED_M", (3, 171))
    monkeypatch.setattr(X, "SPECIAL", (("special1", 3, 1), ("special2", 6, 2)))
    monkeypatch.setattr(X, "PROBE_M", {1: 4, 2: 6})
    monkeypatch.setattr(X, "AR_M", (2, 86))


def _case(phase: str, name: str, M: int, seed: int = 1) -> dict:
    return {"phase": phase, "case": name, "M": M, "key": f"{phase}/{name}", "seed": seed, **X.expect(phase, M)}


def _fused_words(case: dict, fma: int) -> dict[str, np.ndarray]:
    ops = {k: np.load(p) for k, p in case["ops"].items()}
    shot, cluster = (case["shot"], case["cluster"]) if case["path"] == X.FUSED else (2, 1)
    normed, res = K.mnnvl_norm_rows(ops["parts"], ops["res"], ops["w"], shot, cluster, X.EPS, bool(fma))
    return {"normed": normed, "residual": res}


def _ar_words(case: dict, stage: int) -> dict[str, np.ndarray]:
    parts = np.load(case["ops"]["parts"])
    return {"out": K.custom_rows(parts.reshape(X.WORLD, -1), stage).reshape(parts.shape[1:])}


def _write(case: dict, words: dict[str, np.ndarray], flip: tuple | None = None, rank: int | None = None) -> None:
    """Every rank's words, `flip` (name, index) xor 1 on `rank` (on every rank: a kernel's wrong word)."""
    for r in range(X.WORLD):
        d = Path(case["out"]) / f"rank{r}"
        d.mkdir(parents=True, exist_ok=True)
        for name, a in words.items():
            a = a.copy()
            if flip and flip[0] == name and rank in (None, r):
                a[flip[1]] ^= 1
            np.save(d / f"{name}.npy", a)


def _made(tmp_path: Path, case: dict) -> dict:
    case.update(X.make_operands(tmp_path, case), out=str(tmp_path / "words" / case["key"].replace("/", "_")))
    return case


def _fused_job(tmp_path, M: int, fma: int, *, name: str = "t", flip=None, rank: int | None = None) -> dict:
    case = _made(tmp_path, _case("fused", name, M))
    _write(case, _fused_words(case, fma), flip, rank)
    return X.jobs_of([case], False)[0]


def _ar_job(tmp_path, M: int, stage: int, *, symm_mem: bool = False, flip=None, rank: int | None = None) -> dict:
    case = _made(tmp_path, _case("allreduce", f"M{M}", M))
    _write(case, _ar_words(case, stage), flip, rank)
    return X.jobs_of([case], symm_mem)[0]


def test_the_statics_are_glm53s() -> None:
    cfg = json.loads((Path(__file__).parents[1] / "check" / "fixtures" / "glm_moe_dsa" / "config.json").read_text())
    assert (cfg["hidden_size"], cfg["rms_norm_eps"], cfg["dtype"]) == (X.HIDDEN, X.EPS, "bfloat16") and X.WORLD == 8


def test_the_dispatch_at_glm53s_boundaries() -> None:
    assert X.FUSED_MAX == 42 and [X.expected_path(M) for M in (42, 43)] == [X.FUSED, X.UNFUSED]
    assert [X.shot_of(M) for M in (10, 11)] == [1, 2] and [X.stage_of(M) for M in (21, 22)] == [1, 2]
    assert [X.expected_comm(M, True) for M in (21, 22, 682, 683)] == ["ca_comm", "symm_mem_comm", "symm_mem_comm", "symm_mem_comm"]
    assert [X.expected_comm(M, False) for M in (21, 22, 682, 683)] == ["ca_comm", "ca_comm", "ca_comm", "pynccl_comm"]
    assert X.expected_comm(5461, True) == "symm_mem_comm" and X.expected_comm(5461, False) == X.expected_comm(5462, True) == "pynccl_comm"
    assert {M: T.mnnvl_grid(M, X.HIDDEN)[1] for M in (16, 17, 33, 34)} == {16: 8, 17: 4, 33: 4, 34: 2}


def test_the_cases_cover_every_shot_geometry_and_communicator() -> None:
    geoms = {(1, 8), (2, 8), (2, 4), (2, 2)}
    for quick in (False, True):
        cs = X.cases(quick, set(X.PHASES))
        assert {(c["shot"], c["cluster"]) for c in cs if c.get("path") == X.FUSED} == geoms
        assert X.UNFUSED in {c.get("path") for c in cs}
        assert {(c["comm"], c["stage"]) for c in cs if c["phase"] == "allreduce"} >= {("ca_comm", 2), ("pynccl_comm", 2)}
        assert len({c["seed"] for c in cs}) == len({c["key"] for c in cs}) == len(cs)
    full = X.cases(False, set(X.PHASES))
    assert {(c["comm"], c["stage"]) for c in full if c["phase"] == "allreduce"} == {("ca_comm", 1), ("ca_comm", 2), ("pynccl_comm", 2)}
    assert {X.expected_comm(c["M"], True) for c in full if c["phase"] == "allreduce"} == {"ca_comm", "symm_mem_comm"}
    assert all(X.shot_of(M) == s for _, M, s in X.SPECIAL) and all(X.shot_of(M) == s for s, M in X.PROBE_M.items())
    assert X.cases(False, {"allreduce"}) and all(c["phase"] == "allreduce" for c in X.cases(False, {"allreduce"}))


def test_the_special_tokens_hold_what_they_name(tmp_path) -> None:
    """Every -0 partial sums to +0 (sanitized); the near-cancelling pair flushes to -0, which one-shot keeps and two-shot's add onto
    +0 does not; `ORDER` sums to 1 and its reverse to 1 + 2^-7 in ascending rank order; a NaN partial makes its token's residual
    NaN and its every normed word NaN; and an all-reduce keeps -0."""
    nan = lambda a: (a & 0x7FFF) > 0x7F80  # noqa: E731
    for name, M, shot in X.SPECIAL:
        case = _made(tmp_path, _case("fused", name, M))
        w = _fused_words(case, 1)
        assert (w["residual"][0] == 0).all() and (w["normed"][0] & 0x7FFF == 0).all()
        assert (w["residual"][1, :64] == (X.NEG0 if shot == 1 else 0)).all() and w["residual"][1, 128:130].tolist() == [0x3F80, 0x3F81]
        assert (np.load(case["ops"]["parts"])[:, 1, 64:128] & 0x7F80 == 0).all()
        assert nan(w["residual"][2, 5]) and nan(w["normed"][2]).all() and not nan(w["residual"][2, 6])
    case = _made(tmp_path, _case("allreduce", "M3", 3))
    for stage in (1, 2):
        out = _ar_words(case, stage)["out"][0]
        assert (out[:8] == X.NEG0).all() and (out[8:16] == 0).all() and (out[16:24] & 0x7FFF).any() and nan(out[24])


def test_probe_tokens_tell_the_contractions_apart(small) -> None:
    for shot, M in X.PROBE_M.items():
        parts, res, w, info = X.probe_tokens(np.random.default_rng(shot), M, shot)
        a, b = (K.mnnvl_norm_rows(parts, res, w, shot, T.mnnvl_grid(M, X.HIDDEN)[1], X.EPS, f)[0] for f in (False, True))
        assert info["found"] == M and parts.shape == (X.WORLD, M, X.HIDDEN) and (a != b).any(axis=1).all()


def test_probe_screening_falls_back_to_random_tokens(small, monkeypatch) -> None:
    monkeypatch.setattr(X, "SCREEN_MAX", 1)
    parts, res, _, info = X.probe_tokens(np.random.default_rng(0), 300, 2)
    assert info["found"] < 300 and info["screened"] == X.SCREEN_CHUNK and parts.shape[1] == res.shape[0] == 300


@pytest.mark.slow
def test_the_capture_seeds_find_every_probe_token() -> None:
    """The run's own probe operands, at GLM-5.3's width and seeds (about 20 s): every token of both probe calls tells the two
    contractions apart. Catches a seed, scale or geometry change that would leave a shot's `fma` undecided on the 8-GPU node,
    where the run can't be redone cheaply; the small-width tests can't, since the screening rate depends on the width."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        for c in X.cases(False, {"fused"}):
            if c["case"].startswith("probe"):
                assert X.make_operands(Path(d), c)["screen"]["found"] == c["M"], c["key"]


@pytest.mark.parametrize("M", [3, 6])
@pytest.mark.parametrize("fma", [0, 1])
def test_a_fused_launch_takes_the_kernel_words(small, tmp_path, M: int, fma: int) -> None:
    job = _fused_job(tmp_path, M, fma, name="probe1" if M == 3 else "probe2")
    assert (job["shot"], job["cluster"]) == (1 if M == 3 else 2, 2)
    t = X.compare(job)
    assert "error" not in t, t
    assert (t["words"], t["mismatch"], t["normed_words"], t["rank_mismatch"]) == (M * N, 0, M * N, 0)
    assert t[f"fma{fma}_mismatch"] == 0 and t[f"fma{1 - fma}_mismatch"] == t["fma_differ"] > 0 and X.fma_of(t) == fma
    assert (t["reference"], t["twin_mismatch"], t["first_mismatch"], t[f"first_fma{fma}_mismatch"]) == (6, 0, None, None)
    assert t[f"first_fma{1 - fma}_mismatch"]["what"] == f"normed fma={1 - fma}"


@pytest.mark.parametrize("flip,field", [(("residual", (2, 100)), "first_mismatch"), (("normed", (1, 7)), "first_fma1_mismatch")])
def test_a_flipped_fused_word_is_found_and_located(small, tmp_path, flip: tuple, field: str) -> None:
    t = X.compare(_fused_job(tmp_path, 3, 1, flip=flip))
    assert t[field]["at"] == list(flip[1]) and t[field]["kernel"] == t[field]["want"] ^ 1
    assert (t["mismatch"], t["fma1_mismatch"]) == ((1, 0) if flip[0] == "residual" else (0, 1)) and t["rank_mismatch"] == 0


def test_a_rank_that_differs_is_counted(small, tmp_path) -> None:
    t = X.compare(_fused_job(tmp_path, 3, 1, flip=("normed", (0, 9)), rank=5))
    assert (t["rank_mismatch"], t["fma1_mismatch"], t["mismatch"]) == (1, 0, 0)
    t = X.compare(_ar_job(tmp_path, 2, 1, flip=("out", (1, 4)), rank=7))
    assert (t["rank_mismatch"], t["mismatch"]) == (1, 0)


def test_an_unfused_launch_is_compared_across_ranks_only(small, tmp_path) -> None:
    job = _fused_job(tmp_path, 171, 1, flip=("residual", (0, 0)), rank=2)
    assert job["model"] is False and "shot" not in job
    t = X.compare(job)
    assert (t["rank_mismatch"], t["words"], t["reference"]) == (1, 0, 0)


@pytest.mark.parametrize("M,stage", [(2, 1), (86, 2)])
def test_a_custom_all_reduce_takes_the_kernel_words(small, tmp_path, M: int, stage: int) -> None:
    job = _ar_job(tmp_path, M, stage)
    assert job["stage"] == stage and job["model"]
    t = X.compare(job)
    assert (t["words"], t["mismatch"], t["rank_mismatch"], t["twin_mismatch"]) == (M * N, 0, 0, 0)
    assert t["reference"] == len(X.custom_sample(M * N)) == 7 + 2 * X.WORLD
    t = X.compare(_ar_job(tmp_path, M, stage, flip=("out", (M - 1, N - 2))))
    assert t["mismatch"] == 1 and t["first_mismatch"]["at"] == [M * N - 2]


def test_the_order_elements_tell_the_stages_apart(small, tmp_path) -> None:
    """At each 2-stage owner's first two elements: `ORDER` from rank 0 (the 1-stage kernel's only owner) sums to 1 and its
    reverse to 1 + 2^-7; from the 2-stage kernel's other owners at least one of the two differs, as the ties fall (1 + 2^-7 when
    five or six 2^-26 come before the 1, 1 when four or fewer do). Random words can't tell the stages apart: the f32 sum of 8
    bf16 partials of one scale is almost always exact."""
    job = _ar_job(tmp_path, 86, 2)
    parts = np.load(job["ops"]["parts"]).reshape(X.WORLD, -1)
    at = X.order_elements(parts.shape[1])
    assert len(at) == 2 * X.WORLD and (parts[:, at[0]] == X.ORDER).all() and (parts[:, at[1]] == X.ORDER[::-1]).all()
    one, two = (K.custom_rows(parts, s)[at] for s in (1, 2))
    assert one.tolist() == [0x3F80, 0x3F81] * X.WORLD
    want = [(0x3F80, 0x3F81), (0x3F81, 0x3F81), (0x3F81, 0x3F80), (0x3F81, 0x3F80)] + [(0x3F80, 0x3F80)] * 4
    assert two.tolist() == [w for pair in want for w in pair]
    assert X.compare({**job, "stage": 1})["mismatch"] == 9


def test_symmetric_memory_is_compared_with_both_stages(small, tmp_path) -> None:
    job = _ar_job(tmp_path, 86, 2, symm_mem=True)
    assert job["stage"] is None and job["model"]
    t = X.compare(job)
    assert t["stage2_mismatch"] == 0 and t["stage1_mismatch"] > 0 and t["words"] == 0
    assert X.jobs_of([{**_case("allreduce", "M2731", 2731), "ops": {}, "out": ""}], False)[0]["model"] is False


def test_fma_is_decided_only_when_the_probes_tell() -> None:
    t = {"normed_words": 9, "fma_differ": 2, "fma1_mismatch": 0, "fma0_mismatch": 2}
    assert X.fma_of(t) == 1 and X.fma_of({**t, "fma1_mismatch": 2, "fma0_mismatch": 0}) == 0
    assert X.fma_of({**t, "fma_differ": 0}) is X.fma_of({**t, "fma0_mismatch": 0}) is X.fma_of({**t, "fma0_mismatch": 1, "fma1_mismatch": 1}) is None
    assert X.fma_of({**t, "normed_words": 0}) is None


def _call(case: dict) -> dict:
    if case["path"] == X.UNFUSED:
        return {"M": case["M"], "can_use": False, "max_token_num": 0, "path": X.UNFUSED}
    return {"M": case["M"], "can_use": True, "max_token_num": X.FUSED_MAX, "path": X.FUSED, "backend": "mnnvl", "shot": case["shot"],
            "weight_bias": 0.0, "epsilon": X.EPS}


def _rank_case(case: dict, symm_mem: bool, us: float) -> dict:
    if case["phase"] == "fused":
        return {"calls": [_call(case)] * 2, "deterministic": True, "us": us, "kernels": ["void allreduce_fusion_kernel"]}
    comm = X.expected_comm(case["M"], symm_mem)
    kernels = [f"void vllm::cross_device_reduce_{case['stage']}stage<__nv_bfloat16, 8>(...)"] if comm == "ca_comm" else []
    return {"comms": [comm] * 2, "deterministic": True, "us": us, "kernels": kernels}


def _records(cases_: list[dict], symm_mem: bool = False, us: float = 3.0) -> list[dict]:
    return [{"rank": r, "cases": {c["key"]: _rank_case(c, symm_mem, us) for c in cases_}, "errors": {}, "communicators": {}}
            for r in range(X.WORLD)]


@pytest.mark.parametrize("change,ok", [({}, True), ({"backend": "trtllm"}, False), ({"shot": 1}, False), ({"weight_bias": 1.0}, False),
                                       ({"epsilon": 1e-6}, False), ({"max_token_num": 128}, False), ({"path": X.UNFUSED}, False),
                                       ({"can_use": False}, False)])
def test_a_fused_call_is_on_its_expected_path(change: dict, ok: bool) -> None:
    case = _case("fused", "M11", 11)
    recs = _records([case])
    recs[4]["cases"][case["key"]]["calls"][1] = {**_call(case), **change}
    assert X.launch_ok(case, recs, False) is ok
    past = _case("fused", "M43", 43)
    assert X.launch_ok(past, _records([past]), False)
    recs = _records([past])
    recs[0]["cases"][past["key"]]["calls"] = [{**_call(case), "M": 43}] * 2
    assert not X.launch_ok(past, recs, False)


def test_an_all_reduce_is_on_its_expected_communicator_and_kernel() -> None:
    case = _case("allreduce", "M22", 22)
    assert X.launch_ok(case, _records([case]), False) and X.launch_ok(case, _records([case], True), True)
    assert not X.launch_ok(case, _records([case], True), False) and not X.launch_ok(case, _records([case]), True)
    recs = _records([case])
    recs[0]["cases"][case["key"]]["kernels"] = ["void vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>(...)"]
    assert not X.launch_ok(case, recs, False)
    recs[0]["cases"][case["key"]]["kernels"] = {"error": "profiler unavailable"}
    assert X.launch_ok(case, recs, False)
    for mutate in (lambda r: r[3]["cases"][case["key"]].update(deterministic=False), lambda r: r[6]["errors"].update({case["key"]: "boom"}),
                   lambda r: r[2]["cases"].pop(case["key"]), lambda r: r.pop()):
        recs = _records([case])
        mutate(recs)
        assert not X.launch_ok(case, recs, False)


def test_the_programs_bind_at_every_glm53_shape() -> None:
    cs = X.cases(False, set(X.PHASES))
    got = X.bind_check(cs)
    assert got["ok"] and len(got["bindings"]) == 2 * 4 + 6
    assert got["bindings"]["AllReduceRmsNormMnnvl_v1{WORLD=8,N=6144,SHOT=2,CLUSTER=2,EPS=1e-05,FMA=True}"]
    assert got["bindings"]["AllReduceCustom_v1{WORLD=8,SIZE=4190208,STAGE=2}"]


def _doc() -> dict:
    """A GO record at GLM-5.3's statics without words: one launch per fused shot and an unfused one, a 2-stage custom all-reduce
    and an NCCL one."""
    cs = [_case("fused", "M10", 10), _case("fused", "M42", 42), _case("fused", "M43", 43), _case("allreduce", "M64", 64),
          _case("allreduce", "M683", 683)]
    zero = {k: 0 for k in X.TOTALS}
    comps = [{"key": "fused/M10", **zero, "words": 9, "normed_words": 9, "fma0_mismatch": 2, "fma_differ": 2, "reference": 4},
             {"key": "fused/M42", **zero, "words": 9, "normed_words": 9, "fma1_mismatch": 3, "fma_differ": 3, "reference": 4},
             {"key": "fused/M43", **zero}, {"key": "allreduce/M64", **zero, "words": 9, "reference": 7}, {"key": "allreduce/M683", **zero}]
    doc = {"schema": X.SCHEMA, "gate": {"ok": True}, "errors": {}, "bind": {"ok": True}, "case_list": cs, "ranks": _records(cs),
           "comparisons": comps}
    doc["fma"] = {str(s): X.fma_of(X._shot_totals(doc, s)) for s in (1, 2)}
    doc["cases"] = X.case_verdicts(doc)
    return doc


def test_each_case_has_its_own_verdict() -> None:
    doc = _doc()
    assert doc["fma"] == {"1": 1, "2": 0} and all(c["ok"] for c in doc["cases"].values())
    assert doc["cases"]["fused/M10"]["path"] == [X.FUSED] * 2 and doc["cases"]["allreduce/M683"]["path"] == ["pynccl_comm"] * 2
    for mutate, bad in [(lambda d: d["comparisons"][3].update(mismatch=1), "allreduce/M64"),
                        (lambda d: d["comparisons"][4].update(rank_mismatch=1), "allreduce/M683"),
                        (lambda d: d["comparisons"][0].update(reference=0), "fused/M10"),
                        (lambda d: d["comparisons"][1].update(twin_mismatch=1), "fused/M42"),
                        (lambda d: d["comparisons"].pop(2), "fused/M43"),
                        (lambda d: d["ranks"][1]["errors"].update({"fused/M43": "boom"}), "fused/M43"),
                        (lambda d: d["ranks"][7]["cases"]["fused/M10"]["calls"][0].update(shot=2), "fused/M10"),
                        (lambda d: d["fma"].update({"2": 1}), "fused/M42"),
                        (lambda d: d["fma"].update({"1": None}), "fused/M10")]:
        doc = _doc()
        mutate(doc)
        got = X.case_verdicts(doc)
        assert {k for k, c in got.items() if not c["ok"]} == {bad}


@pytest.mark.parametrize("path,value", [(("gate", "ok"), False), (("errors", "rank3/fused/M10"), "boom"), (("bind", "ok"), False),
                                        (("fma", "2"), None), (("fma", "1"), None), (("cases", "allreduce/M683", "ok"), False)])
def test_the_verdict_needs_every_check(path: tuple, value) -> None:
    assert X._verdict(_doc())
    doc = copy.deepcopy(_doc())
    node = doc
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    assert not X._verdict(doc)
    assert not X._verdict({**_doc(), "ranks": _doc()["ranks"][:7]})
    one_phase = _doc()
    one_phase["case_list"] = [c for c in one_phase["case_list"] if c["phase"] == "fused"]
    assert not X._verdict(one_phase)                                             # a run without the all-reduce phase is not a GO


def _fake_spawn(flip: dict):
    """`spawn_ranks` stood in for: every rank's words from the models (a fused shot's at `FMA`, a custom all-reduce's at its stage;
    a launch the capture doesn't model, the same words on every rank) and its records, the time 3 us switched and 2 us served."""

    def spawn(work, env, cases_):
        symm = env == X.DEFAULT_ENV
        for c in cases_:
            c["out"] = str(work / c["key"].replace("/", "_"))
            if c["phase"] == "fused":
                words = _fused_words(c, FMA.get(c.get("shot"), 1))
            else:
                words = _ar_words(c, c["stage"] if X.expected_comm(c["M"], symm) == "ca_comm" else 2)
            _write(c, words, flip.get(c["key"]))
        return _records(cases_, symm, 2.0 if symm else 3.0)

    return spawn


@pytest.mark.slow
def test_main_end_to_end_on_stubbed_ranks(small, tmp_path, monkeypatch, capsys) -> None:
    """`main` with the rank processes stood in for (`_fake_spawn`) and the gate passed: the operands, the probes' screening, the
    comparison jobs through the spawned worker pool at the jobs' own widths and geometries, each shot's decided `fma`, the per-case
    verdicts, the served-default run and the switch's cost, the sealed record and the exit status, so a slip in the orchestration
    shows here and not in the 8-GPU node's first hour.  About 35 s: `main` runs twice, and each modelled fused launch evaluates the reference 6 times."""
    monkeypatch.setattr(X, "gate", lambda *a: {"ok": True, "checks": {}})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    flip = {"fused/M3": ("residual", (1, 5))}
    monkeypatch.setattr(X, "spawn_ranks", _fake_spawn(flip))
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1"]) == 1
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is False and not doc["errors"] and doc["fma"] == {"1": 1, "2": 0}
    assert {k for k, c in doc["cases"].items() if not c["ok"]} == {"fused/M3"} and doc["totals"]["mismatch"] == 1
    assert set(doc["cases"]) == {"fused/M3", "fused/M171", "fused/special1", "fused/special2", "fused/probe1", "fused/probe2",
                                 "allreduce/M2", "allreduce/M86"}
    assert doc["bind"]["ok"] and {c["key"]: c["screen"]["found"] for c in doc["case_list"] if "screen" in c} == {"fused/probe1": 4,
                                                                                                                "fused/probe2": 6}
    served = doc["served_default"]["cases"]
    assert served["allreduce/M86"]["comms"] == ["symm_mem_comm"] * 2 and served["allreduce/M86"]["stage2_mismatch"] == 0
    assert served["allreduce/M2"]["mismatch"] == 0 and doc["switch_cost"] == {"allreduce/M2": 1.5, "allreduce/M86": 1.5}
    printed = capsys.readouterr().out
    assert "case fused/M3" in printed and "GLM53-TP8-ALLREDUCE NO-GO" in printed and "no-go cases=['fused/M3']" in printed
    flip.clear()
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1", "--no-default"]) == 0
    printed = capsys.readouterr().out
    assert "GLM53-TP8-ALLREDUCE GO" in printed and "served default" not in printed


def test_a_failed_gate_runs_nothing(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(X, "gate", lambda *a: {"ok": False, "checks": {"world": False}})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    monkeypatch.setattr(X, "spawn_ranks", lambda *a: pytest.fail("ranks spawned past a failed gate"))
    with pytest.raises(SystemExit) as e:
        X.main(["--out", str(tmp_path), "--scratch", str(tmp_path)])
    assert e.value.code == 3 and "GATE-FAIL" in capsys.readouterr().out
    assert json.loads((tmp_path / X.RECORD).read_text())["ok"] is False


def test_the_record_is_a_registered_property() -> None:
    from verity_vllm.properties.record import seal

    doc = seal(_doc(), X._verdict(_doc()))
    assert REGISTRY["glm53_tp8_allreduce_capture"].schema == X.SCHEMA and doc["ok"] and doc["digest"] == digest_of(doc)
