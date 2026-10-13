"""The GLM-5.3 TP8 all-reduce capture without a GPU: vLLM's dispatch at GLM-5.3's boundaries in the served engine and under the
switch, the special and probe operands, the probes' trees and FTZ read back from words the models make (and refused when the
switch's adds are not one f32 tree per owner), the comparison jobs over rank words made here by the models (a flipped word found
and located, a rank that differs, a shot's contraction decided by the probes), each launch's path, communicator and kernel
checks, the bindings, the routes and the verdict, and `main` end to end with the rank processes stood in for (the GPU run is
`tests/properties/glm53_tp8_allreduce_capture_gpu.py --out ...` on an 8x H200 node).  Most tests run at N = 1536 = 3 * 2^9
tokens wide: rcp(N) is inexact there as at GLM-5.3's 6144, so the two contractions of the norm differ on some tokens (at a power
of two they never do), and the fused cluster is 2 CTAs."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tests.properties import glm53_tp8_allreduce_capture_gpu as X
from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.registry import tp_allreduce as T
from verity_vllm.program.registry import tp_allreduce_check as K
from verity_vllm.properties.record import REGISTRY, digest_of

N = 1536
FMA = {1: 1, 2: 0}
BALANCED = (((0, 1), (2, 3)), ((4, 5), (6, 7)))


def _random_tree(rng):
    nodes = list(range(X.WORLD))
    while len(nodes) > 1:
        i, j = sorted(rng.choice(len(nodes), 2, replace=False))
        b, a = nodes.pop(j), nodes.pop(i)
        nodes.append((a, b))
    return T.multimem_tree(nodes[0], X.WORLD)


TREES = tuple(_random_tree(np.random.default_rng(s)) for s in range(X.WORLD))


@pytest.fixture
def small(monkeypatch):
    """The capture at N wide: N's FlashInfer budget (170 tokens), FlashInfer's one-shot strategy to 4 tokens in place of N's 42 (so
    that a two-shot launch's probe tokens are few to screen), and token counts on both sides of the boundaries: multimem from 171
    tokens, the custom all-reduce 2-stage there."""
    monkeypatch.setattr(X, "HIDDEN", N)
    monkeypatch.setattr(X, "FUSED_MAX", T.fused_max_tokens(N, X.WORLD))
    monkeypatch.setattr(X, "shot_of", lambda M: 1 if M <= 4 else 2)
    monkeypatch.setattr(X, "SCREEN_CHUNK", 200)
    monkeypatch.setattr(X, "FUSED_M", (3, 171))
    monkeypatch.setattr(X, "SPECIAL", (("special1", 3, 1), ("special2", 6, 2)))
    monkeypatch.setattr(X, "PROBE_M", {1: 4, 2: 6})
    monkeypatch.setattr(X, "AR_M", (2, 6, 171))
    monkeypatch.setattr(X, "PROBE_M_AR", (171,))
    monkeypatch.setattr(X, "SWITCH_M", (171,))


def _case(phase: str, name: str, M: int, seed: int = 1) -> dict:
    return {"phase": phase, "case": name, "M": M, "key": f"{phase}/{name}", "seed": seed, "spawn": X.SPAWN_OF[phase],
            **X.expect(phase, M)}


def _fused_words(case: dict, fma: int) -> dict[str, np.ndarray]:
    ops = {k: np.load(p) for k, p in case["ops"].items()}
    shot, cluster = (case["shot"], case["cluster"]) if case["path"] == X.FUSED else (2, 1)
    normed, res = K.mnnvl_norm_rows(ops["parts"], ops["res"], ops["w"], shot, cluster, X.EPS, bool(fma))
    return {"normed": normed, "residual": res}


def _ar_words(case: dict, trees=TREES, ftz: bool = True) -> dict[str, np.ndarray]:
    """An all-reduce's words from its communicator's model (NCCL's: any sum, the same on every rank)."""
    parts = np.load(case["ops"]["parts"])
    flat = parts.reshape(X.WORLD, -1)
    comm = case["comm"]
    if comm == "fi_ar_comm":
        out = K.mnnvl_rows(flat, case["shot"])
    elif comm == "symm_mem_comm":
        out = K.multimem_rows(flat, trees, ftz)
    else:
        out = K.custom_rows(flat, case["stage"] or 2)
    return {"out": out.reshape(parts.shape[1:])}


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
    return X.jobs_of([case])[0]


def _ar_job(tmp_path, M: int, phase: str = "allreduce", *, flip=None, rank: int | None = None, trees=TREES, ftz=True) -> dict:
    case = _made(tmp_path, _case(phase, f"M{M}", M))
    _write(case, _ar_words(case, TREES, True), flip, rank)
    return X.jobs_of([case], trees, ftz)[0]


# --- dispatch and cases ----------------------------------------------------------------------------------------------------------

def test_the_statics_are_glm53s() -> None:
    cfg = json.loads((Path(__file__).parents[1] / "check" / "fixtures" / "glm_moe_dsa" / "config.json").read_text())
    assert (cfg["hidden_size"], cfg["rms_norm_eps"], cfg["dtype"]) == (X.HIDDEN, X.EPS, "bfloat16") and X.WORLD == 8


def test_the_dispatch_at_glm53s_boundaries() -> None:
    """Served (custom all-reduce off): FlashInfer to 42 tokens, multimem to 5461 (64 MiB), NCCL past; switched: the custom
    all-reduce from 43 to 682, always 2-stage, NCCL past."""
    assert X.FUSED_MAX == 42 and [X.expected_path(M) for M in (42, 43)] == [X.FUSED, X.UNFUSED]
    assert [X.shot_of(M) for M in (10, 11)] == [1, 2] and [X.stage_of(M) for M in (21, 22, 43)] == [1, 2, 2]
    assert [X.expected_comm(M, False) for M in (1, 42, 43, 256, 2048, 5461, 5462)] == \
        ["fi_ar_comm"] * 2 + ["symm_mem_comm"] * 4 + ["pynccl_comm"]
    assert [X.expected_comm(M, True) for M in (42, 43, 682, 683)] == ["fi_ar_comm", "ca_comm", "ca_comm", "pynccl_comm"]
    assert {M: T.mnnvl_grid(M, X.HIDDEN)[1] for M in (16, 17, 33, 34)} == {16: 8, 17: 4, 33: 4, 34: 2}
    assert X.SPAWNS["served"] == (X.SERVED_ENV, False) and X.SPAWNS["switch"][1] is True
    assert X.SWITCH_ENV["VLLM_ALLREDUCE_USE_SYMM_MEM"] == "0" and X.SERVED_ENV["VLLM_ALLREDUCE_USE_SYMM_MEM"] == "1"


def test_the_cases_cover_every_route() -> None:
    geoms = {(1, 8), (2, 8), (2, 4), (2, 2)}
    for quick in (False, True):
        cs = X.cases(quick, set(X.PHASES))
        assert {(c["shot"], c["cluster"]) for c in cs if c.get("path") == X.FUSED} == geoms and X.UNFUSED in {c.get("path") for c in cs}
        ar = [c for c in cs if c["phase"] == "allreduce"]
        assert {c["comm"] for c in ar} == {"fi_ar_comm", "symm_mem_comm"} and {c["shot"] for c in ar if c["shot"]} == {1, 2}
        assert 1 in {c["M"] for c in ar} and 256 in {c["M"] for c in ar}
        assert {(c["comm"], c["stage"]) for c in cs if c["phase"] == "switch"} == {("ca_comm", 2), ("pynccl_comm", None)}
        probes = {c["M"]: c["seed"] for c in cs if c["phase"] == "probe"}
        assert probes and probes == {c["M"]: c["seed"] for c in cs if c["phase"] == "again"}
        assert all(c["comm"] == "symm_mem_comm" for c in cs if c["phase"] in ("probe", "again"))
        assert {c["spawn"] for c in cs} == set(X.SPAWNS) and len({c["key"] for c in cs}) == len(cs)
        assert len({c["seed"] for c in cs if c["phase"] != "again"}) == len(cs) - len(probes)
    assert all(X.shot_of(M) == s for _, M, s in X.SPECIAL) and all(X.shot_of(M) == s for s, M in X.PROBE_M.items())
    assert X.cases(False, {"switch"}) and all(c["phase"] == "switch" for c in X.cases(False, {"switch"}))


def test_the_special_tokens_hold_what_they_name(tmp_path) -> None:
    """Every -0 partial sums to +0 (sanitized); the near-cancelling pair flushes to -0, which one-shot keeps and two-shot's add onto
    +0 does not; `ORDER` sums to 1 and its reverse to 1 + 2^-7 in ascending rank order; a NaN partial makes its token's residual
    NaN and its every normed word NaN; and an all-reduce of IEEE adds keeps -0."""
    nan = lambda a: (a & 0x7FFF) > 0x7F80  # noqa: E731
    for name, M, shot in X.SPECIAL:
        case = _made(tmp_path, _case("fused", name, M))
        w = _fused_words(case, 1)
        assert (w["residual"][0] == 0).all() and (w["normed"][0] & 0x7FFF == 0).all()
        assert (w["residual"][1, :64] == (X.NEG0 if shot == 1 else 0)).all() and w["residual"][1, 128:130].tolist() == [0x3F80, 0x3F81]
        assert (np.load(case["ops"]["parts"])[:, 1, 64:128] & 0x7F80 == 0).all()
        assert nan(w["residual"][2, 5]) and nan(w["normed"][2]).all() and not nan(w["residual"][2, 6])
    case = _made(tmp_path, _case("switch", "M43", 43))
    parts = np.load(case["ops"]["parts"]).reshape(X.WORLD, -1)
    for out in (K.custom_rows(parts, 1), K.custom_rows(parts, 2), K.multimem_rows(parts, (BALANCED,) * 8, False)):
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


# --- the probes' trees -----------------------------------------------------------------------------------------------------------

def test_the_probe_operands_hold_their_pairs_and_specials() -> None:
    """At every pair slot +B on i, -B on j and 2^0..2^5 on the others in ascending rank order; at every special slot its words; and
    at GLM-5.3's probe sizes every owner's slice is whole periods."""
    size = 1024
    parts = X.probe_parts(np.random.default_rng(0), size)
    owner, slot, rot = X.probe_slots(size)
    for f, big in enumerate(X.BIG):
        for p, (i, j) in enumerate(X.PAIRS):
            at = slot == f * len(X.PAIRS) + p
            others = [r for r in range(X.WORLD) if r not in (i, j)]
            assert (parts[i, at] == big).all() and (parts[j, at] == big | 0x8000).all()
            assert (parts[others][:, at] == np.array(X.SMALL, np.uint16)[:, None]).all()
    halves = np.flatnonzero(slot == 2 * len(X.PAIRS) + 1)
    assert (parts[rot[halves], halves] == 0x0040).all() and (parts[(rot[halves] + 1) % 8, halves] == 0x0040).all()
    assert (parts[:, slot == 2 * len(X.PAIRS) + 3] == X.NEG0).all()
    assert sorted(set(owner.tolist())) == list(range(8)) and X.PERIOD == 64
    for M in X.PROBE_M_AR:
        assert all((hi - lo) % X.PERIOD == 0 for lo, hi, _ in T.multimem_owners(X.WORLD, M * X.HIDDEN))


def test_the_probe_reads_any_trees_and_ftz_back() -> None:
    """Words from the model at 40 random sets of one tree per owner, IEEE or FTZ, give back those trees and that FTZ."""
    size = 1024
    parts = X.probe_parts(np.random.default_rng(2), size)
    rng = np.random.default_rng(3)
    for k in range(40):
        trees = tuple(_random_tree(rng) for _ in range(X.WORLD))
        out = K.multimem_rows(parts, trees, bool(k % 2))
        got = X.decode_probe(out, size)
        assert got["trees"] == list(trees) and X.decide_ftz(out, parts)["ftz"] is bool(k % 2)
        assert all(o["families_agree"] and o["absorbing"] == [True, True] for o in got["owners"].values())


def test_the_capture_probes_read_trees_at_glm53s_sizes(tmp_path) -> None:
    """The run's own probe operands at GLM-5.3's width and seeds give back distinct trees per owner, and the `again` probes'
    operands are the first probes'."""
    cs = X.cases(False, {"probe", "again"})
    for c in cs:
        c.update(X.make_operands(tmp_path, c))
    for c in [c for c in cs if c["phase"] == "probe"]:
        parts = np.load(c["ops"]["parts"]).reshape(X.WORLD, -1)
        again = next(a for a in cs if a["phase"] == "again" and a["M"] == c["M"])
        assert np.array_equal(parts, np.load(again["ops"]["parts"]).reshape(X.WORLD, -1))
        out = K.multimem_rows(parts, TREES, False)
        assert X.decode_probe(out, parts.shape[1])["trees"] == list(TREES) and X.decide_ftz(out, parts)["ftz"] is False


def _f64_sum(parts: np.ndarray) -> np.ndarray:
    v = ((parts.astype(np.uint32) << 16).view(np.float32)).astype(np.float64).sum(0).astype(np.float32)
    return DR.f2fp_bf16(v.view(np.uint32))


def test_a_switch_no_f32_tree_states_is_refused() -> None:
    """An f64 accumulator (2^40 + 63 exact, 2^100 absorbing): no tree, the families disagree; an order that changes along an
    owner's slice (each period another tree): no tree; a probe's words outside 0..63: named."""
    size = 1024
    parts = X.probe_parts(np.random.default_rng(4), size)
    got = X.decode_probe(_f64_sum(parts), size)
    assert got["trees"] is None and all(o["absorbing"] == [False, True] and not o["families_agree"] for o in got["owners"].values())
    a, b = K.multimem_rows(parts, (BALANCED,) * 8, False), K.multimem_rows(parts, (T.chain(8),) * 8, False)
    _, slot, _ = X.probe_slots(size)
    mixed = np.where((np.arange(size) // X.PERIOD) % 2 == 0, a, b)
    got = X.decode_probe(mixed, size)
    assert got["trees"] is None and not any(o["one_cluster_per_pair"] for o in got["owners"].values())
    bad = a.copy()
    bad[slot == 0] = 0x3FC0
    got = X.decode_probe(bad, size)
    assert got["trees"] is None and got["owners"]["0"]["bad_words"] == {"0:0,1": [0x3FC0]}


def test_ftz_is_decided_only_by_a_whole_flush() -> None:
    """Flushing inputs only or results only fits neither model: FTZ is undecided."""
    size = 1024
    parts = X.probe_parts(np.random.default_rng(5), size)
    out = K.multimem_rows(parts, (BALANCED,) * 8, False)
    _, slot, _ = X.probe_slots(size)
    names = [n for n, *_ in X.SPECIALS]
    at = {n: slot == 2 * len(X.PAIRS) + names.index(n) for n in X.FLUSH}
    assert X.decide_ftz(out, parts)["words"] == {"tiny": [1], "halves": [0x80], "cancel": [1], "neg_tiny": [0x8001]}
    inputs_only, results_only = out.copy(), out.copy()
    for n, w in (("tiny", 0), ("halves", 0), ("neg_tiny", X.NEG0)):
        inputs_only[at[n]] = w
    for n, w in (("tiny", 0), ("cancel", 0), ("neg_tiny", X.NEG0)):
        results_only[at[n]] = w
    for o in (inputs_only, results_only):
        assert X.decide_ftz(o, parts) == {**X.decide_ftz(o, parts), "ftz": None, "fits": []}


# --- comparisons -----------------------------------------------------------------------------------------------------------------

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
    t = X.compare(_ar_job(tmp_path, 2, flip=("out", (1, 4)), rank=7))
    assert (t["rank_mismatch"], t["mismatch"]) == (1, 0)


def test_an_unfused_launch_is_compared_across_ranks_only(small, tmp_path) -> None:
    job = _fused_job(tmp_path, 171, 1, flip=("residual", (0, 0)), rank=2)
    assert job["model"] is False and "shot" not in job
    t = X.compare(job)
    assert (t["rank_mismatch"], t["words"], t["reference"]) == (1, 0, 0)


@pytest.mark.parametrize("phase,M,model", [("allreduce", 2, "mnnvl"), ("allreduce", 6, "mnnvl"), ("allreduce", 171, "multimem"),
                                           ("probe", 171, "multimem"), ("switch", 171, "custom")])
def test_an_all_reduce_takes_the_kernel_words(small, tmp_path, phase: str, M: int, model: str) -> None:
    job = _ar_job(tmp_path, M, phase)
    assert job["model"] == model
    t = X.compare(job)
    assert "error" not in t, t
    sample = X.multimem_sample(M * N) if model == "multimem" else X.custom_sample(M * N)
    assert (t["words"], t["mismatch"], t["rank_mismatch"], t["twin_mismatch"], t["reference"]) == (M * N, 0, 0, 0, len(sample))
    t = X.compare(_ar_job(tmp_path, M, phase, flip=("out", (M - 1, N - 2))))
    assert t["mismatch"] == 1 and t["first_mismatch"]["at"] == [M * N - 2]


def test_a_multimem_launch_is_modelled_only_at_pinned_trees(small, tmp_path) -> None:
    """Without the probes' trees a multimem launch has no words compared (so no GO); at other trees or the other FTZ its words
    differ."""
    job = _ar_job(tmp_path, 171, trees=None)
    assert job["model"] is None and X.compare(job)["words"] == 0
    assert X.compare(_ar_job(tmp_path, 171, phase="probe", trees=(BALANCED,) * 8))["mismatch"] > 0
    assert X.compare(_ar_job(tmp_path, 171, phase="probe", ftz=False))["mismatch"] > 0


def test_the_order_elements_tell_the_stages_apart(small, tmp_path) -> None:
    """At each 2-stage owner's first two elements: `ORDER` from rank 0 (the 1-stage kernel's only owner) sums to 1 and its
    reverse to 1 + 2^-7; from the 2-stage kernel's other owners at least one of the two differs, as the ties fall (1 + 2^-7 when
    five or six 2^-26 come before the 1, 1 when four or fewer do)."""
    job = _ar_job(tmp_path, 171, "switch")
    parts = np.load(job["ops"]["parts"]).reshape(X.WORLD, -1)
    at = X.order_elements(parts.shape[1])
    assert len(at) == 2 * X.WORLD and (parts[:, at[0]] == X.ORDER).all() and (parts[:, at[1]] == X.ORDER[::-1]).all()
    one, two = (K.custom_rows(parts, s)[at] for s in (1, 2))
    assert one.tolist() == [0x3F80, 0x3F81] * X.WORLD
    want = [(0x3F80, 0x3F81), (0x3F81, 0x3F81), (0x3F81, 0x3F80), (0x3F81, 0x3F80)] + [(0x3F80, 0x3F80)] * 4
    assert two.tolist() == [w for pair in want for w in pair]
    assert X.compare({**job, "stage": 1})["mismatch"] == 9


def test_fma_is_decided_only_when_the_probes_tell() -> None:
    t = {"normed_words": 9, "fma_differ": 2, "fma1_mismatch": 0, "fma0_mismatch": 2}
    assert X.fma_of(t) == 1 and X.fma_of({**t, "fma1_mismatch": 2, "fma0_mismatch": 0}) == 0
    assert X.fma_of({**t, "fma_differ": 0}) is X.fma_of({**t, "fma0_mismatch": 0}) is X.fma_of({**t, "fma0_mismatch": 1, "fma1_mismatch": 1}) is None
    assert X.fma_of({**t, "normed_words": 0}) is None


# --- launches, bindings, routes, verdicts ----------------------------------------------------------------------------------------

KERNELS = {"ca_comm": "void vllm::cross_device_reduce_{stage}stage<__nv_bfloat16, 8>(...)",
           "symm_mem_comm": "void c10d::symmetric_memory::multimem_all_reduce_kernel<c10::BFloat16, 16>(...)",
           "fi_ar_comm": "void flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel<8, __nv_bfloat16, false, 0>(...)"}


def _call(case: dict) -> dict:
    if case["path"] == X.UNFUSED:
        return {"M": case["M"], "can_use": False, "max_token_num": 0, "path": X.UNFUSED}
    return {"M": case["M"], "can_use": True, "max_token_num": X.FUSED_MAX, "path": X.FUSED, "backend": "mnnvl", "shot": case["shot"],
            "weight_bias": 0.0, "epsilon": X.EPS}


def _rank_case(case: dict, us: float) -> dict:
    if case["phase"] == "fused":
        return {"calls": [_call(case)] * 2, "deterministic": True, "us": us, "kernels": ["void allreduce_fusion_kernel"]}
    comm = case["comm"]
    kernels = [KERNELS[comm].format(stage=case["stage"])] if comm in KERNELS else []
    return {"comms": [comm] * (4 if case["phase"] in ("probe", "again") else 2), "deterministic": True, "us": us, "kernels": kernels}


def _records(cases_: list[dict], us: float = 3.0) -> list[dict]:
    return [{"rank": r, "cases": {c["key"]: _rank_case(c, us) for c in cases_}, "errors": {}, "communicators": {}}
            for r in range(X.WORLD)]


@pytest.mark.parametrize("change,ok", [({}, True), ({"backend": "trtllm"}, False), ({"shot": 1}, False), ({"weight_bias": 1.0}, False),
                                       ({"epsilon": 1e-6}, False), ({"max_token_num": 128}, False), ({"path": X.UNFUSED}, False),
                                       ({"can_use": False}, False)])
def test_a_fused_call_is_on_its_expected_path(change: dict, ok: bool) -> None:
    case = _case("fused", "M11", 11)
    recs = _records([case])
    recs[4]["cases"][case["key"]]["calls"][1] = {**_call(case), **change}
    assert X.launch_ok(case, recs) is ok
    past = _case("fused", "M43", 43)
    assert X.launch_ok(past, _records([past]))
    recs = _records([past])
    recs[0]["cases"][past["key"]]["calls"] = [{**_call(case), "M": 43}] * 2
    assert not X.launch_ok(past, recs)


@pytest.mark.parametrize("phase,M", [("allreduce", 11), ("allreduce", 43), ("probe", 256), ("switch", 64), ("switch", 683)])
def test_an_all_reduce_is_on_its_expected_communicator_and_kernel(phase: str, M: int) -> None:
    case = _case(phase, f"M{M}", M)
    assert X.launch_ok(case, _records([case]))
    recs = _records([case])
    recs[3]["cases"][case["key"]]["comms"][1] = "pynccl_comm" if case["comm"] != "pynccl_comm" else "ca_comm"
    assert not X.launch_ok(case, recs)
    if case["comm"] in ("ca_comm", "symm_mem_comm"):
        recs = _records([case])
        recs[0]["cases"][case["key"]]["kernels"] = ["void vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>(...)"]
        assert not X.launch_ok(case, recs)
        recs[0]["cases"][case["key"]]["kernels"] = {"error": "profiler unavailable"}
        assert X.launch_ok(case, recs)
    for mutate in (lambda r: r[3]["cases"][case["key"]].update(deterministic=False), lambda r: r[6]["errors"].update({case["key"]: "boom"}),
                   lambda r: r[2]["cases"].pop(case["key"]), lambda r: r.pop()):
        recs = _records([case])
        mutate(recs)
        assert not X.launch_ok(case, recs)


def test_the_programs_bind_at_every_glm53_shape() -> None:
    """The fused norm at 4 geometries x 2 contractions, FlashInfer's all-reduce at 5 M, multimem at 4 sizes at the probes' trees,
    the custom all-reduce at 4."""
    cs = X.cases(False, set(X.PHASES))
    got = X.bind_check(cs, list(TREES), True)
    assert got["ok"] and len(got["bindings"]) == 8 + 5 + 4 + 4
    assert got["bindings"]["AllReduceRmsNormMnnvl_v1{WORLD=8,N=6144,SHOT=2,CLUSTER=2,EPS=1e-05,FMA=True}"]
    assert got["bindings"]["AllReduceCustom_v1{WORLD=8,SIZE=4190208,STAGE=2}"]
    assert got["bindings"]["AllReduceMnnvl_v1{WORLD=8,N=258048,SHOT=2}"]
    trees = "(" + ",".join(X.tree_str(t) for t in TREES) + ")"
    assert got["bindings"][f"AllReduceMultimem_v1{{WORLD=8,SIZE=1572864,TREES={trees},FTZ=True}}"]
    assert X.bind_check(cs, None, None)["ok"]


def _doc() -> dict:
    """A GO record at GLM-5.3's statics without words, prefill served: a launch per fused shot and an unfused one, FlashInfer's
    all-reduce, a multimem one, the probes in both spawns, a 2-stage custom all-reduce and an NCCL one under the switch."""
    cs = [_case("fused", "M10", 10), _case("fused", "M42", 42), _case("fused", "M43", 43), _case("allreduce", "M1", 1),
          _case("allreduce", "M64", 64), _case("probe", "M64", 64), _case("again", "M64", 64), _case("switch", "M64", 64),
          _case("switch", "M683", 683)]
    zero = {k: 0 for k in X.TOTALS}
    comps = [{"key": "fused/M10", **zero, "words": 9, "normed_words": 9, "fma0_mismatch": 2, "fma_differ": 2, "reference": 4},
             {"key": "fused/M42", **zero, "words": 9, "normed_words": 9, "fma1_mismatch": 3, "fma_differ": 3, "reference": 4},
             {"key": "fused/M43", **zero}, *[{"key": k, **zero, "words": 9, "reference": 7} for k in
                                             ("allreduce/M1", "allreduce/M64", "probe/M64", "again/M64", "switch/M64")],
             {"key": "switch/M683", **zero}]
    ranks = {s: _records([c for c in cs if c["spawn"] == s]) for s in X.SPAWNS}
    doc = {"schema": X.SCHEMA, "gate": {"ok": True}, "errors": {}, "bind": {"ok": True}, "case_list": cs, "ranks": ranks,
           "comparisons": comps, "probes": {"trees": list(TREES), "agree": True, "ftz": True}}
    doc["fma"] = {str(s): X.fma_of(X._shot_totals(doc, s)) for s in (1, 2)}
    return _derive(doc)


def _derive(doc: dict) -> dict:
    doc["cases"] = X.case_verdicts(doc)
    doc["routes"] = X.routes(doc)
    return doc


def test_each_case_has_its_own_verdict() -> None:
    doc = _doc()
    assert doc["fma"] == {"1": 1, "2": 0} and all(c["ok"] for c in doc["cases"].values())
    assert doc["cases"]["fused/M10"]["path"] == [X.FUSED] * 2 and doc["cases"]["switch/M683"]["path"] == ["pynccl_comm"] * 2
    assert doc["cases"]["probe/M64"]["path"] == ["symm_mem_comm"] * 4
    for mutate, bad in [(lambda d: d["comparisons"][4].update(mismatch=1), "allreduce/M64"),
                        (lambda d: d["comparisons"][8].update(rank_mismatch=1), "switch/M683"),
                        (lambda d: d["comparisons"][0].update(reference=0), "fused/M10"),
                        (lambda d: d["comparisons"][1].update(twin_mismatch=1), "fused/M42"),
                        (lambda d: d["comparisons"][6].update(words=0), "again/M64"),
                        (lambda d: d["comparisons"].pop(2), "fused/M43"),
                        (lambda d: d["ranks"]["served"][1]["errors"].update({"fused/M43": "boom"}), "fused/M43"),
                        (lambda d: d["ranks"]["switch"][2]["cases"]["switch/M64"].update(comms=["symm_mem_comm"] * 2), "switch/M64"),
                        (lambda d: d["ranks"]["served"][7]["cases"]["fused/M10"]["calls"][0].update(shot=2), "fused/M10"),
                        (lambda d: d["fma"].update({"2": 1}), "fused/M42"),
                        (lambda d: d["fma"].update({"1": None}), "fused/M10")]:
        doc = _doc()
        mutate(doc)
        got = X.case_verdicts(doc)
        assert {k for k, c in got.items() if not c["ok"]} == {bad}


def test_prefill_takes_the_served_route_else_the_switch() -> None:
    doc = _doc()
    assert doc["routes"] == {"fused": True, "decode": True, "served": True, "switch": True} and X.prefill_of(doc["routes"]) == "served"
    for mutate in (lambda d: d["probes"].update(trees=None), lambda d: d["probes"].update(agree=False),
                   lambda d: d["probes"].update(ftz=None), lambda d: d["comparisons"][5].update(mismatch=3),
                   lambda d: d["comparisons"][6].update(rank_mismatch=1), lambda d: d["comparisons"][4].update(twin_mismatch=1)):
        doc = _doc()
        mutate(doc)
        _derive(doc)
        assert not doc["routes"]["served"] and X.prefill_of(doc["routes"]) == "switch"
    doc = _doc()
    doc["probes"]["trees"] = None
    doc["comparisons"][7]["mismatch"] = 1
    assert X.prefill_of(_derive(doc)["routes"]) is None
    doc = _doc()
    doc["case_list"] = [c for c in doc["case_list"] if c["phase"] != "again"]
    assert not _derive(doc)["routes"]["served"]


@pytest.mark.parametrize("path,value", [(("gate", "ok"), False), (("errors", "served/rank3/fused/M10"), "boom"), (("bind", "ok"), False),
                                        (("errors", "bind"), "boom"), (("routes", "fused"), False), (("routes", "decode"), False),
                                        (("errors", "served_again/compare/again/M64"), "boom")])
def test_the_verdict_needs_every_check(path: tuple, value) -> None:
    assert X._verdict(_doc())
    doc = copy.deepcopy(_doc())
    node = doc
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    assert not X._verdict(doc)
    short = _doc()
    short["ranks"]["served"] = short["ranks"]["served"][:7]
    assert not X._verdict(short)


def test_the_verdict_ignores_the_route_prefill_does_not_take() -> None:
    doc = _doc()
    doc["errors"]["switch/ranks"] = "boom"
    doc["routes"]["switch"] = False
    assert X._verdict(doc)
    doc = _doc()
    doc["routes"]["served"] = False
    assert X._verdict(doc)
    doc["errors"]["switch/rank2/switch/M64"] = "boom"
    assert not X._verdict(doc)
    doc = _doc()
    doc["routes"].update(served=False, switch=False)
    assert not X._verdict(doc)


def _fake_spawn(flip: dict, trees_of: dict):
    """`spawn_ranks` stood in for: every rank's words from the models (a fused shot's at `FMA`, an all-reduce's by its
    communicator, multimem at `trees_of[spawn]` and FTZ) and its records, the time 3 us served and 2 us switched."""

    def spawn(work, env, custom_ar, cases_):
        spawn_name = next(k for k, (e, ca) in X.SPAWNS.items() if e == env and ca == custom_ar and work.name == k)
        for c in cases_:
            c["out"] = str(work / c["key"].replace("/", "_"))
            if c["phase"] == "fused":
                words = _fused_words(c, FMA.get(c.get("shot"), 1))
            else:
                words = _ar_words(c, trees_of[spawn_name], True)
            _write(c, words, flip.get(c["key"]))
        return _records(cases_, 2.0 if custom_ar else 3.0)

    return spawn


@pytest.mark.slow
def test_main_end_to_end_on_stubbed_ranks(small, tmp_path, monkeypatch, capsys) -> None:
    """`main` with the rank processes stood in for (`_fake_spawn`) and the gate passed: the operands, the probes' screening and
    trees, the comparison jobs through the spawned worker pool at the jobs' own widths and geometries, each shot's decided `fma`,
    the per-case verdicts, the routes, the switch's cost, the sealed record, the saved probe words and the exit status, so a slip
    in the orchestration shows here and not in the 8-GPU node's first hour.  About 40 s: `main` runs three times."""
    monkeypatch.setattr(X, "gate", lambda *a: {"ok": True, "checks": {}})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    flip = {"fused/M3": ("residual", (1, 5))}
    trees_of = {"served": TREES, "served_again": TREES, "switch": TREES}
    monkeypatch.setattr(X, "spawn_ranks", _fake_spawn(flip, trees_of))
    out = tmp_path / "out"
    argv = ["--out", str(out), "--scratch", str(tmp_path), "--workers", "1"]
    assert X.main(argv) == 1
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is False and not doc["errors"] and doc["fma"] == {"1": 1, "2": 0}
    assert {k for k, c in doc["cases"].items() if not c["ok"]} == {"fused/M3"} and doc["totals"]["mismatch"] == 1
    assert set(doc["cases"]) == {"fused/M3", "fused/M171", "fused/special1", "fused/special2", "fused/probe1", "fused/probe2",
                                 "allreduce/M2", "allreduce/M6", "allreduce/M171", "probe/M171", "again/M171", "switch/M171"}
    assert doc["bind"]["ok"] and doc["probes"]["agree"] and doc["probes"]["ftz"] is True
    assert [T.multimem_tree(t, 8) for t in doc["probes"]["trees"]] == list(TREES) and doc["prefill"] == "served"
    assert doc["switch_cost"] == {"M171": round(2 / 3, 3)} and sorted(doc["words"]) == [
        "words/again_M171.npy", "words/allreduce_M171.npy", "words/probe_M171.npy"]
    printed = capsys.readouterr().out
    assert "case fused/M3" in printed and "GLM53-TP8-ALLREDUCE NO-GO" in printed and "no-go cases=['fused/M3']" in printed
    flip.clear()
    assert X.main(argv) == 0
    printed = capsys.readouterr().out.splitlines()[-1]
    assert printed.startswith("GLM53-TP8-ALLREDUCE GO") and "prefill=served" in printed and "ftz=True" in printed
    assert all(X.tree_str(t) in printed for t in TREES)
    trees_of["served_again"] = (BALANCED,) * 8
    assert X.main(argv) == 0
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["prefill"] == "switch" and doc["probes"]["agree"] is False and doc["probes"]["again_words_equal"] == {"again/M171": False}
    assert "prefill=switch trees=none ftz=None" in capsys.readouterr().out.splitlines()[-1]


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
