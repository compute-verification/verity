"""The GLM-5.3 TP8 all-reduce capture without a GPU: vLLM's dispatch at GLM-5.3's and DeepSeek-V4-Pro's boundaries at each setting
(vLLM's defaults, the custom all-reduce off, the switch), the served setting read from the workload, the special and probe
operands, the probes' trees and FTZ read back from words the models make (and refused when the switch's adds are not one f32 tree
per owner), the comparison jobs over rank words made here by the models (a flipped word found and located, a rank that differs, a
shot's contraction decided by the probes), each launch's path, communicator and kernel checks, each spawn's communicators, the
bindings, the routes, the verdicts and the coverage, `tp_allreduce.CAPTURED` against the dispatch and the capture map, and `main`
end to end with the rank processes stood in for (the GPU run is `tests/properties/glm53_tp8_allreduce_capture_gpu.py --out ...` on
an 8x H200 node).  Most tests run at N = 1536 = 3 * 2^9 tokens wide (and 1792 for the h7168 pass): rcp(N) is inexact there as at
GLM-5.3's 6144, so the two contractions of the norm differ on some tokens (at a power of two they never do), and the fused cluster
is 2 CTAs."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3", "deepseek-ai/DeepSeek-V4-Pro")

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tests.properties import glm53_tp8_allreduce_capture_gpu as X
from verity.definitions.numerics import derived_rows as DR
from verity.definitions.registry import tp_allreduce as T
from verity.definitions.registry import tp_allreduce_check as K
from verity.definitions.registry import tp_allreduce_nodes as TN
from verity_vllm.properties.record import REGISTRY, digest_of

N = 1536
N7 = 1792
FMA = {1: 1, 2: 0}
BALANCED = (((0, 1), (2, 3)), ((4, 5), (6, 7)))
CMAP = Path(__file__).parents[2] / "manifests" / "capture-maps" / "glm-moe-dsa-eager-hopper-tp8-v0.json"


def _random_tree(rng):
    nodes = list(range(X.WORLD))
    while len(nodes) > 1:
        i, j = sorted(rng.choice(len(nodes), 2, replace=False))
        b, a = nodes.pop(j), nodes.pop(i)
        nodes.append((a, b))
    return TN.multimem_tree(nodes[0], X.WORLD)


TREES = tuple(_random_tree(np.random.default_rng(s)) for s in range(X.WORLD))


@pytest.fixture
def small(monkeypatch):
    """The capture at N wide: N's FlashInfer budget (170 tokens), FlashInfer's one-shot strategy to 4 tokens in place of N's 42 (so
    that a two-shot launch's probe tokens are few to screen), and token counts on both sides of the boundaries: multimem from 171
    tokens, the custom all-reduce 2-stage there; the h7168 pass at N7 wide, FlashInfer to 146 tokens and multimem at 148."""
    monkeypatch.setattr(X, "HIDDEN", N)
    monkeypatch.setattr(X, "FUSED_MAX", T.fused_max_tokens(N, X.WORLD))
    monkeypatch.setattr(X, "shot_of", lambda M, N=None: 1 if M <= 4 else 2)
    monkeypatch.setattr(X, "SCREEN_CHUNK", 200)
    monkeypatch.setattr(X, "FUSED_M", (3, 171))
    monkeypatch.setattr(X, "SPECIAL", (("special1", 3, 1), ("special2", 6, 2)))
    monkeypatch.setattr(X, "PROBE_M", {1: 4, 2: 6})
    monkeypatch.setattr(X, "AR_M", (2, 6, 171))
    monkeypatch.setattr(X, "PROBE_M_AR", (171,))
    monkeypatch.setattr(X, "NCCL_PROBE_M", ())
    monkeypatch.setattr(X, "SWITCH_M", (171,))
    monkeypatch.setattr(X, "DSV4_HIDDEN", N7)
    monkeypatch.setattr(X, "DSV4_AR_M", (2, 146, 148))
    monkeypatch.setattr(X, "DSV4_PROBE_M", (148,))


@pytest.fixture
def nccl_small(small, monkeypatch):
    """`small` with symmetric memory's cap at 171 tokens of N, so that NCCL takes 172, its NCCL probe's M."""
    monkeypatch.setattr(T, "SYMM_MEM_MAX_BYTES", 171 * N * 2)
    monkeypatch.setattr(X, "AR_M", (2, 171, 172))
    monkeypatch.setattr(X, "NCCL_PROBE_M", (172,))


RING = (3, 1, 4, 0, 7, 5, 2, 6)


def _ring(flat: np.ndarray, chunk: int = 4096, ring=RING, f32: bool = False) -> np.ndarray:
    """A ring all-reduce's words: chunk k of `chunk` elements reduced around `ring` from its (k + 1)-th rank, each hop's add rounded
    to bf16 (`nccl_tree_rows`, NCCL's `FuncSum` on bf16), or in f32 rounded once (`f32`)."""
    W, size = flat.shape
    out = np.empty(size, np.uint16)
    for k, lo in enumerate(range(0, size, chunk)):
        order = [ring[(k + 1 + i) % W] for i in range(W)]
        cols = flat[:, lo:lo + chunk]
        out[lo:lo + chunk] = DR.f2fp_bf16(K.tree_sum(cols, T.chain(W, order), False)) if f32 else K.nccl_tree_rows(cols, order[::-1])
    return out


def _case(phase: str, name: str, M: int, seed: int = 1, served: str = "defaults") -> dict:
    s = X.spawns(served)[X.SPAWN_OF[phase]]
    return {"phase": phase, "case": name, "M": M, "key": f"{phase}/{name}", "seed": seed, "spawn": X.SPAWN_OF[phase],
            "setting": s["setting"], "H": s["H"], **X.expect(phase, M, s["setting"], s["H"])}


def _comms(setting: str) -> dict:
    """A rank's `communicators` at `setting` as vLLM makes them on 8 H200s."""
    custom, symm = T.SETTINGS[setting]
    c = {"fi_ar_comm": True, "qr_comm": False, "fi_pcie_ipc_ar_comm": False, "aiter_ar_comm": False, "ca_comm": custom,
         "symm_mem_comm": symm, "pynccl_comm": True, "custom_all_reduce_enabled": custom,
         "fi_ar_comm.max_workspace_size": T.FI_MAX_BYTES, "fi_ar_comm.max_num_tokens": 0}
    if custom:
        c.update({"ca_comm.max_size": T.CUSTOM_SYMM_MAX_BYTES if symm else T.CUSTOM_MAX_BYTES, "ca_comm.fully_connected": True})
    if symm:
        c.update({"symm_mem_comm.max_size": T.SYMM_MEM_MAX_BYTES, "symm_mem_comm.world_size": X.WORLD})
    return c


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
    elif comm == "pynccl_comm" and X.nccl_modelled(case):
        out = _ring(flat)
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
    """At vLLM's defaults and with the custom all-reduce off: FlashInfer to 42 tokens, multimem to 5461 (64 MiB), NCCL past;
    switched: the custom all-reduce from 43 to 682, always 2-stage, NCCL past."""
    assert X.FUSED_MAX == 42 and [X.expected_path(M) for M in (42, 43)] == [X.FUSED, X.UNFUSED]
    assert [X.shot_of(M) for M in (10, 11)] == [1, 2] and [X.stage_of(M) for M in (21, 22, 43)] == [1, 2, 2]
    for setting in ("defaults", "custom-off"):
        assert [X.expected_comm(M, setting) for M in (1, 21, 22, 42, 43, 256, 2048, 5461, 5462)] == \
            ["fi_ar_comm"] * 4 + ["symm_mem_comm"] * 4 + ["pynccl_comm"]
    assert [X.expected_comm(M, "switch") for M in (42, 43, 682, 683)] == ["fi_ar_comm", "ca_comm", "ca_comm", "pynccl_comm"]
    assert {M: T.mnnvl_grid(M, X.HIDDEN)[1] for M in (16, 17, 33, 34)} == {16: 8, 17: 4, 33: 4, 34: 2}
    assert X.SWITCH_ENV["VLLM_ALLREDUCE_USE_SYMM_MEM"] == "0" and X.SERVED_ENV["VLLM_ALLREDUCE_USE_SYMM_MEM"] == "1"


def test_the_dispatch_at_deepseek_v4_pros_boundaries() -> None:
    """At 7168 wide and vLLM's defaults: FlashInfer one-shot to 9 tokens, two-shot to 36, multimem from 37 to 4681, NCCL past."""
    H = X.DSV4_HIDDEN
    assert T.fused_max_tokens(H, X.WORLD) == 36 and [X.shot_of(M, H) for M in (9, 10)] == [1, 2]
    assert [X.expected_comm(M, "defaults", H) for M in (1, 9, 10, 18, 19, 36, 37, 4681, 4682)] == \
        ["fi_ar_comm"] * 6 + ["symm_mem_comm"] * 2 + ["pynccl_comm"]
    assert {X.expected_comm(M, "defaults", H) for M in X.DSV4_AR_M} == {"fi_ar_comm", "symm_mem_comm"}
    assert {X.shot_of(M, H) for M in X.DSV4_AR_M if M <= 36} == {1, 2}


@pytest.mark.parametrize("H", [1536, 6144, 7168])
def test_the_custom_all_reduce_takes_nothing_beside_symmetric_memory(H: int) -> None:
    """With symmetric memory on, the custom all-reduce's 256 KiB is inside FlashInfer's 0.5 MiB, so at vLLM's defaults every
    all-reduce takes the route it takes with the custom all-reduce off; under the switch it takes everything past FlashInfer below
    8 MiB."""
    assert T.CUSTOM_SYMM_MAX_BYTES < T.FI_MAX_BYTES
    Ms = range(1, 6000)
    assert [X.expected_comm(M, "defaults", H) for M in Ms] == [X.expected_comm(M, "custom-off", H) for M in Ms]
    assert "ca_comm" not in {X.expected_comm(M, "defaults", H) for M in Ms}
    fi = T.fused_max_tokens(H, X.WORLD)
    assert {X.expected_comm(M, "switch", H) for M in range(fi + 1, 6000) if M * H * 2 < T.CUSTOM_MAX_BYTES} == {"ca_comm"}


def test_the_served_setting_follows_the_workload(tmp_path) -> None:
    """``disable_custom_all_reduce`` false or unstated (vLLM's default), or no workload file: vLLM's defaults; true: the custom
    all-reduce off; anything else refused.  The workload in the tree, when there is one, serves vLLM's defaults."""
    p = tmp_path / "w.json"
    for args, setting, source in [({"disable_custom_all_reduce": False}, "defaults", "w.json"),
                                  ({"disable_custom_all_reduce": True}, "custom-off", "w.json"),
                                  ({"max_model_len": 288}, "defaults", "vLLM's default: w.json states none")]:
        p.write_text(json.dumps({"engine": {"args": args}}))
        assert X.workload_setting(p) == {"setting": setting, "disable_custom_all_reduce": args.get("disable_custom_all_reduce"),
                                         "source": source}
    assert X.workload_setting(tmp_path / "absent.json")["setting"] == "defaults"
    p.write_text(json.dumps({"engine": {"args": {"disable_custom_all_reduce": "false"}}}))
    with pytest.raises(ValueError):
        X.workload_setting(p)
    assert X.WORKLOAD_PATH.name == f"{X.WORKLOAD}.json" and X.WORKLOAD_PATH.parent.name == "workloads"
    if X.WORKLOAD_PATH.is_file():
        assert X.workload_setting()["setting"] == "defaults"


def test_the_spawns_and_the_custom_off_variant() -> None:
    """The served spawns run the served setting; the switch and h7168 spawns theirs whatever it is.  Custom-off, the two GO runs'
    setting, launches what they launched: the same cases, seeds and routes as vLLM's defaults."""
    for served, custom in (("defaults", True), ("custom-off", False)):
        sp = X.spawns(served)
        assert set(sp) == {*X.GLM_SPAWNS, "h7168"} and set(X.SPAWN_OF.values()) == set(sp)
        for name in ("served", "served_again"):
            assert sp[name] == {"setting": served, "H": X.HIDDEN, "custom_all_reduce": custom, "symm_mem": True,
                                "env": X.SERVED_ENV}
        assert sp["switch"] == {"setting": "switch", "H": X.HIDDEN, "custom_all_reduce": True, "symm_mem": False,
                                "env": X.SWITCH_ENV}
        assert sp["h7168"] == {"setting": "defaults", "H": X.DSV4_HIDDEN, "custom_all_reduce": True, "symm_mem": True,
                               "env": X.SERVED_ENV}
    route = lambda c: (c["key"], c["seed"], c.get("comm"), c.get("path"), c.get("shot"), c.get("stage"), c["H"])  # noqa: E731
    on, off = (X.cases(False, set(X.PHASES), X.spawns(s)) for s in ("defaults", "custom-off"))
    assert [route(c) for c in on] == [route(c) for c in off]
    assert {c["setting"] for c in off if c["spawn"] in ("served", "served_again")} == {"custom-off"}
    glm = X.cases(False, {"fused", "allreduce", "probe", "again", "switch"})
    assert [route(c) for c in glm] == [route(c) for c in on if c["spawn"] != "h7168" and c["phase"] not in X.NCCL_PROBE_PHASES]


def test_the_cases_cover_every_route() -> None:
    geoms = {(1, 8), (2, 8), (2, 4), (2, 2)}
    for quick in (False, True):
        cs = X.cases(quick, set(X.PHASES))
        assert {(c["shot"], c["cluster"]) for c in cs if c.get("path") == X.FUSED} == geoms and X.UNFUSED in {c.get("path") for c in cs}
        ar = [c for c in cs if c["phase"] == "allreduce"]
        assert {c["comm"] for c in ar} - {"pynccl_comm"} == {"fi_ar_comm", "symm_mem_comm"} and {c["shot"] for c in ar if c["shot"]} == {1, 2}
        assert 1 in {c["M"] for c in ar} and 256 in {c["M"] for c in ar}
        assert {c["M"] for c in ar if c["comm"] == "pynccl_comm"} == (set() if quick else set(X.NCCL_PROBE_M))
        assert {c["M"] for c in ar if c["comm"] == "symm_mem_comm"} >= ({43, 256} if quick else {43, 2048, 2049, 4096, 5461})
        assert {(c["comm"], c["stage"]) for c in cs if c["phase"] == "switch"} == {("ca_comm", 2), ("pynccl_comm", None)}
        probes = {c["M"]: c["seed"] for c in cs if c["phase"] == "probe"}
        assert probes and probes == {c["M"]: c["seed"] for c in cs if c["phase"] == "again"} and max(probes) == 5461
        nccl = {c["M"]: c["seed"] for c in cs if c["phase"] == "nccl_probe"}
        assert set(nccl) == set(X.NCCL_PROBE_M) == {5462, 8192} and nccl == {c["M"]: c["seed"] for c in cs if c["phase"] == "nccl_again"}
        assert all(c["comm"] == "symm_mem_comm" for c in cs if c["phase"] in X.PROBE_PHASES)
        assert all(c["comm"] == "pynccl_comm" and X.nccl_modelled(c) for c in cs if c["phase"] in X.NCCL_PROBE_PHASES)
        assert {c["spawn"] for c in cs} == set(X.spawns("defaults")) and len({c["key"] for c in cs}) == len(cs)
        assert len({c["seed"] for c in cs if c["phase"] not in ("again", "nccl_again")}) == len(cs) - len(probes) - len(nccl)
        h = [c for c in cs if c["spawn"] == "h7168"]
        assert {c["H"] for c in h} == {X.DSV4_HIDDEN} and {c["H"] for c in cs if c["spawn"] != "h7168"} == {X.HIDDEN}
        assert {c["comm"] for c in h} == {"fi_ar_comm", "symm_mem_comm"} and {c["shot"] for c in h if c["shot"]} == {1, 2}
        assert {c["M"] for c in h if c["phase"] == "h7168_probe"} == set(X.DSV4_PROBE_M) <= {c["M"] for c in h if c["phase"] == "h7168"}
        assert "ca_comm" not in {c.get("comm") for c in cs if c["phase"] != "switch"}
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
    for M, H in [(M, X.HIDDEN) for M in X.PROBE_M_AR] + [(M, X.DSV4_HIDDEN) for M in X.DSV4_PROBE_M]:
        assert all((hi - lo) % X.PERIOD == 0 for lo, hi, _ in T.multimem_owners(X.WORLD, M * H))


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


def _read_probes_back(tmp_path, cs: list[dict]) -> None:
    for c in cs:
        c.update(X.make_operands(tmp_path, c))
    for c in [c for c in cs if c["phase"] in ("probe", "h7168_probe")]:
        parts = np.load(c["ops"]["parts"]).reshape(X.WORLD, -1)
        assert parts.shape[1] == c["M"] * c["H"]
        if c["phase"] == "probe":
            again = next(a for a in cs if a["phase"] == "again" and a["M"] == c["M"])
            assert np.array_equal(parts, np.load(again["ops"]["parts"]).reshape(X.WORLD, -1))
        out = K.multimem_rows(parts, TREES, False)
        assert X.decode_probe(out, parts.shape[1])["trees"] == list(TREES) and X.decide_ftz(out, parts)["ftz"] is False


def test_the_capture_probes_read_trees_at_glm53s_sizes(tmp_path) -> None:
    """The run's own probe operands at GLM-5.3's and DeepSeek-V4-Pro's widths and seeds, to the 2048 tokens the eager workload
    batches, give back distinct trees per owner, and the `again` probes' operands are the first probes'."""
    _read_probes_back(tmp_path, [c for c in X.cases(False, {"probe", "again", "h7168_probe"}) if c["M"] <= X.STEP_TOKENS])


@pytest.mark.slow
def test_the_multimem_probe_reads_trees_at_64_mib(tmp_path) -> None:
    """The probe at 5461 tokens, symmetric memory's last M (64 MiB, about 10 s and 2 GB): its operands at the run's seed, read back
    through the model at 8 random trees and IEEE adds.  Catches an owner slice of `multimem_owners` that is not whole periods, or a
    seed or layout change, at the size that lets `CAPTURED` reach 5461; the smaller probes' tests can't, since the slices'
    remainders depend on the size."""
    cs = [c for c in X.cases(False, {"probe", "again"}) if c["M"] == max(X.PROBE_M_AR)]
    assert [c["M"] for c in cs] == [5461, 5461]
    _read_probes_back(tmp_path, cs)


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

@pytest.mark.slow
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


@pytest.mark.slow
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
                                           ("probe", 171, "multimem"), ("switch", 171, "custom"), ("h7168", 146, "mnnvl"),
                                           ("h7168", 148, "multimem"), ("h7168_probe", 148, "multimem")])
def test_an_all_reduce_takes_the_kernel_words(small, tmp_path, phase: str, M: int, model: str) -> None:
    job = _ar_job(tmp_path, M, phase)
    H = N7 if phase.startswith("h7168") else N
    assert job["model"] == model
    t = X.compare(job)
    assert "error" not in t, t
    sample = X.multimem_sample(M * H) if model == "multimem" else X.custom_sample(M * H)
    assert (t["words"], t["mismatch"], t["rank_mismatch"], t["twin_mismatch"], t["reference"]) == (M * H, 0, 0, 0, len(sample))
    t = X.compare(_ar_job(tmp_path, M, phase, flip=("out", (M - 1, H - 2))))
    assert t["mismatch"] == 1 and t["first_mismatch"]["at"] == [M * H - 2]


def test_a_spawn_takes_its_own_pinned_trees(small, tmp_path) -> None:
    """h7168's multimem launches are modelled at its own probe's trees (`pins`), GLM-5.3's at theirs."""
    glm, h = (_made(tmp_path, _case(p, "M", M)) for p, M in (("allreduce", 171), ("h7168", 148)))
    jobs = X.jobs_of([glm, h], TREES, True, {"h7168": ((BALANCED,) * 8, False)})
    assert [(j["trees"], j["ftz"]) for j in jobs] == [(TREES, True), ((BALANCED,) * 8, False)]
    assert X.jobs_of([h], TREES, True, {"h7168": (None, None)})[0]["model"] is None


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


# --- NCCL's runs -----------------------------------------------------------------------------------------------------------------

def test_the_nccl_probe_runs_its_periods_over_the_flat_tensor() -> None:
    """Every `PERIOD` elements from 0 hold the 56 pairs then the specials, each period's specials one rank on, whatever the owners."""
    size = 64 * 20 + 32
    parts = X.nccl_probe_parts(np.random.default_rng(0), size)
    for k in range(size // X.PERIOD):
        at = k * X.PERIOD
        for f, big in enumerate(X.BIG):
            for p, (i, j) in enumerate(X.PAIRS):
                e = at + f * len(X.PAIRS) + p
                assert parts[i, e] == big and parts[j, e] == big | 0x8000
        assert (parts[:, at + 2 * len(X.PAIRS) + 3] == X.NEG0).all() and parts[k % 8, at + 2 * len(X.PAIRS)] == 0x0001


@pytest.mark.parametrize("f32", [False, True])
def test_the_nccl_decoder_recovers_a_simulated_ring(f32: bool) -> None:
    """A ring all-reduce whose chunks start one rank further round a random ring each, at bf16 adds per hop or f32 adds: every
    chunk one run, its chain the ring from its start (the first two added in either order), and the model that made the words
    the one that fits every word."""
    size, chunk = 6144, 256
    parts = X.nccl_probe_parts(np.random.default_rng(1), size)
    ring = tuple(int(r) for r in np.random.default_rng(2).permutation(X.WORLD))
    out = _ring(parts, chunk, ring, f32)
    dec = X.decode_runs(out, size)
    assert dec["undecoded"] == 0 and [r[:2] for r in dec["runs"]] == [[lo, lo + chunk] for lo in range(0, size, chunk)]
    for k, (_, _, t) in enumerate(dec["runs"]):
        want = [ring[(k + 1 + i) % X.WORLD] for i in range(X.WORLD)][::-1]
        got = dec["chains"][t]
        assert list(got[:-2]) == want[:-2] and set(got[-2:]) == set(want[-2:]), k
    assert len(dec["trees"]) == X.WORLD
    fit = X.nccl_fit(out, parts, dec)
    assert fit["model"] == ("f32" if f32 else "bf16") and fit["fit"][fit["model"]] == 0
    assert np.array_equal(X.nccl_rows(parts, dec, fit["model"]), out)
    if f32:
        assert fit["fit"]["bf16"] > 0


def test_a_run_edge_inside_a_period_is_undecoded() -> None:
    """Chunks of 96 elements put an edge inside every third period: those periods name no tree and are counted; chunks of 128 put
    none inside one."""
    size = 6144
    parts = X.nccl_probe_parts(np.random.default_rng(3), size)
    dec = X.decode_runs(_ring(parts, 96), size)
    bad = [r for r in dec["runs"] if r[2] < 0]
    assert bad and bad[0][:2] == [64, 128] and dec["undecoded"] == sum(hi - lo for lo, hi, _ in bad)
    assert X.decode_runs(_ring(parts, 128), size)["undecoded"] == 0
    assert X.chain_of(((0, 1), (2, 3))) is None and X.chain_of((((0, 1), 2), 3)) == (3, 2, 1, 0)


def _nccl_cases(tmp_path, again_ring=RING) -> tuple[dict, dict, dict]:
    probe, again, ar = (_made(tmp_path, _case(p, "M172", 172)) for p in ("nccl_probe", "nccl_again", "allreduce"))
    for c, ring in ((probe, RING), (again, again_ring), (ar, RING)):
        _write(c, {"out": _ring(np.load(c["ops"]["parts"]).reshape(X.WORLD, -1), 4096, ring).reshape(172, N)})
    return probe, again, ar


def test_an_nccl_launch_takes_its_probes_runs(nccl_small, tmp_path) -> None:
    """NCCL's launch at the NCCL probe's M is modelled at the runs both probes decoded: every word, every rank and the reference
    (`AllReduceNcclTreeCoordinate_v1` at each run's chain) equal, a flipped word found; an again probe whose words differ, or a
    launch at an M no probe decoded, is not modelled."""
    probe, again, ar = _nccl_cases(tmp_path)
    assert {c["comm"] for c in (probe, again, ar)} == {"pynccl_comm"} and all(X.nccl_modelled(c) for c in (probe, again, ar))
    got = X.nccl_analysis([probe, again])
    assert got["agree"] and got["again_words_equal"] == {"nccl_again/M172": True} and set(got["runs"]) == {"172"}
    dec = got["runs"]["172"]
    assert dec["fit"] == "bf16" and len(dec["runs"]) == 65 and len(dec["trees"]) == X.WORLD and None not in dec["chains"]
    job = X.jobs_of([ar], nccl=got["runs"])[0]
    assert job["model"] == "nccl"
    t = X.compare(job)
    assert "error" not in t, t
    assert (t["words"], t["mismatch"], t["rank_mismatch"], t["twin_mismatch"]) == (172 * N, 0, 0, 0)
    assert t["reference"] == len(X.nccl_sample(172 * N, dec)) > 65
    _write(ar, {"out": _ring(np.load(ar["ops"]["parts"]).reshape(X.WORLD, -1)).reshape(172, N)}, ("out", (171, N - 2)))
    t = X.compare(job)
    assert t["mismatch"] == 1 and t["first_mismatch"]["at"] == [172 * N - 2]
    assert X.jobs_of([ar], nccl={})[0]["model"] is None
    _, again, _ = _nccl_cases(tmp_path, (0, 1, 2, 3, 4, 5, 6, 7))
    assert not X.nccl_analysis([probe, again])["agree"]


def test_the_nccl_log_is_read() -> None:
    """NCCL's version, each collective's tuning by its bytes (named, numbered or with its channels), and device 0's init lines."""
    import tempfile

    lines = {0: ["h:1:1 [0] NCCL INFO NCCL version 2.27.5+cuda12.9", "h:1:1 [0] NCCL INFO Channel 00/24 : 0 1 2 3 4 5 6 7",
                 "h:1:1 [0] NCCL INFO AllReduce: 67117056 Bytes -> Algo RING proto SIMPLE channel{Lo..Hi}={0..23}",
                 "h:1:1 [0] NCCL INFO AllReduce: 67117056 Bytes -> Algo RING proto SIMPLE channel{Lo..Hi}={0..23}",
                 "h:1:1 [0] NCCL INFO AllReduce: 4 Bytes -> Algo 0 proto 0 time 9.5"],
             3: ["h:4:4 [3] NCCL INFO Channel 00/24 : 0 1 2 3 4 5 6 7",
                 "h:4:4 [3] NCCL INFO AllReduce: 67117056 Bytes -> Algo RING proto SIMPLE channel{Lo..Hi}={0..23}"]}
    with tempfile.TemporaryDirectory() as d:
        for r, ls in lines.items():
            (Path(d) / f"nccl.h.{r}.log").write_text("\n".join(ls) + "\n")
        got = X.nccl_log(Path(d))
    assert got["files"] == 2 and got["version"] == ["2.27.5+cuda12.9"] and got["init"] == ["NCCL version 2.27.5+cuda12.9", "Channel 00/24 : 0 1 2 3 4 5 6 7"]
    assert got["tuning"] == [{"func": "AllReduce", "bytes": 4, "algo": "TREE", "proto": "LL", "channels": None, "calls": 1, "devs": [0]},
                             {"func": "AllReduce", "bytes": 67117056, "algo": "RING", "proto": "SIMPLE", "channels": "0..23", "calls": 3,
                              "devs": [0, 3]}]
    assert X.nccl_choice({"served": got}, 5462 * 6144 * 2) == [{"algo": "RING", "proto": "SIMPLE", "channels": "0..23", "calls": 3}]


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


def _records(cases_: list[dict], us: float = 3.0, setting: str = "defaults") -> list[dict]:
    return [{"rank": r, "cases": {c["key"]: _rank_case(c, us) for c in cases_}, "errors": {}, "communicators": _comms(setting)}
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


@pytest.mark.parametrize("phase,M", [("allreduce", 11), ("allreduce", 43), ("probe", 256), ("switch", 64), ("switch", 683),
                                     ("h7168", 36), ("h7168", 37), ("h7168_probe", 37), ("allreduce", 5461), ("allreduce", 8192),
                                     ("nccl_probe", 5462), ("nccl_again", 8192)])
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
    """The fused norm at 4 geometries x 2 contractions, FlashInfer's all-reduce at 5 M, multimem at 7 sizes at the probes' trees
    (to 5461 tokens), the custom all-reduce at 4, NCCL at none; at DeepSeek-V4-Pro's width FlashInfer's at 6 M and multimem at 3."""
    cs = X.cases(False, set(X.PHASES))
    glm, h = [c for c in cs if c["spawn"] in X.GLM_SPAWNS], [c for c in cs if c["spawn"] == "h7168"]
    got = X.bind_check(glm, list(TREES), True)
    assert got["ok"] and len(got["bindings"]) == 8 + 5 + 7 + 4
    assert got["bindings"]["AllReduceRmsNormMnnvl_v1{WORLD=8,N=6144,SHOT=2,CLUSTER=2,EPS=1e-05,FMA=True}"]
    assert got["bindings"]["AllReduceCustom_v1{WORLD=8,SIZE=4190208,STAGE=2}"]
    assert got["bindings"]["AllReduceMnnvl_v1{WORLD=8,N=258048,SHOT=2}"]
    trees = "(" + ",".join(X.tree_str(t) for t in TREES) + ")"
    assert got["bindings"][f"AllReduceMultimem_v1{{WORLD=8,SIZE=1572864,TREES={trees},FTZ=True}}"]
    assert X.bind_check(glm, None, None)["ok"]
    got = X.bind_check(h, list(TREES), False)
    assert got["ok"] and len(got["bindings"]) == 6 + 3 and got["bindings"]["AllReduceMnnvl_v1{WORLD=8,N=258048,SHOT=2}"]
    assert got["bindings"][f"AllReduceMultimem_v1{{WORLD=8,SIZE={37 * 7168},TREES={trees},FTZ=False}}"]


def test_each_spawn_holds_its_settings_communicators() -> None:
    """Every rank's communicators at its spawn's setting pass; the custom all-reduce on when it should be off or at the cap it
    has without symmetric memory, symmetric memory off, another all-reduce on, or a rank missing, fail."""
    for served in ("defaults", "custom-off"):
        for name, s in X.spawns(served).items():
            assert X.dispatch_check(s, [_comms(s["setting"])] * X.WORLD)["ok"], name
    sp = X.spawns("defaults")
    for name, change in [("served", {"ca_comm.max_size": T.CUSTOM_MAX_BYTES}), ("served", {"symm_mem_comm": False}),
                         ("served", {"custom_all_reduce_enabled": False}), ("served", {"qr_comm": True}),
                         ("switch", {"ca_comm.max_size": T.CUSTOM_SYMM_MAX_BYTES}), ("h7168", {"fi_ar_comm.max_workspace_size": 1}),
                         ("h7168", {"ca_comm.fully_connected": False})]:
        comms = [_comms(sp[name]["setting"])] * X.WORLD
        comms[5] = {**comms[5], **change}
        got = X.dispatch_check(sp[name], comms)
        assert not got["ok"] and set(got["differ"]) == {"rank5"} and set(got["differ"]["rank5"]) == set(change), (name, change)
    off = X.spawns("custom-off")["served"]
    assert not X.dispatch_check(off, [_comms("defaults")] * X.WORLD)["ok"]
    assert not X.dispatch_check(sp["served"], [_comms("defaults")] * 7)["ok"]
    assert not X.dispatch_check(sp["served"], [None] * X.WORLD)["ok"]


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
    sp = X.spawns("defaults")
    ranks = {s: _records([c for c in cs if c["spawn"] == s], 3.0, sp[s]["setting"]) for s in X.GLM_SPAWNS}
    doc = {"schema": X.SCHEMA, "gate": {"ok": True}, "errors": {}, "bind": {"ok": True}, "case_list": cs, "ranks": ranks,
           "comparisons": comps, "probes": {"trees": list(TREES), "agree": True, "ftz": True},
           "dispatch": {s: X.dispatch_check(sp[s], [r["communicators"] for r in recs]) for s, recs in ranks.items()}}
    doc["fma"] = {str(s): X.fma_of(X._shot_totals(doc, s)) for s in (1, 2)}
    return _derive(doc)


def _derive(doc: dict) -> dict:
    doc["cases"] = X.case_verdicts(doc)
    doc["routes"] = X.routes(doc)
    doc["h7168"] = X.h7168_verdict(doc)
    doc["nccl"] = X.nccl_verdict(doc)
    return doc


def _with_nccl(doc: dict, ok: bool = True) -> dict:
    """`doc` with NCCL's launch at 5462 tokens and its two NCCL probes, decoded into one run on a chain."""
    cs = [_case("allreduce", "M5462", 5462), _case("nccl_probe", "M5462", 5462), _case("nccl_again", "M5462", 5462)]
    zero = {k: 0 for k in X.TOTALS}
    doc["case_list"] += cs
    doc["comparisons"] += [{"key": c["key"], **zero, "words": 9, "reference": 7, "mismatch": int(not ok and i == 0)}
                           for i, c in enumerate(cs)]
    sp = X.spawns("defaults")
    for s in ("served", "served_again"):
        doc["ranks"][s] = _records([c for c in doc["case_list"] if c["spawn"] == s])
        doc["dispatch"][s] = X.dispatch_check(sp[s], [r["communicators"] for r in doc["ranks"][s]])
    tree = T.chain(X.WORLD, RING)
    doc["nccl_probes"] = {"agree": True, "runs": {"5462": {"trees": [tree], "chains": [X.chain_of(tree)], "runs": [[0, 5462 * 6144, 0]],
                                                           "fit": "bf16"}}}
    doc["nccl_log"] = {"served": {"tuning": [{"func": "AllReduce", "bytes": 5462 * 6144 * 2, "algo": "RING", "proto": "SIMPLE",
                                              "channels": "0..23", "calls": 30, "devs": [0]}]}}
    return _derive(doc)


def _with_h7168(doc: dict, ok: bool = True) -> dict:
    """`doc` with the h7168 pass: FlashInfer's all-reduce at 36 tokens, a multimem one and its probe at 37."""
    cs = [_case("h7168", "M36", 36), _case("h7168", "M37", 37), _case("h7168_probe", "M37", 37)]
    zero = {k: 0 for k in X.TOTALS}
    doc["case_list"] += cs
    doc["comparisons"] += [{"key": c["key"], **zero, "words": 9, "reference": 7, "mismatch": int(not ok and i == 1)}
                           for i, c in enumerate(cs)]
    doc["ranks"]["h7168"] = _records(cs)
    doc["dispatch"]["h7168"] = X.dispatch_check(X.spawns("defaults")["h7168"], [r["communicators"] for r in doc["ranks"]["h7168"]])
    doc["h7168_probes"] = {"trees": list(TREES), "agree": True, "ftz": True}
    doc["h7168_bind"] = {"ok": True}
    return _derive(doc)


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
    doc["comparisons"][8]["rank_mismatch"] = 1
    assert X.prefill_of(_derive(doc)["routes"]) == "switch" and not doc["cases"]["switch/M683"]["ok"]
    doc = _doc()
    doc["probes"]["trees"] = None
    doc["comparisons"][7]["mismatch"] = 1
    assert X.prefill_of(_derive(doc)["routes"]) is None
    doc = _doc()
    doc["case_list"] = [c for c in doc["case_list"] if c["phase"] != "again"]
    assert not _derive(doc)["routes"]["served"]


@pytest.mark.parametrize("path,value", [(("gate", "ok"), False), (("errors", "served/rank3/fused/M10"), "boom"), (("bind", "ok"), False),
                                        (("errors", "bind"), "boom"), (("routes", "fused"), False), (("routes", "decode"), False),
                                        (("errors", "served_again/compare/again/M64"), "boom"), (("dispatch", "served", "ok"), False),
                                        (("dispatch", "served_again", "ok"), False)])
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
    doc["dispatch"]["switch"]["ok"] = False
    assert X._verdict(doc)
    doc = _doc()
    doc["routes"]["served"] = False
    assert X._verdict(doc)
    doc["errors"]["switch/rank2/switch/M64"] = "boom"
    assert not X._verdict(doc)
    doc = _doc()
    doc["routes"]["served"] = False
    doc["dispatch"]["switch"]["ok"] = False
    assert not X._verdict(doc)
    doc = _doc()
    doc["routes"].update(served=False, switch=False)
    assert not X._verdict(doc)


def test_the_h7168_verdict_stands_apart_from_glm53s() -> None:
    """h7168 is GO on its own launches, probe, bindings and dispatch, and GLM-5.3's verdict reads none of them; skipped when it
    did not run."""
    assert _doc()["h7168"] == {"ok": None}
    doc = _with_h7168(_doc())
    assert doc["h7168"] == {"ok": True, "trees": list(TREES), "ftz": True, "same_as_glm53": True} and X._verdict(doc)
    for mutate in (lambda d: d["comparisons"][-2].update(mismatch=1), lambda d: d["h7168_probes"].update(agree=False),
                   lambda d: d["h7168_bind"].update(ok=False), lambda d: d["dispatch"]["h7168"].update(ok=False),
                   lambda d: d["errors"].update({"h7168/rank1/h7168/M36": "boom"}), lambda d: d["ranks"]["h7168"].pop()):
        doc = _with_h7168(_doc())
        mutate(doc)
        _derive(doc)
        assert doc["h7168"]["ok"] is False and X._verdict(doc)
    doc = _with_h7168(_doc())
    doc["h7168_probes"]["trees"] = [BALANCED] * 8
    assert _derive(doc)["h7168"]["same_as_glm53"] is False and doc["h7168"]["ok"]


def test_the_nccl_verdict_stands_apart_from_glm53s() -> None:
    """NCCL is GO on its probes' runs, its launches at their M and both served spawns, with its runs, model, chains and tuning per
    M, and GLM-5.3's verdict reads none of it; skipped when no NCCL probe ran.  Its launches are covered only when it is GO."""
    assert _doc()["nccl"] == {"ok": None}
    doc = _with_nccl(_doc())
    assert doc["nccl"]["ok"] is True and X._verdict(doc)
    m = doc["nccl"]["M"]["5462"]
    assert (m["runs"], m["model"], m["chains"]) == (1, "bf16", [X.chain_of(T.chain(X.WORLD, RING))])
    assert m["tuning"] == [{"algo": "RING", "proto": "SIMPLE", "channels": "0..23", "calls": 30}]
    assert {"setting": "defaults", "N": 6144, "route": "pynccl_comm", "M": [5462]} in X.coverage(doc, True)
    for mutate in (lambda d: d["comparisons"][-3].update(mismatch=1), lambda d: d["nccl_probes"].update(agree=False),
                   lambda d: d["dispatch"]["served_again"].update(ok=False), lambda d: d["ranks"]["served_again"].pop(),
                   lambda d: d["ranks"]["served"][2]["cases"]["nccl_probe/M5462"].update(comms=["symm_mem_comm"] * 2)):
        doc = _with_nccl(_doc())
        mutate(doc)
        _derive(doc)
        assert doc["nccl"]["ok"] is False and "pynccl_comm" not in {r["route"] for r in X.coverage(doc, True)}
    doc = _with_nccl(_doc())
    doc["comparisons"][-3]["mismatch"] = 1
    assert X._verdict(_derive(doc)) and not doc["cases"]["allreduce/M5462"]["ok"]


def test_the_coverage_is_what_a_go_record_shows() -> None:
    """Per setting, hidden size and route, the token counts of a GO record's launches: the fused calls on FlashInfer's path, its
    and multimem's and the custom all-reduce's, not NCCL's or the norm module's; a route with a NO-GO launch, or a family whose
    verdict fails, has none."""
    doc = _with_h7168(_doc())
    rows = X.coverage(doc, True)
    assert rows == [{"setting": "defaults", "N": 6144, "route": "fi_ar_comm", "M": [1]},
                    {"setting": "defaults", "N": 6144, "route": "fused", "M": [10, 42]},
                    {"setting": "defaults", "N": 6144, "route": "symm_mem_comm", "M": [64]},
                    {"setting": "defaults", "N": 7168, "route": "fi_ar_comm", "M": [36]},
                    {"setting": "defaults", "N": 7168, "route": "symm_mem_comm", "M": [37]},
                    {"setting": "switch", "N": 6144, "route": "ca_comm", "M": [64]}]
    assert [r for r in X.coverage(doc, False) if r["N"] == 6144] == []
    doc = _with_h7168(_doc(), ok=False)
    assert [r for r in X.coverage(doc, True) if r["N"] == 7168] == []
    doc = _doc()
    doc["comparisons"][4]["mismatch"] = 1
    _derive(doc)
    assert "symm_mem_comm" not in {r["route"] for r in X.coverage(doc, True)}


# --- what is covered --------------------------------------------------------------------------------------------------------------

def _bind_captured(c: T.Captured, M: int):
    from verity.circuits.defs import bind

    e = X.expect("fused" if c.route == "fused" else "allreduce", M, c.setting, c.N)
    size = M * c.N
    if c.route == "fused":
        return bind(T.AllReduceRmsNormMnnvl, WORLD=X.WORLD, N=c.N, SHOT=e["shot"], CLUSTER=e["cluster"], EPS=X.EPS, **c.statics)
    if c.route == "fi_ar_comm":
        return bind(T.AllReduceMnnvl, WORLD=X.WORLD, N=size, SHOT=e["shot"], **c.statics)
    if c.definition == "AllReduceMultimem_v1":
        return bind(T.AllReduceMultimem, WORLD=X.WORLD, SIZE=size, **c.statics)
    return bind(T.AllReduceCustom, WORLD=X.WORLD, SIZE=size, **c.statics)


def _node_of(c: T.Captured) -> dict:
    """The node record whose routes `c` is, by its node tree."""
    return next(r for r in TN.NODE_RECORDS if TN.node_key(r)[2:] == c.node)


def test_each_captured_route_is_the_dispatchs_at_its_setting() -> None:
    """Every H200 `tp_allreduce.CAPTURED` entry is, at each M it covers, the route vLLM's dispatch takes at its setting (the fused path to
    FlashInfer's budget, the custom all-reduce 2-stage throughout), cites exactly the GO runs at that setting (a node record's
    entry, its route's), and its Definition binds at its first and last M with what they read (the node's tree on every owner,
    IEEE adds, FMA in both shots); `covered` names it there, a node record's only at its device and stated tree (or none stated,
    `UNSTATED_NODE`'s), and nowhere past it."""
    assert X.tree_str(T.GLM53_H200_TREE) == "((((((0,(1,2)),3),5),6),4),7)"
    assert T.CAPTURE_RUNS == ("art:4a6503ce29cbd49b808dffe32566e4f74e3933199fdd347dec4a2238e73058f7",
                              "art:aef3c209534e1cf90640cd9bda04a931d041f8553437e917346d6e24283507ea",
                              "art:6562ff46905ed10fe4372ee070a97ecc6d36725f313e4834a44e78fb3acb3f7a")
    runs = {"custom-off": T.CAPTURE_RUNS[:2], "defaults": T.CAPTURE_RUNS[2:], "switch": T.CAPTURE_RUNS}
    assert [c.definition for c in T.CAPTURED if c.setting not in T.SETTINGS] == ["AllReduceNcclTree_v1"]
    for c in (c for c in T.CAPTURED if c.setting in T.SETTINGS):
        assert c.N in (X.HIDDEN, X.DSV4_HIDDEN) and (c.world, c.transport) == (X.WORLD, None)
        if c.node is None:
            assert c.runs == runs[c.setting] and c.device is None
            facts = {}
        else:
            rec = _node_of(c)
            assert c.runs == next(tuple(r["run_records"]) for r in rec["routes"] if (r["setting"], r["N"], r["route"]) == c[:3])
            assert c.device == TN.node_key(rec)[0] == ("9.0", 132) and c.definition == "AllReduceMultimem_v1"
            facts = {"device": c.device, "trees": c.node[0], "ftz": c.node[1]}
            assert (T.covered(c.setting, c.N, c.route, c.lo) is c) == (c.node == T.UNSTATED_NODE[1:])
        for M in range(c.lo, c.hi + 1):
            if c.route == "fused":
                assert X.expected_path(M) == X.FUSED, (c, M)
            else:
                assert X.expected_comm(M, c.setting, c.N) == c.route, (c, M)
            if c.route == "ca_comm":
                assert X.stage_of(M, c.N) == c.statics["STAGE"], (c, M)
            assert T.covered(c.setting, c.N, c.route, M, **facts) is c
        assert T.covered(c.setting, c.N, c.route, c.lo - 1, **facts) is None
        assert T.covered(c.setting, c.N, c.route, c.hi + 1, **facts) is None
        for M in (c.lo, c.hi):
            assert _bind_captured(c, M).id.split("{")[0] == c.definition


def test_vllms_defaults_are_covered_through_the_captured_range() -> None:
    """The GO capture covers every GLM-5.3 call at vLLM's defaults through 2048 tokens, and DeepSeek-V4-Pro's standalone
    all-reduces there through 2048. It covers neither width past that range, DeepSeek-V4-Pro at another setting, nor NCCL;
    a setting `SETTINGS` does not name is refused."""
    routes = ("fused", "fi_ar_comm", "ca_comm", "symm_mem_comm", "pynccl_comm")
    for M in (1, 9, 10, 11, 36, 37, 42, 43, 256, 682, 683, 2048, 5461, 5462):
        for n in (X.HIDDEN, X.DSV4_HIDDEN):
            expected = X.expected_comm(M, "defaults", n)
            assert (T.covered("defaults", n, expected, M) is not None) == (M <= 2048)
            assert all(T.covered("defaults", n, r, M) is None for r in routes if r != expected and r != "fused")
        assert (T.covered("defaults", X.HIDDEN, "fused", M) is not None) == (M <= X.FUSED_MAX)
        assert T.covered("defaults", X.DSV4_HIDDEN, "fused", M) is None
        assert all(T.covered(s, X.DSV4_HIDDEN, r, M) is None for s in ("custom-off", "switch") for r in routes)
        assert all(T.covered(s, X.HIDDEN, "pynccl_comm", M) is None for s in T.SETTINGS)
    with pytest.raises(ValueError, match="setting 'served'"):
        T.covered("served", X.HIDDEN, "fi_ar_comm", 1)


def test_the_capture_map_claims_no_all_reduce_capture_has_not_covered() -> None:
    """The capture map's `tp-allreduce` and `fused-allreduce-rmsnorm` Programs may be "exists", and their Definitions' conformance
    more than not-yet-established, only when `CAPTURED` covers every call GLM-5.3's served engine makes at the workload's setting
    (`workload_setting`) up to the 2048 tokens a step batches: the fused path to FlashInfer's budget and each all-reduce on the route
    `expected_comm` names, NCCL's never.  Every H200 `CAPTURED` Definition is one its Program lists, and both notes cite `CAPTURED`."""
    from verity.circuits.defs import REGISTRY as DEFS
    from verity.circuits.defs import conformance_label

    cmap = json.loads(CMAP.read_text())
    setting = X.workload_setting()["setting"]
    Ms = range(1, X.STEP_TOKENS + 1)
    calls = {"fused-allreduce-rmsnorm": [("fused", M) for M in Ms if X.expected_path(M) == X.FUSED],
             "tp-allreduce": [(X.expected_comm(M, setting), M) for M in Ms]}
    for pid, cs in calls.items():
        p = cmap["programs"][pid]
        uncovered = [(r, M) for r, M in cs if T.covered(setting, X.HIDDEN, r, M) is None]
        assert "tp_allreduce.CAPTURED" in p["note"], pid
        if uncovered:
            assert p["status"] != "exists", (pid, setting, uncovered[:3])
            assert all(conformance_label(DEFS.defs[d].conformance) == "not-yet-established" for d in p["definitions"]), pid
        else:
            assert all(conformance_label(DEFS.defs[d].conformance) == "exact-model-tested" for d in p["definitions"]), pid
    for c in (c for c in T.CAPTURED if c.setting in T.SETTINGS):
        assert c.definition in cmap["programs"]["fused-allreduce-rmsnorm" if c.route == "fused" else "tp-allreduce"]["definitions"]


# --- node records -----------------------------------------------------------------------------------------------------------------

H200 = ("9.0", 132)
NODE_TREE = "(((((((0,2),3),1),5),6),4),7)"


def _tree_strs(trees) -> set[str]:
    return {X.tree_str(TN.multimem_tree(t, X.WORLD)) for t in trees}


def test_the_node_records_are_the_two_h200_nodes() -> None:
    """`NODE_RECORDS` holds two H200 nodes.  The first read `NODE_TREE` on every owner, FTZ off, and NCCL's all-reduces past symmetric
    memory as NVLS in one f32 sum on that tree, so it covers symmetric memory to 5461 tokens and NCCL from 5462 to 8192 at vLLM's
    defaults, both at that tree (offset-free).  The second is `CAPTURE_RUNS`' node, `GLM53_H200_TREE` on every owner and no NCCL
    logged, its routes at the runs `CAPTURE_RUNS` names, and `UNSTATED_NODE`."""
    new, old = TN.NODE_RECORDS
    assert _tree_strs(new["multimem"]["trees"]) == {NODE_TREE} and _tree_strs(old["multimem"]["trees"]) == {X.tree_str(T.GLM53_H200_TREE)}
    assert TN.node_key(new)[0] == TN.node_key(old)[0] == H200 and new["multimem"]["ftz"] is old["multimem"]["ftz"] is False
    assert [(r["setting"], r["N"], r["route"], r["lo"], r["hi"]) for r in new["routes"]] == [
        ("defaults", 6144, "pynccl_comm", 5462, 8192), ("defaults", 6144, "symm_mem_comm", 43, 5461)]
    assert new["nccl"]["fit"] == "f32" and [(s["M"], s["algo"], s["proto"], s["channels"], len(s["runs"])) for s in new["nccl"]["sizes"]] == [
        (5462, "NVLS", "SIMPLE", "0..15", 1), (8192, "NVLS", "SIMPLE", "0..15", 1)]
    assert X.tree_str(TN.nvls_tree(new["nccl"], X.WORLD)) == NODE_TREE and new["versions"]["nccl"] == "2.29.7+cuda12.9"
    assert old["nccl"] is None and tuple(s["run_record"] for s in old["source"]) == T.CAPTURE_RUNS
    assert [(r["setting"], r["N"], r["route"], r["lo"], r["hi"], tuple(r["run_records"])) for r in old["routes"]] == [
        ("custom-off", 6144, "symm_mem_comm", 43, 2048, T.CAPTURE_RUNS[:2]), ("defaults", 6144, "symm_mem_comm", 43, 2048, T.CAPTURE_RUNS[2:]),
        ("defaults", 7168, "symm_mem_comm", 37, 2048, T.CAPTURE_RUNS[2:])]
    assert T.UNSTATED_NODE == (TN.node_key(old)[0], *TN.node_key(old)[2:])
    nvls = T.covered("defaults", 6144, "pynccl_comm", 5462, device=H200, trees=new["multimem"]["trees"], ftz=False)
    assert nvls.statics == {"TREES": TN.node_key(new)[2], "FTZ": False} and T.offset_free(nvls, X.WORLD)


def test_covered_takes_the_stated_node_tree() -> None:
    """At the first node's stated tree, symmetric memory covers 43..5461 tokens at vLLM's defaults and NCCL 5462..8192, each at that
    tree.  At `CAPTURE_RUNS`' tree, or with none stated (`UNSTATED_NODE`), symmetric memory covers up to 2048 at its own tree and
    NCCL nothing.  Half a statement (TREES or FTZ alone), another tree, FTZ on, a tree not one per owner, or another device
    covers neither."""
    new_t, old_t = TN.NODE_RECORDS[0]["multimem"]["trees"], [T.GLM53_H200_TREE] * X.WORLD

    def at(route: str, M: int, device=H200, **stated):
        return T.covered("defaults", 6144, route, M, device=device, **stated)

    for M in (43, 2048, 5461):
        assert at("symm_mem_comm", M, trees=new_t, ftz=False).statics["TREES"] == TN.node_key(TN.NODE_RECORDS[0])[2]
    for M in (5462, 8192):
        assert at("pynccl_comm", M, trees=new_t, ftz=False).definition == "AllReduceMultimem_v1"
    assert at("pynccl_comm", 8193, trees=new_t, ftz=False) is None
    for stated in ({"trees": old_t, "ftz": False}, {}):
        assert at("symm_mem_comm", 2048, **stated).statics["TREES"] == TN.multimem_trees(old_t, X.WORLD)
        assert at("symm_mem_comm", 2049, **stated) is None and at("pynccl_comm", 5462, **stated) is None
    for stated in ({"trees": new_t}, {"ftz": False}, {"trees": new_t, "ftz": True}, {"trees": [BALANCED] * X.WORLD, "ftz": False},
                   {"trees": new_t[:7], "ftz": False}):
        assert at("symm_mem_comm", 256, **stated) is None, stated
    assert at("symm_mem_comm", 256, device=T.RTX_PRO_6000, trees=new_t, ftz=False) is None


def test_a_registration_is_refused_by_name(monkeypatch) -> None:
    """`node_refusal` names why a registration's stated node tree is refused: `multimem_tree_unrecorded` for a tree, FTZ, device or
    world no record holds, or a tree that is not one per owner; `multimem_tree_unstated` for half a statement, and for none once
    `UNSTATED_NODE` is None, under which `covered` covers no symmetric-memory call stated none (and still FlashInfer's)."""
    new_t = TN.NODE_RECORDS[0]["multimem"]["trees"]
    for trees, ftz in ((new_t, False), ([T.GLM53_H200_TREE] * X.WORLD, False), (None, None)):
        assert T.node_refusal(trees, ftz, device=H200) is None
    for trees, ftz, device, world in ((new_t, True, H200, 8), ([BALANCED] * X.WORLD, False, H200, 8), (new_t, False, T.RTX_PRO_6000, 8),
                                      (new_t[:7], False, H200, 8), (new_t, False, H200, 4)):
        assert T.node_refusal(trees, ftz, device=device, world=world).startswith(f"{T.TREE_UNRECORDED}: "), (ftz, device, world)
    assert T.node_refusal(new_t, None, device=H200).startswith(f"{T.TREE_UNSTATED}: ")
    monkeypatch.setattr(T, "UNSTATED_NODE", None)
    assert T.node_refusal(None, None, device=H200).startswith(f"{T.TREE_UNSTATED}: ")
    assert T.covered("defaults", 6144, "symm_mem_comm", 256, device=H200) is None
    assert T.covered("defaults", 6144, "fi_ar_comm", 1, device=H200) is not None


def _set(path: tuple, value):
    def mutate(rec):
        *head, last = path
        for k in head:
            rec = rec[k]
        rec[last] = value
    return mutate


@pytest.mark.parametrize("mutate,match", [
    (_set(("schema",), "tp-allreduce-node/0"), "schema"),
    (_set(("device", "compute_capability"), "9"), "not major.minor"),
    (_set(("world",), 1), "world is 1"),
    (lambda r: r["versions"].pop("torch"), "versions has"),
    (_set(("versions", "nccl"), 2.29), "versions.nccl"),
    (lambda r: r["multimem"]["trees"].pop(), "one per owner"),
    (_set(("multimem", "ftz"), "no"), "multimem.ftz"),
    (_set(("multimem", "probes"), [[6144]]), r"multimem.probes\[0\]"),
    (_set(("nccl", "fit"), "fp16"), "nccl.fit"),
    (_set(("nccl", "fit"), "bf16"), "pynccl_comm route needs nccl.fit"),
    (_set(("nccl", "sizes", 0, "bytes"), 1), r"nccl.sizes\[0\].bytes"),
    (_set(("nccl", "sizes", 0, "runs"), [[0, 1, 0]]), "runs end at 1"),
    (_set(("routes", 0, "setting"), "served"), r"routes\[0\] is setting 'served'"),
    (_set(("routes", 0, "route"), "ca_comm"), r"routes\[0\] is route 'ca_comm'"),
    (_set(("routes", 1, "lo"), 9999), "lo 9999 > hi"),
    (lambda r: r["routes"].append(copy.deepcopy(r["routes"][0])), "repeats"),
    (_set(("routes", 0, "run_records"), ["art:" + "1" * 64]), "not in source"),
    (_set(("source", 0, "run_record"), None), r"source\[0\].run_record"),
])
def test_a_node_record_is_refused_by_its_field(mutate, match: str) -> None:
    """A record `tp_allreduce` reads (`check_node_record`'s format, then `SETTINGS` where it keys the routes) is refused by its
    field."""
    rec = copy.deepcopy(TN.NODE_RECORDS[0])
    T._node_entries(TN.check_node_record(rec))
    mutate(rec)
    with pytest.raises(ValueError, match=match):
        T._node_entries(TN.check_node_record(rec))


def _registered(doc: dict) -> dict:
    doc["gate"] = {"ok": True, "name": "NVIDIA H200", "compute_cap": "9.0", "sms": 132, "driver": "595.91.07"}
    doc["environment"] = {"torch": "2.13.0+cu129", "vllm": "0.28.1rc1.dev472+gd9105ea80", "flashinfer": "0.6.18"}
    return doc


def test_a_register_run_writes_a_node_record_that_round_trips(tmp_path, monkeypatch) -> None:
    """`node_record` of a run `register_verdict` holds is an uncited node record whose JSON reads back equal, keyed on the probes' trees
    and FTZ at the gate's device: NCCL's ring (bf16 hops) recorded with its tuning but no NCCL route.  Cited with its run, it loads
    (`load_node_records`, two of one node refused) and `covered` takes its symmetric-memory route at its trees.  An NCCL fit
    without one logged tuning, or probes that disagree, hold no record."""
    doc = _registered(_with_nccl(_doc()))
    assert X.register_verdict(doc)["ok"]
    rec = X.node_record(doc, "c0ffee")
    assert json.loads(json.dumps(rec)) == rec and TN.node_key(rec) == (H200, X.WORLD, TN.multimem_trees(TREES, X.WORLD), True)
    assert rec["versions"] == {"driver": "595.91.07", "nccl": None, "torch": "2.13.0+cu129", "vllm": "0.28.1rc1.dev472+gd9105ea80",
                               "flashinfer": "0.6.18"}
    assert rec["multimem"]["probes"] == [[6144, 64]] and rec["source"] == [{"run_record": None, "commit": "c0ffee"}]
    assert rec["nccl"]["fit"] == "bf16" and [(s["M"], s["algo"], s["channels"]) for s in rec["nccl"]["sizes"]] == [(5462, "RING", "0..23")]
    assert [(r["setting"], r["N"], r["route"], r["lo"], r["hi"]) for r in rec["routes"]] == [("defaults", 6144, "symm_mem_comm", 64, 64)]
    with pytest.raises(ValueError, match="run_record"):
        TN.check_node_record(rec)
    art = "art:" + "0" * 64
    rec["source"][0]["run_record"], rec["routes"][0]["run_records"] = art, [art]
    path = tmp_path / "nodes.json"
    path.write_text(json.dumps([rec]))
    assert TN.load_node_records(path) == (rec,)
    monkeypatch.setattr(T, "CAPTURED", (*T.CAPTURED, *T._node_entries(rec)))
    got = T.covered("defaults", 6144, "symm_mem_comm", 64, device=H200, trees=list(TREES), ftz=True)
    assert got.runs == (art,) and got.statics == {"TREES": TN.multimem_trees(TREES, X.WORLD), "FTZ": True}
    path.write_text(json.dumps([rec, rec]))
    with pytest.raises(ValueError, match="two node records"):
        TN.load_node_records(path)
    untuned = _registered(_with_nccl(_doc()))
    untuned["nccl_log"] = {}
    assert not X.register_verdict(untuned)["ok"]
    split = _registered(_with_nccl(_doc()))
    split["probes"]["agree"] = False
    assert not X.register_verdict(split)["ok"]


def _fake_spawn(flip: dict, trees_of: dict):
    """`spawn_ranks` stood in for: every rank's words from the models (a fused shot's at `FMA`, an all-reduce's by its
    communicator, multimem at `trees_of[spawn]` and FTZ), and its records with its setting's communicators, the time 3 us served
    and 2 us switched; refused unless the spawn's environment and custom all-reduce are its setting's."""

    def spawn(work, env, custom_ar, cases_):
        name, setting = work.name, cases_[0]["setting"]
        assert {c["spawn"] for c in cases_} == {name} and {c["setting"] for c in cases_} == {setting}
        assert env == X.SETTING_ENV[setting] and custom_ar is T.SETTINGS[setting][0], (name, setting)
        for c in cases_:
            c["out"] = str(work / c["key"].replace("/", "_"))
            if c["phase"] == "fused":
                words = _fused_words(c, FMA.get(c.get("shot"), 1))
            else:
                words = _ar_words(c, trees_of[name], True)
            _write(c, words, flip.get(c["key"]))
        return _records(cases_, 2.0 if name == "switch" else 3.0, setting)

    return spawn


@pytest.mark.slow
def test_main_end_to_end_on_stubbed_ranks(small, tmp_path, monkeypatch, capsys) -> None:
    """`main` with the rank processes stood in for (`_fake_spawn`) and the gate passed: the served setting read from the workload,
    each spawn's setting, the operands, the probes' screening and trees, the comparison jobs through the spawned worker pool at the
    jobs' own widths and geometries, each shot's decided `fma`, the per-case verdicts, the routes, the switch's cost, the h7168
    verdict, the coverage, the sealed record, the saved probe words and the exit status, so a slip in the orchestration shows here
    and not in the 8-GPU node's first hour.  About a minute: `main` runs three times."""
    monkeypatch.setattr(X, "gate", lambda *a: {"ok": True, "checks": {}})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    workload = tmp_path / "workload.json"
    workload.write_text(json.dumps({"engine": {"args": {"disable_custom_all_reduce": False}}}))
    monkeypatch.setattr(X, "WORKLOAD_PATH", workload)
    flip = {"fused/M3": ("residual", (1, 5))}
    trees_of = {"served": TREES, "served_again": TREES, "switch": TREES, "h7168": TREES}
    monkeypatch.setattr(X, "spawn_ranks", _fake_spawn(flip, trees_of))
    out = tmp_path / "out"
    argv = ["--out", str(out), "--scratch", str(tmp_path), "--workers", "1"]
    assert X.main(argv) == 1
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is False and not doc["errors"] and doc["fma"] == {"1": 1, "2": 0}
    assert doc["served"] == {"setting": "defaults", "disable_custom_all_reduce": False, "source": "workload.json"}
    assert {k: s["setting"] for k, s in doc["spawns"].items()} == {
        "served": "defaults", "served_again": "defaults", "switch": "switch", "h7168": "defaults"}
    assert {k for k, c in doc["cases"].items() if not c["ok"]} == {"fused/M3"} and doc["totals"]["mismatch"] == 1
    assert set(doc["cases"]) == {"fused/M3", "fused/M171", "fused/special1", "fused/special2", "fused/probe1", "fused/probe2",
                                 "allreduce/M2", "allreduce/M6", "allreduce/M171", "probe/M171", "again/M171", "switch/M171",
                                 "h7168/M2", "h7168/M146", "h7168/M148", "h7168_probe/M148"}
    assert doc["bind"]["ok"] and doc["h7168_bind"]["ok"] and doc["probes"]["agree"] and doc["probes"]["ftz"] is True
    assert all(d["ok"] for d in doc["dispatch"].values()) and set(doc["dispatch"]) == set(X.spawns("defaults"))
    assert [TN.multimem_tree(t, 8) for t in doc["probes"]["trees"]] == list(TREES) and doc["prefill"] == "served"
    assert doc["h7168"] == {"ok": True, "trees": doc["probes"]["trees"], "ftz": True, "same_as_glm53": True}
    assert doc["switch_cost"] == {"M171": round(2 / 3, 3)} and sorted(doc["words"]) == [
        "words/again_M171.npy", "words/allreduce_M171.npy", "words/h7168_M148.npy", "words/h7168_probe_M148.npy",
        "words/probe_M171.npy"]
    assert {(r["N"], r["route"]): r["M"] for r in doc["coverage"]} == {(N7, "fi_ar_comm"): [2, 146], (N7, "symm_mem_comm"): [148]}
    printed = capsys.readouterr().out
    assert "case fused/M3" in printed and "GLM53-TP8-ALLREDUCE NO-GO" in printed and "no-go cases=['fused/M3']" in printed
    flip.clear()
    assert X.main(argv) == 0
    doc = json.loads((out / X.RECORD).read_text())
    assert {(r["setting"], r["N"], r["route"]): r["M"] for r in doc["coverage"]} == {
        ("defaults", N, "fi_ar_comm"): [2, 6], ("defaults", N, "fused"): [3, 4, 6], ("defaults", N, "symm_mem_comm"): [171],
        ("switch", N, "ca_comm"): [171], ("defaults", N7, "fi_ar_comm"): [2, 146], ("defaults", N7, "symm_mem_comm"): [148]}
    printed = capsys.readouterr().out.splitlines()[-1]
    assert printed.startswith("GLM53-TP8-ALLREDUCE GO") and "served=defaults prefill=served" in printed and "ftz=True" in printed
    assert "h7168=GO h7168_trees=same h7168_ftz=True" in printed and all(X.tree_str(t) in printed for t in TREES)
    trees_of["served_again"] = (BALANCED,) * 8
    flip["h7168/M146"] = ("out", (0, 5))
    assert X.main([*argv, "--served", "custom-off"]) == 0
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["served"]["setting"] == "custom-off" and doc["spawns"]["served"]["custom_all_reduce"] is False
    assert all(r["communicators"]["ca_comm"] is False for r in doc["ranks"]["served"]) and doc["dispatch"]["served"]["ok"]
    assert doc["prefill"] == "switch" and doc["probes"]["agree"] is False and doc["probes"]["again_words_equal"] == {"again/M171": False}
    assert doc["h7168"]["ok"] is False and doc["h7168"]["same_as_glm53"] is False and not doc["cases"]["h7168/M146"]["ok"]
    assert {r["setting"] for r in doc["coverage"]} == {"custom-off", "switch"} and all(r["N"] == N for r in doc["coverage"])
    printed = capsys.readouterr().out.splitlines()[-1]
    assert "served=custom-off prefill=switch trees=none ftz=None" in printed and "h7168=NO-GO" in printed


def test_main_models_nccl_at_its_probes_runs(nccl_small, tmp_path, monkeypatch, capsys) -> None:
    """`main` over the all-reduce and NCCL probe phases with the rank processes stood in for, NCCL's words a simulated ring: the
    NCCL probes decoded into runs in both sets of rank processes, the NCCL launch modelled at them through the spawned worker
    pool, the nccl verdict, its record and its line, and the NCCL words saved, so a slip in the NCCL orchestration shows here and
    not on the 8-GPU node."""
    monkeypatch.setattr(X, "gate", lambda *a: {"ok": True, "checks": {}})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    workload = tmp_path / "workload.json"
    workload.write_text(json.dumps({"engine": {"args": {"disable_custom_all_reduce": False}}}))
    monkeypatch.setattr(X, "WORKLOAD_PATH", workload)
    monkeypatch.setattr(X, "spawn_ranks", _fake_spawn({}, {"served": TREES, "served_again": TREES}))
    out = tmp_path / "out"
    argv = ["--out", str(out), "--scratch", str(tmp_path), "--workers", "1", "--only", "allreduce,nccl_probe,nccl_again"]
    assert X.main(argv) == 1
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and not doc["errors"] and doc["ok"] is False
    assert {k for k, c in doc["cases"].items() if not c["ok"]} == {"allreduce/M171"}
    assert doc["nccl"]["ok"] is True and doc["nccl"]["M"]["172"]["model"] == "bf16" and doc["nccl"]["M"]["172"]["runs"] == 65
    assert doc["nccl_probes"]["agree"] and doc["nccl_probes"]["per_probe"]["nccl_again/M172"]["undecoded"] == 0
    assert {"words/allreduce_M172.npy", "words/nccl_probe_M172.npy", "words/nccl_again_M172.npy"} <= set(doc["words"])
    assert set(doc["nccl_log"]) == {"served", "served_again"} and doc["coverage"] == []
    assert "nccl=GO nccl_models=['bf16']" in capsys.readouterr().out.splitlines()[-1]


def _nvls_spawn(trees_of: dict):
    """`_fake_spawn` with NCCL's words one f32 sum over the whole call on `RING`'s chain (NVLS), and each rank process's NCCL log
    naming its version and NVLS for its every NCCL all-reduce."""
    base = _fake_spawn({}, trees_of)

    def spawn(work, env, custom_ar, cases_):
        recs = base(work, env, custom_ar, cases_)
        lines = ["h:1:1 [0] NCCL INFO NCCL version 2.29.7+cuda12.9"]
        for c in (c for c in cases_ if c.get("comm") == "pynccl_comm"):
            flat = np.load(c["ops"]["parts"]).reshape(X.WORLD, -1)
            _write(c, {"out": _ring(flat, flat.shape[1], RING, True).reshape(c["M"], -1)})
            lines.append(f"h:1:1 [0] NCCL INFO AllReduce: {c['M'] * X.HIDDEN * 2} Bytes -> Algo NVLS proto SIMPLE channel{{Lo..Hi}}={{0..15}}")
        Path(work).mkdir(parents=True, exist_ok=True)
        (Path(work) / "nccl.h.0.log").write_text("\n".join(lines) + "\n")
        return recs

    return spawn


def test_main_registers_the_node_on_stubbed_ranks(nccl_small, tmp_path, monkeypatch, capsys) -> None:
    """`main --register` with the rank processes stood in for, NCCL's words one f32 sum on one tree: only `REGISTER_PHASES` run,
    and the node record it writes is the probes' trees and FTZ at the gate's device, NCCL's NVLS tuning and run, and symmetric
    memory's and NCCL's routes at their probes' M, exit 0 and its line.  Probes whose trees differ between the two sets of rank
    processes write no record and exit 1."""
    monkeypatch.setattr(X, "gate", lambda *a: {"ok": True, "checks": {}, "name": "NVIDIA H200", "compute_cap": "9.0", "sms": 132,
                                               "driver": "595.91.07"})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    monkeypatch.setattr(X, "_commit", lambda: "c0ffee")
    workload = tmp_path / "workload.json"
    workload.write_text(json.dumps({"engine": {"args": {"disable_custom_all_reduce": False}}}))
    monkeypatch.setattr(X, "WORKLOAD_PATH", workload)
    trees_of = {"served": TREES, "served_again": TREES}
    monkeypatch.setattr(X, "spawn_ranks", _nvls_spawn(trees_of))
    out, node = tmp_path / "out", tmp_path / "node.json"
    argv = ["--out", str(out), "--scratch", str(tmp_path), "--workers", "1", "--register", str(node)]
    assert X.main(argv) == 0
    doc = json.loads((out / X.RECORD).read_text())
    assert set(doc["cases"]) == {"probe/M171", "again/M171", "nccl_probe/M172", "nccl_again/M172"} and doc["register"] == {"ok": True}
    rec = TN.check_node_record(json.loads(node.read_text()), cited=False)
    assert TN.node_key(rec) == (H200, X.WORLD, TN.multimem_trees(TREES, X.WORLD), True) and rec["versions"]["nccl"] == "2.29.7+cuda12.9"
    assert rec["nccl"]["fit"] == "f32" and [(s["M"], s["algo"], s["proto"], s["channels"], len(s["runs"])) for s in rec["nccl"]["sizes"]] == [
        (172, "NVLS", "SIMPLE", "0..15", 1)]
    assert [(r["route"], r["lo"], r["hi"]) for r in rec["routes"]] == [("pynccl_comm", 172, 172), ("symm_mem_comm", 171, 171)]
    assert rec["multimem"]["probes"] == [[N, 171]] and rec["source"] == [{"run_record": None, "commit": "c0ffee"}]
    assert any(line.startswith(f"GLM53-TP8-NODE-RECORD GO {node}") for line in capsys.readouterr().out.splitlines())
    node.unlink()
    trees_of["served_again"] = (BALANCED,) * 8
    assert X.main(argv) == 1
    assert not node.exists() and "GLM53-TP8-NODE-RECORD NO-GO" in capsys.readouterr().out


def test_a_failed_gate_runs_nothing(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(X, "gate", lambda *a: {"ok": False, "checks": {"world": False}})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    monkeypatch.setattr(X, "spawn_ranks", lambda *a: pytest.fail("ranks spawned past a failed gate"))
    with pytest.raises(SystemExit) as e:
        X.main(["--out", str(tmp_path), "--scratch", str(tmp_path)])
    assert e.value.code == 3 and "GATE-FAIL" in capsys.readouterr().out
    assert json.loads((tmp_path / X.RECORD).read_text())["ok"] is False


def test_the_gate_wants_flashinfer_jit_tools(monkeypatch) -> None:
    """Each rank JIT-builds FlashInfer's TRT-LLM all-reduce, so without ninja or nvcc on PATH the gate fails before a rank spawns."""
    import subprocess

    import torch

    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": True, "checks": {"driver": True}})
    monkeypatch.setattr(X.C, "_run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=""))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: X.WORLD)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "NVIDIA H200")
    monkeypatch.setattr(X.shutil, "which", lambda t: f"/usr/bin/{t}")
    assert X.gate(575, "9.0", 132)["ok"]
    monkeypatch.setattr(X.shutil, "which", lambda t: None if t == "ninja" else f"/usr/bin/{t}")
    rec = X.gate(575, "9.0", 132)
    assert not rec["ok"] and rec["checks"]["jit_tools"] is False and rec["jit_tools"]["ninja"] is None


def test_the_record_is_a_registered_property() -> None:
    from verity_vllm.properties.record import seal

    doc = seal(_doc(), X._verdict(_doc()))
    assert REGISTRY["glm53_tp8_allreduce_capture"].schema == X.SCHEMA and doc["ok"] and doc["digest"] == digest_of(doc)
