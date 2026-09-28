"""The Match's fold binds the top-p select under the sampler construction the Build's Programs declare (`sampling_event.
program_sampler_construction`, passed as `pipeline.match --program-dir`), never under the capture target's default alone.

A single-request Program's select is specialised on its constant split count (`GumbelTopPTokenSelect_v2{V,S}`, `registry.topp_words`),
which has no shared-greedy form: it compares as the plain `GumbelTopPTokenSelect_v1{V}` (`split_selects.as_fold_selects`), so its fold
binds that one.  A Program whose selects read the `splits` Input (every batched component) declares S4's `shared-greedy` and folds to
`GumbelTopPTokenSelectSharedGreedy_v1{V}`, as before.  #101's fourth Match (r20260928-175437-cfe5) failed on exactly this pair: derived
`GumbelTopPTokenSelect_v1` x32, record `GumbelTopPTokenSelectSharedGreedy_v1` x32.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from verity.ml.circuits.defs import bind
from verity_vllm.check.match import split_selects
from verity_vllm.observe.fold.patterns.sampling import SCRATCH_TOPP_SELECT, TokenSelectGumbelTopP, with_sampler_construction
from verity_vllm.observe.fold.resolver import Profile
from verity_vllm.program import sampling_event as SE
from verity_vllm.program.frontend.target_profile import DEFAULT_TARGET, TargetProfile
from verity_vllm.program.registry import sampling as SM
from verity_vllm.program.registry import sampling_vocab
from verity_vllm.program.registry import topp_words

VLLM_DIR = Path(__file__).resolve().parents[2]
PLAIN, SHARED = "greedy-check-per-stage", "shared-greedy"
#: #101's fourth try (train V fe7931d5, run r20260928-175437-cfe5): its Build (`build_request/`) and its records (the Match's files)
BUILD_101 = "art:7fef3bd2ef46404067d0c4a534cd0fef72e730f4d01c8b7a6a20022eb5edd9f7"
RECORDS_101 = "art:89aa13c1fa28b559c1928526dedf8175f34587f7a6ae82bd1fb10555ed35999d"
ROW_101 = "llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__stoch-t0.8-p0.95__bi-eager"


def test_the_construction_table_is_the_registrys():
    assert set(SE.CONSTRUCTION_OF.values()) == set(TargetProfile.SAMPLER_CONSTRUCTIONS) and set(SE.CONSTRUCTION_OF) <= set(SE.SAMPLING_EVENT_FAMILIES)
    for plain_id, shared in sampling_vocab.SHARED_GREEDY.items():
        assert SE.CONSTRUCTION_OF[plain_id] == PLAIN and SE.CONSTRUCTION_OF[shared.id] == SHARED
        plain = next(d for d in (SM.GumbelTokenSelect, SM.GumbelTopPTokenSelect) if d.id == plain_id)
        assert sampling_vocab.select_for(plain, SE.CONSTRUCTION_OF[shared.id]) is shared and sampling_vocab.select_for(plain, SE.CONSTRUCTION_OF[plain_id]) is plain
    assert SE.CONSTRUCTION_OF[topp_words.GumbelTopPTokenSelectV2.id] == PLAIN, "the specialised select has no shared-greedy form"


def test_a_programs_selects_declare_one_construction(tmp_path):
    assert SE.sampler_construction_of(["GumbelTopPTokenSelect_v2{V=64,S=4}", "Gemm_v1{K=4,N=4}"]) == PLAIN
    assert SE.sampler_construction_of(["s:GumbelTopPTokenSelectSharedGreedy_v1{V=64}"]) == SHARED
    assert SE.sampler_construction_of(["TokenSelect_v1{V=64}"]) is None
    with pytest.raises(ValueError, match="two sampler constructions"):
        SE.sampler_construction_of(["GumbelTopPTokenSelect_v2{V=64,S=4}", "GumbelTopPTokenSelectSharedGreedy_v1{V=64}"])
    (tmp_path / "result.json").write_text(json.dumps({"program": {}}))
    with pytest.raises(SystemExit, match="no spec_histogram"):
        SE.program_sampler_construction([str(tmp_path)])


def _program_dir(tmp_path: Path, V: int, S: int | None) -> tuple[Path, str]:
    """A stand-in request Program (`StochasticServe` through the vLLM sampler rule): single-request at a constant `S`, else reading the
    `splits` Input as every batched component does; its dir holds the `result.json` spec histogram the Build writes.  -> (dir, select)."""
    from verity_vllm.pipeline.build import _spec_histogram
    from verity_vllm.program.frontend import derive_examples as ex
    from verity_vllm.program.frontend.derive import derive_report
    from verity_vllm.program.frontend.rules import DEFAULT_RULES
    from verity_vllm.program.frontend.rules.base import RuleSet
    from verity_vllm.program.frontend.rules.vllm_sampling import VLLM_SAMPLING_RULES
    from verity_vllm.program.registry import b1
    LP, STEPS = 3, 1
    C = dict(b1.B0_CONFIG, NL=1, V=V, MAX_POS=64)
    rules = RuleSet("default+stoch", list(DEFAULT_RULES.rules) + list(VLLM_SAMPLING_RULES), list(DEFAULT_RULES.families))
    with torch.device("meta"):
        m = ex.StochasticServe(C, STEPS, constant_S=S, temperature=0.8, top_p=0.95)
    inputs = (torch.empty(LP, dtype=torch.int32, device="meta"), torch.empty(1, dtype=torch.int64, device="meta"))
    if S is None:
        inputs += (torch.empty(STEPS + 1, dtype=torch.int32, device="meta"),)
    rep = derive_report(m, inputs, mode="profile", profile="b1-eager-v3", config=C, rules=rules, name="StochServe")
    assert rep.ok, rep.unsupported
    hist = _spec_histogram(rep.program)
    d = tmp_path / ("single" if S is not None else "batched")
    d.mkdir()
    (d / "result.json").write_text(json.dumps({"spec_histogram": hist}))
    select = next(k for k in hist if SE.family_of(k) in SE.CONSTRUCTION_OF)
    return d, select


def _fold_select(V: int, pattern: TokenSelectGumbelTopP, target: TargetProfile | None = None) -> str:
    """The spec the fold's top-p select binds at its gather (`TokenSelectGumbelTopP._stage3`) under a capture header declaring `target`
    (None: a header with no declared target, as #101's capture folds under the default)."""
    from verity_vllm.observe.events import DispatchEnter, DispatchExit, RunHeader, ViewDesc
    from verity_vllm.observe.fold.resolver import StepContext
    from verity_vllm.observe.fold.tree import TreeNode
    logits = ViewDesc(1, 0, (1, V), (V, 1), "bfloat16", "cuda:0")
    local_argmax, idx, out = ViewDesc(2, 0, (1, 4), (4, 1), "int64", "cuda:0"), *(ViewDesc(a, 0, (1, 1), (1, 1), "int64", "cuda:0") for a in (3, 4))
    node = TreeNode(DispatchEnter(1, None, 0, "sample", "aten.gather.default", "aten::gather(Tensor self, int dim, Tensor index, *, bool "
                                  "sparse_grad=False) -> Tensor", [local_argmax, 1, idx], {}, 7, None), DispatchExit(1, out))
    header = RunHeader("verity-capture/log/v1", "2026-09-28T00:00:00Z", "toy", {"target": target.to_json()} if target else {}, {}, {})
    stage2 = {"local_argmax": local_argmax, "bf16": logits, "S": 4, "kernels": (), "specs": [("r0", 5, 7, 0.8, 0.9)]}
    ctx = StepContext(header, None, None, [], {}, {}, None, scratch={SCRATCH_TOPP_SELECT: {"stage2": {idx.alloc: stage2}}})
    return pattern._stage3(node, ctx).spec.id


def _as_compared(spec: str) -> str:
    rows = [{"spec": spec, "k": 0}]
    split_selects.as_fold_selects(rows)
    return rows[0]["spec"]


def _profile(*patterns) -> Profile:
    return Profile("stand-in", list(patterns), {}, None)


@pytest.mark.parametrize("S, construction, folded", [(4, PLAIN, "GumbelTopPTokenSelect_v1"), (None, SHARED, "GumbelTopPTokenSelectSharedGreedy_v1")])
def test_the_fold_binds_the_construction_the_program_declares(tmp_path, S, construction, folded):
    """Single-request: the Program's v2 select declares the plain construction and the fold binds `GumbelTopPTokenSelect_v1{V}`, the spec
    the compare reads the v2 select as; without the Program's construction the fold binds S4's shared-greedy default, #101's failure.
    Batched: the Program's `SharedGreedy_v1` declares `shared-greedy` and the fold binds it exactly as it did before."""
    V = 64
    d, select = _program_dir(tmp_path, V, S)
    assert (SE.family_of(select) == "GumbelTopPTokenSelect_v2") == (S is not None)
    assert SE.program_sampler_construction([str(d)]) == construction
    profile = with_sampler_construction(_profile(TokenSelectGumbelTopP(), object()), construction)
    pattern = next(p for p in profile.patterns if isinstance(p, TokenSelectGumbelTopP))
    fold = _fold_select(V, pattern)
    assert fold == bind(getattr(SM, folded.removesuffix("_v1")), V=V).id == _as_compared(select)
    default = _fold_select(V, TokenSelectGumbelTopP())
    assert (default == fold) == (construction == SHARED), "the default is S4's shared-greedy: a single-request Program no longer folds under it"


def test_a_declared_target_construction_still_holds_without_program_dirs():
    V = 64
    tp = DEFAULT_TARGET.with_overrides(sampler_construction=PLAIN)
    assert with_sampler_construction(_profile(TokenSelectGumbelTopP()), None).patterns[0].construction is None
    assert _fold_select(V, TokenSelectGumbelTopP(), tp) == bind(SM.GumbelTopPTokenSelect, V=V).id
    with pytest.raises(SystemExit, match="not one of"):
        with_sampler_construction(_profile(TokenSelectGumbelTopP()), "plain")


def test_the_match_driver_passes_the_programs_construction_to_every_fold(tmp_path):
    from verity_vllm.pipeline import cli
    from verity_vllm.pipeline.match import Plan
    d, _select = _program_dir(tmp_path, 64, 4)
    wl = tmp_path / "workload.json"
    wl.write_text(json.dumps({"requests": [{"request_id": "r-0", "prompt_token_ids": [1, 2, 3], "max_tokens": 2}]}))
    base = ["--case", "LLAMA32_1B", "--profile", "generic", "--workload", str(wl), "--out", str(tmp_path / "match")]
    plan = Plan(cli.parse("match", [*base, "--program-dir", str(d)], {}))
    for stage in ("fold", "fold_2"):
        cmd = plan.cmd(stage)
        assert cmd[cmd.index("--sampler-construction") + 1] == PLAIN, stage
    assert "--sampler-construction" not in Plan(cli.parse("match", base, {})).cmd("fold")


def _store():
    from tests.regression import store_io
    if os.environ.get("VERITY_SKIP_STORE") == "1" or not store_io.reachable():
        pytest.skip("the evidence store is not reachable here")
    return store_io


def test_101s_build_declares_the_plain_construction_and_its_select_folds_as_it_compares(tmp_path):
    """#101's fourth Build: its request Program's selects are `GumbelTopPTokenSelect_v2{V=128256,S=32}`, which declare the plain
    construction, and the fold's select under it is exactly what the compare reads them as."""
    build = _store().fetch(BUILD_101, to=tmp_path / "build")
    hist = json.load(open(build / "build_request" / "result.json"))["spec_histogram"]
    selects = {k: n for k, n in hist.items() if SE.family_of(k) in SE.CONSTRUCTION_OF}
    assert selects and {SE.family_of(k) for k in selects} == {"GumbelTopPTokenSelect_v2"} and sum(selects.values()) == 32, selects
    assert SE.program_sampler_construction([str(build / "build_request")]) == PLAIN
    pattern = with_sampler_construction(_profile(TokenSelectGumbelTopP()), PLAIN).patterns[0]
    assert {_as_compared(k) for k in selects} == {_fold_select(128256, pattern)} == {"GumbelTopPTokenSelect_v1{V=128256}"}


def test_101s_records_fold_canonical_equal_under_the_programs_construction(tmp_path):
    """#101's fourth Match, replayed on CPU from its records: the preserved capture folded under the construction its Build declares, then
    the Match's own comparison (`program-compare`, as the row runs it) against the Build's request Program: every family's count equal
    but the dense SiLU's version (the Build predates SiluMul_v2), the selects among them.  The records
    tree holds `match/` (the capture log among it) but not the row-level `match_compare.json`; that file, the failing comparison, is
    checked only where a tree carries it.  A store that lists or fetches nothing skips by name."""
    store = _store()
    try:
        build = store.fetch(BUILD_101, to=tmp_path / "build")
    except store.StoreUnavailable as e:
        pytest.skip(f"the evidence store is not reachable: the fetch of {BUILD_101} returned nothing: {e}"[:400])
    rec = tmp_path / "records"
    listing = store._cli("fetch", RECORDS_101, "--list", "--json", check=False)
    try:
        paths = [str(r["path"]) for r in json.loads(listing.stdout)] if listing.returncode == 0 else []
    except (ValueError, TypeError, KeyError):
        paths = []
    capture = next((p for p in paths if p.endswith("match/capture/log.jsonl.gz")), None)
    compare = next((p for p in paths if p.rsplit("/", 1)[-1] == "match_compare.json"), None)
    if capture is None:
        pytest.skip(f"the evidence store is not reachable: it lists no match/capture/log.jsonl.gz in {RECORDS_101} ({len(paths)} files, "
                    f"rc {listing.returncode}: {listing.stderr.strip()[-200:]})")
    p = store._cli("fetch", RECORDS_101, "--to", str(rec), *(x for q in (capture, compare) if q for x in ("--path", q)), check=False)
    log = rec / capture
    if p.returncode != 0 or not log.is_file():
        pytest.skip(f"the evidence store is not reachable: the fetch of {capture} from {RECORDS_101} returned nothing (rc {p.returncode}: "
                    f"{p.stderr.strip()[-200:]})")
    if compare is not None and (rec / compare).is_file():
        assert json.load(open(rec / compare))["histogram_diff_T_normalised"] == {"GumbelTopPTokenSelectSharedGreedy_v1{V=128256}": [0, 32],
                                                                                   "GumbelTopPTokenSelect_v1{V=128256}": [32, 0]}
    run = tmp_path / "fold"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(VLLM_DIR), os.environ.get("PYTHONPATH", "")])}
    fold = subprocess.run([sys.executable, "-m", "verity_vllm.pipeline.cli", "match", "--worker", "fold", "--log", str(log), "--profile",
                           _profile_of_101(), "--out", str(run / "fold_summary.json"), "--persist-dir", str(run),
                           "--sampler-construction", SE.program_sampler_construction([str(build / "build_request")])],
                          cwd=VLLM_DIR, env=env, capture_output=True, text=True)
    assert fold.returncode == 0, fold.stderr[-3000:]
    cmp = subprocess.run([sys.executable, "-m", "verity_vllm.pipeline.cli", "program-compare", "--a", str(build / "build_request"),
                          "--b", str(run), "--out", str(tmp_path / "match_compare.json")], cwd=VLLM_DIR, env=env, capture_output=True, text=True)
    assert cmp.returncode == 0, cmp.stderr[-3000:]
    res = json.load(open(tmp_path / "match_compare.json"))
    # #101's Build predates SiluMul_v2 (#820) and binds SiluMul_v1; the fold binds the record target's vocabulary's silu_mul (v2)
    assert res["histogram_diff_T_normalised"] == {"SiluMul_v1{I=8192}": [4592, 0], "SiluMul_v2{I=8192}": [0, 4592]}, res["histogram_diff_T_normalised"]


def _profile_of_101() -> str:
    from verity_vllm.engine.profiles import profile_module_for_case
    return profile_module_for_case(json.load(open(VLLM_DIR / "workloads" / f"{ROW_101}.json"))["case"])
