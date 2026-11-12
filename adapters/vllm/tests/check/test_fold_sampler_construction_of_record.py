"""Workload 101's fourth Build and Match, fetched from the evidence store: the Build's request Program declares the plain
construction, and its preserved capture folds under it as the Match's comparison reads it (`test_fold_sampler_construction`).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.check.test_fold_sampler_construction import PLAIN, VLLM_DIR, _as_compared, _fold_select, _profile
from verity_vllm.observe.fold.patterns.sampling import TokenSelectGumbelTopP, with_sampler_construction
from verity_vllm.program import sampling_event as SE

#: workload 101's fourth try: its Build (`build_request/`) and its records (the Match's files)
BUILD_101 = "art:7fef3bd2ef46404067d0c4a534cd0fef72e730f4d01c8b7a6a20022eb5edd9f7"
RECORDS_101 = "art:89aa13c1fa28b559c1928526dedf8175f34587f7a6ae82bd1fb10555ed35999d"
ROW_101 = "llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__stoch-t0.8-p0.95__bi-eager"


def _store():
    from tests.regression import store_io
    if os.environ.get("VERITY_SKIP_STORE") == "1" or not store_io.reachable():
        pytest.skip("the evidence store is not reachable here")
    return store_io


@pytest.mark.slow
def test_101s_build_declares_the_plain_construction_and_its_select_folds_as_it_compares(tmp_path):
    """Workload 101's fourth Build: its request Program's selects are `GumbelTopPTokenSelect_v2{V=128256,S=32}`, which declare the plain
    construction, and the fold's select under it is exactly what the compare reads them as."""
    build = _store().fetch(BUILD_101, to=tmp_path / "build")
    hist = json.load(open(build / "build_request" / "result.json"))["spec_histogram"]
    selects = {k: n for k, n in hist.items() if SE.family_of(k) in SE.CONSTRUCTION_OF}
    assert selects and {SE.family_of(k) for k in selects} == {"GumbelTopPTokenSelect_v2"} and sum(selects.values()) == 32, selects
    assert SE.program_sampler_construction([str(build / "build_request")]) == PLAIN
    pattern = with_sampler_construction(_profile(TokenSelectGumbelTopP()), PLAIN).patterns[0]
    assert {_as_compared(k) for k in selects} == {_fold_select(128256, pattern)} == {"GumbelTopPTokenSelect_v1{V=128256}"}


@pytest.mark.slow
def test_101s_records_fold_canonical_equal_under_the_programs_construction(tmp_path):
    """Workload 101's fourth Match, replayed on CPU from its records: the preserved capture folded under the construction its Build declares, then
    the Match's own comparison (`program-compare`, as the workload runs it) against the Build's request Program: every family's count equal
    but the dense SiLU's version (the Build predates SiluMul_v2), the selects among them.  The records
    tree holds `match/` (the capture log among it) but not the workload-level `match_compare.json`; that file, the failing comparison, is
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
    # workload 101's Build predates SiluMul_v2 and binds SiluMul_v1; the fold binds the record target's vocabulary's silu_mul (v2)
    assert res["histogram_diff_T_normalised"] == {"SiluMul_v1{I=8192}": [4592, 0], "SiluMul_v2{I=8192}": [0, 4592]}, res["histogram_diff_T_normalised"]


def _profile_of_101() -> str:
    from verity_vllm.observe.fold.profiles import profile_module_for_case
    return profile_module_for_case(json.load(open(VLLM_DIR / "workloads" / f"{ROW_101}.json"))["case"])
