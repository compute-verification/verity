"""Workload 101's fourth Build, fetched from the evidence store: its request Program declares the plain construction, and the
fold's select under it is what the Match's comparison reads (`test_fold_sampler_construction`).
"""
from __future__ import annotations

import json
import os

import pytest

from tests.check.test_fold_sampler_construction import (
    PLAIN,
    _as_compared,
    _fold_select,
    _profile,
)
from verity_vllm.observe.fold.patterns.sampling import (
    TokenSelectGumbelTopP,
    with_sampler_construction,
)
from verity_vllm.program import sampling_event as SE

#: workload 101's fourth try: its Build (`build_request/`)
BUILD_101 = "art:7fef3bd2ef46404067d0c4a534cd0fef72e730f4d01c8b7a6a20022eb5edd9f7"


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
    hist = json.loads((build / "build_request" / "result.json").read_text())["spec_histogram"]
    selects = {k: n for k, n in hist.items() if SE.family_of(k) in SE.CONSTRUCTION_OF}
    assert selects and {SE.family_of(k) for k in selects} == {"GumbelTopPTokenSelect_v2"} and sum(selects.values()) == 32, selects
    assert SE.program_sampler_construction([str(build / "build_request")]) == PLAIN
    pattern = with_sampler_construction(_profile(TokenSelectGumbelTopP()), PLAIN).patterns[0]
    assert {_as_compared(k) for k in selects} == {_fold_select(128256, pattern)} == {"GumbelTopPTokenSelect_v1{V=128256}"}
