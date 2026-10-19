"""The test signer's session logs (`mc_cases.ts.session_log`) go under the suite's basetemp (`pous_test_signer.DIR`)."""

from __future__ import annotations

import pytest

import mc_cases


@pytest.fixture(autouse=True, scope="session")
def _test_signer_dir(tmp_path_factory):
    mc_cases.ts.DIR = tmp_path_factory.mktemp("test-signer")
    yield
    mc_cases.ts.DIR = None
