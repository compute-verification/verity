"""The repository's `slow` marker (declared in `pyproject.toml`): tests that take minutes are skipped unless `--slow`
is given, whichever test paths a run names."""
from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--slow", action="store_true", default=False, help="run the tests marked slow")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--slow"):
        return
    skip = pytest.mark.skip(reason="slow: opt in with --slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
