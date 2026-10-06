import pytest


@pytest.fixture(autouse=True)
def _outside_any_run(monkeypatch):
    """A benchmark's `run_result.write` writes into ``$RESEARCH_RUN_DIR``: under a `research run` of the suites that is the
    run's own result.json, which a test's verdict must never replace."""
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
