import pytest


@pytest.fixture(autouse=True)
def _outside_any_run(monkeypatch):
    """`Checks.finish` writes ``$OUT/<script>.verdict.json`` and, under `research run`, ``$RESEARCH_RUN_DIR/result.json``:
    under a recorded run of the suites those are the run's own, which a test must never write."""
    for k in ("OUT", "RESEARCH_RUN_DIR", "RESEARCH_RUN_ID", "RESEARCH_SOURCE_SHA"):
        monkeypatch.delenv(k, raising=False)
