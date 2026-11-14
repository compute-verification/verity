import pytest


@pytest.fixture(autouse=True)
def _outside_a_research_run(monkeypatch):
    """A suite runs in its caller's environment, `verity check --record`'s research run among them: no test writes
    into that run's result."""
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
