import pytest


@pytest.fixture(autouse=True)
def event_log(tmp_path, monkeypatch):
    """Every hook a test runs records its event (`settings.record_event`) here, not in the VM's log."""
    path = tmp_path / "agent-guard-events.jsonl"
    monkeypatch.setenv("AGENT_GUARD_EVENT_LOG", str(path))
    return path
