import pytest


@pytest.fixture(autouse=True)
def event_log(tmp_path, monkeypatch):
    """Every hook a test runs records its event (`settings.record_event`) here, not in the VM's log."""
    path = tmp_path / "agent-guard-events.jsonl"
    monkeypatch.setenv("AGENT_GUARD_EVENT_LOG", str(path))
    return path


@pytest.fixture(autouse=True)
def no_tmux_server(tmp_path, monkeypatch):
    """A hook asks tmux which sessions there are before a `tmux kill-session` (`kept_session.sessions`): one a test runs finds
    an empty socket directory, never the tmux server of the machine the tests run on."""
    monkeypatch.setenv("TMUX_TMPDIR", str(tmp_path))
    monkeypatch.delenv("TMUX", raising=False)
