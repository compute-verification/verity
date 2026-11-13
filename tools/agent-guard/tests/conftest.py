import subprocess

import pytest


@pytest.fixture(autouse=True, scope="session")
def scratch_event_log(tmp_path_factory):
    """No hook a test starts records its event (`settings.record_event`) in the machine's log, whatever environment the test
    hands it. The session's environment names this scratch log from before the first module's fixture is built, so every
    copy of the environment holds the name; and a child a test starts with an environment that lacks it (one built from
    nothing, or with every `AGENT_GUARD_*` variable taken out) is given it."""
    path = tmp_path_factory.mktemp("events") / "agent-guard-events.jsonl"
    start = subprocess.Popen.__init__

    def start_with_the_log_named(self, *args, **kwargs):
        if kwargs.get("env") is not None:
            kwargs["env"] = {"AGENT_GUARD_EVENT_LOG": str(path), **kwargs["env"]}
        start(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("AGENT_GUARD_EVENT_LOG", str(path))
        mp.setattr(subprocess.Popen, "__init__", start_with_the_log_named)
        yield path


@pytest.fixture(autouse=True)
def event_log(tmp_path, monkeypatch):
    """What the hooks a test runs in its own environment record: a log of the test's own."""
    path = tmp_path / "agent-guard-events.jsonl"
    monkeypatch.setenv("AGENT_GUARD_EVENT_LOG", str(path))
    return path


@pytest.fixture(autouse=True)
def no_tmux_server(tmp_path, monkeypatch):
    """A hook asks tmux which sessions there are before a `tmux kill-session` (`kept_session.sessions`): one a test runs finds
    an empty socket directory, never the tmux server of the machine the tests run on."""
    monkeypatch.setenv("TMUX_TMPDIR", str(tmp_path))
    monkeypatch.delenv("TMUX", raising=False)
