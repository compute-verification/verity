"""`research auth request` (research/auth.py) against a scripted API server, on a fake clock that its sleeps advance."""
from __future__ import annotations

import stat
from pathlib import Path

import pytest

from research import auth
from research.jobs.client import Unreachable

TOKEN, SECRET = "tok-9f3a2c-not-for-printing", "poll-5e1d-not-for-printing"
CREATED = {"id": "req1", "approve_url": "https://site.example/auth/requests/req1", "user_code": "WXYZ-2345",
           "poll_secret": SECRET, "interval_s": 5, "expires_at": "2026-09-30T00:30:00Z"}
START = auth._seconds("2026-09-30T00:00:00Z")


class Server:
    """Answers the request, then one scripted answer per poll: (status, body), or an exception to raise."""

    def __init__(self, polls, created=(201, CREATED)):
        self.created, self.polls, self.calls = created, list(polls), []

    def call(self, method, path, body=None):
        self.calls.append((method, path, body))
        if path == "/api/auth/requests":
            if isinstance(self.created, Exception):
                raise self.created
            return self.created
        answer = self.polls.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class Clock:
    def __init__(self):
        self.t, self.slept = START, []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def run(server, tmp_path, *args, clock=None):
    clock = clock or Clock()
    lines = []
    code = auth.main(["request", "--name", "dispatch-rc", "--scopes", "jobs:work", "--file", str(tmp_path / "jobs" / "dispatch-rc.token"),
                      *args], client=server, sleep=clock.sleep, clock=clock, out=lines.append)
    return code, "\n".join(lines), clock


def test_an_approved_request_writes_the_token_with_mode_600_and_never_prints_it(tmp_path):
    server = Server([(200, {"state": "pending"}), (200, {"state": "signed-in", "github_login": "neekon"}),
                     (200, {"state": "signed-in", "github_login": "neekon"}),
                     (200, {"state": "approved", "github_login": "neekon", "token": TOKEN, "name": "dispatch-rc", "scopes": ["jobs:work"],
                            "expires_at": "2026-12-29T00:00:00Z"})])
    code, out, clock = run(server, tmp_path, "--note", "the RC's dispatcher")
    path = tmp_path / "jobs" / "dispatch-rc.token"
    assert code == 0 and path.read_text() == TOKEN + "\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert TOKEN not in out and SECRET not in out
    assert "https://site.example/auth/requests/req1" in out and "WXYZ-2345" in out
    assert out.count("signed in as neekon") == 1 and "approved for neekon: wrote `dispatch-rc` (jobs:work, expires 2026-12-29" in out
    method, route, body = server.calls[0]
    assert (method, route) == ("POST", "/api/auth/requests") and body["name"] == "dispatch-rc" and body["scopes"] == ["jobs:work"]
    assert body["note"] == "the RC's dispatcher" and body["client"].startswith("research ") and body["host"]
    assert all(c[2] == {"poll_secret": SECRET} and c[1] == "/api/auth/requests/req1/poll" for c in server.calls[1:])
    assert clock.slept == [5, 5, 5, 5]
    assert list(path.parent.iterdir()) == [path]


def test_a_denied_request_writes_nothing(tmp_path):
    code, out, _ = run(Server([(200, {"state": "signed-in", "github_login": "someone"}), (200, {"state": "denied"})]), tmp_path)
    assert code == 1 and "denied" in out and not (tmp_path / "jobs" / "dispatch-rc.token").exists()


def test_polling_backs_off_when_asked_and_when_the_site_is_away(tmp_path):
    server = Server([(429, {"interval_s": 12}), Unreachable("POST poll: timed out"), Unreachable("again"),
                     (200, {"state": "approved", "token": TOKEN, "scopes": ["jobs:work"]})])
    code, out, clock = run(server, tmp_path)
    assert code == 0 and clock.slept == [5, 12, 24, 48]
    assert out.count("can't be reached") == 1


def test_a_request_that_nobody_approves_expires(tmp_path):
    code, out, clock = run(Server([(200, {"state": "pending"})] * 1000), tmp_path)
    assert code == 1 and "expired" in out
    assert clock.t > auth._seconds(CREATED["expires_at"])
    code, out, _ = run(Server([(200, {"state": "expired"})]), tmp_path)
    assert code == 1 and "expired" in out
    code, out, _ = run(Server([(200, {"state": "collected"})]), tmp_path)
    assert code == 1 and "already collected" in out


def test_narrowed_scopes_are_said(tmp_path):
    server, clock, lines = Server([(200, {"state": "approved", "token": TOKEN, "scopes": ["jobs:enqueue"]})]), Clock(), []
    code = auth.main(["request", "--name", "coordinator", "--scopes", "jobs:enqueue,trains:write", "--file", str(tmp_path / "token")],
                     client=server, sleep=clock.sleep, clock=clock, out=lines.append)
    assert code == 0 and any("narrowed the scopes you asked for (jobs:enqueue, trains:write)" in line for line in lines)


def test_an_existing_file_is_kept_unless_forced(tmp_path, capsys):
    path = tmp_path / "jobs" / "dispatch-rc.token"
    path.parent.mkdir()
    path.write_text("old\n")
    code, _, _ = run(Server([]), tmp_path)
    assert code == 2 and path.read_text() == "old\n" and "exists" in capsys.readouterr().err
    code, _, _ = run(Server([(200, {"state": "approved", "token": TOKEN})]), tmp_path, "--force")
    assert code == 0 and path.read_text() == TOKEN + "\n"


def test_the_site_unreachable_or_refusing_at_the_start(tmp_path):
    code, out, _ = run(Server([], created=Unreachable("no route")), tmp_path)
    assert code == 4 and "can't be reached" in out
    code, out, _ = run(Server([], created=(400, {"error": "unknown scope jobs:everything"})), tmp_path)
    assert code == 1 and "unknown scope" in out


@pytest.mark.parametrize("args,why", [(["--name", "Dispatch RC"], "--name"), (["--scopes", "jobs work"], "--scopes"), (["--scopes", ","], "--scopes")])
def test_bad_arguments_are_refused_before_anything_is_sent(tmp_path, capsys, args, why):
    base = {"--name": "dispatch-rc", "--scopes": "jobs:work"}
    base.update(dict(zip(args[::2], args[1::2])))
    argv = ["request", *[x for kv in base.items() for x in kv], "--file", str(tmp_path / "t")]
    assert auth.main(argv, client=Server([])) == 2 and why in capsys.readouterr().err


def test_research_auth_is_wired():
    from research.cli import main

    with pytest.raises(SystemExit) as e:
        main(["auth", "request", "--help"])
    assert e.value.code == 0


def test_the_http_client_sends_no_token_when_asking_for_one():
    from research.jobs.client import HTTPClient

    assert HTTPClient("https://site.example", anonymous=True)._token is None
    assert HTTPClient("https://site.example", token="t")._token == "t"
