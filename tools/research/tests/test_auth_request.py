"""`research auth` (research/auth.py) against a scripted API server in the shape of `internal/token-requests-spec.md`, on a fake
clock that its sleeps advance."""
from __future__ import annotations

import json
import stat

import pytest

from research import auth
from research.jobs.client import Unreachable

TOKEN, CODE = "dummy-token-value", "dummy-device-code"
CREATED = {"request_id": "r-1", "user_code": "BCDF-GHJK", "device_code": CODE, "verification_uri": "https://site.example/device",
           "verification_uri_complete": "https://site.example/device?code=BCDF-GHJK", "expires_in": 900, "interval": 5, "status": "pending"}
PENDING = (400, {"error": "authorization_pending", "status": "waiting_for_sign_in", "login": None, "interval": 5})
WAITING = (400, {"error": "authorization_pending", "status": "waiting_for_approval", "login": "neekon", "interval": 5})
ISSUED = (200, {"token": TOKEN, "name": "dispatch-rc", "scopes": ["jobs:work"], "expires_at": "2026-12-29T00:00:00Z", "login": "neekon",
                "line": None, "daily_usd": None, "replaces": False})


class Server:
    """Answers the request, then one scripted answer per poll: (status, body), or an exception to raise."""

    def __init__(self, polls, created=(201, CREATED)):
        self.created, self.polls, self.calls = created, list(polls), []

    def call(self, method, path, body=None):
        self.calls.append((method, path, body))
        if path == "/api/token-requests":
            if isinstance(self.created, Exception):
                raise self.created
            return self.created
        answer = self.polls.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class Clock:
    def __init__(self):
        self.t, self.slept = 0.0, []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def request(server, tmp_path, *args, name="dispatch-rc", scopes="jobs:work"):
    clock, lines = Clock(), []
    code = auth.main(["request", "--name", name, "--scopes", scopes, "--file", str(tmp_path / "jobs" / "dispatch-rc.token"), *args],
                     client=server, sleep=clock.sleep, clock=clock, out=lines.append)
    return code, "\n".join(lines), clock


def test_an_issued_token_is_written_with_mode_600_and_never_printed(tmp_path):
    server = Server([PENDING, WAITING, WAITING, ISSUED])
    code, out, clock = request(server, tmp_path, "--days", "30", "--agent", "bc-8ece7cde", "--why", "the RC's dispatcher")
    path = tmp_path / "jobs" / "dispatch-rc.token"
    assert code == 0 and path.read_text() == TOKEN + "\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    meta = json.loads(auth.meta_path(path).read_text())
    assert meta == {"name": "dispatch-rc", "scopes": ["jobs:work"], "expires_at": "2026-12-29T00:00:00Z", "login": "neekon",
                    "line": None, "daily_usd": None, "days": 30}
    assert TOKEN not in out and CODE not in out
    assert "https://site.example/device?code=BCDF-GHJK" in out and "BCDF-GHJK" in out
    assert out.count("waiting for you to open the link") == 1 and out.count("signed in as @neekon; waiting for approval") == 1
    assert "wrote `dispatch-rc` (jobs:work, expires 2026-12-29T00:00:00Z, for @neekon)" in out
    method, route, body = server.calls[0]
    assert (method, route) == ("POST", "/api/token-requests")
    assert body["name"] == "dispatch-rc" and body["scopes"] == ["jobs:work"] and body["days"] == 30 and body["machine"]
    assert body["agent"] == "bc-8ece7cde" and body["why"] == "the RC's dispatcher"
    assert all(c[1] == "/api/token-requests/poll" and c[2] == {"device_code": CODE} for c in server.calls[1:])
    assert clock.slept == [5, 5, 5, 5]
    assert sorted(p.name for p in path.parent.iterdir()) == ["dispatch-rc.token", "dispatch-rc.token.json"]


@pytest.mark.parametrize("error,says", [("access_denied", "denied"), ("expired_token", "expired"), ("invalid_grant", "doesn't know")])
def test_a_request_that_stops_writes_nothing(tmp_path, error, says):
    code, out, _ = request(Server([WAITING, (400, {"error": error, "message": "…"})]), tmp_path)
    assert code == 1 and says in out and not (tmp_path / "jobs" / "dispatch-rc.token").exists()


def test_polling_slows_down_when_asked_and_when_the_site_is_away(tmp_path):
    server = Server([(400, {"error": "slow_down", "interval": 10}), (429, {"error": "rate_limited"}), Unreachable("POST poll: HTTP 503"),
                     Unreachable("again"), ISSUED])
    code, out, clock = request(server, tmp_path)
    assert code == 0 and clock.slept == [5, 10, 20, 40, 60]
    assert out.count("can't be reached") == 1


def test_it_gives_up_after_30_minutes(tmp_path):
    code, out, clock = request(Server([PENDING] * 1000), tmp_path)
    assert code == 1 and "giving up" in out and clock.t > auth.GIVE_UP_S


def test_narrowed_scopes_and_a_replaced_token_are_said(tmp_path):
    issued = (200, {**ISSUED[1], "name": "coordinator", "scopes": ["jobs:enqueue"], "replaces": True})
    code, out, _ = request(Server([issued]), tmp_path, name="coordinator", scopes="jobs:enqueue,events:read")
    assert code == 0 and "narrowed the scopes you asked for (jobs:enqueue, events:read)" in out
    assert "replaces the live token `coordinator` at its first use" in out


def test_an_existing_file_is_kept_unless_forced(tmp_path, capsys):
    path = tmp_path / "jobs" / "dispatch-rc.token"
    path.parent.mkdir()
    path.write_text("old\n")
    code, _, _ = request(Server([]), tmp_path)
    assert code == 2 and path.read_text() == "old\n" and "research auth renew" in capsys.readouterr().err
    code, _, _ = request(Server([ISSUED]), tmp_path, "--force")
    assert code == 0 and path.read_text() == TOKEN + "\n"


def test_renew_sends_the_token_as_bearer_and_asks_for_the_same(tmp_path, monkeypatch):
    request(Server([ISSUED]), tmp_path, "--days", "30")
    path = tmp_path / "jobs" / "dispatch-rc.token"
    seen = []
    renewed = (200, {**ISSUED[1], "token": "dummy-renewed-token", "replaces": True})

    def client(url=None, token=None, anonymous=False):
        seen.append((token, anonymous))
        return Server([renewed], created=(201, {**CREATED, "status": "approved"}))
    monkeypatch.setattr(auth, "HTTPClient", client)
    clock, lines = Clock(), []
    assert auth.main(["renew", "--file", str(path)], sleep=clock.sleep, clock=clock, out=lines.append) == 0
    assert seen == [(TOKEN, False)] and path.read_text() == "dummy-renewed-token\n"
    assert "renewing `dispatch-rc` (jobs:work) for 30 days: approved at once" in lines[0]
    assert not any("open this link" in line for line in lines)
    assert json.loads(auth.meta_path(path).read_text())["days"] == 30


def test_renew_needs_what_request_wrote(tmp_path, capsys):
    assert auth.main(["renew", "--file", str(tmp_path / "missing")], client=Server([])) == 2
    assert "as `research auth request` writes them" in capsys.readouterr().err


def test_the_site_unreachable_or_refusing_at_the_start(tmp_path):
    code, out, _ = request(Server([], created=Unreachable("no route")), tmp_path)
    assert code == 4 and "can't be reached" in out
    code, out, _ = request(Server([], created=(400, {"error": "invalid_request", "message": "scopes: unknown scope jobs:everything"})), tmp_path)
    assert code == 1 and "invalid_request: scopes: unknown scope jobs:everything" in out


@pytest.mark.parametrize("args,why", [
    (["--name", "Dispatch RC"], "--name"), (["--name", "d"], "--name"), (["--name", "dispatch_rc"], "--name"),
    (["--scopes", "jobs work"], "--scopes"), (["--scopes", ","], "--scopes"), (["--days", "0"], "--days"), (["--days", "400"], "--days"),
])
def test_bad_arguments_are_refused_before_anything_is_sent(tmp_path, capsys, args, why):
    base = {"--name": "dispatch-rc", "--scopes": "jobs:work"}
    base.update(dict(zip(args[::2], args[1::2])))
    server = Server([])
    argv = ["request", *[x for kv in base.items() for x in kv], "--file", str(tmp_path / "t")]
    assert auth.main(argv, client=server) == 2 and why in capsys.readouterr().err and server.calls == []


def test_a_store_scope_is_a_scope():
    assert auth.SCOPE.fullmatch("write:fixture/v1") and auth.SCOPE.fullmatch("jobs:enqueue")


def test_research_auth_is_wired():
    from research.cli import main

    with pytest.raises(SystemExit) as e:
        main(["auth", "request", "--help"])
    assert e.value.code == 0


def test_the_http_client_sends_no_token_when_asking_for_one():
    from research.jobs.client import HTTPClient

    assert HTTPClient("https://site.example", anonymous=True)._token is None
    assert HTTPClient("https://site.example", token="t")._token == "t"
