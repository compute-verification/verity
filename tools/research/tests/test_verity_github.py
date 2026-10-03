"""verity_github.py, the git/gh credential helper: Verity's App token from the broker first, Cursor's injected token only while
the broker is down, never another credential, and a repo-local install that wins over Cursor's global rewrite for verity alone."""
from __future__ import annotations

import base64
import datetime as dt
import io
import json
import os
import subprocess
import urllib.error
from pathlib import Path

import pytest

from research import verity_github as vg

T = 2_000_000_000.0
QUERY = "protocol=https\nhost=github.com\npath=danielreuter/verity.git\nusername=verity-agent\n\n"
BROKER_TOKEN = "broker-dummy-token"


def cursor_token(exp: float) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    return vg.CURSOR_APP_PREFIX + "dummy." + payload + ".not-a-signature"


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Broker:
    """The broker's HTTP replies, in order: a dict is a JSON body, an int an HTTP error, an exception is raised."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    def open(self, request, timeout):
        self.calls += 1
        assert request.full_url == vg.BROKER and request.headers["Authorization"] == "Bearer oidc-dummy"
        reply = self.replies.pop(0)
        if isinstance(reply, int):
            raise urllib.error.HTTPError(vg.BROKER, reply, "dummy", {}, None)
        if isinstance(reply, Exception):
            raise reply
        return Response(json.dumps(reply).encode())


def minted(expires: float, token: str = BROKER_TOKEN) -> dict:
    stamp = dt.datetime.fromtimestamp(expires, dt.timezone.utc).isoformat().replace("+00:00", "Z")
    return {"token": token, "expires_at": stamp}


@pytest.fixture
def helper(tmp_path, monkeypatch):
    """The helper with its cache in tmp_path, the OIDC socket answering, and a fake gh whose token `world.cursor` sets."""
    monkeypatch.setattr(vg, "ROOT", tmp_path / "verity-auth")
    monkeypatch.setattr(vg, "oidc_token", lambda: "oidc-dummy")
    gh, ran = tmp_path / "gh", tmp_path / "gh-ran"
    gh.write_text(f'#!/bin/sh\ntouch {ran}\ncat {tmp_path / "cursor-token"} 2>/dev/null\n')
    gh.chmod(0o755)
    vg.ROOT.mkdir()
    (vg.ROOT / "config.json").write_text(json.dumps({"gh": str(gh)}))

    class World:
        gh_ran = ran

        def cursor(self, token):
            (tmp_path / "cursor-token").write_text(token + "\n")

        def use(self, *replies):
            self.broker = Broker(*replies)
            monkeypatch.setattr(vg.urllib.request, "build_opener", lambda *handlers: self.broker)

        def get(self, now):
            out = io.StringIO()
            vg.credential("get", io.StringIO(QUERY), out, clock=lambda: now)
            return out.getvalue()

        def cache(self):
            return json.loads((vg.ROOT / "token.json").read_text())

    world = World()
    world.use()                                     # no reply queued: nothing in a test reaches the real broker
    return world


def test_the_broker_token_is_served_and_cached_until_it_nears_expiry(helper):
    helper.use(minted(T + 3600), minted(T + 3600 + 3000, "second-dummy-token"))
    assert helper.get(T) == f"username=verity-agent\npassword={BROKER_TOKEN}\n\n"
    assert helper.cache()["source"] == "broker" and (vg.ROOT / "token.json").stat().st_mode & 0o777 == 0o600
    assert helper.get(T + 3600 - vg.MARGIN - 1).endswith(f"password={BROKER_TOKEN}\n\n") and helper.broker.calls == 1
    assert helper.get(T + 3600 - vg.MARGIN).endswith("password=second-dummy-token\n\n"), "renewed inside the margin"
    assert not helper.gh_ran.exists(), "Cursor's token is never read while the broker answers"


@pytest.mark.parametrize("outage", [503, 429, urllib.error.URLError("unreachable")])
def test_a_broker_outage_falls_back_to_cursors_token_and_asks_the_broker_again_after_a_minute(helper, capsys, outage):
    helper.cursor(cursor_token(T + 3000))
    helper.use(outage, minted(T + 3600))
    assert helper.get(T).endswith(f"password={cursor_token(T + 3000)}\n\n")
    assert helper.cache()["source"] == "cursor" and "broker unavailable" in capsys.readouterr().err
    assert helper.get(T + vg.FALLBACK_RETRY - 1).endswith(f"password={cursor_token(T + 3000)}\n\n") and helper.broker.calls == 1
    assert helper.get(T + vg.FALLBACK_RETRY).endswith(f"password={BROKER_TOKEN}\n\n") and helper.cache()["source"] == "broker"


def test_an_unavailable_identity_socket_is_an_outage_too(helper, monkeypatch):
    def down():
        raise vg.Unavailable("Cursor identity socket is unavailable")
    monkeypatch.setattr(vg, "oidc_token", down)
    helper.cursor(cursor_token(T + 3000))
    assert helper.get(T).endswith(f"password={cursor_token(T + 3000)}\n\n")


def test_a_refusal_fails_without_falling_back_and_stops_git_from_prompting(helper):
    helper.cursor(cursor_token(T + 3000))
    helper.use(403)
    out = io.StringIO()
    with pytest.raises(vg.AuthError, match="refused this identity"):
        vg.credential("get", io.StringIO(QUERY), out, clock=lambda: T)
    assert out.getvalue() == "quit=true\n\n" and not helper.gh_ran.exists()


@pytest.mark.parametrize(("token", "why"), [
    (cursor_token(T + vg.MARGIN), "Cursor's credential has expired"),
    ("github_pat_" "dummy0000000000", "no usable Cursor credential"),
    ("", "no usable Cursor credential"),
])
def test_only_a_live_cursor_app_token_is_a_fallback(helper, token, why):
    helper.cursor(token)
    helper.use(503)
    out = io.StringIO()
    with pytest.raises(vg.AuthError, match=why):
        vg.credential("get", io.StringIO(QUERY), out, clock=lambda: T)
    assert out.getvalue() == "quit=true\n\n" and not (vg.ROOT / "token.json").exists()


@pytest.mark.parametrize("query", [
    QUERY.replace("danielreuter/verity.git", "danielreuter/research-notes.git"),
    QUERY.replace("host=github.com", "host=gitlab.com"),
    QUERY.replace("protocol=https", "protocol=http"),
    QUERY.replace("username=verity-agent", "username=notes"),
])
def test_other_repositories_hosts_and_users_get_nothing(helper, query):
    helper.use()
    out = io.StringIO()
    vg.credential("get", io.StringIO(query), out, clock=lambda: T)
    assert out.getvalue() == "" and helper.broker.calls == 0


def test_erase_drops_the_cache_only_for_the_token_it_holds(helper):
    helper.use(minted(T + 3600))
    helper.get(T)
    vg.credential("erase", io.StringIO(QUERY.replace("\n\n", "\npassword=other\n\n")), io.StringIO())
    assert helper.cache()["token"] == BROKER_TOKEN
    vg.credential("erase", io.StringIO(QUERY.replace("\n\n", f"\npassword={BROKER_TOKEN}\n\n")), io.StringIO())
    assert not (vg.ROOT / "token.json").exists()


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], check=check, capture_output=True, text=True)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A verity checkout on a VM whose global config carries Cursor's rewrite, with a gh on PATH that holds no token."""
    repo = tmp_path / "verity"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _git(repo, "remote", "add", "origin", "https://github.com/danielreuter/verity")
    with open(os.environ["GIT_CONFIG_GLOBAL"], "a") as cfg:
        for prefix in ("https://github.com/", "git@github.com:"):
            cfg.write(f'[url "https://x-access-token:cursor-dummy@github.com/"]\n\tinsteadOf = {prefix}\n')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text("#!/bin/sh\nexit 1\n")
    (bin_dir / "gh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(tmp_path))
    return repo


def test_install_routes_verity_alone_to_the_helper_and_git_never_prompts(checkout, tmp_path):
    assert vg.main(["install", str(checkout), "--rc", str(tmp_path / ".bashrc")]) == 0
    assert _git(checkout, "remote", "get-url", "origin").stdout.strip() == "https://verity-agent@github.com/danielreuter/verity"
    other = _git(checkout, "ls-remote", "--get-url", "https://github.com/danielreuter/research-notes").stdout.strip()
    assert other == "https://x-access-token:cursor-dummy@github.com/danielreuter/research-notes", "other repos keep Cursor's rule"
    # No identity socket and no Cursor token: the installed copy refuses, and git neither prompts nor asks another helper.
    fill = subprocess.run(["git", "-C", str(checkout), "credential", "fill"], capture_output=True, text=True,
                          input="url=https://verity-agent@github.com/danielreuter/verity\n\n",
                          env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
    assert fill.returncode != 0 and "password=" not in fill.stdout
    assert "verity-github: Broker unavailable and no usable Cursor credential remains" in fill.stderr
    auth = checkout / ".git" / "verity-auth"
    assert (auth / "verity-github.py").read_bytes() == Path(vg.__file__).read_bytes()
    assert (auth.stat().st_mode & 0o777, (auth / "config.json").stat().st_mode & 0o777) == (0o700, 0o600)


def test_install_is_idempotent_and_adds_the_gh_path_once(checkout, tmp_path):
    rc = tmp_path / ".bashrc"
    rc.write_text("export A=1")
    for _ in range(2):
        assert vg.main(["install", str(checkout), "--rc", str(rc)]) == 0
    lines = rc.read_text().splitlines()
    assert lines[0] == "export A=1" and lines.count(f"export PATH={checkout}/.git/verity-auth/bin:\"$PATH\"") == 1
    rewrites = _git(checkout, "config", "--local", "--get-all",
                    "url.https://verity-agent@github.com/danielreuter/verity.insteadOf").stdout.split()
    assert len(rewrites) == 6
    helpers = _git(checkout, "config", "--local", "--get-all", "credential.https://verity-agent@github.com.helper").stdout
    assert helpers.splitlines()[0] == "" and len(helpers.splitlines()) == 2, "inherited helpers are reset, then ours alone"


def test_install_refuses_a_checkout_of_another_repository(checkout, capsys):
    _git(checkout, "remote", "set-url", "origin", "https://github.com/someone/else")
    assert vg.main(["install", str(checkout)]) == 1
    assert "Install only in a clean danielreuter/verity checkout" in capsys.readouterr().err
    assert not (checkout / ".git" / "verity-auth").exists()
