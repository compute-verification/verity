#!/usr/bin/env python3
"""Repo-local git and gh credentials for Cursor cloud agents in danielreuter/verity, from Verity's own GitHub App.

    python3 tools/research/src/research/verity_github.py install /workspace [--rc ~/.bashrc]
    verity_github.py credential get|store|erase          (git's credential helper protocol, on stdin)
    verity_github.py gh ARGS...                          (gh with the same token, through the bin/gh wrapper)

Cursor pushes each VM a one-hour `ghs_` token of its own GitHub App, written into a global `url.<token>.insteadOf` rule and
gh's hosts.yml. Nothing on the VM can renew it, so when Cursor's push stops, git and gh fail with 401 until the next push.
This helper asks the broker instead: it exchanges the agent's Cursor OIDC identity (the agent socket, audience BROKER) for a
one-hour installation token of the `verity-agents` App, which covers REPO only (contents and pull requests write, metadata
read). The App's private key stays in the broker; nothing long-lived reaches the VM. When the broker is unreachable or
answers 429/5xx, the helper uses Cursor's current token, if it is still valid, and asks the broker again after 60 s. When
the broker refuses this identity, it fails without falling back: a refusal is a decision, not an outage.

`install` copies this file into the checkout's git directory (`.git/verity-auth/`, outside the worktree, so a branch switch
never changes it) and writes repo-local config: an `insteadOf` rule to `https://verity-agent@github.com/REPO`, which is
longer than Cursor's global rule and so wins for REPO alone, and a credential helper scoped to that URL. It also writes
`bin/gh`, which runs the real gh with the token as GH_TOKEN; `--rc` adds that directory to PATH in the given shell rc file.
Install is idempotent: run it from the environment's start script so every VM, including a reset or snapshot-booted one,
has it. The token is cached in `token.json` (mode 0600) and renewed when fewer than MARGIN seconds remain. Stdlib only.
"""
import base64
import contextlib
import datetime
import fcntl
import http.client
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

BROKER = "https://website-docs-sage.vercel.app/api/agent-github/token"
REPO = "danielreuter/verity"
USERNAME = "verity-agent"
#: Cursor's GitHub App: the only fallback credential the helper accepts, never a user's PAT
CURSOR_APP_PREFIX = "ghs_" "1210556_"
MARGIN = 300
FALLBACK_RETRY = 60
ROOT = Path(__file__).resolve().parent
RC_MARK = "# verity-github: gh with Verity's App token"


class AuthError(Exception):
    pass


class Unavailable(AuthError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise AuthError("Credential service redirected; refusing to forward identity")


class UnixHTTP(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("cursor-agent", timeout=10)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


def oidc_token():
    conn = UnixHTTP(os.environ.get("CURSOR_AGENT_SOCKET", "/run/cursor/api.sock"))
    try:
        conn.request("POST", "/v1/tokens/oidc", json.dumps({"aud": BROKER}), {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status == 403:
            raise AuthError("Cursor refused this agent's identity")
        if response.status != 200:
            raise Unavailable("Cursor identity service is unavailable")
        value = json.loads(response.read(32768))
        token = value.get("token")
        if not isinstance(token, str) or not token or any(c.isspace() for c in token):
            raise AuthError("Cursor returned an invalid identity")
        return token
    except (OSError, http.client.HTTPException) as error:
        raise Unavailable("Cursor identity socket is unavailable") from error
    finally:
        conn.close()


def exchange(now):
    identity = oidc_token()
    request = urllib.request.Request(BROKER, data=b"", method="POST", headers={"Authorization": "Bearer " + identity})
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=25) as response:
            data = json.loads(response.read(32768))
    except urllib.error.HTTPError as error:
        if error.code == 429 or error.code >= 500:
            raise Unavailable("GitHub broker is unavailable") from error
        raise AuthError("GitHub broker refused this identity") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise Unavailable("GitHub broker is unreachable") from error
    try:
        expires = datetime.datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00")).timestamp()
        value = {"token": data["token"], "expires": expires, "source": "broker"}
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise AuthError("Broker returned a malformed credential") from error
    if not usable(value, now):
        raise AuthError("Broker returned a stale or malformed credential")
    return value


def usable(value, now):
    return (isinstance(value, dict) and isinstance(value.get("token"), str)
            and bool(value["token"]) and not any(c.isspace() for c in value["token"])
            and isinstance(value.get("expires"), (int, float))
            and now + MARGIN < value["expires"] <= now + 3900)


def real_gh():
    return json.loads((ROOT / "config.json").read_text())["gh"]


def cursor_fallback(now):
    """Cursor's current short-lived App token, from gh's hosts.yml (which Cursor rewrites on each push)."""
    clean = {k: v for k, v in os.environ.items() if k not in ("GH_TOKEN", "GITHUB_TOKEN", "GH_HOST")}
    result = subprocess.run([real_gh(), "auth", "token", "--hostname", "github.com"],
                            capture_output=True, text=True, env=clean, timeout=10)
    token = result.stdout.strip() if result.returncode == 0 else ""
    if not token.startswith(CURSOR_APP_PREFIX):
        raise AuthError("Broker unavailable and no usable Cursor credential remains")
    try:
        part = token[len(CURSOR_APP_PREFIX):].split(".")[1]
        payload = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
        value = {"token": token, "expires": payload["exp"], "source": "cursor"}
        if not usable(value, now):
            raise ValueError("expired")
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise AuthError("Broker unavailable and Cursor's credential has expired") from error
    # GitHub validates this token; the expiry decoded above only controls local caching.
    print("verity-github: broker unavailable; using Cursor's existing short-lived token", file=sys.stderr)
    return value


@contextlib.contextmanager
def locked_cache():
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(ROOT / "token.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a+") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def read_cache():
    try:
        fd = os.open(ROOT / "token.json", os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd) as saved:
            stat = os.fstat(saved.fileno())
            if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
                return None
            return json.load(saved)
    except (OSError, ValueError):
        return None


def save_cache(value):
    # The lock serializes readers, so the file is written in place: no temporary copy of the token is left behind.
    fd = os.open(ROOT / "token.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as saved:
        os.fchmod(saved.fileno(), 0o600)
        json.dump(value, saved)


def get_token(clock=time.time):
    with locked_cache():
        now = clock()
        cached = read_cache()
        if usable(cached, now) and (cached.get("source") != "cursor" or cached.get("retry_at", 0) > now):
            return cached["token"]
        try:
            fresh = exchange(now)
        except Unavailable:
            fresh = cursor_fallback(now)
            fresh["retry_at"] = now + FALLBACK_RETRY
        save_cache(fresh)
        return fresh["token"]


def credential(action, stream=sys.stdin, out=sys.stdout, clock=time.time):
    fields = {}
    for line in stream:
        if line in ("\n", "\r\n"):
            break
        key, sep, value = line.rstrip("\r\n").partition("=")
        if sep:
            fields[key] = value
    if fields.get("protocol") != "https" or fields.get("host") != "github.com" or fields.get("path") not in (REPO, REPO + ".git"):
        return
    if fields.get("username") not in (None, USERNAME):
        return
    if action == "get":
        try:
            token = get_token(clock)
        except Exception as error:
            # quit=true stops git from prompting or asking another helper, which would hide the failure.
            out.write("quit=true\n\n")
            if isinstance(error, AuthError):
                raise
            raise AuthError("GitHub credential is unavailable") from error
        out.write(f"username={USERNAME}\npassword={token}\n\n")
    elif action == "erase":
        with locked_cache():
            cached = read_cache()
            if cached and fields.get("password") == cached.get("token"):
                (ROOT / "token.json").unlink(missing_ok=True)
    # `store` is a no-op: git hands back the password this helper gave it.


def install(repo, rc=None):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    prefixes = ["https://github.com/" + REPO, "git@github.com:" + REPO,
                "ssh://git@github.com/" + REPO, "ssh://git@github.com:22/" + REPO,
                "git+ssh://git@github.com/" + REPO, "git+ssh://git@github.com:22/" + REPO]
    origin = git("config", "--get", "remote.origin.url")
    if origin not in prefixes + [p + ".git" for p in prefixes]:
        raise AuthError("Install only in a clean " + REPO + " checkout")
    directory = Path(git("rev-parse", "--path-format=absolute", "--git-common-dir")) / "verity-auth"
    directory.mkdir(mode=0o700, exist_ok=True)
    os.chmod(directory, 0o700)
    config = directory / "config.json"
    gh = json.loads(config.read_text())["gh"] if config.exists() else shutil.which("gh")
    if not gh or str(directory) in str(Path(gh).resolve()):
        raise AuthError("Install requires the original GitHub CLI on PATH")
    helper = directory / "verity-github.py"
    if helper.resolve() != Path(__file__).resolve():
        shutil.copyfile(__file__, helper)
    helper.chmod(0o700)
    config.write_text(json.dumps({"gh": gh}))
    config.chmod(0o600)
    (directory / "bin").mkdir(mode=0o700, exist_ok=True)
    wrapper = directory / "bin" / "gh"
    wrapper.write_text("#!/bin/sh\nexec python3 " + shlex.quote(str(helper)) + ' gh "$@"\n')
    wrapper.chmod(0o700)
    target = f"https://{USERNAME}@github.com/{REPO}"
    key = "url." + target + ".insteadOf"
    subprocess.run(["git", "-C", str(repo), "config", "--local", "--unset-all", key], capture_output=True)
    for prefix in prefixes:
        git("config", "--local", "--add", key, prefix)
    scope = f"credential.https://{USERNAME}@github.com."
    subprocess.run(["git", "-C", str(repo), "config", "--local", "--unset-all", scope + "helper"], capture_output=True)
    git("config", "--local", "--add", scope + "helper", "")
    git("config", "--local", "--add", scope + "helper", "!python3 " + shlex.quote(str(helper)) + " credential")
    git("config", "--local", scope + "useHttpPath", "true")
    line = "export PATH=" + shlex.quote(str(directory / "bin")) + ':"$PATH"'
    if rc is None:
        print("Installed for " + REPO + ". Add this to this agent's shell setup:")
        print(line)
        return
    rc = Path(rc).expanduser()
    existing = rc.read_text() if rc.exists() else ""
    if line not in existing.splitlines():
        with rc.open("a") as f:
            f.write(("" if existing.endswith("\n") or not existing else "\n") + RC_MARK + "\n" + line + "\n")
    print("Installed for " + REPO + "; gh's PATH entry is in " + str(rc))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        command = argv[0] if argv else ""
        if command == "install" and len(argv) == 2:
            install(Path(argv[1]).resolve())
        elif command == "install" and len(argv) == 4 and argv[2] == "--rc":
            install(Path(argv[1]).resolve(), argv[3])
        elif command == "credential" and len(argv) == 2:
            credential(argv[1])
        elif command == "gh":
            env = dict(os.environ, GH_TOKEN=get_token(), GIT_TERMINAL_PROMPT="0")
            os.execve(real_gh(), [real_gh(), *argv[1:]], env)
        else:
            raise AuthError("Usage: verity_github.py install REPO [--rc FILE] | credential ACTION | gh ARGS...")
    except AuthError as error:
        print("verity-github: " + str(error), file=sys.stderr)
        return 1
    except Exception:
        # Never render an upstream response, command output or credential in an error.
        print("verity-github: credential operation failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
