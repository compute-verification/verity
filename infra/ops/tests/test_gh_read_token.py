"""The control pod's read-only GitHub token (infra/control/gh_read_token.sh), which every watcher that reads GitHub takes, and
the watchers' `gh` (infra/control/bin/gh), with openssl, curl and python3.12 stood in: the request asks GitHub for read
permissions only, its cache is a file no write token is kept in, and `gh` takes the token of the moment at each call."""
import json
import os
import re
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

CONTROL = Path(__file__).resolve().parents[3] / "infra" / "control"
READ, WRITE, NOTES = (CONTROL / f"gh_{n}_token.sh" for n in ("read", "app", "notes"))
# curl's stand-in records what the helper asks GitHub for (the URL and the body after -d) and answers with $REPLY
CURL = """#!/bin/sh
while [ $# -gt 0 ]; do
  case "$1" in
    -d) printf '%s\\n' "$2" >> "$ASKED";;
    https://*) printf '%s\\n' "$1" >> "$ASKED";;
  esac
  shift
done
cat "$REPLY"
"""


def script(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


class Pod:
    """A pod's programs under a tmp dir: the App's key, GitHub's reply to the exchange, and what each helper was asked for."""

    def __init__(self, tmp: Path):
        bin_ = tmp / "bin"
        script(bin_ / "openssl", "#!/bin/sh\ncat\n")
        script(bin_ / "curl", CURL)
        script(bin_ / "python3.12", f'#!/bin/sh\nexec {sys.executable} "$@"\n')
        self.reply, self.asked, self.read_cache, self.write_cache = (tmp / n for n in ("reply", "asked", "read-cache", "write-cache"))
        (tmp / "app.pem").write_text("not-to-be-copied\n")
        self.env = {"PATH": f"{bin_}:/usr/bin:/bin", "GH_APP_KEY": str(tmp / "app.pem"), "ASKED": str(self.asked),
                    "REPLY": str(self.reply), "GH_READ_CACHE": str(self.read_cache), "GH_APP_CACHE": str(self.write_cache)}
        self.answer("dummy-minted")

    def answer(self, token: str | None, message: str = "") -> None:
        self.reply.write_text(json.dumps({"token": token} if token else {"message": message}))

    def run(self, helper: Path, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([str(helper), *args], env=self.env, capture_output=True, text=True)

    def requests(self) -> list[tuple[str, dict]]:
        lines = self.asked.read_text().splitlines() if self.asked.exists() else []
        return [(url, json.loads(body)) for url, body in zip(lines[::2], lines[1::2])]


def test_the_read_helper_asks_github_for_read_permissions_only_on_the_two_repositories(tmp_path):
    pod = Pod(tmp_path)
    r = pod.run(READ)
    assert (r.returncode, r.stdout) == (0, "dummy-minted\n"), r.stderr
    (url, body), = pod.requests()
    assert url == "https://api.github.com/app/installations/173153984/access_tokens"
    assert body == {"repositories": ["verity", "infra"], "permissions": {"contents": "read", "pull_requests": "read"}}
    assert set(body["permissions"].values()) == {"read"}
    pod.answer(None, "Bad credentials")
    r = pod.run(READ)
    assert r.returncode != 0 and r.stdout == "" and "gh_read_token: GitHub refused: Bad credentials" in r.stderr


def test_the_cached_read_token_is_in_a_file_of_its_own_so_no_reader_gets_a_write_token_and_no_writer_a_read_one(tmp_path):
    """Both helpers keep their token 45 min. Were they to share a cache, whichever minted first would hand its token to the
    other's callers: a watcher would hold a token that writes, or drive one that can't."""
    pod = Pod(tmp_path)
    pod.write_cache.write_text("dummy-write\n")
    assert pod.run(READ, "--cached").stdout == "dummy-minted\n"
    assert pod.read_cache.read_text() == "dummy-minted\n" and stat.S_IMODE(pod.read_cache.stat().st_mode) == 0o600
    assert pod.write_cache.read_text() == "dummy-write\n" and [p.name for p in tmp_path.glob("read-cache*")] == ["read-cache"]
    pod.answer("dummy-later")
    assert pod.run(READ, "--cached").stdout == "dummy-minted\n" and len(pod.requests()) == 1     # within 45 min: no new exchange
    assert pod.run(WRITE, "--cached").stdout == "dummy-write\n" and len(pod.requests()) == 1
    old = pod.read_cache.stat().st_mtime - 2701
    os.utime(pod.read_cache, (old, old))
    assert pod.run(READ, "--cached").stdout == "dummy-later\n" and pod.read_cache.read_text() == "dummy-later\n"
    assert [body["permissions"] for _, body in pod.requests()] == [{"contents": "read", "pull_requests": "read"}] * 2
    defaults = [re.search(r"^cache=\$\{(\w+):-(\S+)\}$", f.read_text(), re.M).groups() for f in (READ, WRITE, NOTES)]
    assert defaults[0] == ("GH_READ_CACHE", "/dev/shm/verity-read-token") and len({path for _, path in defaults}) == 3


def test_the_watchers_gh_takes_the_read_token_of_the_moment_at_each_call_whatever_token_its_loop_started_with(tmp_path):
    """A loop's environment is read once, when it starts, and a token lives an hour: `gh` on the loops' PATH asks the read
    helper at each call, and a GH_TOKEN the loop was started with is not the one GitHub's CLI gets."""
    token = tmp_path / "token"
    helper = script(tmp_path / "helper", '#!/bin/sh\n[ "$*" = --cached ] || exit 2\ncat "$TOKEN"\n')
    gh = script(tmp_path / "gh", '#!/bin/sh\necho "$GH_TOKEN $*"\n')
    env = {"PATH": "/usr/bin:/bin", "WATCHERS_GH_TOKEN": str(helper), "WATCHERS_GH": str(gh), "TOKEN": str(token),
           "GH_TOKEN": "dummy-from-the-start"}
    for value in ("dummy-one", "dummy-two"):
        token.write_text(value + "\n")
        r = subprocess.run([str(CONTROL / "bin" / "gh"), "api", "x y"], env=env, capture_output=True, text=True, check=True)
        assert r.stdout == f"{value} api x y\n"
    token.unlink()
    r = subprocess.run([str(CONTROL / "bin" / "gh"), "api", "x"], env=env, capture_output=True, text=True)
    assert r.returncode == 1 and r.stdout == ""
    wrapper = (CONTROL / "bin" / "gh").read_text()
    assert "${WATCHERS_GH_TOKEN:-/data/infra/gh_read_token.sh}" in wrapper and "${WATCHERS_GH:-/data/watchers/bin/gh}" in wrapper
    path = tomllib.loads((CONTROL / "watchers.toml").read_text())["env"]["PATH"].split(":")
    assert path[:2] == ["{src}/infra/control/bin", "{dir}/bin"]           # the wrapper is found before the CLI it runs
