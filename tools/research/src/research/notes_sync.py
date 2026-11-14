"""`research notes sync` / `research notes scan`: the notes tree as a git repo with a remote that several machines write.

    research notes sync [--path P]... [-m MSG] [--retries N] [--remote NAME] [--root DIR]
    research notes scan [--max-bytes N] [--json] [--root DIR]

`sync` commits the caller's changes (every change, or only under `--path`; files over 1 MB, evidence and renders, and the
steward's own state stay out, as with `snapshot`), then fetches, rebases onto the remote branch, and pushes; a push that loses a race fetches, rebases and pushes
again, at most `--retries` times.  A rebase conflict never drops either side: the rebase is aborted, the local commit stays, the
exit is 3 and the conflicted files are named.  Before anything is pushed, the outgoing commits' added lines are scanned for key
material (`KEY_PATTERNS`, and the value of any secret-named environment variable); a hit refuses the push (exit 4), naming the
file.  With no remote configured the commit is local and the report says so.

One sync of a clone runs at a time, so several agents can share a clone: a sync holds the clone's lock (`_lock`, a flock on
`<git dir>/notes-sync.lock`) from before its commit to after its push, and another sync of the clone waits for it, at most
LOCK_WAIT_S, saying on stderr whom it waits for (the holder's pid, machine and agent); past that it commits nothing and exits
1.  Before it commits, a sync puts right what an earlier git left in the clone (`settle`), each with one `SYNC-WARN` line: a
rebase or merge in progress that no git process is running is ended, and a HEAD left detached with nothing in progress and the
branch at or behind it gets the branch checked out where it is.  A HEAD detached anywhere else is refused (exit 2).  Ending a
rebase, a found one or the sync's own, never loses what was written under it: `git rebase --abort` resets every tracked file,
so the files changed in the working tree are saved first and put back after (`end_operation`), and the sync's own rebase is
run again when what stopped it was such a file.  What cannot be put back (the file changed on both sides since, a conflicted
file someone wrote into, a file written where a found rebase had taken one out of the tree, a commit made on its detached
HEAD) stays on a local branch `rescue-<UTC time>`, and the sync fails naming it.  Other git commands in a clone (the watcher's
`snapshot`, the standby's `pull`, the archive) do not take the lock.

A commit made here is the clone's own (`identity_env`: `sync`'s and its rebase's, the archive's, `snapshot`'s): where the
configuration git reads in the clone names an author and a committer, git is not handed the calling shell's GIT_AUTHOR_NAME,
GIT_AUTHOR_EMAIL, GIT_COMMITTER_NAME and GIT_COMMITTER_EMAIL, so a name a script exported never reaches the notes.  A clone
that names none commits as the variables say.  A commit the rebase replays keeps the author it was made with.

A lane note (`lanes/**/*.md`) the sync would add or change stays out of the commit when its filename stamp `YYYYMMDDTHHMMZ` (a
path new since HEAD) or its front matter's `created:` (new or changed since HEAD) is more than 5 minutes ahead of the sync's
clock: one `SYNC-REFUSED <path>: ...` line each, with the value `date -u` gives now, the rest synced, exit 5 when nothing else
failed.  What HEAD already has is never re-checked, and a stamp stops being ahead once the clock passes it.

Evidence and renders never enter the notes' history: a file added or changed under `renders/`, `campaigns/*/assets/` or
`lanes/*/evidence/` stays out of the commit and is named in the report; it belongs in the evidence store (`research data put
... --preserve`), cited by its `art:` id.  Deleting one commits like any other change.

A note (`lanes/**/*.md`, `kb/**/*.md`) whose front matter is broken also stays out: opened with `---` and never closed, a line
that is not `key: value` (or a list item, an indented continuation, a comment), or a `created:` that is not a UTC time (the stamp
check above cannot read it).  Front matter is optional: a note without it is fine, since its lane, kind and time are in its path.
`SYNC-WARN <path>: ...` lines (committed anyway) name a `lane:` that is not the note's folder (a handoff's lane is its recipient)
and a `created:` more than a day from the file name's stamp.  Only content the repository has never held is checked (a move, or
an unchanged file, is not), and only for problems HEAD's copy of the path did not already have.

Once pushed, the `art:` ids and run ids that the pushed commits' added lines cite in notes are looked up on the evidence store's
remote, what every reader resolves (not this machine's catalog): an `art:` prefix must name a manifest, a run id an attempt or,
failing that, a label (a run with `custody waived` is accounted for).  Each note citing one the remote lacks gets a
`SYNC-WARN <path>: cites ...` line saying how to publish it; nothing is refused.  At most 64 ids and 20 s a sync; with no usable
remote there is no check, and `RESEARCH_NOTES_CITE_CHECK=0` turns it off.

`scan` reads every blob in the repo's history: files over `--max-bytes` and key material, each with the path it was committed
under.  Exit 1 when it finds anything; run it before the first push and after (acceptance A5).

With `RESEARCH_NOTES_SYNC=1`, `research notes checkpoint | bind | relaunch` run `sync` for the lane directories they wrote once
they succeed (a cloud agent's default; the laptop keeps the watcher's snapshot until the switch-over).

With a notes token (`notes_token`), every fetch, pull and push of a github.com notes remote (`remote_target`: `sync`, the standby
steward's pull, budgets.toml's fetch) goes to its explicit-user URL `https://x-access-token@github.com/<owner>/<repo>.git` with a
credential helper that reads the token from the environment (`RESEARCH_NOTES_TOKEN`).  A cloud VM's global `url.<cursor[bot]
token>.insteadOf https://github.com/` matches only the plain URL, so it no longer turns the request into the bot's, which the
notes repo refuses; the token itself is never in a URL, an argument or a message.  A Cursor agent's token is the agents' GitHub
broker's, asked for with its Cursor identity whatever the variable held, so a cloud VM needs no secret and one it still
injects is not used.  Where the broker gives none (no Cursor identity: a laptop, the control pod), the token is the variable as it
was set, and with neither nothing changes.  When GitHub refuses the variable's token (401 or 403), `sync`, `pull` and the clone
say so in one line and go on with the broker's where it gives one (`refused_given`).  With a token, a notes command (and machine
resolution, which reads `<notes>/machines.d`) on a machine without the notes clone clones `NOTES_URL` there first
(`ensure_clone`).
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

SYNC_ENV = "RESEARCH_NOTES_SYNC"
TOKEN_ENV = "RESEARCH_NOTES_TOKEN"
SOURCE_ENV, EXPIRES_ENV = "RESEARCH_NOTES_TOKEN_SOURCE", "RESEARCH_NOTES_TOKEN_EXPIRES"
#: the agents' GitHub broker: a Cursor agent's identity buys an hour's token for the notes repository
GITHUB_BROKER = "https://website-docs-sage.vercel.app/api/agent-github/token"
REFRESH_S, BROKER_RETRY_S = 300, 60
NOTES_URL = "https://github.com/compute-verification/research-notes.git"
#: a clone made before the notes moved into the organization still names the old address; the token is the organization's
MOVED = {"danielreuter/research-notes": "compute-verification/research-notes"}
GITHUB_URL_RE = re.compile(r"^(?:https://(?:[^@/]+@)?github\.com/|(?:(?:git\+)?ssh://)?git@github\.com(?::22)?[:/])"
                           r"(?P<repo>[\w.-]+/[\w.-]+?)(?:\.git)?/?$")
#: the shell expands the variable when git asks for the password: the token's value is never in git's arguments or config
CREDENTIAL_HELPER = f'!f(){{ echo username=x-access-token; echo "password=${TOKEN_ENV}"; }}; f'
MAX_BYTES = 1 << 20
FUTURE_RC = 5
FUTURE_TOLERANCE = dt.timedelta(minutes=5)
STAMP_RE = re.compile(r"^(\d{8}T\d{4}Z)-")
CREATED_RE = re.compile(r"^created:\s*[\"']?((\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)\s*(Z|[+-]\d{2}:?\d{2})?)[\"']?\s*$",
                        re.M | re.I)
KEY_PATTERNS = {
    "aws-access-key-id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "apikey": re.compile(r"(?i)\bapi_?key\b\s*[=:]\s*[\"']?[A-Za-z0-9_\-]{16,}"),
    "private-key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "session-token": re.compile(r"(?i)\bsession_?token\b\"?\s*[=:]\s*[\"']?[A-Za-z0-9/+=_\-]{16,}"),
}
SECRET_ENV_RE = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD)", re.I)
NEVER_COMMIT = ("steward-state.json", ".steward-state.json.tmp")
EVIDENCE_PATH = re.compile(r"^(?:renders/|campaigns/[^/]+/assets/|lanes/[^/]+/evidence/)")
#: an index.lock this old, with no git process in the repository, was left by a git that died mid-write
STALE_LOCK_S = 600
#: the clone's sync lock, in its git directory; how long a sync waits for another sync of the clone, and how often it looks again
LOCK_FILE, LOCK_WAIT_S, LOCK_POLL_S = "notes-sync.lock", 120.0, 0.25
#: how long a sync that finds a rebase or merge in progress, or HEAD detached, waits for a git process at work in the clone to end
GIT_WAIT_S = 10.0
#: a local branch that keeps what a sync could not put back; never pushed, and never taken for the notes branch
RESCUE = "rescue-"
#: `git status --porcelain` codes of a path with an unresolved conflict
UNMERGED = ("DD", "AU", "UD", "UA", "DU", "AA", "UU")


def is_evidence(path: str) -> bool:
    """A path under the notes' evidence and render locations, which never enter its history."""
    return bool(EVIDENCE_PATH.match(path))
FRONT_LINE_RE = re.compile(r"^(?:[A-Za-z_][\w.-]*:(?:[ \t].*)?|[ \t]+\S.*|-(?:[ \t].*)?|#.*|[ \t]*)$")
FRONT_LANE_RE = re.compile(r"^lane:[ \t]*[\"'`]?([^\s\"'`]+)", re.M)
MAIL_RE = re.compile(r"handoff|asks|reply")
CREATED_STAMP_SLACK = dt.timedelta(days=1)
CITE_ENV = "RESEARCH_NOTES_CITE_CHECK"
CITE_ART_RE = re.compile(r"\bart:([0-9a-f]{8,64})\b")
CITE_RUN_RE = re.compile(r"\br\d{8}-\d{6}-[0-9a-f]{4}\b")
CITE_MAX = 64
CITE_WAIT_S = 20.0
CITE_HINTS = {"run": "a run is in the store once its attempt is on the remote: `research data custody <run> --publish` where its "
                     "run dir is, else `research data label <run> custody waived --by <you>`",
              "art": "`research data push <art>` where it was put"}


#: what names a commit's author and committer from a shell: git reads these before the repository's configuration
IDENTITY_ENV = ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL")


def identity_env(root: Path) -> dict[str, str] | None:
    """The environment git commits in under `root`: the caller's less IDENTITY_ENV where the configuration git reads there
    (the clone's, the user's, the machine's) names an author and a committer, each with an address, so a commit is theirs
    whatever a shell exported; else None, the caller's as it is: a clone that names none commits as the variables say, and
    without them as git guesses or refuses.  GIT_AUTHOR_DATE and GIT_COMMITTER_DATE stay."""
    env = {k: v for k, v in os.environ.items() if k not in IDENTITY_ENV}
    if len(env) == len(os.environ):
        return None
    r = subprocess.run(["git", "-C", str(root), "config", "--get-regexp", r"^(user|author|committer)\.(name|email)$"],
                       capture_output=True, text=True, errors="replace", timeout=60, env=env)
    named = {f[0] for f in (line.split(None, 1) for line in r.stdout.splitlines()) if len(f) == 2}
    whole = all(f"{who}.{k}" in named or f"user.{k}" in named for who in ("author", "committer") for k in ("name", "email"))
    return env if whole else None


def _run(root: Path, *args: str, timeout: float = 120, input: str | None = None, index: Path | None = None) -> subprocess.CompletedProcess:
    """git in `root`, as the clone's own author and committer (`identity_env`); output that is not UTF-8 (a diff of a binary
    or Latin-1 file) is decoded with replacement characters.  With `index`, git reads and writes that index file instead of
    the clone's.
    Commits are unsigned: the notes repo needs no signatures, and an agent VM's signing key is often missing from its agent,
    which would make every commit and rebase here fail."""
    env = identity_env(root)
    if index is not None:
        env = {**(os.environ if env is None else env), "GIT_INDEX_FILE": str(index)}
    return subprocess.run(["git", "-C", str(root), "-c", "commit.gpgsign=false", *args], capture_output=True, text=True,
                          errors="replace", timeout=timeout, input=input, env=env)


def _out(root: Path, *args: str) -> str | None:
    r = _run(root, *args)
    return r.stdout if r.returncode == 0 else None


clock = time.time
NO_IDENTITY = "no Cursor identity here"
_BROKER_REFUSED: list[tuple[float, str]] = []       # (when, why) the broker last gave no token; empty once it gives one
_SET_ASIDE: list[str] = []                          # non-empty once the broker's token replaced one the variable was given


def _held() -> tuple[str | None, bool, float]:
    """(the token in the environment, whether `notes_token` put it there from the broker, when that one expires)."""
    tok, source, expires = (os.environ.get(k) for k in (TOKEN_ENV, SOURCE_ENV, EXPIRES_ENV))
    return tok or None, source == "broker", float(expires or 0) if source == "broker" else 0.0


def notes_token(now: float | None = None) -> str | None:
    """A token for the notes repository.  On a Cursor agent, an hour's token from the agents' GitHub broker (`GITHUB_BROKER`, scope
    `notes`), asked for with the agent's Cursor identity whatever RESEARCH_NOTES_TOKEN held: this puts it in the environment with
    its source and expiry, so git's credential helper and child processes read it, and mints again REFRESH_S before it expires.
    Where the broker gives none (`broker_why`: no Cursor identity, as on a laptop or the control pod; a refusal; an outage),
    RESEARCH_NOTES_TOKEN as it was set, else None; a broker that gave none is asked again only after BROKER_RETRY_S."""
    now = clock() if now is None else now
    tok, minted, expires = _held()
    if tok and minted and expires > now + REFRESH_S:
        return tok
    if _BROKER_REFUSED and now < _BROKER_REFUSED[-1][0] + BROKER_RETRY_S:
        return tok
    try:
        from .slack import _NoRedirect, oidc_token
        req = urllib.request.Request(GITHUB_BROKER, data=json.dumps({"scope": "notes"}).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {oidc_token(GITHUB_BROKER)}",
                                              "Content-Type": "application/json"})
        with urllib.request.build_opener(_NoRedirect).open(req, timeout=25) as r:
            d = json.loads(r.read(32768))
        new = d["token"]
        expires = dt.datetime.fromisoformat(d["expires_at"].replace("Z", "+00:00")).timestamp()
        if not isinstance(new, str) or not new or any(c.isspace() for c in new):
            raise ValueError("malformed token")
    except Exception as e:  # noqa: BLE001  (no Cursor identity, a refusal or an outage: the caller goes on with what it was given)
        code = getattr(e, "code", None)
        _BROKER_REFUSED[:] = [(now, f"the broker answered HTTP {code}" if isinstance(code, int) else
                               NO_IDENTITY if type(e).__name__ == "SlackError" else f"the broker gave no usable answer ({type(e).__name__})")]
        return tok
    if tok and not minted:
        _SET_ASIDE.append(TOKEN_ENV)
    _BROKER_REFUSED.clear()
    os.environ.update({TOKEN_ENV: new, SOURCE_ENV: "broker", EXPIRES_ENV: f"{expires:.0f}"})
    return new


def broker_why() -> str | None:
    """Why the broker last gave no token (`NO_IDENTITY`, its HTTP status, the class of an outage); None once it gave one, or
    before it was asked."""
    return _BROKER_REFUSED[-1][1] if _BROKER_REFUSED else None


def token_source() -> tuple[str | None, str]:
    """(where `notes_token` gets the token now, the same in words for a message): "broker", with whether a RESEARCH_NOTES_TOKEN
    this process was given is now unused; "env", the variable, with why the broker gave none; None, with why there is none."""
    tok, minted = notes_token(), _held()[1]
    if tok and minted:
        return "broker", "the GitHub broker's token, by this agent's Cursor identity" + (
            f"; {TOKEN_ENV} is set here too and is not used: a Cursor agent no longer needs it" if _SET_ASIDE else "")
    why = broker_why() or "the broker was not asked"
    return ("env", f"{TOKEN_ENV}, since the GitHub broker gave no token ({why})") if tok else \
        (None, f"git's own: there is no notes token ({why}, and {TOKEN_ENV} is not set)")


#: what git says when GitHub answers a credential with 401 or 403
TOKEN_REFUSED = ("Authentication failed", "Invalid username or token", "returned error: 401", "returned error: 403", "denied to")


def refused_given() -> bool:
    """GitHub refused the notes token (401 or 403) and it was RESEARCH_NOTES_TOKEN as this process was given it, not the broker's:
    say so in one line on stderr, never the token, and ask the broker now.  Whether the broker gave one, so the refused request
    is worth making again; False, and nothing said, when the refused token was the broker's or there was none."""
    tok, minted, _ = _held()
    if not tok or minted:
        return False
    _BROKER_REFUSED.clear()                                 # asked now, though it may have given none a moment ago
    notes_token()
    again = _held()[1]
    print(f"notes: {TOKEN_ENV} was refused by GitHub: it is expired or revoked; "
          + ("going on with the GitHub broker's token" if again else f"the GitHub broker gave no token to go on with ({broker_why()})"),
          file=sys.stderr)
    return again


def _authed(target: Callable[[], tuple[list[str], str]], git: Callable[[list[str], str], subprocess.CompletedProcess]
            ) -> subprocess.CompletedProcess:
    """`git(options, repository)` for what `target` gives now (`remote_target`, `token_url`); when GitHub refuses
    RESEARCH_NOTES_TOKEN as given (`TOKEN_REFUSED`, `refused_given`), once more with the broker's token."""
    r = git(*target())
    if r.returncode and any(k in r.stderr for k in TOKEN_REFUSED) and refused_given():
        r = git(*target())
    return r


def token_url(url: str) -> tuple[list[str], str] | None:
    """(git options, explicit-user URL) that reach the github.com repository `url` names with the notes token (`notes_token`,
    module docstring); None when there is none or `url` is not github.com's."""
    m = GITHUB_URL_RE.match(url.strip())
    if not (m and notes_token()):
        return None
    repo = MOVED.get(m["repo"], m["repo"])
    return ["-c", "credential.helper=", "-c", f"credential.helper={CREDENTIAL_HELPER}"], f"https://x-access-token@github.com/{repo}.git"


def remote_target(root: Path, remote: str = "origin") -> tuple[list[str], str]:
    """(git options, what to name as the repository) for a fetch, pull or push of the notes clone's `remote`: `token_url` of its
    URL as configured (`remote get-url` prints it rewritten by insteadOf), else ([], remote) as before."""
    url = _out(root, "config", "--get", f"remote.{remote}.url")
    return (token_url(url) if url else None) or ([], remote)


RACE = ("non-fast-forward", "fetch first", "[rejected]", "failed to update ref", "stale info")
REFUSED = ("Authentication failed", "could not read Username", "Permission denied", "denied to", "returned error: 401",
           "returned error: 403", "Repository not found", "does not appear to be a git repository")


def refused(stderr: str) -> bool:
    """Whether a failed fetch or push was refused (credentials, permissions, no such repository), which no retry changes. Anything
    else, a lost push race or a lock or transfer error two machines' syncs hit under heavy load, is retried."""
    return any(k in stderr for k in REFUSED)


def token_hint(root: Path, remote: str = "origin") -> str:
    """'' unless there is no notes token here and `remote` is github.com's; then why git failed and what to do, for the end of
    a fetch or push failure (a laptop's own credentials need no token, so having none is no refusal by itself)."""
    url = _out(root, "config", "--get", f"remote.{remote}.url") or ""
    if not GITHUB_URL_RE.match(url.strip()) or notes_token():
        return ""
    return (f" -> no notes token here ({broker_why()}), so git used this machine's own credentials, which the notes repo refuses "
            f"on a cloud VM: a Cursor agent's token comes from the GitHub broker (`research doctor` checks it), so sync again in "
            f"a minute; where there is no Cursor identity, set {TOKEN_ENV}")


def pull(root: Path, remote: str = "origin") -> subprocess.CompletedProcess:
    """`git pull --ff-only` of the current branch from `remote` (`remote_target`), its remote-tracking branch updated as a plain
    pull does."""
    branch = (_out(root, "symbolic-ref", "--short", "HEAD") or "").strip()

    def git(opts: list[str], where: str) -> subprocess.CompletedProcess:
        if not opts or not branch:
            return _run(root, "pull", "-q", "--ff-only", timeout=300)
        return _run(root, *opts, "pull", "-q", "--ff-only", where, f"+refs/heads/{branch}:refs/remotes/{remote}/{branch}", timeout=300)

    return _authed(lambda: remote_target(root, remote), git)


_CLONE_LOCK = threading.Lock()


def ensure_clone(root: Path) -> str | None:
    """Clone NOTES_URL to `root` when `root` is missing and there is a notes token (a fresh cloud VM), its `origin` the plain
    URL: a line saying it did, or why it failed; None when there was nothing to do.  One thread of a process clones at a time
    (`research run`'s machine probes run in parallel, so a clone directory named by pid alone would be each other's), and each
    clones into a directory of its own, so another process cloning beside it never touches it."""
    root = Path(root).expanduser()
    if root.exists() or not token_url(NOTES_URL):
        return None
    with _CLONE_LOCK:
        if root.exists():
            return None
        return _clone(root)


def _clone(root: Path) -> str:
    root.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{root.name}.clone-", dir=root.parent))
    r = _authed(lambda: token_url(NOTES_URL) or ([], NOTES_URL), lambda opts, url: subprocess.run(
        ["git", *opts, "clone", "-q", url, str(tmp)], capture_output=True, text=True, errors="replace", timeout=900))
    if r.returncode == 0:
        _run(tmp, "remote", "set-url", "origin", NOTES_URL)
        try:
            tmp.rename(root)
        except OSError:
            pass                                            # another command cloned it first
    shutil.rmtree(tmp, ignore_errors=True)
    if not root.is_dir():
        return f"notes: clone of {NOTES_URL} into {root} failed: {r.stderr.strip()[-300:]}"
    return f"notes: cloned {NOTES_URL} into {root}"


def _secret_values(environ: dict | None = None) -> list[str]:
    env = os.environ if environ is None else environ
    return [v for k, v in env.items() if SECRET_ENV_RE.search(k) and v and len(v) >= 16]


def find_keys(text: str, secrets: list[str]) -> list[str]:
    """The names of the key patterns (and `env-secret` for a secret env var's value) that `text` contains."""
    hits = [name for name, rx in KEY_PATTERNS.items() if rx.search(text)]
    return hits + (["env-secret"] if any(s in text for s in secrets) else [])


def _changed(root: Path, paths: list[str] | None, max_bytes: int) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    """(paths to commit, paths skipped as over max_bytes, evidence files left out, {staged rename's new path: its old path}) among
    the working tree's changes, limited to `paths` when given."""
    st = _out(root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", *(paths or []))
    entries, keep, big, evidence, renames, i = (st or "").split("\0"), [], [], [], {}, 0
    while i < len(entries):
        e = entries[i]
        i += 1
        if not e:
            continue
        code, path = e[:2], e[3:]
        if code[0] in "RC":
            keep.append(entries[i])
            if code[0] == "R":
                renames[path] = entries[i]
            i += 1
        if path in NEVER_COMMIT:
            continue
        p = root / path
        if p.is_file() and is_evidence(path):
            evidence.append(path)
            continue
        (big if p.is_file() and p.stat().st_size > max_bytes else keep).append(path)
    return keep, big, evidence, renames


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def created_at(text: str) -> tuple[str, dt.datetime] | None:
    """(as written, time) of the front matter's `created:` (ISO, seconds optional, UTC unless it gives an offset), or None."""
    end = text.find("\n---", 3) if text.startswith("---\n") else -1
    m = CREATED_RE.search(text[4:end]) if end >= 0 else None
    if m is None:
        return None
    try:
        t = dt.datetime.fromisoformat(f"{m[2]}T{m[3]}{m[4] if m[4] and m[4].upper() != 'Z' else ''}")
    except ValueError:
        return None
    return m[1], t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def future_stamps(root: Path, paths: list[str], now: dt.datetime) -> dict[str, str]:
    """{path: why} for the lane notes among `paths` whose filename stamp (a path HEAD lacks) or `created:` (a value HEAD's copy
    lacks) is more than FUTURE_TOLERANCE after `now` (UTC); `why` gives the value `date -u` prints at `now`."""
    out = {}
    for path in paths:
        p = root / path
        if not (path.startswith("lanes/") and path.endswith(".md") and p.is_file()):
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        old = _out(root, "show", f"HEAD:{path}")
        found = []
        if old is None and (m := STAMP_RE.match(p.name)):
            try:
                found.append(("stamp", m[1], dt.datetime.strptime(m[1], "%Y%m%dT%H%MZ").replace(tzinfo=dt.timezone.utc), "%Y%m%dT%H%MZ"))
            except ValueError:
                pass
        c, prev = created_at(text), created_at(old) if old is not None else None
        if c and (prev is None or prev[1] != c[1]):
            found.append(("created:", *c, "%Y-%m-%dT%H:%MZ"))
        why = [f"{what} {was} is {math.ceil((t - now).total_seconds() / 60)} min ahead; use {now.strftime(fmt)} (date -u +{fmt})"
               for what, was, t, fmt in found if t - now > FUTURE_TOLERANCE]
        if why:
            out[path] = "; ".join(why)
    return out


def header_issues(path: str, text: str, now: dt.datetime) -> tuple[list[str], list[str]]:
    """(refusals, warnings) for one note's front matter (module docstring); ([], []) for a note without front matter."""
    lines = text.split("\n")
    if lines[0].rstrip() != "---":
        return [], []
    end = next((i for i in range(1, len(lines)) if lines[i].rstrip() == "---"), None)
    if end is None:
        return ["front matter opened with --- is never closed: add a --- line under the header"], []
    refuse = [f"front matter line {n} is not `key: value`: {line.strip()[:60]!r}"
              for n, line in enumerate(lines[1:end], 2) if not FRONT_LINE_RE.match(line.rstrip("\r"))]
    block, warn, c = "\n".join(lines[1:end]), [], created_at(text)
    if (m := re.search(r"^created:[ \t]*(.*?)[ \t\r]*$", block, re.M)) and c is None:
        refuse.append(f"created: {m[1]!r} is not a UTC time; use {now:%Y-%m-%dT%H:%MZ} (date -u +%Y-%m-%dT%H:%MZ)")
    parts = path.split("/")
    if parts[0] == "lanes" and len(parts) > 2 and (m := FRONT_LANE_RE.search(block)) and m[1] != parts[1]:
        warn.append(f"lane: {m[1]} but the note is in lanes/{parts[1]}/" + (" (a handoff's lane is its recipient)" if MAIL_RE.search(parts[-1]) else ""))
    if c and (s := STAMP_RE.match(parts[-1])):
        try:
            off = c[1] - dt.datetime.strptime(s[1], "%Y%m%dT%H%MZ").replace(tzinfo=dt.timezone.utc)
        except ValueError:
            off = dt.timedelta(0)
        if abs(off) > CREATED_STAMP_SLACK:
            warn.append(f"created: {c[0]} is {abs(off).total_seconds() / 3600:.0f} h from the file name's stamp {s[1]}")
    return refuse, warn


def _known_blobs(root: Path, paths: list[str]) -> set[str]:
    """The paths whose current content the repository already holds as a blob (a move, a revert, an unchanged file)."""
    hashes = _run(root, "hash-object", "--stdin-paths", input="\n".join(paths) + "\n").stdout.split()
    if len(hashes) != len(paths):
        return set()
    check = _run(root, "cat-file", "--batch-check", input="\n".join(hashes) + "\n").stdout.splitlines()
    missing = {line.split()[0] for line in check if line.endswith(" missing")}
    return {p for p, h in zip(paths, hashes) if h not in missing}


def header_problems(root: Path, paths: list[str], now: dt.datetime) -> tuple[dict[str, str], list[str]]:
    """({path: why} to leave out, [`path: warning`]) for the notes among `paths` with new content: the problems of
    `header_issues` that HEAD's copy of the path did not already have."""
    notes = [p for p in paths if p.startswith(("lanes/", "kb/")) and p.endswith(".md") and (root / p).is_file()]
    known = _known_blobs(root, notes) if notes else set()
    refused, warned = {}, []
    for path in notes:
        if path in known:
            continue
        try:
            refuse, warn = header_issues(path, (root / path).read_text(errors="replace"), now)
        except OSError:
            continue
        if (refuse or warn) and (old := _out(root, "show", f"HEAD:{path}")) is not None:
            old_refuse, old_warn = header_issues(path, old, now)
            refuse, warn = [x for x in refuse if x not in old_refuse], [x for x in warn if x not in old_warn]
        if refuse:
            refused[path] = "; ".join(refuse)
        warned += [f"{path}: {x}" for x in warn]
    return refused, warned


def commit(root: Path, paths: list[str] | None = None, message: str | None = None, max_bytes: int = MAX_BYTES,
           now: dt.datetime | None = None) -> dict:
    """Commit the caller's changes (all, or under `paths`) but evidence and renders (`evidence`), lane notes stamped ahead of `now`
    (`future`: `path: why`) and notes with a broken header (`header`: `path: why`); `warn` lists `path: warning`.  Retries a held
    index lock (another writer on this machine) briefly."""
    keep, big, evidence, renames = _changed(root, paths, max_bytes)
    now = now or _utcnow()
    future = future_stamps(root, keep, now)
    header, warn = header_problems(root, [p for p in keep if p not in future], now)
    if future or header or evidence:
        out = {*future, *header, *evidence}
        out |= {renames[p] for p in out if p in renames}                  # a file left out keeps its old path: never half a move
        keep = [p for p in keep if p not in out]
        _run(root, "--literal-pathspecs", "reset", "-q", "--", *sorted(out))  # a staged copy would be committed anyway
    left = {"skipped": big, "evidence": evidence, "future": [f"{p}: {why}" for p, why in future.items()],
            "header": [f"{p}: {why}" for p, why in header.items()], "warn": warn}
    if not keep:
        return {"commit": None, "paths": 0, **left}
    msg = message or f"notes sync {time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())}: {len(keep)} paths"
    moved_from = set(renames.values())          # a staged move's old path: its removal is staged already, and `add` finds no such path
    to_add = [p for p in keep if p not in moved_from]
    for attempt in range(10):
        add = _run(root, "--literal-pathspecs", "add", "-A", "--pathspec-from-file=-", "--pathspec-file-nul", input="\0".join(to_add)) \
            if to_add else None
        if add is None or add.returncode == 0:
            c = _run(root, "commit", "-q", "--no-verify", "-m", msg)
            if c.returncode == 0:
                return {"commit": (_out(root, "rev-parse", "--short", "HEAD") or "").strip(), "paths": len(keep), **left}
            if "index.lock" not in c.stderr and "nothing to commit" in (c.stdout + c.stderr):
                return {"commit": None, "paths": 0, **left}
            err = c.stderr
        else:
            err = add.stderr
        if "index.lock" not in err:
            raise RuntimeError(f"git commit failed: {err.strip()[-300:]}")
        time.sleep(0.2 * (attempt + 1))
    raise RuntimeError("git index is locked by another writer (index.lock); try again")


def outgoing_keys(root: Path, base: str | None, secrets: list[str]) -> list[str]:
    """`path: pattern` for key material in lines the commits after `base` (all history when None) add."""
    rng = [f"{base}..HEAD"] if base else ["HEAD"]
    log = _out(root, "log", "-p", "--no-color", "--format=", "--unified=0", *rng) or ""
    hits, path = [], None
    for line in log.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else line[4:]
        elif line.startswith("+") and not line.startswith("+++"):
            for name in find_keys(line, secrets):
                if f"{path}: {name}" not in hits:
                    hits.append(f"{path}: {name}")
    return hits


def outgoing_citations(root: Path, base: str | None) -> dict[str, list[str]]:
    """{note path: [`art:<hex>` | run id, ...]} cited in the lines the commits after `base` (the last commit when None: a first
    push) add to notes under lanes/ and kb/; a moved note adds no lines."""
    rng = [f"{base}..HEAD"] if base else ["-1", "HEAD"]
    log = _out(root, "log", "-p", "-M", "--no-color", "--format=", "--unified=0", *rng, "--", "*.md") or ""
    out: dict[str, list[str]] = {}
    path = None
    for line in log.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") and line[6:].startswith(("lanes/", "kb/")) else None
        elif path and line.startswith("+"):
            for cited in ["art:" + m[1] for m in CITE_ART_RE.finditer(line)] + CITE_RUN_RE.findall(line):
                if cited not in out.setdefault(path, []):
                    out[path].append(cited)
    return out


def cite_check_on() -> bool:
    return os.environ.get(CITE_ENV, "").strip().lower() not in ("0", "off", "no", "false")


def missing_citations(cited: list[str], remote=None, workers: int = 8, wait_s: float = CITE_WAIT_S) -> tuple[set[str], str | None]:
    """(the ids among `cited` the store's remote does not have, why the answer is partial or None), `workers` lookups at a time
    for at most `wait_s` (daemon threads: a remote that hangs never holds up the exit).  (set(), None) with no usable remote: none
    configured, or its credentials absent here."""
    from .store.labels_sync import remote_labels_for  # noqa: PLC0415
    from .store.local import attempt_key  # noqa: PLC0415
    from .store.remote import ConfigError, load_remote  # noqa: PLC0415

    if remote is None:
        try:
            remote = load_remote()
        except ConfigError:
            return set(), None
    if remote is None or not cited:
        return set(), None

    def have(c: str) -> bool:
        if c.startswith("art:"):
            return bool(remote.list(f"manifests/{c[4:]}"))
        return remote.head(attempt_key(c)) is not None or bool(remote_labels_for(remote, c))

    todo, answers = list(cited), {}

    def work() -> None:
        while todo:
            try:
                c = todo.pop()
            except IndexError:
                return
            try:
                answers[c] = have(c)
            except Exception as e:  # noqa: BLE001
                answers[c] = e

    threads = [threading.Thread(target=work, daemon=True) for _ in range(max(1, min(workers, len(cited))))]
    for t in threads:
        t.start()
    deadline = time.monotonic() + wait_s
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    got = dict(answers)
    failed = [c for c in cited if isinstance(got.get(c), Exception)]
    late = [c for c in cited if c not in got]
    why = ([f"the store's remote did not answer for {len(failed)} cited ids ({type(got[failed[0]]).__name__})"] if failed else []) \
        + ([f"{len(late)} cited ids not answered within {wait_s:g} s"] if late else [])
    return {c for c in cited if got.get(c) is False}, "; ".join(why) or None


def citation_warnings(cites: dict[str, list[str]], remote=None) -> list[str]:
    """`path: cites ...` for each note among `cites` (`outgoing_citations`) citing an `art:` id or run id the store's remote does
    not have, with how to publish it; a `citation check: ...` line when the check was cut short (module docstring)."""
    ids = list(dict.fromkeys(c for v in cites.values() for c in v))
    if not ids or not cite_check_on():
        return []
    missing, why = missing_citations(ids[:CITE_MAX], remote)
    out = []
    for path, v in cites.items():
        if gone := [c for c in v if c in missing]:
            hints = [CITE_HINTS[k] for k in ("run", "art") if any(c.startswith("art:") == (k == "art") for c in gone)]
            out.append(f"{path}: cites {', '.join(gone)}, which the evidence store does not have ({'; '.join(hints)})")
    if why:
        out.append(f"citation check: {why}")
    if len(ids) > CITE_MAX:
        out.append(f"citation check: {len(ids) - CITE_MAX} more cited ids not looked up (at most {CITE_MAX} a sync)")
    return out


def git_running(root: Path, procs: Path = Path("/proc")) -> int | None:
    """The pid of a git process at work in the repository (its working directory is `root` or under it); None when there is
    none, and where there is no /proc to look in."""
    real = Path(root).resolve()
    for proc in procs.glob("[0-9]*"):
        try:
            if (proc / "comm").read_text().strip() != "git":
                continue
            cwd = Path(os.readlink(proc / "cwd")).resolve()
        except OSError:
            continue
        if cwd == real or real in cwd.parents:
            return int(proc.name)
    return None


def clear_stale_lock(root: Path, now: float | None = None, procs: Path = Path("/proc")) -> bool:
    """Remove the repository's index.lock when it is over STALE_LOCK_S old and no git process works in the repository; whether it
    did.  A git killed mid-write leaves it, and every later sync then fails at its rebase until someone clears it by hand."""
    gitdir = _gitdir(root)
    if gitdir is None:
        return False
    lock = gitdir / "index.lock"
    try:
        if (now if now is not None else time.time()) - lock.stat().st_mtime < STALE_LOCK_S:
            return False
    except FileNotFoundError:
        return False
    if git_running(root, procs) is not None:
        return False
    lock.unlink(missing_ok=True)
    return True


def _gitdir(root: Path) -> Path | None:
    out = (_out(root, "rev-parse", "--absolute-git-dir") or "").strip()
    return Path(out) if out else None


def _branch(root: Path) -> str:
    """The branch HEAD is on; '' when HEAD is detached."""
    return (_out(root, "symbolic-ref", "-q", "--short", "HEAD") or "").strip()


class Busy(Exception):
    """Another sync held the clone's lock for the whole wait; the text names it."""


def _who() -> str:
    """This sync, as one that waits for it names it: its pid and machine, the agent that runs it (`remote.sender`), else the
    lane or the user, and when it took the lock."""
    from .remote import sender

    who = sender() or os.environ.get("RESEARCH_LANE") or os.environ.get("USER") or "an unnamed caller"
    return f"pid {os.getpid()} on {socket.gethostname()}, {who}, since {_utcnow():%H:%M:%SZ}"


def _lock(gitdir: Path, wait_s: float) -> int:
    """A descriptor holding the clone's sync lock, an exclusive flock on `<git dir>/notes-sync.lock`, after waiting at most
    `wait_s` for another sync of the clone to let it go; `Busy` after that.  The kernel drops the lock when its holder exits,
    however it exits, so a killed sync leaves nothing to clear, and git never inherits it.  The holder writes who it is into the
    file (`_who`), and a sync that waits says once, on stderr, whom it waits for."""
    fd = os.open(gitdir / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o666)
    start, said = time.monotonic(), False
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            holder = os.pread(fd, 400, 0).decode(errors="replace").strip()      # empty for the instant before it writes it
            if time.monotonic() - start >= wait_s:
                os.close(fd)
                raise Busy(holder or "which did not say who it is") from None
            if holder and not said:
                print(f"notes sync: waiting up to {wait_s:g} s for another sync of this clone to end: {holder}", file=sys.stderr, flush=True)
                said = True
            time.sleep(LOCK_POLL_S)
    os.ftruncate(fd, 0)
    os.pwrite(fd, _who().encode(), 0)
    return fd


def in_progress(gitdir: Path) -> str | None:
    """"rebase" or "merge" when the repository is in the middle of one (its state is in the git directory), else None."""
    if (gitdir / "rebase-merge").is_dir() or (gitdir / "rebase-apply" / "rebasing").exists():
        return "rebase"
    return "merge" if (gitdir / "MERGE_HEAD").exists() else None


def notes_branch(root: Path, remote: str = "origin") -> str | None:
    """The branch a clone whose HEAD is detached belongs on: the one the remote's HEAD names (`refs/remotes/<remote>/HEAD`, which
    a clone sets), else the only local branch that is not a `rescue-` one; None when neither says."""
    local = [b for b in (_out(root, "for-each-ref", "--format=%(refname:short)", "refs/heads") or "").split() if not b.startswith(RESCUE)]
    named = (_out(root, "symbolic-ref", "-q", "--short", f"refs/remotes/{remote}/HEAD") or "").strip().removeprefix(f"{remote}/")
    return named if named in local else local[0] if len(local) == 1 else None


def _merged(path: Path) -> bytes | None:
    """A file's bytes with the labels off its conflict markers, which differ with the command that wrote them; None when it is
    not a file."""
    return re.sub(rb"(?m)^(<{7}|\|{7}|>{7}) .*$", rb"\1", path.read_bytes()) if path.is_file() else None


def _keep(root: Path, gitdir: Path, found: bool, orig: str = "") -> dict | None:
    """What ending a rebase or merge would discard from the working tree, as a commit on HEAD (`commit`) made through an index of
    its own, so the clone's index and files are not touched: every tracked file as the working tree has it.  `paths` are the
    files changed since they were last staged, which is what someone wrote under the operation: each is put back after (`base`
    is the tree they are a change to).  A file left in conflict is the stopped step's own, and git writes it again at the next
    conflict; with `found` (an operation that may have sat stopped for long), `aside` are the conflicted files that are no
    longer what git wrote there (`git checkout --merge` writes that again, into a directory of its own), which someone wrote
    into since: the commit holds those as they are, and they are not put back.  So it is with `in_way`, the untracked files at
    paths the commit a found rebase began from (`orig`) has: a rebase takes a file its branch added out of the tree until it
    picks that commit again, a line appended meanwhile makes a new file there, and git will not abort over it.  None when there
    is nothing to keep."""
    entries = (_out(root, "status", "--porcelain=v1", "-z", f"--untracked-files={'all' if found and orig else 'no'}") or "").split("\0")
    unmerged, paths, untracked, i = [], [], [], 0
    while i < len(entries):
        code, path = entries[i][:2], entries[i][3:]
        i += 2 if code[:1] in ("R", "C") else 1
        if path and (code in UNMERGED or code[1] != " "):
            (unmerged if code in UNMERGED else untracked if code == "??" else paths).append(path)
    in_way = sorted(set(untracked) & set((_out(root, "ls-tree", "-r", "-z", "--name-only", orig) or "").split("\0"))) if untracked else []
    index, aside = gitdir / f"notes-sync.index.{os.getpid()}", []
    try:
        shutil.copyfile(gitdir / "index", index)
        if found and unmerged:
            with tempfile.TemporaryDirectory(prefix="notes-sync-") as tmp:
                _run(root, "-c", "core.hooksPath=/dev/null", f"--work-tree={tmp}", "--literal-pathspecs", "checkout", "-q", "--merge",
                     "--", *unmerged, index=index)
                aside = [p for p in unmerged if (was := _merged(Path(tmp) / p)) is None or was != _merged(root / p)]
        if not (paths or aside or in_way):
            return None
        if unmerged:
            _run(root, "update-index", "--force-remove", "-z", "--stdin", input="\0".join(unmerged), index=index)
        base = _run(root, "write-tree", index=index).stdout.strip()
        _run(root, "add", "-u", index=index)
        if there := [p for p in aside + in_way if (root / p).is_file()]:
            _run(root, "--literal-pathspecs", "add", "-f", "--", *there, index=index)
        tree = _run(root, "write-tree", index=index).stdout.strip()
    finally:
        for p in (index, index.with_name(f"{index.name}.lock")):
            p.unlink(missing_ok=True)
    made = _run(root, "commit-tree", tree, "-p", "HEAD", "-m", "notes sync: what the working tree held when a rebase or merge was ended")
    if not (base and tree and made.returncode == 0):
        raise RuntimeError(f"could not save the working tree before ending a rebase or merge; nothing was changed: {made.stderr.strip()[-200:]}")
    return {"commit": made.stdout.strip(), "base": base, "paths": paths, "aside": aside, "in_way": in_way}


def _rescue(root: Path, commit: str) -> str:
    """A new local branch `rescue-<UTC time>` at `commit`; its name."""
    stamp = f"{RESCUE}{_utcnow():%Y%m%dT%H%M%SZ}"
    for name in (stamp, *(f"{stamp}-{n}" for n in range(2, 100))):
        if _run(root, "branch", "-q", "--no-track", name, commit).returncode == 0:
            return name
    raise RuntimeError(f"could not make a branch {stamp} at {commit[:12]} to keep the working tree on; nothing was changed")


def _patch(root: Path, old: str, new: str, *paths: str) -> bytes:
    """The change from tree `old` to tree `new` (in `paths`, else everywhere) as a patch `git apply` takes."""
    return subprocess.run(["git", "-C", str(root), "-c", "diff.noprefix=false", "--literal-pathspecs", "diff", "--binary", "--no-color",
                           "--no-ext-diff", "--no-textconv", old, new, "--", *paths], capture_output=True, timeout=120).stdout


def _apply(root: Path, patch: bytes) -> bool:
    """Apply `patch` to the working tree, whole or not at all (git refuses a patch any hunk of which does not fit, and changes
    nothing); True when the tree already holds what it adds, and when it is applied."""
    if not patch.strip():
        return True
    git = ["git", "-C", str(root), "apply", "--whitespace=nowarn"]
    if subprocess.run([*git, "--reverse", "--check"], input=patch, capture_output=True, timeout=120).returncode == 0:
        return True
    return subprocess.run(git, input=patch, capture_output=True, timeout=120).returncode == 0


def _unstash(root: Path, stash: str) -> bool:
    """Put back the uncommitted changes a rebase stashed before it began (its autostash); whether they are back.  The stash goes
    into `git stash list` first, and stays there when it does not apply."""
    if _run(root, "stash", "store", "-m", "notes sync: the autostash of a rebase that never ended", stash).returncode:
        return False
    if not _apply(root, _patch(root, f"{stash}^1", stash)):
        return False
    if (_out(root, "rev-parse", "-q", "--verify", "refs/stash") or "").strip() == stash:
        _run(root, "stash", "drop", "-q")
    return True


def end_operation(root: Path, gitdir: Path, op: str, found: bool = False) -> dict:
    """End the rebase or merge in progress in the clone (`git rebase --abort`, `git merge --abort`) and keep what that discards.
    git's abort resets the tracked files, so lines someone wrote into one under the operation would be gone, held by no commit:
    they are saved first (`_keep`, on a new `rescue-` branch) and put back after as a patch, and the branch is deleted once all
    of it is back.  `found`: the operation was found in progress with no process running it, however long ago it stopped; then
    a conflicted file someone wrote into since, a file written where a rebase had taken one out of the tree (removed from the
    tree once the branch holds it, so the abort can run), and a commit made on a rebase's detached HEAD that the branch has no
    equal of, stay on the rescue branch.  A rebase that died before it wrote where it came from (only its autostash is there, and HEAD
    never moved) is cleared by removing its directory, its autostash put back (`_unstash`).
    The report: `over` (nothing is in progress now), `put_back` (files), and `lost`: None when everything local is back in the
    tree, else what is not and where it is kept, for the caller to fail with."""
    state = next((gitdir / d for d in ("rebase-merge", "rebase-apply") if (gitdir / d).is_dir()), None) if op == "rebase" else None

    def read(name: str) -> str:
        return (state / name).read_text().strip() if state and (state / name).is_file() else ""

    head, onto, orig, autostash = (_out(root, "rev-parse", "HEAD") or "").strip(), read("onto"), read("orig-head"), read("autostash")
    picked = {f[1] for f in (line.split() for line in read("rewritten-list").splitlines()) if len(f) == 2}
    out = {"over": True, "put_back": 0, "lost": None}
    try:
        kept = _keep(root, gitdir, found, orig)
        rescue = _rescue(root, kept["commit"]) if kept else None
    except (OSError, RuntimeError) as e:
        return {**out, "over": False, "lost": f"the {op} in progress was not ended ({e})"}
    in_way = kept["in_way"] if kept else []
    for p in in_way:
        (root / p).unlink()                                     # the rescue branch holds it, and git's abort refuses to write over it
    aborted = _run(root, op, "--abort")
    if aborted.returncode and in_progress(gitdir):
        for p in in_way:
            (root / p).write_bytes(subprocess.run(["git", "-C", str(root), "cat-file", "blob", f"{kept['commit']}:{p}"],
                                                  capture_output=True, check=True, timeout=120).stdout)
        if rescue:
            _run(root, "branch", "-q", "-D", rescue)            # the abort changed no file, so the tree still holds all of it
        if state is None or read("head-name"):
            why = " ".join(aborted.stderr.split())[-300:] or "git gave no reason"
            return {**out, "over": False, "lost": f"the {op} in progress could not be ended ({why}); no file was changed"}
        shutil.rmtree(state)
        if autostash and not _unstash(root, autostash):
            out["lost"] = ("the uncommitted changes a rebase stashed before it died no longer apply to the files: they are the "
                           "first entry of `git stash list` (git stash pop, then resolve)")
        return out
    missing = [p for p in (kept["paths"] if kept else []) if not _apply(root, _patch(root, kept["base"], kept["commit"], p))]
    out["put_back"] = len(kept["paths"]) - len(missing) if kept else 0
    aside = kept["aside"] if kept else []
    extra = [line[2:] for line in (_out(root, "cherry", orig, head, onto) or "").splitlines()
             if line.startswith("+ ") and line[2:] not in picked] if found and head and onto and orig else []
    if not (missing or aside or in_way or extra):
        if rescue:
            _run(root, "branch", "-q", "-D", rescue)
        return out
    rescue = rescue or _rescue(root, head)
    what = [f"{len(x)} {name} ({', '.join(x[:3])})" for x, name in (
        (missing, f"file(s) written while the {op} was in progress could not be put back"),
        (aside, f"file(s) the {op} left in conflict were written into since"),
        (in_way, f"file(s) were written where the {op} had taken a file of the branch's out of the tree"),
        ([c[:7] for c in extra], "commit(s) made on its detached HEAD are not on the branch")) if x]
    return {**out, "lost": f"{'; '.join(what)}: the branch {rescue} keeps them as they were (git diff HEAD {rescue}); "
                           f"delete it once they are back"}


def settle(root: Path, gitdir: Path, remote: str = "origin") -> tuple[str, list[str], str | None]:
    """The clone back on its branch with nothing in progress, whatever an earlier git left behind, before a sync commits:
    (the branch, one line for each thing put right, why it could not be: then the sync commits nothing).  The caller holds the
    clone's lock, so no other sync is at work in it.
    A rebase or merge in progress that no git process is running is ended (`end_operation`).  A HEAD left detached with nothing
    in progress and the notes branch (`notes_branch`) at or behind it, as a rebase leaves it that died after its first step or
    lost its state directory, gets the branch checked out where HEAD is (`git checkout -B`), which moves no file and drops no
    commit.  Any other detached HEAD is someone's to resolve, and is refused as before."""
    said: list[str] = []
    op, head = in_progress(gitdir), (_out(root, "rev-parse", "--short", "HEAD") or "").strip()
    if not op and (branch := _branch(root)):
        return branch, said, None
    start = time.monotonic()
    while (pid := git_running(root)) is not None and time.monotonic() - start < GIT_WAIT_S:
        time.sleep(LOCK_POLL_S)                                 # a git that only reads is gone at the next look
    if pid is not None:
        return "", said, (f"a {op} is in progress" if op else "HEAD is detached") + \
            f" and git (pid {pid}) is at work in this clone; nothing committed: sync again when it has ended"
    if op:
        ended = end_operation(root, gitdir, op, found=True)
        if not ended["over"]:
            return "", said, f"{ended['lost']}; nothing committed"
        said.append(f"ended a {op} left in progress that no process was running (HEAD was at {head}): the local commits are kept"
                    + (f", and {ended['put_back']} file(s) written under it are back in the tree" if ended["put_back"] else ""))
        if ended["lost"]:
            return "", said, f"{ended['lost']}; nothing committed"
    branch = _branch(root)
    if not branch:
        want = notes_branch(root, remote)
        if not want or _run(root, "merge-base", "--is-ancestor", f"refs/heads/{want}", "HEAD").returncode:
            return "", said, f"HEAD is detached at {head}" + (f", which {want} is not behind" if want else "") + \
                "; nothing committed: check out the notes branch"
        back = _run(root, "checkout", "-q", "-B", want, "HEAD")
        if back.returncode:
            why = (back.stderr.strip().splitlines() or ["git gave no reason"])[-1][:200]
            return "", said, f"HEAD is detached at {head} and {want} could not be checked out there ({why}); nothing committed"
        said.append(f"HEAD was detached at {head} with nothing in progress and {want} at or behind it: checked {want} out there "
                    f"(git checkout -B {want} HEAD), which moved no file and dropped no commit")
        branch = want
    return branch, said, None


def sync(root: Path, paths: list[str] | None = None, message: str | None = None, retries: int = 5, remote: str = "origin",
         max_bytes: int = MAX_BYTES, secrets: list[str] | None = None, now: dt.datetime | None = None) -> dict:
    """Commit, then fetch / rebase / push with bounded retries (module docstring), one sync of a clone at a time (`_lock`)
    and from a clone put right first (`settle`).  The report: `commit`, `paths`, `skipped`,
    `evidence` (evidence and renders left out), `future` (notes left out, stamped ahead of `now`), `header` (notes left out, broken
    front matter), `warn` (SYNC-WARN lines: front matter, what `settle` put right, and once pushed, citations the store lacks),
    `pushed` (bool), `attempts`, and on failure `error` plus `conflict` (files)
    or `keys` (hits); `rc` is the exit code."""
    root = Path(root)
    gitdir = _gitdir(root)
    if gitdir is None:
        return {"rc": 2, "error": f"{root} is not a git repository"}
    try:
        lock = _lock(gitdir, LOCK_WAIT_S)
    except Busy as e:
        return {"rc": 1, "pushed": False, "attempts": 0, "commit": None,
                "error": f"another sync of this clone was still running after {LOCK_WAIT_S:g} s ({e}); nothing committed: sync again"}
    try:
        return _sync(root, gitdir, paths, message, retries, remote, max_bytes, secrets, now)
    finally:
        os.close(lock)


def _sync(root: Path, gitdir: Path, paths: list[str] | None, message: str | None, retries: int, remote: str, max_bytes: int,
          secrets: list[str] | None, now: dt.datetime | None) -> dict:
    """`sync`, holding the clone's lock."""
    stale = clear_stale_lock(root)
    branch, said, error = settle(root, gitdir, remote)
    if stale:
        said.insert(0, f"removed a stale .git/index.lock (over {STALE_LOCK_S} s old, no git running here)")
    if error:
        return {"rc": 2, "pushed": False, "attempts": 0, "commit": None, "warn": said, "error": error}
    rep = {"rc": 0, "pushed": False, "attempts": 0, **commit(root, paths, message, max_bytes, now)}
    rep["warn"] = [*rep.get("warn", []), *said]
    if rep["future"] or rep["header"]:
        rep["rc"] = FUTURE_RC
    if _out(root, "remote", "get-url", remote) is None:
        return {**rep, "note": f"no remote {remote!r}: committed locally only"}
    secrets = _secret_values() if secrets is None else secrets
    tracking = f"refs/remotes/{remote}/{branch}"

    def net(verb: str, refspec: str) -> subprocess.CompletedProcess:
        return _authed(lambda: remote_target(root, remote), lambda opts, where: _run(root, *opts, verb, "-q", where, refspec, timeout=300))

    for n in range(retries + 1):
        rep["attempts"] = n + 1
        f = net("fetch", f"+refs/heads/{branch}:{tracking}")
        has_remote = f.returncode == 0
        if not has_remote and "couldn't find remote ref" not in f.stderr:
            if n < retries and not refused(f.stderr):
                time.sleep(min(2.0, 0.1 * 2 ** n))
                continue
            return {**rep, "rc": 1, "error": f"git fetch {remote} failed: {f.stderr.strip()[-300:]}{token_hint(root, remote)}"}
        base = tracking if has_remote else None
        if has_remote and _run(root, "merge-base", "--is-ancestor", tracking, "HEAD").returncode != 0:
            rb = _run(root, "rebase", "--autostash", tracking, timeout=300)
            if rb.returncode:
                conflict = [x for x in (_out(root, "diff", "--name-only", "--diff-filter=U") or "").splitlines() if x]
                ended = end_operation(root, gitdir, "rebase") if in_progress(gitdir) else {"over": True, "put_back": 0, "lost": None}
                if not _branch(root) and not in_progress(gitdir):
                    # an abort that lost its rebase-merge directory (as on NFS) leaves HEAD detached at the remote branch
                    _run(root, "checkout", "-q", branch)
                if ended["lost"]:
                    return {**rep, "rc": 3, "conflict": conflict, "error": f"the rebase stopped, and {ended['lost']}; then sync again"}
                if ended["put_back"] and not conflict and n < retries:
                    continue                                    # it stopped at a file someone wrote under it: again, with that stashed
                return {**rep, "rc": 3, "conflict": conflict,
                        "error": "rebase conflict; nothing was dropped (the local commit is kept, the rebase aborted): resolve "
                                 + (", ".join(conflict) or rb.stderr.strip()[-200:]) + ", then sync again"}
        if base and (_out(root, "rev-parse", "HEAD") or "").strip() == (_out(root, "rev-parse", tracking) or "").strip():
            return {**rep, "pushed": False, "note": "already on the remote"}
        keys = outgoing_keys(root, base, secrets)
        if keys:
            return {**rep, "rc": 4, "keys": keys, "error": "key material in outgoing commits; nothing pushed: " + "; ".join(keys[:5])}
        cites = outgoing_citations(root, base) if cite_check_on() else {}
        p = net("push", f"HEAD:refs/heads/{branch}")
        if p.returncode == 0:
            net("fetch", f"+refs/heads/{branch}:{tracking}")
            return {**rep, "pushed": True, "warn": rep["warn"] + citation_warnings(cites)}
        if refused(p.stderr) or n == retries and not any(k in p.stderr for k in RACE):
            return {**rep, "rc": 1, "error": f"git push failed (local commit kept): {p.stderr.strip()[-300:]}{token_hint(root, remote)}"}
        time.sleep(min(2.0, 0.1 * 2 ** n))
    return {**rep, "rc": 1, "error": f"lost the push race {retries + 1} times (local commit kept); sync again"}


def lag(root: Path, remote: str = "origin") -> tuple[int, int] | None:
    """(ahead, behind): commits HEAD has that the remote-tracking branch lacks, and the reverse; None without one."""
    branch = (_out(root, "symbolic-ref", "--short", "HEAD") or "").strip()
    counts = _out(root, "rev-list", "--left-right", "--count", f"HEAD...refs/remotes/{remote}/{branch}") if branch else None
    if not counts or len(counts.split()) != 2:
        return None
    ahead, behind = counts.split()
    return int(ahead), int(behind)


def scan(root: Path, max_bytes: int = MAX_BYTES, secrets: list[str] | None = None) -> dict:
    """Every blob in the history: `big` (path, bytes) over max_bytes, `keys` (path: pattern) with key material."""
    secrets = _secret_values() if secrets is None else secrets
    objs = _out(root, "rev-list", "--objects", "--all") or ""
    paths: dict[str, str] = {}
    for line in objs.splitlines():
        sha, _, path = line.partition(" ")
        if path:
            paths.setdefault(sha, path)
    if not paths:
        return {"blobs": 0, "big": [], "keys": []}
    check = _run(root, "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)", input="\n".join(paths) + "\n")
    blobs = [(s, int(n)) for s, t, n in (x.split() for x in check.stdout.splitlines() if x.count(" ") == 2) if t == "blob"]
    big = sorted((paths[s], n) for s, n in blobs if n > max_bytes)
    keys = []
    small = [s for s, n in blobs if n <= max_bytes]
    if small:
        proc = subprocess.run(["git", "-C", str(root), "cat-file", "--batch"], input=("\n".join(small) + "\n").encode(),
                              capture_output=True, timeout=600)
        data, i = proc.stdout, 0
        while i < len(data):
            nl = data.index(b"\n", i)
            sha, _, size = data[i:nl].decode().split()
            body = data[nl + 1:nl + 1 + int(size)]
            i = nl + 1 + int(size) + 1
            if b"\0" in body[:8192]:
                continue
            for name in find_keys(body.decode(errors="replace"), secrets):
                keys.append(f"{paths[sha]}: {name}")
    return {"blobs": len(blobs), "big": [{"path": p, "bytes": n} for p, n in big], "keys": sorted(set(keys))}


def render_sync(rep: dict) -> str:
    head = (f"notes sync: committed {rep['commit']} ({rep['paths']} paths)" if rep.get("commit") else "notes sync: nothing new to commit")
    head += "; pushed" if rep.get("pushed") else ""
    if rep.get("attempts", 0) > 1:
        head += f" after {rep['attempts']} attempts"
    parts = [head]
    if rep.get("skipped"):
        parts.append(f"skipped {len(rep['skipped'])} over 1 MB (put them in the store): {', '.join(rep['skipped'][:3])}")
    if rep.get("evidence"):
        parts.append(f"left out {len(rep['evidence'])} evidence/render files (the evidence store holds them: research data put ... "
                     f"--preserve, cite the art: id): {', '.join(rep['evidence'][:3])}")
    if rep.get("future"):
        parts.append(f"left out {len(rep['future'])} stamped ahead of the clock (SYNC-REFUSED)")
    if rep.get("header"):
        parts.append(f"left out {len(rep['header'])} with broken front matter (SYNC-REFUSED)")
    if rep.get("warn"):
        parts.append(f"{len(rep['warn'])} warning{'s' if len(rep['warn']) > 1 else ''} (SYNC-WARN)")
    if rep.get("note"):
        parts.append(rep["note"])
    if rep.get("error"):
        parts.append(f"FAILED: {rep['error']}")
    return "; ".join(parts)


def refused_lines(rep: dict) -> list[str]:
    """`SYNC-REFUSED` for each note left out, then `SYNC-WARN` for each note committed with a warning."""
    return [f"SYNC-REFUSED {x}" for x in rep.get("future", []) + rep.get("header", [])] + [f"SYNC-WARN {x}" for x in rep.get("warn", [])]


def auto_sync(root: Path, lanes: list[str]) -> tuple[str | None, int]:
    """After a notes write command: `sync` the lane directories it wrote, when `RESEARCH_NOTES_SYNC=1`.  (report lines, FUTURE_RC
    when a note was left out for its stamp or its front matter, else 0; other failures stay advisory, the next sync retries them);
    (None, 0) when off."""
    if os.environ.get(SYNC_ENV, "").strip() not in ("1", "true", "yes", "on"):
        return None, 0
    rep = sync(root, paths=[f"lanes/{x}" for x in lanes if x], message=f"notes: {', '.join(x for x in lanes if x)}")
    return "\n".join([render_sync(rep), *refused_lines(rep)]), FUTURE_RC if rep.get("future") or rep.get("header") else 0


def main(argv: list[str], default_root: Path) -> int:
    ap = argparse.ArgumentParser(prog="research notes", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sy = sub.add_parser("sync", help="commit your changes, rebase onto the remote, push (one sync of a clone at a time; bounded retries; a "
                                     "conflict drops nothing; "
                                     "exit 5: lane notes stamped over 5 min ahead of the clock, or with broken front matter, were left out)")
    sy.add_argument("--path", action="append", default=[], metavar="P", help="only changes under P (repeatable; default every change)")
    sy.add_argument("-m", "--message", default=None)
    sy.add_argument("--retries", type=int, default=5)
    sy.add_argument("--remote", default="origin")
    sy.add_argument("--root", type=Path, default=None)
    sc = sub.add_parser("scan", help="every blob in the notes history: files over 1 MB and key material (exit 1 if any)")
    sc.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    sc.add_argument("--json", action="store_true")
    sc.add_argument("--root", type=Path, default=None)
    ns = ap.parse_args(argv)
    root = ns.root or default_root
    if ns.cmd == "sync":
        rep = sync(root, ns.path or None, ns.message, max(0, ns.retries), ns.remote)
        print("\n".join([render_sync(rep), *refused_lines(rep)]), file=sys.stdout if rep["rc"] == 0 else sys.stderr)
        return rep["rc"]
    rep = scan(root, ns.max_bytes)
    if ns.json:
        print(json.dumps(rep, indent=1))
    else:
        for b in rep["big"]:
            print(f"BIG  {b['path']}  {b['bytes'] / 2**20:.1f} MB")
        for k in rep["keys"]:
            print(f"KEY  {k}")
        print(f"scanned {rep['blobs']} blobs: {len(rep['big'])} over {ns.max_bytes} bytes, {len(rep['keys'])} with key material")
    return 1 if rep["big"] or rep["keys"] else 0
