"""`research doctor`: whether this agent's VM can do its job, checked at session start, without printing a value.

  research doctor [--role lane|coordinator|infra] [--json]

Each check is `ok`, `skip` (it does not apply here) or a `finding`, which says what failed and how to fix it:

  secrets     every name the role needs (`SECRETS`) and every name this VM injects ($CLOUD_AGENT_INJECTED_SECRET_NAMES) is set;
              a name `research` took back from the tmux server (`secret_hint.recover`) counts, and is named
  github      `gh auth status` exits 0
  git-push    `git push --dry-run` of HEAD to a branch nobody has (`PROBE_REF`) on the checkout's origin: GitHub accepts the
              credential git pushes with
  notes-push  the same for the notes clone, through `notes_sync.remote_target` (RESEARCH_NOTES_TOKEN's helper when it is set)
  store       the evidence store's remote answers a HEAD of an absent key; a 403 is a refused credential
  slack       `auth.test` is ok, with the bot token or through the broker (coordinator and infra: a lane asks its coordinator)
  ssh:NAME    `ssh -o BatchMode=yes ... true`, as `research run --on` reaches NAME, for each machine whose entry names an ssh host
  venv        `uv sync --locked --check --offline` of the checkout's lock against the venv this runs in (else the checkout's
              .venv), workspace members aside: a worktree's come from PYTHONPATH
  disk        free space on the filesystems of the runs root, the store, the checkout and $TMPDIR, against the floor `research
              run` refuses under ($RESEARCH_DISK_FLOOR_GB)
  clock       this clock against the Date header of CLOCK_URL (Cloudflare's, as R2 is), within CLOCK_SKEW_MAX_S

The probes run at once, each bounded by DEADLINE_S, and what they print is discarded: a finding names an exit code, an HTTP
status or a class of error, and its fix says how to see the tool's own message. Exit 0 when nothing is a finding, 1 when
something is, 2 on a usage error; `--json` prints `research/doctor/v1`.
"""
from __future__ import annotations

import argparse
import email.utils
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from . import notes_sync, remote, runs, secret_hint, slack
from .pods import registry, runpod
from .store import local as store_local
from .store import remote as store_remote

SCHEMA = "research/doctor/v1"
ROLES = ("lane", "coordinator", "infra")
#: The Cursor secrets each role needs, by name; a tuple is alternatives, any one of which will do, and a role needs the names of
#: the roles before it too. GitHub and Slack need none: they go through Verity's brokers with the agent's Cursor identity
#: (.git/verity-auth, `research msg`). HF_TOKEN is optional (the Hub repos we fetch are public); Notion is an MCP server's.
SECRETS: dict[str, tuple[str | tuple[str, ...], ...]] = {
    "lane": ("R2_ENDPOINT", "R2_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",      # the store's remote (store/remote.py)
             "RUNPOD_API_KEY", "RUNPOD_SSH_KEY_B64",                                       # pods, and ssh to every machine (pods/runpod.py)
             "RESEARCH_NOTES_TOKEN"),                                                      # notes pushes (notes_sync.py)
    "coordinator": (),
    "infra": ("NEBIUS_SA_ID", "NEBIUS_SA_PUBLIC_KEY_ID", ("NEBIUS_SA_PRIVATE_KEY_B64", "NEBIUS_SA_PRIVATE_KEY")),   # pods/nebius/common.sh
}
SLACK_ROLES = ("coordinator", "infra")
DEADLINE_S = 25.0
PROBE_TIMEOUT_S = 15.0
SSH_CONNECT_S = 8
CLOCK_URL = "https://cloudflare.com/cdn-cgi/trace"      # uncached: github.com's CDN answers with Dates up to 10 s old
CLOCK_SKEW_MAX_S = 30.0         # R2 refuses a request 15 min off; notes refuse a stamp 5 min ahead; NTP keeps a VM within ms
PROBE_REF = "refs/heads/research-doctor/dry-run"
STORE_PROBE_KEY = "research-doctor/absent"
UV_SYNC = ("uv", "sync", "--all-packages", "--extra", "torch-cpu")     # AGENTS.md's
REPO = Path(__file__).resolve().parents[4]
DASHBOARD = "the Cursor Dashboard (Cloud Agents > Secrets)"
OK, FINDING, SKIP = "ok", "finding", "skip"

clock = time.time


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str | None = None


def run(argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> tuple[int | None, str]:
    """(exit code, output) of `argv`, stdin closed: None when it gave no answer within PROBE_TIMEOUT_S, 127 when it could not
    start. The output is for telling one failure from another, never for printing."""
    try:
        p = subprocess.run(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, errors="replace", timeout=PROBE_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        return None, ""
    except OSError:
        return 127, ""
    return p.returncode, p.stdout


def _alts(need: str | tuple[str, ...]) -> tuple[str, ...]:
    return (need,) if isinstance(need, str) else need


def needs(role: str) -> list[tuple[str, ...]]:
    return [_alts(n) for r in ROLES[:ROLES.index(role) + 1] for n in SECRETS[r]]


def check_secrets(role: str) -> Check:
    injected = [n for n in os.environ.get("CLOUD_AGENT_INJECTED_SECRET_NAMES", "").split(",") if n]
    want = needs(role)
    named = {n for alts in want for n in alts}
    want += [(n,) for n in injected if n not in named]
    missing = ["|".join(alts) for alts in want if not any(os.environ.get(n) for n in alts)]
    lost = [m for m in missing if any(n in injected for n in m.split("|"))]
    unset = [m for m in missing if m not in lost]
    if missing:
        fixes = []
        if lost:
            fixes.append(f"{', '.join(lost)}: this VM injects {'it' if len(lost) == 1 else 'them'}, but neither this shell nor the tmux "
                         f"server has {'it' if len(lost) == 1 else 'them'}: {secret_hint.FRESH_SHELL} (`tmux new-session`, then "
                         "`research doctor` there); if that shell lacks them too, the VM lost its secrets (an exec-daemon restart) "
                         "and a new agent VM gets them at boot")
        if unset:
            fixes.append(f"{', '.join(unset)}: add {'it' if len(unset) == 1 else 'them'} in {DASHBOARD}; a new agent VM gets them")
        return Check("secrets", FINDING, f"not set here: {', '.join(missing)}", "; ".join(fixes))
    took = [n for n in secret_hint.RECOVERED if os.environ.get(n)]
    detail = f"the {len(needs(role))} names the {role} role needs" + (f" and the {len(injected)} this VM injects" if injected else "") + " are set"
    if took:
        shown = ", ".join(took[:4]) + (f" and {len(took) - 4} more" if len(took) > 4 else "")
        detail += (f"; {len(took)} taken back from the tmux server because this shell had lost them ({shown}): run tools other than "
                   "`research` from a tmux login shell")
    return Check("secrets", OK, detail)


def check_github() -> Check:
    rc, _ = run(["gh", "auth", "status", "--hostname", "github.com"], env={**os.environ, "GH_PROMPT_DISABLED": "1"})
    if rc == 0:
        return Check("github", OK, "`gh auth status` accepts its token")
    if rc == 127:
        return Check("github", FINDING, "gh is not on PATH", "install the GitHub CLI (https://cli.github.com)")
    if rc is None:
        return Check("github", FINDING, f"`gh auth status` gave no answer within {PROBE_TIMEOUT_S:g} s", "retry; `gh auth status` shows where it hangs")
    return Check("github", FINDING, f"`gh auth status` exits {rc}: gh's token is missing, expired or revoked",
                 "`gh auth status` says which (it masks the token); on a Cursor VM gh's token comes from Verity's GitHub broker "
                 "(.git/verity-auth) or Cursor's App token, which a new agent VM renews")


#: (what git says, what it means, whether the credential is the fix), the first that matches
PUSH_FAILURES = (
    (re.compile(r"\b403\b|Permission to \S+ denied|denied to ", re.I), "GitHub refused the credential write access", True),
    (re.compile(r"Authentication failed|could not read (Username|Password)|terminal prompts disabled|Invalid username or password|\b401\b",
                re.I), "no usable credential (missing, expired or revoked)", True),
    (re.compile(r"Could not resolve host|Failed to connect|Connection (timed out|refused|reset)|Operation timed out|Network is unreachable",
                re.I), "the remote is unreachable", False),
    (re.compile(r"Repository not found|does not appear to be a git repository|\b404\b", re.I),
     "no such repository, or none this credential can see", True),
)


def push_probe(name: str, root: Path, opts: list[str], where: str, credential_fix: str) -> Check:
    """`git push --dry-run` of HEAD to PROBE_REF: GitHub asks for the push credential before it answers, and nothing is sent."""
    rc, out = run(["git", "-C", str(root), *opts, "push", "--dry-run", "--quiet", where, f"HEAD:{PROBE_REF}"],
                  env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if rc == 0:
        return Check(name, OK, f"origin of {root} accepts a push (dry run)")
    see = f"`git -C {root} push --dry-run origin HEAD:{PROBE_REF}` shows git's message"
    if rc is None:
        return Check(name, FINDING, f"`git push --dry-run` from {root} gave no answer within {PROBE_TIMEOUT_S:g} s", f"retry; {see}")
    what, credential = next(((w, c) for pat, w, c in PUSH_FAILURES if pat.search(out)), (f"git exits {rc}", False))
    return Check(name, FINDING, f"origin of {root} refuses a push (dry run): {what}", f"{credential_fix}; {see}" if credential else see)


def _git_root(where: Path) -> Path | None:
    try:
        p = subprocess.run(["git", "-C", str(where), "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return Path(p.stdout.strip()) if p.returncode == 0 and p.stdout.strip() else None


def checkout() -> Path | None:
    """The git checkout the caller works in (the working directory's), else the one this package runs from."""
    return _git_root(Path.cwd()) or _git_root(REPO)


def _has_origin(root: Path) -> bool:
    return subprocess.run(["git", "-C", str(root), "config", "--get", "remote.origin.url"], capture_output=True, check=False).returncode == 0


def check_git_push(root: Path | None) -> Check:
    if root is None:
        return Check("git-push", SKIP, "not in a git checkout")
    if not _has_origin(root):
        return Check("git-push", SKIP, f"{root} has no origin")
    return push_probe("git-push", root, [], "origin",
                      "the credential git pushes to github.com with is missing or expired: on a Cursor VM it comes from Verity's GitHub "
                      "broker (.git/verity-auth) or Cursor's App token in ~/.gitconfig, which a new agent VM renews")


def check_notes_push() -> Check:
    root = registry.notes_root()
    if _git_root(root) != root.resolve():
        return Check("notes-push", FINDING, f"{root} is not a git clone: notes written here reach nobody",
                     "with RESEARCH_NOTES_TOKEN set, any `research notes` command clones it (kb/cloud-lane-setup.md, section 1)")
    if not _has_origin(root):
        return Check("notes-push", FINDING, f"the notes clone {root} has no origin", f"`git -C {root} remote add origin {notes_sync.NOTES_URL}`")
    opts, where = notes_sync.remote_target(root)
    fix = (f"{notes_sync.TOKEN_ENV} was refused (expired or rotated): a new one goes in {DASHBOARD}" if os.environ.get(notes_sync.TOKEN_ENV)
           else f"set {notes_sync.TOKEN_ENV} (a Cursor secret): without it git pushes as the VM's GitHub App, which the notes repo "
                "refuses (kb/cloud-lane-setup.md, section 1)")
    return push_probe("notes-push", root, opts, where, fix)


def check_store() -> Check:
    try:
        r = store_remote.load_remote()
    except store_remote.ConfigError as e:
        return Check("store", FINDING, f"the store's remote is misconfigured: {e}", "fix what it names (tools/research/README.md, Credentials)")
    if r is None:
        return Check("store", FINDING, "no store remote: a run here is preserved nowhere, and `research run` refuses without --local-only",
                     f"set R2_ENDPOINT, R2_BUCKET, AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in {DASHBOARD}, or write ~/.research/store.toml")
    kind = type(r).__name__
    try:
        r.head(STORE_PROBE_KEY)
    except urllib.error.HTTPError as e:
        return Check("store", FINDING, f"the store's remote ({kind}) refused a HEAD: HTTP {e.code}",
                     "the key pair (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) was revoked or rotated, or a temporary credential "
                     f"(AWS_SESSION_TOKEN) expired: a new pair goes in {DASHBOARD}; `research data mint-credential` mints a temporary one")
    except Exception as e:  # noqa: BLE001  (a transport's errors are many; only the class is shown)
        return Check("store", FINDING, f"the store's remote ({kind}) gave no answer: {type(e).__name__}",
                     "retry; `research data preserved RUN` on any run shows the remote's own error")
    return Check("store", OK, f"the store's remote ({kind}) answers")


def check_slack() -> Check:
    try:
        api = slack.Slack(os.environ.get("SLACK_BOT_TOKEN"), transport=lambda url, data, headers: slack.post(url, data, headers, timeout=PROBE_TIMEOUT_S))
        api.call("auth.test")
    except slack.SlackError as e:
        return Check("slack", FINDING, str(e), "the using-slack skill: a Cursor agent goes through the broker with no token; @infra holds the bot's")
    return Check("slack", OK, f"auth.test is ok ({api.via})")


#: (what ssh says, what it means), the first that matches
SSH_FAILURES = (
    (re.compile(r"Permission denied|Too many authentication failures", re.I), "refused the key"),
    (re.compile(r"Host key verification failed|IDENTIFICATION HAS CHANGED", re.I), "its host key changed"),
    (re.compile(r"Could not resolve|timed out|Connection refused|No route to host|Network is unreachable|closed by remote|reset by peer",
                re.I), "unreachable"),
)


def ssh_machines() -> dict[str, dict]:
    """The machines `research run --on` knows (machines.toml and the registry) whose entries name an ssh host."""
    return {n: m for n, m in sorted(remote.load_machines().items()) if m.get("provider") == "ssh"}


def check_ssh(name: str, machine: dict) -> Check:
    check = f"ssh:{name}"
    see = f"`research pods ssh {name} -- true` shows ssh's message"
    try:
        t = remote.ssh_host_target({**machine, "name": name})
        argv = remote.ssh_argv(t["host"], t["port"], user=t["user"], connect_timeout=SSH_CONNECT_S)
    except (remote.Refused, runpod.PodError) as e:
        return Check(check, FINDING, str(e), "fix what it names")
    who = f"{t['user']}@{t['host']}:{t['port']}"
    rc, out = run([*argv, "true"])
    if rc == 0:
        return Check(check, OK, f"{who} answers")
    if rc is None:
        return Check(check, FINDING, f"{who} gave no answer within {PROBE_TIMEOUT_S:g} s", f"retry; {see}")
    what = next((w for pat, w in SSH_FAILURES if pat.search(out)), f"ssh exits {rc}")
    fix = (f"the key `research` uses (RUNPOD_SSH_KEY_B64, else {runpod.SSH_KEY}) is not authorized there: ask @infra to add its public "
           f"key; {see}") if what == "refused the key" else f"{see}; ask @infra if it persists"
    return Check(check, FINDING, f"{who}: {what}", fix)


def active_venv() -> Path | None:
    return Path(sys.prefix) if sys.prefix != sys.base_prefix else None


def check_venv(root: Path | None, venv: Path | None = None) -> Check:
    if root is None or not (root / "uv.lock").is_file():
        return Check("venv", SKIP, "no uv.lock in a checkout here")
    venv = venv or active_venv() or root / ".venv"
    own = venv.resolve() == (root / ".venv").resolve()
    sync = f"`cd {root} && {' '.join(UV_SYNC)}`"
    if not (venv / "pyvenv.cfg").is_file():
        return Check("venv", FINDING, f"no venv at {venv}", sync)
    rc, out = run([*UV_SYNC, "--locked", "--check", "--offline", "--no-install-workspace", "--inexact"], cwd=root,
                  env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(venv)})
    if rc == 0:
        return Check("venv", OK, f"{venv} has what {root / 'uv.lock'} pins (workspace members aside)")
    if rc == 127:
        return Check("venv", FINDING, "uv is not on PATH", "install uv (https://docs.astral.sh/uv/)")
    if rc is None:
        return Check("venv", FINDING, f"`uv sync --check` gave no answer within {PROBE_TIMEOUT_S:g} s", f"retry in {root}")
    if "needs to be updated" in out:
        return Check("venv", FINDING, f"{root / 'uv.lock'} is out of date with the checkout's pyproject.toml files",
                     f"`uv lock` in {root}, and commit it")
    pins = re.findall(r"^ \+ (\S+)", out, re.M)
    if not pins:
        return Check("venv", FINDING, f"`uv sync --check` exits {rc}",
                     f"`cd {root} && {' '.join(UV_SYNC)} --locked --check` shows uv's message")
    shown = ", ".join(pins[:6]) + (f" and {len(pins) - 6} more" if len(pins) > 6 else "")
    fix = sync if own else (f"{venv} is another checkout's: sync it there once that checkout has this lock, or give this checkout its "
                            f"own ({sync}, then {root / '.venv/bin/python'}); a sync from here would point its editables at {root}")
    return Check("venv", FINDING, f"{venv} lacks what {root / 'uv.lock'} pins: {shown}", fix)


def _existing(p: Path) -> Path:
    p = p.expanduser().absolute()
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def check_disk(root: Path | None) -> Check:
    seen: dict[int, Path] = {}
    for p in (runs.runs_root(), store_local.store_root(), *([root] if root else []), Path(tempfile.gettempdir())):
        seen.setdefault(os.stat(_existing(p)).st_dev, p)
    refusals = [why for p in seen.values() if (why := runs.disk_floor_refusal(p))]
    if refusals:
        found = [r.partition(" Free space first: ") for r in refusals]
        return Check("disk", FINDING, "; ".join(head for head, _, _ in found), "free space first: " + found[0][2])
    least = min(runs.free_bytes(_existing(p)) for p in seen.values())
    return Check("disk", OK, f"{least / 1e9:.1f} GB free at the least on {len(seen)} filesystem{'s' if len(seen) != 1 else ''} "
                             f"(runs, store, checkout, $TMPDIR), over the floor (${runs.DISK_FLOOR_ENV})")


def server_date(url: str = CLOCK_URL) -> float:
    """The time `url`'s Date header gives a HEAD, in seconds since the epoch; its one-second resolution is the caller's."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "research-doctor"})
    try:
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_S) as r:
            date = r.headers.get("Date")
    except urllib.error.HTTPError as e:
        date = e.headers.get("Date")
    if not date:
        raise ValueError("no Date header")
    return email.utils.parsedate_to_datetime(date).timestamp()


def check_clock() -> Check:
    host = urllib.parse.urlsplit(CLOCK_URL).netloc
    before = clock()
    try:
        server = server_date() + 0.5            # a Date header truncates to the second
    except (OSError, ValueError, TypeError) as e:
        return Check("clock", FINDING, f"no Date from {host} to compare with: {type(e).__name__}",
                     f"check the network: `curl -sI {CLOCK_URL}` shows whether {host} answers")
    skew = (before + clock()) / 2 - server
    if abs(skew) > CLOCK_SKEW_MAX_S:
        return Check("clock", FINDING, f"this clock is {skew:+.0f} s off {host}'s",
                     "`timedatectl` shows whether it syncs; R2 refuses a request 15 min off and notes refuse a stamp 5 min ahead")
    return Check("clock", OK, f"within {CLOCK_SKEW_MAX_S:g} s of {host}'s ({skew:+.1f} s)")


Probe = tuple[str, Callable[[], Check]]


def checks(role: str, root: Path | None) -> list[Probe]:
    out: list[Probe] = [("secrets", lambda: check_secrets(role)), ("github", check_github),
                        ("git-push", lambda: check_git_push(root)), ("notes-push", check_notes_push), ("store", check_store)]
    if role in SLACK_ROLES:
        out.append(("slack", check_slack))
    try:
        machines = ssh_machines()
    except (OSError, ValueError) as e:          # tomllib.TOMLDecodeError is a ValueError
        out.append(("ssh", lambda e=e: Check("ssh", FINDING, f"the machines config is unreadable: {type(e).__name__}",
                                             "`research pods registry` names the file")))
    else:
        out += [(f"ssh:{n}", lambda n=n, m=m: check_ssh(n, m)) for n, m in machines.items()] or \
               [("ssh", lambda: Check("ssh", SKIP, "no machine in machines.toml or the registry names an ssh host"))]
    out += [("venv", lambda: check_venv(root)), ("disk", lambda: check_disk(root)), ("clock", check_clock)]
    return out


def gather(probes: list[Probe], deadline_s: float = DEADLINE_S) -> list[Check]:
    """Every probe at once, in daemon threads: one still running at the deadline is a finding, and is left behind."""
    done: dict[str, Check] = {}

    def one(name: str, probe: Callable[[], Check]) -> None:
        try:
            done[name] = probe()
        except Exception as e:  # noqa: BLE001  (a probe's own failure is a finding; its message might quote what it read)
            done[name] = Check(name, FINDING, f"the probe itself failed: {type(e).__name__}", "tell @infra, with `research doctor --json`")

    threads = [threading.Thread(target=one, args=p, daemon=True) for p in probes]
    for t in threads:
        t.start()
    end = time.monotonic() + deadline_s
    for t in threads:
        t.join(max(0.0, end - time.monotonic()))
    got = dict(done)
    return [got.get(name) or Check(name, FINDING, f"no answer within {deadline_s:g} s", "retry; a probe this slow is a finding of its own")
            for name, _ in probes]


def render(role: str, results: list[Check]) -> str:
    n = sum(c.status == FINDING for c in results)
    w = max(len(c.name) for c in results)
    lines = [f"research doctor ({role}): " + (f"{n} finding{'s' if n != 1 else ''}" if n else "ok")]
    for c in results:
        lines.append(f"{'FINDING' if c.status == FINDING else c.status:<8} {c.name:<{w}}  {c.detail}")
        if c.fix:
            lines.append(f"{'':<8} {'':<{w}}  fix: {c.fix}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research doctor", description=__doc__.split("\n\n")[0])
    ap.add_argument("--role", choices=ROLES, default="lane", help="what this VM is for: a coordinator adds Slack, infra adds Nebius")
    ap.add_argument("--json", action="store_true", help=f"print {SCHEMA}")
    ns = ap.parse_args(argv)
    results = gather(checks(ns.role, checkout()))
    n = sum(c.status == FINDING for c in results)
    if ns.json:
        print(json.dumps({"schema": SCHEMA, "role": ns.role, "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(clock())),
                          "ok": not n, "findings": n, "checks": [asdict(c) for c in results]}, indent=1))
    else:
        print(render(ns.role, results))
    return 1 if n else 0
