"""`research queue`: the merge queue.  The verified queue's loop (`research queue vqueue`, `research.vqueue_loop`) merges the
PRs ready to land onto `main` as candidates, and moves `main` only to a candidate with no new failure (`research.vqueue`).
"""
from __future__ import annotations

import argparse
import ast
import calendar
import concurrent.futures
import contextlib
import dataclasses
import difflib
import fcntl
import fnmatch
import functools
import hashlib
import importlib.util
import json
import math
import os
import posixpath
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
import weakref
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Collection, Iterable

from . import linear, quick_tier, suite_layout, waitcap
from .telemetry import end_state
from .gates import WORKTREES_LOCK, git_env, worktrees_shared
from .store import ids, vocab

BRANCH, SHADOW_BRANCH = "next", "queue-shadow"
RULES = "ci/queue.toml"
RULES_BEFORE_MOVE = "tools/check/queue.toml"
DRIVE_FLAGS = "tools/check/drive.toml"
PINS_FILES = ("ci/pins.py", "tools/check/pins.py")
MODULES, HEADERS = "tools/verity/lean/modules.py", "tools/verity/lean/headers.py"
PORTED_ROOTS = ("security/Definitions.lean", "verity/protocols/proofs/flock/verifier/Flock.lean")
PORT_INPUTS = ("*.lean", "*lakefile.toml", "*lakefile.lean", "*lake-manifest.json", "*lean-toolchain")
MARK_TIME_RE = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")
AUTO_SHARDS_RE = re.compile(r"auto(:[1-9]\d*)?")
IDENTITY = {"GIT_AUTHOR_NAME": "research queue", "GIT_AUTHOR_EMAIL": "queue@research.invalid",
            "GIT_COMMITTER_NAME": "research queue", "GIT_COMMITTER_EMAIL": "queue@research.invalid"}
DATES = ("GIT_AUTHOR_DATE", "GIT_COMMITTER_DATE")
STALL = ("-c", "http.lowSpeedLimit=1000", "-c", "http.lowSpeedTime=120")
GH_TIMEOUT_S = 120
GH_PAGES = 10
GH_PER_PAGE = 100
LANE_RE = re.compile(r"^[ \t>*_]*Lane[:\s]+`?([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)`?", re.M)
REGISTRY = Path(".agents/skills/using-slack/registry.json")
AGENT_RE = re.compile(r"\bbc-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
TRAILER_RE = re.compile(r"^(PR|Head|Owner): (.+)$", re.M)
MOVE_RE = re.compile(r"^[ \t>*_]*Move[:\s]+`?(tools/move/[\w.-]+\.toml)`?[ \t.]*$", re.M)
ISSUE_KEY_RE = re.compile(r"[A-Z][A-Z0-9]*-\d+")
ISSUE_LINE = "Refs {key}"
NO_ISSUE_LINE = "Issue: none: {reason}"
NO_ISSUE_RE = re.compile(r"^[ \t>*_]*Issue:[ \t]*none:[ \t]*(\S.*?)[ \t]*$", re.M)
GOAL_LINE = "Goal {id}"
GOAL_RE = re.compile(r"^[ \t>*_]*Goal:?[ \t]+`?([A-Za-z0-9](?:[\w./-]*[A-Za-z0-9])?)`?[ \t.]*$", re.M)
GOALS_CACHE = Path("~/.research/cache/goals.json")
ISSUE_LINK_RE = re.compile(r"(?i)\b(?:refs?|references|part of|contributes to|towards?|relates to|related to|close[sd]?|closing|"
                           r"fix(?:e[sd])?|fixing|resolve[sd]?|resolving|complete[sd]?|completing|implement(?:s|ed|ing)?)"
                           r"[ \t:]+(?-i:([A-Z][A-Z0-9]*-\d+))\b")
MOVE_TRAILER_RE = re.compile(r"^Move: (tools/move/[\w.-]+\.toml)$", re.M)
RENAME, MOVE_CHECK, RESTACK = "tools/move/rename.py", "tools/check/move_check.py", "tools/move/restack.py"
MOVE_DIR, MOVE_PINS = "tools/move/", "tools/check/pins.py"
MOVE_LINES = "tools/move/move_lines.toml"
RESTACK_WAIT_S = 2 * 3600
RESTACK_LOST_S = RESTACK_WAIT_S + 300
RESTACK_OUT = "restack.json"
RESTACK_JOB = f'd=$1; shift; exec timeout -s INT -k 60 {RESTACK_WAIT_S} "$@" > "$d/{RESTACK_OUT}"'
RESTACK_INTERRUPTED = ("130", "143")
RESTACKING = "restacking across the move"
LANDED_WINDOW = 2000
SLUG_RE = re.compile(r"github\.com[:/]+([^/]+/[^/]+?)(?:\.git)?/?$")
REPO_ARG_RE = re.compile(r"[\w.-]+/[\w.-]+")
CLONE_MARK = "research.queueclone"
WORKDIR = Path.home() / ".research" / "queue" / "verity"
AUTH_FAILED_RE = re.compile(r"could not read (Username|Password)|Authentication failed|unable to get password")
REPO = "compute-verification/verity"
MEMBER_ASSOCIATIONS = ("OWNER", "MEMBER")
AUTHORS = ("verity-agents[bot]", "verity-drive[bot]", "danielreuter")


class QueueError(Exception):
    pass


class Restacking(Exception):
    """A head a background job is carrying across a move: its PR waits off `next` this sync."""


def _git(repo: Path, *args: str, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    full = git_env(GIT_TERMINAL_PROMPT="0", **(env or {}))
    if env is not None and env.keys() == IDENTITY.keys():
        for k in DATES:
            full.pop(k, None)
    with worktrees_shared(repo, full) if _subcommand(args) == "fetch" else contextlib.nullcontext():
        r = subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, capture_output=True, text=True,
                           errors="surrogateescape", env=full)
    if check and r.returncode:
        raise QueueError(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()[-400:]}")
    return r


def _subcommand(args: tuple[str, ...]) -> str:
    i = 0
    while args[i:i + 1] == ("-c",):
        i += 2
    return args[i] if i < len(args) else ""


def _out(repo: Path, *args: str) -> str:
    return _git(repo, *args).stdout.strip()


def credential_args(repo: Path, remote: str) -> tuple[str, ...]:
    if not re.match(r"https?://", remote):
        return ()
    found = _git(repo, "config", "--local", "--get-urlmatch", "credential", remote, check=False).stdout.splitlines()
    conf = dict(line.split(" ", 1) for line in found if " " in line)
    if not conf.get("credential.helper"):
        if _git(repo, "config", "--get-urlmatch", "credential.helper", remote, check=False).stdout.strip() or not (gh := shutil.which("gh")):
            return ()
        conf = {"credential.helper": f"!{shlex.quote(gh)} auth git-credential"}
    return ("-c", "credential.helper=", *(o for k, v in conf.items() for o in ("-c", f"{k}={v}")))


def redacted(url: str) -> str:
    return re.sub(r"(?<=://)[^/@]*@", "", url)


def slug(url: str) -> str | None:
    return (m := SLUG_RE.search(url)) and m.group(1)


def same_repository(a: str, b: str) -> bool:
    sa, sb = slug(a), slug(b)
    return (sa or "").lower() == (sb or "").lower() if sa or sb else a.rstrip("/") == b.rstrip("/")


def remote_and_repo(remote: str | None, repo: str | None, cwd: Path) -> tuple[str, str]:
    if repo is not None and not REPO_ARG_RE.fullmatch(repo):
        raise QueueError(f"--repo {repo!r}: OWNER/NAME on GitHub, as {REPO}")
    if repo and remote is None:
        return f"https://github.com/{repo}.git", repo
    remote = remote or _out(cwd, "remote", "get-url", "origin")
    named = slug(remote)
    if repo and named and named.lower() != repo.lower():
        raise QueueError(f"--remote {redacted(remote)} is {named} on GitHub, not --repo {repo}: pass --repo alone (its remote is "
                         f"https://github.com/{repo}.git), or a --remote of {repo}")
    if not (repo := repo or named):
        raise QueueError(f"can't tell OWNER/NAME from {redacted(remote)!r}: pass --repo")
    return remote, repo


@dataclass(frozen=True)
class PR:
    number: int
    head: str
    title: str = ""
    body: str = ""
    draft: bool = False
    labels: tuple[str, ...] = ()
    base: str = "main"
    branch: str = ""
    fork: bool = False
    # : None when the head's repository was deleted
    head_repo: str | None = REPO
    author: str = ""
    association: str = "MEMBER"

    @classmethod
    def from_api(cls, d: dict) -> "PR":
        repo = lambda side: ((d.get(side) or {}).get("repo") or {}).get("full_name")  # noqa: E731
        return cls(number=int(d["number"]), head=d["head"]["sha"], title=d.get("title") or "", body=d.get("body") or "",
                   draft=bool(d.get("draft")), labels=tuple(sorted(x["name"] for x in d.get("labels") or [])),
                   base=(d.get("base") or {}).get("ref", "main"), branch=d["head"].get("ref") or "",
                   fork=repo("head") != repo("base"), head_repo=repo("head"), author=(d.get("user") or {}).get("login") or "",
                   association=d.get("author_association") or "")

    def outside(self, repo: str) -> str | None:
        fix = "a maintainer opens it from a branch of this repository"
        if self.head_repo != repo:
            return f"its head is on {self.head_repo or 'a deleted repository'}, not {repo}: {fix}"
        if self.association not in MEMBER_ASSOCIATIONS and self.author not in AUTHORS:
            return (f"its author {self.author or '(none)'} is no owner or member of the organization "
                    f"({self.association or 'no association'}) and none of {', '.join(AUTHORS)}: {fix}")
        return None

    @property
    def owner(self) -> str | None:
        if m := LANE_RE.search(self.body):
            return m.group(1)
        if m := AGENT_RE.search(self.body):
            return m.group(0)
        return None

    @property
    def move(self) -> str | None:
        m = MOVE_RE.search(self.body)
        return m.group(1) if m else None


class GitHub:
    def __init__(self, repo: str):
        self.repo = repo

    def open_prs(self) -> list[PR]:
        out: dict[int, PR] = {}
        with concurrent.futures.ThreadPoolExecutor(GH_PAGES) as ex:
            first = 1
            while True:
                pages = list(ex.map(self._page, range(first, first + GH_PAGES)))
                for page in pages:
                    for d in page:
                        out.setdefault(int(d["number"]), PR.from_api(d))
                    if len(page) < GH_PER_PAGE:
                        return list(out.values())
                first += GH_PAGES

    def pr(self, number: int) -> PR:
        return PR.from_api(self._api(f"pulls/{number}"))

    def _page(self, n: int) -> list[dict]:
        page = self._api(f"pulls?state=open&per_page={GH_PER_PAGE}&page={n}")
        return page if isinstance(page, list) else [page]

    def _api(self, path: str):
        what = f"gh api repos/{self.repo}/{path.partition('?')[0]}"
        try:
            r = subprocess.run(["gh", "api", f"repos/{self.repo}/{path}"], capture_output=True, text=True, timeout=GH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise QueueError(f"{what}: no answer in {GH_TIMEOUT_S} s") from None
        if r.returncode:
            raise QueueError(f"{what}: {(r.stderr or r.stdout).strip()[-300:]}")
        return json.loads(r.stdout)

    def branch_tip(self, branch: str) -> str | None:
        try:
            r = subprocess.run(["gh", "api", f"repos/{self.repo}/git/ref/heads/{branch}", "--jq", ".object.sha"], capture_output=True,
                               text=True, timeout=GH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return None
        return r.stdout.strip() or None if r.returncode == 0 else None

    def close(self, number: int, comment: str) -> None:
        try:
            r = subprocess.run(["gh", "pr", "close", str(number), "--repo", self.repo, "--comment", comment], capture_output=True,
                               text=True, timeout=GH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise QueueError(f"gh pr close {number}: no answer in {GH_TIMEOUT_S} s") from None
        if r.returncode:
            raise QueueError(f"gh pr close {number}: {(r.stderr or r.stdout).strip()[-300:]}")


class PublicGitHub(GitHub):
    def _api(self, path: str):
        import urllib.request
        url = f"https://api.github.com/repos/{self.repo}/{path}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"}),
                                        timeout=GH_TIMEOUT_S) as r:
                return json.load(r)
        except (OSError, ValueError) as e:
            raise QueueError(f"{url}: {e}") from None


def landing_waiter(repo: str | None = None, store_root: str | None = None) -> Callable[[list[int]], bool]:
    from .store.local import LocalStore

    state: dict = {}

    def waiting(numbers: list[int]) -> bool:
        if "heads" not in state:
            try:
                slug = repo or ((m := SLUG_RE.search(_out(Path.cwd(), "remote", "get-url", "origin"))) and m.group(1))
                hold = Rules().hold_label
                state["heads"] = {pr.number: pr.head for pr in GitHub(slug).open_prs() if hold not in pr.labels}
                state["labels"] = StoreLabels(LocalStore(store_root), log=lambda s: None)
                state["labels"].prefetch(state["heads"].items())
            except (QueueError, OSError) as e:
                print(f"linear audit: can't read the open PRs ({e}), so no issue counts as waiting on landing",
                      file=sys.stderr)
                state["heads"], state["labels"] = {}, None
        heads, labels = state["heads"], state["labels"]
        return bool(numbers) and all(n in heads and labels.ready(n, heads[n]) for n in numbers)

    return waiting


@dataclass(frozen=True)
class Rules:
    hold_label: str = "hold"
    source: str = RULES
    marked_after: str | None = None
    author: tuple[str, str] | None = None

    @classmethod
    def parse(cls, text: str, source: str = RULES) -> "Rules":
        doc = tomllib.loads(text)
        unknown = set(doc) - {"hold_label", "marked_after", "author"}
        if unknown:
            raise QueueError(f"{source}: unknown keys {sorted(unknown)}")
        cut = doc.get("marked_after")
        if cut is not None and not (isinstance(cut, str) and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", cut)):
            raise QueueError(f"{source}: marked_after {cut!r}: a UTC time as YYYY-MM-DDTHH:MM:SSZ, as labels record it")
        author = doc.get("author")
        if author is not None:
            if why := _author_why(author):
                raise QueueError(f'{source}: author {why}: write `author = {{ name = "NAME", email = "ADDRESS" }}`, the two '
                                 "the queue's commits carry as author and committer, or leave `author` out and they carry "
                                 f"{IDENTITY['GIT_AUTHOR_NAME']} <{IDENTITY['GIT_AUTHOR_EMAIL']}>")
            author = (author["name"], author["email"])
        return cls(doc.get("hold_label", "hold"), source, cut, author)

    @property
    def identity(self) -> dict[str, str]:
        if self.author is None:
            return IDENTITY
        name, email = self.author
        return {"GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email, "GIT_COMMITTER_NAME": name, "GIT_COMMITTER_EMAIL": email}


def _author_why(author: object) -> str | None:
    if not isinstance(author, dict):
        return f"is {author!r}, not a table"
    if unknown := sorted(set(author) - {"name", "email"}):
        return f"has unknown keys {unknown}"
    for key in ("name", "email"):
        if key not in author:
            return f"names no {key}"
        value = author[key]
        if not isinstance(value, str):
            return f"names {key} {value!r}, not a string"
        if not value.strip():
            return f"names an empty {key}"
        if bad := sorted(set(value) & set("\0\n<>")):
            return f"names {key} {value!r}, which holds {' and '.join(repr(c) for c in bad)}"
    return None


LANDINGS = ("review",)
GATE_PATH_RE = re.compile(r"[\w.-]+(?:/[\w.-]+)*\.py")
CHECK_STEPS = ("preflight-lock", "preflight-lints", "preflight-fixtures", "flock-circuit-build", "lean-build",
               "lean-unit-cut", "lean-agreement", "lean-audit", "lean-suites", "pytest", "circuit-check", "flock-rust-tests",
               "ratchet-gate")
LOWRISK_EXCLUDE_RE = re.compile(r"(?:[\w.-]+/)+(?:[\w.-]+)?|[\w.*?\[\]-]+")
HANDLE_RE = re.compile(r"@[a-z0-9][a-z0-9._-]*")
MODES = ("off", "shadow", "on")
CLOSE_UNLANDABLE = ("off", "log", "on")
COMMAND_LINE = ("on", "marked_after", "regenerate_locks", "after_landing_no_cache")
# : Keys no code reads, which code deployed ahead of `main` may still read, so even a strict parse ignores them.
DELETED_KEYS = ("max_lean_checks", "nonlean_first", "reviewed_first")
FIRST_RETIRED = ("--first is retired: the verified queue's priority file (`research queue vqueue --priority-file`) holds the "
                 "landing order; list the PRs there, first to land first")
SIGNED_GUARDS = (str(REGISTRY), "tools/research/src/research/labelauth.py", "tools/research/src/research/queue.py",
                 "tools/research/src/research/landing.py")
STAFFS_RE = re.compile(r"[a-z0-9*?][a-z0-9._*?-]*")


def signed_mode(value: object) -> str | None:
    if isinstance(value, bool):
        return "on" if value else "off"
    return "shadow" if value == "shadow" else None


@dataclass(frozen=True)
class DriveFlags:
    on: str | None = None
    marked_after: str | None = None
    regenerate_locks: bool = False
    land: str = "review"
    land_into: str | None = None
    after_landing_no_cache: bool = False
    landing_gates: tuple[str, ...] = ()
    warn_gates: tuple[str, ...] = ()
    warn_steps: tuple[str, ...] = ()
    verdict_first: bool = True
    stale_days: int = 0
    stale_close_days: int = 0
    stale_keep: tuple[int, ...] = ()
    close_unlandable: str = "off"
    after_landing_shards: int | str = 1
    command_line: tuple[str, ...] = ("on",)
    lowrisk_lane: bool = True
    lowrisk_per_tick: int = 20
    lowrisk_excludes: tuple[str, ...] = ()
    verdict_excludes: tuple[str, ...] = ()
    verdict_plain: tuple[str, ...] = ()
    executor: str = "off"
    executor_enqueue: tuple[str, ...] = ()
    verdict_carry: str = "off"
    bottom_ring: tuple[str, ...] = ("@ci", "@top")
    bottom_wait_min: int = 30
    signed_verdicts: str = "off"
    restack_jobs: int = 3

    @property
    def plain_excludes(self) -> tuple[str, ...]:
        if not (self.lowrisk_excludes and self.verdict_excludes):
            return ()
        return tuple(dict.fromkeys([x for x in self.lowrisk_excludes if x not in self.verdict_plain] + list(self.verdict_excludes)))

    @classmethod
    def parse(cls, text: str, source: str = "drive.toml", strict: bool = False) -> "DriveFlags":
        try:
            doc = tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            raise QueueError(f"{source}: not TOML: {e}") from None
        keys = [f.name for f in dataclasses.fields(cls)]
        for k in list(doc):
            if k not in keys:
                near = difflib.get_close_matches(k, keys, n=1)
                if strict and k not in DELETED_KEYS:
                    raise QueueError(f"{source}: unknown key {k!r}{f' (did you mean {near[0]!r}?)' if near else ''}: this "
                                     f"research reads only {', '.join(keys)}; rename or remove it, or update this research")
                print(f"{source}: ignoring key {k!r}, which this research doesn't read "
                      + ("(its code was deleted: remove it)" if k in DELETED_KEYS else "(a newer one may)"), file=sys.stderr)
                del doc[k]

        def bad(k: str, want: str) -> QueueError:
            return QueueError(f"{source}: {k} = {doc[k]!r}: {want}")

        def names(k: str) -> tuple[str, ...]:
            v = doc[k]
            if not (isinstance(v, list) and all(isinstance(x, str) and re.fullmatch(r"[\w.-]+", x) for x in v)):
                raise bad(k, "a list of names, as [\"a\", \"b\"]")
            return tuple(v)

        got: dict = {}
        if "on" in doc:
            if not (on := names("on")):
                raise bad("on", "at least one machine (train.sh's: a machine, or `auto`)")
            got["on"] = ",".join(on)
        if "after_landing_shards" in doc:
            v = doc["after_landing_shards"]
            if not (isinstance(v, str) and AUTO_SHARDS_RE.fullmatch(v)) and (type(v) is not int or v < 1):
                raise bad("after_landing_shards", "a whole number, 1 or a shard plan's N in tools/check/ci.toml, or "
                                                  "\"auto\" or \"auto:MAX\" (the shard planner's N, at most MAX)")
            got["after_landing_shards"] = v
        for k, modes in (("executor", MODES), ("verdict_carry", MODES), ("close_unlandable", CLOSE_UNLANDABLE)):
            if k in doc:
                if doc[k] not in modes:
                    raise bad(k, f"one of {', '.join(modes)}")
                got[k] = doc[k]
        if "executor_enqueue" in doc:
            v = doc["executor_enqueue"]
            if not (isinstance(v, list) and v and all(isinstance(x, str) and x for x in v)):
                raise bad("executor_enqueue", "a command as a non-empty list of its words, as [\"python3\", \"enqueue.py\"]")
            got["executor_enqueue"] = tuple(v)
        if "marked_after" in doc:
            if not (isinstance(doc["marked_after"], str) and MARK_TIME_RE.fullmatch(doc["marked_after"])):
                raise bad("marked_after", "a UTC time as YYYY-MM-DDTHH:MM:SSZ, as labels record it")
            got["marked_after"] = doc["marked_after"]
        if "lowrisk_per_tick" in doc:
            if type(doc["lowrisk_per_tick"]) is not int or doc["lowrisk_per_tick"] < 1:
                raise bad("lowrisk_per_tick", "a whole number, 1 or more (`lowrisk_lane = false` turns the lane off)")
            got["lowrisk_per_tick"] = doc["lowrisk_per_tick"]
        if "restack_jobs" in doc:
            if type(doc["restack_jobs"]) is not int or doc["restack_jobs"] < 1:
                raise bad("restack_jobs", "a whole number, 1 or more")
            got["restack_jobs"] = doc["restack_jobs"]
        for k in ("lowrisk_excludes", "verdict_excludes", "verdict_plain"):
            if k in doc:
                v = doc[k]
                if not (isinstance(v, list) and v and all(isinstance(x, str) and LOWRISK_EXCLUDE_RE.fullmatch(x)
                                                             and ".." not in x.split("/") for x in v)):
                    raise bad(k, "a non-empty list of directories from the repository's root, each ending in `/`, files' paths "
                                 "from it and file name globs, as [\"*.lean\", \"verity/core/\", \"verity/__init__.py\"]")
                got[k] = tuple(v)
        if plain := got.get("verdict_plain"):
            if dup := sorted({x for x in plain if plain.count(x) > 1}):
                raise bad("verdict_plain", f"{', '.join(dup)} written more than once: keep each entry once")
            if stray := [x for x in plain if x not in got.get("lowrisk_excludes", ())]:
                raise bad("verdict_plain", f"{', '.join(stray)} is no entry of lowrisk_excludes: each entry is one of "
                                           "lowrisk_excludes, written character for character as it is there; copy it from "
                                           "there, or remove it")
            if both := [x for x in plain if x in got.get("verdict_excludes", ())]:
                raise bad("verdict_plain", f"{', '.join(both)} is also an entry of verdict_excludes, which keeps it on the "
                                           "whole check: remove it from one of the two")
        if "bottom_ring" in doc:
            v = doc["bottom_ring"]
            if not (isinstance(v, list) and v and all(isinstance(x, str) and HANDLE_RE.fullmatch(x) for x in v)):
                raise bad("bottom_ring", "a non-empty list of handles, each `@` and its lower-case name, as [\"@ci\"]")
            got["bottom_ring"] = tuple(v)
        if "bottom_wait_min" in doc:
            if type(doc["bottom_wait_min"]) is not int or doc["bottom_wait_min"] < 1:
                raise bad("bottom_wait_min", "a whole number of minutes, 1 or more")
            got["bottom_wait_min"] = doc["bottom_wait_min"]
        for k in ("regenerate_locks", "verdict_first", "lowrisk_lane"):
            if k in doc:
                if not isinstance(doc[k], bool):
                    raise bad(k, "true or false")
                got[k] = doc[k]
        if "signed_verdicts" in doc:
            if (mode := signed_mode(doc["signed_verdicts"])) is None:
                raise bad("signed_verdicts", "false, \"shadow\" (count every mark and verdict, and log each that doesn't "
                                             "authenticate) or true")
            got["signed_verdicts"] = mode
        if "land" in doc:
            if doc["land"] == "check":
                raise bad("land", "train landing was deleted with its code: drive lands on review only; set "
                                  "`land = \"review\"` or remove the key")
            if doc["land"] not in LANDINGS:
                raise bad("land", f"one of {', '.join(LANDINGS)}")
            got["land"] = doc["land"]
        if "land_into" in doc:
            from .merge import CHECKED

            if doc["land_into"] not in ("main", CHECKED):
                raise bad("land_into", f"\"{CHECKED}\" (main follows it to fully checked commits) or \"main\"")
            got["land_into"] = doc["land_into"]
        if "after_landing_no_cache" in doc:
            if not isinstance(doc["after_landing_no_cache"], bool):
                raise bad("after_landing_no_cache", "true or false")
            got["after_landing_no_cache"] = doc["after_landing_no_cache"]
        for k in ("landing_gates", "warn_gates"):
            if k in doc:
                v = doc[k]
                if not (isinstance(v, list) and all(isinstance(x, str) and GATE_PATH_RE.fullmatch(x) and ".." not in x.split("/")
                                                    for x in v)):
                    raise bad(k, "a list of Python scripts by their path from the repository's root, as "
                                 "[\"adapters/x/tools/gate.py\"]")
                got[k] = tuple(v)
        if "warn_steps" in doc:
            if not (isinstance(doc["warn_steps"], list) and all(x in CHECK_STEPS for x in doc["warn_steps"])):
                raise bad("warn_steps", f"a list of the check's steps, of {', '.join(CHECK_STEPS)}")
            got["warn_steps"] = tuple(doc["warn_steps"])
        for k in ("stale_days", "stale_close_days"):
            if k in doc:
                if type(doc[k]) is not int or doc[k] < 0:
                    raise bad(k, "a whole number of days, 0 for off")
                got[k] = doc[k]
        if "stale_keep" in doc:
            if not (isinstance(doc["stale_keep"], list) and all(type(n) is int and n > 0 for n in doc["stale_keep"])):
                raise bad("stale_keep", "a list of PR numbers, as [2692, 2693]")
            got["stale_keep"] = tuple(doc["stale_keep"])
        if "command_line" in doc:
            v = doc["command_line"]
            if not (isinstance(v, list) and all(x in COMMAND_LINE for x in v)):
                raise bad("command_line", f"a list of the keys a command-line flag sets, of {', '.join(COMMAND_LINE)}")
            got["command_line"] = tuple(v)
        if got.get("stale_close_days") and not 0 < got.get("stale_days", 0) <= got["stale_close_days"]:
            raise bad("stale_close_days", "0, or at least stale_days, which must be on: drive closes only a stale PR")
        if got.get("signed_verdicts") == "on" and got.get("lowrisk_lane", cls.lowrisk_lane) and \
                (open_ := [p for p in SIGNED_GUARDS if lowrisk([p], got.get("lowrisk_excludes", cls.lowrisk_excludes))]):
            raise bad("signed_verdicts", f"true lets a change to {' or '.join(open_)} land on its ready mark alone, so a lane "
                                         "could make itself another handle's holder or loosen the check: add each to "
                                         "`lowrisk_excludes`, or set `lowrisk_lane = false`")
        return cls(**got)

    def line(self) -> str:
        return " ".join(f"{k}={','.join(map(str, v)) if isinstance(v, tuple) else v}"
                        for k, v in dataclasses.asdict(self).items())


EVERY = "*"


def priority_list(path: Path, flag: str = "--priority-file", comments: bool = False) -> tuple[int, ...]:
    return listed(path, flag, comments)[0]


def listed(path: Path, flag: str = "--priority-file", comments: bool = False, every: bool = False) -> tuple[tuple[int, ...], bool]:
    try:
        if not path.exists():
            return (), False
        text = path.read_text()
    except (OSError, UnicodeDecodeError) as e:
        raise QueueError(f"{flag} {path} can't be read ({getattr(e, 'strerror', None) or e}): make it a text file of "
                         "PR numbers separated by spaces or commas, or remove it") from None
    words = [w for w in re.split(r"[\s,]+", re.sub(r"#.*", "", text) if comments else text) if w]
    star = every and EVERY in words
    if bad := [w for w in words if not re.fullmatch(r"[1-9][0-9]*", w) and not (every and w == EVERY)]:
        raise QueueError(f"{flag} {path} holds {', '.join(map(repr, bad[:5]))}, which "
                         f"{'is not a PR number' if len(bad) == 1 else 'are not PR numbers'}: write PR numbers separated by "
                         f"spaces or commas{f', or {EVERY} for every PR' if every else ''}")
    return tuple(dict.fromkeys(int(w) for w in words if w != EVERY)), star


GATE_TIMEOUT_S = 600
GATE_FIX_LINES = 15
GATE_VERDICTS = 1000
WARN_GATE_SAYS = "A warn gate refuses nothing: the PR lands, and its owner fixes it in their next PR"
RATCHET_SAYS = ("The ratchet gate refuses nothing here, since it can read a violation whose line moved, or an import cycle through a "
                "module the PR adds, as new: check's `ratchet-gate` step, off the merge gate, judges the PR's merge again in its "
                "candidate")
COLLIDE_SAYS = ("Nothing is refused for it: the verified queue refuses whichever of the two it merges second, for its conflict "
                "with `main` by then. To be the second without that refusal, wait for the PR named to land, merge `main` and mark "
                "the new head; or, where the PR named is your own, merge its head (`git merge SHA`), say in the body that this PR "
                "is stacked on it, and mark the new head, which lands after it")
MERGES = 2000


def kept_verdicts(state: dict, key: str, main: str, gates: tuple[str, ...]) -> dict[str, list[str]]:
    cache = state.get(key) or {}
    return (cache.get("of") or {}) if (cache.get("main"), cache.get("gates")) == (main, list(gates)) else {}


def flags_at(repo: Path, ref: str) -> DriveFlags:
    r = _git(repo, "show", f"{ref}:{DRIVE_FLAGS}", check=False)
    return DriveFlags.parse(r.stdout, f"{DRIVE_FLAGS} at {ref}") if r.returncode == 0 else DriveFlags()


def gates_at(repo: Path, ref: str, key: str = "landing_gates") -> tuple[str, ...]:
    r = _git(repo, "show", f"{ref}:{DRIVE_FLAGS}", check=False)
    if r.returncode:
        return ()
    source = f"{DRIVE_FLAGS} at {ref}"
    try:
        gates = tomllib.loads(r.stdout).get(key)
    except tomllib.TOMLDecodeError as e:
        raise QueueError(f"{source}: not TOML: {e}") from None
    if gates is None:
        return ()
    return getattr(DriveFlags.parse(f"{key} = {json.dumps(gates)}\n", source), key)


def land_into_key(repo: Path, ref: str) -> str | None:
    r = _git(repo, "show", f"{ref}:{DRIVE_FLAGS}", check=False)
    if r.returncode:
        return None
    source = f"{DRIVE_FLAGS} at {ref}"
    try:
        into = tomllib.loads(r.stdout).get("land_into")
    except tomllib.TOMLDecodeError as e:
        raise QueueError(f"{source}: not TOML: {e}") from None
    return None if into is None else DriveFlags.parse(f"land_into = {json.dumps(into)}\n", source).land_into


def into_at(repo: Path, ref: str) -> str:
    return land_into_key(repo, ref) or "main"


def signed_at(repo: Path, ref: str) -> str:
    r = _git(repo, "show", f"{ref}:{DRIVE_FLAGS}", check=False)
    if r.returncode:
        return "off"
    source = f"{DRIVE_FLAGS} at {ref}"
    try:
        value = tomllib.loads(r.stdout).get("signed_verdicts", False)
    except tomllib.TOMLDecodeError as e:
        raise QueueError(f"{source}: not TOML: {e}") from None
    if (mode := signed_mode(value)) is None:
        raise QueueError(f"{source}: signed_verdicts = {value!r}: false, \"shadow\" or true")
    return mode


def _handles_at(repo: Path, ref: str) -> list[dict]:
    r = _git(repo, "show", f"{ref}:{REGISTRY}", check=False)
    try:
        reg = json.loads(r.stdout).get("handles", []) if r.returncode == 0 else []
    except (ValueError, AttributeError) as e:
        raise QueueError(f"{REGISTRY} at {ref} can't be parsed ({e}), so who holds each handle is unknown: fix it") from None
    return [h for h in reg if isinstance(h, dict) and "handle" in h]


def holders_at(repo: Path, ref: str) -> dict[str, str]:
    return {a: h["handle"] for h in _handles_at(repo, ref) for a in AGENT_RE.findall(str(h.get("holder", "")))}


def staffs_of(handles: list[dict], source: str) -> dict[str, tuple[str, ...]]:
    names = [_who(str(h["handle"])) for h in handles]
    out: dict[str, tuple[str, ...]] = {}
    for h in handles:
        if (globs := h.get("staffs")) is None:
            continue
        bad = None
        if not (isinstance(globs, list) and all(isinstance(g, str) and STAFFS_RE.fullmatch(g) for g in globs)):
            bad = "it is a list of names or globs of names, lower-case, as [\"review-*\"]"
        elif taken := sorted({n for g in globs for n in names if fnmatch.fnmatchcase(n, g)}):
            bad = f"it matches {', '.join(taken)}, and a handle is delegated only by its own holder"
        if bad:
            raise QueueError(f"{source}: {h['handle']}'s staffs = {globs!r}: {bad}: fix it")
        out[_who(str(h["handle"]))] = tuple(globs)
    return out


def staffs_at(repo: Path, ref: str) -> dict[str, tuple[str, ...]]:
    return staffs_of(_handles_at(repo, ref), f"{REGISTRY} at {ref}")


def run_gate(repo: Path, main: str, head: str, path: str) -> tuple[int | None, str]:
    r = _git(repo, "show", f"{main}:{path}", check=False)
    if r.returncode:
        return None, f"{path} isn't at `main` {main[:12]}"
    with tempfile.TemporaryDirectory(prefix="queue-gate-", ignore_cleanup_errors=True) as d:
        script = Path(d) / Path(path).name
        script.write_text(r.stdout, errors="surrogateescape")
        try:
            p = subprocess.run([sys.executable, str(script), main, head, "--json"], cwd=repo, capture_output=True, text=True,
                               stdin=subprocess.DEVNULL, timeout=GATE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return None, f"{path} ran over {GATE_TIMEOUT_S} s"
    return p.returncode, p.stdout if p.returncode in (0, 1) else (p.stderr or p.stdout).strip()[-400:]


def scripts_at(repo: Path, at: str, paths: Iterable[str], suffix: str = ".py") -> dict[str, str]:
    out = {}
    for line in _out(repo, "ls-tree", "-z", at, "--", *paths).split("\0"):
        meta, _, path = line.partition("\t")
        if path.endswith(suffix) and meta.split()[1:2] == ["blob"]:
            out[path] = meta.split()[2]
    return out


def move_scripts(repo: Path, at: str) -> dict[str, str]:
    return {**scripts_at(repo, at, (MOVE_DIR, MOVE_CHECK, MOVE_PINS)), **scripts_at(repo, at, (MOVE_LINES,), "")}


def restack_files(repo: Path, at: str) -> dict[str, str]:
    return scripts_at(repo, at, (RESTACK, MOVE_LINES), "")


MOVE_SCRIPTS = (f"{MOVE_DIR}*.py", MOVE_LINES)


def path_rows(text: str) -> list[tuple[str, str]]:
    data = tomllib.loads(text)
    if not isinstance(got := data.get("words_in") or [], list):
        raise ValueError(f"holds a words_in that is no array of tables ({type(got).__name__} {got!r:.60})")
    tables = [data.get("words"), *(s.get("words") for s in data.get("words_in") or [] if isinstance(s, dict))]
    rows = [(o, n) for t in tables if isinstance(t, dict) for o, n in t.items()]
    return list(dict.fromkeys((o, n) for o, n in rows if isinstance(n, str) and "/" in o and "/" in n))


def write_scripts(repo: Path, scripts: dict[str, str], dest: Path) -> Path:
    for path, blob in scripts.items():
        f = dest / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(_git(repo, "cat-file", "blob", blob).stdout, errors="surrogateescape")
    return dest


def gate_refusal(repo: Path, main: str, head: str, gates: Iterable[str],
                 run: Callable[[Path, str, str, str], tuple[int | None, str]] = run_gate,
                 onto: str | None = None, kind: str = "landing gate") -> tuple[list[str], list[str]]:
    what, merged = (head[:12], "") if onto is None else (f"its merge onto `{onto}` ({head[:12]})", f" on its merge onto `{onto}`")
    rerun = "in the branch" if onto is None else f"on the branch merged with `origin/{onto}`"
    refused, unjudged = [], []
    for path in gates:
        rc, out = run(repo, main, head, path)
        if rc == 0:
            continue
        failures = None
        if rc == 1:
            with contextlib.suppress(ValueError, TypeError, KeyError, AttributeError):
                failures = [(str(f["why"]), [str(x) for x in (f.get("fix") or [])][:GATE_FIX_LINES])
                            for f in json.loads(out)["failures"]]
        if not failures:
            why = out if rc is None else f"exit {rc}" + (f": {out}" if rc != 1 else ": no failure it names")
            unjudged.append(f"the {kind} {path} can't judge {what} on `main` {main[:12]} ({why.strip()[-400:]})")
            continue
        for why, fix in failures:
            refused.append(f"the {kind} {path} at `main` {main[:12]}{merged}: {why}" + "".join(f"; {x}" for x in fix)
                           + f" (`python3 {path} origin/main HEAD` {rerun} runs it)")
    return refused, unjudged


def label_target(pr: int, head: str) -> str:
    return f"pr:{pr}@{head}"


def _who(name: str) -> str:
    return name.strip().lstrip("@").lower()


def unreviewed(labels: Iterable, owners: Iterable[str | None]) -> str | None:
    mine = {_who(o) for o in owners if o}
    newest: dict[str, object] = {}
    for lab in sorted(labels, key=lambda lab: lab.ts):
        newest[_who(lab.by)] = lab
    if held := [f"{lab.by.lstrip('@')} ({lab.ref or 'no reason'})" for lab in newest.values() if lab.value == "hold"]:
        return f"held by {', '.join(held)}"
    if any(lab.value == "land" and by not in mine for by, lab in newest.items()):
        return None
    if own := [lab.by.lstrip("@") for by, lab in newest.items() if by in mine]:
        return f"no land verdict but its owner's own ({', '.join(own)}), which doesn't count"
    return "no land verdict"


LEAN_LOCK = "lean-audit.json"
LEAN_SOURCES = (".lean", "lakefile.lean", "lakefile.toml", "lean-toolchain", "lake-manifest.json")


LEAN_PROOFS_TREE = "security/"
LEAN_SPEC = ("security/Definitions/", "security/Definitions.lean", "security/lakefile.toml", "security/lake-manifest.json",
             "security/lean-toolchain")


def lean_change(repo: Path, commit: str, base: str | None = None) -> bool:
    files = changed(repo, base or f"{commit}^1", commit)
    if any(Path(f).name == LEAN_LOCK or f.endswith(LEAN_SOURCES) and f.startswith(LEAN_SPEC) for f in files):
        return True
    if not (lean := [f for f in files if f.endswith(LEAN_SOURCES) and not f.startswith(LEAN_PROOFS_TREE)]):
        return False
    pkgs = {str(Path(f).parent) for f in _out(repo, "ls-tree", "-r", "--name-only", commit).splitlines() if Path(f).name == LEAN_LOCK}
    return any(p == "." or f.startswith(p + "/") for f in lean for p in pkgs)


def lowrisk(files: Iterable[str], excludes: Iterable[str]) -> bool:
    dirs, paths, names = [], set(), []
    for x in excludes:
        if x.endswith("/"):
            dirs.append(x)
        elif "/" in x:
            paths.add(x)
        else:
            names.append(x)
    files, dirs = list(files), tuple(dirs)
    return bool(files and (dirs or paths or names)) and not any(
        suite_layout.held(f, dirs) or f in paths or any(fnmatch.fnmatchcase(PurePosixPath(f).name, n) for n in names)
        for f in files)


LOWRISK_ROW = "low-risk: lands on its mark"


def own_files(repo: Path, main: str, head: str) -> frozenset[str]:
    return frozenset(_out(repo, "diff", "--name-only", "--no-renames", f"{main}...{head}").splitlines())


def lands_on_mark(flags: DriveFlags, pr: PR, why: str | None, files: Iterable[str]) -> bool:
    return (flags.lowrisk_lane and not pr.move and not (why or "").startswith("held")
            and lowrisk(files, flags.lowrisk_excludes))


PREFETCH_JOBS = 128


class StoreLabels:
    def __init__(self, store, log: Callable[[str], None] = print, since: str | None = None, verify=None,
                 change: Callable[[str], str | None] | None = None, moved: Callable[[str, str], str | None] | None = None,
                 owners: Callable[[int], Iterable[str | None]] | None = None):
        self.store, self.log, self.since, self.verify, self.change, self.moved = store, log, since, verify, change, moved
        self.owners = owners
        self.shadow = False
        self.unauthenticated: dict[tuple[int, str, str, str, str], str] = {}
        self._pulled: set[int] = set()
        self._newest: dict[tuple[int, str], object] = {}
        self._fresh: set[str] = set()
        self._dirs: dict[str, list[str]] | None = None
        self._said: set[tuple[int, str, str]] = set()

    def prefetch(self, heads: Iterable[tuple[int, str]]) -> None:
        from .store import labels_sync

        heads = list(heads)
        targets, prs = [label_target(n, h) for n, h in heads], sorted({n for n, _ in heads})
        if not targets or getattr(self.store, "remote", None) is None:
            return
        try:
            if self.change is None:
                labels_sync.pull_targets(self.store, targets, jobs=PREFETCH_JOBS)
            else:
                labels_sync.pull_prefixes(self.store, [label_target(n, "") for n in prs], jobs=PREFETCH_JOBS)
        except Exception:  # noqa: BLE001
            return
        self._fresh.update(targets)
        if self.change is not None:
            self._pulled.update(prs)
            self._dirs = None

    def ready(self, pr: int, head: str) -> bool:
        labs = sorted(self._counted(pr, head, vocab.READY)[0], key=lambda lab: lab.ts)
        self._newest[(pr, head)] = labs[-1] if labs else None
        return bool(labs) and str(labs[-1].value).lower() == "true" and (self.since is None or labs[-1].ts >= self.since)

    def reviews(self, pr: int, head: str) -> list:
        return self._counted(pr, head, vocab.REVIEWED)[0]

    def ready_by(self, pr: int, head: str) -> str | None:
        labs = [lab for lab in sorted(self._counted(pr, head, vocab.READY)[0], key=lambda lab: lab.ts) if str(lab.value).lower() == "true"]
        return labs[-1].by if labs else None

    def refusals(self, pr: int, head: str, key: str) -> list[str]:
        return self._counted(pr, head, key)[1]

    def lands(self, pr: int, head: str) -> bool:
        marks = [lab for lab in self._verified(pr, head, self._labels(pr, head, vocab.READY))[0] if str(lab.value).lower() == "true"]
        mine = {*(self.owners(pr) if self.owners is not None else ()), max(marks, key=lambda lab: lab.ts).by if marks else None}
        return unreviewed(self._verified(pr, head, self._labels(pr, head, vocab.REVIEWED))[0], mine) is None

    def _counted(self, pr: int, head: str, key: str) -> tuple[list, list[str]]:
        counted, refused = self._verified(pr, head, self._labels(pr, head, key))
        if counted or self.change is None or not (others := self._other_heads(pr, head, key)) or not (own := self.change(head)):
            return counted, refused
        came, moved = [], []
        for old, labs in others.items():
            if (same := self.change(old) == own) or (self.moved is not None and self.lands(pr, old)
                                                     and self.moved(old, head) == own):
                got, why = self._verified(pr, old, labs)
                counted, refused = [*counted, *got], [*refused, *why]
                (came if same else moved).extend([old] if got else [])
        if (came or moved) and (pr, head, key) not in self._said:
            self._said.add((pr, head, key))
            for heads, how in ((came, "theirs"), (moved, "theirs across the moves `main` landed since, as a restack carries it")):
                if heads:
                    self.log(f"research queue: #{pr} at {head[:8]} takes the `{key}` labels at {', '.join(c[:8] for c in heads)}: "
                             f"none of its own counts, and its own change is {how} (patch-id {own[:12]})")
        return counted, refused

    def _verified(self, pr: int, head: str, labs: list) -> tuple[list, list[str]]:
        if self.verify is None:
            return labs, []
        from .labelauth import landing

        beside = self.store.labels(label_target(pr, head), key=vocab.LABEL_AUTH)
        counted, refused = [], []
        for lab in labs:
            if not landing(lab):
                counted.append(lab)
            elif not self.shadow:
                if why := self.verify.why(lab, beside):
                    refused.append(f"{lab.key} = {lab.value} by {lab.by} at {lab.ts}: {why}")
                else:
                    counted.append(lab)
            else:
                try:
                    why = self.verify.why(lab, beside)
                except Exception as e:  # noqa: BLE001 -- the shadow changes nothing the queue lands
                    why = f"it can't be judged: {type(e).__name__}: {e}"
                if why:
                    self.unauthenticated[(pr, head, lab.key, lab.by, lab.ts)] = why
                counted.append(lab)
        return counted, refused

    def marked_by(self, pr: int, head: str) -> str | None:
        lab = self._newest.get((pr, head))
        return None if lab is None else lab.by

    def marked_at(self, pr: int, head: str) -> str | None:
        lab = self._newest.get((pr, head))
        return None if lab is None else lab.ts

    def before_cutoff(self, pr: int, head: str) -> str | None:
        lab = self._newest.get((pr, head))
        if lab is None or self.since is None or str(lab.value).lower() != "true" or lab.ts >= self.since:
            return None
        return lab.ts

    def _other_heads(self, pr: int, head: str, key: str) -> dict[str, list]:
        from .store import labels_sync

        prefix = label_target(pr, "")
        if pr not in self._pulled:
            try:
                labels_sync.pull_prefix(self.store, prefix)
            except Exception as e:  # noqa: BLE001  (no pull: local labels alone may be stale)
                raise QueueError(f"STALE: the labels of {prefix}* weren't pulled from the store's remote ({type(e).__name__}: {e}), "
                                 f"so ready marks on #{pr}'s other heads here may be old") from None
            self._pulled.add(pr)
            self._dirs = None
        if self._dirs is None:
            root, self._dirs = Path(self.store.root) / "labels", {}
            for name in sorted(os.listdir(root)) if root.is_dir() else ():
                at, sep, old = name.rpartition("@")
                if sep:
                    self._dirs.setdefault(at + sep, []).append(old)
        out = {}
        for old in self._dirs.get(ids.target_dirname(prefix), ()):
            if old != head and ids.is_pr(label_target(pr, old)) and (labs := self.store.labels(label_target(pr, old), key=key)):
                out[old] = labs
        return out

    def _labels(self, pr: int, head: str, key: str) -> list:
        from .store import labels_sync
        from .store.local import StoreError

        target = label_target(pr, head)
        if target in self._fresh or pr in self._pulled:
            return self.store.labels(target, key=key)
        try:
            labels_sync.read_through(self.store, [target], lambda s: self.log(f"research queue: {s}"), strict=True)
        except StoreError as e:
            raise QueueError(f"STALE: the labels of {target} weren't pulled from the store's remote ({e}), so ready marks here may "
                             f"be old; not answering from them") from None
        return self.store.labels(target, key=key)


@dataclass(frozen=True)
class Member:
    pr: int
    head: str
    owner: str
    commit: str


@dataclass
class Entry:
    pr: PR
    state: str = "ready"
    why: str = ""
    after: tuple[int, ...] = ()
    older: tuple[int, ...] = ()
    held: dict[int, str] = field(default_factory=dict)
    position: int | None = None
    commit: str | None = None
    idle_since: float | None = None

    def contains(self, n: int) -> str:
        return f"#{n}'s commits (at an older head)" if n in self.older else f"#{n}"

    def holds(self, n: int) -> str:
        return f"#{n}'s commits at its older head {self.held[n][:12]}" if n in self.older else f"#{n}'s head {self.held[n][:12]}"


def contained(repo: Path, base: str, head: str, heads: dict[str, int]) -> list[int]:
    return sorted({heads[h] for h in _out(repo, "rev-list", f"{base}..{head}").split() if h in heads})


def carried_heads(repo: Path, landed: str, head: str) -> list[tuple[int, str, str]]:
    log = _out(repo, "log", "--format=%H%x00%P%x00%B%x1e", "--grep=^Queue: #", f"{landed}..{head}")
    made = (queue_commit(*r.strip().split("\x00", 2)) for r in log.split("\x1e") if r.strip())
    return [(m.pr, m.head, m.commit) for m in made
            if m and _git(repo, "merge-base", "--is-ancestor", m.head, landed, check=False).returncode]


def stale_carried(repo: Path, carried: Iterable[tuple[int, str, str]], heads: dict[int, str],
                  admitted: Callable[[int], str | None], landed: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for n, h, c in carried:
        if n in out:
            continue
        at, cur = f"it carries #{n}'s head {h[:12]} through `{BRANCH}`'s merge {c[:12]}", heads.get(n)
        fix = f"merge origin/main and #{n}'s current head instead of `{BRANCH}`, or drop #{n}'s commits"
        if cur is None:
            out[n] = f"{at}, and #{n} is closed: drop #{n}'s commits (rebuild the branch on origin/main, without `{BRANCH}`)"
        elif h != cur and _git(repo, "merge-base", "--is-ancestor", h, cur, check=False).returncode:
            out[n] = f"{at}, which neither `{landed}` nor #{n}'s current head {cur[:12]} holds: {fix}"
        elif (no := admitted(n)) is not None:
            out[n] = f"{at}, and #{n} isn't admitted at {cur[:12]} ({no}): {fix}"
    return out


def _brought(parents: dict[str, list[str]], heads: Iterable[tuple[int, str]]) -> dict[str, int]:
    at: dict[str, int] = {}
    for i, h in heads:
        todo = [h]
        while todo:
            if (c := todo.pop()) in parents and c not in at:
                at[c] = i
                todo += parents[c]
    return at


def above(repo: Path, base: str, heads: Iterable[str]) -> dict[str, tuple[list[str], set[str]]]:
    heads = list(dict.fromkeys(heads))
    if not heads:
        return {}
    r = subprocess.run(["git", "rev-list", "--parents", "--stdin"], cwd=repo, input="".join(f"{h}\n" for h in [*heads, f"^{base}"]),
                       capture_output=True, text=True, env=git_env(GIT_TERMINAL_PROMPT="0"))
    if r.returncode:
        raise QueueError(f"git rev-list --parents {base}.. ({len(heads)} heads): {(r.stderr or r.stdout).strip()[-400:]}")
    parents = {c: ps for c, *ps in (line.split() for line in r.stdout.splitlines())}
    out = {}
    for h in heads:
        below: set[str] = set()
        todo = [h] if h in parents else []
        while todo:
            if (c := todo.pop()) not in below:
                below.add(c)
                todo += [p for p in parents[c] if p in parents]
        children = dict.fromkeys(below, 0)
        for c in below:
            for p in parents[c]:
                if p in children:
                    children[p] += 1
        order, todo = [], [h] if below else []
        while todo:
            order.append(c := todo.pop())
            for p in parents[c]:
                if p in children:
                    children[p] -= 1
                    if not children[p]:
                        todo.append(p)
        line, c = set(), h
        while c in parents:
            line.add(c)
            c = parents[c][0] if parents[c] else ""
        out[h] = (order, line)
    return out


def based_on(pr: PR, prs: Iterable[PR]) -> str:
    parent = next((p for p in prs if p.branch == pr.base and p.number != pr.number), None)
    return f"{pr.base} (#{parent.number}'s head)" if parent else pr.base


def _message(pr: PR) -> str:
    return f"Queue: #{pr.number} {pr.title}".rstrip() + f"\n\nPR: #{pr.number}\nHead: {pr.head}\nOwner: {pr.owner}\n"


def _move_message(pr: PR, tip: str, into: str, main: str) -> str:
    return (f"Queue: #{pr.number} {pr.title}".rstrip() + f"\n\n`python3 {RENAME} --map {pr.move}` at the root of {tip}, regenerated "
            f"by the queue for #{pr.number} and certified by `{MOVE_CHECK} {tip[:12]} HEAD`, both as `{into}` {main[:12]} has them."
            f"\n\nPR: #{pr.number}\nHead: {pr.head}\nOwner: {pr.owner}\nMove: {pr.move}\n")


def queue_commit(commit: str, parents: str, body: str) -> Member | None:
    t = {k: v.strip() for k, v in TRAILER_RE.findall(body)}
    if (len(parents.split()) > 1 or MOVE_TRAILER_RE.search(body)) and body.startswith("Queue: #") and {"PR", "Head", "Owner"} <= set(t) \
            and re.fullmatch(r"#\d+", t["PR"]) and re.fullmatch(r"[0-9a-f]{40}", t["Head"]):
        return Member(int(t["PR"][1:]), t["Head"], t["Owner"], commit)
    return None


class Queue:
    def __init__(self, workdir: Path, *, remote: str, github, ready: Callable[[int, str], bool],
                 store=None, shadow: bool = False, rules: Rules | None = None, log: Callable[[str], None] = print,
                 auth: tuple[str, ...] = (),
                 before_cutoff: Callable[[int, str], str | None] | None = None,
                 marked_by: Callable[[int, str], str | None] | None = None,
                 marked_at: Callable[[int, str], str | None] | None = None,
                 prefetch: Callable[[list[tuple[int, str]]], None] | None = None, verdict_first: bool | None = None,
                 land_into: str | None = None):
        self.workdir, self.remote, self.github, self.ready, self.store = Path(workdir), remote, github, ready, store
        self.land_into, self._into = land_into, None
        self.into_fetched = False
        self.shadow, self.log, self._rules, self.auth = shadow, log, rules, auth
        self.before_cutoff, self.marked_by, self.marked_at, self.prefetch = before_cutoff, marked_by, marked_at, prefetch
        self.verdict_first = verdict_first
        self.lift: list[int] = []
        self._lifted: list[int] = []
        self.park: list[int] = []
        self._parked: list[int] = []
        self.now: Callable[[], float] = time.time
        self.priority: tuple[int, ...] = ()
        self.priority_file: Path | None = None
        self._prioritized: set[int] = set()
        self.unlock: Callable[[], None] = lambda: None
        self.branch = SHADOW_BRANCH if shadow else BRANCH
        self.state_path = self.workdir.parent / f"{self.workdir.name}.state.json"
        self._touched: dict[str, set[str]] = {}
        self._registry: tuple[set, dict[str, str], set[str]] | None = None
        self.listed: list[PR] = []
        self._moves: list[Move] = []
        self._upcoming: list[str] = []
        self._made: dict[str, dict] = {}
        self._restacks: dict[str, dict] = {}
        self._restacking: dict[str, dict] = {}
        self._used: set[str] = set()
        self._words_of: dict[str, set[str]] = {}
        self.restack_jobs = DriveFlags.restack_jobs
        self._restack_starts = False
        self._merge_records: dict[str, dict] = {}
        self._merge_serial = 0
        self._merge_keys: dict[tuple[str, str], str] = {}
        self._context: str | None = None
        self._merged_from: dict[str, tuple[str, str]] = {}
        self._merges_reused = 0
        self._own: dict[tuple[str, str], str | None] = {}
        self._kept: dict[str, str | None] | None = None
        self._kept_at = Path()
        self._mapped: dict[tuple[str, str], str | None] = {}
        self._mapped_kept: dict[str, str] | None = None
        self._mapped_at: Path | None = None
        self._mapped_elsewhere: dict[str, str] = {}
        self._mapped_seen: tuple[tuple[str, int, int], ...] = ()
        self._mapped_failed: dict[str, str] = {}
        self._crossed: dict[tuple[str, str], list[tuple[str, list[str]]]] = {}
        self._mapped_wanted: dict[str, tuple[str, str, list[tuple[str, list[str]]]]] = {}
        self.mapped_jobs = 0

    def _remote(self, cwd: Path, *args: str, ok: tuple[int, ...] | None = (0,)) -> subprocess.CompletedProcess:
        r = _git(cwd, *STALL, *self.auth, *args, check=False)
        if ok is not None and r.returncode not in ok:
            err = (r.stderr or r.stdout).strip()[-400:]
            hint = ("; no credential helper applies to the remote outside the caller's checkout: set one in git's global config, "
                    "or put an authenticated `gh` on PATH") if not self.auth and AUTH_FAILED_RE.search(err) else ""
            raise QueueError(f"git {' '.join(args)}: {err}{hint}")
        return r

    def _ours(self) -> bool:
        return (_git(self.workdir, "config", "--local", "--get", CLONE_MARK, check=False).stdout.strip() == "true"
                or self.workdir.resolve() == WORKDIR.resolve())

    def _clone(self) -> None:
        if fresh := not (self.workdir / ".git").exists():
            self.workdir.parent.mkdir(parents=True, exist_ok=True)
            self._remote(self.workdir.parent, "clone", "-q", "--config", f"{CLONE_MARK}=true", self.remote, str(self.workdir))
        if fresh or self._ours():
            # the caller's origin as it is now: a token embedded in the URL the clone was made from rotates
            _git(self.workdir, "remote", "set-url", "origin", self.remote)
        elif not same_repository(origin := _git(self.workdir, "remote", "get-url", "origin", check=False).stdout.strip(),
                                 self.remote):
            here, there = redacted(origin), redacted(self.remote)
            raise QueueError(f"{self.workdir} is a checkout the queue didn't clone (its git config has no {CLONE_MARK}), whose "
                             f"origin, {here or 'unset'}, is another repository than the queue's remote {there}, and the queue "
                             f"never repoints such a checkout: pass --workdir a directory of its own, where the queue clones "
                             f"{there}" + (f", or --remote {here}, the repository that checkout is a clone of" if here else ""))
        from .clone import Refused, set_drivers

        try:
            if (changed := set_drivers(self.workdir)) and not fresh:
                self.log(f"research queue: set the record merge drivers in {self.workdir}'s config, which lacked or held "
                         f"another value for {', '.join(changed)}")
        except Refused as e:
            raise QueueError(str(e)) from None
        _git(self.workdir, "merge", "--abort", check=False)
        _git(self.workdir, "reset", "-q", "--hard")
        _git(self.workdir, "clean", "-qfdx", "-e", "/.merge_file_*")

    def fetch(self, prs: Iterable[PR] = ()) -> dict[int, str]:
        prs = list(prs)
        self._registry, self._into = None, None
        specs = ["+refs/heads/main:refs/remotes/origin/main", *(f"+refs/pull/{p.number}/head:refs/queue/pr/{p.number}" for p in prs)]
        if self._remote(self.workdir, "fetch", "-q", "--prune", "origin", *specs, ok=None).returncode:
            self._remote(self.workdir, "fetch", "-q", "--prune", "origin", specs[0])
            for p, spec in zip(prs, specs[1:], strict=True):
                if (r := self._remote(self.workdir, "fetch", "-q", "origin", spec, ok=None)).returncode:
                    _git(self.workdir, "update-ref", "-d", f"refs/queue/pr/{p.number}", check=False)
                    self.log(f"research queue: #{p.number}'s pull ref can't be fetched, so it waits this sync: "
                             f"{(r.stderr or r.stdout).strip()[-200:]}")
        self.into_fetched = False
        if (into := self.into) != "main":
            if self._remote(self.workdir, "ls-remote", "--exit-code", "origin", f"refs/heads/{into}", ok=(0, 2)).returncode == 0:
                self._remote(self.workdir, "fetch", "-q", "origin", f"+refs/heads/{into}:{self.base}")
                self.into_fetched = True
            else:
                _git(self.workdir, "update-ref", self.base, "refs/remotes/origin/main")
        if _git(self.workdir, "merge-base", "--is-ancestor", self.base, "HEAD", check=False).returncode:
            _git(self.workdir, "checkout", "-q", "--detach", "--force", self.base)
        # exit 2: no such branch; any other failure is the remote's, never an empty queue
        if self._remote(self.workdir, "ls-remote", "--exit-code", "origin", f"refs/heads/{self.branch}", ok=(0, 2)).returncode == 0:
            self._remote(self.workdir, "fetch", "-q", "origin", f"+refs/heads/{self.branch}:refs/remotes/origin/{self.branch}")
        else:
            _git(self.workdir, "update-ref", "-d", f"refs/remotes/origin/{self.branch}", check=False)
        if not prs:
            return {}
        refs = dict(line.split() for line in _out(self.workdir, "for-each-ref", "--format=%(refname) %(objectname)",
                                                  "refs/queue/pr/").splitlines())
        return {p.number: refs[ref] for p in prs if (ref := f"refs/queue/pr/{p.number}") in refs}

    def fetch_prs(self, prs: list[PR]) -> tuple[list[PR], dict[int, Entry]]:
        fetched = self.fetch(prs)
        consistent, waiting = [], {}
        for p in prs:
            if fetched.get(p.number) != p.head and p.branch and self._head_from_branch(p):
                fetched[p.number] = p.head
            if p.number not in fetched:
                waiting[p.number] = Entry(p, state="waiting", why="GitHub has no pull ref for its head yet; retried at the next reading")
                continue
            if fetched[p.number] == p.head:
                consistent.append(p)
                continue
            why = f"GitHub's API head {p.head[:12]} and pull ref {fetched[p.number][:12]} disagree; retried at the next reading"
            waiting[p.number] = Entry(p, state="waiting", why=why)
            self.log(f"research queue: #{p.number}'s API head {p.head[:12]} and fetched pull ref "
                     f"{fetched[p.number][:12]} disagree; retried at the next reading")
        return consistent, waiting

    def _head_from_branch(self, p: PR) -> bool:
        ref = f"refs/queue/branch/{p.number}"
        r = self._remote(self.workdir, "fetch", "-q", "origin", f"+refs/heads/{p.branch}:{ref}", ok=None)
        tip = _git(self.workdir, "rev-parse", "-q", "--verify", ref, check=False).stdout.strip() if not r.returncode else ""
        _git(self.workdir, "update-ref", "-d", ref, check=False)
        if tip != p.head:
            return False
        _git(self.workdir, "update-ref", f"refs/queue/pr/{p.number}", tip)
        self.log(f"research queue: #{p.number}'s pull ref isn't at its API head {p.head[:12]}; took the head from its branch")
        return True

    def load_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {"ejected": {}, "conflicts": {}}

    def save_state(self, state: dict) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
        tmp.replace(self.state_path)

    def rules(self) -> Rules:
        if self._rules is None:
            self._rules = rules_at(self.workdir, "origin/main")
            if self._rules is None:
                raise QueueError(f"`main` has no {RULES}: pass --rules FILE")
        return self._rules

    @property
    def identity(self) -> dict[str, str]:
        return IDENTITY if self._rules is None else self._rules.identity

    @property
    def into(self) -> str:
        if self.land_into is not None:
            return self.land_into
        if self._into is None:
            self._into = into_at(self.workdir, "refs/remotes/origin/main")
        return self._into

    @property
    def base(self) -> str:
        return f"refs/remotes/origin/{self.into}"

    @property
    def slug(self) -> str:
        return getattr(self.github, "repo", None) or REPO

    def chain(self) -> list[Member]:
        return chain_of(self.workdir, self.base, f"refs/remotes/origin/{self.branch}")

    def own_change(self, head: str, landing: str | None = None) -> str | None:
        if (key := (landing or self.base, head)) in self._own:
            return self._own[key]
        if not (mb := _git(self.workdir, "merge-base", key[0], head, check=False).stdout.strip()):
            self._own[key] = None
            return None
        if self._kept is None:
            self._kept_at = self.workdir / _out(self.workdir, "rev-parse", "--git-common-dir") / "own-changes.json"
            self._kept = json.loads(self._kept_at.read_text()) if self._kept_at.is_file() else {}
        if (at := f"{mb} {head}") not in self._kept:
            self._kept[at] = own_change(self.workdir, mb, head)
            self._kept = dict(list(self._kept.items())[-OWN_KEEP:])
            tmp = self._kept_at.with_name(f"{self._kept_at.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self._kept))
            tmp.replace(self._kept_at)
        self._own[key] = self._kept[at]
        return self._own[key]

    def mapped_change(self, old: str, new: str) -> str | None:
        if (key := (old, new)) in self._mapped:
            return self._mapped[key]
        onto = _git(self.workdir, "merge-base", self.base, new, check=False).stdout.strip()
        if not onto or not (moves := self._moves_crossed(onto, old)) or (
                (got := self._mapped_got(at := mapped_key(old, moves))) is None and at in self._mapped_failed):
            self._mapped[key] = None
            return None
        if got is None:
            if self.mapped_jobs:
                self._mapped_wanted.setdefault(at, (old, onto, moves))
                return None
            self._mapped_done(at, *self._mapped_one(old, onto, moves))
            got = self._mapped_got(at)
        self._mapped[key] = got
        return got

    def work_out_mapped(self) -> int:
        from .landing import _hhmm, _job_end

        self._kept_mapped()
        drop, now, ended = self._mapped_at.with_suffix(".d"), self.now(), []
        running = set()
        for mark in sorted(drop.glob(f"*{MAPPING_MARK}")):
            result = mark.with_suffix(MAPPED_SUFFIX)
            try:
                job = json.loads(mark.read_text())
                end, at = _job_end(job, result.name), job["at"]
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
                job, end, at = {}, f"its marker doesn't read: {type(e).__name__}: {e}", ""
            if end is None and now <= job.get("until", now):
                running.add(mark.name)
                continue
            mark.unlink(missing_ok=True)
            if not isinstance(end, tuple) and job.get("dir"):
                shutil.rmtree(job["dir"], ignore_errors=True)
            if isinstance(end, tuple) and not result.is_file():
                rc, tail = end[0], " | ".join(end[1].splitlines()[-3:])
                write_mapped(result, at, None, f"its mapping job exited {rc} with none written" + (f": {tail}" if tail else ""))
            if result.is_file():
                ended += [at] if at else []
                continue
            why = end or f"none written by {_hhmm(job['until'])}, its time"
            self.log(f"research queue: the mapping job of {mark.stem} ended with no result ({why}): a later tick that wants "
                     "it starts it again")
        self._read_mapped_elsewhere()
        for at in ended:
            if self._mapped_got(at) is None and at in self._mapped_failed:
                self.log(f"research queue: {at[:8]}'s own change isn't mapped across the moves `main` landed since: "
                         f"{self._mapped_failed[at]}")
        wanted, self._mapped_wanted = self._mapped_wanted, {}
        for at, (old, onto, moves) in wanted.items():
            if len(running) >= self.mapped_jobs:
                break
            mark = drop / f"{mapped_name(at)}{MAPPING_MARK}"
            if mark.name in running or self._mapped_got(at) is not None or at in self._mapped_failed:
                continue
            self._start_mapping(mark, at, old, onto, moves, now)
            running.add(mark.name)
            self.log(f"research queue: mapping {old[:8]} across the {len(moves)} move(s) `main` landed since, in a detached "
                     f"job ({len(running)} of {self.mapped_jobs} running)")
        return len(ended)

    def _start_mapping(self, mark: Path, at: str, old: str, onto: str, moves: list[tuple[str, list[str]]], now: float) -> None:
        from .landing import detach

        mark.parent.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, "-c", MAP_JOB, str(Path(__file__).resolve().parents[1]), str(self.workdir), old, onto,
                json.dumps(moves), str(mark.with_suffix(MAPPED_SUFFIX)), str(MERGED_ROOT or "")]
        env = tree_env(_script_env())
        job = {**detach(self.workdir, argv, env), "out": str(mark.parent), "at": at, "until": now + _mapped_lost(moves)}
        tmp = mark.with_name(f"{mark.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(job))
        tmp.replace(mark)

    def _moves_crossed(self, onto: str, old: str) -> list[tuple[str, list[str]]]:
        if (key := (onto, old)) not in self._crossed:
            self._crossed[key] = moves_between(self.workdir, onto, old)
        return self._crossed[key]

    def _kept_mapped(self) -> dict[str, str]:
        if self._mapped_kept is None:
            if self._mapped_at is None:
                self._mapped_at = self.workdir / _out(self.workdir, "rev-parse", "--git-common-dir") / "mapped-changes.json"
            self._mapped_kept = json.loads(self._mapped_at.read_text()) if self._mapped_at.is_file() else {}
            self._read_mapped_elsewhere()
        return self._mapped_kept

    def _read_mapped_elsewhere(self) -> None:
        files = sorted(self._mapped_at.with_suffix(".d").glob(f"*{MAPPED_SUFFIX}"))
        if (seen := tuple((p.name, (s := p.stat()).st_mtime_ns, s.st_size) for p in files)) == self._mapped_seen:
            return
        self._mapped_seen, got, refused = seen, {}, {}
        for p in files:
            try:
                doc = dict(json.loads(p.read_text()))
            except (OSError, ValueError, TypeError) as e:
                self.log(f"research queue: {p} isn't read: {type(e).__name__}: {e}")
                continue
            got.update({at: v for at, v in doc.items() if isinstance(v, str)})
            refused.update({at: str(v["refused"]) for at, v in doc.items() if isinstance(v, dict) and "refused" in v})
        self._mapped_elsewhere = got
        self._mapped_failed.update(refused)

    def _mapped_got(self, at: str) -> str | None:
        return self._kept_mapped().get(at) or self._mapped_elsewhere.get(at)

    def _mapped_one(self, old: str, onto: str, moves: list[tuple[str, list[str]]]) -> tuple[str | None, str]:
        return map_one(self.workdir, old, onto, moves)

    def _mapped_done(self, at: str, got: str | None, why: str) -> None:
        if got is None:
            self._mapped_failed[at] = why or UNMAPPED
            self.log(f"research queue: {at[:8]}'s own change isn't mapped across the moves `main` landed since: "
                     f"{self._mapped_failed[at]}")
            return
        kept = self._kept_mapped()
        kept[at] = got
        self._mapped_kept = kept = dict(list(kept.items())[-OWN_KEEP:])
        tmp = self._mapped_at.with_name(f"{self._mapped_at.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(kept))
        tmp.replace(self._mapped_at)

    def _marked_again(self, p: PR, ejected: dict) -> bool:
        if self.marked_at is None or not self.ready(p.number, p.head):
            return False
        when, at = self.marked_at(p.number, p.head), ejected.get("at", "")
        if len(at) == len("2026-10-12T21:09Z"):
            at = at[:-1] + ":59Z"
        return when is not None and when > at

    def entries(self, prs: list[PR], state: dict, base: str | None = None, landed: str | None = None,
                queued: str | None = None, open_numbers: Collection[int] | None = None,
                old_landed: str | None = None) -> dict[int, Entry]:
        rules, base = self.rules(), base or self.base
        landed = landed or self.base
        heads = {p.head: p.number for p in prs}
        out: dict[int, Entry] = {}
        below: dict[int, set[str]] = {}
        order: dict[int, list[str]] = {}
        line: dict[int, set[str]] = {}
        if self.prefetch is not None:
            self.prefetch([(p.number, p.head) for p in prs if rules.hold_label not in p.labels and p.owner is not None])
        reach = above(self.workdir, base, (p.head for p in prs))
        old = {} if open_numbers is None else old_next(self.workdir, (p.head for p in prs if reach[p.head][0]), landed,
                                                       open_numbers)
        for p in prs:
            e = out[p.number] = Entry(p)
            ejected = state["ejected"].get(str(p.number))
            if why := p.outside(self.slug):
                e.state, e.why = "refused", why
            elif carried := [old[c] for c in reach[p.head][0] if c in old]:
                e.state, e.why = "refused", old_next_refusal(carried, self.into)
            elif rules.hold_label in p.labels:
                e.state, e.why = "waiting", f"held (`{rules.hold_label}` label)"
            elif p.owner is None:
                e.state, e.why = "waiting", "no owner: add a `Lane {name}` line to the PR body"
            elif ejected and ejected["head"] == p.head and not self._marked_again(p, ejected):
                e.state, e.why = "refused", f"ejected at {p.head[:8]}: {ejected['why']} (a ready mark re-admits it)"
            elif self.ready(p.number, p.head):
                pass
            elif self.before_cutoff and (when := self.before_cutoff(p.number, p.head)):
                e.state, e.why = "waiting", (f"ignored: marked before cutoff (its ready mark at {p.head[:8]} is from {when}, "
                                             f"before `marked_after`): re-mark it or close it")
            else:
                e.state, e.why = "waiting", f"not marked ready at {p.head[:8]}: `research queue ready {p.number} --on MACHINE` (or --local; --mark without the tier)"
            order[p.number], line[p.number] = reach[p.head]
            below[p.number] = set(order[p.number])
            e.held = {heads[h]: h for h in order[p.number] if h in heads and heads[h] != p.number}
            e.after = tuple(sorted(e.held))
        members = chain_of(self.workdir, landed, queued or f"refs/remotes/origin/{self.branch}")
        pos = {m.pr: i for i, m in enumerate(members) if m.pr in out and m.head == out[m.pr].pr.head}
        parents = {c: ps for c, *ps in map(str.split, _out(self.workdir, "rev-list", "--parents", f"{landed}..{members[-1].commit}")
                                           .splitlines())} if pos else {}
        first = {n: (e.state, e.why, e.after, e.held) for n, e in out.items()}
        on = functools.cache(lambda c: landed != base and not _git(self.workdir, "merge-base", "--is-ancestor", c, landed,
                                                                   check=False).returncode)
        on_old = functools.cache(lambda c: bool(old_landed) and not _git(self.workdir, "merge-base", "--is-ancestor", c,
                                                                          old_landed, check=False).returncode)
        seen = state.get("seen") or {}
        carriers = set(pos)
        while True:
            brought = _brought(parents, sorted((pos[n], members[pos[n]].head) for n in carriers))
            for n, e in out.items():
                e.state, e.why, e.after, e.held = *first[n][:3], dict(first[n][3])
            own = {n: {c for c in below[n].difference(*(below[a] for a in e.after)) if brought.get(c, math.inf) >= pos.get(n, math.inf)}
                   for n, e in out.items()}
            branch: dict[str, list[int]] = {}
            for n, cs in own.items():
                for c in cs & line[n]:
                    branch.setdefault(c, []).append(n)
            for n, e in out.items():
                older: dict[int, str] = {}
                for c in (c for c in order[n] if c in own[n]):
                    at = [q for q in seen.get(c, ()) if q in out and c in below[q]]
                    for q in [*([q for q in at if out[q].state == "ready"] or at), *(branch.get(c, ()) if c not in line[n] else ())]:
                        if q != n:
                            older.setdefault(q, c)
                e.older = tuple(sorted(older))
                e.after = tuple(sorted(e.after + e.older))
                e.held |= older
            for e in out.values():
                e.after = tuple(a for a in e.after if not on(e.held[a]) and not self._carried(e.pr, out[a], state))
                e.older = tuple(a for a in e.older if a in e.after)
                e.held = {a: e.held[a] for a in e.after}
            changed = True
            while changed:
                changed = False
                for e in out.values():
                    if e.state == "ready" and (blocking := [a for a in e.after
                                                            if out[a].state != "ready" and not on_old(e.held[a])]):
                        e.state, e.why, changed = "waiting", f"contains {e.contains(blocking[0])}, which isn't admitted", True
            if not (fell := {n for n in carriers if out[n].state != "ready"}):
                return out
            carriers -= fell

    @staticmethod
    def _seen(state: dict, prs: list[PR], open_numbers: set[int]) -> None:
        seen = state["seen"] = {h: kept for h, ns in (state.get("seen") or {}).items()
                                if (kept := [n for n in ns if n in open_numbers])}
        new: dict[str, list[int]] = {}
        for p in prs:
            if p.head not in seen:
                new.setdefault(p.head, []).append(p.number)
        seen.update(new)

    def _carried(self, p: PR, a: Entry, state: dict) -> bool:
        if a.state == "ready" or a.pr.owner is None or self.rules().hold_label in a.pr.labels:
            return False
        who = _who(a.pr.owner)
        ejected = state["ejected"].get(str(a.pr.number)) or {}
        return (_who(p.owner or "") == who
                or (ejected.get("head") == a.pr.head and _who(ejected.get("by") or "") == who)
                or (self.marked_by is not None and not self.ready(a.pr.number, a.pr.head)
                    and not (self.before_cutoff and self.before_cutoff(a.pr.number, a.pr.head))
                    and _who(self.marked_by(a.pr.number, a.pr.head) or "") == who))

    def _stale_carry(self, pr: PR, entries: dict[int, Entry]) -> str | None:
        if not (carried := [c for c in carried_heads(self.workdir, self.base, pr.head) if c[0] != pr.number]):
            return None
        def why(n: int) -> str | None:
            if (e := entries[n]).state == "ready":
                return None
            return f"not marked ready at {e.pr.head[:8]}" if e.why.startswith("not marked ready") else e.why

        heads = {n: e.pr.head for n, e in entries.items()}
        return "; ".join(stale_carried(self.workdir, carried, heads, why, self.into).values()) or None

    def order(self, chain: list[Member], entries: dict[int, Entry], lift: Iterable[int] = (),
              park: Iterable[int] = ()) -> list[int]:
        ready = {n for n, e in entries.items() if e.state == "ready"}
        stays = {m.pr for m in chain if self._pending(m, entries)}
        kept = [m.pr for m in chain if m.pr in ready and self._same_or_fixed(m, entries) or m.pr in stays]
        at = {m.pr: m.head for m in chain}
        parked = {n for n in park if n in kept and (entries[n].pr.head == at[n] or n in stays)}
        grown = bool(parked)
        while grown:
            grown = False
            for n in ready.union(kept) - parked:
                if parked.intersection(entries[n].after):
                    parked.add(n)
                    grown = True
        out: list[int] = []
        placing: set[int] = set()

        def place(n: int) -> None:
            if n in out or n in placing:
                return
            placing.add(n)
            for a in entries[n].after:
                place(a)
            out.append(n)

        new = sorted(ready - set(kept))
        cold = {n for n in new if self._forces_cold(n)}
        self._lifted = []
        first = [n for n in self.priority if n in ready]
        below = set(first)
        for n in lift:
            if (n in kept and n not in parked and n not in below and entries[n].pr.head == at.get(n)
                    and all(a in self._lifted or a in below for a in entries[n].after)):
                self._lifted.append(n)
        self._parked = [n for n in kept if n in parked] + sorted(parked - set(kept))
        ahead = [n for n in new if n in self._prioritized]
        for n in (first + self._lifted + kept + ahead
                  + [n for n in new if n not in cold and n not in ahead] + sorted(cold.difference(ahead))):
            if n not in parked:
                place(n)
        for n in self._parked:
            place(n)
        return out

    def flag(self, name: str) -> bool:
        own = getattr(self, name)
        return own if own is not None else getattr(flags_at(self.workdir, "refs/remotes/origin/main"), name)

    def _same_or_fixed(self, m: Member, entries: dict[int, Entry]) -> bool:
        new = entries[m.pr].pr.head
        return new == m.head or not _git(self.workdir, "merge-base", "--is-ancestor", m.head, new, check=False).returncode

    def _pending(self, m: Member, entries: dict[int, Entry]) -> bool:
        e = entries.get(m.pr)
        if e is None or e.pr.head == m.head or e.state != "waiting" or not e.why.startswith("not marked ready"):
            return False
        return bool(self.ready(m.pr, m.head)) and self._same_or_fixed(m, entries)

    def _forces_cold(self, n: int) -> bool:
        r = _git(self.workdir, "diff", "--name-only", f"refs/remotes/origin/main...refs/queue/pr/{n}", check=False)
        return r.returncode == 0 and forces_cold(r.stdout.split())

    def _merge(self, tip: str, pr: PR, members: list[Member], head: str | None = None) -> tuple[str | None, str]:
        head = head or pr.head
        rec = self._recorded(tip, head)
        if "conflict" in rec:
            return None, self._unmerged("", rec["conflict"], members)
        if "err" in rec:
            return None, self._unmerged(rec["err"], [], members)
        if rec.get("pins"):
            self.log(f"research queue: #{pr.number} conflicts only in pins: {', '.join(rec['pins'])} rebuilt from the merged tree")
        if rec.get("gone"):
            self.log(f"research queue: #{pr.number} deletes {', '.join(rec['gone'])}, which `{self.branch}` changed only by "
                     "porting it to the module system: deleted in the merge")
        if rec.get("tip_gone"):
            self.log(f"research queue: `{self.branch}` deletes {', '.join(rec['tip_gone'])}, which #{pr.number} changed only by "
                     "porting it to the module system: deleted in the merge")
        if rec.get("merged"):
            return tip, ""
        if "why" in rec:
            return None, rec["why"]
        if files := rec.get("ported"):
            pins = rec.get("ported_pins")
            self.log(f"research queue: #{pr.number}'s merged tree holds Lean made before the module system: {', '.join(files)} "
                     f"ported by `{MODULES}`" + (f", and the pins {', '.join(pins)} rebuilt" if pins else ""))
        commit = _git(self.workdir, "commit-tree", rec["tree"], "-p", tip, "-p", head, "-m", _message(pr),
                      env=self.identity).stdout.strip()
        self._merged_from[commit] = (self._merge_key(tip, head), head)
        return commit, ""

    def _recorded(self, tip: str, head: str) -> dict:
        key = self._merge_key(tip, head)
        rec = self._merge_records.get(key)
        if rec is not None and "tree" in rec and not self._has_tree(rec["tree"]):
            rec = None
        if rec is None:
            rec = self._merged(tip, head)
            if "err" not in rec and "why" not in rec:
                self._merge_records[key] = rec
        else:
            self._merges_reused += 1
        rec["at"] = self._merge_serial
        return rec

    def _keep_merges(self, state: dict) -> None:
        had = state.get("merges") or {}
        merged = {**(had.get("of") or {}), **self._merge_records}
        self._merge_records = dict(sorted(merged.items(), key=lambda kv: -kv[1].get("at", 0))[:MERGES])
        state["merges"] = {"serial": max(self._merge_serial, int(had.get("serial") or 0)), "of": self._merge_records}

    def _merged(self, tip: str, head: str) -> dict:
        tree, files, r = merged_tree(self.workdir, tip, head, plain=False)
        if r.returncode == 129 and "attr-source" in r.stderr:
            raise QueueError(f"this git has no `--attr-source`, which merging `{self.branch}` needs: install git >= 2.40")
        if tree is None:
            return {"conflict": files} if files else {"err": (r.stderr or r.stdout).strip()[-200:]}
        rec: dict = {}
        if files:
            rec.update(zip(("pins", "gone", "tip_gone"), taken(self.workdir, tree, head, files)))
        elif not _git(self.workdir, "merge-base", "--is-ancestor", head, tip, check=False).returncode:
            return {"merged": True}
        tree, files, pins, why = ported(self.workdir, tree, base=tip)
        if tree is None:
            return {**rec, "why": why}
        return {k: v for k, v in {**rec, "tree": tree, "ported": files, "ported_pins": pins}.items() if v}

    def _merge_key(self, tip: str, head: str) -> str:
        if (key := self._merge_keys.get((tip, head))) is None:
            tree = _out(self.workdir, "rev-parse", f"{tip}^{{tree}}")
            bases = _git(self.workdir, "merge-base", "--all", tip, head, check=False).stdout.split()
            key = hashlib.sha256("\0".join([self._merge_context(), tree, head, *bases]).encode()).hexdigest()
            self._merge_keys[(tip, head)] = key
        return key

    def _merge_context(self) -> str:
        if self._context is None:
            from . import clone, gates

            config = [ln for ln in _git(self.workdir, "config", "--list", check=False).stdout.splitlines()
                      if not ln.startswith(("remote.", "url.", "http.", "credential.", "user.", "branch."))]
            drivers = subprocess.run(["python3", "-c", "import sys; print(sys.version)"], capture_output=True, text=True,
                                     env=git_env()).stdout
            parts = [*(Path(f).read_text() for f in (__file__, clone.__file__, gates.__file__)),
                     json.dumps(clone.RECORD_DRIVER, sort_keys=True), _out(self.workdir, "version"), sys.version, drivers, *config]
            self._context = hashlib.sha256("\0".join(parts).encode()).hexdigest()
        return self._context

    def _has_tree(self, tree: str) -> bool:
        return not _git(self.workdir, "cat-file", "-e", f"{tree}^{{tree}}", check=False).returncode

    def _lock_drops(self, tip: str, head: str, commit: str) -> list[str]:
        key, merged = self._merged_from.get(commit, ("", ""))
        if merged != head or (rec := self._merge_records.get(key)) is None:
            return lock_drops(self.workdir, tip, head, commit)
        if "drops" not in rec:
            rec["drops"] = lock_drops(self.workdir, tip, head, commit)
        return list(rec["drops"])

    @staticmethod
    def _move_key(tip: str, pr: PR, main: str) -> str:
        return f"{tip}:{main}:{pr.move}:{pr.number}:{pr.head}"

    def _regenerate(self, tip: str, pr: PR, main: str) -> tuple[str | None, str]:
        key = self._move_key(tip, pr, main)
        self._used.add(key)
        got = self._made.get(key) or {}
        if not got.get("why") and not (got.get("commit") and self._has(got["commit"])):
            commit, why, report = self._regenerated(tip, pr, main)
            got = self._made[key] = {"commit": commit, "certificate": report} if commit else {"why": why}
        return got.get("commit"), got.get("why", "")

    def _scripts_below(self, tip: str, main: str) -> list[str]:
        scripts, ours = move_scripts(self.workdir, main), move_scripts(self.workdir, tip)
        return sorted(p for p in ours.keys() | scripts.keys() if p.startswith(MOVE_DIR) and ours.get(p) != scripts.get(p))

    def _regenerated(self, tip: str, pr: PR, main: str) -> tuple[str | None, str, dict | None]:
        scripts, into = move_scripts(self.workdir, main), self.into
        missing = [f"{p} at `{into}` {main[:12]}" for p in (RENAME, MOVE_CHECK) if p not in scripts]
        if _git(self.workdir, "cat-file", "-e", f"{tip}:{pr.move}", check=False).returncode:
            missing.insert(0, f"{pr.move} at {tip[:12]}")
        if missing:
            return None, (f"its move can't be regenerated: there's no {' or '.join(missing)}; land the map first, then mark the "
                          "move again"), None
        if changed := self._scripts_below(tip, main):
            return None, (f"the members below it change {', '.join(changed)} from `{into}` {main[:12]}: its move runs "
                          f"`{into}`'s scripts, and the members above it are carried across it with its base's, so it waits "
                          f"until those members land on `{into}`"), None
        try:
            rows = path_rows(_out(self.workdir, "show", f"{tip}:{pr.move}"))
        except tomllib.TOMLDecodeError as e:
            return None, f"its map {pr.move} at {tip[:12]} doesn't parse ({e}): fix the map, then mark the move again", None
        except ValueError as e:
            return None, f"its map {pr.move} at {tip[:12]} {e}: fix the map, then mark the move again", None
        with scratch() as tmp:
            wt = Path(tmp, "wt")
            with worktrees_locked(self.workdir):
                _git(self.workdir, "worktree", "add", "-q", "--detach", str(wt), tip)
            try:
                rename = write_scripts(self.workdir, scripts, Path(tmp, "rename")) / RENAME
                r = subprocess.run([sys.executable, str(rename), "--map", pr.move, "--root", str(wt)], cwd=wt,
                                   capture_output=True, text=True, env=_script_env())
                if r.returncode:
                    return None, f"`{RENAME} --map {pr.move}` failed on {tip[:12]} (exit {r.returncode}): {_tail(r)}", None
                _git(wt, "add", "-A")
                tree = _out(wt, "write-tree")
                if tree == _out(self.workdir, "rev-parse", f"{tip}^{{tree}}"):
                    return None, (f"`{RENAME} --map {pr.move}` changes nothing on {tip[:12]}: the move is in it already, or "
                                  "its map moves nothing there"), None
                commit = _git(self.workdir, "commit-tree", tree, "-p", tip, "-m", _move_message(pr, tip, into, main),
                              env=self.identity).stdout.strip()
                check = write_scripts(self.workdir, scripts, Path(tmp, "check")) / MOVE_CHECK
                c = subprocess.run([sys.executable, str(check), tip, commit, *(f"--map={o}={n}" for o, n in rows), "--json"],
                                   cwd=wt, capture_output=True, text=True, env=_script_env())
                said = f"`{MOVE_CHECK} {tip[:12]} {commit[:12]}`" + (f" with {len(rows)} --map rows" if rows else "")
                judge = f"{said} couldn't judge its move regenerated on {tip[:12]} (exit {c.returncode})"
                try:
                    report = json.loads(c.stdout)
                except ValueError as e:
                    return None, f"{judge}: no JSON report ({e}): {_tail(c, 4)}", None
                if not isinstance(report, dict):
                    return None, f"{judge}: {_tail(c, 4)}", None
                if c.returncode or report.get("certified") is not True:
                    refused = "; ".join(f"{f.get('path')}: {f.get('reason')}" for f in (report.get("refusals") or [])[:3])
                    return None, (f"its move regenerated on {tip[:12]} as {commit[:12]} isn't certified ({said} exit "
                                  f"{c.returncode}): {_tail(c, 1)}" + (f": {refused}" if refused else "")), None
                self.log(f"research queue: #{pr.number}'s move ({pr.move}) regenerated on {tip[:12]} as {commit[:12]}: "
                         f"{_tail(c, 1)}")
                return commit, "", report
            finally:
                with worktrees_locked(self.workdir):
                    _git(self.workdir, "worktree", "remove", "--force", str(wt), check=False)
                    _git(self.workdir, "worktree", "prune", check=False)

    def _merge_through(self, tip: str, pr: PR, members: list[Member]) -> tuple[str | None, str]:
        if not (pending := [mv for mv in self._moves if not self._contains(mv.commit, pr.head)]):
            return self._merge(tip, pr, members)
        if not any(self._sensitive(pr.head, mv) for mv in pending):
            commit, why = self._merge(tip, pr, members)
            if commit is not None:
                return commit, why
        head, names = pr.head, ", ".join(f"{mv.commit[:12]} ({mv.map})" for mv in pending)
        for mv in pending:
            head, why = self._restack(head, mv, pr.head)
            if head is None:
                return None, f"it doesn't cross the move {mv.commit[:12]} ({mv.map}): {why}"
        commit, why = self._merge(tip, pr, members, head)
        if commit is None:
            return None, f"{why}, restacked through the move {names} as {head[:12]}"
        self.log(f"research queue: #{pr.number} at {pr.head[:8]} restacked through the move {names} as {head[:12]}")
        return commit, why

    def _sensitive(self, head: str, mv: Move) -> bool:
        if mv.map not in self._words_of:
            got = map_retires(self.workdir, f"{mv.commit}^1", mv.map)
            self._words_of[mv.map] = set(got.words) if got else set()
        r = _git(self.workdir, "diff", "-U0", "--no-color", "--no-ext-diff", f"{mv.commit}^1...{head}", check=False)
        low = r.stdout.lower()
        return bool(r.returncode) or any(w in low for w in self._words_of[mv.map])

    def _restack(self, head: str, mv: Move, of: str) -> tuple[str | None, str]:
        key = f"{head}:{mv.commit}"
        self._used.add(key)
        got = self._restacks.get(key)
        if got is None or (got.get("result") and not self._has(got["result"])):
            self._restacks.pop(key, None)
            if key not in self._restacking and self._launch_restack(head, mv, of):
                before = self._moves[:self._moves.index(mv)] if mv in self._moves else []
                for h in dict.fromkeys(self._upcoming):
                    k = f"{h}:{mv.commit}"
                    if h == head or k in self._restacks or k in self._restacking or self._contains(mv.commit, h) \
                            or not all(self._contains(b.commit, h) for b in before) or not self._sensitive(h, mv):
                        continue
                    if not self._launch_restack(h, mv, h):
                        break
            if (got := self._restacks.get(key)) is None:
                raise Restacking(self._restacking_why(key, mv))
        return got.get("result"), got.get("why", "")

    def _launch_restack(self, head: str, mv: Move, of: str, tries: int = 1, since: float | None = None) -> bool:
        key = f"{head}:{mv.commit}"
        if not self._restack_starts or len(self._restacking) >= self.restack_jobs:
            return False
        files = restack_files(self.workdir, self.base)
        if RESTACK not in files:
            self._restacks[key] = {"why": f"`{self.into}` has no {RESTACK}"}
            return False
        from .landing import LAUNCHES, detach

        root = Path(tempfile.gettempdir()) / LAUNCHES
        root.mkdir(parents=True, exist_ok=True)
        out = tempfile.mkdtemp(prefix="restack-", dir=root)
        path = write_scripts(self.workdir, files, Path(out)) / RESTACK
        argv = ["sh", "-c", RESTACK_JOB, "sh", out, sys.executable, str(path), head, "--move-commit", mv.commit, "--script", RENAME,
                "--script-arg=--map", f"--script-arg={mv.map}", "--main", "refs/remotes/origin/main", "--json"]
        now = self.now()
        self._restacking[key] = {**detach(self.workdir, argv, _script_env()), "out": out, "of": of, "map": mv.map, "at": now,
                                 "since": now if since is None else since, "tries": tries}
        self._save_restacks()
        self.log(f"research queue: restacking {head[:12]} across the move {mv.commit[:12]} ({mv.map}) in the background"
                 + (f", try {tries}" if tries > 1 else ""))
        return True

    def _save_restacks(self) -> None:
        self.save_state({**self.load_state(), "restacks": self._restacks, "restacking": self._restacking})

    def _restacking_why(self, key: str, mv: Move) -> str:
        from .landing import _hhmm

        what = f"{RESTACKING} {mv.commit[:12]} ({mv.map})"
        if job := self._restacking.get(key):
            return f"{what} since {_hhmm(job['since'])}"
        if not self._restack_starts:
            return f"{what}: a dry run starts no restack"
        return f"{what} once one of the {len(self._restacking)} restacks running ends"

    def _collect_restacks(self, wanted: set[str]) -> None:
        ended = False
        for key, job in list(self._restacking.items()):
            if (end := restack_end(job, self.now())) is None:
                continue
            ended = True
            del self._restacking[key]
            (head, move), (said, how, interrupted) = key.split(":"), end
            if job["of"] not in wanted or not self._has(head):
                self.log(f"research queue: the restack of {head[:12]} across the move {move[:12]} is dropped: "
                         + (f"its PR's head {job['of'][:8]} changed" if job["of"] not in wanted else "the clone lost it"))
            elif said is not None:
                self._restacks[key] = said
            elif interrupted:
                self.log(f"research queue: the restack of {head[:12]} across the move {move[:12]} was interrupted ({how}); "
                         "again, not counted as a try")
                self._launch_restack(head, Move(move, job["map"]), job["of"], job["tries"], job["since"])
            elif job["tries"] < 2:
                self.log(f"research queue: the restack of {head[:12]} across the move {move[:12]} ended with no result ({how}); "
                         "once more")
                self._launch_restack(head, Move(move, job["map"]), job["of"], job["tries"] + 1, job["since"])
            else:
                self._restacks[key] = {"why": f"{RESTACK} failed ({how})"}
        if ended:
            self._save_restacks()

    def _has(self, commit: str) -> bool:
        return not _git(self.workdir, "cat-file", "-e", f"{commit}^{{commit}}", check=False).returncode

    def _contains(self, ancestor: str, commit: str) -> bool:
        return not _git(self.workdir, "merge-base", "--is-ancestor", ancestor, commit, check=False).returncode

    def _gated(self, main: str, gates: tuple[str, ...], tip: str, commit: str, pr: PR, before: dict[str, list[str]],
               verdicts: dict[str, list[str]], kind: str = "landing gate") -> tuple[list[str], list[str]]:
        trees = _out(self.workdir, "rev-parse", f"{tip}^{{tree}}", f"{commit}^{{tree}}").split()
        key = hashlib.sha256("\0".join([*trees, _message(pr)]).encode()).hexdigest()
        if key in before:
            verdicts[key] = before[key]
            return list(before[key]), []
        refused, unjudged = gate_refusal(self.workdir, main, commit, gates, onto=self.branch, kind=kind)
        if not unjudged:
            verdicts[key] = refused
        return refused, unjudged

    def _read_priority(self, state: dict) -> tuple[int, ...]:
        try:
            got = priority_list(self.priority_file)
        except QueueError as e:
            if state.get("priority_error") != str(e):
                self.log(f"research queue: {e}; drive's priority list is empty until the file is fixed")
            state["priority_error"], state["priority"] = str(e), []
            return ()
        state.pop("priority_error", None)
        if list(got) != state.get("priority", []):
            self.log(f"research queue: drive's priority list ({self.priority_file}), first to land first: "
                     + (" ".join(f"#{n}" for n in got) or "empty"))
            state["priority"] = list(got)
        return got

    def _unmerged(self, err: str, files: list[str], members: list[Member]) -> str:
        if not files:
            return f"doesn't merge onto `{self.branch}`: {err}"
        by = touched_by(self.workdir, members, files, self._touched)
        return f"conflicts with `{self.branch}` in {', '.join(files)}" + (f" (touched by {', '.join(by)})" if by else "")

    def handle(self, pr: PR) -> str | None:
        if (got := self._read_registry()) is None:
            return None
        names, holders, _ = got
        by = self.marked_by(pr.number, pr.head) if self.marked_by else None
        suffix = pr.branch.rpartition("-")[2] if "-" in pr.branch else ""
        tail = sorted({n for a, n in holders.items() if len(suffix) == 4 and a.endswith(suffix)})
        for c in (pr.owner, holders.get(pr.owner or ""), by, tail[0] if len(tail) == 1 else None):
            if c in names:
                return c
        return None

    def _read_registry(self) -> tuple[set, dict[str, str], set[str]] | None:
        if self._registry is None:
            r = _git(self.workdir, "show", f"origin/main:{REGISTRY}", check=False)
            try:
                reg = json.loads(r.stdout).get("handles", []) if r.returncode == 0 else []
            except (ValueError, AttributeError):
                return None
            self._registry = ({h.get("handle") for h in reg if isinstance(h, dict)},
                              {a: h["handle"] for h in reg if isinstance(h, dict) and "handle" in h
                               for a in AGENT_RE.findall(str(h.get("holder", "")))},
                              {h["linear"] for h in reg if isinstance(h, dict) and isinstance(h.get("linear"), str)})
        return self._registry

    def issue_teams(self) -> set[str]:
        return got[2] if (got := self._read_registry()) else set()

    def sync(self, dry_run: bool = False) -> tuple[list[Member], dict[int, Entry]]:
        """Off drive it refuses before anything."""
        self._drive_only(self.branch, "queue sync", dry_run)
        self._clone()
        state = self.load_state()
        if self.priority_file is not None:
            self.priority = self._read_priority(state)
        open_prs = self.github.open_prs()
        prs, waiting = self.fetch_prs([p for p in open_prs if p.base == "main"])
        landed, self._moves = queue_log(self.workdir, self.base)
        prs = [p for p in prs if p.head not in landed
               and _git(self.workdir, "merge-base", "--is-ancestor", p.head, self.base, check=False).returncode]
        self._made, self._restacks, self._used = state.get("moves_made") or {}, state.get("restacks") or {}, set()
        self._restacking, self._restack_starts = state.get("restacking") or {}, not dry_run
        merges = state.get("merges") or {}
        self._merge_records = {**(merges.get("of") or {}), **self._merge_records}
        self._merge_serial, self._merges_reused = int(merges.get("serial") or 0) + 1, 0
        self._merge_keys, self._context, self._merged_from = {}, None, {}
        self._seen(state, prs, {p.number for p in open_prs})
        chain = self.chain()
        entries = self.entries(prs, state, open_numbers={p.number for p in open_prs})
        entries.update(waiting)
        listed = [n for n in self.priority if n in entries]
        self._prioritized = set(listed).union(*(entries[n].after for n in listed))
        pending = {m.pr: m.head for m in chain if self._pending(m, entries)}
        on = {m.pr for m in chain}
        out = {int(n) for n, c in state.get("conflicts", {}).items()
               if int(n) in entries and int(n) not in on and c["head"] == entries[int(n)].pr.head}
        out |= {n for n, e in entries.items() if n not in on and out.intersection(e.after)}

        def last(order: list[int]) -> list[int]:
            return [n for n in order if n not in out] + [n for n in order if n in out]

        main = _out(self.workdir, "rev-parse", self.base)
        order = last(self.order(chain, entries, self.lift, self.park))
        heads = {n: pending.get(n, entries[n].pr.head) for n in order}

        def kept(order: list[int]) -> int:
            want = [(n, heads[n]) for n in order]
            return next((i for i, (m, w) in enumerate(zip(chain, want)) if (m.pr, m.head) != w), min(len(chain), len(want)))

        k = kept(order)
        if k < max((i + 1 for i, n in enumerate(order) if n in self._lifted), default=0):
            self.log(f"research queue: {' '.join(f'#{n}' for n in self._lifted)} (reviewed) go to the bottom of `{self.branch}`, "
                     "below what waits for a review")
        parked = set(self._parked)
        low = next((i for i, m in enumerate(chain) if m.pr in parked), len(chain))
        if any(m.pr not in parked and m.pr in heads for m in chain[low:]):
            self.log(f"research queue: {' '.join(f'#{n}' for n in self._parked)} (held, or containing a held PR) go to the top "
                     f"of `{self.branch}`, so what was above them lands")
        members = chain[:k]
        tip = members[-1].commit if members else main
        dropped: set[int] = set()
        conflicts: dict[str, dict] = {}
        waits: dict[str, dict] = {}
        gates = gates_at(self.workdir, main) if order[k:] else ()
        warns = gates_at(self.workdir, main, "warn_gates") if order[k:] else ()
        before, verdicts = kept_verdicts(state, "gate_verdicts", main, gates), {}
        warn_before, warn_verdicts = kept_verdicts(state, "warn_verdicts", main, warns), {}
        open_heads = {p.number: p.head for p in open_prs}
        warned = {n: w for n, w in (state.get("warned") or {}).items() if open_heads.get(int(n)) == w["head"]}
        merged = False
        if tip != main:
            self._moves = queue_log(self.workdir, tip)[1]
        self._upcoming = [heads[n] for n in order[k:] if not entries[n].pr.move]
        if not dry_run:
            self._collect_restacks({*heads.values(), *open_heads.values()})

        def refuse(n: int, why: str) -> None:
            e = entries[n]
            e.state, e.why = "refused", why
            dropped.add(n)
            conflicts[str(n)] = {"head": e.pr.head, "why": why, "handle": self.handle(e.pr)}

        def wait(n: int, why: str) -> None:
            e = entries[n]
            e.state, e.why = "waiting", why
            dropped.add(n)
            waits[str(n)] = {"head": e.pr.head, "why": why, "handle": self.handle(e.pr)}
            if state.get("waiting", {}).get(str(n), {}).get("head") != e.pr.head:
                self.log(f"research queue: #{n} at {e.pr.head[:8]} waits off `{self.branch}`: {why}")

        for n in order[k:]:
            e = entries[n]
            if why := self._stale_carry(e.pr, entries):
                refuse(n, why)
                continue
            if blocked := [a for a in e.after if a in dropped]:
                wait(n, f"contains {e.contains(blocked[0])}, which didn't merge")
                continue
            if e.pr.move:
                commit, why = self._regenerate(tip, e.pr, main)
            else:
                try:
                    commit, why = self._merge_through(tip, e.pr, members)
                except Restacking as r:
                    wait(n, str(r))
                    continue
            merged = True
            if not e.pr.move and commit not in (None, tip) and (drops := self._lock_drops(tip, e.pr.head, commit)):
                commit, why = None, lock_drop_refusal(drops, self.branch)
            if commit is not None and gates:
                refused, unjudged = self._gated(main, gates, tip, commit, e.pr, before, verdicts)
                if unjudged and not refused:
                    wait(n, f"a landing gate can't judge it yet: {' | '.join(unjudged)}; retried next sync")
                    continue
                if refused or unjudged:
                    commit, why = None, " | ".join(refused + unjudged)
            if commit is None:
                refuse(n, why)
                continue
            if warns:
                said, unjudged = self._gated(main, warns, tip, commit, e.pr, warn_before, warn_verdicts, kind="warn gate")
                if said:
                    warned[str(n)] = {"head": e.pr.head, "why": " | ".join(said), "handle": self.handle(e.pr)}
                if unjudged:
                    self.log(f"research queue: #{n} at {e.pr.head[:8]}: {' | '.join(unjudged)}; a warn gate keeps no PR out")
            tip = commit
            members.append(Member(n, e.pr.head, e.pr.owner or "", commit))
            if e.pr.move:
                self._moves.append(Move(commit, e.pr.move))
        if merged:
            _git(self.workdir, "checkout", "-q", "--detach", tip)
        for i, m in enumerate(members, 1):
            entries[m.pr].state, entries[m.pr].position, entries[m.pr].commit = "queued", i, m.commit
        old = _git(self.workdir, "rev-parse", "--verify", "-q", f"refs/remotes/origin/{self.branch}", check=False).stdout.strip()
        if tip != old and not dry_run:
            self._remote(self.workdir, "push", "-q", f"--force-with-lease=refs/heads/{self.branch}:{old}", "origin",
                         f"{tip}:refs/heads/{self.branch}")
        self.log(f"research queue: `{self.branch}` at {tip[:12]}: {len(members)} PR(s) above `{self.into}`"
                 + (f", kept {k}, merged {len(members) - k}" if members else "")
                 + (f", merges reused for {self._merges_reused}" if self._merges_reused else "")
                 + (f", gates' verdicts reused for {reused}" if (reused := len(verdicts.keys() & before.keys())
                                                                  + len(warn_verdicts.keys() & warn_before.keys())) else "")
                 + (" (dry run: not pushed)" if dry_run else ""))
        if not dry_run:
            state["conflicts"], state["waiting"] = conflicts, waits
            state["moves_made"] = {k: v for k, v in self._made.items() if k in self._used}
            moves = {mv.commit for mv in self._moves}
            state["restacks"] = {k: v for k, v in self._restacks.items() if k in self._used or k.rpartition(":")[2] in moves}
            state["restacking"] = self._restacking
            for key, gs, kept, new in (("gate_verdicts", gates, before, verdicts), ("warn_verdicts", warns, warn_before,
                                                                                   warn_verdicts)):
                if gs:
                    newest = {**{h: v for h, v in kept.items() if h not in new}, **new}
                    state[key] = {"main": main, "gates": list(gs), "of": dict(list(newest.items())[-GATE_VERDICTS:])}
            state["warned"] = warned
            self._keep_merges(state)
            self.save_state(state)
        return members, entries

    def _drive_only(self, ref: str, verb: str, dry_run: bool) -> None:
        from .merge import DRIVE_ONLY, off_drive, on_drive

        if not dry_run and ref in DRIVE_ONLY and not on_drive():
            raise QueueError(off_drive(ref, verb))

    def eject(self, pr: int, reason: str, by: str, sha: str | None = None) -> PR:
        self._clone()
        prs = {p.number: p for p in self.github.open_prs()}
        if pr not in prs:
            raise QueueError(f"#{pr} isn't an open PR")
        if sha and not prs[pr].head.startswith(sha):
            raise QueueError(f"#{pr} is at {prs[pr].head[:12]} now, not {sha[:12]}: its owner pushed since, so nothing is ejected; "
                             f"the new head is checked on its own")
        state = self.load_state()
        state["ejected"][str(pr)] = {"head": prs[pr].head, "why": reason, "by": by, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if self.store is not None and not self.shadow:
            from .store.cli import _label_write_through

            target = label_target(pr, prs[pr].head)
            self.store.add_label(target, vocab.READY, False, by, ref=reason)
            said, lost = _label_write_through(self.store, target)
            self.log(f"research queue: #{pr}'s ready mark at {prs[pr].head[:12]} is withdrawn ({target}); {said}")
            if lost:
                raise QueueError(f"#{pr}'s withdrawn mark didn't reach the store's remote, so other queues may still admit it: {said}")
        self.save_state(state)
        return prs[pr]

    def tell_ejected(self, p: PR, reason: str, by: str, post: Callable[..., bool] | None = None) -> int:
        from .landing import CI, _merge_thread, merge_thread_post

        owner = self.handle(p)
        to = tuple(dict.fromkeys((CI, *([f"@{owner}"] if owner else []))))
        text = (f"{' '.join(to)} queue: #{p.number} at {p.head[:8]} is ejected from `{self.branch}` by {by}: {reason}. "
                f"Fix it and re-mark it; the next sync rebuilds `{self.branch}` without it")
        if self.shadow:
            self.log(f"research queue: would tell {' '.join(to)}: {text} [shadow]")
            return 0
        if (post or merge_thread_post)(text, to, sender=by):
            self.log(f"research queue: told {' '.join(to)} as @{by}: #{p.number} ejected")
            return 0
        print(f"research queue: #{p.number} is ejected, but posting its notice as @{by} failed, so no one was told. Send it as "
              f"your own handle: research msg send --as HANDLE --thread {_merge_thread()} --to {' '.join(to)} --text {text!r}",
              file=sys.stderr)
        return 1

    def status(self, stale: Callable[[list[PR]], dict[int, float]] | None = None,
               fetched: Callable[[], Callable[[list[PR]], dict[int, float]] | None] | None = None,
               loop: bool = False, reviews: Callable[[int, str], list] | None = None) -> dict[int, Entry]:
        self._clone()
        state = self.load_state()
        open_prs = self.listed = self.github.open_prs()
        prs, waiting = self.fetch_prs([p for p in open_prs if p.base == "main"])
        self.unlock()
        if fetched is not None:
            stale = fetched()
        main = _out(self.workdir, "rev-parse", "refs/remotes/origin/main")
        prefetch = self.prefetch
        if prefetch is not None:
            prefetch([(p.number, p.head) for p in open_prs])
            self.prefetch = None
        try:
            old = _git(self.workdir, "rev-parse", "-q", "--verify", "refs/remotes/origin/checked^{commit}", check=False).stdout.strip()
            onto = {"base": main, "landed": main, "queued": main, "old_landed": old or None} if loop else {}
            entries = self.entries(prs, state, open_numbers={p.number for p in open_prs}, **onto)
            entries.update(waiting)
            entries.update(self.off_main(open_prs))
            idle = stale(open_prs) if stale else {}
        finally:
            self.prefetch = prefetch
        by_number = {p.number: p for p in open_prs}
        for n, since in idle.items():
            p = by_number[n]
            e = entries.setdefault(n, Entry(p, state="waiting", why=f"based on {based_on(p, open_prs)}, and not marked ready at "
                                                                    f"{p.head[:8]}"))
            e.idle_since = since
            e.why = f"stale, idle since {time.strftime('%Y-%m-%d', time.gmtime(since))}: {e.why}"
        conflicts, waits = ({}, {}) if loop else (state.get("conflicts", {}), state.get("waiting", {}))
        if loop:
            reach = above(self.workdir, main, (e.pr.head for e in entries.values() if e.state == "ready"))
            for e in entries.values():
                if e.state == "ready" and not reach[e.pr.head][0]:
                    e.state, e.why = "queued", "landed: its head is on `main`"
        else:
            for i, m in enumerate(self.chain(), 1):
                if m.pr in entries and entries[m.pr].pr.head == m.head:
                    entries[m.pr].state, entries[m.pr].position, entries[m.pr].commit = "queued", i, m.commit
            if self.into != "main":
                landed = queue_log(self.workdir, self.base)[0]
                reach = above(self.workdir, self.base, (e.pr.head for e in entries.values() if e.state == "ready"))
                for e in entries.values():
                    if e.state == "ready" and (e.pr.head in landed or not reach[e.pr.head][0]):
                        e.state, e.why = "queued", (f"landed on `{self.into}`: it reaches `main` when drive's full check of a "
                                                    f"`{self.into}` commit at or above it passes every step on the merge gate")
        files_of = self.conflicts_with(main, [e.pr.head for n, e in entries.items() if e.state == "ready"
                                              and (loop or not e.pr.move)
                                              and conflicts.get(str(n), {}).get("head") != e.pr.head
                                              and waits.get(str(n), {}).get("head") != e.pr.head])
        maps: dict[tuple[str, str], Retired | None] = {}

        def held(a: int) -> str | None:
            why = unreviewed(reviews(a, entries[a].pr.head), ()) if reviews else None
            return why if why and why.startswith("held") else None

        for n in contained_first(entries):
            e = entries[n]
            if loop and (e.state == "ready" or e.why.startswith("contains")) and (below := held_below(entries, n, held)):
                e.state, e.why = "waiting", held_waits(below[0], entries[below[0]].pr.head, below[1])
            elif e.state == "ready" and (out := [a for a in e.after if a in entries and entries[a].state in ("refused", "waiting")]):
                e.state, e.why = "waiting", f"contains {e.contains(out[0])}, which is {entries[out[0]].state}: {entries[out[0]].why}"
            elif e.state == "ready" and (c := conflicts.get(str(n))) and c["head"] == e.pr.head:
                e.state, e.why = "refused", f"{c['why']} (at the last sync)"
            elif e.state == "ready" and (w := waits.get(str(n))) and w["head"] == e.pr.head:
                e.state, e.why = "waiting", f"{w['why']} (at the last sync)"
            elif e.state == "ready" and (loop or not e.pr.move) and (files := files_of[e.pr.head]):
                e.state, e.why = "refused", (f"conflicts with `main` in {', '.join(files)}: merge `main` into the branch and "
                                             f"re-mark it")
            elif e.state == "ready" and loop and (lacked := lacked_moves(self.workdir, main, e.pr.head, maps=maps)):
                e.state, e.why = "refused", lacks_why(lacked)
            elif e.state == "ready":
                e.why = ("admitted: the verified queue merges it into a candidate on `main` once it has a land verdict"
                         + (", or on its mark alone once its move is certified on `main`, as drive.toml allows" if e.pr.move else
                            ", or on its mark alone when it is low-risk"))
        return entries

    def conflicts_with(self, main: str, heads: Iterable[str]) -> dict[str, list[str]]:
        from .clone import RECORD_DRIVER

        path = self.workdir.parent / f"{self.workdir.name}.conflicts.json"
        key = {"main": main, "at": _out(self.workdir, "rev-parse", "HEAD"), "drivers": RECORD_DRIVER}
        doc = json.loads(path.read_text()) if path.is_file() else {}
        got = dict(doc.get("of") or {}) if doc.get("key") == key else {}
        if todo := [h for h in dict.fromkeys(heads) if h not in got]:
            with concurrent.futures.ThreadPoolExecutor(os.cpu_count() or 1) as ex:
                got.update(zip(todo, ex.map(lambda h: conflicting_files(self.workdir, main, h), todo)))
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps({"key": key, "of": got}))
            tmp.replace(path)
        return got

    def off_main(self, prs: list[PR]) -> dict[int, Entry]:
        off = [p for p in prs if p.base != "main"]
        if off and self.prefetch is not None:
            self.prefetch([(p.number, p.head) for p in off])
        return {p.number: Entry(p, state="waiting", why=f"based on {based_on(p, prs)}, not main: retarget it to main")
                for p in off if self.ready(p.number, p.head)}


_COLD_PREFIXES = ("verity/circuits/", "verity/circuits/ops/", "verity/circuits/silicon/",
                  "tools/verity/circuit_check/", "adapters/vllm/verity_vllm/", "verity/protocols/proofs/flock/prover/python/",
                  "verity/protocols/resource_accounting/compute/pouw/verifier/", "verity/protocols/resource_accounting/memory/pous/verifier/", "tools/pous_grader/",
                  "verity/protocols/resource_accounting/memory/pous/attester/challenger_session.py", "verity/protocols/resource_accounting/memory/pous/attester/memory_challenger.py",
                  "verity/protocols/resource_accounting/memory/poub/verifier/commitments/",
                  "verity/commitments/frame_v3/", "verity/commitments/gates/", "verity/commitments/hm96/",
                  "verity/commitments/__init__.py", "verity/commitments/advice.py", "verity/commitments/coin_tree.py",
                  "verity/commitments/domains.py", "verity/commitments/history.py", "verity/commitments/identity.py",
                  "verity/commitments/indexed.py", "verity/commitments/layout.py", "verity/commitments/limits.py",
                  "verity/commitments/merkle.py", "verity/commitments/multiproof.py", "tests/verity/commitments/pyproject.toml",
                  "verity/commitments/row_table.py", "verity/commitments/rowleaf.py", "verity/commitments/scheme.py",
                  "verity/commitments/window.py",
                  "verity/circuits/private_circuits/", "benchmarks/numerical/python/")


def contained_first(entries: dict[int, Entry]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()

    def place(n: int) -> None:
        if n in seen:
            return
        seen.add(n)
        for a in entries[n].after:
            if a in entries:
                place(a)
        out.append(n)

    for n in sorted(entries):
        place(n)
    return out


def forces_cold(files: Iterable[str]) -> bool:
    for f in files:
        if f.endswith(".md") or "/tests/" in f:
            continue
        if f.startswith(_COLD_PREFIXES) or f in ("uv.lock", "pyproject.toml") or f.endswith(".circuit-check.py"):
            return True
    return False


def merged_tree(repo: Path, base: str, head: str, *, plain: bool = True, config: tuple[str, ...] | None = None,
                merge_base: str | None = None) -> tuple[str | None, list[str], subprocess.CompletedProcess]:
    if config is None:
        with base_drivers(repo, base) as config:
            return merged_tree(repo, base, head, plain=plain, config=config, merge_base=merge_base)
    at = (f"--merge-base={merge_base}",) if merge_base else ()
    r = _git(repo, *config, f"--attr-source={base}", "merge-tree", "--write-tree", "--name-only", "--no-messages", *at, base,
             head, check=False)
    if r.returncode == 129 and "attr-source" in r.stderr:
        if not plain:
            return None, [], r
        r = _git(repo, *config, "merge-tree", "--write-tree", "--name-only", "--no-messages", *at, base, head, check=False)
    if r.returncode == 1:
        files = sorted({f for f in r.stdout.splitlines()[1:] if f})
        tree, rest = port_deletions(repo, r.stdout.split()[0], base, head, files)
        return (repinned(repo, tree, base, head, rest) if rest else tree), files, r
    return (None if r.returncode else r.stdout.split()[0]), [], r


def port_deletions(repo: Path, tree: str, ours: str, theirs: str, conflicted: list[str]) -> tuple[str, list[str]]:
    lean = [f for f in conflicted if f.endswith(".lean")]
    if not lean:
        return tree, conflicted
    bases = _git(repo, "merge-base", "--all", ours, theirs, check=False).stdout.split()
    sides = _blobs(repo, [f"{c}:{f}" for c in (*bases, ours, theirs) for f in lean])
    take: list[str] = []
    for porter, deleter in ((ours, theirs), (theirs, ours)):
        held = [f for f in lean if f"{porter}:{f}" in sides and f"{deleter}:{f}" not in sides
                and any(f"{b}:{f}" in sides for b in bases)]
        if held and on_modules(repo, porter)[0]:
            take += _only_ported(repo, porter, bases, held, sides)
    if not take:
        return tree, conflicted
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        env = {"GIT_INDEX_FILE": str(Path(tmp, "index"))}
        _git(repo, "read-tree", tree, env=env)
        _git(repo, "update-index", "--force-remove", *take, env=env)
        return _git(repo, "write-tree", env=env).stdout.strip(), [f for f in conflicted if f not in take]


def _only_ported(repo: Path, side: str, bases: list[str], files: list[str], sides: dict[str, bytes]) -> list[str]:
    out: list[str] = []
    tried: set[tuple[tuple[str, bytes], ...]] = set()
    for b in bases:
        todo = [f for f in files if f not in out and f"{b}:{f}" in sides]
        if not todo or (texts := tuple((f, sides[f"{b}:{f}"]) for f in todo)) in tried:
            continue
        tried.add(texts)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            env = {"GIT_INDEX_FILE": str(Path(tmp, "index"))}
            _git(repo, "read-tree", side, env=env)
            for f in todo:
                mode, _, sha = _out(repo, "ls-tree", b, "--", f).split("\t", 1)[0].split()
                _git(repo, "update-index", "--cacheinfo", f"{mode},{sha},{f}", env=env)
            new, _, _, _ = _codemod(repo, _git(repo, "write-tree", env=env).stdout.strip())
        port = _blobs(repo, [f"{new}:{f}" for f in todo]) if new else {}
        out += [f for f in todo if port.get(f"{new}:{f}") == sides[f"{side}:{f}"]]
    return out


def taken(repo: Path, tree: str, head: str, files: list[str]) -> tuple[list[str], list[str], list[str]]:
    held = _blobs(repo, [f"{c}:{f}" for c in (tree, head) for f in files])
    gone = [f for f in files if f"{tree}:{f}" not in held]
    return ([f for f in files if f"{tree}:{f}" in held], [f for f in gone if f"{head}:{f}" not in held],
            [f for f in gone if f"{head}:{f}" in held])


_DRIVER_PATH = re.compile(r"(?<![\w./-])[\w.-]+(?:/[\w.-]+)+")


@contextlib.contextmanager
def base_drivers(repo: Path, base: str):
    from .clone import RECORD_DRIVER

    attrs = _git(repo, "show", f"{base}:.gitattributes", check=False)
    names = dict.fromkeys(m for ln in attrs.stdout.splitlines() if not ln.lstrip().startswith("#")
                          for m in re.findall(r"(?:^|\s)merge=(\S+)", ln)) if not attrs.returncode else {}
    cmds = {k: c for k in (f"merge.{n}.driver" for n in names)
            if (c := RECORD_DRIVER.get(k) or _git(repo, "config", "--get", k, check=False).stdout.strip())}
    with tempfile.TemporaryDirectory(prefix="research-drivers-", ignore_cleanup_errors=True) as tmp:
        held = {s.split(":", 1)[1] for s in _blobs(repo, [f"{base}:{p}" for c in cmds.values() for p in _DRIVER_PATH.findall(c)])}
        files = []
        for d in sorted({posixpath.dirname(p) for p in held}):
            ls = _git(repo, "ls-tree", "-z", base, "--", f"{d}/").stdout
            files += [e.split("\t", 1)[1] for e in ls.split("\0") if e and e.split()[1] == "blob"]
        for spec, data in _blobs(repo, [f"{base}:{f}" for f in files]).items():
            (out := Path(tmp) / spec.split(":", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
        def moved(m: re.Match) -> str:
            return shlex.quote(str(Path(tmp) / m[0])) if m[0] in held else m[0]

        yield tuple(o for k, c in cmds.items() for o in ("-c", f"{k}={_DRIVER_PATH.sub(moved, c)}"))


def conflicting_files(repo: Path, base: str, head: str) -> list[str]:
    tree, files, _ = merged_tree(repo, base, head)
    return [] if tree else files


def pins_at(repo: Path, ref: str) -> dict[str, tuple[str | None, str]]:
    for path in PINS_FILES:
        if (r := _git(repo, "show", f"{ref}:{path}", check=False)).returncode == 0:
            break
    else:
        return {}
    try:
        [calls] = [n.value.elts for n in ast.parse(r.stdout).body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Tuple)
                   and any(isinstance(t, ast.Name) and t.id == "PINS" for t in n.targets)]
        out = {}
        for c in calls:
            args = {**dict(zip(("path", "key", "under"), c.args)), **{k.arg: k.value for k in c.keywords}}
            path, key, under = (ast.literal_eval(args[a]) for a in ("path", "key", "under"))
            out[path] = (key, under)
        return out
    except (SyntaxError, ValueError, KeyError, AttributeError):
        return {}


def _blobs(repo: Path, specs: list[str]) -> dict[str, bytes]:
    out = subprocess.run(["git", "cat-file", "--batch"], cwd=repo, input="".join(f"{s}\n" for s in specs).encode(),
                         capture_output=True, env=git_env(GIT_TERMINAL_PROMPT="0"), check=True).stdout
    got, i = {}, 0
    for s in specs:
        j = out.index(b"\n", i)
        head, i = out[i:j].split(), j + 1
        if head[-1] in (b"missing", b"ambiguous"):
            continue
        if head[1] == b"blob":
            got[s] = out[i:i + int(head[2])]
        i += int(head[2]) + 1
    return got


def repinned(repo: Path, tree: str, ours: str, theirs: str, conflicted: list[str]) -> str | None:
    pins = pins_at(repo, ours)
    if not conflicted or any(p not in pins for p in conflicted):
        return None
    bases = _git(repo, "merge-base", "--all", ours, theirs, check=False).stdout.split()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        env = {"GIT_INDEX_FILE": str(Path(tmp, "index"))}
        _git(repo, "read-tree", tree, env=env)
        for path in conflicted:
            key, under = pins[path]
            texts = _blobs(repo, [f"{c}:{path}" for c in (ours, theirs, *bases)])
            if not (sides := [texts[s] for s in (f"{ours}:{path}", f"{theirs}:{path}") if s in texts]):
                return None
            recorded: dict[str, str] = {}
            try:
                for text in (b.decode() for b in sides):
                    pairs = ([ln.split("  ", 1)[::-1] for ln in text.splitlines() if ln.strip()] if key is None
                             else json.loads(text)[key].items())
                    recorded |= {p: h for p, h in pairs if p not in recorded}
                kept = sides[0] if key is None else _kept(key, texts.get(f"{ours}:{path}"), texts.get(f"{theirs}:{path}"),
                                                          [texts.get(f"{b}:{path}") for b in bases])
            except (ValueError, KeyError, TypeError, AttributeError):
                return None
            if kept is None:
                return None
            full = {p: posixpath.normpath(posixpath.join(under, p)) for p in recorded}
            found = _blobs(repo, [f"{tree}:{f}" for f in full.values()])
            digests = {p: hashlib.sha256(found[f"{tree}:{f}"]).hexdigest() for p, f in full.items() if f"{tree}:{f}" in found}
            text = ("".join(f"{h}  {p}\n" for p, h in digests.items()) if key is None
                    else _with_json(kept.decode(), key, digests))
            blob = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=repo, input=text.encode(), capture_output=True,
                                  env=git_env(GIT_TERMINAL_PROMPT="0"), check=True).stdout.decode().strip()
            mode = _out(repo, "ls-tree", tree, "--", path).split(" ", 1)[0] or "100644"
            _git(repo, "update-index", "--add", "--cacheinfo", f"{mode},{blob},{path}", env=env)
        return _git(repo, "write-tree", env=env).stdout.strip()


def _kept(key: str, ours: bytes | None, theirs: bytes | None, bases: list[bytes | None]) -> bytes | None:
    def rest(text: bytes | None) -> dict | None:
        return None if text is None else {k: v for k, v in json.loads(text).items() if k != key}

    o, t, was = rest(ours), rest(theirs), [rest(b) for b in bases]
    if o == t or (was and all(w == t for w in was)):
        return ours
    return theirs if was and all(w == o for w in was) else None


def _with_json(text: str, key: str, value: dict) -> str:
    doc = json.loads(text)
    if m := re.search(rf"^([ \t]*){re.escape(json.dumps(key))}:\s*(?=\{{)", text, re.M):
        old, end = json.JSONDecoder().raw_decode(text, m.end())
        if old == doc[key]:
            return text[:m.end()] + json.dumps(value, indent=1).replace("\n", "\n" + m.group(1)) + text[end:]
    return json.dumps({**doc, key: value}, indent=1) + "\n"


def ported(repo: Path, tree: str, base: str | None = None) -> tuple[str | None, list[str], list[str], str]:
    on, why = on_modules(repo, tree)
    if not on:
        return (None if why else tree), [], [], why
    if base is not None and not _git(repo, "diff", "--quiet", base, tree, "--", *PORT_INPUTS, MODULES, HEADERS,
                                     check=False).returncode:
        return tree, [], [], ""
    new, files, pins, why = _codemod(repo, tree)
    return new, list(files), list(pins), why


_ROOTS_READ = ("import json, sys, types\nd = json.load(sys.stdin)\nm = types.ModuleType('headers')\n"
               "exec(compile(d['headers'], 'headers.py', 'exec'), m.__dict__)\n"
               "print(json.dumps([(h := m.header(s)) is not None and h.module for s in d['roots']]))\n")


def on_modules(repo: Path, tree: str) -> tuple[bool, str]:
    specs = [f"{tree}:{p}" for p in (MODULES, HEADERS, *PORTED_ROOTS)]
    blobs = _blobs(repo, specs)
    if len(blobs) < len(specs):
        return False, ""
    doc = {"headers": blobs[specs[1]].decode(errors="replace"),
           "roots": [blobs[s].decode(errors="replace") for s in specs[2:]]}
    r = subprocess.run([sys.executable, "-c", _ROOTS_READ], input=json.dumps(doc), capture_output=True, text=True,
                       env=_script_env())
    if r.returncode:
        return False, (f"`{HEADERS}` failed to read the headers of {', '.join(PORTED_ROOTS)} (exit {r.returncode}): "
                       f"{_tail(r, 3)}")
    return all(json.loads(r.stdout.splitlines()[-1])), ""


@functools.lru_cache(maxsize=256)
def _codemod(repo: Path, tree: str) -> tuple[str | None, tuple[str, ...], tuple[str, ...], str]:
    commit = _git(repo, "commit-tree", tree, "-m", "ported", env=IDENTITY).stdout.strip()
    with merged_worktree(repo, commit) as wt:
        r = subprocess.run([sys.executable, MODULES], cwd=wt, capture_output=True, text=True, env=_script_env())
        if r.returncode:
            return None, (), (), (f"`{MODULES}` failed on the merged tree (exit {r.returncode}): {_tail(r, 3)}; port by hand "
                                  "what it names")
        _git(wt, "add", "-u")
        diff = _git(wt, "diff", "--cached", "-z", "--no-renames", "--name-only", commit).stdout
        if not (files := tuple(f for f in diff.split("\0") if f)):
            return tree, (), (), ""
        pins = _repinned_files(repo, wt, tree, files)
        _git(wt, "add", "-u")
        return _out(wt, "write-tree"), files, pins, ""


def _repinned_files(repo: Path, wt: Path, tree: str, files: tuple[str, ...]) -> tuple[str, ...]:
    before = _blobs(repo, [f"{tree}:{f}" for f in files])
    moved = {f: (hashlib.sha256(before[f"{tree}:{f}"]).hexdigest(), hashlib.sha256((wt / f).read_bytes()).hexdigest())
             for f in files if f"{tree}:{f}" in before and (wt / f).is_file()}
    out = []
    for path, (key, under) in pins_at(repo, tree).items():
        if not (pin := wt / path).is_file():
            continue
        text = new = pin.read_bytes().decode(errors="surrogateescape")
        for f, (old, now) in moved.items():
            rel = posixpath.relpath(f, under) if under else f
            new = (re.sub(rf"(?m)^{old}(?=  {re.escape(rel)}$)", now, new) if key is None
                   else re.sub(rf'({re.escape(json.dumps(rel))}\s*:\s*"){old}"', rf'\g<1>{now}"', new))
        if new != text:
            pin.write_bytes(new.encode(errors="surrogateescape"))
            out.append(path)
    return tuple(out)


def rules_at(repo: Path, ref: str) -> Rules | None:
    for path in (RULES, RULES_BEFORE_MOVE):
        r = _git(repo, "show", f"{ref}:{path}", check=False)
        if r.returncode == 0:
            return Rules.parse(r.stdout, path)
    return None


def chain_of(repo: Path, base: str, ref: str) -> list[Member]:
    if _git(repo, "rev-parse", "--verify", "-q", ref, check=False).returncode:
        return []
    if _git(repo, "merge-base", "--is-ancestor", base, ref, check=False).returncode:
        return []
    out = []
    log = _out(repo, "log", "--first-parent", "--reverse", "--format=%H%x00%B%x1e", f"{base}..{ref}")
    for rec in filter(None, (x.strip() for x in log.split("\x1e"))):
        commit, body = rec.split("\x00", 1)
        t = dict(TRAILER_RE.findall(body))
        if not {"PR", "Head", "Owner"} <= set(t):
            return []
        out.append(Member(int(t["PR"].lstrip("#")), t["Head"], t["Owner"], commit))
    return out


def queue_merges(repo: Path, base: str, ref: str) -> list[Member]:
    out = []
    log = _out(repo, "log", "--first-parent", "--reverse", "--format=%H%x00%B%x1e", f"{base}..{ref}")
    for rec in filter(None, (x.strip() for x in log.split("\x1e"))):
        commit, body = rec.split("\x00", 1)
        t = dict(TRAILER_RE.findall(body))
        if {"PR", "Head", "Owner"} <= set(t):
            out.append(Member(int(t["PR"].lstrip("#")), t["Head"], t["Owner"], commit))
    return out


@dataclass(frozen=True)
class Move:
    commit: str
    map: str


def queue_log(repo: Path, ref: str, n: int = LANDED_WINDOW) -> tuple[set[str], list[Move]]:
    heads, moves = set(), []
    log = _git(repo, "log", "--first-parent", f"-n{n}", "--format=%H%x00%B%x1e", ref, check=False).stdout
    for rec in filter(None, (x.strip() for x in log.split("\x1e"))):
        commit, body = rec.split("\x00", 1)
        t = dict(TRAILER_RE.findall(body))
        if {"PR", "Head"} <= set(t):
            heads.add(t["Head"].strip())
            if m := MOVE_TRAILER_RE.search(body):
                moves.append(Move(commit, m.group(1)))
    return heads, moves[::-1]


OLD_NEXT_SHOWN = 3


def old_next(repo: Path, heads: Iterable[str], landed: str, open_prs: Collection[int]) -> dict[str, Member]:
    heads = list(dict.fromkeys(heads))
    if not heads:
        return {}
    env = git_env(GIT_TERMINAL_PROMPT="0")
    r = subprocess.run(["git", "log", "--stdin", "--grep=^Queue: #", "--format=%H%x00%P%x00%B%x1e"], cwd=repo,
                       input="".join(f"{h}\n" for h in [*heads, f"^{landed}"]), capture_output=True, text=True,
                       errors="surrogateescape", env=env)
    if r.returncode:
        raise QueueError(f"git log {landed}.. ({len(heads)} heads): {(r.stderr or r.stdout).strip()[-400:]}")
    found = {m.commit: m for rec in r.stdout.split("\x1e") if rec.strip()
             if (m := queue_commit(*rec.strip().split("\x00", 2))) and m.pr not in open_prs}
    if not found:
        return {}
    done = queue_log(repo, landed)[0]
    if not (maybe := sorted({m.head for m in found.values()} - done)):
        return {}
    have = subprocess.run(["git", "cat-file", "--batch-check=%(objectname)"], cwd=repo, input="".join(f"{h}\n" for h in maybe),
                          capture_output=True, text=True, env=env).stdout.splitlines()
    off = subprocess.run(["git", "rev-list", "--ignore-missing", "--stdin"], cwd=repo,
                         input="".join(f"{h}\n" for h in [*maybe, f"^{landed}"]), capture_output=True, text=True, env=env)
    if off.returncode:
        raise QueueError(f"git rev-list {landed}.. ({len(maybe)} heads): {(off.stderr or off.stdout).strip()[-400:]}")
    on = {h for h in have if not h.endswith(" missing")}.difference(off.stdout.split())
    return {c: m for c, m in found.items() if m.head not in done and m.head not in on}


def old_next_refusal(old: list[Member], into: str) -> str:
    shown = ", ".join(f"{m.commit[:12]} (#{m.pr})" for m in old[:OLD_NEXT_SHOWN])
    more = f" and {len(old) - OLD_NEXT_SHOWN} more" if len(old) > OLD_NEXT_SHOWN else ""
    return (f"its head carries {len(old)} merge{'s' if len(old) > 1 else ''} of an old `{BRANCH}` that `{into}` doesn't hold, "
            f"of PRs closed unlanded ({shown}{more}), which would land with it: retarget the PR to the demo branch it builds on, "
            f"or restack its own commits onto `main` (a branch from `origin/main` with each cherry-picked, and the branch of each "
            f"open PR it needs merged, never `{BRANCH}`) and mark that head")


def move_of(repo: Path, commit: str) -> str | None:
    m = MOVE_TRAILER_RE.search(_git(repo, "log", "-1", "--format=%B", commit, check=False).stdout)
    return m.group(1) if m else None


def _script_env() -> dict[str, str]:
    env = git_env(GIT_TERMINAL_PROMPT="0", PYTHONDONTWRITEBYTECODE="1", **IDENTITY)
    for k in DATES:
        env.pop(k, None)
    return env


def _tail(r: subprocess.CompletedProcess, n: int = 6) -> str:
    return " | ".join(((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-n:])


def restack_end(job: dict, now: float) -> tuple[dict | None, str, bool] | None:
    from .landing import _job_end

    if (end := _job_end(job, RESTACK_OUT)) is None and now - job["at"] <= RESTACK_LOST_S:
        return None
    if not isinstance(end, tuple):
        shutil.rmtree(job["dir"], ignore_errors=True)
    unread = ""
    try:
        (x,) = json.loads(Path(job["out"], RESTACK_OUT).read_text())
        said = {"result": x["result"]} if x["status"] in ("clean", "up-to-date") else {"why": restack_why(x)}
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
        said, unread = None, f"; its results don't read ({type(e).__name__}: {e})"
    shutil.rmtree(job["out"], ignore_errors=True)
    if end is None:
        return said, f"no exit written {RESTACK_LOST_S // 60} min after its start", False
    if isinstance(end, str):
        return said, end, True
    rc, tail, _ = end
    stopped = f", stopped {RESTACK_WAIT_S // 60} min after its start" if rc == "124" else ""
    tail = " | ".join(tail.splitlines()[-6:])
    return (said, f"exit {rc}{stopped}" + (f": {tail}" if tail else "") + (unread if rc == "0" else ""),
            rc in RESTACK_INTERRUPTED)


def changed(repo: Path, base: str, commit: str) -> set[str]:
    return set(_out(repo, "diff", "--name-only", "--no-renames", base, commit).split())


def touched_by(repo: Path, members: list[Member], files: list[str], cache: dict[str, set[str]] | None = None) -> list[str]:
    cache = {} if cache is None else cache
    for m in members:
        if m.commit not in cache:
            cache[m.commit] = set(_out(repo, "diff", "--name-only", f"{m.commit}^1", m.commit).split())
    return [f"#{m.pr}" for m in members if set(files) & cache[m.commit]]


def _lock(workdir: Path):
    workdir.parent.mkdir(parents=True, exist_ok=True)
    f = open(workdir.parent / f"{workdir.name}.lock", "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise QueueError(f"another `research queue` holds {f.name}") from None
    return f


def _render(entries: dict[int, Entry]) -> str:
    rank = {"queued": 0, "ready": 1, "waiting": 2, "refused": 3}
    rows = sorted(entries.values(), key=lambda e: (rank[e.state], e.position or 0, e.pr.number))
    lines = []
    for e in rows:
        where = f"{e.position:>3}" if e.position else "  -"
        lines.append(f"{where}  #{e.pr.number:<5} {e.pr.head[:8]}  {e.state:<8} {(e.pr.owner or '-')[:24]:<24} {e.why}")
    return "\n".join(lines) or "no open PRs"


SUITES = quick_tier.SUITES
QUICK_TIER = ["uv", "run", "--locked", "--all-packages", "--extra", "torch-cpu", "python", SUITES, "--quick"]
LEAN_BUILDS = ("verity/protocols/proofs/flock/verifier",)
LEAN_AUDIT: list[str] | None = None


def lean_audit(repo: Path, base: str, at: str) -> list[str] | None:
    if LEAN_AUDIT and any(f.endswith(LEAN_SOURCES) or Path(f).name == LEAN_LOCK for f in changed(repo, base, at)):
        return LEAN_AUDIT
    return None


def allowed(repo: Path, base: str | None) -> Callable[[dict], bool] | None:
    from . import policy

    pol = policy.at(repo, base) if base else None
    if not isinstance(pol, policy.Policy) or not pol.allow:
        return None
    now = time.time()
    return lambda row: pol.suite(str(row.get("name")), row, now).covered


class TierRefused(QueueError):
    """Why the quick tier has no commit to run at: the head isn't on the remote, or doesn't merge onto `main`'s tip."""


def onto_main(repo: Path, main: str, pr: PR) -> str:
    if not _git(repo, "merge-base", "--is-ancestor", main, pr.head, check=False).returncode:
        return pr.head
    tree, files, r = merged_tree(repo, main, pr.head)
    if tree is None:
        raise TierRefused(f"it conflicts with `main` at {main[:12]} in {', '.join(files)}: merge `main` into the branch" if files else
                          f"`git merge-tree` of it onto `main` exited {r.returncode} ({(r.stderr or r.stdout).strip()[-200:]})")
    tree, _, _, why = ported(repo, tree, base=main)
    if tree is None:
        raise TierRefused(why)
    return _git(repo, "commit-tree", tree, "-p", main, "-p", pr.head, "-m", f"quick tier: #{pr.number} onto main {main[:12]}",
                env=IDENTITY).stdout.strip()


class QuickTier:
    WORKTREE_SUFFIX = ""

    def __init__(self, repo: Path, remote: str, root: Path | None = None):
        self.repo, self.remote = Path(repo), remote
        self.root = Path(root or Path.home() / ".research" / "queue" / "ready")

    def __call__(self, pr: PR) -> str | None:
        """None when the tier passes on ``pr``'s head merged onto `main`'s tip; else what failed, for the refusal."""
        try:
            with self._checkout(pr) as (wt, at, base):
                out = self.root / "logs" / f"{pr.number}-{pr.head[:12]}"
                doc = quick_tier.tier(wt, out, at, base, QUICK_TIER, LEAN_BUILDS, merged=None if at == pr.head else pr.head,
                                      audit=lean_audit(self.repo, base, at))
                return quick_tier.judge(doc, at, covers=allowed(self.repo, base))
        except TierRefused as e:
            return str(e)

    @contextlib.contextmanager
    def _checkout(self, pr: PR):
        common = Path(_out(self.repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
        wt = self.root / (hashlib.sha256(str(common).encode()).hexdigest()[:12] + self.WORKTREE_SUFFIX)
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / f"{wt.name}.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            _git(self.repo, "fetch", "-q", self.remote, "+refs/heads/main:refs/queue/ready/main",
                 f"+refs/pull/{pr.number}/head:refs/queue/ready/pr")
            if _git(self.repo, "cat-file", "-e", f"{pr.head}^{{commit}}", check=False).returncode:
                raise TierRefused(f"{pr.head[:12]} isn't on {self.remote} (pushed again since?)")
            main = _out(self.repo, "rev-parse", "refs/queue/ready/main")
            if (at := onto_main(self.repo, main, pr)) != pr.head:
                print(f"research queue: #{pr.number}'s head {pr.head[:12]} merged onto `main` at {main[:12]}: the tier runs on "
                      f"{at[:12]}", flush=True)
            _git(self.repo, "worktree", "prune")
            if not (wt / ".git").exists():
                shutil.rmtree(wt, ignore_errors=True)
                _git(self.repo, "worktree", "add", "-q", "--detach", "--force", str(wt), at)
            else:
                _git(wt, "checkout", "-q", "--detach", "--force", at)
            _git(wt, "clean", "-qffdx", "-e", "/.venv", *(x for d in LEAN_BUILDS for x in ("-e", f"/{d}/.lake")))
            yield wt, at, main


REMOTE_TIMEOUT_S = 3 * 3600
REMOTE_WAIT_S = 6 * 3600
REMOTE_GRACE_S = 1800
REMOTE_POLL_S = 30
#: slot.py's exit when no check slot came within --wait-max (its NOT_STARTED): the tier never ran
SLOT_NOT_STARTED = 75
SLOT = ("python3", "tools/check/slot.py")
#: `research queue ready`'s exit when the tier never started: nothing failed, and nothing is marked
READY_NOT_STARTED = 3
SPILL_CPUS, SPILL_MEM_GB = 4, 16
POOL_SLOTS = "/workspace/research/locks/{pool}/slots"
LANE_RUN = ('`research run --on auto --cwd source -- bash -lc "uv sync -q --all-packages --extra torch-cpu && '
            'uv run tools/check/suites.py --changed $(git fetch -q origin main && git merge-base HEAD FETCH_HEAD) --quick"`')


class NotStarted(str):
    """Why a quick tier never ran (no check slot within its wait, no machine to run it on): not a failure of the PR's head."""


class StillGoing(str):
    """The caller stopped waiting with the tier's run still going: nothing failed or was cancelled."""


def place(spec: str, repo: Path, read: Callable[[list[str]], dict] | None = None,
          spill: Callable[[Path], tuple[tuple[str, str] | None, str]] | None = None) -> tuple[str | None, str]:
    from .store import tools_registry

    path = tools_registry.resolve("tools/check/check.py", lambda p: (repo / p).is_file()) or "tools/check/check.py"
    s = importlib.util.spec_from_file_location("queue_check", repo / path)
    check = importlib.util.module_from_spec(s)
    sys.modules[s.name] = check
    s.loader.exec_module(check)
    names = list(check.AUTO) if spec == "auto" else [n for n in spec.split(",") if n]
    docs = {n: ({**d, "lean": {"pools": None}} if isinstance(d, dict) else d) for n, d in (read or check.read_slots)(names).items()}
    got, why, skipped = check.choose(names, docs)
    why += "".join(f"; skipped {s}" for s in skipped)
    if got is None and spec == "auto":
        target, how = (spill or _spill)(repo)
        if target is not None:
            return f"{target[0]}:{target[1]}", f"{how}, as none of {', '.join(names)} could be read ({why})"
        why += f"; no research pool either: {how}"
    return got, why


def _spill(repo: Path) -> tuple[tuple[str, str] | None, str]:
    from . import auto

    try:
        target, why = auto.choose(auto.read(repo, SPILL_CPUS), SPILL_CPUS, SPILL_MEM_GB)
    except auto.Unplaceable as e:
        return None, str(e)
    return (target, why) if target is not None else (None, why)


def _launch(argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True)


def quick_run(rid: str, on: str, wt: Path, at: str, base: str, *, merged: str | None = None, audit: list[str] | None = None,
              by: str = "", project: str = "verity", wait_s: float = REMOTE_WAIT_S, timeout_s: float = REMOTE_TIMEOUT_S,
              pool: str = "", priority: bool = False) -> list[str]:
    slot = (["env", f"CHECK_SLOTS={POOL_SLOTS.format(pool=pool)}"] if pool else []) + list(SLOT)
    job = Path(quick_tier.__file__)
    args = [*(x for d in LEAN_BUILDS for x in ("--lean", d)), "--head", at, "--base", base,
            *(["--merged", merged] if merged else []), *(["--lean-audit", shlex.join(audit)] if audit else []),
            "--", *QUICK_TIER]
    opts = [*(["--priority"] if priority else []), *(["--by", by] if by else []), "--wait-max", f"{wait_s:g}",
            "--run-max", f"{timeout_s:g}"]
    return [sys.executable, "-m", "research", "run", "--id", rid, "--on", on, "--project", project,
            "--source", str(wt), "--cwd", "source", "--send", str(job), "--require-result",
            "--timeout", f"{wait_s + timeout_s:g}", "--custody-ttl", str(math.ceil(wait_s + timeout_s + REMOTE_GRACE_S)),
            "--declared-output", "out/*", "--", *slot, *opts,
            "--", "sh", "-c", f'exec python3 "$RESEARCH_RUN_DIR/inputs/{job.name}" {shlex.join(args)}']


class RemoteQuickTier(QuickTier):
    WORKTREE_SUFFIX = "-ship"

    def __init__(self, repo: Path, remote: str, on: str, *, by: str = "", project: str = "verity", root: Path | None = None,
                 timeout_s: float = REMOTE_TIMEOUT_S, wait_s: float = REMOTE_WAIT_S, max_wait_s: float = waitcap.WAIT_LIMIT_S,
                 poll_s: float = REMOTE_POLL_S, launch: Callable[[list[str], Path], subprocess.CompletedProcess] = _launch,
                 fetch: Callable[[Path], dict] | None = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, place: Callable[[str, Path], tuple[str | None, str]] = place):
        super().__init__(repo, remote, root)
        self.on, self.by, self.project, self.poll_s = on, by, project, poll_s
        self.timeout_s, self.wait_s, self.max_wait_s = timeout_s, wait_s, max_wait_s
        self.launch, self.fetch, self.sleep, self.clock, self.place = launch, fetch, sleep, clock, place

    def __call__(self, pr: PR) -> str | None:
        from . import runs as R

        on, pool = self.on, ""
        if on == "auto" or "," in on:
            got, why = self.place(on, self.repo)
            if got is None:
                return NotStarted(f"--on {on}: {why}")
            print(f"research queue: --on {on}: {got}: {why}", flush=True)
            on, _, pool = got.partition(":")
        rid = R.new_run_id()
        try:
            with self._checkout(pr) as (wt, at, base):
                r = self.launch(quick_run(rid, on, wt, at, base, merged=None if at == pr.head else pr.head,
                                          audit=lean_audit(self.repo, base, at), by=self.by, project=self.project,
                                          wait_s=self.wait_s, timeout_s=self.timeout_s, pool=pool), wt)
        except TierRefused as e:
            return str(e)
        if r.returncode:
            return f"`research run --on {on}` failed (exit {r.returncode}): {(r.stderr or r.stdout or '').strip()[-400:]}"
        print(f"research queue: the quick tier runs as {rid} on {on}, in {'a slot of its ' + pool + ' pool' if pool else 'a check slot'}; waiting for it to end "
              f"(`research inspect {rid}`)", flush=True)
        d = R.runs_root() / rid
        status, why = self._wait(d, on)
        return why or judge_run(d, rid, on, at, status, allowed(self.repo, base))

    def _wait(self, d: Path, on: str) -> tuple[dict, str | None]:
        from . import remote
        from . import runs as R

        status: dict = {}
        bound = self.wait_s + self.timeout_s + REMOTE_GRACE_S
        limit = min(bound, self.max_wait_s)
        deadline = self.clock() + limit
        fetch = self.fetch or (lambda d: remote.fetch(d, timeout=deadline - self.clock()))
        while self.clock() < deadline:
            self.sleep(min(self.poll_s, deadline - self.clock()))
            try:
                rec = fetch(d)
            except Exception as e:  # noqa: BLE001  (a fetch that fails is a poll that saw nothing)
                rec = {"last_fetch_error": {"stderr": f"{type(e).__name__}: {e}"}}
            status = _json_file(d / "status.json")
            if status.get("state") in R.TERMINAL:
                return status, None
            if rec.get("launcher_exited") or rec.get("runner_gone"):
                try:
                    fetch(d)
                except Exception:  # noqa: BLE001
                    pass
                status = _json_file(d / "status.json")
                if status.get("state") in R.TERMINAL:
                    return status, None
            if gone := rec.get("launcher_exited"):
                return status, f"run {d.name}'s runner on {on} exited before the run ended ({gone.get('last_line') or 'no log line'})"
            if gone := rec.get("runner_gone"):
                return status, f"run {d.name} on {on} ended without recording it: {gone['why']}"
        if limit < bound:
            return status, StillGoing(waitcap.still_going(
                f"the quick tier's run {d.name} on {on} ({status.get('state') or 'without a status'})",
                f"research queue judge {d.name} --by {self.by or 'LANE'}", limit))
        return status, (f"run {d.name} on {on} didn't end within {bound:g} s (last seen "
                        f"{status.get('state') or 'without a status'}): `research inspect {d.name}`")


def tier_report(d: Path, rid: str, on: str, status: dict):
    from .result import check

    res = check(d, rid)
    if not res.present and status.get("rc") == SLOT_NOT_STARTED:
        return None, NotStarted(f"run {rid} on {on} waited for a check slot until its --wait-max and never ran the tier "
                                f"(`research inspect {rid}`)")
    if not res.present:
        return None, f"run {rid} on {on} ended {status.get('state')} (exit {status.get('rc')}) without a result.json (`research inspect {rid}`)"
    if not res.valid:
        return None, f"run {rid}'s result.json is unreadable: {'; '.join(res.errors)[:400]}"
    if (kind := res.doc.get("kind")) != quick_tier.KIND:
        return None, f"run {rid}'s result is a {kind or 'result with no kind'}, not a {quick_tier.KIND} report"
    return res, None


def judge_report(res, rid: str, on: str, head: str, status: dict, covers: Callable[[dict], bool] | None = None) -> str | None:
    tier, output = res.doc.get("tier"), f"its output is in run {rid}'s log on {on}"
    if (why := quick_tier.judge(tier, head, output, covers)) is not None:
        return why
    rc, validation = (0, "passed") if quick_tier.judge(tier, head, output) is None else (1, "failed")
    if (status.get("state"), status.get("rc"), res.validation_status) != (end_state(rc), rc, validation):
        return (f"run {rid}'s facts show a pass, but it ended {status.get('state')} (exit {status.get('rc')}) with validation "
                f"{res.validation_status}")
    return None


def judge_run(d: Path, rid: str, on: str, head: str, status: dict, covers: Callable[[dict], bool] | None = None) -> str | None:
    res, why = tier_report(d, rid, on, status)
    return why or judge_report(res, rid, on, head, status, covers)


def _json_file(p: Path) -> dict:
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def same_tree_mark(store, repo: Path, remote: str, pr: PR) -> str | None:
    _fetch_heads(repo, remote, [pr])
    if not (want := _tree(repo, pr.head)):
        return None
    marked = StoreLabels(store)._other_heads(pr.number, pr.head, vocab.READY)
    return next((old for old, labs in sorted(marked.items()) if any(str(lab.value).lower() == "true" for lab in labs)
                 and _tree(repo, old) == want), None)


def same_tree_prs(repo: Path, remote: str, prs: list[PR], head: str) -> list[PR]:
    _fetch_heads(repo, remote, prs)
    want = _tree(repo, head)
    return [p for p in prs if want and _tree(repo, p.head) == want] if prs else []


def _fetch_heads(repo: Path, remote: str, prs: list[PR]) -> None:
    """Each PR's head into a ref of its own, so a tier's fetch under the worktree's lock never meets this one."""
    if prs:
        _git(repo, "fetch", "-q", remote, *(f"+refs/pull/{p.number}/head:refs/queue/same-tree/{p.number}" for p in prs), check=False)


def _tree(repo: Path, commit: str) -> str:
    return _git(repo, "rev-parse", "-q", "--verify", f"{commit}^{{tree}}", check=False).stdout.strip()


OWN_KEEP = 4096


def own_change(repo: Path, base: str, head: str) -> str | None:
    diff = subprocess.run(["git", "diff-tree", "-r", "-p", "--full-index", base, head], cwd=repo, capture_output=True,
                          env=git_env(GIT_TERMINAL_PROMPT="0"))
    if diff.returncode or not diff.stdout:
        return None
    got = subprocess.run(["git", "patch-id", "--verbatim"], cwd=repo, input=diff.stdout, capture_output=True).stdout.split()
    return got[0].decode() if got else None


MOVE_LINE_RE = re.compile(r"^(?:Move|Layout-move): (tools/move/[\w.-]+\.toml)(?: --no-lock)?$", re.M)
_MAPPED = ("import sys; sys.path.insert(0, sys.argv[1]); import restack as R\n"
           "x, m = sys.argv[2:4]\n"
           "R.SCRIPT, R.SCRIPT_ARGS = R.move_line(m, None)\n"
           "base = R.move_base(m, None)\n"
           "own = getattr(R, 'own_map', lambda *a: None)(m, base, R.SCRIPT_ARGS, None)\n"
           "tree = R.run_script(x, base, sys.executable, None, *([(m, own)] if own else []))[0]\n"
           "if own or getattr(R, 'hand_of', lambda *a: [])(m, None):\n"
           "    on_base = R.commit_tree(R.check_generated(m, base, sys.executable, None), [], 'the script on the base', None)\n"
           "    tree, files = R.merge_trees(R.commit_tree(tree, [x], 'the script', None), m, None, base=on_base)\n"
           "    if files:\n"
           "        sys.exit('its hand changes conflict with the head in ' + ', '.join(files))\n"
           "print('mapped', tree)\n")
MAPPED_RE = re.compile(r"^mapped ([0-9a-f]{40,64})$", re.M)
MAPPED_TIMEOUT_S = 1800.0
#: what a run of a tree's own code never gets (`tree_env`): the queue's GitHub and Slack credentials, any of which can push
SCRIPT_UNSET = ("GH_TOKEN", "GITHUB_TOKEN", "GH_CONFIG_DIR", "SLACK_BOT_TOKEN")
STORE_UNSET = ("AWS_", "R2_", "RESEARCH_STORE", "RESEARCH_EXPORT_", "CLOUDFLARE_", "STORE_")
UNLANDED = "no land verdict of its own counts, and a move's script runs a tree's own code, so it runs only on a head a reviewer passed"


def tree_env(env: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in env.items() if k not in SCRIPT_UNSET and not k.startswith(STORE_UNSET)}


def scratch() -> tempfile.TemporaryDirectory:
    root = Path(MERGED_ROOT or tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=f"{MERGED_PREFIX}{os.getpid()}-", dir=root, ignore_cleanup_errors=True)


def mapped_tree(repo: Path, commit: str, move: str, restack: Path,
                timeout: float = MAPPED_TIMEOUT_S, env: dict[str, str] | None = None) -> tuple[str | None, str]:
    with scratch() as d:
        rc, out = run_group([sys.executable, "-c", _MAPPED, str(restack), commit, move], repo, timeout,
                            env={**tree_env({**_script_env(), **(env or {})}), "TMPDIR": d,
                                 "GH_CONFIG_DIR": str(Path(d, "no-gh"))})
        left = any(Path(d).iterdir())
    if left:
        with open(Path(repo, _out(repo, "rev-parse", "--git-common-dir")) / WORKTREES_LOCK, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            _git(repo, "worktree", "prune", check=False)
    if rc is None:
        return None, f"ran over {timeout:.0f} s"
    got = MAPPED_RE.findall(out)
    return (got[-1], "") if rc == 0 and got else (None, f"exit {rc}: {' | '.join(out.strip().splitlines()[-3:])}")


def _own_onto(repo: Path, onto: str, commit: str) -> str | None:
    mb = _git(repo, "merge-base", onto, commit, check=False).stdout.strip()
    return own_change(repo, mb, commit) if mb else None


def mapped_change(repo: Path, head: str, onto: str, moves: list[tuple[str, list[str]]], restack: Path,
                  env: dict[str, str] | None = None) -> tuple[str | None, str]:
    x = head
    for i, (move, parents) in enumerate(moves):
        if len(parents) != 1:
            return None, f"the move {move[:12]} has {len(parents)} parents"
        base = parents[0]
        if late := next((mb for mb in _git(repo, "merge-base", "--all", x, onto, check=False).stdout.split()
                         if _git(repo, "merge-base", "--is-ancestor", mb, base, check=False).returncode), None):
            if i < len(moves) - 1:
                continue
            return None, (f"it has {late[:12]}, which came after {base[:12]}, the base of the move {move[:12]}: restack.py "
                          "refuses it (based-after)")
        if _git(repo, "merge-base", "--is-ancestor", base, x, check=False).returncode:
            before = _own_onto(repo, onto, x)
            r = _git(repo, "merge-tree", "--write-tree", "--name-only", "--no-messages", "-z", x, base, check=False, env=env)
            got = [s for s in r.stdout.split("\0") if s]
            if r.returncode or not got:
                return None, (f"it doesn't merge with {base[:12]}, the base of the move {move[:12]}: "
                              f"{', '.join(got[1:4]) or (r.stderr or '').strip()[-200:]}")
            x = _git(repo, "commit-tree", got[0], "-p", x, "-p", base, "-m", f"Merge {base}", env=IDENTITY).stdout.strip()
            if _own_onto(repo, onto, x) != before:
                return None, f"its merge with {base[:12]}, the base of the move {move[:12]}, changes its own change"
        tree, why = mapped_tree(repo, x, move, restack, env=env)
        if tree is None:
            return None, f"the move {move[:12]}'s script fails on it: {why}"
        x = _git(repo, "commit-tree", tree, "-p", x, "-p", move, "-m", f"Mapped across {move}", env=IDENTITY).stdout.strip()
    return _own_onto(repo, onto, x), ""


def move_lines_at(repo: Path, at: str) -> dict[str, dict]:
    r = _git(repo, "show", f"{at}:{MOVE_LINES}", check=False)
    try:
        data = tomllib.loads(r.stdout) if not r.returncode else {}
    except tomllib.TOMLDecodeError as e:
        print(f"research queue: {MOVE_LINES} at {at[:12]} doesn't parse ({e}); its corrections are left out, and restack.py refuses the table itself", file=sys.stderr)
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def moves_between(repo: Path, onto: str, head: str) -> list[tuple[str, list[str]]]:
    lines = move_lines_at(repo, onto)
    log = _git(repo, "log", "--topo-order", "--reverse", "--no-merges", "--format=%H %P%x00%B%x1e", onto, f"^{head}",
               check=False).stdout
    found = []
    for rec in filter(None, (s.strip() for s in log.split("\x1e"))):
        line, _, body = rec.partition("\x00")
        commit, *parents = line.split()
        entry = lines.get(commit, {})
        if entry.get("move") is not False and MOVE_LINE_RE.search(str(entry.get("line") or body)):
            found.append((commit, parents))
    return found


@dataclass(frozen=True)
class Retired:
    names: frozenset[str] = frozenset()
    words: frozenset[str] = frozenset()

    def named(self, text: str) -> list[str]:
        low = text.lower()
        return sorted({n for n in self.names if n in text} | {w for w in self.words if w in low})


def map_retires(repo: Path, at: str, path: str) -> Retired | None:
    r = _git(repo, "show", f"{at}:{path}", check=False)
    try:
        data = tomllib.loads(r.stdout) if not r.returncode else None
    except tomllib.TOMLDecodeError as e:
        print(f"research queue: the move map {path} at {at[:12]} doesn't parse ({e}), so a head that lacks its move "
              "stays out", file=sys.stderr)
        data = None
    if isinstance(data, dict) and not isinstance(data.get("words_in") or [], list):
        print(f"research queue: the move map {path} at {at[:12]} holds a words_in that is no array of tables, so a head "
              "that lacks its move stays out", file=sys.stderr)
        data = None
    if not isinstance(data, dict):
        return None

    def table(key: str, of: dict = data) -> dict:
        return t if isinstance(t := of.get(key), dict) else {}

    paths = {o for o, n in table("paths").items() if o != n}
    paths |= {p for k in ("retired", "drop") if isinstance(ps := table(k).get("paths"), list) for p in ps if isinstance(p, str)}
    modules = {o for k in ("modules", "aliases") for o, n in table(k).items() if o != n and "." in o}
    words = {*table("words"), *(w for s in data.get("words_in") or [] if isinstance(s, dict) for w in table("words", s))}
    return Retired(frozenset(p.rstrip("/") for p in paths if p.strip("/")) | frozenset(modules),
                   frozenset(w.lower() for w in words if w))


def lacked_moves(repo: Path, main: str, head: str, files: Iterable[str] | None = None,
                 maps: dict[tuple[str, str], Retired | None] | None = None) -> list[tuple[str, str, list[str] | None]]:
    if not (moves := moves_between(repo, main, head)):
        return []
    maps = {} if maps is None else maps
    text = own_text(repo, main, head, files)
    out = []
    for commit, parents in moves:
        if not (m := MOVE_LINE_RE.search(_out(repo, "log", "-1", "--format=%B", commit))):
            continue
        if (key := (parents[0] if parents else f"{commit}^", m[1])) not in maps:
            maps[key] = map_retires(repo, *key)
        if (got := maps[key]) is None or (names := got.named(text)):
            out.append((commit, m[1], None if got is None else names))
    return out


def own_text(repo: Path, main: str, head: str, files: Iterable[str] | None = None) -> str:
    files = own_files(repo, main, head) if files is None else files
    diff = _out(repo, "diff", "-U0", "--no-color", "--no-ext-diff", "--no-renames", f"{main}...{head}")
    return "\n".join([*sorted(files), *(x for x in diff.splitlines() if x.startswith("+") and not x.startswith("+++"))])


def crossed(repo: Path, main: str, head: str, moves: Iterable[tuple[str, str]], files: Iterable[str] | None = None,
            maps: dict[tuple[str, str], Retired | None] | None = None) -> frozenset[str]:
    maps = {} if maps is None else maps
    text = own_text(repo, main, head, files)

    def read(at: str, path: str) -> Retired | None:
        if (at, path) not in maps:
            maps[at, path] = map_retires(repo, at, path)
        return maps[at, path]

    out = set()
    for move, path in moves:
        if (got := read(move, path)) is None:
            continue
        if (also := read(main, path)) is not None:
            got = Retired(got.names | also.names, got.words | also.words)
        if not got.named(text):
            out.add(move)
    return frozenset(out)


def lacks_why(lacked: list[tuple[str, str, list[str] | None]]) -> str:
    def said(names: list[str] | None) -> str:
        if names is None:
            return "whose map can't be read at its base, so what its own change names of it can't be told"
        shown = ", ".join(f"`{x}`" for x in names[:5]) + (f" and {len(names) - 5} more" if len(names) > 5 else "")
        return f"whose retired names its own change names ({shown})"

    moves = "; and ".join(f"the move {c[:12]} `main` landed ({m}), {said(ns)}" for c, m, ns in lacked)
    return (f"it lacks {moves}: merge `main` into it and rename them, or cross it with `tools/move/restack.py`, and mark the "
            "new head ready")


def merge_main(pr: str, branch: str | None) -> str:
    return (f"merge `main` into {pr} where no move `main` landed lies between; where one does (a commit with a `Move:` or "
            f"`Layout-move:` line that {pr} lacks), a plain merge is clean and wrong, since the move's script never runs on "
            f"its own lines and the files it adds stay at their old paths: carry it against the move's own commit first, "
            f"`python3 tools/move/restack.py {branch or 'BRANCH'} --main MOVE`, then merge today's `main` into the carried result, a merge inside the new layout, and check that the "
            "merge's tree holds its own files at the new paths")


def held_below(entries: dict[int, Entry], n: int, held: Callable[[int], str | None],
               seen: frozenset[int] = frozenset()) -> tuple[int, str] | None:
    for a in entries[n].after:
        if a in seen or a not in entries:
            continue
        e = entries[a]
        if hold := e.why if e.why.startswith("held (") else held(a) if e.state == "ready" else None:
            return a, hold
        if got := held_below(entries, a, held, seen | {n}):
            return got
    return None


def held_waits(a: int, head: str, hold: str) -> str:
    label = re.match(r"held \(`(.+)` label\)", hold)
    release = (f"the `{label[1]}` label comes off #{a}" if label else
               f"#{a}'s holder gives that head a `land` verdict (`research queue review {a} --land`)")
    return (f"it waits on #{a}'s head {head[:12]}, which is {hold}: it is taken once {release}, or once it no longer holds "
            f"#{a}'s commits")


def mapped_key(head: str, moves: list[tuple[str, list[str]]]) -> str:
    return f"{head} {hashlib.sha512(' '.join(m for m, _ in moves).encode()).hexdigest()}"


def map_one(repo: Path, head: str, onto: str, moves: list[tuple[str, list[str]]],
            env: dict[str, str] | None = None) -> tuple[str | None, str]:
    try:
        with scratch() as d:
            restack = write_scripts(repo, restack_files(repo, onto), Path(d)) / MOVE_DIR
            return mapped_change(repo, head, onto, moves, restack, env)
    except (QueueError, OSError, subprocess.SubprocessError) as e:
        return None, f"{type(e).__name__}: {e}"


UNMAPPED = "it has no own change once mapped"
MAP_JOB = "import sys; sys.path.insert(0, sys.argv[1]); from research.queue import map_job; map_job(*sys.argv[2:])"
MAPPED_SUFFIX, MAPPING_MARK = ".json", ".running"
MAPPED_SLACK_S = 600.0


def mapped_name(at: str) -> str:
    return f"{at[:12]}-{hashlib.sha512(at.encode()).hexdigest()[:16]}"


def _mapped_lost(moves: list[tuple[str, list[str]]]) -> float:
    return len(moves) * MAPPED_TIMEOUT_S + MAPPED_SLACK_S


def write_mapped(path: Path, at: str, got: str | None, why: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({at: got if got is not None else {"refused": why or UNMAPPED}}) + "\n")
    tmp.replace(path)


def map_job(repo: str, head: str, onto: str, moves: str, out: str, root: str) -> None:
    global MERGED_ROOT
    MERGED_ROOT = Path(root) if root else None
    crossed = [(m, list(parents)) for m, parents in json.loads(moves)]
    write_mapped(Path(out), mapped_key(head, crossed), *map_one(Path(repo), head, onto, crossed))


def map_moves(repo: Path, heads: Iterable[str], main: str, jobs: int,
              refuse: Callable[[str], str | None]) -> dict[str, tuple[str, str | None, str]]:
    todo = {h: moves for h in heads if (moves := moves_between(repo, main, h))}
    refused = {h: why for h in todo if (why := refuse(h))}
    with concurrent.futures.ThreadPoolExecutor(max(1, jobs)) as pool:
        got = {h: pool.submit(map_one, repo, h, main, moves) for h, moves in todo.items() if h not in refused}
    return {h: (mapped_key(h, todo[h]), *(got[h].result() if h in got else (None, refused[h]))) for h in todo}


CROSS_REPORT, CROSS_BUNDLE, CROSS_MAPPED, CROSS_PUSHED = "cross.json", "crossed.bundle", "mapped-changes.json", "pushed.json"
CROSS_NEW, CROSS_PR = "refs/cross/new/", "refs/cross/pr/"
# : `research queue cross` never crosses a head with no land verdict of its own: `restack.py` runs the head's own code.
CROSS_UNLANDED = ("no land verdict of its own counts, so its mark wouldn't carry, and restack.py runs a head's own code, which "
                  "runs only on a head a reviewer passed: its owner restacks it")
LEASE_RE = re.compile(r"\(stale info\)")
CROSS_NEVER = ("main", "checked", "next")
CROSS_KEPT = True


def _commit(repo: Path, sha: str) -> bool:
    return not _git(repo, "cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode


def cross_drivers(repo: Path, main: str, into: Path) -> dict[str, str]:
    drivers: dict[str, str] = {}
    for line in _git(repo, "config", "--get-regexp", r"^merge\..*\.driver$", check=False).stdout.splitlines():
        key, _, cmd = line.partition(" ")
        drivers[key] = cmd
    got = []
    for i, (key, cmd) in enumerate(drivers.items()):
        words = cmd.split(" ")
        if len(words) < 2 or words[0] != "python3" or os.path.isabs(words[1]):
            continue
        if (src := _git(repo, "show", f"{main}:{words[1]}", check=False)).returncode:
            continue
        (f := into / str(i) / Path(words[1]).name).parent.mkdir(parents=True)
        f.write_text(src.stdout, errors="surrogateescape")
        got.append((key, " ".join(["python3", str(f), *words[2:]])))
    quote = lambda s: "'" + s.replace("'", "'\\''") + "'"  # noqa: E731
    return {"GIT_CONFIG_PARAMETERS": " ".join(f"{quote(k)}={quote(v)}" for k, v in got)} if got else {}


def restack_why(x: dict) -> str:
    if x.get("status") == "conflicted":
        return f"{RESTACK} conflicts at its {x.get('step')} merge in {', '.join(x.get('files') or [])}"
    return f"{RESTACK}: {x.get('status')}: {((x.get('error') or '').splitlines() or [''])[0][:300]}"


def named_moves(repo: Path, head: str, main: str, moves: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    shared, out = _git(repo, "merge-base", "--all", head, main, check=False).stdout.split(), []
    for i, (move, parents) in enumerate(moves):
        later = [q[0] for _, q in moves[i + 1:] if len(q) == 1]
        if (len(parents) == 1 and later and _out(repo, "rev-list", "-n1", *shared, f"^{parents[0]}")
                and not _out(repo, "rev-list", "-n1", move, "--not", *later)):
            continue
        out.append((move, parents))
        shared.append(move)
    return out


def cross_one(repo: Path, head: str, main: str, moves: list[tuple[str, list[str]]], onto: str | None = None,
              kept: bool = False, src: str | None = None, drivers: dict[str, str] | None = None) -> dict:
    with scratch() as d:
        try:
            restack = write_scripts(repo, restack_files(repo, main), Path(d)) / RESTACK
        except (QueueError, OSError) as e:
            return {"why": f"{type(e).__name__}: {e}"}
        if not restack.is_file():
            return {"why": f"{main[:12]} has no {RESTACK}"}
        env = tree_env(_script_env())
        named = named_moves(repo, head, main, moves)
        argv = [sys.executable, str(restack), head, *(a for m, _ in named for a in ("--move-commit", m)), "--onto", onto or main,
                "--main", main, "--json", *(("--kept",) if kept else ())]
        rc, out = run_group(argv, repo, RESTACK_WAIT_S,
                            env={**env, **(drivers or {}), "TMPDIR": d, "GH_CONFIG_DIR": str(Path(d, "no-gh"))})
    if rc is None:
        return {"why": f"{RESTACK} ran over {RESTACK_WAIT_S // 60} min"}
    try:
        (x,) = json.JSONDecoder().raw_decode(out.lstrip())[0]
        status = x["status"]
    except (ValueError, TypeError, KeyError):
        return {"why": f"{RESTACK} exit {rc}: {' | '.join(out.strip().splitlines()[-3:])}"}
    if status != "clean" or rc:
        return {"why": restack_why(x) if status != "clean" else f"{RESTACK} said clean, but exited {rc}", "status": status}
    new, src = x["result"], src or head
    base = _git(repo, "merge-base", main, new, check=False).stdout.strip()
    crossed = moves_between(repo, base, src) if base else []
    own = own_change(repo, base, new) if base else None
    mapped, why = map_one(repo, src, base, crossed, drivers) if crossed else (None, f"{new[:12]} crosses no move of {main[:12]}")
    if own is None or own != mapped:
        whose = "its old one" if src == head else f"that of {src[:12]}, whose verdict it takes,"
        return {"new": new, "why": f"{RESTACK} crossed it clean as {new[:12]}, but its own change there ({(own or 'none')[:12]}) "
                                   f"isn't {whose} mapped across the moves ({(mapped or why)[:300]}), so no mark or verdict "
                                   "would carry"}
    return {"new": new, "own": own, "key": mapped_key(src, crossed)}


def cross_plan(repo: Path, prs: Iterable[PR], main: str, named: list[str], ready: Callable[[int, str], bool],
               verdict: Callable[[int, str], str | None], only: Collection[int] = (),
               slug: str = REPO) -> tuple[dict[int, dict], list[dict]]:
    def has(a: str, b: str) -> bool:
        return not _git(repo, "merge-base", "--is-ancestor", a, b, check=False).returncode

    opened = {p.number: p for p in prs if p.base == "main"}
    todo: dict[int, dict] = {}
    skipped: list[dict] = []
    off: dict[int, str] = {}

    def skip(p: PR, kind: str, why: str) -> None:
        off[p.number] = why
        if not only or p.number in only:
            skipped.append({"pr": p.number, "head": p.head, "kind": kind, "why": why})

    for n, p in sorted(opened.items()):
        found = [] if p.outside(slug) or not _commit(repo, p.head) else moves_between(repo, main, p.head)
        moves = [mv for mv in found if mv[0] in named] if named else found
        if (why := p.outside(slug)) is not None:
            skip(p, "outside", why)
        elif not _commit(repo, p.head):
            skip(p, "missing", "its head can't be fetched")
        elif not moves:
            skip(p, "already crossed" if named else "no moves to cross",
                 "it has the moves" if named else f"it lacks no move {main[:12]} has")
        elif only and n not in only:
            off[n] = "not asked for (--only)"
        elif not ready(n, p.head):
            skip(p, "not marked", "not marked ready at its head")
        elif (src := verdict(n, p.head)) is None:
            skip(p, "no land verdict", CROSS_UNLANDED)
        else:
            todo[n] = {"pr": n, "branch": p.branch, "old": p.head, "src": src, "moves": moves, "below": None}
    off_main = {n: set(_out(repo, "rev-list", t["old"], "--not", main).split()) for n, t in todo.items()}
    holds = {n: [m for m, q in opened.items() if m != n and q.head != todo[n]["old"] and q.head in off_main[n]] for n in todo}
    while gone := [n for n in todo if any(m not in todo for m in holds[n])]:
        for n in gone:
            m = next(m for m in holds[n] if m not in todo)
            skip(opened[n], "stack", f"it holds #{m}'s head, which this run doesn't cross: {off.get(m, 'refused')}")
            del todo[n]
    for n in list(todo):
        if below := holds[n]:
            top = [m for m in below if all(has(opened[k].head, opened[m].head) for k in below)]
            if top:
                todo[n]["below"] = top[0]
            else:
                skip(opened[n], "stack", f"it holds the heads of {', '.join(f'#{m}' for m in sorted(below))}, which aren't one "
                                         "stack")
                del todo[n]
    while gone := [n for n, t in todo.items() if t["below"] is not None and t["below"] not in todo]:
        for n in gone:
            skip(opened[n], "stack", f"it holds #{todo[n]['below']}'s head, which this run doesn't cross: "
                                     f"{off.get(todo[n]['below'], 'refused')}")
            del todo[n]
    return todo, skipped


def cross_all(repo: Path, todo: dict[int, dict], main: str, jobs: int, kept: bool = False,
              log: Callable[[str], None] = print, root: Callable[[str], dict[str, str] | str] | None = None,
              drivers: dict[str, str] | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    above: dict[int | None, list[int]] = {}
    for n, t in sorted(todo.items()):
        above.setdefault(t["below"], []).append(n)
    if kept and jobs > 1 and todo:
        moves = max((t["moves"] for t in todo.values()), key=len)
        got = cross_one(repo, moves[0][1][0], main, moves, kept=True, drivers=drivers)
        log(f"research queue cross: the moves' checks kept, restacking their base {moves[0][1][0][:12]}: "
            f"{got.get('why', 'clean')}")
    base: dict[str, concurrent.futures.Future] = {}

    def gate(n: int, got: dict) -> dict:
        if root is None or "own" not in got:
            return got
        new = got["new"]
        if isinstance(fails := root(new), str):
            got = {"why": f"crossed clean as {new[:12]}, but the root suite didn't run on the crossed tree: {fails}", "new": new}
        elif bad := [t for t in fails if t not in (b if isinstance(b := base["main"].result(), dict) else {})]:
            got = {"why": f"crossed clean as {new[:12]}, but the root suite fails on the crossed tree: {bad[0]}"
                          + (f" (and {len(bad) - 1} more)" if len(bad) > 1 else ""), "new": new, "root": bad}
        log(f"research queue cross: #{n}: the root suite on {new[:12]}: {got.get('why', 'passes')}")
        return got

    def stack(n: int, onto: str | None = None) -> list[tuple[int, dict]]:
        got = cross_one(repo, todo[n]["old"], main, todo[n]["moves"], onto=onto, kept=kept, src=todo[n].get("src"),
                        drivers=drivers)
        log(f"research queue cross: #{n} at {todo[n]['old'][:12]}: " + (f"crossed as {got['new'][:12]}" if "own" in got
                                                                         else got["why"]))
        got = gate(n, got)
        if onto is not None and got.get("status") == "based-after":
            got = {"stack": f"{RESTACK} won't cross it onto #{todo[n]['below']}'s crossing, whose old head it holds (based-after), "
                            "so its owners restack the stack"}
        out = [(n, got)]
        for m in above.get(n, []):
            out += stack(m, got["new"]) if "own" in got else [(k, {"under": n}) for k in _stacked(above, m)]
        return out

    with concurrent.futures.ThreadPoolExecutor(max(1, jobs)) as pool:
        if root is not None and todo:
            base["main"] = pool.submit(root, main)
            base["main"].add_done_callback(lambda f: log(
                f"research queue cross: the root suite on main {main[:12]}: " + (
                    f"didn't run ({r}), so every failure on a crossed tree counts" if isinstance(r := f.result(), str)
                    else f"fails {', '.join(r)}, which no crossing is refused for" if r else "passes")))
        done = [x for f in [pool.submit(stack, n) for n in above.get(None, [])] for x in f.result()]
    crossed, refused, skipped = [], [], []
    for n, got in sorted(done, key=lambda x: x[0]):
        t = {k: todo[n].get(k) for k in ("pr", "branch", "old", "src", "below")}
        if "own" in got:
            crossed.append({**t, "new": got["new"], "own": got["own"], "key": got["key"]})
        elif "under" in got or "stack" in got:
            skipped.append({"pr": n, "head": t["old"], "kind": "stack",
                            "why": got.get("stack") or f"#{got['under']} below it in its stack doesn't cross"})
        else:
            refused.append({"pr": n, "old": t["old"], "why": got["why"], **({"new": got["new"]} if "new" in got else {}),
                            **({"root": got["root"]} if "root" in got else {})})
    return crossed, refused, skipped


def _stacked(above: dict[int | None, list[int]], n: int) -> list[int]:
    return [n, *(k for m in above.get(n, []) for k in _stacked(above, m))]


def cross_table(report: dict) -> str:
    kinds: dict[str, int] = {}
    for s in report["skipped"]:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    lines = [f"research queue cross: main {report['main'][:12]}, moves {' '.join(m[:12] for m in report['moves']) or 'none'}",
             f"crossed {len(report['crossed'])}, refused {len(report['refused'])}, skipped {len(report['skipped'])}"
             + (f" ({', '.join(f'{k} {v}' for k, v in sorted(kinds.items()))})" if kinds else "")]
    lines += [f"  crossed #{c['pr']} {c['old'][:12]} -> {c['new'][:12]}"
              + (f" onto #{c['below']}'s" if c.get("below") else "") + "  carries its mark and verdict"
              for c in report["crossed"]]
    lines += [f"  refused #{r['pr']} {r['old'][:12]}: {r['why']}" for r in report["refused"]]
    lines += [f"  skipped #{s['pr']} {s['head'][:12]}: {s['why']}" for s in report["skipped"] if s["kind"] == "stack"]
    lines += [f"  pushed #{n}: {said}" for n, said in sorted(report.get("pushed", {}).items(), key=lambda x: int(x[0]))]
    return "\n".join(lines)


def cross_write(repo: Path, out: Path, report: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / CROSS_REPORT).write_text(json.dumps(report, indent=1) + "\n")
    (out / CROSS_MAPPED).write_text(json.dumps({c["key"]: c["own"] for c in report["crossed"]}) + "\n")
    if report["crossed"]:
        for c in report["crossed"]:
            _git(repo, "update-ref", f"{CROSS_NEW}{c['pr']}", c["new"])
        _git(repo, "bundle", "create", "-q", str(out / CROSS_BUNDLE), *(f"{CROSS_NEW}{c['pr']}" for c in report["crossed"]),
             "--not", report["main"], *(c["old"] for c in report["crossed"]))


def cross_push(repo: Path, report: dict, remote: str, opened: dict[int, PR] | None = None,
               log: Callable[[str], None] = print) -> dict[str, str]:
    def has(a: str, b: str) -> bool:
        return not _git(repo, "merge-base", "--is-ancestor", a, b, check=False).returncode

    said: dict[str, str] = {}
    for c in sorted(report["crossed"], key=lambda c: _depth(report["crossed"], c)):
        n, old, new, branch = c["pr"], c["old"], c["new"], c["branch"]
        p = (opened or {}).get(n)
        if opened is not None and (p is None or p.base != "main"):
            said[str(n)] = "not pushed: it isn't open into `main` any more"
        elif p is not None and p.head != old:
            said[str(n)] = f"not pushed: owner pushed since ({p.head[:12]})"
        elif c.get("below") is not None and not said.get(str(c["below"]), "").startswith("pushed"):
            said[str(n)] = f"not pushed: #{c['below']} below it in its stack wasn't"
        elif not branch or branch in CROSS_NEVER or branch.startswith("refs/"):
            said[str(n)] = f"not pushed: its PR names {f'the branch {branch}' if branch else 'no branch'}, never pushed"
        elif not (has(old, new) and has(report["main"], new) and all(has(m, new) for m in report["moves"])) or own_change(
                repo, _git(repo, "merge-base", report["main"], new, check=False).stdout.strip() or report["main"], new) != c["own"]:
            said[str(n)] = f"not pushed: {new[:12]} isn't the crossing the report describes"
        else:
            r = _git(repo, *STALL, "push", f"--force-with-lease=refs/heads/{branch}:{old}", remote, f"{new}:refs/heads/{branch}",
                     check=False)
            said[str(n)] = (f"pushed {new[:12]} to {branch}" if not r.returncode else "not pushed: owner pushed since"
                            if LEASE_RE.search(r.stderr) else f"not pushed: {(r.stderr or r.stdout).strip()[-300:]}")
        log(f"research queue cross: #{n}: {said[str(n)]}")
    return said


def _depth(crossed: list[dict], c: dict) -> int:
    by = {x["pr"]: x for x in crossed}
    d, below = 0, c.get("below")
    while below is not None and below in by:
        d, below = d + 1, by[below].get("below")
    return d


PREFLIGHT_TEST_PATHS = ("tests/test_module_strings.py", "tests/test_pins.py", "tests/test_doc_paths.py",
                        "tests/test_repository.py", "tests/tools/research/test_pythonpath.py")


def _by_suite(paths: tuple[str, ...]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    groups: dict[str, list[str]] = {}
    for path in paths:
        d, sep, rest = path.rpartition("tests/")
        groups.setdefault(d.rstrip("/") or ".", []).append(sep + rest)
    return tuple((d, tuple(fs)) for d, fs in groups.items())


def reaches(paths: Iterable[str], reach: Iterable[str]) -> bool:
    return any(f.startswith(r) or f"/{r}" in f if r.endswith("/") else f == r for f in paths for r in reach)


PREFLIGHT_TESTS = _by_suite(PREFLIGHT_TEST_PATHS)
PREFLIGHT_LINTS = (("tests/test_no_wall_clock.py", ("tests/",)), ("tests/adapters/vllm/tests/lint", ("adapters/vllm/",)))
#: config and noqa comments), the repository's lints and lists, the root ruff.toml, and the root pyproject.toml's ruff pin.
PREFLIGHT_RATCHET = ("adapters/vllm/tools/ratchet_gate.py", ("adapters/vllm/", "tests/lint/", "ruff.toml", "pyproject.toml"))
PREFLIGHT_UNCOLLECTED: dict[str, str] = {}
PREFLIGHT_SYNC = ("uv", "sync", "--locked", "--all-packages", "--extra", "torch-cpu", "-q")
PREFLIGHT_CHECKS_S = 120.0
PREFLIGHT_COLLIDE_S = 30.0
PREFLIGHT_COLLIDE_NAMED = 5
PREFLIGHT_JOBS = 2
# : Below this share of available memory no suite is collected, so a mark never OOMs the VM's other agents.
PREFLIGHT_MIN_FREE = 0.25
PREFLIGHT_LEAN_MODULES = 40
PREFLIGHT_LEAN_S = 180.0
PREFLIGHT_LEAN_ON = ("vy-nebius-cpu-9", "vy-nebius-cpu-5")
PREFLIGHT_LEAN_POOL: str | None = None
LEAN_BUILDER, LEAN_BUILD_LINE = "tools/verity/lean/lean_changed.py", "lean-build: "
LEAN_SLOTS = Path(os.environ.get("LEAN_SLOTS", "/workspace/research/locks/lean-slots"))
LEAN_FIXTURES = "tools/verity/lean/"
MERGED_PREFIX = "research-merged-"
MERGED_ROOT: Path | None = None
_TOUCHED = ("import json, sys; sys.path.insert(0, 'tools/check'); import suites as S; pkgs = S.members(); "
            "chosen = S.touched(S.discover(), pkgs, sys.argv[1], sys.argv[2], S.locked()); "
            "ins = {s: S.inputs_of(s, pkgs) for s in chosen}; own = {s: [p for p in ps if p != '.'] for s, ps in ins.items()}; "
            "fx = S.fetched_fixtures(sys.argv[2], {p for ps in own.values() for p in ps}); "
            "print(json.dumps({'suites': [s.path for s in chosen], 'fixtures': {s.path: sorted(p for p in fx "
            "if S.covered(p, own[s], S.dropped(s, ins[s]))) for s in chosen}}))")
PREFLIGHT_FETCH_S = 45.0
def free_memory() -> float | None:
    try:
        info = dict(ln.split(":", 1) for ln in Path("/proc/meminfo").read_text().splitlines() if ":" in ln)
        return int(info["MemAvailable"].split()[0]) / int(info["MemTotal"].split()[0])
    except (OSError, KeyError, ValueError, ZeroDivisionError):
        return None


_SUMMARY_RE = re.compile(r"^(?:FAILED|ERROR) (.+?)(?: - (.*))?$", re.M)
_LOCK_STALE = "needs to be updated"


class Ran(str):
    pass


@dataclass
class Touched:
    suites: list[str] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    tail: str = ""
    fixtures: dict[str, list[str]] = field(default_factory=dict)


def failed_row(suite: str, failures: dict[str, str]) -> dict:
    return {"name": suite, "suite": suite, "status": "failed", "failed_tests": list(failures), "log": None,
            "summary": list(failures.values())}


def pytest_failures(run: Callable[[list[str], Path], tuple[subprocess.CompletedProcess | None, str]],
                    py: str | tuple[str, ...], cwd: Path, args: list[str]) -> dict[str, str] | str:
    r, why = run([*((py,) if isinstance(py, str) else py), "-P", "-m", "pytest", "-q", "-p", "no:cacheprovider", *args], cwd)
    if r is None:
        return why
    if r.returncode in (0, 5):
        return {}
    found = {m.group(1): m.group(0) for m in _SUMMARY_RE.finditer(r.stdout)}
    return found or {f"exit {r.returncode}": f"pytest exited {r.returncode}: {(r.stdout + r.stderr).strip()[-300:]}"}


@contextlib.contextmanager
def worktrees_locked(repo: Path):
    common = Path(_out(repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    with open(common / "train.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        with open(common / WORKTREES_LOCK, "a") as w:
            fcntl.flock(w, fcntl.LOCK_EX)
            yield


@contextlib.contextmanager
def merged_worktree(repo: Path, commit: str, root: Path | None = None):
    root = Path(root or MERGED_ROOT or tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    for d in root.glob(f"{MERGED_PREFIX}*"):
        pid = d.name.removeprefix(MERGED_PREFIX).split("-", 1)[0]
        if pid.isdigit() and int(pid) != os.getpid() and not Path(f"/proc/{pid}").exists():
            shutil.rmtree(d, ignore_errors=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"{MERGED_PREFIX}{os.getpid()}-", dir=root))
    wt = tmp / "tree"
    try:
        with worktrees_locked(repo):
            _git(repo, "worktree", "prune", check=False)
            _git(repo, "worktree", "add", "-q", "--detach", str(wt), commit)
        yield wt
    finally:
        with worktrees_locked(repo):
            _git(repo, "worktree", "remove", "--force", str(wt), check=False)
            shutil.rmtree(tmp, ignore_errors=True)
            _git(repo, "worktree", "prune", check=False)


CROSS_ROOT_SUITE = ["tests", "-m", "not slow", "-n", "4"]
CROSS_ROOT_WAIT_S = 30 * 60
CROSS_ROOT_SYNC: tuple[str, ...] | None = PREFLIGHT_SYNC
CROSS_ROOT_PYTHON: str | None = None


def root_failures(repo: Path, commit: str, sync: tuple[str, ...] | None = None,
                  python: str | None = None) -> dict[str, str] | str:
    env = tree_env(git_env())

    def run(argv: list[str], cwd: Path) -> tuple[subprocess.CompletedProcess | None, str]:
        rc, out = run_group(argv, cwd, CROSS_ROOT_WAIT_S, env=env)
        if rc is None:
            return None, f"`{' '.join(argv[:4])}` ran over {CROSS_ROOT_WAIT_S // 60} min"
        return subprocess.CompletedProcess(argv, rc, out, ""), ""

    try:
        with merged_worktree(repo, commit) as wt:
            if not (wt / "tests").is_dir():
                return {}
            if sync is not None:
                r, why = run(list(sync), wt)
                if r is None or r.returncode:
                    return why or f"`{' '.join(sync)}` exited {r.returncode}: {r.stdout.strip()[-300:]}"
            return pytest_failures(run, python or str(wt / ".venv" / "bin" / "python"), wt, root_suite(wt))
    except (QueueError, OSError, subprocess.SubprocessError) as e:
        return f"{type(e).__name__}: {e}"


def root_suite(wt: Path) -> list[str]:
    under = sorted(d.relative_to(wt).as_posix() for d in (wt / suite_layout.MIRROR).iterdir() if d.is_dir())
    return [*CROSS_ROOT_SUITE, *(f"--ignore={d}" for d in under if not suite_layout.root_owns(d + "/"))]


@dataclass
class LocalSuites:
    python: str
    run: Callable[[list[str], Path], tuple[subprocess.CompletedProcess | None, str]]
    jobs: int = PREFLIGHT_JOBS
    free: Callable[[], float | None] = free_memory
    suites: list[str] | None = None
    args: tuple[str, ...] = ("--collect-only",)
    fetch: Callable[[Path, list[str]], tuple[set[str], str]] | None = None
    fixtures: dict[str, list[str]] = field(default_factory=dict)

    def __call__(self, wt: Path, commit: str, base: str) -> Touched:
        out = Touched(fixtures=dict(self.fixtures))
        suites = self.suites
        if suites is None:
            r, why = self.run([self.python, "-P", "-c", _TOUCHED, base, _out(wt, "rev-parse", f"{commit}^{{tree}}")], wt)
            try:
                said = json.loads(r.stdout.strip().splitlines()[-1]) if r is not None and r.returncode == 0 else None
                suites, out.fixtures = (said["suites"], said["fixtures"]) if said is not None else (None, {})
            except (ValueError, IndexError, KeyError, TypeError):
                suites = None
            if suites is None:
                out.notes.append(f"no suite collected: `suites.py` couldn't say which the change touches "
                                 f"({why or (r.stderr or r.stdout).strip()[-200:]})")
                suites = []
            out.notes += [f"{s} not collected: {PREFLIGHT_UNCOLLECTED[s]}" for s in suites if s in PREFLIGHT_UNCOLLECTED]
            suites = [s for s in suites if s not in PREFLIGHT_UNCOLLECTED]
        todo = [s for s in suites if (wt / s).is_dir()]
        if todo and (f := self.free()) is not None and f < PREFLIGHT_MIN_FREE:
            out.notes.append(f"no suite collected: {f:.0%} of memory available, under {PREFLIGHT_MIN_FREE:.0%}; its candidate's "
                             "check still collects them")
            todo = []
        need = sorted({p for s in todo for p in out.fixtures.get(s, ())})
        if need and self.fetch is not None:
            missing, why = self.fetch(wt, need)
            if blocked := [s for s in todo if missing & set(out.fixtures.get(s, ()))]:
                shown = ", ".join(sorted(missing)[:3]) + (f" and {len(missing) - 3} more" if len(missing) > 3 else "")
                out.notes.append(f"{', '.join(blocked)} not judged: {len(missing)} of the {len(need)} fixture(s) the suites read "
                                 f"weren't fetched ({why or 'no source held them'}: {shown}); its candidate's check fetches them")
                todo = [s for s in todo if s not in blocked]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(self.jobs, len(todo) or 1))) as pool:
            for s, res in zip(todo, pool.map(lambda s: pytest_failures(self.run, self.python, wt / s, list(self.args)), todo)):
                if isinstance(res, str):
                    out.notes.append(f"collect:{s} not checked: {res}")
                    continue
                out.suites.append(s)
                if res:
                    out.failed.append(failed_row(s, res))
        if self.suites is None:
            out.notes.append(Ran(f"collected {len(todo)} suite(s)" + (f" ({', '.join(todo)})" if len(todo) <= 6 else "")))
        return out


def touched_suites(repo: Path, commit: str, base: str, how: LocalSuites, *, wt: Path | None = None,
                   root: Path | None = None) -> Touched:
    if wt is not None:
        return how(wt, commit, base)
    try:
        with merged_worktree(repo, commit, root) as tree:
            return how(tree, commit, base)
    except QueueError as e:
        why = f"no worktree of {commit[:12]}: {e}"
        return Touched(notes=[why], tail=why)


_IMPORT = re.compile(r"(?:public\s+)?(?:meta\s+)?import\s+(?:all\s+)?(.*)")


def lean_imports(text: str) -> list[str]:
    out, depth = [], 0
    for line in text.splitlines():
        s = line.strip()
        while s:
            if depth:
                i, j = s.find("/-"), s.find("-/")
                if j < 0 and i < 0:
                    s = ""
                elif i >= 0 and (j < 0 or i < j):
                    depth, s = depth + 1, s[i + 2:]
                else:
                    depth, s = depth - 1, s[j + 2:].strip()
            elif s.startswith("/-"):
                depth, s = 1, s[2:]
            else:
                break
        if not s or s.startswith("--") or s in ("module", "prelude"):
            continue
        if not (m := _IMPORT.fullmatch(s)):
            break
        out += m.group(1).split("--", 1)[0].split()
    return out


@dataclass
class LakePackage:
    path: str
    src: str
    roots: tuple[str, ...]
    requires: tuple[str, ...]

    def module(self, f: str) -> str | None:
        rel = posixpath.relpath(f, self.src)
        return None if rel.startswith("../") or not rel.endswith(".lean") else rel.removesuffix(".lean").replace("/", ".")

    def builds(self, module: str) -> bool:
        return any(module == r or module.startswith(r + ".") for r in self.roots)


def lake_packages(wt: Path, files: list[str]) -> dict[str, LakePackage]:
    have = set(files)
    out = {}
    for f in files:
        if not f.endswith("/lean-audit.json") or f.startswith(LEAN_FIXTURES):
            continue
        d = f.rsplit("/", 1)[0]
        src, roots, requires = ".", [], []
        try:
            if f"{d}/lakefile.toml" in have:
                doc = tomllib.loads((wt / d / "lakefile.toml").read_text())
                src = doc.get("srcDir", ".")
                roots = [r for lib in doc.get("lean_lib", []) for r in (lib.get("roots") or [lib.get("name")]) if isinstance(r, str)]
                requires = [r["path"] for r in doc.get("require", []) if isinstance(r.get("path"), str)]
            elif f"{d}/lakefile.lean" in have:
                text = (wt / d / "lakefile.lean").read_text()
                src = (m.group(1) if (m := re.search(r'srcDir\s*:=\s*"([^"]*)"', text)) else ".")
                roots = re.findall(r"lean_lib\s+«?([\w.]+)»?", text)
            else:
                continue
            if f"{d}/lake-manifest.json" in have:
                manifest = json.loads((wt / d / "lake-manifest.json").read_text())
                requires += [p["dir"] for p in manifest.get("packages", []) if p.get("type") == "path" and isinstance(p.get("dir"), str)]
        except (OSError, ValueError, AttributeError, TypeError):
            continue
        out[d] = LakePackage(d, posixpath.normpath(posixpath.join(d, src)), tuple(roots),
                             tuple(sorted({posixpath.normpath(posixpath.join(d, p)) for p in requires} - {d})))
    return out


@dataclass
class LeanGraph:
    module: dict[str, str] = field(default_factory=dict)
    imports: dict[str, list[str]] = field(default_factory=dict)
    targets: dict[str, str] = field(default_factory=dict)

    def upstream(self, f: str) -> list[str]:
        files = {m: g for g, m in self.module.items()}
        seen, todo = set(), list(self.imports.get(self.module.get(f, ""), []))
        while todo:
            if (m := todo.pop()) in seen or m not in files:
                continue
            seen.add(m)
            todo += self.imports.get(m, [])
        return sorted(files[m] for m in seen)


def lean_graph(wt: Path, files: list[str], changed: list[str]) -> LeanGraph:
    pkgs = lake_packages(wt, files)
    owner = lambda f: max((p for p in pkgs if f.startswith(p + "/")), key=len, default=None)    # noqa: E731
    touched = {p for f in changed if f.endswith(".lean") and (p := owner(f)) and not f.endswith("/lakefile.lean")}
    graph = LeanGraph()
    if not touched:
        return graph
    scope = {p for p, pkg in pkgs.items() if p in touched or touched & set(pkg.requires)}
    for f in files:
        if f.endswith(".lean") and not f.endswith("lakefile.lean") and (p := owner(f)) in scope and (m := pkgs[p].module(f)):
            graph.module[f] = m
            try:
                graph.imports[m] = lean_imports((wt / f).read_text(errors="replace"))
            except OSError:
                graph.imports[m] = []
    start = {m for f in changed if (p := owner(f)) in touched and f.endswith(".lean") and (m := pkgs[p].module(f))}
    importers: dict[str, set[str]] = {}
    for m, imps in graph.imports.items():
        for i in imps:
            importers.setdefault(i, set()).add(m)
    reached, todo = set(start), list(start)
    while todo:
        for m in importers.get(todo.pop(), ()):
            if m not in reached:
                reached.add(m)
                todo.append(m)
    graph.targets = {f: m for f, m in graph.module.items() if m in reached and pkgs[owner(f)].builds(m)}
    return graph


@dataclass
class LeanBuilt:
    failed: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    where: str = ""
    passed: list[str] = field(default_factory=list)


class LeanTimedOut(str):
    """The Lean step built nothing within its time: a note under "checks skipped", refusing nothing."""


LEAN_UNBUILT_FIX = "mark again once {on} builds them in its `build` Lean slots (`research status` there says why not)"


def run_group(argv: list[str], cwd: Path, left: float, env: dict[str, str] | None = None) -> tuple[int | None, str]:
    """Runs in a process group of its own, killed whole once ``left`` seconds pass."""
    p = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace",
                         stdin=subprocess.DEVNULL, start_new_session=True, env=env)
    try:
        out, err = p.communicate(timeout=max(left, 0))
        return p.returncode, out + err
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGKILL)
        out, err = p.communicate()
        return None, out + err


RUN_STARTED_RE = re.compile(r"^research: run (\S+) (?:launched|refused) on ", re.M)


@dataclass
class LakeBuild:
    on: tuple[str, ...] = ()
    pool: str | None = PREFLIGHT_LEAN_POOL
    by: str | None = None
    slots: Path = LEAN_SLOTS
    run: Callable[[list[str], Path, float], tuple[int | None, str]] = run_group
    runs: Callable[[], Path] | None = None
    clock: Callable[[], float] = time.monotonic

    def warm(self) -> bool:
        try:
            return any(ln.split("#")[0].split()[:1] == ["build"] for ln in self.slots.read_text().splitlines())
        except OSError:
            return False

    def __call__(self, wt: Path, files: list[str], left: float) -> LeanBuilt | str:
        if left <= 0:
            return LeanTimedOut("the Lean step's budget was spent before it started")
        if not (wt / LEAN_BUILDER).is_file():
            return f"the merged tree has no {LEAN_BUILDER}, which builds them"
        if self.warm():
            rc, text = self.run([sys.executable, LEAN_BUILDER, "--build", *files], wt, left)
            res = self._read(rc, text, "here", left)
            return res if isinstance(res, (LeanBuilt, LeanTimedOut)) else f"{res}: {LEAN_UNBUILT_FIX.format(on='this node')}"
        if not self.on:
            return (f"this machine keeps no warm Lean tree (no `build` pool in {self.slots}; an agent VM builds nothing) and was "
                    f"given no node to build on: `ready --on MACHINE` builds them on {' or '.join(PREFLIGHT_LEAN_ON)}")
        end, tried = self.clock() + left, []
        for i, node in enumerate(self.on):
            if (rest := end - self.clock()) <= 0:
                return LeanTimedOut("; ".join([*tried, f"no time was left to build on {node}"]))
            rc, text, where = self._remote(node, wt, files, rest, wait=i == len(self.on) - 1)
            res = self._read(rc, text, where, rest)
            if isinstance(res, LeanTimedOut):
                return LeanTimedOut("; ".join([*tried, res]))
            if isinstance(res, LeanBuilt):
                res.passed = tried
                return res
            tried.append(res)
        return f"{'; '.join(tried)}: {LEAN_UNBUILT_FIX.format(on=' or '.join(self.on))}"

    @staticmethod
    def _read(rc: int | None, text: str, where: str, left: float) -> LeanBuilt | str:
        if rc is None:
            return LeanTimedOut(f"its build {where} hadn't ended after {left:.0f} s (a wait in its pool's line counts)"
                                f"{': ' + text if text else ''}" if where != "here" else
                                f"its build here was still going after {left:.0f} s{': ' + text if text else ''}")
        doc = next((ln[len(LEAN_BUILD_LINE):] for ln in reversed(text.splitlines()) if ln.startswith(LEAN_BUILD_LINE)), None)
        try:
            doc = json.loads(doc) if doc is not None else None
        except ValueError:
            doc = None
        if not isinstance(doc, dict):
            return f"its build {where} exited {rc} without a result line: {text.strip()[-300:]}"
        built, skipped = doc.get("built") or {}, doc.get("skipped") or {}
        notes = [f"{p} not built {where}: {why}" if where != "here" else f"{p} not built: {why}" for p, why in skipped.items()]
        failed = {f: str(e) for f, e in (doc.get("failed") or {}).items()}
        notes += [f"`lake build` in {p} exited {c} with no Lean error in a file" for p, c in built.items()
                  if c and not any(f.startswith(p + "/") for f in failed)]
        if not built and not failed:
            return "; ".join(notes) or f"its build {where} built nothing (exit {rc})"
        return LeanBuilt(failed, notes, where)

    def _remote(self, node: str, wt: Path, files: list[str], left: float, wait: bool = True) -> tuple[int | None, str, str]:
        from . import runs as R

        secs = f"{max(left - 10, 1):.0f}"
        argv = [sys.executable, "-m", "research", "run", "--on", node, "--stay",
                *(["--pool", self.pool, "--pool-wait", secs] if self.pool else []),
                *(["--as", self.by] if self.by else []), "--project", "verity", "--source", str(wt), "--cwd", "clone",
                "--timeout", f"{secs}s", "--wait", "--wait-every", "5", "--wait-max", secs, "--",
                "python3", LEAN_BUILDER, *([] if wait else ["--no-wait"]), "--build", *files]
        rc, said = self.run(argv, wt, left)
        where = f"on {node}"
        if not (m := RUN_STARTED_RE.search(said)):
            return rc, said.strip()[-300:] if rc is not None else "", where
        rid, where = m.group(1), f"on {node} (run {m.group(1)})"
        self.run([sys.executable, "-m", "research", "fetch", rid, "--on", node, "--all"], wt, 60)
        log = (self.runs or R.runs_root)() / rid / "stdout.log"
        text = log.read_text(errors="replace") if log.is_file() else ""
        rc = None if rc == waitcap.STILL_GOING else rc
        return rc, text or (said.strip()[-300:] if rc is not None else ""), where


LOCK_DROP_FIX = "restore it verbatim from `{onto}` in a lock-only commit (`git show origin/{onto}:PATH`) and mark that head"
LOCK_DROP_SHOWN = 60


def _unowned(rec):
    return {k: v for k, v in rec.items() if k != "owner"} if isinstance(rec, dict) else rec


def _texts(section: str, entries: dict, g: str) -> list[str]:
    if g not in entries:
        return []
    v = entries[g]
    if section == "inline":
        return list(v) if isinstance(v, dict) else []
    return [v if isinstance(v, str) else json.dumps(v, sort_keys=True)]


LOCK_NAME = re.compile(r"(?<![\w.'!?])([^\W\d][\w'!?]*(?:\.[^\W\d][\w'!?]*)*)")


def _held_to_form(lock: dict) -> set[str]:
    claims = lock.get("claims") if isinstance(lock.get("claims"), dict) else {}
    ws, gs, generic = (d if isinstance(d, dict) else {}
                       for d in (claims.get("witnesses"), lock.get("witnesses"), claims.get("generic")))
    names = [n for w in ws.values() if isinstance(w, dict)
             for n in [w.get("fails"), *(w.get("meets") if isinstance(w.get("meets"), list) else [])]]
    for w in gs.values():
        if isinstance(w, dict):
            n = w.get("nontrivial")
            names += [n] if isinstance(n, str) else n if isinstance(n, list) else []
            names += w.get("honest") if isinstance(w.get("honest"), list) else []
    return {n for n in [*names, *generic.values()] if isinstance(n, str)}


def _read_hashes(lock: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for group in (lock.get("reads") or {}).values():
        if isinstance(group, dict) and isinstance(defs := group.get("definitions"), dict):
            out.update(defs)
    return out


def dropped_entries(base: dict, merged: dict) -> list[str]:
    listed, was = merged.get("guarantees") or {}, base.get("guarantees") or {}
    old, new = _read_hashes(base), _read_hashes(merged)
    inert = _held_to_form(merged)

    def restated(g: str) -> bool:
        names = LOCK_NAME.findall(" ".join((listed[g] if isinstance(listed[g], dict) else {}).get("signature") or []))
        return _unowned(listed[g]) != _unowned(was.get(g)) or any(
            old[n] != new[n] for n in names if n in old and n in new)

    out = []
    for section in ("inline", "frozen"):
        b, m = ((d.get("shape") or {}).get(section) or {} for d in (base, merged))
        if not isinstance(b, dict) or not isinstance(m, dict):
            continue
        moved = {t for g in m if g not in b for t in _texts(section, m, g)}
        for g in b:
            if g not in listed or restated(g):
                continue
            had, kept = _texts(section, b, g), _texts(section, m, g)
            gone = [t for t in had if t not in kept and t not in moved]
            if section == "frozen":
                out += [f"shape.frozen[{g}]" for _ in gone if g not in m and g not in inert]
            elif len(gone) > len([t for t in kept if t not in had]):
                out += [f"shape.inline[{g}]: " + (t if len(t) <= LOCK_DROP_SHOWN else t[:LOCK_DROP_SHOWN] + "…") for t in gone]
    return out


def lock_drops(repo: Path, base: str, head: str, merged: str | None = None) -> list[str]:
    r = _git(repo, "diff", "--name-only", "--no-renames", f"{base}...{head}", check=False)
    locks = [f for f in r.stdout.splitlines() if posixpath.basename(f) == LEAN_LOCK and not f.startswith(LEAN_FIXTURES)]
    if r.returncode or not locks:
        return []
    if merged is None:
        argv = ("merge-tree", "--write-tree", "--name-only", "--no-messages", base, head)
        m = _git(repo, f"--attr-source={base}", *argv, check=False)
        if m.returncode == 129 and "attr-source" in m.stderr:
            m = _git(repo, *argv, check=False)
        if m.returncode not in (0, 1) or not m.stdout.split():
            return []
        merged = m.stdout.split()[0]
    fork = _git(repo, "merge-base", base, head, check=False).stdout.strip()
    blobs = _blobs(repo, [f"{c}:{f}" for f in locks for c in (base, merged, fork, head) if c])
    out = []
    for f in locks:
        if f"{base}:{f}" not in blobs or f"{merged}:{f}" not in blobs:
            continue
        try:
            was, now, had, has = (json.loads(blobs[k]) if k in blobs else None
                                  for k in (f"{base}:{f}", f"{merged}:{f}", f"{fork}:{f}", f"{head}:{f}"))
        except ValueError as e:
            out.append(f"{f}: not JSON on `{BRANCH}`, merged, at the merge base or at the head ({e}), so what it drops can't be "
                       "told")
            continue
        if isinstance(was, dict) and isinstance(now, dict):
            own = _own_drops(had, has)
            out += [f"{f}: {d}" for d in dropped_entries(was, now) if d not in own]
    return out


def _own_drops(fork: object, head: object) -> set[str]:
    if not (isinstance(fork, dict) and isinstance(head, dict)):
        return set()
    return {d for d in dropped_entries(fork, head) if d.startswith("shape.inline[")}


def lock_drop_refusal(drops: list[str], onto: str = BRANCH) -> str:
    return (f"its lock merge onto `{onto}` drops {len(drops)} entr{'y' if len(drops) == 1 else 'ies'} `{onto}`'s statements still "
            f"need: {'; '.join(drops)}: {LOCK_DROP_FIX.format(onto=onto)}")


class Preflight:
    def __init__(self, repo: Path, remote: str, store, *, sync: tuple[str, ...] | None = PREFLIGHT_SYNC, python: str | None = None,
                 budget_s: float = PREFLIGHT_CHECKS_S, jobs: int = PREFLIGHT_JOBS, clock: Callable[[], float] = time.monotonic,
                 root: Path | None = None, log: Callable[[str], None] = print,
                 free: Callable[[], float | None] = free_memory,
                 fetch: Callable[[Path, list[str], float], tuple[set[str], str]] | None = None,
                 lean: Callable[[Path, list[str], float], LeanBuilt | str] | None = None,
                 lean_s: float = PREFLIGHT_LEAN_S, lean_modules: int = PREFLIGHT_LEAN_MODULES,
                 collide_s: float = PREFLIGHT_COLLIDE_S, collide_jobs: int | None = None):
        self.repo, self.remote, self.store, self.sync, self.python = Path(repo), remote, store, sync, python
        self.budget_s, self.jobs, self.clock, self.log, self.free = budget_s, jobs, clock, log, free
        self.fetch = fetch or self._fetch_fixtures
        self.lean, self.lean_s, self.lean_modules = lean, lean_s, lean_modules
        self.collide_s, self.collide_jobs = collide_s, max(1, collide_jobs or os.cpu_count() or 1)
        self.root = Path(root or tempfile.gettempdir())

    def __call__(self, pr: PR, prs: list[PR]) -> str | None:
        start = self.clock()
        try:
            base, onto, members = self._fetch(pr)
        except QueueError as e:
            self.log(f"research queue: preflight: #{pr.number} at {pr.head[:12]}: checks skipped: all ({e}); its candidate's "
                     "check still tests the head")
            return None
        if why := self._old_next(pr, prs):
            return why
        notes: list[str] = []
        found = self._contained(pr, prs) + self._gates(pr, notes)
        tree, files, r = None, [], None
        if not pr.move:
            tree, files, r = merged_tree(self.repo, base, pr.head)
            if tree is not None:
                self._collisions(pr, prs, base, onto)
        if pr.move:
            notes.append(f"the merged tree's checks: a move ({pr.move}), which its candidate's check checks")
        elif tree is None and files and (moves := self._only_moved(base, files)):
            notes.append(f"the merged tree's checks: it conflicts with {onto} in {', '.join(files)}, files only the move "
                         f"{', '.join(moves)} changed: restack it across the move (`tools/move/restack.py`) and mark the new "
                         "head")
        elif tree is None and files:
            by = touched_by(self.repo, members, files)
            found.insert(0, f"it conflicts with {onto} in {', '.join(files)}" + (f" (touched by {', '.join(by)})" if by else "")
                         + ": merge `main` once those land, or settle it with their owners; never merge `next` into the branch "
                           "(it carries unlanded PRs)")
            notes.append("the merged tree's checks: it doesn't merge")
        elif tree is None:
            notes.append(f"the merged tree's checks: `git merge-tree` exited {r.returncode} ({(r.stderr or r.stdout).strip()[-200:]})")
        else:
            if files:
                pins, gone, tip_gone = taken(self.repo, tree, pr.head, files)
                if pins:
                    notes.append(Ran(f"conflicts only in pins: {', '.join(pins)}, rebuilt from the merged tree"))
                if gone:
                    notes.append(Ran(f"deletes {', '.join(gone)}, which {onto} changed only by porting it to the module "
                                     "system: deleted in the merge"))
                if tip_gone:
                    notes.append(Ran(f"{onto} deletes {', '.join(tip_gone)}, which the head changed only by porting it to the "
                                     "module system: deleted in the merge"))
            tree, files, pins, why = ported(self.repo, tree, base=base)
            if tree is None:
                found.append(why)
                notes.append("the merged tree's checks: its Lean doesn't take the module system")
            else:
                if files:
                    notes.append(Ran(f"Lean made before the module system: {', '.join(files)} ported by `{MODULES}`"
                                     + (f", and the pins {', '.join(pins)} rebuilt" if pins else "")))
                if drops := lock_drops(self.repo, base, pr.head, tree):
                    found.append(lock_drop_refusal(drops, onto.split("`")[1]))
                try:
                    found += self._merged(pr, base, tree, onto, notes, members)
                except QueueError as e:
                    notes.append(f"the merged tree's checks ({e})")
        skipped = [n for n in notes if not isinstance(n, Ran)]
        took = "; ".join([f"{self.clock() - start:.0f} s", *(n for n in notes if isinstance(n, Ran))])
        if found:
            return (f"the preflight of its head merged onto {onto} refused it ({took}"
                    + (f"; checks skipped: {'; '.join(skipped)}" if skipped else "") + "):\n  - " + "\n  - ".join(found)
                    + "\n  (`--no-preflight` skips the preflight: the lander's)")
        said = f"checks skipped: {'; '.join(skipped)}; the checks that ran passed" if skipped else "nothing refused"
        self.log(f"research queue: preflight: #{pr.number} at {pr.head[:12]} merged onto {onto}: {said} ({took})")
        return None

    def _old_next(self, pr: PR, prs: list[PR]) -> str | None:
        if pr.base != "main":
            return None
        into = into_at(self.repo, "refs/queue/preflight/main")
        if _git(self.repo, "rev-parse", "-q", "--verify", f"refs/queue/preflight/{into}^{{commit}}", check=False).returncode:
            into = "main"
        try:
            old = old_next(self.repo, [pr.head], f"refs/queue/preflight/{into}", {p.number for p in prs})
        except QueueError as e:
            self.log(f"research queue: preflight: whether #{pr.number}'s head carries an old `{BRANCH}` isn't known ({e}); the "
                     "sync judges it")
            return None
        return old_next_refusal(list(old.values()), into) if old else None

    def _only_moved(self, base: str, files: list[str]) -> list[str]:
        moves = queue_log(self.repo, base)[1]
        moved = set().union(*(changed(self.repo, f"{mv.commit}^1", mv.commit) for mv in moves))
        return [f"{mv.commit[:12]} ({mv.map})" for mv in moves] if moves and set(files) <= moved else []

    def _fetch(self, pr: PR) -> tuple[str, str, list[Member]]:
        _git(self.repo, *STALL, "fetch", "-q", self.remote, "+refs/heads/main:refs/queue/preflight/main",
             f"+refs/pull/{pr.number}/head:refs/queue/preflight/pr")
        missing = lambda: _git(self.repo, "cat-file", "-e", f"{pr.head}^{{commit}}", check=False).returncode   # noqa: E731
        if missing() and pr.branch:
            _git(self.repo, *STALL, "fetch", "-q", self.remote, f"+refs/heads/{pr.branch}:refs/queue/preflight/branch", check=False)
        if missing():
            raise QueueError(f"{pr.head[:12]} isn't on {self.remote} (pushed again since?)")
        main, into = _out(self.repo, "rev-parse", "refs/queue/preflight/main"), "main"
        if (written := land_into_key(self.repo, "refs/queue/preflight/main")) == "main":
            return main, f"`main` at {main[:12]}", []
        if (land := written or "main") != "main" and not _git(
                self.repo, *STALL, "fetch", "-q", self.remote, f"+refs/heads/{land}:refs/queue/preflight/{land}", check=False).returncode:
            main, into = _out(self.repo, "rev-parse", f"refs/queue/preflight/{land}"), land
        r = _git(self.repo, *STALL, "ls-remote", "--exit-code", self.remote, f"refs/heads/{BRANCH}", check=False)
        if r.returncode == 2:
            _git(self.repo, "update-ref", "-d", f"refs/queue/preflight/{BRANCH}", check=False)
            return main, f"`{into}` at {main[:12]} (no `{BRANCH}`)", []
        if r.returncode:
            raise QueueError(f"git ls-remote {self.remote}: {(r.stderr or r.stdout).strip()[-200:]}")
        _git(self.repo, *STALL, "fetch", "-q", self.remote, f"+refs/heads/{BRANCH}:refs/queue/preflight/{BRANCH}")
        tip = _out(self.repo, "rev-parse", f"refs/queue/preflight/{BRANCH}")
        if into == "main" and _git(self.repo, "merge-base", "--is-ancestor", main, tip, check=False).returncode:
            return main, f"`{into}` at {main[:12]} (`{BRANCH}` at {tip[:12]} doesn't hold it)", []
        members = queue_merges(self.repo, main, tip)
        if own := next((m for m in reversed(members) if m.pr == pr.number), None):
            below = _out(self.repo, "rev-parse", f"{own.commit}^1")
            return below, f"`{BRANCH}` below its merge of #{pr.number} at {own.head[:8]} ({below[:12]})", members[:members.index(own)]
        return tip, f"`{BRANCH}` at {tip[:12]}", members

    def _contained(self, pr: PR, prs: list[PR]) -> list[str]:
        main = "refs/queue/preflight/main"
        heads = {p.head: p.number for p in prs}
        land = f"refs/queue/preflight/{into_at(self.repo, main)}"
        landed = land if not _git(self.repo, "rev-parse", "-q", "--verify", f"{land}^{{commit}}", check=False).returncode else main
        carried = [c for c in carried_heads(self.repo, landed, pr.head) if c[0] != pr.number]
        inside = set(contained(self.repo, main, pr.head, heads)) - {pr.number}
        if not inside and not carried:
            return []
        rules = rules_at(self.repo, main) or Rules()
        labels = StoreLabels(self.store, log=lambda s: print(s, file=sys.stderr), since=rules.marked_after)
        q = Queue(self.repo, remote=self.remote, github=None, ready=lambda n, h: (n, h) == (pr.number, pr.head) or labels.ready(n, h),
                  store=self.store, rules=rules, log=self.log, before_cutoff=labels.before_cutoff, marked_by=labels.marked_by,
                  marked_at=labels.marked_at, prefetch=labels.prefetch)
        labels.change = functools.partial(q.own_change, landing=landed)
        judged = inside | {n for n, _, _ in carried}
        try:
            entries = q.entries([pr, *(p for p in prs if p.number in judged)], {"ejected": {}}, base=main, landed=landed,
                                queued=f"refs/queue/preflight/{BRANCH}")
        except QueueError as e:
            self.log(f"research queue: preflight: can't tell whether the open PRs this head contains are admitted ({e})")
            return []

        def why(n: int) -> str:
            e, lab = entries[n], labels._newest.get((n, entries[n].pr.head))
            if lab is not None and str(lab.value).lower() == "false" and lab.ref:
                return f"ejected at {e.pr.head[:8]} by {lab.by}: {lab.ref}"
            return f"not marked ready at {e.pr.head[:8]}" if e.why.startswith("not marked ready") else e.why

        held = [n for n in entries[pr.number].after if entries[n].state != "ready"]
        stale = stale_carried(self.repo, (c for c in carried if c[0] not in held), {n: e.pr.head for n, e in entries.items()},
                              lambda n: None if entries[n].state == "ready" else why(n), landed.rsplit("/", 1)[-1])
        return [f"it contains {entries[pr.number].holds(n)}, and #{n} isn't admitted ({why(n)}), so the verified queue holds this PR "
                f"behind it: mark #{n} ready first, so it lands first, or drop #{n}'s commits from this branch (if this PR carries "
                f"#{n}'s change, #{n}'s owner withdraws its mark: `research data label pr:{n}@{entries[n].pr.head[:12]} ready false "
                f"--by OWNER`)" for n in held] + list(stale.values())

    def _gates(self, pr: PR, notes: list[str]) -> list[str]:
        main = _out(self.repo, "rev-parse", "refs/queue/preflight/main")
        try:
            gates, warns = gates_at(self.repo, main), gates_at(self.repo, main, "warn_gates")
        except QueueError as e:
            notes.append(f"no landing gate run ({e})")
            return []
        refused, unjudged = gate_refusal(self.repo, main, pr.head, gates)
        notes += unjudged
        for why in gate_refusal(self.repo, main, pr.head, warns, kind="warn gate")[0]:
            self.log(f"research queue: preflight: warning: #{pr.number} at {pr.head[:12]}: {why}. {WARN_GATE_SAYS}")
        return refused

    def _collisions(self, pr: PR, prs: list[PR], base: str, onto: str) -> None:
        """A warning, never a refusal."""
        say = lambda what: self.log(f"research queue: preflight: warning: #{pr.number} at {pr.head[:12]}: {what}")  # noqa: E731
        some = lambda xs: ", ".join(xs[:PREFLIGHT_COLLIDE_NAMED]) + (                                              # noqa: E731
            f" and {len(xs) - PREFLIGHT_COLLIDE_NAMED} more" if len(xs) > PREFLIGHT_COLLIDE_NAMED else "")
        try:
            found, unread = self._collide(pr, prs, base)
        except Exception as e:  # noqa: BLE001  (a read that only warns never costs the mark)
            say(f"the marked PRs it doesn't merge with weren't read ({type(e).__name__}: {e}); the queue's merge judges each")
            return
        if found:
            say(f"it doesn't merge with {len(found)} marked PR(s), each merged onto {onto} and this head onto that, as a "
                "candidate holding both merges them:\n    "
                + "\n    ".join(f"#{p.number} ({p.owner or 'no lane named'}) at {p.head[:12]}, in {some(files)}" for p, files in found)
                + f"\n  {COLLIDE_SAYS}")
        if unread:
            say("whether it merges with these PRs wasn't read: "
                + "; ".join(f"{', '.join(f'#{p.number}' for p in ps)} ({why})" for why, ps in unread.items())
                + f". The read has {self.collide_s:g} s of its own and refuses nothing; the queue's merge judges each")

    def _collide(self, pr: PR, prs: list[PR], base: str) -> tuple[list[tuple[PR, list[str]]], dict[str, list[PR]]]:
        deadline = self.clock() + self.collide_s
        spent = lambda: self.clock() >= deadline                                                                  # noqa: E731
        absent = lambda p: _git(self.repo, "cat-file", "-e", f"{p.head}^{{commit}}", check=False).returncode      # noqa: E731
        rules = rules_at(self.repo, "refs/queue/preflight/main") or Rules()
        labels = StoreLabels(self.store, log=lambda s: print(s, file=sys.stderr), since=rules.marked_after)
        others = [p for p in prs if p.number != pr.number and p.base == "main" and not p.move]
        labels.prefetch([(p.number, p.head) for p in others])
        marked: list[PR] = []
        unread: dict[str, list[PR]] = {}
        for p in others:
            if spent():
                unread.setdefault("their marks not read in time", []).append(p)
            elif labels.ready(p.number, p.head):
                marked.append(p)
        if marked and spent():
            return [], {**unread, "not merged in time": marked}
        if fetch := [p for p in marked if absent(p)]:
            _git(self.repo, *STALL, "fetch", "-q", self.remote,
                 *(f"+refs/pull/{p.number}/head:refs/queue/preflight/marked/{p.number}" for p in fetch), check=False)
            if gone := [p for p in fetch if absent(p)]:
                unread["their heads not fetched"] = gone
                marked = [p for p in marked if p not in gone]

        def paths(head: str) -> set[str]:
            mb = _git(self.repo, "merge-base", base, head, check=False).stdout.strip()
            return changed(self.repo, mb, head) if mb else set()

        def files(o: PR) -> list[str]:
            if any(not _git(self.repo, "merge-base", "--is-ancestor", a, b, check=False).returncode
                   for a, b in ((o.head, pr.head), (pr.head, o.head))):
                return []
            if (tree := merged_tree(self.repo, base, o.head)[0]) is None:
                return []
            below = _out(self.repo, "commit-tree", tree, "-p", base, "-p", o.head, "-m", f"preflight: #{o.number} onto {base[:12]}")
            return conflicting_files(self.repo, below, pr.head)

        found: list[tuple[PR, list[str]]] = []
        with concurrent.futures.ThreadPoolExecutor(self.collide_jobs) as pool:
            mine = paths(pr.head)
            todo = [o for o, theirs in zip(marked, pool.map(lambda o: paths(o.head), marked)) if mine & theirs]
            for i in range(0, len(todo), self.collide_jobs):
                if spent():
                    unread["not merged in time"] = todo[i:]
                    break
                group = todo[i:i + self.collide_jobs]
                found += [(o, fs) for o, fs in zip(group, pool.map(files, group)) if fs]
        return found, unread

    def _merged(self, pr: PR, base: str, tree: str, onto: str, notes: list[str], members: list[Member] | None = None) -> list[str]:
        commit = _out(self.repo, "commit-tree", tree, "-p", base, "-p", pr.head, "-m", f"preflight: #{pr.number} onto {base[:12]}")
        fixtures: dict[str, list[str]] = {}
        out: list[str] = []
        with merged_worktree(self.repo, commit, self.root) as wt:
            graph, lean_failed, lean_deadline = self._lean(wt, base, tree, notes, out)
            deadline = self.clock() + self.budget_s
            failed = self._checks(wt, base, tree, None, notes, deadline, fixtures)
            again: dict[str, dict[str, str]] = {}
            lean_again: dict[str, str] | str = {}
            if any(failed.values()) or lean_failed:
                _git(wt, "checkout", "-q", "--detach", "--force", base)
                _git(wt, "clean", "-qffdx", "-e", "/.venv")
                again = (self._checks(wt, base, tree, {k for k, v in failed.items() if v}, [], deadline, fixtures)
                         if any(failed.values()) else {})
                lean_again = self._lean_at_base(wt, lean_failed, lean_deadline) if lean_failed else {}
            if ratchet := self._ratchet(wt, base, commit, tree, notes, deadline):
                self.log(f"research queue: preflight: warning: #{pr.number} at {pr.head[:12]}: the ratchet gate "
                         f"({PREFLIGHT_RATCHET[0]} as {onto} has it) fails the merge: " + "\n    ".join(ratchet)
                         + f"\n  {RATCHET_SAYS}")
        if not any(failed.values()) and not lean_failed:
            return out
        for name, fails in failed.items():
            if fails and name not in again:
                notes.append(f"{name} failed on the merged tree and couldn't run on {onto} to compare: {'; '.join(fails.values())}")
            elif new := [line for k, line in sorted(fails.items()) if k not in again[name]]:
                out.append(self._finding(name, new))
        if isinstance(lean_again, str):
            notes.append(f"Lean: {', '.join(sorted(lean_failed))} failed to build on the merged tree and couldn't be built on {onto} "
                         f"to compare ({lean_again}): {'; '.join(lean_failed.values())}")
        else:
            changed = set(_out(self.repo, "diff", "--name-only", "--no-renames", base, tree).split())
            cache: dict[str, set[str]] = {}
            out += [self._lean_finding(f, line, graph, members or [], f in changed, cache)
                    for f, line in sorted(lean_failed.items()) if f not in lean_again]
        return out

    def _lean(self, wt: Path, base: str, tree: str, notes: list[str], found: list[str]) -> tuple[LeanGraph, dict[str, str], float]:
        deadline = self.clock() + self.lean_s
        if self.lean is None:
            return LeanGraph(), {}, deadline
        changed = _out(self.repo, "diff", "--name-only", "--no-renames", base, tree).split()
        graph = lean_graph(wt, _out(wt, "ls-files").splitlines(), changed)
        if not graph.targets:
            return graph, {}, deadline
        if len(graph.targets) > self.lean_modules:
            notes.append(f"Lean: not built: {len(graph.targets)} modules are or import what it changes, over the "
                         f"{self.lean_modules} the preflight builds; the lock run after its landing builds them")
            return graph, {}, deadline
        res = self.lean(wt, sorted(graph.targets), deadline - self.clock())
        if isinstance(res, LeanTimedOut):
            notes.append(f"lean build timed out ({len(graph.targets)} module(s) not built: {res})")
            return graph, {}, deadline
        if isinstance(res, str):
            found.append(f"Lean: {len(graph.targets)} module(s) that are or import what it changes couldn't be built: {res}")
            return graph, {}, deadline
        notes += [f"Lean: {n}" for n in res.notes]
        notes.append(Ran(f"Lean: built {len(graph.targets)} module(s) {res.where or 'here'}, {len(res.failed)} failed"
                         + (f" (not built before that: {'; '.join(res.passed)})" if res.passed else "")))
        return graph, res.failed, deadline

    def _lean_at_base(self, wt: Path, failed: dict[str, str], deadline: float) -> dict[str, str] | str:
        there = sorted(f for f in failed if (wt / f).is_file())
        if not there:
            return {}
        res = self.lean(wt, there, deadline - self.clock())
        return res if isinstance(res, str) else res.failed

    def _lean_finding(self, f: str, line: str, graph: LeanGraph, members: list[Member], own: bool,
                      cache: dict[str, set[str]]) -> str:
        mod, stack = graph.module.get(f, f), "merge or stack on that PR, never merge `next` (it carries unlanded PRs)"
        if owners := [] if own else touched_by(self.repo, members, [f], cache):
            return f"Lean: `{mod}` ({f}), of {', '.join(owners)} on `{BRANCH}`, fails to build with this head merged: {line}: {stack}"
        whose = "this head's own" if own else "on `main`"
        if ups := touched_by(self.repo, members, graph.upstream(f), cache):
            return (f"Lean: `{mod}` ({f}), {whose}, fails to build with this head merged: {line}: it imports what "
                    f"{', '.join(ups)} on `{BRANCH}` changed: {stack}")
        return f"Lean: `{mod}` ({f}), {whose}, fails to build with this head merged and not on `{BRANCH}`: {line}: fix the branch"

    @staticmethod
    def _finding(name: str, lines: list[str]) -> str:
        shown = "; ".join(lines[:5]) + (f"; and {len(lines) - 5} more" if len(lines) > 5 else "")
        kind, _, where = name.partition(":")
        if kind == "sync":
            return (f"`uv sync --locked` on the merged tree: {shown}: the merge leaves `uv.lock` stale; run `uv lock` on the branch "
                    f"once `main` has the other side, and push it")
        if kind == "pins":
            return (f"`pins.py --stale` on the merged tree: {shown}: regenerate each with its command, never by hand "
                    f"(`python3 tools/check/pins.py --regen`); when the head alone is fresh, the other side is a PR on "
                    f"`next`: regenerate once it lands")
        if kind == "tests":
            return (f"the invariant tests in {where} fail on the merged tree: {shown}: reproduce on `git merge origin/next` in a "
                    f"scratch worktree, fix the branch, push and mark the new head")
        return (f"`pytest --collect-only` in {where} fails on the merged tree: {shown}: a test reads what another queued PR "
                f"changes or what only an unlanded PR adds; make it read what `main` and `next` hold, or wait for that PR")

    def _run(self, argv: list[str], cwd: Path, deadline: float,
             extra: dict[str, str] | None = None) -> tuple[subprocess.CompletedProcess | None, str]:
        left = deadline - self.clock()
        if left <= 0:
            return None, "the budget was spent before it started"
        env = {k: v for k, v in os.environ.items() if k not in quick_tier.CALLER_ENV and not k.startswith("PYTEST_")}
        env["COLUMNS"] = "300"
        env.update(extra or {})
        try:
            return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, errors="replace",
                                  timeout=left if math.isfinite(left) else None), ""
        except subprocess.TimeoutExpired:
            return None, f"still running when the {self.budget_s:g} s budget ran out"
        except OSError as e:
            return None, f"{argv[0]} didn't start ({e})"

    def _fetch_fixtures(self, wt: Path, paths: list[str], deadline: float) -> tuple[set[str], str]:
        left = min(deadline - self.clock(), PREFLIGHT_FETCH_S)
        if left <= 0:
            return set(paths), "the budget was spent before the fetch started"
        root = getattr(self.store, "root", None)
        argv = [sys.executable, "-P", "-m", "research", "data", "fetch-fixtures", *(["--store", str(root)] if root else []),
                "--repo", str(wt), "--json", *(str(wt / p) for p in paths)]
        try:
            r = subprocess.run(argv, capture_output=True, text=True, errors="replace", stdin=subprocess.DEVNULL, timeout=left,
                               check=False)
        except subprocess.TimeoutExpired:
            return set(paths), f"`research data fetch-fixtures` still running after {left:.0f} s"
        except OSError as e:
            return set(paths), f"`research data fetch-fixtures` didn't start ({e})"
        try:
            rep = json.loads(r.stdout)
            missing = {e["path"] for e in rep["entries"] if e["status"] in ("missing", "stale")}
        except (ValueError, KeyError, TypeError):
            return set(paths), f"`research data fetch-fixtures` exited {r.returncode}: {(r.stderr or r.stdout).strip()[-200:]}"
        if not missing:
            return missing, ""
        return missing, ("no source here: no reachable private remote, no public URL" if not rep.get("sources")
                         else f"not on {', '.join(rep['sources'])}")

    def _checks(self, wt: Path, base: str, tree: str, only: set[str] | None, notes: list[str],
                deadline: float, fixtures: dict[str, list[str]]) -> dict[str, dict[str, str]]:
        want = (lambda name: True) if only is None else (lambda name: name in only)
        out: dict[str, dict[str, str]] = {}

        def got(name: str, res: dict[str, str] | str) -> None:
            if isinstance(res, str):
                notes.append(f"{name} not checked: {res}")
            else:
                out[name] = res

        if want("pins"):
            if (wt / "tools/check/pins.py").is_file():
                r, why = self._run([sys.executable, "tools/check/pins.py", "--stale"], wt, deadline)
                got("pins", why if r is None else {} if r.returncode == 0 else
                    {ln.split(" is stale", 1)[0]: ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip()}
                    or {f"exit {r.returncode}": f"pins.py --stale exited {r.returncode}"})
        if self.sync is not None:
            r, why = self._run(list(self.sync), wt, deadline)
            text = "" if r is not None and r.returncode == 0 else why or (r.stderr or r.stdout).strip()[-300:]
            if r is not None and (not text or _LOCK_STALE in text) and want("sync"):
                out["sync"] = {"uv.lock": text.splitlines()[-1]} if text else {}
            if text:
                if r is None or _LOCK_STALE not in text:
                    notes.append(f"the merged tree's tests not run: `{' '.join(self.sync)}` failed ({text})")
                return out
        py = self.python or str(wt / ".venv" / "bin" / "python")
        run = lambda argv, cwd: self._run(argv, cwd, deadline)                         # noqa: E731
        for d, files in PREFLIGHT_TESTS:
            if want(f"tests:{d}") and (present := [f for f in files if (wt / d / f).is_file()]):
                got(f"tests:{d}", pytest_failures(run, py, wt / d, present))
        lints = [(path, reach) for path, reach in PREFLIGHT_LINTS if want(f"tests:{path}") and (wt / path).exists()]
        changed = _out(self.repo, "diff", "--name-only", base, tree).split() if lints else []
        for path, reach in lints:
            if reaches(changed, reach):
                d, _, rest = path.rpartition("tests/")
                got(f"tests:{path}", pytest_failures(run, py, wt / (d.rstrip("/") or "."), ["tests/" + rest]))
        names = None if only is None else [n.partition(":")[2] for n in only if n.startswith("collect:")]
        fetch = lambda wt_, paths: self.fetch(wt_, paths, deadline)                    # noqa: E731
        res = touched_suites(self.repo, tree, base, LocalSuites(py, run, self.jobs, self.free, names, fetch=fetch,
                                                                fixtures=fixtures), wt=wt)
        fixtures.update(res.fixtures)
        notes += res.notes
        failed = {row["suite"]: dict(zip(row["failed_tests"], row["summary"])) for row in res.failed}
        for s in res.suites:
            out[f"collect:{s}"] = failed.get(s, {})
        return out

    def _ratchet(self, wt: Path, base: str, commit: str, tree: str, notes: list[str], deadline: float) -> list[str]:
        path, reach = PREFLIGHT_RATCHET
        if (_git(self.repo, "cat-file", "-e", f"{base}:{path}", check=False).returncode
                or not reaches(_out(self.repo, "diff", "--name-only", base, tree).split(), reach)):
            return []
        script = wt.parent / Path(path).name
        script.write_text(_git(self.repo, "show", f"{base}:{path}").stdout, errors="surrogateescape")
        r, why = self._run([self.python or str(wt / ".venv" / "bin" / "python"), str(script), base, commit, "--json"], wt,
                           deadline, {"RATCHET_GATE_CACHE": "", "TMPDIR": str(wt.parent)})
        if r is not None:
            with contextlib.suppress(ValueError, TypeError, KeyError, AttributeError):
                found = [str(x) for f in json.loads(r.stdout)["failures"] for x in (f["why"], *(f.get("fix") or []))]
                if r.returncode in (0, 1) and bool(found) == (r.returncode == 1):
                    return list(dict.fromkeys(found))
            why = f"exited {r.returncode}: {(r.stderr or r.stdout).strip()[-300:]}"
        notes.append(f"ratchet-gate not checked: {why}")
        return []


class MarkOnly:
    def __call__(self, pr: PR) -> None:
        return None


FAST_KIND = "lean-fast/v1"
CHECK_KINDS = ("check/v1", "check-shard/v1", "check-deferred/v1")
LEAN_AUDIT_STEP = "lean-audit"
KEY_COMMAND = "tools/verity/lean/lean_changed.py"


def ejection(store, pr: PR) -> str | None:
    try:
        labs = StoreLabels(store, log=lambda s: print(s, file=sys.stderr))._labels(pr.number, pr.head, vocab.READY)
    except QueueError as e:
        print(f"research queue ready: can't tell whether #{pr.number}'s head was ejected ({e}); its candidate's check still checks it",
              file=sys.stderr)
        return None
    last = labs[-1] if labs else None
    if last is None or str(last.value).lower() != "false" or not last.ref:
        return None
    return f"{last.ref} (ejected by {last.by} at {last.ts})"


SHA_READS, SHA_WAIT_S = 6, 5.0


def off_main_note(pr: PR, prs: list[PR]) -> str | None:
    if pr.base == "main":
        return None
    parent = next((p for p in prs if p.branch == pr.base and p.number != pr.number), None)
    until = f"until #{parent.number} lands, one stack level per landing" if parent else "until it is"
    return (f"#{pr.number} is based on {based_on(pr, prs)}, not main: the verified queue won't see it {until}. Retarget it to main (the queue orders "
            f"stacked PRs itself: a PR containing another waits for it); retargeting keeps the head, so a ready mark at "
            f"{pr.head[:12]} counts from then on, and a new push needs a new mark unless its own change is the same")


def ready(github, store, number: int | None, by: str, tier: Callable[[PR], str | None], again: bool = False,
          preflight: Callable[[PR, list[PR]], str | None] | None = None, sha: str | None = None,
          sleep: Callable[[float], None] = time.sleep, as_json: bool = False,
          goals: Callable[[PR], str | None] | None = None) -> int:
    with contextlib.redirect_stdout(sys.stderr) if as_json else contextlib.nullcontext():
        rc, pr, prs = _ready(github, store, number, by, tier, again, preflight, sha, sleep, goals)
        note = off_main_note(pr, prs) if pr is not None else None
        if note:
            print(f"research queue ready: {note}", file=sys.stderr)
    if as_json:
        print(json.dumps({"pr": number, "head": pr and pr.head, "base": pr and pr.base, "exit": rc, "warning": note}))
    return rc


def _ready(github, store, number: int | None, by: str, tier: Callable[[PR], str | None], again: bool,
           preflight: Callable[[PR, list[PR]], str | None] | None, sha: str | None,
           sleep: Callable[[float], None], goals: Callable[[PR], str | None] | None = None) -> tuple[int, PR | None, list[PR]]:
    if not sha and number is not None and (p := next((x for x in github.open_prs() if x.number == number), None)) and p.branch \
            and (tip := getattr(github, "branch_tip", lambda _b: None)(p.branch)):
        sha = tip
    for i in range(SHA_READS if sha else 1):
        prs = github.open_prs() if number is not None else []
        pr = next((p for p in prs if p.number == number), None)
        if pr is None or not sha or pr.head.startswith(sha):
            break
        if i + 1 < SHA_READS:
            sleep(SHA_WAIT_S)
    if pr is None:
        print(f"research queue ready: {'#' + str(number) if number is not None else 'a PR number'} is not an open PR", file=sys.stderr)
        return 2, pr, prs
    if why := pr.outside(github.repo):
        print(f"research queue ready: #{pr.number} is not marked ready: {why}", file=sys.stderr)
        return 1, pr, prs
    if sha and not pr.head.startswith(sha):
        print(f"research queue ready: #{pr.number} is not marked: GitHub has its head at {pr.head[:12]}, not {sha[:12]}, after "
              f"{SHA_READS} reads {SHA_WAIT_S:g} s apart. Push {sha[:12]} to the PR's branch (or pass the head GitHub shows), then "
              "run this again", file=sys.stderr)
        return 1, pr, prs
    if goals is not None and (why := goals(pr)):
        print(f"research queue ready: warning: #{pr.number} at {pr.head[:12]}: {why}", file=sys.stderr)
    if not again and (why := ejection(store, pr)):
        print(f"research queue ready: #{pr.number} is not marked ready at {pr.head[:12]}: that head was ejected: {why}. Push the fix "
              f"and mark the new head; if the ejection was wrong (a machine fault), say so in the merge thread and pass --again",
              file=sys.stderr)
        return 1, pr, prs
    if preflight is not None and (why := preflight(pr, prs)):
        print(f"research queue ready: #{pr.number} is not marked ready at {pr.head[:12]}: {why}", file=sys.stderr)
        return 1, pr, prs
    if (repo := getattr(tier, "repo", None)) is not None:
        try:
            old = same_tree_mark(store, repo, tier.remote, pr)
        except QueueError as e:
            print(f"research queue: {e}; running the quick tier", file=sys.stderr)
            old = None
        if old:
            print(f"research queue: #{pr.number}'s head {pr.head[:12]} has the tree of {old[:12]}, marked ready after its quick tier "
                  f"passed: the same files, so the tier isn't run again", flush=True)
            return _mark(store, pr, by, None, "ready"), pr, prs
    if isinstance(tier, MarkOnly):
        print(f"research queue: #{pr.number} at {pr.head[:12]}: marked on {by}'s word, with no quick tier", flush=True)
        return _mark(store, pr, by, None, "ready"), pr, prs
    print(f"research queue: the quick tier of #{pr.number}'s head {pr.head[:12]} on its merge onto `main`'s tip: `suites.py --quick` "
          "on the suites it changes", flush=True)
    return _mark(store, pr, by, tier(pr), "ready"), pr, prs


def judge(github, store, ref: str, by: str, on: str | None = None, fetch: Callable[[Path], dict] | None = None,
          runs_root: Path | None = None, repo: Path | None = None, url: str | None = None,
          goals: Callable[[PR], str | None] | None = None) -> int:
    from . import cli, remote
    from . import runs as R

    def refuse(why: str) -> int:
        print(f"research queue judge: {why}", file=sys.stderr)
        return 1

    try:
        d = cli._run_dir(R.runs_root(runs_root), ref, on)
        rec = (fetch or remote.fetch)(d)
    except (FileNotFoundError, LookupError, remote.Refused) as e:
        return refuse(str(e))
    rid, machine, status = d.name, rec.get("machine") or "its machine", _json_file(d / "status.json")
    if status.get("state") not in R.TERMINAL:
        err = (rec.get("last_fetch_error") or {}).get("stderr")
        return refuse(f"run {rid} hasn't ended (last seen {status.get('state') or 'without a status'}"
                      f"{'; its machine is unreachable: ' + err.strip()[-200:] if err else ''}): `research inspect {rid}`")
    res, why = tier_report(d, rid, machine, status)
    if why:
        return refuse(why)
    head = tier.get("head") if isinstance(tier := res.doc.get("tier"), dict) else None
    of = (tier.get("merged") if isinstance(tier, dict) else None) or head
    open_prs = github.open_prs()
    prs = [p for p in open_prs if p.head == of]
    if not prs and repo is not None and isinstance(of, str):
        if len(prs := same_tree_prs(repo, url, open_prs, of)) == 1:
            print(f"research queue: #{prs[0].number}'s head {prs[0].head[:12]} has the tree of {of[:12]}, where run {rid} ran the "
                  f"quick tier: the same files", flush=True)
    if len(prs) != 1:
        return refuse(f"run {rid} ran the quick tier at {str(of)[:12]}, " + (
            f"the head of each of {', '.join('#' + str(p.number) for p in prs)}: mark one with `research queue ready`" if prs else
            "which is no open PR's head now, nor its tree (pushed again since, or closed?)"))
    if why := prs[0].outside(github.repo):
        return refuse(f"#{prs[0].number} is not marked ready: {why}")
    if goals is not None and (why := goals(prs[0])):
        print(f"research queue judge: warning: #{prs[0].number} at {prs[0].head[:12]}: {why}", file=sys.stderr)
    covers = allowed(repo or Path.cwd(), tier.get("base") if isinstance(tier, dict) else None)
    return _mark(store, prs[0], by, judge_report(res, rid, machine, head, status, covers), "judge")


def _mark(store, pr: PR, by: str, failed: str | None, command: str) -> int:
    from .labelauth import AuthError
    from .store.cli import _label_write_through

    if isinstance(failed, StillGoing):
        print(f"research queue {command}: #{pr.number} is not marked ready at {pr.head[:12]} yet. {failed}", file=sys.stderr)
        return waitcap.STILL_GOING
    if isinstance(failed, NotStarted):
        print(f"research queue {command}: #{pr.number} is not marked ready at {pr.head[:12]}: the quick tier never started: {failed}. "
              f"Nothing failed: run it again (`--on auto` takes the machine with a free check slot). A PR lands once its candidate's full "
              f"check passes, so the quick tier isn't required: `ready N --mark --by LANE` marks it; to run its tests anyway: {LANE_RUN}", file=sys.stderr)
        return READY_NOT_STARTED
    if failed is not None:
        print(f"research queue {command}: #{pr.number} is not marked ready at {pr.head[:12]}: the quick tier failed: {failed}. Push a "
              f"fix, or run it again: passes are cached by input, so a re-run on the same cache runs only what failed", file=sys.stderr)
        return 1
    target = label_target(pr.number, pr.head)
    try:
        store.add_label(target, vocab.READY, True, by)
    except AuthError as e:
        print(f"research queue {command}: #{pr.number} is not marked ready at {pr.head[:12]}: {e}", file=sys.stderr)
        return 1
    said, lost = _label_write_through(store, target)
    print(f"research queue: #{pr.number} is ready at {pr.head[:12]} ({target}); {said}")
    return 1 if lost else 0


def waited_min(marked: str | None, now: float) -> int | None:
    if not marked:
        return None
    return int((now - calendar.timegm(time.strptime(marked, "%Y-%m-%dT%H:%M:%SZ"))) // 60)


def review(q: Queue, store, labels: StoreLabels, pr: int | None, verdict: str | None, ref: str | None, by: str | None,
           sha: str | None = None, as_json: bool = False, over: float | None = None,
           now: Callable[[], float] = time.time) -> int:
    from .labelauth import AuthError
    from .store.cli import _label_write_through

    q._clone()
    opened = {p.number: p for p in q.github.open_prs()}
    if pr is None:
        onto = [p for p in opened.values() if p.base == "main"]
        q.fetch(onto)
        q.unlock()
        main = _out(q.workdir, "rev-parse", "refs/remotes/origin/main")
        flags = flags_at(q.workdir, main)
        here = [p for p in onto if not _git(q.workdir, "cat-file", "-e", f"{p.head}^{{commit}}", check=False).returncode
                and _git(q.workdir, "merge-base", "--is-ancestor", p.head, main, check=False).returncode]
        labels.prefetch([(p.number, p.head) for p in here])
        rows = []
        for p in here:
            if not q.ready(p.number, p.head):
                continue
            if (why := unreviewed(labels.reviews(p.number, p.head), review_owners(q, labels, p.number, p.head, p.owner, opened))) is None:
                continue
            files = own_files(q.workdir, main, p.head)
            if lands_on_mark(flags, p, why, files):
                why = LOWRISK_ROW
            if refused := labels.refusals(p.number, p.head, vocab.REVIEWED):
                why += f"; not authenticated: {'; '.join(refused)}"
            base = _out(q.workdir, "merge-base", main, p.head)
            marked = q.marked_at(p.number, p.head) if q.marked_at else None
            rows.append({"pr": p.number, "head": p.head, "owner": p.owner, "lean": lean_change(q.workdir, p.head, base),
                         "files": sorted(files), "why": why, "marked": marked, "waited_min": waited_min(marked, now())})
        rows.sort(key=lambda r: (r["marked"] or "", r["pr"]))
        lacking = len(rows)
        if over is not None:
            rows = [r for r in rows if r["waited_min"] is None or r["waited_min"] > over]
        if as_json:
            print(json.dumps(rows, indent=1))
        else:
            for r in rows:
                shown = ", ".join(r["files"][:8]) + (f" (+{len(r['files']) - 8} more)" if len(r["files"]) > 8 else "")
                ago = f" ({r['waited_min']} min ago)" if r["waited_min"] is not None else ""
                print(f"#{r['pr']} {r['head'][:12]} {r['owner']} Lean change: {'yes' if r['lean'] else 'no'}, marked "
                      f"{r['marked'] or '?'}{ago}: {r['why']}\n    {shown}")
            print(f"research queue review: {lacking} marked PR(s) not on `main` lack a land verdict"
                  + (f"; {len(rows)} of them marked over {over:g} min ago or at a time not known" if over is not None else ""))
        return 0
    q.fetch()
    chain = [] if land_into_key(q.workdir, "refs/remotes/origin/main") == "main" else q.chain()
    m = next((m for m in chain if m.pr == pr), None)
    live = (p := opened.get(pr)) and ((p.branch and getattr(q.github, "branch_tip", lambda _b: None)(p.branch)) or p.head)
    if sha:
        # a full SHA is checked too: a mistyped one would record a verdict on a commit that doesn't exist, which never counts
        known = {h for h in (m and m.head, live) if h and h.startswith(sha)}
        r = _git(q.workdir, "rev-parse", "--verify", "-q", f"{sha}^{{commit}}", check=False)
        if len(known) != 1 and r.returncode:
            raise QueueError(f"review: --sha {sha} is neither #{pr}'s head on `{q.branch}` nor on GitHub, nor a commit the queue's "
                             f"clone has: pass {'the' if len(sha) < 40 else 'its'} full 40-character SHA")
        sha = known.pop() if len(known) == 1 else r.stdout.strip()
    if sha is None and m and live and live != m.head:
        raise QueueError(f"review: #{pr} is on `{q.branch}` at {m.head[:12]}, but GitHub has its head at {live[:12]}, pushed since "
                         f"drive's last sync: pass --sha with the head you read ({live} or {m.head})")
    head, owner = (sha or m.head, m.owner) if m and (sha in (None, m.head)) else (sha, None)
    if owner is None:
        if (open_pr := opened.get(pr)) is None:
            raise QueueError(f"review: #{pr} is neither on `{q.branch}` nor open")
        head, owner = head or open_pr.head, open_pr.owner
    if not owner or _who(by) in {_who(o) for o in review_owners(q, labels, pr, head, owner, opened) if o}:
        raise QueueError(f"review: #{pr} is {by}'s own PR (its owner, its owner's handle or its ready mark's author; or it has no "
                         "owner), and an owner's verdict doesn't count: a reviewer who isn't its owner reviews it")
    target = label_target(pr, head)
    try:
        store.add_label(target, vocab.REVIEWED, verdict, by, ref=ref or None)
    except AuthError as e:
        raise QueueError(f"review: #{pr}'s `{verdict}` at {head[:12]} isn't written: {e}") from None
    said, lost = _label_write_through(store, target)
    print(f"research queue: #{pr} reviewed `{verdict}` at {head[:12]} by {by} ({target}); {said}")
    return 1 if lost else 0


def delegate(store, agent: str, lane: str, days: int, holders: dict[str, str] | None,
             mint: Callable[[str], str] | None = None, now: Callable[[], float] = time.time,
             staffs: dict[str, tuple[str, ...]] | None = None) -> int:
    from . import labelauth
    from .store.cli import _label_write_through

    if not AGENT_RE.fullmatch(agent):
        raise QueueError(f"delegate: {agent!r} isn't a Cursor agent id (bc-, then a UUID)")
    if not 1 <= days <= labelauth.DELEGATION_MAX_DAYS:
        raise QueueError(f"delegate: --days {days}: 1 to {labelauth.DELEGATION_MAX_DAYS}")
    until = time.strftime(labelauth.TS_FORMAT, time.gmtime(now() + days * 86400))
    try:
        value = labelauth.delegation(lane, agent, until, mint or labelauth.cursor_mint)
        me = labelauth.unverified_claims(value["token"]).get("cloud_agent_id")
    except labelauth.AuthError as e:
        raise QueueError(f"delegate: {agent} can't be delegated {lane}: {e}") from None
    mine = _who(holders.get(me, "")) if holders is not None and isinstance(me, str) else ""
    if not labelauth.speaks_for(mine, _who(lane), staffs or {}):
        raise QueueError(f"delegate: this agent ({me}) doesn't hold {lane} in main's registry ({REGISTRY}), nor a handle whose "
                         f"`staffs` there names it, so its delegation would never count: {lane}'s holder runs it, or the "
                         "holder of the handle that staffs it")
    if agent == me:
        raise QueueError(f"delegate: {agent} is this agent, and an agent's delegation to itself never counts: delegate {lane} "
                         "to the reader's or the worker's own agent")
    target = labelauth.delegation_target(agent)
    store.add_label(target, vocab.LABEL_DELEGATION, value, mine)
    said, lost = _label_write_through(store, target)
    print(f"research queue: {agent} signs as {lane} until {until}, while this agent holds {mine} in main's registry "
          f"({target}); {said}")
    return 1 if lost else 0


def label_signer(store, repo: Path | None):
    from . import labelauth

    main, holders = "refs/remotes/origin/main", None
    if on := repo is not None and signed_at(repo, main) == "on":
        holders = holders_at(repo, main)
    return labelauth.Signer(store, labelauth.cursor_mint, required=on, holders=holders,
                            log=lambda s: print(f"research queue: {s}", file=sys.stderr))


def label_verifier(repo: Path, ref: str, keys: Path, log: Callable[[str], None] = print):
    from . import labelauth

    holders, staffs = holders_at(repo, ref), staffs_at(repo, ref)
    try:
        return labelauth.Verifier(labelauth.issuer_keys(keys, log=log), holders, staffs)
    except labelauth.AuthError as e:
        raise QueueError(f"no ready mark or land verdict can be authenticated (drive.toml's `signed_verdicts`): {e}") from None


def review_owners(q: Queue, labels: StoreLabels, pr: int, head: str, owner: str | None,
                  opened: dict[int, PR] | None = None) -> set[str | None]:
    p = (opened or {}).get(pr) or PR(pr, head, body=f"Lane {owner}")
    return {owner, q.handle(p), labels.ready_by(pr, head)}


def broker_gh(top: Path) -> str:
    common = _git(top, "rev-parse", "--path-format=absolute", "--git-common-dir", check=False).stdout.strip()
    wrapper = Path(common) / "verity-auth" / "bin" / "gh" if common else None
    return str(wrapper) if wrapper and wrapper.is_file() else (shutil.which("gh") or "gh")


def issue_line(spec: str) -> str:
    if spec.startswith("none:"):
        if not (reason := spec.removeprefix("none:").strip()):
            raise QueueError("open: --issue none:REASON needs the reason this PR has no Linear issue")
        return NO_ISSUE_LINE.format(reason=reason)
    if not ISSUE_KEY_RE.fullmatch(spec):
        raise QueueError(f"open: --issue {spec!r} is neither a Linear key (like ENG-123) nor none:REASON")
    return ISSUE_LINE.format(key=spec)


def with_issue(body: str, spec: str) -> str:
    line = issue_line(spec)
    return body if line in body.splitlines() else f"{body.rstrip()}\n\n{line}\n"


def issue_app(key: str) -> tuple[linear.Linear, str] | Exception:
    try:
        app = linear.Linear()
        return app, app.issue_id(key)
    except linear.NoIssue as e:
        raise QueueError(f"--issue {key}: {e}: give the PR's issue, or none:REASON; nothing sent") from None
    except (linear.LinearError, OSError) as e:
        return e


def link_issue(key: str, url: str, app: tuple[linear.Linear, str] | Exception | None = None) -> str:
    got = app or issue_app(key)
    try:
        if isinstance(got, Exception):
            raise got
        return f"{url} {'linked' if got[0].link_pr(got[1], url) else 'was linked already'} to {key} through the Linear app"
    except (linear.LinearError, OSError) as e:
        n = (m := re.search(r"/pull/(\d+)$", url)) and m.group(1)
        raise QueueError(f"{url} is not linked to {key} in Linear ({e}); link it with "
                         f"`research queue link {n or 'N'} --issue {key}`") from None


def goal_statuses(cache: Path | None = None) -> dict[str, str]:
    from . import goals
    from .slack import SlackError

    path = (cache or GOALS_CACHE).expanduser()
    try:
        got = {str(g["id"]): str(g.get("status")) for g in goals.Route().call()}
    except (SlackError, OSError, ValueError, KeyError, TypeError) as e:
        try:
            kept = json.loads(path.read_text())
            got = {str(g["id"]): str(g["status"]) for g in kept["goals"]}
        except (OSError, ValueError, KeyError, TypeError):
            raise QueueError(f"the Goals page can't be read ({e}) and this machine has no copy of it ({path})") from None
        print(f"research queue: the Goals page can't be read ({e}): using its {len(got)} goals as this machine read them at "
              f"{kept.get('read') or 'an unrecorded time'} ({path})", file=sys.stderr)
        return got
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"read": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                   "goals": [{"id": k, "status": v} for k, v in sorted(got.items())]}, indent=1) + "\n")
        tmp.replace(path)
    except OSError as e:
        print(f"research queue: the Goals page's goals aren't cached at {path} ({e})", file=sys.stderr)
    return got


def live_goals(names: Iterable[str], status: dict[str, str]) -> list[str]:
    return sorted({n for n in names if status.get(n, "done") != "done"})


def with_goals(body: str, key: str, app: tuple[linear.Linear, str] | Exception,
               read: Callable[[], dict[str, str]] | None = None) -> str:
    def warn(why: str) -> None:
        print(f"research queue open: warning: --issue {key}: {why}", file=sys.stderr)

    try:
        if isinstance(app, Exception):
            raise app
        labels = app[0].issue_labels(key)
    except (linear.LinearError, OSError) as e:
        warn(f"the Linear app can't read its labels ({e}): no `Goal` line written")
        return body
    try:
        status = (read or goal_statuses)()
    except QueueError as e:
        warn(f"{e}: no `Goal` line written")
        return body
    if not (live := live_goals(labels, status)):
        done = [n for n in labels if status.get(n) == "done"]
        warn(f"no label of it names a live goal of the Goals page (its labels: {', '.join(labels) or 'none'}"
             f"{'; done goals among them: ' + ', '.join(done) if done else ''}): label {key} with the goal it serves "
             f"(its id on the page, `research goals show`)")
    for g in GOAL_RE.findall(body):
        if g not in live:
            why = (f"a goal no label of {key} names" if g not in labels else "a done goal" if status.get(g) == "done" else
                   "no goal of the Goals page")
            warn(f"the body's `Goal {g}` names {why}" + (f": name one of {key}'s live goals ({', '.join(live)}), or drop the "
                                                        "line and `open` writes them" if live else ""))
    if GOAL_RE.search(body) or not live:
        return body
    lines = body.splitlines()
    at = lines.index(ISSUE_LINE.format(key=key)) + 1
    return "\n".join([*lines[:at], *(GOAL_LINE.format(id=g) for g in live), *lines[at:]]) + "\n"


def goal_gate(pr: PR, read: Callable[[], dict[str, str]] | None = None) -> str | None:
    if NO_ISSUE_RE.search(pr.body):
        return None
    fix = (f"add a `Goal <id>` line naming the live goal it serves (its id on the Goals page, `research goals show`), or an "
           f"`Issue: none: REASON` line, to its body (`research queue edit {pr.number} --body-file F`)")
    if not (named := GOAL_RE.findall(pr.body)):
        return f"its body names no goal: {fix}"
    try:
        status = (read or goal_statuses)()
    except QueueError as e:
        return f"{e}: its `Goal` lines go unchecked"
    if live_goals(named, status):
        return None
    dead = "; ".join(f"{g} is done" if g in status else f"the Goals page has no goal {g}" for g in dict.fromkeys(named))
    return f"its body names no live goal ({dead}): {fix}"


def named_issue(body: str, teams: Iterable[str] = ()) -> str | None:
    if m := ISSUE_LINK_RE.search(body):
        return m.group(1)
    if teams and (m := re.search(rf"\b(?:{'|'.join(map(re.escape, sorted(teams)))})-\d+\b", body)):
        return m.group(0)
    if m := NO_ISSUE_RE.search(body):
        return f"none: {m.group(1)}"
    return None


def _unnamed(prs: Iterable[PR], teams: Iterable[str]) -> str:
    prs = sorted(prs, key=lambda p: p.number)
    rows = [p for p in prs if named_issue(p.body, teams) is None]
    if not rows:
        return ""
    head = (f"\n\nno Linear issue named ({len(rows)} of {len(prs)} open PRs; add a `Refs KEY` or `Issue: none: REASON` line "
            "with `research queue edit N --body-file F`):")
    return head + "".join(f"\n  #{p.number:<5} {(p.owner or '-')[:24]:<24} {p.title}" for p in rows)


def open_pr(top: Path, repo: str, head: str, title: str, body: str, *, base: str = "main", sha: str | None = None,
            draft: bool = False, gh: str | None = None, run=subprocess.run) -> str:
    found = _git(top, "ls-remote", "origin", f"refs/heads/{head}", check=False).stdout.split()
    if not found:
        raise QueueError(f"open: no branch {head!r} on origin: push it first")
    if sha and not found[0].startswith(sha):
        raise QueueError(f"open: origin's {head} is at {found[0][:12]}, not {sha}: the head moved; re-check it with its owner")
    call = _gh_call("open", top, gh, run)
    existing = json.loads(call("pr", "list", "--repo", repo, "--head", head, "--state", "open", "--json", "number,url") or "[]")
    if existing:
        raise QueueError(f"open: {head} already has PR #{existing[0]['number']} ({existing[0]['url']}); nothing opened")
    out = call("pr", "create", "--repo", repo, "--head", head, "--base", base, "--title", title, "--body", body,
               *(["--draft"] if draft else []))
    return out.strip().splitlines()[-1]


def _gh_call(verb: str, top: Path, gh: str | None, run) -> Callable[..., str]:
    gh = gh or broker_gh(top)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

    def call(*args: str) -> str:
        r = run([gh, *args], cwd=top, capture_output=True, text=True, env=env, timeout=120)
        if r.returncode:
            raise QueueError(f"{verb}: gh {args[0]} {args[1]}: {(r.stderr or r.stdout).strip()[-300:]}")
        return r.stdout

    return call


CURSOR_BODY = ("<!-- CURSOR_AGENT_PR_BODY_BEGIN -->", "<!-- CURSOR_AGENT_PR_BODY_END -->")
CURSOR_FOOTER_RE = re.compile(r'<div><a href="https://cursor\.com/agents/bc-[^"]*cursor_ref=pr_footer')


def edited_body(old: str, new: str) -> str:
    begin, end = CURSOR_BODY
    foot = next((old[m.start():].strip() for m in reversed(list(CURSOR_FOOTER_RE.finditer(old)))
                 if end not in old or m.start() > old.rfind(end)), "")
    if m := CURSOR_FOOTER_RE.search(new):
        new = new[:m.start()]
    text = new.replace(begin, "").replace(end, "").strip("\n")
    if begin in old and end in old:
        text = f"{begin}\n{text}\n{end}"
    return text + (f"\n\n{foot}" if foot else "") + "\n"


def edit_pr(top: Path, repo: str, number: int, body: str, *, title: str | None = None, dry_run: bool = False,
            gh: str | None = None, run=subprocess.run) -> tuple[str, str]:
    call = _gh_call("edit", top, gh, run)
    cur = json.loads(call("pr", "view", str(number), "--repo", repo, "--json", "body,url"))
    text = edited_body(cur.get("body") or "", body)
    now = PR(number, "", body=text).owner
    if (was := LANE_RE.search(cur.get("body") or "")) and _who(now or "") != _who(was.group(1)):
        raise QueueError(f"edit: #{number}'s new body names owner {now or 'none'}, and the PR's is {was.group(1)}: keep its "
                         f"`Lane {was.group(1)}` line; nothing edited")
    if not dry_run:
        call("pr", "edit", str(number), "--repo", repo, *(("--title", title) if title else ()), "--body", text)
    return cur.get("url") or f"#{number}", text


def _stats(a: argparse.Namespace) -> int:
    from . import queue_stats
    from .merge import fetch_run
    from .store.local import LocalStore

    top = Path(_out(Path.cwd(), "rev-parse", "--show-toplevel"))
    _git(top, *STALL, "fetch", "-q", "origin", "main", check=False)
    heads = {}
    for body in _out(top, "log", "--first-parent", "--format=%B%x1e", "-n", "5000", "origin/main").split("\x1e"):
        t = dict(TRAILER_RE.findall(body))
        if "PR" in t and "Head" in t:
            heads.setdefault(int(t["PR"].lstrip("#")), t["Head"])
    labels = StoreLabels(LocalStore(a.store), log=lambda s: None)

    def marked(pr: int):
        head = heads.get(pr)
        if head is None:
            return None
        labels.ready(pr, head)
        at = labels.marked_at(pr, head)
        return queue_stats.ts(at) if at else None

    def run_dir(rid: str) -> Path:
        got = fetch_run(rid, ["steps.jsonl"], top, refresh=True)
        if got and not (got[0] / "slot.json").is_file():
            fetch_run(rid, ["slot.json"], top, refresh=True)
        return got[0] if got else Path("/nonexistent")

    return queue_stats.main(a.log, a.hours, a.until, marked, run_dir, a.json)


def _checked_heads(a: argparse.Namespace) -> int:
    from .merge import Refused, checked_heads
    from .store import bucket_index as BI
    from .store.local import LocalStore

    idx = BI.reader(LocalStore(a.store).root)
    if idx is None:
        print(f"research queue checked-heads: no store index: set ${BI.READER_ENV} (store_private's read-only URL) or build one "
              "here with `research data index`", file=sys.stderr)
        return 1
    try:
        heads = checked_heads(Path(_out(Path.cwd(), "rev-parse", "--show-toplevel")), a.tip, idx)
    except (QueueError, Refused, RuntimeError) as e:
        print(f"research queue checked-heads: {e}", file=sys.stderr)
        return 1
    if a.out is not None:
        a.out.write_text("".join(c + "\n" for c, _ in reversed(heads)))
        print(f"research queue checked-heads: wrote {len(heads)} checked train heads (newest first) to {a.out}, an "
              "optional verifier --main file (first-parent by default)", file=sys.stderr)
        return 0
    print(json.dumps([{"commit": c, "run": r} for c, r in heads], indent=1) if a.json else "\n".join(c for c, _ in heads))
    return 0


def _map_moves(a: argparse.Namespace) -> int:
    from .store.local import LocalStore

    top = Path(_out(Path.cwd(), "rev-parse", "--show-toplevel"))
    pairs = [x.partition("@") for x in (a.pr, *a.shard_run) if x]
    if bad := ["".join(p) for p in pairs if not (p[0].isdigit() and p[1] and p[2])]:
        raise SystemExit(f"research queue map-moves: {', '.join(bad)}: give each old head as N@HEAD, its PR's number and the head")
    prs = {h: int(n) for n, _, h in pairs}
    heads, labels = list(prs), StoreLabels(LocalStore(a.store), log=lambda s: print(s, file=sys.stderr))

    def refuse(h: str) -> str | None:
        try:
            return None if labels.lands(prs[h], h) else UNLANDED
        except QueueError as e:
            return str(e)

    def lacks(h: str) -> bool:
        return bool(_git(top, "cat-file", "-e", f"{h}^{{commit}}", check=False).returncode)

    _git(top, *STALL, "fetch", "-q", "origin", "main", check=False)
    for h in filter(lacks, heads):
        _git(top, *STALL, "fetch", "-q", "origin", h, check=False)
    missing = set(filter(lacks, heads))
    got = map_moves(top, heads, _out(top, "rev-parse", a.tip), a.jobs, refuse)
    for h in heads:
        _, own, why = got.get(h, ("", None, "no such commit here or on origin" if h in missing else "it crosses no move"))
        print(f"research queue map-moves: {h[:12]}: {own or why}", file=sys.stderr)
    text = json.dumps({key: own for key, own, _ in got.values() if own})
    if a.out is not None:
        a.out.write_text(text + "\n")
    else:
        print(text)
    return 0


def _cross(a: argparse.Namespace) -> int:
    from .clone import Refused, set_drivers
    from .store.local import LocalStore

    top = Path(_out(Path.cwd(), "rev-parse", "--show-toplevel"))
    remote, slug_ = remote_and_repo(a.remote, a.repo, top)
    push_to = "origin" if a.remote is None and a.repo is None else remote
    log = lambda s: print(s, file=sys.stderr)  # noqa: E731
    store = LocalStore(a.store)
    if a.from_dir is None and store.remote is None:
        raise SystemExit(f"research queue cross: the evidence store has no remote here ({store.remote_error or 'none configured'}), "
                         f"so no mark or land verdict can be read; run it from a shell with the store's secrets, or on a machine "
                         f"with a read-only store key: `research run --on auto --project verity --cwd clone --store-read -- "
                         f"research queue cross --repo {slug_}`")
    gh = GitHub(slug_) if shutil.which("gh") else PublicGitHub(slug_)

    def fetch(*shas: str) -> None:
        if (want := [s for s in shas if not _commit(top, s)]) and _git(top, *STALL, "fetch", "-q", "--no-tags", remote, *want,
                                                                         check=False).returncode:
            for s in want:
                _git(top, *STALL, "fetch", "-q", "--no-tags", remote, s, check=False)

    _git(top, *STALL, "fetch", "-q", "--no-tags", remote, "+refs/heads/main:refs/remotes/origin/main", check=False)
    if a.from_dir is not None:
        report = json.loads((a.from_dir / CROSS_REPORT).read_text())
        fetch(report["main"], *(c["old"] for c in report["crossed"]))
        if report["crossed"] and (r := _git(top, "fetch", "-q", str(a.from_dir / CROSS_BUNDLE), f"+{CROSS_NEW}*:{CROSS_NEW}*",
                                            check=False)).returncode:
            raise SystemExit(f"research queue cross: {a.from_dir / CROSS_BUNDLE} doesn't fetch: {r.stderr.strip()[-300:]}")
        report["pushed"] = cross_push(top, report, push_to, {p.number: p for p in gh.open_prs()}, log)
        (a.from_dir / CROSS_PUSHED).write_text(json.dumps(report["pushed"], indent=1) + "\n")
        print(cross_table(report))
        return 0
    if _git(top, "rev-parse", "--verify", "-q", f"{a.main}^{{commit}}", check=False).returncode:
        fetch(a.main)
    main = _out(top, "rev-parse", "--verify", f"{a.main}^{{commit}}")
    fetch(*a.move_commit)
    named = [_out(top, "rev-parse", "--verify", f"{m}^{{commit}}") for m in a.move_commit]
    for m in named:
        if _git(top, "merge-base", "--is-ancestor", m, main, check=False).returncode:
            raise SystemExit(f"research queue cross: --move-commit {m[:12]} isn't on --main {main[:12]}")
        if not MOVE_LINE_RE.search(_out(top, "log", "-1", "--format=%B", m)):
            raise SystemExit(f"research queue cross: --move-commit {m[:12]} has no `Move:` or `Layout-move:` line")
    prs = [p for p in gh.open_prs() if p.base == "main"]
    _git(top, *STALL, "fetch", "-q", "--no-tags", remote, *(f"+refs/pull/{p.number}/head:{CROSS_PR}{p.number}" for p in prs),
         check=False)
    fetch(*(p.head for p in prs))
    labels = StoreLabels(store, log=log)
    q = Queue(top, remote=remote, github=gh, ready=labels.ready, store=store, log=log, marked_by=labels.marked_by)
    opened = {p.number: p for p in prs}
    owns: dict[str, str | None] = {}

    def change(h: str) -> str | None:
        if h not in owns:
            fetch(h)
            mb = _git(top, "merge-base", main, h, check=False).stdout.strip() if _commit(top, h) else ""
            owns[h] = own_change(top, mb, h) if mb else None
        return owns[h]

    flags = flags_at(top, main)
    labels.since = flags.marked_after or ((rules_at(top, main) or Rules()).marked_after)
    labels.change, labels.owners = change, lambda n: {opened[n].owner, q.handle(opened[n])} if n in opened else set()
    if flags.signed_verdicts == "on":
        labels.verify = label_verifier(top, main, Path.home() / ".research" / "cache" / "issuer-keys.json", log)
    labels.prefetch([(p.number, p.head) for p in prs])

    def verdict(n: int, head: str) -> str | None:
        if labels.lands(n, head):
            return head
        own = change(head)
        return next((o for o in labels._other_heads(n, head, vocab.REVIEWED) if own and change(o) == own and labels.lands(n, o)),
                    None)

    try:
        set_drivers(top)
    except Refused as e:
        raise SystemExit(f"research queue cross: the merge drivers restack.py merges with can't be set here: {e}") from None
    todo, skipped = cross_plan(top, prs, main, named, labels.ready, verdict, set(a.only or ()), slug_)
    log(f"research queue cross: {len(todo)} to cross, {len(skipped)} skipped; crossing {a.jobs} at once")
    with scratch() as d:
        crossed, refused, stacked = cross_all(top, todo, main, a.jobs, kept=CROSS_KEPT, log=log,
                                              root=lambda c: root_failures(top, c, CROSS_ROOT_SYNC, CROSS_ROOT_PYTHON),
                                              drivers=cross_drivers(top, main, Path(d)))
    report = {"main": main, "moves": named or list(dict.fromkeys(m for t in todo.values() for m, _ in t["moves"])),
              "crossed": crossed, "refused": refused, "skipped": skipped + stacked}
    out = a.out or Path("cross")
    cross_write(top, out, report)
    if a.push:
        report["pushed"] = cross_push(top, report, push_to, {p.number: p for p in gh.open_prs()}, log)
        (out / CROSS_PUSHED).write_text(json.dumps(report["pushed"], indent=1) + "\n")
    print(cross_table(report))
    log(f"research queue cross: the report and the bundle are in {out}")
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["vqueue"]:
        from .vqueue_loop import main as vqueue_main

        return vqueue_main(argv[1:])
    if argv[:1] == ["stranded"]:
        from .stranded import main as stranded_main

        return stranded_main(argv[1:])
    ap = argparse.ArgumentParser(prog="research queue", description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["status", "sync", "eject", "ready", "judge", "open", "edit", "link", "stats",
                                            "checked-heads", "review", "delegate", "shadow-report", "vqueue", "map-moves",
                                            "cross", "stranded"])
    ap.add_argument("pr", nargs="?", help="eject, ready, review, edit, link: the PR number; judge: the quick tier's run id; "
                    "delegate: the Cursor agent id to delegate --by's handle to; map-moves: the first old head, as N@HEAD")
    ap.add_argument("shard_run", nargs="*", metavar="N@HEAD", help="map-moves: the other old heads")
    ap.add_argument("--jobs", type=int, default=1, metavar="N", help="map-moves: the heads mapped at once, each running a move's "
                    "script on a whole tree in a worktree of its own under --preflight-root; cross: the stacks crossed at once "
                    "(default 1)")
    ap.add_argument("--main", default="origin/main", metavar="REF", help="cross: the moved `main` whose moves the heads cross "
                    "and whose tools/move/restack.py crosses them (default origin/main)")
    ap.add_argument("--move-commit", action="append", default=[], metavar="SHA", help="cross: a move to cross, a commit of "
                    "--main with a `Move:` or `Layout-move:` line, repeated (default: every move --main has and a head lacks); "
                    "a head crosses those of them restack.py crosses finding them itself, which skips one the branch it "
                    "carries is based after where a later one's base holds it (`named_moves`)")
    ap.add_argument("--only", type=int, nargs="+", metavar="N", help="cross: only these PRs; one stacked on a PR it doesn't name "
                    "is skipped, as its stack doesn't cross")
    ap.add_argument("--push", action="store_true", help="cross: push each crossing that is clean to its PR's own branch, with a "
                    "lease on its old head (default: a dry run, which pushes nothing)")
    ap.add_argument("--from", dest="from_dir", type=Path, metavar="DIR", help="cross --push: push what a run left in DIR (its "
                    f"{CROSS_REPORT} and {CROSS_BUNDLE}, from `research fetch RUN --all`) instead of crossing anew")
    ap.add_argument("--reason", help="eject: why, for the owner's notice")
    ap.add_argument("--by", default=os.environ.get("RESEARCH_LANE"),
                    help="eject, ready, judge: who ejects it, the handle its notice is posted as (default merge-queue, which no "
                         "one holds, so the notice fails) or says it's ready (no default: the owner's lane), "
                         "default $RESEARCH_LANE")
    ap.add_argument("--shadow", action="store_true", help=f"build `{SHADOW_BRANCH}`, never push `main`, send no notices")
    ap.add_argument("--dry-run", action="store_true", help="sync: decide and report, push nothing, tell no one; open, edit: "
                                                           "say what it would send, and send nothing")
    ap.add_argument("--workdir", type=Path, default=None, help="the queue's own clone (default: ~/.research/queue/NAME for --repo "
                                                              "OWNER/NAME, else ~/.research/queue/verity)")
    ap.add_argument("--remote", default=None, help="the repository to clone (default: https://github.com/OWNER/NAME.git for --repo "
                                                   "OWNER/NAME, else this checkout's origin); one naming another repository on "
                                                   "GitHub than --repo is refused")
    ap.add_argument("--repo", default=None, help="OWNER/NAME on GitHub (default: from the remote's URL)")
    ap.add_argument("--rules", type=Path, default=None, help=f"the admission rules (default: {RULES} on `main`)")
    ap.add_argument("--store", default=None, help="evidence store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--json", action="store_true", help="status: as JSON; ready: one JSON object on stdout (pr, head, base, exit, "
                    "warning), the rest on stderr")
    ap.add_argument("--on", metavar="MACHINE", help="ready: run the quick tier as a `research run` there, in a check slot (`auto` or "
                                                    "M1,M2: the one with a free check slot, as `check.py --on` picks), waiting for it "
                                                    f"{waitcap.WAIT_LIMIT_S} s at most (then exit {waitcap.STILL_GOING}, the run going "
                                                    "on: `judge RUN` marks the PR once it ends); judge: look "
                                                    "for a run another VM launched there only")
    ap.add_argument("--local", action="store_true", help="ready: run the quick tier here")
    ap.add_argument("--mark", action="store_true", help="ready: mark it with no quick tier (its candidate's check tests it)")
    ap.add_argument("--again", action="store_true", help="ready: mark a head that was ejected (only when the ejection was wrong)")
    ap.add_argument("--preflight-root", type=Path, default=None, metavar="DIR",
                    help="ready, sync: where merged trees go, ready's preflight's and every other (default: the temporary "
                         "directory); a tree must share a filesystem with uv's cache for its sync to hardlink (on a host whose "
                         "TMPDIR is a network volume, pass a directory on its local disk)")
    ap.add_argument("--no-preflight", action="store_true",
                    help="ready: skip the preflight: the head merged onto the branch `Preflight._fetch` fetches must merge cleanly, contain "
                         "no open PR that isn't admitted, and pass `pins.py --stale`, the invariant tests (module strings, pins, "
                         "doc paths, pythonpath) and `pytest --collect-only` of the suites its change touches, in a temporary "
                         f"worktree, in about {PREFLIGHT_CHECKS_S:g} s, and build the Lean modules that are or import what it "
                         f"changes (at most {PREFLIGHT_LEAN_MODULES}, in {PREFLIGHT_LEAN_S:g} s, on a warm Lean tree: here on a "
                         f"node, else on the first of {', '.join(PREFLIGHT_LEAN_ON)} that starts the build, whatever the tier; "
                         "one no node can build refuses the mark, one not "
                         "built in time is a skipped check), and its warnings with it (the vLLM ratchet gate's, and the marked PRs "
                         f"the head doesn't merge with, read in {PREFLIGHT_COLLIDE_S:g} s of their own before those checks)")
    ap.add_argument("--marked-after", metavar="YYYY-MM-DDTHH:MM:SSZ", help="status/sync: admit only ready marks written at "
                                                                           "or after this time (default: `marked_after` in the rules file)")
    ap.add_argument("--first", help=argparse.SUPPRESS)
    ap.add_argument("--priority-file", type=Path, metavar="PATH",
                    help="sync: the priority list, the landing order: a file of PR numbers separated by spaces or "
                         "commas, first to land first (a number repeated keeps its first place), read again at each sync (none: "
                         "an empty list; one it can't read: an error logged once, and an empty list); the verified queue reads "
                         "its own (`research queue vqueue --priority-file`)")
    ap.add_argument("--land", action="store_true", help="review N: the verdict that lands it")
    ap.add_argument("--hold", metavar="REASON", help="review N: the verdict that keeps it, and every PR containing it, out of "
                                                     "the verified queue's candidates")
    ap.add_argument("--ref", help="review N --land: the note or reason the verdict rests on")
    ap.add_argument("--over", type=float, metavar="MINUTES", help="review: list only the PRs marked more than MINUTES ago (or at "
                                                                  "a time not known)")
    ap.add_argument("--head", help="open: the pushed branch to open a PR for")
    ap.add_argument("--sha", help="open: refuse unless origin's branch is at this commit (the owner's pinned head); ready: mark only "
                                  "this head, refusing when GitHub still shows another after a few reads (push && ready, safely); "
                                  "eject: only while this is the PR's head, so an eject never withdraws the mark of a head "
                                  "pushed meanwhile")
    ap.add_argument("--title", help="open: the PR's title; edit: its new title")
    ap.add_argument("--body-file", type=Path, help="open: the PR's body, verbatim but for --issue's last line (its first line "
                                                   "`Lane NAME` names the owner); "
                                                   "edit: the PR's whole new body, in place of all its text")
    ap.add_argument("--issue", metavar="KEY|none:REASON",
                    help="open (required): the PR's Linear issue, a key like ENG-123, written into the body as a `Refs KEY` line "
                         "and linked to the PR through the Linear app (Linear's GitHub integration is off for this "
                         "repository, so the line alone links nothing), with a `Goal <id>` line after it for each live goal "
                         "of the Goals page a label of the issue names (a warning on stderr when none does); or none:REASON, written as an "
                         "`Issue: none: REASON` line, for a PR with no issue; link (required): the KEY to link the open PR to")
    ap.add_argument("--base", default="main", help="open: the base branch (default main)")
    ap.add_argument("--tip", default="origin/main", help="checked-heads: whose first-parent line; map-moves: the landing branch "
                    "whose moves the heads cross (default origin/main)")
    ap.add_argument("--out", type=Path, metavar="FILE", help="checked-heads: write the checked train heads newest first "
                    "to FILE (an optional source for the verifier's --main file) instead of listing them on stdout; map-moves: "
                    f"write the JSON there instead; cross: the directory for {CROSS_REPORT}, {CROSS_BUNDLE} and {CROSS_MAPPED} "
                    "(default ./cross; in a `research run`, give it \"$OUT\")")
    ap.add_argument("--draft", action="store_true", help="open: as a draft (default: ready for review)")
    ap.add_argument("--log", type=Path, help="stats: a log drive wrote (its stdout, every tick's), for the split of drive's "
                    "landings; without it, the queue's table")
    ap.add_argument("--tips", type=int, default=10, metavar="N", help="shadow-report: the newest N tips whose shadow records "
                    "compare a check with the executor's verdicts (default 10)")
    ap.add_argument("--short", action="store_true", help="stats: the table without the column saying how each number is measured")
    ap.add_argument("--rounds", action="store_true", help="stats: one row for each candidate built in the window, instead of the table")
    ap.add_argument("--reds", action="store_true", help="stats: each candidate being checked, its members above `main`, its shards' "
                    "runs and steps, and only the reds on the merge gate, instead of the table")
    ap.add_argument("--hours", type=float, default=24.0, help="stats: the window's length (default 24)")
    ap.add_argument("--days", type=int, default=7, help="delegate: how long the delegation lasts (default 7, at most 14)")
    ap.add_argument("--until", metavar="YYYY-MM-DDTHH:MM:SSZ", help="stats: the window's end (default now); for the table, a past "
                    "time with its UTC offset")
    a = ap.parse_args(argv)
    if a.command == "vqueue":
        ap.error("vqueue comes first, its options after it: research queue vqueue --help")
    if a.first is not None:
        ap.error(FIRST_RETIRED)
    global MERGED_ROOT
    MERGED_ROOT = a.preflight_root or MERGED_ROOT
    if a.command == "stats":
        if a.log is None:
            from . import queue_table

            if a.rounds and a.reds:
                ap.error("stats: --rounds or --reds, not both")
            return queue_table.main(a.store, a.json, short=a.short, until=a.until, by_round=a.rounds, by_red=a.reds)
        return _stats(a)
    if a.command == "checked-heads":
        return _checked_heads(a)
    if a.command == "map-moves":
        return _map_moves(a)
    if a.command == "cross":
        if a.from_dir is not None and not a.push:
            ap.error("cross: --from DIR goes with --push: it pushes what a dry run left in DIR")
        return _cross(a)
    if a.command == "shadow-report":
        from .executor import shadow_records, shadow_report
        from .store.local import LocalStore

        text, _ = shadow_report(shadow_records(LocalStore(a.store), log=lambda s: print(s, file=sys.stderr)), max(a.tips, 1))
        print(text)
        return 0
    if a.issue is not None and a.command not in ("open", "link"):
        ap.error("--issue is for open and link only")
    if a.command == "link":
        if not (a.pr and a.pr.isdigit() and a.issue and ISSUE_KEY_RE.fullmatch(a.issue)) or a.dry_run:
            ap.error("link needs the PR number and --issue KEY (its Linear issue, like ENG-123), and has no --dry-run")
        try:
            repo = remote_and_repo(a.remote, a.repo, Path.cwd())[1]
            print(f"research queue link: {link_issue(a.issue, f'https://github.com/{repo}/pull/{a.pr}')}")
            return 0
        except (QueueError, OSError) as e:
            print(f"research queue: {e}", file=sys.stderr)
            return 1
    if a.command == "open":
        if not (a.head and a.title and a.body_file and a.issue):
            ap.error("open needs --head BRANCH, --title T, --body-file F and --issue KEY (the PR's Linear issue, like ENG-123, "
                     "or none:REASON)")
        try:
            body = with_issue(a.body_file.read_text(), a.issue)
            top = Path(_out(Path.cwd(), "rev-parse", "--show-toplevel"))
            repo = remote_and_repo(a.remote, a.repo, top)[1]
            app = issue_app(a.issue) if ISSUE_KEY_RE.fullmatch(a.issue) else None
            if app is not None:
                body = with_goals(body, a.issue, app)
            if a.dry_run:
                link = f"; would link it to {a.issue} through the Linear app" if app else ""
                print(f"research queue open: would open {a.head}{'@' + a.sha[:12] if a.sha else ''} onto {a.base} in {repo}"
                      f"{' as a draft' if a.draft else ''}: {a.title!r} ({len(body)} bytes of body, ending "
                      f"{body.rstrip().splitlines()[-1]!r}{link}; dry run)")
                return 0
            url = open_pr(top, repo, a.head, a.title, body, base=a.base, sha=a.sha, draft=a.draft)
        except (QueueError, OSError) as e:
            print(f"research queue: {e}", file=sys.stderr)
            return 1
        print(url)
        if app:
            try:
                print(f"research queue open: {link_issue(a.issue, url, app)}")
            except QueueError as e:
                print(f"research queue: {e}", file=sys.stderr)
        return 0
    if a.command == "edit":
        if not (a.pr and a.pr.isdigit() and a.body_file):
            ap.error("edit needs the PR number and --body-file F")
        try:
            top = Path(_out(Path.cwd(), "rev-parse", "--show-toplevel"))
            repo = remote_and_repo(a.remote, a.repo, top)[1]
            url, body = edit_pr(top, repo, int(a.pr), a.body_file.read_text(), title=a.title, dry_run=a.dry_run)
            titled = f" and its title with {a.title!r}" if a.title else ""
            if a.dry_run:
                print(f"research queue edit: would replace {url}'s body{titled} with ({len(body)} bytes; dry run):\n{body}",
                      end="")
            else:
                print(f"research queue edit: {url}'s body replaced ({len(body)} bytes){titled}")
            return 0
        except (QueueError, OSError) as e:
            print(f"research queue: {e}", file=sys.stderr)
            return 1
    if a.priority_file and a.command != "sync":
        ap.error("--priority-file is for sync only")
    if a.marked_after and not MARK_TIME_RE.fullmatch(a.marked_after):
        ap.error(f"--marked-after {a.marked_after!r}: a UTC time as YYYY-MM-DDTHH:MM:SSZ, as labels record it")
    if a.dry_run and a.command != "sync":
        ap.error(f"--dry-run is for sync only: `{a.command}` would act for real")
    if a.command == "ready" and bool(a.on) + a.local + a.mark != 1:
        ap.error("ready takes one of --on MACHINE (the quick tier as a run there, in a check slot), --local (here) and --mark (no tier)")
    if a.no_preflight and a.command != "ready":
        ap.error("--no-preflight is for ready only")
    if a.command == "judge" and not a.pr:
        ap.error("judge needs the quick tier's run id")
    if a.command in ("ready", "judge") and not a.by:
        ap.error(f"{a.command} writes the owner's ready mark: pass --by <lane> (or set $RESEARCH_LANE)")
    if (a.land or a.hold is not None or a.ref) and not (a.command == "review" and a.pr):
        ap.error("--land, --hold and --ref are for review N only")
    if a.over is not None and not (a.command == "review" and not a.pr):
        ap.error("--over is for the review listing only (review with no PR number)")
    if a.command == "review" and a.pr and (a.land == (a.hold is not None) or not a.by):
        ap.error("review N records one verdict: --land [--ref NOTE] or --hold REASON, with --by <reviewer lane> (or $RESEARCH_LANE)")
    if a.command == "delegate" and not (a.pr and a.by):
        ap.error("delegate AGENT lets a Cursor agent (a worker) sign as your handle: pass its agent id and --by <your lane>")
    a.by = a.by or "merge-queue"
    if a.command not in ("judge", "delegate") and a.pr is not None:
        if not a.pr.isdigit():
            ap.error(f"{a.command}: {a.pr!r} is not a PR number")
        a.pr = int(a.pr)
    from .store.local import LocalStore

    try:
        remote, repo = remote_and_repo(a.remote, a.repo, Path.cwd())
        workdir = a.workdir or (WORKDIR.with_name(repo.split("/")[1].lower()) if a.repo else WORKDIR)
        store = LocalStore(a.store)
        here = _git(Path.cwd(), "rev-parse", "--show-toplevel", check=False).stdout.strip()
        main_ref, keys = "refs/remotes/origin/main", workdir.parent / f"{workdir.name}.issuer-keys.json"
        if a.command == "delegate":
            return delegate(store, a.pr, a.by, a.days, holders_at(Path(here), main_ref) if here else None,
                            staffs=staffs_at(Path(here), main_ref) if here else None)
        if a.command in ("judge", "ready"):
            store = label_signer(store, Path(here) if here else None)
        if a.command == "judge":
            return judge(GitHub(repo), store, a.pr, a.by, on=a.on, repo=Path(here) if here else None, url=remote,
                         goals=goal_gate)
        if a.command == "ready":
            top = Path(_out(Path.cwd(), "rev-parse", "--show-toplevel"))
            tier = MarkOnly() if a.mark else RemoteQuickTier(top, remote, a.on, by=a.by) if a.on else QuickTier(top, remote)
            pre = None if a.no_preflight else Preflight(top, remote, store, lean=LakeBuild(on=PREFLIGHT_LEAN_ON, by=a.by),
                                                             root=a.preflight_root)
            return ready(GitHub(repo), store, a.pr, a.by, tier, again=a.again, preflight=pre, sha=a.sha, as_json=a.json,
                         goals=goal_gate)
        labels = StoreLabels(store, log=lambda s: print(s, file=sys.stderr), since=a.marked_after)
        q = Queue(workdir, remote=remote, github=GitHub(repo), ready=labels.ready, store=store, shadow=a.shadow,
                  rules=Rules.parse(a.rules.read_text(), str(a.rules)) if a.rules else None,
                  auth=credential_args(Path.cwd(), remote),
                  before_cutoff=labels.before_cutoff, marked_by=labels.marked_by, marked_at=labels.marked_at,
                  prefetch=labels.prefetch)
        lock = _lock(workdir)
        q.unlock = lock.close
        own = weakref.WeakMethod(q.own_change)
        labels.change = lambda head: own()(head)
        if not a.marked_after and a.command != "status":
            if a.rules is None:
                q._clone()
                q.fetch()
            labels.since = q.rules().marked_after
        if a.command == "status":
            from .stale import stale_prs

            def at_main() -> Callable[[list[PR]], dict[int, float]]:
                if not a.marked_after:
                    labels.since = q.rules().marked_after
                r = _git(q.workdir, "show", f"refs/remotes/origin/main:{DRIVE_FLAGS}", check=False)
                got = DriveFlags.parse(r.stdout, f"{DRIVE_FLAGS} at main") if r.returncode == 0 else DriveFlags()
                return lambda prs: stale_prs(q, prs, q.now(), got.stale_days,
                                             lambda n, h: [lab.ts for lab in labels.reviews(n, h)], keep=got.stale_keep)

            with contextlib.redirect_stdout(sys.stderr) if a.json else contextlib.nullcontext():
                entries = q.status(fetched=at_main, loop=True, reviews=labels.reviews)
                teams = q.issue_teams()
            if a.json:
                print(json.dumps([{"pr": e.pr.number, "head": e.pr.head, "owner": e.pr.owner, "draft": e.pr.draft, "state": e.state,
                                   "why": e.why, "position": e.position, "commit": e.commit, "after": list(e.after),
                                   "issue": named_issue(e.pr.body, teams),
                                   "stale": e.idle_since is not None,
                                   "idle_since": None if e.idle_since is None else
                                   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(e.idle_since))}
                                  for e in sorted(entries.values(), key=lambda e: e.pr.number)], indent=1))
            else:
                print(_render(entries) + _unnamed(q.listed, teams))
            return 0
        if a.command == "review":
            if a.pr is None and here and signed_at(Path(here), main_ref) == "on":
                labels.verify = label_verifier(Path(here), main_ref, keys, lambda s: print(f"research queue: {s}", file=sys.stderr))
            return review(q, label_signer(store, Path(here) if here else None) if a.pr else store, labels, a.pr,
                          "land" if a.land else "hold" if a.pr else None, a.hold or a.ref, a.by, sha=a.sha, as_json=a.json,
                          over=a.over)
        from .merge import off_drive, on_drive

        if a.command == "sync" and not (a.dry_run or on_drive()):
            raise QueueError(off_drive(q.branch, "queue sync"))
        if a.command == "sync":
            q.priority_file = a.priority_file
            q.sync(dry_run=a.dry_run)
            return 0
        if a.pr is None or not a.reason:
            ap.error("eject needs a PR number and --reason")
        told = q.tell_ejected(q.eject(a.pr, a.reason, a.by, sha=a.sha), a.reason, a.by)
        print(f"research queue: #{a.pr}'s mark is withdrawn; the verified queue leaves it out until a ready mark re-admits it, "
              "and a candidate in flight holding it still lands it as built unless the eject is ci's or not by its "
              "owner")
        return told
    except QueueError as e:
        print(f"research queue: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
