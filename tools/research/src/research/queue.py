"""`research queue`: the merge queue (infra plan change 5).  `next` is `main` plus every admitted PR, one merge commit each, in the
order they were admitted.  Drive (`research.drive`) fast-forwards `main` to the top of the longest prefix of `next` whose every
member has a reviewer's land verdict at its head: PRs land on an eyeball review, and the full check runs after landing.

    research queue status [--json]                 each open PR: its place in `next`, what it waits for, or why it was refused;
                                                   `stale` when it is unmarked at its head and idle drive.toml's `stale_days`
                                                   (`research.stale`), which drive closes at `stale_close_days`
    research queue sync [--dry-run]                admit and drop PRs, merge them onto `next`, push it
    research queue eject PR --reason TEXT [--by HANDLE]
                                                   keep the PR's current head out of `next`; a new push or ready mark re-admits it;
                                                   posted as HANDLE to @ci and the owner in the merge thread (exit 1 if it couldn't be)
    research queue ready PR (--on MACHINE | --local | --mark) [--by LANE] [--no-preflight] [--sha SHA] [--json]
                                                   the owner's word that the PR is ready at its current head (a new push needs it again;
                                                   --sha: only at SHA, refused when GitHub still has another head after a few reads),
                                                   once the quick tier passes there: as a run on MACHINE in a check slot
                                                   (`RemoteQuickTier`; `auto` or M1,M2: the one with a free slot), or here
                                                   (`QuickTier`); exit 3 when it never started (no slot within its wait), 79 when
                                                   --on's run is still going after 20 min (research.waitcap: `judge RUN` marks it
                                                   once it ends); or, with
                                                   --mark, on the owner's word alone (`MarkOnly`).  First the records gate
                                                   (`RecordsGate`, which --no-preflight keeps) refuses a head that changes a Lean
                                                   lock against `main` unless a records run or Lean check in the store passed
                                                   that package, of the head or of a commit whose files of the package and the
                                                   packages it requires (tests and docs aside), lean-toolchain and the Lean
                                                   checker (tools/verity/lean) are the head's, naming the run to start; then,
                                                   in about a minute, the preflight (`Preflight`) refuses, with
                                                   the fix, a head that conflicts with a
                                                   fresh `next` (naming the paths), contains an open PR that isn't admitted
                                                   (naming it), fails a landing gate of drive.toml at `main` (with
                                                   the gate's fix), or, merged onto `next` in a temporary worktree, fails `pins.py
                                                   --stale`, tests/test_module_strings.py, test_pins.py, test_doc_paths.py,
                                                   test_repository.py, tools/research/tests/test_pythonpath.py, or `pytest
                                                   --collect-only` in a suite
                                                   whose inputs its change touches, or breaks the build of a Lean module that
                                                   is or imports what it changes (on a warm Lean tree: here on a node, else
                                                   with --on on vy-nebius-cpu-3), naming the PR on `next` that owns it;
                                                   --no-preflight skips it (the lander's)
    research queue drive (--on MACHINE | --config PATH [--config-from-tree]) [--marked-after TS] [--first N,...]
                         [--[no-]regenerate-locks] [--[no-]after-landing-no-cache] [--dry-run]
                                                   one tick of the landing automation (`research.drive`): sync and land the
                                                   reviewed prefix of `next`, a Lean change once its Lean check passed, then
                                                   check `main` (--after-landing-no-cache: reusing no pass but
                                                   circuit-check's); tell @ci the rest.
                                                   --config reads the flags (`DriveFlags`) from PATH at `main` in the queue's clone,
                                                   fresh each tick, so a change to them is a reviewed commit (drive's of record:
                                                   tools/verity/check/drive.toml); a flag given here overrides the file's, the
                                                   effective flags print once a tick, and --config-from-tree reads this file system's
    research queue review [N (--land [--ref NOTE] | --hold REASON) --by LANE] [--sha SHA] [--json]
                                                   without N: `next`'s members above `main` that no land verdict at their head
                                                   lands, bottom first (number, head, owner, files, whether it waits for a Lean
                                                   check: `lean_change`); with N: a reviewer's verdict, a `reviewed` label on
                                                   `pr:N@HEAD` (its head on `next`, or --sha, a full or short SHA), which drive's
                                                   review landing counts unless LANE is the PR's owner, its handle or its ready
                                                   mark's author, refused here; without --sha, refused when GitHub has another
                                                   head than `next` (pushed since drive's last sync)
    research queue baseline SHA --by ci             record `main` at SHA as green for the review landing, which otherwise first
                                                   runs the after-landing run on `main` itself, and lands nothing until it ends
                                                   (on the control pod: `/data/infra/drive.sh queue baseline SHA --by ci`)
    research queue lean-check COMMIT [--on MACHINES] [--by LANE] [--again]
                                                   a hand Lean check (`train.sh COMMIT MACHINES --deferred lean-audit`, MACHINES
                                                   default drive.toml's `on`) of a member's commit on `next`, recorded in drive's
                                                   state at its launch, so it holds `next` and lands what it covers as drive's own
                                                   (`Driver.hand_lean`); refused off `next`, under a running check at or above
                                                   it, or where drive records a passed one at it (--again reruns it)
                                                   (on the control pod: `/data/infra/drive.sh queue lean-check COMMIT`)
    research queue judge RUN [--on MACHINE] [--by LANE]
                                                   the mark `ready --on` writes, from its quick-tier run RUN that already ended
                                                   (its caller gone), for the open PR at the head the tier ran at, or at its tree
    research queue open --head BRANCH --title T --body-file F [--sha SHA] [--base main] [--draft]
                                                   open a PR for any pushed branch through gh (the repo's broker wrapper when the
                                                   checkout has one), for a lead whose PR tool opens PRs only from its own run
                                                   branch; refuses when BRANCH isn't on the remote, isn't at --sha, or already
                                                   has an open PR; prints the PR's URL (`open_pr`)
    research queue edit PR --body-file F [--dry-run]
                                                   replace the PR's body with F outright, through gh as `open` does, keeping only
                                                   the Cursor PR tool's markers and footer (`edit_pr`); refuses a body that drops or
                                                   changes the PR's `Lane NAME` owner; --dry-run prints the body it would write
    research queue stats --log FILE [--hours H] [--until TS] [--json]
                                                   where landed PRs' time went, from drive's log (`research.queue_stats`)
    research queue checked-heads [--tip REV] [--json] [--out FILE]
                                                   the commits on REV's first-parent line (default origin/main) whose own `check`
                                                   passed and landed them, oldest first, one a line: each train's head, by drive's
                                                   `landed` labels in the store index (`merge.checked_heads`), less a head whose
                                                   train a later `git revert` names. --out FILE writes them newest first, an
                                                   optional source for the verifier's --main file (`merge.main_line`), for a
                                                   deployment that wants credit's line to be the checked train heads rather than
                                                   the raw `git rev-list --first-parent origin/main` infra writes by default
    common flags: [--shadow] [--workdir DIR] [--rules FILE] [--repo OWNER/NAME] [--remote URL]

Admission.  A PR is admitted while it is open against `main`, has an owner, carries no hold label, and is marked ready at its head;
no change waits on a per-head grant.  A PR based on another branch is never admitted: `ready` marks it and says so,
and `status` lists it, once marked, as waiting to be retargeted to `main` (a PR containing another waits for it there).  Ready is a store label `ready = true` on `pr:{number}@{head sha}` that
`research queue ready PR` writes, so a new push needs a new one (a restack too: the tier ran on the old tree; not a head whose tree is a marked head's,
as a merge of `main` that changes no file leaves it: `ready` marks that one without a tier), and only after the quick tier passes at that
head: `suites.py --quick` on the suites the PR changes, from a clean checkout, either as a `research run` on a machine (`--on`,
`RemoteQuickTier`: in one of its check slots, so several PRs' tiers run at once and the caller only waits) or here (`--local`,
`QuickTier`); a failure refuses and names the tests, and a remote run whose result can't be read, or that reports another head, is a
failure.  `--mark` (`MarkOnly`) writes the mark with no tier, and the after-landing run is the head's first full check.  Draft state is ignored: agents open PRs as drafts, and
the ready label is what says one is done.  The hold label keeps a PR out whatever else holds.  The owner is the lane its body names
on a `Lane {name}` line, else the Cursor agent id its body carries.  The rules file (`ci/queue.toml` on
`main`, `tools/verity/check/queue.toml` until the repository move, or `--rules`) names the hold label.  Ready marks are read after
pulling the target's labels from the store's remote; when that pull can't happen (no remote here, one
that's unusable, a failed pull) the command fails with `STALE`, never answering from this machine's labels alone.

Preflight.  What breaks or cancels a train is cheap to catch at mark time: a conflict with a PR already on `next`, a stale Lake
pin, a module string naming a module only an unlanded PR adds, a workspace member without its REAL_ROOTS entry, a test that reads
a row another queued PR refits away (an error at collection), one lane's unlanded branch carried in two PRs, and a failure of
`tests/test_repository.py` (a Markdown cap, a report-genre name), which the preflight runs too, as it runs the landing gates the
sync would (on the head; the sync, on its merge onto `next`).  A Lean change can pass its own Lean check and break a module
another PR on `next` owns (a changed signature, dropped premises, a dropped import), which drive's Lean check of the stack finds
only an hour later; the preflight builds those modules (`LakeBuild`).  So `ready` runs `Preflight` before the mark and before any quick tier: a refusal names what it found and the fix, a
failure `next` has too refuses nothing, and what keeps a check from running is said and refuses nothing (the after-landing run
checks it).  `--no-preflight` skips it, for the lander.

Records gate.  A head whose own records run printed `AUDIT: FAIL` fails a train's Lean check, about 50 minutes later.
So `ready` (and `judge`) first refuse a head whose change against `main` touches a `lean-audit.json`
unless the store holds a records run or a Lean check of that head that passed each such package on its lock, or of a commit whose
inputs for the package are the head's (its files and those of the packages it requires, tests and docs aside; `lean-toolchain`
and the checker whole), or vouches for it through a head that only merged `main` or committed the record a run regenerated (`RecordsGate`).  It reads the store's remote
and refuses when it can't; `--no-preflight` doesn't skip it.

`next`.  Its first-parent chain above `main` is the queue, and each merge commit's trailers name its PR, head and owner: git holds
the queue's state.  A sync keeps the longest prefix of that chain which still matches the admitted PRs (same PR, same head, same
order) and merges the rest on top, each by `git merge-tree` and `git commit-tree` (`Queue._merge`: no worktree, so a candidate
that conflicts costs one call).  A merge whose only conflicts are in pins (`tools/verity/check/pins.py`'s `PINS`, generated
sha256 listings of Lean inputs) is kept, each conflicted pin regenerated from the merged tree (`repinned`): two PRs that each
change a pinned input both change its pin.  So a new push moves its PR to the tip, a dropped PR takes the commits after it
along (they're merged again), and a `main` that moved outside the queue rebuilds `next` on it, unless it moved only by files a member whose Lean
check runs doesn't change (`hold`): then every sync, drive's or a hand one, leaves `next` as it is until that check ends.  Only drive
syncs (`Queue._drive_only`): `eject` only withdraws the mark, for drive's next tick, and a hand `sync` on the control pod
(`Queue.hand_sync`) waits as drive's tick does for a running Lean check whose prefix holds an ejected member (`lean_waits`) and
keeps `next` up to drive's running and passed Lean checks (`lean_hold`).  A
PR whose head contains another open PR's head, or by ancestry an older one (a head the queue saw it at, or its branch merged in:
`Queue.entries`; never by PR number, and never for commits the two merely share), goes
in after it and waits while that one waits, unless it carries that one's change: the same owner, or that one's owner withdrew its
mark (`Queue._carried`); drive tells the owner of an unmarked one that a marked PR waits for, once per pair of heads
(`Driver.tell_unmarked`).  What a sync adds on top goes non-Lean first
(drive.toml's `nonlean_first`, on by default: `Queue._nonlean_order`): each PR whose own change is no Lean change (`lean_change`
on its diff against `main`, or against the PR it's stacked on) and contains none of those merged with it goes below every Lean
change, `--first`'s order and then admission's within each part, so it doesn't wait above a Lean change for its Lean check.
Members on `next` never move for it, merged again or not: those above a moved or dropped member are merged again in their order,
and only a `--first` PR placed again, or a `next` rebuilt on a `main` that moved off it, sorts what is above it.  Nor do they
move but for drive's lift (`Queue.lift`, `Driver._lift`): while no Lean check of `next` runs
or passed, the members with a land verdict go below those waiting for one (drive.toml's `reviewed_first`), those that change no
Lean first, each after the PRs it contains and none past a held member, and land at once.  A PR that doesn't merge cleanly is left out of that sync, and its owner is told
which files conflict and which queued PRs touched them; while its head stays, later syncs try it, and the PRs that contain it,
above everything else, whatever `--first` or non-Lean first would say, so a sync with nothing new keeps `next`.  So is a PR a landing gate refuses (drive.toml's `landing_gates` at
`main`, `gate_refusal`) on its merge onto `next` (`tools/verity/check/gates/markdown_caps.py`: one it puts over a cap),
with the gate's fix: each PR the sync merges is judged, a new one or one merged again above a
dropped or moved member, never one in the prefix it keeps, so a refusal never takes a running check's commit off `next`; a merge
with the message, tree and parent tree of one judged under the same `main` takes that verdict (`Queue._gated`).  A PR
that contains one left out waits, recorded (the state's `waiting`) and told as a conflict is; `status` shows it waiting and
names the PR it contains and why that one is out, judging each PR after those it contains (else it shows
'admitted' for hours while one it contains conflicts with `main`).

Landing.  `drive` (`research.drive`) is the loop's tick that needs no agent: it lands `next`'s reviewed prefix, a Lean change
once its Lean check passed, by a fast-forward of `main` (no new merge commit), and checks `main` after each landing, reverting
what that check fails; it tells `@ci` what it refuses.  A push refused because `main` moved leaves `main` alone; the next sync
rebuilds `next` on the new `main`.  A landing that changes what a guarantee says sends the spec-change DM of what changed (`spec_alert`).

Notices.  A refusal (a conflict, an ejection) is posted in the merge thread to `@ci` and the owner's handle, once per PR head:
drive tells each conflict its sync finds, and `eject` posts as its ejector.  Nothing is written to the notes: a handoff note
there reached its owner only when the VM that ran the sync pushed its notes, inside that lane's commit.

`--shadow` builds `queue-shadow` instead of `next`, never pushes `main` and sends no notices, so the queue can run beside the trains
with no effect on them; its `sync`, `land` and `drive` still run only on drive (`--dry-run` shows the same off it).

Files.  The queue works only in its own clone (`--workdir`, default ~/.research/queue/verity), and fetches with forced refspecs: a
local copy of a branch that wasn't force-fetched once put a train on a stale base.  `state.json` beside the clone keeps ejections and
the notices sent; losing it costs a retry.  One queue command runs at a time per workdir (`queue.lock`).  The clone authenticates as
the caller's checkout does (`credential_args`): a cloud VM sets its credential helper in the checkout's own config, which a clone
elsewhere never reads.  Git never prompts, so a remote the queue can't read fails the command instead of answering from what the
clone fetched last.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import contextlib
import dataclasses
import difflib
import fcntl
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from . import quick_tier, waitcap
from .gates import git_env
from .store import ids, vocab

BRANCH, SHADOW_BRANCH = "next", "queue-shadow"
RULES = "ci/queue.toml"
#: where the rules are until the repository move puts `tools/verity/check/` at `ci/`
RULES_BEFORE_MOVE = "tools/verity/check/queue.toml"
#: drive's flags of record (`DriveFlags`, `--config`)
DRIVE_FLAGS = "tools/verity/check/drive.toml"
#: the pins tool, whose `PINS` names the generated files a merge rebuilds (`repinned`): after the repository move, and before it
PINS_FILES = ("ci/pins.py", "tools/verity/check/pins.py")
MARK_TIME_RE = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")
IDENTITY = {"GIT_AUTHOR_NAME": "research queue", "GIT_AUTHOR_EMAIL": "queue@research.invalid",
            "GIT_COMMITTER_NAME": "research queue", "GIT_COMMITTER_EMAIL": "queue@research.invalid"}
DATES = ("GIT_AUTHOR_DATE", "GIT_COMMITTER_DATE")
#: a transfer from the remote under 1 KB/s for 2 min aborts: git has no timeout, and a stalled HTTPS fetch would hold a drive tick
#: for minutes, the next tick retrying it
STALL = ("-c", "http.lowSpeedLimit=1000", "-c", "http.lowSpeedTime=120")
#: `gh api`'s limit: gh has no timeout, and a listing of the open PRs can hold a drive tick on a dead connection for
#: minutes while the same call answers in a second from a fresh one
GH_TIMEOUT_S = 120
#: the lane's handle ends on a letter or digit, so a sentence's period (`Lane proofs.`) isn't part of it
LANE_RE = re.compile(r"^[ \t>*_]*Lane[:\s]+`?([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)`?", re.M)
REGISTRY = Path(".agents/skills/using-slack/registry.json")
AGENT_RE = re.compile(r"\bbc-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
TRAILER_RE = re.compile(r"^(PR|Head|Owner): (.+)$", re.M)
SLUG_RE = re.compile(r"github\.com[:/]+([^/]+/[^/]+?)(?:\.git)?/?$")
AUTH_FAILED_RE = re.compile(r"could not read (Username|Password)|Authentication failed|unable to get password")
#: what decides how a check runs and what it passes: a `main` that changed any of it re-checks the queue rather than land on it
CHECK_TOOLING = ("tools/verity/check/", "tools/verity/lean/", "tools/verity/circuit_check/")


class QueueError(Exception):
    pass


def _git(repo: Path, *args: str, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    full = git_env(GIT_TERMINAL_PROMPT="0", **(env or {}))
    if env is IDENTITY:                     # the queue's merges carry the time they're made: a date a caller pins stops `log --since`
        for k in DATES:
            full.pop(k, None)
    r = subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, capture_output=True, text=True,
                       errors="surrogateescape", env=full)
    if check and r.returncode:
        raise QueueError(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()[-400:]}")
    return r


def _out(repo: Path, *args: str) -> str:
    return _git(repo, *args).stdout.strip()


def credential_args(repo: Path, remote: str) -> tuple[str, ...]:
    """`git -c` options that let the queue's own clone authenticate to `remote` as the checkout `repo` does: the `credential.*`
    settings its own config has for the remote (a cloud VM's helper lives there), else `gh auth git-credential` when no helper
    applies to the remote at all and `gh` is on PATH.  None when git's global config already serves it, or the remote isn't http."""
    if not re.match(r"https?://", remote):
        return ()
    found = _git(repo, "config", "--local", "--get-urlmatch", "credential", remote, check=False).stdout.splitlines()
    conf = dict(line.split(" ", 1) for line in found if " " in line)
    if not conf.get("credential.helper"):
        if _git(repo, "config", "--get-urlmatch", "credential.helper", remote, check=False).stdout.strip() or not (gh := shutil.which("gh")):
            return ()
        conf = {"credential.helper": f"!{shlex.quote(gh)} auth git-credential"}
    return ("-c", "credential.helper=", *(o for k, v in conf.items() for o in ("-c", f"{k}={v}")))


# ---- the open PRs -------------------------------------------------------------------------------------------------------------------

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

    @classmethod
    def from_api(cls, d: dict) -> "PR":
        return cls(number=int(d["number"]), head=d["head"]["sha"], title=d.get("title") or "", body=d.get("body") or "",
                   draft=bool(d.get("draft")), labels=tuple(sorted(x["name"] for x in d.get("labels") or [])),
                   base=(d.get("base") or {}).get("ref", "main"), branch=d["head"].get("ref") or "")

    @property
    def owner(self) -> str | None:
        """The lane a `Lane {name}` line names, else the Cursor agent id the body carries (its PR footer), else None."""
        if m := LANE_RE.search(self.body):
            return m.group(1)
        if m := AGENT_RE.search(self.body):
            return m.group(0)
        return None


def _json_values(text: str) -> Iterable:
    """The JSON values `gh api --paginate` prints one after another (one array per page)."""
    dec, i = json.JSONDecoder(), 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        v, i = dec.raw_decode(text, i)
        yield v


class GitHub:
    """The open PRs of a repository, through `gh api` (the cloud VMs' authenticated GitHub CLI)."""

    def __init__(self, repo: str):
        self.repo = repo

    def open_prs(self) -> list[PR]:
        try:
            r = subprocess.run(["gh", "api", "--paginate", f"repos/{self.repo}/pulls?state=open&per_page=100"], capture_output=True,
                               text=True, timeout=GH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise QueueError(f"gh api repos/{self.repo}/pulls: no answer in {GH_TIMEOUT_S} s") from None
        if r.returncode:
            raise QueueError(f"gh api repos/{self.repo}/pulls: {(r.stderr or r.stdout).strip()[-300:]}")
        return [PR.from_api(d) for page in _json_values(r.stdout) for d in (page if isinstance(page, list) else [page])]

    def branch_tip(self, branch: str) -> str | None:
        """``branch``'s commit as git has it now (the refs API, which a push moves at once; a PR's head lags it by seconds), or
        None when it can't be read (a fork's branch, no answer)."""
        try:
            r = subprocess.run(["gh", "api", f"repos/{self.repo}/git/ref/heads/{branch}", "--jq", ".object.sha"], capture_output=True,
                               text=True, timeout=GH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return None
        return r.stdout.strip() or None if r.returncode == 0 else None

    def close(self, number: int, comment: str) -> None:
        """Close PR ``number`` with ``comment`` (drive's stale rule, `research.stale`), as `gh` is authenticated here: drive's App."""
        try:
            r = subprocess.run(["gh", "pr", "close", str(number), "--repo", self.repo, "--comment", comment], capture_output=True,
                               text=True, timeout=GH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise QueueError(f"gh pr close {number}: no answer in {GH_TIMEOUT_S} s") from None
        if r.returncode:
            raise QueueError(f"gh pr close {number}: {(r.stderr or r.stdout).strip()[-300:]}")


# ---- rules and labels ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Rules:
    hold_label: str = "hold"
    source: str = RULES
    #: ready marks older than this don't count (`--marked-after` overrides it), so every VM's queue admits what drive admits
    marked_after: str | None = None

    @classmethod
    def parse(cls, text: str, source: str = RULES) -> "Rules":
        doc = tomllib.loads(text)
        unknown = set(doc) - {"hold_label", "marked_after"}
        if unknown:
            raise QueueError(f"{source}: unknown keys {sorted(unknown)}")
        cut = doc.get("marked_after")
        if cut is not None and not (isinstance(cut, str) and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", cut)):
            raise QueueError(f"{source}: marked_after {cut!r}: a UTC time as YYYY-MM-DDTHH:MM:SSZ, as labels record it")
        return cls(doc.get("hold_label", "hold"), source, cut)


#: drive's ways of landing (`DriveFlags.land`): `check`, which landed trains on their own check, was deleted with
#: its code
LANDINGS = ("review",)
#: a landing gate's path (`DriveFlags.landing_gates`)
GATE_PATH_RE = re.compile(r"[\w.-]+(?:/[\w.-]+)*\.py")


@dataclass(frozen=True)
class DriveFlags:
    """`research queue drive`'s flags: the defaults, then what `--config` reads (`DRIVE_FLAGS` on `main`, so a change to them is
    a reviewed commit and the next tick takes it), then each flag the command line gives (`override`)."""
    on: str | None = None
    marked_after: str | None = None
    regenerate_locks: bool = False
    first: tuple[int, ...] = ()
    #: how `next` lands: "review", each member on a reviewer's verdict, the full check running after landing (`Driver`)
    land: str = "review"
    #: the after-landing run reuses no pass but circuit-check's per-target ones (`check.py --no-cache --circuit-cache`)
    after_landing_no_cache: bool = False
    #: scripts a PR passes to be admitted (`gate_refusal`: its head at the ready mark's preflight, its merge onto `next` at the
    #: sync), each by its path from the repository's root, run as it is at `main`; drive's landing runs none
    landing_gates: tuple[str, ...] = ()
    #: the most Lean checks of the review landing running at once: a lower one and those stacked above it (`drive.lean_stack`)
    max_lean_checks: int = 2
    #: what a sync adds onto `next` goes non-Lean changes first (`Queue._nonlean_order`), so they don't wait above a
    #: Lean change for its Lean check
    nonlean_first: bool = True
    #: while no Lean check of `next` runs, members with a land verdict go below those waiting for one (`Driver._lift`), so a
    #: member waiting for its review holds only itself
    reviewed_first: bool = True
    #: days an open PR unmarked at its head stays idle before it is stale (`research.stale`: `status` says so, drive rings its
    #: owner once a day), and before drive closes it; 0 is off
    stale_days: int = 0
    stale_close_days: int = 0
    #: PRs the stale rule never judges: their owners keep them open with a reason
    stale_keep: tuple[int, ...] = ()

    @classmethod
    def parse(cls, text: str, source: str = "drive.toml", strict: bool = False) -> "DriveFlags":
        """``text``'s flags. ``strict`` (drive loading its own flags): any key this code doesn't read refuses, so a drive that
        lags `main` stops rather than run without a guard `main` added. Otherwise (a client reading `main`'s flags) such a
        key is ignored with a note, so a client older than `main` keeps working; `tests/test_drive_toml.py` holds `main`'s
        drive.toml to the strict parse, which is where a typo is caught."""
        try:
            doc = tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            raise QueueError(f"{source}: not TOML: {e}") from None
        keys = [f.name for f in dataclasses.fields(cls)]
        for k in list(doc):
            if k not in keys:
                near = difflib.get_close_matches(k, keys, n=1)
                if strict:
                    raise QueueError(f"{source}: unknown key {k!r}{f' (did you mean {near[0]!r}?)' if near else ''}: drive "
                                     f"reads only {', '.join(keys)}; rename or remove it, or update this research")
                print(f"{source}: ignoring key {k!r}, which this research doesn't read (a newer one may)", file=sys.stderr)
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
        if "max_lean_checks" in doc:
            if type(doc["max_lean_checks"]) is not int or doc["max_lean_checks"] < 1:
                raise bad("max_lean_checks", "a whole number, 1 or more")
            got["max_lean_checks"] = doc["max_lean_checks"]
        if "marked_after" in doc:
            if not (isinstance(doc["marked_after"], str) and MARK_TIME_RE.fullmatch(doc["marked_after"])):
                raise bad("marked_after", "a UTC time as YYYY-MM-DDTHH:MM:SSZ, as labels record it")
            got["marked_after"] = doc["marked_after"]
        for k in ("regenerate_locks", "nonlean_first", "reviewed_first"):
            if k in doc:
                if not isinstance(doc[k], bool):
                    raise bad(k, "true or false")
                got[k] = doc[k]
        if "first" in doc:
            if not (isinstance(doc["first"], list) and all(type(n) is int and n > 0 for n in doc["first"])):
                raise bad("first", "a list of PR numbers, as [1858, 1860]")
            got["first"] = tuple(doc["first"])
        if "land" in doc:
            if doc["land"] == "check":
                raise bad("land", "train landing was deleted with its code (#5070): drive lands on review only; set "
                                  "`land = \"review\"` or remove the key")
            if doc["land"] not in LANDINGS:
                raise bad("land", f"one of {', '.join(LANDINGS)}")
            got["land"] = doc["land"]
        if "after_landing_no_cache" in doc:
            if not isinstance(doc["after_landing_no_cache"], bool):
                raise bad("after_landing_no_cache", "true or false")
            got["after_landing_no_cache"] = doc["after_landing_no_cache"]
        if "landing_gates" in doc:
            v = doc["landing_gates"]
            if not (isinstance(v, list) and all(isinstance(x, str) and GATE_PATH_RE.fullmatch(x) and ".." not in x.split("/")
                                                for x in v)):
                raise bad("landing_gates", "a list of Python scripts by their path from the repository's root, as "
                                           "[\"integrations/x/tools/gate.py\"]")
            got["landing_gates"] = tuple(v)
        for k in ("stale_days", "stale_close_days"):
            if k in doc:
                if type(doc[k]) is not int or doc[k] < 0:
                    raise bad(k, "a whole number of days, 0 for off")
                got[k] = doc[k]
        if "stale_keep" in doc:
            if not (isinstance(doc["stale_keep"], list) and all(type(n) is int and n > 0 for n in doc["stale_keep"])):
                raise bad("stale_keep", "a list of PR numbers, as [2692, 2693]")
            got["stale_keep"] = tuple(doc["stale_keep"])
        if got.get("stale_close_days") and not 0 < got.get("stale_days", 0) <= got["stale_close_days"]:
            raise bad("stale_close_days", "0, or at least stale_days, which must be on: drive closes only a stale PR")
        return cls(**got)

    def override(self, a: argparse.Namespace) -> "DriveFlags":
        """These flags with each one ``a`` (the command line) gives explicitly in its place."""
        given = {k: getattr(a, k) for k in ("on", "marked_after", "regenerate_locks", "after_landing_no_cache")
                 if getattr(a, k, None) is not None}
        if a.first is not None:
            given["first"] = tuple(int(n) for n in a.first.split(",") if n)
        return dataclasses.replace(self, **given)

    def line(self) -> str:
        return " ".join(f"{k}={','.join(map(str, v)) if isinstance(v, tuple) else v}"
                        for k, v in dataclasses.asdict(self).items())


def drive_flags(path: str, clone: Path | None) -> tuple[DriveFlags, str]:
    """The flags in ``path`` at `main` in the queue's ``clone`` (fetched), or in the file itself when ``clone`` is None
    (`--config-from-tree`); with where they came from."""
    if clone is None:
        try:
            return DriveFlags.parse(Path(path).read_text(), path, strict=True), path
        except OSError as e:
            raise QueueError(f"--config {path}: {e.strerror}") from None
    if Path(path).is_absolute() or path.startswith(".."):
        raise QueueError(f"--config {path}: a path from the repository's root, read at `main` (or --config-from-tree for this file)")
    sha = _out(clone, "rev-parse", "refs/remotes/origin/main")
    r = _git(clone, "show", f"{sha}:{path}", check=False)
    if r.returncode:
        raise QueueError(f"--config {path}: `main` ({sha[:12]}) has no such file: land it first, or pass --config-from-tree")
    where = f"{path} at main {sha[:12]}"
    return DriveFlags.parse(r.stdout, where, strict=True), where


#: how long a landing gate may run before it counts as unable to judge (`run_gate`)
GATE_TIMEOUT_S = 600
#: the fix items of a gate's failure a refusal quotes
GATE_FIX_LINES = 15
#: the most landing-gate verdicts the queue's state keeps for one `main` (`Queue._gated`), the newest
GATE_VERDICTS = 1000


def flags_at(repo: Path, ref: str) -> DriveFlags:
    """drive.toml at ``ref``; the defaults when ``ref`` has none (QueueError when it can't be parsed)."""
    r = _git(repo, "show", f"{ref}:{DRIVE_FLAGS}", check=False)
    return DriveFlags.parse(r.stdout, f"{DRIVE_FLAGS} at {ref}") if r.returncode == 0 else DriveFlags()


def gates_at(repo: Path, ref: str) -> tuple[str, ...]:
    """drive.toml's `landing_gates` at ``ref``; none when ``ref`` has no drive.toml (QueueError when it isn't TOML or the gates
    are malformed).  Only that key is read: a branch's older code reads `main`'s drive.toml here (the ready mark's preflight),
    and a key added since must not stop it running the gates."""
    r = _git(repo, "show", f"{ref}:{DRIVE_FLAGS}", check=False)
    if r.returncode:
        return ()
    source = f"{DRIVE_FLAGS} at {ref}"
    try:
        gates = tomllib.loads(r.stdout).get("landing_gates")
    except tomllib.TOMLDecodeError as e:
        raise QueueError(f"{source}: not TOML: {e}") from None
    if gates is None:
        return ()
    return DriveFlags.parse(f"landing_gates = {json.dumps(gates)}\n", source).landing_gates


def run_gate(repo: Path, main: str, head: str, path: str) -> tuple[int | None, str]:
    """The landing gate ``path`` as it is at ``main`` (so the change it judges can't weaken it), run as `python3 PATH MAIN HEAD
    --json` in ``repo``: (its exit, its output), or (None, why it didn't run)."""
    r = _git(repo, "show", f"{main}:{path}", check=False)
    if r.returncode:
        return None, f"{path} isn't at `main` {main[:12]}"
    with tempfile.TemporaryDirectory(prefix="queue-gate-") as d:
        script = Path(d) / Path(path).name
        script.write_text(r.stdout, errors="surrogateescape")
        try:
            p = subprocess.run([sys.executable, str(script), main, head, "--json"], cwd=repo, capture_output=True, text=True,
                               stdin=subprocess.DEVNULL, timeout=GATE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return None, f"{path} ran over {GATE_TIMEOUT_S} s"
    return p.returncode, p.stdout if p.returncode in (0, 1) else (p.stderr or p.stdout).strip()[-400:]


def gate_refusal(repo: Path, main: str, head: str, gates: Iterable[str],
                 run: Callable[[Path, str, str, str], tuple[int | None, str]] = run_gate,
                 onto: str | None = None) -> tuple[list[str], list[str]]:
    """What the landing ``gates`` say of ``head`` against ``main`` (each judges the change from their merge base): (each failure,
    with its fix and how to run the gate; each gate that couldn't judge, with why).  ``head`` is a PR's head, or with ``onto``
    its merge onto that branch, as the sync judges it: what the members below add counts too (a merge can take
    a file over its cap with the members below it, though its head alone is under).  A gate exits 0 to pass, and 1 with
    `--json`'s `{"failures": [{"why": ..., "fix": [...]}]}` to fail; anything else is unable to judge."""
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
            unjudged.append(f"the landing gate {path} can't judge {what} on `main` {main[:12]} ({why.strip()[-400:]})")
            continue
        for why, fix in failures:
            refused.append(f"the landing gate {path} at `main` {main[:12]}{merged}: {why}" + "".join(f"; {x}" for x in fix)
                           + f" (`python3 {path} origin/main HEAD` {rerun} runs it)")
    return refused, unjudged


def label_target(pr: int, head: str) -> str:
    return f"pr:{pr}@{head}"


def lean_running(state: dict, pr: int) -> str | None:
    """The run ids of a running review-landing Lean check (`drive["lean"]`) whose checked prefix holds ``pr``, or None: an
    ejection of ``pr`` waits to rebuild `next` until it ends, as a sync would take its commit off `next` and cancel it
    mid-audit.  A stacked one (`Driver._stack_lean`) doesn't hold it."""
    for rec in ((state.get("drive") or {}).get("lean") or {}).values():
        if rec.get("state") is None and not rec.get("stacked") and any(n == pr for n, _ in rec.get("members") or ()):
            return " ".join(rec.get("ids") or ()) or "(launching)"
    return None


def lean_hold(state: dict, chain: list[Member]) -> int:
    """How many members at the bottom of ``chain`` drive's Lean checks keep in place: up to the highest commit of one in drive's
    state (`drive["lean"]`) that runs, launches or passed, as drive's review tick holds them (`Driver._running_lean`,
    `_landable`).  A passed one holds whether or not its prefix lands yet; drive's next tick decides that."""
    lean = (state.get("drive") or {}).get("lean") or {}
    held = {c for c, rec in lean.items() if rec.get("state") in (None, "passed")}
    return max((i + 1 for i, m in enumerate(chain) if m.commit in held), default=0)


def lean_waits(state: dict, ready: Callable[[int, str], bool]) -> str | None:
    """The commit of a running Lean check of drive's, not a stacked one, whose checked prefix holds a member whose mark is
    withdrawn (``ready``), or None: a rebuild from that member would take the check's commit off `next`, so drive's tick waits for
    the check to end (`Driver._review`), and so does a hand sync."""
    for c, rec in ((state.get("drive") or {}).get("lean") or {}).items():
        if rec.get("state") is None and not rec.get("stacked") and any(not ready(n, h) for n, h in rec.get("members") or ()):
            return c
    return None


def _who(name: str) -> str:
    return name.strip().lstrip("@").lower()


def unreviewed(labels: Iterable, owners: Iterable[str | None]) -> str | None:
    """Why the `reviewed` ``labels`` on one head don't land it, or None when they do: each reviewer's newest verdict counts, a
    hold by anyone keeps it, and a land counts only by a reviewer who is none of ``owners`` (the PR's owning lane or agent id,
    that owner's handle, its ready mark's author: `review_owners`), compared without case or a leading `@`."""
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


#: a Lake package's lock, and the files of a package its Lean check reads
LEAN_LOCK = "lean-audit.json"
LEAN_SOURCES = (".lean", "lakefile.lean", "lakefile.toml", "lean-toolchain", "lake-manifest.json")


#: the analysis tree: a Lean change only here that touches no lock is proofs, which lands on its land verdict and is checked after
#: landing, a failure reverting it; Lean anywhere else runs as code (C-Flock's verifier and the ZK
#: server, the PoUS grader, the certifier program, the NCI check, the remote audit's verifier) and keeps its Lean check first
LEAN_PROOFS_TREE = "Security/"


def lean_change(repo: Path, commit: str, base: str | None = None) -> bool:
    """Whether ``commit``'s own change (against its first parent, or ``base``) waits for a Lean check before a review landing: it
    touches a lock, so neither a guarantee's statement nor a definition it reads reaches `main` without its record (the
    after-landing lean-audit, `--no-regenerate`, fails on a record difference), or a Lean source outside `LEAN_PROOFS_TREE` of a
    Lake package that has a lock, Lean that runs as code.  A change to `Security/`'s Lean alone is proofs: it lands on its
    verdict, and the after-landing lean-audit reverts it on a failure."""
    files = changed(repo, base or f"{commit}^1", commit)
    if any(Path(f).name == LEAN_LOCK for f in files):
        return True
    if not (lean := [f for f in files if f.endswith(LEAN_SOURCES) and not f.startswith(LEAN_PROOFS_TREE)]):
        return False
    pkgs = {str(Path(f).parent) for f in _out(repo, "ls-tree", "-r", "--name-only", commit).splitlines() if Path(f).name == LEAN_LOCK}
    return any(p == "." or f.startswith(p + "/") for f in lean for p in pkgs)


class StoreLabels:
    """Ready marks from the evidence store: `ready` labels on `pr:{n}@{head}`, pulled from the remote first.  Without that pull it
    raises: a cloud VM's shell that lost its store secrets has no remote, and its local labels are whatever it last pulled."""

    def __init__(self, store, log: Callable[[str], None] = print, since: str | None = None):
        self.store, self.log, self.since = store, log, since
        self._pulled: set[int] = set()
        self._newest: dict[tuple[int, str], object] = {}
        self._fresh: set[str] = set()

    def prefetch(self, heads: Iterable[tuple[int, str]]) -> None:
        """Pull the labels of each `pr:{n}@{head}` from the store's remote at once, so `ready` answers each from them without a
        pull of its own (one listing per open PR, in turn, would be seconds of every drive tick).  A failed pull leaves each
        target to its own, which says why."""
        from .store import labels_sync

        targets = [label_target(n, h) for n, h in heads]
        if not targets or getattr(self.store, "remote", None) is None:
            return
        try:
            labels_sync.pull_targets(self.store, targets, jobs=labels_sync.SYNC_JOBS)
        except Exception:  # noqa: BLE001
            return
        self._fresh.update(targets)

    def ready(self, pr: int, head: str) -> bool:
        """Whether the newest `ready` label on `pr:{n}@{head}` is `true`, written at or after ``since`` (`--marked-after`) when
        given.  `research queue eject` writes `ready = false` there, so an ejection holds on every VM's queue until the owner
        marks it again or pushes a new head (an ejection kept in the ejecting VM's state would be lost to drive, which re-admits)."""
        labs = sorted(self._labels(pr, head, vocab.READY), key=lambda lab: lab.ts)
        self._newest[(pr, head)] = labs[-1] if labs else None
        return bool(labs) and str(labs[-1].value).lower() == "true" and (self.since is None or labs[-1].ts >= self.since)

    def reviews(self, pr: int, head: str) -> list:
        """The `reviewed` labels on `pr:{n}@{head}` (`research queue review`), pulled as `ready`'s are."""
        return self._labels(pr, head, vocab.REVIEWED)

    def ready_by(self, pr: int, head: str) -> str | None:
        """Who wrote the newest `ready = true` label on `pr:{n}@{head}`, whenever it was written."""
        labs = [lab for lab in sorted(self._labels(pr, head, vocab.READY), key=lambda lab: lab.ts) if str(lab.value).lower() == "true"]
        return labs[-1].by if labs else None

    def marked_by(self, pr: int, head: str) -> str | None:
        """Who wrote the newest `ready` label `ready` read on `pr:{n}@{head}`."""
        lab = self._newest.get((pr, head))
        return None if lab is None else lab.by

    def marked_at(self, pr: int, head: str) -> str | None:
        """When the newest `ready` label `ready` read on `pr:{n}@{head}` was written."""
        lab = self._newest.get((pr, head))
        return None if lab is None else lab.ts

    def before_cutoff(self, pr: int, head: str) -> str | None:
        """When the newest `ready` label `ready` read on `pr:{n}@{head}` is `true` but older than ``since``, its time: a mark
        drive ignores, which `status` names so the owner re-marks or closes the PR."""
        lab = self._newest.get((pr, head))
        if lab is None or self.since is None or str(lab.value).lower() != "true" or lab.ts >= self.since:
            return None
        return lab.ts

    def _other_heads(self, pr: int, head: str, key: str) -> dict[str, list]:
        """{head: its ``key`` labels} for each head of pull request ``pr`` but ``head`` that has some, after pulling every head's
        labels from the store's remote (once per PR); without that pull it raises, as `_labels` does."""
        from .store import labels_sync

        prefix = label_target(pr, "")
        if pr not in self._pulled:
            try:
                labels_sync.pull_prefix(self.store, prefix)
            except Exception as e:  # noqa: BLE001  (no pull: local labels alone may be stale)
                raise QueueError(f"STALE: the labels of {prefix}* weren't pulled from the store's remote ({type(e).__name__}: {e}), "
                                 f"so ready marks on #{pr}'s other heads here may be old") from None
            self._pulled.add(pr)
        out = {}
        for d in sorted((Path(self.store.root) / "labels").glob(f"{ids.target_dirname(prefix)}*")):
            old = d.name.rpartition("@")[2]
            if old != head and ids.is_pr(label_target(pr, old)) and (labs := self.store.labels(label_target(pr, old), key=key)):
                out[old] = labs
        return out

    def _labels(self, pr: int, head: str, key: str) -> list:
        from .store import labels_sync
        from .store.local import StoreError

        target = label_target(pr, head)
        if target in self._fresh:
            return self.store.labels(target, key=key)
        try:
            labels_sync.read_through(self.store, [target], lambda s: self.log(f"research queue: {s}"), strict=True)
        except StoreError as e:
            raise QueueError(f"STALE: the labels of {target} weren't pulled from the store's remote ({e}), so ready marks here may "
                             f"be old; not answering from them") from None
        return self.store.labels(target, key=key)


# ---- the queue ----------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Member:
    """One PR on `next`: its merge commit, and the head it merged."""
    pr: int
    head: str
    owner: str
    commit: str


@dataclass
class Entry:
    """One open PR's standing in a sync."""
    pr: PR
    state: str = "ready"                  # ready | queued | waiting | refused
    why: str = ""
    after: tuple[int, ...] = ()           # open PRs whose current or an older head its head contains (`Queue.entries`)
    older: tuple[int, ...] = ()           # those of `after` whose current head it doesn't contain
    position: int | None = None           # 1-based, on `next`
    commit: str | None = None
    idle_since: float | None = None       # stale (`research.stale`): idle since then

    def contains(self, n: int) -> str:
        """How a reason names ``n``, one of `after`."""
        return f"#{n}'s commits (at an older head)" if n in self.older else f"#{n}"


def contained(repo: Path, base: str, head: str, heads: dict[str, int]) -> list[int]:
    """The PRs of ``heads`` ({head: PR number}) whose head ``head`` contains: one of its commits not on ``base``, ``head``
    itself included (the caller leaves its own PR out).  A PR goes on `next` after those it contains, and after those whose
    commits it holds at an older head (`Entry.after`, `Queue.entries`)."""
    return sorted({heads[h] for h in _out(repo, "rev-list", f"{base}..{head}").split() if h in heads})


def based_on(pr: PR, prs: Iterable[PR]) -> str:
    """``pr``'s base branch, with the open PR whose head that branch is (`BRANCH (#P's head)`) when one is."""
    parent = next((p for p in prs if p.branch == pr.base and p.number != pr.number), None)
    return f"{pr.base} (#{parent.number}'s head)" if parent else pr.base


def _message(pr: PR) -> str:
    return f"Queue: #{pr.number} {pr.title}".rstrip() + f"\n\nPR: #{pr.number}\nHead: {pr.head}\nOwner: {pr.owner}\n"


class Queue:
    def __init__(self, workdir: Path, *, remote: str, github, ready: Callable[[int, str], bool],
                 store=None, shadow: bool = False, rules: Rules | None = None, log: Callable[[str], None] = print,
                 auth: tuple[str, ...] = (), first: tuple[int, ...] = (),
                 before_cutoff: Callable[[int, str], str | None] | None = None,
                 marked_by: Callable[[int, str], str | None] | None = None,
                 marked_at: Callable[[int, str], str | None] | None = None,
                 prefetch: Callable[[list[tuple[int, str]]], None] | None = None, nonlean_first: bool | None = None,
                 reviewed_first: bool | None = None):
        self.workdir, self.remote, self.github, self.ready, self.store = Path(workdir), remote, github, ready, store
        self.shadow, self.log, self._rules, self.auth, self.first = shadow, log, rules, auth, tuple(first)
        self.before_cutoff, self.marked_by, self.marked_at, self.prefetch = before_cutoff, marked_by, marked_at, prefetch
        #: how many members at the bottom of `next` a running check covers, or a passed Lean check drive is to land
        #: (`Driver._landable`): `first` goes above them, so no reorder takes that check's commit off `next`
        self.hold = 0
        #: whether a sync merges non-Lean changes first (`_nonlean_order`); None: drive.toml's `nonlean_first` at `main`
        self.nonlean_first = nonlean_first
        #: whether drive's lift puts members with a land verdict below those without (`Driver._lift`); None: drive.toml's
        #: `reviewed_first` at `main`
        self.reviewed_first = reviewed_first
        #: each (head, base)'s `lean_change`
        self._lean_of: dict[tuple[str, str], bool] = {}
        #: members a running check (or a `hold`ing passed one) covers that keep their place at the head it checks though their PR's head has moved
        #: (`Driver`: a new head doesn't cancel a running check, and its PR rejoins above once that check ends)
        self.frozen: set[int] = set()
        #: members `next` keeps that this sync moves below the rest, in this order, at the heads it holds (`Driver._lift`: those
        #: with a land verdict, non-Lean first, while no Lean check of `next` runs or passed); what `order` moved of them
        self.lift: list[int] = []
        self._lifted: list[int] = []
        self.now: Callable[[], float] = time.time
        self.branch = SHADOW_BRANCH if shadow else BRANCH
        self.state_path = self.workdir.parent / f"{self.workdir.name}.state.json"
        #: whether git takes `--attr-source` (`_merge`), until it refuses it once
        self._attr_source = True
        #: the files each merge commit of `next` changed (`touched_by`)
        self._touched: dict[str, set[str]] = {}
        #: main's registry as `handle` reads it (its handles, {agent id: handle}), until the next `fetch`
        self._registry: tuple[set, dict[str, str]] | None = None

    # -- the clone and its state

    def _remote(self, cwd: Path, *args: str, ok: tuple[int, ...] | None = (0,)) -> subprocess.CompletedProcess:
        """A git command that talks to the remote, with `auth` (`credential_args`); one that exits outside `ok` raises."""
        r = _git(cwd, *STALL, *self.auth, *args, check=False)
        if ok is not None and r.returncode not in ok:
            err = (r.stderr or r.stdout).strip()[-400:]
            hint = ("; no credential helper applies to the remote outside the caller's checkout: set one in git's global config, "
                    "or put an authenticated `gh` on PATH") if not self.auth and AUTH_FAILED_RE.search(err) else ""
            raise QueueError(f"git {' '.join(args)}: {err}{hint}")
        return r

    def _clone(self) -> None:
        if not (self.workdir / ".git").exists():
            self.workdir.parent.mkdir(parents=True, exist_ok=True)
            self._remote(self.workdir.parent, "clone", "-q", self.remote, str(self.workdir))
        # the caller's origin as it is now: a token embedded in the URL the clone was made from rotates
        _git(self.workdir, "remote", "set-url", "origin", self.remote)
        # `.gitattributes` names this driver for every lean-audit.json; without it in the clone's config git merges the records
        # as text, and any two PRs that each add a record conflict
        from .clone import RECORD_DRIVER

        for k, v in RECORD_DRIVER.items():
            _git(self.workdir, "config", k, v)
        _git(self.workdir, "merge", "--abort", check=False)
        _git(self.workdir, "reset", "-q", "--hard")
        _git(self.workdir, "clean", "-qfdx")

    def fetch(self, prs: Iterable[PR] = ()) -> dict[int, str]:
        """Forced refspecs only: `main`, the queue's branch when it exists, and each open PR's head as `refs/queue/pr/{n}`;
        return each fetched pull ref's commit, which can briefly differ from GitHub's API head, and none for a PR whose pull ref
        GitHub doesn't have (it waits, `fetch_prs`)."""
        prs = list(prs)
        self._registry = None
        specs = ["+refs/heads/main:refs/remotes/origin/main", *(f"+refs/pull/{p.number}/head:refs/queue/pr/{p.number}" for p in prs)]
        if self._remote(self.workdir, "fetch", "-q", "--prune", "origin", *specs, ok=None).returncode:
            # one pull ref GitHub hasn't made yet (a PR with only its `merge` ref) fails the whole
            # fetch, and every tick with it: fetch `main` alone, raising as before, then each PR's ref, leaving out what's missing
            self._remote(self.workdir, "fetch", "-q", "--prune", "origin", specs[0])
            for p, spec in zip(prs, specs[1:], strict=True):
                if (r := self._remote(self.workdir, "fetch", "-q", "origin", spec, ok=None)).returncode:
                    _git(self.workdir, "update-ref", "-d", f"refs/queue/pr/{p.number}", check=False)
                    self.log(f"research queue: #{p.number}'s pull ref can't be fetched, so it waits this sync: "
                             f"{(r.stderr or r.stdout).strip()[-200:]}")
        # the merge drivers (`clone.RECORD_DRIVER`) run from this checkout, so one that `main` has moved past judges merges with old
        # drivers (an old clone refuses a lean-audit.json conflict main's merge.py resolves).  Drive's checkout, `next`'s tip, contains `main` and stays
        if _git(self.workdir, "merge-base", "--is-ancestor", "refs/remotes/origin/main", "HEAD", check=False).returncode:
            _git(self.workdir, "checkout", "-q", "--detach", "--force", "refs/remotes/origin/main")
        # exit 2: no such branch; any other failure is the remote's, never an empty queue
        if self._remote(self.workdir, "ls-remote", "--exit-code", "origin", f"refs/heads/{self.branch}", ok=(0, 2)).returncode == 0:
            self._remote(self.workdir, "fetch", "-q", "origin", f"+refs/heads/{self.branch}:refs/remotes/origin/{self.branch}")
        else:
            _git(self.workdir, "update-ref", "-d", f"refs/remotes/origin/{self.branch}", check=False)
        heads = {p.number: _git(self.workdir, "rev-parse", "-q", "--verify", f"refs/queue/pr/{p.number}", check=False) for p in prs}
        return {n: r.stdout.strip() for n, r in heads.items() if not r.returncode}

    def fetch_prs(self, prs: list[PR]) -> tuple[list[PR], dict[int, Entry]]:
        """Fetch ``prs`` and separate any whose API head disagrees with its pull ref, so one GitHub propagation race cannot stop
        the queue.  Such a PR waits outside ``next`` until a later snapshot agrees rather than admitting an unverified old head,
        unless its branch's tip is the API head."""
        fetched = self.fetch(prs)
        consistent, waiting = [], {}
        for p in prs:
            if fetched.get(p.number) != p.head and p.branch and self._head_from_branch(p):
                fetched[p.number] = p.head
            if p.number not in fetched:
                waiting[p.number] = Entry(p, state="waiting", why="GitHub has no pull ref for its head yet; retrying next sync")
                continue
            if fetched[p.number] == p.head:
                consistent.append(p)
                continue
            why = f"GitHub's API head {p.head[:12]} and pull ref {fetched[p.number][:12]} disagree; retrying next sync"
            waiting[p.number] = Entry(p, state="waiting", why=why)
            self.log(f"research queue: #{p.number}'s API head {p.head[:12]} and fetched pull ref "
                     f"{fetched[p.number][:12]} disagree; retrying next sync")
        return consistent, waiting

    def _head_from_branch(self, p: PR) -> bool:
        """Take ``p``'s head from its branch when the branch's tip is the API head: GitHub can leave refs/pull/N/head behind its
        branch and API head for hours."""
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

    def chain(self) -> list[Member]:
        """The members on the queue's branch above `main`, oldest first; none when the branch is missing or `main` isn't below it."""
        return chain_of(self.workdir, "origin/main", f"refs/remotes/origin/{self.branch}")

    # -- admission

    def _marked_again(self, p: PR, ejected: dict) -> bool:
        """Whether ``p``'s owner marked it ready after its ejection at this head: the store's newest mark decides on every VM, so
        this queue's own record of the ejection can't hold a PR that mark admits: otherwise a PR re-marked at its ejected head
        would stay off `next` on the ejecting VM's queue only."""
        if self.marked_at is None or not self.ready(p.number, p.head):
            return False
        when, at = self.marked_at(p.number, p.head), ejected.get("at", "")
        if len(at) == len("2026-10-12T21:09Z"):
            at = at[:-1] + ":59Z"                       # recorded to the minute: a mark in that minute may predate the ejection
        return when is not None and when > at

    def entries(self, prs: list[PR], state: dict, base: str = "origin/main") -> dict[int, Entry]:
        """Each open PR's standing.  A PR P goes after, and waits while it isn't admitted, each other open PR Q whose current
        head P's head contains, or an older head of Q's, by ancestry alone (never by PR number).  P holds an older head of Q
        when one of P's own commits (those in ``base``.. its head less those below the heads of open PRs it contains) is

        - a head the queue saw Q at, and the first open PR it saw there (`seen`), that Q's head still contains, as when P
          branched from Q and Q moved on unmarked (else P would land Q's unmarked commits under P's mark); or
        - one of Q's own commits on Q's first-parent line that P has only through a merge, as when P merged Q's branch and Q
          moved on (P waits for Q, and Q for nothing).

        Commits two PRs merely share wait for nothing: a third branch both merged (on neither first-parent line), or a fork
        point the queue never saw either PR at (on both).  One `rev-list` and one `rev-list --first-parent` a PR, not a merge
        base for each pair."""
        rules = self.rules()
        heads = {p.head: p.number for p in prs}
        out: dict[int, Entry] = {}
        below: dict[int, set[str]] = {}
        line: dict[int, set[str]] = {}
        if self.prefetch is not None:
            self.prefetch([(p.number, p.head) for p in prs if rules.hold_label not in p.labels and p.owner is not None])
        for p in prs:
            e = out[p.number] = Entry(p)
            ejected = state["ejected"].get(str(p.number))
            if rules.hold_label in p.labels:
                e.state, e.why = "waiting", f"held (`{rules.hold_label}` label)"
            elif p.owner is None:
                e.state, e.why = "waiting", "no owner: add a `Lane {name}` line to the PR body"
            elif ejected and ejected["head"] == p.head and not self._marked_again(p, ejected):
                e.state, e.why = "refused", f"ejected at {p.head[:8]}: {ejected['why']} (a new push or ready mark re-admits it)"
            elif self.ready(p.number, p.head):
                pass
            elif self.before_cutoff and (when := self.before_cutoff(p.number, p.head)):
                e.state, e.why = "waiting", (f"ignored: marked before cutoff (its ready mark at {p.head[:8]} is from {when}, "
                                             f"before `marked_after`): re-mark it or close it")
            else:
                e.state, e.why = "waiting", f"not marked ready at {p.head[:8]}: `research queue ready {p.number} --on MACHINE` (or --local; --mark without the tier)"
            below[p.number] = set(_out(self.workdir, "rev-list", f"{base}..{p.head}").split())
            line[p.number] = set(_out(self.workdir, "rev-list", "--first-parent", f"{base}..{p.head}").split())
            e.after = tuple(sorted(heads[h] for h in below[p.number] if h in heads and heads[h] != p.number))
        own = {n: below[n].difference(*(below[a] for a in e.after)) for n, e in out.items()}
        branch: dict[str, list[int]] = {}                   # commit: the PRs it is an own commit on the first-parent line of
        for n, cs in own.items():
            for c in cs & line[n]:
                branch.setdefault(c, []).append(n)
        seen = state.get("seen") or {}
        for n, e in out.items():
            held = {q for c in own[n] for q in seen.get(c, ()) if q in out and c in below[q]}
            held |= {q for c in own[n] - line[n] for q in branch.get(c, ())}
            e.older = tuple(sorted(held - {n}))
            e.after = tuple(sorted(e.after + e.older))
        for e in out.values():
            e.after = tuple(a for a in e.after if not self._carried(e.pr, out[a], state))
            e.older = tuple(a for a in e.older if a in e.after)
        changed = True
        while changed:                                      # a PR waits while a PR whose commits it contains isn't admitted
            changed = False
            for e in out.values():
                if e.state == "ready" and (blocking := [a for a in e.after if out[a].state != "ready"]):
                    e.state, e.why, changed = "waiting", f"contains {e.contains(blocking[0])}, which isn't admitted", True
        return out

    @staticmethod
    def _seen(state: dict, prs: list[PR], open_numbers: set[int]) -> None:
        """Record in ``state``'s `seen` ({head: PRs}) each head of ``prs`` no open PR was seen at before, for the PRs at it now,
        so a PR holding it later holds an older head of theirs (`entries`); a closed PR's records go."""
        seen = state["seen"] = {h: kept for h, ns in (state.get("seen") or {}).items()
                                if (kept := [n for n in ns if n in open_numbers])}
        new: dict[str, list[int]] = {}
        for p in prs:
            if p.head not in seen:
                new.setdefault(p.head, []).append(p.number)
        seen.update(new)

    def _carried(self, p: PR, a: Entry, state: dict) -> bool:
        """Whether ``p`` carries the change of ``a``, a PR whose head or own commits it contains (`entries`) and that isn't
        marked ready: the same owner
        rebuilt it as ``p``, or ``a``'s owner withdrew its mark (`eject`, or a `ready = false` label).  Then ``p`` doesn't wait
        for ``a`` and goes on without it below, so a lane that rebuilds a PR needn't drop the superseded one by hand, which no
        one's tooling may close."""
        if a.state == "ready" or a.pr.owner is None or self.rules().hold_label in a.pr.labels:
            return False
        who = _who(a.pr.owner)
        ejected = state["ejected"].get(str(a.pr.number)) or {}
        return (_who(p.owner or "") == who
                or (ejected.get("head") == a.pr.head and _who(ejected.get("by") or "") == who)
                or (self.marked_by is not None and not self.ready(a.pr.number, a.pr.head)
                    and not (self.before_cutoff and self.before_cutoff(a.pr.number, a.pr.head))    # then it's a `true`
                    and _who(self.marked_by(a.pr.number, a.pr.head) or "") == who))

    def order(self, chain: list[Member], entries: dict[int, Entry], lift: Iterable[int] = ()) -> list[int]:
        """Members that keep their head, or move it to a descendant (`_same_or_fixed`), keep their order; the rest follow by PR
        number, each after the PRs it contains.  The PRs
        in `first` (`--first`) that are ready go below all of them, in the order given, so they land first; but
        above the bottom `hold` members, which a running check covers.  With no `hold`, the members of ``lift`` at the heads
        `next` holds go above `first` and below the rest, each only after the PRs it contains (`_lifted`)."""
        ready = {n for n, e in entries.items() if e.state == "ready"}
        kept = [m.pr for m in chain
                if m.pr in ready and self._same_or_fixed(m, entries) or self._frozen(m, entries) or self._pending(m, entries)]
        held: list[int] = []
        for m in chain[:self.hold]:
            if m.pr not in kept:
                break
            held.append(m.pr)
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
        at = {m.pr: m.head for m in chain}
        self._lifted = []
        for n in () if held else lift:
            if n in kept and entries[n].pr.head == at.get(n) and all(a in self._lifted for a in entries[n].after):
                self._lifted.append(n)
        # a newly ready PR that makes circuit-check run cold goes above the warm ones; members already on `next` keep their place,
        # and so the check that covers them
        for n in held + [f for f in self.first if f in ready] + self._lifted + kept + [n for n in new if n not in cold] + sorted(cold):
            place(n)
        return out

    def flag(self, name: str) -> bool:
        """drive.toml's ``name`` (`nonlean_first`, `reviewed_first`) as this queue takes it: its own setting, else `main`'s."""
        own = getattr(self, name)
        return own if own is not None else getattr(flags_at(self.workdir, "refs/remotes/origin/main"), name)

    def _same_or_fixed(self, m: Member, entries: dict[int, Entry]) -> bool:
        """Whether member ``m``'s PR is at the head `next` holds, or at a head that descends from it (a fix on top, or a merge of a
        PR below it): such a re-mark keeps its place, since its new head is preflighted and checked like any other, and a PR
        re-merged each time a stacked PR below it takes a fix isn't sent behind the rest of the queue."""
        new = entries[m.pr].pr.head
        return new == m.head or not _git(self.workdir, "merge-base", "--is-ancestor", m.head, new, check=False).returncode

    def _pending(self, m: Member, entries: dict[int, Entry]) -> bool:
        """Whether member ``m`` stays at its admitted head while its PR's new head, a descendant of it, waits for its ready mark:
        then the mark swaps the head in place (`_same_or_fixed`) instead of the PR, pushed one commit on, dropping out while its
        preflight runs and rejoining at the back.  Only while the admitted head is still marked: an ejection withdraws that
        mark."""
        e = entries.get(m.pr)
        if e is None or e.pr.head == m.head or e.state != "waiting" or not e.why.startswith("not marked ready"):
            return False
        return bool(self.ready(m.pr, m.head)) and self._same_or_fixed(m, entries)

    def _frozen(self, m: Member, entries: dict[int, Entry]) -> bool:
        """Whether member ``m`` stays at its old head: `frozen` names it, its PR is still open, neither ejected nor held, and that
        head is still marked ready: a head whose mark was withdrawn doesn't ride the running trains until their checks fail on
        it."""
        e = entries.get(m.pr)
        return (m.pr in self.frozen and e is not None and e.pr.head != m.head and e.state != "refused"
                and not e.why.startswith("held") and bool(self.ready(m.pr, m.head)))

    def _forces_cold(self, n: int) -> bool:
        r = _git(self.workdir, "diff", "--name-only", f"refs/remotes/origin/main...refs/queue/pr/{n}", check=False)
        return r.returncode == 0 and forces_cold(r.stdout.split())

    def _nonlean_order(self, tail: list[int], heads: dict[int, str], entries: dict[int, Entry]) -> list[int]:
        """``tail``, the PRs a sync adds onto `next` (all it merges after a `--first` PR placed again, or on a `main` that moved off
        `next`), at ``heads``: those whose own change is no Lean change (`_own_lean`)
        and that contain no Lean change of ``tail`` first, then the rest, each part in ``tail``'s order (`first`'s, then
        admission's).  A Lean change waits for a Lean check (`drive`'s review landing), and what sits above it waits with it.
        Only the tail moves: a rebuild of what the sync keeps would take a running Lean check's commit off `next`."""
        late: set[int] = set()
        for n in tail:
            if any(a in late for a in entries[n].after) or self._own_lean(heads[n], [heads.get(a, entries[a].pr.head)
                                                                                      for a in entries[n].after]):
                late.add(n)
        return [n for n in tail if n not in late] + [n for n in tail if n in late]

    def _own_lean(self, head: str, below: list[str]) -> bool:
        """Whether ``head``'s own change is a Lean change (`lean_change`): against the head it is stacked on (of ``below``, the
        open PRs' heads it contains, the one that contains the rest), else against its merge base with `main`."""
        def within(a: str, b: str) -> bool:
            return not _git(self.workdir, "merge-base", "--is-ancestor", a, b, check=False).returncode

        base = next((b for b in below if all(within(a, b) for a in below)), None)
        if base is None:
            r = _git(self.workdir, "merge-base", "refs/remotes/origin/main", head, check=False)
            base = r.stdout.strip() if r.returncode == 0 else "refs/remotes/origin/main"
        if (head, base) not in self._lean_of:
            self._lean_of[(head, base)] = lean_change(self.workdir, head, base)
        return self._lean_of[(head, base)]

    # -- building `next`

    def _merge(self, tip: str, pr: PR, members: list[Member]) -> tuple[str | None, str]:
        """(``pr``'s head merged onto ``tip``, "") or (None, why not), as `git merge --no-ff` in a checkout of ``tip`` gives them,
        without the checkout: `git merge-tree` merges, with ``tip``'s merge drivers (`--attr-source`), and `git commit-tree`
        makes the merge commit, so a candidate that conflicts costs one call (merging each candidate in the worktree, with
        many of them conflicting, takes tens of minutes).  A merge whose every conflict is in a pin is kept, each pin rebuilt
        from the merged tree (`repinned`).  A git without `--attr-source` (before 2.40) merges in the worktree
        (`_merge_in_worktree`), which refuses any conflict."""
        if not self._attr_source:
            return self._merge_in_worktree(tip, pr, members)
        r = _git(self.workdir, f"--attr-source={tip}", "merge-tree", "--write-tree", "--name-only", "--no-messages", tip, pr.head,
                 check=False)
        if r.returncode == 129 and "attr-source" in r.stderr:
            self._attr_source = False
            return self._merge_in_worktree(tip, pr, members)
        if r.returncode == 1:
            files = list(dict.fromkeys(r.stdout.splitlines()[1:]))
            if not (tree := repinned(self.workdir, r.stdout.split()[0], tip, pr.head, files)):
                return None, self._unmerged(r, files, members)
            self.log(f"research queue: #{pr.number} conflicts only in pins: {', '.join(files)} rebuilt from the merged tree")
        elif r.returncode:
            return None, self._unmerged(r, [], members)
        elif not _git(self.workdir, "merge-base", "--is-ancestor", pr.head, tip, check=False).returncode:
            return tip, ""                                      # already merged: `merge --no-ff` makes no commit either
        else:
            tree = r.stdout.split()[0]
        commit = _git(self.workdir, "commit-tree", tree, "-p", tip, "-p", pr.head, "-m", _message(pr), env=IDENTITY).stdout.strip()
        return commit, ""

    def _gated(self, main: str, gates: tuple[str, ...], tip: str, commit: str, pr: PR, before: dict[str, list[str]],
               verdicts: dict[str, list[str]]) -> tuple[list[str], list[str]]:
        """`gate_refusal` of ``commit``, ``pr``'s merge onto ``tip``, or the refusals ``before`` (the last syncs' verdicts under
        this `main` and these gates) holds for a merge with the same message, tree and parent tree: a lift or a moved member
        merges every member above it again, and those whose members below are the same set have the same trees, each ~15 s of
        gates.  The gates read only `main`, the merge's trees and message, and the members below, each judged at its own merge.
        What the gates say goes into ``verdicts``; a gate that couldn't judge is never kept."""
        trees = _out(self.workdir, "rev-parse", f"{tip}^{{tree}}", f"{commit}^{{tree}}").split()
        key = hashlib.sha256("\0".join([*trees, _message(pr)]).encode()).hexdigest()
        if key in before:
            verdicts[key] = before[key]
            return list(before[key]), []
        refused, unjudged = gate_refusal(self.workdir, main, commit, gates, onto=self.branch)
        if not unjudged:
            verdicts[key] = refused
        return refused, unjudged

    def _merge_in_worktree(self, tip: str, pr: PR, members: list[Member]) -> tuple[str | None, str]:
        """`_merge` by `git merge --no-ff` in the clone's worktree, checked out at ``tip``."""
        _git(self.workdir, "checkout", "-q", "--detach", tip)
        r = _git(self.workdir, "merge", "--no-ff", "--no-edit", "-m", _message(pr), pr.head, check=False, env=IDENTITY)
        if r.returncode == 0:
            return _out(self.workdir, "rev-parse", "HEAD"), ""
        files = _out(self.workdir, "diff", "--name-only", "--diff-filter=U").split()
        _git(self.workdir, "merge", "--abort", check=False)
        _git(self.workdir, "reset", "-q", "--hard", tip)
        return None, self._unmerged(r, files, members)

    def _unmerged(self, r: subprocess.CompletedProcess, files: list[str], members: list[Member]) -> str:
        if not files:
            return f"doesn't merge onto `{self.branch}`: {(r.stderr or r.stdout).strip()[-200:]}"
        by = touched_by(self.workdir, members, files, self._touched)
        return f"conflicts with `{self.branch}` in {', '.join(files)}" + (f" (touched by {', '.join(by)})" if by else "")

    def handle(self, pr: PR) -> str | None:
        """The Slack handle of the lane that owns ``pr``, from main's registry (`.agents/skills/using-slack/registry.json`): the
        lane its body names, the agent id it carries, the ready mark's author, else the agent whose id ends in its branch's
        suffix (`cursor/<name>-4292`); None when none of them is a handle."""
        if (got := self._read_registry()) is None:
            return None
        names, holders = got
        by = self.marked_by(pr.number, pr.head) if self.marked_by else None
        suffix = pr.branch.rpartition("-")[2] if "-" in pr.branch else ""
        tail = sorted({n for a, n in holders.items() if len(suffix) == 4 and a.endswith(suffix)})
        for c in (pr.owner, holders.get(pr.owner or ""), by, tail[0] if len(tail) == 1 else None):
            if c in names:
                return c
        return None

    def _read_registry(self) -> tuple[set, dict[str, str]] | None:
        """(the handles, {agent id: handle}) of `main`'s registry, read once per `fetch`: a review tick asks `handle` of each
        member of `next` (`Driver._owners`), and each `git show` is ~70 ms in drive's clone on the control pod's network volume,
        times over a hundred members.  None when it can't be parsed."""
        if self._registry is None:
            r = _git(self.workdir, "show", f"origin/main:{REGISTRY}", check=False)
            try:
                reg = json.loads(r.stdout).get("handles", []) if r.returncode == 0 else []
            except (ValueError, AttributeError):
                return None
            self._registry = ({h.get("handle") for h in reg if isinstance(h, dict)},
                              {a: h["handle"] for h in reg if isinstance(h, dict) and "handle" in h
                               for a in AGENT_RE.findall(str(h.get("holder", "")))})
        return self._registry

    def sync(self, dry_run: bool = False) -> tuple[list[Member], dict[int, Entry]]:
        """Rebuild `next` (see the module's `next`).  Off drive it refuses before anything (`_drive_only`)."""
        self._drive_only(self.branch, "queue sync", dry_run)
        self._clone()
        state = self.load_state()
        open_prs = self.github.open_prs()
        prs, waiting = self.fetch_prs([p for p in open_prs if p.base == "main"])
        self._seen(state, prs, {p.number for p in open_prs})
        chain = self.chain()
        entries = self.entries(prs, state)
        entries.update(waiting)
        frozen = {m.pr: m.head for m in chain if self._frozen(m, entries) or self._pending(m, entries)}
        # what the last sync left out at its head is tried again above the rest: in `--first`'s place or among members `next`
        # keeps, it would be left out again and every member above it merged again, each tick keeping none of them
        on = {m.pr for m in chain}
        out = {int(n) for n, c in state.get("conflicts", {}).items()
               if int(n) in entries and int(n) not in on and c["head"] == entries[int(n)].pr.head}
        out |= {n for n, e in entries.items() if n not in on and out.intersection(e.after)}

        def last(order: list[int]) -> list[int]:
            return [n for n in order if n not in out] + [n for n in order if n in out]

        main = _out(self.workdir, "rev-parse", "refs/remotes/origin/main")
        flag: list[bool] = []

        def nonlean() -> bool:
            if not flag:
                flag.append(self.flag("nonlean_first"))
            return flag[0]

        order = last(self.order(chain, entries, self.lift))
        heads = {n: frozen.get(n, entries[n].pr.head) for n in order}

        def kept(order: list[int]) -> int:
            want = [(n, heads[n]) for n in order]
            return next((i for i, (m, w) in enumerate(zip(chain, want)) if (m.pr, m.head) != w), min(len(chain), len(want)))

        k = kept(order)
        if lifted := k < len(self._lifted):
            self.log(f"research queue: {' '.join(f'#{n}' for n in self._lifted)} (reviewed) go to the bottom of `{self.branch}`, "
                     "below what waits for a review or a Lean check")
        # members `next` holds are merged again in their order above a moved or dropped member, and only what comes new goes
        # non-Lean first: a re-sort of the whole tail would undo the lift's order, and the next lift would merge it all again.
        # A `--first` PR placed again reorders from it; a `next` rebuilt on a moved `main` holds no member, so all of it is sorted
        refirst = not lifted and set(self.first) & set(order[k:])
        j = k if refirst else max(k, max((i + 1 for i, n in enumerate(order) if n in on), default=0))
        if len(order) - j > 1 and nonlean():
            order = last(order[:j] + self._nonlean_order(order[j:], heads, entries))
            k = kept(order)
        members = chain[:k]
        tip = members[-1].commit if members else _out(self.workdir, "rev-parse", "origin/main")
        dropped: set[int] = set()
        conflicts: dict[str, dict] = {}
        waits: dict[str, dict] = {}
        gates = gates_at(self.workdir, main) if order[k:] else ()
        cache = state.get("gate_verdicts") or {}
        before = (cache.get("of") or {}) if (cache.get("main"), cache.get("gates")) == (main, list(gates)) else {}
        verdicts: dict[str, list[str]] = {}
        merged = False
        for n in order[k:]:
            e = entries[n]
            if blocked := [a for a in e.after if a in dropped]:
                e.state, e.why = "waiting", f"contains {e.contains(blocked[0])}, which didn't merge"
                dropped.add(n)
                waits[str(n)] = {"head": e.pr.head, "why": e.why, "handle": self.handle(e.pr)}
                if state.get("waiting", {}).get(str(n), {}).get("head") != e.pr.head:
                    self.log(f"research queue: #{n} at {e.pr.head[:8]} waits off `{self.branch}`: {e.why}")
                continue
            (commit, why), merged = self._merge(tip, e.pr, members), True
            if commit is not None and gates:
                refused, unjudged = self._gated(main, gates, tip, commit, e.pr, before, verdicts)
                if refused or unjudged:
                    commit, why = None, " | ".join(refused + unjudged)
            if commit is None:
                e.state, e.why = "refused", why
                dropped.add(n)
                conflicts[str(n)] = {"head": e.pr.head, "why": why, "handle": self.handle(e.pr)}
                continue
            tip = commit
            members.append(Member(n, e.pr.head, e.pr.owner or "", commit))
        if merged:                  # drive runs train.sh and check.py from the worktree: `next`'s tip after a sync
            _git(self.workdir, "checkout", "-q", "--detach", tip)
        for i, m in enumerate(members, 1):
            entries[m.pr].state, entries[m.pr].position, entries[m.pr].commit = "queued", i, m.commit
        old = _git(self.workdir, "rev-parse", "--verify", "-q", f"refs/remotes/origin/{self.branch}", check=False).stdout.strip()
        if tip != old and not dry_run:
            self._remote(self.workdir, "push", "-q", f"--force-with-lease=refs/heads/{self.branch}:{old}", "origin",
                         f"{tip}:refs/heads/{self.branch}")
        self.log(f"research queue: `{self.branch}` at {tip[:12]}: {len(members)} PR(s) above `main`"
                 + (f", kept {k}, merged {len(members) - k}" if members else "")
                 + (f", gates' verdicts reused for {reused}" if (reused := len(verdicts.keys() & before.keys())) else "")
                 + (" (dry run: not pushed)" if dry_run else ""))
        if not dry_run:
            state["conflicts"], state["waiting"] = conflicts, waits
            if gates:
                newest = {**{h: v for h, v in before.items() if h not in verdicts}, **verdicts}
                state["gate_verdicts"] = {"main": main, "gates": list(gates), "of": dict(list(newest.items())[-GATE_VERDICTS:])}
            self.save_state(state)
        return members, entries

    def _drive_only(self, ref: str, verb: str, dry_run: bool) -> None:
        """Refuse a push of `main` or `next` (`merge.DRIVE_ONLY`) here unless this is drive (`merge.on_drive`), whatever the caller:
        GitHub's rulesets don't stop verity-agents[bot], and a VM's `research queue eject` that force-pushes a rebuild of `next`
        drops the Lean checks running on it.  `--shadow` gets no pass: its push of
        `queue-shadow` passes here, but `research queue sync`, `land` and `drive` refuse off drive with it as without, though
        it never pushes `main` or `next`."""
        from .merge import DRIVE_ONLY, off_drive, on_drive

        if not dry_run and ref in DRIVE_ONLY and not on_drive():
            raise QueueError(off_drive(ref, verb))

    def hand_sync(self, dry_run: bool = False) -> str | None:
        """`research queue sync` on the control pod, a sync that isn't drive's tick, under the holds drive's review tick keeps
        (`Driver._review`): none while a running Lean check of drive's holds a member whose mark is withdrawn (`lean_waits`), and
        `next` kept in place up to its highest commit with a running or passed one (`lean_hold`).  Why it synced nothing, or
        None."""
        self._drive_only(self.branch, "queue sync", dry_run)
        self._clone()
        self.fetch()
        state = self.load_state()
        if c := lean_waits(state, self.ready):
            why = f"`{self.branch}` waits for the Lean check of {c[:12]} to end before an ejection rebuilds it"
            self.log(f"research queue: {why}")
            return why
        chain = self.chain()
        self.hold = lean_hold(state, chain)
        self.frozen = {m.pr for m in chain[:self.hold]}
        if self.hold:
            self.log(f"research queue: drive's Lean checks keep `{self.branch}`'s bottom {self.hold} member(s) in place")
        self.sync(dry_run)
        return None

    def eject(self, pr: int, reason: str, by: str, sha: str | None = None) -> PR:
        """Withdraw ``pr``'s mark and keep it off the queue's branch; with ``sha``, only while that is still its head, so an
        eject that waits for drive's tick doesn't withdraw the mark of a head its owner pushed meanwhile."""
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
        """A hand ejection's notice, posted as ``by`` in the merge thread to `@ci` and the lane that owns ``p`` (only a handle's
        holder may post as it).  Drive's
        own ejections are told by `Driver.refuse`.  1, with the text to send, when the post failed."""
        from .drive import CI, _merge_thread, merge_thread_post

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

    # -- status

    def status(self, stale: Callable[[list[PR]], dict[int, float]] | None = None, close_days: int = 0) -> dict[int, Entry]:
        """Each open PR's entry; with ``stale`` (`research.stale.stale_prs` over the open PRs), each stale one says so, an unmarked
        PR based on another branch among them, and ``close_days`` (drive's `stale_close_days`) when drive closes it."""
        self._clone()
        state = self.load_state()
        open_prs = self.github.open_prs()
        prs, waiting = self.fetch_prs([p for p in open_prs if p.base == "main"])
        entries = self.entries(prs, state)
        entries.update(waiting)
        entries.update(self.off_main(open_prs))
        by_number = {p.number: p for p in open_prs}
        for n, since in (stale(open_prs) if stale else {}).items():
            p = by_number[n]
            e = entries.setdefault(n, Entry(p, state="waiting", why=f"based on {based_on(p, open_prs)}, and not marked ready at "
                                                                    f"{p.head[:8]}"))
            e.idle_since = since
            e.why = (f"stale, idle since {time.strftime('%Y-%m-%d', time.gmtime(since))}"
                     + (f" (drive closes it {close_days} days idle; push, mark it ready or close it)" if close_days else "")
                     + f": {e.why}")
        for i, m in enumerate(self.chain(), 1):
            if m.pr in entries and entries[m.pr].pr.head == m.head:
                entries[m.pr].state, entries[m.pr].position, entries[m.pr].commit = "queued", i, m.commit
        conflicts, waits = state.get("conflicts", {}), state.get("waiting", {})
        main = _out(self.workdir, "rev-parse", "refs/remotes/origin/main")
        for n in contained_first(entries):
            e = entries[n]
            if e.state == "ready" and (out := [a for a in e.after if a in entries and entries[a].state in ("refused", "waiting")]):
                e.state, e.why = "waiting", f"contains {e.contains(out[0])}, which is {entries[out[0]].state}: {entries[out[0]].why}"
            elif e.state == "ready" and (c := conflicts.get(str(n))) and c["head"] == e.pr.head:
                e.state, e.why = "refused", f"{c['why']} (at the last sync)"
            elif e.state == "ready" and (w := waits.get(str(n))) and w["head"] == e.pr.head:
                e.state, e.why = "waiting", f"{w['why']} (at the last sync)"
            elif e.state == "ready" and (files := conflicting_files(self.workdir, main, e.pr.head)):
                e.state, e.why = "refused", (f"conflicts with `main` in {', '.join(files)}: merge `main` into the branch and "
                                             f"re-mark it")
            elif e.state == "ready":
                e.why = f"admitted: the next sync merges it onto `{self.branch}`, and drive lands it on a land verdict"
        return entries

    def off_main(self, prs: list[PR]) -> dict[int, Entry]:
        """The open PRs among ``prs`` based on a branch other than `main` that are marked ready at their head, each waiting with
        the fix: drive admits only PRs based on `main`, so without this such a PR's mark went unseen until its parent landed and
        someone retargeted it."""
        off = [p for p in prs if p.base != "main"]
        if off and self.prefetch is not None:
            self.prefetch([(p.number, p.head) for p in off])
        return {p.number: Entry(p, state="waiting", why=f"based on {based_on(p, prs)}, not main: retarget it to main")
                for p in off if self.ready(p.number, p.head)}


#: what a change must touch to move many circuit-check targets' cache keys: the circuits package (every Definition imports its
#: library), the checker, the lock and the workspace's member list (installed distributions), a sidecar (the registries), and
#: what every target's entry reads or the registries import (an entry can read tens of files of vLLM's `program/` and these
#: packages, so a change only to `verity_vllm/pipeline/` and `config.py` reruns hundreds of targets)
_COLD_PREFIXES = ("verity/ml/circuits/", "tools/verity/circuit_check/", "integrations/vllm/verity_vllm/",
                  "verity/core/protocols/pouw/", "verity/core/protocols/pous/", "verity/ml/flock/python/",
                  "verity/ml/commitments/", "verity/ml/private_circuits/", "benchmarks/numerical/python/")


def contained_first(entries: dict[int, Entry]) -> list[int]:
    """``entries``' numbers, each after the PRs it contains (`Entry.after`), so a PR is judged after what it contains."""
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
    """Whether a change to ``files`` makes the next check's circuit-check run cold (hours, not seconds), so a sync merges it
    above the warm PRs (`Queue.order`): a scheduling estimate, never a gate."""
    for f in files:
        if f.endswith(".md") or "/tests/" in f:
            continue
        if f.startswith(_COLD_PREFIXES) or f in ("uv.lock", "pyproject.toml") or f.endswith(".circuit-check.py"):
            return True
    return False


def conflicting_files(repo: Path, base: str, head: str) -> list[str]:
    """The files a merge of ``head`` onto ``base`` would leave conflicted, from `git merge-tree` (no worktree touched), so
    `status` names a conflict on any VM, not only on the one whose last sync recorded it; [] when it merges cleanly.  It
    merges as `Queue._merge` does: with ``base``'s merge drivers (`--attr-source`, where the clone has them installed), and
    a conflict only in pins counts as none, since the sync rebuilds them (`repinned`; a conflict over PoUS's TRUSTED.sha256,
    which the sync merges, refuses nothing)."""
    r = _git(repo, f"--attr-source={base}", "merge-tree", "--write-tree", "--name-only", "--no-messages", base, head,
             check=False)
    if r.returncode == 129 and "attr-source" in r.stderr:
        r = _git(repo, "merge-tree", "--write-tree", "--name-only", "--no-messages", base, head, check=False)
    if r.returncode != 1:
        return []
    files = sorted({f for f in r.stdout.splitlines()[1:] if f})
    return [] if repinned(repo, r.stdout.split()[0], base, head, files) else files


def pins_at(repo: Path, ref: str) -> dict[str, tuple[str | None, str]]:
    """{path: (key, under)} of each `Pin` in the pins tool's `PINS` on ``ref`` (`PINS_FILES`), read from its text with `ast`,
    since research imports nothing from Verity; {} when ``ref`` holds none or it doesn't parse."""
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
    """{spec: its blob's bytes} for each of ``specs`` (`<tree>:<path>`) that names a blob, by one `git cat-file --batch`."""
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
    """The merged ``tree`` with each conflicted pin rebuilt, when every path in ``conflicted`` is a pin (`pins_at` ``ours``);
    else None.  A pin records the union of the paths both sides' versions record, OURS's in its order then THEIRS's new
    ones, less those the merged tree lacks, each with the sha256 of its blob there; a JSON pin keeps the rest of OURS's text.
    `pins.py --merged` would run each pin's command instead, which needs a checkout and Lake: two PRs that each changed a
    pinned input both changed the pin, and drive would eject the second (as over the PoUS TRUSTED.sha256)."""
    pins = pins_at(repo, ours)
    if not conflicted or any(p not in pins for p in conflicted):
        return None
    with tempfile.TemporaryDirectory() as tmp:
        env = {"GIT_INDEX_FILE": str(Path(tmp, "index"))}
        _git(repo, "read-tree", tree, env=env)
        for path in conflicted:
            key, under = pins[path]
            if not (sides := _blobs(repo, [f"{ours}:{path}", f"{theirs}:{path}"])):
                return None
            recorded: dict[str, str] = {}
            try:
                for text in (b.decode() for b in sides.values()):
                    pairs = ([ln.split("  ", 1)[::-1] for ln in text.splitlines() if ln.strip()] if key is None
                             else json.loads(text)[key].items())
                    recorded |= {p: h for p, h in pairs if p not in recorded}
            except (ValueError, KeyError, TypeError, AttributeError):
                return None
            full = {p: posixpath.normpath(posixpath.join(under, p)) for p in recorded}
            found = _blobs(repo, [f"{tree}:{f}" for f in full.values()])
            digests = {p: hashlib.sha256(found[f"{tree}:{f}"]).hexdigest() for p, f in full.items() if f"{tree}:{f}" in found}
            text = ("".join(f"{h}  {p}\n" for p, h in digests.items()) if key is None
                    else _with_json(next(iter(sides.values())).decode(), key, digests))
            blob = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=repo, input=text.encode(), capture_output=True,
                                  env=git_env(GIT_TERMINAL_PROMPT="0"), check=True).stdout.decode().strip()
            mode = _out(repo, "ls-tree", tree, "--", path).split(" ", 1)[0] or "100644"
            _git(repo, "update-index", "--add", "--cacheinfo", f"{mode},{blob},{path}", env=env)
        return _git(repo, "write-tree", env=env).stdout.strip()


def _with_json(text: str, key: str, value: dict) -> str:
    """``text``, a JSON object written with `indent=1`, with its top-level ``key`` set to ``value`` and every other byte kept."""
    doc = json.loads(text)
    if m := re.search(rf"^([ \t]*){re.escape(json.dumps(key))}:\s*(?=\{{)", text, re.M):
        old, end = json.JSONDecoder().raw_decode(text, m.end())
        if old == doc[key]:
            return text[:m.end()] + json.dumps(value, indent=1).replace("\n", "\n" + m.group(1)) + text[end:]
    return json.dumps({**doc, key: value}, indent=1) + "\n"


def rules_at(repo: Path, ref: str) -> Rules | None:
    """The admission rules on ``ref`` (`RULES`, else `RULES_BEFORE_MOVE`), or None when it holds neither."""
    for path in (RULES, RULES_BEFORE_MOVE):
        r = _git(repo, "show", f"{ref}:{path}", check=False)
        if r.returncode == 0:
            return Rules.parse(r.stdout, path)
    return None


def chain_of(repo: Path, base: str, ref: str) -> list[Member]:
    """The members on ``ref`` above ``base``, oldest first, from the trailers of its first-parent merges; none when ``ref`` is
    missing, ``base`` isn't below it, or a commit on that chain isn't the queue's."""
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
            return []                                   # a commit the queue didn't make: rebuild from `main`
        out.append(Member(int(t["PR"].lstrip("#")), t["Head"], t["Owner"], commit))
    return out


def changed(repo: Path, base: str, commit: str) -> set[str]:
    return set(_out(repo, "diff", "--name-only", "--no-renames", base, commit).split())


def built_on(repo: Path, ref: str) -> tuple[str, list[Member]]:
    """(the base ``ref`` was built on, its members oldest first), whether or not `main` has moved since; ("", []) when ``ref`` is
    missing or none of its first-parent chain is in `main`."""
    if _git(repo, "rev-parse", "--verify", "-q", ref, check=False).returncode:
        return "", []
    # not `merge-base`: a member stacked on a PR that has since landed on `main` gives a second, wrong common ancestor
    # (which empties the chain and cancels a disjoint check)
    base = next((c for c in _out(repo, "rev-list", "--first-parent", ref).split()
                 if not _git(repo, "merge-base", "--is-ancestor", c, "origin/main", check=False).returncode), "")
    if not base:
        return "", []
    out = []
    for rec in filter(None, (x.strip() for x in _out(repo, "log", "--first-parent", "--reverse",
                                                      "--format=%H%x00%B%x1e", f"{base}..{ref}").split("\x1e"))):
        commit, body = rec.split("\x00", 1)
        t = dict(TRAILER_RE.findall(body))
        if not {"PR", "Head", "Owner"} <= set(t):
            return base, []
        out.append(Member(int(t["PR"].lstrip("#")), t["Head"], t["Owner"], commit))
    return base, out


def hold(repo: Path, branch: str, running: Callable[[str, str], bool]) -> str | None:
    """Why `branch` stays on its base, or None: a check of one of its members is running (``running(commit, base)``: drive's
    Lean checks) and `main` moved by files none of the members up to it change, and by no check tooling (`CHECK_TOOLING`).  A
    rebuild on the new `main` would waste that check.  Every sync keeps it, an eject's included: rebuilding every PR onto a
    `main` drive is holding `next` off would cancel every running check."""
    base, chain = built_on(repo, f"refs/remotes/origin/{branch}")
    main = _out(repo, "rev-parse", "origin/main")
    if not chain or main == base:
        return None
    moved = changed(repo, base, main)
    if any(p.startswith(CHECK_TOOLING) for p in moved):
        return None
    for m in reversed(chain):
        if running(m.commit, base):
            if moved & changed(repo, base, m.commit):
                return None
            return (f"`main` moved to {main[:12]} by files `{branch}` doesn't change while {m.commit[:12]}'s check runs: "
                    f"`{branch}` stays on {base[:12]} until it ends")
    return None


def touched_by(repo: Path, members: list[Member], files: list[str], cache: dict[str, set[str]] | None = None) -> list[str]:
    """`#N` for each member whose merge changed one of ``files``; ``cache`` keeps each merge's files across calls (a sync names
    the members behind each of its conflicts)."""
    cache = {} if cache is None else cache
    for m in members:
        if m.commit not in cache:
            cache[m.commit] = set(_out(repo, "diff", "--name-only", f"{m.commit}^1", m.commit).split())
    return [f"#{m.pr}" for m in members if set(files) & cache[m.commit]]


# ---- command line -------------------------------------------------------------------------------------------------------------------

def _lock(workdir: Path):
    workdir.parent.mkdir(parents=True, exist_ok=True)
    f = open(workdir.parent / f"{workdir.name}.lock", "w")  # noqa: SIM115  (held until the process exits)
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


#: the quick tier, run from the checkout's root: the suites the PR changes since `--changed REV`, without the tests marked `slow`
SUITES = quick_tier.SUITES
QUICK_TIER = ["uv", "run", "--locked", "--all-packages", "--extra", "torch-cpu", "python", SUITES, "--quick"]
#: the Lake packages whose builds the suites' tests run, built before `suites.py` as `check` builds them before those suites: each
#: test worker runs `lake build` there, and workers building one from cold at once fail (`lake build` exit 1, a binary missing)
LEAN_BUILDS = ("verity/core/service/flock",)


class QuickTier:
    """`suites.py --quick --changed {merge base with main}` at a PR's head, from a clean checkout (`quick_tier.tier`, judged by
    `quick_tier.judge`): one worktree of the caller's repository per machine under `root`, checked out at the head and cleaned of
    everything but its `.venv` and the `.lake` of each of `LEAN_BUILDS` for each PR (one PR at a time: `{name}.lock`), so its
    environment syncs only what moved and Lake rebuilds only what changed.  Each of `LEAN_BUILDS` is built first (with the PATH's
    `lake`, else elan's; none, or a failed build, refuses).  Passes are cached by input (`suites.py`), so a re-run after a failure
    runs only what failed."""

    #: the worktree's name after the hash of the caller's git common dir: a remote tier ships its own (`-ship`), so a local tier,
    #: which holds its worktree's lock for the whole tier, never holds up a remote one's checkout from the same repository
    WORKTREE_SUFFIX = ""

    def __init__(self, repo: Path, remote: str, root: Path | None = None):
        self.repo, self.remote = Path(repo), remote
        self.root = Path(root or Path.home() / ".research" / "queue" / "ready")

    def __call__(self, pr: PR) -> str | None:
        """None when the tier passes at ``pr``'s head; else what failed, for the refusal."""
        with self._checkout(pr) as (wt, base):
            if base is None:
                return f"{pr.head[:12]} isn't on {self.remote} (pushed again since?)"
            out = self.root / "logs" / f"{pr.number}-{pr.head[:12]}"
            return quick_tier.judge(quick_tier.tier(wt, out, pr.head, base, QUICK_TIER, LEAN_BUILDS), pr.head)

    @contextlib.contextmanager
    def _checkout(self, pr: PR):
        """Under the worktree's lock: the worktree at ``pr``'s head, cleaned, and the head's merge base with `main` (None when the
        head isn't on the remote)."""
        common = Path(_out(self.repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
        wt = self.root / (hashlib.sha256(str(common).encode()).hexdigest()[:12] + self.WORKTREE_SUFFIX)
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / f"{wt.name}.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            _git(self.repo, "fetch", "-q", self.remote, "+refs/heads/main:refs/queue/ready/main",
                 f"+refs/pull/{pr.number}/head:refs/queue/ready/pr")
            if _git(self.repo, "cat-file", "-e", f"{pr.head}^{{commit}}", check=False).returncode:
                yield wt, None
                return
            _git(self.repo, "worktree", "prune")
            if not (wt / ".git").exists():
                shutil.rmtree(wt, ignore_errors=True)
                _git(self.repo, "worktree", "add", "-q", "--detach", "--force", str(wt), pr.head)
            else:
                _git(wt, "checkout", "-q", "--detach", "--force", pr.head)
            _git(wt, "clean", "-qffdx", "-e", "/.venv", *(x for d in LEAN_BUILDS for x in ("-e", f"/{d}/.lake")))
            yield wt, _out(self.repo, "merge-base", "refs/queue/ready/main", pr.head)


#: what a remote quick tier may take once it holds a check slot (`slot.py --run-max`), how long it may wait in the machine's line for
#: one first (`slot.py --wait-max`; the run's own timeout on the machine is the two together), and how much longer the caller waits
#: for its records to say it ended (custody's upload, a machine briefly unreachable) before it gives up on the run
REMOTE_TIMEOUT_S = 3 * 3600
REMOTE_WAIT_S = 6 * 3600
REMOTE_GRACE_S = 1800
REMOTE_POLL_S = 30
#: slot.py's exit when no check slot came within --wait-max (its NOT_STARTED): the tier never ran
SLOT_NOT_STARTED = 75
#: the remote tier's command, slot.py from the shipped tree as the launcher of a machine with /etc/vy/runs-via-slot accepts it
#: (pods/nebius/slot_only.py's `through_slot`, which the suites that read this module may not read)
SLOT = ("python3", "tools/verity/check/slot.py")
#: `research queue ready`'s exit when the tier never started: nothing failed, and nothing is marked
READY_NOT_STARTED = 3
#: a lane's own test run, which no landing needs (a PR lands on an eyeball review)
LANE_RUN = ('`research run --on vy-nebius-cpu-K --cwd source -- bash -lc "uv sync -q --all-packages --extra torch-cpu && '
            'uv run tools/verity/check/suites.py --changed $(git rev-parse origin/main) --quick"`')


class NotStarted(str):
    """Why a quick tier never ran (no check slot within its wait, no machine to run it on): not a failure of the PR's head."""


class StillGoing(str):
    """`ready --on`'s caller stopped waiting at research.waitcap's cap with the tier's run still going: nothing failed or was
    cancelled, and `research queue judge RUN` marks the PR once the run has ended."""


def place(spec: str, repo: Path, read: Callable[[list[str]], dict] | None = None) -> tuple[str | None, str]:
    """`--on auto` (check.py's AUTO) or `--on M1,M2`: the machine `check.py --on` would take for a lane check now (`choose` over
    each machine's `read_slots`, from the caller's checkout), with the Lean audit slots left out, since the quick tier takes none;
    None when no machine's slots could be read. And why, with each machine skipped."""
    from .store import tools_registry

    path = tools_registry.resolve("tools/verity/check/check.py", lambda p: (repo / p).is_file()) or "tools/verity/check/check.py"
    s = importlib.util.spec_from_file_location("queue_check", repo / path)
    check = importlib.util.module_from_spec(s)
    sys.modules[s.name] = check
    s.loader.exec_module(check)
    names = list(check.AUTO) if spec == "auto" else [n for n in spec.split(",") if n]
    docs = {n: ({**d, "lean": {"pools": None}} if isinstance(d, dict) else d) for n, d in (read or check.read_slots)(names).items()}
    got, why, skipped = check.choose(names, docs)
    return got, why + "".join(f"; skipped {s}" for s in skipped)


def _launch(argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True)


class RemoteQuickTier(QuickTier):
    """The quick tier as a `research run` on machine ``on`` (`auto` or `M1,M2`: the one `place` picks when the tier starts): a
    worktree of its own (`WORKTREE_SUFFIX`), checked out and cleaned as `QuickTier` does, is the run's `--source`, and the worktree's
    lock is held only while it is checked out and shipped, so several PRs' tiers run at once.  On the machine the run waits for a
    free check slot (the shipped tree's `tools/verity/check/slot.py`, as `check.py --record --on` takes one: the run's command is
    slot.py's, which a machine with /etc/vy/runs-via-slot requires) and runs `quick_tier.py` (sent with the run) in that tree.
    The caller polls the run's records (`research fetch`) until it ends and judges the facts in its result.json as the local tier's are judged; a run that ends without them, with facts from another run
    or head, or other than `done` with exit 0, is a failure.  A run that waited `wait_s` for a slot and gave up never started
    (`NotStarted`), and neither did one with no machine to run on.  The tier has `timeout_s` from taking its slot (`--run-max`), so
    the run on the machine is bounded by the two together (`--timeout`), and the caller's wait by that plus REMOTE_GRACE_S of
    ``clock`` time (a fetch can take its ssh's own timeout, so polls are not counted), past which it ends with the poll under way.
    The caller waits at most ``max_wait_s`` (research.waitcap's cap) of that: then the run goes on and the wait ends `StillGoing`."""

    WORKTREE_SUFFIX = "-ship"

    def __init__(self, repo: Path, remote: str, on: str, *, by: str = "", project: str = "verity", root: Path | None = None,
                 timeout_s: float = REMOTE_TIMEOUT_S, wait_s: float = REMOTE_WAIT_S, max_wait_s: float = waitcap.WAIT_CAP_S,
                 poll_s: float = REMOTE_POLL_S, launch: Callable[[list[str], Path], subprocess.CompletedProcess] = _launch,
                 fetch: Callable[[Path], dict] | None = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, place: Callable[[str, Path], tuple[str | None, str]] = place):
        super().__init__(repo, remote, root)
        self.on, self.by, self.project, self.poll_s = on, by, project, poll_s
        self.timeout_s, self.wait_s, self.max_wait_s = timeout_s, wait_s, max_wait_s
        self.launch, self.fetch, self.sleep, self.clock, self.place = launch, fetch, sleep, clock, place

    def __call__(self, pr: PR) -> str | None:
        from . import runs as R

        on = self.on
        if on == "auto" or "," in on:
            got, why = self.place(on, self.repo)
            if got is None:
                return NotStarted(f"--on {on}: {why}")
            print(f"research queue: --on {on}: {got}: {why}", flush=True)
            on = got
        job = Path(quick_tier.__file__)
        rid = R.new_run_id()
        with self._checkout(pr) as (wt, base):
            if base is None:
                return f"{pr.head[:12]} isn't on {self.remote} (pushed again since?)"
            args = [*(x for d in LEAN_BUILDS for x in ("--lean", d)), "--head", pr.head, "--base", base, "--", *QUICK_TIER]
            opts = [*(["--by", self.by] if self.by else []), "--wait-max", f"{self.wait_s:g}", "--run-max", f"{self.timeout_s:g}"]
            r = self.launch([sys.executable, "-m", "research", "run", "--id", rid, "--on", on, "--project", self.project,
                             "--source", str(wt), "--cwd", "source", "--send", str(job), "--require-result",
                             "--timeout", f"{self.wait_s + self.timeout_s:g}", "--custody-ttl", str(math.ceil(self.wait_s + self.timeout_s + REMOTE_GRACE_S)),
                             "--declared-output", "out/*", "--", *SLOT, *opts,
                             "--", "sh", "-c", f'exec python3 "$RESEARCH_RUN_DIR/inputs/{job.name}" {shlex.join(args)}'], wt)
        if r.returncode:
            return f"`research run --on {on}` failed (exit {r.returncode}): {(r.stderr or r.stdout or '').strip()[-400:]}"
        print(f"research queue: the quick tier runs as {rid} on {on}, in a check slot; waiting for it to end "
              f"(`research inspect {rid}`)", flush=True)
        d = R.runs_root() / rid
        status, why = self._wait(d, on)
        return why or judge_run(d, rid, on, pr.head, status)

    def _wait(self, d: Path, on: str) -> tuple[dict, str | None]:
        """The run's status.json once it says the run ended, from its records fetched every `poll_s` until the wait's deadline; or why
        the wait gave up."""
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
                # a fetch copies status.json before it probes the runner, which can record its end and exit in between
                # (a run that passed can read as gone): fetch once more before calling the run lost
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
    """Run ``rid``'s result.json (its records in ``d``, its `status.json` ``status``) as a `quick_tier` report, and None; or None and
    why it isn't one."""
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


def judge_report(res, rid: str, on: str, head: str, status: dict) -> str | None:
    """None when the report ``res`` of run ``rid``, ended with ``status``, shows the tier passed at ``head``; else what failed."""
    if (why := quick_tier.judge(res.doc.get("tier"), head, f"its output is in run {rid}'s log on {on}")) is not None:
        return why
    if status.get("state") != "done" or status.get("rc") != 0 or res.validation_status != "passed":
        return (f"run {rid}'s facts show a pass, but it ended {status.get('state')} (exit {status.get('rc')}) with validation "
                f"{res.validation_status}")
    return None


def judge_run(d: Path, rid: str, on: str, head: str, status: dict) -> str | None:
    """None when ended run ``rid`` (records in ``d``) reports a pass of the quick tier at ``head``; else what failed: a run that ends
    without the facts, with facts from another run or head, or other than `done` with exit 0, is a failure."""
    res, why = tier_report(d, rid, on, status)
    return why or judge_report(res, rid, on, head, status)


def _json_file(p: Path) -> dict:
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def same_tree_mark(store, repo: Path, remote: str, pr: PR) -> str | None:
    """Another head of ``pr`` marked ready whose tree is ``pr.head``'s (a merge of `main` that changes no file), or None.  The tier
    at that head ran these exact files, and when a clean merge of `main` leaves the tree as it was, the files the new head changes
    against `main` are among those the marked head changed against its merge base, so the tier would run no suite it didn't.  None
    when either commit isn't here after fetching the PR's head (a rebase drops the old one), and QueueError when the PR's labels
    can't be pulled from the store's remote."""
    _fetch_heads(repo, remote, [pr])
    if not (want := _tree(repo, pr.head)):
        return None
    marked = StoreLabels(store)._other_heads(pr.number, pr.head, vocab.READY)
    return next((old for old, labs in sorted(marked.items()) if any(str(lab.value).lower() == "true" for lab in labs)
                 and _tree(repo, old) == want), None)


def same_tree_prs(repo: Path, remote: str, prs: list[PR], head: str) -> list[PR]:
    """The PRs of ``prs`` whose head has ``head``'s tree, after fetching their heads (none when ``head`` isn't here)."""
    _fetch_heads(repo, remote, prs)
    want = _tree(repo, head)
    return [p for p in prs if want and _tree(repo, p.head) == want] if prs else []


def _fetch_heads(repo: Path, remote: str, prs: list[PR]) -> None:
    """Each PR's head into a ref of its own, so a tier's fetch under the worktree's lock never meets this one."""
    if prs:
        _git(repo, "fetch", "-q", remote, *(f"+refs/pull/{p.number}/head:refs/queue/same-tree/{p.number}" for p in prs), check=False)


def _tree(repo: Path, commit: str) -> str:
    return _git(repo, "rev-parse", "-q", "--verify", f"{commit}^{{tree}}", check=False).stdout.strip()


#: the invariant tests the preflight runs on the merged tree, from the root, each a common way a train breaks: a module string
#: naming a module only an unlanded PR adds, stale Lake pins, a doc path, a workspace member without its REAL_ROOTS entry, and
#: `check`'s preflight-lints on test_repository.py (a skill over the Markdown cap, a report-genre name); about 5 s
PREFLIGHT_TEST_PATHS = ("tests/test_module_strings.py", "tests/test_pins.py", "tests/test_doc_paths.py",
                        "tests/test_repository.py", "tools/research/tests/test_pythonpath.py")


def _by_suite(paths: tuple[str, ...]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """`paths` grouped by the directory pytest runs each in (the part before `tests/`), each relative to it."""
    groups: dict[str, list[str]] = {}
    for path in paths:
        d, sep, rest = path.rpartition("tests/")
        groups.setdefault(d.rstrip("/") or ".", []).append(sep + rest)
    return tuple((d, tuple(fs)) for d, fs in groups.items())


#: `PREFLIGHT_TEST_PATHS` by the directory pytest runs in
PREFLIGHT_TESTS = _by_suite(PREFLIGHT_TEST_PATHS)
#: lints the preflight runs on the merged tree only when the change reaches what they judge (a path starting with, or holding
#: a directory, of the prefix): they take about 20 s and 40 s, and a train that fails one of them loses its hour-long check
PREFLIGHT_LINTS = (("tests/test_no_wall_clock.py", ("tests/",)), ("integrations/vllm/tests/lint", ("integrations/vllm/",)))
#: suites whose collection is no cheap check, each with why; the preflight names them and leaves them to the train's check
PREFLIGHT_UNCOLLECTED: dict[str, str] = {}
#: the merged tree's environment, as the quick tier's `uv run --locked` has it (a fresh tree syncs from uv's cache in seconds)
PREFLIGHT_SYNC = ("uv", "sync", "--locked", "--all-packages", "--extra", "torch-cpu", "-q")
#: what the merged tree's checks may take together; what hasn't finished by then is named and refuses nothing.  Every suite's
#: collection but circuit-check's takes about 18 s on 4 workers of a 4-core VM; `PREFLIGHT_LINTS` add up to a minute
PREFLIGHT_BUDGET_S = 120.0
#: collections at once: four torch-importing `--collect-only` runs reach about 8 GB, half of a marker's 16 GB VM
PREFLIGHT_JOBS = 2
#: below this share of memory available, no suite is collected, so a mark never OOMs the VM's other agents (the shell hook's `AGENT_GUARD_MIN_AVAIL`)
PREFLIGHT_MIN_FREE = 0.25
#: the Lean step's bounds (`Preflight._lean`): the most modules it builds (those the change touches and every module importing one,
#: transitively, each rebuilt since an import changed), and the seconds its builds may take, on the merged tree and, for what fails
#: there, on `next`; past either it notes why and refuses nothing.  With `PREFLIGHT_BUDGET_S` the whole preflight stays within
#: 5 minutes, and a warm tree rebuilds a few changed modules and their importers in a minute or two
PREFLIGHT_LEAN_MODULES = 40
PREFLIGHT_LEAN_S = 180.0
#: where the Lean step builds for `ready --on` when this machine keeps no warm Lean tree (cpu-3's `build` Lean slots, its research
#: pool the least loaded; cpu-1's pool is congested).  The one place to move it
PREFLIGHT_LEAN_ON = "vy-nebius-cpu-3"
#: the `research run` pool it takes there: none, as `verity lean try --on vy-nebius-cpu-3` runs, since the node's `build` Lean slots
#: are the gate; a node without them would take its `research` pool
PREFLIGHT_LEAN_POOL: str | None = None
#: the merged tree's builder (`lean_changed.py --build FILE...`, on a `build` Lean slot's warm tree), and its last line's prefix
LEAN_BUILDER, LEAN_BUILD_LINE = "tools/verity/lean/lean_changed.py", "lean-build: "
#: `lean_slot.py`'s SLOTS: a machine whose file lists a `build` pool keeps warm Lean trees; an agent VM has none
LEAN_SLOTS = Path(os.environ.get("LEAN_SLOTS", "/workspace/research/locks/lean-slots"))
#: the Lake packages the Lean step reads: a directory with a lakefile and a `lean-audit.json`, the auditor's fixtures aside
LEAN_FIXTURES = "tools/verity/lean/"
#: the temporary worktrees' prefix (`merged_worktree`), `<tmp>/research-merged-<pid>-*`: one whose process is gone is swept (a kill
#: skips `finally`)
MERGED_PREFIX = "research-merged-"
#: prints, as JSON, the paths of the suites `suites.py`'s `touched` selects for the change from commit argv[1] to tree argv[2]
#: (`suites`) and, for each, the fetched registry entries under its inputs (`fixtures`), as `suites.py` picks what to fetch before
#: it runs them: the repository's own suite reads tracked files only, so it needs none
_TOUCHED = ("import json, sys; sys.path.insert(0, 'tools/verity/check'); import suites as S; pkgs = S.members(); "
            "chosen = S.touched(S.discover(), pkgs, sys.argv[1], sys.argv[2], S.locked()); "
            "ins = {s: S.inputs_of(s, pkgs) for s in chosen}; own = {s: [p for p in ps if p != '.'] for s, ps in ins.items()}; "
            "fx = S.fetched_fixtures(sys.argv[2], {p for ps in own.values() for p in ps}); "
            "print(json.dumps({'suites': [s.path for s in chosen], 'fixtures': {s.path: sorted(p for p in fx "
            "if S.covered(p, own[s], S.dropped(s, ins[s]))) for s in chosen}}))")
#: what the preflight's fetch of its suites' fixtures (`research data fetch-fixtures`) may take of `PREFLIGHT_BUDGET_S`: the
#: machine's store keeps every blob it fetches, so a later preflight here copies them from it in seconds
PREFLIGHT_FETCH_S = 45.0
#: a line of pytest's short summary: `FAILED tests/x.py::test_y - why`, `ERROR tests/x.py - why` (a collection error)
def free_memory() -> float | None:
    """The share of this machine's memory available (`/proc/meminfo`'s MemAvailable over MemTotal), or None where unreadable."""
    try:
        info = dict(ln.split(":", 1) for ln in Path("/proc/meminfo").read_text().splitlines() if ":" in ln)
        return int(info["MemAvailable"].split()[0]) / int(info["MemTotal"].split()[0])
    except (OSError, KeyError, ValueError, ZeroDivisionError):
        return None


_SUMMARY_RE = re.compile(r"^(?:FAILED|ERROR) (.+?)(?: - (.*))?$", re.M)
_LOCK_STALE = "needs to be updated"

@dataclass
class Touched:
    """What `touched_suites` gave.  ``failed``: each failed suite in `suites.py --json`'s shape (`quick_tier.ROW_KEYS`, status
    failed, with each failing test's summary line under `summary`); ``suites``: those that ran here; ``ids``: the runs launched;
    ``notes``: what kept a suite or a launch from running; ``tail``: the launch's output; ``fixtures``: each selected suite's
    fetched fixtures (`LocalSuites`)."""
    suites: list[str] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    tail: str = ""
    fixtures: dict[str, list[str]] = field(default_factory=dict)


def failed_row(suite: str, failures: dict[str, str]) -> dict:
    """Suite ``suite``'s failures, {node id: its summary line}, as `suites.py --json` reports a failed suite."""
    return {"name": suite, "suite": suite, "status": "failed", "failed_tests": list(failures), "log": None,
            "summary": list(failures.values())}


def pytest_failures(run: Callable[[list[str], Path], tuple[subprocess.CompletedProcess | None, str]], py: str, cwd: Path,
                    args: list[str]) -> dict[str, str] | str:
    """{node id: its summary line} of what failed in `pytest ARGS` in ``cwd`` (empty: passed), or why it didn't run."""
    r, why = run([py, "-P", "-m", "pytest", "-q", "-p", "no:cacheprovider", *args], cwd)
    if r is None:
        return why
    if r.returncode in (0, 5):                                   # 5: nothing collected
        return {}
    found = {m.group(1): m.group(0) for m in _SUMMARY_RE.finditer(r.stdout)}
    return found or {f"exit {r.returncode}": f"pytest exited {r.returncode}: {(r.stdout + r.stderr).strip()[-300:]}"}


@contextlib.contextmanager
def worktrees_locked(repo: Path):
    """``repo``'s worktrees to this caller alone: train.sh's lock (`<git common dir>/train.lock`), so drive's launches at once
    and train.sh never add, remove or prune a worktree beside another (two at once fail with "Could not write new index
    file")."""
    common = Path(_out(repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    with open(common / "train.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


@contextlib.contextmanager
def merged_worktree(repo: Path, commit: str, root: Path | None = None):
    """A detached worktree of ``repo`` at ``commit`` in a temporary directory under ``root`` (`MERGED_PREFIX`), removed with it
    whatever happens; one a dead process left is swept first."""
    root = Path(root or tempfile.gettempdir())
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


@dataclass
class LocalSuites:
    """`touched_suites` here, as the preflight runs them: `pytest ARGS` (`--collect-only`) in each suite `suites.py`'s own `touched`
    selects (run with the tree's ``python``; `PREFLIGHT_UNCOLLECTED` left out), or in ``suites`` alone when given, ``jobs`` at once,
    each through ``run`` (the caller's budget); none while under `PREFLIGHT_MIN_FREE` of memory is available.  First ``fetch``
    (the tree, the paths; gives the paths still missing and why) brings the fetched fixtures under those suites' inputs into the
    tree (`_TOUCHED`'s, or ``fixtures`` with ``suites``), since a test may read one at import; a suite whose fixtures it couldn't
    fetch isn't collected, and one note says so."""
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
            out.notes.append(f"no suite collected: {f:.0%} of memory available, under {PREFLIGHT_MIN_FREE:.0%}; the after-landing "
                             "run still collects them")
            todo = []
        need = sorted({p for s in todo for p in out.fixtures.get(s, ())})
        if need and self.fetch is not None:
            missing, why = self.fetch(wt, need)
            if blocked := [s for s in todo if missing & set(out.fixtures.get(s, ()))]:
                shown = ", ".join(sorted(missing)[:3]) + (f" and {len(missing) - 3} more" if len(missing) > 3 else "")
                out.notes.append(f"{', '.join(blocked)} not judged: {len(missing)} of the {len(need)} fixture(s) the suites read "
                                 f"weren't fetched ({why or 'no source held them'}: {shown}); the after-landing run fetches them")
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
            out.notes.append(f"collected {len(todo)} suite(s)" + (f" ({', '.join(todo)})" if len(todo) <= 6 else ""))
        return out


def touched_suites(repo: Path, commit: str, base: str, how: LocalSuites, *, wt: Path | None = None,
                   root: Path | None = None) -> Touched:
    """The suites the change from ``base`` to ``commit`` (a merged tree) touches, run at ``commit`` by ``how`` within the
    caller's budget (`LocalSuites`: the preflight, a PR's head merged onto `next`).  In ``wt`` when given (checked out at
    ``commit``), else in a temporary worktree of ``repo`` at ``commit`` (`merged_worktree`), removed once ``how`` returns.  The
    failures come back in `suites.py --json`'s shape, in `Touched.failed`.  A worktree that can't be made is a note, and
    nothing runs."""
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
    """The modules a Lean file's header imports (`import M`, `public import M`, `meta import M`, `import all M`), read up to its
    first command, past comments (nested `/- -/` too), `module` and `prelude`."""
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
    """A Lake package of a tree: its directory (``path``, from the root), where its modules' names start (``src``, its `srcDir`),
    its libraries' roots (a module under one is what `lake build +M` builds) and the packages it requires by path (its manifest's
    `path` entries, the inherited ones too)."""
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
    """The Lake packages of the tree ``wt`` holding ``files``: each directory with a `lean-audit.json` and a lakefile, outside
    `LEAN_FIXTURES`."""
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
    """The Lean step's view of a merged tree: {file: module} of each `.lean` file of the packages a change reaches, its imports
    among them, and ``targets``, {file: module} of what the step builds."""
    module: dict[str, str] = field(default_factory=dict)
    imports: dict[str, list[str]] = field(default_factory=dict)
    targets: dict[str, str] = field(default_factory=dict)

    def upstream(self, f: str) -> list[str]:
        """The files ``f``'s module imports, transitively, among those the graph holds."""
        files = {m: g for g, m in self.module.items()}
        seen, todo = set(), list(self.imports.get(self.module.get(f, ""), []))
        while todo:
            if (m := todo.pop()) in seen or m not in files:
                continue
            seen.add(m)
            todo += self.imports.get(m, [])
        return sorted(files[m] for m in seen)


def lean_graph(wt: Path, files: list[str], changed: list[str]) -> LeanGraph:
    """What the Lean step builds for a change to ``changed`` (paths from the root, deleted ones too) on the tree ``wt`` holding
    ``files``: for each Lake package a changed `.lean` file belongs to (`lake_packages`, the deepest holding it), the changed
    modules and every module importing one, transitively (`lean_imports`), among that package's own files and those of every
    package requiring it by path; only modules under a library's root, which `lake build +M` names.  Empty without a Lean change."""
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
    """What the Lean step's build gave: {file: its first Lean error} for each file that failed, and what it said otherwise."""
    failed: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def run_group(argv: list[str], cwd: Path, left: float) -> tuple[int | None, str]:
    """``argv``'s exit status and output (stderr after stdout), in a process group of its own that is killed whole once ``left``
    seconds pass (a `lake build` it started included, which would otherwise go on in a slot's tree), and then None."""
    p = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace",
                         stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        out, err = p.communicate(timeout=max(left, 0))
        return p.returncode, out + err
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGKILL)
        out, err = p.communicate()
        return None, out + err


#: `research run`'s line naming the run it started (or refused), as `verity lean try` reads it
RUN_STARTED_RE = re.compile(r"^research: run (\S+) (?:launched|refused) on ", re.M)


@dataclass
class LakeBuild:
    """The Lean step's builds (`Preflight._lean`): the tree's `LEAN_BUILDER` (`lean_changed.py --build FILE...`), which builds the
    files' modules on the warm tree of a node's `build` Lean slot as `verity lean try` builds one, and ends with one
    `LEAN_BUILD_LINE`.  It runs here when this machine's Lean slot file (``slots``) lists a `build` pool, in a process group
    killed at the step's deadline; else, given ``on`` (`ready --on`: `PREFLIGHT_LEAN_ON`), as a `research run` of the tree there
    on its `build` Lean slot, or in ``pool`` (a slot named for ``by``) when one is given, bounded by the same deadline; else not
    at all, and says so: an agent VM builds nothing (the team skill, "Machines"), and its preflight never starts a run unasked."""
    on: str | None = None
    pool: str | None = PREFLIGHT_LEAN_POOL
    by: str | None = None
    slots: Path = LEAN_SLOTS
    run: Callable[[list[str], Path, float], tuple[int | None, str]] = run_group
    runs: Callable[[], Path] | None = None

    def warm(self) -> bool:
        try:
            return any(ln.split("#")[0].split()[:1] == ["build"] for ln in self.slots.read_text().splitlines())
        except OSError:
            return False

    def __call__(self, wt: Path, files: list[str], left: float) -> LeanBuilt | str:
        if left <= 0:
            return "the Lean step's budget was spent before it started"
        if not (wt / LEAN_BUILDER).is_file():
            return f"the merged tree has no {LEAN_BUILDER}"
        if self.warm():
            rc, text = self.run([sys.executable, LEAN_BUILDER, "--build", *files], wt, left)
            where = "here"
        elif self.on:
            rc, text, where = self._remote(wt, files, left)
        else:
            return (f"this machine keeps no warm Lean tree (no `build` pool in {self.slots}; an agent VM builds nothing): "
                    f"`ready --on MACHINE` builds them on {PREFLIGHT_LEAN_ON}")
        if rc is None:
            return (f"its build {where} hadn't ended after {left:.0f} s (a wait in its pool's line counts){': ' + text if text else ''}"
                    if where != "here" else f"its build here was still going after {left:.0f} s{': ' + text if text else ''}")
        doc = next((ln[len(LEAN_BUILD_LINE):] for ln in reversed(text.splitlines()) if ln.startswith(LEAN_BUILD_LINE)), None)
        try:
            doc = json.loads(doc) if doc is not None else None
        except ValueError:
            doc = None
        if not isinstance(doc, dict):
            return f"its build {where} exited {rc} without a result line: {text.strip()[-300:]}"
        built, skipped = doc.get("built") or {}, doc.get("skipped") or {}
        notes = [f"{p} not built: {why}" for p, why in skipped.items()]
        failed = {f: str(e) for f, e in (doc.get("failed") or {}).items()}
        notes += [f"`lake build` in {p} exited {c} with no Lean error in a file" for p, c in built.items()
                  if c and not any(f.startswith(p + "/") for f in failed)]
        if not built and not failed:
            return "; ".join(notes) or f"its build {where} built nothing (exit {rc})"
        return LeanBuilt(failed, notes)

    def _remote(self, wt: Path, files: list[str], left: float) -> tuple[int | None, str, str]:
        """`LEAN_BUILDER --build` as a `research run` of ``wt``'s HEAD on ``on``, waited for within ``left``; its exit status (None:
        it didn't end), its stdout, and where it ran."""
        from . import runs as R

        secs = f"{max(left - 10, 1):.0f}"
        argv = [sys.executable, "-m", "research", "run", "--on", self.on, *(["--pool", self.pool, "--pool-wait", secs] if self.pool else []),
                *(["--as", self.by] if self.by else []), "--project", "verity", "--source", str(wt), "--cwd", "clone",
                "--timeout", secs, "--wait", "--wait-every", "5", "--wait-max", secs, "--",
                "python3", LEAN_BUILDER, "--build", *files]
        rc, said = self.run(argv, wt, left)
        where = f"on {self.on}"
        if not (m := RUN_STARTED_RE.search(said)):
            return rc, said.strip()[-300:] if rc is not None else "", where
        rid, where = m.group(1), f"on {self.on} (run {m.group(1)})"
        self.run([sys.executable, "-m", "research", "fetch", rid, "--on", self.on, "--all"], wt, 60)
        log = (self.runs or R.runs_root)() / rid / "stdout.log"
        text = log.read_text(errors="replace") if log.is_file() else ""
        return (None if rc == waitcap.STILL_GOING else rc), text, where


class Preflight:
    """`ready`'s preflight, before the mark (and before the quick tier): refused, each refusal naming its fix, when the head
    (a) conflicts with `next` (`conflicting_files`, no worktree), naming the paths and the PRs on `next` that touched them;
    (b) merged onto `next`, fails `pins.py --stale`, a test of `PREFLIGHT_TEST_PATHS`, a lint of `PREFLIGHT_LINTS` the change
    reaches, or `pytest --collect-only` in a suite whose inputs
    the change touches (`touched_suites` with `LocalSuites`, from `next` to the merged tree), or the merged tree's `uv sync --locked`
    finds its lock stale; (c) contains another open PR's head that isn't admitted (no ready mark at that head, held, or no
    owner), naming each; (d) fails one of drive.toml's `landing_gates` at `main` (`_gates`), with the gate's fix; or (e) merged
    onto `next`, breaks the build of a Lean module that is or imports, transitively, a module it changes, in that module's package
    or one requiring it by path (`lean_graph`), built by ``lean`` (`LakeBuild`, on a warm Lean tree) within `lean_s`, naming the
    module, its first error and the PR on `next` that owns it (`touched_by`), or for a module of its own, those that changed what it
    imports.  A train most often breaks on one of (a) to (c), and a Lean break is most often one of (e).

    `next` is fetched fresh; a PR already on it is merged onto the commit below its old head, as drive's next sync would rebuild
    it.  The checks of (b) and (e) run in a temporary worktree of the caller's repository at the merge (`merged_worktree`), (e)
    first, within `lean_s`, while the tree is clean (a `research run` ships it), then (b) with its own environment (`sync`;
    ``python`` stands in for its interpreter), under `budget_s` together, and the worktree is removed whatever happens.  Before
    (b), the fetched fixtures under the collected suites' inputs come into it (``fetch``, by default `_fetch_fixtures`: through
    ``store``, within `PREFLIGHT_FETCH_S` of the budget); a suite whose fixtures can't be fetched (no credentials, no source) is
    named once and not judged.  A failure that `next` has too (a pin already stale there, a module two PRs on `next` already
    broke) isn't the PR's: each failed check runs again at `next` in the same worktree, within the same budget, and only what
    fails on the merge alone refuses.  What keeps a check from running (no `next` to fetch, the budget spent, labels that can't
    be pulled, more than `lean_modules` Lean modules, no warm Lean tree) is said and refuses nothing: the preflight saves a
    revert, and the after-landing run still checks the head."""

    def __init__(self, repo: Path, remote: str, store, *, sync: tuple[str, ...] | None = PREFLIGHT_SYNC, python: str | None = None,
                 budget_s: float = PREFLIGHT_BUDGET_S, jobs: int = PREFLIGHT_JOBS, clock: Callable[[], float] = time.monotonic,
                 root: Path | None = None, log: Callable[[str], None] = print,
                 free: Callable[[], float | None] = free_memory,
                 fetch: Callable[[Path, list[str], float], tuple[set[str], str]] | None = None,
                 lean: Callable[[Path, list[str], float], LeanBuilt | str] | None = None,
                 lean_s: float = PREFLIGHT_LEAN_S, lean_modules: int = PREFLIGHT_LEAN_MODULES):
        self.repo, self.remote, self.store, self.sync, self.python = Path(repo), remote, store, sync, python
        self.budget_s, self.jobs, self.clock, self.log, self.free = budget_s, jobs, clock, log, free
        self.fetch = fetch or self._fetch_fixtures
        self.lean, self.lean_s, self.lean_modules = lean, lean_s, lean_modules
        self.root = Path(root or tempfile.gettempdir())

    def __call__(self, pr: PR, prs: list[PR]) -> str | None:
        """None when the head may be marked; else the refusal."""
        start = self.clock()
        try:
            base, onto, members = self._fetch(pr)
        except QueueError as e:
            self.log(f"research queue: preflight not run ({e}); the after-landing run still tests the head")
            return None
        notes: list[str] = []
        found = self._contained(pr, prs) + self._gates(pr, notes)
        if files := conflicting_files(self.repo, base, pr.head):
            by = touched_by(self.repo, members, files)
            found.insert(0, f"it conflicts with {onto} in {', '.join(files)}" + (f" (touched by {', '.join(by)})" if by else "")
                         + ": merge `main` once those land, or settle it with their owners; never merge `next` into the branch "
                           "(it carries unlanded PRs)")
        else:
            try:
                found += self._merged(pr, base, onto, notes, members)
            except QueueError as e:
                notes.append(f"the merged tree's checks not run ({e})")
        took = f"{self.clock() - start:.0f} s" + (f"; {'; '.join(notes)}" if notes else "")
        if found:
            return (f"the preflight of its head merged onto {onto} refused it ({took}):\n  - " + "\n  - ".join(found)
                    + "\n  (`--no-preflight` skips the preflight: the lander's)")
        self.log(f"research queue: preflight: #{pr.number} at {pr.head[:12]} merged onto {onto}: nothing refused ({took})")
        return None

    def _fetch(self, pr: PR) -> tuple[str, str, list[Member]]:
        """The commit to merge the head onto (`next`, below the PR's own member when it has one; `main` when there's no `next`),
        how to name it, and `next`'s members."""
        _git(self.repo, *STALL, "fetch", "-q", self.remote, "+refs/heads/main:refs/queue/preflight/main",
             f"+refs/pull/{pr.number}/head:refs/queue/preflight/pr")
        missing = lambda: _git(self.repo, "cat-file", "-e", f"{pr.head}^{{commit}}", check=False).returncode   # noqa: E731
        if missing() and pr.branch:         # GitHub's pull ref lags a push by seconds; the branch has the head `ready` read
            _git(self.repo, *STALL, "fetch", "-q", self.remote, f"+refs/heads/{pr.branch}:refs/queue/preflight/branch", check=False)
        if missing():
            raise QueueError(f"{pr.head[:12]} isn't on {self.remote} (pushed again since?)")
        main = _out(self.repo, "rev-parse", "refs/queue/preflight/main")
        r = _git(self.repo, *STALL, "ls-remote", "--exit-code", self.remote, f"refs/heads/{BRANCH}", check=False)
        if r.returncode == 2:
            return main, f"`main` at {main[:12]} (no `{BRANCH}`)", []
        if r.returncode:
            raise QueueError(f"git ls-remote {self.remote}: {(r.stderr or r.stdout).strip()[-200:]}")
        _git(self.repo, *STALL, "fetch", "-q", self.remote, f"+refs/heads/{BRANCH}:refs/queue/preflight/{BRANCH}")
        tip = _out(self.repo, "rev-parse", f"refs/queue/preflight/{BRANCH}")
        members = chain_of(self.repo, main, tip)
        if own := next((m for m in members if m.pr == pr.number), None):
            below = _out(self.repo, "rev-parse", f"{own.commit}^1")
            return below, f"`{BRANCH}` below its merge of #{pr.number} at {own.head[:8]} ({below[:12]})", members[:members.index(own)]
        return tip, f"`{BRANCH}` at {tip[:12]}", members

    def _contained(self, pr: PR, prs: list[PR]) -> list[str]:
        """(c): each open PR whose head this head contains and that the sync wouldn't admit, so that it holds this PR behind it,
        with the fix.  The sync's own admission judges it (`Queue.entries` over this head, taken as marked, and the PRs it
        contains): not marked ready at that head, ejected there, held, with no owner, or containing such a PR itself, unless this
        PR carries its change (`Queue._carried`), so such a mark is refused here rather than taken and left for the next sync
        to say "contains #N, which isn't admitted"."""
        main = "refs/queue/preflight/main"
        heads = {p.head: p.number for p in prs}
        inside = set(contained(self.repo, main, pr.head, heads)) - {pr.number}
        if not inside:
            return []
        rules = rules_at(self.repo, main) or Rules()
        labels = StoreLabels(self.store, log=lambda s: print(s, file=sys.stderr), since=rules.marked_after)
        q = Queue(self.repo, remote=self.remote, github=None, ready=lambda n, h: (n, h) == (pr.number, pr.head) or labels.ready(n, h),
                  store=self.store, rules=rules, log=self.log, before_cutoff=labels.before_cutoff, marked_by=labels.marked_by,
                  marked_at=labels.marked_at, prefetch=labels.prefetch)
        try:
            entries = q.entries([pr, *(p for p in prs if p.number in inside)], {"ejected": {}}, base=main)
        except QueueError as e:
            self.log(f"research queue: preflight: can't tell whether the open PRs this head contains are admitted ({e})")
            return []

        def why(n: int) -> str:
            e, lab = entries[n], labels._newest.get((n, entries[n].pr.head))
            if lab is not None and str(lab.value).lower() == "false" and lab.ref:
                return f"ejected at {e.pr.head[:8]} by {lab.by}: {lab.ref}"
            return f"not marked ready at {e.pr.head[:8]}" if e.why.startswith("not marked ready") else e.why

        return [f"it contains #{n}'s head {entries[n].pr.head[:12]}, and #{n} isn't admitted ({why(n)}), so drive holds this PR "
                f"behind it: mark #{n} ready first, so it lands first, or drop #{n}'s commits from this branch (if this PR carries "
                f"#{n}'s change, #{n}'s owner withdraws its mark: `research data label pr:{n}@{entries[n].pr.head[:12]} ready false "
                f"--by OWNER`)" for n in entries[pr.number].after if entries[n].state != "ready"]

    def _gates(self, pr: PR, notes: list[str]) -> list[str]:
        """(d): what drive.toml's `landing_gates` at `main` refuse (`gate_refusal`).  A gate that can't judge is a note: the sync
        that would merge the head onto `next` runs the gates again, and leaves out a head they don't pass."""
        main = _out(self.repo, "rev-parse", "refs/queue/preflight/main")
        try:
            gates = gates_at(self.repo, main)
        except QueueError as e:
            notes.append(f"no landing gate run ({e})")
            return []
        refused, unjudged = gate_refusal(self.repo, main, pr.head, gates)
        notes += unjudged
        return refused

    def _merged(self, pr: PR, base: str, onto: str, notes: list[str], members: list[Member] | None = None) -> list[str]:
        """(b) and (e), in a temporary worktree at the merge of the head onto ``base``."""
        r = _git(self.repo, "merge-tree", "--write-tree", "--no-messages", base, pr.head, check=False)
        if r.returncode:
            notes.append(f"the merged tree's checks not run: `git merge-tree` exited {r.returncode} ({(r.stderr or r.stdout).strip()[-200:]})")
            return []
        tree = r.stdout.split()[0]
        commit = _out(self.repo, "commit-tree", tree, "-p", base, "-p", pr.head, "-m", f"preflight: #{pr.number} onto {base[:12]}")
        fixtures: dict[str, list[str]] = {}
        with merged_worktree(self.repo, commit, self.root) as wt:
            graph, lean_failed, lean_deadline = self._lean(wt, base, tree, notes)
            deadline = self.clock() + self.budget_s
            failed = self._checks(wt, base, tree, None, notes, deadline, fixtures)
            if not any(failed.values()) and not lean_failed:
                return []
            _git(wt, "checkout", "-q", "--detach", "--force", base)
            _git(wt, "clean", "-qffdx", "-e", "/.venv")
            again = (self._checks(wt, base, tree, {k for k, v in failed.items() if v}, [], deadline, fixtures)
                     if any(failed.values()) else {})
            lean_again = self._lean_at_base(wt, lean_failed, lean_deadline) if lean_failed else {}
        out = []
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

    def _lean(self, wt: Path, base: str, tree: str, notes: list[str]) -> tuple[LeanGraph, dict[str, str], float]:
        """(e): the modules of the merged tree in ``wt`` that the change from ``base`` touches and every module importing one, transitively
        (`lean_graph`), built by ``lean`` (`LakeBuild`) within `lean_s`: the graph, {file: its first Lean error} of each that failed,
        and the step's deadline.  Past `lean_modules` modules, or when the build can't run, a note and nothing failed."""
        deadline = self.clock() + self.lean_s
        if self.lean is None:
            return LeanGraph(), {}, deadline
        changed = _out(self.repo, "diff", "--name-only", "--no-renames", base, tree).split()
        graph = lean_graph(wt, _out(wt, "ls-files").splitlines(), changed)
        if not graph.targets:
            return graph, {}, deadline
        if len(graph.targets) > self.lean_modules:
            notes.append(f"Lean: not built: {len(graph.targets)} modules are or import what it changes, over the "
                         f"{self.lean_modules} the preflight builds; drive's Lean check of `{BRANCH}` builds them")
            return graph, {}, deadline
        res = self.lean(wt, sorted(graph.targets), deadline - self.clock())
        if isinstance(res, str):
            notes.append(f"Lean: {len(graph.targets)} module(s) not built: {res}")
            return graph, {}, deadline
        notes += [f"Lean: {n}" for n in res.notes]
        notes.append(f"Lean: built {len(graph.targets)} module(s), {len(res.failed)} failed")
        return graph, res.failed, deadline

    def _lean_at_base(self, wt: Path, failed: dict[str, str], deadline: float) -> dict[str, str] | str:
        """Which of the files that ``failed`` on the merged tree fail on `next` too (checked out in ``wt``), or why that couldn't be
        built: a file `next` lacks fails only on the merge."""
        there = sorted(f for f in failed if (wt / f).is_file())
        if not there:
            return {}
        res = self.lean(wt, there, deadline - self.clock())
        return res if isinstance(res, str) else res.failed

    def _lean_finding(self, f: str, line: str, graph: LeanGraph, members: list[Member], own: bool,
                      cache: dict[str, set[str]]) -> str:
        """The refusal for ``f``, which fails to build on the merged tree and not on `next`, naming the PR on `next` that owns it
        (`touched_by`), or, when it is the head's own (``own``) or no member's, those that changed what it imports."""
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
                    f"(`python3 tools/verity/check/pins.py --regen`); when the head alone is fresh, the other side is a PR on "
                    f"`next`: regenerate once it lands")
        if kind == "tests":
            return (f"the invariant tests in {where} fail on the merged tree: {shown}: reproduce on `git merge origin/next` in a "
                    f"scratch worktree, fix the branch, push and mark the new head")
        return (f"`pytest --collect-only` in {where} fails on the merged tree: {shown}: a test reads what another queued PR "
                f"changes or what only an unlanded PR adds; make it read what `main` and `next` hold, or wait for that PR")

    def _run(self, argv: list[str], cwd: Path, deadline: float) -> tuple[subprocess.CompletedProcess | None, str]:
        """``argv`` in ``cwd`` with what's left of the budget, outside the caller's environment (`quick_tier.CALLER_ENV`,
        pytest's own variables); and why it didn't finish."""
        left = deadline - self.clock()
        if left <= 0:
            return None, "the budget was spent before it started"
        env = {k: v for k, v in os.environ.items() if k not in quick_tier.CALLER_ENV and not k.startswith("PYTEST_")}
        env["COLUMNS"] = "300"                                         # pytest's summary lines, whole
        try:
            return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, errors="replace", timeout=left), ""
        except subprocess.TimeoutExpired:
            return None, f"still running when the {self.budget_s:g} s budget ran out"
        except OSError as e:
            return None, f"{argv[0]} didn't start ({e})"

    def _fetch_fixtures(self, wt: Path, paths: list[str], deadline: float) -> tuple[set[str], str]:
        """The fetched registry entries ``paths`` brought into ``wt`` by `research data fetch-fixtures` (`suites.py`'s fetch, every
        byte checked against its id) through ``store``, whose blobs stay there for the next preflight on this machine, within
        `PREFLIGHT_FETCH_S` of what's left by ``deadline``: the paths still missing, and why."""
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
        """{check: {what failed: its line}} for the checks of (b) on the tree checked out in ``wt`` (``only`` those named), each
        that ran by ``deadline``; why each other didn't run goes to ``notes``.  ``fixtures`` gets each collected suite's fetched
        fixtures on the first pass (``only`` None), and a later pass fetches them again."""
        want = (lambda name: True) if only is None else (lambda name: name in only)
        out: dict[str, dict[str, str]] = {}

        def got(name: str, res: dict[str, str] | str) -> None:
            if isinstance(res, str):
                notes.append(f"{name} not checked: {res}")
            else:
                out[name] = res

        if want("pins"):
            if (wt / "tools/verity/check/pins.py").is_file():
                r, why = self._run([sys.executable, "tools/verity/check/pins.py", "--stale"], wt, deadline)
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
            if any(f.startswith(r) or f"/{r}" in f for f in changed for r in reach):
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


class MarkOnly:
    """`ready --mark`: the owner's word alone, with no quick tier, so marking needs no machine and no wait; the after-landing
    run tests the head.  A ready post with no mark behind it admits nothing, so the mark is what an owner gives."""

    def __call__(self, pr: PR) -> None:
        return None


#: the Lean fast path's result kind (`lean_changed.py --records`), and the Lean check's step in a check's result
FAST_KIND, LEAN_AUDIT_STEP = "lean-fast/v1", "lean-audit"
#: the fast path's command, which `RecordsGate` names for each package it can't vouch for
RECORDS_COMMAND = ("research run --on vy-nebius-cpu-1 --pool research --project verity --source . --cwd clone --declared-output "
                   "'lean-audit/*' -- python3 tools/verity/lean/lean_changed.py --records {pkg} --update")
#: where a records run keeps, under its `--out`, the record `--update` wrote when it differs from the commit's lock
RECORD_KEPT = "record/" + LEAN_LOCK
#: a requirer's report in a records run's files (`lean_changed.reports`: `requirer-<path, each run of other characters `_`>/`)
REQUIRER_AUDIT = re.compile(r"(?:^|/)requirer-([^/]+)/audit\.json$")
#: how many commits below a head `RecordsGate` follows one that only merged `main`, or only committed a record a run regenerated
RECORDS_WALK = 8
#: the attempts `RecordsGate` lists from the store's remote: those started after the oldest commit it may read, less the skew of a
#: machine's clock, and never more than the lookback before now (an older run the catalog here lacks isn't seen)
RECORDS_SKEW_S, RECORDS_LOOKBACK_S = 3600, 3 * 86400
#: the repository's paths besides the packages' own that decide a records run's verdict: the toolchain they may share, and the
#: checker `RECORDS_COMMAND` runs (`lean_changed.py`, with the `lean_audit.py`, `check.py`, `merge.py` and `sandbox.sh` it loads,
#: and the `common.py` the audit's key counts among its files; `lean_slot.py` only takes a slot)
RECORDS_SHARED = ("lean-toolchain", "tools/verity/lean", "tools/verity/check/common.py")
#: the most commits with runs in the window, the newest first, that `RecordsGate` reads in its search for a run on the same inputs
#: (a safety bound): those this clone lacks are fetched by sha, `RECORDS_FETCH_CHUNK` a `git fetch`, a chunk that fails split in
#: halves until the commits the remote won't serve stand alone, in `RECORDS_FETCH_CALLS` fetches at most; one that can't be read
#: vouches for nothing
RECORDS_FETCH, RECORDS_FETCH_CHUNK, RECORDS_FETCH_CALLS = 1000, 200, 48
#: the files of a package that a records run doesn't read, and so leaves out of its inputs (`RecordsGate._differs`): those in a
#: `tests/` or `test/` directory of it, Markdown and READMEs; everything else stays, as a build may read any file
RECORDS_UNREAD_DIRS = frozenset({"tests", "test"})
_SHA = re.compile(r"[0-9a-f]{40}")


def _slug(pkg: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", pkg)


def _blob(repo: Path, commit: str, path: str) -> bytes | None:
    r = subprocess.run(["git", "cat-file", "blob", f"{commit}:{path}"], cwd=repo, capture_output=True, env=git_env(), check=False)
    return r.stdout if r.returncode == 0 else None


@dataclass
class RecordsRun:
    """A records run (`lean_changed.py --records PKG`) of one commit, as `research run` published it: its result (`FAST_KIND`:
    PKG, `validation.status`, `path`: `slot`, `dev-cache` or `reused`) and its run files ({path: sha256}), which hold check.py's
    `audit.json` for PKG and for each requirer (`requirer-<path>/`) and, when `--update` wrote a record that differs from the
    commit's lock, `record/lean-audit.json`."""
    id: str
    pkg: str
    passed: bool
    path: str
    files: dict[str, str]
    art: str | None

    def kept_record(self) -> str | None:
        """The sha256 of the record `--update` wrote that differs from the commit's lock, or None when there's none."""
        return next((s for p, s in sorted(self.files.items()) if (p == RECORD_KEPT or p.endswith("/" + RECORD_KEPT))
                     and "requirer-" not in p), None)

    def audits(self) -> list[str]:
        """check.py's `audit.json` files in the run: PKG's first, then each requirer's."""
        top = [p for p in self.files if (p == "audit.json" or p.endswith("/audit.json")) and "requirer-" not in p
               and not re.search(r"(?:^|/)audit-[^/]*/", p)]
        return sorted(top) + sorted(p for p in self.files if REQUIRER_AUDIT.search(p))

    def reports_of(self, pkg: str) -> list[str]:
        """The per-package reports (`audit-<name>/report.json` beside an `audit.json`) that may be ``pkg``'s: a package audited
        beside PKG or a requirer, its `proved_in` one (`Security/Proofs` in a run of `Security`); its path is in the report."""
        each = (posixpath.join(posixpath.dirname(a), f"audit-{posixpath.basename(pkg)}", "report.json") for a in self.audits())
        return [p for p in each if p in self.files]


class RecordsGate:
    """`ready`'s records gate, before the preflight and the mark: a head whose change against `main` (from their merge base)
    touches a Lake package's lock (`lean-audit.json`) is refused unless the store holds, for each such package, a records run of
    that exact head in which the package passed on the head's own lock (as PKG, or as a requirer PKG's run audited), or a Lean
    check of the head (a result whose `lean-audit` step passed without regenerating that lock).  A records run that printed
    `AUDIT: FAIL` still writes a record, which reads as a pass, and a head marked on it fails a train's Lean check, about 50
    minutes later.

    Two heads are vouched for by the commit below them, followed `RECORDS_WALK` commits at most: one that commits exactly the
    record a records run of its one parent regenerated for a package that passed there (`--update`'s `record/lean-audit.json`),
    changing nothing but locks; and one that only merged `main` into a vouched commit, every file it changed being main's whole
    but a lock, which is main's or `tools/verity/lean/merge.py`'s merge of the two sides (the fast path reports `path: reused`
    for it: a candidate, which the train's Lean check rebuilds and regenerates where it differs).

    A package is also vouched for by such a pass at any other commit whose inputs for it are the head's (`_inputs`): every file of
    the package's directory, lock included, and of each in-repo Lake package it requires by path (its `lake-manifest.json`'s and
    `lakefile.toml`'s, transitively) or its lock names under `proved_in`, less those a records run doesn't read (in a `tests/` or
    `test/` directory, Markdown, READMEs: `RECORDS_UNREAD_DIRS`), the same blob there as at the head; and `RECORDS_SHARED` (the
    root `lean-toolchain` and the checker: `tools/verity/lean/` and `tools/verity/check/common.py`) the same whole, so a pass
    under another checker vouches for nothing.  A PR that carries another PR's Lean change needs no run of its own for the
    packages that change passed in.  The candidates are the commits of the attempts in the window read, the newest first
    (`RECORDS_FETCH` of them, fetched by sha when this clone lacks them); a commit that can't be read vouches for nothing.

    The store is read from its remote (`refresh(attempts_only=True, since=...)`, the attempts since the oldest commit the walk may
    read or the newest that changed those inputs was written, and never more than `RECORDS_LOOKBACK_S` back); a store with no
    usable remote, or one that can't be listed, refuses: the gate never fails open.  `--no-preflight` doesn't skip it: the lander
    marks heads whose locks were checked like any other's."""

    def __init__(self, repo: Path, remote: str, store, *, log: Callable[[str], None] = print,
                 now: Callable[[], float] = time.time, lock_merge: Path | None = None):
        self.repo, self.remote, self.store, self.log, self.now = Path(repo), remote, store, log, now
        self.lock_merge = lock_merge or self.repo / "tools" / "verity" / "lean" / "merge.py"
        self._runs: dict[str, tuple[list[RecordsRun], list[str]]] = {}
        self._merges: dict[str, bool] = {}
        self._since = ""
        self._candidates: list[str] | None = None
        self._readable: list[str] | None = None
        self.notes: list[str] = []

    def __call__(self, pr: PR) -> str | None:
        """None when the head may be marked; else the refusal, naming the command to start for each package."""
        try:
            locks = self._locks(pr)
        except QueueError as e:
            return (f"the records gate can't tell which Lean locks it changes ({e}), so it refuses: it never fails open. Run "
                    f"this again once the remote can be read")
        if not locks:
            return None
        inputs: dict[str, list[str]] = {}
        for pkg in locks:
            try:
                inputs[pkg] = self._inputs(pr.head, pkg)
            except (QueueError, ValueError, TypeError, AttributeError) as e:   # no search on the same inputs for it; the rest stands
                self.notes.append(f"{pkg}'s inputs at {pr.head[:12]} can't be read ({type(e).__name__}: {e})")
        try:
            self._read(pr.head, sorted({p for ps in inputs.values() for p in ps}))
        except Exception as e:  # noqa: BLE001  (no store, no remote, a failed listing: refuse, never answer from what's here)
            return (f"it changes the Lean lock of {', '.join(locks)}, and the store's records runs can't be read "
                    f"({type(e).__name__}: {e}), so the records gate refuses: it never fails open. Run this where the store's remote "
                    f"is usable (R2_ENDPOINT and R2_BUCKET, or ~/.research/store.toml)")
        missing = []
        for pkg in locks:
            if how := self._vouch(pr.head, pkg, 0) or (pkg in inputs and self._same_inputs(pr.head, pkg, inputs[pkg])):
                self.log(f"research queue: records gate: {pkg}: {how}")
            else:
                missing.append(pkg)
        if not missing:
            return None
        lines = []
        for pkg in missing:
            lines += self._failures(pr.head, pkg)
            lines.append(f"start `{RECORDS_COMMAND.format(pkg=pkg)}` and mark once it passed; when its run files hold "
                         f"`lean-audit/{RECORD_KEPT}`, commit exactly that over {pkg}/{LEAN_LOCK} and mark that commit, which "
                         f"needs no run of its own")
        notes = f" ({'; '.join(self.notes[:3])})" if self.notes else ""
        return (f"it changes the Lean lock of {', '.join(missing)} against `main`, and the store holds no records run of "
                f"{pr.head[:12]} in which {'it' if len(missing) == 1 else 'each'} passed on the head's own lock, nor a Lean check of "
                f"it, nor such a run below a head that only merged `main` or committed the record a run regenerated, nor at a "
                f"commit whose inputs for it are the head's{notes}:\n  - "
                + "\n  - ".join(lines) + "\n  (`--no-preflight` doesn't skip this gate)")

    def _locks(self, pr: PR) -> dict[str, str]:
        """{package: its lock's path} for each lock the head's change against `main` touches and the head still has."""
        _git(self.repo, *STALL, "fetch", "-q", self.remote, "+refs/heads/main:refs/queue/records/main",
             f"+refs/pull/{pr.number}/head:refs/queue/records/pr")
        if _git(self.repo, "cat-file", "-e", f"{pr.head}^{{commit}}", check=False).returncode:
            raise QueueError(f"{pr.head[:12]} isn't on {self.remote} (pushed again since?)")
        files = _out(self.repo, "diff", "--name-only", "--no-renames", f"refs/queue/records/main...{pr.head}").splitlines()
        return {posixpath.dirname(f) or ".": f for f in sorted(files) if posixpath.basename(f) == LEAN_LOCK
                and _git(self.repo, "cat-file", "-e", f"{pr.head}:{f}", check=False).returncode == 0}

    def _read(self, head: str, inputs: list[str]) -> None:
        """Bring this store's attempts up to date from its remote, from the oldest commit the walk may read or the write of the
        newest that changed ``inputs`` (an author's date, which a rebase keeps), whichever is older."""
        if self.store.remote is None:
            raise QueueError("this machine's store has no remote, so the runs other machines published aren't here")
        times = [int(t) for t in _out(self.repo, "log", "--first-parent", f"-{RECORDS_WALK + 1}", "--format=%ct", head).split()]
        if inputs:
            times += [int(t) for t in _out(self.repo, "log", "-1", "--format=%at", head, "--", *inputs).split()]
        start = max(min(times) - RECORDS_SKEW_S, self.now() - RECORDS_LOOKBACK_S)
        self._since = time.strftime("r%Y%m%d-%H%M%S", time.gmtime(start))
        rep = self.store.refresh(attempts_only=True, since=self._since)
        if errs := rep.get("errors"):
            self.notes.append(f"{len(errs)} attempt(s) on the remote couldn't be read, e.g. {errs[0]}")

    def _inputs(self, head: str, pkg: str) -> list[str]:
        """The paths whose files decide ``pkg``'s records at ``head``: the package, each in-repo Lake package it requires by
        path (its `lake-manifest.json`'s and `lakefile.toml`'s, transitively) or its lock names under `proved_in`, and
        `RECORDS_SHARED`.  Raises on a file it can't parse or a path outside the repository."""
        seen: list[str] = []
        todo = [pkg]
        while todo:
            p = todo.pop()
            if p in seen:
                continue
            seen.append(p)
            at = (lambda f: f) if p == "." else (lambda f: f"{p}/{f}")
            named: list = []
            if (b := _blob(self.repo, head, at("lake-manifest.json"))) is not None:
                named += [d.get("dir") for d in json.loads(b).get("packages") or [] if d.get("type") == "path"]
            if (b := _blob(self.repo, head, at("lakefile.toml"))) is not None:
                named += [r.get("path") for r in tomllib.loads(b.decode()).get("require") or [] if "path" in r]
            if (b := _blob(self.repo, head, at(LEAN_LOCK))) is not None:
                named += list(json.loads(b).get("proved_in") or [])
            for d in named:
                q = posixpath.normpath(posixpath.join(p, d)) if isinstance(d, str) else ""
                if not q or posixpath.isabs(q) or q == ".." or q.startswith("../"):
                    raise QueueError(f"{p} names {d!r}, which isn't a package of this repository")
                todo.append(q)
        return sorted({*seen, *RECORDS_SHARED})

    def _present(self, commits: list[str]) -> set[str]:
        """The ``commits`` this clone can read, by one `git cat-file --batch-check`."""
        r = subprocess.run(["git", "cat-file", "--batch-check=%(objecttype)"], cwd=self.repo, capture_output=True, text=True,
                           input="".join(f"{c}^{{commit}}\n" for c in commits), env=git_env(), check=False)
        lines = r.stdout.splitlines()
        if r.returncode or len(lines) != len(commits):
            raise QueueError(f"git cat-file --batch-check: {r.stderr.strip()[-300:]}")
        return {c for c, ln in zip(commits, lines) if ln.split()[-1] == "commit"}

    def _commits(self) -> list[str]:
        """The clean commits of the attempts since the window's start, the newest first, `RECORDS_FETCH` at most."""
        if self._candidates is None:
            rows = self.store.index.sql(
                "SELECT source_commit, MAX(id) FROM attempts WHERE id > ? AND source_dirty = 0 AND source_commit IS NOT NULL "
                "GROUP BY source_commit ORDER BY 2 DESC", (self._since,))
            self._candidates = [c for c, _ in rows if _SHA.fullmatch(str(c))][:RECORDS_FETCH]
        return self._candidates

    def _readable_commits(self) -> list[str]:
        """`_commits` this clone can read once it fetched those it lacks from the remote by sha, in `RECORDS_FETCH_CHUNK`s,
        a chunk that fails split in halves until the commits the remote won't serve stand alone, `RECORDS_FETCH_CALLS` fetches
        at most."""
        if self._readable is None:
            cands = self._commits()
            have = self._present(cands)
            lacking = [c for c in cands if c not in have]
            todo = [lacking[i:i + RECORDS_FETCH_CHUNK] for i in range(0, len(lacking), RECORDS_FETCH_CHUNK)]
            calls = 0
            while todo and calls < RECORDS_FETCH_CALLS:
                part = todo.pop(0)
                calls += 1
                if _git(self.repo, *STALL, "fetch", "-q", self.remote, *part, check=False).returncode and len(part) > 1:
                    todo[:0] = [part[:len(part) // 2], part[len(part) // 2:]]
            if lacking:
                have = self._present(cands)
            if unread := len(cands) - len(have):
                self.notes.append(f"{unread} of {len(cands)} commit(s) with runs can't be read here or from {self.remote}")
            self._readable = [c for c in cands if c in have]
        return self._readable

    def _differs(self, head: str, commits: list[str], inputs: list[str]) -> set[str]:
        """The ``commits`` whose ``inputs`` aren't the head's, by one `git diff-tree --stdin`: a file of `RECORDS_SHARED` that
        differs, or one of a package that a records run reads (`RECORDS_UNREAD_DIRS`, Markdown and READMEs aside)."""
        if not commits:
            return set()
        r = subprocess.run(["git", "diff-tree", "--stdin", "--always", "-r", "-z", "--name-only", "--no-renames", "--", *inputs],
                           cwd=self.repo, capture_output=True, input="".join(f"{c} {head}\n" for c in commits).encode(),
                           env=git_env(), check=False)
        if r.returncode:
            raise QueueError(f"git diff-tree --stdin: {r.stderr.decode(errors='replace').strip()[-300:]}")
        shared = [p for p in inputs if p in RECORDS_SHARED]
        pkgs = sorted((p for p in inputs if p not in RECORDS_SHARED), key=len, reverse=True)
        out, at, n = set(), None, 0
        for tok in r.stdout.decode(errors="surrogateescape").split("\0"):
            if n < len(commits) and tok == commits[n]:          # `--always`: each pair's header, in order, its paths after it
                at, n = tok, n + 1
                continue
            if not tok or at in out:
                continue
            if at is None:
                raise QueueError(f"git diff-tree --stdin printed {tok!r} before any commit")
            if any(tok == s or tok.startswith(s + "/") for s in shared):
                out.add(at)
                continue
            p = next((p for p in pkgs if p == "." or tok.startswith(p + "/")), None)
            rel = tok if p in (None, ".") else tok[len(p) + 1:]
            name = posixpath.basename(rel)
            if not (RECORDS_UNREAD_DIRS & set(rel.split("/")[:-1]) or name.lower().endswith(".md") or name.startswith("README")):
                out.add(at)
        if n != len(commits):                                    # a pair it skipped (an object it couldn't read) is no match
            raise QueueError(f"git diff-tree --stdin compared {n} of {len(commits)} commits")
        return out

    def _same_inputs(self, head: str, pkg: str, inputs: list[str]) -> str | None:
        """Why a pass of ``pkg`` at another commit whose ``inputs`` are the head's (`_differs`) vouches for it, or None."""
        try:
            cands = [c for c in self._readable_commits() if c != head]
            differ = self._differs(head, cands, inputs)
        except Exception as e:  # noqa: BLE001  (a search that can't finish vouches for nothing)
            self.notes.append(f"{pkg}'s inputs can't be compared ({type(e).__name__}: {e})")
            return None
        for c in cands:
            if c not in differ and (how := self._passed(c, pkg)):
                return f"a {how[0]} of {c[:12]} ({how[1]}) with identical inputs"
        return None

    def _at(self, commit: str) -> tuple[list[RecordsRun], list[str]]:
        """(the records runs of ``commit``, the ids of its attempts whose result's `lean-audit` step passed), newest first."""
        if commit in self._runs:
            return self._runs[commit]
        runs, checks = [], []
        rows = self.store.index.sql("SELECT id, manifest_json FROM attempts WHERE source_commit = ? AND source_dirty = 0 "
                                    "ORDER BY start_utc DESC, id DESC", (commit,))
        for rid, doc in rows:
            out = (json.loads(doc).get("outputs") if doc else None) or {}
            try:
                meta = self.store.get_manifest(out["result"]).meta if out.get("result") else {}
                if meta.get("kind") == FAST_KIND:
                    fp = meta.get("workload_fingerprint") or {}
                    files = {f["path"]: f["sha256"] for f in self.store.listing(out["run_files"])} if out.get("run_files") else {}
                    runs.append(RecordsRun(rid, str(fp.get("package")), (meta.get("validation") or {}).get("status") == "passed",
                                           str(fp.get("path") or "slot"), files, out.get("run_files")))
                elif any(isinstance(s, dict) and s.get("step") == LEAN_AUDIT_STEP and s.get("status") == "passed"
                         for s in meta.get("steps") or []):
                    checks.append(rid)
            except Exception as e:  # noqa: BLE001  (a run whose records can't be read vouches for nothing; say so)
                self.notes.append(f"{rid}'s result or files can't be read ({type(e).__name__})")
        self._runs[commit] = (runs, checks)
        return runs, checks

    def _audited(self, run: RecordsRun, pkg: str) -> tuple[bool, str | None]:
        """(whether ``pkg`` passed its audit in ``run``, the first line of its first failure when it failed).  A run that passed
        passed every package check.py audited in it: PKG (on the `reused` path too, where nothing ran), each requirer, and
        each package audited beside one, read from its own report.  A failed run's are read from its `audit.json` files and
        those reports."""
        if run.passed:
            if pkg == run.pkg or any(m.group(1) == _slug(pkg) for p in run.files if (m := REQUIRER_AUDIT.search(p))):
                return True, None
            return any(self._reported(run, p, pkg) == (True, None) for p in run.reports_of(pkg)), None
        for p in [*run.audits(), *run.reports_of(pkg)]:
            got = self._reported(run, p, pkg)
            if got is not None:
                return got
        return False, (f"records run {run.id} failed with no report of {pkg}" if pkg == run.pkg else None)

    def _reported(self, run: RecordsRun, path: str, pkg: str) -> tuple[bool, str | None] | None:
        """What run file ``path`` (an `audit.json`, or one package's `report.json`) says of ``pkg``; None when it doesn't name it."""
        try:
            with tempfile.TemporaryDirectory() as d:
                doc = json.loads((self.store.fetch(run.art, Path(d), paths=[path]) / path).read_text())
        except Exception as e:  # noqa: BLE001
            return False, f"records run {run.id}'s {path} can't be read ({type(e).__name__})"
        reps = doc.get("packages") if path.endswith("/audit.json") or path == "audit.json" else [doc]
        rep = next((r for r in reps or [] if isinstance(r, dict) and (str(r.get("package")) == pkg
                                                                     or str(r.get("package")).endswith("/" + pkg))), None)
        if rep is None:
            return None
        if not rep.get("failures"):
            return True, None
        return False, f"records run {run.id} failed {pkg}: {str(rep['failures'][0]).strip().splitlines()[0][:300]}"

    def _vouch(self, commit: str, pkg: str, depth: int) -> str | None:
        """Why ``pkg``'s lock at ``commit`` passed its records, or None."""
        if how := self._passed(commit, pkg):
            kind, rid, path = how
            if kind == "Lean check":
                return f"the Lean check {rid} of {commit[:12]} passed it"
            return (f"records run {rid} of {commit[:12]} passed it" +
                    (" (`path: reused`: its records passed on the same inputs)" if path == "reused" else ""))
        if depth >= RECORDS_WALK:
            return None
        parents = _out(self.repo, "rev-list", "--parents", "-n", "1", commit).split()[1:]
        if len(parents) == 2 and self._main_merge(commit, *parents):
            if how := self._vouch(parents[0], pkg, depth + 1):
                return f"{commit[:12]} only merges `main`'s {parents[1][:12]} into {parents[0][:12]}, where {how}"
            return None
        if len(parents) != 1:
            return None
        below = parents[0]
        files = _out(self.repo, "diff", "--name-only", "--no-renames", below, commit).splitlines()
        if not files or any(posixpath.basename(f) != LEAN_LOCK for f in files):
            return None
        lock = f"{pkg}/{LEAN_LOCK}" if pkg != "." else LEAN_LOCK
        if lock not in files:
            how = self._vouch(below, pkg, depth + 1)
            return how and f"{commit[:12]} changes only other packages' locks over {below[:12]}, where {how}"
        sha = hashlib.sha256(_blob(self.repo, commit, lock) or b"").hexdigest()
        for run in self._at(below)[0]:
            if run.pkg == pkg and run.kept_record() == sha and self._audited(run, pkg)[0]:
                return f"{commit[:12]} commits the record records run {run.id} of {below[:12]} regenerated, where {pkg} passed"
        return None

    def _passed(self, commit: str, pkg: str) -> tuple[str, str, str] | None:
        """(`records run` or `Lean check`, its id, a records run's `path`) of the first run of ``commit`` that passed ``pkg`` on
        the commit's own lock, or None."""
        runs, checks = self._at(commit)
        for run in runs:
            if self._audited(run, pkg)[0] and not (run.pkg == pkg and run.kept_record()):
                return "records run", run.id, run.path
        for rid in checks:
            if self._checked(rid, pkg):
                return "Lean check", rid, ""
        return None

    def _checked(self, rid: str, pkg: str) -> bool:
        """Whether Lean check ``rid`` passed ``pkg`` on the commit's own lock: it regenerated no record of it."""
        from .merge import regenerated_records

        try:
            regen = regenerated_records(self.store, rid)
        except Exception as e:  # noqa: BLE001  (`Unreadable`: vouches for nothing)
            self.notes.append(f"the Lean check {rid}'s report can't be read ({type(e).__name__})")
            return False
        return regen is not None and pkg not in regen

    def _main_merge(self, commit: str, ours: str, theirs: str) -> bool:
        """Whether ``commit`` only merges `main`'s ``theirs`` into ``ours``: each file it changes over ``ours`` is main's whole, a
        lock aside, which may be `merge.py`'s merge of the two sides instead."""
        if commit not in self._merges:
            ok = _git(self.repo, "merge-base", "--is-ancestor", theirs, "refs/queue/records/main", check=False).returncode == 0
            base = _out(self.repo, "merge-base", ours, theirs) if ok else ""
            for f in _out(self.repo, "diff", "--name-only", "--no-renames", ours, commit).splitlines() if ok else []:
                mine = _blob(self.repo, commit, f)
                if mine == _blob(self.repo, theirs, f):
                    continue
                if not (posixpath.basename(f) == LEAN_LOCK and mine is not None and self._merged_lock(base, ours, theirs, f, mine)):
                    ok = False
                    break
            self._merges[commit] = ok
        return self._merges[commit]

    def _merged_lock(self, base: str, ours: str, theirs: str, path: str, mine: bytes) -> bool:
        """Whether ``mine`` is `tools/verity/lean/merge.py`'s merge (this checkout's) of ``path`` at ``base``, ``ours``, ``theirs``."""
        tool = self.lock_merge
        if not tool.is_file():
            return False
        with tempfile.TemporaryDirectory() as d:
            sides = []
            for name, c in (("base", base), ("ours", ours), ("theirs", theirs)):
                (p := Path(d) / name).write_bytes(_blob(self.repo, c, path) or b"")
                sides.append(str(p))
            try:
                r = subprocess.run([sys.executable, str(tool), *sides, path], cwd=self.repo, capture_output=True, timeout=300,
                                   check=False)
                return r.returncode == 0 and json.loads(Path(sides[1]).read_text()) == json.loads(mine)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                return False

    def _failures(self, head: str, pkg: str) -> list[str]:
        """What the records runs of ``head`` say against ``pkg``: the newest failing run's first failure line, or a passing run whose
        `--update` wrote a record the head doesn't hold."""
        out = []
        for run in self._at(head)[0]:
            ok, line = self._audited(run, pkg)
            if ok and run.pkg == pkg and run.kept_record():
                out.append(f"records run {run.id} of {head[:12]} passed {pkg} only with the record its `--update` wrote, which "
                           f"differs from the head's lock: commit its `{RECORD_KEPT}`")
                break
            if line:
                out.append(line)
                break
        return out


def ejection(store, pr: PR) -> str | None:
    """Why ``pr``'s current head was ejected, when the newest `ready` label on it is `research queue eject`'s `false` (its reason in
    `ref`), pulled from the store's remote first; else None, and None when that pull fails: this saves a revert, and the after-landing
    run checks the head."""
    try:
        labs = StoreLabels(store, log=lambda s: print(s, file=sys.stderr))._labels(pr.number, pr.head, vocab.READY)
    except QueueError as e:
        print(f"research queue ready: can't tell whether #{pr.number}'s head was ejected ({e}); the after-landing run still checks it",
              file=sys.stderr)
        return None
    last = labs[-1] if labs else None
    if last is None or str(last.value).lower() != "false" or not last.ref:
        return None
    return f"{last.ref} (ejected by {last.by} at {last.ts})"


#: how many times `ready --sha` reads GitHub's open PRs, and the seconds between reads, before refusing a head GitHub doesn't show
SHA_READS, SHA_WAIT_S = 6, 5.0


def off_main_note(pr: PR, prs: list[PR]) -> str | None:
    """What `ready` tells the owner of ``pr`` when it isn't based on `main` (drive admits only PRs that are), or None."""
    if pr.base == "main":
        return None
    parent = next((p for p in prs if p.branch == pr.base and p.number != pr.number), None)
    until = f"until #{parent.number} lands, one stack level per landing" if parent else "until it is"
    return (f"#{pr.number} is based on {based_on(pr, prs)}, not main: drive won't see it {until}. Retarget it to main (drive orders "
            f"stacked PRs itself: a PR containing another waits for it); retargeting keeps the head, so a ready mark at "
            f"{pr.head[:12]} counts from then on, and a new push needs a new mark as always")


def ready(github, store, number: int | None, by: str, tier: Callable[[PR], str | None], again: bool = False,
          preflight: Callable[[PR, list[PR]], str | None] | None = None, sha: str | None = None,
          sleep: Callable[[float], None] = time.sleep, as_json: bool = False,
          records: Callable[[PR], str | None] | None = None) -> int:
    """`research queue ready PR`: run the quick tier at the PR's current head (`tier`, a `RemoteQuickTier` or a `QuickTier`), and once
    it passes write `ready = true` on `pr:{n}@{that head}` and push it to the store's remote.  A head whose tree is that of another
    head marked ready (`same_tree_mark`) is marked without running the tier again.  A head drive or the lander ejected is refused
    unless ``again`` (a re-mark at an ejected head would fail a second train on the same line).  Before any of it,
    ``records`` (`RecordsGate`, which `--no-preflight` keeps) refuses a head that changes a Lean lock no records run passed, of
    it or of a commit with the same inputs for that package,
    then ``preflight`` (`Preflight`; None with `--no-preflight`) refuses a head that would break `next`, with the fix.  With ``sha``
    (`--sha`) it marks only that head: GitHub's head lags a push by seconds (`git push && research queue ready N --mark` can mark
    the old head), so it reads again, up to `SHA_READS` times, and refuses when GitHub still has another head.  A PR based
    on another branch is marked all the same, and told that drive won't see it until it is based on `main` (`off_main_note`).
    With ``as_json`` (`--json`) the rest goes to stderr, and stdout gets one object: pr, head, base, exit and warning (that note)."""
    with contextlib.redirect_stdout(sys.stderr) if as_json else contextlib.nullcontext():
        rc, pr, prs = _ready(github, store, number, by, tier, again, preflight, sha, sleep, records)
        note = off_main_note(pr, prs) if pr is not None else None
        if note:
            print(f"research queue ready: {note}", file=sys.stderr)
    if as_json:
        print(json.dumps({"pr": number, "head": pr and pr.head, "base": pr and pr.base, "exit": rc, "warning": note}))
    return rc


def _ready(github, store, number: int | None, by: str, tier: Callable[[PR], str | None], again: bool,
           preflight: Callable[[PR, list[PR]], str | None] | None, sha: str | None,
           sleep: Callable[[float], None], records: Callable[[PR], str | None] | None = None) -> tuple[int, PR | None, list[PR]]:
    if not sha and number is not None and (p := next((x for x in github.open_prs() if x.number == number), None)) and p.branch \
            and (tip := getattr(github, "branch_tip", lambda _b: None)(p.branch)):
        sha = tip                                           # a mark seconds after a push would preflight GitHub's old head
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
    if sha and not pr.head.startswith(sha):
        print(f"research queue ready: #{pr.number} is not marked: GitHub has its head at {pr.head[:12]}, not {sha[:12]}, after "
              f"{SHA_READS} reads {SHA_WAIT_S:g} s apart. Push {sha[:12]} to the PR's branch (or pass the head GitHub shows), then "
              "run this again", file=sys.stderr)
        return 1, pr, prs
    if not again and (why := ejection(store, pr)):
        print(f"research queue ready: #{pr.number} is not marked ready at {pr.head[:12]}: that head was ejected: {why}. Push the fix "
              f"and mark the new head; if the ejection was wrong (a machine fault), say so in the merge thread and pass --again",
              file=sys.stderr)
        return 1, pr, prs
    if records is not None and (why := records(pr)):
        print(f"research queue ready: #{pr.number} is not marked ready at {pr.head[:12]}: {why}", file=sys.stderr)
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
    print(f"research queue: the quick tier at #{pr.number}'s head {pr.head[:12]}: `suites.py --quick` on the suites it changes", flush=True)
    return _mark(store, pr, by, tier(pr), "ready"), pr, prs


def judge(github, store, ref: str, by: str, on: str | None = None, fetch: Callable[[Path], dict] | None = None,
          runs_root: Path | None = None, repo: Path | None = None, url: str | None = None,
          records: Callable[[PR], str | None] | None = None) -> int:
    """`research queue judge RUN`: what `ready --on` does once its run ends, for a quick-tier run that already ended (its waiting
    client gone): fetch its records (`research fetch`; a run another VM launched is adopted from its machine, ``on`` or each), judge
    them as `RemoteQuickTier` does, at the head the tier ran at, and mark the open PR at that head ready, or, when none is there now
    and ``repo`` (a clone of ``url``) is given, the one open PR whose head has that head's tree (`same_tree_mark`'s reason).
    Refused when the run can't be fetched, hasn't ended, isn't a quick tier, failed, or no open PR is at that head or tree now."""
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
    open_prs = github.open_prs()
    prs = [p for p in open_prs if p.head == head]
    if not prs and repo is not None and isinstance(head, str):
        if len(prs := same_tree_prs(repo, url, open_prs, head)) == 1:
            print(f"research queue: #{prs[0].number}'s head {prs[0].head[:12]} has the tree of {head[:12]}, where run {rid} ran the "
                  f"quick tier: the same files", flush=True)
    if len(prs) != 1:
        return refuse(f"run {rid} ran the quick tier at {str(head)[:12]}, " + (
            f"the head of each of {', '.join('#' + str(p.number) for p in prs)}: mark one with `research queue ready`" if prs else
            "which is no open PR's head now, nor its tree (pushed again since, or closed?)"))
    if records is not None and (why := records(prs[0])):
        return refuse(f"#{prs[0].number} is not marked ready at {prs[0].head[:12]}: {why}")
    return _mark(store, prs[0], by, judge_report(res, rid, machine, head, status), "judge")


def _mark(store, pr: PR, by: str, failed: str | None, command: str) -> int:
    """`ready = true` on `pr:{n}@{head}`, pushed to the store's remote, unless the quick tier ``failed`` there (the refusal)."""
    from .store.cli import _label_write_through

    if isinstance(failed, StillGoing):
        print(f"research queue {command}: #{pr.number} is not marked ready at {pr.head[:12]} yet. {failed}", file=sys.stderr)
        return waitcap.STILL_GOING
    if isinstance(failed, NotStarted):
        print(f"research queue {command}: #{pr.number} is not marked ready at {pr.head[:12]}: the quick tier never started: {failed}. "
              f"Nothing failed: run it again (`--on auto` takes the machine with a free check slot). A PR lands on an eyeball review, "
              f"and the quick tier isn't required: `ready N --mark --by LANE` marks it; to run its tests anyway: {LANE_RUN}", file=sys.stderr)
        return READY_NOT_STARTED
    if failed is not None:
        print(f"research queue {command}: #{pr.number} is not marked ready at {pr.head[:12]}: the quick tier failed: {failed}. Push a "
              f"fix, or run it again: passes are cached by input, so a re-run on the same cache runs only what failed", file=sys.stderr)
        return 1
    target = label_target(pr.number, pr.head)
    store.add_label(target, vocab.READY, True, by)
    said, lost = _label_write_through(store, target)
    print(f"research queue: #{pr.number} is ready at {pr.head[:12]} ({target}); {said}")
    return 1 if lost else 0


def review(q: Queue, store, labels: StoreLabels, pr: int | None, verdict: str | None, ref: str | None, by: str | None,
           sha: str | None = None, as_json: bool = False) -> int:
    """`research queue review`: without ``pr``, list `next`'s members above `main` that no land verdict at their head lands
    (`unreviewed`), bottom first, the order drive lands them in; with it, write ``verdict`` (`land` or `hold`, ``ref`` its note or
    reason) by ``by`` on the PR's head on `next` (or ``sha``, or its open head when it isn't on `next`), refusing its owner.  Without
    ``sha`` it refuses when GitHub has a head other than `next`'s, as the reviewer may have read either; a short ``sha`` is the
    known head or clone commit it starts."""
    from .store.cli import _label_write_through

    q._clone()
    q.fetch()
    main = _out(q.workdir, "rev-parse", "refs/remotes/origin/main")
    chain = [m for m in q.chain() if _git(q.workdir, "merge-base", "--is-ancestor", m.commit, main, check=False).returncode]
    opened = {p.number: p for p in q.github.open_prs()}
    if pr is None:
        labels.prefetch([(m.pr, m.head) for m in chain])
        rows = []
        for m in chain:
            if (why := unreviewed(labels.reviews(m.pr, m.head), review_owners(q, labels, m.pr, m.head, m.owner, opened))) is None:
                continue
            files = sorted(changed(q.workdir, f"{m.commit}^1", m.commit))
            rows.append({"pr": m.pr, "head": m.head, "owner": m.owner, "lean": lean_change(q.workdir, m.commit), "files": files,
                         "why": why})
        if as_json:
            print(json.dumps(rows, indent=1))
        else:
            for r in rows:
                shown = ", ".join(r["files"][:8]) + (f" (+{len(r['files']) - 8} more)" if len(r["files"]) > 8 else "")
                print(f"#{r['pr']} {r['head'][:12]} {r['owner']} waits for a Lean check: {'yes' if r['lean'] else 'no'}: "
                      f"{r['why']}\n    {shown}")
            print(f"research queue review: {len(rows)} of {len(chain)} member(s) of `{q.branch}` above `main` lack a land verdict")
        return 0
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
    store.add_label(target, vocab.REVIEWED, verdict, by, ref=ref or None)
    said, lost = _label_write_through(store, target)
    print(f"research queue: #{pr} reviewed `{verdict}` at {head[:12]} by {by} ({target}); {said}")
    return 1 if lost else 0


def review_owners(q: Queue, labels: StoreLabels, pr: int, head: str, owner: str | None,
                  opened: dict[int, PR] | None = None) -> set[str | None]:
    """Whose verdict on ``pr`` at ``head`` doesn't count (`unreviewed`): its owner (`PR.owner`, or its merge's `Owner`), that
    owner's Slack handle through the registry (`Queue.handle`), and who wrote its ready mark (`StoreLabels.ready_by`)."""
    p = (opened or {}).get(pr) or PR(pr, head, body=f"Lane {owner}")
    return {owner, q.handle(p), labels.ready_by(pr, head)}


def baseline(q: Queue, sha: str, by: str, steps: str | None = None) -> int:
    """`research queue baseline SHA --by ci [--steps STEP,...]`: record `main` at SHA as verified by the after-landing run
    (`drive["postland"]`), on the steps ci knows are green there (every after-landing step by default; with --steps, only those,
    as a red baseline run records: a later failure of another step reverts nothing), so the review landing lands without first running the after-landing run on `main` itself
    (`Driver._unbaselined`), and a later failure reverts only a landing above SHA.  It drops a bisection still open (it reverted
    nothing), and is refused while a post-landing run runs, for a SHA `main` doesn't hold, and off drive (`/data/infra/drive.sh
    queue baseline SHA --by ci` on the control pod)."""
    from .drive import AFTER_LANDING

    if _who(by) != "ci":
        raise QueueError(f"baseline: only ci records a green `main` (--by ci), not {by}")
    green = [x for x in (steps or "").split(",") if x] or list(AFTER_LANDING)
    if unknown := [x for x in green if x not in AFTER_LANDING]:
        raise QueueError(f"baseline: --steps {', '.join(unknown)}: not after-landing steps ({', '.join(AFTER_LANDING)})")
    q._clone()
    q.fetch()
    full = _git(q.workdir, "rev-parse", "--verify", "-q", f"{sha}^{{commit}}", check=False).stdout.strip()
    if not full or _git(q.workdir, "merge-base", "--is-ancestor", full, "refs/remotes/origin/main", check=False).returncode:
        raise QueueError(f"baseline: {sha} is not a commit of `main`")
    state = q.load_state()
    pl = state.setdefault("drive", {}).setdefault("postland", {"verified": full, "pending": [], "run": None})
    if pl.get("run"):
        raise QueueError(f"baseline: the post-landing run {' '.join(pl['run'].get('ids') or [])} is running; its end verifies "
                         "`main` or acts on it (a lost one, in 3 h): record a baseline once it has ended")
    if b := pl.pop("bisect", None):
        print(f"research queue: dropped the bisection of `main` {b['tip'][:12]}'s failed after-landing run; it reverted nothing")
    on = lambda a, b: not _git(q.workdir, "merge-base", "--is-ancestor", a, b, check=False).returncode  # noqa: E731
    pl["verified"] = full
    pl["verified_as"] = {"steps": green, "no_cache": None, "by": by,
                         "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    pl["pending"] = [p for p in pl.get("pending", []) if not on(p, full)]
    if pl.get("failed") and on(pl["failed"], full):
        del pl["failed"]
    q.save_state(state)
    print(f"research queue: `main` {full[:12]} recorded as verified by the after-landing run, by {by}: the review landing lands "
          f"on it; {len(pl['pending'])} landing(s) above it wait for their run")
    return 0


def broker_gh(top: Path) -> str:
    """The gh to open PRs with: the checkout's broker wrapper (`<git common dir>/verity-auth/bin/gh`, which trades the agent's
    identity for an installation token) when it has one, else gh on PATH."""
    common = _git(top, "rev-parse", "--path-format=absolute", "--git-common-dir", check=False).stdout.strip()
    wrapper = Path(common) / "verity-auth" / "bin" / "gh" if common else None
    return str(wrapper) if wrapper and wrapper.is_file() else (shutil.which("gh") or "gh")


def open_pr(top: Path, repo: str, head: str, title: str, body: str, *, base: str = "main", sha: str | None = None,
            draft: bool = False, gh: str | None = None, run=subprocess.run) -> str:
    """Open a PR for ``head``, a branch already pushed to origin, and return its URL.  Refuses (`QueueError`) when the branch
    isn't on origin, when ``sha`` is given and the branch isn't at it (the owner's pinned head), and when an open PR already has
    that head, so a relayed request can never open a duplicate or a moved head."""
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
    """`gh ARGS` in ``top`` (the broker's gh by default, `broker_gh`), returning its stdout; a failure raises, naming ``verb``."""
    gh = gh or broker_gh(top)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

    def call(*args: str) -> str:
        r = run([gh, *args], cwd=top, capture_output=True, text=True, env=env, timeout=120)
        if r.returncode:
            raise QueueError(f"{verb}: gh {args[0]} {args[1]}: {(r.stderr or r.stdout).strip()[-300:]}")
        return r.stdout

    return call


#: the markup a Cursor agent's PR tool keeps in a body: it rewrites only the text between these markers, so any text above them
#: (the body `open` gave the PR) stays, and its footer, the agent's link whose id `PR.owner` reads, follows them
CURSOR_BODY = ("<!-- CURSOR_AGENT_PR_BODY_BEGIN -->", "<!-- CURSOR_AGENT_PR_BODY_END -->")
CURSOR_FOOTER_RE = re.compile(r'<div><a href="https://cursor\.com/agents/bc-[^"]*cursor_ref=pr_footer')


def edited_body(old: str, new: str) -> str:
    """``new`` as the whole of a PR body that was ``old``: none of ``old``'s text stays, only the platform's markup, which is
    kept deliberately.  ``new`` goes between Cursor's markers when ``old`` has them, so the tool's next update replaces it
    rather than leaving it above, and ``old``'s footer follows it (else a PR keeps the body it was opened with above every
    later one).  Markers or a footer ``new`` brings along are dropped."""
    begin, end = CURSOR_BODY
    foot = next((old[m.start():].strip() for m in reversed(list(CURSOR_FOOTER_RE.finditer(old)))
                 if end not in old or m.start() > old.rfind(end)), "")
    if m := CURSOR_FOOTER_RE.search(new):
        new = new[:m.start()]
    text = new.replace(begin, "").replace(end, "").strip("\n")
    if begin in old and end in old:
        text = f"{begin}\n{text}\n{end}"
    return text + (f"\n\n{foot}" if foot else "") + "\n"


def edit_pr(top: Path, repo: str, number: int, body: str, *, dry_run: bool = False, gh: str | None = None,
            run=subprocess.run) -> tuple[str, str]:
    """Replace PR ``number``'s body with ``body`` outright (`edited_body`) and return its URL and the body it now has; with
    ``dry_run``, change nothing.  Refused (`QueueError`) when the PR's body names its owner on a `Lane NAME` line and the new
    one names another owner or none: an edit never moves a PR to another lane or out of the queue."""
    call = _gh_call("edit", top, gh, run)
    cur = json.loads(call("pr", "view", str(number), "--repo", repo, "--json", "body,url"))
    text = edited_body(cur.get("body") or "", body)
    now = PR(number, "", body=text).owner
    if (was := LANE_RE.search(cur.get("body") or "")) and _who(now or "") != _who(was.group(1)):
        raise QueueError(f"edit: #{number}'s new body names owner {now or 'none'}, and the PR's is {was.group(1)}: keep its "
                         f"`Lane {was.group(1)}` line; nothing edited")
    if not dry_run:
        call("pr", "edit", str(number), "--repo", repo, "--body", text)
    return cur.get("url") or f"#{number}", text


def _stats(a: argparse.Namespace) -> int:
    """`research queue stats` (`queue_stats`): each landed PR's head from `main`'s merges onto `next`, its ready mark from the
    store, the landing check's records fetched with `research status --also`."""
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
        # a landing's check has ended (--refresh); one name a fetch, since slot.json beside steps.jsonl fetched neither
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


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research queue", description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["status", "sync", "eject", "ready", "judge", "drive", "open", "edit", "stats",
                                            "checked-heads", "review", "baseline", "lean-check"])
    ap.add_argument("pr", nargs="?", help="eject, ready, review, edit: the PR number; judge: the quick tier's run id; baseline: the "
                    "commit of `main` ci knows is green; lean-check: the commit of `next` to Lean-check")
    ap.add_argument("--reason", help="eject: why, for the owner's notice")
    ap.add_argument("--steps", help="baseline: the after-landing steps green at SHA (default every one): a later failure of "
                    "another step reverts nothing")
    ap.add_argument("--by", default=os.environ.get("RESEARCH_LANE"),
                    help="eject, ready, judge: who ejects it, the handle its notice is posted as (default merge-queue, which no "
                         "one holds, so the notice fails) or says it's ready (no default: the owner's lane), "
                         "default $RESEARCH_LANE")
    ap.add_argument("--shadow", action="store_true", help=f"build `{SHADOW_BRANCH}`, never push `main`, send no notices")
    ap.add_argument("--dry-run", action="store_true", help="sync/drive: decide and report, push and launch nothing, tell no "
                                                           "one; open, edit: say what it would send, and send nothing")
    ap.add_argument("--workdir", type=Path, default=Path.home() / ".research" / "queue" / "verity", help="the queue's own clone")
    ap.add_argument("--remote", default=None, help="the repository to clone (default: this checkout's origin)")
    ap.add_argument("--repo", default=None, help="OWNER/NAME on GitHub (default: from the remote's URL)")
    ap.add_argument("--rules", type=Path, default=None, help=f"the admission rules (default: {RULES} on `main`)")
    ap.add_argument("--store", default=None, help="evidence store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--json", action="store_true", help="status: as JSON; ready: one JSON object on stdout (pr, head, base, exit, "
                    "warning), the rest on stderr")
    ap.add_argument("--on", metavar="MACHINE", help="ready: run the quick tier as a `research run` there, in a check slot (`auto` or "
                                                    "M1,M2: the one with a free check slot, as `check.py --on` picks), waiting for it "
                                                    f"{waitcap.WAIT_CAP_S} s at most (then exit {waitcap.STILL_GOING}, the run going "
                                                    "on: `judge RUN` marks the PR once it ends); judge: look "
                                                    "for a run another VM launched there only; lean-check: train.sh's MACHINES "
                                                    "(default drive.toml's `on` at `main`)")
    ap.add_argument("--local", action="store_true", help="ready: run the quick tier here")
    ap.add_argument("--mark", action="store_true", help="ready: mark it with no quick tier (the after-landing run tests it)")
    ap.add_argument("--again", action="store_true", help="ready: mark a head that was ejected (only when the ejection was wrong); "
                                                          "lean-check: rerun a commit whose recorded Lean check passed")
    ap.add_argument("--no-preflight", action="store_true",
                    help="ready: skip the preflight (the lander's): the head merged onto a fresh `next` must merge cleanly, contain "
                         "no open PR that isn't admitted, and pass `pins.py --stale`, the invariant tests (module strings, pins, "
                         "doc paths, pythonpath) and `pytest --collect-only` of the suites its change touches, in a temporary "
                         f"worktree, in about {PREFLIGHT_BUDGET_S:g} s, and build the Lean modules that are or import what it "
                         f"changes (at most {PREFLIGHT_LEAN_MODULES}, in {PREFLIGHT_LEAN_S:g} s, on a warm Lean tree: here on a "
                         f"node, else with --on on {PREFLIGHT_LEAN_ON}); the records gate (a changed Lean lock needs a records "
                         "run that passed it, of the head or of a commit whose files of its package and the packages it requires "
                         "(tests and docs aside), lean-toolchain and the Lean checker (tools/verity/lean) are the head's) still runs")
    ap.add_argument("--regenerate-locks", action=argparse.BooleanOptionalAction,
                    help="drive: land a commit whose Lean check passed by regenerating lock records with those locks committed "
                         "on top (`Regenerated:`), which `research merge`'s gate accepts on that check (CI-64)")
    ap.add_argument("--marked-after", metavar="YYYY-MM-DDTHH:MM:SSZ", help="status/sync/drive: admit only ready marks written at "
                                                                           "or after this time (default: `marked_after` in the rules file)")
    ap.add_argument("--first", metavar="N[,N...]", help="sync/drive: put these ready PRs at the bottom of `next`, in this "
                                                        "order, ahead of members that keep their place (a fix `main` "
                                                        "needs)")
    ap.add_argument("--after-landing-no-cache", action=argparse.BooleanOptionalAction,
                    help="drive: run the after-landing check reusing no pass but circuit-check's per-target ones "
                         "(check.py --no-cache --circuit-cache)")
    for old in ("--max-running", "--train-size"):
        ap.add_argument(old, type=int, metavar="N", help="drive: accepted and ignored (train landing was deleted, #5070)")
    ap.add_argument("--land", action="store_true", help="review N: the verdict that lands it")
    ap.add_argument("--hold", metavar="REASON", help="review N: the verdict that keeps it, and everything above it, on `next`")
    ap.add_argument("--ref", help="review N --land: the note or reason the verdict rests on")
    ap.add_argument("--config", metavar="PATH",
                    help=f"drive: read its flags from PATH (`DriveFlags`, e.g. {DRIVE_FLAGS}) at `main` in the queue's clone, "
                         "fresh each tick; a flag given here overrides the file's")
    ap.add_argument("--config-from-tree", action="store_true", help="drive: read --config's PATH from this file system instead")
    ap.add_argument("--head", help="open: the pushed branch to open a PR for")
    ap.add_argument("--sha", help="open: refuse unless origin's branch is at this commit (the owner's pinned head); ready: mark only "
                                  "this head, refusing when GitHub still shows another after a few reads (push && ready, safely); "
                                  "eject: only while this is the PR's head, so an eject that waits for drive's tick never hits a "
                                  "head pushed meanwhile")
    ap.add_argument("--title", help="open: the PR's title")
    ap.add_argument("--body-file", type=Path, help="open: the PR's body, verbatim (its first line `Lane NAME` names the owner); "
                                                   "edit: the PR's whole new body, in place of all its text")
    ap.add_argument("--base", default="main", help="open: the base branch (default main)")
    ap.add_argument("--tip", default="origin/main", help="checked-heads: whose first-parent line (default origin/main)")
    ap.add_argument("--out", type=Path, metavar="FILE", help="checked-heads: write the checked train heads newest first "
                    "to FILE (an optional source for the verifier's --main file) instead of listing them on stdout")
    ap.add_argument("--draft", action="store_true", help="open: as a draft (default: ready for review)")
    ap.add_argument("--log", type=Path, help="stats: drive's log (its stdout, every tick's)")
    ap.add_argument("--hours", type=float, default=24.0, help="stats: the window's length (default 24)")
    ap.add_argument("--until", metavar="YYYY-MM-DDTHH:MM:SSZ", help="stats: the window's end (default now)")
    a = ap.parse_args(argv)
    if a.command == "stats":
        if a.log is None:
            ap.error("stats needs --log FILE (drive's log)")
        return _stats(a)
    if a.command == "checked-heads":
        return _checked_heads(a)
    if a.command == "open":
        if not (a.head and a.title and a.body_file):
            ap.error("open needs --head BRANCH, --title T and --body-file F")
        try:
            top = Path(_out(Path.cwd(), "rev-parse", "--show-toplevel"))
            remote = a.remote or _out(top, "remote", "get-url", "origin")
            repo = a.repo or ((m := SLUG_RE.search(remote)) and m.group(1))
            if not repo:
                raise QueueError(f"can't tell OWNER/NAME from {remote!r}: pass --repo")
            if a.dry_run:                                   # --dry-run never opens a real PR
                print(f"research queue open: would open {a.head}{'@' + a.sha[:12] if a.sha else ''} onto {a.base} in {repo}"
                      f"{' as a draft' if a.draft else ''}: {a.title!r} ({len(a.body_file.read_text())} bytes of body; dry run)")
                return 0
            print(open_pr(top, repo, a.head, a.title, a.body_file.read_text(), base=a.base, sha=a.sha, draft=a.draft))
            return 0
        except (QueueError, OSError) as e:
            print(f"research queue: {e}", file=sys.stderr)
            return 1
    if a.command == "edit":
        if not (a.pr and a.pr.isdigit() and a.body_file):
            ap.error("edit needs the PR number and --body-file F")
        try:
            top = Path(_out(Path.cwd(), "rev-parse", "--show-toplevel"))
            remote = a.remote or _out(top, "remote", "get-url", "origin")
            repo = a.repo or ((m := SLUG_RE.search(remote)) and m.group(1))
            if not repo:
                raise QueueError(f"can't tell OWNER/NAME from {remote!r}: pass --repo")
            url, body = edit_pr(top, repo, int(a.pr), a.body_file.read_text(), dry_run=a.dry_run)
            if a.dry_run:
                print(f"research queue edit: would replace {url}'s body with ({len(body)} bytes; dry run):\n{body}", end="")
            else:
                print(f"research queue edit: {url}'s body replaced ({len(body)} bytes)")
            return 0
        except (QueueError, OSError) as e:
            print(f"research queue: {e}", file=sys.stderr)
            return 1
    if a.first and not re.fullmatch(r"\d+(,\d+)*", a.first):
        ap.error(f"--first {a.first!r}: PR numbers separated by commas")
    if a.marked_after and not MARK_TIME_RE.fullmatch(a.marked_after):
        ap.error(f"--marked-after {a.marked_after!r}: a UTC time as YYYY-MM-DDTHH:MM:SSZ, as labels record it")
    if (a.config or a.config_from_tree) and a.command != "drive":
        ap.error("--config and --config-from-tree are for drive only")
    if a.config_from_tree and not a.config:
        ap.error("--config-from-tree needs --config PATH")
    if a.dry_run and a.command not in ("sync", "drive"):
        ap.error(f"--dry-run is for sync and drive only: `{a.command}` would act for real")
    if a.command == "ready" and bool(a.on) + a.local + a.mark != 1:
        ap.error("ready takes one of --on MACHINE (the quick tier as a run there, in a check slot), --local (here) and --mark (no tier)")
    if a.no_preflight and a.command != "ready":
        ap.error("--no-preflight is for ready only")
    if a.command == "judge" and not a.pr:
        ap.error("judge needs the quick tier's run id")
    if a.command == "drive" and not a.on and not a.config:
        ap.error(f"{a.command} needs --on MACHINE (train.sh's: a machine, or `auto`) for the checks it launches")
    if a.command in ("ready", "judge") and not a.by:
        ap.error(f"{a.command} writes the owner's ready mark: pass --by <lane> (or set $RESEARCH_LANE)")
    if (a.land or a.hold is not None or a.ref) and not (a.command == "review" and a.pr):
        ap.error("--land, --hold and --ref are for review N only")
    if a.command == "review" and a.pr and (a.land == (a.hold is not None) or not a.by):
        ap.error("review N records one verdict: --land [--ref NOTE] or --hold REASON, with --by <reviewer lane> (or $RESEARCH_LANE)")
    if a.command == "baseline" and not (a.pr and a.by):
        ap.error("baseline SHA records a green `main` for the review landing: pass the commit and --by ci")
    if a.command == "lean-check" and not (a.pr and a.by):
        ap.error("lean-check COMMIT launches a Lean check of a commit of `next` through drive: pass the commit and --by <lane> "
                 "(or set $RESEARCH_LANE)")
    a.by = a.by or "merge-queue"
    if a.command not in ("judge", "baseline", "lean-check") and a.pr is not None:
        if not a.pr.isdigit():
            ap.error(f"{a.command}: {a.pr!r} is not a PR number")
        a.pr = int(a.pr)
    from .store.local import LocalStore

    try:
        remote = a.remote or _out(Path.cwd(), "remote", "get-url", "origin")
        repo = a.repo or ((m := SLUG_RE.search(remote)) and m.group(1))
        if not repo:
            raise QueueError(f"can't tell OWNER/NAME from {remote!r}: pass --repo")
        store = LocalStore(a.store)
        if a.command == "judge":
            here = _git(Path.cwd(), "rev-parse", "--show-toplevel", check=False).stdout.strip()
            return judge(GitHub(repo), store, a.pr, a.by, on=a.on, repo=Path(here) if here else None, url=remote,
                         records=RecordsGate(Path(here) if here else Path.cwd(), remote, store))
        if a.command == "ready":
            top = Path(_out(Path.cwd(), "rev-parse", "--show-toplevel"))
            tier = MarkOnly() if a.mark else RemoteQuickTier(top, remote, a.on, by=a.by) if a.on else QuickTier(top, remote)
            pre = None if a.no_preflight else Preflight(top, remote, store,
                                                        lean=LakeBuild(on=PREFLIGHT_LEAN_ON if a.on else None, by=a.by))
            return ready(GitHub(repo), store, a.pr, a.by, tier, again=a.again, preflight=pre, sha=a.sha, as_json=a.json,
                         records=RecordsGate(top, remote, store))
        labels = StoreLabels(store, log=lambda s: print(s, file=sys.stderr), since=a.marked_after)
        q = Queue(a.workdir, remote=remote, github=GitHub(repo), ready=labels.ready, store=store, shadow=a.shadow,
                  rules=Rules.parse(a.rules.read_text(), str(a.rules)) if a.rules else None,
                  auth=credential_args(Path.cwd(), remote), first=tuple(int(n) for n in (a.first or "").split(",") if n),
                  before_cutoff=labels.before_cutoff, marked_by=labels.marked_by, marked_at=labels.marked_at,
                  prefetch=labels.prefetch)
        lock = _lock(a.workdir)  # noqa: F841
        if ignored := [f"--{k.replace('_', '-')}" for k in ("max_running", "train_size") if getattr(a, k) is not None]:
            q.log(f"research queue drive: {' and '.join(ignored)} ignored: train landing was deleted (#5070); drop them from "
                  "drive's args")
        flags = DriveFlags().override(a)
        if a.config:
            if not a.config_from_tree:
                q._clone()
                q.fetch()
            base, where = drive_flags(a.config, None if a.config_from_tree else q.workdir)
            flags = base.override(a)
            if not flags.on:
                raise QueueError(f"drive needs a machine for the checks it launches: set `on` in {where}, or pass --on MACHINE")
            q.first, q.nonlean_first, q.reviewed_first = flags.first, flags.nonlean_first, flags.reviewed_first
            a.marked_after = flags.marked_after
            labels.since = a.marked_after
            q.log(f"research queue drive: flags from {where}: {flags.line()}")
        if not a.marked_after:
            if a.rules is None:
                q._clone()
                q.fetch()
            labels.since = q.rules().marked_after
        if a.command == "status":
            from .stale import stale_prs

            q._clone()
            q.fetch()
            r = _git(q.workdir, "show", f"refs/remotes/origin/main:{DRIVE_FLAGS}", check=False)
            got = DriveFlags.parse(r.stdout, f"{DRIVE_FLAGS} at main") if r.returncode == 0 else DriveFlags()
            entries = q.status(lambda prs: stale_prs(q, prs, q.now(), got.stale_days,
                                                     lambda n, h: [lab.ts for lab in labels.reviews(n, h)], keep=got.stale_keep),
                               got.stale_close_days)
            if a.json:
                print(json.dumps([{"pr": e.pr.number, "head": e.pr.head, "owner": e.pr.owner, "draft": e.pr.draft, "state": e.state,
                                   "why": e.why, "position": e.position, "commit": e.commit, "after": list(e.after),
                                   "stale": e.idle_since is not None,
                                   "idle_since": None if e.idle_since is None else
                                   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(e.idle_since))}
                                  for e in sorted(entries.values(), key=lambda e: e.pr.number)], indent=1))
            else:
                print(_render(entries))
            return 0
        if a.command == "review":
            return review(q, store, labels, a.pr, "land" if a.land else "hold" if a.pr else None, a.hold or a.ref, a.by,
                          sha=a.sha, as_json=a.json)
        from .merge import DRIVE_ENV, off_drive, on_drive

        if a.command == "baseline":
            if not on_drive():
                raise QueueError(f"baseline: drive's state is the control pod's, and this isn't drive (no {DRIVE_ENV}=1): run "
                                 f"`/data/infra/drive.sh queue baseline {a.pr} --by {a.by}` there")
            return baseline(q, a.pr, a.by, a.steps)
        if a.command == "lean-check":
            if not on_drive():
                raise QueueError(f"lean-check: drive's state is the control pod's, and this isn't drive (no {DRIVE_ENV}=1): run "
                                 f"`/data/infra/drive.sh queue lean-check {a.pr}` there")
            from .drive import Driver

            if not a.on:
                q._clone()
                q.fetch()
            if not (on := a.on or drive_flags(DRIVE_FLAGS, q.workdir)[0].on):
                raise QueueError(f"lean-check: {DRIVE_FLAGS} at `main` names no machines (`on`): pass --on MACHINES")
            rec = Driver(q, on=on).hand_lean(a.pr, a.by, again=a.again)
            print(f"research queue: the Lean check of {a.pr} on {on} is drive's, by {a.by}: "
                  f"{' '.join(rec['ids']) or 'launching (log ' + rec['launch']['dir'] + '/log)'}")
            return 0

        if a.command == "sync" and not (a.dry_run or on_drive()):
            raise QueueError(off_drive(q.branch, "queue sync"))
        if a.command == "drive":
            q._drive_only("main", "once", a.dry_run)          # its landings push `main`, its syncs `next`
        if a.command == "sync":
            q.hand_sync(dry_run=a.dry_run)
            return 0
        if a.command == "drive":
            from .drive import Driver

            Driver(q, on=flags.on, regenerate_locks=flags.regenerate_locks, after_landing_no_cache=flags.after_landing_no_cache,
                   reviews=labels.reviews, ready_by=labels.ready_by, max_lean_checks=flags.max_lean_checks,
                   stale_days=flags.stale_days, stale_close_days=flags.stale_close_days,
                   stale_keep=flags.stale_keep).tick(dry_run=a.dry_run)
            return 0
        if a.pr is None or not a.reason:
            ap.error("eject needs a PR number and --reason")
        # no sync here, on drive or off: drive's tick rebuilds `next` with its hold, its `first` and its flags (an
        # eject's own sync would rebuild every member of `next` and drop the Lean checks running on it)
        told = q.tell_ejected(q.eject(a.pr, a.reason, a.by, sha=a.sha), a.reason, a.by)
        if on_drive() and (busy := lean_running(q.load_state(), a.pr)):
            print(f"research queue: #{a.pr}'s mark is withdrawn; the Lean check {busy} covers it, so drive rebuilds `{q.branch}` "
                  f"the tick after that check ends, and lands nothing from #{a.pr} up meanwhile")
        else:
            print(f"research queue: #{a.pr}'s mark is withdrawn; drive's next tick rebuilds `{q.branch}` without it (only drive's "
                  "tick pushes it)")
        return told
    except QueueError as e:
        print(f"research queue: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
