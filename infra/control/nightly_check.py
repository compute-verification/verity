"""The nightly check of `main`, which the control pod starts by itself: once a day, from `--hour` UTC, one uncached check of
`main`'s tip, recorded on `--on` in a slot of its `--pool`; then the loop reads the run's result, compares its verdicts part by
part with the recorded check's at the same commit, posts both once in #agent-alerts and labels the commit. A nightly uncached
check must agree with the cached ones (AGENTS.md, "Checking and merging"); this is what runs it and says what it found.

Each pass does the next thing the night needs and saves the state after every step, so a restarted loop goes on from where it
stopped and nothing is launched or posted twice:

    prepare   fetch origin's `main` into the loop's own clone (`--work`/repo, made on first use, over https with the credential
              helper) and add a worktree of its tip under `--work`/trees. Both are on the pod's own disk: on /data's NFS a tree
              takes minutes to add and remove. Origin unreachable is tried again each pass, and is an error after PREPARE_S.
    launch    in that worktree, with its own tools/research/src alone on PYTHONPATH and TMPDIR `--tmp`:
                  python tools/verity/check/check.py --record --on NODE --pool POOL --base <git's empty tree> --no-cache --timeout S
              the dispatcher's `audit-main` command (`research.jobs.dispatch.command`) in a pool of its own: every path counts as
              changed, so every step runs and the Lean agreement's inputs are sent. The launch returns once the run is launched;
              the run is the one its `research: run R launched` line names, and a launch that names none is an error. Its output
              is kept beside the state (`NAME.launch.log`, emptied before each launch), where a loop restarted during the
              launch finds it: the launch runs in a session of its own and outlives the loop, and no night launches twice.
              The worktree is removed once the launch has returned, and nothing a removal does or fails to do changes the
              night: what is left is removed by the next night's prepare.
    watch     each pass fetches the run's records and reads them as the dispatcher reads a check's (`dispatch.classify`): its
              exit code and check's result.json, with each failed step, each failed suite (suites.json, suites-lean.json)
              and `cache_contradictions`. A run that hasn't ended LOST_S after its own timeout is lost.
    compare   a run that ended with a verdict, once: each of its parts against the recorded checks of the same commit, which
              are the store's attempts of the clean commit whose result is a check's, but this run and any other uncached
              one. A part is a suite (its row in a suite report), a Lean package (its report in lean-audit/audit.json and
              its inputs key in the result's `passes`) or a step (its record in the result; every step but those of
              BY_PART, whose parts are the suites and the Lean packages). The recorded verdict on a part is the newest
              recorded check's that holds one. The part agrees when that verdict is the uncached one, under the same key
              where both sides hold one, and for a Lean package with the same report outside RUN_FIELDS. A comparison
              that can't be read is tried again each pass, and after COMPARE_S the night reports without one and says so.
    report    the outcome and the comparison, once each: a post in #agent-alerts, the rings (below), and the label
              `nightly_check` on `commit:<sha>` with the run as its ref. One that fails is tried again each pass until the
              next night is due.

The outcomes: `passed`, the check passed and its result names no cached pass it contradicts; `contradicted`, its result names
one, a cached pass that answers for something the uncached run failed, whatever the check's verdict (check records a
contradiction only for a suite that failed fresh); `failed`, the check failed and names none; `error`, no verdict: the run
never started (slot.py's exit 75), timed out, was refused or lost, or nothing was launched. A pass says what check's own
`--no-cache` and `cache_contradictions` say, and no more than they cover. The state's `last` is the newest outcome and when
it was read, which the monitors' `nightly-check` reads to ring infra when there has been none for 36 h.

The comparison changes no outcome and no label. It adds to the post how many parts agree, how many disagree and how many
have no stored verdict, and names each part that disagrees with both verdicts, the key and the run that recorded the other
one (with the run that stored a suite's reused pass). It reaches what check's own `cache_contradictions` doesn't, which
is a suite's fresh failure against a pass the uncached run finds in its own cache: a recorded failure the uncached run
passed, a Lean package, a step, a key or a Lean report that differs, on whatever machine the recorded check ran. A part
with no stored verdict is listed, but a step the recorded check skipped by name (lean-agreement when its pinned upstream
build was not sent), which is said to be skipped in the recorded check, with its reason; a commit with no recorded check is
said to have none, never to agree. A part an earlier
recorded check of the commit failed and a later one passed, as a check run again after a failure leaves it (or passed
and then failed), is counted and named in one more sentence, and is no disagreement: the later verdict is the recorded
one. It does not cover: each test's own outcome (a suite is one part, read by its row's status); the uncached run's keys
for the steps check caches whole, which `--no-cache` never computes, so such a step's verdicts are compared and its key
is the recorded side's alone; the key of a suite a recorded check ran in parts or passed by a traced pass, whose row holds
the part's key or the suite's traced key and not the key of its inputs, so only its verdict is compared; circuit-check's
targets, which are one part together; an earlier recorded check's key and Lean report, where a later one holds the part;
a step several shards of one check each run, as each runs the preflight, on which the shard launched last holds the
recorded verdict and another shard's failure reads as an earlier check's; and a part the uncached run skipped, which is
listed when a recorded check has it.

The rings, each an inbox thread: nobody's for a pass with no part that disagrees (one post, naming nobody); infra's for any
other outcome and for a comparison that could not be read; infra's and ci's when a part disagrees, whatever the outcome.

The first pass on a pod starts no night for a day whose hour has passed: its first night is the next one.

usage (the watchers' `nightly-check` loop on the control pod, infra/control/watchers.toml, which holds the node, the pool, the
timeout and the hour; research importable, $SLACK_BOT_TOKEN and the store's key set):
  python3 infra/control/nightly_check.py --on NODE --pool POOL --timeout S --hour H --work DIR --state FILE [--tmp DIR]
      [--every 300] [--once] [--dry-run]
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import http.client
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from research import monitors, slack
from research.jobs import dispatch

SCHEMA = "verity/nightly-check/v1"
STAMP, iso = monitors.STAMP, monitors.iso       # the state's times, as the monitors' nightly-check reads them
URL = "https://github.com/compute-verification/verity.git"
HELPER = "!/data/infra/gh_app_token.sh git"     # hands git the App's token when git asks: never in a config or an argument
CHECK = "tools/verity/check/check.py"
KIND = "audit-main"                             # the dispatcher's name for this check, whose records `dispatch.classify` reads
SUITE_FILES = ("suites.json", "suites-lean.json")     # check's suite reports, beside its result.json in the run dir
LEAN_AUDIT = "lean-audit/audit.json"            # the Lean check's report there: each package's
#: the steps whose parts are compared one by one, as suites and Lean packages; every other step of a result is one part, so
#: a step check gains is compared from the night it first runs
BY_PART = ("pytest", "lean-suites", "lean-audit")
#: the fields of a Lean package's report that say how its check ran; the others say what it found. A reused pass's report is
#: the one its storing run wrote, so these are that run's: its times, sample, seal and replay, its node's (`sandbox`,
#: `build_threads`, `peak_gib`, and whether it used its replay and facts caches, `replay_cache`, `facts_cache`), the content
#: address of its build (`facts_key`: it folds the `.olean` files, whose bytes differ between machines, and only a records
#: run's `--facts-cache` writes it) and the records run whose facts it was judged from (`judged_from`)
RUN_FIELDS = frozenset({"seconds", "sampled", "staged", "replay", "pipeline", "sandbox", "build_threads", "peak_gib",
                        "replay_cache", "facts_cache", "facts_key", "judged_from"})
VERDICTS = ("passed", "failed")
LOST = ("launcher-exited", "runner-gone")       # a run that ended leaving no status (research.cli.ENDED)
OUTCOMES = ("passed", "contradicted", "failed", "error")
RINGS = OUTCOMES[1:]
RING_OF = {"ring": "infra", "ring_ci": "ci"}    # a report's rings, as the night's state names them, and whose inbox each rings
BY = "nightly-check"                            # who the label is by
PREPARE_S = 3600        # origin unreachable this long after the night's first try: an error
LOST_S = 3600           # a run that hasn't ended this long after its own timeout is lost
COMPARE_S = 3600        # a comparison unread this long after the outcome was read: the night reports without one
BEFORE_S = 3600         # a commit's recorded checks are looked for from this long before it was made: their ids are another clock's
GIT_S = 1800            # one git command; the first fetch brings all of main's history
KILL_S = 30             # a launch past its time, between its SIGTERM and its SIGKILL
SHOWN = 8               # names in one list of a post


@dataclass(frozen=True)
class Settings:
    on: str
    pool: str
    timeout: int
    hour: int
    work: Path
    state: Path
    tmp: Path | None = None
    url: str = URL
    helper: str = HELPER

    @property
    def launch_log(self) -> Path:
        return self.state.with_name(self.state.stem + ".launch.log")


@dataclass(frozen=True)
class Read:
    """What a pass read of the night's run: `outcome` None while it goes on, else its outcome, the words its post says and
    how long the run took."""
    outcome: str | None
    detail: str
    minutes: int | None = None


def log(msg: str) -> None:
    print(f"{time.strftime(STAMP, time.gmtime())} {msg}", flush=True)


def at(stamp: str) -> float:
    return float(calendar.timegm(time.strptime(stamp, STAMP)))


def command(cfg: Settings, python: str = sys.executable) -> list[str]:
    """The launch, run in a worktree of the commit: `dispatch.command`'s `audit-main`, in `cfg.pool` and with the run's timeout."""
    return [python, CHECK, "--record", "--on", cfg.on, "--pool", cfg.pool, "--base", dispatch.EMPTY_TREE, "--no-cache",
            "--timeout", str(cfg.timeout)]


def due(state: dict, cfg: Settings, now: float) -> str | None:
    """The UTC day whose night starts now: today, from `cfg.hour` on, unless today's has started."""
    t = time.gmtime(now)
    day = time.strftime("%Y-%m-%d", t)
    return day if t.tm_hour >= cfg.hour and (state.get("night") or {}).get("day") != day else None


def listed(items: list[str]) -> str:
    return ", ".join(items[:SHOWN]) + (f" and {len(items) - SHOWN} more" if len(items) > SHOWN else "")


def some(names: list[str]) -> str:
    return listed([f"`{n}`" for n in names])


def judge(s: dict, result: dict, suites: list[dict], cfg: Settings) -> Read:
    """A run's records (`research.runs.summarize`, check's result.json and its suite reports' rows) as the night's outcome."""
    state, rc = s.get("state"), s.get("rc")
    ob = dispatch.classify(KIND, s)
    if ob.state == "running" and state not in LOST:
        return Read(None, ob.summary)
    minutes = round(s["wall_s"] / 60) if isinstance(s.get("wall_s"), (int, float)) else None
    if ob.state == "passed" and not s.get("result_valid"):
        # the dispatcher takes exit 0 and a result that says passed; a pass here also says nothing was contradicted, which
        # only this run's own valid result can
        return Read("error", f"it exited 0, but its result.json isn't this run's valid result ({s.get('result')}), so what it "
                    "contradicted can't be read.", minutes)
    if ob.state in ("passed", "failed"):
        steps = [str(x.get("step")) for x in result.get("steps") or [] if isinstance(x, dict) and x.get("status") == "failed"]
        red = sorted({str(r.get("suite")) for r in suites if isinstance(r, dict) and r.get("status") == "failed"})
        said = [c for c in result.get("cache_contradictions") or [] if isinstance(c, dict) and c.get("contradicts")]
        failed = ("failed steps: " + (some(steps) if steps else "none named in its result")
                  + ("; failed suites: " + some(red) if red else "") + ".") if ob.state == "failed" else ""
        if said:
            tests = "; ".join(f"`{c.get('suite')}`: {', '.join(str(t) for t in c['contradicts'][:3])}"
                              + (f" and {len(c['contradicts']) - 3} more" if len(c["contradicts"]) > 3 else "") for c in said[:SHOWN])
            more = f"; and {len(said) - SHOWN} more suites" if len(said) > SHOWN else ""
            return Read("contradicted", f"the uncached check {ob.state}, and a cached pass answers for what it failed, so a cache "
                        f"key misses an input or a test isn't a function of its inputs: {tests}{more}." + (f" Its {failed}" if failed else ""),
                        minutes)
        return Read(ob.state, f"{failed} Its result names no cached pass it contradicts." if failed else "", minutes)
    if rc == monitors.NOT_STARTED:
        why = f"it never started: no slot of `{cfg.on}`'s `{cfg.pool}` pool took it (exit {rc}), so nothing was checked"
    elif s.get("class") == "STAGE_TIMEOUT":
        why = f"it timed out at its {cfg.timeout} s"
    elif state in LOST:
        gone = (s.get("remote_rec") or {}).get("runner_gone" if state == "runner-gone" else "launcher_exited") or {}
        why = f"it was lost on `{cfg.on}`: {gone.get('why') or gone.get('last_line') or 'no reason recorded'}"
    elif state == "refused":
        why = f"`{cfg.on}` refused it, so nothing ran: {s.get('why') or 'no reason recorded'}"
    else:
        why = "it ended without a verdict"
    return Read("error", f"{why} ({ob.summary if ob.state == 'error' else state}).", minutes)


# ---- the comparison with the recorded check ---------------------------------------------------------------------------------

def found(field: str, value) -> str:
    """A digest of one field of a Lean package's report, less the build time each heartbeat carries (`module_seconds`, null
    where the build was reused)."""
    if field == "heartbeats" and isinstance(value, list):
        value = sorted(json.dumps({k: v for k, v in h.items() if k != "module_seconds"}, sort_keys=True) if isinstance(h, dict)
                       else json.dumps(h) for h in value)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verdicts(run: str, result: dict, suites: list[dict], audit: dict) -> dict[tuple[str, str], list[dict]]:
    """A check's records (its result, its suite reports' rows and its Lean report) as {(kind, name): its verdicts on that part}:
    each suite's row that ran or was reused, each Lean package's report (a package the report lists as failed is failed,
    report or none), each step that ran but those of BY_PART, and each step check skipped by name, `skipped` with its reason
    (the step's `detail` up to its parenthesis) as its `how`. A verdict holds its `run` and `status`; the `key` it was given
    under, where an uncached and a recorded check hold the same kind of key (a Lean package's inputs key; a suite's inputs
    key, which the row of a part of a split suite and the row of a reused traced pass don't hold: theirs are the part's key
    and the suite's traced key); a Lean package's `report`, each field outside RUN_FIELDS by digest; `how` a pass it
    reused came to it (a Lean package's names lean-audit's cache key, its own hash of the package's inputs, which is never
    the inputs key and which an uncached check doesn't print); and its `row`, which part of a split suite it is a verdict
    on (None: the whole part)."""
    out: dict[tuple[str, str], list[dict]] = {}

    def add(kind: str, name, status: str, key=None, how: str = "", report: dict | None = None, row: str | None = None) -> None:
        out.setdefault((kind, str(name)), []).append(
            {"run": run, "status": status, "key": key, "how": how, "report": report, "row": row})

    for r in suites:
        status = {"cached": "passed"}.get(r.get("status"), r.get("status")) if isinstance(r, dict) else None
        if status not in VERDICTS:
            continue
        since = (r.get("since") or {}) if r["status"] == "cached" else {}
        traced = "traced " if since.get("traced") else ""
        add("suite", r.get("suite"), status, None if r.get("part") or traced else r.get("key"),
            "" if r["status"] != "cached" else f"a {traced}pass `{since['run']}` stored" if since.get("run") else f"a {traced}pass it reused",
            row=json.dumps(r["part"], sort_keys=True) if r.get("part") else None)
    keys = {pkg: key for key, pkg in (result.get("passes") or {}).items()}
    reused, failed = audit.get("cached") or {}, audit.get("failed") or []
    for r in audit.get("packages") or []:
        if isinstance(r, dict) and isinstance(pkg := r.get("package"), str):
            add("Lean package", pkg, "passed" if r.get("failures") == [] and pkg not in failed else "failed", keys.get(pkg),
                f"a pass reused from lean-audit's cache, where its key is `{str(reused[pkg])[:12]}`" if pkg in reused else "",
                {k: found(k, v) for k, v in r.items() if k not in RUN_FIELDS})
    for pkg in failed:
        if ("Lean package", str(pkg)) not in out:        # one whose check left no report
            add("Lean package", pkg, "failed", keys.get(pkg))
    for s in result.get("steps") or []:
        if isinstance(s, dict) and isinstance(s.get("step"), str) and s["step"] not in BY_PART and s.get("status") in VERDICTS:
            hit = s["cache"].get("hit") if isinstance(s.get("cache"), dict) else None
            add("step", s["step"], s["status"], how=f"a pass reused under key `{str(hit)[:12]}`" if hit else "")
        elif isinstance(s, dict) and isinstance(s.get("step"), str) and s["step"] not in BY_PART and s.get("status") == "skipped":
            add("step", s["step"], "skipped", how=str(s.get("detail") or "").removeprefix("skipped:").split(" (")[0].strip())
    return out


def told(part: tuple[str, str], u: dict, stored: list[dict]) -> str | None:
    """The line for a part the recorded check disagrees with the uncached check on, or None when every recorded verdict that
    counts (`stored`: `tally`'s) is the uncached one, under its key and with its report: the part, the uncached verdict and
    key, then each recorded verdict with its run, how it came by a pass it reused, the key where it is another, and the
    fields of a Lean report that differ."""
    there, differs = [], False
    for s in stored:
        key = f", key `{s['key'][:12]}`" if s["key"] and s["key"] != u["key"] else ""
        mine, theirs = u["report"], s["report"]
        fields = sorted(k for k in {*mine, *theirs} if mine.get(k) != theirs.get(k)) if mine is not None and theirs is not None else []
        differs = differs or s["status"] != u["status"] or bool(key and u["key"]) or bool(fields)
        there.append(f"{s['status']} in `{s['run']}`" + (f" ({s['how']})" if s["how"] else "") + key
                     + (f", its report differing in {some(fields)}" if fields else ""))
    if not differs:
        return None
    return f"{part[0]} `{part[1]}`: {u['status']} here" + (f", key `{u['key'][:12]}`" if u["key"] else "") + "; " + "; ".join(there)


def tally(mine: dict[tuple[str, str], list[dict]], theirs: dict[str, dict]) -> dict:
    """The uncached check's verdicts against the recorded checks' (`theirs`: {run: its `verdicts`}), part by part, as the night's
    state keeps the comparison: the recorded `runs`, how many parts `agree`, a line for each that disagrees (`told`), the parts
    with no stored verdict (`unstored`), those a recorded check skipped by name and none gave a verdict on (`skipped`, each
    with the newest reason a skip gave), those only a recorded check has a verdict on (`unchecked`), and those an `earlier`
    recorded check gave another verdict than a later one, under the earlier verdict. A skip is no verdict: the uncached
    check's own skips are left out, and a recorded check's never hides another's verdict on the part.

    The recorded verdict that counts on a part is the newest recorded check's that holds one, a run's id starting with when
    it was launched: a check run again after a failure doesn't count against the night by its first try. It is the newest
    row by row for a suite recorded in parts, each of which a check holds in the shard that ran it."""
    stored: dict[tuple[str, str], list[dict]] = {}
    skips: dict[tuple[str, str], str] = {}
    earlier: dict[str, set[str]] = {status: set() for status in VERDICTS}
    mine = {part: got for part, v in mine.items() if (got := [g for g in v if g["status"] in VERDICTS])}
    for run in sorted(theirs, reverse=True):
        for part, got in theirs[run].items():
            newer = {v["row"]: v["status"] for v in stored.get(part, [])}
            for v in got:
                if v["status"] not in VERDICTS:
                    skips[part] = skips.get(part) or v["how"]
                elif v["row"] not in newer:
                    stored.setdefault(part, []).append(v)
                elif v["status"] != newer[v["row"]]:
                    earlier[v["status"]].add(f"{part[0]} `{part[1]}`")
    out = {"runs": sorted(theirs), "agree": 0, "disagree": [], "unstored": [], "skipped": [],
           "unchecked": [f"{kind} `{name}`" for kind, name in sorted(set(stored) - set(mine))],
           "earlier": {status: sorted(parts) for status, parts in earlier.items()}}
    for part in sorted(mine):
        u = mine[part][0]
        if len(mine[part]) > 1:                  # the uncached check ran it in parts: one verdict, failed when a part failed
            u = {**u, "status": "failed" if any(v["status"] == "failed" for v in mine[part]) else "passed", "key": None}
        if part not in stored and part in skips:
            out["skipped"].append([f"{part[0]} `{part[1]}`", skips[part]])
        elif part not in stored:
            out["unstored"].append(f"{part[0]} `{part[1]}`")
        elif line := told(part, u, sorted(stored[part], key=lambda v: v["run"])):
            out["disagree"].append(line)
        else:
            out["agree"] += 1
    return out


def said(c: dict | None) -> str:
    """The comparison's sentences in the night's post: none for a night with no verdicts to compare."""
    if not c:
        return ""
    if "unread" in c:
        return f" Its verdicts could not be compared with the recorded check's: {c['unread']}."
    if not c["runs"]:
        return " The store holds no recorded check of this commit, so its verdicts were compared with none."
    return (f" Against the recorded check of this commit ({some(c['runs'])}), part by part: {c['agree']} agree, "
            f"{len(c['disagree'])} disagree, {len(c['unstored'])} with no stored verdict"
            + (f" ({listed(c['unstored'])})" if c["unstored"] else "")
            + "".join(f", {part} skipped in the recorded check" + (f": {why}" if why else "") for part, why in c.get("skipped", []))
            + "; circuit-check is one part, its targets not compared one by one."
            + (f" Only the recorded check has a verdict on {listed(c['unchecked'])}." if c["unchecked"] else "")
            + "".join(f" {len(parts)} part{'s' if len(parts) > 1 else ''} {was} in an earlier recorded check of this commit and "
                      f"{then} in a later one: {listed(parts)}."
                      for was, then in (("failed", "passed"), ("passed", "failed")) if (parts := c["earlier"][was]))
            + "".join(f"\n\u2022 {line}" for line in c["disagree"]))


def rung(night: dict) -> list[str]:
    """Whose inbox the night's post rings: nobody's for a pass no part of which disagrees, infra's for any other outcome and for
    a comparison that could not be read, infra's and ci's when a part disagrees."""
    c = night.get("compared") or {}
    if c.get("disagree"):
        return ["infra", "ci"]
    return ["infra"] if night["outcome"] in RINGS or "unread" in c else []


def text(night: dict, tags: dict[str, str]) -> str:
    """The night's one post: a pass that rings nobody is one paragraph naming nobody; anything else leads with the tags of those
    it rings (`rung`)."""
    commit = f"`{night['commit'][:12]}`" if night.get("commit") else "a commit it never read"
    run = f"run `{night['run']}`" if night.get("run") else "no run"
    took = f", {night['minutes']} min" if night.get("minutes") is not None else ""
    c, to = night.get("compared") or {}, rung(night)
    if not to:
        return f"Nightly check of `main` passed: {commit}, {run}{took}.{said(c)}"
    title = {"passed": "passed", "contradicted": "contradicts the cache", "failed": "failed", "error": "gave no verdict"}[night["outcome"]]
    title += (", and disagrees with the recorded check" if c.get("disagree") else
              ", and was not compared with the recorded check" if "unread" in c else "")
    detail = f" {night['detail']}" if night["detail"] else ""
    return f"{' '.join(tags[n] for n in to)} *Nightly check of `main` {title}* at {commit} ({run}{took}):{detail}{said(c)}"


# ---- one pass -------------------------------------------------------------------------------------------------------------------

def finish(state: dict, night: dict, outcome: str, detail: str, now: float, fx, minutes: int | None = None) -> None:
    night.update(phase="reporting", outcome=outcome, detail=detail, minutes=minutes, at=iso(now))
    state["last"] = {k: night.get(k) for k in ("day", "commit", "run", "outcome", "at")}
    fx.log(f"{night['day']}: {outcome}" + (f": {detail}" if detail else ""))


def tidy(night: dict, fx) -> None:
    """Remove the night's worktree. Whatever the removal raises is logged and the night goes on."""
    try:
        fx.cleanup(Path(night["tree"]))
    except Exception as e:  # noqa: BLE001  (no cleanup decides the night: the next prepare removes what is left)
        fx.log(f"{night['day']}: {night['tree']} was not removed ({type(e).__name__}: {e}); the night goes on, and the next "
               f"night's prepare removes what is left")


def launched(state: dict, night: dict, cfg: Settings, fx, now: float, rc: int | None, out: str) -> None:
    """The launch has returned (`rc` None: in a loop that stopped before it did): the run its output names, else an error."""
    tidy(night, fx)
    if m := dispatch.LAUNCHED.search(out):
        night.update(run=m.group(1), launched=iso(now), phase="running")
        fx.log(f"{night['day']}: run {m.group(1)} launched on {cfg.on} at {night['commit'][:12]}")
        return
    tail = " | ".join(ln.strip() for ln in out.splitlines()[-3:] if ln.strip())[-500:]
    how = f"exit {rc}" if rc is not None else "the loop was restarted during it"
    finish(state, night, "error", f"the launch named no run ({how}), so nothing is known to run: "
           f"{tail or 'it printed nothing'}", now, fx)


def start(state: dict, night: dict, cfg: Settings, fx, now: float, save: Callable[[dict], None]) -> None:
    try:
        commit, tree = fx.prepare(night["day"])
        fx.forget_launch()                       # the last night's log: a loop restarted during this launch must not read that run as this night's
    except Exception as e:  # noqa: BLE001  (whatever stops it is tried again next pass, and is the night's error after PREPARE_S)
        why, waited = f"{type(e).__name__}: {e}"[-400:], now - at(night["began"])
        if waited < PREPARE_S:
            fx.log(f"{night['day']}: main's tree isn't ready ({why}); trying again next pass")
            return
        finish(state, night, "error", f"`main`'s tree could not be prepared for {int(waited // 60)} min, so nothing was "
               f"launched: {why}", now, fx)
        return
    night.update(commit=commit, tree=str(tree), phase="launching", launching=iso(now))
    save(state)                                  # before the launch: a loop restarted during it reads the launch's log
    argv = command(cfg)
    fx.log(f"{night['day']}: launch: (cd {tree} && {shlex.join(argv)})")
    try:
        rc, out = fx.launch(argv, tree)
    except (OSError, subprocess.SubprocessError) as e:
        finish(state, night, "error", f"the launch could not start, so nothing runs: {f'{type(e).__name__}: {e}'[-400:]}", now, fx)
        tidy(night, fx)
        return
    launched(state, night, cfg, fx, now, rc, out)


def resume(state: dict, night: dict, cfg: Settings, fx, now: float, save: Callable[[dict], None]) -> None:
    """A loop restarted during the launch: the launch, in its own session, may still be going, so its log is read again each
    pass until it names a run or the launch's time is up."""
    out = fx.launch_output()
    if dispatch.LAUNCHED.search(out) or now - at(night["launching"]) >= dispatch.LAUNCH_TIMEOUT_S:
        launched(state, night, cfg, fx, now, None, out)
    else:
        fx.log(f"{night['day']}: restarted during the launch, whose log names no run yet; reading it again next pass")


def watch(state: dict, night: dict, cfg: Settings, fx, now: float, save: Callable[[dict], None]) -> None:
    try:
        got = fx.read(night["run"])
    except Exception as e:  # noqa: BLE001  (a pass that can't read the run reads it again; the deadline below bounds the wait)
        got = Read(None, f"its records could not be read ({type(e).__name__}: {e})")
    if got.outcome is not None:
        finish(state, night, got.outcome, got.detail, now, fx, got.minutes)
        if got.outcome != "error":
            night["phase"] = "comparing"         # a check that gave a verdict has verdicts to compare
    elif (over := now - at(night["launched"]) - cfg.timeout) >= LOST_S:
        finish(state, night, "error", f"it was lost: not ended {int(over // 60)} min after its {cfg.timeout} s timeout "
               f"({got.detail}). `research status {night['run']}` says what is known.", now, fx)
    else:
        fx.log(f"{night['day']}: run {night['run']}: {got.detail}")


def compare(state: dict, night: dict, cfg: Settings, fx, now: float, save: Callable[[dict], None]) -> None:
    """The ended run's verdicts against the recorded check's, once: a comparison that can't be read is read again each pass, and
    after COMPARE_S the night reports without one."""
    try:
        night["compared"] = fx.compare(night["run"], night["commit"])
    except Exception as e:  # noqa: BLE001  (the store or the run's machine can't be read: read again, then said in the post)
        why = f"{type(e).__name__}: {e}"[-300:]
        if now - at(night["at"]) < COMPARE_S:
            fx.log(f"{night['day']}: its verdicts aren't compared with the recorded check's yet ({why}); trying again next pass")
            return
        night["compared"] = {"unread": why}
    night["phase"] = "reporting"
    fx.log(f"{night['day']}: compared:{said(night['compared'])}")


def owed(night: dict) -> list[str]:
    """What the night's report still lacks."""
    to = rung(night)
    need = (("post", True), *((ring, who in to) for ring, who in RING_OF.items()), ("label", bool(night.get("commit"))))
    return [what for what, wanted in need if wanted and what not in night]


def report(state: dict, night: dict, cfg: Settings, fx, now: float, save: Callable[[dict], None]) -> None:
    words = text(night, fx.tags)
    for what in owed(night):
        if what in RING_OF and "post" not in night:
            continue                             # a ring carries the post's link
        try:
            if what == "post":
                night["post"] = fx.post(words)
            elif what in RING_OF:
                night[what] = fx.ring(night["post"], words, RING_OF[what])
            else:
                night["label"] = fx.label(night["commit"], night["outcome"], night.get("run"))
        except Exception as e:  # noqa: BLE001  (Slack's or the store's refusal: said here, tried again next pass, never the night's result)
            fx.log(f"{night['day']}: no {what} yet ({type(e).__name__}: {e}); trying again next pass")
            continue
        save(state)                              # done once: never posted, rung or labelled twice
        fx.log(f"{night['day']}: {what}: {night[what]}")
    if not owed(night):
        night["phase"] = "done"


PHASES = {"preparing": start, "launching": resume, "running": watch, "comparing": compare, "reporting": report}


def advance(state: dict, cfg: Settings, fx, now: float, save: Callable[[dict], None]) -> None:
    """The night as far as this pass can take it: each phase runs once, and one that stays where it was waits for the next pass."""
    night = state["night"]
    while night["phase"] != "done":
        was = night["phase"]
        PHASES[was](state, night, cfg, fx, now, save)
        save(state)
        if night["phase"] == was:
            return


def step(state: dict, cfg: Settings, fx, now: float, save: Callable[[dict], None]) -> dict:
    """One pass at `now`: the night in progress goes on, and a new one starts when it is due and none is before it."""
    if "since" not in state:
        state.update(schema=SCHEMA, since=iso(now))
        if day := due(state, cfg, now):
            state["night"] = {"day": day, "phase": "done", "skipped": "the loop first ran after that day's hour"}
            fx.log(f"first pass: {day}'s {cfg.hour:02d}:00Z has passed, so the first night is the next day's")
    if (state.get("night") or {}).get("phase", "done") != "done":
        advance(state, cfg, fx, now, save)
    night = state.get("night")
    if (night is None or night["phase"] in ("done", "reporting")) and (day := due(state, cfg, now)):
        if night and night["phase"] == "reporting":
            fx.log(f"{night['day']}: its report still lacks {', '.join(owed(night))}: given up, since {day}'s night is due")
        state["night"] = {"day": day, "phase": "preparing", "began": iso(now)}
        advance(state, cfg, fx, now, save)
    save(state)
    return state


# ---- the control pod ----------------------------------------------------------------------------------------------------------

def rows(path: Path, kind: type):
    """The JSON file's document when it is a `kind`, else an empty one; a file that isn't there holds none."""
    if not path.exists():
        return kind()
    doc = json.loads(path.read_text())
    return doc if isinstance(doc, kind) else kind()


def suite_rows(d: Path) -> list[dict]:
    """The rows of check's suite reports in the directory `d`."""
    return [r for f in SUITE_FILES for r in rows(d / f, list)]


def read(run: str, cfg: Settings, fetch: Callable | None = None, root: Path | None = None) -> Read:
    """Fetch the run's records, with check's suite reports, and judge them; a machine that can't be reached is read again."""
    from research import remote, runs

    d = runs.resolve(runs.runs_root(root), run)
    rec = (fetch or remote.fetch)(d, also=SUITE_FILES)
    if rec.get("last_fetch_error"):
        return Read(None, f"`{cfg.on}` unreachable (rc {rec['last_fetch_error'].get('rc')})")
    s = runs.summarize(d)
    if s["state"] == "refused":
        s["why"] = rows(d / "refused.json", dict).get("reason") or s.get("why")
    return judge(s, rows(d / "result.json", dict), suite_rows(d), cfg)


def stored(commit: str, but: str, since: str, store=None) -> dict[str, dict]:
    """{run: its `verdicts`} of the recorded checks of the clean `commit` the store holds: every attempt there whose result is a
    check's, but the run `but` (the night's own) and any other uncached one, which stores no verdict. The store's attempts
    are listed from its remote first, those since `since` (a run id's prefix), and each check's suite reports and Lean report
    come from its files there. Raises when a result or a report the store lists can't be fetched: a recorded check is never
    read as holding fewer verdicts than it stored."""
    from research.queue import CHECK_KINDS
    from research.store.local import LocalStore

    store, reports = store or LocalStore(), [*SUITE_FILES, LEAN_AUDIT]
    store.refresh(attempts_only=True, since=since)
    out = {}
    for run, doc in store.index.sql("SELECT id, manifest_json FROM attempts WHERE source_commit = ? AND source_dirty = 0 ORDER BY id",
                                    (commit,)):
        doc = json.loads(doc) if doc else {}
        outs = doc.get("outputs") or {}
        if run == but or not outs.get("result") or any("--no-cache" in str(a) for a in doc.get("argv") or []):
            continue
        result = store.get_manifest(outs["result"])
        if result.kind not in CHECK_KINDS:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            files = outs.get("run_files") or outs.get("run_record")
            if files and store.listing(files, reports):          # a shard that ran no suite and no Lean check holds none of them
                store.fetch(files, Path(tmp), paths=reports)
            out[run] = verdicts(run, result.meta or {}, suite_rows(Path(tmp)), rows(Path(tmp) / LEAN_AUDIT, dict))
    return out


def compared(run: str, commit: str, since: str, fetch: Callable | None = None, root: Path | None = None, store=None) -> dict:
    """The ended run's verdicts against the recorded checks' of `commit` (`tally`): its Lean report is fetched beside the records
    `read` fetched, and theirs are read from the store (`stored`). Raises when its machine can't be reached."""
    from research import remote, runs

    d = runs.resolve(runs.runs_root(root), run)
    rec = (fetch or remote.fetch)(d, also=(LEAN_AUDIT,))
    if rec.get("last_fetch_error"):
        raise ConnectionError(f"the run's machine is unreachable (rc {rec['last_fetch_error'].get('rc')})")
    mine = verdicts(run, rows(d / "result.json", dict), suite_rows(d), rows(d / LEAN_AUDIT, dict))
    return tally(mine, stored(commit, run, since, store))


class Pod:
    """What a pass does outside its state, on the control pod: git, the launch, the run's records, Slack and the store."""

    def __init__(self, cfg: Settings, log: Callable[[str], None] = log, reg: dict | None = None, api=None, routes=None):
        self.cfg, self.log, self.routes = cfg, log, routes
        self.repo, self.trees = cfg.work / "repo", cfg.work / "trees"
        self.reg = reg if reg is not None else slack.load_registry(slack.registry_path())
        self.api = api if api is not None else slack.client(argparse.Namespace(dry_run=False))
        self.alerts = slack.channel_id_for(self.reg, "agent-alerts")
        self.tags = {name: slack.tag(slack.entry(self.reg, name)) for name in RING_OF.values()}

    def git(self, *args: str, check: bool = True) -> str:
        r = subprocess.run(["git", "-C", str(self.repo), *args], capture_output=True, text=True, timeout=GIT_S,
                           env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
        if check and r.returncode != 0:
            raise subprocess.SubprocessError(f"git {' '.join(args[:2])} exited {r.returncode}: {r.stderr.strip()[-300:]}")
        return r.stdout.strip()

    def prepare(self, day: str) -> tuple[str, Path]:
        """Origin's `main` in the loop's clone, and a worktree of its tip for the night `day`: (the commit, the worktree)."""
        if not (self.repo / "HEAD").exists():
            self.repo.mkdir(parents=True, exist_ok=True)
            self.git("init", "-q", "--bare")
        for key, value in (("remote.origin.url", self.cfg.url), ("credential.helper", self.cfg.helper)):
            if value and self.git("config", "--get", key, check=False) != value:
                self.git("config", key, value)
        self.git("fetch", "-q", "origin", "+refs/heads/main:refs/remotes/origin/main")
        commit = self.git("rev-parse", "--verify", "refs/remotes/origin/main^{commit}")
        tree = self.trees / f"{day}-{commit[:12]}"
        for old in sorted(self.trees.glob("*")):         # what an earlier cleanup left; only the night's own path must be free
            try:
                shutil.rmtree(old)
            except OSError as e:
                if old == tree:
                    raise
                self.log(f"prepare: {old} left in place ({type(e).__name__}: {e})")
        self.git("worktree", "prune")
        self.trees.mkdir(parents=True, exist_ok=True)
        self.git("worktree", "add", "--detach", "--quiet", str(tree), commit)
        return commit, tree

    def cleanup(self, tree: Path) -> None:
        """Remove the worktree `tree`; raises when it is still there."""
        self.git("worktree", "remove", "--force", str(tree), check=False)
        if tree.exists():
            shutil.rmtree(tree)
        self.git("worktree", "prune", check=False)

    def launch(self, argv: list[str], tree: Path) -> tuple[int, str]:
        """`argv` in `tree`, with the tree's own research first and alone on PYTHONPATH, its output in the launch log: (its
        exit code, its output). In a session of its own, so a restart of the loop doesn't end a launch that is under way."""
        env = {**os.environ, "PYTHONPATH": str(tree / "tools" / "research" / "src")}
        if self.cfg.tmp is not None:
            self.cfg.tmp.mkdir(parents=True, exist_ok=True)
            env["TMPDIR"] = str(self.cfg.tmp)
        with open(self.cfg.launch_log, "wb") as f:
            p = subprocess.Popen(argv, cwd=tree, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            try:
                rc = p.wait(timeout=dispatch.LAUNCH_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)             # its own session: check.py, `research run` and its ssh
                try:
                    rc = p.wait(timeout=KILL_S)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    rc = p.wait()
        out = self.launch_output()
        for ln in out.splitlines():
            self.log(f"launch: {ln}")
        return rc, out

    def launch_output(self) -> str:
        return self.cfg.launch_log.read_text(errors="replace") if self.cfg.launch_log.exists() else ""

    def forget_launch(self) -> None:
        self.cfg.launch_log.unlink(missing_ok=True)

    def read(self, run: str) -> Read:
        return read(run, self.cfg)

    def compare(self, run: str, commit: str) -> dict:
        """The run's verdicts against the recorded checks' of `commit`, which were launched after it was made."""
        made = int(self.git("show", "-s", "--format=%ct", commit))
        return compared(run, commit, "r" + time.strftime("%Y%m%d-%H%M%S", time.gmtime(made - BEFORE_S)))

    def post(self, words: str) -> str:
        return slack._post(self.api, self.alerts, "infra", words)

    def ring(self, ts: str, words: str, name: str) -> str:
        """Ring `name`'s inbox thread with the post at `ts`, as the agent watch's relay rings an alert's owner."""
        to, r = (self.routes or slack.Routes()).resolve(name)
        if not r or r.get("closed"):
            return f"{name} has no open inbox: the post stands in #agent-alerts, unrung"
        said, link = slack.plain(words), slack.permalink(self.reg, self.alerts, ts)
        if len(said) + 1 + len(link) > slack.DOORBELL_MAX:      # what a doorbell carries of a post: the link stays inside it
            said = said[:slack.DOORBELL_MAX - len(link) - 2].rstrip() + "\u2026"
        self.api.call("chat.postMessage", channel=r.get("channel") or slack.channel_id_for(self.reg, slack.FRONT_DOOR),
                      thread_ts=r["inbox_ts"], username="router", unfurl_links="false", unfurl_media="false",
                      text=slack.doorbell("infra", ts, ts, f"{said} {link}"))
        return f"rang @{to}"

    def label(self, commit: str, outcome: str, run: str | None) -> str:
        """`nightly_check = outcome` on `commit:<sha>`, pushed to the store's remote as the queue pushes a ready mark."""
        from research.store import vocab
        from research.store.cli import _label_write_through
        from research.store.local import LocalStore

        if why := vocab.check(vocab.NIGHTLY_CHECK, outcome):
            raise ValueError(why)
        store, target = LocalStore(), f"commit:{commit}"
        store.add_label(target, vocab.NIGHTLY_CHECK, outcome, BY, ref=run)
        said, _lost = _label_write_through(store, target)
        return f"{target} {vocab.NIGHTLY_CHECK}={outcome}; {said}"


def load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=1) + "\n")
    tmp.replace(path)


def parse(argv: list[str] | None = None) -> tuple[Settings, argparse.Namespace]:
    ap = argparse.ArgumentParser(prog="nightly_check", description=__doc__.split("\n\n")[0])
    ap.add_argument("--on", required=True, metavar="NODE", help="the machine the check is recorded on")
    ap.add_argument("--pool", required=True, help="the pool on that machine whose slot the check waits for")
    ap.add_argument("--timeout", required=True, type=int, metavar="S", help="the run's timeout, in seconds")
    ap.add_argument("--hour", required=True, type=int, choices=range(24), metavar="H", help="the UTC hour a night starts from")
    ap.add_argument("--work", required=True, type=Path, help="the loop's clone and worktrees, on the pod's own disk")
    ap.add_argument("--state", required=True, type=Path)
    ap.add_argument("--tmp", type=Path, help="TMPDIR of the launch")
    ap.add_argument("--url", default=URL, help="origin")
    ap.add_argument("--credential-helper", default=HELPER, help="git's credential helper for origin ('' for none)")
    ap.add_argument("--every", type=float, default=300.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="say what this pass would do, and do none of it")
    ns = ap.parse_args(argv)
    return Settings(ns.on, ns.pool, ns.timeout, ns.hour, ns.work, ns.state, ns.tmp, ns.url, ns.credential_helper), ns


def main(argv: list[str] | None = None) -> int:
    cfg, ns = parse(argv)
    if ns.dry_run:
        state, now = load_state(cfg.state), time.time()
        night = state.get("night") or {}
        print(f"state {cfg.state}: night {night.get('day')} {night.get('phase')}; last {json.dumps(state.get('last'))}")
        day = due(state, cfg, now)
        print("a night would start now: " + ("no: a first pass after the hour starts none that day" if day and "since" not in state
                                             else day or "no") + f" (once a day, from {cfg.hour:02d}:00Z)")
        print(f"launch, in a worktree of origin/main under {cfg.work}: {shlex.join(command(cfg))}")
        return 0
    pod = Pod(cfg)
    log(f"a night a day from {cfg.hour:02d}:00Z, on {cfg.on} in its {cfg.pool} pool, timeout {cfg.timeout} s; state {cfg.state}; "
        f"the launch, in a worktree of origin/main under {cfg.work}: {shlex.join(command(cfg))}")
    while True:
        state = load_state(cfg.state)
        try:
            step(state, cfg, pod, time.time(), lambda s: save_state(cfg.state, s))
        except (OSError, subprocess.SubprocessError, slack.SlackError, http.client.HTTPException, ValueError, LookupError) as e:
            log(f"pass failed: {type(e).__name__}: {e}")    # the next pass goes on from the state as it was last saved
        if ns.once:
            return 0
        time.sleep(ns.every)


if __name__ == "__main__":
    sys.exit(main())
