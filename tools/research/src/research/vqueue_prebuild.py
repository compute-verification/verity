"""The Lean builds the verified queue's loop starts ahead of the candidates that will hold them, so a candidate's records run
and check compile only what no ready PR's build compiled: `tools/verity/check/lean_prebuild.py` (`SCRIPT`, the loop's own).

While a PR with a Lean change (`vqueue.Ready.lean`) waits ready, merging onto `main` and in no candidate, `Prebuilds.tick`
launches ``lean_prebuild.py MAIN HEAD --base MAIN --on MACHINE --mem-gb auto --repo CLONE --as ci`` in a session of its own
(`landing.detach`) on a machine of ``machines`` (`--prebuild-on`) that runs none, in rank order, a PR never built first:
once per head and `main`'s Lean inputs (`vqueue_slim.lean_key`), so a PR is built again only once `main`'s Lean moved. The
run audits the change in the machine's pool `research`, off the check slots, and its command names the head, so the
launcher of the candidate's records run and check that holds the PR finds its bundles and Lean verdicts however old
(`common.merged_heads`, `lean_audit.shared_bundles`, `check.verdict_packs`): their audit restores every module the build
compiled and reuses the replay of every module it replayed, and a candidate with exactly the built Lean content takes each
package's pass whole. A change that reaches no audited package launches no run (`lean_prebuild.NOTHING`).

Each launch is a `prebuild` line of the loop's log, and its end another, `done`, with its `state` (`STATES`) and the run it
launched; a machine runs one prebuild at a time, counted as running `RUNS_S` from its launch."""
from __future__ import annotations

import os
import re
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from .landing import LAUNCH_LOST_S, detach
from .vqueue import Ready

SCRIPT = Path(__file__).resolve().parents[4] / "tools" / "verity" / "check" / "lean_prebuild.py"
#: how long after its launch a prebuild's run counts as running on its machine: an audit of a change under most of
#: security/Proofs compiles and replays its modules in about an hour
RUNS_S = 2 * 3600
#: the lane its pool slot names
LANE = "ci"
#: a launch's end: its run launched, its change reached no audited package, it launched nothing, or it was lost
STATES = ("launched", "nothing", "failed", "lost")
LAUNCHED = re.compile(r"^lean-prebuild: launched as (r\S+?):", re.M)
NOTHING = "reaches no audited Lean package"


def argv(main: str, head: str, machine: str, repo: Path) -> list[str]:
    """The prebuild of ``head`` merged onto ``main``, read against ``main``, from ``repo``'s commits."""
    return [sys.executable, str(SCRIPT), main, head, "--base", main, "--on", machine, "--mem-gb", "auto", "--repo", str(repo),
            "--as", LANE]


def ended(launch: dict) -> tuple[str, str] | None:
    """(its exit, its output's tail) once the detached launch ``launch`` (`landing.detach`) ended, its exit "lost" once its
    process is gone with no exit written; None while it runs."""
    d = Path(launch["dir"])
    try:
        tail = (d / "log").read_text(errors="replace").strip()[-600:]
    except OSError:
        tail = ""
    if (d / "rc").is_file():
        return (d / "rc").read_text().strip(), tail
    try:
        os.kill(int(launch["pid"]), 0)
    except ProcessLookupError:
        return "lost", tail
    return None


def outcome(rc: str, tail: str) -> tuple[str, str | None]:
    """(its `STATES` state, the run it launched) of a launch that ended with exit ``rc`` and output ``tail``."""
    if rc == "0" and (m := LAUNCHED.search(tail)):
        return "launched", m[1]
    if rc == "0" and NOTHING in tail:
        return "nothing", None
    return ("lost" if rc == "lost" else "failed"), None


class Prebuilds:
    """The prebuilds of ready Lean changes on ``machines``, from the loop's clone ``repo``: ``launch(repo, argv)`` starts one
    detached and returns its record (`landing.detach`'s), ``read`` reads one (`ended`)."""

    def __init__(self, machines: Iterable[str], repo: Path, now: Callable[[], float] = time.time,
                 launch: Callable[[Path, list[str]], dict] | None = None,
                 read: Callable[[dict], tuple[str, str] | None] = ended):
        self.machines, self.repo, self.now, self.read = [m for m in machines if m], Path(repo), now, read
        self.launch = launch or (lambda repo, args: detach(repo, args, {**os.environ, "GIT_TERMINAL_PROMPT": "0"}))

    def tick(self, lines: list[dict], ready: Iterable[Ready], held: set[int], main: str, key: str | None) -> list[dict]:
        """The `prebuild` lines' fields this tick adds to ``lines`` (the log's): the end of each launch that ended (`done`),
        then a launch for each PR of ``ready`` (in rank order) due one, on a machine running none. A PR is due one when it
        has a Lean change, merges onto ``main``, is a member of no candidate (``held``) and has no prebuild at its head
        against `main`'s Lean key ``key``; none is due while ``key`` is unknown."""
        now, out = self.now(), []
        ends = {r["launch"]["dir"]: r for r in lines if r.get("done") and isinstance(r.get("launch"), dict)}
        starts = [r for r in lines if not r.get("done") and isinstance(r.get("launch"), dict)]
        for r in starts:
            if r["launch"]["dir"] in ends:
                continue
            got = self.read(r["launch"])
            if got is None and now - float(r["launch"].get("at") or 0) < LAUNCH_LOST_S:
                continue
            state, run = outcome(*(got or ("lost", "no run id printed within LAUNCH_LOST_S")))
            ends[r["launch"]["dir"]] = end = {**_fields(r), "launch": r["launch"], "state": state, "run": run,
                                              "why": (got or ("", ""))[1] or f"its launch was {state}", "done": True}
            out.append(end)
        busy = {r["machine"] for r in starts
                if (e := ends.get(r["launch"]["dir"])) is None
                or e["state"] == "launched" and now - float(r["launch"].get("at") or 0) < RUNS_S}
        free = [m for m in self.machines if m not in busy]
        if not free or key is None:
            return out
        built = {(r.get("heads") or [None])[0] for r in lines}
        tried = {((r.get("heads") or [None])[0], r.get("lean_key")) for r in lines}
        due = sorted((x for x in ready if x.lean and not x.conflict and x.number not in held and (x.head, key) not in tried),
                     key=lambda x: x.head in built)
        for x, machine in zip(due, free):
            fields = {"number": x.number, "heads": [x.head], "base": main, "machine": machine, "lean_key": key}
            try:
                launch = {**self.launch(self.repo, argv(main, x.head, machine, self.repo)), "at": now}
            except OSError as e:
                out.append({**fields, "state": "failed", "run": None, "why": f"not launched: {e}", "done": True})
                continue
            out.append({**fields, "launch": launch,
                        "why": f"#{x.number}'s Lean change built on {machine} ahead of its candidate"
                               + (", again: `main`'s Lean moved since its last" if x.head in built else "")})
        return out


def _fields(r: dict) -> dict:
    return {k: r.get(k) for k in ("number", "heads", "base", "machine", "lean_key")}
