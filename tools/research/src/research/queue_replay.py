"""`python3 -m research.queue_replay`: the verified queue replayed on its own records, under other rules.

The question it answers: had the queue run under another rule over the same hours, with the same changes marked at the same
times, the same changes at fault and checks as long as they were, how soon would each change have reached `main`, how many
rounds would have been lost, how many check places would it have needed, and how long would `main` have been red.

What is read is what `research queue stats` reads (`queue_table.collect`, over a longer window), and beside it: the fields
of the loop's decision log that reader leaves out (a build whose records run goes first, a failed candidate's suspects), each
version of the loop's flags with the time it took effect, the files each change touches, the test suites of this checkout
with the paths each reads (`tools/verity/check/suites.py`), and how long each suite took in the checks' shards.

`trace` turns that into the replay's input: each change with the hours the queue could take it (its mark, its land verdict,
a hold, a conflict), whether a real check failed on it (`Fault`), and how long real checks took.  `Replay` then steps through
the loop's own tick times.  The policy is the loop's: every tick it builds a `vqueue.Snapshot` of the replayed world and
applies what `vqueue.decide` returns, so the first row (`RULES`) is the loop's policy on its recorded arrivals, and its
distance from what was measured is the replay's error.  The other rows change one thing each (`Rule`).

What the replay assumes, which every table it prints repeats (`ASSUMPTIONS`): owners mark when they did, and fix a change at
fault as long after it left the queue as they did; a check takes as long as a real check of its kind built near that time;
a change fails every check that holds it if a real check failed on it, and each real failure with nobody at fault fails the
first check started after it; a conflict with `main` is found when it was; places are not rationed, their peak is reported.

`collect` does the reading; everything below it is a pure function of the document, so a saved document replays the same
way every time (the draws are seeded).  Stdlib only."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import queue_table as qt
from . import vqueue as vq

#: a check's shards, each of which holds one check place, and the places a records run holds
SHARDS, RECORDS_PLACES = 3, 1
#: the check places a rule's peak is reported against: those there are, and a larger pool
PLACES, PLACES_SOON = 22, 26
#: a Lean source and a lock, as drive's Lean-change test reads them (`queue.lean_change`)
LEAN_SOURCES = (".lean", "lakefile.lean", "lakefile.toml", "lean-toolchain", "lake-manifest.json")
LEAN_LOCK, LEAN_PROOFS_TREE = "lean-audit.json", "security/"
LEAN_SPEC = ("security/Definitions/", "security/Definitions.lean", "security/lakefile.toml", "security/lake-manifest.json",
             "security/lean-toolchain")
#: how many of the real checks nearest in time a replayed check's length is drawn from
NEAREST = 8
#: how long after a real candidate's drop its Lean members still count as pinned, in seconds: the loop builds them again at once
PIN_GRACE_S = 600.0
#: a rerun of one member's failed tasks, in seconds, when nothing else is known of it
RERUN_S = 900.0
#: what a member taken from the queue by another says, in the loop's words (`queue_table.drop_cause` reads them)
TAKEN = {"hold": "held by a reviewer", "withdrawn": "its ready mark was withdrawn by another",
         "fault": "held for the red in a candidate"}

EXTRA_SCRIPT = r'''
import glob, json, os, sys
live, since, wide, pins = sys.argv[1:5]
lines = []
with open(os.path.join(live, "decisions.jsonl"), errors="replace") as f:
    for raw in f:
        try:
            r = json.loads(raw)
        except ValueError:
            continue
        if r.get("at", "") < since or r.get("shadow"):
            continue
        o = r.get("outcome") or {}
        if r.get("action") == "regenerate" or r.get("records_first") or (r.get("action") == "drop" and o):
            lines.append({"at": r["at"], "action": r["action"], "sha": r.get("sha"), "why": (r.get("why") or "")[:200],
                          "records_first": bool(r.get("records_first")), "state": o.get("state"),
                          "suspects": o.get("suspects")})
flags = []
for p in glob.glob(os.path.join(live, "args*")):
    try:
        flags.append([os.path.getmtime(p), open(p).read()])
    except OSError:
        pass
def numbers(path):
    try:
        return sorted({int(w) for line in open(path) for w in line.split("#")[0].split() if w.isdigit()})
    except OSError:
        return []
print(json.dumps({"lines": lines, "flags": sorted(flags), "wide": numbers(wide), "pins": numbers(pins)}))
'''

SUITE_SCRIPT = r'''
import json, sys
out = {}
for rid in sys.argv[1:]:
    rows = []
    for name in ("suites.json", "suites-lean.json"):
        try:
            got = json.load(open("/workspace/research/runs/" + rid + "/" + name))
        except (OSError, ValueError):
            continue
        rows += [[r.get("name"), r.get("seconds")] for r in got if isinstance(r, dict) and r.get("status") in ("passed", "failed")]
    out[rid] = rows
print(json.dumps(out))
'''


# ---- reading ----------------------------------------------------------------------------------------------------------------


def suite_map() -> dict:
    """This checkout's test suites by directory: each one's name, the paths its key reads (`suites.inputs_of`), the suites
    nested in its directory that it leaves out (`suites.dropped`) and whether it reads Markdown; and the runner's files, a
    change to which reruns every suite."""
    spec = importlib.util.spec_from_file_location("verity_check_suites", qt.CHECKOUT / "tools/verity/check/suites.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    pkgs, out = mod.members(), {}
    for s in mod.discover():
        paths = mod.inputs_of(s, pkgs)
        out[s.path] = {"name": s.name, "inputs": paths, "cut": mod.dropped(s, paths), "prose": mod.markdown(s)}
    return {"suites": out, "runner": [*mod.RUNNER, "uv.lock"]}


def soundness_paths() -> list[str]:
    """The paths of the units on the soundness path, from `tests/test_boundaries.py` (`SOUNDNESS_PATH`, `PLACE`)."""
    tree = ast.parse((qt.CHECKOUT / "tests/test_boundaries.py").read_text())
    found = {}
    for node in tree.body:
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(target, ast.Name) and target.id in ("SOUNDNESS_PATH", "PLACE") and node.value is not None:
            try:
                found[target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    place = found.get("PLACE") or {}
    return sorted({f"verity/{p}" for unit in found.get("SOUNDNESS_PATH") or () for p in place.get(unit, ())})


def head_files(head: str) -> list[str] | None:
    """The files a head that never landed changes from where it left `main`, from GitHub's compare (which lists nothing for
    a head `main` holds); None when it gave no answer."""
    try:
        p = subprocess.run(["gh", "api", f"repos/{qt.REPO}/compare/main...{head}", "--jq", "[.files[].filename]"],
                           capture_output=True, text=True, timeout=90)
        got = json.loads(p.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return got if isinstance(got, list) and all(isinstance(f, str) for f in got) else None


def collect(hours: float = 72.0, known: dict | None = None) -> dict:
    """`queue_table.collect` over the last ``hours``, with the replay's own readings under ``replay``.  ``known`` is an
    earlier document: the files of a PR it holds are not asked for again."""
    was, qt.WINDOW_H = qt.WINDOW_H, hours
    try:
        doc = qt.collect(sample_s=1.0)
    finally:
        qt.WINDOW_H = was
    from .vqueue_loop import PINS, WIDE

    lead = datetime.fromtimestamp(doc["now"] - (hours + 24) * 3600, UTC)
    extra, err = qt._remote(qt.CONTROL, EXTRA_SCRIPT, qt.LIVE, f"{lead:%Y-%m-%d}T00:00:00Z", str(WIDE), str(PINS))
    if extra is None:
        doc["errors"].append(f"{qt.CONTROL}: the replay's fields: {err}")
    rep = dict(extra or {"lines": [], "flags": [], "wide": [], "pins": []})
    decisions = doc["control"].get("decisions") or []
    prs = sorted({str(m) for r in decisions if r["action"] == "build" and not r.get("shadow") for m in r.get("members") or []}
                 | set(doc["open"]))
    landed = {h for r in decisions if r["action"] == "advance" and r.get("done") and not r.get("shadow") for h in r.get("heads") or []}
    heads = sorted({f"{pr}@{h}" for r in decisions if r["action"] == "build" and not r.get("shadow")
                    for pr, h in zip(r.get("members") or [], r.get("heads") or []) if h not in landed})
    files = dict(((known or {}).get("replay") or {}).get("files") or {})
    ask = [pr for pr in prs if pr not in files]
    old = [t for t in heads if t not in files]
    with ThreadPoolExecutor(8) as ex:
        files.update({pr: got for pr, got in zip(ask, ex.map(qt.pr_files, ask)) if got is not None})
        files.update({t: got for t, got in zip(old, ex.map(lambda t: head_files(t.partition("@")[2]), old)) if got})
    rep["files"] = {k: files[k] for k in [*prs, *heads] if k in files}
    listed = subprocess.run(["git", "ls-files", f"*{LEAN_LOCK}"], cwd=qt.CHECKOUT, capture_output=True, text=True).stdout
    rep["locks"] = sorted({line.rpartition("/")[0] or "." for line in listed.splitlines()})
    rep.update(suite_map())
    rep["soundness"] = soundness_paths()
    cands = qt.candidates(decisions, qt.window_runs(doc), qt.window_start(doc), tuple(doc.get("off_gate") or qt.OFF_GATE))
    by_node: dict[str, list[str]] = {}
    for c in cands:
        if c.get("verdict_state") == "pass":
            for s in c["shards"]:
                by_node.setdefault(s["node"], []).append(s["id"])
    seconds: dict[str, list[float]] = {}
    with ThreadPoolExecutor(8) as ex:
        for node, (got, err) in zip(by_node, ex.map(lambda kv: qt._remote(kv[0], SUITE_SCRIPT, *kv[1]), by_node.items())):
            if got is None:
                doc["errors"].append(f"{node}: the suites' seconds: {err}")
            for rows in (got or {}).values():
                for name, took in rows:
                    if isinstance(took, (int, float)):
                        seconds.setdefault(name, []).append(took)
    rep["seconds"] = {name: qt.pct(v, .5) for name, v in seconds.items()}
    doc["replay"] = rep
    return doc


def cut(doc: dict, hours: float) -> dict:
    """``doc`` cut to its last ``hours`` and to what the replay reads of it, small enough to keep as a test's fixture: the
    loop's lines of those hours, the shards of their candidates, the labels of their changes with who set each replaced by
    a number, for each change the few of its files that decide its kind and the suites it reruns, and of each suite the
    inputs those files are under."""
    day = doc["now"] - hours * 3600
    rep = doc.get("replay") or {}

    def since(when: str) -> bool:
        return qt.ts(when) >= day

    def short(head: str) -> str:
        return head[:10]

    def line(r: dict, built: set[str]) -> dict:
        keep = {"build": ("base", "members", "heads", "why", "done"), "advance": ("members", "heads", "done"),
                "slim": ("ids", "verdict", "why"), "drop": ("why", "failed"), "conflict": ("number", "heads"),
                "eject": ("number", "heads", "why")}[r["action"]]
        if (r.get("sha") or "")[:12] not in built:
            keep += ("members", "heads")
        out = {k: r[k] for k in ("action", "at", *keep) if r.get(k) not in (None, [], "", False)}
        out.update({k: r[k][:12] for k in ("sha", "base") if k in out or (k == "sha" and r.get(k))})
        if "heads" in out:
            out["heads"] = [short(h) for h in out["heads"]]
        if "why" in out:
            # the reason as its readers read it, without the run or the role it names
            out["why"] = re.sub(r"\br\d{8}-\d{6}-[0-9a-f]{4}\b", "a run", re.sub(r"withdrawn by \S+", "withdrawn by another", out["why"]))[:160]
            assert qt.drop_cause(out["why"]) == qt.drop_cause(r["why"]), r["why"]
        return out

    def steps(shard: dict) -> list[dict]:
        """A shard's steps that say how it went: each that failed, was stopped or has no end, and the last to end on the gate."""
        off = tuple(doc.get("off_gate") or qt.OFF_GATE)
        on = [st for st in shard.get("steps") or [] if st.get("step") not in off]
        odd = [st for st in on if st.get("state") in ("failed", "stopped") or not st.get("end")]
        last = max((st for st in on if st.get("end")), key=lambda st: st["end"], default=None)
        return [{x: st.get(x) for x in ("step", "state", "end")} for st in ([*odd, last] if last is not None and last not in odd else odd)]

    raw = [r for r in doc["control"].get("decisions") or []
           if since(r["at"]) and not r.get("shadow") and r["action"] in ("build", "slim", "drop", "advance", "conflict", "eject")]
    shas = {(r.get("sha") or "")[:12] for r in raw if r["action"] == "build"}
    lines = [line(r, shas) for r in raw if r["action"] != "slim" or r.get("ids") or r.get("verdict")
             or (r.get("why") or "").startswith(("judged", "started"))]
    ids = {i for r in lines for i in r.get("ids") or ()}
    members = {str(m) for r in raw for m in [*(r.get("members") or ()), r.get("number")] if m}
    who: dict = {}
    labels = {f"{t.partition('@')[0]}@{short(t.partition('@')[2])}": [dict(lab, by=who.setdefault(lab.get("by"), len(who) + 1)) for lab in labs]
              for t, labs in doc["labels"].items() if t.partition("@")[0] in members}
    prs = members
    heads = {h for r in lines for h in r.get("heads") or ()} | {t.partition("@")[2] for t in labels}
    restacks = {short(a): short(b) for a, b in (doc.get("restacks") or {}).items()}
    for _ in range(4):
        heads |= {x for a, b in restacks.items() if a in heads or b in heads for x in (a, b)}
    was = Suites(rep, 0.0, 0.0)
    sound, locks = rep.get("soundness") or [], rep.get("locks") or ["."]

    def few(files: list[str]) -> list[str]:
        files = sorted(files)
        want, got = was.touched(frozenset(files)), set(was.whole)
        tests = [lambda f: f in was.runner, lambda f: lean_compiles([f, "x.lean"], locks), lambda f: f.endswith(LEAN_SOURCES),
                 lambda f: bool(qt.LEAN_FILE.search(f)), lambda f: _under(f, sound), lambda f: not f.endswith(".md"),
                 lambda f: "/tests/" not in f"/{f}"]
        out = {next(f for f in files if test(f)) for test in tests if any(test(f) for f in files)}
        for f in files:
            if not want <= (got := got | was.touched(frozenset({*out})) if out else got):
                if was.touched(frozenset({f})) - got:
                    out.add(f)
        return sorted(out or files[:1])

    files = {}
    for k, v in (rep.get("files") or {}).items():
        pr, _, h = k.partition("@")
        if v and pr in prs and (not h or short(h) in heads):
            files[f"{pr}@{short(h)}" if h else pr] = few(v)
            assert was.touched(frozenset(files[f"{pr}@{short(h)}" if h else pr])) == was.touched(frozenset(v)), k
    kept = [f for v in files.values() for f in v]
    # a suite goes by its directory: its seconds are all its name is read for
    suites = {path: {"name": path, "prose": s["prose"], "inputs": [q for q in s["inputs"] if q == "." or any(_under(f, [q]) for f in kept)],
                     "cut": [c for c in s["cut"] if any(_under(f, [c]) for f in kept)]} for path, s in (rep.get("suites") or {}).items()}
    runs = {n: {"runs": [{k: (steps(r) if k == "steps" else v)
                          for k, v in r.items() if k in ("id", "kind", "sha", "shard", "took", "end", "state", "failed", "steps")}
                         for r in d["runs"] if r["id"] in ids or (r.get("kind") == "shard" and r.get("sha") in shas)]}
            for n, d in doc["nodes"].items()}
    flags = sorted(rep.get("flags") or [], key=lambda x: x[0])
    return {"now": doc["now"], "day": f"{datetime.fromtimestamp(day, UTC):%Y-%m-%dT%H:%M:%SZ}", "errors": [], "off_gate": doc.get("off_gate"),
            "control": {"decisions": lines, "ticks": [t for t in doc["control"].get("ticks") or [] if since(t)]},
            "nodes": {n: d for n, d in runs.items() if d["runs"]}, "labels": labels,
            "open": {pr: short(h) for pr, h in doc["open"].items() if pr in prs},
            "restacks": {a: b for a, b in restacks.items() if a in heads},
            "lean": {k: v for k, v in (doc.get("lean") or {}).items() if k in shas},
            "replay": {"lines": [{k: v for k, v in r.items() if v not in (None, [], False)} for r in rep.get("lines") or [] if since(r["at"])],
                       "flags": [f for f in flags if f[0] < day][-1:] + [f for f in flags if f[0] >= day],
                       "wide": rep.get("wide") or [], "pins": rep.get("pins") or [], "files": files, "locks": locks,
                       "suites": suites, "runner": rep.get("runner") or [], "soundness": sound,
                       "seconds": {path: (rep.get("seconds") or {}).get(s["name"]) for path, s in (rep.get("suites") or {}).items()}}}


# ---- the trace --------------------------------------------------------------------------------------------------------------


@dataclass
class Change:
    """One head of a PR the queue could take, with every head a restack made of it."""
    pr: int
    head: str
    spans: list[list]                    # [from, until (None: until it lands), why it ended] while the queue could take it
    rank: float                          # the queue's order: when the loop first built it, else when it could first take it
    files: frozenset[str]
    lean: bool = False                   # its Lean change compiles (`vqueue.Ready.lean`)
    audit: bool = False                  # it changes a Lean input (`queue_table.LEAN_FILE`)
    sound: bool = False                  # it changes a file on the soundness path
    mark: float | None = None            # the mark its real landing counts from
    landed: float | None = None          # when `main` really moved to it
    pins: list[tuple[float, float]] = field(default_factory=list)   # while a real candidate held it: a Lean change counts
                                         # as pinned then (pins are set by hand), and from its last build on if it landed
    conflicts: list[tuple[float, float]] = field(default_factory=list)   # while it did not merge onto `main`
    fault: int | None = None             # the real failure it was at fault for (`Fault`)
    left: float | None = None            # when it last left a real candidate
    after: tuple[int, str] | None = None  # the head at fault this one fixes
    delay: float = 0.0                   # how long after that head left the queue this one could be taken

    @property
    def key(self) -> tuple[int, str]:
        return self.pr, self.head

    @property
    def every(self) -> bool:
        return True

    @property
    def plain(self) -> bool:
        """No Lean input and nothing on the soundness path."""
        return not self.audit and not self.sound

    @property
    def prose(self) -> bool:
        return bool(self.files) and all(f.endswith(".md") for f in self.files)

    @property
    def tests(self) -> bool:
        return bool(self.files) and all("/tests/" in f"/{f}" for f in self.files) and not self.audit


@dataclass
class Fault:
    """A real failed check and the changes blamed for it: named by the loop (an eject, a drop naming a held or ejected
    member), else those whose mark another withdrew or a reviewer held after it, else its members that never landed."""
    id: int
    at: float                            # when the loop dropped the first candidate that failed on it
    changes: list[tuple[int, str]]
    suites: tuple[str, ...]              # the suites that failed first, by directory; none when no node recorded one
    delay: float | None = None           # from that drop until the blamed changes left the queue, None when the loop ejected
    how: str = ""                        # how the blamed changes were found


@dataclass
class Trace:
    start: float
    end: float
    ticks: list[float]
    changes: dict[tuple[int, str], Change]
    faults: dict[int, Fault]
    flags: list[tuple[float, dict]]      # (from when, the loop's flags), oldest first
    hands: list[tuple[float, frozenset[int]]]   # each candidate dropped by hand: when, and its members
    passes: list[tuple[float, bool, float]]     # each real check that passed: its build, whether it changed Lean, seconds
    fails: list[tuple[float, float]]            # each real check that failed: its build, seconds to the loop's drop
    records: list[float]                 # each real records run that passed, in seconds
    records_from: float | None           # when the loop first ran a Lean batch's records before its check
    blame_from: float | None             # when the loop first ejected one member for a failure
    mixed_until: float                   # the loop's last batch that mixed Lean changes with others
    capped: float                        # when its flags first took no Lean change into a batch (`--lean 0`): from then
                                         # until `mixed_until`, the ones it mixed in were those pinned
    flakes: list[float]                  # each real check that failed with no change at fault: when it was built
    wide: frozenset[int]
    suites: dict
    measured: dict


def _flags(text: str) -> dict:
    """The loop's flags as its parser reads its arguments file: the last value of each, comments left out."""
    out: dict = {}
    for line in text.splitlines():
        if not line.lstrip().startswith("#"):
            for name, value in re.findall(r"(?:^|\s)--(batch|depth|lean)[ =](\d+)", line):
                out[name] = int(value)
            if re.search(r"(?:^|\s)--no-records\b", line):
                out["records"] = False
    return out


def lean_compiles(files, locks) -> bool:
    """Drive's Lean-change test on a list of files, for a change that compiles: a lock, the spec or a Lean source outside
    the analysis tree in a package with a lock (``locks``: their directories) changes, and a Lean source is among the files."""
    files = list(files)
    change = any(f.rpartition("/")[2] == LEAN_LOCK or (f.endswith(LEAN_SOURCES) and f.startswith(LEAN_SPEC)) for f in files) \
        or any(f.endswith(LEAN_SOURCES) and not f.startswith(LEAN_PROOFS_TREE) and _under(f, locks) for f in files)
    return change and any(f.endswith(LEAN_SOURCES) for f in files)


def _under(path: str, prefixes) -> bool:
    return any(p == "." or path == p or path.startswith(p.rstrip("/") + "/") for p in prefixes)


class Suites:
    """The suites a set of files reruns, as `suites.touched` reads paths, and what a rerun of some of them costs."""

    def __init__(self, rep: dict, launch_s: float, check_s: float):
        self.by_path = rep.get("suites") or {}
        self.runner = set(rep.get("runner") or ())
        self.seconds = {path: (rep.get("seconds") or {}).get(s["name"]) for path, s in self.by_path.items()}
        self.whole = frozenset(p for p, s in self.by_path.items() if "." in s["inputs"])
        self.index: dict[str, list[str]] = {}
        for path, s in self.by_path.items():
            for p in s["inputs"]:
                self.index.setdefault(p.rstrip("/"), []).append(path)
        known = [v for v in self.seconds.values() if v]
        self.launch_s = launch_s
        #: how many suites' seconds a check gets through in one second of its own, from a whole check's length
        self.speed = max(1.0, sum(known) / max(60.0, check_s - launch_s)) if known else 1.0
        self._touched: dict[frozenset[str], frozenset[str]] = {}

    def touched(self, files: frozenset[str]) -> frozenset[str]:
        if files not in self._touched:
            if any(f in self.runner for f in files):
                got = frozenset(self.by_path)
            else:
                found = set(self.whole)
                for f in files:
                    p, near = f, set()
                    while True:
                        near.update(self.index.get(p, ()))
                        if "/" not in p:
                            break
                        p = p.rpartition("/")[0]
                    for path in near - found:
                        s = self.by_path[path]
                        inner = [c for c in s["cut"] if _under(f, [c])]
                        if (s["prose"] or not f.endswith(".md")) and _under(f, [q for q in s["inputs"] if not inner or _under(q, inner)]):
                            found.add(path)
                got = frozenset(found)
            self._touched[files] = got
        return self._touched[files]

    def wall(self, suites) -> float:
        """Seconds a check of only ``suites`` takes: its launch, then the longer of its longest suite and all of them at
        the speed a whole check gets through suites."""
        took = [self.seconds.get(p) or 60.0 for p in suites]
        return self.launch_s + (max(max(took), sum(took) / self.speed) if took else 0.0)

    def inputs(self, path: str) -> frozenset[str] | None:
        """What a failure of the suite at ``path`` blames, as `vqueue.meets` reads inputs; None for a suite this checkout
        doesn't have.  The suite that reads the whole tree blames nobody by its inputs: the loop names its failing test's."""
        s = self.by_path.get(path)
        return None if s is None else frozenset() if "." in s["inputs"] else frozenset(s["inputs"])


def trace(doc: dict) -> Trace:
    """The replay's input from a collected document (`collect`)."""
    rep = doc.get("replay") or {}
    decisions = [r for r in doc["control"].get("decisions") or [] if not r.get("shadow")]
    labels, restacks = doc["labels"], doc.get("restacks") or {}
    cands = qt.candidates(decisions, qt.window_runs(doc), qt.window_start(doc), tuple(doc.get("off_gate") or qt.OFF_GATE))
    moved = [qt.ts(r["at"]) for r in decisions if r["action"] == "advance" and r.get("done")]
    first = next((c for c in cands if c["outcome"] == "green"), None)
    start = first["build"] if first else qt.window_start(doc)
    end = doc["now"]

    def root(head: str) -> str:
        return qt.chain(restacks, head)[-1]

    built: dict[tuple[int, str], list[float]] = {}
    gone: dict[tuple[int, str], float] = {}
    landed: dict[tuple[int, str], float] = {}
    heads: dict[tuple[int, str], list[str]] = {}
    members_of: dict[str, list[tuple[int, str]]] = {}
    for r in decisions:
        at = qt.ts(r["at"])
        pairs = [(int(pr), root(h)) for pr, h in zip(r.get("members") or [], r.get("heads") or [])]
        for (pr, _), h in zip(pairs, r.get("heads") or []):
            if h not in heads.setdefault((pr, root(h)), []):
                heads[pr, root(h)].append(h)
        if r["action"] == "build":
            members_of[(r.get("sha") or "")[:12]] = pairs
            for k in pairs:
                built.setdefault(k, []).append(at)
        elif r["action"] == "advance" and r.get("done"):
            landed.update({k: at for k in pairs})
        elif r["action"] == "drop":
            for k in members_of.get((r.get("sha") or "")[:12], ()):
                gone[k] = max(gone.get(k, 0), at)
    for target in labels:
        pr, _, h = target.partition("@")
        if h not in heads.setdefault((int(pr), root(h)), []):
            heads[int(pr), root(h)].append(h)

    files_of = {k: frozenset(f) for k, f in (rep.get("files") or {}).items()}
    sound, locks = rep.get("soundness") or [], rep.get("locks") or ["."]
    changes: dict[tuple[int, str], Change] = {}
    closed: dict[tuple[int, str], list[tuple[float, str]]] = {}
    for key, hs in heads.items():
        pr = key[0]
        labs = sorted({(lab["ts"], lab["key"], str(lab["value"]).lower(), lab.get("by")) for h in hs
                       for lab in labels.get(f"{pr}@{h}") or []})
        owner, marked, held, land, open_at, spans, ends = None, False, False, None, None, [], []
        for when, name, value, by in labs:
            at = qt.ts(when)
            if name == "ready":
                marked = value == "true"
                owner = owner or (by if marked else None)
                why = "owner" if by == owner else "withdrawn"
            else:
                held = value == "hold"
                land = land or (at if value == "land" else None)
                why = "hold"
            if marked and not held and open_at is None:
                open_at = at
            elif open_at is not None and not (marked and not held):
                spans.append([open_at, at, why])
                ends.append((at, why))
                open_at = None
        if open_at is not None:
            spans.append([open_at, None, ""])
        builds = sorted(built.get(key, ()))
        takes = []
        for a, b, why in spans:
            before = next((x for x in builds if x >= a and (b is None or x < b)), None)
            when = min(max(a, land) if land is not None else float("inf"), before if before is not None else float("inf"))
            if when != float("inf") and (b is None or when < b):
                takes.append([when, b, why])
        if key in landed and not takes:
            takes.append([min(builds or [landed[key]]), None, ""])
        elif key in landed:
            takes[-1][1] = None
        if not takes:
            continue
        f = next((files_of[f"{pr}@{h}"] for h in hs if f"{pr}@{h}" in files_of), None) or files_of.get(str(pr), frozenset())
        # a change the loop never built comes after every one it did: what kept it back (the priority file) is not kept
        ch = Change(pr, hs[0], takes, builds[0] if builds else end + takes[0][0], f, lean_compiles(f, locks),
                    any(qt.LEAN_FILE.search(x) for x in f), any(_under(x, sound) for x in f),
                    landed=landed.get(key), left=gone.get(key))
        ch.head = key[1]
        if key in landed:
            ch.mark = qt.mark_at([lab for h in hs for lab in labels.get(f"{pr}@{h}") or []], landed[key])
        changes[key] = ch
        closed[key] = ends
    held_at: dict[str, float] = {}
    for r in decisions:
        sha = (r.get("sha") or "")[:12]
        if r["action"] == "build":
            held_at[sha] = qt.ts(r["at"])
        elif sha in held_at and (r["action"] == "drop" or (r["action"] == "advance" and r.get("done"))):
            for k in members_of.get(sha, ()):
                if k in changes:
                    by_hand = r["action"] == "drop" and qt.drop_cause(r.get("why") or "") == "by hand"
                    changes[k].pins.append((held_at[sha], float("inf") if r["action"] == "advance" else
                                            qt.ts(r["at"]) + (0.0 if by_hand else PIN_GRACE_S)))
            del held_at[sha]
    for sha, at in held_at.items():
        for k in members_of.get(sha, ()):
            if k in changes:
                changes[k].pins.append((at, float("inf")))
    for r in decisions:
        if r["action"] == "conflict":
            at = qt.ts(r["at"])
            for h in r.get("heads") or []:
                ch = changes.get((int(r.get("number") or 0), root(h)))
                if ch is not None and not any(a <= at < b for a, b in ch.conflicts):
                    ch.conflicts.append((at, next((x for x in sorted(built.get(ch.key, ())) if x > at), float("inf"))))

    # the changes at fault
    ejects = [(qt.ts(r["at"]), (int(r["number"]), root(h))) for r in decisions if r["action"] == "eject"
              for h in (r.get("heads") or [])[:1]]
    faults: dict[int, Fault] = {}
    clean: list[float] = []
    for c in cands:
        failed = (c["outcome"] == "red" and "infrastructure" not in (c.get("cause"), c.get("kind"))) or c.get("verdict_state") == "fail"
        if not failed or c.get("cause") == "dropped base" or c["build"] < start:
            continue
        at, mine = c.get("dropped") or c.get("verdict") or c["build"], members_of.get(c["sha"], [])
        if any(changes[k].fault is not None for k in mine if k in changes):
            continue
        named = [k for k in mine if re.match(rf"#{k[0]}\b", c.get("why") or "")]
        named += [k for when, k in ejects if k in mine and abs(when - at) < 900 and k not in named]
        ejected = bool(named) and any(k in mine and abs(when - at) < 900 for when, k in ejects)
        how = "named by the loop"
        if not named:
            how = "found by a hold or a mark another withdrew"
            # a mark another withdrew, or a hold, soon after: not the loop's own eject for a later failure
            named = [k for k in mine if any(at - 1800 <= when <= at + 3 * 3600 and why != "owner"
                                            and not any(e == k and abs(when - x) < 300 for x, e in ejects)
                                            for when, why in closed.get(k, ()))]
        if not named:
            how = "found by never landing"
            never = [k for k in mine if k not in landed]
            named = never if 0 < len(never) <= max(3, len(mine) // 4) else []
        named = [k for k in named if k in changes and k not in landed]
        if not named:
            clean.append(c["build"])
            continue
        suites = tuple(dict.fromkeys(f[0] for s in c["shards"] for f in sorted(s.get("failed") or [], key=lambda f: f[1] or 0) if f[1]))
        out = [when for k in named for when, why in closed.get(k, ()) if when >= c["build"] and why != "owner"]
        fault = Fault(len(faults), at, named, suites[:1], None if ejected else max(0.0, min(out) - at) if out else None, how)
        faults[fault.id] = fault
        for k in named:
            changes[k].fault = fault.id
            changes[k].spans = [[changes[k].spans[0][0], None, ""]]
    # a PR closed since that the loop did not land (merged another way, or given up): the queue could take it until it last
    # left a real candidate
    for ch in changes.values():
        if ch.landed is None and ch.fault is None and str(ch.pr) not in doc["open"]:
            cut = ch.left or 0.0
            ch.spans = [[a, cut if b is None or b > cut else b, why if b is not None and b <= cut else "owner"]
                        for a, b, why in ch.spans if a < cut]
    changes = {k: ch for k, ch in changes.items() if ch.spans}
    delays = [f.delay for f in faults.values() if f.delay is not None]
    for f in faults.values():
        if f.delay is None and not any(k == e for _, e in ejects for k in f.changes):
            f.delay = qt.pct(delays, .5) or 0.0
    # a later head of a PR: the one before it stops being taken when it can be; after a head at fault, it is its fix
    by_pr: dict[int, list[Change]] = {}
    for ch in changes.values():
        by_pr.setdefault(ch.pr, []).append(ch)
    for chs in by_pr.values():
        chs.sort(key=lambda ch: ch.spans[0][0])
        for old, new in zip(chs, chs[1:]):
            if old.fault is not None:
                new.after, new.delay = old.key, max(0.0, new.spans[0][0] - (old.left or faults[old.fault].at))
            elif old.landed is None:
                cut = new.spans[0][0]
                old.spans = [[a, cut if b is None or b > cut else b, why if b is not None and b <= cut else "owner"]
                             for a, b, why in old.spans if a < cut]

    lean = doc.get("lean") or {}
    passes = [(c["build"], bool(qt.lean_count(lean, c["sha"])), c["verdict"] - c["build"]) for c in cands
              if c.get("verdict_state") == "pass" and c["build"] >= start]
    fails = [(c["build"], (c.get("dropped") or c["verdict"]) - c["build"]) for c in cands
             if c["build"] >= start and c.get("verdict_state") == "fail"]
    lines = rep.get("lines") or []
    began = {r["sha"]: qt.ts(r["at"]) for r in reversed(lines) if r["action"] == "regenerate" and r["why"] == "started"}
    records = [qt.ts(r["at"]) - began[r["sha"]] for r in lines
               if r["action"] == "regenerate" and r["why"] == "judged pass" and r["sha"] in began]
    first_records = [qt.ts(r["at"]) for r in lines if r.get("records_first")]
    stamped = [(at, _flags(text)) for at, text in sorted(rep.get("flags") or [], key=lambda x: x[0])]
    capped = min((at for at, f in stamped if f.get("lean") == 0 and at >= start), default=end)
    mixes = [qt.ts(r["at"]) for r in decisions if r["action"] == "build" and start <= qt.ts(r["at"])
             and 0 < sum(changes[k].lean for k in members_of[(r.get("sha") or "")[:12]] if k in changes) < len(r.get("members") or [])]
    launch = qt.pct([max(s["took"] for s in c["shards"]) - c["build"] for c in cands
                     if c["shards"] and all(s.get("took") for s in c["shards"]) and c["build"] >= start], .5) or 540.0
    plain = qt.pct([s for _, lean_change, s in passes if not lean_change], .5) or 3000.0
    parts = [p for p in qt.parts(doc) if start <= p["landed"] <= end]
    lost: dict[str, int] = {}
    for c in cands:
        if c["build"] >= start and c["outcome"] in ("red", "dropped"):
            lost[c["cause"]] = lost.get(c["cause"], 0) + len(c["members"])
    measured = {"landed": len(parts), "minutes": sorted(p["total"] for p in parts), "lost": lost,
                "built": sum(1 for c in cands if c["build"] >= start), "moves": sum(1 for m in moved if start <= m <= end),
                "reds": sum(1 for c in cands if c["build"] >= start and c["outcome"] == "red")}
    return Trace(start, end, [t for t in map(qt.ts, doc["control"].get("ticks") or []) if start <= t <= end], changes, faults,
                 stamped, [(c["dropped"], frozenset(c["members"])) for c in cands if c.get("cause") == "by hand" and c["dropped"] >= start],
                 passes, fails, records, min(first_records, default=None), min((when for when, _ in ejects), default=None),
                 max(mixes, default=start), capped, sorted(clean), frozenset(rep.get("wide") or ()), Suites(rep, launch, plain), measured)


def gated(tr: Trace, f: Fault) -> bool:
    """Whether a run of the suites each blamed change reruns would have shown the failure ``f`` before the mark: the suite
    that failed first is known and reads files of every change blamed.  (Not known from the record: whether it fails without
    the other members of its candidate.)"""
    return bool(f.suites) and all(set(f.suites) <= tr.suites.touched(tr.changes[k].files) for k in f.changes)


# ---- the rules --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """One row: what the replayed loop does differently from the loop as it runs now."""
    name: str
    backtrack: str = ""            # what is undone when the rule goes wrong
    recorded: bool = False         # the loop's flags, its blame and the drops by hand as they were at each time; else
    batch: int = 32                # these flags throughout, the records run before a Lean batch's check, and the loop
    depth: int = 6                 # naming one member for a failure whose suite reads no other member's files
    lean: int = 0
    beside: bool = False           # plain batches are built beside a Lean batch's pending records run
    keep: bool = False             # a failed check removes only the change at fault, and what is rebuilt keeps its passes
    keep_all: bool = False         # under `keep`, so does what is rebuilt when a failure names nobody or several, or when
                                   # others take a member from the queue; else those candidates are dropped as today
    side: int = 0                  # so many batches checked beside each other, each on `main`, and no chain
    alone: bool = False            # each candidate above the lowest is also checked alone on `main`
    first: str = ""                # the changes that land on their mark and verdict and are checked after: plain | prose | tests
    period: float = 0.0            # a candidate is built every so many seconds whatever is below it, and no depth
    plain_first: bool = False      # with nothing in check, the plain batch is built before a Lean batch
    after: int = 2                 # under `first`, so many checks of `main` at once
    gate: bool = False             # a change is marked only once the suites it reruns pass on it merged onto `main`: a real
                                   # failure in a suite that reads the files of each change blamed for it never reaches the queue
    overlap: bool = False          # a Lean batch's check starts with its records run and stands when the run ends
    back: bool = False             # under `first`, a failure naming several changes reverts each, back to the queue's check;
                                   # else the change at fault stays on `main` as long as it really stayed in the queue
    pin_all: bool = True           # every Lean change the queue may take is pinned; else one is while a real candidate held it


@dataclass
class _Cand:
    sha: str
    base: str
    changes: list[Change]
    built: float
    start: float | None = None           # when its check started; None while it waits for its records run
    pass_at: float = float("inf")
    fail_at: float | None = None
    faults: set = field(default_factory=set)
    records_at: float | None = None
    reruns: int = 0
    twin: tuple[float, float, str] | None = None   # its check alone on `main`: started, passes, on which `main`
    left: tuple[float, str] | None = None          # when it left the queue, and why ("" when `main` moved to it)
    union: float | None = None                     # beside others: when its union with what landed since may land
    flake: float | None = None                     # the real failure with nobody at fault that it stands for
    counted: float = 0.0                           # its check's places are counted up to here already (a kept check)

    @property
    def members(self) -> tuple[int, ...]:
        return tuple(ch.pr for ch in self.changes)

    @property
    def heads(self) -> tuple[str, ...]:
        return tuple(ch.head for ch in self.changes)


class Replay:
    """One run of the queue over a trace under a rule; ``seed`` fixes its draws."""

    def __init__(self, tr: Trace, rule: Rule, seed: int = 0):
        self.tr, self.rule, self.rng = tr, rule, random.Random(seed)
        self.t = tr.start
        self.main = "main"
        self.cands: dict[str, _Cand] = {}
        self.side: list[_Cand] = []                    # Lean batches in their records run, beside the chain
        self.history: list[vq.Outcome] = []
        self.built: list[_Cand] = []                   # every candidate built, in order
        self.landed: dict[tuple[int, str], float] = {}
        self.out: dict[tuple[int, str], tuple[float, str]] = {}
        self.arrive: dict[tuple[int, str], float] = {}
        self.removal: dict[int, float] = {}            # a fault's changes leave the queue by hand at this time
        self.lost: dict[str, int] = {}
        self.places: list[tuple[float, float, int]] = []
        self.counts = {"built": 0, "reds": 0, "moves": 0, "rechecked": 0, "unions": 0, "reverts": 0, "on_red": 0, "false_reds": 0,
                       "same_file": 0}
        self.union_s: list[float] = []
        self.red: list[tuple[float, float]] = []       # while `main` held a change at fault
        self.on_main: list[tuple[float, Change]] = []  # changes landed before their check, not yet in one
        self.checked: set[tuple[int, str]] = set()     # changes reverted as suspects: checked before they land again
        self.red_until = -float("inf")                 # a change at fault known to be on `main` until then
        self.post: list[dict] = []
        self.last_build = -float("inf")
        self.n = 0
        self.hands = sorted(tr.hands)
        self.flakes = sorted(tr.flakes)
        self.passes = {k: sorted(x for x in tr.passes if x[1] == k) or sorted(tr.passes) for k in (False, True)}
        self.fails = sorted(tr.fails)
        self.next = {ch.after: ch for ch in tr.changes.values() if ch.after is not None}
        for f in tr.faults.values() if rule.gate else ():
            if gated(tr, f):
                for key in f.changes:
                    # the head at fault is never marked; its fix is, when it really was
                    self.out[key] = (tr.start, "gate")
                    self.arrive[key] = tr.changes[key].left or f.at

    # -- the world at a tick

    def flags(self) -> dict:
        r = self.rule
        if not r.recorded:
            return {"batch": r.batch, "depth": 99 if r.period else r.depth, "lean": r.lean, "records": not r.beside}
        now = {"batch": 16, "depth": 1, "lean": 3}
        for at, f in self.tr.flags:
            if at <= self.t:
                now.update(f)
        now["records"] = self.tr.records_from is not None and self.t >= self.tr.records_from and now.get("records", True)
        return now

    def takes(self, ch: Change) -> bool:
        """Whether the queue could take ``ch`` now: marked with a land verdict, not held, not landed, not at fault and out."""
        if ch.key in self.landed or ch.key in self.out:
            return False
        shift = 0.0
        if ch.after is not None:
            if ch.after not in self.arrive:
                return False
            shift = self.arrive[ch.after] + ch.delay - ch.spans[0][0]
        return any(a + shift <= self.t and (b is None or self.t < b + shift) for a, b, _ in ch.spans)

    def ready(self) -> dict[int, Change]:
        out: dict[int, Change] = {}
        for ch in self.tr.changes.values():
            if self.takes(ch) and (ch.pr not in out or out[ch.pr].spans[0][0] < ch.spans[0][0]):
                out[ch.pr] = ch
        return out

    def _files(self, ch: Change) -> frozenset[str]:
        return ch.files | {f"fault/{ch.fault}"} if ch.fault is not None else ch.files

    def _why_gone(self, ch: Change) -> str:
        if ch.key in self.out:
            return self.out[ch.key][1]
        ended = [(b, why) for _, b, why in ch.spans if b is not None and b <= self.t]
        return max(ended)[1] if ended else "owner"

    def _draw(self, rows: list[tuple], at: float) -> float:
        near = sorted(rows, key=lambda x: abs(x[0] - at))[:NEAREST]
        return self.rng.choice(near)[-1] if near else 3600.0

    def _blame(self, fault: int) -> frozenset[str]:
        """The inputs a failure on ``fault`` names: with the suite that failed known and the loop naming one member, that
        suite's; else every member's."""
        own = frozenset({f"fault/{fault}"})
        if self.rule.recorded and (self.tr.blame_from is None or self.t < self.tr.blame_from):
            return own | {"**"}
        named = [self.tr.suites.inputs(p) for p in self.tr.faults[fault].suites]
        return own | {"**"} if not named or None in named else own.union(*named)

    def _verdict(self, c: _Cand) -> vq.Verdict:
        if c.fail_at is not None and self.t >= c.fail_at:
            failed = tuple(vq.Failed(f"fault-{f}", f"{f}@{c.sha}", self._blame(f) if isinstance(f, int) else frozenset({f}),
                                     tests=frozenset({str(f)}), base_tests=frozenset()) for f in sorted(c.faults, key=str))
            return vq.Verdict(vq.FAIL, failed, ended=self.t >= c.pass_at)
        return vq.Verdict(vq.PASS, ended=True) if self.t >= c.pass_at else vq.Verdict()

    def _candidate(self, c: _Cand) -> vq.Candidate:
        if c.start is None:
            records = vq.RecordsRun(vq.PASS if self.t >= c.records_at else vq.PENDING)
            return vq.Candidate(c.sha, c.base, c.members, c.heads, records=records, waits=True)
        return vq.Candidate(c.sha, c.base, c.members, c.heads, self._verdict(c), reruns=c.reruns)

    def snapshot(self, cands: list[_Cand], main: str, depth: int | None = None, busy=(), pins: bool = True) -> tuple[vq.Snapshot, dict]:
        f, ready = self.flags(), self.ready()
        held = {ch.key for c in cands for ch in c.changes}
        order = sorted((ch for ch in ready.values() if ch.pr not in busy), key=lambda ch: (ch.rank, ch.pr))
        mixed = self.rule.recorded and self.t <= self.tr.mixed_until
        if mixed:
            # the loop then took Lean changes into any batch, as changes with none: up to its cap, the first so many in
            # order; once the cap was none, those pinned then
            over = [ch for ch in order if ch.lean and ch.key not in held]
            over = over[f["lean"]:] if self.t < self.tr.capped else [ch for ch in over if not any(a <= self.t < b for a, b in ch.pins)]
            order, pins = [ch for ch in order if ch not in over], False
        rd = tuple(vq.Ready(ch.pr, ch.head, self._files(ch), ch.lean and not mixed, i,
                            ch.key not in held and any(a <= self.t < b for a, b in ch.conflicts)) for i, ch in enumerate(order))
        flying, left, pulled = [], [], set()
        for c in cands:
            for ch in c.changes:
                if ready.get(ch.pr) is not ch:
                    flying.append(vq.Ready(ch.pr, ch.head, self._files(ch), ch.lean and not mixed))
                    if (why := self._why_gone(ch)) in TAKEN:
                        pulled.add(ch.pr)
                        left.append((ch.pr, ch.head, TAKEN[why]))
        # as it ran, a Lean change was pinned by hand, about when a real candidate took it; now every one the queue may take is
        pinned = frozenset(ch.pr for ch in order if ch.lean and (self.rule.pin_all or any(a <= self.t < b for a, b in ch.pins))) \
            if pins else frozenset()
        snap = vq.Snapshot(main, rd, tuple(self._candidate(c) for c in cands), tuple(self.history), f["batch"],
                           f["depth"] if depth is None else depth, lean=f["lean"], now=self.t, records=f["records"],
                           left=tuple(left), wide=self.tr.wide, pinned=pinned, pulled=frozenset(pulled), flying=tuple(flying))
        return snap, ready

    # -- what happens to a candidate

    def _use(self, start: float | None, end: float, n: int) -> None:
        if start is not None and end > start:
            self.places.append((start, end, n))

    def _check(self, c: _Cand, at: float | None = None) -> None:
        """Start ``c``'s check: it fails if it holds a change at fault, or if it is the first check started since a real
        check failed with nobody at fault."""
        c.start = self.t if at is None else at
        c.pass_at = c.start + self._draw(self.passes[any(ch.audit for ch in c.changes)], c.start)
        c.faults = {ch.fault for ch in c.changes if ch.fault is not None}
        if not c.faults and self.flakes and self.flakes[0] <= c.start:
            c.faults, c.flake = {f"flake-{c.sha}"}, self.flakes.pop(0)
        c.fail_at = c.start + min(self._draw(self.fails, c.start), c.pass_at - c.start) if c.faults else None

    def _new(self, changes: list[Change], base: str) -> _Cand:
        self.n += 1
        self.counts["built"] += 1
        self.last_build = self.t
        self.built.append(_Cand(f"c{self.n:04d}", base, changes, self.t))
        return self.built[-1]

    def _busy(self) -> int:
        """The check places in use now."""
        n = SHARDS * len(self.post)
        for c in [*self.cands.values(), *self.side]:
            if c.start is None:
                n += RECORDS_PLACES
            elif self.t < min(c.pass_at, c.fail_at or float("inf")):
                n += SHARDS
            if c.twin is not None and self.t < c.twin[1]:
                n += SHARDS
        return n

    def _leave(self, c: _Cand, cause: str | None) -> None:
        """``c`` leaves the queue: its places are freed, and with a ``cause`` its members' round is lost."""
        self.cands.pop(c.sha, None)
        c.left = (self.t, cause or "")
        if c.flake is not None and cause != "red" and any(isinstance(f, str) for f in c.faults):
            self.flakes.insert(0, c.flake)        # it left before its failure showed: the next check takes it
        if c.start is None:
            self._use(c.built, min(self.t, c.records_at), RECORDS_PLACES)
        else:
            if c.records_at is not None and not c.counted:
                self._use(c.built, c.records_at, RECORDS_PLACES)
            self._use(max(c.start, c.counted), min(self.t, c.pass_at, c.fail_at or float("inf")), SHARDS)
        if c.twin is not None:
            self._use(c.twin[0], min(self.t, c.twin[1]), SHARDS)
        if cause is not None:
            self.lost[cause] = self.lost.get(cause, 0) + len(c.changes)

    def _land(self, ch: Change) -> None:
        self.landed[ch.key] = self.t

    def _gone(self, key: tuple[int, str], why: str) -> None:
        """A change at fault leaves the queue; its fix arrives as long after as it really did."""
        if key not in self.out:
            self.out[key] = (self.t, why)
            self.arrive[key] = self.t

    def _above(self, sha: str) -> list[_Cand]:
        out, top = [], sha
        while nxt := next((c for c in self.cands.values() if c.base == top), None):
            out.append(nxt)
            top = nxt.sha
        return out

    # -- the loop's actions

    def apply(self, acts: list, ready: dict[int, Change], on_main=None) -> None:
        ejected = [a for a in acts if isinstance(a, vq.Eject)]
        failed = next((a for a in acts if isinstance(a, vq.Drop) and a.outcome is not None and a.outcome.state == vq.FAIL), None)
        saved: set[str] = set()
        if failed is not None and failed.sha in self.cands:
            self.counts["reds"] += 1
            f = self.cands[failed.sha]
            for fault in (x for x in f.faults if isinstance(x, int)):
                if not any((e.number, e.head) in self.tr.faults[fault].changes for e in ejected):
                    self.removal.setdefault(fault, self.t + (self.tr.faults[fault].delay or 0.0))
            if self.rule.keep and (ejected or self.rule.keep_all):
                # with nobody named, every member of the failed candidate goes back to the queue, and those above it go on
                saved = self._recheck(f, {(e.number, e.head) for e in ejected} or {ch.key for ch in f.changes}, "red")
            elif self.rule.alone:
                saved = self._switch(f)
        elif self.rule.keep_all:
            # a candidate dropped for a member others took from the queue goes on without it, as do those above it
            for a in acts:
                if isinstance(a, vq.Drop) and a.sha in self.cands and (cause := qt.drop_cause(a.why)) in ("hold", "withdrawn mark"):
                    c = self.cands[a.sha]
                    pulled = {ch.key for ch in c.changes if ready.get(ch.pr) is not ch}
                    if pulled:
                        saved = self._recheck(c, pulled, cause)
                    break
        for a in acts:
            if isinstance(a, vq.Eject):
                self._gone((a.number, a.head), "fault")
            elif isinstance(a, vq.Advance) and a.sha in self.cands:
                c = self.cands[a.sha]
                if on_main is not None and not on_main(c):
                    saved.add(a.sha)
                    continue
                self.main = c.sha
                self.counts["moves"] += 1
                for ch in c.changes:
                    self._land(ch)
                self._leave(c, None)
                self._moved()
            elif isinstance(a, vq.Drop) and a.sha in self.cands and a.sha not in saved:
                if a.outcome is not None:
                    self.history.append(a.outcome)
                self._leave(self.cands[a.sha], qt.drop_cause(a.why))
            elif isinstance(a, vq.Drop) and a.outcome is not None and a.sha in saved and a.outcome.state != vq.PASS:
                self.history.append(a.outcome)
            elif isinstance(a, vq.Build):
                # what was kept is on the chain again: the batch the loop would build in its place is not built
                if saved or (self.rule.period and self.t - self.last_build < self.rule.period):
                    continue
                c = self._new([ready[m] for m in a.members], a.base)
                if a.records or (self.rule.side and any(ch.lean for ch in c.changes)):
                    c.records_at = self.t + (self.rng.choice(self.tr.records) if self.tr.records else 3600.0)
                elif self.rule.overlap and any(ch.lean for ch in c.changes):
                    self._check(c)
                    took = self.rng.choice(self.tr.records) if self.tr.records else 3600.0
                    self._use(self.t, self.t + took, RECORDS_PLACES)
                    c.pass_at = max(c.pass_at, self.t + took)
                elif self.rule.beside and any(ch.lean for ch in c.changes):
                    c.records_at = self.t + (self.rng.choice(self.tr.records) if self.tr.records else 3600.0)
                    self.side.append(c)
                    continue
                else:
                    self._check(c)
                    if self.rule.alone and c.base != self.main and self._busy() + 2 * SHARDS <= PLACES:
                        c.twin = (self.t, self.t + c.pass_at - c.start, self.main)
                self.cands[c.sha] = c
            elif isinstance(a, vq.Commit) and a.sha in self.cands:
                self._check(self.cands[a.sha])
            elif isinstance(a, vq.Rerun) and a.sha in self.cands:
                c = self.cands[a.sha]
                c.reruns, c.faults = c.reruns + 1, {x for x in c.faults if isinstance(x, int)}
                c.fail_at = self.t + RERUN_S if c.faults else None
                c.pass_at = max(c.pass_at, self.t + RERUN_S)

    def _moved(self) -> None:
        """`main` moved: a check alone on the old `main` says nothing of the new one, so each is started again."""
        if self.rule.alone:
            for c in self.cands.values():
                if c.twin is not None and c.base != self.main:
                    self._use(c.twin[0], min(self.t, c.twin[1]), SHARDS)
                    c.twin = None
                    if self._busy() + SHARDS <= PLACES:
                        c.twin = (self.t, self.t + self._draw(self.passes[any(ch.audit for ch in c.changes)], self.t), self.main)

    def _recheck(self, f: _Cand, culprits: set[tuple[int, str]], cause: str) -> set[str]:
        """The candidate ``f`` without ``culprits`` (the changes at fault, or those taken from the queue), and each candidate
        above it on the new one, built at once: a task whose inputs the removed changes don't touch keeps its pass, so each
        check ends when its own would have, or after a rerun of the suites that read the removed files, whichever is later.
        The round of each removed change is lost to ``cause``."""
        gone = frozenset().union(*(ch.files for ch in f.changes if ch.key in culprits))
        rerun = self.tr.suites.wall(self.tr.suites.touched(gone)) if gone else self.tr.suites.launch_s
        base, saved = f.base, set()
        if cause == "red":
            f.flake = None
        for old in [f, *self._above(f.sha)]:
            keep = [ch for ch in old.changes if ch.key not in culprits]
            saved.add(old.sha)
            self._leave(old, None)
            if len(keep) < len(old.changes):
                self.lost[cause] = self.lost.get(cause, 0) + len(old.changes) - len(keep)
            if not keep:
                continue
            new = self._new(keep, base)
            new.start, new.records_at = old.start, old.records_at
            if old.start is None:
                new.built = old.built
            else:
                new.faults = {ch.fault for ch in keep if ch.fault is not None}
                new.pass_at = self.t + max(old.pass_at - self.t, rerun)
                new.fail_at = max(old.fail_at or 0.0, self.t + self.tr.suites.launch_s) if new.faults else None
                new.counted = self.t
            self.counts["rechecked"] += len(keep)
            self.cands[new.sha] = new
            base = new.sha
        return saved

    def _switch(self, f: _Cand) -> set[str]:
        """The candidate on a failed one goes on with its check alone on `main`, when that was started on this `main`."""
        nxt = next((c for c in self.cands.values() if c.base == f.sha), None)
        if nxt is None or nxt.twin is None or nxt.twin[2] != self.main or f.base != self.main:
            return set()
        self._use(nxt.start, self.t, SHARDS)
        nxt.base, nxt.start, nxt.pass_at, nxt.twin = f.base, nxt.twin[0], nxt.twin[1], None
        if nxt.faults:
            nxt.fail_at = max(self.t, nxt.fail_at or self.t)
        return {nxt.sha}

    # -- one tick

    def step(self, t: float) -> None:
        self.t = t
        for fault, at in list(self.removal.items()):
            if at <= t:
                for key in self.tr.faults[fault].changes:
                    self._gone(key, "fault")
                del self.removal[fault]
        while self.rule.recorded and self.hands and self.hands[0][0] <= t:
            _, members = self.hands.pop(0)
            hit = max(self.cands.values(), key=lambda c: len(members & set(c.members)), default=None)
            if hit is not None and members & set(hit.members):
                for c in reversed(self._above(hit.sha)):
                    self._leave(c, "dropped base")
                self._leave(hit, "by hand")
            # the loop records a drop by hand as an undecided outcome of its members, newer than any failure that capped them
            held = [ch for ch in self.tr.changes.values() if ch.pr in members]
            self.history.append(vq.Outcome(tuple(ch.pr for ch in held), tuple(ch.head for ch in held), vq.PENDING))
        if self.rule.first:
            self._first()
        if self.rule.side:
            return self._side()
        chain = list(self.cands.values())
        busy = {ch.pr for c in self.side for ch in c.changes}
        if self.rule.beside:
            for c in [c for c in self.side if t >= c.records_at and len(self.cands) < self.flags()["depth"]][:1]:
                self.side.remove(c)
                tops = [x for x in self.cands.values() if not any(y.base == x.sha for y in self.cands.values())]
                c.base = tops[0].sha if tops else self.main
                self._check(c)
                self.cands[c.sha] = c
                chain = list(self.cands.values())
        pins = not (self.rule.plain_first and not self.cands) and not self.side
        snap, ready = self.snapshot(chain, self.main, busy=busy, pins=pins)
        acts = vq.decide(snap)
        self.apply(acts, ready)
        if self.rule.alone:
            # a check alone runs only on a place nothing else needs: the newest is cancelled first
            for c in sorted(self.cands.values(), key=lambda c: -c.built):
                if self._busy() <= PLACES:
                    break
                if c.twin is not None:
                    self._use(c.twin[0], min(self.t, c.twin[1]), SHARDS)
                    c.twin = None
        if self.rule.plain_first and not pins and not self.side:
            return
        if self.rule.beside and self.side and any(isinstance(a, vq.Build) and self.side[-1].built == t for a in acts):
            # the Lean batch went beside the chain: this tick's build is still free for a plain batch
            snap, ready = self.snapshot(list(self.cands.values()), self.main, busy={ch.pr for c in self.side for ch in c.changes},
                                        pins=False)
            self.apply([a for a in vq.decide(snap) if isinstance(a, vq.Build)], ready)

    def _side(self) -> None:
        """Batches beside each other, each on the `main` it was built on: one that passes lands at once if `main` hasn't
        moved since, else after a rerun of the suites that read files of both it and what landed since (the union's check).
        One Lean batch is in its records run or its check at a time: the records of a second would be stale when the first lands."""
        for c in sorted(self.cands.values(), key=lambda c: c.built):
            if c.start is None:
                if self.t < c.records_at:
                    continue
                self._check(c)
            snap, ready = self.snapshot([c], c.base, depth=0)
            self.apply(vq.decide(snap), ready, on_main=self._union)
        if sum(1 for c in self.cands.values() if c.union is None) < self.rule.side:
            busy = {ch.pr for c in self.cands.values() for ch in c.changes}
            lean = any(ch.lean for c in self.cands.values() for ch in c.changes)
            snap, ready = self.snapshot([], self.main, depth=1, busy=busy, pins=not lean)
            self.apply([a for a in vq.decide(snap) if isinstance(a, vq.Build)], ready)

    def _union(self, c: _Cand) -> bool:
        """Whether ``c``, which passed, lands now.  It does when nothing landed since it was built and no union is ahead of
        it; else its union with what landed since, and with the unions ahead of it, reruns the suites that read files of
        both sides, and it lands when that ends and those ahead have landed."""
        if c.union is None:
            ahead = [u for u in self.cands.values() if u.union is not None]
            since = [self.tr.changes[key] for key, at in self.landed.items() if at > c.built] + [ch for u in ahead for ch in u.changes]
            if not since:
                return True
            mine = frozenset().union(*(ch.files for ch in c.changes))
            theirs = frozenset().union(*(ch.files for ch in since))
            both = self.tr.suites.touched(mine) & self.tr.suites.touched(theirs)
            took = self.tr.suites.wall(both) if both else 0.0
            self.counts["unions"] += 1
            self.counts["same_file"] += bool(mine & theirs)
            self.union_s.append(took)
            self._use(self.t, self.t + took, SHARDS)
            c.union = max([self.t + took, *(u.union for u in ahead)])
        return self.t >= c.union

    def _first(self) -> None:
        """Changes of the rule's class land on their mark and verdict; `main` is then checked, so many checks at once, each
        over what landed since the one before it.  A change at fault stays on `main` until a check fails on it and the loop
        reverts it: at once when the failure names it alone; else as long after as it really was taken from the queue, or,
        under `Rule.back`, at once with every other change the failure names, each of which the queue then checks before it
        lands again.  Under `Rule.back` a failure with nobody at fault reverts every change of its check the same way."""
        kind, back = self.rule.first, self.rule.back
        for p in list(self.post):
            if p["fail_at"] is not None and self.t >= p["fail_at"]:
                again: list[Change] = list(p["changes"]) if p["flake"] and back else []
                for ch in p["changes"]:
                    if ch.fault is not None and ch.key in self.landed:
                        suspects = [o for o in p["changes"] if vq.meets(self._files(o), self._blame(ch.fault))]
                        if len(suspects) > 1 and back:
                            self.red.append((self.landed[ch.key], self.t))
                            again += [o for o in suspects if o not in again]
                            continue
                        at = self.t if len(suspects) == 1 else self.t + (self.tr.faults[ch.fault].delay or 0.0)
                        self.red.append((self.landed[ch.key], at))
                        self.red_until = max(self.red_until, at)
                        self.counts["reverts"] += 1
                        self.out[ch.key] = (at, "fault")
                        self.arrive[ch.key] = at
                        del self.landed[ch.key]
                again = [o for o in again if o.key in self.landed]
                if again:
                    # the loop cannot name one change: each suspect is reverted and goes through the queue's check before it lands
                    for o in again:
                        del self.landed[o.key]
                        self.checked.add(o.key)
                    self.counts["reverts"] += len(again)
                    self.history.append(vq.Outcome(tuple(o.pr for o in again), tuple(o.head for o in again), vq.FAIL,
                                                   tuple(o.pr for o in again), self.main))
                self.counts["false_reds"] += p["flake"]
                self._use(p["start"], self.t, SHARDS)
                self.post.remove(p)
            elif self.t >= p["pass_at"]:
                self._use(p["start"], p["pass_at"], SHARDS)
                self.post.remove(p)
        red = self.t < self.red_until or any(ch.fault is not None and ch.key in self.landed for _, ch in self.on_main) or any(
            ch.fault is not None and ch.key in self.landed for p in self.post for ch in p["changes"])
        for ch in sorted(self.ready().values(), key=lambda ch: ch.rank):
            if getattr(ch, kind) and ch.key not in self.checked and not any(a <= self.t < b for a, b in ch.conflicts) and not any(
                    ch in c.changes for c in self.cands.values()):
                self._land(ch)
                self.on_main.append((self.t, ch))
                self.counts["on_red"] += red
                red = red or ch.fault is not None
        if self.on_main and len(self.post) < self.rule.after:
            changes = [ch for _, ch in self.on_main]
            self.on_main = []
            took = self._draw(self.passes[False], self.t)
            flake = not any(ch.fault is not None for ch in changes) and bool(self.flakes) and self.flakes[0] <= self.t
            if flake:
                self.flakes.pop(0)
            fails = any(ch.fault is not None for ch in changes) or flake
            self.post.append({"start": self.t, "changes": changes, "pass_at": self.t + took, "flake": flake,
                              "fail_at": self.t + min(self._draw(self.fails, self.t), took) if fails else None})

    def run(self) -> "Replay":
        for t in self.tr.ticks:
            self.step(t)
        self.t = self.tr.end
        for c in [*self.cands.values(), *self.side]:
            self._leave(c, None)
        for p in self.post:
            self._use(p["start"], min(self.t, p["pass_at"]), SHARDS)
        return self


# ---- the measures -----------------------------------------------------------------------------------------------------------


def peak(places: list[tuple[float, float, int]], start: float, end: float) -> dict:
    """The most check places held at once, the places held nine tenths of the time or less, and the share of the time more
    than `PLACES` were held."""
    events = sorted([(a, n) for a, _, n in places] + [(b, -n) for _, b, n in places])
    now, most, last, held = 0, 0, start, {}
    for at, n in events:
        held[now] = held.get(now, 0.0) + max(0.0, min(at, end) - last)
        last, now = max(last, min(at, end)), now + n
        most = max(most, now)
    held[now] = held.get(now, 0.0) + max(0.0, end - last)
    total, acc, p90 = sum(held.values()) or 1.0, 0.0, 0
    for n in sorted(held):
        acc += held[n]
        if acc >= .9 * total:
            p90 = n
            break
    return {"peak": most, "p90_places": p90, "over": sum(v for n, v in held.items() if n > PLACES) / total}


def measure(tr: Trace, run: Replay) -> dict:
    """One run's numbers, over the changes that really landed in the window: minutes from the mark to `main` (a change
    the run did not land by the window's end has none)."""
    who = [ch for ch in tr.changes.values() if ch.landed is not None and ch.mark is not None and tr.start <= ch.landed <= tr.end]
    wait = {ch.key: (run.landed[ch.key] - ch.mark) / 60 if ch.key in run.landed else float("inf") for ch in who}
    minutes = sorted(wait.values())
    lean = [wait[ch.key] <= qt.TARGET_MIN for ch in who if ch.lean]
    other = [wait[ch.key] <= qt.TARGET_MIN for ch in who if not ch.lean]
    new = [wait[ch.key] <= qt.TARGET_MIN for ch in who if ch.mark >= tr.start - 60 * qt.TARGET_MIN]
    red = sorted(run.red)
    merged: list[list[float]] = []
    for a, b in red:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, min(b, tr.end)])
    return {"n": len(minutes), "within": sum(1 for m in minutes if m <= qt.TARGET_MIN) / len(minutes) if minutes else 0.0,
            "p50": qt.pct(minutes, .5), "p90": qt.pct(minutes, .9), "unlanded": sum(1 for m in minutes if m == float("inf")),
            "lost": sum(run.lost.values()), "by_cause": dict(run.lost), **run.counts, **peak(run.places, tr.start, tr.end),
            "within_lean": sum(lean) / len(lean) if lean else 0.0, "within_other": sum(other) / len(other) if other else 0.0,
            "within_new": sum(new) / len(new) if new else 0.0,
            "landed_all": len(run.landed), "red_times": len(merged), "red_min": sum(b - a for a, b in merged) / 60,
            "union_min": (qt.pct(run.union_s, .5) or 0.0) / 60}


def measured(tr: Trace) -> dict:
    """What really happened over the replay's window, in `measure`'s terms."""
    m = tr.measured
    minutes = m["minutes"]
    who = [ch for ch in tr.changes.values() if ch.landed is not None and ch.mark is not None and tr.start <= ch.landed <= tr.end]
    kind = {k: [(ch.landed - ch.mark) / 60 <= qt.TARGET_MIN for ch in who if ch.lean == k] for k in (True, False)}
    new = [(ch.landed - ch.mark) / 60 <= qt.TARGET_MIN for ch in who if ch.mark >= tr.start - 60 * qt.TARGET_MIN]
    return {"n": len(minutes), "within": sum(1 for x in minutes if x <= qt.TARGET_MIN) / len(minutes) if minutes else 0.0,
            "p50": qt.pct(minutes, .5), "p90": qt.pct(minutes, .9), "lost": sum(m["lost"].values()), "by_cause": m["lost"],
            "built": m["built"], "reds": m["reds"], "moves": m["moves"], "landed": m["landed"],
            "within_lean": sum(kind[True]) / len(kind[True]) if kind[True] else 0.0, "n_lean": len(kind[True]),
            "within_other": sum(kind[False]) / len(kind[False]) if kind[False] else 0.0,
            "within_new": sum(new) / len(new) if new else 0.0, "n_new": len(new)}


def row(tr: Trace, rule: Rule, seeds: int) -> dict:
    """``rule`` over ``seeds`` runs: each number's mean, and the least and most of the share within two hours."""
    runs = [measure(tr, Replay(tr, rule, seed).run()) for seed in range(seeds)]
    out: dict = {"rule": rule.name, "backtrack": rule.backtrack, "seeds": seeds}
    for k in runs[0]:
        if k != "by_cause":
            vals = [r[k] for r in runs]
            out[k] = sum(vals) / len(vals) if all(v != float("inf") for v in vals) else float("inf")
    out["within_range"] = [min(r["within"] for r in runs), max(r["within"] for r in runs)]
    causes = {c for r in runs for c in r["by_cause"]}
    out["by_cause"] = {c: sum(r["by_cause"].get(c, 0) for r in runs) / len(runs) for c in sorted(causes)}
    return out


def pairs(tr: Trace, doc: dict) -> dict:
    """Of each two real batches built one after the other with a known suite map: how often they share no suite but those
    that read the whole tree, and the seconds a check of the suites they do share takes."""
    decisions = [r for r in doc["control"].get("decisions") or [] if r["action"] == "build" and not r.get("shadow")
                 and qt.ts(r["at"]) >= tr.start]
    restacks = doc.get("restacks") or {}
    batches = []
    for r in decisions:
        chs = [tr.changes.get((int(pr), qt.chain(restacks, h)[-1])) for pr, h in zip(r.get("members") or [], r.get("heads") or [])]
        files = frozenset().union(*(ch.files for ch in chs if ch is not None)) if chs else frozenset()
        if files:
            batches.append(tr.suites.touched(files))
    whole = tr.suites.whole
    shared = [a & b for a, b in zip(batches, batches[1:])]
    cost = [tr.suites.wall(s) for s in shared]
    return {"pairs": len(shared), "none_shared": sum(1 for s in shared if s <= whole),
            "shared_p50": qt.pct([len(s - whole) for s in shared], .5), "suites": len(tr.suites.by_path),
            "rerun_p50_min": (qt.pct(cost, .5) or 0) / 60, "rerun_p90_min": (qt.pct(cost, .9) or 0) / 60,
            "whole_min": tr.suites.wall(whole) / 60}


def lean_first(tr: Trace, doc: dict) -> list[dict]:
    """Each time the loop built a Lean batch whose records run comes first: how many other candidates were in the queue then
    and how many changes they held, whether a drop came in the five minutes before, how many plain changes the queue could
    take that were in no candidate, and the minutes until the loop next built a batch with a plain change (to the window's
    end when it built none)."""
    decisions = [r for r in doc["control"].get("decisions") or [] if not r.get("shadow")]
    restacks = doc.get("restacks") or {}
    builds, began, gone = [], {}, {}
    for r in decisions:
        sha, at = (r.get("sha") or "")[:12], qt.ts(r["at"])
        if r["action"] == "build" and r.get("done"):
            got = [tr.changes.get((int(pr), qt.chain(restacks, h)[-1])) for pr, h in zip(r.get("members") or [], r.get("heads") or [])]
            builds.append((at, sha, [ch for ch in got if ch is not None], r.get("why") or ""))
            began.setdefault(sha, at)
        elif r["action"] == "drop" or (r["action"] == "advance" and r.get("done")):
            gone.setdefault(sha, at)
    held_by = {sha: chs for _, sha, chs, _ in builds}
    drops = sorted(qt.ts(r["at"]) for r in decisions if r["action"] == "drop" and qt.drop_cause(r.get("why") or "") != "on main")
    out = []
    for at, sha, mine, why in builds:
        if not why.startswith("its records run first") or at < tr.start:
            continue
        # a candidate from before the window with no recorded end is one the loop forgot at a restart
        flying = [k for k, b in began.items() if k != sha and b < at < gone.get(k, tr.start if b < tr.start else float("inf"))]
        held = {ch.key for k in [sha, *flying] for ch in held_by[k]}
        waiting = [ch for ch in tr.changes.values() if ch.plain and ch.key not in held and not (ch.landed and ch.landed <= at)
                   and any(a <= at and (b is None or at < b) for a, b, _ in ch.spans)]
        nxt = min((b for b, _, chs, _ in builds if b > at and any(ch.plain for ch in chs)), default=tr.end)
        out.append({"at": at, "lean": len(mine), "in_queue": len(flying), "held": sum(len(held_by[k]) for k in flying),
                    "after_drop": any(at - 300 <= d <= at for d in drops), "plain_waiting": len(waiting), "minutes": (nxt - at) / 60})
    return out


RULES = (
    Rule("0. For reference: every change, Lean ones too, lands the moment the queue may take it and is checked after",
         "a revert", first="every"),
    Rule("1. The loop as it ran: its flags, its blame, its pins and the drops by hand at their times", recorded=True, pin_all=False),
    Rule("1b. The loop with a Lean batch's records run first and nothing built meanwhile, left alone: batch 32, depth 6, "
         "pins as they were", pin_all=False),
    Rule("2. Plain batches build beside a Lean batch's records run; a failed check removes only the change at fault and what "
         "is rebuilt keeps its passes", "none: the same checks, scheduled differently", beside=True, keep=True, keep_all=True),
    Rule("2a. Only: plain batches build beside a Lean batch's records run", "none", beside=True),
    Rule("2b. Row 1b, and only: a failed check removes only the change at fault and what is rebuilt keeps its passes", "none",
         keep=True, keep_all=True, pin_all=False),
    Rule("2c. Row 2, but passes are kept only when the loop names one change at fault", "none", beside=True, keep=True),
    Rule("3. Batches side by side on `main`, 6 in check at once, each union rechecked on the suites both sides read",
         "a union that fails is dropped and its batch built again on `main`", side=6, beside=True),
    Rule("3b. The same, 4 in check at once", "the same", side=4, beside=True),
    Rule("4. Row 2, and each candidate above the lowest is also checked alone on `main` while places are idle",
         "none: a check alone that is not needed is cancelled", beside=True, keep=True, keep_all=True, alone=True),
    Rule("4b. Row 2a, and the same checks alone", "the same", beside=True, alone=True),
    Rule("5. Row 2, and a plain change lands on its mark and verdict and is checked after",
         "the loop reverts the change a later check fails on", beside=True, keep=True, keep_all=True, first="plain"),
    Rule("5b. Row 5, but a failed check of `main` that names several changes reverts each of them at once, and they are checked "
         "before they land again", "the same, and the reverts of changes not at fault", beside=True, keep=True, keep_all=True, first="plain",
         back=True),
    Rule("5c. Row 5 with four checks of `main` at once", "the same", beside=True, keep=True, keep_all=True, first="plain", after=4),
    Rule("5d. Row 5, and a change is marked only once the suites it reruns pass on it merged onto `main`", "the same",
         beside=True, keep=True, keep_all=True, first="plain", gate=True),
    Rule("6a. Row 2, and a change of prose only (every file Markdown) lands on its mark and verdict", "a revert",
         beside=True, keep=True, keep_all=True, first="prose"),
    Rule("6b. Row 2, and a change of tests only (every file under a tests directory) lands on its mark and verdict", "a revert",
         beside=True, keep=True, keep_all=True, first="tests"),
    Rule("6c. Row 2, and a candidate every 10 minutes whatever is below it, no depth",
         "none: more candidates are dropped on a red", beside=True, keep=True, keep_all=True, period=600.0),
    Rule("6d. Row 2 with batch 16", "none", beside=True, keep=True, keep_all=True, batch=16),
    Rule("6e. Row 2 with batch 64, depth 3", "none", beside=True, keep=True, keep_all=True, batch=64, depth=3),
    Rule("6f. Row 2, and a Lean batch's check starts with its records run, not after it", "the check is run again when the "
         "records commit changes more than the records (not priced: here it never does)", beside=True, keep=True, keep_all=True, overlap=True),
    Rule("6g. Row 2, and a change is marked only once the suites it reruns pass on it merged onto `main`",
         "none: the run before the mark is the cost", beside=True, keep=True, keep_all=True, gate=True),
    Rule("7. Row 1b, and with nothing in check the plain batch is built before a Lean batch", "none", plain_first=True,
         pin_all=False),
)

ASSUMPTIONS = (
    "Owners mark when they did; a change at fault is fixed as long after it leaves the queue as it really was.",
    "The queue can take a change from its mark and its land verdict on, or from when the loop first built it if that was "
    "sooner (a change admitted with no verdict), while it was not held and its mark stood.",
    "A check takes as long as one of the real checks of its kind (Lean change or not) built nearest that time, drawn at "
    "random; a failing one shows as soon as one of the real failed checks nearest that time did; a records run takes as long "
    "as one of the real ones that passed.",
    "A change a real check failed on fails every check that holds it; no other change fails.  Each real check that failed "
    "with nobody at fault fails the first check started after it was built.",
    "A change conflicts with `main` when it really did, whatever landed before it in the replay.",
    "The queue's order is the order the loop first built the changes in (its priority file's history is not kept).",
    "Check places are not rationed: a rule's peak is reported against the places there are.  A check holds three places, a "
    "records run one, a union's check three.",
    "Every Lean change the queue may take is pinned; in the rows that say pins as they were, one is pinned while a real "
    "candidate held it, and from its last build on if it landed.  One Lean batch is in its records run at a time; side by "
    "side, one in its records run or its check.",
    "A kept pass: a candidate rebuilt without some changes ends when its check would have, or after a check of the suites "
    "that read the removed files, whichever is later.  A union's check never fails.",
    "Landing before the check: a check of `main` covers what landed since the one before it and fails only for a change at "
    "fault among those; a red is found when that check fails.",
)


def render(tr: Trace, rows: list[dict], extra: dict, held: list[dict]) -> str:
    m = measured(tr)
    span = (tr.end - tr.start) / 3600

    found: dict[str, int] = {}
    for f in tr.faults.values():
        found[f.how] = found.get(f.how, 0) + 1

    def fmt(x):
        return "not by the window's end" if x != x or x == float("inf") else f"{x:.0f}"

    out = [f"Window: {span:.0f} hours, {len(tr.ticks)} ticks of the loop, {m['n']} changes landed with a mark, {m['n_new']} of "
           f"them marked in the window or less than two hours before it, {m['n_lean']} of them Lean changes (a Lean source that "
           f"compiles); {len(tr.faults)} "
           f"failed checks with a change at fault ({', '.join(f'{n} {how}' for how, n in sorted(found.items()))}), {len(tr.flakes)} "
           "with nobody at fault.", "",
           "| Rule | On `main` within 2 h of the mark | Of those marked in the window; of the Lean changes; of the others | Mark "
           "to `main`, median (min) | Ninth decile (min) | PR-rounds lost | Landed in the window | Check places at peak (of "
           f"{PLACES}, soon {PLACES_SOON}) | `main` red |", "|---|---|---|---|---|---|---|---|---|",
           f"| Measured | {m['within']:.0%} | {m['within_new']:.0%}; {m['within_lean']:.0%}; {m['within_other']:.0%} | "
           f"{fmt(m['p50'])} | {fmt(m['p90'])} | {m['lost']} | {m['landed']} | not read | never |"]
    for r in rows:
        red = (f"{r['red_times']:.0f} times, {r['red_min']:.0f} min in all; {r['on_red']:.0f} changes landed on a red `main`; "
               f"{r['reverts']:.0f} reverts") if r["reverts"] or r["red_times"] else "never"
        over = f", over {PLACES} for {r['over']:.0%} of the time" if r["over"] >= 0.005 else ""
        out.append(f"| {r['rule']} | {r['within']:.0%} ({r['within_range'][0]:.0%} to {r['within_range'][1]:.0%}) | "
                   f"{r['within_new']:.0%}; {r['within_lean']:.0%}; {r['within_other']:.0%} | {fmt(r['p50'])} | {fmt(r['p90'])} | "
                   f"{r['lost']:.0f} | "
                   f"{r['landed_all']:.0f} | {r['peak']:.0f}{over} | {red} |")
    out += ["", f"Each replayed row is the mean of {rows[0]['seeds'] if rows else 0} runs; in brackets, the least and the most of "
            "the runs.", "",
            f"Real batches built one after the other: {extra['none_shared']} of {extra['pairs']} pairs share no suite but those "
            f"reading the whole tree; they share {extra['shared_p50']:.0f} of {extra['suites']} suites at the median, and a check "
            f"of the shared suites takes {extra['rerun_p50_min']:.0f} minutes at the median, {extra['rerun_p90_min']:.0f} at the "
            f"ninth decile (the whole-tree suites alone: {extra['whole_min']:.0f}).", ""]
    if held:
        alone = [h for h in held if not h["in_queue"]]
        out += [f"Lean batches whose records run came first: {len(held)}.  Until the loop next built a batch with a plain change: "
                f"{', '.join(f'{h['minutes']:.0f}' for h in held)} minutes, with {min(h['plain_waiting'] for h in held)} to "
                f"{max(h['plain_waiting'] for h in held)} plain changes the queue could take in no candidate.  Built with nothing "
                f"else in the queue: {len(alone)} ({', '.join(f'{h['minutes']:.0f}' for h in alone)} minutes); within five minutes "
                f"of a drop: {sum(1 for h in held if h['after_drop'])}.", ""]
    out += ["Assumptions:", *(f"- {a}" for a in ASSUMPTIONS)]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m research.queue_replay", description=__doc__.split("\n\n")[0])
    ap.add_argument("--doc", type=Path, help="a saved document (`--save`); without it the records are read now")
    ap.add_argument("--save", type=Path, help="write the document read to this file; one already there says each PR's files")
    ap.add_argument("--hours", type=float, default=72.0, help="how far back the records are read")
    ap.add_argument("--seeds", type=int, default=5, help="runs of each rule, each with its own draws")
    ap.add_argument("--rules", default="", help="the rows to print, by the start of their names, comma-separated (default: all)")
    ap.add_argument("--json", action="store_true", help="print the rows as JSON")
    ap.add_argument("--cut", type=Path, help="write the document cut to its last `--hours`, small, to this file and stop")
    a = ap.parse_args(argv)
    doc = json.loads(a.doc.read_text()) if a.doc else collect(
        a.hours, json.loads(a.save.read_text()) if a.save and a.save.exists() else None)
    if a.save:
        a.save.write_text(json.dumps(doc, default=str))
    if a.cut:
        a.cut.write_text(json.dumps(cut(doc, a.hours), separators=(",", ":"), sort_keys=True) + "\n")
        return 0
    tr = trace(doc)
    want = [w.strip() for w in a.rules.split(",") if w.strip()]
    rows = [row(tr, rule, a.seeds) for rule in RULES if not want or any(rule.name.startswith(w) for w in want)]
    extra, held = pairs(tr, doc), lean_first(tr, doc)
    print(json.dumps({"measured": measured(tr), "rows": rows, "pairs": extra, "records_first": held}, indent=1, default=str)
          if a.json else render(tr, rows, extra, held))
    return 0


if __name__ == "__main__":
    sys.exit(main())
