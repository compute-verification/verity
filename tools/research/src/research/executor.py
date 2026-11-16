"""Drive's side of the task executor: enqueue a check and read each task's verdict back by its key."""
from __future__ import annotations

import hashlib
import heapq
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

from .store import vocab

PASS, FAIL, PENDING = "pass", "fail", "pending"
ERROR = "error"
BLOCKED = "blocked"
NONE, MISSING = "none", "missing"
AFTER_LANDING_PRIORITY = 1
TASKS_GRAPH_KIND = "tasks-graph/v1"
GRAPH_V2 = "verity-graph/v2"
TASKS_V1 = "tasks/v1"
LISTED = {TASKS_V1: (vocab.TASKS_GRAPH, TASKS_GRAPH_KIND), GRAPH_V2: (vocab.VERITY_GRAPH, GRAPH_V2)}
KEY_TAG = b"verity-key/v1\n"
V2_FIELDS = {"id", "kind", "env", "argv", "vars", "inputs", "reads", "outputs"}
SWITCH_STREAK = 3
SHADOW_WAIT_S = 6 * 3600
OLD_PASS = ("passed", "skipped", "reused", "cached")
OLD_FAIL = ("failed",)
SUITE_NO_VERDICT = ("skipped",)
SPLIT_RE = re.compile(r":\d+/\d+$")
AUDIT_RE = re.compile(r"^AUDIT (?:\S*?/lean-audit-scratch-[^/\s]+/)?(\S+?): (PASS|FAIL)\b", re.M)
REUSED_RE = re.compile(r"^lean-audit: (\S+?): PASS reused\b", re.M)


def task_target(key: str) -> str:
    return f"task:{key.removeprefix('sha512:')}"


def canonical(x) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def task_key(task: dict, files: dict[str, str], env: str, reads: dict[str, dict[str, str]]) -> str:
    doc = {"env": env, "kind": task["kind"], "argv": list(task["argv"]), "vars": dict(task["vars"]),
           "inputs": sorted([p, files[p]] for p in task["inputs"]),
           "reads": sorted([r["task"], r["output"], reads[r["task"]][r["output"]]] for r in task["reads"]),
           "outputs": sorted([o["name"], o["path"]] for o in task["outputs"])}
    return "sha512:" + hashlib.sha512(KEY_TAG + canonical(doc)).hexdigest()


def fold_results(docs: Iterable[object], key: str) -> tuple[str, dict | None]:
    got = [d for d in docs if isinstance(d, dict) and d.get("key") == key and d.get("status") in (PASS, FAIL)]
    first = next((d for s in (FAIL, PASS) for d in got if d["status"] == s), None)
    return (first["status"], first) if first else (PENDING, None)


def check_id(graph: dict) -> str:
    return f"{graph.get('kind') or 'after-landing'}:{graph['commit']}"


def _digest(doc) -> str:
    return hashlib.sha512(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def keyed(graph: dict) -> dict:
    check = check_id(graph)
    return {**graph, "tasks": [t if t.get("key") else {**t, "key": _digest({"check": check, "id": t["id"]})}
                               for t in graph["tasks"]]}


def fold(values: Iterable[str]) -> str:
    vs = list(values)
    if FAIL in vs and ERROR not in vs[len(vs) - vs[::-1].index(FAIL):]:
        return FAIL
    return PASS if vs and vs[-1] == PASS else PENDING


class Transport(Protocol):
    def enqueue(self, graph_file: Path | None, priority: int, check: str, schema: str | None = None) -> None:
        ...

    def graph(self, check: str, schema: str = TASKS_V1) -> tuple[str, dict] | None:
        ...

    def labels(self, keys: list[str]) -> dict[str, list[str]]:
        ...

    def results(self, keys: list[str]) -> dict[str, list[object]]:
        ...


class Executor:
    def __init__(self, transport: Transport):
        self.transport = transport

    def enqueue(self, graph: dict, priority: int) -> dict[str, str]:
        g = keyed(graph)
        with tempfile.TemporaryDirectory(prefix="executor-") as d:
            f = Path(d) / "graph.json"
            f.write_text(json.dumps(g))
            self.transport.enqueue(f, priority, check_id(g))
        return {t["id"]: t["key"] for t in g["tasks"]}

    def enqueue_check(self, check: str, priority: int) -> None:
        self.transport.enqueue(None, priority, check)

    def graph(self, check: str) -> tuple[str, dict[str, str]] | None:
        if (got := self.transport.graph(check)) is None:
            return None
        gid, g = got
        if not isinstance(g, dict) or g.get("schema") != "tasks/v1" or "commit" not in g or check_id(g) != check:
            what = check_id(g) if isinstance(g, dict) and "commit" in g else "no tasks/v1 check"
            raise ValueError(f"the executor's graph {gid} of {check} is of {what}")
        return gid, {t["id"]: t["key"] for t in keyed(g)["tasks"]}

    def verdicts(self, keys: Iterable[str]) -> dict[str, str]:
        ks = list(dict.fromkeys(keys))
        got = self.transport.labels(ks)
        return {k: fold(got.get(k) or ()) for k in ks}

    def enqueue_graph(self, check: str, priority: int) -> None:
        self.transport.enqueue(None, priority, check, GRAPH_V2)

    def graph_v2(self, check: str) -> tuple[str, dict] | None:
        if (got := self.transport.graph(check, GRAPH_V2)) is None:
            return None
        gid, g = got
        if not isinstance(g, dict) or g.get("schema") != GRAPH_V2 or not isinstance(g.get("tasks"), list):
            raise ValueError(f"the executor's graph {gid} of {check} is no {GRAPH_V2} graph")
        if g.get("commit") != check.partition(":")[2]:
            raise ValueError(f"the executor's graph {gid} of {check} is of commit {g.get('commit')}")
        files, envs = g.get("files") or {}, g.get("envs") or {}
        ids = {t.get("id") for t in g["tasks"] if isinstance(t, dict)}
        for t in g["tasks"]:
            if not (isinstance(t, dict) and V2_FIELDS <= t.keys() and t["env"] in envs and set(t["inputs"]) <= files.keys()
                    and all(isinstance(r, dict) and r.get("task") in ids for r in t["reads"])):
                raise ValueError(f"the executor's graph {gid} of {check} has a task with no key to compute: "
                                 f"{t.get('id') if isinstance(t, dict) else repr(t)}")
        return gid, g

    def settle(self, graph: dict, keyed: dict[str, str] | None = None) -> dict[str, tuple[str, dict | None]]:
        tasks = {t["id"]: t for t in graph["tasks"]}
        ids = list(tasks)
        order = {i: n for n, i in enumerate(ids)}
        files, envs = graph.get("files") or {}, graph.get("envs") or {}
        out: dict[str, tuple[str, dict | None]] = {}
        keys = {i: t["key"] for i, t in tasks.items() if t.get("key")}
        unread = {i: len(t["reads"]) for i, t in tasks.items() if i not in keys}
        blocked: set[str] = set()
        pending: set[str] = set()
        readers: dict[str, list[tuple[str, dict]]] = {}
        for i in unread:
            for r in tasks[i]["reads"]:
                readers.setdefault(r["task"], []).append((i, r))
        fetch = dict(keys)
        later = {i for i, n in unread.items() if not n}
        now: list[int] = []
        at = -1

        def settled(j: str) -> None:
            s = out[j]
            for i, r in readers.get(j, ()):
                if i in out or i in keys:
                    continue
                unread[i] -= 1
                if s[0] in (FAIL, BLOCKED) or s[0] == PASS and r["output"] not in (s[1].get("outputs") or {}):
                    blocked.add(i)
                elif s[0] == PENDING:
                    pending.add(i)
                if i in blocked or i in pending or not unread[i]:
                    if order[i] > at:
                        heapq.heappush(now, order[i])
                    else:
                        later.add(i)

        while fetch or later:
            if fetch:
                got = self.transport.results(list(dict.fromkeys(fetch.values())))
                out.update({i: fold_results(got.get(k) or (), k) for i, k in fetch.items()})
                for i in fetch:
                    settled(i)
                fetch = {}
            for i in later:
                heapq.heappush(now, order[i])
            later = set()
            while now:
                i = ids[at := heapq.heappop(now)]
                if i in out or i in keys:
                    continue
                if i in blocked or i in pending:
                    out[i] = (BLOCKED if i in blocked else PENDING, None)
                    settled(i)
                else:
                    t = tasks[i]
                    keys[i] = fetch[i] = task_key(t, files, envs[t["env"]]["digest"],
                                                  {r["task"]: out[r["task"]][1]["outputs"] for r in t["reads"]})
            at = -1
        if keyed is not None:
            keyed.update(keys)
        return {i: out.get(i, (PENDING, None)) for i in tasks}


class FileTransport:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def enqueue(self, graph_file: Path | None, priority: int, check: str, schema: str | None = None) -> None:
        graph = json.loads(graph_file.read_text()) if graph_file is not None else None
        with (self.root / "enqueued.jsonl").open("a") as f:
            f.write(json.dumps({"check": check, "priority": priority, "graph": graph, **({"schema": schema} if schema else {})})
                    + "\n")

    def enqueued(self) -> list[dict]:
        p = self.root / "enqueued.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines()] if p.is_file() else []

    def _graph_file(self, check: str, schema: str) -> Path:
        return self.root / "graphs" / schema.replace("/", "_") / f"{check.replace(':', '_')}.json"

    def list_graph(self, check: str, graph: dict) -> None:
        p = self._graph_file(check, graph.get("schema") or TASKS_V1)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(graph))

    def graph(self, check: str, schema: str = TASKS_V1) -> tuple[str, dict] | None:
        p = self._graph_file(check, schema)
        return (str(p), json.loads(p.read_text())) if p.is_file() else None

    def finish(self, key: str, verdict: str) -> None:
        with (self.root / "verdicts.jsonl").open("a") as f:
            f.write(json.dumps({"key": key, "verdict": verdict}) + "\n")

    def finish_result(self, result: dict) -> None:
        with (self.root / "results.jsonl").open("a") as f:
            f.write(json.dumps({"key": result["key"], "result": result}) + "\n")

    def _lines(self, name: str, field: str, keys: list[str]) -> dict[str, list]:
        p, want, out = self.root / name, set(keys), {}
        for line in p.read_text().splitlines() if p.is_file() else ():
            rec = json.loads(line)
            if rec["key"] in want:
                out.setdefault(rec["key"], []).append(rec[field])
        return out

    def labels(self, keys: list[str]) -> dict[str, list[str]]:
        return self._lines("verdicts.jsonl", "verdict", keys)

    def results(self, keys: list[str]) -> dict[str, list[object]]:
        return self._lines("results.jsonl", "result", keys)


class QueueTransport:
    """infra's queue, reached through its enqueue command: research imports nothing from `infra/`."""

    by: str | None = None

    def __init__(self, store, command: tuple[str, ...] = (), cwd: Path | None = None):
        self.store, self.command, self.cwd = store, tuple(command), cwd

    def _labels(self, target: str, key: str) -> list:
        return [lab for lab in self.store.labels(target, key=key) if self.by is None or lab.by == self.by]

    def enqueue(self, graph_file: Path | None, priority: int, check: str, schema: str | None = None) -> None:
        if not self.command:
            raise RuntimeError("drive.toml names no `executor_enqueue` command: set it to infra's enqueue command, the words "
                               "of its argv")
        graph = ["--graph", str(graph_file)] if graph_file is not None else []
        r = subprocess.run([*self.command, *graph, "--priority", str(priority), "--check", check,
                            *(["--schema", schema] if schema else [])],
                           cwd=self.cwd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.returncode:
            raise RuntimeError(f"{' '.join(self.command)} exited {r.returncode}: {(r.stderr or r.stdout).strip()[-400:]}")

    def graph(self, check: str, schema: str = TASKS_V1) -> tuple[str, dict] | None:
        from .store import labels_sync

        label, want = LISTED[schema]
        target = f"commit:{check.partition(':')[2]}"
        if not (labs := self._labels(target, label)) and getattr(self.store, "remote", None) is not None:
            labels_sync.pull_targets(self.store, [target])
            labs = self._labels(target, label)
        if not labs:
            return None
        art = labs[-1].value
        if (kind := self.store.get_manifest(art).kind) != want:
            raise ValueError(f"{target}'s {label} label names {art}, a {kind}, not a {want}")
        return art, json.loads(Path(self.store.fetch(art)).read_text())

    def _values(self, keys: list[str], label: str) -> dict[str, list]:
        from .queue import PREFETCH_JOBS
        from .store import labels_sync

        have = {k: [lab.value for lab in self._labels(task_target(k), label)] for k in keys}
        stale = (lambda v: fold(v) != PASS) if label == vocab.TASK_VERDICT else (lambda v: not v)
        if (todo := [k for k, v in have.items() if stale(v)]) and getattr(self.store, "remote", None) is not None:
            labels_sync.pull_targets(self.store, [task_target(k) for k in todo], jobs=PREFETCH_JOBS)
            have.update({k: [lab.value for lab in self._labels(task_target(k), label)] for k in todo})
        return have

    def labels(self, keys: list[str]) -> dict[str, list[str]]:
        return self._values(keys, vocab.TASK_VERDICT)

    def results(self, keys: list[str]) -> dict[str, list[object]]:
        return self._values(keys, vocab.TASK_RESULT)


def old_verdict(status: str | None) -> str | None:
    return PASS if status in OLD_PASS else FAIL if status in OLD_FAIL else None


def old_verdicts(run_dirs: Iterable[Path]) -> dict[str, str]:
    from .landing import PREFLIGHT, step_records, suite_rows

    got: dict[str, set[str]] = {}

    def put(part: str, v: str | None) -> None:
        if v is not None:
            got.setdefault(part, set()).add(v)

    for d in run_dirs:
        for st in step_records(d):
            if st.get("step") in PREFLIGHT:
                continue
            put(st.get("step") or "", old_verdict(st.get("status")))
            if st.get("step") == "lean-audit" and st.get("log"):
                try:
                    log = (d / Path(st["log"]).name).read_text(errors="replace")
                except OSError:
                    log = ""
                for pkg, said in AUDIT_RE.findall(log):
                    put(f"lean-audit:{pkg}", PASS if said == "PASS" else FAIL)
                for pkg in REUSED_RE.findall(log):
                    put(f"lean-audit:{pkg}", PASS)
        for name, s in suite_rows(d).items():
            if s.get("status") not in SUITE_NO_VERDICT:
                put(f"{s['step']}:{name}", old_verdict(s.get("status")))
    return {p: FAIL if FAIL in vs else PASS for p, vs in got.items() if p}


def compare(ids: dict[str, str], old: dict[str, str], new: dict[str, str]) -> dict:
    units: dict[str, tuple[str, str]] = {}
    jobs: dict[str, list[str]] = {}
    split: dict[str, list[str]] = {}
    only = []
    by_part = {p.partition(":")[0] for p in old if ":" in p}
    for tid, key in ids.items():
        v, job = new.get(key, PENDING), tid.partition(":")[0]
        jobs.setdefault(job, []).append(v)
        if tid in old:
            units[tid] = (old[tid], v)
        elif (whole := SPLIT_RE.sub("", tid)) != tid and whole != job and whole in old:
            split.setdefault(whole, []).append(v)
        elif tid == job or job in by_part or job not in old:
            units[tid] = (NONE, v)
            only.append(tid)

    def folded(vs: list[str]) -> str:
        return FAIL if FAIL in vs else PENDING if PENDING in vs else PASS

    for job, vs in jobs.items():
        if job in old:
            units[job] = (old[job], folded(vs))
    units.update({whole: (old[whole], folded(vs)) for whole, vs in split.items()})
    missing = sorted(p for p in old if p not in units)
    units.update({p: (old[p], MISSING) for p in missing})
    rows = sorted(units.items())
    return {"units": len(rows), "agree": sum(o == n for _, (o, n) in rows),
            "pending": sorted(u for u, (o, n) in rows if n == PENDING and o != NONE),
            "disagree": [{"unit": u, "old": o, "executor": n} for u, (o, n) in rows
                         if o != n and (n != PENDING or o == NONE)],
            "missing": missing, "executor_only": sorted(only)}


def agrees(rec: dict) -> bool:
    return (not rec.get("error") and not rec.get("incomplete") and rec.get("units", 0) > 0
            and rec.get("agree") == rec.get("units") and not rec.get("pending") and not rec.get("disagree")
            and rec.get("missing") == [] and rec.get("executor_only") == [])


@dataclass
class Shadowed:
    tip: str
    ts: str
    rec: dict


def shadow_records(store, pull: bool = True, log: Callable[[str], None] = print) -> list[Shadowed]:
    from .store import ids, labels_sync

    if pull and getattr(store, "remote", None) is not None:
        try:
            labels_sync.pull_prefix(store, "commit:")
        except Exception as e:  # noqa: BLE001  (the local labels still answer, said so)
            log(f"research queue shadow-report: the store's remote wasn't read ({type(e).__name__}: {e}); local records only")
    out = []
    for d in sorted((Path(store.root) / "labels").glob(f"{ids.target_dirname('commit:')}*")):
        tip = d.name.partition("_")[2]
        if labs := store.labels(f"commit:{tip}", key=vocab.EXECUTOR_SHADOW):
            out.append(Shadowed(tip, labs[-1].ts, labs[-1].value))
    return sorted(out, key=lambda s: s.ts, reverse=True)


def shadow_report(records: list[Shadowed], tips: int) -> tuple[str, bool]:
    shown = records[:tips]
    streak = next((i for i, s in enumerate(records) if not agrees(s.rec)), len(records))
    lines = [f"executor shadow: the newest {len(shown)} of {len(records)} checked tips compared"]
    for s in shown:
        r = s.rec
        what = (f"error: {r['error']}" if r.get("error") else f"incomplete: {r['incomplete']}" if r.get("incomplete") else
                f"agree {r.get('agree', 0)}/{r.get('units', 0)}, disagree {len(r.get('disagree') or ())}"
                + (f", pending {len(r['pending'])}" if r.get("pending") else "")
                + ("" if "missing" in r else ", no coverage counts")
                + (f", missing in executor {len(r['missing'])}" if r.get("missing") else "")
                + (f", executor only {len(r['executor_only'])}" if r.get("executor_only") else ""))
        lines.append(f"{s.tip[:12]}  {s.ts}  {what}  (run {r.get('run') or '-'})")
        for x in r.get("disagree") or ():
            lines.append(f"    {x['unit']}: " + ("missing in executor (run " + x["old"] + ")" if x["executor"] == MISSING else
                                                 f"executor only (executor {x['executor']})" if x["old"] == NONE else
                                                 f"run {x['old']}, executor {x['executor']}"))
        lines += [f"    {u}: executor pending" for u in r.get("pending") or ()]
    ok = streak >= SWITCH_STREAK
    lines.append(f"switch gate ({SWITCH_STREAK} tips in a row agreeing fully, none missing in executor or executor only, none "
                 f"incomplete): {'met' if ok else 'not met'} "
                 f"(the newest {streak} agree)")
    return "\n".join(lines), ok

