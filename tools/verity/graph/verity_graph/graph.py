"""The task graph of a checkout (graph/v2): each family's tasks (`families`), completed from its `tasks.toml` table and checked
against the schema, the digest of every file a task or an env reads, each env's digest, and the key of each task that reads no
artifact (`contract.key`).  It reads only the tree: the bytes depend on the commit alone, not on the checkout's path, the
machine, the environment or the time."""
from __future__ import annotations

import fnmatch
import heapq
import json
import operator
import subprocess
import tomllib
from collections.abc import Iterable, Iterator
from pathlib import Path

from . import contract, families

#: this package's directory in the tree, its families' table and its runners
HOME = "tools/verity/graph"
SPEC = f"{HOME}/tasks.toml"
RUNNERS = f"{HOME}/verity_graph/runners"
#: a graph/v2 task's fields, in the order the listing writes them; a lister gives all but the core's (`CORE_FIELDS`)
TASK_FIELDS = ("id", "kind", "about", "env", "argv", "vars", "inputs", "reads", "outputs", "resources", "timeout_s", "priority",
               "key")
CORE_FIELDS = ("priority", "key")
FAMILY_FIELDS = {"files": list, "kinds": list, "env": str, "about": str, "resources": dict, "timeout_s": int}
RESOURCES = ("cores", "mem_gb", "disk_gb")
#: what `[slot]` bounds: the largest slot an executor leases a task, which no task's resources may pass
SLOT = ("cores", "mem_gb")
#: what `verity graph run` writes in a task's work directory beside its outputs and reads
LOG = "task.log"
#: a field name no table, task or result may have: who is paged is drive's policy, from paths
FORBIDDEN = "owner"
#: a task's fields `complete` holds to an exact shape, which no `owner` field gets past
SHAPED = ("inputs", "reads", "outputs")
READ = frozenset(("task", "output"))
_read = operator.itemgetter("task", "output")


class GraphError(ValueError):
    """The tree's families or tasks don't make a graph/v2."""


def matches(path: str, pattern: str) -> bool:
    """``pattern`` (fnmatch over the whole path, so `*` crosses `/`; a leading `**/` also matches at the root) covers ``path``."""
    return fnmatch.fnmatchcase(path, pattern) or (pattern.startswith("**/") and fnmatch.fnmatchcase(path, pattern[3:]))


def git(root: Path, *argv: str) -> bytes:
    r = subprocess.run(["git", "-C", str(root), *argv], capture_output=True)
    if r.returncode:
        raise GraphError(f"git {' '.join(argv)} in {root} exited {r.returncode}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout


def tracked(root: Path) -> list[str]:
    """The tree's files as git has them, tracked and untracked-unignored (POSIX, relative, sorted), but those deleted."""
    out = git(root, "ls-files", "-z", "-co", "--exclude-standard").split(b"\0")
    return sorted({p for p in (x.decode() for x in out if x) if (root / p).is_file()})


def commit(root: Path) -> str | None:
    """HEAD's commit when the tree is clean (no change to a tracked file, no untracked-unignored file), else None: a dirty tree's
    graph names no commit, so it is never taken for one."""
    if git(root, "--no-optional-locks", "status", "--porcelain", "-z", "--untracked-files=normal"):
        return None
    return git(root, "rev-parse", "HEAD").decode().strip()


def _forbid(where: str, doc: object) -> None:
    if isinstance(doc, dict):
        if FORBIDDEN in doc:
            raise GraphError(f"{where} has an `{FORBIDDEN}` field: who is paged is drive's policy, from paths, never the graph's")
        for k, v in doc.items():
            _forbid(f"{where}.{k}", v)
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            _forbid(f"{where}[{i}]", v)


def spec(root: Path) -> dict:
    """`tasks.toml`, checked: {"slot": {cores, mem_gb}, "env": {name: {"spec": [glob]}}, "family": {name: table}}."""
    try:
        doc = tomllib.loads((root / SPEC).read_text())
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise GraphError(f"{SPEC}: {e}") from e
    _forbid(SPEC, doc)
    if extra := set(doc) - {"slot", "env", "family"}:
        raise GraphError(f"{SPEC}: tables {', '.join(sorted(extra))}: it holds [slot], [env.*] and [family.*] only")
    slot = doc.get("slot")
    if not isinstance(slot, dict) or set(slot) != set(SLOT) or not all(
            isinstance(v, int | float) and not isinstance(v, bool) and v > 0 for v in slot.values()):
        raise GraphError(f"{SPEC}: [slot] holds {' and '.join(SLOT)}, each a positive number: the largest slot an executor "
                         "leases a task")
    envs, fams = doc.get("env", {}), doc.get("family", {})
    for name, e in envs.items():
        if set(e) != {"spec"} or not isinstance(e["spec"], list) or not all(isinstance(p, str) for p in e["spec"]):
            raise GraphError(f"{SPEC}: [env.{name}] holds `spec`, a list of paths and globs, alone")
    for name, t in fams.items():
        for f, ty in FAMILY_FIELDS.items():
            if not isinstance(t.get(f), ty):
                raise GraphError(f"{SPEC}: [family.{name}] needs `{f}` ({ty.__name__})")
        if extra := set(t) - set(FAMILY_FIELDS):
            raise GraphError(f"{SPEC}: [family.{name}] fields {', '.join(sorted(extra))}: it has {', '.join(FAMILY_FIELDS)}")
        if not t["kinds"] or t["env"] not in envs:
            raise GraphError(f"{SPEC}: [family.{name}] needs a kind and an env of [env.*]")
        _resources(f"[family.{name}]", t["resources"], slot)
    return {"slot": slot, "env": envs, "family": fams}


def _resources(where: str, r: dict, slot: dict) -> None:
    if set(r) != set(RESOURCES) or not all(isinstance(v, int | float) and not isinstance(v, bool) and v > 0 for v in r.values()):
        raise GraphError(f"{where}: resources are {', '.join(RESOURCES)}, each a positive number")
    if over := [f"{k} {r[k]} (its largest {slot[k]})" for k in SLOT if r[k] > slot[k]]:
        raise GraphError(f"{where}: asks {', '.join(over)}, more than {SPEC}'s [slot], the largest slot an executor leases a "
                         "task: no slot would take it, so it and every task reading it would wait queued. Ask about what "
                         "its passing runs reach, or raise [slot] once the executor's slots are that large")


def covered(root: Path, family: str, files: list[str]) -> list[str]:
    """The files of ``files`` that `[family.<family>]`'s `files` globs cover: the ones its lister makes tasks of."""
    pats = spec(root)["family"][family]["files"]
    return [f for f in files if any(matches(f, p) for p in pats)]


def runner_sources(kind: str) -> list[str]:
    """A kind's runner, its module and the dispatch beside it: inputs of every task of the kind."""
    return [f"{RUNNERS}/__init__.py", f"{RUNNERS}/{kind.replace('-', '_')}.py"]


def _path(where: str, p: object) -> str:
    if not isinstance(p, str) or not p or p.startswith("/") or "\\" in p or any(x in ("", ".", "..") for x in p.split("/")):
        raise GraphError(f"{where}: {p!r} is not a relative POSIX path inside the work directory")
    return p


def complete(t: object, family: str, table: dict, envs: dict, files: set[str], slot: dict) -> dict:
    """``t``, a task family ``family``'s lister gave, with what it left out taken from its table, checked, its resources within
    ``slot`` (`[slot]`)."""
    if not isinstance(t, dict):
        raise GraphError(f"family {family} listed {t!r}, not a task")
    _forbid(f"family {family}'s task {t.get('id')!r}", {k: v for k, v in t.items() if k not in SHAPED})
    tid = t.get("id")
    if extra := set(t) - (set(TASK_FIELDS) - set(CORE_FIELDS)):
        raise GraphError(f"{tid}: fields {', '.join(sorted(extra))} aren't graph/v2's (a lister leaves `priority` and `key` to "
                         "the core)")
    kind = t.get("kind", table["kinds"][0])
    if kind not in table["kinds"]:
        raise GraphError(f"{tid}: kind {kind!r} isn't one of [family.{family}]'s kinds ({', '.join(table['kinds'])})")
    if not isinstance(tid, str) or not tid.startswith(f"{kind}:") or len(tid) == len(kind) + 1:
        raise GraphError(f"{tid!r}: a task's id is `<kind>:<name>`, here `{kind}:...`")
    env = t.get("env", table["env"])
    if env not in envs:
        raise GraphError(f"{tid}: env {env!r} isn't one of [env.*] ({', '.join(envs)})")
    argv = t.get("argv", ["verity", "graph", "run", tid])
    v = t.get("vars", {})
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
        raise GraphError(f"{tid}: argv is a non-empty list of strings")
    if not isinstance(v, dict) or not all(isinstance(k, str) and isinstance(x, str) for k, x in v.items()):
        raise GraphError(f"{tid}: vars is {{name: string}}")
    inputs = t.get("inputs", [])
    if not isinstance(inputs, list) or not all(isinstance(p, str) for p in inputs) or len(set(inputs)) != len(inputs):
        raise GraphError(f"{tid}: inputs is a list of distinct paths")
    inputs = sorted({*inputs, *runner_sources(kind)})
    if missing := [p for p in inputs if p not in files]:
        raise GraphError(f"{tid}: inputs {', '.join(missing[:5])} aren't files of the tree"
                         + ("" if missing[0] not in runner_sources(kind) else f" (kind {kind} has no runner)"))
    reads = t.get("reads", [])
    if not isinstance(reads, list) or not all(isinstance(r, dict) and r.keys() == READ and isinstance(r["task"], str)
                                              and isinstance(r["output"], str) for r in reads):
        raise GraphError(f"{tid}: reads is a list of {{task, output}}")
    reads = sorted(reads, key=_read)
    if len(set(map(_read, reads))) != len(reads):
        raise GraphError(f"{tid}: a read is listed twice")
    outputs = t.get("outputs", [])
    if not isinstance(outputs, list) or not all(isinstance(o, dict) and set(o) == {"name", "path"} for o in outputs):
        raise GraphError(f"{tid}: outputs is a list of {{name, path}}")
    if len({o["name"] for o in outputs}) != len(outputs) or not all(isinstance(o["name"], str) and o["name"] for o in outputs):
        raise GraphError(f"{tid}: outputs need distinct names")
    for o in outputs:
        _path(f"{tid}'s output {o['name']}", o["path"])
    resources = {**table["resources"], **t.get("resources", {})}
    _resources(tid, resources, slot)
    timeout = t.get("timeout_s", table["timeout_s"])
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
        raise GraphError(f"{tid}: timeout_s is a positive integer")
    about = t.get("about")
    if about is None:
        try:
            about = table["about"].format(id=tid, kind=kind, name=tid[len(kind) + 1:], **v)
        except (KeyError, IndexError, ValueError) as e:
            raise GraphError(f"[family.{family}]'s about {table['about']!r} names {e}, which {tid} lacks") from e
    if not isinstance(about, str) or not about:
        raise GraphError(f"{tid}: about is a non-empty string")
    return {"id": tid, "kind": kind, "about": about, "env": env, "argv": list(argv), "vars": dict(v), "inputs": inputs,
            "reads": reads, "outputs": sorted(({"name": o["name"], "path": o["path"]} for o in outputs), key=lambda o: o["name"]),
            "resources": resources, "timeout_s": timeout, "priority": 0, "key": None}


def family_of(kind: str, fams: dict) -> str:
    """The family whose table lists ``kind``."""
    owners = [n for n, t in fams.items() if kind in t["kinds"]]
    if len(owners) != 1:
        raise GraphError(f"kind {kind!r} is in {'no' if not owners else ' and '.join(owners)} [family.*] table's kinds")
    return owners[0]


def order(tasks: list[dict]) -> list[dict]:
    """``tasks`` with every task after the tasks it reads, ties by id; GraphError on a cycle or a read of a task not among them."""
    by = {t["id"]: t for t in tasks}
    need = {t["id"]: {r["task"] for r in t["reads"]} for t in tasks}
    if bad := sorted(f"{i} reads {d}" for i, ds in need.items() for d in ds if d not in by):
        raise GraphError(f"reads of tasks not in the graph: {'; '.join(bad[:5])}")
    done: list[dict] = []
    ready = [i for i, ds in need.items() if not ds]
    heapq.heapify(ready)
    readers: dict[str, list[str]] = {}
    for i, ds in need.items():
        for d in ds:
            readers.setdefault(d, []).append(i)
    left = {i: len(ds) for i, ds in need.items()}
    while ready:
        i = heapq.heappop(ready)
        done.append(by[i])
        for r in readers.get(i, ()):
            left[r] -= 1
            if not left[r]:
                heapq.heappush(ready, r)
    if len(done) != len(tasks):
        raise GraphError(f"the reads make a cycle through {', '.join(sorted(set(by) - {t['id'] for t in done})[:5])}")
    return done


def priorities(tasks: list[dict]) -> dict[str, int]:
    """Each task's priority: how many reads the longest chain of them through it makes, the tasks that wait on it one after
    another and those it waits on so; 0 for a task that reads nothing and that nothing reads.  The Lean modules under a lock
    and the lock itself have the graph's highest."""
    done = order(tasks)
    readers: dict[str, list[str]] = {}
    for t in tasks:
        for r in t["reads"]:
            readers.setdefault(r["task"], []).append(t["id"])
    below: dict[str, int] = {}
    for t in done:
        below[t["id"]] = max((below[r["task"]] + 1 for r in t["reads"]), default=0)
    above: dict[str, int] = {}
    for t in reversed(done):
        above[t["id"]] = max((above[r] + 1 for r in readers.get(t["id"], ())), default=0)
    return {i: below[i] + above[i] for i in below}


def _check_reads(tasks: list[dict]) -> None:
    """Each task's reads name outputs of ``tasks``, and its outputs, reads and log land on distinct paths of its work directory,
    none inside another (`_misread` says which don't)."""
    out = {t["id"]: {o["name"]: o["path"] for o in t["outputs"]} for t in tasks}
    up = {p: p.rpartition("/")[0] for o in out.values() for p in o.values()} | {LOG: LOG.rpartition("/")[0]}
    dirs = {d: tuple("/".join(d.split("/")[:i]) for i in range(1, d.count("/") + 2)) if d else () for d in set(up.values())}
    by = {t["id"]: t for t in tasks}
    for t in tasks:
        try:
            paths = [*out[t["id"]].values(), LOG, *[out[r["task"]][r["output"]] for r in t["reads"]]]
        except KeyError:
            _misread(t, by)
            continue
        at = set(paths)
        above: set[str] = set()
        for d in {up[p] for p in at}:
            above.update(dirs[d])
        if len(at) != len(paths) or not above.isdisjoint(at):
            _misread(t, by)


def _misread(t: dict, by: dict) -> None:
    paths = [(o["path"], f"its output {o['name']}") for o in t["outputs"]] + [(LOG, "the log")]
    for r in t["reads"]:
        src = by.get(r["task"])
        out = next((o for o in (src or {}).get("outputs", []) if o["name"] == r["output"]), None)
        if out is None:
            raise GraphError(f"{t['id']} reads {r['task']}'s {r['output']}, which "
                             + ("isn't a task" if src is None else "it doesn't output"))
        paths.append((out["path"], f"{r['task']}'s {r['output']}"))
    at: dict[str, str] = {}
    for p, what in paths:
        if p in at:
            raise GraphError(f"{t['id']}: {at[p]} and {what} both land at {p} in its work directory")
        at[p] = what
    for p, what in paths:
        parts = p.split("/")
        if up := next((q for q in ("/".join(parts[:i]) for i in range(1, len(parts))) if q in at), None):
            raise GraphError(f"{t['id']}: {what} lands inside {at[up]} ({up}) in its work directory")


def listing(root: Path, only: set[str] | None = None, listers: dict | None = None) -> dict:
    """The graph/v2 document of the tree at ``root``: every family's tasks, or with ``only`` those families' and every task
    they read, transitively.  ``listers`` ({family: tasks(root, files)}) defaults to `families.listers()`."""
    root = Path(root)
    s = spec(root)
    fams = s["family"]
    listers = families.listers() if listers is None else listers
    if extra := sorted(set(listers) - set(fams)):
        raise GraphError(f"families/{', '.join(extra)}: a lister with no [family.*] table in {SPEC}")
    if only is not None and (unknown := sorted(set(only) - set(fams))):
        raise GraphError(f"no family {', '.join(unknown)}: {SPEC} declares {', '.join(sorted(fams))}")
    files = tracked(root)
    have = set(files)
    tasks: dict[str, dict] = {}
    todo = sorted(fams if only is None else only)
    listed: set[str] = set()
    while todo:
        name = todo.pop(0)
        listed.add(name)
        if name not in listers:
            continue
        new = [complete(t, name, fams[name], s["env"], have, s["slot"]) for t in listers[name](root, files)]
        for c in new:
            if c["id"] in tasks:
                raise GraphError(f"{c['id']} is listed twice")
            tasks[c["id"]] = c
        kinds = {i.partition(":")[0] for i in {r["task"] for c in new for r in c["reads"]}}
        todo = sorted(set(todo) | ({family_of(k, fams) for k in kinds} - listed))
    return _document(root, s, files, sorted(tasks.values(), key=lambda t: t["id"]), commit(root))


def task_graph(root: Path, task_id: str, listers: dict | None = None) -> dict:
    """The part of the tree's graph/v2 that running ``task_id`` needs, as `listing` makes it: that task and every task it reads,
    transitively, the digests of their inputs and the envs, without completing, digesting or keying the rest of their families
    (`verity graph run`).  Its `commit` is null and its priorities count only the chains it holds, none of the tasks that read
    ``task_id``, since nothing that runs a task reads either."""
    root = Path(root)
    s = spec(root)
    fams = s["family"]
    listers = families.listers() if listers is None else listers
    files = tracked(root)
    have = set(files)
    listed: dict[str, dict] = {}
    tasks: dict[str, dict] = {}
    todo = [task_id]
    while todo:
        tid = todo.pop()
        if tid in tasks:
            continue
        name = family_of(tid.partition(":")[0], fams)
        if name not in listed:
            listed[name] = {t.get("id"): t for t in (listers[name](root, files) if name in listers else []) if isinstance(t, dict)}
        if tid not in listed[name]:
            raise GraphError(f"no task {tid} on this tree: `verity graph` lists them")
        tasks[tid] = complete(listed[name][tid], name, fams[name], s["env"], have, s["slot"])
        todo += [r["task"] for r in tasks[tid]["reads"]]
    return _document(root, s, files, sorted(tasks.values(), key=lambda t: t["id"]), None)


def part(doc: dict, task_id: str, by: dict | None = None) -> dict:
    """What running ``task_id`` reads of the graph/v2 ``doc`` (``by``: its tasks by id), which an executor that listed once
    hands the run (`verity graph run --graph`): the task, its env, the digests of its inputs and its env's spec files, and of
    each task it reads the id and the outputs it reads.  It grows with the task's own reads, not with every task under it."""
    by = {t["id"]: t for t in doc["tasks"]} if by is None else by
    task = by[task_id]
    read: dict[str, set[str]] = {}
    for r in task["reads"]:
        read.setdefault(r["task"], set()).add(r["output"])
    env = doc["envs"][task["env"]]
    producers = [{"id": i, "outputs": [o for o in by[i]["outputs"] if o["name"] in names]} for i, names in read.items()]
    return {"schema": contract.SCHEMA_GRAPH, "commit": doc["commit"], "envs": {task["env"]: env},
            "files": {p: doc["files"][p] for p in sorted({*task["inputs"], *env["spec"]})},
            "tasks": sorted([task, *producers], key=lambda t: t["id"])}


def given(root: Path, path: Path, task_id: str) -> dict:
    """The graph/v2 at ``path`` (`verity graph run --graph`): the listing an executor made once, packed or not (`unpack`), or
    its `part` for ``task_id``, so that no lister runs.  Taken when it holds the task, the outputs the task reads and its env,
    and each file the task's key reads (its inputs and its env's spec) has the digest it names here, so that no result is
    keyed from another tree's bytes."""
    try:
        doc = json.loads(Path(path).read_bytes())
    except (OSError, ValueError) as e:
        raise GraphError(f"--graph {path}: {e}") from e
    try:
        if doc["schema"] == contract.SCHEMA_PACKED:
            doc = unpack(doc)
        if doc["schema"] != contract.SCHEMA_GRAPH:
            raise ValueError(f"schema {doc['schema']!r}")
        by = {t["id"]: t for t in doc["tasks"]}
        if (task := by.get(task_id)) is not None:
            spec, env = doc["envs"][task["env"]]["spec"], doc["envs"][task["env"]]["digest"]
            lacks = [f"{r['task']}'s {r['output']}" for r in task["reads"]
                     if r["output"] not in {o["name"] for o in by.get(r["task"], {}).get("outputs", [])}]
            need = sorted({*task["inputs"], *spec})
            named = {p: doc["files"].get(p) for p in need}
    except (KeyError, TypeError, AttributeError, ValueError) as e:
        raise GraphError(f"--graph {path} isn't graph/v2 ({type(e).__name__}: {e}): `verity graph --json` lists it, or run "
                         "without --graph") from e
    if task is None:
        raise GraphError(f"no task {task_id} in --graph {path}")
    if lacks:
        raise GraphError(f"--graph {path} lacks {', '.join(lacks[:5])}, which {task_id} reads")
    if stale := [p for p in need if not (root / p).is_file() or contract.file_digest(root / p) != named[p]]:
        raise GraphError(f"--graph {path} isn't this checkout's: {', '.join(stale[:5])} "
                         f"{'differs' if len(stale) == 1 else 'differ'} here from its digests (`verity graph --json` lists "
                         "this checkout, or run without --graph)")
    if env != contract.env_digest(spec, named):
        raise GraphError(f"--graph {path}: env {task['env']}'s digest isn't the one its spec files give")
    return doc


def _runs(indices: Iterable[int]) -> list[int]:
    """Increasing ``indices`` as runs of consecutive ones, two integers a run: how far past the previous run's end it starts
    (the first, past 0), and its length."""
    out: list[int] = []
    end = 0
    for i in indices:
        if out and i == end:
            out[-1] += 1
        else:
            out += [i - end, 1]
        end = i + 1
    return out


def _indices(runs: object) -> Iterator[int]:
    """The indices `_runs` gave as ``runs``; ValueError on what it never gives."""
    if not isinstance(runs, list) or len(runs) % 2 or not all(type(x) is int for x in runs):
        raise ValueError(f"runs {str(runs)[:40]} aren't pairs of integers")
    end = 0
    for gap, n in zip(runs[::2], runs[1::2]):
        if gap < 0 or n < 1:
            raise ValueError(f"a run starts {gap} past the last and is {n} long")
        yield from range(end + gap, end + gap + n)
        end += gap + n


def pack(doc: dict) -> dict:
    """The listing ``doc`` packed (`contract.SCHEMA_PACKED`), which `unpack` makes ``doc`` again: each task's `inputs` as runs
    of indices into `files`' paths in order (`_runs`), and its `reads` as {output name: runs of indices into `tasks`}.  A
    suite's test files share most of their thousands of inputs, and a Lean module reads two outputs of each of the hundreds
    of modules its compile loads: graph/v2 spells out each path and task id, where here an input or a read is a few bytes,
    most of them in runs of neighbours."""
    at = {p: i for i, p in enumerate(sorted(doc["files"]))}
    of = {t["id"]: i for i, t in enumerate(doc["tasks"])}
    tasks = []
    for t in doc["tasks"]:
        if t["inputs"] != sorted(set(t["inputs"])) or list(map(_read, t["reads"])) != sorted(set(map(_read, t["reads"]))):
            raise GraphError(f"{t['id']}: its inputs and its reads aren't graph/v2's, sorted and each once")
        reads: dict[str, list[int]] = {}
        for r in t["reads"]:
            reads.setdefault(r["output"], []).append(of[r["task"]])
        tasks.append({**t, "inputs": _runs(at[p] for p in t["inputs"]),
                      "reads": {o: _runs(ix) for o, ix in sorted(reads.items())}})
    return {**doc, "schema": contract.SCHEMA_PACKED, "tasks": tasks}


def unpack(doc: dict) -> dict:
    """The listing `pack` packed as ``doc``, graph/v2 again; GraphError when ``doc`` is no packed listing."""
    try:
        if doc["schema"] != contract.SCHEMA_PACKED:
            raise ValueError(f"schema {doc['schema']!r}")
        paths, ids = sorted(doc["files"]), [t["id"] for t in doc["tasks"]]
        tasks = [{**t, "inputs": [paths[i] for i in _indices(t["inputs"])],
                  "reads": sorted(({"task": ids[i], "output": o} for o, runs in t["reads"].items() for i in _indices(runs)),
                                  key=_read)} for t in doc["tasks"]]
    except (KeyError, TypeError, AttributeError, IndexError, ValueError) as e:
        raise GraphError(f"no packed listing ({type(e).__name__}: {e})") from e
    return {**doc, "schema": contract.SCHEMA_GRAPH, "tasks": tasks}


def _document(root: Path, s: dict, files: list[str], ts: list[dict], at: str | None) -> dict:
    """graph/v2 of the completed tasks ``ts``, sorted by id, at commit ``at``: their reads checked, each one's priority among
    them (`priorities`), the digests of their inputs and of every env's spec files, each env's digest, and the key of each task
    that reads no artifact."""
    _check_reads(ts)
    rank = priorities(ts)
    envs = {name: sorted(f for f in files if any(matches(f, p) for p in e["spec"])) for name, e in sorted(s["env"].items())}
    needed = sorted({p for t in ts for p in t["inputs"]} | {p for spec_files in envs.values() for p in spec_files})
    digests = {p: contract.file_digest(root / p) for p in needed}
    env_docs = {name: {"spec": sp, "digest": contract.env_digest(sp, digests)} for name, sp in envs.items()}
    key = contract.keyer(digests)
    for t in ts:
        t["priority"] = rank[t["id"]]
        if not t["reads"]:
            t["key"] = key(t, None, env_docs[t["env"]]["digest"])
    return {"schema": contract.SCHEMA_GRAPH, "commit": at, "envs": env_docs, "files": digests, "tasks": ts}
