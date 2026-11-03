"""The task graph of a checkout (graph/v2): each family's tasks (`families`), completed from its `tasks.toml` table and checked
against the schema, the digest of every file a task or an env reads, each env's digest, and the key of each task that reads no
artifact (`contract.key`).  It reads only the tree: the bytes depend on the commit alone, not on the checkout's path, the
machine, the environment or the time."""
from __future__ import annotations

import fnmatch
import subprocess
import tomllib
from heapq import heapify, heappop, heappush
from itertools import chain
from operator import itemgetter, methodcaller
from pathlib import Path

from . import contract, families

#: this package's directory in the tree, its families' table and its runners
HOME = "tools/verity/graph"
SPEC = f"{HOME}/tasks.toml"
RUNNERS = f"{HOME}/verity_graph/runners"
#: a graph/v2 task's fields, in the order the listing writes them; a lister gives all but `key`
TASK_FIELDS = ("id", "kind", "about", "env", "argv", "vars", "inputs", "reads", "outputs", "resources", "timeout_s", "key")
READ_FIELDS = {"task", "output"}
FAMILY_FIELDS = {"files": list, "kinds": list, "env": str, "about": str, "resources": dict, "timeout_s": int}
RESOURCES = ("cores", "mem_gb", "disk_gb")
#: what `verity graph run` writes in a task's work directory beside its outputs and reads
LOG = "task.log"
#: a field name no table, task or result may have: who is paged is drive's policy, from paths
FORBIDDEN = "owner"
_SCALARS = {str, int, float, bool, type(None)}


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
    """GraphError naming the first place in ``doc`` a dict holds a FORBIDDEN key. The search takes a list of scalars, or of
    plain dicts, whole: a task reads thousands of outputs."""
    stack = [doc]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            if FORBIDDEN in x:
                _name(where, doc)
            x = x.values()
        elif not isinstance(x, list):
            continue
        kinds = set(map(type, x))
        if kinds <= _SCALARS:
            continue
        if kinds == {dict}:
            if FORBIDDEN in set().union(*x):
                _name(where, doc)
            stack.append(list(chain.from_iterable(map(dict.values, x))))
        else:
            stack += x


def _name(where: str, doc: object) -> None:
    if isinstance(doc, dict):
        if FORBIDDEN in doc:
            raise GraphError(f"{where} has an `{FORBIDDEN}` field: who is paged is drive's policy, from paths, never the graph's")
        for k, v in doc.items():
            _name(f"{where}.{k}", v)
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            _name(f"{where}[{i}]", v)


def spec(root: Path) -> dict:
    """`tasks.toml`, checked: {"env": {name: {"spec": [glob]}}, "family": {name: table}}."""
    try:
        doc = tomllib.loads((root / SPEC).read_text())
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise GraphError(f"{SPEC}: {e}") from e
    _forbid(SPEC, doc)
    if extra := set(doc) - {"env", "family"}:
        raise GraphError(f"{SPEC}: tables {', '.join(sorted(extra))}: it holds [env.*] and [family.*] only")
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
        _resources(f"[family.{name}]", t["resources"])
    return {"env": envs, "family": fams}


def _resources(where: str, r: dict) -> None:
    if set(r) != set(RESOURCES) or not all(isinstance(v, int | float) and not isinstance(v, bool) and v > 0 for v in r.values()):
        raise GraphError(f"{where}: resources are {', '.join(RESOURCES)}, each a positive number")


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


def _all(xs: list, ty: type) -> bool:
    """Every one of ``xs`` is a ``ty``, decided by their exact types first, which takes thousands at C speed."""
    return set(map(type, xs)) <= {ty} or all(isinstance(x, ty) for x in xs)


def _plain_reads(reads: list) -> bool:
    """Every one of ``reads`` is a plain dict of `task` and `output`, each a plain string: a task's thousands, at C speed."""
    return (set(map(type, reads)) <= {dict} and set(map(frozenset, reads)) <= {frozenset(READ_FIELDS)}
            and set(map(type, chain.from_iterable(map(dict.values, reads)))) <= {str})


def complete(t: object, family: str, table: dict, envs: dict, files: set[str]) -> dict:
    """``t``, a task family ``family``'s lister gave, with what it left out taken from its table, checked."""
    if not isinstance(t, dict):
        raise GraphError(f"family {family} listed {t!r}, not a task")
    _forbid(f"family {family}'s task {t.get('id')!r}", t)
    tid = t.get("id")
    if extra := set(t) - set(TASK_FIELDS[:-1]):
        raise GraphError(f"{tid}: fields {', '.join(sorted(extra))} aren't graph/v2's (a lister leaves `key` to the core)")
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
    if not isinstance(inputs, list) or not _all(inputs, str) or len(set(inputs)) != len(inputs):
        raise GraphError(f"{tid}: inputs is a list of distinct paths")
    inputs = sorted({*inputs, *runner_sources(kind)})
    if not files.issuperset(inputs):
        missing = [p for p in inputs if p not in files]
        raise GraphError(f"{tid}: inputs {', '.join(missing[:5])} aren't files of the tree"
                         + ("" if missing[0] not in runner_sources(kind) else f" (kind {kind} has no runner)"))
    reads = t.get("reads", [])
    if not isinstance(reads, list) or not (_plain_reads(reads) or all(
            isinstance(r, dict) and r.keys() == READ_FIELDS and isinstance(r["task"], str) and isinstance(r["output"], str)
            for r in reads)):
        raise GraphError(f"{tid}: reads is a list of {{task, output}}")
    pairs = sorted(map(itemgetter("task", "output"), reads))
    if len(set(pairs)) != len(pairs):
        raise GraphError(f"{tid}: a read is listed twice")
    outputs = t.get("outputs", [])
    if not isinstance(outputs, list) or not all(isinstance(o, dict) and set(o) == {"name", "path"} for o in outputs):
        raise GraphError(f"{tid}: outputs is a list of {{name, path}}")
    if len({o["name"] for o in outputs}) != len(outputs) or not all(isinstance(o["name"], str) and o["name"] for o in outputs):
        raise GraphError(f"{tid}: outputs need distinct names")
    for o in outputs:
        _path(f"{tid}'s output {o['name']}", o["path"])
    resources = {**table["resources"], **t.get("resources", {})}
    _resources(tid, resources)
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
            "reads": [{"task": a, "output": b} for a, b in pairs],
            "outputs": sorted(({"name": o["name"], "path": o["path"]} for o in outputs), key=lambda o: o["name"]),
            "resources": resources, "timeout_s": timeout, "key": None}


def family_of(kind: str, fams: dict) -> str:
    """The family whose table lists ``kind``."""
    owners = [n for n, t in fams.items() if kind in t["kinds"]]
    if len(owners) != 1:
        raise GraphError(f"kind {kind!r} is in {'no' if not owners else ' and '.join(owners)} [family.*] table's kinds")
    return owners[0]


def order(tasks: list[dict]) -> list[dict]:
    """``tasks`` with every task after the tasks it reads, ties by id; GraphError on a cycle or a read of a task not among them."""
    by = {t["id"]: t for t in tasks}
    need = {t["id"]: set(map(itemgetter("task"), t["reads"])) for t in tasks}
    if bad := sorted(f"{i} reads {d}" for i, ds in need.items() for d in ds - by.keys()):
        raise GraphError(f"reads of tasks not in the graph: {'; '.join(bad[:5])}")
    done: list[dict] = []
    seen: set[str] = set()
    ready = [i for i, ds in need.items() if not ds]
    heapify(ready)
    readers: dict[str, list[str]] = {}
    for i, ds in need.items():
        for d in ds:
            readers.setdefault(d, []).append(i)
    left = {i: len(ds) for i, ds in need.items()}
    while ready:
        i = heappop(ready)
        done.append(by[i])
        seen.add(i)
        for r in readers.get(i, ()):
            left[r] -= 1
            if not left[r]:
                heappush(ready, r)
    if len(done) != len(tasks):
        raise GraphError(f"the reads make a cycle through {', '.join(sorted(set(by) - seen)[:5])}")
    return done


def _landing(e: tuple[str, str | None, str | None]) -> str:
    """What lands at a path of a task's work directory, for a message: (path, output name, None) its output, (LOG, None, None)
    the log, (path, task, output) a read."""
    _p, a, b = e
    return "the log" if a is None else f"its output {a}" if b is None else f"{a}'s {b}"


def _clash(paths: list[str]) -> bool:
    """Whether two of ``paths`` are one, or one is a directory above another."""
    if len(at := set(paths)) != len(paths):
        return True
    dirs: set[str] = set()
    for d in set(map(itemgetter(0), map(methodcaller("rpartition", "/"), paths))):
        while d and d not in dirs:
            dirs.add(d)
            d = d.rpartition("/")[0]
    return not at.isdisjoint(dirs)


def _check_reads(tasks: list[dict]) -> None:
    outs = {t["id"]: {o["name"]: o["path"] for o in t["outputs"]} for t in tasks}
    for t in tasks:
        paths = [o["path"] for o in t["outputs"]]
        paths.append(LOG)
        try:
            paths += [outs[r["task"]][r["output"]] for r in t["reads"]]
        except KeyError:
            _lands(t, outs)
        if _clash(paths):
            _lands(t, outs)


def _lands(t: dict, outs: dict[str, dict[str, str]]) -> None:
    """GraphError naming the first of task ``t``'s reads of no output, or the first two things that land on one path of its
    work directory or one inside the other."""
    lands: list[tuple[str, str | None, str | None]] = [(o["path"], o["name"], None) for o in t["outputs"]]
    lands.append((LOG, None, None))
    for r in t["reads"]:
        src = outs.get(r["task"])
        if (p := None if src is None else src.get(r["output"])) is None:
            raise GraphError(f"{t['id']} reads {r['task']}'s {r['output']}, which "
                             + ("isn't a task" if src is None else "it doesn't output"))
        lands.append((p, r["task"], r["output"]))
    at: dict[str, tuple[str, str | None, str | None]] = {}
    for e in lands:
        if e[0] in at:
            raise GraphError(f"{t['id']}: {_landing(at[e[0]])} and {_landing(e)} both land at {e[0]} in its work directory")
        at[e[0]] = e
    for e in lands:
        p = e[0]
        i = p.find("/")
        while i != -1:
            if (up := p[:i]) in at:
                raise GraphError(f"{t['id']}: {_landing(e)} lands inside {_landing(at[up])} ({up}) in its work directory")
            i = p.find("/", i + 1)


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
    wanted: set[str] = set()
    while todo:
        name = todo.pop(0)
        listed.add(name)
        if name not in listers:
            continue
        kinds: dict[str, None] = {}
        for t in listers[name](root, files):
            c = complete(t, name, fams[name], s["env"], have)
            if c["id"] in tasks:
                raise GraphError(f"{c['id']} is listed twice")
            tasks[c["id"]] = c
            kinds.update(dict.fromkeys(r["task"].partition(":")[0] for r in c["reads"]))
        wanted.update(family_of(k, fams) for k in kinds)
        todo = sorted(set(todo) | (wanted - listed))
    return _document(root, s, files, sorted(tasks.values(), key=lambda t: t["id"]), commit(root))


def task_graph(root: Path, task_id: str, listers: dict | None = None) -> dict:
    """The part of the tree's graph/v2 that running ``task_id`` needs, as `listing` makes it: that task and every task it reads,
    transitively, the digests of their inputs and the envs, without completing, digesting or keying the rest of their families
    (`verity graph run`).  Its `commit` is null, since nothing that runs a task reads one."""
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
        tasks[tid] = complete(listed[name][tid], name, fams[name], s["env"], have)
        todo += [r["task"] for r in tasks[tid]["reads"]]
    return _document(root, s, files, sorted(tasks.values(), key=lambda t: t["id"]), None)


def _document(root: Path, s: dict, files: list[str], ts: list[dict], at: str | None) -> dict:
    """graph/v2 of the completed tasks ``ts``, sorted by id, at commit ``at``: their reads checked, the digests of their inputs
    and of every env's spec files, each env's digest, and the key of each task that reads no artifact."""
    _check_reads(ts)
    order(ts)
    envs = {name: sorted(f for f in files if any(matches(f, p) for p in e["spec"])) for name, e in sorted(s["env"].items())}
    needed = sorted({p for t in ts for p in t["inputs"]} | {p for spec_files in envs.values() for p in spec_files})
    digests = {p: contract.file_digest(root / p) for p in needed}
    env_docs = {name: {"spec": sp, "digest": contract.env_digest(sp, digests)} for name, sp in envs.items()}
    key = contract.keyer(digests)
    for t in ts:
        if not t["reads"]:
            t["key"] = key(t, None, env_docs[t["env"]]["digest"])
    return {"schema": contract.SCHEMA_GRAPH, "commit": at, "envs": env_docs, "files": digests, "tasks": ts}


def closure(doc: dict, ids: set[str]) -> list[dict]:
    """The tasks of ``doc`` named by ``ids`` and every task they read, transitively, in `order`."""
    by = {t["id"]: t for t in doc["tasks"]}
    keep, todo = set(), sorted(ids)
    while todo:
        i = todo.pop()
        if i in keep:
            continue
        if i not in by:
            raise GraphError(f"no task {i} in the graph")
        keep.add(i)
        todo += [r["task"] for r in by[i]["reads"]]
    return order([by[i] for i in sorted(keep)])
