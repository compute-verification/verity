"""lean-env's runner: make the `lean` env a task names (`families/lean_env.py`) as lean-module's runner makes it, kernel-replay
its modules (`Replay.lean --env-out`), and write their declarations' axiom sets as the env's record, which every
lean-module task on the env reads. Under the `whole` scope the modules are every `.olean` of the fetched packages' builds
and of the toolchain that `lean` finds there (`lean_module.env_modules`); under `imports`, the task's `modules` and every
module they import, transitively.

Each module's declarations go to the kernel once, private ones and proof bodies included, against the modules it imports.
One of a module under the module system that the kernel stops on (its recursion) is checked again on the constants
`lean` checked it on, its imports' public parts, and a round names those (`on_public_imports`). Modules that can't be
imported together (each of the toolchain's executables declares `main`) are replayed in rounds:
`Replay.lean` defers a module that clashes with one before it, and the next round imports the earlier rounds' modules and
takes their sets from their record so far.

An env layered on another (the task reads that env's `axioms`, its base) starts from the base's record: each module the
base recorded must be one of this env's with the bytes the base replayed (`lean_module.env_digests`, else an error), and
its line goes into this record as the base has it; the first round replays the rest, importing the base's modules and
taking their sets from its record, as a later round does an earlier one's. The judging below is of the modules this task
replayed, the base's task having judged its own.

The record is JSON written a field a line, so that a module's runner reads the fields before its axioms alone
(`lean_module.env_header`), and `Replay.lean --env` parses one module's line when it first needs it:

    {
    "schema":"verity-lean-env-axioms/v1",
    "env":<the env's `families.lean_env.env_key`>,
    "toolchain":<its `lean-toolchain` line>,
    "package":<the task's package>,
    "scope":<the task's scope>,
    "base":<the task of the env it is layered on, null for none>,
    "files":{<module>:<its `lean_module.module_digest`>,...},
    "sets":[[<axiom>,...],...],
    "axioms":{
    "<module>":{"<constant>":<the index of its set in `sets`>,...},
    ...
    }
    }

A constant that uses no axiom is left out, and a theorem two modules declare with one statement takes the union of the
copies' sets under each. Every replayed module has a line, and a digest of the parts the replay read.

The task fails when the kernel rejects a declaration, and when a declaration of a fetched package's module uses an axiom
outside `lean_module.STANDARD` (an axiom it declares, `sorryAx` from a `sorry`, `Lean.ofReduceBool` from `native_decide`),
unless it is unsafe: compiled code, which the kernel never checks and no checked declaration may mention. Lake declares
each member of its open type families as an axiom (`family_def`, kept apart by their names), so a fetched package's code
that uses Lake's API (a linter's executable) may use those, the toolchain's `Lake` modules' own and no others (`judge`).
A declaration `UNPROVED` lists, one its package itself names as unproved, may use `sorryAx` and no other: a module of the
tree that reaches it takes `sorryAx` through the record and fails, and an entry that uses `sorryAx` no more fails the env
until it leaves the list.
The toolchain's modules are recorded and not judged, since Init declares those axioms and the toolchain's own definitions
use them (`Lean.reduceBool` takes `Lean.trustCompiler`): a module of the tree that reaches one takes it through the record
and fails.
An env that can't be made, one with no build of any fetched package that Mathlib's cache doesn't supply (made without
warm dependencies or the pinned bundle, which hold what the Lean check built of ArkLib and the rest), a module under the
module system whose private part is missing (its public part makes its theorems axioms), a replay that ends without a
verdict and one that leaves out a module of the set are errors, as is a base record that isn't a whole replay on this
toolchain or names a module this env lacks or holds other bytes of."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .. import ROOT
from ..families import lean_env as family
from ..families.lean_env import SCOPES
from ..graph import LOG, RUNNERS
from . import Timeout, call
from . import lean_module as lm

#: the files of the Lean tool a task runs, and the env's making it shares with lean-module: inputs of every lean-env task
TOOLS = ("tools/verity/lean/Replay.lean", f"{RUNNERS}/lean_module.py")
SCHEMA = lm.ENV_SCHEMA
#: {a fetched package's module: its declarations that may use `sorryAx`}, each one its package lists as unproved
#: (ArkLib's `scripts/axiom_baseline.json`)
UNPROVED = {"ArkLib.Data.Fin.Basic": ("Fin.sumCases",),
            "ArkLib.Data.MvPolynomial.Interpolation": ("MvPolynomial.basis",)}


def _dump(x: object) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write(path: Path, head: dict, files: dict[str, str], records: dict[str, dict[str, list[str]]]) -> None:
    """The record at ``path`` of ``records``, {module: {constant: its axioms}}, with ``head``'s fields and ``files``."""
    index: dict[tuple[str, ...], int] = {}
    mods = sorted(records)
    lines = [_dump({c: index.setdefault(tuple(s), len(index)) for c, s in records[m].items() if s}) for m in mods]
    with open(path, "w") as f:
        f.write("{\n")
        for k, x in {"schema": SCHEMA, **head, "files": files, "sets": [list(s) for s in index]}.items():
            f.write(f"{_dump(k)}:{_dump(x)},\n")
        f.write('"axioms":{\n')
        for i, (m, line) in enumerate(zip(mods, lines)):
            f.write(f"{_dump(m)}:{line}{',' if i + 1 < len(mods) else ''}\n")
        f.write("}\n}\n")


def judge(records: dict[str, dict[str, list[str]]], unsafe: dict[str, set[str]], toolchain: set[str],
          unproved: dict[str, tuple[str, ...]], base: frozenset[str] = frozenset()) -> tuple[
              dict[str, list[str]], dict[str, int], list[str]]:
    """The axioms outside `lean_module.STANDARD` that safe declarations of the modules but ``base``'s use, as ({axiom: the
    fetched packages' declarations that use it}, {axiom: how many of the ``toolchain``'s modules' declarations use it}),
    and the declarations ``unproved`` lists in a recorded module that use no `sorryAx`. A fetched package's declaration may
    also use an axiom of Lake's open type families, one that a module of the toolchain's `Lake` declares (its
    `family_def`s); one a fetched package declares, and Init's, are outside. One ``unproved`` lists may also use
    `sorryAx`."""
    lake = {c for m in toolchain if m.partition(".")[0] == "Lake" for c, s in records[m].items() if c in s}
    outside: dict[str, list[str]] = {}
    unjudged: dict[str, int] = {}
    for m in sorted(set(records) - base):
        for c, s in sorted(records[m].items()):
            for a in s if c not in unsafe.get(m, ()) else ():
                if a in lm.STANDARD:
                    continue
                if m in toolchain:
                    unjudged[a] = unjudged.get(a, 0) + 1
                elif a not in lake and not (a == "sorryAx" and c in unproved.get(m, ())):
                    outside.setdefault(a, []).append(c)
    stale = sorted(c for m, cs in unproved.items() if m in records for c in cs if "sorryAx" not in records[m].get(c, ()))
    return outside, unjudged, stale


def _tail(work: Path, since: int, n: int = 20) -> list[str]:
    with open(work / LOG, "rb") as f:
        f.seek(since)
        return [line[:500] for line in f.read().decode(errors="replace").splitlines()[1:] if line.strip()][-n:]


def run(task: dict, work: Path) -> dict:
    v = task["vars"]
    pkg = ROOT / v["package"]
    out = work / next(o["path"] for o in task["outputs"] if o["name"] == "axioms")
    if v.get("scope") not in SCOPES:
        return {"status": "error", "detail": {"error": f"scope {v.get('scope')!r} isn't one of {', '.join(SCOPES)}"}}
    try:
        base = family.read(task)
    except ValueError as e:
        return {"status": "error", "detail": {"error": str(e)}}
    if base and v["scope"] != "whole":
        return {"status": "error", "detail": {"error": f"only the whole scope layers an env, and this one is {v['scope']}"}}
    steps: dict[str, float] = {}
    t = time.monotonic()
    try:
        tc = lm.toolchain(pkg, work)
        deps = lm.packages(pkg, work)
        replay = lm.program("Replay", tc, work)
    except (lm.EnvError, Timeout) as e:
        return {"status": "error", "detail": {"error": f"the lean env: {type(e).__name__}: {e}"}}
    if deps is not None and (bare := [n for n in lm.fetched(pkg) if n not in lm.la.cache_covered(deps)]) and not any(
            (deps / n / ".lake" / "build").is_dir() for n in bare):
        return {"status": "error", "detail": {"error": (
            f"the env holds no build of {', '.join(bare)}, which Mathlib's cache doesn't supply: this worker made it "
            "without warm dependencies or the pinned bundle, and its record would leave their modules out. Run the task "
            f"where the Lean check of {v['package']} has passed (`lean_audit.WarmDeps`) or where `setup.sh` can fetch "
            f"the bundle, after removing {deps}")}}
    libs = lm.env_libs(pkg, tc, deps)
    found = lm.env_modules(libs)
    steps["env"] = round(time.monotonic() - t, 3)
    head = {"env": family.env_key(pkg), "toolchain": (pkg / "lean-toolchain").read_text().strip(),
            "package": v["package"], "scope": v["scope"], "base": base[0] if base else None}
    detail: dict = {"libs": [str(d) for d in libs], "seconds": steps}
    records: dict[str, dict[str, list[str]]] = {}
    given: dict[str, str] = {}
    if base:
        t = time.monotonic()
        try:
            body = json.loads((work / base[1]).read_text())
            given, sets = body["files"], body["sets"]
            records = {m: {c: sets[i] for c, i in cs.items()} for m, cs in body["axioms"].items()}
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
            return {"status": "error", "detail": detail | {"error": f"{base[0]}'s record: {type(e).__name__}: {e}"}}
        if (body.get("schema"), body.get("toolchain"), body.get("scope")) != (SCHEMA, head["toolchain"], "whole") or (
                set(records) != set(given)):
            return {"status": "error", "detail": detail | {"error": (
                f"{base[0]}'s record isn't a whole replay on {head['toolchain']} ({SCHEMA}) with a line and a digest "
                "for each of its modules")}}
        if absent := sorted(set(given) - set(found)):
            return {"status": "error", "detail": detail | {"absent": absent[:20], "error": (
                f"{base[0]} recorded {len(absent)} modules this env doesn't hold")}}
        mine = lm.env_digests(head["env"], libs, sorted(given))
        if differ := [m for m in sorted(given) if mine[m] != given[m]]:
            return {"status": "error", "detail": detail | {"differ": differ[:20], "error": (
                f"this worker's env holds other bytes than {base[0]} replayed for {len(differ)} of its modules")}}
        steps["base"] = round(time.monotonic() - t, 3)
        detail |= {"base": base[0], "base_modules": len(given)}
    env = {**os.environ, "LEAN_PATH": os.pathsep.join(map(str, libs[:-1])), "LEAN_SYSROOT": str(tc),
           "LEAN_NUM_THREADS": str(task.get("resources", {}).get("cores", 1))}
    roots = sorted(set(found) - set(given)) if v["scope"] == "whole" else sorted(json.loads(v.get("modules") or "[]"))
    todo, flag = roots, "--closure" if v["scope"] == "whole" else "--closure-of"
    unsafe: dict[str, set[str]] = {}
    rounds: list[dict] = []
    later: set[str] = set()
    detail["rounds"] = rounds
    while todo:
        d = work / "replay" / str(len(rounds) + 1)
        d.mkdir(parents=True, exist_ok=True)
        (d / "modules").write_text("".join(f"{m}\n" for m in todo))
        argv = [str(replay), str(d / "out.json")]
        if records:
            write(d / "env.json", head, {}, records)
            argv += ["--env", str(d / "env.json")]
        argv += ["--env-out", str(d / "sets.jsonl"), flag, f"@{d / 'modules'}"]
        since = (work / LOG).stat().st_size if (work / LOG).exists() else 0
        t = time.monotonic()
        rc = call(argv, work, work, env)
        got = json.loads((d / "out.json").read_text()) if (d / "out.json").is_file() else {}
        rounds.append({"modules": got.get("modules"), "replayed": got.get("replayed"), "skipped": got.get("skipped"),
                       "imported": got.get("imported"), "on_public_imports": got.get("on_public_imports"),
                       "seconds": round(time.monotonic() - t, 3), "replay_seconds": got.get("seconds")})
        if rc == 1 and got.get("rejected"):
            return {"status": "fail", "detail": detail | {"error": "the kernel rejected declarations",
                                                          "rejected": got["rejected"]}}
        if rc or not (d / "sets.jsonl").is_file():
            return {"status": "error", "detail": detail | {"error": f"the replay exited {rc} without a verdict",
                                                           "log": _tail(work, since)}}
        deferred: list[str] | None = None
        with open(d / "sets.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if "module" in r:
                    records[r["module"]], unsafe[r["module"]] = r["axioms"], set(r["unsafe"])
                else:
                    deferred = r["deferred"]
                    for m, cs in r["joined"].items():
                        for c, s in cs.items():
                            records.setdefault(m, {})[c] = sorted({*records[m].get(c, ()), *s})
        if deferred is None or (deferred and not got.get("modules")):
            return {"status": "error", "detail": detail | {"error": "the replay deferred every module it was given"
                                                           if deferred else "the replay wrote no deferred line"}}
        later |= set(deferred)
        todo, flag = deferred, "--closure"
    steps["replay"] = round(sum(r["seconds"] for r in rounds), 3)
    if missing := sorted(set(found if v["scope"] == "whole" else roots) - set(records)):
        return {"status": "error", "detail": detail | {"missing": missing[:20], "error": (
            f"the replay left out {len(missing)} modules of the set")}}
    t = time.monotonic()
    files = given | lm.env_digests(head["env"], libs, sorted(set(records) - set(given)))
    if nowhere := sorted(m for m, x in files.items() if x is None):
        return {"status": "error", "detail": detail | {"nowhere": nowhere[:20], "error": (
            f"{len(nowhere)} replayed modules have no `.olean` in the env's directories")}}
    steps["digests"] = round(time.monotonic() - t, 3)
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, head, files, records)
    outside, unjudged, stale = judge(records, unsafe,
                                     {m for m in records if m in found and found[m].is_relative_to(libs[-1])}, UNPROVED,
                                     frozenset(given))
    detail |= {"modules": len(records), "constants": sum(map(len, records.values())),
               "replayed": sum(r["replayed"] or 0 for r in rounds), "deferred": sorted(later),
               "record_bytes": out.stat().st_size, "toolchain_outside_standard": unjudged,
               "unproved": sorted(c for m, cs in UNPROVED.items() if m in records for c in cs if c not in stale)}
    if outside:
        return {"status": "fail", "detail": detail | {
            "error": f"declarations of fetched packages use axioms other than {', '.join(lm.STANDARD)}",
            "axioms": {a: names[:20] for a, names in sorted(outside.items())},
            "declarations": {a: len(names) for a, names in sorted(outside.items())}}}
    if stale:
        return {"status": "fail", "detail": detail | {"stale": stale, "error": (
            f"`lean_env.UNPROVED` lists {', '.join(stale)}, which use no `sorryAx` now: remove them from it")}}
    return {"status": "pass", "detail": detail}
