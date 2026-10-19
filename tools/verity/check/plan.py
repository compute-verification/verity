"""`check.py --plan [COMMIT]`: what `check` would reuse at COMMIT and what it would run, running no test, build or audit.

    uv run --locked --all-packages --extra torch-cpu python tools/verity/check/check.py --plan [COMMIT] [--json]
        [--base REF] [--packs-only] [--no-gather] [--from RUN]... [--verdicts-in PACK]...

COMMIT defaults to HEAD: this checkout as it is, uncommitted files included, as `check` would see it here.  Another commit is
checked out in a temporary worktree and planned there, in that tree's own environment (`DECLARED`, the declared command's `uv
run`), by that commit's own key code: its `suites.py`, its `check.py` (`env`, `STEP_KEYS`), its `lean_audit.py` and its
`circuit_check`.  Run it where `check` runs (a check node): the suites' and circuit-check's keys hold the interpreter, the base
image and the installed versions, so a machine unlike the one that earned a pass never finds it.

Each part of the check, with its key at COMMIT and the recorded passes under that key:

    suite:NAME            a suite, in pytest or (`AFTER_LEAN`) lean-suites: `suites.key` under that step's environment, a pass
                          `cached` that `holds`, or the suite's traced pass (`PerTest.whole`); a suite whose key reads what
                          lean-build builds (a declared artifact under the verifier's `.lake/`) is `unkeyed`
    circuit-check:TARGET  each target `circuit-check --all` checks, at the keys its task list's cache entry holds
                          (`circuit_check.cache.lookup`, as `cli.run` reads them); with that entry in no source the step is one
                          part, `circuit-check`, `unkeyed`: its targets' keys need the task list built (every Definition)
    lean-audit:PACKAGE    `lean_audit.key`, a pass whose report has the kernel replay (`check.lean_fast`'s rule)
    flock-rust-tests, lean-agreement
                          `STEP_KEYS`, under the step environment of each check slot here (the key reads VERITY_TEST_CACHE,
                          which a slot sets to its own test cache), so a pass counts in the slot that earned it
    preflight-lints       the repository's own suite (tests/), which holds `LINTS`
    lean                  `skipped`, in place of every Lean part, on a tree without the verifier's Lean package (check
                          skips its Lean group there): a skip passes nothing
    the rest              `KEYLESS`: no key, so each must run; lean-changed is no verdict, never required

The passes are looked up in this machine's caches (`local`: $CHECK_CACHE, $VERITY_TEST_CACHE and every check slot's test
cache; not with --packs-only), in the verdict pack of each recorded check `gather_verdicts` would take (none with
--no-gather), of each RUN given with --from, and in each PACK given (which no gate cites).  A pass found in a recorded check's
pack names that attempt and the pack's sha256, which `research merge --verdicts` cites.

A part is `passed` (a pass is found), `must-run` (none under its key), `unkeyed` (its key can't be computed without running a
step first: it must run), `skipped` or `no-verdict`.  lean-agreement is required when the job's `merge_requires` covers the change from
--base (default: COMMIT's first parent).  The plan is fully passed when every required part is passed.  `--json` prints the
plan as `KIND`.  Exit 0 fully passed, 1 not, 2 usage.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
KIND = "check-plan/v1"
#: how a check runs a tree (`tools/verity/check/tool.py`'s command, less `python check.py`)
DECLARED = ["run", "--locked", "--all-packages", "--extra", "torch-cpu", "python"]
#: the parts no key covers, and why: each must run
KEYLESS = {
    "preflight-lock": "`uv lock --check` keeps no pass",
    "preflight-fixtures": "the fixtures' fetch from the store keeps no pass",
    "flock-circuit-build": "`cargo check` of flock-circuit keeps no pass",
    "lean-build": "`lake build` of the verifier keeps no pass",
    "lean-unit-cut": "the partition check runs the verifier lean-build builds, and keeps no pass",
}
#: the parts of one step the text lists that didn't pass (`--json` lists every part)
SHOWN = 25

#: circuit-check's task list and each target's entry, looked up as `circuit_check.cli.run` does with `--cache`, in each source's
#: cache: argv[1] is [[source, dir], ...]; prints one line, `CIRCUIT-PLAN {json}`
CIRCUIT_PROGRAM = r'''
import json, sys
from pathlib import Path
from circuit_check import cache as CC, checks as CK, cli, targets as TG
CC.trace(cli.REPO)
TG.load_registries()
opts = CK.Options()
listed = CC.key({"target": ["task-list", "--all"], "as_call": None,
                 "options": {k: getattr(opts, k) for k in opts.__dataclass_fields__}})
dirs = json.loads(sys.argv[1])
out = {"repo": str(cli.REPO), "untraced": CC.untraced()[:3], "list": None, "targets": []}
for name, d in dirs:
    hit = CC.lookup(Path(d), listed, cli.REPO, moved=False)
    if hit is None:
        continue
    out["list"] = {"source": name, "entry": hit[1]}
    for (kind, label), k in zip(hit[0]["targets"], hit[0]["keys"]):
        found = []
        for n2, d2 in dirs:
            g = CC.lookup(Path(d2), k, cli.REPO, moved=False)
            if g is not None and not cli._errored(g[0]):
                found.append([n2, g[1]])
        out["targets"].append({"kind": kind, "target": label, "key": CC._dir(Path(), k).as_posix(), "found": found})
    break
print("CIRCUIT-PLAN " + json.dumps(out))
'''


@dataclass
class Source:
    """Where passes are looked up: a check cache root and the suites' cache roots, and the attempt whose pack they came from."""
    name: str
    check: Path
    tests: list[Path]
    attempt: str | None = None
    sha256: str | None = None

    def where(self, run: str | None = None) -> dict:
        return {"source": self.name, **({"attempt": self.attempt, "pack_sha256": self.sha256} if self.attempt else {}),
                **({"run": run} if run else {})}


@dataclass
class Part:
    part: str
    step: str
    key: str | None
    state: str
    found: list[dict] = field(default_factory=list)
    why: str | None = None
    required: bool = True

    def doc(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@contextmanager
def environ(e: dict[str, str]):
    """This process's environment is ``e`` inside: what a step's own process would see (`suites.environment` reads it)."""
    was = dict(os.environ)
    os.environ.clear()
    os.environ.update(e)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(was)


def step_env(C) -> dict[str, str]:
    """The environment `check` gives its steps inside a recorded run (`check.env()` with the store required and the vLLM
    reference regression on), less the run's custody dir, which no key reads."""
    e = C.env()
    e[C.STORE_REQUIRED_ENV] = "1"
    e[C.REGRESSION_ENV] = "1"
    e.pop(C.CUSTODY_ENV, None)
    return e


def lean_env(e: dict[str, str]) -> dict[str, str]:
    """lean-suites' environment: the step's, with elan's bin first on the PATH as `ELAN` puts it."""
    home = e.get("HOME") or str(Path.home())
    return {**e, "PATH": f"{home}/.elan/bin:{e.get('PATH', '')}"}


def slot_caches(C) -> list[Path]:
    """Each check slot's test cache on this machine (`slot.py` sets VERITY_TEST_CACHE to it): the slots its config names and
    those whose cache is here."""
    base = Path.home() / ".cache" / "verity"
    names: set[str] = set()
    try:
        names |= {s["name"] for s in C.SLOT.parse(C.SLOT.SLOTS.read_text())}
    except (OSError, ValueError, KeyError, AttributeError):
        pass
    names |= {p.name.removeprefix("tests-slot-") for p in base.glob("tests-slot-*") if p.is_dir()}
    return [base / f"tests-slot-{n}" for n in sorted(names)]


def local_source(C) -> Source:
    S = C.SUITES
    tests = [Path(os.environ.get("VERITY_TEST_CACHE") or S.CACHE), *slot_caches(C)]
    return Source("local", C.cache_dir(), list(dict.fromkeys(tests)))


def unpack(V, pack: Path, d: Path, attempt: str | None) -> Source:
    """A pack's entries (`verdicts.import_`, which refuses a pack that isn't one) under ``d``, as a source of their own."""
    roots = {"check": d / "check", "tests": d / "tests"}
    was = V.roots
    V.roots = lambda: roots
    try:
        V.import_(pack)
    finally:
        V.roots = was
    sha = hashlib.sha256(pack.read_bytes()).hexdigest()
    return Source(f"pack:{attempt}" if attempt else f"file:{pack.name}", roots["check"], [roots["tests"]], attempt, sha)


def suite_parts(C, sources: list[Source], lean: bool) -> list[Part]:
    """One part per suite, as `suites.py`'s own plan keys and finds it (its `main`), under pytest's or lean-suites' environment."""
    S = C.SUITES
    e = step_env(C)
    with environ(e):
        env = {"pytest": S.environment()}
    with environ(lean_env(e)):
        env["lean-suites"] = S.environment()
    pkgs, suites = S.members(), S.discover()
    tree = S.worktree()
    lock, provided = S.locked(), set(importlib.metadata.packages_distributions())
    inputs = {s: S.inputs_of(s, pkgs) for s in suites}
    cuts = {s: S.dropped(s, inputs[s]) for s in suites}
    fixtures = S.fetched_fixtures(tree, {p for ps in inputs.values() for p in ps})
    built = (C.FLOCK_LEAN / ".lake").resolve()
    was = S.CACHE
    out = []
    try:
        for s in suites:
            step = "lean-suites" if lean and s.name in C.AFTER_LEAN else "pytest"
            if needs := [a for a in s.artifacts if (C.ROOT / a).resolve().is_relative_to(built)]:
                out.append(Part(f"suite:{s.name}", step, None, "unkeyed", why=f"its key reads {needs[0]}, which lean-build builds"))
                continue
            ids = S.input_ids(tree, sorted(set(inputs[s]) | set(S.RUNNER)), S.markdown(s), cuts[s])
            ids.update({f"fixture:{p}": art for p, art in fixtures.items() if S.covered(p, inputs[s], cuts[s])})
            deps = S.resolved(S.roots(s, pkgs), lock, s.path)
            k = S.key(s, ids, env[step], [], deps)
            found, per = [], None
            for src in sources:
                for root in src.tests:
                    S.CACHE = root
                    hit = S.cached(k)
                    if hit and S.holds(hit, tree, lock, provided):
                        found.append(src.where(hit.get("run")))
                        break
                    per = per or S.PerTest(s, ids, env[step], [[]], deps, tree)
                    per.dir = S.per_test_dir(s)
                    if w := per.whole(lock, provided):
                        found.append({**src.where(w.get("run")), "traced": True})
                        break
            out.append(Part(f"suite:{s.name}", step, k, "passed" if found else "must-run", found))
    finally:
        S.CACHE = was
    return out


def lean_parts(C, sources: list[Source]) -> list[Part]:
    """One part per Lake package with a lock: its audit's key (`lean_audit.key`), and a kept pass with the kernel replay."""
    LA = C._lean_audit()
    files = C.tracked_files(C.ROOT)
    if files is None:
        return [Part("lean-audit", "lean-audit", None, "unkeyed", why="the tree's files can't be listed")]
    fast = C._load("lean_changed", C.LEAN_JOBS).FAST_KIND
    out = []
    for p in LA.packages(files):
        k = LA.key(p, files, LA.FLAGS)
        found = []
        for src in sources:
            try:
                rep = json.loads((src.check / "lean-audit" / f"{k}.json").read_text())["report"]
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if (isinstance(rep, dict) and not rep.get("failures") and rep.get("kind") != fast
                    and isinstance(rep.get("seconds"), dict) and "replay" in rep["seconds"]):
                found.append(src.where())
        out.append(Part(f"lean-audit:{p}", "lean-audit", k, "passed" if found else "must-run", found))
    return out


def keyed_step_parts(C, sources: list[Source], names: list[str]) -> list[Part]:
    """`STEP_KEYS`' steps: the key under the step environment as it is here and in each check slot (its VERITY_TEST_CACHE)."""
    e = step_env(C)
    variants = [e, *({**e, "VERITY_TEST_CACHE": str(p)} for p in slot_caches(C) if str(p) != e.get("VERITY_TEST_CACHE"))]
    out = []
    for name in names:
        keys = list(dict.fromkeys(k for v in variants if (k := C.STEP_KEYS[name](v))))
        if not keys:
            why = ("its key reads `rustc -vV` of the toolchain check_build.sh pins, which isn't installed here (the step installs "
                   "it)" if name == "flock-rust-tests" else "its key can't be computed here (the tree's files can't be listed)")
            out.append(Part(name, name, None, "unkeyed", why=why))
            continue
        found = [{**src.where(), "key": k} for k in keys for src in sources if (src.check / name / f"{k}.json").is_file()]
        out.append(Part(name, name, found[0]["key"] if found else keys[0], "passed" if found else "must-run", found,
                        why=None if len(keys) == 1 else f"keyed under {len(keys)} slots' environments"))
    return out


def circuit_parts(C, sources: list[Source]) -> list[Part]:
    """circuit-check's targets at its task list's keys, looked up in each source by the tree's own `circuit_check`
    (`CIRCUIT_PROGRAM`, in a process of its own under the step environment, since it traces what it reads)."""
    if not (C.ROOT / "tools" / "verity" / "circuit_check").is_dir():
        return [Part("circuit-check", "circuit-check", None, "unkeyed", why="the tree has no tools/verity/circuit_check")]
    dirs = [[s.name, str(s.check / "circuit-check")] for s in sources]
    r = subprocess.run([sys.executable, "-c", CIRCUIT_PROGRAM, json.dumps(dirs)], cwd=C.ROOT, env=step_env(C),
                       capture_output=True, text=True)
    line = next((x for x in reversed(r.stdout.splitlines()) if x.startswith("CIRCUIT-PLAN ")), None)
    if r.returncode or line is None:
        return [Part("circuit-check", "circuit-check", None, "unkeyed",
                     why=f"its keys couldn't be read (exit {r.returncode}): {(r.stderr or r.stdout).strip()[-300:]}")]
    got = json.loads(line.removeprefix("CIRCUIT-PLAN "))
    if Path(got["repo"]).resolve() != C.ROOT.resolve():
        return [Part("circuit-check", "circuit-check", None, "unkeyed",
                     why=f"{sys.executable} imports circuit_check from {got['repo']}, not this tree")]
    if got["list"] is None:
        return [Part("circuit-check", "circuit-check", None, "unkeyed",
                     why="its task list's entry is in no source, so its targets' keys need the task list built (every Definition)")]
    by = {s.name: s for s in sources}
    return [Part(f"circuit-check:{t['target']}", "circuit-check", t["key"], "passed" if t["found"] else "must-run",
                 [{**by[n].where(), "entry": entry} for n, entry in t["found"]]) for t in got["targets"]]


def required_agreement(commit: str, base: str | None, repo: Path = ROOT) -> bool:
    """Whether lean-agreement's `merge_requires` (in COMMIT's ci.toml) covers a file the change from ``base`` changes, as
    `research merge`'s `required_steps` reads it."""
    from research.merge import rule_touches

    r = subprocess.run(["git", "show", f"{commit}:tools/verity/check/ci.toml"], cwd=repo, capture_output=True, text=True)
    rule = next((j.get("merge_requires") for j in tomllib.loads(r.stdout).get("job", []) if j.get("name") == "lean-agreement"),
                None) if r.returncode == 0 else None
    if not rule or not base:
        return False
    d = subprocess.run(["git", "diff", "--no-renames", "--name-only", "-z", base, commit], cwd=repo, capture_output=True, text=True)
    return bool(rule_touches(list(rule), [p for p in d.stdout.split("\0") if p]))


def plan_tree(tree: Path, sources: list[Source], agreement: bool, local: bool = False, circuit=circuit_parts) -> list[Part]:
    """Every part of the check of ``tree`` (which this process's interpreter runs, as `check` would), by that tree's own key
    code, and the passes ``sources`` hold under each key (with ``local``, this machine's caches first, added to them)."""
    C = _load("planned_check", tree / "tools" / "verity" / "check" / "check.py")
    if local:
        sources.insert(0, local_source(C))
    jobs = [j.name for j in C.JOBS]
    lean = (C.FLOCK_LEAN / "lakefile.toml").exists()
    suites = suite_parts(C, sources, lean)
    own = next((s.name for s in C.SUITES.discover() if s.path == "."), None)
    root = next((p for p in suites if p.part == f"suite:{own}"), None)
    parts = []
    for name, _argv, _cwd in C.preflight():
        if name == "preflight-lints" and root is not None:
            parts.append(Part(name, name, root.key, root.state, root.found, f"the repository's own suite ({root.part}) holds it"))
        else:
            parts.append(Part(name, name, None, "unkeyed", why=KEYLESS.get(name, "no key covers it")))
    parts += [p for p in suites if p.step == "pytest"]
    for j in jobs:
        if j == "circuit-check":
            parts += circuit(C, sources)
        elif j == "flock-rust-tests":
            parts += keyed_step_parts(C, sources, [j])
        elif not lean and j.startswith("lean"):
            if not any(p.part == "lean" for p in parts):
                parts.append(Part("lean", "lean", None, "skipped", why=f"check skips its Lean steps: there is no "
                                  f"{C.FLOCK_LEAN.relative_to(C.ROOT)}/lakefile.toml"))
        elif j == "lean-audit":
            parts += lean_parts(C, sources)
        elif j == "lean-agreement":
            [p] = keyed_step_parts(C, sources, [j])
            p.required = agreement
            parts.append(p)
        elif j == "lean-suites":
            parts += [p for p in suites if p.step == "lean-suites"]
        elif j == "lean-changed":
            parts.append(Part(j, j, None, "no-verdict", why="it vouches for nothing: the audit decides", required=False))
        elif j in KEYLESS:
            parts.append(Part(j, j, None, "unkeyed", why=KEYLESS[j]))
        elif j != "pytest":
            parts.append(Part(j, j, None, "unkeyed", why="the plan doesn't know this job's key"))
    return parts


def document(parts: list[Part], commit: str, tree: str | None, base: str | None, sources: list[Source], notes: list[str]) -> dict:
    req = [p for p in parts if p.required]
    count = {s: sum(p.state == s for p in req) for s in ("passed", "must-run", "unkeyed", "skipped")}
    return {"kind": KIND, "commit": commit, "tree": tree, "base": base,
            "sources": [{"source": s.name, **({"attempt": s.attempt, "pack_sha256": s.sha256} if s.attempt else {})} for s in sources],
            "notes": notes, "summary": {"parts": len(parts), "required": len(req), **count},
            "fully_passed": all(p.state == "passed" for p in req), "parts": [p.doc() for p in parts]}


def text(doc: dict) -> list[str]:
    """The plan as lines: each part that must run, then per step how many of its parts passed and from where."""
    lines = [f"plan: {doc['commit'][:12]} (tree {str(doc.get('tree'))[:12]}): sources {', '.join(s['source'] for s in doc['sources']) or 'none'}"]
    lines += [f"plan: note: {n}" for n in doc["notes"]]
    steps: dict[str, list[dict]] = {}
    for p in doc["parts"]:
        steps.setdefault(p["step"], []).append(p)
    for step, ps in steps.items():
        ok = [p for p in ps if p["state"] == "passed"]
        srcs = sorted({f.get("attempt") or f["source"] for p in ok for f in p["found"][:1]})
        lines.append(f"plan: {step}: {len(ok)} of {len(ps)} parts passed" + (f" (first found in {', '.join(srcs)})" if srcs else ""))
        shown = [p for p in ps if p["state"] != "passed" or len(ps) == 1]
        for p in shown[:SHOWN]:
            where = (p["found"][0].get("attempt") or p["found"][0]["source"]) if p["found"] else ""
            lines.append(f"plan:   {p['part']:60} {p['state']:9} {str(p.get('key'))[:16]:16} {where}"
                         + ("" if p.get("required", True) else " (not required)") + (f"  {p['why']}" if p.get("why") else ""))
        if len(shown) > SHOWN:
            lines.append(f"plan:   ... and {len(shown) - SHOWN} more not passed (--json lists every part)")
    s = doc["summary"]
    lines.append(f"plan: {'FULLY PASSED' if doc['fully_passed'] else 'not fully passed'}: {s['passed']} of {s['required']} required "
                 f"parts passed, {s['must-run']} must run, {s['unkeyed']} unkeyed" + (f", {s['skipped']} skipped" if s.get("skipped") else ""))
    return lines


def _git(*args: str, repo: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def gather(C, d: Path, a, notes: list[str]) -> list[tuple[str | None, Path]]:
    """The packs to look passes up in: what `gather_verdicts` would take (unless --no-gather), each --from RUN's, each
    --verdicts-in PACK."""
    from research.store.local import LocalStore

    packs: list[tuple[str | None, Path]] = []
    store = None
    try:
        if not a.no_gather or a.from_:
            store = LocalStore()
        if not a.no_gather:
            packs += C.gathered_packs(d, store)
        for rid in a.from_:
            if (p := C.attempt_pack(store, rid, d)) is None:
                notes.append(f"--from {rid}: that attempt published no verdict pack")
            else:
                packs.append((rid, p))
    except Exception as e:  # noqa: BLE001  (no pack: say so, and plan on what is here)
        notes.append(f"no verdicts gathered from the store ({type(e).__name__}: {e})")
    packs += [(None, Path(p)) for p in a.verdicts_in]
    return list({(rid, str(p)): (rid, p) for rid, p in packs}.values())


def _clean_env(tree: Path) -> dict[str, str]:
    """This process's environment for a planning process in ``tree``, as a check run there would have it: without this
    checkout's virtualenv and `uv run`'s depth (the tree's own `uv run` sets both), and with this checkout's path written as
    ``tree``'s wherever it appears (`check._env_key` writes either as `<tree>`)."""
    e = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "UV_RUN_RECURSION_DEPTH")}
    path = e.get("PATH", "").split(os.pathsep)
    if os.environ.get("VIRTUAL_ENV") and str(Path(os.environ["VIRTUAL_ENV"]) / "bin") in path:
        path.remove(str(Path(os.environ["VIRTUAL_ENV"]) / "bin"))       # the one `uv run` put first
    e["PATH"] = os.pathsep.join(path)
    return {k: v.replace(str(ROOT), str(tree)) for k, v in e.items()}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="check.py --plan", description=__doc__.split("\n\n")[0])
    ap.add_argument("commit", nargs="?", default="HEAD")
    ap.add_argument("--json", action="store_true", help=f"print the plan as {KIND}")
    ap.add_argument("--base", help="the change lean-agreement's merge_requires is read on (default: COMMIT's first parent)")
    ap.add_argument("--packs-only", action="store_true", help="look passes up in packs only, not in this machine's caches")
    ap.add_argument("--no-gather", action="store_true", help="take no pack from the newest recorded checks")
    ap.add_argument("--from", dest="from_", action="append", default=[], metavar="RUN", help="also RUN's verdict pack")
    ap.add_argument("--verdicts-in", action="append", default=[], metavar="PACK", help="also this pack (cited by no gate)")
    ap.add_argument("--tree", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--pack", action="append", default=[], help=argparse.SUPPRESS)
    ap.add_argument("--agreement", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    return ap


def inner(a) -> dict:
    """`--tree T`: plan T in this process, which is T's own interpreter, from the packs `--pack RUN=PATH` names (RUN empty for
    a pack no attempt published) and, unless --packs-only, this machine's caches."""
    V = _load("plan_verdicts", HERE / "verdicts.py")
    tree = a.tree.resolve()
    notes: list[str] = []
    with tempfile.TemporaryDirectory(prefix="check-plan-packs-") as d:
        sources = []
        for i, spec in enumerate(a.pack):
            rid, _, path = spec.partition("=")
            try:
                sources.append(unpack(V, Path(path), Path(d) / str(i), rid or None))
            except (V.BadPack, OSError, EOFError) as e:
                notes.append(f"{rid or path}: its pack is refused ({type(e).__name__}: {e})")
        parts = plan_tree(tree, sources, a.agreement, local=not a.packs_only)
        commit = _git("rev-parse", "HEAD", repo=tree)
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no", repo=tree))
        if dirty:
            notes.append("the tree has uncommitted changes, which the plan keys as they are")
        return document(parts, commit, _git("rev-parse", "HEAD^{tree}", repo=tree) if not dirty else None, a.base, sources, notes)


def main(argv: list[str], host=None) -> int:
    """`check.py --plan ...`; ``host`` is the check.py module running it (its `gathered_packs`), else this directory's."""
    a = parser().parse_args(argv)
    if a.tree is not None:
        doc = inner(a)
        if a.out:
            a.out.write_text(json.dumps(doc) + "\n")
        else:
            print(json.dumps(doc, indent=1) if a.json else "\n".join(text(doc)))
        return 0 if doc["fully_passed"] else 1
    try:
        commit = _git("rev-parse", "--verify", f"{a.commit}^{{commit}}")
    except subprocess.CalledProcessError:
        print(f"check.py --plan: no commit {a.commit}", file=sys.stderr)
        return 2
    head = _git("rev-parse", "HEAD")
    base = a.base or (_git("rev-parse", f"{commit}^") if _git("rev-list", "--parents", "-n", "1", commit).count(" ") else None)
    host = host or _load("plan_host_check", HERE / "check.py")
    notes: list[str] = []
    with tempfile.TemporaryDirectory(prefix="check-plan-") as d:
        packs = gather(host, Path(d), a, notes)
        args = [*(["--packs-only"] if a.packs_only else []), *(["--agreement"] if required_agreement(commit, base) else []),
                *(f"--pack={rid or ''}={p}" for rid, p in packs), *(["--base", base] if base else [])]
        out = Path(d) / "plan.json"
        if commit == head:
            r = main(["--tree", str(ROOT), "--out", str(out), *args])
        else:
            tree = Path(d) / "tree"
            _git("worktree", "add", "-q", "--detach", str(tree), commit)
            try:
                uv = os.environ.get("UV") or shutil.which("uv") or "uv"
                r = subprocess.run([uv, "--quiet", *DECLARED, str(HERE / "plan.py"), "--tree", str(tree), "--out", str(out), *args],
                                   cwd=tree, env=_clean_env(tree)).returncode
            finally:
                subprocess.run(["git", "worktree", "remove", "--force", str(tree)], cwd=ROOT, capture_output=True)
        if not out.is_file():
            print(f"check.py --plan: planning {commit[:12]} failed (exit {r})", file=sys.stderr)
            return 2
        doc = json.loads(out.read_text())
    doc["notes"] = notes + doc["notes"]
    print(json.dumps(doc, indent=1) if a.json else "\n".join(text(doc)))
    return 0 if doc["fully_passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
