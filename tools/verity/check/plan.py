"""`check.py --plan [COMMIT]`: what `check` would reuse at COMMIT and what it would run, running no test, build or audit.

    uv run --locked --all-packages --extra torch-cpu python tools/verity/check/check.py --plan [COMMIT] [--json]
        [--base REF] [--packs-only] [--no-gather] [--from RUN]... [--verdicts-in PACK]... [--on MACHINE]

COMMIT defaults to HEAD: this checkout as it is, uncommitted files included, as `check` would see it here.  Another commit is
checked out in a temporary worktree and planned there, in that tree's own environment (`DECLARED`, the declared command's `uv
run`), by that commit's own key code: its `suites.py`, its `check.py` (`env`, `STEP_KEYS`), its `lean_audit.py` and its
`circuit_check`.  Run it where `check` runs (a check node): the suites' and circuit-check's keys hold the interpreter, the base
image and the installed versions, so a machine unlike the one that earned a pass never finds it.  `--on MACHINE` does that from
anywhere the store is (`on_machine`): the packs are gathered here, and COMMIT (always a worktree of it, so committed files only)
is planned on MACHINE by a short `research run --wait` that ships it with this plan.py, verdicts.py and those packs.  Its keys
match passes earned on a machine of MACHINE's environment only (the vy-nebius-cpu nodes have no nvidia-smi, the GPU nodes do).

Each part of the check, with its key at COMMIT and the recorded passes under that key:

    suite:NAME            a suite, in pytest or (`AFTER_LEAN`) lean-suites: `suites.key` under that step's environment, a pass
                          `cached` that `holds`, or the suite's traced pass (`PerTest.whole`); what lean-build builds keys a
                          suite by the verifier's sources (`suites.built_from`), and a suite whose key reads it otherwise (a
                          tree whose suites.py keys the bytes) is `unkeyed`
    circuit-check:TARGET  each target `circuit-check --all` checks, at the keys its task list's cache entry holds
                          (`circuit_check.cache.lookup`, as `cli.run` reads them); with that entry in no source the step is one
                          part, `circuit-check`, `unkeyed`: its targets' keys need the task list built (every Definition)
    lean-audit:PACKAGE    `lean_audit.key`, a pass whose report has the kernel replay (`check.lean_fast`'s rule)
    flock-rust-tests, lean-agreement, preflight-lints
                          `STEP_KEYS`, under the step environment of each check slot here (the key reads VERITY_TEST_CACHE,
                          which a slot sets to its own test cache), so a pass counts in the slot that earned it; preflight-lints
                          passes too where the repository's own suite (tests/), which holds `LINTS`, passed
    lean                  `skipped`, in place of every Lean part, on a tree without the verifier's Lean package (check
                          skips its Lean group there): a skip passes nothing
    lean-changed          no verdict, never required: it vouches for nothing, the audit decides
    ratchet-gate          no verdict, never required: it judges the merges above the check's base, not COMMIT's tree, and
                          the train's check runs it

The steps that build or fetch what other steps read (`IMPLIED`) keep no verdict of their own.  Each is
`implied` when every part that reads what it makes has passed (`implied_by`), and `must-run` when one hasn't.  Why those passes
answer for each:

    preflight-lock        `uv lock --check`: uv.lock is what the workspace's pyproject files resolve to.  A plan of a commit
                          runs in that commit's tree under `uv run --locked` (`DECLARED`, and `research merge`'s `PLAN`), which
                          refuses a lock that isn't current, so a plan exists only where it would pass; and every suite's key
                          holds the lock's resolution of what the suite imports.  Implied by every suite.
    preflight-fixtures    fetches each fixture `fixtures/artifacts.json` lists that git doesn't hold.  A suite's key holds the
                          art id of each such fixture it reads (`suites.fetched_fixtures`), and a suite whose fixture can't be
                          fetched fails, never passes; a fixture no suite reads decides no verdict.  Implied by every suite.
    flock-circuit-build   `check_build.sh`'s `cargo check` of flock-circuit in each CPU feature set.  flock-rust-tests' key
                          (`rust_tests_key`) holds everything the build reads (the script, its pinned Flock commit and
                          toolchain, the patches, the live crate), and its tests compile those crates in release.  They don't
                          compile the `flock-circuit` binary target or the `sha512,glue` set, so an error only there is what
                          this leaves to the post-landing run.  Implied by flock-rust-tests.
    lean-build            `lake build` of the verifier.  Its audit (`lean-audit:` the verifier's package) builds every file of
                          the package from the same sources and toolchain, its key holding them, and replays each declaration;
                          the suites that run its binaries are keyed by those sources (`suites.built_from`) and ran on them,
                          as lean-agreement did when the change requires it.  Implied by those.
    lean-unit-cut         `unit_cut_agree.py` on 500 random cuts against the binary lean-build built.  verity-flock's
                          `test_unit_cut_agrees_with_the_reference` runs it on the first 300 of the same seed, against the
                          binary its key now names, so this leaves the last 200 cuts to the post-landing run.  Implied by
                          the suite that runs it.

The passes are looked up in this machine's caches (`local`: $CHECK_CACHE, $VERITY_TEST_CACHE and every check slot's test
cache; not with --packs-only), in the verdict pack of each recorded check `gather_verdicts` would take (none with
--no-gather), in both packs of each recorded check of COMMIT and of its post-landing runs (`--deferred`, which drive runs on
the `main` it landed: none with --no-gather) and of each RUN given with --from (the
passes it stored, `verdicts.tar.gz`, and those it reused, `verdicts-reused.tar.xz`: `check.Reused`), and in each PACK given
(which no gate cites).  A pass found in a recorded check's pack names that attempt, the pack's sha256, which `research merge
--verdicts` cites, the run that earned it, and the files it is (`paths`).

A part is `passed` (a pass is found), `implied` (above), `must-run` (none under its key), `unkeyed` (its key can't be computed
without running a step first: it must run), `skipped` or `no-verdict`.  lean-agreement is required when the job's
`merge_requires` covers the change from --base (default: COMMIT's first parent).  The plan is fully passed when every required
part is passed or implied.  `--json` prints the
plan as `KIND`.  Exit 0 fully passed, 1 not, 2 usage.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shlex
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
#: the steps that build or fetch what other parts read, and keep no verdict of their own (the module docstring says why each
#: is satisfied by the passes of the parts that read what it makes): {step: (what it makes, the parts that read it)}
IMPLIED = {
    "preflight-lock": ("the current uv.lock every suite's key resolves", lambda C, parts: _named(parts, "suite:")),
    "preflight-fixtures": ("the fetched fixtures, each in the key of every suite that reads it",
                           lambda C, parts: _named(parts, "suite:")),
    "flock-circuit-build": ("the patched Flock crates, which flock-rust-tests compiles from what its key holds",
                            lambda C, parts: _named(parts, "flock-rust-tests")),
    "lean-build": ("the verifier, built from the sources its audit and the suites that run it are keyed by",
                   lambda C, parts: [p.part for p in parts if p.reads == "lean-build" or (p.part == "lean-agreement" and p.required)
                                     or p.part == f"lean-audit:{C.FLOCK_LEAN.relative_to(C.ROOT).as_posix()}"]),
    "lean-unit-cut": ("the partition check of the first 300 of its 500 cuts, which verity-flock runs",
                      lambda C, parts: _named(parts, "suite:verity-flock")),
}


def _named(parts: list["Part"], prefix: str) -> list[str]:
    return [p.part for p in parts if p.part.startswith(prefix)]
#: a recorded check's packs (`check.PACK_FILES`): the passes its steps stored, and those it found here before they ran, which it
#: reused (`check.Reused`)
PACK, REUSED = "verdicts.tar.gz", "verdicts-reused.tar.xz"
#: the states of a part the plan counts as passed
OK = ("passed", "implied")
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
    out["list"] = {"source": name, "entry": hit[1], "key": CC._dir(Path(), listed).as_posix()}
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

    def where(self, run: str | None = None, paths: list[Path] = ()) -> dict:
        """A pass found here: the source, the attempt and pack it came from, the run that earned it, and the files it is (as pack
        paths, `check/...` and `tests/...`, which `verdicts.export_paths` packs)."""
        return {"source": self.name, **({"attempt": self.attempt, "pack_sha256": self.sha256} if self.attempt else {}),
                **({"run": run} if run else {}), **({"paths": [self.pack_path(p) for p in paths]} if paths else {})}

    def pack_path(self, p: Path) -> str:
        if p.is_relative_to(self.check):
            return f"check/{p.relative_to(self.check).as_posix()}"
        root = next(r for r in self.tests if p.is_relative_to(r))
        return f"tests/{p.relative_to(root).as_posix()}"


@dataclass
class Part:
    part: str
    step: str
    key: str | None
    state: str
    found: list[dict] = field(default_factory=list)
    why: str | None = None
    required: bool = True
    #: the build step whose output its key names (a suite keyed by the verifier's sources)
    reads: str | None = None
    #: the parts whose passes satisfy it (`IMPLIED`)
    implied_by: list[str] | None = None

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


def local_source(C, own: bool = False) -> Source:
    """This machine's caches: $CHECK_CACHE, $VERITY_TEST_CACHE and (unless ``own``, a check's own caches alone) every check
    slot's test cache."""
    S = C.SUITES
    tests = [Path(os.environ.get("VERITY_TEST_CACHE") or S.CACHE), *([] if own else slot_caches(C))]
    return Source("own" if own else "local", C.cache_dir(), list(dict.fromkeys(tests)))


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
            if needs := [a for a in s.artifacts if (C.ROOT / a).resolve().is_relative_to(built)
                         and not (hasattr(S, "built_from") and S.built_from(a))]:
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
                        found.append(src.where(hit.get("run"), [root / f"{k}.json"]))
                        break
                    per = per or S.PerTest(s, ids, env[step], [[]], deps, tree)
                    per.dir = S.per_test_dir(s)
                    if w := per.whole(lock, provided):
                        files = [per.dir / f"{S.WHOLE}--{w['key']}.json", per.dir / S.TRACED / f"{w['traced']}.json"]
                        found.append({**src.where(w.get("run"), files), "traced": True})
                        break
            reads = "lean-build" if any((C.ROOT / a).resolve().is_relative_to(built) for a in s.artifacts) else None
            out.append(Part(f"suite:{s.name}", step, k, "passed" if found else "must-run", found, reads=reads))
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
            entry = src.check / "lean-audit" / f"{k}.json"
            try:
                rep = json.loads(entry.read_text())["report"]
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if (isinstance(rep, dict) and not rep.get("failures") and rep.get("kind") != fast
                    and isinstance(rep.get("seconds"), dict) and "replay" in rep["seconds"]):
                found.append(src.where(paths=[entry]))
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
        found = [{**src.where(paths=[src.check / name / f"{k}.json"]), "key": k} for k in keys for src in sources
                 if (src.check / name / f"{k}.json").is_file()]
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
    lister = by[got["list"]["source"]]
    task_list = lister.pack_path(lister.check / "circuit-check" / got["list"]["key"] / f"{got['list']['entry']}.json")

    def found(n: str, key: str, entry: str) -> dict:
        # a target's key is known only from the task list's entry, so a pack that answers for the target holds both
        f = by[n].where(paths=[by[n].check / "circuit-check" / key / f"{entry}.json"])
        return {**f, "paths": [*f["paths"], task_list], "entry": entry}

    return [Part(f"circuit-check:{t['target']}", "circuit-check", t["key"], "passed" if t["found"] else "must-run",
                 [found(n, t["key"], entry) for n, entry in t["found"]]) for t in got["targets"]]


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


def plan_tree(tree: Path, sources: list[Source], agreement: bool, local: bool = False, circuit=circuit_parts,
              own: bool = False) -> list[Part]:
    """Every part of the check of ``tree`` (which this process's interpreter runs, as `check` would), by that tree's own key
    code, and the passes ``sources`` hold under each key (with ``local``, this machine's caches first, added to them: ``own``,
    only the caches this process's check uses)."""
    C = _load("planned_check", tree / "tools" / "verity" / "check" / "check.py")
    if local:
        sources.insert(0, local_source(C, own))
    jobs = [j.name for j in C.JOBS]
    lean = (C.FLOCK_LEAN / "lakefile.toml").exists()
    suites = suite_parts(C, sources, lean)
    own = next((s.name for s in C.SUITES.discover() if s.path == "."), None)
    root = next((p for p in suites if p.part == f"suite:{own}"), None)
    parts = []
    for name, _argv, _cwd in C.preflight():
        if name in C.STEP_KEYS:
            [p] = keyed_step_parts(C, sources, [name])
            if p.state != "passed" and name == "preflight-lints" and root is not None and root.state == "passed":
                p = Part(name, name, p.key, "passed", root.found, f"the repository's own suite ({root.part}), which holds it, passed")
            parts.append(p)
        elif name == "preflight-lints" and root is not None:
            parts.append(Part(name, name, root.key, root.state, root.found, f"the repository's own suite ({root.part}) holds it"))
        elif name in IMPLIED:
            parts.append(Part(name, name, None, "implied"))
        else:
            parts.append(Part(name, name, None, "unkeyed", why="no key covers it"))
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
        elif j == "ratchet-gate":
            parts.append(Part(j, j, None, "no-verdict", why="it judges the merges above the check's base, not the tree: "
                              "the train's check runs it", required=False))
        elif j in IMPLIED:
            parts.append(Part(j, j, None, "implied"))
        elif j != "pytest":
            parts.append(Part(j, j, None, "unkeyed", why="the plan doesn't know this job's key"))
    by = {p.part: p for p in parts}
    for p in parts:
        if p.state == "implied":
            makes, readers = IMPLIED[p.part]
            p.implied_by = readers(C, parts)
            short = [n for n in p.implied_by if by[n].state != "passed"]
            if not p.implied_by or short:
                p.state = "must-run"
                p.why = f"it makes {makes}: " + (f"{', '.join(short[:5])} not passed" if short else "no part here reads it")
            else:
                p.why = f"it makes {makes}: all {len(p.implied_by)} passed"
    return parts


def document(parts: list[Part], commit: str, tree: str | None, base: str | None, sources: list[Source], notes: list[str]) -> dict:
    req = [p for p in parts if p.required]
    count = {s: sum(p.state == s for p in req) for s in ("passed", "implied", "must-run", "unkeyed", "skipped")}
    return {"kind": KIND, "commit": commit, "tree": tree, "base": base,
            "sources": [{"source": s.name, **({"attempt": s.attempt, "pack_sha256": s.sha256} if s.attempt else {})} for s in sources],
            "notes": notes, "summary": {"parts": len(parts), "required": len(req), **count},
            "fully_passed": all(p.state in OK for p in req), "parts": [p.doc() for p in parts]}


def text(doc: dict) -> list[str]:
    """The plan as lines: each part that must run, then per step how many of its parts passed and from where."""
    lines = [f"plan: {doc['commit'][:12]} (tree {str(doc.get('tree'))[:12]}): sources {', '.join(s['source'] for s in doc['sources']) or 'none'}"]
    lines += [f"plan: note: {n}" for n in doc["notes"]]
    steps: dict[str, list[dict]] = {}
    for p in doc["parts"]:
        steps.setdefault(p["step"], []).append(p)
    for step, ps in steps.items():
        ok = [p for p in ps if p["state"] in OK]
        srcs = sorted({f.get("attempt") or f["source"] for p in ok for f in p["found"][:1]})
        lines.append(f"plan: {step}: {len(ok)} of {len(ps)} parts passed" + (f" (first found in {', '.join(srcs)})" if srcs else ""))
        shown = [p for p in ps if p["state"] not in OK or len(ps) == 1]
        for p in shown[:SHOWN]:
            where = (p["found"][0].get("attempt") or p["found"][0]["source"]) if p["found"] else ""
            lines.append(f"plan:   {p['part']:60} {p['state']:9} {str(p.get('key'))[:16]:16} {where}"
                         + ("" if p.get("required", True) else " (not required)") + (f"  {p['why']}" if p.get("why") else ""))
        if len(shown) > SHOWN:
            lines.append(f"plan:   ... and {len(shown) - SHOWN} more not passed (--json lists every part)")
    s = doc["summary"]
    lines.append(f"plan: {'FULLY PASSED' if doc['fully_passed'] else 'not fully passed'}: {s['passed'] + s.get('implied', 0)} of "
                 f"{s['required']} required parts passed" + (f" ({s['implied']} implied by those that read what they build)"
                                                               if s.get("implied") else "")
                 + f", {s['must-run']} must run, {s['unkeyed']} unkeyed" + (f", {s['skipped']} skipped" if s.get("skipped") else ""))
    return lines


def _git(*args: str, repo: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


#: the states of a recorded check whose packs hold passes (`check.gathered_packs`'): a pack holds only passes
ENDED = ("done", "failed", "cancelled")


def own_checks(store, commit: str) -> list[str]:
    """The recorded checks of ``commit`` that ended, and its post-landing runs (`check.py --deferred`, recorded without `--tool
    check`, so found by their argv), newest first."""
    rows = store.index.sql(f"SELECT id, tool FROM attempts WHERE source_commit = ? AND state IN ({', '.join('?' * len(ENDED))}) "
                           "ORDER BY start_utc DESC", (commit, *ENDED))
    return [rid for rid, tool in rows if tool == "check" or _post_landing(store, rid)]


def _post_landing(store, rid: str) -> bool:
    try:
        argv = (store.get_attempt(rid) or {}).get("argv") or []
    except Exception:  # noqa: BLE001  (an attempt that can't be read is no source)
        return False
    return "--deferred" in argv and any(str(x).endswith("tools/verity/check/check.py") for x in argv)


def gather(C, d: Path, a, notes: list[str], commit: str | None = None) -> list[tuple[str | None, Path]]:
    """The packs to look passes up in: what `gather_verdicts` would take (unless --no-gather); both packs of each recorded check
    of ``commit`` (unless --no-gather) and of each --from RUN, the passes it stored and those it reused (`REUSED`, a pack too
    big to take from every check); each --verdicts-in PACK."""
    from research.store.local import LocalStore

    packs: list[tuple[str | None, Path]] = []
    store = None
    try:
        if not a.no_gather or a.from_:
            store = LocalStore()
        if not a.no_gather:
            packs += C.gathered_packs(d, store)
        own = own_checks(store, commit) if commit and not a.no_gather else []
        taken = {rid for rid, _ in packs}
        for rid in dict.fromkeys([*a.from_, *own]):
            got = [(rid, p) for name in (PACK, REUSED) if not (name == PACK and rid in taken)
                   and (p := C.attempt_pack(store, rid, d, name)) is not None]
            if not got and rid in a.from_ and rid not in taken:
                notes.append(f"--from {rid}: that attempt published no verdict pack")
            packs += got
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


#: a plan on a node (`--on`): the project its run belongs to, its bound in seconds (the tree's environment is synced first where
#: that machine has none for it yet), and where it writes the plan
REMOTE_PROJECT, REMOTE_TIMEOUT_S, REMOTE_OUT = "verity", 1800, "out/plan.json"
#: the pool it runs in (`research run --pool`): a research slot at nice 19 beside the merge checks where the machine takes runs
#: only through slots (vy-nebius-cpu-1), and the command as it is (slot.py with no slots file) where it takes them directly
REMOTE_POOL = "research"
#: the run id `research run` prints when it launches one (`research.queue.LAUNCHED_RE`)
LAUNCHED = re.compile(r"\brun (r20\d{6}-\d{6}-[0-9a-f]{4}) launched\b")


def on_machine(machine: str, commit: str, d: Path, packs: list[tuple[str | None, Path]], flags: list[str], out: Path,
               notes: list[str], run=subprocess.run) -> int:
    """`--on MACHINE`: plan ``commit`` on MACHINE, a check node, where the keys are that machine's (its interpreter, base image,
    the tools on its PATH and its check slots' caches), as a `research run --wait` of a worktree of ``commit`` (`--source`,
    `--cwd source`, in `REMOTE_POOL`): this checkout's plan.py and verdicts.py, sent with the run, plan that tree (`--tree .`) in its own
    environment (`DECLARED`), on the packs gathered here, sent too, since a run on a node can't fetch the store's blobs.  Its
    shell is no login shell: a profile's PATH would key flock-rust-tests, lean-agreement and preflight-lints unlike a check's.  The
    plan comes back from the run's `REMOTE_OUT` to ``out``.  Returns the run's exit, 2 when it launched none."""
    if machine == "auto" or "," in machine:
        print(f"check.py --plan --on {machine}: name one machine; its keys are that machine's", file=sys.stderr)
        return 2
    tree, send = d / "on-tree", d / "send"
    send.mkdir()
    names = []
    for i, (rid, p) in enumerate(packs):
        shutil.copyfile(p, send / (n := f"pack-{i}{''.join(p.suffixes[-2:])}"))
        names.append((rid, n))
    cmd = " ".join(['I="$RESEARCH_RUN_DIR/inputs";', 'mkdir -p "$RESEARCH_RUN_DIR/out";', "exec uv --quiet", *DECLARED,
                    '"$I/plan.py"', "--tree", ".", "--out", f'"$RESEARCH_RUN_DIR/{REMOTE_OUT}"', *map(shlex.quote, flags),
                    *(shlex.quote(f"--pack={rid or ''}=") + f'"$I/{n}"' for rid, n in names)])
    sends = [HERE / "plan.py", HERE / "verdicts.py", *(send / n for _, n in names)]
    _git("worktree", "add", "-q", "--detach", str(tree), commit, repo=ROOT)
    try:
        r = run([sys.executable, "-m", "research", "run", "--on", machine, "--project", REMOTE_PROJECT, "--source", str(tree),
                 "--cwd", "source", "--pool", REMOTE_POOL, "--as", os.environ.get("RESEARCH_LANE") or "ci",
                 "--timeout", str(REMOTE_TIMEOUT_S), "--wait", "--declared-output", REMOTE_OUT,
                 *(x for f in sends for x in ("--send", str(f))), "--", "bash", "-c", cmd],
                cwd=ROOT, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(tree)], cwd=ROOT, capture_output=True)
    said = f"{r.stdout}\n{r.stderr}".strip()
    if not (ids := LAUNCHED.findall(said)):
        print(f"check.py --plan --on {machine}: `research run` launched no run (exit {r.returncode}): {said[-400:]}", file=sys.stderr)
        return 2
    rid = ids[-1]
    notes.append(f"planned on {machine} in run {rid}, on the {len(names)} packs gathered where it was asked")
    got = run([sys.executable, "-m", "research", "status", rid, "--json", "--also", REMOTE_OUT], cwd=ROOT, capture_output=True,
              text=True, stdin=subprocess.DEVNULL)
    try:
        shutil.copyfile(Path(json.loads(got.stdout)["runs"][0]["dir"]) / REMOTE_OUT, out)
    except (ValueError, KeyError, IndexError, TypeError, OSError) as e:
        print(f"check.py --plan --on {machine}: run {rid} (exit {r.returncode}) left no {REMOTE_OUT} here "
              f"({type(e).__name__}: {e}): {said[-300:]}", file=sys.stderr)
    return r.returncode


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="check.py --plan", description=__doc__.split("\n\n")[0])
    ap.add_argument("commit", nargs="?", default="HEAD")
    ap.add_argument("--json", action="store_true", help=f"print the plan as {KIND}")
    ap.add_argument("--base", help="the change lean-agreement's merge_requires is read on (default: COMMIT's first parent)")
    ap.add_argument("--packs-only", action="store_true", help="look passes up in packs only, not in this machine's caches")
    ap.add_argument("--no-gather", action="store_true", help="take no pack from the newest recorded checks")
    ap.add_argument("--from", dest="from_", action="append", default=[], metavar="RUN", help="also RUN's verdict pack")
    ap.add_argument("--verdicts-in", action="append", default=[], metavar="PACK", help="also this pack (cited by no gate)")
    ap.add_argument("--on", metavar="MACHINE", help="plan on MACHINE, a check node, as a short `research run` (`on_machine`)")
    ap.add_argument("--tree", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--pack", action="append", default=[], help=argparse.SUPPRESS)
    ap.add_argument("--agreement", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--own", action="store_true", help=argparse.SUPPRESS)
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
        parts = plan_tree(tree, sources, a.agreement, local=not a.packs_only, own=a.own)
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
        packs = gather(host, Path(d), a, notes, commit)
        flags = [*(["--packs-only"] if a.packs_only else []), *(["--agreement"] if required_agreement(commit, base) else []),
                 *(["--base", base] if base else [])]
        args = [*flags, *(f"--pack={rid or ''}={p}" for rid, p in packs)]
        out = Path(d) / "plan.json"
        if a.on:
            r = on_machine(a.on, commit, Path(d), packs, flags, out, notes)
        elif commit == head:
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
