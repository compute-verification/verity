"""The build family: the check's steps that build the Flock verifier and run what it built, and Flock's Rust, as
`tools/verity/check/check.py` runs them (its `lean_steps` and `groups`), each step one task.

- `lean-build:verity/protocols/flock/verifier`: `lake build` of the Lean verifier, which reads its package and the packages of
  the tree it requires by path; its outputs are the executables its lakefile's
  `defaultTargets` name, each at its path in a built tree (`.lake/build/bin/<exe>`, where the suites that run them look), and
  the rest of the build, each of those packages' `.lake/build` in one archive (`TREE`, its var `builds` the packages).
- `lean-unit-cut:verity/protocols/flock/agreement/unit_cut_agree.py`: the script on the flock-verify it reads, the Lean partition
  check's conformance with `verity.circuits.partition`; its inputs are the script's import closure (`test_file.Imports`: it
  reads no data), its output the script's result.json.
- `lean-agreement:verity/protocols/flock/agreement`: `check.py --lean-agreement` on the flock-verify it reads, against upstream's Rust
  verifier on the upstream build and inputs `upstream.json` pins; its output is ci.py's per-set verdicts.
- `lean-suites:SUITE`: `suites.py` on one of the suites check runs after lean-build (`suites.AFTER_LEAN`, read from its source:
  `lean_suites`, whose test files the test-file family leaves out), on every executable lean-build outputs and the build they
  were linked in, unpacked into the tree, where the tests' own `lake build`s find it, and
  `lean-suites:SUITE:K/M` on part K of M of one split by test file (`PARTS`) or by test id (the `parts` it declares), so a
  task holds about ten minutes of a suite where its tests allow; its inputs are the whole tree, as tasks/v1 keys a job that
  declares none, since those suites build and run Lake packages no tracer follows; its output is suites.py's report. A suite
  that declares `parallel` takes the family's slot, and one that doesn't one process (`suites.TEST_PROCESS_GB`, or its
  `worker_gb`); its timeout is its own `timeout_s`, else the family's.
- `flock-circuit-build:verity/protocols/flock/prover/live`: `check_build.sh`, `cargo check` of flock-circuit in each CPU feature set a pod
  builds. It outputs nothing: `cargo check` builds nothing a reader runs.
- `flock-rust-tests:verity/protocols/flock/prover/live`: `check_build.sh test`, the Rust tests in release on the same checkout, which it makes
  itself: it reads nothing of flock-circuit-build's.

A tree without the verifier's Lean package or its pin lists none of the Lean tasks, as check skips them there; one without
`check_build.sh` none of the Rust ones."""
from __future__ import annotations

import ast
import json
import tomllib
from pathlib import Path

from ..runners.lean_build import TREE
from .test_file import Imports, Layout, Tree, discover

LEAN = "verity/protocols/flock/verifier"
VERIFIER = "verity/protocols/flock/agreement"
FLOCK = "verity/protocols/flock/prover"
UNIT_CUT = f"{VERIFIER}/unit_cut_agree.py"
SUITES = "tools/verity/check/suites.py"
#: the lean-suites suites listed in parts, {suite: parts}: `suites.py --files K/M` on a timing record that has none of their
#: files (`lean_suites.run`), so each part is every M-th of the suite's test files in path order, a function of the commit alone,
#: and the parts of one tree hold each file once whichever machines run them.  Each is as many as keep its parts near ten minutes
#: at its files' recorded seconds, or near its longest file.  A suite that declares `[tool.verity.tests] parts` is in those
#: parts by test id, as check runs it, and isn't listed here
PARTS = {"verity-flock": 6, "verity-vllm": 2, "pous-check": 3, "verity-pouw-check-difftest": 2,
         "verity/protocols/certifier/verifier": 2}
#: a lean-suites task's disk: the suites build Lake packages of their own beside the verifier's
SUITES_DISK_GB = 60
#: where lean-build's output holding the rest of its build (`lean_build.TREE`: each `.lake/build` but the executables) is in a work
#: directory
TREE_PATH = "lean-build/lake-build.tar.gz"
#: what `check.py --lean-agreement` reads of the tree outside the verifier's directory: check.py, the modules it loads as it
#: starts and the job declaration and policy they read, with the Slack registry the declaration checks its handles against
#: (`tests/test_build.py` holds this list to what an audit hook sees it open)
CHECK_READS = (*(f"tools/verity/check/{m}" for m in ("check.py", "ci.toml", "common.py", "declaration.py", "live.py",
                                                     "policy.toml", "records.py", "settings.py", "shards.py", "slot.py",
                                                     "spans.py", "suites.py", "verdicts.py")),
               ".agents/skills/using-slack/registry.json")


def required(root: Path, files: list[str]) -> list[str]:
    """The packages of the tree the Lean verifier requires by path from outside its directory, by its manifest (which lists
    those its requirements require too) or its lakefile, each from the root and ending in "/" (`check.required_in_tree`)."""
    entries: list[dict] = []
    if f"{LEAN}/lake-manifest.json" in files:
        entries += json.loads((root / LEAN / "lake-manifest.json").read_text()).get("packages", [])
    if f"{LEAN}/lakefile.toml" in files:
        entries += tomllib.loads((root / LEAN / "lakefile.toml").read_text()).get("require", [])
    top, own = root.resolve(), (root / LEAN).resolve()
    dirs = {(own / p).resolve() for e in entries if (p := e.get("dir") or e.get("path")) is not None}
    return sorted(d.relative_to(top).as_posix() + "/" for d in dirs if d.is_relative_to(top) and not d.is_relative_to(own))


def package(root: Path, files: list[str]) -> list[str]:
    """The files `lake build` of the Lean verifier can read: its package's and those of the packages of the tree it requires
    by path (`required`), less each one's lock (`lean-audit.json`, which only the Lean check reads) and any Lake package
    nested in the verifier's while its lakefile requires nothing (`check.agreement_closure`'s cut)."""
    pre = LEAN + "/"
    lakefiles = [f for f in files if f.startswith(pre) and f.rsplit("/", 1)[-1] in ("lakefile.toml", "lakefile.lean")]
    top = [f for f in lakefiles if f.count("/") == pre.count("/")]
    nested = tuple(f.rsplit("/", 1)[0] + "/" for f in lakefiles if f not in top)
    if any("require" in (root / f).read_text() for f in top):
        nested = ()
    packages = (pre, *required(root, files))
    locks = {p + "lean-audit.json" for p in packages}
    return [f for f in files if f.startswith(packages) and f not in locks and not f.startswith(nested)]


def executables(root: Path, files: list[str]) -> list[str]:
    """The executables `lake build` builds in the verifier: the `lean_exe`s its lakefile's `defaultTargets` name."""
    if f"{LEAN}/lakefile.toml" not in files:
        return []
    doc = tomllib.loads((root / LEAN / "lakefile.toml").read_text())
    exes = {e.get("name") for e in doc.get("lean_exe", [])}
    return sorted(t for t in doc.get("defaultTargets", []) if t in exes)


def rust_inputs(files: list[str]) -> list[str]:
    """What `check_build.sh` reads of the tree: the script, which pins the Flock commit and the Rust toolchain, the patches it
    applies and the live crate it links in (`check.rust_tests_inputs`)."""
    pre = FLOCK + "/"
    return [f for f in files if f == pre + "check_build.sh" or f.startswith(pre + "live/")
            or (f.startswith(pre + "flock-") and f.endswith("-b684b12.patch"))]


def suites_literals(root: Path, files: list[str], *names: str) -> dict[str, object]:
    """The module-level literals ``names`` as suites.py's source writes them, read without running it (those it lacks left out)."""
    out: dict[str, object] = {}
    if SUITES not in files:
        return out
    for node in ast.parse((root / SUITES).read_text()).body:
        if isinstance(node, ast.Assign) and (hit := [t.id for t in node.targets if isinstance(t, ast.Name) and t.id in names]):
            out[hit[0]] = ast.literal_eval(node.value)
    return out


def after_lean(root: Path, files: list[str]) -> list[str]:
    """`suites.AFTER_LEAN` as suites.py's source writes it, the suites check's lean-suites step runs; none when it has none."""
    return [str(s) for s in suites_literals(root, files, "AFTER_LEAN").get("AFTER_LEAN", ())]


def lean_suites(root: Path, files: list[str]) -> list[str]:
    """The suites the lean-suites tasks run, each by its name or path: `after_lean` where the tree has the verifier lean-build
    builds (`executables`), none where it hasn't, as check runs them with the rest of the suites there.  The test-file family
    lists none of their test files, so each runs once, after the build it needs."""
    return after_lean(root, files) if executables(root, files) else []


def suite_tasks(root: Path, files: list[str], build: str, bin_: str, exes: list[str]) -> list[dict]:
    """A lean-suites task for each suite of `lean_suites`, or one for each of its parts (`suites.py NAME --files K/M`): by test
    id the `[tool.verity.tests] parts` it declares, else its `PARTS`.  Each has its suite's resources and timeout: a suite that
    declares `[tool.verity.tests] parallel` the family's slot, else one process of its `worker_gb` (`suites.TEST_PROCESS_GB`
    undeclared); its `timeout_s`, else the family's.  A name no suite has is listed all the same, so suites.py's refusal of it
    is the task's error."""
    names = lean_suites(root, files)
    if not names:
        return []
    tree = Tree(root, files)
    by = {k: s for s in discover(tree) for k in (s.name, s.path)}
    one_gb = suites_literals(root, files, "TEST_PROCESS_GB").get("TEST_PROCESS_GB")
    out = []
    for name in names:
        s = by.get(name)
        tests = tree.toml(s.dir + "/pyproject.toml" if s.dir else "pyproject.toml").get("tool", {}).get(
            "verity", {}).get("tests", {}) if s else {}
        gb = (s.mem_gb if s else None) or one_gb
        resources = {"disk_gb": SUITES_DISK_GB} if tests.get("parallel") is True else {
            "cores": 1, **({"mem_gb": gb} if gb else {}), "disk_gb": SUITES_DISK_GB}
        task = {"kind": "lean-suites", "env": "py", "inputs": list(files),
                "reads": [*({"task": build, "output": e} for e in exes), {"task": build, "output": TREE}],
                "outputs": [{"name": "report", "path": "suites-lean.json"}], "resources": resources,
                **({"timeout_s": s.timeout_s} if s and s.timeout_s else {})}
        by_id = tests.get("parts") if type(tests.get("parts")) is int else None
        staged = {"bin": bin_, "tree": TREE_PATH}
        if m := by_id or PARTS.get(s.name if s else name):
            of = f"{name}'s tests by id" if by_id else f"{name}'s test files (one in {m}, by path)"
            out += [{"id": f"lean-suites:{name}:{k}/{m}", **task, "vars": {"suites": name, "files": f"{k}/{m}", **staged},
                     "about": f"suites.py --lean-built on part {k} of {m} of {of}"} for k in range(1, m + 1)]
        else:
            out.append({"id": f"lean-suites:{name}", **task, "vars": {"suites": name, **staged},
                        "about": f"suites.py --lean-built on {name}, as check's lean-suites step runs it"})
    return out


def unit_cut_inputs(root: Path, files: list[str]) -> list[str]:
    """`unit_cut_agree.py` and what running it imports: `python SCRIPT` puts its directory on `sys.path`."""
    tree = Tree(root, files)
    return sorted(Imports(tree, Layout(tree)).closure([UNIT_CUT], [VERIFIER], [UNIT_CUT]))


def tasks(root: Path, files: list[str]) -> list[dict]:
    out = []
    if (rust := rust_inputs(files)) and f"{FLOCK}/check_build.sh" in rust:
        out.append({"id": f"flock-circuit-build:{FLOCK}/live", "kind": "flock-circuit-build", "env": "rust",
                    "about": "cargo check flock-circuit in each CPU feature set a pod builds (check_build.sh)", "inputs": rust})
        out.append({"id": f"flock-rust-tests:{FLOCK}/live", "kind": "flock-rust-tests", "env": "rust",
                    "about": "cargo test, in release, of the live crate and the crates the patches change (check_build.sh test)",
                    "inputs": rust})
    exes = executables(root, files)
    if not exes:
        return out
    build = f"lean-build:{LEAN}"
    bin_ = f"{LEAN}/.lake/build/bin"
    out.append({"id": build, "kind": "lean-build", "env": "lean", "about": f"lake build the Lean verifier ({LEAN})",
                "vars": {"builds": " ".join([LEAN, *(d.rstrip("/") for d in required(root, files))])},
                "inputs": package(root, files), "outputs": [*({"name": e, "path": f"{bin_}/{e}"} for e in exes),
                                                            {"name": TREE, "path": TREE_PATH}]})
    if "flock-verify" in exes and UNIT_CUT in files:
        out.append({"id": f"lean-unit-cut:{UNIT_CUT}", "kind": "lean-unit-cut", "env": "py",
                    "about": "the Lean partition check's codes against verity.circuits.partition's on random cuts",
                    "vars": {"verify": f"{bin_}/flock-verify"}, "inputs": unit_cut_inputs(root, files),
                    "reads": [{"task": build, "output": "flock-verify"}],
                    "outputs": [{"name": "result", "path": "lean-unit-cut/result.json"}],
                    "resources": {"cores": 1, "mem_gb": 4, "disk_gb": 1}, "timeout_s": 900})
    if "flock-verify" in exes and f"{VERIFIER}/upstream.json" in files:
        out.append({"id": f"lean-agreement:{VERIFIER}", "kind": "lean-agreement", "env": "py",
                    "about": "agree with upstream's Rust verifier on the pinned upstream build and inputs (ci.py)",
                    "vars": {"verify": f"{bin_}/flock-verify"},
                    "inputs": sorted({*(f for f in files if f.startswith(VERIFIER + "/")), *(f for f in CHECK_READS if f in files)}),
                    "reads": [{"task": build, "output": "flock-verify"}],
                    "outputs": [{"name": "sets", "path": "flock-agreement/ci/sets.json"}]})
    return out + suite_tasks(root, files, build, bin_, exes)
