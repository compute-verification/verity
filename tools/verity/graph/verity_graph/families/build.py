"""The build family: the check's steps that build the Flock verifier and run what it built, and Flock's Rust, as
`tools/verity/check/check.py` runs them (its `lean_steps` and `groups`), each step one task.

- `lean-build:verity/core/service/flock`: `lake build` of the Lean verifier; its outputs are the executables its lakefile's
  `defaultTargets` name, each at its path in a built tree (`.lake/build/bin/<exe>`, where the suites that run them look), and
  `build`, the rest of that `.lake/build` packed (`.lake/build.tar.gz`), with which a reader's `lake build` finds every
  target built (`lean_outputs`). The test files of the suites check runs after lean-build read them all (`lean_suites`: the
  test-file family's tasks of those suites).
- `lean-unit-cut:verity/ml/flock/verifier/unit_cut_agree.py`: the script on the flock-verify it reads, the Lean partition
  check's conformance with `verity.ml.circuits.partition`; its inputs are the script's import closure (`test_file.Imports`: it
  reads no data), its output the script's result.json.
- `lean-agreement:verity/ml/flock/verifier`: `check.py --lean-agreement` on the flock-verify it reads, against upstream's Rust
  verifier on the upstream build and inputs `upstream.json` pins; its output is ci.py's per-set verdicts.
- `flock-circuit-build:verity/ml/flock/live`: `check_build.sh`, `cargo check` of flock-circuit in each CPU feature set a pod
  builds. It outputs nothing: `cargo check` builds nothing a reader runs.
- `flock-rust-tests:verity/ml/flock/live`: `check_build.sh test`, the Rust tests in release on the same checkout, which it makes
  itself: it reads nothing of flock-circuit-build's.

A tree without the verifier's Lean package or its pin lists none of the Lean tasks, as check skips them there; one without
`check_build.sh` none of the Rust ones."""
from __future__ import annotations

import ast
import tomllib
from pathlib import Path

from .test_file import Imports, Layout, Tree

LEAN = "verity/core/service/flock"
BUILD = f"lean-build:{LEAN}"
BIN = f"{LEAN}/.lake/build/bin"
VERIFIER = "verity/ml/flock/verifier"
FLOCK = "verity/ml/flock"
UNIT_CUT = f"{VERIFIER}/unit_cut_agree.py"
SUITES = "tools/verity/check/suites.py"
#: what `check.py --lean-agreement` reads of the tree outside the verifier's directory: check.py, the modules it loads as it
#: starts and the job declaration and policy they read, with the Slack registry the declaration checks its handles against
#: (`tests/test_build.py` holds this list to what an audit hook sees it open)
CHECK_READS = (*(f"tools/verity/check/{m}" for m in ("check.py", "ci.toml", "common.py", "declaration.py", "live.py",
                                                     "policy.toml", "settings.py", "shards.py", "slot.py", "spans.py",
                                                     "suites.py", "verdicts.py")),
               ".agents/skills/using-slack/registry.json")


def package(root: Path, files: list[str]) -> list[str]:
    """The Lean verifier's files `lake build` can read: its package's, less its lock (`lean-audit.json`, which only the Lean
    check reads) and any Lake package nested in it while its lakefile requires nothing (`check.agreement_closure`'s cut)."""
    pre = LEAN + "/"
    lakefiles = [f for f in files if f.startswith(pre) and f.rsplit("/", 1)[-1] in ("lakefile.toml", "lakefile.lean")]
    top = [f for f in lakefiles if f.count("/") == pre.count("/")]
    nested = tuple(f.rsplit("/", 1)[0] + "/" for f in lakefiles if f not in top)
    if any("require" in (root / f).read_text() for f in top):
        nested = ()
    return [f for f in files if f.startswith(pre) and f != pre + "lean-audit.json" and not f.startswith(nested)]


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
    """The suites whose test files run on what lean-build built, each by its name or path: `after_lean` where the tree has the
    verifier lean-build builds (`executables`), none where it hasn't, as check runs them with the rest of the suites there."""
    return after_lean(root, files) if executables(root, files) else []


def lean_outputs(root: Path, files: list[str]) -> list[dict]:
    """lean-build's outputs, none where the tree lacks the verifier: each of its `executables` at `.lake/build/bin/<exe>`, and
    `build`, the rest of its `.lake/build` packed (`runners/lean_build.py`'s `pack`, which `place` unpacks)."""
    exes = executables(root, files)
    return [*({"name": e, "path": f"{BIN}/{e}"} for e in exes), {"name": "build", "path": f"{LEAN}/.lake/build.tar.gz"}] if exes else []


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
    out.append({"id": BUILD, "kind": "lean-build", "env": "lean", "about": f"lake build the Lean verifier ({LEAN})",
                "inputs": package(root, files), "outputs": lean_outputs(root, files)})
    if "flock-verify" in exes and UNIT_CUT in files:
        out.append({"id": f"lean-unit-cut:{UNIT_CUT}", "kind": "lean-unit-cut", "env": "py",
                    "about": "the Lean partition check's codes against verity.ml.circuits.partition's on random cuts",
                    "vars": {"verify": f"{BIN}/flock-verify"}, "inputs": unit_cut_inputs(root, files),
                    "reads": [{"task": BUILD, "output": "flock-verify"}],
                    "outputs": [{"name": "result", "path": "lean-unit-cut/result.json"}],
                    "resources": {"cores": 1, "mem_gb": 4, "disk_gb": 1}, "timeout_s": 900})
    if "flock-verify" in exes and f"{VERIFIER}/upstream.json" in files:
        out.append({"id": f"lean-agreement:{VERIFIER}", "kind": "lean-agreement", "env": "py",
                    "about": "agree with upstream's Rust verifier on the pinned upstream build and inputs (ci.py)",
                    "vars": {"verify": f"{BIN}/flock-verify"},
                    "inputs": sorted({*(f for f in files if f.startswith(VERIFIER + "/")), *(f for f in CHECK_READS if f in files)}),
                    "reads": [{"task": BUILD, "output": "flock-verify"}],
                    "outputs": [{"name": "sets", "path": "flock-agreement/ci/sets.json"}]})
    return out
