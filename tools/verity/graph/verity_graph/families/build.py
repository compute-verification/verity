"""The build family: the check's steps that build the Flock verifier and run what it built, as `tools/verity/check/check.py`
runs them (its `lean_steps` and `groups`).

- `lean-build:verity/core/service/flock`: `lake build` of the Lean verifier; its outputs are the executables its lakefile's
  `defaultTargets` name, each at its path in a built tree (`.lake/build/bin/<exe>`, where the suites that run them look).
- `lean-agreement:verity/ml/flock/verifier`: `check.py --lean-agreement` on the flock-verify it reads, against upstream's Rust
  verifier on the upstream build and inputs `upstream.json` pins; its output is ci.py's per-set verdicts.
- `flock-circuit-build:verity/ml/flock/live`: `check_build.sh`, `cargo check` of flock-circuit in each CPU feature set a pod
  builds. It outputs nothing: `cargo check` builds nothing a reader runs.

A tree without the verifier's Lean package or its pin lists none of the Lean tasks, as check skips them there."""
from __future__ import annotations

import tomllib
from pathlib import Path

LEAN = "verity/core/service/flock"
VERIFIER = "verity/ml/flock/verifier"
FLOCK = "verity/ml/flock"
#: what `check.py --lean-agreement` reads of the tree outside the verifier's directory: check.py, the modules it loads as it
#: starts and the job declaration they read, with the Slack registry the declaration checks its handles against
#: (`tests/test_build.py` holds this list to what an audit hook sees it open)
CHECK_READS = (*(f"tools/verity/check/{m}" for m in ("check.py", "ci.toml", "common.py", "declaration.py", "live.py",
                                                     "settings.py", "shards.py", "slot.py", "spans.py", "suites.py",
                                                     "verdicts.py")),
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


def tasks(root: Path, files: list[str]) -> list[dict]:
    out = []
    if (rust := rust_inputs(files)) and f"{FLOCK}/check_build.sh" in rust:
        out.append({"id": f"flock-circuit-build:{FLOCK}/live", "kind": "flock-circuit-build", "env": "rust",
                    "about": "cargo check flock-circuit in each CPU feature set a pod builds (check_build.sh)", "inputs": rust})
    exes = executables(root, files)
    if not exes:
        return out
    build = f"lean-build:{LEAN}"
    out.append({"id": build, "kind": "lean-build", "env": "lean", "about": f"lake build the Lean verifier ({LEAN})",
                "inputs": package(root, files),
                "outputs": [{"name": e, "path": f"{LEAN}/.lake/build/bin/{e}"} for e in exes]})
    if "flock-verify" in exes and f"{VERIFIER}/upstream.json" in files:
        out.append({"id": f"lean-agreement:{VERIFIER}", "kind": "lean-agreement", "env": "py",
                    "about": "agree with upstream's Rust verifier on the pinned upstream build and inputs (ci.py)",
                    "vars": {"verify": f"{LEAN}/.lake/build/bin/flock-verify"},
                    "inputs": sorted({*(f for f in files if f.startswith(VERIFIER + "/")), *(f for f in CHECK_READS if f in files)}),
                    "reads": [{"task": build, "output": "flock-verify"}],
                    "outputs": [{"name": "sets", "path": "flock-agreement/ci/sets.json"}]})
    return out
