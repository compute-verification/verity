#!/usr/bin/env python3
"""check.py NAME HEAD [HEAD ...]: the superseded-versions test on #8463's head merged with each HEAD in turn.

Makes a scratch worktree at BASE, merges the heads one after another, runs the test file with pytest on the merged tree (every
workspace package resolved from that tree) and writes what the test's own functions find to out/NAME.json.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/agent/repos/verity")
WORK = Path("/tmp/cir208")
BASE = "b1409cffb06f0e1b46ca8ee97fc85c28c412523b"
SP = REPO / ".venv/lib/python3.12/site-packages"
PY = REPO / ".venv/bin/python"
TEST = "tools/verity/circuit_check/tests/test_superseded_versions.py"

DETAIL = r"""
import json, sys
sys.path.insert(0, "tools/verity/circuit_check/tests")
import test_superseded_versions as t
import verity, circuit_check, verity_vllm
here = str(t.REPO)
assert all(m.__file__.startswith(here) for m in (verity, circuit_check, verity_vllm)), (verity.__file__, circuit_check.__file__)
ids = t.registered()
assert all(not f.startswith("/") for f, _ in ids.values()), [f for f, _ in ids.values() if f.startswith("/")][:3]
now = t.superseded(ids)
entries, held = t.listed()
f = t.findings(now, entries, held)
new = sorted(i for i in now if i not in entries and i not in held)
print(json.dumps({"ids": {i: [file, boolean] for i, (file, boolean) in ids.items()}, "superseded": len(now),
                  "entries": len(entries),
                  "new": [{"id": i, "file": now[i][0], "by": now[i][1], "boolean": ids[i][1]} for i in new],
                  "gone": f["gone"], "moved": f["moved"], "twice": f["twice"]}))
"""


#: run after the registries load: import every module the merged heads change, and report what that registers beyond them
EXTRA = r"""
import importlib, json, sys
from circuit_check import sidecars, targets
from verity.circuits.defs import REGISTRY
targets.load_registries()
before = set(REGISTRY.defs)
errors = {}
for mod in json.loads(sys.argv[1]):
    try:
        importlib.import_module(mod)
    except BaseException as e:
        errors[mod] = f"{type(e).__name__}: {e}"[:160]
extra = {}
for i in sorted(set(REGISTRY.defs) - before):
    d = REGISTRY.defs[i]
    try:
        extra[i] = [sidecars.rel(sys.modules[sidecars.owner(d)].__file__), getattr(d, "word", None) is not None, targets._authored(d)]
    except Exception as e:
        extra[i] = [f"{type(e).__name__}: {e}"[:120], None, None]
print(json.dumps({"extra": extra, "import_errors": errors}))
"""

PACKAGES = (("verity/", ""), ("adapters/vllm/verity_vllm/", "adapters/vllm/"),
            ("verity/protocols/flock/prover/python/verity_flock/", "verity/protocols/flock/prover/python/"),
            ("benchmarks/numerical/python/verity_numerical/", "benchmarks/numerical/python/"))


def modules(files: list[str]) -> list[str]:
    """The importable modules among ``files``: under a package of the tree, no test, sidecar or conftest."""
    out = []
    for f in files:
        parts = Path(f).parts
        if not f.endswith(".py") or "tests" in parts or parts[-1] == "conftest.py" or "." in parts[-1][:-3]:
            continue
        root = max((r for pre, r in PACKAGES if f.startswith(pre)), key=len, default=None)
        if root is None:
            continue
        rel = f[len(root):-3].replace("/", ".")
        out.append(rel[:-len(".__init__")] if rel.endswith(".__init__") else rel)
    return sorted(set(out))


def roots() -> list[str]:
    out: list[str] = []
    for f in sorted(SP.glob("_editable_impl_*.pth")):
        line = f.read_text().strip().splitlines()[0]
        if line.startswith("/"):
            out.append(line)
        else:
            for m in re.findall(r"/agent/repos/verity[^'\"]*", (SP / f"{line.split()[1]}.py").read_text()):
                p = Path(m)
                out.append(str(p.parent.parent if p.name == "__init__.py" else p.parent))
    return list(dict.fromkeys(out))


#: `--resolve=ours|theirs`: take that side of each conflicting hunk (a scratch resolution, to read the registries of a union)
RESOLVE = [f"-X{a.split('=', 1)[1]}" for a in sys.argv[1:] if a.startswith("--resolve=")]


def git(*args: str, cwd: Path = REPO, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=check)


def main(name: str, heads: list[str]) -> int:
    wt = WORK / "wt" / name
    git("worktree", "remove", "--force", str(wt), check=False)
    git("worktree", "add", "-q", "--detach", str(wt), BASE)
    out: dict = {"name": name, "base": BASE[:10], "heads": [], "conflicts": {}}
    for h in heads:
        sha = git("rev-parse", h).stdout.strip()
        out["heads"].append(sha[:10])
        m = git("merge", "--no-edit", "--no-ff", *RESOLVE, sha, cwd=wt, check=False)
        if m.returncode:
            out["conflicts"][sha[:10]] = git("diff", "--name-only", "--diff-filter=U", cwd=wt).stdout.split()
            out["merge_error"] = (m.stdout + m.stderr)[-600:]
            break
    if not out["conflicts"]:
        env = {**os.environ, "PYTHONPATH": ":".join(r.replace(str(REPO), str(wt), 1) for r in roots())}
        env.pop("VIRTUAL_ENV", None)
        p = subprocess.run([str(PY), "-m", "pytest", "-q", "-p", "no:cacheprovider", TEST], cwd=wt, env=env,
                           capture_output=True, text=True)
        out["pytest"] = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-400:]
        out["pytest_failed"] = sorted(set(re.findall(r"^FAILED \S+::(\S+)", p.stdout, re.M)))
        d = subprocess.run([str(PY), "-c", DETAIL], cwd=wt, env=env, capture_output=True, text=True)
        if d.returncode:
            out["detail_error"] = d.stderr[-1500:]
        else:
            out.update(json.loads(d.stdout.splitlines()[-1]))
        changed = git("diff", "--name-only", "--diff-filter=AM", BASE, "HEAD", "--", "*.py", cwd=wt).stdout.split()
        out["changed_modules"] = mods = modules(changed)
        x = subprocess.run([str(PY), "-c", EXTRA, json.dumps(mods)], cwd=wt, env=env, capture_output=True, text=True)
        if x.returncode:
            out["extra_error"] = x.stderr[-1500:]
        else:
            out.update(json.loads(x.stdout.splitlines()[-1]))
    (WORK / "out").mkdir(exist_ok=True)
    (WORK / "out" / f"{name}.json").write_text(json.dumps(out, indent=1) + "\n")
    brief = {k: v for k, v in out.items() if k != "ids"}
    brief["registered"] = len(out.get("ids", {}))
    print(json.dumps(brief))
    if not out["conflicts"] and "--keep" not in sys.argv:
        git("worktree", "remove", "--force", str(wt), check=False)
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(main(args[0], args[1:]))
