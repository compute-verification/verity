"""Scratch, on a branch of its own and never for landing: elaborate `benchmarks/two_stage/tests/TwoGuard.lean` on a
node's warm Lean tree without running it.

    research run --on auto --as proofs --project verity --source . --cwd source -- python3 scratch-widezk/twoguard_elab.py

The script is in no Lake package, so `lean_changed.py --build` and `--try` skip it, and no check compiles it. This takes
the tree as `benchmarks/two_stage/tests/guard_node.py` does (the node's `build` Lean slot, the tree's lock, the sync of
`security/Proofs`' closure from this checkout, the tree's kept dependencies), builds the two modules the script imports
there, and then, still holding them, runs `lake env lean` on the script from the package's directory: Lean elaborates
every declaration of it and the kernel checks each, and `main` is not run. Lean's output goes to stdout and to
``$OUT/twoguard-elab.log``. Exit: lean's status on the script, the build's when the imports do not build, 2 when it got
no slot, tree or dependencies."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "verity" / "lean"))
import lean_changed as LC  # noqa: E402

PKG = "security/Proofs"
MODULES = ("Proofs.Flock.SpotFlock.Code", "Proofs.Flock.SamplingGeneric.TwoAt")
SCRIPT = ROOT / "benchmarks" / "two_stage" / "tests" / "TwoGuard.lean"


def elaborate(tree: Path, lake: str) -> int:
    t0 = time.monotonic()
    p = subprocess.run([lake, "env", "lean", str(SCRIPT)], cwd=tree / PKG, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    text = p.stdout + p.stderr
    if out := os.environ.get("OUT"):
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "twoguard-elab.log").write_text(text)
    print(text, end="" if text.endswith("\n") or not text else "\n")
    errors = [ln for ln in text.splitlines() if ": error" in ln]
    print(f"twoguard-elab: lake env lean {SCRIPT.relative_to(ROOT)}: exit {p.returncode} in {time.monotonic() - t0:.0f} s, "
          f"{len(errors)} error line(s), {sum(': warning' in ln for ln in text.splitlines())} warning line(s)", flush=True)
    return p.returncode


def main() -> int:
    lake = shutil.which("lake") or str(Path.home() / ".elan" / "bin" / "lake")
    files = LC.LA.COMMON.tracked_files(LC.ROOT)
    if files is None:
        print("twoguard-elab: the tree's files cannot be listed")
        return 2
    os.environ.pop(LC.LEAN_SLOT.HELD, None)
    try:
        slot = LC.LEAN_SLOT.hold(LC.FAST_POOL, or_any=False)
    except LC.LEAN_SLOT.Refused as e:
        print(f"twoguard-elab: no `{LC.FAST_POOL}` Lean slot here ({e})")
        return 2
    try:
        tree = LC.made(LC.records_tree())
        with open(tree / ".lock", "w") as held:
            if not LC.hold_tree(held, tree, print):
                return 2
            links, left = LC.read_links(tree), LC.left_by(tree)
            LC.mark(tree, package=PKG, phase="sync")
            n = LC.sync(LC.closure(PKG), files, tree)
            there = [q for q in [*LC.closure(PKG), (left or {}).get("package")] if isinstance(q, str) and (tree / q).is_dir()]
            LC.repair(tree, list(dict.fromkeys(there)), links, left, print)
            LC.write_links(tree, links)
            LC.mark(tree, package=PKG, phase="deps")
            deps = LC.LA.COMMON.cache_dir() / "lean-deps"
            if not ((tree / PKG / ".lake" / "packages").is_dir() and PKG not in links) and \
                    not LC.link(PKG, tree, deps, links, print, copy=True):
                print(f"twoguard-elab: {PKG}: no dependencies kept on this node")
                return 2
            LC.write_links(tree, links)
            LC.mark(tree, package=PKG, phase="build")
            print(f"twoguard-elab: {tree} ({n} files changed there): lake build {' '.join(MODULES)}", flush=True)
            t0 = time.monotonic()
            rc, text = LC.build(PKG, tree, lake, [f"+{m}" for m in MODULES], LC.sandboxed())
            print(f"twoguard-elab: lake build of the script's imports: exit {rc} in {time.monotonic() - t0:.0f} s, "
                  f"{len(LC.compiled(text))} module(s) compiled", flush=True)
            if rc == 0:
                rc = elaborate(tree, lake)
            (tree / LC.DIRTY).unlink(missing_ok=True)
            return rc
    finally:
        if slot is not None:
            os.close(slot)
            os.environ.pop(LC.LEAN_SLOT.HELD, None)


if __name__ == "__main__":
    sys.exit(main())
