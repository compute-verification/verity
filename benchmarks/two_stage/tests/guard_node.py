"""`TwoGuard.lean` on a node's warm Lean tree:

    research run --on NODE --pool research --as proofs --project verity --source . --cwd clone -- \\
      python3 benchmarks/two_stage/tests/guard_node.py DIR check|diag [M2]
    ... -- python3 benchmarks/two_stage/tests/guard_node.py --pytest [PYTEST ARGS]

It holds the node's `build` Lean slot and its tree as `lean_changed.py --try` does (the slot, the tree's lock, the sync of
`Security/Proofs`' closure from this checkout, the tree's kept dependencies), builds `Proofs.Flock.SpotFlock.Code` there,
and then, still holding them, runs `lake env lean --run TwoGuard.lean` on its arguments in the tree's `Security/Proofs`,
or with ``--pytest`` the guard's test (`test_guard.py`) with ``VERITY_TWO_STAGE_GUARD`` naming that directory. Exit:
the run's status, 2 when it got no slot, tree or dependencies."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tools" / "verity" / "lean"))
import lean_changed as LC  # noqa: E402

PKG, MODULE = "Security/Proofs", "Proofs.Flock.SpotFlock.Code"


def main(argv: list[str]) -> int:
    lake = shutil.which("lake") or str(Path.home() / ".elan" / "bin" / "lake")
    files = LC.LA.COMMON.tracked_files(LC.ROOT)
    if files is None:
        print("guard-node: the tree's files cannot be listed")
        return 2
    os.environ.pop(LC.LEAN_SLOT.HELD, None)
    try:
        slot = LC.LEAN_SLOT.hold(LC.FAST_POOL, or_any=False)
    except LC.LEAN_SLOT.Refused as e:
        print(f"guard-node: no `{LC.FAST_POOL}` Lean slot here ({e})")
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
                print(f"guard-node: {PKG}: no dependencies kept on this node")
                return 2
            LC.write_links(tree, links)
            LC.mark(tree, package=PKG, phase="build")
            print(f"guard-node: {tree} ({n} files changed there): lake build +{MODULE}", flush=True)
            rc, _ = LC.build(PKG, tree, lake, [f"+{MODULE}"], LC.sandboxed())
            if rc == 0:
                if argv[:1] == ["--pytest"]:
                    env = {**os.environ, "VERITY_TWO_STAGE_GUARD": str(tree / PKG)}
                    rc = subprocess.run([sys.executable, "-m", "pytest", "-q", str(HERE / "test_guard.py"), *argv[1:]],
                                        cwd=ROOT, env=env).returncode
                else:
                    rc, _ = LC.build(PKG, tree, lake, sandbox=False,
                                     cmd=["env", "lean", "--run", str(HERE / "TwoGuard.lean"), *argv])
            (tree / LC.DIRTY).unlink(missing_ok=True)
            return rc
    finally:
        if slot is not None:
            os.close(slot)
            os.environ.pop(LC.LEAN_SLOT.HELD, None)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
