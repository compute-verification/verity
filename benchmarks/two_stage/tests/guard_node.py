"""`TwoGuard.lean` on a node's warm Lean tree:

    research run --on NODE --pool research --as proofs --project verity --source . --cwd clone -- \\
      python3 benchmarks/two_stage/tests/guard_node.py [--interpret] DIR check|diag [M2]
    ... -- python3 benchmarks/two_stage/tests/guard_node.py --pytest [PYTEST ARGS]

It holds the node's `build` Lean slot and its tree as `lean_changed.py --try` does (the slot, the tree's lock, the sync of
`Security/Proofs`' closure from this checkout, the tree's kept dependencies) and builds the two modules the script imports
there. Then, still holding them, it compiles the script as an executable of `Security/Proofs` (`twoguard`, from a config of
its own: the package's with the executable added, beside the script's copy under the package's `.lake/twoguard/`) and runs
it on its arguments, or with ``--pytest`` the guard's test (`test_guard.py`) with ``VERITY_TWO_STAGE_GUARD`` naming it.
``--interpret`` runs the script in Lean's interpreter instead (`lake env lean --run`), whose every setup of a statement
takes minutes. Exit: the run's status, 2 when it got no slot, tree or dependencies."""
from __future__ import annotations

import os
import resource
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tools" / "verity" / "lean"))
import lean_changed as LC  # noqa: E402

PKG = "Security/Proofs"
MODULES = ("Proofs.Flock.SpotFlock.Code", "Proofs.Flock.SamplingGeneric.TwoAt")
EXE = "twoguard"
#: the interpreter's thread stack, in KiB (`lean --tstack`): reserved, and touched only as deep as the parsers recurse. Every
#: thread of Lean's gets one, so the script runs with one worker (`--threads=1`), or the node refuses the reservations.
TSTACK_KIB = 128 << 20
#: the executable's stack, in bytes (its main thread's `RLIMIT_STACK`)
STACK = 64 << 30


def raise_stack() -> None:
    _, hard = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK, (STACK if hard == resource.RLIM_INFINITY else min(STACK, hard), hard))


def exe_config(tree: Path) -> list[str]:
    """`lake` arguments that build `twoguard` in ``tree``'s `Security/Proofs`: the package's config with the executable
    added, rooted at the script's copy, both under the package's `.lake/twoguard/`."""
    d = tree / PKG / ".lake" / EXE
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "TwoGuard.lean", d / "TwoGuard.lean")
    cfg = d / "lakefile.toml"
    cfg.write_text((tree / PKG / "lakefile.toml").read_text()
                   + f'\n[[lean_exe]]\nname = "{EXE}"\nroot = "TwoGuard"\nsrcDir = "Proofs/.lake/{EXE}"\n')
    return ["-f", str(cfg), "build", EXE]


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
            print(f"guard-node: {tree} ({n} files changed there): lake build {' '.join(MODULES)}", flush=True)
            rc, _ = LC.build(PKG, tree, lake, [f"+{m}" for m in MODULES], LC.sandboxed())
            if rc == 0 and argv[:1] == ["--interpret"]:
                rc, _ = LC.build(PKG, tree, lake, sandbox=False, cmd=[
                    "env", "lean", "--threads=1", f"--tstack={TSTACK_KIB}", "--run", str(HERE / "TwoGuard.lean"), *argv[1:]])
            elif rc == 0:
                print(f"guard-node: lake build {EXE}", flush=True)
                rc, _ = LC.build(PKG, tree, lake, sandbox=LC.sandboxed(), cmd=exe_config(tree))
                exe = tree / PKG / ".lake" / "build" / "bin" / EXE
                if rc == 0 and argv[:1] == ["--pytest"]:
                    env = {**os.environ, "VERITY_TWO_STAGE_GUARD": str(exe)}
                    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
                    rc = subprocess.run([uv, "run", "--no-project", "--with", "pytest", "python", "-m", "pytest", "-q",
                                         str(HERE / "test_guard.py"), *argv[1:]], cwd=ROOT, env=env).returncode
                elif rc == 0:
                    rc = subprocess.run([str(exe), *argv], preexec_fn=raise_stack).returncode
            (tree / LC.DIRTY).unlink(missing_ok=True)
            return rc
    finally:
        if slot is not None:
            os.close(slot)
            os.environ.pop(LC.LEAN_SLOT.HELD, None)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
