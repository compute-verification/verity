"""`TwoGuard.lean` on a node's warm Lean tree:

    research run --on NODE --pool research --as proofs --project verity --source . --cwd clone -- \\
      python3 benchmarks/two_stage/tests/guard_node.py DIR check|diag [M2] [+ DIR check|diag [M2] ...]
    ... -- python3 benchmarks/two_stage/tests/guard_node.py --pytest [PYTEST ARGS]

It holds the node's `build` Lean slot and its tree as `lean_changed.py --try` does (the slot, the tree's lock, the sync of
`Security/Proofs`' closure from this checkout, the tree's kept dependencies) and builds the two modules the script imports
there. Then, still holding them, it runs the script in Lean's interpreter (`lake env lean --run`) with the code of Flock's
verifier and SPOT's (`NATIVE`) loaded as native code, linked into one shared library from their
archives, which the interpreter calls in place of interpreting them: interpreted, one setup of a statement takes minutes.
The rest (the model, Mathlib) stays interpreted: the kept dependencies are read-only, so nothing that imports Mathlib
compiles here. `.lake/twoguard/twoguard` under the package runs it on each run its arguments name, all at once, each
one's output in ``$OUT/guard-<i>.log`` (`together`); ``--pytest`` runs the guard's test (`test_guard.py`) with
``VERITY_TWO_STAGE_GUARD`` naming it. Exit: the run's status, 2 when it got no slot, tree or
dependencies."""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from verity_config.settings import of

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tools" / "verity" / "lean"))
import lean_changed as LC  # noqa: E402

PKG = "Security/Proofs"
MODULES = ("Proofs.Flock.SpotFlock.Code", "Proofs.Flock.SamplingGeneric.TwoAt")
#: the libraries run natively: SPOT's verifier and every library it imports, none of which imports
#: Mathlib (VStar and Flock import each other, so they link together, which Lake's per-library shared libraries can't)
NATIVE = ("flock_verifier/VStar", "flock_verifier/Flock", "core_protocol/CoreProtocol", "verity_difftest/Difftest",
          "lms/Lms", "core_spot/Spot")
#: the interpreter's thread stack, in KiB (`lean --tstack`): reserved, and touched only as deep as the parsers recurse. Every
#: thread of Lean's gets one, so the script runs with one worker (`--threads=1`), or the node refuses the reservations.
TSTACK_KIB = 128 << 20


def runner(tree: Path, lake: str) -> Path | None:
    """`twoguard` under ``tree``'s `Security/Proofs/.lake/twoguard/`: the script in the interpreter with `NATIVE` loaded;
    None when they don't build or link."""
    rc, text = LC.build(PKG, tree, lake, sandbox=LC.sandboxed(), cmd=["query", *(f"{lib}:static.export" for lib in NATIVE)])
    archives = [line for line in text.splitlines() if line.endswith(".a.export")]
    if rc or len(archives) != len(NATIVE):
        print(f"guard-node: the native libraries' archives: lake query exited {rc} and named {len(archives)}")
        return None
    pkg = tree / PKG
    prefix = subprocess.run([lake, "env", "lean", "--print-prefix"], cwd=pkg, capture_output=True, text=True).stdout.strip()
    d = pkg / ".lake" / "twoguard"
    d.mkdir(parents=True, exist_ok=True)
    so = d / "libtwoguard.so"
    link = [str(Path(prefix) / "bin" / "leanc"), "-shared", "-o", str(so),
            "-Wl,--whole-archive", *archives, "-Wl,--no-whole-archive"]
    print(f"guard-node: {shlex.join(link)}", flush=True)
    if subprocess.run(link).returncode:
        return None
    exe = d / "twoguard"
    exe.write_text(f"#!/bin/sh\ncd {shlex.quote(str(pkg))} && exec {shlex.quote(lake)} env lean --threads=1 "
                   f"--tstack={TSTACK_KIB} --load-dynlib={shlex.quote(str(so))} --run "
                   f"{shlex.quote(str(HERE / 'TwoGuard.lean'))} \"$@\"\n")
    exe.chmod(0o755)
    return exe


def together(exe: Path, argv: list[str]) -> int:
    """``exe`` on each run ``argv`` names (``DIR MODE [M2]``, runs separated by ``+``) at once, each's output in
    ``$OUT/guard-<i>.log``; prints each one's exit and result line. -> 0 when every run printed its result."""
    out = Path(of(__file__).OUT or ".")
    jobs, cur = [], []
    for a in [*argv, "+"]:
        if a != "+":
            cur.append(a)
        elif cur:
            jobs, cur = [*jobs, cur], []
    procs = []
    for i, job in enumerate(jobs):
        log = out / f"guard-{i}.log"
        print(f"guard-node: run {i}: {shlex.join(job)} > {log}", flush=True)
        procs.append((log, subprocess.Popen([str(exe), *job], stdout=open(log, "w"), stderr=subprocess.STDOUT)))
    rc = 0
    for i, (log, p) in enumerate(procs):
        code = p.wait()
        last = [ln for ln in log.read_text(errors="replace").splitlines() if ln.startswith("GUARD-RESULT ")]
        print(f"guard-node: run {i} exited {code}: {last[-1] if last else 'no result'}", flush=True)
        rc = rc or (0 if last else 1)
    return rc


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
            exe = runner(tree, lake) if rc == 0 else None
            if exe is None:
                rc = rc or 1
            elif argv[:1] == ["--pytest"]:
                env = {**os.environ, "VERITY_TWO_STAGE_GUARD": str(exe)}
                uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
                rc = subprocess.run([uv, "run", "--no-project", "--with", "pytest", "python", "-m", "pytest", "-q",
                                     str(HERE / "test_guard.py"), *argv[1:]], cwd=ROOT, env=env).returncode
            else:
                rc = together(exe, argv)
            (tree / LC.DIRTY).unlink(missing_ok=True)
            return rc
    finally:
        if slot is not None:
            os.close(slot)
            os.environ.pop(LC.LEAN_SLOT.HELD, None)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
