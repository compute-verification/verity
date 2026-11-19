"""Every `lake build` or `lake exe` a Python file of the tree starts holds the tree's one Lake build lock,
`verity/circuits/difftest/.lake/harness-build.lock` (`verity_difftest.BUILD_LOCK`), which check's lean-build step takes too
(`tools/check/suites.py`'s `LAKE_BUILD`). Lake builds a package's path requires in each one's own `.lake`, so two builds of
two packages at once (two suites' fixtures in one tree) write one dependency's files together and fail on each other's
half-written `.olean` or `.c.o`. `lake exe` builds a stale target before running it; `lake env` builds nothing and writes no
`.lake`, so it needs no lock.

The scan: a list or tuple whose first element is ``"lake"`` and that names ``build`` or ``exe`` (or passes on ``*args``), and
a string with ``lake build`` or ``lake exe`` in a shell command's position, each outside a ``with`` on ``BUILD_LOCK`` (or a
string naming the lock). What may stay is listed in ``ALLOWED`` (a path, or a directory ending in ``/``) with its reason;
the list only shrinks.
"""
from __future__ import annotations

import ast
import re

from test_repository import ROOT, _tracked

HELPERS = ("`verity_difftest.build(package, exe)`, or `verity.protocols.proofs.spot.verifier.one_stage.verifier."
           "executable(exe, package)` in a module that imports only `verity`")
LOCK = "harness-build.lock"
#: what a file holding either form contains, which most files don't: only those are parsed
MENTIONS = ('"lake",', "'lake',", "&& lake ", "; lake ", "exec lake ", "|| lake ")
SHELL = re.compile(r"(?:&&|;|\bexec|\|\|)\s*lake\s+(?:-d\s+\S+\s+)?(?:build|exe)\b")
FLOCK_SUITE = ("Flock's own suite: it builds Flock's verifier's `defaultTargets` (or a scratch package requiring it), which "
               "check's lean-build step builds first under the lock, so in a check these builds write nothing")
ALLOWED: dict[str, str] = {
    "tools/verity/lean/": "the Lean check, which builds in scratch trees of its own under its own tree locks",
    "tests/tools/": "the tools' tests: command lines the tools classify, stub `lake`s and scratch copies of fixtures",
    "tools/check/pins.py": "a pin's regeneration command, run by hand (`pins.py --write`), never by a suite",
    "adapters/vllm/tools/served_run/local_gate.py": "builds in the tree it extracts for the gate, which nothing else builds in",
    **{f"tests/verity/protocols/proofs/flock/prover/{f}": FLOCK_SUITE for f in (
        "test_flock_rows_vectors.py", "test_inner_registered.py", "test_lean_coin_open.py", "test_lean_coin_tree.py",
        "test_lean_forged_proofs.py", "test_lean_hello.py", "test_lean_hidden_outputs.py", "test_lean_identifier.py",
        "test_lean_ks_form.py", "test_lean_out_group_slot.py", "test_lean_plain_hidden_tables.py",
        "test_lean_plain_leaves.py", "test_lean_plain_tree.py", "test_lean_public_inputs.py", "test_lean_rec_coins.py",
        "test_lean_record_checks.py", "test_lean_region_words.py", "test_lean_registered_bit_rows.py",
        "test_lean_registered_reads.py", "test_lean_served_registered.py", "test_lean_session_tables.py",
        "test_lean_shared_reads.py", "test_lean_typed_reads.py", "test_lean_typed_template.py",
        "test_lean_verify_members.py", "test_lean_width_refusals.py", "test_lean_witness.py", "test_lean_zk_mask.py",
        "test_lean_zk.py", "test_pouw_rows.py", "test_read_tmpl.py", "test_read_tmpl_stager.py", "test_rec_algebra.py",
        "test_rec_full_draw.py", "test_rec_links.py", "test_rec_opening_unit.py", "test_stream_zk_server.py",
        "test_universal_same_shape.py", "test_wired_links.py", "test_zk_server_agreement.py", "test_zk_server_contract.py",
        "test_zk_server_frame.py")},
}


def unlocked(src: str) -> list[int]:
    """The lines of `src` that start `lake build` or `lake exe` outside a ``with`` on the lock (the module docstring)."""
    tree = ast.parse(src)
    held, docs = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.With) and any("BUILD_LOCK" in ast.unparse(i.context_expr) or LOCK in ast.unparse(i.context_expr)
                                           for i in n.items):
            held.update(id(m) for m in ast.walk(n))
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and ast.get_docstring(n, False):
            docs.add(id(n.body[0].value))
    lines = []
    for n in ast.walk(tree):
        if id(n) in held or id(n) in docs:
            continue
        if isinstance(n, (ast.List, ast.Tuple)) and n.elts and isinstance(n.elts[0], ast.Constant) and n.elts[0].value == "lake":
            words = [e.value for e in n.elts[1:4] if isinstance(e, ast.Constant)]
            if {"build", "exe"} & set(words) or (len(n.elts) > 1 and isinstance(n.elts[1], ast.Starred)):
                lines.append(n.lineno)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and SHELL.search(n.value) and LOCK not in n.value:
            lines.append(n.lineno)
    return sorted(lines)


def allowed(path: str) -> str | None:
    return next((k for k in ALLOWED if path == k or (k.endswith("/") and path.startswith(k))), None)


def test_every_lake_build_a_python_file_starts_holds_the_trees_build_lock() -> None:
    offenders, used = [], set()
    for path in _tracked():
        if not path.endswith(".py"):
            continue
        src = (ROOT / path).read_text(encoding="utf-8")
        if not any(m in src for m in MENTIONS) or not (lines := unlocked(src)):
            continue
        if key := allowed(path):
            used.add(key)
        else:
            offenders += [f"{path}:{line}" for line in lines]
    assert not offenders, (f"`lake build` or `lake exe` outside the tree's Lake build lock ({LOCK}): build through {HELPERS}, "
                           "or hold `BUILD_LOCK` around it (`with open(BUILD_LOCK, \"a\") as lock: fcntl.flock(lock, "
                           "fcntl.LOCK_EX)`):\n" + "\n".join(offenders))
    assert not set(ALLOWED) - used, f"ALLOWED entries that match nothing any more: {sorted(set(ALLOWED) - used)}"


def test_the_scan_sees_each_form() -> None:
    src = '''
"""(cd X && lake build Y): a docstring, read by no shell"""
import subprocess
from verity_difftest import BUILD_LOCK

def f(pkg, args):
    subprocess.run(["lake", "build", "a"], cwd=pkg)
    subprocess.run(("lake", "exe", "b"))
    subprocess.run(["lake", "-d", str(pkg), "build"])
    subprocess.run(["lake", *args])
    subprocess.run(["bash", "-c", "cd p && lake build c"])
    subprocess.run(["lake", "env", "lean", "x.lean"])
    raise RuntimeError("lake build a failed")
    with open(BUILD_LOCK, "a") as lock:
        subprocess.run(["lake", "build", "d"])
    subprocess.run(["bash", "-c", "flock v/.lake/harness-build.lock lake build; cd q && lake build e"])
'''
    assert unlocked(src) == [7, 8, 9, 10, 11]
