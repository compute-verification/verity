"""One flag name per concept across the workspace's command lines, so an agent who learned one CLI can use the next.

The CLIs (`research`, `cluster`, `ops`, `circuit-check`, `verity`, `verity-bench`, `verity-vllm`) grew separately, and the
same idea picked up different flags: the machine a command acts on was `--on`, `--node` or `--machine`; the handle acting was
`--as`, `--by` or `--handle`, and `--by` also meant a filter or a time bucket. The canonical names:

- `--on MACHINE`: the machine a command acts on or runs on;
- `--as HANDLE`: the handle acting, signing or posting;
- `--json`: machine-readable output on stdout.

A flag named here as a synonym fails the test wherever it is newly defined. KNOWN lists where each was already defined when
this test landed; it only shrinks. An entry goes when its owner renames the flag (keeping the old spelling as a hidden alias
that says which flag to use), and a stale entry fails too, so the list can't hide a fixed file.
"""
from __future__ import annotations

import ast
import collections
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: the packages whose argparse definitions are the CLIs
CLI_ROOTS = (
    "tools/research/src",
    "infra/cluster/src",
    "infra/ops",
    "benchmarks/harness/verity_bench",
    "integrations/vllm/verity_vllm",
    "tools/verity/circuit_check/src",
    "tools/verity/cli",
)

#: a non-canonical flag -> what to use instead
SYNONYMS = {
    "--node": "--on (the machine a command acts on)",
    "--machine": "--on (the machine a command acts on)",
    "--handle": "--as (the handle acting)",
    "--by": "--as for the handle acting, or a name that says what it filters (e.g. --labeled-by, --bucket)",
}

#: (file, flag): how many definitions existed when this test landed (14 Oct 2026); shrink it, never grow it
KNOWN = {
    ("infra/cluster/src/cluster/cli.py", "--by"): 1,
    ("infra/ops/verity_ops/deploy.py", "--by"): 2,
    ("infra/ops/verity_ops/keeper.py", "--by"): 1,
    ("tools/research/src/research/approaches.py", "--by"): 1,
    ("tools/research/src/research/cli.py", "--by"): 1,
    ("tools/research/src/research/monitors.py", "--by"): 2,
    ("tools/research/src/research/monitors.py", "--machine"): 2,
    ("tools/research/src/research/notes_archive.py", "--by"): 1,
    ("tools/research/src/research/pods/nebius/dispatch.py", "--by"): 1,
    ("tools/research/src/research/pods/nebius/keeper.py", "--by"): 1,
    ("tools/research/src/research/pods/nebius/lean_sandbox.py", "--by"): 1,
    ("tools/research/src/research/pods/nebius/lockout/lockout.py", "--node"): 1,
    ("tools/research/src/research/pods/nebius/n1_alerts.py", "--node"): 1,
    ("tools/research/src/research/pods/nebius/resource_probe.py", "--node"): 1,
    ("tools/research/src/research/pods/nebius/write_probe.py", "--handle"): 1,
    ("tools/research/src/research/pods/nebius/write_probe.py", "--node"): 1,
    ("tools/research/src/research/pods/part.py", "--by"): 1,
    ("tools/research/src/research/queue.py", "--by"): 1,
    ("tools/research/src/research/retention.py", "--by"): 2,
    ("tools/research/src/research/store/cli.py", "--by"): 2,
}


def _definitions() -> collections.Counter:
    listed = subprocess.run(["git", "ls-files", "--", *(f"{r}/**.py" for r in CLI_ROOTS)], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.split()
    found: collections.Counter = collections.Counter()
    for rel in listed:
        if "/tests/" in rel:
            continue
        for node in ast.walk(ast.parse((ROOT / rel).read_text(), rel)):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and arg.value in SYNONYMS:
                        found[(rel, arg.value)] += 1
    return found


def test_no_new_synonym_flags():
    new = {key: n for key, n in _definitions().items() if n > KNOWN.get(key, 0)}
    assert not new, "\n".join(
        f"{rel} defines {flag} ({n}x; known {KNOWN.get((rel, flag), 0)}): use {SYNONYMS[flag]} instead"
        for (rel, flag), n in sorted(new.items()))


def test_known_list_has_no_stale_entries():
    found = _definitions()
    stale = {key: n for key, n in KNOWN.items() if found.get(key, 0) < n}
    assert not stale, "\n".join(
        f"KNOWN lists {rel} {flag} {n}x but only {found.get((rel, flag), 0)} remain: lower or delete its entry "
        "in tests/test_cli_conventions.py" for (rel, flag), n in sorted(stale.items()))
