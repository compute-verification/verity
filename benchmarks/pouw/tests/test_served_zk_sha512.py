"""The served -h3 window's own values are SHA-512 (core's `derive` or SHA-512 itself), each pinned: the statement's salt key
and tree bindings, the pass's name in the window's context, and the committer's choice of calls.  The window's modules call
SHA-256 only to name a cubin in provenance no check compares, and BLAKE3 nowhere."""
from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

from verity.primitives.randomness import derive

SERVED_ZK = Path(__file__).resolve().parents[1] / "served_zk"
sys.path.insert(0, str(SERVED_ZK))
import rows_bench as RB  # noqa: E402
import served_commit as SC  # noqa: E402
import statement as ST  # noqa: E402

#: (module, the SHA-256 call) left as they are: a cubin's file identity, recorded as metadata only
PROVENANCE = {("statement.py", "hashlib.sha256(raw)"), ("rows_bench.py", "hashlib.sha256(image)"),
              ("rows_bench.py", "hashlib.sha256(rimage)")}


def test_the_statements_salt_key_is_derived_on_sha512():
    salt = bytes(range(32))
    assert ST.salt_key(salt) == derive(salt, "verity/pouw/served-zk-statement-salt/v1", {}).value
    assert ST.salt_key(salt).hex() == "892e7f399cb48e991c3ccdfacca74500cfab3bc838d2d23758c89de42837ab72"


def test_the_statements_tree_bindings_are_sha512():
    b = RB.binding(ST.PREFIX, "x/k4096/m32")
    assert b == hashlib.sha512(b"verity/pouw/served-zk/statement/v1/x/k4096/m32").digest()
    assert b.hex().startswith("69c7e8352772e57ad10d9be2f6f41d6ae5ec5fe3c8a8ae3b8b775d6415b7d0e3")


def test_a_pass_is_named_by_its_manifests_sha512():
    assert SC.manifest_name(b"{}") == hashlib.sha512(b"{}").hexdigest()


def test_the_calls_are_a_subset_under_a_derived_key():
    calls = [{"index": i} for i in (9, 2, 40, 7, 13, 5)]
    among = sorted(calls, key=lambda u: u["index"])
    picked = derive(bytes(32), SC.CALLS_DOMAIN, {}).subset(3, 6, "calls")
    assert SC.choose_calls(bytes(32), calls, 3) == [among[i] for i in picked]
    assert [u["index"] for u in SC.choose_calls(bytes(32), calls, 3)] == [5, 7, 40]
    assert SC.choose_calls(bytes(32), calls, 9) == among


def test_the_windows_modules_hash_with_sha256_only_for_provenance_and_never_blake3():
    """rows_bench's `main` times -h2's BLAKE3 kernels beside the statement's as a microbenchmark; the statement reads only
    its `Commit` and `constants`, so BLAKE3 is checked absent from the window's and the statement's own modules."""
    found = set()
    for name in ("served_commit.py", "statement.py", "rows_bench.py"):
        tree = ast.parse((SERVED_ZK / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "sha256":
                found.add((name, ast.unparse(node)))
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                assert not any("blake3" in n for n in [getattr(node, "module", None) or "", *(a.name for a in node.names)]), name
        if name != "rows_bench.py":
            assert not any(w in ast.unparse(tree).lower() for w in ("blake3", "b3s")), name
    assert found == PROVENANCE
