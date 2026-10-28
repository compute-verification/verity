"""the not-executed placement of the layouts-only MANIFEST COVERAGE runs at B=1.

commit_delta.main() places the population over the executed prefix of record in three sites: the binding-map coverage (alpha), the
layouts-only manifest coverage (beta) and the sampled replay (gamma).  Gated `if len(reqs) > 1 and ...` (B >= 2 only), (beta)
would require a B=1 EOS-stopped request's CAP-shape manifest in full and fail COMMIT at MANIFEST COVERAGE on exactly the
never-executed steps, which (alpha) and (gamma) classify.  This test reads the SOURCE (torch-free, the class of test_commit_delta_cli.py) and asserts that the `if`
whose body calls `restrict_manifest_identities` tests the manifest only -- never the request count -- so one request maps to r0 and the
same restrict_* run as at B >= 2.
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "verity_vllm" / "pipeline" / "commit.py"


def _placement_ifs() -> list[ast.If]:
    tree = ast.parse(SRC.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    out: list[ast.If] = []
    for node in ast.walk(main):
        if not isinstance(node, ast.If):
            continue
        calls = [c for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "restrict_manifest_identities"]
        if calls and not any(isinstance(inner, ast.If) and inner is not node and any(
                isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "restrict_manifest_identities" for c in ast.walk(inner))
                             for inner in ast.walk(node)):
            out.append(node)
    return out


def _mentions_len_reqs(test: ast.expr) -> bool:
    for n in ast.walk(test):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "len" and n.args \
                and isinstance(n.args[0], ast.Name) and n.args[0].id == "reqs":
            return True
    return False


def test_the_manifest_coverage_placement_is_not_gated_on_the_request_count_f_r17b_42_ii():
    ifs = _placement_ifs()
    assert ifs, "the layouts-only manifest coverage no longer places the requirement over the executed prefix (restrict_manifest_identities gone)"
    for node in ifs:
        assert not _mentions_len_reqs(node.test), (
            "the manifest-coverage placement is gated on len(reqs) -- at B=1 the cap-shape manifest would be required in full "
            f"(line {node.lineno}: {ast.unparse(node.test)[:120]})")
        src = ast.unparse(node.test)
        assert "required_manifest" in src and "identities" in src, f"the placement's own condition (the manifest with identities) is gone: {src[:120]}"


def test_the_gate_the_sync_twin_failed_on_would_be_caught():
    """The test's own negative: the pre-fix condition is what the assertion rejects."""
    pre = ast.parse('if len(reqs) > 1 and isinstance(required_manifest, dict) and required_manifest.get("identities"):\n    pass').body[0]
    post = ast.parse('if isinstance(required_manifest, dict) and required_manifest.get("identities"):\n    pass').body[0]
    assert _mentions_len_reqs(pre.test) and not _mentions_len_reqs(post.test)
