"""benchmarks/pouw/pearl_c_vllm/served_step.py's arm: it runs only declared schemes, by name, and names any other."""

from __future__ import annotations

from conftest import load


def test_a_served_arm_runs_only_declared_schemes_by_name_and_names_any_other():
    from verity.protocols.run import schemes

    SS = load("pearl_c_vllm/served_step.py")
    assert SS.undeclared(["pearl-plus"]) is None
    for other in ("pouw:pearl-c-nvfp4-v0", "pouw:pearl-c-sm120-v1-h3", "pouw:pearl-c-nope", schemes.get("pearl-plus").option):
        assert SS.undeclared(["pearl-plus", other]).startswith(f"{other} is no declared scheme's name (")
        assert "pearl-plus" in SS.undeclared([other]).partition("(")[2]
