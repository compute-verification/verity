from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bench_arm
import stack


def test_shared_benchmark_arms_name_the_plain_control_and_require_isolation():
    assert (bench_arm.PLAIN.id, bench_arm.PLAIN.control, bench_arm.PLAIN.precision) == (
        "bandwidth-plain",
        None,
        "fp8",
    )
    assert (bench_arm.AUDIT.id, bench_arm.AUDIT.control, bench_arm.AUDIT.precision) == (
        "bandwidth-audit",
        "bandwidth-plain",
        "fp8",
    )
    assert not bench_arm.PLAIN.resident and not bench_arm.AUDIT.resident


def test_library_inputs_fail_closed_and_kernel_digest_binds_contents(tmp_path, monkeypatch):
    for variable in bench_arm._LIB_VARS.values():
        monkeypatch.delenv(variable, raising=False)
    with pytest.raises(RuntimeError, match="VERITY_VLLM_BENCH_WT_LIB"):
        bench_arm._libraries()

    first, second = tmp_path / "first.so", tmp_path / "second.so"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    got = bench_arm._digest([first, second])
    want = hashlib.sha256()
    for path in (first, second):
        name = path.name.encode()
        want.update(len(name).to_bytes(4, "big"))
        want.update(name)
        want.update(path.read_bytes())
    assert got == want.hexdigest()
    second.write_bytes(b"changed")
    assert bench_arm._digest([first, second]) != got


def test_plain_install_gate_does_not_require_encoded_receipt_tree():
    plain = {"gates": {"layers": 128, "configs_checked": 5160}}
    encoded = {"gates": {**plain["gates"], "step": {32: {"ok": True}}}}
    assert bench_arm._install_gate_passed(plain, encoded=False)
    assert not bench_arm._install_gate_passed(plain, encoded=True)
    assert bench_arm._install_gate_passed(encoded, encoded=True)


def test_plain_stack_counts_weight_read_passes_for_filler_share():
    plain = SimpleNamespace(encoded=False, nvec={8: 10}, passes=0)
    stack.Stack.begin_pass(plain, 8)
    assert plain.passes == 1
