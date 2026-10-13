from __future__ import annotations

import hashlib

import pytest

import bench_arm


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
