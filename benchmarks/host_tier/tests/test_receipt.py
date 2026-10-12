"""The host CPUs' receipts (`receipt.cpp`) against their definitions in `reference.py`: the sketches and the leaf for both
key widths, with SHA-256 portable and on the SHA extensions, SHA-256 itself across its padding cases, and the gate
refusing a wrong sketch. The GPU's half is checked by loop.cu's gate on the node. Needs g++ and AVX-512 VNNI."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import reference  # noqa: E402


def _flags() -> set[str]:
    try:
        lines = Path("/proc/cpuinfo").read_text().splitlines()
    except OSError:
        return set()
    return next((set(l.split(":", 1)[1].split()) for l in lines if l.startswith("flags")), set())


@pytest.fixture(scope="module")
def gate_cpu(tmp_path_factory):
    flags = _flags()
    if not shutil.which("g++") or not {"avx512f", "avx512bw", "avx512_vnni", "sha_ni"} <= flags:
        pytest.skip("needs g++ and a CPU with AVX-512 VNNI and the SHA extensions")
    exe = tmp_path_factory.mktemp("gate") / "gate_cpu"
    subprocess.run(["g++", "-O2", "-march=native", "-I", str(HERE), str(HERE / "tests" / "gate_cpu.cpp"),
                    str(HERE / "receipt.cpp"), "-o", str(exe)], check=True)
    return exe


@pytest.mark.parametrize("ks", [4, 16])
@pytest.mark.parametrize("sha_ni", [0, 1])
def test_cpu_receipts_match_the_reference(gate_cpu, tmp_path, ks, sha_ni):
    gate = tmp_path / "gate.json"
    gate.write_text(subprocess.run([str(gate_cpu), str(ks), str(sha_ni)], check=True, capture_output=True, text=True).stdout)
    assert reference.main(str(gate)) == 0


@pytest.mark.parametrize("sha_ni", [0, 1])
def test_sha256_matches_hashlib(gate_cpu, sha_ni):
    out = subprocess.run([str(gate_cpu), "sha", str(sha_ni)], check=True, capture_output=True, text=True).stdout
    for line in out.splitlines():
        n, digest = line.split()
        assert digest == hashlib.sha256(bytes((7 * i + 1) & 0xFF for i in range(int(n)))).hexdigest(), n


def test_the_gate_refuses_a_wrong_sketch(gate_cpu, tmp_path):
    g = json.loads(subprocess.run([str(gate_cpu), "4", "0"], check=True, capture_output=True, text=True).stdout)
    g["cpu_sketches"][5] += 1
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps(g))
    assert reference.main(str(gate)) == 3
