"""`kept_bootstrap.sh` keeps pod_bootstrap.sh's environment under a key of what it is built from: built with no card and
then its taps under the lease's prefix, reused while the key matches, built anew when it doesn't. Run on a tree whose
pod_bootstrap.sh is a stand-in that records each call; the wheel, nvidia-smi and nvcc are stand-ins too."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OPS = "adapters/vllm/verity_vllm/ops"
TAPS = ("pod_fa2_tap.sh", "pod_fa3_tap.sh", "pod_norm_tap.sh", "pod_router_tap.sh", "cuda_headers.sh")


def _exe(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)


def _tree(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    tree, bin_, boot = tmp_path / "tree", tmp_path / "bin", tmp_path / "boot"
    (tree / "benchmarks/one_stage").mkdir(parents=True)
    shutil.copy(ROOT / "benchmarks/one_stage/kept_bootstrap.sh", tree / "benchmarks/one_stage")
    for d in ("adapters/vllm/manifests", "adapters/vllm/verity_vllm/acquire"):
        (tree / d).parent.mkdir(parents=True, exist_ok=True)
        (tree / d).symlink_to(ROOT / d)
    for f in TAPS:
        (tree / OPS).mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / OPS / f, tree / OPS / f)
    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("vllm-0.dist-info/METADATA", "Name: vllm\nRequires-Dist: torch==2.9.0\nRequires-Dist: numpy\n")
    (boot / "wheels").mkdir(parents=True)
    (boot / "wheels/vllm-0.whl").write_bytes(wheel.getvalue())
    _exe(tree / OPS / "pod_bootstrap.sh",
         f'W="vllm-0.whl"\nWSHA={hashlib.sha256(wheel.getvalue()).hexdigest()}\n'
         '# curl -fL -o x "https://wheels.vllm.ai/0/vllm-0.whl"\n'
         f'echo "$WORK $* ${{LEASED:-unleased}}" >> {tmp_path}/calls\n'
         'mkdir -p "$WORK/venv312/bin" && printf \'#!/bin/sh\\nexit 0\\n\' > "$WORK/venv312/bin/python" && chmod +x "$WORK/venv312/bin/python"')
    _exe(bin_ / "nvidia-smi", f"cat {tmp_path}/arch")
    _exe(tmp_path / "cuda/bin/nvcc", "echo 'Cuda compilation tools, release 12.9, V12.9.86'; echo 'Build cuda_12.9'")
    _exe(bin_ / "lease", 'LEASED=leased exec "$@"')
    (bin_ / "python3").symlink_to(sys.executable)
    (tmp_path / "arch").write_text("9.0\n")
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path), "BOOT_ROOT": str(boot),
           "CUDA_HOME": str(tmp_path / "cuda")}
    return tree, env


def _boot(tree: Path, env: dict[str, str], out: Path) -> tuple[subprocess.CompletedProcess, Path]:
    done = subprocess.run(["bash", "benchmarks/one_stage/kept_bootstrap.sh", "0", "LLAMA32_1B_INT8", str(out), "lease"],
                          cwd=tree, env=env, capture_output=True, text=True)
    return done, Path(done.stdout.splitlines()[-1]) if done.returncode == 0 else Path()


def test_built_without_a_card_then_its_taps_under_the_lease_and_reused_while_its_key_matches(tmp_path):
    tree, env = _tree(tmp_path)
    done, d = _boot(tree, env, tmp_path / "out1")
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "calls").read_text().splitlines() == [
        f"{d} --cpu --cases LLAMA32_1B_INT8 --out {tmp_path / 'out1/cpu'} unleased",
        f"{d} --gpu --cases LLAMA32_1B_INT8 --out {tmp_path / 'out1/gpu'} leased"]
    key = (d / "key.txt").read_text()
    assert d.parent == tmp_path / "boot" and (d / "ready").is_file() and (tmp_path / "out1/key.txt").read_text() == key
    assert " torch==2.9.0\n" in key and "\narch 9.0\n" in key and f"\ncuda {tmp_path / 'cuda'} Cuda compilation tools" in key
    for f in (*(f"verity_vllm/ops/{t}" for t in TAPS), "verity_vllm/acquire/norm_tap_src/", "verity_vllm/acquire/fa3_tap_src/",
              "manifests/checkpoints.json"):
        assert f"  {f}" in key, f

    done, again = _boot(tree, env, tmp_path / "out2")
    assert done.returncode == 0 and again == d and "reusing" in done.stdout
    assert len((tmp_path / "calls").read_text().splitlines()) == 2


def test_another_card_architecture_is_another_key_and_a_new_build(tmp_path):
    tree, env = _tree(tmp_path)
    _, d = _boot(tree, env, tmp_path / "out1")
    (tmp_path / "arch").write_text("12.0\n")
    done, other = _boot(tree, env, tmp_path / "out2")
    assert done.returncode == 0 and other != d and other.parent == d.parent and (d / "ready").is_file()
    assert [ln.split()[0] for ln in (tmp_path / "calls").read_text().splitlines()] == [str(d), str(d), str(other), str(other)]


def test_an_unfinished_directory_is_built_again(tmp_path):
    tree, env = _tree(tmp_path)
    _, d = _boot(tree, env, tmp_path / "out1")
    (d / "ready").unlink()
    done, again = _boot(tree, env, tmp_path / "out2")
    assert done.returncode == 0 and again == d and "building" in done.stdout
    assert len((tmp_path / "calls").read_text().splitlines()) == 4
