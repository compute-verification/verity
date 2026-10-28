"""verity_vllm/ops/pod_bootstrap.sh: the mode switch (gpu / cpu), the staging marker step 3 writes, and the readiness record.

The script is infra (venv, wheels, checkpoints, native taps) and cannot run here; what CAN be checked mechanically is that it parses,
that `--dry-run` resolves the mode the way the header promises (nvidia-smi on PATH -> gpu; absent -> cpu; `--cpu` forces), that a
dry run touches nothing, that step 3 marks a checkpoint staged only when it downloaded and checked it, and that the step-5 readiness
heredoc is valid Python that, in cpu mode, writes a `verity-vllm/readiness/v1` record with `mode: cpu`, `cuda_available`, the CPU
checks and no GPU check, so a GPU-less pod gets its readiness.json instead of dying at step 4 with BOOTSTRAP_FAIL_CUDA."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

VLLM = Path(__file__).resolve().parents[2]
SCRIPT = VLLM / "verity_vllm" / "ops" / "pod_bootstrap.sh"
TREE = VLLM.parents[1]
BASH = shutil.which("bash") or "/bin/bash"


def _bin(tmp_path: Path, *, nvidia_smi: bool) -> Path:
    """A PATH with only what the argument-parsing prologue needs (`dirname`), plus an optional fake `nvidia-smi`."""
    b = tmp_path / "bin"
    b.mkdir()
    dirname = shutil.which("dirname")
    assert dirname, "dirname not found"
    os.symlink(dirname, b / "dirname")
    if nvidia_smi:
        (b / "nvidia-smi").write_text("#!/bin/sh\necho fake\n")
        (b / "nvidia-smi").chmod(0o755)
    return b


def _dry_run(tmp_path: Path, *args: str, nvidia_smi: bool) -> subprocess.CompletedProcess:
    env = {"PATH": str(_bin(tmp_path, nvidia_smi=nvidia_smi)), "WORK": str(tmp_path / "work"), "HOME": str(tmp_path)}
    return subprocess.run([BASH, str(SCRIPT), "--dry-run", "--out", str(tmp_path / "out"), *args], capture_output=True, text=True, env=env)


def test_script_parses() -> None:
    r = subprocess.run([BASH, "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_dry_run_auto_selects_gpu_when_nvidia_smi_present(tmp_path: Path) -> None:
    r = _dry_run(tmp_path, nvidia_smi=True)
    assert r.returncode == 0, r.stderr
    assert "mode=gpu" in r.stdout and "4 native taps (hidden_gpu" in r.stdout and r.stdout.rstrip().endswith("BOOTSTRAP-DRY-RUN")
    assert "no nvidia-smi" not in r.stdout


def test_dry_run_auto_selects_cpu_without_nvidia_smi(tmp_path: Path) -> None:
    r = _dry_run(tmp_path, nvidia_smi=False)
    assert r.returncode == 0, r.stderr
    assert "no nvidia-smi on PATH: cpu mode" in r.stdout and "mode=cpu" in r.stdout and "4 native taps SKIPPED (cpu mode)" in r.stdout


def test_cpu_flag_forces_cpu_mode_even_with_a_gpu(tmp_path: Path) -> None:
    r = _dry_run(tmp_path, "--cpu", "--cases", "B0,B1", nvidia_smi=True)
    assert r.returncode == 0, r.stderr
    assert "mode=cpu cases=B0,B1" in r.stdout and "SKIPPED (cpu mode)" in r.stdout


def test_gpu_flag_forces_gpu_mode_without_nvidia_smi(tmp_path: Path) -> None:
    r = _dry_run(tmp_path, "--gpu", nvidia_smi=False)
    assert r.returncode == 0, r.stderr
    assert "mode=gpu" in r.stdout and "no nvidia-smi" not in r.stdout


def test_dry_run_touches_nothing(tmp_path: Path) -> None:
    _dry_run(tmp_path, "--cpu", nvidia_smi=False)
    assert not (tmp_path / "out").exists() and not (tmp_path / "work").exists()


def test_unknown_argument_exits_2(tmp_path: Path) -> None:
    env = {"PATH": str(_bin(tmp_path, nvidia_smi=False)), "HOME": str(tmp_path)}
    r = subprocess.run([BASH, str(SCRIPT), "--tpu"], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "unknown arg --tpu" in r.stderr


def _readiness_source() -> str:
    """The step-5 Python heredoc, as the script feeds it to $PY."""
    text = SCRIPT.read_text()
    m = re.search(r'\$PY - "\$ROOT" "\$CORE" "\$OUT/readiness\.json" "\$HG_SO" "\$CC" "\$RESEARCH" "\$MODE" "\$CORE/verity/core" <<\'EOF\'.*?\n(.*?)\nEOF\n', text, re.S)
    assert m, "step-5 readiness heredoc not found (argument list or heredoc marker changed)"
    return m.group(1)


def test_readiness_heredoc_is_python() -> None:
    compile(_readiness_source(), "verity_vllm/ops/pod_bootstrap.sh#readiness", "exec")


def test_readiness_cpu_mode_record(tmp_path: Path) -> None:
    """Run the readiness Python in cpu mode against this tree.  Environment checks (vllm, Python 3.12) may fail here -- the record is
    written regardless (that is the invariant) and its shape is what is asserted; `ok` must equal the AND of the checks."""
    pytest.importorskip("torch")
    src = tmp_path / "readiness.py"
    src.write_text(_readiness_source())
    out = tmp_path / "readiness.json"
    root, core, research = VLLM, TREE, TREE / "tools" / "research" / "src"
    verifiers = TREE / "verity" / "core"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(p) for p in (root, core, research)),
               https_proxy="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9")   # the network probe fails fast, recorded as an error
    r = subprocess.run([sys.executable, str(src), str(root), str(core), str(out), "", "", str(research), "cpu", str(verifiers)],
                       capture_output=True, text=True, env=env)
    assert out.exists(), f"readiness.json not written\nstdout: {r.stdout}\nstderr: {r.stderr}"
    rec = json.loads(out.read_text())
    assert rec["schema"] == "verity-vllm/readiness/v1" and rec["mode"] == "cpu"
    assert rec["cuda_available"] is False or rec["cuda_available"] is True   # recorded from torch, not assumed
    checks = rec["checks"]
    assert "torch_cpu" in checks and checks["torch_cpu"]["ok"], checks["torch_cpu"]
    assert checks["torch_cpu"]["value"]["cuda_available"] == rec["cuda_available"]
    assert not {"torch_cuda", "hidden_gpu_so", "fa2_tap_so", "fa3_tap_so"} & set(checks), "GPU checks must not appear in cpu mode"
    for name in ("python_3_12", "vllm_import", "vllm_commit_d9105ea80", "verity_from_tree", "verity_vllm_from_tree",
                 "research_telemetry_from_tree", "sampled_proofs_from_tree", "distributions_from_tree", "numpy_import",
                 "no_old_tree_on_path"):
        assert name in checks, name
    for name in ("verity_from_tree", "verity_vllm_from_tree", "research_telemetry_from_tree", "sampled_proofs_from_tree", "distributions_from_tree",
                 "numpy_import"):
        assert checks[name]["ok"], (name, checks[name])
    assert set(checks["distributions_from_tree"]["value"]) == {"catalog", "kernels", "core"}
    assert rec["ok"] == all(c["ok"] for c in checks.values())
    assert (r.returncode == 0) == rec["ok"]
    meas = rec["measurements"]
    assert set(meas) == {"network", "disk", "memory", "cpu"}
    assert meas["cpu"]["nproc"] == os.cpu_count()
    assert "total_bytes" in meas["memory"] and "cgroup_max_bytes" in meas["memory"]
    assert rec["identities"]["torch"] == checks["torch_cpu"]["value"] and rec["identities"]["native"] == {}
    assert rec["identities"]["tree"] == {"integration_root": str(root), "verity_core": str(core), "research": str(research)}


def test_gpu_mode_record_shape_is_unchanged() -> None:
    """The gpu-mode record carries no `mode` key and the GPU check names (readers treat 'no mode' as gpu)."""
    src = _readiness_source()
    assert 'if not gpu:\n    r["mode"] = mode' in src
    for name in ("torch_cuda", "hidden_gpu_so", "fa2_tap_so", "fa3_tap_so"):
        assert f'check("{name}"' in src, name


HUB_STUB = """import os
from pathlib import Path
def snapshot_download(repo, revision, allow_patterns, cache_dir):
    d = Path(os.environ["STUB_LOCAL"]); d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{}"); (d / "model.safetensors").write_text(os.environ["STUB_BYTES"])
    return str(d)
"""


def test_step3_stages_a_checkpoint_it_downloads_and_checks_and_no_other(tmp_path: Path) -> None:
    """The step-3 heredoc writes the staging marker (repo, revision, UTC time, dir) that the step records with `verity-vllm records
    staged` only when this run downloaded the weights and every shard's sha256 checked."""
    m = re.search(r'\$PY - "\$CASE" manifests/checkpoints\.json "\$OUT/staged-\$CASE" <<\'EOF\'.*?\n(.*?)\nEOF\n', SCRIPT.read_text(), re.S)
    assert m, "step-3 checkpoint heredoc not found (argument list or heredoc marker changed)"
    (tmp_path / "step3.py").write_text(m.group(1))
    (tmp_path / "hub" / "huggingface_hub").mkdir(parents=True)
    (tmp_path / "hub" / "huggingface_hub" / "__init__.py").write_text(HUB_STUB)
    local, rev = tmp_path / "ck", "a" * 40
    (tmp_path / "ckpts.json").write_text(json.dumps({"checkpoints": [{"role": "T", "repo": "o/m", "revision": rev, "local_path": str(local),
        "safetensors_sha256": {"model.safetensors": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"}}]}))
    marker = tmp_path / "staged-T"

    def step3(stub_bytes: str) -> int:
        marker.unlink(missing_ok=True)
        env = dict(os.environ, PYTHONPATH=str(tmp_path / "hub"), HF_HOME=str(tmp_path / "hf"), STUB_LOCAL=str(local), STUB_BYTES=stub_bytes)
        return subprocess.run([sys.executable, str(tmp_path / "step3.py"), "T", str(tmp_path / "ckpts.json"), str(marker)],
                              capture_output=True, text=True, env=env).returncode

    assert step3("hello") == 0
    repo, revision, at, where = marker.read_text().split()
    assert (repo, revision, where) == ("o/m", rev, str(local)) and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", at)
    assert step3("hello") == 0 and not marker.exists(), "weights already on the host are not a staging"
    (local / "model.safetensors").unlink()
    assert step3("tampered") == 1 and not marker.exists(), "a shard whose sha256 fails is not staged"
