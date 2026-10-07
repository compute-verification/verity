"""verity_vllm/ops/pod_bootstrap.sh: the mode switch (gpu / cpu) and the readiness record it writes.

The script is infra (venv, wheels, checkpoints, native taps) and cannot run here; what CAN be checked mechanically is that it parses,
that `--dry-run` resolves the mode the way the header promises (nvidia-smi on PATH -> gpu; absent -> cpu; `--cpu` forces), that a
dry run touches nothing, and that the step-5 readiness heredoc is valid Python that, in cpu mode, writes a `verity-vllm/readiness/v1`
record with `mode: cpu`, `cuda_available`, the CPU checks and no GPU check.  (Before 2026-09-23 a GPU-less pod died at step 4 with
BOOTSTRAP_FAIL_CUDA before readiness.json existed.)"""

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
    m = re.search(r'\$PY - "\$ROOT" "\$CORE" "\$OUT/readiness\.json" "\$HG_SO" "\$CC" "\$RESEARCH" "\$MODE" "\$CORE/experimental" <<\'EOF\'.*?\n(.*?)\nEOF\n', text, re.S)
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
    experimental = TREE / "experimental"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(p) for p in (root, core, research, TREE / "catalog", TREE / "kernels", experimental)),
               https_proxy="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9")   # the network probe fails fast, recorded as an error
    r = subprocess.run([sys.executable, str(src), str(root), str(core), str(out), "", "", str(research), "cpu", str(experimental)],
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
                 "research_telemetry_from_tree", "sampled_proofs_from_tree", "sampled_proofs_via_pth", "distributions_from_tree", "numpy_import",
                 "no_old_tree_on_path"):
        assert name in checks, name
    for name in ("verity_from_tree", "verity_vllm_from_tree", "research_telemetry_from_tree", "sampled_proofs_from_tree", "distributions_from_tree",
                 "numpy_import"):
        assert checks[name]["ok"], (name, checks[name])
    assert set(checks["distributions_from_tree"]["value"]) == {"catalog", "kernels", "experimental"}
    assert rec["ok"] == all(c["ok"] for c in checks.values())
    assert (r.returncode == 0) == rec["ok"]
    meas = rec["measurements"]
    assert set(meas) == {"network", "disk", "memory", "cpu"}
    assert meas["cpu"]["nproc"] == os.cpu_count()
    assert "total_bytes" in meas["memory"] and "cgroup_max_bytes" in meas["memory"]
    assert rec["identities"]["torch"] == checks["torch_cpu"]["value"] and rec["identities"]["native"] == {}
    assert rec["identities"]["tree"] == {"integration_root": str(root), "verity_core": str(core), "research": str(research)}


def test_bootstrap_writes_the_experimental_pth_and_readiness_checks_it() -> None:
    """Gate and row scripts set their own PYTHONPATH: the venv's site-packages carries a .pth for verity/experimental/, and readiness
    imports sampled proofs' plans (verity.experimental.protocols.sampled_proofs.plan) with verity/experimental/ removed from PYTHONPATH."""
    text = SCRIPT.read_text()
    assert 'echo "$CORE/$G" > "$SITE/verity-$G.pth"' in text
    assert text.index('"$SITE/verity-$G.pth"') < text.index("# ---- 5. readiness")
    assert 'check("sampled_proofs_via_pth"' in _readiness_source()


def test_the_pth_check_passes_on_a_fresh_venv_and_fails_without_the_pth(tmp_path: Path) -> None:
    """A fresh Python 3.12 venv as bootstrap leaves it: verity-experimental.pth in site-packages and a PYTHONPATH (core, research,
    the integration) that omits verity/experimental/.  The readiness check's function imports sampled proofs' plans from the tree
    through the .pth -- its dependency verity from PYTHONPATH -- and fails once the .pth is removed."""
    uv = shutil.which("uv") or shutil.which(os.path.expanduser("~/.local/bin/uv"))
    if not uv:
        pytest.skip("uv not installed")
    venv = tmp_path / "venv"
    if subprocess.run([uv, "venv", "-q", "-p", "3.12", str(venv)], capture_output=True).returncode != 0:
        pytest.skip("no Python 3.12 for uv venv")
    py, site = venv / "bin" / "python", venv / "lib" / "python3.12" / "site-packages"
    experimental, core, research = TREE / "experimental", TREE, TREE / "tools" / "research" / "src"
    src = _readiness_source()
    fn = src[src.index("def experimental_without_pythonpath():"):src.index('check("sampled_proofs_via_pth"')]
    probe = tmp_path / "probe.py"
    probe.write_text("import os, subprocess, sys\nexperimental = sys.argv[1]\n" + fn + "print(experimental_without_pythonpath())\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(p) for p in (VLLM, core, research, experimental)))
    (site / "verity-experimental.pth").write_text(f"{experimental}\n")
    ok = subprocess.run([str(py), str(probe), str(experimental)], capture_output=True, text=True, env=env)
    assert ok.returncode == 0 and ok.stdout.strip().startswith(str(experimental.resolve())), ok.stdout + ok.stderr
    (site / "verity-experimental.pth").unlink()
    bad = subprocess.run([str(py), str(probe), str(experimental)], capture_output=True, text=True, env=env)
    assert bad.returncode != 0 and "without experimental on PYTHONPATH" in bad.stderr


def test_bootstrap_writes_a_pth_for_each_distribution(tmp_path: Path) -> None:
    """A row's stages set PYTHONPATH to the tree, the core and research, so the catalog, the kernels and experimental (each a
    distribution under its grouping, imported as verity.ml.circuits, verity.ml.kernels, verity.experimental) reach them through a .pth."""
    text = SCRIPT.read_text()
    start = text.index('for G in $DISTS; do\n')
    loop = text[start:text.index("\ndone\n", start) + len("\ndone")]
    assert start < text.index("# ---- 5. readiness")
    core = TREE.resolve()
    script = f'fail() {{ echo "FAIL $*"; exit 3; }}\nCORE={core}; DISTS="catalog kernels experimental"; SITE={tmp_path}\n{loop}\n'
    r = subprocess.run([BASH, "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert {p.name: p.read_text() for p in tmp_path.glob("*.pth")} == {
        "verity-catalog.pth": f"{core / 'catalog'}\n", "verity-kernels.pth": f"{core / 'kernels'}\n",
        "verity-experimental.pth": f"{core / 'experimental'}\n"}


def test_gpu_mode_record_shape_is_unchanged() -> None:
    """The gpu-mode record carries no `mode` key and the same check names as before 2026-09-23 (readers treat 'no mode' as gpu)."""
    src = _readiness_source()
    assert 'if not gpu:\n    r["mode"] = mode' in src
    for name in ("torch_cuda", "hidden_gpu_so", "fa2_tap_so", "fa3_tap_so"):
        assert f'check("{name}"' in src, name
