"""`proof_layers.sh` is the README's commands for the proof-layers table's three benchmarks, as one command: its runs
carry the settings and sampling designs the README's by-hand lines carry, its steps run in the README's order, it reads
only the settings it names, and it refuses before its first step, a missing setting by name. Read as text, and run only
as far as a refusal: nothing is built."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCRIPT = HERE.parent / "proof_layers.sh"
TEXT = SCRIPT.read_text()
README = (HERE.parent / "README.md").read_text()
SETTINGS = ["A5_CARD", "A5_CORES", "WORK", "GPU_LEASE_WHO"]


def by_hand() -> str:
    """The README's by-hand block of the proof-layers section: the code block after the one with the script's line."""
    section = README.split("## The three benchmarks of the proof-layers table", 1)[1].split("\n## ", 1)[0]
    blocks = re.findall(r"~~~sh\n(.*?)~~~", section, re.S)
    assert len(blocks) == 2 and "proof_layers.sh" in blocks[0], "the section holds the one line, then the by-hand block"
    return blocks[1]


def test_the_script_parses():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_its_runs_and_partitions_are_the_readmes():
    hand = by_hand()
    laws = dict(b.split("=", 1) for b in re.search(r"for b in (.*?); do", hand).group(1).split())
    runs = {m[1]: m[0] for m in re.findall(r"A5_COINS=challenger (A5_PLAIN=1 )?A5_PARTITION=\$WORK/partition-(\w+)\.json", hand)}
    assert list(laws) == list(runs) == ["flock", "zk", "sampling"]
    script_laws = {name: law for law, name in re.findall(r"\$\(part (\S+) (\w+)\)", TEXT)}
    script_runs = {name: plain for plain, name in re.findall(r'\$\(run "(A5_PLAIN=1 )?" (\w+)\)', TEXT)}
    assert script_laws == laws
    assert script_runs == runs
    assert "bash benchmarks/one_stage/a5_llama1b_h200.sh" in TEXT
    assert "pod_bootstrap.sh --cases LLAMA32_1B_INT8" in TEXT and "pod_bootstrap.sh --cases LLAMA32_1B_INT8" in hand
    assert "verity bench prove layers --flock" in TEXT and "--recursion" not in TEXT


def test_its_steps_run_in_the_readmes_order():
    steps = re.search(r"^STEPS=\((.*?)\)$", TEXT, re.M).group(1).split()
    assert steps == ["bootstrap", "partition-flock", "partition-zk", "partition-sampling", "flock", "zk", "sampling", "layers"]
    assert set(re.findall(r"^  \[([\w-]+)\]=", TEXT, re.M)) == set(steps)


def test_it_reads_only_the_settings_it_names():
    assigned = set(re.findall(r"(?:^|[\s;(])(?:local )?([A-Za-z_]\w*)=", TEXT)) | {"s", "t", "n", "f", "o", "k"}
    read = set(re.findall(r"\$\{?!?([A-Za-z_]\w*)", TEXT)) - (assigned - set(SETTINGS))
    # HOME and PATH are the shell's; RESEARCH_RUN_DIR is a research run's own directory, the results' default
    assert read - {"HOME", "PATH", "RESEARCH_RUN_DIR", "SECONDS", "PIPESTATUS"} == set(SETTINGS)
    header = TEXT.split("set -uo pipefail", 1)[0]
    assert all(f"#   {s} " in header for s in SETTINGS)


def test_every_refusal_comes_before_the_first_step():
    first_step = TEXT.index("trap finish EXIT")
    refusals = [m.start() for m in re.finditer(r"\brefuse \"", TEXT)]
    assert refusals and max(refusals) < first_step
    assert TEXT.count('rm -rf "$WORK"') == 1 and TEXT.index('rm -rf "$WORK"') > TEXT.index("finish() {")


@pytest.mark.parametrize("missing", ["A5_CARD", "A5_CORES", "WORK"])
def test_a_missing_setting_is_refused_by_name(missing, tmp_path):
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "A5_CARD": "0", "A5_CORES": "0", "WORK": str(tmp_path / "w")}
    del env[missing]
    run = subprocess.run(["bash", str(SCRIPT), "--dry-run"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert run.returncode == 2
    assert run.stderr.startswith(f"proof_layers.sh: no {missing}:")
    assert not (tmp_path / "w").exists()


def test_a_lease_without_its_owner_is_refused(tmp_path):
    (tmp_path / "gpu-lease").write_text("#!/bin/sh\nexit 9\n")
    (tmp_path / "gpu-lease").chmod(0o755)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "HOME": str(tmp_path), "A5_CARD": "0", "A5_CORES": "0", "WORK": str(tmp_path / "w")}
    run = subprocess.run(["bash", str(SCRIPT), "--dry-run"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert run.returncode == 2
    assert run.stderr.startswith("proof_layers.sh: no GPU_LEASE_WHO:")
