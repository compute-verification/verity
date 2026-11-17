"""`proof_layers.sh` runs the proof-layers table's three benchmarks as one command: its runs carry the settings and
sampling designs of the README's section on them, its steps run in that section's order, it reads only the settings it
names, and it refuses before its first step, a missing setting by name. Read as text, and run only as far as a refusal:
nothing is built. The suite keys without its Markdown, so the section's runs are written out here (`DESIGNS`, `PLAIN`)."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCRIPT = ROOT / "benchmarks" / "one_stage" / "proof_layers.sh"
TEXT = SCRIPT.read_text()
SETTINGS = ["A5_CARD", "A5_CORES", "WORK", "GPU_LEASE_WHO", "A5_STEPS"]
DESIGNS = {"flock": "stratified-rate:880/880", "zk": "$ZK_DESIGN", "sampling": "subset:256"}
PLAIN = {"flock": "A5_PLAIN=1 ", "zk": "", "sampling": ""}


def test_the_script_parses():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_its_runs_and_partitions_are_the_readmes():
    script_designs = {name: design for design, name in re.findall(r"\$\(part (\S+) (\w+)\)", TEXT)}
    script_runs = {name: plain for plain, name in re.findall(r'\$\(run "(A5_PLAIN=1 )?" (\w+)\)', TEXT)}
    assert script_designs == DESIGNS and "ZK_DESIGN=subset:880 " in TEXT
    assert script_runs == PLAIN
    assert "A5_COINS=challenger $1A5_PARTITION=" in TEXT and "bash benchmarks/one_stage/a5_llama1b_h200.sh" in TEXT
    assert "pod_bootstrap.sh --cases LLAMA32_1B_INT8" in TEXT
    assert "verity bench prove layers --flock" in TEXT and "--recursion" not in TEXT


@pytest.mark.parametrize("steps", ["", "1-16"])
def test_its_partition_files_are_the_trees_with_the_layers_sampling_design_and_a5_steps_scope(steps, tmp_path):
    """Unset, A5_STEPS leaves the tree's partition file as it is but for its sampling design; set, it gives each file its
    scope's steps, and the zero-knowledge layer every proof block by rate."""

    lines = TEXT.splitlines()
    start = lines.index(next(x for x in lines if x.startswith("part() {")))
    defs = "\n".join([x for x in lines if x.startswith(("q() {", "PART=", "ZK_DESIGN="))] + lines[start:lines.index("}", start) + 1])
    tree = json.loads((ROOT / "benchmarks" / "one_stage" / "llama32_1b_rope_d64_l0.json").read_text())
    want = {"flock": "stratified-rate:880/880", "zk": "rate:1/1" if steps else "subset:880"}
    for name, design in (("flock", want["flock"]), ("zk", "$ZK_DESIGN")):
        script = f'WORK={tmp_path}; A5_STEPS={steps}\n{defs}\neval "$(part {design} {name})"'
        subprocess.run(["bash", "-c", script], cwd=ROOT, check=True)
        made = json.loads((tmp_path / f"partition-{name}.json").read_text())
        assert made.pop("scope") == ({**tree["scope"], "steps": [1, 16]} if steps else tree["scope"])
        assert list(made) == [k for k in tree if k != "scope"] and want[name] in made.values()
        assert all(made[k] == tree[k] for k in made if made[k] != want[name])



def test_its_steps_run_in_the_readmes_order():
    steps = re.search(r"^STEPS=\((.*?)\)$", TEXT, re.M).group(1).split()
    assert steps == ["bootstrap", "partition-flock", "partition-zk", "partition-sampling", "flock", "zk", "sampling", "layers"]
    assert set(re.findall(r"^  \[([\w-]+)\]=", TEXT, re.M)) == set(steps)


def test_it_reads_only_the_settings_it_names():
    assigned = set(re.findall(r"(?:^|[\s;(])(?:local )?([A-Za-z_]\w*)=", TEXT)) | {"s", "t", "n", "f", "o", "k"}
    read = set(re.findall(r"\$\{?!?([A-Za-z_]\w*)", TEXT)) - (assigned - set(SETTINGS))
    # HOME and PATH are the shell's; RESEARCH_RUN_DIR is a research run's own directory, the results' default
    assert read - {"HOME", "PATH", "RESEARCH_RUN_DIR", "SECONDS", "PIPESTATUS", "BASH_REMATCH"} == set(SETTINGS)
    header = TEXT.split("set -uo pipefail", 1)[0]
    assert all(f"#   {s} " in header for s in SETTINGS)


def test_it_refuses_under_96_gb_of_host_memory_available():
    assert re.findall(r'\[ "\$\{mem_gb:-0\}" -ge (\d+) \]', TEXT) == ["96"]
    assert "under 96 GB of\n# host memory available" in TEXT


def test_it_refuses_under_96_gb_free_under_work():
    assert re.findall(r'\[ "\$\{free_gb:-0\}" -ge (\d+) \]', TEXT) == ["96"]
    assert "not 96: WORK peaks at about 21 GB, and 75 GB must stay free beside it" in TEXT
    assert "under 96 GB free under WORK (75 GB left beside WORK's peak of about 21 GB)" in TEXT


def test_every_refusal_comes_before_the_first_step():
    first_step = TEXT.index("trap finish EXIT")
    refusals = [m.start() for m in re.finditer(r"\brefuse \"", TEXT)]
    assert refusals and max(refusals) < first_step
    assert TEXT.count('rm -rf "$WORK"') == 1 and TEXT.index('rm -rf "$WORK"') > TEXT.index("finish() {")


@pytest.mark.parametrize("missing", ["A5_CARD", "A5_CORES", "WORK"])
def test_a_missing_setting_is_refused_by_name(missing, tmp_path):
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "A5_CARD": "0", "A5_CORES": "0", "WORK": str(tmp_path / "w")}
    del env[missing]
    run = subprocess.run(["bash", str(SCRIPT), "--dry-run"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert run.returncode == 2
    assert run.stderr.startswith(f"proof_layers.sh: no {missing}:")
    assert not (tmp_path / "w").exists()


def test_a_lease_without_its_owner_is_refused(tmp_path):
    (tmp_path / "gpu-lease").write_text("#!/bin/sh\nexit 9\n")
    (tmp_path / "gpu-lease").chmod(0o755)
    env = {"PATH": f"{tmp_path}:{os.defpath}", "HOME": str(tmp_path), "A5_CARD": "0", "A5_CORES": "0", "WORK": str(tmp_path / "w")}
    run = subprocess.run(["bash", str(SCRIPT), "--dry-run"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert run.returncode == 2
    assert run.stderr.startswith("proof_layers.sh: no GPU_LEASE_WHO:")


@pytest.mark.parametrize("steps", ["8-1", "8", "a-b"])
def test_a5_steps_that_are_no_range_of_steps_are_refused(steps, tmp_path):
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "A5_CARD": "0", "A5_CORES": "0", "WORK": str(tmp_path / "w"),
           "A5_STEPS": steps}
    run = subprocess.run(["bash", str(SCRIPT), "--dry-run"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert run.returncode == 2 and run.stderr.startswith(f"proof_layers.sh: A5_STEPS={steps}: LO-HI")
