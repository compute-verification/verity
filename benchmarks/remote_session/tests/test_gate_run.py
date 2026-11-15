"""`gate_run.sh --dry-run`: the prover run it would launch and what its gate steps are given. HOST_GIB is the node's
`--host-gib` and that much more of the lease's host memory, and SESSIONS, WINDOWS, WINDOW_MS and SAMPLES reach gate.sh
in place of PACED; with none of them the prover's run and the gate steps' variables are the launcher's defaults. A
HOST_GIB that isn't whole GiB, and a form gate.sh refuses, end it before anything is written or run. The launcher is
a copy in a throwaway repository whose origin's main names one role; `uv` is a stub on PATH that records
each `research run` and answers with a run id, so nothing is launched."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

RS = Path(__file__).resolve().parents[1]
SNAP = "/workspace/hf/hub/models--org--tiny/snapshots/abc123"
NODE_ARGS = f"--decode-s 60 --warm-openings 2773 --wait-s 1800 --model-dir {SNAP}"
SET = ("HOST_GIB", "SESSIONS", "WINDOWS", "WINDOW_MS", "SAMPLES", "PACED", "PACE_PASS_MS", "LEASE_MEM_GB", "LEASE_ON",
       "LEASE_BY", "LEASE_FROM", "ENGINES", "ROLE", "PLAN", "ARENA_GIB", "UTIL", "PORT", "COMMIT", "PIN", "GATE_KEY",
       "GATE_ON", "PROVER_NODE", "PROVER_RUN", "CIRCUIT", "MAXMIN", "GATE_MEM", "SLOTS", "FETCH")
GIT = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
       "GIT_COMMITTER_EMAIL": "t@t"}


def git(r: Path, *cmd: str) -> str:
    return subprocess.run(["git", *cmd], cwd=r, env=GIT, capture_output=True, text=True, check=True).stdout.strip()


def dry_run(tmp_path: Path, **given: str) -> tuple[subprocess.CompletedProcess, list[str], list[list[str]]]:
    """gate_run.sh --dry-run with GIVEN and nothing else of SET, step 1 skipped: its result, the words of the prover's
    run it printed and the words of each `research run` it started, in order (the probe, then the fire)."""
    r, origin, bin_, home = tmp_path / "repo", tmp_path / "origin.git", tmp_path / "bin", tmp_path / "home"
    pin = r / "verity/protocols/pouw/verifier/lean/check/models.json"
    for d in (pin.parent, r / "benchmarks/remote_session", bin_, home):
        d.mkdir(parents=True)
    pin.write_text(json.dumps({"TINY": {"repo": "org/tiny", "revision": "abc123"}}))
    for name in ("gate_run.sh", "gate.sh"):
        shutil.copy(RS / name, r / "benchmarks/remote_session" / name)
    git(tmp_path, "init", "-q", "--bare", str(origin))
    git(r, "init", "-q", "-b", "main")
    git(r, "add", ".")
    git(r, "commit", "-q", "-m", "c0")
    git(r, "remote", "add", "origin", str(origin))
    git(r, "push", "-q", "origin", "main")
    (bin_ / "uv").write_text(f'#!/bin/bash\n(IFS=$\'\\x1f\'; echo "$*") >> {tmp_path}/calls\n'
                             'echo "run r20200101-000000-abcd"\n')
    (bin_ / "uv").chmod(0o755)
    env = {k: v for k, v in GIT.items() if k not in SET}
    env |= {"PATH": f"{bin_}{os.pathsep}{env['PATH']}", "HOME": str(home), "TMPDIR": str(tmp_path),
            "CHECKPOINT": "0", "ROLE": "TINY", **given}
    p = subprocess.run(["bash", str(r / "benchmarks/remote_session/gate_run.sh"), "--dry-run", "ma-remote-coins-9"],
                       cwd=r, env=env, capture_output=True, text=True)
    printed = [line for line in p.stdout.splitlines() if line.startswith("DRY: research run ")]
    calls = (tmp_path / "calls").read_text().splitlines() if (tmp_path / "calls").exists() else []
    return p, printed[0].split() if printed else [], [c.split("\x1f") for c in calls]


def envs(words: list[str]) -> dict[str, str]:
    """A `research run`'s --env settings."""
    return dict(words[i + 1].split("=", 1) for i, w in enumerate(words) if w == "--env")


def after(words: list[str], flag: str) -> str:
    return words[words.index(flag) + 1]


FORM = {"SESSIONS": "1", "WINDOWS": "7", "WINDOW_MS": "60000", "SAMPLES": "33273"}


@pytest.mark.parametrize("given, node_args, mem_gb, form", [
    ({}, NODE_ARGS, "256", {}),
    ({"HOST_GIB": "8", **FORM}, f"{NODE_ARGS} --host-gib 8", "264", FORM),
    ({"HOST_GIB": "08", "ENGINES": "4", "PACED": "1:1000:2773:7", "PACE_PASS_MS": "46"},
     f"{NODE_ARGS} --host-gib 8", "136", {"PACED": "1:1000:2773:7", "PACE_PASS_MS": "46"}),
    ({"HOST_GIB": "8", "LEASE_MEM_GB": "300"}, f"{NODE_ARGS} --host-gib 8", "300", {}),
], ids=["default", "host-part-and-one-coin-a-window", "host-part-paced", "lease-memory-given"])
def test_a_dry_run_prints_the_provers_run_with_the_host_part_and_hands_the_sessions_form_to_the_gate_steps(
        tmp_path, given, node_args, mem_gb, form):
    """HOST_GIB reaches the node as --host-gib and adds to the 32 GB an engine the lease is asked for, unless
    LEASE_MEM_GB says otherwise; without it the node's arguments name no host part. The probe and the fire get the
    form's variables that are set and no other."""
    p, prover, calls = dry_run(tmp_path, **given)
    assert p.returncode == 0, p.stdout + p.stderr
    line = " ".join(prover)
    assert f"--env NODE_ARGS={node_args} --env VERIFIER_SSH=" in line
    assert after(prover, "--mem-gb") == mem_gb
    assert [after(c, "--on") for c in calls] == ["vy-nebius-cpu-5", "vy-nebius-cpu-5"]
    assert [c[-1] for c in calls] == [f'bash "$RESEARCH_RUN_DIR/inputs/gate.sh" {verb}' for verb in ("probe", "fire")]
    for c in calls:
        got = envs(c)
        assert got["DRY"] == "1" and got["MODEL"] == "TINY"
        assert {k: got[k] for k in ("PACED", "PACE_PASS_MS", *FORM) if k in got} == form


@pytest.mark.parametrize("host_gib", ["8.5", "-1", "8G", "eight"])
def test_a_host_part_that_is_not_whole_gib_is_refused_before_anything_is_written_or_run(tmp_path, host_gib):
    p, prover, calls = dry_run(tmp_path, HOST_GIB=host_gib)
    assert p.returncode == 3 and f"HOST_GIB is {host_gib}: the store's host part in whole GiB, 0 for none" in p.stdout
    assert prover == [] and calls == [] and not (tmp_path / "home" / ".research").exists()


@pytest.mark.parametrize("given, why", [
    ({**FORM, "PACED": "1:1000:2773:7"}, "the host's run takes one form, so unset SESSIONS or both of the others"),
    ({**FORM, "PACE_PASS_MS": "46"}, "the host's run takes one form, so unset SESSIONS or both of the others"),
    ({"WINDOWS": "7", "WINDOW_MS": "60000", "SAMPLES": "33273"},
     "SESSIONS is unset: set it, or unset them for PACED's sessions"),
], ids=["sessions-and-paced", "sessions-and-a-pace", "windows-without-sessions"])
def test_both_forms_of_the_hosts_run_or_half_of_one_is_refused_before_anything_is_written_or_run(tmp_path, given, why):
    p, prover, calls = dry_run(tmp_path, **given)
    assert p.returncode == 3 and why in p.stdout, p.stdout + p.stderr
    assert prover == [] and calls == [] and not (tmp_path / "home" / ".research").exists()
