"""`gate_run.sh --dry-run`: the prover run it would launch and what its gate steps are given. HOST_GIB is the node's
`--host-gib` and that much more of the lease's host memory, LEAF_CTAS is the node's `--leaf-ctas`, and SESSIONS,
WINDOWS, WINDOW_MS and SAMPLES reach gate.sh in place of PACED; with none of them the prover's run and the gate steps'
variables are the launcher's defaults, and no model reaches gate.sh. A HOST_GIB that isn't whole GiB, a LEAF_CTAS
that isn't a count of CTAs, a form gate.sh refuses, and a host whose address the environment doesn't hold, end it
before anything is written or run; the gate steps are handed the two auditor devices' addresses. The launcher is
a copy in a throwaway repository whose origin's main names one role; `uv` is a stub on PATH that records
each `research run` and answers with a run id, so nothing is launched."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

RS = (Path(__file__).resolve().parents[3] / "benchmarks" / "poub_session")
SNAP = "/workspace/hf/hub/models--org--tiny/snapshots/abc123"
NODE_ARGS = f"--decode-s 60 --warm-openings 2773 --wait-s 1800 --model-dir {SNAP}"
SET = ("HOST_GIB", "LEAF_CTAS", "BIND", "SESSIONS", "WINDOWS", "WINDOW_MS", "SAMPLES", "PACED", "PACE_PASS_MS", "LEASE_MEM_GB", "LEASE_ON",
       "LEASE_BY", "LEASE_FROM", "ENGINES", "ROLE", "PLAN", "ARENA_GIB", "UTIL", "PORT", "COMMIT", "PIN", "GATE_KEY",
       "GATE_ON", "PROVER_NODE", "PROVER_RUN", "CIRCUIT", "MAXMIN", "GATE_MEM", "GATE_POOL", "SLOTS", "FETCH")
#: the hosts' addresses as the launcher's environment holds them, here documentation addresses
HOSTS = {"VY_CHALLENGER_1_HOST": "192.0.2.21", "VY_AUDITOR_1_HOST": "192.0.2.22", "VY_NEBIUS_2_HOST": "192.0.2.23"}
GIT = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
       "GIT_COMMITTER_EMAIL": "t@t"}


def git(r: Path, *cmd: str) -> str:
    return subprocess.run(["git", *cmd], cwd=r, env=GIT, capture_output=True, text=True, check=True).stdout.strip()


def dry_run(tmp_path: Path, words: tuple[str, ...] = ("--dry-run",), **given: str,
            ) -> tuple[subprocess.CompletedProcess, list[str], list[list[str]]]:
    """gate_run.sh --dry-run (or WORDS) with GIVEN, the hosts' addresses and nothing else of SET, step 1 skipped: its
    result, the words of the prover's run it printed and the words of each `research run` it started, in order (the
    probe, then the fire)."""
    r, origin, bin_, home = tmp_path / "repo", tmp_path / "origin.git", tmp_path / "bin", tmp_path / "home"
    pin = r / "verity/protocols/accounting/compute/pouw/verifier/lean/check/models.json"
    for d in (pin.parent, r / "benchmarks/poub_session", bin_, home):
        d.mkdir(parents=True)
    pin.write_text(json.dumps({"TINY": {"repo": "org/tiny", "revision": "abc123"}}))
    for name in ("gate_run.sh", "gate.sh"):
        shutil.copy(RS / name, r / "benchmarks/poub_session" / name)
    git(tmp_path, "init", "-q", "--bare", str(origin))
    git(r, "init", "-q", "-b", "main")
    git(r, "add", ".")
    git(r, "commit", "-q", "-m", "c0")
    git(r, "remote", "add", "origin", str(origin))
    git(r, "push", "-q", "origin", "main")
    (bin_ / "uv").write_text(f'#!/bin/bash\n(IFS=$\'\\x1f\'; echo "$*") >> {tmp_path}/calls\n'
                             'echo "run r20200101-000000-abcd"\n')
    (bin_ / "uv").chmod(0o755)
    env = {k: v for k, v in GIT.items() if k not in (*SET, "PROVER_SSH", "VERIFIER_SSH")}
    env |= {**HOSTS, "PATH": f"{bin_}{os.pathsep}{env['PATH']}", "HOME": str(home), "TMPDIR": str(tmp_path),
            "CHECKPOINT": "0", "ROLE": "TINY", **given}
    p = subprocess.run(["bash", str(r / "benchmarks/poub_session/gate_run.sh"), *words, "ma-remote-coins-9"],
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


@pytest.mark.parametrize("leaf_ctas", ["0", "016", "-1", "160.5", "1000", "many"])
def test_a_leaf_pass_that_is_not_a_count_of_ctas_is_refused_before_anything_is_written_or_run(tmp_path, leaf_ctas):
    p, prover, calls = dry_run(tmp_path, LEAF_CTAS=leaf_ctas)
    assert p.returncode == 3, p.stdout + p.stderr
    assert f"LEAF_CTAS is {leaf_ctas}: the leaf pass's reading CTAs, 1 to 999, or unset for today's pass" in p.stdout
    assert prover == [] and calls == [] and not (tmp_path / "home" / ".research").exists()


@pytest.mark.parametrize("unset, whose", [("VY_CHALLENGER_1_HOST", "the challenger host"),
                                          ("VY_AUDITOR_1_HOST", "the verifier host"),
                                          ("VY_NEBIUS_2_HOST", "the prover's node, or PROVER_SSH")])
def test_a_host_whose_address_the_environment_doesnt_hold_is_refused_by_name_before_anything_is_written_or_run(
        tmp_path, unset, whose):
    p, prover, calls = dry_run(tmp_path, **{unset: ""})
    assert p.returncode == 3 and f"{unset} is unset: the address of {whose}" in p.stdout, p.stdout + p.stderr
    assert prover == [] and calls == [] and not (tmp_path / "home" / ".research").exists()


def test_a_prover_ssh_that_is_given_reads_no_address_of_the_provers_node_and_whatif_reads_no_hosts(tmp_path):
    """PROVER_SSH given empty is GATE_ON's own /etc/vy/peers line, so the gate steps get none and the prover node's
    address isn't asked for; whatif reaches no host, so it gets as far as its records with no address at all."""
    p, _, calls = dry_run(tmp_path, VY_NEBIUS_2_HOST="", PROVER_SSH="")
    assert p.returncode == 0 and len(calls) == 2, p.stdout + p.stderr
    assert all("PROVER_SSH" not in envs(c) for c in calls)
    p, _, calls = dry_run(tmp_path / "whatif", ("whatif",), **dict.fromkeys(HOSTS, ""))
    assert p.returncode == 3 and "no fire and prover runs in " in p.stdout and calls == [], p.stdout + p.stderr
