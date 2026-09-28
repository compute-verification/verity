"""Every test gets its own evidence store: `research run` publishes an attempt at the end, and that must never land in the
developer's ~/.research/store.  And every test starts without credentials or a recording run's variables: tests start runs of
their own, which would inherit them."""
from __future__ import annotations

import os
import re
import signal
from pathlib import Path

import pytest

#: a cloud VM's Cursor secrets stand in for the laptop's config files (pods/runpod.py, store/remote.py), a recorded check's step
#: holds its run's variables, and a pod's shell its own: no test may reach the real account or bucket, or act as a run, through them
#: (nor extend or read a real machine's lease: a Nebius host exports LEASE_DIR; nor post to the real Slack workspace; nor, with those
#: cleared, read as a cloud VM's shell that lost its injected secrets: secret_hint.py)
CLEARED = ("AWS_", "R2_", "RESEARCH_", "RUNPOD_", "LEASE_", "SLACK_", "CLOUD_AGENT_", "LINEAR_")


@pytest.fixture(autouse=True)
def _isolated_store(monkeypatch, tmp_path_factory):
    for k in list(os.environ):
        if k.startswith(CLEARED):
            monkeypatch.delenv(k)
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path_factory.mktemp("store")))
    # ... and never the developer's ~/.research/store.toml remote (an S3 config without credentials in the test env would otherwise
    # be what every test sees).  Tests that want a remote point RESEARCH_STORE_CONFIG at their own file.
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path_factory.mktemp("cfg") / "absent-store.toml"))
    # ... so a local `research run` would refuse for want of a remote, unless told what preserves its records: a test's run goes
    # with its tmp dir. The tests of that refusal unset this.
    monkeypatch.setenv("RESEARCH_LOCAL_ONLY", "a test's run: its store is a tmp dir")
    monkeypatch.setenv("RESEARCH_MACHINES_D", "none")      # the pod registry (pods/registry.py) is each test's explicit choice
    monkeypatch.setenv("RESEARCH_MERGE_MIRROR", "")        # nor does a merge's push reach node 1's bare repository
    # ... nor the public export: an empty URL is no public source (unset, as the clear above leaves it, would mean fetchset.PUBLIC_URL)
    monkeypatch.setenv("RESEARCH_PUBLIC_URL", "")
    # ... nor the machine's hard deadline: lease.sh defaults to /etc/research/deadline, which a Nebius host has
    monkeypatch.setenv("LEASE_DEADLINE_FILE", str(tmp_path_factory.mktemp("lease") / "absent-deadline"))
    # ... nor its GPU split: gpu-lease and the runner read /etc/vy/direct-gpus, which says `none` on a cut-over Nebius host
    monkeypatch.setenv("GPU_LEASE_ALLOWED_FILE", str(tmp_path_factory.mktemp("gpu-lease") / "absent-direct-gpus"))
    # ... nor its bookings: gpu-lease refuses a lease that overlaps a window in /workspace/pouw/fill/windows on vy-nebius-2
    monkeypatch.setenv("GPU_LEASE_WINDOWS", str(tmp_path_factory.mktemp("gpu-lease") / "absent-windows"))
    monkeypatch.setenv("DIRECT_CPUS_FILE", str(tmp_path_factory.mktemp("cpus") / "absent-direct-cpus"))     # nor its CPU map
    # ... nor cpu-1's rule that a run comes through a check slot (/etc/vy/runs-via-slot)
    monkeypatch.setenv("RUNS_VIA_SLOT_FILE", str(tmp_path_factory.mktemp("slot-only") / "absent-runs-via-slot"))
    # ... nor its learned bundle sizes: on vy-nebius-1 the pacer mirrors them to /workspace/jobs/dispatch/bundle-sizes.json
    monkeypatch.setenv("VY_BUNDLE_SIZES", str(tmp_path_factory.mktemp("bundle-sizes") / "bundle-sizes.json"))
    # ... nor its dispatcher's settings: importing dispatch.py loads $VY_DISPATCH_ROOT/dispatch.env (default
    # /workspace/jobs/dispatch, node 1's loop's) into os.environ outside monkeypatch, so every later test in the worker would see
    # node 1's CPU slices
    monkeypatch.setenv("VY_DISPATCH_ROOT", str(tmp_path_factory.mktemp("dispatch") / "absent"))
    # ... nor the agent's Cursor identity: without SLACK_BOT_TOKEN, research msg asks this socket for an OIDC token for its broker
    monkeypatch.setenv("CURSOR_AGENT_SOCKET", str(tmp_path_factory.mktemp("cursor") / "absent.sock"))


@pytest.fixture(autouse=True)
def _isolated_git(monkeypatch, tmp_path_factory):
    """Git reads neither the machine's system config nor its global one, but this file: a cloud VM's global config signs every
    commit through an agent that can fail mid-suite (`git commit` exit 128, "Couldn't find key in agent?"), and sets
    core.fsmonitor and push.autoSetupRemote, which a pod's doesn't. A repository's own config and `-c` still win over it."""
    cfg = tmp_path_factory.mktemp("git") / "config"
    cfg.write_text("[user]\n\tname = t\n\temail = t@t\n[commit]\n\tgpgsign = false\n[tag]\n\tgpgsign = false\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def reap(tmp_path):
    """However the test ends, SIGTERM the process group of every process whose argv names its tmp_path: the detached processes
    it starts (n1_lease's fences, a lease loop, a pod guard) lead their own sessions, with their `sleep` children in the group.
    A process that exits meanwhile is skipped (a check runs other suites' processes beside these), and pytest's own group is
    never signalled. On vy-nebius-1 the old per-lock scan raised on such a race and left fences holding /workspace for hours."""
    yield
    own = os.getpgrp()
    named = re.compile(re.escape(os.fsencode(tmp_path)) + rb"(/|\0|$)")    # not a sibling test's .../test_x10 for .../test_x1
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if named.search((proc / "cmdline").read_bytes()):
                pgid = os.getpgid(int(proc.name))
                if pgid != own:
                    os.killpg(pgid, signal.SIGTERM)
        except OSError:
            continue
