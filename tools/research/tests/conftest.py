"""Every test gets its own evidence store: `research run` publishes an attempt at the end, and that must never land in the
developer's ~/.research/store.  And every test starts without credentials or a recording run's variables: tests start runs of
their own, which would inherit them."""
from __future__ import annotations

import os

import pytest

#: a cloud VM's Cursor secrets stand in for the laptop's config files (pods/runpod.py, store/remote.py), a recorded check's step
#: holds its run's variables, and a pod's shell its own: no test may reach the real account or bucket, or act as a run, through them
#: (nor extend or read a real machine's lease: a Nebius host exports LEASE_DIR; nor post to the real Slack workspace)
CLEARED = ("AWS_", "R2_", "RESEARCH_", "RUNPOD_", "LEASE_", "SLACK_")


@pytest.fixture(autouse=True)
def _isolated_store(monkeypatch, tmp_path_factory):
    for k in list(os.environ):
        if k.startswith(CLEARED):
            monkeypatch.delenv(k)
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path_factory.mktemp("store")))
    # ... and never the developer's ~/.research/store.toml remote (an S3 config without credentials in the test env would otherwise
    # be what every test sees).  Tests that want a remote point RESEARCH_STORE_CONFIG at their own file.
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path_factory.mktemp("cfg") / "absent-store.toml"))
    monkeypatch.setenv("RESEARCH_MACHINES_D", "none")      # the pod registry (pods/registry.py) is each test's explicit choice
    # ... nor the public export: an empty URL is no public source (unset, as the clear above leaves it, would mean fetchset.PUBLIC_URL)
    monkeypatch.setenv("RESEARCH_PUBLIC_URL", "")
    # ... nor the machine's hard deadline: lease.sh defaults to /etc/research/deadline, which a Nebius host has
    monkeypatch.setenv("LEASE_DEADLINE_FILE", str(tmp_path_factory.mktemp("lease") / "absent-deadline"))
    # ... nor its GPU split: gpu-lease and the runner read /etc/vy/direct-gpus, which says `none` on a cut-over Nebius host
    monkeypatch.setenv("GPU_LEASE_ALLOWED_FILE", str(tmp_path_factory.mktemp("gpu-lease") / "absent-direct-gpus"))
    monkeypatch.setenv("DIRECT_CPUS_FILE", str(tmp_path_factory.mktemp("cpus") / "absent-direct-cpus"))     # nor its CPU map
    # ... nor the agent's Cursor identity: without SLACK_BOT_TOKEN, research slack asks this socket for an OIDC token for its broker
    monkeypatch.setenv("CURSOR_AGENT_SOCKET", str(tmp_path_factory.mktemp("cursor") / "absent.sock"))
