"""A leased deployments-gpu Job mounts circuits' lease pool whoever renders it. On Oct 1 node 2's Build hand-back ran
`dispatch.py submit --task 1` over ssh without VY_LEASE_HOSTDIRS, and five circuits Commits leased GPUs from proofs' pool."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from research.pods import registry

pytest.importorskip("yaml")
NEBIUS = Path(registry.__file__).resolve().parent / "nebius"
TEMPLATE = ("name: config-run\n---\nname: build\nresources:\n  cpus: 4\n  memory: 8\n  labels:\n    kueue.x-k8s.io/queue-name: q\n"
            "run: echo build\n---\nname: gpu\nresources:\n  cpus: 4\n  memory: 64\n  accelerators: RTXPRO6000:1\n  labels:\n"
            "    kueue.x-k8s.io/queue-name: deployments-gpu\nrun: echo gpu\n")


def load(tmp_path, monkeypatch, hostdirs: str | None):
    if hostdirs is None:
        monkeypatch.delenv("VY_LEASE_HOSTDIRS", raising=False)
    else:
        monkeypatch.setenv("VY_LEASE_HOSTDIRS", hostdirs)
    monkeypatch.setenv("VY_DISPATCH_ROOT", str(tmp_path / "dispatch"))
    monkeypatch.delenv("PACK_COMMITS", raising=False)
    spec = importlib.util.spec_from_file_location("dispatch", NEBIUS / "dispatch.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "dispatch", mod)
    spec.loader.exec_module(mod)
    (tmp_path / "sky" / "jobs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "sky" / "jobs" / "config-run.yaml").write_text(TEMPLATE)
    monkeypatch.setattr(mod, "TEMPLATES", tmp_path / "sky" / "jobs")
    return mod


def pool(mod, tmp_path, queue: str | None = None) -> str:
    item = {"template": "config-run", "tree": str(tmp_path), "lease": "self", "env": {"ROW": "r"}, **({"queue": queue} if queue else {})}
    job = mod.render(item, "vllm-epoch-run/cov-gm302", 1, 0)
    (vol,) = [v for v in job["spec"]["template"]["spec"]["volumes"] if v["name"] == "gpu-lease"]
    return vol["hostPath"]["path"]


def test_without_the_variable_a_leased_commit_mounts_circuits_pool(tmp_path, monkeypatch):
    mod = load(tmp_path, monkeypatch, None)
    assert pool(mod, tmp_path) == "/run/gpu-lease-circuits"
    assert pool(mod, tmp_path, queue="provers") == "/run/gpu-lease"


def test_the_variable_still_overrides_the_default(tmp_path, monkeypatch):
    assert pool(load(tmp_path, monkeypatch, ""), tmp_path) == "/run/gpu-lease"
    assert pool(load(tmp_path, monkeypatch, "deployments-gpu=/run/gpu-lease-x"), tmp_path) == "/run/gpu-lease-x"
