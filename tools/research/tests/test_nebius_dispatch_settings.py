"""Node 1's dispatcher loop runs each tick in a fresh process from the environment it started with, so an edit to dispatch.env
applies at the next tick and a key removed from it drops out. On 2 Oct a lend of cores 128-159 to provers never reached the
running loop, which had read the file once, at its start, and two agents had to restart it by hand."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from research.pods import registry

DISPATCH = Path(registry.__file__).resolve().parent / "nebius" / "dispatch.py"
#: what a tick's process would render with: the settings a fresh import of dispatch.py resolves from its environment
PROBE = ("import importlib.util, json, sys; spec = importlib.util.spec_from_file_location('d', sys.argv[1]); "
         "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
         "print(json.dumps({'prover': m.PROVER_CPUS, 'cpus': m.CPUS, 'hostdirs': m.LEASE_HOSTDIRS}))")


class Stop(Exception):
    pass


def test_each_tick_reads_dispatch_env_as_it_is_now_and_a_removed_key_drops_out(tmp_path, monkeypatch, capsys):
    root = tmp_path / "dispatch"
    root.mkdir()
    env_file = root / "dispatch.env"
    env_file.write_text("# node 1\nVY_PROVER_CPUS=128-191\nVY_LEASE_HOSTDIRS=deployments-gpu=/run/lent\nVY_DISPATCH_CPUS=0-1\n")
    monkeypatch.setenv("VY_DISPATCH_ROOT", str(root))
    monkeypatch.setenv("VY_DISPATCH_CPUS", "96-127")      # the caller's own setting still wins over the file's
    for k in ("VY_PROVER_CPUS", "VY_LEASE_HOSTDIRS"):
        monkeypatch.delenv(k, raising=False)
    before = dict(os.environ)
    spec = importlib.util.spec_from_file_location("dispatch_settings", DISPATCH)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)              # as the loop does at its start: dispatch.env into its own os.environ
        assert os.environ["VY_PROVER_CPUS"] == "128-191" and mod.PROVER_CPUS == "128-191"

        seen, rcs, real_run = [], iter([0, 0, 0, 1, mod.TICK_FAILED]), subprocess.run

        def run(argv, env):
            assert argv == [sys.executable, str(DISPATCH.resolve()), "loop", "--once"]
            out = real_run([sys.executable, "-c", PROBE, str(DISPATCH)], env=env, capture_output=True, text=True, check=True)
            seen.append(json.loads(out.stdout))
            return subprocess.CompletedProcess(argv, next(rcs))

        edits = iter(["VY_PROVER_CPUS=160-191\nVY_LEASE_HOSTDIRS=deployments-gpu=/run/lent\n",    # the lend ends
                      "# no keys: every setting falls back to its default\n", "", ""])

        def sleep(s):
            assert s == 7
            text = next(edits, None)
            if text is None:
                raise Stop
            env_file.write_text(text)

        monkeypatch.setattr(mod.subprocess, "run", run)
        monkeypatch.setattr(mod.time, "sleep", sleep)
        with pytest.raises(Stop):
            mod.main(["loop", "--every", "7"])
    finally:
        for k in set(os.environ) - set(before):
            del os.environ[k]

    default_hostdirs = {"deployments-gpu": "/run/gpu-lease-circuits"}
    assert seen == [{"prover": "128-191", "cpus": "96-127", "hostdirs": {"deployments-gpu": "/run/lent"}},
                    {"prover": "160-191", "cpus": "96-127", "hostdirs": {"deployments-gpu": "/run/lent"}},
                    {"prover": "128-175", "cpus": "96-127", "hostdirs": default_hostdirs},
                    {"prover": "128-175", "cpus": "96-127", "hostdirs": default_hostdirs},
                    {"prover": "128-175", "cpus": "96-127", "hostdirs": default_hostdirs}]
    # a tick that died before printing its own `tick failed` is reported by the loop, and one that printed it isn't reported twice
    err = capsys.readouterr().err
    assert err.count("tick failed: exit 1") == 1 and f"exit {mod.TICK_FAILED}" not in err
