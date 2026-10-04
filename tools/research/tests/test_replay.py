"""research.replay: a replay set's attempts classified from their records (replayable / needs window / not replayable / unknown), and
the relaunch each would get.  A tmp store, a tmp machines file, run records written the way the machine's runner writes them."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from research import replay as RP
from research.cli import main as research_main
from research.store import ids
from research.store.local import LocalStore

SET = "warden/re-verify"
SHA = "c" * 40


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_RUNS", str(tmp_path / "runs"))
    (tmp_path / "machines.toml").write_text(
        '[machines.vy-nebius-1]\nprovider = "ssh"\nhost = "10.0.0.1"\nproject = "verity"\n'
        '[machines.vy-control]\nprovider = "runpod"\npod_id = "p1"\nproject = "verity"\n'
        '[machines.other]\nprovider = "ssh"\nhost = "10.0.0.2"\nproject = "elsewhere"\n')
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    return tmp_path


@pytest.fixture
def st(env) -> LocalStore:
    return LocalStore()


def _launch_argv(run_id: str, opts: list[str], command: list[str], wrap: list[str] | None = None) -> list[str]:
    """remote._launch_request's argv for the runner, with cli._run_remote's harness args and remote.launch's --tree-sha."""
    return [*(wrap or []), "python3", "-m", "research", "run", "--runs-dir", "/r/runs", "--id", run_id, "--claimed", "--cost-json", "{}",
            "--remote-json", '{"machine": "vy-nebius-1"}', "--custody-dir", f"/r/requests/{run_id}/custody", "--stage", "run",
            "--project", "verity", *opts, "--source-identity", "{}", "--tree-sha", SHA, "--", *command]


def _attempt(run_id: str, *, machine: str | None = "vy-nebius-1", source_sha: str | None = SHA, argv=None, env=None, outputs=None,
             exclusive=False, mem_gb=None) -> dict:
    remote = {"machine": machine, "pod_id": None, "root": "/r", "source_sha": source_sha, "source_dir": None, "source_archive_sha256": None,
              "source_verified": None} if machine else None
    return {"schema": "research/attempt/v0.1", "id": run_id, "tool": {"name": "python", "version": None}, "derivation": "drv:" + "a" * 64,
            "reuse_key": None, "argv": argv or ["python", "-m", "x"], "params": {"argv": argv or ["python", "-m", "x"], **({"env": env} if env else {})},
            "key_params": ["argv"], "inputs": {}, "conditions": {"exec.machine": machine}, "key_conditions": [],
            "source": {"commit": SHA, "tree": "d" * 40, "dirty": False, "dirty_digest": None, "lockfiles": {}}, "closure": {},
            "execution": {"start_utc": "2026-10-01T00:00:00Z", "end_utc": "2026-10-01T00:01:00Z", "wall_s": 60.0, "host": "h", "machine": machine,
                          "project": "verity", "exclusive": exclusive, "mem_gb_reserved": mem_gb, "remote": remote},
            "cost": None, "status": {"state": "done", "rc": 0, "class": "SUCCESS", "failure": None, "validation": None, "result": "absent"},
            "outputs": outputs or {}, "telemetry": None}


def _job(command_redacted=()) -> dict:
    return {"schema": "telemetry/job/v0.2", "ownership": {"project": "verity"}, "execution": {"command": ["python", "-m", "x"],
                                                                                               "command_redacted": list(command_redacted)}}


def _record(st: LocalStore, tmp: Path, run_id: str, launch_argv: list[str], *, inputs: dict[str, bytes] | None = None, trimmed=(),
            job: dict | None = None) -> str:
    d = tmp / "rec" / run_id
    d.mkdir(parents=True)
    (d / "launch.json").write_text(json.dumps({"run_id": run_id, "argv": launch_argv}))
    (d / "job.json").write_text(json.dumps(job or _job()))
    for name, data in (inputs or {}).items():
        (d / "inputs").mkdir(exist_ok=True)
        (d / "inputs" / name).write_bytes(data)
    meta = {"run": run_id, "trimmed": [{"path": p, "bytes": 1 << 30, "sha256": "e" * 64} for p in trimmed]}
    return st.put_artifact("run-record/v1", meta, tree=d)


def _run(st: LocalStore, tmp: Path, run_id: str, opts=(), command=("python", "-m", "x"), wrap=None, label=True, **kw) -> str:
    rec_kw = {k: kw.pop(k) for k in ("inputs", "trimmed", "job") if k in kw}
    art = _record(st, tmp, run_id, _launch_argv(run_id, list(opts), list(command), wrap), **rec_kw)
    st.put_attempt(_attempt(run_id, argv=list(command), outputs={"run_record": art}, **kw))
    if label:
        st.add_label(run_id, "replay_set", SET, by="t")
    return art


def _by_run(rep: dict) -> dict[str, dict]:
    return {x["run"]: x for x in rep["attempts"] if x["run"]}


def _dry(st: LocalStore, env: Path, **kw) -> dict:
    from research import remote
    return RP.dry_run(st, SET, runs_root=env / "runs", machines=remote.load_machines(), **kw)


R1, R2, R3 = "r20261001-000000-aaaa", "r20261001-000000-bbbb", "r20261001-000000-cccc"


def test_a_shipped_on_run_relaunches_with_only_the_source_swapped(st, env):
    _run(st, env, R1, opts=["--timeout", "600", "--cwd", "source", "--tool", "check", "--campaign=c1"], exclusive=True, mem_gb=8.0,
         env={"FOO": "1"})
    x = _by_run(_dry(st, env, at="main"))[R1]
    assert (x["class"], x["machine"], x["commit"]) == (RP.REPLAYABLE, "vy-nebius-1", SHA)
    assert x["command"] == ["research", "run", "--on", "vy-nebius-1", "--project", "verity", "--source", "<checkout of main>", "--exclusive",
                            "--mem-gb", "8", "--env", "FOO=1", "--env", f"RESEARCH_REPLAY_OF={R1}", "--stage", "run", "--timeout", "600",
                            "--cwd", "source", "--tool", "check", "--campaign=c1", "--", "python", "-m", "x"]
    assert x["prepare"] == []


def test_send_inputs_come_from_the_run_record_and_a_trimmed_one_is_not_replayable(st, env):
    art = _run(st, env, R1, inputs={"spec.json": b"{}"})
    _run(st, env, R2, trimmed=["inputs/traces.bin"])
    rep = _by_run(_dry(st, env))
    assert rep[R1]["class"] == RP.REPLAYABLE
    assert rep[R1]["prepare"] == [["research", "data", "fetch", art, "--path", "inputs", "--to", f"replay-inputs/{R1}"]]
    assert rep[R1]["command"][rep[R1]["command"].index("--send") + 1] == f"replay-inputs/{R1}/inputs/spec.json"
    assert rep[R2]["class"] == RP.NOT_REPLAYABLE and "inputs/traces.bin" in rep[R2]["reason"] and "trimmed" in rep[R2]["reason"]


def test_secrets_a_local_run_no_source_and_a_gone_machine_are_not_replayable(st, env):
    _run(st, env, R1, env={"HF_TOKEN": "<redacted>"})
    _run(st, env, R2, job=_job(command_redacted=["--api-key"]))
    _run(st, env, R3, source_sha=None)
    local, gone, foreign = "r20261001-000000-dddd", "r20261001-000000-eeee", "r20261001-000000-ffff"
    st.put_attempt(_attempt(local, machine=None))
    st.add_label(local, "replay_set", SET, by="t")
    _run(st, env, gone, machine="vy-sp1")
    _run(st, env, foreign, machine="other")
    rep = _by_run(_dry(st, env))
    assert {r: x["class"] for r, x in rep.items()} == dict.fromkeys(rep, RP.NOT_REPLAYABLE)
    assert "--env HF_TOKEN" in rep[R1]["reason"] and "--api-key" in rep[R2]["reason"]
    assert "no source tree" in rep[R3]["reason"] and "ran locally" in rep[local]["reason"]
    assert "'vy-sp1' is not in" in rep[gone]["reason"] and "belongs to project" in rep[foreign]["reason"]
    assert rep[R1]["command"] is None
    moved = _by_run(_dry(st, env, on="vy-control"))
    assert moved[gone]["class"] == RP.REPLAYABLE and moved[gone]["command"][:4] == ["research", "run", "--on", "vy-control"]


def test_a_gpu_lease_needs_a_window_and_prints_the_queue_command(st, env):
    wrap = ["env", "GPU_LEASE_WHO=pouw-c4:lane-k", f"RESEARCH_RUN_ID={R1}", "CUDA_VISIBLE_DEVICES=", "systemd-run", "--user", "--scope",
            "--unit", f"cluster-{R1}", "-p", "MemoryMax=48G", "-p", "AllowedCPUs=0-7,16", "--", "sh", "-c", "check", "cpuset-check",
            "0-7,16", "taskset", "-c", "0-7,16"]
    lease = ["/workspace/pouw/infra/bin/gpu-lease", "1", "--wait", "--max-min", "40", "--mem-gb", "48", "--timed", "--"]
    _run(st, env, R1, opts=["--timeout", "3000"], command=[*lease, "python", "bench.py"], wrap=wrap)
    _run(st, env, R2, command=["sh", "-c", "gpu-lease 2 -- python x.py"])
    rep = _by_run(_dry(st, env, at="abc123"))
    x = rep[R1]
    assert x["class"] == RP.NEEDS_WINDOW and "booked window" in x["reason"]
    assert x["command"] == ["research", "run", "--queue", "--source", "<checkout of abc123>", "--project", "verity", "--on", "vy-nebius-1",
                            "--kind", "pouw-c4", "--gpus", "1", "--max-min", "40", "--mem-gb", "48", "--cpus", "9", "--quiet",
                            "--env", f"RESEARCH_REPLAY_OF={R1}", "--stage", "run", "--timeout", "3000", "--", "python", "bench.py"]
    assert rep[R2]["class"] == RP.NEEDS_WINDOW and "inside it" in rep[R2]["reason"]
    assert rep[R2]["command"][:3] == ["research", "run", "--on"] and rep[R2]["command"][-3:] == ["sh", "-c", "gpu-lease 2 -- python x.py"]


def test_a_queued_cpu_run_relaunches_through_the_queue(st, env):
    wrap = ["python3", "-m", "research", "disk-hold", "--gb", "20", "--path", "/r/runs", "--", "env", "GPU_LEASE_WHO=lane-j",
            "systemd-run", "-p", "MemoryMax=16G", "-p", "AllowedCPUs=4-7", "--", "taskset", "-c", "4-7"]
    _run(st, env, R1, opts=["--timeout", "36000"], wrap=wrap)
    x = _by_run(_dry(st, env))[R1]
    assert x["class"] == RP.REPLAYABLE
    assert x["command"][:12] == ["research", "run", "--queue", "--source", "<checkout of REV>", "--project", "verity", "--on", "vy-nebius-1",
                                 "--mem-gb", "16", "--cpus"]
    assert x["command"][x["command"].index("--custody-ttl") + 1] == "11h", "a 10 h run outlives the default 6 h custody key"


def test_artifact_and_derivation_members_stand_for_their_attempts(st, env, tmp_path):
    _run(st, env, R1, label=False)
    res = st.put_artifact("bench-result/v1", {"x": 1})
    doc = _attempt(R2, outputs={"result": res, "run_record": _record(st, tmp_path, R2, _launch_argv(R2, [], ["python", "-m", "x"]))})
    st.put_attempt(doc)
    hand = st.put_artifact("fixture/v1", {"y": 2})
    for t in (res, hand, R2, "drv:" + "a" * 64):
        st.add_label(t, "replay_set", SET, by="t")
    rep = _dry(st, env)
    assert rep["members"] == 4
    runs = _by_run(rep)
    assert set(runs) == {R1, R2}, "the derivation's attempts, each once"
    assert runs[R2]["targets"] == sorted([res, R2, "drv:" + "a" * 64])
    orphan = [x for x in rep["attempts"] if x["run"] is None]
    assert orphan[0]["targets"] == [hand] and "put by hand" in orphan[0]["reason"]


def test_records_not_in_the_local_store_are_unknown_until_fetched(env, monkeypatch):
    (env / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{env / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(env / "store.toml"))
    st = LocalStore()
    art = _run(st, env, R1)
    assert st.push(art)["preserved"]
    m = st.get_manifest(art)
    for f in m.payload.files:
        st.object_path(f.sha256).unlink()
    assert _by_run(_dry(st, env))[R1]["class"] == RP.UNKNOWN
    st.manifest_path(ids.hex_of(art)).unlink()
    x = _by_run(_dry(st, env))[R1]
    assert x["class"] == RP.UNKNOWN and "--fetch" in x["reason"]
    assert _by_run(_dry(st, env, fetch=True))[R1]["class"] == RP.REPLAYABLE


def test_without_a_run_record_a_full_local_copy_of_the_run_dir_serves(st, env):
    st.put_attempt(_attempt(R1))
    st.add_label(R1, "replay_set", SET, by="t")
    d = env / "runs" / R1
    d.mkdir(parents=True)
    (d / "launch.json").write_text(json.dumps({"argv": _launch_argv(R1, [], ["python", "-m", "x"])}))
    (d / "job.json").write_text(json.dumps(_job()))
    x = _by_run(_dry(st, env))[R1]
    assert x["class"] == RP.NOT_REPLAYABLE and "--send" in x["reason"] and f"research fetch {R1} --all" in x["reason"]
    (d / "inputs").mkdir()
    (d / "inputs" / "a.txt").write_text("a")
    (d / "preserved.json").write_text("{}")
    x = _by_run(_dry(st, env))[R1]
    assert x["class"] == RP.REPLAYABLE and x["prepare"] == [["cp", "-r", f"{d}/inputs", f"replay-inputs/{R1}/"]]
    shutil.rmtree(d)
    assert "no run record in the store" in _by_run(_dry(st, env))[R1]["reason"]


def test_runner_argv_drops_the_runners_own_flags_in_both_spellings():
    wrap, opts, cmd = RP.runner_argv(["py", "-m", "research", "run", "--id=r1", "--claimed", "--tree-sha", SHA, "--timeout", "5", "--",
                                      "a", "--id", "x"])
    assert (wrap, opts, cmd) == ([], ["--timeout", "5"], ["a", "--id", "x"])
    with pytest.raises(ValueError):
        RP.runner_argv(["py", "x.py"])


def test_cli_dry_run_json_summary_and_refusals(st, env, capsys):
    _run(st, env, R1)
    _run(st, env, R2, env={"AWS_SECRET_ACCESS_KEY": "<redacted>"})
    assert research_main(["replay", SET, "--dry-run", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["summary"] == {RP.REPLAYABLE: 1, RP.NEEDS_WINDOW: 0, RP.NOT_REPLAYABLE: 1, RP.UNKNOWN: 0}
    assert research_main(["replay", SET, "--dry-run", "--at", "main"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == f"{SET} at main"
    assert out.splitlines()[-1] == f"{SET}: 2 member(s), 2 attempt(s): 1 replayable, 0 needs window, 1 not replayable"
    assert research_main(["replay", "warden/re-check", "--dry-run"]) == 2
    assert research_main(["replay", SET]) == 2
    assert research_main(["replay", "pous/re-run", "--dry-run"]) == 1
    assert "no target is labelled replay_set=pous/re-run" in capsys.readouterr().err
