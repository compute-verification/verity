"""Resume and the stores: which runs count as a point's, across machines and remotes, and a requeued or timed point's records."""
from __future__ import annotations

import calendar
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from bench_standins import FakeFactory, inputs

import verity_bench.cli as cli
from verity_bench import jobs
from verity_bench.cli import main
from verity_bench.vllm import result as result_format
from verity_bench.vllm.presets import (
    GPUS,
    expand_sweep,
    resolve_spec,
)
from verity_bench.vllm.runner import (
    run_benchmark,
)

COMMITTED = calendar.timegm((2026, 10, 21, 12, 0, 0))
RTX = GPUS["rtx-pro-6000"].nvidia_smi_name


def _repo(tmp_path):
    """A workspace checkout whose one commit, made at COMMITTED, is the commit the lookup reads its tree from."""

    (repo := tmp_path / "repo").mkdir()
    (repo / "uv.lock").write_text("lock\n")
    (repo / "pyproject.toml").write_text("")
    dates = f"GIT_AUTHOR_DATE='@{COMMITTED} +0000' GIT_COMMITTER_DATE='@{COMMITTED} +0000'"
    subprocess.run(f"git init -q && git add . && {dates} git -c user.name=t -c user.email=t@example.com "
                   "-c commit.gpgsign=false commit -qm points", shell=True, cwd=repo, check=True)
    return repo, subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _attempt(run, identity, repo, *, validation="passed", rc=0, dirty=False, tree=None, tool="vllm_bench_point"):
    """An attempt as `research run` records it from `repo` (from another commit's `tree`, or another `tool`)."""

    from research import env
    from research.store import derivation

    source = {**env.source_identity(env.source_conditions(repo)), **({"tree": tree} if tree else {}), "dirty": dirty}
    drv = derivation.derivation_id({"name": tool, "version": "0"}, {}, {"identity": identity}, {}, ["hw.gpu.family"])
    return {"schema": "research/attempt/v0.1", "id": run, "tool": {"name": tool, "version": "0"}, "derivation": drv,
            "reuse_key": derivation.reuse_key(drv, source), "argv": [], "params": {"identity": identity}, "inputs": {},
            "key_params": ["identity"], "conditions": {}, "key_conditions": [], "source": source, "closure": None,
            "execution": {"start_utc": f"{run[1:5]}-{run[5:7]}-{run[7:9]}T{run[10:12]}:{run[12:14]}:{run[14:16]}Z"},
            "cost": None, "status": {"state": "done", "rc": rc, "validation": validation}, "outputs": {}, "telemetry": None}


def test_resume_counts_only_a_clean_exited_passing_run_of_the_same_identity_and_commit(tmp_path):
    """Validation comes from result.json alone, so a run killed after writing it (rc 143,
    preempted) read as passed, and so did a run from a dirty tree, whose code its commit doesn't name."""

    from research.store.local import LocalStore

    (repo, commit), ident = _repo(tmp_path), "f" * 64
    store, config = _stores(tmp_path, [_attempt("r20261021-130000-0001", ident, repo),
                                       _attempt("r20261021-140000-0002", ident, repo),
                                       _attempt("r20261021-150000-0003", ident, repo, rc=143),
                                       _attempt("r20261021-160000-0004", ident, repo, dirty=True),
                                       _attempt("r20261021-170000-0005", ident, repo, validation="failed", rc=3),
                                       _attempt("r20261021-180000-0006", ident, repo, tree="b" * 40),
                                       _attempt("r20261021-190000-0007", "e" * 64, repo),
                                       _attempt("r20261021-200000-0008", ident, repo, tool="something_else")])
    assert jobs.StoreCompletion(LocalStore(store, config_path=config), root=repo)(ident, commit) == "r20261021-140000-0002"

    root = Path(__file__).resolve().parents[3]
    spec = resolve_spec("llama31-8b-fp8-1gpu")
    pins = jobs.toolchain(root, spec, commit=commit)
    point = expand_sweep("quick")[0]
    identity = jobs.point_identity(spec.config_sha256, pins, point)
    assert identity != jobs.point_identity(spec.config_sha256, replace(pins, vllm_pin="0" * 40), point)
    assert identity != jobs.point_identity("0" * 64, pins, point)


def _stores(tmp_path, remote_attempts):
    """A fresh local store whose remote is a directory holding `remote_attempts`, so nothing reaches the network."""

    from research.store import canon

    remote = tmp_path / "remote"
    (remote / "attempts").mkdir(parents=True)
    for doc in remote_attempts:
        (remote / "attempts" / f"{doc['id']}.json").write_bytes(canon.dumps(doc))
    config = tmp_path / "store.toml"
    config.write_text(f'[remote]\ntype = "fs"\npath = "{remote}"\n')
    return tmp_path / "local", config


def test_a_point_another_machine_ran_counts_and_an_absent_point_stays_missing(tmp_path):
    """The lookup once read only this machine's catalog, so a fresh VM saw every point as missing (and `run` would
    have launched them again). It pulls the remote's attempts since the commit, once, and nothing older."""

    from research.store.local import LocalStore

    (repo, commit), found, absent, failed = _repo(tmp_path), "1" * 64, "2" * 64, "3" * 64
    remote_attempts = [
        _attempt("r20261021-130000-0001", found, repo),
        _attempt("r20261021-140000-0002", failed, repo, validation="failed", rc=3),
        _attempt("r20261018-120000-0003", absent, repo),  # before the commit's margin: never fetched
        _attempt("r20261021-150000-0004", absent, repo, tree="b" * 40),
    ]
    root, config = _stores(tmp_path, remote_attempts)
    store = LocalStore(root, config_path=config)
    logged = []
    lookup = jobs.StoreCompletion(store, root=repo, log=logged.append)

    assert jobs.StoreCompletion(LocalStore(root, config_path=config), root=repo, local_only=True)(found, commit) is None
    assert lookup(found, commit) == "r20261021-130000-0001"
    assert lookup(absent, commit) is None
    assert lookup(failed, commit) is None
    assert sorted(store.attempt_ids()) == ["r20261021-130000-0001", "r20261021-140000-0002", "r20261021-150000-0004"]
    assert logged == [f"pulled 3 attempts started since r20261020 from {store.remote.describe()} (0 of 3 were here)"]

    later = _attempt("r20261021-160000-0005", absent, repo)
    (tmp_path / "remote" / "attempts" / f"{later['id']}.json").write_bytes(json.dumps(later).encode())
    assert lookup(absent, commit) is None, "one pull per commit"
    again = jobs.StoreCompletion(store, root=repo, log=logged.append)
    assert again(found, commit) == "r20261021-130000-0001" and len(logged) == 1, "a local hit reads no remote"
    assert again(absent, commit) == "r20261021-160000-0005"
    assert logged[-1].startswith("pulled 1 attempts") and logged[-1].endswith("(3 of 4 were here)")


def test_without_a_remote_the_lookup_refuses_unless_told_to_read_the_catalog_alone(tmp_path):
    from research.store.local import LocalStore

    repo, commit = _repo(tmp_path)
    config = tmp_path / "store.toml"
    config.write_text("")
    store = LocalStore(tmp_path / "local", config_path=config)
    with pytest.raises(RuntimeError, match="--local-only"):
        jobs.StoreCompletion(store, root=repo)("1" * 64, commit)
    assert jobs.StoreCompletion(store, root=repo, local_only=True)("1" * 64, commit) is None


def test_check_on_a_fresh_store_counts_the_points_the_remote_holds(tmp_path, monkeypatch, capsys):
    repo, commit = _repo(tmp_path)
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", commit)
    monkeypatch.chdir(repo)
    points = jobs.make_plan("llama31-8b-fp8-1gpu", root=repo, lookup=lambda identity, commit: None).points
    ran = [_attempt(f"r20261021-13000{i}-000{i}", item.identity, repo) for i, item in enumerate(points[1:])]
    store, config = _stores(tmp_path, ran)
    monkeypatch.setenv("RESEARCH_STORE", str(store))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(config))

    assert main(["check", "llama31-8b-fp8-1gpu", "--local-only"]) == 1
    assert f"0/{len(points)} points" in capsys.readouterr().out
    assert main(["check", "llama31-8b-fp8-1gpu"]) == 1
    out = capsys.readouterr().out
    assert f"{len(points) - 1}/{len(points)} points" in out
    assert f"{points[0].point.id}\t{points[0].identity}\tmissing" in out
    assert all(f"{item.identity}\t{run['id']}" in out for item, run in zip(points[1:], ran))


def test_a_requeued_point_deletes_its_earlier_attempts_verdict_before_it_runs(tmp_path, monkeypatch):
    """A preempted point reruns whole; if it reran in its run directory and died before writing a
    new result.json, the earlier attempt's verdict would stand beside the new attempt's half-written files."""

    root = Path(__file__).resolve().parents[3]
    plan = jobs.make_plan("llama31-8b-fp8-1gpu", root=root, commit="a" * 40, lookup=lambda identity, commit: None)
    item = plan.points[0]
    (tmp_path / "result.json").write_text('{"validation": {"status": "passed"}}')
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", "a" * 40)
    monkeypatch.chdir(root)

    def preempted(*args, **kwargs):
        raise RuntimeError("preempted")

    monkeypatch.setattr(cli, "run_benchmark", preempted)
    monkeypatch.setattr(result_format.kit, "host_stamp", lambda: {"card": RTX})
    with pytest.raises(RuntimeError, match="preempted"):
        cli.main(["_vllm-point", "--spec", "llama31-8b-fp8-1gpu", "--point", item.point.id, "--identity", item.identity])
    assert not (tmp_path / "result.json").exists()


def test_a_timed_point_records_its_numbers_in_its_runs_result(tmp_path, monkeypatch):
    """`_vllm-point` on one timed point, run on the fake engine at the test's preset and arms, writes its result and
    then each of the point's numbers into it through the kit's writer (`result.record`), named ``ARM.METRIC``."""

    root = Path(__file__).resolve().parents[3]
    plan = jobs.make_plan("llama31-8b-fp8-1gpu", root=root, commit="a" * 40, lookup=lambda identity, commit: None)
    item = plan.points[0]
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", "a" * 40)
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261114-000000-0000")
    monkeypatch.delenv("OUT", raising=False)
    monkeypatch.chdir(root)
    preset, _, arms = inputs()
    ran = []

    def timed(_preset, _arms, points, **_kwargs):
        ran.append(points)
        return run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=FakeFactory(), git_commit="abc")

    monkeypatch.setattr(cli, "run_benchmark", timed)
    monkeypatch.setattr(result_format.kit, "host_stamp", lambda: {"commit": "abc", "card": RTX})
    assert cli.main(["_vllm-point", "--spec", "llama31-8b-fp8-1gpu", "--point", item.point.id,
                     "--identity", item.identity]) == 0
    assert ran == [(item.point,)]
    held = json.loads((tmp_path / "result.json").read_text())
    assert held["run_id"] == "r20261114-000000-0000" and held["validation"]["status"] == "passed"
    got = {m["name"]: m for m in held["measurements"]}
    doc = json.loads((tmp_path / "out" / "benchmark.json").read_text())
    (row,) = doc["points"]
    assert got and {m["point"] for m in got.values()} == {item.point.id} == {row["point"]["id"]}
    assert {m["mark"] for m in got.values()} == {"measured"}
    for arm, a in row["arms"].items():
        for metric, m in a["metrics"].items():
            assert got[f"{arm}.{metric}"]["value"] == m["mean"]
            assert (f"{arm}.{metric}.ratio" in got) == (arm != doc["baseline"])
