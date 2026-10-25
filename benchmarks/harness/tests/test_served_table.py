from __future__ import annotations

import calendar
import json
from pathlib import Path

import pytest

from verity_bench import served_table
from verity_bench.cli import main
from verity_bench.vllm.arms import Arm
from verity_bench.vllm.presets import expand_sweep, load_spec, resolve_spec
from verity_bench.vllm.runner import run_benchmark

COMMIT_SPEC = "llama31-8b-fp8-1gpu-commit-served"
ADDED = calendar.timegm((2026, 10, 24, 1, 0, 0))
# Time per output token by arm; every arm slows by the same factor each rep, so the paired ratios are exact.
SCALE = {"stock-fp8": 2.0, "commit": 2.4, "stock-fp8@compiled": 1.0, "commit@compiled": 1.2}


class ScaledEngine:
    def __init__(self, arm, scale):
        self.arm, self.scale, self.reps = arm, scale, 0
        self.identity = {"vllm_version": "0.fake+gd9105ea80", "kernel_digest": None, "engine_overrides": {}, "kv_cache_tokens": 1}

    def gate(self, point):
        return {f"{point.id}-0": (7, 9)}

    def custom_gate(self):
        return {"passed": True, "provided": False}

    def warmup(self, point):
        return {"wall_s": 0.5}

    def measure(self, point):
        from verity_bench.vllm.runner import _token_digest

        self.reps += 1
        tpot = self.scale * SCALE[self.arm.id] * (1 + self.reps / 10)
        return {"token_digest": _token_digest({f"{point.id}-0": (7, 9)}),
                "metrics": {"time_per_output_token_s": tpot, "decode_time_per_token_s": 0.8 * tpot,
                            "time_to_first_token_s": 1.0, "tokens_per_s": 1 / tpot, "peak_gpu_memory_bytes": 1.0}}

    def close(self):
        pass


class ScaledFactory:
    def __init__(self, commit_scale):
        self.commit_scale = commit_scale

    def create(self, arm, preset, *, resident):
        return ScaledEngine(arm, self.commit_scale if arm.id.startswith("commit") else 1.0)


def _result(point_id: str, commit_scale: float, reps: int = 2) -> dict:
    [point] = [p for p in expand_sweep("quick") if p.id == point_id]
    arms = (Arm("stock-fp8"), Arm("commit", control="stock-fp8"), Arm("stock-fp8@compiled"),
            Arm("commit@compiled", control="stock-fp8@compiled"))
    return run_benchmark(load_spec(COMMIT_SPEC).preset, arms, (point,), sweep="quick", reps=reps,
                         factory=ScaledFactory(commit_scale), git_commit="c" * 40)


class Producer:
    """A machine that ran points: it publishes each result, its attempt and its labels to the remote directory."""

    def __init__(self, tmp_path: Path):
        from research.store.local import LocalStore

        self.tmp_path, self.remote = tmp_path, tmp_path / "remote"
        (self.remote / "attempts").mkdir(parents=True)
        self.config = tmp_path / "store.toml"
        self.config.write_text(f'[remote]\ntype = "fs"\npath = "{self.remote}"\n')
        self.store = LocalStore(tmp_path / "producer", config_path=self.config)

    def ran(self, run: str, spec: str, point: str, doc: dict | None, *, validation="passed", rc=0) -> str | None:
        from research.store import canon

        art = None
        if doc is not None:
            tree = self.tmp_path / "files" / run / "out"
            tree.mkdir(parents=True)
            (tree / "benchmark.json").write_text(json.dumps(doc))
            files = self.store.put_artifact("run-files/v1", {}, tree=tree.parent)
            art = self.store.put_artifact("verity-vllm-benchmark/v1", {"run_id": run}, {"run_files": files})
            self.store.push(art)
            self.store.push(files)
        start = f"{run[1:5]}-{run[5:7]}-{run[7:9]}T{run[10:12]}:{run[12:14]}:{run[14:16]}Z"
        attempt = {
            "schema": "research/attempt/v0.1", "id": run, "tool": {"name": "vllm_bench_point", "version": "0"},
            "derivation": "drv:" + "0" * 64, "reuse_key": None, "argv": [],
            "params": {"spec": served_table.spec_path(spec), "point": point, "identity": "1" * 64},
            "key_params": ["identity"], "inputs": {}, "conditions": {}, "key_conditions": [],
            "source": {"commit": "c" * 40, "dirty": False}, "closure": None, "execution": {"start_utc": start},
            "cost": None, "status": {"state": "done" if rc == 0 else "failed", "rc": rc, "validation": validation},
            "outputs": {"result": art} if art else {}, "telemetry": None,
        }
        (self.remote / "attempts" / f"{run}.json").write_bytes(canon.dumps(attempt))
        return art

    def label(self, target: str, key: str, value: str, ts: str) -> None:
        self.store.add_label(target, key, value, by="benchmarks", ts=ts)
        self.store.push_labels([target])


def _fresh(tmp_path: Path, producer: Producer):
    from research.store.local import LocalStore

    return LocalStore(tmp_path / "fresh", config_path=producer.config)


def _commit_rows(table: dict) -> dict[str, dict]:
    return {row["point"]: row for row in table["rows"] if row["protocol"] == "Commit"}


def test_the_served_table_takes_each_points_newest_passing_unsuperseded_run_from_a_fresh_stores_remote(tmp_path):
    """The table is rendered from the store, never from numbers in code: on a fresh machine it finds every point's
    runs on the remote, a newer passing run supersedes an older one, a `superseded_by` label passes a result over, a
    failed run never counts, and the caveat is the result's and its run's `label` and `note`."""

    producer = Producer(tmp_path)
    producer.ran("r20261024-020000-0001", COMMIT_SPEC, "b8-p512-mixed", _result("b8-p512-mixed", 1.0))
    newest_b8 = producer.ran("r20261024-030000-0002", COMMIT_SPEC, "b8-p512-mixed", _result("b8-p512-mixed", 1.25, reps=3))
    producer.ran("r20261024-040000-0003", COMMIT_SPEC, "b8-p512-mixed", None, validation="failed", rc=3)
    kept_b32 = producer.ran("r20261024-020000-0004", COMMIT_SPEC, "b32-p512-mixed", _result("b32-p512-mixed", 1.1))
    replaced = producer.ran("r20261024-030000-0005", COMMIT_SPEC, "b32-p512-mixed", _result("b32-p512-mixed", 9.0))
    producer.label(replaced, "superseded_by", "r20261024-020000-0004", "2026-10-24T05:00:00Z")
    producer.label(newest_b8, "label", "served Commit at b8 | mechanism evidence only", "2026-10-24T05:00:00Z")
    producer.label("r20261024-030000-0002", "note", "no pinned sm_120 build", "2026-10-24T05:00:01Z")

    logged = []
    table = served_table.build(store=_fresh(tmp_path, producer), root=tmp_path, added=lambda spec: ADDED,
                               log=logged.append)

    assert logged == [f"pulled 5 attempts started since r20261023 from {producer.store.remote.describe()} (0 of 5 were here)"]
    rows = _commit_rows(table)
    b8, b32 = rows["b8-p512-mixed"], rows["b32-p512-mixed"]
    assert (b8["run"], b8["art"], b8["reps"]) == ("r20261024-030000-0002", newest_b8, 3)
    assert b8["over_served"]["time_per_output_token_s"]["mean"] == pytest.approx(1.2 * 1.25)
    assert b8["over_served"]["decode_time_per_token_s"]["mean"] == pytest.approx(1.2 * 1.25)
    assert b8["over_served"]["time_per_output_token_s"]["n"] == 3
    assert (b8["own_arm"], b8["eager_stock"]) == ("commit", "stock-fp8")
    assert b8["over_eager"]["time_per_output_token_s"]["mean"] == pytest.approx(1.2 * 1.25)
    assert [(c["key"], c["value"]) for c in b8["caveats"]] == [
        ("label", "served Commit at b8 | mechanism evidence only"), ("note", "no pinned sm_120 build")]
    assert (b32["run"], b32["art"]) == ("r20261024-020000-0004", kept_b32)
    assert b32["over_served"]["time_per_output_token_s"]["mean"] == pytest.approx(1.2 * 1.1)
    assert table["skipped"] == [{"spec": COMMIT_SPEC, "point": "b32-p512-mixed", "run": "r20261024-030000-0005",
                                 "art": replaced, "why": "superseded by r20261024-020000-0004"}]

    others = [row for row in table["rows"] if row["protocol"] != "Commit"] + table["factors"]
    assert others and all(row["run"] is None and row["over_served" if "protocol" in row else "factor"] is None
                          for row in others)

    markdown = served_table.render(table)
    [line] = [line for line in markdown.splitlines() if line.startswith("| Commit |") and "b8-p512-mixed" in line]
    assert "| 1.500× ± 0.000 | 1.500× ± 0.000 | 1.500× ± 0.000 | 3 | r20261024-030000-0002 |" in line
    assert "served Commit at b8 \\| mechanism evidence only / no pinned sm_120 build" in line
    assert "| PoUS bandwidth audit | `llama31-8b-fp8-1gpu-bandwidth-served` | b8-p512-mixed | — | — | — | — | no passing run |" in markdown
    assert "r20261024-030000-0005" in markdown.split("Passed over")[1]


def test_local_only_reads_the_catalog_alone(tmp_path):
    producer = Producer(tmp_path)
    producer.ran("r20261024-020000-0001", COMMIT_SPEC, "b8-p512-mixed", _result("b8-p512-mixed", 1.0))

    def never(spec):
        raise AssertionError("a local-only table reads no remote")

    table = served_table.build(store=_fresh(tmp_path, producer), root=tmp_path, local_only=True, added=never)
    assert all(row["run"] is None for row in table["rows"])


def test_the_served_table_command_prints_markdown_and_json(tmp_path, monkeypatch, capsys):
    producer = Producer(tmp_path)
    producer.ran("r20261024-020000-0001", COMMIT_SPEC, "b8-p512-mixed", _result("b8-p512-mixed", 1.0))
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "fresh"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(producer.config))
    monkeypatch.setattr(served_table, "spec_added", lambda root, spec: ADDED)

    assert main(["served-table"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Served overhead on Llama-3.1-8B") and "| r20261024-020000-0001 |" in out
    assert main(["served-table", "--json"]) == 0
    table = json.loads(capsys.readouterr().out)
    assert table["schema"] == served_table.SCHEMA
    assert _commit_rows(table)["b8-p512-mixed"]["run"] == "r20261024-020000-0001"


def test_every_protocol_row_names_arms_its_specs_run_and_an_eager_stock_only_where_the_spec_is_eager():
    for protocol in (*served_table.PROTOCOLS, served_table.FACTORS):
        for spec in protocol.specs:
            resolved = resolve_spec(spec)
            assert served_table.SERVED in resolved.arms, spec
            assert protocol.arm in resolved.arms, spec
            assert (protocol.own_arm or protocol.arm) in resolved.arms, spec
            assert (protocol.eager_stock is not None) == resolved.eager, spec
            if protocol.eager_stock:
                assert protocol.eager_stock in resolved.arms, spec
            assert resolved.preset.gpu_id == "rtx-pro-6000" and resolved.preset.gpu_count == 1, spec
