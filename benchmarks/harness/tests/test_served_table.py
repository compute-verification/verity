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
COMMIT = served_table.Protocol("Commit", (COMMIT_SPEC,), "commit@compiled", "stock-fp8", own_arm="commit")
BY_SPEC = {
    "protocols": (served_table.Protocol("PoUS bandwidth audit", ("llama31-8b-fp8-1gpu-bandwidth-served-r6",),
                                        "bandwidth-audit", None), COMMIT),
    "factors": served_table.Protocol("conversion factor", ("llama31-8b-fp8-1gpu-compiled-full",), served_table.SERVED,
                                     "stock-fp8"),
    "breakdowns": (),
}
LIVE_ARM, LIVE_CONTROL = "dither-a@x", "exact-fused"
# Time per output token by arm; every arm slows by the same factor each rep, so the paired ratios are exact.
SCALE = {"stock-fp8": 2.0, "commit": 2.4, "stock-fp8@compiled": 1.0, "commit@compiled": 1.2}


class ScaledEngine:
    def __init__(self, arm, scale, gpu):
        self.arm, self.scale, self.reps = arm, scale, 0
        self.identity = {"vllm_version": "0.fake+gd9105ea80", "kernel_digest": None, "engine_overrides": {}, "kv_cache_tokens": 1,
                         "gpu": gpu}

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
    def __init__(self, commit_scale, gpu):
        self.commit_scale, self.gpu = commit_scale, gpu

    def create(self, arm, preset, *, resident):
        return ScaledEngine(arm, self.commit_scale if arm.id.startswith("commit") else 1.0, self.gpu)


RTX, H200 = "NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA H200"


def _result(point_id: str, commit_scale: float, reps: int = 2, gpu: str = RTX) -> dict:
    [point] = [p for p in expand_sweep("quick") if p.id == point_id]
    arms = (Arm("stock-fp8"), Arm("commit", control="stock-fp8"), Arm("stock-fp8@compiled"),
            Arm("commit@compiled", control="stock-fp8@compiled"))
    return run_benchmark(load_spec(COMMIT_SPEC).preset, arms, (point,), sweep="quick", reps=reps,
                         factory=ScaledFactory(commit_scale, gpu), git_commit="c" * 40)


class Producer:
    """A machine that ran points: it publishes each result, its attempt and its labels to the remote directory."""

    def __init__(self, tmp_path: Path):
        from research.store.local import LocalStore

        self.tmp_path, self.remote = tmp_path, tmp_path / "remote"
        (self.remote / "attempts").mkdir(parents=True)
        self.config = tmp_path / "store.toml"
        self.config.write_text(f'[remote]\ntype = "fs"\npath = "{self.remote}"\n')
        self.store = LocalStore(tmp_path / "producer", config_path=self.config)

    def _tree(self, run: str, files: dict[str, str]) -> str:
        root = self.tmp_path / "files" / run
        for path, text in files.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(text)
        art = self.store.put_artifact("run-files/v1", {}, tree=root)
        self.store.push(art)
        return art

    def _attempt(self, run: str, tool: str, params: dict, outputs: dict, *, validation, rc, machine=None) -> None:
        from research.store import canon

        start = f"{run[1:5]}-{run[5:7]}-{run[7:9]}T{run[10:12]}:{run[12:14]}:{run[14:16]}Z"
        attempt = {
            "schema": "research/attempt/v0.1", "id": run, "tool": {"name": tool, "version": "0"},
            "derivation": "drv:" + "0" * 64, "reuse_key": None, "argv": [], "params": params,
            "key_params": ["identity"] if "identity" in params else [], "inputs": {}, "conditions": {},
            "key_conditions": [], "source": {"commit": "c" * 40, "dirty": False}, "closure": None,
            "execution": {"start_utc": start, **({"machine": machine} if machine else {})},
            "cost": None, "status": {"state": "done" if rc == 0 else "failed", "rc": rc, "validation": validation},
            "outputs": outputs, "telemetry": None,
        }
        (self.remote / "attempts" / f"{run}.json").write_bytes(canon.dumps(attempt))

    def ran(self, run: str, spec: str, point: str, doc: dict | None, *, validation="passed", rc=0) -> str | None:
        art = None
        if doc is not None:
            files = self._tree(run, {"out/benchmark.json": json.dumps(doc)})
            art = self.store.put_artifact("verity-vllm-benchmark/v1", {"run_id": run}, {"run_files": files})
            self.store.push(art)
        self._attempt(run, "vllm_bench_point",
                      {"spec": served_table.spec_path(spec), "point": point, "identity": "1" * 64},
                      {"result": art} if art else {}, validation=validation, rc=rc)
        return art

    def table(self, runs, label: str) -> str:
        """A table art whose meta names its runs, as a record's table arts do."""

        path = self.tmp_path / "files" / f"table-{label}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(label)
        art = self.store.put_artifact("verity-bench-table/v1", {"runs": runs}, file=path)
        self.store.push(art)
        return art

    def live(self, run: str, summary: dict, stop: dict, *, rc=0) -> str:
        """A live PoUW run: its run record holds sbench's summary and stop records under out/."""

        record = self._tree(run, {"out/summary.json": json.dumps(summary), "out/stop.json": json.dumps(stop)})
        self._attempt(run, "bash", {}, {"result": None, "run_record": record}, validation=None, rc=rc,
                      machine="vy-dc-h200")
        return record

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
                               log=logged.append, **BY_SPEC)

    remote = producer.store.remote.describe()
    assert logged == [f"pulled 3 labels from {remote} (0 of 3 were here)",
                      f"pulled 5 attempts started since r20261023 from {remote} (0 of 5 were here)"]
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
    assert table["live"] == []

    others = [row for row in table["rows"] if row["protocol"] != "Commit"] + table["factors"]
    assert others and all(row["run"] is None and row["over_served" if "protocol" in row else "factor"] is None
                          for row in others)

    markdown = served_table.render(table)
    [line] = [line for line in markdown.splitlines() if line.startswith("| Commit |") and "b8-p512-mixed" in line]
    assert line.startswith(f"| Commit | {RTX} | `{COMMIT_SPEC}` | b8-p512-mixed |")
    assert "| 1.500× ± 0.000 | 1.500× ± 0.000 | 1.500× ± 0.000 | 3 | r20261024-030000-0002 |" in line
    assert "served Commit at b8 \\| mechanism evidence only / no pinned sm_120 build" in line
    assert ("| PoUS bandwidth audit | — | `llama31-8b-fp8-1gpu-bandwidth-served-r6` | b8-p512-mixed | — | — | — | — "
            "| no passing run |") in markdown
    assert "r20261024-030000-0005" in markdown.split("Passed over")[1]


def test_a_labelled_run_is_its_protocols_row_over_a_newer_unlabelled_one(tmp_path):
    """A record replaces the newest-run lookup of the same spec and point, while a protocol without a labelled record
    still takes the newest passing run."""

    producer = Producer(tmp_path)
    producer.ran("r20261024-020000-0001", COMMIT_SPEC, "b8-p512-mixed", _result("b8-p512-mixed", 1.0))
    producer.ran("r20261024-030000-0002", COMMIT_SPEC, "b8-p512-mixed", _result("b8-p512-mixed", 1.25))
    producer.label("r20261024-020000-0001", "run_of_record", "commit", "2026-10-24T05:00:00Z")
    of_record = served_table.Protocol("of record", (COMMIT_SPEC,), "commit@compiled", "stock-fp8", own_arm="commit",
                                      record="commit")

    table = served_table.build(store=_fresh(tmp_path, producer), root=tmp_path, added=lambda spec: ADDED,
                               log=lambda _: None, protocols=(of_record, COMMIT), factors=BY_SPEC["factors"],
                               breakdowns=())
    assert [(row["point"], row["run"]) for row in table["rows"] if row["protocol"] == "of record"] == [
        ("b8-p512-mixed", "r20261024-020000-0001")]
    assert _commit_rows(table)["b8-p512-mixed"]["run"] == "r20261024-030000-0002"


def test_local_only_reads_the_catalog_alone(tmp_path):
    producer = Producer(tmp_path)
    producer.ran("r20261024-020000-0001", COMMIT_SPEC, "b8-p512-mixed", _result("b8-p512-mixed", 1.0))

    def never(spec):
        raise AssertionError("a local-only table reads no remote")

    table = served_table.build(store=_fresh(tmp_path, producer), root=tmp_path, local_only=True, added=never,
                               **BY_SPEC)
    assert all(row["run"] is None for row in table["rows"])


def test_a_run_of_record_is_a_labelled_run_or_a_labelled_arts_runs_from_a_fresh_stores_remote(tmp_path):
    """A protocol's record replaces the newest-run lookup: the runs labelled `run_of_record=NAME`, and the runs a
    labelled art's `runs` meta names (a list, or a map from point to run, read in sweep order), in label order, are read
    from the remote without a catalog pull, every point of each is a row, the record art's labels are caveats beside the
    run's, a target's latest label names its record, and a record run that isn't a passing point run is listed with
    why rather than read."""

    producer = Producer(tmp_path)
    b32 = producer.ran("r20261024-020000-0001", COMMIT_SPEC, "b32-p512-mixed", _result("b32-p512-mixed", 1.1))
    producer.ran("r20261024-020000-0002", COMMIT_SPEC, "b8-p512-mixed", _result("b8-p512-mixed", 1.25, gpu=H200))
    newer = producer.ran("r20261024-030000-0003", COMMIT_SPEC, "b8-p512-mixed", _result("b8-p512-mixed", 9.0))
    producer.ran("r20261024-040000-0004", COMMIT_SPEC, "b8-p512-mixed", None, validation="failed", rc=3)
    listed = producer.table(["r20261024-020000-0001"], "listed")
    mapped = producer.table({"b8-p512-mixed": "r20261024-020000-0002", "b32-p512-mixed": "r20261024-020000-0001"},
                            "mapped")
    producer.label(listed, "label", "prices it, doesn't credit it", "2026-10-24T05:00:00Z")
    for target, record, ts in ((listed, "by-art", "05:00:01"), (mapped, "by-map", "05:00:02"),
                               ("r20261024-020000-0001", "factor", "05:00:03"), (newer, "by-id", "05:00:04"),
                               ("r20261024-040000-0004", "by-id", "05:00:05"),
                               ("r20261024-020000-0002", "by-id", "05:00:06"), (newer, "retired", "05:00:07")):
        producer.label(target, "run_of_record", record, f"2026-10-24T{ts}Z")
    protocols = (
        served_table.Protocol("by art", (), "commit@compiled", "stock-fp8", own_arm="commit", record="by-art"),
        served_table.Protocol("by map", (), "commit@compiled", None, record="by-map"),
        served_table.Protocol("by id", (), "commit@compiled", None, record="by-id"),
    )

    def never(spec):
        raise AssertionError("a table of records pulls no catalog")

    table = served_table.build(store=_fresh(tmp_path, producer), root=tmp_path, added=never, protocols=protocols,
                               factors=served_table.Protocol("factor", (), served_table.SERVED, "stock-fp8",
                                                             record="factor"), breakdowns=())

    rows = {(row["protocol"], row["point"]): row for row in table["rows"]}
    art_row = rows["by art", "b32-p512-mixed"]
    assert (art_row["run"], art_row["art"], art_row["record"], art_row["spec"]) == (
        "r20261024-020000-0001", b32, listed, COMMIT_SPEC)
    assert art_row["over_served"]["time_per_output_token_s"]["mean"] == pytest.approx(1.2 * 1.1)
    assert art_row["over_eager"]["time_per_output_token_s"]["mean"] == pytest.approx(1.2 * 1.1)
    assert [(c["target"], c["value"]) for c in art_row["caveats"]] == [(listed, "prices it, doesn't credit it")]
    assert [point for protocol, point in rows if protocol == "by map"] == ["b8-p512-mixed", "b32-p512-mixed"]
    assert rows["by map", "b8-p512-mixed"]["over_eager"] is None
    by_id = rows["by id", "b8-p512-mixed"]
    assert (by_id["run"], by_id["record"], by_id["device"], art_row["device"]) == ("r20261024-020000-0002", None, H200, RTX)
    assert by_id["run"] != newer and by_id["over_served"]["time_per_output_token_s"]["mean"] == pytest.approx(1.5)
    assert [point for protocol, point in rows if protocol == "by id"] == ["—", "b8-p512-mixed"]
    assert rows["by id", "—"]["run"] is None
    [failed] = table["skipped"]
    assert (failed["run"], failed["spec"]) == ("r20261024-040000-0004", COMMIT_SPEC)
    assert failed["why"].startswith("not a passing vllm_bench_point run (tool vllm_bench_point, validation failed, rc 3")
    [factor] = table["factors"]
    assert factor["factor"]["time_per_output_token_s"]["mean"] == pytest.approx(0.5)

    markdown = served_table.render(table)
    assert f"| r20261024-020000-0001 | `{b32[:16]}` (record `{listed[:16]}`) | prices it, doesn't credit it |" in markdown
    assert f"| by id | {H200} | `{COMMIT_SPEC}` | b8-p512-mixed | 1.500× ± 0.000 |" in markdown
    assert "r20261024-040000-0004" in markdown.split("Passed over")[1]


def _live_summary(steps, controls, *, live=True, tokens_equal=True):
    """sbench's summary.json and stop.json for one PoUW arm beside its control, as the live runs write them."""

    summary = {
        "arms": ["exact", LIVE_CONTROL, LIVE_ARM], "b": 32, "gate_failed": [], "ref_of": {LIVE_ARM: "exact"},
        "gate": {LIVE_ARM: {"check": {"ok": True, "complete": True, "t": 64, "form_basis": "measured only"}}},
        "gate_tokens_equal_ref_arm": {LIVE_ARM: [True, True, True]},
        "decode_tokens_vs_ref": {LIVE_ARM: [{"requests_equal": 32 if tokens_equal else 31, "first_diff": []}] * 3},
        "per_arm": {LIVE_ARM: {"tpot_all": steps}, LIVE_CONTROL: {"tpot_all": controls}, "exact": {"tpot_all": [1.0]}},
        "stop": {LIVE_ARM: {"error": None, "close_error": None,
                            "end_verdict": {"ok": True, "live_ok": True} if live else None}},
    }
    coins = {"end": {"live": {"coins": {"source": "off-node", "ok": True}}}} if live else {}
    return summary, {LIVE_ARM: {"stopped": True, **coins}}


def test_a_live_pouw_row_is_its_runs_median_decode_step_over_its_controls_and_stands_only_on_live_gated_runs(tmp_path):
    """Kernels' live rows are read from each run's record, never typed: the runs are the entries of the `runs` meta of
    an art labelled `run_of_record=pouw-live`, the ratio is the arm's median decode step over its control's in the same
    run, the run's labels are its caveat, and a run whose coins weren't live off-node or whose tokens differ from the
    int8-exact engine's is passed over with why."""

    producer = Producer(tmp_path)
    record = producer.live("r20261024-210000-0001", *_live_summary([16.44, 16.49, 16.48], [3.74, 3.744, 3.78]))
    producer.live("r20261024-210000-0002", *_live_summary([14.53, 14.29, 14.52], [3.04, 3.06, 3.05], live=False))
    producer.live("r20261024-210000-0003", *_live_summary([14.0, 14.0, 14.0], [3.0, 3.0, 3.0], tokens_equal=False))
    producer.label("r20261024-210000-0001", "label", "H200 gated live: gamma 4.40", "2026-10-24T21:41:42Z")
    runs = ("r20261024-210000-0001", "r20261024-210000-0002", "r20261024-210000-0003", "r20991231-000000-0000")
    live = producer.table([{"device": "H200", "run": run, "arm": LIVE_ARM, "control": LIVE_CONTROL} for run in runs],
                          "live")
    producer.label(live, "run_of_record", served_table.LIVE, "2026-10-24T21:41:43Z")

    table = served_table.build(store=_fresh(tmp_path, producer), root=tmp_path, protocols=(),
                               factors=served_table.Protocol("factor", (), served_table.SERVED, "stock-fp8"),
                               breakdowns=())

    [row] = table["live"]
    assert (row["run"], row["record"], row["machine"], row["batch"], row["t"], row["reps"], row["coins"]) == (
        "r20261024-210000-0001", record, "vy-dc-h200", 32, 64, 3, "off-node")
    assert (row["step_ms"], row["control_ms"]) == (16.48, 3.744)
    assert row["ratio"] == pytest.approx(16.48 / 3.744)
    assert [c["value"] for c in row["caveats"]] == ["H200 gated live: gamma 4.40"]
    why = {s["run"]: s["why"] for s in table["skipped"]}
    assert why["r20261024-210000-0002"] == ("no passing live window at the run's end (coins not live); coins absent, "
                                            "not passing off-node coins")
    assert why["r20261024-210000-0003"] == "timed tokens differ from exact's"
    assert why["r20991231-000000-0000"] == "not in the store or on its remote"

    markdown = served_table.render(table)
    assert (f"| H200 | vy-dc-h200 | 32 | 64 | 16.480 | `{LIVE_CONTROL}` | 3.744 | 4.402× | 3 | off-node | "
            f"r20261024-210000-0001 | `{record[:16]}` | H200 gated live: gamma 4.40 |") in markdown
    assert "per step, not per token" in markdown
    with pytest.raises(ValueError, match="label that art instead"):
        served_table.live_runs(None, ["r20261024-210000-0001"])


def test_the_served_table_command_prints_markdown_and_json(tmp_path, monkeypatch, capsys):
    producer = Producer(tmp_path)
    producer.ran("r20261024-020000-0001", COMMIT_SPEC, "b8-p512-mixed", _result("b8-p512-mixed", 1.0))
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "fresh"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(producer.config))
    monkeypatch.setattr(served_table, "spec_added", lambda root, spec: ADDED)
    monkeypatch.setattr(served_table, "PROTOCOLS", BY_SPEC["protocols"])
    monkeypatch.setattr(served_table, "FACTORS", BY_SPEC["factors"])
    monkeypatch.setattr(served_table, "BREAKDOWNS", BY_SPEC["breakdowns"])

    assert main(["served-table"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Served overhead on Llama-3.1-8B") and "| r20261024-020000-0001 |" in out
    assert main(["served-table", "--json"]) == 0
    table = json.loads(capsys.readouterr().out)
    assert table["schema"] == served_table.SCHEMA
    assert _commit_rows(table)["b8-p512-mixed"]["run"] == "r20261024-020000-0001"


def test_every_record_has_its_own_name_in_the_vocabularys_form():
    from research.store import vocab

    names = [entry.record for entry in (*served_table.PROTOCOLS, served_table.FACTORS, *served_table.BREAKDOWNS)]
    assert len(set(names + [served_table.LIVE])) == len(names) + 1
    assert all(vocab.check(served_table.RUN_OF_RECORD, name) is None for name in [*names, served_table.LIVE])


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
