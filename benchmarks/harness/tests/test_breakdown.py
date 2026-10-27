"""`verity-bench breakdown` (`verity_bench.breakdown`) on fake engines whose arms report the bandwidth audit's spans the
way `bench_arm`'s extras do: the rows sum to the spans arms' audit-over-plain, each row is the part it names, the view
reads local results or the store's newest passing run that times every arm, and no gate or ratio reads a span."""
from __future__ import annotations

import json

import pytest

from test_served_table import ADDED, Producer, _fresh
from verity_bench import breakdown as B
from verity_bench.cli import main
from verity_bench.vllm.arms import Arm
from verity_bench.vllm.presets import expand_sweep, load_spec
from verity_bench.vllm.runner import _token_digest, run_benchmark

TOKENS, GRAPH_STEPS, EAGER_STEPS = 416, 63, 4
# ms a step by part, top-up off: the spans arms with the top-up on stretch every part by 10%
PLAIN_GRAPH = {"linear": 6.0, "lm_head": 0.8, "attention": 0.6, "model": 1.4, "between": 0.12}
AUDIT_GRAPH = {"linear": 15.5, "lm_head": 3.8, "commit": 2.5, "attention": 1.5, "model": 1.45, "kv_receipts": 0.3,
               "between": 0.13, "pass": 0.08}
PLAIN_EAGER = {"prefill_decode": 21.0, "prefill_gemm": 20.0, "lm_head": 0.8}
AUDIT_EAGER = {"prefill_decode": 25.0, "prefill_gemm": 20.0, "lm_head": 8.0, "model": 100.0, "commit": 0.5}
OUTSIDE = {B.PLAIN_OFF: 0.01, B.AUDIT_OFF: 0.03}         # ms a token in no part
CONTENTION = {B.PLAIN_SPANS: 0.1, B.AUDIT_SPANS: 1.9}    # ms a token the top-up adds beside the stretch
CREDITED = {B.PLAIN: 1.8, B.AUDIT: 7.2}


def _spanned(graph, eager):
    return (sum(graph.values()) * GRAPH_STEPS + sum(eager.values()) * EAGER_STEPS) / TOKENS


def _shape(arm_id):
    """An arm's parts a step by kind and its time per output token (ms), or None for a credited arm (no spans)."""

    if arm_id in CREDITED:
        return None
    audit = "audit" in arm_id
    graph, eager = (AUDIT_GRAPH, AUDIT_EAGER) if audit else (PLAIN_GRAPH, PLAIN_EAGER)
    if arm_id in CONTENTION:
        graph, eager = ({k: 1.1 * v for k, v in parts.items()} for parts in (graph, eager))
        off = B.AUDIT_OFF if audit else B.PLAIN_OFF
        tpot = _spanned(AUDIT_GRAPH if audit else PLAIN_GRAPH, AUDIT_EAGER if audit else PLAIN_EAGER) + OUTSIDE[off] \
            + CONTENTION[arm_id]
    else:
        tpot = _spanned(graph, eager) + OUTSIDE[arm_id]
    return graph, eager, tpot


class SpanEngine:
    def __init__(self, arm, span_scale):
        self.arm, self.span_scale, self.reps = arm, span_scale, 0
        self.identity = {"vllm_version": "0.fake+gd9105ea80", "kernel_digest": None, "engine_overrides": {},
                         "kv_cache_tokens": 1}

    def gate(self, point):
        return {f"{point.id}-0": (7, 9)}

    def custom_gate(self):
        return {"passed": True, "provided": True}

    def warmup(self, point):
        return {"wall_s": 0.5}

    def measure(self, point):
        self.reps += 1
        drift = 1 + self.reps / 100           # every arm slows alike rep to rep: the rows pair by rep
        shape = _shape(self.arm.id)
        extras, spans = {"filler_share": 0.0 if "no-topup" in self.arm.id else 0.7}, {}
        if shape is None:
            tpot = CREDITED[self.arm.id]
        else:
            graph, eager, tpot = shape
            for kind, parts, steps in (("graph", graph, GRAPH_STEPS), ("eager", eager, EAGER_STEPS)):
                head = f"span_per_step.{kind}."
                extras.update({f"{head}{part}_ms": self.span_scale * ms for part, ms in parts.items()})
                extras.update({f"{head}step_ms": self.span_scale * sum(parts.values()), f"{head}steps": float(steps),
                               f"{head}append_ms": 5.0, f"{head}marks": 358.0,
                               f"{head}tokens_per_step": 8.0 if kind == "graph" else 643.0})
                spans[kind] = {"ms": {part: self.span_scale * ms * steps for part, ms in parts.items()}}
            extras["span.topup_busy_share"] = 0.0 if "no-topup" in self.arm.id else 0.9
        tpot_s = tpot * drift / 1e3
        return {"token_digest": _token_digest({f"{point.id}-0": (7, 9)}), "generated_tokens": TOKENS,
                "metrics": {"time_per_output_token_s": tpot_s, "time_to_first_token_s": 1.0, "tokens_per_s": 1 / tpot_s,
                            "peak_gpu_memory_bytes": 1.0},
                "spans": {"span": spans}, "extras": extras}

    def close(self):
        pass


class SpanFactory:
    def __init__(self, span_scale=1.0):
        self.span_scale = span_scale

    def create(self, arm, preset, *, resident):
        return SpanEngine(arm, self.span_scale)


def _result(point_id="b8-p512-mixed", *, arms=B.ARMS, span_scale=1.0, reps=2):
    [point] = [p for p in expand_sweep("quick") if p.id == point_id]
    resolved = tuple(Arm(arm, control=None if arm == B.PLAIN else B.PLAIN) for arm in arms)
    return run_benchmark(load_spec(B.SPEC).preset, resolved, (point,), sweep="quick", reps=reps,
                         factory=SpanFactory(span_scale), git_commit="c" * 40)


def _rows(entry):
    return {row["row"]: row["ms_per_token"]["mean"] for row in entry["rows"]}


def test_the_rows_sum_to_the_spans_arms_audit_over_plain_and_each_is_the_part_it_names():
    [row] = _result(reps=4)["points"]
    got = B.attribution(row)
    drift = sum(1 + rep / 100 for rep in range(1, 5)) / 4
    per_token = lambda ms, steps: ms * steps / TOKENS  # noqa: E731
    total = (_shape(B.AUDIT_SPANS)[2] - _shape(B.PLAIN_SPANS)[2]) * drift
    assert got["total"]["mean"] == pytest.approx(total) and got["total"]["n"] == 4
    assert got["sum"] == pytest.approx(total)
    assert got["credited"]["mean"] == pytest.approx((CREDITED[B.AUDIT] - CREDITED[B.PLAIN]) * drift)
    rows = _rows(got)
    assert list(rows) == ["top-up contention", "decode linear", "decode lm_head", "decode commit", "decode attention",
                          "decode, the rest", "prefill steps", "outside every span"]
    assert rows["top-up contention"] == pytest.approx((CONTENTION[B.AUDIT_SPANS] - CONTENTION[B.PLAIN_SPANS]) * drift)
    # every part from the top-up-off arms, whose spans the top-up doesn't stretch
    assert rows["decode linear"] == pytest.approx(per_token(15.5 - 6.0, GRAPH_STEPS))
    assert rows["decode commit"] == pytest.approx(per_token(2.5, GRAPH_STEPS))
    assert rows["decode, the rest"] == pytest.approx(per_token(1.45 + 0.3 + 0.13 + 0.08 - 1.4 - 0.12, GRAPH_STEPS))
    assert rows["prefill steps"] == pytest.approx(per_token(sum(AUDIT_EAGER.values()) - sum(PLAIN_EAGER.values()),
                                                            EAGER_STEPS))
    # the time in no part, here the fakes' OUTSIDE scaled by the run's drift with the rest of the time per token
    expected_outside = total - sum(value for name, value in rows.items() if name != "outside every span")
    assert rows["outside every span"] == pytest.approx(expected_outside)
    shares = {row["row"]: row["share"] for row in got["rows"]}
    assert sum(shares.values()) == pytest.approx(1.0)


def test_each_spans_arms_step_by_part_is_its_reps_mean():
    [row] = _result()["points"]
    entry = B.point(row, {"result": "x"})
    assert list(entry["arms"]) == list(B.SPANS_ARMS)
    audit = entry["arms"][B.AUDIT_OFF]
    assert audit["kinds"]["graph"]["parts_ms"] == pytest.approx(AUDIT_GRAPH)
    assert audit["kinds"]["eager"]["parts_ms"] == pytest.approx(AUDIT_EAGER)
    assert (audit["kinds"]["graph"]["steps"], audit["kinds"]["eager"]["tokens_per_step"]) == (GRAPH_STEPS, 643.0)
    assert audit["topup_busy_share"] == 0.0 and entry["arms"][B.AUDIT_SPANS]["topup_busy_share"] == 0.9
    assert entry["arms"][B.PLAIN_SPANS]["kinds"]["graph"]["parts_ms"]["linear"] == pytest.approx(6.6)


def test_spans_enter_no_gate_and_no_ratio():
    """The same runs with every span a thousand times larger: the gates, every metric and every ratio are equal; only
    the spans and the extras that carry them differ."""

    small, large = _result(span_scale=1.0), _result(span_scale=1000.0)
    assert small["gates"] == large["gates"] and all(g["passed"] for g in small["gates"])
    for arm in B.ARMS:
        a, b = small["points"][0]["arms"][arm], large["points"][0]["arms"][arm]
        assert a["metrics"] == b["metrics"]
        if arm in B.SPANS_ARMS:
            assert a["span_totals"] != b["span_totals"] and a["extras"] != b["extras"]
    assert B.attribution(small["points"][0])["rows"][0] == B.attribution(large["points"][0])["rows"][0]


def test_an_arm_without_spans_or_a_missing_arm_is_refused():
    [row] = _result(arms=B.ARMS[:-1])["points"]
    with pytest.raises(ValueError, match=f"no timed {B.AUDIT_OFF}"):
        B.attribution(row)
    [row] = _result()["points"]
    for rep in row["arms"][B.AUDIT_SPANS]["reps"]:
        rep["extras"] = {"filler_share": 0.7}
    with pytest.raises(ValueError, match=f"{B.AUDIT_SPANS} at b8-p512-mixed rep 0 recorded no spans"):
        B.attribution(row)


def test_the_breakdown_command_reads_result_files(tmp_path, capsys):
    paths = []
    for point in ("b8-p512-mixed", "b32-p512-mixed"):
        paths.append(tmp_path / f"{point}.json")
        paths[-1].write_text(json.dumps(_result(point)))
    assert main(["breakdown", *map(str, paths), "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema"] == B.SCHEMA and [p["point"] for p in doc["points"]] == ["b8-p512-mixed", "b32-p512-mixed"]
    assert main(["breakdown", str(paths[0])]) == 0
    markdown = capsys.readouterr().out
    assert "| top-up contention |" in markdown and "| **step** |" in markdown and "ratio" not in markdown.split("\n## ")[1]


def test_the_breakdown_takes_each_points_newest_passing_run_that_times_every_arm_from_the_remote(tmp_path):
    producer = Producer(tmp_path)
    kept = producer.ran("r20261024-020000-0001", B.SPEC, "b8-p512-mixed", _result("b8-p512-mixed"))
    old_arms = producer.ran("r20261024-030000-0002", B.SPEC, "b8-p512-mixed",
                            _result("b8-p512-mixed", arms=(B.PLAIN, B.AUDIT)))
    table = B.build(store=_fresh(tmp_path, producer), root=tmp_path, added=lambda spec: ADDED, log=lambda _: None)
    [b8] = table["points"]
    assert (b8["point"], b8["source"]["run"], b8["source"]["art"]) == ("b8-p512-mixed", "r20261024-020000-0001", kept)
    assert [(s["run"], s["art"], s["why"]) for s in table["skipped"]] == [
        ("r20261024-030000-0002", old_arms, f"no timed {', '.join(B.SPANS_ARMS)} at b8-p512-mixed"),
        (None, None, "no passing run")]
    assert table["skipped"][1]["point"] == "b32-p512-mixed"
    assert "r20261024-020000-0001" in B.render(table)


def test_the_served_table_puts_a_records_breakdown_beside_the_protocols_row(tmp_path):
    """`served_table.BREAKDOWNS` reads the breakdown at each run of its labelled record art, not the newest run, and
    passes over a record run without the spans arms."""

    from verity_bench import served_table

    producer = Producer(tmp_path)
    producer.ran("r20261024-020000-0001", B.SPEC, "b8-p512-mixed", _result("b8-p512-mixed", reps=3))
    producer.ran("r20261024-020000-0002", B.SPEC, "b32-p512-mixed", _result("b32-p512-mixed", arms=(B.PLAIN, B.AUDIT)))
    producer.ran("r20261024-030000-0003", B.SPEC, "b8-p512-mixed", _result("b8-p512-mixed", span_scale=2.0))
    record = producer.table({"b8-p512-mixed": "r20261024-020000-0001", "b32-p512-mixed": "r20261024-020000-0002"},
                            "split")
    producer.label(record, "run_of_record", "split", "2026-10-24T05:00:00Z")
    breakdowns = (served_table.Breakdown("PoUS bandwidth audit", "split"),)

    def never(spec):
        raise AssertionError("a table of records pulls no catalog")

    table = served_table.build(store=_fresh(tmp_path, producer), root=tmp_path, added=never, protocols=(),
                               factors=served_table.Protocol("factor", (), served_table.SERVED, "stock-fp8"),
                               breakdowns=breakdowns)

    [b8] = table["breakdowns"]
    assert (b8["protocol"], b8["point"], b8["run"], b8["record"], b8["reps"]) == (
        "PoUS bandwidth audit", "b8-p512-mixed", "r20261024-020000-0001", record, 3)
    assert b8["sum"] == pytest.approx(b8["total"]["mean"])
    assert _rows(b8) == pytest.approx(_rows(B.attribution(_result("b8-p512-mixed", reps=3)["points"][0])))
    [skipped] = [s for s in table["skipped"] if s["run"] == "r20261024-020000-0002"]
    assert skipped["why"] == f"no timed {', '.join(B.SPANS_ARMS)} at b32-p512-mixed"

    markdown = served_table.render(table)
    section = markdown.split("## Where the bandwidth audit's time goes")[1]
    assert "| row | b8-p512-mixed, ms an output token | share |" in section
    assert "| decode linear: the weights' codec decode" in section
    assert "run r20261024-020000-0001" in section and f"(record `{record[:16]}`)" in section
