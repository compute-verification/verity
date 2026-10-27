from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bench_arm
import step_spans as SS

SETTINGS = ("VERITY_VLLM_BENCH_TOPUP", "VERITY_VLLM_BENCH_SPANS", "VERITY_VLLM_BENCH_KV_RECEIPTS",
            "VERITY_VLLM_BENCH_CODEC_ROUNDS", "VERITY_VLLM_BENCH_TOPUP_SMS")
OLD_SNAPSHOT = {"filler_bytes", "passes", "registered_weight_bytes"}
OLD_EXTRAS = {"filler_bytes", "registered_weight_bytes", "filler_share", "receipt_passes", "filler_ctas"}


class FakeTopup:
    """topup.cu's log and progress as `Topup` reads them: chunks (t0 ms, t1 ms, bytes, ...) finished so far."""

    def __init__(self) -> None:
        self.records: list[tuple] = []
        self.so = SimpleNamespace(tu_progress=self._progress)

    def chunk(self, t0: float, t1: float, nbytes: float = float(512 << 20)) -> None:
        self.records.append((t0, t1, nbytes, 0, 0, len(self.records)))

    def log(self, first: int = 0) -> list[tuple]:
        return self.records[first:]

    def _progress(self, out) -> None:
        out[0] = int(sum(r[2] for r in self.records))


class FakeSpans:
    """The stack's device accumulators, driven by an injected global timer through `step_spans.account`."""

    def __init__(self) -> None:
        self.main, self.side = [0] * SS.SLOTS, [0] * SS.SLOTS
        self.steps, self.tokens = [0, 0], [0, 0]
        self.now = 5 * 10**9

    def decode_step(self, gaps_ns: list[int], batch: int) -> None:
        """One graph step: its first mark sets the kind, each gap ends at the next of the stack's marks."""
        self.steps[0] += 1
        self.tokens[0] += batch
        parts = ["between", "commit", "append_wait", "pass"] + ["model", "linear", "model", "attention",
                                                                 "kv_receipts"] + ["model", "lm_head"]
        for i, part in enumerate(parts):
            self.now += gaps_ns[i % len(gaps_ns)]
            SS.account(self.main, SS.P[part], 0 if i == 0 else -1, self.now)

    def read(self) -> dict:
        out = SS.decode(self.main, self.side, self.steps, self.tokens)
        self.main[0] = self.side[0] = 0
        return out


def _installed(*, encoded: bool = True, topup: bool = True, spans: bool = False) -> bench_arm._Installed:
    """An installed arm's state without a GPU: the stack's install record passed, a fake top-up and spans."""
    state = object.__new__(bench_arm._Installed)
    state.encoded, state.use_topup, state.use_spans, state.ctas = encoded, topup, spans, 32
    gates = {"layers": 128, "configs_checked": 5160, "step": {32: {"ok": True}}}
    stack = SimpleNamespace(linears=[SimpleNamespace(nbytes=1000)], passes=0, kv_receipts=encoded,
                            spans=FakeSpans() if spans else None)
    state.install = SimpleNamespace(record={"gates": gates}, stack=stack)
    state.topup = FakeTopup() if topup else None
    state.log = bench_arm._TopupLog(state.topup)
    state.started, state.closed, state.kernel = True, False, ""
    state.snapshots, state.run_start, state.measured, state.empty_runs = [], None, 0, 0
    state.topup_sms, state.green, state.step_stream, state.off_step_runs = 0, None, None, 0
    return state


def _partitioned(*, confined: bool = True) -> bench_arm._Installed:
    """An installed arm with the top-up on 8 of 188 SMs, as `green.partition` and `spread_checks` record it."""
    state = _installed()
    state.topup_sms = 8
    state.green = {"fill_sms": 8, "step_sms": 180, "min_partition": 8, "alignment": 8, "device_sms": 188,
                   "spread": {"step": 180, "step_graph": 180, "side": 180, "side_graph": 180 if confined else 188,
                              "fill": 8, "device_graph": 188, "confined": confined}}
    return state


def _arm(arm_id: str, state: bench_arm._Installed) -> bench_arm.BandwidthArm:
    arm = bench_arm.BandwidthArm(arm_id, encoded=state.encoded, control="bandwidth-plain")
    arm.state = state
    return arm


def _line(arm_id: str, phase: str = "") -> SimpleNamespace:
    return SimpleNamespace(arm=SimpleNamespace(id=arm_id), phase=phase)


def test_shared_benchmark_arms_name_the_plain_control_and_require_isolation():
    assert (bench_arm.PLAIN.id, bench_arm.PLAIN.control, bench_arm.PLAIN.precision) == (
        "bandwidth-plain",
        None,
        "fp8",
    )
    assert (bench_arm.AUDIT.id, bench_arm.AUDIT.control, bench_arm.AUDIT.precision) == (
        "bandwidth-audit",
        "bandwidth-plain",
        "fp8",
    )
    assert not bench_arm.PLAIN.resident and not bench_arm.AUDIT.resident


def test_library_inputs_fail_closed_and_kernel_digest_binds_contents(tmp_path, monkeypatch):
    for variable in bench_arm._LIB_VARS.values():
        monkeypatch.delenv(variable, raising=False)
    with pytest.raises(RuntimeError, match="VERITY_VLLM_BENCH_WT_LIB"):
        bench_arm._libraries()

    first, second = tmp_path / "first.so", tmp_path / "second.so"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    got = bench_arm._digest([first, second])
    want = hashlib.sha256()
    for path in (first, second):
        name = path.name.encode()
        want.update(len(name).to_bytes(4, "big"))
        want.update(name)
        want.update(path.read_bytes())
    assert got == want.hexdigest()
    second.write_bytes(b"changed")
    assert bench_arm._digest([first, second]) != got


def test_plain_install_gate_does_not_require_encoded_receipt_tree():
    plain = {"gates": {"layers": 128, "configs_checked": 5160}}
    encoded = {"gates": {**plain["gates"], "step": {32: {"ok": True}}}}
    assert bench_arm._install_gate_passed(plain, encoded=False)
    assert not bench_arm._install_gate_passed(plain, encoded=True)
    assert bench_arm._install_gate_passed(encoded, encoded=True)


def test_lm_head_captures_mid_serve_without_emptying_the_allocator_cache():
    """The arms' lm_head captures its CUDA graphs lazily, while vLLM serves. torch.cuda.graph (or any empty_cache) there
    hands vLLM's cached activations back to the driver, and the first timed run after a warm-up cudaMallocs them again
    beside the filler."""
    tree = ast.parse((Path(bench_arm.__file__).parent / "stack.py").read_text())
    lm_head = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LmHead")
    apply = next(n for n in lm_head.body if isinstance(n, ast.FunctionDef) and n.name == "apply")
    calls = {ast.unparse(n.func) for n in ast.walk(apply) if isinstance(n, ast.Call)}
    assert not calls & {"torch.cuda.graph", "torch.cuda.empty_cache"}, (
        "LmHead.apply must capture with CUDAGraph.capture_begin/capture_end under torch.cuda.stream, "
        "not torch.cuda.graph, which empties the allocator's cache")
    assert {"graph.capture_begin", "graph.capture_end"} <= calls


def test_the_ablations_run_under_ids_of_their_own_and_the_defaults_are_the_credited_lines(monkeypatch):
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    assert bench_arm.CREDITED == (bench_arm.PLAIN.id, bench_arm.AUDIT.id)
    want = {
        "PLAIN": ("bandwidth-plain", None, False, True, False),
        "AUDIT": ("bandwidth-audit", "bandwidth-plain", True, True, False),
        "PLAIN_NO_TOPUP": ("bandwidth-plain-no-topup", "bandwidth-plain", False, False, False),
        "AUDIT_NO_TOPUP": ("bandwidth-audit-no-topup", "bandwidth-plain", True, False, False),
        "PLAIN_SPANS": ("bandwidth-plain-spans", "bandwidth-plain", False, True, True),
        "AUDIT_SPANS": ("bandwidth-audit-spans", "bandwidth-plain", True, True, True),
    }
    for name, (arm_id, control, encoded, topup, spans) in want.items():
        arm = getattr(bench_arm, name)
        assert (arm.id, arm.control, arm.encoded, arm.precision, arm.resident) == (arm_id, control, encoded, "fp8", False)
        assert arm.settings() == {"topup": topup, "spans": spans, "kv_receipts": True, "codec_rounds": 3,
                                  "topup_sms": 0}, name


def test_the_environment_sets_only_what_an_arm_leaves_open(monkeypatch):
    monkeypatch.setenv("VERITY_VLLM_BENCH_TOPUP", "off")
    monkeypatch.setenv("VERITY_VLLM_BENCH_SPANS", "off")
    assert not bench_arm.AUDIT.settings()["topup"]
    assert bench_arm.AUDIT_SPANS.settings()["spans"] and bench_arm.AUDIT_SPANS.settings()["topup"] is False
    assert bench_arm.BandwidthArm("x", encoded=True, control=None, topup=True).settings()["topup"]
    monkeypatch.setenv("VERITY_VLLM_BENCH_TOPUP", "no")
    with pytest.raises(RuntimeError, match="VERITY_VLLM_BENCH_TOPUP must be 'on' or 'off'"):
        bench_arm.AUDIT.settings()


def test_the_top_up_sms_are_the_arms_else_the_environments_else_interleaved(monkeypatch):
    monkeypatch.delenv("VERITY_VLLM_BENCH_TOPUP_SMS", raising=False)
    assert bench_arm.AUDIT.settings()["topup_sms"] == int(bench_arm.TOPUP_SMS)
    monkeypatch.setenv("VERITY_VLLM_BENCH_TOPUP_SMS", "16")
    assert bench_arm.AUDIT.settings()["topup_sms"] == bench_arm.PLAIN.settings()["topup_sms"] == 16
    assert bench_arm.BandwidthArm("x", encoded=True, control=None, topup_sms=0).settings()["topup_sms"] == 0
    for bad in ("-8", "eight", ""):
        monkeypatch.setenv("VERITY_VLLM_BENCH_TOPUP_SMS", bad)
        with pytest.raises(RuntimeError, match="VERITY_VLLM_BENCH_TOPUP_SMS must be a nonnegative integer"):
            bench_arm.AUDIT.settings()


def test_a_partition_passes_only_confined_and_with_every_run_on_the_steps_stream():
    """Interleaved, the gate and extras say nothing of a partition; partitioned, they name it, and a stream of the
    step's that ran outside its SMs, or a run whose current stream was not the step's, fails the gate."""
    interleaved = _arm("bandwidth-audit", _installed()).gate(_line("bandwidth-audit"))
    assert interleaved["topup_sms"] == 0 and "partition" not in interleaved
    state = _partitioned()
    arm = _arm("bandwidth-audit", state)
    gate = arm.gate(_line("bandwidth-audit"))
    assert gate["passed"] and gate["topup_sms"] == 8 and gate["partition"]["step_sms"] == 180
    assert gate["partition"]["off_step_runs"] == 0
    arm.spans(_line("bandwidth-audit"))
    state.topup.chunk(0.0, 1.0)
    arm.spans(_line("bandwidth-audit"))
    extras = arm.extras(_line("bandwidth-audit"))
    assert extras.keys() == OLD_EXTRAS | {"topup_sms", "step_sms"}
    assert (extras["topup_sms"], extras["step_sms"]) == (8, 180)
    state.off_step_runs = 1
    assert not arm.gate(_line("bandwidth-audit"))["passed"]
    assert not _arm("bandwidth-audit", _partitioned(confined=False)).gate(_line("bandwidth-audit"))["passed"]


def test_a_run_with_the_top_up_off_never_yields_a_credited_line():
    """A credited id's gate fails (the harness keeps no points), naming the ablation arm; under an id of its own the line
    is kept and says it is not credited. The line is judged by the harness's id for it, so a wrapper that reuses the
    module's gate under another id is judged as that id."""
    for arm_id in bench_arm.CREDITED:
        state = _installed(encoded=arm_id == "bandwidth-audit", topup=False)
        gate = _arm(arm_id, state).gate(_line(arm_id))
        assert not gate["passed"] and not gate["topup"]
        assert not gate["credit"]["credited"] and "AUDIT_NO_TOPUP" in gate["credit"]["fix"]
        assert any("top-up off" in why for why in gate["credit"]["refused"])
    state = _installed(topup=False)
    for arm_id in ("bandwidth-audit-no-topup", "bandwidth-audit-no-filler"):
        gate = _arm("bandwidth-audit", state).gate(_line(arm_id))
        assert gate["passed"] and not gate["credit"]["credited"] and gate["credit"]["refused"]
    on = _arm("bandwidth-audit", _installed()).gate(_line("bandwidth-audit"))
    assert on["passed"] and on["credit"] == {"credited": True, "refused": [], "passes": True}


def test_a_top_up_that_reads_nothing_in_a_measured_run_refuses_credit():
    state = _installed()
    arm = _arm("bandwidth-audit", state)
    arm.begin_run(_line("bandwidth-audit", "gate"))
    arm.end_run(_line("bandwidth-audit", "gate"))          # a gate or warm-up run is not judged
    arm.begin_run(_line("bandwidth-audit", "measure"))
    state.topup.chunk(0.0, 3.0)
    arm.end_run(_line("bandwidth-audit", "measure"))
    assert arm.gate(_line("bandwidth-audit"))["passed"]
    arm.begin_run(_line("bandwidth-audit", "measure"))      # paused: tu_pause, or a top-up stalled
    arm.end_run(_line("bandwidth-audit", "measure"))
    gate = arm.gate(_line("bandwidth-audit"))
    assert not gate["passed"] and gate["measured_runs"] == 2
    assert gate["credit"]["refused"] == ["the top-up read nothing in 1 measured run(s)"]


def test_spans_refuse_credit_and_no_span_enters_a_verdict():
    state = _installed(spans=True)
    assert not _arm("bandwidth-audit", state).gate(_line("bandwidth-audit"))["passed"]
    arm = _arm("bandwidth-audit-spans", state)
    before = arm.gate(_line("bandwidth-audit-spans"))
    for gaps in ([1, 2, 3], [10**9, 1, 10**8]):            # any timings at all
        state.install.stack.spans.decode_step(gaps, 8)
        state.topup.chunk(0.0, 1e9)
        arm.spans(_line("bandwidth-audit-spans"))
        arm.extras(_line("bandwidth-audit-spans"))
    assert arm.gate(_line("bandwidth-audit-spans")) == before and before["passed"] and before["spans"]


def test_with_spans_off_the_arm_reports_what_it_reported_before():
    state = _installed()
    arm = _arm("bandwidth-audit", state)
    assert arm.spans(_line("bandwidth-audit")).keys() == OLD_SNAPSHOT
    for t0 in (0.0, 4.0, 8.0):
        state.topup.chunk(t0, t0 + 3.5, 1000.0 + t0)
    state.install.stack.passes = 64
    after = arm.spans(_line("bandwidth-audit"))
    assert after["filler_bytes"] == sum(record[2] for record in state.topup.log())
    extras = arm.extras(_line("bandwidth-audit"))
    assert extras.keys() == OLD_EXTRAS
    assert extras["registered_weight_bytes"] == 64 * 1000 and extras["receipt_passes"] == 64


def test_spans_reach_the_harness_per_step_and_sum_to_its_steps():
    """The harness's per-rep `spans` is the cumulative snapshot's change over the run, and the arm's extras divide it by
    the run's steps: the main stream's parts sum to its steps' time; the top-up overlaps it."""
    state = _installed(spans=True)
    arm, line, spans = _arm("bandwidth-audit-spans", state), _line("bandwidth-audit-spans"), state.install.stack.spans
    gaps = [120_000, 40_000, 5_000, 30_000, 70_000, 900_000, 60_000, 400_000, 200_000, 80_000, 1_300_000]
    spans.decode_step(gaps, 8)                             # a warm-up, before the run's first read
    arm.spans(line)
    spans.now += 3 * 10**9                                 # the harness between runs
    for _ in range(10):
        spans.decode_step(gaps, 8)
    state.topup.chunk(10.0, 14.0)
    after = arm.spans(line)
    assert {"graph", "eager", "topup"} <= after["span"].keys() and after["span"]["graph"]["steps"] == 11
    extras = arm.extras(line)
    step_ms = sum(gaps) / 1e6
    # the run's first `between` began before the read restarted the clock: the only time not in a part
    assert extras["span_per_step.graph.step_ms"] == pytest.approx(step_ms - gaps[0] / 1e6 / 10, rel=1e-9)
    assert extras["span_per_step.graph.step_ms"] == pytest.approx(step_ms, rel=0.02)
    assert extras["span_per_step.graph.linear_ms"] == pytest.approx(gaps[5] / 1e6)
    assert extras["span.main_ms"] == pytest.approx(10 * extras["span_per_step.graph.step_ms"])
    assert extras["span.topup_busy_ms"] == 4.0
    assert extras["span.topup_busy_share"] == pytest.approx(4.0 / extras["span.main_ms"])
    assert OLD_EXTRAS <= extras.keys()


def test_with_spans_off_the_stack_launches_no_mark():
    """Every mark goes through Stack.mark or the append's guarded side marks, and the stack has spans only when its
    install's args ask: off, vLLM's graphs capture what they captured before."""
    tree = ast.parse((Path(bench_arm.__file__).parent / "stack.py").read_text())
    stack = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Stack")
    mark = next(n for n in stack.body if isinstance(n, ast.FunctionDef) and n.name == "mark")
    assert ast.unparse(mark.body[0].test) == "self.spans is not None"
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    side = [n for n in calls if n.func.attr == "mark_side"]
    guarded = {id(c) for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "spans is not None"
               for c in ast.walk(n)}
    assert side and all(id(n) in guarded for n in side)
    assert not [n for n in calls if n.func.attr == "mark" and ast.unparse(n.func.value) == "self.spans"
                and n not in ast.walk(mark)]
    made = [n for n in ast.walk(tree) if isinstance(n, ast.If) and "getattr(args, 'spans', False)" in ast.unparse(n.test)]
    assert len(made) == 1 and "SS.Spans" in ast.unparse(made[0])
    assert ast.unparse(tree).count("SS.Spans(") == 1
