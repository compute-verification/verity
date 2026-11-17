"""The arms and their engines: reference and experimental arms, the engine process, warmups and timed reps, the gates,
kv pools and serving defaults."""
from __future__ import annotations

import gc
import logging
import sys
import tempfile
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest
from bench_standins import FakeFactory, inputs
from verity_vllm.engine.env import INDUCTOR_DIR_MARK

from verity_bench import settings
from verity_bench.vllm.arms import (
    build_bound,
    resolve_arms,
)
from verity_bench.vllm.engine import (
    ProcessEngineFactory,
    _kv_cache_tokens,
    _ProcessEngine,
    engine_args,
)
from verity_bench.vllm.presets import (
    SweepPoint,
    expand_sweep,
    load_spec,
    local_checkpoint,
)
from verity_bench.vllm.result import (
    validate,
)
from verity_bench.vllm.runner import (
    _warm,
    run_benchmark,
)
from verity_bench.vllm.timing import (
    HostProbe,
    run_trace,
)


def test_stock_precisions_are_independent_reference_arms():
    arms = resolve_arms("stock,stock-fp8")
    assert [(arm.id, arm.precision, arm.control) for arm in arms] == [
        ("stock", "bfloat16", None),
        ("stock-fp8", "fp8", None),
    ]


def test_experimental_arm_can_require_an_isolated_engine(monkeypatch):
    module = ModuleType("bench_isolated_arm")
    module.ARM = SimpleNamespace(id="isolated", install=lambda target: None, resident=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    arm, = resolve_arms("isolated", (f"{module.__name__}:ARM",))
    assert arm.experimental and not arm.resident


def test_a_protocol_entry_that_is_refused_binds_no_build_and_says_so(caplog):
    unknown, _control = resolve_arms("no-such-protocol:x,stock-fp8")
    with caplog.at_level(logging.INFO, logger="verity_bench.vllm.arms"):
        assert build_bound(unknown) == ()
    assert "'no-such-protocol:x' refused" in caplog.text and "the engine's build refuses it" in caplog.text


def test_a_dead_engine_process_reports_its_own_error_not_a_connection_reset():
    class DeadConn:
        def __init__(self, final=None):
            self.final = final

        def send(self, _):
            raise BrokenPipeError

        def poll(self, _timeout):
            return self.final is not None

        def recv(self):
            return self.final

    class DeadProcess:
        exitcode = -9

        def join(self):
            pass

    point = SweepPoint(batch=8, prompt_tokens=512, traffic="mixed", output_tokens=64)
    drained = _ProcessEngine(DeadProcess(), DeadConn({"error": "RuntimeError: CUDA out of memory", "trace": ""}), {})
    with pytest.raises(RuntimeError, match="died during gate: RuntimeError: CUDA out of memory"):
        drained.gate(point)
    silent = _ProcessEngine(DeadProcess(), DeadConn(), {})
    with pytest.raises(RuntimeError, match=r"exit code -9 \(SIGKILL: likely out of memory; its own last error is on the job's stderr\)$"):
        silent.gate(point)

    class TornConn(DeadConn):
        def poll(self, _timeout):
            raise ConnectionResetError

    torn = _ProcessEngine(DeadProcess(), TornConn(), {})
    with pytest.raises(RuntimeError, match=r"exit code -9 .*; its last message could not be read \(ConnectionResetError\)$"):
        torn.gate(point)


def test_an_engines_kv_pool_is_its_blocks_times_their_size_and_none_where_the_engine_hides_either():
    def llm(cache):
        return SimpleNamespace(vllm=SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=SimpleNamespace(cache_config=cache))))

    assert _kv_cache_tokens(llm(SimpleNamespace(num_gpu_blocks=28_672, block_size=16))) == 458_752
    assert _kv_cache_tokens(llm(SimpleNamespace(num_gpu_blocks=None, block_size=16))) is None
    assert _kv_cache_tokens(llm(SimpleNamespace(block_size=16))) is None
    assert _kv_cache_tokens(SimpleNamespace(vllm=SimpleNamespace())) is None


def test_closing_an_engine_whose_worker_already_sent_its_error_keeps_that_error(capsys):
    """A gate that fails in the worker (an arm's refusal) ends the worker's loop; the runner's cleanup then closes the
    engine, and that close must not replace the gate's error with a connection error, only say on stderr that the
    handshake broke."""

    class ExitingConn:
        def send(self, _):
            pass

        def recv(self):
            raise EOFError

        def close(self):
            pass

    class ExitingProcess:
        def __init__(self, code):
            self.exitcode = code

        def is_alive(self):
            return True

        def join(self):
            pass

    _ProcessEngine(ExitingProcess(0), ExitingConn(), {}).close()
    assert "left before the close handshake (EOFError)" in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="exited with code 1 while closing"):
        _ProcessEngine(ExitingProcess(1), ExitingConn(), {}).close()


def test_local_checkpoint_is_content_bound(tmp_path):
    (tmp_path / "config.json").write_bytes(b"config")
    (tmp_path / "weights.safetensors").write_bytes(b"weights")
    preset = local_checkpoint(load_spec("llama31-8b-fp8-1gpu").preset, tmp_path)
    assert preset.model == str(tmp_path)
    assert preset.revision is preset.case is None
    assert len(preset.model_digest) == 64
    (tmp_path / "weights.safetensors").write_bytes(b"changed")
    assert local_checkpoint(load_spec("llama31-8b-fp8-1gpu").preset, tmp_path).model_digest != preset.model_digest


def test_resident_engines_interleave_arms_rep_by_rep_and_discard_warmups():
    preset, points, arms = inputs()
    factory = FakeFactory()
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=factory, git_commit="abc")
    measured = [e[1] for e in factory.events if e[0] == "measure"]
    assert measured == ["stock-fp8", "candidate", "candidate", "stock-fp8"]
    assert doc["execution"]["interleave"] == "resident"
    assert all(doc["points"][0]["arms"][a.id]["metrics"]["time_per_output_token_s"]["n"] == 2 for a in arms)
    assert doc["points"][0]["arms"]["candidate"]["metrics"]["time_per_output_token_s"]["ratio_to_stock"]["mean"] == 2


@pytest.mark.parametrize("resident_failure", [None, "candidate"])
def test_every_engine_runs_the_warmups_before_its_first_timed_rep_and_the_schedule_keeps_them(resident_failure):
    """A `blocks` engine's first timed rep after one warm-up was slow on the bandwidth arms:
    the count is a spec knob in the identity, each engine runs all of them before it times anything, and each warm-up's
    own summary stays in the schedule."""

    preset, points, arms = inputs()
    factory = FakeFactory(resident_failure=resident_failure)
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=3, warmups=3, factory=factory, git_commit="abc")
    assert doc["identity"]["warmups"] == doc["execution"]["warmups"] == 3
    lifetimes: dict[str, list[list[str]]] = {}
    for event in factory.events:
        if event[0] == "create":
            lifetimes.setdefault(event[1], []).append([])
        elif event[0] in ("warmup", "measure"):
            lifetimes[event[1]][-1].append(event[0])
    timed = [runs for arm in lifetimes.values() for runs in arm if "measure" in runs]
    assert timed and all(runs[:3] == ["warmup"] * 3 and "warmup" not in runs[3:] for runs in timed)
    warm = [s for s in doc["execution"]["schedule"] if s["phase"] == "warmup"]
    assert len(warm) == 3 * len(timed) and [s["warmup"] for s in warm[:3]] == [0, 1, 2]
    assert all(s["run"]["wall_s"] == 0.5 for s in warm)
    with pytest.raises(ValueError, match="warm-up"):
        run_benchmark(preset, arms, points, sweep="quick", reps=1, warmups=0, factory=FakeFactory(), git_commit="abc")
    with pytest.raises(ValueError, match="max_warmups"):
        run_benchmark(preset, arms, points, sweep="quick", reps=1, warmups=3, max_warmups=2, factory=FakeFactory(),
                      git_commit="abc")


def test_an_engine_warms_past_its_minimum_until_two_runs_agree_and_says_when_it_never_did():
    """A run's plain engines warmed in 7.6 s, 2.3 s, 1.26 s and then ran at 1.26 s: a fixed count that
    happens to cover one host's stall is luck, so past `warmups` an engine warms until two consecutive runs agree."""

    class Stalls:
        def __init__(self, walls):
            self.walls = list(walls)

        def warmup(self, point):
            return {"wall_s": self.walls.pop(0)}

    point = expand_sweep("quick")[1]
    schedule = []
    _warm(Stalls([7.6, 2.27, 1.26, 1.262, 9.9]), point, 2, 8, schedule, {"arm": "plain"})
    assert [s["run"]["wall_s"] for s in schedule] == [7.6, 2.27, 1.26, 1.262]
    assert "settled" not in schedule[-1]
    schedule = []
    _warm(Stalls([1.0, 1.01, 9.9]), point, 2, 8, schedule, {"arm": "stock"})
    assert len(schedule) == 2
    schedule = []
    _warm(Stalls([5.0, 4.0, 3.0, 2.0, 1.0]), point, 1, 3, schedule, {"arm": "never"})
    assert len(schedule) == 3 and schedule[-1]["settled"] is False
    schedule = []
    _warm(Stalls([5.0, 4.0]), point, 2, 2, schedule, {"arm": "fixed"})
    assert len(schedule) == 2 and "settled" not in schedule[-1]


def test_a_timed_rep_keeps_its_step_times_request_times_and_host_counters():
    """A slow rep must say where its time went: which steps were slow, which requests waited, and whether the host was
    busy, ran slower code or waited on the device."""

    timing = {
        "steps": [{"s": 0.0123456789, "prefill_tokens": 512, "decode_tokens": 0},
                  {"s": 0.002, "prefill_tokens": 0, "decode_tokens": 8}],
        "requests": {"r-0": {"arrival_s": 0.0, "first_output_s": 0.0141},
                     "r-1": {"arrival_s": 0.0005, "first_output_s": None}},
    }
    assert run_trace(timing, start=12.5, ready=10.0) == {
        "since_ready_s": 2.5,
        "step_s": [0.012346, 0.002],
        "step_prefill_tokens": [512, 0],
        "step_decode_tokens": [0, 8],
        "arrival_s": [0.0, 0.0005],
        "first_output_s": [0.0141, None],
    }

    allocations = iter([{"num_device_alloc": 10, "num_alloc_retries": 0}, {"num_device_alloc": 13, "num_alloc_retries": 1}])
    probe = HostProbe(lambda: next(allocations))
    gc.collect()
    host = probe.close()
    assert host.pop("device") == {"num_device_alloc": 3, "num_alloc_retries": 1}
    assert set(host) == {"loadavg_1m", "process_cpu_s", "thread_cpu_s", "involuntary_switches", "voluntary_switches",
                         "minor_faults", "major_faults", "gc_s"}
    assert host["gc_s"] > 0 and all(v >= 0 for v in host.values())
    assert probe._on_gc not in gc.callbacks
    assert "device" not in HostProbe().close()


def test_resident_failure_falls_back_to_blocks():
    preset, points, arms = inputs()
    factory = FakeFactory(resident_failure="candidate")
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=3, factory=factory, git_commit="abc")
    assert doc["execution"]["interleave"] == "blocks"
    assert "does not fit" in doc["execution"]["resident_failure"]
    measured = [e[1] for e in factory.events if e[0] == "measure"]
    assert measured == ["stock-fp8", "stock-fp8", "candidate", "candidate", "candidate", "stock-fp8"]
    assert all(doc["points"][0]["arms"][a.id]["metrics"]["time_per_output_token_s"]["n"] == 3 for a in arms)


def test_an_engine_that_fails_to_start_closes_the_engines_already_up():
    preset, points, arms = inputs()
    factory = FakeFactory()
    create = factory.create

    def refuse(arm, preset, *, resident):
        if arm.id == "candidate":
            raise RuntimeError("Refused: protocol options refused")
        return create(arm, preset, resident=resident)

    factory.create = refuse
    with pytest.raises(RuntimeError, match="Refused"):
        run_benchmark(preset, arms, points, sweep="quick", reps=1, factory=factory, git_commit="abc")
    assert factory.events == [("create", "stock-fp8", True), ("close", "stock-fp8")]


@pytest.mark.parametrize("factory", [FakeFactory(gate_ok=False), FakeFactory(bad_tokens="candidate")])
def test_a_failed_gate_records_the_failure_and_stops_before_timing(factory):
    preset, points, arms = inputs()
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=factory, git_commit="abc")
    assert any(not g["passed"] for g in doc["gates"])
    assert doc["points"] == []
    assert not any(e[0] == "measure" for e in factory.events)
    validate(doc)


@pytest.mark.parametrize("resident_failure", [None, "candidate"])
def test_an_arm_gate_is_read_again_after_its_timed_runs_and_its_failure_drops_the_points(resident_failure):
    """An arm judges its warm-up and timed runs only once they have run: served_commit's gate names a rootless warm-up
    in its ``warmups`` and fails a measured run without a root, so the verdict must hold what the gate says after them."""

    preset, points, arms = inputs()
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=FakeFactory(), git_commit="abc")
    closing = [g for g in doc["gates"] if g["kind"] == "arm-closing"]
    assert all(g["passed"] for g in closing) and doc["points"]
    assert [g["arm"] for g in closing] == ["stock-fp8", "candidate"]

    factory = FakeFactory(timed_ok=False, resident_failure=resident_failure)
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=factory, git_commit="abc")
    assert all(g["passed"] for g in doc["gates"] if g["kind"] == "arm")
    closing = [g for g in doc["gates"] if g["kind"] == "arm-closing"]
    assert closing and not any(g["passed"] for g in closing)
    if resident_failure:
        # blocks mode stops opening engines at the first failed gate (its points are dropped either way)
        assert (closing[0]["arm"], closing[0]["cycle"]) == (arms[0].id, 0) and len(closing) == 1
    assert doc["points"] == []
    validate(doc)


def test_a_failed_token_gate_records_the_first_divergent_token():
    preset, points, arms = inputs()
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2,
                        factory=FakeFactory(bad_tokens="candidate"), git_commit="abc")
    failed, = [g for g in doc["gates"] if g["kind"] == "tokens" and not g["passed"]]
    assert failed["first_divergence"] == {
        "request": "b32-p512-mixed-0", "position": 0, "token": 8, "control_token": 7,
    }
    assert failed["divergent_requests"] == 1


def test_unequal_kv_pools_fail_the_gate_and_name_the_pin():
    """At b128-p8192 the audit arm's engine got a smaller KV pool than its plain twin's (its encoded store and
    scratch shrink what vLLM's profiling leaves), vLLM scheduled the two arms differently once the point overran the
    pool, and the greedy tokens diverged. Equal pools are a precondition of the token gate, so unequal pools must fail
    loudly on their own, before any timing."""

    preset, points, arms = inputs()
    factory = FakeFactory(kv_tokens={"stock-fp8": 494_064, "candidate": 491_376})
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=factory, git_commit="abc")
    gate, = [g for g in doc["gates"] if g["kind"] == "kv-pool"]
    assert not gate["passed"]
    assert (gate["tokens"], gate["control_tokens"]) == (491_376, 494_064)
    assert "resident_kv_cache_memory_bytes" in gate["detail"] and "block_kv_cache_memory_bytes" in gate["detail"]
    assert doc["points"] == []
    assert not any(e[0] == "measure" for e in factory.events)
    validate(doc)


def test_equal_kv_pools_pass_and_are_recorded_in_the_identity():
    preset, points, arms = inputs()
    factory = FakeFactory(kv_tokens={"stock-fp8": 458_752, "candidate": 458_752})
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=factory, git_commit="abc")
    gate, = [g for g in doc["gates"] if g["kind"] == "kv-pool"]
    assert gate["passed"] and doc["points"]
    assert {a["id"]: a["kv_cache_tokens"] for a in doc["identity"]["arms"]} == {
        "stock-fp8": 458_752, "candidate": 458_752,
    }


def test_blocks_mode_pins_one_kv_pool_for_every_arm_of_the_llama_preset():
    settings = load_spec("llama31-8b-fp8-1gpu").preset.engine_settings
    assert settings["block_kv_cache_memory_bytes"] == 56 * 2**30


def _serving_defaults_preset():
    preset = load_spec("llama32-1b-bf16-rtx-served-arm").preset
    return replace(preset, engine_settings={**preset.engine_settings, "serving_defaults": True})


def test_engine_args_pin_the_pool_in_either_mode_and_serving_defaults_leave_compilation_to_vllm():
    served_preset = _serving_defaults_preset()
    for resident in (True, False):
        args = engine_args(served_preset, "bfloat16", resident=resident, eager=False)
        assert args["kv_cache_memory_bytes"] == 4 * 2**30
        assert "compilation_config" not in args and args["enforce_eager"] is False and args["execution"] == "compiled"
    assert engine_args(served_preset, "bfloat16", resident=True, eager=True)["execution"] == "eager"
    pinned = load_spec("llama31-8b-fp8-1gpu").preset
    resident = engine_args(pinned, "fp8", resident=True, eager=False)
    assert "kv_cache_memory_bytes" not in resident and resident["quantization"] == "fp8"
    assert resident["compilation_config"] == {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                              "cudagraph_capture_sizes": [1, 8, 32, 128]}
    assert engine_args(pinned, "fp8", resident=False, eager=False)["kv_cache_memory_bytes"] == 56 * 2**30
    assert "compilation_config" not in engine_args(pinned, "fp8", resident=False, eager=True)


def test_serving_defaults_give_every_arm_process_one_inductor_cache(monkeypatch):
    spawned = []

    class Context:
        def Pipe(self):
            class End:
                def recv(self):
                    return {"ready": {}}

                def close(self):
                    pass
            return End(), End()

        def Process(self, target, args):
            spawned.append(settings.environ())
            return SimpleNamespace(start=lambda: None, join=lambda: None)

    monkeypatch.setenv(INDUCTOR_DIR_MARK, "")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/elsewhere")
    monkeypatch.setattr(tempfile, "mkdtemp", lambda prefix: f"/tmp/{prefix}x")
    factory = ProcessEngineFactory()
    factory.context = Context()
    preset = _serving_defaults_preset()
    for arm in resolve_arms("stock"):
        factory.create(arm, preset, resident=True)
        factory.create(arm, preset, resident=True)
    assert [e["TORCHINDUCTOR_CACHE_DIR"] for e in spawned] == ["/tmp/verity_bench_inductor_x"] * 2
    assert all(e[INDUCTOR_DIR_MARK] == "1" for e in spawned)
