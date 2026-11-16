"""Served variants and protocol arms: their checkpoints, settings and execution, and the Pearl-C and H1 arms' gates."""
from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from verity_vllm.engine.env import INDUCTOR_DIR_MARK

from verity_bench import settings
from verity_bench.vllm.arms import (
    H1,
    h1_config,
    plugin,
    protocols_config,
    resolve_arms,
    window_steps,
)
from verity_bench.vllm.engine import (
    H1_OPENING_STEPS,
    PASS_COUNTS,
    ProcessEngineFactory,
    _pass_totals,
    _pearl_c_run,
    _VllmEngine,
    arm_execution,
    engine_args,
)
from verity_bench.vllm.presets import (
    SPEC_ROOT,
    SweepPoint,
    expand_sweep,
    load_spec,
    resolve_spec,
    served,
)
from verity_bench.vllm.runner import (
    h1_gate,
    h1_summary,
)


def test_a_served_variant_times_a_builtin_engine_in_its_own_execution_under_an_eager_spec(monkeypatch):
    """Every protocol row times eager against eager while vLLM serves compiled: `llama31-8b-fp8-1gpu-compiled-full` times
    the eager rows' stock FP8 engine against itself at vLLM's serving defaults, eager and compiled, in one spec."""

    spec = resolve_spec("llama31-8b-fp8-1gpu-compiled-full")
    assert spec.eager and "serving_defaults" not in spec.preset.engine_settings
    base, served_eager, compiled = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
    assert [(a.id, a.precision, a.control, a.execution) for a in (base, served_eager, compiled)] == [
        ("stock-fp8", "fp8", None, None),
        ("stock-fp8@eager", "fp8", None, "eager"),
        ("stock-fp8@compiled", "fp8", None, "compiled"),
    ]
    assert compiled.to_json()["execution"] == "compiled"
    assert resolve_arms("stock-fp8,stock-fp8@compiled")[1].control == "stock-fp8"
    for bad in ("stock-fp8@bi-eager", "nope@compiled"):
        with pytest.raises(ValueError, match="BUILTIN@LABEL"):
            resolve_arms(f"stock-fp8,{bad}")

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
            spawned.append(args)
            return SimpleNamespace(start=lambda: None, join=lambda: None)

    monkeypatch.setenv(INDUCTOR_DIR_MARK, "")
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.setattr(tempfile, "mkdtemp", lambda prefix: f"/tmp/{prefix}x")
    factory = ProcessEngineFactory(eager=True)
    factory.context = Context()
    preset, _ = served(spec.preset, spec.eager)
    for arm in (base, served_eager, compiled):
        factory.create(arm, preset, resident=True)
    built = [engine_args(p, a.precision, resident=True, eager=e) for _, a, p, _, e in spawned]
    assert [(b["enforce_eager"], b.get("execution"), "compilation_config" in b) for b in built] == [
        (True, None, False), (True, "eager", False), (False, "compiled", False)]
    assert {b["kv_cache_memory_bytes"] for b in built} == {8 << 30}
    assert {b["max_num_batched_tokens"] for b in built} == {4096}
    assert settings.environ()["TORCHINDUCTOR_CACHE_DIR"] == "/tmp/verity_bench_inductor_x"


def test_a_served_variants_arm_settings_take_only_its_control(tmp_path):
    source = (SPEC_ROOT / "llama31-8b-fp8-1gpu-compiled-full.toml").read_text()
    path = tmp_path / "spec.toml"
    section = '[arm_settings."stock-fp8@compiled"]\ncontrol = "none"'
    assert section in source
    path.write_text(source.replace(section, '[arm_settings."stock-fp8@compiled"]\ncontrol = "stock-fp8@eager"'))
    spec = load_spec(path)
    assert resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)[2].control == "stock-fp8@eager"
    for old, new, why in ((section, '[arm_settings."stock-fp8@compiled"]\nprecision = "bfloat16"', "takes only its control"),
                          (section, '[arm_settings."stock-fp8@compiled"]\ncontrol = "stock"', "control must be another arm"),
                          ('[arm_settings."stock-fp8@compiled"]', '[arm_settings."stock-fp8"]', "must name a protocol arm")):
        assert old in source
        path.write_text(source.replace(old, new))
        with pytest.raises(ValueError, match=why):
            load_spec(path)


def test_a_served_variant_loads_its_own_checkpoint_where_the_presets_is_another_encoding(tmp_path):
    """`llama31-8b-1gpu-pouw-h1-served` times H1 on the int8 pin over stock FP8 as vLLM serves it, which vLLM can't load
    from the pin (`verity_int8`): its served variant names the pin's BF16 base, and only its engine loads it."""

    spec = resolve_spec("llama31-8b-1gpu-pouw-h1-served")
    compiled, int8, h1 = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
    assert [(a.id, a.precision, a.control, a.execution, a.model) for a in (compiled, int8, h1)] == [
        ("stock-fp8@compiled", "fp8", None, "compiled", "unsloth/Llama-3.1-8B"),
        ("stock-int8", "int8_exact", None, None, None),
        ("pouw:eps1-sm120-h1fx1", None, "stock-int8", None, None),
    ]
    assert compiled.to_json()["revision"] == "3f0d51f8e5640f98f1a96ea9044a0e55c0a83814"
    preset, eager = served(spec.preset, spec.eager)
    built = [arm_execution(a, preset, eager) for a in (compiled, int8, h1)]
    assert [(p.model, p.revision, p.case, e) for p, e in built] == [
        ("unsloth/Llama-3.1-8B", "3f0d51f8e5640f98f1a96ea9044a0e55c0a83814", None, False),
        ("verity-int8/Llama-3.1-8B-INT8", "723d342ac443b040931242c768df342b0f6f15a7", "LLAMA31_8B_INT8", True),
        ("verity-int8/Llama-3.1-8B-INT8", "723d342ac443b040931242c768df342b0f6f15a7", "LLAMA31_8B_INT8", True),
    ]
    args = engine_args(built[0][0], compiled.precision, resident=True, eager=False)
    assert (args["quantization"], args["execution"], args["max_num_batched_tokens"]) == ("fp8", "compiled", 64)

    source = (SPEC_ROOT / "llama31-8b-1gpu-pouw-h1-served.toml").read_text()
    path = tmp_path / "spec.toml"
    pin = 'revision = "3f0d51f8e5640f98f1a96ea9044a0e55c0a83814"\n'
    for old, new, why in ((pin, "", "set both"),
                          ('control = "stock-int8"\n', f'control = "stock-int8"\nmodel = "x/y"\n{pin}',
                           "only a served variant names its own checkpoint")):
        assert old in source
        path.write_text(source.replace(old, new))
        with pytest.raises(ValueError, match=why):
            load_spec(path)


def test_a_protocol_arm_takes_its_precision_control_and_pouw_settings_from_the_spec(monkeypatch):
    """`pouw:pearl-c-sm120-v1-h3` was refused before any engine started: the harness
    gave every protocol arm the preset's FP8 and no pouw settings. Its spec's `arm_settings` give it the BF16 checkpoint,
    stock FP8 as its control, and the device, kernel and setup-built cubin, with the harness's challenger as its coins,
    which the adapter's refusals accept."""

    from verity.protocols.resource_accounting.compute.pouw.verifier import verifier
    from verity.protocols.resource_accounting.compute.pouw.verifier.coins import CoinCommitment
    from verity_vllm.protocols import refusals
    from verity_vllm.protocols.interface import Context

    spec = resolve_spec("llama31-8b-1gpu-pouw-protocol-served")
    control, arm, _ = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
    assert (control.id, control.precision) == ("stock-fp8", "fp8")
    assert (arm.protocols, arm.precision, arm.control) == (("pouw:pearl-c-sm120-v1-h3",), "bfloat16", None)
    assert arm.protocols_config == {"pouw": {"device": "cuda", "kernel": "verity/protocols/resource_accounting/compute/pouw/prover/kernels/pearl_c_sm120",
                                             "cubin": "$PEARLC_CUBIN", "schedule": "deferred"}}
    assert arm.to_json()["protocols_config"] == arm.protocols_config
    settings = spec.arm_settings["pouw:pearl-c-sm120-v1-h3"]
    other = replace(settings, protocols_config={"pouw": {**settings.protocols_config["pouw"], "schedule": "serial"}})
    assert spec.config_sha256 != replace(spec, arm_settings={arm.id: other}).config_sha256
    assert resolve_arms(spec.arms)[1].control == "stock-fp8" and resolve_arms(spec.arms)[1].precision is None

    with pytest.raises(ValueError, match=r"\$PEARLC_CUBIN"):
        protocols_config(arm, {})
    monkeypatch.setattr(verifier, "coins", lambda n: b"\1" * n)
    config = protocols_config(arm, {"PEARLC_CUBIN": "/run/out/setup/pearl_c_sm120.cubin"})
    assert config["pouw"]["cubin"] == "/run/out/setup/pearl_c_sm120.cubin"
    assert isinstance(config["pouw"]["coins"].commitment, CoinCommitment)
    ctx = Context(stage="serve", enabled=arm.protocols, settings=config)
    assert refusals(ctx) == ()
    assert "device only" in refusals(Context(stage="serve", enabled=arm.protocols))[0]


def test_pearlpluss_h200_spec_serves_the_1bs_bf16_checkpoint_through_sm90s_kernel_from_its_setup(monkeypatch):
    """`served_pearlplus.sh`'s spec: Llama 3.2 1B's BF16 checkpoint through `pouw:pearl-c-h100-v1-h3` on one H200, its kernel
    directory and cubin both the setup's (pearl_c_sm90.sh exports them), serial, beside stock FP8 eager and compiled; the
    adapter's refusals accept it."""

    from verity.protocols.resource_accounting.compute.pouw.verifier import verifier
    from verity_vllm.protocols import refusals
    from verity_vllm.protocols.interface import Context

    spec = resolve_spec("llama32-1b-1gpu-pouw-protocol-served-h200")
    assert (spec.preset.model, spec.preset.gpu_id, spec.preset.gpu_count) == ("unsloth/Llama-3.2-1B", "h200", 1)
    assert spec.setup == "benchmarks/harness/setup/pearl_c_sm90.sh"
    control, arm, compiled = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
    assert [(a.id, a.precision, a.control, a.execution) for a in (control, compiled)] == [
        ("stock-fp8", "fp8", None, None), ("stock-fp8@compiled", "fp8", None, "compiled")]
    assert (arm.protocols, arm.precision, arm.control) == (("pouw:pearl-c-h100-v1-h3",), "bfloat16", None)
    assert arm.protocols_config == {"pouw": {"device": "cuda", "kernel": "$PEARLC_KERNEL", "cubin": "$PEARLC_CUBIN"}}
    with pytest.raises(ValueError, match=r"\$PEARLC_KERNEL"):
        protocols_config(arm, {"PEARLC_CUBIN": "/run/out/setup/pearl_c_sm90/pearl_c_sm90.cubin"})
    monkeypatch.setattr(verifier, "coins", lambda n: b"\1" * n)
    config = protocols_config(arm, {"PEARLC_KERNEL": "/tree/benchmarks/pouw/pearl_c_sm90",
                                    "PEARLC_CUBIN": "/run/out/setup/pearl_c_sm90/pearl_c_sm90.cubin"})
    assert (config["pouw"]["kernel"], config["pouw"]["cubin"]) == ("/tree/benchmarks/pouw/pearl_c_sm90",
                                                                   "/run/out/setup/pearl_c_sm90/pearl_c_sm90.cubin")
    assert refusals(Context(stage="serve", enabled=arm.protocols, settings=config)) == ()


def test_served_passes_arm_writes_that_arm_as_serve_openais_protocols_config(monkeypatch, tmp_path):
    """``served_passes.py arm`` writes the H200 spec's Pearl-C arm as serve-openai's Pearl+ arm takes it
    (`serve_pearl.config_of`): `serve_pearl.CONFIG`'s keys alone, the setup's paths read in, no challenges flipped (the
    server flips each window's); an unset ``$`` value is refused, naming it."""
    from benchmarks.pouw.pearl_c_vllm import served_passes as SP
    from verity.protocols.resource_accounting.compute.pouw.verifier import verifier
    from verity_vllm.pipeline import serve_pearl as SPL

    def flipped(n):
        raise AssertionError("served_passes.py arm flipped challenges")
    monkeypatch.setattr(verifier, "coins", flipped)
    spec, out = str(SPEC_ROOT / "llama32-1b-1gpu-pouw-protocol-served-h200.toml"), tmp_path / "config.json"
    monkeypatch.delenv("PEARLC_KERNEL", raising=False)
    monkeypatch.setenv("PEARLC_CUBIN", "/run/out/setup/pearl_c_sm90/pearl_c_sm90.cubin")
    with pytest.raises(ValueError, match=r"\$PEARLC_KERNEL"):
        SP.main(["arm", "--spec", spec, "--out", str(out)])
    monkeypatch.setenv("PEARLC_KERNEL", "/tree/benchmarks/pouw/pearl_c_sm90")
    assert SP.main(["arm", "--spec", spec, "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc == {"pouw": {"device": "cuda", "kernel": "/tree/benchmarks/pouw/pearl_c_sm90",
                            "cubin": "/run/out/setup/pearl_c_sm90/pearl_c_sm90.cubin"}}
    assert SPL.config_of(str(out)) == doc


def test_arm_settings_name_a_protocol_arm_of_the_spec_and_a_control_it_runs(tmp_path):
    source = (SPEC_ROOT / "llama31-8b-1gpu-pouw-protocol-served.toml").read_text()
    path = tmp_path / "spec.toml"
    for control, want in (("stock-fp8", "stock-fp8"), ("none", None)):
        path.write_text(source.replace('control = "none"', f'control = "{control}"', 1))
        spec = load_spec(path)
        assert resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)[1].control == want
    for old, new, why in (('arms = ["stock-fp8", "pouw:pearl-c-sm120-v1-h3", "stock-fp8@compiled"]',
                           'arms = ["stock-fp8", "pouw:pearl-c-h100-v1-h3", "stock-fp8@compiled"]', "must name a protocol arm"),
                          ('[arm_settings."pouw:pearl-c-sm120-v1-h3"]\n', '[arm_settings.stock]\n', "must name a protocol arm"),
                          ('precision = "bfloat16"', 'precision = "int4"', "precision is one of"),
                          ('control = "none"', 'control = "stock"', "control must be another arm"),
                          ('control = "none"', 'control = "pouw:pearl-c-sm120-v1-h3"', "control must be another arm")):
        assert old in source
        path.write_text(source.replace(old, new, 1))
        with pytest.raises(ValueError, match=why):
            load_spec(path)
    with pytest.raises(ValueError, match="not in --arms"):
        resolve_arms("stock-fp8", (), load_spec("llama31-8b-1gpu-pouw-protocol-served").arm_settings)


def test_a_pearl_c_protocol_arms_passes_are_the_adapters_and_the_harness_only_reads_them():
    """The adapter ends a Pearl-C pass at each ``execute_model`` (`pouw.install`); the harness wraps nothing, finds the
    device run through the option's ``end_pass`` and reports the running totals of the passes it ended."""
    from verity_torch.pouw.pearl_c.run_device import PearlCDeviceRun

    class Run(PearlCDeviceRun):
        def __init__(self):
            self.details = {"passes": []}

        def end_pass(self):
            self.details["passes"].append({"units": 224, "tiles": 5000, "screened_rows": 64, "failing_rows": 0,
                                           "unsure_rows": 1})

    run = Run()
    option = SimpleNamespace(end_pass=run.end_pass, release=lambda: None)
    llm = SimpleNamespace(protocols=SimpleNamespace(get=lambda name: option if name == "pouw" else None))
    assert _pearl_c_run(llm) is run
    assert _pass_totals(run) == dict.fromkeys(("passes", *PASS_COUNTS), 0)
    run.end_pass()
    run.end_pass()
    assert _pass_totals(run) == {"passes": 2, "units": 448, "tiles": 10000, "screened_rows": 128,
                                        "failing_rows": 0, "unsure_rows": 2}
    other = SimpleNamespace(protocols=SimpleNamespace(get=lambda name: SimpleNamespace(release=lambda: None)))
    assert _pearl_c_run(other) is None
    assert _pearl_c_run(SimpleNamespace(protocols=None)) is None


def test_pearl_cs_served_row_is_its_protocol_row_beside_the_compiled_engine():
    """`llama31-8b-1gpu-pouw-protocol-served` measures Pearl-C over the compiled engine in one run, so every arm's engine
    shares its step budget, sequence bound and KV pool."""

    spec = resolve_spec("llama31-8b-1gpu-pouw-protocol-served")
    pearl = "pouw:pearl-c-sm120-v1-h3"
    arms = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
    assert [(a.id, a.precision, a.control, a.execution) for a in arms] == [
        ("stock-fp8", "fp8", None, None),
        (pearl, "bfloat16", None, None),
        ("stock-fp8@compiled", "fp8", None, "compiled"),
    ]
    preset, eager = served(spec.preset, spec.eager)
    for resident in (True, False):
        built = [engine_args(p, a.precision or p.precision, resident=resident, eager=e)
                 for a in arms for p, e in [arm_execution(a, preset, eager)]]
        assert [(b["enforce_eager"], b.get("execution")) for b in built] == [(True, None), (True, None), (False, "compiled")]
        shared = ("max_num_batched_tokens", "max_num_seqs", "max_model_len", "gpu_memory_utilization",
                  "kv_cache_memory_bytes")
        assert len({tuple(b[k] for k in shared) for b in built}) == 1
        assert built[0]["kv_cache_memory_bytes"] == 8 << 30 and built[0]["max_num_batched_tokens"] == 4096

    b32 = resolve_spec("llama31-8b-1gpu-pouw-protocol-served-b32")
    assert (b32.arms, b32.arm_settings, b32.setup, b32.eager, b32.preset) == (
        spec.arms, spec.arm_settings, spec.setup, spec.eager, replace(spec.preset, id=b32.id))
    assert b32.reps == 6 and b32.sweep == replace(spec.sweep, id="b32", batches=(32,))


def test_an_h1_arm_is_gated_against_exact_int8_and_each_run_gets_a_window_the_llm_accepts():
    """`pouw:eps1-sm120-h1fx1` serves the int8_exact arm's own y, so its token control is that engine without PoUW
    (`stock-int8`). The harness hands `verity_vllm.LLM` only H1's public parameters, T the engine's step budget and S
    from the run's step bound, and the LLM fills the model runner and in-process coins after the build; a spec can't
    supply them, nor S."""

    from verity.errors import Refused
    from verity.protocols.resource_accounting.compute.pouw.prover.serving import H1_SCHEME
    from verity_vllm.llm import protocol_context, resolve_options

    spec = resolve_spec("llama31-8b-1gpu-pouw-h1")
    control, arm = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
    assert (control.id, control.precision, control.control) == ("stock-int8", "int8_exact", None)
    assert (arm.id, arm.protocols, arm.precision, arm.control) == (H1, (H1,), None, "stock-int8")
    assert f"pouw:{H1_SCHEME}" == H1
    preset, eager = served(spec.preset, spec.eager)
    args = engine_args(preset, "int8_exact", resident=True, eager=eager)
    assert (args["quantization"], args["attention_backend"], args["enforce_eager"]) == ("verity_int8", "CUSTOM", True)
    assert args["max_num_batched_tokens"] == spec.arm_settings[H1].protocols_config["pouw"]["rows"] == 64
    assert args["max_num_seqs"] == 64 and engine_args(load_spec("llama31-8b-fp8-1gpu").preset, "fp8", resident=True,
                                                      eager=True)["max_num_seqs"] == 128
    with pytest.raises(ValueError, match="eagerly only"):
        engine_args(preset, "int8_exact", resident=True, eager=False)

    b8, b32 = expand_sweep("quick")
    assert (window_steps(b8, 64), window_steps(b32, 64)) == (130, 294)
    config = protocols_config(arm, {})
    assert config == {"pouw": {"rows": 64, "admission": "per-round"}}
    assert h1_config(config, H1_OPENING_STEPS) == {"pouw": {"rows": 64, "admission": "per-round", "steps": 1}}
    for steps in (H1_OPENING_STEPS, 294):
        ctx = protocol_context(resolve_options({"protocols": H1, "protocols_config": h1_config(config, steps)}))
        assert ctx.settings_of("pouw")["steps"] == steps and ctx.filled == {"pouw": ("runner", "coins", "rounds")}
    with pytest.raises(Refused, match=r"'rows' \(T"):
        protocol_context(resolve_options({"protocols": H1, "protocols_config": config}))
    for named in ({"steps": 10}, {"coins": object()}, {"runner": object()}):
        with pytest.raises(ValueError, match="each window's"):
            h1_config({"pouw": {"rows": 64, **named}}, 294)


def test_each_h1_run_opens_its_own_window_through_the_llm_and_closes_it_after_serving(monkeypatch):
    """Per run, the H1 engine calls `verity_vllm.LLM.reinstall` with the run's S before serving, reads that window
    (`pouw.h1_run`) and closes it with `LLM.close` after serving, even when serving fails; each window's summary is the
    arm's gate, and a timed rep's is its `extras.h1_window`."""

    from verity_vllm.protocols import pouw

    calls: list[tuple[Any, ...]] = []
    window = SimpleNamespace(verdicts={0: {"credit": True}}, refused=[], order_ok=True, steps=130, rows=64, calls=[0],
                             requests=[0], refused_requests=[], coins_from="in-process",
                             details={"steps_served": 106, "static": True, "u": "1/2"}, request_gaps=lambda: [])

    class LLM:
        def reinstall(self, config: dict[str, Any]) -> None:
            calls.append(("reinstall", config))

        def close(self) -> None:
            calls.append(("close",))

    def serve(point: SweepPoint, phase: str) -> tuple[dict[str, Any], dict[str, Any]]:
        calls.append(("serve", phase))
        if phase == "warmup":
            raise RuntimeError("serving failed")
        return {}, {"extras": {}}

    monkeypatch.setattr(pouw, "h1_run", lambda entry: calls.append(("h1_run", entry)) or window)
    engine = object.__new__(_VllmEngine)
    engine.llm, engine.h1 = LLM(), {"config": {"pouw": {"rows": 64, "admission": "per-round"}}, "rows": 64,
                                    "windows": []}
    monkeypatch.setattr(engine, "_serve", serve)
    b8 = expand_sweep("quick")[0]
    engine._run(b8, "gate")
    with pytest.raises(RuntimeError, match="serving failed"):
        engine._run(b8, "warmup")
    out = engine._run(b8, "measure")[1]
    opened = ("reinstall", {"pouw": {"rows": 64, "admission": "per-round", "steps": 130}})
    assert calls == [opened, ("h1_run", H1), ("serve", "gate"), ("close",)] + [
        opened, ("h1_run", H1), ("serve", "warmup"), ("close",), opened, ("h1_run", H1), ("serve", "measure"), ("close",)]
    assert [w["phase"] for w in engine.h1["windows"]] == ["gate", "warmup", "measure"]
    assert out["extras"]["h1_window"] is engine.h1["windows"][-1] and out["extras"]["h1_window"]["steps_served"] == 106
    assert engine.custom_gate()["passed"]


def test_an_h1_window_passes_its_gate_only_static_credited_and_entered():
    def run(**over):
        base = dict(verdicts={0: {"credit": True}, 1: {"credit": True}}, refused=[], order_ok=True, steps=130, rows=64,
                    calls=[0, 1], requests=[0], refused_requests=[], coins_from="in-process",
                    details={"steps_served": 120, "static": True, "u": "1/2"}, request_gaps=lambda: [])
        return SimpleNamespace(**{**base, **over})

    good = h1_summary(run())
    assert good["ok"] and good["served_calls"] == good["credit_gate_passed"] == 2 and good["steps_served"] == 120
    bad = [h1_summary(run(refused=[{"step": 130, "linear": 0, "m": 8}])),
           h1_summary(run(details={"static": False})),
           h1_summary(run(verdicts={0: {"credit": False, "reason": "credit: served-output-not-credited-kernel"}})),
           h1_summary(run(request_gaps=lambda: ["step 3: 1 row of a request with no entry"])),
           h1_summary(run(details={"static": True, "unopened": "refused"})),
           h1_summary(run(verdicts={}))]
    assert not any(b["ok"] for b in bad)
    assert bad[2]["credit_gate_refusals"] == ["credit: served-output-not-credited-kernel"]
    assert h1_gate([good, good])["passed"] and not h1_gate([]).get("passed")
    gate = h1_gate([good, bad[0]])
    assert not gate["passed"] and gate["provided"] and gate["failed"] == [bad[0]] and "credits nothing" in gate["coins"]


def test_pearl_cs_whole_step_arm_builds_vllms_full_decode_graphs_beside_the_served_protocol_arm(monkeypatch):
    """`llama31-8b-1gpu-pearl-c-whole-served` times the whole-step arm beside the protocol arm as served (its settings the
    protocol row's) and the compiled engine. Under the eager spec, the arm's overrides build vLLM's FULL decode graphs at
    the preset's capture sizes without torch.compile, at the `compiled` label, which `verity_vllm.LLM` accepts, and its
    run takes the graph path's options and serial schedule."""

    from verity_vllm.llm import resolve_options

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    spec = resolve_spec("llama31-8b-1gpu-pearl-c-whole-served")
    pearl = "pouw:pearl-c-sm120-v1-h3"
    assert spec.arm_settings[pearl] == resolve_spec("llama31-8b-1gpu-pouw-protocol-served").arm_settings[pearl]
    arms = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
    assert [(a.id, a.precision, a.control, a.execution, a.experimental) for a in arms] == [
        ("stock-fp8", "fp8", None, None, False), (pearl, "bfloat16", None, None, False),
        ("stock-fp8@compiled", "fp8", None, "compiled", False), ("pouw-pearl-c-whole", "bfloat16", None, None, True)]
    assert arms[-1].engine_label == "graphs-mode0" and arms[-1].to_json()["engine_label"] == "graphs-mode0"
    whole = plugin(arms[-1])
    preset, eager = served(spec.preset, spec.eager)
    preset, eager = arm_execution(arms[-1], preset, eager)
    built = {**engine_args(preset, "bfloat16", resident=False, eager=eager), **whole.engine_overrides(None)}
    assert built["compilation_config"] == {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                           "cudagraph_capture_sizes": list(spec.preset.engine_settings["capture_sizes"])}
    opts = resolve_options(built)
    assert opts["enforce_eager"] is False and opts["execution"] == "compiled"
    assert whole.run_options == {"graphs": True, "whole_defer": True, "whole_defer_a": True, "whole_fork_stats": True,
                                 "lag_ends": True, "whole_tree_sm": True, "aside_ends": True, "whole_prefetch_b": True,
                                 "whole_eager": "deferred-a"} and whole.schedule == "serial"
    monkeypatch.setenv("PEARLC_WHOLE_PREFETCH_B", "0")
    assert "whole_prefetch_b" not in type(whole)().run_options
    monkeypatch.setenv("PEARLC_WHOLE_EAGER", "deferred")
    assert type(whole)().run_options["whole_eager"] == "deferred"
    monkeypatch.setenv("PEARLC_WHOLE_EAGER", "0")
    assert "whole_eager" not in type(whole)().run_options


def test_pearl_cs_whole_step_arm_on_an_h200_runs_pearl_c_sm90s_kernels_under_the_h100_record(monkeypatch):
    """`llama31-8b-1gpu-pearl-c-whole-served-h200` is the whole-served spec's preset on one H200, the whole-step arm beside
    compiled stock, its setup building pearl_c_sm90's cubin; with PEARLC_KERNEL that directory, the arm takes its record
    (`pearl-c-h100-v1-h3`, a scheme the device path runs) and its build.sh cubin, and builds the same engine."""
    from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c import SCHEMES

    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root))
    spec, rtx = resolve_spec("llama31-8b-1gpu-pearl-c-whole-served-h200"), resolve_spec("llama31-8b-1gpu-pearl-c-whole-served")
    assert (spec.preset.gpu_id, spec.preset.engine_settings, spec.preset.model) == ("h200", rtx.preset.engine_settings,
                                                                                     rtx.preset.model)
    assert (root / spec.setup).is_file() and "pearl_c_sm90/build.sh" in (root / spec.setup).read_text()
    arms = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
    assert [(a.id, a.execution, a.experimental) for a in arms] == [("stock-fp8@compiled", "compiled", False),
                                                                   ("pouw-pearl-c-whole", None, True)]
    from benchmarks.pouw.pearl_c_vllm import bench_arm
    assert set(bench_arm.KERNEL_SCHEMES.values()) <= set(SCHEMES)
    assert all((root / "benchmarks/pouw" / d).is_dir() for d in bench_arm.KERNEL_SCHEMES)
    monkeypatch.setenv("PEARLC_KERNEL", str(root / "verity/protocols/resource_accounting/compute/pouw/prover/kernels/pearl_c_sm90"))
    monkeypatch.delenv("PEARLC_SCHEME", raising=False)
    monkeypatch.delenv("PEARLC_CUBIN", raising=False)
    whole = bench_arm.PearlCWholeArm()
    assert whole.scheme == "pearl-c-h100-v1-h3" and whole.kernel_dir.name == "pearl_c_sm90"
    assert whole.kernel_digest(None) is None or (whole.kernel_dir / "pearl_c_sm90.cubin").is_file()
    preset, eager = served(spec.preset, spec.eager)
    preset, eager = arm_execution(arms[-1], preset, eager)
    built = {**engine_args(preset, "bfloat16", resident=False, eager=eager), **whole.engine_overrides(None)}
    assert built["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert whole.run_options == {"graphs": True, "whole_defer": True, "whole_defer_a": True, "whole_fork_stats": True,
                                 "lag_ends": True, "whole_tree_sm": True, "aside_ends": True} and whole.schedule == "serial"
