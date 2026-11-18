"""Served variants and protocol arms: their checkpoints, settings and execution, and the Pearl-C arms' gates."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from verity_vllm.engine.env import INDUCTOR_DIR_MARK

from verity_bench import jobs, settings
from verity_bench.vllm.arms import plugin, protocols_config, resolve_arms
from verity_bench.vllm.engine import PASS_COUNTS, ProcessEngineFactory, _pass_totals, _pearl_c_run, arm_execution, engine_args
from verity_bench.vllm.presets import SPEC_ROOT, ArmSettings, load_spec, resolve_spec, served


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


def test_a_protocol_arm_takes_its_precision_control_and_pouw_settings_from_the_spec(monkeypatch):
    """A protocol arm was refused before any engine started while the harness gave every protocol arm the preset's FP8
    and no pouw settings. Its spec's `arm_settings` give it the BF16 checkpoint, stock FP8 as its control, and the device,
    kernel and setup-built cubin, which are the spec's identity, with the harness's challenger as its coins, which the
    adapter's refusals accept."""

    from verity.protocols.resource_accounting.compute.pouw.verifier import verifier
    from verity.protocols.resource_accounting.compute.pouw.verifier.coins import CoinCommitment
    from verity_vllm.protocols import refusals
    from verity_vllm.protocols.interface import Context

    spec, pearl = resolve_spec("llama32-1b-1gpu-pouw-protocol-served@h200"), "pouw:pearl-c-h100-v1-h3"
    control, arm, _ = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
    assert (control.id, control.precision) == ("stock-fp8", "fp8")
    assert (arm.protocols, arm.precision, arm.control) == ((pearl,), "bfloat16", None)
    assert arm.protocols_config == {"pouw": {"device": "cuda", "kernel": "$PEARLC_KERNEL", "cubin": "$PEARLC_CUBIN"}}
    assert arm.to_json()["protocols_config"] == arm.protocols_config
    settings = spec.arm_settings[pearl]
    other = replace(settings, protocols_config={"pouw": {**settings.protocols_config["pouw"], "schedule": "serial"}})
    assert spec.config_sha256 != replace(spec, arm_settings={arm.id: other}).config_sha256
    assert resolve_arms(spec.arms)[1].control == "stock-fp8" and resolve_arms(spec.arms)[1].precision is None

    with pytest.raises(ValueError, match=r"\$PEARLC_CUBIN"):
        protocols_config(arm, {"PEARLC_KERNEL": "/tree/benchmarks/pouw/pearl_c_sm90"})
    monkeypatch.setattr(verifier, "coins", lambda n: b"\1" * n)
    config = protocols_config(arm, {"PEARLC_KERNEL": "/tree/benchmarks/pouw/pearl_c_sm90",
                                    "PEARLC_CUBIN": "/run/out/setup/pearl_c_sm90/pearl_c_sm90.cubin"})
    assert config["pouw"]["cubin"] == "/run/out/setup/pearl_c_sm90/pearl_c_sm90.cubin"
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

    spec = resolve_spec("llama32-1b-1gpu-pouw-protocol-served@h200")
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


def test_the_h200_pearl_c_setup_serves_without_batch_invariance_and_with_sm90s_bf16_rows_off(tmp_path):
    """The H200 spec's point, launched as `run_point.sh` launches it (its setup sourced, then the point's Python), with
    only the off-card build stood in: the serve gets no VLLM_BATCH_INVARIANT even when the job has one, and
    PEARLC_BF16_ROWS=0 unless the job sets it."""

    root = Path(__file__).resolve().parents[3]
    stand_in = tmp_path / "stand-in"
    stand_in.mkdir()
    (stand_in / "bash").write_text(f'#!/bin/sh\ncase "$1" in */pearl_c_sm90/build.sh) mkdir -p "$2" && '
                                   f': > "$2/pearl_c_sm90.cubin"; exit 0 ;; esac\nexec {shutil.which("bash")} "$@"\n')
    python = tmp_path / "venv312" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\necho "${VLLM_BATCH_INVARIANT-unset} $PEARLC_BF16_ROWS ${PEARLC_CUBIN#$RESEARCH_RUN_DIR/}"\n')
    for script in (stand_in / "bash", python):
        script.chmod(0o755)
    spec = resolve_spec("llama32-1b-1gpu-pouw-protocol-served@h200")
    # the setup reads the build's target from the run's machine's card (the fixture's is an H100: sm_90a)
    (tmp_path / "run" / "inputs").mkdir(parents=True)
    shutil.copy(Path(__file__).parent / "fixtures" / "machines" / "one-card.toml", tmp_path / "run" / "inputs" / "m.toml")
    env = {"PATH": f"{stand_in}:/usr/bin:/bin", "VERITY_BENCH_VENV": str(tmp_path / "venv312"), "CUDA_HOME": str(tmp_path),
           "RESEARCH_RUN_DIR": str(tmp_path / "run"), "RESEARCH_MACHINE": "m", "VLLM_BATCH_INVARIANT": "1",
           **jobs.point_environment(spec)}

    def served(**job: str) -> str:
        ran = subprocess.run(["bash", "benchmarks/harness/run_point.sh"], cwd=root, env={**env, **job},
                             capture_output=True, text=True)
        assert ran.returncode == 0, ran.stderr
        return ran.stdout.strip()

    assert served() == "unset 0 out/setup/pearl_c_sm90/pearl_c_sm90.cubin"
    assert served(PEARLC_BF16_ROWS="1") == "unset 1 out/setup/pearl_c_sm90/pearl_c_sm90.cubin"


def test_served_passes_arm_writes_that_arm_as_serve_openais_protocols_config(monkeypatch, tmp_path):
    """``served_passes.py arm`` writes the H200 spec's Pearl-C arm as serve-openai's ``--protocols-config`` takes it
    (`serve_openai.scheme_settings`): the scheme's own settings alone, the setup's paths read in, no challenges flipped
    (the server flips each epoch's); an unset ``$`` value is refused, naming it."""
    from benchmarks.pouw.pearl_c_vllm import served_passes as SP
    from verity.protocols.resource_accounting.compute.pouw.verifier import verifier
    from verity.protocols.resource_accounting.scheme import settings
    from verity_vllm import messages as M

    def flipped(n):
        raise AssertionError("served_passes.py arm flipped challenges")
    monkeypatch.setattr(verifier, "coins", flipped)
    spec, out = str(SPEC_ROOT / "llama32-1b-1gpu-pouw-protocol-served@h200.toml"), tmp_path / "config.json"
    monkeypatch.delenv("PEARLC_KERNEL", raising=False)
    monkeypatch.setenv("PEARLC_CUBIN", "/run/out/setup/pearl_c_sm90/pearl_c_sm90.cubin")
    with pytest.raises(ValueError, match=r"\$PEARLC_KERNEL"):
        SP.main(["arm", "--spec", spec, "--out", str(out)])
    monkeypatch.setenv("PEARLC_KERNEL", "/tree/benchmarks/pouw/pearl_c_sm90")
    assert SP.main(["arm", "--spec", spec, "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc == {"pouw": {"kernel": "/tree/benchmarks/pouw/pearl_c_sm90",
                            "cubin": "/run/out/setup/pearl_c_sm90/pearl_c_sm90.cubin"}}
    settings(M.option("pouw:pearl-c-h100-v1-h3"), {**doc["pouw"], "coins": "in-process"})


def test_served_passes_refuses_the_rtx_pro_6000s_pearl_c_arm_by_name():
    """No declared scheme serves Pearl-C on sm_120 (`pouw:pearl-c-sm120-v1-h3` is no declared scheme's option), so
    ``served_passes.py`` refuses an arm of that entry by name (`undeclared`, which `pearl_c_arm` asks before any engine or
    challenger), and its default spec is Pearl+'s."""
    from benchmarks.pouw.pearl_c_vllm import served_passes as SP

    assert "pouw:pearl-c-sm120-v1-h3: no declared scheme serves Pearl-C on that card" in SP.undeclared(
        ["pouw:pearl-c-sm120-v1-h3"])
    assert SP.undeclared(["pouw:pearl-c-h100-v1-h3"]) is None
    assert SP.SPEC.name == "llama32-1b-1gpu-pouw-protocol-served.toml"


def test_served_passes_serve_attaches_pearl_plus_by_name_with_its_settings_and_a_live_sessions_ctx(monkeypatch, tmp_path):
    """``served_passes.py serve`` attaches the arm's declared scheme on the build's exchange, not through the adapter's
    option: the spec's arm settings that are Pearl+'s, its passes retained under DIR, and its salt in-process; under
    the challenger gate, a session an epoch under the lane as the peer, with the gate, its ledger and the private
    directory in its ctx, and refused by name without PEARLC_PRIVATE."""
    from benchmarks.pouw.pearl_c_vllm import served_passes as SP
    from verity.protocols.resource_accounting.scheme import settings
    from verity_vllm import messages as M

    arm = SP.pearl_c_arm(resolve_spec("llama32-1b-1gpu-pouw-protocol-served@h200"))
    scheme = M.option(arm.protocols[0])
    monkeypatch.setenv("PEARLC_KERNEL", "/tree/k")
    monkeypatch.setenv("PEARLC_CUBIN", "/run/k.cubin")
    own, ctx = SP.scheme_settings(arm, scheme, tmp_path)
    assert (own, ctx) == ({"kernel": "/tree/k", "cubin": "/run/k.cubin", "transcript_dir": str(tmp_path),
                           "coins": "in-process"}, {})
    settings(scheme, own)

    class Gate:
        def __init__(self, command):
            self.command = command

        def call(self, *words):
            return {}

        def ledger(self):
            return []

    import verity_vllm.pipeline.commit_coins as CC

    monkeypatch.setattr(CC, "ChallengerGate", Gate)
    monkeypatch.setenv("PEARLC_CHALLENGER_GATE", "ssh gate")
    monkeypatch.setenv("PEARLC_CHALLENGER_PEER", "node-1")
    monkeypatch.delenv("PEARLC_PRIVATE", raising=False)
    with pytest.raises(SystemExit, match="PEARLC_PRIVATE"):
        SP.scheme_settings(arm, scheme, None)
    monkeypatch.setenv("PEARLC_PRIVATE", str(tmp_path / "private"))
    own, ctx = SP.scheme_settings(arm, scheme, None)
    assert {k: own[k] for k in ("challenger", "lane", "peer")} == {"challenger": "127.0.0.1:0", "lane": "vllm",
                                                                   "peer": "node-1"}
    assert "coins" not in own and "transcript_dir" not in own and ctx["private"] == str(tmp_path / "private")
    assert ctx["gate"]() == {} and ctx["ledger"]() == []
    settings(scheme, own)


def test_arm_settings_name_a_protocol_arm_of_the_spec_and_a_control_it_runs(tmp_path):
    spec_id = "llama32-1b-1gpu-pouw-protocol-served"
    source = (SPEC_ROOT / f"{spec_id}.toml").read_text()
    path = tmp_path / "spec.toml"
    for control, want in (("stock-fp8", "stock-fp8"), ("none", None)):
        path.write_text(source.replace('control = "none"', f'control = "{control}"', 1))
        spec = load_spec(path)
        assert resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)[1].control == want
    for old, new, why in (('arms = ["stock-fp8", "pouw:pearl-c-h100-v1-h3", "stock-fp8@compiled"]',
                           'arms = ["stock-fp8", "pouw:ncp-v2", "stock-fp8@compiled"]', "must name a protocol arm"),
                          ('[arm_settings."pouw:pearl-c-h100-v1-h3"]\n', '[arm_settings.stock]\n', "must name a protocol arm"),
                          ('precision = "bfloat16"', 'precision = "int4"', "precision is one of"),
                          ('control = "none"', 'control = "stock"', "control must be another arm"),
                          ('control = "none"', 'control = "pouw:pearl-c-h100-v1-h3"', "control must be another arm")):
        assert old in source
        path.write_text(source.replace(old, new, 1))
        with pytest.raises(ValueError, match=why):
            load_spec(path)
    with pytest.raises(ValueError, match="not in --arms"):
        resolve_arms("stock-fp8", (), load_spec(spec_id).arm_settings)


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
    plain = SimpleNamespace(release=lambda: None)
    llm = SimpleNamespace(protocols=SimpleNamespace(installed=[plain, option]))
    assert _pearl_c_run(llm) is run
    assert _pass_totals(run) == dict.fromkeys(("passes", *PASS_COUNTS), 0)
    run.end_pass()
    run.end_pass()
    assert _pass_totals(run) == {"passes": 2, "units": 448, "tiles": 10000, "screened_rows": 128,
                                        "failing_rows": 0, "unsure_rows": 2}
    other = SimpleNamespace(protocols=SimpleNamespace(installed=[plain]))
    assert _pearl_c_run(other) is None
    assert _pearl_c_run(SimpleNamespace(protocols=None)) is None


def test_pearl_cs_served_row_is_its_protocol_row_beside_the_compiled_engine():
    """`llama32-1b-1gpu-pouw-protocol-served` measures Pearl-C over the compiled engine in one run, so every arm's engine
    shares its step budget, sequence bound and KV pool."""

    spec = resolve_spec("llama32-1b-1gpu-pouw-protocol-served@h200")
    pearl = "pouw:pearl-c-h100-v1-h3"
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
        assert built[0]["kv_cache_memory_bytes"] == 8 << 30 and built[0]["max_num_batched_tokens"] == 8192
    assert engine_args(resolve_spec("llama31-8b-fp8-1gpu").preset, "fp8", resident=True, eager=True)["max_num_seqs"] == 128


def test_pearl_cs_whole_step_arm_builds_vllms_full_decode_graphs_beside_the_compiled_engine(monkeypatch):
    """`llama31-8b-1gpu-pearl-c-whole-served` times the whole-step arm beside the compiled engine, on
    `llama31-8b-fp8-1gpu-compiled-full`'s preset. Under the eager spec, the arm's overrides build vLLM's FULL decode graphs at the preset's capture sizes
    without torch.compile, at the `compiled` label, which `verity_vllm.LLM` accepts, and its run takes the graph path's
    options and serial schedule."""

    from verity_vllm.llm import resolve_options

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    spec = resolve_spec("llama31-8b-1gpu-pearl-c-whole-served")
    full = resolve_spec("llama31-8b-fp8-1gpu-compiled-full").preset
    assert dict(spec.preset.engine_settings) == {**full.engine_settings, "max_num_seqs": 128}
    assert replace(spec.preset, id=full.id, gpu_id=full.gpu_id, engine_settings=full.engine_settings) == full
    arms = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
    assert [(a.id, a.precision, a.control, a.execution, a.experimental) for a in arms] == [
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
    assert "whole_prefetch_b" not in type(whole)(whole.protocol).run_options
    monkeypatch.setenv("PEARLC_WHOLE_EAGER", "deferred")
    assert type(whole)(whole.protocol).run_options["whole_eager"] == "deferred"
    monkeypatch.setenv("PEARLC_WHOLE_EAGER", "0")
    assert "whole_eager" not in type(whole)(whole.protocol).run_options


def test_the_1b_whole_step_spec_is_the_1b_pouw_spec_on_the_h100_with_the_whole_step_arm_beside():
    whole, protocol = (tomllib.loads((SPEC_ROOT / f"{name}.toml").read_text())
                       for name in ("llama32-1b-1gpu-pearl-c-whole-served", "llama32-1b-1gpu-pouw-protocol-served"))
    assert (whole.pop("cards"), protocol.pop("cards")) == (["h100"], ["h200", "h100"])
    assert whole.pop("arms") == [*protocol.pop("arms"), "pouw-pearl-c-whole"]
    assert whole.pop("arm_modules") == ["benchmarks.pouw.pearl_c_vllm.bench_arm:PEARL_C_WHOLE"]
    whole["id"] = whole["preset"]["id"] = protocol["id"]
    assert whole == protocol


@pytest.mark.parametrize("pinned", [True, False])
def test_the_kernel_scheme_a_served_scheme_runs_on_a_card_is_its_declared_rows_name(pinned):
    """`kernel_scheme` reads the scheme's declared ``kernels``, as the PoUW arm does (`bench_arm.py`'s
    ``kernel_scheme(self.protocol, card, pinned=True)``); Pearl+ has no build for the RTX PRO 6000's architecture."""
    from verity_bench.schemes import kernel_scheme

    assert kernel_scheme("pearl-plus", "h100", pinned=pinned) == kernel_scheme("pearl-plus", "h200", pinned=pinned) \
        == "pearl-c-h100-v1-h3"
    with pytest.raises(LookupError, match="^rtx-pro-6000's architecture sm_120a has no build here: pearl-c-h100-v1-h3 is "
                                          "built for sm_90a$"):
        kernel_scheme("pearl-plus", "rtx-pro-6000", pinned=pinned)


def test_pearl_cs_whole_step_arm_on_an_h200_runs_pearl_c_sm90s_kernels_under_the_h100_record(monkeypatch):
    """`llama31-8b-1gpu-pearl-c-whole-served` on one H200 is its preset there, the whole-step arm beside compiled stock
    alone, its setup building pearl_c_sm90's cubin; the arm takes the useful-work scheme by its registered name
    and, on a card of the H200's target, the kernel scheme Pearl+ runs there (`pearl-c-h100-v1-h3`) from the tree's
    pearl_c_sm90 directory or PEARLC_KERNEL and its build.sh cubin, and builds the same engine."""
    from verity_bench.schemes import arch

    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root))
    spec = resolve_spec("llama31-8b-1gpu-pearl-c-whole-served@h200")
    assert spec == resolve_spec("llama31-8b-1gpu-pearl-c-whole-served") and spec.preset.gpu_id == "h200"
    assert (root / spec.setup).is_file() and "pearl_c_sm90/build.sh" in (root / spec.setup).read_text()
    arms = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
    assert [(a.id, a.execution, a.experimental) for a in arms] == [("stock-fp8@compiled", "compiled", False),
                                                                   ("pouw-pearl-c-whole", None, True)]
    assert dict(spec.arm_settings) == {"stock-fp8@compiled": ArmSettings(control="none")}
    from benchmarks.pouw.pearl_c_vllm import bench_arm
    assert bench_arm.PEARL_C_WHOLE.protocol == bench_arm.PEARL_C.protocol == "pearl-plus"
    assert arch(spec.preset.gpu_id) == arch("h100")
    monkeypatch.delenv("PEARLC_KERNEL", raising=False)
    monkeypatch.delenv("PEARLC_CUBIN", raising=False)
    with pytest.raises(NotImplementedError, match=r"^dither is not served: no adapter entry attaches it to a served run"):
        bench_arm.PearlCArm("dither").kernel("h100")
    with pytest.raises(ValueError, match=r"pouw-pearl-c names no useful-work scheme: build it as PearlCArm\(NAME\)"):
        bench_arm.PearlCArm().kernel("h100")
    monkeypatch.setenv("PEARLC_KERNEL", str(root / "elsewhere"))
    assert bench_arm.PearlCWholeArm("pearl-plus").kernel("h100") == ("pearl-c-h100-v1-h3", root / "elsewhere")
    monkeypatch.delenv("PEARLC_KERNEL")
    whole = bench_arm.PearlCWholeArm("pearl-plus")
    assert whole.kernel("h100") == ("pearl-c-h100-v1-h3", bench_arm.KERNELS / "pearl_c_sm90")
    assert whole.kernel_dir.is_dir() and whole.scheme == "pearl-c-h100-v1-h3"
    assert whole.kernel_digest(None) is None or (whole.kernel_dir / "pearl_c_sm90.cubin").is_file()
    preset, eager = served(spec.preset, spec.eager)
    preset, eager = arm_execution(arms[-1], preset, eager)
    built = {**engine_args(preset, "bfloat16", resident=False, eager=eager), **whole.engine_overrides(None)}
    assert built["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert whole.run_options == {"graphs": True, "whole_defer": True, "whole_defer_a": True, "whole_fork_stats": True,
                                 "lag_ends": True, "whole_tree_sm": True, "aside_ends": True} and whole.schedule == "serial"
