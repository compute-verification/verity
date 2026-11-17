"""Arms that override their own engine and sampling, the two arms at `vllm serve`'s defaults (`stock@serve`,
`stock-fp8@serve`), the runner's `alone` lifetime, and the table of where an arm differs from `vllm serve`."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from bench_standins import FakeEngine, FakeFactory, inputs
from verity_config import ConfigError
from verity_config.canonical import dumps, value
from verity_vllm.engine.env import ENGINE
from verity_vllm.engine.hooks import Hooks
from verity_vllm.engine.profiles.dense_generic import pinned_hf_config

from verity_bench import steady_spec
from verity_bench.vllm import engine as E
from verity_bench.vllm import result, serve_defaults as S
from verity_bench.vllm.arms import BUILTINS, Arm, Target, resolve_arms
from verity_bench.vllm.presets import SPEC_ROOT, VLLM_PIN, SweepPoint, load_spec, resolve_spec, served
from verity_bench.vllm.runner import ENGINE_IDENTITY, run_benchmark
from verity_bench.vllm.table import render

ROOT = Path(__file__).resolve().parents[3]
SERVE = ("stock@serve", "stock-fp8@serve")
GAINED = {"llama32-1b-1gpu-pouw-protocol-served-h100", "llama32-1b-1gpu-pouw-protocol-served-h200",
          "llama32-1b-bf16-h100-served-arm", "llama32-1b-bf16-h200-served-arm"}
BANDWIDTH = "llama32-1b-fp8-1gpu-bandwidth-served-h200"
PEARL_H200 = "llama32-1b-1gpu-pouw-protocol-served-h200"


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _specs():
    for path in sorted(SPEC_ROOT.glob("*.toml")):
        if path.name != "pyproject.toml" and not steady_spec.is_steady(path) and "vllm-sweep/v0" not in path.read_text():
            yield path.stem


def test_every_existing_spec_and_arm_keeps_its_identity_and_engine_byte_for_byte(monkeypatch):
    """`fixtures/spec_identities.json` was recorded from main (7db05656deb) before arms had overrides: each spec's
    `config_sha256` and resolved-spec digest, and each arm's record (`Arm.to_json`) and engine arguments resident and in
    blocks. A spec that gained the serve arms is its old self once they are taken out, and nothing else moved."""

    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks"))
    recorded = json.loads((Path(__file__).parent / "fixtures/spec_identities.json").read_text())
    assert set(_specs()) == set(recorded) | {f"{BANDWIDTH}-serve"}
    for name, was in recorded.items():
        spec = resolve_spec(name)
        assert (name in GAINED) == any(a in SERVE for a in spec.arms), name
        spec = replace(spec, arms=tuple(a for a in spec.arms if a not in SERVE))
        assert spec.config_sha256 == was["config_sha256"], name
        assert hashlib.sha256(dumps(value(spec))).hexdigest() == was["resolved_spec_sha256"], name
        preset, eager = served(spec.preset, spec.eager)
        arms = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
        assert {a.id for a in arms} == set(was["arms"]), name
        for arm in arms:
            p, e = E.arm_execution(arm, preset, eager)
            built = {str(r): E.engine_args(p, arm.precision or p.precision, resident=r, eager=e) for r in (True, False)}
            assert {"arm": _digest(arm.to_json()), "engine_args": _digest(built)} == was["arms"][arm.id], (name, arm.id)


def test_a_spec_that_gained_the_serve_arms_keeps_its_id_and_points_keep_their_shape():
    for name in GAINED:
        spec = load_spec(name)
        assert spec.id == name and spec.arms[-2:] == SERVE


def test_the_bandwidth_member_keeps_its_arms_and_its_serve_copy_is_it_with_the_serve_arms():
    member, copy = load_spec(BANDWIDTH), load_spec(f"{BANDWIDTH}-serve")
    assert copy.arms == (*member.arms, *SERVE)
    assert replace(copy, id=member.id, arms=member.arms, preset=replace(copy.preset, id=member.preset.id)) == member


def _spec_with(tmp_path, extra: str) -> Path:
    source = (SPEC_ROOT / "llama31-8b-fp8-1gpu-compiled-full.toml").read_text()
    section = '[arm_settings."stock-fp8@compiled"]\ncontrol = "none"'
    assert section in source
    path = tmp_path / "spec.toml"
    path.write_text(source.replace(section, section + "\n" + extra))
    return path


def test_an_arms_overrides_load_strictly_and_enter_its_record_and_the_specs_digest(tmp_path):
    plain = resolve_spec("llama31-8b-fp8-1gpu-compiled-full")
    path = _spec_with(tmp_path, '\n[arm_settings."stock-fp8@compiled".engine]\nmax_num_seqs = 256\n'
                                'enable_prefix_caching = true\nkv_cache_memory_bytes = "vllm"\n'
                                '\n[arm_settings."stock-fp8@compiled".sampling]\ndetokenize = true\nseed = "vllm"\n')
    spec = resolve_spec(path)
    assert spec.config_sha256 != plain.config_sha256
    arm = resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)[-1]
    assert arm.engine == {"max_num_seqs": 256, "enable_prefix_caching": True, "kv_cache_memory_bytes": "vllm"}
    assert arm.to_json()["sampling"] == {"detokenize": True, "seed": "vllm"} and arm.control is None
    preset, eager = served(spec.preset, spec.eager)
    p, e = E.arm_execution(arm, preset, eager)
    args = E.engine_args(p, arm.precision, resident=True, eager=e, overrides=S.resolve(arm.engine, {}))
    assert args["max_num_seqs"] == 256 and args["enable_prefix_caching"] is True and "kv_cache_memory_bytes" not in args
    for extra, why in (('\n[arm_settings."stock-fp8@compiled".engine]\nmax_num_seqz = 2\n', "not one an arm overrides"),
                       ('\n[arm_settings."stock-fp8@compiled".engine]\ngpu_memory_utilization = 2.0\n', r"\(0, 1\]"),
                       ('\n[arm_settings."stock-fp8@compiled".sampling]\ntemperature = "vllm"\n', "a float at least 0")):
        with pytest.raises(ConfigError, match=why):
            load_spec(_spec_with(tmp_path, extra))


def test_two_arms_differing_only_in_overrides_never_share_an_engine_identity():
    assert "arm_settings" in ENGINE_IDENTITY
    one, two = (Arm("x", engine={"max_num_seqs": n}).to_json() for n in (128, 256))
    assert one != two and "engine" not in Arm("x").to_json() and "alone" not in Arm("x").to_json()


def _context(tmp_path, memory=140 * S.GiB, name="NVIDIA H200"):
    (tmp_path / "generation_config.json").write_text(json.dumps({"do_sample": True, "max_length": 131072,
                                                                 "temperature": 0.6, "top_p": 0.9}))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 131072}))
    preset = resolve_spec(PEARL_H200).preset
    return S.context(memory, name, pinned_hf_config(preset.model, preset.revision), tmp_path)


def _serve_args(arm, ctx, spec=PEARL_H200):
    spec = resolve_spec(spec)
    preset, eager = served(spec.preset, spec.eager)
    p, e = E.arm_execution(arm, preset, eager)
    return E.engine_args(p, arm.precision or p.precision, resident=False, eager=e,
                         overrides=S.resolve(arm.engine or {}, ctx))


@pytest.mark.parametrize("memory, name", [(140 * S.GiB, "NVIDIA H200"), (80 * S.GiB, "NVIDIA H100 80GB HBM3")])
def test_the_serve_arms_build_vllm_serves_engine_on_both_cards(tmp_path, memory, name):
    ctx = _context(tmp_path, memory, name)
    assert ctx["max_model_len"] == 131072 and ctx["generation"] == {"temperature": 0.6, "top_p": 0.9}
    bf16, fp8 = (_serve_args(BUILTINS[a], ctx) for a in SERVE)
    for args in (bf16, fp8):
        assert {k: args.get(k) for k in ("max_num_seqs", "max_num_batched_tokens", "max_model_len", "gpu_memory_utilization",
                                         "enable_prefix_caching", "dtype", "execution", "enforce_eager")} == {
            "max_num_seqs": 1024, "max_num_batched_tokens": 8192, "max_model_len": 131072, "gpu_memory_utilization": 0.92,
            "enable_prefix_caching": True, "dtype": "auto", "execution": "compiled", "enforce_eager": False}
        assert "kv_cache_memory_bytes" not in args and "compilation_config" not in args
    assert "quantization" not in bf16 and fp8["quantization"] == "fp8"
    assert all(BUILTINS[a].alone and BUILTINS[a].sampling == {"detokenize": True} and E.serves(BUILTINS[a]) for a in SERVE)
    assert not E.serves(BUILTINS["stock-fp8"])


def test_vllm_serves_scheduler_limits_follow_the_card_and_its_context_the_checkpoint():
    assert S.card(180 * S.GiB, "NVIDIA B200")["max_num_batched_tokens"] == 16384
    assert S.card(80 * S.GiB, "NVIDIA A100-SXM4-80GB")["max_num_seqs"] == 256
    assert S.max_model_len({"max_position_embeddings": 4096, "seq_length": 2048}) == 2048
    assert S.max_model_len({"max_position_embeddings": 131072, "rope_scaling": {"rope_type": "llama3", "factor": 32}},
                           {"model_max_length": 8192}) == 8192
    with pytest.raises(ValueError, match="by hand"):
        S.max_model_len({"max_position_embeddings": 4096, "rope_scaling": {"rope_type": "yarn", "factor": 4}})
    assert S.generation({"top_k": 50, "temperature": 0.7, "max_length": 9}) == {"temperature": 0.7}


def _differences(arm, ctx, spec=PEARL_H200):
    args = _serve_args(arm, ctx, spec)
    sampling = {**E.REQUEST, **S.resolve_sampling(arm.sampling or {})}
    fresh = bool(args.get("execution")) and not E.serves(arm)
    return S.differences(S.settings(args, sampling, ENGINE, ctx, compile_fresh=fresh), ctx)


def test_stock_at_serve_differs_from_vllm_serve_only_in_the_workloads_requests_and_the_adapters_switches(tmp_path):
    ctx = _context(tmp_path)
    rows = {r["setting"]: r for r in _differences(BUILTINS["stock@serve"], ctx)}
    assert {k for k, r in rows.items() if r["whose"] == S.WORKLOAD} == {
        "temperature", "top_p", "max_tokens", "ignore_eos", "request_seed"}
    assert {k for k, r in rows.items() if r["whose"] == S.ADAPTER} == {
        "VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_USE_FLASHINFER_SAMPLER", "VLLM_DISABLE_SHARED_EXPERTS_STREAM",
        "TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK", "enable_flashinfer_autotune", "VLLM_DISABLE_COMPILE_CACHE"}
    assert not [k for k, r in rows.items() if r["whose"] == S.HARNESS]
    assert (rows["temperature"]["arm"], rows["temperature"]["vllm_serve"]) == (0.0, 0.6)
    eager = {r["setting"]: r for r in _differences(BUILTINS["stock-fp8"], ctx)}
    assert {"enforce_eager", "VLLM_BATCH_INVARIANT", "max_num_seqs", "max_model_len", "gpu_memory_utilization",
            "kv_cache_memory_bytes", "enable_prefix_caching", "quantization", "dtype", "detokenize"} <= set(eager)
    assert {k for k, r in eager.items() if r["whose"] == S.ADAPTER} == {k for k, r in rows.items() if r["whose"] == S.ADAPTER}


def test_every_switch_the_adapter_pins_has_vllms_default_beside_it():
    assert set(ENGINE) <= set(S.DEFAULTS)
    assert all(S.DEFAULTS[k].whose == S.ADAPTER for k in ENGINE)


def test_the_defaults_were_read_at_the_vllm_the_pods_install():
    """A vLLM bump changes `pod_bootstrap.sh`'s wheel, and this fails until `serve_defaults.DEFAULTS` is read again at the
    new commit and `VLLM_COMMIT` names it."""

    bootstrap = (ROOT / "adapters/vllm/verity_vllm/ops/pod_bootstrap.sh").read_text()
    [commit] = set(re.findall(r"wheels\.vllm\.ai/([0-9a-f]{40})/", bootstrap))
    assert S.VLLM_COMMIT == commit == VLLM_PIN
    assert f"+g{commit[:9]}" in bootstrap


class _Llm:
    def __init__(self, events, reset=True):
        self.events, self.reset = events, reset
        hf = SimpleNamespace(vocab_size=1000)
        self.llm_engine = SimpleNamespace(vllm_config=SimpleNamespace(model_config=SimpleNamespace(hf_config=hf)))

    def reset_prefix_cache(self):
        self.events.append("reset")
        return self.reset


def _engine(monkeypatch, arm, prefix_caching, reset=True):
    events: list = []
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw))
    import torch
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(E, "_allocator_counts", lambda: {})

    def arrivals(llm, requests, params, **_):
        events.append(("run", params(requests[0])))
        out = SimpleNamespace(outputs=[SimpleNamespace(token_ids=[5, 6])], num_cached_tokens=0)
        return {r.request_id: out for r in requests}, {"requests": {r.request_id: {"time_to_first_token_s": 0.1}
                                                                   for r in requests}}

    monkeypatch.setattr(E, "run_requests_arrivals", arrivals)
    engine = object.__new__(E._VllmEngine)
    engine.arm, engine.obj, engine.installed, engine.pearl_c, engine.ready_at = arm, None, None, None, 0.0
    engine.target = Target(arm, None, None, Hooks(tag="t"))
    engine.llm = SimpleNamespace(vllm=_Llm(events, reset), protocols=None)
    engine.sampling = {**E.REQUEST, **S.resolve_sampling(arm.sampling or {})}
    engine.prefix_caching = prefix_caching
    return engine, events


def test_a_prefix_caching_engine_empties_its_cache_before_every_run_and_refuses_when_vllm_cant(monkeypatch):
    """Every run repeats its point's prompts, so a serve arm's cache would serve its timed reps from its warm-up's blocks:
    the cache is emptied before each run, and the requests stay those every arm serves."""

    point = SweepPoint(4, 32, "mixed", 8)
    engine, events = _engine(monkeypatch, BUILTINS["stock@serve"], True)
    engine.gate(point)
    assert events[0] == "reset" and events[1][0] == "run"
    assert events[1][1] == {"max_tokens": 8, "seed": 0, "ignore_eos": True, "detokenize": True, "temperature": 0.0}
    engine, events = _engine(monkeypatch, BUILTINS["stock-fp8"], False)
    engine.gate(point)
    assert events == [("run", {"max_tokens": 8, "seed": 0, "ignore_eos": True, "detokenize": False, "temperature": 0.0})]
    engine, _ = _engine(monkeypatch, BUILTINS["stock@serve"], True, reset=False)
    with pytest.raises(RuntimeError, match="did not reset its prefix cache"):
        engine.gate(point)


def _arms():
    preset, points, arms = inputs()
    return preset, points, (*arms, BUILTINS["stock@serve"])


def test_an_alone_arm_is_gated_and_timed_in_its_own_lifetime_after_the_resident_arms():
    preset, points, arms = _arms()
    factory = FakeFactory()
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=factory, git_commit="a" * 40)
    creates = [e for e in factory.events if e[0] == "create"]
    assert creates == [("create", "stock-fp8", True), ("create", "candidate", True), ("create", "stock@serve", False)]
    assert factory.events.index(("close", "stock-fp8")) < factory.events.index(("create", "stock@serve", False))
    alone = [s for s in doc["execution"]["schedule"] if s.get("arm") == "stock@serve"]
    assert {s["mode"] for s in alone} == {"alone"} and [s["rep"] for s in alone if s["phase"] == "timed"] == [0, 1]
    assert doc["execution"]["interleave"] == "resident" and doc["execution"]["alone"]["arms"] == ["stock@serve"]
    assert {"kind": "timed-tokens", "arm": "stock@serve", "control": None, "against": "gate", "point": points[0].id,
            "passed": True} in doc["gates"]
    assert "stock@serve" in doc["points"][0]["arms"] and doc["baseline"] == "stock-fp8"
    [served_doc] = [a for a in doc["identity"]["arms"] if a["id"] == "stock@serve"]
    assert served_doc["alone"] is True and served_doc["engine"]["gpu_memory_utilization"] == "vllm"


def test_an_alone_arm_follows_blocks_too_and_a_failed_gate_times_it_never():
    preset, points, arms = _arms()
    factory = FakeFactory(resident_failure="candidate")
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=factory, git_commit="a" * 40)
    assert doc["execution"]["interleave"] == "blocks" and doc["points"]
    assert [e for e in factory.events if e[0] == "create"][-1] == ("create", "stock@serve", False)
    factory = FakeFactory(bad_tokens="candidate")
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=factory, git_commit="a" * 40)
    assert not doc["points"] and ("gate", "stock@serve", points[0].id) in factory.events
    assert not [e for e in factory.events if e[:2] in (("warmup", "stock@serve"), ("measure", "stock@serve"))]
    with pytest.raises(ValueError, match="no arm's control can be one"):
        run_benchmark(preset, (*arms, Arm("x", control="stock@serve")), points, sweep="quick", reps=1, factory=FakeFactory(),
                      git_commit="a" * 40)


class _Reporting(FakeFactory):
    def create(self, arm, preset, *, resident):
        engine: FakeEngine = super().create(arm, preset, resident=resident)
        rows = [{"setting": "enable_prefix_caching", "arm": arm.alone, "vllm_serve": True, "whose": S.HARNESS,
                 "where": "here"}] if not arm.alone else []
        engine.identity = {**engine.identity, "vllm_serve": {"context": {"card": "NVIDIA H200"}, "differences": rows,
                                                            **({"note": S.NOTE} if arm.alone else {})}}
        return engine


def test_a_result_records_and_renders_where_each_arm_differs_from_vllm_serve_and_a_merge_keeps_it():
    preset, points, arms = _arms()
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=_Reporting(), git_commit="a" * 40)
    serve = doc["vllm_serve"]
    assert serve["vllm"] == S.VLLM_COMMIT and serve["context"] == {"card": "NVIDIA H200"}
    assert set(serve["arms"]) == {a.id for a in arms} and "context" not in serve["arms"]["stock-fp8"]
    table = render(doc)
    assert "`stock@serve`: the requests are the workload's" in table and "| enable_prefix_caching | False | True |" in table
    other = run_benchmark(preset, arms, (SweepPoint(8, 512, "mixed", 64),), sweep="quick", reps=2, factory=_Reporting(),
                          git_commit="a" * 40)
    assert result.merge([doc, other])["vllm_serve"] == serve
    other["vllm_serve"]["context"]["card"] = "NVIDIA H100"
    with pytest.raises(result.IdentityMismatch, match="vllm serve"):
        result.merge([doc, other])
    assert "vllm_serve" not in run_benchmark(preset, arms, points, sweep="quick", reps=1, factory=FakeFactory(),
                                             git_commit="a" * 40)
