"""`pipeline.engine_spec` and `LLM.from_spec`: a strict config v0 document builds the same `LLM` as its keywords would.

* ``[engine]`` is exactly `LLM`'s supported options; an unknown or missing key, a wrongly typed value or another schema
  refuses before any engine is built;
* only the keys a spec sets reach `LLM`, arrays as lists;
* the digest changes with every option but ``protocols_config``, which `LLM` never hashes;
* `LLM.from_spec` records it as ``engine_record["spec_digest"]``;
* every pinned preset under ``benchmarks/specs`` resolves as an engine spec to its checkpoint pin.
"""
from __future__ import annotations

import tomllib
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import verity_config

from verity_vllm.config import EnginePin
from verity_vllm.pipeline import engine_spec as ES
from verity_vllm.pipeline import llm as L

LLAMA = ("unsloth/Llama-3.2-1B", "9535bd9b1d1dea6acafbdc4813b728796aeb28da")
BENCH_SPECS = Path(__file__).resolve().parents[4] / "benchmarks" / "specs"

SPEC = f'''schema = "verity-vllm/engine-spec/v0"
model = "{LLAMA[0]}"
revision = "{LLAMA[1]}"

[engine]
quantization = "fp8"
enforce_eager = false
max_num_seqs = 4
compilation_config = {{ mode = 0, cudagraph_capture_sizes = [1, 8] }}
'''


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "engine.toml"
    p.write_text(text)
    return p


def test_the_engine_table_is_exactly_the_llm_options():
    assert {f.name for f in fields(ES.Engine)} == set(L.SUPPORTED)


def test_a_spec_sets_only_its_keys(tmp_path):
    spec = ES.load(_write(tmp_path, SPEC))
    assert (spec.model, spec.revision) == LLAMA
    assert spec.options() == {"quantization": "fp8", "enforce_eager": False, "max_num_seqs": 4,
                              "compilation_config": {"mode": 0, "cudagraph_capture_sizes": [1, 8]}}


@pytest.mark.parametrize("edit, why", [
    (lambda s: s + "kv_cache_dtype = \"fp8\"\n", "unknown key"),
    (lambda s: s.replace(f'revision = "{LLAMA[1]}"\n', ""), "missing required key 'revision'"),
    (lambda s: s.replace("max_num_seqs = 4", "max_num_seqs = \"4\""), "must be an integer"),
    (lambda s: s.replace("engine-spec/v0", "engine-spec/v1"), "schema must be"),
])
def test_a_spec_not_of_the_declared_shape_refuses(tmp_path, edit, why):
    with pytest.raises(verity_config.ConfigError, match=why):
        ES.load(_write(tmp_path, edit(SPEC)))


def test_the_digest_covers_every_option_but_protocols_config(tmp_path):
    spec = ES.load(_write(tmp_path, SPEC))
    assert spec.digest == ES.load(_write(tmp_path, SPEC)).digest
    assert replace(spec, engine=replace(spec.engine, seed=1)).digest != spec.digest
    assert replace(spec, revision="0" * 40).digest != spec.digest
    assert replace(spec, engine=replace(spec.engine, protocols_config={"pous": {"cpu": 3}})).digest == spec.digest


def test_from_spec_builds_the_llm_of_its_keywords_and_records_the_digest(tmp_path, monkeypatch):
    from verity_vllm.engine import vllm_adapter as va
    seen = []

    def build_engine(checkpoints, **kw):
        seen.append(kw)
        return SimpleNamespace(llm=None, model=None, kwargs={}, env_flags={}, execution=kw["execution"], effective={}, build_s=0.0)

    monkeypatch.setattr(va, "build_engine", build_engine)
    path = _write(tmp_path, SPEC)
    llm = L.LLM.from_spec(path)
    keywords = L.LLM(LLAMA[0], revision=LLAMA[1], **ES.load(path).options())
    assert seen[0] == seen[1]
    assert seen[0]["max_num_seqs"] == 4 and seen[0]["engine_args"]["quantization"] == "fp8"
    assert llm.engine_record["spec_digest"] == ES.load(path).digest
    assert keywords.engine_record["spec_digest"] is None


def test_a_refused_spec_builds_no_engine(tmp_path, monkeypatch):
    from verity_vllm.engine import vllm_adapter as va
    monkeypatch.setattr(va, "build_engine", lambda *a, **kw: pytest.fail("built an engine for a refused spec"))
    with pytest.raises(verity_config.ConfigError):
        L.LLM.from_spec(_write(tmp_path, SPEC + "bogus = 1\n"))


def _presets() -> list[dict]:
    out = []
    for p in sorted(BENCH_SPECS.glob("*.toml")):
        doc = tomllib.loads(p.read_text())
        if doc.get("schema") == "verity-bench/vllm-spec/v0" and doc["preset"].get("revision"):
            out.append(doc["preset"])
    return out


@pytest.mark.parametrize("preset", _presets(), ids=lambda p: p["id"])
def test_every_pinned_benchmark_preset_resolves_as_an_engine_spec(preset):
    spec = ES.from_mapping({"schema": ES.SCHEMA, "model": preset["model"], "revision": preset["revision"],
                            "engine": {"tensor_parallel_size": preset["gpu_count"],
                                       "max_model_len": preset["engine_settings"]["max_model_len"]}})
    pin = EnginePin.resolve(spec.model, spec.revision, world=spec.engine.tensor_parallel_size)
    assert pin.case == preset["case"]
