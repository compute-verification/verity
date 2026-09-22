"""Lane IT: `binding.select_semantic_profile` against every committed production manifest (independent-tests.md §2.4).

The selector was written for the verifier's `--semantic-profile auto` and unit-tested on synthetic manifests
(`test_semantic_profile_selection.py`). Here it sees the 45 real production manifests under `fixtures/results/*/production/`
and must, for each: H100 Qwen2 eager rows -> `qwen2-eager-batch-invariant-hopper-v1`; Ada/A100 rows -> the `-v2` twins
(`qwen2-…-v2`, `llama-…-v2`); never a `status: frozen` v1; and refuse (BindingError, naming the unmet requirement) every row
no shipped profile covers: compiled (`enforce_eager: false`), TP=2, the MoE model.

The checkpoint `config.json` is not local for the Qwen models (only SmolLM2 is in the HF cache), so the architecture is
taken from a repo -> architecture table; the selector reads nothing else from the config.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_vllm_adapter import binding as vbinding

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "fixtures" / "results"
ARCH = {"HuggingFaceTB/SmolLM2-135M": "LlamaForCausalLM", "Qwen/Qwen2.5-1.5B": "Qwen2ForCausalLM",
        "Qwen/Qwen2.5-7B": "Qwen2ForCausalLM", "Qwen/Qwen1.5-MoE-A2.7B-Chat": "Qwen2MoeForCausalLM"}
MANIFESTS = sorted(RESULTS.glob("*/production/manifest.json"))


def expected(m: dict) -> str | None:
    """The profile the brief expects, or None for 'must refuse'."""
    arch = ARCH[m["model"]["repo"]]
    comp = m.get("compilation") or {}
    tp = (m.get("parallelism") or {}).get("tensor_parallel_size")
    eager = comp.get("enforce_eager") is True and comp.get("cudagraph_mode") == "NONE"
    if not eager or tp != 1 or m.get("dtype") != "bfloat16" or m.get("quantization") is not None or (m.get("engine_flags") or {}).get("VLLM_BATCH_INVARIANT") != "1":
        return None
    cc = m["gpu"]["compute_capability"]
    if arch == "Qwen2ForCausalLM":
        return {"9.0": "qwen2-eager-batch-invariant-hopper-v1.json", "8.0": "qwen2-eager-batch-invariant-v2.json", "8.9": "qwen2-eager-batch-invariant-v2.json"}.get(cc)
    if arch == "LlamaForCausalLM":
        return {"8.0": "llama-eager-batch-invariant-v2.json", "8.9": "llama-eager-batch-invariant-v2.json"}.get(cc)
    return None


@pytest.mark.skipif(not MANIFESTS, reason="no committed production manifests")
@pytest.mark.parametrize("path", MANIFESTS, ids=[p.parent.parent.name for p in MANIFESTS])
def test_selector_on_committed_manifest(path):
    m = json.loads(path.read_text())
    config = {"architectures": [ARCH[m["model"]["repo"]]]}
    want = expected(m)
    if want is None:
        with pytest.raises(vbinding.BindingError) as ei:
            vbinding.select_semantic_profile(config, m)
        msg = str(ei.value)
        assert "covered by 0 shipped semantic profiles" in msg, msg  # a refusal, not an ambiguity
        return
    got = vbinding.select_semantic_profile(config, m)
    assert got.name == want, (path.parent.parent.name, got.name, want)
    assert json.loads(got.read_text()).get("status") != "frozen"


def test_every_manifest_is_classified_and_no_frozen_profile_is_ever_selected():
    """One table for the report: which profile (or refusal) each committed row gets."""
    table = {}
    for p in MANIFESTS:
        m = json.loads(p.read_text())
        try:
            got = vbinding.select_semantic_profile({"architectures": [ARCH[m["model"]["repo"]]]}, m).name
        except vbinding.BindingError as e:
            got = "REFUSED: " + str(e).split(";")[0]
        table[p.parent.parent.name] = {"gpu": m["gpu"]["name"], "cc": m["gpu"]["compute_capability"], "model": m["model"]["repo"],
                                       "eager": (m.get("compilation") or {}).get("enforce_eager"), "tp": (m.get("parallelism") or {}).get("tensor_parallel_size"),
                                       "selected": got}
    assert not any("-v1.json" in v["selected"] and "hopper" not in v["selected"] for v in table.values()), table
    out = REPO / "fixtures" / "it" / "results" / "selector-manifests.json"
    if out.parent.is_dir():
        out.write_text(json.dumps({"lane": "IT", "manifests": len(table), "rows": table}, indent=1) + "\n")
