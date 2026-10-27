"""verity_vllm.check.match.compiler_form: the Match reads a fold in the registered compiler's form when its components are in it.
TINY's request of 8 prompt rows and 3 tokens, built unpruned, is what the runtime runs and the fold records: every prompt row
through the last layer, with the v1 Definitions.  Put in the compiler's form it is the Build's request Program instance for
instance and operand for operand, in both operand forms; and the Build's Program is its own fixed point."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from verity.ml.circuits.codec import encode_program
from verity_vllm.check.match import compiler_form as CF
from verity_vllm.check.match.instance_sequence import Prog
from verity_vllm.check.match.program_compare import compare_steps
from verity_vllm.program.predict import build as PB
from verity_vllm.program.predict.phi import Phi

TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4,
        "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5, "tie_word_embeddings": True,
        "max_position_embeddings": 64, "hidden_act": "silu"}
PHI = Phi(model={"repo": "org/m", "revision": "r", "hf_config": TINY}, target={"compute_capability": [12, 0], "num_sms": 188})
DEAD = {"Attention_v5": 7, "Gemm_v2": 21, "RMSNormFusedCuda_v2": 14, "RoPE_v1": 7, "SiluMul_v2": 7}


@pytest.fixture(scope="module")
def descriptors():
    """The Build's request Program (the registered compiler's) and the same request unpruned, as descriptors."""
    pruned = PB.build_program(PHI, "request", LP=8, T=3)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(PB, "dead_free_form", lambda b: (b, list(range(len(b.nodes)))))
        full = PB.build_program(PHI, "request", LP=8, T=3)
    return encode_program(pruned.program), encode_program(full.program)


@pytest.mark.parametrize("compact", [False, True])
def test_the_unpruned_request_in_the_compilers_form_is_the_builds_request_program(descriptors, compact):
    """Before: the fold has the last layer's 56 dead units, the v1 specs and the full qkv on rows 0-6, so the Match's canonical
    sequences differ.  After: the same Program set as the compiler's (spec, operands and output of every instance), the dead units
    dropped by family, every RMSNormTriton, Attention and TokenSelect swapped, and the 7 qkv linears narrowed to their K/V columns."""
    built, ran = descriptors
    A = Prog("built", built, {}, tok_node=0, compact=compact)
    B = Prog("ran", ran, {}, tok_node=0, compact=compact)
    assert not compare_steps(A, B)["canonical_sha256"]["equal"]
    ran_spec = {r["k"]: r["spec"] for r in B.inst}
    facts = B.to_compiler_form()
    assert facts == {"sampling_events": 4, "dead": DEAD,
                     "swapped": {"Attention_v5 -> Attention_v18": 15, "RMSNormTriton_v1 -> RMSNormTriton_v5": 11,
                                 "TokenSelect_v1 -> TokenSelect_v4": 4},
                     "narrowed": {"Gemm_v2": 7}}
    assert len(B.inst) == len(A.inst) == 194
    assert Counter(r["spec"] for r in B.inst) == Counter(r["spec"] for r in A.inst)
    assert [(r["spec"], r["out"]) for r in B.inst] == [(r["spec"], r["out"]) for r in A.inst]
    kv = [r for r in B.inst if ran_spec[r["k"]].startswith("Gemm_v2{K=64,N=128,") and r["spec"] != ran_spec[r["k"]]]
    assert len(kv) == 7 and {(r["spec"].split(",DOT")[0], r["out"]) for r in kv} == {("Gemm_v2{K=64,N=64", "Array<64,Value<16>>")}
    res = compare_steps(A, B)
    assert res["canonical_sha256"]["equal"] and res["totals"]["operands_different"] == 0
    assert res["totals"]["steps_spec_equal"] == len(A.steps) == len(B.steps) == 4


@pytest.mark.parametrize("compact", [False, True])
def test_the_builds_request_program_is_its_own_compilers_form(descriptors, compact):
    """The compiler's form is a fixed point: nothing dead, nothing left to swap, no linear read in part, every instance unchanged."""
    A = Prog("built", descriptors[0], {}, tok_node=0, compact=compact)
    inst = list(A.inst)
    assert A.to_compiler_form() == {"sampling_events": 4, "dead": {}, "swapped": {}, "narrowed": {}}
    assert all(x is y for x, y in zip(A.inst, inst, strict=True))


def _component(tmp_path, name: str, dead_gates: dict | None):
    d = tmp_path / name
    d.mkdir()
    if dead_gates is not None:
        (d / "derive-report.json").write_text(json.dumps({"dead_gates": dead_gates}))
    return str(d)


def test_a_fold_is_put_in_the_form_only_when_every_component_counts_no_dead_gate_exactly(tmp_path, descriptors):
    """A component is in the compiler's form when its derive report counts no dead computed gate exactly; one that counts some,
    counts approximately or has no report keeps the fold as it ran, by name."""
    good = _component(tmp_path, "build_request_LP8_T3", {"dead_computed_gates": 0, "approximate": False})
    dead = _component(tmp_path, "build_step", {"dead_computed_gates": 7138, "approximate": False})
    approx = _component(tmp_path, "build_request_approx", {"dead_computed_gates": 0, "approximate": True})
    bare = _component(tmp_path, "build_request_bare", None)
    assert CF.components_in_form([good, good]) == {"in_form": True, "components": 1, "not_in_form": []}
    assert CF.components_in_form([good, dead, approx, bare]) == {
        "in_form": False, "components": 4, "not_in_form": ["build_step", "build_request_approx", "build_request_bare"]}
    assert not CF.components_in_form([])["in_form"]
    B = Prog("ran", descriptors[1], {}, tok_node=0)
    n = len(B.inst)
    kept = CF.fold_in_form(B, [good, dead])
    assert kept["in_form"] is False and "dead" not in kept and B.fold_form is kept and len(B.inst) == n
    put = CF.fold_in_form(B, [good])
    assert put["dead"] == DEAD and B.fold_form is put and len(B.inst) == n - sum(DEAD.values())


def test_the_linears_n_is_its_top_level_static():
    """A function-valued static names its Definition in a JSON string; an `N=` inside it is not the linear's."""
    s = '{K=64,N=128,DOT={"fn":"Dot_v1{N=3}"}}'
    assert CF._top_n(s) == (8, 11) and s[8:11] == "128"
    t = '{DOT={"fn":"Dot_v1{N=3,M=[1,2]}"},K=64,N=5}'
    assert t[slice(*CF._top_n(t))] == "5"
    assert CF._top_n('{K=64,DOT={"fn":"Dot_v1{N=3}"}}') is None
