"""Held-out classes for the HF 5.x plumbing closures (compose-vs-extend rule): real `transformers` classes NOBODY used to
build those closures — `Qwen3ForCausalLM` (Llama-like + q/k RMSNorm; primary), `OlmoForCausalLM` and `GemmaForCausalLM`
(secondary) — tiny random configs, sdpa, f32, forward + greedy, ZERO transformers patches. Reuses the harness machinery of
`test_derive_realhf` (wrappers, evaluation against torch CPU, JSONL rows). A refusal is a valid result and is recorded verbatim;
this file never patches anything to make a class pass.

Run (pod, `/workspace/venv-realhf`):
    REALHF_OUT=/workspace/derive2-realhf/heldout.jsonl /workspace/venv-realhf/bin/python -m pytest \
        tests/program/test_derive_hf5_heldout.py -q -p no:cacheprovider -s
Knobs as in test_derive_realhf (REALHF_EVAL, REALHF_STEPS, REALHF_ONLY, REALHF_OUT, REALHF_HEAD)."""
from __future__ import annotations

import json
import os
from typing import Any

import pytest
import torch

transformers = pytest.importorskip("transformers")

from verity_vllm.program.frontend.derive import DERIVE_VERSION, derive_report
from tests.program import test_derive_realhf as RH

H, NL, NH, KVH, V, I, MAXPOS, EPS = RH.H, RH.NL, RH.NH, RH.KVH, RH.V, RH.I, RH.MAXPOS, RH.EPS
HEAD_DIM = H // NH

HELDOUT_CLASS = {"qwen3": "Qwen3ForCausalLM", "olmo": "OlmoForCausalLM", "gemma": "GemmaForCausalLM"}


def make_config(arch: str, impl: str = "sdpa", tie: bool = True):
    common = dict(tie_word_embeddings=tie, attn_implementation=impl, use_cache=False)
    if arch == "qwen3":
        from transformers import Qwen3Config
        return Qwen3Config(vocab_size=V, hidden_size=H, intermediate_size=I, num_hidden_layers=NL, num_attention_heads=NH,
                           num_key_value_heads=KVH, max_position_embeddings=MAXPOS, rms_norm_eps=EPS, head_dim=HEAD_DIM,
                           use_sliding_window=False, **common)
    if arch == "olmo":
        from transformers import OlmoConfig
        return OlmoConfig(vocab_size=V, hidden_size=H, intermediate_size=I, num_hidden_layers=NL, num_attention_heads=NH,
                          num_key_value_heads=KVH, max_position_embeddings=MAXPOS, **common)
    if arch == "gemma":
        from transformers import GemmaConfig
        return GemmaConfig(vocab_size=V, hidden_size=H, intermediate_size=I, num_hidden_layers=NL, num_attention_heads=NH,
                           num_key_value_heads=KVH, max_position_embeddings=MAXPOS, rms_norm_eps=EPS, head_dim=HEAD_DIM, **common)
    raise ValueError(arch)


def make_model(arch: str, impl: str = "sdpa", dtype: str = "f32", tie: bool = True) -> torch.nn.Module:
    torch.manual_seed(0)
    cfg = make_config(arch, impl, tie)
    m = getattr(transformers, HELDOUT_CLASS[arch])(cfg)
    if hasattr(m, "set_attn_implementation"):
        m.set_attn_implementation(impl)
    assert m.config._attn_implementation == impl, m.config._attn_implementation
    m = m.to(RH.DT[dtype]).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim == 2 and "embed" not in n:
                p.mul_(8.0)
    return m


CASES = []
for arch in ("qwen3", "olmo", "gemma"):
    CASES.append((f"{arch}-sdpa-f32-tie1-fwd", arch, "forward"))
    CASES.append((f"{arch}-sdpa-f32-tie1-greedy", arch, "greedy"))
if RH.ONLY:
    CASES = [c for c in CASES if RH.ONLY in c[0]]


@pytest.mark.parametrize("case_id,arch,what", CASES, ids=[c[0] for c in CASES])
def test_heldout_case(case_id: str, arch: str, what: str):
    rec: dict[str, Any] = {"case": case_id, "arch": arch, "class": f"transformers.{HELDOUT_CLASS[arch]}", "attn_impl": "sdpa", "dtype": "f32",
                           "tie": True, "what": what, "variant": "held-out (DERIVE-2)", "importer_head": RH.GIT_HEAD,
                           "derive_version": DERIVE_VERSION, "torch": torch.__version__, "transformers": transformers.__version__,
                           "mode": "reference", "dims": dict(H=H, NL=NL, NH=NH, KVH=KVH, V=V, I=I, MAXPOS=MAXPOS, head_dim=HEAD_DIM),
                           "LP": RH.LP, "STEPS": RH.STEPS if what == "greedy" else None, "prompt": RH.PROMPT.tolist(), "probe": 0, "probe_patches": []}
    model = make_model(arch)
    module = RH.Logits(model) if what == "forward" else RH.Greedy(model, RH.STEPS)
    inputs = (RH.PROMPT.clone(),)
    metas = tuple(torch.empty(x.shape, dtype=x.dtype, device="meta") for x in inputs)
    rep = derive_report(module, metas, mode="reference", name=f"HeldOut[{case_id}]",
                        config={**rec["dims"], "dtype": "f32", "impl": "sdpa", "tie": True})
    prov = rep.provenance
    order, counts = RH._rules_in_order(prov)
    rec["rules_in_order"] = order
    rec["rule_counts"] = counts
    if not rep.ok:
        u = rep.unsupported
        rec.update(outcome="unsupported", reason=u.reason, node=u.node, module_path=u.module_path, hint=u.hint, refusal_verbatim=str(u),
                   applications_before_refusal=len(prov.get("applications_before_refusal", [])))
        RH._record(rec)
        print("\nHELDOUT", json.dumps(rec, default=str)[:2500])
        return  # a lawful refusal: reported verbatim, never fixed here
    rec.update(outcome="derived", root=rep.root, digest=rep.digest, flat_digest=rep.flat_digest, gates=rep.gates,
               input_gates=rep.program.input_gates, nodes=len(rep.flat_definition.body.nodes), family=rep.family["family"] if rep.family else None,
               structure=RH._structure(rep, module, inputs))
    if RH.EVAL:
        rec["evaluation"] = RH.evaluate(rep, module, inputs, "f32")
    RH._record(rec)
    print("\nHELDOUT", json.dumps(rec, default=str)[:2500])
    assert rec["structure"]["one_input_per_distinct_leaf"], rec["structure"]
    if RH.EVAL:
        ev = rec["evaluation"]
        assert ev.get("evaluated"), f"{case_id}: derived program could not be evaluated: {ev.get('why')}"
        assert ev["match"], f"{case_id}: derived program disagrees with torch CPU: {ev}"
