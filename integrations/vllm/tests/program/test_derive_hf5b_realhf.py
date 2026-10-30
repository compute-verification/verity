"""Real `transformers` 5.16.1 classes with the embedding `padding_idx` and the bf16 `inv_freq` buffer supported, ZERO transformers
patches, reference mode, sdpa; the harness machinery of `test_derive_realhf` (wrappers, evaluation against torch CPU, JSONL rows) and
the held-out model makers of `test_derive_hf5_heldout` reused unchanged.

Groups (`GROUP` in the JSONL row):
  * `grh14`   — `OlmoForCausalLM` (pad 1), `GemmaForCausalLM` (pad 0), f32: the `padding_idx` targets;
  * `grh7`    — `LlamaForCausalLM`, `Qwen2ForCausalLM`, `Qwen3ForCausalLM` in bf16: the bf16 `inv_freq` targets;
  * `reconf`  — Llama / Qwen2 / GPT-2 (`gelu_pytorch_tanh`) / Qwen3 f32 greedy: the digests `test_derive_realhf` and
                `test_derive_hf5_heldout` report must not move;
  * `heldout` — `Phi3ForCausalLM` (primary), `GPTNeoXForCausalLM`, `StableLmForCausalLM` (secondary), f32: classes nobody used to
                build the `padding_idx` / `inv_freq` support, run once as-is; a refusal is a valid result, never patched here;
    * `open`    — the open gaps (below): must still refuse with their (stable part of the) strings;
    * `heldout4` — `Starcoder2ForCausalLM` (primary), `GraniteForCausalLM`, `CohereForCausalLM` (secondary), f32: classes nobody
                used to build any support here (the `grh14` rows are its targets), run once as-is; refusals are never patched here.

Run (pod, `/workspace/venv-realhf`):
    REALHF_OUT=/workspace/derive3-realhf/hf5b.jsonl /workspace/venv-realhf/bin/python -m pytest \
        tests/program/test_derive_hf5b_realhf.py -q -p no:cacheprovider -s
Knobs as in test_derive_realhf (REALHF_EVAL, REALHF_STEPS, REALHF_ONLY, REALHF_OUT, REALHF_HEAD); REALHF_GROUP=<group> selects a group.
bf16 equality criterion (`test_derive_realhf`'s `evaluate`, unchanged): logits — `allclose(rtol=1/64, atol=1/32)` is the pass/fail `match`, and the
row ALSO carries `exact_words` (bit-exact count) and `max_abs`; greedy tokens — bit-equal or not (`match_mode: exact tokens`)."""
from __future__ import annotations

import json
import os
from typing import Any

import pytest
import torch

transformers = pytest.importorskip("transformers")

from verity_vllm.program.frontend.derive import DERIVE_VERSION, derive_report
from tests.program import test_derive_hf5_heldout as HO
from tests.program import test_derive_realhf as RH

H, NL, NH, KVH, V, I, MAXPOS, EPS = RH.H, RH.NL, RH.NH, RH.KVH, RH.V, RH.I, RH.MAXPOS, RH.EPS
HEAD_DIM = H // NH
GROUP = os.environ.get("REALHF_GROUP", "")

CLASS = {**RH.MODEL_CLASS, **HO.HELDOUT_CLASS, "phi3": "Phi3ForCausalLM", "gpt_neox": "GPTNeoXForCausalLM", "stablelm": "StableLmForCausalLM",
         # `heldout4`: LayerNorm + bias + gelu_pytorch_tanh; residual / logit multipliers; parallel attn+MLP
         "starcoder2": "Starcoder2ForCausalLM", "granite": "GraniteForCausalLM", "cohere": "CohereForCausalLM"}


def make_config(arch: str, impl: str = "sdpa", tie: bool = True, **over):
    """Tiny random configs for this file's held-out classes (the others come from RH / HO). `pad_token_id` is set explicitly where the
    class default would fall outside the tiny vocabulary (Phi3: 32000) — a config value, not a patch."""
    common = dict(tie_word_embeddings=tie, attn_implementation=impl, use_cache=False)
    if arch == "phi3":
        from transformers import Phi3Config
        return Phi3Config(vocab_size=V, hidden_size=H, intermediate_size=I, num_hidden_layers=NL, num_attention_heads=NH,
                          num_key_value_heads=KVH, max_position_embeddings=MAXPOS, original_max_position_embeddings=MAXPOS,
                          rms_norm_eps=EPS, pad_token_id=0, **common, **over)
    if arch == "gpt_neox":
        from transformers import GPTNeoXConfig
        return GPTNeoXConfig(vocab_size=V, hidden_size=H, intermediate_size=I, num_hidden_layers=NL, num_attention_heads=NH,
                             max_position_embeddings=MAXPOS, layer_norm_eps=EPS, **common, **over)
    if arch == "stablelm":
        from transformers import StableLmConfig
        return StableLmConfig(vocab_size=V, hidden_size=H, intermediate_size=I, num_hidden_layers=NL, num_attention_heads=NH,
                              num_key_value_heads=KVH, max_position_embeddings=MAXPOS, layer_norm_eps=EPS, **common, **over)
    if arch == "starcoder2":
        from transformers import Starcoder2Config
        return Starcoder2Config(vocab_size=V, hidden_size=H, intermediate_size=I, num_hidden_layers=NL, num_attention_heads=NH,
                                num_key_value_heads=KVH, max_position_embeddings=MAXPOS, norm_epsilon=EPS, **common, **over)
    if arch == "granite":
        from transformers import GraniteConfig
        return GraniteConfig(vocab_size=V, hidden_size=H, intermediate_size=I, num_hidden_layers=NL, num_attention_heads=NH,
                             num_key_value_heads=KVH, max_position_embeddings=MAXPOS, rms_norm_eps=EPS, pad_token_id=0, **common, **over)
    if arch == "cohere":
        from transformers import CohereConfig
        return CohereConfig(vocab_size=V, hidden_size=H, intermediate_size=I, num_hidden_layers=NL, num_attention_heads=NH,
                            num_key_value_heads=KVH, max_position_embeddings=MAXPOS, layer_norm_eps=EPS, **common, **over)
    raise ValueError(arch)


def make_model(arch: str, dtype: str, impl: str = "sdpa", tie: bool = True, **over) -> torch.nn.Module:
    if arch in RH.MODEL_CLASS:
        return RH.make_model(arch, impl, dtype, tie, **over)
    if arch in HO.HELDOUT_CLASS:
        assert not over
        return HO.make_model(arch, impl, dtype, tie)
    torch.manual_seed(0)
    cfg = make_config(arch, impl, tie, **over)
    m = getattr(transformers, CLASS[arch])(cfg)
    if hasattr(m, "set_attn_implementation"):
        m.set_attn_implementation(impl)
    assert m.config._attn_implementation == impl, m.config._attn_implementation
    m = m.to(RH.DT[dtype]).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim == 2 and "embed" not in n:
                p.mul_(8.0)
    return m


# (case id, group, arch, impl, dtype, what, config overrides, expected refusal fragments (open gaps only))
CASES: list[tuple] = []
for arch in ("olmo", "gemma"):
    for what in ("forward", "greedy"):
        CASES.append((f"{arch}-sdpa-f32-tie1-{'fwd' if what == 'forward' else 'greedy'}", "grh14", arch, "sdpa", "f32", what, {}, ()))
for arch in ("llama", "qwen2", "qwen3"):
    for what in ("forward", "greedy"):
        CASES.append((f"{arch}-sdpa-bf16-tie1-{'fwd' if what == 'forward' else 'greedy'}", "grh7", arch, "sdpa", "bf16", what, {}, ()))
CASES += [
    ("llama-sdpa-f32-tie1-greedy", "reconf", "llama", "sdpa", "f32", "greedy", {}, ()),
    ("qwen2-sdpa-f32-tie1-greedy", "reconf", "qwen2", "sdpa", "f32", "greedy", {}, ()),
    ("gpt2-sdpa-f32-tie1-greedy-gelu_pytorch_tanh", "reconf", "gpt2", "sdpa", "f32", "greedy", {"activation_function": "gelu_pytorch_tanh"}, ()),
    ("qwen3-sdpa-f32-tie1-greedy", "reconf", "qwen3", "sdpa", "f32", "greedy", {}, ()),
]
for arch in ("phi3", "gpt_neox", "stablelm"):
    for what in ("forward", "greedy"):
        CASES.append((f"{arch}-sdpa-f32-tie1-{'fwd' if what == 'forward' else 'greedy'}", "heldout", arch, "sdpa", "f32", what, {}, ()))
CASES.append(("phi3-sdpa-f32-tie1-fwd-partial_rotary_0.5", "heldout", "phi3", "sdpa", "f32", "forward", {"partial_rotary_factor": 0.5}, ()))
# `heldout4` — Starcoder2 (primary), Granite, Cohere (secondary); run once as-is, refusals never patched here
for arch in ("starcoder2", "granite", "cohere"):
    for what in ("forward", "greedy"):
        CASES.append((f"{arch}-sdpa-f32-tie1-{'fwd' if what == 'forward' else 'greedy'}", "heldout4", arch, "sdpa", "f32", what, {}, ()))
CASES += [
    ("llama-eager-f32-tie1-fwd", "open", "llama", "eager", "f32", "forward", {}, ("hidden_state", "lift_fresh_copy")),
    ("gpt2-eager-f32-tie1-fwd-gelu_pytorch_tanh", "open", "gpt2", "eager", "f32", "forward", {"activation_function": "gelu_pytorch_tanh"},
     ("hidden_state", "lift_fresh_copy")),                                                                                        # the same refusal as the eager llama case
    ("gpt2-sdpa-f32-tie1-fwd", "open", "gpt2", "sdpa", "f32", "forward", {}, ("numerics_unregistered", "a Python scalar operand needs a constant kind")),
    ("mistral-sdpa-f32-tie1-fwd-sw4", "open", "mistral", "sdpa", "f32", "forward", {"sliding_window": 4},
     ("decomposition", "the static attn_mask is not the causal pattern")),
]
if GROUP:
    CASES = [c for c in CASES if c[1] == GROUP]
if RH.ONLY:
    CASES = [c for c in CASES if RH.ONLY in c[0]]

# the digests the earlier harnesses report: the re-confirmation rows must reproduce them exactly
EXPECT_DIGEST_PREFIX = {"llama-sdpa-f32-tie1-greedy": "538319a3770a", "qwen2-sdpa-f32-tie1-greedy": "e9d32625a99b",
                        "gpt2-sdpa-f32-tie1-greedy-gelu_pytorch_tanh": "03925eb18841", "qwen3-sdpa-f32-tie1-greedy": "46ed593f82a3"}


def _derive_identity(case_id: str, group: str, dims: dict, dtype: str, impl: str, over: dict) -> tuple[str, dict]:
    """The `name` / `config` handed to `derive_report`. Both are part of the program's identity: the root specialization id is
    `Derived[<name>]_v1{SRC={class, config_sha256, inputs, mode, profile}}` and `program_digest` hashes the root id, so the SAME gates
    under another name or config are another digest. The re-confirmation rows therefore use exactly the identity of the harness
    that reported the digest — `RealHF[<id>]` with `{dims (no head_dim), dtype, impl, tie}` for Llama / Qwen2 / GPT-2
    (`test_derive_realhf.py`), `HeldOut[<id>]` with `{dims + head_dim, …}` for Qwen3 (`test_derive_hf5_heldout.py`).
    Every other case is `HF5b[<id>]` with the full dims and the config overrides (its digests are new, not comparable to theirs)."""
    if group == "reconf" and case_id.startswith("qwen3"):
        return f"HeldOut[{case_id}]", {**dims, "dtype": dtype, "impl": impl, "tie": True}
    if group == "reconf":
        return f"RealHF[{case_id}]", {**{k: v for k, v in dims.items() if k != "head_dim"}, "dtype": dtype, "impl": impl, "tie": True}
    return f"HF5b[{case_id}]", {**dims, "dtype": dtype, "impl": impl, "tie": True, **over}


@pytest.mark.parametrize("case_id,group,arch,impl,dtype,what,over,expect_refusal", CASES, ids=[c[0] for c in CASES])
def test_hf5b_case(case_id: str, group: str, arch: str, impl: str, dtype: str, what: str, over: dict, expect_refusal: tuple):
    rec: dict[str, Any] = {"case": case_id, "group": group, "arch": arch, "class": f"transformers.{CLASS[arch]}", "attn_impl": impl, "dtype": dtype,
                           "tie": True, "what": what, "variant": f"{'DERIVE-4' if group in ('heldout4', 'grh14') else 'DERIVE-3'} {group}",
                           "cfg_over": over, "importer_head": RH.GIT_HEAD,
                           "derive_version": DERIVE_VERSION, "torch": torch.__version__, "transformers": transformers.__version__,
                           "mode": "reference", "dims": dict(H=H, NL=NL, NH=NH, KVH=KVH, V=V, I=I, MAXPOS=MAXPOS, head_dim=HEAD_DIM),
                           "LP": RH.LP, "STEPS": RH.STEPS if what == "greedy" else None, "prompt": RH.PROMPT.tolist(), "probe": 0, "probe_patches": []}
    model = make_model(arch, dtype, impl, True, **over)
    module = RH.Logits(model) if what == "forward" else RH.Greedy(model, RH.STEPS)
    inputs = (RH.PROMPT.clone(),)
    metas = tuple(torch.empty(x.shape, dtype=x.dtype, device="meta") for x in inputs)
    name, config = _derive_identity(case_id, group, rec["dims"], dtype, impl, over)
    rec["derive_name"] = name
    rep = derive_report(module, metas, mode="reference", name=name, config=config)
    prov = rep.provenance
    order, counts = RH._rules_in_order(prov)
    rec["rules_in_order"] = order
    rec["rule_counts"] = counts
    if not rep.ok:
        u = rep.unsupported
        rec.update(outcome="unsupported", reason=u.reason, node=u.node, module_path=u.module_path, hint=u.hint, refusal_verbatim=str(u),
                   applications_before_refusal=len(prov.get("applications_before_refusal", [])))
        RH._record(rec)
        print("\nHF5B", json.dumps(rec, default=str)[:3000])
        if expect_refusal:  # an open gap: the refusal must be the known one
            reason, needle = expect_refusal
            assert u.reason == reason and (needle in str(u.node) or needle in u.hint), str(u)
        return  # a lawful refusal: reported verbatim, never fixed here
    rec.update(outcome="derived", root=rep.root, digest=rep.digest, flat_digest=rep.flat_digest, gates=rep.gates,
               input_gates=rep.program.input_gates, nodes=len(rep.flat_definition.body.nodes), family=rep.family["family"] if rep.family else None,
               structure=RH._structure(rep, module, inputs))
    rot = [a for a in prov["applications"] if a["rule"].startswith("HFRotaryTable")]
    emb = [a for a in prov["applications"] if a["rule"].startswith("Embedding[reference]")]
    rec["rotary_tables"] = [{k: a["preconditions_checked"].get(k) for k in ("T", "D", "dtype", "source_dtype", "source_cast", "fired_at")} for a in rot[:1]]
    rec["embedding_padding_idx"] = sorted({str(a["preconditions_checked"].get("padding_idx")) for a in emb})
    if RH.EVAL:
        rec["evaluation"] = RH.evaluate(rep, module, inputs, dtype)
    RH._record(rec)
    print("\nHF5B", json.dumps(rec, default=str)[:3000])
    assert not expect_refusal, f"{case_id}: an open gap DERIVED — the report must say so (digest {rep.digest[:12]})"
    assert rec["structure"]["one_input_per_distinct_leaf"], rec["structure"]
    if case_id in EXPECT_DIGEST_PREFIX:
        assert rep.digest.startswith(EXPECT_DIGEST_PREFIX[case_id]), (case_id, rep.digest, EXPECT_DIGEST_PREFIX[case_id])
    if RH.EVAL:
        ev = rec["evaluation"]
        assert ev.get("evaluated"), f"{case_id}: derived program could not be evaluated: {ev.get('why')}"
        assert ev["match"], f"{case_id}: derived program disagrees with torch CPU: {ev}"
