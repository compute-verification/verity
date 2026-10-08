"""Lane REAL-HF — held-out test of the importer (`verity_vllm.program.frontend.derive`, frozen at tag `derive-freeze-2` = 2a694d0a)
on REAL `transformers` model classes, not the project's hand-written stand-ins (`derive_examples.HFTinyLM` / `HFGreedy`).

Every case is recorded as one JSONL row (outcome, verbatim refusal, rules fired in order, digest, evaluation vs torch CPU) and
the test itself only fails on a CRASH (unclassified exception), a WRONG program (values / tokens differ from torch CPU beyond the
importer's documented tolerance) or lost sharing. An explicit `Unsupported(...)` is a lawful outcome and is reported, not fixed
(the importer is not modified by this lane; gaps are filed as G-RH-n).

Tiny random-initialised configs (no downloads; HF_HUB_OFFLINE=1): hidden 64, 2 layers, 4 heads, 2 KV heads (GQA models),
vocab 128, intermediate 128, max_position 64; tie_word_embeddings True/False; f32 and bf16; `attn_implementation` eager / sdpa.

Run (pod, `/workspace/venv-realhf`, torch 2.14.0+cu130, transformers 5.16.1):

    cd <tree>/integrations/vllm && export PYTHONPATH=$PWD HF_HUB_OFFLINE=1
    REALHF_OUT=/workspace/realhf/results/realhf.jsonl /workspace/venv-realhf/bin/python -m pytest \
        tests/program/test_derive_realhf.py -q -p no:cacheprovider -s

Environment knobs: REALHF_EVAL=0 (derive only, no evaluation), REALHF_STEPS (greedy decode steps, default 4),
REALHF_ONLY=<substring> (run the cases whose id contains it), REALHF_OUT (JSONL path).
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")

from verity.ml.circuits.codec import program_digest
from verity_vllm.program.frontend.derive import DERIVE_VERSION, DeriveReport, Unsupported, derive_report
from verity_vllm.program.kernels.serve3_reference import materialize
from verity.ml.circuits.types import Array, Record, Value

# ---------------------------------------------------------------------------------------------------------
# dims, environment, output
# ---------------------------------------------------------------------------------------------------------

H, NL, NH, KVH, V, I, MAXPOS, EPS = 64, 2, 4, 2, 128, 128, 64, 1e-5
LP = 8
STEPS = int(os.environ.get("REALHF_STEPS", "4"))
EVAL = os.environ.get("REALHF_EVAL", "1") != "0"
ONLY = os.environ.get("REALHF_ONLY", "")
OUT_PATH = os.environ.get("REALHF_OUT") or "/tmp/realhf/realhf.jsonl"
POS_STATIC = os.environ.get("REALHF_POS") == "static"  # wrappers pass position_ids=arange(T)[None] explicitly (see PROBES)
DT = {"f32": torch.float32, "bf16": torch.bfloat16}


def _git_head() -> str:
    if os.environ.get("REALHF_HEAD"):
        return os.environ["REALHF_HEAD"]
    try:
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


GIT_HEAD = _git_head()


def _record(rec: dict) -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


# ---------------------------------------------------------------------------------------------------------
# the real classes, tiny random configs
# ---------------------------------------------------------------------------------------------------------


def make_config(arch: str, impl: str, tie: bool, **over):
    common = dict(tie_word_embeddings=tie, attn_implementation=impl, use_cache=False)
    if arch == "llama":
        from transformers import LlamaConfig
        return LlamaConfig(vocab_size=V, hidden_size=H, intermediate_size=I, num_hidden_layers=NL, num_attention_heads=NH,
                           num_key_value_heads=KVH, max_position_embeddings=MAXPOS, rms_norm_eps=EPS, **common, **over)
    if arch == "qwen2":
        from transformers import Qwen2Config
        return Qwen2Config(vocab_size=V, hidden_size=H, intermediate_size=I, num_hidden_layers=NL, num_attention_heads=NH,
                           num_key_value_heads=KVH, max_position_embeddings=MAXPOS, rms_norm_eps=EPS, use_sliding_window=False,
                           **common, **over)
    if arch == "mistral":
        from transformers import MistralConfig
        return MistralConfig(vocab_size=V, hidden_size=H, intermediate_size=I, num_hidden_layers=NL, num_attention_heads=NH,
                             num_key_value_heads=KVH, max_position_embeddings=MAXPOS, rms_norm_eps=EPS, **common, **over)
    if arch == "gpt2":
        from transformers import GPT2Config
        return GPT2Config(vocab_size=V, n_positions=MAXPOS, n_embd=H, n_layer=NL, n_head=NH, n_inner=I, layer_norm_epsilon=EPS,
                          resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, **common, **over)
    raise ValueError(arch)


MODEL_CLASS = {"llama": "LlamaForCausalLM", "qwen2": "Qwen2ForCausalLM", "mistral": "MistralForCausalLM", "gpt2": "GPT2LMHeadModel"}


def make_model(arch: str, impl: str, dtype: str, tie: bool, **over) -> nn.Module:
    torch.manual_seed(0)
    cfg = make_config(arch, impl, tie, **over)
    cls = getattr(transformers, MODEL_CLASS[arch])
    m = cls(cfg)
    if hasattr(m, "set_attn_implementation"):
        m.set_attn_implementation(impl)
    assert m.config._attn_implementation == impl, m.config._attn_implementation
    m = m.to(DT[dtype]).eval()
    # random init leaves the norm weights at 1 and the linears at N(0, 0.02): scale the linears up so logits are not all ~0
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim == 2 and "embed" not in n and "wte" not in n and "wpe" not in n:
                p.mul_(8.0)
    tied = m.get_output_embeddings().weight is m.get_input_embeddings().weight
    assert tied == tie, (tied, tie)
    return m


class Logits(nn.Module):
    """`ids[T] int64 -> logits[T, V]`: one HF forward on a fixed prompt (the model's own mask / positions / cache handling)."""

    def __init__(self, model: nn.Module, fkw: Callable[[torch.Tensor], dict] | None = None) -> None:
        super().__init__()
        self.model = model
        self._fkw = fkw

    def forward(self, ids):
        kw = self._fkw(ids) if self._fkw else {}
        if POS_STATIC and "position_ids" not in kw:
            kw["position_ids"] = torch.arange(ids.shape[0]).unsqueeze(0)
        return self.model(input_ids=ids.unsqueeze(0), **kw).logits[0]


class LogitsWithMaskInput(nn.Module):
    """As `Logits` but the 2-D attention mask is a prescribed INPUT value (the way a serving stack passes it)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, ids, attention_mask):
        return self.model(input_ids=ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0)).logits[0]


class LogitsWithPositionInput(nn.Module):
    """As `Logits` but `position_ids` is a prescribed INPUT value."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, ids, position_ids):
        return self.model(input_ids=ids.unsqueeze(0), position_ids=position_ids.unsqueeze(0)).logits[0]


class Greedy(nn.Module):
    """Bounded greedy request analogous to `derive_examples.HFGreedy`: LP int64 prompt tokens, STEPS greedy steps with the prefix
    recomputed each step (no KV cache; the KV dependency is the dataflow). Returns the STEPS+1 selected tokens (i32)."""

    def __init__(self, model: nn.Module, steps: int) -> None:
        super().__init__()
        self.model = model
        self.STEPS = steps

    def forward(self, prompt):
        toks = prompt
        out = []
        for _ in range(self.STEPS + 1):
            kw = {"position_ids": torch.arange(toks.shape[0]).unsqueeze(0)} if POS_STATIC else {}
            logits = self.model(input_ids=toks.unsqueeze(0), use_cache=False, **kw).logits
            nxt = torch.argmax(logits[0, -1:], dim=-1).to(torch.int32)
            out.append(nxt)
            toks = torch.cat([toks, nxt])
        return torch.cat(out)


def _prompt(n: int = LP) -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    return torch.randint(0, V, (n,), generator=g, dtype=torch.int64)


PROMPT = _prompt()


@dataclass
class Case:
    id: str
    arch: str
    impl: str
    dtype: str
    tie: bool
    what: str  # "forward" | "greedy"
    make: Callable[[], tuple[nn.Module, tuple]]  # (wrapper module, example inputs)
    variant: str = ""
    cfg_over: dict = field(default_factory=dict)


def _fwd(arch, impl, dtype, tie, fkw=None, **over):
    def mk():
        m = make_model(arch, impl, dtype, tie, **over)
        return Logits(m, fkw), (PROMPT.clone(),)
    return mk


def _greedy(arch, impl, dtype, tie, **over):
    def mk():
        m = make_model(arch, impl, dtype, tie, **over)
        return Greedy(m, STEPS), (PROMPT.clone(),)
    return mk


CASES: list[Case] = []
for arch in ("llama", "qwen2", "gpt2", "mistral"):
    for impl in ("eager", "sdpa"):
        for dtype in ("f32", "bf16"):
            for tie in (True, False):
                CASES.append(Case(f"{arch}-{impl}-{dtype}-tie{int(tie)}-fwd", arch, impl, dtype, tie, "forward", _fwd(arch, impl, dtype, tie)))
            CASES.append(Case(f"{arch}-{impl}-{dtype}-tie1-greedy", arch, impl, dtype, True, "greedy", _greedy(arch, impl, dtype, True)))

# variants (Llama f32, both attention implementations): the forward-kwargs a serving stack actually passes
for impl in ("eager", "sdpa"):
    CASES += [
        Case(f"llama-{impl}-f32-tie1-fwd-use_cache", "llama", impl, "f32", True, "forward",
             _fwd("llama", impl, "f32", True, fkw=lambda ids: {"use_cache": True}), variant="use_cache=True (DynamicCache built in forward)"),
        Case(f"llama-{impl}-f32-tie1-fwd-mask_static_ones", "llama", impl, "f32", True, "forward",
             _fwd("llama", impl, "f32", True, fkw=lambda ids: {"attention_mask": torch.ones(1, ids.shape[0], dtype=torch.int64)}),
             variant="attention_mask = ones built in forward (static structure)"),
        Case(f"llama-{impl}-f32-tie1-fwd-position_static", "llama", impl, "f32", True, "forward",
             _fwd("llama", impl, "f32", True, fkw=lambda ids: {"position_ids": torch.arange(ids.shape[0]).unsqueeze(0)}),
             variant="position_ids = arange built in forward (static structure)"),
        Case(f"llama-{impl}-f32-tie1-fwd-cache_position_static", "llama", impl, "f32", True, "forward",
             _fwd("llama", impl, "f32", True, fkw=lambda ids: {"cache_position": torch.arange(ids.shape[0])}),
             variant="cache_position = arange built in forward (static structure)"),
    ]
    CASES.append(Case(f"llama-{impl}-f32-tie1-fwd-mask_input", "llama", impl, "f32", True, "forward",
                      (lambda impl=impl: (lambda: (LogitsWithMaskInput(make_model("llama", impl, "f32", True)),
                                                    (PROMPT.clone(), torch.ones(LP, dtype=torch.int64)))))(),
                      variant="attention_mask is a prescribed INPUT value"))
    CASES.append(Case(f"llama-{impl}-f32-tie1-fwd-position_input", "llama", impl, "f32", True, "forward",
                      (lambda impl=impl: (lambda: (LogitsWithPositionInput(make_model("llama", impl, "f32", True)),
                                                    (PROMPT.clone(), torch.arange(LP)))))(),
                      variant="position_ids is a prescribed INPUT value"))
# Mistral with a sliding window smaller than the prompt (the default 4096 never bites at T=8)
for impl in ("eager", "sdpa"):
    CASES.append(Case(f"mistral-{impl}-f32-tie1-fwd-sw4", "mistral", impl, "f32", True, "forward",
                      _fwd("mistral", impl, "f32", True, sliding_window=4), variant="sliding_window=4 (< T=8)", cfg_over={"sliding_window": 4}))
# GPT-2 with the F.gelu(tanh) activation instead of the config default `gelu_new` (the explicit tanh formula)
for impl in ("eager", "sdpa"):
    for dtype in ("f32", "bf16"):
        CASES.append(Case(f"gpt2-{impl}-{dtype}-tie1-fwd-gelu_pytorch_tanh", "gpt2", impl, dtype, True, "forward",
                          _fwd("gpt2", impl, dtype, True, activation_function="gelu_pytorch_tanh"),
                          variant="activation_function=gelu_pytorch_tanh (F.gelu approximate=tanh)", cfg_over={"activation_function": "gelu_pytorch_tanh"}))
        CASES.append(Case(f"gpt2-{impl}-{dtype}-tie1-greedy-gelu_pytorch_tanh", "gpt2", impl, dtype, True, "greedy",
                          _greedy("gpt2", impl, dtype, True, activation_function="gelu_pytorch_tanh"),
                          variant="activation_function=gelu_pytorch_tanh", cfg_over={"activation_function": "gelu_pytorch_tanh"}))

if ONLY:
    CASES = [c for c in CASES if ONLY in c.id]


# ---------------------------------------------------------------------------------------------------------
# PROBES past the first refusal — test-side patches of the *transformers* source (never of the importer), each removing
# exactly one blocker so the NEXT refusal becomes visible. Rows carry `probe` = the patches applied; a result under a probe is
# NOT "the real class derives" and is reported separately from the frozen result (REALHF_PROBE=0, the default).
# ---------------------------------------------------------------------------------------------------------

PROBE = int(os.environ.get("REALHF_PROBE", "0"))


def _probe_patches(level: int) -> list[str]:
    applied: list[str] = []
    if level >= 1:
        # P1: `masking_utils.find_packed_sequence_indices` returns None (what it returns OUTSIDE tracing for one unpadded sequence;
        # under torch.export it cannot evaluate `(mask[:, -1] == 0).all()` and returns the cumsum tensor -> `aten.diff` / `cumsum`).
        import transformers.masking_utils as mu
        mu.find_packed_sequence_indices = lambda position_ids: None
        applied.append("P1:find_packed_sequence_indices->None")
    if level >= 2:
        # P2: `masking_utils.eager_mask` builds the additive mask with a factory zero instead of the literal `torch.tensor(0.0)`
        # (a 0-d tensor literal created in forward is lifted by export and refused as `hidden_state`, XFER-G6 class).
        import transformers.masking_utils as mu
        _sdpa_mask = mu.sdpa_mask

        def eager_mask(batch_size, q_length, kv_length, q_offset=0, kv_offset=0, mask_function=mu.causal_mask_function, attention_mask=None,
                       dtype=torch.float32, allow_is_bidirectional_skip=False, use_vmap=False, device="cpu", **kwargs):
            kwargs.pop("allow_is_causal_skip", None)
            mask = _sdpa_mask(batch_size=batch_size, q_length=q_length, kv_length=kv_length, q_offset=q_offset, kv_offset=kv_offset,
                              mask_function=mask_function, attention_mask=attention_mask, allow_is_causal_skip=False,
                              allow_is_bidirectional_skip=allow_is_bidirectional_skip, use_vmap=use_vmap, device=device, **kwargs)
            if mask is not None:
                mask = torch.where(mask, torch.zeros((), dtype=dtype), torch.finfo(dtype).min)
            return mask

        mu.eager_mask = eager_mask
        mu.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping["eager"] = eager_mask
        applied.append("P2:eager_mask zero literal -> torch.zeros(())")
    if level >= 3:
        # P3: the rotary forward without `.to(dtype=torch.float, device=x.device)` on the inv_freq view (transformers <= 4.5x spelling,
        # which is what `derive_examples.HFRotaryEmbedding` copies); `aten.to.device` has no rule in the importer.
        def rot_forward(self, x, position_ids):
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            position_ids_expanded = position_ids[:, None, :].float()
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

        for modname, clsname in (("llama", "LlamaRotaryEmbedding"), ("qwen2", "Qwen2RotaryEmbedding"), ("mistral", "MistralRotaryEmbedding")):
            mod = __import__(f"transformers.models.{modname}.modeling_{modname}", fromlist=[clsname])
            getattr(mod, clsname).forward = rot_forward
        applied.append("P3:RotaryEmbedding.forward without .to(device) (<=4.5x spelling)")
    if level >= 4:
        # P4: `nn.Dropout` in eval mode is the identity (`aten.dropout(x, p, train=False)` stays in the pre-dispatch graph and the
        # importer classifies `dropout` as `side_input`); GPT-2 keeps `nn.Dropout` modules with p=0.
        _drop_fwd = nn.Dropout.forward

        def drop_forward(self, x):
            return x if not self.training else _drop_fwd(self, x)

        nn.Dropout.forward = drop_forward
        applied.append("P4:nn.Dropout.forward identity in eval mode")
    if level >= 5:
        # P5: `masking_utils.and_masks` seeds its accumulator with `q_idx.new_ones((), dtype=bool)`; `aten.new_ones` on a static index
        # tensor has no rule. Seed with a factory `torch.ones` instead (same value).
        import transformers.masking_utils as mu

        def and_masks(*mask_functions):
            def and_mask(batch_idx, head_idx, q_idx, kv_idx):
                result = torch.ones((), dtype=torch.bool, device=q_idx.device)
                for mask in mask_functions:
                    result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
                return result
            return and_mask

        mu.and_masks = and_masks
        applied.append("P5:and_masks accumulator new_ones -> torch.ones")
    if level >= 6:
        # P6: `masking_utils.create_causal_mask` returns what it returns OUTSIDE tracing for one unpadded prompt: None for sdpa
        # (`_ignore_causal_mask_sdpa` is True at runtime and only disabled under `is_tracing`; `sdpa_attention_forward` then passes
        # is_causal=True, the form the importer's Sdpa rule binds); for eager a 0/finfo.min additive causal mask created outside
        # the trace (a lifted constant, the form EagerAttention's additive branch accepts). Only when no attention_mask / cache is
        # given; sliding-window masks (Mistral) are untouched.
        import transformers.masking_utils as mu
        _ccm = mu.create_causal_mask
        _masks: dict[tuple[int, torch.dtype], torch.Tensor] = {}
        for T in range(1, LP + STEPS + 2):
            for dt in (torch.float32, torch.bfloat16):
                _masks[(T, dt)] = torch.full((1, 1, T, T), torch.finfo(dt).min, dtype=dt).triu(1)

        def create_causal_mask(config, inputs_embeds, attention_mask, past_key_values, position_ids=None, **kw):
            if attention_mask is not None or past_key_values is not None:
                return _ccm(config, inputs_embeds, attention_mask, past_key_values, position_ids=position_ids, **kw)
            if config._attn_implementation == "sdpa":
                return None
            return _masks[(inputs_embeds.shape[1], inputs_embeds.dtype)]

        mu.create_causal_mask = create_causal_mask
        for modname in ("llama", "qwen2", "gpt2", "mistral"):
            mod = __import__(f"transformers.models.{modname}.modeling_{modname}", fromlist=["create_causal_mask"])
            if hasattr(mod, "create_causal_mask"):
                mod.create_causal_mask = create_causal_mask
        applied.append("P6:create_causal_mask -> None (sdpa: is_causal=True) / lifted-constant additive mask (eager)")
    if level >= 7:
        # P7: `integrations.sdpa_attention.sdpa_attention_forward` omits `scale=` when the module's scaling IS the SDPA default
        # D**-0.5 (HF always passes it explicitly; the importer's Sdpa rule binds only the default). Same numerics (D=16: 0.25 exactly).
        import transformers.integrations.sdpa_attention as sa
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        _sdpa_fwd = sa.sdpa_attention_forward

        def sdpa_attention_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
            if scaling is not None and float(scaling) == float(query.shape[-1]) ** -0.5:
                scaling = None
            return _sdpa_fwd(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)

        sa.sdpa_attention_forward = sdpa_attention_forward
        ALL_ATTENTION_FUNCTIONS._global_mapping["sdpa"] = sdpa_attention_forward
        applied.append("P7:sdpa_attention_forward drops scale= when it equals D**-0.5")
    if level >= 8:
        # P8: `integrations.sdpa_attention.use_gqa_in_sdpa` -> False: HF then repeats the KV heads with `repeat_kv` (the <=4.5x path and
        # the stand-in's) instead of SDPA's `enable_gqa=True`, which the importer's Sdpa rule does not bind. Same numerics.
        import transformers.integrations.sdpa_attention as sa
        sa.use_gqa_in_sdpa = lambda *a, **k: False
        applied.append("P8:use_gqa_in_sdpa -> False (repeat_kv instead of enable_gqa)")
    if os.environ.get("REALHF_POS") == "static":
        # Not a transformers patch: the wrappers pass position_ids = arange(T).unsqueeze(0) explicitly (the value HF computes as
        # `arange(T) + past_seen_tokens`; the `+ 0` is what HFRotaryTable's positions scan does not look through).
        applied.append("POS:position_ids=arange(T)[None] passed explicitly")
    return applied


PROBE_APPLIED = _probe_patches(PROBE)


# ---------------------------------------------------------------------------------------------------------
# evaluation of a derived program against the same module on torch CPU (same host)
# ---------------------------------------------------------------------------------------------------------


def _words(t: torch.Tensor, w: int) -> list[int]:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        assert w == 16, w
        return (t.view(torch.int16).numpy().astype(np.int64) & 0xFFFF).ravel().tolist()
    if t.dtype == torch.float32:
        assert w == 32, w
        return (t.view(torch.int32).numpy().astype(np.int64) & 0xFFFFFFFF).ravel().tolist()
    if t.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.bool, torch.uint8):
        return [int(x) & ((1 << w) - 1) for x in t.to(torch.int64).ravel().tolist()]
    raise TypeError(t.dtype)


def _decode(word: int, w: int, like: torch.dtype) -> float | int:
    if like == torch.float32:
        return float(np.array([word], dtype=np.uint32).view(np.float32)[0])
    if like == torch.bfloat16:
        return float(np.array([word << 16], dtype=np.uint32).view(np.float32)[0])
    if word >= 1 << (w - 1):
        word -= 1 << w
    return int(word)


def _leaf_width(T) -> int:
    while isinstance(T, Array):
        T = T.elem
    assert isinstance(T, Value), T
    return T.w


def _derived_tables(rep: DeriveReport, module: nn.Module) -> dict[str, torch.Tensor]:
    """The values a verifier supplies for the rule-declared `derived_table` fields (HF RoPE cos/sin): computed by the source's
    own rotary module from the recorded positions, in the recorded dtype — exactly the recorded derivation."""
    prov = rep.provenance
    fields = [w for w in prov["weights"] if w["kind"] == "derived_table"]
    if not fields:
        return {}
    apps = [a for a in prov["applications"] if a["rule"].startswith("HFRotaryTable")]
    mods = dict(module.named_modules())
    out: dict[str, torch.Tensor] = {}
    cos_fields = [w for w in fields if w["derivation"]["table"] == "cos"]
    for i, cw in enumerate(cos_fields):
        der = cw["derivation"]
        rot = mods[der["source_buffer"].rsplit(".", 1)[0]]
        T = cw["shape"][0]
        positions = apps[i]["preconditions_checked"].get("positions") if i < len(apps) else None
        if positions is None or len(positions) != T:
            positions = list(range(T))
        dt = torch.float32 if der["dtype"] == "f32" else torch.bfloat16
        with torch.no_grad():
            cos, sin = rot(torch.zeros(1, dtype=dt), torch.tensor(positions, dtype=torch.int64).unsqueeze(0))
        out[cw["name"]] = cos[0]
        out[cw["name"].replace(".cos_table", ".sin_table")] = sin[0]
    return out


def evaluate(rep: DeriveReport, module: nn.Module, inputs: tuple, tol_dtype: str) -> dict:
    P = rep.program
    params, _ = P.fn.signature()
    if len(params) != len(inputs) + 1:
        return {"evaluated": False, "why": f"{len(params)} program params vs {len(inputs)} inputs + weights"}
    named = {**dict(module.named_parameters(remove_duplicate=False)), **dict(module.named_buffers(remove_duplicate=False))}
    named.update(_derived_tables(rep, module))
    wrec = params[-1][1]
    assert isinstance(wrec, Record), wrec
    ftypes = dict(wrec.fields)
    words: list[list[int]] = []
    for (pn, pt), x in zip(params[:-1], inputs):
        words.append(_words(x, _leaf_width(pt)))
    wwords: list[int] = []
    for w in rep.provenance["weights"]:
        t = named.get(w["name"])
        if t is None:
            return {"evaluated": False, "why": f"no tensor for weights field {w['name']!r} (kind {w['kind']})"}
        wwords += _words(t, _leaf_width(ftypes[w["name"]]))
    words.append(wwords)
    if sum(len(x) for x in words) != P.input_gates:
        return {"evaluated": False, "why": f"{sum(len(x) for x in words)} words vs {P.input_gates} input gates"}
    try:
        tab = materialize(P, words)
    except Exception as e:  # noqa: BLE001
        return {"evaluated": False, "why": f"materialize: {type(e).__name__}: {str(e)[:300]}"}
    with torch.no_grad():
        want = module(*[x.clone() for x in inputs])
    want_flat = want.reshape(-1)
    outs = P.circuit.root_outputs()
    if len(outs) != want_flat.numel():
        return {"evaluated": True, "match": False, "why": f"{len(outs)} root outputs vs {want_flat.numel()} torch outputs"}
    got = torch.tensor([_decode(tab[g], P.circuit.gate(g).prim.ret.w, want_flat.dtype) for g in outs], dtype=torch.float64)
    want_d = want_flat.to(torch.float64)
    exact = int((got == want_d).sum())
    res = {"evaluated": True, "words": len(outs), "exact_words": exact, "bit_exact": exact == len(outs),
           "gates_evaluated": P.circuit.size}
    if want_flat.is_floating_point():
        d = (got - want_d).abs()
        res["max_abs"] = float(d.max())
        res["max_rel"] = float((d / want_d.abs().clamp_min(1e-6)).max())
        res["want_max_abs"] = float(want_d.abs().max())
        if tol_dtype == "f32":
            # test_derive.py's HF f32 cases: allclose(rtol=1e-5, atol=1e-5); also reported at 1e-4 (the GPT-2 rows)
            res["tolerance"] = "f32: allclose(rtol=1e-5, atol=1e-5) [test_derive.py HF layer]; also rtol=1e-4/atol=1e-5"
            res["match"] = bool(torch.allclose(got, want_d, rtol=1e-5, atol=1e-5))
            res["match_1e-4"] = bool(torch.allclose(got, want_d, rtol=1e-4, atol=1e-5))
        else:
            res["tolerance"] = "bf16: allclose(rtol=1/64, atol=1/32) [test_derive.py HF bf16 layer, ~2 bf16 ulps]"
            res["match"] = bool(torch.allclose(got, want_d, rtol=1 / 64, atol=1 / 32))
        res["match_mode"] = "allclose"
    else:
        res["match"] = bool((got == want_d).all())
        res["match_mode"] = "exact tokens"
        res["got"] = [int(x) for x in got.tolist()]
        res["want"] = [int(x) for x in want_d.tolist()]
    return res


# ---------------------------------------------------------------------------------------------------------
# the run: every case, reported as-is
# ---------------------------------------------------------------------------------------------------------


def _rules_in_order(prov: dict) -> tuple[list[str], dict[str, int]]:
    apps = prov.get("applications") or prov.get("applications_before_refusal") or []
    order: list[str] = []
    counts: dict[str, int] = {}
    for a in apps:
        r = a["rule"]
        if r not in counts:
            order.append(r)
        counts[r] = counts.get(r, 0) + 1
    return order, counts


def _structure(rep: DeriveReport, module: nn.Module, inputs: tuple) -> dict:
    P = rep.program
    seen = {id(p): p for p in list(module.parameters()) + list(module.buffers())}
    distinct = sum(p.numel() for p in seen.values())
    derived = sum(int(np.prod(w["shape"])) for w in rep.provenance["weights"] if w["kind"] == "derived_table")
    n_in = sum(x.numel() for x in inputs)
    tied = [w for w in rep.provenance["weights"] if w["aliases"]]
    return {"input_gates": P.input_gates, "distinct_param_leaves": distinct, "derived_table_leaves": derived, "input_leaves": n_in,
            "one_input_per_distinct_leaf": P.input_gates == distinct + derived + n_in,
            "tied_fields": [(w["name"], w["aliases"]) for w in tied],
            "derived_tables": [(w["name"], w["shape"]) for w in rep.provenance["weights"] if w["kind"] == "derived_table"],
            "root_outputs": len(P.circuit.root_outputs())}


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_realhf_case(case: Case):
    rec: dict[str, Any] = {"case": case.id, "arch": case.arch, "class": f"transformers.{MODEL_CLASS[case.arch]}", "attn_impl": case.impl,
                           "dtype": case.dtype, "tie": case.tie, "what": case.what, "variant": case.variant, "cfg_over": case.cfg_over,
                           "importer_head": GIT_HEAD, "derive_version": DERIVE_VERSION, "torch": torch.__version__,
                           "transformers": transformers.__version__, "mode": "reference", "dims": dict(H=H, NL=NL, NH=NH, KVH=KVH, V=V, I=I, MAXPOS=MAXPOS),
                           "LP": LP, "STEPS": STEPS if case.what == "greedy" else None, "prompt": PROMPT.tolist(),
                           "probe": PROBE, "probe_patches": PROBE_APPLIED}
    try:
        module, inputs = case.make()
    except Exception as e:  # noqa: BLE001
        rec.update(outcome="construct-failed", detail=f"{type(e).__name__}: {str(e)[:400]}")
        _record(rec)
        pytest.fail(f"model construction failed: {rec['detail']}")
    metas = tuple(torch.empty(x.shape, dtype=x.dtype, device="meta") for x in inputs)
    try:
        rep = derive_report(module, metas, mode="reference", name=f"RealHF[{case.id}]", config={**rec["dims"], "dtype": case.dtype, "impl": case.impl, "tie": case.tie})
    except Exception as e:  # noqa: BLE001
        rec.update(outcome="crash", detail=f"{type(e).__name__}: {str(e)[:600]}")
        _record(rec)
        print("\nREALHF", json.dumps(rec, default=str)[:1500])
        pytest.fail(f"CRASH-NOT-REFUSAL {case.id}: {rec['detail']}")
    prov = rep.provenance
    order, counts = _rules_in_order(prov)
    rec["rules_in_order"] = order
    rec["rule_counts"] = counts
    rec["vocabulary"] = prov.get("vocabulary", {}).get("pin") or prov.get("vocabulary", {}).get("name")
    if not rep.ok:
        u = rep.unsupported
        rec["outcome"] = "unsupported"
        rec["reason"] = u.reason
        rec["node"] = u.node
        rec["module_path"] = u.module_path
        rec["hint"] = u.hint
        rec["refusal_verbatim"] = str(u)
        rec["export_ok"] = not (u.hint.startswith("torch.export") or u.reason == "export")
        rec["applications_before_refusal"] = len(prov.get("applications_before_refusal", []))
        _record(rec)
        print("\nREALHF", json.dumps(rec, default=str)[:2000])
        return  # a lawful refusal: reported, never fixed here
    rec["outcome"] = "derived"
    rec["export_ok"] = True
    rec["root"] = rep.root
    rec["digest"] = rep.digest
    rec["flat_digest"] = rep.flat_digest
    rec["gates"] = rep.gates
    rec["input_gates"] = rep.program.input_gates
    rec["nodes"] = len(rep.flat_definition.body.nodes)
    rec["family"] = rep.family["family"] if rep.family else None
    rec["structure"] = _structure(rep, module, inputs)
    if EVAL:
        rec["evaluation"] = evaluate(rep, module, inputs, case.dtype)
    _record(rec)
    print("\nREALHF", json.dumps(rec, default=str)[:2500])
    assert rec["structure"]["one_input_per_distinct_leaf"], rec["structure"]
    if EVAL:
        ev = rec["evaluation"]
        assert ev.get("evaluated"), f"{case.id}: derived program could not be evaluated: {ev.get('why')}"
        assert ev["match"], f"{case.id}: derived program disagrees with torch CPU: {ev}"


def test_realhf_summary_written():
    """Prints the compressed outcome table from the JSONL rows of this run (for the report)."""
    if not os.path.exists(OUT_PATH):
        pytest.skip("no rows")
    rows = [json.loads(l) for l in open(OUT_PATH)]
    rows = [r for r in rows if r.get("importer_head") == GIT_HEAD and r.get("probe", 0) == PROBE]
    print(f"\n{len(rows)} rows @ {GIT_HEAD} probe={PROBE} {PROBE_APPLIED}")
    for r in rows:
        if r.get("outcome") == "derived":
            ev = r.get("evaluation") or {}
            print(f"  {r['case']:52s} DERIVED  nodes={r.get('nodes')} gates={r.get('gates')} digest={str(r.get('digest'))[:12]} "
                  f"match={ev.get('match')} exact={ev.get('exact_words')}/{ev.get('words')} max_abs={ev.get('max_abs')}")
        else:
            print(f"  {r['case']:52s} {str(r.get('outcome')).upper():9s} {r.get('reason')} @ {r.get('module_path')} [{str(r.get('node'))[:60]}] :: {str(r.get('hint'))[:150]}")
