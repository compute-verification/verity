"""P8: model and target facts live only in ``verity_vllm/program/quarantine/`` and the engine profile: the model facts in
``engine/profiles/``, and the κ profile that pins them for a (vLLM, torch, Triton, GPU) tuple in ``observe/fold/profiles/``.

The rule's literal scan: a string literal in code (docstrings and a module's ``__models__`` marker excepted) naming a model
(``model``), a vLLM layer class (``layer-class``), a GPU (``gpu``) or an ``sm_8x``/``sm_9x``/``sm_10x``/``sm_12x``
architecture (``sm-arch``), outside the two homes.  Keyed by the normalized token, counted per function, in ``allowlists/p08_facts.json``.

The scan lives here rather than in ``tests/properties/test_quarantine_lint.py``, which tests the quarantine lint itself.
"""
from __future__ import annotations

import ast
import re
from functools import lru_cache

from . import _ratchet as R

NAME = "p08_facts"
RULE = "P8 (model and target facts live in their two homes)"
HOMES = ("verity_vllm/program/quarantine/", "verity_vllm/engine/profiles/", "verity_vllm/observe/fold/profiles/")
MODEL = re.compile(r"(?i)(?<![a-z0-9])(llama|qwen|gemma|olmoe|olmo|mixtral|mistral|smollm|deepseek|phi-?[0-9]|pythia|"
                   r"gpt-?neox|gpt2|granite|falcon|starcoder|opt-[0-9])")
LAYER_CLASS = re.compile(r"\b((?:QKV|Row|Column|MergedColumn|Replicated|ReplicatedLinear|VocabParallel|Parallel)"
                         r"(?:Parallel)?(?:Linear|Embedding|LMHead)|FusedMoE|MoERunner|SharedFusedMoE|RMSNorm|GemmaRMSNorm|"
                         r"LayerNorm|SiluAndMul|GeluAndMul|RotaryEmbedding|LogitsProcessor|Sampler|"
                         r"[A-Z][A-Za-z0-9]*(?:DecoderLayer|ForCausalLM|Attention|MLP|MoE))\b")
GPU = re.compile(r"(?i)(?<![a-z0-9])(l40s|l40|l4|a100|a10g|a10|a40|a6000|h100|h200|h800|h20|gh200|b200|gb200|b100|rtx ?[2-5]0[6-9]0|"
                 r"4090|3090|5090|t4|v100)(?![a-z0-9])")
SM_ARCH = re.compile(r"(?i)(?<![a-z0-9])(sm_?(?:8[0-9]|9[0-9]|10[0-9]|12[0-9])a?)(?![0-9])")


def tokens(s: str) -> list[tuple[str, str]]:
    out = [("model", m.group(1).lower()) for m in MODEL.finditer(s)]
    out += [("layer-class", m.group(1)) for m in LAYER_CLASS.finditer(s)]
    out += [("gpu", m.group(1).lower().replace(" ", "")) for m in GPU.finditer(s)]
    out += [("sm-arch", m.group(1).lower().replace("sm", "sm_").replace("__", "_")) for m in SM_ARCH.finditer(s)]
    return out


def file_violations(src: str, rel: str) -> list[R.Violation]:
    if rel.startswith(HOMES):
        return []
    tree = R.parse_src(src)
    docs = R.docstring_ids(tree) | R.marker_ids(tree)
    out: list[R.Violation] = []
    for node, symbol in R.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            for kind, tok in sorted(set(tokens(node.value))):
                out.append(R.Violation(rel, kind, symbol, tok, node.lineno))
    return out


@lru_cache(maxsize=None)
def scan() -> tuple[R.Violation, ...]:
    out: list[R.Violation] = []
    for p in R.library_files():
        out += file_violations(R.source(p), R.rel(p))
    return tuple(out)


def test_no_new_violations() -> None:
    R.assert_no_new(NAME, scan(), RULE,
                    "Read model and target facts from verity_vllm/program/quarantine/ or the engine profile, "
                    "not from a literal.")


def test_allowlist_has_no_stale_entries() -> None:
    R.assert_no_stale(NAME, scan(), RULE)


def test_detectors_on_examples() -> None:
    bad = (
        "'''Llama on L40S, sm_89: docstrings are prose.'''\n"
        "MOE_SITE_CLASSES = ('MoERunner', 'FusedMoE')\n"
        "def pick(model):\n"
        "    if 'qwen' in model and DEVICE == 'H100':\n"
        "        return 'sm_90a'\n"
        "    return 'meta-llama/Llama-3.2-1B'\n"
    )
    got = sorted((v.kind, v.detail) for v in file_violations(bad, "verity_vllm/tp/x.py"))
    assert got == [("gpu", "h100"), ("layer-class", "FusedMoE"), ("layer-class", "MoERunner"), ("model", "llama"),
                   ("model", "qwen"), ("sm-arch", "sm_90a")], got
    assert file_violations(bad, "verity_vllm/engine/profiles/x.py") == []
    assert file_violations(bad, "verity_vllm/observe/fold/profiles/x.py") == []
    assert file_violations("x = ('bf16', 'float32', 'tp1', 'layers.0.mlp')\n", "verity_vllm/tp/x.py") == []
    assert file_violations("__models__ = ('deepseek-ai/DeepSeek-V4-Pro',)\n", "verity_vllm/tp/x.py") == []
