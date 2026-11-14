"""vLLM's layer classes as `check.value_check` groups the modules it taps, and as the native committer names them.

A module's group is the first name of ``TAPPED`` in its MRO, else ``LINEAR_GROUP`` when its MRO holds a name of ``LINEAR``, else its
own class name.  ``REEXEC`` are the groups whose forward is a pure function of the tapped inputs, so the check runs them again;
``SAMPLER`` is the runner's sampler's group.
"""
from __future__ import annotations

__all__ = ["ATTENTION", "COLUMN_LINEAR", "EMBEDDING", "GATED_ACTS", "GELU_AND_MUL", "GEMMA_RMS_NORM", "LINEAR", "LINEAR_BASE",
           "LINEAR_GROUP", "LM_HEAD", "LOGITS", "NORMS", "REEXEC", "RMS_NORM", "ROTARY", "ROW_LINEAR", "SAMPLER",
           "SHARDED_ENDINGS", "SILU_AND_MUL", "TAPPED"]

TAPPED = ("RMSNorm", "GemmaRMSNorm", "SiluAndMul", "GeluAndMul", "RotaryEmbedding", "Attention", "VocabParallelEmbedding",
          "LogitsProcessor")
LINEAR = frozenset({"LinearBase", "ColumnParallelLinear", "RowParallelLinear", "ReplicatedLinear", "QKVParallelLinear",
                    "MergedColumnParallelLinear"})
LINEAR_GROUP = "Linear"
REEXEC = (LINEAR_GROUP, "RMSNorm", "GemmaRMSNorm", "SiluAndMul", "GeluAndMul", "RotaryEmbedding")
SAMPLER = "Sampler"

#: The classes `commit.committer.native_host` names one by one in its step bound and its coverage declaration.
RMS_NORM, GEMMA_RMS_NORM, LAYER_NORM = "RMSNorm", "GemmaRMSNorm", "LayerNorm"
SILU_AND_MUL, GELU_AND_MUL = "SiluAndMul", "GeluAndMul"
ROTARY, ATTENTION, EMBEDDING, LOGITS = "RotaryEmbedding", "Attention", "VocabParallelEmbedding", "LogitsProcessor"
LM_HEAD = "ParallelLMHead"
LINEAR_BASE, COLUMN_LINEAR, ROW_LINEAR = "LinearBase", "ColumnParallelLinear", "RowParallelLinear"
#: The norms and the gated activations, each a family the step bound gives one width.
NORMS = frozenset({RMS_NORM, GEMMA_RMS_NORM, LAYER_NORM})
GATED_ACTS = frozenset({SILU_AND_MUL, GELU_AND_MUL, "GeluAndMulSparse", "MulAndSilu", "SiluAndMulWithClamp"})
#: The endings of a class name by which a tensor-parallel rank picks the modules it lists with their sharding attributes:
#: vLLM's parallel linears, its embeddings and its LM head.  ``LOGITS`` is picked by its whole name, beside these.
SHARDED_ENDINGS = ("ParallelLinear", "Embedding", "LMHead")
