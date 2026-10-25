"""vLLM's layer classes as `check.value_check` groups the modules it taps.

A module's group is the first name of ``TAPPED`` in its MRO, else ``LINEAR_GROUP`` when its MRO holds a name of ``LINEAR``, else its
own class name.  ``REEXEC`` are the groups whose forward is a pure function of the tapped inputs, so the check runs them again;
``SAMPLER`` is the runner's sampler's group.
"""
from __future__ import annotations

TAPPED = ("RMSNorm", "GemmaRMSNorm", "SiluAndMul", "GeluAndMul", "RotaryEmbedding", "Attention", "VocabParallelEmbedding",
          "LogitsProcessor")
LINEAR = frozenset({"LinearBase", "ColumnParallelLinear", "RowParallelLinear", "ReplicatedLinear", "QKVParallelLinear",
                    "MergedColumnParallelLinear"})
LINEAR_GROUP = "Linear"
REEXEC = (LINEAR_GROUP, "RMSNorm", "GemmaRMSNorm", "SiluAndMul", "GeluAndMul", "RotaryEmbedding")
SAMPLER = "Sampler"
