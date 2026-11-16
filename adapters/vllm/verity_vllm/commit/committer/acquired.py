"""What the native committers acquire: the module classes whose forward outputs (and input ids) they take, matched on the class
names of a module's MRO; the chunk an acquired tensor's leaves are cut at; and one acquired tensor's record in its step
(`TensorMeta`).  It imports no torch, so a reader of the tables or of a step's layout needs no committer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from verity_vllm import settings

GPU_CHUNK_BYTES = 256

# the construction-time selection: TP1 Commits replace it with the acquisition plan's (apply_plan_to_committer); TP rank committers
# (pipeline/tp/rank_worker.py) have no plan yet and keep it
# module classes whose forward OUTPUT is an Instance output (census §3 `HBM-materialized (native)`); matched on the MRO class names
ACQUIRE_CLASSES = {
    "ColumnParallelLinear", "RowParallelLinear", "QKVParallelLinear", "MergedColumnParallelLinear", "ReplicatedLinear", "LinearBase",
    "RMSNorm", "GemmaRMSNorm", "LayerNorm",
    "SiluAndMul", "GeluAndMul", "GeluAndMulSparse", "MulAndSilu", "SiluAndMulWithClamp",
    "RotaryEmbedding",
    "Attention",
    "VocabParallelEmbedding",
    "LogitsProcessor",
    "Sampler",
}
# classes whose INPUT ids are a root input of the Program (prompt ids / fed-back sampled ids)
ACQUIRE_INPUT_CLASSES = {"VocabParallelEmbedding"}
SKIP_CLASSES = {"ParallelLMHead"}   # its forward is not called by vLLM (LogitsProcessor applies the quant method); the logits are acquired there


def _mro_names(m: Any) -> set[str]:
    return {c.__name__ for c in type(m).__mro__}


class SkipRefused(RuntimeError):
    """VERITY_SKIP names an ablation that cannot be applied as named: the run is refused, never labelled with a skip that did not happen."""


def skip_classes_of_env() -> frozenset[str]:
    """VERITY_SKIP: a comma list of module classes whose module-output acquisition a per-row MARGINAL-cost ablation drops (NOT a commitment)."""
    return frozenset(settings.VERITY_SKIP)


def _dtype_name(t: Any) -> str:
    return str(t.dtype).replace("torch.", "")


@dataclass
class TensorMeta:
    ordinal: int
    name: str            # module name (+ "/in" for an acquired input, "/k" for tuple element k)
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    chunk: int
    first_leaf: int      # rank of the first leaf of this tensor in the step tree
    n_leaves: int
    host: memoryview | None = field(default=None, repr=False)   # pinned bytes (set when copied)
    dev_off: int = 0                                             # arena staging: byte offset of this tensor in its device block
    stream_off: int = -1                                         # packed mode: byte offset of this tensor in the step's packed stream
    dev_block: Any = field(default=None, repr=False)             # arena staging: the device block (released after D2H)
