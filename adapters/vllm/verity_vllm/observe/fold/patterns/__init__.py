"""The fold's kernel patterns, one module per kernel family, and the helpers they share.

Everything here is pure Python over the event trees (`tree.TreeNode`) and the step context
(`resolver.StepContext`); it never reads tensor values.  The patterns are registered by a profile module
(`profiles/vllm_d9105ea80_sm89_eager.py`) which also pins the constexpr / launch-geometry values they accept.

Vocabulary: a pattern maps a raw event subtree to one of the resolver's outcomes: `Compute` (rows of a
registered *specialization*, one *instance* per row), `Transparent` (alias / move / alloc / noop) or `Unsupported`.

Shared helpers: `resolutions` (grouped resolutions, `unsupported`), `ops` (op names, bound arguments),
`geometry` (operand views), `transparent` (transparent ops, scope helpers), `target` (the record target's facts).
Families: `gemm`, `norms`, `rotary`, `kv_cache`, `attention`, `activations`, `embedding`, `sampling`, `fp8`, `collectives`.
`fp8` and `collectives` import their own registries (`registry.fp8`, `registry.b1_tp2`) and are not imported here, so
importing the package does not load those registries.
"""

from __future__ import annotations

from verity_vllm.observe.fold.patterns.activations import SiluAndMul as SiluAndMul
from verity_vllm.observe.fold.patterns.activations import SiluAndMulWithClamp
from verity_vllm.observe.fold.patterns.attention import FA2_OP_NAMES, AttentionScope
from verity_vllm.observe.fold.patterns.embedding import Embedding
from verity_vllm.observe.fold.patterns.gemm import BiasAdd, GemmLaunch, fuse_bias_adds
from verity_vllm.observe.fold.patterns.geometry import (
    GeometryError,
    as_slot_rows,
    as_weight_view,
    is_int,
    is_row_major,
    split_heads,
    sub_rows,
)
from verity_vllm.observe.fold.patterns.kv_cache import (
    KVCacheUpdate,
    kv_group_of,
    kv_layer_of,
    kv_slots,
)
from verity_vllm.observe.fold.patterns.norms import FusedAddRMSNorm, RMSNormTritonLaunch
from verity_vllm.observe.fold.patterns.ops import (
    as_float,
    base_op,
    bound_args,
    first,
    op_name,
    schema_param_names,
    triton_name,
)
from verity_vllm.observe.fold.patterns.resolutions import (
    ComputeGroup,
    TransparentGroup,
    _check_pins,
    _fmt,
    _group_compute,
    attribution_for,
    hint_of,
    members,
    unsupported,
)
from verity_vllm.observe.fold.patterns.rotary import RotaryEmbedding as RotaryEmbedding
from verity_vllm.observe.fold.patterns.rotary import _positions_ok
from verity_vllm.observe.fold.patterns.sampling import (
    LogitsGather,
    TokenSelectArgmax,
    TokenSelectGumbelTopP,
    TokenSelectGumbelTwoStage,
)
from verity_vllm.observe.fold.patterns.target import (
    GEMM_TARGET_SCRATCH_KEY,
    X01_SCRATCH_KEY,
    gemm_target_of,
    target_of,
    vocab_kind,
)
from verity_vllm.observe.fold.patterns.transparent import (
    ALIAS_OPS,
    SCRATCH_INT_FILLS,
    TransparentAten,
    TransparentTritonMetadata,
    classify_transparent,
    find_descendants,
    scope_transparents,
    widened_source,
)

# The package's names.  `RotaryEmbedding` and `SiluAndMul` are exported as the two `import X as X` lines above, since a
# string naming a vLLM layer class is a model fact, which lives in the profiles.
__all__ = [
    "ALIAS_OPS", "AttentionScope", "BiasAdd", "ComputeGroup", "Embedding", "FA2_OP_NAMES", "FusedAddRMSNorm",
    "GEMM_TARGET_SCRATCH_KEY", "GemmLaunch", "GeometryError", "KVCacheUpdate", "LogitsGather", "RMSNormTritonLaunch",
    "SCRATCH_INT_FILLS", "SiluAndMulWithClamp", "TokenSelectArgmax", "TokenSelectGumbelTopP", "TokenSelectGumbelTwoStage",
    "TransparentAten", "TransparentGroup", "TransparentTritonMetadata", "X01_SCRATCH_KEY", "_check_pins", "_fmt",
    "_group_compute", "_positions_ok", "as_float", "as_slot_rows", "as_weight_view", "attribution_for", "base_op", "bound_args",
    "classify_transparent", "dsa_patterns", "dsv4_patterns", "find_descendants", "first", "fuse_bias_adds", "gemm_target_of",
    "hint_of", "is_int", "is_row_major", "kv_group_of", "kv_layer_of", "kv_slots", "members", "op_name", "schema_param_names",
    "scope_transparents", "split_heads", "sub_rows", "target_of", "triton_name", "unsupported", "vocab_kind", "widened_source",
]


def dsa_patterns() -> list:
    """Load GLM-5.3's DSA patterns without making the shared package import them."""
    from verity_vllm.observe.fold.patterns.dsa import dsa_patterns as build
    return build()


def dsv4_patterns() -> list:
    """Load DeepSeek-V4's patterns without making the shared package import them."""
    from verity_vllm.observe.fold.patterns.dsv4 import dsv4_patterns as build
    return build()
