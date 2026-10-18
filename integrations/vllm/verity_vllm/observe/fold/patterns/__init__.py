"""The fold's kernel patterns, one module per kernel family, and the helpers they share.

Everything here is pure Python over the event trees (`tree.TreeNode`) and the step context
(`resolver.StepContext`); it never reads tensor values.  The patterns are registered by a profile module
(`profiles/vllm_d9105ea80_sm89_eager.py`) which also pins the constexpr / launch-geometry values they accept.

Vocabulary (D68): a pattern maps a raw event subtree to one of the resolver's outcomes: `Compute` (rows of a
registered *specialization*, one *instance* per row), `Transparent` (alias / move / alloc / noop) or `Unsupported`.

Shared helpers: `resolutions` (grouped resolutions, `unsupported`, report labels), `ops` (op names, bound arguments),
`geometry` (operand views), `transparent` (transparent ops, scope helpers), `target` (the record target's facts).
Families: `gemm`, `norms`, `rotary`, `kv_cache`, `attention`, `activations`, `embedding`, `sampling`, `fp8`, `collectives`.
`fp8` and `collectives` import their own registries (`registry.fp8`, `registry.b1_tp2`) and are not imported here, so
importing the package does not load those registries.
"""

from __future__ import annotations

from verity_vllm.observe.fold.patterns.activations import SiluAndMul
from verity_vllm.observe.fold.patterns.attention import FA2_OP_NAMES, AttentionScope
from verity_vllm.observe.fold.patterns.embedding import Embedding
from verity_vllm.observe.fold.patterns.gemm import BiasAdd, GemmLaunch, fuse_bias_adds
from verity_vllm.observe.fold.patterns.geometry import GeometryError, as_slot_rows, as_weight_view, is_int, is_row_major, split_heads, sub_rows
from verity_vllm.observe.fold.patterns.kv_cache import KVCacheUpdate, kv_group_of, kv_layer_of, kv_slots
from verity_vllm.observe.fold.patterns.norms import FusedAddRMSNorm, RMSNormTritonLaunch
from verity_vllm.observe.fold.patterns.ops import as_float, base_op, bound_args, first, op_name, schema_param_names, triton_name
from verity_vllm.observe.fold.patterns.resolutions import (ComputeGroup, TransparentGroup, _check_pins, _fmt, _group_compute, attribution_for,
                                                           hint_of, label, members, unsupported)
from verity_vllm.observe.fold.patterns.rotary import RotaryEmbedding, _positions_ok
from verity_vllm.observe.fold.patterns.sampling import (SCRATCH_TOKEN_SELECT, LogitsGather, TokenSelectArgmax, TokenSelectGumbelTopP,
                                                        TokenSelectGumbelTwoStage)
from verity_vllm.observe.fold.patterns.target import GEMM_TARGET_SCRATCH_KEY, X01_SCRATCH_KEY, gemm_target_of, target_of
from verity_vllm.observe.fold.patterns.transparent import (ALIAS_OPS, TransparentAten, TransparentTritonMetadata, classify_transparent,
                                                           find_descendants, scope_transparents, widened_source)
from verity_vllm.observe.fold.resolver import PerRow, Shared


def dsa_patterns() -> list:
    """Load GLM-5.3's DSA patterns without making the shared package import them."""
    from verity_vllm.observe.fold.patterns.dsa import dsa_patterns as build
    return build()
