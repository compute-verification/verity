"""``veritor.tc-ampere-bf16@1`` / ``@2``: the tensor-core gate families a checker recomputes, over ``verity.ml.circuits.silicon``.

A gate family is the checker-facing vocabulary of a lowered relation: the ``op`` names a ``KindProgram`` may use,
each with an arity, argument widths, an output width and executable semantics, and an identity digest the guest
refuses to check a statement against unless it equals the digest of the table it is about to apply
(``archive/sp1/common/src/gateset.rs``).  The semantics themselves are ``verity.ml.circuits.silicon``'s; nothing here
redefines a tensor-core step.

The family is L1's pinned Ampere step plus the two casts, with 16-bit sources:

* ``tc_dot16`` (``[acc, a0..a15, b0..b15]`` -> FP32 word) and ``tc_dot16_0`` (``[a0..a15, b0..b15]`` -> FP32 word):
  :func:`verity.ml.circuits.silicon.silicon.tc_dot` on the Ampere pipeline, bit for bit.  This is the **finite-domain** relation:
  a non-finite operand or accumulator raises and the checker rejects the obligation (the same verdict as the
  Rust ``tc.rs`` and the Lean ``TCB.lean``).  ``verity.ml.circuits.ops.prims.AmpereBF16TcDot16`` (v2) is the *total* extension
  of the same step; the two agree on every finite input, and no registered family binds the total semantics yet
  (``tc_total.rs`` exists in the guest crate without a gate set).  Which relation Verity declares for a sampled
  step is a protocol decision, not a backend one.
* ``f32_to_bf16`` (32 -> 16): round to nearest, ties to even; every NaN becomes one canonical word -- ``0x7FC0``
  in ``@1`` (the software canonicalisation), ``0x7FFF`` in ``@2`` (the measured ``cvt.rn.bf16.f32`` word; the
  serving epilogue).  The versions agree on every non-NaN word.
* ``bf16_to_f32`` (16 -> 32): the exact widening.
* ``in`` and ``weight``: 16-bit source words.

``veritor.tensor-core.<pipeline>@2`` (:func:`make_tensor_core_gate_set`) is the earlier, cast-free family: the
two steps of one pipeline with 32-bit sources; the conformance suite pins its Ampere BF16 digest.

Promoted from veritor ``core/tc.py`` and the gate builders of ``core/silicon.py`` (eb865ce2); the manifests, and
so the digests, are unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

from verity.ml.circuits.silicon.cast import bf16_to_f32_word, f32_to_bf16_hw_word, f32_to_bf16_word
from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16, Pipeline, tc_dot
from verity.ml.circuits.targets.target import TC_AMPERE_BF16_NAME, TC_AMPERE_BF16_VERSION, TC_BF16_NAME_PREFIX, TC_BF16_VERSION

from .gateset import INPUT_SOURCE, WEIGHT_SOURCE, Gate, GateSet

__all__ = [
    "make_tensor_core_gate_set",
    "TC_AMPERE_BF16_HW_VERSION",
    "TC_AMPERE_BF16_NAME",
    "TC_AMPERE_BF16_VERSION",
    "TC_AMPERE_BF16_VERSIONS",
    "TC_BF16_NAME_PREFIX",
    "TC_BF16_VERSION",
    "make_tc_ampere_bf16_gate_set",
    "make_tc_bf16_gate_set",
    "tensor_core_gates",
]

#: The version whose ``f32_to_bf16`` is the hardware ``cvt.rn.bf16.f32`` (NaN -> ``0x7FFF``).
TC_AMPERE_BF16_HW_VERSION = "2"
TC_AMPERE_BF16_VERSIONS = (TC_AMPERE_BF16_VERSION, TC_AMPERE_BF16_HW_VERSION)


def tensor_core_gates(pipeline: Pipeline) -> tuple[Gate, Gate]:
    """``tc_dot{k}`` and ``tc_dot{k}_0``: one ``mma`` step with, and without, an incoming accumulator."""

    k = pipeline.k
    # One operand of a step is ``words_per_step`` words: the ``k`` element codes and, for a block-scaled pipeline
    # (``BlockScaledAlignAdd``: ``scale`` set), its ``k / block`` 8-bit scale bytes after them.  A grouped-sum pipeline
    # has ``words_per_step == k`` and no scales, so its gates (and the family digests) are unchanged.
    n = pipeline.words_per_step
    widths = (pipeline.operand_bits,) * k + (8,) * (n - k)

    def evaluate(args: tuple[int, ...]) -> int:
        return tc_dot(pipeline, args[0], args[1 : 1 + n], args[1 + n :])

    def evaluate_zero(args: tuple[int, ...]) -> int:
        return tc_dot(pipeline, 0, args[:n], args[n:])

    return (
        Gate(f"tc_dot{k}", 1 + 2 * n, 32, replay_cost=k, proof_cost=k, evaluate=evaluate,
             arg_widths=(32, *(widths * 2))),
        Gate(f"tc_dot{k}_0", 2 * n, 32, replay_cost=k, proof_cost=k, evaluate=evaluate_zero,
             arg_widths=widths * 2),
    )


def _bf16_gate_set(pipeline: Pipeline, name: str, version: str, cast: Callable[[int], int]) -> GateSet:
    """The two ``tc_dot16`` steps of a BF16 ``pipeline`` plus the two casts and 16-bit sources, under ``name@version``."""

    if pipeline.operand_bits != 16:
        raise ValueError(f"{pipeline.name} has {pipeline.operand_bits}-bit operands; the BF16 family needs 16")
    step, step_zero = tensor_core_gates(pipeline)
    return GateSet(
        (
            step,
            step_zero,
            Gate("f32_to_bf16", 1, 16, replay_cost=1, proof_cost=1, evaluate=lambda args: cast(args[0]),
                 arg_widths=(32,)),
            Gate("bf16_to_f32", 1, 32, replay_cost=1, proof_cost=1, evaluate=lambda args: bf16_to_f32_word(args[0]),
                 arg_widths=(16,)),
            Gate("in", 0, 16, replay_cost=0, proof_cost=1, source=INPUT_SOURCE),
            Gate("weight", 0, 16, replay_cost=0, proof_cost=1, source=WEIGHT_SOURCE),
        ),
        name=name,
        version=version,
    )


def make_tc_ampere_bf16_gate_set(version: str = TC_AMPERE_BF16_VERSION) -> GateSet:
    """The gate set ``veritor.tc-ampere-bf16@<version>``; see the module docstring."""

    if version not in TC_AMPERE_BF16_VERSIONS:
        raise ValueError(f"{TC_AMPERE_BF16_NAME} has versions {TC_AMPERE_BF16_VERSIONS}, not {version!r}")
    cast: Callable[[int], int] = f32_to_bf16_word if version == TC_AMPERE_BF16_VERSION else f32_to_bf16_hw_word
    return _bf16_gate_set(AMPERE_BF16_M16N8K16, TC_AMPERE_BF16_NAME, version, cast)


def make_tc_bf16_gate_set(pipeline: Pipeline) -> GateSet:
    """``veritor.tc-bf16.<pipeline.name>@1``: the ``veritor.tc-ampere-bf16@1`` vocabulary (two ``tc_dot16`` steps,
    ``f32_to_bf16`` with the canonical ``0x7FC0`` NaN, ``bf16_to_f32``, 16-bit sources) with the steps computed on
    another BF16 pipeline -- the family of the Hopper BF16 draft target (``HOPPER_BF16_M16N8K16``: one group of 16,
    the 26-bit adder, floor -133).  The Ampere family keeps its own name and digest."""

    return _bf16_gate_set(pipeline, f"{TC_BF16_NAME_PREFIX}{pipeline.name}", TC_BF16_VERSION, f32_to_bf16_word)


def make_tensor_core_gate_set(pipeline: Pipeline = AMPERE_BF16_M16N8K16) -> GateSet:
    """``veritor.tensor-core.<pipeline.name>@2``: the two ``tc_dot`` steps of one pipeline plus 32-bit sources."""

    return GateSet(
        (
            *tensor_core_gates(pipeline),
            Gate("in", 0, 32, replay_cost=0, proof_cost=1, source=INPUT_SOURCE),
            Gate("weight", 0, 32, replay_cost=0, proof_cost=1, source=WEIGHT_SOURCE),
        ),
        name=f"veritor.tensor-core.{pipeline.name}",
        version="2",
    )
