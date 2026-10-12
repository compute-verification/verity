"""Subcircuit bodies the interface fixes canonically, so both tracks derive one digest.

A subcircuit that originates at the interface (the serving exporter projects a
captured proof unit to it) is identified by
:func:`~veritor.interface.typed_obligation.subcircuit_digest` of its body.
Two exporters that build "the same" GEMM coordinate must therefore build the
*same body* -- the same incoming order, the same gate order, the same argument
order -- or Track B sees two subcircuits.  This module is where that body is
fixed.

Only the ``veritor.tc-ampere-bf16@1`` GEMM coordinate is pinned so far: the
body of one output word of a BF16 GEMM under the batch-invariant Triton
``matmul_persistent`` kernel, as the numerics lane validated it (one
``mma.sync.m16n8k16`` chain over ascending 16-wide ``k`` slices, then round to
nearest even) and as W3c's ``verity_vllm_synth.lowering.DagBuilder.gemm_coordinate``
builds it.  The interface's family for it is ``veritor.tc-ampere-bf16@1``
(``veritor.core.tc``), whose ``tc_dot16``/``tc_dot16_0``/``f32_to_bf16`` have
the semantics of the serving side's pinned set for exactly these gates.
"""

from __future__ import annotations

from collections.abc import Sequence

from .typed_obligation import GATE, IN, INPUT, Definition, Gate, Subcircuit, subcircuit_digest

__all__ = ["TC_K", "GEMM_WITHIN", "GateSpec", "input_gate", "make_subcircuit", "gemm_coordinate", "gemm_accumulator_fp8",
           "gemm_accumulator_nvfp4", "input_equation", "elementwise"]

#: The ``k`` extent of one Ampere BF16 tensor-core step.
TC_K = 16

#: A gate as the builders spell it: ``(op, args)`` for a primitive call, or the
#: result of :func:`input_gate` for a prescribed-input gate.
type GateSpec = tuple[str, Sequence[tuple[str, int]]] | Gate


def input_gate(width: int, value: int) -> Gate:
    """The prescribed-input gate ``Input[k] : () -> Value<width>`` with prescribed value ``value``."""

    return Gate(INPUT, (), width, value.to_bytes((width + 7) // 8, "big").hex())


def make_subcircuit(
    incoming: Sequence[int],
    gates: Sequence[GateSpec],
    outgoing: Sequence[int],
    definition: str | Definition | None = None,
) -> Subcircuit:
    """A subcircuit with incoming boundary values of the given widths, named by its body digest.

    ``definition`` (a string or a full :class:`Definition`) is the informative
    annotation; it does not enter the digest.
    """

    annotation = Definition(definition) if isinstance(definition, str) else definition
    body = tuple(g if isinstance(g, Gate) else Gate(g[0], tuple(g[1])) for g in gates)
    s = Subcircuit("00" * 32, tuple(incoming), body, tuple(outgoing), annotation)
    return Subcircuit(subcircuit_digest(s), s.incoming, s.gates, s.outgoing, annotation)


#: The registered composite the GEMM coordinate is a subcircuit of (IR §B.7, §B.11
#: answer 2: an approved decomposition *is* a registered composite function
#: definition).  ``Gemm<K,N>(x: Array<K,Value<16>>, W: Array<N,Array<K,Value<16>>>) ->
#: Array<N,Value<16>>`` is the composite ``batch(GemmCoordinate<K>, x, W, axes=(None,
#: 0))``: N dataflow-independent calls, member ``j`` reading ``x`` (shared) and
#: ``W[j, :]`` and producing the one word ``out[j]``; its body is ``tc_dot16_0``,
#: ``tc_dot16`` chained over the 16-slices, ``f32_to_bf16``, every gate one
#: ``Value<w>``.  Those calls and gates are the canonical circuit; a coordinate
#: obligation is a subcircuit of it.  That the batch-invariant serving kernel
#: implements this ``Gemm`` is the serving side's conformance evidence (roles, IR
#: §10: specification = this body over L1's family; reference evaluators =
#: veritor.core.tc / the guest's common crate / Lean TCB; serving implementation
#: = matmul_persistent; proof checker = the guest).
GEMM_WITHIN = "Gemm"


def gemm_coordinate(k: int, *, opened_accumulator: bool = False) -> Subcircuit:
    """One GEMM output coordinate over ``k`` (a multiple of 16) in ``veritor.tc-ampere-bf16@1``.

    Incoming values ``0 .. k`` are the A operands (the activation row,
    ``x[0..k]``), ``k .. 2k`` the B operands (the weight row ``W[j, 0..k]``),
    all 16-bit BF16 words in ascending ``k`` order.  Gate ``0`` is
    ``tc_dot16_0(x[0:16], w[0:16])``; gate ``s`` (``1 <= s < k/16``) is
    ``tc_dot16(acc_{s-1}, x[16s:16s+16], w[16s:16s+16])``; the last gate is
    ``f32_to_bf16(acc)`` -- the BF16 word the kernel stores -- and is the one
    outgoing value.  With ``opened_accumulator`` the cast is omitted and the
    FP32 accumulator itself is outgoing.  The weight words are incoming values
    here -- the ``Input`` gates that prescribe them are outside this subcircuit
    (unsampled by it, IR §E); a plan that samples them puts them in the body.
    """

    if k <= 0 or k % TC_K:
        raise ValueError(f"k={k} must be a positive multiple of {TC_K}")
    steps = k // TC_K
    gates: list[GateSpec] = [
        (f"tc_dot{TC_K}_0", [(IN, i) for i in range(TC_K)] + [(IN, k + i) for i in range(TC_K)])
    ]
    for step in range(1, steps):
        a = [(IN, TC_K * step + i) for i in range(TC_K)]
        b = [(IN, k + TC_K * step + i) for i in range(TC_K)]
        gates.append((f"tc_dot{TC_K}", [(GATE, step - 1), *a, *b]))
    if opened_accumulator:
        definition = Definition("GemmAccumulator", (("K", k),), GEMM_WITHIN)
        return make_subcircuit([16] * (2 * k), gates, [steps - 1], definition)
    gates.append(("f32_to_bf16", [(GATE, steps - 1)]))
    definition = Definition("GemmCoordinate", (("K", k),), GEMM_WITHIN)
    return make_subcircuit([16] * (2 * k), gates, [steps], definition)


def gemm_accumulator_fp8(k: int, *, step_k: int = 32) -> Subcircuit:
    """``GemmAccumulator<K>`` over ``k`` (a multiple of ``step_k``) for the FP8 draft targets: the bare dot with the
    FP32 word outgoing (:data:`verity.ml.circuits.targets.target.FP8_HOPPER_WGMMA` / :data:`FP8_ADA_MMA`).

    Incoming values ``0 .. k`` are the A operands (8-bit E4M3 words), ``k .. 2k`` the B operands, ascending ``k``.
    Gate ``0`` is ``tc_dot{step_k}_0(a[0:step_k], b[0:step_k])``; gate ``s`` is ``tc_dot{step_k}(acc_{s-1}, ...)``;
    the last accumulator word is the one outgoing value -- there is **no** epilogue cast: the scale-and-
    ``cvt.rn.satfinite.e4m3x2`` store of a serving FP8 GEMM is a separate relation.  The ops are those of
    ``veritor.tensor-core.<pipeline>@2`` (:func:`verity_sp1.proofs.families.make_tensor_core_gate_set`) for a
    ``step_k``-wide pipeline; the body is the same for Hopper (one group of 32) and Ada (two of 16): the grouping is
    the gate's semantics, not the program's.  Draft: no guest checks this family yet.
    """

    if k <= 0 or k % step_k:
        raise ValueError(f"k={k} must be a positive multiple of {step_k}")
    steps = k // step_k
    gates: list[GateSpec] = [
        (f"tc_dot{step_k}_0", [(IN, i) for i in range(step_k)] + [(IN, k + i) for i in range(step_k)])
    ]
    for step in range(1, steps):
        a = [(IN, step_k * step + i) for i in range(step_k)]
        b = [(IN, k + step_k * step + i) for i in range(step_k)]
        gates.append((f"tc_dot{step_k}", [(GATE, step - 1), *a, *b]))
    definition = Definition("GemmAccumulator", (("K", k),), GEMM_WITHIN)
    return make_subcircuit([8] * (2 * k), gates, [steps - 1], definition)


def gemm_accumulator_nvfp4(k: int, *, step_k: int = 64, block: int = 16) -> Subcircuit:
    """``GemmAccumulator<K>`` over ``k`` (a multiple of ``step_k``) for the NVFP4 draft target
    (:data:`verity.ml.circuits.targets.target.NVFP4_SM120`): the bare block-scaled dot with the FP32 word outgoing.

    Incoming values: ``0 .. k`` the A element codes (4-bit E2M1), ``k .. k + k/block`` the A scale bytes (8-bit UE4M3,
    one per ``block`` codes, ascending), then the same two runs for B (``k + k/block .. 2k + k/block`` codes,
    ``2k + k/block .. 2(k + k/block)`` scales).  Gate ``s`` is ``tc_dot{step_k}[_0](acc, a codes[s], a scales[s], b codes[s],
    b scales[s])`` -- one operand of a step is its ``step_k`` codes followed by its ``step_k / block`` scale bytes, the
    argument order ``BlockScaledAlignAdd.step`` takes (:func:`verity_sp1.proofs.families.tensor_core_gates`).  No
    epilogue cast (the serving NVFP4 store -- scale, then quantise -- is a separate relation).  Draft: no guest
    checks this family yet.
    """

    if k <= 0 or k % step_k or step_k % block:
        raise ValueError(f"k={k} must be a positive multiple of {step_k} (a multiple of the block {block})")
    steps, spb = k // step_k, step_k // block            # scale bytes per step
    nsc = k // block                                     # scale bytes per operand
    a0, as0, b0, bs0 = 0, k, k + nsc, 2 * k + nsc        # where each run of incoming values starts

    def operand(codes0: int, scales0: int, step: int) -> list[tuple[str, int]]:
        codes = [(IN, codes0 + step_k * step + i) for i in range(step_k)]
        scales = [(IN, scales0 + spb * step + j) for j in range(spb)]
        return codes + scales

    gates: list[GateSpec] = [(f"tc_dot{step_k}_0", operand(a0, as0, 0) + operand(b0, bs0, 0))]
    for step in range(1, steps):
        gates.append((f"tc_dot{step_k}", [(GATE, step - 1), *operand(a0, as0, step), *operand(b0, bs0, step)]))
    definition = Definition("GemmAccumulator", (("K", k),), GEMM_WITHIN)
    widths = [4] * k + [8] * nsc
    return make_subcircuit(widths * 2, gates, [steps - 1], definition)


def input_equation(expected: int, width: int = 16) -> Subcircuit:
    """``InputEq<W>[v]``: the body of one sampled input equation -- a single prescribed ``Input<width>``
    gate with value ``expected`` as the one outgoing value (IR §E; the lowering pins the committed
    position to ``v``).  No incoming values.

    [A3, Q-A56-2 2026-09-10] The canonical constructor, mirrored from the serving exporter
    (``verity_vllm.typed_obligation_export.input_equation_subcircuit``) so the verifier derives the
    expectation's subcircuit digest without importing prover-side code; the body -- and therefore the
    digest -- is identical (the ``Definition`` annotation does not enter the digest).  ``width`` is the
    family's word width (16 for the tc families; a 32-bit boundary token is pinned as its two halves).
    """

    if not 0 <= int(expected) < (1 << width):
        raise ValueError(f"input equation value {expected} is not a {width}-bit word")
    return make_subcircuit([], [input_gate(width, int(expected))], [0], Definition("InputEq", (("W", width),)))


def elementwise(gate: str, arity: int, width: int = 16, *, prim: str = "", kind: str = "") -> Subcircuit:
    """``<gate>`` over ``arity`` incoming ``Value<width>`` words: the body of one elementwise proof unit (a
    batch member or primitive proof unit of a registered primitive the approved guest evaluates as one gate -- ``silu_mul_bf16``
    over the gate/up words, ``bf16_add_f2fp`` over the two residual words, ``rope_out``/``rope_out_add`` over
    ``(x, y, cos, sin)``).  Incoming values are the primitive's operands in its signature order; gate 0 is the one
    outgoing value.  ``prim``/``kind`` annotate (``Definition``; not in the digest).

    [CK-ELEM 2026-09-10] The canonical constructor, so the verifier derives an elementwise expectation's subcircuit
    digest without importing the builder (``e2e.coverage_v1.ck_elem.kind_subcircuit`` builds the same body).
    """

    if arity <= 0:
        raise ValueError(f"arity={arity} must be positive")
    gates: list[GateSpec] = [(gate, [(IN, i) for i in range(arity)])]
    return make_subcircuit([width] * arity, gates, [0], Definition(prim or gate, (), kind or None))
