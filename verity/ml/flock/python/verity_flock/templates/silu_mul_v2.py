"""C-Flock lowering of ``silu-mul-v2`` (SiluMul_v2, vLLM's served ``_C.silu_and_mul``): I identical units read off the IR graph,
each one ``SiluMulBf16_v2`` over (gate[i], up[i]).

``SiluMulBf16_v2`` differs from ``SiluMul_v1``'s primitive on NaN (0x7FFF, not 0x7FC0), on the up word (``+ 0.0f`` turns -0 into
+0), on gate -inf (NaN, not -0), on the gates whose ``expf(-g)`` overflows (g <= -89: -0, not a tiny nonzero) and on gate -0
(kept, not +0).  C-Flock has no piece of its own for it: the unit is the registered Boolean Definition whose word is
``SiluMulBf16_v2``, ``boolean_silu.SiluMulBf16Expf`` (``SiluMulBf16_v4``), run as its builder (``ir_lower._boolean_build``).
That circuit is exact arithmetic on g's fields off a band of exponents and a constant table on it, so it is half
``silu-mul``'s 16-bit lookup in ANDs (1,380 against 2,606).

The unit's rows do not depend on I; its ``flock-ir-unit/v2`` netlist, whose LEAVES line places the units on the row's words, is
pinned per I by :data:`PINS` (GLM-5.3 at TP8: I = 1536 for the dense layers, 256 for the shared expert). ``verity/flock-circuit``
holds a power-of-two number of units an instance, so there a unit is I's odd part of consecutive elements
(:func:`circuit_lowering`, pinned by :data:`CIRCUIT_PINS` where that part is not 1)."""
TEMPLATE = "silu-mul-v2"
UNIT = "silu-mul-v2/bf16/element"
PINS = {256: "a5907d96d7cb5f22067c8a4ffd90a50c6e226ddcd1314a38df49c82ab6f7610d",
        1536: "8f2324d6184b34d7a5ad79e03480e0b6e6e98466d09d0f5f2ed2f53e9b95d06e"}
CIRCUIT_PINS = {1536: "861aaba839ead11166bd84e3f028a39988fc033022aab8f85494838a1e8bb8e7"}


def definition(subcircuit):
    import verity_vllm.program.registry.boolean_silu  # noqa: F401  (registers SiluMulBf16_v4, the unit's builder)

    return subcircuit.definition()


def lowering(subcircuit):
    from verity_flock import ir_lower

    return ir_lower.lowering(definition(subcircuit), UNIT)


def circuit_lowering(subcircuit):
    """The ``verity/flock-circuit`` lowering: I's odd part m of consecutive elements a unit, so an instance holds I / m units, a
    power of two (I = 1536: three elements, 512 units; I = 256: :func:`lowering` itself)."""
    from verity_flock import ir_lower

    I = int(subcircuit.params["I"])
    m = I // (I & -I)
    return ir_lower.lowering(definition(subcircuit), UNIT + (f"/x{m}" if m > 1 else ""), group=m)


#: Components per unit in the frame-v3 statement (a block holds at most 256 unit slots and every chunk its units read).
FRAME_GROUP = 1
FRAME_SCHEMES = ("frame-v3/blake3-keyed",)


def frame_lowering(subcircuit):
    from verity_flock import ir_lower

    return ir_lower.lowering(definition(subcircuit), UNIT + (f"/x{FRAME_GROUP}" if FRAME_GROUP > 1 else ""), group=FRAME_GROUP)


def supports(subcircuit, scheme):
    if scheme in FRAME_SCHEMES:
        return None
    return (f"C-Flock proves {subcircuit.id} under {', '.join(FRAME_SCHEMES)} (verity/flock-ir-frame/v3); binding its rows under "
            f"{scheme} is not wired yet")


def relation(subcircuit, scheme):
    return "irf:silu-mul-v2"


def stage(input_set, lo, hi, out_dir):
    """The frame-v3 instance file for ids [lo, hi) (``verity_flock.ir_frame.stage``): each pod stages its own."""
    from verity_flock import ir_frame

    return ir_frame.stage(frame_lowering(input_set.subcircuit), input_set, lo, hi, out_dir)
