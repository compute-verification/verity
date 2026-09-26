"""C-Flock lowering of ``silu-mul`` (SiluMul_v1): I identical units read off the IR graph, each one ``SiluMulBf16`` over
(gate[i], up[i]), lowered piece by piece (``verity_flock.ir_lower``): ``bf16(silu(g))`` is a 16-bit lookup enumerated from
the primitive's own helpers, then an f32 multiply and the torch bf16 cast.

The unit's ``flock-ir-unit/v1`` netlist does not depend on I and is pinned by :data:`PIN`."""
TEMPLATE = "silu-mul"
UNIT = "silu-mul/bf16/element"
PIN = "3b1ed294ef438c1e4eb1b7d61eacd3b7a88ed78d1713e5a72752c35d6f5f96f7"


def definition(subcircuit):
    return subcircuit.definition()


def lowering(subcircuit):
    from verity_flock import ir_lower

    return ir_lower.lowering(definition(subcircuit), UNIT)


#: Components per unit in the frame-v3 statement (a block holds at most 256 unit slots and every chunk its units read).
FRAME_GROUP = 2
FRAME_SCHEMES = ("frame-v3/blake3-keyed",)


def frame_lowering(subcircuit):
    from verity_flock import ir_lower

    return ir_lower.lowering(definition(subcircuit), UNIT + (f"/x{FRAME_GROUP}" if FRAME_GROUP > 1 else ""), group=FRAME_GROUP)


def supports(subcircuit, scheme):
    if scheme in FRAME_SCHEMES:
        return None
    return (f"C-Flock proves {subcircuit.id} under {', '.join(FRAME_SCHEMES)} (verity/flock-ir-frame/v1); binding its rows under "
            f"{scheme} is not wired yet")


def relation(subcircuit, scheme):
    return "irf:silu-mul"


def stage(input_set, lo, hi, out_dir):
    """The frame-v3 instance file for ids [lo, hi) (``verity_flock.ir_frame.stage``): each pod stages its own."""
    from verity_flock import ir_frame

    return ir_frame.stage(frame_lowering(input_set.subcircuit), input_set, lo, hi, out_dir)
