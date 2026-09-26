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
    return (f"C-Flock lowers {subcircuit.id} exactly (verity_flock.templates.silu_mul: unit {UNIT}) and proves it with public "
            f"unit inputs and outputs (verity/flock-ir-block/v1); binding its rows under {scheme} is not wired yet")


def stage(input_set, lo, hi, out_dir):
    from verity_flock import ir_lower

    return ir_lower.stage(lowering(input_set.subcircuit), input_set, lo, hi, out_dir)
