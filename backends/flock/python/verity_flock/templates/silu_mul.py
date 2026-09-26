"""C-Flock lowering of ``silu-mul`` (SiluMul_v1): I identical units read off the IR graph, each one ``SiluMulBf16`` over
(gate[i], up[i]), lowered piece by piece (``verity_flock.ir_lower``): ``bf16(silu(g))`` is a 16-bit lookup enumerated from
the primitive's own helpers, then an f32 multiply and the torch bf16 cast.

The unit's ``flock-ir-unit/v1`` netlist does not depend on I and is pinned by :data:`PIN`."""
TEMPLATE = "silu-mul"
UNIT = "silu-mul/bf16/element"
PIN = "f40ec4ad44935001918c3b9f75e8d8a1560ee8a0e30ca5a6d20cec97414c1fb0"


def definition(subcircuit):
    return subcircuit.definition()


def lowering(subcircuit):
    from verity_flock import ir_lower

    return ir_lower.lowering(definition(subcircuit), UNIT)


def supports(subcircuit, scheme):
    return (f"C-Flock lowers {subcircuit.id} exactly (verity_flock.templates.silu_mul: unit {UNIT}) and proves it with public "
            f"unit inputs and outputs (verity/flock-ir-block/v1); binding its rows under {scheme} is not wired yet")


def stage(input_set, lo, hi, out_dir):
    from verity_flock import ir_lower

    return ir_lower.stage(lowering(input_set.subcircuit), input_set, lo, hi, out_dir)
