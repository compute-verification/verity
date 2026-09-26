"""C-Flock lowering of ``rope-head`` (RoPEHead_v1): D/2 identical units read off the IR graph, each the pair
``RopeOut`` + ``RopeOutAdd`` over (x[i], x[i + D/2], cos[i], sin[i]), lowered piece by piece (``verity_flock.ir_lower``).

The unit's ``flock-ir-unit/v1`` netlist does not depend on D (only the arrangement does) and is pinned by :data:`PIN`."""
TEMPLATE = "rope-head"
UNIT = "rope-head/neox-bf16/pair"
PIN = "2eaa652ff26483e788b8d3dd54abd134981605dd0927d2273cccf1ea8e646e14"


def definition(subcircuit):
    return subcircuit.definition()


def lowering(subcircuit):
    from verity_flock import ir_lower

    return ir_lower.lowering(definition(subcircuit), UNIT)


def supports(subcircuit, scheme):
    return (f"C-Flock lowers {subcircuit.id} exactly (verity_flock.templates.rope_head: unit {UNIT}) and proves it with public "
            f"unit inputs and outputs (verity/flock-ir-block/v1); binding its rows under {scheme} is not wired yet")


def stage(input_set, lo, hi, out_dir):
    from verity_flock import ir_lower

    return ir_lower.stage(lowering(input_set.subcircuit), input_set, lo, hi, out_dir)
