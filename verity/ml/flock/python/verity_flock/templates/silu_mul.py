"""Flock lowering of ``silu-mul`` (SiluMul_v1): I identical units read off the IR graph, each one ``SiluMulBf16`` over
(gate[i], up[i]), lowered piece by piece (``verity_flock.ir_lower``): ``bf16(silu(g))`` is a 16-bit lookup enumerated from
the primitive's own helpers, then an f32 multiply and the torch bf16 cast.

The unit's rows do not depend on I; its ``flock-ir-unit/v2`` netlist, whose LEAVES line places the units on the row's words, is
pinned per I by :data:`PINS` (the frame statement's two-element unit by :data:`FRAME_PINS`)."""
TEMPLATE = "silu-mul"
UNIT = "silu-mul/bf16/element"
PINS = {8192: "a698c50c235f36fbfac74a545158fcc2579a7645433560aa55f53fcf6eadb537",
        9728: "049d020b1a37757a0af483157a7bf31c5e3fbc853fd8e956e00c96971e1f9639",
        14336: "67ae5f764e08b0e4e3dc3e861158eaba8f4e437ea8c8eb9243ed63dc57361758"}
FRAME_PINS = {8192: "973065fe03d854e63476a60f39dceaae530dc0a6f8851118f5c99e3a8baf1067",
              9728: "6d5f25ce6290b113bad320974d69d82f8f66eaec747e8886063e7334a244ded2",
              14336: "8b65ec7a9ef165c9184a0a27b6a84772808b955495fa34e0d12e6a8b86fb7e4d"}


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
    return (f"C-Flock proves {subcircuit.id} under {', '.join(FRAME_SCHEMES)} (verity/flock-ir-frame/v3); binding its rows under "
            f"{scheme} is not wired yet")


def relation(subcircuit, scheme):
    return "irf:silu-mul"


def stage(input_set, lo, hi, out_dir):
    """The frame-v3 instance file for ids [lo, hi) (``verity_flock.ir_frame.stage``): each pod stages its own."""
    from verity_flock import ir_frame

    return ir_frame.stage(frame_lowering(input_set.subcircuit), input_set, lo, hi, out_dir)
