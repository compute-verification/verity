"""C-Flock lowering of ``token-select-chunk`` (TokenSelectChunk_v2{C}): one V-chunk of the registered compiler's greedy select
(``dead_free.TokenSelect_v4``) as ONE unit (``ir_lower.whole``: the chunk's scan carries its best from lane to lane, so its gate
graph is no row of identical units), lowered piece by piece from the word Definition. At C = 64 its input row is 64 bf16 logits,
one SHA-512 block (1,024 bits).

The unit is the whole instance, so its netlist depends on C and is pinned per C by :data:`PINS`."""
TEMPLATE = "token-select-chunk"
UNIT = "token-select-chunk/v2/chunk"
PINS = {64: "802a2f1ac3709b52ce0930508777e4400eb2d53ec62b1084efd525ffb4cb196c"}


def definition(subcircuit):
    return subcircuit.definition()


def lowering(subcircuit):
    from verity_flock import ir_lower

    return ir_lower.lowering(definition(subcircuit), UNIT, unit_fn=ir_lower.whole)


FRAME_SCHEMES = ("frame-v3/blake3-keyed",)


def frame_lowering(subcircuit):
    return lowering(subcircuit)


def supports(subcircuit, scheme):
    if scheme in FRAME_SCHEMES:
        return None
    return (f"C-Flock proves {subcircuit.id} under {', '.join(FRAME_SCHEMES)} (verity/flock-ir-frame/v3); binding its rows under "
            f"{scheme} is not wired yet")


def relation(subcircuit, scheme):
    return "irf:token-select-chunk"


def stage(input_set, lo, hi, out_dir):
    """The frame-v3 instance file for ids [lo, hi) (``verity_flock.ir_frame.stage``): each pod stages its own."""
    from verity_flock import ir_frame

    return ir_frame.stage(frame_lowering(input_set.subcircuit), input_set, lo, hi, out_dir)
