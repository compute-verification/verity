"""C-Flock lowering of ``token-select-combine`` (TokenSelectCombine_v2{V,C}): the registered compiler's greedy token from
logits[0] and the row's V-chunk partials (``dead_free.TokenSelect_v4``'s last call) as ONE unit (``ir_lower.whole``: the combine
carries its running best across chunks, so its gate graph is no row of identical units), lowered piece by piece from the word
Definition. Its output is the egress id, one 32-bit word.

The unit is the whole instance, so its netlist depends on every static and is pinned per (V, C) by :data:`PINS`."""
TEMPLATE = "token-select-combine"
UNIT = "token-select-combine/v2/row"
PINS = {(200, 64): "d36c1846d96f0a707aa5fa696f305096cba2bd743cb875758f00c7171e9ec6d3",
        (128256, 64): "b0c99cf74ec9aa32d0cb35c711dd6f1d205373ef906f7a1b4f2b3aef7078227d"}


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
    return "irf:token-select-combine"


def stage(input_set, lo, hi, out_dir):
    """The frame-v3 instance file for ids [lo, hi) (``verity_flock.ir_frame.stage``): each pod stages its own."""
    from verity_flock import ir_frame

    return ir_frame.stage(frame_lowering(input_set.subcircuit), input_set, lo, hi, out_dir)
