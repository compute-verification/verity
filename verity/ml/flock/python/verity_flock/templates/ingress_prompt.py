"""C-Flock lowering of ``ingress-prompt`` (IngressPrompt_v1{MODEL,LP,V}): the I/O shell's ingress parse as ONE unit
(``ir_lower.whole``: the parse's fields read different bytes of one body, so its gate graph is no row of identical units), lowered
piece by piece from the word Definition. Its input row is the body, whole SHA-512 blocks of bytes (ingress v0 pads to them).

The unit is the whole instance, so its netlist depends on every static and is pinned per (MODEL, LP, V) by :data:`PINS`."""
TEMPLATE = "ingress-prompt"
UNIT = "ingress-prompt/v0/body"
PINS = {("m", 3, 1000): "14c0b236f6fafca82aab4142bbb36f2fbe74a725c95a4774059c3c2b35259395",
        ("served-model", 16, 128256): "26b26dda22045fb57d027b370bf0790f89239e1dacf8c9de9ef84d34a22e9f81"}


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
    return "irf:ingress-prompt"


def stage(input_set, lo, hi, out_dir):
    """The frame-v3 instance file for ids [lo, hi) (``verity_flock.ir_frame.stage``): each pod stages its own."""
    from verity_flock import ir_frame

    return ir_frame.stage(frame_lowering(input_set.subcircuit), input_set, lo, hi, out_dir)
