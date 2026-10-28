"""Flock lowering of ``gumbel-top-p-token-select`` (GumbelTopPTokenSelect_v1{V}): V identical lane units lowered from the
IR composites of the Definition's three scans (``TemperatureLane``, ``TopPMaskStep``'s select, ``GumbelSelectStep``), the
row-level values (the top-p keep word, the Gumbel noise, the temperature's row scalars) native public cut words, under
``verity/flock-ir-sampling/v1`` (``verity_flock.ir_sampling``).

The lane netlist does not depend on V (only the arrangement does) and is pinned by :data:`PIN`."""
TEMPLATE = "gumbel-top-p-token-select"
UNIT = "gumbel-top-p-token-select/fp32/lane"
PIN = "c9aeca364cb08b68a174fa9811ff505306c6a8f9af7458689ff23e66409cc1b6"
FRAME_SCHEMES = ("frame-v3/blake3-keyed",)


def definition(subcircuit):
    return subcircuit.definition()


def frame_lowering(subcircuit):
    from verity_flock import ir_sampling

    return ir_sampling.lane(int(subcircuit.params["V"]))


def lowering(subcircuit):
    return frame_lowering(subcircuit)


def supports(subcircuit, scheme):
    if scheme in FRAME_SCHEMES:
        return None
    return (f"C-Flock proves {subcircuit.id} under {', '.join(FRAME_SCHEMES)} (verity/flock-ir-sampling/v1); binding its rows under "
            f"{scheme} is not wired yet")


def relation(subcircuit, scheme):
    return "irf:gumbel-top-p-token-select"


def stage(input_set, lo, hi, out_dir):
    """The ``flock-ir-sampling/v1`` instance file for ids [lo, hi) (``verity_flock.ir_sampling.stage``): each pod stages its own."""
    from verity_flock import ir_sampling

    return ir_sampling.stage(frame_lowering(input_set.subcircuit), input_set, lo, hi, out_dir)
