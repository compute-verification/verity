"""C-Flock lowering of ``silu-mul`` (SiluMul_v1): I identical units read off the IR graph, each one ``SiluMulBf16`` over
(gate[i], up[i]), lowered piece by piece (``verity_flock.ir_lower``): ``bf16(silu(g))`` is a 16-bit lookup enumerated from
the primitive's own helpers, then an f32 multiply and the torch bf16 cast.

The unit's rows do not depend on I; its ``flock-ir-unit/v2`` netlist, whose LEAVES line places the units on the row's words, is
pinned per I by :data:`PINS` (the frame statement's two-element unit by :data:`FRAME_PINS`)."""
TEMPLATE = "silu-mul"
UNIT = "silu-mul/bf16/element"
PINS = {8192: "be5a090be33ef6ad7d8cc0242610d038c87adb11c93f4ba1328f19263a42f25b",
        9728: "1747e5c5bd676392db03ec383fc3acf6655b6066619f748d26b6380e25062e0d",
        14336: "20e14a658b21fff2e3dd001ccb64ec9ca67ff5e4e7a28b9ca311011404a15726"}
FRAME_PINS = {8192: "823415f4d07973827c800d8bb64baedaeab62ca240ff8a5569f28a1beea12282",
              9728: "5b903f64ceeb414a4528ab36ceafde474afc2a9a0594ac2fb9fd0bb71bf19ea9",
              14336: "8291dc1369457102112b6fe221f245544e819411e16bf160ae3f04651d257a05"}


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
