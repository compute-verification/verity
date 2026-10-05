"""B-Ligero lowering of ``silu-mul`` (SiluMul_v1): not written yet."""
TEMPLATE = "silu-mul"


def supports(subcircuit, scheme):
    return "B-Ligero has no lowering of SiluMul_v1 yet (after C-Flock's gadget library settles)"
