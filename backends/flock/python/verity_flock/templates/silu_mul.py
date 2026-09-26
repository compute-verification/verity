"""C-Flock lowering of ``silu-mul`` (SiluMul_v1): not written yet."""
TEMPLATE = "silu-mul"


def supports(subcircuit, scheme):
    return "C-Flock has no lowering of SiluMul_v1 yet (lane C-Flock generic IR lowering: per-op gadgets, then elementwise ops)"
