"""C-Flock lowering of ``rmsnorm-triton`` (RMSNormTriton_v1): not written yet."""
TEMPLATE = "rmsnorm-triton"


def supports(subcircuit, scheme):
    return "C-Flock has no lowering of RMSNormTriton_v1 yet (lane C-Flock generic IR lowering: per-op gadgets, then elementwise ops)"
