"""C-Flock lowering of ``rmsnorm-fused-cuda`` (RMSNormFusedCuda_v2): not written yet."""
TEMPLATE = "rmsnorm-fused-cuda"


def supports(subcircuit, scheme):
    return "C-Flock has no lowering of RMSNormFusedCuda_v2 yet (lane C-Flock generic IR lowering: per-op gadgets, then elementwise ops)"
