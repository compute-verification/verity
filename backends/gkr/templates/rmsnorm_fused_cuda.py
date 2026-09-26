"""A-GKR lowering of ``rmsnorm-fused-cuda`` (RMSNormFusedCuda_v2): not written yet."""
TEMPLATE = "rmsnorm-fused-cuda"


def supports(subcircuit, scheme):
    return "A-GKR has no lowering of RMSNormFusedCuda_v2 yet (after C-Flock's gadget library settles)"
