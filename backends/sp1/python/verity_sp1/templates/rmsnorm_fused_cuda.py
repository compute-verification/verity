"""D-SP1 lowering of ``rmsnorm-fused-cuda`` (RMSNormFusedCuda_v2): not written yet."""
TEMPLATE = "rmsnorm-fused-cuda"


def supports(subcircuit, scheme):
    return "D-SP1 has no guest for RMSNormFusedCuda_v2 yet (lane D-SP1: the IR evaluator in the zkVM, starting with a one-input RMSNorm spike)"
