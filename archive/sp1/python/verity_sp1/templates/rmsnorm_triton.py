"""D-SP1 lowering of ``rmsnorm-triton`` (RMSNormTriton_v1): not written yet."""
TEMPLATE = "rmsnorm-triton"


def supports(subcircuit, scheme):
    return "D-SP1 has no guest for RMSNormTriton_v1 yet (lane D-SP1: the IR evaluator in the zkVM, starting with a one-input RMSNorm spike)"
