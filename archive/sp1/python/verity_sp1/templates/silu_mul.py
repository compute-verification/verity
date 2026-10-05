"""D-SP1 lowering of ``silu-mul`` (SiluMul_v1): not written yet."""
TEMPLATE = "silu-mul"


def supports(subcircuit, scheme):
    return "D-SP1 has no guest for SiluMul_v1 yet (lane D-SP1: the IR evaluator in the zkVM)"
