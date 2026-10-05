"""D-SP1 lowering of ``attention-head`` (FA2 AttentionHead_v3): not written yet."""
TEMPLATE = "attention-head"


def supports(subcircuit, scheme):
    return "D-SP1 has no guest for AttentionHead_v3 yet (lane D-SP1: the IR evaluator in the zkVM)"
