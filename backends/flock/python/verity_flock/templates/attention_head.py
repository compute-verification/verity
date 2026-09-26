"""C-Flock lowering of ``attention-head`` (FA2 AttentionHead_v3): not written yet."""
TEMPLATE = "attention-head"


def supports(subcircuit, scheme):
    return "C-Flock has no lowering of AttentionHead_v3 yet (lane C-Flock generic IR lowering: after the elementwise ops)"
