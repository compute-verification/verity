"""B-Ligero lowering of ``attention-head`` (FA2 AttentionHead_v3): not written yet."""
TEMPLATE = "attention-head"


def supports(subcircuit, scheme):
    return "B-Ligero has no lowering of AttentionHead_v3 yet (after C-Flock's gadget library settles)"
