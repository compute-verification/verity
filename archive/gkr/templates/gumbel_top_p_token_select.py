"""A-GKR lowering of ``gumbel-top-p-token-select`` (GumbelTopPTokenSelect_v1): not written yet."""
TEMPLATE = "gumbel-top-p-token-select"


def supports(subcircuit, scheme):
    return "A-GKR has no lowering of GumbelTopPTokenSelect_v1 yet (after C-Flock's gadget library settles)"
