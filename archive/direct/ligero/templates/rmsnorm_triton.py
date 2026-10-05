"""B-Ligero lowering of ``rmsnorm-triton`` (RMSNormTriton_v1): not written yet."""
TEMPLATE = "rmsnorm-triton"


def supports(subcircuit, scheme):
    return "B-Ligero has no lowering of RMSNormTriton_v1 yet (after C-Flock's gadget library settles)"
