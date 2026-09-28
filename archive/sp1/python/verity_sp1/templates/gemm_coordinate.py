"""D-SP1 lowering of ``gemm-coordinate``: the committed guest (frame-v3 SHA-256 rows, vllm-v1 SHA-256 position leaves) at K = 1536
on the Ampere BF16 step."""
TEMPLATE = "gemm-coordinate"
K = 1536
SEMANTICS = "sm80.mma.m16n8k16.bf16"
SCHEMES = {"frame-v3/sha-256": "frame-v3", "vllm-v1/sha-256": "vllm-v1"}


def supports(subcircuit, scheme):
    if subcircuit.params.get("semantics") != SEMANTICS:
        return f"the SP1 committed guest evaluates {SEMANTICS} (verity.ml.circuits.silicon AMPERE_BF16_M16N8K16), not {subcircuit.params.get('semantics')}"
    if int(subcircuit.params["K"]) != K:
        return f"the SP1 committed guest's statements are built at K = {K}; K = {subcircuit.params['K']} is not"
    if scheme not in SCHEMES:
        return f"the SP1 committed guest opens {sorted(SCHEMES)}, not {scheme}"
    return None


def relation(subcircuit, scheme):
    return f"bf16-ampere+{SCHEMES[scheme]}"
