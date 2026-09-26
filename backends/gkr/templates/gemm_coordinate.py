"""A-GKR lowering of ``gemm-coordinate``: only route (a) binds the operands (``tools/cell.py``: A-GKR with the sigma link to
Flock's keyed-BLAKE3 row leaves, on ``bf16-ampere+blake3``); A-GKR alone keeps x and W private and bound to nothing."""
TEMPLATE = "gemm-coordinate"
K = 1536
SEMANTICS = "sm80.mma.m16n8k16.bf16"
SCHEME = "frame-v3/blake3-keyed"


def supports(subcircuit, scheme):
    if subcircuit.params.get("semantics") != SEMANTICS:
        return f"A-GKR's linked statement is {SEMANTICS} (tools/cell.py REL bf16-ampere), not {subcircuit.params.get('semantics')}"
    if int(subcircuit.params["K"]) != K:
        return f"A-GKR's circuits are exported at K = {K} (gpu.v2.export); K = {subcircuit.params['K']} is not"
    if scheme != SCHEME:
        return f"A-GKR binds operands only through route (a)'s link to Flock's {SCHEME} leaves, not {scheme}"
    return None


def relation(subcircuit, scheme):
    return "bf16-ampere+blake3"
