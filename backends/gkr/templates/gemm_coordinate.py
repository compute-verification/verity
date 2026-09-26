"""A-GKR lowering of ``gemm-coordinate``: only route (a) binds the operands (``tools/cell.py``: A-GKR with the sigma link to
Flock's keyed-BLAKE3 row leaves, on ``bf16-ampere+blake3``); A-GKR alone keeps x and W private and bound to nothing.

At vLLM's reduction lengths (``REAL_KS``) the statement is ``bf16-ampere-k<K>+blake3``: the same circuit files with ``steps`` =
K / 16 (``verifier/pins.txt``), and Flock's row of 2K bytes is K / 512 whole BLAKE3 chunks.  ``stage`` writes a range of an input
set, which the cell reads as it is (``tools/cell.py statement --input-set``)."""
TEMPLATE = "gemm-coordinate"
K = 1536
REAL_KS = (2048, 8192)
SEMANTICS = "sm80.mma.m16n8k16.bf16"
SCHEME = "frame-v3/blake3-keyed"


def supports(subcircuit, scheme):
    if subcircuit.params.get("semantics") != SEMANTICS:
        return f"A-GKR's linked statement is {SEMANTICS} (tools/cell.py REL bf16-ampere), not {subcircuit.params.get('semantics')}"
    k = int(subcircuit.params["K"])
    if k != K and k not in REAL_KS:
        return f"A-GKR's linked circuits are pinned at K = {K} and {list(REAL_KS)} (verifier/pins.txt); K = {k} is not"
    if scheme != SCHEME:
        return f"A-GKR binds operands only through route (a)'s link to Flock's {SCHEME} leaves, not {scheme}"
    return None


def relation(subcircuit, scheme):
    k = int(subcircuit.params["K"])
    return "bf16-ampere+blake3" if k == K else f"bf16-ampere-k{k}+blake3"


def stage(input_set, lo, hi, out_dir):
    """Ids ``[lo, hi)`` of an input set as a set of their own under ``out_dir`` (``input_sets.subset``)."""
    from pathlib import Path

    from verity_numerical.bench.input_sets import subset
    if lo == 0 and hi == input_set.n:
        return Path(input_set.path)
    return subset(input_set.path, out_dir, lo, hi)
