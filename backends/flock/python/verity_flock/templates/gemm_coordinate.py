"""C-Flock lowering of ``gemm-coordinate``: flock-pure's pinned unit netlist (``verity_flock.lowering``), at K = 1536 only.

The pod scripts (``pod/30-cell.sh``, ``verity_flock.bench``) take the relation name and regenerate that relation's frozen
instance set themselves (``verity_flock.instances``), so there is no ``stage``: a cell runs on the frozen set only."""
TEMPLATE = "gemm-coordinate"
K = 1536
SCHEME = "frame-v3/blake3-keyed"
RELATIONS = {"sm90.mma.m16n8k16.bf16": "bf16-hopper", "sm89.mma.m16n8k32.e4m3": "fp8-ada", "sm90.wgmma.m64n8k32.e4m3": "fp8-hopper"}


def supports(subcircuit, scheme):
    sem = subcircuit.params.get("semantics")
    if sem not in RELATIONS:
        return f"C-Flock lowers the semantics {sorted(RELATIONS)} (verity_flock.lowering.PIPES), not {sem}"
    if int(subcircuit.params["K"]) != K:
        return (f"flock-pure's unit netlist and instance files are pinned at K = {K} (lowering.PINS, instances.K); "
                f"K = {subcircuit.params['K']} needs its own pinned lowering (lane C-Flock GEMM at real K)")
    if scheme != SCHEME:
        return f"flock-pure proves keyed-BLAKE3 row leaves ({SCHEME}), not {scheme}"
    return None


def relation(subcircuit, scheme):
    return RELATIONS[subcircuit.params["semantics"]]
