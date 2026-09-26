"""C-Flock lowering of ``gemm-coordinate``: flock-pure's pinned unit netlists (``verity_flock.lowering``) under the
``verity/flock-pure-block/v2`` layouts.

K = 1536 is the frozen sets' K (``Chunk(3)`` for BF16, ``Fp8`` for E4M3). Any other K lowers when a row is whole BLAKE3
chunks, at least two (``Chunk(n)``, n = K x word bytes / 1024: K = 2048 and 8192 today); ``stage`` writes an input set's
range as the instance file (``instances.write_set``, rows per VU). A proof holds at most 32,768 / n VUs (CN2); larger
batches are sub-batches."""
from pathlib import Path

TEMPLATE = "gemm-coordinate"
K = 1536
SCHEME = "frame-v3/blake3-keyed"
RELATIONS = {"sm80.mma.m16n8k16.bf16": "bf16-ampere", "sm90.mma.m16n8k16.bf16": "bf16-hopper",
             "sm90.wgmma.m64n8k16.bf16": "bf16-hopper-wgmma", "sm89.mma.m16n8k32.e4m3": "fp8-ada",
             "sm90.wgmma.m64n8k32.e4m3": "fp8-hopper"}
CHUNK = 1024


def chunks(subcircuit) -> int | None:
    """``n`` of the ``Chunk(n)`` layout a K lowers to, or None at the frozen K (its own layouts)."""
    k = int(subcircuit.params["K"])
    if k == K:
        return None
    return k * (1 if "e4m3" in subcircuit.params["semantics"] else 2) // CHUNK


def supports(subcircuit, scheme):
    sem = subcircuit.params.get("semantics")
    if sem not in RELATIONS:
        return f"C-Flock lowers the semantics {sorted(RELATIONS)} (verity_flock.lowering.PIPES), not {sem}"
    k = int(subcircuit.params["K"])
    if k != K:
        row = k * (1 if "e4m3" in sem else 2)
        if row % CHUNK or row // CHUNK < 2:
            return (f"flock-pure-block's Chunk(n) layout needs a row of whole BLAKE3 chunks, at least two (CN3); K = {k} "
                    f"gives a {row}-byte row")
    if scheme != SCHEME:
        return f"flock-pure proves keyed-BLAKE3 row leaves ({SCHEME}), not {scheme}"
    return None


def relation(subcircuit, scheme):
    return RELATIONS[subcircuit.params["semantics"]]


def stage(input_set, lo, hi, out_dir) -> Path:
    """VUs ``[lo, hi)`` of ``input_set`` as a ``flock-pure-instances/v1`` file (accumulators re-chained under the relation's
    model and checked against the set's y)."""
    from verity_flock import instances

    rel = RELATIONS[input_set.subcircuit.params["semantics"]]
    out = Path(out_dir) / f"instances-{lo}-{hi}.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    instances.write_set(out, rel, input_set.path, lo, hi)
    return out
