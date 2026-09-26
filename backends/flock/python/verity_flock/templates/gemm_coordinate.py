"""C-Flock lowering of ``gemm-coordinate``: flock-pure's pinned unit netlists (``verity_flock.lowering``) under the
``verity/flock-pure-block/v2`` layouts.

K = 1536 is the frozen sets' K: ``Chunk(3)`` / ``ShaBf16`` for BF16, ``Fp8`` / ``ShaFp8`` for E4M3, ``Fp4`` / ``ShaFp4`` for
NVFP4 (keyed-BLAKE3 / SHA-256 rows). Any other K lowers when a row is whole BLAKE3 chunks, at least two (``Chunk(n)``,
n = K x word bytes / 1024), or n whole chunks and a 512-byte partial one (``ChunkTail(n)``, n + 1 blocks per VU), keyed-BLAKE3
only. ``stage`` writes an input set's range as the instance file
(``instances.write_set``, rows per VU). A proof holds at most 32,768 / n VUs (CN2); larger batches are sub-batches."""
from pathlib import Path

TEMPLATE = "gemm-coordinate"
K = 1536
SCHEMES = {"frame-v3/blake3-keyed": "blake3", "frame-v3/sha-256": "sha256"}
RELATIONS = {"sm80.mma.m16n8k16.bf16": "bf16-ampere", "sm90.mma.m16n8k16.bf16": "bf16-hopper",
             "sm90.wgmma.m64n8k16.bf16": "bf16-hopper-wgmma", "sm89.mma.m16n8k32.e4m3": "fp8-ada",
             "sm90.wgmma.m64n8k32.e4m3": "fp8-hopper", "sm120.mma.m16n8k64.e2m1.nvf4": "fp4-nvf4"}
#: Chunk(n) with a red-team grant (red-team-flock, extended 2026-09-26 10:05Z to every n from 2 to 64); ChunkTail(n) is under review.
GRANTED_CHUNKS = set(range(2, 65))
TAIL = 512
CHUNK = 1024


def row_bytes(subcircuit) -> int:
    sem, k = subcircuit.params["semantics"], int(subcircuit.params["K"])
    return k // 2 + k // 16 if "e2m1" in sem else k * (1 if "e4m3" in sem else 2)


def chunks(subcircuit) -> int | None:
    """``n`` of the ``Chunk(n)`` / ``ChunkTail(n)`` layout a K lowers to, or None at the frozen K (its own layouts)."""
    return None if int(subcircuit.params["K"]) == K else row_bytes(subcircuit) // CHUNK


def tail(subcircuit) -> bool:
    return int(subcircuit.params["K"]) != K and row_bytes(subcircuit) % CHUNK == TAIL


def layout(subcircuit) -> str | None:
    n = chunks(subcircuit)
    return None if n is None else f"ChunkTail({n})" if tail(subcircuit) else f"Chunk({n})"


def max_vus_per_proof(subcircuit) -> int:
    """CN2: m <= 35 (blocks per VU x VUs <= 32,768; K = 1536 layouts by their ``k_log``)."""
    n = chunks(subcircuit)
    if n is not None:
        return 32768 // (n + tail(subcircuit))
    return {"bf16": 10922, "e4m3": 16384, "e2m1": 32768}[subcircuit.params["datatype"]]


def supports(subcircuit, scheme):
    sem = subcircuit.params.get("semantics")
    if sem not in RELATIONS:
        return f"C-Flock lowers the semantics {sorted(RELATIONS)} (verity_flock.lowering.PIPES), not {sem}"
    if scheme not in SCHEMES:
        return f"flock-pure proves keyed-BLAKE3 or SHA-256 row leaves ({sorted(SCHEMES)}), not {scheme}"
    k = int(subcircuit.params["K"])
    if k == K:
        return None
    if "e2m1" in sem:
        return f"the NVFP4 layouts (Fp4 / ShaFp4) are K = {K} rows; K = {k} needs its own layout"
    row = row_bytes(subcircuit)
    if row % CHUNK not in (0, TAIL) or row < 2 * CHUNK and row % CHUNK == 0 or row < CHUNK:
        return (f"flock-pure-block's Chunk(n) / ChunkTail(n) layouts need a row of whole BLAKE3 chunks (at least two) or whole "
                f"chunks and a 512-byte partial one (CN3); K = {k} gives a {row}-byte row ({row / CHUNK:g} chunks)")
    if scheme != "frame-v3/blake3-keyed":
        return f"Chunk(n) rows are keyed-BLAKE3 only; SHA-256 rows exist at K = {K}"
    return None


def granted(subcircuit, scheme) -> str | None:
    """None when a red-team grant covers the layout this statement lowers to, else what is missing (a cell may still run)."""
    n = chunks(subcircuit)
    if n is not None and tail(subcircuit):
        return f"ChunkTail({n}) has no red-team grant yet (red-team-flock is reviewing PR #70)"
    if n is not None and n not in GRANTED_CHUNKS:
        return f"Chunk({n}) is outside red-team-flock's Chunk(n) grant (n from 2 to 64)"
    return None


def relation(subcircuit, scheme):
    return RELATIONS[subcircuit.params["semantics"]]


def leaf(scheme) -> str:
    return SCHEMES[scheme]


def stage(input_set, lo, hi, out_dir, scheme="frame-v3/blake3-keyed") -> Path:
    """VUs ``[lo, hi)`` of ``input_set`` as a ``flock-pure-instances/v1`` file (accumulators re-chained under the relation's
    model and checked against the set's y)."""
    from verity_flock import instances

    rel = RELATIONS[input_set.subcircuit.params["semantics"]]
    out = Path(out_dir) / f"instances-{lo}-{hi}.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    instances.write_set(out, rel, input_set.path, lo, hi, SCHEMES[scheme])
    return out
