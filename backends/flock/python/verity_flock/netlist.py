"""``verity/flock-netlist/v1``: any lowered IR template (``flock-ir-unit/v2`` netlists, :mod:`ir_lower`) proved with its
frame-v3 rows hashed in the circuit and nothing else public (``live/src/netlist.rs`` is the statement).

Private by default. The public data are the frame-v3 roots, the row digests they are built from (the serving tree's
leaves), the declared outputs and the layout below. Every other word (every compression's chaining value, every chunk
value, every word a unit, a tail stage or a lookup reads from another) stays inside the witness, joined by copy
constraints (Δ) inside a block: no public intermediate word and no verifier-evaluated tail on private data.

One or more whole instances (VUs) per block. A block is the same for every block (Flock's ``A = I ⊗ A_0 + Δ``) and holds
aligned ranges of slots of a few types:

* ``comp``: BLAKE3 compressions (2^14 bits), a range per input port for its chunk runs (a run = one chunk's blocks) and, for a
  port of several chunks, a range for its parent-tree compressions (heap order, the root first), so a row's digest is the
  output of one compression and every chunk value stays private;
* ``unit``: the template's identical units (2^unit_log bits);
* ``mask``: 2^14 free cells (``A = B = I``), reserved for zero-knowledge masks and filled by the prover's mask hook (zeros in
  M0), apart from the forced-zero padding (positions no range covers).

A padding instance (the block's unused instances and every block past the last) is the all-zero instance; its digests and
outputs are pinned here (``dummy``), so the verifier knows every public value of every block.

The composite netlist (``flock-netlist/v1``) is the statement's meaning: a ``META`` line (ports, ranges, the compression
slots' roles, the units' leaf and output maps, the pinned dummy values, the leaf-commitment scheme), then each slot type's
``flock-ir-unit/v2`` netlist. Its sha256 is the pin. The verifier derives Δ, the public regions and their values from it
and from its own public instance file (no rows), never from the prover's.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import ir_lower as IL

FORMAT = "flock-netlist/v1"
STATEMENT = "verity/flock-netlist/v1"
SCHEME = "frame-v3/blake3-keyed"
INST_FORMAT = "flock-netlist-inst/v1"
#: The leaf-commitment scheme of the proof's own Merkle trees (Ligerito levels), pinned in the statement and the backend
#: identity; the reserved sizes are what Halevi–Micali (HM96, the O(k) variant as in B-Ligero's leaves) needs, so it drops in
#: without changing the layout: randomness x of 6k + 4 bits at k = 160 (121 bytes), a universal hash key of (|x| + 256) bits.
LEAF_SCHEME = {"id": "flock-leaf/salted-sha256/v1", "salt_bytes": 32, "reserved_salt_bytes": 128, "reserved_key_bytes": 160}
CHUNK = 1024
BLOCK = 64
SUB_LOG = 14
MASK_LOG = 14
K_MAX = 26

# the BLAKE3 compression slot's columns (flock_prover::r1cs_hashes::blake3)
CV_BASE, OUT_LO_BASE, M_BASE, PARAMS_BASE, Z_CONST = 0, 256, 512, 1024, 11706
CHUNK_START, CHUNK_END, PARENT, ROOT, KEYED_HASH = 1, 2, 4, 8, 16


def _pow2(x: int) -> int:
    return 1 << max(0, (x - 1).bit_length())


def _log2(x: int) -> int:
    assert x & (x - 1) == 0 and x > 0, x
    return x.bit_length() - 1


@dataclass
class Port:
    name: str
    words: int
    chunks: int
    nb: int          # blocks per chunk (a power of two; one short chunk allowed)

    @property
    def bytes(self) -> int:
        return 2 * self.words


def _port(name: str, words: int) -> Port:
    b = 2 * words
    chunks = -(-b // CHUNK)
    nb = min(b, CHUNK) // BLOCK
    if b % BLOCK or nb & (nb - 1) or (chunks > 1 and (b % CHUNK or chunks & (chunks - 1))):
        raise ValueError(f"port {name}: {b}-byte rows are not whole chunks of a power-of-two count (or one power-of-two short chunk)")
    return Port(name, words, chunks, nb)


def _ranges(items: list[tuple[str, int, int, dict]]) -> tuple[list[dict], int]:
    """Place ``(type, slot_log, count, extra)`` ranges, largest first, each aligned to its own power-of-two span."""
    order = sorted(range(len(items)), key=lambda i: (-(_pow2(items[i][2]) << items[i][1]), i))
    at, out = 0, [None] * len(items)
    for i in order:
        typ, sl, n, extra = items[i]
        span = _pow2(n) << sl
        at = -(-at // span) * span
        out[i] = {"type": typ, "slot_log": sl, "pos0": at >> sl, "count": n, **extra}
        at += span
    return out, max(20, (at - 1).bit_length())


def compose(low: IL.Lowering, sub, vus_per_block: int | None = None) -> str:
    """The ``flock-netlist/v1`` text for a template instance lowered to identical units (no cut yet: every unit input is an
    input leaf or a zero word, every output leaf a unit's returned word)."""
    U = low.units
    if U.cut or U.tail_rets:
        raise NotImplementedError(f"{low.name}: cut templates (a tail in the circuit) are not composed yet")
    if any(p.dtype != "u16" for p in (*sub.inputs, *sub.outputs)):
        raise ValueError(f"{low.name}: frame-v3 rows here are u16 words")
    lv = low.leaves
    ports = [_port(p.name, w) for p, (w, _) in zip(sub.inputs, lv["in_ports"])]
    outs = [[p.name, w] for p, (w, _) in zip(sub.outputs, lv["out_ports"])]
    ul = max(7, (low.layout.useful - 1).bit_length())
    if low.layout.useful == 1 << ul:
        raise ValueError(f"{low.name}: a unit fills its slot; a zero leaf needs a forced-zero row")

    def fits(G: int):
        items = []
        for pi, p in enumerate(ports):
            items.append(("comp", SUB_LOG, _pow2(G * p.chunks) * p.nb, {"role": "runs", "port": pi}))
            if p.chunks > 1:
                items.append(("comp", SUB_LOG, G * p.chunks, {"role": "parents", "port": pi}))
        items.append(("unit", ul, G * U.n, {"net": "unit"}))
        items.append(("mask", MASK_LOG, 1, {"net": "mask"}))
        return _ranges(items)

    if vus_per_block is None:
        G = 1
        while True:
            _, k = fits(2 * G)
            if k > 20 or 2 * G * U.n > 4096:
                break
            G *= 2
    else:
        G = vus_per_block
    ranges, k_log = fits(G)
    if k_log > K_MAX:
        raise ValueError(f"{low.name}: an instance needs a block of 2^{k_log} bits (at most 2^{K_MAX})")
    for r in ranges:
        if r["type"] == "comp" and r["role"] == "runs":
            p = ports[r["port"]]
            runs = r["count"] // p.nb
            r["slots"] = [[g, c] if g < G else None for g, c in ([divmod(q, p.chunks) for q in range(runs)])]
    dummy = _dummy(low, sub, ports)
    meta = {"format": FORMAT, "statement": STATEMENT, "scheme": SCHEME, "template": sub.template_name if hasattr(sub, "template_name") else None,
            "subcircuit": sub.id, "unit": low.name, "unit_sha256": low.sha256, "vus_per_block": G, "k_log": k_log,
            "ports": [[p.name, p.words, p.chunks, p.nb] for p in ports], "out_ports": outs, "ranges": ranges,
            "units_per_vu": U.n, "unit_log": ul, "leaves_in": lv["in"], "leaves_out": lv["out"], "row_key": _row_key().hex(),
            "dummy": dummy, "leaf_scheme": LEAF_SCHEME, "mask_log": MASK_LOG}
    head = f"{FORMAT} {low.name}\nMETA {json.dumps(meta, sort_keys=True, separators=(',', ':'))}\n"
    unit_text = IL.netlist(low.layout, low.name, None, None)
    return head + f"NET unit {unit_text.count(chr(10))}\n" + unit_text


def _row_key() -> bytes:
    from verity.commitments.rowleaf import ROLE_X, blake3_row_key

    return blake3_row_key(ROLE_X)


def _dummy(low: IL.Lowering, sub, ports: list[Port]) -> dict:
    """The all-zero instance's row digests (per port) and output words (every output leaf), as the padding instances hold."""
    import blake3

    key = _row_key()
    n_in = sum(p.words for p in ports)
    outs, _, ok = low.evaluate(np.zeros((1, n_in), dtype=np.uint64))
    assert int(ok) & 1, "the all-zero instance does not satisfy the unit rows"
    return {"digests": [blake3.blake3(bytes(p.bytes), key=key).hexdigest() for p in ports], "outputs": [int(x) for x in outs[0]]}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --- instance files -----------------------------------------------------------------------------------------------------------

def write(text: str, input_set, lo: int, hi: int, path, public_only: bool) -> dict:
    """``flock-netlist-inst/v1``: one JSON header line (the set, the range, the composite netlist's sha256, the frame-v3
    statement: roots, domains, bindings), then, for the prover only, every instance's input rows (ports in order, u16 LE),
    then every instance's row digests (32 bytes per port) and output words (u16 LE, output leaves in order). The verifier's
    own file (``public_only``) has no rows: it never holds a private word."""
    from . import ir_frame

    meta = json.loads(text.split("\n", 2)[1][5:])
    sub = input_set.subcircuit
    n = hi - lo
    rows = {p.name: np.asarray(input_set.port(p.name, lo, hi), dtype=np.uint16).reshape(n, -1) for p in sub.inputs}
    outs = {p.name: np.asarray(input_set.port(p.name, lo, hi), dtype=np.uint16).reshape(n, -1) for p in sub.outputs}

    class _Plan:
        in_ports = [(nm, w) for nm, w, _, _ in meta["ports"]]
        out_ports = [tuple(x) for x in meta["out_ports"]]

    fv, digests = ir_frame.statement(_Plan, input_set, lo, hi, rows, outs)
    head = {"format": INST_FORMAT, "statement": STATEMENT, "scheme": SCHEME, "set": input_set.name, "subcircuit": sub.id,
            "content_digest": input_set.content_digest, "range": [lo, hi], "instances": n, "netlist_sha256": digest(text),
            "frame_v3": fv, "rows": not public_only}
    body = b""
    if not public_only:
        body += b"".join(np.concatenate([rows[nm] for nm, _ in _Plan.in_ports], axis=1).astype("<u2")[i].tobytes() for i in range(n))
    body += b"".join(digests[nm][i].tobytes() for i in range(n) for nm, _ in _Plan.in_ports)
    body += np.concatenate([outs[nm] for nm, _ in _Plan.out_ports], axis=1).astype("<u2").tobytes()
    with open(path, "wb") as f:
        f.write(json.dumps(head, sort_keys=True).encode() + b"\n")
        f.write(body)
    return head


def stage(low: IL.Lowering, input_set, lo: int, hi: int, out_dir, public_only: bool = False) -> Path:
    """``<out_dir>/net.txt`` (the composite netlist) and ``<out_dir>/inst-<n>.bin`` (``pub-<n>.bin`` for the verifier)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = compose(low, input_set.subcircuit)
    (out / "net.txt").write_text(text)
    path = out / f"{'pub' if public_only else 'inst'}-{hi - lo}.bin"
    write(text, input_set, lo, hi, path, public_only)
    return path
