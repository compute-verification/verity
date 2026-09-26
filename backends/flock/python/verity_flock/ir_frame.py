"""frame-v3 binding of a lowered IR template: the statement ``verity/flock-ir-frame/v1`` (``live/src/ir_frame.rs``).

Every input port of every instance is one frame-v3 row leaf, ``blake3-keyed/row/v2`` under the x-row key, one tree per port
under its own domain; every output word is one ``u16`` word leaf, one tree per output port: the GEMM cells' x, W and y. The
rows stay private: the proof hashes them with Flock's BLAKE3 compression circuit, and each unit's input bits are copies of
the message bits they hash. The outputs are public, as GEMM's y is.

Layout. A unit reads bytes of some 1024-byte BLAKE3 chunks of its instance's rows; units and the chunks they read form
components, which must all have one shape. A block holds ``G`` components: their chunks' compressions at 2^14-bit slots
``0 .. comp_slots`` (``nb`` per chunk, a power of two, chunk slots a power of two), then their units at 2^unit_log slots
``unit_pos0 ..`` (``unit_pos0`` a multiple of the unit count, a power of two), so the counters, the chunk values and the unit
outputs are each one aligned subcube. Each chunk's chaining value (for a one-chunk row, its ROOT output, the row digest) is
a prover-committed public; the verifier folds every row's chunk values into its digest natively and checks the digests and
the output words against the frame-v3 roots it recomputes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import ir_lower as IL

FORMAT = "flock-ir-frame/v1"
STATEMENT = "verity/flock-ir-frame/v1"
SCHEME = "frame-v3/blake3-keyed"
BINDING_TAG = "verity/flock-ir-frame/binding/v1"
CHUNK = 1024
BLOCK = 64
SUB_LOG = 14
MAX_SLOTS = 256
K_MIN, K_MAX = 20, 22
OWNER = -1


def _pow2(x: int) -> bool:
    return x > 0 and x & (x - 1) == 0


@dataclass
class Plan:
    """How a template's instances fill blocks. ``comps[c]`` = (its chunks as (input port, chunk index, bytes), its
    instance-local units); ``wiring[k][j]`` = (local chunk slot, byte offset in the chunk) of leaf input port j of a component's
    unit k (the same for every component)."""

    low: IL.Lowering
    in_ports: list[tuple[str, int]]
    out_ports: list[tuple[str, int]]
    comps: list[tuple[list[tuple[int, int, int]], list[int]]]
    wiring: list[list[tuple[int, int]]]
    G: int
    nb: int
    comp_slots: int
    unit_log: int
    unit_pos0: int
    units_per_block: int
    k_log: int

    @property
    def chunks_per_comp(self) -> int:
        return len(self.comps[0][0])

    @property
    def units_per_comp(self) -> int:
        return len(self.comps[0][1])

    def blocks(self, n: int) -> int:
        return -(-n * len(self.comps) // self.G)


def plan(low: IL.Lowering, sub, k_max: int = K_MAX) -> Plan:
    if any(p.dtype != "u16" for p in (*sub.inputs, *sub.outputs)):
        raise ValueError(f"{low.name}: frame-v3 rows here are u16 words")
    in_ports = [(p.name, p.words) for p in sub.inputs]
    out_ports = [(p.name, p.words) for p in sub.outputs]
    starts = np.cumsum([0] + [w for _, w in in_ports])
    row_bytes = [2 * w for _, w in in_ports]

    def chunk_of(pos: int) -> tuple[int, int, int]:
        p = int(np.searchsorted(starts, pos, side="right") - 1)
        byte = 2 * (pos - int(starts[p]))
        return p, byte // CHUNK, byte % CHUNK

    parent: dict = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    U = low.units
    reads = []
    for u in range(U.n):
        cs = [chunk_of(p) for k, p in U.in_src[u] if k == "leaf"]
        reads.append(cs)
        find(("u", u))
        for p, c, _ in cs:
            parent[find(("c", p, c))] = find(("u", u))
    groups: dict = {}
    for u in range(U.n):
        groups.setdefault(find(("u", u)), []).append(u)
    comps, shapes = [], []
    for units in sorted(groups.values(), key=min):
        chunks = sorted({(p, c) for u in units for p, c, _ in reads[u]})
        slot = {pc: i for i, pc in enumerate(chunks)}
        sizes = tuple(min(CHUNK, row_bytes[p] - CHUNK * c) for p, c in chunks)
        wiring = tuple(tuple((slot[(p, c)], off) for p, c, off in reads[u]) for u in units)
        shapes.append((tuple(p for p, _ in chunks), tuple(row_bytes[p] <= CHUNK for p, _ in chunks), sizes, wiring))
        comps.append(([(p, c, s) for (p, c), s in zip(chunks, sizes)], units))
    if any(s != shapes[0] for s in shapes):
        raise ValueError(f"{low.name}: its unit-and-chunk components are not all one shape ({len(set(shapes))} shapes)")
    sizes = set(shapes[0][2])
    if len(sizes) != 1 or next(iter(sizes)) % BLOCK or not _pow2(next(iter(sizes)) // BLOCK):
        raise ValueError(f"{low.name}: chunks of {sorted(sizes)} bytes (need one size, a power-of-two multiple of {BLOCK})")
    nb = next(iter(sizes)) // BLOCK
    useful = low.layout.useful
    ul = max(7, (useful - 1).bit_length())
    cpc, upc = len(comps[0][0]), len(comps[0][1])
    best = None
    for G in range(1, MAX_SLOTS + 1):
        Q, units = G * cpc, G * upc
        C = Q * nb
        if C > MAX_SLOTS or units > MAX_SLOTS:
            break
        if not (_pow2(Q) and _pow2(units)):
            continue
        pos0 = -(-(C << SUB_LOG) // (1 << ul))
        pos0 = -(-pos0 // units) * units
        k_log = max(K_MIN, (((pos0 + units) << ul) - 1).bit_length())
        if k_log > k_max:
            continue
        fill = ((C << SUB_LOG) + units * useful) / (1 << k_log)
        if best is None or fill > best[0] + 1e-9:
            best = (fill, G, C, pos0, units, k_log)
    if best is None:
        raise ValueError(f"{low.name}: no block of at most 2^{k_max} bits holds whole components ({cpc} chunks x {nb} compressions, "
                         f"{upc} units of 2^{ul}) with power-of-two chunk and unit counts")
    _, G, C, pos0, units, k_log = best
    return Plan(low, in_ports, out_ports, comps, [list(w) for w in shapes[0][3]], G, nb, C, ul, pos0, units, k_log)


# --- frame-v3 statement commitments ------------------------------------------------------------------------------------------

def statement(pl: Plan, input_set, lo: int, hi: int, rows: dict[str, np.ndarray], outs: dict[str, np.ndarray]):
    """The frame-v3 roots over instances [lo, hi) of ``input_set``: a row-leaf tree per input port, a word-leaf tree per output
    port (instance-major words), each under a domain bound to the set, the range, the port and its schema. Returns the
    header's ``frame_v3`` entry and the row digests (port -> (n, 32) uint8)."""
    import blake3
    from verity.commitments.frame_v3 import FrameV3, Port, RowLeaf, WordLeaf
    from verity.commitments.identity import identity_digest
    from verity.commitments.indexed import RangeIndexedDomain
    from verity.commitments.merkle import CommitmentDomain
    from verity.commitments.rowleaf import ROLE_X, SCHEMA_BLAKE3_ROW, blake3_row_key

    n = hi - lo
    schemas = {**{nm: SCHEMA_BLAKE3_ROW for nm, _ in pl.in_ports}, **{nm: "u16" for nm, _ in pl.out_ports}}
    counts = {**{nm: n for nm, _ in pl.in_ports}, **{nm: n * w for nm, w in pl.out_ports}}
    bind, doms = {}, {}
    for name in schemas:
        bind[name] = bytes.fromhex(identity_digest(BINDING_TAG, {"set": input_set.name, "content_digest": input_set.content_digest,
                                                                 "lo": int(lo), "hi": int(hi), "port": name, "schema": schemas[name]}))
        doms[name] = CommitmentDomain(bind[name], OWNER, RangeIndexedDomain(0, counts[name]))
    fv = FrameV3({**{nm: Port(doms[nm], RowLeaf("blake3", ROLE_X, 16, w)) for nm, w in pl.in_ports},
                  **{nm: Port(doms[nm], WordLeaf("u16", 2)) for nm, w in pl.out_ports}})
    key = blake3_row_key(ROLE_X)
    digests, roots = {}, {}
    for name, _ in pl.in_ports:
        r = np.ascontiguousarray(rows[name], dtype="<u2")
        digests[name] = np.frombuffer(b"".join(blake3.blake3(r[i].tobytes(), key=key).digest() for i in range(n)),
                                      dtype=np.uint8).reshape(n, 32)
        roots[name] = fv.root(name, [fv.tree_leaf(name, i, digests[name][i].tobytes()) for i in range(n)]).hex()
    for name, _ in pl.out_ports:
        flat = np.asarray(outs[name], dtype=np.uint64).reshape(-1)
        roots[name] = fv.root(name, [fv.tree_leaf(name, r, fv.leaf_layout(name, r).digest(int(v).to_bytes(2, "big")))
                                     for r, v in enumerate(flat.tolist())]).hex()
    return {"roots": roots, "domain_ids": {k: d.domain_id.hex() for k, d in doms.items()}, "bindings": {k: b.hex() for k, b in bind.items()},
            "owner": OWNER, "schemas": schemas, "key": key.hex()}, digests


# --- the instance file -------------------------------------------------------------------------------------------------------

def write(pl: Plan, input_set, lo: int, hi: int, path, cuts: np.ndarray | None = None) -> dict:
    """``flock-ir-frame/v1``: one JSON header line (the plan, every block's chunks and units, the frame-v3 statement), then every
    instance's input rows (ports in order, words little-endian), every instance's row digests (32 bytes per input port), every
    unit's returned-output words (``ret_words`` x 16 bytes, units instance-major) and, for a cut template, every instance's cut
    words (u32 LE; the prover's: the verifier takes them from the proof's publics)."""
    low, sub, n = pl.low, input_set.subcircuit, hi - lo
    U, lay = low.units, low.layout
    rows = {p.name: np.asarray(input_set.port(p.name, lo, hi), dtype=np.uint16).reshape(n, -1) for p in sub.inputs}
    outs = {p.name: np.asarray(input_set.port(p.name, lo, hi), dtype=np.uint16).reshape(n, -1) for p in sub.outputs}
    fv, digests = statement(pl, input_set, lo, hi, rows, outs)
    total = n * len(pl.comps)
    blocks = []
    for b in range(pl.blocks(n)):
        chunks, units = [], []
        for gi in range(b * pl.G, (b + 1) * pl.G):
            if gi >= total:
                chunks += [None] * pl.chunks_per_comp
                units += [None] * pl.units_per_comp
                continue
            i, c = divmod(gi, len(pl.comps))
            cs, us = pl.comps[c]
            chunks += [[i, p, ci] for p, ci, _ in cs]
            units += [i * U.n + u for u in us]
        blocks.append([chunks, units])
    ret = lay.out_groups[0]
    n_ret = len(ret[2])
    fout = IL.flat_ports(input_set, sub.outputs, lo, hi)
    out_bytes = IL._pack_words(low.unit_outputs(fout, cuts)[:n_ret], ret[2], ret[1],
                               [c - ret[0] * IL.WORD for c in lay.out_port_cols[:n_ret]])
    leaf_ports = [j for j, (k, _) in enumerate(U.in_src[0]) if k == "leaf"]
    head = {"format": FORMAT, "statement": STATEMENT, "scheme": SCHEME, "unit": low.name, "unit_sha256": low.sha256,
            "set": input_set.name, "subcircuit": sub.id, "content_digest": input_set.content_digest, "range": [lo, hi], "instances": n,
            "units_per_instance": U.n, "units": n * U.n,
            "in_ports": [[nm, w] for nm, w in pl.in_ports], "out_ports": [[nm, w] for nm, w in pl.out_ports],
            "layout": {"sub_log": SUB_LOG, "comp_slots": pl.comp_slots, "nb": pl.nb, "chunks_per_block": pl.G * pl.chunks_per_comp,
                       "units_per_block": pl.units_per_block, "unit_log": pl.unit_log, "unit_pos0": pl.unit_pos0, "k_log": pl.k_log,
                       "G": pl.G, "chunks_per_comp": pl.chunks_per_comp, "units_per_comp": pl.units_per_comp,
                       "chunk_ports": [p for p, _, _ in pl.comps[0][0]] * pl.G,
                       "single_chunk": [2 * pl.in_ports[p][1] <= CHUNK for p, _, _ in pl.comps[0][0]] * pl.G,
                       "wiring": [[[q + pl.chunks_per_comp * (k // pl.units_per_comp), off] for q, off in pl.wiring[k % pl.units_per_comp]]
                                  for k in range(pl.units_per_block)],
                       "leaf_cols": [lay.in_port_cols[j] for j in leaf_ports],
                       "ret_group": [ret[0], ret[1]], "ret_cols": [c - ret[0] * IL.WORD for c in lay.out_port_cols[:n_ret]],
                       "out_leaf": [[p for k, p in U.out_src[u] if k == "ret"] for u in range(U.n)]},
            "blocks": blocks, "cut_words": len(U.cut), "frame_v3": fv}
    body = b"".join(np.concatenate([rows[nm] for nm, _ in pl.in_ports], axis=1).astype("<u2")[i].tobytes() for i in range(n))
    body += b"".join(digests[nm][i].tobytes() for i in range(n) for nm, _ in pl.in_ports)
    body += out_bytes
    if U.cut:
        body += np.asarray(cuts, dtype="<u4").tobytes()
    with open(path, "wb") as f:
        f.write(json.dumps(head, sort_keys=True).encode() + b"\n")
        f.write(body)
    return head


def stage(low: IL.Lowering, input_set, lo: int, hi: int, out_dir, k_max: int = K_MAX) -> Path:
    """Write ids [lo, hi) of ``input_set`` as ``<out_dir>/frame-<n>.bin`` and the unit netlist as ``<out_dir>/net.txt``: the
    verifier's own staging (red-team-flock-2 IR2)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "net.txt").write_text(low.text)
    sub = input_set.subcircuit
    pl = plan(low, sub, k_max)
    path = out / f"frame-{hi - lo}.bin"
    if low.units.cut:
        raise NotImplementedError(f"{low.name}: the frame statement's cut words (committed publics + native tail) are not wired yet")
    write(pl, input_set, lo, hi, path)
    return path
