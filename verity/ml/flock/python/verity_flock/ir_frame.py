"""frame-v3 binding of a lowered IR template: the statement ``verity/flock-ir-frame/v3`` (``live/src/ir_frame.rs``).

Every input port of every instance is one frame-v3 row leaf, ``blake3-keyed/row/v2`` under the x-row key, one tree per port
under its own domain; every output word is one ``u16`` word leaf, one tree per output port: the GEMM cells' x, W and y. The
rows stay private: the proof hashes them with Flock's BLAKE3 compression circuit, and each unit's input bits are copies of
the message bits they hash. The outputs are public, as GEMM's y is.

Layout. A run is ``nb`` consecutive compressions of a 1024-byte BLAKE3 chunk (a whole chunk when that fits, else a half,
...); units and the runs they read form components, which must all have one shape. A block holds ``G`` components: their
runs' compressions at 2^14-bit slots ``0 .. comp_slots`` (run slots padded with empty ones to a power of two), then their
units at 2^unit_log slots ``unit_pos0 ..`` (``unit_pos0`` a multiple of the unit count, a power of two), so each region is one
aligned subcube. Every compression's counter, block length and flags, and every run's input chaining value (the key for a
chunk's first run, else the previous run's output), are verifier-fixed region claims, so a chunk's runs may sit in different
blocks with one circuit for all. Each run's output chaining value (for a one-chunk row, its ROOT output, the row digest) is a
prover-committed public; the verifier chains a chunk's runs through them, folds every row's chunk values into its digest
natively and checks the digests and the output words against the frame-v3 roots it recomputes.

A cut template's cut words (``ir_lower.cut_words``) follow in the file: public words of the units that read or compute them,
each unit's cut ports a region claim; the verifier evaluates the netlist's pinned tail on them before any session.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import ir_lower as IL

FORMAT = "flock-ir-frame/v3"
STATEMENT = "verity/flock-ir-frame/v3"
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
    """How a template's instances fill blocks. A run is ``nb`` consecutive compressions of one chunk (``chunk_nb`` per chunk,
    ``nb`` dividing it): run ``r`` of chunk ``c`` of an input port hashes its bytes ``1024 c + 64 nb r ..`` (``64 nb`` of them).
    ``comps[c]`` = (its runs as (input port, chunk index, run index), its instance-local units); ``wiring[k][j]`` = (local run
    slot, byte offset in the run) of leaf input port j of a component's unit k (the same for every component)."""

    low: IL.Lowering
    in_ports: list[tuple[str, int]]
    out_ports: list[tuple[str, int]]
    comps: list[tuple[list[tuple[int, int, int]], list[int]]]
    wiring: list[list[tuple[int, int]]]
    G: int
    nb: int
    chunk_nb: int
    comp_slots: int
    unit_log: int
    unit_pos0: int
    units_per_block: int
    k_log: int
    #: input ports the units read as public words (hashed by the verifier, not in the circuit)
    public_ports: list = field(default_factory=list)

    @property
    def chunks_per_comp(self) -> int:
        return len(self.comps[0][0])

    @property
    def units_per_comp(self) -> int:
        return len(self.comps[0][1])

    @property
    def chunk_slots(self) -> int:
        """Run slots per block: the ``G`` components' runs, then empty slots up to a power of two."""
        return self.comp_slots // self.nb

    def blocks(self, n: int) -> int:
        return -(-n * len(self.comps) // self.G)


def _components(U, reads, run_bytes: int):
    """Units and the runs of ``run_bytes`` they read, joined into components: ``(comps, wiring)``, or ``None`` when the
    components are not all one shape."""
    parent: dict = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    runs = [[(p, byte // CHUNK, byte % CHUNK // run_bytes, byte % run_bytes) for p, byte in reads[u]] for u in range(U.n)]
    for u in range(U.n):
        find(("u", u))
        for p, c, r, _ in runs[u]:
            parent[find(("r", p, c, r))] = find(("u", u))
    groups: dict = {}
    for u in range(U.n):
        groups.setdefault(find(("u", u)), []).append(u)
    comps, shapes = [], []
    for units in sorted(groups.values(), key=min):
        rs = sorted({(p, c, r) for u in units for p, c, r, _ in runs[u]})
        slot = {x: i for i, x in enumerate(rs)}
        shapes.append((tuple(p for p, _, _ in rs), tuple(tuple((slot[(p, c, r)], off) for p, c, r, off in runs[u]) for u in units)))
        comps.append((rs, units))
    if any(s != shapes[0] for s in shapes):
        return None
    return comps, [list(w) for w in shapes[0][1]]


def plan(low: IL.Lowering, sub, k_max: int = K_MAX) -> Plan:
    """The largest run (a whole chunk, then halves, ...) whose components fit a block of at most 2^k_max bits, and the block
    packing with the best fill for it."""
    if any(p.dtype != "u16" for p in (*sub.inputs, *sub.outputs)):
        raise ValueError(f"{low.name}: frame-v3 rows here are u16 words")
    in_ports, out_ports = _ports(low, sub)
    starts = np.cumsum([0] + [w for _, w in in_ports])
    row_bytes = [2 * w for _, w in in_ports]
    sizes = {min(CHUNK, rb - CHUNK * c) for rb in row_bytes for c in range(-(-rb // CHUNK))}
    size = next(iter(sizes))
    if len(sizes) != 1 or size % BLOCK or not _pow2(size // BLOCK):
        raise ValueError(f"{low.name}: chunks of {sorted(sizes)} bytes (need one size, a power-of-two multiple of {BLOCK})")
    chunk_nb = size // BLOCK
    U = low.units

    def at(pos: int) -> tuple[int, int]:
        p = int(np.searchsorted(starts, pos, side="right") - 1)
        return p, 2 * (pos - int(starts[p]))

    reads = [[at(p) for k, p in U.in_src[u] if k == "leaf"] for u in range(U.n)]
    useful = low.layout.useful
    ul = max(7, (useful - 1).bit_length())
    nb = chunk_nb
    while nb >= 1:
        found = _components(U, reads, BLOCK * nb)
        if found is not None:
            comps, wiring = found
            cpc, upc = len(comps[0][0]), len(comps[0][1])
            best = None
            for G in range(1, MAX_SLOTS + 1):
                Q, units = 1 << (G * cpc - 1).bit_length(), G * upc
                C = Q * nb
                if C > MAX_SLOTS or units > MAX_SLOTS:
                    break
                if not _pow2(units):
                    continue
                pos0 = -(-(C << SUB_LOG) // (1 << ul))
                pos0 = -(-pos0 // units) * units
                k_log = max(K_MIN, (((pos0 + units) << ul) - 1).bit_length())
                if k_log > k_max:
                    continue
                fill = ((C << SUB_LOG) + units * useful) / (1 << k_log)
                if best is None or fill > best[0] + 1e-9:
                    best = (fill, G, C, pos0, units, k_log)
            if best is not None:
                _, G, C, pos0, units, k_log = best
                return Plan(low, in_ports, out_ports, comps, wiring, G, nb, chunk_nb, C, ul, pos0, units, k_log)
        nb //= 2
    raise ValueError(f"{low.name}: no run of a chunk's compressions gives components of one shape that fit a block of at most "
                     f"2^{k_max} bits (units of 2^{ul}, a power-of-two unit count)")


def _ports(low: IL.Lowering, sub) -> tuple[list, list]:
    """``(name, words)`` per input and output port, the words from the lowering's pinned leaf maps (a key-count port has
    none in the subcircuit)."""
    lv = low.leaves
    return ([(p.name, w) for p, (w, _) in zip(sub.inputs, lv["in_ports"])], [(p.name, w) for p, (w, _) in zip(sub.outputs, lv["out_ports"])])


def plan_tc(low: IL.Lowering, sub) -> Plan:
    """Tensor-core steps (``ir_lower.tc_units``) of an attention head: the public query port is not hashed here; the two key
    ports (K, V; D words per key, a run of ``2 D`` bytes = one key's row) are. One block per group of 16 keys: run slots
    ``[K keys 0..15 | V keys 0..15]``, unit slots ``[QK (key j, step s) at D/16 j + s | PV column d at D + d]``; a key past T is
    an empty run slot, its QK steps empty unit slots, and a PV step reads the empty slot's zero words where the IR pads."""
    in_ports, out_ports = _ports(low, sub)
    starts = [int(x) for x in np.cumsum([0] + [w for _, w in in_ports])]
    public = {pos for _, pos in low.units.public}
    pub = [p for p in range(len(in_ports)) if all(q in public for q in range(starts[p], starts[p + 1]))]
    rows = [p for p in range(len(in_ports)) if p not in pub]
    if len(pub) != 1 or len(rows) != 2:
        raise ValueError(f"{low.name}: a tensor-core frame needs one public port and two key ports")
    pk, pv = rows
    D = in_ports[pub[0]][1]
    T = in_ports[pk][1] // D
    key_bytes, steps = 2 * D, D // 16
    if in_ports[pv][1] != T * D or key_bytes % BLOCK or CHUNK % key_bytes or D % 16:
        raise ValueError(f"{low.name}: keys of {D} words do not tile BLAKE3 chunks")
    nb = key_bytes // BLOCK

    def run(p, c):
        byte = c * key_bytes
        return (p, byte // CHUNK, byte % CHUNK // key_bytes)

    lv = low.leaves["in"]
    ng = -(-T // 16)
    comps = [([run(pk, 16 * g + j) if 16 * g + j < T else None for j in range(16)]
              + [run(pv, 16 * g + j) if 16 * g + j < T else None for j in range(16)], [None] * (2 * D)) for g in range(ng)]
    for u, leaves in enumerate(lv):
        first = next(x for x in leaves if x >= 0)
        p = next(q for q in rows if starts[q] <= first < starts[q + 1])
        c, w = divmod(first - starts[p], D)
        if p == pk:
            s = w // 16
            want = [starts[pk] + c * D + 16 * s + t for t in range(16)]
            g, slot = c // 16, steps * (c % 16) + s
        else:
            g, d = c // 16, w
            want = [starts[pv] + (16 * g + t) * D + d if 16 * g + t < T else -1 for t in range(16)]
            slot = D + d
        if leaves != want or comps[g][1][slot] is not None:
            raise ValueError(f"{low.name}: unit {u} is not a QK or PV step of its key group")
        comps[g][1][slot] = u
    wiring = [[(j, 32 * s + 2 * t) for t in range(16)] for j in range(16) for s in range(steps)] + \
             [[(16 + t, 2 * d) for t in range(16)] for d in range(D)]
    ul = max(7, (low.layout.useful - 1).bit_length())
    Q, units = 32, 2 * D
    C = Q * nb
    pos0 = -(-(C << SUB_LOG) // (1 << ul))
    pos0 = -(-pos0 // units) * units
    k_log = max(K_MIN, (((pos0 + units) << ul) - 1).bit_length())
    if k_log > K_MAX or units > MAX_SLOTS or C > MAX_SLOTS:
        raise ValueError(f"{low.name}: a 16-key group does not fit a block of at most 2^{K_MAX} bits")
    return Plan(low, in_ports, out_ports, comps, wiring, 1, nb, CHUNK // BLOCK, C, ul, pos0, units, k_log, public_ports=pub)


# --- frame-v3 statement commitments ------------------------------------------------------------------------------------------

def statement(pl: Plan, input_set, lo: int, hi: int, rows: dict[str, np.ndarray], outs: dict[str, np.ndarray]):
    """The frame-v3 roots over instances [lo, hi) of ``input_set``: a row-leaf tree per input port, a word-leaf tree per output
    port (instance-major words), each under a domain bound to the set, the range, the port and its schema. Returns the
    header's ``frame_v3`` entry and the row digests (port -> (n, 32) uint8)."""
    import blake3
    from verity.ml.commitments.frame_v3 import FrameV3, Port, RowLeaf, WordLeaf
    from verity.ml.commitments.identity import identity_digest
    from verity.ml.commitments.indexed import RangeIndexedDomain
    from verity.ml.commitments.merkle import CommitmentDomain
    from verity.ml.commitments.rowleaf import ROLE_X, SCHEMA_BLAKE3_ROW, blake3_row_key

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
    """``flock-ir-frame/v3``: one JSON header line (the plan, every block's runs as ``[instance, port, chunk, run]`` and units, the
    frame-v3 statement), then every instance's input rows (ports in order, words little-endian), every instance's row digests
    (32 bytes per input port), the outputs: every unit's returned-output words (``ret_words`` x 16 bytes, units instance-major)
    or, when the tail computes them, every instance's output words (u16 LE), and, for a cut template, every instance's cut
    words (u32 LE)."""
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
            chunks += [None if x is None else [i, *x] for x in cs]
            units += [None if u is None else i * U.n + u for u in us]
        blocks.append([chunks + [None] * (pl.chunk_slots - len(chunks)), units])
    fout = IL.flat_ports(input_set, sub.outputs, lo, hi)
    tail_out = bool(U.tail_rets)
    n_ret = 0 if tail_out else len(lay.out_groups[0][2])
    ret = None if tail_out else lay.out_groups[0]
    if tail_out:
        out_bytes = np.asarray(fout, dtype="<u2").tobytes()
    else:
        out_bytes = IL._pack_words(low.unit_outputs(fout, cuts)[:n_ret], ret[2], ret[1],
                                   [c - ret[0] * IL.WORD for c in lay.out_port_cols[:n_ret]])
    leaf_ports = [j for j, (k, _) in enumerate(U.in_src[0]) if k in ("leaf", "zero")]
    head = {"format": FORMAT, "statement": STATEMENT, "scheme": SCHEME, "unit": low.name, "unit_sha256": low.sha256,
            "set": input_set.name, "subcircuit": sub.id, "content_digest": input_set.content_digest, "range": [lo, hi], "instances": n,
            "units_per_instance": U.n, "units": n * U.n,
            "in_ports": [[nm, w] for nm, w in pl.in_ports], "out_ports": [[nm, w] for nm, w in pl.out_ports],
            "public_ports": pl.public_ports, "outputs": "tail" if tail_out else "units",
            "layout": {"sub_log": SUB_LOG, "comp_slots": pl.comp_slots, "nb": pl.nb, "chunks_per_block": pl.chunk_slots,
                       "units_per_block": pl.units_per_block, "unit_log": pl.unit_log, "unit_pos0": pl.unit_pos0, "k_log": pl.k_log,
                       "G": pl.G, "chunks_per_comp": pl.chunks_per_comp, "units_per_comp": pl.units_per_comp,
                       "wiring": [[[q + pl.chunks_per_comp * (k // pl.units_per_comp), off] for q, off in pl.wiring[k % pl.units_per_comp]]
                                  for k in range(pl.units_per_block)],
                       "leaf_cols": [lay.in_port_cols[j] for j in leaf_ports],
                       "ret_group": [ret[0], ret[1]] if ret else None,
                       "ret_cols": [c - ret[0] * IL.WORD for c in lay.out_port_cols[:n_ret]] if ret else [],
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


def serve_args(path) -> list[str]:
    """Extra ``flock-ir-frame serve`` arguments for a staged file: none for this statement."""
    return []


def stage(low: IL.Lowering, input_set, lo: int, hi: int, out_dir, k_max: int = K_MAX) -> Path:
    """Write ids [lo, hi) of ``input_set`` as ``<out_dir>/frame-<n>.bin`` and the unit netlist as ``<out_dir>/net.txt``: the
    verifier's own staging (red-team-flock-2 IR2)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "net.txt").write_text(low.text)
    sub = input_set.subcircuit
    pl = plan_tc(low, sub) if low.units.tail_rets else plan(low, sub, k_max)
    path = out / f"frame-{hi - lo}.bin"
    cuts = IL.cut_words(low, IL.flat_ports(input_set, sub.inputs, lo, hi)) if low.units.cut else None
    write(pl, input_set, lo, hi, path, cuts)
    return path
