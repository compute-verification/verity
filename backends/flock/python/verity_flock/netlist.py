"""``verity/flock-netlist/v1``: any lowered IR template (``flock-ir-unit/v2`` netlists, :mod:`ir_lower`) proved with its
frame-v3 rows hashed in the circuit and nothing else public (``live/src/netlist.rs`` is the statement).

Private by default. The public data are the frame-v3 roots, the row digests they are built from (the serving tree's
leaves), the declared outputs and the layout below. Every other word (every compression's chaining value, every chunk
value, every word a unit, a tail stage or a lookup reads from another) stays inside the witness, joined by copy
constraints (Δ) inside a block: no public intermediate word, and the verifier evaluates no part of the computation (a tail,
the Gumbel noise of a public seed included, is a slot type of the circuit or the template has no statement here).

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


#: MUFU lookup slots: low decoder bits per table (``lookup.rs``); the high decoder takes the rest
LOOKUP_LO = {24: 14, 23: 13}


def _decode_rows(k: int) -> int:
    return 0 if k == 1 else (1 << k) + _decode_rows(k // 2) + _decode_rows(k - k // 2)


def lookup_rows(n: int, lo: int, bits: int) -> int:
    """Rows of a lookup slot (``lookup.rs`` LookupNet::build): index word, decoders, products, output word, constant."""
    r = 128 + _decode_rows(lo) + _decode_rows(n - lo) + bits * (1 << (n - lo))
    return -(-r // 128) * 128 + 128 + 1


def _slot_log(useful: int) -> int:
    return max(7, (useful - 1).bit_length())


def _wires(low: IL.Lowering, tail) -> list:
    """Instance-local wires ``[dst net, dst index, dst port, src net, src index, src port]``: every cut input port of every unit
    and every input port of every stage and lookup, from the port that computes it (a unit's cut output, a stage's output, a
    lookup's value)."""
    U = low.units
    n_ret = sum(1 for k, _ in U.out_src[0] if k == "ret")
    producer: dict[int, tuple] = {}
    for u, src in enumerate(U.out_src):
        for j, (kind, k) in enumerate(src):
            if kind == "cut":
                producer[k] = ("unit", u, j)
    last = len(tail.stages) - 1
    for j, tag in enumerate(tail.stages[last].outs):
        if tag[0] == "cut":
            producer[tag[1]] = (f"stage{last}", 0, j)
    local, seen = [], {}
    for j, (table, _) in enumerate(tail.lookups):
        local.append(seen.get(table, 0))
        seen[table] = seen.get(table, 0) + 1
    wires = []
    for u, src in enumerate(U.in_src):
        for j, (kind, k) in enumerate(src):
            if kind == "cut":
                wires.append(["unit", u, j, *producer[k]])
    for i, st in enumerate(tail.stages):
        prev = tail.stages[i - 1] if i else None
        for j, tag in enumerate(st.ins):
            if tag[0] in ("pass", "ctx"):
                wires.append([f"stage{i}", 0, j, f"stage{i - 1}", 0, prev.outs.index(tag)])
            elif tag[0] == "lookup":
                wires.append([f"stage{i}", 0, j, f"lookup:{tail.lookups[tag[1]][0]}", local[tag[1]], 0])
            else:
                wires.append([f"stage{i}", 0, j, *producer[tag[1]]])
    for q, (table, si) in enumerate(tail.lookups):
        wires.append([f"lookup:{table}", local[q], 0, f"stage{si}", 0, tail.stages[si].outs.index(("index", q))])
    del n_ret
    return wires


def compose(low: IL.Lowering, sub, vus_per_block: int | None = None, forged: bool = False) -> str:
    """The ``flock-netlist/v1`` text for a template lowered to identical units and, for a cut lowering, its tail in the circuit
    (:mod:`tail`: stage netlists and MUFU lookup slots, wired to the units' cut ports)."""
    from . import tail as TL

    U = low.units
    if U.tail_rets:
        raise NotImplementedError(f"{low.name}: a tail that computes the outputs (tensor-core steps) is not in the circuit yet; "
                                  f"the statement has no verifier-evaluated form")
    if any(p.dtype != "u16" for p in (*sub.inputs, *sub.outputs)):
        raise ValueError(f"{low.name}: frame-v3 rows here are u16 words")
    tail = TL.lower_tail(low, forged) if U.cut else None
    stages = [TL.stage_layout(st) for st in tail.stages] if tail else []
    lookups: dict[str, int] = {}
    for table, _ in (tail.lookups if tail else []):
        lookups[table] = lookups.get(table, 0) + 1
    lv = low.leaves
    ports = [_port(p.name, w) for p, (w, _) in zip(sub.inputs, lv["in_ports"])]
    outs = [[p.name, w] for p, (w, _) in zip(sub.outputs, lv["out_ports"])]
    ul = _slot_log(low.layout.useful)
    if low.layout.useful == 1 << ul:
        raise ValueError(f"{low.name}: a unit fills its slot; a zero leaf needs a forced-zero row")
    lk = {t: {"table": t, "sha256": IL.TABLES[t], "in_bits": TL.TABLE_BITS[t], "out_bits": TL.VALUE_BITS, "lo_bits": LOOKUP_LO[TL.TABLE_BITS[t]]}
          for t in lookups}

    def fits(G: int):
        items = []
        for pi, p in enumerate(ports):
            items.append(("comp", SUB_LOG, _pow2(G * p.chunks) * p.nb, {"role": "runs", "port": pi}))
            if p.chunks > 1:
                items.append(("comp", SUB_LOG, G * p.chunks, {"role": "parents", "port": pi}))
        items.append(("unit", ul, G * U.n, {"net": "unit"}))
        for i, lay in enumerate(stages):
            items.append(("unit", _slot_log(lay.useful), G, {"net": f"stage{i}"}))
        for t, c in lookups.items():
            L = lk[t]
            items.append(("unit", _slot_log(lookup_rows(L["in_bits"], L["lo_bits"], L["out_bits"])), G * c, {"net": f"lookup:{t}"}))
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
    per_vu = {"unit": U.n, **{f"stage{i}": 1 for i in range(len(stages))}, **{f"lookup:{t}": c for t, c in lookups.items()}}
    meta = {"format": FORMAT, "statement": STATEMENT, "scheme": SCHEME, "template": sub.template.id,
            "subcircuit": sub.id, "unit": low.name, "unit_sha256": low.sha256, "vus_per_block": G, "k_log": k_log,
            "ports": [[p.name, p.words, p.chunks, p.nb] for p in ports], "out_ports": outs, "ranges": ranges,
            "units_per_vu": U.n, "unit_log": ul, "leaves_in": lv["in"], "leaves_out": lv["out"], "row_key": _row_key().hex(),
            "dummy": dummy, "leaf_scheme": LEAF_SCHEME, "mask_log": MASK_LOG, "per_vu": per_vu, "lookups": lk,
            "wires": _wires(low, tail) if tail else []}
    head = f"{FORMAT} {low.name}\nMETA {json.dumps(meta, sort_keys=True, separators=(',', ':'))}\n"
    texts = [("unit", IL.netlist(low.layout, low.name, None, None))]
    texts += [(f"stage{i}", IL.netlist(lay, f"{low.name}/tail/stage{i}", None, None)) for i, lay in enumerate(stages)]
    return head + "".join(f"NET {nm} {t.count(chr(10))}\n{t}" for nm, t in texts)


def _row_key() -> bytes:
    from verity.commitments.rowleaf import ROLE_X, blake3_row_key

    return blake3_row_key(ROLE_X)


def _dummy(low: IL.Lowering, sub, ports: list[Port]) -> dict:
    """The all-zero instance's row digests (per port) and output words (every output leaf), as the padding instances hold."""
    import blake3

    key = _row_key()
    n_in = sum(p.words for p in ports)
    zero = np.zeros((1, n_in), dtype=np.uint64)
    outs, _, ok = low.evaluate(zero, IL.cut_words(low, zero) if low.units.cut else None)
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
    if low.units.cut:
        IL.write_tables(out / "tables")
        if not (out / "net-forged.txt").exists():
            (out / "net-forged.txt").write_text(compose(low, input_set.subcircuit, forged=True))
    path = out / f"{'pub' if public_only else 'inst'}-{hi - lo}.bin"
    write(text, input_set, lo, hi, path, public_only)
    return path


def lowering_for_set(input_set) -> IL.Lowering:
    """The template's C-Flock lowering for a set (``lowering_for_set`` when the template's statement depends on the set)."""
    from verity_numerical.bench import lowerings

    mod = lowerings.registry("C-Flock").module(input_set.subcircuit.template.id)
    return mod.lowering_for_set(input_set) if hasattr(mod, "lowering_for_set") else mod.frame_lowering(input_set.subcircuit)


def main(argv=None) -> int:
    """``python -m verity_flock.netlist --input-set DIR --n N [N ...] --out DIR``: the composite netlist, then for each N the
    prover's ``inst-N.bin`` and the verifier's own ``pub-N.bin`` (instances [0, N)); prints the netlist's sha256 (the pin)."""
    import argparse

    from verity_numerical.bench.input_sets import InputSet

    ap = argparse.ArgumentParser()
    ap.add_argument("--input-set", required=True)
    ap.add_argument("--n", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    s = InputSet.open(a.input_set)
    low = lowering_for_set(s)
    for n in a.n:
        stage(low, s, 0, n, a.out)
        stage(low, s, 0, n, a.out, public_only=True)
    print(json.dumps({"subcircuit": s.subcircuit.id, "unit": low.name, "pin": digest((Path(a.out) / "net.txt").read_text()), "n": a.n}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
