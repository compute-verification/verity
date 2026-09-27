"""``verity/flock-circuit``: any lowered IR template (``flock-ir-unit/v2`` circuits, :mod:`ir_lower`) proved with its serving
rows committed in the circuit and nothing else public (``live/src/circuit.rs`` is the statement).

Rows are ``frame-v3-sha512`` leaves under ``hm96-sha512/row/v1`` (``verity.commitments.rowleaf``): a row's inner digest is
``x = SHA-512(sha512_row_prefix || row)`` (``sha512/row/v1``), its hiding commit string ``b || c`` with ``b = x XOR M(key) y`` and
``c = SHA-512(salt_prefix || y)`` for a 192-byte salt ``y`` (``hm96.SHA512``, key its ``default_key``), and the frame leaf binds
``hm96.SHA512.tree_leaf(key, b || c)``. Both hashes are proved here: the rows and their salts are private.

Private by default. The public data are the frame-v3-sha512 roots, each row's ``b || c`` (the serving tree's leaves are
functions of it), the declared outputs, and the layout below. Every other word (every compression's chaining value, every
word a unit, a tail stage or a lookup reads from another) stays inside the witness, joined by copy constraints (Δ) inside a
block: no public intermediate word, and the verifier evaluates no part of the computation (a tail, the Gumbel noise of a
public seed included, is a slot type of the circuit or the template has no statement here).

One or more whole instances (VUs) per block. A block is the same for every block (Flock's ``A = I ⊗ A_0 + Δ``) and holds aligned
ranges of slots of a few types:

* ``sha512x3``: three independent SHA-512 compressions (:mod:`sha512_circuit`, 2^18 bits): a row's blocks after its constant
  first block, chained by Δ (the first from the prefix's midstate, a Δ constant), the row's bytes the message bits;
* ``hm96``: one per input port (``per_vu`` a power of two): the row's padding compression (the padding block a Δ constant),
  then ``b || c`` from the salt; its ``bc`` output is the public region of the row;
* ``unit``: the template's identical units (2^unit_log bits), each input bit a copy of the message bit that hashes it;
* ``mask``: 2^14 free cells (``A = B = I``), reserved for zero-knowledge masks and filled by the prover's mask hook (zeros in
  M0), apart from the forced-zero padding (positions no range covers).

A padding instance (the block's unused instances and every block past the last) is the all-zero instance with zero salts; its
``b || c`` and outputs are pinned here (``dummy``), so the verifier knows every public value of every block.

The composite circuit (``flock-circuit``) is the statement's meaning: a ``META`` line (ports, ranges, each row's constants,
the wires, the units' leaf and output maps, the pinned dummy values, the leaf-commitment scheme, the program digests, the
partition digest), then each slot type's ``flock-ir-unit/v2`` circuit. Its sha256 is the pin. The verifier derives Δ, the public
regions and their values from it and from its own public instance file (no rows, no salts), never from the prover's; the
instance file's header carries each block's unit indices (the VUs of the partition it proves).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import ir_lower as IL

FORMAT = "flock-circuit"
STATEMENT = "verity/flock-circuit"
SCHEME = "frame-v3-sha512/hm96-sha512"
ROW_SCHEMA = "hm96-sha512/row/v1"
INST_FORMAT = "flock-circuit-inputs"
#: The leaf-commitment scheme of the proof's own Merkle trees (Ligerito levels), pinned in the statement and the backend
#: identity and swappable: ``hm96-sha512/v1`` (``verity.commitments.hm96.SHA512``, its PROTOCOL.md section 7) over the
#: SHA-512 column digest, leaf = ``tree_leaf(key, b || c)``, an opening = the column and its 192-byte salt (``opened_salts``).
#: Salts are a ChaCha20 expansion of a fresh 256-bit OS key per proof (Daniel, 2026-09-27: salts drawn leaf by leaf from the OS
#: cost +96% prover time at m = 33), modelled as uniform like the OS generator's own ChaCha20 output; the key is never reused or
#: stored. The level-0 salts are shared by both reps (one root). Key: the scheme's ``default_key``, one constant of the statement for every
#: tree, pinned by its SHA-512. Never a witness: a prover choosing the key after commitment solves for one that opens a leaf to
#: any value.
LEAF_SCHEME = {"id": "hm96-sha512/v1", "inner": "sha512 column digest", "salt_bytes": 192,
               "salt_source": "chacha20 expansion of a fresh 256-bit os key per proof, never reused or stored", "key": "default_key, a statement constant for every tree",
               "key_sha512": "afc5a60c85d8aa75cb4c7f726588d299a1712e1c6a621f09ca64809a5fedd744"
                             "a0f6b0201d05e82b29cd5b274e6c29670cd4e4ab10831911fb1f9e0c7019df22"}
SHA_NET, HM_NET, PER_SLOT = "sha512x3", "hm96", 3
SHA_LOG = HM_LOG = 18
MASK_LOG = 14
K_MAX = 26
#: The partition digest here covers this lowering's own split of a VU into slots (its unit circuit, leaf and output maps and
#: wires). Provisional: the partition checker's partition digest replaces it (commitments-decisions-routing §1).
PARTITION_RULE = "flock-circuit/unit-cover/v0"
SALT_BYTES, BC_BYTES = 192, 128


def _pow2(x: int) -> int:
    return 1 << max(0, (x - 1).bit_length())


def _log2(x: int) -> int:
    assert x & (x - 1) == 0 and x > 0, x
    return x.bit_length() - 1


@dataclass
class Port:
    name: str
    words: int

    @property
    def bytes(self) -> int:
        return 2 * self.words

    @property
    def comps(self) -> int:
        """Compressions after the constant first block (the padding block is the hm96 slot's)."""
        return self.bytes // 128


def _port(name: str, words: int) -> Port:
    if (2 * words) % 128:
        raise ValueError(f"port {name}: {2 * words}-byte rows are not whole SHA-512 blocks")
    return Port(name, words)


def _row_consts(p: Port) -> dict:
    """A row's constants: ``sha512/row/v1``'s first block, the chaining value after it, the padding block."""
    from . import sha512_circuit as S

    prefix = S.sha512_row_prefix(p.words)
    return {"prefix": prefix.hex(), "midstate": [f"{w:016x}" for w in S.midstate(prefix)], "pad": S.pad_tail(128 + p.bytes).hex()}


def _ranges(items: list[tuple[str, int, int, dict]]) -> tuple[list[dict], int]:
    """Place ``(type, slot_log, count, extra)`` ranges, largest first, each aligned to its own power-of-two span; then the
    ``packed`` ones (no region reads them), each aligned to its slot only and spanning its count."""
    plain = [i for i in range(len(items)) if not items[i][3].get("packed")]
    order = sorted(plain, key=lambda i: (-(_pow2(items[i][2]) << items[i][1]), i))
    at, out = 0, [None] * len(items)
    for i in order:
        typ, sl, n, extra = items[i]
        span = _pow2(n) << sl
        at = -(-at // span) * span
        out[i] = {"type": typ, "slot_log": sl, "pos0": at >> sl, "count": n, **extra}
        at += span
    for i in (i for i in range(len(items)) if items[i][3].get("packed")):
        typ, sl, n, extra = items[i]
        at = -(-at // (1 << sl)) << sl
        out[i] = {"type": typ, "slot_log": sl, "pos0": at >> sl, "count": n, **extra}
        at += n << sl
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


def _row_slots(ports: list[Port]) -> tuple[int, list[list[tuple[int, int]]]]:
    """Per VU: the sha512x3 slots, and each port's compressions as ``(slot, sub)`` (three per slot, ports in order)."""
    at, where = 0, []
    for p in ports:
        where.append([divmod(at + k, PER_SLOT) for k in range(p.comps)])
        at += p.comps
    return -(-at // PER_SLOT), where


def _row_wires(ports: list[Port]) -> list:
    """Each row's chain (compression k's ``cv`` from compression k-1's ``out``) and its hm96 slot's ``cv`` (from the last)."""
    _, where = _row_slots(ports)
    wires = []
    for p, w in enumerate(where):
        for k in range(1, len(w)):
            wires.append([SHA_NET, w[k][0], 2 * w[k][1], SHA_NET, w[k - 1][0], w[k - 1][1]])
        wires.append([HM_NET, p, 0, SHA_NET, w[-1][0], w[-1][1]])
    return wires


def compose(low: IL.Lowering, sub, vus_per_block: int | None = None, forged: bool = False) -> str:
    """The ``flock-circuit`` text for a template lowered to identical units and, for a cut lowering, its tail in the circuit
    (:mod:`tail`: stage circuits and MUFU lookup slots, wired to the units' cut ports)."""
    from . import sha512_circuit as S
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
    n_sha, _ = _row_slots(ports)
    n_hm = _pow2(len(ports))

    def fits(G: int):
        items = [("unit", SHA_LOG, G * n_sha, {"net": SHA_NET, "packed": True}), ("unit", HM_LOG, G * n_hm, {"net": HM_NET}),
                 ("unit", ul, G * U.n, {"net": "unit"})]
        for i, lay in enumerate(stages):
            items.append(("unit", _slot_log(lay.useful), G, {"net": f"stage{i}"}))
        for t, c in lookups.items():
            L = lk[t]
            items.append(("unit", _slot_log(lookup_rows(L["in_bits"], L["lo_bits"], L["out_bits"])), G * c, {"net": f"lookup:{t}"}))
        items.append(("mask", MASK_LOG, 1, {"net": "mask"}))
        return _ranges(items)

    if vus_per_block is None:
        G, k1 = 1, fits(1)[1]
        while 2 * G * U.n <= 4096 and fits(2 * G)[1] <= max(k1, 22):
            G *= 2
    else:
        G = vus_per_block
    ranges, k_log = fits(G)
    if k_log > K_MAX:
        raise ValueError(f"{low.name}: an instance needs a block of 2^{k_log} bits (at most 2^{K_MAX})")
    per_vu = {"unit": U.n, SHA_NET: n_sha, HM_NET: n_hm, **{f"stage{i}": 1 for i in range(len(stages))},
              **{f"lookup:{t}": c for t, c in lookups.items()}}
    tail_wires = _wires(low, tail) if tail else []
    cover = {"rule": PARTITION_RULE, "unit_sha256": low.sha256, "units_per_vu": U.n, "leaves_in": lv["in"], "leaves_out": lv["out"],
             "tail_wires": tail_wires, "stages": [digest(_expanded(lay, f"{low.name}/tail/stage{i}")) for i, lay in enumerate(stages)],
             "lookups": lk}
    meta = {"format": FORMAT, "statement": STATEMENT, "scheme": SCHEME, "row_schema": ROW_SCHEMA, "template": sub.template.id,
            "subcircuit": sub.id, "unit": low.name, "unit_sha256": low.sha256, "vus_per_block": G, "k_log": k_log,
            "ports": [[p.name, p.words, p.comps] for p in ports], "out_ports": outs, "ranges": ranges, "rows": [_row_consts(p) for p in ports],
            "units_per_vu": U.n, "unit_log": ul, "leaves_in": lv["in"], "leaves_out": lv["out"], "row_key_sha512": LEAF_SCHEME["key_sha512"],
            "dummy": _dummy(low, ports), "leaf_scheme": LEAF_SCHEME, "mask_log": MASK_LOG, "per_vu": per_vu, "lookups": lk,
            "wires": _row_wires(ports) + tail_wires, "program_digests": sub.program_digests(),
            "partition": {"rule": PARTITION_RULE, "digest": hashlib.sha256(json.dumps(cover, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}}
    head = f"{FORMAT} {low.name}\nMETA {json.dumps(meta, sort_keys=True, separators=(',', ':'))}\n"
    texts = [("unit", _expanded(low.layout, low.name)),
             (SHA_NET, _expanded(IL.layout(S.compression3(), ["cv0", "msg0", "cv1", "msg1", "cv2", "msg2"], ["out0", "out1", "out2"]), SHA_NET)),
             (HM_NET, _expanded(IL.layout(S.hm96_row(), ["cv", "pad", "y"], ["bc"]), HM_NET))]
    texts += [(f"stage{i}", _expanded(lay, f"{low.name}/tail/stage{i}")) for i, lay in enumerate(stages)]
    return head + "".join(f"CIRCUIT {nm} {t.count(chr(10))}\n{t}" for nm, t in texts)


def _expanded(lay: IL.Layout, name: str) -> str:
    """A slot type's expanded circuit as ``flock-ir-unit/v2`` text (``ir_lower``'s writer)."""
    return IL.netlist(lay, name, None, None)



def _commit_strings(ports: list[Port], rows: dict, salts: dict) -> dict[str, list[bytes]]:
    """Each row's ``b || c`` (``hm96.SHA512`` over its ``sha512/row/v1`` digest and its salt)."""
    from verity.commitments import hm96
    from verity.commitments.rowleaf import ROLE_X, sha512_row_layout

    inst = hm96.SHA512
    out = {}
    for p in ports:
        lay = sha512_row_layout(ROLE_X, 16, p.words)
        r = np.ascontiguousarray(rows[p.name], dtype="<u2")
        out[p.name] = [inst.commit_string(inst.default_key, lay.digest(r[i].tobytes()), salts[p.name][i]) for i in range(len(r))]
    return out


def _dummy(low: IL.Lowering, ports: list[Port]) -> dict:
    """The all-zero instance's ``b || c`` (zero rows, zero salts; per port) and output words (every output leaf), as the padding
    instances hold."""
    n_in = sum(p.words for p in ports)
    zero = np.zeros((1, n_in), dtype=np.uint64)
    outs, _, ok = low.evaluate(zero, IL.cut_words(low, zero) if low.units.cut else None)
    assert int(ok) & 1, "the all-zero instance does not satisfy the unit rows"
    bc = _commit_strings(ports, {p.name: np.zeros((1, p.words), np.uint16) for p in ports}, {p.name: [bytes(SALT_BYTES)] for p in ports})
    return {"bc": [bc[p.name][0].hex() for p in ports], "outputs": [int(x) for x in outs[0]]}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --- instance files -----------------------------------------------------------------------------------------------------------

def statement(ports: list[Port], out_ports: list[tuple[str, int]], input_set, lo: int, hi: int, rows: dict, outs: dict, salts: dict):
    """The frame-v3-sha512 roots over instances [lo, hi): an ``hm96-sha512/row/v1`` tree per input port, a ``u16`` word tree per
    output port (instance-major words), each under a domain bound to the set, the range, the port and its schema. Returns the
    header's ``frame_v3`` entry and every row's ``b || c``."""
    from verity.commitments import hm96
    from verity.commitments.frame_v3 import FrameV3, RowLeaf, WordLeaf
    from verity.commitments.frame_v3 import Port as FPort
    from verity.commitments.identity import identity_digest
    from verity.commitments.indexed import RangeIndexedDomain
    from verity.commitments.merkle import CommitmentDomain
    from verity.commitments.rowleaf import ROLE_X, row_leaf

    from . import ir_frame

    inst, n = hm96.SHA512, hi - lo
    schemas = {**{p.name: ROW_SCHEMA for p in ports}, **{nm: "u16" for nm, _ in out_ports}}
    counts = {**{p.name: n for p in ports}, **{nm: n * w for nm, w in out_ports}}
    bind, doms = {}, {}
    for name in schemas:
        bind[name] = bytes.fromhex(identity_digest(ir_frame.BINDING_TAG, {"set": input_set.name, "content_digest": input_set.content_digest,
                                                                          "lo": int(lo), "hi": int(hi), "port": name, "schema": schemas[name]}))
        doms[name] = CommitmentDomain(bind[name], ir_frame.OWNER, RangeIndexedDomain(0, counts[name]), hash="sha512")
    fv = FrameV3({**{p.name: FPort(doms[p.name], RowLeaf("sha512", ROLE_X, 16, p.words)) for p in ports},
                  **{nm: FPort(doms[nm], WordLeaf("u16", 2)) for nm, _ in out_ports}})
    bcs = _commit_strings(ports, rows, salts)
    roots = {}
    for p in ports:
        leaves = [row_leaf(doms[p.name], i, inst.tree_leaf(inst.default_key, bcs[p.name][i]), ROW_SCHEMA) for i in range(n)]
        roots[p.name] = fv.root(p.name, leaves).hex()
    for name, _ in out_ports:
        flat = np.asarray(outs[name], dtype=np.uint64).reshape(-1)
        roots[name] = fv.root(name, [fv.tree_leaf(name, r, fv.leaf_layout(name, r).digest(int(v).to_bytes(2, "big")))
                                     for r, v in enumerate(flat.tolist())]).hex()
    return {"frame": "frame-v3-sha512", "roots": roots, "domain_ids": {k: d.domain_id.hex() for k, d in doms.items()},
            "bindings": {k: b.hex() for k, b in bind.items()}, "owner": ir_frame.OWNER, "schemas": schemas,
            "row_key_sha512": hashlib.sha512(inst.default_key).hexdigest()}, bcs


def write(text: str, input_set, lo: int, hi: int, path, public_only: bool) -> dict:
    """``flock-circuit-inputs``: one JSON header line (the set, the range, each block's unit indices, the composite circuit's
    sha256, the frame-v3-sha512 statement), then, for the prover only, every instance's input rows (ports in order, u16 LE) and
    every row's salt (192 bytes, instance-major, ports in order; fresh from the OS generator), then every row's ``b || c``
    (128 bytes) and every instance's output words (u16 LE, output leaves in order). The verifier's own file (``public_only``)
    has neither rows nor salts: it never holds a private word. Staging draws the salts; a serving committer draws its own."""
    meta = json.loads(text.split("\n", 2)[1][5:])
    sub = input_set.subcircuit
    n = hi - lo
    ports = [Port(nm, w) for nm, w, _ in meta["ports"]]
    out_ports = [tuple(x) for x in meta["out_ports"]]
    rows = {p.name: np.asarray(input_set.port(p.name, lo, hi), dtype=np.uint16).reshape(n, -1) for p in sub.inputs}
    outs = {p.name: np.asarray(input_set.port(p.name, lo, hi), dtype=np.uint16).reshape(n, -1) for p in sub.outputs}
    salts = {p.name: [os.urandom(SALT_BYTES) for _ in range(n)] for p in ports}
    fv, bcs = statement(ports, out_ports, input_set, lo, hi, rows, outs, salts)
    G = meta["vus_per_block"]
    units = {"vu_range": [lo, hi], "vus_per_block": G, "units_per_vu": meta["units_per_vu"],
             "blocks": [[lo + b * G, min(G, n - b * G)] for b in range(-(-n // G))]}
    head = {"format": INST_FORMAT, "statement": STATEMENT, "scheme": SCHEME, "set": input_set.name, "subcircuit": sub.id,
            "content_digest": input_set.content_digest, "range": [lo, hi], "instances": n, "circuit_sha256": digest(text),
            "units": units, "frame_v3": fv, "rows": not public_only}
    body = b""
    if not public_only:
        body += b"".join(np.concatenate([rows[p.name] for p in ports], axis=1).astype("<u2")[i].tobytes() for i in range(n))
        body += b"".join(salts[p.name][i] for i in range(n) for p in ports)
    body += b"".join(bcs[p.name][i] for i in range(n) for p in ports)
    body += np.concatenate([outs[nm] for nm, _ in out_ports], axis=1).astype("<u2").tobytes()
    with open(path, "wb") as f:
        f.write(json.dumps(head, sort_keys=True).encode() + b"\n")
        f.write(body)
    return head


def stage(low: IL.Lowering, input_set, lo: int, hi: int, out_dir, public_only: bool = False) -> Path:
    """``<out_dir>/circuit.txt`` (the composite circuit) and ``<out_dir>/inst-<n>.bin`` (``pub-<n>.bin`` for the verifier)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = compose(low, input_set.subcircuit)
    (out / "circuit.txt").write_text(text)
    if low.units.cut:
        IL.write_tables(out / "tables")
        if not (out / "circuit-forged.txt").exists():
            (out / "circuit-forged.txt").write_text(compose(low, input_set.subcircuit, forged=True))
    path = out / f"{'pub' if public_only else 'inst'}-{hi - lo}.bin"
    write(text, input_set, lo, hi, path, public_only)
    return path


def lowering_for_set(input_set) -> IL.Lowering:
    """The template's C-Flock lowering for a set (``lowering_for_set`` when the template's statement depends on the set)."""
    from verity_numerical.bench import lowerings

    mod = lowerings.registry("C-Flock").module(input_set.subcircuit.template.id)
    return mod.lowering_for_set(input_set) if hasattr(mod, "lowering_for_set") else mod.frame_lowering(input_set.subcircuit)


def main(argv=None) -> int:
    """``python -m verity_flock.circuit --input-set DIR --n N [N ...] --out DIR``: the composite circuit, then for each N the
    prover's ``inst-N.bin`` and the verifier's own ``pub-N.bin`` (instances [0, N)); prints the circuit's sha256 (the pin)."""
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
    print(json.dumps({"subcircuit": s.subcircuit.id, "unit": low.name, "pin": digest((Path(a.out) / "circuit.txt").read_text()), "n": a.n}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
