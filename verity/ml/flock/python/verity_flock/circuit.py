"""``verity/flock-circuit``: any lowered IR template (``flock-ir-unit/v2`` circuits, :mod:`ir_lower`) proved with its serving
rows committed in the circuit and nothing else public (``live/src/circuit.rs`` is the statement).

Rows are ``frame-v3-sha512`` leaves under ``hm96-sha512/row/v1`` (``verity.ml.commitments.rowleaf``): a row's inner digest is
``x = SHA-512(sha512_row_prefix || row)`` (``sha512/row/v1``), its hiding commit string ``b || c`` with ``b = x XOR M(key) y`` and
``c = SHA-512(salt_prefix || y)`` for a 192-byte salt ``y`` (``hm96.SHA512``, key its ``default_key``), and the frame leaf binds
``hm96.SHA512.tree_leaf(key, b || c)``. Both hashes are proved here: the rows and their salts are private. A port may instead
hold a bit row of any ``Port.bits`` bits below 2^64 (``sha512/row/v2``, its leaf ``hm96-sha512/row/v2``; META names each port's
schema and length in ``row_schemas``), or a segmented bit row (``sha512/row-seg/v1``, its leaf ``hm96-sha512/row-seg/v1``: each
1 KiB segment its own chain from its own prefix's midstate, the outer message's bytes copies of the segments' digests, its
last block the hm96 slot's): its ``ceil(bits / 16)`` u16 leaves the row's bits low first, its message after the
prefix ``P(bits)`` the row's ``ceil(bits / 8)`` bytes then SHA-512's padding. The verifiers derive from the length the
prefix, the blocks, and every message bit at or past it (the last byte's fill bits, then the padding: a whole padding block,
a last block of the row's bytes and padding, or an extra block of padding alone) as Δ constants.

A row is an input port's words or an output port's (an output row: the port's words, zero-padded to whole SHA-512 blocks, its
message bits copies of the bits the circuit computes). META ``ports`` lists the input rows, then one output row per output port;
``input_ports`` counts the input rows. An output is a value like an input: committed under its hiding leaf, never published, and
the committed output row is what a unit that reads it opens.

Private by default. The public data are the frame-v3-sha512 roots, each row's ``b || c`` (the serving tree's leaves are
functions of it), and the layout below. Every other word (every output word, every compression's chaining value, every word a
unit, a tail stage or a lookup reads from another) stays inside the witness, joined by copy constraints (Δ) inside a block: no
public value, and the verifier evaluates no part of the computation (a tail, the Gumbel noise of a public seed included, is a
slot type of the circuit or the template has no statement here). Private means absent from the public file: M0's transcript is
a function of the witness (``NON_ZK_PROOF``, its mask slots zeros), so only ``--zk`` targets hiding from the transcript.

One or more whole instances (VUs) per block. A block is the same for every block (Flock's ``A = I ⊗ A_0 + Δ``) and holds aligned
ranges of slots of a few types:

* ``sha512x3``: three independent SHA-512 compressions (:mod:`sha512_circuit`, 2^18 bits): a row's blocks after its constant
  first block but its last, chained by Δ (the first from the prefix's midstate, a Δ constant), the row's bits the message
  bits (a bit row's message bits past its length Δ constants);
* ``hm96``: one per row (``per_vu`` a power of two): the row's last compression (its last block: a ``sha512/row/v1`` row's
  padding block, a Δ constant; a bit row's last bytes, if any, and padding, its bits past the length Δ constants; its
  chaining value the previous compression's, or for a row of no other compression the midstate), then ``b || c`` from the
  salt; its ``bc`` output is the public region of the row;
* ``unit``: the template's identical units (2^unit_log bits), each input bit a copy of the message bit that hashes it (a bit
  row's leaf bits past its length: of the forced-zero row), each output row's message bit a copy of the output bit it
  commits (zero past the port's words);
* ``mask``: 2^14 free cells (``A = B = I``), reserved for zero-knowledge masks and filled by the prover's mask hook (zeros in
  M0), apart from the forced-zero padding (positions no range covers).

A padding instance (the block's unused instances and every block past the last) is the all-zero instance with zero salts; its
rows' ``b || c`` (its output rows hold the zero instance's outputs) and outputs are pinned here (``dummy``), so the verifier
knows every public value of every block.

The composite circuit (``flock-circuit``) is the statement's meaning: a ``META`` line (ports, ranges, each row's constants,
the wires, the units' leaf and output maps, the pinned dummy values, the leaf-commitment scheme, the program digests, the
partition digest), then each slot type's ``flock-ir-unit/v2`` circuit; the templated statement (``flock-circuit/tmpl``,
`compose`'s ``layout``) gives the unit as its ``flock-ir-tmpl/v2`` template in META instead. Its SHA-512 (:func:`digest`) is the pin. The verifier derives Δ, the public
regions and their values from it and from its own public instance file (no rows, no salts), never from the prover's; the
instance file's header carries each block's unit indices (the VUs of the partition it proves).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import ir_lower as IL

FORMAT = "flock-circuit"
STATEMENT = "verity/flock-circuit"
#: The templated statement (PRF-133, the Lean verifier's ``Tags.circuitTmpl``): `compose`'s ``layout``
TMPL_FORMAT = "flock-circuit/tmpl"
TMPL_STATEMENT = "verity/flock-circuit/tmpl"
SCHEME = "frame-v3-sha512/hm96-sha512"
ROW_SCHEMA = "hm96-sha512/row/v1"
INST_FORMAT = "flock-circuit-inputs"
#: Each row tree's frame domain: owner ``OWNER``, binding ``identity_digest_sha512(BINDING_TAG, {set, lo, hi, port, schema})``
#: (64 bytes), so a root is bound to its input set, its range of instances, its port and the port's row schema.
BINDING_TAG = "verity/flock-circuit/binding/v2"
OWNER = -1
#: The leaf-commitment scheme of the proof's own Merkle trees (Ligerito levels), pinned in the statement and the backend
#: identity and swappable: ``hm96-sha512/v1`` (``verity.ml.commitments.hm96.SHA512``, its PROTOCOL.md section 7) over the
#: SHA-512 column digest, leaf = ``tree_leaf(key, b || c)``, an opening = the column and its 192-byte salt (``opened_salts``).
#: Salts are the OS's own bytes, every leaf's 192 read a tree at a time in bulk (``zk_hooks::LeafSalts``; Daniel, 9 Oct, decision
#: 4(a)), never reused or stored, as ``salt_source`` says. The Lean verifier pins this object (``Flock/Tags.lean``'s
#: ``leafSchemeHm96Os``; statements composed up to verity ``b3727bd9`` name a per-proof ChaCha20 key, ``@b3727bd9``), and
#: ``live/src/circuit.rs``'s ``leaf_scheme`` is the same object. The level-0 salts are shared by
#: both reps (one root). Key: the scheme's ``default_key``, one constant of the statement for every
#: tree, pinned by its SHA-512. Never a witness: a prover choosing the key after commitment solves for one that opens a leaf to
#: any value.
LEAF_SCHEME = {"id": "hm96-sha512/v1", "inner": "sha512 column digest", "salt_bytes": 192,
               "salt_source": "os bytes (getrandom), 192 per leaf, each tree's read once, never reused or stored",
               "key": "default_key, a statement constant for every tree",
               "key_sha512": "afc5a60c85d8aa75cb4c7f726588d299a1712e1c6a621f09ca64809a5fedd744"
                             "a0f6b0201d05e82b29cd5b274e6c29670cd4e4ab10831911fb1f9e0c7019df22"}
#: The leaves of a statement whose trees need no hiding, a recursive session's inner statement (`verity_flock.rec_live`): its
#: firewall commits every tree top under a salt of its own and the openings stay in V*'s witness, so a leaf is the column's
#: SHA-512 digest itself (``HashKind::Sha512``, the Lean verifier's ``MerkleScheme.sha512``) and an opening carries no salt.
#: ZK off only: ``flock-circuit --zk`` refuses it (``live/src/circuit.rs``'s ``leaf_scheme_plain`` is the same object).
LEAF_SCHEME_PLAIN = {"id": "flock-leaf/sha512-unsalted", "inner": "sha512 column digest", "salt_bytes": 0,
                     "use": "a recursive session's inner statement: the gateway commits every tree top under a salt, ZK off"}
SHA_NET, HM_NET, PER_SLOT = "sha512x3", "hm96", 3
SHA_LOG = HM_LOG = 18
MASK_LOG = 14
K_MAX = 27
#: The partition is a named query of the program, P = Q(C) (Daniel, 2026-09-27): its digest is SHA-512 over the tag and
#: canon({program, query, version, params}). Until core's verity/partition/v1 lands (#111), the query here is this lowering's
#: own split of a VU into slots (its unit circuit, leaf and output maps, wires), under a provisional tag.
PARTITION_RULE = "flock-circuit/unit-cover/v0"
SALT_BYTES, BC_BYTES = 192, 128


PARTITION_V1 = "verity/partition/v1"
#: META's bindings: what the statement binds about the circuit, not the circuit. A unit's class (:func:`class_digest`) leaves
#: them out, so a partition can name the class its digest is bound beside.
BINDING_KEYS = ("partition", "program_sha512", "program_digests")


_COMPACT = json.JSONEncoder(sort_keys=True, separators=(",", ":"))
_HASH_CHUNK = 1 << 24
#: compose's stand-in for META ``template``, whose lines it serializes on their own: a template's META is most of the file
_SPLICE = "\0template\0"


def meta_of(text: str) -> dict:
    """The circuit file's META (its second line), parsed without copying the rest of the file."""
    i = text.index("\n") + 1
    return json.loads(text[i + 5:text.index("\n", i)])


def _sha512(pieces) -> hashlib._Hash:
    """SHA-512 of the UTF-8 of ``pieces`` concatenated, encoded :data:`_HASH_CHUNK` characters at a time."""
    h, buf, size = hashlib.sha512(), [], 0
    for s in pieces:
        for k in range(0, len(s), _HASH_CHUNK):
            buf.append(s[k:k + _HASH_CHUNK])
            size += len(buf[-1])
            if size >= _HASH_CHUNK:
                h.update("".join(buf).encode())
                buf, size = [], 0
    h.update("".join(buf).encode())
    return h


def class_digest(text: str) -> str:
    """The class of every instance the circuit proves: SHA-512 of the circuit file with META's :data:`BINDING_KEYS` removed
    (META re-serialized compact with sorted keys)."""
    i = text.index("\n") + 1
    j = text.index("\n", i)
    meta = {k: v for k, v in json.loads(text[i + 5:j]).items() if k not in BINDING_KEYS}
    rest = (text[k:k + _HASH_CHUNK] for k in range(j, len(text), _HASH_CHUNK))
    return _sha512(itertools.chain([text[:i], "META "], _COMPACT.iterencode(meta), rest)).hexdigest()


def set_statics(input_set) -> dict:
    """The instance statics of a set's instances (a statement covers one binding: attention's T)."""
    sub = input_set.subcircuit
    return dict(sub.instance_statics_of({p.name: input_set.port(p.name, 0, 1)[0] for p in sub.inputs}))


def program_digests(sub, statics: dict | None = None) -> dict[str, str]:
    """META ``program_digests``: core's ``program_digest`` of the subcircuit's Definition at the instance statics, keyed by its
    descriptor id (``verity.ml.circuits.codec``'s, the id the canonical partition and the verifier name a template by); a silicon
    reference no core Definition wraps keeps its own key."""
    from verity.ml.circuits.codec import _spec_id

    digests = sub.program_digests([statics or {}])
    try:
        key = _spec_id(sub.definition(**(statics or {})))
    except (AssertionError, AttributeError, TypeError, ValueError):
        return digests
    (value,) = digests.values()
    return {key: value}


def program_sha512(sub, statics: dict | None = None) -> str:
    """META ``program_sha512``: core's ``program_digest`` of the subcircuit's Definition at its default statics; a caller that
    builds the partition over another Program passes that Program's digest instead."""
    from verity.ml.circuits.codec import encode_program, program_digest
    from verity.ml.circuits.program import Program

    try:
        desc = encode_program(Program(sub.definition(**(statics or {}))))
    except (TypeError, ValueError, AttributeError):
        # a silicon reference no core Definition wraps: its digests as the program
        return hashlib.sha512(json.dumps(sub.program_digests([statics or {}]), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return program_digest(desc)


def partition_digest(obj: dict) -> str:
    """``SHA-512(rule || 0 || canon({program, query, version, params}))``, 64 bytes: ``verity/partition/v1``'s shape under the
    provisional :data:`PARTITION_RULE`."""
    return hashlib.sha512(PARTITION_RULE.encode() + b"\0" + json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _pow2(x: int) -> int:
    return 1 << max(0, (x - 1).bit_length())


def _log2(x: int) -> int:
    assert x & (x - 1) == 0 and x > 0, x
    return x.bit_length() - 1


@dataclass
class Port:
    """An input port's rows: ``words`` u16 leaves, framed as ``sha512/row/v1`` x rows of u16 words (whole SHA-512 blocks), or
    (``bits``) as core's bit row ``sha512/row/v2`` of ``bits`` bits (``verity.ml.commitments.rowleaf``): its leaves the row's bits
    low first, ``words = ceil(bits / 16)``, the bits past ``bits`` zero."""

    name: str
    words: int
    bits: int | None = None
    #: a segmented bit row (``sha512/row-seg/v1``), its schema in place of ``sha512/row/v2``
    seg: bool = False

    @property
    def n_bits(self) -> int:
        """The row's length in bits (a ``sha512/row/v1`` row's: its words')."""
        return 16 * self.words if self.bits is None else self.bits

    @property
    def bytes(self) -> int:
        """The row bytes its digest hashes after the prefix: ``2 * words``, or a bit row's ``ceil(bits / 8)``."""
        return 2 * self.words if self.bits is None else -(-self.bits // 8)

    @property
    def comps(self) -> int:
        """Compressions after the constant first block: the message's blocks but its last, which the hm96 slot compresses (the
        row's last bytes, if any, then SHA-512's padding: ``ceil((bytes + 17) / 128)`` blocks in all)."""
        if self.seg:
            from verity.ml.commitments.rowleaf import sha512_row_seg_segments

            segs = sha512_row_seg_segments(self.bits)
            return sum(-(-((m + 7) // 8 + 17) // 128) for m in segs) + -(-(64 * len(segs) + 17) // 128) - 1
        return -(-(self.bytes + 17) // 128) - 1

    def chain_start(self, k: int) -> bool:
        """Whether compression ``k`` (``comps``: the hm96 slot's) starts a chain from a constant midstate: a row's first, or a
        segmented row's segment's or outer message's first. Every other reads the previous compression's output."""
        if not self.seg:
            return k == 0
        from verity.ml.commitments.rowleaf import sha512_row_seg_segments

        first = 0
        for m in sha512_row_seg_segments(self.bits):
            if k <= first:
                return k == first
            first += -(-((m + 7) // 8 + 17) // 128)
        return k == first

    def message(self) -> bytes:
        """The message after the prefix with every row bit zero: ``bytes`` zero bytes, then SHA-512's padding. A message bit is
        the row's bit ``8 j + t`` (bit ``t`` of byte ``j``) below ``n_bits``, and this constant from there on (the fill bits
        of the last row byte, then the padding)."""
        from . import sha512_circuit as S

        return bytes(self.bytes) + S.pad_tail(128 + self.bytes)

    @property
    def schema(self) -> str:
        """The frame leaf's schema: ``hm96-sha512/row/v1``, a bit row's ``hm96-sha512/row/v2``, a segmented row's
        ``hm96-sha512/row-seg/v1``."""
        from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG, SCHEMA_HM96_SHA512_ROW_V2

        return ROW_SCHEMA if self.bits is None else SCHEMA_HM96_SHA512_ROW_SEG if self.seg else SCHEMA_HM96_SHA512_ROW_V2

    def digest(self, row: bytes) -> bytes:
        """The row's inner digest of its ``bytes`` row bytes (``sha512/row/v1``, ``/v2`` or ``sha512/row-seg/v1``)."""
        from verity.ml.commitments.rowleaf import sha512_row_seg_digest

        return sha512_row_seg_digest(row, self.bits) if self.seg else self.layout().digest(row)

    def prefix(self) -> bytes:
        """The row's first block: core's prefix, or a segmented row's outer ``Q(bits)``."""
        from verity.ml.commitments.rowleaf import sha512_row_seg_prefix

        return sha512_row_seg_prefix(self.bits) if self.seg else self.layout().prefix

    @property
    def tail_bytes(self) -> int:
        """The bytes the padding follows: the row's, or a segmented row's outer message's (its segments' digests)."""
        if self.seg:
            from verity.ml.commitments.rowleaf import sha512_row_seg_segments

            return 64 * len(sha512_row_seg_segments(self.bits))
        return self.bytes

    def leaf(self):
        from verity.ml.commitments.frame_v3 import BitRowLeaf, RowLeaf
        from verity.ml.commitments.rowleaf import ROLE_X

        return RowLeaf("sha512", ROLE_X, 16, self.words) if self.bits is None else BitRowLeaf(self.bits)

    def layout(self):
        """The row's inner digest layout (core's: its prefix, and its length, which is the row's own); a segmented row has none."""
        if self.seg:
            raise ValueError(f"port {self.name}: a segmented row's digest is two levels of SHA-512, not one layout")
        return self.leaf().layout()


def _port(name: str, words: int, bits: int | None = None, seg: bool = False) -> Port:
    """A ``sha512/row/v1`` port of whole SHA-512 blocks, or a bit row of any u64 length in its ``ceil(bits / 16)`` leaves (the
    verifiers derive its fill bits and padding as Δ constants)."""
    if bits is None:
        if words <= 0 or (2 * words) % 128:
            raise ValueError(f"port {name}: {2 * words}-byte rows are not whole SHA-512 blocks (sha512/row/v1)")
        return Port(name, words)
    if isinstance(bits, bool) or not isinstance(bits, int) or not 0 <= bits < 1 << 64:
        raise ValueError(f"port {name}: a row's length is a u64 number of bits, not {bits!r}")
    if words != -(-bits // 16):
        raise ValueError(f"port {name}: a row of {bits} bits is {-(-bits // 16)} u16 leaves, not {words}")
    return Port(name, words, bits, seg)


def _meta_ports(meta: dict) -> list[Port]:
    """META's rows, each a bit row (or a segmented one) of the length its ``row_schemas`` entry names (absent: ``sha512/row/v1``
    rows)."""
    from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG, SCHEMA_HM96_SHA512_ROW_V2

    if "row_schemas" not in meta:
        return [Port(nm, w) for nm, w, _ in meta["ports"]]
    if any(s not in (SCHEMA_HM96_SHA512_ROW_V2, SCHEMA_HM96_SHA512_ROW_SEG) for s, _bits in meta["row_schemas"]):
        raise ValueError(f"row_schemas: each port's is {SCHEMA_HM96_SHA512_ROW_V2} or {SCHEMA_HM96_SHA512_ROW_SEG}")
    return [_port(nm, w, bits, s == SCHEMA_HM96_SHA512_ROW_SEG) for (nm, w, _), (s, bits) in zip(meta["ports"], meta["row_schemas"], strict=True)]


def out_row(name: str, words: int, bit_row: bool = False) -> Port:
    """Output port ``name``'s row: its ``words`` output words, zero-padded to whole SHA-512 blocks (64 words each); in a
    statement of bit rows (``row_schemas``), a bit row of exactly those words' bits."""
    w = -(-words // 64) * 64
    return Port(name, w, 16 * w if bit_row else None)


def out_rows(outs: dict, out_ports: list[tuple[str, int]], rows: list[Port], counts: list[int] | None = None) -> dict:
    """Each output port's words ``{port: (n, words)}`` as its output row's (zero past the port's words); ``counts`` (META
    ``out_row_ports``): each row's run of output ports, their words concatenated in port order. Output ports past the rows'
    runs (META ``pinned_outputs``) have no row."""
    counts = counts if counts is not None else [1] * len(rows)
    out, at = {}, 0
    for k, p in zip(counts, rows, strict=True):
        run = out_ports[at:at + k]
        v = np.concatenate([np.asarray(outs[nm], dtype=np.uint16).reshape(-1, w) for nm, w in run], axis=1)
        out[p.name] = np.zeros((len(v), p.words), dtype=np.uint16)
        out[p.name][:, :v.shape[1]] = v
        at += k
    return out


def _out_ports(outs: list, bit_rows: bool, seg_out_rows, v2_out_rows=()) -> tuple[list[Port], list[int]]:
    """The output rows and each one's run of output ports: one row per port (``out_row``), each run in ``seg_out_rows``
    (consecutive output ports, in order) one ``sha512/row-seg/v1`` row of exactly their words, and each run in ``v2_out_rows``
    (one port or several) one ``sha512/row/v2`` row of exactly their words, each named after its first."""
    names = [nm for nm, _ in outs]
    starts, covered = {}, set()
    for seg, runs in ((True, seg_out_rows), (False, v2_out_rows)):
        for run in runs:
            i = names.index(run[0]) if run and run[0] in names else -1
            if i < 0 or list(run) != names[i:i + len(run)] or covered & set(run):
                raise ValueError(f"output row {list(run)}: a run of consecutive output ports, each in one row")
            starts[i] = (len(run), seg)
            covered |= set(run)
    rows, counts, j = [], [], 0
    while j < len(outs):
        if j in starts:
            k, seg = starts[j]
            w = sum(x for _, x in outs[j:j + k])
            rows.append(Port(outs[j][0], w, 16 * w, seg))
            counts.append(k)
            j += k
        else:
            rows.append(out_row(*outs[j], bit_row=bit_rows))
            counts.append(1)
            j += 1
    return rows, counts


def _row_consts(p: Port) -> dict:
    """A row's constants: its schema's first block (core's prefix), the chaining value after it, SHA-512's padding after its
    bytes (a whole block for ``sha512/row/v1``)."""
    from . import sha512_circuit as S

    prefix = p.prefix()
    return {"prefix": prefix.hex(), "midstate": [f"{w:016x}" for w in S.midstate(prefix)], "pad": S.pad_tail(128 + p.tail_bytes).hex()}


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
    for si, st in enumerate(tail.stages):
        for s in range(st.count):
            for j, tag in enumerate(st.tags(s)[1]):
                if tag[0] == "cut":
                    producer[tag[1]] = (f"stage{si}", s, j)
    public = dict(getattr(U, "public", []))
    local, seen = [], {}
    for j, (table, _) in enumerate(tail.lookups):
        local.append(seen.get(table, 0))
        seen[table] = seen.get(table, 0) + 1
    wires = []
    for u, src in enumerate(U.in_src):
        for j, (kind, k) in enumerate(src):
            if kind == "cut" and k not in public:
                wires.append(["unit", u, j, *producer[k]])
    for i, st in enumerate(tail.stages):
        prev = tail.stages[i - 1] if i else None
        for s in range(st.count):
            for j, tag in enumerate(st.tags(s)[0]):
                if tag[0] in ("pass", "ctx", "lookup") and (st.count != 1 or (tag[0] != "lookup" and prev.count != 1)):
                    raise NotImplementedError(f"stage{i}: a repeated stage reads cut words only (no pass, context or lookup port)")
                if tag[0] in ("pass", "ctx"):
                    wires.append([f"stage{i}", 0, j, f"stage{i - 1}", 0, prev.outs.index(tag)])
                elif tag[0] == "lookup":
                    wires.append([f"stage{i}", 0, j, f"lookup:{tail.lookups[tag[1]][0]}", local[tag[1]], 0])
                else:
                    wires.append([f"stage{i}", s, j, *producer[tag[1]]])
    for q, (table, si) in enumerate(tail.lookups):
        wires.append([f"lookup:{table}", local[q], 0, f"stage{si}", 0, tail.stages[si].outs.index(("index", q))])
    del n_ret
    return wires


def _ret_ports(st, slotted: bool = False) -> list:
    """META ``out_net_ports``: each output word's port of the last stage, in output order; a repeated stage's, or every stage's
    when ``slotted``, as ``[slot, port]``."""
    at = {tag[1]: (s, j) for s in range(st.count) for j, tag in enumerate(st.tags(s)[1]) if tag[0] == "ret"}
    if sorted(at) != list(range(len(at))):
        raise ValueError("the last stage's returned outputs are not each output word once")
    return [at[o][1] if st.count == 1 and not slotted else list(at[o]) for o in range(len(at))]


def _row_slots(ports: list[Port]) -> tuple[int, list[list[tuple[int, int]]]]:
    """Per VU: the sha512x3 slots, and each port's compressions as ``(slot, sub)`` (three per slot, ports in order)."""
    at, where = 0, []
    for p in ports:
        where.append([divmod(at + k, PER_SLOT) for k in range(p.comps)])
        at += p.comps
    return -(-at // PER_SLOT), where


def _row_wires(ports: list[Port]) -> list:
    """Each row's chain (compression k's ``cv`` from compression k-1's ``out``) and its hm96 slot's ``cv`` (from the last; a
    row of no compression gives its hm96 slot the midstate, a Δ constant). A segmented row's chains restart at each chain start
    (``Port.chain_start``), whose midstate is a Δ constant, and the outer message's digest bits are Δ's copies, not wires."""
    _, where = _row_slots(ports)
    wires = []
    for p, w in enumerate(where):
        for k in range(1, len(w)):
            if not ports[p].chain_start(k):
                wires.append([SHA_NET, w[k][0], 2 * w[k][1], SHA_NET, w[k - 1][0], w[k - 1][1]])
        if w and not ports[p].chain_start(len(w)):
            wires.append([HM_NET, p, 0, SHA_NET, w[-1][0], w[-1][1]])
    return wires


def _split_leaves(leaves: list[list[int]], in_ports: list[tuple[str, int]], public: dict) -> tuple[list, list, list]:
    """Each flat input leaf as ``(public, index)``: a row word's index among the rows' words, or a public word's among the public
    ports' words; then the units' ``leaves_in`` (a public leaf there a zero leaf, -1) and ``leaves_public``."""
    owner, r, p = [], 0, 0
    for nm, w in in_ports:
        owner += [(nm in public, (p if nm in public else r) + k) for k in range(w)]
        p, r = (p + w, r) if nm in public else (p, r + w)
    li = [[-1 if x < 0 or owner[x][0] else owner[x][1] for x in u] for u in leaves]
    lp = [[owner[x][1] if x >= 0 and owner[x][0] else -1 for x in u] for u in leaves]
    return owner, li, lp


def aligned_blocks(words: list[int]) -> list[tuple[int, int]]:
    """Distinct words as aligned power-of-two blocks ``(w, e)`` (``w`` a multiple of ``2^e``), in order: each maximal run of
    consecutive words cut from its start, every block the largest aligned one that fits in what is left of the run."""
    out, ws = [], sorted(set(words))
    i = 0
    while i < len(ws):
        a = b = ws[i]
        while i < len(ws) and ws[i] == b:
            i, b = i + 1, b + 1
        while a < b:
            e = (b - a).bit_length() - 1
            while a % (1 << e):
                e -= 1
            out.append((a, e))
            a += 1 << e
    return out


def public_blocks(group: tuple, li: list[list[int]], lp: list[list[int]]) -> list[tuple[int, int]]:
    """The ``in`` regions' words in a unit slot (PROTOCOL.md §16.10): the 128-bit words holding a public leaf in some unit, as
    :func:`aligned_blocks`, inside the leaf group, every leaf in them a public or a zero leaf in every unit."""
    col, words, bits = group
    if any(b != 16 for b in bits):
        raise ValueError("public inputs: the unit's leaves are not 16-bit words")
    pub = {col + 16 * j // 128 for u in lp for j, x in enumerate(u) if x >= 0}
    if not pub:
        raise ValueError("public inputs: no unit reads a public word")
    blocks = aligned_blocks(list(pub))
    for w, e in blocks:
        if w < col or w + (1 << e) > col + words:
            raise ValueError(f"public inputs: the in region's words {w}..{w + (1 << e)} leave the unit's leaf group")
    for j in range(len(bits)):
        if col + 16 * j // 128 in pub and any(lp[u][j] < 0 and li[u][j] >= 0 for u in range(len(li))):
            raise ValueError(f"public inputs: leaf {j} reads a row word inside word {col + 16 * j // 128}, which holds a public "
                             f"word: lay the public ports out as whole words")
    return blocks


def compose(low: IL.Lowering, sub, vus_per_block: int | None = None, forged: bool = False, partition: str | None = None,
            program: str | None = None, statics: dict | None = None, port_bits: list | None = None,
            public_inputs: dict | None = None, seg_ports: frozenset[str] = frozenset(), seg_out_rows: tuple = (),
            v2_out_rows: tuple = (), links: list | None = None, definition: str | None = None,
            leaf_scheme: dict | None = None, extra_meta: dict | None = None, tail=None, layout: str | None = None,
            pinned_outputs: tuple = ()) -> str:
    """The ``flock-circuit`` text for a template lowered to identical units and, for a cut lowering, its tail in the circuit
    (:mod:`tail`: stage circuits and MUFU lookup slots, wired to the units' cut ports). ``partition``: the caller's
    ``verity/partition/v1`` digest (64-byte hex), else the provisional unit cover; ``program``: the Program's SHA-512 the
    partition names, else :func:`program_sha512` of the subcircuit. ``port_bits``: each input port's bit row length (its
    ``sha512/row/v2`` rows; the output rows then bit rows of all their words), which META names per port (``row_schemas``)
    in place of the one ``row_schema``; ``seg_ports``: the input ports among them that are segmented rows (``sha512/row-seg/v1``,
    PoUW's per-call role-X rows); ``seg_out_rows``: runs of consecutive output ports each committed as one segmented row of
    exactly their words, named after its first (META ``out_row_ports``), so that a unit reading the row as one input port
    has the writer's leaf; ``v2_out_rows``: runs (one port or several) each committed as one ``sha512/row/v2`` row of exactly
    their words (PoUW's ok, tile digest and role-W rows), in place of a lone port's row zero-padded to whole blocks.
    ``leaf_scheme``: the proof's own Merkle leaves (META ``leaf_scheme``), :data:`LEAF_SCHEME` by default or
    :data:`LEAF_SCHEME_PLAIN`. ``pinned_outputs``: the last output ports' names (META ``pinned_outputs``), which have no
    row: each of their bits Δ's copy of the unit's forced-zero row, so the proof holds only where they are zero (a writer's
    ``S[j] XOR y``); bit rows only, and the all-zero instance's must be zero.

    ``public_inputs`` (``{port: {"seed": ..., "derivation": ...}}``): input ports that are not rows but public words the verifier
    recomputes from public data, each naming its seed and derivation (PROTOCOL.md §16.10). The units read them from ``in``
    regions (META ``public_inputs``, ``leaves_public``), which must be whole words of the unit's leaf group (:func:`public_blocks`).
    The data guard counts a port whose derivation it does not list as data (``verity.ml.circuits.operations.public_data``).

    ``links`` (``[{"port", "leaves": [lo, hi]}]``, META ``links``, PROTOCOL.md §16.13): input rows that hold template parameter
    leaves ``[lo, hi)`` as an output row holds outputs (a word per leaf), so that a reader whose leaves are another unit's outputs
    commits its writer's output row; the verifiers compare the two rows' ``b || c``. ``definition``: META ``definition``, the
    descriptor id of the Definition the unit lowers, which a template query may list (a unit class's own template id is
    ``unit-class/...``). ``extra_meta``: keys a statement adds to META for its verifier (`rec_residuals.HOLO_KEY`), none of
    them one META already has.

    ``tail`` (a :class:`tail.Tail`): the stages as given, in place of :func:`tail.lower_tail`'s, for a lowering whose split is
    its own (`multi_unit`; `read_slots`, whose tail carries its wires): a stage of ``count`` slots per VU is a second repeated
    unit type, wired slot by slot (META ``per_vu``), and returned outputs on a repeated last stage (or a ``slotted_rets`` tail's)
    are ``out_net_ports`` of ``[slot, port]``.

    ``layout`` (a layout mode, `tmpl.mode_of`; TMPL1's is ``iv/hybrid/4``): the templated statement
    (:data:`TMPL_STATEMENT`, PRF-133, the Lean verifier's ``HmOut.parseTmpl``), the same statement with its unit given as
    the unit's ``flock-ir-tmpl/v2`` template (`tmpl.emit` of the flat unit, or a structural lowering's own template,
    `class_statement.TmplLowering`, whose only statement this is and whose ``unit_sha256`` is over the template's text):
    META ``template`` its lines and ``layout`` the mode, in place of the ``CIRCUIT unit`` section and the flat META
    ``template``. The verifier unrolls the template and commits its columns at the layout's positions (`tmpl.layout`), so
    the unit's range and ``unit_log`` are that net's slot. A unit class's ``definition`` has no templated form."""
    for name, v in (("partition", partition), ("program", program)):
        if v is not None and (len(v) != 128 or v != v.lower() or any(ch not in "0123456789abcdef" for ch in v)):
            raise ValueError(f"{name}: a SHA-512 digest is 128 lowercase hex digits")
    from . import sha512_circuit as S
    from . import tail as TL

    U = low.units
    if U.tail_rets and not U.cut:
        raise NotImplementedError(f"{low.name}: returned outputs without a cut tail; the statement has no verifier-evaluated form")
    if any(p.dtype != "u16" for p in (*sub.inputs, *sub.outputs)):
        raise ValueError(f"{low.name}: frame-v3 rows here are u16 words")
    if tail is None:
        tail = TL.lower_tail(low, forged) if U.cut else None
    stages = [TL.stage_layout(st) for st in tail.stages] if tail else []
    lookups: dict[str, int] = {}
    for table, _ in (tail.lookups if tail else []):
        lookups[table] = lookups.get(table, 0) + 1
    lv = low.leaves
    public_inputs = dict(public_inputs or {})
    in_ports = [(p.name, w) for p, (w, _) in zip(sub.inputs, lv["in_ports"])]
    if set(public_inputs) - {nm for nm, _ in in_ports}:
        raise ValueError(f"public inputs {sorted(set(public_inputs) - {nm for nm, _ in in_ports})}: not input ports")
    for nm, v in public_inputs.items():
        if set(v) != {"seed", "derivation"} or not all(isinstance(v[k], str) and v[k] for k in v):
            raise ValueError(f"public input {nm}: it names its recomputation, a non-empty seed and derivation")
    if port_bits is not None and None in [b for (nm, _w), b in zip(in_ports, port_bits) if nm not in public_inputs]:
        raise ValueError(f"{low.name}: bit rows on every input row or on none")
    bits_of = port_bits or [None] * len(sub.inputs)
    if seg_ports and (port_bits is None or not seg_ports <= {p.name for p in sub.inputs}):
        raise ValueError(f"{low.name}: segmented rows are bit rows of input ports: {sorted(seg_ports)}")
    ins = [_port(nm, w, b, nm in seg_ports) for (nm, w), b in zip(in_ports, bits_of, strict=True) if nm not in public_inputs]
    pins = [{"port": nm, "words": w, **public_inputs[nm]} for nm, w in in_ports if nm in public_inputs]
    owner, leaves_in, leaves_public = _split_leaves(lv["in"], in_ports, public_inputs)
    if pins:
        public_blocks(low.layout.in_groups[0], leaves_in, leaves_public)
    outs = [[p.name, w] for p, (w, _) in zip(sub.outputs, lv["out_ports"])]
    if (seg_out_rows or v2_out_rows) and port_bits is None:
        raise ValueError(f"{low.name}: output rows of exactly their words are bit rows (row_schemas)")
    k_pin = len(pinned_outputs)
    if pinned_outputs and (port_bits is None or layout is not None or list(pinned_outputs) != [nm for nm, _ in outs[len(outs) - k_pin:]]):
        raise ValueError(f"{low.name}: pinned outputs {list(pinned_outputs)} are the last output ports, of a flat statement of bit rows")
    orows, counts = _out_ports(outs[:len(outs) - k_pin], bool(port_bits), seg_out_rows, v2_out_rows)
    ports = ins + orows
    if links:
        named = {p.name: p for p in ins}
        seen: list[tuple[int, int]] = []
        for x in links:
            p, lohi = named.get(x.get("port")), x.get("leaves")
            if set(x) != {"port", "leaves"} or p is None or not (isinstance(lohi, list) and len(lohi) == 2 and 0 <= lohi[0] < lohi[1]):
                raise ValueError(f"link {x}: an input row and a leaf range [lo, hi)")
            if len(orows) != 1 or p.bits is not None or p.words != orows[0].words or any(a < lohi[1] and lohi[0] < b for a, b in seen):
                raise ValueError(f"link {x}: a u16 row of the one output row's words, its leaves disjoint from every other link's")
            seen.append(tuple(lohi))
    template = getattr(low, "template", None)
    if template is not None and layout is None:
        raise ValueError(f"{low.name}: a unit given as its template has only the templated statement: name its layout")
    useful = None if template is not None else low.layout.useful
    if layout is not None:
        from . import tmpl as TM

        if definition is not None:
            raise ValueError(f"{low.name}: a unit class (META definition) has no templated statement: META template is its unit's")
        if template is None:
            template, stats = TM.emit(TM.of_layout(low.layout, low.name))
            order = stats["order"]
        else:
            order = TM.ordered(template)
        if not order["ok"]:
            raise ValueError(f"{low.name}: the unit's template is not in the order Tmpl.parse checks: {order}")
        useful = TM.layout(template, layout)[template.root].size
    ul = _slot_log(useful)
    if useful == 1 << ul:
        raise ValueError(f"{low.name}: a unit fills its slot; a zero leaf needs a forced-zero row")
    lk = {t: {"table": t, "sha256": IL.TABLES[t], "in_bits": TL.TABLE_BITS[t], "out_bits": TL.VALUE_BITS, "lo_bits": LOOKUP_LO[TL.TABLE_BITS[t]]}
          for t in lookups}
    n_sha, _ = _row_slots(ports)
    if n_sha == 0:
        raise ValueError(f"{low.name}: no row has a compression before its last block (the {SHA_NET} range, the pin's, would hold no slot)")
    n_hm = _pow2(len(ports))

    def fits(G: int):
        # a range no region reads is packed (the compressions, the lookups, and the units when the tail returns the outputs)
        items = [("unit", SHA_LOG, G * n_sha, {"net": SHA_NET, "packed": True}), ("unit", HM_LOG, G * n_hm, {"net": HM_NET}),
                 ("unit", ul, G * U.n, {"net": "unit", **({"packed": True} if U.tail_rets and not pins else {})})]
        for i, lay in enumerate(stages):
            items.append(("unit", _slot_log(lay.useful), G * tail.stages[i].count, {"net": f"stage{i}"}))
        for t, c in lookups.items():
            L = lk[t]
            items.append(("unit", _slot_log(lookup_rows(L["in_bits"], L["lo_bits"], L["out_bits"])), G * c, {"net": f"lookup:{t}", "packed": True}))
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
    tmpl_text = TM.write(template) if template is not None else None
    per_vu = {"unit": U.n, SHA_NET: n_sha, HM_NET: n_hm, **{f"stage{i}": tail.stages[i].count for i in range(len(stages))},
              **{f"lookup:{t}": c for t, c in lookups.items()}}
    tail_wires = (tail.wires if tail.wires is not None else _wires(low, tail)) if tail else []
    public = dict(getattr(U, "public", []))
    # a public cut word (attention's query) is a row word: its unit input copies the row's message bits, zero above 16
    leaf_cuts = [["unit", u, j, public[k]] for u, src in enumerate(U.in_src) for j, (kind, k) in enumerate(src) if kind == "cut" and k in public]
    if any(owner[x[3]][0] for x in leaf_cuts):
        raise NotImplementedError(f"{low.name}: a cut input read from a public input port")
    leaf_cuts = [[*x[:3], owner[x[3]][1]] for x in leaf_cuts]
    out_net = f"stage{len(stages) - 1}" if U.tail_rets else "unit"
    out_net_ports = _ret_ports(tail.stages[-1], tail.slotted_rets) if U.tail_rets else []
    programs = program_digests(sub, statics)
    unit = tmpl_text if hasattr(low, "template") else _expanded(low.layout, low.name)
    unit_sha256 = low.sha256_of(unit)
    stage_texts = [_expanded(lay, f"{low.name}/tail/stage{i}") for i, lay in enumerate(stages)]
    cover = {"program": programs, "query": f"flock-circuit/unit-cover:{low.name}", "version": 0,
             "params": {"unit_sha256": unit_sha256, "units_per_vu": U.n, "leaves_in": lv["in"], "leaves_out": lv["out"], "tail_wires": tail_wires,
                        "stages": [digest(t) for t in stage_texts], "lookups": lk}}
    row_schema = {"row_schemas": [[p.schema, p.bits] for p in ports]} if port_bits else {"row_schema": ROW_SCHEMA}
    meta = {"format": FORMAT, "statement": STATEMENT, "scheme": SCHEME, **row_schema, "template": sub.template.id,
            "subcircuit": sub.id, "unit": low.name, "unit_sha256": unit_sha256, "vus_per_block": G, "k_log": k_log,
            "ports": [[p.name, p.words, p.comps] for p in ports], "input_ports": len(ins), "out_ports": outs, "ranges": ranges,
            "rows": [_row_consts(p) for p in ports], "units_per_vu": U.n, "unit_log": ul, "leaves_in": leaves_in, "leaves_out": lv["out"],
            **({"public_inputs": pins, "leaves_public": leaves_public} if pins else {}),
            "row_key_sha512": LEAF_SCHEME["key_sha512"], "dummy": _dummy(low, in_ports, ins, outs, orows, sub, counts),
            "leaf_scheme": leaf_scheme or LEAF_SCHEME,
            **({"out_row_ports": counts} if seg_out_rows or max(counts, default=1) > 1 else {}),
            **({"pinned_outputs": list(pinned_outputs)} if pinned_outputs else {}),
            **({"links": links} if links else {}), **({"definition": definition} if definition else {}),
            # only for a template that needs them (attention, GEMM): absent means the unit returns the outputs and no leaf cut
            **({"out_net": out_net, "out_net_ports": out_net_ports} if U.tail_rets else {}), **({"leaf_cuts": leaf_cuts} if leaf_cuts else {}), "mask_log": MASK_LOG, "per_vu": per_vu, "lookups": lk,
            "wires": _row_wires(ports) + tail_wires, "program_digests": programs,
            "program_sha512": program or program_sha512(sub, statics),
            "partition": ({"rule": PARTITION_V1, "digest": partition} if partition else
                          {"rule": PARTITION_RULE, "query": cover["query"], "version": cover["version"], "digest": partition_digest(cover)})}
    if pinned_outputs and any(meta["dummy"]["outputs"][sum(w for _, w in outs[:len(outs) - k_pin]):]):
        raise ValueError(f"{low.name}: the all-zero instance's pinned outputs are not zero (it pads every block)")
    if template is not None:
        meta.update(format=TMPL_FORMAT, statement=TMPL_STATEMENT, template=_SPLICE, layout=layout)
    if extra_meta:
        if clash := sorted(set(extra_meta) & set(meta)):
            raise ValueError(f"extra_meta: {clash} are META's own keys")
        meta.update(extra_meta)
    meta_json = [json.dumps(meta, sort_keys=True, separators=(",", ":"))]
    if template is not None:
        a, b = meta_json[0].split(json.dumps(_SPLICE))
        meta_json = [a, json.dumps(tmpl_text.splitlines(), separators=(",", ":")), b]
    texts = [] if template is not None else [("unit", unit)]
    texts += [(SHA_NET, _expanded(IL.layout(S.compression3(), ["cv0", "msg0", "cv1", "msg1", "cv2", "msg2"], ["out0", "out1", "out2"]), SHA_NET)),
              (HM_NET, _expanded(IL.layout(S.hm96_row(), ["cv", "pad", "y"], ["bc"]), HM_NET))]
    texts += [(f"stage{i}", t) for i, t in enumerate(stage_texts)]
    return "".join([f"{meta['format']} {low.name}\nMETA ", *meta_json, "\n",
                    *(s for nm, t in texts for s in (f"CIRCUIT {nm} {t.count(chr(10))}\n", t))])


def _expanded(lay: IL.Layout, name: str) -> str:
    """A slot type's expanded circuit as ``flock-ir-unit/v2`` text (``ir_lower``'s writer)."""
    return IL.netlist(lay, name, None, None)



def _commit_strings(ports: list[Port], rows: dict, salts: dict) -> dict[str, list[bytes]]:
    """Each row's ``b || c`` (``hm96.SHA512`` over its row digest, ``sha512/row/v1``, ``/v2`` or ``row-seg/v1`` of the leaves' first
    ``Port.bytes`` bytes, and its salt). Refuses a bit row whose leaves have a bit set past its length."""
    from verity.ml.commitments import hm96

    inst = hm96.SHA512
    out = {}
    for p in ports:
        r = np.ascontiguousarray(rows[p.name], dtype="<u2")
        if p.bits is not None and 16 * p.words > p.bits:
            if np.any(r[:, -1] >> np.uint16(p.bits - 16 * (p.words - 1))):
                raise ValueError(f"port {p.name}: a row of {p.bits} bits has a bit set past its length")
        out[p.name] = [inst.commit_string(inst.default_key, p.digest(r[i].tobytes()[:p.bytes]), salts[p.name][i]) for i in range(len(r))]
    return out


def _dummy(low: IL.Lowering, in_ports: list[tuple[str, int]], ins: list[Port], out_ports: list, orows: list[Port], sub=None,
           counts: list[int] | None = None) -> dict:
    """The all-zero instance's output words (every output leaf) and every row's ``b || c`` under zero salts (its zero input rows,
    then its output rows), as the padding instances hold (a tail that returns the outputs: the template's reference evaluation of
    the zero instance, or the lowering's own where it evaluates the tail, ``evaluates_rets``). Its public inputs are zero too."""
    n_in = sum(w for _, w in in_ports)
    zero = np.zeros((1, n_in), dtype=np.uint64)
    reads_cut = any(k == "cut" for src in low.units.in_src for k, _ in src)
    outs, _, ok = low.evaluate(zero, IL.cut_words(low, zero) if low.units.cut and reads_cut else None)
    assert int(ok) & 1, "the all-zero instance does not satisfy the unit rows"
    if low.units.tail_rets and not getattr(low, "evaluates_rets", False):
        ev = sub.evaluate({nm: [np.zeros(w, dtype=np.uint16)] for nm, w in in_ports})
        outs = np.concatenate([np.asarray(ev[p.name][0], dtype=np.uint64) for p in sub.outputs])[None, :]
    words = [int(x) for x in outs[0]]
    at = np.cumsum([0] + [w for _, w in out_ports])
    rows = {p.name: np.zeros((1, p.words), np.uint16) for p in ins}
    rows.update(out_rows({nm: words[at[j]:at[j + 1]] for j, (nm, _) in enumerate(out_ports)}, [tuple(x) for x in out_ports], orows, counts))
    ports = ins + orows
    bc = _commit_strings(ports, rows, {p.name: [bytes(SALT_BYTES)] for p in ports})
    return {"bc": [bc[p.name][0].hex() for p in ports], "outputs": words}


def coins(live: list[dict]) -> str | None:
    """The coins M0's sessions ran on, as their LIVE records name them: ``os-seed-prf`` when the verdict opens the step-0 seed (the
    binary's default), ``live-os`` when the identity names per-round OS coins (``FC_COINS=os``); None for anything else (a
    ``--zk`` session's committed coin tree) and for sessions that differ."""
    modes = {"os-seed-prf" if (x.get("verdict") or {}).get("coin_seed") else
             "live-os" if str((x.get("identity") or {}).get("coins", "")).startswith("live-os") else None for x in live}
    return modes.pop() if len(modes) == 1 else None


def digest(text: str) -> str:
    """The circuit's pin: SHA-512 of the file (every identity and binding digest of the statement is SHA-512)."""
    return _sha512([text]).hexdigest()


# --- instance files -----------------------------------------------------------------------------------------------------------

def statement(ports: list[Port], input_set, lo: int, hi: int, rows: dict, salts: dict):
    """The frame-v3-sha512 roots over instances [lo, hi): an ``hm96-sha512/row/v1`` tree per row (``hm96-sha512/row/v2`` for a
    bit row; the input rows, then the output rows; over ``rows[port]``: the instances' rows, or a shared-row file's table),
    each under a domain bound to the set, the range, the port and its schema. Returns the header's ``frame_v3`` entry and
    every row's ``b || c``."""
    from verity.ml.commitments import hm96
    from verity.ml.commitments.frame_v3 import BitRowLeaf, FrameV3
    from verity.ml.commitments.frame_v3 import Port as FPort
    from verity.ml.commitments.identity import identity_digest_sha512
    from verity.ml.commitments.indexed import RangeIndexedDomain
    from verity.ml.commitments.merkle import CommitmentDomain
    from verity.ml.commitments.rowleaf import row_leaf

    inst = hm96.SHA512
    schemas = {p.name: p.schema for p in ports}
    bind, doms = {}, {}
    for p in ports:
        bind[p.name] = identity_digest_sha512(BINDING_TAG, {"set": input_set.name, "lo": int(lo), "hi": int(hi), "port": p.name,
                                                            "schema": p.schema})
        doms[p.name] = CommitmentDomain(bind[p.name], OWNER, RangeIndexedDomain(0, len(rows[p.name])), hash="sha512")
    # the roots read only the domains: a segmented row's port is configured with its bit row's leaf, which binds no schema here
    fv = FrameV3({p.name: FPort(doms[p.name], BitRowLeaf(p.bits) if p.seg else p.leaf()) for p in ports})
    bcs = _commit_strings(ports, rows, salts)
    roots = {}
    for p in ports:
        leaves = [row_leaf(doms[p.name], i, inst.tree_leaf(inst.default_key, bc), p.schema) for i, bc in enumerate(bcs[p.name])]
        roots[p.name] = fv.root(p.name, leaves).hex()
    return {"frame": "frame-v3-sha512", "roots": roots, "domain_ids": {k: d.domain_id.hex() for k, d in doms.items()},
            "bindings": {k: b.hex() for k, b in bind.items()}, "owner": OWNER, "schemas": schemas,
            "row_key_sha512": hashlib.sha512(inst.default_key).hexdigest()}, bcs


def stage_salts(input_set, lo: int, hi: int, ports: list[Port], key: bytes | None, tables: dict | None = None) -> dict[str, list[bytes]]:
    """Every row's salt (instances [lo, hi); ``tables``, a shared-row file's ``{port: R_p}``: every table row's): fresh from the
    OS generator, or (``key``) SHAKE-256 of the key, the set and the row, so a benchmark's verifier pod can stage its own public
    file. A key makes the salts known to whoever holds it; the verifier's checking process never reads them either way (its file
    has none). A serving committer draws its own."""
    rows = {p.name: range(tables[p.name]) if tables else range(lo, hi) for p in ports}
    if key is None:
        return {p.name: [os.urandom(SALT_BYTES) for _ in rows[p.name]] for p in ports}
    tag = b"verity/flock-circuit/stage-salt/v1\0" + key + bytes.fromhex(input_set.content_digest)
    kind = b"table\0" if tables else b""
    return {p.name: [hashlib.shake_256(tag + p.name.encode() + b"\0" + kind + i.to_bytes(8, "big")).digest(SALT_BYTES) for i in rows[p.name]]
            for p in ports}


def share(rows: dict) -> tuple[dict, dict]:
    """Each port's distinct rows, in first-occurrence order, and every instance's ref into them: a shared-row file's tables."""
    tables, refs = {}, {}
    for name, r in rows.items():
        _, first, inv = np.unique(r, axis=0, return_index=True, return_inverse=True)
        order = np.argsort(first)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        tables[name], refs[name] = r[first[order]], rank[inv.reshape(-1)]
    return tables, refs


def check_shared_reads(meta: dict, n: int, counts: dict | None, refs: dict | None) -> None:
    """META ``shared_reads`` (``[{port, group, position}]``) held to the public file the Lean verifier refuses otherwise
    (``Flock.HmIn.parse`` and ``checkReads``, with its refusals): input row ``port`` is one table row per group of ``group``
    instances, instance ``i`` reading row ``i // group``, and public input ``position`` is ``i % group`` (seed ``instance``,
    derivation ``mod:G``): one writer a window of each shared row. ``counts`` and ``refs`` (``{port: R_p}``,
    ``{port: (n,)}``): a shared-row file's tables, else ``None``."""
    if "shared_reads" not in meta:
        return
    if "public_inputs" not in meta:
        raise ValueError("META: shared_reads without public_inputs (a shared read's position is a public input)")
    rs = meta["shared_reads"]
    if not isinstance(rs, list):
        raise ValueError("META shared_reads: not a list")  # noqa: TRY004 (one refusal type, the verifier's text)
    if not rs:
        raise ValueError("META: shared_reads names no port")
    if counts is None or refs is None:
        raise ValueError("shared reads: the public file is not a shared-row file (header shared_rows)")
    rows = [p.name for p in _meta_ports(meta)[:meta["input_ports"]]]
    pins = {pp["port"]: pp for pp in meta["public_inputs"]}
    seen = set()
    for r in rs:
        if not isinstance(r, dict):
            raise ValueError("META: a shared read is not an object")  # noqa: TRY004
        port, pos, g = r.get("port"), r.get("position"), r.get("group")
        if (sorted(r) != ["group", "port", "position"] or not isinstance(port, str) or not port or not isinstance(pos, str)
                or not pos or type(g) is not int or g <= 0):
            raise ValueError("META: a shared read names no port, position, or group (a positive count of instances)")
        if port in seen:
            raise ValueError(f"shared read {port}: named twice")
        seen.add(port)
        if port not in rows:
            raise ValueError(f"shared read {port}: not an input row")
        if pos not in pins:
            raise ValueError(f"shared read {port}: its position {pos} is not a public input")
        if pins[pos].get("seed") != "instance" or pins[pos].get("derivation") != f"mod:{g}":
            raise ValueError(f"shared read {port}: its position {pos} is not the instance index mod {g} (seed instance, "
                             f"derivation mod:{g})")
        if n % g:
            raise ValueError(f"shared read {port}: {n} instances are not whole groups of {g}")
        if counts[port] != n // g:
            raise ValueError(f"shared read {port}: its table is not one row per group of {g} ({n // g} rows)")
        ref = np.asarray(refs[port]).reshape(-1)
        bad = np.flatnonzero(ref != np.arange(n) // g)
        if len(bad):
            i = int(bad[0])
            raise ValueError(f"shared read {port}: instance {i} reads table row {int(ref[i])}, not its group's {i // g}")


PUBLIC_INPUTS_FORMAT = "verity/flock-public-inputs/v1"


def _public_bytes(pins: list[dict], words: dict, n: int) -> bytes:
    """Every instance's public words, u16 LE, instance-major, the ports in META order."""
    if not pins:
        return b""
    w = np.concatenate([np.asarray(words[pp["port"]], dtype=np.uint16).reshape(n, pp["words"]) for pp in pins], axis=1)
    return np.ascontiguousarray(w.astype("<u2")).tobytes()


def write_public_inputs(text: str, words: dict, n: int, path) -> None:
    """The verifier's own ``--public-inputs`` file (PROTOCOL.md §16.10): a header line ``{format, instances, ports}``, its ports
    META's ``public_inputs`` (each with its seed and derivation), then ``words`` (``{port: (n, words)}``, the verifier's own
    recomputation of the population's public words) as the public file holds them."""
    pins = meta_of(text).get("public_inputs", [])
    head = {"format": PUBLIC_INPUTS_FORMAT, "instances": n, "ports": pins}
    Path(path).write_bytes(json.dumps(head, sort_keys=True).encode() + b"\n" + _public_bytes(pins, words, n))


def public_file(head: dict, public: bytes) -> bytes:
    """The verifier's own ``flock-circuit-inputs`` file of header ``head`` (``rows`` false) and its public bytes, its header
    line as C-Flock writes it (`Instances::public_file`: compact, key-sorted, UTF-8), so the file's SHA-512 is the prover's
    record's ``public_file_sha512``."""
    line = json.dumps(dict(head, rows=False), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return line.encode() + b"\n" + public


def write(text: str, input_set, lo: int, hi: int, path, pub_path=None, salt_key: bytes | None = None, indices: list | None = None,
          share_rows: bool = False, rows: dict | None = None, outs: dict | None = None, tables: dict | None = None,
          refs: dict | None = None, salts: dict | None = None, registered: dict | None = None,
          public_words: dict | None = None, identifier: dict | None = None, digests: tuple[str, str] | None = None,
          meta: dict | None = None) -> dict:
    """``flock-circuit-inputs``: one JSON header line (the set's name, the range, each block's unit indices, the composite
    circuit's sha512, the frame-v3-sha512 statement; never the set's content digest, an unsalted hash of its values), then, for
    the prover only, every instance's rows (its input rows, then its output rows: ports in order, u16 LE) and every row's salt
    (192 bytes, instance-major, ports in order; :func:`stage_salts`), then every row's ``b || c`` (128 bytes). ``path`` gets the prover's file (``None``: none), ``pub_path`` the verifier's own file
    from the same salts, which has neither rows nor salts: it never holds a value, an output word included.

    ``share_rows``: each input port's distinct rows are one committed table (header ``shared_rows: {port: R_p}``, the port's
    frame-v3 tree over them; :func:`share`), and the rows, salts and ``b || c`` are the tables' (port-major, table order), with
    every instance's refs (u32 LE, instance-major, ports in order) after the ``b || c``: the refs show which instances read
    equal rows, so this dedupe suits rows whose equality is the program's structure (``tables`` below is value-independent). An
    output row is its instance's own
    (its table the instances' output rows in order, never deduplicated, so equal outputs stay apart). ``rows`` and ``outs``
    (``{port: (n, words)}``) replace the set's range as the instances' words.

    ``tables`` and ``refs`` (``{port: (R_p, words)}``, ``{port: (n,)}``, input ports): the shared-row tables and refs as given,
    in their order and with no dedupe (a value-independent layout: equal rows stay separate rows, each with its own salt).
    ``salts`` (``{port: [192-byte salt, ...]}``, every row port's, one per table row, or per instance without tables): the
    committer's own salts.

    ``registered`` (``{port: {"value", "positions", "paths"}}``, with ``tables``): the rows whose table rows are rows of a
    registered value, committed with the registration's salts (``salts``), each table row's position and frame-v3 path to the
    registered root (``verity.core.protocols.sampled_proofs.one_stage.registered``, committed in the port's row format: its
    ``schema``); the header's ``registered``. A port
    is an input port (its table's rows) or an output port's row (its table the instances' output rows, one each: instance
    ``i``'s output is the registered row at ``positions[i]``, under that row's registered salt ``salts[port][i]``). A padding
    instance's rows keep their zero salts and pinned ``dummy`` either way: the verifier opens only the instances' rows.

    A circuit with public inputs (META ``public_inputs``, PROTOCOL.md §16.10): the header names them as META does, and both files
    end with every instance's public words (u16 LE, instance-major, the ports in order): ``public_words``
    (``{port: (n, words)}``), else the set's.

    ``identifier``: the header's ``identifier`` (PROTOCOL.md §16.10), a JSON object C-Flock does not read, on which canonical JSON
    is RFC 8785 (``verity.ml.circuits.codec.jcs_domain_error``). The public digest binds it, and the verifier holds it to its own
    (``--identifier F``).

    ``digests`` and ``meta``: ``text``'s (:func:`digest`, :func:`class_digest`; :func:`meta_of`), from a caller that writes many
    statements of one circuit (`class_statement.UnitStager`); else computed here."""
    from verity.ml.circuits.codec import jcs_domain_error

    if identifier is not None:
        bad = jcs_domain_error(identifier, "identifier") if isinstance(identifier, dict) else "identifier: not a JSON object"
        if bad:
            raise ValueError(f"{bad} (the header's identifier is a JSON object on which canonical JSON is RFC 8785)")
    if meta is None:
        meta = meta_of(text)
    sub = input_set.subcircuit
    n = hi - lo
    pins = meta.get("public_inputs", [])
    if public_words is None:
        public_words = {pp["port"]: (rows[pp["port"]] if rows is not None and pp["port"] in rows else input_set.port(pp["port"], lo, hi))
                        for pp in pins}
    pub_tail = _public_bytes(pins, public_words, n)
    ports = _meta_ports(meta)
    ins, orows = ports[:meta["input_ports"]], ports[meta["input_ports"]:]
    out_ports = [tuple(x) for x in meta["out_ports"]]
    if outs is None:
        outs = {p.name: np.asarray(input_set.port(p.name, lo, hi), dtype=np.uint16).reshape(n, -1) for p in sub.outputs}
    if (tables is None) != (refs is None) or (tables is not None and not share_rows):
        raise ValueError("tables and refs are given together, with share_rows")
    if registered is not None:
        names = [p.name for p in ports]
        if len(set(names)) != len(names):
            dup = next(nm for i, nm in enumerate(names) if nm in names[:i])
            raise ValueError(f"registered reads: the circuit names port {dup} twice, and a registered read names one port")
        if tables is None or salts is None or not registered or not set(registered) <= {p.name for p in ports}:
            raise ValueError("registered: some rows' tables (input ports' or output rows'), given with their tables, refs and "
                             "registered salts")
        by_name = {p.name: p for p in ports}
        for name, e in registered.items():
            rows_p = n if by_name[name] in orows else len(np.asarray(tables[name]).reshape(-1, by_name[name].words))
            if set(e) != {"value", "positions", "paths"} or len(e["positions"]) != rows_p or len(e["paths"]) != rows_p:
                raise ValueError(f"registered port {name}: a value, and one position and path per table row")
    if tables is not None:
        if set(tables) != {p.name for p in ins} or set(refs) != {p.name for p in ins}:
            raise ValueError("tables and refs: one per input port")
        rows = {p.name: np.ascontiguousarray(tables[p.name], dtype=np.uint16).reshape(-1, p.words) for p in ins}
        refs = {p.name: np.asarray(refs[p.name], dtype=np.int64).reshape(-1) for p in ins}
        for p in ins:
            r = refs[p.name]
            if len(r) != n or not len(rows[p.name]) or r.min(initial=0) < 0 or r.max(initial=0) >= len(rows[p.name]):
                raise ValueError(f"port {p.name}: one ref per instance, each a row of its table")
    else:
        if rows is None:
            rows = {p.name: np.asarray(input_set.port(p.name, lo, hi), dtype=np.uint16).reshape(n, -1) for p in sub.inputs}
        refs = None
        if share_rows:
            rows, refs = share({p.name: rows[p.name] for p in ins})
    rows = {**{p.name: rows[p.name] for p in ins}, **out_rows(outs, out_ports, orows, meta.get("out_row_ports"))}
    if refs is not None:
        refs.update({p.name: np.arange(n) for p in orows})
    check_shared_reads(meta, n, {p.name: len(rows[p.name]) for p in ins} if refs is not None else None, refs)
    if salts is None:
        salts = stage_salts(input_set, lo, hi, ports, salt_key, {p.name: len(rows[p.name]) for p in ports} if share_rows else None)
    elif any(len(salts[p.name]) != len(rows[p.name]) or any(len(s) != SALT_BYTES for s in salts[p.name]) for p in ports):
        raise ValueError("salts: one 192-byte salt per row")
    fv, bcs = statement(ports, input_set, lo, hi, rows, salts)
    G = meta["vus_per_block"]
    indices = list(range(lo, hi)) if indices is None else [int(u) for u in indices]
    if len(indices) != n or any(a >= b for a, b in zip(indices, indices[1:])) or min(indices, default=0) < 0:
        raise ValueError("unit indices: one global index per instance, ascending")
    pin, cls = digests or (digest(text), class_digest(text))
    units = {"vus_per_block": G, "units_per_vu": meta.get("units_per_vu", 1), "indices": indices,
             "blocks": [indices[b:b + G] for b in range(0, n, G)], "classes": [cls] * n}
    head = {"format": INST_FORMAT, "statement": meta.get("statement", STATEMENT), "scheme": SCHEME, "set": input_set.name, "subcircuit": sub.id,
            "range": [lo, hi], "instances": n, "circuit_sha512": pin, "units": units, "frame_v3": fv, "rows": True}
    if pins:
        head["public_inputs"] = pins
    if identifier is not None:
        head["identifier"] = identifier
    if registered is not None:
        head["registered"] = {name: {"value": e["value"], "positions": [int(x) for x in e["positions"]], "paths": list(e["paths"])}
                              for name, e in registered.items()}
    if refs is not None:
        head["shared_rows"] = {p.name: len(rows[p.name]) for p in ports}
        public = b"".join(bc for p in ports for bc in bcs[p.name])
        public += np.ascontiguousarray(np.stack([refs[p.name] for p in ports], axis=1).astype("<u4")).tobytes()
    else:
        public = b"".join(bcs[p.name][i] for i in range(n) for p in ports)
    public += pub_tail
    if path is not None:
        if refs is not None:
            private = b"".join(np.ascontiguousarray(rows[p.name].astype("<u2")).tobytes() for p in ports)
            private += b"".join(s for p in ports for s in salts[p.name])
        else:
            # the ports' rows side by side, instance-major: one concatenation (inside the loop it was quadratic in n)
            private = np.ascontiguousarray(np.concatenate([rows[p.name] for p in ports], axis=1).astype("<u2")).tobytes()
            private += b"".join(salts[p.name][i] for i in range(n) for p in ports)
        with open(path, "wb") as f:
            f.write(json.dumps(head, sort_keys=True).encode() + b"\n" + private + public)
    if pub_path is not None:
        Path(pub_path).write_bytes(public_file(head, public))
    return head


def stage(low: IL.Lowering, input_set, lo: int, hi: int, out_dir, public_only: bool = False, salt_key: bytes | None = None,
          indices: list | None = None, partition: str | None = None, program: str | None = None, share_rows: bool = False,
          public_inputs: dict | None = None, identifier: dict | None = None, leaf_scheme: dict | None = None) -> Path:
    """``<out_dir>/circuit.txt`` (the composite circuit), the prover's ``<out_dir>/inst-<n>.bin`` and the verifier's own
    ``pub-<n>.bin`` from the same salts (``share_rows``: shared-row files, :func:`write`); returns the prover's path, or
    (``public_only``: a verifier staging its own public file, which needs the prover's ``salt_key``) writes and returns only
    ``pub-<n>.bin``. ``public_inputs`` (:func:`compose`): also the verifier's ``public-inputs-<n>.bin``, from the set's words, which
    stand in for the verifier's own recomputation (a caller that holds the derivation writes it with
    :func:`write_public_inputs`). ``identifier``: both files' header ``identifier`` (:func:`write`). ``leaf_scheme``
    (:func:`compose`): the proof's own Merkle leaves."""
    if public_only and salt_key is None:
        raise ValueError("a verifier stages its own public file only from the prover's staging salt key (salts are otherwise fresh)")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    statics = set_statics(input_set)
    text = compose(low, input_set.subcircuit, partition=partition, program=program, statics=statics, public_inputs=public_inputs,
                   leaf_scheme=leaf_scheme)
    (out / "circuit.txt").write_text(text)
    if low.units.cut:
        IL.write_tables(out / "tables")
        if not (out / "circuit-forged.txt").exists():
            (out / "circuit-forged.txt").write_text(compose(low, input_set.subcircuit, forged=True, partition=partition, program=program,
                                                            statics=statics, public_inputs=public_inputs, leaf_scheme=leaf_scheme))
    inst, pub = out / f"inst-{hi - lo}.bin", out / f"pub-{hi - lo}.bin"
    write(text, input_set, lo, hi, None if public_only else inst, pub, salt_key, indices, share_rows=share_rows, identifier=identifier)
    if public_inputs:
        write_public_inputs(text, {nm: input_set.port(nm, lo, hi) for nm in public_inputs}, hi - lo, out / f"public-inputs-{hi - lo}.bin")
    return pub if public_only else inst


def lowering_for_set(input_set) -> IL.Lowering:
    """The template's C-Flock lowering for a set (``lowering_for_set`` when the template's statement depends on the set)."""
    from verity_numerical.bench import lowerings

    mod = lowerings.registry("C-Flock").module(input_set.subcircuit.template.id)
    if hasattr(mod, "circuit_lowering"):
        return mod.circuit_lowering(input_set.subcircuit)
    return mod.lowering_for_set(input_set) if hasattr(mod, "lowering_for_set") else mod.frame_lowering(input_set.subcircuit)


def main(argv=None) -> int:
    """``python -m verity_flock.circuit --input-set DIR --n N [N ...] --out DIR [--partition H --program H --unit-indices FILE]``:
    the composite circuit, then for each N the prover's ``inst-N.bin`` and the verifier's own ``pub-N.bin`` (instances [0, N),
    their global unit indices from FILE's JSON list, else [0, N)); prints the pin and the class. ``--class-only``: compose and
    print only (the class a partition names). ``--typed``: the typed statement instead. ``--public-input PORT SEED DERIVATION``
    (repeatable): an input port the verifier recomputes from public data (:func:`compose`'s ``public_inputs``)."""
    import argparse

    from verity_numerical.bench.input_sets import InputSet

    ap = argparse.ArgumentParser()
    ap.add_argument("--input-set", required=True)
    ap.add_argument("--n", type=int, nargs="+", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--partition", default=None, help="the verity/partition/v1 digest (SHA-512 hex) the statement binds")
    ap.add_argument("--program", default=None, help="the SHA-512 of the Program the partition is over")
    ap.add_argument("--unit-indices", default=None, help="a JSON list: the global unit index of each staged instance, ascending")
    ap.add_argument("--class-only", action="store_true")
    ap.add_argument("--share-rows", action="store_true", help="each input port's distinct rows as one committed table, refs per instance")
    ap.add_argument("--typed", action="store_true", help="the typed statement (verity_flock.typed_statement.stage)")
    ap.add_argument("--public-input", nargs=3, action="append", default=[], metavar=("PORT", "SEED", "DERIVATION"),
                    help="an input port the verifier recomputes from public data, naming its seed and derivation (PROTOCOL.md §16.10)")
    a = ap.parse_args(argv)
    public_inputs = {p: {"seed": sd, "derivation": d} for p, sd, d in a.public_input} or None
    if public_inputs and a.typed:
        ap.error("--public-input: a typed statement's instance reads only rows")
    s = InputSet.open(a.input_set)
    low = lowering_for_set(s)
    if a.class_only:
        text = compose(low, s.subcircuit, partition=a.partition, program=a.program, statics=set_statics(s), public_inputs=public_inputs)
        print(json.dumps({"subcircuit": s.subcircuit.id, "unit": low.name, "class_sha512": class_digest(text),
                          "program_sha512": program_sha512(s.subcircuit, set_statics(s))}))
        return 0
    idx = json.loads(Path(a.unit_indices).read_text()) if a.unit_indices else None
    if a.typed:
        from verity_flock import typed_statement
    stage_ = typed_statement.stage if a.typed else stage
    for n in a.n:
        stage_(low, s, 0, n, a.out, indices=idx[:n] if idx is not None else None, partition=a.partition, program=a.program,
               share_rows=a.share_rows, **({"public_inputs": public_inputs} if public_inputs else {}))
    text = (Path(a.out) / "circuit.txt").read_text()
    print(json.dumps({"subcircuit": s.subcircuit.id, "unit": low.name, "pin": digest(text), "class_sha512": class_digest(text), "n": a.n}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
