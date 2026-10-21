"""The H1 tile unit as a C-Flock statement, and the window's commitments it is opened against: what the committer, the
prover and the verifier of `run.py` share.

* **The unit** is `circuit.h1zy.PouwH1TileZy` over `circuit.h1tile.PouwH1Tile{EXACT=GemmAccumulatorS8_v2{K, sm120}, R, C,
  PAIRS=the full grid}` (`h1window.tile_unit`): the H1 tile on its rows' codes, with its z/y record checked (RT4-INT-33).
  The statement's Definition is the window's template at K (`Shape.definition`, `h1window.template`: one call of the unit,
  its A and B rows ``a_k<K>`` and ``b_k<K>``). Its statement (`stager`): input rows ``a0..a{R-1}`` (``a_k<K>[i]``, `ports`)
  and ``b0..b{C-1}``, one ``hm96-sha512/row/v2`` bit row per A row and B row (8 K + 32 bits: the int8 codes' bytes, then the
  row's f32 scale, s_a for an A row and s_w for a B row, `h1zy.ROW`); ``zy``, the unit's record (48 R C bits: its cells'
  s32 z, row-major, then their bf16 y, `h1zy.RECORD`, what the served ``scaled_mm_zy`` writes); the public row ``pub``
  (``salt || position || roots``, 1600 bits: the window's salt, the unit's position and the window's roots; the salt is
  PoUW's window salt `pouw.salt.pouw_salt`, `unit_salt`); and two output rows, ``receipt`` (o0: CP-W's 2 (KS + m) words)
  and ``ok`` (o1: one word, 1 exactly when the record is the tile's z and its dequant), each one ``row/v2`` row of exactly
  its words. z is no output: the record, an input, carries it.
* **The public row** stands for public input ports, which the Lean verifier refuses as a form its soundness proof does
  not cover (`ProvedScope.statementRules`). Its hm96 salt is public (`pub_salt`), so the verifier computes the row's
  ``b || c`` from its own salt, position and roots, and its public file binds the proof to them; its content is public, so
  the public salt hides nothing that needed hiding.
* **The ok row** is a row of known content: the verifier computes its ``b || c`` from the one word 1 (`OK`) under a public
  salt (`ok_salt`), so a proof it accepts is one whose unit computed 1. A record that is not the tile's product and its
  dequant makes the unit compute 0, whose row is not that ``b || c`` (binding fails), and a prover that commits 1 anyway
  commits a wrong witness.
* **The window's commitments** are those rows' ``b || c`` (hm96-sha512 under the committer's salts) as frame-v3-sha512
  leaves (`rowleaf.row_leaf`) of four trees: ``a``, the window's A rows (the activations every call reads, each with its
  s_a) in a slot of `slot` rows, the token rows padded with zero rows of scale 0; ``b``, every call's B rows (the model's,
  each with its s_w), call after call; ``zy``, every tile's record, row u unit u's; and ``receipt`` over the window's
  tiles. ``a``, ``b`` and ``zy`` are the registered values ``a/<K>``, ``b/<K>`` and ``zy`` of the window's Program
  (`register`: `registered.Registered.commit` under the Program's digest, ``row/v2`` rows of 16-bit leaves, `value_words`
  a row), so the Lean verifier opens every row a statement reads against their roots; ``receipt`` is the window's own tree
  (`domain`), opened outside the proof. The roots ``(root(a), root(b))`` are every unit's ``roots`` input.
  `commit_strings` computes a row's ``b || c`` with C-Flock's own function on the port the statement's META declares
  (`stage.py` holds the two equal), so a proof's committed rows are the window's leaves exactly when its rows and salts
  are the committed ones.
* **The window's public Program** (`h1window.program`, `Shape.window`): every tile of the window as one call of the unit
  (the smoke's `TILE`, 4 × 8), each call its own linear at K, every call reading the one A block (the slot), its
  parameters ``a/<K>``, ``b/<K>`` and ``zy`` the trees ``a``, ``b`` and ``zy``; unit u is the tile `window.py` numbers u,
  it reads row u of ``zy``, and its outputs are row u of ``receipt`` and ``ok``. Its `Q_template_instances` partition
  derives the units, and the statement binds it (`Shape.stager`'s ``bind``), so the Lean verifier derives each instance's
  unit from its own copy of the program (``--program``, ``--partition``) and the indices the public file states, and
  `unit_reads` is the row of each tree each of the unit's ports reads. The slot is the window's size class and the calls'
  B rows the model's, so the layout is protocol structure and the real token count stays the prover's. Each circuit port
  reads one tree in every unit, as a registered port reads one value.
* **The verifier's public file** (`verifier_pub`) is built from the opened leaves and the verifier's own public and ok
  rows alone: a shared-row file (`circuit.write`'s), its header the statement's (its circuit pin, its units' global
  indices, each port's frame-v3 root over its table's ``b || c``) with each registered port's table the registered rows its
  units read (``registered``: their positions, the verifier's own, and the opened paths), then the ``b || c`` and each
  unit's refs. Lean ``verify --zk`` holds a session to that file, so it accepts only a proof whose committed rows are the
  window's, whose salt, position and roots are the verifier's and whose ok is 1; with ``--registered``
  (`registered_reads`, the verifier's own reads from its Program) it also opens each A row, B row and record against its
  registered root at the position the verifier's Program reads.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, tree_levels
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2, row_leaf
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw.circuit import h1window as HW
from verity.core.protocols.pouw.circuit import seeds as SD
from verity.core.protocols.sampled_proofs.one_stage import registered as G

HM96 = hm96.SHA512
SALT_BYTES = 192
M32 = (1 << 32) - 1
#: the public row's parameters, in its bit order
PUBLIC = ("salt", "position", "roots")
PUB_BITS = 512 + 64 + 1024
#: STAND-INS for what the network certifier and the window's registration give a certified window, which this harness
#: runs without: the certifier's reveal(p) (`stand_in_reveal`), the registered stream, and the registered prover's id
STAND_IN_REVEAL_TAG = b"verity/pouw/h1-prove/stand-in-reveal/v0\0"
STREAM = hashlib.sha512(b"verity/pouw/h1-prove/stand-in-stream/v0").digest()
PROVER = PS.prover_id(b"verity/pouw/h1-prove/stand-in-prover-key/v0")
PUB_SALT_TAG = b"verity/pouw/h1-prove/pub-row-salt/v0\0"
OK_SALT_TAG = b"verity/pouw/h1-prove/ok-row-salt/v0\0"
WINDOW_TAG = "verity/pouw/h1-prove/window/v0"
#: the ok row the verifier accepts: the unit's one output bit, 1, in its one word
OK = np.ones(1, np.uint16)


def tile() -> str:
    """The H1 tile under the unit: ``H1_TILE`` v1 (`h1tile.PouwH1Tile`, z the exact A Bᵀ) or v3 (`h1tile.PouwH1Tile3`, the
    one stream: z = O' - sum_d 128^d D_d of the folded words, with its wrap correction and dither peel; A Bᵀ too)."""
    import os
    t = os.environ.get("H1_TILE", "v1")
    if t not in ("v1", "v3"):
        raise ValueError(f"H1_TILE={t!r}: v1 or v3")
    return t


def stand_in_reveal(coin: bytes) -> bytes:
    """A STAND-IN for the network certifier's salt for the window, its ``reveal(p)`` (`pouw.salt.revealed`), which this
    harness has no certifier to give: SHA-512 of the epoch's opened salt coin (`pouw.coins`) under this harness's tag.
    Like the certifier's salt it is the auditor's, drawn before the window and opened only after the window's roots are
    recorded; unlike it, no certifier signs it or bounds when it is revealed."""
    return hashlib.sha512(STAND_IN_REVEAL_TAG + bytes(coin)).digest()


def unit_salt(reveal: bytes, window: int, stream: bytes = STREAM, prover: bytes = PROVER) -> bytes:
    """`pouw.salt`, the units' one coin: PoUW's window salt, `pouw.salt.pouw_salt(reveal, window, stream, prover)`, here of
    the stand-in reveal (`stand_in_reveal`), the epoch's number as the window, and the stand-in stream and prover."""
    return PS.pouw_salt(reveal, window, stream, prover)


def pub_salt(window: str, position: int) -> bytes:
    """The public row's hm96 salt, which anyone derives from the window and the unit's position."""
    return hashlib.shake_256(PUB_SALT_TAG + window.encode() + b"\0" + int(position).to_bytes(8, "big")).digest(SALT_BYTES)


def ok_salt(window: str, position: int) -> bytes:
    """The ok row's hm96 salt, which anyone derives from the window and the unit's position: the row's content is the one
    the verifier accepts (`OK`), so the salt hides nothing."""
    return hashlib.shake_256(OK_SALT_TAG + window.encode() + b"\0" + int(position).to_bytes(8, "big")).digest(SALT_BYTES)


#: the smoke's R × C tile (the served path's is 32 × 32, `pouw_h1.TILE`)
TILE = (4, 8)


def slot(rows: int, tile: tuple[int, int] = TILE) -> int:
    return HW.slot(rows, tile)


def window_shape(com: dict) -> tuple[tuple[tuple[str, int], ...], int]:
    """A window commitment's public shape: its calls' names and B rows, and its A slot."""
    return tuple((c["name"], int(c["n"])) for c in com["calls"]), int(com["slot"])


def window_roots(com: dict) -> tuple[bytes, bytes]:
    """A window commitment's roots (root(a), root(b)), every unit's ``roots`` input."""
    return bytes.fromhex(com["root_a"]), bytes.fromhex(com["root_b"])


#: the statement's output rows, by `h1window.unit_reads`' keys
OUT_PORTS = {"receipt": "o0", "ok": "o1"}
#: the window's registered trees, each the Program parameter `value` names
REGISTERED = ("a", "b", "zy")


def ports(template) -> dict[str, str]:
    """The statement's port of each row a unit of ``template`` (the window's at its K, `h1window.template`) reads, by
    `h1window.unit_reads`' and `registered.reads`' keys (its parameters' ``a_k<K>[i]``, ``b_k<K>[j]`` and ``zy``): A row i's
    ``a<i>``, B row j's ``b<j>``, the record's ``zy``."""
    (a, at), (b, bt), (zy, _t) = template.signature()[0][:3]
    return {**{f"{a}[{i}]": f"a{i}" for i in range(at.n)}, **{f"{b}[{j}]": f"b{j}" for j in range(bt.n)}, zy: "zy"}


def window_ports(program) -> dict[str, str]:
    """`ports` of the window ``program``'s template: the statement is one K's circuit, so a window of two K's is refused."""
    tiles = program.fn.bindings["TILES"]
    if len(tiles) != 1:
        raise ValueError(f"h1_prove's statement is one K's circuit; the window has K = {[k for k, _t in tiles]}")
    return ports(tiles[0][1])


def unit_reads(program, tp, u: int) -> dict[str, tuple[str, int]]:
    """`h1window.unit_reads` by the statement's ports (`window_ports`, `OUT_PORTS`; ``salt``, ``position`` and ``roots``
    the public row's), each read's parameter by the tree that commits it (``a/<K>`` the tree ``a``, ``b/<K>`` the tree
    ``b``, ``zy`` the tree ``zy``)."""
    port = {**window_ports(program), **OUT_PORTS}
    return {port.get(key, key): (param.split("/")[0], row) for key, (param, row) in HW.unit_reads(program, tp, u).items()}


# -- the registered trees a, b and zy ---------------------------------------------------------------------------------------


def value(tree: str, k: int) -> str:
    """The window Program's parameter that tree ``a``, ``b`` or ``zy`` registers: ``a/<K>``, ``b/<K>``, ``zy``."""
    return "zy" if tree == "zy" else f"{tree}/{int(k)}"


def tree_of(port: str) -> str:
    """The registered tree a statement port reads: ``a<i>`` tree ``a``, ``b<j>`` tree ``b``, ``zy`` tree ``zy``."""
    return "zy" if port == "zy" else port[0]


def value_words(name: str, tile: tuple[int, int] = TILE) -> int:
    """A registered value's row width in 16-bit leaves: ``a/<K>`` and ``b/<K>`` K/2 + 2 (`h1zy.ROW`), ``zy`` 3 R C
    (`h1zy.RECORD`)."""
    return 3 * tile[0] * tile[1] if name == "zy" else int(name.split("/")[1]) // 2 + 2


def register(program: str, name: str, words: np.ndarray, salts) -> G.Registered:
    """A window's rows of the registered value ``name`` (its bit rows' u16 words, `row_words` or `record_words`), committed
    under the window Program's digest ``program`` in the tile's ports' format (``row/v2``, 16-bit leaves) with one of
    ``salts`` a row."""
    return G.Registered.commit(program, name, np.asarray(words).tolist(), [bytes(s) for s in salts], G.WORD_BITS, V2)


def registered_domain(program: str, name: str, leaves: int) -> CommitmentDomain:
    return G.domain(program, name, value_words(name), int(leaves))


def registered_levels(program: str, name: str, bcs: list[bytes]):
    """The registered tree of ``name``'s rows ``b || c`` (`register`'s, from its ``bcs`` alone)."""
    dom = registered_domain(program, name, len(bcs))
    return tree_levels(dom, [G.leaf(dom, i, bc, V2) for i, bc in enumerate(bcs)])


def verify_registered(program: str, name: str, leaves: int, root: bytes, index: int, bc: bytes, sibs: list[str]) -> bool:
    """``bc`` is row ``index`` of the registered value ``name`` under ``root``: the check flock-verify ``--registered``
    makes of each table row (`registered.verify_read`)."""
    return G.verify_read(registered_domain(program, name, leaves), root, index, bc, bytes.fromhex("".join(sibs)), V2)


def registered_reads(program, query, roots: dict[str, str]) -> dict:
    """The verifier's ``--registered`` file of a window (flock-verify ``verify --zk --registered``): for each registered
    port (`window_ports`), the registered value it reads (``roots``: ``{a/<K>: root, b/<K>: root, zy: root}``, the
    window's), the value's rows, width and domain, and the row each unit of the verifier's own Program reads
    (`registered.reads_file` over ``program`` and ``query``, the window's). The one place h1_prove derives it:
    `served.register_served` takes its place once it reads ``row/v2`` trees."""
    return G.reads_file(program, query, {v: value_words(v) for v in roots}, roots, ports=window_ports(program),
                        schemas={v: V2 for v in roots})


def table(positions) -> tuple[list[int], list[int], list[int]]:
    """A registered port's shared-row table over its instances' positions (`registered.table`): the table's positions,
    each instance's ref into it, and the first instance that reads each table row."""
    tpos, refs = G.table([int(p) for p in positions])
    first: dict[int, int] = {}
    for i, r in enumerate(refs):
        first.setdefault(r, i)
    return tpos, refs, [first[t] for t in range(len(tpos))]


def statement_tables(z: dict, in_ports: list[str], out_ports: list[str], k: int) -> dict:
    """The prover's shared-row statement of `window.py open`'s rows ``z`` (its rows.npz): each registered port's table
    (``r_<port>``, the rows' positions) the distinct positions its instances read in first-read order (`table`), under
    their registered salts and opened paths (``p_<port>``), and every other input port's one row per instance; as
    `UnitStager.stage` takes them: ``tables``, ``refs``, ``salts`` (every port's, one per table row) and ``registered``."""
    n = len(z["units"])
    tables, refs, salts, registered = {}, {}, {}, {}
    for p in in_ports:
        w, s = np.asarray(z[f"w_{p}"], np.uint16), [bytes(x) for x in z[f"s_{p}"]]
        if f"r_{p}" not in z:
            tables[p], refs[p], salts[p] = w, list(range(n)), s
            continue
        tpos, refs[p], first = table(z[f"r_{p}"])
        paths = [str(x) for x in z[f"p_{p}"]]
        for i, t in enumerate(refs[p]):
            j = first[t]
            if (s[i], paths[i]) != (s[j], paths[j]) or not np.array_equal(w[i], w[j]):
                raise ValueError(f"port {p}: instances {j} and {i} read position {tpos[t]} as two rows")
        tables[p], salts[p] = w[first], [s[j] for j in first]
        registered[p] = {"value": value(tree_of(p), k), "positions": tpos, "paths": [paths[j] for j in first]}
    for p in out_ports:
        salts[p] = [bytes(x) for x in z[f"s_{p}"]]
    return {"tables": tables, "refs": refs, "salts": salts, "registered": registered}


@dataclass(frozen=True)
class Shape:
    """One H1 tile unit's shape: R A rows and C B rows at contraction K."""

    k: int
    r: int = TILE[0]
    c: int = TILE[1]

    @cached_property
    def layout(self):
        from verity.core.protocols.pouw.circuit import h1tile as H
        return H.Layout(self.r, self.c, self.k)

    @property
    def ab_words(self) -> int:
        """An A or B row's words: K codes and the f32 scale (`h1zy.ROW`)."""
        return self.k // 2 + 2

    @property
    def record_words(self) -> int:
        return 3 * self.r * self.c

    @property
    def receipt_words(self) -> int:
        return 4 * self.layout.sets

    @property
    def in_rows(self) -> list[str]:
        return [f"a{i}" for i in range(self.r)] + [f"b{j}" for j in range(self.c)] + ["zy"]

    def ports(self) -> list:
        """The statement's rows as C-Flock's ports: the A and B rows, the record, the public row, then the receipt's and
        ok's output rows."""
        from verity_flock import circuit as FC
        return ([FC._port(n, self.ab_words, 16 * self.ab_words) for n in self.in_rows[:-1]]
                + [FC._port("zy", self.record_words, 16 * self.record_words), FC._port("pub", PUB_BITS // 16, PUB_BITS)]
                + [FC._port("o0", self.receipt_words, 16 * self.receipt_words), FC._port("o1", len(OK), 16 * len(OK))])

    def unit(self):
        """The tile unit (`h1window.tile_unit`, over the v1 or v3 H1 tile by `tile`)."""
        from verity.core.protocols.pouw.circuit import h1tile as H
        return HW.tile_unit(self.k, (self.r, self.c), H.PouwH1Tile3 if tile() == "v3" else None)

    def definition(self):
        """The statement's Definition: the window's template at K (`h1window.template`, one call of `unit`), the one its
        partition lists."""
        return HW.template(self.k, self.unit())

    def window(self, calls, m: int):
        """The window's public Program (`h1window.program` over this unit: its ``calls`` ``(name, B rows)``, each its own
        linear at K, all reading the one A block of slot ``m``) and its `Q_template_instances` query."""
        program = HW.program([(name, self.k, n) for name, n in calls], [(name, 0) for name, _n in calls], m,
                             (self.r, self.c), unit=lambda _k: self.unit())
        return program, HW.query(program)

    def stager(self, name: str, circuit=None, bind: dict | None = None):
        """The unit's `UnitStager`: its circuit, built once (the long step at K = 4096) and written to ``circuit``, bound to
        ``bind`` (``{"partition", "program"}``: the window's partition and Program, `Shape.window`)."""
        from verity_flock import class_statement as CS
        template = self.definition()
        reads = {port: key for key, port in ports(template).items()}
        bits = {p.name: p.bits for p in self.ports()}
        in_rows = [(p, V2, bits[p], [reads[p]]) for p in self.in_rows] + [("pub", V2, PUB_BITS, list(PUBLIC))]
        return CS.UnitStager(template, name, in_rows, [(V2, (0,)), (V2, (1,))], circuit=circuit, bind=bind)


# -- words ------------------------------------------------------------------------------------------------------------------

def row_words(codes: np.ndarray, scales) -> np.ndarray:
    """int8 code rows ``(n, K)`` and their f32 scales ``(n,)`` as their bit rows' u16 words ``(n, K / 2 + 2)``: code t's
    bits are leaves 8 t .. 8 t + 7, then the scale's 32 bits, little-endian (`h1zy.ROW`, #3053's A table leaf)."""
    b = np.ascontiguousarray(np.asarray(codes).astype(np.int8).view(np.uint8))
    b = b.reshape(len(b), -1)
    s = np.ascontiguousarray(np.asarray(scales, "<f4").reshape(len(b))).view(np.uint8).reshape(len(b), 4)
    return np.ascontiguousarray(np.concatenate([b, s], axis=1)).view("<u2").astype(np.uint16)


def split_row(words) -> tuple[list[int], int]:
    """`row_words`' one row back: its codes (u8 encodings) and its scale (the f32 encoding)."""
    b = np.asarray(words, np.uint16).astype("<u2").tobytes()
    return list(b[:-4]), int.from_bytes(b[-4:], "little")


def dequant(z, sa, sw) -> np.ndarray:
    """`h1zy.dequant` on arrays (z s32, s_a and s_w f32, broadcast): RN_bf16(RN32(RN32(RN32(z) · s_a) · s_w)), the served
    epilogue, as bf16 encodings, every NaN 0x7FFF."""
    with np.errstate(over="ignore", invalid="ignore"):
        f = (np.asarray(z, np.int64).astype(np.int32).astype(np.float32) * np.asarray(sa, np.float32)) * np.asarray(sw, np.float32)
    u = np.asarray(f, np.float32).view(np.uint32).astype(np.uint64)
    return np.where(np.isnan(f), 0x7FFF, ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF).astype(np.uint16)


def records(a: np.ndarray, b: np.ndarray, sa, sw) -> tuple[np.ndarray, np.ndarray]:
    """``scaled_mm_zy``'s record of A's rows and B's rows (int8 codes) at their scales (f32): z = A Bᵀ exact, as s32, and
    y = `dequant`(z) at each cell's row s_a and column s_w, each ``(rows of A, rows of B)`` (`h1zy.records` on arrays)."""
    z = (np.asarray(a, np.int64) @ np.asarray(b, np.int64).T).astype(np.int32)
    return z, dequant(z, np.asarray(sa, np.float32)[:, None], np.asarray(sw, np.float32)[None, :])


def record_words(z, y) -> np.ndarray:
    """A unit's record, its cells' z (s32) and y (bf16 encodings), row-major, as its bit row's 3 R C u16 words: z's 32-bit
    words little-endian, then y's (`h1zy.RECORD`, #3053's z table leaf)."""
    zw = np.ascontiguousarray((np.asarray(z, np.int64).reshape(-1) & M32).astype("<u4")).view("<u2")
    return np.concatenate([zw, np.asarray(y, np.uint16).reshape(-1)]).astype(np.uint16)


def split_record(words, cells: int) -> tuple[list[int], list[int]]:
    """`record_words`' one row back: its cells' z (s32 words, two's complement in 32 bits) and y (bf16 encodings)."""
    w = np.asarray(words, np.uint16)
    return [int(v) for v in np.ascontiguousarray(w[:2 * cells].astype("<u2")).view("<u4")], [int(v) for v in w[2 * cells:]]


def pub_words(salt: bytes, position: int, roots: tuple[bytes, bytes]) -> np.ndarray:
    """The public row's 100 u16 words: the unit's salt, position and roots leaves (`seeds.leaves`' bits), in that order,
    bit t of the row bit t % 16 of word t // 16."""
    bits = SD.leaves(salt, True) + SD.position_leaves(position, True) + SD.leaves(roots[0], True) + SD.leaves(roots[1], True)
    assert len(bits) == PUB_BITS
    return np.packbits(np.array(bits, np.uint8), bitorder="little").view("<u2").astype(np.uint16)


def receipt_words(rc: list[int]) -> np.ndarray:
    return np.array([int(v) & M32 for v in rc], dtype="<u4").view("<u2").astype(np.uint16)


def commit_strings(port, rows: np.ndarray, salts: list[bytes]) -> list[bytes]:
    """Each row's ``b || c`` under its salt, by C-Flock's own function (`circuit._commit_strings`) on `port`."""
    from verity_flock import circuit as FC
    return FC._commit_strings([port], {port.name: np.asarray(rows, np.uint16).reshape(len(salts), -1)}, {port.name: salts})[port.name]


# -- the window's own tree --------------------------------------------------------------------------------------------------

def domain(window: str, tree: str, count: int) -> CommitmentDomain:
    b = identity_digest_sha512(WINDOW_TAG, {"window": window, "tree": tree, "leaves": count, "schema": V2})
    return CommitmentDomain(b, -1, RangeIndexedDomain(0, count), hash="sha512")


def leaves(dom: CommitmentDomain, bcs: list[bytes]) -> list[bytes]:
    return [row_leaf(dom, i, HM96.tree_leaf(HM96.default_key, bc), V2) for i, bc in enumerate(bcs)]


def levels(dom: CommitmentDomain, bcs: list[bytes]):
    return tree_levels(dom, leaves(dom, bcs))


def path(lv, index: int) -> list[str]:
    out = []
    for level in lv[:-1]:
        out.append(level[index ^ 1].hex())
        index >>= 1
    return out


def verify(dom: CommitmentDomain, root: bytes, index: int, bc: bytes, sibs: list[str]) -> bool:
    from verity.core.protocols.pouw import core_check as CC
    return CC.verify_path(dom, root, index, leaves_one(dom, index, bc), [bytes.fromhex(s) for s in sibs])


def leaves_one(dom: CommitmentDomain, index: int, bc: bytes) -> bytes:
    return row_leaf(dom, index, HM96.tree_leaf(HM96.default_key, bc), V2)


# -- the verifier's public file ---------------------------------------------------------------------------------------------

def _frame(ports: list, set_name: str, n: int, bcs: dict[str, list[bytes]]) -> dict:
    """`circuit.statement`'s frame-v3 entry over given ``b || c`` (the verifier's: it never holds a row) of ``n`` instances,
    each port's tree over its table's rows."""
    from verity_flock import circuit as FC
    from verity.ml.commitments.frame_v3 import BitRowLeaf, FrameV3
    from verity.ml.commitments.frame_v3 import Port as FPort
    bind, doms = {}, {}
    for p in ports:
        bind[p.name] = identity_digest_sha512(FC.BINDING_TAG, {"set": set_name, "lo": 0, "hi": int(n), "port": p.name, "schema": p.schema})
        doms[p.name] = CommitmentDomain(bind[p.name], FC.OWNER, RangeIndexedDomain(0, len(bcs[p.name])), hash="sha512")
    fv = FrameV3({p.name: FPort(doms[p.name], BitRowLeaf(p.bits) if p.seg else p.leaf()) for p in ports})
    roots = {p.name: fv.root(p.name, [row_leaf(doms[p.name], i, HM96.tree_leaf(HM96.default_key, bc), p.schema)
                                      for i, bc in enumerate(bcs[p.name])]).hex() for p in ports}
    return {"frame": "frame-v3-sha512", "roots": roots, "domain_ids": {k: d.domain_id.hex() for k, d in doms.items()},
            "bindings": {k: b.hex() for k, b in bind.items()}, "owner": FC.OWNER, "schemas": {p.name: p.schema for p in ports},
            "row_key_sha512": hashlib.sha512(HM96.default_key).hexdigest()}


def circuit_meta(circuit: Path) -> dict:
    """A circuit file's META (its second line), without reading the gates."""
    with open(circuit) as f:
        f.readline()
        return json.loads(f.readline()[5:])


def forge_statement(honest: Path, out: Path, pub: bytes, instance: int, port: str, words: np.ndarray, circuit: Path) -> dict:
    """A consistent statement with a wrong witness: ``honest``'s prover file (`circuit.write`'s shared-row file of
    ``circuit``) with instance ``instance``'s row of output ``port`` (its table's row ``instance``) replaced by ``words``,
    and ``pub`` (the verifier's own file of a window that committed those words under the same salt) as its header,
    ``b || c`` and refs. Every committed row is then the opened window's, so binding holds, and the replaced row is not
    what the circuit computes on the instance's inputs. Writes ``out``/inst.bin and pub.bin (``pub`` itself)."""
    from verity_flock import circuit as FC
    data = (honest / "inst.bin").read_bytes()
    nl = data.index(b"\n") + 1
    head = json.loads(data[:nl])
    meta = circuit_meta(circuit)
    ports = FC._meta_ports(meta)
    k = [p.name for p in ports].index(port)
    if "shared_rows" not in head or k < meta["input_ports"]:
        raise ValueError(f"{honest}/inst.bin: an output row of a shared-row file is replaced, not {port} of this one")
    n, counts = head["instances"], [int(head["shared_rows"][p.name]) for p in ports]
    W = sum(c * p.words for c, p in zip(counts, ports))
    rows = bytearray(data[nl:nl + 2 * W])
    salts = data[nl + 2 * W:nl + 2 * W + SALT_BYTES * sum(counts)]
    if len(data) != nl + 2 * W + (SALT_BYTES + 128) * sum(counts) + 4 * n * len(ports):
        raise ValueError(f"{honest}/inst.bin: not a shared-row file of its ports' tables without public words")
    at = 2 * (sum(c * p.words for c, p in zip(counts[:k], ports[:k])) + instance * ports[k].words)
    row = sum(counts[:k]) + instance
    words = np.asarray(words, "<u2").reshape(-1)
    if len(words) != ports[k].words:
        raise ValueError(f"{port}: {len(words)} words, not {ports[k].words}")
    rows[at:at + 2 * len(words)] = words.tobytes()
    pnl = pub.index(b"\n") + 1
    phead, body = json.loads(pub[:pnl]), pub[pnl:]
    if len(body) != 128 * sum(counts) + 4 * n * len(ports) or {x: phead[x] for x in phead if x != "frame_v3"} != dict(
            {x: head[x] for x in head if x != "frame_v3"}, rows=False):
        raise ValueError("the verifier's file is not this statement's but for its frame roots")
    salt, bc = salts[SALT_BYTES * row:SALT_BYTES * (row + 1)], body[128 * row:128 * (row + 1)]
    if commit_strings(ports[k], words[None, :], [salt])[0] != bc:
        raise ValueError(f"{port}: the verifier's b || c of instance {instance} is not these words under the prover's salt")
    out.mkdir(parents=True, exist_ok=True)
    (out / "inst.bin").write_bytes(json.dumps(dict(phead, rows=True), sort_keys=True).encode() + b"\n" + bytes(rows) + salts + body)
    (out / "pub.bin").write_bytes(pub)
    return {"instance": instance, "port": port, "row_words_changed": int(np.count_nonzero(
        np.frombuffer(data[nl + at:nl + at + 2 * len(words)], "<u2") != words))}


def verifier_pub(circuit_head: dict, set_name: str, subcircuit: str, bcs: list[dict[str, bytes]], indices: list[int],
                 reads: list[dict[str, tuple[str, int, str]]]) -> bytes:
    """The verifier's own public file for n units (`circuit.write`'s ``pub_path`` of a shared-row statement):
    ``circuit_head`` the statement's circuit-side fields (its META, pin and class, which the verifier reads off the
    circuit file it holds), each unit's global index in the window's Program (``indices``, ascending: the drawn units),
    each unit's rows' ``b || c`` (port name -> 128 bytes: the window's opened leaves, and its own public and ok rows'), and
    each unit's registered reads (``reads``: port -> (value, position, path), the position the verifier's Program reads
    and the opened path, hex). A registered port's table is its units' distinct positions in first-read order (`table`),
    the header's ``registered`` their positions and paths; every other port's is one row per unit."""
    from verity_flock import circuit as FC
    meta, pin, cls = circuit_head["meta"], circuit_head["pin"], circuit_head["class"]
    if meta.get("public_inputs"):
        raise ValueError("the unit's statement has public input ports, which the Lean verifier refuses")
    ports = FC._meta_ports(meta)
    n = len(bcs)
    indices = [int(u) for u in indices]
    if len(indices) != n or indices != sorted(set(indices)) or len(reads) != n:
        raise ValueError(f"{n} units need {n} ascending global indices and reads, not {indices}")
    tabs, refs, registered = {}, {}, {}
    for p in ports:
        if p.name not in reads[0]:
            tabs[p.name], refs[p.name] = [u[p.name] for u in bcs], list(range(n))
            continue
        tpos, refs[p.name], first = table([r[p.name][1] for r in reads])
        for i, t in enumerate(refs[p.name]):
            j = first[t]
            if (bcs[i][p.name], reads[i][p.name]) != (bcs[j][p.name], reads[j][p.name]):
                raise ValueError(f"port {p.name}: units {j} and {i} read position {tpos[t]} as two rows")
        tabs[p.name] = [bcs[j][p.name] for j in first]
        registered[p.name] = {"value": reads[first[0]][p.name][0], "positions": tpos,
                              "paths": [reads[j][p.name][2] for j in first]}
    fv = _frame(ports, set_name, n, tabs)
    vb = meta["vus_per_block"]
    units = {"vus_per_block": vb, "units_per_vu": meta.get("units_per_vu", 1), "indices": indices,
             "blocks": [indices[b:b + vb] for b in range(0, n, vb)], "classes": [cls] * n}
    head = {"format": FC.INST_FORMAT, "statement": meta.get("statement", FC.STATEMENT), "scheme": FC.SCHEME, "set": set_name,
            "subcircuit": subcircuit, "range": [0, n], "instances": n, "circuit_sha512": pin, "units": units, "frame_v3": fv,
            "rows": False, "registered": registered, "shared_rows": {p.name: len(tabs[p.name]) for p in ports}}
    return (json.dumps(head, sort_keys=True).encode() + b"\n" + b"".join(bc for p in ports for bc in tabs[p.name])
            + np.ascontiguousarray(np.stack([refs[p.name] for p in ports], axis=1).astype("<u4")).tobytes())
