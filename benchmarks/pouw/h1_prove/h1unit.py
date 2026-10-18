"""The H1 tile unit as a C-Flock statement, and the window's commitments it is opened against: what the committer, the
prover and the verifier of `run.py` share.

* **The unit** is `circuit.h1tile.PouwH1Tile{EXACT=GemmAccumulatorS8_v2{K, sm120}, R, C, PAIRS=the full grid}`, the
  Definition the compile pass binds for an s8 grid tile (`h1tile._Unit`). Its statement (`stager`): input rows ``a0..a{R-1}``
  and ``b0..b{C-1}``, one ``hm96-sha512/row/v2`` bit row per A row and B row (8 K bits, the int8 codes' bytes), the public
  row ``pub`` (``salt || position || roots``, 1600 bits: the window's salt, the unit's position and its seed group's
  roots), and two output rows, ``z`` (o0: the cells' exact s32 products) and ``receipt`` (o1: CP-W's 2 (KS + m) words),
  each one ``row/v2`` row of exactly its words.
* **The public row** stands for public input ports, which the Lean verifier refuses as a form its soundness proof does
  not cover (`ProvedScope.statementRules`). Its hm96 salt is public (`pub_salt`), so the verifier computes the row's
  ``b || c`` from its own salt, position and roots, and its public file binds the proof to them; its content is public, so
  the public salt hides nothing that needed hiding.
* **The window's commitments** are those rows' ``b || c`` (hm96-sha512 under the committer's salts) as frame-v3-sha512
  leaves (`rowleaf.row_leaf`) of one tree per side and call (A rows, B rows) and one per output (z, receipt) over the
  window's tiles. `commit_strings` computes a row's ``b || c`` with C-Flock's own function on the port the statement's
  META declares (`stage.py` holds the two equal), so a proof's committed rows are the window's leaves exactly when its
  rows and salts are the committed ones.
* **The verifier's public file** (`verifier_pub`) is built from the opened leaves and the verifier's own public row alone:
  the statement's header (its circuit pin, each port's frame-v3 root over those ``b || c``) and the ``b || c``. Lean
  ``verify --zk`` holds a session to that file, so it accepts only a proof whose committed rows are the window's and whose
  salt, position and roots are the verifier's.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, tree_levels
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2, row_leaf
from verity.core.protocols.pouw.circuit import seeds as SD

HM96 = hm96.SHA512
SALT_BYTES = 192
#: the public row's parameters, in its bit order
PUBLIC = ("salt", "position", "roots")
PUB_BITS = 512 + 64 + 1024
SALT_TAG = b"verity/pouw/h1-prove/salt/v0\0"
PUB_SALT_TAG = b"verity/pouw/h1-prove/pub-row-salt/v0\0"
WINDOW_TAG = "verity/pouw/h1-prove/window/v0"


def unit_salt(coin: bytes) -> bytes:
    """`pouw.salt`, the units' one coin: SHA-512 of the epoch's opened salt coin under this harness's tag."""
    return hashlib.sha512(SALT_TAG + bytes(coin)).digest()


def pub_salt(window: str, position: int) -> bytes:
    """The public row's hm96 salt, which anyone derives from the window and the unit's position."""
    return hashlib.shake_256(PUB_SALT_TAG + window.encode() + b"\0" + int(position).to_bytes(8, "big")).digest(SALT_BYTES)


@dataclass(frozen=True)
class Shape:
    """One H1 tile unit's shape: R A rows and C B rows at contraction K."""

    k: int
    r: int = 4
    c: int = 8

    @cached_property
    def layout(self):
        from verity.core.protocols.pouw.circuit import h1tile as H
        return H.Layout(self.r, self.c, self.k)

    @property
    def z_words(self) -> int:
        return 2 * self.r * self.c

    @property
    def receipt_words(self) -> int:
        return 4 * self.layout.sets

    @property
    def in_rows(self) -> list[str]:
        return [f"a{i}" for i in range(self.r)] + [f"b{j}" for j in range(self.c)]

    def ports(self) -> list:
        """The statement's rows as C-Flock's ports: the A and B rows, the public row, then z's and the receipt's output rows."""
        from verity_flock import circuit as FC
        return ([FC._port(n, self.k // 2, 8 * self.k) for n in self.in_rows] + [FC._port("pub", PUB_BITS // 16, PUB_BITS)]
                + [FC._port("o0", self.z_words, 16 * self.z_words), FC._port("o1", self.receipt_words, 16 * self.receipt_words)])

    def definition(self):
        from verity.ml.circuits.defs import bind
        from verity.ml.circuits.ops.boolean import gemm as BG
        from verity.core.protocols.pouw.circuit import h1tile as H
        exact = bind(BG.GemmAccumulatorS8, K=self.k, **BG.GEMM_SM120_S8)
        return bind(H.PouwH1Tile, EXACT=exact, R=self.r, C=self.c,
                    PAIRS=tuple((i, j) for i in range(self.r) for j in range(self.c)))

    def stager(self, name: str, circuit=None):
        """The unit's `UnitStager`: its circuit, built once (the long step at K = 4096) and written to ``circuit``."""
        from verity_flock import class_statement as CS
        in_rows = ([(f"a{i}", V2, 8 * self.k, [f"a[{i}]"]) for i in range(self.r)]
                   + [(f"b{j}", V2, 8 * self.k, [f"b[{j}]"]) for j in range(self.c)] + [("pub", V2, PUB_BITS, list(PUBLIC))])
        return CS.UnitStager(self.definition(), name, in_rows, [(V2, (0,)), (V2, (1,))], circuit=circuit)


# -- words ------------------------------------------------------------------------------------------------------------------

def row_words(codes: np.ndarray) -> np.ndarray:
    """int8 code rows ``(n, K)`` as their bit rows' u16 words ``(n, K / 2)``: code t's bits are leaves 8 t .. 8 t + 7."""
    b = np.ascontiguousarray(np.asarray(codes).astype(np.int8).view(np.uint8))
    return b.reshape(len(b), -1).view("<u2").astype(np.uint16)


def pub_words(salt: bytes, position: int, roots: tuple[bytes, bytes]) -> np.ndarray:
    """The public row's 100 u16 words: the unit's salt, position and roots leaves (`seeds.leaves`' bits), in that order,
    bit t of the row bit t % 16 of word t // 16."""
    bits = SD.leaves(salt, True) + SD.position_leaves(position, True) + SD.leaves(roots[0], True) + SD.leaves(roots[1], True)
    assert len(bits) == PUB_BITS
    return np.packbits(np.array(bits, np.uint8), bitorder="little").view("<u2").astype(np.uint16)


def z_words(z: list[int]) -> np.ndarray:
    return np.array([int(v) & 0xFFFFFFFF for v in z], dtype="<u4").view("<u2").astype(np.uint16)


def receipt_words(rc: list[int]) -> np.ndarray:
    return np.array([int(v) & 0xFFFFFFFF for v in rc], dtype="<u4").view("<u2").astype(np.uint16)


def commit_strings(port, rows: np.ndarray, salts: list[bytes]) -> list[bytes]:
    """Each row's ``b || c`` under its salt, by C-Flock's own function (`circuit._commit_strings`) on `port`."""
    from verity_flock import circuit as FC
    return FC._commit_strings([port], {port.name: np.asarray(rows, np.uint16).reshape(len(salts), -1)}, {port.name: salts})[port.name]


# -- the window's trees -----------------------------------------------------------------------------------------------------

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
    """`circuit.statement`'s frame-v3 entry over given ``b || c`` (the verifier's: it never holds a row)."""
    from verity_flock import circuit as FC
    from verity.ml.commitments.frame_v3 import BitRowLeaf, FrameV3
    from verity.ml.commitments.frame_v3 import Port as FPort
    bind, doms = {}, {}
    for p in ports:
        bind[p.name] = identity_digest_sha512(FC.BINDING_TAG, {"set": set_name, "lo": 0, "hi": int(n), "port": p.name, "schema": p.schema})
        doms[p.name] = CommitmentDomain(bind[p.name], FC.OWNER, RangeIndexedDomain(0, n), hash="sha512")
    fv = FrameV3({p.name: FPort(doms[p.name], BitRowLeaf(p.bits) if p.seg else p.leaf()) for p in ports})
    roots = {p.name: fv.root(p.name, [row_leaf(doms[p.name], i, HM96.tree_leaf(HM96.default_key, bc), p.schema)
                                      for i, bc in enumerate(bcs[p.name])]).hex() for p in ports}
    return {"frame": "frame-v3-sha512", "roots": roots, "domain_ids": {k: d.domain_id.hex() for k, d in doms.items()},
            "bindings": {k: b.hex() for k, b in bind.items()}, "owner": FC.OWNER, "schemas": {p.name: p.schema for p in ports},
            "row_key_sha512": hashlib.sha512(HM96.default_key).hexdigest()}


def verifier_pub(circuit_head: dict, set_name: str, subcircuit: str, bcs: list[dict[str, bytes]]) -> bytes:
    """The verifier's own public file for n units (`circuit.write`'s ``pub_path``): ``circuit_head`` the statement's
    circuit-side fields (its META, pin and class, which the verifier reads off the circuit file it holds), and each unit's
    rows' ``b || c`` (port name -> 128 bytes): the window's opened leaves, and its own public row's."""
    from verity_flock import circuit as FC
    meta, pin, cls = circuit_head["meta"], circuit_head["pin"], circuit_head["class"]
    if meta.get("public_inputs"):
        raise ValueError("the unit's statement has public input ports, which the Lean verifier refuses")
    ports = FC._meta_ports(meta)
    n = len(bcs)
    fv = _frame(ports, set_name, n, {p.name: [u[p.name] for u in bcs] for p in ports})
    G = meta["vus_per_block"]
    indices = list(range(n))
    units = {"vus_per_block": G, "units_per_vu": meta.get("units_per_vu", 1), "indices": indices,
             "blocks": [indices[b:b + G] for b in range(0, n, G)], "classes": [cls] * n}
    head = {"format": FC.INST_FORMAT, "statement": meta.get("statement", FC.STATEMENT), "scheme": FC.SCHEME, "set": set_name,
            "subcircuit": subcircuit, "range": [0, n], "instances": n, "circuit_sha512": pin, "units": units, "frame_v3": fv,
            "rows": False}
    return json.dumps(head, sort_keys=True).encode() + b"\n" + b"".join(u[p.name] for u in bcs for p in ports)
