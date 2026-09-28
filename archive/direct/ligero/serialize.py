"""Language-neutral files for the B-Ligero chain proofs (lane b-verifier): ``ligero-system/v1`` (the compiled
constraint system = the verification key), ``ligero-statement/v1`` (the verifier's configuration and public words),
``ligero-proof/v1`` (the transcript).  A reader that round-trips, a file-level Python verifier (the cross-check the
Rust verifier in ``archive/ligero-verify`` is compared against), and the generators of the honest / negative /
mutated proof sets.

All integers little-endian.  Field elements are ``u32`` in ``[0, p)``; the reader does not check canonicity (the
verifier does).  Layout (every array is written without a length prefix when the length is fixed by the header):

~~~text
ligero-system/v1     magic "LIGSYS01" | u32 m | u32 n_pins | n_pins x (str name, u32 row) | u32 L | L x linear
                     | u32 Q | Q x (expr a, expr b, expr c) | u8 has_chain | [3 x u32 c rows | 3 x expr y | expr y16]
                     | 32 B sys_id (protocol.system_id, the Python digest)
ligero-system/v2     magic "LIGSYS02" | as v1, but the chain's trailing ``expr y16`` is replaced by
                     u32 n_end | n_end x (expr Y, u32 lo, u32 bits): the chain end in components (chain.py; the FP4
                     relation's full FP32 word as sign / exponent / fraction bit fields, each < p)
    linear           u32 nnz | nnz x (u32 row, u32 coef) | u32 const | u32 pub_pin (0xFFFFFFFF = 0)
    expr             u32 nnz | nnz x (u32 row, u32 coef) | u32 const
    str              u32 len | utf-8 bytes
ligero-statement/v2  magic "LIGSTM02" | u32 layout (1 = chain) | u32 l | u32 n | u32 D | u32 t | f64 target_log2
                     | u8 zk | u32 t_pad | u8 mode (0 = interactive: step-0 coins, challenges from the coins alone;
                     1 = fiat-shamir: no coins, challenges from the transcript; PROTOCOL.md 8c after b-zk-fix)
                     | u32 n_proofs | u32 n_vus | u32 steps | u32 K
                     | l x K u16 a | l x K u16 b | l x u16 y16
ligero-statement/v3  magic "LIGSTM03" | the v2 body | the authentication block (authentication = included, auth.py):
                     str version ("ligero-b/auth/v1") | str schema ("u16") | u32 n_vus | n_vus x u32 vu_index
                     | 3 x tree (a, b, y): 32 B binding | u8 owner + 2 | u64 count | 32 B root
ligero-system/v3     magic "LIGSYS03" | as v2, but the chain block is u32 n_links | n_links x u32 c | n_links x expr y
                     | u8 has_y16 | [expr y16] or [u32 n_end | n_end x (expr Y, u32 lo, u32 bits)]: a chain with more
                     than the accumulator's three linked rows (hashchain.py: + 16 sponge capacity lanes; chain.py gives
                     the extra links monomial coefficients).  Written only when n_links != 3 (v1 / v2 otherwise).
ligero-statement/v4  magic "LIGSTM04" | the v2 header through u32 K | u8 word_bytes | u8 y_bytes | str relation
                     | l x K a | l x K b | l x y16 (u8/u16/u32 per the byte widths) | u8 has_auth | [the v3 block]
                     The relation-named statement: every registered relation of relations.py but the Ampere BF16
                     unit (bf16-hopper: 16/16-bit words; fp8-hopper / fp8-ada: 8-bit operands and 22-bit public words
                     in 32-bit slots), the words hashed into the statement digest exactly as written
                     (protocol.RelationHooks word_dtype / y16_dtype).  The Ampere BF16 relation (`relation = "bf16"`,
                     vu.py; no name in the file) is always written as v2 / v3 so that the Rust verifier can check it.
                     (Lane b-hopper had written this layout under "LIGSTM03" before v3 was taken by the authentication
                     block above; the merge lane/b-merge-h100 made every relation-named statement v4 -- b-hopper's
                     LIGSTM03 dumps died with their pod, so no file of that layout exists.)
ligero-statement/v5  magic "LIGSTM05" | the v2 header through u32 K | u8 y_bytes | str relation | str authentication
                     ("included-hash": hashchain.py / hashauth.py -- the operand words are NOT in the statement)
                     | n_vus x 16 u32 digests (per VU: the x row's 8 Poseidon2 lanes, then the W column's; the public
                     table the proof pins is derived: is_end = 1 and these lanes at the VU's last column, 0 elsewhere)
                     | l x y16 (y_bytes) | the hash-auth block: str version ("ligero-b/auth/v2h") | str schema
                     ("poseidon2-babybear-w24/row/v2h") | 32 B Poseidon2 constants digest | u32 n_vus
                     | n_vus x (u32 vu_index, u32 x_index, u32 w_index) | 3 x tree (a, b, y) as in v3
                     | 3 x (u32 n_sib | n_sib x 32 B): the SHA-256 multiproofs of the a, b, y trees (in the STATEMENT:
                     the proof file is untouched).  The verifier checks the Ligero proof against the digest table,
                     then the digests as leaf/v2h leaves at the claimed ranks under the roots.
ligero-statement/v6  magic "LIGSTM06" | the v5 layout with the ``l x y16`` vector replaced by ``n_vus x y`` (y_bytes): the
                     chain-end words in VU order (lane tier0-bytes).  The public y16 vector of a hashed relation is zero
                     off the chain ends (relchain.HashedRelationRunner), so the reader rebuilds it as
                     y16[v * steps + steps - 1] = y[v], 0 elsewhere -- the statement digest, the challenges and the proof
                     bytes are those of the v5 file (the writer refuses a y16 with a nonzero word off the ends).  At
                     l = 16384 / 341 VUs this is 91 -> 27 kB per sub-batch.  Written by default for hashed relations
                     (``LIGERO_STMT_TRIM=0`` writes v5); v5 files are still read.
ligero-proof/v1      magic "LIGPRF01" | u8 hash (1 = blake3, 2 = sha256) | u8 has_coins | u8 zk | u8 has_chain
                     | u32 D | u32 w_len | u32 h_len | u32 q_len | u32 v_len | u32 M | u32 t | u32 depth
                     | [c1 c2 r1 s1 r2 s2, 32 B each] | 32 B root | D x w_len u32 w | D x h_len u32 h
                     | D x q_len u32 q | D x v_len u32 v | t x u32 cols | M x t u32 opened (row-major)
                     | t x depth x 32 B paths (leaf level first)
ligero-proof/v2      magic "LIGPRF02" | the v1 body | 3 x (u32 n_sib | n_sib x 32 B): the Merkle multiproofs of the
                     a, b, y trees (canonical order of verity.ml.commitments.multiproof; SHA-256 nodes)
ligero-system/v4     magic "LIGSYS04" | the v3 body through the chain end (the chain block always in the v3 form:
                     u32 n_links | ...) | the fingerprint block (chain.py "Fingerprint constraints"): u32 n_op
                     | n_op x 16 u32 lane rows | u32 coords | u8 mode (0 = coin: rho from challenge 1, F a prover
                     message; 1 = public: rho, F statement values) | 32 B sys_id.  Written for a chain with ``fp``.
ligero-proof/v3      magic "LIGPRF03" | the v1 body | u32 n_fp | n_fp x u32 F: the fingerprint message of a coin-mode
                     system (n_vus x n_op x coords, VU-major; protocol.Proof.fp), sent after coin 1 (interactive) /
                     absorbed into challenge 2 (Fiat-Shamir)
ligero-statement/v6  magic "LIGSTM06" | the v2 header through u32 K | u8 y_bytes | str relation | str authentication
                     ("included-hash-shared": relchain.SharedHashedRunner -- the operand rows are hashed ONCE per
                     sub-batch in a paired hash-side proof) | l x y16 (y_bytes) | the hash side's configuration
                     u32 l_h | u32 n_h | u32 D_h | u32 t_h | u32 t_pad_h | u32 n_units | u32 coords
                     | n_units x (u32 role, 8 u32 digest): the hash side's units (distinct x rows then distinct W
                     columns, ascending rank: hashchain.shared_units of the block below) and their Poseidon2 digests
                     | the hash-auth block of v5 (indices, trees, multiproofs).  Derived: the GEMM side's table
                     [is_end] (l, 1) and index rows (n_vus, 3); the hash side's table [is_end | role | d0..d7]
                     (l_h, 10) and units (n_units, 2).  The pair is (this, a v3 proof of the GEMM side, a v1 proof of
                     the hash side); the verifier checks G, derives rho itself, maps F through the units and checks H
                     on (table, units, rho, F_H), then the digests as leaf/v2h leaves under the roots.
~~~
A BF16 statement without an authentication block is written as v2 and a proof without multiproofs as v1 (byte-identical
to before these extensions); the verifier refuses an authenticated statement with a v1 proof and vice versa.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import auth as auth_mod
from .compile import P, System

SYS_MAGIC, STMT_MAGIC, PROOF_MAGIC = b"LIGSYS01", b"LIGSTM02", b"LIGPRF01"
SYS_MAGIC_V2 = b"LIGSYS02"     # a chain system whose end is in components (chain.py): v1 with the y16 expr replaced
STMT_MAGIC_V3, PROOF_MAGIC_V2 = b"LIGSTM03", b"LIGPRF02"
STMT_MAGIC_V4 = b"LIGSTM04"
SYS_MAGIC_V3 = b"LIGSYS03"     # a chain with n_links != 3 linked rows (the hashed relation's sponge capacities)
STMT_MAGIC_V5 = b"LIGSTM05"    # authentication = included-hash: digests + leaf indices + roots + multiproofs, no operand words
STMT_MAGIC_V6 = b"LIGSTM06"    # v5 with the chain-end words only (n_vus, not l): the trimmed hashed statement (lane tier0-bytes)
#: Hashed relations write ``ligero-statement/v6`` unless ``LIGERO_STMT_TRIM=0`` (then the v5 layout, byte-identical to before).
HASHED_STATEMENT_TRIMMED = os.environ.get("LIGERO_STMT_TRIM", "1") != "0"
SYS_MAGIC_V4 = b"LIGSYS04"     # a chain with fingerprint constraints (chain.py; the row-sharing systems of hashchain.py)
PROOF_MAGIC_V3 = b"LIGPRF03"   # the fingerprint message F after the v1 body (a coin-mode fingerprinted system)
#: STMT_MAGIC_V6 also heads the row-sharing pair (authentication = included-hash-shared, relchain.SharedHashedRunner):
#: the two v6 layouts are told apart by the authentication string after the relation name (read_statement).
AUTH_HASH = "included-hash"
AUTH_SHARED = "included-hash-shared"
FP_MODES = {"coin": 0, "public": 1}
_DTYPES = {1: "<u1", 2: "<u2", 4: "<u4"}
#: The relation a v2 / v3 statement (no name in the file) is: the Ampere BF16 unit of ``vu.py``.
BF16_RELATION = "bf16"
MODE_CODES = {"interactive": 0, "fiat-shamir": 1}
MODE_NAMES = {v: k for k, v in MODE_CODES.items()}
HASH_IDS = {"blake3": 1, "sha256": 2}
NO_PUB = 0xFFFFFFFF


# ---------------------------------------------------------------------------------------------------------------------
# Byte helpers
# ---------------------------------------------------------------------------------------------------------------------

class _W:
    def __init__(self):
        self.parts: list[bytes] = []

    def u8(self, x): self.parts.append(struct.pack("<B", x))
    def u32(self, x): self.parts.append(struct.pack("<I", x))
    def u64(self, x): self.parts.append(struct.pack("<Q", x))
    def f64(self, x): self.parts.append(struct.pack("<d", x))
    def raw(self, b: bytes): self.parts.append(bytes(b))

    def s(self, text: str):
        b = text.encode()
        self.u32(len(b))
        self.raw(b)

    def arr(self, a, dtype):
        self.raw(np.ascontiguousarray(np.asarray(a)).astype(dtype).tobytes())

    def bytes(self) -> bytes:
        return b"".join(self.parts)


class _R:
    def __init__(self, data: bytes):
        self.d, self.at = data, 0

    def take(self, n: int) -> bytes:
        if self.at + n > len(self.d):
            raise ValueError("truncated file")
        b = self.d[self.at:self.at + n]
        self.at += n
        return b

    def u8(self): return struct.unpack("<B", self.take(1))[0]
    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def u64(self): return struct.unpack("<Q", self.take(8))[0]
    def f64(self): return struct.unpack("<d", self.take(8))[0]
    def s(self): return self.take(self.u32()).decode()

    def arr(self, shape, dtype):
        n = int(np.prod(shape)) if len(shape) else 1
        sz = np.dtype(dtype).itemsize
        return np.frombuffer(self.take(n * sz), dtype=dtype).reshape(shape).copy()

    def done(self):
        if self.at != len(self.d):
            raise ValueError(f"trailing bytes: {len(self.d) - self.at}")


def _expr_w(w: _W, lin: dict, c: int):
    items = sorted(lin.items())
    w.u32(len(items))
    for j, v in items:
        w.u32(j)
        w.u32(v % P)
    w.u32(c % P)


def _expr_r(r: _R):
    nnz = r.u32()
    lin = {}
    for _ in range(nnz):
        j = r.u32()
        lin[j] = r.u32()
    return lin, r.u32()


# ---------------------------------------------------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------------------------------------------------

def system_bytes(sys: System) -> bytes:
    from .protocol import system_id
    w = _W()
    w.raw(SYS_MAGIC)
    w.u32(sys.m)
    w.u32(len(sys.pins))
    for idx, name in sys.pins:
        w.s(name)
        w.u32(idx)
    pin_pos = {name: i for i, (_, name) in enumerate(sys.pins)}
    w.u32(len(sys.linear))
    for c in sys.linear:
        _expr_w(w, c.lin, c.c)
        w.u32(NO_PUB if c.pub is None else pin_pos[c.pub])
    w.u32(len(sys.quadratic))
    for q in sys.quadratic:
        for e in (q.a, q.b, q.c):
            _expr_w(w, e.lin, e.c)
    w.u8(1 if sys.chain is not None else 0)
    fp = sys.chain.get("fp") if sys.chain is not None else None
    v3 = sys.chain is not None and (len(sys.chain["c"]) != 3 or fp is not None)
    if sys.chain is not None:
        if v3:
            w.u32(len(sys.chain["c"]))
        for idx in sys.chain["c"]:
            w.u32(idx)
        for e in sys.chain["y"]:
            _expr_w(w, e.lin, e.c)
        if v3:
            w.u8(1 if sys.chain.get("y16") is not None else 0)
        if sys.chain.get("y16") is not None:
            _expr_w(w, sys.chain["y16"].lin, sys.chain["y16"].c)
        else:
            # component ends (chain.py; only ``ligero-system/v2`` files carry them): u32 n_end | n_end x (expr, u32 lo, u32 bits)
            w.u32(len(sys.chain["y_end"]))
            for Y, lo, bits in sys.chain["y_end"]:
                _expr_w(w, Y.lin, Y.c)
                w.u32(int(lo))
                w.u32(int(bits))
        if fp is not None:
            # ligero-system/v4: the fingerprint block (chain.py): lane rows per operand slot, coordinates, mode
            from .chain import FP_LANES
            w.u32(len(fp["lanes"]))
            for op in fp["lanes"]:
                if len(op) != FP_LANES:
                    raise ValueError("a fingerprinted operand slot has 16 lane rows")
                for r_ in op:
                    w.u32(int(r_))
            w.u32(int(fp["coords"]))
            w.u8(FP_MODES[fp["mode"]])
    w.raw(system_id(sys))
    out = w.bytes()
    if fp is not None:
        out = SYS_MAGIC_V4 + out[len(SYS_MAGIC):]
    elif v3:
        out = SYS_MAGIC_V3 + out[len(SYS_MAGIC):]
    elif sys.chain is not None and sys.chain.get("y16") is None:
        out = SYS_MAGIC_V2 + out[len(SYS_MAGIC):]
    return out


def read_system(data: bytes) -> dict:
    r = _R(data)
    magic = r.take(8)
    if magic not in (SYS_MAGIC, SYS_MAGIC_V2, SYS_MAGIC_V3, SYS_MAGIC_V4):
        raise ValueError("not a ligero-system/v1, /v2, /v3 or /v4 file")
    v2 = magic == SYS_MAGIC_V2
    v4 = magic == SYS_MAGIC_V4
    v3 = magic == SYS_MAGIC_V3 or v4
    m = r.u32()
    pins = [(r.s(), r.u32()) for _ in range(r.u32())]
    linear = []
    for _ in range(r.u32()):
        lin, c = _expr_r(r)
        pub = r.u32()
        linear.append((lin, c, None if pub == NO_PUB else pub))
    quadratic = [tuple(_expr_r(r) for _ in range(3)) for _ in range(r.u32())]
    chain = None
    if r.u8():
        n_links = r.u32() if v3 else 3
        c_rows = [r.u32() for _ in range(n_links)]
        ys = [_expr_r(r) for _ in range(n_links)]
        if v3:
            v2 = not bool(r.u8())
        if v2:
            # component ends: 1..3 of them, each a bit field of a 32-bit word below p (chain.check_component, the Rust
            # parser's rule) -- a file with any other widths is refused here, not at proof time
            from .chain import END_FAMILIES, check_component
            n_end = r.u32()
            if not 1 <= n_end <= len(END_FAMILIES):
                raise ValueError(f"system file: a chain end has 1..{len(END_FAMILIES)} components, got {n_end}")
            y_end = []
            for x in range(n_end):
                Y = _expr_r(r)
                y_end.append((Y, *check_component(x, r.u32(), r.u32())))
            chain = {"c": c_rows, "y": ys, "y16": None, "y_end": y_end}
        else:
            y16 = _expr_r(r)
            chain = {"c": c_rows, "y": ys, "y16": y16}
        if v4:
            from .chain import FP_LANES
            n_op = r.u32()
            if not 1 <= n_op <= 8:
                raise ValueError("system file: fingerprinted operand slots must be 1..8")
            lanes = [[r.u32() for _ in range(FP_LANES)] for _ in range(n_op)]
            coords = r.u32()
            mode = r.u8()
            if not 1 <= coords <= 32 or mode not in (0, 1):
                raise ValueError("system file: fingerprint coordinates 1..32, mode 0 / 1")
            chain["fp"] = {"lanes": lanes, "coords": coords, "mode": "coin" if mode == 0 else "public"}
    sys_id = r.take(32)
    r.done()
    return {"m": m, "pins": pins, "linear": linear, "quadratic": quadratic, "chain": chain, "sys_id": sys_id}


# ---------------------------------------------------------------------------------------------------------------------
# Statement
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Statement:
    l: int
    n: int
    D: int
    t: int
    target_log2: float
    zk: bool
    t_pad: int
    mode: str          # "interactive" | "fiat-shamir"
    n_proofs: int
    n_vus: int
    steps: int
    a: np.ndarray      # (l, K) operand words (uint16 for the BF16 relation)
    b: np.ndarray      # (l, K) operand words
    y16: np.ndarray    # (l,) the chain-end public words (uint16 for the BF16 relation)
    auth: auth_mod.AuthStatement | None = None   # authentication = included: the committed-object block (v3 / v4)
    relation: str = BF16_RELATION   # "bf16" (v2 / v3 files carry no name) | a relations.py name (bf16-hopper, fp8-hopper, fp8-ada; v4)
    word_bytes: int = 2             # bytes per operand word in the file (2: v2 / v3)
    y_bytes: int = 2                # bytes per public word in the file (2: v2 / v3)
    hash_auth: object | None = None  # authentication = included-hash (v5): hashauth.HashAuth; ``a`` is then the (l, 17) public
    #                                  table [is_end | d0..d15] and ``b`` the (n_vus, 3) leaf triples (what the digest binds)
    hash_sibs: list | None = None    # v5: the three multiproofs (a, b, y), lists of 32-byte siblings
    shared: "SharedSide | None" = None   # v6 (authentication = included-hash-shared): the hash side; ``a`` is then the
    #                                      (l, 1) table [is_end] and ``b`` the (n_vus, 3) index rows
    trimmed: bool = False            # hashed statement written as v6 (chain-end words only) instead of v5 (the l-vector)
    row_words: int | None = None     # v5 / v6 as read: the header's K (``_K_of`` = K_VU; not in the statement digest, so
    #                                  ``verify_files`` compares it with K_VU)

    @property
    def v4(self) -> bool:
        """Written as the relation-named ``ligero-statement/v4``: any relation but the BF16 unit's 16/16-bit words."""
        return self.hash_auth is None and (self.word_bytes != 2 or self.y_bytes != 2 or self.relation != BF16_RELATION)

    @property
    def v5(self) -> bool:
        return self.hash_auth is not None and self.shared is None

    @property
    def v6(self) -> bool:
        return self.shared is not None

    @property
    def authentication(self) -> str:
        if self.shared is not None:
            return AUTH_SHARED
        return AUTH_HASH if self.hash_auth is not None else "included" if self.auth is not None else "excluded"

    def digests(self) -> np.ndarray:
        """v5: (n_vus, 2 x digest_elems) the published digest lanes (from the public table's chain-end columns)."""
        ends = np.arange(self.n_vus) * self.steps + self.steps - 1
        return np.asarray(self.a, dtype=np.int64)[ends, 1:]

    def y_words(self) -> np.ndarray:
        """v6: (n_vus,) the chain-end public words; refuses a y16 vector with a nonzero word off the ends (the v6 file
        could not represent it, and the statement digest hashes the whole vector)."""
        ends = np.arange(self.n_vus) * self.steps + self.steps - 1
        y = np.asarray(self.y16, dtype=np.int64)
        mask = np.ones(len(y), dtype=bool)
        mask[ends] = False
        if (y[mask] != 0).any():
            raise ValueError("trimmed statement: y16 has a nonzero word off the chain ends")
        return y[ends]

    @property
    def base_relation(self) -> str:
        """The registered relation behind ``relation`` (``"fp8-ada+blake3"`` -> ``"fp8-ada"``)."""
        from .leaf import registry as leaf_registry
        return leaf_registry.split_relation_name(self.relation)[0]

    @property
    def leaf(self) -> str:
        """v5: the leaf scheme the relation string names (``"poseidon2"`` when it carries no ``+<leaf>`` suffix)."""
        from .leaf import registry as leaf_registry
        return leaf_registry.split_relation_name(self.relation)[1]

    def leaf_scheme(self):
        from .leaf import registry as leaf_registry
        return leaf_registry.for_relation(self.relation)

    def config(self):
        from .protocol import Config
        return Config(self.l, self.n, self.D, self.t, self.target_log2, self.zk, self.t_pad, self.mode)

    @property
    def interactive(self) -> bool:
        return self.mode == "interactive"

    def layout(self):
        from .chain import Layout
        return Layout(self.l, self.n_vus, self.steps)


@dataclass
class SharedSide:
    """The hash side of a v6 statement (``relchain.SharedHashedRunner``): its configuration (the same target, rate, mode
    and 2 n_proofs as the GEMM side, sized for its own ``l``), the units ``(n_units, 2) [role, rank]`` in the order
    ``hashchain.shared_units`` derives from the authentication block, their digests ``(n_units, 8)`` and the fingerprint
    coordinates ``coords`` (what the proof's F takes: (n_vus, 2, coords))."""
    l: int
    n: int
    D: int
    t: int
    t_pad: int
    coords: int
    units: np.ndarray
    digests: np.ndarray

    def config(self, st: "Statement"):
        from .protocol import Config
        return Config(self.l, self.n, self.D, self.t, st.target_log2, st.zk, self.t_pad, st.mode)

    def layout(self, steps: int):
        from .chain import Layout
        return Layout(self.l, len(self.units), steps)

    def table(self, steps: int) -> np.ndarray:
        """The hash side's public table (l_h, 10) ``[is_end | role | d0..d7]`` (``hashchain.row_hash_hints``)."""
        from . import hashchain
        n_units = len(self.units)
        tab = np.zeros((self.l, 2 + hashchain.N_ROW_DIGEST), dtype=np.int64)
        ends = np.arange(n_units) * steps + steps - 1
        tab[ends, 0] = 1
        tab[:n_units * steps, 1] = np.repeat(self.units[:, 0], steps)
        tab[ends, 2:] = self.digests
        return tab


def _auth_block_w(w: _W, auth: auth_mod.AuthStatement):
    w.s(auth_mod.AUTH_VERSION)
    w.s(auth_mod.SCHEMA)
    w.u32(len(auth.vu_index))
    w.arr(np.asarray(auth.vu_index, dtype=np.int64), "<u4")
    for tr in auth.trees:
        assert len(tr.binding) == 32 and len(tr.root) == 32
        w.raw(tr.binding)
        w.u8(tr.owner + 2)
        w.u64(tr.count)
        w.raw(tr.root)


def _auth_block_r(r: _R) -> auth_mod.AuthStatement:
    version, schema = r.s(), r.s()
    if version != auth_mod.AUTH_VERSION or schema != auth_mod.SCHEMA:
        raise ValueError(f"unknown authentication block {version!r}/{schema!r}")
    n_vus = r.u32()
    vu_index = [int(x) for x in r.arr((n_vus,), "<u4")]
    trees = []
    for name in auth_mod.TREE_NAMES:
        binding = r.take(32)
        owner = r.u8() - 2
        count = r.u64()
        root = r.take(32)
        trees.append(auth_mod.TreeRef(name, binding, owner, count, root))
    return auth_mod.AuthStatement(tuple(vu_index), tuple(trees))


def _hash_auth_block_w(w: _W, st: Statement):
    from . import hashauth
    auth = st.hash_auth
    scheme = st.leaf_scheme()
    w.s(hashauth.HASH_AUTH_VERSION)
    w.s(scheme.schema)
    w.raw(bytes.fromhex(scheme.params_sha256))
    w.u32(auth.n_vus)
    w.arr(auth.indices(), "<u4")
    for tr in auth.trees:
        assert len(tr.binding) == 32 and len(tr.root) == 32
        w.raw(tr.binding)
        w.u8(tr.owner + 2)
        w.u64(tr.count)
        w.raw(tr.root)
    sibs = st.hash_sibs if st.hash_sibs is not None else [[], [], []]
    if len(sibs) != 3:
        raise ValueError("three multiproofs (a, b, y)")
    for sb in sibs:
        w.u32(len(sb))
        for x in sb:
            if len(x) != 32:
                raise ValueError("malformed sibling")
            w.raw(x)


def _hash_auth_block_r(r: _R, n_vus: int, scheme):
    from . import hashauth
    version, schema = r.s(), r.s()
    if version != hashauth.HASH_AUTH_VERSION or schema != scheme.schema:
        raise ValueError(f"unknown authentication block {version!r}/{schema!r} (the relation names leaf scheme {scheme.name!r})")
    if r.take(32) != bytes.fromhex(scheme.params_sha256):
        raise ValueError(f"{scheme.name} parameters differ from the verifier's")
    if r.u32() != n_vus:
        raise ValueError("authentication block does not match the layout")
    idx = r.arr((n_vus, 3), "<u4").astype(np.int64)
    trees = []
    for name in auth_mod.TREE_NAMES:
        binding = r.take(32)
        owner = r.u8() - 2
        count = r.u64()
        root = r.take(32)
        trees.append(auth_mod.TreeRef(name, binding, owner, count, root))
    auth = hashauth.HashAuth(tuple(int(v) for v in idx[:, 0]), tuple(int(v) for v in idx[:, 1]), tuple(int(v) for v in idx[:, 2]),
                             tuple(trees))
    sibs = []
    for _ in range(3):
        n = r.u32()
        if n > 1 << 24:
            raise ValueError("multiproof too long")
        sibs.append([r.take(32) for _ in range(n)])
    return auth, sibs


def statement_bytes(st: Statement) -> bytes:
    if st.word_bytes not in _DTYPES or st.y_bytes not in _DTYPES:
        raise ValueError(f"statement word widths {st.word_bytes}/{st.y_bytes} bytes are not 1, 2 or 4")
    if st.v6:
        return _statement_bytes_v6(st)
    if st.v5:
        return _statement_bytes_v5(st, trimmed=st.trimmed)
    w = _W()
    w.raw(STMT_MAGIC_V4 if st.v4 else STMT_MAGIC if st.auth is None else STMT_MAGIC_V3)
    w.u32(1)
    for x in (st.l, st.n, st.D, st.t):
        w.u32(x)
    w.f64(st.target_log2)
    w.u8(int(st.zk))
    w.u32(st.t_pad)
    w.u8(MODE_CODES[st.mode])
    w.u32(st.n_proofs)
    w.u32(st.n_vus)
    w.u32(st.steps)
    K = int(np.asarray(st.a).shape[1])
    w.u32(K)
    if st.v4:
        w.u8(st.word_bytes)
        w.u8(st.y_bytes)
        w.s(st.relation)
    w.arr(st.a, _DTYPES[st.word_bytes])
    w.arr(st.b, _DTYPES[st.word_bytes])
    w.arr(st.y16, _DTYPES[st.y_bytes])
    if st.v4:
        w.u8(int(st.auth is not None))
    if st.auth is not None:
        _auth_block_w(w, st.auth)
    return w.bytes()


def _header_w(w: _W, st: Statement, magic: bytes):
    w.raw(magic)
    w.u32(1)
    for x in (st.l, st.n, st.D, st.t):
        w.u32(x)
    w.f64(st.target_log2)
    w.u8(int(st.zk))
    w.u32(st.t_pad)
    w.u8(MODE_CODES[st.mode])
    w.u32(st.n_proofs)
    w.u32(st.n_vus)
    w.u32(st.steps)


def _statement_bytes_v5(st: Statement, trimmed: bool = False) -> bytes:
    """The hashed statement: v5 (the l-vector y16) or, ``trimmed``, v6 (the n_vus chain-end words; same digest)."""
    w = _W()
    _header_w(w, st, STMT_MAGIC_V6 if trimmed else STMT_MAGIC_V5)
    w.u32(_K_of(st))
    w.u8(st.y_bytes)
    w.s(st.relation)            # "<rel>" (poseidon2) | "<rel>+<leaf>": the leaf scheme travels in the relation string
    w.s(AUTH_HASH)
    d = st.digests()
    if d.shape != (st.n_vus, 2 * st.leaf_scheme().digest_elems):
        raise ValueError("digest table shape")
    w.arr(d, "<u4")
    w.arr(st.y_words() if trimmed else st.y16, _DTYPES[st.y_bytes])
    _hash_auth_block_w(w, st)
    return w.bytes()


def _K_of(st: Statement) -> int:
    """The v5 / v6 header's K: the words per committed row, the relation's K (``relations.Relation.K``; K_VU for every
    relation registered at 1536 and for a base the registry does not know)."""
    from .leaf import registry as leaf_registry
    from .relations import K_VU, RELATIONS
    rel = RELATIONS.get(leaf_registry.split_relation_name(st.relation)[0])
    return rel.K if rel is not None else K_VU


def hashed_statement_format() -> str:
    """The format name the dump path writes for a hashed relation (``relchain.bench_vu``'s ``statement_format``)."""
    return "ligero-statement/v6" if HASHED_STATEMENT_TRIMMED else "ligero-statement/v5"


def _statement_bytes_v6(st: Statement) -> bytes:
    from . import hashchain
    sh = st.shared
    w = _W()
    _header_w(w, st, STMT_MAGIC_V6)
    w.u32(_K_of(st))
    w.u8(st.y_bytes)
    w.s(st.relation)
    w.s(AUTH_SHARED)
    w.arr(st.y16, _DTYPES[st.y_bytes])
    n_units = len(sh.units)
    for x in (sh.l, sh.n, sh.D, sh.t, sh.t_pad, n_units, sh.coords):
        w.u32(int(x))
    units = np.asarray(sh.units, dtype=np.int64)
    dg = np.asarray(sh.digests, dtype=np.int64)
    if units.shape != (n_units, 2) or dg.shape != (n_units, hashchain.N_ROW_DIGEST):
        raise ValueError("hash side: units (n_units, 2) and digests (n_units, 8)")
    w.arr(np.concatenate([units[:, :1], dg], 1), "<u4")
    _hash_auth_block_w(w, st)
    return w.bytes()


def _read_v6(r: _R, l, n, D, t, target, zk, t_pad, mode_code, n_proofs, n_vus, steps, K) -> Statement:
    from . import hashchain
    from .leaf import registry as leaf_registry
    yb = r.u8()
    if yb not in _DTYPES:
        raise ValueError(f"statement word width {yb} bytes is not 1, 2 or 4")
    relation = r.s()
    if r.s() != AUTH_SHARED:
        raise ValueError("v6 statement names an authentication other than included-hash-shared")
    if n_vus * steps > l:
        raise ValueError("layout does not fit the columns")
    try:
        scheme = leaf_registry.for_relation(relation)
    except (ValueError, KeyError) as e:
        raise ValueError(f"v6 statement: {e}") from None
    y16 = r.arr((l,), _DTYPES[yb])
    l_h, n_h, D_h, t_h, t_pad_h, n_units, coords = (r.u32() for _ in range(7))
    if n_units == 0 or n_units * steps > l_h or not 1 <= coords <= 32:
        raise ValueError("hash side: units do not fit its columns, or bad fingerprint coordinates")
    rows = r.arr((n_units, 1 + hashchain.N_ROW_DIGEST), "<u4").astype(np.int64)
    auth, sibs = _hash_auth_block_r(r, n_vus, scheme)
    r.done()
    # the units are DERIVED from the block (hashchain.shared_units); the file's roles must agree
    units, _, _ = hashchain.shared_units(auth)
    if len(units) != n_units or not np.array_equal(units[:, 0], rows[:, 0]):
        raise ValueError("hash side: units do not match the authentication block")
    is_end = np.zeros((l, 1), dtype=np.int64)
    is_end[np.arange(n_vus) * steps + steps - 1, 0] = 1
    shared = SharedSide(l_h, n_h, D_h, t_h, t_pad_h, coords, units, rows[:, 1:])
    return Statement(l, n, D, t, target, zk, t_pad, MODE_NAMES[mode_code], n_proofs, n_vus, steps, is_end, auth.indices(), y16,
                     None, relation, 4, yb, auth, sibs, shared, row_words=K)


def _read_v5(r: _R, l, n, D, t, target, zk, t_pad, mode_code, n_proofs, n_vus, steps, K, trimmed: bool = False) -> Statement:
    from . import hashchain
    from .leaf import registry as leaf_registry
    yb = r.u8()
    if yb not in _DTYPES:
        raise ValueError(f"statement word width {yb} bytes is not 1, 2 or 4")
    relation = r.s()
    if r.s() != AUTH_HASH:
        raise ValueError("v5 statement names an authentication other than included-hash")
    if n_vus * steps > l:
        raise ValueError("layout does not fit the columns")
    try:
        scheme = leaf_registry.for_relation(relation)
    except (ValueError, KeyError) as e:
        raise ValueError(f"v5 statement: {e}") from None
    d = r.arr((n_vus, 2 * scheme.digest_elems), "<u4").astype(np.int64)
    if trimmed:
        # v6: the n_vus chain-end words; the public vector is zero off the ends (relchain.HashedRelationRunner)
        yw = r.arr((n_vus,), _DTYPES[yb])
        y16 = np.zeros(l, dtype=yw.dtype)
        y16[np.arange(n_vus) * steps + steps - 1] = yw
    else:
        y16 = r.arr((l,), _DTYPES[yb])
    auth, sibs = _hash_auth_block_r(r, n_vus, scheme)
    r.done()
    table = hashchain.public_table(d, n_vus, steps, l)
    return Statement(l, n, D, t, target, zk, t_pad, MODE_NAMES[mode_code], n_proofs, n_vus, steps, table, auth.indices(), y16,
                     None, relation, 4, yb, auth, sibs, trimmed=trimmed, row_words=K)


def read_statement(data: bytes) -> Statement:
    r = _R(data)
    magic = r.take(8)
    if magic not in (STMT_MAGIC, STMT_MAGIC_V3, STMT_MAGIC_V4, STMT_MAGIC_V5, STMT_MAGIC_V6):
        raise ValueError("not a ligero-statement/v2, /v3, /v4, /v5 or /v6 file")
    if r.u32() != 1:
        raise ValueError("only chain mode (1) is defined")
    l, n, D, t = r.u32(), r.u32(), r.u32(), r.u32()
    target = r.f64()
    zk = bool(r.u8())
    t_pad = r.u32()
    mode_code = r.u8()
    if mode_code not in MODE_NAMES:
        raise ValueError(f"unknown mode code {mode_code}")
    n_proofs, n_vus, steps, K = r.u32(), r.u32(), r.u32(), r.u32()
    if magic == STMT_MAGIC_V6:
        at = r.at
        r.u8()
        r.s()
        shared = r.s() == AUTH_SHARED
        r.at = at
        if shared:
            return _read_v6(r, l, n, D, t, target, zk, t_pad, mode_code, n_proofs, n_vus, steps, K)
    if magic in (STMT_MAGIC_V5, STMT_MAGIC_V6):
        return _read_v5(r, l, n, D, t, target, zk, t_pad, mode_code, n_proofs, n_vus, steps, K, trimmed=magic == STMT_MAGIC_V6)
    relation, wb, yb = BF16_RELATION, 2, 2
    if magic == STMT_MAGIC_V4:
        wb, yb = r.u8(), r.u8()
        if wb not in _DTYPES or yb not in _DTYPES:
            raise ValueError(f"statement word widths {wb}/{yb} bytes are not 1, 2 or 4")
        relation = r.s()
    a = r.arr((l, K), _DTYPES[wb])
    b = r.arr((l, K), _DTYPES[wb])
    y16 = r.arr((l,), _DTYPES[yb])
    has_auth = magic == STMT_MAGIC_V3 or (magic == STMT_MAGIC_V4 and bool(r.u8()))
    auth = _auth_block_r(r) if has_auth else None
    r.done()
    return Statement(l, n, D, t, target, zk, t_pad, MODE_NAMES[mode_code], n_proofs, n_vus, steps, a, b, y16, auth, relation, wb, yb)


# ---------------------------------------------------------------------------------------------------------------------
# Proof
# ---------------------------------------------------------------------------------------------------------------------

def _u4_view(a) -> memoryview:
    """``a`` as ``<u4`` bytes without a copy when it already is a C-contiguous little-endian uint32 array (the prover's
    opened columns and test messages are), else one converted copy."""
    u = np.ascontiguousarray(np.asarray(a), dtype="<u4")
    return memoryview(u).cast("B")


def proof_buffers(pf) -> list:
    """The ``ligero-proof`` byte stream of :func:`proof_bytes` as a list of buffers: the small fields packed into ``bytes``,
    every large array (w, h, q, v, the opened columns) a view of the proof's own memory -- the live sender writes them to the
    socket as they are (``live.send_frame_parts``), so a proof is never copied on the prover's side."""
    w = _W()
    out: list = []

    def flush():
        if w.parts:
            out.append(w.bytes())
            w.parts.clear()

    def big(a):
        flush()
        out.append(_u4_view(a))

    _proof_stream(pf, w, big)
    flush()
    return out


def proof_bytes(pf) -> bytes:
    w = _W()
    _proof_stream(pf, w, lambda a: w.arr(a, "<u4"))
    return w.bytes()


def _proof_stream(pf, w: "_W", big) -> None:
    """Write proof ``pf`` in the ligero-proof/v1 | v2 | v3 layout: the header fields through ``w``, each large ``<u4`` array
    through ``big(array)``."""
    from .protocol import COIN_BYTES
    fp = getattr(pf, "fp", None)
    if fp is not None and pf.auth is not None:
        raise ValueError("a fingerprint message and multiproofs in one transcript: no such format")
    w.raw(PROOF_MAGIC_V3 if fp is not None else PROOF_MAGIC if pf.auth is None else PROOF_MAGIC_V2)
    hid = 1 if pf.hash_name.startswith("blake3") else 2 if pf.hash_name.startswith("sha256") else 0
    w.u8(hid)
    has_coins = pf.coin_commit is not None and pf.coins is not None
    w.u8(int(has_coins))
    w.u8(int(pf.v is not None))
    w.u8(int(pf.q is not None))
    D = pf.w.shape[0]
    M, t = pf.opened.shape
    depth = len(pf.paths[0]) if pf.paths else 0
    for x in (D, pf.w.shape[1], pf.h.shape[1], pf.q.shape[1] if pf.q is not None else 0,
              pf.v.shape[1] if pf.v is not None else 0, M, t, depth):
        w.u32(x)
    if has_coins:
        for x in (pf.coin_commit[0], pf.coin_commit[1], pf.coins.r1, pf.coins.s1, pf.coins.r2, pf.coins.s2):
            assert len(x) == COIN_BYTES
            w.raw(x)
    assert len(pf.root) == 32
    w.raw(pf.root)
    big(pf.w)
    big(pf.h)
    if pf.q is not None:
        big(pf.q)
    if pf.v is not None:
        big(pf.v)
    w.arr(np.asarray(pf.cols, dtype=np.int64), "<u4")
    big(pf.opened)
    for pth in pf.paths:
        assert len(pth) == depth
        for sib in pth:
            assert len(sib) == 32
            w.raw(sib)
    if pf.auth is not None:
        assert len(pf.auth) == len(auth_mod.TREE_NAMES)
        for sibs in pf.auth:
            w.u32(len(sibs))
            for sib in sibs:
                assert len(sib) == 32
                w.raw(sib)
    if fp is not None:
        F = np.ascontiguousarray(np.asarray(fp, dtype=np.int64))
        w.u32(F.size)
        w.arr(F.reshape(-1), "<u4")


def read_proof(data: bytes, fp_shape: tuple | None = None):
    """A ``protocol.Proof`` from the file (no canonicity checks: the verifier's job).  ``fp_shape``: the (n_vus, n_op,
    coords) a v3 file's fingerprint message takes (from the statement and the system; flat when None)."""
    from .protocol import Coins, Proof
    r = _R(data)
    magic = r.take(8)
    if magic not in (PROOF_MAGIC, PROOF_MAGIC_V2, PROOF_MAGIC_V3):
        raise ValueError("not a ligero-proof/v1, /v2 or /v3 file")
    hid, has_coins, has_v, has_q = r.u8(), r.u8(), r.u8(), r.u8()
    D, w_len, h_len, q_len, v_len, M, t, depth = (r.u32() for _ in range(8))
    cc = coins = None
    if has_coins:
        c1, c2, r1, s1, r2, s2 = (r.take(32) for _ in range(6))
        cc, coins = (c1, c2), Coins(r1, s1, r2, s2)
    root = r.take(32)
    w = r.arr((D, w_len), "<u4")
    h = r.arr((D, h_len), "<u4")
    q = r.arr((D, q_len), "<u4") if has_q else None
    v = r.arr((D, v_len), "<u4") if has_v else None
    cols = [int(x) for x in r.arr((t,), "<u4")]
    opened = r.arr((M, t), "<u4")
    paths = [[r.take(32) for _ in range(depth)] for _ in range(t)]
    auth = fp = None
    if magic == PROOF_MAGIC_V2:
        auth = [[r.take(32) for _ in range(r.u32())] for _ in auth_mod.TREE_NAMES]
    if magic == PROOF_MAGIC_V3:
        n_fp = r.u32()
        if n_fp > 1 << 24:
            raise ValueError("fingerprint message too long")
        fp = r.arr((n_fp,), "<u4")
        if fp_shape is not None:
            if int(np.prod(fp_shape)) != n_fp:
                raise ValueError(f"fingerprint message holds {n_fp} words, the statement fixes {fp_shape}")
            fp = fp.reshape(fp_shape)
    r.done()
    name = {1: "blake3 (file)", 2: "sha256 (file)"}.get(hid, "unknown (file)")
    pf = Proof(root, w, h, cols, opened, paths, name, {}, q, v, cc, coins, auth)
    pf.fp = fp
    return pf


def read_proof_head(data: bytes):
    """The head of a proof file -- coin commitments, coins, root and the test message (w, h, q, v) -- without reading the
    opened columns or the paths behind it: what the live verifier's per-proof checks compare (the Rust verifier reads the
    whole file).  ``ValueError`` on a truncated or foreign head."""
    from types import SimpleNamespace

    from .protocol import Coins
    r = _R(data)
    magic = r.take(8)
    if magic not in (PROOF_MAGIC, PROOF_MAGIC_V2, PROOF_MAGIC_V3):
        raise ValueError("not a ligero-proof/v1, /v2 or /v3 file")
    _hid, has_coins, has_v, has_q = r.u8(), r.u8(), r.u8(), r.u8()
    D, w_len, h_len, q_len, v_len, _M, _t, _depth = (r.u32() for _ in range(8))
    cc = coins = None
    if has_coins:
        c1, c2, r1, s1, r2, s2 = (r.take(32) for _ in range(6))
        cc, coins = (c1, c2), Coins(r1, s1, r2, s2)
    root = r.take(32)
    w = r.arr((D, w_len), "<u4")
    h = r.arr((D, h_len), "<u4")
    q = r.arr((D, q_len), "<u4") if has_q else None
    v = r.arr((D, v_len), "<u4") if has_v else None
    return SimpleNamespace(magic=magic, coin_commit=cc, coins=coins, root=root, w=w, h=h, q=q, v=v, fp=None)


# ---------------------------------------------------------------------------------------------------------------------
# The Python verifier on files (the cross-check)
# ---------------------------------------------------------------------------------------------------------------------

def read_coins(raw: bytes):
    """A coins file as ``ligero-verify --coins`` reads it: the verifier's own ``r1 || s1 || r2 || s2``, 128 raw bytes or
    256 hex digits (+ newline)."""
    from .protocol import Coins
    if len(raw) in (256, 257):
        raw = bytes.fromhex(raw.decode().strip())
    if len(raw) != 128:
        raise ValueError(f"coins must hold r1 || s1 || r2 || s2 (128 bytes, raw or hex); got {len(raw)} bytes")
    return Coins(raw[0:32], raw[32:64], raw[64:96], raw[96:128])


def verify_files(stmt_path: Path, proof_path: Path, device: str = "cpu", runner=None,
                 relation: str | None = None, own_coins=None) -> tuple[bool, str, float]:
    """Run ``protocol.verify`` on the two files with the statement's configuration; returns (ok, reason, seconds).
    ``relation``: which relation's system to check against (a v4 statement names its own; a v2 / v3 statement is the
    Ampere BF16 relation unless told otherwise -- the system id in the statement digest rejects a mismatch either way).
    ``own_coins`` (interactive mode): the verifier's OWN step-0 coins, as ``ligero-verify --coins``; without them the
    transcript's openings are replayed (below)."""
    from . import protocol as pm
    try:
        st = read_statement(Path(stmt_path).read_bytes())
    except ValueError as e:
        return False, f"unreadable statement: {e}", 0.0
    if relation is not None:
        if st.v4 and st.relation != relation:
            return False, f"statement names relation {st.relation!r}, verifier asked for {relation!r}", 0.0
        st.relation = relation
    try:
        pf = read_proof(Path(proof_path).read_bytes(), fp_shape=(st.n_vus, 2, st.shared.coords) if st.v6 else None)
        if st.v6:
            # the pair: <stem>.proof is the GEMM side (a v3 file, F after the body), <stem>.hproof the hash side (v1)
            from .relchain import SharedProof
            pf_h = read_proof(Path(proof_path).with_suffix(".hproof").read_bytes())
            if pf.fp is None or pf_h.fp is not None:
                return False, "shared pair: the GEMM side carries the fingerprint message, the hash side none", 0.0
            pf = SharedProof(pf, pf_h, {}, st.shared.layout(st.steps))
    except (ValueError, OSError) as e:
        return False, f"unreadable proof: {e}", 0.0
    R = runner or _runner(device, st)
    if _relation_of(R) != st.relation:
        return False, f"statement names relation {st.relation!r}, the runner proves {_relation_of(R)!r}", 0.0
    if st.authentication != getattr(R, "AUTH", "excluded" if st.auth is None else st.authentication):
        return False, f"statement authentication {st.authentication!r}, the runner proves {getattr(R, 'AUTH', 'excluded')!r}", 0.0
    # Interactive mode: ``verify`` demands the verifier's OWN step-0 coins.  A verifier reading a file has none, so the
    # only thing it can do is REPLAY the transcript's openings as its coins (the "coins are not this verifier's coins"
    # check is then vacuous; what remains is c_i = H(r_i || s_i) and the coin-only challenge derivation).
    coins = own_coins if (st.interactive and own_coins is not None) else pf.coins if (st.interactive and pf.coins is not None) else None
    if st.interactive and coins is None:
        return False, "interactive transcript carries no coins", 0.0
    if (st.v5 or st.v6) and st.row_words is not None:
        from .chain import Layout
        # the VU shape first (protocol.layout_error, as protocol.verify would), then the header K, which is not in the
        # statement digest: every hashed writer emits K_VU (``_K_of``; for fp4-nvf4 24 x 64 codes, not steps x WORDS)
        why = pm.layout_error(Layout(st.l, st.n_vus, st.steps), _hooks_of(R))
        if why is None and st.row_words != _K_of(st):
            why = f"statement: K = {st.row_words} words per row, a hashed statement's row is K = {_K_of(st)}"
        if why is not None:
            return False, why, 0.0
    t0 = time.perf_counter()
    try:
        if st.v6:
            # the paired statement: G on (is_end, indices, y16), the verifier's own rho, H on (table, units, rho, F_H), the
            # digests under the roots (relchain.SharedHashedRunner.verify_vus).  ``coins``: G's (own or replayed) and H's
            # replayed from its transcript unless given as a pair.
            import torch
            sh = st.shared
            if R.cfg(st.l) != st.config():
                R._cfg[st.l] = st.config()
            if R.cfg_h(sh.l) != sh.config(st):
                R._cfg_h[sh.l] = sh.config(st)
            ok, why = R.verify_vus(pf, (st.a.astype(np.int64), st.b.astype(np.int64), torch.as_tensor(st.y16.astype(np.int64)),
                                        st.hash_auth, st.hash_sibs, sh.table(st.steps), sh.units), st.layout(), coins=coins)
        elif st.v5:
            # the hashed statement: R.verify_vus runs protocol.verify against the digest table (the auth block in the
            # statement digest through the hooks' tag) and then the multiproofs of the published digests (hashauth)
            import torch
            if R.cfg(st.l) != st.config():
                R._cfg[st.l] = st.config()
            ok, why = R.verify_vus(pf, (st.a.astype(np.int64), st.b.astype(np.int64), torch.as_tensor(st.y16.astype(np.int64)),
                                        st.hash_auth, st.hash_sibs), st.layout(), coins=coins)
        else:
            ok, why = pm.verify(R.p, R.sys, R.tb, st.config(), None, st.a.astype(np.int64), st.b.astype(np.int64), None,
                                pf, device, chain=(st.layout(), st.y16.astype(np.int64)), coins=coins, rel=_hooks_of(R), auth=st.auth)
    except Exception as e:  # noqa: BLE001 -- the Python verifier is not reject-first on malformed input
        ok, why = False, f"python verifier raised {type(e).__name__}: {e}"
    return ok, why, time.perf_counter() - t0


def _relation_of(R) -> str:
    """The relation a runner proves: its ``relation`` attribute (``relchain.RelationChainRunner``: the registered name),
    else the Ampere BF16 unit (``vu.ChainRunner``)."""
    return getattr(R, "relation", BF16_RELATION)


def _hooks_of(R):
    """The runner's ``protocol.RelationHooks`` (``BF16_HOOKS`` for ``vu.ChainRunner``, which carries none)."""
    from .protocol import BF16_HOOKS
    return getattr(R, "hooks", BF16_HOOKS)


def verify_batch_dir(dir: Path, device: str = "cpu", target_bits: float = 128.0) -> dict:
    """The Python batch verifier on files (PROTOCOL.md 8c, "Batch verification"; the cross-check of ``ligero-verify
    batch``): every ``DIR/*.proof`` with its ``DIR/*.stmt`` (and ``DIR/*.coins`` when present: that proof's own
    step-0 coins) through ``protocol.verify`` -- own statement, own proof, own checks; then ``protocol.batch_bound``
    on THIS verifier's recomputation of every proof's bound (``soundness(statement.config())``) and every statement's
    ``n_proofs`` against the number of sub-batches presented, at ``target_bits``.  The first rejected sub-batch
    rejects the batch with ``"<name>: <reason>"`` (the accounting is reported anyway); the reason strings and their
    precedence are those of the Rust verifier."""
    from . import protocol as pm
    d = Path(dir)
    proofs = sorted(p for p in d.glob("*.proof") if not p.name.startswith("._"))
    if not proofs:
        raise FileNotFoundError(f"no *.proof in {d}")
    t_all = time.perf_counter()
    R = None
    items: list[dict] = []
    for p in proofs:
        sp = p.with_suffix(".stmt")
        st = read_statement(sp.read_bytes())
        if R is None:
            R = _runner(device, st)
        cp = p.with_suffix(".coins")
        own = read_coins(cp.read_bytes()) if cp.is_file() else None
        if st.v6 and own is not None:
            # the pair's coins: <stem>.hcoins the hash side's (replayed from its transcript when absent)
            hp = p.with_suffix(".hcoins")
            own = (own, read_coins(hp.read_bytes())) if hp.is_file() else own
        ok, why, secs = verify_files(sp, p, device, runner=R, own_coins=own)
        if st.v6:
            # the pair: G's bound with the fingerprint term and H's, each sized for 2 n_proofs (SharedHashedRunner.batch_bound)
            sh = st.shared
            b = [pm.soundness(st.config(), R.tb.L, R.tb.Q, 1, fp=(sh.coords, st.steps, 2),
                              chain_extras=R.chain_extras)["per_proof_total_log2"],
                 pm.soundness(sh.config(st), R.tb_h.L, R.tb_h.Q, 1, chain_extras=R.chain_extras_h)["per_proof_total_log2"]]
            items.append({"name": p.name, "accepted": ok, "reason": why, "seconds": secs, "own_coins": own is not None,
                          "per_proof_log2": b, "n_proofs": 2 * st.n_proofs, "pair": True})
            continue
        b = pm.soundness(st.config(), R.tb.L, R.tb.Q, 1, chain_extras=getattr(R, "chain_extras", 0))["per_proof_total_log2"]
        items.append({"name": p.name, "accepted": ok, "reason": why, "seconds": secs, "own_coins": own is not None,
                      "per_proof_log2": b, "n_proofs": st.n_proofs})
    per, claimed, names = [], [], []
    for it in items:
        if it.get("pair"):
            per += it["per_proof_log2"]
            claimed += [it["n_proofs"]] * 2
            names += [it["name"] + ".G", it["name"] + ".H"]
        else:
            per.append(it["per_proof_log2"])
            claimed.append(it["n_proofs"])
            names.append(it["name"])
    rejected = [it for it in items if not it["accepted"]]
    if rejected:
        out = {"accepted": False, "reason": f"{rejected[0]['name']}: {rejected[0]['reason']}", "n_subbatches": len(items),
               "target_bits": float(target_bits), "batch_log2": pm.log2_sum(per), "per_proof_log2": per, "n_proofs_claimed": claimed}
    else:
        out = pm.batch_bound(per, claimed, target_bits, names)
    out["batch_bits"] = -out["batch_log2"]
    return {**out, "n_accepted": len(items) - len(rejected), "n_rejected": len(rejected), "seconds": time.perf_counter() - t_all,
            "verifier": "python (archive.direct.ligero.protocol.verify + batch_bound)", "items": items}


def _runner(device: str, st: Statement | None = None):
    zk = st.zk if st is not None else False
    n_proofs = st.n_proofs if st is not None else 1
    target = st.target_log2 if st is not None else -128.0
    mode = st.mode if st is not None else "interactive"
    name = st.relation if st is not None else BF16_RELATION
    if name == "fp4-nvf4":
        if st is not None and st.v5:        # fp4-nvf4+poseidon2: the hashed statement (fp4/hashed.py on the generic hashed runner)
            from .fp4.hashed import fp4_hashed_relation
            from .relchain import HashedRelationRunner
            return HashedRelationRunner(fp4_hashed_relation(), device, target, 2, n_proofs=n_proofs, zk=zk, mode=mode)
        from .fp4.chain import FP4ChainRunner
        return FP4ChainRunner(device, target, 2, n_proofs=n_proofs, zk=zk, mode=mode)
    if name != BF16_RELATION:
        # any registered relation (relations.py: bf16-hopper, fp8-hopper, fp8-ada) runs on the generic chain runner; a v5
        # statement's "<rel>+<leaf>" names the leaf scheme of its hashed runner
        from .leaf import registry as leaf_registry
        from .relations import RELATIONS
        from .relchain import HashedRelationRunner, RelationChainRunner, SharedHashedRunner
        base, leaf = leaf_registry.split_relation_name(name)
        if base not in RELATIONS:
            raise ValueError(f"no runner for relation {name!r} (known: {BF16_RELATION!r}, 'fp4-nvf4', {sorted(RELATIONS)})")
        if st is not None and st.v6:
            if leaf != "poseidon2":     # split_relation_name canonicalises "" / "hash" to "poseidon2"
                raise ValueError(f"relation {name!r}: row sharing (v6) hashes with Poseidon2 only (hashchain.row_hash_system)")
            return SharedHashedRunner(RELATIONS[base], device, target, 2, n_proofs=n_proofs, zk=zk, mode=mode)
        if st is not None and st.v5:
            return HashedRelationRunner(RELATIONS[base], device, target, 2, n_proofs=n_proofs, zk=zk, mode=mode, leaf=leaf)
        if base != name:
            raise ValueError(f"relation {name!r}: a leaf suffix names a hashed (v5) statement, this file is not v5")
        return RelationChainRunner(RELATIONS[name], device, target, 2, n_proofs=n_proofs, zk=zk, mode=mode)
    from .vu import ChainRunner
    return ChainRunner(device, target, 2, "torch", n_proofs=n_proofs, zk=zk, mode=mode)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------------------------------------------------

def _statement_of(R, pubs, lay, cfg, n_proofs: int, mode: str | None = None, auth=None) -> Statement:
    """The statement of a runner's public inputs, in the runner's relation and word widths (``protocol.RelationHooks``):
    the Ampere BF16 unit is written as v2 / v3 -- the Rust verifier's formats -- and its relation is implied by the
    system id; every registered relation (bf16-hopper; fp8-hopper / fp8-ada: 8-bit operands, 32-bit slots for the
    22-bit public words) is named in the file and takes v4."""
    hooks = _hooks_of(R)
    if len(pubs) == 7:                     # relchain.SharedHashedRunner: (is_end, indices, y16, HashAuth, multiproofs, table_h, units) -> v6
        is_end, idx, y16, hauth, sibs, table_h, units = pubs
        units = np.asarray(units, dtype=np.int64)
        table_h = np.asarray(table_h, dtype=np.int64)
        lay_h = R.layout_h(len(units), lay.l)
        cfg_h = R.cfg_h(lay_h.l)
        ends_h = np.arange(len(units)) * lay_h.steps + lay_h.steps - 1
        shared = SharedSide(cfg_h.l, cfg_h.n, cfg_h.D, cfg_h.t, cfg_h.t_pad, R.fp_coords, units, table_h[ends_h, 2:])
        return Statement(cfg.l, cfg.n, cfg.D, cfg.t, cfg.target_log2, cfg.zk, cfg.t_pad, mode or cfg.mode, n_proofs,
                         lay.n_vus, lay.steps, np.asarray(is_end, dtype=np.int64), np.asarray(idx, dtype=np.int64),
                         np.asarray(y16, dtype=np.int64), None, _relation_of(R), 4, np.dtype(hooks.y16_dtype).itemsize,
                         hauth, [list(x) for x in sibs], shared)
    if len(pubs) == 5:                     # relchain.HashedRelationRunner: (table, indices, y16, HashAuth, multiproofs) -> v5
        a, b, y16, hauth, sibs = pubs
        return Statement(cfg.l, cfg.n, cfg.D, cfg.t, cfg.target_log2, cfg.zk, cfg.t_pad, mode or cfg.mode, n_proofs,
                         lay.n_vus, lay.steps, np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64),
                         np.asarray(y16, dtype=np.int64), None, _relation_of(R), 4, np.dtype(hooks.y16_dtype).itemsize,
                         hauth, [list(x) for x in sibs], trimmed=HASHED_STATEMENT_TRIMMED)
    a, b, y16 = pubs
    return Statement(cfg.l, cfg.n, cfg.D, cfg.t, cfg.target_log2, cfg.zk, cfg.t_pad, mode or cfg.mode, n_proofs,
                     lay.n_vus, lay.steps, np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64),
                     np.asarray(y16, dtype=np.int64), auth, _relation_of(R), np.dtype(hooks.word_dtype).itemsize,
                     np.dtype(hooks.y16_dtype).itemsize)


def _write(out: Path, stem: str, st: Statement, pf, manifest: dict, kind: str, R, device: str, extra: dict | None = None):
    sp, pp = out / f"{stem}.stmt", out / f"{stem}.proof"
    sp.write_bytes(statement_bytes(st))
    pp.write_bytes(proof_bytes(pf))
    # round trip + the Python verifier on the FILES
    rt = read_proof(pp.read_bytes())
    assert rt.root == pf.root and np.array_equal(rt.opened, pf.opened) and rt.cols == pf.cols and rt.paths == pf.paths
    assert np.array_equal(rt.w, pf.w) and np.array_equal(rt.h, pf.h)
    assert (rt.q is None) == (pf.q is None) and (rt.q is None or np.array_equal(rt.q, pf.q))
    assert (rt.v is None) == (pf.v is None) and (rt.v is None or np.array_equal(rt.v, pf.v))
    assert (rt.auth is None) == (pf.auth is None) and (rt.auth is None or rt.auth == [list(s) for s in pf.auth])
    st2 = read_statement(sp.read_bytes())
    assert np.array_equal(st2.a, st.a) and np.array_equal(st2.y16, st.y16) and st2.t == st.t
    assert st2.auth == st.auth
    ok, why, secs = verify_files(sp, pp, device, runner=R)
    entry = {"kind": kind, "stmt": sp.name, "proof": pp.name, "stmt_sha256": sha256_file(sp), "proof_sha256": sha256_file(pp),
             "proof_bytes": pp.stat().st_size, "python_verdict": {"accepted": ok, "reason": why, "seconds": secs},
             **(extra or {})}
    manifest["files"].append(entry)
    print(f"{stem}: {kind} -> python {ok} ({why}) {pp.stat().st_size / 1e6:.2f} MB", flush=True)
    return entry


def _manifest(out: Path, args, R, auth: bool = False) -> dict:
    from .run import _git_commit
    hooks = _hooks_of(R)
    v4 = _relation_of(R) != BF16_RELATION or np.dtype(hooks.word_dtype).itemsize != 2 or np.dtype(hooks.y16_dtype).itemsize != 2
    v5 = getattr(R, "AUTH", None) == AUTH_HASH
    v6 = getattr(R, "AUTH", None) == AUTH_SHARED
    sys_v2 = R.sys.chain is not None and R.sys.chain.get("y16") is None      # component chain ends (the FP4 relation)
    sys_v3 = R.sys.chain is not None and len(R.sys.chain["c"]) != 3          # more linked rows (the hashed relation)
    sys_v4 = R.sys.chain is not None and R.sys.chain.get("fp") is not None   # fingerprint constraints (row sharing)
    fmt = {"system": "ligero-system/v4" if sys_v4 else "ligero-system/v3" if sys_v3 else "ligero-system/v2" if sys_v2 else "ligero-system/v1",
           "statement": "ligero-statement/v6" if v6 else hashed_statement_format() if v5 else "ligero-statement/v4" if v4 else "ligero-statement/v3" if auth else "ligero-statement/v2",
           "proof": "ligero-proof/v3" if v6 else "ligero-proof/v2" if auth else "ligero-proof/v1"}
    if v6:
        fmt["system_h"], fmt["proof_h"] = "ligero-system/v4", "ligero-proof/v1"
    from . import witness as witness_mod
    return {"format": fmt, "relation": _relation_of(R), "authentication": getattr(R, "AUTH", "included" if auth else "excluded"),
            "commit": _git_commit(), "device": args.device, "system": R.sys.summary(),
            "zk": R.zk, "mode": R.mode, "impl": getattr(args, "impl", witness_mod.IMPL), "files": []}


def _save_manifest(out: Path, man: dict):
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str))


def _write_system(out: Path, R, man: dict):
    data = system_bytes(R.sys)
    (out / "system.bin").write_bytes(data)
    rd = read_system(data)
    assert rd["m"] == R.sys.m and len(rd["quadratic"]) == len(R.sys.quadratic)
    man["system_file"] = {"path": "system.bin", "sha256": hashlib.sha256(data).hexdigest(), "sys_id": rd["sys_id"].hex(),
                          "bytes": len(data)}
    if getattr(R, "sys_h", None) is not None:          # the pair's hash side (relchain.SharedHashedRunner)
        data_h = system_bytes(R.sys_h)
        (out / "system_h.bin").write_bytes(data_h)
        rd_h = read_system(data_h)
        assert rd_h["m"] == R.sys_h.m and rd_h["chain"]["fp"]["mode"] == "public"
        man["system_h_file"] = {"path": "system_h.bin", "sha256": hashlib.sha256(data_h).hexdigest(), "sys_id": rd_h["sys_id"].hex(),
                                "bytes": len(data_h)}


def gen_set(args) -> int:
    """The honest sub-batched set: ``--total-vus`` VUs as ceil(total / (batch // 96)) proofs (b-zk's 25 x 170)."""
    from verity_numerical.bench import instances

    from .vu import STEPS, ChainRunner, _vu_data
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_proof = args.batch // STEPS
    total = args.total_vus
    n_proofs = -(-total // per_proof)
    R = ChainRunner(args.device, args.target, 2, "torch", n_proofs=n_proofs, zk=args.zk, mode=args.mode)
    man = _manifest(out, args, R)
    man["set"] = {"total_vus": total, "batch": args.batch, "per_proof": per_proof, "n_proofs": n_proofs, "zk": args.zk,
                  "mode": args.mode}
    _write_system(out, R, man)
    vu = instances.load("vu-k1536", Path(args.root))
    lay = R.layout(per_proof)
    cfg = R.cfg(lay.l)
    man["set"].update({"l": lay.l, "n": cfg.n, "t": cfg.t, "D": cfg.D, "k": cfg.k})
    for i, lo in enumerate(range(0, total, per_proof)):
        hi = min(lo + per_proof, total)
        vus = [_vu_data(vu, j % vu.n) for j in range(lo, hi)]
        pf, pubs, lay_i = R.prove_vus(vus, check=False, min_l=lay.l)
        st = _statement_of(R, pubs, lay_i, R.cfg(lay_i.l), n_proofs)
        _write(out, f"sub_{i:02d}", st, pf, man, "honest", R, args.device,
               {"vus": [lo, hi], "prover_seconds": pf.timings.get("total")})
        _save_manifest(out, man)
    return 0


def gen_negatives(args) -> int:
    """The 52 ``vu-k1536-neg`` instances as chain claims (vu.gate_vu step 2): the prover is run on the negative's
    operands with the honest accumulators and the WRONG claimed word (it emits: the claim is false, not malformed);
    when no honest witness exists (off-domain operands) the prover raises, and the statement is paired with a proof
    of a different statement (the trivial VU)."""
    from verity_numerical.bench import instances
    from verity_numerical.checker.step import step_output

    from .vu import STEP, STEPS, ChainRunner
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R = ChainRunner(args.device, args.target, 2, "torch", zk=args.zk, mode=args.mode)
    man = _manifest(out, args, R)
    _write_system(out, R, man)
    neg = instances.load("vu-k1536-neg", Path(args.root))
    cols = {n: i for i, n in enumerate(neg.columns)}
    foreign, _fpubs, _flay = R.prove_vus([([0] * (STEP * STEPS), [0] * (STEP * STEPS), [0] * STEPS, 0)], check=False)
    for i in range(neg.n):
        row = neg.index[i]
        x, w, y16 = neg.instance(i)
        meta = {"neg_id": i, "mistake": row[cols["mistake"]], "verdict": row[cols["verdict"]]}
        try:
            accs, c = [], 0
            for s in range(STEPS):
                c = step_output(R.p, c, x[16 * s:16 * s + 16], w[16 * s:16 * s + 16])
                accs.append(c)
            pf, pubs, lay = R.prove_vus([(x, w, accs, int(y16))], check=False)
            meta["prover"] = "emitted (false claim)"
        except Exception as e:  # noqa: BLE001
            lay = R.layout(1)
            a = np.zeros((lay.l, STEP), dtype=np.int64)
            b = np.zeros((lay.l, STEP), dtype=np.int64)
            for s in range(STEPS):
                a[s], b[s] = x[16 * s:16 * s + 16], w[16 * s:16 * s + 16]
            yv = np.zeros(lay.l, dtype=np.int64)
            yv[STEPS - 1] = int(y16)
            pf, pubs = foreign, (a, b, yv)
            meta["prover"] = f"refused ({type(e).__name__}); foreign proof (trivial VU) attached"
        st = _statement_of(R, pubs, lay, R.cfg(lay.l), 1)
        _write(out, f"neg_{i:02d}", st, pf, man, "negative", R, args.device, meta)
    _save_manifest(out, man)
    return 0


def gen_small(args) -> int:
    """B = 1 and B = 64 honest proofs (single proof each)."""
    from verity_numerical.bench import instances

    from .vu import ChainRunner, _vu_data
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R = ChainRunner(args.device, args.target, 2, "torch", zk=args.zk, mode=args.mode)
    man = _manifest(out, args, R)
    _write_system(out, R, man)
    vu = instances.load("vu-k1536", Path(args.root))
    for B in args.sizes:
        vus = [_vu_data(vu, j) for j in range(B)]
        pf, pubs, lay = R.prove_vus(vus, check=False)
        st = _statement_of(R, pubs, lay, R.cfg(lay.l), 1)
        _write(out, f"b{B}", st, pf, man, "honest", R, args.device, {"vus": [0, B]})
    _save_manifest(out, man)
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# Mutations (on the serialised files: one protocol element each)
# ---------------------------------------------------------------------------------------------------------------------

def _flip_byte(b: bytes, i: int = 0) -> bytes:
    ba = bytearray(b)
    ba[i] ^= 1
    return bytes(ba)


def mutations(pf, st: Statement) -> list[tuple[str, object, Statement]]:
    """(name, proof, statement) per mutation; ``pf``/``st`` are not modified."""
    import copy

    from .protocol import Coins
    out = []

    def P_(name):
        p2 = copy.deepcopy(pf)
        return name, p2

    M, t = pf.opened.shape
    m_rows = M - (6 * st.D if st.zk else 0)
    n, p2 = P_("opened_value_plus1"); p2.opened[0, 0] = (int(p2.opened[0, 0]) + 1) % P; out.append((n, p2, st))
    n, p2 = P_("opened_value_minus1"); p2.opened[m_rows - 1, t - 1] = (int(p2.opened[m_rows - 1, t - 1]) - 1) % P; out.append((n, p2, st))
    if st.zk:
        D = st.D
        n, p2 = P_("opened_mask_row_prox_plus1"); p2.opened[m_rows, 0] = (int(p2.opened[m_rows, 0]) + 1) % P; out.append((n, p2, st))
        n, p2 = P_("opened_mask_row_chain_a_plus1"); p2.opened[m_rows + 4 * D, 0] = (int(p2.opened[m_rows + 4 * D, 0]) + 1) % P; out.append((n, p2, st))
        n, p2 = P_("opened_mask_row_chain_b_plus1"); p2.opened[M - 1, t - 1] = (int(p2.opened[M - 1, t - 1]) + 1) % P; out.append((n, p2, st))
    n, p2 = P_("opened_value_noncanonical_p"); p2.opened[0, 0] = P; out.append((n, p2, st))
    n, p2 = P_("merkle_sibling_flipped"); p2.paths[0][0] = _flip_byte(p2.paths[0][0]); out.append((n, p2, st))
    n, p2 = P_("merkle_paths_truncated"); p2.paths = [pth[:-1] for pth in p2.paths]; out.append((n, p2, st))
    n, p2 = P_("merkle_paths_extended"); p2.paths = [pth + [pth[-1]] for pth in p2.paths]; out.append((n, p2, st))
    n, p2 = P_("root_flipped"); p2.root = _flip_byte(p2.root); out.append((n, p2, st))
    n, p2 = P_("w_perturbed"); p2.w[0, 0] = (int(p2.w[0, 0]) + 1) % P; out.append((n, p2, st))
    n, p2 = P_("w_noncanonical_p"); p2.w[0, 0] = P; out.append((n, p2, st))
    n, p2 = P_("h_perturbed"); p2.h[0, 0] = (int(p2.h[0, 0]) + 1) % P; out.append((n, p2, st))
    n, p2 = P_("h_last_coefficient_perturbed"); p2.h[-1, -1] = (int(p2.h[-1, -1]) + 1) % P; out.append((n, p2, st))
    if pf.q is not None:
        n, p2 = P_("q_perturbed"); p2.q[0, 0] = (int(p2.q[0, 0]) + 1) % P; out.append((n, p2, st))
        n, p2 = P_("q_sum_preserving_perturbation"); p2.q[0, 1] = (int(p2.q[0, 1]) + 1) % P; p2.q[0, 2] = (int(p2.q[0, 2]) - 1) % P; out.append((n, p2, st))
    if pf.v is not None:
        n, p2 = P_("v_perturbed"); p2.v[0, 0] = (int(p2.v[0, 0]) + 1) % P; out.append((n, p2, st))
        n, p2 = P_("v_perturbed_in_H_kernel"); p2.v[0, 0] = (int(p2.v[0, 0]) + 1) % P; p2.v[0, st.l] = (int(p2.v[0, st.l]) - 1) % P; out.append((n, p2, st))
    n, p2 = P_("t_minus_1_columns"); p2.cols = p2.cols[:-1]; p2.opened = p2.opened[:, :-1].copy(); p2.paths = p2.paths[:-1]; out.append((n, p2, st))
    n, p2 = P_("duplicated_column_index"); p2.cols[1] = p2.cols[0]; p2.opened[:, 1] = p2.opened[:, 0]; p2.paths[1] = list(p2.paths[0]); out.append((n, p2, st))
    n, p2 = P_("columns_swapped"); p2.cols[0], p2.cols[1] = p2.cols[1], p2.cols[0]; p2.opened[:, [0, 1]] = p2.opened[:, [1, 0]]; p2.paths[0], p2.paths[1] = p2.paths[1], p2.paths[0]; out.append((n, p2, st))
    n, p2 = P_("column_index_out_of_range"); p2.cols[0] = st.n; out.append((n, p2, st))
    n, p2 = P_("opened_row_dropped"); p2.opened = p2.opened[:-1].copy(); out.append((n, p2, st))
    n, p2 = P_("extra_opened_row"); p2.opened = np.concatenate([p2.opened, p2.opened[:1]]); out.append((n, p2, st))
    if pf.coins is not None:
        c = pf.coins
        n, p2 = P_("coin_opening_r1_mismatch"); p2.coins = Coins(_flip_byte(c.r1), c.s1, c.r2, c.s2); out.append((n, p2, st))
        n, p2 = P_("coin_opening_s2_mismatch"); p2.coins = Coins(c.r1, c.s1, c.r2, _flip_byte(c.s2)); out.append((n, p2, st))
        n, p2 = P_("coin_commitment_c1_flipped"); p2.coin_commit = (_flip_byte(pf.coin_commit[0]), pf.coin_commit[1]); out.append((n, p2, st))
        n, p2 = P_("coins_stripped"); p2.coin_commit = None; p2.coins = None; out.append((n, p2, st))
        # the challenge blocks: a CONSISTENT re-opening of a slot (fresh r_i, s_i and c_i = H(r_i || s_i)) changes the
        # derived challenge (interactive: r / cols are functions of the coins alone) while the transcript stays as it was
        from .protocol import coin_commit
        r1n, s1n = bytes(32 * [7]), bytes(32 * [9])
        n, p2 = P_("challenge1_coin_reopened_consistently"); p2.coins = Coins(r1n, s1n, c.r2, c.s2); p2.coin_commit = (coin_commit(r1n, s1n), pf.coin_commit[1]); out.append((n, p2, st))
        r2n, s2n = bytes(32 * [11]), bytes(32 * [13])
        n, p2 = P_("challenge2_coin_reopened_consistently"); p2.coins = Coins(c.r1, c.s1, r2n, s2n); p2.coin_commit = (pf.coin_commit[0], coin_commit(r2n, s2n)); out.append((n, p2, st))
    else:
        # Fiat-Shamir transcript: the challenge inputs are root (challenge 1) and root, w, h, q, v (challenge 2); a
        # planted coin block must be refused (the mode has no step 0)
        from .protocol import coin_commit
        r1n, s1n, r2n, s2n = bytes(32 * [7]), bytes(32 * [9]), bytes(32 * [11]), bytes(32 * [13])
        n, p2 = P_("coins_planted_in_fs_transcript"); p2.coins = Coins(r1n, s1n, r2n, s2n); p2.coin_commit = (coin_commit(r1n, s1n), coin_commit(r2n, s2n)); out.append((n, p2, st))
    # statement-side
    st2 = copy.deepcopy(st); end = st.steps - 1; st2.y16[end] ^= 1; out.append(("statement_y16_flipped", pf, st2))
    st2 = copy.deepcopy(st); st2.a[0, 0] ^= 1; out.append(("statement_operand_a_flipped", pf, st2))
    st2 = copy.deepcopy(st); st2.b[st.steps - 1, 15] ^= 0x80; out.append(("statement_operand_b_flipped", pf, st2))
    st2 = copy.deepcopy(st); st2.a[0, 0] = 0x7F80; out.append(("statement_operand_off_domain_inf", pf, st2))
    st2 = copy.deepcopy(st); st2.t = st.t - 1; out.append(("statement_t_minus_1", pf, st2))
    st2 = copy.deepcopy(st); st2.zk = not st.zk; st2.t_pad = 0 if st.zk else 256; out.append(("statement_zk_mode_flipped", pf, st2))
    st2 = copy.deepcopy(st); st2.mode = "fiat-shamir" if st.interactive else "interactive"; out.append(("statement_mode_flipped", pf, st2))
    st2 = copy.deepcopy(st); st2.D = st.D - 1; out.append(("statement_D_minus_1", pf, st2))
    st2 = copy.deepcopy(st); st2.t_pad = st.t_pad + 1; out.append(("statement_t_pad_plus_1", pf, st2))
    if st.auth is not None and pf.auth is not None:
        out += auth_mutations(pf, st)
    return out


def auth_mutations(pf, st: Statement) -> list[tuple[str, object, Statement]]:
    """File-level negatives of the authentication block (``authentication = included``).  The transcript-side ones
    leave the Ligero proof valid and break only the multiproofs (the auth check must catch them); the statement-side
    ones change what the verifier binds (root, indices) without re-proving, so the challenge binding catches them."""
    import copy

    from .auth import AuthStatement, TreeRef
    out = []

    def P_(name):
        return name, copy.deepcopy(pf)

    n, p2 = P_("auth_multiproof_a_truncated"); p2.auth[0] = p2.auth[0][:-1]; out.append((n, p2, st))
    n, p2 = P_("auth_multiproof_b_extended"); p2.auth[1] = p2.auth[1] + [p2.auth[1][-1]]; out.append((n, p2, st))
    n, p2 = P_("auth_multiproof_y_sibling_flipped"); p2.auth[2][0] = _flip_byte(p2.auth[2][0]); out.append((n, p2, st))
    n, p2 = P_("auth_multiproof_a_sibling_flipped"); p2.auth[0][0] = _flip_byte(p2.auth[0][0]); out.append((n, p2, st))
    n, p2 = P_("auth_multiproofs_swapped_a_b"); p2.auth[0], p2.auth[1] = p2.auth[1], p2.auth[0]; out.append((n, p2, st))
    n, p2 = P_("auth_multiproofs_stripped"); p2.auth = None; out.append((n, p2, st))
    st2 = copy.deepcopy(st); st2.auth = None; out.append(("statement_auth_stripped", pf, st2))

    def with_tree(name, i, **kw):
        st2 = copy.deepcopy(st)
        trees = list(st2.auth.trees)
        d = trees[i].to_json(); d.update(kw)
        trees[i] = TreeRef.from_json(d)
        st2.auth = AuthStatement(st2.auth.vu_index, tuple(trees))
        out.append((name, pf, st2))

    with_tree("statement_auth_root_a_flipped", 0, root=_flip_byte(st.auth.trees[0].root).hex())
    with_tree("statement_auth_root_b_flipped", 1, root=_flip_byte(st.auth.trees[1].root).hex())
    with_tree("statement_auth_root_y_flipped", 2, root=_flip_byte(st.auth.trees[2].root).hex())
    with_tree("statement_auth_binding_a_flipped", 0, binding=_flip_byte(st.auth.trees[0].binding).hex())
    with_tree("statement_auth_count_a_plus_1", 0, count=st.auth.trees[0].count + 1)
    with_tree("statement_auth_owner_b_as_boundary", 1, owner=-1)
    st2 = copy.deepcopy(st)
    st2.auth = AuthStatement(tuple(v + 1 for v in st.auth.vu_index), st.auth.trees)
    out.append(("statement_auth_vu_index_shifted", pf, st2))
    return out


def gen_auth_negatives(args) -> int:
    """``authentication = included`` negatives for both verifiers (vu.auth_negatives, on the trees of ``--total-vus``
    VUs or the cached trees): the prover-side ones (a Ligero-valid proof over words that are not the committed words;
    over the committed words at the wrong index; under a different root) and the file-level ``auth_mutations``.
    The honest included proof is written too (``auth_honest``)."""
    from verity_numerical.bench import instances

    from .vu import ChainRunner, auth_negatives, auth_trees_for
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R = ChainRunner(args.device, args.target, 2, "torch", zk=args.zk, mode=args.mode)
    man = _manifest(out, args, R, auth=True)
    _write_system(out, R, man)
    vu = instances.load("vu-k1536", Path(args.root))
    trees, info = auth_trees_for(vu, Path(args.root), args.total_vus, args.auth_cache, args.auth_procs)
    man["auth"] = info
    cases = auth_negatives(R, trees, vu, args.n_vus, procs=args.auth_procs)
    for name, pf, pubs, lay, auth, meta in cases:
        st = _statement_of(R, pubs, lay, R.cfg(lay.l), 1, auth=auth)
        kind = "honest" if name == "auth_honest" else "auth_negative"
        _write(out, name, st, pf, man, kind, R, args.device, meta)
    # the file-level mutations of the honest included proof
    st = read_statement((out / "auth_honest.stmt").read_bytes())
    pf = read_proof((out / "auth_honest.proof").read_bytes())
    for name, p2, st2 in auth_mutations(pf, st):
        sp, pp = out / f"mut_{name}.stmt", out / f"mut_{name}.proof"
        sp.write_bytes(statement_bytes(st2))
        pp.write_bytes(proof_bytes(p2))
        ok, why, secs = verify_files(sp, pp, args.device, runner=R)
        man["files"].append({"kind": "auth_mutation", "mutation": name, "stmt": sp.name, "proof": pp.name,
                             "stmt_sha256": sha256_file(sp), "proof_sha256": sha256_file(pp),
                             "python_verdict": {"accepted": ok, "reason": why, "seconds": secs}})
        print(f"mut {name}: python {ok} ({why})", flush=True)
    _save_manifest(out, man)
    return 0


def gen_mutations(args) -> int:
    """Mutated copies of one honest proof file (``--base STEM`` in ``--src``), each written as its own pair."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.src)
    st = read_statement((src / f"{args.base}.stmt").read_bytes())
    pf = read_proof((src / f"{args.base}.proof").read_bytes())
    R = _runner(args.device, st)
    man = {"format": "ligero-proof/v1 mutations", "base": args.base, "src": str(src), "files": []}
    for name, p2, st2 in mutations(pf, st):
        sp, pp = out / f"mut_{name}.stmt", out / f"mut_{name}.proof"
        sp.write_bytes(statement_bytes(st2))
        pp.write_bytes(proof_bytes(p2))
        ok, why, secs = verify_files(sp, pp, args.device, runner=R)
        man["files"].append({"kind": "mutation", "mutation": name, "stmt": sp.name, "proof": pp.name,
                             "stmt_sha256": sha256_file(sp), "proof_sha256": sha256_file(pp),
                             "python_verdict": {"accepted": ok, "reason": why, "seconds": secs}})
        print(f"mut {name}: python {ok} ({why})", flush=True)
    # sub-batch swap: proof i with statement j (needs a set with >= 2 honest sub-batches)
    if args.swap_src:
        ss = Path(args.swap_src)
        pairs = [("sub_00", "sub_01"), ("sub_01", "sub_00")]
        for pa, pb in pairs:
            if (ss / f"{pa}.proof").exists() and (ss / f"{pb}.stmt").exists():
                name = f"subbatch_swapped_{pa}_proof_{pb}_stmt"
                sp, pp = out / f"mut_{name}.stmt", out / f"mut_{name}.proof"
                sp.write_bytes((ss / f"{pb}.stmt").read_bytes())
                pp.write_bytes((ss / f"{pa}.proof").read_bytes())
                st_b = read_statement(sp.read_bytes())
                ok, why, secs = verify_files(sp, pp, args.device, runner=_runner(args.device, st_b))
                man["files"].append({"kind": "mutation", "mutation": name, "stmt": sp.name, "proof": pp.name,
                                     "stmt_sha256": sha256_file(sp), "proof_sha256": sha256_file(pp),
                                     "python_verdict": {"accepted": ok, "reason": why, "seconds": secs}})
                print(f"mut {name}: python {ok} ({why})", flush=True)
    _save_manifest(out, man)
    return 0


def gen_witness_mutations(args) -> int:
    """Negatives that reach the TESTS (a transcript mutation is caught earlier by the Fiat-Shamir binding): the
    witness is edited before commitment (``prove(mutate=...)``, vu.gate_vu step 3) so root, messages and openings are
    all consistent with a wrong witness -- one row per row class, a broken chain link, a nonzero c_0, and (ZK) a
    committed mask row edited after the messages were formed is impossible without the tree, so the mask rows are
    exercised by the plain value mutations instead."""
    from verity_numerical.bench import instances

    from .vu import STEPS, ChainRunner, _vu_data
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R = ChainRunner(args.device, args.target, 2, "torch", zk=args.zk, mode=args.mode)
    man = _manifest(out, args, R)
    _write_system(out, R, man)
    vu = instances.load("vu-k1536", Path(args.root))
    base = [_vu_data(vu, 0), _vu_data(vu, 1)]
    picks, seen = [], set()
    for r in R.sys.rows:
        if r.kind not in seen:
            seen.add(r.kind)
            picks.append(r)
    rng = np.random.default_rng(7)
    cases = []
    for r in picks:
        col, delta = int(rng.integers(0, 2 * STEPS)), int(rng.integers(1, 3))
        cases.append((f"row_{r.kind}_{r.name.replace('[', '_').replace(']', '').replace('.', '_')}_col{col}",
                      lambda W, idx=r.idx, col=col, delta=delta: W.__setitem__((idx, col), (int(W[idx, col]) + delta) % P)))
    c_rows = R.sys.chain["c"]
    cases.append(("chain_link_broken_cf40", lambda W: W.__setitem__((c_rows[2], 40), (int(W[c_rows[2], 40]) + 1) % P)))
    cases.append(("chain_c0_nonzero_vu1", lambda W: W.__setitem__((c_rows[1], STEPS), 1)))
    cases.append(("all_bits_of_one_column_flipped", lambda W: W.__setitem__((slice(None), 5), (1 - W[:, 5]) % P)))
    for name, mut in cases:
        pf, pubs, lay = R.prove_vus(base, check=False, mutate=mut)
        st = _statement_of(R, pubs, lay, R.cfg(lay.l), 1)
        _write(out, f"wmut_{name}", st, pf, man, "witness_mutation", R, args.device, {"mutation": name})
    _save_manifest(out, man)
    return 0


def gen_forged_interactive(args) -> int:
    """DISCREPANCY BV-D7 demonstration: interactive-mode transcripts are not transferable evidence.  In the interactive
    mode the challenges are functions of the verifier's coins alone (PROTOCOL.md 8c after b-zk-fix); a prover that
    CHOOSES the coins knows ``r, rho, rc, cols`` before committing and can build a tableau that passes every column
    test for ANY public statement -- which is exactly what the b-zk-fix simulator (``redteam/simulator.py``) does.
    Here the simulator is run as a cheating prover: honest VU operands with a WRONG claimed word (and the 52
    ``vu-k1536-neg`` claims), coins sampled by the prover itself, written as ordinary proof files.  A file verifier
    that replays the transcript's coins (Rust without ``--coins``, Python ``verify_files``) accepts them; the same
    files are rejected under any other verifier's coins.  Fiat-Shamir mode has no such transcript."""
    from verity_numerical.bench import instances
    from verity_numerical.checker.step import step_output

    from . import protocol as pm
    from .redteam import simulator
    from .vu import STEP, STEPS, ChainRunner, _vu_data
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R = ChainRunner(args.device, args.target, 2, "torch", zk=True, mode="interactive")
    man = _manifest(out, args, R)
    man["forgery"] = "redteam/simulator.simulate run as a cheating prover with prover-chosen coins (BV-D7)"
    _write_system(out, R, man)
    vu = instances.load("vu-k1536", Path(args.root))
    cases = []
    x, w, _accs, y = _vu_data(vu, 0)
    cases.append(("honest_operands_wrong_word", x, w, int(y) ^ 1, {"true_y16": int(y)}))
    if args.negatives:
        neg = instances.load("vu-k1536-neg", Path(args.root))
        cols = {n: i for i, n in enumerate(neg.columns)}
        for i in range(neg.n):
            row = neg.index[i]
            xn, wn, y16n = neg.instance(i)
            cases.append((f"neg_{i:02d}", xn, wn, int(y16n), {"neg_id": i, "mistake": row[cols["mistake"]], "verdict": row[cols["verdict"]]}))
    rng = np.random.default_rng(11)
    for name, xx, ww, claimed, meta in cases:
        lay = R.layout(1)
        cfg = R.cfg(lay.l)
        a = np.zeros((lay.l, STEP), dtype=np.int64)
        b = np.zeros((lay.l, STEP), dtype=np.int64)
        for s in range(STEPS):
            a[s], b[s] = xx[16 * s:16 * s + 16], ww[16 * s:16 * s + 16]
        yv = np.zeros(lay.l, dtype=np.int64)
        yv[STEPS - 1] = claimed
        try:
            true_word = 0
            for s in range(STEPS):
                true_word = step_output(R.p, true_word, xx[16 * s:16 * s + 16], ww[16 * s:16 * s + 16])
            meta["checker_word"] = int(true_word)
            meta["claim_is_false"] = int(true_word) != claimed
        except Exception as e:  # noqa: BLE001 -- off-domain operands: the checker itself refuses
            meta["checker_word"] = f"checker refused ({type(e).__name__})"
            meta["claim_is_false"] = True
        V = pm.Coins.sample()                                   # the PROVER plays step 0
        try:
            t0 = time.perf_counter()
            pf = simulator.simulate(V, R.p, R.sys, R.tb, cfg, lay, yv, a, b, args.device, rng=rng)
            meta["simulator_seconds"] = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            meta["simulator"] = f"raised {type(e).__name__}: {e}"
            man["files"].append({"kind": "forged_interactive", "name": name, **meta})
            print(f"forged {name}: simulator raised {type(e).__name__}: {e}", flush=True)
            continue
        if not isinstance(pf, pm.Proof):
            meta["simulator"] = f"abort transcript {pf}"
            man["files"].append({"kind": "forged_interactive", "name": name, **meta})
            continue
        ok_o, why_o = R.verify_vus(pf, (a, b, yv), lay, coins=pm.Coins.sample())
        meta["python_under_other_verifiers_coins"] = {"accepted": ok_o, "reason": why_o}
        st = _statement_of(R, (a, b, yv), lay, cfg, 1)
        _write(out, f"forged_{name}", st, pf, man, "forged_interactive", R, args.device, meta)
    _save_manifest(out, man)
    return 0


def cmd_verify(args) -> int:
    ok, why, secs = verify_files(Path(args.statement), Path(args.proof), args.device, relation=args.relation)
    res = {"accepted": ok, "reason": why, "seconds": secs, "verifier": "python (archive.direct.ligero.protocol.verify)",
           "relation": args.relation or "(from the statement)"}
    print(json.dumps(res))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))
    return 0 if ok else 1


def cmd_verify_batch(args) -> int:
    res = verify_batch_dir(Path(args.dir), args.device, args.target_bits)
    for it in res["items"]:
        print(f"{it['name']}: {'ACCEPT' if it['accepted'] else 'reject'} ({it['reason']}) {it['seconds']:.3f}s "
              f"per-proof 2^{it['per_proof_log2']:.2f} n_proofs={it['n_proofs']}", flush=True)
    print(f"{res['n_subbatches']} proofs: {res['n_accepted']} accepted, {res['n_rejected']} rejected; batch "
          f"{'ACCEPT' if res['accepted'] else 'reject'} ({res['reason']}); batch bits {res['batch_bits']:.6f} at target {res['target_bits']}",
          flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))
    else:
        print(json.dumps(res))
    return 0 if res["accepted"] else 1


def cmd_dump_system(args) -> int:
    from verity_numerical.checker.params import REAL

    from .unit import compile_unit
    data = system_bytes(compile_unit(REAL, chain=True))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_bytes(data)
    rd = read_system(data)
    print(json.dumps({"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "sys_id": rd["sys_id"].hex(),
                      "m": rd["m"], "L": len(rd["linear"]), "Q": len(rd["quadratic"])}))
    return 0


def main(argv=None) -> int:
    from verity_numerical.bench import instances
    ap = argparse.ArgumentParser(prog="python -m archive.direct.ligero.serialize")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--root", default=str(instances.DEFAULT_ROOT))
        s.add_argument("--device", default="cpu")
        s.add_argument("--target", type=float, default=-128.0)
        s.add_argument("--zk", action="store_true")
        s.add_argument("--mode", default="interactive", choices=("interactive", "fiat-shamir"))
        s.add_argument("--out", required=True)

    s = sub.add_parser("gen-set"); common(s)
    s.add_argument("--batch", type=int, default=16384)
    s.add_argument("--total-vus", type=int, default=4096)
    s = sub.add_parser("gen-negatives"); common(s)
    s = sub.add_parser("gen-small"); common(s)
    s.add_argument("--sizes", type=int, nargs="+", default=[1, 64])
    s = sub.add_parser("gen-witness-mutations"); common(s)
    s = sub.add_parser("gen-forged-interactive"); common(s)
    s.add_argument("--negatives", action="store_true", help="also the 52 vu-k1536-neg claims")
    s = sub.add_parser("gen-auth-negatives"); common(s)
    s.add_argument("--total-vus", type=int, default=64, help="VUs committed in the trees (4096 = the frozen set)")
    s.add_argument("--n-vus", type=int, default=2, help="VUs per negative proof")
    s.add_argument("--auth-cache", default=None, help="directory of cached trees (built there when absent)")
    s.add_argument("--auth-procs", type=int, default=None)
    s = sub.add_parser("gen-mutations")
    s.add_argument("--src", required=True)
    s.add_argument("--base", default="b1")
    s.add_argument("--swap-src", default=None)
    s.add_argument("--device", default="cpu")
    s.add_argument("--out", required=True)
    s = sub.add_parser("verify")
    s.add_argument("--statement", required=True)
    s.add_argument("--proof", required=True)
    s.add_argument("--device", default="cpu")
    s.add_argument("--json", default=None)
    s.add_argument("--relation", default=None, help="the relations.py name to check against (v2 / v3 statements: bf16 unless given; v4 statements name their own)")
    s = sub.add_parser("verify-batch", help="the Python batch verifier on DIR/*.{stmt,proof[,coins]} (PROTOCOL.md 8c); exit 1 on reject")
    s.add_argument("--dir", required=True)
    s.add_argument("--device", default="cpu")
    s.add_argument("--target-bits", type=float, default=128.0, help="the batch target t: reject unless sum_i 2^-b_i <= 2^-t")
    s.add_argument("--json", default=None)
    s = sub.add_parser("dump-system")
    s.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {"gen-set": gen_set, "gen-negatives": gen_negatives, "gen-small": gen_small, "gen-mutations": gen_mutations,
            "gen-witness-mutations": gen_witness_mutations, "gen-forged-interactive": gen_forged_interactive,
            "gen-auth-negatives": gen_auth_negatives, "verify": cmd_verify, "verify-batch": cmd_verify_batch,
            "dump-system": cmd_dump_system}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
