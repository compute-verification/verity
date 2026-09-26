"""V[B] as a program: typed registers and batched operations, each operation one table of one Boolean gadget.

A register holds an array of bits, bytes or F elements (GF(2^128), PROTOCOL.md §2.1). An operation is either free (a
rewiring or a GF(2)-linear map: gathers, XORs, conversions, products by constants) or a table of gadget instances whose
AND count is the gadget's (`gadgets`). Assertions are registers that must be zero. This is the circuit's "tables plus
glue" form (Project store `docs/circuit-privacy.md` I.3): each costly operation is a table, the register flow is the glue.

The builder sees only shapes, never values, so a program is a function of the bounds it was built from; `digest()` hashes
the whole op list. `evaluate` runs it natively on named inputs; `gadgets.check_program` runs its costly operations through
their Boolean circuits.
"""
from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass, field

import numpy as np

from . import gf

KINDS = ("bit", "byte", "F")


@dataclass(eq=False)
class Reg:
    prog: "Program"
    id: int
    kind: str
    shape: tuple[int, ...]

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    def __xor__(self, other: "Reg") -> "Reg":
        return self.prog.xor(self, other)

    def __getitem__(self, idx) -> "Reg":
        return self.prog.take(self, idx)

    def reshape(self, *shape) -> "Reg":
        return self.prog.reshape(self, shape)


@dataclass
class Op:
    name: str
    ins: tuple[int, ...]
    outs: tuple[int, ...]
    params: dict = field(default_factory=dict)
    component: str = ""
    ands: int = 0


def _canon(x) -> bytes:
    if isinstance(x, np.ndarray):
        return b"nd" + str(x.dtype).encode() + repr(x.shape).encode() + hashlib.sha256(np.ascontiguousarray(x).tobytes()).digest()
    if isinstance(x, dict):
        return b"{" + b",".join(_canon(k) + b":" + _canon(v) for k, v in sorted(x.items())) + b"}"
    if isinstance(x, (list, tuple)):
        return b"[" + b",".join(_canon(v) for v in x) + b"]"
    return repr(x).encode()


class Program:
    def __init__(self, name: str):
        self.name = name
        self.regs: list[Reg] = []
        self.ops: list[Op] = []
        self.inputs: dict[str, tuple[str, Reg]] = {}
        self.asserts: list[tuple[Reg, str]] = []
        self._component = ""

    # --- structure --------------------------------------------------------------------------------------------------

    def _reg(self, kind: str, shape) -> Reg:
        assert kind in KINDS
        r = Reg(self, len(self.regs), kind, tuple(int(s) for s in shape))
        self.regs.append(r)
        return r

    def _op(self, op_name, ins, outs, ands=0, **params):
        self.ops.append(Op(op_name, tuple(r.id for r in ins), tuple(r.id for r in outs), params, self._component, int(ands)))

    @contextlib.contextmanager
    def component(self, name: str):
        prev = self._component
        self._component = name if not prev else f"{prev}/{name}"
        try:
            yield
        finally:
            self._component = prev

    def input(self, name: str, role: str, kind: str, shape) -> Reg:
        assert role in ("public", "witness") and name not in self.inputs
        r = self._reg(kind, shape)
        self.inputs[name] = (role, r)
        self._op("input", (), (r,), name=name, role=role)
        return r

    def public(self, name, kind, shape) -> Reg:
        return self.input(name, "public", kind, shape)

    def witness(self, name, kind, shape) -> Reg:
        return self.input(name, "witness", kind, shape)

    def const(self, kind: str, value) -> Reg:
        v = np.asarray(value, dtype=np.uint64 if kind == "F" else np.uint8)
        shape = v.shape[:-1] if kind == "F" else v.shape
        r = self._reg(kind, shape)
        self._op("const", (), (r,), value=v)
        return r

    def fconst(self, xs) -> Reg:
        a = np.asarray(xs, dtype=object)
        return self.const("F", gf.to_arr(a.reshape(-1)).reshape(a.shape + (2,)))

    def digest(self) -> str:
        h = hashlib.sha256(self.name.encode())
        for r in self.regs:
            h.update(_canon((r.kind, r.shape)))
        for o in self.ops:
            h.update(_canon((o.name, o.ins, o.outs, o.params, o.ands)))
        for r, why in self.asserts:
            h.update(_canon((r.id, why)))
        return h.hexdigest()

    def ands(self) -> int:
        return sum(o.ands for o in self.ops)

    def costs(self, depth: int = 1) -> dict[str, int]:
        out: dict[str, int] = {}
        for o in self.ops:
            if o.ands:
                k = "/".join(o.component.split("/")[:depth]) or "-"
                out[k] = out.get(k, 0) + o.ands
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def io_sizes(self) -> dict[str, int]:
        """Bits of public input and of witness input."""
        bits = {"bit": 1, "byte": 8, "F": 128}
        out = {"public": 0, "witness": 0}
        for role, r in self.inputs.values():
            out[role] += r.size * bits[r.kind]
        return out

    # --- free operations ---------------------------------------------------------------------------------------------

    def xor(self, a: Reg, b: Reg) -> Reg:
        assert a.kind == b.kind, (a.kind, b.kind)
        r = self._reg(a.kind, np.broadcast_shapes(a.shape, b.shape))
        self._op("xor", (a, b), (r,))
        return r

    def take(self, a: Reg, idx) -> Reg:
        """A static rearrangement: `a[idx]` with numpy indexing on a's shape."""
        flat = np.arange(a.size, dtype=np.int64).reshape(a.shape)[idx]
        return self.gather(a, np.asarray(flat))

    def gather(self, a: Reg, flat_idx: np.ndarray) -> Reg:
        flat_idx = np.asarray(flat_idx, dtype=np.int64)
        assert flat_idx.size == 0 or (flat_idx.min() >= 0 and flat_idx.max() < a.size)
        r = self._reg(a.kind, flat_idx.shape)
        self._op("gather", (a,), (r,), idx=flat_idx)
        return r

    def reshape(self, a: Reg, shape) -> Reg:
        shape = tuple(shape)
        if -1 in shape:
            k = shape.index(-1)
            rest = int(np.prod([s for s in shape if s != -1]))
            shape = shape[:k] + (a.size // rest,) + shape[k + 1:]
        assert int(np.prod(shape)) == a.size
        return self.gather(a, np.arange(a.size, dtype=np.int64).reshape(shape))

    def concat(self, regs: list[Reg], axis: int = 0) -> Reg:
        kind = regs[0].kind
        assert all(r.kind == kind for r in regs)
        shapes = [r.shape for r in regs]
        out_shape = list(shapes[0])
        out_shape[axis] = sum(s[axis] for s in shapes)
        r = self._reg(kind, out_shape)
        self._op("concat", tuple(regs), (r,), axis=axis)
        return r

    def xorsum(self, a: Reg, axis: int) -> Reg:
        shape = a.shape[:axis] + a.shape[axis + 1:]
        r = self._reg(a.kind, shape)
        self._op("xorsum", (a,), (r,), axis=axis)
        return r

    def bits(self, a: Reg) -> Reg:
        """bytes (..., n) -> bits (..., 8n) LSB first; F (...) -> bits (..., 128), bit i = coefficient of x^i."""
        if a.kind == "byte":
            r = self._reg("bit", a.shape[:-1] + (8 * a.shape[-1],))
        else:
            assert a.kind == "F"
            r = self._reg("bit", a.shape + (128,))
        self._op("bits", (a,), (r,))
        return r

    def bytes_(self, a: Reg) -> Reg:
        """bits (..., 8n) -> bytes (..., n); F (...) -> bytes (..., 16) (le64 lo, le64 hi)."""
        if a.kind == "bit":
            assert a.shape[-1] % 8 == 0
            r = self._reg("byte", a.shape[:-1] + (a.shape[-1] // 8,))
        else:
            r = self._reg("byte", a.shape + (16,))
        self._op("bytes", (a,), (r,))
        return r

    def field(self, a: Reg) -> Reg:
        """bytes (..., 16) or bits (..., 128) -> F (...)."""
        assert a.shape[-1] == (16 if a.kind == "byte" else 128)
        r = self._reg("F", a.shape[:-1])
        self._op("field", (a,), (r,))
        return r

    def fmulc(self, a: Reg, c: int) -> Reg:
        """a · c for a constant c: GF(2)-linear, no AND."""
        r = self._reg("F", a.shape)
        self._op("fmulc", (a,), (r,), c=int(c))
        return r

    def linmap(self, a: Reg, mat: np.ndarray) -> Reg:
        """bits (..., l) -> bits (..., n) through a constant GF(2) matrix (n, l): XORs only."""
        mat = np.asarray(mat, dtype=np.uint8)
        assert a.kind == "bit" and a.shape[-1] == mat.shape[1]
        r = self._reg("bit", a.shape[:-1] + (mat.shape[0],))
        self._op("linmap", (a,), (r,), mat=mat)
        return r

    def bnot(self, a: Reg) -> Reg:
        r = self._reg("bit", a.shape)
        self._op("bnot", (a,), (r,))
        return r

    def assert_zero(self, a: Reg, why: str):
        self.asserts.append((a, why))

    def assert_eq(self, a: Reg, b: Reg, why: str):
        self.assert_zero(self.xor(a, b), why)

    # --- operations with ANDs -------------------------------------------------------------------------------------------

    def fmul(self, a: Reg, b: Reg) -> Reg:
        from .gadgets import GF_MUL_ANDS
        assert a.kind == b.kind == "F"
        shape = np.broadcast_shapes(a.shape, b.shape)
        r = self._reg("F", shape)
        self._op("fmul", (a, b), (r,), ands=int(np.prod(shape, dtype=np.int64)) * GF_MUL_ANDS)
        return r

    def band(self, a: Reg, b: Reg) -> Reg:
        assert a.kind == b.kind == "bit"
        shape = np.broadcast_shapes(a.shape, b.shape)
        r = self._reg("bit", shape)
        self._op("band", (a, b), (r,), ands=int(np.prod(shape, dtype=np.int64)))
        return r

    def fscale(self, bit: Reg, a: Reg) -> Reg:
        """bit · a: one AND per bit of a."""
        assert bit.kind == "bit" and a.kind == "F"
        shape = np.broadcast_shapes(bit.shape, a.shape)
        r = self._reg("F", shape)
        self._op("fscale", (bit, a), (r,), ands=128 * int(np.prod(shape, dtype=np.int64)))
        return r

    def sha(self, algo: str, state: Reg, blocks: Reg) -> Reg:
        """Merkle–Damgård compressions: state (..., H) bytes, blocks (..., nb, B) bytes -> state (..., H)."""
        from .gadgets import sha_ands, SHA_STATE, SHA_BLOCK
        H, Bk = SHA_STATE[algo], SHA_BLOCK[algo]
        assert state.kind == blocks.kind == "byte" and state.shape[-1] == H and blocks.shape[-1] == Bk
        lead = np.broadcast_shapes(state.shape[:-1], blocks.shape[:-2])
        r = self._reg("byte", lead + (H,))
        n = int(np.prod(lead, dtype=np.int64)) * blocks.shape[-2]
        self._op(algo, (state, blocks), (r,), ands=n * sha_ands(algo))
        return r

    def benes(self, x: Reg, sw: Reg) -> Reg:
        """A Beneš network over x (n, w) bits with switch settings sw (2 log n - 1, n/2)."""
        n, w = x.shape
        stages = 2 * (n.bit_length() - 1) - 1
        assert x.kind == sw.kind == "bit" and n & (n - 1) == 0 and sw.shape == (stages, n // 2)
        r = self._reg("bit", (n, w))
        self._op("benes", (x, sw), (r,), ands=stages * (n // 2) * w)
        return r


# ---------------------------------------------------------------------------------------------------------------------
# native evaluation
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Result:
    vals: list
    failed: list[tuple[str, int]]

    @property
    def accepted(self) -> bool:
        return not self.failed


def _bits_le(a: np.ndarray) -> np.ndarray:
    return np.unpackbits(a[..., None], axis=-1, bitorder="little").reshape(a.shape[:-1] + (8 * a.shape[-1],))


def _f_bits(a: np.ndarray) -> np.ndarray:
    return _bits_le(np.ascontiguousarray(a).view(np.uint8).reshape(a.shape[:-1] + (16,)))


def _f_of_bytes(b: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(b).view("<u8").reshape(b.shape[:-1] + (2,)).astype(np.uint64)


def benes_stage_perms(n: int):
    """For each of the 2 log n - 1 switch layers, the static permutation applied just before it (None: identity)."""
    L = n.bit_length() - 1
    out = [None]
    idx = np.arange(n)
    for l in range(L - 1):
        B = n >> l
        blk, p = divmod(idx, B)
        perm_to = blk * B + (p % 2) * (B // 2) + p // 2
        inv = np.empty(n, dtype=np.int64)
        inv[perm_to] = idx
        out.append(inv)
    for l in reversed(range(L - 1)):
        B = n >> l
        blk, p = divmod(idx, B)
        src = blk * B + (p % 2) * (B // 2) + p // 2
        out.append(src)
    return out


def benes_apply(x: np.ndarray, sw: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    x = x.copy()
    for perm, s in zip(benes_stage_perms(n), sw):
        if perm is not None:
            x = x[perm]
        a, b = x[0::2], x[1::2]
        d = (a ^ b) * s.astype(x.dtype)[:, None]
        x[0::2] ^= d
        x[1::2] ^= d
    return x


def evaluate(prog: Program, inputs: dict[str, np.ndarray], stop_early: bool = False) -> Result:
    from .gadgets import sha_native
    vals: list = [None] * len(prog.regs)
    shapes = {r.id: r for r in prog.regs}
    for o in prog.ops:
        ins = [vals[i] for i in o.ins]
        out = shapes[o.outs[0]] if o.outs else None
        if o.name == "input":
            v = np.asarray(inputs[o.params["name"]])
            want = out.shape + ((2,) if out.kind == "F" else ())
            if v.shape != want:
                raise ValueError(f"input {o.params['name']}: shape {v.shape}, want {want}")
            v = v.astype(np.uint64 if out.kind == "F" else np.uint8)
        elif o.name == "const":
            v = o.params["value"]
        elif o.name == "xor":
            v = ins[0] ^ ins[1]
        elif o.name == "gather":
            a = ins[0]
            flat = a.reshape((-1, 2)) if out.kind == "F" else a.reshape(-1)
            v = flat[o.params["idx"]]
        elif o.name == "concat":
            v = np.concatenate([np.broadcast_to(x, x.shape) for x in ins], axis=o.params["axis"])
        elif o.name == "xorsum":
            v = np.bitwise_xor.reduce(ins[0], axis=o.params["axis"])
        elif o.name == "bits":
            a = ins[0]
            v = _bits_le(a) if a.dtype == np.uint8 else _f_bits(a)
        elif o.name == "bytes":
            a = ins[0]
            if a.dtype == np.uint8:
                v = np.packbits(a.reshape(a.shape[:-1] + (-1, 8)), axis=-1, bitorder="little")[..., 0]
            else:
                v = np.ascontiguousarray(a).view(np.uint8).reshape(a.shape[:-1] + (16,))
        elif o.name == "field":
            a = ins[0]
            if a.shape[-1] == 128 and out.shape == a.shape[:-1] and shapes[o.ins[0]].kind == "bit":
                a = np.packbits(a.reshape(a.shape[:-1] + (16, 8)), axis=-1, bitorder="little")[..., 0]
            v = _f_of_bytes(a)
        elif o.name == "fmulc":
            v = gf.vmul(ins[0], gf.to_arr([o.params["c"]])[0])
        elif o.name == "linmap":
            v = ((ins[0].astype(np.int32) @ o.params["mat"].T.astype(np.int32)) & 1).astype(np.uint8)
        elif o.name == "bnot":
            v = ins[0] ^ np.uint8(1)
        elif o.name == "fmul":
            v = gf.vmul(ins[0], ins[1])
        elif o.name == "band":
            v = ins[0] & ins[1]
        elif o.name == "fscale":
            bit, a = ins
            v = a * bit.astype(np.uint64)[..., None]
        elif o.name in ("sha256", "sha512"):
            v = sha_native(o.name, ins[0], ins[1])
        elif o.name == "benes":
            v = benes_apply(ins[0], ins[1])
        else:
            raise ValueError(o.name)
        if out is not None:
            want = out.shape + ((2,) if out.kind == "F" else ())
            v = np.broadcast_to(v, want) if v.shape != want else v
            vals[o.outs[0]] = v
    failed = []
    for r, why in prog.asserts:
        bad = int(np.count_nonzero(np.any(vals[r.id].reshape(r.size, -1) != 0, axis=-1)))
        if bad:
            failed.append((why, bad))
    return Result(vals, failed)
