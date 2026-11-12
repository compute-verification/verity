"""A PTX interpreter wide enough for the four kernels vLLM's `rejection_sample` launches, as Triton 3.7.1 emits them.

Every thread of every block is a row of numpy arrays; the threads at the lowest program counter step together, so the
lanes of a warp meet at each `shfl.sync` and the warps of a block at each `bar.sync` whenever the control flow is uniform.
A `shfl.sync` whose partner lane is somewhere else is recorded in `Run.undefined` (CUDA leaves its result undefined) and
reads the partner's register as it stands.

Arithmetic: integers wrap at their width; `add/sub/mul.f32` are numpy float32 (round to nearest, subnormals kept);
`fma.rn.f32` is `topp_split.fma32`; `fma.rn.ftz.f32` and `add.rz.ftz.f32` are `numerics.libdevice`'s; `ex2.approx.f32` is
MUFU.EX2 (`ops.mufu`) with ptxas's expansion below -126; `max.f32` returns the operand that is no NaN.
"""
from __future__ import annotations

import re

import numpy as np

from verity.definitions.numerics.libdevice import add_rz_f32, fma_f32
from verity.definitions.ops.mufu import mufu_ex2_words, table
from verity.definitions.registry.topp_split import fma32

U64 = np.uint64
M32 = U64(0xFFFFFFFF)
M64 = (1 << 64) - 1
END = 1 << 60


def _split(rest: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in rest:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _operand(tok: str):
    tok = tok.strip()
    if tok.startswith("["):
        inner = tok[1:-1].replace(" ", "")
        base, _, off = inner.partition("+")
        return ("mem", _operand(base), int(off or 0))
    if tok.startswith("{"):
        return ("vec", [t.strip() for t in tok[1:-1].split(",")])
    if re.fullmatch(r"%(r|rd|p)\d+", tok):
        return ("reg", tok)
    if tok in ("%tid.x", "%ctaid.x", "%ctaid.y"):
        return ("special", tok)
    if re.fullmatch(r"0f[0-9A-Fa-f]{8}", tok):
        return ("imm", int(tok[2:], 16))
    if re.fullmatch(r"-?\d+", tok):
        return ("imm", int(tok) & M64)
    if re.fullmatch(r"0x[0-9A-Fa-f]+", tok):
        return ("imm", int(tok, 16))
    return ("sym", tok)


def parse(text: str):
    """(instructions, labels, number of parameters): an instruction is (predicate or None, opcode, operands)."""
    prog, labels, nparams, inside = [], {}, 0, False
    for raw in text.splitlines():
        line = raw.split("//")[0].strip()
        if not line:
            continue
        if not inside:
            nparams += line.startswith(".param")
            inside = line == "{"
            continue
        if line == "}":
            break
        if line.endswith(":"):
            labels[line[:-1]] = len(prog)
            continue
        if line.startswith(".") or not line.endswith(";"):
            continue
        m = re.match(r"(?:@(!?)(%p\d+)\s+)?(\S+)\s*(.*);$", line)
        pred = (m[2], bool(m[1])) if m[2] else None
        op = m[3]
        prog.append((pred, op, [_operand(t) for t in _split(m[4])] if not op.startswith("bra") else [("sym", m[4].strip())]))
    return prog, labels, nparams


def _f32(x):
    return (x & M32).astype(np.uint32).view(np.float32)


def _bits(f):
    return np.ascontiguousarray(f, dtype=np.float32).view(np.uint32).astype(U64)


def _signed(x, w):
    return (x & M32).astype(np.uint32).view(np.int32).astype(np.int64) if w == 32 else np.ascontiguousarray(x).view(np.int64)


def _ex2(t):
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        small = t < np.float32(-126.0)
        half = np.ascontiguousarray(np.where(small, (t * np.float32(0.5)).astype(np.float32), t), dtype=np.float32)
        e = mufu_ex2_words(half.view(np.uint32), table("ex2")).view(np.float32)
        return np.where(small, (e * e).astype(np.float32), e).astype(np.float32)


_FCMP = {"eq": lambda a, b: a == b, "ne": lambda a, b: (a != b) & ~(np.isnan(a) | np.isnan(b)), "lt": lambda a, b: a < b,
         "le": lambda a, b: a <= b, "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b,
         "ltu": lambda a, b: ~(a >= b), "leu": lambda a, b: ~(a > b), "gtu": lambda a, b: ~(a <= b),
         "geu": lambda a, b: ~(a < b), "nan": lambda a, b: np.isnan(a) | np.isnan(b)}
_ICMP = {"eq": lambda a, b: a == b, "ne": lambda a, b: a != b, "lt": lambda a, b: a < b, "le": lambda a, b: a <= b,
         "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b}


class Run:
    def __init__(self, text: str, threads: int, grid, params, mem32: np.ndarray, smem_words: int = 64, max_steps: int = 2_000_000):
        self.prog, self.labels, nparams = parse(text)
        self.params = [int(p) & M64 for p in params] + [0] * (nparams - len(params))
        assert len(self.params) == nparams, (len(params), nparams)
        gx, gy = (list(grid) + [1])[:2]
        blocks = gx * gy
        n = blocks * threads
        self.n, self.mem = n, mem32
        self.block = np.repeat(np.arange(blocks), threads)
        self.special = {"%tid.x": np.tile(np.arange(threads), blocks).astype(U64), "%ctaid.x": (self.block % gx).astype(U64),
                        "%ctaid.y": (self.block // gx).astype(U64)}
        self.smem = np.zeros((blocks, smem_words), np.uint32)
        self.reg: dict[str, np.ndarray] = {}
        self.pc = np.zeros(n, np.int64)
        self.undefined: list[tuple[int, int]] = []
        self.steps = 0
        while True:
            cur = int(self.pc.min())
            if cur >= END:
                break
            idx = np.flatnonzero(self.pc == cur)
            self.step(cur, idx)
            self.steps += 1
            assert self.steps < max_steps, "step limit"

    def get(self, o, idx):
        kind = o[0]
        if kind == "reg":
            r = self.reg.get(o[1])
            if r is None:
                r = self.reg[o[1]] = np.zeros(self.n, bool if o[1].startswith("%p") else U64)
            return r[idx]
        if kind == "imm":
            return np.full(len(idx), o[1], U64)
        if kind == "special":
            return self.special[o[1]][idx]
        if kind == "sym" and o[1] == "global_smem":
            return np.zeros(len(idx), U64)
        raise ValueError(o)

    def put(self, o, idx, value):
        r = self.reg.get(o[1])
        if r is None:
            r = self.reg[o[1]] = np.zeros(self.n, bool if o[1].startswith("%p") else U64)
        r[idx] = value

    def addr(self, o, idx):
        return (self.get(o[1], idx) + U64(o[2])).astype(np.int64)

    def step(self, cur: int, idx: np.ndarray) -> None:
        pred, op, a = self.prog[cur]
        self.pc[idx] = cur + 1
        if op == "bra.uni" or op == "bra":
            take = idx if pred is None else idx[self.get(("reg", pred[0]), idx) != pred[1]]
            self.pc[take] = self.labels[a[0][1]]
            return
        if op.startswith("ret"):
            self.pc[idx] = END
            return
        if pred is not None:
            idx = idx[self.get(("reg", pred[0]), idx) != pred[1]]
            if not len(idx):
                return
        parts = op.split(".")
        head, typ = parts[0], parts[-1]
        w = 64 if typ.endswith("64") else 32
        mask = U64(M64) if w == 64 else M32
        if op == "bar.sync":
            return
        if head == "ld":
            if parts[1] == "param":
                self.put(a[0], idx, U64(self.params[int(a[1][1][1].rsplit("_", 1)[1])]) & mask)
            elif parts[1] == "shared":
                at = self.addr(a[1], idx)
                self.put(a[0], idx, self.smem[self.block[idx], at >> 2].astype(U64))
            else:
                at = self.addr(a[1], idx)
                assert not (at & 3).any() and (at >= 0).all() and ((at >> 2) + (4 if "v4" in parts else w // 32) <= len(self.mem)).all(), (cur, op)
                names = a[0][1]
                if "v4" in parts:
                    for k, name in enumerate(names):
                        self.put(("reg", name), idx, self.mem[(at >> 2) + k].astype(U64))
                elif w == 64:
                    self.put(("reg", names[0]), idx, self.mem[at >> 2].astype(U64) | (self.mem[(at >> 2) + 1].astype(U64) << U64(32)))
                else:
                    self.put(("reg", names[0]), idx, self.mem[at >> 2].astype(U64))
            return
        if head == "st":
            at = self.addr(a[0], idx)
            if parts[1] == "shared":
                self.smem[self.block[idx], at >> 2] = (self.get(a[1], idx) & M32).astype(np.uint32)
                return
            assert not (at & 3).any() and (at >= 0).all() and ((at >> 2) + w // 32 <= len(self.mem)).all(), (cur, op)
            assert len(np.unique(at)) == len(at), ("two threads store to one address", cur, op)
            v = self.get(("reg", a[1][1][0]), idx)
            self.mem[at >> 2] = (v & M32).astype(np.uint32)
            if w == 64:
                self.mem[(at >> 2) + 1] = (v >> U64(32)).astype(np.uint32)
            return
        if head == "shfl":
            partner = idx ^ int(a[2][1])
            away = int((self.pc[partner] != cur + 1).sum())
            if away:
                self.undefined.append((cur, away))
            self.put(a[0], idx, self.get(a[1], partner))
            return
        if head == "mov" and a[0][0] == "vec":
            x = self.get(a[1], idx)
            for k, name in enumerate(a[0][1]):
                if name != "_":
                    self.put(("reg", name), idx, (x >> U64(32 * k)) & M32)
            return
        if head == "mov":
            self.put(a[0], idx, self.get(a[1], idx) != 0 if typ == "pred" else self.get(a[1], idx) & mask)
            return
        if typ == "pred":
            x = self.get(a[1], idx)
            if head == "not":
                self.put(a[0], idx, ~x)
            else:
                y = self.get(a[2], idx)
                self.put(a[0], idx, x & y if head == "and" else x | y if head == "or" else x ^ y)
            return
        if head == "setp":
            x, y = self.get(a[1], idx), self.get(a[2], idx)
            if typ == "f32":
                with np.errstate(invalid="ignore"):
                    self.put(a[0], idx, _FCMP[parts[1]](_f32(x), _f32(y)))
            elif typ[0] == "s":
                self.put(a[0], idx, _ICMP[parts[1]](_signed(x, w), _signed(y, w)))
            else:
                self.put(a[0], idx, _ICMP[parts[1]](x & mask, y & mask))
            return
        if head == "selp":
            self.put(a[0], idx, np.where(self.get(a[3], idx), self.get(a[1], idx), self.get(a[2], idx)) & mask)
            return
        if head == "cvt":
            x = self.get(a[1], idx)
            src = parts[-1]
            if parts[1] == "rn" and parts[2] == "f32" and src == "s32":
                self.put(a[0], idx, _bits(_signed(x, 32).astype(np.float32)))
            elif parts[1] == "s64" and src == "s32":
                self.put(a[0], idx, _signed(x, 32).view(U64))
            elif (parts[1], src) in (("u64", "u32"), ("u32", "u64")):
                self.put(a[0], idx, x & M32)
            else:
                raise NotImplementedError(op)
            return
        if typ == "f32":
            x = _f32(self.get(a[1], idx))
            with np.errstate(invalid="ignore", over="ignore", under="ignore"):
                if op == "ex2.approx.f32":
                    r = _ex2(x)
                else:
                    y = _f32(self.get(a[2], idx))
                    if op == "add.f32":
                        r = x + y
                    elif op == "sub.f32":
                        r = x - y
                    elif op == "mul.f32":
                        r = x * y
                    elif op == "max.f32":
                        r = np.fmax(x, y)
                    elif op == "fma.rn.f32":
                        r = fma32(x, y, _f32(self.get(a[3], idx)))
                    elif op == "fma.rn.ftz.f32":
                        r = fma_f32(x, y, _f32(self.get(a[3], idx)))
                    elif op == "add.rz.ftz.f32":
                        r = add_rz_f32(x, y)
                    else:
                        raise NotImplementedError(op)
            self.put(a[0], idx, _bits(r))
            return
        x = self.get(a[1], idx)
        if head == "not":
            self.put(a[0], idx, ~x & mask)
            return
        y = self.get(a[2], idx)
        if op.startswith("mul.wide") or op.startswith("mad.wide"):
            p = (_signed(x, 32) * _signed(y, 32)).view(U64) if typ == "s32" else (x & M32) * (y & M32)
            self.put(a[0], idx, p + self.get(a[3], idx) if head == "mad" else p)
        elif op.startswith("mul.hi"):
            assert typ == "u32"
            self.put(a[0], idx, ((x & M32) * (y & M32)) >> U64(32))
        elif op.startswith("mul.lo"):
            self.put(a[0], idx, (x * y) & mask)
        elif head == "add":
            self.put(a[0], idx, (x + y) & mask)
        elif head == "sub":
            self.put(a[0], idx, (x - y) & mask)
        elif head in ("and", "or", "xor"):
            self.put(a[0], idx, (x & y if head == "and" else x | y if head == "or" else x ^ y) & mask)
        elif head == "shl":
            self.put(a[0], idx, np.where(y >= U64(w), U64(0), (x << np.minimum(y, U64(63))) & mask))
        elif head == "shr" and typ[0] == "s":
            self.put(a[0], idx, (_signed(x, w) >> np.minimum(y, U64(w - 1)).astype(np.int64)).view(U64) & mask)
        elif head == "shr":
            self.put(a[0], idx, np.where(y >= U64(w), U64(0), (x & mask) >> np.minimum(y, U64(63))))
        elif op == "max.s32":
            self.put(a[0], idx, np.where(_signed(x, 32) > _signed(y, 32), x, y) & mask)
        else:
            raise NotImplementedError(op)


class Heap:
    """Tensors in one 32-bit-word memory, each at a 64-byte boundary, the first past address 4096."""

    def __init__(self, words: int):
        self.mem = np.full(words, 0xA5A5A5A5, np.uint32)
        self.top = 4096
        self.at: dict[str, tuple[int, np.dtype, tuple]] = {}

    def put(self, name: str, array: np.ndarray) -> int:
        array = np.ascontiguousarray(array)
        base = self.top
        raw = np.frombuffer(array.tobytes(), np.uint32)
        self.mem[base >> 2:(base >> 2) + len(raw)] = raw
        self.at[name] = (base, array.dtype, array.shape)
        self.top = (base + array.nbytes + 127) & ~63
        assert self.top >> 2 <= len(self.mem), "heap too small"
        return base

    def empty(self, name: str, shape, dtype) -> int:
        base = self.top
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        self.at[name] = (base, np.dtype(dtype), tuple(shape))
        self.top = (base + nbytes + 127) & ~63
        assert self.top >> 2 <= len(self.mem), "heap too small"
        return base

    def get(self, name: str) -> np.ndarray:
        base, dtype, shape = self.at[name]
        n = int(np.prod(shape)) * dtype.itemsize // 4
        return np.frombuffer(self.mem[base >> 2:(base >> 2) + n].tobytes(), dtype).reshape(shape).copy()
