"""`h1c.cu` from torch: the served window's tables, `served_h1/commit.py`'s format bit for bit, built on the device.

    lib = H1C("libh1c.so", key)                  # the window's 64-byte key (the salts'); hm96's byte tables go up once
    st = lib.set([Tab(...), ...])                # one launch set: its device table and buffers, allocated once
    lib.build(st, step)                          # domains, segments, leaves, levels on the current stream; step (1,) int32
    st.root(i)                                   # table i's root: 8 int64 big-endian words, a device view

A table's domain is `commit.domain(fields, count, kind)`, with `splice` naming the fields that move with the step (each
a s + b), so a step's sequence is the same launches with the step read from the device.  Its label is `lab[0]` + str(s)
+ `lab[1]` (or `lab[0]` alone).  `Tab.out` points a table's root into another table's leaves (a given table: no data).
"""
from __future__ import annotations

import ctypes as C
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks" / "pouw" / "served_h1"))

import commit as CM  # noqa: E402
from verity.ml.commitments import identity as ID  # noqa: E402
from verity.ml.commitments import merkle as MK  # noqa: E402
from verity.ml.commitments import rowleaf as RL  # noqa: E402
from verity.ml.commitments.indexed import RangeIndexedDomain  # noqa: E402
from verity.ml.commitments.merkle import merkle_depth  # noqa: E402

U64_8, U64_16 = C.c_uint64 * 8, C.c_uint64 * 16


class _Const(C.Structure):
    _fields_ = [("salt_prefix", U64_16), ("leaf_prefix", U64_16), ("head", C.c_uint8 * 128), ("frame", C.c_uint8 * 64),
                ("schema", C.c_uint8 * 32), ("qtag", C.c_uint8 * 48), ("stag", C.c_uint8 * 48), ("head_len", C.c_int),
                ("frame_len", C.c_int), ("schema_len", C.c_int), ("qtag_len", C.c_int), ("stag_len", C.c_int)]


class _Dom(C.Structure):
    _fields_ = [("bind", C.c_void_p), ("js", C.c_void_p * 3), ("bind_len", C.c_int), ("js_len", C.c_int * 3),
                ("nval", C.c_int), ("va", C.c_uint64 * 2), ("vb", C.c_uint64 * 2), ("pos", U64_8)]


class _Tab(C.Structure):
    _fields_ = [("data", C.c_void_p), ("tail", C.c_void_p), ("stride", C.c_longlong), ("tail_stride", C.c_longlong),
                ("lab", C.c_void_p * 2), ("lab_len", C.c_int * 2), ("labs", C.c_int), ("len", C.c_int), ("body", C.c_int),
                ("count", C.c_int), ("depth", C.c_int), ("dm", _Dom), ("dom", C.c_void_p), ("level", C.c_void_p),
                ("segs", C.c_void_p), ("root", C.c_void_p)]


def words(b: bytes) -> np.ndarray:
    """Bytes (a multiple of 8) as big-endian u64 words, in int64."""
    return np.frombuffer(b, dtype=">u8").astype("<u8").view("<i8").copy()


def unwords(w) -> bytes:
    a = w.cpu().numpy() if isinstance(w, torch.Tensor) else np.asarray(w)
    return a.view("<u8").astype(">u8").tobytes()


def _bytes_into(arr, b: bytes) -> None:
    if len(b) > len(arr):
        raise ValueError(f"{len(b)} bytes into {len(arr)}")
    for i, x in enumerate(b):
        arr[i] = x


@dataclass
class Dom:
    kind: str
    fields: dict
    count: int
    splice: dict = field(default_factory=dict)   # field -> (a, b): its value a s + b

    def at(self, s: int) -> CM.CommitmentDomain:
        f = dict(self.fields)
        for k, (a, b) in self.splice.items():
            f[k] = a * s + b
        return CM.domain(f, self.count, self.kind)

    def template(self):
        f, sent = dict(self.fields), {}
        for i, (k, ab) in enumerate(self.splice.items()):
            v = 987654321012345 + 1111 * i
            f[k] = v
            sent[str(v).encode()] = ab
        js = ID.canonical_json_bytes(f)
        pieces, vals, rest = [], [], js
        for t in sorted(sent, key=js.index):
            pre, rest = rest.split(t, 1)
            pieces.append(pre)
            vals.append(sent[t])
        pieces.append(rest)
        tag = f"verity/pouw/served-h1/{self.kind}/v0".encode()
        bind = ID._TAGGED_SHA512_PREFIX + len(tag).to_bytes(4, "big") + tag
        return bind, pieces, vals, RangeIndexedDomain(0, self.count).identity_sha512()


@dataclass
class Tab:
    dom: Dom
    count: int
    data: torch.Tensor | None = None     # row i's first `body` bytes at data + i stride (bytes)
    stride: int = 0
    body: int = 0
    length: int = 0
    tail: torch.Tensor | None = None     # then length - body bytes at tail + i tail_stride
    tail_stride: int = 0
    lab: tuple = ("", "")                 # lab[0] + str(s) + lab[1], or lab[0] alone (labs False)
    labs: bool = True
    out: int | None = None               # where the root goes (a device address), else the set's own


class Set:
    def __init__(self, tabs: list[Tab], dev):
        self.tabs, self.n = tabs, len(tabs)
        blobs, offs = [], {}

        def put(b: bytes) -> int:
            if b not in offs:
                offs[b] = sum(len(x) for x in blobs)
                blobs.append(b + bytes(-len(b) % 8))
            return offs[b]

        tpls = [t.dom.template() for t in tabs]
        for (bind, pieces, _, _), t in zip(tpls, tabs):
            put(bind)
            for j in range(3):
                put(pieces[j] if j < len(pieces) else b"")
            for x in t.lab:
                put(x.encode())
        self.arena = torch.frombuffer(bytearray(b"".join(blobs) or bytes(8)), dtype=torch.uint8).to(dev)
        base = self.arena.data_ptr()
        depths = [merkle_depth(t.count) for t in tabs]
        nseg = [(t.length + 1023) // 1024 if t.data is not None else 0 for t in tabs]
        self.level_off, at = [], 0
        for d in depths:
            self.level_off.append(at)
            at += 8 * (2 * (1 << d) - 1)
        self.level = torch.empty(max(at, 8), dtype=torch.int64, device=dev)
        self.seg_off, at = [], 0
        for t, ns in zip(tabs, nseg):
            self.seg_off.append(at)
            at += 8 * t.count * ns
        self.segs = torch.empty(max(at, 8), dtype=torch.int64, device=dev)
        self.dom = torch.empty(8 * self.n, dtype=torch.int64, device=dev)
        self.roots = torch.empty(8 * self.n, dtype=torch.int64, device=dev)
        host = (_Tab * self.n)()
        for i, (t, (bind, pieces, vals, pos), d) in enumerate(zip(tabs, tpls, depths)):
            dm = _Dom()
            dm.bind, dm.bind_len = base + offs[bind], len(bind)
            for j in range(3):
                p = pieces[j] if j < len(pieces) else b""
                dm.js[j], dm.js_len[j] = base + offs[p], len(p)
            dm.nval = len(vals)
            for j, (a, b) in enumerate(vals):
                dm.va[j], dm.vb[j] = a, b
            for j, w in enumerate(np.frombuffer(pos, ">u8")):
                dm.pos[j] = int(w)
            x = _Tab()
            x.data = t.data.data_ptr() if t.data is not None else None
            x.tail = t.tail.data_ptr() if t.tail is not None else None
            x.stride, x.tail_stride = t.stride, t.tail_stride
            for j in range(2):
                x.lab[j], x.lab_len[j] = base + offs[t.lab[j].encode()], len(t.lab[j].encode())
            x.labs = int(t.labs)
            x.len, x.body, x.count, x.depth = t.length, t.body, t.count, d
            x.dm = dm
            x.dom = self.dom.data_ptr() + 64 * i
            x.level = self.level.data_ptr() + 8 * self.level_off[i]
            x.segs = self.segs.data_ptr() + 8 * self.seg_off[i]
            x.root = t.out if t.out is not None else self.roots.data_ptr() + 64 * i
            host[i] = x
        self.host = host
        self.dev = torch.frombuffer(bytearray(bytes(host)), dtype=torch.uint8).to(dev)

    def root(self, i: int) -> torch.Tensor:
        return self.roots[8 * i:8 * i + 8]

    def leaves_addr(self, i: int) -> int:
        """Table i's level 0 (its leaves) as a device address: where given leaves go."""
        return self.level.data_ptr() + 8 * self.level_off[i]

    def leaf(self, i: int, r: int) -> torch.Tensor:
        o = self.level_off[i] + 8 * r
        return self.level[o:o + 8]


class H1C:
    def __init__(self, path: str, key: bytes, dev=None):
        dev = dev or torch.device("cuda")
        self.lib = C.CDLL(path)
        self.lib.h1c_build.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_void_p]
        self.lib.h1c_const.argtypes = [C.c_void_p]
        sz = (C.c_int * 3)()
        self.lib.h1c_sizes(sz)
        if tuple(sz) != (C.sizeof(_Const), C.sizeof(_Dom), C.sizeof(_Tab)):
            raise RuntimeError(f"h1c.cu's structs are {tuple(sz)} bytes, here {(C.sizeof(_Const), C.sizeof(_Dom), C.sizeof(_Tab))}")
        inst = CM.INST
        k = _Const()
        for j, w in enumerate(np.frombuffer(inst.salt_prefix, ">u8")):
            k.salt_prefix[j] = int(w)
        for j, w in enumerate(np.frombuffer(inst.leaf_prefix(CM.KEY), ">u8")):
            k.leaf_prefix[j] = int(w)
        head = CM.SALT_TAG + bytes(key)
        for name, b in (("head", head), ("frame", MK._FRAME), ("schema", CM.SCHEMA.encode()),
                        ("qtag", RL.SHA512_ROW_SEG_TAG), ("stag", RL.SHA512_ROW_SEG_SEGMENT_TAG)):
            _bytes_into(getattr(k, name), b)
            setattr(k, f"{name}_len", len(b))
        rc = self.lib.h1c_const(C.byref(k))
        if rc:
            raise RuntimeError(f"h1c_const returned {rc}")
        tabs = inst.byte_tables(CM.KEY)
        raw = b"".join(v.to_bytes(64, "little") for row in tabs for v in row)
        self.btab = torch.from_numpy(words(raw)).to(dev)
        self.dev = dev

    def set(self, tabs: list[Tab]) -> Set:
        return Set(tabs, self.dev)

    def build(self, st: Set, step: torch.Tensor, domains: bool = True) -> None:
        rc = self.lib.h1c_build(C.addressof(st.host), st.dev.data_ptr(), st.n, step.data_ptr(), self.btab.data_ptr(),
                                int(domains), torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"h1c_build returned {rc}")
