"""libvc's Python binding (ctypes): plans built from the references' own prefix chunks, launches on raw device pointers.

A committer passes device pointers as ints (``tensor.data_ptr()``) and its stream as an int
(``torch.cuda.current_stream().cuda_stream``); nothing here allocates on the device or synchronizes. Every prefix the
library hashes comes from its owner's reference: ``rowleaf`` (P, Q), ``hm96.SHA512`` (the salt prefix, the leaf prefix and
the key) and ``pearl_c.S512_LABELS`` (-h3's seed-A label), so a plan is the reference's format by construction. `Cuda` is a
minimal libcudart wrapper for the tests and the conformance run.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path

from verity_config.settings import of

HERE = Path(__file__).resolve().parent
settings = of(__file__)
ABI = 1
ROW_NONE, ROW_V2, ROW_SEG, ROW_TILE = 0, 1, 2, 3
SALT_GIVEN, SALT_PRF, SALT_CHACHA64, SALT_CHACHA_RFC = 0, 1, 2, 3
FORM_AUTO, FORM_SPLIT, FORM_FUSED, FORM_FOLD, FORM_SPEC = 0, 1, 2, 3, 4
SALT_SOURCES = {"given": SALT_GIVEN, "prf": SALT_PRF, "chacha64": SALT_CHACHA64, "chacha-rfc": SALT_CHACHA_RFC}
SCHEMA_LENGTHS = (18, 22)
REC_BYTES = 192
TILE = 64


@dataclass(frozen=True)
class Spec:
    """What a plan is built from (`vc_spec`), every block already the reference's bytes."""

    domain_id: bytes
    schema: bytes | None = None
    salt_prefix: bytes | None = None
    leaf_prefix: bytes | None = None
    key: bytes | None = None
    row_scheme: int = ROW_NONE
    row_bits: int = 0
    row_prefix: bytes | None = None
    row_prefix_b: bytes | None = None
    tile_rows: int = 0
    tile_cols: int = 0
    salt_source: int = SALT_GIVEN
    salt_key: bytes | None = None
    salt_nonce: int = 0
    seed_label: bytes | None = None

    def __post_init__(self) -> None:
        _length(self.domain_id, 64, "domain id")
        if self.schema is not None and len(self.schema) not in SCHEMA_LENGTHS:
            raise ValueError(f"a frame leaf schema is {SCHEMA_LENGTHS} bytes here, not {len(self.schema)}")
        for name, n in (("salt_prefix", 128), ("leaf_prefix", 128), ("key", 256), ("row_prefix", 128), ("row_prefix_b", 128),
                        ("seed_label", 128)):
            if getattr(self, name) is not None:
                _length(getattr(self, name), n, name)
        hm = (self.salt_prefix, self.leaf_prefix, self.key)
        if any(v is not None for v in hm) and not all(v is not None for v in hm):
            raise ValueError("hm96 leaves take the salt prefix, the leaf prefix and the key together")
        if self.salt_source == SALT_PRF:
            _length(self.salt_key, 64, "PRF salt key")
        elif self.salt_source in (SALT_CHACHA64, SALT_CHACHA_RFC):
            _length(self.salt_key, 32, "ChaCha20 salt key")
        elif self.salt_source != SALT_GIVEN:
            raise ValueError(f"unknown salt source {self.salt_source}")
        if not 0 <= self.salt_nonce < 1 << 64:
            raise ValueError("the salt nonce is a u64")
        if self.row_scheme in (ROW_V2, ROW_SEG) and not (0 < self.row_bits < 1 << 32 and self.row_prefix is not None):
            raise ValueError("row digests take the row's length in bits (below 2^32) and its prefix block")
        if self.row_scheme == ROW_TILE:
            if self.tile_rows % 2 or not self.tile_cols or self.tile_cols % TILE or self.row_prefix is None:
                raise ValueError("tiles take even rows, columns a multiple of 64 and the full tile's Q block")
            if self.tile_rows % TILE and self.row_prefix_b is None:
                raise ValueError("a last row block of fewer than 64 rows takes its own Q block")


def _length(v: bytes | None, n: int, name: str) -> None:
    if v is None or len(v) != n:
        raise ValueError(f"the {name} is {n} bytes")


# --- specs from the references ---------------------------------------------------------------------------------------


def _hm96() -> dict:
    from verity.ml.commitments import hm96
    key = hm96.SHA512.default_key
    return {"salt_prefix": hm96.SHA512.salt_prefix, "leaf_prefix": hm96.SHA512.leaf_prefix(key), "key": key}


def _salt(salt: tuple) -> dict:
    kind, *args = salt
    source = SALT_SOURCES[kind]
    if source == SALT_GIVEN:
        return {"salt_source": source}
    if source == SALT_PRF:
        return {"salt_source": source, "salt_key": bytes(args[0])}
    return {"salt_source": source, "salt_key": bytes(args[0]), "salt_nonce": int(args[1]) if len(args) > 1 else 0}


def row_v2_spec(domain_id: bytes, row_bits: int, **kw) -> dict:
    from verity.ml.commitments.rowleaf import sha512_row_v2_prefix
    return {"domain_id": domain_id, "row_scheme": ROW_V2, "row_bits": row_bits, "row_prefix": sha512_row_v2_prefix(row_bits), **kw}


def row_seg_spec(domain_id: bytes, row_bits: int, **kw) -> dict:
    from verity.ml.commitments.rowleaf import sha512_row_seg_prefix
    return {"domain_id": domain_id, "row_scheme": ROW_SEG, "row_bits": row_bits, "row_prefix": sha512_row_seg_prefix(row_bits), **kw}


def hm96_rows(domain_id: bytes, row_bits: int, salt: tuple = ("given",), schema: str = "hm96-sha512/row/v2") -> Spec:
    """hm96-sha512 leaves over sha512/row/v2 rows (the served activations' and outputs' rows)."""
    return Spec(**row_v2_spec(domain_id, row_bits, schema=schema.encode(), **_hm96(), **_salt(salt)))


def chunk_rows(domain_id: bytes, k: int, chunks: int, salt: tuple = ("given",)) -> tuple[Spec, int]:
    """An H1 table's chunk leaves at K = k (vc_chunk_rows_launch's, chunks = weights.chunks(k)): their hm96-sha512/row/v2
    spec under the value's domain, and the least out_stride their launch takes."""
    lb = k // chunks + 4
    return hm96_rows(domain_id, 8 * lb, salt), -(-lb // 16) * 16


def served_rows(domain_id: bytes, n_words: int, salt: tuple = ("given",)) -> Spec:
    """verity_vllm's serving rows: u16 words, x their sha512/row/v1 digest (role x), hm96-sha512/row/v1 leaves."""
    from verity.ml.commitments.rowleaf import ROLE_X, SCHEMA_HM96_SHA512_ROW, sha512_row_prefix
    return Spec(domain_id=domain_id, row_scheme=ROW_V2, row_bits=16 * n_words, row_prefix=sha512_row_prefix(ROLE_X, 16, n_words),
                schema=SCHEMA_HM96_SHA512_ROW.encode(), **_hm96(), **_salt(salt))


def hm96_digests(domain_id: bytes, salt: tuple = ("given",), schema: str = "hm96-sha512/row-seg/v1", row_bits: int = 0) -> Spec:
    """hm96-sha512 leaves over given 64-byte inner digests x; with row_bits, the plan also digests sha512/row-seg/v1 rows (A's tree)."""
    rows = row_seg_spec(domain_id, row_bits) if row_bits else {"domain_id": domain_id}
    return Spec(**rows, schema=schema.encode(), **_hm96(), **_salt(salt))


def frame_spec(domain_id: bytes, schema: str) -> Spec:
    """frame-v3-sha512 leaves over 64-byte values, pads, nodes and the empty root."""
    return Spec(domain_id=domain_id, schema=schema.encode())


def tiles_spec(domain_id: bytes, rows: int, cols: int) -> Spec:
    """Pearl-C -h3's tile leaves of a rows x cols C~ / U pair (sha512/row-seg/v1 over each tile's messages)."""
    from verity.ml.commitments.rowleaf import sha512_row_seg_prefix
    full = sha512_row_seg_prefix(8 * TILE * 512)
    last = rows % TILE
    return Spec(domain_id=domain_id, row_scheme=ROW_TILE, tile_rows=rows, tile_cols=cols, row_prefix=full,
                row_prefix_b=sha512_row_seg_prefix(8 * last * 512) if last else None)


def seeds_spec(domain_id: bytes, row_bits: int) -> Spec:
    """Pearl-C -h3's row digests (sha512/row-seg/v1 of each FP32 row) and row seeds (pearl_c.row_seed)."""
    from verity.core.protocols.pouw.schemes.pearl_c import S512_LABELS
    return Spec(**row_seg_spec(domain_id, row_bits), seed_label=S512_LABELS["seed-A"])


# --- the library ---------------------------------------------------------------------------------------------------

_P = ctypes.c_void_p
_U64 = ctypes.c_uint64
_I = ctypes.c_int


class _CSpec(ctypes.Structure):
    _fields_ = [("domain_id", _P), ("schema", _P), ("schema_len", _I), ("salt_prefix", _P), ("leaf_prefix", _P), ("key", _P),
                ("row_scheme", _I), ("row_bits", _U64), ("row_prefix", _P), ("row_prefix_b", _P), ("tile_rows", ctypes.c_uint32),
                ("tile_cols", ctypes.c_uint32), ("salt_source", _I), ("salt_key", _P), ("salt_nonce", _U64), ("seed_label", _P)]


# name: (restype, argtypes)
_SIGS = {
    "vc_plan_bytes": (_I, []),
    "vc_plan_build": (_I, [_P, _P]),
    "vc_tables_bytes": (_U64, [_P]),
    "vc_tables_launch": (_I, [_P, _P, _P]),
    "vc_empty_root": (_I, [_P, _P]),
    "vc_init": (_I, []),
    "vc_sha512_launch": (_I, [_P, _U64, _U64, _U64, _P, _P]),
    "vc_compress_launch": (_I, [_P, _P, _U64, _P, _P]),
    "vc_chunk_rows_launch": (_I, [_P, _U64, _P, _U64, ctypes.c_uint32, ctypes.c_uint32, _P, _U64, _P]),
    "vc_rows_scratch_bytes": (_U64, [_P, _U64]),
    "vc_rows_launch": (_I, [_P, _P, _P, _U64, _U64, _P, _P, _P]),
    "vc_tiles_count": (_U64, [_P]),
    "vc_tiles_scratch_bytes": (_U64, [_P]),
    "vc_tiles_launch": (_I, [_P, _P, _P, _P, _P, _P, _P]),
    "vc_row_seeds_launch": (_I, [_P, _P, _U64, _P, _P, ctypes.c_uint32, _P, _P]),
    "vc_tree_bytes": (_U64, [_U64]),
    "vc_tree_width": (_U64, [_U64]),
    "vc_salt_scratch_bytes": (_U64, [_U64]),
    "vc_salts_launch": (_I, [_P, _P, _P, _U64, _U64, _U64, _P, _P, _P]),
    "vc_leaves_x_launch": (_I, [_P, _P, _U64, _U64, _U64, _P, _P, _P, _P, _P]),
    "vc_leaves_rows_launch": (_I, [_P, _P, _U64, _U64, _U64, _U64, _P, _P, _P, _P, _P]),
    "vc_leaves_fused_launch": (_I, [_P, _P, _P, _U64, _P, _P, _U64, _U64, _U64, _P, _P, _P, _P]),
    "vc_frame_leaves_launch": (_I, [_P, _P, _U64, _U64, _U64, _P, _P]),
    "vc_nodes_launch": (_I, [_P, _P, _P, _U64, _I, _U64, _P]),
    "vc_tree_launch": (_I, [_P, _P, _U64, _P]),
    "vc_tree_arity_bytes": (_U64, [_U64, _I]),
    "vc_tree_arity_width": (_U64, [_U64, _I]),
    "vc_tree_arity_launch": (_I, [_P, _P, _U64, _I, _P]),
    "vc_commit_scratch_bytes": (_U64, [_U64]),
    "vc_commit_rows_launch": (_I, [_P, _P, _P, _U64, _P, _U64, _P, _P, _P, _I, _P]),
    "vc_open_bytes": (_U64, [_U64]),
    "vc_open_launch": (_I, [_P, _P, _P, _P, _P, _P, _U64, _P, _U64, _P, _P]),
}

OPEN_HEAD = 16 + 192 + 128 + 64


@dataclass(frozen=True)
class VcOpen:
    """One vc_open_launch record: ``offset`` is the leaf's byte offset in the levels (None outside the tree)."""

    position: int
    offset: int | None
    salt: bytes
    bc: bytes
    leaf: bytes
    path: tuple[bytes, ...]


def parse_openings(raw: bytes, n_leaves: int, m: int) -> list[VcOpen]:
    """vc_open_launch's m records, read back from the device (``raw``: m * vc_open_bytes(n_leaves) bytes)."""
    depth = (n_leaves - 1).bit_length() if n_leaves > 1 else 0
    size = OPEN_HEAD + 64 * depth
    if len(raw) != size * m:
        raise ValueError(f"{m} openings of a {n_leaves}-leaf tree are {size * m} bytes, not {len(raw)}")
    out = []
    for i in range(m):
        r = raw[size * i:size * (i + 1)]
        pos, off = int.from_bytes(r[:8], "little"), int.from_bytes(r[8:16], "little")
        out.append(VcOpen(pos, None if off == (1 << 64) - 1 else off, r[16:208], r[208:336], r[336:400],
                           tuple(r[OPEN_HEAD + 64 * k:OPEN_HEAD + 64 * (k + 1)] for k in range(depth))))
    return out


class LaunchError(RuntimeError):
    pass


class Lib:
    """libvc.so (``VC_LIB``, else build/libvc.so beside this file). A launch that returns nonzero raises `LaunchError`."""

    def __init__(self, path: str | os.PathLike | None = None) -> None:
        self.path = str(path or settings.VC_LIB or HERE / "build" / "libvc.so")
        self.so = ctypes.CDLL(self.path)
        for name, (res, args) in _SIGS.items():
            f = getattr(self.so, name)
            f.restype, f.argtypes = res, args
        rc = self.so.vc_init()
        if rc:
            raise LaunchError(f"vc_init: CUDA error {rc}")

    def __getattr__(self, name: str):
        f = getattr(self.so, "vc_" + name)
        if f.restype is not _I or not name.endswith("_launch"):
            return f

        def call(*args):
            rc = f(*args)
            if rc:
                raise LaunchError(f"vc_{name}: {'bad argument' if rc < 0 else f'CUDA error {rc}'}")
        return call

    def plan(self, spec: Spec) -> Plan:
        return Plan(self, spec)


class Plan:
    """A built plan (host memory) and the references it was built from; `ptr` is what a launch takes."""

    def __init__(self, lib: Lib, spec: Spec) -> None:
        self.spec = spec
        self._keep = []
        c = _CSpec()
        for name, _ in _CSpec._fields_:
            v = getattr(spec, name, None)
            if name == "schema_len":
                c.schema_len = len(spec.schema) if spec.schema is not None else 0
            elif isinstance(v, bytes):
                buf = ctypes.create_string_buffer(v, len(v))
                self._keep.append(buf)
                setattr(c, name, ctypes.addressof(buf))
            elif v is not None:
                setattr(c, name, v)
        self.buf = ctypes.create_string_buffer(lib.so.vc_plan_bytes())
        rc = lib.so.vc_plan_build(ctypes.byref(c), self.buf)
        if rc:
            raise ValueError(f"vc_plan_build refused the spec ({rc})")
        self.ptr = ctypes.addressof(self.buf)
        self.lib = lib

    @property
    def tables_bytes(self) -> int:
        return self.lib.so.vc_tables_bytes(self.ptr)

    def empty_root(self) -> bytes:
        out = ctypes.create_string_buffer(64)
        self.lib.so.vc_empty_root(self.ptr, out)
        return out.raw


# --- libcudart, for tests --------------------------------------------------------------------------------------------


class Cuda:
    """The few runtime calls the tests make (device memory, a stream, graph capture), on libcudart.so."""

    def __init__(self, path: str | None = None) -> None:
        cands = [path] if path else [settings.VC_CUDART, "/usr/local/cuda/lib64/libcudart.so", "libcudart.so", "libcudart.so.12",
                                     "libcudart.so.13"]
        err = None
        for p in [c for c in cands if c]:
            try:
                self.rt = ctypes.CDLL(p)
                break
            except OSError as e:
                err = e
        else:
            raise OSError(f"no libcudart: {err}")

    def _ok(self, rc: int, what: str) -> None:
        if rc:
            self.rt.cudaGetErrorString.restype = ctypes.c_char_p
            raise LaunchError(f"{what}: CUDA error {rc} ({(self.rt.cudaGetErrorString(rc) or b'').decode()})")

    def malloc(self, n: int) -> int:
        p = ctypes.c_void_p()
        self._ok(self.rt.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(max(n, 16))), "cudaMalloc")
        return p.value

    def free(self, p: int) -> None:
        self.rt.cudaFree(ctypes.c_void_p(p))

    # transfers go through the caller's (non-blocking) stream and wait for it: the legacy stream doesn't order against it
    def h2d(self, dst: int, data: bytes, s: int) -> None:
        self._ok(self.rt.cudaMemcpyAsync(ctypes.c_void_p(dst), data, ctypes.c_size_t(len(data)), 1, ctypes.c_void_p(s)), "cudaMemcpyAsync H2D")
        self.sync(s)

    def d2h(self, src: int, n: int, s: int) -> bytes:
        out = ctypes.create_string_buffer(n)
        self._ok(self.rt.cudaMemcpyAsync(out, ctypes.c_void_p(src), ctypes.c_size_t(n), 2, ctypes.c_void_p(s)), "cudaMemcpyAsync D2H")
        self.sync(s)
        return out.raw

    def memset(self, p: int, v: int, n: int, s: int) -> None:
        self._ok(self.rt.cudaMemsetAsync(ctypes.c_void_p(p), v, ctypes.c_size_t(n), ctypes.c_void_p(s)), "cudaMemsetAsync")

    def stream(self) -> int:
        s = ctypes.c_void_p()
        self._ok(self.rt.cudaStreamCreateWithFlags(ctypes.byref(s), 1), "cudaStreamCreate")
        return s.value

    def sync(self, s: int) -> None:
        self._ok(self.rt.cudaStreamSynchronize(ctypes.c_void_p(s)), "cudaStreamSynchronize")

    def capture(self, s: int, enqueue) -> int:
        """enqueue(s) under global stream capture; returns the instantiated graph."""
        self._ok(self.rt.cudaStreamBeginCapture(ctypes.c_void_p(s), 0), "cudaStreamBeginCapture")
        try:
            enqueue(s)
        finally:
            g = ctypes.c_void_p()
            rc = self.rt.cudaStreamEndCapture(ctypes.c_void_p(s), ctypes.byref(g))
        self._ok(rc, "cudaStreamEndCapture")
        ex = ctypes.c_void_p()
        self._ok(self.rt.cudaGraphInstantiate(ctypes.byref(ex), g, ctypes.c_ulonglong(0)), "cudaGraphInstantiate")
        return ex.value

    def replay(self, graph: int, s: int) -> None:
        self._ok(self.rt.cudaGraphLaunch(ctypes.c_void_p(graph), ctypes.c_void_p(s)), "cudaGraphLaunch")
