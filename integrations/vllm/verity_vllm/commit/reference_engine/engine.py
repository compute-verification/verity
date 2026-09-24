"""The committer: per-step subtree over packed 64-byte position-identified leaves, request root over step roots, openings and
the verifier (EXPERIMENTAL, CMT-1).

Reused, not forked (imported from `verity_vllm`): `hashing.node_hash / lift_hash` (production node / lift frames),
`commit.path_shape / fold_path` (the production opening verifier's core), `commit_bench.semantic.pos_leaf`
(C0 leaf: SHA-256("verity/pos-leaf/v0" || u64 nbytes || value)), `commit_bench.fasttree.fold_levels`.

Tree shape (two levels of production trees):
    step tree     : leaves = pos_leaf(64-byte chunk k of the step stream), production pairing / lift rule -> step_root[s]
    request tree  : level-0 nodes = step_root[0..S), same rule -> merkle_root
    R = SHA-256(ROOT_TAG || program_digest || ctx_digest || positions_digest || u64 S || u64 N_leaves || merkle_root)

Backends: "host" = hashlib per leaf (the slow reference, Python loop), "torch" = `torch_sha256` on the stream's device
(GPU-resident when the stream is a CUDA tensor; identical digests -- asserted in tests).
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_POC = Path(__file__).resolve().parents[3] / "vllm-poc"
if _POC.is_dir() and str(_POC) not in sys.path:
    sys.path.insert(0, str(_POC))

from verity_vllm.commit import hashing  # noqa: E402
from verity_vllm.commit.merkle import fold_path, path_shape  # noqa: E402
from verity_vllm.commit import fasttree as ft  # noqa: E402
from verity_vllm.commit.semantic_layout import POS_LEAF_TAG, pos_leaf  # noqa: E402

from .positions import LEAF_BYTES, StepLayout, positions_digest, positions_document  # noqa: E402

ROOT_TAG = b"verity/cmt-ref/root/v0"
LEAF_PREIMAGE_BYTES = len(POS_LEAF_TAG) + 8 + LEAF_BYTES     # 90 B -> 2 SHA blocks
NODE_PREIMAGE_BYTES = 103                                    # 2 SHA blocks
LIFT_PREIMAGE_BYTES = 63                                     # 2 SHA blocks (63 + 9 > 64)


def sha_blocks(n: int) -> int:
    return (n + 8) // 64 + 1


@dataclass(frozen=True)
class Context:
    profile_id: str
    request_id: str
    prompt_len: int
    model: str = ""
    extra: dict = field(default_factory=dict)

    def digest(self) -> bytes:
        doc = {"profile_id": self.profile_id, "request_id": self.request_id, "prompt_len": self.prompt_len, "model": self.model, **self.extra}
        return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).digest()

    def to_json(self) -> dict:
        return {"profile_id": self.profile_id, "request_id": self.request_id, "prompt_len": self.prompt_len, "model": self.model, **self.extra}


def bind_root(program_digest: bytes, ctx_digest: bytes, positions_dig: bytes, n_steps: int, n_leaves: int, merkle_root: bytes) -> bytes:
    for d in (program_digest, ctx_digest, positions_dig, merkle_root):
        if len(d) != 32:
            raise ValueError("32-byte digests expected")
    return hashlib.sha256(ROOT_TAG + program_digest + ctx_digest + positions_dig + struct.pack(">QQ", n_steps, n_leaves) + merkle_root).digest()


# --- leaf hashing backends ----------------------------------------------------------------------------------------------------


def leaf_digests_host(stream: bytes | memoryview) -> list[bytes]:
    """The reference: one hashlib call per 64-byte leaf (zero-padded tail)."""
    mv = memoryview(stream)
    n = (len(mv) + LEAF_BYTES - 1) // LEAF_BYTES
    out = []
    for k in range(n):
        chunk = mv[k * LEAF_BYTES: (k + 1) * LEAF_BYTES]
        if len(chunk) < LEAF_BYTES:
            chunk = bytes(chunk) + bytes(LEAF_BYTES - len(chunk))
        out.append(pos_leaf(chunk))
    return out


def _as_padded_leaf_matrix(stream_t, n_leaves: int):
    import torch
    n = n_leaves * LEAF_BYTES
    if stream_t.numel() == n:
        buf = stream_t
    else:
        buf = torch.zeros(n, dtype=torch.uint8, device=stream_t.device)
        buf[: stream_t.numel()] = stream_t
    return buf.view(n_leaves, LEAF_BYTES)


def leaf_digests_torch(stream_t, n_leaves: int, *, compiled: bool = False, chunk: int = 1 << 20):
    """torch backend: uint8 [n] tensor (any device) -> uint8 [n_leaves, 32] on the same device."""
    import torch

    from . import torch_sha256 as TS
    mat = _as_padded_leaf_matrix(stream_t, n_leaves)
    if n_leaves <= chunk:
        return TS.pos_leaf_digests(mat, compiled=compiled)
    outs = [TS.pos_leaf_digests(mat[i: i + chunk], compiled=compiled) for i in range(0, n_leaves, chunk)]
    return torch.cat(outs, dim=0)


# --- the committer ---------------------------------------------------------------------------------------------------------------


@dataclass
class StepCommit:
    layout: StepLayout
    root: bytes | None
    leaf_count: int
    levels_host: list[list[bytes]] | None = None    # host backend: every level (bytes); torch-hybrid: the upper (small) levels
    levels_torch: Any = None                          # torch backend: list of uint8 [n, 32] tensors (on device) -- the lower levels
    root_t: Any = None                                # torch backend, deferred root: uint8 [32] on device until finish()
    timing: dict = field(default_factory=dict)


class Committer:
    """One request: `commit_step` per engine step (streaming), `finish` -> (R, positions document).

    torch backend knobs: `host_levels_below` = when a tree level has at most that many nodes it is copied to the host (one small
    D2H, a sync) and the remaining levels are folded with hashlib (the upper levels are launch-bound on the GPU: a few nodes per
    kernel).  0 = fold everything on the device and defer the root read to `finish()` (no per-step sync at all)."""

    def __init__(self, program_digest: bytes, ctx: Context, *, backend: str = "host", compiled: bool = False, keep_levels: bool = True,
                 host_levels_below: int = 0) -> None:
        if len(program_digest) != 32:
            raise ValueError("program digest: 32 bytes")
        if backend not in ("host", "torch"):
            raise ValueError(backend)
        self.program_digest, self.ctx, self.backend, self.compiled, self.keep_levels = program_digest, ctx, backend, compiled, keep_levels
        self.host_levels_below = int(host_levels_below)
        self.steps: list[StepCommit] = []
        self.root: bytes | None = None
        self.merkle_root: bytes | None = None
        self.top_levels: list[list[bytes]] | None = None
        self.positions: dict | None = None
        self.hash_calls = 0
        self.hash_blocks = 0

    # -- per step --------------------------------------------------------------------------------------------------------
    def commit_step(self, layout: StepLayout, stream) -> StepCommit:
        """`stream`: bytes / bytearray / memoryview / numpy uint8 / torch uint8 tensor of `layout.stream_bytes` bytes."""
        if self.root is not None:
            raise RuntimeError("finished")
        if len(self.steps) != layout.step:
            raise ValueError(f"steps must arrive in order: got {layout.step}, expected {len(self.steps)}")
        layout.freeze()
        n = layout.leaf_count
        t0 = time.perf_counter()
        if self.backend == "host":
            data = stream if isinstance(stream, (bytes, bytearray, memoryview)) else _to_host_bytes(stream)
            if len(data) != layout.stream_bytes:
                raise ValueError(f"stream has {len(data)} bytes, layout says {layout.stream_bytes}")
            leaves = leaf_digests_host(data)
            t1 = time.perf_counter()
            levels = ft.fold_levels(list(leaves))
            root = levels[-1][0]
            t2 = time.perf_counter()
            sc = StepCommit(layout, root, n, levels_host=levels if self.keep_levels else [levels[0][:0]] + [[root]],
                            timing={"leaf_s": t1 - t0, "tree_s": t2 - t1, "wall_s": t2 - t0})
            self.hash_calls += n + sum(len(lv) for lv in levels[1:])
            self.hash_blocks += n * sha_blocks(LEAF_PREIMAGE_BYTES) + sum(len(lv) for lv in levels[1:]) * 2
        else:
            import torch

            from . import torch_sha256 as TS
            if not isinstance(stream, torch.Tensor):
                stream = torch.frombuffer(bytearray(_to_host_bytes(stream)), dtype=torch.uint8)
            if stream.dtype != torch.uint8 or stream.dim() != 1:
                raise ValueError("torch backend expects a flat uint8 tensor")
            if stream.numel() != layout.stream_bytes and stream.numel() != layout.padded_bytes:
                raise ValueError(f"stream has {stream.numel()} bytes, layout says {layout.stream_bytes}")
            leaves = leaf_digests_torch(stream, n, compiled=self.compiled)
            levels = TS.fold_levels(leaves, compiled=self.compiled, stop_at=self.host_levels_below)
            t1 = time.perf_counter()
            host_levels = None
            if self.host_levels_below > 0 and int(levels[-1].shape[0]) > 1:
                small = levels[-1].cpu().numpy()                         # one small D2H (<= host_levels_below x 32 B), a sync
                host_levels = ft.fold_levels([bytes(r.tobytes()) for r in small])
                root, root_t = host_levels[-1][0], None
            elif self.host_levels_below > 0:
                root, root_t = bytes(levels[-1][0].cpu().numpy().tobytes()), None
            else:
                root, root_t = None, levels[-1][0]                       # deferred: read in finish()
            t2 = time.perf_counter()
            keep_t = levels if self.keep_levels else [levels[0][:0], levels[-1]]
            sc = StepCommit(layout, root, n, levels_host=host_levels, levels_torch=keep_t, root_t=root_t,
                            timing={"device_enqueue_s": t1 - t0, "host_tail_s": t2 - t1, "wall_s": t2 - t0,
                                    "note": "torch backend: enqueue is asynchronous; host_tail includes the sync when host_levels_below > 0"})
            n_dev_nodes = sum(int(lv.shape[0]) for lv in levels[1:])
            n_host_nodes = sum(len(lv) for lv in host_levels[1:]) if host_levels else 0
            self.hash_calls += n + n_dev_nodes + n_host_nodes
            self.hash_blocks += n * 2 + (n_dev_nodes + n_host_nodes) * 2
        self.steps.append(sc)
        return sc

    # -- finish ----------------------------------------------------------------------------------------------------------
    def finish(self) -> tuple[bytes, dict]:
        if self.root is not None:
            return self.root, self.positions
        if not self.steps:
            raise RuntimeError("no steps")
        deferred = [s for s in self.steps if s.root is None]
        if deferred:                                   # one D2H for every deferred step root
            import torch
            roots = torch.stack([s.root_t for s in deferred]).cpu().numpy()
            for s, r in zip(deferred, roots, strict=True):
                s.root = bytes(r.tobytes())
        step_roots = [s.root for s in self.steps]
        self.top_levels = ft.fold_levels(list(step_roots))
        self.merkle_root = self.top_levels[-1][0]
        self.hash_calls += sum(len(lv) for lv in self.top_levels[1:])
        self.hash_blocks += sum(len(lv) for lv in self.top_levels[1:]) * 2
        layouts = [s.layout for s in self.steps]
        n_leaves = sum(s.leaf_count for s in self.steps)
        self.positions = positions_document(layouts, {**self.ctx.to_json(), "program_digest": self.program_digest.hex(),
                                                       "n_steps": len(self.steps), "n_leaves": n_leaves,
                                                       "step_roots": [r.hex() for r in step_roots], "merkle_root": self.merkle_root.hex()})
        pdig = positions_digest(self.positions)
        self.root = bind_root(self.program_digest, self.ctx.digest(), pdig, len(self.steps), n_leaves, self.merkle_root)
        self.positions["context"]["root"] = self.root.hex()
        self.positions["context"]["root_tag"] = ROOT_TAG.decode()
        return self.root, self.positions

    # -- accounting ------------------------------------------------------------------------------------------------------
    def persistent_bytes(self) -> dict:
        """Retained state after finish: roots vs (optionally) the levels kept for opening generation."""
        levels = 0
        for s in self.steps:
            if s.levels_host:
                levels += sum(32 * len(lv) for lv in s.levels_host)
            if s.levels_torch is not None:
                levels += sum(int(lv.numel()) for lv in s.levels_torch)
        return {"roots_bytes": 32 * (len(self.steps) + 1), "levels_bytes": levels,
                "minimum_retained_bytes": 32 * (len(self.steps) + 1) + 32 * sum(s.leaf_count for s in self.steps),
                "note": "minimum = root + step roots + leaf digests (paths regenerable from leaf digests); levels = full tree kept for O(1) openings"}


def _to_host_bytes(x) -> bytes:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().contiguous().cpu().numpy().tobytes()
    except ImportError:
        pass
    import numpy as np
    if isinstance(x, np.ndarray):
        return x.tobytes()
    return bytes(x)


# --- openings ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Opening:
    """Everything a verifier needs besides the public (root, positions): the identity requested, the leaf bytes, both paths."""
    step: int
    class_id: str
    instance: str
    element: int
    value: bytes                       # the claimed value bytes (item width)
    leaf_index: int
    leaf: bytes                        # the 64-byte leaf
    path: tuple[bytes | None, ...]     # step tree path (production shape rule)
    step_root: bytes
    top_path: tuple[bytes | None, ...]

    def to_json(self) -> dict:
        return {"step": self.step, "class_id": self.class_id, "instance": self.instance, "element": self.element, "value": self.value.hex(),
                "leaf_index": self.leaf_index, "leaf": self.leaf.hex(), "path": [None if p is None else p.hex() for p in self.path],
                "step_root": self.step_root.hex(), "top_path": [None if p is None else p.hex() for p in self.top_path]}

    @classmethod
    def from_json(cls, d: dict) -> Opening:
        return cls(d["step"], d["class_id"], d["instance"], d["element"], bytes.fromhex(d["value"]), d["leaf_index"], bytes.fromhex(d["leaf"]),
                   tuple(None if p is None else bytes.fromhex(p) for p in d["path"]), bytes.fromhex(d["step_root"]),
                   tuple(None if p is None else bytes.fromhex(p) for p in d["top_path"]))

    def replace(self, **kw) -> Opening:
        import dataclasses
        return dataclasses.replace(self, **kw)


def _path_from_levels_host(levels: list[list[bytes]], index: int) -> tuple[bytes | None, ...]:
    out: list[bytes | None] = []
    cur = index
    for level in levels[:-1]:
        out.append(None if (cur == len(level) - 1 and len(level) % 2 == 1) else level[cur ^ 1])
        cur //= 2
    return tuple(out)


def _path_from_levels_torch(levels, index: int) -> tuple[bytes | None, ...]:
    out: list[bytes | None] = []
    cur = index
    for level in levels[:-1]:
        m = int(level.shape[0])
        out.append(None if (cur == m - 1 and m % 2 == 1) else bytes(level[cur ^ 1].cpu().numpy().tobytes()))
        cur //= 2
    return tuple(out)


def open_value(c: Committer, step: int, class_id: str, instance: str, element: int, stream_leaf_bytes: bytes | None = None,
               stream=None) -> Opening:
    """Opening for one value.  The leaf bytes come from `stream_leaf_bytes` (64 B) or are cut from `stream` (the step's
    committed stream: bytes / torch uint8); the paths come from the retained levels."""
    if c.root is None:
        raise RuntimeError("finish() first")
    sc = c.steps[step]
    k, off, ib = sc.layout.rank(class_id, instance, element)
    if stream_leaf_bytes is None:
        if stream is None:
            raise ValueError("need the step stream or the leaf bytes")
        lo = k * LEAF_BYTES
        if hasattr(stream, "numel"):
            chunk = stream[lo: lo + LEAF_BYTES]
            leaf = bytes(chunk.cpu().numpy().tobytes())
        else:
            leaf = bytes(memoryview(stream)[lo: lo + LEAF_BYTES])
        leaf = leaf + bytes(LEAF_BYTES - len(leaf))
    else:
        leaf = bytes(stream_leaf_bytes)
    if sc.levels_torch is not None and int(sc.levels_torch[0].shape[0]) == sc.leaf_count:
        path = _path_from_levels_torch(sc.levels_torch, k)                       # device levels (all, or the lower ones)
        if sc.levels_host is not None:                                             # hybrid: the upper levels live on the host
            top_k = k >> (len(sc.levels_torch) - 1)
            path = path + _path_from_levels_host(sc.levels_host, top_k)
    elif sc.levels_host is not None and len(sc.levels_host[0]) == sc.leaf_count:
        path = _path_from_levels_host(sc.levels_host, k)
    else:
        raise RuntimeError("levels not retained; regenerate the step tree to open")
    top = _path_from_levels_host(c.top_levels, step)
    return Opening(step, class_id, instance, element, leaf[off: off + ib], k, leaf, path, sc.root, top)


def verify_opening(root: bytes, positions: dict, op: Opening, program_digest: bytes | None = None) -> tuple[bool, str]:
    """Production-core verification against the ORIGINAL bound root.  The verifier derives the position from `positions`
    (public static context) -- a prover-supplied leaf index that disagrees is refused -- checks both path shapes with
    `path_shape`, folds with `fold_path`, rebinds with `bind_root`, and finally extracts the value at the derived offset."""
    try:
        ctx = positions["context"]
        layouts = [StepLayout.from_json(d) for d in positions["steps"]]
        S = len(layouts)
        if not 0 <= op.step < S:
            return False, "step out of range"
        L = layouts[op.step]
        k, off, ib = L.rank(op.class_id, op.instance, op.element)
        if k != op.leaf_index:
            return False, "leaf index disagrees with the layout"
        if len(op.leaf) != LEAF_BYTES:
            return False, "leaf size"
        if len(op.value) != ib or op.leaf[off: off + ib] != op.value:
            return False, "value bytes do not match the leaf at the derived offset"
        n = L.leaf_count
        shape = path_shape(n, k)
        if len(op.path) != len(shape) or any((p is None) == has for p, has in zip(op.path, shape, strict=True)):
            return False, "step path shape"
        if fold_path(pos_leaf(op.leaf), k, op.path) != op.step_root:
            return False, "step root mismatch"
        tshape = path_shape(S, op.step)
        if len(op.top_path) != len(tshape) or any((p is None) == has for p, has in zip(op.top_path, tshape, strict=True)):
            return False, "top path shape"
        merkle = fold_path(op.step_root, op.step, op.top_path)
        n_leaves = sum(l.leaf_count for l in layouts)
        pdig = positions_digest(positions)
        pd = program_digest if program_digest is not None else bytes.fromhex(ctx["program_digest"])
        c = Context(ctx["profile_id"], ctx["request_id"], int(ctx["prompt_len"]), ctx.get("model", ""),
                    {k2: v for k2, v in ctx.items() if k2 not in ("profile_id", "request_id", "prompt_len", "model", "program_digest", "n_steps", "n_leaves",
                                                                      "step_roots", "merkle_root", "root", "root_tag")})
        R = bind_root(pd, c.digest(), pdig, S, n_leaves, merkle)
        if R != root:
            return False, "root mismatch"
        return True, "ok"
    except Exception as e:  # noqa: BLE001 -- a verifier never raises on a malformed opening
        return False, f"malformed: {type(e).__name__}: {e}"


class Verifier:
    """Same checks as `verify_opening`, with the positions document parsed and digested ONCE (verify_opening re-parses per call:
    ~26 ms for this request's 917 KB document vs ~0.2 ms per opening here).  Fail-closed: any exception -> (False, reason)."""

    def __init__(self, root: bytes, positions: dict, program_digest: bytes | None = None) -> None:
        self.root = root
        ctx = positions["context"]
        self.layouts = [StepLayout.from_json(d) for d in positions["steps"]]
        self.S = len(self.layouts)
        self.n_leaves = sum(l.leaf_count for l in self.layouts)
        self.pdig = positions_digest(positions)
        self.pd = program_digest if program_digest is not None else bytes.fromhex(ctx["program_digest"])
        self.ctx = Context(ctx["profile_id"], ctx["request_id"], int(ctx["prompt_len"]), ctx.get("model", ""),
                           {k2: v for k2, v in ctx.items() if k2 not in ("profile_id", "request_id", "prompt_len", "model", "program_digest", "n_steps",
                                                                              "n_leaves", "step_roots", "merkle_root", "root", "root_tag")})

    def verify(self, op: Opening) -> tuple[bool, str]:
        try:
            if not 0 <= op.step < self.S:
                return False, "step out of range"
            L = self.layouts[op.step]
            k, off, ib = L.rank(op.class_id, op.instance, op.element)
            if k != op.leaf_index:
                return False, "leaf index disagrees with the layout"
            if len(op.leaf) != LEAF_BYTES:
                return False, "leaf size"
            if len(op.value) != ib or op.leaf[off: off + ib] != op.value:
                return False, "value bytes do not match the leaf at the derived offset"
            shape = path_shape(L.leaf_count, k)
            if len(op.path) != len(shape) or any((p is None) == has for p, has in zip(op.path, shape, strict=True)):
                return False, "step path shape"
            if fold_path(pos_leaf(op.leaf), k, op.path) != op.step_root:
                return False, "step root mismatch"
            tshape = path_shape(self.S, op.step)
            if len(op.top_path) != len(tshape) or any((p is None) == has for p, has in zip(op.top_path, tshape, strict=True)):
                return False, "top path shape"
            merkle = fold_path(op.step_root, op.step, op.top_path)
            if bind_root(self.pd, self.ctx.digest(), self.pdig, self.S, self.n_leaves, merkle) != self.root:
                return False, "root mismatch"
            return True, "ok"
        except Exception as e:  # noqa: BLE001
            return False, f"malformed: {type(e).__name__}: {e}"


def production_leaf_check(op: Opening) -> bool:
    """The production `hashing` module reproduces our leaf / node / lift digests (same frames): cross-check on one opening."""
    leaf = hashing.leaf_hash if False else pos_leaf  # pos_leaf is the C0 frame; node/lift are production's
    digest, cur = leaf(op.leaf), op.leaf_index
    for sib in op.path:
        digest = hashing.lift_hash(digest) if sib is None else (hashing.node_hash(digest, sib) if cur % 2 == 0 else hashing.node_hash(sib, digest))
        cur //= 2
    return digest == op.step_root
