"""Adapters that present every commitment construction under test through ONE interface, so the security tests in
`test_security.py` run unchanged against each of them (and against a sibling lane's engine once an adapter is added here).

    scheme.commit(values: list[bytes], ctx: bytes) -> Committed          # values in tree order; ctx = context the root is bound to
    scheme.open(c: Committed, i: int) -> Opening
    scheme.verify(root: bytes, n: int, ctx: bytes, opening) -> bool     # verifier side: only (root, n, ctx) + the opening

Adapters:
  (the Δ_commit committer `native_host` exposes `verify(RunCommitment, Opening)` — the verifier's held commitment — and is
   tested directly in `test_native_host_security.py` through its offline core `commit_offline`)
  * `C0PrimitiveScheme`  — the bare production primitive both native_host and fa2_commit build on: `pos_leaf` + `fasttree.fold_levels`
                           + `path_in_levels` + `commit.fold_path`/`path_shape` (verity_vllm, the production opening verifier core)
  * `ProtocolMerkleScheme` — `veritor.protocol.merkle` (CommitmentDomain / MerkleTree / verify_opening): the SP1 statement framing
                           (`zk/sp1/common` reproduces these roots; multiproof witness format 2 folds the same nodes)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from verity_capture.experimental.fa2_commit import engine as eng


@dataclass
class Committed:
    root: bytes
    n: int
    ctx: bytes
    state: Any


class C0PrimitiveScheme:
    """pos_leaf leaves (one per value), fasttree fold, path_in_levels; verifier = commit.path_shape + fold_path + bind_root(ctx)."""

    name = "c0_primitive"

    def _bind(self, ctx: bytes, n: int, merkle: bytes) -> bytes:
        z = hashlib.sha256(ctx).digest()
        return eng.bind_root(z, z, z, z, n, merkle)

    def commit(self, values: list[bytes], ctx: bytes) -> Committed:
        leaves = [eng.pos_leaf(v) for v in values]
        levels = eng.ft.fold_levels(list(leaves))
        return Committed(self._bind(ctx, len(values), levels[-1][0]), len(values), ctx, (levels, list(values)))

    def open(self, c: Committed, i: int):
        levels, values = c.state
        return (i, values[i], eng.path_in_levels(levels, i))

    def verify(self, root: bytes, n: int, ctx: bytes, opening) -> bool:
        i, value, path = opening
        if not 0 <= i < n:
            return False
        try:
            shape = eng.path_shape(n, i)
        except Exception:  # noqa: BLE001
            return False
        if len(path) != len(shape) or any((p is None) == has for p, has in zip(path, shape, strict=True)):
            return False
        return self._bind(ctx, n, eng.fold_path(eng.pos_leaf(value), i, list(path))) == root

    @staticmethod
    def with_value(opening, value: bytes):
        return (opening[0], value, opening[2])

    @staticmethod
    def with_index(opening, index: int, identity=None):
        return (index, opening[1], opening[2])

    @staticmethod
    def with_path(opening, path):
        return (opening[0], opening[1], tuple(path))


class ProtocolMerkleScheme:
    name = "veritor_protocol_merkle"
    SCHEMA = "bytes"

    def _domain(self, ctx: bytes, n: int):
        from veritor.core import RangeIndexedDomain
        from veritor.protocol.merkle import CommitmentDomain

        return CommitmentDomain(hashlib.sha256(ctx).digest(), -1, RangeIndexedDomain(0, n))

    def commit(self, values: list[bytes], ctx: bytes) -> Committed:
        from veritor.protocol.merkle import MerkleTree

        dom = self._domain(ctx, len(values))
        tree = MerkleTree(dom, {i: v for i, v in enumerate(values)}, lambda _p: self.SCHEMA)
        return Committed(tree.commitment.root, len(values), ctx, tree)

    def open(self, c: Committed, i: int):
        return c.state.open(i)

    def verify(self, root: bytes, n: int, ctx: bytes, opening) -> bool:
        from veritor.core import VerificationLimits
        from veritor.protocol.merkle import verify_opening
        from veritor.protocol.messages import Commitment

        try:
            return verify_opening(self._domain(ctx, n), Commitment(root, n), opening, self.SCHEMA, VerificationLimits())
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def with_value(opening, value: bytes):
        from veritor.protocol.messages import Opening

        return Opening(opening.position, value, opening.path)

    @staticmethod
    def with_index(opening, index: int, identity=None):
        from veritor.protocol.messages import Opening

        return Opening(index, opening.value, opening.path)

    @staticmethod
    def with_path(opening, path):
        from veritor.protocol.messages import Opening

        return Opening(opening.position, opening.value, tuple(path))


class CmtRefScheme:
    """CMT-1's engine (`verity_capture.experimental.commit`, merged from lane/r9-cmt-ref): packed 64-B pos-leaf step tree + request
    root bound to (program digest, ctx, positions digest); public state = (root, positions document).  One step; value i = segment
    `t<i>` of dtype i64 (8 bytes = one element, so an opening carries the whole value); verifier state kept per root."""

    name = "cmt_ref_engine"

    def __init__(self) -> None:
        from verity_capture.experimental import commit as C

        self.C = C
        self._public: dict[bytes, tuple[dict, bytes]] = {}       # root -> (positions, ctx)

    def _ctx(self, ctx: bytes):
        return self.C.Context("profile-test", ctx.decode(errors="replace"), 7, "model-test")

    def commit(self, values: list[bytes], ctx: bytes) -> Committed:
        C = self.C
        c = C.Committer(hashlib.sha256(b"program").digest(), self._ctx(ctx), backend="host")
        L = C.StepLayout(0, len(values))
        segs = [L.add(f"t{i}", "x", "i64", (len(v) // 8,)) for i, v in enumerate(values)]
        buf = bytearray(L.stream_bytes)
        for s, v in zip(segs, values, strict=True):
            buf[s.offset:s.offset + s.nbytes] = v
        c.commit_step(L, bytes(buf))
        root, positions = c.finish()
        self._public[root] = (positions, ctx)
        return Committed(root, len(values), ctx, (c, bytes(buf)))

    def open(self, c: Committed, i: int):
        com, buf = c.state
        return self.C.open_value(com, 0, f"t{i}", "x", 0, stream=buf)

    def verify(self, root: bytes, n: int, ctx: bytes, opening) -> bool:
        pub = self._public.get(root)
        if pub is None:
            return False                                   # the verifier never received this root
        positions, ctx0 = pub
        if n != len(positions["steps"][0]["segments"]):
            return False                                   # a verifier expecting another N
        if ctx != ctx0:                                    # a verifier holding another context: its rebind differs
            import copy

            positions = copy.deepcopy(positions)
            positions["context"]["request_id"] = ctx.decode(errors="replace")
        ok, _why = self.C.verify_opening(root, positions, opening)
        return bool(ok)

    @staticmethod
    def with_value(opening, value: bytes):
        # the adversary rewrites the value AND the leaf slice it sits in (else the verifier's leaf/value consistency check fires first)
        leaf = bytearray(opening.leaf)
        off = None
        for o in range(0, len(leaf) - len(opening.value) + 1):
            if leaf[o:o + len(opening.value)] == opening.value:
                off = o
                break
        if off is not None and len(value) == len(opening.value):
            leaf[off:off + len(value)] = value
        return opening.replace(value=value, leaf=bytes(leaf))

    @staticmethod
    def with_index(opening, index: int, identity=None):
        return opening.replace(class_id=f"t{index}")

    @staticmethod
    def with_path(opening, path):
        return opening.replace(path=tuple(path))


def _schemes() -> list:
    out = [C0PrimitiveScheme()]
    try:
        import veritor.protocol.merkle  # noqa: F401  (the `veritor` package needs Python >= 3.12: PEP 695 `type` statements)

        out.append(ProtocolMerkleScheme())
    except Exception as e:  # noqa: BLE001
        print(f"[schemes] veritor.protocol.merkle not importable here ({type(e).__name__}); scheme skipped")
    try:
        out.append(CmtRefScheme())
    except Exception as e:  # noqa: BLE001 -- the sibling's engine is optional until merged
        print(f"[schemes] cmt_ref engine not available: {e!r}")
    return out


SCHEMES = _schemes()
