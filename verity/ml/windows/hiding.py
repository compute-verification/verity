"""The host committer's ``hm96-sha256/v1`` hiding leaves (``verity.ml.commitments.hm96``, spec ``PROTOCOL.md`` beside it), batched
over one step's unsalted leaf digests, and the committer's choice of leaf rule (:class:`LeafSchemeRule`, opt-in, off by default).
``tests/commit/test_hiding.py`` pins it to core's reference and vectors.

Each leaf gets 128 fresh salt bytes from the OS generator (one call per step), retained with the step: a leaf opens with its
value and its salt.  ``M_key * y`` gathers core's byte tables (per salt byte position, the XOR of the columns its set bits
select) with numpy over the whole step.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence

import numpy as np

from verity.ml.commitments import hm96, vllm_v1

NAME = hm96.NAME
SALT_BYTES = hm96.SALT_BYTES
LEAF_SCHEMES = (vllm_v1.NAME, NAME)
_BATCH = 4096


class HidingLeaves:
    """The batch path of one hm96 instance (``hm96.SHA256`` by default, the committer's opt-in; ``hm96.SHA512`` for the
    benchmark until the re-baseline wires SHA-512 trees) under ``key`` (the instance's pinned key by default)."""

    def __init__(self, key: bytes | None = None, instance: hm96.Instance = hm96.SHA256) -> None:
        self.instance = inst = instance
        self.key = inst.check_key(inst.default_key if key is None else key)
        self.digest = inst.scheme_digest(self.key)
        self._d, self._y = inst.digest_bytes, inst.salt_bytes
        flat = b"".join(t.to_bytes(self._d, "little") for table in inst.byte_tables(self.key) for t in table)
        self._tables = np.frombuffer(flat, dtype=np.uint64).reshape(self._y, 256, self._d // 8)
        self._salt_h = hashlib.new(inst.hash, inst.salt_prefix)
        self._leaf_h = hashlib.new(inst.hash, inst.leaf_prefix(self.key))

    def masks(self, salts: bytes, n: int) -> np.ndarray:
        """``M_key * y`` of ``n`` concatenated salts, as ``(n, digest bytes)``."""
        y = np.frombuffer(salts, dtype=np.uint8).reshape(n, self._y)
        out = np.zeros((n, self._d // 8), dtype=np.uint64)
        for s in range(0, n, _BATCH):
            yt = np.ascontiguousarray(y[s:s + _BATCH].T)
            acc = out[s:s + _BATCH]
            for p in range(self._y):
                acc ^= self._tables[p][yt[p]]
        return out.view(np.uint8).reshape(n, self._d)

    def leaves_with(self, inner: Sequence[bytes], salts: bytes) -> list[bytes]:
        """The tree leaves of the unsalted digests ``inner`` under the given concatenated salts."""
        n, d, y = len(inner), self._d, self._y
        if len(salts) != n * y:
            raise ValueError(f"{n} leaves take {n * y} salt bytes, not {len(salts)}")
        if not n:
            return []
        b = (np.frombuffer(b"".join(inner), dtype=np.uint8).reshape(n, d) ^ self.masks(salts, n)).tobytes()
        ys, bs = memoryview(salts), memoryview(b)
        out = []
        for i in range(n):
            h = self._salt_h.copy()
            h.update(ys[i * y:(i + 1) * y])
            g = self._leaf_h.copy()
            g.update(bs[i * d:(i + 1) * d])
            g.update(h.digest())
            out.append(g.digest())
        return out

    def leaves(self, inner: Sequence[bytes]) -> tuple[list[bytes], bytes]:
        """Fresh salts for every leaf of ``inner``: the tree leaves and the concatenated salts (the stored bytes)."""
        salts = os.urandom(len(inner) * self._y)
        return self.leaves_with(inner, salts), salts

    def leaf(self, inner_digest: bytes, salt: bytes) -> bytes:
        """One leaf through core's reference (the verifier's path)."""
        return self.instance.tree_leaf(self.key, self.instance.commit_string(self.key, inner_digest, salt))


class LeafSchemeRule:
    """The native host committer's leaf rule (a mixin of ``NativeHostCommitter``): ``vllm-v1`` position leaves (the default, the
    recorded roots), or ``hm96-sha256/v1`` leaves over them.  Under hm96 every step's salts are retained (the stored bytes), openings
    carry them, and the step roots bind the scheme's digest through the context digest.

    hm96 fails closed on the path the leaves take: only ``NativeHostCommitter``'s own host path salts (``_salted``), a committer
    whose step path is another (the GPU chunk tree, a subclass's native hashing) is refused at construction, the context digest
    binds the scheme only for a step whose leaves were salted (every step root is bound through it: ``commit_block_offline`` and the
    native collector's roots included), and a step without one salt per leaf neither opens nor finalizes.  Padding steps keep
    position leaves and are refused."""

    name: str
    hiding: HidingLeaves | None = None

    def _init_leaf_scheme(self, leaf_scheme: str, other_path: bool) -> None:
        """`other_path`: the committer's steps do not take the host path (the GPU chunk tree, or an overridden step path)."""
        if leaf_scheme not in LEAF_SCHEMES:
            raise ValueError(f"leaf_scheme must be one of {LEAF_SCHEMES}, not {leaf_scheme!r}")
        if leaf_scheme == NAME and other_path:
            raise ValueError(f"{NAME} leaves are salted on NativeHostCommitter's host path only; {type(self).__name__} with these options "
                             f"hashes its leaves elsewhere (GPU chunk tree or native hashing), which has no hiding-leaf rule")
        self.leaf_scheme = leaf_scheme
        self.hiding = HidingLeaves() if leaf_scheme == NAME else None
        self._salts: dict[int, bytes] = {}
        if self.hiding is not None:
            self.name = f"{self.name}+hm96"

    def _leaf_ctx(self, step: int) -> bytes:
        """What the context digest of `step` appends: nothing under vllm-v1 (today's roots); under hm96 the scheme's digest, and only
        for a step whose leaves were salted, so no path can bind an hm96 root over unsalted leaves."""
        if self.hiding is None:
            return b""
        if step not in self._salts:
            raise RuntimeError(f"step {step}: its leaves were not salted, and an {NAME} root binds salted leaves only (host path)")
        return b"/leaf=" + self.hiding.digest

    def _salted(self, step: int, leaves: list[bytes]) -> list[bytes]:
        """A step's tree leaves from its position-leaf digests, the step's fresh salts retained under hm96 (none for an empty step)."""
        if self.hiding is None:
            return leaves
        out, self._salts[step] = self.hiding.leaves(leaves)
        return out

    def _unsalted(self, step: int, n_leaves: int) -> bool:
        """Under hm96, whether `step` lacks exactly one salt per leaf: its openings do not verify and the run does not finalize."""
        return self.hiding is not None and len(self._salts.get(step, b"")) != n_leaves * self._salt_bytes()

    def _salt_bytes(self) -> int:
        """The configured instance's salt size (128 bytes for hm96-sha256/v1, 192 for hm96-sha512/v1)."""
        return self.hiding.instance.salt_bytes if self.hiding is not None else 0

    def _salts_of(self, step: int, lo: int, hi: int) -> bytes:
        """The salts of leaves [lo, hi) of `step` (empty under vllm-v1 leaves)."""
        y = self._salt_bytes()
        return b"" if self.hiding is None else self._salts[step][lo * y:hi * y]

    def _leaf_rule(self, inner: list[bytes], salts: bytes) -> list[bytes] | None:
        """The tree leaves of position-leaf digests `inner` under this rule; None for a salt string of the wrong size."""
        if self.hiding is None:
            return None if salts else inner
        y = self._salt_bytes()
        if len(salts) != len(inner) * y:
            return None
        return [self.hiding.leaf(x, salts[i * y:(i + 1) * y]) for i, x in enumerate(inner)]

    def _refuse_padding(self, padding: list) -> None:
        if padding and self.hiding is not None:
            raise ValueError(f"padding steps are position-leaf blocks: not implemented under {NAME} leaves")

    def _leaf_record(self, steps: list) -> dict:
        """The run record's leaf-scheme fields; under hm96, refuses a run with a step that lacks one salt per leaf."""
        bad = [s.step for s in steps if self._unsalted(s.step, s.n_leaves)]
        if bad:
            raise RuntimeError(f"{NAME}: steps {bad[:8]} lack one salt per leaf; refusing to finalize a run whose roots bind {NAME}")
        return {"leaf_scheme": self.leaf_scheme, "salt_bytes": sum(len(s) for s in self._salts.values())}
