"""The native host committer's choice of leaf rule (:class:`LeafSchemeRule`, opt-in, off by default): ``vllm-v1`` position
leaves, or ``hm96-sha256/v1`` hiding leaves over them (`hiding.HidingLeaves`), each step's salts (128 fresh bytes a leaf, one OS
call per step) retained with the step.  ``tests/commit/test_hiding.py`` pins it.
"""
from __future__ import annotations

from verity.ml.commitments import vllm_v1

from verity.ml.windows.hiding import NAME, HidingLeaves

LEAF_SCHEMES = (vllm_v1.NAME, NAME)


class LeafSchemeRule:
    """The native host committer's leaf rule (a mixin of ``NativeHostCommitter``): ``vllm-v1`` position leaves (the default, the
    recorded roots), or ``hm96-sha256/v1`` leaves over them.  Under hm96 every step's salts are retained (the stored bytes), openings
    carry them, and the step roots bind the scheme's digest through the context digest.

    hm96 fails closed on the path the leaves take: only ``NativeHostCommitter``'s own host path salts (``_salted``), a committer
    whose step path is another (the GPU chunk tree, a subclass's native hashing) is refused at construction, the context digest
    binds the scheme only for a step whose leaves were salted (every step root is bound through it: ``commit_block_offline`` and the
    native collector's roots included), and a step without one salt per leaf neither opens nor finalizes."""

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

    def _leaf_record(self, steps: list) -> dict:
        """The run record's leaf-scheme fields; under hm96, refuses a run with a step that lacks one salt per leaf."""
        bad = [s.step for s in steps if self._unsalted(s.step, s.n_leaves)]
        if bad:
            raise RuntimeError(f"{NAME}: steps {bad[:8]} lack one salt per leaf; refusing to finalize a run whose roots bind {NAME}")
        return {"leaf_scheme": self.leaf_scheme, "salt_bytes": sum(len(s) for s in self._salts.values())}
