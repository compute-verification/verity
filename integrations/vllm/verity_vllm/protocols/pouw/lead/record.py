"""The lead's pass record on the serving side: each call's header, root(A_j) and words root in call order, published as
core's `PassCommitment` before any sample.

Both roots are the committer's (A_j's and the words' frame-v3 leaves, each at its own position), which the record never
recomputes: it fixes their order. A call's A root comes before its words, and calls come in order. `publish` freezes
the record: activation_roots is the SHA-512 over the A roots in call order, transcript_root over the words roots,
records_root over the calls' headers (six u64be each: pass, call, linear, m, K, n), units the calls and tiles their
committed MMA tiles of words, (R/16)·(C/8)·S a call. A call after publication, and a sample before it (`commitment`),
are refused.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

from verity.core.protocols.pouw.serving import PassCommitment

from verity_vllm.protocols.pouw.lead.feed import Call, Shape

ROOT_BYTES = 64


class RecordError(ValueError):
    """A call or a sample the pass's record refuses."""


def header(call: Call) -> bytes:
    return b"".join(int(v).to_bytes(8, "big") for v in (call.pass_index, call.index, call.linear, call.m, call.k, call.n))


def tiles(sh: Shape) -> int:
    """The call's committed MMA tiles of words: row blocks of 16, column blocks of 8, steps."""
    return (sh.R // 16) * (sh.C // 8) * sh.S


def _sha512(parts: Sequence[bytes]) -> bytes:
    return hashlib.sha512(b"".join(parts)).digest()


def _root(call: Call, what: str, root: bytes) -> bytes:
    if len(root) != ROOT_BYTES:
        raise RecordError(f"call {call.index}: {what} root of {len(root)} bytes; it is {ROOT_BYTES}")
    return bytes(root)


class PassRecord:
    """Pass ``pass_index``'s record on rank ``rank`` of ``world``; ``scheme``, ``salt``, ``weights_root``, ``kernel`` and
    ``controls`` go into its `PassCommitment` as given."""

    def __init__(self, scheme: str, rank: int, world: int, pass_index: int, salt: bytes, weights_root: bytes,
                 kernel: str | None = None, controls: Sequence[str] = ()) -> None:
        self.scheme, self.rank, self.world, self.pass_index = scheme, rank, world, pass_index
        self.salt, self.weights_root = bytes(salt), bytes(weights_root)
        self.kernel, self.controls = kernel, tuple(controls)
        self.calls: list[Call] = []
        self.a_roots: list[bytes] = []
        self.words_roots: list[bytes] = []
        self.published: PassCommitment | None = None

    def _open(self, call: Call, what: str) -> None:
        if self.published is not None:
            raise RecordError(f"pass {self.pass_index}: {what} of call {call.index} after the pass's record was published")
        if call.pass_index != self.pass_index:
            raise RecordError(f"call {call.index} of pass {call.pass_index} in pass {self.pass_index}'s record")

    def a_root(self, call: Call, root: bytes) -> None:
        """The call's root(A_j): the next call's, once, before its words."""
        self._open(call, "A")
        if call.index != len(self.a_roots) or len(self.a_roots) != len(self.words_roots):
            raise RecordError(f"pass {self.pass_index}: A of call {call.index}, the record's next is call "
                              f"{len(self.words_roots)} and its A comes once, before its words")
        root = _root(call, "an A", root)
        self.calls.append(call)
        self.a_roots.append(root)

    def words(self, call: Call, root: bytes) -> None:
        """The call's words root, after its root(A_j)."""
        self._open(call, "words")
        if len(self.a_roots) != len(self.words_roots) + 1 or self.calls[-1] != call:
            raise RecordError(f"pass {self.pass_index}: words of call {call.index} before its A root")
        self.words_roots.append(_root(call, "a words", root))

    def publish(self) -> PassCommitment:
        """The pass's `PassCommitment`, once every call's words are recorded; published once."""
        if self.published is not None:
            raise RecordError(f"pass {self.pass_index}: its record is published once")
        if len(self.a_roots) != len(self.words_roots):
            raise RecordError(f"pass {self.pass_index}: call {self.calls[-1].index} has its A root and no words")
        self.published = PassCommitment(
            self.scheme, self.rank, self.world, self.pass_index, self.salt, self.weights_root,
            transcript_root=_sha512(self.words_roots), activation_roots=_sha512(self.a_roots), units=len(self.calls),
            tiles=sum(tiles(c.shape) for c in self.calls), records_root=_sha512([header(c) for c in self.calls]),
            kernel=self.kernel, controls=self.controls)
        return self.published

    def commitment(self) -> PassCommitment:
        """What a sample reads: the published record. Refused before publication."""
        if self.published is None:
            raise RecordError(f"pass {self.pass_index}: no sample before the pass's record is published")
        return self.published
