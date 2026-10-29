"""The lead's commitments on the serving side: each call's root(A_j), sent up before its coins, and the pass's record,
published as core's `PassCommitment` before any sample.

* **root(A_j)** (`PassRecord.a_root`): the call's A, its s8 rows xq [m, K] zero-padded to R = 16⌈m/16⌉ rows (Lean's
  `padRows`), as one byte stream in rows-sha512's rows (`commit.step_rows`: hm96-sha512 hiding leaves in a
  frame-v3-sha512 tree, a 64-byte root). The tree's domain binds the lead (`PROGRAM`), the call (its pass, index and
  linear) and its geometry (R, K); its salts come from the call's key, SHA-512(`A_TAG` ‖ Kp ‖ u64be(pass) ‖ u64be(call)),
  Kp the pass's 64-byte key. This is the committer's format until the lead fixes its keyed-leaf bytes; `a_root` is the
  one function that changes then.
* **The pass's record** (`PassRecord`): each call's header, root(A_j) and words root (the call's committed words, hashed
  by the committer) in call order. A call's root comes before its words, and calls come in order. `publish` freezes the
  record as `PassCommitment`: activation_roots is the SHA-512 over the A roots in call order, transcript_root over the
  words roots, records_root over the calls' headers (six u64be each: pass, call, linear, m, K, n), units the calls and
  tiles their committed MMA tiles of words (Lean's `rowBlocks · colBlocks · S`). A call after publication, and a sample
  before it (`commitment`), are refused.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from verity.core.protocols.pouw.serving import PassCommitment
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.merkle import CommitmentDomain, tree_levels

from verity_vllm.commit import step_rows as SR
from verity_vllm.protocols.pouw.lead.feed import LAYOUT, Call, Shape

A_TAG = "verity-vllm/pouw-lead-a/v1"
#: the lead in the A tree's domain: its layout
PROGRAM = identity_digest_sha512(A_TAG, {"layout": LAYOUT})
KEY_BYTES = SR.KEY_BYTES
ROOT_BYTES = SR.ROOT_BYTES


class RecordError(ValueError):
    """A call or a sample the pass's record refuses."""


def header(call: Call) -> bytes:
    return b"".join(int(v).to_bytes(8, "big") for v in (call.pass_index, call.index, call.linear, call.m, call.k, call.n))


def call_key(kp: bytes, call: Call) -> bytes:
    """The salt key of the call's A tree under the pass's key ``kp``."""
    if len(kp) != KEY_BYTES:
        raise RecordError(f"a pass key is {KEY_BYTES} bytes")
    return hashlib.sha512(A_TAG.encode() + kp + header(call)[:16]).digest()


def a_domain(call: Call, rows: int) -> CommitmentDomain:
    sh = call.shape
    ctx = identity_digest_sha512(A_TAG, {"pass": call.pass_index, "call": call.index, "linear": call.linear})
    geo = identity_digest_sha512(A_TAG, {"rows": sh.R, "k": sh.k})
    return SR.domain(PROGRAM, ctx, geo, rows)


def a_root(xq: np.ndarray, call: Call, kp: bytes) -> bytes:
    """root(A_j) of the call's s8 rows ``xq`` [m, K] under the pass's key ``kp``."""
    sh = call.shape
    xq = np.asarray(xq)
    if xq.shape != (sh.m, sh.k) or xq.dtype != np.int8:
        raise RecordError(f"call {call.index}: A is {xq.dtype} {xq.shape}, the call's is int8 ({sh.m}, {sh.k})")
    data = np.zeros((sh.R, sh.k), dtype=np.int8)
    data[:sh.m] = xq
    hs = SR.tensor_hs(data.tobytes(), call_key(kp, call))
    dom = a_domain(call, len(hs))
    return tree_levels(dom, [SR.leaf(dom, i, h) for i, h in enumerate(hs)])[-1][0]


def tiles(sh: Shape) -> int:
    """The call's committed MMA tiles of words: row blocks of 16, column blocks of 8, steps."""
    return (sh.R // 16) * (sh.C // 8) * sh.S


def _sha512(parts: Sequence[bytes]) -> bytes:
    return hashlib.sha512(b"".join(parts)).digest()


class PassRecord:
    """Pass ``pass_index``'s record on rank ``rank`` of ``world``, its A trees salted from ``key``; ``scheme``, ``salt``,
    ``weights_root``, ``kernel`` and ``controls`` go into its `PassCommitment` as given."""

    def __init__(self, scheme: str, rank: int, world: int, pass_index: int, key: bytes, salt: bytes, weights_root: bytes,
                 kernel: str | None = None, controls: Sequence[str] = ()) -> None:
        if len(key) != KEY_BYTES:
            raise RecordError(f"a pass key is {KEY_BYTES} bytes")
        self.scheme, self.rank, self.world, self.pass_index = scheme, rank, world, pass_index
        self.key, self.salt, self.weights_root = bytes(key), bytes(salt), bytes(weights_root)
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

    def a_root(self, call: Call, xq: np.ndarray) -> bytes:
        """The call's root(A_j), recorded: the next call's, before its words."""
        self._open(call, "A")
        if call.index != len(self.a_roots) or len(self.a_roots) != len(self.words_roots):
            raise RecordError(f"pass {self.pass_index}: A of call {call.index}, the record's next is call "
                              f"{len(self.words_roots)} and its A comes once, before its words")
        root = a_root(xq, call, self.key)
        self.calls.append(call)
        self.a_roots.append(root)
        return root

    def words(self, call: Call, root: bytes) -> None:
        """The call's words root, after its root(A_j)."""
        self._open(call, "words")
        if len(self.a_roots) != len(self.words_roots) + 1 or self.calls[-1] != call:
            raise RecordError(f"pass {self.pass_index}: words of call {call.index} before its A root")
        if len(root) != ROOT_BYTES:
            raise RecordError(f"call {call.index}: a words root of {len(root)} bytes; it is {ROOT_BYTES}")
        self.words_roots.append(bytes(root))

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
