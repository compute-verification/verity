"""The lead's pass record on the serving side. A pass is one window: one engine step's forward, its calls' root(A_j)s and
the one root of its words.

The window opens in this order. The record takes the weights' root, the committer's hm96 leaf over the window's key
(``key_leaf``, which binds the key the window's coins derive from) and the window's call count (``calls``), all three
fixed before the certifier samples the window's salt salt_w. Then the record takes salt_w (`salted`). Then come the
calls, in order: each one's root(A_j), at a position j below ``calls`` (`a_root`), so the committer's positions are
injective. Once every call is in, the forward's words root comes (`words`): one words instance a forward, its calls'
words in call order.

Both kinds of root are the committer's, and the record never recomputes them: it fixes their order. `publish` freezes
the record as core's `PassCommitment`:

- ``salt`` is salt_w;
- ``activation_roots`` is the SHA-512 over the A roots in call order, and ``transcript_root`` the words root;
- ``records_root`` is the SHA-512 over the window's header (``key_leaf``, then ``calls`` as a u64be) and the calls'
  headers (six u64be each: pass, call, linear, m, K, n);
- ``units`` counts the calls, and ``tiles`` their committed MMA tiles of words, (R/16)·(C/8)·S a call.

A sample before publication (`commitment`) is refused, and so is anything recorded after it.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

from verity.core.protocols.pouw.serving import PassCommitment

from verity_vllm.protocols.pouw.lead.feed import Call, Shape

ROOT_BYTES = 64
#: a window's A positions f·calls + j stay below 2^63, where its words positions begin
MAX_CALLS = 2**63


class RecordError(ValueError):
    """A call or a sample the pass's record refuses."""


def header(call: Call) -> bytes:
    return b"".join(int(v).to_bytes(8, "big") for v in (call.pass_index, call.index, call.linear, call.m, call.k, call.n))


def tiles(sh: Shape) -> int:
    """The call's committed MMA tiles of words: row blocks of 16, column blocks of 8, steps."""
    return (sh.R // 16) * (sh.C // 8) * sh.S


def _sha512(parts: Sequence[bytes]) -> bytes:
    return hashlib.sha512(b"".join(parts)).digest()


class PassRecord:
    """Pass ``pass_index``'s record on rank ``rank`` of ``world``: one window of at most ``calls`` calls, under the
    weights' root ``weights_root`` and the key's leaf ``key_leaf``. ``scheme``, ``kernel`` and ``controls`` go into its
    `PassCommitment` as given."""

    def __init__(self, scheme: str, rank: int, world: int, pass_index: int, weights_root: bytes, key_leaf: bytes,
                 calls: int, kernel: str | None = None, controls: Sequence[str] = ()) -> None:
        self.scheme, self.rank, self.world, self.pass_index = scheme, rank, world, pass_index
        if not 1 <= calls < MAX_CALLS:
            raise RecordError(f"pass {pass_index}: a window of {calls} calls; it holds 1 to 2^63 - 1")
        self.weights_root = self._digest("weights' root", weights_root)
        self.key_leaf = self._digest("key leaf", key_leaf)
        self.window_calls = calls
        self.kernel, self.controls = kernel, tuple(controls)
        self.salt: bytes | None = None
        self.calls: list[Call] = []
        self.a_roots: list[bytes] = []
        self.words_root: bytes | None = None
        self.published: PassCommitment | None = None

    def _digest(self, what: str, value: bytes) -> bytes:
        if len(value) != ROOT_BYTES:
            raise RecordError(f"pass {self.pass_index}: a {what} of {len(value)} bytes; it is {ROOT_BYTES}")
        return bytes(value)

    def _open(self, what: str) -> None:
        if self.published is not None:
            raise RecordError(f"pass {self.pass_index}: {what} after the pass's record was published")

    def salted(self, salt: bytes) -> None:
        """The window's salt salt_w, which the certifier sampled once it held the weights' root, the key leaf and the
        call count: once, before any A root."""
        self._open("the window's salt")
        if self.salt is not None:
            raise RecordError(f"pass {self.pass_index}: the window's salt comes once")
        self.salt = self._digest("salt", salt)

    def a_root(self, call: Call, root: bytes) -> None:
        """The call's root(A_j): the next call's, after the window's salt and before the words root, at a position below
        the window's call count."""
        what = f"A of call {call.index}"
        self._open(what)
        if call.pass_index != self.pass_index:
            raise RecordError(f"call {call.index} of pass {call.pass_index} in pass {self.pass_index}'s record")
        if self.salt is None:
            raise RecordError(f"pass {self.pass_index}: {what} before the window's salt")
        if self.words_root is not None:
            raise RecordError(f"pass {self.pass_index}: {what} after the forward's words root")
        if call.index != len(self.calls):
            raise RecordError(f"pass {self.pass_index}: {what}; the record's next is call {len(self.calls)}")
        if call.index >= self.window_calls:
            raise RecordError(f"pass {self.pass_index}: {what} is past the window's {self.window_calls} calls, fixed "
                              "before its salt")
        root = self._digest("call's A root", root)
        self.calls.append(call)
        self.a_roots.append(root)

    def words(self, root: bytes) -> None:
        """The forward's words root: once, after its calls' A roots."""
        self._open("the words root")
        if not self.calls:
            raise RecordError(f"pass {self.pass_index}: the words root before any call's A root")
        if self.words_root is not None:
            raise RecordError(f"pass {self.pass_index}: the words root comes once")
        self.words_root = self._digest("words root", root)

    def publish(self) -> PassCommitment:
        """The pass's `PassCommitment`, once the words root is recorded; published once."""
        if self.published is not None:
            raise RecordError(f"pass {self.pass_index}: its record is published once")
        if self.words_root is None or self.salt is None:
            raise RecordError(f"pass {self.pass_index}: published before its forward's words root")
        window = self.key_leaf + self.window_calls.to_bytes(8, "big")
        self.published = PassCommitment(
            self.scheme, self.rank, self.world, self.pass_index, self.salt, self.weights_root,
            transcript_root=self.words_root, activation_roots=_sha512(self.a_roots), units=len(self.calls),
            tiles=sum(tiles(c.shape) for c in self.calls), records_root=_sha512([window, *map(header, self.calls)]),
            kernel=self.kernel, controls=self.controls)
        return self.published

    def commitment(self) -> PassCommitment:
        """What a sample reads: the published record. Refused before publication."""
        if self.published is None:
            raise RecordError(f"pass {self.pass_index}: no sample before the pass's record is published")
        return self.published
