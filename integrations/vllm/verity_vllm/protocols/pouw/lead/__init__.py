"""PoUW's lead on the int8 plain arm (``tag-blc4a-t64-rot/served/s8/k32/w/l1w/l1a/min``): the serving side of each
noised linear call. `feed` is a call's coin feed and the in-process certifier that stands in for the network certifier's
coin link; `record` is the pass's record.

`LeadRun` is one install's lead (`pouw.install` under `SCHEME`). Each engine step's forward is one pass and one window,
whose calls are fixed in number at install: the hooked linears (``calls``). At the pass's first call the window opens:

1. a fresh key ks, and the committer's hm96 leaf over it (`LeadBackend.key_leaf`);
2. the pass's record of the weights' root, that key leaf and ``calls`` (`record.PassRecord`);
3. the window's salt salt_w, which the coin link samples once it holds all three (`feed.CoinLink.open`), recorded
   (`record.PassRecord.salted`).

Then the lead serves each linear call on the arm through the arm's ``gemm`` (`engine.int8_experts.PlainGemm`). The arm
quantizes the call's rows as it always does, then

1. root(A_j) of the call's s8 rows, the committer's at the call's position j < ``calls`` under ks and salt_w
   (`LeadBackend.a_root`), is recorded in the pass's record (`record.PassRecord.a_root`);
2. the call's coins come from the coin link, released after that root (`feed.CoinLink.release`);
3. the lead's kernel and committer run on the rows and the coins (`LeadBackend.call`), giving the served y in the plain
   arm's place: the layout's fold of the noisy word (A·B plus the coin terms), so not the plain arm's y.

At the step's end (`LeadRun.end_pass`), the committer's root of the forward's one words instance (`LeadBackend.words_root`)
is recorded, and the pass's `PassCommitment` is published, before any sample. A call made outside a step (an engine's
dummy run) joins the next step's window, and a window past ``calls`` calls is refused. The coins are this process's
(`feed.IN_PROCESS`) until the network certifier's coin link replaces them, and they credit nothing.
"""
from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from verity.core.protocols.pouw.serving import PassCommitment

from verity_vllm.protocols.pouw.lead import feed as F
from verity_vllm.protocols.pouw.lead import record as R

__all__ = ["SCHEME", "LeadBackend", "WindowKeys"]

SCHEME = "lead-blc4a-t64-rot"
#: the pouw settings the lead never reads: its coins come off the coin link, call by call, and it retains nothing
OTHERS = ("coins", "draws", "tamper", "device", "kernel", "cubin", "transcript_dir")
UNCOVERED = "every GEMM outside LinearBase (lm_head, attention)"
#: a window's key ks, the committer's: fresh for each window, never published
KEY_BYTES = 64
#: what `LeadBackend` implements
BACKEND = ("load", "key_leaf", "a_root", "call", "words_root")


@dataclass(frozen=True)
class WindowKeys:
    """Pass ``index``'s window as the committer reads it: its one forward is forward 0, of at most ``calls`` calls, under
    the key ``key`` (ks) and the certifier's salt ``salt`` (salt_w)."""

    index: int
    calls: int
    key: bytes
    salt: bytes


class LeadBackend(Protocol):
    """The lead's kernel and its committer on the device. ``variant`` is the kernel's id, which each pass's commitment
    carries."""

    variant: str | None

    def load(self, w: int, layer: Any) -> None:
        """Linear ``w``'s weight operands, from ``layer.weight`` [N, K] int8 and ``layer.weight_scale`` [N, 1], once an
        install."""

    def key_leaf(self, key: bytes) -> bytes:
        """The 64-byte hm96 leaf over a window's key ``key``, which binds the key before the certifier samples the window's
        salt: with that salt, the key derives the hiding of the committer's leaves."""

    def a_root(self, w: int, call: F.Call, xq: Any, window: WindowKeys) -> bytes:
        """root(A_j) of the call's s8 rows ``xq`` [m, K]: 64 bytes, the keyed root of A_j's instance at position
        ``call.index`` of ``window`` (the committer's 1 KiB segments, each its own positioned frame-v3 leaf)."""

    def call(self, w: int, layer: Any, xq: Any, s_a: Any, feed: F.Feed) -> Any:
        """Linear ``w``'s call on the s8 rows ``xq`` [m, K] (scales ``s_a`` [m]) under the coins ``feed``: the served y
        [m, N] bf16 in the plain arm's place, the layout's fold of the noisy word. Its committed words join the window's
        words instance."""

    def words_root(self, window: WindowKeys) -> bytes:
        """The 64-byte keyed root of ``window``'s words instance: its forward's calls' committed words in call order."""


def refusals(scheme: str, settings: Any) -> list[str]:
    """Why `scheme`'s parsed pouw `settings` set the lead's settings wrongly, one reason each."""
    if scheme != SCHEME:
        return [] if settings.backend is None else [f"{scheme}: the pouw setting 'backend' is the lead's ({SCHEME})"]
    out = []
    if settings.backend is None:
        out.append(f"{SCHEME}: the pouw setting 'backend' is the lead's kernel and committer (`LeadBackend`: "
                   f"{', '.join(BACKEND)})")
    taken = [k for k in OTHERS if getattr(settings, k) is not None] + (["retain_steps"] if settings.retain_steps else [])
    if taken:
        out.append(f"{SCHEME}: takes each call's coins off its coin link after the call's root(A_j) and retains nothing, "
                   f"so the pouw settings {', '.join(taken)} are other schemes'")
    return out


class LeadRun:
    """One install's lead on rank ``rank`` of ``world``: each call served through ``backend`` under coins from ``link``,
    each window of at most ``calls`` calls under a fresh key (``key(KEY_BYTES)``), its record committed to the weights'
    root ``weights_root``. ``commits`` holds every published pass's `PassCommitment` (``passes``) in order."""

    def __init__(self, entry: str, backend: LeadBackend, link: F.CoinLink, weights_root: bytes, calls: int, *,
                 rank: int = 0, world: int = 1, key: Callable[[int], bytes] = secrets.token_bytes) -> None:
        self.entry, self.backend, self.link, self.weights_root = entry, backend, link, bytes(weights_root)
        self.calls, self.rank, self.world, self.key = calls, rank, world, key
        self.kernel = getattr(backend, "variant", None)
        self.passes: list[PassCommitment] = []
        self.retain_dir = None
        self.index = 0
        self.record: R.PassRecord | None = None
        self.window: WindowKeys | None = None
        #: the call in flight, until its kernel has served it
        self.unserved: F.Call | None = None
        self.commits: dict[str, Any] = {"scheme": SCHEME, "layout": F.LAYOUT, "coin_source": link.source,
                                        "weights_root": self.weights_root.hex(), "calls": calls, "passes": []}
        self.details: dict[str, Any] = {"scheme": SCHEME, "layout": F.LAYOUT, "coin_source": link.source,
                                        "kernel": self.kernel, "calls": 0, "passes": 0, "uncovered": UNCOVERED}

    def _open(self) -> tuple[R.PassRecord, WindowKeys]:
        """The next window: its key's leaf and the weights' root and call count recorded, then the link's salt."""
        ks = self.key(KEY_BYTES)
        leaf = self.backend.key_leaf(ks)
        record = R.PassRecord(SCHEME, self.rank, self.world, self.index, self.weights_root, leaf, self.calls,
                              kernel=self.kernel)
        salt = self.link.open(self.index, leaf, self.weights_root, self.calls)
        record.salted(salt)
        self.record, self.window = record, WindowKeys(self.index, self.calls, ks, record.salt)
        return self.record, self.window

    def call(self, w: int, layer: Any, xq: Any, s_a: Any) -> Any:
        """Linear ``w``'s call on the s8 rows ``xq`` [m, K] and their scales ``s_a``: the served y."""
        record, window = (self.record, self.window) if self.window is not None else self._open()
        (m, k), n = tuple(xq.shape), int(layer.weight.shape[0])
        call = self.unserved = F.Call(window.index, len(record.calls), w, m, k, n)
        root = self.backend.a_root(w, call, xq, window)
        record.a_root(call, root)
        y = self.backend.call(w, layer, xq, s_a, self.link.release(call, root))
        self.unserved = None
        self.details["calls"] += 1
        return y

    def end_pass(self, step: bool = False) -> PassCommitment | None:
        """Publish the open window's record, after the committer's words root, and leave the next window to open at its
        first call; a pass with no calls ends nothing, and the last published pass's commitment is returned. A window
        with a call that raised is never published: no retries."""
        if self.window is None or self.record is None:
            return self.passes[-1] if self.passes else None
        if self.unserved is not None:
            raise R.RecordError(f"pass {self.window.index}: call {self.unserved.index} raised before its kernel served "
                                "it, so the pass is never published")
        self.record.words(self.backend.words_root(self.window))
        out = self.record.publish()
        self.passes.append(out)
        self.commits["passes"].append(out.to_json())
        self.details["passes"] = len(self.passes)
        self.record, self.window, self.index = None, None, out.pass_index + 1
        return out

    def finish(self) -> None:
        self.end_pass()
