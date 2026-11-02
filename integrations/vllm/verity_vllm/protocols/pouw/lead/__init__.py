"""PoUW's lead on the int8 plain arm (``tag-blc4a-t64-rot/served/s8/k32/w/l1w/l1a/min``): the serving side of each
noised linear call. `feed` is a call's coin feed and the in-process certifier that stands in for the network certifier's
coin link; `record` is the pass's record.

`LeadRun` is one install's lead (`pouw.install` under `SCHEME`). Each engine step's forward is one pass and one window,
whose calls are fixed at install: one call of each hooked linear (``linears``, each one's n x k). The window's keyed
digests are the committer's (`commit.lead_rows.WindowKeying`), in the one order it enforces. At the pass's first call the
window opens:

1. a fresh window keying, which takes fresh OS bytes for the window's secret ks_w and gives its keyLeaf, ks_w's hm96
   leaf;
2. its registration, the weights' root, keyLeaf and ``calls``, recorded in the pass's record (`record.PassRecord`);
3. the window's salt salt_w, which the coin link samples once it holds all three (`feed.CoinLink.open`), recorded
   (`record.PassRecord.salted`) and taken by the keying;
4. the domain of the forward's one words instance (`lead_rows.WindowKeying.words_domain`), whose leaves the first call's m
   fixes: each linear's chunks at m rows (`words_leaves`).

Then the lead serves each linear call on the arm through the arm's ``gemm`` (`engine.int8_experts.PlainGemm`). The arm
quantizes the call's rows as it always does, then

1. root(A_j), the keyed digest of the call's s8 rows at its position j < ``calls`` of the window's one forward
   (`lead_rows.WindowKeying.a_root`, hashed on the host), is recorded in the pass's record (`record.PassRecord.a_root`);
2. the call's coins come from the coin link, released after that root (`feed.CoinLink.release`);
3. the lead's kernel runs on the rows and the coins (`LeadBackend.call`), giving the served y in the plain arm's place:
   the plain arm's y bit for bit, every coin term folded out of the word exactly. Its export's committed words join the
   words instance on the device at once, under the window's domain.

At the step's end (`LeadRun.end_pass`), the words instance's subtree top (`LeadBackend.words`) is keyed into the forward's
words root (`words_root`) and recorded, and the pass's `PassCommitment` is published, before any sample. A call made
outside a step (an engine's dummy run) joins the next step's window. A window past ``calls`` calls is refused, and so is a
call off its linear's n x k or on other than the window's m rows, since the words were counted at that m: a forward whose
linears see different m, as a MoE layer's experts do, is not covered. The coins are this process's (`feed.IN_PROCESS`)
until the network certifier's coin link replaces them, and they credit nothing.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from verity.core.protocols.pouw.serving import PassCommitment
from verity.ml.commitments.merkle import CommitmentDomain

from verity_vllm.commit import lead_rows as L
from verity_vllm.errors import Refused
from verity_vllm.protocols.pouw.lead import feed as F
from verity_vllm.protocols.pouw.lead import record as R

SCHEME = "lead-blc4a-t64-rot"
#: the pouw settings the lead never reads: its coins come off the coin link, call by call, and it retains nothing
OTHERS = ("coins", "draws", "tamper", "device", "kernel", "cubin", "transcript_dir")
UNCOVERED = "every GEMM outside LinearBase (lm_head, attention)"
#: what `LeadBackend` implements
BACKEND = ("load", "call", "words")
#: a window is one forward, forward 0 of its keying's positions
FORWARD = 0


class LeadBackend(Protocol):
    """The lead's kernel on the device. ``variant`` is the kernel's id, which each pass's commitment carries."""

    variant: str | None

    def load(self, w: int, layer: Any) -> None:
        """Linear ``w``'s weight operands, from ``layer.weight`` [N, K] int8 and ``layer.weight_scale`` [N, 1], once an
        install."""

    def call(self, w: int, layer: Any, xq: Any, s_a: Any, feed: F.Feed, dom: CommitmentDomain) -> Any:
        """Linear ``w``'s call on the s8 rows ``xq`` [m, K] (scales ``s_a`` [m]) under the coins ``feed``: the served y
        [m, N] bf16 in the plain arm's place, the plain arm's y bit for bit (`engine.int8_exact.dequantize` of the exact
        product), whatever the coins. Its export's chunks become the next leaves of the window's words instance under
        ``dom`` (`lead_rows.WindowKeying.words_domain`, one domain for each of a window's calls) at once: added to the
        `lead_rows.DeviceWords` over ``dom`` that the backend makes at the window's first call, so no export outlives
        its hashing."""

    def words(self) -> bytes:
        """The top of the window's words instance (`lead_rows.DeviceWords.top`), the root's two children or the one
        chunk, once at the window's end, after which the backend holds none of the window."""


def words_leaves(m: int, linears: Sequence[tuple[int, int]]) -> int:
    """The leaves of a window's words instance at ``m`` rows a call: each linear's chunks (`lead_rows.chunk_count`), one
    call of each n x k in ``linears``."""
    return sum(L.chunk_count(m, n, F.steps(k)) for n, k in linears)


def words_root(window: L.WindowKeying, backend: LeadBackend, leaves: int) -> bytes:
    """The forward's words root, ``window``'s keyed digest at the forward's words position of the top of its words
    instance of ``leaves`` leaves, which the backend hashed on the device call by call (`LeadBackend.words`): the serving
    side's one call into the committer for it, which refuses a top of the wrong size
    (`lead_rows.WindowKeying.words_top`). No committed word comes to the host."""
    return window.words_top(FORWARD, leaves, backend.words())


def _rows(xq: Any) -> bytes:
    """A_j's bytes on the host: the call's m rows of K int8, row-major."""
    return xq.contiguous().cpu().numpy().tobytes()


def refusals(scheme: str, settings: Any) -> list[str]:
    """Why `scheme`'s parsed pouw `settings` set the lead's settings wrongly, one reason each."""
    if scheme != SCHEME:
        return [] if settings.backend is None else [f"{scheme}: the pouw setting 'backend' is the lead's ({SCHEME})"]
    out = []
    if settings.backend is None:
        out.append(f"{SCHEME}: the pouw setting 'backend' is the lead's kernel (`LeadBackend`: "
                   f"{', '.join(BACKEND)})")
    taken = [k for k in OTHERS if getattr(settings, k) is not None] + (["retain_steps"] if settings.retain_steps else [])
    if taken:
        out.append(f"{SCHEME}: takes each call's coins off its coin link after the call's root(A_j) and retains nothing, "
                   f"so the pouw settings {', '.join(taken)} are other schemes'")
    return out


class LeadRun:
    """One install's lead on rank ``rank`` of ``world``: each call served through ``backend`` under coins from ``link``.
    A window is one call of each of ``linears`` (each one's n x k, ``calls`` of them) on one m, keyed by a fresh
    ``keying(calls)`` (`lead_rows.WindowKeying`), its record committed to the weights' root ``weights_root``.
    ``commits`` holds every published pass's `PassCommitment` (``passes``) in order."""

    def __init__(self, entry: str, backend: LeadBackend, link: F.CoinLink, weights_root: bytes,
                 linears: Sequence[tuple[int, int]], *, rank: int = 0, world: int = 1,
                 keying: Callable[[int], L.WindowKeying] = L.WindowKeying) -> None:
        self.entry, self.backend, self.link, self.weights_root = entry, backend, link, bytes(weights_root)
        self.linears = tuple((int(n), int(k)) for n, k in linears)
        self.calls, self.rank, self.world, self.keying = len(self.linears), rank, world, keying
        self.kernel = getattr(backend, "variant", None)
        self.passes: list[PassCommitment] = []
        self.retain_dir = None
        self.index = 0
        self.record: R.PassRecord | None = None
        self.window: L.WindowKeying | None = None
        #: the open window's m, its words instance's leaves and their domain, fixed at its first call
        self.m: int | None = None
        self.leaves: int | None = None
        self.dom: CommitmentDomain | None = None
        #: the call in flight, until its kernel has served it
        self.unserved: F.Call | None = None
        self.commits: dict[str, Any] = {"scheme": SCHEME, "layout": F.LAYOUT, "coin_source": link.source,
                                        "weights_root": self.weights_root.hex(), "calls": self.calls, "passes": []}
        self.details: dict[str, Any] = {"scheme": SCHEME, "layout": F.LAYOUT, "coin_source": link.source,
                                        "kernel": self.kernel, "calls": 0, "passes": 0, "uncovered": UNCOVERED}

    def _open(self, m: int) -> tuple[R.PassRecord, L.WindowKeying]:
        """The next window: its keying's registration recorded, then the link's salt, recorded and keyed, then its words
        instance's domain, of `words_leaves` at the first call's ``m``."""
        window = self.keying(self.calls)
        weights, leaf, calls = window.register(self.weights_root)
        record = R.PassRecord(SCHEME, self.rank, self.world, self.index, weights, leaf, calls, kernel=self.kernel)
        record.salted(self.link.open(self.index, leaf, weights, calls))
        window.open(record.salt)
        leaves = words_leaves(m, self.linears)
        self.dom = window.words_domain(FORWARD, leaves)
        self.record, self.window, self.m, self.leaves = record, window, m, leaves
        return record, window

    def call(self, w: int, layer: Any, xq: Any, s_a: Any) -> Any:
        """Linear ``w``'s call on the s8 rows ``xq`` [m, K] and their scales ``s_a``: the served y. A call off linear
        ``w``'s n x k, or on other than the window's m, is refused, and its window is never published."""
        (m, k), n = tuple(xq.shape), int(layer.weight.shape[0])
        record, window = (self.record, self.window) if self.window is not None else self._open(m)
        call = self.unserved = F.Call(record.pass_index, len(record.calls), w, m, k, n)
        if m != self.m:
            raise Refused(f"pass {record.pass_index}: call {call.index} (linear {w}) has {m} rows, and the window's "
                          f"words were counted at its first call's {self.m}: a window's calls share one m")
        if not 0 <= w < self.calls or self.linears[w] != (n, k):
            raise Refused(f"pass {record.pass_index}: call {call.index} is {n} x {k} on linear {w}, not one of the "
                          f"{self.calls} linears the window's words were counted for")
        root = window.a_root(FORWARD, call.index, _rows(xq))
        record.a_root(call, root)
        y = self.backend.call(w, layer, xq, s_a, self.link.release(call, root), self.dom)
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
            raise R.RecordError(f"pass {self.record.pass_index}: call {self.unserved.index} raised before its kernel "
                                "served it, so the pass is never published")
        self.record.words(words_root(self.window, self.backend, self.leaves))
        out = self.record.publish()
        self.passes.append(out)
        self.commits["passes"].append(out.to_json())
        self.details["passes"] = len(self.passes)
        self.record, self.window, self.index = None, None, out.pass_index + 1
        self.m = self.leaves = self.dom = None
        return out

    def finish(self) -> None:
        self.end_pass()
