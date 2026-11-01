"""PoUW's lead on the int8 plain arm (``tag-blc4a-t64-rot/served/s8/k32/w/l1w/l1a/min``): the serving side of each
noised linear call. `feed` is a call's coin feed and the in-process certifier that stands in for the network certifier's
coin link; `record` is the pass's record.

`LeadRun` is one install's lead (`pouw.install` under `SCHEME`). Each engine step's forward is one pass and one window,
whose calls are fixed in number at install: the hooked linears (``calls``). The window's keyed digests are the
committer's (`commit.lead_rows.WindowKeying`), in the one order it enforces. At the pass's first call the window opens:

1. a fresh window keying, which takes fresh OS bytes for the window's secret ks_w and gives its keyLeaf, ks_w's hm96
   leaf;
2. its registration, the weights' root, keyLeaf and ``calls``, recorded in the pass's record (`record.PassRecord`);
3. the window's salt salt_w, which the coin link samples once it holds all three (`feed.CoinLink.open`), recorded
   (`record.PassRecord.salted`) and taken by the keying.

Then the lead serves each linear call on the arm through the arm's ``gemm`` (`engine.int8_experts.PlainGemm`). The arm
quantizes the call's rows as it always does, then

1. root(A_j), the keyed digest of the call's s8 rows at its position j < ``calls`` of the window's one forward
   (`lead_rows.WindowKeying.a_root`, hashed on the host), is recorded in the pass's record (`record.PassRecord.a_root`);
2. the call's coins come from the coin link, released after that root (`feed.CoinLink.release`);
3. the lead's kernel runs on the rows and the coins (`LeadBackend.call`), giving the served y in the plain arm's place:
   the plain arm's y bit for bit, every coin term folded out of the word exactly.

At the step's end (`LeadRun.end_pass`), the root of the forward's one words instance (`words_root`) is recorded, and the
pass's `PassCommitment` is published, before any sample. A call made outside a step (an engine's dummy run) joins the
next step's window, and a window past ``calls`` calls is refused. The coins are this process's (`feed.IN_PROCESS`) until
the network certifier's coin link replaces them, and they credit nothing.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from verity.core.protocols.pouw.serving import PassCommitment

from verity_vllm.commit import lead_rows as L
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

    def call(self, w: int, layer: Any, xq: Any, s_a: Any, feed: F.Feed) -> Any:
        """Linear ``w``'s call on the s8 rows ``xq`` [m, K] (scales ``s_a`` [m]) under the coins ``feed``: the served y
        [m, N] bf16 in the plain arm's place, the plain arm's y bit for bit (`engine.int8_exact.dequantize` of the exact
        product), whatever the coins. Its committed words join the window's words instance."""

    def words(self) -> Sequence[bytes]:
        """The committed words of the window's calls, one stream a call in call order (`lead_rows.stream` of the
        kernel's export), handed over once at the window's end, after which the backend holds none."""


def words_root(window: L.WindowKeying, backend: LeadBackend) -> bytes:
    """The forward's words root, ``window``'s keyed digest of its calls' committed words at the forward's words position:
    the serving side's one call into the committer for it. Every committed word comes to the host and is hashed there
    (`lead_rows.WindowKeying.words_root`); a subtree top computed on the device is what replaces that, here and in
    `LeadBackend.words`."""
    return window.words_root(FORWARD, backend.words())


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
    """One install's lead on rank ``rank`` of ``world``: each call served through ``backend`` under coins from ``link``,
    each window of at most ``calls`` calls keyed by a fresh ``keying(calls)`` (`lead_rows.WindowKeying`), its record
    committed to the weights' root ``weights_root``. ``commits`` holds every published pass's `PassCommitment`
    (``passes``) in order."""

    def __init__(self, entry: str, backend: LeadBackend, link: F.CoinLink, weights_root: bytes, calls: int, *,
                 rank: int = 0, world: int = 1,
                 keying: Callable[[int], L.WindowKeying] = L.WindowKeying) -> None:
        self.entry, self.backend, self.link, self.weights_root = entry, backend, link, bytes(weights_root)
        self.calls, self.rank, self.world, self.keying = calls, rank, world, keying
        self.kernel = getattr(backend, "variant", None)
        self.passes: list[PassCommitment] = []
        self.retain_dir = None
        self.index = 0
        self.record: R.PassRecord | None = None
        self.window: L.WindowKeying | None = None
        #: the call in flight, until its kernel has served it
        self.unserved: F.Call | None = None
        self.commits: dict[str, Any] = {"scheme": SCHEME, "layout": F.LAYOUT, "coin_source": link.source,
                                        "weights_root": self.weights_root.hex(), "calls": calls, "passes": []}
        self.details: dict[str, Any] = {"scheme": SCHEME, "layout": F.LAYOUT, "coin_source": link.source,
                                        "kernel": self.kernel, "calls": 0, "passes": 0, "uncovered": UNCOVERED}

    def _open(self) -> tuple[R.PassRecord, L.WindowKeying]:
        """The next window: its keying's registration recorded, then the link's salt, recorded and keyed."""
        window = self.keying(self.calls)
        weights, leaf, calls = window.register(self.weights_root)
        record = R.PassRecord(SCHEME, self.rank, self.world, self.index, weights, leaf, calls, kernel=self.kernel)
        record.salted(self.link.open(self.index, leaf, weights, calls))
        window.open(record.salt)
        self.record, self.window = record, window
        return record, window

    def call(self, w: int, layer: Any, xq: Any, s_a: Any) -> Any:
        """Linear ``w``'s call on the s8 rows ``xq`` [m, K] and their scales ``s_a``: the served y."""
        record, window = (self.record, self.window) if self.window is not None else self._open()
        (m, k), n = tuple(xq.shape), int(layer.weight.shape[0])
        call = self.unserved = F.Call(record.pass_index, len(record.calls), w, m, k, n)
        root = window.a_root(FORWARD, call.index, _rows(xq))
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
            raise R.RecordError(f"pass {self.record.pass_index}: call {self.unserved.index} raised before its kernel "
                                "served it, so the pass is never published")
        self.record.words(words_root(self.window, self.backend))
        out = self.record.publish()
        self.passes.append(out)
        self.commits["passes"].append(out.to_json())
        self.details["passes"] = len(self.passes)
        self.record, self.window, self.index = None, None, out.pass_index + 1
        return out

    def finish(self) -> None:
        self.end_pass()
