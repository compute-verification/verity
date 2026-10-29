"""PoUW's lead on the int8 plain arm (``tag-blc4a-t64-rot/served/s8/k32/w/l1w/l1a/min``, Lean's `Pouw.Lead.lead`): the
serving side of each noised linear call. `feed` is a call's coin feed and the in-process certifier that stands in for the
network certifier's coin link; `commit` is root(A_j) and the pass's record.

`LeadRun` is one install's lead (`pouw.install` under `SCHEME`). It serves each linear call on the arm through the arm's
``gemm`` (`engine.int8_experts.PlainGemm`): the arm quantizes the call's rows as it always does, then

1. root(A_j) of the call's s8 rows, recorded in the pass's record (`commit.PassRecord.a_root`);
2. the call's coins from the coin link, released after that root (`feed.CoinLink.release`);
3. the lead's kernel and committer (`LeadBackend.call`) on the rows and the coins: the served y, the plain arm's, and the
   root of the call's words;
4. the words root, recorded after root(A_j) (`commit.PassRecord.words`).

Each engine step's calls are one pass, whose `PassCommitment` is published at the step's end (`LeadRun.end_pass`), before
any sample. The coins are this process's (`feed.IN_PROCESS`) until the network certifier's coin link replaces them, and
credit nothing.
"""
from __future__ import annotations

import secrets
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
from verity.core.protocols.pouw.serving import PassCommitment

from verity_vllm.protocols.pouw.lead import commit as C
from verity_vllm.protocols.pouw.lead import feed as F

SCHEME = "lead-blc4a-t64-rot"
#: the pouw settings the lead never reads: its coins come off the coin link, call by call, and it retains nothing
OTHERS = ("coins", "draws", "tamper", "device", "kernel", "cubin", "transcript_dir")
UNCOVERED = "every GEMM outside LinearBase (lm_head, attention)"


class LeadBackend(Protocol):
    """The lead's kernel and its committer on the device. ``variant`` is the kernel's id, which each pass's commitment
    carries."""

    variant: str | None

    def load(self, w: int, layer: Any) -> None:
        """Linear ``w``'s weight operands, from ``layer.weight`` [N, K] int8 and ``layer.weight_scale`` [N, 1], once an
        install."""

    def call(self, w: int, layer: Any, xq: Any, s_a: Any, feed: F.Feed) -> tuple[Any, bytes]:
        """Linear ``w``'s call on the s8 rows ``xq`` [m, K] (scales ``s_a`` [m]) under the coins ``feed``: the served y
        [m, N] bf16, the plain arm's, and the 64-byte root of the call's committed words."""


def refusals(scheme: str, settings: Any) -> list[str]:
    """Why `scheme`'s parsed pouw `settings` set the lead's settings wrongly, one reason each."""
    if scheme != SCHEME:
        return [] if settings.backend is None else [f"{scheme}: the pouw setting 'backend' is the lead's ({SCHEME})"]
    out = []
    if settings.backend is None:
        out.append(f"{SCHEME}: the pouw setting 'backend' is the lead's kernel and committer (`LeadBackend`: load, call)")
    taken = [k for k in OTHERS if getattr(settings, k) is not None] + (["retain_steps"] if settings.retain_steps else [])
    if taken:
        out.append(f"{SCHEME}: takes each call's coins off its coin link after the call's root(A_j) and retains nothing, "
                   f"so the pouw settings {', '.join(taken)} are other schemes'")
    return out


def _host(t: Any) -> np.ndarray:
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


class LeadRun:
    """One install's lead on rank ``rank`` of ``world``: each call served through ``backend`` under coins from ``link``,
    each pass's record keyed afresh (``key(64)``) and committed to the weights' root ``weights_root``. ``commits`` holds
    every published pass's `PassCommitment` (``passes``) in order."""

    def __init__(self, entry: str, backend: LeadBackend, link: F.CoinLink, weights_root: bytes, *, rank: int = 0,
                 world: int = 1, key: Callable[[int], bytes] = secrets.token_bytes) -> None:
        self.entry, self.backend, self.link, self.weights_root = entry, backend, link, bytes(weights_root)
        self.rank, self.world, self.key = rank, world, key
        self.kernel = getattr(backend, "variant", None)
        self.passes: list[PassCommitment] = []
        self.retain_dir = None
        self.record = self._record(0)
        self.commits: dict[str, Any] = {"scheme": SCHEME, "layout": F.LAYOUT, "coin_source": link.source,
                                        "weights_root": self.weights_root.hex(), "passes": []}
        self.details: dict[str, Any] = {"scheme": SCHEME, "layout": F.LAYOUT, "coin_source": link.source,
                                        "kernel": self.kernel, "calls": 0, "passes": 0, "uncovered": UNCOVERED}

    def _record(self, index: int) -> C.PassRecord:
        return C.PassRecord(SCHEME, self.rank, self.world, index, self.key(C.KEY_BYTES), b"", self.weights_root,
                            kernel=self.kernel)

    def call(self, w: int, layer: Any, xq: Any, s_a: Any) -> Any:
        """Linear ``w``'s call on the s8 rows ``xq`` [m, K] and their scales ``s_a``: the served y."""
        (m, k), n = tuple(xq.shape), int(layer.weight.shape[0])
        call = F.Call(self.record.pass_index, len(self.record.calls), w, m, k, n)
        root = self.record.a_root(call, _host(xq))
        y, words = self.backend.call(w, layer, xq, s_a, self.link.release(call, root))
        self.record.words(call, words)
        self.details["calls"] += 1
        return y

    def end_pass(self, step: bool = False) -> PassCommitment | None:
        """Publish the open pass's record and open the next; a pass with no calls ends nothing, and the last published
        pass's commitment is returned."""
        if not self.record.calls:
            return self.passes[-1] if self.passes else None
        out = self.record.publish()
        self.passes.append(out)
        self.commits["passes"].append(out.to_json())
        self.details["passes"] = len(self.passes)
        self.record = self._record(out.pass_index + 1)
        return out

    def finish(self) -> None:
        self.end_pass()
