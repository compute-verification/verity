"""What a windowed step keeps on host for openings (`native_collect`'s windowed staging): its pinned windows (`_HostWindows`) or
byte ranges of its stream (`_HostRanges`)."""

from __future__ import annotations

from typing import Any

import torch

from verity_vllm.commit.committer.native_host import GPU_CHUNK_BYTES

class _HostWindows:
    """The raw bytes of one step retained on host as pinned windows (P0 windowed staging); quacks like the contiguous `keep` tensor for
    openings (`blk[lo:hi]` -> CPU uint8 tensor, may span two windows), `numel()`, `is_cuda`.  `wins` = [(k, hv, n)] with k the window
    index (window k covers bytes [k W, k W + n)); the list may be SPARSE (retain=sample: only the windows holding challenged leaves are
    kept -- the replay opening path at batch sizes whose whole raw stream does not fit the pod's host cgroup); a slice touching a window
    that was not retained is an error, never zeros."""
    is_cuda = False

    def __init__(self, wins: list[tuple[int, Any, int]], used: int, W: int, chunk: int = GPU_CHUNK_BYTES) -> None:
        self.wins = {int(k): (hv, int(n)) for k, hv, n in wins if n}
        self.used = int(used)
        self.W = int(W)
        self.n = -(-self.used // chunk) * chunk            # logical (padded) length of the step, not the retained bytes

    def numel(self) -> int:
        return self.n

    @property
    def retained_bytes(self) -> int:
        return sum(n for _hv, n in self.wins.values())

    def holds(self, lo: int, hi: int) -> bool:
        """True when every byte of [lo, hi) (clipped to the padded step) is retained -- the opening path asks BEFORE slicing, so a leaf
        whose bytes were not kept (retain=sample; a window that did not land) is routed to replay instead of raising."""
        lo, hi = int(lo), int(min(hi, self.n))
        if hi <= lo:
            return True
        for k in range(lo // self.W, (hi - 1) // self.W + 1):
            w = self.wins.get(k)
            if w is None:
                return False
            base = k * self.W
            if min(hi, base + self.W) > base + w[1]:               # a short (last) window: bytes past its `n` are not held
                return False
        return True

    def __getitem__(self, sl: slice) -> Any:
        lo, hi = int(sl.start or 0), int(min(sl.stop if sl.stop is not None else self.n, self.n))
        if hi <= lo:
            return torch.empty(0, dtype=torch.uint8)
        parts = []
        for k in range(lo // self.W, (hi - 1) // self.W + 1):
            w = self.wins.get(k)
            if w is None:
                raise KeyError(f"host window {k} (bytes [{k * self.W}, {(k + 1) * self.W})) was not retained (retain=sample)")
            hv, n = w
            base = k * self.W
            a, b = max(lo, base) - base, min(hi, base + n) - base
            if b > a:
                parts.append(hv[a:b])
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    def cpu(self) -> "_HostWindows":
        return self


class _HostRanges:
    """The raw bytes of one step retained on host as pinned BYTE RANGES of the step stream: bounded retain=host (each window as it
    lands, or carved out of a learning step's transient copies) and a replay's challenged chunks (bounded retain=sample by chunk).
    `ranges` = [(lo, hi, hv)] disjoint, sorted, hv a pinned uint8 view of exactly hi - lo bytes; a slice may span consecutive ranges
    (a window boundary splits one retained region in two) but never a gap: a gap is a KeyError, never zeros.  Same duck type as
    `_HostWindows` / the contiguous `keep` tensor."""
    is_cuda = False

    def __init__(self, ranges: list[tuple[int, int, Any]], used: int, chunk: int = GPU_CHUNK_BYTES) -> None:
        self.ranges = sorted(((int(lo), int(hi), hv) for lo, hi, hv in ranges if hi > lo), key=lambda r: r[0])
        self.used = int(used)
        self.n = -(-self.used // chunk) * chunk
        self._lows = [r[0] for r in self.ranges]

    def numel(self) -> int:
        return self.n

    @property
    def retained_bytes(self) -> int:
        return sum(hi - lo for lo, hi, _hv in self.ranges)

    def _cover(self, lo: int, hi: int) -> list[tuple[int, int, Any]] | None:
        """The consecutive retained ranges covering [lo, hi) with no gap, or None."""
        import bisect
        i = bisect.bisect_right(self._lows, lo) - 1
        if i < 0:
            return None
        out, pos = [], lo
        while pos < hi:
            if i >= len(self.ranges):
                return None
            rlo, rhi, hv = self.ranges[i]
            if rlo > pos or rhi <= pos:
                return None
            out.append((rlo, rhi, hv))
            pos = rhi
            i += 1
        return out

    def holds(self, lo: int, hi: int) -> bool:
        lo, hi = int(lo), int(min(hi, self.n))
        return hi <= lo or self._cover(lo, hi) is not None

    def __getitem__(self, sl: slice) -> Any:
        lo, hi = int(sl.start or 0), int(min(sl.stop if sl.stop is not None else self.n, self.n))
        if hi <= lo:
            return torch.empty(0, dtype=torch.uint8)
        cov = self._cover(lo, hi)
        if cov is None:
            raise KeyError(f"bytes [{lo}, {hi}) of the step stream were not retained on host: open by replay")
        parts = [hv[max(lo, rlo) - rlo:min(hi, rhi) - rlo] for rlo, rhi, hv in cov]
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    def cpu(self) -> "_HostRanges":
        return self
