"""What a windowed step keeps on host for openings (`native_collect`'s windowed staging): its pinned windows (`_HostWindows`) or, with
value families excluded (`--retain-exclude`), byte ranges of its stream (`_HostRanges`), and the families and chunk ranges excluded."""

from __future__ import annotations

from typing import Any

import torch

from verity_vllm.commit.committer.native_host import GPU_CHUNK_BYTES
from verity_vllm.commit.committer.plan_cache import PlanMetas

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
    """The raw bytes of one step retained on host as pinned BYTE RANGES of the step stream (retain=host with
    `VERITY_RETAIN_EXCLUDE`): the chunk-aligned interior of every tensor whose value family is excluded (the FA3 hidden M1 plane at
    3,545 tokens: 93 % of the prefill step) is NOT copied off the staging window -- those leaves open by REPLAY against the original
    root; every other chunk (module outputs, the boundary chunks an excluded tensor shares with its neighbours) is kept and
    opens from host as before.  `ranges` = [(lo, hi, hv)] disjoint, sorted, hv a pinned uint8 view of exactly hi - lo bytes; a slice
    may span consecutive ranges (a window boundary splits one retained region in two) but never a gap: a gap is a KeyError, never
    zeros.  Same duck type as `_HostWindows` / the contiguous `keep` tensor."""
    is_cuda = False

    def __init__(self, ranges: list[tuple[int, int, Any]], used: int, excluded: dict[str, int] | None = None, chunk: int = GPU_CHUNK_BYTES) -> None:
        self.ranges = sorted(((int(lo), int(hi), hv) for lo, hi, hv in ranges if hi > lo), key=lambda r: r[0])
        self.used = int(used)
        self.n = -(-self.used // chunk) * chunk
        self.excluded = dict(excluded or {})                     # family -> bytes not retained (for the record)
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
            fams = ",".join(sorted(self.excluded)) or "-"
            raise KeyError(f"bytes [{lo}, {hi}) of the step stream were not retained on host (retain=host excluding {fams}): open by replay")
        parts = [hv[max(lo, rlo) - rlo:min(hi, rhi) - rlo] for rlo, rhi, hv in cov]
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    def cpu(self) -> "_HostRanges":
        return self


#: The RETAIN-windowed pool: the value families NO C2 consumer reads back from host: the attention
#: launch's M1 export stream (FA2 on Ada/Ampere, FA3 on Hopper).  oracle_compare classifies `hidden_stream` out before any read
#: (`no oracle family`), sampled_replay has no VU row for it (`fa2_hidden_m1_stream: attention interior, no VU row of its own`), the
#: boundary linkage reads ids only; its leaves are COMMITTED (coverage + openings) and open by REPLAY against the original root
#: (openings_after_release).  It is 84 % of the raw stream at 1024/128 (planner, two workloads: hidden 100,126 of 119,879 MiB and 122,740
#: of 146,955 MiB), so `--retain-exclude hidden_m1` (= both families, whichever tap the pod has) keeps the C2 byte source whole and the
#: pinned pool at the 16 % the C2 forms read -- byte-identical oracle / sampled-replay / linkage forms by construction.
HIDDEN_M1_FAMILIES = frozenset({"fa2_hidden_m1", "fa3_hidden_m1"})
RETAIN_EXCLUDE_ALIASES = {"hidden_m1": HIDDEN_M1_FAMILIES, "hidden": HIDDEN_M1_FAMILIES}


def retain_exclude_families(spec: str | None) -> set[str]:
    """`VERITY_RETAIN_EXCLUDE` / `--retain-exclude` -> the set of value families (native_host.FAMILY_RULES names) whose chunk-aligned
    interior is not copied to host: comma-separated family names (`_stream` suffix tolerated) and the aliases `hidden_m1` / `hidden`."""
    out: set[str] = set()
    for f in (spec or "").split(","):
        f = f.strip().removesuffix("_stream")
        if not f:
            continue
        out |= set(RETAIN_EXCLUDE_ALIASES.get(f, {f}))
    return out


def _excluded_chunk_ranges(metas: list, exclude: set[str], chunk: int = GPU_CHUNK_BYTES) -> list[tuple[int, int, str]]:
    """The chunk-aligned INTERIOR [lo, hi) of every tensor of an excluded value family in step-stream order (merged when adjacent): the
    bytes retain=host + exclude does not copy off the staging window.  Boundary chunks shared with a neighbouring tensor are retained
    (the neighbour opens from host); a tensor shorter than one chunk contributes nothing."""
    from verity_vllm.commit.committer.native_host import value_family
    out: list[tuple[int, int, str]] = []
    spans = zip(metas.names, metas.offsets, metas.sizes) if isinstance(metas, PlanMetas) else (   # name, byte offset in the step stream, nbytes
        (m.name, int(m.dev_off) if int(getattr(m, "dev_off", -1)) >= 0 else int(m.stream_off), int(m.nbytes)) for m in metas)
    for name, off, nbytes in spans:
        fam = value_family(name)
        if fam not in exclude:
            continue
        lo = -(-off // chunk) * chunk
        hi = ((off + nbytes) // chunk) * chunk
        if hi <= lo:
            continue
        if out and out[-1][1] == lo and out[-1][2] == fam:
            out[-1] = (out[-1][0], hi, fam)
        else:
            out.append((lo, hi, fam))
    return out


def outside(excl: list[tuple[int, int, str]], lo: int, hi: int) -> list[tuple[int, int]]:
    """[lo, hi) minus the excluded chunk ranges `excl` (`_excluded_chunk_ranges`; absolute step-stream offsets): the bytes kept."""
    out, pos = [], lo
    for elo, ehi, _fam in excl:
        if ehi <= pos:
            continue
        if elo >= hi:
            break
        if elo > pos:
            out.append((pos, min(elo, hi)))
        pos = max(pos, ehi)
        if pos >= hi:
            break
    if pos < hi:
        out.append((pos, hi))
    return out
