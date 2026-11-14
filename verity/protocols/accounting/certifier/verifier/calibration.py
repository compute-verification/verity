"""The honest prover's clock syncs in hindsight (`choose_syncs`): each window's sync from Σ_sync, chosen from all of
the window's frames on every egress link, which the live provers' choices are held against."""

from __future__ import annotations

from collections.abc import Sequence

from verity.protocols.accounting.certifier.verifier.schedule import ClockSync, LogicalFrame, SyncSet, eligible


def choose_syncs(links: Sequence[tuple[Sequence[LogicalFrame], Sequence[int]]], sync_set: SyncSet,
                 window_buckets: int) -> tuple[list[ClockSync], list[list[int]], int]:
    """Replay the honest prover's choice for one configuration, window by window. `links` holds each egress link's
    frames and their ready buckets. Under one clock per spatial unit every egress link reads the window's sync, so the
    prover declares one value per window: the value met on every link with the least total hand time over the window's
    frames on all links, which is their total shaping delay; ties go to the smaller δ, then the larger ρ. (Taking the
    smallest δ first would pick a slow ρ to save one bucket, and hold every later frame for seconds.) A window that no
    value meets on every link is a miss, which is a violation; the replay continues with the most lenient value. Returns
    the chosen syncs, each link's actual hand times (a late frame goes when ready, delaying those behind it) and the
    number of missed windows.

    Each link's frames are in the Program's canonical order. For a batching engine that order is step-major, so a long
    session anchored in one window interleaves with sessions anchored in later ones, and anchors need not be monotone.
    On each link, each window's first frame must come after the first frame of every earlier window (requests are
    admitted in order). While window w is chosen, later windows take their most eager value (least δ, largest ρ). That
    only lowers hand times, so a window found met stays met once later windows are fixed. A larger δ or a smaller ρ
    only raises hand times, so a value met on every link exists exactly when the most lenient one is met on each."""
    wins = [[f.anchor // window_buckets for f in frames] for frames, _ in links]
    firsts: list[dict[int, int]] = []
    lasts: list[dict[int, int]] = []
    for win in wins:
        first: dict[int, int] = {}
        last: dict[int, int] = {}
        for k, w in enumerate(win):
            first.setdefault(w, k)
            last[w] = k
        if any(first[a] > first[b] for a, b in zip(sorted(first), sorted(first)[1:])):
            raise ValueError("each window's first frame must follow the first frame of every earlier window")
        firsts.append(first)
        lasts.append(last)
    if not any(wins):
        return [], [[] for _ in links], 0
    windows = max(max(win) for win in wins if win) + 1
    deltas, rhos = sorted(set(sync_set.deltas)), sorted(set(sync_set.rhos))
    eager, lenient = ClockSync(deltas[0], rhos[-1]), ClockSync(deltas[-1], rhos[0])
    syncs = [eager] * windows
    misses, h_prefix, k0 = 0, [0] * len(links), [0] * len(links)

    def cost(w: int, sync: ClockSync) -> int | None:
        """The window's total hand time on every link under `sync`, or None if a frame of the window is late."""
        syncs[w] = sync
        total = 0
        for i, (frames, ready) in enumerate(links):
            if w not in firsts[i]:
                continue
            h, win = h_prefix[i], wins[i]
            for k in range(firsts[i][w], lasts[i][w] + 1):
                h = max(eligible(frames[k], syncs[win[k]]), h)
                if win[k] == w:
                    if ready[k] > h:
                        return None
                    total += h
        return total

    for w in range(windows):
        if not any(w in first for first in firsts):
            continue
        for i, (frames, _) in enumerate(links):
            if w in firsts[i]:
                for k in range(k0[i], firsts[i][w]):
                    h_prefix[i] = max(eligible(frames[k], syncs[wins[i][k]]), h_prefix[i])
                k0[i] = firsts[i][w]
        best: tuple[int, int, int, ClockSync] | None = None
        for i_rho, rho in enumerate(rhos):
            # a larger δ only raises hand times, so the least met δ for this ρ is found by bisection
            if (c := cost(w, ClockSync(deltas[-1], rho))) is None:
                continue
            lo, hi, c_hi = 0, len(deltas) - 1, c
            while lo < hi:
                mid = (lo + hi) // 2
                if (c := cost(w, ClockSync(deltas[mid], rho))) is None:
                    lo = mid + 1
                else:
                    hi, c_hi = mid, c
            key = (c_hi, lo, -i_rho, ClockSync(deltas[lo], rho))
            if best is None or key[:3] < best[:3]:
                best = key
        if best is None:
            misses += 1
            syncs[w] = lenient
        else:
            syncs[w] = best[3]
    hands = []
    for (frames, ready), win in zip(links, wins):
        link_hands, h, actual = [], 0, 0
        for k, f in enumerate(frames):
            h = max(eligible(f, syncs[win[k]]), h)
            actual = max(h, ready[k], actual)
            link_hands.append(actual)
        hands.append(link_hands)
    return syncs, hands, misses
