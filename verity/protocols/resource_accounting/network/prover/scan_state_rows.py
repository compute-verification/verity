"""The scan state's rows a window reads: the network attester's scan of its ingress rows, from the empty state at
bucket 0, as a window of ``buckets`` buckets from scan bucket t0 reads it (`ingress_scan.window_state`), with the
request each of the window's slots starts.

A window here is any object with ``scan.shape`` (an `ingress_scan.Class`'s statics by name), ``buckets``, ``slots``, and ``anchor(k)`` and
``lane(k)``, slot k's bucket and lane."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from verity.protocols.resource_accounting.network.prover import ingress_scan as IS


def _scan(scan: IS.Class, ingress: Sequence[bytes], outputs: Mapping[int, Sequence[int]]) -> list[IS.State]:
    got = IS.scan(scan, list(ingress), {int(q): tuple(v) for q, v in outputs.items()})
    if got is None:
        raise ValueError(f"the scan at {scan} refuses the network attester's ingress rows to bucket {len(ingress) - 1}: "
                         f"a request it can't hold (more than {scan.KI} prompt ids or {scan.KO} outputs) or rows that "
                         "aren't the network attester's")
    return got[0]


def scan_anchors(cls: Any, ingress: Sequence[bytes], outputs: Mapping[int, Sequence[int]]) -> dict[int, int]:
    """Each request's anchor in the scan of ``ingress`` by the window ``cls``, as `scan_state` scans it: the bucket whose
    Programs start it (its slot's ``occ`` bit 0 set in the state before it), by its ordinal q. ValueError where the scan
    refuses the rows."""
    out: dict[int, int] = {}
    for t, st in enumerate(_scan(IS.Class(**cls.scan.shape), ingress, outputs)):
        for s in st.slots:
            if s.occ & 1:
                out.setdefault(s.q, t + 1)
    return out


def scan_state(cls: Any, ingress: Sequence[bytes], outputs: Mapping[int, Sequence[int]],
               t0: int = 0) -> tuple[list[list[int]], list[int | None]]:
    """The scan as the window ``cls`` from scan bucket ``t0`` reads it: its state rows (`ingress_scan.window_state`'s)
    and the request each slot's Program starts, slot k's the ordinal q of the request whose Program starts at its anchor
    in its lane (``occ``'s bit 0 set in the row there), None for a slot no request starts. ``ingress`` is the network
    attester's registered ingress rows (a row a bucket, `network_attester_rows.window_rows`) to the window's last row,
    each request's Program giving its output ids ``outputs[q]``. ValueError where the scan refuses the rows."""
    scan, n = IS.Class(**cls.scan.shape), t0 + cls.buckets - 1
    if len(ingress) < n:
        raise ValueError(f"the window from scan bucket {t0} reads the scan's states to bucket {n - 1}: give the ingress "
                         f"rows to it, {n}, not {len(ingress)}")
    states = _scan(scan, ingress[:n], outputs)
    rows = IS.window_state(scan, [IS.encode(scan, s) for s in states], t0, cls.buckets)
    at = {a: states[t0 + a - 1] if t0 + a else IS.empty(scan) for a in {cls.anchor(k) for k in range(cls.slots)}}
    starts = []
    for k in range(cls.slots):
        s = at[cls.anchor(k)].slots[cls.lane(k)]
        starts.append(s.q if s.occ & 1 else None)
    return rows, starts


def standin_rows(cls: Any, slots: Sequence[tuple[Sequence[int], int]]) -> list[list[int]]:
    """The state rows where no scan ran that give slot k its row (occupancy, prompt length, ids) and ``max_tokens``,
    ``slots[k]``, at its anchor and lane, every other field zero."""
    scan = IS.Class(**cls.scan.shape)
    states = [IS.empty(scan) for _ in range(cls.buckets)]
    for k, (row, max_tokens) in enumerate(slots):
        s = states[cls.anchor(k)].slots[cls.lane(k)]
        s.occ, s.fields, s.max_tokens = int(row[0]), int(row[1]), int(max_tokens)
        for p, v in enumerate(row[2:]):
            s.ids[IS.position(scan, p)] = int(v)
    return [IS.encode(scan, st) for st in states]


__all__ = ["scan_anchors", "scan_state", "standin_rows"]
