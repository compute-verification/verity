"""How `Prog` canonicalises a read of a request's peers Input: the received-collective partials of a TP rank.

The derived rank Program lowers one Input per `peers_<k>` (one field at offset 0); the fold lowers ONE homogeneous peers Record per
request, its fields at increasing offsets (`Prog.peer_fields[node]` = sorted [(leaf offset, k)], set by the loaders).  A read
canonicalises as ("PEER", k, lo, hi), lo/hi relative to its field on both sides; a run over several fields is split at the field
boundaries first (`_split_peer`), so a WORLD > 2 fold compares with the rank Program.
"""
from __future__ import annotations


class PeerReads:
    """`Prog`'s reads of peers Input nodes (`instance_sequence.Prog`, and through it `batch_decomp.Projection`)."""

    def _peer_src(self, idx: int, lo: int, hi: int):
        """("PEER", k, lo', hi') for a read of leaf range [lo, hi) of peers Input node `idx`; None when the range is not inside one field."""
        import bisect
        flds = getattr(self, "peer_fields", {}).get(idx)
        if not flds:
            return None
        offs = [o for o, _ in flds]
        j = bisect.bisect_right(offs, lo) - 1
        if j < 0:
            return ("PEER?", idx, lo, hi)
        base, k = flds[j]
        end = flds[j + 1][0] if j + 1 < len(flds) else None
        if end is not None and hi > end:
            return ("PEER?", idx, lo, hi)      # canon splits a read at field boundaries first (`_split_peer`); unreachable from it
        return ("PEER", k, lo - base, hi - base)

    def _split_peer(self, run) -> list[tuple]:
        """A run reading several adjacent fields of one peers Input node as one run per field, in leaf order (the same leaves).  At
        WORLD > 2 the fold reads a collective's peers as ONE run over the request's homogeneous peers Record (fields at increasing
        offsets) where the rank Program reads one `peers_<k>` Input per peer: split, each part is one field's ("PEER", k, ..) on both
        sides, and the compare still checks the ordinals in order.  Any other run unchanged."""
        space, idx, lo, hi = run
        flds = getattr(self, "peer_fields", {}).get(idx) if space == "n" else None
        if not flds or len(flds) < 2:
            return [run]
        import bisect
        j = bisect.bisect_right(flds, (lo, float("inf"))) - 1
        if j < 0 or j + 1 >= len(flds) or hi <= flds[j + 1][0]:
            return [run]
        parts, a = [], lo
        while j + 1 < len(flds) and hi > flds[j + 1][0]:
            b = flds[j + 1][0]
            parts.append((space, idx, a, b))
            a, j = b, j + 1
        parts.append((space, idx, a, hi))
        return parts

    def _peer_srcP(self, p: tuple, unfold: bool) -> list[tuple]:
        """`_peer_src` over a progression of reads of one peers Input node: uniform ("PEER", k, (lo', hi'), (dL, dH), c)
        iff every run's [lo, hi) lies inside the SAME field (the whole span does: fields are disjoint and sorted), else the record's
        per-run answer (a run outside every field or spanning two is ("PEER?", ..) run by run)."""
        import bisect
        _space, (idx0, lo0, hi0), (_dI, dL, dH), c = p
        flds = getattr(self, "peer_fields", {}).get(idx0)
        if flds and c > 1:
            span_lo, span_hi = min(lo0, lo0 + (c - 1) * dL), max(hi0, hi0 + (c - 1) * dH)
            offs = [o for o, _ in flds]
            j = bisect.bisect_right(offs, span_lo) - 1
            if j >= 0:
                base, k = flds[j]
                end = flds[j + 1][0] if j + 1 < len(flds) else None
                if end is None or span_hi <= end:
                    return [("PEER", k, (lo0 - base, hi0 - base), (dL, dH), c)]
        return self._per_run(p, unfold)
