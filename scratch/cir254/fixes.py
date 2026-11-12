"""CIR-254: changes tried for their effect on staging memory, each set on the modules for one process (`stage_profile.py
--fix A,B`), none an edit of the tree.

    commit-sum     `DsaCountSum_v1`'s builder commits the sum it returns (`forms.Circuit.commit`: one row a bit that is not
                   already one committed bit), so a count going up the tree is W + 1 columns, not an XOR of its whole subtree
    commit-split   `DsaCountSplit_v1`'s builder commits the two counts it returns
    commit-tag     `DsaButterflyTag_v1`'s builder commits the tag it returns
    last-use       no change to the circuit: ``wires`` records each word's last read, and the record gives what the lowering
                   would hold at most if it dropped a word after its last reader (``rec["last_use"]``)

A commit is a row of the lowered unit and no gate of the Definition (`trace.TracingCircuit.commit` returns the word), so the
Definition, its description and its id stay as they are, and the unit's circuit and pin change.
"""
from __future__ import annotations

import sys


def _committing(build):
    def run(C, *a):
        return [C.commit(list(w)) for w in build(C, *a)]
    return run


class _Recording(dict):
    """``wires`` with each key's last read: the count of words there were when it was last read."""

    def __init__(self, *a):
        super().__init__(*a)
        self.last: dict[int, int] = {}

    def __getitem__(self, k):
        self.last[k] = len(self)
        return dict.__getitem__(self, k)


def _frontier(wires: _Recording, keep: set) -> dict:
    """The most bytes of sets (and their entries) alive at once when a word is dropped after its last read (a word nothing
    read is dropped as it is made; ``keep`` stay to the end): each set counted once, alive from its first word to the last
    read of any word that holds it."""
    n = len(wires)
    first: dict[int, int] = {}
    until: dict[int, int] = {}
    size: dict[int, tuple[int, int]] = {}
    for i, (k, word) in enumerate(wires.items()):
        end = n if k in keep else wires.last.get(k, i)
        for f in word:
            s = id(f.s)
            if s not in first:
                first[s] = i
                size[s] = (sys.getsizeof(f.s) + 48, len(f.s))
                until[s] = end
            elif end > until[s]:
                until[s] = end
    delta_b = [0] * (n + 2)
    delta_e = [0] * (n + 2)
    total_b = total_e = 0
    for s, i in first.items():
        b, e = size[s]
        total_b += b
        total_e += e
        delta_b[i] += b
        delta_e[i] += e
        delta_b[until[s] + 1] -= b
        delta_e[until[s] + 1] -= e
    peak_b = peak_e = cur_b = cur_e = at = 0
    for i in range(n + 1):
        cur_b += delta_b[i]
        cur_e += delta_e[i]
        if cur_b > peak_b:
            peak_b, peak_e, at = cur_b, cur_e, i
    return {"words": n, "sets": len(first), "all_bytes": total_b, "all_entries": total_e, "peak_bytes": peak_b,
            "peak_entries": peak_e, "peak_at_word": at, "never_read": sum(1 for k in wires if k not in wires.last and k not in keep)}


def apply(names: str, rec: dict) -> None:
    from verity.definitions.registry import boolean_dsa_topk_mask as BM
    from verity_flock import ir_lower as IL

    for name in names.split(","):
        if name == "commit-sum":
            BM.DsaCountSum.build = _committing(BM.DsaCountSum.build)
        elif name == "commit-split":
            BM.DsaCountSplit.build = _committing(BM.DsaCountSplit.build)
        elif name == "commit-tag":
            BM.DsaButterflyTag.build = _committing(BM.DsaButterflyTag.build)
        elif name == "last-use":
            inner = IL.lower_gates
            depth = [0]

            def lower_gates(C, gates, wires, *a, **kw):
                if depth[0] or "last_use" in rec or not hasattr(gates, "returns"):
                    return inner(C, gates, wires, *a, **kw)
                depth[0] += 1
                try:
                    rw = _Recording(wires)
                    inner(C, gates, rw, *a, **kw)
                finally:
                    depth[0] -= 1
                rec["last_use"] = _frontier(rw, set(gates.returns))
                wires.update(rw)
                return wires

            IL.lower_gates = lower_gates
        else:
            raise ValueError(f"--fix {name}?")
