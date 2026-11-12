"""CIR-254: changes tried for their effect on staging memory, each set on the modules for one process (`stage_profile.py
--fix A,B`), none an edit of the tree.

    commit-sum     `DsaCountSum_v1`'s builder commits the sum it returns (`forms.Circuit.commit`: one row a bit that is not
                   already one committed bit), so a count going up the tree is W + 1 columns, not an XOR of its whole subtree
    commit-split   `DsaCountSplit_v1`'s builder commits the two counts it returns
    commit-tag     `DsaButterflyTag_v1`'s builder commits the tag it returns
    last-use       no change to the circuit: ``wires`` records each word's last read, and the record gives what the lowering
                   would hold at most if it dropped a word after its last reader (``rec["last_use"]``)
    free-words     no change to the circuit (the pin must be the one without): `ir_lower.lower_gates` drops a gate's word from
                   ``wires`` once the last gate that reads it is lowered.  The last reader of every gate is read off the gates
                   in one pass before the lowering (`_last_readers`); the words the caller reads after (the call's returns)
                   stay.  What the change to `lower_gates` itself would be: that pass, and a `del` after each step.
    ctl-piece      no change to the circuit (the pin must be the one without): `DsaButterflyCtl_v1` gets the builder its
                   neighbours have (`_piece`: ``d.build = own_gates(d)``), so the lowering keeps a word for what a switch
                   returns and none for each Xor and Select inside it

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


def _last_readers(gates):
    """For every gate of the circuit, the index of the last gate of ``gates`` that reads it (-1: none does), as a lowering by
    `ir_lower.lower_gates` reads: a gate reads its operands, and a builder-lowered activation reads every leaf of its
    parameters when it is lowered, whether or not a gate of its own does."""
    import numpy as np
    from verity_flock import ir_lower as IL

    circ = gates.circuit
    last = np.full(circ.size, -1, dtype=np.int64)
    seen: set[tuple[int, int]] = set()
    n = 0
    for g in circ.gates(gates.lo, gates.hi):
        n += 1
        i = g.index
        for o in g.operands:
            last[o] = i
        act = IL._boolean_activation(g)
        if act is not None:
            sc = act[0]
            key = (sc.offset, id(sc.spec))
            if key not in seen:
                seen.add(key)
                end = sc.offset + sc.spec.gates - 1
                for pi, (_, t) in enumerate(sc.spec.signature()[0]):
                    for j in range(t.leaves):
                        o = IL.param_leaf(sc, pi, j)
                        if last[o] < end:
                            last[o] = end
    return last, n, len(seen)


class _Freeing(dict):
    """``wires`` that drops a word once no gate still to be lowered reads it.  A word set at gate j says every gate below
    the word set before it is lowered (gates are lowered in order, an activation's returns set after its builder has read
    its parameters), so a word whose last reader is below that one is dropped then; ``keep`` never are."""

    def __init__(self, base: dict, last, keep: set):
        import heapq

        super().__init__(base)
        self.last, self.keep, self.push, self.pop = last, keep, heapq.heappush, heapq.heappop
        self.heap = sorted((int(last[k]), k) for k in base)
        self.prev = -1
        self.dropped = self.most = 0

    def _made(self, j: int) -> None:
        self.push(self.heap, (int(self.last[j]), j))
        h, prev, keep = self.heap, self.prev, self.keep
        while h and h[0][0] < prev:
            o = self.pop(h)[1]
            if o not in keep and dict.pop(self, o, None) is not None:
                self.dropped += 1
        self.prev = j
        if len(self) > self.most:
            self.most = len(self)

    def __setitem__(self, j, word):
        new = j not in self
        dict.__setitem__(self, j, word)
        if new:
            self._made(j)

    def setdefault(self, j, word):
        if j in self:
            return dict.__getitem__(self, j)
        dict.__setitem__(self, j, word)
        self._made(j)
        return word


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
        elif name == "ctl-piece":
            BM.DsaButterflyCtl.build = BM.own_gates(BM.DsaButterflyCtl)
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
        elif name == "free-words":
            inner_f = IL.lower_gates
            depth_f = [0]

            def lower_gates_freeing(C, gates, wires, *a, **kw):
                if depth_f[0] or "free_words" in rec or not hasattr(gates, "returns"):
                    return inner_f(C, gates, wires, *a, **kw)
                import time

                t = time.time()
                last, n, acts = _last_readers(gates)
                rec["free_words"] = {"gates": n, "activations": acts, "last_readers_seconds": round(time.time() - t, 1),
                                     "last_readers_bytes": int(last.nbytes)}
                keep = set(gates.returns) | set(kw.get("reads", ()) or (a[1] if len(a) > 1 else ()))
                fw = _Freeing(wires, last, keep)
                depth_f[0] += 1
                try:
                    inner_f(C, gates, fw, *a, **kw)
                finally:
                    depth_f[0] -= 1
                rec["free_words"].update({"words_set": fw.dropped + len(fw), "dropped": fw.dropped, "most_at_once": fw.most,
                                          "left": len(fw)})
                wires.clear()
                wires.update(fw)
                return wires

            IL.lower_gates = lower_gates_freeing
        else:
            raise ValueError(f"--fix {name}?")
