"""What the grade and the library read of one traced step, in Boolean gates (training_slowdown.gates) and bits. No
torch: a `StepCircuit` saved with `StepCircuit.save` is measured again with
``python -m training_slowdown.capture.circuit F.npz``.

`StepCircuit` is training_slowdown.capture.structure's dataflow of a program recorded from its start, restricted to its measured
step R (`Capture.region`) and the values R reads from earlier steps that depend on their token data (`carried`: θ' and
the run's other persistent state). Each value has its predecessors, phase, whether it is token-indexed (`tok`), whether
it depends on an earlier step's token data (`dep`) or on step R's own (`tokdep`), its bits, and the gates of the op that
made it (each counted op at its circuit Definition's price, `gates.per_mac`, split over the op's values by elements).
An op is a token op when it reads or makes a token-indexed value.

`measure` gives, per step:

- `gates`, |C_R|: the step's gates;
- `affected_share` = |A ∩ ∪T| / |C_R|, the token ops that depend on θ';
- τ: the token inputs' positions the step reads (its targets included), per consumed token (`tau_ids`, ids, priced at
  ⌈log2 V⌉ by the caller), and `tau_other_bits`: any other input a token op reads, and 64 bits (a seed) per random op
  one reads, over the consumed tokens. The consumed tokens are the union of the positions the token-index ops read in
  each token input; a step whose token-index ops read no token input (a program-made source) counts every
  token-index read.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

FORWARD, BACKWARD, OTHER = 0, 1, 2
COMPUTED, TOKEN_INPUT, OTHER_INPUT, CARRIED, RANDOM, ATTR = range(6)
SEED_BITS = 64
#: the circuit's fields saved as JSON beside its arrays
JSON_FIELDS = ("read_backs", "products")


@dataclass
class StepCircuit:
    indptr: np.ndarray
    indices: np.ndarray
    phase: np.ndarray
    tok: np.ndarray
    dep: np.ndarray
    tokdep: np.ndarray
    kind: np.ndarray
    #: whether a value reads one outside the circuit (an earlier step's, or θ): a carried value that does is a source
    outside: np.ndarray
    read_next: np.ndarray
    bits: np.ndarray
    width: np.ndarray
    gates: np.ndarray
    node: np.ndarray
    #: token reads: the reading op, its storage key (an index into `keys`), whether it is a token-index op, positions
    tr_op: np.ndarray
    tr_key: np.ndarray
    tr_index: np.ndarray
    tr_indptr: np.ndarray
    tr_pos: np.ndarray
    keys: np.ndarray
    token_key: np.ndarray
    region: int
    step_tokens: int
    #: the recording's read-backs that decide what the graph can't carry (`data_dependent` and `carried_scalars`); None
    #: for a circuit no recording listed them for
    read_backs: dict | None = None
    #: the step's matrix products (training_slowdown.capture.products' `rows`); None for a circuit recorded before
    #: training_slowdown 4.8
    products: dict | None = None

    def save(self, path: Path) -> None:
        arrays = {f.name: getattr(self, f.name) for f in fields(self)
                  if f.name not in JSON_FIELDS and getattr(self, f.name) is not None}
        for name in JSON_FIELDS:
            if getattr(self, name) is not None:
                arrays[name] = np.array(json.dumps(getattr(self, name)))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "StepCircuit":
        z = np.load(path, allow_pickle=False)
        out = {f.name: (int(z[f.name]) if f.name in ("region", "step_tokens") else z[f.name])
               for f in fields(cls) if f.name not in JSON_FIELDS and f.name in z.files}
        return cls(**out, **{name: json.loads(str(z[name])) if name in z.files else None for name in JSON_FIELDS})

    def __len__(self) -> int:
        return len(self.phase)

    def preds(self) -> list[list[int]]:
        ip, ix = self.indptr.tolist(), self.indices.tolist()
        return [ix[ip[v]:ip[v + 1]] for v in range(len(self))]

    def positions(self, i: int) -> np.ndarray:
        return self.tr_pos[self.tr_indptr[i]:self.tr_indptr[i + 1]]


# ---- cuts ---------------------------------------------------------------------------------------------------------------

class Net:
    """A flow network for Dinic's algorithm: edge e and its reverse e ^ 1."""

    def __init__(self, n: int) -> None:
        self.n, self.head = n, [-1] * n
        self.to: list[int] = []
        self.cap: list[int] = []
        self.nxt: list[int] = []

    def add(self, u: int, v: int, c: int) -> None:
        self.to += [v, u]
        self.cap += [c, 0]
        self.nxt += [self.head[u], self.head[v]]
        self.head[u], self.head[v] = len(self.to) - 2, len(self.to) - 1

    def _levels(self, s: int) -> list[int]:
        level = [-1] * self.n
        level[s] = 0
        q = [s]
        for u in q:
            e = self.head[u]
            while e != -1:
                w = self.to[e]
                if self.cap[e] > 0 and level[w] < 0:
                    level[w] = level[u] + 1
                    q.append(w)
                e = self.nxt[e]
        return level

    def max_flow(self, s: int, t: int) -> tuple[int, list[bool]]:
        """The flow's value and the nodes the residual network reaches from `s` (the min cut's source side)."""
        total = 0
        while True:
            level = self._levels(s)
            if level[t] < 0:
                return total, [x >= 0 for x in level]
            it = list(self.head)
            while True:
                stack, path = [s], []
                pushed = 0
                while stack:
                    u = stack[-1]
                    if u == t:
                        pushed = min(self.cap[e] for e in path)
                        for e in path:
                            self.cap[e] -= pushed
                            self.cap[e ^ 1] += pushed
                        break
                    e = it[u]
                    while e != -1 and not (self.cap[e] > 0 and level[self.to[e]] == level[u] + 1):
                        e = self.nxt[e]
                    it[u] = e
                    if e == -1:
                        level[u] = -1
                        stack.pop()
                        if path:
                            path.pop()
                            it[stack[-1]] = self.nxt[it[stack[-1]]]
                        continue
                    stack.append(self.to[e])
                    path.append(e)
                if not pushed:
                    break
                total += pushed


# ---- the quantities -----------------------------------------------------------------------------------------------------

def _union_size(c: StepCircuit, reads) -> int:
    by: dict[int, list[np.ndarray]] = {}
    for i in reads:
        by.setdefault(int(c.tr_key[i]), []).append(c.positions(i))
    return sum(len(np.unique(np.concatenate(ps))) for ps in by.values())


def measure(c: StepCircuit) -> dict:
    """The step's quantities (module docstring)."""
    n = len(c)
    preds = c.preds()
    tok, dep = c.tok.astype(bool), c.dep.astype(bool)
    node = c.node.astype(np.int64)
    ops = int(node.max()) + 1 if n and node.max() >= 0 else 0
    src = np.repeat(np.arange(n), np.diff(c.indptr))
    pred = c.indices.astype(np.int64)

    def any_pred(flag: np.ndarray) -> np.ndarray:
        return np.bincount(src, weights=flag[pred].astype(float), minlength=n) > 0 if len(pred) else np.zeros(n, bool)

    def per_op(flag: np.ndarray) -> np.ndarray:
        on = node >= 0
        return np.bincount(node[on], weights=flag[on].astype(float), minlength=ops) > 0

    op_gates = np.bincount(node[node >= 0], weights=c.gates[node >= 0], minlength=ops)
    op_tok = per_op(tok | any_pred(tok))
    op_dep = per_op(dep)
    total = float(op_gates.sum())

    index = [i for i in range(len(c.tr_op)) if c.tr_index[i]]
    index = [i for i in index if c.token_key[c.tr_key[i]]] or index
    consumed = _union_size(c, index)
    ids = _union_size(c, [i for i in range(len(c.tr_op)) if c.token_key[c.tr_key[i]]])
    read_by_tok = set()
    for v in np.nonzero((node >= 0) & op_tok[np.maximum(node, 0)])[0].tolist():
        read_by_tok.update(preds[v])
    other_bits = sum(int(c.bits[p]) for p in read_by_tok if c.kind[p] == OTHER_INPUT)
    random_ops = {int(node[v]) for v in np.nonzero(c.kind == RANDOM)[0].tolist()}
    seeds = len({o for o in random_ops if op_tok[o]} | {int(node[p]) for p in read_by_tok if c.kind[p] == RANDOM})
    per = max(consumed, 1)
    return {"gates": total, "affected_share": float(op_gates[op_tok & op_dep].sum()) / total if total else 0.0,
            "tau_ids": ids / per, "tau_other_bits": (other_bits + SEED_BITS * seeds) / per}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("circuit", type=Path, help="a StepCircuit saved by StepCircuit.save")
    a = ap.parse_args(argv)
    print(json.dumps(measure(StepCircuit.load(a.circuit)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
