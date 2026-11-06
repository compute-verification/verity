"""The training formalism v2's quantities measured on one traced step, in Boolean gates (training_slowdown.gates) and
bits. No torch: a `StepCircuit` saved with `StepCircuit.save` is measured again with
``python -m training_slowdown.capture.circuit F.npz``.

`StepCircuit` is training_slowdown.capture.structure's dataflow of a program recorded from its start, restricted to its measured
step R (`Trace.region`) and the values R reads from earlier steps that depend on their token data (`carried`: θ' and
the run's other persistent state). Each value has its predecessors, phase, whether it is token-indexed (`tok`), whether
it depends on an earlier step's token data (`dep`) or on step R's own (`tokdep`), its bits, and the gates of the op that
made it (each counted op at its circuit Definition's price, `gates.per_mac`, split over the op's values by elements).
An op is a token op when it reads or makes a token-indexed value: its gates are its tokens' token blocks', one MAC of a
contraction over tokens (dW) to each token. A value stands for all its elements, so a share of one token's work is a
share of its ops' gates, and a cut's bits over the step's tokens are a token's.

`measure` gives, per step:

- σ = the token ops' share of the step's gates (forward and backward; the optimizer reads no token-indexed value);
  `affected_share` = |A ∩ ∪T| / |C_R|, the token ops that depend on θ'; `affected_all_share` = every affected op's
  (A, the optimizer's included).
- tokens: the step's consumed tokens, the union of the positions the token-index ops read in each token input, and
  in each sampled id the step feeds back (a generation round's decode). An index computed from token data otherwise (a
  sampled head's unique(y), a QAT code) is no token, though training_slowdown.flops counts its reads too; a step whose
  token-index ops read no token input (a program-made source) counts every token-index read.
  `token_factor` = tokens over the record's count for the step.
- τ: the token inputs' positions the step reads (its targets included), per consumed token (`tau_ids`, ids, priced at
  ⌈log2 V⌉ by the caller), and `tau_other_bits`: any other input a token op reads, and 64 bits (a seed) per random op
  one reads, over the consumed tokens.
- a: the narrowest cut across the forward's affected path (its affected token-indexed values), from where the token's
  data first meets θ' to the values the loss reduces, in bits per consumed token: an upper bound (a witness). `a_x`:
  the shares of one token's work done where the cut's values are made (its first and last).
- u_in: the update bits a block reads to do the first share x of one token's affected work, in program order
  (the forward, then the backward): at each op, the narrowest cut between the carried values and every update value
  (affected and free of the step's tokens: θ' as the step reads it, cast, merged or perturbed) the token's ops read so
  far; at most `max_cuts` cuts, each over a run of ops (an upper bound). One token is the first token-index read's
  batch: the ops its token-indexed values reach.
- t (`tail_share`): the share of that work in ops that read no update value at all, wherever they sit (attention's
  products, dW, elementwise products of activations); `frozen_tail_share`: only the contiguous ones between the
  forward's last update read and the backward's first (frozen layers above the highest update).
- Δ_grad = `grad_cut_bits` + `state_bits`: the narrowest cut between the step's token side and where its weight-shaped
  token-dependent values (gradients, the update) meet a carried value or reach step R + 1, plus the carried values the
  step's token ops don't read (optimizer moments, error feedback). The cut prices each value at the width the circuit
  passes it, its dtype's (a torch.bool plane 1 bit an entry, int8 and uint8 8, f16 16, a packed word its packing), so
  a message coded inside the step is measured at its coded width. Its encode, decode and sum are traced ops, counted in
  |C| by the rule every op is (training_slowdown.gates): their products at the elementwise price, an integer one at bf16's for
  want of a Definition, and their comparisons, casts, shifts, masks and sums at nothing. The values a reduction over
  the step's tokens makes (a dW product, a loss) are priced at the accumulator's width, 32 bits an element at least,
  since a block holding part of the tokens passes the accumulator's partial.
- `update_cut_bits`: the narrowest cut between the carried values and every update value any token op of the step
  reads: the trace's own encoding of θ' (compare 8 |E|).
- `partial_cut_bits`: the gradient's cut from one token's group alone (its own dW products): the partial a block
  holding the group's work passes, at its passed width, which the one bound's reduction sensitivity reads
  (training_slowdown.formalism).
- `data_dependent` and `carried_scalars`: the recording's read-backs that decide what the graph can't carry, which the
  circuit keeps (`read_backs`, saved with it); absent for a circuit no recording listed them for, which
  training_slowdown.formalism refuses as unchecked.
- `products`: the step's matrix products (training_slowdown.capture.products' rows), which the circuit keeps too; absent
  for a circuit recorded before training_slowdown 4.8.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

FORWARD, BACKWARD, OTHER = 0, 1, 2
COMPUTED, TOKEN_INPUT, OTHER_INPUT, CARRIED, RANDOM, ATTR = range(6)
SEED_BITS = 64
ACCUMULATOR_BITS = 32
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
    #: the recording's read-backs that decide what the graph can't carry (`data_dependent` and `carried_scalars`),
    #: which `measure` returns; None for a circuit no recording listed them for
    read_backs: dict | None = None
    #: the step's matrix products (training_slowdown.capture.products' `rows`), which `measure` returns; None for a
    #: circuit recorded before training_slowdown 4.8
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


def vertex_cut(preds: list[list[int]], cap: dict[int, int | None], sources, sinks) -> tuple[int | None, list[int]]:
    """The least capacity of a set of the values `cap` names (a value's capacity, None: not cuttable) meeting every path
    from `sources` to `sinks` along `preds` through them (sources and sinks may be cut), and one such set; None when
    every separating set holds an uncuttable value."""
    succ: dict[int, list[int]] = {v: [] for v in cap}
    for v in cap:
        for p in preds[v]:
            if p in succ:
                succ[p].append(v)
    fwd = set(s for s in sources if s in cap)
    stack = list(fwd)
    while stack:
        for w in succ[stack.pop()]:
            if w not in fwd:
                fwd.add(w)
                stack.append(w)
    back = set(t for t in sinks if t in fwd)
    stack = list(back)
    while stack:
        for p in preds[stack.pop()]:
            if p in fwd and p not in back:
                back.add(p)
                stack.append(p)
    keep = sorted(back)
    if not keep:
        return 0, []
    idx = {v: i for i, v in enumerate(keep)}
    inf = sum(c for v in keep if (c := cap[v]) is not None) + 1
    net = Net(2 * len(keep) + 2)
    S, T = 2 * len(keep), 2 * len(keep) + 1
    for i, v in enumerate(keep):
        net.add(2 * i, 2 * i + 1, inf if cap[v] is None else cap[v])
        for p in preds[v]:
            if p in idx:
                net.add(2 * idx[p] + 1, 2 * i, inf)
    for s in sources:
        if s in idx:
            net.add(S, 2 * idx[s], inf)
    for t in sinks:
        if t in idx:
            net.add(2 * idx[t] + 1, T, inf)
    total, seen = net.max_flow(S, T)
    if total >= inf:
        return None, []
    return total, [v for i, v in enumerate(keep) if seen[2 * i] and not seen[2 * i + 1]]


# ---- the quantities -----------------------------------------------------------------------------------------------------

def _union_size(c: StepCircuit, reads) -> int:
    by: dict[int, list[np.ndarray]] = {}
    for i in reads:
        by.setdefault(int(c.tr_key[i]), []).append(c.positions(i))
    return sum(len(np.unique(np.concatenate(ps))) for ps in by.values())


def _profile(cuts: list[tuple[float, int]]) -> list[list[float]]:
    """Points (x, bits) from (0, 0) to (1, u(1)), u stepping up to each cut's bits at its share."""
    pts, last = [[0.0, 0.0]], 0
    for x, bits in cuts:
        if bits > last:
            if x > pts[-1][0]:
                pts.append([x, float(last)])
            pts.append([x, float(bits)])
            last = bits
    if pts[-1][0] < 1.0:
        pts.append([1.0, float(last)])
    return pts


def _update(c: StepCircuit, preds: list[list[int]]) -> tuple[set[int], dict[int, int], list[int]]:
    """The update values (affected and free of the step's tokens), their bits, and the carried ones among them."""
    U = set(np.nonzero(c.dep.astype(bool) & ~c.tokdep.astype(bool))[0].tolist())
    return U, {v: int(c.bits[v]) for v in U}, [v for v in U if c.kind[v] == CARRIED and (c.outside[v] or not preds[v])]


def path(c: StepCircuit, preds: list[list[int]] | None = None) -> tuple[list[int], int]:
    """One token's traced path (module docstring): the ops that depend on θ' and make or read a value its group's
    token-indexed values reach, in program order, and the group's tokens (the first token-index read's batch)."""
    preds = c.preds() if preds is None else preds
    n, node = len(c), c.node.astype(np.int64)
    ops = int(node.max()) + 1 if n and node.max() >= 0 else 0
    index = [i for i in range(len(c.tr_op)) if c.tr_index[i]]
    index = [i for i in index if c.token_key[c.tr_key[i]]] or index
    if not index:
        return [], 0
    first = min(index, key=lambda i: int(c.tr_op[i]))
    key, fp = int(c.tr_key[first]), c.positions(first)
    seed_ops = {int(c.tr_op[i]) for i in range(len(c.tr_op)) if int(c.tr_key[i]) == key
                and len(np.intersect1d(c.positions(i), fp, assume_unique=True))}
    tok_l, node_l = c.tok.astype(bool).tolist(), node.tolist()
    inside = [False] * n
    for v in range(n):
        if tok_l[v] and (node_l[v] in seed_ops or any(inside[p] for p in preds[v])):
            inside[v] = True
    touched = np.array([inside[v] or any(inside[p] for p in preds[v]) for v in range(n)], dtype=bool)
    on = node >= 0
    op_group = np.bincount(node[on], weights=touched[on].astype(float), minlength=ops) > 0
    op_dep = np.bincount(node[on], weights=c.dep.astype(bool)[on].astype(float), minlength=ops) > 0
    return np.nonzero(op_group & op_dep)[0].tolist(), len(np.unique(fp))


def measure(c: StepCircuit, max_cuts: int = 128) -> dict:
    """The step's quantities (module docstring)."""
    n = len(c)
    preds = c.preds()
    tok, dep, tokdep = c.tok.astype(bool), c.dep.astype(bool), c.tokdep.astype(bool)
    upd = dep & ~tokdep
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
    reads_upd = any_pred(upd)
    op_reads_upd = per_op(reads_upd)
    op_phase = np.full(ops, OTHER)
    for v in np.nonzero(node >= 0)[0][::-1]:
        op_phase[node[v]] = c.phase[v]
    total = float(op_gates.sum())

    def share(mask) -> float:
        return float(op_gates[mask].sum()) / total if total else 0.0

    # tokens and τ
    index = [i for i in range(len(c.tr_op)) if c.tr_index[i]]
    index = [i for i in index if c.token_key[c.tr_key[i]]] or index
    consumed = _union_size(c, index)
    data = [i for i in range(len(c.tr_op)) if c.token_key[c.tr_key[i]]]
    ids = _union_size(c, data)
    tok_op_values = np.nonzero((node >= 0) & op_tok[np.maximum(node, 0)])[0]
    read_by_tok = set()
    for v in tok_op_values.tolist():
        read_by_tok.update(preds[v])
    other_bits = sum(int(c.bits[p]) for p in read_by_tok if c.kind[p] == OTHER_INPUT)
    random_ops = {int(node[v]) for v in np.nonzero(c.kind == RANDOM)[0].tolist()}
    seeds = len({o for o in random_ops if op_tok[o]} | {int(node[p]) for p in read_by_tok if c.kind[p] == RANDOM})
    per = max(consumed, 1)
    tok_l, node_l = tok.tolist(), node.tolist()

    # a: the forward's dependent token-indexed values, from their first to the ones the loss reads (the loss: a dependent
    # forward value no dependent token-indexed value comes from, so not a statistic the forward reads back)
    fwd_l = (c.phase == FORWARD).tolist()
    D = (tok & dep & (c.phase == FORWARD)).tolist()
    dep_l = dep.tolist()
    reaches = [False] * n
    for v in range(n - 1, -1, -1):
        if fwd_l[v] and (D[v] or reaches[v]):
            for p in preds[v]:
                reaches[p] = True
    dvals = [v for v in range(n) if D[v]]
    ends = {p for v in range(n) if fwd_l[v] and dep_l[v] and not tok_l[v] and not reaches[v] for p in preds[v] if D[p]}
    if not ends:
        succ_d = {p for v in dvals for p in preds[v] if D[p]}
        ends = {v for v in dvals if v not in succ_d}
    entries = {v for v in dvals if not any(D[p] for p in preds[v])}
    both = entries & ends
    cap = {v: (None if v in ends else int(c.bits[v])) for v in dvals if v not in both}
    a_cut, a_set = vertex_cut(preds, cap, entries - both, ends - both)

    # one token: the first token-index read's batch, and the ops its token-indexed values reach (`path`)
    order, group_tokens = path(c, preds)
    op_group = np.zeros(ops, dtype=bool)
    op_group[order] = True

    # the update values, and the cut to the ones read so far
    U, ucap, carried = _update(c, preds)
    sources = set(carried)
    targets_of: dict[int, set[int]] = {}
    for v in np.nonzero((node >= 0) & op_group[np.maximum(node, 0)])[0].tolist():
        targets_of.setdefault(int(node[v]), set()).update(p for p in preds[v] if p in U)
    g_total = float(op_gates[op_group].sum())
    x_before, x_after, acc, seen, events = {}, {}, 0.0, set(), []
    for o in order:
        x_before[o] = acc / g_total if g_total else 0.0
        acc += op_gates[o]
        x_after[o] = acc / g_total if g_total else 0.0
        new = targets_of.get(o, set()) - seen
        if new:
            seen |= new
            events.append((o, frozenset(seen)))
    chunk = max(1, math.ceil(len(events) / max_cuts))
    cuts = []
    for j in range(0, len(events), chunk):
        run = events[j:j + chunk]
        bits, _ = vertex_cut(preds, ucap, carried, run[-1][1])
        cuts.append((x_before[run[0][0]], bits or 0))
    u = _profile(cuts)
    free = op_group & ~op_reads_upd
    t = float(op_gates[free].sum()) / g_total if g_total else 0.0
    fwd_reads = [o for o in order if op_phase[o] == FORWARD and op_reads_upd[o]]
    lo = fwd_reads[-1] if fwd_reads else -1
    early = (c.phase == FORWARD) & (node >= 0) & (node <= lo)
    op_reads_early = per_op(any_pred(early))
    hi = next((o for o in order if op_phase[o] == BACKWARD and (op_reads_upd[o] or op_reads_early[o])), ops)
    tail = [o for o in order if lo < o < hi]
    frozen = float(op_gates[tail].sum()) / g_total if g_total and fwd_reads else 0.0
    a_x = sorted(x_after[node_l[v]] for v in a_set if node_l[v] in x_after)

    # the whole step's update read, the gradient's cut and the carried state
    all_targets = set()
    for v in tok_op_values.tolist():
        all_targets.update(p for p in preds[v] if p in U)
    update_cut, _ = vertex_cut(preds, ucap, carried, all_targets)
    weightlike = set()
    stack = list(all_targets)
    while stack:
        v = stack.pop()
        if v in weightlike:
            continue
        weightlike.add(v)
        if v not in sources:
            stack.extend(p for p in preds[v] if p in U)
    state = sum(int(c.bits[v]) for v in carried if v not in weightlike)
    G = ~tok & tokdep & dep
    g_vals = np.nonzero(G)[0].tolist()
    g_entries = {v for v in g_vals if any(tokdep[p] and not G[p] for p in preds[v])}
    gcap = {v: int(c.bits[v]) // max(int(c.width[v]), 1) * max(ACCUMULATOR_BITS, int(c.width[v])) if v in g_entries
            else int(c.bits[v]) for v in g_vals}
    g_sinks = {v for v in gcap if c.read_next[v]}
    for v in np.nonzero(reads_upd)[0].tolist():
        g_sinks.update(p for p in preds[v] if p in gcap)
    grad_cut, _ = vertex_cut(preds, gcap, g_entries, g_sinks)
    own = {v for v in g_entries if node_l[v] >= 0 and op_group[node_l[v]]}
    partial_cut, _ = vertex_cut(preds, gcap, own, g_sinks) if own else (None, [])

    return {"region": c.region, "gates": total, "affected_share": share(op_tok & op_dep), "sigma": share(op_tok),
            "affected_all_share": share(op_dep), "tokens": consumed, "step_tokens": c.step_tokens,
            "token_factor": consumed / c.step_tokens if c.step_tokens and consumed else 1.0,
            "tau_ids": ids / per, "tau_other_bits": (other_bits + SEED_BITS * seeds) / per,
            "a_bits": None if a_cut is None else a_cut / per, "a_cut_bits": a_cut,
            "a_x": [a_x[0], a_x[-1]] if a_x else None,
            "u_in": u, "u1_bits": u[-1][1], "u_cuts": len(cuts), "tail_share": t, "frozen_tail_share": frozen,
            "group_gates": g_total, "group_tokens": group_tokens, "update_cut_bits": update_cut,
            "grad_cut_bits": grad_cut, "partial_cut_bits": partial_cut, "state_bits": state,
            "delta_grad_bits": None if grad_cut is None else grad_cut + state} | (c.read_backs or {}) | (
        {"products": c.products} if c.products is not None else {})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("circuit", type=Path, help="a StepCircuit saved by StepCircuit.save")
    ap.add_argument("--max-cuts", type=int, default=128)
    a = ap.parse_args(argv)
    print(json.dumps(measure(StepCircuit.load(a.circuit), a.max_cuts), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
