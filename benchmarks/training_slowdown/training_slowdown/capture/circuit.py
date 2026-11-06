"""The training formalism v2's quantities measured on one traced step, in Boolean gates (te_core.gates) and bits. No torch: a `StepCircuit` saved by ``python -m te_core.capture.script --circuit F.npz`` is measured again with
``python -m te_core.capture.circuit F.npz``.

`StepCircuit` is te_core.capture.structure's dataflow of a program recorded from its start, restricted to its measured
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
  in each sampled id the step feeds back (a generation round's decode, te_core.capture.script). An index computed
  from token data otherwise (a sampled head's unique(y), a QAT code) is no token, though te_core.flops counts its
  reads too; a step whose token-index ops read no token input (a program-made source) counts every token-index read.
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
  |C| by the rule every op is (te_core.gates): their products at the elementwise price, an integer one at bf16's for
  want of a Definition, and their comparisons, casts, shifts, masks and sums at nothing. The values a reduction over
  the step's tokens makes (a dW product, a loss) are priced at the accumulator's width, 32 bits an element at least,
  since a block holding part of the tokens passes the accumulator's partial.
- `update_cut_bits`: the narrowest cut between the carried values and every update value any token op of the step
  reads: the trace's own encoding of θ' (compare 8 |E|).
- `partial_cut_bits`: the gradient's cut from one token's group alone (its own dW products): the partial a block
  holding the group's work passes, at its passed width, which the one bound's reduction kind reads (te_core.formalism).
- `entries`, reading A's inputs (A1's approved reading: a block reads at least the update bits its work reads,
  counted once, plus each token's computed state where its work on the token begins, the context it reads and doesn't
  compute included). One token's path (its group's affected ops in program order, as for u_in) is cut into at most
  `segments` runs: one starts at the forward's end, about half at the forward's narrowest boundaries (where the charged
  bits the rest of the forward reads drop, so an entry lands inside a code), and the rest at about equal gates. For
  each run of segments i through j, what a block doing that work for one of the group's `tokens` reads beyond θ:
  `frontier_bits[i][j - i]`, per token, the computed values its ops read and don't make (the token's state at the
  entry, a skip connection, a statistic; values computed from θ and token data alone are free, the block computing
  them from the token's id; an op none of whose values is read, such as a slice only a zeros_like reads, reads
  nothing); `ids[i][j - i]`, the token-input positions its ops read
  per token (1, the token's id, where they read only such state); `u_bits[i][j - i]`, once a block, the narrowest cut from the carried
  values to the update values its ops read, plus 64 bits a random op it reads. Per segment: its affected `gates` per
  token, its `phase` (the least of its ops'), and, from a trace whose variant halves the context (`positions` its
  length; None otherwise), `context_fwd_bits` and `context_bwd_bits`: the bits per other position of the sequence
  that its token-mixing ops read (an op reading or making a value quadratic in the context, at the least per-position
  bits of its affected token-indexed operands), earlier positions in the forward and later ones in the backward; and its
  first op (`first_ops`), so a run's ops on the path (`path`) are `run_ops`'s. Reading A's checker, `check`, counts the
  gates and reads of any set of the trace's ops by the same rule: a witness block, of which the runs are the default
  packer's (te_core.formalism.packer).
- `data_dependent` and `carried_scalars`: the recording's read-backs that decide what the graph can't carry
  (te_core.capture.script), which the circuit keeps (`read_backs`, saved with it); absent for a circuit no recording
  listed them for, which te_core.formalism refuses as unchecked.
- `products`: the step's matrix products as te_core.io_bound reads them (te_core.capture.products' rows), which the
  circuit keeps too; absent for a circuit recorded before te_core 4.8.
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
    #: the recording's read-backs that decide what the graph can't carry (te_core.capture.script's `data_dependent` and
    #: `carried_scalars`), which `measure` returns; None for a circuit no recording listed them for
    read_backs: dict | None = None
    #: whether a value is quadratic in the context (attention's scores and probabilities), and the context's length;
    #: None and 0 for a trace whose variant doesn't halve the context
    mix: np.ndarray | None = None
    ctx: int = 0
    #: the step's matrix products as te_core.io_bound reads them (te_core.capture.products' `rows`), which `measure`
    #: returns; None for a circuit recorded before te_core 4.8
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
        out = {f.name: (int(z[f.name]) if f.name in ("region", "step_tokens", "ctx") else z[f.name])
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


def _op_gates(c: StepCircuit) -> np.ndarray:
    node = c.node.astype(np.int64)
    ops = int(node.max()) + 1 if len(c) and node.max() >= 0 else 0
    return np.bincount(node[node >= 0], weights=c.gates[node >= 0], minlength=ops)


def _update(c: StepCircuit, preds: list[list[int]]) -> tuple[set[int], dict[int, int], list[int]]:
    """The update values (affected and free of the step's tokens), their bits, and the carried ones among them."""
    U = set(np.nonzero(c.dep.astype(bool) & ~c.tokdep.astype(bool))[0].tolist())
    return U, {v: int(c.bits[v]) for v in U}, [v for v in U if c.kind[v] == CARRIED and (c.outside[v] or not preds[v])]


def _ops_reads(c: StepCircuit, preds: list[list[int]], ops) -> tuple[dict[int, list[int]], dict[int, set[int]]]:
    """Each op of `ops`'s values, and the values it reads: none for an op none of whose values is read."""
    by_op: dict[int, list[int]] = {}
    for v in np.nonzero(np.isin(c.node, list(ops)))[0].tolist():
        by_op.setdefault(int(c.node[v]), []).append(v)
    used = c.read_next.astype(bool).copy()
    used[c.indices] = True
    return by_op, {o: {p for v in by_op.get(o, []) for p in preds[v]} if any(used[v] for v in by_op.get(o, []))
                   else set() for o in ops}


def _charger(c: StepCircuit, U: set[int], tokens: int):
    """What reading a value costs a block beyond θ (module docstring), memoised: ("u", 0) an update value, ("seed", its
    op) a random one, ("bits", its bits over `tokens`) another input or a computed value that depends on θ', ("data", 0)
    one computed from θ and token data alone, ("", 0) θ and the token inputs."""
    dep, tokdep = c.dep.astype(bool), c.tokdep.astype(bool)
    charges: dict[int, tuple[str, float]] = {}

    def charge(p: int) -> tuple[str, float]:
        k = c.kind[p]
        if p in U:
            return "u", 0.0
        if k == TOKEN_INPUT:
            return "", 0.0
        if k == RANDOM:
            return "seed", float(c.node[p])
        if k == OTHER_INPUT or (k == COMPUTED and dep[p]):
            return "bits", int(c.bits[p]) / tokens
        if k == COMPUTED and tokdep[p]:
            return "data", 0.0
        return "", 0.0

    def charged(p: int) -> tuple[str, float]:
        return charges[p] if p in charges else charges.setdefault(p, charge(p))

    return charged


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


def check(c: StepCircuit, ops, tokens: int = 1, preds: list[list[int]] | None = None) -> dict:
    """A witness block's count (reading A, module docstring): the gates of the ops `ops`, as traced, and what they read
    beyond θ, each over `tokens` (a block doing that work for one of the tokens the ops process): `frontier_bits`, the
    values they read and don't make, charged as `_charger` charges them; `ids`, the token-input positions they read (1,
    the token's id, where they read only state computed from token data); `u_bits`, the narrowest cut from the carried
    values to the update values they read, plus 64 bits a random op they read, once. The default packer's blocks are
    such witnesses (a run of `entries`' segments, `run_ops`), and a block any other set of the trace's ops gives is
    counted by the same rule."""
    preds = c.preds() if preds is None else preds
    ops = sorted({int(o) for o in ops})
    U, ucap, carried = _update(c, preds)
    by_op, reads_of = _ops_reads(c, preds, ops)
    charged = _charger(c, U, tokens)
    made = {v for o in ops for v in by_op.get(o, [])}
    bits, targets, seeds, data = 0.0, set(), set(), False
    for p in set().union(*reads_of.values()) - made:
        kind, x = charged(p)
        if kind == "u":
            targets.add(p)
        elif kind == "seed":
            seeds.add(x)
        elif kind == "bits":
            bits += x
        elif kind == "data":
            data = True
    on = set(ops)
    direct = _union_size(c, [r for r in range(len(c.tr_op))
                             if c.token_key[c.tr_key[r]] and int(c.tr_op[r]) in on]) / tokens
    g = _op_gates(c)
    return {"ops": len(ops), "gates": float(g[[o for o in ops if o < len(g)]].sum()) / tokens, "frontier_bits": bits,
            "ids": direct if direct or not data else 1.0,
            "u_bits": (vertex_cut(preds, ucap, carried, targets)[0] or 0) + SEED_BITS * len(seeds)}


def run_ops(order: list[int], entries: dict, i: int, j: int) -> list[int]:
    """The ops of `entries`' segments i through j on the path `order` (`path`)."""
    first = entries["first_ops"]
    hi = first[j + 1] if j + 1 < len(first) else math.inf
    return [o for o in order if first[i] <= o < hi]


def _cut_path(order: list[int], op_gates: np.ndarray, op_phase: np.ndarray, by_op: dict[int, list[int]],
              reads_of: dict[int, set[int]], charged, segments: int) -> dict[int, int]:
    """Each op of the path `order`'s segment, at most `segments` of them: one starts at the forward's end, about half at
    the forward's narrowest boundaries (where the charged bits the rest of the forward reads and the path made before
    drop, the least first), and the rest at about equal gates."""
    n = len(order)
    at = {o: t for t, o in enumerate(order)}
    made = {v: at[o] for o in order for v in by_op.get(o, [])}
    last: dict[int, int] = {}
    for t, o in enumerate(order):
        if op_phase[o] == FORWARD:
            for p in reads_of[o]:
                last[p] = t
    diff = np.zeros(n + 1)
    for p, t in last.items():
        kind, x = charged(p)
        lo = made.get(p, -1) + 1
        if kind == "bits" and lo <= t:
            diff[lo] += x
            diff[t + 1] -= x
    live = np.cumsum(diff)[:n]
    fwd = [t for t in range(1, n) if op_phase[order[t - 1]] == FORWARD]
    bounds = {0, *[t for t in fwd if op_phase[order[t]] != FORWARD][:1]}
    drops = sorted((t for t in fwd if live[t] < live[t - 1]), key=lambda t: (live[t], t))
    bounds |= set(drops[:max(segments // 2 - len(bounds), 0)])
    g = op_gates[order]
    total, rest = float(g.sum()), segments - len(bounds)
    if total and rest > 0:
        raw = ((np.cumsum(g) - g) * (rest + 1) / total).astype(int)
        bounds |= {t for t in range(1, n) if raw[t] != raw[t - 1]}
    b = np.array(sorted(bounds))
    return {o: int(np.searchsorted(b, t, side="right")) - 1 for t, o in enumerate(order)}


def _entries(c: StepCircuit, preds: list[list[int]], order: list[int], op_gates: np.ndarray, op_phase: np.ndarray,
             U: set[int], ucap: dict[int, int], carried: list[int], tokens: int, segments: int) -> dict | None:
    """Reading A's inputs on one token's traced path (module docstring): its ops in program order cut into runs
    (`_cut_path`), and for each run of them, i through j at `[i][j - i]`, what a block doing that work for one of the
    group's tokens reads."""
    if not order or not tokens:
        return None
    tok, dep = c.tok.astype(bool), c.dep.astype(bool)
    mix = c.mix.astype(bool) if c.mix is not None else None
    by_op, reads_of = _ops_reads(c, preds, order)
    charged = _charger(c, U, tokens)
    seg_of = _cut_path(order, op_gates, op_phase, by_op, reads_of, charged, segments)
    n_seg = int(max(seg_of.values())) + 1
    made = [set() for _ in range(n_seg)]
    reads = [set() for _ in range(n_seg)]
    gates = [0.0] * n_seg
    phase = [OTHER] * n_seg
    ctx = {FORWARD: [0.0] * n_seg, BACKWARD: [0.0] * n_seg}
    for o in order:
        s, vs, rd = seg_of[o], by_op.get(o, []), reads_of[o]
        gates[s] += float(op_gates[o]) / tokens
        phase[s] = min(phase[s], int(op_phase[o]))
        made[s].update(vs)
        reads[s] |= rd
        if mix is not None and op_phase[o] in ctx and (any(mix[x] for x in vs) or any(mix[p] for p in rd)):
            per = [int(c.bits[p]) / tokens for p in rd if tok[p] and dep[p] and not mix[p]]
            ctx[int(op_phase[o])][s] += min(per, default=0.0)

    token_reads = [[] for _ in range(n_seg)]
    for r in range(len(c.tr_op)):
        if c.token_key[c.tr_key[r]] and int(c.tr_op[r]) in seg_of:
            token_reads[seg_of[int(c.tr_op[r])]].append(r)

    frontier, ids, u_bits = [], [], []
    cut_of: dict[frozenset, int] = {}
    for i in range(n_seg):
        seen, inside, bits, direct, data, targets, seeds = set(), set(), 0.0, 0.0, False, set(), set()
        read_at: list[int] = []
        row_f, row_i, row_u = [], [], []
        for j in range(i, n_seg):
            inside |= made[j]
            if token_reads[j]:
                read_at += token_reads[j]
                direct = _union_size(c, read_at) / tokens
            for p in reads[j] - seen - inside:
                seen.add(p)
                kind, x = charged(p)
                if kind == "u":
                    targets.add(p)
                elif kind == "seed":
                    seeds.add(x)
                elif kind == "bits":
                    bits += x
                elif kind == "data":
                    data = True
            key = frozenset(targets)
            if key not in cut_of:
                cut_of[key] = vertex_cut(preds, ucap, carried, key)[0] or 0
            row_f.append(bits)
            row_i.append(direct if direct or not data else 1.0)
            row_u.append(cut_of[key] + SEED_BITS * len(seeds))
        frontier.append(row_f)
        ids.append(row_i)
        u_bits.append(row_u)
    first = [0] * n_seg
    for o in reversed(order):
        first[seg_of[o]] = o
    return {"segments": n_seg, "first_ops": first, "gates": gates, "phase": phase, "frontier_bits": frontier, "ids": ids,
            "u_bits": u_bits,
            "context_fwd_bits": ctx[FORWARD] if mix is not None else None,
            "context_bwd_bits": ctx[BACKWARD] if mix is not None else None,
            "positions": c.ctx or None, "tokens": tokens}


def measure(c: StepCircuit, max_cuts: int = 128, segments: int = 24) -> dict:
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
    entries = _entries(c, preds, order, op_gates, op_phase, U, ucap, carried, group_tokens, segments)

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
            "delta_grad_bits": None if grad_cut is None else grad_cut + state, "entries": entries} | (c.read_backs or {}) | (
        {"products": c.products} if c.products is not None else {})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("circuit", type=Path, help="a StepCircuit saved by te_core.capture.script --circuit")
    ap.add_argument("--max-cuts", type=int, default=128)
    a = ap.parse_args(argv)
    print(json.dumps(measure(StepCircuit.load(a.circuit), a.max_cuts), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
