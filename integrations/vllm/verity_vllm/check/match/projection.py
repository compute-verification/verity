"""A request's projection of the batched fold: the instances resolved to it, segmented at its own TokenSelects, with every read of
another request's prompt, seed, peers, witness or node canonicalised as a cross-request read (`XREQ`)."""

from __future__ import annotations

import collections

from verity_vllm.check.match import prefix_reads as PX
from verity_vllm.check.match import program_compare as C
from verity_vllm.check.match import supplied_inputs as SI

XREQ = "XREQ"


class Projection(PX.CachedKv, C.RootView):
    """Request r's view of the oracle RootView: the subset of instances attributed to r, segmented at r's TokenSelects (a warm request's reads
    of its cached prefix: `prefix_reads.CachedKv`)."""

    def __init__(self, oracle: C.RootView, label: str, inst_idx: list[int], tok_node: int, weights_node: "int | set[int] | frozenset[int]",
                 seed_node: int | None = None, peer_node: int | None = None):
        # no super().__init__: reuse the decoded oracle, never decode twice
        self.label = label
        self._fold = oracle
        self.digest = f"{oracle.digest}#{label}"
        self.nodes = oracle.nodes
        self.inputs = oracle.inputs
        # the request's OWN seed parameter canonicalises as ("SEED", ..) (the derived Program's `seed`); another request's is xreq
        self.seed_nodes = {seed_node} if seed_node is not None else set()
        # the request's OWN `peers_<i>` node canonicalises as ("PEER", k, ..) (the rank Program's `peers_<k>`); another request's is
        # xreq.  `peer_fields` keeps only this request's node so `RootView._src`'s peer branch never answers for a foreign one.
        self.peer_node = peer_node
        self.peer_fields = {peer_node: oracle.peer_fields[peer_node]} if peer_node is not None and peer_node in getattr(oracle, "peer_fields", {}) else {}
        self._all_peer_nodes = set(getattr(oracle, "peer_fields", {}))
        self.peer_index = getattr(oracle, "peer_index", {})
        # the request's OWN witness parameters canonicalise as ("WIT", name, ..) (the rank Program's witness input of that name); another
        # request's is xreq.  Each is request i's for its header index i = this prompt's node, as a `run_<i>` below.
        w_index = getattr(oracle, "witness_index", {})
        self.witness_fields = {k: nm for k, nm in getattr(oracle, "witness_fields", {}).items() if w_index.get(k) == tok_node}
        self._all_witness_nodes = set(getattr(oracle, "witness_fields", {}))
        # the registered `values` are shared like the weights; a `run_<i>` is request i's (the header index = this prompt's node)
        owner = getattr(oracle, "supplied_owner", {})
        self.supplied_fields = {k: f for k, f in getattr(oracle, "supplied_fields", {}).items() if owner.get(k) in (None, tok_node)}
        self.supplied_owner = {k: owner.get(k) for k in self.supplied_fields}
        self._all_supplied = set(getattr(oracle, "supplied_fields", {}))
        self.param_names = getattr(oracle, "param_names", [])
        self.inst = [oracle.inst[i] for i in inst_idx]
        self.weight_fields = {}
        self.tok_node = tok_node
        # the shared weights = a SET of Input nodes (one per leaf field of a mixed-width Record; one node for a
        # homogeneous one).  An int keeps the older call form (`n_input_nodes - 1`) meaning the same single node.
        self.weights_nodes = frozenset(weights_node) if isinstance(weights_node, (set, frozenset, list, tuple)) else frozenset({weights_node})
        self.weights_node = weights_node if isinstance(weights_node, int) else (min(self.weights_nodes) if self.weights_nodes else None)
        self.k2i = {r["k"]: i for i, r in enumerate(self.inst)}
        self.oracle_rows = None
        self.oracle_wmap = getattr(oracle, "oracle_wmap", None)
        self._wlos = getattr(oracle, "_wlos", None)
        self.oracle_wmap_overlaps = getattr(oracle, "oracle_wmap_overlaps", None)
        self.compact = getattr(oracle, "compact", False)   # the oracle's operand representation (runs | progressions)
        # Record-global offset of each weights Input node's first leaf (compare.load_oracle): without it every
        # multi-node weight read would be located at offset 0 of the field map -- the wrong field (or `W?`) -- on the projection side only
        self.input_base = getattr(oracle, "input_base", {})
        # the fold's bound top-p split constants (compare.RootView._bind_splits): select node -> S; the
        # request's e-th select canonicalises its splits operand as ("SPLITS", e, e+1) against the Program's `splits` input entry e
        self.splits_values = getattr(oracle, "splits_values", {})
        self.splits_nodes = getattr(oracle, "splits_nodes", set())
        self.xreq = collections.Counter()
        self.xreq_examples: list[dict] = []
        self._segment()

    def _src(self, run, unfold: bool):
        space, idx, lo, hi = run
        if space == "n":
            if idx in self.inputs:
                if idx in self._all_supplied:
                    if idx in self.supplied_fields:
                        return SI.src(self, idx, lo, hi)
                    self._xreq("run-param", idx, lo, hi)
                    return (XREQ, "run", idx, lo, hi)
                if idx == self.tok_node:
                    return ("TOK", lo, hi)
                if idx in self.seed_nodes:
                    return ("SEED", lo, hi)
                if idx in self.splits_nodes:
                    return ("SPLITS", lo, hi)
                if idx == self.peer_node:
                    return self._peer_src(idx, lo, hi)     # the request's own received-collective peer input
                if idx in self.witness_fields:
                    return ("WIT", self.witness_fields[idx], lo, hi)
                if idx not in self.weights_nodes:
                    kind = self._foreign_kind(idx)
                    self._xreq(f"{kind}-param", idx, lo, hi)
                    return (XREQ, kind, idx, lo, hi)
            elif idx not in self.k2i:
                self._xreq("node", idx, lo, hi)
                return (XREQ, "node", idx, lo, hi)
        return super()._src(run, unfold)

    def _srcP(self, p: tuple, unfold: bool) -> list[tuple]:
        """`_src` for one run progression `(space, (idx0, lo0, hi0), (dI, dL, dH), count)` (compare.RootView._srcP): the request's
        own inputs by kind, another request's reads as XREQ counted per RUN, everything else the oracle RootView's mapping.  Uniform only
        when every run of the progression is provably of one kind (one node, or a node range wholly outside this request's instances
        and the inputs); otherwise the record's mapping run by run."""
        space, base, delta, c = p
        if space == "n":
            idx0, lo0, hi0 = base
            dI, dL, dH = delta
            if dI == 0 or c == 1:
                if idx0 in self.inputs:
                    if idx0 in self._all_supplied:
                        if idx0 in self.supplied_fields:
                            return SI.srcP(self, p, unfold)
                        self._xreq("run-param", (idx0, lo0, hi0), (0, dL, dH), c)
                        return [(XREQ, "run", (idx0, lo0, hi0), (0, dL, dH), c)]
                    if idx0 == self.tok_node:
                        return [("TOK", (lo0, hi0), (dL, dH), c)]
                    if idx0 in self.seed_nodes:
                        return [("SEED", (lo0, hi0), (dL, dH), c)]
                    if idx0 in self.splits_nodes:
                        return [("SPLITS", (lo0, hi0), (dL, dH), c)]
                    if idx0 == self.peer_node:
                        return self._peer_srcP(p, unfold)     # the request's own received-collective peer input
                    if idx0 in self.witness_fields:
                        return [("WIT", self.witness_fields[idx0], (lo0, hi0), (dL, dH), c)]
                    if idx0 not in self.weights_nodes:
                        kind = self._foreign_kind(idx0)
                        self._xreq(f"{kind}-param", (idx0, lo0, hi0), (0, dL, dH), c)
                        return [(XREQ, kind, (idx0, lo0, hi0), (0, dL, dH), c)]
                elif idx0 not in self.k2i:
                    self._xreq("node", (idx0, lo0, hi0), (0, dL, dH), c)
                    return [(XREQ, "node", (idx0, lo0, hi0), (0, dL, dH), c)]
                return super()._srcP(p, unfold)
            # a window over several producer nodes: all of them another request's (none ours, none an input) -> one XREQ progression
            lo_i, hi_i = min(idx0, idx0 + (c - 1) * dI), max(idx0, idx0 + (c - 1) * dI)
            if lo_i > self._max_input and self._own_count(lo_i, hi_i) == 0:
                self._xreq("node", (idx0, lo0, hi0), (dI, dL, dH), c)
                return [(XREQ, "node", (idx0, lo0, hi0), (dI, dL, dH), c)]
            if lo_i > self._max_input and self._own_count(lo_i, hi_i) == hi_i - lo_i + 1:   # wholly ours: the RootView mapping decides
                return super()._srcP(p, unfold)
            return self._per_run(p, unfold)
        return super()._srcP(p, unfold)

    def _foreign_kind(self, idx: int) -> str:
        """Which parameter of another request Input node `idx` is: its seed, peers or witness, else its prompt."""
        if idx in getattr(self, "_all_seed_nodes", ()):
            return "seed"
        if idx in self._all_peer_nodes:
            return "peers"
        return "witness" if idx in self._all_witness_nodes else "prompt"

    @property
    def _max_input(self) -> int:
        m = getattr(self, "_max_input_cache", None)
        if m is None:
            m = self._max_input_cache = max(self.inputs) if self.inputs else -1
        return m

    def _own_count(self, lo_i: int, hi_i: int) -> int:
        """How many of this request's instance node indices lie in [lo_i, hi_i]."""
        import bisect
        own = getattr(self, "_own_sorted", None)
        if own is None:
            own = self._own_sorted = sorted(self.k2i)
        return bisect.bisect_right(own, hi_i) - bisect.bisect_left(own, lo_i)

    def _xreq(self, kind: str, *ref) -> None:
        """One cross-request read: `ref` = (idx, lo, hi) for a run, or (base, delta, count) for a progression of `count` runs -- counted
        and exemplified per RUN either way (the counts and the first 12 examples are on the record)."""
        if len(ref) == 3 and not isinstance(ref[0], tuple):
            idx, lo, hi = ref
            self.xreq[kind] += 1
            if len(self.xreq_examples) < 12:
                self.xreq_examples.append({"kind": kind, "ref": [idx, lo, hi]})
            return
        base, delta, c = ref
        self.xreq[kind] += c
        k = 0
        while len(self.xreq_examples) < 12 and k < c:
            self.xreq_examples.append({"kind": kind, "ref": [b + k * d for b, d in zip(base, delta)]})
            k += 1


def engine_steps_of(P: Projection, meta_of: dict[int, dict]) -> list[list[int]]:
    """Per projection step: the sorted engine steps its instances came from (1 each unless the prefill was chunked)."""
    out = []
    for idxs in P.steps:
        out.append(sorted({meta_of[P.inst[i]["k"]]["step"] for i in idxs if P.inst[i]["k"] in meta_of}))
    return out
