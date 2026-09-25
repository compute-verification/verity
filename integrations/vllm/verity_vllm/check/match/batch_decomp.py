"""Batch decomposition (M-BATCH-DECOMP, R13 X-09): Match at B>1 by projecting the batched eager fold onto its requests.

Under batch invariance a batched kernel launch (one GEMM with M = Σ tokens of the co-scheduled requests, one fused norm over
the step's token rows, ...) is the token-row-wise union of the per-request Calls the padded per-request Programs describe. The
fold already emits ONE Call per token row (`fold.py`, "Per-row emission"), so the batched execution IS a sequence of
per-token-row Calls; what Match lacked was the occurrence resolution of each Call to its request and a per-request comparison.

Occurrence resolution (glossary; the code still says 'attribution': the fold's own naming rule, `fold._name` /
`patterns.attribution_for`): a Call with `rows == num_tokens` of its engine step is token row `row` -> `StepStart.rows[row] =
(request, position)`; with `rows == len(logits_rows)` it is logits row `row` -> `StepStart.logits_rows[row]`. When both counts
coincide (a pure decode step) the two resolutions must
name the same request, else the Call is AMBIGUOUS. Any other row count is UNATTRIBUTABLE. Both refuse (fail-closed):
no Call is dropped, guessed or shared.

Projection: request r's Calls in fold order, segmented at r's own `TokenSelect` Calls into r's steps (prefill,
then one per decode). Operand sources are canonicalised exactly as `compare.Prog.canon` does, relative to the projection:
`(TOK, lo, hi)` for r's OWN prompt parameter, `(W, field, lo, hi)` for weights, `(N, step_r, index_r, lo, hi)` for a node
produced by one of r's Calls. A read of ANOTHER request's prompt parameter or of a node resolved to another request
is a cross-request dependency (`XREQ`): it is counted, reported with examples, and fails the sweep row — token-row independence is
exactly what BI guarantees, so such a read is either a non-row-separable op (must be listed explicitly) or a defect.

Comparison: each projection against the derived request-wrapper Program for the request's OWN shape `(LP_r, T_r)` —
`LP_r` = its prompt length (RunHeader), `T_r` = its TokenSelect count − 1 (decode steps actually served) — with
`compare.compare_steps` (spec sequence per step, 1:1 wiring, canonical sha256). One derivation per distinct `(LP, T)`,
reusable across requests and rows. The verdict is PASS iff every request is canonical-equal AND every Call is
resolved AND no XREQ AND every needed derivation is present. Missing derivations are listed (`missing_derived`) and
exit 4 so the driver (`row_pod.sh`) can derive them and rerun; refusal/mismatch exits 3.

    python -m verity_vllm.check.match.batch_decomp --oracle <match dir: program.json + instances.jsonl + capture/log.jsonl.gz>
        --derived <dir with descriptor.json.gz + derive-report.json + artifact.json> [--derived <dir> ...] --out decomp.json

PROTOCOL SUMMARY (Match stage, B>1: the per-request decomposition of the batched fold -- X-09; occurrence resolution at B>1)
  Purpose: resolve every Call of the batched eager fold to the request that produced it, project each request's Calls into
    its own steps, and compare that projection canonically against the Build's request-wrapper Program of the request's OWN shape.
  Inputs: --oracle <match dir> (program.json + instances.jsonl = the fold; capture/log.jsonl.gz = RunHeader / StepStart rows);
    --derived <build_request*> dirs (descriptor.json.gz, derive-report.json, artifact.json), one per distinct (LP, T).
  Outputs: --out match_decomp.json {verdict PASS|FAIL|MISSING_DERIVED, requests[{canonical_equal, LP, T, steps, totals}],
    attribution{unattributed}, xreq_total, missing_derived, within_step_criterion, reasons}.  Exit 0 PASS, 4 missing derived
    shapes (row_pod derives them and reruns once), 3 refusal / mismatch.
  Invariants: no Call is dropped, guessed or shared -- an AMBIGUOUS or UNATTRIBUTABLE token-row count refuses; a read of another
    request's prompt or node is XREQ and fails the sweep row; a request is compared at (LP_r, T_r) = (its prompt length, its served decode
    steps), never at the row envelope; the operand counts are the positional comparison, the verdict follows --within-step (dag of
    record; sequence recorded beside it).  Inside GM-01 (MATCH_PIPELINE=shared) the same decompose() writes this document via --decomp-out.
  Protocol spec: Notion -> "vLLM integration — protocol spec (Build · Match · Commit)".
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import re
import sys
import time

from verity_vllm.check.match import program_compare as C
from verity_vllm.check.program_ops import Ops
from verity_vllm.program.sampling_event import is_sampling_event as _is_sampling_event  # [R15 stoch]

XREQ = "XREQ"
_TAG_RE = re.compile(r'^\{"\$e":\s*"(\w+)"')


# ------------------------------------------------------------------------------------------------ log: header + steps


def read_steps(log_path: str) -> tuple[dict, dict[int, dict]]:
    """RunHeader (as dict) and per engine step {rows: [(req, pos)], logits_rows: [(req, pos)], logits_indices: [int]}."""
    header = None
    steps: dict[int, dict] = {}
    opener = gzip.open if log_path.endswith(".gz") else open
    with opener(log_path, "rt") as f:
        for line in f:
            m = _TAG_RE.match(line)  # the writer puts "$e" first (`events.to_json`); decode only the two record kinds we need
            tag = m.group(1) if m else None
            if tag == "StepStart":
                d = json.loads(line)
                s = int(d["step"])
                if s in steps:
                    raise ValueError(f"log has two StepStart records for step {s}")
                steps[s] = {"rows": [(r, int(p)) for r, p in d["rows"]], "logits_rows": [(r, int(p)) for r, p in d["logits_rows"]],
                            "logits_indices": [int(x) for x in d.get("logits_indices") or []]}
            elif tag == "RunHeader":
                if header is not None:
                    raise ValueError("log has two RunHeader records")
                header = json.loads(line)
    if header is None:
        raise ValueError(f"{log_path}: no RunHeader")
    return header, steps


def request_params(header: dict) -> tuple[list[dict], dict[str, int], dict[str, str]]:
    """(requests in header order = prompt parameter order, request_id -> prompt param index, output id -> parent id).
    Mirrors `fold._header`: a forked child (`workload.children`) reads its PARENT's prompt parameter."""
    reqs = list((header.get("workload") or {}).get("requests") or [])
    if not reqs:
        raise ValueError("RunHeader.workload has no requests")
    idx = {r["request_id"]: i for i, r in enumerate(reqs)}
    parent = {r["request_id"]: r["request_id"] for r in reqs}
    for c in (header.get("workload") or {}).get("children") or []:
        if c.get("parent") not in idx:
            raise ValueError(f"child {c.get('request_id')!r} names unknown parent {c.get('parent')!r}")
        idx[c["request_id"]] = idx[c["parent"]]
        parent[c["request_id"]] = c["parent"]
    return reqs, idx, parent


# ------------------------------------------------------------------------------------------------ attribution


SHARED = "*weight-only*"   # owner sentinel of a weight-only instance (F-r19-int-15); never a request id


def is_weight_only(r: dict, weights_nodes: frozenset, shared_nodes: set, compact: bool) -> bool:
    """[R19, F-r19-int-15] True when every node operand of fold instance `r` is a shared weight or an earlier weight-only instance (node id
    in `shared_nodes`) and there is at least one: the instance carries no request's data (gemma2's RMSNorm scale `1 + w`).  The rule
    `decompose` and GM-01 (global_match.check, G2) apply after the meta channel and before the row-count rule -- one definition."""
    if compact:
        from verity_vllm.program.compact import idx_values as _idx_values
    reads = False
    for a in r["args"]:
        for run in a:   # record (space, idx, lo, hi) | compact (space, (idx0, lo, hi), (dI, dL, dH), count): space / idx only
            if run[0] != "n":
                continue
            for idx in (_idx_values(run) if compact else (run[1],)):
                if idx in weights_nodes or idx in shared_nodes:
                    reads = True
                else:
                    return False
    return reads


def shared_legs(per_req: dict, shared_idx: list, steps: dict) -> dict:
    """[R19, F-r19-int-15] Each request's instance list with the weight-only instances of every engine step the request has rows in
    (StepStart / logits rows) merged in, in node order: the request's own derive computes that node inside its forward at that step.
    A request with no rows in the step does not see it.  Request order is preserved."""
    present: dict[int, set[str]] = {s: {r for r, _ in st.get("rows", [])} | {r for r, _ in st.get("logits_rows", [])} for s, st in steps.items()}
    return {rid: sorted(idxs + [i for i, s in shared_idx if rid in present.get(s, ())]) for rid, idxs in per_req.items()}


def attribute(meta: dict, st: dict) -> tuple[str | None, str]:
    """(request_id | None, how) for one fold instance meta {step,row,rows} given its engine step's StepStart."""
    row, rows = meta.get("row"), meta.get("rows")
    if row is None or rows is None:
        return None, "no-row"
    n_tok, n_log = len(st["rows"]), len(st["logits_rows"])
    if rows == n_tok and rows == n_log:
        if row >= n_tok:
            return None, "row-out-of-range"
        a, b = st["rows"][row][0], st["logits_rows"][row][0]
        return (a, "token=logits") if a == b else (None, f"ambiguous:{a}!={b}")
    if rows == n_tok:
        return (st["rows"][row][0], "token") if row < n_tok else (None, "row-out-of-range")
    if rows == n_log:
        return (st["logits_rows"][row][0], "logits") if row < n_log else (None, "row-out-of-range")
    # [R14 moe] fused-MoE expert instances (MoeExpertGemm / SiluMul / MoeExpertGemmW) run over the TOPK*tokens pair rows, pair-major
    # p = t*TOPK + k (the Program's addressing, independent of the sorted block schedule): row p belongs to token p // TOPK.  The R13 MoE
    # X-09 rule; without it every MoE-block instance is "attributed to no request" (OLMoE b1: 110 208 of 165 950).
    if n_tok and rows > n_tok and rows % n_tok == 0:
        k = rows // n_tok
        return (st["rows"][row // k][0], f"moe-pair(x{k})") if row < rows else (None, "row-out-of-range")
    return None, f"rows={rows}!=tokens({n_tok})|logits({n_log})"


# ------------------------------------------------------------------------------------------------ projection


def weights_input_nodes(B: "C.Prog", n_request_params: int, n_seed_params: int = 0, n_peer_params: int = 0) -> set[int]:
    """[R16 global, F-fp8-GM1 / M-0823] The fold's shared WEIGHTS parameter as a SET of Input nodes: every Input node of the oracle
    Prog that no request owns.  The fold's root parameters are `prompt_<r>` (one per request, header order), then `seed_<i>` (one per
    seeded request, [R15 stoch]), then ONE `weights` Record (`fold._header`).  `verity.ir.program._lower_input` lowers a width-homogeneous
    Record to ONE `Input` batch (the bf16 dense models: `n_input_nodes - 1` was that node) but a MIXED-width Record to one batch PER LEAF
    FIELD (Qwen3-4B-FP8: e4m3 weight + f32 `weight_scale_inv` per linear + bf16 norms -> 435 nodes at B=1), so the last Input node is not
    "the weights" and the node count says nothing about the number of requests.  The request-owned Inputs are exactly the first
    `n_request_params + n_seed_params` nodes (the codec emits one node per prompt/seed parameter, in parameter order, before the Record's
    leaves); everything after them is the weights, shared by every request subcircuit exactly once (one global address per leaf).
    Refuses by name a Prog whose Input nodes cannot hold the header's parameters or whose request-owned Inputs are not the leading ones.
    [R17 D90] a tensor-parallel rank fold adds ONE `peers_<i>` Record per request (homogeneous bf16 -> one Input node each) after the seeds
    and before the weights: `n_peer_params` of them, request-owned."""
    n_owned = n_request_params + n_seed_params + n_peer_params
    inputs = sorted(B.inputs)
    if len(inputs) <= n_owned:
        raise ValueError(f"program has {len(inputs)} input nodes but the header has {n_request_params} requests + {n_seed_params} seeds "
                         f"+ {n_peer_params} peers (>= 1 weights node expected)")
    owned = inputs[:n_owned]
    if owned != list(range(n_owned)):
        raise ValueError(f"the request-owned input nodes are not the leading Input nodes 0..{n_owned - 1}: {owned[:8]}")
    w = set(inputs[n_owned:])
    # the codec's layout contract (compare.load_oracle `input_base`): the weights leaves are contiguous after the owned parameters
    if min(w) != n_owned:
        raise ValueError(f"first weights input node is {min(w)}, expected {n_owned} (request params {n_request_params}, seeds {n_seed_params}, "
                         f"peers {n_peer_params})")
    # cross-check against the fold's OWN statement of where the weights parameter is: load_oracle keys `input_base` by every Input node at or
    # after the weights parameter index the instances actually read (`operand_refs` space "p", fold contract: weights = the LAST parameter).
    # A header claiming more (or fewer) requests than the Program has prompt parameters puts the boundary elsewhere -> refused by name.
    base = getattr(B, "input_base", None)
    if base:
        if set(base) != w:
            raise ValueError(f"header and Program disagree on the input layout: the header's {n_request_params} requests + {n_seed_params} seeds put "
                             f"the weights at input nodes {min(w)}..{max(w)}, the fold's instances read the weights parameter as input nodes "
                             f"{min(base)}..{max(base)}")
    return w


class Projection(C.Prog):
    """Request r's view of the oracle Prog: the subset of instances attributed to r, segmented at r's TokenSelects."""

    def __init__(self, oracle: C.Prog, label: str, inst_idx: list[int], tok_node: int, weights_node: "int | set[int] | frozenset[int]",
                 seed_node: int | None = None, peer_node: int | None = None):  # noqa: D107
        # no super().__init__: reuse the decoded oracle, never decode twice
        self.label = label
        self.digest = f"{oracle.digest}#{label}"
        self.nodes = oracle.nodes
        self.inputs = oracle.inputs
        # [R15 stoch] the request's OWN seed parameter canonicalises as ("SEED", ..) (the derived Program's `seed`); another request's is xreq
        self.seed_nodes = {seed_node} if seed_node is not None else set()
        # [R17 D90] the request's OWN `peers_<i>` node canonicalises as ("PEER", k, ..) (the rank Program's `peers_<k>`); another request's is
        # xreq.  `peer_fields` keeps only this request's node so `Prog._src`'s peer branch never answers for a foreign one.
        self.peer_node = peer_node
        self.peer_fields = {peer_node: oracle.peer_fields[peer_node]} if peer_node is not None and peer_node in getattr(oracle, "peer_fields", {}) else {}
        self._all_peer_nodes = set(getattr(oracle, "peer_fields", {}))
        self.peer_index = getattr(oracle, "peer_index", {})
        self.param_names = getattr(oracle, "param_names", [])
        self.inst = [oracle.inst[i] for i in inst_idx]
        self.weight_fields = {}
        self.tok_node = tok_node
        # [R16 global, F-fp8-GM1] the shared weights = a SET of Input nodes (one per leaf field of a mixed-width Record; one node for a
        # homogeneous one).  An int keeps the pre-R16 call form (`n_input_nodes - 1`) meaning the same single node.
        self.weights_nodes = frozenset(weights_node) if isinstance(weights_node, (set, frozenset, list, tuple)) else frozenset({weights_node})
        self.weights_node = weights_node if isinstance(weights_node, int) else (min(self.weights_nodes) if self.weights_nodes else None)
        self.k2i = {r["k"]: i for i, r in enumerate(self.inst)}
        self.oracle_rows = None
        self.oracle_wmap = getattr(oracle, "oracle_wmap", None)
        self._wlos = getattr(oracle, "_wlos", None)
        self.oracle_wmap_overlaps = getattr(oracle, "oracle_wmap_overlaps", None)
        self.compact = getattr(oracle, "compact", False)   # [R17 mfast] the oracle's operand representation (runs | progressions)
        # [R16 global, F-fp8-GM1] Record-global offset of each weights Input node's first leaf (compare.load_oracle): without it every
        # multi-node weight read would be located at offset 0 of the field map -- the wrong field (or `W?`) -- on the projection side only
        self.input_base = getattr(oracle, "input_base", {})
        # [R16 stoch/global, COORD M-0169 (4)] the fold's bound top-p split constants (compare.Prog._bind_splits): select node -> S; the
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
                if idx == self.tok_node:
                    return ("TOK", lo, hi)
                if idx in self.seed_nodes:
                    return ("SEED", lo, hi)
                if idx in self.splits_nodes:
                    return ("SPLITS", lo, hi)
                if idx == self.peer_node:
                    return self._peer_src(idx, lo, hi)     # [R17 D90] the request's own received-collective peer input
                if idx not in self.weights_nodes:
                    kind = "seed" if idx in getattr(self, "_all_seed_nodes", ()) else ("peers" if idx in self._all_peer_nodes else "prompt")
                    self._xreq(f"{kind}-param", idx, lo, hi)
                    return (XREQ, kind, idx, lo, hi)
            elif idx not in self.k2i:
                self._xreq("node", idx, lo, hi)
                return (XREQ, "node", idx, lo, hi)
        return super()._src(run, unfold)

    def _srcP(self, p: tuple, unfold: bool) -> list[tuple]:
        """[R17 mfast] `_src` for one run progression `(space, (idx0, lo0, hi0), (dI, dL, dH), count)` (compare.Prog._srcP): the request's
        own inputs by kind, another request's reads as XREQ counted per RUN, everything else the oracle Prog's mapping.  Uniform only
        when every run of the progression is provably of one kind (one node, or a node range wholly outside this request's instances
        and the inputs); otherwise the record's mapping run by run."""
        space, base, delta, c = p
        if space == "n":
            idx0, lo0, hi0 = base
            dI, dL, dH = delta
            if dI == 0 or c == 1:
                if idx0 in self.inputs:
                    if idx0 == self.tok_node:
                        return [("TOK", (lo0, hi0), (dL, dH), c)]
                    if idx0 in self.seed_nodes:
                        return [("SEED", (lo0, hi0), (dL, dH), c)]
                    if idx0 in self.splits_nodes:
                        return [("SPLITS", (lo0, hi0), (dL, dH), c)]
                    if idx0 == self.peer_node:
                        return self._peer_srcP(p, unfold)     # [R17 D90] the request's own received-collective peer input
                    if idx0 not in self.weights_nodes:
                        kind = "seed" if idx0 in getattr(self, "_all_seed_nodes", ()) else ("peers" if idx0 in self._all_peer_nodes else "prompt")
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
            if lo_i > self._max_input and self._own_count(lo_i, hi_i) == hi_i - lo_i + 1:   # wholly ours: the Prog mapping decides
                return super()._srcP(p, unfold)
            return self._per_run(p, unfold)
        return super()._srcP(p, unfold)

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


def _producer(P: C.Prog, t) -> int | None:
    if t[0] != "N":
        return None
    return P.steps[t[1]][t[2]]


def _producers(P: C.Prog, t) -> list[int]:
    """Every producer instance of a canonical N source, record form or [R17 mfast] progression."""
    from verity_vllm.program.compact import n_producers
    return n_producers(P.steps, t)


def live_cone(P: C.Prog) -> set[int]:
    """Instances in the backward cone of the LAST TokenSelect: everything the served token sequence depends on (each TokenSelect
    feeds the next step's embedding, so the chain reaches every served TokenSelect). What falls outside is DEAD for serving — with a
    chunked prefill vLLM v1 runs the sampler on every scheduled request, including a partially-prefilled one, and DISCARDS that
    token: the fold records that final-norm row / lm_head row / TokenSelect, the request-wrapper Program never computes it.
    Producers precede consumers, so one reverse pass closes the cone."""
    order = [i for idxs in P.steps for i in idxs]
    ts = [i for i in order if _is_sampling_event(P.inst[i]["spec"])]   # [R15 stoch] any sampling-event family
    if not ts:
        return set(order)
    live = {ts[-1]}
    for i in reversed(order):
        if i in live:
            for arg in P.canon(i, False):
                for t in arg:
                    if t[0] == "N":
                        live.update(_producers(P, t))
    return live


def dag_hashes(P: C.Prog, only: set[int] | None = None) -> dict[int, str]:
    """Emission-order-insensitive canonical form: every instance hashed by (spec, form, operand sources, out type) where a node
    source carries its PRODUCER'S HASH (Merkle) instead of its (step, index) position. Two Programs with equal multisets hold the
    same computations over the same weights/prompt reads with the same dependency trees; only the order in which the fold emitted
    them (and the engine-step segmentation) is abstracted. Used for a request whose prefill the engine CHUNKED over several steps:
    the eager fold then emits chunk-major (all ops of rows 0..c-1, then all ops of rows c..LP-1) while the request-wrapper Program
    is op-major over LP rows — the same DAG, a different sequence. `only` restricts the multiset to those instances (live cone);
    producers precede consumers in node order."""
    import hashlib
    hashes: dict[int, str] = {}
    compact = getattr(P, "compact", False)
    if compact:
        from verity_vllm.program.compact import expand_mixed
    for s, idxs in enumerate(P.steps):
        for i in idxs:
            if only is not None and i not in only:
                continue
            r = P.inst[i]
            parts = []
            for arg in P.canon(i, False):
                if compact:   # [R17 mfast] the record's per-run payload: a gather's producers carry one hash EACH
                    arg = expand_mixed(arg)
                row = []
                for t in arg:
                    if t[0] == "N":
                        ii = P.steps[t[1]][t[2]]
                        row.append(["N", hashes.get(ii, f"forward-ref-or-dead:{ii}"), *t[3:]])
                    else:
                        row.append(list(t))
                parts.append(row)
            hashes[i] = hashlib.sha256(json.dumps([r["spec"], r["form"], parts, r["out"]], separators=(",", ":"), default=str).encode()).hexdigest()
    return hashes


def dag_multiset(P: C.Prog, only: set[int] | None = None) -> collections.Counter:
    return collections.Counter(dag_hashes(P, only).values())


def engine_steps_of(P: Projection, meta_of: dict[int, dict]) -> list[list[int]]:
    """Per projection step: the sorted engine steps its instances came from (1 each unless the prefill was chunked)."""
    out = []
    for idxs in P.steps:
        out.append(sorted({meta_of[P.inst[i]["k"]]["step"] for i in idxs if P.inst[i]["k"] in meta_of}))
    return out


# ------------------------------------------------------------------------------------------------ derived index


def derived_shape(d: str) -> tuple[int, int] | None:
    """(LP, T) of a derived request-wrapper dir: derive_step's construction_manifest.json `structural_inputs`."""
    for fn in ("construction_manifest.json", "artifact.json"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            si = (json.load(open(p)).get("structural_inputs") or {})
            if si.get("LP") is not None and si.get("T") is not None:
                return int(si["LP"]), int(si["T"])
    rep = os.path.join(d, "derive-report.json")
    if os.path.exists(rep):
        r = json.load(open(rep))
        for k in ("structural_inputs", "shape"):
            si = r.get(k) or (r.get("provenance") or {}).get(k) or {}
            if si.get("LP") is not None and si.get("T") is not None:
                return int(si["LP"]), int(si["T"])
    return None


# ------------------------------------------------------------------------------------------------ driver


# [COORD ruling M-0177 on DECISION M-0021] the within-step criterion of record from r15-int-2: `dag` accepts a request whose projection
# is the same live-cone DAG multiset as its Program although the fold's emission SEQUENCE differs (a co-scheduled prefill displaced
# by a running decode, M-ARRIVAL-ORDER); `sequence` is the r14-v1 criterion (dag accepted for a chunked prefill only). Both facts are
# recorded per request either way (`sequence_equal`, `dag_multiset_equal`); the global checker (GM-01) applies the per-step form and gates.
WITHIN_STEP_CRITERIA = ("sequence", "dag")
DEFAULT_WITHIN_STEP = "dag"

# [R18 x09, F-r18-x09-1] PHASE ACCOUNTING of the per-request leg.  `X09_PROFILE_OUT=<path>` writes one JSON sidecar beside the document
# (never into it: the X-09 document's bytes are the record's) with, per request, wall seconds and counters for every phase of the leg --
# projection build, shape, component load, compare_steps (sequence criterion), and the DAG diagnostic (live_cone A / P, dag_hashes A / P,
# multiset diff) -- plus the sizes that drive them (instances, live-cone sizes, operand runs expanded, per-run fallbacks).  Inert unset.
# `X09_FORCE_DAG=1` (diagnostic ONLY, never on a row of record) runs the DAG diagnostic even when the sequence criterion held, so the
# DAG path's cost can be measured on a PASS record; the verdict fields are computed exactly as before (`canonical_equal` = the same
# expression), the extra `dag_multiset*` fields are then present where the record would have omitted them.
PROFILE_ENV, FORCE_DAG_ENV = "X09_PROFILE_OUT", "X09_FORCE_DAG"
_profile_counters = collections.Counter()   # process-local: per-run fallback / expansion counts (compare.Prog hooks below), per leg


def _profile_on() -> bool:
    return bool(os.environ.get(PROFILE_ENV, "").strip())


def _fast_stats():
    m = sys.modules.get("verity_vllm.check.match.global_match_fast")
    return getattr(m, "stats", None) if m is not None else None


class _Phase:
    """Wall-clock + counter deltas of one named phase of a leg (a context manager)."""

    def __init__(self, prof: dict, name: str):
        self.prof, self.name = prof, name

    def __enter__(self):
        self.t0 = time.time()
        st = _fast_stats()
        self.s0 = dict(st) if st is not None else None
        self.c0 = dict(_profile_counters)
        return self

    def __exit__(self, *exc):
        d = {"seconds": round(time.time() - self.t0, 3)}
        st = _fast_stats()
        if st is not None and self.s0 is not None:
            delta = {k: int(v - self.s0.get(k, 0)) for k, v in st.items() if v - self.s0.get(k, 0)}
            if delta:
                d["fast_stats"] = delta
        cd = {k: int(v - self.c0.get(k, 0)) for k, v in _profile_counters.items() if v - self.c0.get(k, 0)}
        if cd:
            d["counters"] = cd
        self.prof.setdefault("phases", {})[self.name] = d
        return False


def _install_profile_counters() -> None:
    """Count the record's per-run fallbacks / progression expansions inside compare.Prog while profiling (content-neutral wrappers)."""
    if getattr(C.Prog, "_x09_prof_installed", False):
        return
    orig_per_run, orig_srcP, orig_src = C.Prog._per_run, C.Prog._srcP, C.Prog._src

    def _per_run(self, p, unfold):
        _profile_counters["per_run_progressions"] += 1
        _profile_counters["per_run_runs"] += p[-1]
        return orig_per_run(self, p, unfold)

    def _srcP(self, p, unfold):
        _profile_counters["srcP_calls"] += 1
        return orig_srcP(self, p, unfold)

    def _src(self, run, unfold):
        _profile_counters["src_calls"] += 1
        return orig_src(self, run, unfold)

    C.Prog._per_run, C.Prog._srcP, C.Prog._src = _per_run, _srcP, _src
    C.Prog._x09_prof_installed = True


class _NoPhase:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _n_runs_of(arg) -> int:
    """Operand runs of one operand in either form (record: one tuple per run; compact: progressions with a trailing count)."""
    if not arg:
        return 0
    t = arg[0]
    if isinstance(t[1], tuple):   # compact (space, (idx, lo, hi), (dI, dL, dH), count)
        return sum(p[-1] for p in arg)
    return len(arg)


def record_ops() -> Ops:   # the checker of record's operations: X-09's default, GM-01's unless MATCH_IMPL=fast
    return Ops("record", C.load_oracle, C.load_derived, Projection, C.compare_steps, dag_hashes, live_cone)


def decompose(oracle_dir: str, derived_dirs: list[str], log_path: str | None = None, within_step: str = DEFAULT_WITHIN_STEP,
              oracle: "C.Prog | None" = None, components: "dict[str, C.Prog] | None" = None, ops: Ops | None = None) -> dict:
    if within_step not in WITHIN_STEP_CRITERIA:
        raise ValueError(f"within_step must be one of {WITHIN_STEP_CRITERIA}, got {within_step!r}")
    ops = ops or record_ops()
    t0 = time.time()
    profiling = _profile_on()
    force_dag = os.environ.get(FORCE_DAG_ENV, "").strip() == "1"
    run_prof: dict = {"schema": "cb-a/x09-profile/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "oracle_dir": oracle_dir,
                      "force_dag": force_dag, "pid": os.getpid(), "phases": {}, "requests": []}
    if profiling:
        _install_profile_counters()
    log_path = log_path or os.path.join(oracle_dir, "capture", "log.jsonl.gz")
    header, steps = read_steps(log_path)
    reqs, param_of, parent_of = request_params(header)
    # [R15 global, GM-01 memory] the caller (global_match.check) may pass the fold it has already decoded: the fold Prog is read-only
    # after load_oracle and at B16 1024/128 it is ~40 GB of Python objects -- decoding it a second time in the same process doubled
    # GM-01's peak RSS (87 GB on cp-dev1) for nothing
    with _Phase(run_prof, "fold_load") if profiling else _NoPhase():
        B = oracle if oracle is not None else ops.load_oracle(oracle_dir, "oracle")
    if profiling:
        run_prof["fold_preloaded"] = oracle is not None
        run_prof["fold_instances"] = len(B.inst)
        run_prof["fold_compact"] = bool(getattr(B, "compact", False))
    n_in = B.n_input_nodes
    _attr_phase = _Phase(run_prof, "attribution") if profiling else _NoPhase()
    _attr_phase.__enter__()
    # [R15 stoch] a stochastic fold declares one `seed_<i>` parameter per seeded request between the prompts and the weights (fold._header)
    # (`Input64_v1` nodes in request order: the i-th seed node is the i-th request's -- every request of a stochastic row is seeded,
    # derive_step.sampler_geometry refuses otherwise)
    seed_node_of: dict[str, int] = {}
    for k, i in sorted(getattr(B, "seed_index", {}).items()):
        if i >= len(reqs):
            raise ValueError(f"program has {len(B.seed_index)} seed input nodes but the header has {len(reqs)} requests")
        seed_node_of[reqs[i]["request_id"]] = k
    # [R17 D90] a tensor-parallel rank fold declares one `peers_<i>` parameter per request (program.json peers_layout -> load_oracle
    # `peer_index`: node -> request index), request-owned like the prompt; a header/Program disagreement on the count is refused by name
    peer_node_of: dict[str, int] = {}
    for k, i in sorted(getattr(B, "peer_index", {}).items()):
        if i >= len(reqs):
            raise ValueError(f"program has a peers input node for request index {i} but the header has {len(reqs)} requests")
        peer_node_of[reqs[i]["request_id"]] = k
    if peer_node_of and len(peer_node_of) != len(reqs):
        raise ValueError(f"program has {len(peer_node_of)} peers input nodes for {len(reqs)} requests (a TP rank fold declares one per request)")
    # [R16 global, F-fp8-GM1 / M-0823] the shared weights are the Input nodes no request owns -- ONE node for a homogeneous bf16 Record,
    # one per leaf field for a mixed-width one (FP8: 435) -- never `n_input_nodes - 1`; refused by name when the layout does not hold
    weights_nodes = weights_input_nodes(B, len(reqs), len(seed_node_of), len(peer_node_of))
    meta_of = {o["node"] + n_in: o for o in B.oracle_rows}

    # attribution of every instance, in node order (producers precede consumers). Channels, in order:
    #   meta     -- the fold recorded `request` on the instance (fold._compute_emit, X-09+);
    #   dataflow -- every request-owned operand the instance reads (its own prompt parameter / a node attributed earlier) names ONE request
    #               (row independence: an instance over one request's data is that request's; several requests -> non-separable, refused);
    #   weight-only -- [R19, F-r19-int-15] every node operand is a shared weight or an earlier weight-only instance (and there is at least
    #               one): the instance carries no request's data -- gemma2's RMSNorm scale `1 + w` (AddScalarBf16 N=2304, rows=1, per norm
    #               per step; 11235 on #57 B8 1024/128) is the first such node in the table (llama/mistral/olmoe/qwen norms multiply by w
    #               directly).  It is SHARED: owned by no request, read like a weight, and projected into the leg of every request that has
    #               rows in its engine step -- the request's own derive computes the same node inside its forward.  Before this rule the
    #               row-count channel refused it ("rows=1 != tokens|logits") and every norm reading it counted as a cross-request read.
    #   rows     -- no request-owned operand at all: the fold's row-count rule (token rows / logits rows).
    # A per-position Attention group (rows=|positions sharing T|) is the case only dataflow gets right on pre-X-09 folds.
    shared_idx: list[tuple[int, int]] = []   # (instance index, engine step) of every weight-only instance (owner SHARED, module level)
    owners_of_param: dict[int, set[str]] = collections.defaultdict(set)
    for rid, pidx in param_of.items():
        owners_of_param[pidx].add(rid)
    for rid, k in seed_node_of.items():
        owners_of_param[k].add(rid)
    for rid, k in peer_node_of.items():
        owners_of_param[k].add(rid)      # [R17 D90] a collective instance reading request r's peers node is r's (dataflow channel)
    owner_of_node: dict[int, str] = {}
    per_req: dict[str, list[int]] = collections.OrderedDict()
    unattributed: list[dict] = []
    how = collections.Counter()
    disagree = 0
    compact = getattr(B, "compact", False)
    if compact:
        from verity_vllm.program.compact import idx_values as _idx_values
    for i, r in enumerate(B.inst):
        o = meta_of.get(r["k"])
        if o is None:
            unattributed.append({"i": i, "k": r["k"], "spec": r["spec"], "why": "no instances.jsonl row"})
            continue
        rid, h = o.get("request"), "meta"
        if rid is None:
            deps: set = set()
            reads_weight = False        # a shared weight node or an earlier weight-only instance among the operands
            for a in r["args"]:
                for run in a:   # record (space, idx, lo, hi) | compact (space, (idx0, lo, hi), (dI, dL, dH), count): space / idx only
                    if run[0] != "n":
                        continue
                    for idx in (_idx_values(run) if compact else (run[1],)):
                        if idx in B.inputs:
                            if idx not in weights_nodes:
                                deps |= owners_of_param.get(idx, {("?", "param", idx)})
                            else:
                                reads_weight = True
                        else:
                            own = owner_of_node.get(idx, ("?", "node", idx))
                            if own == SHARED:
                                reads_weight = True
                            else:
                                deps.add(own)
            unknown = {d for d in deps if isinstance(d, tuple)}
            owners = deps - unknown
            if unknown:
                h = "reads unattributed " + ", ".join(f"{d[1]} {d[2]}" for d in sorted(unknown))[:200]
            elif len(owners) == 1:
                rid, h = next(iter(owners)), "dataflow"
            elif len(owners) > 1:
                h = "reads several requests: " + ", ".join(sorted(owners))[:200]
            elif reads_weight:
                rid, h = SHARED, "weight-only"
            else:
                st = steps.get(o["step"])
                if st is None:
                    h = f"no StepStart for step {o['step']}"
                else:
                    rid, h = attribute(o, st)
                    h = "rows:" + h
        how[h.split(":")[0] if h.startswith("rows") else h.split(" ")[0]] += 1
        if rid is None:
            unattributed.append({"i": i, "k": r["k"], "spec": r["spec"], "step": o["step"], "row": o["row"], "rows": o["rows"], "why": h})
            continue
        if rid == SHARED:
            owner_of_node[r["k"]] = SHARED
            shared_idx.append((i, o["step"]))
            continue
        # cross-check against the row-count rule whenever it applies unambiguously (diagnostic; the dataflow channel is authoritative)
        st = steps.get(o["step"])
        if st is not None and h != "rows":
            rr, hh = attribute(o, st)
            if rr is not None and rr != rid and hh in ("token", "logits"):
                disagree += 1
        owner_of_node[r["k"]] = rid
        per_req.setdefault(rid, []).append(i)
    _attr_phase.__exit__(None, None, None)

    # derived index by (LP, T)
    derived: dict[tuple[int, int], str] = {}
    bad_derived = []
    for d in derived_dirs:
        sh = derived_shape(d)
        if sh is None:
            bad_derived.append(d)
        else:
            derived.setdefault(sh, d)
    A_cache: dict[tuple[int, int], C.Prog] = {}
    components = components if components is not None else {}  # decoded component Progs keyed by dir, shared with the caller

    res = {"schema": "cb-a/batch-decomp/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "oracle_dir": oracle_dir,
           "log": log_path, "n_requests_header": len(reqs), "n_output_sequences": len(per_req), "engine_steps": len(steps),
           "oracle": {"digest": B.digest, "instances": len(B.inst), "n_input_nodes": n_in,
                      # [R16 global, F-fp8-GM1] the shared weights as a node RANGE (contiguous after the request-owned parameters)
                      "weights_input_nodes": {"n": len(weights_nodes), "first": min(weights_nodes), "last": max(weights_nodes),
                                              "request_owned_inputs": len(reqs) + len(seed_node_of)},
                      "weight_map": {"n_ranges": len(B.oracle_wmap), "n_overlaps": len(B.oracle_wmap_overlaps)}},
           "attribution": {"how": dict(how), "unattributed": len(unattributed), "unattributed_examples": unattributed[:12],
                           "attributed": sum(len(v) for v in per_req.values()), "row_rule_disagreements": disagree,
                           # [R19, F-r19-int-15] weight-only instances: owned by no request, projected into every participating leg
                           "weight_only_shared": len(shared_idx),
                           "weight_only_specs": dict(collections.Counter(B.inst[i]["spec"].split("{")[0] for i, _ in shared_idx).most_common(6))},
           "derived_available": {f"LP{lp}_T{t}": d for (lp, t), d in sorted(derived.items())}, "derived_unreadable": bad_derived,
           "requests": [], "missing_derived": []}
    missing: set[tuple[int, int]] = set()
    all_equal = True

    def shape_of(rid: str, idxs: list[int]):
        """(LP, T) of a request from the fold alone (the leg recomputes the same; here to preload its component before the legs fork)."""
        pidx = param_of.get(rid)
        if pidx is None:
            return None
        LP = len(reqs[pidx]["prompt_token_ids"])
        n_ts = sum(1 for i in idxs if _is_sampling_event(B.inst[i]["spec"]))
        chunks = max(1, sum(1 for s, st in steps.items() if any(r == rid and pos < LP for r, pos in st.get("rows", []))))
        return (LP, n_ts - chunks)

    def leg(item) -> tuple:
        """[R17 mfast] X-09 for ONE request: its Projection of the fold, the comparison with its component (sequence + DAG criteria),
        its cross-request reads.  Reads the shared fold / components / attribution, writes nothing shared: the record's loop body,
        returning (entry, missing shape | None) instead of appending.  Runs in this process, or on a forked worker (fast path)."""
        rid, idxs = item
        pidx = param_of.get(rid)
        ent = {"request_id": rid, "parent": parent_of.get(rid), "prompt_param": pidx, "n_instances": len(idxs)}
        prof: dict = {"request_id": rid, "n_instances": len(idxs), "pid": os.getpid(), "t_start": time.time()} if profiling else None
        if prof is not None:
            ent["_profile"] = prof
        if pidx is None:
            ent["error"] = "request not in RunHeader.workload (no prompt parameter)"
            return ent, None
        LP = len(reqs[pidx]["prompt_token_ids"])
        with _Phase(prof, "projection_build") if prof is not None else _NoPhase():
            P = ops.projection(B, f"oracle[{rid}]", idxs, tok_node=pidx, weights_node=weights_nodes, seed_node=seed_node_of.get(parent_of.get(rid, rid)),
                               peer_node=peer_node_of.get(parent_of.get(rid, rid)))
            P._all_seed_nodes = set(seed_node_of.values())
        with _Phase(prof, "shape") if prof is not None else _NoPhase():
            n_ts = sum(1 for r in P.inst if _is_sampling_event(r["spec"]))
            # prefill chunks: engine steps in which the request had PROMPT rows (position < LP). vLLM v1 samples (and discards) a token
            # after every partial chunk, so the fold holds one TokenSelect per chunk + one per served decode step: T = n_ts - chunks.
            prefill_steps = sorted(s for s, st in steps.items() if any(r == rid and pos < LP for r, pos in st.get("rows", [])))
            chunks = max(1, len(prefill_steps))
            T = n_ts - chunks
            es = engine_steps_of(P, meta_of)
        ent.update({"LP": LP, "T": T, "token_selects": n_ts, "steps": len(P.steps), "engine_steps_per_step": es[:4] + (["..."] if len(es) > 4 else []),
                    "prefill_chunks": chunks, "prefill_engine_steps": prefill_steps[:8],
                    "chunked_steps": sum(1 for e in es if len(e) != 1), "engine_step_range": [min(min(e) for e in es if e), max(max(e) for e in es if e)] if es else None})
        if prof is not None:
            prof.update({"LP": LP, "T": T, "steps": len(P.steps), "prefill_chunks": chunks,
                         "p_runs_total": sum(_n_runs_of(a) for r in P.inst for a in r["args"]) if len(P.inst) <= 2_000_000 else None})
        if n_ts == 0:
            ent["error"] = "no TokenSelect attributed to the request"
            return ent, None
        d = derived.get((LP, T))
        if d is None:
            ent["derived"] = None
            return ent, (LP, T)
        with _Phase(prof, "component_load") if prof is not None else _NoPhase():
            A = A_cache.get((LP, T))
            if A is None:
                A = components.get(d)
                if A is None:
                    A = components[d] = ops.load_derived(d, f"derived[LP{LP}_T{T}]")
                A_cache[(LP, T)] = A
        if prof is not None:
            prof.update({"a_instances": len(A.inst), "a_steps": len(A.steps),
                         "a_runs_total": sum(_n_runs_of(a) for r in A.inst for a in r["args"]) if len(A.inst) <= 2_000_000 else None})
        with _Phase(prof, "compare_steps") if prof is not None else _NoPhase():
            cmp = ops.compare_steps(A, P, unfold_a=False)
        ent["derived"] = d
        ent["derived_digest"] = A.digest
        ent["xreq"] = {"counts": dict(P.xreq), "examples": P.xreq_examples}
        seq_equal = bool(cmp["canonical_sha256"]["equal"]) and not P.xreq
        ent["sequence_equal"] = seq_equal
        ent["canonical_sha256"] = cmp["canonical_sha256"]
        ent["criterion"] = "sequence"
        chunked = chunks > 1
        if prof is not None:
            prof.update({"sequence_equal": seq_equal, "xreq_total": sum(P.xreq.values()),
                         "steps_spec_equal": cmp["totals"].get("steps_spec_equal", 0), "cmp_steps": len(cmp["steps"])})
        if (not seq_equal and not P.xreq) or (force_dag and not P.xreq):
            # Order-insensitive DAG form over the LIVE cone (see dag_multiset / live_cone). It is the ACCEPTED criterion only for a
            # request whose prefill the engine chunked: the fold is then chunk-major and carries one discarded sample per partial
            # chunk, so the sequence criterion cannot hold by construction. Otherwise it is reported as a diagnostic only.
            with _Phase(prof, "live_cone_A") if prof is not None else _NoPhase():
                la = ops.live_cone(A)
            with _Phase(prof, "live_cone_P") if prof is not None else _NoPhase():
                lb = ops.live_cone(P)
            with _Phase(prof, "dag_hashes_A") if prof is not None else _NoPhase():
                ha = ops.dag_hashes(A, la)
            with _Phase(prof, "dag_hashes_P") if prof is not None else _NoPhase():
                hb = ops.dag_hashes(P, lb)
            with _Phase(prof, "multiset_diff") if prof is not None else _NoPhase():
                ma, mb = collections.Counter(ha.values()), collections.Counter(hb.values())
                only_a, only_b = ma - mb, mb - ma
                diff = only_a + only_b
                spec = lambda Q, i: Q.inst[i]["spec"].split("{")[0]  # noqa: E731
                ent["dag_multiset_equal"] = not diff
                ent["dag_multiset"] = {
                    "a_live": len(la), "a_dead": len(A.inst) - len(la), "b_live": len(lb), "b_dead": len(P.inst) - len(lb),
                    "a_dead_specs": dict(collections.Counter(spec(A, i) for i in range(len(A.inst)) if i not in la)),
                    "b_dead_specs": dict(collections.Counter(spec(P, i) for i in range(len(P.inst)) if i not in lb)),
                    "only_a": sum(only_a.values()), "only_b": sum(only_b.values()),
                    "only_a_specs": dict(collections.Counter(spec(A, i) for i, h in ha.items() if h in only_a).most_common(6)),
                    "only_b_specs": dict(collections.Counter(spec(P, i) for i, h in hb.items() if h in only_b).most_common(6))}
            if prof is not None:
                prof.update({"a_live": len(la), "b_live": len(lb), "dag_only_a": sum(only_a.values()), "dag_only_b": sum(only_b.values())})
            if not seq_equal and not diff and chunked:
                ent["criterion"] = "dag-multiset over the live cone (chunked prefill)"
            elif not seq_equal and not diff and within_step == "dag":
                ent["criterion"] = "dag-multiset over the live cone (within-step criterion of record, COORD M-0177; sequence differs)"
        ent["canonical_equal"] = seq_equal or (ent.get("dag_multiset_equal") is True and (chunked or within_step == "dag"))
        ent["totals"] = cmp["totals"]
        ent["a_steps"], ent["b_steps"] = cmp["a"]["steps"], cmp["b"]["steps"]
        bad = [s for s in cmp["steps"] if not s["spec_seq_equal"] or (s.get("wiring") or {}).get("operands_different")]
        ent["first_divergences"] = [{"step": s["step"], "n_a": s["n_a"], "n_b": s["n_b"], "div": s.get("first_divergences_T_normalised", [])[:2],
                                     "wiring_examples": (s.get("wiring") or {}).get("examples", [])[:3]} for s in bad[:3]]
        ent["n_divergent_steps"] = len(bad)
        if prof is not None:
            prof["seconds"] = round(time.time() - prof["t_start"], 3)
        return ent, None

    if shared_idx:
        # [R19, F-r19-int-15] a weight-only instance of engine step s belongs to the projection of every request with rows in s (the
        # request's own derive computes that node inside its forward at that step); a request with no rows in s does not see it.
        # Legs stay in request order; each leg's indices stay in node order (producers precede consumers).  GM-01 builds the same legs.
        items = list(shared_legs(per_req, shared_idx, steps).items())
    else:
        items = list(per_req.items())
    if ops.legs is not None:
        # [R17 mfast, BRIEF §5 step 3] the per-request comparisons are separable once attribution is done (above, single and exact): the
        # fast path runs the legs on forked workers over the SAME fold / components (copy-on-write) -- every component is loaded here
        # first so no worker loads its own copy; the aggregation below is the record's loop, in request order
        to_load: list[tuple[str, str]] = []
        for rid, idxs in items:
            sh = shape_of(rid, idxs)
            d = derived.get(sh) if sh is not None else None
            if d is not None and d not in components and all(d != x[0] for x in to_load):
                to_load.append((d, f"derived[LP{sh[0]}_T{sh[1]}]"))
        if to_load:
            # the components are built on the workers too, one per leg (2 s to build vs 0.1 s to pickle a component back: the build is
            # the descriptor walk), then held by the parent so the legs below share them copy-on-write; a build failure raises, as the
            # record's load in the loop does
            components.update(zip((d for d, _ in to_load), ops.legs(to_load, lambda it: ops.load_derived(*it), lambda it: 1)))
        with _Phase(run_prof, "legs") if profiling else _NoPhase():
            outs = ops.legs(items, leg, lambda it: len(it[1]))
    else:
        with _Phase(run_prof, "legs") if profiling else _NoPhase():
            outs = [leg(it) for it in items]
    for ent, miss in outs:
        if profiling:
            p = ent.pop("_profile", None)   # the sidecar's, never the document's
            if p is not None:
                run_prof["requests"].append(p)
        else:
            ent.pop("_profile", None)
        if miss is not None:
            missing.add(miss)
        elif ent.get("error"):
            all_equal = False
        elif ent.get("derived") is not None:
            all_equal = all_equal and ent["canonical_equal"]
        res["requests"].append(ent)
    res["within_step_criterion"] = within_step
    res["missing_derived"] = [[lp, t] for lp, t in sorted(missing)]
    res["xreq_total"] = sum(sum(e.get("xreq", {}).get("counts", {}).values()) for e in res["requests"])
    complete = not unattributed and not missing and not bad_derived
    res["verdict"] = "PASS" if (complete and all_equal and per_req) else ("MISSING_DERIVED" if missing and not unattributed else "FAIL")
    res["reasons"] = ([f"{len(unattributed)} unattributed instances"] if unattributed else []) + \
                     ([f"missing derived Programs for {res['missing_derived']}"] if missing else []) + \
                     ([f"{len(bad_derived)} derived dir(s) unreadable: {[os.path.basename(x) for x in bad_derived[:3]]}"] if bad_derived else []) + \
                     ([f"{res['xreq_total']} cross-request operand reads"] if res["xreq_total"] else []) + \
                     [f"{e['request_id']}: not canonical-equal" for e in res["requests"] if e.get("canonical_equal") is False and not e.get("xreq", {}).get("counts")] + \
                     [f"{e['request_id']}: {e['error']}" for e in res["requests"] if e.get("error")]
    res["seconds"] = round(time.time() - t0, 1)
    if profiling:
        run_prof["seconds"] = res["seconds"]
        run_prof["verdict"] = res["verdict"]
        run_prof["x09_legs"] = "forked" if ops.legs is not None else "inline"
        st = _fast_stats()
        if st is not None:
            run_prof["fast_stats_final"] = {k: int(v) for k, v in sorted(st.items())}
        try:
            import resource
            run_prof["max_rss_mib_parent"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
            run_prof["max_rss_mib_children"] = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024)
        except Exception:   # noqa: BLE001 - telemetry only
            pass
        out = os.environ[PROFILE_ENV]
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        json.dump(run_prof, open(out, "w"), indent=1, default=str)
    return res


def main(argv=None) -> int:
    from verity_vllm.pipeline.gc_tuning import tune as _gc_tune   # process-level cyclic-GC thresholds; content-neutral (VERITY_GC_THRESHOLDS)
    _gc_tune()
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True, help="match dir: program.json + instances.jsonl (+ capture/log.jsonl.gz)")
    ap.add_argument("--derived", action="append", default=[], help="derived request-wrapper dir (repeatable; shape read from artifact.json)")
    ap.add_argument("--log", default=None, help="capture log (default <oracle>/capture/log.jsonl.gz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--within-step", choices=WITHIN_STEP_CRITERIA, default=DEFAULT_WITHIN_STEP,
                    help="within-step criterion of record: dag (COORD M-0177; live-cone DAG multiset accepted for every request) | sequence (r14-v1)")
    a = ap.parse_args(argv)
    res = decompose(a.oracle, a.derived, a.log, within_step=a.within_step)
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    brief = {"verdict": res["verdict"], "within_step": res["within_step_criterion"], "reasons": res["reasons"],
             "requests": [{k: e.get(k) for k in ("request_id", "LP", "T", "n_instances", "steps", "canonical_equal", "sequence_equal", "dag_multiset_equal", "criterion", "chunked_steps")}
                                                                                 for e in res["requests"]],
             "attribution": {k: v for k, v in res["attribution"].items() if k != "unattributed_examples"}, "missing_derived": res["missing_derived"],
             "xreq_total": res["xreq_total"], "seconds": res["seconds"]}
    print(json.dumps(brief, indent=1, default=str))
    return 0 if res["verdict"] == "PASS" else (4 if res["verdict"] == "MISSING_DERIVED" else 3)


if __name__ == "__main__":
    sys.exit(main())
