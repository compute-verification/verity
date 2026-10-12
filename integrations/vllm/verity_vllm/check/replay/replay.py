"""verity_vllm/check/replay/replay.py — the derived Program as its own oracle.

The fold produces an *executable* Program (every instance has an exact body), so its correctness is checked by
replaying it against the evidence L1 snapshotted, in three tiers.  Each tier returns a JSON-able dict with a
tier-level `status` in {"pass", "fail", "skipped", "error"}; per-instance entries carry the same four statuses plus
"computed" (evaluated, but no snapshot of the output exists to compare with — normal in tiers B/C).

    uv run verity-vllm replay --log data/logs/m1.jsonl.gz --snapshots data/logs/m1 \
        --profile vllm_d9105ea80_sm89_eager --tier a --k 3 --seed S [--jobs N] --out out/capture/reports/replay_a.json
    ... --tier b --step 1 --max-arg-leaves 4000000 [--layers 0,1,2] [--max-instances N] [--evaluator gates|twins|auto]
    ... --tier c                                     # pod only: hours for a real model

A. `tier_a` — **body exactness**.  For every specialization present in the Program, up to `k` instances (seeded
   uniform sample over the snapshot steps) are evaluated exactly as `compare.compare_subcircuits` does: operands by
   provenance (`compare.Values`: prompt / checkpoint / the producer's output snapshot), expected outputs from the
   instance's own event snapshot, `verity.ml.circuits.evaluate.evaluate_call`, bit-exact compare.  An instance whose
   argument leaves exceed `max_arg_leaves` (Embedding and the lm_head Gemm: 28M-leaf weight operands) is evaluated
   at a *child unit* found by descending with `FunctionCall`/`Scope.child` — one `GemmCoordinate`, one gather gate —
   whose argument gates are looked up in the parent's arguments by gate index (only the needed leaves are
   resolved) and whose return gates are a subset of the parent's return gates (expected values by gate index).
   `status` is "pass" iff every spec has >= min(k, evaluable) passing evaluations and no fail/error; a spec with no
   evaluable instance (no snapshot step contains it, oversize with no fitting child, operands whose producers were
   never snapshotted) makes it "fail" and is listed in `uncovered_specs` with the reason.
B. `tier_b` — **wiring exactness of one step**.  Every instance of `step` is replayed in node order; an operand
   produced by an instance already replayed comes from the *computed* table (the wiring under test), otherwise from
   the producer's snapshot (earlier steps; an instance skipped under the budget), parameters from prompt/checkpoint.
   Outputs are compared with the snapshot when one exists; the first divergence is recorded and the computed value
   keeps flowing (a body or memory-model error shows as one localized divergence and its cascade).  Skips under the
   explicit budgets (`max_arg_leaves`, `max_instances`, `layers`) do not fail the tier; consumers of a skipped
   producer fall back to its snapshot and the report is flagged `partial`.  Skips for missing evidence (a producer
   from a step without snapshots) make the status "skipped".
C. `tier_c` — **end to end, pod only**.  The same engine over all steps with NO interior snapshots: every operand
   comes from the computed table or the parameters; the next step's token is the computed TokenSelect of the
   previous step (checked against `result.sampled`; on a mismatch the computed token is reported and kept).  Hours
   on a CPU pod for a real model (the laptop measures ~40 us per gate: ~5 min per SmolLM2 decode step, ~3 h for the
   32-row prefill); tested here on a synthetic toy only.

Resource discipline (DESIGN §11): one `Replayer` holds one instance's transcript at a time; `computed` outputs are
freed once their last consumer (in body order) has been replayed.  Measured on the laptop (2026-09-16), M1: fold +
Program 2.1 s / 0.19 GB, plus 0.51 GB once per process when the first RMSNorm gate lazily loads the MUFU tables
(`verity_vllm.program.kernels.rms_relation.tables`); a decode-step instance costs 0.01 s (RoPE, norms) to 5.2 s
(`Gemm_v1{K=576,N=3072}`); tier A k=2 30 s / 1.0 GB, tier B on 2 of 30 layers of a decode step 23 s / 0.85 GB
(`max_rss_gb` in the reports is the process high-water mark, that baseline included).  Torch-free.  Words are
encodings (Python ints of the bit pattern) exactly as `compare_subcircuits` passes them.

Evaluators (h4).  `--evaluator gates|twins|auto` (default `auto`) selects how tiers B and C evaluate an instance:
`gates` is `evaluate_call` on the registered body; `twins` uses the twin kernel of `verity_vllm.program.kernels.twins`
wherever one claims the specialization; `auto` uses a twin only after core `self_check` passed on >= 3 random instances
of the specialization, else the gates.  A twin that declines an operand (non-finite words outside its exact domain)
falls back to the gates for that instance.  Every per-instance entry records `evaluator` ("gates" | "twins") and
`twin_id`; step reports carry `evaluators: {twins, gates, self_check}`.  Tier A always uses the gates (it is the
body-exactness tier).  Pass/fail, `partial`/`skipped` and every other report field are unchanged by the choice of evaluator.
"""
from __future__ import annotations

import glob
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from dataclasses import dataclass

from verity.ml.circuits.evaluate import evaluate_call
from verity_vllm.config import UsageError, option
from verity_vllm.program.frontend.rules.family import _field_intervals
from verity.ml.circuits.layout import FunctionCall
from verity.ml.circuits.refs import runs

from verity_vllm.check.fold_compare import SnapshotStore, Values, WeightSource, _attach_specs, _component_ranges, _jsonable, _max_rss_gb, fold_log, load_profile
from verity_vllm.check.replay.tier_a_pool import _distinct_step_order, evaluate_strata
from verity_vllm.observe.fold.fold import FoldResult
from verity_vllm.commit import challenge as CH

TIER_STATUSES = ("pass", "fail", "skipped", "error")
EVALUATORS = ("gates", "twins", "auto")
_LAYER_RE = re.compile(r"layers\.(\d+)")


class Skipped(LookupError):
    """An evaluation that cannot be performed: `budget=True` for an explicit budget (`max_arg_leaves`, `max_instances`,
    `layers`), else missing evidence (no snapshot / no weight source)."""

    def __init__(self, reason: str, budget: bool = False) -> None:
        super().__init__(reason)
        self.reason, self.budget = reason, budget


def layer_of(hint: str | None) -> int | None:
    m = _LAYER_RE.search(hint or "")
    return int(m.group(1)) if m else None


def _diff(got: np.ndarray, expect: np.ndarray) -> tuple[int, list[int]]:
    if got.size != expect.size:
        raise ValueError(f"{got.size} return encodings vs {expect.size} snapshot words")
    neq = np.nonzero(got != expect)[0]
    return int(neq.size), [int(i) for i in neq[:4]]


# ---------------------------------------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------------------------------------


class Replayer:
    """Evaluates root instances of a folded Program with the reference evaluator.

    `chain=True` keeps a table `computed[node] -> output words` and feeds it to consumers (tiers B/C);
    `snapshots=True` lets an operand whose producer is not in the table come from the producer's output snapshot
    (`compare.Values`); parameters always come from the prompt / checkpoint.  `scope` (a node range) tells which
    producers belong to the current replay: a fallback to the snapshot of a producer *inside* the scope means the
    replay is `partial` (the producer was skipped), one *outside* is an ordinary boundary from an earlier step."""

    def __init__(self, result: FoldResult, store: SnapshotStore | None = None, weights: WeightSource | None = None, *,
                 chain: bool = True, snapshots: bool = True, max_arg_leaves: int | None = None,
                 evaluator: str = "auto", self_check_n: int = 3, seed: int = 0) -> None:
        if result.program is None:
            raise ValueError("fold ran without Program(); replay needs the circuit")
        if evaluator not in EVALUATORS:
            raise ValueError(f"evaluator {evaluator!r} not in {EVALUATORS}")
        _attach_specs(result)
        self.r = result
        self.store, self.weights = store, weights
        self.chain, self.snapshots = chain, bool(snapshots and store is not None)
        self.max_arg_leaves = max_arg_leaves
        self.evaluator = evaluator
        self.self_check_n, self.seed = self_check_n, seed
        self._twins: dict[str, Any] = {}                 # spec id -> admitted twin (None: this spec uses the gates)
        self.self_checks: dict[str, dict] = {}           # spec id -> the self-check record that admitted / rejected its twin
        self.evaluator_counts: Counter = Counter()       # "twins" / "gates" over the instances evaluated by `replay_node`
        if evaluator != "gates" and chain:
            from verity_vllm.program.kernels.twins import preload
            preload()                                    # tables + models before the per-instance allocations (tiers B/C)
        self.vals = Values(result, store, weights)
        self.by_node: dict[int, dict] = self.vals.by_node
        self.program = result.program
        self.circuit = self.program.circuit
        self.root = self.program.root
        self.n_in = len(self.root.body.nodes) - len(result.body.nodes)
        self.W = len(result.params) - 1
        self.computed: dict[int, np.ndarray] = {}
        self.tokens: dict[int, int] = {}                 # TokenSelect node -> computed token
        self.skipped_nodes: set[int] = set()
        self.scope: tuple[int, int] = (0, len(result.body.nodes))
        self._frees_at: dict[int, list[int]] | None = None
        self._ret_index: dict[int, dict[int, int]] = {}
        self._field_cache: dict[int, np.ndarray] = {}
        self._out_cache: dict[int, np.ndarray] = {}
        self._root_offsets = np.array(self.root.body.offsets, dtype=np.int64)
        self._input_intervals = self._input_gate_intervals()
        # leaf-typed fields of the weights Record: (lo, hi, name) in leaf order (any weight source with `words(lo, hi)`)
        self._weight_intervals = _field_intervals(result.params[-1][1], "")
        self._weight_starts = np.array([lo for lo, _, _ in self._weight_intervals], dtype=np.int64)
        self.output_nodes: dict[int, tuple[str, int]] = {}   # TokenSelect node -> (request, step) from the fold's outputs
        for rid, seq in result.outputs.items():
            steps = [s for s, _ in result.sampled.get(rid, [])]
            for j, refs in enumerate(seq):
                for space, idx, _, _ in runs(refs):
                    if space == "n" and j < len(steps):
                        self.output_nodes[idx] = (rid, steps[j])

    # -- structure ----------------------------------------------------------------------------------------

    def _input_gate_intervals(self) -> list[tuple[int, int, int, int]]:
        """(gate_lo, gate_hi, param, first leaf) of the Input gates, per parameter, in leaf order."""
        out = []
        for pi in range(len(self.r.params)):
            off = 0
            for space, _, lo, hi in runs(self.program.input_coll(pi).refs):
                if space != "g":
                    raise ValueError(f"input collection of parameter {pi} is not over gates ({space!r})")
                out.append((lo, hi, pi, off))
                off += hi - lo
        return out

    def name(self, k: int) -> str:
        m = self.by_node.get(k)
        return m["name"] if m else f"node {k}"

    def call(self, k: int) -> FunctionCall:
        return FunctionCall(self.circuit, self.circuit.root_scope, self.n_in + k)

    def arg_leaves(self, k: int) -> int:
        return sum(a.refs.count for a in self.r.body.nodes[k].args)

    def out_leaves(self, k: int) -> int:
        return sum(hi - lo for lo, hi in _component_ranges(self.by_node[k]))

    def ret_index(self, k: int) -> dict[int, int]:
        """Gate -> output leaf position of root instance `k` (its return gates in signature order)."""
        d = self._ret_index.get(k)
        if d is None:
            d = {g: j for j, g in enumerate(self.call(k).returns)}
            self._ret_index[k] = d
        return d

    def frees_at(self) -> dict[int, list[int]]:
        """node -> producers whose last consumer (in body order) is that node; the body's return refs are never freed."""
        if self._frees_at is None:
            last: dict[int, int] = {}
            for j, node in enumerate(self.r.body.nodes):
                for a in node.args:
                    for space, idx, _, _ in runs(a.refs):
                        if space == "n":
                            last[idx] = j
            for space, idx, _, _ in runs(self.r.body.ret):
                if space == "n":
                    last.pop(idx, None)
            fa: dict[int, list[int]] = defaultdict(list)
            for idx, j in last.items():
                fa[j].append(idx)
            self._frees_at = dict(fa)
        return self._frees_at

    def release(self, k: int) -> None:
        """Free computed outputs whose consumers have all been replayed (called after node `k`)."""
        for j in self.frees_at().get(k, ()):
            self.computed.pop(j, None)

    # -- values ---------------------------------------------------------------------------------------------

    def operand_words(self, k: int) -> tuple[list[int], dict]:
        """Argument encodings of root instance `k` in signature order; `info` lists the producers that were not in
        the computed table: `fallback` (inside the scope: partial) and `boundary` (outside: an earlier step)."""
        arrays, info = self.operand_arrays(k)
        words = np.concatenate([a.astype(np.uint64, copy=False) for a in arrays]) if arrays else np.zeros(0, dtype=np.uint64)
        return words.tolist(), info

    def operand_arrays(self, k: int) -> tuple[list[np.ndarray], dict]:
        """`operand_words` before flattening: one unsigned-integer array per signature argument (its reference runs
        concatenated, native word width — uint16 bf16 / uint32 / uint64 from the computed table), same `info`."""
        node = self.r.body.nodes[k]
        arrays: list[np.ndarray] = []
        info: dict[str, list[int]] = {"fallback": [], "boundary": []}
        lo_s, hi_s = self.scope
        for a in node.args:
            parts: list[np.ndarray] = []
            for space, idx, lo, hi in runs(a.refs):
                if space == "n":
                    w = self.computed.get(idx) if self.chain else None
                    if w is None:
                        pm = self.by_node.get(idx)
                        where = f"node {idx} ({pm['name']}, step {pm['step']})" if pm else f"node {idx}"
                        if not self.snapshots:
                            raise Skipped(f"operand from {where} is not computed and there are no interior snapshots")
                        try:
                            w = np.asarray(self.vals.of_run("n", idx, lo, hi))
                        except LookupError as e:
                            raise Skipped(f"operand from {where} has no snapshot: {e}") from None
                        (info["fallback"] if lo_s <= idx < hi_s and self.chain else info["boundary"]).append(idx)
                    else:
                        w = w[lo:hi]
                else:
                    try:
                        w = np.asarray(self.vals.of_run(space, idx, lo, hi))
                    except LookupError as e:
                        raise Skipped(f"parameter {idx} leaves [{lo}, {hi}): {e}") from None
                parts.append(w)
            if len(parts) == 1:
                arrays.append(parts[0])
            else:
                arrays.append(np.concatenate([p.astype(np.uint64, copy=False) for p in parts]) if parts else np.zeros(0, dtype=np.uint64))
        info["fallback"] = sorted(set(info["fallback"]))
        info["boundary"] = sorted(set(info["boundary"]))
        return arrays, info

    def expected_words(self, meta: dict) -> np.ndarray | None:
        """Snapshot words of the instance's outputs (its own event, after the call), None when no snapshot exists."""
        if self.store is None:
            return None
        out = []
        for j, view in enumerate(meta["output_views"]):
            if view is None:
                raise Skipped(f"output {j} view unknown")
            s = self.store.find(view, at=meta["event"])
            if s is None:
                return None
            out.append(self.store.extract(s, view).astype(np.uint64))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.uint64)

    def evaluate(self, k: int, args: list[int]) -> np.ndarray:
        """`evaluate_call` of root instance `k` on argument encodings; returns the return encodings."""
        got = evaluate_call(self.circuit, self.call(k), args)
        return np.array(got, dtype=np.uint64)

    def twin_of(self, k: int):
        """The admitted twin (`verity_vllm.program.kernels.twins`) of root instance `k`'s specialization as (twin_id, its
        instance form, the statics), or None (the gates), per `self.evaluator`.  The first instance of a spec runs the
        twin's self-check (`twins.admit`) and records it in `self.self_checks`; `auto` falls back to the gates on a
        failed check, `twins` raises for every instance of that spec (loud), `gates` never uses a twin."""
        if self.evaluator == "gates":
            return None
        m = self.by_node[k]
        sid = m["spec"]
        if sid in self._twins:
            tw = self._twins[sid]
            if isinstance(tw, str):
                raise RuntimeError(tw)
            return tw
        from verity_vllm.program.kernels import kernel_registry as KR, twins
        spec, tw = m["_spec"], None
        if KR.covers(spec, twins.KERNEL) is None:
            rec = twins.admit(spec, n=self.self_check_n, seed=self.seed)
            self.self_checks[sid] = rec
            if rec["status"] == "pass":
                tw = (rec["twin_id"], KR.instance(spec, twins.KERNEL), dict(spec.statics))
            elif self.evaluator == "twins":
                self._twins[sid] = f"twin {rec['twin_id']} of {sid} failed its self-check ({rec['status']}): {rec['mismatches'][:1]}"
                raise RuntimeError(self._twins[sid])
        self._twins[sid] = tw
        return tw

    def evaluate_arrays(self, k: int, arrays: list[np.ndarray]) -> tuple[np.ndarray, dict]:
        """`evaluate` from `operand_arrays` through the selected evaluator: the admitted twin when it exists and claims
        the operands, else the gates.  Returns (return encodings, {"evaluator": "twins"|"gates", "twin_id"[, "declined"]})."""
        tw = self.twin_of(k)
        ev: dict[str, Any] = {"evaluator": "gates", "twin_id": tw[0] if tw is not None else None}
        if tw is not None:
            from verity_vllm.program.kernels.kernel_registry import Declined
            try:
                got = np.asarray(tw[1](*arrays, **tw[2]), dtype=np.uint64)
            except Declined as e:
                ev["declined"] = str(e)
            else:
                ev["evaluator"] = "twins"
                self.evaluator_counts["twins"] += 1
                return got, ev
        words = np.concatenate([a.astype(np.uint64, copy=False) for a in arrays]).tolist() if arrays else []
        self.evaluator_counts["gates"] += 1
        return self.evaluate(k, words), ev

    def output_words(self, k: int) -> np.ndarray:
        """All output words of root instance `k`: computed, else its snapshot."""
        w = self.computed.get(k) if self.chain else None
        if w is None:
            w = self._out_cache.get(k)
        if w is None:
            try:
                w = np.asarray(self.vals.of_run("n", k, 0, self.out_leaves(k))).astype(np.uint64)
            except LookupError as e:
                raise Skipped(f"outputs of node {k} ({self.name(k)}, step {self.by_node[k]['step']}) have no snapshot: {e}") from None
            self._out_cache[k] = w
        return w

    def _weight_words_at(self, leaves: np.ndarray) -> np.ndarray:
        if self.weights is None or not self.weights.available():
            raise Skipped("no weight source (checkpoint safetensors / cos_sin fixture)")
        iv = self._weight_intervals
        ii = np.searchsorted(self._weight_starts, leaves, side="right") - 1
        out = np.empty(leaves.size, dtype=np.uint64)
        for i in np.unique(ii):
            sel = ii == i
            flo, fhi, _ = iv[int(i)]
            w = self._field_cache.get(int(i))
            if w is None:
                w = self.weights.words(flo, fhi)
                self._field_cache[int(i)] = w
            rel = leaves[sel] - flo
            if rel.min() < 0 or rel.max() >= w.size:
                raise Skipped(f"weight leaves outside field {iv[int(i)][2]}")
            out[sel] = w[rel]
        return out

    def _param_words_at(self, pi: int, leaves: np.ndarray) -> np.ndarray:
        if pi == self.W:
            return self._weight_words_at(leaves)
        toks = self.r.requests[pi]["prompt_token_ids"]
        return np.array([int(toks[int(l)]) & 0xFFFFFFFF for l in leaves], dtype=np.uint64)

    def gate_words(self, gates: Iterable[int]) -> list[int]:
        """Encodings of root gates: Input gates from the parameters, other gates from the producing root instance's
        outputs (computed table, else snapshot).  Resolves only the gates asked for (a child's arguments), never a
        whole weight operand."""
        g = np.fromiter((int(x) for x in gates), dtype=np.int64)
        out = np.empty(g.size, dtype=np.uint64)
        if g.size == 0:
            return []
        ks = np.searchsorted(self._root_offsets, g, side="right") - 1
        for kk in np.unique(ks):
            sel = ks == kk
            gs = g[sel]
            if kk < self.n_in:
                vals = np.empty(gs.size, dtype=np.uint64)
                done = np.zeros(gs.size, dtype=bool)
                for glo, ghi, pi, leaf0 in self._input_intervals:
                    m = (gs >= glo) & (gs < ghi)
                    if m.any():
                        vals[m] = self._param_words_at(pi, gs[m] - glo + leaf0)
                        done |= m
                if not done.all():
                    raise ValueError(f"gate {int(gs[~done][0])} is an Input gate of no parameter")
                out[sel] = vals
            else:
                flat = int(kk) - self.n_in
                rix = self.ret_index(flat)
                try:
                    pos = np.array([rix[int(x)] for x in gs], dtype=np.int64)
                except KeyError as e:
                    raise Skipped(f"gate {e.args[0]} is interior to node {flat} ({self.name(flat)}), not one of its outputs") from None
                out[sel] = self.output_words(flat)[pos]
        return out.tolist()

    # -- one instance ---------------------------------------------------------------------------------------

    def replay_node(self, k: int, *, compare: bool = True) -> dict:
        """Tier B/C step for root instance `k`: operands (computed table first), evaluate, store, compare with the
        output snapshot when one exists.  Raises `Skipped`; evaluator errors propagate."""
        m = self.by_node[k]
        entry: dict[str, Any] = {"node": k, "name": m["name"], "spec": m["spec"], "step": m["step"]}
        t0 = time.time()
        leaves = self.arg_leaves(k)
        if self.max_arg_leaves is not None and leaves > self.max_arg_leaves:
            raise Skipped(f"{leaves} argument leaves exceed max_arg_leaves={self.max_arg_leaves}", budget=True)
        arrays, info = self.operand_arrays(k)
        got, ev = self.evaluate_arrays(k, arrays)
        if self.chain:
            self.computed[k] = got
        if k in self.output_nodes and got.size == 1:
            self.tokens[k] = int(got[0])
        entry.update(status="computed", n_args=int(sum(a.size for a in arrays)), n_returns=int(got.size), partial=info["fallback"],
                     boundary=len(info["boundary"]), **ev)
        if compare:
            expect = self.expected_words(m)
            if expect is not None:
                n_diff, first = _diff(got, expect)
                entry.update(status="pass" if not n_diff else "fail", n_diff=n_diff, first_diff=first)
        entry["seconds"] = round(time.time() - t0, 3)
        return entry

    def evaluate_instance(self, m: dict) -> dict:
        """Tier A whole-instance evaluation, exactly `compare_subcircuits`' procedure (operands by provenance)."""
        k = m["node"]
        entry: dict[str, Any] = {"node": k, "name": m["name"], "spec": m["spec"], "step": m["step"], "granularity": "instance"}
        t0 = time.time()
        args, _ = self.operand_words(k)
        expect = self.expected_words(m)
        if expect is None:
            raise Skipped(f"no snapshot of event {m['event']} covers the outputs")
        got = self.evaluate(k, args)
        n_diff, first = _diff(got, expect)
        entry.update(status="pass" if not n_diff else "fail", n_args=len(args), n_returns=int(got.size), n_diff=n_diff,
                     first_diff=first, evaluator="gates", twin_id=None, seconds=round(time.time() - t0, 3))
        return entry

    # -- child units (oversize instances) ----------------------------------------------------------------------

    def plan_child(self, k: int, budget: int, rng: random.Random, depth: int = 3) -> list[tuple[int, int]] | None:
        """A descent path `[(node index in the enclosing body, member count), ...]` from root instance `k` to a unit
        whose argument leaves fit `budget`, whose argument gates all lie outside the instance (they are the
        instance's own arguments) and whose return gates are among the instance's return gates.  None if no such
        unit exists within `depth` levels."""
        root = self.call(k)
        rix = self.ret_index(k)
        return self._descend(root, rix, root, [], budget, rng, depth)

    def _descend(self, root: FunctionCall, rix: dict[int, int], cur: FunctionCall, path: list, budget: int,
                 rng: random.Random, depth: int) -> list[tuple[int, int]] | None:
        if cur.is_primitive:
            return None
        sc = cur.scope()
        body = sc.spec.body
        order = list(range(len(body.nodes)))
        rng.shuffle(order)
        rlo, rhi = root.lo, root.hi
        for j in order:
            node = body.nodes[j]
            nm = 1 if node.form in ("primitive", "call") else node.n
            child = FunctionCall(self.circuit, sc, j, rng.randrange(nm))
            if not all(g in rix for g in child.returns):
                continue
            leaves = sum(t.leaves for t in child.arg_types())
            if leaves <= budget:
                if all(not (rlo <= g < rhi) for g in child.args):
                    return path + [(j, nm)]
                continue
            if depth > 0:
                sub = self._descend(root, rix, child, path + [(j, nm)], budget, rng, depth - 1)
                if sub is not None:
                    return sub
        return None

    def evaluate_child(self, m: dict, path: list[tuple[int, int]], members: list[int]) -> dict:
        """Tier A evaluation of one child unit of instance `m` along `path` at `members` (one per level, tier A draws them):
        argument values by gate from the instance's arguments, expected returns by gate from the instance's output snapshot."""
        k = m["node"]
        t0 = time.time()
        call = self.call(k)
        labels = []
        for (j, _), member in zip(path, members):
            call = FunctionCall(self.circuit, call.scope(), j, member)
            labels.append(f"{call.id}#{member}")
        entry: dict[str, Any] = {"node": k, "name": m["name"], "spec": m["spec"], "step": m["step"],
                                 "granularity": f"child:{call.id}", "child": {"path": labels, "gates": [call.lo, call.hi]}}
        self._out_cache.clear()
        expect_all = self.expected_words(m)
        if expect_all is None:
            raise Skipped(f"no snapshot of event {m['event']} covers the outputs")
        rix = self.ret_index(k)
        # P5 (D85): admissibility was established by `plan_child` on ONE random member; the members drawn here may differ.
        # A unit whose returns are not all instance outputs, or that reads an instance-interior value, is not a semantic
        # failure of the run -- classify it instead of failing the tier with a bare KeyError.
        root = self.call(k)
        if not all(g in rix for g in call.returns):
            raise Skipped(f"child {call.id} members {labels} return an interior value (member-dependent admissibility)")
        if any(root.lo <= g < root.hi for g in call.args):
            raise Skipped(f"child {call.id} members {labels} read an interior value (member-dependent admissibility)")
        args = self.gate_words(call.args)
        pos = [rix[g] for g in call.returns]
        expect = expect_all[np.array(pos, dtype=np.int64)]
        got = np.array(evaluate_call(self.circuit, call, args), dtype=np.uint64)
        n_diff, first = _diff(got, expect)
        entry.update(status="pass" if not n_diff else "fail", n_args=len(args), n_returns=int(got.size), n_diff=n_diff,
                     first_diff=first, evaluator="gates", twin_id=None, seconds=round(time.time() - t0, 3))
        return entry

    def outcome(self, m: dict, path: list[tuple[int, int]] | None, members: list[int] | None) -> tuple[str, Any]:
        """One tier A evaluation (the whole instance when `path` is None) as ("ok", entry), ("skipped", reason) or ("error", message)."""
        try:
            return "ok", (self.evaluate_instance(m) if path is None else self.evaluate_child(m, path, members))
        except Skipped as ex:
            return "skipped", ex.reason
        except Exception as ex:  # noqa: BLE001 - evaluator / MUFU table problems are reported, not fatal
            return "error", f"{type(ex).__name__}: {ex}"


# ---------------------------------------------------------------------------------------------------------
# tier A — body exactness per specialization
# ---------------------------------------------------------------------------------------------------------


def tier_a(result: FoldResult, store: SnapshotStore, weights: WeightSource | None, *, k: int = 3, seed: int,
           max_arg_leaves: int = 4_000_000, steps: Iterable[int] | None = None, jobs: int = 1) -> dict:
    """Body exactness: up to `k` seeded-random instances per specialization (stratified by the full specialization
    label), evaluated bit-exactly against their output snapshots; oversize instances at a child unit.  See the module
    docstring for the pass rule.  `jobs` > 1 evaluates on that many forked workers with the same draw (`tier_a_pool`):
    the report is the same apart from `jobs`, `seconds` and `max_rss_gb`."""
    t0 = time.time()
    rng = CH.stream(seed)
    snap_steps = sorted(set(store.steps) if steps is None else set(store.steps) & set(int(s) for s in steps))
    report: dict[str, Any] = {"tier": "a", "status": "pass", "k": k, "seed": seed, "jobs": jobs, "snapshot_steps": snap_steps,
                              "max_arg_leaves": max_arg_leaves, "coverage": {}, "instances": [], "uncovered_specs": []}
    try:
        rep = Replayer(result, store, weights, chain=False, snapshots=True)
    except ValueError as e:
        report.update(status="skipped", reason=str(e), seconds=round(time.time() - t0, 3), max_rss_gb=round(_max_rss_gb(), 3))
        return report
    by_spec: dict[str, list[dict]] = defaultdict(list)
    for m in result.instances:
        by_spec[m["spec"]].append(m)
    evaluate_strata(rep, report, by_spec, snap_steps, k=k, max_arg_leaves=max_arg_leaves, rng=rng, jobs=jobs)
    # Served tokens (D82): every TokenSelect instance at a snapshot step, evaluated from its observed operands (the
    # logits boundary); the computed token must equal the token the engine served for that (request, step).  This
    # proves TokenSelect given those logits, not the logits from the roots (that is what the sampled bodies + the
    # exhaustive provenance establish jointly).
    report["tokens"] = _served_tokens(rep, result, snap_steps)
    st = Counter(e["status"] for e in report["instances"])
    report["summary"] = dict(st)
    if st.get("fail") or st.get("error"):
        report["status"] = "fail"
        report["status_reason"] = f"{st.get('fail', 0)} failing and {st.get('error', 0)} erroring evaluations"
    elif report["tokens"].get("mismatch"):
        report["status"] = "fail"
        report["status_reason"] = f"{len(report['tokens']['mismatch'])} served token(s) not reproduced from the logits boundary"
    elif report["uncovered_specs"]:
        report["status"] = "fail"
        report["status_reason"] = f"{len(report['uncovered_specs'])} of {len(by_spec)} specializations have no passing evaluation"
    elif not st.get("pass"):
        report["status"] = "skipped"
        report["status_reason"] = "nothing evaluated"
    report["n_specs"] = len(by_spec)
    report["n_specs_covered"] = sum(1 for c in report["coverage"].values() if c["pass"] and not (c["fail"] or c["error"])
                                    and c["pass"] >= min(k, c["evaluable"]))
    report["evaluators"] = {"twins": 0, "gates": sum(1 for e in report["instances"] if e["status"] in ("pass", "fail")), "self_check": {}}
    report["vu_population"] = vu_population(result, rep, max_arg_leaves)   # D85: informational, never gates
    report["seconds"] = round(time.time() - t0, 3)
    report["max_rss_gb"] = round(_max_rss_gb(), 3)
    return report


def vu_population(result: FoldResult, rep: Replayer, max_arg_leaves: int | None, *, depth: int = 3, max_units_per_spec: int = 200_000) -> dict:
    """What the sampled population actually covers (D85, `vu_query`): per specialization the instance / VU counts and
    granularity, the oversize instances that have NO admissible unit (zero mass: a sampler over this population never
    checks them), and for every oversize specialization whether its child units tile the instance's returns
    (`partial` = some output leaves of the instance are checkable by no unit).  Informational; failures here are reported,
    never raised."""
    t0 = time.time()
    try:
        from verity_vllm.check.replay import vu_query as vq
        budget = int(max_arg_leaves) if max_arg_leaves else 1 << 62
        ix = vq.index(result, max_arg_leaves=budget, depth=depth)
        summ = ix.summary()
        tiling: dict[str, dict] = {}
        seen_spec: set[str] = set()
        for kk, m in ix.by_node.items():
            spec = m["spec"]
            if ix.fits[kk] or spec in seen_spec:
                continue
            seen_spec.add(spec)
            n_ret = len(rep.ret_index(kk))
            covered: set[int] = set()
            n_units = 0
            for vu in ix.of_instance(kk):
                covered.update(vq.returns_of(result, vu))
                n_units += 1
                if n_units >= max_units_per_spec:
                    break
            tiling[spec] = {"instance_node": kk, "returns": n_ret, "covered_returns": len(covered), "units_examined": n_units,
                            "partial": len(covered) < n_ret, "truncated": n_units >= max_units_per_spec}
        zero_mass = sorted({m["spec"] for kk, m in ix.by_node.items() if not ix.fits[kk] and not ix.per_instance[kk]})
        return {"status": "ok", **summ, "zero_mass_specs": zero_mass, "tiling": tiling,
                "partial_tiling_specs": sorted(s for s, t in tiling.items() if t["partial"]), "seconds": round(time.time() - t0, 3)}
    except Exception as e:  # noqa: BLE001 - informational
        return {"status": "error", "error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - t0, 3)}


def _served_tokens(rep: "Replayer", result: FoldResult, snap_steps: list[int]) -> dict:
    """Evaluate every TokenSelect instance at a snapshot step from its observed operands and compare with the served token."""
    out: dict[str, Any] = {"steps": [], "evaluated": 0, "equal": 0, "skipped": 0, "mismatch": [], "per_step": {}}
    nodes = sorted(rep.output_nodes)
    for k in nodes:
        m = rep.by_node.get(k)
        if m is None or m["step"] not in snap_steps:
            continue
        rid, st = rep.output_nodes[k]
        served = next((int(t) for s_, t in result.sampled.get(rid, []) if s_ == st), None)
        ps = out["per_step"].setdefault(str(st), {"rows": 0, "equal": 0, "skipped": 0})
        ps["rows"] += 1
        try:
            args, _ = rep.operand_words(k)
            got = rep.evaluate(k, args)
        except Skipped as ex:
            out["skipped"] += 1
            ps["skipped"] += 1
            ps.setdefault("skip_reason", ex.reason[:120])
            continue
        out["evaluated"] += 1
        tok = int(got[0]) if got.size == 1 else None
        if tok is not None and served is not None and tok == served:
            out["equal"] += 1
            ps["equal"] += 1
        else:
            out["mismatch"].append({"node": k, "request": rid, "step": st, "computed": tok, "served": served})
    out["steps"] = sorted(int(x) for x in out["per_step"])
    out["all_rows_equal"] = bool(out["evaluated"]) and not out["mismatch"] and not out["skipped"]
    return out


# ---------------------------------------------------------------------------------------------------------
# tier B — wiring exactness of one step;  tier C — end to end
# ---------------------------------------------------------------------------------------------------------


def _step_metas(result: FoldResult, step: int) -> list[dict]:
    return [m for m in result.instances if m["step"] == step]


def _replay_step(rep: Replayer, metas: list[dict], *, compare: bool, max_instances: int | None, layers: set[int] | None,
                 progress=None) -> dict:
    """Replay `metas` (one step, node order) through `rep`; the per-step part of the tier B/C reports."""
    t0 = time.time()
    rep.scope = (metas[0]["node"], metas[-1]["node"] + 1)
    entries: list[dict] = []
    first_div = None
    n_eval = n_cmp = n_div = 0
    for i, m in enumerate(metas):
        k = m["node"]
        try:
            if max_instances is not None and i >= max_instances:
                raise Skipped(f"beyond max_instances={max_instances}", budget=True)
            if layers is not None:
                L = layer_of(m["hint"])
                if L is not None and L not in layers:
                    raise Skipped(f"layer {L} not in the selected layers", budget=True)
            e = rep.replay_node(k, compare=compare)
        except Skipped as ex:
            rep.skipped_nodes.add(k)
            e = {"node": k, "name": m["name"], "spec": m["spec"], "step": m["step"], "status": "skipped", "reason": ex.reason,
                 "budget": ex.budget}
        except Exception as ex:  # noqa: BLE001 - evaluator problems are reported per instance
            rep.skipped_nodes.add(k)
            e = {"node": k, "name": m["name"], "spec": m["spec"], "step": m["step"], "status": "error",
                 "reason": f"{type(ex).__name__}: {ex}"}
        rep.release(k)
        entries.append(e)
        if e["status"] in ("pass", "fail", "computed"):
            n_eval += 1
        if e["status"] in ("pass", "fail"):
            n_cmp += 1
        if e["status"] == "fail":
            n_div += 1
            if first_div is None:
                first_div = {kk: e[kk] for kk in ("node", "name", "spec", "n_diff", "first_diff")}
        if progress is not None and (i + 1) % 50 == 0:
            progress(i + 1, len(metas), e)
    skipped = [e for e in entries if e["status"] == "skipped"]
    errors = [e for e in entries if e["status"] == "error"]
    partial = any(e.get("partial") for e in entries if e["status"] != "skipped")
    # the step's TokenSelect instances: computed token vs the log's sampled token
    tok_rows = [(m["row"], m["node"]) for m in metas if m["node"] in rep.output_nodes]
    tok_rows.sort()
    computed = [rep.tokens.get(node) for _, node in tok_rows]
    sampled = []
    for _, node in tok_rows:
        rid, st = rep.output_nodes[node]
        sampled.append(next((int(t) for s, t in rep.r.sampled.get(rid, []) if s == st), None))
    if tok_rows and all(c is not None for c in computed):
        equal = computed == sampled
    else:
        equal = None
    step = metas[0]["step"]
    return {"step": step, "n_instances": len(metas), "n_evaluated": n_eval, "n_compared": n_cmp, "n_diverged": n_div,
            "n_skipped": len(skipped), "n_errors": len(errors), "partial": partial, "first_divergence": first_div,
            # KV / state linkage (gate G5): instances of this step that consumed an operand produced OUTSIDE the step's scope,
            # i.e. at an earlier step (attention over K/V rows written by earlier steps; the residual is intra-step)
            "n_from_earlier_steps": sum(1 for e in entries if e.get("boundary")),
            "n_earlier_step_operands": sum(int(e.get("boundary") or 0) for e in entries),
            "tokens": {"computed": computed, "sampled": sampled, "equal": equal},
            "skipped": [{"node": e["node"], "name": e["name"], "spec": e["spec"], "reason": e["reason"], "budget": e["budget"]} for e in skipped[:20]],
            "skipped_reasons": dict(Counter(("budget: " if e["budget"] else "evidence: ") + e["reason"].split(":")[0] for e in skipped)),
            "n_budget_skips": sum(1 for e in skipped if e["budget"]),
            "errors": [{"node": e["node"], "name": e["name"], "spec": e["spec"], "reason": e["reason"]} for e in errors[:20]],
            "divergences": [{"node": e["node"], "name": e["name"], "spec": e["spec"], "n_diff": e["n_diff"], "first_diff": e["first_diff"]}
                            for e in entries if e["status"] == "fail"][:20],
            "evaluators": _evaluators(rep, entries),
            "instances": entries, "seconds": round(time.time() - t0, 3)}


def _evaluators(rep: Replayer, entries: list[dict]) -> dict:
    """Which evaluator produced the evaluated entries, and the twin self-checks this Replayer ran so far."""
    return {"twins": sum(1 for e in entries if e.get("evaluator") == "twins"),
            "gates": sum(1 for e in entries if e.get("evaluator") == "gates"),
            "declined": sum(1 for e in entries if e.get("declined")),
            "mode": rep.evaluator,
            "self_check": {sid: {kk: v for kk, v in r.items() if kk != "statics"} for sid, r in sorted(rep.self_checks.items())}}


def _step_status(s: dict) -> tuple[str, str]:
    evidence = s["n_skipped"] - s["n_budget_skips"]
    if s["n_diverged"] or s["n_errors"] or s["tokens"]["equal"] is False:
        return "fail", (f"{s['n_diverged']} divergences, {s['n_errors']} errors"
                        + (", sampled token mismatch" if s["tokens"]["equal"] is False else ""))
    if evidence:
        return "skipped", f"{evidence} instances skipped for missing evidence (not the explicit budget)"
    if not s["n_evaluated"]:
        return "skipped", "nothing evaluated"
    return "pass", ""


def tier_b(result: FoldResult, store: SnapshotStore, weights: WeightSource | None, *, step: int, max_arg_leaves: int | None = None,
           compare_outputs: bool = True, max_instances: int | None = None, layers: Iterable[int] | None = None,
           progress=None, evaluator: str = "auto", self_check_n: int = 3, seed: int = 0) -> dict:
    """Wiring exactness of one step (see the module docstring).  Budgets: `max_arg_leaves` (None = unlimited; the
    lm_head Gemm / Embedding need ~1 GB of Python ints at 28M leaves — pod only), `max_instances` (a prefix of the
    step's instances in node order), `layers` (only these layers' instances plus the non-layer ones: embedding, final
    norm, lm_head, sampler).  Budget skips never fail the tier; consumers fall back to snapshots and are `partial`."""
    t0 = time.time()
    base: dict[str, Any] = {"tier": "b", "step": step, "max_arg_leaves": max_arg_leaves, "max_instances": max_instances,
                            "layers": sorted(layers) if layers is not None else None, "compare_outputs": compare_outputs,
                            "evaluator": evaluator}
    metas = _step_metas(result, step)
    if not metas:
        return {**base, "status": "skipped", "reason": f"no instances at step {step}", "seconds": 0.0, "max_rss_gb": round(_max_rss_gb(), 3)}
    try:
        rep = Replayer(result, store, weights, chain=True, snapshots=True, max_arg_leaves=max_arg_leaves,
                       evaluator=evaluator, self_check_n=self_check_n, seed=seed)
    except ValueError as e:
        return {**base, "status": "skipped", "reason": str(e), "seconds": 0.0, "max_rss_gb": round(_max_rss_gb(), 3)}
    s = _replay_step(rep, metas, compare=compare_outputs, max_instances=max_instances,
                     layers=set(int(l) for l in layers) if layers is not None else None, progress=progress)
    status, why = _step_status(s)
    report = {**base, **s, "status": status}
    if why:
        report["status_reason"] = why
    report["seconds"] = round(time.time() - t0, 3)
    report["max_rss_gb"] = round(_max_rss_gb(), 3)
    return report


def _producers(result: FoldResult, k: int) -> set[int]:
    """Root instances whose outputs instance `k` reads."""
    out: set[int] = set()
    for a in result.body.nodes[k].args:
        for space, idx, _, _ in runs(a.refs):
            if space == "n":
                out.add(idx)
    return out


def plan_chains(result: FoldResult, snap_steps: list[int], *, n: int, rng: random.Random, depth: int = 3, max_nodes: int = 8,
                arg_leaves=None, max_arg_leaves: int | None = None) -> list[list[int]]:
    """Choose `n` connected subgraphs (D82 chaining smoke test): a seeded consumer at a snapshot step and its in-step
    producers up to `depth` levels back (at most `max_nodes`, every node within `max_arg_leaves`), preferring slices
    whose members span >= 2 specializations.  Returns node lists in node order; slices never share a seed."""
    by_node = {m["node"]: m for m in result.instances}
    # seeds round-robin over the snapshot steps (prefill has many more instances than a decode step; slices should
    # cover both regimes), each step's candidates shuffled
    cands = [m["node"] for m in _distinct_step_order([m for m in result.instances if m["step"] in snap_steps], rng)]
    fits = (lambda k: arg_leaves is None or max_arg_leaves is None or arg_leaves(k) <= max_arg_leaves)
    slices: list[list[int]] = []
    used: set[int] = set()
    for seed in cands:
        if len(slices) >= n:
            break
        if seed in used or not fits(seed):
            continue
        step = by_node[seed]["step"]
        members = {seed}
        frontier = [seed]
        for _ in range(depth):
            nxt = []
            for c in frontier:
                for p in sorted(_producers(result, c)):
                    if p in members or p not in by_node or by_node[p]["step"] != step or not fits(p):
                        continue
                    members.add(p)
                    nxt.append(p)
                    if len(members) >= max_nodes:
                        break
                if len(members) >= max_nodes:
                    break
            frontier = nxt
            if not frontier or len(members) >= max_nodes:
                break
        if len(members) < 2:
            continue
        specs = {by_node[k]["spec"].split("{")[0] for k in members}
        if len(specs) < 2 and len(slices) < n:      # keep looking for a boundary-crossing slice; accept as a last resort
            continue
        slices.append(sorted(members))
        used |= members
    if len(slices) < n:                               # fall back to same-spec slices when nothing else crosses a boundary
        for seed in cands:
            if len(slices) >= n:
                break
            if seed in used or not fits(seed):
                continue
            ps = [p for p in _producers(result, seed) if p in by_node and by_node[p]["step"] == by_node[seed]["step"] and fits(p)]
            if ps:
                slices.append(sorted({seed, *ps[:max_nodes - 1]}))
                used |= set(slices[-1])
    return slices


def tier_chain(result: FoldResult, store: SnapshotStore, weights: WeightSource | None, *, n: int = 3, seed: int, depth: int = 3,
               max_nodes: int = 8, max_arg_leaves: int | None = 4_000_000, steps: Iterable[int] | None = None) -> dict:
    """Chaining smoke test (D82): `n` small connected subgraphs replayed with *computed* intermediate values — a
    consumer inside a slice takes its operand from the producer's computed output, never from the snapshot — and
    every member compared bit-exactly with its output snapshot.  Catches evaluator / encoding / Refs integration
    errors that per-instance evaluation from snapshots cannot see.  Pass iff every slice's members pass and every
    slice chained >= 1 producer->consumer edge."""
    t0 = time.time()
    rng = CH.stream(seed)
    snap_steps = sorted(set(store.steps) if steps is None else set(store.steps) & set(int(s) for s in steps))
    report: dict[str, Any] = {"tier": "chain", "status": "pass", "n": n, "seed": seed, "depth": depth, "max_nodes": max_nodes,
                              "max_arg_leaves": max_arg_leaves, "snapshot_steps": snap_steps, "slices": []}
    try:
        rep = Replayer(result, store, weights, chain=True, snapshots=True, max_arg_leaves=max_arg_leaves)
    except ValueError as e:
        report.update(status="skipped", reason=str(e), seconds=round(time.time() - t0, 3), max_rss_gb=round(_max_rss_gb(), 3))
        return report
    slices = plan_chains(result, snap_steps, n=n, rng=rng, depth=depth, max_nodes=max_nodes, arg_leaves=rep.arg_leaves,
                         max_arg_leaves=max_arg_leaves)
    if not slices:
        report.update(status="skipped", reason="no connected slice within the budget at a snapshot step")
    for nodes in slices:
        rep.computed.clear()
        rep.scope = (nodes[0], nodes[-1] + 1)
        members = set(nodes)
        entries: list[dict] = []
        edges = 0
        for k in nodes:
            m = rep.by_node[k]
            inside = sorted(p for p in _producers(result, k) if p in members)
            try:
                e = rep.replay_node(k, compare=True)
                # a member producer that was not fed from the computed table breaks the chain
                fell = [p for p in inside if p in (e.get("partial") or [])]
                e["chained_from"] = [p for p in inside if p not in fell]
                e["unchained"] = fell
                edges += len(e["chained_from"])
                if fell:
                    e["status"] = "fail"
                    e["reason"] = f"member producers {fell} came from snapshots, not the computed table"
                elif e["status"] == "computed":
                    e["status"] = "fail"
                    e["reason"] = "no output snapshot to compare with"
            except Skipped as ex:
                e = {"node": k, "name": m["name"], "spec": m["spec"], "step": m["step"], "status": "skipped", "reason": ex.reason}
            except Exception as ex:  # noqa: BLE001
                e = {"node": k, "name": m["name"], "spec": m["spec"], "step": m["step"], "status": "error", "reason": f"{type(ex).__name__}: {ex}"}
            e.pop("partial", None)
            entries.append(e)
        ok = all(e["status"] == "pass" for e in entries) and edges >= 1
        report["slices"].append({"nodes": nodes, "step": rep.by_node[nodes[0]]["step"], "specs": sorted({rep.by_node[k]["spec"] for k in nodes}),
                                 "edges_chained": edges, "status": "pass" if ok else "fail", "members": entries})
    fails = [sl for sl in report["slices"] if sl["status"] != "pass"]
    report["n_slices"] = len(report["slices"])
    report["n_edges_chained"] = sum(sl["edges_chained"] for sl in report["slices"])
    if fails:
        report["status"] = "fail"
        report["status_reason"] = f"{len(fails)} of {len(report['slices'])} slices failed"
    report["seconds"] = round(time.time() - t0, 3)
    report["max_rss_gb"] = round(_max_rss_gb(), 3)
    return report


def tier_c(result: FoldResult, weights: WeightSource | None, *, steps: Iterable[int] | None = None, max_arg_leaves: int | None = None,
           progress=None, evaluator: str = "auto", self_check_n: int = 3, seed: int = 0) -> dict:
    """End to end — **pod only** for a real model (hours on a CPU: ~40 us per gate, 5e8 gates for M1).  The replay
    engine over all steps (or `steps`, a prefix of the served steps) with no interior snapshots: every operand is
    computed or a parameter; the next step's token is the previous step's computed TokenSelect (the fold wired the
    decode Embedding to it), checked against `result.sampled` per step — a mismatch is reported and the computed
    token keeps flowing.  `max_arg_leaves` is accepted for tests only: a budget skip of the lm_head cascades into
    skips of everything downstream."""
    t0 = time.time()
    all_steps = [s["step"] for s in result.steps]
    steps_l = [int(s) for s in steps] if steps is not None else all_steps
    base: dict[str, Any] = {"tier": "c", "steps": steps_l, "max_arg_leaves": max_arg_leaves, "evaluator": evaluator}
    try:
        rep = Replayer(result, None, weights, chain=True, snapshots=False, max_arg_leaves=max_arg_leaves,
                       evaluator=evaluator, self_check_n=self_check_n, seed=seed)
    except ValueError as e:
        return {**base, "status": "skipped", "reason": str(e), "seconds": 0.0, "max_rss_gb": round(_max_rss_gb(), 3)}
    per_step = []
    for st in steps_l:
        metas = _step_metas(result, st)
        if not metas:
            per_step.append({"step": st, "status": "skipped", "reason": "no instances"})
            continue
        s = _replay_step(rep, metas, compare=False, max_instances=None, layers=None, progress=progress)
        s.pop("instances")
        status, why = _step_status(s)
        s["status"] = status
        if why:
            s["status_reason"] = why
        per_step.append(s)
    # final tokens per request over the replayed steps
    tokens: dict[str, dict] = {}
    for rid, seq in result.sampled.items():
        comp, samp = [], []
        for node, (r, st) in rep.output_nodes.items():
            if r == rid and st in steps_l:
                comp.append((st, rep.tokens.get(node)))
        comp.sort()
        want = {st: int(t) for st, t in seq if st in steps_l}
        samp = [want.get(st) for st, _ in comp]
        got = [t for _, t in comp]
        tokens[rid] = {"steps": [st for st, _ in comp], "computed": got, "sampled": samp,
                       "equal": (got == samp) if got and all(t is not None for t in got) else None,
                       "first_mismatch_step": next((st for (st, g), s_ in zip(comp, samp) if g != s_), None)}
    n_div = sum(s.get("n_diverged", 0) for s in per_step)
    n_err = sum(s.get("n_errors", 0) for s in per_step)
    n_skip = sum(s.get("n_skipped", 0) for s in per_step)
    if any(t["equal"] is False for t in tokens.values()) or n_err:
        status, why = "fail", "sampled token mismatch" if n_err == 0 else f"{n_err} evaluator errors"
    elif n_skip or not per_step or all(t["equal"] is None for t in tokens.values()):
        status, why = "skipped", f"{n_skip} instances skipped; tokens not fully computed"
    else:
        status, why = "pass", ""
    report = {**base, "status": status, "per_step": per_step, "tokens": tokens,
              "n_instances": sum(s.get("n_instances", 0) for s in per_step), "n_evaluated": sum(s.get("n_evaluated", 0) for s in per_step),
              "n_diverged": n_div, "n_skipped": n_skip, "n_errors": n_err,
              "evaluators": {"twins": sum(s.get("evaluators", {}).get("twins", 0) for s in per_step),
                             "gates": sum(s.get("evaluators", {}).get("gates", 0) for s in per_step),
                             "declined": sum(s.get("evaluators", {}).get("declined", 0) for s in per_step),
                             "mode": evaluator,
                             "self_check": {sid: {kk: v for kk, v in r.items() if kk != "statics"} for sid, r in sorted(rep.self_checks.items())}}}
    if why:
        report["status_reason"] = why
    report["seconds"] = round(time.time() - t0, 3)
    report["max_rss_gb"] = round(_max_rss_gb(), 3)
    return report


# ---------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------


def summary_line(rep: dict) -> str:
    t = rep.get("tier")
    if t == "a":
        return (f"tier a  {rep['status']:7s} specs {rep.get('n_specs_covered', 0)}/{rep.get('n_specs', 0)} covered, k={rep['k']} "
                f"seed={rep['seed']} steps={rep['snapshot_steps']} evaluations={rep.get('summary')} uncovered={len(rep.get('uncovered_specs', []))} "
                f"{rep.get('seconds')}s {rep.get('max_rss_gb')}GB")
    ev = rep.get("evaluators") or {}
    ev_s = f"evaluators twins={ev.get('twins', 0)} gates={ev.get('gates', 0)} declined={ev.get('declined', 0)} " if ev else ""
    if t == "b":
        return (f"tier b  {rep['status']:7s} step {rep['step']}: {rep.get('n_evaluated', 0)}/{rep.get('n_instances', 0)} evaluated, "
                f"{rep.get('n_compared', 0)} compared, {rep.get('n_diverged', 0)} diverged, {rep.get('n_skipped', 0)} skipped "
                f"({rep.get('n_budget_skips', 0)} budget), partial={rep.get('partial')} tokens={rep.get('tokens')} {ev_s}"
                f"{rep.get('seconds')}s {rep.get('max_rss_gb')}GB")
    if t == "c":
        return (f"tier c  {rep['status']:7s} steps {rep['steps'][:1]}..{rep['steps'][-1:]}: {rep.get('n_evaluated', 0)}/{rep.get('n_instances', 0)} "
                f"evaluated, {rep.get('n_skipped', 0)} skipped, tokens={ {r: t['equal'] for r, t in rep.get('tokens', {}).items()} } {ev_s}"
                f"{rep.get('seconds')}s {rep.get('max_rss_gb')}GB")
    return f"tier {t} {rep.get('status')}"


@dataclass(frozen=True, kw_only=True)
class Options:
    log: str = option("--log")
    snapshots: str | None = option("--snapshots", default=None, help="directory with snapshots.json and values/ (tiers a, b)")
    profile: str = option("--profile")
    tier: str = option("--tier", choices=("a", "b", "c", "chain"))
    chains: int = option("--chains", type=int, default=3, help="tier chain: number of connected slices")
    chain_depth: int = option("--chain-depth", type=int, default=3)
    chain_max_nodes: int = option("--chain-max-nodes", type=int, default=8)
    step: int | None = option("--step", type=int, default=None, help="tier b: the step to replay; tier c: replay steps 0..STEP")
    k: int = option("--k", type=int, default=3)
    jobs: int = option("--jobs", "-j", type=int, default=1, help="tier a: evaluate on this many forked workers (tier_a_pool); the same seed gives the same instances, members and report")
    seed: int | None = option("--seed", type=int, default=None, help="tiers a, chain: the sample's seed (required); tiers b, c: the twin self-check's (default 0)")
    max_arg_leaves: int | None = option("--max-arg-leaves", type=int, default=None, help="tier a default 4000000; tiers b/c default unlimited")
    max_instances: int | None = option("--max-instances", type=int, default=None, help="tier b: replay only this prefix of the step's instances")
    layers: str | None = option("--layers", default=None, help="tier b: comma-separated layers to replay (non-layer instances always)")
    no_compare: bool = option("--no-compare", default=False, action="store_true", help="tier b: do not compare outputs with snapshots")
    safetensors: str | None = option("--safetensors", default=None, help="glob of the checkpoint's model*.safetensors (default: HF cache of the profile's model)")
    cos_sin: str | None = option("--cos-sin", default=None, help="uint16 .npy of the rotary cos_sin table (default: <snapshots>/cos_sin_cache.npy, else the B0 fixture)")
    progress: bool = option("--progress", default=False, action="store_true", help="print progress every 50 instances (tiers b, c)")
    evaluator: str = option("--evaluator", default="auto", choices=EVALUATORS, help="tiers b, c: gates (evaluate_call), twins (verity_vllm.program.kernels.twins wherever one exists; a failed self-check is " "an error), auto (a twin after its self-check passed, else gates; default)")
    self_check_n: int = option("--self-check-n", type=int, default=3, help="random instances per specialization in the twin self-check (>= 3)")
    out: str = option("--out")


def main(a: Options) -> int:
    if a.self_check_n < 3:
        raise UsageError("--self-check-n must be >= 3")
    if a.jobs < 1:
        raise UsageError("--jobs must be >= 1")
    if a.seed is None and a.tier in ("a", "chain"):
        raise UsageError(f"--seed is required for tier {a.tier}: its sample is drawn from it")

    t0 = time.time()
    if a.evaluator != "gates" and a.tier in ("b", "c"):
        from verity_vllm.program.kernels.twins import preload
        preload()      # MUFU tables + C++ models before the Program and the checkpoint are resident (lower high-water mark)
    profile = load_profile(a.profile)
    repo = (profile.config or {}).get("model")
    result = fold_log(a.log, profile, with_program=True)
    fold_s = round(time.time() - t0, 3)
    snaps = Path(a.snapshots) if a.snapshots else None
    store = SnapshotStore(snaps) if snaps is not None and (snaps / "snapshots.json").exists() else None
    cos_sin = Path(a.cos_sin) if a.cos_sin else ((snaps / "cos_sin_cache.npy") if snaps is not None and (snaps / "cos_sin_cache.npy").exists() else None)
    weights = WeightSource(result.params[-1][1], [Path(p) for p in sorted(glob.glob(a.safetensors))] if a.safetensors else None, cos_sin, repo=repo,
                           config=dict(profile.config or {}))   # [SHARED-CODE lane cov-ln] the profile's family / dtype facts

    def progress(i, n, e):
        print(f"  [{i}/{n}] {e.get('name')} {e['status']} {e.get('seconds', '')}s rss={_max_rss_gb():.2f}GB", file=sys.stderr, flush=True)

    prog = progress if a.progress else None
    if a.tier == "a":
        if store is None:
            rep = {"tier": "a", "status": "skipped", "reason": "no snapshots.json"}
        else:
            rep = tier_a(result, store, weights, k=a.k, seed=a.seed, max_arg_leaves=a.max_arg_leaves if a.max_arg_leaves is not None else 4_000_000, jobs=a.jobs)
    elif a.tier == "chain":
        if store is None:
            rep = {"tier": "chain", "status": "skipped", "reason": "no snapshots.json"}
        else:
            rep = tier_chain(result, store, weights, n=a.chains, seed=a.seed, depth=a.chain_depth, max_nodes=a.chain_max_nodes,
                             max_arg_leaves=a.max_arg_leaves if a.max_arg_leaves is not None else 4_000_000)
    elif a.tier == "b":
        if store is None or a.step is None:
            rep = {"tier": "b", "status": "skipped", "reason": "tier b needs --snapshots and --step"}
        else:
            layers = [int(x) for x in a.layers.split(",") if x.strip()] if a.layers else None
            rep = tier_b(result, store, weights, step=a.step, max_arg_leaves=a.max_arg_leaves, compare_outputs=not a.no_compare,
                         max_instances=a.max_instances, layers=layers, progress=prog, evaluator=a.evaluator, self_check_n=a.self_check_n, seed=a.seed or 0)
    else:
        steps = [s["step"] for s in result.steps if a.step is None or s["step"] <= a.step]
        rep = tier_c(result, weights, steps=steps, max_arg_leaves=a.max_arg_leaves, progress=prog, evaluator=a.evaluator,
                     self_check_n=a.self_check_n, seed=a.seed or 0)
    rep = {"log": a.log, "profile": a.profile, "model": repo, "digest": result.digest, "fold_seconds": fold_s,
           "weights_available": weights.available(), **rep}
    rep["total_seconds"] = round(time.time() - t0, 3)
    rep["max_rss_gb"] = round(_max_rss_gb(), 3)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(rep), indent=1))
    print(summary_line(rep), file=sys.stderr)
    print(str(out))
    return 0 if rep["status"] == "pass" else 1
