"""Structural / unfolding comparator (CB-A): a derived Program (flat request or padded rectangle) against the accepted
record's Program, or a padded Program against its flat counterpart. Comparison ONLY — nothing here feeds construction.

Unit of comparison: the root body's *instance sequence* (every non-Input root node: spec id, form, operand reference
runs), segmented into engine steps at `TokenSelect` boundaries (one sampling event per step; the oracle's per-instance
`step` field cross-checks the segmentation). Per step:

  1. spec sequence — exact; then with attention `T` normalised (the only per-position static);
  2. operand wiring — every operand run is canonicalised to a source: (`W`, weight field name, lo, hi) for an input
     weight, (`TOK`, lo, hi) for the prompt, (`N`, producer step, producer index-within-step, lo, hi) for a node
     output — and compared 1:1 when the spec sequences align (the weight name comes from each program's own record:
     the derive report's field list / the oracle's `instances.jsonl` hints); cross-step (`KV`) sources are reported apart;
  3. unfolding (padded vs flat) — the padded program's pad-kind instances (`Const*`, `I32Eq`, `BitNot`, `BitAnd`,
     `SelectI32`, `batch(SelectBf16)`) are removed, every operand that reads a mux is followed through the mux's
     live branch (`a`) to the underlying producer, and the result is compared with the flat program as in 1–2. The
     muxes themselves are audited: per decode step ONE live bit keys the token slot and every layer's K and V row,
     against ONE zero gate; a mux anywhere else is reported (a real mismatch, never normalised).

Every difference is reported with counts and examples; nothing is dropped silently.

    verity-vllm program-compare --a <dir with descriptor.json.gz + derive-report.json> \\
        --b <oracle dir with program.json + instances.jsonl | derived dir> --out compare.json [--unfold-a]
"""

from __future__ import annotations

import collections
import difflib
import gzip
import json
import os
import re
import sys
import time
from dataclasses import dataclass

from verity.ml.circuits.program import RUN, input_id
from verity_vllm.check.match import split_selects
from verity_vllm.check.match.canonical_operands import as_progression as _asP, weight_field_name as _field_name
from verity_vllm.check.match import supplied_inputs as SI
from verity_vllm.check.match.peer_reads import PeerReads
from verity_vllm.check.match.sampler_geometry import constant_splits_agree
from verity_vllm.config import option
from verity_vllm.program.sampling_event import is_sampling_event as _is_sampling_event  # [R15 stoch] the step-boundary families

PAD_SPECS = ("Const", "I32Eq_v1", "BitNot_v1", "BitAnd_v1", "SelectI32_v1", "SelectBf16_v1")
_T_RE = re.compile(r"\{T=\d+,")
# [R16 stoch/global, COORD M-0169 (4)] the derived top-p split count S: the fold emits it as `Const32[S]` "splits", the 6th operand
# (index 5) of the top-p select, the one sampling event with a 6th operand (patterns.TokenSelectGumbelTopP._stage3); the request Program
# takes it as the request-owned INPUT `splits` (int32 [T+1], after the seed) and event s reads `splits[s:s+1]` (vllm_meta.StochasticRequest).
# compare BINDS the two: the fold's constant at the request's event s IS the value of the Program's input entry s -- canonicalised
# ("SPLITS", s, s+1) on both sides; the constant's VALUE is kept (`Prog.splits_values`, by select node) for GM-01's per-engine-step
# relation check (`global_match.sampler_geometry_check`: S_t == SplitsFor_v1(|live(t)|, num_SMs)).  Nothing about S is a static of the
# Definition.
_CONST_RE = re.compile(r"^Const(\d+)\[0x([0-9a-fA-F]+)\]_v1$")
SPLITS_OPERAND = 5


def const_value(spec: str) -> tuple[int, int] | None:
    """(width, value) of a codec `Const<w>[0x..]_v1` spec (a fold / Program constant instance); None for any other spec."""
    m = _CONST_RE.match(spec.split(":", 1)[-1])
    return (int(m.group(1)), int(m.group(2), 16)) if m else None


def _registry():
    """The registry every module loaded (`verity_vllm.program.registry.load`); a module that fails to import is named on stderr (its
    Programs, e.g. Hopper's under Python < 3.12, then fail to decode with that reason)."""
    from verity_vllm.program.registry.catalog import load
    reg, errors = load()
    for e in errors:
        print(f"[compare] registry module unavailable ({e}): Programs naming its Definitions will not decode", file=sys.stderr)
    return reg


class Prog(PeerReads):
    """A decoded root body as an instance sequence."""

    compact = False   # operand representation: runs (record) | run progressions ([R17 mfast] cb_a.compact); set per instance in __init__

    def __init__(self, label: str, desc: dict, weight_fields: dict[int, str], tok_node: int, oracle_rows: list[dict] | None = None,
                 compact: bool = False):
        from verity.ml.circuits.codec import program_digest
        self.label = label
        self.compact = bool(compact)
        self.digest = desc.get("digest") or program_digest(desc)
        self.inputs: dict[int, tuple[str, int]] = {}
        self.inst: list[dict] = []  # k, spec, form, args=[runs | progressions], out
        if self.compact:
            # [R17 mfast] the root body read straight from the descriptor JSON into run progressions: no Node / Coll / Refs objects for
            # the root, no per-row expansion of a Strided (compact.lean_root; the non-root Definitions are decoded as decode_program does)
            from verity_vllm.program.compact import lean_root
            lr = lean_root(desc, _registry())
            for k, fid, form, args6, out, n in lr["nodes"]:
                if form == "batch" and input_id(fid) is not None:
                    self.inputs[k] = (fid, n)
                    continue
                self.inst.append({"k": k, "spec": fid if form != "batch" else f"batch:{fid}", "form": form, "args": args6, "out": out})
            self.n_nodes = lr["n_nodes"]
            self.nodes = None
            self.param_names = lr["params"]
            del lr
        else:
            from verity.ml.circuits.codec import decode_program
            _spec_id, runs = self._node_functions()
            dec = decode_program(desc, _registry())
            root = dec.root
            body = root.body if hasattr(root, "body") else root.defn.body
            for k, n in enumerate(body.nodes):
                fid = _spec_id(n.fn)
                if n.form == "batch" and input_id(fid) is not None:
                    self.inputs[k] = (fid, n.n)
                    continue
                self.inst.append({"k": k, "spec": fid if n.form != "batch" else f"batch:{fid}", "form": n.form,
                                  "args": [runs(a.refs) for a in n.args], "out": str(n.out)})
            # [R16 global, F-global-mem / GM-01 at LP 4096] the decoded IR (`body.nodes`: node objects + Refs trees, O(LP^2) leaves for the
            # attention K/V gathers of a long prefill) is NOT retained: every comparison reads `inst` (spec, form, operand RUNS, out) only.
            # Keeping `self.nodes` alive doubled the resident Program (fold 148 GB / comparator 172 GiB / GM-01 > 351 GiB on the 4096/512
            # row, capacity M-0815).  `n_nodes` keeps the count; nothing in cb_a read the node objects after this loop.
            self.n_nodes = len(body.nodes)
            self.nodes = None
            self.param_names = [n for n, _ in getattr(body, "params", ())]   # empty for a lowered root; kept for callers that inspect it
            del dec, root, body
        if len(self.inst) > 50_000:   # the decoded body is a reference cycle (nodes <-> body); collect it now, not at an arbitrary later point
            import gc
            gc.collect()
        # [R15 stoch] the seed parameters (a stochastic Program's declared runtime inputs: the request wrapper's `seed`, the fold's
        # `seed_<i>` per seeded request in request order, fold._header) are lowered to `Input64_v1` batch nodes of one 64-bit value
        # (the root's params are lowered to Input batches, `program._lower_root`; names are not kept).  They canonicalise as
        # ("SEED", lo, hi) on both sides; `seed_index[k]` = the ordinal of seed node k = the index of the request it belongs to.
        # A seed is bound per run: a `RegisteredInput64_v1` node is never one.
        seed_in = sorted(k for k, (fid, n) in self.inputs.items() if input_id(fid) == (RUN, 64) and n == 1)
        self.seed_nodes = set(seed_in)
        self.seed_index = {k: i for i, k in enumerate(seed_in)}
        self.weight_fields = weight_fields  # input node index -> field name
        self.tok_node = tok_node
        # [R17 D90] the request's PEER inputs (a TP rank Program's `peers_<k>` = the other rank's partial of its k-th collective, issue order):
        # `peer_fields[node]` = sorted [(leaf offset, k)] of the fields the Input node holds -- the derived Program lowers one Input per
        # `peers_<k>` (one field at offset 0), the fold ONE homogeneous Record per request (its fields at increasing offsets, program.json
        # `peers_layout`).  A read canonicalises as ("PEER", k, lo, hi) with lo/hi relative to the field on BOTH sides.  Set by the loaders.
        self.peer_fields: dict[int, list[tuple[int, int]]] = {}
        self.peer_index: dict[int, int] = {}       # fold side: peers Input node -> request index (batch_decomp ownership)
        self.supplied_fields: dict[int, list[tuple[int, str, int]]] = {}   # `supplied_inputs`, set by the loaders
        self.supplied_owner: dict[int, int | None] = {}
        self.split_statics = split_selects.as_fold_selects(self.inst)       # a select specialised on S compares as the fold's
        self.k2i = {r["k"]: i for i, r in enumerate(self.inst)}
        self.oracle_rows = oracle_rows
        self._bind_splits()
        self._segment()

    def _node_functions(self):   # the run-form decode's per-node spec id and operand runs; a subclass may return functions equal to these
        from verity.ml.circuits import codec, refs
        return codec._spec_id, refs.runs

    def _bind_splits(self) -> None:
        """[R16 stoch/global, COORD M-0169 (4)] Bind the derived split count S of every top-p TokenSelect (its operand SPLITS_OPERAND).

        Fold side: the operand is a `Const32[S]` instance emitted right before the select -- the constant is REMOVED from the instance
        sequence (it is the value of a prescribed input, not a computation of the Program), its value kept in `splits_values[select node]`
        and the operand canonicalises as ("SPLITS", e, e+1) with e the select's event ordinal in this Prog's segmentation (a request
        projection: the request's e-th sampling event).  Program side: the operand reads an Input batch that is neither the prompt nor a
        seed nor a weight field -- the request-owned `splits` input; `splits_nodes` holds it and every read of it canonicalises as
        ("SPLITS", lo, hi) = ("SPLITS", s, s+1) for event s.  Both sides therefore agree exactly when the fold's e-th select of the request
        carries the constant the Program's event e reads: no value is compared here (GM-01 checks S_t against the live count).  A select
        whose splits operand is neither form (the R15 {V,S} static form has no 6th operand) binds nothing and compares as before; a
        `Const32` read by anything other than a select's splits operand is never removed.  A single-request Program's own binds alike."""
        self.splits_values: dict[int, int] = {}      # select node k -> S (fold side)
        self.splits_nodes: set[int] = set()          # Input nodes read as the splits operand (Program side)
        bound_consts: dict[int, int] = {}            # const instance index -> select node k
        readers: collections.Counter = collections.Counter()   # const node k -> number of operand runs reading it (any instance)
        compact = self.compact
        if not compact:
            for r in self.inst:
                for a in r["args"]:
                    for run in a:
                        if run[0] == "n":
                            readers[run[1]] += 1
        else:
            # [R17 mfast] Only the reader counts of the CANDIDATE constants are consulted below (a Const32 instance read as a select's splits
            # operand), so count per progression: a progression over one node (dI == 0) is `count` reads of it; a node-index progression
            # (dI != 0: the gathers, 97 % of all runs) is one read of every candidate it visits -- found by bisection in the sorted
            # candidates instead of enumerating its runs.  Exactly the record's per-run count on every consulted key.
            import bisect
            cands: set[int] = set()
            for r in self.inst:
                sp = r["spec"].split(":", 1)[-1]
                if not _is_sampling_event(sp) or len(r["args"]) <= SPLITS_OPERAND:
                    continue
                a = r["args"][SPLITS_OPERAND]
                if len(a) == 1 and a[0][0] == "n" and a[0][3] == 1 and a[0][1][0] not in self.inputs:
                    cands.add(a[0][1][0])
            cs = sorted(cands)
            for r in self.inst:
                for a in r["args"]:
                    for run in a:
                        if run[0] != "n":
                            continue
                        idx0, dI, c = run[1][0], run[2][0], run[3]
                        if c == 1 or dI == 0:
                            readers[idx0] += c
                        elif cs:
                            lo, hi = (idx0, idx0 + (c - 1) * dI) if dI > 0 else (idx0 + (c - 1) * dI, idx0)
                            i = bisect.bisect_left(cs, lo)
                            while i < len(cs) and cs[i] <= hi:
                                if (cs[i] - idx0) % dI == 0:
                                    readers[cs[i]] += 1
                                i += 1
        for i, r in enumerate(self.inst):
            sp = r["spec"].split(":", 1)[-1]
            if not _is_sampling_event(sp) or len(r["args"]) <= SPLITS_OPERAND:
                continue
            a = r["args"][SPLITS_OPERAND]
            if len(a) != 1 or a[0][0] != "n":
                continue
            if compact:
                if a[0][3] != 1:
                    continue   # several runs: never the single Const32 read
                k, lo, hi = a[0][1]
            else:
                _, k, lo, hi = a[0]
            if k in self.inputs:
                if k != self.tok_node and k not in self.seed_nodes and k not in self.weight_fields:
                    self.splits_nodes.add(k)
                continue
            j = self.k2i.get(k)
            cv = const_value(self.inst[j]["spec"]) if j is not None else None
            if cv is None or cv[0] != 32 or (lo, hi) != (0, 1) or readers[k] != 1 or self.inst[j]["args"]:
                continue   # not the fold's splits constant (or shared / not a plain Const32): left in place, compared as an instance
            bound_consts[j] = r["k"]
            self.splits_values[r["k"]] = cv[1]
        if bound_consts:
            self.inst = [r for i, r in enumerate(self.inst) if i not in bound_consts]
            self.k2i = {r["k"]: i for i, r in enumerate(self.inst)}
        self.n_splits_consts_bound = len(bound_consts)

    def _segment(self):
        """Steps at TokenSelect boundaries (inclusive)."""
        steps, cur = [], []
        for i, r in enumerate(self.inst):
            cur.append(i)
            if _is_sampling_event(r["spec"]):   # [R15 stoch] TokenSelect_v1 | GumbelTopPTokenSelect_v1 | GumbelTokenSelect_v1
                steps.append(cur)
                cur = []
        if cur:
            steps.append(cur)  # trailing nodes (e.g. the padded program's post-sample muxes / cat)
        self.steps = steps
        self.i2step = {}
        for s, idxs in enumerate(steps):
            for j, i in enumerate(idxs):
                self.i2step[i] = (s, j)
        # [R17 mfast] for the compact canon: `_eq_until[s]` = the last step t >= s such that steps s..t all have len(steps[s]) instances
        # (a gather whose producers are one per step maps uniformly across equal-sized steps)
        n = len(steps)
        eq = [0] * n
        for s in range(n - 1, -1, -1):
            eq[s] = eq[s + 1] if s + 1 < n and len(steps[s + 1]) == len(steps[s]) else s
        self._eq_until = eq

    def canon(self, i: int, unfold: bool = False) -> list[list[tuple]]:
        """Canonical sources of instance i's operands.  Record form: one tuple per run, `(.., lo, hi)`.  Compact form (self.compact):
        one tuple per progression of the MAPPED runs, `(.., lo0, ln, stride, count)` -- the same head per kind, the greedy RLE of the
        record's per-run list (compact.rle_mapped), so two operands are canonical-equal iff the record says so."""
        out = []
        r = self.inst[i]
        bound = r["k"] in getattr(self, "splits_values", {})
        if self.compact:
            from verity_vllm.program.compact import rle_mapped
            for ai, a in enumerate(r["args"]):
                if bound and ai == SPLITS_OPERAND:
                    e = self.i2step[i][0]
                    out.append([("SPLITS", (e, e + 1), (0, 0), 1)])
                    continue
                mapped: list[tuple] = []
                for p in a:
                    mapped.extend(self._srcP(p, unfold))
                out.append(rle_mapped(mapped))
            return out
        for ai, a in enumerate(r["args"]):
            if bound and ai == SPLITS_OPERAND:
                # [M-0169 (4)] the fold's Const32[S] at this select = the Program's `splits` input entry of the same event ordinal
                e = self.i2step[i][0]
                out.append([("SPLITS", e, e + 1)])
                continue
            out.append([self._src(r, unfold) for run in a for r in self._split_peer(run)])
        return out

    def _srcP(self, p: tuple, unfold: bool) -> list[tuple]:
        """[R17 mfast] canonical source(s) of one run PROGRESSION `(space, (idx0, lo0, hi0), (dI, dL, dH), count)`: ONE tuple when the
        record's per-run mapping is provably affine over the progression (the same producer node; producers consecutive inside one
        step; one producer per step across equal-sized steps; a weights window inside one field), else the record's mapping run by run
        (re-normalised by the caller).  Heads and semantics are `_src`'s: the expansion of the result is the record's per-run list."""
        space, base, delta, c = p
        if space != "n":
            return [p]
        idx0, lo0, hi0 = base
        dI, dL, dH = delta
        if idx0 in self.inputs:
            if dI != 0:
                return self._per_run(p, unfold)   # a window sliding over several parameters: never one kind
            if idx0 in self.supplied_fields:
                return SI.srcP(self, p, unfold)
            if idx0 == self.tok_node:
                return [("TOK", (lo0, hi0), (dL, dH), c)]
            if idx0 in self.seed_nodes:
                return [("SEED", (lo0, hi0), (dL, dH), c)]
            if idx0 in getattr(self, "splits_nodes", ()):
                return [("SPLITS", (lo0, hi0), (dL, dH), c)]
            if idx0 in getattr(self, "peer_fields", ()):
                return self._peer_srcP(p, unfold)   # [R17 D90] the k-th received-collective peer input of the request
            if idx0 in self.weight_fields:
                return [("W", _field_name(self.weight_fields[idx0]), (lo0, hi0), (dL, dH), c)]
            wr = getattr(self, "oracle_wmap", None)
            if wr:
                import bisect
                b0 = getattr(self, "input_base", {}).get(idx0, 0)
                last_lo, last_hi = lo0 + (c - 1) * dL, hi0 + (c - 1) * dH
                glo, ghi = min(lo0, last_lo) + b0, max(hi0, last_hi) + b0
                j = bisect.bisect_right(self._wlos, lo0 + b0) - 1
                # uniform iff the fields are disjoint (else the record's per-run bisect may land inside another field) and the whole
                # span of the progression lies in the first run's field -- then every run's own lookup is this field
                if not getattr(self, "oracle_wmap_overlaps", None) and 0 <= j < len(wr) and wr[j][0] <= glo and ghi <= wr[j][1]:
                    off = wr[j][0]
                    return [("W", _field_name(wr[j][2]), (lo0 + b0 - off, hi0 + b0 - off), (dL, dH), c)]
                return self._per_run(p, unfold)
            return [("W", f"input#{idx0}", (lo0, hi0), (dL, dH), c)]
        j0 = self.k2i[idx0]
        if unfold and self.is_mux(j0):
            if dI != 0:
                return self._per_run(p, unfold)
            live_arg = self.inst[j0]["args"][1]
            if len(live_arg) == 1 and live_arg[0][0] == "n" and live_arg[0][3] == 1:   # exactly one run, as the record requires
                i2, lo2, _hi2 = live_arg[0][1]
                return self._srcP(("n", (i2, lo2 + lo0, lo2 + hi0), (0, dL, dH), c), unfold)
            return [("MUX-UNFOLD-FAILED", (j0, lo0, hi0), (0, dL, dH), c)]
        s0, jj0 = self.i2step[j0]
        if c == 1 or dI == 0:
            return [("N", (s0, jj0, lo0, hi0), (0, 0, dL, dH), c)]
        # producers idx0, idx0+dI, ...: instance indices affine in k iff no non-instance node (input / bound const) lies in between
        # (k2i[idx] = idx - #missing below idx, so equal spans mean nothing is missing); then either all in one step (jj affine) or one
        # per `ds` steps of equal size (s affine, jj constant); anything else: the record's mapping per run
        j_last = self.k2i.get(idx0 + (c - 1) * dI)
        if j_last is None or j_last - j0 != (c - 1) * dI:
            return self._per_run(p, unfold)
        sl, jjl = self.i2step[j_last]
        if sl == s0:
            return [("N", (s0, jj0, lo0, hi0), (0, dI, dL, dH), c)]
        size = len(self.steps[s0])
        if size and dI % size == 0:
            ds = dI // size
            lo_s, hi_s = min(s0, sl), max(s0, sl)
            if s0 + (c - 1) * ds == sl and jjl == jj0 and self._eq_until[lo_s] >= hi_s:
                return [("N", (s0, jj0, lo0, hi0), (ds, 0, dL, dH), c)]
        return self._per_run(p, unfold)

    def _per_run(self, p: tuple, unfold: bool) -> list[tuple]:
        """The record's `_src` on every run of a progression (a peers read split as the record splits it), each as a one-run progression."""
        from verity_vllm.program.compact import expand
        return [_asP(self._src(r, unfold)) for run in expand([p]) for r in self._split_peer(run)]

    def _src(self, run, unfold: bool):
        space, idx, lo, hi = run
        if space != "n":
            return (space, idx, lo, hi)
        if idx in self.inputs:
            if idx in self.supplied_fields:
                return SI.src(self, idx, lo, hi)
            if idx == self.tok_node:
                return ("TOK", lo, hi)
            if idx in self.seed_nodes:
                return ("SEED", lo, hi)
            if idx in getattr(self, "splits_nodes", ()):
                return ("SPLITS", lo, hi)   # [M-0169 (4)] the request-owned derived split count, entry per sampling event
            if idx in getattr(self, "peer_fields", ()):
                return self._peer_src(idx, lo, hi)   # [R17 D90] the k-th received-collective peer input of the request
            if idx in self.weight_fields:
                return ("W", _field_name(self.weight_fields[idx]), lo, hi)
            wr = getattr(self, "oracle_wmap", None)
            if wr:  # locate the field range containing lo (offsets are Record-global: node base + in-node offset, [R15 fp8])
                import bisect
                b0 = getattr(self, "input_base", {}).get(idx, 0)
                glo, ghi = lo + b0, hi + b0
                j = bisect.bisect_right(self._wlos, glo) - 1
                if 0 <= j < len(wr) and wr[j][0] <= glo and ghi <= wr[j][1]:
                    return ("W", _field_name(wr[j][2]), glo - wr[j][0], ghi - wr[j][0])
                return ("W?", idx, lo, hi)
            return ("W", f"input#{idx}", lo, hi)
        j = self.k2i[idx]
        if unfold and self.is_mux(j):
            # follow the live branch: SelectI32(live, y, pad) / batch(SelectBf16, live, row, zero) -> operand 1
            live_arg = self.inst[j]["args"][1]
            if len(live_arg) == 1 and live_arg[0][0] == "n":
                if self.compact:   # [R17 mfast] one progression of exactly one run, else not the single-run live operand the rule wants
                    if live_arg[0][3] != 1:
                        return ("MUX-UNFOLD-FAILED", j, lo, hi)
                    i2, lo2, _hi2 = live_arg[0][1]
                else:
                    s2, i2, lo2, hi2 = live_arg[0]
                return self._src(("n", i2, lo2 + lo, lo2 + hi), unfold)
            return ("MUX-UNFOLD-FAILED", j, lo, hi)
        s, jj = self.i2step[j]
        return ("N", s, jj, lo, hi)

    def is_pad(self, i: int) -> bool:
        sp = self.inst[i]["spec"].split(":", 1)[-1]
        return any(sp.startswith(p) for p in PAD_SPECS)

    def is_mux(self, i: int) -> bool:
        sp = self.inst[i]["spec"].split(":", 1)[-1]
        return sp.startswith("SelectI32_v1") or sp.startswith("SelectBf16_v1")


def load_derived(d: str, label: str, cls: type[Prog] = Prog, compact: bool = False) -> Prog:
    desc = json.load(gzip.open(os.path.join(d, "descriptor.json.gz")))
    rep = json.load(open(os.path.join(d, "derive-report.json")))
    prov = rep.get("provenance", rep)
    src = prov.get("source") if isinstance(prov.get("source"), dict) else {}
    # the module's USER inputs precede the weight fields: `example_inputs` records one entry per input (the greedy wrapper's `prompt`;
    # [R15 stoch] the stochastic wrapper's `prompt` + `seed`); a report without it is the one-input greedy wrapper
    user_inputs = src.get("example_inputs") or src.get("inputs") or [[0]]
    n_user = max(1, len(user_inputs))
    reg = SI.derived_registration(prov, n_user, d)   # a `registered_values` derivation's `values`, between them
    fields = {n_user + bool(reg) + w["field"]: w["name"] for w in prov["weights"]}  # derive: user inputs first, then one Input batch per field
    P = cls(label, desc, fields, tok_node=0, compact=compact)
    SI.set_supplied(P, [(n_user, None, reg)] if reg else [], d)
    # [R17 D90] a TP rank Program (derive_step --tp): `tp_links.json` names the user inputs in order (`prompt`, `peers_0`, ... = the k-th
    # collective's peer partial, issue order); each `peers_<k>` is its own Input node (one field at offset 0).  Refused by name when the
    # link table and the report disagree on how many user inputs there are.
    tl = os.path.join(d, "tp_links.json")
    if os.path.exists(tl):
        links = json.load(open(tl))
        names = list(links.get("user_inputs") or [])
        if len(names) != n_user:
            raise ValueError(f"{d}: tp_links.json names {len(names)} user inputs, derive-report has {n_user}")
        for j, nm in enumerate(names):
            if nm.startswith("peers_"):
                if j not in P.inputs:
                    raise ValueError(f"{d}: user input {j} ({nm}) is not an Input node of the Program")
                P.peer_fields[j] = [(0, int(nm[len("peers_"):]))]
        P.tp = {"rank": links.get("rank"), "world": links.get("world")}
        P.tp_links = links.get("links") or []
    return P


def load_oracle(d: str, label: str, cls: type[Prog] = Prog, compact: bool = False) -> Prog:
    # [R17 mfast, rev F-r17-12] the checker reads the FILES OF RECORD and nothing beside them: a compact copy written into the fold dir by
    # the record's producer (unit 7's sidecars, withdrawn) is prover input whose content nothing the checker owns authenticates -- a forged
    # sidecar with the sources' true digests/sizes/sha256 pasted in was accepted (M-0497).  `instances.jsonl` itself may be stored in either
    # form (F2, run_config.INSTANCES_SCHEMA_V2: operand references as progressions); this loader keeps only the parameter-space items --
    # "p" runs and [GLM-5.3] "P" progressions, read without expansion (a scattered weight operand is ~1e6 runs) -- so its semantics do not
    # depend on the form.  The form read is stated (LOAD_NOTES -> impl.config.fold_load) so two documents of one capture under two stores are distinguishable by content.
    # [R17 mfast n2, COORD R17-37 (2) / F-r17-15 (iii)] the two record files are hashed AS READ (one pass: every byte the loader consumes
    # goes through sha256 before it is decoded), and the digests are stated beside the form (LOAD_NOTES -> impl.fold_load[dir].program_sha256 /
    # instances_sha256): a Match document -- the record checker's as much as the fast one's -- is tied to the IMMUTABLE INPUT ARTIFACTS by
    # content, so the audit tool compares the two documents' digests with the row's files (and with each other) instead of trusting that
    # two oracle_dir paths named the same bytes.  ~2-3 s per GB at load; nothing else changes (rows, form handling, the Prog).
    import hashlib
    raw = open(os.path.join(d, "program.json"), "rb").read()
    program_sha = hashlib.sha256(raw).hexdigest()
    pj = json.loads(raw)
    del raw
    desc = pj["program"]
    desc.setdefault("digest", pj.get("digest"))
    rows = []
    ih = hashlib.sha256()
    with open(os.path.join(d, "instances.jsonl"), "rb") as f:
        for line in f:
            ih.update(line)
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if "schema" in o and "node" not in o:   # the header line
                LOAD_NOTES[d] = {"source": "record files", "instances_schema": o.get("schema"),
                                 "operand_refs_form": o.get("operand_refs_form") or "runs", "program_sha256": program_sha}
            if "node" in o:
                # [R16 global, GM-01 memory at LP 4096] only the PARAMETER-space runs ("p": prompt / weights reads -- O(instances)) of each
                # operand are kept from `operand_refs`; the node-space runs ("n": the attention K/V reads, O(LP^2) over a long prefill) are
                # the same references `Prog.inst[*].args` holds and are never read from the rows.  Operand arity is preserved (one list per
                # operand, possibly empty) so `operand_roots[k]` still names operand k below.
                rows.append({"node": o["node"], "step": o.get("step"), "row": o.get("row"), "rows": o.get("rows"), "hint": o.get("hint"),
                             "spec": o.get("spec"), "operand_refs": [[r for r in a if r[0] in ("p", "P")] for a in (o.get("operand_refs") or [])],
                             "pattern": o.get("pattern"),
                             "request": o.get("request"),  # [X-09] the fold's own attribution when recorded
                             "operand_roots": o.get("operand_roots")})  # [R15 fp8] per-operand weight names (multi-weight Computes)
    if d not in LOAD_NOTES:     # a pre-header instances.jsonl (no schema line): the form is the default, the digests are still stated
        LOAD_NOTES[d] = {"source": "record files", "instances_schema": None, "operand_refs_form": "runs", "program_sha256": program_sha}
    LOAD_NOTES[d]["instances_sha256"] = ih.hexdigest()
    return _finish_oracle(d, label, pj, desc, rows, cls, compact)


LOAD_NOTES: dict[str, dict] = {}   # fold dir -> {"source": "record files", "instances_schema": <header schema>, "operand_refs_form": runs|progressions}


def _finish_oracle(d: str, label: str, pj: dict, desc: dict, rows: list[dict], cls: type[Prog] = Prog, compact: bool = False) -> Prog:
    """The rest of load_oracle over the retained rows."""
    # weight layout: the WEIGHTS parameter's offset ranges -> weight name from the hints ("module/param.name"). The fold lays
    # out one prompt parameter per request followed by the weights (`fold._header`): weights = param n_input_nodes-1 (== 1
    # only at B=1; [R13 X-09] the hard-coded 1 read request 1's PROMPT as the weight map at B>1).
    n_in = pj.get("n_input_nodes", 2)
    # [R15 fp8] the weights Record is the LAST root parameter (fold contract) -- its index is the largest `p` index any instance reads,
    # not `n_input_nodes - 1`: a Record that is not width-homogeneous (FP8: e4m3 weights, f32 block scales, bf16 norms) is encoded as
    # ONE Input node per leaf field (436 for Qwen3-4B-FP8 at B=1), so the node count no longer says how many prompt parameters there are.
    p_idx = {i for o in rows for a in (o.get("operand_refs") or []) for r in a for i in _p_indices(r)}
    w_param = max(p_idx) if p_idx else n_in - 1
    # [R17 D90] the fold states its parameter layout by name (program.json `params`): the weights index is read from it when present
    # (exact, even for a fold whose instances read no weight -- a fixture) and cross-checked against the instances' largest "p" index.
    names = [n for n, _t in (pj.get("params") or [])]
    if names:
        if names[-1] != "weights":
            raise ValueError(f"{d}: program.json params end with {names[-1]!r}, the fold contract puts `weights` LAST")
        if p_idx and max(p_idx) != len(names) - 1:
            raise ValueError(f"{d}: instances read parameter {max(p_idx)} as the largest, program.json puts `weights` at {len(names) - 1}")
        w_param = len(names) - 1
    p = cls(label, desc, {}, tok_node=0, oracle_rows=rows, compact=compact)
    # [R17 D90] a tensor-parallel rank fold: one homogeneous `peers_<i>` Record per request = ONE Input node at the parameter's index (every
    # parameter before the weights is homogeneous, one node each); its fields (issue order, leaf offsets) come from `peers_layout`.  The k of a
    # field is its ordinal among the REQUEST's fields = the rank Program's `peers_<k>` for that request.
    for ent in pj.get("peers_layout") or []:
        node = int(ent["param"])
        if node not in p.inputs:
            raise ValueError(f"{d}: peers parameter {ent.get('name')!r} (index {node}) is not an Input node of the Program")
        flds = sorted((int(f["offset"]), int(f["ordinal"])) for f in ent.get("fields") or [])
        if not flds:
            raise ValueError(f"{d}: peers parameter {ent.get('name')!r} has no fields")
        if sum(1 for _o, k in flds) != len({k for _o, k in flds}):
            raise ValueError(f"{d}: peers parameter {ent.get('name')!r} has duplicate field ordinals")
        p.peer_fields[node] = flds
        p.peer_index[node] = int(ent["request_index"])
    p.tp = pj.get("tp")
    w_nodes = SI.load_oracle_supplied(p, pj, d)   # a `registered_values` fold's supplied parameters
    del pj, desc   # [R16 global] the descriptor JSON (12.5 GB on disk for the 4096/512 fold) is not needed past the decode
    # Input node -> offset of its first leaf inside the weights parameter (0 for a single homogeneous Record node -- the pre-R15 layout):
    # the codec emits the prompt parameters' nodes first (one each, `w_param` of them), then the Record's leaf batches in field order.
    w_nodes = w_nodes if w_nodes is not None else {k for k in p.inputs if k >= w_param}
    base, acc = {}, 0
    for k in sorted(p.inputs):
        if k in w_nodes:
            base[k] = acc
            acc += int(p.inputs[k][1])
    p.input_base = base
    # Field = the union of every range the instances read under one weight name (the fold's hint `module/param`; the
    # RoPE table is hinted `module/rope.q|k` per position -> the module's `cos_sin_cache`). Equal ranges keep the first name.
    # [R15 fp8] a Compute that reads several weights (the block-scaled FP8 GEMM: `weight` AND `weight_scale_inv`) records the name of
    # each weight operand in `operand_roots` (fold meta, from `Compute.roots`); operand k is then spanned under ITS name, never
    # the hint's -- the union under one name would fuse two fields into a range the derived program does not have.
    span: dict[str, list[int]] = {}
    for o in rows:
        hint = o.get("hint") or ""
        mod, _, tail = hint.rpartition("/")
        if tail in ("rope.q", "rope.k"):
            name = f"{mod}.cos_sin_cache"
        else:
            name = tail or hint
        oroots = o.get("operand_roots") or []
        for k, a in enumerate(o.get("operand_refs") or []):
            nm = oroots[k] if k < len(oroots) and oroots[k] else name
            for r in a:
                if (w := _p_w_span(r, w_param)) is not None:
                    sp = span.setdefault(nm, [w[0], w[1]])
                    sp[0], sp[1] = min(sp[0], w[0]), max(sp[1], w[1])
    byrange: dict[tuple[int, int], str] = {}
    for name, (lo, hi) in span.items():
        byrange.setdefault((lo, hi), name)
    p.oracle_wmap = sorted((lo, hi, nm) for (lo, hi), nm in byrange.items())
    # fields must not overlap (else the map is ambiguous): report, never guess
    p.oracle_wmap_overlaps = [(a, b) for a, b in zip(p.oracle_wmap, p.oracle_wmap[1:]) if a[1] > b[0]]
    p._wlos = [lo for lo, _, _ in p.oracle_wmap]
    p.n_input_nodes = n_in
    # [R16 global, GM-01 memory at LP 4096] the (parameter-space) `operand_refs` served only `w_param` and the weight-field map above; they
    # are dropped from the retained rows.  `operand_roots` (one short name per operand) stays.  Consumers of `oracle_rows` (batch_decomp,
    # global_match.canonicalize_record) read node/step/row/rows/request/spec.
    for o in rows:
        o["operand_refs"] = None
    return p


from verity_vllm.program.compact import expand_mixed as _expand_mixed, n_crosses_step as _n_crosses_step, p_item_indices as _p_indices, p_item_weight_span as _p_w_span  # noqa: E402  [R17 mfast; GLM-5.3 "P"]


def _norm_T(spec: str) -> str:
    return _T_RE.sub("{T=*,", spec)


# [R18 x09, F-r18-x09-1] The divergence REPORT of a step whose spec sequences differ.  The record ran difflib.SequenceMatcher
# (autojunk=False) over the WHOLE step: O(n*m) worst case, and a prefill step of an LP-row request-wrapper Program is ONE step of
# LP x layers x ops instances (329,730 for LP 1024 x 32 layers).  When the sequences differ throughout -- a Build for another target's
# kernel families (#60: Hopper Build, L40S fold), or a displaced co-scheduled request under the `dag` within-step criterion -- that one
# alignment ran 40 min to 3 h per leg at 100 % CPU, flat memory (measured n^2.2: 32 K items 13.8 s, 329.7 K items hours).  No verdict
# depends on it: spec_seq_equal / spec_multiset_equal / canonical_sha256 / the DAG multiset carry the check; the alignment only fills
# `first_divergences_T_normalised` ("the first 6 divergent opcodes, 3 examples each") and `n_divergent_opcodes`.  Now the common prefix
# and suffix are located in O(n) and the matcher runs over the differing CORE only, over at most `_DIVERGENCE_WINDOW` items of each
# side -- the first divergences lie at the start of the core, exactly where the report looks.  A core longer than the window is reported
# truncated BY NAME (`divergence_window`): the count then covers the window, never a guess.  An equal step never enters this path, so a
# PASS record's bytes are unchanged.  The matcher's cost is ~ W^2 x matching blocks (measured on a 9-op row with 5 of 9 equal: W 512
# 0.1 s, 1024 0.8 s, 2048 6 s, 4096 49 s), so the window is 1024: <= 1 s per divergent step, whatever the step holds.
_DIVERGENCE_WINDOW = 1024


def _divergence_report(a: list, b: list) -> dict:
    """Non-equal opcodes of `a` vs `b` (T-normalised spec sequences of one step) in FULL-sequence coordinates, bounded (see above)."""
    n, m = len(a), len(b)
    lim = min(n, m)
    p = 0
    while p < lim and a[p] == b[p]:
        p += 1
    s = 0
    while s < lim - p and a[n - 1 - s] == b[m - 1 - s]:
        s += 1
    ca, cb = a[p:n - s], b[p:m - s]
    W = _DIVERGENCE_WINDOW
    truncated = len(ca) > W or len(cb) > W
    sm = difflib.SequenceMatcher(a=ca[:W], b=cb[:W], autojunk=False)
    ops = [(tag, i1 + p, i2 + p, j1 + p, j2 + p) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]
    out = {"ops": ops}
    if truncated:
        out["window"] = {"common_prefix": p, "common_suffix": s, "core_a": len(ca), "core_b": len(cb), "window": W, "truncated": True}
    return out


def compare_steps(A: Prog, B: Prog, unfold_a: bool = False, *, canon_step_relative=None, canonical_hash=None) -> dict:
    """Spec sequences + wiring per step. `unfold_a`: drop A's pad-kind instances and follow muxes (padded vs flat)."""
    canon_step_relative, canonical_hash = canon_step_relative or _canon_step_relative, canonical_hash or _canonical_hash
    def seq(P: Prog, unfold: bool):   # each step's instances, pad kinds dropped when unfolding; empty segments dropped
        return [k for k in ([i for i in idxs if not (unfold and P.is_pad(i))] for idxs in P.steps) if k]

    sa, sb = seq(A, unfold_a), seq(B, False)
    def _side(P: Prog, seqs):
        d = {"label": P.label, "digest": P.digest, "instances": len(P.inst), "steps": len(seqs), "inputs": len(P.inputs)}
        # [M-0169 (4)] the derived split count binding, when present on this side (fold: bound Const32[S] per select; Program: splits input)
        sv = getattr(P, "splits_values", {})
        if sv:
            d["splits_bound"] = {"form": "Const32[S] per top-p select (removed from the sequence, value kept)", "selects": len(sv),
                                 "values": sorted(set(sv.values()))}
        elif getattr(P, "splits_nodes", ()):
            d["splits_bound"] = {"form": "request-owned `splits` Input read per event", "nodes": sorted(P.splits_nodes)}
        return d

    res = {"a": _side(A, sa), "b": _side(B, sb), "steps": [], "totals": collections.Counter()}
    for s in range(max(len(sa), len(sb))):
        ia = sa[s] if s < len(sa) else []
        ib = sb[s] if s < len(sb) else []
        spa = [A.inst[i]["spec"] for i in ia]
        spb = [B.inst[i]["spec"] for i in ib]
        st = {"step": s, "n_a": len(ia), "n_b": len(ib), "spec_seq_equal": spa == spb,
              "spec_seq_equal_T_normalised": [_norm_T(x) for x in spa] == [_norm_T(x) for x in spb]}
        ca, cb = collections.Counter(spa), collections.Counter(spb)
        st["spec_multiset_equal"] = ca == cb
        if ca != cb:
            st["only_in_a"] = {k: v for k, v in (ca - cb).items()}
            st["only_in_b"] = {k: v for k, v in (cb - ca).items()}
        if spa != spb:
            div = _divergence_report([_norm_T(x) for x in spa], [_norm_T(x) for x in spb])   # [R18 x09] bounded, see _divergence_report
            ops = div["ops"]
            st["first_divergences_T_normalised"] = [{"op": t, "a": spa[i1:min(i2, i1 + 3)], "b": spb[j1:min(j2, j1 + 3)], "a_range": [i1, i2], "b_range": [j1, j2]}
                                                    for t, i1, i2, j1, j2 in ops[:6]]
            st["n_divergent_opcodes"] = len(ops)
            if "window" in div:
                st["divergence_window"] = div["window"]
        # wiring: 1:1 when aligned (T-normalised alignment suffices: same instance order)
        if st["spec_seq_equal_T_normalised"]:
            same = diff = 0
            by_spec_arg: collections.Counter = collections.Counter()
            examples = []
            kv_cross = 0
            for j, (i, k) in enumerate(zip(ia, ib)):
                # remap producer indices through the (possibly filtered) sequences
                cA = canon_step_relative(A, i, sa, unfold_a)
                cB = canon_step_relative(B, k, sb, False)
                for ai, (x, y) in enumerate(zip(cA, cB)):
                    if x == y:
                        same += 1
                    else:
                        diff += 1
                        by_spec_arg[f"{_norm_T(A.inst[i]['spec'])}#arg{ai}"] += 1
                        if len(examples) < 12:
                            examples.append({"j": j, "spec": A.inst[i]["spec"], "arg": ai, "a": _short(_runs_of(A, x)), "b": _short(_runs_of(B, y))})
                    if any(_n_crosses_step(t, s) for t in x):
                        kv_cross += 1
                if len(cA) != len(cB):
                    diff += abs(len(cA) - len(cB))
                    by_spec_arg[f"{_norm_T(A.inst[i]['spec'])}#arity"] += 1
            st["wiring"] = {"operands_equal": same, "operands_different": diff, "different_by_spec_arg": dict(by_spec_arg.most_common(20)),
                            "cross_step_operands_in_a": kv_cross, "examples": examples}
        res["steps"].append(st)
        res["totals"]["steps_spec_equal"] += int(st["spec_seq_equal"])
        res["totals"]["steps_spec_equal_T_normalised"] += int(st["spec_seq_equal_T_normalised"])
        if "wiring" in st:
            res["totals"]["operands_equal"] += st["wiring"]["operands_equal"]
            res["totals"]["operands_different"] += st["wiring"]["operands_different"]
    res["totals"] = dict(res["totals"])
    res["canonical_sha256"] = {"a": canonical_hash(A, sa, unfold_a), "b": canonical_hash(B, sb, False)}
    res["canonical_sha256"]["equal"] = res["canonical_sha256"]["a"] == res["canonical_sha256"]["b"]
    split_selects.hold(A, B, res, constant_splits_agree(A, B))
    return res


def _canonical_hash(P: Prog, seqs: list[list[int]], unfold: bool, canon_step_relative=None) -> str:
    """sha256 of the expanded instance sequence in canonical form: (spec, form, canonical operand sources, out type) per
    instance, producers as (step, index-within-step) — independent of input layout, node numbering and root naming."""
    import hashlib
    canon_step_relative = canon_step_relative or _canon_step_relative
    h = hashlib.sha256()
    compact = getattr(P, "compact", False)
    for idxs in seqs:
        for i in idxs:
            r = P.inst[i]
            cs = canon_step_relative(P, i, seqs, unfold)
            if compact:   # [R17 mfast] the record's BYTES: the run-per-tuple serialisation (canonical_sha256 is on X-09's record)
                cs = [_expand_mixed(c) for c in cs]
            h.update(json.dumps([r["spec"], r["form"], [list(map(list, c)) for c in cs], r["out"]],
                                separators=(",", ":"), default=str).encode())
        h.update(b"|step|")
    return h.hexdigest()


def _short(x):
    return [list(t) if len(str(t)) < 120 else str(t)[:120] for t in x][:6]


def _runs_of(P: Prog, canon_arg) -> list:
    """A canonical operand as the record prints it: per run.  Identity for a record-form Prog; expands a compact Prog's progressions
    (only the first 6 runs are ever printed -- bounded work)."""
    if not getattr(P, "compact", False):
        return canon_arg
    return _expand_mixed(canon_arg)[:6]


def _canon_step_relative(P: Prog, i: int, seqs: list[list[int]], unfold: bool):
    """Canonical operands with node producers expressed as (N, step, index within the FILTERED step sequence)."""
    pos = getattr(P, "_pos_cache", None)
    if pos is None or P._pos_cache_key != (unfold, id(seqs)):
        pos = {}
        for s, idxs in enumerate(seqs):
            for j, ii in enumerate(idxs):
                pos[ii] = (s, j)
        P._pos_cache, P._pos_cache_key = pos, (unfold, id(seqs))
    out = []
    compact = getattr(P, "compact", False)
    for arg in P.canon(i, unfold):
        if compact:   # [R17 mfast] the record's per-run tuples: each run's producer is remapped on its own
            arg = _expand_mixed(arg)
        row = []
        for t in arg:
            if t[0] == "N":
                s, jj = t[1], t[2]
                ii = P.steps[s][jj]
                if ii in pos:
                    row.append(("N",) + pos[ii] + t[3:])
                else:
                    row.append(("N-PAD", s, jj) + t[3:])  # producer is a pad-kind node that was not unfolded
            else:
                row.append(t)
        out.append(tuple(row))
    return out


def audit_padding(P: Prog) -> dict:
    """The padded program's mux discipline: per decode step, one live bit keys 1 SelectI32 + 2*NL SelectBf16 batches,
    against one Const16 zero; guards = I32Eq + BitNot + BitAnd per event. Segments are TokenSelect-terminated, so segment
    s+1 holds event s's guard (it follows the sample) and decode step s's row muxes (the step-boundary cache clear)."""
    per_step = []
    for s, idxs in enumerate(P.steps):
        muxes = [i for i in idxs if P.is_mux(i)]
        if not muxes:
            per_step.append({"step": s, "muxes": 0})
            continue
        lives = collections.Counter()
        zeros = collections.Counter()
        kinds = collections.Counter()
        for i in muxes:
            r = P.inst[i]
            kinds[r["spec"]] += 1
            if getattr(P, "compact", False):
                from verity_vllm.program.compact import expand, first_runs
                lives[str(expand(r["args"][0]))] += 1
                if len(r["args"]) > 2 and "SelectBf16" in r["spec"]:
                    zeros[str(first_runs(r["args"][2], 1))] += 1
            else:
                lives[str(r["args"][0])] += 1
                if len(r["args"]) > 2 and "SelectBf16" in r["spec"]:
                    zeros[str(r["args"][2][:1])] += 1  # the zero row: one broadcast Const16 per step
        guards = collections.Counter(P.inst[i]["spec"] for i in idxs if P.is_pad(i) and not P.is_mux(i))
        per_step.append({"step": s, "muxes": len(muxes), "mux_kinds": dict(kinds), "distinct_live_refs": len(lives), "distinct_zero_refs": len(zeros),
                         "guards_and_consts": dict(guards)})
    return {"per_step": per_step}


def spec_histograms(P: Prog) -> dict:
    h = collections.Counter(r["spec"] for r in P.inst)
    hT = collections.Counter(_norm_T(r["spec"]) for r in P.inst)
    return {"exact": dict(sorted(h.items())), "T_normalised": dict(sorted(hT.items()))}


def per_step_templates(P: Prog) -> list[dict]:
    out = []
    for s, idxs in enumerate(P.steps):
        h = collections.Counter(_norm_T(P.inst[i]["spec"]) for i in idxs)
        out.append({"step": s, "n": len(idxs), "T_normalised_histogram": dict(sorted(h.items()))})
    return out


@dataclass(frozen=True, kw_only=True)
class Options:
    a: str = option("--a", help="derived dir (descriptor.json.gz + derive-report.json)")
    b: str = option("--b", help="oracle dir (program.json + instances.jsonl) or derived dir")
    out: str = option("--out")
    unfold_a: bool = option("--unfold-a", default=False, action="store_true",
                            help="A is a padded program: drop pad kinds, follow muxes")
    label_a: str = option("--label-a", default="derived")
    label_b: str = option("--label-b", default="oracle")


def main(a: Options) -> int:
    t0 = time.time()
    A = load_derived(a.a, a.label_a)
    B = load_oracle(a.b, a.label_b) if os.path.exists(os.path.join(a.b, "program.json")) else load_derived(a.b, a.label_b)
    res = {"schema": "cb-a/compare/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "a_dir": a.a, "b_dir": a.b, "unfold_a": a.unfold_a}
    res["digests"] = {"a": A.digest, "b": B.digest, "equal": A.digest == B.digest}
    res["histograms"] = {"a": spec_histograms(A), "b": spec_histograms(B)}
    ha, hb = res["histograms"]["a"]["T_normalised"], res["histograms"]["b"]["T_normalised"]
    res["histogram_diff_T_normalised"] = {k: [ha.get(k, 0), hb.get(k, 0)] for k in sorted(set(ha) | set(hb)) if ha.get(k, 0) != hb.get(k, 0)}
    res["templates"] = {"a": per_step_templates(A), "b": per_step_templates(B)}
    if B.oracle_rows is not None:
        # cross-check the TokenSelect segmentation against the oracle's per-instance step field
        mism = 0
        node_step = {o["node"] + B.n_input_nodes: o["step"] for o in B.oracle_rows}
        for i, r in enumerate(B.inst):
            s, _ = B.i2step[i]
            if node_step.get(r["k"]) != s:
                mism += 1
        res["oracle_segmentation_mismatches"] = mism
        res["oracle_weight_map"] = {"n_ranges": len(B.oracle_wmap), "first": B.oracle_wmap[:5], "overlaps": B.oracle_wmap_overlaps[:5],
                                    "n_overlaps": len(B.oracle_wmap_overlaps)}
        B.weight_fields_ranges = B.oracle_wmap
    res["compare"] = compare_steps(A, B, unfold_a=a.unfold_a)
    if a.unfold_a:
        res["padding_audit_a"] = audit_padding(A)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    t = res["compare"]["totals"]
    print(json.dumps({"digests_equal": res["digests"]["equal"], "canonical_equal": res["compare"]["canonical_sha256"]["equal"], "steps_a": res["compare"]["a"]["steps"], "steps_b": res["compare"]["b"]["steps"],
                      "instances": [res["compare"]["a"]["instances"], res["compare"]["b"]["instances"]], "totals": t,
                      "histogram_diff_T_normalised": res["histogram_diff_T_normalised"], "seconds": res["seconds"]}, indent=1))
    return 0
