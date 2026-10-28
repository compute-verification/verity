"""A decoded root body as the comparator reads it (`RootView`): its instance sequence, segmented into engine steps at the sampling
events, and the canonical source of every operand run (the forms are `program_compare`'s, its docstring states them)."""

from __future__ import annotations

import collections
import re
import sys

from verity.ml.circuits.program import RUN, input_id
from verity_vllm.check.match import compiler_form
from verity_vllm.check.match import prefix_reads as PX
from verity_vllm.check.match import split_selects
from verity_vllm.check.match import supplied_inputs as SI
from verity_vllm.check.match.canonical_operands import as_progression as _asP, weight_field_name as _field_name
from verity_vllm.check.match.peer_reads import PeerReads
from verity_vllm.program.sampling_event import is_sampling_event as _is_sampling_event  # the step-boundary families

PAD_SPECS = ("Const", "I32Eq_v1", "BitNot_v1", "BitAnd_v1", "SelectI32_v1", "SelectBf16_v1")
# the derived top-p split count S: the fold emits it as `Const32[S]` "splits", the 6th operand
# (index 5) of the top-p select, the one sampling event with a 6th operand (patterns.TokenSelectGumbelTopP._stage3); the request Program
# takes it as the request-owned INPUT `splits` (int32 [T+1], after the seed) and event s reads `splits[s:s+1]` (vllm_meta.StochasticRequest).
# compare BINDS the two: the fold's constant at the request's event s IS the value of the Program's input entry s -- canonicalised
# ("SPLITS", s, s+1) on both sides; the constant's VALUE is kept (`RootView.splits_values`, by select node) for GM-01's per-engine-step
# relation check (`global_match.sampler_geometry_check`: S_t == SplitsFor_v1(|live(t)|, num_SMs)).  Nothing about S is a static of the
# Definition.
_CONST_RE = re.compile(r"^Const(\d+)\[0x([0-9a-fA-F]+)\]_v1$")
SPLITS_OPERAND = 5


def const_value(spec: str) -> tuple[int, int] | None:
    """(width, value) of a codec `Const<w>[0x..]_v1` spec (a fold / Program constant instance); None for any other spec."""
    m = _CONST_RE.match(spec.split(":", 1)[-1])
    return (int(m.group(1)), int(m.group(2), 16)) if m else None


def _registry():
    """The registry every module loaded (`verity.ml.definitions.registry.load`); a module that fails to import is named on stderr (its
    Programs, e.g. Hopper's under Python < 3.12, then fail to decode with that reason)."""
    from verity.ml.definitions.registry.catalog import load
    reg, errors = load()
    for e in errors:
        print(f"[compare] registry module unavailable ({e}): Programs naming its Definitions will not decode", file=sys.stderr)
    return reg


class RootView(PeerReads, PX.PrefixInput):
    """A decoded root body as an instance sequence."""

    compact = False   # operand representation: runs (record) | run progressions (cb_a.compact); set per instance in __init__
    fold_form: dict | None = None   # a fold's `compiler_form.fold_in_form` record, set by its loader's caller

    def __init__(self, label: str, desc: dict, weight_fields: dict[int, str], tok_node: int, oracle_rows: list[dict] | None = None,
                 compact: bool = False):
        from verity.ml.circuits.codec import program_digest
        self.label = label
        self.compact = bool(compact)
        self.digest = desc.get("digest") or program_digest(desc)
        self.inputs: dict[int, tuple[str, int]] = {}
        self.inst: list[dict] = []  # k, spec, form, args=[runs | progressions], out
        if self.compact:
            # the root body read straight from the descriptor JSON into run progressions: no Node / Coll / Refs objects for
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
            descriptor_id, runs = self._node_functions()
            dec = decode_program(desc, _registry())
            root = dec.root
            body = root.body if hasattr(root, "body") else root.defn.body
            for k, n in enumerate(body.nodes):
                fid = descriptor_id(n.fn)
                if n.form == "batch" and input_id(fid) is not None:
                    self.inputs[k] = (fid, n.n)
                    continue
                self.inst.append({"k": k, "spec": fid if n.form != "batch" else f"batch:{fid}", "form": n.form,
                                  "args": [runs(a.refs) for a in n.args], "out": str(n.out)})
            # the decoded IR (node objects + Refs trees, O(LP^2) leaves for a long prefill's K/V gathers) is not retained: comparisons read
            # `inst` only, and keeping it doubled the resident Program (GM-01 > 351 GiB at LP 4096).  `n_nodes` keeps the count.
            self.n_nodes = len(body.nodes)
            self.nodes = None
            self.param_names = [n for n, _ in getattr(body, "params", ())]   # empty for a lowered root; kept for callers that inspect it
            del dec, root, body
        if len(self.inst) > 50_000:   # the decoded body is a reference cycle (nodes <-> body); collect it now, not at an arbitrary later point
            import gc
            gc.collect()
        # the seed parameters (a stochastic Program's declared runtime inputs: the request wrapper's `seed`, the fold's
        # `seed_<i>` per seeded request in request order, fold._header) are lowered to `Input64_v1` batch nodes of one 64-bit value
        # (the root's params are lowered to Input batches, `program._lower_root`; names are not kept).  They canonicalise as
        # ("SEED", lo, hi) on both sides; `seed_index[k]` = the ordinal of seed node k = the index of the request it belongs to.
        # A seed is bound per run: a `RegisteredInput64_v1` node is never one.
        seed_in = sorted(k for k, (fid, n) in self.inputs.items() if input_id(fid) == (RUN, 64) and n == 1)
        self.seed_nodes = set(seed_in)
        self.seed_index = {k: i for i, k in enumerate(seed_in)}
        self.weight_fields = weight_fields  # input node index -> field name
        self.tok_node = tok_node
        # the request's PEER inputs (a TP rank Program's `peers_<k>` = the other rank's partial of its k-th collective, issue order):
        # `peer_fields[node]` = sorted [(leaf offset, k)] of the fields the Input node holds -- the derived Program lowers one Input per
        # `peers_<k>` (one field at offset 0), the fold ONE homogeneous Record per request (its fields at increasing offsets, program.json
        # `peers_layout`).  A read canonicalises as ("PEER", k, lo, hi) with lo/hi relative to the field on BOTH sides.  Set by the loaders.
        self.peer_fields: dict[int, list[tuple[int, int]]] = {}
        self.peer_index: dict[int, int] = {}       # fold side: peers Input node -> request index (batch_decomp ownership)
        # a rank Program's WITNESS inputs (a user input the link table names neither a prompt nor a peer input: the MoE align `schedule`):
        # Input node -> its name; a read canonicalises as ("WIT", name, lo, hi), never as an unnamed weight.  Set by `load_derived`.
        self.witness_fields: dict[int, str] = {}
        self.witness_index: dict[int, int] = {}    # fold side: witness Input node -> request index (batch_decomp ownership)
        self.supplied_fields: dict[int, list[tuple[int, str, int]]] = {}   # `supplied_inputs`, set by the loaders
        self.supplied_owner: dict[int, int | None] = {}
        self.split_statics = split_selects.as_fold_selects(self.inst)       # a select specialised on S compares as the fold's
        self.k2i = {r["k"]: i for i, r in enumerate(self.inst)}
        self.oracle_rows = oracle_rows
        self._bind_splits()
        self._segment()

    def _node_functions(self):   # the run-form decode's per-node spec id and operand runs; a subclass may return functions equal to these
        from verity.ml.circuits import codec, refs
        return codec.descriptor_id, refs.runs

    def _bind_splits(self) -> None:
        """Bind the derived split count S of every top-p TokenSelect (its operand SPLITS_OPERAND).
        Fold side: the `Const32[S]` emitted right before the select is REMOVED from the sequence (a prescribed input's value), kept in
        `splits_values[select node]`, and the operand canonicalises as ("SPLITS", e, e+1), e the select's event ordinal.  Program side:
        the request-owned `splits` Input (neither prompt, seed nor weight; `splits_nodes`) read as ("SPLITS", s, s+1) for event s.  Both
        agree exactly when the fold's e-th select carries the constant the Program's event e reads (GM-01 checks S_t against the live
        count).  Any other select binds nothing; a `Const32` read by anything but a splits operand is never removed."""
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
            # Only the reader counts of the CANDIDATE constants are consulted below (a Const32 instance read as a select's splits
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

    def to_compiler_form(self) -> dict:
        """Put this fold's instances in the registered compiler's form (`compiler_form.compiled`) and resegment them; returns what
        that dropped, swapped and narrowed.  Call it before any comparison reads the sequence."""
        self.inst, facts = compiler_form.compiled(self.inst, self.compact)
        self.k2i = {r["k"]: i for i, r in enumerate(self.inst)}
        self.__dict__.pop("_fast_canon", None)
        self._segment()
        return facts

    def _segment(self):
        """Steps at TokenSelect boundaries (inclusive)."""
        steps, cur = [], []
        for i, r in enumerate(self.inst):
            cur.append(i)
            if _is_sampling_event(r["spec"]):   # TokenSelect_v1 | GumbelTopPTokenSelect_v1 | GumbelTokenSelect_v1
                steps.append(cur)
                cur = []
        if cur:
            steps.append(cur)  # trailing nodes (e.g. the padded program's post-sample muxes / cat)
        self.steps = steps
        self.i2step = {}
        for s, idxs in enumerate(steps):
            for j, i in enumerate(idxs):
                self.i2step[i] = (s, j)
        # for the compact canon: `_eq_until[s]` = the last step t >= s such that steps s..t all have len(steps[s]) instances
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
                out.append(rle_mapped(PX.merge_reads(mapped, True) if self.pfx else mapped))
            return out
        for ai, a in enumerate(r["args"]):
            if bound and ai == SPLITS_OPERAND:
                # the fold's Const32[S] at this select = the Program's `splits` input entry of the same event ordinal
                e = self.i2step[i][0]
                out.append([("SPLITS", e, e + 1)])
                continue
            mapped = [self._src(r, unfold) for run in a for r in self._split_peer(run)]
            out.append(PX.merge_reads(mapped, False) if self.pfx else mapped)
        return out

    def _srcP(self, p: tuple, unfold: bool) -> list[tuple]:
        """Canonical source(s) of one run PROGRESSION `(space, (idx0, lo0, hi0), (dI, dL, dH), count)`: ONE tuple when the
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
                return self._peer_srcP(p, unfold)   # the k-th received-collective peer input of the request
            if idx0 in self.pfx:
                return self._pfx_srcP(p, unfold)
            if idx0 in getattr(self, "witness_fields", ()):
                return [("WIT", self.witness_fields[idx0], (lo0, hi0), (dL, dH), c)]
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
                return ("SPLITS", lo, hi)   # the request-owned derived split count, entry per sampling event
            if idx in getattr(self, "peer_fields", ()):
                return self._peer_src(idx, lo, hi)   # the k-th received-collective peer input of the request
            if idx in self.pfx:
                return self._pfx_src(idx, lo, hi)
            if idx in getattr(self, "witness_fields", ()):
                return ("WIT", self.witness_fields[idx], lo, hi)
            if idx in self.weight_fields:
                return ("W", _field_name(self.weight_fields[idx]), lo, hi)
            wr = getattr(self, "oracle_wmap", None)
            if wr:  # locate the field range containing lo (offsets are Record-global: node base + in-node offset,)
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
                if self.compact:   # one progression of exactly one run, else not the single-run live operand the rule wants
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
