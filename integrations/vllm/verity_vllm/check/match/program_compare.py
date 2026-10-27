"""Structural / unfolding comparator (CB-A): a derived Program (flat request or padded rectangle) against the accepted
record's Program, or a padded Program against its flat counterpart. Comparison ONLY — nothing here feeds construction.

Unit of comparison: the root body's *instance sequence* (every non-Input root node: spec id, form, operand reference
runs), segmented into engine steps at `TokenSelect` boundaries (one sampling event per step; the oracle's per-instance
`step` field cross-checks the segmentation). Per step:

  1. spec sequence — exact; then with attention `T` normalised (the only per-position static);
  2. operand wiring — every operand run is canonicalised to a source: (`W`, weight field name, lo, hi) for an input
     weight, (`TOK`, lo, hi) for the prompt, (`WIT`, name, lo, hi) for a rank Program's witness input (`load_derived`),
     (`N`, producer step, producer index-within-step, lo, hi) for a node
     output — and compared 1:1 when the spec sequences align (the weight name comes from each program's own record:
     the derive report's field list / the oracle's `instances.jsonl` hints); cross-step (`KV`) sources are reported apart;
  3. unfolding (padded vs flat) — the padded program's pad-kind instances (`Const*`, `I32Eq`, `BitNot`, `BitAnd`,
     `SelectI32`, `batch(SelectBf16)`) are removed, every operand that reads a mux is followed through the mux's
     live branch (`a`) to the underlying producer, and the result is compared with the flat program as in 1–2. The
     muxes themselves are audited: per decode step ONE live bit keys the token slot and every layer's K and V row,
     against ONE zero gate; a mux anywhere else is reported (a real mismatch, never normalised).

Every difference is reported with counts and examples; nothing is dropped silently.  `--operands progressions` holds both sides'
operands in compact form (program.compact), for Programs whose strided reads are ~1e6 runs each (GLM-5.3's W_UK_T / W_UV).

    verity-vllm program-compare --a <dir with descriptor.json.gz + derive-report.json> \\
        --b <oracle dir with program.json + instances.jsonl | derived dir> --out compare.json [--unfold-a | --operands progressions]
"""

from __future__ import annotations

import collections
import difflib
import json
import os
import re
import time
from dataclasses import dataclass

from verity_vllm.check.match import compiler_form as CF
from verity_vllm.check.match import split_selects
from verity_vllm.check.match.instance_sequence import Prog
from verity_vllm.check.match.program_loaders import load_derived, load_oracle
from verity_vllm.check.match.sampler_geometry import constant_splits_agree
from verity_vllm.config import UsageError, option
from verity_vllm.program.compact import expand_mixed as _expand_mixed, hash_operand as _hash_operand, n_crosses_step as _n_crosses_step  #
# Imported from here by modules open PRs edit (tests/program/test_compact.py, tests/check/test_glm53_rank_rehearsal.py) and
# by check/match/resolve_decomp.py, slated for deletion; they move to instance_sequence / program_loaders after those land.
from verity_vllm.check.match.instance_sequence import const_value  # noqa: F401
from verity_vllm.check.match.program_loaders import LOAD_NOTES  # noqa: F401

_T_RE = re.compile(r"\{T=\d+,")


def _norm_T(spec: str) -> str:
    return _T_RE.sub("{T=*,", spec)


# The divergence REPORT of a step whose spec sequences differ.  difflib.SequenceMatcher over a WHOLE step is
# O(n*m): a prefill step of an LP-row Program is LP x layers x ops instances, and a step differing throughout ran 40 min to 3 h per leg.
# No verdict depends on it (spec_seq_equal / spec_multiset_equal / canonical_sha256 / the DAG multiset carry the check); it only fills
# `first_divergences_T_normalised` and `n_divergent_opcodes`.  So the common prefix and suffix are located in O(n) and the matcher runs
# over the differing CORE only, at most `_DIVERGENCE_WINDOW` items of each side (the first divergences lie at its start); a longer core
# is reported truncated BY NAME (`divergence_window`).  An equal step never enters this path.  Cost ~ W^2 (W 1024: 0.8 s, 4096: 49 s).
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
        # the derived split count binding, when present on this side (fold: bound Const32[S] per select; Program: splits input)
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
            div = _divergence_report([_norm_T(x) for x in spa], [_norm_T(x) for x in spb])   # bounded, see _divergence_report
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
            ops = [_hash_operand(c) for c in cs] if compact else [list(map(list, c)) for c in cs]   # the record's BYTES (compact.hash_operand)
            h.update(json.dumps([r["spec"], r["form"], ops, r["out"]],
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
        P._pos_cache, P._pos_cache_key, P._pos_ident = pos, (unfold, id(seqs)), not unfold and seqs == P.steps
    compact = getattr(P, "compact", False)
    if compact and P._pos_ident:   # the remap is the identity: canon's normal-form progressions compare exactly as their runs
        return [tuple(arg) for arg in P.canon(i, False)]
    out = []
    for arg in P.canon(i, unfold):
        if compact:   # the record's per-run tuples: each run's producer is remapped on its own
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
    operands: str = option("--operands", default="runs", choices=("runs", "progressions"), help="both sides' operand form (progressions: compact)")


def main(a: Options) -> int:
    t0, cp = time.time(), a.operands == "progressions"
    if cp and a.unfold_a:
        raise UsageError("--operands progressions compares without unfolding only: an unfolded side's operands are runs")
    A = load_derived(a.a, a.label_a, compact=cp)
    B = load_oracle(a.b, a.label_b, compact=cp) if os.path.exists(os.path.join(a.b, "program.json")) else load_derived(a.b, a.label_b, compact=cp)
    res = {"schema": "cb-a/compare/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "a_dir": a.a, "b_dir": a.b, "unfold_a": a.unfold_a}
    if B.oracle_rows is not None:
        res["fold_form"] = CF.fold_in_form(B, [a.a])
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
