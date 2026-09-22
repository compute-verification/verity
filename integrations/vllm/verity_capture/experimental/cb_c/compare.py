"""Independent structural / unfolding comparator (CB-C, builder-validation owner).

A mechanically constructed Program (CB-A's step descriptor, CB-B's request descriptor, ...) against the accepted
oracle record (`program.json` + `instances.jsonl` of `B0-cov_b0_b1_c256`). Written from the descriptor codec and
the registry semantics only (verity.ir/codec.py, refs.py; registry/b1.py, serve3.py) -- not derived from either
bet's own comparator.

    python -m verity_capture.experimental.cb_c.compare --mine <descriptor.json[.gz] | program.json wrapper>
        --oracle <dir with program.json + instances.jsonl> --out <json> [--md <md>]
        [--mine-instances <cb-a instances.json.gz | instances.jsonl>] [--mine-step-regex 't=(\\d+)/']
        [--align mine_step:oracle_step,...] [--pad-regex REGEX] [--active-steps N]

What is compared, per aligned (mine step, oracle step) pair -- every difference is reported verbatim, then
classified as `representational` (a different but equivalent way to write the same circuit: input layout,
concatenation splitting, step-boundary Inputs standing for another step's values) or `semantic` (different
operator, static parameter, type, operand value, or dependency):

  1. operator/spec sequence     ordered list of callee specialization ids (`Name_vN{statics}`) of the step's nodes;
                                on mismatch, the family sequence (names only) and a per-position static diff.
  2. operand wiring             every operand reference run rewritten in producer-relative coordinates:
                                (`input`, width, offset, count) for root Input gates, (`node`, producer step,
                                index of the producer within its step, base, count[, stride]) for node outputs.
                                Node-to-node wiring must be identical. Input offsets are compared through a
                                mapping learnt from the matched positions; it must be a consistent bijection.
  3. types                      argument types and output type of every node (interned type tables de-interned).
  4. KV / state dependencies    operand runs whose producer lies in an earlier step: census per (consumer family,
                                producer family, lag) and per step; a step-program's Inputs that stand for
                                earlier-step values are listed as `state-boundary` differences.
  5. active vs padded region    nodes matched by `--pad-regex` / padding families (PadGuard, PadDecodeStep,
                                Select*, BitAnd, BitNot, I32Eq) are the padding scaffold; steps beyond
                                `--active-steps` are the inactive suffix; both are reported separately.
  6. expanded-circuit equality  for a region where 1-4 hold with a consistent bijection, the expanded gate
                                circuits are equal iff the *definitions* of every callee reachable from the
                                region's specs are equal on both sides (bodies compared node by node after
                                de-interning). Both facts are checked and the conclusion is stated; gate counts
                                per spec are computed from the definitions as a cross-check.

Nothing is normalised silently: the only rewrites are the de-interning of the v1 tables, the resolution of
`alias` parts to the referenced sequence (the codec's own decode rule) and the merge of adjacent contiguous
affine runs on one producer (declared in the output under `declared_normalisations`).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Any

SCHEMA = "verity-cbc/compare/v1"
PAD_FAMILIES = {"PadGuard", "PadDecodeStep", "SelectBf16", "SelectF32", "SelectI32", "BitAnd", "BitNot", "I32Eq"}


# ---------------------------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------------------------

def _open(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def load_descriptor(path: str) -> tuple[dict, dict]:
    with _open(path) as f:
        d = json.load(f)
    if "program" in d and "definitions" not in d:
        return d["program"], {k: v for k, v in d.items() if k != "program"}
    return d, {}


class Desc:
    """A descriptor with the v1 tables de-interned and the root body's references resolved."""

    def __init__(self, desc: dict, label: str) -> None:
        self.label = label
        self.desc = desc
        self.schema = desc.get("schema")
        self.v1 = self.schema == "verity-ir/descriptor/v1"
        self.callees = desc.get("callees", [])
        self.types = desc.get("types", [])
        self.defs = desc["definitions"]
        self.root_id = desc["root"]
        self.program_fn = desc.get("program_fn")
        body = self.defs[self.root_id]["body"]
        self.params = body["params"]
        self.ret_type = self.type_str(body["ret_type"])
        self.nodes_json = body["nodes"]
        self.ret_json = body["ret"]
        # resolved nodes
        self.nodes: list[dict] = []
        self.input_nodes: list[int] = []
        for k, nd in enumerate(self.nodes_json):
            fn = self.callee(nd["fn"])
            args = []
            for a in nd["args"]:
                args.append({"type": self.type_str(a["type"]), "runs": self.runs(a["refs"], k)})
            node = {"k": k, "fn": fn, "family": spec_family(fn), "form": nd["form"], "n": nd.get("n", 1), "axes": nd.get("axes", []),
                    "out": self.type_str(nd["out"]), "args": args}
            if not nd["args"] and spec_family(fn).startswith("Input"):
                node["input"] = True
                self.input_nodes.append(k)
            self.nodes.append(node)
        self.ret_runs = self.runs(self.ret_json, len(self.nodes))

    def callee(self, f) -> str:
        return self.callees[f] if isinstance(f, int) else f

    def type_str(self, t) -> str:
        tj = self.types[t] if isinstance(t, int) else t
        return json.dumps(tj, separators=(",", ":"))

    def runs(self, j, k: int) -> list[tuple]:
        """Reference JSON -> list of runs ("A", node, base, count) | ("S", node, base, inner, ostride, istride, count) | ("E", node, leaf)."""
        if "alias" in j:
            kk, a = j["alias"]
            if kk >= k:
                raise ValueError(f"{self.label}: alias to node {kk} at/after node {k}")
            return list(self.nodes[kk]["args"][a]["runs"])
        if "concat" in j:
            out: list[tuple] = []
            for x in j["concat"]:
                out.extend(self.runs(x, k))
            return _merge_affine(out)
        if "affine" in j:
            sp, i, b, c = j["affine"]
            assert sp == "n", sp
            return [("A", i, b, c)] if c else []
        if "strided" in j:
            sp, i, b, inner, os_, is_, c = j["strided"]
            assert sp == "n", sp
            if is_ == 1 and os_ == inner:
                return [("A", i, b, c)]
            return [("S", i, b, inner, os_, is_, c)]
        if "explicit" in j:
            return _merge_affine([("A", r[1], r[2], 1) for r in j["explicit"] if r[0] == "n"])
        raise ValueError(f"unknown refs {j}")

    def definition_canon(self, fid: str, seen: dict[str, Any]) -> Any:
        """De-interned definition JSON of `fid`, callees replaced by their ids (recursively canonicalised into `seen`)."""
        if fid in seen:
            return seen[fid]
        d = self.defs[fid]
        if d["form"] == "primitive":
            c = {"form": "primitive", "id": d["id"], "params": d["params"], "ret": d["ret"]}
            seen[fid] = c
            return c
        b = d["body"]
        nodes = []
        for nd in b["nodes"]:
            cf = self.callee(nd["fn"])
            nodes.append({"fn": cf, "form": nd["form"], "n": nd.get("n", 1), "axes": nd.get("axes", []), "n_xs": nd.get("n_xs", 0),
                          "out": self.type_str(nd["out"]),
                          "args": [{"type": self.type_str(a["type"]), "refs": _refs_canon(a["refs"])} for a in nd["args"]]})
            self.definition_canon(cf, seen)
        c = {"form": "composite", "id": d["id"], "bindings": _bindings_canon(d["bindings"]), "params": b["params"],
             "ret_type": self.type_str(b["ret_type"]), "nodes": nodes, "ret": _refs_canon(b["ret"])}
        seen[fid] = c
        return c


def _refs_canon(j) -> Any:
    # alias parts inside definitions refer to nodes of the same body; keep them literal (both writers use the codec).
    return j


def _bindings_canon(b: dict) -> dict:
    out = {}
    for k, v in b.items():
        out[k] = v
    return out


def _merge_affine(runs: list[tuple]) -> list[tuple]:
    out: list[tuple] = []
    for r in runs:
        if out and r[0] == "A" and out[-1][0] == "A" and out[-1][1] == r[1] and out[-1][2] + out[-1][3] == r[2]:
            out[-1] = ("A", r[1], out[-1][2], out[-1][3] + r[3])
        else:
            out.append(r)
    return out


_SPEC_RE = re.compile(r"^([^{]*?)_v(\d+)(\{.*\})?$")


def spec_family(fid: str) -> str:
    m = _SPEC_RE.match(fid)
    return m.group(1) if m else fid


def spec_parse(fid: str) -> tuple[str, str, dict]:
    m = _SPEC_RE.match(fid)
    if not m:
        return fid, "", {}
    name, ver, st = m.group(1), m.group(2), m.group(3) or ""
    statics: dict[str, str] = {}
    if st:
        body = st[1:-1]
        depth = 0
        cur = ""
        parts = []
        for ch in body:
            if ch in "{[(":
                depth += 1
            elif ch in "}])":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(cur)
                cur = ""
            else:
                cur += ch
        if cur:
            parts.append(cur)
        for p in parts:
            if "=" in p:
                kk, vv = p.split("=", 1)
                statics[kk.strip()] = vv.strip()
    return name, ver, statics


# ---------------------------------------------------------------------------------------------------------------
# step segmentation
# ---------------------------------------------------------------------------------------------------------------

def load_oracle_steps(path: str, n_input_nodes: int, n_nodes: int) -> tuple[dict[int, int], dict[int, str], dict]:
    """instances.jsonl -> {root node k: step}, {k: name}; root node of instance j = n_input_nodes + j."""
    step_of: dict[int, int] = {}
    name_of: dict[int, str] = {}
    header: dict = {}
    with _open(path) as f:
        for line in f:
            r = json.loads(line)
            if "schema" in r and "node" not in r:
                header = r
                continue
            k = n_input_nodes + int(r["node"])
            step_of[k] = int(r["step"])
            name_of[k] = r.get("name", "")
    for k in range(n_input_nodes):
        step_of[k] = -1
    return step_of, name_of, header


def load_mine_steps(D: Desc, inst_path: str | None, step_regex: str | None, prompt_len: int | None = None) -> tuple[dict[int, int], dict[int, str], str]:
    step_of: dict[int, int] = {k: -1 for k in D.input_nodes}
    name_of: dict[int, str] = {}
    how = ""
    if inst_path:
        with _open(inst_path) as f:
            text = f.read()
        if inst_path.endswith(".jsonl"):
            n_in = len(D.input_nodes)
            rx0 = re.compile(step_regex) if step_regex else None
            for line in text.splitlines():
                r = json.loads(line)
                if "node" in r and "step" in r:
                    k = n_in + int(r["node"])
                    step_of[k] = int(r["step"])
                    name_of[k] = r.get("name", "")
                elif "k" in r:  # cb-b body.jsonl: body index + name `t=<position>/...`; step from the position and the prompt length
                    k = n_in + int(r["k"])
                    name_of[k] = r.get("name", "")
                    m = rx0.search(name_of[k]) if rx0 and name_of[k] else None
                    if m and prompt_len is not None:
                        t = int(m.group(1))
                        step_of[k] = 0 if t < prompt_len else t - prompt_len + 1
                    elif m:
                        step_of[k] = int(m.group(1))
            how = f"instances/body jsonl ({inst_path}; step = 0 if position < prompt_len {prompt_len} else position - prompt_len + 1)"
        else:
            j = json.loads(text)
            rows = j.get("rows", j.get("instances", j)) if isinstance(j, dict) else j
            # cb-a/instance-sequence/v1: rows in root-node order after the Input nodes
            n_in = len(D.input_nodes)
            for i, r in enumerate(rows):
                k = n_in + (int(r["node"]) if isinstance(r, dict) and "node" in r else i)
                step_of[k] = int(r.get("step", 0)) if isinstance(r, dict) else 0
                name_of[k] = (r.get("name") or r.get("label") or "") if isinstance(r, dict) else ""
            how = f"instance sequence ({inst_path}, schema {j.get('schema') if isinstance(j, dict) else '?'})"
    if step_regex == "tokenselect":  # structural rule: a sampling event (TokenSelect) closes a step; usable when names carry no position
        n_ts = 0
        for k, nd in enumerate(D.nodes):
            if k in step_of:
                continue
            step_of[k] = -1 if nd.get("input") else n_ts
            if nd["family"] == "TokenSelect":
                n_ts += 1
        return step_of, name_of, "structural: step = number of TokenSelect nodes before the node (a sampling event closes a step)"
    rx = re.compile(step_regex) if step_regex else None
    for k, nd in enumerate(D.nodes):
        if k in step_of:
            continue
        s = 0
        if rx and name_of.get(k):
            m = rx.search(name_of[k])
            if m:
                t = int(m.group(1))
                s = (0 if t < prompt_len else t - prompt_len + 1) if prompt_len is not None else t
        step_of[k] = s
    if not how:
        how = "single step (all non-Input nodes are step 0)" if not rx else f"regex {step_regex} on node names"
    return step_of, name_of, how


# ---------------------------------------------------------------------------------------------------------------
# per-step canonical form
# ---------------------------------------------------------------------------------------------------------------

class Side:
    def __init__(self, D: Desc, step_of: dict[int, int], name_of: dict[int, str]) -> None:
        self.D = D
        self.step_of = step_of
        self.name_of = name_of
        self.steps: dict[int, list[int]] = defaultdict(list)
        for k in range(len(D.nodes)):
            if D.nodes[k].get("removed"):
                continue
            self.steps[step_of.get(k, 0)].append(k)
        self.index_in_step: dict[int, int] = {}
        for s, ks in self.steps.items():
            for i, k in enumerate(ks):
                self.index_in_step[k] = i
        self.input_width: dict[int, int] = {}
        self.input_count: dict[int, int] = {}
        for k in D.input_nodes:
            nd = D.nodes[k]
            m = re.match(r"Input(\d+)", nd["family"])
            self.input_width[k] = int(m.group(1)) if m else -1
            self.input_count[k] = nd["n"]

    def canon_run(self, r: tuple, consumer_step: int) -> tuple:
        """Producer-relative coordinates. Input runs keep (width, input-node ordinal, base) -- mapped later."""
        i = r[1]
        if i in self.input_width:
            ordinal = self.D.input_nodes.index(i)
            if r[0] == "A":
                return ("input", self.input_width[i], ordinal, r[2], r[3])
            return ("input-strided", self.input_width[i], ordinal) + tuple(r[2:])
        ps = self.step_of.get(i, 0)
        rel = ("same" if ps == consumer_step else ("earlier", consumer_step - ps))
        if r[0] == "A":
            return ("node", rel, self.index_in_step[i], self.D.nodes[i]["family"], r[2], r[3])
        return ("node-strided", rel, self.index_in_step[i], self.D.nodes[i]["family"]) + tuple(r[2:])


def _leaves(runs: list[tuple]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r in runs:
        if r[0] == "A":
            out.extend((r[1], r[2] + i) for i in range(r[3]))
        elif r[0] == "S":
            _, i, b, inner, os_, is_, c = r
            for o in range(c // inner if inner else 0):
                out.extend((i, b + o * os_ + j * is_) for j in range(inner))
        else:
            out.append((r[1], r[2]))
    return out


def verify_and_project_padding(D: Desc, step_of: dict[int, int], name_of: dict[int, str]) -> dict:
    """serve3 padding check + projection of the scaffold (PadGuard / live-keyed SelectBf16 muxes / their constants) onto the
    flat computation it wraps. Every scaffold node is checked against `verity_vllm/registry/serve3.py`:
      * `PadGuard_v1{EOS}(live, y)`: `live` is `Const1[0x1]` (the first event) or the previous guard's `live_out` (slot 1);
        `y` is a `TokenSelect` output of the same step (the RAW sample -- the guard must not read a muxed token);
      * `batch(SelectBf16_v1, live, row, zero)`: `live` is the guard `live_out` of the *previous* sampling event (the step's
        liveness), `row` a computed value of this step, `zero` the `Const16[0x0000]` row;
      * the only consumers of a raw `TokenSelect` are its guard; the next step's Embedding reads the guard's `tok_out`.
    The projection rewires every consumer of a guard's `tok_out` to the raw sample and of a mux output to its computed row, then
    removes the scaffold. This is the serve3-prescribed *representational* difference between a padded and a flat Program; the
    verification results are reported so the projection is never a silent normalisation."""
    fam = lambda k: D.nodes[k]["family"]  # noqa: E731
    fn = lambda k: D.nodes[k]["fn"]  # noqa: E731
    muxes = [k for k in range(len(D.nodes)) if fam(k) == "SelectBf16" and D.nodes[k]["form"] == "batch"]
    consts_live = [k for k in range(len(D.nodes)) if fn(k).startswith("Const1[0x1]")]
    consts_zero = [k for k in range(len(D.nodes)) if fn(k).startswith("Const16[0x0000]")]
    problems: list[str] = []
    alias: dict[tuple[int, int], tuple[int, int]] = {}  # (scaffold node, leaf index) -> (flat node, leaf index)
    guard_step: dict[int, int] = {}
    live_chain: list[dict] = []
    guard_forms = Counter()
    single = lambda k, a: (lambda L: L[0] if len(L) == 1 else None)(_leaves(D.nodes[k]["args"][a]["runs"]))  # noqa: E731
    # guard records: {"key": node that carries live_out, "live_out": (node, slot), "tok_out": (node, slot), "live_in": leaf,
    #                 "y": leaf, "eos": spec, "nodes": [...]} -- from the registered composite or from its 6 inlined gates
    recs: list[dict] = []
    for g in range(len(D.nodes)):
        if fam(g) == "PadGuard":
            li, yi = single(g, 0), single(g, 1)
            if li is None or yi is None:
                problems.append(f"guard {g}: live/y are not single values")
                continue
            recs.append({"key": g, "live_out": (g, 1), "tok_out": (g, 0), "live_in": li, "y": yi, "eos": fn(g), "nodes": [g], "form": "PadGuard_v1"})
            guard_forms["PadGuard_v1 composite"] += 1
    # inlined: live_out = BitAnd(live_in, BitNot(I32Eq(y, Const32[EOS]))); tok_out = SelectI32(live_in, y, Const32[0])
    sel_by_live_y: dict[tuple, list[int]] = defaultdict(list)
    for k in range(len(D.nodes)):
        if fam(k) == "SelectI32" and len(D.nodes[k]["args"]) == 3:
            sel_by_live_y[(single(k, 0), single(k, 1))].append(k)
    for b in range(len(D.nodes)):
        if fam(b) != "BitAnd" or len(D.nodes[b]["args"]) != 2:
            continue
        li, ns = single(b, 0), single(b, 1)
        if ns is None or fam(ns[0]) != "BitNot":
            problems.append(f"BitAnd {b}: second operand is not BitNot(stop)")
            continue
        st = single(ns[0], 0)
        if st is None or fam(st[0]) != "I32Eq":
            problems.append(f"BitAnd {b}: BitNot operand is not I32Eq(y, EOS)")
            continue
        ya, ca = single(st[0], 0), single(st[0], 1)
        if ya is not None and fam(ya[0]).startswith("Const32"):
            ya, ca = ca, ya
        if ya is None or ca is None or not fam(ca[0]).startswith("Const32"):
            problems.append(f"I32Eq {st[0]}: operands are not (y, Const32[EOS])")
            continue
        sels = sel_by_live_y.get((li, ya), [])
        if len(sels) != 1:
            problems.append(f"BitAnd {b}: expected exactly one SelectI32(live, y, Const32[0]) with the same live/y, found {len(sels)}")
            continue
        sel = sels[0]
        pad = single(sel, 2)
        if pad is None or not fn(pad[0]).startswith("Const32[0x00000000]"):
            problems.append(f"SelectI32 {sel}: pad operand is {fn(pad[0]) if pad else None}, not Const32[0x00000000]")
        recs.append({"key": b, "live_out": (b, 0), "tok_out": (sel, 0), "live_in": li, "y": ya, "eos": f"inlined guard, stop = I32Eq(y, {fn(ca[0])})",
                     "nodes": [b, ns[0], st[0], sel, ca[0]] + ([pad[0]] if pad else []), "form": "inlined"})
        guard_forms["inlined 6-gate guard (I32Eq, BitNot, BitAnd, SelectI32, 2 consts)"] += 1
    recs.sort(key=lambda r: r["y"][0])  # sampling-event order
    live_out_of: dict[tuple[int, int], int] = {}
    prev: dict | None = None
    guards = []
    for r in recs:
        g = r["key"]
        guards.append(g)
        (ln, li_), (yn, yi_) = r["live_in"], r["y"]
        if prev is None:
            if ln not in consts_live:
                problems.append(f"guard {g}: first live input is {fam(ln)} not Const1[0x1]")
        elif (ln, li_) != prev["live_out"]:
            problems.append(f"guard {g}: live input is ({fam(ln)} {ln}, {li_}) not the previous guard's live_out {prev['live_out']}")
        if fam(yn) != "TokenSelect":
            problems.append(f"guard {g}: y is {fam(yn)} not the raw TokenSelect sample")
        # scaffold nodes are named `s=<event>/guard` (or unnamed): their step is their sample's step
        st_ = step_of.get(yn, 0)
        for k in r["nodes"]:
            step_of[k] = st_
        guard_step[g] = st_
        alias[r["tok_out"]] = (yn, yi_)  # tok_out -> raw sample
        live_out_of[r["live_out"]] = g
        live_chain.append({"guard": g, "form": r["form"], "step": st_, "live_in": (fam(ln), ln, li_), "y": (fam(yn), yn, yi_), "eos": r["eos"]})
        prev = r
    scaffold_extra = {k for r in recs for k in r["nodes"]}
    guard_of_step = {s: g for g, s in guard_step.items()}
    mux_stats = Counter()
    for m in sorted(muxes):
        nd = D.nodes[m]
        if len(nd["args"]) != 3:
            problems.append(f"mux {m}: {len(nd['args'])} args")
            continue
        ll, rl, zl = (_leaves(nd["args"][i]["runs"]) for i in range(3))
        n = nd["n"]
        s = step_of.get(rl[0][0], 0) if rl else step_of.get(m, 0)  # the mux's step is its computed row's step
        step_of[m] = s
        for z in zl:
            step_of[z[0]] = s
        if len(ll) != 1 or ll[0] not in live_out_of:
            problems.append(f"mux {m} (step {s}): select bit is not a guard live_out: {[(fam(a), a, b) for a, b in ll][:2]}")
        elif guard_step.get(live_out_of[ll[0]]) != s - 1:
            problems.append(f"mux {m} (step {s}): keyed on guard of step {guard_step.get(live_out_of[ll[0]])}, expected the previous sampling event (step {s - 1})")
        if len(rl) != n or len(zl) != n:
            problems.append(f"mux {m}: row/zero widths {len(rl)}/{len(zl)} vs n={n}")
            continue
        if any(z[0] not in consts_zero for z in zl):
            problems.append(f"mux {m}: zero operand is not Const16[0x0000]")
        if any(step_of.get(r[0], 0) != s for r in rl):
            problems.append(f"mux {m}: computed row not from step {s}")
        for i, (rn, ri) in enumerate(rl):
            alias[(m, i)] = (rn, ri)
        mux_stats[fam(rl[0][0])] += 1
    # raw-sample consumers: only its guard (and nothing else) may read a TokenSelect output
    consumers: dict[int, list[int]] = defaultdict(list)
    for k, nd in enumerate(D.nodes):
        for a in nd["args"]:
            for r in a["runs"]:
                if fam(r[1]) == "TokenSelect":
                    consumers[r[1]].append(k)
    raw_leaks = {ts: [(fam(c), c) for c in cs if c not in scaffold_extra] for ts, cs in consumers.items()}
    raw_leaks = {ts: v for ts, v in raw_leaks.items() if v}
    if raw_leaks:
        problems.append(f"raw TokenSelect outputs read by non-guard nodes: {list(raw_leaks.items())[:5]}")
    ret_leaves = _leaves(D.ret_runs)
    ret_from_guards = all(n in scaffold_extra for n, _ in ret_leaves)
    # projection
    scaffold = scaffold_extra | set(muxes) | set(consts_live) | set(consts_zero)
    live_slots = set(live_out_of)

    def project(runs: list[tuple]) -> list[tuple]:
        if not any(r[1] in scaffold for r in runs):
            return runs
        out: list[tuple] = []
        for r in runs:
            if r[1] not in scaffold:
                out.append(r)
                continue
            for (n, i) in _leaves([r]):
                seen = 0
                while n in scaffold and (n, i) in alias and seen < 8:
                    n, i = alias[(n, i)]
                    seen += 1
                if n in scaffold:
                    problems.append(f"leaf ({fam(n)} {n}, {i}) has no projection (live bits / constants are not flat values)")
                out.append(("A", n, i, 1))
        return _merge_affine(out)

    for k, nd in enumerate(D.nodes):
        if k in scaffold:
            continue
        for a in nd["args"]:
            a["runs"] = project(a["runs"])
    tok_leaves = [(n, i) for n, i in ret_leaves if (n, i) not in live_slots]
    D.ret_runs = project(_merge_affine([("A", n, i, 1) for n, i in tok_leaves]))
    D.ret_type_projected = f"tok-slots-only from {D.ret_type}"
    for k in scaffold:
        D.nodes[k]["removed"] = True
    return {"guards": len(guards), "muxes": len(muxes), "mux_members": sum(D.nodes[m]["n"] for m in muxes), "mux_row_sources": dict(mux_stats),
            "live_consts": len(consts_live), "zero_consts": len(consts_zero),
            "live_chain": live_chain, "guards_per_step": {str(s): g for s, g in sorted(guard_of_step.items())},
            "ret_reads_guards_only": ret_from_guards, "raw_sample_leaks": {str(k): v for k, v in raw_leaks.items()},
            "eos_statics": sorted({r["eos"] for r in recs}), "guard_forms": dict(guard_forms),
            "problems": problems, "ok": not problems, "scaffold_nodes_removed": len(scaffold)}


def compare_steps(M: Side, O: Side, ms: int, os_: int, input_map: dict, R: dict) -> dict:
    """Compare mine step `ms` with oracle step `os_`. `input_map` accumulates (mine input width, ordinal, base, count)
    -> oracle (width, ordinal, base, count) correspondences across all steps (must be a consistent bijection)."""
    mk = M.steps.get(ms, [])
    ok = O.steps.get(os_, [])
    out: dict[str, Any] = {"mine_step": ms, "oracle_step": os_, "mine_nodes": len(mk), "oracle_nodes": len(ok)}
    m_specs = [M.D.nodes[k]["fn"] for k in mk]
    o_specs = [O.D.nodes[k]["fn"] for k in ok]
    out["spec_sequence_equal"] = m_specs == o_specs
    m_fam = [spec_family(s) for s in m_specs]
    o_fam = [spec_family(s) for s in o_specs]
    out["family_sequence_equal"] = m_fam == o_fam
    out["spec_histogram"] = {"mine": dict(Counter(m_specs)), "oracle": dict(Counter(o_specs))}
    hist_diff = {}
    for s in set(m_specs) | set(o_specs):
        a, b = m_specs.count(s), o_specs.count(s)
        if a != b:
            hist_diff[s] = {"mine": a, "oracle": b}
    out["spec_histogram_diff"] = hist_diff
    if not out["spec_sequence_equal"]:
        first = next((i for i in range(min(len(m_specs), len(o_specs))) if m_specs[i] != o_specs[i]), min(len(m_specs), len(o_specs)))
        out["first_spec_divergence"] = {"position": first,
                                        "mine": m_specs[first] if first < len(m_specs) else None,
                                        "oracle": o_specs[first] if first < len(o_specs) else None,
                                        "mine_name": M.name_of.get(mk[first], "") if first < len(mk) else None,
                                        "oracle_name": O.name_of.get(ok[first], "") if first < len(ok) else None}
        if out["family_sequence_equal"]:
            diffs = Counter()
            examples = []
            for i, (a, b) in enumerate(zip(m_specs, o_specs)):
                if a != b:
                    na, va, sa = spec_parse(a)
                    nb, vb, sb = spec_parse(b)
                    keyd = tuple(sorted((kk, sa.get(kk), sb.get(kk)) for kk in set(sa) | set(sb) if sa.get(kk) != sb.get(kk)))
                    diffs[(na, va, vb, keyd)] += 1
                    if len(examples) < 8:
                        examples.append({"position": i, "mine": a, "oracle": b})
            out["static_parameter_diffs"] = [{"family": k[0], "mine_version": k[1], "oracle_version": k[2],
                                              "statics": [{"name": d[0], "mine": d[1], "oracle": d[2]} for d in k[3]], "count": v}
                                             for k, v in diffs.items()]
            out["static_parameter_examples"] = examples
    # wiring / types on the matched prefix (same family at the same position)
    n = min(len(mk), len(ok))
    wiring_mismatch: list[dict] = []
    type_mismatch: list[dict] = []
    state_boundary: list[dict] = []
    input_conflicts: list[dict] = []
    matched = 0
    cross_step_m = Counter()
    cross_step_o = Counter()
    for i in range(n):
        a, b = M.D.nodes[mk[i]], O.D.nodes[ok[i]]
        if a["family"] != b["family"]:
            break
        matched += 1
        if a["out"] != b["out"] or [x["type"] for x in a["args"]] != [x["type"] for x in b["args"]]:
            if len(type_mismatch) < 20:
                type_mismatch.append({"position": i, "mine": {"out": a["out"], "args": [x["type"] for x in a["args"]]},
                                      "oracle": {"out": b["out"], "args": [x["type"] for x in b["args"]]}})
            else:
                type_mismatch.append({"position": i})
        if len(a["args"]) != len(b["args"]):
            wiring_mismatch.append({"position": i, "spec": a["fn"], "reason": f"arity {len(a['args'])} vs {len(b['args'])}"})
            continue
        for j, (aa, bb) in enumerate(zip(a["args"], b["args"])):
            ra = [M.canon_run(r, ms) for r in aa["runs"]]
            rb = [O.canon_run(r, os_) for r in bb["runs"]]
            for r in ra:
                if r[0].startswith("node") and r[1] != "same":
                    cross_step_m[(a["family"], r[3], r[1][1])] += 1
            for r in rb:
                if r[0].startswith("node") and r[1] != "same":
                    cross_step_o[(b["family"], r[3], r[1][1])] += 1
            if len(ra) != len(rb):
                # a step-program Input standing for an earlier step's node output (or vice versa) is the common cause
                kinds_a = Counter(r[0] for r in ra)
                kinds_b = Counter(r[0] for r in rb)
                if len(state_boundary) < 40:
                    state_boundary.append({"position": i, "spec": a["fn"], "arg": j, "mine_runs": len(ra), "oracle_runs": len(rb),
                                           "mine_kinds": dict(kinds_a), "oracle_kinds": dict(kinds_b),
                                           "mine_first": [list(map(str, r)) for r in ra[:3]], "oracle_first": [list(map(str, r)) for r in rb[:3]]})
                else:
                    state_boundary.append({"position": i, "arg": j})
                continue
            for x, y in zip(ra, rb):
                if x[0] == "input" and y[0] == "input":
                    key = (x[1], x[2], x[3], x[4])
                    val = (y[1], y[2], y[3], y[4])
                    prev = input_map.get(key)
                    if prev is None:
                        input_map[key] = val
                    elif prev != val:
                        if len(input_conflicts) < 20:
                            input_conflicts.append({"position": i, "spec": a["fn"], "arg": j, "mine": list(key), "oracle": list(val), "earlier": list(prev)})
                        else:
                            input_conflicts.append({"position": i})
                    if x[1] != y[1] or x[4] != y[4]:
                        wiring_mismatch.append({"position": i, "spec": a["fn"], "arg": j, "kind": "input width/count", "mine": list(map(str, x)), "oracle": list(map(str, y))})
                elif x[0].startswith("node") and y[0].startswith("node"):
                    if x != y:
                        if x[1] != y[1] and x[1] != "same" and y[1] != "same":
                            kind = "cross-step lag differs"
                        elif x[1] != y[1]:
                            kind = "same-step vs earlier-step producer"
                        else:
                            kind = "node wiring"
                        if len(wiring_mismatch) < 40:
                            wiring_mismatch.append({"position": i, "spec": a["fn"], "arg": j, "kind": kind, "mine": list(map(str, x)), "oracle": list(map(str, y))})
                        else:
                            wiring_mismatch.append({"position": i, "arg": j})
                else:
                    if len(state_boundary) < 40:
                        state_boundary.append({"position": i, "spec": a["fn"], "arg": j, "mine": list(map(str, x)), "oracle": list(map(str, y)),
                                               "kind": "Input on one side, node output on the other"})
                    else:
                        state_boundary.append({"position": i, "arg": j})
    out["matched_prefix"] = matched
    out["wiring_mismatches"] = len(wiring_mismatch)
    out["wiring_mismatch_examples"] = wiring_mismatch[:40]
    out["type_mismatches"] = len(type_mismatch)
    out["type_mismatch_examples"] = type_mismatch[:20]
    out["state_boundary_differences"] = len(state_boundary)
    out["state_boundary_examples"] = state_boundary[:40]
    out["input_map_conflicts"] = len(input_conflicts)
    out["input_map_conflict_examples"] = input_conflicts[:20]
    out["cross_step_refs"] = {"mine": [{"consumer": k[0], "producer": k[1], "lag": k[2], "runs": v} for k, v in sorted(cross_step_m.items())],
                              "oracle": [{"consumer": k[0], "producer": k[1], "lag": k[2], "runs": v} for k, v in sorted(cross_step_o.items())]}
    out["cross_step_refs_equal"] = cross_step_m == cross_step_o
    out["verdict"] = ("EXACT" if out["spec_sequence_equal"] and not wiring_mismatch and not type_mismatch and not state_boundary and not input_conflicts
                      else "STRUCTURE-EQUAL-UP-TO-STATE-BOUNDARY" if out["spec_sequence_equal"] and not wiring_mismatch and not type_mismatch and not input_conflicts
                      else "FAMILY-EQUAL-STATICS-DIFFER" if out["family_sequence_equal"] and not out["spec_sequence_equal"]
                      else "MISMATCH")
    return out


def definitions_equal(M: Desc, O: Desc, specs: set[str]) -> dict:
    """Every definition reachable from `specs` on both sides, canonicalised and compared."""
    sm: dict[str, Any] = {}
    so: dict[str, Any] = {}
    missing_m, missing_o, differ = [], [], []
    for s in sorted(specs):
        if s not in M.defs:
            missing_m.append(s)
            continue
        if s not in O.defs:
            missing_o.append(s)
            continue
        M.definition_canon(s, sm)
        O.definition_canon(s, so)
    for fid in sorted(set(sm) & set(so)):
        if json.dumps(sm[fid], sort_keys=True) != json.dumps(so[fid], sort_keys=True):
            differ.append(fid)
    return {"n_reachable_mine": len(sm), "n_reachable_oracle": len(so), "missing_in_mine": missing_m, "missing_in_oracle": missing_o,
            "differing": differ[:50], "n_differing": len(differ), "equal": not (missing_m or missing_o or differ)}


def gate_counts(D: Desc, specs: set[str]) -> dict[str, int]:
    """Gates of one call of each spec, from the definitions (primitive = 1 gate; batch = n * gates(fn); scan = n * gates)."""
    memo: dict[str, int] = {}

    def g(fid: str) -> int:
        if fid in memo:
            return memo[fid]
        d = D.defs[fid]
        if d["form"] == "primitive":
            memo[fid] = 1
            return 1
        total = 0
        for nd in d["body"]["nodes"]:
            cf = D.callee(nd["fn"])
            total += nd.get("n", 1) * g(cf)
        memo[fid] = total
        return total

    return {s: g(s) for s in sorted(specs) if s in D.defs}


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------

def parse_align(s: str | None, M: Side, O: Side) -> list[tuple[int, int]]:
    if s:
        pairs = []
        for p in s.split(","):
            a, b = p.split(":")
            pairs.append((int(a), int(b)))
        return pairs
    ms = sorted(k for k in M.steps if k >= 0)
    os_ = sorted(k for k in O.steps if k >= 0)
    return list(zip(ms, os_))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mine", required=True)
    ap.add_argument("--mine-instances", default=None)
    ap.add_argument("--mine-step-regex", default=r"t=(\d+)/")
    ap.add_argument("--mine-label", default="mine")
    ap.add_argument("--mine-prompt-len", type=int, default=None, help="prompt length: node names carry positions t=<pos>; step = 0 for pos < LP else pos-LP+1")
    ap.add_argument("--oracle", required=True, help="dir with program.json + instances.jsonl (or a constructed Program dir with body.jsonl)")
    ap.add_argument("--oracle-instances", default=None)
    ap.add_argument("--oracle-prompt-len", type=int, default=None)
    ap.add_argument("--align", default=None, help="mine_step:oracle_step,... (default: positional)")
    ap.add_argument("--pad-regex", default=r"(^|/)(pad|inactive|dead)")
    ap.add_argument("--active-steps", type=int, default=None, help="mine steps >= this are the inactive suffix")
    ap.add_argument("--project-padding", action="store_true",
                    help="verify the serve3 scaffold (PadGuard / live-keyed SelectBf16 / constants) and project it out before comparing (reported, never silent)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    R: dict[str, Any] = {"schema": SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": sys.argv,
                         "mine": a.mine, "oracle": a.oracle, "mine_label": a.mine_label,
                         "declared_normalisations": ["v1 callee/type tables de-interned", "alias parts resolved to the referenced sequence (codec decode rule)",
                                                     "adjacent contiguous affine runs on one producer merged", "strided runs with inner_stride 1 and outer_stride == inner rewritten as affine"]}
    m_desc, m_wrap = load_descriptor(a.mine)
    o_desc, o_wrap = load_descriptor(f"{a.oracle}/program.json")
    M = Desc(m_desc, "mine")
    O = Desc(o_desc, "oracle")
    R["oracle_wrapper"] = o_wrap
    R["mine_wrapper"] = m_wrap
    R["roots"] = {"mine": M.root_id, "oracle": O.root_id, "mine_program_fn": M.program_fn, "oracle_program_fn": O.program_fn,
                  "mine_ret_type": M.ret_type, "oracle_ret_type": O.ret_type, "mine_params": M.params, "oracle_params": O.params}
    R["sizes"] = {"mine_nodes": len(M.nodes), "oracle_nodes": len(O.nodes), "mine_input_nodes": len(M.input_nodes), "oracle_input_nodes": len(O.input_nodes),
                  "mine_defs": len(M.defs), "oracle_defs": len(O.defs)}
    R["inputs"] = {"mine": [{"k": k, "family": M.nodes[k]["family"], "n": M.nodes[k]["n"], "type": M.nodes[k]["out"]} for k in M.input_nodes],
                   "oracle": [{"k": k, "family": O.nodes[k]["family"], "n": O.nodes[k]["n"], "type": O.nodes[k]["out"]} for k in O.input_nodes]}
    if os.path.exists(f"{a.oracle}/instances.jsonl"):
        o_step, o_name, o_hdr = load_oracle_steps(f"{a.oracle}/instances.jsonl", o_wrap.get("n_input_nodes", len(O.input_nodes)), len(O.nodes))
        o_how = "instances.jsonl"
    else:  # a constructed Program as the reference (body.jsonl names): same segmentation rule as --mine
        o_step, o_name, o_how = load_mine_steps(O, a.oracle_instances or f"{a.oracle}/body.jsonl", a.mine_step_regex, a.oracle_prompt_len)
        o_hdr = {}
    m_step, m_name, how = load_mine_steps(M, a.mine_instances, a.mine_step_regex, a.mine_prompt_len)
    R["segmentation"] = {"mine": how, "oracle": o_how, "oracle_header": o_hdr}
    if a.project_padding:
        R["padding_projection"] = verify_and_project_padding(M, m_step, m_name)
        R["declared_normalisations"].append("PADDING PROJECTION (--project-padding): PadGuard tok_out -> raw TokenSelect sample; live-keyed SelectBf16 mux -> its computed row; "
                                            "guards/muxes/Const1[0x1]/Const16[0x0000] removed; live-bit ret slots dropped. Scaffold verified against serve3.py first (see padding_projection)")
    Ms, Os = Side(M, m_step, m_name), Side(O, o_step, o_name)
    R["steps"] = {"mine": {str(s): len(ks) for s, ks in sorted(Ms.steps.items())}, "oracle": {str(s): len(ks) for s, ks in sorted(Os.steps.items())}}
    # padding scaffold
    pad_rx = re.compile(a.pad_regex)
    pad_nodes = [k for k, nd in enumerate(M.nodes) if nd["family"] in PAD_FAMILIES or (m_name.get(k) and pad_rx.search(m_name[k]))]
    R["padding"] = {"scaffold_nodes": len(pad_nodes), "scaffold_families": dict(Counter(M.nodes[k]["family"] for k in pad_nodes)),
                    "inactive_suffix_steps": [s for s in Ms.steps if a.active_steps is not None and s >= a.active_steps],
                    "inactive_suffix_nodes": sum(len(Ms.steps[s]) for s in Ms.steps if a.active_steps is not None and s >= a.active_steps)}
    pairs = parse_align(a.align, Ms, Os)
    if a.active_steps is not None:
        pairs = [(m, o) for m, o in pairs if m < a.active_steps]
    R["alignment"] = [{"mine": m, "oracle": o} for m, o in pairs]
    input_map: dict = {}
    per_step = []
    for m, o in pairs:
        per_step.append(compare_steps(Ms, Os, m, o, input_map, R))
    R["per_step"] = per_step
    # program return: producer-relative form on both sides (mine after projection, if any)
    last_m = max((m for m, _ in pairs), default=0)
    last_o = max((o for _, o in pairs), default=0)
    rm = [Ms.canon_run(r, last_m) for r in M.ret_runs]
    ro = [Os.canon_run(r, last_o) for r in O.ret_runs]
    R["ret"] = {"mine": rm, "oracle": ro, "equal": rm == ro, "mine_ret_type": getattr(M, "ret_type_projected", M.ret_type), "oracle_ret_type": O.ret_type}
    # input bijection check
    inv: dict = {}
    non_inj = []
    for k, v in input_map.items():
        if v in inv and inv[v] != k:
            non_inj.append({"oracle": list(v), "mine_a": list(inv[v]), "mine_b": list(k)})
        inv[v] = k
    R["input_bijection"] = {"pairs": len(input_map), "non_injective": len(non_inj), "examples": non_inj[:20],
                            "consistent": not non_inj and all(s["input_map_conflicts"] == 0 for s in per_step)}
    # definitions reachable from the compared region
    specs = set()
    for m, o in pairs:
        specs |= {M.nodes[k]["fn"] for k in Ms.steps.get(m, [])}
        specs |= {O.nodes[k]["fn"] for k in Os.steps.get(o, [])}
    specs = {s for s in specs if not spec_family(s).startswith("Input")}
    R["definitions"] = definitions_equal(M, O, specs)
    gm, go = gate_counts(M, {s for s in specs if s in M.defs}), gate_counts(O, {s for s in specs if s in O.defs})
    R["gates_per_spec"] = {"mine": gm, "oracle": go, "equal_on_shared": all(gm[s] == go[s] for s in set(gm) & set(go))}
    # region gate totals
    def region_gates(S: Side, G: dict[str, int], steps: list[int]) -> int:
        return sum(S.D.nodes[k]["n"] * G.get(S.D.nodes[k]["fn"], 0) for s in steps for k in S.steps.get(s, []) if not S.D.nodes[k].get("input"))
    R["region_gates"] = {"mine": region_gates(Ms, gm, [m for m, _ in pairs]), "oracle": region_gates(Os, go, [o for _, o in pairs])}
    all_exact = all(s["verdict"] == "EXACT" for s in per_step) and R["ret"]["equal"]
    all_struct = all(s["verdict"] in ("EXACT", "STRUCTURE-EQUAL-UP-TO-STATE-BOUNDARY") for s in per_step)
    R["expanded_circuit"] = {
        "claim": "the expanded gate circuits of the compared region are equal iff (a) node sequences, wiring and types are equal with a consistent input bijection and (b) every reachable definition is equal on both sides",
        "a_node_level": "EXACT" if all_exact else ("EXACT-UP-TO-STATE-BOUNDARY" if all_struct else "NOT-EQUAL"),
        "b_definitions_equal": R["definitions"]["equal"],
        "input_bijection_consistent": R["input_bijection"]["consistent"],
        "verdict": ("EQUAL" if all_exact and R["definitions"]["equal"] and R["input_bijection"]["consistent"]
                    else "EQUAL-MODULO-STATE-BOUNDARY-INPUTS" if all_struct and R["definitions"]["equal"] and R["input_bijection"]["consistent"]
                    else "NOT-EQUAL")}
    R["summary"] = {"verdicts": Counter(s["verdict"] for s in per_step), "seconds": round(time.time() - t0, 2)}
    R["summary"]["verdicts"] = dict(R["summary"]["verdicts"])
    with open(a.out, "w") as f:
        json.dump(R, f, indent=1, default=str)
    if a.md:
        with open(a.md, "w") as f:
            f.write(render_md(R))
    print(json.dumps({"verdicts": R["summary"]["verdicts"], "expanded": R["expanded_circuit"]["verdict"], "definitions_equal": R["definitions"]["equal"],
                      "input_bijection": R["input_bijection"]["consistent"], "seconds": R["summary"]["seconds"]}))
    return 0


def render_md(R: dict) -> str:
    L = [f"# compare: {R['mine_label']} vs oracle — {R['utc']}", "",
         f"- mine: `{R['mine']}` root `{R['roots']['mine']}` ({R['sizes']['mine_nodes']} root nodes, {R['sizes']['mine_input_nodes']} Input nodes)",
         f"- oracle: `{R['oracle']}` root `{R['roots']['oracle']}` ({R['sizes']['oracle_nodes']} root nodes, {R['sizes']['oracle_input_nodes']} Input nodes)",
         f"- ret types: mine `{R['roots']['mine_ret_type']}` / oracle `{R['roots']['oracle_ret_type']}`; ret wiring equal (after any projection): {R.get('ret', {}).get('equal')}",
         f"- segmentation: mine {R['segmentation']['mine']}; steps mine {R['steps']['mine']} / oracle {R['steps']['oracle']}",
         f"- padding scaffold nodes: {R['padding']['scaffold_nodes']} {R['padding']['scaffold_families']}; inactive suffix steps {R['padding']['inactive_suffix_steps']} ({R['padding']['inactive_suffix_nodes']} nodes)",
         (f"- padding projection (serve3 check): ok={R['padding_projection']['ok']} guards={R['padding_projection']['guards']} muxes={R['padding_projection']['muxes']} "
          f"(members {R['padding_projection']['mux_members']}, rows from {R['padding_projection']['mux_row_sources']}) eos={R['padding_projection']['eos_statics']} "
          f"ret_reads_guards_only={R['padding_projection']['ret_reads_guards_only']} raw_sample_leaks={len(R['padding_projection']['raw_sample_leaks'])} "
          f"problems={R['padding_projection']['problems'][:5]}") if R.get("padding_projection") else "- padding projection: off",
         f"- declared normalisations: {'; '.join(R['declared_normalisations'])}", "",
         "| mine step | oracle step | nodes m/o | spec seq | family seq | matched prefix | wiring ≠ | type ≠ | state-boundary | input conflicts | cross-step deps equal | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in R["per_step"]:
        L.append(f"| {s['mine_step']} | {s['oracle_step']} | {s['mine_nodes']}/{s['oracle_nodes']} | {s['spec_sequence_equal']} | {s['family_sequence_equal']} | {s['matched_prefix']} | "
                 f"{s['wiring_mismatches']} | {s['type_mismatches']} | {s['state_boundary_differences']} | {s['input_map_conflicts']} | {s['cross_step_refs_equal']} | **{s['verdict']}** |")
    L += ["", f"- input bijection: {R['input_bijection']}"[:600],
          f"- definitions reachable from the region: equal={R['definitions']['equal']} (mine {R['definitions']['n_reachable_mine']}, oracle {R['definitions']['n_reachable_oracle']}; "
          f"missing in mine {len(R['definitions']['missing_in_mine'])}, in oracle {len(R['definitions']['missing_in_oracle'])}, differing {R['definitions']['n_differing']})",
          f"- gates per spec equal on shared specs: {R['gates_per_spec']['equal_on_shared']}; region gates mine {R['region_gates']['mine']:,} / oracle {R['region_gates']['oracle']:,}",
          f"- **expanded-circuit verdict: {R['expanded_circuit']['verdict']}** ({R['expanded_circuit']['a_node_level']}; definitions equal {R['expanded_circuit']['b_definitions_equal']})", ""]
    for s in R["per_step"]:
        if s["verdict"] != "EXACT":
            L.append(f"## step {s['mine_step']} vs {s['oracle_step']}: {s['verdict']}")
            if s.get("first_spec_divergence"):
                L.append(f"- first spec divergence: `{json.dumps(s['first_spec_divergence'])}`")
            for d in s.get("static_parameter_diffs", [])[:20]:
                L.append(f"- statics differ in `{d['family']}` x{d['count']}: " + ", ".join(f"{x['name']}: mine {x['mine']} / oracle {x['oracle']}" for x in d["statics"]))
            if s["spec_histogram_diff"]:
                L.append(f"- spec histogram diff: `{json.dumps(s['spec_histogram_diff'])[:1500]}`")
            for w in s["wiring_mismatch_examples"][:10]:
                L.append(f"- wiring: `{json.dumps(w)}`")
            for w in s["type_mismatch_examples"][:5]:
                L.append(f"- type: `{json.dumps(w)}`")
            for w in s["state_boundary_examples"][:6]:
                L.append(f"- state boundary: `{json.dumps(w)[:700]}`")
            if s["cross_step_refs"]["mine"] or s["cross_step_refs"]["oracle"]:
                L.append(f"- cross-step refs mine: `{json.dumps(s['cross_step_refs']['mine'])[:800]}`")
                L.append(f"- cross-step refs oracle: `{json.dumps(s['cross_step_refs']['oracle'])[:800]}`")
            L.append("")
    if R["definitions"]["differing"]:
        L.append("## differing definitions")
        for d in R["definitions"]["differing"][:50]:
            L.append(f"- `{d}`")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
