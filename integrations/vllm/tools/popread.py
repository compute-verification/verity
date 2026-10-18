"""POPREAD: whether each unit's cross-unit reads are fixed by public structure or depend on a committed value.

A Q_word unit (`verity/partition/v1`, X = 16) lies inside one Call, a root node of the Program.  It reads other units through
its Call's operands and through its Definition's own wiring, and descriptor/v1 references are all static (affine, strided,
explicit, concat), so within one Program no read position depends on a value.  A value can still steer what a unit reads in
two ways, and this tool names both, per Call family:

* **VALUE-DEPENDENT**: the unit reads a Program input whose binding to other units' outputs is planned from committed values.
  Today that is a warm request's `prefix` input, bound to its producers' KV rows by prompt-block equality (`prefix_plan`), and
  `plan` tests a planner for it by changing committed values with the public launch context held fixed.
* **PUBLIC, selecting in the unit**: the unit reads every candidate at fixed positions and a selection primitive inside it
  (`Gather<T>x<V>`, `Select<T>`) picks one by a value.  The read set is public; the selection's index is named (the Call
  family or input that produces it).

`--workload-program` adds the activity the stop rule keys on committed tokens: a request's steps from 1 + LAG on run only if
no earlier token was an EOS, so their units are programmed (their reads stay public) but their activity is value-keyed.  Under
a stop rule with ignore_eos no request stops early, and the counts say what an EOS stop would key.

    python tools/popread.py descriptor DESCRIPTOR.json.gz [--units] [--workload-program workload_program.json] --out OUT.json
    python tools/popread.py served BUILD.json [--units] [--ignore-eos] [--lag N] --out OUT.json   (a served-compiled P_served)
    python tools/popread.py plan WORKLOAD.json [...] --planner prefix_plan.py --out OUT.json
    python tools/popread.py table OUT.json [...] [--md OUT.md]
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

SCHEMA = "popread/v1"
UNIT_BITS = 16
INPUT = re.compile(r"^(Registered)?Input\d+_v\d+$")
SELECT = re.compile(r"^(?:Gather(?P<t>[A-Za-z0-9]*?)x(?P<v>\d+)(?:u\d+)?|Select(?P<s>[A-Za-z0-9]+))_v\d+$")
INDEX_PARAMS = ("token", "idx", "index", "c", "cond", "pred", "sel")
CONST = re.compile(r"^Const\d+\[")
#: input groups whose binding to other units' outputs is planned from committed values, and the value
VALUE_BOUND = {"prefix": "prompt_token_ids: the cached length C and each block's producer come from prompt-block equality "
                         "(prefix_plan.plan), so which KV rows the warm request reads depends on the prompts"}


def base(fid: str) -> str:
    return fid.split("{", 1)[0]


def group(name: str) -> str:
    """An input's group: its name without a request prefix (`r3.`) or an index (`peers_12`, `weights[4]`)."""
    name = re.sub(r"^r\d+\.", "", name)
    return re.sub(r"(_\d+|\[\d+\])+$", "", name.split(".")[0]) or name


def load(path: str | Path) -> dict[str, Any]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def members(node: dict[str, Any]) -> int:
    return int(node.get("n", 1)) if node["form"] in ("batch", "scan") else 1


def ref_parts(refs: dict[str, Any], nodes: list[dict[str, Any]]) -> Iterator[tuple[str, int]]:
    """The (space, index) of every part a reference sequence reads, aliases resolved."""
    stack = [refs]
    while stack:
        j = stack.pop()
        if "concat" in j:
            stack.extend(j["concat"])
        elif "alias" in j:
            k, a = j["alias"]
            stack.append(nodes[k]["args"][a]["refs"])
        elif "affine" in j:
            if j["affine"][3]:
                yield j["affine"][0], j["affine"][1]
        elif "strided" in j:
            if j["strided"][6]:
                yield j["strided"][0], j["strided"][1]
        elif "explicit" in j:
            yield from {(s, i) for s, i, _ in j["explicit"]}


class Walk:
    """A raw descriptor/v1's Definitions: gates per activation and the selection primitives inside, with what indexes them."""

    def __init__(self, desc: dict[str, Any]) -> None:
        self.defs, self.callees = desc["definitions"], desc["callees"]
        self._gates: dict[str, int] = {}
        self._sel: dict[str, Counter] = {}

    def callee(self, node: dict[str, Any]) -> str:
        return self.callees[node["fn"]]

    def gates(self, fid: str) -> int:
        got = self._gates.get(fid)
        if got is None:
            d = self.defs[fid]
            got = 1 if d["form"] == "primitive" else sum(members(n) * self.gates(self.callee(n)) for n in d["body"]["nodes"])
            self._gates[fid] = got
        return got

    def _label(self, nodes: list[dict[str, Any]], k: int) -> tuple[str, str]:
        fid = self.callee(nodes[k])
        return ("v", "const") if CONST.match(fid) else ("v", base(fid))

    def selects(self, fid: str) -> Counter:
        """{(selection primitive, frozenset of index sources): count per activation}, a source ("p", i) for the Definition's
        parameter i or ("v", family) for the body node that computes the index."""
        got = self._sel.get(fid)
        if got is not None:
            return got
        d, got = self.defs[fid], Counter()
        if d["form"] == "primitive":
            if SELECT.match(fid):
                names = [p[0] for p in d["params"]]
                i = next((names.index(n) for n in INDEX_PARAMS if n in names), 0)
                got[(fid, frozenset({("p", i)}))] = 1
        else:
            nodes = d["body"]["nodes"]
            for n in nodes:
                sub = self.selects(self.callee(n))
                if not sub:
                    continue
                m = members(n)
                for (prim, srcs), c in sub.items():
                    new: set = set()
                    for s in srcs:
                        if s[0] != "p":
                            new.add(s)
                            continue
                        for space, i in ref_parts(n["args"][s[1]]["refs"], nodes):
                            new.add(("p", i) if space == "p" else self._label(nodes, i))
                    got[(prim, frozenset(new))] += m * c
        self._sel[fid] = got
        return got


def root_selects(desc: dict[str, Any]) -> Counter:
    """A one-Definition Program's selections, an index source ("p", i) for the Definition's parameter i (its i-th Input node)
    or ("v", family) for the body node that computes it."""
    W = Walk(desc)
    nodes = W.defs[desc["root"]]["body"]["nodes"]
    param = {}
    for k, n in enumerate(nodes):
        if INPUT.match(W.callee(n)):
            param[k] = len(param)
    got: Counter = Counter()
    for k, n in enumerate(nodes):
        if k in param:
            continue
        for (prim, isrc), c in W.selects(W.callee(n)).items():
            new: set = set()
            for s in isrc:
                if s[0] != "p":
                    new.add(s)
                    continue
                for _space, i in ref_parts(n["args"][s[1]]["refs"], nodes):
                    new.add(("p", param[i]) if i in param else W._label(nodes, i))
            got[(prim, frozenset(new))] += members(n) * c
    return got


def select_kind(prim: str) -> tuple[str, int]:
    m = SELECT.match(prim)
    return (f"Gather{m['t']}x{m['v']}", int(m["v"])) if m["v"] else (f"Select{m['s']}", 2)


def input_names(desc: dict[str, Any], n_inputs: int) -> list[str]:
    """The root's input batches' names, in order: the correspondence's run inputs, then its weights."""
    corr = (desc.get("annotations") or {}).get("correspondence") or {}
    names = list(corr.get("inputs") or []) + [f"weight:{w}" for w in corr.get("weights") or []]
    return names if len(names) == n_inputs else [f"input_{k}" for k in range(n_inputs)]


def input_group(name: str) -> str:
    return "weight" if name.startswith("weight:") else group(name)


def classify(families: dict[str, dict[str, Any]], value_bound: dict[str, str]) -> None:
    for f in families.values():
        bound = sorted(g for g in f["reads"]["inputs"] if g in value_bound)
        f["verdict"] = "VALUE-DEPENDENT" if bound else "PUBLIC"
        f["value"] = "; ".join(f"{g}: {value_bound[g]}" for g in bound) or None


def _family(name: str) -> dict[str, Any]:
    return {"family": name, "calls": 0, "gates": 0, "units": 0, "eos_keyed": {"calls": 0, "gates": 0, "units": 0},
            "eos_step_keyed": {"calls": 0, "gates": 0, "units": 0}, "reads": {"inputs": set(), "calls": set()}, "selects": Counter()}


def _finish(out: dict[str, Any], families: dict[str, dict[str, Any]], value_bound: dict[str, str], have_units: bool) -> dict:
    classify(families, value_bound)
    T = {"calls": sum(f["calls"] for f in families.values()), "gates": sum(f["gates"] for f in families.values()),
         "units": sum(f["units"] for f in families.values()) if have_units else None}
    rows = []
    for f in sorted(families.values(), key=lambda f: -(f["units"] if have_units else f["gates"])):
        sel = defaultdict(lambda: {"per_family": 0, "index": set()})
        for (kind, V, idx), c in f["selects"].items():
            sel[(kind, V)]["per_family"] += c
            sel[(kind, V)]["index"].update(idx)
        rows.append({**{k: f[k] for k in ("family", "calls", "gates", "verdict", "value", "eos_keyed", "eos_step_keyed")},
                     "units": f["units"] if have_units else None,
                     "share_gates": f["gates"] / T["gates"] if T["gates"] else 0.0,
                     "share_units": (f["units"] / T["units"] if T["units"] else 0.0) if have_units else None,
                     "reads": {"inputs": sorted(f["reads"]["inputs"]), "calls": sorted(f["reads"]["calls"])},
                     "selects": [{"select": k, "V": V, "count": s["per_family"], "index": sorted(s["index"])}
                                 for (k, V), s in sorted(sel.items())]})

    def total(pred: Callable[[dict], bool], key: str = "") -> dict[str, Any]:
        sub = [r for r in rows if pred(r)]
        get = (lambda r, k: r[key][k]) if key else (lambda r, k: r[k])
        o = {k: sum(get(r, k) or 0 for r in sub) for k in ("calls", "gates", "units")}
        o["share_gates"] = o["gates"] / T["gates"] if T["gates"] else 0.0
        o["share_units"] = (o["units"] / T["units"] if T["units"] else 0.0) if have_units else None
        return o

    out.update(totals=T, families=rows, summary={
        "value_dependent": total(lambda r: r["verdict"] == "VALUE-DEPENDENT"),
        "in_unit_select": total(lambda r: any(s["V"] > 2 for s in r["selects"])),
        "eos_keyed": total(lambda r: True, "eos_keyed"),
        "eos_step_keyed": total(lambda r: True, "eos_step_keyed"),
    })
    return out


def step_of_calls(wp: dict[str, Any], n_calls: int) -> tuple[list[int] | None, int, str | None]:
    """Call ordinal -> its request-relative step, from a workload Program's chronology, whose node ranges index the request
    body's nodes after its Input batches (`program.workload`), and LAG; None when the ranges don't tile this Program's Calls."""
    ch, lag = wp.get("chronology") or {}, int((wp.get("lag") or {}).get("value") or 0)
    spans = [(m["nodes"][0], m["nodes"][1], int(m["s"])) for e in ch.get("engine_steps") or () for m in e["members"]]
    if len({m["request_id"] for e in ch.get("engine_steps") or () for m in e["members"]}) != 1:
        return None, lag, "the chronology has several requests: per-request steps need the component's own ranges"
    end = max((hi for _lo, hi, _s in spans), default=0)
    if end != n_calls:
        return None, lag, f"the chronology's node ranges end at {end}, not this Program's {n_calls} Calls"
    step = [0] * n_calls
    for lo, hi, s in spans:
        step[lo:hi] = [s] * (hi - lo)
    return step, lag, None


def q_word_units(path: str | Path) -> tuple[dict[int, int], int]:
    """Q_word{X=16, W=32} v1's units per Call (root node index), and the population."""
    from verity.ml.circuits import partition_object as PO
    from verity_vllm.program.boolean import load_program
    program, _digest = load_program(Path(path))
    part = PO.evaluate(program, PO.query(X=UNIT_BITS))
    return {k: c.units for k, c in part.calls}, part.population


def analyze_descriptor(desc: dict[str, Any], *, units: dict[int, int] | None = None, workload_program: dict | None = None,
                       value_bound: dict[str, str] = VALUE_BOUND) -> dict[str, Any]:
    W = Walk(desc)
    nodes = W.defs[desc["root"]]["body"]["nodes"]
    is_in = [bool(INPUT.match(W.callee(n))) for n in nodes]
    names = input_names(desc, sum(is_in))
    label: list[str] = []
    it = iter(names)
    for k, n in enumerate(nodes):
        label.append(f"input:{input_group(next(it))}" if is_in[k] else f"call:{base(W.callee(n))}")
    n_calls = len(nodes) - sum(is_in)
    step, lag, why = step_of_calls(workload_program, n_calls) if workload_program else (None, 0, "no workload Program given")
    fams: dict[str, dict[str, Any]] = {}
    by_step: dict[int, dict[str, Any]] = {}
    ordinal = -1
    for k, n in enumerate(nodes):
        if is_in[k]:
            continue
        ordinal += 1
        fid = W.callee(n)
        f = fams.setdefault(base(fid), _family(base(fid)))
        g, u = W.gates(fid) * members(n), (units or {}).get(k, 0)
        f["calls"] += 1
        f["gates"] += g
        f["units"] += u
        if step is not None:
            b = by_step.setdefault(step[ordinal], {"s": step[ordinal], "calls": 0, "gates": 0, "units": 0, "families": Counter()})
            b["calls"], b["gates"], b["units"] = b["calls"] + 1, b["gates"] + g, b["units"] + u
            b["families"][base(fid)] += u if units is not None else g
        if step is not None and step[ordinal] >= 1 + lag:
            for key, v in (("calls", 1), ("gates", g), ("units", u)):
                f["eos_keyed"][key] += v
                f["eos_step_keyed"][key] += v
        srcs = [{label[i] for _s, i in ref_parts(a["refs"], nodes)} for a in n["args"]]
        for s in srcs:
            for lab in s:
                kind, _, what = lab.partition(":")
                f["reads"]["inputs" if kind == "input" else "calls"].add(what)
        for (prim, isrc), c in W.selects(fid).items():
            idx = set()
            for s in isrc:
                idx |= srcs[s[1]] if s[0] == "p" else {f"in-call:{s[1]}"}
            kind, V = select_kind(prim)
            f["selects"][(kind, V, frozenset(idx))] += c * members(n)
    ignore = bool(((workload_program or {}).get("stop_rule") or {}).get("ignore_eos"))
    out = {"schema": SCHEMA, "adapter": "descriptor/v1", "root": desc["root"][:160],
           "inputs": dict(Counter(input_group(x) for x in names)), "value_bound": value_bound,
           "eos": _eos(lag, step is not None, ignore, why),
           "by_step": [dict(b, families=dict(b["families"]), per_family="units" if units is not None else "gates")
                       for _s, b in sorted(by_step.items())]}
    return _finish(out, fams, value_bound, units is not None)


IGNORE_EOS = "the row serves with ignore_eos, so every request runs to max_tokens"


def _eos(lag: int, counted: bool, ignore: bool, why: str | None) -> dict[str, Any]:
    """`counted`: the keyed counts are filled (what an EOS stop would key); `applied`: the row's stop rule honors EOS, so they are."""
    return {"lag": lag, "keyed_from_step": 1 + lag, "counted": counted, "applied": counted and not ignore,
            "why_not": why if not counted else (IGNORE_EOS if ignore else None)}


def _value_label(r: dict[str, Any], producer: dict[str, str]) -> str:
    if "weight" in r:
        return "input:weight"
    v = r.get("value", "")
    return f"input:{v.split('/', 1)[1]}" if v.startswith("c00.input/") else f"call:{producer.get(v, '?')}"


def analyze_served(build: dict[str, Any], *, units: bool = False, ignore_eos: bool = False, lag: int = 0) -> dict[str, Any]:
    """A served-compiled Build's P_served: per step, each Call's Definition applied written width / its return leaves times per
    token row, attention once per query row and layer, the logits and the sampled id once per request.  The plan runs every
    request to max_tokens, so a request's rows from its own step 1 + LAG on are the activity an EOS stop would key
    (`eos_keyed`), and every row of an engine step holding one of them is in a batch whose shape that stop would change
    (`eos_step_keyed`: the step's tokens, padding, capture mode, Gemm M and attention splits).  `ignore_eos`: the row serves
    with it (the Build doesn't record the sampling), so no stop happens and both counts are what one would key."""
    from verity.ml.circuits.codec import encode_program
    from verity.ml.circuits.cut import evaluate_definition
    from verity.ml.circuits.program import Program
    from verity_vllm.query.call_scope import specialization

    P = build["program"]
    width = {k: int(v["width"]) for k, v in P["values"].items()}
    producer = {w: c.get("base") or "Gemm" for c in P["calls"] for w in c.get("writes") or ()}
    for a in P.get("attention") or ():
        producer[a["name"]] = "Attention"
    producer["logits"] = base(P["logits"].get("name") or "logits")
    arrive = {r["request_id"]: int(r["arrive"]) for r in (build.get("workload") or {}).get("schedule") or ()}
    memo: dict[str, Any] = {}
    cut_memo: dict = {}

    def info(spec: str, statics: dict[str, Any]) -> dict[str, Any]:
        got = memo.get(spec)
        if got is None:
            d = specialization(base(spec), statics)
            sel = root_selects(encode_program(Program(d)))
            ret = d.signature()[1]
            got = memo[spec] = {"gates": d.gates, "ret": ret.leaves, "selects": sel,
                                "units": evaluate_definition(d, UNIT_BITS, cut_memo).units if units else 0}
        return got

    fams: dict[str, dict[str, Any]] = {}

    def add(fam: str, spec: str, statics: dict[str, Any], apps: int, reads: list[dict[str, Any]], eos_apps: int,
            step_keyed: bool) -> None:
        if apps <= 0:
            return
        i, f = info(spec, statics), fams.setdefault(fam, _family(fam))
        f["calls"] += apps
        f["gates"] += apps * i["gates"]
        f["units"] += apps * i["units"]
        for field, n in (("eos_keyed", eos_apps), ("eos_step_keyed", apps if step_keyed else 0)):
            for key, v in (("calls", n), ("gates", n * i["gates"]), ("units", n * i["units"])):
                f[field][key] += v
        labs = [_value_label(r, producer) for r in reads]
        for lab in labs:
            kind, _, what = lab.partition(":")
            f["reads"]["inputs" if kind == "input" else "calls"].add(what)
        for (prim, isrc), c in i["selects"].items():
            idx = {labs[s[1]] if s[0] == "p" and s[1] < len(labs) else f"in-call:{s[1]}" for s in isrc}
            kind, V = select_kind(prim)
            f["selects"][(kind, V, frozenset(idx))] += apps * c

    for plan in build["steps_plan"]:
        s, tokens = int(plan["step"]), int(plan["tokens"])
        keyed = {r["request_id"]: s - arrive.get(r["request_id"], 0) >= 1 + lag for r in plan["requests"]}
        eos_rows = sum(int(r["tokens"]) for r in plan["requests"] if keyed[r["request_id"]])
        any_keyed = any(keyed.values())
        for c in P["calls"]:
            if "definition" in c:
                spec, statics = c["definition"], c["statics"]
            else:
                g = plan["gemms"][",".join(str(x) for x in c["gemm"])]
                spec, statics = g["definition"], g["statics"]
            per_row = max(width[c["writes"][0]] // info(spec, statics)["ret"], 1)
            add(base(spec), spec, statics, tokens * per_row, c["reads"], eos_rows * per_row, any_keyed)
        for r in plan["requests"]:
            k = keyed[r["request_id"]]
            for a in r["attention"]:
                for att in P.get("attention") or ():
                    per = max(width[att["q"]["value"]] // info(a["definition"], a["statics"])["ret"], 1)
                    add(base(a["definition"]), a["definition"], a["statics"], per, [att["q"], att["k"], att["v"]], per * k, any_keyed)
            lg, sm = plan["logits"], P["sampled"]
            add(base(lg["definition"]), lg["definition"], lg["statics"], 1, [P["logits"]["x"], P["logits"]["w"]], int(k), any_keyed)
            add(sm["base"], sm["definition"], sm["statics"], 1, [{"value": "logits"}], int(k), any_keyed)
    out = {"schema": SCHEMA, "adapter": "P_served", "program_digest": build.get("program_digest"),
           "schedule": build.get("workload"), "value_bound": VALUE_BOUND,
           "inputs": {"ids": 1, "positions": 1, "weight": len({r["weight"] for c in P["calls"] for r in c["reads"] if "weight" in r})},
           "plan_inputs": "steps_plan reads the declared schedule only (arrive, prompt_tokens, max_tokens) and the dispatch "
                          "(capture sizes, max_num_seqs, split_cap, sms): no token value",
           "eos": _eos(lag, True, ignore_eos, None)}
    return _finish(out, fams, VALUE_BOUND, units)


def load_planner(path: str | Path, name: str = "plan") -> Callable[[dict], Any]:
    spec = importlib.util.spec_from_file_location(f"popread_planner_{Path(path).stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def _variants(w: dict[str, Any], seed: int) -> Iterator[tuple[str, dict[str, Any]]]:
    """The workload with its committed values changed and its public launch context (lengths, arrivals, engine) held: every
    prompt's ids replaced by fresh ones, and every prompt made a copy of the first's prefix."""
    rng = random.Random(seed)
    reqs = w.get("requests") or []
    if not any(isinstance(r.get("prompt_token_ids"), list) for r in reqs):
        return
    fresh = [dict(r, prompt_token_ids=[rng.randrange(1, 1 << 15) for _ in r["prompt_token_ids"]]) for r in reqs]
    yield "distinct prompts", dict(w, requests=fresh)
    first = max(reqs, key=lambda r: len(r["prompt_token_ids"]))["prompt_token_ids"]
    same = [dict(r, prompt_token_ids=list(first[:len(r["prompt_token_ids"])])) for r in reqs]
    yield "one shared prompt", dict(w, requests=same)


def plan_check(w: dict[str, Any], planner: Callable[[dict], Any], seed: int = 0) -> dict[str, Any]:
    """Whether `planner(workload)` (the Program shapes and input bindings it plans) changes when only committed values change."""
    got = planner(w)
    changed = [{"variant": name, "plan": p} for name, v in _variants(w, seed) if (p := planner(v)) != got]
    return {"plan": got, "verdict": "VALUE-DEPENDENT" if changed else "PUBLIC",
            "value": "prompt_token_ids" if changed else None, "changed_by": changed,
            "held": ["prompt_len", "max_tokens", "arrive_step", "engine"]}


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.2f}%"


def render(reports: Iterable[dict[str, Any]]) -> str:
    out = []
    for r in reports:
        if "families" not in r:
            out.append(f"### {r.get('name', '?')}: plan\n\n{r.get('verdict')} (value: {r.get('value')}); "
                       f"changed by: {', '.join(c['variant'] for c in r.get('changed_by') or []) or 'nothing'}\n")
            continue
        t, s = r["totals"], r["summary"]
        out.append(f"### {r.get('name', '?')}\n\n{t['calls']:,} Calls, {t['gates']:,} gates"
                   + (f", {t['units']:,} Q_word units" if t["units"] is not None else "") + "\n")
        out.append("| family | calls | units | share units | gates | share gates | cross-unit reads | in-unit selection (index) |")
        out.append("|---|---:|---:|---:|---:|---:|---|---|")
        for f in r["families"]:
            sel = "; ".join(f"{x['select']} x{x['count']:,} by {', '.join(x['index'])}" for x in f["selects"]) or "-"
            reads = f["verdict"] + (f" ({f['value']})" if f["value"] else "")
            out.append(f"| {f['family']} | {f['calls']:,} | {'-' if f['units'] is None else format(f['units'], ',')} | "
                       f"{_fmt(f['share_units'])} | {f['gates']:,} | {_fmt(f['share_gates'])} | {reads} | {sel} |")
        vd, sel = s["value_dependent"], s["in_unit_select"]
        out.append(f"\nVALUE-DEPENDENT: {_fmt(vd['share_units'])} of units, {_fmt(vd['share_gates'])} of gates. "
                   f"PUBLIC with an in-unit gather (V > 2): {_fmt(sel['share_units'])} of units, {_fmt(sel['share_gates'])} of gates. "
                   + _render_eos(r["eos"], s) + "\n")
    return "\n".join(out)


def _render_eos(e: dict[str, Any], s: dict[str, Any]) -> str:
    if not e.get("counted", e.get("applied")):
        return f"EOS: {e['why_not']}."
    own, step = s["eos_keyed"], s.get("eos_step_keyed") or s["eos_keyed"]
    keyed = (f"{_fmt(own['share_units'])} of units ({_fmt(own['share_gates'])} of gates) are a request's steps >= "
             f"{e['keyed_from_step']}, and {_fmt(step['share_units'])} of units ({_fmt(step['share_gates'])} of gates) are in "
             f"an engine step holding one")
    return f"Activity keyed on EOS: {keyed}." if e["applied"] else f"EOS: {e['why_not']}; were EOS honored, {keyed}."


def _bound(items: list[str]) -> dict[str, str]:
    vb = dict(VALUE_BOUND)
    for it in items:
        g, _, why = it.partition("=")
        vb[g] = why
    return vb


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("descriptor")
    d.add_argument("descriptor")
    d.add_argument("--units", action="store_true", help="Q_word units per Call (decodes the Program through the registry)")
    d.add_argument("--workload-program", help="workload_program.json: its chronology attributes root nodes to steps")
    d.add_argument("--value-bound", action="append", default=[], metavar="GROUP=WHY")
    d.add_argument("--name", default=None)
    d.add_argument("--out", required=True)
    s = sub.add_parser("served")
    s.add_argument("build")
    s.add_argument("--units", action="store_true")
    s.add_argument("--ignore-eos", action="store_true", help="the row serves with ignore_eos (served_match does)")
    s.add_argument("--lag", type=int, default=0, help="the engine's LAG: steps a request runs past its EOS")
    s.add_argument("--name", default=None)
    s.add_argument("--out", required=True)
    p = sub.add_parser("plan")
    p.add_argument("workload", nargs="+")
    p.add_argument("--planner", required=True, help="a planner module's path, PATH[:FUNC] (default FUNC plan)")
    p.add_argument("--out", required=True)
    t = sub.add_parser("table")
    t.add_argument("reports", nargs="+")
    t.add_argument("--md", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "descriptor":
        desc = load(a.descriptor)
        units = None
        if a.units:
            units, population = q_word_units(a.descriptor)
        r = analyze_descriptor(desc, units=units, value_bound=_bound(a.value_bound),
                               workload_program=json.loads(Path(a.workload_program).read_text()) if a.workload_program else None)
        r["name"] = a.name or a.descriptor
        if units is not None:
            r["population"] = population
        Path(a.out).write_text(json.dumps(r, indent=1, default=sorted))
    elif a.cmd == "served":
        r = analyze_served(json.loads(Path(a.build).read_text()), units=a.units, ignore_eos=a.ignore_eos, lag=a.lag)
        r["name"] = a.name or a.build
        Path(a.out).write_text(json.dumps(r, indent=1, default=sorted))
    elif a.cmd == "plan":
        path, _, fn = a.planner.partition(":")
        planner = load_planner(path, fn or "plan")
        reports = [dict(plan_check(json.loads(Path(w).read_text()), planner), name=Path(w).stem) for w in a.workload]
        Path(a.out).write_text(json.dumps(reports if len(reports) > 1 else reports[0], indent=1))
    else:
        reps: list[dict[str, Any]] = []
        for x in a.reports:
            j = json.loads(Path(x).read_text())
            reps.extend(j if isinstance(j, list) else [j])
        md = render(reps)
        if a.md:
            Path(a.md).write_text(md)
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
