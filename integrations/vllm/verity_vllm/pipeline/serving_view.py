"""A served row's request Programs arranged the way vLLM serves them (`vllm-serving-view/v1`): engine steps, the requests in each
step, each one's model runner (the model's modules, the sampler, the state read across steps) and the TP ranks.

A view, not a Program: every node is a set of Calls of the Build's own request Programs; nothing is re-derived or re-digested.  Read
from the Build's records and the IR, never from names:

  engine steps  the Build's chronology (`build_workload/chronology.json`): per engine step e its members (request r, its step s, the
                phase).  A Build that kept only its request Programs has no chronology: its steps are the request steps (arrival
                not recorded), and the view says so
  request step  per Call, by dataflow.  The sampler of step s produces element s of the request Program's return (the served tokens:
                the descriptor's root `ret`), or with no descriptor it is the s-th Call of a sampling-event Definition
                (`program.sampling_event`, the Build's own step boundary).  A Call's step is the first step whose sampler needs it (it
                is the sampler or one of its ancestors); a Call no sampler needs takes max over the Calls it reads of their step, plus
                one past a sampler
  model runner  per (request, step): the sampler = the step's sampler Call, its constants (Calls reading no Call, read by it alone)
                and the request inputs it reads; the model = the step's other Calls by recorded owning module
                (`Correspondence.module_of`, nested as `program_graph` nests them: the nearest enclosing module path); state
                edges = reads of Values produced at an earlier step, other than the previous sampler's token: the KV cache
  TP ranks      per module body, each rank's request Program's Calls; a collective edge runs from the Calls rank b feeds its Call
                reading `peers_<k>[lo:hi]` with to rank a's Call reading the same `peers_<k>[lo:hi]` (the same collective)
  gates         a Call's gates are its specialization's (`SpecializedDefinition.gates`); a request's Calls tile its address-map range
                in body order (checked), so every node's gate count is exact and a contiguous node has a gate range
  words         Q_word_v1{X,W} per Call (`query.word.unit_rule` of its specialization): units and committed interior words

The leaves (a module body or sampler, one rank, one request step) partition every Call of every request Program exactly once; the
generator fails unless they do, unless every member's and every request's sums are the Program's, and unless the members equal the
node ranges the chronology records.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from verity_vllm.config import option
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.pipeline import program_graph as PG
from verity_vllm.program.sampling_event import is_sampling_event
from verity.ml.definitions import word as word_query
from verity_vllm.query.program_view import _read_instance_header, from_instances, iter_instance_rows, spec_base, spec_statics

SCHEMA = "vllm-serving-view/v1"
PEER = re.compile(r"peers_(\d+)$")
EXPANDED = (0, 1)


def _h(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _runs(ks: Iterable[int]) -> list[list[int]]:
    """Sorted Call ids as half-open runs [lo, hi)."""
    out: list[list[int]] = []
    for k in ks:
        if out and out[-1][1] == k:
            out[-1][1] = k + 1
        else:
            out.append([k, k + 1])
    return out


@dataclass
class RequestProgram:
    """One request Program of one rank, with its Calls' steps, modules, gates and words."""

    label: str
    rank: int
    path: str
    digest: str
    n: int
    step: np.ndarray
    samplers: list[int]
    sampler_rule: str
    module: list[str]
    gates: np.ndarray
    units: np.ndarray
    words: np.ndarray
    spec: list[str]
    keys: list[tuple[str, str]]
    inputs: dict[int, str]
    cols: Any
    shapes: list[str]
    constant_of: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    sampler_inputs: dict[int, dict[str, list[list[int]]]] = field(default_factory=dict)
    peer_runs: dict[tuple[int, int, int], int] = field(default_factory=dict)
    call_words: dict[tuple[int, int], int] = field(default_factory=dict)
    input_words: dict[tuple[int, int], int] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)


def request_program_paths(root: str | Path) -> list[tuple[int, str]]:
    """(rank, instances.json.gz) of every request Program under a Build: per rank every `build_request_LP*` (B > 1: `build_request`
    is then one of them again, the component of record) else `build_request`; `build_step`'s Program is not a request's."""
    by_rank: dict[int, list[str]] = {}
    for p in sorted(Path(root).glob("**/instances.json.gz")):
        if not p.parent.name.startswith("build_request"):
            continue
        m = re.search(r"/rank(\d+)/", str(p))
        by_rank.setdefault(int(m.group(1)) if m else 0, []).append(str(p))
    out = []
    for rank, ps in sorted(by_rank.items()):
        lp = [p for p in ps if Path(p).parent.name.startswith("build_request_LP")]
        out += [(rank, p) for p in (lp or ps)]
    return out


def _returned_calls(path: str, n: int) -> list[int] | None:
    """The Calls producing the request Program's return elements, in order (its descriptor beside `instances.json.gz`), or None."""
    d = Path(path).parent / "descriptor.json.gz"
    if not d.exists():
        return None
    desc = json.load(gzip.open(d, "rt"))
    body = desc["definitions"][desc["root"]]["body"]
    K = len(body["nodes"]) - n
    ret = body["ret"]
    out = []
    for r in ret["concat"] if "concat" in ret else [ret]:
        (space, idx, _base, count), = r.values()
        if space != "n" or count != 1 or idx < K:
            raise ValueError(f"{d}: a return element that is not one Call's Value: {r}")
        out.append(idx - K)
    return out


def _step_of_calls(cols: Any, samplers: list[int]) -> tuple[np.ndarray, dict[str, int]]:
    """Per Call its request step: the first step whose sampler needs it, else (needed by none) the forward rule."""
    n = cols.n
    off = np.frombuffer(cols.read_off, dtype=np.int64)
    prod = np.frombuffer(cols.read_call, dtype=np.int32)
    is_s = np.zeros(n, dtype=np.int64)
    is_s[samplers] = 1
    INF = np.int64(1 << 40)
    need = np.full(n, INF, dtype=np.int64)
    for s, k in enumerate(samplers):
        need[k] = min(need[k], s)
    fwd = np.zeros(n, dtype=np.int64)
    for k in range(n - 1, -1, -1):
        if need[k] < INF:
            ps = prod[off[k]:off[k + 1]]
            ps = ps[ps >= 0]
            if ps.size:
                np.minimum.at(need, ps, need[k])
    for k in range(n):
        ps = prod[off[k]:off[k + 1]]
        ps = ps[ps >= 0]
        if ps.size:
            fwd[k] = int((fwd[ps] + is_s[ps]).max())
    dead = need >= INF
    step = np.where(dead, fwd, need)
    seg = np.cumsum(is_s) - is_s
    return step, {"needed_by_a_sampler": int((~dead).sum()), "needed_by_none": int(dead.sum()),
                  "equal_forward_rule": int((step == fwd).sum()), "later_than_forward_rule": int((step > fwd).sum()),
                  "equal_body_order_segments": int((step == seg).sum())}


def load_request_program(rank: int, path: str, *, acc: PG._Acc, X: int, W: int, memo: dict) -> RequestProgram:
    hdr = _read_instance_header(path)
    P = from_instances(path)
    cols = P.columns
    n = cols.n
    ret = _returned_calls(path, n)
    if ret is not None:
        samplers, rule = ret, "return"
    else:
        samplers = [k for k in range(n) if is_sampling_event(cols.specs[cols.spec_idx[k]])]
        rule = "sampling-event"
    step, chk = _step_of_calls(cols, samplers)
    C = Correspondence(P)
    mods = [C.module_of(k) for k in range(n)]
    spec_gates: dict[str, int] = {}
    spec_q: dict[str, tuple[int, int]] = {}
    for s in set(cols.specs):
        fn = word_query.specialization(spec_base(s), spec_statics(s))
        spec_gates[s] = fn.gates
        r = word_query.unit_rule(fn, X, W, 0, memo=memo)
        spec_q[s] = (int(r["units_per_call"]), int(r["committed_interior_words"]))
    specs = [cols.specs[cols.spec_idx[k]] for k in range(n)]
    acc.add(P, hdr.get("digest"), Path(path).parent.name)
    vals: dict[tuple[str, str], dict[str, set]] = {}
    for s in set(zip(mods, specs)):
        for a, v in spec_statics(s[1]).items():
            vals.setdefault((s[0], spec_base(s[1])), {}).setdefault(a, set()).add(json.dumps(v, sort_keys=True, default=str))
    shape_of = {}
    for m, s in set(zip(mods, specs)):
        var = {a for a, vs in vals.get((m, spec_base(s)), {}).items() if len(vs) > 1}
        shape_of[(m, s)] = spec_base(s) + json.dumps({a: v for a, v in spec_statics(s).items() if a not in var}, sort_keys=True, default=str) \
            + ("|" + ",".join(sorted(var)) if var else "")
    rp = RequestProgram(label=Path(path).parent.name, rank=rank, path=path, digest=hdr.get("digest"), n=n, step=step, samplers=samplers,
                        sampler_rule=rule, module=mods, gates=np.array([spec_gates[s] for s in specs], dtype=np.int64),
                        units=np.array([spec_q[s][0] for s in specs], dtype=np.int64),
                        words=np.array([spec_q[s][1] for s in specs], dtype=np.int64), spec=specs, keys=acc.call_keys[-1],
                        inputs={j: q.name for j, q in enumerate(P.params)}, cols=cols, shapes=[shape_of[(m, s)] for m, s in zip(mods, specs)], checks=chk)
    names = {p.name: p.id for p in P.params}
    if "splits" in names:
        pread_off = np.frombuffer(cols.pread_off, dtype=np.int64)
        pread_call = np.frombuffer(cols.pread_call, dtype=np.int32)
        readers = [k for k in range(n) if (pread_call[pread_off[k]:pread_off[k + 1]] == names["splits"]).any()]
        rp.checks["splits_readers_are_the_samplers"] = readers == sorted(samplers)
    off = np.frombuffer(cols.read_off, dtype=np.int64)
    prod = np.frombuffer(cols.read_call, dtype=np.int32).astype(np.int64)
    cons = np.repeat(np.arange(n, dtype=np.int64), np.diff(off))
    m = prod >= 0
    lo_r, hi_r, reads_call = np.full(n, n, dtype=np.int64), np.full(n, -1, dtype=np.int64), np.zeros(n, dtype=np.int64)
    np.minimum.at(lo_r, prod[m], cons[m])
    np.maximum.at(hi_r, prod[m], cons[m])
    np.add.at(reads_call, cons[m], 1)
    only = np.where(lo_r == hi_r, lo_r, -1)
    is_s = np.zeros(n, dtype=bool)
    is_s[samplers] = True
    rp.constant_of = np.where((reads_call == 0) & (only >= 0) & is_s[np.clip(only, 0, n - 1)], only, -1)
    smp = set(samplers)
    for r in iter_instance_rows(path):
        i = int(r["i"])
        for operand in r.get("args") or ():
            for run in operand:
                if run[0] == "n":
                    key = (int(run[1]), i)
                    rp.call_words[key] = rp.call_words.get(key, 0) + run[3] - run[2]
                elif run[0] == "p":
                    key = (i, int(run[1]))
                    rp.input_words[key] = rp.input_words.get(key, 0) + run[3] - run[2]
                    name = rp.inputs[int(run[1])]
                    if i in smp and name != "weights":
                        rp.sampler_inputs.setdefault(i, {}).setdefault(name, []).append([run[2], run[3]])
                    mm = PEER.match(name)
                    if mm:
                        key3 = (int(mm.group(1)), int(run[2]), int(run[3]))
                        if rp.peer_runs.setdefault(key3, i) != i:
                            raise ValueError(f"{path}: peers_{key3[0]}[{key3[1]}:{key3[2]}] read by two Calls ({rp.peer_runs[key3]}, {i})")
    return rp


class _Nodes:
    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []

    def add(self, kind: str, parent: str | None, label: str, **kw: Any) -> dict[str, Any]:
        node = {"id": f"n{len(self.nodes)}", "kind": kind, "parent": parent, "label": label, "children": [], **kw}
        self.nodes.append(node)
        if parent is not None:
            self.nodes[int(parent[1:])]["children"].append(node["id"])
        return node


FIELDS = ("calls", "gates", "units", "committed_interior_words")


def _sum(node: dict[str, Any], rp: RequestProgram, ks: np.ndarray) -> None:
    node["calls"] = node.get("calls", 0) + int(ks.size)
    node["gates"] = node.get("gates", 0) + int(rp.gates[ks].sum())
    node["units"] = node.get("units", 0) + int(rp.units[ks].sum())
    node["committed_interior_words"] = node.get("committed_interior_words", 0) + int(rp.words[ks].sum())


def serving_view(build_dir: str | Path, *, row: dict[str, Any] | None = None, graph: dict[str, Any] | None = None,
                 X: int = 16, W: int = 32) -> dict[str, Any]:
    """The row's serving view from its Build directory (the sweep row layout: `build_workload/`, `build_request*/`, `rank<r>/`), with
    its program graph (`program.json`) when given for the group ids' cross-check and the Q_word totals."""
    root = Path(build_dir)
    wl = root / "build_workload"
    chron = json.loads((wl / "chronology.json").read_text()) if (wl / "chronology.json").exists() else None
    amap = json.loads((wl / "address_map.json").read_text()) if (wl / "address_map.json").exists() else None
    paths = request_program_paths(root)
    if not paths:
        raise ValueError(f"{root}: no request Programs (build_request*/instances.json.gz)")
    acc, memo = PG._Acc(), {}
    graph_ranks = min(r for r, _p in paths)
    progs = [load_request_program(r, p, acc=acc, X=X, W=W, memo=memo) for r, p in paths if r == graph_ranks]
    gid = {key: f"g{i}" for i, key in enumerate(sorted(acc.groups, key=lambda k: acc.groups[k]["first"]))}
    progs += [load_request_program(r, p, acc=acc, X=X, W=W, memo=memo) for r, p in paths if r != graph_ranks]
    ranks = sorted({p.rank for p in progs})
    by_label: dict[str, dict[int, RequestProgram]] = {}
    for p in progs:
        by_label.setdefault(p.label, {})[p.rank] = p
    by_digest = {p.digest: p.label for p in progs if p.rank == ranks[0]}

    # requests: the address map's (request id -> its Program by digest), else one per request Program
    if amap is not None:
        requests = [{"request_id": r["request_id"], "label": by_digest[r["program_digest"]], "gates": r["gates"], "program_digest": r["program_digest"]}
                    for r in amap["requests"]]
    else:
        requests = [{"request_id": lab, "label": lab, "gates": None, "program_digest": by_label[lab][ranks[0]].digest} for lab in by_label]
    if chron is not None:
        steps = [{"e": s["e"], "members": [(m["request_id"], m["s"], m["phase"]) for m in s["members"]], "co_scheduled": s["co_scheduled"],
                  "mixed_prefill_decode": s["mixed_prefill_decode"]} for s in chron["engine_steps"]]
        recorded = {(m["request_id"], m["s"]): (m.get("nodes"), m.get("gates")) for s in chron["engine_steps"] for m in s["members"]}
    else:
        T = max(int(p.step.max()) for p in progs) + 1
        steps = [{"e": s, "members": [(r["request_id"], s, "prefill" if s == 0 else "decode") for r in requests
                                      if s <= int(by_label[r["label"]][ranks[0]].step.max())], "co_scheduled": None, "mixed_prefill_decode": None}
                 for s in range(T)]
        recorded = {}

    N = _Nodes()
    top = N.add("serving", None, "serving", workload_digest=(amap or {}).get("workload_digest"), gates_total=(amap or {}).get("gates"),
                input_gates=(amap or {}).get("input_gates"), shared=(amap or {}).get("shared"), requests=[r["request_id"] for r in requests],
                ranks=ranks)
    leaf_of = {(rq["request_id"], rank): np.full(by_label[rq["label"]][rank].n, -1, dtype=np.int64) for rq in requests for rank in ranks}
    member_of: dict[tuple[str, int], str] = {}
    req_by_id = {r["request_id"]: r for r in requests}
    hashes: dict[str, str] = {}
    for st in steps:
        en = N.add("engine_step", top["id"], f"engine step {st['e']}", e=st["e"], co_scheduled=st["co_scheduled"],
                   mixed_prefill_decode=st["mixed_prefill_decode"], collapsed=st["e"] not in EXPANDED)
        for rid, s, phase in st["members"]:
            rq = req_by_id[rid]
            mem = N.add("member", en["id"], f"{rid} step {s}", request_id=rid, s=s, phase=phase, program_digest=rq["program_digest"])
            member_of[(rid, s)] = mem["id"]
            _member(N, mem, rq, [by_label[rq["label"]][rank] for rank in ranks], gid, leaf_of, hashes)
    for node in reversed(N.nodes):                                  # children have larger ids: subtree totals bottom-up
        if node["kind"] in ("module", "engine_step", "serving"):
            kids = [N.nodes[int(c[1:])] for c in node["children"]]
            if node["kind"] == "module":
                node["body_calls"] = sum(k["calls"] for k in kids if k["kind"] == "rank")
            for f in FIELDS:
                node[f] = sum(k.get(f, 0) for k in kids)
    checks = _checks(N, requests, ranks, by_label, leaf_of, member_of, recorded)
    edges, checks["edges"] = _edges(N, by_label, requests, ranks, leaf_of)
    checks["ok"] = checks["ok"] and checks["edges"]["ok"]
    if graph is not None:
        checks["graph"] = _graph_check(graph, acc, gid, N)
    out = {"schema": SCHEMA, "row": row or {}, "build": {"dir": str(root), "chronology": chron is not None, "address_map": amap is not None,
                                                          "request_programs": [{"label": p.label, "rank": p.rank, "program_digest": p.digest, "calls": p.n,
                                                                                "gates": int(p.gates.sum()), "sampler_rule": p.sampler_rule}
                                                                               for p in progs]},
           "query": word_query.query_id(X, W, 0), "requests": requests,
           "steps": [dict(st, members=[{"request_id": a, "s": b, "phase": c, "node": member_of[(a, b)]} for a, b, c in st["members"]]) for st in steps],
           "nodes": N.nodes, "edges": edges, "checks": checks, "rules": RULES}
    if not checks["ok"]:
        raise ValueError(f"serving view does not partition the Build's Calls: {json.dumps(checks)[:2000]}")
    return out


def _member(N: _Nodes, mem: dict[str, Any], rq: dict[str, Any], progs: list[RequestProgram], gid: dict, leaf_of: dict, hashes: dict) -> None:
    """A member's model runner (the model's module tree, the sampler) with one leaf per (module body or sampler, rank)."""
    rid, s = mem["request_id"], mem["s"]
    runner = N.add("runner", mem["id"], "model runner")
    model = N.add("model", runner["id"], "model")
    sampler = N.add("sampler", runner["id"], "sampler")
    mod_nodes: dict[str, dict[str, Any]] = {}
    body_keys: dict[str, Counter] = {}
    for j, p in enumerate(progs):
        ks = np.nonzero(p.step == s)[0]
        if j == 0:
            runs = mem["call_runs"] = _runs(ks.tolist())
            if rq["gates"] is not None and len(runs) == 1:
                base = rq["gates"][0] + int(p.gates[:runs[0][0]].sum())
                mem["gate_range"] = [base, base + int(p.gates[ks].sum())]
        own = p.samplers[s:s + 1]
        smp = set(own) | {k for k in ks.tolist() for o in own if p.constant_of[k] == o}
        if smp:
            arr = np.array(sorted(smp), dtype=np.int64)
            rn = N.add("rank", sampler["id"], f"rank {p.rank}", rank=p.rank, spec=p.spec[own[0]], call=int(own[0]), module=p.module[own[0]],
                       group=gid.get(p.keys[own[0]]), inputs=p.sampler_inputs.get(own[0], {}),
                       with_calls=[{"call": int(k), "spec": p.spec[k], "group": gid.get(p.keys[k])} for k in arr.tolist() if k != own[0]])
            _sum(rn, p, arr)
            _sum(sampler, p, arr)
            sampler.setdefault("spec", p.spec[own[0]])
            leaf_of[(rid, p.rank)][arr] = int(rn["id"][1:])
        model_ks = np.array([k for k in ks.tolist() if k not in smp], dtype=np.int64)
        _modules(N, model, mod_nodes, body_keys, p, model_ks, gid, leaf_of[(rid, p.rank)])
        _sum(model, p, model_ks)
        _sum(mem, p, ks)
        _sum(runner, p, ks)

    def subtree(nid: str) -> str:
        nd = N.nodes[int(nid[1:])]
        if nid not in hashes:
            kids = [c for c in nd["children"] if N.nodes[int(c[1:])]["kind"] == "module"]
            hashes[nid] = _h([sorted(body_keys.get(nd.get("path"), Counter()).items()), [subtree(c) for c in kids]])
        return hashes[nid]

    for node in mod_nodes.values():
        node["subtree_hash"] = subtree(node["id"])
    model["subtree_hash"] = _h([subtree(c) for c in model["children"] if N.nodes[int(c[1:])]["kind"] == "module"])
    for node in [model, *mod_nodes.values()]:
        seen: dict[str, str] = {}
        for c in node["children"]:
            cn = N.nodes[int(c[1:])]
            if cn["kind"] != "module":
                continue
            if cn["subtree_hash"] in seen:
                cn["fold_of"] = seen[cn["subtree_hash"]]
            else:
                seen[cn["subtree_hash"]] = c
    mem["structure_hash"] = _h([model["subtree_hash"], sampler.get("spec", "").split("{")[0]])


def _modules(N: _Nodes, model: dict, mod_nodes: dict, body_keys: dict, p: RequestProgram, model_ks: np.ndarray, gid: dict, leaf: np.ndarray) -> None:
    """One rank's model Calls of a member under their module nodes (created on first sight, parents first), one leaf per body."""
    present = {p.module[k] for k in model_ks.tolist()}
    known = set(present)
    first: dict[str, int] = {}
    for k in model_ks.tolist():
        q = p.module[k]
        first.setdefault(q, k)
        while "." in q:
            q = q.rsplit(".", 1)[0]
            known.add(q)
            first.setdefault(q, k)
    known.discard("")
    for m in sorted(known | ({""} & present), key=lambda x: (first.get(x, 1 << 62), x)):
        if m in mod_nodes:
            continue
        par = PG._parent(m, known) if m else None
        parent = model if not m or par == "" else mod_nodes[par]
        mod_nodes[m] = N.add("module", parent["id"], m or "(no module)", path=m, body=m in present)
    by_mod: dict[str, list[int]] = {}
    for k in model_ks.tolist():
        by_mod.setdefault(p.module[k], []).append(k)
    for m, kl in by_mod.items():
        arr = np.array(kl, dtype=np.int64)
        node = mod_nodes[m]
        rn = N.add("rank", node["id"], f"rank {p.rank}", rank=p.rank)
        _sum(rn, p, arr)
        node.setdefault("groups", sorted({gid[p.keys[k]] for k in kl if p.keys[k] in gid}, key=lambda g: int(g[1:])))
        body_keys.setdefault(m, Counter()).update(p.shapes[k] for k in kl)
        leaf[arr] = int(rn["id"][1:])


def _checks(N: _Nodes, requests: list[dict], ranks: list[int], by_label: dict, leaf_of: dict, member_of: dict, recorded: dict) -> dict[str, Any]:
    """Every Call of every (request, rank) Program in exactly one leaf; each request's members sum to its Program; its Calls' gates
    equal its address-map span; each member equals the node and gate range the chronology records for it."""
    checks: dict[str, Any] = {"partition": {}, "members": {}, "gates": {}}
    part_ok = True
    for rq in requests:
        for rank in ranks:
            lf = leaf_of[(rq["request_id"], rank)]
            part_ok &= not (lf < 0).any()
            checks["partition"][f"{rq['request_id']}/rank{rank}"] = {"calls": int(lf.size), "unplaced": int((lf < 0).sum())}
    leaves = [n for n in N.nodes if n["kind"] == "rank"]
    total_calls = sum(by_label[rq["label"]][rank].n for rq in requests for rank in ranks)
    checks["partition"].update(ok=part_ok and sum(n["calls"] for n in leaves) == total_calls, leaves=len(leaves), calls=total_calls)
    for rq in requests:
        p = by_label[rq["label"]][ranks[0]]
        mems = [N.nodes[int(v[1:])] for (rid, _s), v in member_of.items() if rid == rq["request_id"]]
        checks["members"][rq["request_id"]] = {"steps": len(mems), "samplers": len(p.samplers), "sampler_rule": p.sampler_rule,
                                               "calls": sum(m["calls"] for m in mems), "program_calls": p.n * len(ranks),
                                               "ok": sum(m["calls"] for m in mems) == p.n * len(ranks), **p.checks}
        if rq["gates"] is not None:
            span = rq["gates"][1] - rq["gates"][0]
            checks["gates"][rq["request_id"]] = {"address_map_span": span, "calls_gates": int(p.gates.sum()), "ok": span == int(p.gates.sum())}
    agree, differ, absent = 0, [], 0
    for (rid, s), nid in member_of.items():
        nodes_, gates_ = recorded.get((rid, s), (None, None))
        if not nodes_:
            absent += 1
            continue
        mem = N.nodes[int(nid[1:])]
        if mem["call_runs"] == [list(nodes_)] and (gates_ is None or mem.get("gate_range") == list(gates_)):
            agree += 1
        else:
            differ.append(f"{rid}/{s}")
    checks["chronology_ranges"] = {"agree": agree, "differ": differ, "not_recorded": absent, "ok": not differ}
    checks["ok"] = (checks["partition"]["ok"] and all(v["ok"] for v in checks["members"].values())
                    and all(v["ok"] for v in checks["gates"].values()) and checks["chronology_ranges"]["ok"])
    return checks


KINDS = ("flow", "token", "state", "backward")


def _edges(N: _Nodes, by_label: dict, requests: list[dict], ranks: list[int], leaf_of: dict) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Leaf-to-leaf edges: `flow` inside one request step and rank, `token` from a sampler to the next step, `state` from an earlier step
    (the KV cache), `collective` between ranks; `backward` (a read from a later step) must not occur.  Returns (edges, checks)."""
    agg: dict[tuple[str, int, int], list[int]] = {}
    chk: dict[str, Any] = {"backward_reads": 0, "collective_word_mismatch": 0}
    for rq in requests:
        for rank in ranks:
            p = by_label[rq["label"]][rank]
            lf = leaf_of[(rq["request_id"], rank)]
            cols = p.cols
            is_s = np.zeros(p.n, dtype=np.int64)
            is_s[p.samplers] = 1
            off = np.frombuffer(cols.read_off, dtype=np.int64)
            prod = np.frombuffer(cols.read_call, dtype=np.int32).astype(np.int64)
            cons = np.repeat(np.arange(p.n, dtype=np.int64), np.diff(off))
            m = prod >= 0
            prod, cons = prod[m], cons[m]
            kind = _kind(p.step[prod], p.step[cons], is_s[prod])
            code = (kind << 60) | (lf[prod] << 30) | lf[cons]
            u, c = np.unique(code[lf[prod] != lf[cons]], return_counts=True)
            wa = np.array([a for a, _b in p.call_words], dtype=np.int64)
            wb = np.array([b for _a, b in p.call_words], dtype=np.int64)
            wv = np.array(list(p.call_words.values()), dtype=np.int64)
            wcode = (_kind(p.step[wa], p.step[wb], is_s[wa]) << 60) | (lf[wa] << 30) | lf[wb]
            words = Counter()
            for k, w in zip(wcode.tolist(), wv.tolist()):
                words[k] += w
            for k, n_reads in zip(u.tolist(), c.tolist()):
                e = agg.setdefault((KINDS[k >> 60], (k >> 30) & ((1 << 30) - 1), k & ((1 << 30) - 1)), [0, 0])
                e[0] += n_reads
                e[1] += words.get(k, 0)
            chk["backward_reads"] += int((kind == 3).sum())
        if len(ranks) > 1:
            chk["collective_word_mismatch"] += _collectives(by_label, rq, ranks, leaf_of, agg)
    out = [{"kind": kind, "from": N.nodes[a]["id"], "to": N.nodes[b]["id"], "reads": r, "words": w}
           for (kind, a, b), (r, w) in sorted(agg.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0]))]
    chk["by_kind"] = {k: {"edges": sum(1 for e in out if e["kind"] == k), "reads": sum(e["reads"] for e in out if e["kind"] == k),
                          "words": sum(e["words"] for e in out if e["kind"] == k)} for k in (*KINDS, "collective")}
    chk["ok"] = chk["backward_reads"] == 0 and chk["collective_word_mismatch"] == 0
    return out, chk


def _kind(sa: np.ndarray, sb: np.ndarray, from_sampler: np.ndarray) -> np.ndarray:
    return np.where(sa == sb, 0, np.where((from_sampler == 1) & (sb == sa + 1), 1, np.where(sb > sa, 2, 3))).astype(np.int64)


def _collectives(by_label: dict, rq: dict, ranks: list[int], leaf_of: dict, agg: dict) -> int:
    """Rank b's Calls feeding its Call that reads `peers_<k>[lo:hi]` -> rank a's Call reading `peers_<k>[lo:hi]` (TP2: b = the other
    rank), with the words b's operand runs name there; returns the rank-a Calls whose peer words differ from the words fed on rank b."""
    if len(ranks) != 2:
        raise ValueError(f"collective edges are stated for two ranks, the Build has {len(ranks)}")
    bad = 0
    a_rank, b_rank = ranks
    for a, b in ((a_rank, b_rank), (b_rank, a_rank)):
        pa, pb = by_label[rq["label"]][a], by_label[rq["label"]][b]
        fed: dict[int, list[tuple[int, int]]] = {}
        targets = set(pb.peer_runs.values())
        for (src, dst), w in pb.call_words.items():
            if dst in targets:
                fed.setdefault(dst, []).append((src, w))
        got: Counter = Counter()
        want: Counter = Counter()
        pairs = set()
        for key3, ca in pa.peer_runs.items():
            cb = pb.peer_runs.get(key3)
            if cb is None:
                raise ValueError(f"{rq['request_id']}: rank {a} reads peers_{key3[0]}[{key3[1]}:{key3[2]}], rank {b} has no Call reading it")
            if pa.spec[ca] != pb.spec[cb] or int(pa.step[ca]) != int(pb.step[cb]):
                raise ValueError(f"{rq['request_id']}: peers_{key3[0]}[{key3[1]}:{key3[2]}] is read by {pa.spec[ca]} (step {pa.step[ca]}) on rank {a} "
                                 f"and by {pb.spec[cb]} (step {pb.step[cb]}) on rank {b}")
            want[ca] += key3[2] - key3[1]
            pairs.add((cb, ca))
        for cb, ca in pairs:
            for src, ws in fed.get(cb, []):
                e = agg.setdefault(("collective", int(leaf_of[(rq["request_id"], b)][src]), int(leaf_of[(rq["request_id"], a)][ca])), [0, 0])
                e[0] += 1
                e[1] += ws
                got[ca] += ws
        bad += sum(1 for ca in want if got[ca] != want[ca])
    return bad


def _graph_check(graph: dict[str, Any], acc: PG._Acc, gid: dict, N: _Nodes) -> dict[str, Any]:
    """The view's group ids against the row's program.json: the same (module, Definition, Calls) per id; Q_word totals."""
    gg = {x["id"]: x for x in graph["groups"]}
    diff = []
    for key, g in gid.items():
        x = gg.get(g)
        mine = acc.groups[key]
        if x is None or (x["module"], x["definition"]) != (key[0], mine["family"]):
            diff.append(g)
    view_units = sum(n.get("units", 0) for n in N.nodes if n["kind"] == "rank")
    view_words = sum(n.get("committed_interior_words", 0) for n in N.nodes if n["kind"] == "rank")
    q = graph.get("q_word_v1") or {}
    return {"groups": len(gid), "graph_groups": len(gg), "differ": diff[:20], "differ_n": len(diff),
            "q_word_units": {"view": view_units, "graph": q.get("units")}, "q_word_committed_interior_words": {"view": view_words, "graph": q.get("committed_interior_words")}}


RULES = {
    "engine_step": "the Build's chronology (build_workload/chronology.json): members (request id, request step s, phase) per engine step e; "
                   "without one, the request steps (arrival not recorded); where the chronology records a member's node and gate range, the "
                   "member's Calls must be exactly that range (checks.chronology_ranges)",
    "request_step": "per Call: the first step s whose sampler needs it (the sampler or an ancestor of it); a Call no sampler needs: max over the "
                    "Calls it reads of their step, +1 past a sampler; the sampler of step s: the Call producing element s of the request Program's "
                    "return (sampler_rule 'return'), else the s-th Call of a sampling-event Definition (sampler_rule 'sampling-event')",
    "sampler": "the step's sampler Call and its constants (Calls that read no Call and are read by the sampler alone); inputs = the request inputs "
               "it reads as [lo, hi) leaf runs (splits: the one leaf s)",
    "module": "Correspondence.module_of (the recorded owning module); parent = the nearest enclosing module path (torch named_modules nesting)",
    "rank": "a leaf: one rank's Calls of one module body (or the sampler) at one request step; the leaves partition every request Program's Calls",
    "totals": "calls, gates, units, committed_interior_words of an inner node = the sum over its subtree (body_calls: a module's own Calls)",
    "gates": "a Call's specialization's gate count (SpecializedDefinition.gates); a request's Calls tile its address-map range in body order, "
             "so gate_range = the member's Calls when they are one run",
    "units, committed_interior_words": "Q_word_v1 per Call (query.word.unit_rule of its specialization), summed",
    "groups": "program_graph's group ids (the same accumulator, the same Programs and order as program_graphs.py: rank 0 on a TP row)",
    "subtree_hash": "sha256 over the module's body (its Calls' shapes with counts: Definition family, the statics constant over the Program's "
                    "Calls of that family in that module, the names of the others) and its module children's subtree hashes in order; "
                    "fold_of = the first sibling with the same hash",
    "structure_hash": "a member's model subtree hash and its sampler's Definition: equal members have the same structure (decode steps differ only in T)",
    "edges": "Values read between leaves: flow (same member, same rank), token (a sampler's output read at the next step), state (read at a later "
             "step: the KV cache), collective (TP: the Calls rank b feeds its Call reading peers_<k>[lo:hi] with -> rank a's Call reading peers_<k>[lo:hi]); "
             "words = the leaves the operand runs name",
}


def site_tree(view: dict[str, Any]) -> dict[str, Any]:
    """The docs-site tree: the view's nodes nested, engine steps collapsed but for EXPANDED, a collapsed step carrying only its
    summary and `same_structure_as` (the first expanded step whose members have the same structure hashes), folded module siblings
    as one entry with `repeat` (their labels and node ids)."""
    nodes = {n["id"]: n for n in view["nodes"]}
    keep = ("id", "kind", "label", "calls", "body_calls", "gates", "units", "committed_interior_words", "e", "request_id", "s", "phase", "path", "spec",
            "rank", "gate_range", "groups", "group", "co_scheduled", "mixed_prefill_decode", "inputs", "with_calls")

    def nest(nid: str) -> dict[str, Any]:
        n = nodes[nid]
        out = {k: n[k] for k in keep if k in n and n[k] is not None}
        kids, folded = [], {}
        for c in n["children"]:
            cn = nodes[c]
            if cn.get("fold_of"):
                folded.setdefault(cn["fold_of"], []).append(c)
                continue
            kids.append(c)
        out["children"] = []
        for c in kids:
            sub = nest(c)
            if c in folded:
                sub["repeat"] = [{"label": nodes[x]["label"], "id": x} for x in folded[c]]
            out["children"].append(sub)
        return out

    root = nodes["n0"]
    tree = {k: root[k] for k in ("id", "kind", "label", "workload_digest", "gates_total", "requests", "ranks", *FIELDS) if k in root}
    tree["children"] = []
    shapes: dict[tuple, int] = {}
    for c in root["children"]:
        st = nodes[c]
        mems = tuple(nodes[m].get("structure_hash") for m in st["children"])
        if not st["collapsed"]:
            tree["children"].append(nest(c))
            shapes.setdefault(mems, st["e"])
            continue
        entry = {k: st[k] for k in ("id", "kind", "label", "e", "co_scheduled", "mixed_prefill_decode") if st.get(k) is not None}
        entry.update(collapsed=True, calls=sum(nodes[m]["calls"] for m in st["children"]), gates=sum(nodes[m]["gates"] for m in st["children"]),
                     members=[{k: nodes[m][k] for k in ("id", "request_id", "s", "phase", "calls", "gates", "gate_range") if k in nodes[m]}
                              for m in st["children"]])
        if mems in shapes:
            entry["same_structure_as"] = shapes[mems]
        tree["children"].append(entry)
    return {"schema": SCHEMA + "/site", "row": view["row"], "build": {k: v for k, v in view["build"].items() if k != "dir"},
            "expanded": list(EXPANDED), "tree": tree, "checks": {"ok": view["checks"]["ok"]}}


@dataclass(frozen=True, kw_only=True)
class Options:
    """write a served row's serving view (engine steps, requests, model runner, TP ranks) from its Build"""

    out: str = option("out", help="the view JSON to write")
    build: str = option("build", help="the row's Build directory (build_workload/, build_request*/, rank<r>/)")
    site: str | None = option("--site", default=None, help="also write the docs-site tree here")
    graph: str | None = option("--graph", default=None, help="the row's program.json: cross-check the group ids and Q_word totals")
    row: str | None = option("--row", default=None, help="JSON object: the row block (row number, key, record, note)")
    word: str = option("--word", default="16/32", help="Q_word_v1{X,W} for the per-node units and committed words")


def main(a: Options) -> int:
    X, W, _extras = word_query.parse(a.word)
    v = serving_view(a.build, row=json.loads(a.row) if a.row else None, graph=json.loads(Path(a.graph).read_text()) if a.graph else None, X=X, W=W)
    Path(a.out).write_text(json.dumps(v, separators=(",", ":"), default=str) + "\n")
    if a.site:
        Path(a.site).write_text(json.dumps(site_tree(v), separators=(",", ":"), default=str) + "\n")
    print(json.dumps({"out": a.out, "nodes": len(v["nodes"]), "edges": len(v["edges"]), "checks_ok": v["checks"]["ok"]}))
    return 0
