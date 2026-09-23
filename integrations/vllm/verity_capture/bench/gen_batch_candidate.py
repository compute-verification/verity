"""GEN ov-batch lane — request-level provenance from the FROZEN CANDIDATE alone (torch-free; no fold, no log, no patterns).

D85 persists `program.json` / `instances.jsonl` next to `fold_summary.json`; this diagnostic answers the batch lane's mechanism
questions from those files (plus the lane's `attribution.json`, written on the pod from the log's public step facts, and the
workload file), so nothing is folded on the laptop:

  1. per engine step: which request's rows produced the K/V that each request's layer-0 attention reads (by producer request
     AND producer step) — cross-request / cross-step KV aliasing under prefix caching;
  2. the prompt positions (of which `prompt_i` parameter) upstream of every sampled token at step 0;
  3. declared prompt positions never consumed by any instance, and — for a request admitted with `num_computed > 0` — the
     *prefix-alias obligation* `prompt_r[j] == prompt_r'[j]` for the aliased positions, checked against the workload (the
     request record is a frozen root class; the equality is what vLLM's block hash asserts and Verity never observes);
  4. the Program's return: which TokenSelect instances each request's returned `Array(n, I32)` is made of, and whether a returned
     sample was *discarded* by the engine (logits row position < prompt_len - 1, i.e. `seq_len < prompt_len` after the step:
     chunked prefill) — the runtime facts `StepStart.seq_lens` / `logits_rows` are in `attribution.json`.

    uv run python -m verity_capture.bench.gen_batch_candidate --run-dir out/gen/lanes/ov-batch/runs/b0_prefix64_2req \
        --workload workloads/workload_b0_prefix64_2req.json --out out/gen/lanes/ov-batch/runs/b0_prefix64_2req/candidate.md

Laptop cost: B0 prefix64 (38 k instances) ~0.3 GB / 5 s; Qwen conc4 (82 k) ~0.6 GB / 15 s.  DIAGNOSTIC only (RULES §1: not a gate).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import sys
from pathlib import Path

from verity.ir.refs import refs_from_json, runs
from verity.ir.types import type_from_json


def _ranges(ps) -> str:
    ps = sorted(set(ps))
    if not ps:
        return "-"
    out, a, b = [], ps[0], ps[0]
    for p in ps[1:]:
        if p == b + 1:
            b = p
        else:
            out.append(f"{a}" if a == b else f"{a}..{b}")
            a = b = p
    out.append(f"{a}" if a == b else f"{a}..{b}")
    return ",".join(out) + f" ({len(ps)})"


def _open(path: Path):
    """The plain file, or its committed `.gz` twin (the lane commits the frozen candidate gzipped: < 20 MB per run)."""
    if path.exists():
        return path.open("rt")
    if path.with_name(path.name + ".gz").exists():
        return gzip.open(path.with_name(path.name + ".gz"), "rt")
    raise FileNotFoundError(f"{path} (or {path.name}.gz)")


class Candidate:
    """program.json + instances.jsonl (+ attribution.json, workload)."""

    def __init__(self, run_dir: Path, workload: Path | None):
        self.run_dir = run_dir
        with _open(run_dir / "program.json") as f:
            prog = json.load(f)
        self.digest = prog["digest"]
        self.n_input_nodes = int(prog["n_input_nodes"])
        desc = prog["program"]
        self.desc = desc
        root = desc["definitions"][desc["root"]]
        self.root_body = root["body"]
        self.attr = json.loads((run_dir / "attribution.json").read_text()) if (run_dir / "attribution.json").exists() else None
        self.fold = json.loads((run_dir / "fold_summary.json").read_text()) if (run_dir / "fold_summary.json").exists() else None
        # The fold's parameters (`prompt_0, prompt_1, ..., weights`; `("p", k, leaf)` in instances.jsonl) are the root circuit's first
        # `n_input_nodes` nodes (Input32 / Input16 batches) in the same order; names from fold_summary.params, types from the nodes.
        types = desc.get("types")
        in_nodes = self.root_body["nodes"][: self.n_input_nodes]
        names = [n for n, _ in (self.fold or {}).get("params") or []]
        if len(names) != len(in_nodes):
            names = [f"prompt_{k}" if (desc.get("callees", [None] * 999)[nd["fn"]] if isinstance(nd.get("fn"), int) else nd.get("fn")) == "Input32_v1" else f"in{k}"
                     for k, nd in enumerate(in_nodes)]
        self.params = [(names[k], type_from_json(types[nd["out"]] if isinstance(nd["out"], int) else nd["out"])) for k, nd in enumerate(in_nodes)] \
            if in_nodes else [(n, type_from_json(t)) for n, t in self.root_body.get("params") or []]
        self.prompt_params = {i: (n, t.leaves) for i, (n, t) in enumerate(self.params) if n.startswith("prompt")}
        self.insts: list[dict] = []
        with _open(run_dir / "instances.jsonl") as f:
            self.inst_header = json.loads(next(f))
            for line in f:
                self.insts.append(json.loads(line))
        if self.inst_header.get("operand_refs_form") == "progressions":   # [R17 mfast F2] v2 file: node-space progressions -> the run tuples
            from verity_capture.experimental.cb_a.compact import runs_from_file_items
            for m in self.insts:
                if m.get("operand_refs"):
                    m["operand_refs"] = [[list(r) for r in runs_from_file_items(a)] for a in m["operand_refs"]]   # the v1 json shape
        self.by_node = {m["node"]: m for m in self.insts if m.get("node") is not None}
        self._rows_at: dict[int, int] = {}
        for m in self.insts:
            self._rows_at[m["step"]] = max(self._rows_at.get(m["step"], 0), m["rows"])
        self._names: dict[str, list[str]] = {}
        self.workload = json.loads(workload.read_text()) if workload else None
        # request order: the fold names parameters prompt_<k> in RunHeader.workload order == the workload file order
        self.req_ids = [r["request_id"] for r in (self.workload or {}).get("requests", [])] or \
            [r["request_id"] for r in (self.fold or {}).get("requests", [])]
        self.req_index = {r: i for i, r in enumerate(self.req_ids)}
        self.prompt_len = {r["request_id"]: len(r["prompt_token_ids"]) for r in (self.workload or {}).get("requests", [])} or \
            {r["request_id"]: r["prompt_len"] for r in (self.fold or {}).get("requests", [])}
        self._rowmap: dict[int, dict[int, tuple[int, int]]] = {}

    # rows -> (request index, position) ----------------------------------------------------------------------
    def rows_at(self, step: int) -> int:
        return self._rows_at.get(step, 0)

    def row_map(self, step: int) -> dict[int, tuple[int, int]]:
        """row -> (request index, position).  From attribution.json (log facts) when present; the Embedding instances'
        operand refs (`("p", prompt_k, pos)`) give the prompt rows independently and are cross-checked (`row_map_check`)."""
        if step in self._rowmap:
            return self._rowmap[step]
        out: dict[int, tuple[int, int]] = {}
        if self.attr:
            st = next((s for s in self.attr["steps"] if s["step"] == step), None)
            if st:
                for rid, e in st["requests"].items():
                    for i, p in zip(_expand(e["row_index"]), _expand(e["positions"])):
                        out[i] = (self.req_index.get(rid, -1), p)
        if not out:
            out = self.embedding_rows(step)
        self._rowmap[step] = out
        return out

    def embedding_rows(self, step: int) -> dict[int, tuple[int, int]]:
        n = self.rows_at(step)
        out: dict[int, tuple[int, int]] = {}
        for m in self.insts:
            if m["step"] == step and m["spec"].startswith("Embedding_v") and m["rows"] == n:
                for sp, idx, lo, hi in m["operand_refs"][0]:
                    if sp == "p" and idx in self.prompt_params:
                        out[m["row"]] = (list(self.prompt_params).index(idx), lo)
        return out

    def row_map_check(self, step: int) -> dict:
        a, b = self.row_map(step), self.embedding_rows(step)
        common = {i: (a[i], b[i]) for i in b if i in a}
        return {"prompt_rows_from_embedding": len(b), "rows_from_attribution": len(a),
                "agree": sum(1 for x, y in common.values() if x == y), "disagree": [(i, x, y) for i, (x, y) in common.items() if x != y][:8]}

    def who(self, node: int) -> tuple[int, int, int] | None:
        """(request index, position, step) of the row an instance computed, or None for non-row instances."""
        m = self.by_node.get(node)
        if m is None:
            return None
        rm = self.row_map(m["step"])
        if m["rows"] == self.rows_at(m["step"]) and m["row"] in rm:
            r, p = rm[m["row"]]
            return (r, p, m["step"])
        return None

    # per-operand sources -----------------------------------------------------------------------------------
    def operand_names(self, spec: str, n: int) -> list[str]:
        if spec in self._names:
            return self._names[spec]
        d = self.desc["definitions"].get(spec)
        if d is None:
            base = spec.split("{")[0]
            d = next((v for k, v in self.desc["definitions"].items() if k.split("{")[0] == base and v.get("params")), None)
        if d and d.get("params") and len(d["params"]) == n:
            out = [p[0] for p in d["params"]]
        elif spec.startswith("Attention_v") and n == 3:
            out = ["q", "kc", "vc"]            # the profile's Attention_v3 operand order (DESIGN §6); the codec inlines the per-head bodies
        else:
            out = [f"arg{i}" for i in range(n)]
        self._names[spec] = out
        return out

    def sources(self, m: dict, operand: int) -> dict[str, list[int]]:
        """'req<r>@step<s>:<Spec>' -> positions (rows of producer instances), 'param:<name>' -> leaves, 'other' -> nodes."""
        by: dict[str, list[int]] = collections.defaultdict(list)
        for sp, idx, lo, hi in m["operand_refs"][operand]:
            if sp == "n":
                w = self.who(idx)
                pm = self.by_node.get(idx)
                if w is not None:
                    by[f"req{w[0]}@step{w[2]}:{pm['spec'].split('{')[0]}"].append(w[1])
                else:
                    by[f"other:{pm['spec'].split('{')[0] if pm else idx}"].append(idx)
            elif sp == "p":
                by[f"param:{self.params[idx][0]}"].extend(range(lo, hi))
            else:
                by[f"{sp}:{idx}"].extend(range(lo, hi))
        return by

    # 1. layer-0 attention K/V sources per request and step ----------------------------------------------------
    def attention_sources(self, layer: int = 0) -> list[dict]:
        out = []
        for m in self.insts:
            if not m["spec"].startswith("Attention_v") or not re.search(rf"layers\.{layer}(\.|/|$)", m.get("hint") or m.get("name") or ""):
                continue
            names = self.operand_names(m["spec"], len(m["operand_refs"]))
            q_src = self.sources(m, 0)
            q_who = next((k for k in q_src if k.startswith("req")), None)
            if q_who is None:
                continue
            r = int(q_who[3:].split("@")[0])
            pos = q_src[q_who][0]
            ops = {names[i]: {k: _ranges(v) for k, v in self.sources(m, i).items()} for i in range(1, len(names))}
            out.append({"step": m["step"], "request": r, "position": pos, "spec": m["spec"], "node": m["node"], "operands": ops})
        out.sort(key=lambda d: (d["step"], d["request"], d["position"]))
        return out

    # 2. prompt positions upstream of a node ------------------------------------------------------------------
    def upstream_prompt_positions(self, node: int) -> tuple[dict[str, set[int]], int]:
        seen: set[int] = set()
        stack = [node]
        pos: dict[str, set[int]] = collections.defaultdict(set)
        while stack:
            j = stack.pop()
            if j in seen:
                continue
            seen.add(j)
            m = self.by_node.get(j)
            if m is None:
                continue
            for refs in m["operand_refs"]:
                for sp, idx, lo, hi in refs:
                    if sp == "n":
                        if idx not in seen:
                            stack.append(idx)
                    elif sp == "p" and idx in self.prompt_params:
                        pos[self.params[idx][0]].update(range(lo, hi))
        return pos, len(seen)

    def token_select(self, step: int) -> list[dict]:
        return sorted((m for m in self.insts if m["spec"].startswith("TokenSelect_v") and m["step"] == step), key=lambda m: m["row"])

    def logits_rows(self, step: int) -> list[tuple[str, int]] | None:
        if not self.attr:
            return None
        st = next((s for s in self.attr["steps"] if s["step"] == step), None)
        return [(r, int(p)) for r, p in st["logits_rows"]] if st else None

    # 3. declared vs consumed ---------------------------------------------------------------------------------
    def prompt_consumption(self) -> list[dict]:
        consumed: dict[int, set[int]] = collections.defaultdict(set)
        consumers: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
        for m in self.insts:
            for refs in m["operand_refs"]:
                for sp, idx, lo, hi in refs:
                    if sp == "p" and idx in self.prompt_params:
                        consumed[idx].update(range(lo, hi))
                        consumers[idx][m["spec"].split("{")[0]] += 1
        out = []
        for idx, (name, leaves) in self.prompt_params.items():
            c = consumed.get(idx, set())
            out.append({"param": name, "declared": leaves, "consumed": len(c), "never_consumed": _ranges(set(range(leaves)) - c),
                        "consumers": dict(consumers[idx])})
        return out

    def admitted_with_cache(self) -> list[dict]:
        """Requests whose first step has num_computed > 0 (from attribution.json), with the alias obligation checked."""
        if not self.attr:
            return []
        out = []
        first: dict[str, dict] = {}
        for st in self.attr["steps"]:
            for rid, e in st["requests"].items():
                if rid not in first:
                    first[rid] = {"step": st["step"], **e}
        att0 = self.attention_sources(0)
        for rid, e in first.items():
            c = int(e.get("num_computed") or 0)
            if c <= 0:
                continue
            r = self.req_index[rid]
            # producer request(s) of the K over positions [0, c) at this request's first attention row
            row = next((a for a in att0 if a["step"] == e["step"] and a["request"] == r), None)
            producers = sorted({k.split("@")[0] for op in (row or {}).get("operands", {}).values() for k in op if k.startswith("req") and not k.startswith(f"req{r}@")})
            obligation = []
            for pr in producers:
                r2 = int(pr[3:])
                if self.workload:
                    a = self.workload["requests"][r]["prompt_token_ids"][:c]
                    b = self.workload["requests"][r2]["prompt_token_ids"][:c]
                    obligation.append({"alias": f"prompt_{r}[0..{c - 1}] == prompt_{r2}[0..{c - 1}]", "holds_in_workload": a == b,
                                       "first_difference": next((j for j in range(c) if a[j] != b[j]), None)})
            out.append({"request": r, "request_id": rid, "first_step": e["step"], "num_computed": c, "prompt_len": self.prompt_len.get(rid),
                        "kv_producers_over_cached_prefix": producers, "obligation": obligation})
        return out

    # 4. the return ------------------------------------------------------------------------------------------
    def ret_runs(self) -> list[tuple[str, int, int, int]]:
        return _ret_runs(self.root_body, self.root_body["ret"])

    def returned_tokens(self) -> list[dict]:
        rr = self.ret_runs()
        nodes = []
        for sp, idx, lo, hi in rr:
            if sp != "n":
                nodes.append(("non-node", sp, idx, lo, hi))
                continue
            k = idx - self.n_input_nodes
            m = self.by_node.get(k)
            nodes.append(("node", k, m))
        # group by request via the TokenSelect instance's logits row (attribution) -- in step order
        per_req: dict[int, list[dict]] = collections.defaultdict(list)
        unknown = []
        consumers = collections.Counter()
        for m in self.insts:
            for refs in m["operand_refs"]:
                for sp, idx, lo, hi in refs:
                    if sp == "n":
                        consumers[idx] += 1
        for kind, *rest in nodes:
            if kind != "node" or rest[1] is None:
                unknown.append(rest)
                continue
            k, m = rest
            lr = self.logits_rows(m["step"])
            if lr is None or m["row"] >= len(lr):
                unknown.append((k, m["step"], m["row"]))
                continue
            rid, pos = lr[m["row"]]
            lp = self.prompt_len.get(rid)
            tok = None
            if self.attr:
                st = next((s for s in self.attr["steps"] if s["step"] == m["step"]), None)
                if st and m["row"] < len(st.get("sampled") or []):
                    tok = st["sampled"][m["row"]]
            per_req[self.req_index.get(rid, -1)].append({"step": m["step"], "logits_row": m["row"], "position": pos, "token": tok, "node": k,
                                                         "consumed_by_instances": consumers.get(k, 0),
                                                         "committed": (pos >= lp - 1) if lp is not None else None})
        out = []
        for r in sorted(per_req):
            seq = per_req[r]
            out.append({"request": r, "returned": len(seq), "committed": sum(1 for s in seq if s["committed"]),
                        "discarded": [s for s in seq if s["committed"] is False], "max_tokens": (self.workload or {}).get("requests", [{}] * (r + 1))[r].get("max_tokens") if self.workload else None,
                        "steps": [s["step"] for s in seq]})
        return out + ([{"unknown": unknown}] if unknown else [])


def _expand(rng: str) -> list[int]:
    out: list[int] = []
    for part in str(rng).split(","):
        if part in ("", "-"):
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _ret_runs(body: dict, j) -> list[tuple[str, int, int, int]]:
    """Runs of a root-body refs JSON (codec v1: `{"alias": [k, a]}` = node k's argument a; `{"concat": [...]}`)."""
    if isinstance(j, dict) and "alias" in j:
        k, a = j["alias"]
        return _ret_runs(body, body["nodes"][k]["args"][a]["refs"])
    if isinstance(j, dict) and "concat" in j:
        out: list[tuple[str, int, int, int]] = []
        for x in j["concat"]:
            out.extend(_ret_runs(body, x))
        return out
    return [tuple(r) for r in runs(refs_from_json(j))]


def analyze(run_dir: Path, workload: Path | None, layer: int = 0) -> dict:
    c = Candidate(run_dir, workload)
    d: dict = {"run_dir": str(run_dir), "digest": c.digest, "n_instances": len(c.insts), "n_input_nodes": c.n_input_nodes,
               "params": [(n, repr(t)) for n, t in c.params], "requests": [(i, r[:8], c.prompt_len.get(r)) for i, r in enumerate(c.req_ids)],
               "attribution_present": c.attr is not None, "row_map_check_step0": c.row_map_check(0)}
    att = c.attention_sources(layer)
    # first row per (step, request)
    first: dict[tuple[int, int], dict] = {}
    for a in att:
        key = (a["step"], a["request"])
        if key not in first or a["position"] < first[key]["position"]:
            first[key] = a
    d["attention_first_row"] = [first[k] for k in sorted(first)]
    # cross-request / cross-step summary over ALL layer-`layer` attention instances: producer (request, step) of K per (step, reader request)
    summary: dict[tuple[int, int], collections.Counter] = collections.defaultdict(collections.Counter)
    for a in att:
        for op in a["operands"].values():
            for k in op:
                if k.startswith("req"):
                    summary[(a["step"], a["request"])][k.split(":")[0]] += 1
    d["kv_sources_by_step"] = [{"step": s, "reader": r, "producers": dict(sorted(v.items()))} for (s, r), v in sorted(summary.items())]
    d["sampled_token_upstream_step0"] = []
    lr0 = c.logits_rows(0)
    for m in c.token_select(0):
        pos, n = c.upstream_prompt_positions(m["node"])
        who = f"req{c.req_index.get(lr0[m['row']][0], -1)}@{lr0[m['row']][1]}" if lr0 and m["row"] < len(lr0) else "?"
        d["sampled_token_upstream_step0"].append({"logits_row": m["row"], "request": who, "upstream_instances": n,
                                                  "prompt_positions": {k: _ranges(v) for k, v in sorted(pos.items())}})
    d["prompt_consumption"] = c.prompt_consumption()
    d["admitted_with_cache"] = c.admitted_with_cache()
    d["returned"] = c.returned_tokens()
    d["ret_type"] = (c.fold or {}).get("ret_type")
    if c.fold:
        d["fold_requests"] = c.fold.get("requests")
    return d


def render(d: dict) -> str:
    L = [f"# Candidate provenance — `{d['run_dir']}`", "", f"- Program digest `{d['digest'][:16]}…`, {d['n_instances']} instances, "
         f"{d['n_input_nodes']} input nodes; parameters: {', '.join(f'`{n}: {t}`' for n, t in d['params'] if not n.startswith('weights'))}",
         f"- requests (index, id, prompt_len): {d['requests']}; attribution.json present: {d['attribution_present']}; "
         f"step-0 row map cross-check (attribution vs Embedding operand refs): {d['row_map_check_step0']}", ""]
    L += ["## 1. Layer-0 attention, first row of each request at each step: where q / K / V come from", ""]
    for a in d["attention_first_row"]:
        L.append(f"- step {a['step']} **req{a['request']}** position {a['position']} `{a['spec']}`")
        for name, src in a["operands"].items():
            L.append(f"  - `{name}` <- " + "; ".join(f"{k} positions {v}" for k, v in src.items()))
    L += ["", "### K/V producers of every layer-0 attention row, by (step, reader) -> producer request@step (instance counts)", ""]
    for s in d["kv_sources_by_step"]:
        L.append(f"- step {s['step']} reader req{s['reader']}: {s['producers']}")
    L += ["", "## 2. Prompt positions upstream of each sampled token at step 0 (transitive closure over the Program)", ""]
    for s in d["sampled_token_upstream_step0"]:
        L.append(f"- logits row {s['logits_row']} ({s['request']}): {s['upstream_instances']} upstream instances; " + "; ".join(f"`{k}` {v}" for k, v in s["prompt_positions"].items()))
    L += ["", "## 3. Declared prompt parameters vs positions actually read", ""]
    for p in d["prompt_consumption"]:
        L.append(f"- `{p['param']}`: {p['consumed']}/{p['declared']} consumed; never consumed: {p['never_consumed']}; consumers: {p['consumers']}")
    if d["admitted_with_cache"]:
        L += ["", "### Requests admitted with `num_computed > 0` — the prefix-alias obligation", ""]
        for a in d["admitted_with_cache"]:
            L.append(f"- req{a['request']} (`{a['request_id'][:8]}`) first step {a['first_step']}: num_computed {a['num_computed']} of prompt_len {a['prompt_len']}; "
                     f"K/V over the cached prefix produced by {a['kv_producers_over_cached_prefix']}; obligation: {a['obligation']}")
    L += ["", "## 4. The Program's return: TokenSelect instances per request, committed vs discarded", ""]
    L.append(f"- fold `ret_type`: `{d.get('ret_type')}`; fold requests: {d.get('fold_requests')}")
    for r in d["returned"]:
        if "unknown" in r:
            L.append(f"- unattributed return leaves: {r['unknown'][:8]}")
            continue
        L.append(f"- req{r['request']}: {r['returned']} returned leaves, {r['committed']} committed (position >= prompt_len-1), max_tokens {r['max_tokens']}; steps {_ranges(r['steps'])}")
        for s in r["discarded"]:
            L.append(f"  - DISCARDED by the engine but returned by the Program: step {s['step']} logits row {s['logits_row']} position {s['position']} token {s['token']} "
                     f"(TokenSelect node {s['node']}, consumed by {s['consumed_by_instances']} instances)")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True, help="directory with program.json, instances.jsonl [, attribution.json, fold_summary.json]")
    ap.add_argument("--workload", default=None, help="the workload file (request order, prompt ids for the alias obligation)")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", required=True, help="markdown output (a .json sibling is written too)")
    a = ap.parse_args(argv)
    d = analyze(Path(a.run_dir), Path(a.workload) if a.workload else None, a.layer)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(d))
    out.with_suffix(".json").write_text(json.dumps(d, indent=1, default=str) + "\n")
    print(out.read_text(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
