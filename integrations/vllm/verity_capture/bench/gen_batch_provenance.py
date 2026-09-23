"""GEN batch lane — request-level provenance of the derived Program (torch-free; small models only).

Folds a run's log with `verity_capture.fold`, then answers, per request at step 0:
  1. whose instances feed the first attention row's K/V cache operands (own rows, another request's rows);
  2. which prompt positions (of which `prompt_i` parameter) are upstream of the request's sampled token;
  3. which prompt positions are never consumed by any instance (declared inputs the Program does not read).

Evidence for the prefix-caching mechanism note (cross-request KV state) and the concurrency isolation claim.
Laptop-safe for B0 (SmolLM2-135M: ~0.5 GB RSS, ~10 s); do not run on >= 1B logs on the laptop.

    uv run python -m verity_capture.bench.gen_batch_provenance --run-dir out/gen/lanes/batch/runs/b0_prefix64_2req \
        --profile vllm_d9105ea80_sm89_eager --out out/gen/lanes/batch/runs/b0_prefix64_2req/provenance.md
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

from verity.ir.refs import Affine, Concat, Explicit, Strided

from ..compare import load_profile
from ..events import RunHeader, StepStart
from ..fold import fold
from ..log import read_log


def _sources(refs, prompt_idx: set[int], leaves: bool):
    """(space, idx) -> leaf offsets (prompt parameters only, when `leaves`); descriptor-level, no weight enumeration."""
    out: dict[tuple[str, int], set[int]] = collections.defaultdict(set)

    def want(sp, idx):
        return leaves and sp == "p" and idx in prompt_idx

    def walk(r):
        if isinstance(r, Concat):
            for p in r.parts:
                walk(p)
        elif isinstance(r, Affine):
            out[(r.space, r.idx)].update(range(r.base, r.base + r.count) if want(r.space, r.idx) else ())
        elif isinstance(r, Strided):
            out[(r.space, r.idx)].update((r[j][2] for j in range(r.count)) if want(r.space, r.idx) else ())
        elif isinstance(r, Explicit):
            for it in r.items:
                out[(it[0], it[1])].update([it[2]] if want(it[0], it[1]) else ())
        else:
            it = r[0]
            out[(it[0], it[1])].update((r[j][2] for j in range(r.count)) if want(it[0], it[1]) else ())

    walk(refs)
    return out


def analyze(log: Path, profile_id: str, step: int = 0) -> dict:
    """`log` is a raw observation log (or a run directory holding `capture/log.jsonl.gz`)."""
    log = Path(log)
    if log.is_dir():
        log = log / "capture" / "log.jsonl.gz"
    events = list(read_log(log))
    hdr = next(e for e in events if isinstance(e, RunHeader))
    req_index = {r["request_id"]: i for i, r in enumerate(hdr.workload["requests"])}
    s0 = next(e for e in events if isinstance(e, StepStart) and e.step == step)
    rows = s0.rows
    res = fold(events, load_profile(profile_id), strict=False, with_program=True)
    nodes = res.body.nodes
    pnames = [n for n, _ in res.params]
    prompt_idx = {i for i, n in enumerate(pnames) if n.startswith("prompt")}
    meta_by_node = {m["node"]: m for m in res.instances}
    cache: dict[int, dict] = {}

    def srcs(j: int) -> dict:
        if j not in cache:
            d: dict = collections.defaultdict(set)
            for arg in nodes[j].args:
                for key, lv in _sources(arg.refs, prompt_idx, True).items():
                    d[key].update(lv)
            cache[j] = d
        return cache[j]

    def row_req(m) -> tuple[str, int] | None:
        """(request, position) of an instance whose compute spans the whole token batch (row index = batch row)."""
        return rows[m["row"]] if m["step"] == step and m["rows"] == len(rows) else None

    def q_row_req(m) -> tuple[str, int] | None:
        """(request, position) of an attention instance via its `q` operand's producer (attention computes are grouped by T)."""
        for (sp, idx) in _sources(nodes[m["node"]].args[0].refs, prompt_idx, False):
            if sp == "n" and idx in meta_by_node:
                rr = row_req(meta_by_node[idx])
                if rr is not None:
                    return rr
        return None

    out = {"run": str(log), "step": step, "requests": [{"index": req_index[r["request_id"]], "request_id": r["request_id"],
                                                             "prompt_len": len(r["prompt_token_ids"])} for r in hdr.workload["requests"]],
           "block_tables": {f"req{req_index[r]}": bt for r, bt in s0.block_tables.items()}, "block_size": s0.block_size,
           "num_computed": {f"req{req_index[r]}": v for r, v in s0.num_computed.items()},
           "rows": {f"req{req_index[r]}": [min(p for rr, p in rows if rr == r), max(p for rr, p in rows if rr == r)] for r in dict.fromkeys(r for r, _ in rows)},
           "fold": {"instances": len(res.instances), "unresolved": res.unresolved, "errors": len(res.errors)},
           "attention_layer0": [], "sampled_token_upstream": [], "prompt_consumption": []}

    # 1. K/V cache operands of each request's first attention row at layer 0
    att = [m for m in res.instances if m["spec"].startswith("Attention_v3") and m["step"] == step and re.search(r"layers\.0\.", m["hint"])]
    first_by_req: dict[str, tuple[int, dict]] = {}
    for m in att:
        rr = q_row_req(m)
        if rr is None:
            continue
        if rr[0] not in first_by_req or rr[1] < first_by_req[rr[0]][0]:
            first_by_req[rr[0]] = (rr[1], m)
    for rid, (position, m) in sorted(first_by_req.items(), key=lambda kv: req_index[kv[0]]):
        node = nodes[m["node"]]
        params, _ = node.fn.signature()
        ops = []
        for (arg, (pn, _pt)) in zip(node.args, params):
            by = collections.defaultdict(set)
            for (sp, idx) in _sources(arg.refs, prompt_idx, False):
                if sp == "n":
                    mm = meta_by_node.get(idx)
                    rr = row_req(mm) if mm else None
                    if rr is not None:
                        by[(f"req{req_index[rr[0]]}", mm["spec"].split("{")[0])].add(rr[1])
                    else:
                        by[("other", str(idx))].add(-1)
                else:
                    by[("param", pnames[idx])].add(-1)
            ops.append({"operand": pn, "from": {f"{k[0]}:{k[1]}": (f"{min(v)}..{max(v)}" if min(v) >= 0 else "-") + f" ({len(v)})" for k, v in by.items()}})
        out["attention_layer0"].append({"request": f"req{req_index[rid]}", "row": m["row"], "position": position, "spec": m["spec"], "operands": ops})

    # 2. prompt positions upstream of each sampled token at this step
    lr = s0.logits_rows
    for m in [m for m in res.instances if m["spec"].startswith("TokenSelect") and m["step"] == step]:
        seen: set[int] = set()
        stack = [m["node"]]
        pos: dict[str, set[int]] = collections.defaultdict(set)
        while stack:
            j = stack.pop()
            if j in seen:
                continue
            seen.add(j)
            for (sp, idx), lv in srcs(j).items():
                if sp == "n":
                    if idx not in seen:
                        stack.append(idx)
                elif idx in prompt_idx:
                    pos[pnames[idx]].update(lv)
        who = f"req{req_index[lr[m['row']][0]]}@{lr[m['row']][1]}" if m["rows"] == len(lr) else "?"
        out["sampled_token_upstream"].append({"logits_row": m["row"], "request": who, "upstream_nodes": len(seen),
                                              "prompt_positions": {k: f"{min(v)}..{max(v)} ({len(v)})" for k, v in sorted(pos.items())}})

    # 3. declared prompt positions never consumed
    consumed: dict[str, set[int]] = collections.defaultdict(set)
    for j in range(len(nodes)):
        for (sp, idx), lv in srcs(j).items():
            if sp == "p" and idx in prompt_idx:
                consumed[pnames[idx]].update(lv)
    for n_, t in res.params:
        if n_.startswith("prompt"):
            missing = sorted(set(range(t.leaves)) - consumed.get(n_, set()))
            out["prompt_consumption"].append({"param": n_, "declared": t.leaves, "consumed": len(consumed.get(n_, set())),
                                              "never_consumed": (f"{missing[0]}..{missing[-1]} ({len(missing)})" if missing else "none")})
    return out


def render(d: dict) -> str:
    L = [f"# Request-level provenance — step {d['step']} of `{d['run']}`", ""]
    L.append(f"- requests: {[(r['index'], r['request_id'][:8], r['prompt_len']) for r in d['requests']]}")
    L.append(f"- rows at this step (position range per request): {d['rows']}")
    L.append(f"- `num_computed` before this step: {d['num_computed']}; block size {d['block_size']}; block tables: {d['block_tables']}")
    L.append(f"- fold: {d['fold']}")
    L += ["", "## Layer-0 attention of each request's first row: where the K/V cache operands come from", ""]
    for a in d["attention_layer0"]:
        L.append(f"- **{a['request']}** first row (position {a['position']}), `{a['spec']}`")
        for o in a["operands"]:
            L.append(f"  - `{o['operand']}` <- " + ", ".join(f"{k} positions {v}" for k, v in o["from"].items()))
    L += ["", "## Prompt positions upstream of each sampled token (transitive closure over the Program)", ""]
    for s in d["sampled_token_upstream"]:
        L.append(f"- logits row {s['logits_row']} ({s['request']}): {s['upstream_nodes']} upstream instances; " + "; ".join(f"`{k}` {v}" for k, v in s["prompt_positions"].items()))
    L += ["", "## Declared prompt parameters vs. positions actually read", ""]
    for p in d["prompt_consumption"]:
        L.append(f"- `{p['param']}`: {p['consumed']}/{p['declared']} positions consumed; never consumed: {p['never_consumed']}")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True, help="run directory (capture/log.jsonl.gz) or a raw log path")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--out", required=True, help="markdown output path (a .json sibling is written too)")
    a = ap.parse_args(argv)
    d = analyze(Path(a.run_dir), a.profile, a.step)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(d))
    import json
    out.with_suffix(".json").write_text(json.dumps(d, indent=1) + "\n")
    print(out.read_text(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
