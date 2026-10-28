"""The Call boundaries a coarser proof block would keep, from a served Build's request Programs: per Call (what the Commit's Call
boundaries are), per block (a decoder layer's attention block and its MLP block) and per layer, each at one token; per kept boundary
its bytes, and per granularity the bytes and the hashing time they leave at the rows-sha512 floor.  Reads the Build only; measurement
only, for what per-layer proof blocks would cost the proof; nothing here is a scheme.

    python granularity.py BUILD/build_request [BUILD/build_request_LP*_T* ...] [--json OUT.json]
                          (--rate-256, --rate-4k: the floor in GB/s at 256-byte and 4 KiB rows)

A request Program (`descriptor.json.gz`) is its root body: the prescribed inputs, then one Call per op per token row (the attention
Call at one query position).  Each Call's owning module is the Build's runtime correspondence (`call_name_of`; the descriptor's
`node_hints` when the Build has none).  The proof blocks:

  call    every Call its own proof block: every leaf one Call reads of another is kept.
  block   a decoder layer's attention block and its MLP block, at one token.  The attention block is the layer's children (in Program
          order) up to the one whose Calls read other tokens' values (self-attention), so its input norm too; the MLP block is the
          rest, the post-attention norm and the MLP or MoE block.  A Call of the layer's own body is the MLP block's when it reads a
          value of the MLP block or of the attention child (a residual add after attention), else the attention block's.
  layer   a decoder layer at one token: what crosses is the residual stream between layers and the K and V that later tokens'
          attention reads.
  layer-folded  the layer proof block with the residual stream folded: a layer hands the next layer (or the final norm) one hidden vector
          per token, the sum of its output and its residual, instead of the two the next layer's fused add-norm reads; per layer
          and token the larger of what the next layer reads of each producer.  The same values only when every reader of the two
          reads them through their sum in the stream's dtype (`folded_readers` names the Definitions; vLLM's CUDA fused add-norm
          rounds the sum and normalizes that, its native one, which Inductor compiles, normalizes the fp32 sum).

A decoder layer is a ModuleList element: a Call's module path up to its first numeric component.  A module outside every layer (the
embedding, the final norm, the logits, the sampler) is its own proof block at each token.  A Call's token is its embedding Call's: a Call
reading one token's values is that token's, and attention, which reads the K and V of every earlier token, is its latest producer's
(causal).  A token's step is its embedding Call's fx node (one per engine step).  A leaf is kept when a Call of another proof block
reads it, or when the Program returns it; its bytes are its declared width.  Hash ms is bytes over the floor's rate, without per-step padding.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

UNITS = ("call", "block", "layer", "layer-folded")
FOLDED = ("next layer", "out of layers")
NUM = re.compile(r"\d+")


def load(build_dir: str):
    """(calls, names, K, ret runs): the root body's Calls (after its K prescribed-input nodes), each Call's CallName, and the runs of
    the body's return."""
    from verity.ml.circuits.codec import decode_program
    from verity.ml.circuits.program import is_input
    from verity.ml.circuits.refs import runs
    from verity_vllm.correspondence.reader_for_query import call_name_of, parse_call_name
    from verity_vllm.correspondence.runtime import RuntimeCorrespondence
    from verity_vllm.query.program_view import _registry

    with gzip.open(Path(build_dir) / "descriptor.json.gz", "rt") as f:
        desc = json.load(f)
    reg, _errors = _registry()
    body = decode_program(desc, reg).defs[desc["root"]].body
    K = 0
    while K < len(body.nodes) and body.nodes[K].form == "batch" and is_input(body.nodes[K].fn):
        K += 1
    calls = body.nodes[K:]
    ann = desc.get("annotations") or {}
    if ann.get("correspondence"):
        names = [call_name_of(r) for r in RuntimeCorrespondence.records(ann["correspondence"])]
        source = "correspondence"
    else:
        hints = ann.get("node_hints") or {}
        names = [parse_call_name(str(hints.get(str(k), ""))) for k in range(len(calls))]
        source = "node_hints"
    if len(names) != len(calls):
        raise ValueError(f"{build_dir}: {len(names)} named Calls for {len(calls)} Calls")
    return calls, names, K, runs(body.ret), source


def layer_of(module: str) -> tuple[str | None, str]:
    """(the decoder layer, its child) a module path is in: the path up to its first numeric component, and the next component."""
    parts = module.split(".")
    for j, p in enumerate(parts):
        if NUM.fullmatch(p):
            return ".".join(parts[:j + 1]), parts[j + 1] if j + 1 < len(parts) else ""
    return None, ""


def analyze(build_dir: str, rates: dict[int, float]) -> dict:
    from verity.ml.circuits.refs import runs
    from verity_vllm.query.program_view import _spec_id, ports_of_type

    calls, names, K, ret, source = load(build_dir)
    n = len(calls)
    reads = [[(r[1] - K, r[2], r[3]) for a in c.args for r in runs(a.refs) if r[0] == "n" and r[1] >= K] for c in calls]
    family = [_spec_id(c.fn).split("{")[0] for c in calls]
    module = [nm.module for nm in names]
    layer, child = zip(*(layer_of(m) for m in module)) if n else ((), ())

    token = [-1] * n
    step_of_token: list[int] = []
    steps: dict[str, int] = {}
    for i in range(n):
        if family[i].startswith("Embedding"):
            token[i] = len(step_of_token)
            step_of_token.append(steps.setdefault(names[i].node, len(steps)))
        else:
            token[i] = max((token[p] for p, _lo, _hi in reads[i]), default=-1)

    cross = [False] * n                                   # the Call reads another token's value (attention)
    for i in range(n):
        cross[i] = any(token[p] != token[i] for p, _lo, _hi in reads[i] if token[p] >= 0)
    first: dict[tuple[str, str], int] = {}
    attn_child: dict[str, str] = {}
    for i in range(n):
        if layer[i] is not None:
            first.setdefault((layer[i], child[i]), i)
            if cross[i] and child[i]:
                attn_child.setdefault(layer[i], child[i])
    block = [""] * n
    for i in range(n):
        lay = layer[i]
        if lay is None:
            continue
        a = attn_child.get(lay)
        if child[i]:
            block[i] = "attn" if a is None or first[(lay, child[i])] <= first[(lay, a)] else "mlp"
        else:
            block[i] = "mlp" if any(layer[p] == lay and (block[p] == "mlp" or child[p] == a) for p, _lo, _hi in reads[i]) else "attn"

    def unit(g: str, i: int):
        if g == "call":
            return i
        if layer[i] is None:
            return (module[i], token[i])
        return (layer[i], block[i], token[i]) if g == "block" else (layer[i], token[i])

    def kind(p: int, i: int) -> str:
        if layer[p] is None:
            return "into layers" if layer[i] is not None else "outside layers"
        if layer[i] is None:
            return "out of layers"
        if layer[p] != layer[i]:
            return "next layer"
        if token[p] != token[i]:
            return "later token"
        return "other block" if block[p] != block[i] else "within block"

    units = {g: [unit(g, i) for i in range(n)] for g in UNITS}
    kept: dict[str, dict[int, set[tuple[int, int]]]] = {g: defaultdict(set) for g in UNITS}
    why: dict[str, dict[int, set[str]]] = {g: defaultdict(set) for g in UNITS}
    fold: dict[int, set[tuple[int, int]]] = defaultdict(set)     # layer-folded: what the next layer reads, one vector per layer and token
    fold_readers: set[str] = set()
    for i in range(n):
        for p, lo, hi in reads[i]:
            k = kind(p, i)
            for g in UNITS:
                if units[g][p] == units[g][i]:
                    continue
                if g == "layer-folded" and k in FOLDED:
                    fold[p].add((lo, hi))
                    fold_readers.add(family[i])
                else:
                    kept[g][p].add((lo, hi))
                    why[g][p].add(k)
    for r in ret:
        if r[0] == "n":
            for g in UNITS:
                kept[g][r[1] - K].add((r[2], r[3]))
                why[g][r[1] - K].add("program output")

    port_memo: dict[int, list] = {}

    def nbytes(p: int, ivs: set[tuple[int, int]]) -> int:
        tid = id(calls[p].out)
        ports = port_memo.get(tid)
        if ports is None:
            ports = port_memo[tid] = [(q.lo, q.hi, q.width) for q in ports_of_type(calls[p].out)]
        merged: list[list[int]] = []
        for lo, hi in sorted(ivs):
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        bits = 0
        for lo, hi in merged:
            for qlo, qhi, w in ports:
                a, b = max(lo, qlo), min(hi, qhi)
                if a < b:
                    bits += (b - a) * (w if w is not None else max(calls[p].out.leaf_widths()[a:b]))
        return bits // 8

    n_tokens = len(step_of_token)
    prefill = sum(1 for s in step_of_token if s == 0)
    out: dict = {"build": str(build_dir), "names": source, "calls": n, "tokens": n_tokens, "prefill_tokens": prefill,
                 "steps": len(steps), "layers": len({x for x in layer if x is not None}),
                 "attention_child": sorted(set(attn_child.values())), "units": {}}
    for g in UNITS:
        groups: dict[tuple, dict] = {}
        tot = defaultdict(int)
        for p, ivs in kept[g].items():
            b = nbytes(p, ivs)
            ph = "prefill" if token[p] >= 0 and step_of_token[token[p]] == 0 else "decode"
            tot[ph] += b
            mod = re.sub(r"\.\d+(\.|$)", r".*\1", module[p], count=1) if layer[p] is not None else module[p]
            key = (mod, family[p], ",".join(sorted(why[g][p])))
            e = groups.setdefault(key, {"module": mod, "definition": family[p], "read_by": sorted(why[g][p]), "calls": 0, "bytes": 0,
                                        "first": p, "producers": set()})
            e["calls"] += 1
            e["bytes"] += b
            e["producers"].add(p)
        if g == "layer-folded" and fold:
            stream: dict[tuple, tuple[int, int]] = {}
            for p, ivs in fold.items():
                stream[(layer[p], token[p])] = max(stream.get((layer[p], token[p]), (0, p)), (nbytes(p, ivs), p))
            e = groups.setdefault(("stream",), {"module": re.sub(r"\.\d+$", ".*", next(iter(stream))[0]), "definition": "residual sum",
                                                "read_by": sorted({k for p in fold for k in why["layer"][p] if k in FOLDED}),
                                                "calls": 0, "bytes": 0, "first": min(fold), "producers": set()})
            for b, p in stream.values():
                tot["prefill" if token[p] >= 0 and step_of_token[token[p]] == 0 else "decode"] += b
                e["calls"] += 1
                e["bytes"] += b
                e["producers"].add(p)
        total = tot["prefill"] + tot["decode"]
        decode_steps = max(1, len(steps) - 1)
        out["units"][g] = {
            "bytes": total, "prefill_bytes": tot["prefill"], "decode_bytes": tot["decode"],
            "decode_bytes_per_step": tot["decode"] / decode_steps,
            "hash_ms": {str(row): round(total / rate / 1e6, 3) for row, rate in rates.items()},
            "decode_hash_ms_per_step": {str(row): round(tot["decode"] / decode_steps / rate / 1e6, 4) for row, rate in rates.items()},
            "kept": sorted(groups.values(), key=lambda e: e["first"]),
        }
        if g == "layer-folded":
            out["units"][g]["folded_readers"] = sorted(fold_readers)
        for e in out["units"][g]["kept"]:
            e["bytes_per_token"] = round(e["bytes"] / max(1, n_tokens), 1)
            e["layers"] = len({layer[i] for i in e.pop("producers")})
            del e["first"]
    return out


def render(s: dict) -> str:
    lines = [f"## {s['build']}", "",
             f"{s['calls']} Calls, {s['tokens']} tokens ({s['prefill_tokens']} prefill, {s['steps'] - 1} decode steps), {s['layers']} layers; "
             f"modules from {s['names']}; attention child {', '.join(s['attention_child']) or '-'}", ""]
    rows = sorted(next(iter(s["units"].values()))["hash_ms"], key=int)
    base = s["units"]["call"]["bytes"]
    lines += ["| unit | MB | vs call | prefill MB | decode KB / step | " + " | ".join(f"hash ms @ {r} B rows" for r in rows)
              + " | " + " | ".join(f"decode ms / step @ {r} B" for r in rows) + " |",
              "|---|---:|---:|---:|---:|" + "---:|" * (2 * len(rows))]
    for g, u in s["units"].items():
        lines.append(f"| {g} | {u['bytes'] / 1e6:.2f} | {u['bytes'] / base:.3f} | {u['prefill_bytes'] / 1e6:.2f} | "
                     f"{u['decode_bytes_per_step'] / 1e3:.1f} | " + " | ".join(f"{u['hash_ms'][r]:.3f}" for r in rows) + " | "
                     + " | ".join(f"{u['decode_hash_ms_per_step'][r]:.4f}" for r in rows) + " |")
    for g in ("layer", "layer-folded", "block"):
        readers = s["units"][g].get("folded_readers")
        lines += ["", f"{g}: the Call boundaries kept" + (f" (the residual sum's readers: {', '.join(readers)})" if readers else ""), "",
                  "| producer module | Definition | read by | Calls | B / token | layers | B / token / layer |",
                  "|---|---|---|---:|---:|---:|---:|"]
        for e in s["units"][g]["kept"]:
            per = f"{e['bytes_per_token'] / e['layers']:.0f}" if e["layers"] and "*" in e["module"] else ""
            lines.append(f"| {e['module']} | {e['definition']} | {', '.join(e['read_by'])} | {e['calls']} | {e['bytes_per_token']:.0f} | "
                         f"{e['layers'] if '*' in e['module'] else ''} | {per} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("builds", nargs="+", help="request Program dirs of a Build (each holds descriptor.json.gz)")
    ap.add_argument("--json")
    ap.add_argument("--rate-256", type=float, default=66.12, help="GB/s, rows_floor minb2 at 256-byte rows")
    ap.add_argument("--rate-4k", type=float, default=230.61, help="GB/s, rows_floor minb2 at 4 KiB rows")
    a = ap.parse_args()
    res = []
    for b in a.builds:
        s = analyze(b, {256: a.rate_256, 4096: a.rate_4k})
        res.append(s)
        print(render(s), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
