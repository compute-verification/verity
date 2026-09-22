"""program_diff — did the Program *structure* stay the same between runs?  (coverage campaign D88 §7, lane cov-moe)

    python -m verity_capture.bench.program_diff RUN_DIR [RUN_DIR ...] [--out diff.json]

Reads each run's `program.json` (the persisted Verity Program, `verity.ir.codec` v1 descriptor) and reports, per run: the
digest, the Definition set (primitive ids, composite specialization ids with their statics), the root's instance census
(instances per specialization = the root body's callee histogram) and the root's own statics; then pairwise: definitions
only in one run, common specialization ids whose encoding (statics or body) differs, root statics that differ, and the
census delta.  Runs on the pod next to the run dirs — program.json is never pulled to the laptop.  Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def canonical(obj, callees: list, types: list):
    """A definition's encoding with the descriptor's table indices resolved: a body node's `fn` (index into `callees`) becomes
    the callee's id, and `type` / `out` (indices into `types`) become the type descriptor itself.  Two runs' tables are
    numbered differently as soon as their Definition sets differ (one more Attention_v3{T=…} shifts every later index), so
    the raw JSON of the same Definition differs between runs while the Definition does not."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "fn" and isinstance(v, int) and v < len(callees):
                out[k] = callees[v]
            elif k in ("type", "out") and isinstance(v, int) and v < len(types):
                out[k] = types[v]
            else:
                out[k] = canonical(v, callees, types)
        return out
    if isinstance(obj, list):
        return [canonical(v, callees, types) for v in obj]
    return obj


def load(run: Path) -> dict:
    doc = json.loads((run / "program.json").read_text())
    prog = doc["program"]
    defs = prog["definitions"]
    callees = prog.get("callees") or []
    types = prog.get("types") or []
    root = defs[prog["root"]]
    census = Counter()
    for node in root.get("body", {}).get("nodes", []):
        fn = node.get("fn") if isinstance(node, dict) else None
        if isinstance(fn, int) and fn < len(callees):
            census[callees[fn]] += 1
        elif isinstance(fn, str):
            census[fn] += 1
    return {
        "run": run.name, "digest": doc.get("digest"), "n_instances": doc.get("n_instances"), "n_input_nodes": doc.get("n_input_nodes"),
        "profile": doc.get("profile"), "root": prog["root"], "program_fn": prog.get("program_fn"),
        "primitives": sorted(k for k, d in defs.items() if d.get("form") == "primitive"),
        "composites": {k: d.get("bindings", {}) for k, d in defs.items() if d.get("form") == "composite"},
        "encodings": {k: json.dumps(canonical(d, callees, types), sort_keys=True) for k, d in defs.items()},
        "root_bindings": root.get("bindings", {}), "census": dict(census),
    }


def _short(s: str, n: int = 160) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


def pairwise(a: dict, b: dict) -> dict:
    da, db = set(a["encodings"]), set(b["encodings"])
    only_a, only_b = sorted(da - db), sorted(db - da)
    common = da & db
    differ = sorted(k for k in common if a["encodings"][k] != b["encodings"][k])
    root_a, root_b = a["root_bindings"], b["root_bindings"]
    root_diff = {k: {"a": root_a.get(k), "b": root_b.get(k)} for k in sorted(set(root_a) | set(root_b)) if root_a.get(k) != root_b.get(k)}
    ca, cb = a["census"], b["census"]
    census_delta = {k: [ca.get(k, 0), cb.get(k, 0)] for k in sorted(set(ca) | set(cb)) if ca.get(k, 0) != cb.get(k, 0)}
    # specialization ids that differ only by statics: same base id (before '{') present with other statics on the other side
    base = lambda k: k.split("{", 1)[0]
    bases_a, bases_b = {base(k) for k in only_a}, {base(k) for k in only_b}
    return {"a": a["run"], "b": b["run"], "digest_equal": a["digest"] == b["digest"],
            "definitions_only_in_a": only_a, "definitions_only_in_b": only_b,
            "same_base_other_statics": sorted(bases_a & bases_b),
            "common_definitions_with_different_encoding": differ,
            "root_statics_differ": root_diff, "census_delta_instances_per_spec": census_delta}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-position", default="Attention_v3{,AttentionHead_v3{",
                    help="comma list of id prefixes folded into one census line per base (the per-KV-length attention specializations)")
    a = ap.parse_args(argv)
    fold = tuple(p for p in a.per_position.split(",") if p)
    infos = [load(Path(r)) for r in a.runs]
    for i in infos:
        print(f"== {i['run']}: digest {i['digest']}  instances {i['n_instances']}  input nodes {i['n_input_nodes']}  profile {i['profile']}")
        print(f"   {len(i['primitives'])} primitives, {len(i['composites'])} composite specializations; root {i['root']}")
        folded: Counter = Counter()
        spans: dict[str, list[str]] = {}
        for k, v in sorted(i["census"].items(), key=lambda kv: -kv[1]):
            if k.startswith(fold):
                b = k.split("{", 1)[0]
                folded[b] += v
                spans.setdefault(b, []).append(k)
                continue
            print(f"   {v:>8}  {k}")
        for b, v in folded.items():
            print(f"   {v:>8}  {b}{{...}}  ({len(spans[b])} per-position specializations)")
        rb = {k: _short(json.dumps(v, sort_keys=True)) for k, v in i["root_bindings"].items()}
        print(f"   root statics: {rb}")
    pairs = []
    for x in range(len(infos)):
        for y in range(x + 1, len(infos)):
            p = pairwise(infos[x], infos[y])
            pairs.append(p)
            core = lambda ks: sorted(k for k in ks if not k.startswith(fold) and not k.startswith("Root["))
            print(f"\n-- {p['a']} vs {p['b']}: digest equal = {p['digest_equal']}")
            print(f"   definitions only in {p['a']} ({len(p['definitions_only_in_a'])}): {p['definitions_only_in_a'][:12]}")
            print(f"   definitions only in {p['b']} ({len(p['definitions_only_in_b'])}): {p['definitions_only_in_b'][:12]}")
            print(f"   ... of which outside the per-position / Root families: {core(p['definitions_only_in_a'])} | {core(p['definitions_only_in_b'])}")
            print(f"   same Definition, other statics: {p['same_base_other_statics']}")
            print(f"   common ids with a different (canonical) encoding: {p['common_definitions_with_different_encoding'][:40]}")
            print(f"   root statics that differ: { {k: {kk: _short(json.dumps(vv, sort_keys=True)) for kk, vv in v.items()} for k, v in p['root_statics_differ'].items()} }")
            delta = p["census_delta_instances_per_spec"]
            print(f"   instances per spec (a, b) where different, outside the per-position families: { {k: v for k, v in delta.items() if not k.startswith(fold)} }")
            print(f"   ... per-position specs with a different count: {sum(1 for k in delta if k.startswith(fold))}")
    if a.out:
        Path(a.out).write_text(json.dumps({"runs": [{k: v for k, v in i.items() if k != 'encodings'} for i in infos], "pairs": pairs}, indent=1, sort_keys=True))
        print(f"\n[program_diff] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
