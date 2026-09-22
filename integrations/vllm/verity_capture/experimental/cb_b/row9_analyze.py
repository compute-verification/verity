"""Row-9 census (CB-B): the canonical VU interface of the padded (Serve@3) Program split into the active region and the
inactive padded suffix, per CMT-1 value-family class.

    python -m verity_capture.experimental.cb_b.row9_analyze --runs <row9 dir> --manifest <CMT-1 interface_manifest.json>
        --active-events 15 --out <json>  [--pair r155x14 --pair r155x64]

Inputs: for each pair `<runs>/<pair>/{flat,padded}/{instances.jsonl,census_vu/census.json,program.json}` written by
`row9_prepare` + `vu_census`.  Attribution: a VU class row of the census names its root-body entry (`root_spec`); the
VUs of the class are spread evenly over the root nodes of that specialization (the census's own uniformity check —
`nonuniform == []` — is what makes this exact), every root node has a step (`instances.jsonl`), and a step is *active*
iff its sampling event physically occurred in the record (`--active-events` = number of sampled tokens).  Inactive-step
nodes are further split by what they read (root-body descriptor, `n` references): padding machinery (inserted guards /
muxes / constants), body nodes reading a value produced in an active step (carried KV state, the last live token),
and body nodes reading only inactive-step values / weights.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from typing import Any

from verity_capture.bench.vu_census import _compiled_location, _eager_location, _family

KEY_FIELDS = ("root_family", "logical_class", "vu_kind", "prim", "width_bits", "fanout", "eager_class", "compiled_class")


def load_instances(p: str) -> dict[int, dict]:
    out = {}
    with open(p) as f:
        next(f)
        for line in f:
            o = json.loads(line)
            out[o["node"]] = o
    return out


def class_index(manifest: dict) -> dict[tuple, dict]:
    idx = {}
    for c in manifest["classes"]:
        k = tuple(c["census_key"][f] for f in KEY_FIELDS)
        idx[k] = {"id": c["id"], "row": manifest["class_to_row"].get(c["id"]), "role": c.get("role")}
    return idx


def rows_index(manifest: dict) -> dict[str, dict]:
    return {r["id"]: {"name": r["name"], "family": r["family"], "required": r["required"]} for r in manifest["rows"]}


def reads_of_root(program_json: str) -> tuple[list[list[int]], int]:
    """root node i -> the root node indices it reads (via `n` references); n_input_nodes."""
    import verity_vllm.registry.b1  # noqa: F401
    import verity_vllm.registry.serve3  # noqa: F401
    from verity.ir.codec import decode_program
    from verity.ir.defs import REGISTRY
    from verity.ir.refs import runs

    with open(program_json) as f:
        rec = json.load(f)
    desc = rec.get("program", rec)
    body = decode_program(desc, REGISTRY).root.body
    reads = []
    for n in body.nodes:
        s = set()
        for c in n.args:
            for sp, idx, lo, hi in runs(c.refs):
                if sp == "n":
                    s.add(idx)
        reads.append(sorted(s))
    return reads, int(rec.get("n_input_nodes", 0))


def bucket_nodes(inst: dict[int, dict], reads: list[list[int]], n_in: int, active_events: int) -> dict[int, str]:
    """root-body node k -> bucket: active | active-machinery | inactive-machinery | inactive-reads-active | inactive-interior."""
    step_of = {k: o["step"] for k, o in inst.items()}
    b = {}
    for k, o in inst.items():
        st = o["step"]
        mach = bool(o.get("inserted"))
        if st < active_events:
            b[k] = "active-machinery" if mach else "active"
            continue
        if mach:
            b[k] = "inactive-machinery"
            continue
        ra = False
        for r in reads[n_in + k]:
            j = r - n_in
            if j >= 0 and j in step_of and step_of[j] < active_events:
                ra = True
                break
        b[k] = "inactive-reads-active" if ra else "inactive-interior"
    return b


def attribute(census: dict, inst: dict[int, dict], buckets: dict[int, str], cls_idx: dict) -> dict:
    """Per (class id, bucket): VU count / bytes; per bucket totals; per step totals.  Root-input rows go to `inputs`."""
    nodes_by_spec: dict[str, list[int]] = collections.defaultdict(list)
    for k, o in inst.items():
        nodes_by_spec[o["spec"]].append(k)
    per = collections.defaultdict(lambda: {"vus": 0.0, "count": 0.0, "bytes": 0.0})       # count = interface values (out gates)
    per_step = collections.defaultdict(lambda: {"vus": 0.0, "count": 0.0, "bytes": 0.0})
    unattributed = []
    for r in census["classes"]:
        spec = r["root_spec"]
        nodes = nodes_by_spec.get(spec)
        wsum = sum(w for w, _, _, _ in r["out"])
        if r["pattern"] == "input" or spec.startswith("Input"):
            dist = {"inputs": 1.0}
        elif not nodes:
            unattributed.append({"root_spec": spec, "total": r["total"], "pattern": r["pattern"]})
            dist = {"unattributed": 1.0}
        else:
            cnt = collections.Counter(buckets[k] for k in nodes)
            dist = {bk: c / len(nodes) for bk, c in cnt.items()}
            for k in nodes:
                per_step[inst[k]["step"]]["vus"] += r["total"] / len(nodes)
                per_step[inst[k]["step"]]["count"] += r["total"] * len(r["out"]) / len(nodes)
                per_step[inst[k]["step"]]["bytes"] += r["total"] * wsum / 8 / len(nodes)
        for bk, frac in dist.items():
            per[("VU:" + _family(spec), bk)]["vus"] += r["total"] * frac
        for w, prim, cls, fo in r["out"]:
            eager_cls, _ = _eager_location(cls, r["pattern"])
            comp_cls, _ = _compiled_location(cls, spec, prim)
            key = (_family(spec), cls, r["vu_kind"], prim, w, fo, eager_cls, comp_cls)
            ci = cls_idx.get(key)
            cid = ci["id"] if ci else "NEW:" + "/".join(str(x) for x in key)
            for bk, frac in dist.items():
                d = per[(cid, bk)]
                d["count"] += r["total"] * frac
                d["bytes"] += r["total"] * w / 8 * frac
    return {"per_class_bucket": {f"{c}\t{b}": v for (c, b), v in per.items()}, "per_step": dict(sorted(per_step.items())),
            "unattributed": unattributed}


def totals(att: dict) -> dict[str, dict]:
    t = collections.defaultdict(lambda: {"vus": 0.0, "count": 0.0, "bytes": 0.0})
    for kb, v in att["per_class_bucket"].items():
        _, b = kb.split("\t")
        for f in ("vus", "count", "bytes"):
            t[b][f] += v[f]
            t["all"][f] += v[f]
    return dict(t)


def analyze_pair(runs: str, pair: str, manifest: dict, active_events: int) -> dict:
    cls_idx = class_index(manifest)
    out: dict[str, Any] = {"pair": pair, "active_events": active_events}
    for side in ("flat", "padded"):
        d = os.path.join(runs, pair, side)
        census = json.load(open(os.path.join(d, "census_vu", "census.json")))
        inst = load_instances(os.path.join(d, "instances.jsonl"))
        reads, n_in = reads_of_root(os.path.join(d, "program.json"))
        buckets = bucket_nodes(inst, reads, n_in, active_events)
        att = attribute(census, inst, buckets, cls_idx)
        tot = totals(att)
        steps = sorted({o["step"] for o in inst.values()})
        node_buckets = collections.Counter(buckets.values())
        spec_buckets = collections.defaultdict(collections.Counter)
        for k, bk in buckets.items():
            spec_buckets[bk][_family(inst[k]["spec"])] += 1
        out[side] = {
            "program_digest": census["program_digest"], "gates": census["gates"], "root_nodes": census["root_nodes"],
            "vu_count": census["vu_count"], "n_classes": census["n_classes"], "n_nonuniform": census["n_nonuniform"],
            "n_unsupported": census["n_unsupported"], "logical_bytes": census["aggregate"]["logical_bytes"],
            "steps": len(steps), "active_steps": sum(1 for s in steps if s < active_events), "inactive_steps": sum(1 for s in steps if s >= active_events),
            "node_buckets": dict(node_buckets), "spec_families_by_bucket": {b: dict(c) for b, c in spec_buckets.items()},
            "totals_by_bucket": tot, "per_step": att["per_step"], "per_class_bucket": att["per_class_bucket"],
            "unattributed": att["unattributed"],
            "check_vu_sum": tot["all"]["vus"], "check_values_sum": tot["all"]["count"], "check_bytes_sum": tot["all"]["bytes"],
        }
    return out


def _fmt_b(b: float) -> str:
    return f"{b / 1e6:,.2f} MB" if b >= 1e6 else f"{b / 1e3:,.1f} kB"


def render_md(res: dict, manifest: dict, meta: dict) -> str:
    rows = rows_index(manifest)
    L = []
    L.append("# Row 9 census — inactive padded interiors of the Serve@3 rectangle (CB-B)\n")
    L.append(f"Record `{meta['record']}`: prompt {meta['prompt_len']}, max_tokens {meta['max_tokens']}, finish_reason `{meta['finish_reason']}` "
             f"— {meta['active_events']} sampling events physically occurred (steps 0..{meta['active_events'] - 1}); no EOS was sampled.  "
             f"Active region := those steps.  Inactive suffix := decode steps whose event never occurred in serving (present in the "
             f"rectangle only).  Census: canonical VU query `{meta['query_id'][:16]}…` (W_max 32), `verity_capture/bench/vu_census.py` "
             f"{meta['census_version']}; classes = CMT-1's 54 value-family classes (`interface_manifest.json`, matched on the census key).\n")
    L.append("Method: every VU class of the census names one root-body entry; its VUs are spread evenly over the root nodes of that "
             "specialization (exact: the census reports 0 non-uniform classes); each root node has a step; inactive-step nodes are "
             "split by their `n`-references in the root body.  Numbers are exact counts of the Programs listed (measured).\n")
    L.append("## Headline\n")
    L.append("| T_max | active events | inactive steps | padded VUs | inactive VUs (share of all / of non-input) | padded bytes | inactive bytes (share of all / of per-request excl. weights) | prescribed (machinery) | unconstrained-but-required |")
    L.append("|---|---:|---:|---:|---|---:|---|---|---|")
    for pr in res["pairs"]:
        p = pr["padded"]
        T = p["totals_by_bucket"]
        allv, allb = T["all"]["vus"], T["all"]["bytes"]
        inp = T.get("inputs", {}).get("vus", 0)
        req = p["logical_bytes"]["per_request_excl_weights_excl_constants"]
        iv = sum(T[b]["vus"] for b in T if b.startswith("inactive"))
        ib = sum(T[b]["bytes"] for b in T if b.startswith("inactive"))
        mach = T.get("inactive-machinery", {"vus": 0, "bytes": 0})
        unc_v = iv - mach["vus"]
        unc_b = ib - mach["bytes"]
        L.append(f"| {p['steps'] - 1} | {pr['active_events']} | {p['inactive_steps']} | {p['vu_count']:,} | {iv:,.0f} ({100 * iv / allv:.2f} % / {100 * iv / (allv - inp):.2f} %) | "
                 f"{_fmt_b(allb)} | {_fmt_b(ib)} ({100 * ib / allb:.2f} % / {100 * ib / req:.2f} %) | {mach['vus']:,.0f} VUs, {_fmt_b(mach['bytes'])} | {unc_v:,.0f} VUs, {_fmt_b(unc_b)} |")
    L.append("")
    L.append("Scaling: every inactive decode step costs the same as an active one (the canonical attention VU count depends on the number of "
             "128-key blocks, constant for KV length ≤ 256 here), so inactive bytes ≈ (T_max − n_actual) × per-decode-step bytes; the padded "
             "rectangle's *machinery* (guards + row muxes + zero rows) is ~0.2 % of the inactive VUs — the cost of padding is the interior "
             "body of the dead steps, not the muxes.\n")
    for pr in res["pairs"]:
        f, p = pr["flat"], pr["padded"]
        L.append(f"## {pr['pair']}: T_max = {p['steps'] - 1} decode steps (rectangle prefill + {p['steps'] - 1}); active {p['active_steps']} steps, inactive {p['inactive_steps']} steps\n")
        L.append("| Program | gates | root nodes | VUs | interface bytes (all, incl. weights) | per-request bytes (excl. weights/consts) |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for name, s in (("flat (active only, T_max steps)", f), ("padded Serve@3 rectangle", p)):
            lb = s["logical_bytes"]
            L.append(f"| {name} `{s['program_digest'][:12]}…` | {s['gates']:,} | {s['root_nodes']:,} | {s['vu_count']:,} | {_fmt_b(lb['total_incl_weights_and_constants'])} | {_fmt_b(lb['per_request_excl_weights_excl_constants'])} |")
        L.append("")
        L.append("Padded Program, by region (VUs / interface bytes; `inputs` = root weight + prompt batches, counted once):\n")
        L.append("| bucket | root nodes | VUs | share of VUs | interface values | bytes | share of bytes |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        T = p["totals_by_bucket"]
        allv, allc, allb = T["all"]["vus"], T["all"]["count"], T["all"]["bytes"]
        order = ["inputs", "active", "active-machinery", "inactive-reads-active", "inactive-interior", "inactive-machinery", "unattributed"]
        for b in order:
            if b not in T:
                continue
            nb = p["node_buckets"].get(b, 2 if b == "inputs" else 0)
            L.append(f"| {b} | {nb:,} | {T[b]['vus']:,.0f} | {100 * T[b]['vus'] / allv:.2f} % | {T[b]['count']:,.0f} | {_fmt_b(T[b]['bytes'])} | {100 * T[b]['bytes'] / allb:.2f} % |")
        inact = sum(T[b]["vus"] for b in T if b.startswith("inactive"))
        inactc = sum(T[b]["count"] for b in T if b.startswith("inactive"))
        inactb = sum(T[b]["bytes"] for b in T if b.startswith("inactive"))
        L.append(f"| **inactive suffix total** | {sum(p['node_buckets'].get(b, 0) for b in p['node_buckets'] if b.startswith('inactive')):,} | **{inact:,.0f}** | **{100 * inact / allv:.2f} %** | **{inactc:,.0f}** | **{_fmt_b(inactb)}** | **{100 * inactb / allb:.2f} %** |")
        L.append("")
        req = p["logical_bytes"]["per_request_excl_weights_excl_constants"]
        L.append(f"Check: Σ buckets = {allv:,.0f} VUs (census {p['vu_count']:,}), {_fmt_b(allb)} (census {_fmt_b(p['logical_bytes']['total_incl_weights_and_constants'])}).  "
                 f"**Inactive suffix = {100 * inactb / allb:.2f} % of all interface bytes, {100 * inactb / req:.2f} % of the per-request bytes (excl. weights / constants, {_fmt_b(req)}), "
                 f"{100 * inact / (allv - T.get('inputs', {}).get('vus', 0)):.2f} % of the non-input VUs.**  Per inactive decode step: "
                 f"{p['per_step'].get(str(p['steps'] - 1), p['per_step'].get(p['steps'] - 1, {})).get('vus', 0):,.0f} VUs / {_fmt_b(p['per_step'].get(str(p['steps'] - 1), p['per_step'].get(p['steps'] - 1, {})).get('bytes', 0))}.\n")
        L.append("Inactive-step root nodes by definition family and bucket:\n")
        for b in ("inactive-reads-active", "inactive-interior", "inactive-machinery", "active-machinery"):
            fams = p["spec_families_by_bucket"].get(b)
            if fams:
                L.append(f"- `{b}`: " + ", ".join(f"{k} ×{v}" for k, v in sorted(fams.items(), key=lambda x: -x[1])))
        L.append("")
        # per-class table
        L.append("Per value-family class (CMT-1 ids; `NEW:` = classes that exist only in the padded Program; counts are interface values = out gates):\n")
        L.append("| class | CMT-1 row | active values | active bytes | inactive values | inactive bytes | inactive share |")
        L.append("|---|---|---:|---:|---:|---:|---:|")
        agg = collections.defaultdict(lambda: {"a": 0.0, "ab": 0.0, "i": 0.0, "ib": 0.0})
        for kb, v in p["per_class_bucket"].items():
            cid, b = kb.split("\t")
            if cid.startswith("VU:"):
                continue
            g = agg[cid]
            if b.startswith("inactive"):
                g["i"] += v["count"]
                g["ib"] += v["bytes"]
            else:
                g["a"] += v["count"]
                g["ab"] += v["bytes"]
        c2r = manifest["class_to_row"]
        for cid, g in sorted(agg.items(), key=lambda x: -(x[1]["ab"] + x[1]["ib"])):
            rid = c2r.get(cid, "-")
            if isinstance(rid, list):
                rid = ",".join(str(x.get("row", x)) if isinstance(x, dict) else str(x) for x in rid)
            elif isinstance(rid, dict):
                rid = str(rid.get("row", rid))
            rname = rows.get(rid, {}).get("name", "padding machinery (not in CMT-1)" if cid.startswith("NEW:") else "-")
            tot = g["a"] + g["i"]
            L.append(f"| `{cid[:60]}` | {rid}: {rname[:48]} | {g['a']:,.0f} | {_fmt_b(g['ab'])} | {g['i']:,.0f} | {_fmt_b(g['ib'])} | {100 * g['i'] / tot if tot else 0:.1f} % |")
        L.append("")
        # per step
        L.append("Per step (padded Program; VUs / bytes; A = active, I = inactive):\n")
        L.append("| step | region | VUs | interface values | bytes |")
        L.append("|---|---|---:|---:|---:|")
        ps = p["per_step"]
        keys = sorted(ps, key=int)
        show = keys if len(keys) <= 20 else keys[:3] + keys[pr["active_events"] - 2: pr["active_events"] + 2] + keys[-2:]
        last = None
        for k in show:
            if last is not None and int(k) != int(last) + 1:
                L.append("| … | | | |")
            L.append(f"| {k} | {'A' if int(k) < pr['active_events'] else 'I'} | {ps[k]['vus']:,.0f} | {ps[k]['count']:,.0f} | {_fmt_b(ps[k]['bytes'])} |")
            last = k
        L.append("")
        if p["unattributed"]:
            L.append(f"Unattributed census rows: {p['unattributed'][:5]}\n")
    L.append("## Prescribed vs unconstrained values in the inactive suffix (`verity_vllm/registry/serve3.py`, spec §3/§6/§8)\n")
    L.append("Source-inspected.  Serve@3 masks *only at the step interface*: no gate inside a decode-step body is guarded.\n")
    L.append("- **Prescribed** (a value the specification fixes in an inactive step, `live_in = 0`): the guard's outputs "
             "`tok_out = SelectI32(live, y, Const32[0]) = 0` and `live_out = BitAnd(0, ·) = 0`; every appended cache word "
             "`k_new, v_new = SelectBf16(live, computed, 0x0000) = 0` (the `batch(SelectBf16_v1)` mux outputs); the zero rows / "
             "`Const32[EOS]` / `Const32[0]` constants.  These are exactly the `inactive-machinery` VUs above (guard + mux + const "
             "nodes), plus the *carried* active state read by the suffix (the KV rows of active steps are the active Program's values).")
    L.append("- **Unconstrained-but-required** (the interior body over masked inputs): everything else in an inactive step — the "
             "embedding of the pad token 0, every GEMM / RMSNorm / RoPE / SiluMul / attention / logits value, the raw sample `y` "
             "(the stop test `I32Eq(y, EOS)` is computed and then absorbed by `BitAnd(0, ·)`), and the *un-muxed* K/V rows before the "
             "mux.  They are fully determined by the Program (weights, pad token, carried state) but no serving run ever produces "
             "them; the interface must nevertheless commit / verify them because the rectangle's Definitions are unconditional.  "
             "These are the `inactive-reads-active` (attention over carried KV + the first inactive step's reads of the last "
             "active guard) and `inactive-interior` buckets above.")
    L.append("- Nothing in the inactive suffix is *free*: the inactive-step relation `live_in = 0 ⟹ (tok_out, k_new, v_new, live_out) = (0, 0…0, 0…0, 0)` "
             "is a property of the composite (spec §3), but the canonical W_max = 32 partition still exposes the body's interior VUs.\n")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pair", action="append", default=None)
    ap.add_argument("--active-events", type=int, required=True)
    ap.add_argument("--record", default="B0-cov_b0_b1_c256")
    ap.add_argument("--prompt-len", type=int, default=155)
    ap.add_argument("--max-tokens", type=int, default=15)
    ap.add_argument("--finish-reason", default="length")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    manifest = json.load(open(a.manifest))
    pairs = a.pair or ["r155x14", "r155x64"]
    res = {"schema": "verity-gen/row9-census/v1", "record": a.record, "active_events": a.active_events, "pairs": []}
    for pr in pairs:
        res["pairs"].append(analyze_pair(a.runs, pr, manifest, a.active_events))
    c0 = json.load(open(os.path.join(a.runs, pairs[0], "padded", "census_vu", "census.summary.json")))
    meta = {"record": a.record, "prompt_len": a.prompt_len, "max_tokens": a.max_tokens, "finish_reason": a.finish_reason,
            "active_events": a.active_events, "query_id": c0["query_id"], "census_version": c0["version"]}
    res["meta"] = meta
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    with open(os.path.splitext(a.out)[0] + ".md", "w") as f:
        f.write(render_md(res, manifest, meta))
    for pr in res["pairs"]:
        p = pr["padded"]
        T = p["totals_by_bucket"]
        print(pr["pair"], "vu", p["vu_count"], "sum", round(T["all"]["vus"]), "values", round(T["all"]["count"]), "bytes", round(T["all"]["bytes"]), "buckets",
              {b: (round(v["vus"]), round(v["bytes"])) for b, v in T.items()}, "unattributed", len(p["unattributed"]))
        print("  reads-active families:", p["spec_families_by_bucket"].get("inactive-reads-active"))
        print("  interior families:", p["spec_families_by_bucket"].get("inactive-interior"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
