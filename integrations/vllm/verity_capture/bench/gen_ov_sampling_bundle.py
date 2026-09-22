"""[ov-sampling] the SWARM-20260917 artifact bundle of one fetched run dir, as Markdown (one section) + JSON.

    python -m verity_capture.bench.gen_ov_sampling_bundle out/gen/lanes/ov-sampling/runs/<cfg>[_ovs] [--json OUT]

Reads only the small fetched files (card.json, gates.json, stages.json, run.json, fold_summary.json, resolution.json,
replay_a.json, replay_chain.json, root_policy.json, census_reconcile.json, nonint/noninterference.json, ov_recompute.json)
and the sizes / sha256 of the three persisted Program files when present.  Torch-free.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def _load(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _sha(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _instances_specs(p: Path) -> Counter:
    c: Counter = Counter()
    if not p.exists():
        return c
    with open(p) as f:
        for line in f:
            try:
                c[json.loads(line)["spec"]] += 1
            except Exception:  # noqa: BLE001
                pass
    return c


def _accesses_summary(p: Path) -> dict | None:
    if not p.exists():
        return None
    n = 0
    kinds: Counter = Counter()
    with gzip.open(p, "rt") as f:
        for line in f:
            n += 1
            try:
                kinds[json.loads(line).get("kind", "?")] += 1
            except Exception:  # noqa: BLE001
                pass
    return {"records": n, "kinds": dict(kinds)}


def bundle(run: Path) -> dict:
    card = _load(run / "card.json") or {}
    gates = _load(run / "gates.json") or {}
    stages = _load(run / "stages.full.json") or _load(run / "stages.json") or {}   # stages.full.json: the first full pass when the gates were refreshed in place
    manifest = _load(run / "run.json") or {}
    fold = _load(run / "fold_summary.json") or {}
    res = _load(run / "resolution.json") or {}
    ra = _load(run / "replay_a.json") or {}
    rc = _load(run / "replay_chain.json") or {}
    rp = _load(run / "root_policy.json") or {}
    cr = _load(run / "census_reconcile.json") or {}
    ni = _load(run / "nonint" / "noninterference.json") or {}
    ovr = _load(run / "ov_recompute.json") or {}
    if not ovr and run.name.endswith("_ovs"):
        ovr = _load(run.parent / run.name[:-4] / "ov_recompute.json") or {}   # the recomputation ran on the base run dir (same capture)

    persisted = {}
    for name in ("program.json", "instances.jsonl", "accesses.jsonl.gz"):
        p = run / name
        persisted[name] = {"present": p.exists(), "bytes": p.stat().st_size if p.exists() else None, "sha256": _sha(p)}
    frozen = all(v["present"] for v in persisted.values()) and bool(fold.get("digest"))

    g = card.get("gates") or {}
    gate_rows = [(k, g[k].get("status"), (g[k].get("reason") or "")[:220]) for k in sorted(g)]
    unsup_ops = Counter()
    for ex in (res.get("unsupported_examples") or []):
        unsup_ops[ex.get("op", "?")] += 1
    examples = []
    seen = set()
    for ex in (res.get("unsupported_examples") or []):
        key = (ex.get("op"), str(ex.get("reason"))[:80])
        if key in seen:
            continue
        seen.add(key)
        examples.append({"op": ex.get("op"), "reason": str(ex.get("reason"))[:300]})
        if len(examples) >= 6:
            break

    vu = ra.get("coverage") or (ra.get("summary") or {}).get("vu_population") or {}
    if not vu and isinstance(g.get("G4"), dict):
        vu = ((g["G4"].get("detail") or {}).get("vu_population")) or {}
    # the card's VU line is the canonical one
    vu_line = None
    cm = run / "CARD.md"
    if cm.exists():
        for line in cm.read_text().splitlines():
            if line.startswith("VU coverage"):
                vu_line = line.split(")", 1)[-1].strip() if ")" in line else line
                vu_line = line.strip()
                break

    tier_a = {"status": ra.get("status"), "reason": ra.get("status_reason"), "summary": ra.get("summary"),
              "quarantine_specs": ((g.get("G4") or {}).get("detail") or {}).get("quarantine_specs")}
    fails: Counter = Counter()
    for i in ra.get("instances") or []:
        if i.get("status") not in (None, "ok", "pass", "skipped"):
            fails[i.get("spec")] += 1
    tier_a["failing_specs"] = dict(fails)
    tokens = ((g.get("G5") or {}).get("detail") or {}).get("tokens")

    specs = _instances_specs(run / "instances.jsonl")
    gumbel = {k: v for k, v in specs.items() if "Gumbel" in k}
    stage_bad = [(s["stage"], s["status"], (s.get("reason") or "")[:100]) for s in stages.get("stages", []) if s.get("status") != "ok"]
    spend = card.get("spend")

    return {
        "run": str(run), "config_id": manifest.get("config_id") or card.get("config_id"), "profile": manifest.get("profile"),
        "workload": manifest.get("workload_path"), "workload_sha": (manifest.get("workload") or {}).get("sha256"),
        "sampling": ovr.get("sampling") or (card.get("workload") or {}).get("sampling"),
        "verdict": card.get("verdict") or gates.get("verdict"), "gates": gate_rows,
        "program_frozen": frozen, "persisted": persisted, "fold_digest": fold.get("digest"), "spec": fold.get("spec"),
        "fold": {k: fold.get(k) for k in ("n_instances", "n_steps", "unsupported", "unresolved", "errors", "error_classes", "ret_type", "params")},
        "fold_notes": [n for n in (fold.get("fold") or {}).get("notes", []) if "sampled row" in n or "produced no" in n][:6] if isinstance(fold.get("fold"), dict) else None,
        "resolution": {"totals": res.get("totals"), "quarantine_specs": res.get("quarantine_specs"), "unsupported_ops": dict(unsup_ops), "examples": examples},
        "vu_line": vu_line, "tier_a": tier_a, "tokens": tokens, "chain": {"status": rc.get("status"), "n_slices": rc.get("n_slices")},
        "root_policy": {"status": rp.get("status"), "classes": rp.get("classes") or rp.get("by_class"), "unknown": rp.get("unknown")},
        "census": {"residue": (cr.get("totals") or {}).get("residue_kernels"), "residue_names": (cr.get("totals") or {}).get("residue_names")},
        "nonint": {"PASS": ni.get("PASS"), "tokens_equal": ni.get("tokens_equal_accepted"), "hash_equal": ni.get("hash_equal"), "hash_comparisons": ni.get("hash_comparisons")},
        "gumbel_instances": {"count": sum(gumbel.values()), "distinct_specializations": len(gumbel)},
        "instances_by_family": dict(Counter(k.split("_v")[0] for k in specs.elements())),
        "stages_not_ok": stage_bad, "spend": spend,
        "recompute": ({**ovr.get("summary", {}), "temperature": [{k: t.get(k) for k in ("step", "lanes", "div_full_model_equal", "ieee_div_equal", "ieee_mul_rcp_equal")} for t in ovr.get("temperature", [])][:2]} if ovr else None),
        "accesses": _accesses_summary(run / "accesses.jsonl.gz"),
        "quarantine": _quarantine(_load(run / "quarantine.json")),
    }


def _quarantine(q: dict | None) -> dict | None:
    if not q:
        return None
    return {"status": q.get("status"), "differential": q.get("differential"), "negative_matcher_tests": q.get("negative_matcher_tests"),
            "definitions": {k: {"differential": v.get("differential"), "device": (v.get("produce") or {}).get("device"),
                                "edge_kinds_missing": v.get("edge_kinds_missing")} for k, v in (q.get("definitions") or {}).items()}}


def _specs(specs) -> str:
    """`GumbelArgmaxBf16_v1{V=151936,SEED=0,POS=31}` × 32 → one line per Definition with the SEED / POS ranges."""
    import re
    by: dict[str, list[dict]] = {}
    for s in specs or []:
        m = re.match(r"([A-Za-z0-9_]+)\{(.*)\}$", str(s))
        if not m:
            by.setdefault(str(s), []); continue
        kv = dict(p.split("=", 1) for p in m.group(2).split(",") if "=" in p)
        by.setdefault(m.group(1), []).append(kv)
    out = []
    for name, rows in by.items():
        if not rows:
            out.append(f"`{name}`"); continue
        desc = []
        for k in rows[0]:
            vals = sorted({r.get(k) for r in rows}, key=lambda v: (len(str(v)), str(v)))
            desc.append(f"{k}={vals[0]}" if len(vals) == 1 else (f"{k}∈{{{','.join(map(str, vals))}}}" if len(vals) <= 4 else f"{k}∈{vals[0]}..{vals[-1]}"))
        out.append(f"`{name}{{{','.join(desc)}}}` × {len(rows)}")
    return "; ".join(out) if out else "none"


def markdown(b: dict) -> str:
    L = []
    L.append(f"- config id `{b['config_id']}`, profile `{b['profile']}`, workload `{b['workload']}` (sha256 `{b['workload_sha']}`), sampling `{json.dumps(b['sampling'])}`")
    pf = b["persisted"]
    L.append(f"- Program frozen: **{'yes' if b['program_frozen'] else 'no'}** — " + ", ".join(
        f"`{k}` {'present' if v['present'] else 'MISSING'}" + (f" ({v['bytes']} B, sha256 `{v['sha256'][:12]}`)" if v["present"] else "") for k, v in pf.items())
        + (f"; fold digest `{b['fold_digest'][:16]}`" if b["fold_digest"] else "; no fold digest"))
    f = b["fold"]
    L.append(f"- fold: {f.get('n_instances')} instances / {f.get('n_steps')} steps; Unsupported {f.get('unsupported')}, Unresolved {f.get('unresolved')}, errors {f.get('errors')} {json.dumps(f.get('error_classes'))}; return type `{f.get('ret_type')}`")
    r = b["resolution"]
    if r["unsupported_ops"] or r["examples"]:
        L.append(f"- Unsupported ops (resolution.json): `{json.dumps(r['unsupported_ops'])}`; first examples:")
        for ex in r["examples"]:
            L.append(f"  - `{ex['op']}`: {ex['reason']}")
    else:
        L.append("- Unsupported: none (resolution.json has no unsupported examples)")
    if r.get("quarantine_specs"):
        L.append(f"- quarantine specs in use: {_specs(r['quarantine_specs'])}")
    gi = b["gumbel_instances"]
    if gi["count"]:
        L.append(f"- `GumbelArgmaxBf16_v1` instances: {gi['count']} over {gi['distinct_specializations']} distinct specializations (one per (SEED, POS))")
    L.append(f"- VU coverage: {b['vu_line'] or 'n/a (no CARD.md)'}")
    ta = b["tier_a"]
    L.append(f"- tier a (sampled VUs, bit exact): status `{ta['status']}` {ta['summary']}" + (f"; failing specs `{json.dumps(ta['failing_specs'])}`" if ta["failing_specs"] else "") + (f"; quarantine specs evaluated: {_specs(ta['quarantine_specs'])}" if ta.get("quarantine_specs") else ""))
    L.append(f"- served tokens from logits (G5 tokens): `{json.dumps(b['tokens'])[:300]}`")
    L.append(f"- chained slices: `{b['chain']['status']}` ({b['chain']['n_slices']} slices); root policy `{b['root_policy']['status']}`; census residue {b['census']['residue']} kernels {json.dumps(b['census']['residue_names'])[:200] if b['census']['residue_names'] else ''}")
    L.append(f"- control non-interference: PASS={b['nonint']['PASS']}, hashes equal {b['nonint']['hash_equal']}/{b['nonint']['hash_comparisons']}")
    if b["gates"]:
        L.append(f"- gates: " + "; ".join(f"{k} **{s}**" for k, s, _ in b["gates"]) + f" → verdict **{b['verdict']}**")
        for k, s, reason in b["gates"]:
            if s == "fail" and reason:
                L.append(f"  - {k}: {reason}")
    else:
        L.append(f"- gates: n/a (no card: stages not ok {b['stages_not_ok']})")
    if b["stages_not_ok"]:
        L.append(f"- stages not ok: `{json.dumps(b['stages_not_ok'])[:400]}`")
    if b.get("recompute"):
        L.append(f"- lane recomputation (ov_recompute.json): `{json.dumps(b['recompute'])[:400]}`")
    if b.get("quarantine"):
        q = b["quarantine"]
        L.append(f"- quarantine admission (quarantine.json): status `{q['status']}`, differential {q['differential']}, negative_matcher_tests `{q['negative_matcher_tests']}`; "
                 + "; ".join(f"`{k}` {v['differential']['passed']}/{v['differential']['total']} on {v['device']}" for k, v in q["definitions"].items()))
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    b = bundle(Path(a.run))
    if a.json:
        Path(a.json).write_text(json.dumps(b, indent=1, default=str))
    print(markdown(b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
