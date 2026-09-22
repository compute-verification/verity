"""[R8 C-codec] Corpus driver: inventory, A3 model fit on the training split, encode/decode/check every (run, phi level, codec) point.

  python -m verity_capture.experimental.aot_codec.evaluate inventory --records <dir-with-extracted-records> --out <artefact dir>
  python -m verity_capture.experimental.aot_codec.evaluate run --records <dir> --out <artefact dir> --train-regex '^B0-cov_b0_' [--model <frozen.json>]

Outputs (all under --out): inventory.jsonl/.md, hints/<run>.json (the extracted ScheduleHint + audit + check), a3_model.json (frozen, sha256
recorded per row), results.jsonl (one row per point), tables.md.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from . import codecs, facts, model
from .bitio import bits_for

LEVELS = ["phi0", "phi1a", "phi1", "phi2"]


def git_rev() -> str:
    """Commit of the code being run: `git rev-parse` in a checkout, else the shipped tree's EXPORT.json (record_v5/ship.sh), else unknown."""
    here = Path(__file__).resolve()
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here.parent, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        pass
    for p in here.parents:
        if (p / "EXPORT.json").exists():
            try:
                e = json.loads((p / "EXPORT.json").read_text())
                return e["commit"] + ("+dirty" if e.get("dirty_patch_bytes") else "")
            except Exception:
                break
    return "unknown"


def classify(run_id: str) -> dict:
    m = re.match(r"^(?P<fam>[A-Z0-9_]+)-cov_(?P<case>[a-z0-9_]+?)_(?P<prof>b1|b8-static|b8-mixed)_(?P<ctx>c\d+)(?P<suffix>.*)$", run_id)
    if not m:
        return {"family": run_id.split("-")[0], "profile": "?", "context": "?", "suffix": ""}
    return {"family": m["fam"], "profile": {"b1": "B1", "b8-static": "B8-static", "b8-mixed": "B8-mixed"}[m["prof"]], "context": m["ctx"].upper(),
            "suffix": m["suffix"]}


def declared_bound_bits(phi: dict, h: dict) -> dict:
    """Max A1 code length over the declared admissible class (not the realized schedule).  Class: N requests as in phi, <= max_num_seqs
    rows per step, every row may be any prefill chunk size <= its prompt, T <= T_bound where
      phi2: T_bound = declared expected_engine_steps;  phi1a/phi1: max arrival + serial completion;  phi0: serial completion
      (arrivals assumed to fall before the serial completion of the earlier requests -- the class must be bounded somewhere)."""
    N = len(phi["requests"])
    S = min(int(phi["engine"]["max_num_seqs"]), N)
    budget = max(int(phi["engine"]["max_num_batched_tokens"]) - (S - 1), 1)
    chunks = [math.ceil(q["prompt_len"] / budget) for q in phi["requests"]]
    serial = sum(q["max_tokens"] + c for q, c in zip(phi["requests"], chunks))
    if phi.get("expected_engine_steps") is not None:
        T = int(phi["expected_engine_steps"])
    elif facts_has(phi, "arrive_step"):
        T = max(q["arrive_step"] for q in phi["requests"]) + serial
    else:
        T = serial
    framing = 2 * (T + 1).bit_length() - 1  # Elias-gamma of T
    req = 0
    for q in phi["requests"]:
        if "arrive_step" not in q:
            req += bits_for(max(T, 1))
        if "n_generated" not in q:
            req += bits_for(q["max_tokens"] + 1)
        req += 2  # finish reason when n_generated < max_tokens
    maxL = max(q["prompt_len"] for q in phi["requests"])
    per_step = bits_for(S + 1) + S * (bits_for(N) + bits_for(maxL))          # A1 explicit step: n_rows, (request, chunk) per row
    # A2 worst case over the same class: clean=0, arrivals as gamma deltas, fr_rule=0 (+2 bits per stopped request), per-request ngen flag+value,
    # T_eq=0 + gamma(T), all_steps_match=0 + one gamma-coded gap per escaped step (+ the terminating gap), and every escaped step explicit
    # with per-row nc_eq flag + logits flag + explicit logits list (worst).  bound(k) = at most k escaped steps; bound(T) = every step
    # escaped (the class C1 bound; T + 1 zero gaps of 1 bit each).
    a2_fixed = 1 + 1 + 1 + framing + 1
    for q in phi["requests"]:
        if "arrive_step" not in q:
            a2_fixed += 2 * (2 * T + 1).bit_length() - 1  # gamma of zig-zag delta, |delta| <= T
        if "n_generated" not in q:
            a2_fixed += 1 + bits_for(q["max_tokens"] + 1)
        a2_fixed += 2
    gap = 2 * (T + 1).bit_length() - 1  # gamma of a gap <= T (A2's raw integer code; A3 pays the histogram escape + the same gamma)
    a2_step = per_step + S + 1 + bits_for(S + 1) + S * (bits_for(N) + 1)  # explicit rows (+nc_eq each), logits flag + explicit logits
    a2_k = {k: a2_fixed + (k + 1) * gap + k * a2_step for k in (0, 1, 2, 5)}
    return {"T_bound": T, "bits_A1_language": framing + 2 + req + T * per_step, "bits_A2_language": a2_fixed + (T + 1) + T * a2_step,
            "bits_A2_k_escapes": a2_k, "bits_A2_clean": 1 + sum(2 * (2 * T + 1).bit_length() - 1 for q in phi["requests"] if "arrive_step" not in q)}


def facts_has(phi: dict, field: str) -> bool:
    return bool(phi["requests"]) and all(field in q for q in phi["requests"])


def load_all(records_dir: Path, only: list[str] | None = None) -> list[dict]:
    out = []
    for d in sorted(p for p in Path(records_dir).iterdir() if p.is_dir() and (p / "spec.json").exists()):
        if only and d.name not in only:
            continue
        t0 = time.time()
        r = facts.load_record(d)
        r["load_s"] = round(time.time() - t0, 2)
        out.append(r)
    return out


def inventory(records: list[dict], out: Path) -> list[dict]:
    rows = []
    for r in records:
        h, audit = facts.extract_hint(r)
        chk = facts.check_record(r, h)
        ps = facts.program_stats(r)
        io = facts.io_stats(r, h)
        cls = classify(r["run_id"])
        phi0 = facts.phi_from_record(r)
        rows.append({"run_id": r["run_id"], **cls, "model": phi0["model"], "engine": phi0["engine"], "verdict": ps["verdict"],
                     "n_requests": len(h["requests"]), "n_steps": len(h["steps"]), "declared_engine_steps": h.get("_declared_engine_steps"),
                     "prompt_tokens": io["prompt_tokens"], "generated_tokens": io["generated_tokens"], "max_concurrent": max(len(s["rows"]) for s in h["steps"]),
                     "arrivals": [q["arrive_step"] for q in h["requests"]], "n_generated": [q["n_generated"] for q in h["requests"]],
                     "max_tokens": [q["max_tokens"] for q in h["requests"]], "prompt_lens": [q["prompt_len"] for q in h["requests"]],
                     "finish_reasons": sorted(set(q["finish_reason"] for q in h["requests"])),
                     "program_bytes": ps["program_bytes"], "n_instances": ps["n_instances"], "program_digest": ps["program_digest"],
                     "steps_padded": sum(1 for e in r["steps"] if e.get("num_tokens_padded", e["num_tokens"]) != e["num_tokens"]),
                     "phases": {k: sum(1 for e in r["steps"] if e.get("phase_hint") == k) for k in ("prefill", "decode", "mixed")},
                     "audit_events": audit["events"], "unexpressible": audit["unexpressible"], "structural_check_ok": chk["ok"],
                     "log_fields": sorted(r["steps"][0].keys()) if r["steps"] else [], "load_s": r["load_s"]})
        (out / "hints").mkdir(parents=True, exist_ok=True)
        (out / "hints" / f"{r['run_id']}.json").write_text(json.dumps({"hint": facts.strip_private(h), "declared_engine_steps": h.get("_declared_engine_steps"),
                                                                       "audit": audit, "check": chk, "hint_sha256": facts.digest(facts.strip_private(h))}, indent=1))
    (out / "inventory.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    lines = ["| run | family | profile | ctx | verdict | N | T | decl T | prompt tok | gen tok | max conc | padded steps | phases p/d/m | program B | instances | audit events | check |", "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|"]
    for x in rows:
        ph = x["phases"]
        lines.append(f"| `{x['run_id']}` | {x['family']} | {x['profile']} | {x['context']} | {x['verdict']} | {x['n_requests']} | {x['n_steps']} | {x['declared_engine_steps']} | {x['prompt_tokens']} | {x['generated_tokens']} | {x['max_concurrent']} | {x['steps_padded']} | {ph['prefill']}/{ph['decode']}/{ph['mixed']} | {x['program_bytes']} | {x['n_instances']} | {len(x['audit_events'])} | {'ok' if x['structural_check_ok'] else 'FAIL'} |")
    (out / "inventory.md").write_text("# Corpus inventory (measured from the archived records)\n\n" + "\n".join(lines) + "\n")
    return rows


def run(records: list[dict], out: Path, train_regex: str, model_path: Path | None, rev: str) -> None:
    prepared = []
    for r in records:
        h, audit = facts.extract_hint(r)
        chk = facts.check_record(r, h)
        phi0 = facts.phi_from_record(r)
        prepared.append((r, h, audit, chk, phi0))
    train_ids = [r["run_id"] for r, *_ in prepared if re.search(train_regex, r["run_id"])]
    if model_path and Path(model_path).exists():
        m, msha = model.load(model_path)
    else:
        training = [(facts.phi_variant(phi0, h, lvl), facts.strip_private(h)) for r, h, a, c, phi0 in prepared if r["run_id"] in train_ids for lvl in LEVELS]
        m = model.fit(training, train_ids)
        msha = model.freeze(m, out / "a3_model.json")
    rows = []
    for r, h, audit, chk, phi0 in prepared:
        hs = facts.strip_private(h)
        cls = classify(r["run_id"])
        ps = facts.program_stats(r)
        io = facts.io_stats(r, h)
        split = "train" if r["run_id"] in train_ids else "heldout"
        for lvl in LEVELS:
            phi = facts.phi_variant(phi0, h, lvl)
            bound = declared_bound_bits(phi, hs)
            for name in codecs.CODECS:
                t0 = time.time()
                err = None
                try:
                    res = codecs.encode(name, phi, hs, m)
                    back = codecs.decode(name, phi, res.bits, m)
                    ok = facts.canonical_json(back) == facts.canonical_json(hs)
                    bits, marks, ideal = res.n_bits, res.marks, res.ideal_bits
                except Exception as e:  # unexpressible -> recorded, not hidden
                    ok, bits, marks, ideal, err = False, None, {}, None, f"{type(e).__name__}: {e}"
                nreq = len(hs["requests"])
                rows.append({"run_id": r["run_id"], **cls, "split": split, "verdict": ps["verdict"], "phi": lvl, "codec": name,
                             "bits": bits, "ideal_bits": ideal, "roundtrip_ok": ok, "structural_check_ok": chk["ok"], "error": err,
                             "bits_per_request": (bits / nreq) if bits is not None else None,
                             "bits_per_generated_token": (bits / io["generated_tokens"]) if bits is not None and io["generated_tokens"] else None,
                             "bits_per_io_token": (bits / io["io_tokens"]) if bits is not None and io["io_tokens"] else None,
                             "marks": marks, "n_requests": nreq, "n_steps": len(hs["steps"]), "prompt_tokens": io["prompt_tokens"],
                             "generated_tokens": io["generated_tokens"], "io_bytes_2B_per_token": io["io_bytes_2B_per_token"],
                             "program_bytes": ps["program_bytes"], "n_instances": ps["n_instances"], "program_digest": ps["program_digest"],
                             "declared_bound": bound, "model_sha256": msha if name == "A3" else None, "code_rev": rev, "encode_decode_s": round(time.time() - t0, 3)})
    (out / "results.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    (out / "tables.md").write_text(tables(rows, m, msha, rev))


def _stats(xs: list[float]) -> str:
    if not xs:
        return "—"
    xs = sorted(xs)
    p95 = xs[min(len(xs) - 1, int(math.ceil(0.95 * len(xs))) - 1)]
    return f"{statistics.mean(xs):.2f} / {statistics.median(xs):.2f} / {p95:.2f} / {xs[-1]:.2f}"


def tables(rows: list[dict], m: dict, msha: str, rev: str) -> str:
    ok_rows = [x for x in rows if x["bits"] is not None]
    L = [f"# Advice-length tables — code rev `{rev[:12]}`, A3 model sha256 `{msha[:12]}` (fitted on {len(m['fitted_on'])} runs: {', '.join(m['fitted_on'])})", ""]
    L.append("All numbers **measured** (exact bit counts of `encode`; every listed point round-trips `decode(phi, encode(phi,h)) == h`; points that failed are listed separately).")
    L.append("Format: mean / median / p95 / max over runs.  'ideal' = Σ −log2 p of the A3 symbol stream (calculated; the emitted length is ⌈·⌉+≤1).")
    for split in ("all", "train", "heldout"):
        sub = [x for x in ok_rows if split == "all" or x["split"] == split]
        if not sub:
            continue
        L += ["", f"## Headline — advice bits per request ({split}; {len({x['run_id'] for x in sub})} runs)", "",
              "| phi | A0 explicit | A1 remove-derivable | A2 predict+residual | A3 compressed | A3 ideal |", "|---|---|---|---|---|---|"]
        for lvl in LEVELS:
            cells = []
            for name in codecs.CODECS:
                cells.append(_stats([x["bits_per_request"] for x in sub if x["phi"] == lvl and x["codec"] == name]))
            cells.append(_stats([x["ideal_bits"] / x["n_requests"] for x in sub if x["phi"] == lvl and x["codec"] == "A3" and x["ideal_bits"] is not None]))
            L.append(f"| {lvl} | " + " | ".join(cells) + " |")
        L += ["", f"### Total bits per run ({split})", "", "| phi | A0 | A1 | A2 | A3 |", "|---|---|---|---|---|"]
        for lvl in LEVELS:
            L.append(f"| {lvl} | " + " | ".join(_stats([x["bits"] for x in sub if x["phi"] == lvl and x["codec"] == n]) for n in codecs.CODECS) + " |")
        L += ["", f"### Bits per generated token / per (input+output) token ({split})", "", "| phi | codec | per generated token | per I/O token |", "|---|---|---|---|"]
        for lvl in LEVELS:
            for n in codecs.CODECS:
                L.append(f"| {lvl} | {n} | {_stats([x['bits_per_generated_token'] for x in sub if x['phi'] == lvl and x['codec'] == n])} | {_stats([x['bits_per_io_token'] for x in sub if x['phi'] == lvl and x['codec'] == n])} |")
    # category breakdown (sum over runs, per phi, per codec)
    L += ["", "## Breakdown by advice category (bits summed over all runs; A3 = ideal fractional bits)", ""]
    cats = ["framing", "request-static", "arrival", "stop", "membership", "position", "chunk", "logits", "escape"]
    L += ["| phi | codec | " + " | ".join(cats) + " | total |", "|---|---|" + "---|" * (len(cats) + 1)]
    for lvl in LEVELS:
        for n in codecs.CODECS:
            sub = [x for x in ok_rows if x["phi"] == lvl and x["codec"] == n]
            tot = {c: sum(x["marks"].get(c, 0) for x in sub) for c in cats}
            L.append(f"| {lvl} | {n} | " + " | ".join(f"{tot[c]:.1f}" for c in cats) + f" | {sum(tot.values()):.1f} |")
    # size ratios and declared bound
    L += ["", "## Size relative to the Program and to ordinary request I/O (per run, phi1, A2 / A3)", "",
          "| run | split | verdict | N | T | Program bytes | instances | I/O bytes (2 B/token) | A0 bits | A1 bits | A2 bits | A3 bits | A2 bits / Program bit | A2 bits / I/O bit | declared bound A1 (phi1) | declared bound A1 (phi2) |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    by = {}
    for x in ok_rows:
        by.setdefault(x["run_id"], {})[(x["phi"], x["codec"])] = x
    for rid, d in sorted(by.items()):
        x = d.get(("phi1", "A2")) or next(iter(d.values()))
        pb = x["program_bytes"] or 0
        g = lambda k: (d[k]["bits"] if k in d else "—")
        a2 = d.get(("phi1", "A2"), {}).get("bits")
        b1 = d.get(("phi1", "A1"), {}).get("declared_bound", {}).get("bits_A1_language", "—")
        b2 = d.get(("phi2", "A1"), {}).get("declared_bound", {}).get("bits_A1_language", "—")
        L.append(f"| `{rid}` | {x['split']} | {x['verdict']} | {x['n_requests']} | {x['n_steps']} | {pb} | {x['n_instances']} | {x['io_bytes_2B_per_token']} | {g(('phi1','A0'))} | {g(('phi1','A1'))} | {a2} | {g(('phi1','A3'))} | "
                 f"{(a2 / (8 * pb)) if a2 is not None and pb else float('nan'):.2e} | {(a2 / (8 * x['io_bytes_2B_per_token'])) if a2 is not None else float('nan'):.2e} | {b1} | {b2} |")
    bad = [x for x in rows if not x["roundtrip_ok"]]
    L += ["", f"## Points that did not round-trip / were not expressible: {len(bad)}", ""]
    for x in bad[:50]:
        L.append(f"- `{x['run_id']}` {x['phi']} {x['codec']}: {x['error']}")
    return "\n".join(L) + "\n"


def builder_validity(out: Path) -> dict[str, str]:
    """Per-run verdict of C-builder's check from a read-only snapshot of their pod output (`builder_check_snapshot.jsonl`, copied from
    their frozen runs-v5/corpus_results.jsonl, builder aot-build/eager/v5, 08:40Z): 'valid' iff their verdict starts with IDENTICAL, else their status; missing -> 'pending'."""
    snap = out / "builder_check_snapshot.jsonl"
    if not snap.exists():
        return {}
    v = {}
    for line in snap.read_text().splitlines():
        x = json.loads(line)
        verdict = str(x.get("verdict") or x.get("status") or "?")
        v[x["record"]] = "valid" if verdict.startswith("IDENTICAL") else ("unexpressible-v1" if verdict.startswith("UNEXPRESSIBLE") else verdict)
    return v


def report_tables(out: Path, builder_valid: dict[str, str] | None = None) -> str:
    """Per-run rows for REPORT.md from results.jsonl (+ inventory.jsonl): all four codecs at all phi levels, declared bounds, split,
    builder validity, generalization verdict; then the prediction-vs-coding decomposition."""
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    inv = {x["run_id"]: x for x in (json.loads(l) for l in (out / "inventory.jsonl").read_text().splitlines())} if (out / "inventory.jsonl").exists() else {}
    by: dict[str, dict] = {}
    for x in rows:
        by.setdefault(x["run_id"], {})[(x["phi"], x["codec"])] = x
    L = ["| run | family | profile | ctx | split | builder v1 | N | T | A0 | A1 φ0/φ1a/φ1/φ2 | A2 φ0/φ1a/φ1/φ2 | A3 φ0/φ1a/φ1/φ2 | declared φ1: A1 C1 / A2 clean / A2 k≤1 / A2 k≤5 / A2 C1 | escapes | verdict |",
         "|---|---|---|---|---|---|---:|---:|---:|---|---|---|---|---|---|"]
    for rid, d in sorted(by.items()):
        x = d[("phi1", "A2")]
        db = x["declared_bound"]
        ok_all = all(v["roundtrip_ok"] for v in d.values())
        v = lambda c: "/".join(str(d[(l, c)]["bits"]) for l in LEVELS)
        esc = sum(1 for l in LEVELS for c in ("A2",) if d[(l, c)]["marks"].get("escape", 0) > 2)  # more than clean + all_steps_match -> a step escaped
        bv = builder_valid.get(rid, "pending") if builder_valid else "—"
        n_esc = d[("phi1", "A2")]["bits"] - 1
        verdict = ("EXACT, 0 step escapes" if ok_all and n_esc == 0 else f"EXACT, {n_esc} escape bits at φ1" if ok_all else "ROUNDTRIP FAIL")
        L.append(f"| `{rid}` | {x['family']} | {x['profile']} | {x['context']} | {x['split']} | {bv} | {x['n_requests']} | {x['n_steps']} | {d[('phi0','A0')]['bits']} | {v('A1')} | {v('A2')} | {v('A3')} | "
                 f"{d[('phi1','A1')]['declared_bound']['bits_A1_language']} / {db.get('bits_A2_clean', '—')} / {db.get('bits_A2_k_escapes', {}).get('1', '—')} / {db.get('bits_A2_k_escapes', {}).get('5', '—')} / {db['bits_A2_language']} | {esc} | {verdict} |")
    # prediction vs coding: mean bits per run, A1 -> A2 (prediction) and A2 -> A3 (entropy coding)
    L += ["", "| phi | A0 mean | A1 mean | A2 mean | A3 mean | A0→A1 (remove derivable) | A1→A2 (prediction) | A2→A3 (entropy coding) | A3 ideal mean |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for lvl in LEVELS:
        m = {c: statistics.mean(x["bits"] for x in rows if x["phi"] == lvl and x["codec"] == c and x["bits"] is not None) for c in codecs.CODECS}
        ideal = statistics.mean(x["ideal_bits"] for x in rows if x["phi"] == lvl and x["codec"] == "A3" and x["ideal_bits"] is not None)
        L.append(f"| {lvl} | {m['A0']:.1f} | {m['A1']:.1f} | {m['A2']:.2f} | {m['A3']:.2f} | −{m['A0']-m['A1']:.1f} ({100*(m['A0']-m['A1'])/m['A0']:.2f}%) | −{m['A1']-m['A2']:.1f} ({100*(m['A1']-m['A2'])/m['A1']:.1f}% of A1) | −{m['A2']-m['A3']:.2f} ({100*(m['A2']-m['A3'])/m['A2']:.0f}% of A2) | {ideal:.2f} |")
    # held-out vs train at each level, A3
    L += ["", "| phi | split | runs | A2 bits/run mean | A3 bits/run mean | A3 ideal mean | max A3 |", "|---|---|---:|---:|---:|---:|---:|"]
    for lvl in LEVELS:
        for split in ("train", "heldout"):
            sub = [x for x in rows if x["phi"] == lvl and x["split"] == split and x["codec"] == "A3"]
            sub2 = [x for x in rows if x["phi"] == lvl and x["split"] == split and x["codec"] == "A2"]
            if sub:
                L.append(f"| {lvl} | {split} | {len(sub)} | {statistics.mean(x['bits'] for x in sub2):.2f} | {statistics.mean(x['bits'] for x in sub):.2f} | {statistics.mean(x['ideal_bits'] for x in sub):.3f} | {max(x['bits'] for x in sub)} |")
    return "\n".join(L) + "\n"


def extra_logs(paths: list[Path], out: Path, model_path: Path, rev: str, phi_overrides: dict | None = None) -> str:
    """Out-of-corpus generalization: bare capture logs (no spec.json / experiment.json, RunHeader workload only) encoded with the FROZEN
    corpus model.  Every log is held-out by construction (no per-log decoder change; a log the language cannot express is reported as such,
    not patched).  Writes extra_logs.jsonl + returns the markdown table."""
    m, msha = model.load(model_path)
    L = [f"| log | model | engine (seqs/budget) | N (logical) | T | check | A1 φ0/φ1a/φ1/φ2 | A2 φ0/φ1a/φ1/φ2 | A3 φ0/φ1a/φ1/φ2 | audit events | verdict |",
         "|---|---|---|---:|---:|---|---|---|---|---|---|"]
    recs = []
    for lp in paths:
        run_id = lp.parent.parent.name if lp.parent.name == "capture" else (lp.parent.name if lp.name == "log.jsonl.gz" else lp.name.split(".")[0])
        row = {"log": str(lp), "run_id": run_id, "code_rev": rev, "model_sha256": msha, "phi_overrides": dict(phi_overrides or {})}
        try:
            r = facts.load_log_only(lp, run_id)
            r["phi_overrides"] = dict(phi_overrides or {})  # operational facts the RunHeader does not record (e.g. num_gpu_blocks)
            h, audit = facts.extract_hint(r)
        except Exception as e:  # noqa: BLE001 -- the failure *is* the finding
            row.update({"expressible": False, "error": f"{type(e).__name__}: {e}"})
            recs.append(row)
            L.append(f"| `{run_id}` | | | | | | | | | | **NOT EXPRESSIBLE**: {e} |")
            continue
        chk = facts.check_record(r, h)
        phi0 = facts.phi_from_record(r)
        hs = facts.strip_private(h)
        bits, ok, enc_errors = {}, True, {}
        for lvl in LEVELS:
            phi = facts.phi_variant(phi0, h, lvl)
            for name in codecs.CODECS:
                try:  # a schedule outside a codec's alphabet is a *finding* (the language cannot express it), reported per point, never patched here
                    res = codecs.encode(name, phi, hs, m)
                    back = codecs.decode(name, phi, res.bits, m)
                except Exception as e:  # noqa: BLE001
                    enc_errors[f"{lvl}/{name}"] = f"{type(e).__name__}: {e}"
                    bits[f"{lvl}/{name}"] = None
                    ok = False
                    continue
                good = facts.canonical_json(back) == facts.canonical_json(hs)
                ok &= good
                bits[f"{lvl}/{name}"] = res.n_bits if good else None
        row.update({"expressible": not audit["unexpressible"], "unexpressible": audit["unexpressible"][:5], "events": audit["events"][:8], "check_ok": chk["ok"],
                    "model": phi0["model"], "engine": phi0["engine"], "n_requests": len(h["requests"]), "n_steps": len(h["steps"]), "roundtrip_ok": ok, "bits": bits,
                    "encode_errors": enc_errors, "requests": h["requests"]})
        recs.append(row)
        v = lambda c: "/".join(str(bits[f"{l}/{c}"]) for l in LEVELS)
        ev = "; ".join(sorted({x["kind"] for x in audit["events"]})) or "—"
        verdict = ("EXACT" if ok and chk["ok"] and not audit["unexpressible"] else
                   "**NOT EXPRESSIBLE (encode)**: " + "; ".join(sorted(set(enc_errors.values()))) if enc_errors else
                   "ROUNDTRIP FAIL" if not ok else "UNEXPRESSIBLE: " + "; ".join(sorted({x["kind"] for x in audit["unexpressible"]})) if audit["unexpressible"] else "CHECK FAIL")
        L.append(f"| `{run_id}` | {phi0['model']} | {phi0['engine']['max_num_seqs']}/{phi0['engine']['max_num_batched_tokens']} | {len(h['requests'])} | {len(h['steps'])} | {chk['ok']} | {v('A1')} | {v('A2')} | {v('A3')} | {ev} | {verdict} |")
    (out / "extra_logs.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in recs))
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["inventory", "run", "report", "extra"])
    ap.add_argument("--phi", action="append", default=[], help="extra: operational phi fact key=value applied to every log (e.g. num_gpu_blocks=20)")
    ap.add_argument("--records", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--train-regex", default=r"^B0-cov_b0_")
    ap.add_argument("--model", default=None)
    ap.add_argument("--logs", nargs="*", default=None, help="extra: bare capture logs (log.jsonl.gz) to encode with the frozen model")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rev = git_rev()
    if a.cmd == "report":
        (out / "report_tables.md").write_text(report_tables(out, builder_validity(out)))
        print(f"[evaluate] report tables -> {out / 'report_tables.md'}", file=sys.stderr)
        return 0
    if a.cmd == "extra":
        if not a.logs or not a.model:
            ap.error("extra needs --logs and --model")
        md = extra_logs([Path(p) for p in a.logs], out, Path(a.model), rev, {k: json.loads(v) for k, _, v in (s.partition("=") for s in a.phi)})
        (out / "extra_logs.md").write_text(f"# Out-of-corpus logs, frozen model `{Path(a.model).name}`, code `{rev}`\n\n" + md)
        print(md, file=sys.stderr)
        return 0
    if not a.records:
        ap.error("--records is required for inventory/run")
    recs = load_all(Path(a.records), a.only)
    print(f"[evaluate] {len(recs)} records loaded", file=sys.stderr)
    if a.cmd == "inventory":
        rows = inventory(recs, out)
        print(f"[evaluate] inventory: {len(rows)} runs -> {out / 'inventory.md'}", file=sys.stderr)
    else:
        run(recs, out, a.train_regex, Path(a.model) if a.model else None, rev)
        print(f"[evaluate] results -> {out / 'results.jsonl'}, {out / 'tables.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
