#!/usr/bin/env python3
"""[NIGHT 2026-09-12, E8/G4] SmolLM2-135M proving work versus context length -- the E8 fields of the measured table
(``out/night/table/SmolLM2-135M__<prompt>x<decode>.json``, schema of ``out/briefs/NIGHT-2026-09-12.md``).

For every configuration (prompt x decode; ``STEPS = decode - 1`` as ``record_v5`` names it: ``max_tokens = STEPS + 1``):
* the ONE request (the shipped ``record_v5.workload_v5`` rule: B0 corpus prompt 0 truncated for prompt <= 32, synthetic seeded token ids
  ``random.Random(1)`` for longer prompts -- D4's 6.5K practice; greedy, batch-invariant) and its ``request_id``;
* the population by E6's rule (``population_v3.expected_population`` on the frozen TINY instance with LP/STEPS substituted; MODELED by rule,
  equal to the MEASURED pinned population 7,254,531 at 4 x 9), N_R = 1, N_V, the kind table;
* E|Q| = s . N_V at the instance's s = 2^-14 and, for 6500 x 1600, the s that keeps E|Q| <= ~1,000 with the certified U from the shipped
  calculator (``veritor.analysis.certified.certified_bound`` on the population summary; W = NOUT x ceil(log2 V) bits);
* expected statements / proof objects under the registry's batching (``elem_builders.MAX_UNITS_BY_FAMILY``: 64 units per statement, one
  family per statement; GEMM coordinates n = 64 per statement -- R1/R2's strategy, PROPOSED for the exporter kind; whole-scan select one
  statement per VU; Gather: no approved checker -> uncovered, G1 (b)): E[ceil(X_k / m)] with X_k ~ Poisson(s n_k);
* proving: per-kind seconds MEASURED on the 4090 (``pod_prove_kinds.py`` records) x statements = MODELED prove_s; peak GPU memory MEASURED.

    PYTHONPATH=. .venv/bin/python e2e/demo_core/night/context_table.py --records out/night/e8/prove/records.jsonl --out out/night/table
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
for p in (REPO, REPO / "vllm-poc", REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from e2e import population_v3 as P  # noqa: E402
from e2e.coverage_v1 import builders as B  # noqa: E402
from e2e.coverage_v1.elem_builders import DEFAULT_MAX_UNITS, MAX_UNITS_BY_FAMILY  # noqa: E402
from record_v5 import config as C  # noqa: E402
from record_v5 import workload_v5 as W  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-135M"
INSTANCE = REPO / "out/eod/e6/tiny/protocol_instance.json"
CONFIGS = [(4, 9), (32, 16), (128, 64), (512, 128), (2048, 256), (6500, 1600)]
S_INSTANCE = Fraction(1, 2**14)
DELTA = Fraction(1, 10**12)
GEMM_CHUNK = 4000      # the shipped record_v5/obligations_v5.py: ALL selected GemmCoordinate units (every K) pooled, `--chunk 4000` coordinates per statement
ELEM_CHUNK = 50        # the shipped record_v5/obligations_v5.py ELEM_CHUNK: min(50, MAX_UNITS_BY_FAMILY) elementwise units per statement (integrator 23:05Z 2026-09-11)
GEMM_REC_CLASS = "gemm474"   # the RECORD-BACKED GEMM statement measured on the pod: the archived V5-32x15 chunk00 (474 coordinates, K = 576/1536 mix, request 7cc7c3bb)
TABLE_KINDS = {"MufuEx2Ftz_v1|primitive", "RsqrtApprox_v1|primitive", "MufuSqrtFtz_v1|primitive", "DivFullRcp_v1|primitive", "DivFullScaleA_v1|primitive", "Fa2InvSum_v1|primitive"}

# kind -> (measured batch name in the pod records, units per statement, note).  A statement of the class costs what the named batch cost.
def prove_class(kind: str) -> tuple[str, int, str]:
    fn = kind.split("|", 1)[0]
    if kind.startswith("GemmCoordinate_v1{"):
        return "GEMM-POOL", GEMM_CHUNK, "all GemmCoordinate kinds pooled into statements of <= 4,000 coordinates (obligations_v5 --chunk 4000); per-coordinate cost = the RECORD-BACKED 474-coordinate statement's MEASURED prove_s / 474 (K mix as sampled); the synthetic per-K classes K576/n64, K1536/n64 (word-per-leaf trees) are kept in per_kind as upper bounds"
    if kind == "TokenSelect_v1|whole-scan":
        return "record_2663944e_t3", 1, "select/record_2663944e_t3 (REAL TINY whole-scan statement, 49,152 logits; select guest 33892d5e)"
    if kind == "AmpereBF16TcDot16_v1|batch":
        return "tc-dot16-pv_n64", ELEM_CHUNK, "a56/tc-dot16-pv_n64 (4 tc_dot16 per unit; real V rows) -- upper bound for the 1-gate batch unit"
    if fn.startswith("DotBf16_v1"):
        return "dot-bf16-K128_n64", ELEM_CHUNK, "a56/dot-bf16-K128_n64 (K = 128: 8 tc gates per unit; the K = 64 call has 4) -- upper bound"
    if kind in TABLE_KINDS:
        return "FIT:6940000", ELEM_CHUNK, "MUFU table kinds: 50 units x 135k cycles (the 16 KiB table-row leaf hashed in the guest; E7 D23) + ~190k fixed = 6.9M cycles through the MEASURED machine fit prove_s = a + b x Mcycles"
    if fn in ("F32MulFtz_v1", "F32AddFtz_v1", "F32SubFtz_v1", "F32FmaFtz_v1", "F32FmaSubFtz_v1", "F32Max_v1", "GuardNegInfZero_v1"):
        return "chunk00@ftz", ELEM_CHUNK, "ftz/chunk00 (E7/E4's 64-unit veritor.ftz-f32@1 conformance statement, PROVED under the consolidated guest 666e937f; synthetic word-per-leaf trees, 6.8 Mcycles) -- upper bound for a record-backed 64-unit FTZ batch (~1.7 s, the elem class)"
    if kind.startswith("GatherBf16x49152"):
        return "", 0, "no approved checker (veritor.embed-gather@1 unresolved; G1 (b)) -- NOT PROVABLE tonight"
    return "core-silu-add-64u-chunk00", ELEM_CHUNK, "elem/core-silu-add-64u-chunk00 (64 one-gate units, elem-bf16; REAL CORE statement) per 50-unit statement (ELEM_CHUNK) -- the one-gate elem/reduce class (upper bound: 64 measured vs 50 shipped)"


def poisson_ceil_expect(lam: float, m: int) -> float:
    """E[ceil(X / m)] for X ~ Poisson(lam): the expected number of m-unit statements a kind contributes.  Exact (tail sum) for lam <= 2,000;
    above that lam / m + 1/2 (the ceil's mean excess; the Poisson spread is negligible against lam / m)."""

    if lam <= 0 or m <= 0:
        return 0.0
    if lam > 2_000:
        return lam / m + 0.5
    # E[ceil(X/m)] = sum_{j>=1} P(X > (j-1) m); the Poisson pmf is accumulated once
    pmf, term = [], math.exp(-lam)
    i = 0
    while True:
        pmf.append(term)
        i += 1
        term *= lam / i
        if term < 1e-15 and i > lam:
            break
    cdf = 0.0
    total = 0.0
    for k, pk in enumerate(pmf):
        cdf += pk
        # P(X > k) = 1 - cdf(k); summed at every k that is a multiple of m (k = (j-1) m, j >= 1)
        if k % m == 0:
            total += max(0.0, 1.0 - cdf)
    return total


def poisson_sf(lam: float, k: int) -> float:
    """P(X > k), X ~ Poisson(lam)."""

    if k < 0:
        return 1.0
    # sum_{i<=k} e^-lam lam^i / i!  in log space
    acc, term = 0.0, math.exp(-lam)
    for i in range(k + 1):
        if i > 0:
            term *= lam / i
        acc += term
        if term < 1e-300 and i > lam:
            break
    return max(0.0, 1.0 - acc)


def population_at(lp: int, steps: int) -> dict[str, Any]:
    inst = json.loads(INSTANCE.read_text())
    i = copy.deepcopy(inst)
    idn = i["program"]["identity"]
    idn["LP"], idn["STEPS"] = lp, steps
    idn["function"] = f"Serve_v1{{C={{{idn['config_sha12']}}},LP={lp},STEPS={steps}}}"
    return P.expected_population(i)


def request_at(lp: int, decode: int) -> tuple[dict[str, Any], str]:
    doc = C.default_doc("TINY-B0-4x8-1req", B_ex=0, f_policy=C.F_POLICIES[0])
    doc["workload"].update({"prompt_len": lp, "max_tokens": decode, "requests": 1})
    doc["epoch"]["requests"] = 1
    cfg = C.parse(doc)
    wl = W.make_workload(cfg)
    return wl["requests"][0], wl["request_origins"][0]["origin"]


def certified_u(ep: dict[str, Any], s: Fraction) -> dict[str, Any]:
    from veritor.analysis.certified import Bank, Family, Summary, certified_bound
    from veritor.core.policy import VerificationPolicy

    nv = int(ep["vu_count_per_ru"])
    W_bits = int(ep["nout"]) * (49152 - 1).bit_length()
    fams = tuple(Family(k["kind"], int(k["count"]), min(int(k["declared_bits"]), W_bits)) for k in ep["kinds"])
    cb = certified_bound(Summary(fams, (Bank("ru", 1, nv, W_bits),), W_bits), VerificationPolicy(q=Fraction(1), s=s), DELTA, allowance=0, certify_all=False)
    return {"s": f"1/{s.denominator}", "U_bits": cb.bits, "precap_bits": cb.precap_bits, "W_bits": W_bits, "capped": cb.capped, "U_equals_W": cb.bits == W_bits,
            "delta": str(DELTA), "q": "1", "N_R": 1, "B_ex": 0,
            "calculator": "veritor.analysis.certified.certified_bound (Appendix A; Summary = one family per kind (count, min(declared bits, W)), one bank: N_R = 1 RU of N_V units behind W = NOUT x 16 bits)"}


def machine_fit(measured: dict[str, dict[str, Any]]) -> dict[str, float]:
    """prove_s = a + b x Mcycles over the MEASURED (cycles, prove_seconds) points: a = the smallest measured prove_s (the one-unit floor),
    b = the median marginal rate (prove_s - a) / Mcycles over the points above 5 Mcycles (a plain least-squares line has a negative intercept
    here, dominated by the 947 Mcycle point)."""

    pts = [(float(r["execute"]["total_cycles"]) / 1e6, float(r["prove"]["prove_seconds"])) for k, r in measured.items()
           if not k.startswith("_") and (r.get("execute") or {}).get("total_cycles") and (r.get("prove") or {}).get("prove_seconds")]
    if len(pts) < 2:
        return {}
    a = min(y for _, y in pts)
    big = sorted((y - a) / x for x, y in pts if x > 5.0)
    if not big:
        return {}
    b = big[len(big) // 2]
    resid = max(abs(y - (a + b * x)) / y for x, y in pts)
    return {"a_s": a, "b_s_per_Mcycle": b, "points": len(pts), "max_rel_residual": resid, "cycles_range_M": [min(x for x, _ in pts), max(x for x, _ in pts)]}


def load_measured(records: Path | None) -> dict[str, dict[str, Any]]:
    """batch name -> the best (min prove_seconds over reps) MEASURED record; plus gpu peak and context."""

    out: dict[str, dict[str, Any]] = {}
    if records is None or not records.exists():
        return out
    for line in records.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("schema") == "veritor/night-e8/pod-context/v1":
            out.setdefault("_context", r)
            continue
        if r.get("schema") != "veritor/night-e8/prove-record/v1" or not (r.get("prove") or {}).get("ok"):
            continue
        name = r["name"]
        if name.startswith("n") and name[1:].isdigit():  # gemm/coordinate/<family>/K<k>/n<size>
            name = f"{Path(r['batch']).parent.name}/{name}"
        if "/ftz/" in str(r.get("batch", "")):  # E7's FTZ chunks share chunkNN names with the elem/e7 batches
            name = f"{name}@ftz"
        prev = out.get(name)
        if prev is None or float(r["prove"]["prove_seconds"]) < float(prev["prove"]["prove_seconds"]):
            out[name] = r
    return out


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    except Exception:
        return "?"


def field(value: Any, label: str, where: str = "", **extra: Any) -> dict[str, Any]:
    d = {"value": value, "label": label}
    if where:
        d["where"] = where
    d.update(extra)
    return d


def row(lp: int, decode: int, measured: dict[str, dict[str, Any]], s_alt: Fraction | None) -> dict[str, Any]:
    steps = decode - 1
    ep = population_at(lp, steps)
    nv = int(ep["vu_count_per_ru"])
    req, origin = request_at(lp, decode)
    ctx = measured.get("_context") or {}
    fit = machine_fit(measured)
    gpu = ctx.get("gpu") or "NVIDIA GeForce RTX 4090"
    pod = "night-e8-prove (r0dtdsgz281ux8, RTX 4090 24 GB)"
    kinds_out = []
    total_stmts = 0.0
    total_prove = 0.0
    prove_modeled_complete = True
    peak_mib = 0
    uncovered_eq = 0.0
    gemm_eq = 0.0
    gemm_rec = measured.get(GEMM_REC_CLASS)
    for k in sorted(ep["kinds"], key=lambda k: -int(k["count"])):
        n = int(k["count"])
        eq = float(S_INSTANCE) * n
        cls, m, note = prove_class(k["kind"])
        entry: dict[str, Any] = {"kind": k["kind"], "count": n, "E_Q": round(eq, 3), "family": getattr(B.BUILDERS.get(k["kind"]), "family", None), "class": cls, "units_per_statement": m, "class_note": note}
        if cls == "GEMM-POOL":
            gemm_eq += eq
            kk = "K576" if "K=576" in k["kind"] else "K1536"
            syn = measured.get(f"{kk}/n64")
            entry["statements"] = "pooled (see GemmCoordinate pool)"
            entry["synthetic_upper_bound"] = {"class": f"{kk}/n64", "prove_s_per_64_coordinates": round(float(syn["prove"]["prove_seconds"]), 3) if syn else None,
                                              "note": "synthetic word-per-leaf trees (gemm_synth.py): K + K + 1 leaf openings per coordinate; NOT the record's leaf layout"}
            if gemm_rec is not None:
                entry["prove_s_per_coordinate"] = round(float(gemm_rec["prove"]["prove_seconds"]) / 474, 4)
                entry["prove_s_label"] = "MEASURED per coordinate: the RECORD-BACKED 474-coordinate statement (archived V5-32x15 chunk00, request 7cc7c3bb, K mix as sampled) on the 4090 / 474"
            kinds_out.append(entry)
            continue
        if m == 0:
            entry["statements"] = None
            entry["status"] = "NOT PROVABLE (no approved checker)"
            uncovered_eq += eq
        else:
            st = poisson_ceil_expect(eq, m)
            entry["statements"] = round(st, 4)
            total_stmts += st
            rec = measured.get(cls)
            if cls.startswith("FIT:") and fit:
                cyc = int(cls.split(":")[1])
                ps = fit["a_s"] + fit["b_s_per_Mcycle"] * cyc / 1e6
                entry["prove_s_per_statement"] = round(ps, 3)
                entry["prove_s_label"] = f"MODELED: machine fit over {fit['points']} MEASURED points (a = {fit['a_s']:.2f} s, b = {fit['b_s_per_Mcycle']:.3f} s/Mcycle) at {cyc / 1e6:.1f} Mcycles"
                entry["cycles_per_statement"] = cyc
                total_prove += ps * st
            elif rec is None:
                entry["prove_s_per_statement"] = None
                entry["prove_s_label"] = "NOT RUN (kind class not yet measured)"
                prove_modeled_complete = False
            else:
                ps = float(rec["prove"]["prove_seconds"])
                entry["prove_s_per_statement"] = round(ps, 3)
                entry["prove_s_label"] = "MEASURED per statement class on the 4090 (host prove_seconds, best of reps, setup excluded)"
                entry["cycles_per_statement"] = (rec.get("execute") or {}).get("total_cycles")
                total_prove += ps * st
                if rec.get("gpu_peak_used_mib"):
                    peak_mib = max(peak_mib, int(rec["gpu_peak_used_mib"]))
        kinds_out.append(entry)
    # the GEMM coordinate pool: statements of <= 4,000 coordinates (all K together), cost per coordinate from the record-backed measurement
    if gemm_eq > 0:
        st = poisson_ceil_expect(gemm_eq, GEMM_CHUNK)
        total_stmts += st
        pool: dict[str, Any] = {"kind": "GemmCoordinate pool (all K; obligations_v5 --chunk 4000)", "E_Q": round(gemm_eq, 3), "units_per_statement": GEMM_CHUNK, "statements": round(st, 4), "class": GEMM_REC_CLASS}
        if gemm_rec is not None:
            per_coord = float(gemm_rec["prove"]["prove_seconds"]) / 474
            pool["prove_s_per_coordinate"] = round(per_coord, 4)
            pool["prove_s_per_statement"] = round(per_coord * min(GEMM_CHUNK, gemm_eq / max(st, 1e-9)), 3)
            pool["prove_s_label"] = ("MODELED = E|Q|_gemm x MEASURED per-coordinate seconds (record-backed 474-coordinate statement: "
                                     f"{gemm_rec['execute']['total_cycles']:,} cycles, {float(gemm_rec['prove']['prove_seconds']):.1f} s on the 4090); linear in coordinates, K mix as sampled at 32x15")
            pool["cycles_per_coordinate"] = round(int(gemm_rec["execute"]["total_cycles"]) / 474)
            total_prove += per_coord * gemm_eq
            if gemm_rec.get("gpu_peak_used_mib"):
                peak_mib = max(peak_mib, int(gemm_rec["gpu_peak_used_mib"]))
        else:
            pool["prove_s_per_statement"] = None
            pool["prove_s_label"] = "NOT RUN (record-backed GEMM statement not yet measured)"
            prove_modeled_complete = False
        kinds_out.append(pool)
    u14 = certified_u(ep, S_INSTANCE)
    alt = certified_u(ep, s_alt) if s_alt else None
    doc = {
        "schema": "veritor/night/table-row/v1", "written_by": "E8 (G4) -- e2e/demo_core/night/context_table.py", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": MODEL, "architecture": "llama (LlamaForCausalLM; 30 layers, hidden 576, intermediate 1536, 9 heads / 3 kv heads, V = 49,152)",
        "prompt_len": lp, "max_tokens": decode, "STEPS": steps, "program_identity": ep["program_identity"]["function"],
        "request": {"label": "PROPOSED (the shipped record_v5.workload_v5 rule; one request, greedy, batch-invariant, no prefix caching)", "request_id": req["request_id"], "origin": origin,
                    "note": "prompt <= 32: B0 corpus prompt 0 truncated (4 x 9 is the TINY request 2663944e...); prompt > 32: synthetic seeded token ids (D4's 6.5K practice), a labelled difference from natural text"},
        "N_R": field(1, "MODELED", "one request = one RU (P1/cu=whole)"),
        "N_V": field(nv, "MODELED (E6's rule: population_v3.expected_population on the TINY instance with LP/STEPS substituted; MEASURED-equal to the pinned 7,254,531 at 4 x 9)", "laptop"),
        "kinds": len(ep["kinds"]), "kind_table_digest": ep.get("kind_table_digest"),
        "serve_s": field(None, "NOT RUN"), "serve_gpu": field(None, "NOT RUN"), "reconstruction_s": field(None, "NOT RUN (E1's rows)"), "reconstruction_hw": field(None, "NOT RUN"),
        "interior_words": field(None, "NOT RUN (E2's field)"), "interior_leaves": field(None, "NOT RUN (E2's field)"), "trees": field(None, "NOT RUN (E2's field)"),
        "commit_s": field(None, "NOT RUN (E2's field)"), "commit_peak_rss_gb": field(None, "NOT RUN (E2's field)"),
        "s": field("1/16384", "the instance's s = 2^-14 (E6 pin, L1's value)"),
        "Q": field(round(float(S_INSTANCE) * nv, 2), "MODELED E|Q| = s x N_V (q = 1, N_R = 1); a run draws |Q| ~ Binomial(N_V, s)", "laptop"),
        "Q_uncovered_expected": field(round(uncovered_eq, 3), "MODELED E|Q| landing on kinds with no approved checker tonight (Gather; G1 (b))"),
        "statements": field(round(total_stmts, 2), "MODELED: sum over kinds of E[ceil(X_k / m_k)], X_k ~ Poisson(s n_k), m_k = the shipped obligations_v5 batching (elementwise ELEM_CHUNK = 50 per family; ALL GemmCoordinate units pooled at 4,000 per statement; whole-scan select 1)"),
        "proof_objects": field(round(total_stmts, 2), "MODELED = statements (one proof object per statement)"),
        "prove_s": field(round(total_prove, 2) if prove_modeled_complete else None, ("MODELED = sum_k statements_k x MEASURED per-kind seconds (4090)" if prove_modeled_complete else "NOT RUN (per-kind measurement incomplete)"), pod),
        "prove_gpu": field(gpu, "MEASURED (per-kind statements proved on this GPU)", pod),
        "prove_peak_mem_gb": field(round(peak_mib / 1024, 2) if peak_mib else None, "MEASURED max over the proved statement classes (nvidia-smi memory.used peak, 0.2 s sampling)" if peak_mib else "NOT RUN", pod),
        "verify_live_s": field(None, "NOT RUN"), "reverify_s": field(None, "NOT RUN"), "reverify_peak_rss_mb": field(None, "NOT RUN"),
        "verdict_live": field(None, "NOT RUN"), "verdict_reverify": field(None, "NOT RUN"), "verdict_t1": field(None, "NOT RUN"),
        "pod_cost_usd": field(None, "NOT RUN (per-kind measurement pod shared across the six rows: see out/night/e8/pods.md)"),
        "certified_U_at_s": u14, "certified_U_at_alt_s": alt,
        "revision": git_rev(), "record_path": None,
        "per_kind": kinds_out,
        "measurement_context": {k: ctx.get(k) for k in ("utc", "host_info", "gpu", "gpu_idle_used_mib", "sp1_prover") if k in ctx},
        "machine_fit": fit,
        "per_process_setup_s": field(round(min((float(r["prove"]["setup_seconds"]) for k, r in measured.items() if not k.startswith("_") and (r.get("prove") or {}).get("setup_seconds")), default=0.0), 2) if fit else None,
                                     "MEASURED: the GPU prover's per-process setup (one host process per proof, as the live driver runs it); NOT in prove_s; wall = statements x (setup + prove)"),
    }
    return doc


E8_OWNED = ("s", "Q", "Q_uncovered_expected", "statements", "proof_objects", "prove_s", "prove_gpu", "prove_peak_mem_gb", "certified_U_at_s", "certified_U_at_alt_s",
            "per_kind", "measurement_context", "request", "program_identity", "STEPS")
IDENTITY = ("model", "architecture", "prompt_len", "max_tokens")


def merge_row(target: Path, doc: dict[str, Any]) -> None:
    """E8's fields into the shared table file (out/night/table/<model>__<p>x<d>.json).  The integrator's zk/campaign/night_table.py unions the
    row files of every worktree field-wise (latest utc wins), so this file carries ONLY what E8 measured or modeled: the E8-owned fields, the
    MEASURED serve fields, E8's N_V as ``N_V_e8`` (never ``N_V`` itself: the population owner's / E2's form must not be overridden by a newer
    utc), plus the identity fields.  Other agents' fields already in the file (a merged worktree) are kept."""

    target.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(target.read_text()) if target.exists() else {}
    for k in IDENTITY:
        cur.setdefault(k, doc[k])
    for k in E8_OWNED:
        if k in doc:
            cur[k] = doc[k]
    for k in ("serve_s", "serve_gpu"):  # E8's MEASURED serve (or the lane's refusal) unless someone else measured it
        if k in doc and not ((cur.get(k) or {}).get("label") or "").startswith("MEASURED") and (doc[k].get("label") or "") != "NOT RUN":
            cur[k] = doc[k]
    cur["N_V_e8"] = {**doc["N_V"], "note": "E8's form: E6's frozen TINY instance rule (whole-scan token selection, compiled literals) with LP/STEPS substituted, N_R = 1 request"}
    wb = cur.get("written_by")
    cur["written_by"] = sorted(set((wb if isinstance(wb, list) else ([wb] if wb else [])) + [doc["written_by"]]))
    cur["utc"] = doc["utc"]
    target.write_text(json.dumps(cur, indent=1) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records", default=None, help="pod_prove_kinds.py records.jsonl (MEASURED per-kind seconds)")
    ap.add_argument("--out", default=str(REPO / "out/night/e8/table"), help="E8's rows (E8 fields only)")
    ap.add_argument("--merge-into", default=None, help="the shared table dir (out/night/table): E8's fields are MERGED into the existing row files field by field "
                    "(E8 owns the proving/sampling fields; N_R / N_V / kinds are setdefault -- E2's form stays if already written; E8's N_V is added as N_V_e8)")
    ap.add_argument("--configs", default=",".join(f"{p}x{d}" for p, d in CONFIGS))
    ap.add_argument("--serve", default=None, help="out/night/e8/serve/serve_measured.json (MEASURED serve_v5 recorded-pass seconds per shape -> serve_s / serve_gpu)")
    a = ap.parse_args()
    serve = json.loads(Path(a.serve).read_text()) if a.serve else {"shapes": {}}
    measured = load_measured(Path(a.records) if a.records else None)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for spec in a.configs.split(","):
        lp, d = (int(x) for x in spec.split("x"))
        s_alt = Fraction(1, 2**27) if (lp, d) == (6500, 1600) else None
        doc = row(lp, d, measured, s_alt)
        sv = serve["shapes"].get(f"{lp}x{d}")
        if (serve.get("refused") or {}).get(f"{lp}x{d}"):
            doc["serve_s"] = field(None, "NOT RUN: " + serve["refused"][f"{lp}x{d}"])
        if sv:
            doc["serve_s"] = field(round(float(sv["recorded_pass_wall_s"]), 3), "MEASURED: serve_v5 recorded pass (matched C, one request; recording on/off "
                                   f"{[round(x, 3) for x in sv['recording_on_wall_s']]}/{[round(x, 3) for x in sv['recording_off_wall_s']]}, overhead x{sv['production_overhead_ratio']:.3f}; "
                                   f"engine build {sv['engine_build_s']:.1f} s excluded)", sv["hardware"])
            doc["serve_gpu"] = field(sv["hardware"].split(" (")[0], "MEASURED", sv["hardware"])
        path = out / f"SmolLM2-135M__{lp}x{d}.json"
        path.write_text(json.dumps(doc, indent=1) + "\n")
        if a.merge_into:
            merge_row(Path(a.merge_into) / path.name, doc)
        print(f"{path.name}: N_V={doc['N_V']['value']:,} E|Q|={doc['Q']['value']} statements={doc['statements']['value']} prove_s={doc['prove_s']['value']} ({doc['prove_s']['label'][:40]}) "
              f"U@2^-14={doc['certified_U_at_s']['U_bits']} bits (W={doc['certified_U_at_s']['W_bits']})" + (f" alt s={doc['certified_U_at_alt_s']['s']} U={doc['certified_U_at_alt_s']['U_bits']}" if doc["certified_U_at_alt_s"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
