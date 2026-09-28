"""The exhaustive WP1 comparison: one recorded pod run (stage ``wp1.exhaustive``), prints counts, writes a JSON summary.

    python exhaustive_driver.py --out $RESEARCH_RUN_DIR [--harness PATH] [--random N] [--procs P] [--pytest]

Steps (the letters are the lane's test plan):
  0  the fast pytest suite of this directory (``--pytest``; includes the z3 proof of D3 when z3 is importable)
  a  every BF16 encoding x three multipliers: hawkeye_close vs silicon decode, non-finite flagged alike
  b  every 256 x 256 significand product, both signs
  c  the boundary groups, three forms
  d  the transition unit: every structured case, N random cases, every all-finite hardware record (golden, A100, 4090)
  e  the compiled pinned simulator (``--harness``) on every case of (d), bit for bit against both Python evaluators
  f  D3: every (E, v) with E in [-132, 254] and v < 2^12, all three forms; every E x bit length x patterns
  g  the adapter: every subnormal significand and 3,072 mantissa patterns per exponent -- left inverse of operator float and
     fixed point of the wrapper's final group_sum

The summary is ``<out>/wp1_exhaustive.json``; ``<out>/result.json`` is the research envelope (validation passed iff every
mismatch counter is zero).  Stdlib only besides ``verity`` and ``verity_numerical`` on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from verity.ml.circuits.silicon import silicon as S
from verity.ml.circuits.silicon.errors import InvalidArtifact
from verity_numerical.reference import hawkeye_close as H
from wp1_cases import (
    Case,
    boundary_groups,
    canon,
    canon_term,
    golden_records,
    random_cases,
    realise,
    special_records,
    structured_cases,
    term_of,
)

HERE = Path(__file__).resolve().parent
PIPE = S.AMPERE_BF16_M16N8K16


# --- workers -------------------------------------------------------------------------------------------------------------------


def eval_cases(cases: list[Case]) -> list[str]:
    """Per case: ``acc a b sil close wrapper d`` (hex words; ``-`` when absent; ``sil`` is ``rejected`` if silicon raised)."""

    out = []
    for c in cases:
        try:
            sil = f"{S.tc_dot(PIPE, c.acc, list(c.a), list(c.b)):08x}"
        except InvalidArtifact:
            sil = "rejected"
        close = H.tc_dot16(c.acc, c.a, c.b)
        wrapper = f"{H.wrapper_element(c.a, c.b):08x}" if c.acc & 0x7FFFFFFF == 0 else "-"
        d = "-" if c.d is None else f"{c.d:08x}"
        a = "".join(f"{w:04x}" for w in c.a)
        b = "".join(f"{w:04x}" for w in c.b)
        out.append(f"{c.acc:08x} {a} {b} {sil} {close:08x} {wrapper} {d} {c.source}")
    return out


def eval_random_shard(spec: tuple[int, int]) -> list[str]:
    n, seed = spec
    return eval_cases(random_cases(n, seed))


def d3_shard(E: int) -> tuple[int, int]:
    compared = mismatches = 0
    for v in range(1 << 12):
        for sign in (False, True):
            g = realise(sign, E, v)
            a = canon(H.group_sum(g))
            b = canon(H.group_sum_closed_form(g))
            c = canon_term(S.group_sum([term_of(t) for t in g], width=PIPE.width, zero_exponent=PIPE.zero_exponent))
            d = canon(H.normalise_closed_form(sign, E, v))
            compared += 1
            if not (a == b == c == d):
                mismatches += 1
    return compared, mismatches


def d3_structure_shard(E: int) -> tuple[int, int]:
    rng = random.Random(E)
    compared = mismatches = 0
    for l in range(0, 31):
        vs = [0] if l == 0 else {1 << (l - 1), (1 << l) - 1, (1 << (l - 1)) | 1, rng.getrandbits(l) | (1 << (l - 1))}
        for v in vs:
            for sign in (False, True):
                g = realise(sign, E, v)
                a = canon(H.group_sum(g))
                b = canon(H.group_sum_closed_form(g))
                c = canon_term(S.group_sum([term_of(t) for t in g], width=PIPE.width, zero_exponent=PIPE.zero_exponent))
                compared += 1
                if not (a == b == c):
                    mismatches += 1
    return compared, mismatches


def adapter_shard(spec: tuple[int, int, int]) -> tuple[int, int]:
    """(exponent, lo, hi): every significand in [lo, hi) at that exponent: decode(float(g)) == g, group_sum([g]) == g, silicon agrees."""

    exponent, lo, hi = spec
    compared = mismatches = 0
    for m in range(lo, hi):
        for sign in (False, True):
            g = H.Gfloat(sign, exponent, m)
            word = H.gfloat_to_float_bits(g)
            ok = H.fp32_to_gfloat(word) == g and H.group_sum([g]) == g and canon_term(S.fp32_term(word)) == canon(g)
            ok = ok and S.pack_fp32(term_of(g)) == word
            compared += 1
            if not ok:
                mismatches += 1
    return compared, mismatches


# --- steps ----------------------------------------------------------------------------------------------------------------------


def step_a() -> dict:
    finite = nonfinite = mismatches = 0
    for word in range(1 << 16):
        if H.is_finite_bf16(word):
            finite += 1
            for other in (0x3F80, 0x0080, 0x0001):
                if canon(H.multiply_bfloat16_to_gfloat(word, other)) != canon_term(S.bf16_product(word, other)):
                    mismatches += 1
        else:
            nonfinite += 1
            try:
                S.bf16_product(word, 0x3F80)
                mismatches += 1
            except InvalidArtifact:
                pass
            try:
                H.tc_dot16(0, [word] + [0] * 15, [0x3F80] * 16)
                mismatches += 1
            except H.NonFinite:
                pass
    return {"encodings": 1 << 16, "finite": finite, "non_finite": nonfinite, "comparisons": 3 * finite + 2 * nonfinite, "mismatches": mismatches}


def step_b() -> dict:
    compared = mismatches = 0
    for sa in range(256):
        a = (sa & 0x7F) | ((0x7F if sa & 0x80 else 0) << 7)
        for sb in range(256):
            b = (sb & 0x7F) | ((0x7F if sb & 0x80 else 0) << 7)
            for sign in (0, 0x8000):
                got = H.multiply_bfloat16_to_gfloat(a ^ sign, b)
                compared += 1
                if canon(got) != canon_term(S.bf16_product(a ^ sign, b)) or (got.significand and got.significand != (sa * sb) << 9):
                    mismatches += 1
    return {"compared": compared, "mismatches": mismatches}


def step_c() -> dict:
    mismatches = 0
    for name, group in boundary_groups():
        a = canon(H.group_sum(group))
        b = canon(H.group_sum_closed_form(group))
        c = canon_term(S.group_sum([term_of(g) for g in group], width=PIPE.width, zero_exponent=PIPE.zero_exponent))
        if not (a == b == c):
            mismatches += 1
    return {"groups": len(boundary_groups()), "mismatches": mismatches}


def tally(lines: list[str]) -> dict:
    """The (d) counters over the evaluated lines, by top-level source."""

    per: dict[str, Counter] = {}
    for line in lines:
        acc, a, b, sil, close, wrapper, d, source = line.split(" ")
        c = per.setdefault(source.split("/")[0], Counter())
        c["cases"] += 1
        if sil == "rejected":
            c["silicon_rejected"] += 1
            continue
        if d != "-":
            c["hardware_compared"] += 1
            if sil != d:
                c["silicon_ne_hardware"] += 1
        if H.is_finite_fp32(int(sil, 16)):
            c["finite_results"] += 1
            if close != sil:
                c["close_ne_silicon_on_finite_result"] += 1
        else:
            c["saturating_results"] += 1
            if close == sil:
                c["close_eq_silicon_on_saturating_result"] += 1
        if wrapper != "-":
            c["wrapper_checked"] += 1
            if wrapper != close:
                c["wrapper_ne_close"] += 1
    return {k: dict(v) for k, v in sorted(per.items())}


def step_e(lines: list[str], harness: str, out: Path) -> dict:
    inputs = out / "cases.txt"
    with open(inputs, "w") as f:
        for line in lines:
            acc, a, b = line.split(" ", 3)[:3]
            f.write(f"{acc} {a} {b}\n")
    t0 = time.time()
    with open(inputs) as f:
        proc = subprocess.run([harness], stdin=f, capture_output=True, text=True, check=True)
    got = proc.stdout.splitlines()
    c = Counter(lines=len(got), seconds=round(time.time() - t0, 1), cpp_matmul_checked=0, cpp_adapter_ne_close=0,
                cpp_finalised_ne_cpp_adapter=0, cpp_matmul_ne_cpp_adapter=0, cpp_matmul_ne_python_wrapper=0,
                cpp_adapter_ne_silicon_on_finite_result=0)  # zeros written out explicitly so the record says so
    if len(got) != len(lines):
        c["line_count_mismatch"] = 1
    for line, cpp in zip(lines, got):
        _, _, _, sil, close, wrapper, _, _ = line.split(" ")
        adapter, finalised, matmul = cpp.split(" ")
        if adapter != close:
            c["cpp_adapter_ne_close"] += 1
        if finalised != adapter:
            c["cpp_finalised_ne_cpp_adapter"] += 1
        if matmul != "-":
            c["cpp_matmul_checked"] += 1
            if matmul != adapter:
                c["cpp_matmul_ne_cpp_adapter"] += 1
            if wrapper != "-" and matmul != wrapper:
                c["cpp_matmul_ne_python_wrapper"] += 1
        if sil != "rejected" and H.is_finite_fp32(int(sil, 16)) and adapter != sil:
            c["cpp_adapter_ne_silicon_on_finite_result"] += 1
    return dict(c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--harness", default=None, help="compiled hawkeye_harness (step e); omitted = step e not run")
    ap.add_argument("--random", type=int, default=1_000_000)
    ap.add_argument("--procs", type=int, default=max(1, min(64, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--pytest", action="store_true", help="run the fast suite first (needs pytest; z3 for the SMT proof)")
    ap.add_argument("--quick", action="store_true", help="smoke test: a few shards of (f) and (g) only")
    args = ap.parse_args()
    exponents = [H.E_ZERO, -127, -126, -125, 0, 127, 254] if args.quick else list(range(H.E_ZERO, 255))
    sub_shards = 2 if args.quick else 64
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    summary: dict = {
        "schema": "wp1/exhaustive-summary/v0",
        "environment": {
            "hostname": platform.node(),
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
            "procs": args.procs,
        },
        "steps": {},
    }
    steps = summary["steps"]

    def report(name: str, value: dict) -> None:
        steps[name] = value
        print(name, json.dumps(value), flush=True)

    if args.pytest:
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(HERE)],
            capture_output=True,
            text=True,
        )
        tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip()[-300:]
        (out / "pytest.txt").write_text(proc.stdout + proc.stderr)
        report("0_pytest", {"returncode": proc.returncode, "summary": tail, "seconds": round(time.time() - t0, 1)})

    report("a_bf16_encodings", step_a())
    report("b_significand_products", step_b())
    report("c_boundary_groups", step_c())

    with Pool(args.procs) as pool:
        # (d) structured + fixtures + random
        t0 = time.time()
        fixed = structured_cases() + golden_records() + special_records("a100") + special_records("4090")
        chunk = 2000
        lines: list[str] = []
        for part in pool.map(eval_cases, [fixed[i : i + chunk] for i in range(0, len(fixed), chunk)]):
            lines.extend(part)
        shards = max(1, args.random // 5000)
        per = args.random // shards
        for part in pool.map(eval_random_shard, [(per, 20260921 + i) for i in range(shards)]):
            lines.extend(part)
        d = tally(lines)
        d["_total"] = {
            "cases": len(lines),
            "seconds": round(time.time() - t0, 1),
            "close_ne_silicon_on_finite_result": sum(v.get("close_ne_silicon_on_finite_result", 0) for v in d.values()),
            "silicon_ne_hardware": sum(v.get("silicon_ne_hardware", 0) for v in d.values()),
            "wrapper_ne_close": sum(v.get("wrapper_ne_close", 0) for v in d.values()),
            "silicon_rejected": sum(v.get("silicon_rejected", 0) for v in d.values()),
        }
        report("d_transition_unit", d)
        with gzip.open(out / "cases_evaluated.txt.gz", "wt") as f:
            f.write("\n".join(lines) + "\n")

        # (e) the compiled simulator on the same lines
        if args.harness:
            report("e_compiled_simulator", step_e(lines, args.harness, out))
        del lines

        # (f) D3
        t0 = time.time()
        exhaustive = pool.map(d3_shard, exponents)
        structure = pool.map(d3_structure_shard, exponents)
        report(
            "f_d3_normalisation",
            {
                "exhaustive_E": [H.E_ZERO, 254] if not args.quick else exponents,
                "exhaustive_v_below": 1 << 12,
                "exhaustive_compared": sum(c for c, _ in exhaustive),
                "exhaustive_mismatches": sum(m for _, m in exhaustive),
                "structure_bit_lengths": [0, 30],
                "structure_compared": sum(c for c, _ in structure),
                "structure_mismatches": sum(m for _, m in structure),
                "smt_proof": "see 0_pytest (test_closed_form_equals_successive_shifts_over_the_whole_reachable_domain_by_smt)",
                "seconds": round(time.time() - t0, 1),
            },
        )

        # (g) the adapter
        t0 = time.time()
        specs = [(-126, max(1, lo), lo + (1 << 17)) for lo in range(0, 1 << 23, 1 << 17)][:sub_shards]  # every subnormal significand
        sub = pool.map(adapter_shard, specs)
        normal = pool.map(adapter_normal_shard, [e for e in exponents if -126 <= e <= 127])  # 1024 x 3 mantissa patterns per exponent
        report(
            "g_adapter",
            {
                "subnormal_compared": sum(c for c, _ in sub),
                "subnormal_mismatches": sum(m for _, m in sub),
                "normal_compared": sum(c for c, _ in normal),
                "normal_mismatches": sum(m for _, m in normal),
                "seconds": round(time.time() - t0, 1),
            },
        )

    summary["seconds"] = round(time.time() - t_all, 1)
    mismatch_keys = ("mismatches", "exhaustive_mismatches", "structure_mismatches", "subnormal_mismatches", "normal_mismatches",
                     "cpp_adapter_ne_close", "cpp_finalised_ne_cpp_adapter", "cpp_matmul_ne_cpp_adapter",
                     "cpp_matmul_ne_python_wrapper", "cpp_adapter_ne_silicon_on_finite_result", "line_count_mismatch")
    failures = []
    for name, value in steps.items():
        if name == "d_transition_unit":
            for k in ("close_ne_silicon_on_finite_result", "silicon_ne_hardware", "wrapper_ne_close", "silicon_rejected"):
                if value["_total"][k]:
                    failures.append(f"{name}.{k}={value['_total'][k]}")
        elif name == "0_pytest":
            if value["returncode"] != 0:
                failures.append("pytest failed")
        else:
            for k in mismatch_keys:
                if value.get(k):
                    failures.append(f"{name}.{k}={value[k]}")
    summary["failures"] = failures
    (out / "wp1_exhaustive.json").write_text(json.dumps(summary, indent=1))
    run_id = os.environ.get("RESEARCH_RUN_ID")
    if run_id:
        result = {
            "schema": "research/result/v0.1",
            "run_id": run_id,
            "workload_fingerprint": {"driver": "benchmarks/numerical/tests/reference/exhaustive_driver.py", "random": args.random,
                                     "harness": bool(args.harness)},
            "validation": {"status": "passed" if not failures else "failed", "checker": "wp1 exhaustive_driver",
                           "detail": "; ".join(failures) or "every counter zero", "evidence": ["wp1_exhaustive.json"]},
            "measurements": [{"name": "wall", "value": summary["seconds"], "unit": "s"}],
            "measurement_files": ["wp1_exhaustive.json"],
            "artifacts": ["cases_evaluated.txt.gz"],
        }
        tmp = out / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=1))
        os.replace(tmp, out / "result.json")
    print("FAILURES" if failures else "ALL COUNTERS ZERO", failures, f"{summary['seconds']}s", flush=True)
    return 1 if failures else 0


def adapter_normal_shard(exponent: int) -> tuple[int, int]:
    compared = mismatches = 0
    for hi in range(1024):
        for lo in (0, 1, 0x1FFF):
            m = (1 << 23) | (hi << 13) | lo
            for sign in (False, True):
                g = H.Gfloat(sign, exponent, m)
                word = H.gfloat_to_float_bits(g)
                ok = H.fp32_to_gfloat(word) == g and H.group_sum([g]) == g and canon_term(S.fp32_term(word)) == canon(g)
                ok = ok and S.pack_fp32(term_of(g)) == word
                compared += 1
                if not ok:
                    mismatches += 1
    return compared, mismatches


if __name__ == "__main__":
    sys.exit(main())
