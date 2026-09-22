"""Run every red-team attack against ``verity.ml.tc.relation`` (and ``silicon.tc_dot``) and write the findings report.

    python -m verity_numerical.redteam.campaign --out backends/numerical/reports/redteam_relation.json [--full]

``--full`` is the pod-sized sweep (more instances, longer z3 timeouts); the default finishes on a laptop in a few
minutes.  Every finding's ``reproducer`` names the function that rebuilds it.  Under ``research run`` (``RESEARCH_RUN_DIR``
set) the report goes into the run directory and a ``result.json`` envelope is written beside it.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import z3

from verity.ml.tc import relation as R
from verity.ml.tc.silicon import AMPERE_BF16_M16N8K16 as AMP
from verity.ml.tc.total import HOPPER_BF16_WGMMA_K16 as HOP

from . import cases as C
from . import forge as F
from . import harness as H
from . import relation_target as RT
from . import z3_relation as Z

PIPES = {"ampere": AMP, "hopper": HOP}
FIELDS = {"p40": RT.P40, "m61": RT.M61, "goldilocks": RT.GOLDILOCKS, "m89": (1 << 89) - 1}
HERE = "backends/numerical/python/verity_numerical/redteam"
TESTS = "backends/numerical/tests/redteam"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


def run(out: Path, *, full: bool = False, seed: int = 0) -> H.Report:
    rng = random.Random(seed)
    rep = H.Report(meta={"branch": _git("branch", "--show-current"), "commit": _git("rev-parse", "--short", "HEAD"),
                         "z3": z3.get_version_string(), "python": platform.python_version(), "full": full,
                         "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    golden, measured = C.golden_cases()
    n_inst = 24 if full else 6
    z3_to = 300_000 if full else 60_000
    z3_to_p40 = 300_000 if full else 20_000       # the 40-bit whole-group queries are the slow ones (UNKNOWN is listed)
    fid = iter(range(1, 10_000))
    nid = lambda prefix: f"{prefix}{next(fid)}"  # noqa: E731

    # ---- (i) completeness --------------------------------------------------------------------------------------------
    for pname, pipe in PIPES.items():
        structured = C.structured_cases(pipe)
        for align in (R.ALIGN_DIV, R.ALIGN_TABLE):
            t = RT.relation_target(pipe, align)
            cs = structured + (golden if pipe is AMP else [])
            f = rep.add(H.run_completeness(t, cs, fid=nid("C")))
            f.reproducer = f"{TESTS}/test_redteam_relation_completeness.py"
            if pipe is AMP:   # the witness's output is the *measured* word
                bad = [n for n, x in golden if t.output_of(t.gen(x)) != measured[n]]
                f.detail["golden_measured_mismatches"] = bad
                if bad:
                    f.status = "FAIL"
    f = rep.add(H.run_completeness(RT.silicon_target(AMP), golden, fid=nid("C")))
    f.reproducer = f"{TESTS}/test_redteam_harness.py"

    # ---- (ii) mutations, (iv) coordinated -----------------------------------------------------------------------------
    for pname, pipe in PIPES.items():
        sample = C.structured_cases(pipe)[:28] + (rng.sample(golden, 12) if pipe is AMP else [])
        t = RT.relation_target(pipe)
        f = rep.add(H.run_mutations(t, sample, rng=rng, fid=nid("M")))
        f.reproducer = f"{HERE}/harness.py::run_mutations"
        f = rep.add(H.run_coordinated(t, sample[:10], rounds=150 if not full else 600, rng=rng, fid=nid("X")))
        f.reproducer = f"{HERE}/harness.py::run_coordinated"
    s = RT.silicon_target(AMP)
    rep.add(H.run_mutations(s, golden[:20], fid=nid("M"))).reproducer = f"{TESTS}/test_redteam_harness.py"
    rep.add(H.run_coordinated(s, golden[:50], rounds=0, fid=nid("X"))).reproducer = f"{TESTS}/test_redteam_harness.py"
    # the harness notices a weakened checker (self-test): drop the predicate binding out.M
    weak = RT.relation_target(AMP, dropped=("g1.out.M",))
    f = rep.add(H.run_mutations(weak, golden[:2], rng=rng, fid=nid("M")))
    f.attack = "mutation (self-test: relation with g1.out.M dropped)"
    f.reproducer = f"{TESTS}/test_redteam_harness.py::test_harness_detects_weakened_checker"

    # ---- (iii) z3: gadgets --------------------------------------------------------------------------------------------
    for pname, pipe in PIPES.items():
        pool = golden if pipe is AMP else [c for c in C.structured_cases(pipe) if c[0].startswith("random")]
        insts = rng.sample(pool, 3)
        for cname, x in insts:
            w = R.step_witness(pipe, R.decode_state(x[0], pipe), list(x[1]), list(x[2]))
            g = w["groups"][0]
            t1 = next((t for t in g["terms"] if t["aligned"] and t["d"]), None)
            if t1 is None:      # no term is shifted in this group: the align gadget has nothing to mis-split
                t1 = g["terms"][0]
            scaled = t1["aligned"] * t1["pow"] + t1["r"]
            exps = [e for _, e, _ in R._terms(R.params(pipe), w["in"], w["prod"][:pipe.groups[0]])]
            for fname, field in [("Z", None), *FIELDS.items()]:
                queries = {
                    "align (remainder/quotient mis-split)": lambda: Z.align_query(pipe, field, t1["d"], scaled, timeout_ms=z3_to),
                    "final shift (mis-split)": lambda: Z.final_query(pipe, field, g["magW"], g["e_out"], timeout_ms=z3_to),
                    "leading-bit position": lambda: Z.norm_query(pipe, field, g["mag_t"], timeout_ms=z3_to),
                    "E* above the maximum": lambda: Z.estar_query(pipe, field, exps, mode="above", timeout_ms=z3_to),
                    "E* at a non-maximal term": lambda: Z.estar_query(pipe, field, exps, mode="nonmax", timeout_ms=z3_to),
                    "sign flip of the total": lambda: Z.sign_query(pipe, field, g["total"], timeout_ms=z3_to),
                }
                for qname, q in queries.items():
                    st, model, dt = q()
                    detail = {"model": model, "field": field}
                    if qname.startswith("sign") and st == "SAT" and g["total"] == 0:
                        st = "NOTE"        # neg_t is free on a zero total; neg_fin = neg_t * (1 - z_out) hides it
                        detail["unobservable"] = "total = 0: z_t = z_out = 1, the output is the canonical zero either way"
                    rep.add(H.Finding(nid("Z"), f"relation[{pname},{fname}]", f"z3 gadget: {qname}", st, dt, case=cname,
                                      detail=detail, reproducer=f"{HERE}/z3_relation.py"))
            if pipe is AMP and cname == insts[0][0]:
                st, model, dt = Z.sign_query(pipe, None, 0, timeout_ms=z3_to)
                rep.add(H.Finding(nid("N"), f"relation[{pname},Z]", "z3 gadget: sign of a zero total (neg_t free, unobservable)",
                                  "NOTE" if st == "SAT" else st, dt, case="total = 0", detail={"model": model},
                                  reproducer=f"{HERE}/z3_relation.py::sign_query"))

    # ---- (iii) z3: whole groups and steps -----------------------------------------------------------------------------
    for pname, pipe in PIPES.items():
        pool = golden if pipe is AMP else C.structured_cases(pipe)
        insts = rng.sample(pool, n_inst)
        for fname, field in [("Z", None), ("goldilocks", RT.GOLDILOCKS), ("p40", RT.P40)]:
            if fname != "Z":
                insts_f = insts[: (8 if full else 3)]
            else:
                insts_f = insts
            to = z3_to_p40 if fname == "p40" else z3_to
            for cname, x in insts_f:
                w = R.step_witness(pipe, R.decode_state(x[0], pipe), list(x[1]), list(x[2]))
                y = R.pack_state(w["out"])
                for gi in range(len(pipe.groups)):
                    st, model, dt = Z.group_query(pipe, field, w, gi, timeout_ms=to)
                    detail: dict = {"field": field}
                    if st == "SAT":
                        w2 = Z.witness_from_model(pipe, w, model, field)
                        F._regroup_following(pipe, w2, gi)
                        y2 = R.pack_state(w2["out"])
                        ok, viol = RT.relation_check(RT.RelationSemantics(pipe, R.ALIGN_DIV, field), x, y2, w2)
                        gh, gf = w["groups"][gi], w2["groups"][gi]
                        detail.update(replayed_through_check_step=ok, violations=viol[:5], honest_output=f"{y:#010x}",
                                      forged_output=f"{y2:#010x}", input=RT.fmt_x(x),
                                      changed_columns={k: [gh[k], gf[k]] for k in gf if k not in ("terms", "out") and gf[k] != gh[k]},
                                      changed_terms={i: {k: [th[k], tf[k]] for k in tf if tf[k] != th[k]}
                                                     for i, (th, tf) in enumerate(zip(gh["terms"], gf["terms"])) if th != tf})
                        if not ok:
                            st = "SAT-NOT-REPLAYED"
                    rep.add(H.Finding(nid("Z"), f"relation[{pname},{fname}]", f"z3 whole group {gi}: another output state",
                                      st, dt, case=cname, detail=detail,
                                      reproducer=f"{HERE}/z3_relation.py::group_query + witness_from_model"))
                    if fname != "Z":
                        st2, _, dt2 = Z.group_query(pipe, field, w, gi, timeout_ms=to, extra_ranges={"magW": pipe.width})
                        rep.add(H.Finding(nid("Z"), f"relation[{pname},{fname}]+RANGE(magW)",
                                          f"z3 whole group {gi} with the repair RANGE(magW, W) added", st2, dt2, case=cname,
                                          detail={"field": field}, reproducer=f"{HERE}/z3_relation.py::group_query(extra_ranges)"))
                if fname == "Z":
                    st, model, dt = Z.step_query(pipe, None, w, timeout_ms=z3_to)
                    rep.add(H.Finding(nid("Z"), f"relation[{pname},Z]", "z3 whole step (groups chained): another output state",
                                      st, dt, case=cname, reproducer=f"{HERE}/z3_relation.py::step_query"))

    # ---- closed-form forgeries over prime fields ----------------------------------------------------------------------
    for pname, pipe in PIPES.items():
        cs = C.structured_cases(pipe) + (golden if pipe is AMP else [])
        t0 = time.time()
        for fname, field in FIELDS.items():
            sem = RT.RelationSemantics(pipe, R.ALIGN_DIV, field)
            for gadget, fn in F.FORGERIES.items():
                t1 = time.time()
                wrong, same, rejected, first = 0, 0, 0, None
                groups = 0
                for cname, x in cs:
                    w = R.step_witness(pipe, R.decode_state(x[0], pipe), list(x[1]), list(x[2]))
                    y = R.pack_state(w["out"])
                    for gi in range(len(pipe.groups)):
                        groups += 1
                        fg = fn(pipe, w, field, gi)
                        if fg is None:
                            continue
                        ok, viol = RT.relation_check(sem, x, fg.forged_output, fg.witness)
                        if not ok:
                            rejected += 1
                            continue
                        if fg.forged_output != y:
                            wrong += 1
                            if first is None:
                                first = {"case": cname, "group": gi, "input": RT.fmt_x(x), "honest_output": f"{y:#010x}",
                                         "forged_output": f"{fg.forged_output:#010x}", **fg.detail}
                        else:
                            same += 1
                status = "SAT" if wrong else ("NOTE" if same else "UNSAT")
                rep.add(H.Finding(nid("F"), f"relation[{pname},{fname}]", f"closed-form forgery: {gadget}", status,
                                  time.time() - t1, case=f"{len(cs)} cases / {groups} groups",
                                  detail={"field": field, "accepted_wrong_output": wrong, "accepted_same_output": same,
                                          "rejected": rejected, "log2_threshold": F.thresholds(pipe)[gadget], "example": first},
                                  reproducer=f"{HERE}/forge.py::{fn.__name__}"))
        rep.meta[f"forgeries_seconds_{pname}"] = round(time.time() - t0, 1)

    # ---- observations -------------------------------------------------------------------------------------------------
    rep.add(H.Finding(nid("N"), "relation", "census claim: 'a field of at least 40 bits' suffices", "NOTE", 0.0,
                      detail={"unsupported": [
                          "aligned * pow = scaled - r with rinv < 2^A pins aligned only for p >= 2^(2A) (2^52 Ampere, 2^54 Hopper)",
                          "mag24 * pow_s = magW - r_s with rsinv < 2^K pins mag24 only for p >= 2^(24+K) (2^59 / 2^60)",
                          "normB = mag_t * normpow in the window pins L only for p >= 2^(2B) (2^60 / 2^64; Goldilocks by 2^32)",
                          "magW * 2^(B-W) + r_n = normB pins magW over no prime field: magW is never range-checked",
                          "'the range lookups are what pin a witness to the integers': false for magW, r (below 2^(2A)), r_s (below 2^(24+K))",
                          "'one extra limb split each' for a 31-bit field: soundness needs the factors split so that every partial product stays below p, not only the honest values",
                          "the census counts no decode of an arbitrary transition-unit accumulator c (the VU starts at the constant zero state; the backend transition unit does not)",
                          "test_every_column_is_constrained perturbs one column at a time over the integers; every finding here is a coordinated multi-column witness, three of them field-only",
                      ]}, reproducer=f"{HERE}/forge.py (module docstring)"))
    rep.add(H.Finding(nid("N"), "relation", "witness non-uniqueness: neg_t is free when total = 0", "NOTE", 0.0,
                      detail={"observable": False, "why": "neg_fin = neg_t * (1 - z_out) and ovf_eff = ovf * (1 - z_t) both vanish"},
                      reproducer=f"{HERE}/z3_relation.py::sign_query(total=0)"))
    rep.meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    rep.write(out)
    return rep


def _write_result(run_dir: Path, out: Path, rep: H.Report) -> None:
    """The ``research`` result envelope: the findings report is the evidence; ``validation`` is *not* 'the relation
    is sound' but 'the campaign ran to completion' (SAT findings are the deliverable, not a failure)."""
    counts = rep.to_json()["counts"]
    doc = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
           "workload_fingerprint": {"stage": "redteam.search", **rep.meta},
           "validation": {"status": "passed", "checker": "verity_numerical.redteam.campaign",
                          "detail": f"findings by status: {counts}", "evidence": [out.name]},
           "measurements": [{"name": f"findings_{k.lower().replace('-', '_')}", "value": v, "unit": "count"} for k, v in counts.items()],
           "measurement_files": [out.name], "artifacts": [out.name]}
    tmp = run_dir / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, run_dir / "result.json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    default_out = Path(run_dir) / "redteam_relation.json" if run_dir else Path("backends/numerical/reports/redteam_relation.json")
    ap.add_argument("--out", type=Path, default=default_out)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rep = run(a.out, full=a.full, seed=a.seed)
    print(rep.table())
    print(rep.to_json()["counts"])
    if run_dir:
        _write_result(Path(run_dir), a.out, rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
