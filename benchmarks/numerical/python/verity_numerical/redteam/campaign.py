"""Run every red-team attack against ``verity.ml.circuits.silicon.relation`` (and ``silicon.tc_dot``) and write the findings report.

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

from verity.ml.circuits.silicon import relation as R
from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16 as AMP
from verity.ml.circuits.silicon.total import HOPPER_BF16_WGMMA_K16 as HOP

from . import cases as C
from . import forge as F
from . import harness as H
from . import relation_target as RT
from . import z3_relation as Z

PIPES = {"ampere": AMP, "hopper": HOP}
FIELDS = {"p40": RT.P40, "m61": RT.M61, "goldilocks": RT.GOLDILOCKS, "m89": (1 << 89) - 1}
HERE = "benchmarks/numerical/python/verity_numerical/redteam"
TESTS = "benchmarks/numerical/tests/redteam"


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


def run_checker(out: Path, *, full: bool = False, seed: int = 0) -> H.Report:
    """Wave 2: the WP2 checker contracts (``--target checker``).  Mutation / coordinated attacks on every hint column
    under its declared range over Z and F_p; z3 per gadget over Goldilocks, p40 and Mersenne-31 with Boolean case
    splits (the U3 re-encoding); the full group step (concrete, symbolic, selector-split, bit-vector); U4 range
    containment; the census deltas.  Every SAT model is replayed through ``FieldCtx`` before it is reported."""
    from verity_numerical.checker import REAL
    from verity_numerical.checker import search as S

    from . import census_review as CR
    from . import checker_target as CT
    from . import z3_checker as ZC

    rng = random.Random(seed)
    rep = H.Report(meta={"branch": _git("branch", "--show-current"), "commit": _git("rev-parse", "--short", "HEAD"),
                         "z3": z3.get_version_string(), "python": platform.python_version(), "full": full,
                         "target": "verity_numerical.checker (WP2 contracts)", "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    fid = iter(range(1, 10_000))
    nid = lambda prefix: f"{prefix}{next(fid)}"  # noqa: E731
    z3_to = 300_000 if full else 30_000
    cases = CT.step_cases(24 if full else 12, seed)
    fields = [("Z", None), ("goldilocks", CT.GOLDILOCKS), ("p40", CT.P40), ("m31", CT.M31)]

    # ---- completeness, mutations, coordinated: unit ------------------------------------------------------------------
    for fname, field in fields:
        t = CT.bound_checker_target(REAL, field)
        f = rep.add(H.run_completeness(t, cases, fid=nid("C")))
        f.reproducer = f"{HERE}/checker_target.py::bound_checker_target"
        if field in (CT.P40, CT.M31) and f.status == "FAIL":
            f.status, f.attack = "NOTE", "completeness (F4: a 32-bit word does not fit; expected for p < 2^32)"
        sample = cases if full else cases[:36]
        f = rep.add(H.run_mutations(t, sample, rng=rng, fid=nid("M")))
        f.reproducer = f"{HERE}/harness.py::run_mutations(checker_target)"
        f = rep.add(H.run_coordinated(t, sample[:12], rounds=600 if full else 150, rng=rng, fid=nid("X")))
        f.reproducer = f"{HERE}/harness.py::run_coordinated(checker_target)"
    # ---- the VU: word-chained units + epilogue -----------------------------------------------------------------------
    golden, _ = C.golden_cases()
    from verity_numerical.checker import Reject as _Reject
    from verity_numerical.checker.reference import vu as _vu

    chains = []
    while len(chains) < (6 if full else 3):
        picks = rng.sample(golden, 3)
        x = (sum((list(c[1][1]) for c in picks), []), sum((list(c[1][2]) for c in picks), []))
        try:
            _vu(REAL, *x)          # 114 of the 360 records saturate: an infinite intermediate word is outside the VU's domain
        except _Reject:
            continue
        chains.append((f"vu3[{len(chains)}]", x))
    for fname, field in fields[:2]:
        tv = CT.checker_vu_target(REAL, field)
        f = rep.add(H.run_completeness(tv, chains, fid=nid("C")))
        f.reproducer = f"{HERE}/checker_target.py::checker_vu_target"
        f = rep.add(H.run_mutations(tv, chains, rng=rng, fid=nid("M"), limit_per_case=None if full else 120))
        f.attack = "mutation (VU: 3 chained units + epilogue, every hint incl. the epilogue's)"
        f.reproducer = f"{HERE}/checker_target.py::checker_vu_target"
    rep.add(H.Finding(nid("N"), "checker[vu]", "forged intermediate FP32 word between units", "NOTE", 0.0,
                      detail={"why": "check_vu_expr passes unit i's *output expression* as unit i+1's c: the word is not a hint, so no "
                                     "forgery surface exists in the checker; a trace/AIR that materialises c_{i+1} as a column must bind "
                                     "it with one linear identity to unit i's word (census_review: +1 lin, +1 column per unit)"},
                      reproducer=f"{HERE}/census_review.py"))

    # ---- z3: per gadget, Boolean case split, three fields ------------------------------------------------------------
    for fname, field in [("goldilocks", CT.GOLDILOCKS), ("p40", CT.P40), ("m31", CT.M31)]:
        for name, spec in S.SPECS.items():
            r = ZC.gadget_split(spec, REAL, modulus=field, timeout_ms=z3_to)
            st = {"sat": "SAT", "unsat": "UNSAT", "unknown": "UNKNOWN"}[r["status"]]
            detail = {"field": field, "cases": [(c["case"], c["status"], c.get("seconds")) for c in r["cases"]]}
            if st == "SAT":
                detail["model"] = r["model"]
                detail["replayed_in_FieldCtx"] = _replay_gadget(spec, field, r["model"])
                detail["missing_bound"] = _missing_bound(name, field)
            rep.add(H.Finding(nid("Z"), f"checker[{name},{fname}]", "z3 gadget, Boolean hints split: output differs from reference",
                              st, r["seconds"], case=f"{len(r['cases'])} cases", detail=detail,
                              reproducer=f"{HERE}/z3_checker.py::gadget_split"))
    # U3 verbatim: signed_sum over F_(2^31-1) as stated, unsplit (the checker's own UNKNOWN) and split
    r0 = S.z3_gadget(S.SPECS["signed_sum"], REAL, timeout_ms=z3_to, modulus=S.M31)
    rep.add(H.Finding(nid("Z"), "checker[signed_sum,m31]", "U3 as stated (checker.search.z3_gadget, no split)",
                      {"sat": "SAT", "unsat": "UNSAT", "unknown": "UNKNOWN"}[r0.status], r0.seconds,
                      reproducer="verity_numerical/checker/search.py::z3_gadget(modulus=M31)"))
    # ---- z3: the full group step ------------------------------------------------------------------------------------
    r = S.z3_group_step(REAL, timeout_ms=z3_to, concrete=S.concrete_group_inputs())
    rep.add(H.Finding(nid("Z"), "checker[group_step,Z]", "U2 concrete input, Int encoding (checker.search)", r.status.upper(), r.seconds,
                      reproducer="verity_numerical/checker/search.py::z3_group_step(concrete)"))
    r = ZC.bv_group_step(REAL, timeout_ms=20_000, concrete=S.concrete_group_inputs())   # z3 overshoots BV timeouts (381 s at 60 s)
    rep.add(H.Finding(nid("Z"), "checker[group_step,Z]", "U2 concrete input, 128-bit bit-vector encoding (tables as shifts)",
                      r["status"].upper(), r["seconds"], detail={"tables_agree": ZC.bv_tables_agree(REAL)},
                      reproducer=f"{HERE}/z3_checker.py::bv_group_step"))
    r = ZC.int_group_step_split(REAL, timeout_ms=z3_to if full else 60_000, selectors=(1, 1))
    rep.add(H.Finding(nid("Z"), "checker[group_step,Z]", "U2 symbolic input, Int encoding, one-hot selectors fixed (1 of 100 cases)",
                      r["status"].upper(), r["seconds"], detail=r, reproducer=f"{HERE}/z3_checker.py::int_group_step_split"))
    r = ZC.int_group_step_vs_reference(REAL, timeout_ms=60_000, concrete=S.concrete_group_inputs())
    rep.add(H.Finding(nid("Z"), "checker[group_step,Z]", "U2 concrete input, one hint copy vs the composed reference", r["status"].upper(),
                      r["seconds"], reproducer=f"{HERE}/z3_checker.py::int_group_step_vs_reference"))
    r = ZC.int_group_step_vs_reference(REAL, timeout_ms=z3_to if full else 60_000, selector=1)
    rep.add(H.Finding(nid("Z"), "checker[group_step,Z]", "U2 symbolic input, one hint copy vs reference, selector fixed (1 of 10 cases)",
                      r["status"].upper(), r["seconds"], reproducer=f"{HERE}/z3_checker.py::int_group_step_vs_reference"))
    r = ZC.linear_group_step(REAL, timeout_ms=60_000, concrete=S.concrete_group_inputs())
    rep.add(H.Finding(nid("Z"), "checker[group_step,Z]", "U2 concrete input, linearised (QF_LIA: Boolean and table-output products split)",
                      r["status"].upper(), r["seconds"], reproducer=f"{HERE}/z3_checker.py::linear_group_step"))
    r = ZC.linear_group_step(REAL, timeout_ms=z3_to if full else 60_000)
    rep.add(H.Finding(nid("Z"), "checker[group_step,Z]", "U2 symbolic input, linearised (QF_LIA)", r["status"].upper(), r["seconds"],
                      reproducer=f"{HERE}/z3_checker.py::linear_group_step"))
    if full:
        r = ZC.bv_group_step(REAL, timeout_ms=z3_to)
        rep.add(H.Finding(nid("Z"), "checker[group_step,Z]", "U2 symbolic input, bit-vector encoding", r["status"].upper(), r["seconds"],
                          reproducer=f"{HERE}/z3_checker.py::bv_group_step"))
        for sel in range(REAL.n + 2):          # the exhaustive selector split, one copy vs the reference: UNSAT on all = U2 closed
            r = ZC.int_group_step_vs_reference(REAL, timeout_ms=z3_to, selector=sel)
            rep.add(H.Finding(nid("Z"), "checker[group_step,Z]", f"U2 symbolic input vs reference, selector case {sel}",
                              r["status"].upper(), r["seconds"], detail={"model": r["model"]},
                              reproducer=f"{HERE}/z3_checker.py::int_group_step_vs_reference"))
        for sel in range(REAL.n + 2):
            r = ZC.linear_group_step(REAL, timeout_ms=z3_to, selectors=(sel, sel))
            rep.add(H.Finding(nid("Z"), "checker[group_step,Z]", f"U2 symbolic input, linearised, both selectors fixed to {sel}",
                              r["status"].upper(), r["seconds"], detail={"model": r["model"]},
                              reproducer=f"{HERE}/z3_checker.py::linear_group_step"))
    # ---- U4 -----------------------------------------------------------------------------------------------------------
    c4 = ZC.containment(REAL)
    rep.add(H.Finding(nid("U"), "checker[contracts]", "U4 range containment along step.py's data flow (real widths)",
                      "NOTE" if c4["failing_edges"] else "PASS", 0.0,
                      detail={"failing_edges": c4["failing_edges"], "derived": c4["derived"],
                              "contract_to_contract_gaps": [e for e in c4["edges"] if not e["ok"]],
                              "reading": "state_out.e_hat reaches the floor (-132) when z = 1, below pack.e1's declared [-126, ..): harmless "
                                         "because pack multiplies fin by (1 - z), but the contract does not say so; the group-to-group "
                                         "exponent edge is contained only through the per-group ceiling e_hi(g) + growth, not the contract"},
                      reproducer=f"{HERE}/z3_checker.py::containment"))
    # ---- census ---------------------------------------------------------------------------------------------------------
    rep.add(H.Finding(nid("N"), "security.census", "counting conventions: what a backend pays that the census does not count",
                      "NOTE", 0.0, detail=CR.census_delta(REAL), reproducer=f"{HERE}/census_review.py::census_delta"))
    rep.meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    rep.write(out)
    return rep


def run_checker_v2(out: Path, *, full: bool = False, seed: int = 0) -> H.Report:
    """Wave 3: the checker v2 arithmetisation (``--target checker-v2``).  Completeness, mutation and coordinated
    attacks on every v2 hint column over Z, Goldilocks and Mersenne-31 (unit and VU); z3 per v2 gadget over the
    integers and over Goldilocks / M31 (symbolic tables); the two-witness group step.  A SAT or an accepted
    forgery with another output is a finding that kills the corresponding v2 idea."""
    from verity_numerical.checker import REAL
    from verity_numerical.checker.v2 import search as S2

    from . import checker_target as CT

    rng = random.Random(seed)
    rep = H.Report(meta={"branch": _git("branch", "--show-current"), "commit": _git("rev-parse", "--short", "HEAD"),
                         "z3": z3.get_version_string(), "python": platform.python_version(), "full": full,
                         "target": "verity_numerical.checker.v2", "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    fid = iter(range(1, 10_000))
    nid = lambda prefix: f"{prefix}{next(fid)}"  # noqa: E731
    z3_to = 600_000 if full else 60_000
    cases = CT.step_cases(24 if full else 12, seed)
    fields = [("Z", None), ("goldilocks", CT.GOLDILOCKS), ("m31", CT.M31)]
    for fname, field in fields:
        t = CT.bound_checker_target_v2(REAL, field)
        f = rep.add(H.run_completeness(t, cases, fid=nid("C")))
        f.reproducer = f"{HERE}/checker_target.py::bound_checker_target_v2"
        if field == CT.M31 and f.status == "FAIL":
            f.status, f.attack = "NOTE", "completeness (F4: a 32-bit word does not fit; expected for p < 2^32)"
        sample = cases if full else cases[:36]
        f = rep.add(H.run_mutations(t, sample, rng=rng, fid=nid("M")))
        f.reproducer = f"{HERE}/harness.py::run_mutations(bound_checker_target_v2)"
        f = rep.add(H.run_coordinated(t, sample[:12], rounds=600 if full else 150, rng=rng, fid=nid("X")))
        f.reproducer = f"{HERE}/harness.py::run_coordinated(bound_checker_target_v2)"
    golden, _ = C.golden_cases()
    from verity_numerical.checker import Reject as _Reject
    from verity_numerical.checker.reference import vu as _vu

    chains = []
    while len(chains) < (6 if full else 3):
        picks = rng.sample(golden, 3)
        x = (sum((list(c[1][1]) for c in picks), []), sum((list(c[1][2]) for c in picks), []))
        try:
            _vu(REAL, *x)
        except _Reject:
            continue
        chains.append((f"vu3[{len(chains)}]", x))
    for fname, field in fields[:2]:
        tv = CT.checker_vu_target_v2(REAL, field)
        f = rep.add(H.run_completeness(tv, chains, fid=nid("C")))
        f.reproducer = f"{HERE}/checker_target.py::checker_vu_target_v2"
        f = rep.add(H.run_mutations(tv, chains, rng=rng, fid=nid("M"), limit_per_case=None if full else 120))
        f.attack = "mutation (v2 VU: 3 state-chained units + Pack + epilogue, every hint)"
        f.reproducer = f"{HERE}/checker_target.py::checker_vu_target_v2"
    st = {"sat": "SAT", "unsat": "UNSAT", "unknown": "UNKNOWN"}
    for r in S2.integer_queries_v2(REAL, timeout_ms=z3_to):
        status = st.get(r.status, r.status.upper())
        if status == "SAT" and "without" in r.note:
            status = "NOTE"          # the dropped predicate is load-bearing: the expected demonstration
        rep.add(H.Finding(nid("Z"), f"checker-v2[{r.gadget},Z]", f"z3: {r.query}", status, r.seconds,
                          detail={"model": r.model, "note": r.note}, reproducer="verity_numerical/checker/v2/search.py::integer_queries_v2"))
    for r in S2.modular_queries_v2(REAL, timeout_ms=z3_to):
        rep.add(H.Finding(nid("Z"), f"checker-v2[{r.gadget},{r.query.split(',')[0]}]", f"z3 over F_p: {r.query}",
                          st.get(r.status, r.status.upper()), r.seconds, detail={"model": r.model, "note": r.note},
                          reproducer="verity_numerical/checker/v2/search.py::modular_queries_v2"))
    for r in (S2.z3_group_step_v2(REAL, timeout_ms=z3_to, concrete=S2.concrete_group_inputs_v2()),
              S2.z3_group_step_v2(REAL, timeout_ms=z3_to * (3 if full else 1))):
        rep.add(H.Finding(nid("Z"), "checker-v2[group_step,Z]", r.query, st.get(r.status, r.status.upper()), r.seconds,
                          detail={"model": r.model}, reproducer="verity_numerical/checker/v2/search.py::z3_group_step_v2"))
    rep.meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    rep.write(out)
    return rep


def _replay_gadget(spec, field: int, model: dict) -> dict:
    """Run the gadget's own ``check`` on the SAT model's values in ``FieldCtx``: accepted, and the outputs mod p."""
    from verity_numerical.checker import REAL, Reject

    from .checker_target import FieldCtx

    ctx = FieldCtx(field)
    x = {k: int(v) for k, v in model["inputs"].items()}
    h = {k: int(v) for k, v in model["hints"].items()}
    if spec.name == "group_max":
        h = {"E": h["E"], "sel": [h[k] for k in h if k.startswith("sel")]}
    try:
        outs = spec.call(ctx, REAL, x, h)
    except Reject as e:
        return {"accepted": False, "violation": str(e)}
    return {"accepted": True, "outputs_mod_p": [int(o) % field for o in outs], "reference": spec.ref(REAL, x)}


def _missing_bound(name: str, field: int) -> str:
    F = {"align_state": "F1 (lt_pow product wraps: p < 2^50) / F2 (tight quotient bound)",
         "align_product": "F1/F2 (p < 2^32 for the 16-bit lt_pow products)", "leading_bit": "F1 (p < 2^60)",
         "normalise": "F3 (div M' + r reaches 2^31 - 1) / F1", "fp32_decode": "F4 (32-bit word, p < 2^32)",
         "pack": "F4 (32-bit word)", "epilogue": "F4 (32-bit word)", "signed_sum": "p <= 2^31 - 2 (sign pinning)"}
    return F.get(name, "not in the checker's F1-F4 list: NEW") + f" at p = {field}"


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
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", choices=("relation", "checker", "checker-v2"), default="relation",
                    help="relation: verity.ml.circuits.silicon.relation (wave 1); checker: the WP2 checker contracts (wave 2); "
                         "checker-v2: the v2 arithmetisation (wave 3)")
    a = ap.parse_args(argv)
    runner = {"relation": run, "checker": run_checker, "checker-v2": run_checker_v2}[a.target]
    if a.out is None:
        name = f"redteam_{a.target.replace('-', '_')}.json"
        a.out = Path(run_dir) / name if run_dir else Path("backends/numerical/reports") / name
    rep = runner(a.out, full=a.full, seed=a.seed)
    print(rep.table())
    print(rep.to_json()["counts"])
    if run_dir:
        _write_result(Path(run_dir), a.out, rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
