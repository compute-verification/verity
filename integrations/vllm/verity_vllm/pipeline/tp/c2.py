"""A TP Commit's C2 of record per rank, and the Commit's side of a deferred replay: the gates that put a rank's sampled replay, boundary
linkage, population reconciliation and weights pin into its verdict (`replay_gates`, `pin_gate`, `c2_and_pin_gates`), the components
folded over the ranks (`sampled_replay_components`, `query_population_component`, `annotate_value_correspondence`), and `seal`, which
seals every rank's replay bundle of a deferred Commit (`tp-commit --replay-deferred`) and writes `replay_deferred.json`.  `pipeline.tp.commit`
decides in place with them, `pipeline.tp.commit_replay` from the ranks' replays on CPU.  A Commit defers only at a world where deferral
is proven (`DEFER_WORLDS`, `deferral_skipped`)."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from verity_vllm.pipeline import replay_bundle as RB

DEFERRED = "replay_deferred.json"
SCHEMA = "verity-vllm/tp-replay-deferred/v1"
#: the worlds whose Commit defers its ranks' C2 replay, each proven by a row whose deferred arm (the Commit's bundles, then
#: `tp-commit-replay` on CPU) gave the in-process arm's picks, per-rank records, verdict and run root; any other world replays in process
DEFER_WORLDS = (2, 8)


def deferral_skipped(args: Any) -> str | None:
    """Why a `tp-commit --replay-deferred` of `args.tp` ranks replays in process all the same, or None when it defers (or was not asked to).
    It drops the flag, so no rank's finalize defers, and says why in the Commit's log."""
    proven = " and ".join(f"TP{w}" for w in DEFER_WORLDS)
    why = f"world {args.tp}: deferral is proven at {proven} only" if args.replay_deferred and args.tp not in DEFER_WORLDS else None
    if why:
        print(f"[tp2commit] --replay-deferred not taken ({why}): every rank's C2 replay runs in its process", flush=True)
        args.replay_deferred = None
    return why


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_atomic(p: Path, text: str) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, p)


def last_per_rank(rows: list[dict]) -> dict:
    """The per-rank commit blocks of the last instrumented pair (attribution / binding records live there)."""
    for r in reversed(rows):
        if r.get("arm") != "control" and (r.get("commit") or {}).get("per_rank"):
            return r["commit"]["per_rank"]
    return {}


def query_population_gate(sr: dict | None, *, rank: int, required: bool) -> tuple[dict, str | None]:
    """The C2 POPULATION reconciliation of THIS rank's sampled replay against Q(P) -- `population.query_checks`
    (a VU whose address is not a required identity, a required identity no Program row produces, an identity unreachable by alias, ranks
    or block heights that disagree, a required family with no stated mechanism) -- read INTO the TP verdict, fail closed, by the SAME rule
    as the dense committer (commit_verdict._query_population_scope: one rule for every engine).  Returns (record block, FAIL text | None):
    * ok True  -> no fail; the scope note (query of record by id) goes onto the record;
    * ok False -> FAIL naming every non-zero check with its count and first identity (the dense text);
    * ok None / no `query_checks` / no `population` with a required manifest supplied -> FAIL by name ('not reconciled'): the row of
      record of a TP2 B1 256/32 row carried ok False x2 (960 + 960 per rank) under a PASS because nothing here read it; a record that
      cannot say whether it reconciled does not PASS over that silence either;
    * without a required manifest (no Q(P) asked for) the absence is stated, not failed (there was nothing to reconcile against).
    `per_rank_identity_check_ok` is a different check (the binding map's identities vs the manifest) and is not touched."""
    from verity_vllm.check.c2_rules import _query_population_scope
    pop = (sr or {}).get("population") if isinstance((sr or {}).get("population"), dict) else {}
    chk = pop.get("query_checks") if isinstance(pop.get("query_checks"), dict) else None
    note, fail = _query_population_scope(sr if isinstance(sr, dict) else None)
    block = {"rule": "population.query_checks (R17-3 reconciliation against Q(P)) per rank -> verdict, fail closed; commit_verdict._query_population_scope text (F-r17b-38)",
             "rank": int(rank), "present": chk is not None, "ok": (chk or {}).get("ok") if chk is not None else None,
             "counts": ({k: int((v or {}).get("n") or 0) for k, v in chk.items() if isinstance(v, dict) and "n" in v} if chk else None),
             "note": note, "fail": fail}
    if fail is None and chk is not None and chk.get("ok") is False:
        # the dense text needs `query_total` beside the checks; a record with the checks alone still says False -> named, never passed over
        fail = "population.query_checks.ok False -- " + ("; ".join(f"{k} x{n}" for k, n in (block["counts"] or {}).items() if n) or "see population.query_checks")
        block["fail"] = fail
    if fail:
        return block, f"C2 population (rank {rank}, F-r17b-38): {fail}"
    if required and (chk is None or chk.get("ok") is None):
        why = ("no `population.query_checks` on the rank's sampled replay" if chk is None else "`population.query_checks.ok` None (no Q(P) reconciled)")
        block["fail"] = f"NOT reconciled against Q(P) with a required manifest supplied: {why}"
        return block, f"C2 population (rank {rank}, F-r17b-38): {block['fail']}"
    return block, None


def replay_gates(cblock: dict, sr_: dict, lk_: dict, ok: bool, why: str | None, *, rank: int, replay_on: bool, required: bool) -> tuple[bool, str | None]:
    """The rank's C2 of record into its verdict, after the gates before it (in the rank's process, or from its bundle on CPU:
    `pipeline.tp.commit_replay`).  [TP-08 / manbind MAN-06 O1, per rank] sampled exact replay result True AND boundary linkage True
    on this rank; absence, a partial (a stratum not evaluated, an input not retained), population 0 or a mismatch = FAIL named.
    Per rank, the replay's POPULATION reconciled against Q(P) (`query_population_gate`) goes INTO the verdict, fail closed, by the
    dense committer's rule -- not behind the gates above: a PASS-graded replay over a population that does not reconcile is named."""
    cblock["sampled_replay"] = ({k: v for k, v in sr_.items() if k not in ("strata", "trace")} | {"strata_n": len(sr_.get("strata") or {})}) if sr_ else None
    cblock["boundary_linkage"] = lk_ or None
    if replay_on and ok:
        if sr_.get("result") is not True:
            ok, why = False, f"sampled replay (C2 of record, rank {rank}): {sr_.get('grade') or 'not run'} -- {sr_.get('why') or sr_.get('error') or 'no result'}"
        elif lk_.get("result") is not True:
            ok, why = False, f"boundary linkage (rank {rank}): {lk_.get('why') or 'not run'}"
    if replay_on:
        qp_block, qp_fail = query_population_gate(sr_, rank=rank, required=required)
        cblock["query_population"] = qp_block
        if qp_fail is not None:
            ok, why = False, (f"{why}; also: {qp_fail}" if not ok and why else qp_fail)
    return ok, why


def pin_gate(cblock: dict, wp_: dict, ok: bool, why: str | None, *, rank: int, required: bool) -> tuple[bool, str | None]:
    """[manbind MAN-09, per rank] the WEIGHTS OF RECORD: every served shard's sha256 == the digest composed from the
    pinned checkpoint by this rank's Program shard map; False -> FAIL named; absent with --required-manifest -> FAIL 'weights unpinned
    (MAN-09)' (fail closed); --no-weights-pin only records that choice, the verdict still FAILs closed.  The pin is NOT behind the
    earlier gates: a served shard that is not the checkpoint of record is the first-named fault even when its downstream symptoms
    (match oracle mismatches at the first collective input) already failed the rank (measured b1_tp12 neg_flip_weight_r1: oracle 826
    mismatches on rank 1 AND weights_pin False naming down_proj)."""
    cblock["weights_pin"] = ({k: v for k, v in wp_.items() if k != "mismatched"} | {"mismatched": (wp_.get("mismatched") or [])[:8], "mismatched_n": len(wp_.get("mismatched") or [])}) if wp_ else None
    if required:
        pin_why = None
        if wp_.get("result") is False:
            first = ((wp_.get("mismatched") or [{}])[0]).get("served_name") if wp_.get("mismatched") else (wp_.get("missing") or [None])[0]
            pin_why = f"weights of record (rank {rank}): {wp_.get('why')} (first: {first})"
        elif wp_.get("result") is not True:
            pin_why = f"weights unpinned (MAN-09, rank {rank}): {wp_.get('why') or 'no weights of record'}"
        if pin_why is not None:
            why = pin_why + (f"; also: {why}" if not ok and why else "")
            ok = False
    return ok, why


def c2_and_pin_gates(cblock: dict, f: dict, ok: bool, why: str | None, *, rank: int, replay_on: bool, required: bool, bundle: str) -> tuple[bool | None, str | None]:
    """The rank's verdict after its C2 of record and its weights pin, from the gates before them (``ok``, ``why``) and its finalize ``f``.
    A rank whose replay was deferred to its bundle (`--replay-deferred`: sampled-replay grade `deferred`) gets the pin alone: a rank that
    failed already is decided here, any other is PENDING (None) with its pre-replay verdict in ``cblock["replay_deferred"]["pre"]``,
    which `pipeline.tp.commit_replay.fold` passes to `replay_gates` once the rank's replay is on the record."""
    srb = f.get("sampled_replay") or {}
    sr_, lk_ = (srb.get("sampled_replay") or {}), (srb.get("boundary_linkage") or {})
    if replay_on and sr_.get("grade") == "deferred":
        cblock["sampled_replay"], cblock["boundary_linkage"] = sr_, lk_
        cblock["replay_deferred"] = {"pre": [ok, why], "bundle": bundle}
        ok, why = pin_gate(cblock, f.get("weights_pin") or {}, ok, why, rank=rank, required=required)
        return (None, RB.PENDING) if ok else (ok, why)
    ok, why = replay_gates(cblock, sr_, lk_, ok, why, rank=rank, replay_on=replay_on, required=required)
    return pin_gate(cblock, f.get("weights_pin") or {}, ok, why, rank=rank, required=required)


def annotate_unreplayable(srs: list[dict], xr: dict, pair: int, xrank_families: set[str] | list[str]) -> None:
    """Self-describing record: the replay's unreplayable collective families name the pass that evaluated them."""
    byf = xr.get("by_family") or {}
    for sr in srs:
        for u_ in (sr or {}).get("unreplayable") or []:
            fam_ = str(u_.get("family"))
            if fam_ in xrank_families:
                u_["evaluated_by"] = {"pass": "cross_rank_collectives (TP-12, rev F-r16-18 / COORD ruling 27)", "result": xr.get("result"),
                                      "picks": (byf.get(fam_) or {}).get("picks"), "equal": (byf.get(fam_) or {}).get("equal"),
                                      "artifact": f"xrank_pair{pair}.json", "rule": "sampled strata over the committed collective outputs (>= per_stratum per site x kind x phase), not every VU"}


def query_population_component(rows: list[dict], replay_on: bool) -> dict:
    """The population reconciliation per rank, AND-ed, as a named component beside sampled_replay (read from the last
    instrumented pair's per-rank blocks, where the gate wrote it into the verdict); None while a rank's replay is deferred."""
    pr = last_per_rank(rows)
    qp_per_rank = {str(r_): (c.get("query_population") or None) for r_, c in pr.items()}
    deferred = sorted(str(r_) for r_, c in pr.items() if (c.get("sampled_replay") or {}).get("grade") == "deferred")
    return {"result": (None if not replay_on or not qp_per_rank or deferred else all(isinstance(b, dict) and b.get("ok") is True and not b.get("fail") for b in qp_per_rank.values())),
            "per_rank": {r_: ({k: b.get(k) for k in ("present", "ok", "counts", "fail")} if isinstance(b, dict) else None) for r_, b in sorted(qp_per_rank.items())},
            "rule": "population.query_checks.ok True on EVERY rank (R17-3 reconciliation against Q(P)); False / absent with a required manifest = the rank FAILs by name (F-r17b-38)",
            "why": (None if not replay_on else (f"deferred: ranks {deferred} replay from their bundles" if deferred else None)
                    or "; ".join(f"rank {r_}: {b.get('fail')}" for r_, b in sorted(qp_per_rank.items()) if isinstance(b, dict) and b.get("fail")) or None)}


def annotate_value_correspondence(vc: dict | None, sr_comp: dict, lk_comp: dict, qp_comp: dict, wp_comp: dict, xr_comp: dict, fb_comp: dict,
                                  *, replay_on: bool) -> dict:
    """coverage.json's `value_correspondence` with the C2 of record's components on it: ``vc`` as the value check / Match record left it."""
    vc = vc or {"method": "no value check / Match record this attempt", "result": None, "file": None, "components": {}}
    comps = vc.setdefault("components", {})
    comps["sampled_replay"], comps["boundary_linkage"] = sr_comp, lk_comp
    comps["query_population"] = qp_comp
    if qp_comp.get("result") is False and vc.get("result") is not False:
        vc["result"] = False
    comps["weights_pin"] = wp_comp
    if wp_comp.get("result") is not True and vc.get("result") is True:
        vc["result"] = False if wp_comp.get("result") is False else None
    comps["cross_rank_collectives"] = xr_comp
    if xr_comp.get("result") is False and vc.get("result") is not False:
        vc["result"] = False
    comps["fold_match_binding"] = {k: fb_comp.get(k) for k in ("result", "n_bindings", "n_recorder", "n_bad_bindings", "why", "rule")} | {"artifact": "fold_match_binding.json"}
    if fb_comp.get("result") is False and vc.get("result") is not False:
        vc["result"] = False
    if replay_on:
        comps["replay"] = {"result": sr_comp.get("result"), "reference": "see components.sampled_replay (manbind MAN-06 sampled exact replay, per rank)"}
        both = (sr_comp.get("result") is True and lk_comp.get("result") is True)
        if sr_comp.get("deferred"):
            both = None
            vc["result"] = False if vc.get("result") is False else None
        else:
            vc["result"] = (both and vc["result"]) if vc.get("result") is not None else (both if (sr_comp.get("result") is not None) else None)
        vc["method"] = f"sampled-exact-replay({sr_comp.get('result') and 'complete' or (sr_comp.get('per_rank') and next(iter(sr_comp['per_rank'].values())).get('grade')) or 'not-evaluated'}, per rank) + boundary-linkage + " + str(vc.get("method"))
        vc["c2_of_record"] = {"form": "A (sampled replay) per rank", "result": both, "rule": "MAN-06 O1: result true on EVERY rank and pair, else the Commit FAILs named"}
    return vc


#: summary.json's `sampled_replay`: the keys of components.sampled_replay it carries
SR_SUMMARY_KEYS = ("result", "n", "sample", "population", "weights", "weights_self_consistent", "weights_pinned", "weights_note", "unreplayable", "per_rank", "per_run", "why")


def sampled_replay_components(rows: list[dict], versions: dict) -> tuple[dict, dict]:
    """[TP-08 / manbind MAN-06, form (A) shape] `components.sampled_replay` {result, population, n, strata, seed, unreplayable,
    why, per_rank} and `components.boundary_linkage` {result, method, checked, why, per_rank}, folded over the ranks of the LAST
    instrumented pair (AND: one rank not True = not True) with every earlier instrumented pair AND-ed in (`per_run`), so a failing pair
    or rank is named.  A TP2 row's population is the union of the ranks' rank-Program rows; the `ranks` stratum lists the true ranks."""
    pairs = [r["commit"]["per_rank"] for r in rows if r.get("arm") != "control" and (r.get("commit") or {}).get("per_rank")]
    if not versions.get("sampled_replay") or not pairs:
        na = {"result": None, "why": ("not run (--no-sampled-replay or no --required-manifest)" if not versions.get("sampled_replay") else "no instrumented pair finalized")}
        return ({"method": "sampled-exact-replay per rank (manbind sampled_replay)", "population": None, "n": 0, "strata": None, "seed": None, "unreplayable": []} | na,
                {"method": "committed input ids at arrival == declared prompt; later steps == previous step's committed sampled id, per request per rank", "checked": 0} | na)
    pr = pairs[-1]
    srs = {r_: (c.get("sampled_replay") or {}) for r_, c in pr.items()}
    lks = {r_: (c.get("boundary_linkage") or {}) for r_, c in pr.items()}
    per_run = [{r_: {"sampled_replay": (c.get("sampled_replay") or {}).get("result"), "grade": (c.get("sampled_replay") or {}).get("grade"),
                     "boundary_linkage": (c.get("boundary_linkage") or {}).get("result")} for r_, c in p.items()} for p in pairs]
    sr_ok = len(srs) > 0 and all(s.get("result") is True for s in srs.values()) and all(all(v["sampled_replay"] is True for v in p.values()) for p in per_run)
    lk_ok = len(lks) > 0 and all(l.get("result") is True for l in lks.values()) and all(all(v["boundary_linkage"] is True for v in p.values()) for p in per_run)
    smp = {r_: (s.get("sample") or {}) for r_, s in srs.items()}
    popn = {r_: (s.get("population") or {}) for r_, s in srs.items()}
    keys = {r_: (s.get("strata_keys") or {}) for r_, s in srs.items()}
    unrep: dict[str, dict] = {}
    for r_, s in srs.items():
        for u in s.get("unreplayable") or []:
            k = str(u.get("family"))
            unrep.setdefault(k, {"family": k, "n": 0, "why": u.get("why"), "ranks": []})
            unrep[k]["n"] += int(u.get("n") or 0)
            unrep[k]["ranks"].append(int(r_))
    lifetimes: dict[str, int] = {}
    for k_ in keys.values():
        for lv, n_ in ((((k_.get("lifetime") or {}).get("values")) or {}).items()):
            lifetimes[lv] = lifetimes.get(lv, 0) + int(n_)
    sr_comp = {"result": sr_ok if all(s for s in srs.values()) else None,
               "method": ("per rank, in the rank's committing process (tp2_worker finalize): manbind sampled_replay on THIS rank's binding map (identity.rank == r), "
                          "THIS rank's request Programs of record (derive_step --tp 2 --rank r; components per (LP, T)), the rank's committed members and "
                          "its weight SHARDS re-registered at replay; stratified seeded sample, bit-exact compare with the committed output members; rows fed "
                          "by a peer rank's partial have no evaluator and are counted by name"),
               "population": {"n": sum(int(p.get("vus") or 0) for p in popn.values()), "per_rank": {r_: p.get("vus") for r_, p in popn.items()},
                              "source": next((p.get("source") for p in popn.values() if p.get("source")), None),
                              "by_family": {f: sum(int((p.get("by_family") or {}).get(f) or 0) for p in popn.values()) for f in sorted({f for p in popn.values() for f in (p.get("by_family") or {})})}},
               "n": sum(int(m.get("evaluated") or 0) for m in smp.values()),
               "sample": {"picked": sum(int(m.get("picked") or 0) for m in smp.values()), "evaluated": sum(int(m.get("evaluated") or 0) for m in smp.values()),
                          "equal": sum(int(m.get("equal") or 0) for m in smp.values()), "mismatch_n": sum(int(m.get("mismatch_n") or 0) for m in smp.values()),
                          "not_evaluated_n": sum(int(m.get("not_evaluated_n") or 0) for m in smp.values()), "per_stratum": next((m.get("per_stratum") for m in smp.values()), None)},
               "strata": {"n": sum(int(p.get("strata_n") or 0) for p in popn.values()),
                          "ranks": [f"rank {r_}" for r_ in sorted(srs, key=int)],
                          "mechanisms": sorted({m for k_ in keys.values() for m in (k_.get("mechanisms") or [])}),
                          "requests": sorted({q for k_ in keys.values() for q in (k_.get("requests") or [])}),
                          "phases": sorted({q for k_ in keys.values() for q in (k_.get("phases") or [])}),
                          "steps": {"early": sorted({s_ for k_ in keys.values() for s_ in ((k_.get("steps") or {}).get("early") or [])}),
                                    "late": sorted({s_ for k_ in keys.values() for s_ in ((k_.get("steps") or {}).get("late") or [])})},
                          "lifetime": {"values": lifetimes, "ring_slots": next((((k_.get("lifetime") or {}).get("ring_slots")) for k_ in keys.values()), None)},
                          "key": next((k_.get("key") for k_ in keys.values() if k_.get("key")), None),
                          "unevaluated": sorted({u for s in srs.values() for u in (s.get("strata_unevaluated") or [])})[:12]},
               "seed": {r_: {"seed": m.get("seed"), "source": m.get("seed_source")} for r_, m in smp.items()},
               "weights": {r_: (s.get("weights") or {}).get("equal") for r_, s in srs.items()},
               # [M-0614] the per-rank weights check is self-consistency (re-registered shards == the run's root); a pin of record is a Build-side digest
               "weights_self_consistent": {r_: (s.get("weights") or {}).get("self_consistent", (s.get("weights") or {}).get("equal")) for r_, s in srs.items()},
               "weights_pinned": {r_: (s.get("weights") or {}).get("pinned") for r_, s in srs.items()},
               "weights_note": "self-consistency, not a weights-of-record pin (rev-cb M-0610, COORD M-0614; tp M-0632 for the per-rank shard pin)",
               "unreplayable": sorted(unrep.values(), key=lambda u: u["family"]),
               "mismatch": [dict(m, rank=int(r_)) for r_, s in srs.items() for m in (s.get("mismatch") or [])][:12],
               "per_rank": {r_: {k2: s.get(k2) for k2 in ("result", "grade", "why", "rank", "rank_note")} for r_, s in srs.items()},
               "per_run": per_run,
               "why": (None if sr_ok else "; ".join(f"rank {r_}: {s.get('grade')} {s.get('why') or s.get('error') or 'no result'}" for r_, s in srs.items() if s.get("result") is not True) or "a rank of an earlier pair failed")}
    lk_comp = {"result": lk_ok if all(l for l in lks.values()) else None,
               "method": ("per rank, exhaustive over (engine step, request): committed input_token_ids at the arrival step == the declared prompt of the workload; "
                          "at every later step == that request's committed sampled_token_ids of the previous step (manbind boundary_linkage on the rank's map)"),
               "checked": sum(int(l.get("pairs") or 0) for l in lks.values()), "linked": sum(int(l.get("linked") or 0) for l in lks.values()),
               "per_rank": {r_: {k2: l.get(k2) for k2 in ("result", "pairs", "linked", "why")} for r_, l in lks.items()},
               "why": (None if lk_ok else "; ".join(f"rank {r_}: {l.get('why') or 'no result'}" for r_, l in lks.items() if l.get("result") is not True) or "a rank of an earlier pair failed")}
    deferred = sorted({str(r_) for p in pairs for r_, c in p.items() if (c.get("sampled_replay") or {}).get("grade") == "deferred"}, key=int)
    if deferred:
        for comp in (sr_comp, lk_comp):
            comp.update(result=None, deferred={"ranks": deferred}, why=f"deferred: ranks {deferred} replay on CPU from their bundles (`verity-vllm tp-commit-replay`)")
    return sr_comp, lk_comp


def seal(out: Path, summ: dict[str, Any], rows: list[dict[str, Any]], bundles_of: dict[int, dict[int, dict | None]], *, replay_cfg: dict | None,
         required_manifest: str | None, world: int, versions: dict[str, Any], all_pass: bool, vc_base: dict | None, comps: dict[str, dict],
         xrank_pairs: list[dict], xrank_families: list[str], label: str | None) -> None:
    """At the end of a deferred TP Commit: when every check it ran passed and its instrumented runs await their ranks' replay, seals
    every rank's bundle, writes `replay_deferred.json` and makes ``summ`` PENDING; otherwise the Commit is decided and every partial
    bundle goes."""
    trt = [r for r in rows if r["arm"] != "control"]
    pending = [r["pair"] for r in trt if (r.get("commit") or {}).get("commit_pass") is None]
    if not (pending and summ.get("pass") is True and replay_cfg is not None):
        for p, ranks in bundles_of.items():
            for rk in ranks:
                RB.discard(out, p, rank=rk)
        if pending:
            print(f"[tp2commit] REPLAY BUNDLES discarded: the Commit is decided (pass {summ.get('pass')}) with pairs {pending} awaiting a replay", flush=True)
        return
    man_sha = sha256_file(Path(required_manifest)) if required_manifest else None
    bundles = []
    for row in trt:
        pair = int(row["pair"])
        for rk, c in sorted(row["commit"]["per_rank"].items()):
            b = bundles_of[pair][int(rk)]
            ctx = {"tp": {"world": world, "rank": int(rk)}, "pair": pair, "label": label, "code": versions.get("git_head"),
                   "run_root": c["run_root"], "tp_run_root": row["commit"]["tp_run_root"], "weight_registration": c.get("weight_registration"),
                   "replay": replay_cfg, "required_manifest": required_manifest, "required_manifest_sha256": man_sha,
                   "engine_of_record": b["engine_of_record"], "programs": b["programs"]}
            t0 = time.perf_counter()
            m = RB.seal(out, pair, ctx, None, hashed=b["hashed"], rank=int(rk))
            bundles.append({"pair": pair, "rank": int(rk), "path": m["path"], "manifest_sha256": m["sha256"], "bytes": m["bytes"], "files": len(m["files"])})
            print(f"[tp2commit] REPLAY BUNDLE (pair {pair}, rank {rk}): {m['path']} sealed in {time.perf_counter() - t0:.1f} s, {len(m['files'])} files, "
                  f"{m['bytes'] >> 20} MiB, manifest sha256 {m['sha256'][:16]}", flush=True)
    xr_of = {str(r["pair"]): {"result": xr.get("result"), "by_family": xr.get("by_family")} for r, xr in zip(trt, xrank_pairs)}
    doc = {"schema": SCHEMA, "world": world, "pending_pairs": pending, "bundles": bundles, "all_pass": all_pass, "vc_base": vc_base,
           "components": comps, "xrank": xr_of, "xrank_families": xrank_families, "required_manifest": required_manifest,
           "sealed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "replayed": []}
    write_atomic(out / DEFERRED, json.dumps(doc, indent=1, default=str) + "\n")
    summ["pass"] = None
    if summ.get("only_arm"):
        summ["commit_pass"] = None
    summ["replay_deferred"] = {"record": DEFERRED, "pending_pairs": pending, "bundles": bundles}
