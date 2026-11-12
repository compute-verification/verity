"""cover_blocks_build.py BLOCKS.json CUTS.json ACCEPTED.json FRIDAY_RUN_DIR: `cover.json` v2, the v1 record (`cover_build.py`) with the
table by block type beside its rows.

BLOCKS.json is `cover_blocks.sh`'s blocks.json, CUTS.json `cover_list.sh`'s cuts.json, ACCEPTED.json `cover_accepted.py`'s.
Beside this script: `cover_staged.json` (CIR-259's staging steps), `cover_ksteady.json` (HASH-98's Lean rehearsals). The
audit of record is read from its run's files. One row per block type of main's card, then one per unit a list template the
card does not cut can be cut into. A statement counts as accepted only where Lean's verdict in its run's files is accepted.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
J = lambda p: json.loads(Path(p).read_text())  # noqa: E731
blocks_in, cuts, acc = J(sys.argv[1]), J(sys.argv[2])["templates"], J(sys.argv[3])
cover = J(HERE / "cover.json")
staged_steps, ksteady = J(HERE / "cover_staged.json"), J(HERE / "cover_ksteady.json")
card = J(ROOT / "benchmarks/sampling_plan/glm53_blocks.json")["definitions"]
AUDIT_RUN = cover["sources"]["audit_of_record"]
audit = J(Path.home() / f".research/runs/{AUDIT_RUN}/out/sparse-mla-units/summary.json")["rank0"]
audit_ok = audit["audit"]["accepted"]
TREES = {"cir259": "31b1b56aec2a", "r20261112-040916-b4c6": "cda3e822069e", AUDIT_RUN: "da87cdccfcec",
         "ksteady_stage": "62bf29346f76", "ksteady_lean": "33faab2be81c"}
MAIN = cover["tree"]
#: the coordinator's first cut (CIR-271, 20:52Z), a join by exact name
FIRST_CUT = {"own-definition": (6, 8_269_260_792), "boolean-version": (1, 172_200), "none": (247, 3_005_839_000)}

rows_v1 = [r for r in cover["rows"] if r["in_list"]]
inst = {r["template"]: r["instances"] for r in rows_v1}
cut_by = {}
for r in rows_v1:
    for b in r["cut"] or ():
        cut_by.setdefault(b, []).append(r["template"])
uncut = [r["template"] for r in rows_v1 if not r["cut"]]
extra_from = {}
for t in uncut:
    for u, n in cuts[t]["units"]:
        extra_from.setdefault(u, []).append([t, n])


def tree_of(run: str) -> str:
    return TREES.get(run) or TREES["cir259"]


def statements(b: str, def_id: str | None, bv_id: str | None) -> list[dict]:
    """Every Lean verdict read on a statement of `b`: its run, form, Definition, units proved, device and tree."""
    out = []
    for a in acc:
        if b not in (a.get("word"), (a.get("unit") or "").removeprefix("census:")):
            continue
        out.append({"run": a["run"], "tag": a["tag"], "verdict": a["verdict"], "definition": a["definition"], "units": a["n"],
                    "form": "bound", "handed": "the partition object and the Program (B.verify.held)", "device": "cpu",
                    "tree": tree_of(a["run"])})
    for k in ksteady:
        if k["word"] != b:
            continue
        out.append({"run": k["run"], "tag": "lean_verify", "verdict": "accepted" if k["accepted"] else "refused",
                    "definition": k["definition"], "units": k["blocks"], "form": "provisional",
                    "handed": "the Program's word (not the circuit's bound Program) and no partition object; units as stated",
                    "device": (k["prove"] or {}).get("device"), "tree": f"{TREES['ksteady_lean']} (Lean), "
                    f"{TREES['ksteady_stage']} (staged and proved in {k['archive'][:16]})"})
    if b in audit["draw_counts"]:
        out.append({"run": AUDIT_RUN, "tag": "audit", "verdict": "accepted" if audit_ok else "refused", "definition": b,
                    "units": audit["draw_counts"][b], "form": "audit",
                    "handed": "the sampled audit of record (lean-draw and lean-verify per member)", "device": "cpu",
                    "tree": TREES[AUDIT_RUN]})
    for s in out:
        if s["verdict"] != "accepted":
            s["state"] = "none"
        elif s["form"] == "audit" or s["definition"] == def_id:
            s["state"] = "own-definition"
        elif s["definition"] == bv_id or bv_id is None:
            s["state"] = "boolean-version"
        else:
            s["state"] = "other-definition"
    return out


RANK = {"own-definition": 3, "boolean-version": 2, "other-definition": 1, "none": 0}
FORM = {"bound": 2, "audit": 2, "provisional": 1}


def staging_runs(b: str) -> list[dict]:
    out = []
    for s in staged_steps:
        w = s["word"] or (s["name"].removeprefix("census:") if s["name"].startswith("census:") else None)
        if w == b:
            out.append({"run": s["run"], "tag": s["tag"], "rc": s["rc"], "peak_gib": s["peak_gib"], "units": s["n"],
                        "definition": s["definition"], "tree": tree_of(s["run"])})
    for k in ksteady:
        if k["word"] == b and k["stage"]:
            out.append({"run": k["archive"][:16] + "… (" + k["run"] + "'s archive)", "tag": "stage", "rc": k["stage"]["rc"],
                        "peak_gib": k["stage"]["peak_gib"], "units": k["stage"].get("n"), "definition": k["stage"].get("definition"),
                        "tree": TREES["ksteady_stage"]})
    return out


def staged_name(rec: dict, b: str) -> str | None:
    exact = [h for h in rec.get("tables", []) if h["exact"]]
    if exact:
        return exact[0]["name"]
    return f"census:{b}" if rec.get("census_ok") else None


brows = []
for b in list(card) + [u for u in extra_from if u not in card]:
    rec = blocks_in["blocks"].get(b, {})
    d = card.get(b)
    bv = rec.get("boolean_version") or {}
    sts = statements(b, rec.get("def_id"), bv.get("id"))
    best = max(sts, key=lambda s: (RANK[s["state"]], FORM[s["form"]]), default=None)
    state = best["state"] if best else "none"
    runs = staging_runs(b)
    done = [r for r in runs if r["rc"] == 0]
    brows.append({
        "block": b, "on_card": d is not None, "family": d["family"] if d else None,
        "prefill": d["prefill"] if d else 0, "decode": d["decode"] if d else 0,
        "blocks": (d["prefill"] + d["decode"]) if d else 0, "k_log": d["k_log"] if d else None,
        "list_templates": [[t, inst[t]] for t in cut_by.get(b, [])] or [[t, inst[t], n] for t, n in extra_from.get(b, [])] or None,
        "registered": rec.get("template"), "definition": rec.get("def_id"), "owner_file": rec.get("owner_file"),
        "pinned": "this-binding" if rec.get("exact") else ("boolean-version" if bv.get("pins") else "none"),
        "pin_kinds": rec.get("exact") or bv.get("pins") or None, "boolean_version": bv.get("id"),
        "staged": staged_name(rec, b), "zk_refusal": rec.get("zk_refusal"),
        "staging_run": done[0] if done else None, "staging_attempts": [r for r in runs if r["rc"] != 0] or None,
        "accepted": state, "form": best["form"] if best and state != "none" else None,
        "accepted_run": {k: best[k] for k in ("run", "definition", "units", "handed", "device", "tree")} if best and state != "none" else None,
        "statements": sts or None,
        "error": rec.get("error"),
    })

on = [r for r in brows if r["on_card"]]


def tally(key, rows, vals):
    return {v: [sum(1 for r in rows if key(r) == v), sum(r["blocks"] for r in rows if key(r) == v)] for v in vals}


STATES = ("own-definition", "boolean-version", "other-definition", "none")
totals = {
    "card": [len(on), sum(r["blocks"] for r in on)],
    "accepted": tally(lambda r: r["accepted"], on, STATES),
    "accepted_bound": tally(lambda r: r["accepted"] if r["form"] in ("bound", "audit") else "none", on, STATES),
    "staging_run_completed": tally(lambda r: "yes" if r["staging_run"] else "no", on, ("yes", "no")),
    "staged_name": tally(lambda r: "yes" if r["staged"] else "no", on, ("yes", "no")),
    "pinned": tally(lambda r: r["pinned"], on, ("this-binding", "boolean-version", "none")),
    "cut_by_list": tally(lambda r: "yes" if r["list_templates"] else "no", on, ("yes", "no")),
    "extra_units": len(brows) - len(on),
}
assert totals["card"][1] == sum(v[1] for v in totals["accepted"].values())

FIRST_OWN = [k for k in card if k.startswith(("MoeExpertFp8GemmW_v1{E=256,K=256,N=1,", "MoeExpertFp8BlockGemm_v3{K=6144,N=1,"))
             and ("SWAP=true" in k or k.startswith("MoeExpertFp8BlockGemm_v3"))] + ["MoeSum_v1{TOPK=8,H=96}", "RoPEHeadGptJ_v2{D=64}", "SiluMul_v2{I=256}"]
FIRST_BOOL = [k for k in card if k.startswith("MoeRouterSigmoidBias_v1{E=256,TOPK=8,")]
bsum = lambda ks: [len(ks), sum(card[k]["prefill"] + card[k]["decode"] for k in ks)]  # noqa: E731
assert bsum(FIRST_OWN) == list(FIRST_CUT["own-definition"]) and bsum(FIRST_BOOL) == list(FIRST_CUT["boolean-version"]), (bsum(FIRST_OWN), bsum(FIRST_BOOL))
first_state = {k: "own-definition" for k in FIRST_OWN} | {k: "boolean-version" for k in FIRST_BOOL}
vs_first = [{"block": r["block"], "blocks": r["blocks"], "first_cut": first_state.get(r["block"], "none"), "here": r["accepted"],
             "form": r["form"], "run": (r["accepted_run"] or {}).get("run")}
            for r in sorted(on, key=lambda r: -r["blocks"]) if first_state.get(r["block"], "none") != r["accepted"]]

lacks = []
for r in sorted(on, key=lambda r: -r["blocks"]):
    if r["accepted"] != "none" or r["blocks"] <= 3_000_000:
        continue
    why = []
    if not r["registered"]:
        why.append("no registered Definition at this binding" + (f" ({r['error']})" if r["error"] else ""))
    if r["pinned"] == "none":
        why.append("no pin at this binding")
    if not r["staged"]:
        why.append("no stage.py name")
    elif r["zk_refusal"]:
        why.append(f"zk refusal: {r['zk_refusal']}")
    if r["staging_attempts"] and not r["staging_run"]:
        a = r["staging_attempts"][0]
        why.append(f"its staging run {a['run']} ended rc {a['rc']} at {a['peak_gib']} GiB")
    elif not r["staging_run"]:
        why.append("no staging run read")
    if r["statements"]:
        why.append("statements read but none accepted: " + "; ".join(f"{s['run']} {s['verdict']}" for s in r["statements"]))
    if not r["list_templates"]:
        why.append("no list template cuts into it")
    lacks.append({"block": r["block"], "blocks": r["blocks"], "lacks": why})

FIVE = {"RMSNormFusedCudaVector8_v1": "RMSNormFusedCudaVector8_v3", "TokenSelect_v1": "TokenSelect_v4",
        "DsaLatentRow_v1": "DsaLatentRow_v1", "AllGather_v1": "AllGather_v1", "StopCheck_v1": "StopCheck_v1"}
five = []
log = subprocess.check_output(["git", "-C", str(ROOT), "log", "-3", "--format=%h %an <%ae> %ad %s%n%(trailers:key=Co-authored-by,valueonly)",
                               "--date=iso-strict", MAIN, "--", "benchmarks/sampling_plan/glm53_blocks.json"], text=True)
for t in uncut:
    fam = t[: t.index("{")]
    units = []
    for u, n in cuts[t]["units"]:
        rec = blocks_in["blocks"].get(u, {})
        units.append({"unit": u, "per_instance": n, "registered": rec.get("template"), "definition": rec.get("def_id"),
                      "owner_file": rec.get("owner_file"), "pinned": bool(rec.get("exact")),
                      "staged": staged_name(rec, u), "zk_refusal": rec.get("zk_refusal"),
                      "would_add": 8 * inst[t] * n})
    card_has = sorted({k for k, v in card.items() if any(o == fam or o.startswith(fam.rsplit("_v", 1)[0] + "_v") for o in v["of"])})
    five.append({"template": t, "instances_rank0": inst[t], "card_names": card_has or None,
                 "card_blocks": sum(card[k]["prefill"] + card[k]["decode"] for k in card_has),
                 "registered_cuts": units, "whole_would_add": 8 * inst[t]})

FRIDAY_RUN, FRIDAY_TREE, FRIDAY_BRANCH = "r20261112-191841-86b4", "7c2636e740e0", "cursor/proofs-stops-premortem-greedy-7a2e"
fdir = Path(sys.argv[4]) / "out" / "second"
f_st, f_au = J(fdir / "statements.json"), J(fdir / "audit" / "audit.json")
f_out = J(fdir / "summary.json")["outcome"]
#: every member's lean-draw and lean-verify are true, so no map from member to statement is needed (none is read here)
f_all = all(f_au["verdicts"].values()) and len(f_au["verdicts"]) == 3 * len(f_st)
f_ok = {s["template"]: f_all and s["template"] in f_au["statement"] for s in f_st}
f_limits = [f"not main: {FRIDAY_TREE} on {FRIDAY_BRANCH}", "the serve it proved is not checked here (proofs: a stand-in serve of 2 outputs)",
            f"the audit's verdict: accepted {f_au['accepted']}, its verifier of record {f_au['verifiers']['of_record']} failed "
            f"{f_au['verifiers'][f_au['verifiers']['of_record']]['failed']}; outcome {f_out['symbol']} on {f_out['statements']}"]
for r in cover["rows"]:
    if r["in_list"]:
        continue
    t = r["template"]
    if t in f_ok:
        s = next(x for x in f_st if x["template"] == t)
        r["notes"]["friday_run"] = {"run": FRIDAY_RUN, "tree": FRIDAY_TREE, "lean_draw_and_verify": f_ok[t],
                                    "instances": s["header"]["instances"], "drawn": s["draw"]["k"], "limits": f_limits}
        if f_ok[t]:
            r.update(accepted="yes", accepted_definition=None, run=FRIDAY_RUN, source="read")
    elif r["cut"] and all(u in f_ok for u in r["cut"]):
        r["notes"]["friday_run"] = {"run": FRIDAY_RUN, "tree": FRIDAY_TREE, "cut_units_accepted": sum(f_ok[u] for u in r["cut"]),
                                    "cut_units": len(r["cut"]), "limits": f_limits,
                                    "relation": "the registry at the run's tree (greedy_stages.py, the same blob as at the Friday commit) "
                                                "holds GreedySelect_v1 to its body's units (CUTS): 19 GreedyChunk_v2 and GreedyCombine_v2"}
cover["schema"] = "circuits/request-cover/v2"
cover["block_run"] = Path(sys.argv[1]).resolve().parent.parent.name
cover["block_totals"] = totals
cover["first_cut"] = {"totals": FIRST_CUT, "own_definition": FIRST_OWN, "boolean_version": FIRST_BOOL, "differs": vs_first}
cover["runs"] = J(HERE / "cover_runs.json")
fam = lambda t: t.partition("{")[0]  # noqa: E731
OFF_BINDING = {"r20261112-191822-3496": ["DsaMaskPositions_v2{N=40960,AT=0,K=2048}", "DsaTopKMaskWhole_v2{N=40960,TOPK=2048}"],
               "r20261112-195705-2196": ["GreedyChunk_v2{N=128,OFF=0}", "GreedyCombine_v2{V=2560,N=128}"],
               "r20261112-191810-df04": ["GreedyChunk_v2{N=128,OFF=0}", "GreedyCombine_v2{V=2560,N=128}"]}
cover["off_binding"] = [{"run": run, "definition": d, "in_list": d in inst, "family_in_list": sorted(t for t in inst if fam(t) == fam(d))[:5] or None,
                         "on_card": d in card, "family_on_card": sorted(k for k in card if fam(k) == fam(d))[:5] or None}
                        for run, ds in OFF_BINDING.items() for d in ds]
proofs_card = json.loads(subprocess.check_output(["git", "-C", str(ROOT), "show", f"{cover['sources']['friday_at']}:benchmarks/sampling_plan/glm53_blocks.json"]))["definitions"]
cover["proofs_count"] = {
    "card_at": cover["sources"]["friday_at"], "types": len(proofs_card),
    "blocks": sum(v["prefill"] + v["decode"] for v in proofs_card.values()),
    "only_main": [[k, card[k]["prefill"] + card[k]["decode"]] for k in card if k not in proofs_card],
    "only_there": [[k, proofs_card[k]["prefill"] + proofs_card[k]["decode"]] for k in proofs_card if k not in card],
    "changed": [k for k in card if k in proofs_card and (card[k]["prefill"], card[k]["decode"]) != (proofs_card[k]["prefill"], proofs_card[k]["decode"])],
}
cover["lacks_over_3m"] = lacks
cover["card_places"] = {"templates": five, "card_log": log.strip().split("\n")}
cover["blocks"] = brows
rows = cover.pop("rows")
blocks_list = cover.pop("blocks")
head = json.dumps(cover, indent=1)[:-2]
body = ",\n".join(" " + json.dumps(r, separators=(",", ":")) for r in rows)
bbody = ",\n".join(" " + json.dumps(r, separators=(",", ":")) for r in blocks_list)
(HERE / "cover.json").write_text(f'{head},\n "rows": [\n{body}\n ],\n "blocks": [\n{bbody}\n ]\n}}\n')
cover["rows"], cover["blocks"] = rows, blocks_list
assert json.loads((HERE / "cover.json").read_text()) == json.loads(json.dumps(cover))
print(json.dumps(totals, indent=1))
