"""summarize.py OUT: each proof block's numbers (OUT/<tag>/summary.json) and all of them (OUT/summary.json, one table row per
proof block), from
what job.sh leaves there: stage.json and the *.time files, the prover's LIVE lines, the selftests' NEG lines and the Lean
verifier's VERDICT lines. The verifier holds no program for a proof block (`stage.py`: a `UNITS` circuit binds none, and
from a `PARTS` class's program it would derive only whole activations), so it refuses every session once every check
before that refusal has passed (`NO_PROGRAM`): the honest run and the selftest build's honest record pass only when
refused for that alone; every other control passes only when a check rejects it (a forged output word refused)."""
import json
import statistics
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
HONEST = ("honest-own", "st-honest")
#: flock-verify's refusal of a session verified without the verifier's program (`ProvedScope.NO_PROGRAM`)
NO_PROGRAM = "refused: a session without the verifier's program (--program)"
REFUSED = "refused: "


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def tail(p: Path, n: int = 2) -> str:
    try:
        return " | ".join(ln.strip() for ln in p.read_text(errors="replace").strip().splitlines()[-n:])[:600]
    except OSError:
        return ""


def gb(t) -> float | None:
    return round(t["maxrss_kb"] / 2**20, 2) if t else None


def med(xs):
    return round(statistics.median(xs), 3) if xs else None


def lines(p: Path, tag: str) -> list:
    try:
        return [json.loads(ln[len(tag):]) for ln in p.read_text(errors="replace").splitlines() if ln.startswith(tag)]
    except OSError:
        return []


def session(d: Path) -> dict | None:
    runs = lines(d / "prove.out", "LIVE\t")
    if not (d / "prove.out").exists():
        return None
    if not runs:
        return {"error": tail(d / "prove.err", 3) or "no LIVE line"}
    timed = [r for r in runs if not r["warm"]] or runs
    r0, pt = timed[0], load(d / "prove.time")
    return {"accepted": all(r["accepted"] for r in runs), "runs": len(timed), "instances": r0["instances"], "k_log": r0["k_log"],
            "proof_bytes_total": sum(r0["proof_bytes"]), "prove_total_s_median": med([r["prove_total_s"] for r in timed]),
            "rust_verify_s_median": med([r["verdict"]["verify_s"] for r in timed]), "prover_maxrss_gb": gb(pt),
            "prover_wall_s": (pt or {}).get("wall_s"), "serve_maxrss_gb": gb(load(d / "serve.time")),
            "statement_digest": r0["statement_digest"], "circuit_sha512": r0["circuit_sha512"]}


def controls(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("selftest-*.out")):
        neg, t = lines(f, "NEG\t"), load(f.with_suffix(".time")) or {}
        out[f.stem[len("selftest-"):]] = ({"pass": all(x["pass"] for x in neg), "expect": neg[0]["expect"], "accepted": neg[0]["accepted"],
                                          "why": neg[0]["why"][:300], "wall_s": t.get("wall_s")} if neg else
                                         {"pass": False, "exit": t.get("exit"), "error": tail(f.with_suffix(".err"), 3)})
    return out


def lean(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("lean-*.out")):
        tag, t = f.stem[len("lean-"):], load(f.with_suffix(".time")) or {}
        v = lines(f, "VERDICT ")
        why = (v[0].get("why") or "") if v else ""
        got = (None if not v or v[0].get("accepted") is None else "ACCEPT" if v[0]["accepted"] else
               "REFUSED" if why.startswith(REFUSED) else "REJECT")
        expect = "REFUSED" if tag in HONEST else "REJECT"
        ok = got == expect and (tag not in HONEST or why.startswith(NO_PROGRAM))
        out[tag] = {"expect": expect, "verdict": got, "pass": ok, "why": why[:300] if v else tail(f.with_suffix(".err"), 2),
                    "wall_s": t.get("wall_s"), "maxrss_gb": gb(t) if t else None,
                    **({"partition": v[0]["partition"]} if v and v[0].get("partition") else {})}
    return out


def one(d: Path) -> dict:
    st, t = load(d / "stage.json"), load(d / "stage.time")
    rec = {"tag": d.name, "stage_error": None if st else tail(d / "stage.err", 3), "stage_cache": load(d / "stage-cache.json")}
    if st:
        rec.update({k: st.get(k) for k in ("unit", "statics", "n", "definition", "type_definition", "class", "ands", "unit_nnz",
                                           "k_log", "unit_log", "circuit_bytes", "circuit_sha512", "twin", "seconds")},
                   stage_maxrss_gb=gb(t), stage_wall_s=(t or {}).get("wall_s"))
    rec.update(zk=session(d), controls=controls(d), lean=lean(d))
    forged = [c for c in rec["lean"] if c not in HONEST]
    rec["lean_passes_honest"] = bool(rec["lean"]) and all(rec["lean"].get(c, {}).get("pass") for c in HONEST)
    # a rejection counts only beside the honest records passing every check under the same flags: a setup error rejects
    # everything
    rec["forgeries_refused"] = rec["lean_passes_honest"] and bool(forged) and all(rec["lean"][c]["pass"] for c in forged)
    rec["all_pass"] = bool(rec["zk"] and rec["zk"].get("accepted") and rec["lean_passes_honest"] and rec["forgeries_refused"]
                           and all(v["pass"] for v in rec["controls"].values()))
    (d / "summary.json").write_text(json.dumps(rec, indent=1))
    return rec


rows = [one(d) for d in sorted(p for p in OUT.iterdir() if p.is_dir() and (p / "stage.time").exists())]
(OUT / "summary.json").write_text(json.dumps(rows, indent=1))
for r in rows:
    z = r.get("zk") or {}
    print(json.dumps({"unit": r["tag"], "ands": r.get("ands"), "nonzeros": r.get("unit_nnz"), "k_log": r.get("k_log"),
                      "prove_s": z.get("prove_total_s_median"), "peak_gb": z.get("prover_maxrss_gb"),
                      "proof_bytes": z.get("proof_bytes_total"), "lean_verify": r["lean_passes_honest"],
                      "forgery_refused": r["forgeries_refused"], "all_pass": r["all_pass"],
                      "lean": {c: v["verdict"] for c, v in r["lean"].items()}, "error": r.get("stage_error") or z.get("error")}))
