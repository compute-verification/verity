"""summarize.py OUT: each config's numbers (OUT/<role>-k<K>/summary.json) and all of them (OUT/summary.json), from what job.sh
leaves there: stage.json and the *.time files, the prover's LIVE lines, the selftests' NEG lines and the Lean verifier's
VERDICT lines, each Lean check judged against the verdict it must give, and the HZ_LEAN globs that picked the checks run.
A row unit's checks pass `--public-inputs`, a form flock-verify refuses until its soundness proof covers it (T4): each is
REFUSED, its anchor controls too (a refused statement is never accepted, so they say nothing of the anchors), and passes
when not accepted, an honest check only with the refusal as its reason. A plain tile is refused once every check has
passed, its output row holding two output ports, a form the proof does not cover: its honest checks pass only with a
refusal as their reason (REFUSED). A hiding tile's rows (row-seg/v1, row/v2) are covered and its session drawn, so its
honest checks pass only when accepted. Every other tile check passes only when a check rejects it.
Under HZ_STAGE_ONLY=1 nothing is proved or verified, so a config is `stage_only` and its `all_pass` and `lean_checks` are null."""
import json
import os
import statistics
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
EXPECT = {"honest-own": True, "honest-wrong": False, "honest-chosen": False, "st-honest": True}
REFUSAL = "refused: public input ports (--public-inputs): a statement form the soundness proof does not cover"
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
    return round(statistics.median(xs), 4) if xs else None


def session(d: Path) -> dict | None:
    try:
        runs = [json.loads(ln[5:]) for ln in (d / "prove.out").read_text().splitlines() if ln.startswith("LIVE\t")]
    except OSError:
        return None
    if not runs:
        return {"error": tail(d / "prove.err", 3) or "no LIVE line"}
    timed = [r for r in runs if not r["warm"]] or runs
    r0 = timed[0]
    pt, st = load(d / "prove.time"), load(d / "serve.time")
    return {"accepted": all(r["accepted"] for r in runs), "runs": len(timed), "warm": len(runs) - len(timed),
            "instances": r0["instances"], "ands_per_vu": r0["ands_per_vu"], "k_log": r0["k_log"], "dense_m": r0.get("dense_m"),
            "proof_bytes": r0["proof_bytes"], "proof_bytes_total": sum(r0["proof_bytes"]),
            "prove_total_s_median": med([r["prove_total_s"] for r in timed]),
            "prove_rep_s_median": med([x for r in timed for x in r["prove_s"]]),
            "witness_s_median": med([r["witness_s"] for r in timed if r.get("witness_s") is not None]),
            "rust_verify_s_median": med([r["verdict"]["verify_s"] for r in timed]),
            "e2e_s_median": med([r["e2e_s"] for r in timed]), "wire_sent": r0["wire_sent"], "wire_recv": r0["wire_recv"],
            "statement_digest": r0["statement_digest"], "circuit_sha512": r0["circuit_sha512"],
            "prover_maxrss_gb": gb(pt), "prover_wall_s": (pt or {}).get("wall_s"), "serve_maxrss_gb": gb(st)}


def selftests(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("selftest-*.out")):
        case = f.stem[len("selftest-"):]
        neg = [json.loads(ln[4:]) for ln in f.read_text().splitlines() if ln.startswith("NEG\t")]
        t = load(f.with_suffix(".time")) or {}
        out[case] = ({"pass": all(n["pass"] for n in neg), "expect": neg[0]["expect"], "accepted": neg[0]["accepted"],
                      "why": neg[0]["why"][:300], "wall_s": t.get("wall_s"), "maxrss_gb": gb(t) if t else None} if neg else
                     {"pass": False, "exit": t.get("exit"), "error": tail(f.with_suffix(".err"), 3)})
    return out


def lean(d: Path, refused: bool, covered: bool) -> dict:
    out = {}
    for f in sorted(d.glob("lean-*.out")):
        tag = f.stem[len("lean-"):]
        t = load(f.with_suffix(".time")) or {}
        verdict = [json.loads(ln[8:]) for ln in f.read_text(errors="replace").splitlines() if ln.startswith("VERDICT ")]
        accepted = verdict[0].get("accepted") if verdict else None
        why = (verdict[0].get("why") or "") if verdict else ""
        if refused:
            got = None if accepted is None else "ACCEPT" if accepted else "REFUSED" if why == REFUSAL else "REJECT"
            expect, ok = "REFUSED", got == "REFUSED" or (got == "REJECT" and not EXPECT.get(tag, False))
        else:
            got = None if accepted is None else "ACCEPT" if accepted else "REFUSED" if why.startswith(REFUSED) else "REJECT"
            expect = ("ACCEPT" if covered else "REFUSED") if EXPECT.get(tag, False) else "REJECT"
            ok = got == expect
        out[tag] = {"expect": expect, "verdict": got, "pass": ok,
                    "why": why[:300] if verdict else tail(f.with_suffix(".err"), 2),
                    "exit": t.get("exit"), "wall_s": t.get("wall_s"), "maxrss_gb": gb(t) if t else None}
    return out


def one(d: Path) -> dict:
    st, t = load(d / "stage.json"), load(d / "stage.time")
    rec = {"tag": d.name, "stage_time": t, "stage_error": None if st else tail(d / "stage.err", 3),
           "stage_cache": load(d / "stage-cache.json")}
    if st:
        rec.update({k: st.get(k) for k in ("role", "k", "n", "definition", "binding", "staging", "ands", "k_log", "vus_per_block",
                                           "circuit_bytes", "circuit_sha512", "public_inputs", "own_bytes", "own_sha256",
                                           "flipped", "inst_bytes", "pub_bytes", "seconds")},
                   stage_maxrss_gb=gb(t), stage_wall_s=(t or {}).get("wall_s"))
    role = (st or {}).get("role") or d.name
    rec.update(zk=session(d), controls=selftests(d),
               lean=lean(d, not role.startswith(("tile", "hidden")), role.startswith("hidden")),
               lean_checks=os.environ.get("HZ_LEAN") or "*")
    rec["all_pass"] = bool(rec["zk"] and rec["zk"].get("accepted") and rec["lean"]
                           and all(v["pass"] for v in rec["lean"].values()) and all(v["pass"] for v in rec["controls"].values()))
    if os.environ.get("HZ_STAGE_ONLY", "0") == "1":
        rec.update(stage_only=True, lean_checks=None, all_pass=None)
    (d / "summary.json").write_text(json.dumps(rec, indent=1))
    return rec


rows = [one(d) for d in sorted(p for p in OUT.iterdir() if p.is_dir() and (p / "stage.time").exists())]
(OUT / "summary.json").write_text(json.dumps(rows, indent=1))
for r in rows:
    z = r.get("zk") or {}
    print(json.dumps({k: r.get(k) for k in ("tag", "binding", "staging", "ands", "k_log", "circuit_sha512", "stage_maxrss_gb",
                                            "stage_wall_s", "stage_error", "stage_only", "lean_checks", "all_pass")}
                     | {"zk": {k: z.get(k) for k in ("accepted", "k_log", "proof_bytes_total", "prove_total_s_median",
                                                      "rust_verify_s_median", "prover_maxrss_gb", "error")},
                        "controls": {c: v.get("pass") for c, v in r["controls"].items()},
                        "lean": {c: [v["verdict"], v["wall_s"], v["maxrss_gb"]] for c, v in r["lean"].items()}}))
