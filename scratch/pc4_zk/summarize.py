"""summarize.py OUT: each config's numbers (OUT/<role>-k<K>/summary.json) and all of them (OUT/summary.json), from the files
job.sh leaves there: stage.json and *.time, the prover's LIVE lines, the selftests' NEG lines, the Lean verifier's outputs."""
import json
import statistics
import sys
from pathlib import Path

OUT = Path(sys.argv[1])


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


def live(p: Path) -> list[dict]:
    try:
        return [json.loads(ln[5:]) for ln in p.read_text().splitlines() if ln.startswith("LIVE\t")]
    except OSError:
        return []


def med(xs):
    return round(statistics.median(xs), 4) if xs else None


def session(d: Path, out: str, serve: str) -> dict | None:
    runs = live(d / out)
    if not runs:
        return {"error": tail(d / out.replace(".out", ".err"), 3) or "no LIVE line"} if (d / out).exists() else None
    timed = [r for r in runs if not r["warm"]] or runs
    r0 = timed[0]
    return {"accepted": all(r["accepted"] for r in runs), "runs": len(timed), "warm": len(runs) - len(timed),
            "instances": r0["instances"], "ands_per_vu": r0["ands_per_vu"], "k_log": r0["k_log"], "dense_m": r0.get("dense_m"),
            "proof_bytes": r0["proof_bytes"], "proof_bytes_total": sum(r0["proof_bytes"]),
            "prove_total_s_median": med([r["prove_total_s"] for r in timed]),
            "prove_rep_s_median": med([x for r in timed for x in r["prove_s"]]),
            "verify_s_median": med([r["verdict"]["verify_s"] for r in timed]),
            "e2e_s_median": med([r["e2e_s"] for r in timed]), "witness_s_median": med([r["witness_s"] for r in timed if r["witness_s"] is not None]),
            "wire_sent": r0["wire_sent"], "wire_recv": r0["wire_recv"], "zk_hooks": bool(r0["identity"]["hooks"].get("compiler")),
            "statement_digest": r0["statement_digest"], "circuit_sha512": r0["circuit_sha512"],
            "prover_maxrss_gb": gb(load(d / out.replace(".out", ".time"))), "serve_maxrss_gb": gb(load(d / serve)),
            "prover_wall_s": (load(d / out.replace(".out", ".time")) or {}).get("wall_s")}


def selftests(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("selftest-*.out")):
        case = f.stem[len("selftest-"):]
        neg = [json.loads(ln[4:]) for ln in f.read_text().splitlines() if ln.startswith("NEG\t")]
        t = load(f.with_suffix(".time")) or {}
        out[case] = ({"pass": all(n["pass"] for n in neg), "expect": neg[0]["expect"], "accepted": neg[0]["accepted"],
                      "why": neg[0]["why"][:300], "exit": t.get("exit"), "wall_s": t.get("wall_s")} if neg else
                     {"pass": False, "exit": t.get("exit"), "error": tail(f.with_suffix(".err"), 3)})
    return out


def lean(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("lean-*.out")):
        t = load(f.with_suffix(".time")) or {}
        text = f.read_text(errors="replace")
        rec = {"exit": t.get("exit"), "wall_s": t.get("wall_s"), "maxrss_gb": gb(t) if t else None,
               "stderr": tail(f.with_suffix(".err"), 2)}
        verdict = [ln for ln in text.splitlines() if ln.startswith("VERDICT ")]
        if verdict:
            v = json.loads(verdict[0][8:])
            rec.update(accepted=v.get("accepted"), why=(v.get("why") or "")[:300])
        else:
            rec["accepted"] = t.get("exit") == 0
            rec["stdout"] = text[:200]
        out[f.stem[len("lean-"):]] = rec
    return out


def one(d: Path) -> dict:
    st = load(d / "stage.json")
    t = load(d / "stage.time")
    rec = {"tag": d.name, "stage_time": t, "stage_error": None if st else tail(d / "stage.err", 3),
           "skipped": (d / "SKIPPED").read_text().strip() if (d / "SKIPPED").exists() else None}
    if st:
        rec.update(role=st["role"], k=st["k"], definition=st["definition"], ands=st["ands"], n=st["n"],
                   instances_x_ands=st["n"] * st["ands"], circuit_k_log=st["k_log"], circuit_bytes=st["circuit_bytes"],
                   circuit_sha512=st["circuit_sha512"], public_inputs=st["public_inputs"], own_bytes=st["own_bytes"],
                   own_equals_c_flock_writer=st["own_equals_c_flock_writer"], stage_seconds=st["seconds"],
                   stage_maxrss_gb=st["maxrss_gb"], flipped=st["flipped"])
    rec.update(zk=session(d, "prove.out", "serve.time"), controls=selftests(d), nz=session(d, "prove-nz.out", "serve-nz.time"),
               lean=lean(d))
    (d / "summary.json").write_text(json.dumps(rec, indent=1))
    return rec


rows = [one(d) for d in sorted(p for p in OUT.iterdir() if p.is_dir() and (p / "stage.time").exists() or (p / "SKIPPED").exists())]
(OUT / "summary.json").write_text(json.dumps(rows, indent=1))
for r in rows:
    z = r.get("zk") or {}
    print(json.dumps({k: r.get(k) for k in ("tag", "ands", "instances_x_ands", "stage_maxrss_gb", "skipped", "stage_error")}
                     | {"zk": {k: z.get(k) for k in ("accepted", "k_log", "dense_m", "proof_bytes_total", "prove_total_s_median",
                                                      "verify_s_median", "prover_maxrss_gb", "error")},
                        "controls": {c: v.get("pass") for c, v in r["controls"].items()},
                        "lean": {c: [v.get("exit"), v.get("accepted")] for c, v in r["lean"].items()}}))
