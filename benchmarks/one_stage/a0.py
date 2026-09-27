"""One-stage audit, run A0: a template's captured served values through registration, the verifier's draw, M0's prover and
verifier, the Lean verifier's draw checks, and the audit record with its integrity profile; then the negatives.

    PYTHONPATH=<tree>/packages/verity/src:<tree>/backends/numerical/python:<tree>/integrations/vllm:<tree>/protocols/one_stage \\
      python benchmarks/one_stage/a0.py --set DIR --m0-pythonpath PATH --bin flock-circuit --lean flock-verify --out DIR \\
      [--k 16] [--delta 0.01] [--port 7511]

* **Stand-in commitment.** M0 commits the captured values afresh (`frame-v3-sha512`, `hm96-sha512/row/v1` leaves, fresh
  OS salts). The values were read under the served run's `vllm-v1` root and replayed bit-exact there by `vu_export`; that
  link is provenance, not proof, and the registration says so.
* **Partition.** `Q_template_instance` v0 over the population program: one unit per instance, no width rule.
* **Verifier of record.** Lean's `verify` doesn't parse M0's `eb90718f` statement yet, so it is recorded as not run, and
  the audit record is incomplete. Lean checks U1-U3 on the draw; M0's own verifier (`flock-circuit serve`) checks the proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from verity.ir.codec import program_digest
from verity.ir.program import Program
from verity_numerical.bench.input_sets import InputSet
from verity_one_stage import audit, draw as D, partition as P, registration as R

HERE = Path(__file__).resolve().parent
LEAN_VERIFY_NOT_RUN = ("flock-verify parses M0's previous statement formats; the eb90718f format (hm96-sha512 rows in the "
                       "circuit, SHA-512 statement and record) is the Lean verifier lane's follow-up to #113")


def sha512_file(path: Path) -> str:
    return hashlib.sha512(path.read_bytes()).hexdigest()


def run(cmd: list[str], log: Path, env: dict | None = None, timeout: float = 3600) -> subprocess.CompletedProcess:
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})}, timeout=timeout)
    log.write_text(f"$ {' '.join(cmd)}\n# rc {p.returncode} in {time.time() - t:.1f} s\n--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}")
    return p


def stage(set_dir: str, ns: list[int], out: Path, m0_path: str, tampered: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    env = {"PYTHONPATH": m0_path}
    if tampered:
        cmd = [sys.executable, str(HERE / "stage_tampered.py"), "--input-set", set_dir, "--n", str(ns[0]), "--out", str(out)]
    else:
        cmd = [sys.executable, "-m", "verity_flock.circuit", "--input-set", set_dir, "--n", *map(str, ns), "--out", str(out)]
    p = run(cmd, out / "stage.log", env)
    if p.returncode != 0:
        raise RuntimeError(f"staging failed: {p.stderr[-2000:]}")
    text = (out / "circuit.txt").read_text()
    meta = json.loads(text.split("\n", 2)[1][5:])
    return {"dir": out, "meta": meta, "pin": hashlib.sha512(text.encode()).hexdigest()}


def public_header(stage_dir: Path, n: int) -> dict:
    with open(stage_dir / f"pub-{n}.bin", "rb") as f:
        return json.loads(f.readline())


def leaves(meta: dict, n: int) -> dict[str, int]:
    out = {name: n for name, *_ in meta["ports"]}
    out.update({name: n * words for name, words in meta["out_ports"]})
    return out


def stand_in(set_dir: str) -> dict:
    prov = json.loads((Path(set_dir) / "manifest.json").read_text()).get("provenance", {})
    return {"kind": "stand-in", "of": "vllm-v1 commitment of a served run", "row": prov.get("row"), "run": prov.get("research_run") or prov.get("run"),
            "run_root": prov.get("run_root"), "model": prov.get("model"), "program_digests": prov.get("program_digests"),
            "link": "provenance, not proof: vu_export read these values under run_root and replayed each bit-exact"}


def session(bin_: str, st: dict, n: int, law: str, port: int, out: Path) -> tuple[dict | None, dict | None]:
    """One live session: `serve --draw` (the verifier's side: registration, the draw, coins, the verdict) and `prove`."""
    out.mkdir(parents=True, exist_ok=True)
    d = st["dir"]
    tables = d / "tables"
    serve_log = open(out / "serve.log", "w")
    serve = subprocess.Popen([bin_, "serve", "--listen", f"127.0.0.1:{port}", "--out", str(out / "sessions"),
                              "--instances", str(d / f"pub-{n}.bin"), "--circuit", str(d / "circuit.txt"), "--tables", str(tables),
                              "--pin", st["pin"], "--operator", "one-stage-a0", "--draw", law, "--sessions", "1"],
                             stdout=serve_log, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(1800):
            if "SERVING" in (out / "serve.log").read_text() or serve.poll() is not None:
                break
            time.sleep(0.5)
        p = run([bin_, "prove", "--verifier", f"127.0.0.1:{port}", "--instances", str(d / f"inst-{n}.bin"),
                 "--circuit", str(d / "circuit.txt"), "--tables", str(tables), "--runs", "1", "--warm", "0"], out / "prove.log")
        live = next((json.loads(line[5:]) for line in p.stdout.splitlines() if line.startswith("LIVE\t")), None)
        if live is None:
            live = {"accepted": False, "why": f"prover exited {p.returncode}: {p.stderr[-600:]}"}
        serve.wait(timeout=600)
    finally:
        if serve.poll() is None:
            serve.kill()
        serve_log.close()
    recs = sorted((out / "sessions").glob("*/session.json"))
    return (json.loads(recs[-1].read_text()) if recs else None), live


def lean_draw_test(lean: str, cases: list[dict], out: Path, name: str) -> list[dict]:
    f = out / f"{name}.cases.json"
    f.write_text(json.dumps(cases))
    p = run([lean, "draw-test", str(f)], out / f"{name}.lean.log")
    return [json.loads(line.split(" ", 1)[1]) for line in p.stdout.splitlines() if line.startswith("DRAWCHECK")]


def proved_units(header: dict, live: dict) -> list[int]:
    """The drawn statement's units as global unit indices: the population header's index of each drawn position."""
    pop = header["units"]["indices"]
    return [pop[u] for u in (live.get("unit_draw") or {}).get("units", [])]


def audit_session(rec: dict, program, query, header: dict, law: dict, record, live, lean: str, out: Path, delta: float) -> dict:
    """Every check on one session, then the audit record."""
    draw = (record or {}).get("unit_draw")
    verifier = (record or {}).get("verdict") or {}
    lean_u = lean_draw_test(lean, [{"record": draw, "statement": (live or {}).get("unit_draw"),
                                    "instances": (live or {}).get("instances", 0)}], out, "u1-u3") if draw else []
    statement = {"circuit_sha512": (live or {}).get("circuit_sha512"), "statement_digest": (live or {}).get("statement_digest"),
                 "public_sha512": (live or {}).get("public_sha512"), "sigma": (live or {}).get("sigma"),
                 "roots": (live or {}).get("roots"), "coin_commitment": ((record or {}).get("coin_seed") or {}).get("commitment"),
                 "roots_are_registered": (live or {}).get("roots") == rec["roots"]}
    verdicts = {
        "m0-verifier": {"accepted": verifier.get("accepted") is True and statement["roots_are_registered"],
                        "why": verifier.get("why"), "tool": "flock-circuit serve (M0, PR #83 eb90718f)"},
        "lean-draw-u1-u3": {"accepted": bool(lean_u) and all(c["ok"] for c in lean_u), "detail": lean_u,
                            "tool": "flock-verify draw-test (PR #113 9d0d53df)"},
        "lean-verify": {"accepted": None, "why": LEAN_VERIFY_NOT_RUN, "tool": "flock-verify verify --archive"},
    }
    if draw is None:
        return {"accepted": False, "why": "no draw recorded", "verdicts": verdicts, "statement": statement}
    return audit.audit_record(rec, draw, law, proved_units(header, live or {}), (live or {}).get("instances", -1), statement,
                              verdicts, delta)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--m0-pythonpath", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--port", type=int, default=7511)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = InputSet.open(a.set)
    n = s.n
    template = s.subcircuit.definition()
    law_text = f"subset:{a.k}"
    law = D.parse_law(law_text)
    summary: dict = {"set": a.set, "subcircuit": s.subcircuit.id, "n": n, "law": law, "delta": a.delta}

    # 1. serving commits (the stand-in): M0 stages the captured values under frame-v3-sha512 / hm96-sha512
    t = time.time()
    st = stage(a.set, [4, n], out / "stage", a.m0_pythonpath)
    header = public_header(st["dir"], n)
    summary["stage_s"] = round(time.time() - t, 1)

    # 2. the partition: Q_template_instance v0 over the verifier's own population program
    program = P.population_program(template, n)
    query = P.template_instance_query(template.id)
    head_digest = program_digest(Program(template))
    summary["template_program"] = {"statement": st["meta"]["program_digests"], "verifier": {template.id: head_digest},
                                   "ok": st["meta"]["program_digests"].get(template.id) == head_digest}

    # 3. registration, before any draw; the verifier holds its own public file to it
    fv = header["frame_v3"]
    rec = R.record(header["scheme"], fv["roots"], leaves(st["meta"], n), fv["domain_ids"], program, query, stand_in(a.set))
    (out / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    checks = {"registration": R.check(rec, program, query).to_json(),
              "public": R.matches_public(rec, header["scheme"], fv["roots"], fv["domain_ids"], header["instances"]).to_json()}
    summary["registration_digest"] = R.digest(rec)
    summary["partition_digest"] = rec["partition_digest"]
    if not all(c["ok"] for c in checks.values()) or not summary["template_program"]["ok"]:
        summary["checks"] = checks
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print("A0 " + json.dumps({"refused_before_draw": checks}))
        return 1

    # 4-6. the verifier draws after Register, M0 proves the drawn units, M0's verifier and Lean's draw checks
    t = time.time()
    record, live = session(a.bin, st, n, law_text, a.port, out / "session")
    summary["session_s"] = round(time.time() - t, 1)
    result = audit_session(rec, program, query, header, law, record, live, a.lean, out / "session", a.delta)
    result["checks_before_draw"] = checks
    result["timing"] = {k: (live or {}).get(k) for k in ("e2e_s", "witness_s", "prove_total_s", "tp_s", "proof_bytes", "k_log", "units")}
    result["timing"]["verify_s"] = ((record or {}).get("verdict") or {}).get("verify_s")
    (out / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    summary["audit"] = {k: result.get(k) for k in ("accepted", "complete", "proved_units", "wrong_units", "checks")}
    summary["audit"]["verdicts"] = {k: v.get("accepted") for k, v in result.get("verdicts", {}).items()}
    summary["timing"] = result["timing"]

    # negatives: each must be refused
    neg: dict[str, dict] = {}
    other = dict(rec, roots={**rec["roots"], next(iter(rec["roots"])): "00" * 64})
    v = R.matches_public(other, header["scheme"], fv["roots"], fv["domain_ids"], header["instances"])
    neg["registration-root-mismatch"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = R.matches_public(rec, header["scheme"], fv["roots"], {**fv["domain_ids"], next(iter(rec["roots"])): "00" * 64},
                         header["instances"])
    neg["public-file-domain-mismatch"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = R.check(rec, P.population_program(template, n - 1), query)
    neg["registration-partition-mismatch"] = {"refused": not v.ok, "codes": list(v.codes)}
    draw = (record or {}).get("unit_draw") or {"law": "subset", "population": n, "k": a.k, "units": list(range(a.k))}
    for name, bad in (("draw-over-another-population", {**draw, "population": n + 1}),
                      ("draw-outside-the-partition", {**draw, "units": draw["units"][:-1] + [n]}),
                      ("draw-under-another-law", {**draw, "k": a.k - 1, "units": draw["units"][:-1]})):
        v = D.check_draw(bad, rec, law)
        neg[name] = {"refused": not v.ok, "codes": list(v.codes)}
    altered = {**draw, "units": sorted(set(draw["units"][:-1]) | {next(u for u in range(n) if u not in draw["units"])})}
    lean_alt = lean_draw_test(a.lean, [{"record": altered, "statement": draw, "instances": len(draw["units"])},
                                       {"record": draw, "statement": draw, "instances": len(draw["units"]) + 1},
                                       {"record": None, "statement": draw, "instances": len(draw["units"])}], out, "negatives")
    neg["draw-altered-after-session (Lean U1)"] = {"refused": bool(lean_alt) and not lean_alt[0]["ok"], "lean": lean_alt[:1],
                                                   "proved-check": D.check_proved(altered, proved_units(header, live or {}),
                                                                                  (live or {}).get("instances", -1)).to_json()}
    neg["instances-not-the-draw (Lean U3)"] = {"refused": len(lean_alt) > 1 and not lean_alt[1]["ok"], "lean": lean_alt[1:2]}
    neg["draw-dropped-from-record (Lean U1)"] = {"refused": len(lean_alt) > 2 and not lean_alt[2]["ok"], "lean": lean_alt[2:3]}
    for case in ("unit_draw_ignored_refused", "unit_draw_session"):
        p = run([a.bin, "selftest", "--instances", str(st["dir"] / "inst-4.bin"), "--circuit", str(st["dir"] / "circuit.txt"),
                 "--tables", str(st["dir"] / "tables"), "--only", case], out / f"selftest-{case}.log")
        res = next((json.loads(line[4:]) for line in p.stdout.splitlines() if line.startswith("NEG\t")), None)
        key = "prover-ignores-the-draw (M0 selftest)" if case == "unit_draw_ignored_refused" else "m0-selftest-unit-draw-session"
        neg[key] = {"refused" if case == "unit_draw_ignored_refused" else "passed": bool(res and res.get("pass")), "case": res}
    tn = 8
    tst = stage(a.set, [tn], out / "stage-tampered", a.m0_pythonpath, tampered=True)
    theader = public_header(tst["dir"], tn)
    tprog = P.population_program(template, tn)
    tfv = theader["frame_v3"]
    trec = R.record(theader["scheme"], tfv["roots"], leaves(tst["meta"], tn), tfv["domain_ids"], tprog, query,
                    {"kind": "stand-in", "of": "tampered copy of the captured set: bit 0 of every instance's first output word flipped"})
    trecord, tlive = session(a.bin, tst, tn, "subset:2", a.port + 1, out / "session-tampered")
    tres = audit_session(trec, tprog, query, theader, D.parse_law("subset:2"), trecord, tlive, a.lean, out / "session-tampered", a.delta)
    neg["wrong-output-word"] = {"refused": trecord is not None and tres.get("accepted") is False and tres.get("profile") is None,
                                "verifier": ((trecord or {}).get("verdict") or {}).get("why"),
                                "prover": (tlive or {}).get("why"), "draw": (trecord or {}).get("unit_draw")}
    (out / "negatives.json").write_text(json.dumps(neg, indent=1, sort_keys=True))
    summary["negatives"] = {k: v.get("refused", v.get("passed")) for k, v in neg.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    ok = bool(result.get("accepted")) and all(summary["negatives"].values())
    print("A0 " + json.dumps(summary, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
