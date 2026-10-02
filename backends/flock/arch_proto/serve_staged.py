"""serve_staged.sh's driver: a serve session on statements already staged, as they are.

    python3 serve_staged.py BINARY SCRATCH OUT PORT WARM RUNS DIR...

For each staged directory (rec.json, circuit.txt, inst-N.bin, pub-N.bin) the three files are linked into SCRATCH/<shape16> and
``class_statement.prove`` runs there on the GPU (GPU=0: the CPU prover, a dry run): serve on 127.0.0.1, then prove, WARM
warm-up and RUNS timed sessions; the record is the median timed run's, as 70-class-sweep.sh's. OUT/<shape16> keeps live.json
(the record), serve.log, prove.out and prove.err; OUT/summary.jsonl has a line per statement:

- ``build_s``: serve's own statement build ("built in"), once per process, before its first session;
- per session (warm-up first): ``verify_s`` (from Finish to the verdict: the replay of every rep; the build is not in it),
  ``session_s`` (from the session's start), and the verifier's timing and phases summed over the reps;
- ``verify_s_record``: the record's, and ``verify_s_timed_median``.
"""
import json
import os
import re
import shutil
import statistics
import sys
from pathlib import Path

from verity_flock.class_statement import prove


def _sum(into: dict, d: dict) -> None:
    for k, v in d.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            into[k] = into.get(k, 0) + v


def summarize(rec: dict, s: Path, live: dict) -> dict:
    log = (s / "serve.log").read_text() if (s / "serve.log").exists() else ""
    m = re.search(r"built in ([0-9.]+)s", log)
    sessions = []
    for ln in (s / "prove.out").read_text().splitlines() if (s / "prove.out").exists() else []:
        if not ln.startswith("LIVE"):
            continue
        x = json.loads(ln[5:])
        v = x.get("verdict") or {}
        timing, phases = {}, {}
        for r in v.get("reps") or []:
            t = dict(r.get("timing") or {})
            _sum(phases, t.pop("phases", None) or {})
            _sum(timing, t)
        sessions.append({"run": x.get("run"), "warm": x.get("warm"), "accepted": v.get("accepted"), "why": v.get("why"),
                         "verify_s": v.get("verify_s"), "session_s": v.get("session_s"), "handle_s": v.get("handle_s"),
                         "prove_total_s": x.get("prove_total_s"), "e2e_s": x.get("e2e_s"), "wait_s": x.get("wait_s"),
                         "timing": timing, "phases": phases})
    timed = [x["verify_s"] for x in sessions if not x["warm"] and x["verify_s"] is not None]
    st = rec["stage"]
    return {"definition": rec.get("definition"), "shape": rec.get("shape"), "n": st["n"], "k_log": st.get("k_log"),
            "staged_circuit_sha512": st.get("circuit_sha512"), "statement_digest": live.get("statement_digest"),
            "accepted": live.get("accepted"), "build_s": float(m.group(1)) if m else None,
            "build_line": m and log[:m.end()].splitlines()[-1], "verify_s_record": live.get("verify_s"),
            "verify_s_timed_median": statistics.median(timed) if timed else None, "prove_total_s": live.get("prove_total_s"),
            "e2e_s": live.get("e2e_s"), "verifier_peak_gb": live.get("verifier_peak_gb"), "prover_peak_gb": live.get("prover_peak_gb"),
            "proof_bytes": live.get("proof_bytes"), "sessions": sessions}


def main(binary: str, scratch: str, out: str, port: str, warm: str, runs: str, *dirs: str) -> int:
    ok = True
    for d in map(Path, dirs):
        rec = json.loads((d / "rec.json").read_text())
        n, shape = int(rec["stage"]["n"]), rec["shape"][:16]
        s, o = Path(scratch) / shape, Path(out) / shape
        s.mkdir(parents=True)
        o.mkdir(parents=True, exist_ok=True)
        for f in ("circuit.txt", f"inst-{n}.bin", f"pub-{n}.bin"):
            os.symlink(d / f, s / f)
        live = prove(binary, s, n, int(port), gpu=os.environ.get("GPU", "1") == "1", warm=int(warm), runs=int(runs))
        (o / "live.json").write_text(json.dumps(live, indent=1))
        for f in ("serve.log", "prove.out", "prove.err"):
            if (s / f).exists():
                shutil.copyfile(s / f, o / f)
        line = summarize(rec, s, live)
        with open(Path(out) / "summary.jsonl", "a") as fo:
            fo.write(json.dumps(line) + "\n")
        print(f"{line['definition']} n={n}: accepted={line['accepted']} build_s={line['build_s']} "
              f"verify_s={[x['verify_s'] for x in line['sessions']]} record={line['verify_s_record']}", flush=True)
        ok = ok and line["accepted"] is True and all(x["accepted"] for x in line["sessions"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
