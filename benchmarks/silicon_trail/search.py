"""Find the least weight of a cancelling trail (or of a plain trail, or the least active rows after one round).

    python3 search.py --solver KISSAT --w 64 --placement plane --k 3 --lo 2 --hi 160 --jobs 12 --seconds 3000 --out F.json

It asks the solver "is there one of weight at most W?" for several W at once, each a separate solver process, and
narrows [lo, hi]: an UNSAT at W proves every trail weighs more than W, a SAT gives a trail (decoded and checked by
diff.Trail.check, never taken from the solver's word) whose own weight becomes hi.  When fewer W remain than
processes, the spare processes repeat the open W with other seeds and with the solver's --sat and --unsat presets.
It stops when lo = hi (the minimum, proved by the solver's UNSAT at lo - 1) or after --seconds, and writes what it
has: lo, hi, the trail, each solver call with its seconds, and the CPU seconds of all solver processes.

--mode free --rounds N searches plain trails of Keccak-p (no absorb), the setting of the published bounds.
--mode rows searches the least number of active rows of L(a) for a nonzero a in the absorbed lanes.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import diff
import satmodel


CONFIGS = ((), ("--unsat",), ("--sat",))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver", required=True)
    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--placement", default="plane", choices=sorted(satmodel.K.PLACEMENTS))
    ap.add_argument("--mode", default="cancel", choices=("cancel", "free", "rows"))
    ap.add_argument("--k", type=int, default=0, help="steps (cancel)")
    ap.add_argument("--rounds", type=int, default=0, help="rounds (free)")
    ap.add_argument("--lo", type=int, required=True, help="a value already known to be a lower bound")
    ap.add_argument("--hi", type=int, required=True, help="the largest bound asked about")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--seconds", type=float, default=900.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="silicon-trail-"))
    t_start = time.monotonic()
    lo, hi, best = a.lo, None, None     # every trail weighs >= lo; the best found weighs hi
    top = a.hi                          # no bound above this is asked about
    running: dict[int, dict] = {}       # pid -> job
    calls: list[dict] = []
    serial = 0

    def launch(bound: int, variant: int) -> None:
        nonlocal serial
        serial += 1
        m = satmodel.build(a.w, a.placement, bound, k=a.k, rounds=a.rounds, mode=a.mode)
        cnf = work / f"j{serial}.cnf"
        cnf.write_text(m.dimacs())
        size = (m.nvars, len(m.clauses))
        m.clauses = []
        cfg = CONFIGS[variant % len(CONFIGS)]
        cmd = [a.solver, "-q", f"--seed={variant}", *cfg, str(cnf)]
        out = open(work / f"j{serial}.out", "w")
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.DEVNULL)
        running[p.pid] = {"p": p, "bound": bound, "variant": variant, "model": m, "t0": time.monotonic(),
                          "out": work / f"j{serial}.out", "cnf": cnf, "size": size, "cfg": " ".join(cfg)}

    def stop(pid: int, why: str) -> None:
        j = running.pop(pid)
        j["p"].terminate()
        j["p"].wait()
        calls.append({"bound": j["bound"], "variant": j["variant"], "config": j["cfg"], "status": why,
                      "seconds": round(time.monotonic() - j["t0"], 2)})
        j["cnf"].unlink(missing_ok=True)

    def refill() -> None:
        ceiling = (hi - 1) if hi is not None else top
        while len(running) < a.jobs and lo <= ceiling:
            used: dict[int, int] = {}
            for j in running.values():
                used[j["bound"]] = used.get(j["bound"], 0) + 1
            free = [b for b in range(lo, ceiling + 1) if b not in used]
            if free:
                # the unasked bound farthest from every asked one; the ends first
                marks = sorted(used) or []
                if ceiling not in used:
                    pick = ceiling
                elif lo not in used:
                    pick = lo
                else:
                    pick = max(free, key=lambda b: min(abs(b - u) for u in marks))
                launch(pick, 0)
            else:
                pick = min(used, key=lambda b: (used[b], b))
                launch(pick, used[pick])

    refill()
    while running and time.monotonic() - t_start < a.seconds:
        time.sleep(0.2)
        for pid in list(running):
            j = running[pid]
            rc = j["p"].poll()
            if rc is None:
                continue
            running.pop(pid)
            dt = round(time.monotonic() - j["t0"], 2)
            rec = {"bound": j["bound"], "variant": j["variant"], "config": j["cfg"], "seconds": dt,
                   "vars": j["size"][0], "clauses": j["size"][1]}
            j["cnf"].unlink(missing_ok=True)
            if rc == 20:
                rec["status"] = "UNSAT"
                lo = max(lo, j["bound"] + 1)
            elif rc == 10:
                found = satmodel.decode(j["model"], satmodel.parse_model(j["out"].read_text(), j["model"].nvars))
                if isinstance(found, diff.Trail):
                    value, shown = found.weight, found.to_json()
                else:
                    value = found["rows"][0] if a.mode == "rows" else sum(found["weights"])
                    shown = found
                rec["status"], rec["found"] = "SAT", value
                if hi is None or value < hi:
                    hi, best = value, shown
            else:
                rec["status"] = f"exit {rc}"
            calls.append(rec)
            print(f"[{time.monotonic() - t_start:8.1f}s] W={rec['bound']:4d} {rec['status']:6s}"
                  f"{' found ' + str(rec.get('found')) if 'found' in rec else ''} in {dt}s -> lo={lo} hi={hi}", flush=True)
        for pid in list(running):
            b = running[pid]["bound"]
            if b < lo:
                stop(pid, "dropped: below lo")
            elif hi is not None and b >= hi:
                stop(pid, "dropped: at or above hi")
        if hi is not None and lo >= hi:
            break
        refill()
    open_bounds = sorted({j["bound"] for j in running.values()})
    for pid in list(running):
        stop(pid, "stopped: time")
    shutil.rmtree(work, ignore_errors=True)
    cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = {
        "w": a.w, "placement": a.placement, "mode": a.mode, "k": a.k, "rounds": a.rounds,
        "lower_bound": lo, "best_found": hi, "minimum_proved": hi is not None and lo == hi,
        "asked_up_to": top, "open_bounds_at_stop": open_bounds,
        "solver_cpu_seconds": round(cpu.ru_utime + cpu.ru_stime, 1), "wall_seconds": round(time.monotonic() - t_start, 1),
        "jobs": a.jobs, "best": best, "calls": calls,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    what = {"cancel": f"{a.k}-step cancelling trail", "free": f"{a.rounds}-round trail", "rows": "active rows"}[a.mode]
    if result["minimum_proved"]:
        verdict = f"minimum {hi}, proved"
    elif hi is None and lo > top:
        verdict = f"none up to {top}, proved"
    else:
        verdict = f"at least {lo}, best found {hi}, open between"
    print(f"{what}, w={a.w}, {a.placement}: {verdict}; solver CPU {result['solver_cpu_seconds']} s")


if __name__ == "__main__":
    main()
