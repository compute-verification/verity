"""Run a plan's training tasks on this job's GPU, SLOTS at a time, in plan order.

A task whose lr is "best:GROUP" waits until every sweep task of GROUP has finished, then takes the sweep's lr
with the most bits learned on the training question templates (held-out templates stay for evaluation).
"""
import json
import os
import subprocess
import sys

OUT = os.environ["OUT"]


def result(name):
    p = os.path.join(OUT, "results", name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    plan = json.load(open(sys.argv[1]))
    slots = int(os.environ.get("SLOTS", plan.get("slots", 2)))
    pending = list(plan["tasks"])
    done, running, status = set(), {}, {}
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)

    def resolve(t, force=False):
        lr = t["lr"]
        if not isinstance(lr, str):
            return lr
        group = lr.split(":", 1)[1]
        sweep = [s for s in plan["tasks"] if s.get("sweep") and s.get("group") == group]
        if not force and any(s["name"] not in done for s in sweep):
            return None
        scored = [(r["trainq"]["bits"], r["held"]["bits"], r["lr"]) for s in sweep if (r := result(s["name"]))]
        if scored:
            return max(scored)[2]
        return sweep[0]["lr"] if sweep else None

    while pending or running:
        started = True
        while len(running) < slots and started:
            started = False
            for t in pending:
                lr = resolve(t)
                if lr is None:
                    continue
                cfg = {k: v for k, v in t.items() if k not in ("sweep", "group")}
                cfg["lr"] = lr
                log = open(os.path.join(OUT, "logs", t["name"] + ".log"), "w")
                p = subprocess.Popen([sys.executable, "train.py", json.dumps(cfg)], stdout=log, stderr=subprocess.STDOUT)
                running[p.pid] = (t, p, log)
                pending.remove(t)
                print(f"start {t['name']} lr={lr}", flush=True)
                started = True
                break
        if not running:
            for t in pending:
                t["lr"] = resolve(t, force=True)
            if any(t["lr"] is None for t in pending):
                print("unresolvable tasks:", [t["name"] for t in pending], flush=True)
                break
            continue
        pid, st = os.wait()
        if pid not in running:
            continue
        t, p, log = running.pop(pid)
        log.close()
        done.add(t["name"])
        rc = os.waitstatus_to_exitcode(st)
        status[t["name"]] = rc
        r = result(t["name"])
        msg = (f"acc {r['held']['acc_mean']:.3f} bits {r['held']['bits']:.0f} recall {r['recall_frac']:.3f} "
               f"S/H {r['S_over_H']:.3g} tok/s {r['tok_per_s']:.0f}") if r else "no result"
        print(f"done {t['name']} rc={rc} {msg}", flush=True)
        json.dump(status, open(os.path.join(OUT, "driver_status.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
