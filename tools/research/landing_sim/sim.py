"""Landing simulator: replays the ready marks of the last 14 days under the current drive and under the tree queue.

Arrivals are real: every PR head marked ready since START that landed (trace.json) or still waits (waiting.json), at its
mark's time, with its real review delay (a PR with no verdict yet draws one from the landed PRs' delays) and whether its
change is a Lean change (the landed diff; a waiting PR draws it at the landed share). What the trace can't give is a
parameter, calibrated against the checked era's observed throughput, main's move rate and ready-to-main percentiles:
whether a change breaks the check (`p_break`, drawn per PR from its number, the same in every policy), how long a fix takes
to land, how long a run takes.

Policies share one clock (minutes since START) and one PR set. `current` models drive as it ran since 29 Oct: a pass of
`pass0 + pass1 * len(next)` minutes; admission of at most `admit` marked PRs a pass; Lean checks (`lean_min`, at most
`max_lean`) that pin the prefix they cover, so an unreviewed member inside a pinned prefix stops every landing above it;
verdict-first reordering above the pinned prefix; landing of the longest reviewed prefix onto `checked`, low-risk members on
their mark; one after-landing run at a time (`run_min`), green moving `main` to the run's tip, red holding `main` until the
breaker's fix-forward lands (`fix_min` after the red) and a later run passes. Levers switch single mechanisms off.
`tree` models the design: each PR's own cached check at its mark (`own_min`, `own_lean_min`) beside its review; a landing
loop every `step` minutes that stacks up to `spec` speculative candidate trees, each main ⊕ every PR ready to land, checked
incrementally (`int_min`; `int_lean_min` when it holds two Lean changes); a red candidate names its breaker by the failed
task's inputs (`name_p`, else a bisection of log2 n integrations), ejects it with nothing landed, and rechecks itself and the
candidates above it; `main` moves to each green candidate in order. An ejected breaker comes back fixed `fix_min` later and
is reviewed again.
"""
import os

#: the study's data directory: trace.py writes it, the rest read it (`research data fetch art:7ee7a884...` restores it)
DATA = os.environ.get("LANDING_STUDY_DIR", "/tmp/ld")
import argparse
import bisect
import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime

START = datetime(2026, 10, 29, tzinfo=UTC).timestamp()
NOW = datetime(2026, 11, 5, 17, 30, tzinfo=UTC).timestamp()
LOWRISK_EXCL = ("verity/core/", "verity/ml/circuits/", "verity/ml/kernels/", "verity/ml/commitments/", "verity/ml/crypto/",
                "verity/ml/randomness/", "tools/verity/lean/", "tests/", "tools/verity/check/", ".gitattributes", "drive.py",
                "queue.py", "infra/", "tools/research/", "tools/agent-guard/", ".cursor/", "pyproject.toml", "uv.lock",
                "lean-audit.json", "TRUSTED.sha256", "lean-toolchain", "lake-manifest.json")


def excluded(f):
    """drive.toml's lowrisk_excludes: a directory from the root (ending in `/`), else a file name in any directory."""
    for x in LOWRISK_EXCL:
        if x.endswith("/"):
            if f.startswith(x):
                return True
        elif f.rsplit("/", 1)[-1] == x or f.endswith(x) and "*" not in x and "." in x and f.rsplit("/", 1)[-1].endswith(x):
            return True
    return f.endswith(".lean") or "/lakefile." in f or f.startswith("lakefile.")


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def u(n, salt):
    """A uniform draw fixed by the PR number and a salt, the same in every policy."""
    return int(hashlib.sha256(f"{n}:{salt}".encode()).hexdigest()[:12], 16) / 16 ** 12


@dataclass
class P:
    n: int
    r: float                 # ready mark, minutes since START
    v: float                 # review delay, minutes
    lean: bool
    lowrisk: bool
    breaks: bool
    landed_obs: bool
    main_obs: float | None = None
    # simulation state
    landed: float | None = None
    main: float | None = None
    fixed_at: float | None = None
    tries: int = 0


def load(p_break, seed_salt="b"):
    trace = [r for r in json.load(open(DATA + "/trace.json")) if r["pr"] and r["ready_last"]]
    delays = sorted(max(0.0, (r["reviewed_at"] - r["ready_last"]) / 60) for r in trace if r["reviewed_at"])
    lean_share = sum(r["lean"] for r in trace if r["main_at"] and r["main_at"] > START) / max(
        1, sum(1 for r in trace if r["main_at"] and r["main_at"] > START))
    prs = {}
    for r in trace:
        if r["ready_last"] < START and not (r["main_at"] and r["main_at"] >= START):
            continue
        lowrisk = not r["lean"] and not any(excluded(f) for f in r["files"])
        v = max(0.0, (r["reviewed_at"] - r["ready_last"]) / 60) if r["reviewed_at"] else 0.0
        prs[r["pr"]] = P(r["pr"], (r["ready_last"] - START) / 60, v, r["lean"], lowrisk,
                         u(r["pr"], seed_salt) < p_break, True, (r["main_at"] - START) / 60 if r["main_at"] else None)
    for x in json.load(open(DATA + "/waiting.json")):
        if x["pr"] in prs:
            continue
        ready = sorted(ts(a["ts"]) for a in x["labels"] if a["key"] == "ready" and a["value"] is True)
        if not ready:
            continue
        rev = sorted(ts(a["ts"]) for a in x["labels"] if a["key"] == "reviewed" and a["value"] == "land")
        if rev:
            v = max(0.0, (rev[0] - ready[-1]) / 60)
        else:
            # censored: a verdict not given by NOW; draw from the landed delays past the time already waited
            waited = (NOW - ready[-1]) / 60
            tail = [d for d in delays if d > waited] or [waited + 60]
            v = tail[int(u(x["pr"], "v") * len(tail))]
        lean = u(x["pr"], "lean") < lean_share
        prs[x["pr"]] = P(x["pr"], (ready[-1] - START) / 60, v, lean, (not lean) and u(x["pr"], "lr") < 0.42,
                         u(x["pr"], seed_salt) < p_break, False)
    return sorted(prs.values(), key=lambda p: p.r), delays


# ---------------------------------------------------------------- current drive


def sim_current(prs, a, rng):
    """drive since 29 Oct, with levers switched by `a`."""
    t = 0.0
    end = (NOW - START) / 60 + 30 * 1440
    waiting = list(prs)                  # in mark order
    wi = 0
    pending = []                         # marked, not yet admitted
    nxt = []                             # members of next, bottom first
    lean_runs = {}                       # pr -> (start, end) of the Lean check that covers up to it
    checked = []                         # landed, not on main: (pr, landed_at)
    run = None                           # (start, end, tip_index)
    red_until = {}                       # breaker pr -> time its fix lands
    moves = []
    breakers_seen = set()
    diag = []
    while t < end:
        if t < 6.5 * 1440:
            diag.append((len(pending), len(nxt), len(checked)))
        while wi < len(waiting) and waiting[wi].r <= t:
            pending.append(waiting[wi])
            wi += 1
        if wi >= len(waiting) and not pending and not nxt and not checked and run is None:
            break
        # admission
        k = len(pending) if a["admit"] is None else a["admit"]
        adm, pending = pending[:k], pending[k:]
        if a.get("nonlean_first"):
            # what a sync adds goes non-Lean first: below the lowest Lean member no check holds in place yet
            held = max((i + 1 for i, p in enumerate(nxt) if p.n in lean_runs), default=0)
            low = next((i for i in range(held, len(nxt)) if nxt[i].lean), len(nxt))
            nxt = nxt[:low] + [p for p in adm if not p.lean] + nxt[low:] + [p for p in adm if p.lean]
        else:
            nxt += adm
        # Lean checks: pin the prefix up to the highest running or passed check
        running = sum(1 for (s, e) in lean_runs.values() if e > t)
        for i, p in enumerate(nxt):
            if p.lean and p.n not in lean_runs and running < a["max_lean"]:
                lean_runs[p.n] = (t, t + a["lean_min"])
                running += 1
        pinned = 0
        if a["pin"]:
            for i, p in enumerate(nxt):
                if p.n in lean_runs:
                    pinned = i + 1
        # verdict-first above the pinned prefix
        rev = lambda p: t >= p.r + p.v or (a["lowrisk"] and p.lowrisk)
        head, tail = nxt[:pinned], nxt[pinned:]
        tail = [p for p in tail if rev(p)] + [p for p in tail if not rev(p)]
        if not a["pin"]:
            head = [p for p in head if rev(p)] + [p for p in head if not rev(p)]
        nxt = head + tail
        # land: the longest prefix whose members have verdicts and, if Lean, a passed check
        ok = lambda p: rev(p) and (not p.lean or (p.n in lean_runs and lean_runs[p.n][1] <= t))
        landed = []
        if a["set_landing"]:
            landed = [p for p in nxt if ok(p)]
        else:
            for p in nxt:
                if not ok(p):
                    break
                landed.append(p)
        ids = {p.n for p in landed}
        nxt = [p for p in nxt if p.n not in ids]
        for p in landed:
            p.landed = t
            checked.append(p)
            lean_runs.pop(p.n, None)
        # after-landing run
        if run and run[1] <= t:
            s, e, tip = run
            covered = checked[:tip]
            red = [p for p in covered if p.breaks and (p.n not in red_until or red_until[p.n] > s)]
            flake = rng.random() < a["flake"]
            if not red and not flake:
                for p in covered:
                    p.main = e
                checked = checked[tip:]
                moves.append(e)
            else:
                for p in red:
                    if p.n not in red_until:
                        red_until[p.n] = e + a["fix_min"]
                        breakers_seen.add(p.n)
                if a["revert"]:
                    # the culprit leaves checked at once; it comes back fixed, through the queue again
                    for p in red:
                        checked.remove(p)
                        p.landed = None
                        p.r = e + a["fix_min"]
                        p.v = p.v
                        p.breaks = False
                        red_until.pop(p.n, None)
                        bisect.insort(waiting, p, lo=wi, key=lambda q: q.r)
            run = None
        if run is None and checked:
            # a fix-forward that hasn't landed keeps the run red; the run waits for a marked fix up to the window
            if a["revert"] or not any(p.breaks and red_until.get(p.n, 1e18) > t for p in checked if p.n in red_until):
                run = (t, t + a["run_min"], len(checked))
            elif all(red_until[p.n] <= t for p in checked if p.n in red_until):
                run = (t, t + a["run_min"], len(checked))
        t += a["pass0"] + a["pass1"] * len(nxt)
    return moves, diag


# ---------------------------------------------------------------- tree queue


def sim_tree(prs, a, rng):
    step = a["step"]
    t = 0.0
    end = (NOW - START) / 60 + 30 * 1440
    order = list(prs)
    ready = []                   # own check green and reviewed, not in a candidate
    pool = list(prs)
    cands = []                   # dicts: members, start, end, red
    moves = []
    cores = []
    pi = 0
    pool.sort(key=lambda p: p.r)
    waiting = []
    while t < end:
        while pi < len(pool) and pool[pi].r <= t:
            waiting.append(pool[pi])
            pi += 1
        if pi >= len(pool) and not waiting and not cands:
            break
        own = lambda p: a["own_lean_min"] if p.lean else a["own_min"]
        rv = lambda p: 0 if (a["lowrisk"] and p.lowrisk) else p.v
        now_ready = [p for p in waiting if t >= p.r + max(rv(p), own(p))]
        ids = {p.n for p in now_ready}
        waiting = [p for p in waiting if p.n not in ids]
        ready += now_ready
        # finish candidates in order
        while cands and cands[0]["end"] <= t:
            c = cands.pop(0)
            bad = [p for p in c["members"] if p.breaks]
            if not bad:
                for p in c["members"]:
                    p.main = c["end"]
                moves.append(c["end"])
                continue
            # red: eject breakers, recheck this candidate and every one above, nothing landed
            extra = 0 if rng.random() < a["name_p"] else a["int_min"] * math.ceil(math.log2(max(2, len(c["members"]))))
            for p in bad:
                c["members"].remove(p)
                p.breaks = False
                p.r = c["end"] + a["fix_min"]
                p.tries += 1
                bisect.insort(pool, p, lo=pi, key=lambda q: q.r)
            redo = a["int_min"] + extra
            c["start"], c["end"] = t, t + redo
            cands.insert(0, c)
            for d in cands[1:]:
                d["members"] = [p for p in d["members"] if p not in bad]
                d["end"] = max(d["end"], c["end"]) + 0  # rechecked beside it, incrementally
            break
        if len(cands) < a["spec"] and ready:
            m = ready
            ready = []
            dur = a["int_lean_min"] if sum(p.lean for p in m) >= 2 else a["int_min"]
            s = max(t, cands[-1]["start"] if cands else t)
            cands.append({"members": m, "start": t, "end": t + dur})
        # cores in use: own checks running, candidates running
        own_running = sum(1 for p in waiting if p.r <= t < p.r + own(p))
        lean_own = sum(1 for p in waiting if p.lean and p.r <= t < p.r + own(p))
        cores.append((own_running - lean_own) * a["own_cores"] + lean_own * a["lean_cores"] + len(cands) * a["int_cores"])
        t += step
    return moves, cores


# ---------------------------------------------------------------- report


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    lo = int(k)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (k - lo)


def report(name, prs, moves, lo_day=0, hi_day=6.5, extra="", diag=None):
    """Ready-to-main of the PRs marked in [START+lo_day, START+hi_day) days (the checked era), every one run to main."""
    sel = [p for p in prs if lo_day * 1440 <= p.r0 < hi_day * 1440]
    lat = [(p.main - p.r0) / 60 for p in sel if p.main is not None]
    never = sum(1 for p in sel if p.main is None)
    per_day = len([m for m in moves if lo_day * 1440 <= m < hi_day * 1440]) / (hi_day - lo_day)
    landed_in = sum(1 for p in prs if p.main is not None and lo_day * 1440 <= p.main < hi_day * 1440) / (hi_day - lo_day)
    row = {"policy": name, "n": len(sel), "p50_h": round(q(lat, .5), 1), "p90_h": round(q(lat, .9), 1),
           "mean_h": round(statistics.mean(lat), 1) if lat else None, "never": never,
           "main_moves_per_day": round(per_day, 1), "to_main_per_day": round(landed_in, 0),
           "pr_hours_per_week": round(sum(lat) / (hi_day - lo_day) * 7)}
    print(f"{name:42s} n={row['n']:4d} p50={row['p50_h']:6.1f}h p90={row['p90_h']:6.1f}h mean={row['mean_h']}h "
          f"moves/day={row['main_moves_per_day']:4.1f} to-main/day={row['to_main_per_day']:4.0f} PRh/wk={row['pr_hours_per_week']} {extra}")
    return row


CURRENT = dict(admit=4, pass0=40, pass1=0.3, max_lean=10, lean_min=82, pin=True, lowrisk=True, set_landing=False,
               run_min=150, flake=0.1, fix_min=360, revert=False, review_cap=None, land_is_main=False,
               nonlean_first=False)
TREE = dict(review_cap=None, step=5, own_min=20, own_lean_min=82, lowrisk=True, spec=4, int_min=15, int_lean_min=45, name_p=0.9,
            fix_min=360, own_cores=8, lean_cores=32, int_cores=64)


def run(policy, params, p_break, seed=1, load_x=1):
    prs, _ = load(p_break)
    if load_x > 1:
        extra = []
        for k in range(1, load_x):
            for p in prs:
                if p.r >= 0:
                    c = P(p.n + 100000 * k, p.r + 1440 * u(p.n, f"shift{k}") * 0.5, p.v, p.lean, p.lowrisk,
                          u(p.n + 100000 * k, "b") < p_break, False)
                    extra.append(c)
        prs = sorted(prs + extra, key=lambda p: p.r)
    for p in prs:
        p.r0 = p.r
        if params.get("review_cap") is not None:
            p.v = min(p.v, params["review_cap"])
    rng = random.Random(seed)
    if policy == "current":
        moves, diag = sim_current(prs, params, rng)
        if params.get("land_is_main"):
            for p in prs:
                p.main = p.landed if p.landed is not None else p.main
        return prs, moves, ("diag", diag)
    moves, cores = sim_tree(prs, params, rng)
    return prs, moves, cores


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-break", type=float, default=0.04)
    ap.add_argument("--out", default=DATA + "/sim_results.json")
    args = ap.parse_args()
    rows = []
    prs, _ = load(args.p_break)
    for p in prs:
        p.r0 = p.r
    obs = [(p.main_obs - p.r0) / 60 for p in prs if p.main_obs and 0 <= p.r0 < 6.5 * 1440]
    print(f"observed (landed only, marked in the checked era) n={len(obs)} p50={q(obs, .5):.1f}h p90={q(obs, .9):.1f}h")
    print(f"arrivals marked/day in the checked era: "
          f"{sum(1 for p in prs if 0 <= p.r0 < 6.5 * 1440) / 6.5:.0f}, backlog at start {sum(1 for p in prs if p.r0 < 0)}, lean share "
          f"{sum(p.lean for p in prs) / len(prs):.2f}, lowrisk share {sum(p.lowrisk for p in prs) / len(prs):.2f}")
    variants = [
        ("current, review p90 capped at 60 min", "current", {"review_cap": 60}),
        ("current (calibrated)", "current", {}),
        ("current + revert on red (no fix-forward wait)", "current", {"revert": True}),
        ("current + nonlean_first", "current", {"nonlean_first": True}),
        ("current + no prefix pin (set landing)", "current", {"pin": False, "set_landing": True}),
        ("current + admission unbounded, 10-min pass", "current", {"admit": None, "pass0": 10, "pass1": 0.0}),
        ("current + run 60 min (fine-grained cached)", "current", {"run_min": 60}),
        ("current + flake 0", "current", {"flake": 0.0}),
        ("current + Lean check 45 min", "current", {"lean_min": 45}),
        ("current + land_into main (check after)", "current", {"land_is_main": True}),
        ("current + review cap 120 min", "current", {"review_cap": 120}),
        ("current + no pin + revert + admission + pass", "current", {"pin": False, "set_landing": True, "revert": True,
                                                                  "admit": None, "pass0": 10, "pass1": 0.0}),
        ("current + all of the above", "current", {"revert": True, "pin": False, "set_landing": True, "admit": None,
                                                  "pass0": 10, "pass1": 0.0, "run_min": 60, "flake": 0.0,
                                                  "lean_min": 45}),
        ("tree queue (own 20/82, int 15/45, spec 4)", "tree", {}),
        ("tree queue, Lean chain 45", "tree", {"own_lean_min": 45, "int_lean_min": 30}),
        ("tree queue, spec 1 (no speculation)", "tree", {"spec": 1}),
        ("tree queue, name_p 0 (always bisect)", "tree", {"name_p": 0.0}),
        ("tree queue, review capped at 60 min", "tree", {"review_cap": 60}),
        ("tree queue, review capped at 120 min", "tree", {"review_cap": 120}),
        ("tree queue, lowrisk off (every PR reviewed)", "tree", {"lowrisk": False}),
        ("tree queue, Lean 45, review cap 60", "tree", {"review_cap": 60, "own_lean_min": 45, "int_lean_min": 30}),
        ("current, 2x arrivals", "current", {"load_x": 2}),
        ("tree queue, 2x arrivals", "tree", {"load_x": 2}),
        ("tree queue, 3x arrivals", "tree", {"load_x": 3}),
    ]
    for name, pol, over in variants:
        params = dict(CURRENT if pol == "current" else TREE)
        load_x = over.pop("load_x", 1)
        params.update(over)
        prs, moves, cores = run(pol, params, args.p_break, load_x=load_x)
        extra = ""
        if cores and cores[0] == "diag":
            d = cores[1]
            extra = (f"pending mean={statistics.mean(x[0] for x in d):.0f} next mean={statistics.mean(x[1] for x in d):.0f} "
                     f"checked mean={statistics.mean(x[2] for x in d):.0f}")
            cores = None
        if cores:
            extra = f"cores mean={statistics.mean(cores):.0f} p99={q(cores, .99):.0f}"
        row = report(name, prs, moves, extra=extra)
        row["params"] = params
        if cores:
            row["cores_mean"], row["cores_p99"] = statistics.mean(cores), q(cores, .99)
        rows.append(row)
    json.dump({"p_break": args.p_break, "rows": rows}, open(args.out, "w"), indent=1)
