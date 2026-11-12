"""A measure for one run, on a scratch branch only: read a fetched run of job.sh.

    read.py RUN_DIR [NAME ...]      NAME: a directory of RUN_DIR/out with rows/ (default: each pass-*)

For each such suites.py run of the suite: its result (suites.json), and one line per xdist worker, from two sources that are
kept apart: the plugin's rows (the kernel's own figures for that process: VmHWM, and the largest resident set of any process
it started and waited for) and the run's resources.jsonl (every process of the run, sampled each second; PSS every few
seconds), which gives what a worker and everything it started held at once.  GiB throughout (2**30 bytes).
"""
import collections
import datetime
import glob
import json
import os
import sys

G = 2**30
KB = 1024


def epoch(t: str) -> float:
    return datetime.datetime.strptime(t[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp() + float("0" + t[19:-1] or 0)


def hms(t: float) -> str:
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%H:%M:%S")


def load(run: str):
    """Each process (pid, start ticks) -> its samples [(t, rss, hwm, cpu_s)], its first parent pid, and the PSS samples."""
    procs: dict = {}
    pss: list = []                                    # each PSS round: (t, {key: (pss or None when not read, status RSS)})
    cgroup = []
    seen_at: dict = collections.defaultdict(list)     # pid -> [(first t, key)]
    last_key: dict = {}
    with open(os.path.join(run, "resources.jsonl")) as f:
        for line in f:
            if '"kind": "process"' in line or '"kind":"process"' in line:
                r = json.loads(line)
                t = epoch(r["t_utc"])
                for p in r.get("processes", ()):
                    key = (p["pid"], p.get("starttime_ticks"))
                    if key not in procs:
                        procs[key] = {"ppid": p.get("ppid"), "comm": p.get("comm"), "s": []}
                        seen_at[p["pid"]].append((t, key))
                    last_key[p["pid"]] = key
                    procs[key]["s"].append((t, p.get("rss_bytes", 0), p.get("rss_hwm_bytes", 0), p.get("cpu_s", 0.0)))
            elif '"kind": "pss"' in line or '"kind":"pss"' in line:
                r = json.loads(line)
                got = {}
                for p in r.get("processes", ()):
                    # not read: a process over the sampler's --pss-max-rss-gb, or read too recently (its status RSS is there); or gone
                    if (key := last_key.get(p["pid"])) is not None and (p.get("pss_bytes") is not None or p.get("skipped")):
                        got[key] = (p.get("pss_bytes"), p.get("rss_bytes") or p.get("rss_bytes_stat") or 0)
                pss.append((epoch(r["t_utc"]), got))
            elif '"kind": "cgroup"' in line or '"kind":"cgroup"' in line:
                r = json.loads(line)
                cgroup.append((epoch(r["t_utc"]), r["cgroup"].get("memory_current_bytes", 0), r["cgroup"].get("memory_peak_bytes", 0)))
    parent = {}
    for key, p in procs.items():
        first = p["s"][0][0]
        cands = [k for t, k in seen_at.get(p["ppid"], ()) if t <= first + 1.5]
        parent[key] = cands[-1] if cands else None
    kids = collections.defaultdict(list)
    for key, par in parent.items():
        if par is not None:
            kids[par].append(key)
    return procs, pss, cgroup, kids, seen_at


def subtree(kids, key) -> list:
    out, stack = [], list(kids.get(key, ()))
    while stack:
        k = stack.pop()
        out.append(k)
        stack.extend(kids.get(k, ()))
    return out


def at_once(procs, keys, t0, t1, col=1):
    """The most ``keys`` held together in [t0, t1] (the sum of a column of their samples, by whole second), and when."""
    held, n = collections.Counter(), collections.Counter()
    for k in keys:
        for s in procs[k]["s"]:
            if t0 <= s[0] <= t1:
                held[int(s[0])] += s[col]
                n[int(s[0])] += 1
    if not held:
        return 0, None, 0
    when = max(held, key=held.get)
    return held[when], when, max(n.values())


def pss_at_once(pss, keys, t0, t1):
    """The most ``keys`` held together in [t0, t1] by the PSS rounds (a page two processes share counted once), a few seconds
    apart: (with each process the round did not read at its RSS, which counts its shared pages whole: no less than the truth;
    when; with those left out: no more than the truth; the rounds; the rounds with one not read)."""
    keys = set(keys)
    hi, lo, when, rounds, unread = 0, 0, None, 0, 0
    for t, got in pss:
        if not t0 <= t <= t1:
            continue
        mine = [got[k] for k in keys & got.keys()]
        if not mine:
            continue
        rounds += 1
        unread += any(v is None for v, _ in mine)
        a, b = sum(v if v is not None else rss for v, rss in mine), sum(v for v, _ in mine if v is not None)
        if a > hi:
            hi, when = a, t
        lo = max(lo, b)
    return hi, when, lo, rounds, unread


def rows_of(d: str) -> dict:
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "rows", "*.jsonl"))):
        rs = [json.loads(ln) for ln in open(f) if ln.strip()]
        if not rs:
            continue
        pid = next((r["pid"] for r in rs if r["event"] == "start"), rs[0]["pid"])
        out[os.path.basename(f)[:-6]] = [r for r in rs if r["pid"] == pid]
    return out


def gib(kb) -> float:
    return round((kb or 0) * KB / G, 2)


def one(run: str, name: str, res) -> dict:
    procs, pss, cgroup, kids, seen_at = res
    d = os.path.join(run, "out", name)
    rows = rows_of(d)
    ctrl = rows.get("controller", [])
    t0 = next((r["t"] for r in ctrl if r["event"] == "start"), None)
    t1 = next((r["t"] for r in ctrl if r["event"] == "end"), None)
    out: dict = {"name": name, "start": t0 and hms(t0), "end": t1 and hms(t1), "wall_s": t0 and t1 and round(t1 - t0), "workers": []}
    try:
        sj = json.load(open(os.path.join(d, "suites.json")))[0]
        out["suites"] = {k: sj.get(k) for k in ("status", "seconds", "workers", "counts", "outside", "child_outside", "unguarded", "timed_out")}
    except (OSError, ValueError, IndexError):
        out["suites"] = None
    if ctrl:
        out["controller_kids_maxrss"] = gib(ctrl[-1].get("kids_maxrss"))
    build = None
    for w in sorted((k for k in rows if k != "controller"), key=lambda s: int(s[2:])):
        rs = rows[w]
        ev = collections.defaultdict(list)
        for r in rs:
            ev[r["event"]].append(r)
        start, cat, end = (ev[k][0] if ev[k] else None for k in ("start", "catalog", "end"))
        tests = ev["test"]
        last = end or (tests[-1] if tests else cat or start)
        pid = start["pid"] if start else rs[0]["pid"]
        key = next((k for t, k in reversed(seen_at.get(pid, ())) if t0 is None or t <= (t1 or t) + 5), None)
        line: dict = {"worker": w, "pid": pid, "how": cat and cat.get("how"), "calls_from_records": cat and cat.get("calls") is not None,
                      "tests": len(tests), "test_s": round(sum(t["s"] for t in tests)),
                      "own_hwm": gib(last.get("VmHWM")), "own_hwm_at_catalog": cat and gib(cat.get("VmHWM")),
                      "own_rss_at_catalog": cat and gib(cat.get("VmRSS")), "own_rss_at_end": gib(last.get("VmRSS")),
                      "kids_maxrss": gib(last.get("kids_maxrss")), "kids_maxrss_at_catalog": cat and gib(cat.get("kids_maxrss")),
                      "slot_env_in_tests": sorted({str(t.get("slot_env")) for t in tests}), "ended": end is not None}
        # the test that set each high-water mark: the first row at which it reached its last value
        if cat and last.get("VmHWM", 0) <= cat.get("VmHWM", 0):
            line["own_hwm_set_by"] = "making the catalog"
        else:
            line["own_hwm_set_by"] = next((t["test"] for t in tests if t.get("VmHWM") == last.get("VmHWM")), None)
        if cat and last.get("kids_maxrss", 0) <= cat.get("kids_maxrss", 0):
            line["kids_maxrss_set_by"] = "making the catalog" if cat.get("kids_maxrss", 0) > (start or {}).get("kids_maxrss", 0) else "before the session"
        else:
            line["kids_maxrss_set_by"] = next((t["test"] for t in tests if t.get("kids_maxrss") == last.get("kids_maxrss")), None)
        # the tests that raised this process's high-water mark the most, and those whose started processes were the largest
        rises = sorted(((t["VmHWM"] - t["before"]["VmHWM"], t["test"], t["VmHWM"]) for t in tests if t["VmHWM"] > t["before"]["VmHWM"]), reverse=True)
        line["own_rises"] = [{"test": n, "by": gib(by), "to": gib(to)} for by, n, to in rises[:4]]
        krises = sorted(((t["kids_maxrss"], t["test"]) for t in tests if t["kids_maxrss"] > t["before"]["kids_maxrss"]), reverse=True)
        line["kids_rises"] = [{"test": n, "to": gib(to)} for to, n in krises[:4]]
        ba = [r for r in ev["built_apart"] if r.get("test") is None]
        if ba and cat:
            build = {"worker": w, "pid": pid, "key": key, "t0": ba[0]["t"], "t1": cat["t"], "jobs": ba[0].get("jobs"), "budget": ba[0].get("budget"),
                     "entries": ba[0].get("entries"), "counted_sum": ba[0].get("counted_sum"), "counted_max": ba[0].get("counted_max"),
                     "builder_rss_at_start": gib(ba[0].get("VmRSS")), "slot_env": ba[0].get("slot_env"), "workers_env": ba[0].get("workers"),
                     "error": ba[0].get("error")}
        line["built_apart_in_tests"] = [{"test": r.get("test"), "jobs": r.get("jobs"), "budget": r.get("budget"), "entries": r.get("entries")}
                                        for r in ev["built_apart"] if r.get("test") is not None]
        if key is not None:
            sub = subtree(kids, key)
            line["sampled_own_hwm"] = round(max(s[2] for s in procs[key]["s"]) / G, 2)
            line["started"] = len(sub)
            line["sampled_largest_started"] = round(max((max(s[2] for s in procs[k]["s"]) for k in sub), default=0) / G, 2)
            ta = cat["t"] if cat else (t0 or 0)
            tb = (end or last)["t"]
            v, when, lo, rounds, unread = pss_at_once(pss, [key, *sub], ta, tb)
            line["tests_pss_with_started"] = round(v / G, 2)
            line["tests_pss_with_started_lo"] = round(lo / G, 2)
            line["tests_pss_with_started_at"] = when and hms(when)
            line["pss_rounds"], line["pss_rounds_unread"] = rounds, unread
            line["tests_pss_during"] = when and next((t["test"] for t in tests if t["t0"] - 1 <= when <= t["t0"] + t["s"] + 2), None)
            v, when, n = at_once(procs, sub, ta, tb)
            line["tests_started_rss_sum"] = round(v / G, 2)
            line["tests_started_at_once"] = n
            line["tests_started_pss"] = round(pss_at_once(pss, sub, ta, tb)[0] / G, 2)
        out["workers"].append(line)
    if build is not None and build["key"] is not None:
        sub = subtree(kids, build["key"])
        direct = kids.get(build["key"], [])
        b0, b1 = build["t0"], build["t1"]
        alive = [k for k in direct if procs[k]["s"][0][0] <= b1 and procs[k]["s"][-1][0] >= b0]
        v, when, n = at_once(procs, alive, b0, b1)
        build.update(seconds=round(b1 - b0), processes=len(alive), at_once=n, rss_sum=round(v / G, 2), rss_sum_at=when and hms(when),
                     largest_hwm=round(max((max(s[2] for s in procs[k]["s"]) for k in alive), default=0) / G, 2))
        v, when, lo, rounds, unread = pss_at_once(pss, [k for k in sub if procs[k]["s"][0][0] <= b1], b0, b1)
        build.update(pss_sum=round(v / G, 2), pss_sum_lo=round(lo / G, 2), pss_sum_at=when and hms(when), pss_rounds=rounds, pss_rounds_unread=unread)
        # how far each build process grew past the builder's resident set as it forked them: what `heavy.json` counts
        base = build["builder_rss_at_start"] * G
        grown = collections.Counter()
        for k in alive:
            for s in procs[k]["s"]:
                if b0 <= s[0] <= b1:
                    grown[int(s[0])] += max(0, s[1] - base)
        if grown:
            when = max(grown, key=grown.get)
            build.update(grown_sum=round(grown[when] / G, 2), grown_sum_at=hms(when))
        waiting = {}
        for line in out["workers"]:
            k = next((k for t, k in reversed(seen_at.get(line["pid"], ()))), None)
            if k is not None and line["worker"] != build["worker"]:
                waiting[line["worker"]] = round(max((s[1] for s in procs[k]["s"] if b0 <= s[0] <= b1), default=0) / G, 2)
        build["waiting_workers_rss"] = waiting
        del build["key"]
    elif build is not None:
        del build["key"]
    out["build"] = build
    # the whole session at once (the controller and all under it), against what suites.py gave it: its workers' shares
    cpid = ctrl[0]["pid"] if ctrl else None
    ckey = next((k for t, k in reversed(seen_at.get(cpid, ()))), None)
    if ckey is not None and t0 and t1:
        everything = [ckey, *subtree(kids, ckey)]
        spans = {"session": (t0, t1)}
        if build:
            spans.update(build=(build["t0"], build["t1"]), after_build=(build["t1"], t1))
        out["session_pss"] = {}
        for name_, (a, b) in spans.items():
            hi, when, lo, rounds, unread = pss_at_once(pss, everything, a, b)
            out["session_pss"][name_] = {"at_most": round(hi / G, 2), "at": when and hms(when), "at_least": round(lo / G, 2), "rounds": rounds,
                                         "rounds_with_one_unread": unread}
    if t0 and t1:
        cg = [c for c in cgroup if t0 <= c[0] <= t1]
        if cg:
            top = max(cg, key=lambda c: c[1])
            out["cgroup_current_max"] = round(top[1] / G, 2)
            out["cgroup_current_max_at"] = hms(top[0])
    return out


def table(o: dict) -> str:
    head = ("| worker | catalog | own peak (VmHWM) | of which by the catalog | own peak set by | largest process it started (waited-for maxrss) "
            "| set by | it and all it started at once, PSS | tests | test s |\n|---|---|---|---|---|---|---|---|---|---|\n")
    body = ""
    for w in o["workers"]:
        short = lambda s: (s or "?").split("::")[-1][:70]
        body += (f"| {w['worker']} | {w['how']} | {w['own_hwm']} | {w['own_hwm_at_catalog']} | {short(w['own_hwm_set_by'])} | {w['kids_maxrss']} "
                 f"| {short(w['kids_maxrss_set_by'])} | {w.get('tests_pss_with_started', '?')} | {w['tests']} | {w['test_s']} |\n")
    return head + body


def main() -> None:
    run = sys.argv[1].rstrip("/")
    names = sys.argv[2:] or sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(run, "out", "pass-*", "rows")))
    res = load(run)
    got = [one(run, n, res) for n in names]
    json.dump(got, open(os.path.join("/tmp/ccmem", f"peak-{os.path.basename(run)}.json"), "w"), indent=1)
    for o in got:
        print(f"== {o['name']}: {o['start']}-{o['end']} wall {o['wall_s']} s; suites.json {o['suites']}; cgroup at most {o.get('cgroup_current_max')} GiB "
              f"at {o.get('cgroup_current_max_at')}; the controller's largest waited-for process {o.get('controller_kids_maxrss')} GiB")
        if o["build"]:
            print("   build apart:", json.dumps(o["build"]))
        print("   the session's processes at once (PSS; a process not read at its RSS):", json.dumps(o.get("session_pss")))
        print(table(o))
        for w in o["workers"]:
            print(f"   {w['worker']} pid {w['pid']}: sampled own hwm {w.get('sampled_own_hwm')}, started {w.get('started')} (sampled), the largest sampled "
                  f"{w.get('sampled_largest_started')}; in tests: it and what it started PSS {w.get('tests_pss_with_started')} at "
                  f"{w.get('tests_pss_with_started_at')} during {w.get('tests_pss_during')}; what it started alone: PSS {w.get('tests_started_pss')}, "
                  f"RSS sum {w.get('tests_started_rss_sum')}, at once {w.get('tests_started_at_once')}; slot in tests {w['slot_env_in_tests']}; "
                  f"rss at catalog {w['own_rss_at_catalog']}, at end {w['own_rss_at_end']}; ended {w['ended']}; PSS rounds {w.get('pss_rounds')}, "
                  f"{w.get('pss_rounds_unread')} with a process not read (without those: {w.get('tests_pss_with_started_lo')})")
            for r in w["own_rises"]:
                print(f"      own peak rose by {r['by']} to {r['to']}: {r['test']}")
            for r in w["kids_rises"]:
                print(f"      a started process reached {r['to']}: {r['test']}")
            for r in w["built_apart_in_tests"]:
                print(f"      built_apart in a test: {r}")


if __name__ == "__main__":
    main()
