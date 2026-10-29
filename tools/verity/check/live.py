"""A running check's live record (`check-live/v1`), and its reader, `verity ci ls|inspect`.

    uv run python tools/verity/check/live.py ls [--hours H] [--refresh] [--json]
                                            the check runs of the last H hours (default 48), running and ended, newest first:
                                            run, state, commit, machine, started, elapsed
    uv run python tools/verity/check/live.py inspect RUN [--basis N] [--refresh] [--json]
                                            RUN's steps with state, elapsed, an ETA per running step (its median over the last N
                                            recorded checks, default 10) and progress, then the reds so far

The writer.  `check.py` keeps one `Live` per run, beside its `Steps`: every planned step is pending until it starts, running until
its record ends it, then its status.  `Live.write` puts the whole record in the run dir as `check-live.json` (atomically) and, in a
recorded run with a custody key (`check.custody_dir`), publishes it on the store's remote as `live/<run id>/<seq>.json` with that
key: a new immutable object each time (`Remote.put`'s conditional create), so the newest key is the newest record.  It writes
when a step ends and every `EVERY_S` while steps run, and once more when the run ends, with its final state.  A running step
carries its progress (`PROGRESS`): circuit-check's targets done and failed (its `--out`'s `.progress.jsonl`, a line per target
it checked; cache hits come with the step's end), and pytest's and lean-suites' suites passed, failed and reused so far (the
lines `suites.py` prints into the step's log as each suite ends).  A publish that fails is printed once and never stops the check.

The reader reads only recorded data, from the store's remote (this machine's store config or environment): the newest live
record of each run under `live/`, beside the recorded checks with a result this machine's catalog holds, and for the ETA's
basis the ended runs' records and the result manifests (`check/v1` and its
shard and deferred kinds: `meta.steps[].seconds`) of the recorded checks this machine's catalog holds (`--refresh` refreshes its
attempts over the window first: slow on a cold catalog).  A basis takes a step's seconds only where it passed without a cache hit.
What isn't recorded prints as unknown, never as passed: a run with no live record, a step a record doesn't name, and a running
record older than `STALE_S`, which reads as the store's attempt says (ended) or `unknown`.
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import platform
import re
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

SCHEMA = "check-live/v1"
#: the record in the run dir, which `write_result` lists among the run's files
LIVE_FILE = "check-live.json"
#: the store's remote prefix of the records: `live/<run id>/<seq>.json`, `SEQ_DIGITS` digits, so keys sort as they were written
PREFIX = "live/"
SEQ_DIGITS = 6
#: how often the record is written while steps run, besides at each step's end
EVERY_S = 180
#: a running record older than this was last written by a check that may have ended without its final record
STALE_S = 3 * EVERY_S + 120
#: the clock the writer and the reader read (a test injects one)
CLOCK = time.time
#: the ended steps a step's ETA is taken over
BASIS = 10
#: the store attempts of recorded checks (`common.RECORDED_CHECKS`, which this module can't import inside check.py's loader)
RECORDED_CHECKS = "(tool = 'check' OR (tool = 'python3' AND manifest_json LIKE '%tools/verity/check/check.py%'))"
#: the result kinds a check records (`check.py`'s `write_result`)
CHECK_KINDS = ("check/v1", "check-shard/v1", "check-deferred/v1")
ENDED = ("passed", "failed", "stopped", "skipped")


def utc(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def parse_utc(s: str | None) -> float | None:
    try:
        return float(calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")))
    except (TypeError, ValueError):
        return None


#: a line `suites.py` prints as a suite ends (or is reused): its status, padded, then its name
SUITE_LINE = re.compile(r"^(passed|failed|cached|skipped)\s+(\S+)")


def suites_progress(out: Path, step: str) -> dict | None:
    """Suites passed, failed and reused so far, by the step's log, and the failed suites' names."""
    try:
        text = (out / f"{step}.log").read_text(errors="replace")
    except OSError:
        return None
    got: dict[str, list[str]] = {}
    for line in text.splitlines():
        if m := SUITE_LINE.match(line):
            got.setdefault(m[1], []).append(m[2])
    return {"suites_passed": len(got.get("passed", [])), "suites_failed": len(got.get("failed", [])),
            "suites_cached": len(got.get("cached", [])), "failed": got.get("failed", [])[:10]}


def circuit_progress(out: Path, step: str) -> dict | None:
    """Targets checked so far and how many failed, by circuit-check's progress lines, and the first failed targets' names."""
    try:
        lines = (out / "circuit-check.json.progress.jsonl").read_text(errors="replace").splitlines()
    except OSError:
        return None
    done, failed = 0, []
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        done += 1
        if r.get("failures") or r.get("killed"):
            failed.append(str(r.get("target")))
    return {"targets_done": done, "targets_failed": len(failed), "failed": failed[:10]}


PROGRESS: dict[str, Callable[[Path, str], dict | None]] = {
    "pytest": suites_progress, "lean-suites": suites_progress, "circuit-check": circuit_progress}


def custody_put(custody: Path, timeout: float = 30.0) -> Callable[[str, bytes], None]:
    """A `put(key, bytes)` onto the store's remote with the run's custody key (its staged `store.toml` and `cred.json`), never through
    this process's environment, which the steps inherit."""
    import tomllib

    from research.store.remote_s3 import S3Remote

    cfg = tomllib.loads((custody / "store.toml").read_text())["remote"]
    cred = json.loads((custody / "cred.json").read_text())
    remote = S3Remote(cfg["endpoint"], cfg["bucket"], cred["AWS_ACCESS_KEY_ID"], cred["AWS_SECRET_ACCESS_KEY"],
                      region=cfg.get("region", "auto"), timeout=timeout, session_token=cred.get("AWS_SESSION_TOKEN"))

    def put(key: str, data: bytes) -> None:
        with tempfile.TemporaryDirectory(prefix="check-live-") as d:
            (p := Path(d) / "live.json").write_bytes(data)
            remote.put(key, p)

    return put


def machine_of(out: Path) -> str:
    """The machine `research run` placed the run on (its `job.json`), else this host's name."""
    try:
        return json.loads((out / "job.json").read_text())["run"]["remote"]["machine"] or platform.node()
    except (OSError, ValueError, KeyError, TypeError):
        return platform.node()


class Live:
    """One run's live record (the module docstring): `plan`, `started` and `ended` as its steps go, `write` to put it in the run dir and
    publish it with ``put``, and `start` for a thread that writes it every `EVERY_S` and whenever a step ends (`request`)."""

    def __init__(self, out: Path, run_id: str | None = None, put: Callable[[str, bytes], None] | None = None,
                 commit: str | None = None, what: str | None = None, machine: str | None = None):
        self.out, self.put = out, put
        self.run_id = run_id or os.environ.get("RESEARCH_RUN_ID") or "local"
        self.head = {"commit": commit, "what": what or "check", "machine": machine or machine_of(out)}
        self.t0 = CLOCK()
        self.steps: dict[str, dict] = {}
        self.began: dict[str, float] = {}
        self.state = "running"
        self.seq = 0
        self.failures = 0
        self.lock = threading.Lock()
        self.publishing = threading.Lock()
        self.want = threading.Event()
        self.closed = False
        self.thread: threading.Thread | None = None
        (out / LIVE_FILE).unlink(missing_ok=True)

    def plan(self, names: list[str]) -> None:
        with self.lock:
            for n in names:
                self.steps.setdefault(n, {"step": n, "state": "pending"})

    def started(self, name: str) -> None:
        with self.lock:
            self.began[name] = t = CLOCK()
            self.steps[name] = {"step": name, "state": "running", "started_utc": utc(t)}

    def ended(self, rec: dict) -> None:
        name = rec["step"]
        with self.lock:
            s = {"step": name, "state": rec.get("status") or "failed", "seconds": rec.get("seconds")}
            t = self.began.get(name)
            if t is None and isinstance(rec.get("seconds"), (int, float)):
                t = CLOCK() - rec["seconds"]
            if t is not None:
                s["started_utc"] = utc(t)
            s |= {k: rec[k] for k in ("cache", "detail") if rec.get(k)}
            if name in PROGRESS and (p := PROGRESS[name](self.out, name)) is not None:
                s["progress"] = p
            self.steps[name] = s
        self.request()

    def doc(self) -> dict:
        """The record now: each running step's seconds so far and progress read at this moment."""
        now = CLOCK()
        steps = []
        for name, s in self.steps.items():
            s = dict(s)
            if s["state"] == "running":
                s["seconds"] = round(now - self.began[name], 1)
                if name in PROGRESS and (p := PROGRESS[name](self.out, name)) is not None:
                    s["progress"] = p
            steps.append(s)
        slot = {}
        try:
            got = json.loads((self.out / "slot.json").read_text())
            slot = {k: got[k] for k in ("slot", "since", "took") if got.get(k)}
        except (OSError, ValueError, AttributeError):
            pass
        return {"schema": SCHEMA, "run_id": self.run_id, **self.head, **({"slot": slot} if slot else {}), "state": self.state,
                "started_utc": utc(self.t0), "at_utc": utc(now), "seconds": round(now - self.t0, 1), "seq": self.seq, "steps": steps}

    def write(self) -> dict:
        """The record into the run dir, then onto the remote when there is a ``put`` (outside the lock a step's start and end
        take, so a slow remote never holds a step up)."""
        with self.publishing:
            with self.lock:
                doc = self.doc()
                data = (json.dumps(doc, indent=1) + "\n").encode()
                tmp = self.out / (LIVE_FILE + ".tmp")
                tmp.write_bytes(data)
                os.replace(tmp, self.out / LIVE_FILE)
                self.seq += 1
            if self.put is not None:
                try:
                    self.put(f"{PREFIX}{self.run_id}/{doc['seq']:0{SEQ_DIGITS}d}.json", data)
                except Exception as e:  # noqa: BLE001  (a record that isn't published leaves the check as it was)
                    self.failures += 1
                    if self.failures == 1:
                        print(f"check: live: publishing {LIVE_FILE} failed ({type(e).__name__}: {e}); the run dir keeps it",
                              flush=True)
            return doc

    def request(self) -> None:
        self.want.set()

    def start(self) -> None:
        self.thread = threading.Thread(target=self._loop, daemon=True, name="check-live")
        self.thread.start()

    def _loop(self) -> None:
        while not self.closed:
            self.want.wait(EVERY_S)
            self.want.clear()
            if not self.closed:
                self.write()

    def close(self, state: str) -> dict:
        """The run's final record: ``state`` (passed, failed or stopped), written once more."""
        self.closed = True
        self.want.set()
        if self.thread is not None:
            self.thread.join()
        self.state = state
        return self.write()


# ---- the reader -------------------------------------------------------------------------------------------------------------------------

def run_of(key: str) -> str:
    return key[len(PREFIX):].split("/", 1)[0]


def newest_keys(keys: list[str]) -> dict[str, str]:
    """{run id: its newest record's key} of ``keys`` under `PREFIX`."""
    out: dict[str, str] = {}
    for k in keys:
        if k.startswith(PREFIX) and k.endswith(".json") and k.count("/") == 2:
            out[run_of(k)] = max(out.get(run_of(k), k), k)
    return out


def read_json(remote, key: str) -> dict | None:
    with tempfile.TemporaryDirectory(prefix="ci-live-") as d:
        try:
            remote.get(key, Path(d) / "doc.json")
            return json.loads((Path(d) / "doc.json").read_text())
        except (KeyError, OSError, ValueError):
            return None


def live_docs(remote, since: str) -> dict[str, dict]:
    """{run id: its newest live record} of the runs whose ids sort after ``since`` (a run id or its prefix)."""
    keys = [k for k, _ in remote.list_times(PREFIX, start_after=f"{PREFIX}{since}")]
    newest = newest_keys(keys)
    with ThreadPoolExecutor(max_workers=16) as ex:
        docs = dict(zip(newest, ex.map(lambda k: read_json(remote, k), newest.values())))
    return {r: d for r, d in docs.items() if isinstance(d, dict) and d.get("schema") == SCHEMA}


def live_doc(remote, rid: str) -> dict | None:
    newest = newest_keys(remote.list(f"{PREFIX}{rid}/"))
    return read_json(remote, newest[rid]) if rid in newest else None


def since_prefix(hours: float, now: float) -> str:
    return "r" + time.strftime("%Y%m%d-%H%M%S", time.gmtime(now - hours * 3600))


def store_checks(store, since: str, limit: int = 200) -> list[dict]:
    """The recorded checks this machine's catalog holds that started after ``since``, newest first: {run_id, state, commit,
    machine, started_utc, seconds, result}."""
    rows = store.index.sql(f"SELECT id, state, source_commit, host, start_utc, wall_s, manifest_json FROM attempts WHERE {RECORDED_CHECKS} "
                           "AND id > ? ORDER BY id DESC LIMIT ?", (since, limit))
    out = []
    for rid, state, commit, host, start, wall, mj in rows:
        try:
            doc = json.loads(mj)
            result, host = (doc.get("outputs") or {}).get("result"), (doc.get("execution") or {}).get("machine") or host
        except (TypeError, ValueError, AttributeError):
            result = None
        out.append({"run_id": rid, "state": state, "commit": commit, "machine": host, "started_utc": start, "seconds": wall,
                    "result": result})
    return out


def store_steps(store, art: str) -> list[dict] | None:
    """A check's step records, from its result manifest's meta, or None when the manifest isn't a check's."""
    try:
        m = store.get_manifest(art)
    except Exception:  # noqa: BLE001  (unreadable: not in the basis)
        return None
    steps = (m.meta or {}).get("steps") if m.kind in CHECK_KINDS else None
    return steps if isinstance(steps, list) else None


def basis(records: list[list[dict]], n: int = BASIS) -> dict[str, list[float]]:
    """{step: the seconds of its last ``n`` passes without a cache hit} over ``records`` (each a run's steps, newest run first),
    which are live records' steps (`state`) or result manifests' (`status`)."""
    out: dict[str, list[float]] = {}
    for steps in records:
        for s in steps:
            hit = isinstance(s.get("cache"), dict) and "hit" in s["cache"]
            if (s.get("state") or s.get("status")) == "passed" and not hit and isinstance(s.get("seconds"), (int, float)) \
                    and len(out.setdefault(s["step"], [])) < n:
                out[s["step"]].append(float(s["seconds"]))
    return out


def dur(s: float | None) -> str:
    if s is None:
        return "?"
    s = int(round(s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def eta(step: dict, elapsed: float | None, seconds: dict[str, list[float]]) -> str:
    got = seconds.get(step["step"]) or []
    if step.get("state") != "running":
        return ""
    if elapsed is None:
        return "unknown (its record is stale)"
    if not got:
        return "unknown (no recorded basis)"
    med = statistics.median(got)
    left = med - elapsed
    of = f"median {dur(med)} of {len(got)}"
    return f"~{dur(left)} ({of})" if left >= 0 else f"over by {dur(-left)} ({of})"


def progress_text(p: dict | None) -> str:
    if not p:
        return ""
    if "targets_done" in p:
        return f"targets {p['targets_done']} done, {p['targets_failed']} failed"
    return f"suites {p['suites_passed']} passed, {p['suites_failed']} failed, {p['suites_cached']} reused"


def judged(doc: dict, now: float, ended_state: str | None = None) -> str:
    """The run's state as the record shows it at ``now``: a running record older than `STALE_S` is ``ended_state`` (the store's
    attempt, when it has ended) or unknown."""
    state = doc.get("state") or "unknown"
    at = parse_utc(doc.get("at_utc"))
    if state == "running" and (at is None or now - at > STALE_S):
        return f"{ended_state} (no final live record)" if ended_state else f"unknown (last record {dur(now - at) if at else '?'} ago)"
    return state


def reds(doc: dict) -> list[str]:
    out = []
    for s in doc.get("steps") or []:
        p, failed = s.get("progress") or {}, (s.get("progress") or {}).get("failed") or []
        if s.get("state") == "failed":
            out.append(f"{s['step']} failed" + (f": {s['detail']}" if s.get("detail") else ""))
        if failed:
            n = p.get("suites_failed", p.get("targets_failed", len(failed)))
            what = "suite" if "suites_failed" in p else "target"
            out.append(f"{s['step']}: {n} {what}{'s' if n != 1 else ''} failed: {', '.join(failed[:5])}"
                       + (" ..." if n > 5 else ""))
    return out


def render_inspect(rid: str, doc: dict | None, seconds: dict[str, list[float]], now: float, ended_state: str | None = None) -> str:
    """`inspect`'s text: the run's head, one line per step (state, elapsed, ETA, progress), then the reds so far."""
    if doc is None:
        return (f"check {rid}: unknown: no live record on the store's remote (a check from before live records, one still waiting "
                f"for its slot, or one that never started)" + (f"; the store's attempt says {ended_state}" if ended_state else ""))
    state = judged(doc, now, ended_state)
    at = parse_utc(doc.get("at_utc"))
    stale = state != doc.get("state")
    started = parse_utc(doc.get("started_utc"))
    lines = [f"check {rid}  {state}  {doc.get('what') or 'check'} of {(doc.get('commit') or '?')[:12]} on {doc.get('machine') or '?'}",
             f"  started {doc.get('started_utc') or '?'} ({dur(now - started) if started else '?'} ago)"
             + (f", slot {doc['slot'].get('slot')} taken {doc['slot'].get('took')} after waiting since {doc['slot'].get('since')}"
                if doc.get("slot") else "")
             + f"; last record {doc.get('at_utc') or '?'} ({dur(now - at) if at else '?'} ago, #{doc.get('seq')})"]
    rows = [("step", "state", "elapsed", "eta", "progress")]
    for s in doc.get("steps") or []:
        st = s.get("state") or "unknown"
        elapsed = s.get("seconds")
        if st == "running":
            begun = parse_utc(s.get("started_utc"))
            elapsed = None if stale or begun is None else now - begun
            st = "running?" if stale else st
        cache = s.get("cache") or {}
        if "hit" in cache:
            st += " (cached)"
        elif "hits" in cache:
            st += f" ({cache['hits']} reused)"
        rows.append((s["step"], "not run" if st == "pending" and doc.get("state") != "running" else st,
                     dur(elapsed) if elapsed is not None else "", eta(s, None if stale else elapsed, seconds),
                     progress_text(s.get("progress"))))
    w = [max(len(r[i]) for r in rows) for i in range(4)]
    lines += ["  " + "  ".join(c.ljust(w[i]) if i < 4 else c for i, c in enumerate(r)).rstrip() for r in rows]
    red = reds(doc)
    lines.append("reds so far:" + ("" if red else " none recorded"))
    lines += [f"  {r}" for r in red]
    return "\n".join(lines)


def render_ls(rows: list[dict], now: float) -> str:
    out = [("run", "state", "commit", "machine", "started", "elapsed", "what")]
    for r in rows:
        started = parse_utc(r.get("started_utc"))
        out.append((r["run_id"], r["state"], (r.get("commit") or "?")[:12], r.get("machine") or "?", r.get("started_utc") or "?",
                    dur(r.get("seconds")) if r.get("seconds") is not None else dur(now - started) if started else "?",
                    r.get("what") or ""))
    w = [max(len(r[i]) for r in out) for i in range(len(out[0]))]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip() for r in out)


def ls_rows(docs: dict[str, dict], checks: list[dict], now: float, attempt_state: Callable[[str], str | None]) -> list[dict]:
    """One row per run, newest first: its live record's, judged at ``now`` (a stale running one by ``attempt_state``), else the
    catalog's attempt's."""
    rows = {}
    for c in checks:
        rows[c["run_id"]] = {**c, "what": "", "state": {"done": "passed"}.get(c["state"], c["state"])}
    for rid, d in docs.items():
        stale = d.get("state") == "running" and judged(d, now) != "running"
        state = judged(d, now, attempt_state(rid) if stale else None)
        end = parse_utc(d.get("at_utc")) if d.get("state") != "running" else None
        start = parse_utc(d.get("started_utc"))
        rows[rid] = {"run_id": rid, "state": state, "commit": d.get("commit"), "machine": d.get("machine"),
                     "started_utc": d.get("started_utc"), "seconds": end - start if end and start else None,
                     "what": d.get("what") or ""}
    return [rows[k] for k in sorted(rows, reverse=True)]


def _store(refresh_since: str | None):
    from research.store.local import LocalStore

    store = LocalStore()
    if store.remote is None:
        raise SystemExit(f"verity ci: no store remote here ({getattr(store, 'remote_error', None) or 'no store config or R2_* environment'})")
    if refresh_since:
        store.refresh(attempts_only=True, since=refresh_since)
    return store


def _attempt_state(store, rid: str) -> str | None:
    try:
        a = store.get_attempt(rid) or {}
    except Exception:  # noqa: BLE001
        return None
    st = a.get("state") or (a.get("status") or {}).get("state")
    return {"done": "passed"}.get(st, st) if st in ("done", "failed", "cancelled") else None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="verity ci", description="Recorded check runs, read from the store's remote: `ls` and `inspect RUN`.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("ls", help="recent check runs, running and ended")
    pl.add_argument("--hours", type=float, default=48.0)
    pi = sub.add_parser("inspect", help="a run's steps, their ETAs and its reds so far")
    pi.add_argument("run")
    pi.add_argument("--basis", type=int, default=BASIS, help="the ended checks a step's median is taken over")
    pi.add_argument("--hours", type=float, default=96.0, help="how far back the basis looks")
    for p in (pl, pi):
        p.add_argument("--refresh", action="store_true", help="refresh this machine's catalog of attempts over the window first")
        p.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    now = CLOCK()
    since = since_prefix(a.hours, now)
    store = _store(since if a.refresh else None)
    remote = store.remote
    if a.cmd == "ls":
        docs = live_docs(remote, since)
        ran = [c for c in store_checks(store, since, 2000) if c["result"]]      # a twin that never took a slot has none
        rows = ls_rows(docs, ran, now, lambda rid: _attempt_state(store, rid))
        print(json.dumps(rows, indent=1) if a.json else render_ls(rows, now))
        return 0
    doc = live_doc(remote, a.run)
    ended = _attempt_state(store, a.run) if doc is None or judged(doc, now) != doc.get("state") else None
    runs = {r: d["steps"] for r, d in live_docs(remote, since).items() if r != a.run and d.get("state") in ENDED and d.get("steps")}
    prior = [c for c in store_checks(store, since, 2000) if c["result"] and c["run_id"] not in runs and c["run_id"] != a.run]
    for c in prior[:3 * a.basis]:
        if (steps := store_steps(store, c["result"])) is not None:
            runs[c["run_id"]] = steps
    seconds = basis([runs[r] for r in sorted(runs, reverse=True)], a.basis)
    if a.json:
        print(json.dumps({"run_id": a.run, "state": judged(doc, now, ended) if doc else "unknown", "live": doc,
                          "basis": seconds, "reds": reds(doc) if doc else []}, indent=1))
    else:
        print(render_inspect(a.run, doc, seconds, now, ended))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
