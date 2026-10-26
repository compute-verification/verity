"""The cloud watch (INF-73): our cloud infra's health and spend, watched from the control pod so nothing fails or overspends
unnoticed. Standard library only (python3.12 on vy-control-verity; psql for Neon).

    python3.12 cloud_watch.py loop [--config cloud_watch.toml]              every `every_s`: check, post anomalies, digest daily
    python3.12 cloud_watch.py once [--dry-run] [--digest] [--pages N]       one pass; --dry-run prints what it would post and
                                                                            writes no state unless --state names a file;
                                                                            --digest composes the digest now

Every pass runs each check, and each check returns findings (a stable key while the anomaly lasts, and its text), digest lines
and gaps. A finding opens once it has been found on `after` passes in a row (one, or `unreadable_passes` for a check that
raised) and posts once in #agent-alerts naming @infra; it clears once it has been gone `clear_passes` passes in a row, and the
clear is a reply in the open post's thread. A post Slack refuses is tried again next pass. At `digest_hour_utc` each day the
digest posts every check's lines: loop liveness, R2's size by top-level prefix, Neon, spend month to date and projected, the
GitHub rate limit and the public endpoints. A check whose credential isn't in the environment is skipped, and the digest says
once per check which key would close the gap (under *Unmonitored*: `needs KEY: what it would watch`). State: `state` in the config.

The checks (cloud_watch.toml holds every threshold, price and path):

- **loops**: each loop post_start.sh starts leaves a success marker (a log line with its time, a JSON time field, or a file's
  mtime). One with no success in twice its period is an anomaly.
- **r2**: an incremental ListObjectsV2 sweep of each bucket, at most `page_cap` pages (1000 keys, one class-A op) a pass, resumed
  from the last key, so a complete sweep's totals per top-level prefix come every few passes at a bounded cost. Growth between
  the last two sweeps above `max_gib_per_day`, or `mean_ratio` times the 7-day mean, is an anomaly naming the prefix that grew most;
  so is a bucket with `stale_after_h` whose newest object is older than that (the Lean cache's daily publisher).
- **neon**: database size, connections against max_connections, the store index's last write (max `listed_at`), and how many
  tables each of `report_schemas` has (`store_public` is built only once the export's credential is on the pod).
- **spend**: the budgets guard's state (RunPod balance, rate and each line's spend) against budgets.toml's live lines, the
  RunPod balance under its auto top-up (a failed top-up), R2's month at Cloudflare's list price, and, when Nebius credentials and an
  instance list are available, each running VM's line: one on no line, on an expired line, or burning past its line is an anomaly.
- **github**: the token's core rate limit; **jobs**: Control's Job queue, where a job queued past `stale_queued_h` (from its
  `not_before`) means no worker claims its kind; **http**: the public endpoints (Vercel's site, the Slack broker) answer below 500.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

CONFIG = Path(__file__).with_name("cloud_watch.toml")
GIB = 1 << 30
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def epoch(s: str) -> float:
    """`2026-10-18T19:33:23Z`, `2026-10-18 19:33:23+00` or budgets.toml's `2026-10-24T21:00Z`, in epoch seconds."""
    s = s.strip().replace(" ", "T").rstrip("Z")[:19]
    return float(calendar.timegm(time.strptime(s if len(s) > 16 else s + ":00", "%Y-%m-%dT%H:%M:%S")))


def ago(s: float) -> str:
    return f"{s / 60:.0f} min" if s < 5400 else f"{s / 3600:.1f} h" if s < 4 * 86400 else f"{s / 86400:.1f} d"


def usd(x: float) -> str:
    return f"${x:,.2f}" if x < 100 else f"${x:,.0f}"


@dataclass
class Finding:
    key: str
    text: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)       # the digest's lines
    gaps: list[str] = field(default_factory=list)        # `needs KEY: what it would watch`


class Unreadable(Exception):
    """A check that could not read its source: after `unreadable_passes` passes in a row it opens `<check>|unreadable`."""


def http(method: str, url: str, headers: dict | None = None, body: bytes | None = None, timeout: float = 30) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def run(argv: list[str], env: dict, stdin: str = "") -> tuple[int, str, str]:
    p = subprocess.run(argv, input=stdin, env=env, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout, p.stderr


@dataclass
class Ctx:
    cfg: dict
    env: dict
    state: dict
    now: float
    http: Callable = http
    run: Callable = run
    pages: int | None = None                            # --pages: override r2.page_cap for this pass
    clock: Callable[[], float] = time.time              # signing time: R2 refuses a signature over 15 min old, and a long pass outlives `now`


# ---- loops -------------------------------------------------------------------------------------------------------------------------

def tail_match(path: Path, pattern: re.Pattern, chunk: int = 1 << 18, limit: int = 16 << 20) -> str | None:
    """The time group of the last line of `path` matching `pattern`, reading backwards up to `limit` bytes."""
    with open(path, "rb") as f:
        end = f.seek(0, 2)
        pos, carry = end, b""
        while pos > 0 and end - pos < limit:
            start = max(0, pos - chunk)
            f.seek(start)
            block = f.read(pos - start) + carry
            lines = block.split(b"\n")
            carry = lines[0] if start else b""
            for ln in reversed(lines[1:] if start else lines):
                if m := pattern.search(ln.decode(errors="replace")):
                    return m.group(1)
            pos = start
    return None


def marker_time(loop: dict, now: float) -> float | None:
    """When `loop` last succeeded, from its marker; None when the marker holds no success."""
    path = Path(loop["file"])
    if "json" in loop:
        v = json.loads(path.read_text()).get(loop["json"])
        return None if v is None else float(v) if isinstance(v, (int, float)) else epoch(str(v))
    if "pattern" not in loop:
        return path.stat().st_mtime
    got = tail_match(path, re.compile(loop["pattern"]))
    for older in loop.get("rotated", []):              # a log just rotated holds its last success in the rotated copy
        if got is None and Path(older).is_file():
            got = tail_match(Path(older), re.compile(loop["pattern"]))
    if got is None:
        return None
    if re.fullmatch(r"\d\d:\d\dZ", got):                # a time of day (the steward's log): the latest such time not after now
        day = now - now % 86400
        t = day + int(got[:2]) * 3600 + int(got[3:5]) * 60
        return t if t <= now + 60 else t - 86400
    return epoch(got)


def check_loops(ctx: Ctx, rep: Report) -> None:
    for loop in ctx.cfg.get("loop", []):
        name, period = loop["name"], float(loop["period_s"])
        try:
            t = marker_time(loop, ctx.now)
        except FileNotFoundError:
            rep.findings.append(Finding(f"loop|{name}", f"*{name}* has no success marker: `{loop['file']}` is missing. "
                                                         f"Started by {loop.get('started_by', 'post_start.sh')}."))
            rep.lines.append(f"{name}: no marker")
            continue
        except (OSError, ValueError) as e:
            rep.findings.append(Finding(f"loop|{name}", f"*{name}*'s success marker `{loop['file']}` is unreadable ({type(e).__name__})."))
            rep.lines.append(f"{name}: marker unreadable")
            continue
        late = None if t is None else ctx.now - t
        if t is None or late > 2 * period:
            last = "no success line in its log" if t is None else f"last success {iso(t)}, {ago(late)} ago"
            rep.findings.append(Finding(f"loop|{name}", f"*{name}* has not succeeded in twice its period ({ago(2 * period)}): {last}. "
                                                         f"Marker `{loop['file']}`; started by {loop.get('started_by', 'post_start.sh')}."))
        rep.lines.append(f"{name}: " + ("no success line" if t is None else f"ok {ago(max(late, 0))} ago" if late <= 2 * period
                                        else f"LATE, last success {ago(late)} ago"))


# ---- R2 ------------------------------------------------------------------------------------------------------------------------------

def canonical_query(query: dict[str, str]) -> str:
    q = lambda s: urllib.parse.quote(s, safe="-_.~")  # noqa: E731
    return "&".join(f"{q(k)}={q(v)}" for k, v in sorted(query.items()))


def sign(method: str, host: str, path: str, query: dict[str, str], key: str, secret: str, amz_date: str,
         region: str = "auto") -> dict[str, str]:
    """SigV4 headers for a request with an empty body (research.store.remote_s3's scheme)."""
    h = {"host": host, "x-amz-content-sha256": EMPTY_SHA256, "x-amz-date": amz_date}
    signed = ";".join(sorted(h))
    creq = "\n".join([method, urllib.parse.quote(path, safe="-_.~/"), canonical_query(query),
                      "".join(f"{k}:{h[k]}\n" for k in sorted(h)), signed, EMPTY_SHA256])
    scope = f"{amz_date[:8]}/{region}/s3/aws4_request"
    sts = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(creq.encode()).hexdigest()])
    k = ("AWS4" + secret).encode()
    for part in (amz_date[:8], region, "s3", "aws4_request"):
        k = hmac.new(k, part.encode(), hashlib.sha256).digest()
    sig = hmac.new(k, sts.encode(), hashlib.sha256).hexdigest()
    return {**h, "Authorization": f"AWS4-HMAC-SHA256 Credential={key}/{scope},SignedHeaders={signed},Signature={sig}"}


def list_page(ctx: Ctx, b: dict, after: str | None) -> tuple[list[tuple[str, int, float]], bool]:
    """One ListObjectsV2 page of bucket `b` after key `after`: (key, size, last modified) and whether more follow."""
    u = urllib.parse.urlsplit(b["endpoint"])
    query = {"list-type": "2", "max-keys": "1000", **({"start-after": after} if after else {})}
    path = f"/{b['bucket']}"
    headers = sign("GET", u.netloc, path, query, b["key"], b["secret"], time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(ctx.clock())))
    status, body = ctx.http("GET", f"{u.scheme}://{u.netloc}{urllib.parse.quote(path, safe='-_.~/')}?{canonical_query(query)}", headers)
    if status != 200:
        raise Unreadable(f"R2 {b['label']}: ListObjectsV2 answered HTTP {status}")
    root = ET.fromstring(body)
    keys = [(c.findtext(f"{S3_NS}Key"), int(c.findtext(f"{S3_NS}Size") or 0), epoch(c.findtext(f"{S3_NS}LastModified") or "1970-01-01T00:00:00Z"))
            for c in root.iter(f"{S3_NS}Contents")]
    return keys, root.findtext(f"{S3_NS}IsTruncated") == "true"


def top(key: str) -> str:
    return key.split("/", 1)[0] + "/" if "/" in key else "(top level)"


def buckets(ctx: Ctx, rep: Report) -> list[dict]:
    out = []
    for b in ctx.cfg["r2"]["bucket"]:
        need = [b["bucket_env"], b["key_env"], b["secret_env"]]
        endpoint = next((ctx.env[e] for e in b["endpoint_env"] if ctx.env.get(e)), None)
        missing = [e for e in need if not ctx.env.get(e)] + ([] if endpoint else [" or ".join(b["endpoint_env"])])
        if missing:
            rep.gaps.append(f"needs {', '.join(missing)}: R2 bucket `{b['label']}`'s size and growth"
                            + (f", and its newest object within {b['stale_after_h']} h" if b.get("stale_after_h") else ""))
            continue
        out.append({"label": b["label"], "bucket": ctx.env[b["bucket_env"]], "key": ctx.env[b["key_env"]],
                    "secret": ctx.env[b["secret_env"]], "endpoint": endpoint, "stale_after_h": b.get("stale_after_h"),
                    "writer": b.get("writer", "")})
    return out


def growth(sweeps: list[dict]) -> tuple[float, str, float] | None:
    """(bytes a day over the last two sweeps, the prefix that grew most, its bytes a day); None before two sweeps."""
    if len(sweeps) < 2:
        return None
    a, b = sweeps[-2], sweeps[-1]
    days = max((b["t"] - a["t"]) / 86400, 1 / 24)
    delta = {p: b["prefixes"].get(p, [0, 0])[0] - a["prefixes"].get(p, [0, 0])[0] for p in set(a["prefixes"]) | set(b["prefixes"])}
    big = max(delta, key=lambda p: delta[p])
    return sum(delta.values()) / days, big, delta[big] / days


def mean_growth(sweeps: list[dict], now: float, days: float = 7) -> float | None:
    """Bytes a day from the oldest sweep within `days` to the newest; None without two sweeps a day apart in that window."""
    recent = [s for s in sweeps if s["t"] >= now - days * 86400]
    if len(recent) < 2 or recent[-1]["t"] - recent[0]["t"] < 86400:
        return None
    total = lambda s: sum(v[0] for v in s["prefixes"].values())  # noqa: E731
    return (total(recent[-1]) - total(recent[0])) / ((recent[-1]["t"] - recent[0]["t"]) / 86400)


def check_r2(ctx: Ctx, rep: Report) -> None:
    cfg = ctx.cfg["r2"]
    for b in buckets(ctx, rep):
        st = ctx.state.setdefault("r2", {}).setdefault(b["label"], {"after": None, "partial": {}, "started": ctx.now, "sweeps": []})
        pages, done = 0, False
        while pages < (ctx.pages or cfg["page_cap"]):
            keys, more = list_page(ctx, b, st["after"])
            pages += 1
            for k, size, modified in keys:
                acc = st["partial"].setdefault(top(k), [0, 0])
                acc[0] += size
                acc[1] += 1
                st["newest"] = max(st.get("newest") or 0, modified)
            if keys:
                st["after"] = keys[-1][0]
            if not more:
                done = True
                break
        st["list_ops"] = st.get("list_ops", {})
        month = iso(ctx.now)[:7]
        st["list_ops"][month] = st["list_ops"].get(month, 0) + pages
        if done:
            st["sweeps"] = (st["sweeps"] + [{"t": ctx.now, "started": st["started"], "prefixes": st["partial"],
                                             "newest": st.get("newest")}])[-cfg["keep_sweeps"]:]
            st.update(after=None, partial={}, started=ctx.now, newest=None)
        if not st["sweeps"]:
            n = sum(v[1] for v in st["partial"].values())
            rep.lines.append(f"R2 `{b['label']}`: first sweep in progress, {n:,} objects listed so far")
            continue
        last = st["sweeps"][-1]
        size, count = sum(v[0] for v in last["prefixes"].values()), sum(v[1] for v in last["prefixes"].values())
        biggest = sorted(last["prefixes"].items(), key=lambda kv: -kv[1][0])[:5]
        newest = last.get("newest")
        rep.lines.append(f"R2 `{b['label']}`: {size / GIB:,.1f} GiB in {count:,} objects (sweep of {iso(last['t'])}"
                         + (f", newest write {iso(newest)}" if newest else "") + "); largest "
                         + ", ".join(f"`{p}` {v[0] / GIB:,.1f} GiB" for p, v in biggest))
        if b["stale_after_h"] and newest and last["t"] - newest > b["stale_after_h"] * 3600:
            rep.findings.append(Finding(f"r2|{b['label']}|stale", f"*R2 `{b['label']}` has had no new object for {ago(last['t'] - newest)}* "
                                                                   f"(newest {iso(newest)}; expected within {b['stale_after_h']} h). "
                                                                   + b["writer"]))
        g = growth(st["sweeps"])
        if g is None:
            continue
        rate, big, big_rate = g
        mean = mean_growth(st["sweeps"], ctx.now)
        rep.lines.append(f"R2 `{b['label']}` growth: {rate / GIB:+,.1f} GiB/day (most: `{big}` {big_rate / GIB:+,.1f})"
                         + (f", 7-day mean {mean / GIB:+,.1f}" if mean is not None else ""))
        too_fast = rate > cfg["max_gib_per_day"] * GIB
        above_mean = mean is not None and rate > cfg["mean_ratio"] * max(mean, cfg["mean_floor_gib_per_day"] * GIB)
        if too_fast or above_mean:
            why = f"over {cfg['max_gib_per_day']} GiB/day" if too_fast else f"over {cfg['mean_ratio']}x its 7-day mean ({mean / GIB:,.1f})"
            rep.findings.append(Finding(f"r2|{b['label']}|growth", f"*R2 `{b['label']}` growing {rate / GIB:,.1f} GiB/day*, {why}. "
                                                                    f"Biggest writer: `{big}` at {big_rate / GIB:+,.1f} GiB/day."))


# ---- Neon --------------------------------------------------------------------------------------------------------------------------

def pg_env(url: str) -> dict[str, str]:
    """libpq's environment for a postgres URL (research.store.bucket_index.pg_env), so the URL never sits on a command line."""
    u = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qs(u.query)
    env = {"PGHOST": u.hostname or "", "PGPORT": str(u.port or 5432), "PGDATABASE": urllib.parse.unquote(u.path.lstrip("/")),
           "PGUSER": urllib.parse.unquote(u.username or ""), "PGPASSWORD": urllib.parse.unquote(u.password or "")}
    for key, var in (("host", "PGHOST"), ("sslmode", "PGSSLMODE"), ("channel_binding", "PGCHANNELBINDING"), ("options", "PGOPTIONS")):
        if v := query.get(key):
            env[var] = v[-1]
    return {k: v for k, v in env.items() if v}


def scrub(err: str, pg: dict[str, str]) -> str:
    """psql's last error line without the connection's host, address, user, database or password."""
    line = err.strip().splitlines()[-1] if err.strip() else ""
    for k in ("PGPASSWORD", "PGHOST", "PGUSER", "PGDATABASE"):
        if pg.get(k):
            line = line.replace(pg[k], "…")
    return re.sub(r"\(\d+\.\d+\.\d+\.\d+\)|\([0-9a-f:]+:[0-9a-f:]*\)", "(…)", line)[:200]


def check_neon(ctx: Ctx, rep: Report) -> None:
    cfg = ctx.cfg["neon"]
    url = ctx.env.get(cfg["url_env"])
    if not url:
        rep.gaps.append(f"needs {cfg['url_env']}: Neon's size, connections and the store index's last write")
        return
    if not shutil.which(cfg.get("psql", "psql"), path=ctx.env.get("PATH")):
        rep.gaps.append("needs psql: Neon's size, connections and the store index's last write")
        return
    sql = ("select pg_database_size(current_database()), (select count(*) from pg_stat_activity), "
           "(select setting from pg_settings where name = 'max_connections')"
           + "".join(f", (select extract(epoch from max(listed_at)) from {s}.listings)" for s in cfg["schemas"])
           + "".join(f", (select count(*) from pg_stat_user_tables where schemaname = '{s}')" for s in cfg.get("report_schemas", []))
           + ";")
    pg = pg_env(url)
    env = {"PATH": ctx.env.get("PATH", "/usr/bin:/bin"), "PGCONNECT_TIMEOUT": "20", **pg}
    rc, out, err = ctx.run([cfg.get("psql", "psql"), "-X", "-q", "-A", "-t", "-F", "\t", "-v", "ON_ERROR_STOP=1", "-f", "-"], env, sql)
    if rc != 0:
        raise Unreadable(f"Neon: psql exited {rc}: {scrub(err, pg)}")
    size, conns, maxconn, *rest = out.strip().split("\t")
    size, conns, maxconn = int(size), int(conns), int(maxconn)
    writes, tables = rest[:len(cfg["schemas"])], rest[len(cfg["schemas"]):]
    rep.lines.append(f"Neon: {size / GIB:,.2f} GiB, {conns} of {maxconn} connections; store index last write "
                     + ", ".join(f"`{s}` {ago(ctx.now - float(w)) + ' ago' if w else 'never'}" for s, w in zip(cfg["schemas"], writes))
                     + "".join(f"; `{s}` has {n} tables" if int(n) else f"; `{s}` has no tables (never built)"
                               for s, n in zip(cfg.get("report_schemas", []), tables)))
    if size > cfg["size_alert_gib"] * GIB:
        rep.findings.append(Finding("neon|size", f"*Neon database at {size / GIB:,.2f} GiB*, over the {cfg['size_alert_gib']} GiB alert line."))
    if conns > cfg["connections_share"] * maxconn:
        rep.findings.append(Finding("neon|connections", f"*Neon at {conns} of {maxconn} connections* (over {cfg['connections_share']:.0%})."))
    for s, w in zip(cfg["schemas"], writes):
        if not w or ctx.now - float(w) > cfg["index_stale_s"]:
            rep.findings.append(Finding(f"neon|{s}|stale", f"*Store index `{s}` not written for "
                                                          f"{ago(ctx.now - float(w)) if w else 'ever'}* (max `listed_at`; "
                                                          f"alert line {ago(cfg['index_stale_s'])}). Log: `/data/infra/store-index.log`."))


# ---- spend -------------------------------------------------------------------------------------------------------------------------

def budget_lines(ctx: Ctx) -> tuple[dict, str | None]:
    for p in ctx.cfg["spend"]["budgets"]:
        path = Path(os.path.expanduser(p))
        if path.is_file():
            return tomllib.loads(path.read_text()).get("budgets", {}), str(path)
    return {}, None


def line_of(name: str, lines: dict) -> str | None:
    return max((p for p in lines if name.startswith(p)), key=len, default=None)


def accrue(ctx: Ctx, service: str, rate_per_h: float) -> str:
    """Spend this watch has seen this month at `rate_per_h`, integrated over passes (each counted at most 2 x every_s), as text."""
    acc = ctx.state.setdefault("accrued", {}).setdefault(service, {})
    month = iso(ctx.now)[:7]
    if acc.get("month") != month:
        acc.update(month=month, usd=0.0, since=ctx.now, last=ctx.now)
    acc["usd"] += rate_per_h * min(ctx.now - acc["last"], 2 * ctx.cfg["every_s"]) / 3600
    acc["last"] = ctx.now
    return f"{usd(acc['usd'])} seen by this watch since {iso(acc['since'])[:16]}Z"


def month_hours(now: float) -> tuple[float, float]:
    """Hours elapsed in this UTC month, and hours left in it."""
    y, m = time.gmtime(now)[:2]
    start = calendar.timegm((y, m, 1, 0, 0, 0))
    end = calendar.timegm((y + m // 12, m % 12 + 1, 1, 0, 0, 0))
    return (now - start) / 3600, (end - now) / 3600


ACCOUNT_RE = re.compile(r"ACCOUNT \$[\d,.]+ at \$([\d,.]+)/h")


def account_rate(log: Path) -> float | None:
    """RunPod's own spend rate for the account (every pod and volume), from the steward's last `ACCOUNT` line."""
    try:
        got = tail_match(log, ACCOUNT_RE, limit=1 << 20)
    except OSError:
        return None
    return None if got is None else float(got.replace(",", ""))


def pods_this_month(guard: dict, start: float) -> float:
    """The guarded pods' spend since `start`: each pod's `spent`, prorated over the part of its life seen after `start`."""
    total = 0.0
    for p in (guard.get("pods") or {}).values():
        first, last, spent = float(p.get("first_seen") or 0), float(p.get("last_seen") or 0), float(p.get("spent") or 0)
        if last <= start or not spent:
            continue
        total += spent * (1.0 if first >= start or last <= first else (last - start) / (last - first))
    return total


def spend_runpod(ctx: Ctx, rep: Report, lines: dict) -> None:
    cfg = ctx.cfg["spend"]
    path = Path(cfg["guard_state"])
    if not path.is_file():
        rep.gaps.append(f"needs the budgets guard's state (`{path}`, on the control pod): RunPod balance, rate and each line's spend")
        return
    g = json.loads(path.read_text())
    balance, floor = float(g.get("balance") or 0), float((g.get("budgets") or {}).get("balance_floor") or 0)
    rate = account_rate(Path(cfg["steward_log"]))
    if rate is None:                                     # the guard's rate leaves out exempt pods (this one) and volumes
        rate = float(g.get("rate_now") or 0) + ctx.cfg["prices"]["runpod_control_usd_per_hour"]
    elapsed, left = month_hours(ctx.now)
    runway = (balance - floor) / rate if rate > 0 else float("inf")
    pods = pods_this_month(g, ctx.now - elapsed * 3600)
    standing = max(rate - float(g.get("rate_now") or 0), 0)       # what the guard doesn't count: this pod and the volumes
    mtd = pods + standing * elapsed
    rep.lines.append(f"RunPod: balance {usd(balance)} (floor {usd(floor)}), {usd(rate)}/h for the account; this month about "
                     f"{usd(mtd)} ({usd(pods)} of guarded pods, the rest this pod and the volumes), {usd(mtd + rate * left)} "
                     f"projected to month end")
    if balance < cfg["runpod_topup_below_usd"]:
        rep.findings.append(Finding("spend|runpod|runway", f"*RunPod balance {usd(balance)} is under its {usd(cfg['runpod_topup_below_usd'])} "
                                                           f"auto top-up: the top-up failed*, and the balance reaches the guard's "
                                                           f"{usd(floor)} floor in {runway:.0f} h at {usd(rate)}/h, where the guard "
                                                           f"terminates every pod in scope."))
    for prefix, st in (g.get("lines") or {}).items():
        line = lines.get(prefix)
        if not line or epoch(str(line.get("expires", "1970-01-01T00:00Z"))) < ctx.now:
            continue
        cap, spent = line.get("cap_usd"), float(st.get("spent") or 0)
        if line.get("cap_usd_per_day") is not None:
            cap, spent = line["cap_usd_per_day"], float(st.get("window_usd") or 0)
        if cap and spent >= cfg["line_alert_share"] * cap:
            rep.findings.append(Finding(f"spend|runpod|{prefix}", f"*Budget line `{prefix}` at {usd(spent)} of {usd(cap)}*"
                                                                  + (" today" if line.get("cap_usd_per_day") is not None else "")
                                                                  + " on RunPod (the guard terminates its pods at the cap)."))


def nebius_instances(ctx: Ctx, rep: Report) -> list[dict] | None:
    cfg = ctx.cfg["nebius"]
    missing = [e for e in cfg["key_env"] if not ctx.env.get(e)]
    if missing or not cfg.get("instances_command"):
        rep.gaps.append(f"needs {', '.join(missing) or 'nebius.instances_command'}: Nebius VMs' burn against their budget lines, "
                        "and a VM on no line or past its line's expiry")
        return None
    rc, out, err = ctx.run(cfg["instances_command"], ctx.env)
    if rc != 0:
        raise Unreadable(f"Nebius instance list exited {rc}: {err.strip().splitlines()[-1][:200] if err.strip() else ''}")
    return json.loads(out)


def rates(ctx: Ctx) -> dict[str, float]:
    for d in ctx.cfg["nebius"]["machines_d"]:
        p = Path(os.path.expanduser(d))
        if p.is_dir():
            out = {}
            for f in p.glob("*.toml"):
                try:
                    r = tomllib.loads(f.read_text()).get("rate_usd_per_hour")
                except (OSError, tomllib.TOMLDecodeError):
                    continue
                if r is not None:
                    out[f.stem] = float(r)
            return out
    return {}


def spend_nebius(ctx: Ctx, rep: Report, lines: dict) -> None:
    vms = nebius_instances(ctx, rep)
    if vms is None:
        return
    running = [v for v in vms if v.get("state") == "RUNNING"]
    rate_of = rates(ctx)
    burn: dict[str, float] = {}
    unknown = []
    for v in running:
        name, prefix = v["name"], line_of(v["name"], lines)
        rate = rate_of.get(name)
        if rate is None:
            unknown.append(name)
        line = lines.get(prefix) if prefix else None
        exp = epoch(str(line["expires"])) if line and line.get("expires") else None
        if not line:
            rep.findings.append(Finding(f"spend|nebius|{name}|uncovered", f"*Nebius VM `{name}` runs on no budget line*"
                                                                          + (f" ({usd(rate)}/h)" if rate else "") + "."))
        elif exp is not None and exp < ctx.now:
            rep.findings.append(Finding(f"spend|nebius|{name}|expired", f"*Nebius VM `{name}` runs past its line `{prefix}`'s expiry* "
                                                                        f"({line['expires']})" + (f", {usd(rate)}/h" if rate else "") + "."))
        if prefix and rate:
            burn[prefix] = burn.get(prefix, 0.0) + rate
    total = sum(burn.values())
    _, left = month_hours(ctx.now)
    rep.lines.append(f"Nebius: {len(running)} VMs running, {usd(total)}/h known ({usd(total * 24)}/day), "
                     f"{accrue(ctx, 'nebius', total)}, {usd(total * left)} more by month end at this rate"
                     + (f"; no rate registered for {', '.join(f'`{n}`' for n in sorted(unknown))}" if unknown else ""))
    for prefix, rate in sorted(burn.items(), key=lambda kv: -kv[1]):
        line = lines[prefix]
        hours = max(epoch(str(line["expires"])) - ctx.now, 0) / 3600 if line.get("expires") else month_hours(ctx.now)[1]
        rep.lines.append(f"  `{prefix}`: {usd(rate)}/h, " + (f"{usd(rate * 24)}/day of {usd(line['cap_usd_per_day'])}/day"
                                                         if line.get("cap_usd_per_day") is not None else
                                                         f"{usd(rate * hours)} more to its expiry, cap {usd(line.get('cap_usd') or 0)}"))
        if line.get("cap_usd_per_day") is not None and rate * 24 > line["cap_usd_per_day"]:
            rep.findings.append(Finding(f"spend|nebius|{prefix}|day", f"*Budget line `{prefix}` burning {usd(rate * 24)}/day* on Nebius, "
                                                                      f"over its {usd(line['cap_usd_per_day'])}/day."))
        elif line.get("cap_usd") and rate * hours > line["cap_usd"]:
            rep.findings.append(Finding(f"spend|nebius|{prefix}|cap", f"*Budget line `{prefix}`'s running VMs alone reach "
                                                                      f"{usd(rate * hours)} by its expiry*, over its {usd(line['cap_usd'])} cap "
                                                                      f"({usd(rate)}/h for {hours:.0f} h)."))


def spend_r2(ctx: Ctx, rep: Report) -> None:
    p, cfg = ctx.cfg["prices"], ctx.cfg["spend"]
    st = ctx.state.get("r2", {})
    sweeps = [s["sweeps"][-1] for s in st.values() if s.get("sweeps")]
    if not sweeps:
        return
    gb = sum(v[0] for s in sweeps for v in s["prefixes"].values()) / p["r2_gb_bytes"]
    storage = max(gb - p["r2_free_gb"], 0) * p["r2_storage_usd_per_gb_month"]
    # class A ops: at least one PUT per new object (a multipart upload is several) plus our own listings; exact needs analytics
    month = iso(ctx.now)[:7]
    new = 0
    for s in st.values():
        old = [w for w in s.get("sweeps", []) if iso(w["t"])[:7] == month]
        if len(old) >= 2:
            new += max(sum(v[1] for v in old[-1]["prefixes"].values()) - sum(v[1] for v in old[0]["prefixes"].values()), 0)
    ops = new + sum(s.get("list_ops", {}).get(month, 0) for s in st.values())
    elapsed, left = month_hours(ctx.now)
    ops_month = ops * (elapsed + left) / max(elapsed, 1)
    class_a = max(ops_month / 1e6 - p["r2_class_a_free_million"], 0) * p["r2_class_a_usd_per_million"]
    rep.lines.append(f"R2: {gb:,.0f} GB stored, {usd(storage)}/month at list price; class-A ops at least {ops_month / 1e6:,.2f} M "
                     f"this month ({usd(class_a)})")
    if storage + class_a > cfg["r2_alert_usd_per_month"]:
        rep.findings.append(Finding("spend|r2", f"*R2 on pace for {usd(storage + class_a)} this month*, over the "
                                                f"{usd(cfg['r2_alert_usd_per_month'])} alert line ({gb:,.0f} GB stored). No budget line covers R2."))


def check_spend(ctx: Ctx, rep: Report) -> None:
    lines, source = budget_lines(ctx)
    if source is None:
        rep.gaps.append("needs budgets.toml (the steward's notes clone): the budget lines spend is compared with")
    spend_runpod(ctx, rep, lines)
    spend_nebius(ctx, rep, lines)
    spend_r2(ctx, rep)
    rep.gaps.extend(f"needs {g['key']}: {g['closes']}" for g in ctx.cfg["spend"].get("gap", []) if not ctx.env.get(g["key"]))


# ---- GitHub and the public endpoints ---------------------------------------------------------------------------------------------

def check_github(ctx: Ctx, rep: Report) -> None:
    cfg = ctx.cfg["github"]
    token = ctx.env.get(cfg["token_env"])
    if not token:
        rep.gaps.append(f"needs {cfg['token_env']}: the GitHub API rate limit")
        return
    status, body = ctx.http("GET", "https://api.github.com/rate_limit", {"Authorization": f"Bearer {token}",
                                                                         "User-Agent": "verity-cloud-watch"})
    if status != 200:
        raise Unreadable(f"GitHub rate_limit answered HTTP {status}")
    core = json.loads(body)["resources"]["core"]
    rep.lines.append(f"GitHub: {core['remaining']:,} of {core['limit']:,} core requests left, resets {iso(core['reset'])}")
    if core["remaining"] < cfg["remaining_share"] * core["limit"]:
        rep.findings.append(Finding("github|core", f"*GitHub API: {core['remaining']:,} of {core['limit']:,} core requests left* "
                                                   f"for `{cfg['token_env']}` until {iso(core['reset'])}."))


def check_jobs(ctx: Ctx, rep: Report) -> None:
    cfg = ctx.cfg["jobs"]
    token = ctx.env.get(cfg["token_env"])
    if not token:
        rep.gaps.append(f"needs {cfg['token_env']}: the Job queue's queued jobs no worker claims")
        return
    status, body = ctx.http("GET", f"{cfg['url']}/api/jobs?state=queued&limit={cfg['limit']}",
                            {"Authorization": f"Bearer {token}", "User-Agent": "verity-cloud-watch"})
    if status != 200:
        raise Unreadable(f"Job queue answered HTTP {status}")
    jobs = json.loads(body)["jobs"]
    waiting = sorted(((ctx.now - max(epoch(j["enqueued_at"]), epoch(j["not_before"]) if j.get("not_before") else 0), j) for j in jobs),
                     key=lambda aj: aj[0])
    old = [(age, j) for age, j in waiting if age > cfg["stale_queued_h"] * 3600]
    rep.lines.append(f"Job queue: {len(jobs)}{'+' if len(jobs) >= cfg['limit'] else ''} queued"
                     + (f", oldest waiting {ago(waiting[-1][0])}" if waiting else ""))
    if old:
        kinds: dict[str, int] = {}
        for _, j in old:
            kinds[j["kind"]] = kinds.get(j["kind"], 0) + 1
        age, j = old[-1]
        rep.findings.append(Finding("jobs|stale", f"*{len(old)} jobs queued over {cfg['stale_queued_h']} h with no worker claiming them* ("
                                                  + ", ".join(f"{n} `{k}`" for k, n in sorted(kinds.items())) + f"); oldest #{j['id']} "
                                                  f"since {iso(ctx.now - age)}. `research jobs list --state queued`."))


def check_http(ctx: Ctx, rep: Report) -> None:
    for ep in ctx.cfg.get("http", []):
        try:
            status, _ = ctx.http("GET", ep["url"], {"User-Agent": "verity-cloud-watch"}, timeout=20)
        except (urllib.error.URLError, OSError) as e:
            status, why = None, f"unreachable ({type(e).__name__})"
        else:
            why = f"HTTP {status}"
        ok = status is not None and status < 500
        rep.lines.append(f"{ep['name']}: {why}")
        if not ok:
            rep.findings.append(Finding(f"http|{ep['name']}", f"*{ep['name']}* answers {why}: <{ep['url']}>."))


KEY_PREFIX = {"loops": "loop"}
CHECKS: dict[str, Callable[[Ctx, Report], None]] = {"loops": check_loops, "r2": check_r2, "neon": check_neon, "spend": check_spend,
                                                    "github": check_github, "jobs": check_jobs, "http": check_http}


# ---- the pass --------------------------------------------------------------------------------------------------------------------

def evaluate(ctx: Ctx) -> tuple[dict[str, Report], dict[str, Finding]]:
    """Every check's report, and the findings this pass by key (a check that failed: its unreadable finding, counted)."""
    reports, found = {}, {}
    fails = ctx.state.setdefault("fails", {})
    for name, check in CHECKS.items():
        rep = reports[name] = Report()
        try:
            check(ctx, rep)
            fails.pop(name, None)
        except Exception as e:  # noqa: BLE001  (any check's failure is that check unreadable, never a silent watch)
            fails[name] = fails.get(name, 0) + 1
            rep.lines.append(f"{name}: unreadable ({e if isinstance(e, Unreadable) else type(e).__name__})")
            if fails[name] >= ctx.cfg["unreadable_passes"]:
                rep.findings.append(Finding(f"{name}|unreadable", f"*Cloud watch can't read `{name}`* for {fails[name]} passes: "
                                                                  f"{e if isinstance(e, Unreadable) else type(e).__name__}."))
            # a failed check keeps what it had open: what it didn't read hasn't cleared
            for key, o in ctx.state.get("open", {}).items():
                if key.split("|", 1)[0] == KEY_PREFIX.get(name, name) and key != f"{name}|unreadable":
                    found[key] = Finding(key, o["text"])
        for f in rep.findings:
            found[f.key] = f
    return reports, found


def transitions(ctx: Ctx, found: dict[str, Finding]) -> list[tuple[str, str, str | None]]:
    """(key, text, thread) of each post this pass makes: an open (thread None) or a clear (its open post's thread)."""
    opened, gone, out = ctx.state.setdefault("open", {}), ctx.state.setdefault("gone", {}), []
    for key, f in found.items():
        gone.pop(key, None)
        o = opened.setdefault(key, {"since": ctx.now, "text": f.text, "ts": None, "posted": False})
        if not o["posted"]:
            out.append((key, f"*Cloud watch:* <!subteam^{ctx.cfg['mention']}> {f.text}", None))
    for key in [k for k in opened if k not in found]:
        gone[key] = gone.get(key, 0) + 1
        if gone[key] >= ctx.cfg["clear_passes"]:
            o = opened[key]
            if o["posted"]:
                out.append((key, f"*Cloud watch, cleared* after {ago(ctx.now - o['since'])}: {o['text']}", o["ts"]))
            else:
                opened.pop(key)
                gone.pop(key)
    return out


def digest(ctx: Ctx, reports: dict[str, Report]) -> str:
    head = f"*Cloud health and spend, {iso(ctx.now)[:16]}Z* (`infra/control/cloud_watch.py`, INF-73)"
    body = [ln for r in reports.values() for ln in r.lines]
    open_ = [o["text"] for o in ctx.state.get("open", {}).values()]
    gaps = sorted({g for r in reports.values() for g in r.gaps})
    return "\n".join([head, *(f"• {ln}" for ln in body),
                      *([f"*Open* ({len(open_)}):"] + [f"• {t}" for t in open_] if open_ else ["Nothing open."]),
                      *([f"*Unmonitored* ({len(gaps)}), each until its key is in the pod's environment:"] if gaps else []),
                      *(f"• {g}" for g in gaps)])


def post(cfg: dict, env: dict, text: str, thread: str | None, dry: bool, out=None) -> str | None:
    """Post `text` in #agent-alerts (a reply in `thread`); the message's ts, or None when Slack refused it."""
    out = out or sys.stdout
    if dry:
        print(f"would post{' in thread ' + thread if thread else ''}:\n{text}\n", file=out)
        return "DRY-RUN"
    token = env.get("SLACK_BOT_TOKEN")
    if not token:
        print("no SLACK_BOT_TOKEN: not posted", file=out)
        return None
    args = {"channel": cfg["channel"], "username": cfg["username"], "text": text, "unfurl_links": "false", "unfurl_media": "false",
            **({"thread_ts": thread} if thread else {})}
    try:
        status, body = http("POST", "https://slack.com/api/chat.postMessage", {
            "Authorization": f"Bearer {token}", "Content-Type": "application/x-www-form-urlencoded; charset=utf-8"},
            urllib.parse.urlencode(args).encode())
        ans = json.loads(body)
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"post failed: {type(e).__name__}", file=out)
        return None
    if status != 200 or not ans.get("ok"):
        print(f"post refused: HTTP {status} {str(ans.get('error'))[:40]}", file=out)
        return None
    return ans.get("ts")


def one_pass(ctx: Ctx, *, dry: bool = False, force_digest: bool = False, out=None) -> dict[str, Report]:
    reports, found = evaluate(ctx)
    for key, text, thread in transitions(ctx, found):
        ts = post(ctx.cfg, ctx.env, text, thread, dry, out)
        if ts is None:
            continue
        o = ctx.state["open"][key]
        if key in found:
            o.update(posted=True, ts=ts)
        else:
            ctx.state["open"].pop(key)
            ctx.state["gone"].pop(key, None)
    day, hour = iso(ctx.now)[:10], time.gmtime(ctx.now).tm_hour
    if force_digest or (hour >= ctx.cfg["digest_hour_utc"] and ctx.state.get("digest_day") != day):
        if post(ctx.cfg, ctx.env, digest(ctx, reports), None, dry, out) is not None:
            ctx.state["digest_day"] = day
    ctx.state["last_pass"] = ctx.now
    return reports


def load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cloud_watch.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["loop", "once"])
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--state", type=Path, help="default: the config's `state`")
    ap.add_argument("--dry-run", action="store_true", help="print what would be posted; write no state unless --state names it")
    ap.add_argument("--digest", action="store_true", help="compose the digest this pass")
    ap.add_argument("--pages", type=int, help="ListObjectsV2 pages per bucket this pass (default: r2.page_cap)")
    a = ap.parse_args(argv)
    cfg = tomllib.loads(a.config.read_text())
    path = a.state or Path(cfg["state"])
    while True:
        state = load_state(path)
        ctx = Ctx(cfg, dict(os.environ), state, time.time(), pages=a.pages)
        one_pass(ctx, dry=a.dry_run, force_digest=a.digest)
        if not a.dry_run or a.state:
            save_state(path, state)
        print(f"{iso(ctx.now)} pass: {len(state.get('open', {}))} open", flush=True)
        if a.mode == "once":
            return 0
        time.sleep(cfg["every_s"])


if __name__ == "__main__":
    sys.exit(main())
