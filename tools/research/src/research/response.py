"""How fast the leads answer each other on Linear: `research msg linear response`.

A ring is an issue comment whose leading names (`slack.lead_names`, the rule `linear comment` rings by) name handles; each
handle it names, other than the comment's own lane, is one ask. A worker's comment (`lane/NAME`) rings no one
(`linear comment --worker`), so it asks nothing. The ask's answer is the first later comment on the same issue by the rung
handle's lane, its lead or one of its workers. Its minutes run from the ring's `createdAt` to the answer's, Linear's clock,
never a time the text claims. An ask with no answer yet is open, and counts at its age so far, so silence never improves a
percentile. Asks of a ring naming at most MAX_HANDLES handles are counted apart from wider rings (an announcement to every
lead), and the narrow figure is the one a lead is held to.

Who wrote a comment: Linear's `botActor.userDisplayName`, the `createAsUser` name `linear comment --as` gives it (every lane
writes through one app, so the API user is the same for all). A comment under a person's key has no such name: its writer
is the handle its body is signed with after the leading names (`vllm (lead), Fri 13 Nov 4:45 AM PST ...`, or the `@sender:`
a personal key's `linear comment` writes first), and else `unknown`, reported and never guessed.

Percentiles are nearest-rank. Times people read are Pacific (`timefmt.both`); the JSON's are UTC ISO."""
from __future__ import annotations

import math
import re
from datetime import UTC, datetime

from .linear import RING_TAGS_RE, _epoch
from .slack import lead_names
from .timefmt import _day, _utc, both

MAX_HANDLES = 3
UNKNOWN = "unknown"
PERSONAL_RE = re.compile(r"\s*@([a-z0-9][a-z0-9._-]*):\s")       # `Linear.comment` under a personal key: `@sender: body`
SIGNED_RE = re.compile(r"(?:[A-Z]{2,}\.\s+)?([a-z][a-z0-9-]*)(?:/[\w.-]+)?(?:\s*\([^()\n]{1,30}\))?,\s")


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def writer(c: dict, handles: set[str]) -> tuple[str, str]:
    """(who wrote the comment, its body without a personal key's `@sender: `): the app's `--as` name (`lane` or
    `lane/WORKER`), else the handle its body is signed with, else UNKNOWN."""
    body = c.get("body") or ""
    if name := (c.get("botActor") or {}).get("userDisplayName"):
        return name, body
    if (m := PERSONAL_RE.match(body)) and m[1] in handles:
        return m[1], body[m.end():]
    rest = body.lstrip()
    if m := RING_TAGS_RE.match(rest):
        rest = rest[m.end():]
    if (m := SIGNED_RE.match(rest)) and m[1] in handles:
        return m[1], body
    return UNKNOWN, body


def asks(comments: list[dict], reg: dict, now: float, since: float | None = None) -> tuple[list[dict], dict]:
    """Every ask rung by `comments` (Linear's, as `Linear.comments` gives them) at or after `since`, oldest first:
    {issue, url, comment, by, to, width, wide, rung, answered, minutes, open}, `rung` and `answered` UTC ISO and `minutes`
    to the answer, or the age at `now` while open. And what was left out: {unknown: {API name: comments}, worker_rings}."""
    handles = {h["handle"] for h in reg["handles"]}
    rows = []
    for c in comments:
        if not c.get("issue"):
            continue
        by, body = writer(c, handles)
        rows.append({"c": c, "by": by, "lane": by.split("/", 1)[0], "body": body, "t": _epoch(c["createdAt"]),
                     "issue": c["issue"]})
    rows.sort(key=lambda r: r["t"])
    by_issue: dict[str, list[dict]] = {}
    for r in rows:
        by_issue.setdefault(r["issue"]["identifier"], []).append(r)
    out, unknown, worker_rings = [], {}, 0
    for r in rows:
        if r["by"] == UNKNOWN:
            api = (r["c"].get("user") or {}).get("name") or "?"
            unknown[api] = unknown.get(api, 0) + 1
        if since is not None and r["t"] < since:
            continue
        to = [n for n in lead_names(r["body"], reg) if n in handles and n != r["lane"]]
        if not to:
            continue
        if "/" in r["by"]:
            worker_rings += 1
            continue
        later = [x for x in by_issue[r["issue"]["identifier"]] if x["t"] > r["t"]]
        for h in to:
            ans = next((x for x in later if x["lane"] == h), None)
            end = ans["t"] if ans else now
            out.append({"issue": r["issue"]["identifier"], "url": r["issue"].get("url"), "comment": r["c"].get("id"),
                        "by": r["by"], "to": h, "width": len(to), "wide": len(to) > MAX_HANDLES, "rung": _iso(r["t"]),
                        "answered": _iso(ans["t"]) if ans else None, "minutes": round((end - r["t"]) / 60, 1),
                        "open": ans is None})
    return out, {"unknown": dict(sorted(unknown.items())), "worker_rings": worker_rings}


def percentile(values: list[float], p: float) -> float | None:
    """The nearest-rank `p`-th percentile (0 < p <= 100), None of nothing."""
    if not values:
        return None
    v = sorted(values)
    return v[max(math.ceil(p / 100 * len(v)) - 1, 0)]


def figures(rows: list[dict]) -> dict:
    """{received, answered, open, median, p90}: minutes over every ask, an open one at its age so far."""
    mins = [r["minutes"] for r in rows]
    return {"received": len(rows), "answered": sum(not r["open"] for r in rows), "open": sum(r["open"] for r in rows),
            "median": percentile(mins, 50), "p90": percentile(mins, 90)}


def report(found: list[dict], left: dict, since: float, now: float, open_over: float, lane: str | None = None) -> dict:
    """The figures by rung lane and over all, each for asks of at most MAX_HANDLES handles (`narrow`) and wider rings
    (`wide`), and the open asks older than `open_over` minutes, oldest first."""
    rows = [r for r in found if not lane or r["to"] == lane]
    lanes = sorted({r["to"] for r in rows})
    split = lambda rs: {"narrow": figures([r for r in rs if not r["wide"]]),   # noqa: E731
                        "wide": figures([r for r in rs if r["wide"]])}
    return {"since": _iso(since), "now": _iso(now), "max_handles": MAX_HANDLES, "open_over_minutes": open_over,
            "lanes": {x: split([r for r in rows if r["to"] == x]) for x in lanes}, "all": split(rows),
            "open_over": sorted((r for r in rows if r["open"] and r["minutes"] > open_over), key=lambda r: r["rung"]),
            **left, "asks": rows}


def _when(iso: str) -> str:
    """`Fri 13 Nov 10:56 PM PST (06:56Z)`: the Pacific day, then `both`."""
    return f"{_day(_utc(iso), weekday=True)} {both(iso)}"


def _m(v: float | None) -> str:
    return "-" if v is None else f"{v:.0f}"


def render(rep: dict) -> str:
    """`report` as tables people read: one line a lane, narrow asks then wider rings, and the open asks over the limit."""
    head = f"{'lane':<20} {'asks':>5} {'answered':>8} {'open':>5} {'median':>7} {'p90':>6}"

    def line(name: str, f: dict) -> str:
        return f"{name:<20} {f['received']:>5} {f['answered']:>8} {f['open']:>5} {_m(f['median']):>7} {_m(f['p90']):>6}"

    out = [f"Asks rung on Linear from {_when(rep['since'])} to {_when(rep['now'])}: minutes to the rung lane's first comment "
           "after the ring, an open ask at its age so far (nearest-rank median and p90)"]
    for key, what in (("narrow", f"rings naming at most {rep['max_handles']} handles"),
                      ("wide", f"rings naming more than {rep['max_handles']} handles")):
        out += ["", f"{what}:", head]
        out += [line(x, f[key]) for x, f in rep["lanes"].items() if f[key]["received"]]
        out.append(line("all lanes", rep["all"][key]))
    over = rep["open_over"]
    out += ["", f"open over {rep['open_over_minutes']:g} min: {len(over)}"]
    out += [f"  {r['issue']:<10} @{r['to']:<18} rung by {r['by']} {_when(r['rung'])}, open {_m(r['minutes'])} min"
            + (f" (one of {r['width']})" if r["wide"] else "") for r in over]
    if rep["unknown"]:
        out.append("writer unknown (counted for no lane): " + ", ".join(f"{n} comments under {k}" for k, n in rep["unknown"].items()))
    if rep["worker_rings"]:
        out.append(f"not counted: {rep['worker_rings']} workers' comments leading with names (a worker's comment rings no one)")
    return "\n".join(out)
