#!/usr/bin/env python3
"""Top's two loops on the control pod (INF-131; top's spec, #agent-coordination 1792433424.443299), which ran as Cursor timers on
top's VM (asks-relay-v4, admin-hourly-v3) and stopped whenever it was suspended. Each detects and rings: it posts nothing but
inbox doorbells, and whatever needs judgment rings @top instead of being decided.

    top_watch.py asks  [--every 900]  [--state PATH] [--once] [--seed] [--dry-run]    $NOTION_API_KEY, $SLACK_BOT_TOKEN
    top_watch.py sweep [--every 3600] [--state PATH] [--once] [--seed] [--dry-run]    $SLACK_BOT_TOKEN, $GH_TOKEN, $CURSOR_API_KEY,
                                                                                      $LINEAR_CLIENT_ID, $LINEAR_CLIENT_SECRET

**asks**, every 15 min: the Asks for Daniel pages (the registry's #ask-daniel `pages`) edited since the last pass, and every
Default-kind page still Waiting on Daniel past its Deadline.
- Approve ticked: Status Approved, Decision "Approved by Daniel (Approve ticked, <time PT>)".
- Choose set: Status Decided otherwise, Decision "Daniel chose <letter>: <that option's heading on the page>".
- Approve ticked on a Decided by lane page: Status Approved, and " Endorsed by Daniel." appended to its Decision.
- A Default-kind page past its Deadline, still Waiting on Daniel: Status Default applied, Decision "Default applied <time PT>:
  <If no answer>".
- New text under the bold first line of the yellow "Your reply" callout, or a new comment, not starting with an @handle or the
  page's lane (or top) and a colon or a time (`proofs 03:20Z:`): quoted (QUOTE_MAX characters).
Each rings the page's Lane, led by its name, with what to do (`resolve` on its card), and @top with one line, since top writes the
decisions log. The state keeps `since`, the pages decided, the reply pieces seen per page (by block id) and the comment ids
seen.

**sweep**, hourly; each finding rings its owner once (its id kept in `rung` for KEEP_S):
1. `research msg traffic`: an ask unanswered for over an hour from a person or a handle (not a machine: `vy-*`): its addressees.
2. An ACTIVE Verity cloud agent (`GET /v1/agents`) that holds no handle: no registry `holder` and no inbox in the routes names it.
   @top.
3. A new post in #agent-coordination thread RULES_THREAD (deadlocks and rule questions): @top.
4. main's head unchanged for STALL_S (since the sweep first saw it) while `next` is ahead of it (PRs queued): @ci.
5. Once a day after 15:00Z: each H200 under H200_BUSY_PCT mean GPU utilization over 24 h, from fleet.py's samples
   ($FLEET_DIR): @top, with the numbers.
6. The Linear audit (`research.linear`'s `audit`, $LINEAR_CLIENT_ID and $LINEAR_CLIENT_SECRET): one JSON line a pass in
   <state dir>/linear-audit.jsonl, with the counts by lane and kind and #agent-coordination's last 24 h (`traffic --volume`:
   posts, words, posts over 80 words). Each lane with findings, once per LANE_RING_S, with its counts and the command that
   lists them (findings with no lane: @top); once a day after 15:00Z, @top with the totals and their change from the line
   24 h earlier, and the registry's teams the app can't read (`unread`), which the audit skips.

`--seed` runs one pass that records every reply, comment, post and finding as seen or rung and rings nothing (a page's pending
tick is left to the next pass), so a loop taking over from top's timers starts where they are; the watchers supervisor waits for
each state file, which the install seeds. `--dry-run` writes nothing to Notion,
rings nobody and saves no state.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from research import linear, slack

QUOTE_MAX = 600
KEEP_S = 14 * 86400
SINCE_MARGIN_S = 600
UNANSWERED_H = 1
RULES_THREAD = "1792037552.043289"
STALL_S = 3600
H200_BUSY_PCT = 10
REPORT_HOUR_UTC = 15
LANE_RING_S = 12 * 3600
AUDIT_LOG = "linear-audit.jsonl"
DAY_BEFORE_S = 86400 - 1800             # the line 24 h earlier: the newest one at least this old (passes drift within the hour)
REPO = "compute-verification/verity"
OPEN = ("Waiting on Daniel", "Revising")
HANDLE_LEAD = re.compile(r"^\s*@[a-z0-9][a-z0-9._-]*")
OPTION_LETTER = re.compile(r"^([A-E])\.")
STAMP = "%Y-%m-%dT%H:%M:%SZ"


def log(msg: str) -> None:
    print(f"{time.strftime(STAMP, time.gmtime())} {msg}", flush=True)


def iso(t: float) -> str:
    return time.strftime(STAMP, time.gmtime(t))


def pt(t: float) -> str:
    return slack.pacific(datetime.fromtimestamp(t, timezone.utc))


def plain(rich: list | None) -> str:
    return "".join(x.get("plain_text") or (x.get("text") or {}).get("content", "") for x in rich or [])


class Ringer:
    """Ring a name's inbox with a doorbell from `author` (`thread`: the Slack thread it is about, else the inbox's own)."""

    def __init__(self, api: slack.Slack, reg: dict, author: str, dry: bool, routes=None):
        self.api, self.reg, self.author, self.dry = api, reg, author, dry
        self.routes = routes or slack.Routes()

    def __call__(self, name: str, body: str, thread: str | None = None, reply: str | None = None) -> bool:
        to, r = self.routes.resolve(name)
        if not r or r.get("closed"):
            log(f"@{name} has no open inbox: not rung: {body[:200]}")
            return False
        text = slack.doorbell(self.author, thread or r["inbox_ts"], reply or thread or r["inbox_ts"], body)
        if self.dry:
            log(f"would ring @{to}: {text}")
            return True
        self.api.call("chat.postMessage", channel=r.get("channel") or slack.channel_id_for(self.reg, slack.FRONT_DOOR),
                      thread_ts=r["inbox_ts"], username="router", unfurl_links="false", unfurl_media="false", text=text)
        log(f"rang @{to}: {body[:200]}")
        return True


# ---- asks ----

def props(page: dict) -> dict[str, str]:
    return {k: slack.prop_text(v) for k, v in (page.get("properties") or {}).items()}


def card_ts(url: str) -> str | None:
    m = re.search(r"/p(\d{10})(\d{6})", url or "")
    return f"{m[1]}.{m[2]}" if m else None


def query_all(notion, ds: str, body: dict) -> list[dict]:
    out, cursor = [], None
    while True:
        page = notion.call("POST", f"/data_sources/{ds}/query", {**body, "page_size": 100, **({"start_cursor": cursor} if cursor else {})})
        out += page.get("results") or []
        cursor = page.get("next_cursor")
        if not page.get("has_more") or not cursor:
            return out


def page_body(notion, pid: str) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """The reply callout's pieces under its bold first line, in order, as (id, text): its own text's later lines and its
    blocks; and each option's heading by its letter."""
    blocks = notion.call("GET", f"/blocks/{pid}/children?page_size=100").get("results") or []
    reply, headings, found = [], {}, False
    for b in blocks:
        kind = b.get("type")
        v = b.get(kind) or {}
        text = plain(v.get("rich_text"))
        if kind == "callout" and not found and str(v.get("color", "")).startswith("yellow") and "Your reply" in text:
            found = True
            reply += [(f"{b['id']}:{hashlib.sha256(ln.encode()).hexdigest()[:16]}", ln) for ln in text.split("\n")[1:] if ln.strip()]
            if b.get("has_children"):
                for c in notion.call("GET", f"/blocks/{b['id']}/children?page_size=100").get("results") or []:
                    if t := plain((c.get(c.get("type")) or {}).get("rich_text")).strip():
                        reply.append((c["id"], t))
        elif kind == "heading_3" and (m := OPTION_LETTER.match(text)):
            headings.setdefault(m[1], re.sub(r"\s*\(recommended\)\s*$", "", text[len(m[0]):]).strip())
    return reply, headings


def lane_lead(lane: str) -> re.Pattern:
    """How a lane's answer starts: an @handle, or the page's lane or top named with a colon or a time, as lanes sign
    (`proofs:`, `proofs 03:20Z:`, `proofs, 26 Oct 03:20Z:`). `proofs, please …` is still Daniel's."""
    return re.compile(rf"{HANDLE_LEAD.pattern}|^\s*(?:{re.escape(lane)}|top)\s*(?::|,?\s+(?:\d{{1,2}}\s+[A-Z][a-z]{{2}}\s+)?\d{{1,2}}:\d{{2}})")


def daniels(reply: list[tuple[str, str]], seen: set[str], lead: re.Pattern = HANDLE_LEAD) -> list[str]:
    """The pieces new since `seen` that are Daniel's: a lane answers in the callout starting with `lead`, and each new piece
    after that one, up to the next piece already seen, is the lane's too."""
    out, lane = [], False
    for bid, text in reply:
        if bid in seen:
            lane = False
        elif lead.match(text):
            lane = True
        elif not lane:
            out.append(text)
    return out


def comments(notion, pid: str) -> list[dict]:
    return notion.call("GET", f"/comments?block_id={urllib.parse.quote(pid)}&page_size=100").get("results") or []


def decide(p: dict, headings: Callable[[], dict[str, str]], now: float) -> tuple[str, str, str] | None:
    """(Status, Decision, the choice for `resolve`) the page's ticks call for, or None."""
    st, kind = p.get("Status"), p.get("Kind")
    rec = (m[1] if (m := OPTION_LETTER.match(p.get("Recommendation", ""))) else "<value>")
    if st in OPEN and p.get("Approve") == "yes":
        return "Approved", f"Approved by Daniel (Approve ticked, {pt(now)})", rec
    if st in OPEN and p.get("Choose"):
        c = p["Choose"]
        return "Decided otherwise", f"Daniel chose {c}: {headings().get(c, '(no such option on the page)')}", c
    if st == "Decided by lane" and p.get("Approve") == "yes":
        return "Approved", (p.get("Decision", "").rstrip() + " Endorsed by Daniel.").strip(), rec
    dl = p.get("Deadline")
    if st == "Waiting on Daniel" and kind == "Default" and dl and datetime.fromisoformat(dl.replace("Z", "+00:00")).timestamp() <= now:
        dflt = m[1] if (m := OPTION_LETTER.match(p.get("If no answer", ""))) else "<value>"
        return "Default applied", f"Default applied {pt(now)}: {p.get('If no answer', '')}", dflt
    return None


def asks_pass(state: dict, notion, ring: Ringer | None, ds: str, now: float, dry: bool) -> None:
    """One pass; `ring` None seeds (records and rings nothing)."""
    since = state.get("since")
    edited = {"timestamp": "last_edited_time", "last_edited_time": {"on_or_after": iso((since or now - 2 * 86400) - SINCE_MARGIN_S)}}
    overdue = {"and": [{"property": "Status", "status": {"equals": "Waiting on Daniel"}},
                       {"property": "Kind", "select": {"equals": "Default"}},
                       {"property": "Deadline", "date": {"before": iso(now)}}]}
    pages = {p["id"]: p for p in query_all(notion, ds, {"filter": edited}) + query_all(notion, ds, {"filter": overdue})}
    decided, replies, seen = set(state.get("decided", [])), state.setdefault("replies", {}), set(state.get("comments", []))
    said: list = []
    for pid, page in pages.items():
        p, url = props(page), page.get("url", "")
        lane, ask, cid = p.get("Lane") or "top", p.get("Ask", "")[:200], p.get("Card ID") or "<Card ID>"
        thread = card_ts(p.get("Slack card", ""))
        body: dict = {}

        def heads() -> dict[str, str]:
            if "h" not in body:
                body["reply"], body["h"] = page_body(notion, pid)
            return body["h"]

        def tell(what: str, choice: str) -> None:
            if ring is None:
                return
            ring(lane, f"{lane}: {what} on \u201c{ask}\u201d: {url}. Act on it, set the page's Status to match, reply under his text "
                       f"in the callout starting with your handle, and record a decision on the card: research msg ask-daniel "
                       f"resolve {cid} --choice {choice} --note {url}", thread)
            if lane != "top":
                ring("top", f"{lane}'s ask \u201c{ask[:120]}\u201d: {what[:300]} {url}", thread)

        if ring is not None and pid not in decided and (d := decide(p, heads, now)):   # a seed leaves a pending tick to the next pass
            status, decision, choice = d
            if dry:
                log(f"would set {url}: Status {status}, Decision {decision}")
            else:
                notion.call("PATCH", f"/pages/{pid}", {"properties": {"Status": {"status": {"name": status}},
                                                                      "Decision": {"rich_text": [{"text": {"content": decision[:2000]}}]}}})
            tell(decision, choice)
            decided.add(pid)
        heads()
        old, lead = set(replies.get(pid) or []), lane_lead(lane)
        if new := daniels(body["reply"], old, lead):
            tell(f"Daniel replied: \u201c{' '.join(new)[:QUOTE_MAX]}\u201d", "<value>")
        replies[pid] = sorted(old | {bid for bid, _ in body["reply"]})
        try:
            got = comments(notion, pid)
        except slack.SlackError as e:     # a connection without comment capabilities: the callout still relays
            if not said:
                log(f"comments not read: {e}")
            said.append(e)
            got = []
        for c in got:
            if c["id"] in seen:
                continue
            seen.add(c["id"])
            text = plain(c.get("rich_text")).strip()
            if text and not lead.match(text):
                tell(f"Daniel commented: \u201c{text[:QUOTE_MAX]}\u201d", "<value>")
    state.update(since=now, decided=sorted(decided), comments=sorted(seen))


# ---- sweep ----

def once_each(state: dict, fid: str, ring: Ringer | None, name: str, body: str, thread: str | None, now: float,
              reply: str | None = None) -> None:
    rung = state.setdefault("rung", {})
    if fid in rung:
        return
    if ring is None or ring(name, body, thread, reply):
        rung[fid] = now


def unanswered(state, ring, reg, now, traffic: Callable[[], dict]) -> None:
    ch = slack.channel_id_for(reg, slack.FRONT_DOOR)
    for a in traffic().get("unanswered") or []:
        by = str(a.get("by") or "")
        if by.startswith("vy-") or by == "router":
            continue
        for to in a.get("to") or []:
            once_each(state, f"unanswered:{a['ts']}:{to}", ring, to,
                      f"{to}: an ask from {by} has waited over {UNANSWERED_H} h with no answer: "
                      f"{slack.permalink(reg, ch, a['ts'])}. Answer it, or 👀 it and say when.", a["ts"], now)


def holderless(state, ring, now, agents: Callable[[], list[dict]], known: set[str]) -> None:
    for a in agents():
        if a.get("status") == "ACTIVE" and a["id"] not in known:
            once_each(state, f"agent:{a['id']}", ring, "top",
                      f"top: an ACTIVE Verity cloud agent holds no handle and no inbox: {a.get('name') or '(unnamed)'} "
                      f"({a.get('url') or a['id']}), created {a.get('createdAt', '?')}.", None, now)


def known_holders(reg: dict, routes) -> set[str]:
    ids = set(re.findall(r"bc-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", json.dumps(reg.get("handles") or [])))
    return ids | {str(r.get("holder")) for r in routes.all().values() if r.get("holder") and not r.get("closed")}


def rules_thread(state, ring, api, reg, now) -> None:
    ch = slack.channel_id_for(reg, slack.FRONT_DOOR)
    last = state.get("rules_seen") or RULES_THREAD
    msgs = api.paged("conversations.replies", "messages", channel=ch, ts=RULES_THREAD, oldest=last)
    new = sorted((m for m in msgs if float(m["ts"]) > float(last) and m["ts"] != RULES_THREAD), key=lambda m: float(m["ts"]))
    for m in new:
        who = m.get("username") or m.get("user") or "someone"
        if who == "top" or who.startswith("vy-"):
            continue
        once_each(state, f"rules:{m['ts']}", ring, "top",
                  f"top: a new post in the deadlocks and rules thread from {who}: {slack.plain(m.get('text') or '')[:400]} "
                  f"{slack.permalink(reg, ch, m['ts'])}", RULES_THREAD, now, m["ts"])
    if new:
        state["rules_seen"] = new[-1]["ts"]


MAIN_QUERY = """query($owner: String!, $name: String!) { repository(owner: $owner, name: $name) {
  ref(qualifiedName: "refs/heads/main") { target { oid ... on Commit { committedDate } } compare(headRef: "next") { aheadBy } } } }"""


def gh_main() -> dict:
    """main's head, its commit time and how far `next` is ahead, in one small GraphQL read (REST's compare of main...next
    carries every commit and patch: megabytes while a few hundred commits are queued)."""
    owner, name = REPO.split("/")
    req = urllib.request.Request("https://api.github.com/graphql", method="POST",
                                 data=json.dumps({"query": MAIN_QUERY, "variables": {"owner": owner, "name": name}}).encode(),
                                 headers={"Authorization": f"Bearer {os.environ['GH_TOKEN']}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    if out.get("errors"):
        raise ValueError(f"GitHub GraphQL: {out['errors'][0].get('message')}")
    ref = out["data"]["repository"]["ref"]
    return {"sha": ref["target"]["oid"], "committed": ref["target"]["committedDate"], "ahead": ref["compare"]["aheadBy"]}


def main_stalled(state, ring, now, read: Callable[[], dict] = gh_main) -> None:
    """main's head unchanged for STALL_S. A landed train's commit carries the time drive built it on `next`, not when main moved
    to it, so the clock starts at the later of that and the first pass that saw the head."""
    head = read()
    committed = datetime.fromisoformat(head["committed"].replace("Z", "+00:00")).timestamp()
    seen = state.get("main_head") or {}
    if seen.get("sha") != head["sha"]:
        seen = state["main_head"] = {"sha": head["sha"], "seen": now}
    at = max(committed, seen["seen"])
    if now - at >= STALL_S and head["ahead"] > 0:
        once_each(state, f"main:{head['sha']}", ring, "ci",
                  f"ci: main hasn't moved for {(now - at) // 60:.0f} min (at {head['sha'][:10]} since {iso(at)}) while "
                  f"next is {head['ahead']} commits ahead of it: check drive (`research queue status`).", None, now)


def h200s(state, ring, now, samples: Callable[[float, float], list[dict]]) -> None:
    day = iso(now)[:10]
    if time.gmtime(now).tm_hour < REPORT_HOUR_UTC or state.get("h200_day") == day:
        return
    by: dict[str, list[int]] = {}
    for r in samples(now - 86400, now):
        if "h200" in str(r.get("platform", "")).lower() and r.get("read"):
            by.setdefault(r["name"], []).extend(r.get("gpu_util") or [])
    low = {n: sum(u) / len(u) for n, u in sorted(by.items()) if u and sum(u) / len(u) < H200_BUSY_PCT}
    if low:
        once_each(state, f"h200:{day}", ring, "top",
                  f"top: H200s under {H200_BUSY_PCT}% mean GPU utilization over the 24 h to {iso(now)}: "
                  + "; ".join(f"{n} {m:.1f}% ({len(by[n])} GPU samples)" for n, m in low.items())
                  + " (fleet.py's samples).", None, now)
    state["h200_day"] = day


def _change(n: int, then) -> str:
    return f" ({n - then:+d})" if isinstance(then, int) else ""


def linear_audit(state, ring, now, audit: Callable[[], list[dict]], volume: Callable[[], dict], log_path: Path | None,
                 unread: Callable[[], list[str]] = list) -> None:
    """The audit's line appended to `log_path` (None: a dry run, which appends nothing), each lane's ring and @top's daily one.
    A volume that can't be read leaves the line's `volume` null; `unread`, the teams the audit can't see, goes in both."""
    found, blind = audit(), unread()
    counts = linear.tally(found)
    kinds = {k: sum(c.get(k, 0) for c in counts.values()) for k in linear.AUDIT_KINDS}
    try:
        totals = volume()["totals"]
        vol = {k: totals[k] for k in ("posts", "words", "over")}
    except (RuntimeError, ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as e:
        log(f"linear: the volume not read: {type(e).__name__}: {e}")
        vol = None
    earlier = None
    if log_path is not None:
        for raw in log_path.read_text().splitlines() if log_path.is_file() else []:
            try:
                x = json.loads(raw)
            except ValueError as e:
                log(f"linear: skipped a log line that isn't JSON ({e}): {raw[:80]!r}")
                continue
            if str(x.get("t", "")) <= iso(now - DAY_BEFORE_S):
                earlier = x
        with log_path.open("a") as f:
            f.write(json.dumps({"t": iso(now), "findings": len(found), "kinds": kinds, "lanes": counts, "volume": vol,
                                "unread": blind}, sort_keys=True) + "\n")
    bucket = int(now // LANE_RING_S)
    for lane, c in counts.items():
        name, whose = ("top", "issues with no lane") if lane == linear.NO_LANE else (lane, "your open issues")
        once_each(state, f"linear:{lane}:{bucket}", ring, name,
                  f"{name}: the Linear audit finds {sum(c.values())} on {whose} ({', '.join(f'{k} {n}' for k, n in c.items())}): "
                  f"`research msg linear audit --lane {lane}` lists them, one line each with its fix.", None, now)
    if time.gmtime(now).tm_hour < REPORT_HOUR_UTC:
        return

    def then(*keys: str):
        x = earlier
        for k in keys:
            x = x.get(k) if isinstance(x, dict) else None
        return x
    shown = [f"{k} {n}{_change(n, then('kinds', k))}" for k, n in kinds.items() if n or then("kinds", k)]
    body = f"top: the Linear audit at {iso(now)}: {len(found)} findings{_change(len(found), then('findings'))}"
    body += f" ({', '.join(shown)})" if shown else ""
    if vol:
        body += f"; #{slack.FRONT_DOOR}'s last 24 h: " + ", ".join(
            f"{vol[k]} {what}{_change(vol[k], then('volume', k))}"
            for k, what in (("posts", "posts"), ("words", "words"), ("over", f"over {slack.VOLUME_WORDS} words")))
    body += f"; changes from {earlier['t']}" if earlier else "; no line 24 h earlier yet"
    if blind:
        body += f"; not audited, the app can't read team(s) {', '.join(blind)}"
    if counts:
        body += "; by lane: " + ", ".join(
            f"{lane} {sum(c.values())}" for lane, c in sorted(counts.items(), key=lambda kv: -sum(kv[1].values())))
    once_each(state, f"linear:day:{iso(now)[:10]}", ring, "top", body, None, now)


def sweep_pass(state: dict, ring: Ringer | None, checks: dict[str, Callable[[], None]]) -> None:
    for name, check in checks.items():
        try:
            check()
        except Exception as e:  # noqa: BLE001 (one check failing leaves the others)
            log(f"{name} failed: {type(e).__name__}: {e}")


def fleet_samples(start: float, end: float) -> list[dict]:
    p = Path(slack.__file__).resolve().parent / "pods" / "nebius" / "fleet.py"
    spec = importlib.util.spec_from_file_location("fleet", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.load_samples(start, end)


def traffic_json() -> dict:
    r = subprocess.run([sys.executable, "-m", "research", "msg", "traffic", "--stale-h", str(UNANSWERED_H), "--hours", "24",
                        "--json"], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"research msg traffic exited {r.returncode}: {r.stderr.strip()[-300:]}")
    return json.loads(r.stdout)


def linear_check(state, ring, reg, now, log_path: Path | None) -> None:
    lin, owners = linear.Linear(), linear.owners(reg)
    linear_audit(state, ring, now, lambda: lin.audit(owners, now=now), volume_json, log_path, lambda: lin.unread(sorted(owners)))


def volume_json() -> dict:
    r = subprocess.run([sys.executable, "-m", "research", "msg", "traffic", "--volume", "--hours", "24", "--json"],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"research msg traffic --volume exited {r.returncode}: {r.stderr.strip()[-300:]}")
    return json.loads(r.stdout)


def cursor_agents() -> list[dict]:
    from verity_ops.agent_watch import Api, verity

    return [a for a in Api(os.environ["CURSOR_API_KEY"]).agents() if verity(a)]


def prune(state: dict, now: float) -> None:
    state["rung"] = {k: t for k, t in (state.get("rung") or {}).items() if now - t < KEEP_S}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="top_watch.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("asks", "sweep"))
    ap.add_argument("--every", type=float)
    ap.add_argument("--state", type=Path)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--seed", action="store_true", help="one pass that records what it finds and rings nobody")
    ap.add_argument("--dry-run", action="store_true")
    ns = ap.parse_args(argv)
    every = ns.every or (900 if ns.cmd == "asks" else 3600)
    state_path = ns.state or Path(f"/data/watchers/top-{ns.cmd}.json")
    reg = slack.load_registry(slack.registry_path())
    api = slack.Slack(os.environ.get("SLACK_BOT_TOKEN"))      # reads even in a dry run: Ringer is what posts
    while True:
        now = time.time()
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        ring = None if ns.seed else Ringer(api, reg, "asks-relay" if ns.cmd == "asks" else "sweep", ns.dry_run)
        try:
            if ns.cmd == "asks":
                asks_pass(state, slack.Notion(), ring, slack.asks_db(reg)["data_source"], now, ns.dry_run)
            else:
                routes = ring.routes if ring else slack.Routes()
                sweep_pass(state, ring, {
                    "traffic": lambda: unanswered(state, ring, reg, now, traffic_json),
                    "agents": lambda: holderless(state, ring, now, cursor_agents, known_holders(reg, routes)),
                    "rules thread": lambda: rules_thread(state, ring, api, reg, now),
                    "main": lambda: main_stalled(state, ring, now),
                    "h200": lambda: h200s(state, ring, now, fleet_samples),
                    "linear": lambda: linear_check(state, ring, reg, now, None if ns.dry_run else state_path.parent / AUDIT_LOG)})
                prune(state, now)
            if not ns.dry_run:
                tmp = state_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
                tmp.replace(state_path)
            log(f"{ns.cmd} pass done" + (" (seeded: nothing rung)" if ns.seed else ""))
        except (slack.SlackError, OSError, ValueError, KeyError) as e:   # the next pass tries again from the saved state
            log(f"{ns.cmd} pass failed: {type(e).__name__}: {e}")
        if ns.once or ns.seed:
            return 0
        time.sleep(max(1.0, every - (time.time() - now)))


if __name__ == "__main__":
    sys.exit(main())
