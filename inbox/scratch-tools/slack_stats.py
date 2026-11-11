#!/usr/bin/env python3
"""Slack traffic over the last HOURS: per sender handle, message count and word lengths; thread sizes; rings.
Usage: slack_stats.py OUT.json [HOURS=24] [CHANNEL...]. Needs SLACK_BOT_TOKEN (run in tmux)."""
import json, os, re, statistics as st, sys, time, urllib.parse, urllib.request, collections as C

try:  # a long-lived shell loses the VM's injected secrets; research takes them from the tmux server
    from research.secret_hint import recover
    recover()
except ImportError:
    pass

def api(method, **p):
    for _ in range(5):
        req = urllib.request.Request("https://slack.com/api/" + method + "?" + urllib.parse.urlencode(p),
                                     headers={"Authorization": "Bearer " + os.environ["SLACK_BOT_TOKEN"]})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)
        if d.get("error") == "ratelimited":
            time.sleep(20)
            continue
        return d
    return d

out = sys.argv[1]
hours = float(sys.argv[2]) if len(sys.argv) > 2 else 24
chans = sys.argv[3:] or ["C0C5RCXL66N", "C0C5UCA0S0Z", "C0CA1JZSYNR", "C0CAHBAKPL1"]
since = time.time() - 3600 * hours
SENDER = re.compile(r"^(?:\s*(?:<!subteam\^\w+(?:\|[^>]*)?>|<@\w+>|@[\w-]+)[ ,:]*)*\s*\*?([a-z][\w-]+)\*?\s+(?:\(\w[^)]*\)\s+)?\d{1,2}:\d{2}Z")
RING = re.compile(r"<!subteam\^(\w+)")
msgs = []
for ch in chans:
    cursor, roots = None, []
    while True:
        d = api("conversations.history", channel=ch, oldest=f"{since - 86400 * 4:.6f}", limit=200, **({"cursor": cursor} if cursor else {}))
        for m in d.get("messages", []):
            roots.append(m)
        cursor = (d.get("response_metadata") or {}).get("next_cursor")
        if not cursor:
            break
    for r in roots:
        if float(r["ts"]) >= since:
            msgs.append((ch, r["ts"], r["ts"], r.get("text") or "", r.get("username")))
        if r.get("reply_count") and float(r.get("latest_reply", "0")) >= since:
            cur = None
            while True:
                d = api("conversations.replies", channel=ch, ts=r["ts"], oldest=f"{since:.6f}", limit=200, **({"cursor": cur} if cur else {}))
                for m in d.get("messages", [])[0:]:
                    if m["ts"] != r["ts"] and float(m["ts"]) >= since:
                        msgs.append((ch, r["ts"], m["ts"], m.get("text") or "", m.get("username")))
                cur = (d.get("response_metadata") or {}).get("next_cursor")
                if not cur:
                    break
                time.sleep(1)
            time.sleep(1.1)
rows, copies = [], 0
for ch, root, ts, text, user in msgs:
    if user == "router":  # the router's copy of a message already counted in its own thread
        copies += 1
        continue
    m = SENDER.match(text)
    rows.append({"ch": ch, "root": root, "ts": ts, "sender": user or (m.group(1) if m else "?"), "words": len(text.split()),
                 "chars": len(text), "rings": RING.findall(text[:200]), "text": text[:300]})
json.dump(rows, open(out, "w"), indent=0)
by = C.defaultdict(list)
for r in rows:
    by[r["sender"]].append(r["words"])
print(f"{len(rows)} messages in {hours} h over {len(chans)} channels (router copies {copies} not counted); "
      f"threads {len({r['root'] for r in rows})}")
print("sender n median p90 max total_words")
for s, w in sorted(by.items(), key=lambda kv: -sum(kv[1])):
    w.sort()
    print(s, len(w), int(st.median(w)), w[int(0.9 * (len(w) - 1))], w[-1], sum(w))
th = C.Counter(r["root"] for r in rows)
print("biggest threads:", th.most_common(8))
