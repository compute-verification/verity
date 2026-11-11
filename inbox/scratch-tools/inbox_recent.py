#!/usr/bin/env python3
"""Replies in @top's inbox thread from the last N minutes (default 65): `inbox_recent.py [MIN] [--grep TEXT]`.

Slack's conversations.replies ignores an `oldest` with more than six decimals and returns only the parent, so times are
formatted with six (an hourly read on 24 Oct 01:00Z saw 0 replies of 6).
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

try:  # a long-lived shell loses the VM's injected secrets; research takes them from the tmux server
    from research.secret_hint import recover
    recover()
except ImportError:
    pass

INBOX = ("C0C5RCXL66N", "1790877429.351069")


def api(method, **p):
    req = urllib.request.Request("https://slack.com/api/" + method + "?" + urllib.parse.urlencode(p),
                                 headers={"Authorization": "Bearer " + os.environ["SLACK_BOT_TOKEN"]})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    grep = sys.argv[sys.argv.index("--grep") + 1] if "--grep" in sys.argv else None
    if grep in args:
        args.remove(grep)
    now = time.time()
    since = now - 60 * float(args[0] if args else 65)
    out, cursor = [], None
    while True:
        p = dict(channel=INBOX[0], ts=INBOX[1], oldest=f"{since:.6f}", latest=f"{now:.6f}", inclusive="true", limit=200)
        if cursor:
            p["cursor"] = cursor
        d = api("conversations.replies", **p)
        if not d.get("ok"):
            print("error", d.get("error"))
            return 1
        out += [m for m in d["messages"] if float(m["ts"]) > since]
        cursor = (d.get("response_metadata") or {}).get("next_cursor")
        if not cursor:
            break
    hits = [m for m in out if not grep or grep in (m.get("text") or "")]
    print(f"{len(out)} replies, {len(hits)} shown")
    for m in hits:
        print(time.strftime("%H:%MZ", time.gmtime(float(m["ts"]))), m["ts"], (m.get("text") or "")[:400].replace("\n", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
