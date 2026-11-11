#!/usr/bin/env python3
"""Full text of a thread's replies since MIN minutes ago: `thread_full.py THREAD_TS [MIN=30] [CHANNEL=C0C5RCXL66N]`."""
import json, os, sys, time, urllib.parse, urllib.request

try:  # a long-lived shell loses the VM's injected secrets; research takes them from the tmux server
    from research.secret_hint import recover
    recover()
except ImportError:
    pass

def api(method, **p):
    req = urllib.request.Request("https://slack.com/api/" + method + "?" + urllib.parse.urlencode(p),
                                 headers={"Authorization": "Bearer " + os.environ["SLACK_BOT_TOKEN"]})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

ts = sys.argv[1]
since = time.time() - 60 * float(sys.argv[2] if len(sys.argv) > 2 else 30)
ch = sys.argv[3] if len(sys.argv) > 3 else "C0C5RCXL66N"
d = api("conversations.replies", channel=ch, ts=ts, oldest=f"{since:.6f}", inclusive="true", limit=200)
for m in d.get("messages", []):
    if float(m["ts"]) > since:
        print("=====", time.strftime("%H:%MZ", time.gmtime(float(m["ts"]))), m["ts"])
        print(m.get("text") or "")
