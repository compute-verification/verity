import json, os, sys, time, urllib.parse, urllib.request

TOKEN = os.environ["SLACK_BOT_TOKEN"]


def call(method, **params):
    url = f"https://slack.com/api/{method}?" + urllib.parse.urlencode(params)
    for attempt in range(6):
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.load(r)
        except Exception as e:  # noqa: BLE001
            time.sleep(2 ** attempt)
            continue
        if d.get("ok"):
            return d
        if d.get("error") == "ratelimited":
            time.sleep(5)
            continue
        raise SystemExit(f"{method}: {d.get('error')}")
    raise SystemExit(f"{method}: failed")


def replies(channel, ts, oldest):
    out, cursor = [], None
    while True:
        p = dict(channel=channel, ts=ts, limit=200, oldest=oldest)
        if cursor:
            p["cursor"] = cursor
        d = call("conversations.replies", **p)
        out += d["messages"]
        cursor = d.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            return out


def history(channel, oldest):
    out, cursor = [], None
    while True:
        p = dict(channel=channel, limit=200, oldest=oldest)
        if cursor:
            p["cursor"] = cursor
        d = call("conversations.history", **p)
        out += d["messages"]
        cursor = d.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            return out


if __name__ == "__main__":
    mode, channel, *rest = sys.argv[1:]
    oldest = "1792627200"  # 2026-10-22T00:00Z
    if mode == "thread":
        msgs = replies(channel, rest[0], oldest)
        json.dump(msgs, open(rest[1], "w"))
    else:
        msgs = history(channel, oldest)
        json.dump(msgs, open(rest[0], "w"))
    print(len(msgs))
