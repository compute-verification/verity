import json, re, datetime as dt, collections

m = json.load(open("/tmp/reds/thread.json"))
RED = re.compile(r"(suite|step) (\S+) is red on `(main|checked)` ([0-9a-f]{12}) \(run ([^)]*)\): (.*)", re.S)
CATS = [
    ("undeclared-input", r"read outside its declared inputs|undeclared read|declared inputs: "),
    ("lint-ratchet", r"lane text|retired term|silent except|direct environment reads|P8 \(|p08_facts|test_no_new_|lint/|allowlist|test_repo_"),
    ("lock-or-records", r"TRUSTED\.sha256|stale|records? (is|are) stale|lean-audit.json|regenerat|test_trusted_hashes"),
    ("timeout-cap", r"timed out|timeout|Timeout|past its cap|1 h cap"),
    ("machine", r"rc=137|No space left|disk full|MemoryError|Killed|OOM|out of memory"),
    ("no-error-line", r"its log has no error line"),
]


def cat(reason):
    for name, rx in CATS:
        if re.search(rx, reason):
            return name
    return "test-failure"


seen = {}
for x in sorted(m, key=lambda x: float(x["ts"])):
    t = re.sub(r"<@[A-Z0-9]+>|<!subteam\^[A-Z0-9]+(\|[^>]*)?>", "@", x.get("text", "")).replace("\n", " ")
    t = re.sub(r"^(@ )*", "", t)
    if not t.startswith("queue: "):
        continue
    r = RED.search(t)
    if not r:
        continue
    kind, name, br, tip, runs, reason = r.groups()
    when = dt.datetime.fromtimestamp(float(x["ts"]), dt.timezone.utc)
    key = (name, tip)
    culprit = re.search(r"Culprit: (#\d+)|(#\d+) is its culprit|(#\d+) must be reverted", t)
    if key not in seen:
        seen[key] = dict(first=when.isoformat()[:16], branch=br, kind=kind, name=name, tip=tip, runs=runs.split()[:3],
                         cat=cat(reason), reason=reason[:300],
                         culprit=next((g for g in (culprit.groups() if culprit else ()) if g), None))
rows = list(seen.values())
json.dump(rows, open("/tmp/reds/reds.json", "w"), indent=1)
print(len(rows), "distinct (suite|step, tip) reds")
c = collections.Counter((r["first"][:10] >= "2026-10-29", r["cat"]) for r in rows)
for k, v in sorted(c.items()):
    print(("checked-era " if k[0] else "main-era    ") + k[1], v)
tips = collections.defaultdict(list)
for r in rows:
    tips[(r["branch"], r["tip"])].append(r)
print(len(tips), "distinct red tips")
