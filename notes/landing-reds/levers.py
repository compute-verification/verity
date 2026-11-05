"""Lever estimates from the incident table and the stall intervals (art:ca7e94f6...).

Method. A stall's PR-hours per hour is its PR-hours over its length (the mean number of PRs on `checked` during it).
- Revert at once (L2): each code-culprit red on a stall's critical path costs its fix wait; reverting instead costs
  REVERT_H (the ring, then the next pass pushes the revert). Saved stall hours = sum(max(0, fix_wait - REVERT_H)).
- Check before landing (L1): a code-culprit red caught on `next` never lands, so its stall wave goes: its fix wait and
  the extra after-landing cycle it forced (CYCLE_H: verdict gathering, shards, baseline). Saved = fix_wait + CYCLE_H per
  caught red wave, capped at the stall's excess over one cycle.
Open incidents (R22, R24) count their wait to NOW."""
import json, datetime as dt

REVERT_H, CYCLE_H, NOW = 1.0, 2.5, dt.datetime(2026, 11, 5, 18, 0, tzinfo=dt.timezone.utc)
CODE = {"lint-ratchet", "semantic", "lock-or-records", "undeclared-input+lint+lock", "lint+semantic+machine",
        "semantic+undeclared-input", "semantic+lint", "flake+semantic", "rename-or-path", "semantic+lint+test-of-tooling",
        "semantic+test-of-tooling"}
BEFORE = {"R01", "R03", "R05", "R07", "R09", "R10", "R11", "R12", "R13", "R17", "R18", "R21", "R22", "R23", "R24"}


def ts(s):
    return dt.datetime.strptime("2026-" + s.replace("~", ""), "%Y-%m-%d %H:%M").replace(tzinfo=dt.timezone.utc)


iv = json.load(open("/tmp/reds/intervals.json"))
inc = json.load(open("/tmp/reds/incidents.json"))
rate = {}
for i in iv:
    key = f"{i['start'][5:10]} {i['start'][11:16]}"
    rate[key] = (i["pr_h"] / i["h"] if i["h"] else 0, i["h"])
out = {"L1": [0.0, 0.0], "L2": [0.0, 0.0]}
per = {}
for r in inc:
    if r["category"] not in CODE:
        continue
    w = r["fix_wait_h"]
    if w == "":
        continue                                  # wait unknown (R01, R03): counted as saving nothing
    if w is None:
        w = (NOW - ts(r["named"])).total_seconds() / 3600
    k = r["stall"].split("->")[0]
    prh, length = rate.get(k, (0, 0))
    s2 = max(0.0, w - REVERT_H)
    s1 = (w + CYCLE_H) if r["id"] in BEFORE else 0.0
    per.setdefault(k, {"len": length, "prh": prh, "L1": 0.0, "L2": 0.0})
    per[k]["L1"] += s1
    per[k]["L2"] += s2
for k, v in per.items():
    cap = max(0.0, v["len"] - CYCLE_H)
    v["L1"], v["L2"] = min(v["L1"], cap), min(v["L2"], cap)
    for L in ("L1", "L2"):
        out[L][0] += v[L]
        out[L][1] += v[L] * v["prh"]
tot = sum(i["pr_h"] for i in iv)
for L, (h, ph) in out.items():
    print(f"{L}: main-stall hours saved {h:.1f} per 7 days; PR-hours saved {ph:,.0f} of {tot:,} ({100 * ph / tot:.0f}%)")
for k, v in sorted(per.items()):
    print(f"  stall from {k}: {v['len']:.1f} h, {v['prh']:.0f} PRs mean; L1 {v['L1']:.1f} h, L2 {v['L2']:.1f} h")
