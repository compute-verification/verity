"""Every count in CIR-271's cover comments, from cover.json's rows and blocks lists alone: per column, templates and instances
in each state; per block-type column, block types and the card's blocks in each state (and that `block_totals` agrees)."""
import json, sys
from collections import Counter
c = json.load(open(sys.argv[1] if len(sys.argv) > 1 else __file__.replace("cover_check.py", "cover.json")))
rows = [r for r in c["rows"] if r["in_list"]]
state = {"pin": lambda r: r["pin"], "cut": lambda r: "cut" if r["cut"] else "none", "accepted": lambda r: r["accepted"],
         "staged": lambda r: "none" if r["staged"] is None else ("census" if r["staged"].startswith("census:") else "table")}
counts = {c_: {s: [sum(1 for r in rows if f(r) == s), sum(r["instances"] for r in rows if f(r) == s)] for s in sorted(Counter(map(f, rows)))} for c_, f in state.items()}
uncovered = sorted(([r["template"], r["instances"]] for r in rows if r["accepted"] not in ("yes", "boolean-version")), key=lambda x: (-x[1], x[0]))
out = {"templates": len(rows), "instances": sum(r["instances"] for r in rows), "counts": counts,
       "uncovered": [len(uncovered), sum(n for _, n in uncovered)]}
if "blocks" in c:
    on = [b for b in c["blocks"] if b["on_card"]]
    bstate = {"accepted": lambda b: b["accepted"], "form": lambda b: b["form"] or "none",
              "accepted_bound": lambda b: b["accepted"] if b["form"] in ("bound", "audit") else "none",
              "staging_run_completed": lambda b: "yes" if b["staging_run"] else "no",
              "staged_name": lambda b: "yes" if b["staged"] else "no", "pinned": lambda b: b["pinned"],
              "cut_by_list": lambda b: "yes" if b["list_templates"] else "no"}
    bc = {c_: {s: [sum(1 for b in on if f(b) == s), sum(b["blocks"] for b in on if f(b) == s)] for s in sorted(Counter(map(f, on)))} for c_, f in bstate.items()}
    for col, st in c["block_totals"].items():
        if isinstance(st, dict):
            assert all(bc[col].get(s, [0, 0]) == v for s, v in st.items()), (col, st, bc[col])
    big = [[b["block"], b["blocks"]] for b in on if b["accepted"] == "none" and b["blocks"] > 3_000_000]
    out["blocks"] = {"types": len(on), "blocks": sum(b["blocks"] for b in on), "prefill": sum(b["prefill"] for b in on),
                     "decode": sum(b["decode"] for b in on), "counts": bc, "extra_units": len(c["blocks"]) - len(on),
                     "unaccepted_over_3m": [len(big), sum(n for _, n in big)]}
if "bound_totals" in c:
    on = [b for b in c["blocks"] if b["on_card"]]
    bt = {s: [sum(1 for b in on if b["bound_outcome"] == s), sum(b["blocks"] for b in on if b["bound_outcome"] == s)]
          for s in sorted(Counter(b["bound_outcome"] for b in on))}
    assert bt == c["bound_totals"], (bt, c["bound_totals"])
    ty = c["bound_runs"]["types"]
    for t, r in ty.items():
        v = r.get("verdicts") or {}
        h, np_, od = v.get("B.verify.held"), v.get("B.verify.no-program"), v.get("B.verify.other-def")
        held = bool(h and h["exit"] == 0 and h["accepted"] is True)
        ctl = held and all(x and x["exit"] == 1 and x["accepted"] is False for x in (np_, od))
        assert (r["outcome"] == "accepted") == ctl and (r["outcome"] == "held-control-off") == (held and not ctl), t
    for b in on:
        r = ty.get(b["cir272"]["tag"]) if b.get("cir272") else None
        assert (b["form"] == "bound") == (r is None), b["block"]
    total = sum(b["blocks"] for b in on)
    grp = lambda s: "accepted" if s.startswith("accepted") else "not-run" if s.startswith("not-run") else s  # noqa: E731
    shares = {}
    for s, (n, k) in bt.items():
        g = shares.setdefault(grp(s), [0, 0])
        g[0] += n
        g[1] += k
    out["bound"] = {"types": len(on), "blocks": total, "by_outcome": bt,
                    "shares": {g: [n, k, round(100 * k / total, 4)] for g, (n, k) in shares.items()},
                    "tags": dict(Counter(f"{r['outcome']}{'/' + r['why'] if r.get('why') else ''}" for r in ty.values())),
                    "extras": dict(Counter(f"{r['outcome']}{'/' + r['why'] if r.get('why') else ''}" for r in ty.values() if r["extra"]))}
print(json.dumps(out, indent=1))
