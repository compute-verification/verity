"""comment.py: the table of CIR-259's Part 1 comment, from /opt/cursor/artifacts/cir259/table.json (table.py's), one row per
replacement in ROWS' order. Prints Markdown; a tag table.json does not hold prints as a row marked MISSING."""
import json
from pathlib import Path

T = {(r["run"][-4:], r["tag"]): r for r in json.loads(Path("/opt/cursor/artifacts/cir259/table.json").read_text())}

# PARTS type, run's last four, tag, the stager's spec as given (None: census:NAME of the staged Definition)
ROWS = [
    ("moe-gemmw", "d4e3", "gemmw", None),
    ("moe-w13", "8aa4", "w13a", None),
    ("moe-w13, the other variant", "8aa4", "w13b", None),
    ("moe-chunk", "d4e3", "chunk", None),
    ("dsa-logits", "11a8", "logits3", None),
    ("moe-sum", "11a8", "moesum1", None),
    ("silu-mul", "11a8", "silu2", None),
    ("rope", "11a8", "rope2", None),
    ("router", "1a08", "router-row", "router-row"),
    ("router", "1a08", "router-slots", "router-slots"),
    ("dsa-q", "9001", "dsa-q-row", "dsa-q-row"),
    ("rmsnorm-cuda", None, "rmsnorm-v8", "rmsnorm-v8"),
    ("dsa-krow", "8aa4", "krow", None),
    ("dsa-topk", "11a8", "topk", None),
    ("sparse-mla-head", None, "head64", None),
    ("sparse-mla-head", "8aa4", "headend", None),
    ("token-select", None, "greedychunk", None),
    ("token-select", None, "greedycombine", None),
    ("token-select", None, "tokensel", None),
    ("s8-linear", "11a8", "s8", "s8-coord"),
]
NOT = [  # staged before the 13:07Z answer; not replacements
    ("moe-w13's inner coordinate", "8aa4", "w13c"),
    ("moe-w2's inner coordinate", "8aa4", "w2c"),
    ("moe-w2 as the card's whole Definition", "8aa4", "w2g"),
    ("dsa-logits' inner coordinate", "11a8", "logit1"),
]


def secs(wall: str) -> str:
    p = [float(x) for x in wall.split(":")]
    s = p[-1] + 60 * p[-2] + (3600 * p[-3] if len(p) > 2 else 0)
    return f"{s:.1f} s" if s < 600 else f"{s / 60:.1f} min"


def find(run, tag):
    if run:
        return T.get((run, tag))
    hits = [r for (_, t), r in T.items() if t == tag]
    return hits[-1] if hits else None


def lean(r) -> str:
    L = r.get("lean") or {}
    say = lambda k: (L.get(k) or {}).get("says")
    out = []
    held = say("B.statement.held")
    out.append('"units derived"' if held == "ok, units derived" else f"`statement`: {held}")
    ks = say("B.ks-form")
    if ks != "KS-FORM ok":
        out.append(f"`ks-form`: {ks}")
    v = say("B.verify.held")
    if v is None:
        out.append("no session run")
    elif v == "accepted":
        out.append("`verify --zk` accepted")
    else:
        out.append(f"`verify --zk`: {v}")
    od, vd = say("B.statement.other-def"), say("B.verify.other-def")
    prog = "the verifier's program is not the one the partition object names"
    if od and prog in od and (vd is None or prog in vd):
        out.append("refused with another Definition's Program" + (" (at `statement`)" if vd is None else ""))
    else:
        out.append(f"another Program: `statement` {od}; `verify --zk` {vd}")
    np = say("B.verify.no-program")
    if np is not None:
        out.append("and with none" if "a session without the verifier's program" in np else f"no Program: {np}")
    return "; ".join(out)


def row(part, run, tag, spec):
    r = find(run, tag)
    if r is None:
        return f"| `{part}` | {tag} | MISSING | | | | |"
    if not r.get("staged"):
        return f"| `{part}` | `{spec or tag}` | | not staged | {r.get('stage_wall')} at {r.get('stage_peak_gb')} GB | {r.get('refusal')} | [run: {r['run']}] |"
    name = f"`{spec}` (`{r['definition']}`)" if spec else f"`census:{r['definition']}`"
    peak = max((v.get("peak_gb") or 0) for v in (r.get("lean") or {"x": {}}).values())
    return (f"| `{part}` | {name} | {r['n']} | {r['ands']:,}, {r['k_log']} | {secs(r['stage_wall'])} at {r['stage_peak_gb']:.1f} GB"
            f" | {lean(r)}; Lean's peak {peak:.1f} GB | [run: {r['run']}] |")


print("| `PARTS` type | staged with `--bind` as | N | ANDs, `k_log` | staging | Lean verifier, with its partition object and Program | |")
print("|---|---|---|---|---|---|---|")
for x in ROWS:
    print(row(*x))
print()
for part, run, tag in NOT:
    r = find(run, tag)
    print(f"- {part}: `{r['definition']}` N={r['n']}, {r['ands']:,} ANDs, `k_log` {r['k_log']}; {lean(r)} [run: {r['run']}]")
print()
for (run, tag), r in sorted(T.items()):
    p, s = r.get("prove") or {}, r.get("serve") or {}
    if isinstance(p, dict) and p.get("maxrss_kb"):
        print(f"{run} {tag}: prover {p['wall_s']} s at {p['maxrss_kb'] / 2**20:.1f} GB; serve {s.get('wall_s')} s at "
              f"{(s.get('maxrss_kb') or 0) / 2**20:.1f} GB; fail={r.get('session_failed')}")
