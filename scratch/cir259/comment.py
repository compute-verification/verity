"""comment.py: the table of CIR-259's Part 1 comment, from /opt/cursor/artifacts/cir259/table.json (table.py's), one row per
replacement in ROWS' order. Prints Markdown; a tag table.json does not hold prints as a row marked MISSING."""
import json
from pathlib import Path

T = {(r["run"][-4:], r["tag"]): r for r in json.loads(Path("/opt/cursor/artifacts/cir259/table.json").read_text())}

# PARTS type, run's last four, tag, the stager's spec as given (None: census:NAME of the staged Definition)
ROWS = [
    ("moe-gemmw", "d4e3", "gemmw", None),
    ("moe-w13", "8aa4", "w13a", None),
    ("moe-w13, the decode variant", "8aa4", "w13b", None),
    ("moe-chunk", "d4e3", "chunk", None),
    ("dsa-logits", "11a8", "logits3", None),
    ("moe-sum", "11a8", "moesum1", None),
    ("silu-mul", "11a8", "silu2", None),
    ("rope", "11a8", "rope2", None),
    ("router", "1a08", "router-row", "router-row"),
    ("router", "1a08", "router-slots", "router-slots"),
    ("dsa-q", "9dd0", "dsa-q-row", "dsa-q-row"),
    ("rmsnorm-cuda", "9dd0", "rmsnorm-v8", "rmsnorm-v8"),
    ("dsa-krow", "8aa4", "krow", None),
    ("dsa-topk", "11a8", "topk", None),
    ("sparse-mla-head", "9dd0", "head64", None),
    ("sparse-mla-head", "75c2", "head1b", None),
    ("sparse-mla-head", "8aa4", "headend", None),
    ("token-select", "1147", "greedychunk", None),
    ("token-select", "1147", "greedycombine", None),
    ("token-select", "9dd0", "tokensel", None),
    ("s8-linear", "11a8", "s8", "s8-coord"),
]
EXTRA = [  # staged beside the replacements: the census's own names where the replacement is a Boolean version, and a small head block
    ("router, the census's name", "a739", "router-word", None),
    ("dsa-q, the census's name", "a739", "dsa-q-word", None),
    ("sparse-mla-head, one key, the census's name", "75c2", "head1w", None),
]
NOT = [  # staged before the 13:07Z answer; not replacements
    ("`moe-w13`'s inner coordinate", "8aa4", "w13c"),
    ("`moe-w2`'s inner coordinate", "8aa4", "w2c"),
    ("`moe-w2` as the card's whole Definition", "8aa4", "w2g"),
    ("`dsa-logits`' inner coordinate", "11a8", "logit1"),
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
    prog = "the verifier's program is not the one the partition object names"
    if not L:
        b = r["circuit_bytes"]
        return (f"Lean not run: its circuit.txt is {b / 2**30:.1f} GB, and my job held a check to six times that, "
                f"{b * 6 // 2**30 + 4} GB, over its memory")
    held, ks, v = say("B.statement.held"), say("B.ks-form"), say("B.verify.held")
    od, vd, np = say("B.statement.other-def"), say("B.verify.other-def"), say("B.verify.no-program")
    if (held == "ok, units derived" and ks == "KS-FORM ok" and v == "accepted" and od and prog in od and vd and prog in vd
            and np and "a session without the verifier's program" in np):
        return "all four"
    out = ['"units derived"' if held == "ok, units derived" else f"`statement`: {held}"]
    out.append("`ks-form` ok" if ks == "KS-FORM ok" else f'`ks-form`: "{ks}"')
    out.append("no session run" if v is None else "`verify --zk` accepted" if v == "accepted" else f"`verify --zk`: {v}")
    if od and prog in od and (vd is None or prog in vd):
        out.append("refused with another Definition's Program" + (" at `statement`" if vd is None else ""))
    else:
        out.append(f"another Program: `statement` {od}; `verify --zk` {vd}")
    if np is not None:
        out.append("and with none" if "a session without the verifier's program" in np else f"no Program: {np}")
    return "; ".join(out)


def row(part, run, tag, spec):
    r = find(run, tag)
    if r is None:
        return f"| `{part}` | {tag} | MISSING | | | |"
    if not r.get("staged"):
        return f"| {part} | `{spec or tag}` | 1 | | not staged | {r.get('refusal')} [run: {r['run']}] |"
    name = f"`{spec}`, which is `{r['definition']}`" if spec else f"`census:{r['definition']}`"
    peak = max((v.get("peak_gb") or 0) for v in (r.get("lean") or {"x": {}}).values())
    part = f"`{part}`" if " " not in part else part
    most = f"; Lean at most {peak:.1f} GB" if r.get("lean") else ""
    return (f"| {part} | {name} | {r['n']} | {r['ands']:,}, {r['k_log']} | {secs(r['stage_wall'])} at {r['stage_peak_gb']:.1f} GB"
            f" | {lean(r)}{most} [run: {r['run']}] |")


print("| `PARTS` type | staged with `--bind` as | N | ANDs, `k_log` | staging | Lean verifier, with its partition object and Program |")
print("|---|---|---|---|---|---|")
for x in ROWS:
    print(row(*x))
print()
for x in EXTRA:
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
