"""D6 v1.1's build (BASE.pkl) against the same build on V4-Pro's checkpoint shapes (CK.pkl): V4's rows and picks, and
every numeric field of the other models' rows that moved (expected: none).

    python d6_v4_rows.py BASE.pkl CK.pkl [OUT.json]
"""
import json
import pickle
import sys
from pathlib import Path

A, B = (pickle.loads(Path(p).read_bytes()) for p in sys.argv[1:3])
V4, HEAD = "DeepSeek-V4-Pro", "v3.1"


def flat(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, f"{p}/{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from flat(v, f"{p}/{i}")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield p, float(x)


def get(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


FIELDS = {"coverage": lambda r: 1 - r["world"]["uncredited"], "gamma": lambda r: get(r, "world", "gamma"),
          "gamma_add": lambda r: get(r, "world", "gamma_add"), "share": lambda r: get(r, "share", "share"),
          "eps head 1e-6": lambda r: get(r, "eps", "head", 1e-6), "eps_net": lambda r: get(r, "eps", "at", "served", "eps_net"),
          "eps no C_reg 1e-6": lambda r: get(r, "eps", "no_c_reg", 1e-6), "leak_served": lambda r: get(r, "period", "leak_served"),
          "1 - c_inc": lambda r: 1 - get(r, "period", "c_inc"), "cred_macs": lambda r: get(r, "period", "cred_macs"),
          "g": lambda r: get(r, "eps", "g")}
PCT = {"coverage", "share", "eps head 1e-6", "eps_net", "eps no C_reg 1e-6", "leak_served", "1 - c_inc"}


def fmt(name, v):
    if v is None:
        return "-"
    return f"{v:.4%}" if name in PCT else (f"{v:.4e}" if name == "cred_macs" else f"{v:.4f}")


out = {"tau": [get(A, "rule", "tau"), get(B, "rule", "tau")], "rows": {}, "beside": {}, "picks": {}, "other_moved": []}
print("τ:", out["tau"][0], "→", out["tau"][1])
for k in sorted(set(A["rows"]) & set(B["rows"]), key=str):
    ra, rb = A["rows"][k], B["rows"][k]
    if k[3] != V4:
        fa, fb = dict(flat(ra)), dict(flat(rb))
        out["other_moved"] += [[" | ".join(k), p, fa[p], fb.get(p)] for p in fa
                               if fb.get(p) is None or abs(fa[p] - fb[p]) > 1e-12 * max(1, abs(fa[p]))]
        continue
    row = {n: [f(ra), f(rb)] for n, f in FIELDS.items()}
    out["rows"][" | ".join(k)] = row
    if k[0] in (HEAD, "v3", "dual-safe") and k[2] == "measured":
        print(f"{k[0]:9} {k[1]:13} " + "; ".join(f"{n} {fmt(n, a)} → {fmt(n, b)}" for n, (a, b) in row.items()))
print(f"other models' fields that moved: {len(out['other_moved'])}")
for k in sorted(set(A["beside"]) & set(B["beside"]), key=str):
    if k[2] == V4 and k[0] == HEAD:
        a, b = A["beside"][k], B["beside"][k]
        out["beside"][" | ".join(map(str, k))] = {x: [a[x], b[x]] for x in ("share", "leak", "c_inc")}
        print(f"beside {k[1]:13} x {k[3]:<5} share {a['share']:.4%} → {b['share']:.4%}; leak {a['leak']:.4%} → "
              f"{b['leak']:.4%}; 1 − c_inc {1 - a['c_inc']:.4%} → {1 - b['c_inc']:.4%}")
for design in B["picks"]:
    pa, pb_ = A["picks"].get(design, {}), B["picks"][design]
    for k, p in pb_.items():
        q = pa.get(k)
        if p != q:
            out["picks"][f"{design} | {' | '.join(map(str, k))}"] = [q, p]
moved = {k: v for k, v in out["picks"].items() if v[0] is not None}
print(f"picks: {len(out['picks']) - len(moved)} new round classes, {len(moved)} existing ones moved")
for k, (q, p) in sorted(out["picks"].items()):
    if k.startswith(HEAD + " |") and p:
        print(f"  {k}: {'new' if q is None else (q['t'], q['Lp'])} → t {p['t']} L′ {p['Lp']} γ {p['gamma']:.3f} "
              f"leak {p['leak'] if p['leak'] is None else round(p['leak'], 4)}")
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1, default=str))
