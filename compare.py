"""Compare two Builds' Program descriptors (old = main's pins, new = #8047's): every Definition outside the fused norm's own
must be the same body once RMSNormFusedCudaVector8_v1 is read as _v3 and Program digests are masked, and the fused norm
must be called the same number of times at the same statics.  Usage: compare.py OLD_DIR NEW_DIR OUT_JSON"""
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

INTERNAL = re.compile(r"^(RMSNormFusedCudaVector8_v[13]\{|RMSNormPairSumCuda_v1$|RMSNormSumStepCuda_v1$|RMSNormSumInOrderCuda_v1\{)")
HEX = re.compile(r"[0-9a-f]{12,}")


def norm(s: str) -> str:
    return HEX.sub("H", s.replace("RMSNormFusedCudaVector8_v1", "RMSNormFusedCudaVector8_v3"))


def resolve(x, types):
    """Type indices (``type``, ``out``) as the types they index, whose numbering differs between descriptors."""
    if isinstance(x, dict):
        return {k: (types[v] if k in ("type", "out") and isinstance(v, int) else resolve(v, types)) for k, v in x.items()}
    if isinstance(x, list):
        return [resolve(v, types) for v in x]
    return x


def summary(path: Path):
    d = json.load(gzip.open(path))
    callees = [norm(c) for c in d["callees"]]
    types = d.get("types") or []
    bodies, norm_calls, census = {}, Counter(), Counter()
    for k, v in d["definitions"].items():
        nk = norm(k)
        if INTERNAL.match(nk):
            continue
        if v.get("form") != "composite":
            bodies[nk] = norm(json.dumps(v, sort_keys=True))
            continue
        nodes = []
        for n in v["body"]["nodes"]:
            m = dict(n)
            if isinstance(m.get("fn"), int):
                m["fn"] = callees[m["fn"]]
            name = m.get("fn") if isinstance(m.get("fn"), str) else json.dumps(m.get("fn"), sort_keys=True)
            census[norm(name)] += 1
            if "RMSNormFusedCudaVector8_v3{" in norm(name):
                norm_calls[(nk, norm(name), m.get("form"), json.dumps(m.get("n")))] += 1
            nodes.append(resolve(m, types))
        body = dict(v["body"], nodes=nodes)
        bodies[nk] = norm(json.dumps({**v, "body": body}, sort_keys=True))
    return {"bodies": bodies, "norm_calls": norm_calls, "census": census, "root": norm(d.get("root", "")),
            "program_fn": norm(d.get("program_fn", "")), "n_defs": len(d["definitions"])}


def compare(old: Path, new: Path) -> dict:
    a, b = summary(old), summary(new)
    only_old = sorted(set(a["bodies"]) - set(b["bodies"]))
    only_new = sorted(set(b["bodies"]) - set(a["bodies"]))
    differ = sorted(k for k in set(a["bodies"]) & set(b["bodies"]) if a["bodies"][k] != b["bodies"][k])
    cen = {k: (a["census"].get(k, 0), b["census"].get(k, 0)) for k in set(a["census"]) | set(b["census"])
           if a["census"].get(k, 0) != b["census"].get(k, 0)}
    return {"only_old": only_old[:20], "only_new": only_new[:20], "differ": differ[:20], "n_only_old": len(only_old),
            "n_only_new": len(only_new), "n_differ": len(differ), "census_diff": dict(list(cen.items())[:20]),
            "norm_calls_equal": a["norm_calls"] == b["norm_calls"],
            "norm_calls": {" | ".join(k): v for k, v in sorted(b["norm_calls"].items())[:12]},
            "norm_call_total": [sum(a["norm_calls"].values()), sum(b["norm_calls"].values())],
            "root_equal": a["root"] == b["root"], "program_fn_equal": a["program_fn"] == b["program_fn"],
            "n_defs": [a["n_defs"], b["n_defs"]]}


def main() -> int:
    old, new, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    only = sys.argv[4] if len(sys.argv) > 4 else ""
    res, bad = {}, []
    for p in sorted(x for x in new.rglob("descriptor.json.gz") if only in str(x)):
        rel = p.relative_to(new)
        q = old / rel
        if not q.exists():
            res[str(rel)] = {"missing_old": True}
            bad.append(str(rel))
            continue
        r = compare(q, p)
        res[str(rel)] = r
        ok = not (r["n_only_old"] or r["n_only_new"] or r["n_differ"] or r["census_diff"]) and r["norm_calls_equal"] \
            and r["root_equal"] and r["program_fn_equal"]
        r["ok"] = ok
        if not ok:
            bad.append(str(rel))
        print(rel, "OK" if ok else "DIFF", r["norm_call_total"], r["n_defs"], flush=True)
        if not ok:
            print(json.dumps({k: r[k] for k in ("only_old", "only_new", "differ", "census_diff", "root_equal", "program_fn_equal")})[:3000], flush=True)
    out.write_text(json.dumps(res, indent=1))
    print("BAD", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
