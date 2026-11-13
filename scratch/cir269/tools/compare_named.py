"""compare_named.py HEAD_RUN BASE_RUN: each target's report entry at both commits, whole but for timing and memory."""
import json
import os
import sys

ROOT = os.path.expanduser("~/.research/runs")
DROP = {"seconds", "seconds_by_check", "peak_gb", "rss_start_gb"}


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in DROP}
    if isinstance(x, list):
        return [strip(v) for v in x]
    if isinstance(x, str):
        return x.replace(HEAD_DIR, "RUN").replace(BASE_DIR, "RUN")
    return x


def diffs(a, b, path=""):
    if type(a) is not type(b):
        yield path, a, b
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                yield f"{path}.{k}", a.get(k, "<absent>"), b.get(k, "<absent>")
            else:
                yield from diffs(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            yield f"{path}[len]", len(a), len(b)
        for i, (x, y) in enumerate(zip(a, b)):
            yield from diffs(x, y, f"{path}[{i}]")
    elif a != b:
        yield path, a, b


def hooks(run, i):
    p = f"{ROOT}/{run}/out/r/{i}.hooks"
    if not os.path.exists(p):
        return {}
    seen = {}
    for line in open(p):
        name, caller, *_ = json.loads(line)
        seen.setdefault(name, set()).add(caller)
    return {k: sorted(v) for k, v in seen.items()}


head, base = sys.argv[1:3]
HEAD_DIR, BASE_DIR = f"{ROOT}/{head}", f"{ROOT}/{base}"
ids = sorted(int(f[:-5]) for f in os.listdir(f"{HEAD_DIR}/out/r") if f.endswith(".json") and f[:-5].isdigit())
for i in ids:
    pa, pb = f"{HEAD_DIR}/out/r/{i}.json", f"{BASE_DIR}/out/r/{i}.json"
    if not os.path.exists(pb):
        print(f"{i}: no base report yet")
        continue
    A, B = json.load(open(pa)), json.load(open(pb))
    for ta, tb in zip(A["targets"], B["targets"]):
        fa = [(f["check"], f["code"], f["name"].splitlines()[0][:160]) for f in ta.get("failures", [])]
        d = list(diffs(strip(ta), strip(tb)))
        corr = ta.get("correspondence", {})
        real = [r.get("via") for r in corr.get("realizations", [])]
        qc = (ta.get("partition") or {}).get("q_call")
        unch = sorted({(w["check"], w["code"]) for w in ta.get("warnings", [])})
        print(f"{i} {ta['target']}\n  keys compared {len(strip(ta))}; differences {len(d)}; failures {fa}\n"
              f"  vectors {corr.get('vectors')} edge {corr.get('edge_vectors')} via {real}; q_call {qc}; unchecked {unch}\n"
              f"  hooks {hooks(head, i)}")
        for p, x, y in d[:10]:
            print(f"  DIFF {p}: head {json.dumps(x)[:300]} | base {json.dumps(y)[:300]}")
    sd = list(diffs(strip({k: v for k, v in A["summary"].items()}), strip({k: v for k, v in B["summary"].items()})))
    if sd:
        for p, x, y in sd[:5]:
            print(f"  SUMMARY DIFF {p}: head {json.dumps(x)[:200]} | base {json.dumps(y)[:200]}")
