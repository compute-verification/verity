"""CIR-274 DSAMOE: the sweep's tables from the runs' fix_<OUTS>.json files (`run_sweep.sh`), as Markdown on stdout:

    python3 sweep_table.py UNPATCHED_2 UNPATCHED_3 PATCHED_2 PATCHED_3   # each a fix_<OUTS>.json

The lengths each tree refuses, by outputs, with the stuck wire the error names; and, per length, the wires, BatchedGemm's x
wires, SparseMlaSplit's patterns and the chain shapes that repeat a key."""
import json
import re
import sys

names = ("unpatched, 2 outputs", "unpatched, 3 outputs", "patched, 2 outputs", "patched, 3 outputs")
runs = [json.load(open(p))["lengths"] for p in sys.argv[1:5]]
for name, r in zip(names, runs):
    bad = {int(k): v for k, v in r.items() if not v.get("ok")}
    print(f"- {name}: {len(r)} lengths, refused at {sorted(bad) or 'none'}")
    for k, v in sorted(bad.items()):
        m = re.search(r"wire \d+ (\{.*?\}) after its writers: (\d+) ranges", v["error"])
        doc = json.loads(m.group(1).replace("'", '"')) if m else {}
        print(f"  - {k}: {m.group(2) if m else '?'} ranges stuck, {doc.get('reader', '?')} {doc.get('port', '')} <- "
              f"{doc.get('writer', '?')}" if m else f"  - {k}: {v['error'][:200]}")
print()
print("| ids | wires, 2 out | BatchedGemm x | SparseMlaSplit | split tokens | repeated keys (prefill) |")
print("|---|---|---|---|---|---|")
for k in sorted(runs[2], key=int):
    v = runs[2][k]
    if not v.get("ok"):
        print(f"| {k} | refused | | | | |")
        continue
    bg = sum(n for kk, n in v["batched_gemm_x"].items() if "K=512" in kk and "step 0" in kk)
    split = sum(n for s, n in v["sparse_mla_shapes"].items() if int(re.search(r"places=(\d+)", s).group(1)) > 5
                and int(re.search(r"T=(\d+)", s).group(1)) <= int(k)) // 4
    rep = sum(1 for s in v["sparse_mla_shapes"] if not s.endswith("repeated_keys=0") and int(re.search(r"T=(\d+)", s).group(1)) <= int(k))
    print(f"| {k} | {v['wires']:,} | {bg:,} | {v['by_family'].get('SparseMlaSplit', 0):,} | {split} | {rep} |")
