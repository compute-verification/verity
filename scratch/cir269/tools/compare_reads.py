"""Each catalog binding's record's reads at the base against the head: python3 compare_reads.py BASE_READS HEAD_READS
[EXPECTED_FILES].  READS is records-job.sh's reads.json ({binding: {"built", "reads": {kind: {path: digest}}}}).  A read is
(kind, path); EXPECTED_FILES lists, one a line, the paths whose differences are expected (circuit-check's own, the files
#9659 and #9660 change, the catalog's new sidecar).  It prints the bindings in one and not the other, every binding whose
record at the head reads fewer files than at the base with the files it lost, and the other differences by kind."""
import json
import sys
from collections import Counter

base, head = (json.load(open(p)) for p in sys.argv[1:3])
expected = set(open(sys.argv[3]).read().split()) if len(sys.argv) > 3 else set()
B, H = base["bindings"], head["bindings"]
print(f"base: {len(B)} records, version {base['version'][:16]}, {len(base['unstored'])} unstored")
print(f"head: {len(H)} records, version {head['version'][:16]}, {len(head['unstored'])} unstored")
print("bindings only at the base:", sorted(set(B) - set(H)))
print("bindings only at the head:", sorted(set(H) - set(B)))


def files(r):
    return {p for paths in r["reads"].values() for p in paths}


def reads(r):
    return {(k, p): dg for k, paths in r["reads"].items() for p, dg in paths.items()}


fewer, lost_any, gained, changed = [], Counter(), Counter(), Counter()
unexpected = []
for name in sorted(set(B) & set(H)):
    fb, fh = files(B[name]), files(H[name])
    rb, rh = reads(B[name]), reads(H[name])
    if len(fh) < len(fb):
        fewer.append((name, len(fb), len(fh), sorted(fb - fh), sorted(fh - fb)))
    for p in fb - fh:
        lost_any[p] += 1
    for p in fh - fb:
        gained[p] += 1
    for kp in rb.keys() & rh.keys():
        if rb[kp] != rh[kp]:
            changed[kp[1]] += 1
    odd = {p for p in (fb ^ fh) | {kp[1] for kp in rb.keys() & rh.keys() if rb[kp] != rh[kp]}} - expected
    if odd:
        unexpected.append((name, sorted(odd)))
print(f"\nbindings whose record at the head reads fewer files than at the base: {len(fewer)}")
for name, nb, nh, lost, new in fewer:
    print(f"  {name}: {nb} -> {nh}; lost {lost}; gained {new}")
print(f"\nfiles some binding's record stopped reading ({len(lost_any)}):")
for p, n in lost_any.most_common():
    print(f"  {n:4} {p}{'' if p in expected else '   [not expected]'}")
print(f"\nfiles some binding's record started reading ({len(gained)}):")
for p, n in gained.most_common():
    print(f"  {n:4} {p}{'' if p in expected else '   [not expected]'}")
print(f"\nfiles read at both whose digest differs ({len(changed)}):")
for p, n in changed.most_common():
    print(f"  {n:4} {p}{'' if p in expected else '   [not expected]'}")
print(f"\nbindings with a difference outside the expected files: {len(unexpected)}")
for name, odd in unexpected[:40]:
    print(f"  {name}: {odd[:8]}{' ...' if len(odd) > 8 else ''}")
