"""Honest's epilogue writes a committed register for xor-m and xorrot-m, from xor_epilogue_probe.cu's and
xor_wgmma_probe.cu's SASS: stdlib.

    python3 xor_epilogue_sass.py PROBE_SASS [--json OUT]

Counts each kernel's instructions by opcode and takes the base kernel's (IMMA or IGMMA steps alone) out of each variant
of the same M; the difference is the epilogue a block of M steps for one thread's four accumulators (four cells), so a
committed register's share is a quarter of it. The tensor-core count is the same in every kernel of an M. With a
fresh-output IMMA a LOP3 folds two words; with the served form's in-place IGMMA a word is gone at the next step, so
the fold is one write a word.
"""
from __future__ import annotations

import collections
import json
import math
import re
import sys

FUNC = re.compile(r"Function : _Z5probeILi(\d+)EL4Mode(\d)E")
INSN = re.compile(r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)(?:\.[A-Z0-9_.]+)?\s")
MODES = {0: "base", 1: "lead", 2: "xor", 3: "xorrot"}


def counts(text: str) -> dict:
    out: dict = {}
    cur = None
    for line in text.splitlines():
        if "Function :" in line:
            f = FUNC.search(line)
            cur = (int(f.group(1)), MODES[int(f.group(2))]) if f else None
            if cur:
                out[cur] = collections.Counter()
            continue
        i = INSN.search(line)
        if cur and i:
            out[cur][i.group(1)] += 1
    return out


def main() -> None:
    text = open(sys.argv[1]).read()
    c = counts(text)
    res = {}
    ok = True
    for (m, mode), cnt in sorted(c.items()):
        if mode == "base":
            continue
        base = c[(m, "base")]
        diff = {k: cnt[k] - base[k] for k in set(cnt) | set(base) if cnt[k] != base[k]}
        fold, rot = diff.get("LOP3", 0) / 4, diff.get("SHF", 0) / 4
        want_fold = math.ceil((m - 1) / 2) if mode != "lead" else 0
        want_rot = m - 1 if mode == "xorrot" else 0
        rest = {k: v for k, v in diff.items() if k not in ("LOP3", "SHF", "STS")}
        exact = (cnt["IMMA"] == base["IMMA"] == m and fold == want_fold and rot == want_rot
                 and diff.get("STS", 0) == (4 * m if mode == "lead" else 4))
        ok &= exact
        res[f"{mode}-{m}"] = {"imma": cnt["IMMA"], "diff": diff, "fold_lop3_per_register": fold,
                              "shf_per_register": rot, "rest": rest, "as_counted": exact}
        print(f"{'OK  ' if exact else 'FAIL'} {mode}-{m}: IMMA {cnt['IMMA']} (base {base['IMMA']}); a committed "
              f"register: {fold:g} LOP3 (counted {want_fold}), {rot:g} SHF (counted {want_rot}); STS "
              f"{diff.get('STS', 0)}; the loop's other differences {dict(sorted(rest.items()))}")
    w = wcounts(text)
    for (m, mode), cnt in sorted(w.items()):
        if mode != "xor":
            continue
        base = w[(m, "base")]
        diff = {k: cnt[k] - base[k] for k in set(cnt) | set(base) if cnt[k] != base[k]}
        writes = sum(v for k, v in diff.items() if k not in ("STS", "BAR")) / 4
        rest = {k: v for k, v in diff.items() if k not in ("LOP3", "MOV", "STS", "BAR")}
        exact = cnt["IGMMA"] == base["IGMMA"] == m and writes == m and diff.get("STS", 0) == 4
        ok &= exact
        res[f"wgmma-xor-{m}"] = {"igmma": cnt["IGMMA"], "diff": diff, "writes_per_register": writes, "rest": rest,
                                 "per_word": 1 + writes / m, "as_counted": exact}
        print(f"{'OK  ' if exact else 'FAIL'} wgmma-xor-{m}: IGMMA {cnt['IGMMA']} (base {base['IGMMA']}), in place; a "
              f"committed register: {writes:g} writes past the IGMMA's (counted {m}: one a word, since D is C and a "
              f"word is gone at the next step), so honest pays {1 + writes / m:g} a word; the loop's differences "
              f"{dict(sorted(diff.items()))}")
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump(res, f, indent=1, sort_keys=True)
    sys.exit(0 if ok else 1)


WFUNC = re.compile(r"Function : _Z6wprobeILi(\d+)EL5WMode(\d)E")


def wcounts(text: str) -> dict:
    """xor_wgmma_probe.cu's kernels: WGMMA in place, base and xor."""
    out: dict = {}
    cur = None
    for line in text.splitlines():
        if "Function :" in line:
            f = WFUNC.search(line)
            cur = (int(f.group(1)), ("base", "xor")[int(f.group(2))]) if f else None
            if cur:
                out[cur] = collections.Counter()
            continue
        i = INSN.search(line)
        if cur and i:
            out[cur][i.group(1)] += 1
    return out


if __name__ == "__main__":
    main()
