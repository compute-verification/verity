"""Honest's epilogue writes a committed register for xor-m and xorrot-m, from xor_epilogue_probe.cu's SASS: stdlib.

    python3 xor_epilogue_sass.py PROBE_SASS [--json OUT]

Counts each kernel's instructions by opcode and takes the base kernel's (IMMA steps alone) out of each variant of the
same M; the difference is the epilogue a block of M steps for one thread's four accumulators (four cells), so a
committed register's share is a quarter of it. The IMMA count is the same in every kernel of an M.
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
        f = FUNC.search(line)
        if f:
            cur = (int(f.group(1)), MODES[int(f.group(2))])
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
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump(res, f, indent=1, sort_keys=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
