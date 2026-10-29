"""Count `epi_fold.cu`'s unrolled loop body in `cuobjdump -sass`, per output (two a bf16x2 store), per mode, by census
class (`cpw_sass_count.cls`), and check the premise epilogue-fold: the int8 epilogue subtracts a per-row term at no
added instruction an output, as an FFMA's addend (mode 1 against stock's I2F route, mode 0) or in the magic-number
conversion's integer add (mode 4 against mode 3); the exact integer subtraction ahead of I2F (mode 2) is priced beside
them. Prints a table; `--json PATH` writes the counts and checks."""
from __future__ import annotations

import argparse
import collections
import json
import re

from cpw_sass_count import cls, parse

MODES = {0: "stock_i2f", 1: "fold_ffma", 2: "fold_int_i2f", 3: "stock_magic", 4: "fold_magic"}
PAIRS = (("stock_i2f", "fold_ffma", 0), ("stock_magic", "fold_magic", 0), ("stock_i2f", "fold_int_i2f", None))


def count(path: str) -> dict[str, dict]:
    res = {}
    for (arch, k), (ops, where, branches) in parse(path).items():
        m = re.search(r"epiILi(\d)E", k)
        if not m:
            continue
        spans = [(where[t], i) for i, t in branches if t in where and where[t] < i]
        lo, hi = max(spans, key=lambda s: s[1] - s[0]) if spans else (0, len(ops) - 1)
        body = ops[lo:hi + 1]
        outputs = 2 * sum(1 for op in body if op.split(".")[0] in ("STG", "ST"))
        c = collections.Counter(cls(op) for op in body)
        res[MODES[int(m.group(1))]] = {
            "arch": arch, "body": len(body), "outputs": outputs, "per_output": len(body) / outputs,
            "classes_per_output": {k: v / outputs for k, v in c.most_common()},
            "opcodes": dict(collections.Counter(body).most_common())}
    return res


def checks(res: dict) -> list[dict]:
    out = []
    for a, b, want in PAIRS:
        d = res[b]["per_output"] - res[a]["per_output"]
        ca, cb = res[a]["classes_per_output"], res[b]["classes_per_output"]
        moved = {k: cb.get(k, 0) - ca.get(k, 0) for k in set(ca) | set(cb) if cb.get(k, 0) != ca.get(k, 0)}
        out.append({"check": f"{b} against {a}: {d:+g} instructions an output ({moved or 'same classes'})",
                    "added_per_output": d, "moved": moved, "pass": want is None or abs(d - want) < 1e-9})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sass")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = count(a.sass)
    cs = checks(res)
    print("mode | body | outputs | per output | classes an output")
    for name, v in res.items():
        cl = ", ".join(f"{k} {x:g}" for k, x in v["classes_per_output"].items())
        print(f"{name} | {v['body']} | {v['outputs']} | {v['per_output']:g} | {cl}")
    for c in cs:
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {c['check']}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"sass": res, "checks": cs}, fh, indent=1)
    raise SystemExit(0 if all(c["pass"] for c in cs) else 3)
