"""CIR-274 DSAMOE: from a window's stored description alone (`p4_describe.py`'s ``description_<P>_<O>.json``), the inference
block (slot, step) of every unit its wires join, and for each wire pattern whether every (writer unit, reader unit) pair it
joins lies in one block.

The blocks are the description's ``blocks.inference`` entries (`interface.InferenceEntry`, `Blocks.slots_steps`): unit
``at + i`` of an entry has ``i``'s digits in its mixed radix, the first the slowest, its slot the ``slot`` digit and its
step the entry's ``step`` plus its ``step`` digit. A wire (`interface._Pattern`): reader index ``i < prod(radix)`` is
reader unit ``at + i // C`` (``C = cut[1]``, 1 uncut) and reads writer unit ``base + sum(d[k] * step[k])``. Every index
is visited (numpy), so this is the per-pair reading, not a check by pattern fields.

The key wires are the patterns whose reader port is ``tok`` of an ``EmbeddingRead_v1`` template (the expert reads'
key, `p3_keycost.py`'s selection); their writer templates are listed so the selection can be checked.

    python3 p4_blocks.py OUT description_43_2.json [description_43_3.json ...]

Writes OUT/blocks_<P>_<O>.json and prints one line a file."""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def entries_by_template(desc):
    by = defaultdict(list)
    for e in desc["blocks"]["inference"]["entries"]:
        by[e["template"]].append(e)
    out = {}
    for t, es in by.items():
        es.sort(key=lambda e: e["at"])
        out[t] = (np.array([e["at"] for e in es], dtype=np.int64), es)
    return out


def digit(e, axis, i):
    if axis not in e["axes"]:
        return np.zeros_like(i)
    k = e["axes"].index(axis)
    return i // int(np.prod(e["radix"][k + 1:], dtype=np.int64)) % e["radix"][k]


def blocks_of(ents, tid, u):
    """Units ``u`` of template ``tid``: each one's (slot, step), as two arrays."""
    starts, es = ents[tid]
    k = np.searchsorted(starts, u, side="right") - 1
    slot, step = np.empty_like(u), np.empty_like(u)
    for j in np.unique(k):
        e = es[j]
        m = k == j
        i = u[m] - e["at"]
        if (i < 0).any() or (i >= int(np.prod(e["radix"]))).any():
            raise ValueError(f"{tid}: units outside every entry")
        slot[m], step[m] = digit(e, "slot", i), e["step"] + digit(e, "step", i)
    return slot, step


def source(w, i):
    out = np.full(i.shape, w["base"], dtype=np.int64)
    i = i.copy()
    for r, s in zip(reversed(w["radix"]), reversed(w["step"])):
        out += (i % r) * s
        i //= r
    return out


def joins(ents, w):
    """The wire's (reader block, writer block) pairs, as a Counter of ((rslot, rstep), (wslot, wstep)) by index count."""
    n = int(np.prod(w["radix"], dtype=np.int64))
    i = np.arange(n, dtype=np.int64)
    C = w.get("cut", [0, 1])[1]
    rs, rt = blocks_of(ents, w["reader"], w["at"] + i // C)
    ws, wt = blocks_of(ents, w["writer"], source(w, i))
    key = np.stack([rs, rt, ws, wt], axis=1)
    u, c = np.unique(key, axis=0, return_counts=True)
    return {(int(a), int(b), int(x), int(y)): int(k) for (a, b, x, y), k in zip(u, c)}


def short(tid):
    return re.sub(r",DOT=\{.*", ",DOT=…}", tid)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for f in sys.argv[2:]:
        desc = json.loads(Path(f).read_bytes())
        ents = entries_by_template(desc)
        inf = desc["blocks"]["inference"]
        rec = {"file": Path(f).name, "slots": inf["slots"], "steps": inf["steps"], "entries": len(inf["entries"]),
               "templates": len(desc["templates"]), "wires": len(desc["wires"])}
        allw = Counter()
        between: list = []
        key = {"count": 0, "writers": Counter(), "by_reader": defaultdict(Counter), "pairs_index": Counter()}
        for w in desc["wires"]:
            j = joins(ents, w)
            inside = all(a == x and b == y for a, b, x, y in j)
            allw["inside" if inside else "between"] += 1
            if not inside:
                between.append({"reader": short(w["reader"]), "port": w["port"], "writer": short(w["writer"]), "out": w["out"],
                                "at": w["at"], "radix": w["radix"], "base": w["base"], "step": w["step"],
                                "writer step -> reader step": sorted({f"{y}->{b}" for _a, b, _x, y in j})})
            if w["port"] == "tok" and w["reader"].startswith("EmbeddingRead_v1"):
                key["count"] += 1
                key["writers"][short(w["writer"])] += 1
                r = key["by_reader"][short(w["reader"])]
                r["wires"] += 1
                if inside:
                    r["inside"] += 1
                else:
                    r["between"] += 1
                    r["steps " + "+".join(f"{y}->{b}" for _a, b, _x, y in sorted(j) if (_a, b) != (_x, y))] += 1
                for (a, b, x, y), k in j.items():
                    key["pairs_index"][f"writer step {y} -> reader step {b}" + ("" if a == x else " (slots differ)")] += k
                r[f"reader step {sorted({b for _a, b, _x, _y in j})}"] += 1
        rec["all_wires"] = dict(allw)
        rec["wires_between_blocks"] = between
        rec["key_wires"] = {"count": key["count"], "writers": dict(key["writers"]),
                            "inside": sum(r["inside"] for r in key["by_reader"].values()),
                            "between": sum(r["between"] for r in key["by_reader"].values()),
                            "index_pairs_by_steps": dict(key["pairs_index"]),
                            "by_reader": {t: dict(c) for t, c in sorted(key["by_reader"].items())}}
        name = out / f"blocks_{Path(f).stem.removeprefix('description_')}.json"
        name.write_text(json.dumps(rec, indent=1) + "\n")
        print("DSAMOE blocks", rec["file"], json.dumps({k: rec[k] for k in ("slots", "steps", "entries", "wires", "all_wires")}),
              json.dumps({k: v for k, v in rec["key_wires"].items() if k != "by_reader"}), flush=True)


if __name__ == "__main__":
    main()
