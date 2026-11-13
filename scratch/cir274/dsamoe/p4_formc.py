"""CIR-274 DSAMOE: the experts' key wires of a stored window description (`p4_describe.py`), today's and form C's
(`key-wires.md`), each with the inference blocks of its reader units and of its writer units, from the description alone.

Today's: every pattern whose reader port is ``tok`` of an ``EmbeddingRead_v1`` template, as stored. Its MoE node in a
step is read template R's units at that step (R's inference entry there); a node's units are ``(b TOPK + k) T + t``, B
tables of TOPK slots of T tokens, slot k reading the router's output ``o<TOPK+k>`` (its expert id) of token t.

Form C's, written from today's (not built): the node's units in slot-major order ``(k B + b) T + t``, the same allocation
(so the same inference entries), and slot k's run one pattern: ``at`` = the node's first unit + k B T, ``radix`` (B, T),
``base`` today's base of token 0, ``step`` (0, today's token stride, 1 where T = 1), the reader port, writer and ``out``
as today. TOPK patterns a read template a node.

Blocks by `p4_blocks.joins`: every reader index visited. A pattern is "inside" where every (writer unit, reader unit) pair
it joins has one (slot, step).

    python3 p4_formc.py OUT description_43_2.json [description_43_3.json ...]

Writes OUT/keywires_<P>_<O>.json: a summary, ``templates`` (the descriptor ids a pattern's ``reader`` and ``writer`` index),
and ``today`` and ``c``, one pattern a line with its fields, its reader and writer blocks ([slot, step] each) and
``inside``; prints one line a file."""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from p4_blocks import entries_by_template, joins  # noqa: E402

TOPK = 8


def with_blocks(ents, w):
    j = joins(ents, w)
    rb = sorted({(a, b) for a, b, _x, _y in j})
    wb = sorted({(x, y) for _a, _b, x, y in j})
    return {**w, "reader_blocks": [list(b) for b in rb], "writer_blocks": [list(b) for b in wb],
            "inside": all(a == x and b == y for a, b, x, y in j)}


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for f in sys.argv[2:]:
        desc = json.loads(Path(f).read_bytes())
        ents = entries_by_template(desc)
        key = [w for w in desc["wires"] if w["port"] == "tok" and w["reader"].startswith("EmbeddingRead_v1")]
        today = [with_blocks(ents, w) for w in key]
        nodes = defaultdict(list)
        for w, t in zip(key, today):
            (slot, step), = t["reader_blocks"]
            nodes[(w["reader"], slot, step)].append(w)
        c, node_rec = [], []
        for (reader, slot, step), ws in sorted(nodes.items()):
            T = ws[0]["radix"][0]
            if any(w["radix"] != [T] for w in ws) or len(ws) % TOPK:
                raise ValueError(f"{reader} at step {step}: today's key wires are not one radix-(T,) run a (block, slot)")
            B, a0 = len(ws) // TOPK, min(w["at"] for w in ws)
            if sorted(w["at"] for w in ws) != list(range(a0, a0 + B * TOPK * T, T)):
                raise ValueError(f"{reader} at step {step}: today's runs don't tile one allocation of B TOPK T units")
            node_rec.append({"reader": reader, "slot": slot, "step": step, "first_unit": a0, "B": B, "T": T,
                             "entries": [e for e in ents[reader][1]
                                         if e["at"] < a0 + B * TOPK * T and a0 < e["at"] + math.prod(e["radix"])]})
            for k in range(TOPK):
                w0 = min((w for w in ws if (w["at"] - a0) // T % TOPK == k), key=lambda w: w["at"])
                stride = w0["step"][0] if T > 1 else 1
                c.append(with_blocks(ents, {"reader": reader, "port": "tok", "writer": w0["writer"], "out": w0["out"],
                                            "at": a0 + k * B * T, "radix": [B, T], "base": w0["base"], "step": [0, stride]}))
        summ = {"file": Path(f).name,
                "today": {"patterns": len(today), "inside": sum(w["inside"] for w in today)},
                "c": {"patterns": len(c), "inside": sum(w["inside"] for w in c),
                      "reader_units": sum(w["radix"][0] * w["radix"][1] for w in c)},
                "today_reader_units": sum(w["radix"][0] for w in today),
                "nodes": node_rec}
        names = sorted({w[k] for w in today + c for k in ("reader", "writer")})
        ix = {t: i for i, t in enumerate(names)}

        def row(w):
            return json.dumps({**w, "reader": ix[w["reader"]], "writer": ix[w["writer"]]}, separators=(",", ":"))

        lines = [json.dumps({"summary": summ, "templates": names}, indent=1)[:-2] + ',\n "today": [',
                 ",\n".join(row(w) for w in today), '],\n "c": [', ",\n".join(row(w) for w in c), "]}\n"]
        (out / f"keywires_{Path(f).stem.removeprefix('description_')}.json").write_text("\n".join(lines))
        print("DSAMOE keywires", json.dumps({k: summ[k] for k in ("file", "today", "c", "today_reader_units")}), flush=True)


if __name__ == "__main__":
    main()
