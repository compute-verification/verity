"""PADCREDIT R1: a selection applied by arithmetic after the hardness certificate escapes the gated-cone charge.

Mirrors `Accounting.TS.SelectLayer` (cursor/padded-credit-stmt-ea24 at 5d09a7396): Feeds = Late ∪ Gates, SelGood(v) =
every u with u Feeds* v is paid. Every position here is committed before the tile's coin, so there are no Late edges
(as in the static window: rows and ok are in the side rows, committed before the coin).

Case 1 (STATIC-WIN's wiring): the tile reads the side row Pc8RowA writes, codes and ok together. A cheater that forges
ok = 1 on a refused row leaves Pc8RowA unpaid, and the tile isn't good: the cone is charged.
Case 2 (a done flag or EOS mask applied after the forming, by arithmetic): Mask writes codes' = codes * (1 - done),
the tile reads codes' and its gates are {ok}. A wrong Mask writes the zero row; ok, computed on the unmasked row, is
1. Mask writes no gate position and nothing after the coin, so it doesn't feed the tile: the tile is selected and good,
its operand is a row the forming refuses, and its credit has no charge.
Case 3 (the mask before the forming): the forming certifies the masked row, so a zero row gets ok = 0 and no credit.

Runs anywhere: ``uv run python benchmarks/rate_redteam/padcredit_r1_cone_escape.py``.
"""


def refl_trans(feeds, units, v):
    seen, todo = {v}, [v]
    while todo:
        x = todo.pop()
        for u in units:
            if u not in seen and feeds(u, x):
                seen.add(u)
                todo.append(u)
    return seen


def run(case):
    pos_writer = {"row": "RowA", "ok": "RowA", "done": "Done", "masked": "Mask"}
    if case == 1:
        units, reads, gates = ["RowA", "Tile"], {"Tile": ["row", "ok"]}, {"Tile": ["ok"]}
        mode = {"RowA": "whole", "Tile": "paid"}
        operand, ok = [0] * 8, 1
    elif case == 2:
        units = ["RowA", "Done", "Mask", "Tile"]
        reads, gates = {"Mask": ["row", "done"], "Tile": ["masked", "ok"]}, {"Tile": ["ok"]}
        mode = {"RowA": "paid", "Done": "paid", "Mask": "whole", "Tile": "paid"}
        row = [3, -1, 4, 1, -5, 9, 2, -6]
        ok = int(any(row))
        operand = [0] * 8
    else:
        units = ["Done", "Mask", "RowA", "Tile"]
        pos_writer = {"masked": "Mask", "row": "RowA", "ok": "RowA", "done": "Done"}
        reads, gates = {"Mask": ["done"], "RowA": ["masked"], "Tile": ["row", "ok"]}, {"Tile": ["ok"]}
        mode = {"Done": "paid", "Mask": "whole", "RowA": "paid", "Tile": "paid"}
        operand = [0] * 8
        ok = int(any(operand))

    late = lambda u, v: False
    gate = lambda u, v: any(pos_writer[p] == u for p in gates.get(v, []))
    feeds = lambda u, v: late(u, v) or gate(u, v)
    cone = refl_trans(feeds, units, "Tile")
    good = all(mode[u] == "paid" for u in cone)
    selected = ok == 1
    refused = not any(operand)
    return selected, good, refused, sorted(cone)


for case, what in ((1, "tile reads RowA's row (STATIC-WIN)"), (2, "mask after the forming"), (3, "mask before the forming")):
    sel, good, refused, cone = run(case)
    print(f"case {case}, {what}: selected={sel} SelGood={good} operand refused by forming={refused} cone={cone}"
          f" -> credited with no charge on a refused operand: {sel and good and refused}")
