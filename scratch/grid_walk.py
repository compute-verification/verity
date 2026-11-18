"""Replays the grid branch of the Lean decoder's `Widths.partOk` (circuits/lean/Circuits/Program.lean) over the root's
arguments: a grid part takes the fast path only if the argument's type has one width `w` and every lane's source has
`w` (its own one width, or one width between the lane's extremes); otherwise it walks `q.count` leaves, each a `step`
of the Program's budget `WALK = 2^20`. Prints the parts that walk, the running total, and what one step a lane a
period (the Python codec's `Grid.pieces`) would cost instead."""
import gzip, json, sys, time
sys.path.insert(0, ".")
import inference_only_record_reference as R
from verity.circuits.refs import Concat, Grid
from verity.circuits.types import range_widths
WALK = 2 ** 20
text = (gzip.open if sys.argv[1].endswith(".gz") else open)(sys.argv[1], "rt").read()
program = R.reference.desc_program(text)
root = program.root
body = root.body
outs = [n.out for n in body.nodes]
walked = pieces = shown = 0
crossed = None
for k, node in enumerate(body.nodes):
    for i, a in enumerate(node.args):
        t, refs = a.type, a.refs
        parts = refs.parts if isinstance(refs, Concat) else [refs]
        starts = refs.starts if isinstance(refs, Concat) else [0]
        for q, pos in zip(parts, starts):
            if not isinstance(q, Grid):
                continue
            w = t.homogeneous_width()
            def lane_ok(l):
                s, idx, b, e, lv = l
                if s != "n":
                    return True
                S = outs[idx]
                if S.homogeneous_width() == w:
                    return True
                lo = b + sum(min(0, (r - 1) * d) for r, d in lv) + min(0, (q.run - 1) * e)
                hi = b + sum(max(0, (r - 1) * d) for r, d in lv) + max(0, (q.run - 1) * e)
                return set(range_widths(S, lo, hi + 1)) == {w}
            if w is not None and all(lane_ok(l) for l in q.lanes):
                continue
            walked += q.count
            pieces += q.periods * len(q.lanes)
            why = "argument type has more than one width" if w is None else "a lane's source range has another width"
            if shown < 6 or (crossed is None and walked > WALK):
                print(f"walks: root node {k} ({node.form} {node.fn.name if hasattr(node.fn, 'name') else ''}) argument {i} "
                      f"at leaf {pos}: {q.count} leaves ({why}; type {t!s:.120}); total {walked}; part {json.dumps(q.to_json())[:300]}")
                shown += 1
            if crossed is None and walked > WALK:
                crossed = (k, i)
                print(f"  ^ the budget WALK = {WALK} is crossed here: the decoder's refusal")
print(f"total leaves walked {walked} (budget {WALK}); one step a lane a period instead: {pieces}")
