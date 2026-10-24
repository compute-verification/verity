"""Core credit (`credit._window_outcomes`, the Python reference the Lean `PouwCheck.Credit` is difftested against) on
honest windows with GLM-5.3 INT8's layers 0-2 (18 linears, 4 steps, M = 32, tile 32 x 32, r20261023-224227-ea5a's
shapes), built by the credit difftest's case builder (`pouw_credit_cases.honest_window`), on CPU in a few seconds
(VLLM-81).

It credits the window with its Program in GLM's execution order (per layer fused_qkv_a, q_b, indexer.wq_b, o_proj,
gate_up, down) and in module order, refuses the case builder's ROOTS1 negatives on the execution-order layout
(a-row-off, a-call-off, a-call-missing), and refuses ea5a's 70-call layout (calls booked a step early, two refused at
step -1), whose rounds are not equal blocks. ROOT is a checkout with the per-(step, K) A tables (#3360's
cursor/vllm-astep-496a, or a branch that merged it); RECORD is ea5a's ``out/commit/pouw-h1-calls.json``
(``research fetch r20261023-224227-ea5a --all``), read for its shapes, module names and call order:

    git worktree add /tmp/root origin/cursor/vllm-astep-496a
    cd /tmp/root && PYTHONPATH=/tmp/root uv run python benchmarks/pouw/glm_layout/core_glm_order_on_3360.py /tmp/root RECORD.json
"""
import json
import random
import sys

root = sys.argv[1]
sys.path[:0] = [root, f"{root}/verity/core/protocols/pouw/lean/check/tests"]

import pouw_credit_cases as cc  # noqa: E402
from verity.core.protocols.pouw import credit as Cr  # noqa: E402

rec = json.load(open(sys.argv[2]))
mods: dict[int, str] = {}
for c in rec["calls"]:
    mods.setdefault(c["layer"], c["module"])
shapes = rec["public"]["shapes"]
L, S = len(shapes), rec["public"]["steps"]
linears = [[mods[w], k, n] for w, (n, k) in enumerate(shapes)]
of_k: dict[int, list[int]] = {}
for w, (_n, k) in enumerate(shapes):
    of_k.setdefault(k, []).append(w)


def block(s: int, w: int) -> int:
    """`pouw_h1_served.layout`'s A block: each linear its own block of its K, step after step."""
    return s * len(of_k[shapes[w][1]]) + of_k[shapes[w][1]].index(w)


# GLM's execution order in one forward: ea5a's step 1 records it rotated by the two calls booked to the step before
executed = [1, 2, 0, 3, 4, 5, 7, 8, 6, 9, 10, 11, 13, 14, 12, 15, 16, 17]
assert [c["layer"] for c in rec["calls"] if c["step"] == 1] == executed[2:] + executed[:2]


def lay(calls: list[tuple[int, int]]) -> dict:
    return {"calls": [[mods[w], block(s, w)] for s, w in calls], "linears": linears, "m": 32, "tile": [32, 32]}


def run(d: dict, s, rng: random.Random):
    d["ledger"] = cc.rc.ledger_lines(cc.rc.ledger_events(rng, d["window"], s, period=cc.signs(d["weights"])),
                                     cc.rc._key(0))
    return Cr._window_outcomes(*cc.window_args({k: d[k] for k in cc.WINDOW_KEYS}))


def show(name: str, d: dict, got, why) -> None:
    print(f"{name}: {len(d['layout']['calls'])} calls, {cc.units_of(d['layout'])} units, {len(d['rounds'])} rounds, "
          f"K {d['k']}, {len(d['proofs'])} drawn -> " + (f"CREDITED {len(got)} calls" if why is None else f"REFUSED {why}"))


for name, order in (("Program in GLM's execution order", executed), ("Program in module order", list(range(L)))):
    rng = random.Random(81)
    d, s = cc.honest_window(rng, lay=lay([(st, w) for st in range(S) for w in order]))
    show(name, d, *run(d, s, rng))

for off in ("a-row-off", "a-call-off", "a-call-missing"):
    rng = random.Random(81)
    d, s = cc.honest_window(rng, off, lay=lay([(st, w) for st in range(S) for w in executed]))
    show(f"GLM order, {off}", d, *run(d, s, rng))

shifted = [(st, w) for st in range(S - 1) for w in (executed[2:] + executed[:2] if st < S - 2 else executed[2:])] \
    + [(S - 1, w) for w in range(L)]
assert [(c["step"], c["layer"]) for c in rec["calls"]] == shifted
rng = random.Random(81)
d, s = cc.honest_window(rng, lay=lay([(st, w) for st in range(S) for w in executed]))
d["layout"] = lay(shifted)
print("ea5a's 70-call record as the Program, _arow:", Cr._arow(d["rounds"], d["entries"], Cr.commit_of(d["commit"])[0][1],
                                                               d["layout"], Cr._prefix(bytes.fromhex(d["descriptor"]))))
