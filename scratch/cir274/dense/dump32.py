"""CIR-274 DENSE: rank 0's request Program at LP prompt ids, 2 outputs and LAYERS layers (the test's `glm_request`),
node by node: index, family, id, members, axes, output words, return fields, and each argument's runs, with each node's
readers. Run from adapters/vllm:

    uv run python ../../../verity-dense-scratch/scratch/cir274/dense/dump32.py [LP] [LAYERS] [moe]
"""
import sys
from collections import defaultdict

sys.path.insert(0, "tests/program/predict")
from test_node_window import glm_request  # noqa: E402
from verity.circuits.refs import runs  # noqa: E402

LP = int(sys.argv[1]) if len(sys.argv) > 1 else 32
L = int(sys.argv[2]) if len(sys.argv) > 2 else 1
moe = len(sys.argv) > 3 and sys.argv[3] == "moe"
fn = glm_request(LP, L, moe=moe).program.fn
params = [p for p, _ in fn.signature()[0]]
readers = defaultdict(list)
for j, n in enumerate(fn.body.nodes):
    for k, a in enumerate(n.args):
        for sp, idx, lo, hi in runs(a.refs):
            if sp == "n":
                readers[idx].append((j, k, lo, hi))
for j, n in enumerate(fn.body.nodes):
    fam = getattr(n.fn, "name", None) or n.fn.id.split("{")[0]
    try:
        ret = n.fn.signature()[1]
        outs = [(f[0] if isinstance(f, tuple) else "", (f[1] if isinstance(f, tuple) else f).leaves) for f in
                (getattr(ret, "fields", None) or (ret,))]
        ports = [(k, t.leaves, sorted(t.width_set())) for k, t in n.fn.signature()[0]]
    except Exception as e:  # noqa: BLE001
        outs, ports = str(e)[:60], []
    args = []
    for a in n.args:
        rs = []
        for sp, idx, lo, hi in list(runs(a.refs))[:6]:
            rs.append(f"{sp}{params[idx] if sp == 'p' else idx}[{lo},{hi})")
        nr = len(list(runs(a.refs)))
        args.append(" ".join(rs) + (f" ..+{nr - 6}" if nr > 6 else ""))
    rd = sorted({r[0] for r in readers[j]})
    print(f"{j:4d} {n.form:6s} n={n.n} axes={list(n.axes)} out={n.out_leaves} {n.fn.id[:150]}")
    print(f"       ports={ports}")
    print(f"       outs={outs}")
    for k, a in enumerate(args):
        print(f"       arg{k}: {a[:300]}")
    print(f"       readers={rd[:12]}")
