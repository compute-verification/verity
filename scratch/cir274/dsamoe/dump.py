"""CIR-274 DSAMOE: rank 0's request Program at LP prompt ids, 1 layer (dense, or moe), dumped node by node and pickled.

    cd adapters/vllm && uv run python /tmp/dsamoe/dump.py OUT LP LAYERS [moe]
"""
import json
import pickle
import resource
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "tests/program/predict"))
from test_node_window import glm_request  # noqa: E402
from verity.circuits.refs import runs  # noqa: E402

out, LP, layers = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
moe = len(sys.argv) > 4 and sys.argv[4] == "moe"
out.mkdir(parents=True, exist_ok=True)
fn = glm_request(LP, layers, moe=moe).program.fn
tag = f"lp{LP}_l{layers}{'_moe' if moe else ''}"
lines = []
for j, n in enumerate(fn.body.nodes):
    fam = getattr(n.fn, "name", None) or n.fn.id.split("{")[0]
    try:
        sig = n.fn.signature()
        ports = [(k, t.leaves, sorted(t.width_set())) for k, t in sig[0]]
        ret = sig[1]
        outs = [(f[0] if isinstance(f, tuple) else "", (f[1] if isinstance(f, tuple) else f).leaves) for f in (getattr(ret, "fields", None) or (ret,))]
    except Exception as e:  # noqa: BLE001
        ports, outs = str(e), None
    lines.append(f"{j} {n.form} {fam} n={n.n} axes={list(n.axes)} out={n.out_leaves} id={n.fn.id[:300]}\n  ports={ports}\n  outs={outs}\n"
                 + "".join(f"  arg{k}: {list(runs(a.refs))[:12]}\n" for k, a in enumerate(n.args)))
(out / f"{tag}.txt").write_text("".join(lines))
print(tag, len(fn.body.nodes), "peak GiB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576)
