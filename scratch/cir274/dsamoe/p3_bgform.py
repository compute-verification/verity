"""CIR-274 DSAMOE: a prototype, in scratch only, of BatchedGemm's x read when its members' rows come from more than one
writer template (a prefill step that mixes split and unsplit attention chains, `p3_splits.py`). Where one feed over the
whole (B, N, M) grid holds, `batched_gemm` as it is. Else its units member-slowest, ``a + t B N + b N + n``, x fed a run of
like members at a time over (members, B, N), and its output ``Piece(0, B N, Row(t, a, "o0", 1), 1, B N)``. Installed over
`batched_gemm` by monkeypatch, so the tree stays as it is, with the value check's stand-in for a free w (`node_window_values.FREE`), which
assumes the tree's unit order, taught the member-slowest one for the nodes laid out so. Run from the root of the tree:

    .venv/bin/python p3_bgform.py OUT OUTS:LP...

For each length, before and after: whether it builds (the Circuit's `check_wires` and `check_inputs`), its digest, wires,
wires before `coalesce`, and BatchedGemm's x wires. Writes OUT/bgform.json."""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

from verity.circuits.codec import descriptor_id  # noqa: E402
from verity.commitments.windows import interface as IF  # noqa: E402
from verity_vllm.program.predict import node_window as NW  # noqa: E402
from verity_vllm.program.predict import node_window_dense as ND  # noqa: E402
from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402
from verity_vllm.program.predict import node_window_values as NV  # noqa: E402
from verity_vllm.program.predict.steps import plan_steps, request_program  # noqa: E402
from verity_vllm.program.predict.window_builder import Row  # noqa: E402

ORIG = NS.batched_gemm
MIXED = {"nodes": 0, "runs": 0}
#: the nodes laid out member-slowest, whose free w the value check's stand-in reads in that order
SLOWEST: set[int] = set()
_FREE_W = NV.FREE[("BatchedGemm", "w")]


def _batched_gemm_w(ck, node, a, u, words):
    if id(node) not in SLOWEST:
        return _FREE_W(ck, node, a, u, words)
    B, K, N = (int(node.fn.bindings[x]) for x in ("B", "K", "N"))
    t, bn = divmod(u - a, B * N)
    if words != K:
        raise ValueError(f"BatchedGemm w: a row of {words} words, not K = {K}")
    return ck.values.leaves(ck.values.arg(node, 1, t).slice(bn * K, (bn + 1) * K)).astype(np.uint16)


NV.FREE[("BatchedGemm", "w")] = _batched_gemm_w


def batched_gemm_runs(w, j, node):
    spec, M = descriptor_id(node.fn), w.members(node)
    B, K, N = (int(node.fn.bindings[x]) for x in ("B", "K", "N"))
    xs = [w.take(node, 0, b * K, (b + 1) * K, whole=True) for b in range(B)]
    if any(isinstance(x, NS._Free) for x in xs) or NW._lead([x[m] for x in xs for _n in range(N) for m in range(M)],
                                                            (B, N, M)) is not None:
        return ORIG(w, j, node)
    fn = NS._part(spec, "BatchedGemm_v1", B=1, N=1)
    t, a = w.unit(NS._family(node), fn, B * N * M)
    key = [tuple((r.tid, r.out, r.words) for r in xs[0][m]) for m in range(M)]
    s = 0
    for e in range(1, M + 1):
        if e == M or key[e] != key[s]:
            w.feed(t, "x", a + s * B * N, [xs[b][m] for m in range(s, e) for b in range(B) for _n in range(N)], (e - s, B, N))
            MIXED["runs"] += 1
            s = e
    MIXED["nodes"] += 1
    SLOWEST.add(id(node))
    refs = node.args[1].refs
    if len(w.segments(refs.slice(0, K))) == 1:
        w.b.read(t, "w", a, (M, B, N), w.weight(refs, B * N * K), (0, N * K, K))
    else:
        w.feed(t, "w", a, NS._Free([("param", NS.WEIGHTS, NS._label_at(w, refs))]), (M, B, N))
    w.out(j, node, [NS.Piece(0, B * N, Row(t, a, "o0", 1), 1, B * N)])


FAM0 = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES}
FAM1 = {**FAM0, "BatchedGemm": batched_gemm_runs}


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rec = {}
    for arg in sys.argv[2:]:
        OUTS, LP = map(int, arg.split(":"))
        reg = cc.registered(0, 4, "eager")
        advice, ingress = cc.session([(0, LP, OUTS)])
        u = [x for x in plan_steps(reg, advice, ingress).requests if x is not None][0]
        fn = request_program(reg, u).program.fn
        r = {}
        for name, fam in (("before", FAM0), ("after", FAM1)):
            MIXED.update(nodes=0, runs=0)
            t0 = time.perf_counter()
            try:
                c, w = NW.node_window(fn, families=fam)
                IF.check_wires(c.layouts, c.wires, c.cuts, c.order)
                IF.check_inputs(c.layouts, c.cuts, c.wires, c.registered, c.ingress)
                blob = IF.circuit_bytes(c.description())
                bg = sum(1 for x in c.wires if w.family_of(x.reader, x.at) == "BatchedGemm" and x.port == "x")
                r[name] = {"ok": True, "digest": hashlib.sha512(blob).hexdigest(), "bytes": len(blob), "wires": len(c.wires),
                           "wires_before_coalesce": len(w.b.wires), "batched_gemm_x": bg, **{f"mixed_{k}": v for k, v in MIXED.items()}}
            except Exception as e:  # noqa: BLE001 - a refusal is a result
                r[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:1500]}
            r[name]["wall_s"] = round(time.perf_counter() - t0, 1)
        rec[arg] = r
        (out / "bgform.json").write_text(json.dumps(rec, indent=1) + "\n")
        print("DSAMOE bgform", arg, json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
