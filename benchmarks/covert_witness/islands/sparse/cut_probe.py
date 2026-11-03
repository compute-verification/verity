"""cut_probe RUN_ID...: ρ at the reference cell for te_core 4.0-traced rows, as traced and with the cut a replaced by a
rank-2 or rank-1 embedding adapter's (32 or 16 bits), under formalism v2 and under --charge-context.

The trace takes a as the narrowest cut anywhere on a token's affected path and t (the share that reads no θ′ value)
separately, so a rank-r embedding adapter sets a = 16r on any route, whether or not it trains. Reads the toolsmith's
4.0 rescore (art:7bbd4cd32918, `ingest/measured.json`) and the scorer's 02:01Z refresh (art:34c7249f9054, for each
run's start, family and Δ); `TE_CORE` is a checkout's `benchmarks/covert_witness` holding te_core 4.1 (74b520affe8).
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.environ.get("TE_CORE", "/tmp/te41/benchmarks/covert_witness"))
from te_core import formalism as F
from te_core.gates import per_mac

SHAPES = {"s": dict(d_model=512, n_layer=6, vocab=50304, ctx=1024, parameters=45029888, fmac=51342080),
          "m": dict(d_model=768, n_layer=8, vocab=50304, ctx=1024, parameters=95269632, fmac=107893504)}


def tree(art: str) -> str:
    return subprocess.run(["research", "data", "fetch", art], check=True, capture_output=True,
                          text=True).stdout.strip().splitlines()[-1]


def measured(v: dict, a: float, delta: float, changed, shape: dict) -> F.Measured:
    g = v["gates"]
    part = F.Part(g, v["tokens"], v["sigma"] * g, v["affected_share"] * g, 1, v["delta_grad_bits"])
    t = v["tail_share"] if v["tail_share"] < 1 else 0.0
    u_in = tuple(map(tuple, v["u_in"]))
    return F.Measured((part,), v["tau_ids"] * 16, a, F.with_tail(u_in, t), u_in, delta, None, {}, t,
                      F.context(changed, shape) if changed else None)


def main() -> int:
    meas = json.load(open(tree("art:7bbd4cd32918") + "/ingest/measured.json"))
    info = {}
    for line in open(tree("art:34c7249f9054") + "/formalism.jsonl"):
        d = json.loads(line)
        info.setdefault(d["run_id"], (d["start"], d["family"], d["formalism"]["delta_bits"]))
    for rid in sys.argv[1:]:
        v = meas[rid]
        start, family, delta = info[rid]
        sh = SHAPES[start[0]]
        shape = {k: sh[k] for k in ("d_model", "n_layer", "vocab", "ctx", "parameters")}
        c, k = 4096 * sh["fmac"] * per_mac("gemm/bf16"), 80_000
        changed = ["embed.weight"] + [f"blocks.{i}.wq.weight" for i in range(sh["n_layer"])]
        for label, a in (("as traced", v["a_bits"]), ("rank-2 embed", 32.0), ("rank-1 embed", 16.0)):
            v2 = F.at(measured(v, a, delta, None, shape), c, k)
            ctx = F.at(measured(v, a, delta, changed, shape), c, k)
            print(f"{rid} {start} {family} t={v['tail_share']:.3f} {label}: a={a:.0f} s={v2['s']:.0f} "
                  f"s*={v2['s_star']:.1f} rho={v2['rho']:.3g}; charge-context s*={ctx['s_star']:.1f} "
                  f"rho={ctx['rho']:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
