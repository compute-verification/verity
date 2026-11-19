import argparse, json, math, sys, time, traceback
from collections import Counter, defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=27)
ap.add_argument("--floor", type=int, default=25)
ap.add_argument("--group", type=int, default=16)
a = ap.parse_args()
OUT = Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
import faulthandler
_st = open(OUT / f"stacks-{a.mode}.txt", "w")
faulthandler.enable(file=_st, all_threads=True)
faulthandler.dump_traceback_later(300, repeat=True, file=_st)
sys.setrecursionlimit(100000)

from verity.circuits import laying as LY
from verity.circuits import partition_object as PO
from verity.circuits.codec import descriptor_id
from verity.commitments.periods import interface as I

RES = {"mode": a.mode, "k": a.k, "floor": a.floor, "group": a.group}


def save():
    (OUT / f"{a.mode}.json").write_text(json.dumps(RES, indent=1, default=str))


def k_log(ins, outs, ands):
    return I.proof_block_k_log([-(-b // 8) for b in ins], [-(-b // 8) for b in outs], ands)


def fits(ins, outs, ands):
    return len(ins) + len(outs) <= I.MAX_ROWS and k_log(ins, outs, ands) <= a.k


def m_of(ins, outs, ands, units):
    return k_log(ins, outs, ands) + max(3, math.ceil(math.log2(max(1, -(-units // a.group)))))


def enough(ins, outs, ands, units):
    return m_of(ins, outs, ands, units) >= a.floor


def program():
    if a.mode == "tile":
        from verity.circuits.defs import CompositeDefinition, bind
        from verity.circuits.program import Program
        from verity.protocols.resource_accounting.compute.pouw.prover import schemes as PS
        from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_tile as PT
        core = PT.core(PS.get("pearl-c-h100-v1-h3"), 2048, 16, 64, 2048, 16)
        params, ret = core.signature()
        root = CompositeDefinition("PearlPlusCreditedProbe", 1, (), lambda S_: (params, ret),
                                   lambda B, S_, *x: B.call(core, *x), register=False)
        return Program(root, registered=[p for p, _ in params])
    sys.path.insert(0, "tests/adapters/vllm")
    from verity_vllm.program.predict import window_circuit as WCirc
    from verity_vllm.program.predict import window_class as WC
    from tests.program.predict import test_steps as S
    cfg = json.loads(Path("tests/adapters/vllm/tests/program/predict/fixtures/llama32-1b.config.json").read_text())
    reg = S.Registered.of(S.Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg}, target=dict(S.RTXPRO6000)), default_seed=7)
    circuit = WCirc.lower(reg, WC.WindowClass(4, 27, 2), None, requests=[WC.Request(0, j, 27, 2) for j in range(4)]).circuit
    return circuit.program()[0]


def describe(m, blocks, tid):
    per = defaultdict(lambda: [0, 0])
    for lo, hi in blocks:
        ts = {tid[u] for u in range(lo, hi)}
        for u in range(lo, hi):
            per[tid[u]][0] += 1
            per[tid[u]][1] += len(ts) > 1
    costs = [m.block(lo, hi) for lo, hi in blocks]
    ms = [m_of(*c, hi - lo) for c, (lo, hi) in zip(costs, blocks)]
    return {"blocks": len(blocks), "under_floor": sum(x < a.floor for x in ms),
            "m_hist": dict(sorted(Counter(ms).items())), "k_log_hist": dict(sorted(Counter(k_log(*c) for c in costs).items())),
            "mixed_blocks": sum(len({tid[u] for u in range(lo, hi)}) > 1 for lo, hi in blocks),
            "templates_with_units_in_mixed_blocks": {t: {"units": n, "in_mixed": x} for t, (n, x) in sorted(per.items()) if x}}


try:
    t = time.time()
    P0 = program()
    RES["build_s"] = round(time.time() - t, 1)
    save()
    t = time.time()
    P1 = LY.lay(P0, fits)
    q = LY.query(P1)
    m = LY.Measure(P1, q)
    tid = [descriptor_id(P1.root.body.nodes[k].fn) for k, _ in m.units]
    RES.update(lay_s=round(time.time() - t, 1), units=len(m.units), templates=len(q["params"]["templates"]),
               units_of=dict(sorted(Counter(tid).items())))
    save()
    t = time.time()
    plain = LY.pack(P1, fits, q, m)
    RES["plain"] = {**describe(m, plain, tid), "s": round(time.time() - t, 1)}
    save()
    t = time.time()
    try:
        floored = LY.pack(P1, fits, q, m, enough=enough)
        RES["floored"] = {**describe(m, floored, tid), "s": round(time.time() - t, 1),
                          "validate": PO.validate(PO.build(P1, q, [floored])).ok}
    except LY.Unlaid as e:
        RES["floored"] = {"refused": str(e)[:1000], "s": round(time.time() - t, 1)}
except Exception as e:
    RES["error"] = traceback.format_exc()[-4000:]
save()
print(json.dumps(RES)[:3000])
