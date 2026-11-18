"""Throwaway: AND counts of PearlPlusTileCore_v2 today and of deriving its public noise inside (seed_B, lines, Gram)."""
import json
import os
import subprocess
import sys
import time
import traceback


def build(label):
    from verity.circuits.defs import bind
    from verity.circuits.ops.boolean.versions import boolean_version
    from verity.protocols.resource_accounting.compute.pouw.prover.circuit import inert_bound, pc8, pearl_plus_tile as T
    from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_c_noise as N
    from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_c_forming as F
    from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c as C
    from verity.protocols.resource_accounting.compute.pouw.prover.schemes.pearl_c_device import DEVICES

    kind, *rest = label.split("/")
    if kind == "seedB":
        return bind(N.PearlCSeedBV2)
    if kind == "linekey":
        return bind(N.PearlCLineKeyV2)
    if kind == "lineraw":
        return bind(N.PearlCLineRawV2)
    if kind == "linecodes":
        return bind(N.PearlCLineCodesV2)
    if kind == "noiseline":
        return bind(N.PearlCNoiseLineV2)
    if kind == "rowline":
        return bind(N.PearlCRowLineV2, K=int(rest[0]))
    dev, k = rest[0], int(rest[1])
    d = DEVICES[dev]
    at = dict(K=k, R=32, DOT=pc8.ATOMS[d.atom.name], G=d.g)
    if kind == "chain":
        return bind(F.PearlCChain, K=k, DOT=at["DOT"], G=d.g)
    if kind == "gram":
        return bind(F.PearlCGram, **at)
    if kind == "gramsig":
        return bind(F.PearlCGramV2, **{**at, "DOT": boolean_version(at["DOT"])})
    if kind == "core":
        tm, tn, n = (int(x) for x in rest[2:5])
        scheme = C.PearlC(k_min=k, tile=(tm, tn), forming="v1", device=d, hashing="h3")
        rho, rhor = inert_bound._rho(dev, None), inert_bound.row_rho(scheme, k)
        return bind(T.PearlPlusTileCoreV2, K=k, TM=tm, TN=tn, N=n, RHO=rho, RHOR=rhor, DEV=dev)
    raise ValueError(label)


def one(label, path):
    from verity.circuits.program import Program
    from verity.circuits import census as CS
    from verity.circuits.ops.boolean import versions as V
    t0 = time.time()
    try:
        p = Program(build(label))
        c = CS.census(p, boolean=V.boolean_version)
        work = c["totals"].get("work") or {}
        r = {"id": p.fn.id, "ands": sum(w["and"] for w in work.values()),
             "gates": sum(w["gates"] for w in work.values()), "work": work, "s": round(time.time() - t0, 1)}
    except BaseException as e:  # noqa: BLE001
        r = {"error": f"{type(e).__name__}: {str(e)[:800]}", "tb": traceback.format_exc()[-1500:],
             "s": round(time.time() - t0, 1)}
    json.dump(r, open(path, "w"))


def main(out):
    labels = ["seedB", "linekey", "lineraw", "linecodes", "noiseline", "rowline/256"]
    for dev in ("h100", "sm120"):
        labels += [f"core/{dev}/256/1/1/1", f"chain/{dev}/256", f"gram/{dev}/256", f"gramsig/{dev}/256"]
    for dev in ("h100", "sm120"):
        for k in (2048, 8192):
            labels += [f"chain/{dev}/{k}", f"gram/{dev}/{k}", f"core/{dev}/{k}/1/1/1"]
    for dev in ("h100", "sm120"):
        labels += [f"core/{dev}/2048/64/64/2048", f"core/{dev}/8192/64/64/8192"]
    res = {}
    for lab in labels:
        tmp = os.path.join(os.path.dirname(out), "one.json")
        try:
            rc = subprocess.run([sys.executable, __file__, "--one", lab, tmp], timeout=2400).returncode
            r = json.load(open(tmp)) if rc == 0 else {"error": f"child rc={rc}"}
        except subprocess.TimeoutExpired:
            r = {"error": "timeout 2400 s"}
        res[lab] = r
        print(lab, json.dumps({k: v for k, v in r.items() if k != "tb"})[:600], flush=True)
        json.dump(res, open(out, "w"), indent=1)
    return 0 if "ands" in res.get("core/h100/256/1/1/1", {}) else 1


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(sys.argv[2], sys.argv[3])
    else:
        sys.exit(main(sys.argv[1]))
