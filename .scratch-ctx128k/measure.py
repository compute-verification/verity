"""CTX128K (CIR-186): the DSA indexer's circuit at rows up to 131,072 keys, read off the Definitions (`ir_lower.definition_ands`,
never lowered), the host-side Waksman settings' time, and the mask-form prototype's counts.  Writes $OUT/measure.json as it goes."""
from __future__ import annotations

import json
import os
import random
import resource
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(__file__))

OUT = os.environ.get("OUT", ".")
RES: dict = {"items": []}


def save():
    with open(os.path.join(OUT, "measure.json"), "w") as f:
        json.dump(RES, f, indent=1)


def item(name: str, fn):
    t0 = time.time()
    rec: dict = {"name": name}
    try:
        rec["value"] = fn()
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc()[-3000:]
    rec["seconds"] = round(time.time() - t0, 2)
    rec["maxrss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    RES["items"].append(rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "trace"}), flush=True)
    save()


def main():
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_dsa as BD
    from verity.ml.definitions.registry import boolean_dsa_topk as BT
    from verity.ml.definitions.registry import dsa_topk as T
    from verity_flock.ir_lower import definition_ands as ands

    import mask_form as MF

    K = 2048
    item("DsaMqaLogit_v4{H=32,D=128}", lambda: ands(bind(BD.DsaMqaLogitV4, H=32, D=128)))
    for N in (4096, 32768, 131072):
        item(f"DsaMqaLogits_v4{{N={N},H=32,D=128}}", lambda N=N: ands(bind(BD.DsaMqaLogitsV4, N=N, H=32, D=128)))
    for N in (2049, 32768, 131072):
        item(f"waksman_switches({N},{K})", lambda N=N: T.waksman_switches(N, K))
        item(f"DsaWaksmanRows_v2{{N={N},K={K},W=576,E=16}}", lambda N=N: ands(bind(BT.DsaWaksmanRowsV2, N=N, K=K, W=576, E=16)))
        item(f"DsaWaksmanOrder_v2{{N={N},TOPK={K}}}", lambda N=N: ands(bind(BT.DsaWaksmanOrderV2, N=N, TOPK=K)))
        item(f"DsaTopKOneHot_v1{{N={N}}}", lambda N=N: ands(bind(BT.DsaTopKOneHot, N=N)))

        def settings(N=N):
            rng = random.Random(N)
            order = sorted(rng.sample(range(N), K))
            t0 = time.process_time()
            sw = T.waksman_settings(order, N)
            return {"cpu_seconds": round(time.process_time() - t0, 3), "bits": len(sw)}
        item(f"waksman_settings time N={N}", settings)
    for N in (2049, 32768, 131072):
        item(f"DsaTopKVerdict_v1{{N={N},TOPK={K}}}", lambda N=N: ands(bind(BT.DsaTopKVerdict, N=N, TOPK=K)))
        item(f"DsaTopKLongRow_v2{{N={N},TOPK={K}}}", lambda N=N: ands(bind(BT.DsaTopKLongRowV2, N=N, TOPK=K)))
    for N, V in ((2049, 16), (4096, 8), (32768, 4), (131072, 2)):
        item(f"mask_form N={N} TOPK={K}", lambda N=N, V=V: MF.run(N, K, V=V, seed=N))


if __name__ == "__main__":
    main()
