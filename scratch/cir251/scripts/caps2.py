import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from verity.protocols.spot.verifier.one_stage import proofs as PF
from verity_flock import circuit as FC
from verity_flock.class_statement import UnitStager
from verity_vllm.pipeline import chunk_rows as CR, serving_rows as SRW

FILES = ["glm53_all_families_tp8_step16", "glm53_prefill_tp8_step0", "glm53_prefill_tp8_step0_core"]
BIND = {"partition": "0" * 128, "program": "0" * 128}
caps: dict[str, set[int]] = {}
for f in FILES:
    for t, c in json.loads(Path(f"benchmarks/one_stage/{f}.json").read_text())["chunk_cap"].items():
        if t.startswith("DsaRMSNorm") and f.endswith("step16"):
            caps.setdefault(t, set()).add(c)


def rows(in_rows, n, how):
    if how == "zero":
        return {r: np.zeros((n, b // 16), dtype="<u2") for r, _s, b, _ps in in_rows}
    rng = np.random.default_rng(11)
    return {r: np.packbits(rng.integers(0, 2, (n, b), dtype=np.uint8), axis=1, bitorder="little").view("<u2")
            for r, _s, b, _ps in in_rows}


bad = 0
for t in sorted(caps):
    tp = SRW.template(t)
    in_rows, out_rows = CR.unit_rows(tp)
    for n in sorted(caps[t]):
        t0 = time.time()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            st = UnitStager(tp.fn, f"unit/{t}", in_rows, out_rows, circuit=d / "circuit.txt", bind=BIND, **CR.statement(tp))
            got, how = None, None
            for how in ("random", "zero"):
                try:
                    got = st.stage(rows(in_rows, n, how), d / how, b"k" * 32, indices=list(range(n)))
                    break
                except AssertionError as e:
                    print(f"  {t} n={n} {how} rows: {e}", flush=True)
            if got is None:
                bad += 1
                print(json.dumps({"template": t, "cap": n, "staged": False}), flush=True)
                continue
            k_log, vpb = int(st.meta["k_log"]), int(st.meta["vus_per_block"])
            blocks = -(-n // vpb)
            m = k_log + max(0, (blocks - 1).bit_length())
            why = PF.ks_form_why(d / "circuit.txt", d / how / "pub.bin")
            ok = why is None and m <= CR.M_MAX and k_log <= FC.K_MAX and n <= CR.chunk_cap(tp)
            bad += not ok
            print(json.dumps({"template": t, "cap": n, "rows": how, "k_log": k_log, "vus_per_block": vpb, "blocks": blocks,
                              "m": m, "M_MAX": CR.M_MAX, "K_MAX": FC.K_MAX, "ks_form_why": why, "meta": got["meta"],
                              "ok": ok, "seconds": round(time.time() - t0, 1)}, default=str), flush=True)
sys.exit(1 if bad else 0)
