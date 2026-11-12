"""`boolean_version` of the DSA norm's three word Definitions at GLM-5.3's served statics, today and with the three
``opt_in`` marks cleared (the switch's part (b)), and the Boolean versions' descriptor ids.  Builds no circuit.  The
process exits itself past ``--max-gb`` of RSS.  One JSON object on stdout."""
import json
import os
import resource
import sys
import threading
import time

MAX_GB = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0


def guard():
    while True:
        rss = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
        if rss > MAX_GB:
            print(f"GUARD: RSS {rss:.2f} GB past {MAX_GB} GB", file=sys.stderr, flush=True)
            os._exit(9)
        time.sleep(0.2)


threading.Thread(target=guard, daemon=True).start()
t0 = time.monotonic()

from verity.circuits.codec import descriptor_id  # noqa: E402
from verity.circuits.defs import bind  # noqa: E402
from verity.definitions import boolean as BB  # noqa: E402
from verity.definitions.registry import boolean_dsa_norm as BN  # noqa: E402
from verity.definitions.registry import dsa_norm as DN  # noqa: E402

EPS = 1e-5
WORDS = {"DsaRMSNorm{N=2048}": bind(DN.DsaRMSNorm, N=2048, EPS=EPS), "DsaRMSNorm{N=512}": bind(DN.DsaRMSNorm, N=512, EPS=EPS),
         "DsaRMSNormSumSq{N=2048}": bind(DN.DsaRMSNormSumSq, N=2048, EPS=EPS),
         "DsaRMSNormSumSq{N=512}": bind(DN.DsaRMSNormSumSq, N=512, EPS=EPS), "DsaRMSNormApply{G=64}": bind(DN.DsaRMSNormApply, G=64)}


def versions() -> dict[str, str]:
    BB._TABLE.clear()
    BB._VERSIONS.clear()
    return {k: descriptor_id(BB.boolean_version(w)) for k, w in WORDS.items()}


errors = BB.load()
before = versions()
for d in (BN.DsaRMSNormSumSqV3, BN.DsaRMSNormApplyV3, BN.DsaRMSNormV3):
    d.opt_in = False
after = versions()
json.dump({"load_errors": errors, "before": before, "after": after, "seconds": round(time.monotonic() - t0, 1),
           "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}, sys.stdout, indent=1)
