"""The description size (`adapters/vllm/tools/description_sizes.py`'s `description` and `size`, as `program.json` holds it) of
each template the switch names in place of today's: the long row's cut with its five units' `_v2`s, and the DSA norm's
`_v3`s as the partition files would name them.  One JSON line a template, as each finishes; exits itself past ``MAX_GB``."""
import importlib.util
import json
import os
import sys
import threading
import time

MAX_GB = float(sys.argv[1])
ROOT = sys.argv[2]


def guard():
    while True:
        rss = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
        if rss > MAX_GB:
            print(json.dumps({"guard": f"RSS {rss:.2f} GB past {MAX_GB}"}), flush=True)
            os._exit(9)
        time.sleep(0.1)


threading.Thread(target=guard, daemon=True).start()

spec = importlib.util.spec_from_file_location("description_sizes", f"{ROOT}/adapters/vllm/tools/description_sizes.py")
DS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DS)

from verity.circuits.codec import descriptor_id  # noqa: E402
from verity.circuits.defs import bind  # noqa: E402
from verity.definitions.registry import boolean_dsa_norm as BN  # noqa: E402
from verity.definitions.registry import dsa_topk_mask_cut as X  # noqa: E402
from verity.definitions.registry import dsa_topk_reads as R  # noqa: E402

today = set(DS.long_row_units())
R.DsaMaskPositions, R.DsaPositionsMerge = R.DsaMaskPositionsV2, R.DsaPositionsMergeV2
X.DsaTopKMaskWhole, X.DsaTopKDigitPick, X.DsaTopKMaskBlock = X.DsaTopKMaskWholeV2, X.DsaTopKDigitPickV2, X.DsaTopKMaskBlockV2
X.plan.cache_clear()
DS.long_row_units.cache_clear()
new = [(t, u) for t, (u, _) in DS.long_row_units().items() if t not in today]
EPS = 1e-5
new += [(descriptor_id(d), d) for d in (bind(BN.DsaRMSNormApplyV3, G=64), bind(BN.DsaRMSNormSumSqV3, N=512, EPS=EPS),
                                        bind(BN.DsaRMSNormSumSqV3, N=2048, EPS=EPS))]
only = sys.argv[3:]
for t, u in sorted(new, key=lambda x: x[0]):
    if only and not any(o in t for o in only):
        continue
    t0 = time.monotonic()
    desc = DS.description(u)
    n = DS.size(desc)
    print(json.dumps({"template": t, "bytes": n, "past": n > DS.SIZE, "largest": DS.largest(desc),
                      "seconds": round(time.monotonic() - t0, 2)}), flush=True)
