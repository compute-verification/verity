"""`adapters/vllm/tests/pipeline/test_served_unit_rows.py`'s digest of the served chunked templates' statements, computed as the
test computes it, on the three partition files as they are and with the DSA norm's `_v2` templates named as their `_v3`s
(the switch's part (b) carried into the partition files).  The process exits itself past ``--max-gb`` of RSS."""
import hashlib
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

MAX_GB = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd()


def guard():
    while True:
        rss = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
        if rss > MAX_GB:
            print(f"GUARD: RSS {rss:.2f} GB past {MAX_GB} GB", file=sys.stderr, flush=True)
            os._exit(9)
        time.sleep(0.2)


threading.Thread(target=guard, daemon=True).start()
t0 = time.monotonic()

from verity.commitments.windows import interface as IF  # noqa: E402
from verity_vllm.pipeline import serving_rows as SRW, weight_rows as WT  # noqa: E402

SERVED = [ROOT / "benchmarks/one_stage" / f"glm53_{p}.json" for p in ("all_families_tp8_step16", "sparse_mla_tp8_step16", "prefill_tp8_step0")]
PINNED = ("eed75699e3df3e72e98379acfda035f1b2a72d19482b103a1f0d1a8e083833fc"
          "083efb3005628542c59f613d48082ae5b1ff58f2df9eed86cd8646e380d7110a")
RENAME = {"DsaRMSNormSumSq_v2{": "DsaRMSNormSumSq_v3{", "DsaRMSNormApply_v2{": "DsaRMSNormApply_v3{"}

IF.templates_of = lambda _w: []
WT.reads = lambda _ms, _rows, _units: None
IF.circuit_record = lambda _w, _name, _templates, sts, **_kw: sts


def digest(tmpls):
    members = [SimpleNamespace(template=t, schemas=True) for t in tmpls]
    sts = SRW.circuit_record(SimpleNamespace(members=members, row_tables=()), [])
    return len(sts), hashlib.sha512(IF.circuit_bytes(sts)).hexdigest()


def renamed(t):
    for a, b in RENAME.items():
        t = t.replace(a, b)
    return t


now = sorted({t for f in SERVED for t in json.loads(f.read_text())["chunk_cap"]})
switched = sorted({renamed(t) for t in now})
n0, d0 = digest(now)
n1, d1 = digest(switched)
json.dump({"pinned": PINNED, "today": {"count": n0, "sha512": d0, "equals_pin": d0 == PINNED},
           "switched": {"count": n1, "sha512": d1}, "renamed": sorted(set(switched) - set(now)),
           "seconds": round(time.monotonic() - t0, 1),
           "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}, sys.stdout, indent=1)
