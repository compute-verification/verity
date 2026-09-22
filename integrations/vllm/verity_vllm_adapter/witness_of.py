"""Witness accounting of a finished reference verify run (RC): the same bytes the selective verifier
records, computed after the fact from ``verify.json``'s per-VU results and the bundle, so the
full-preparation control's row has the same columns as the selective rows.

    PYTHONPATH=. python -m verity_vllm_adapter.witness_of <bundle> <verify.json> --out <json>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from verity_vllm.capture import CaptureReader
from verity_vllm.description import Description
from verity_vllm_adapter.memo_cache import MemoCache
from verity_vllm_adapter.selective_verifier import witness_accounting


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("verify_json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    reader = CaptureReader(Path(args.bundle))
    description = Description(reader.description)
    v = json.loads(Path(args.verify_json).read_text())
    results = v["report"]["results"]
    items = [(r["canonical_id"], r["status"]) for r in results]
    t0 = time.perf_counter()
    w = witness_accounting(description, reader, MemoCache(reader), items)
    w["vus"] = len(items)
    w["seconds"] = time.perf_counter() - t0
    w["source"] = str(args.verify_json)
    Path(args.out).write_text(json.dumps(w, indent=1) + "\n")
    print(json.dumps({k: w[k] for k in ("vus", "openings", "witness_bytes", "unique_witness_bytes", "unique_weights_value_bytes", "seconds")}))


if __name__ == "__main__":
    main()
