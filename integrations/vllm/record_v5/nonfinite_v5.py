#!/usr/bin/env python
"""[A9 contract 2026-09-10] W-A56-1 -- per RU, the count of COMMITTED BF16 / FP32 (/ FP16) words whose exponent field is all ones
(non-finite: inf / NaN).  The tc checker family rejects such words by declaration (``tc.rs:9-12``), so a non-zero count in a committed
interior is an HC risk to report, not to hide.

One pass over the committed RU trees of J (``adapter/selective/ru/<request_id>/<step>/values.bin``): every ``values.bin`` is memmapped once, every float leaf's bytes are
viewed as words and the exponent mask tested in numpy; derived (prescribed) leaves hold no bytes and are skipped (counted).  Cost = a read
of the captured bytes (D4: 10.43 GB for 2 RUs at 6.5K); ``--budget-s`` caps it -- when exceeded the remaining RUs are written
``UNAVAILABLE`` with the reason, never estimated.  Output ``REC/nonfinite_v5.json`` and a ``costlog.jsonl`` row (class ``other_verifier``:
this is a verifier-side inspection outside the eight classes).

    python -m record_v5.nonfinite_v5 --record REC [--budget-s 900]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from record_v5 import deps

deps.install()

from verity_vllm.capture import CaptureReader  # noqa: E402

MASKS = {"bfloat16": (np.uint16, 7, 0xFF), "float16": (np.uint16, 10, 0x1F), "float32": (np.uint32, 23, 0xFF)}


def scan_ru(ru, per_dtype: dict[str, int], per_op: dict[str, int]) -> tuple[int, int, int]:
    """(words scanned, non-finite words, leaves skipped) for one RU step."""
    words = nonfin = skipped = 0
    raw = ru._raw()  # the memmap, read-only (RUReader keeps one per step)
    for e in ru.index.entries:
        if e.derived is not None:
            skipped += 1
            continue
        m = MASKS.get(e.dtype)
        if m is None:
            continue
        view_t, shift, mask = m
        buf = np.frombuffer(raw[e.offset:e.offset + e.nbytes], dtype=view_t)
        exp = (buf >> np.array(shift, dtype=view_t)) & np.array(mask, dtype=view_t)
        n = int(np.count_nonzero(exp == mask))
        words += buf.size
        if n:
            nonfin += n
            per_dtype[e.dtype] += n
            per_op[e.canonical_id.split("/")[0] if "/" in e.canonical_id else e.canonical_id.rsplit(".", 1)[0]] += n
    return words, nonfin, skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--record", required=True)
    ap.add_argument("--capture", default=None, help="capture dir (default REC/adapter/selective)")
    ap.add_argument("--budget-s", type=float, default=900.0)
    a = ap.parse_args(argv)
    rec = Path(a.record)
    cap_dir = Path(a.capture) if a.capture else rec / "adapter" / "selective"  # A3's selective prover: capture.json + ru/<request_id>/<step>/values.bin here
    out_path = rec / "nonfinite_v5.json"
    if not cap_dir.exists():
        out_path.write_text(json.dumps({"schema": "veritor/a9/nonfinite-v5/v1", "status": "UNAVAILABLE", "reason": f"no committed interiors at {cap_dir} (phase 2 not reached)"}, indent=1) + "\n")
        print(f"[nonfinite_v5] UNAVAILABLE: {cap_dir} missing")
        return 0
    reader = CaptureReader(cap_dir)
    t0 = time.perf_counter()
    per_request: dict[str, dict[str, Any]] = {}
    status = "MEASURED"
    for rid in reader.request_ids():
        keys = [k for k in reader.ru_keys(rid)]
        row = {"ru_steps": len(keys), "words_scanned": 0, "nonfinite_words": 0, "leaves_prescribed_skipped": 0, "per_dtype": defaultdict(int), "per_op": defaultdict(int), "status": "MEASURED"}
        for (r, step) in keys:
            if time.perf_counter() - t0 > a.budget_s:
                row["status"] = f"UNAVAILABLE beyond step {step}: budget {a.budget_s}s exceeded"
                status = "PARTIAL"
                break
            ru = reader.ru(r, step)
            w, n, s = scan_ru(ru, row["per_dtype"], row["per_op"])
            row["words_scanned"] += w
            row["nonfinite_words"] += n
            row["leaves_prescribed_skipped"] += s
            ru._mm = None
        row["per_dtype"] = dict(row["per_dtype"])
        row["per_op"] = dict(sorted(row["per_op"].items(), key=lambda kv: -kv[1])[:20])
        per_request[rid] = row
        if status == "PARTIAL":
            break
    secs = round(time.perf_counter() - t0, 2)
    scanned = {rid for rid in per_request}
    for rid in reader.request_ids():
        if rid not in scanned:
            per_request[rid] = {"status": f"UNAVAILABLE: budget {a.budget_s}s exceeded before this RU", "nonfinite_words": None}
    total = sum(int(r["nonfinite_words"] or 0) for r in per_request.values())
    doc = {"schema": "veritor/a9/nonfinite-v5/v1", "status": status, "seconds": secs, "capture": str(cap_dir), "rule": "exponent field all ones (BF16: bits 14..7; FP32: bits 30..23; FP16: bits 14..10)",
           "why": "W-A56-1: the tc checker family rejects non-finite words by declaration (tc.rs:9-12); a non-zero count in a committed interior is an HC risk",
           "per_ru": per_request, "total_nonfinite_words": total, "hc_risk": total > 0}
    out_path.write_text(json.dumps(doc, indent=1) + "\n")
    with (rec / "costlog.jsonl").open("a") as f:
        f.write(json.dumps({"stage": "W-A56-1 non-finite word scan over the committed RU trees of J (numpy over memmapped values.bin)", "class": "other_verifier", "seconds": secs,
                            "batch": f"{len(scanned)} RU(s)", "label": "MEASURED", "note": "verifier-side inspection outside the eight classes; not summed into the protocol total"}) + "\n")
    print(f"[nonfinite_v5] {status}: {total} non-finite words over {len(scanned)} RU(s) in {secs}s -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
