"""W-A56-2 on the record: the served argmax rule of the pinned instance, read from the committed words.

For every committed step of every retained request in a REC archive (adapter/selective/ru/<request>/<step>/) the RU tree
holds the lm_head logits row (`l=lm_head/op=lm_head.gemm/port=0`, bf16 [V]) and the served token (`l=sampler/op=
sampler.select/port=0`, int32).  We compare the served token with first-max / last-max over the row and count exact ties at
the maximum, NaNs and infinities.  Rows with ties discriminate the tie rule; rows with NaNs would discriminate the NaN rule.

    PYTHONPATH=vllm-poc:src .venv/bin/python out/contract/ck-elem/argmax_rule/measure_archive.py <REC dir> [--out rows.json]

MEASURED on REC-P1-B0 (2026-09-10): 48 rows, served == first_max in 48/48, 4 rows tie at the max (2, 3, 2, 2 tokens) and every one
served the LOWEST index; 0 rows with NaN or inf.  Peak memory ~150 MB (declare 1 GB through lane_run).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
for p in (REPO / "vllm-poc", REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from verity_vllm.capture import CaptureReader  # noqa: E402


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def measure(rec: Path) -> list[dict]:
    sel = rec / "adapter" / "selective"
    reader = CaptureReader(sel)
    rows: list[dict] = []
    for rd in sorted((sel / "ru").iterdir()):
        steps = sorted(int(p.name) for p in rd.iterdir() if p.is_dir())
        if not steps:
            continue
        boundary = reader.boundary(rd.name)
        for step in steps:
            ru = reader.ru(rd.name, step)
            logits_id = [i for i in ru.ids() if "/op=lm_head.gemm/port=0" in i]
            select_id = [i for i in ru.ids() if "/op=sampler.select/port=0" in i]
            if len(logits_id) != 1 or len(select_id) != 1:
                continue
            t = int(logits_id[0].split("/t=")[1].split("/")[0])
            logits = bf16_to_f32(ru.array(logits_id[0]))
            served = int(ru.array(select_id[0]))
            finite = logits[np.isfinite(logits)]
            m = float(np.nanmax(logits)) if logits.size else float("nan")
            srt = np.sort(finite)
            rows.append(
                {
                    "request": rd.name,
                    "step": step,
                    "t": t,
                    "phase": logits_id[0].split("/ph=")[1].split("/")[0],
                    "V": int(logits.size),
                    "served_interior": served,
                    "served_boundary": int(boundary.sampled_token_at(t)),
                    "first_max": int(np.argmax(logits)),
                    "last_max": int(logits.size - 1 - np.argmax(logits[::-1])),
                    "ties_at_max": int(np.sum(logits == m)),
                    "nan": int(np.isnan(logits).sum()),
                    "inf": int(np.isinf(logits).sum()),
                    "max": m,
                    "second": float(srt[-2]) if srt.size > 1 else None,
                }
            )
    return rows


def summary(rows: list[dict]) -> dict:
    ties = [r for r in rows if r["ties_at_max"] > 1]
    return {
        "rows": len(rows),
        "served_equals_first_max": sum(r["served_interior"] == r["first_max"] for r in rows),
        "served_equals_last_max": sum(r["served_interior"] == r["last_max"] for r in rows),
        "interior_equals_boundary": sum(r["served_interior"] == r["served_boundary"] for r in rows),
        "rows_with_ties_at_max": len(ties),
        "tie_rows_served_lowest_index": sum(r["served_interior"] == r["first_max"] for r in ties),
        "tie_rows_served_highest_index": sum(r["served_interior"] == r["last_max"] for r in ties),
        "rows_with_nan": sum(r["nan"] > 0 for r in rows),
        "rows_with_inf": sum(r["inf"] > 0 for r in rows),
        "tie_rule": (
            "MEASURED: lowest index" if ties and all(r["served_interior"] == r["first_max"] for r in ties)
            else "MEASURED: not lowest index" if ties else "UNAVAILABLE: no tie rows in the record"
        ),
        "nan_rule": "UNAVAILABLE: no NaN rows in the record" if not any(r["nan"] for r in rows) else "see rows",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rec", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    rows = measure(a.rec)
    s = summary(rows)
    print(json.dumps(s, indent=1))
    if a.out:
        a.out.write_text(json.dumps({"record": str(a.rec), "summary": s, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
