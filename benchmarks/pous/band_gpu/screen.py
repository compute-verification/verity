"""A decode-only screen of the band's kernel settings on one device: random `C` and tags (a decode's time does not depend on
the data), each setting's decode of every block checked byte for byte against the run of record's (`harness.FROZEN_KERNELS`)
and timed with CUDA events.  An untimed throughput screen that ranks settings; `bench.py` (the kernel gates, encode, the
synthetic loop) measures the one kept.  Exit 1 when a setting's decode differs from the run of record's.

    python benchmarks/pous/band_gpu/screen.py --build-dir DIR --settings '[{"chain_pairs": true}]' --out screen.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from band_gpu import dense, harness, native  # noqa: E402


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--settings", required=True, help="a JSON list of kernel settings over harness.FROZEN_KERNELS")
    ap.add_argument("--segments", type=int, default=30, help="segments of 512 blocks of 64 KiB (30: the 0.5B weight set's)")
    ap.add_argument("--d", type=int, default=12, help="the band's in-degree")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    native.load(a.build_dir)
    dev = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev).manual_seed(0)
    c = torch.randint(0, 256, (a.segments, 512, 65536), dtype=torch.uint8, device=dev, generator=gen)
    tags = torch.randint(0, 256, (a.segments, 32), dtype=torch.uint8, device=dev, generator=gen)

    def decode(settings: dict) -> tuple[torch.Tensor, list[float]]:
        before = harness.set_kernels({})
        try:
            harness.set_kernels({**harness.FROZEN_KERNELS, **settings})
            out = dense.decode_segments(c, tags, a.d)
            times = []
            for _ in range(a.reps):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                dense.decode_segments(c, tags, a.d)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end) / 1e3)
            return out, times
        finally:
            harness.set_kernels(before)

    ref, ref_times = decode({})
    rows = [{"settings": {}, "equal": True, "decode_s": ref_times, "median_s": statistics.median(ref_times)}]
    for settings in json.loads(a.settings):
        out, times = decode(settings)
        rows.append({"settings": settings, "equal": bool(torch.equal(out, ref)), "decode_s": times,
                     "median_s": statistics.median(times)})
        del out
        print(json.dumps(rows[-1]), flush=True)
    base = rows[0]["median_s"]
    for row in rows:
        row["speedup"] = base / row["median_s"]
    doc = {"label": "untimed throughput screen: random C, decode only, against FROZEN_KERNELS",
           "device": harness.gpu_facts(), "segments": a.segments, "d": a.d, "blocks": a.segments * 512, "rows": rows}
    Path(a.out).write_text(json.dumps(doc, indent=1))
    return 0 if all(row["equal"] for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
