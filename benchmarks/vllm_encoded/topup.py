"""The top-up (`topup.cu`) through ctypes, as the bandwidth arms start it (`bench_arm.py`): its functions (`TOPUP`),
`Topup` over them, and the run's seed and fill modes."""
from __future__ import annotations

import ctypes
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/word_codec"))

import cuda_lib  # noqa: E402

RUN_SEED = hashlib.sha256(b"verity/kv-filler/bench/seed").digest()
MODES = {"read": -1, "fill": 4}
FILL_WARPS = {"fill": 4, "fill8": 8, "fill16": 16, "fillg": 4}  # fill-sk16/v0 at that many warps a CTA


def mode_of(name: str) -> int:
    return MODES["fill"] if name in FILL_WARPS else MODES[name]


_v, _i, _q, _d = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64, ctypes.c_double
TOPUP = {   # topup.cu's functions (`cuda_lib.load`): the top-up thread's (tu_*) and fill-sk16/v2's (sq_*)
    "tu_start": [_i, _v, _v, _i, _i, _q, _q, _i, _i, _v, _i],
    "tu_pause": [_i],
    "tu_throttle": [_i],
    "tu_gate": [_i, _i],
    "tu_gate_step": [_i, _v],
    "tu_now_ms": ([_v], _d),
    "tu_progress": [_v],
    "tu_log": [_q, _v, _i],
    "tu_read": [_v, _q, _q, _v, _v],
    "tu_fill_units": [_v, _v, _i, _q, _q, _v, _v, _i, _v],
    "tu_fill_signs": [_v, _v],
    "rd_read_reps": [_i, _i, _v, _q, ctypes.c_uint32, _i, _v, _v],
    "tu_page_read": [_v, _q, _q, _q, _q, _i, _q, _i, _v, _v],
    "sq_start": [_i, _v, _v, _i, _v, _i, _i, _q, _i, _i, _i],
    "sq_idle": [_i, _q],
    "sq_step_begin": [_i, _v],
    "sq_step_end": [_v],
    "sq_step_begin_phase": [_i, _i, _i, _v],
    "sq_step_end_phase": [_i, _v],
    "sq_progress": [_v],
    "sq_prefill_reader": [_i],
    "sq_probe": [_i],
    "sq_shape": [_i, _i],
    "sq_now_ms": ([_v], _d),
}


class Topup:
    def __init__(self, path: str) -> None:
        self.so = cuda_lib.load(Path(path).resolve(), TOPUP)

    def fill_units(self, regions, unit0: int, units: int, seed: bytes, sign_seed: bytes, ctas: int):
        ptrs = (ctypes.c_uint64 * len(regions))(*[p for p, _ in regions])
        sizes = (ctypes.c_uint64 * len(regions))(*[n for _, n in regions])
        out = ctypes.create_string_buffer(32 * units + 4)
        rc = self.so.tu_fill_units(ptrs, sizes, len(regions), unit0, units, seed, sign_seed, ctas, out)
        if rc:
            raise RuntimeError(f"tu_fill_units error {rc}")
        raw = out.raw
        return [raw[32 * k:32 * k + 32] for k in range(units)], int.from_bytes(raw[32 * units:32 * units + 4], "little")

    def start(self, regions, mode: int, chunk: int, depth: int, priority: int, seed: bytes, tile: int = 1 << 16,
              ctas: int = 0, device: int = 0, warps: int = 4) -> None:
        if mode == MODES["fill"] and self.so.tu_fill_warps(warps):
            raise RuntimeError(f"tu_fill_warps({warps}) refused")
        ptrs = (ctypes.c_uint64 * len(regions))(*[p for p, _ in regions])
        sizes = (ctypes.c_uint64 * len(regions))(*[n for _, n in regions])
        rc = self.so.tu_start(device, ptrs, sizes, len(regions), mode, chunk, tile, depth, priority, seed, ctas)
        if rc:
            raise RuntimeError(f"tu_start error {rc}")

    def stop(self) -> None:
        rc = self.so.tu_stop()
        if rc:
            raise RuntimeError(f"tu_stop error {rc}")

    def now(self, stream: int) -> float:
        return self.so.tu_now_ms(stream)

    def log(self, first: int = 0) -> list[tuple[float, float, float, int, int, int]]:
        """Every chunk finished (from the `first`-th): its start and end (ms on the GPU's clock since the start event),
        bytes, region, the first 4 bytes of its root as a big-endian integer, and its number."""
        out = []
        buf = (ctypes.c_double * (6 * 65536))()
        while True:
            n = self.so.tu_log(first, buf, 65536)
            out += [(buf[6 * j], buf[6 * j + 1], buf[6 * j + 2], int(buf[6 * j + 3]), int(buf[6 * j + 4]),
                     int(buf[6 * j + 5])) for j in range(n)]
            first += n
            if n < 65536:
                return out
