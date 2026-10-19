"""What a prover with root can do to one GPU's SM clock through NVML (ctypes, stdlib only), for `clock_ceiling.sh`.

  python clocks.py state UUID                # one JSON line: clocks, limits, VF offsets and their ranges, perf modes
  python clocks.py offset UUID MHZ [--mem]   # set the GPC (or memory) clock VF offset by both NVML calls; rc of each
  python clocks.py lock UUID MIN MAX         # nvmlDeviceSetGpuLockedClocks; MIN and MAX in MHz, or tdp or unlimited

Every set prints {"op", "rc", "error"} and never raises, so a refused knob is a row, not a crash.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys

GRAPHICS, SM, MEM = 0, 1, 2
P0 = 0
#: nvmlClockLimitId_t: special values for nvmlDeviceSetGpuLockedClocks
LIMIT_IDS = {"tdp": 0xFFFFFF01, "unlimited": 0xFFFFFF02}


class ClockOffset(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("type", ctypes.c_int), ("pstate", ctypes.c_int),
                ("clockOffsetMHz", ctypes.c_int), ("minClockOffsetMHz", ctypes.c_int),
                ("maxClockOffsetMHz", ctypes.c_int)]


class PerfString(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("str", ctypes.c_char * 2048)]


def _version(struct: type, v: int = 1) -> int:
    return ctypes.sizeof(struct) | (v << 24)


class Nvml:
    def __init__(self, uuid: str):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self.lib.nvmlErrorString.restype = ctypes.c_char_p
        rc = self.lib.nvmlInit_v2()
        if rc:
            raise SystemExit(f"nvmlInit_v2: {self.err(rc)}")
        self.h = ctypes.c_void_p()
        rc = self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(self.h))
        if rc:
            raise SystemExit(f"nvmlDeviceGetHandleByUUID({uuid}): {self.err(rc)}")

    def err(self, rc: int) -> str:
        return self.lib.nvmlErrorString(rc).decode()

    def call(self, name: str, *args) -> tuple[int, str]:
        fn = getattr(self.lib, name, None)
        if fn is None:
            return -1, "absent from this libnvidia-ml"
        rc = fn(self.h, *args)
        return rc, self.err(rc) if rc else ""

    def uints(self, name: str, n: int, *pre) -> dict:
        out = [ctypes.c_uint() for _ in range(n)]
        rc, e = self.call(name, *pre, *[ctypes.byref(o) for o in out])
        return {"rc": rc, "error": e} if rc else {"value": [o.value for o in out]}

    def ints(self, name: str, n: int, *pre) -> dict:
        out = [ctypes.c_int() for _ in range(n)]
        rc, e = self.call(name, *pre, *[ctypes.byref(o) for o in out])
        return {"rc": rc, "error": e} if rc else {"value": [o.value for o in out]}

    def offsets(self, kind: int) -> dict:
        o = ClockOffset(version=_version(ClockOffset), type=kind, pstate=P0)
        rc, e = self.call("nvmlDeviceGetClockOffsets", ctypes.byref(o))
        if rc:
            return {"rc": rc, "error": e}
        return {"offset": o.clockOffsetMHz, "min": o.minClockOffsetMHz, "max": o.maxClockOffsetMHz}

    def perf_string(self, name: str) -> dict:
        s = PerfString(version=_version(PerfString))
        rc, e = self.call(name, ctypes.byref(s))
        return {"rc": rc, "error": e} if rc else {"value": s.str.decode(errors="replace")}

    def state(self) -> dict:
        u = self.uints
        return {
            "clock_now": {k: u("nvmlDeviceGetClockInfo", 1, t) for k, t in (("graphics", GRAPHICS), ("sm", SM), ("mem", MEM))},
            "clock_max": {k: u("nvmlDeviceGetMaxClockInfo", 1, t) for k, t in (("graphics", GRAPHICS), ("sm", SM), ("mem", MEM))},
            "max_customer_boost": {k: u("nvmlDeviceGetMaxCustomerBoostClock", 1, t) for k, t in (("graphics", GRAPHICS), ("sm", SM))},
            "applications": {k: u("nvmlDeviceGetApplicationsClock", 1, t) for k, t in (("graphics", GRAPHICS), ("mem", MEM))},
            "default_applications": {k: u("nvmlDeviceGetDefaultApplicationsClock", 1, t)
                                     for k, t in (("graphics", GRAPHICS), ("mem", MEM))},
            "p0_min_max": {k: u("nvmlDeviceGetMinMaxClockOfPState", 2, t, P0)
                           for k, t in (("graphics", GRAPHICS), ("sm", SM), ("mem", MEM))},
            "gpc_vf_offset": self.ints("nvmlDeviceGetGpcClkVfOffset", 1),
            "gpc_vf_offset_range": self.ints("nvmlDeviceGetGpcClkMinMaxVfOffset", 2),
            "mem_vf_offset": self.ints("nvmlDeviceGetMemClkVfOffset", 1),
            "mem_vf_offset_range": self.ints("nvmlDeviceGetMemClkMinMaxVfOffset", 2),
            "clock_offsets_p0": {"graphics": self.offsets(GRAPHICS), "mem": self.offsets(MEM)},
            "power_limit_mw": u("nvmlDeviceGetPowerManagementLimit", 1),
            "power_limit_constraints_mw": u("nvmlDeviceGetPowerManagementLimitConstraints", 2),
            "enforced_power_limit_mw": u("nvmlDeviceGetEnforcedPowerLimit", 1),
            "pstate": u("nvmlDeviceGetPerformanceState", 1),
            "perf_modes": self.perf_string("nvmlDeviceGetPerformanceModes"),
            "current_clock_freqs": self.perf_string("nvmlDeviceGetCurrentClockFreqs"),
        }

    def set_offset(self, mhz: int, mem: bool) -> list[dict]:
        rows = []
        old = "nvmlDeviceSetMemClkVfOffset" if mem else "nvmlDeviceSetGpcClkVfOffset"
        rc, e = self.call(old, ctypes.c_int(mhz))
        rows.append({"op": old, "mhz": mhz, "rc": rc, "error": e})
        o = ClockOffset(version=_version(ClockOffset), type=MEM if mem else GRAPHICS, pstate=P0, clockOffsetMHz=mhz)
        rc, e = self.call("nvmlDeviceSetClockOffsets", ctypes.byref(o))
        rows.append({"op": "nvmlDeviceSetClockOffsets", "type": "mem" if mem else "graphics", "mhz": mhz, "rc": rc, "error": e})
        return rows

    def lock(self, lo: str, hi: str) -> dict:
        v = [LIMIT_IDS.get(x, None) or int(x) for x in (lo, hi)]
        rc, e = self.call("nvmlDeviceSetGpuLockedClocks", ctypes.c_uint(v[0]), ctypes.c_uint(v[1]))
        return {"op": "nvmlDeviceSetGpuLockedClocks", "min": lo, "max": hi, "rc": rc, "error": e}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("state")
    s.add_argument("uuid")
    o = sub.add_parser("offset")
    o.add_argument("uuid")
    o.add_argument("mhz", type=int)
    o.add_argument("--mem", action="store_true")
    lk = sub.add_parser("lock")
    lk.add_argument("uuid")
    lk.add_argument("lo")
    lk.add_argument("hi")
    a = p.parse_args(argv)
    n = Nvml(a.uuid)
    if a.cmd == "state":
        print(json.dumps({"op": "state", **n.state()}))
    elif a.cmd == "offset":
        for row in n.set_offset(a.mhz, a.mem):
            print(json.dumps(row))
    else:
        print(json.dumps(n.lock(a.lo, a.hi)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
