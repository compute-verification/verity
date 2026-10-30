"""A probe of the SM clocks the driver reports and the clock offsets it allows, read-only.

Every call is an NVML or nvidia-smi query (no setter), so it runs beside other work. Writes `clock.json` to `$OUT`.
"""

import ctypes
import json
import os
import subprocess

NV = ctypes.CDLL("libnvidia-ml.so.1")
GRAPHICS, SM, MEM = 0, 1, 2


def call(name, *args):
    fn = getattr(NV, name, None)
    if fn is None:
        return "absent"
    return fn(*args)


def uint(name, dev, *pre):
    v = ctypes.c_uint(0)
    rc = call(name, dev, *pre, ctypes.byref(v))
    return v.value if rc == 0 else f"rc={rc}"


class ClockOffset(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("type", ctypes.c_int), ("pstate", ctypes.c_int),
                ("clockOffsetMHz", ctypes.c_int), ("minClockOffsetMHz", ctypes.c_int),
                ("maxClockOffsetMHz", ctypes.c_int)]


def device(i):
    dev = ctypes.c_void_p()
    if call("nvmlDeviceGetHandleByIndex_v2", i, ctypes.byref(dev)) != 0:
        return {"index": i, "error": "no handle"}
    name = ctypes.create_string_buffer(96)
    call("nvmlDeviceGetName", dev, name, 96)
    d = {"index": i, "name": name.value.decode(),
         "max_sm": uint("nvmlDeviceGetMaxClockInfo", dev, SM),
         "max_graphics": uint("nvmlDeviceGetMaxClockInfo", dev, GRAPHICS),
         "max_customer_boost_graphics": uint("nvmlDeviceGetMaxCustomerBoostClock", dev, GRAPHICS),
         "app_graphics": uint("nvmlDeviceGetApplicationsClock", dev, GRAPHICS),
         "default_app_graphics": uint("nvmlDeviceGetDefaultApplicationsClock", dev, GRAPHICS),
         "current_sm": uint("nvmlDeviceGetClockInfo", dev, SM)}
    lo, hi = ctypes.c_uint(0), ctypes.c_uint(0)
    rc = call("nvmlDeviceGetMinMaxClockOfPState", dev, GRAPHICS, 0, ctypes.byref(lo), ctypes.byref(hi))
    d["p0_graphics_min_max"] = [lo.value, hi.value] if rc == 0 else f"rc={rc}"
    off, omin, omax = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)
    rc = call("nvmlDeviceGetGpcClkVfOffset", dev, ctypes.byref(off))
    d["gpc_vf_offset"] = off.value if rc == 0 else f"rc={rc}"
    rc = call("nvmlDeviceGetGpcClkMinMaxVfOffset", dev, ctypes.byref(omin), ctypes.byref(omax))
    d["gpc_vf_offset_min_max"] = [omin.value, omax.value] if rc == 0 else f"rc={rc}"
    for pstate in (0, 2):
        co = ClockOffset(version=ctypes.sizeof(ClockOffset) | (1 << 24), type=GRAPHICS, pstate=pstate)
        rc = call("nvmlDeviceGetClockOffsets", dev, ctypes.byref(co))
        d[f"clock_offsets_p{pstate}"] = ({"offset": co.clockOffsetMHz, "min": co.minClockOffsetMHz,
                                          "max": co.maxClockOffsetMHz} if rc == 0 else f"rc={rc}")
    mem = uint("nvmlDeviceGetMaxClockInfo", dev, MEM)
    if isinstance(mem, int):
        n = ctypes.c_uint(512)
        clocks = (ctypes.c_uint * 512)()
        rc = call("nvmlDeviceGetSupportedGraphicsClocks", dev, mem, ctypes.byref(n), clocks)
        d["supported_graphics_max_at_max_mem"] = max(clocks[: n.value]) if rc == 0 and n.value else f"rc={rc}"
    return d


def main():
    out = {"init": call("nvmlInit_v2")}
    drv = ctypes.create_string_buffer(96)
    call("nvmlSystemGetDriverVersion", drv, 96)
    out["driver"] = drv.value.decode()
    n = ctypes.c_uint(0)
    call("nvmlDeviceGetCount_v2", ctypes.byref(n))
    out["devices"] = [device(i) for i in range(n.value)]
    out["nvidia_smi_query"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,pci.device_id,pci.sub_device_id,vbios_version,clocks.max.sm,"
         "clocks.max.graphics,clocks.sm,clocks.applications.graphics,clocks.default_applications.graphics",
         "--format=csv"], capture_output=True, text=True).stdout
    out["nvidia_smi_clock"] = subprocess.run(["nvidia-smi", "-q", "-d", "CLOCK", "-i", "0"],
                                             capture_output=True, text=True).stdout
    text = json.dumps(out, indent=1)
    print(text)
    with open(os.path.join(os.environ.get("OUT", "."), "clock.json"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
