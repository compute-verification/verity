"""HC-POWER's filler arms (approach pouw/filler-power; note:20261016T0435Z-draft-hc-power): HC-KERN's kern-m2 (dcap.cu's
map 2, W 16, NC 4, S 8192, fill-v1's decode split table) plus the filler during prefill, on a green split of `pre_sms`
SMs. The prefill placement is the only change against kern-m2.

Prefill is the one phase of the suite that runs at the board's 600 W cap (588–605 W read over sustained windows, SM clock
2,085–2,092 MHz). The filler on a few SMs draws less than the linears it displaces, so the prefill on the rest pays its SM
share and no clock (pwr.cu's ppsplit, r20261016-051754-e025): the M = 8,192 linears take 1.038× on 184 SMs and 1.073× on
176, with or without the filler beside them. dcap's gc_split (flags 0) rounds the step partition up to a multiple of 8
SMs, so 4 filler SMs is the only prefill split under the suite's 5% slowdown. fill-v1 and kern-m2 leave prefill without
filler, and it is their lowest entry.

  python3 benchmarks/pouw/filler/score.py benchmarks/pouw/filler/hc_power.py:PowerPre4

kern-m2 is loaded through `arms.load`, which sets DFILL_MAP=2 for every process the scorer starts after loading the arm.
No `from __future__ import annotations`: `arms.load` runs the module without registering it in sys.modules, where
dataclasses looks up string annotations.
"""

from dataclasses import dataclass, field

import arms as A


@dataclass
class PowerPre4:
    """kern-m2 with the filler on 4 SMs during prefill (the linears on 184)."""

    name: str = "power-pre4"
    pre_sms: int = 4
    lineage: dict = field(default_factory=lambda: {
        "parent": "kern-m2 (cursor/hc-kern-ea24@f71ea2b22)", "approach": "pouw/filler-power",
        "mechanism": ["filler:chained-imma-registers", "kernel:map-in-register-order", "kernel:seed-128",
                      "kernel:c-16bit", "kernel:w16-s8192", "place:split", "split:per-point-table", "prefill:split"],
        "hypothesis": "prefill is power-capped and the filler draws less a SM than the linears it displaces, so a "
                      "small prefill split costs only its SM share and lifts the suite's lowest entry"})
    sources: tuple = ("benchmarks/pouw/decode_capacity/dcap.cu", "benchmarks/pouw/filler/hc_kern.py",
                      "benchmarks/pouw/filler/hc_power.py")
    filler: A.Filler | None = None
    choice: str = ("kern-m2's offline table of 11 (B, L) decode splits, and one prefill split chosen on HC-POWER's "
                   "pwr.cu runs; a lookup and a green split at serving time")

    def __post_init__(self):
        self.kern = A.load("benchmarks/pouw/filler/hc_kern.py:KernM2")
        self.filler = self.kern.filler

    def placement(self, phase, b, l):
        if phase == "prefill":
            return A.Placement("split", self.pre_sms)
        return self.kern.placement(phase, b, l)
