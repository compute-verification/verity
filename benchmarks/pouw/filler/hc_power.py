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


#: kern-m2's decode splits with an entry for every scored decode point, raised where HC-POWER's split-only sweep with the
#: scorer's gc_split (pwr.cu, pattern 7; r20261016-050639-8537 and r20261016-052600-c1e9) kept step + filler <= 1.040: the
#: largest measured split under that, else kern-m2's. One 8-SM step past each raised split costs 1.050–1.086, and below
#: 128 step SMs every B 132–256 point costs 1.066–1.121 (cuBLASLt's partition cliff), so those keep 60.
POWER_SPLIT = {(1, 2048): 108, (1, 8192): 108, (1, 32768): 108, (32, 1024): 92, (32, 2048): 108, (32, 8192): 124,
               (32, 24576): 132, (48, 16384): 132, (128, 2048): 92, (128, 6144): 108, (132, 5120): 60, (144, 5120): 60,
               (256, 1024): 60, (256, 2048): 60, (256, 3072): 60}


@dataclass
class PowerSplit(PowerPre4):
    """power-pre4 with the decode splits of `POWER_SPLIT`: four points take 8–16 more filler SMs."""

    name: str = "power-split"
    lineage: dict = field(default_factory=lambda: {
        "parent": "kern-m2 (cursor/hc-kern-ea24@f71ea2b22)", "approach": "pouw/filler-power",
        "mechanism": ["filler:chained-imma-registers", "kernel:map-in-register-order", "kernel:seed-128",
                      "kernel:c-16bit", "kernel:w16-s8192", "place:split", "split:per-point-table",
                      "split:every-scored-point", "prefill:split"],
        "hypothesis": "power binds nowhere, so the split's SM share is the only cost: the largest split a sustained "
                      "sweep keeps under 1.04 at every scored point, and 4 filler SMs during prefill"})
    choice: str = ("an offline table of 15 (B, L) decode splits from HC-POWER's split-only sweep and one prefill split; "
                   "a lookup and a green split at serving time")

    def placement(self, phase, b, l):
        if phase == "prefill":
            return A.Placement("split", self.pre_sms)
        return A.Placement("split", A.point_lookup(POWER_SPLIT, b, l))
