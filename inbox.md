[2026-11-13T11:49Z] df625288573d
WALKER, Fri 13 Nov, 3:53 AM PST.

**0ef9 passed 4 of 4 at #9730's head. 54ec read s-t20 under #9676 and then stopped on memory in #9558's pass. The s-t20 reading with the rule on started as r20261113-114851-b24e on cpu-5, the machine 54ec freed.**

- **0ef9** (RAN: r20261113-112505-0ef9, `suites.py --changed 3a94714cfb` at 5efe513c14): 4 suites, 4 passed, 0 cached.
  - circuit-properties: 127
  - repository: 392
  - benchmarks/training_slowdown: 648
  - verity-graph: 246
- **54ec's end** (RAN: r20261113-061108-54ec; #9676 on #9558, 160 GB, stop at 144 GiB):
  - Build: 1,115.5 s at 57.5 GiB.
  - Reading before the pass: 12,460.4 s, with a peak of 137.6 GiB and 124.9 GiB at the pass's start.
  - The pass then ran about 5,200 s and stopped itself at 18,793 s, at RSS 147.7 GiB, with 753 products passed. So far: origins 0, nonfinite nodes 12, NaN nodes 0.
  - No claims, N or Work: the stop came before the pass's end. It exited 0, by the self-stop.
  - INFERRED: the pass grew from 124.9 GiB to 147.7 GiB on top of what the reading kept.
- **s-t20 with the rule on** (r20261113-114851-b24e): the measurement branch `cursor/trn145-mask-measure-eb6e` (local), which is #9558 at ea11a40e04 plus #9676 plus #9730, merged cleanly. RAN on the VM: test_work's guarded and readings tests pass on it by name (56). The run asks 160 GB, stops at 144 GiB and writes `Work` to JSON. It reads, then runs #9558's pass, so its end gives the reading's time and peak and then the pass's.
- **#9730's body updated** (your items 3 and 4):
  - tiny(2,1024,64) before is #9676's 805.6 s at 10.46 GiB (RAN: b1a5), with the rule-off 796.3 s at 9.44 GiB labelled as rule off.
  - The body says none of the figures ran alone and that the peaks, not the times, are the comparison.
  - Correction to my 3:27 AM message: the 468.9 s s(2,256,512) run was not alone. It shared cpu-11 with the rule-off run, as 383d shared cpu-7 with b1a5. The body says so.

