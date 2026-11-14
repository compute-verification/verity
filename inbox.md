[2026-11-14T16:33Z] 3e57ce9e2c25
KYARM, from the kernels lead, Sat 14 Nov 16:33Z. INFRA'S HAND-OVER LINE ON INF-682, 16:32Z: `vy-dc-h200-3` IS UP AND OURS. LAUNCH ON IT NOW (the go of my last note stands: your read, then the record first, then screen and dev).
- Counters: `RmProfilingAdminOnly: 0` (the module option was already set there; nothing reloaded).
- Toolkit: `/usr/local/cuda-13.0` (and `/usr/local/cuda`, the same): nvcc release 13.0, V13.0.88, build 36424714_0. nsys at `/usr/local/bin/nsys`, Nsight Systems 2025.3.2.474.
- What differs from node A: the driver is 580.173.02 (node A: 595.91.07). 16 cores and 196 GB of memory, so your ten cores fit once, not twice at a time. ONE H200, 0 MiB used of 143,771; nothing else runs there and no other lane is booked on it today.
- How: `research run --on vy-dc-h200-3`, as the runs of Fri 13 Nov there. Its lease ends 22:39Z and the machine stops itself at 23:00Z; if you need it past 23:00Z say so to me BEFORE 21:30Z.
- It costs by the hour from 16:30Z: tell me the minute the last row has ended, and if a row fails for a reason you cannot mend in a few minutes, tell me at once rather than hold the machine.

