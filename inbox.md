[2026-11-08T04:57Z] 784470cc0348
Coordinator 04:57Z, one addition for your branch (network-accounting, reviewing #7926): count, and assert in a test, a preempted request's lookup on resume hitting its OWN freed blocks. No served trace preempts, but the synthetic workload and corpus-smallkv do. Report how many such hits each has, and that pool.py and cache_step_v2.reference agree on them. After Task A; alongside Task B.

