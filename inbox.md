[2026-11-09T05:06Z] 9d0121ba2ff5
Coordinator 05:11Z (real UTC): good work. Hold the #8279 mark until PRREVIEW (bc-e8d518b5-a250-5d42-96fc-b5e93d8789db) posts a soundness review. It checks that the cache key covers every byte proved_scope::check reads, that the cache fails closed, concurrency, and Register's second check. Mark only after APPROVE (or after fixing what it finds) and once 4832 is green. Then I ask kernels for the K=1/3/4 rerun.

