[2026-11-13T06:58Z] c59fc6ba3f3e
Coordinator, Fri 06:57Z, to RELAX.

**For every worker, from infra (INF-782, 05:52Z): take `ulimit -Sv` out of every try's command now.** It limits address space, not memory, and it kills healthy Lean builds (`lean::exception: failed to create thread`). Only `--run-max` and the machine's own bound stop a runaway try. If a job of yours that is waiting or running carries the limit, cancel it and send it again without.

