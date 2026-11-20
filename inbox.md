[2026-11-20T12:29Z] 811b762a97e0
From your coordinator, 12:32Z, a tooling note from top: tries through `research run --on auto` that landed on vy-nebius-cpu-1 (its `build-2` tree) found no dependencies and ran nothing; naming cpu-5 worked. If a run of yours fails in seconds on cpu-1 with a missing dependency or having elaborated/collected nothing, it is that, not your change: rerun naming another node, and say so in your return (run id, node), do not count it as a result.

