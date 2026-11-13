[2026-11-13T23:09Z] 6d00bd66153c
PERIODREAD, coordinator, 23:12Z, one correction to the task I gave you at 23:08Z: do NOT launch with `--on auto` tonight. It can place a run on vy-nebius-cpu-6, which is not ours to use. Name the node on every launch (`research run --on vy-nebius-cpu-N ...`): any node but cpu-6 whose research pool shows a free slot in `research queue stats --json` (`data.nodes[NODE].slots`), with `--mem-wait` if the memory is refused. Everything else stands.

