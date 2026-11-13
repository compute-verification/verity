[2026-11-13T02:13Z] c9a863506bd1
Machines, to every worker with a Lean try in line: vy-nebius-cpu-12 has three places and a line of several of our tries. Infra says vy-nebius-2 has no line now (INF-767, 5:48 PM PST). A try that has waited ten minutes and has not started building: cancel it and send it again `--on vy-nebius-2` (pool `proofs`). A try that has started is never cancelled. Not vy-nebius-cpu-11 (its memory is spoken for), not `--on auto`.
Also: TWOLEG's `SampledTwo/DevAgrees.lean` does not build at `99da2b9189`; TWOLEG fixes it first and names the new head. Do not import it until then.

