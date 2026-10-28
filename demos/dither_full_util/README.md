# dither_full_util: the H200 at full utilization on useful s8 GEMMs

It shows the remote protocol's PoUW half (`demos/README.md`) as PoUW's work unit, not a remote run: Dither's work unit
at full utilization, with no runner yet.

The demo has the GPU run at full honest utilization on useful matmuls, not filler, and certifies as much of that time as it
can. The workload is an 8B model's linear layers at M = 16,384 in s8, back to back at full batch, over seeded weights
registered before the coin. PoUW credits the work through Dither's no-fold receipt, and a live session from the
challenger host supplies each period's coins. The verifier opens the drawn proof blocks, and once their Definition exists
it decides them in C-Flock `--zk` and Lean. The figure the run reports is ε, the share of the GPU's capacity that isn't
certified, with its terms: model slack, idle pipes, ε_skip at δ = 1%, and exchanged bits. Beside ε it reports γ and a
refused negative.

It runs nothing yet (`example.toml`: `run` `moved`, read from its `level = 0`): its runner isn't on `main`, so
`verity example dither_full_util` has nothing to run, and no run of record exists. When the H200's run of record exists,
its id goes in `example.toml`, and a `run.sh` that calls the runner in place makes it `runs`.
