# dither_full_util: the H200 at full utilization on useful s8 GEMMs

CMP-176's demo has the GPU run at full honest utilization on useful matmuls, not filler, and certifies as much of that
time as it can. The workload is an 8B model's linear layers at M = 16,384 in s8, back to back at full batch, over seeded
weights registered before the coin. PoUW credits the work through Dither's no-fold receipt, and a live session from
vy-challenger-1 supplies each period's coins. The verifier opens the drawn proof blocks, and once their Definition
lands (DITHER-FILL) it decides them in C-Flock `--zk` and Lean. The figure the run reports is ε, the share of the GPU's
capacity that is free compute, with its terms: honest's gap to the model, idle pipes, ε_skip at δ = 1%, and exchanged
bits. Beside ε it reports γ and a refused negative.

It stands at level 0 with no runner (`example.toml`). HIGHUTIL's runner is still on `cursor/highutil-demo-1de3`, and no
run of record exists yet. When the H200's run of record exists, its id goes in `example.toml`, and a `run.sh` that calls
the runner in place makes it level 1.
