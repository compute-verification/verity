[2026-11-15T00:07Z] 128689057dc0
From the proofs lead, 00:10Z: benchmarks traced the SCHEME red: spot.prover.cost.glm53's SCHEME moved to benchmarks/sampling_plan/inputs.json in 475519fbd24; the callers in verity-pouw-benchmarks (8 tests a shard) still reach the old place. Mend the callers if they are in our files, else one line for compute-accounting. Then re-run your command.

