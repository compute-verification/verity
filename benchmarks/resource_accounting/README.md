# Resource accounting: the matmul chain

A chain of L dependent calls y_{i+1} = act(W_i · y_i), every W_i its own weight in device memory, at Llama-3.1-8B's
layer shapes and one cube, timed on one RTX PRO 6000 (sm_120) natively and under the protocols, with what each run
certifies beside what it did.

- `sweep.py`: the plan, pure: the shapes, batches, chain length, arms and each arm's divisor.
- `chain.py`: the runner on the GPU (its docstring says what each arm runs, the gates and the timing); `--plan` lists the
  points without a GPU.
- `certified.py`: the certified columns, pure: γ from the pinned price twins (`benchmarks/pouw/harness/price_twins.py`,
  the larger of the two prices; only at a cube, so W_cert is unproved at the Llama shapes), W_cert = (1 − γ)(1 − εs)·ΣWref
  over the credited rows, and each resource's covert residual split into honest overhead and soundness slack. S_cert is
  the PoUS arm's storage profile when it is certified (`storage`); Q_cert is unproved until a bandwidth guarantee is pinned.
- `run.sh`: the recorded run on node 2: GPU 1's ship built outside the lease, `chain.py` under one `gpu-lease 1`
  (30 minutes at most), the retained pass verified after it. `POUS=1` (one shape, `--cwd clone`) adds the PoUS arm: P2's
  timed audit (`benchmarks/pous/pod.sh IMPL=p2-v3-mc-gpu`) on the same GPU, the chain running from its honest audit's
  start to its end and alone again after it; `POUS_IDLE=1` is its control, the chain idle through the same window.
- `render.py`: the table, from a run's `art:` id (or a local `chain.json`), on demand, with the PoUS section when the run
  had that arm. No table is stored.

~~~sh
research run --on vy-nebius-2 --project verity --campaign resource-accounting --source . --cwd source \
  --env GPU_LEASE_WHO=<id> --env SHAPES=o,qkv,gate_up,cube8192 -- bash benchmarks/resource_accounting/run.sh
research run --on vy-nebius-2 --project verity --campaign resource-accounting --source . --cwd clone \
  --env GPU_LEASE_WHO=<id> --env SHAPES=o --env BATCHES=1,128 --env ARMS=native-fp8,pearl-c --env POUS=1 \
  --env POUS_AUDITS=100 --env VERIFY_AT=o:1 -- bash benchmarks/resource_accounting/run.sh
python benchmarks/resource_accounting/render.py art:<id>
~~~

The tests (`tests/`) run the pure parts only: the plan, the certified columns against the pinned twins, the PoUS arm's
S_cert and capacity residual, and the table from a synthetic run.
