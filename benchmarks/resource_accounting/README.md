# Resource accounting: the matmul chain

A chain of L dependent calls y_{i+1} = act(W_i · y_i), every W_i its own weight in device memory, at Llama-3.1-8B's
layer shapes and one cube, timed on one RTX PRO 6000 (sm_120) natively and under the protocols, with what each run
certifies beside what it did.

- `sweep.py`: the plan, pure: the shapes, batches, chain length, arms and each arm's divisor (the fastest native arm of its
  precision the point measured).
- `chain.py`: the runner on the GPU (its docstring says what each arm runs, the gates and the timing); `--plan` lists the
  points without a GPU.
- `tuned.py`: the tuned native arms' GEMM at each point, the fastest correct one of the precision: every cuBLASLt
  heuristic candidate and CUTLASS kernel the PoUW harness's library (`benchmarks/pouw/harness`) has, gated bit-exactly by
  its gate with the known-bad variants rejected, then timed as the chain's L GEMMs on the chain's own operands. The FP8
  chain's weights have a per-tensor scale, which the registry's plain FP8 kernels don't read, so its pick is cuBLASLt's.
- `certified.py`: the certified columns, pure: γ from the pinned price twins (`benchmarks/pouw/harness/price_twins.py`,
  the larger of the two prices, at each Llama linear's (n, k) and the cube), W_cert = (1 − γ)(1 − εs)·ΣWref
  over the credited rows, and each resource's covert residual split into honest overhead and soundness slack, in its weak
  form (worst case over each audit's own coins; a per-run compute residual holds only conditional on `BudgetAbort` for
  the deployed cost model, so none is shown). S_cert is the PoUS arm's storage profile when it is certified (`storage`).
  Q_cert is none: the chains run with no bandwidth arm. First fetches are charged to the bandwidth account, not compute.
- `probe.py`: the native chains at one point under each condition in turn (stream, device synchronization, a held
  stream, NUMA node, a second process beside them), on the host, the device and to launch, then their course around one
  co-tenant at a time (a sleeping process, CPU spinners, an idle CUDA context) with the onset and offset of any change:
  why the chain ran faster beside the harness in a process of its own.
- `run.sh`: the recorded run on node 2: GPU 1's ship built outside the lease, `chain.py` under one `gpu-lease 1`
  (30 minutes at most), the retained pass verified after it. `LEASE` replaces the lease's GPU count and flags, `CHAIN_GPU`
  picks the leased GPU the chain runs on and `HOST_CPUS` pins every host thread; `PHASE` runs one of the three steps over a
  caller's `SCRATCH`, and `LEASE=held` runs the GPU step under the caller's lease. `POUS=1` (one shape, `--cwd clone`) adds the PoUS arm: P2's
  timed audit (`benchmarks/pous/pod.sh IMPL=p2-v3-mc-gpu`) on the same GPU, the chain running from its honest audit's
  start to its end and alone again after it; `POUS_IDLE=1` is its control, the chain idle through the same window.
  `JOINT=separate,inproc` (one shape, native arms, `--cwd clone`) instead runs `chain.py` inside the harness's process
  (`pod.sh BENCH_HOST`), one harness run a round with the chain as its honest audit's load: its responder in a process of
  its own, or a thread of the chain's (no MPS).
  `PROBE=1` (either mode) adds `probe.py` once the shape's points are measured alone; `SIDECAR=1` (either mode, not with
  `PROBE`) holds the course's idle CUDA context in a second process through the run, so every arm is measured at the speed
  it gives the native chains. `SMI_MS=<ms>` (`JOINT`) samples the GPU's clocks, power, temperatures and clock event
  reasons every `<ms>` through the lease into `smi.csv`.
  A tuned arm in `ARMS` needs `TUNED=<build dir|art:id>`,
  the harness's library as its `native/build.sh` makes it (`pouw-harness-build/v1`).
- `window.sh`: the timed window's steps (A, B, C, E and F, `run.sh` each) under one `gpu-lease 8 --wait --timed --max-min 60`:
  every step prepared first over one scratch, the lease taken at the booking's start (`AT`), the steps run in order on GPU 4
  with host threads under `taskset -c 128-159`, each only with its minutes left, the booking's windows-file line deleted
  once the lease ends (`RELEASE`), then the retained passes verified. The rehearsal is the same run on one GPU.
- `render.py`: the table, from a run's `art:` id (or a local `chain.json`), on demand, with the PoUS section when the run
  had that arm, the rounds' (the responder's isolation and the chain's time in each) when it hosted the harness, and the probe's. No table is stored.

~~~sh
research run --on vy-nebius-2 --project verity --campaign resource-accounting --source . --cwd source \
  --env GPU_LEASE_WHO=<id> --env SHAPES=o,qkv,gate_up,cube8192 -- bash benchmarks/resource_accounting/run.sh
research run --on vy-nebius-2 --project verity --campaign resource-accounting --source . --cwd clone \
  --env GPU_LEASE_WHO=<id> --env SHAPES=o --env BATCHES=1,128 --env ARMS=native-fp8,pearl-c --env POUS=1 \
  --env POUS_AUDITS=100 --env VERIFY_AT=o:1 -- bash benchmarks/resource_accounting/run.sh
research run --on vy-nebius-2 --project verity --campaign resource-accounting --source . --cwd clone \
  --env GPU_LEASE_WHO=<id> --env SHAPES=o --env BATCHES=1,128 --env ARMS=native-bf16,native-fp8 \
  --env JOINT=separate,inproc -- bash benchmarks/resource_accounting/run.sh
research run --on vy-nebius-2 --project verity --campaign resource-accounting --source . --cwd clone --no-sampler \
  --env GPU_LEASE_WHO=<id> --env LEGS=A,B,C,F --env AT=<the booking's start, UTC> --env RELEASE='<the booking line>' \
  -- bash benchmarks/resource_accounting/window.sh
python benchmarks/resource_accounting/render.py art:<id>
~~~

The tests (`tests/`) run the pure parts only: the plan, the certified columns against the pinned twins, the PoUS arm's
S_cert and capacity residual, the hosted rounds, the probe's summaries and course, the tune's finalists and the divisor
rule, and the table from a synthetic run.
