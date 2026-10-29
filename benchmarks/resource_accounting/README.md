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
  the larger of the two prices, at each Llama linear's (n, k) and the cube), V = (1 − γ)(1 − εs)·ΣWref
  over the credited rows, and each resource's covert residual split into honest overhead and soundness slack, in its weak
  form (worst case over each audit's own coins; a per-run compute residual holds only conditional on `BudgetAbort` for
  the deployed cost model, so none is shown). S_cert is the PoUS arm's storage profile when it is certified (`storage`).
  Q_cert is `bandwidth.audit.bound` at the red team's grant (T = 1.2 ms, s = 1,849, no RTT), per
  window the verifier accepted within T, with S_chip as the bandwidth harness counts it (`summarize.s_chip_bytes`: the
  caches beside L2 and the SMs' registers and memory) and the side link at PCIe Gen5 x16; `window_params` also reports a
  window on time at T with the RTT at P2's cap inside it. First fetches are charged to the bandwidth account, not compute.
- `joint.py`: the bandwidth audit's arms inside the PoUS harness's process, over its store set up under the bandwidth
  audit's device-bound salt (`audit.draw_salt` over W's commitment and the GPU's id), each window through the bandwidth
  harness's `gpu.Prover` (the warm pass, the L2 scrub, then σ and the tag pass) and the granted `Verifier` in the same process, one
  `ChallengerState` per store: (d), the tag pass over the PoUS store as proof-only reads, every tag timed alone; (e),
  joint v0, `ro-blake3/v1`'s windows beside the chain and P2's audit; (f), joint v1, the honest prover pausing its chain
  at a call boundary for each window, at one window each 10, 100 or 1,000 ms of chain (or none): the bandwidth audit's
  honest overhead.
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
  `JOINT=separate,inproc,joint,joint-v1` (one shape, native arms, `--cwd clone`) instead runs `chain.py` inside the
  harness's process (`pod.sh BENCH_HOST`, the store under the GPU's bandwidth salt, `BANDWIDTH_SALT`), one harness run a
  round with the chain as its honest audit's load: its responder in a process of its own, or a thread of the chain's (no
  MPS), and the joint rounds with `joint.py`'s arms.
  `PROBE=1` (either mode) adds `probe.py` once the shape's points are measured alone; `SIDECAR=1` (either mode, not with
  `PROBE`) holds the course's idle CUDA context in a second process through the run, so every arm is measured at the speed
  it gives the native chains. `SMI_MS=<ms>` (`JOINT`) samples the GPU's clocks, power, temperatures and clock event
  reasons every `<ms>` through the lease into `smi.csv`, on the wall time `joint.py`'s anchor puts its passes on.
  A tuned arm in `ARMS` needs `TUNED=<build dir|art:id>`,
  the harness's library as its `native/build.sh` makes it (`pouw-harness-build/v1`).
- `window.sh`: the timed window's steps (A to F, `run.sh` each) under one `gpu-lease 8 --wait --timed --max-min 60`:
  every step prepared first over one scratch, the lease taken at the booking's start (`AT`), the steps run in order on GPU 4
  with host threads under `taskset -c 128-159`, each only with its minutes left, the booking's windows-file line deleted
  once the lease ends (`RELEASE`), then the retained passes verified. The rehearsal is the same run on one GPU.
- `render.py`: the table, from a run's `art:` id (or a local `chain.json`), on demand, with the PoUS section when the run
  had that arm, the rounds' (the responder's isolation, the chain's time in each, (d), (e) and (f)) when it hosted the
  harness, and the probe's. No table is stored.

~~~sh
research run --on vy-nebius-2 --project verity --campaign resource-accounting --source . --cwd source \
  --env GPU_LEASE_WHO=<id> --env SHAPES=o,qkv,gate_up,cube8192 -- bash benchmarks/resource_accounting/run.sh
research run --on vy-nebius-2 --project verity --campaign resource-accounting --source . --cwd clone \
  --env GPU_LEASE_WHO=<id> --env SHAPES=o --env BATCHES=1,128 --env ARMS=native-fp8,pearl-c --env POUS=1 \
  --env POUS_AUDITS=100 --env VERIFY_AT=o:1 -- bash benchmarks/resource_accounting/run.sh
research run --on vy-nebius-2 --project verity --campaign resource-accounting --source . --cwd clone \
  --env GPU_LEASE_WHO=<id> --env SHAPES=o --env BATCHES=1,128 --env ARMS=native-bf16,native-fp8 \
  --env JOINT=separate,inproc,joint -- bash benchmarks/resource_accounting/run.sh
research run --on vy-nebius-2 --project verity --campaign resource-accounting --source . --cwd clone --no-sampler \
  --env GPU_LEASE_WHO=<id> --env LEGS=A,B,C,F --env AT=<the booking's start, UTC> --env RELEASE='<the booking line>' \
  -- bash benchmarks/resource_accounting/window.sh
python benchmarks/resource_accounting/render.py art:<id>
~~~

The tests (`tests/`) run the pure parts only: the plan, the certified columns against the pinned twins and Q_cert at the
grant, the PoUS arm's S_cert and capacity residual, `joint.py`'s windows and pause hook through `gpu.Prover` over a CPU
tagger with a real challenger state, the probe's summaries and course, the tune's finalists and the divisor rule, and the
table from a synthetic run.
