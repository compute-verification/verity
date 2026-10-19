# One-stage audit runs

The one-stage audit (`verity.core.protocols.sampled_proofs.one_stage`) end to end: serving's committed values through registration, the verifier's
draw, C-Flock's circuit prover (M0) and the Lean verifier, to the audit record with its integrity profile, then the
negatives.

- `a<N>.py`: the driver of run A<N>; its docstring says what the run adds and how to call it.
- `compose_circuit.py`, `stage_subset.py`, `stage_tampered.py`: helpers that run inside M0's tree (its `verity_flock`).
- `pod.sh`: the entry point on a CPU pod.
- `a5.sh`: run A5's entry point on a CPU machine. A5 is a deferred Commit's window, a stand-in linked to its vllm-v1 run root
  and replay bundle: the CPU RoPE d64 Commit fixture, its window from `verity-vllm serving-rows`, and M0 and Lean built from the
  run's tree.
- `a6.sh`: run A6's entry point on a CPU machine, on hidden_zk's cached builds. A6 is the W8A8 coordinates of an int8
  decoder layer (`ScaledMmS8Coordinate_v1` at each K, Llama-3.2-1B's by default) under the stratified law, stand-in values
  staged by C-Flock's `UnitStager` bound to the partition in statements the spec's shape admits (m <= 35), each stratum's
  share proved under `--zk` across its template's statements and held by the Lean verifier with the units it derives itself.
- `a8.sh`: run A8's entry point, on the same builds. A8 is one window of the remote bandwidth audit over a registered store of
  a few thousand blocks with every opening proved by the opening unit (`remote_bandwidth.opening`) under `--zk` and held by
  Lean, honest and with a tampered block, a scan of everything the verifier received for the drawn blocks' bytes, and the
  prover's cost per opening. `a8_gpu.sh` takes its cost points on sm_120 (`flock-circuit prove --zk --gpu`), as a GPU host
  job, and `a8_cost.py` reads a run's sessions into the one cost model's observations (`benchmarks/sampling_plan`).
- `a7.sh`: run A7's entry point, on the same builds; the remote session's checks 1-3 and 5 on each drawn receipt are
  proved, not opened, by the opening unit (`remote_bandwidth.opening`, `verity.ml.commitments.gates.block_check`) in a
  second statement. A7 is the requirement's plan and memory-map units
  (`verity.ml.circuits.requirement_units`), one of each per step of a stand-in decoder's requirement, under the stratified
  law, their witness rows bound to canonical commitments, then wired to the remote bandwidth audit's ledger as its
  requirement link (`audit.RequirementLink`), its ε the most share of the linked credit the units its draw leaves
  unproved can carry and its δ the draw's.
- `plan_shards.py`: the sharded plan unit (`PlanShard_v1`) on a recorded remote session (vllm's `remote_session.json` and the
  remote verifier's `verifier.json`): each pass's plan cut into the store's shards, every instance held to the unit's
  reference, each distinct one to the circuit, and one staged by `UnitStager` for its fit in M0's block. A CPU job; it
  builds nothing.
- `zk_cost.sh`: C-Flock's zero-knowledge session cost on sm_120, for the sampling planner's cost model (`zk_cost.py`, a GPU
  host job on vy-nebius-1, or a `--preemptible` one on vy-nebius-2). A6's W8A8 coordinates at K = 4096 and 14336 are
  staged in statements of 1 to 512 units, and sessions of 1 to 512 drawn units are proved by `flock-circuit prove --zk
  --gpu`, with the firewall's send check on and off. The Lean verifier checks them, and the job fits the per-session,
  per-statement and per-unit costs with their standard errors, writing `costfit` observations.

C-Flock's circuit prover (M0, `verity_flock.circuit`) is on `main`, but a run does not take it from its checkout:
`pod.sh` takes `git archive`s of M0's tree and the Lean verifier's tree at pinned commits, and a captured input set, as
inputs sent with the run; it builds `flock-circuit` and `flock-verify` once per pod, then runs the driver, whose
`--m0-pythonpath` names that M0 tree. The drivers' checks go through the verifier of record
(`one_stage.verifier`), which builds the sampled-proofs Lean in the run's tree on its first check, hence `--cwd clone`:

~~~sh
research run --on POD --source . --cwd clone --send m0.tar --send lean.tar --send set.tar -- bash benchmarks/one_stage/pod.sh
~~~

`check` runs nothing here.
