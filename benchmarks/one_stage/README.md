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
- `a7.sh`: run A7's entry point, on the same builds. A7 is the requirement's plan and memory-map units
  (`verity.ml.circuits.requirement_units`), one of each per step of a stand-in decoder's requirement, under the stratified
  law, their witness rows bound to canonical commitments, then wired to the remote bandwidth audit's ledger as its
  requirement link (`audit.RequirementLink`), its ε the most share of the linked credit the units its draw leaves
  unproved can carry and its δ the draw's.
- `a8.sh`: run A8's entry point, on the same builds. A8 is GLM-5.3 TP8 B1's served step, rank 0: `a8_kinds.py` censuses
  the request Program's root Calls by Definition kind and tries one unit of each through M0; `a8_slice.py` cuts the
  capture's values and the checkpoint's tensors for a few layers off the serving node; `a8.py` stages each lowering kind's
  units from those values, one template each, under the stratified law, proved under `--zk` and held by the Lean verifier;
  `a8_table.py` renders the per-kind table and its gap lines from the audits' summaries.
- `a9.sh`: run A9's entry point, on the same builds. A9 is GLM-5.3's routed-expert FP8 GEMMs in the registry's Boolean
  static pair form (`boolean_fp8_moe`): `a9.py` holds one coordinate of each launch's first routed pair to the capture in
  its word and Boolean forms, cuts an up/gate coordinate whose rows overflow M0's block into its gather and
  block-coordinate units, and saves the units for `a8.py --load-values`, whose audit proves them. `A9_SAVED` audits a
  saved directory without the values stage.

C-Flock's circuit prover (M0, `verity_flock.circuit`) is on `main`, but a run does not take it from its checkout:
`pod.sh` takes `git archive`s of M0's tree and the Lean verifier's tree at pinned commits, and a captured input set, as
inputs sent with the run; it builds `flock-circuit` and `flock-verify` once per pod, then runs the driver, whose
`--m0-pythonpath` names that M0 tree. The drivers' checks go through the verifier of record
(`one_stage.verifier`), which builds the sampled-proofs Lean in the run's tree on its first check, hence `--cwd clone`:

~~~sh
research run --on POD --source . --cwd clone --send m0.tar --send lean.tar --send set.tar -- bash benchmarks/one_stage/pod.sh
~~~

`check` runs nothing here.
