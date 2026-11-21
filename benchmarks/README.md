# Benchmarks

`benchmarks/` holds measurement only: each directory measures something or holds the data measurements are computed against, and nothing outside it reads it. `verity bench NAME [ARGS]` runs a directory's `run.py` or `run.sh`; `verity calibrate NAME` runs `calibration/NAME/`'s; `verity-bench` is the shared harness's own command ([`harness/`](harness/README.md)).

| directory | owner | what it measures | command |
| --- | --- | --- | --- |
| `harness/` | benchmarks | verity-bench: specs, recorded served runs against stock vLLM, the served table, the kit | `verity-bench --help` |
| `overhead/` | benchmarks | γ of a PoUW scheme over stock at a card's operating points | `verity bench overhead` |
| `served_commit/` | circuits | what committing every step's values adds to served vLLM | `verity bench served_commit` |
| `application_runs/` | vllm | one application run's accepting verdicts and minutes | `verity bench application_runs` |
| `application_census/` | — | where each application in applications/ stops under `verity run` | `verity bench application_census` |
| `demo_minutes/` | vllm | the onsite application's minutes from first command to verdict | `onsite_minutes.py` |
| `dsa_topk/` | proofs | the DSA indexer's top-k as a function of its logits: determinism and cost | `verity bench dsa_topk` |
| `replay_ladder/` | proofs | replaying a saturated window gives the served tokens, across dense and MoE models | `verity bench replay_ladder` |
| `serve_tax/` | vllm | each serving change's cost over stock compiled vLLM, and the serving cost | `verity bench serve_tax` |
| `prod_repro/` | vllm | serve, load and replay in one command: what our changes cost over stock | `verity bench prod_repro` |
| `serving_cost/` | benchmarks | the serving cost of a model too large to optimize on, in five stages | `verity bench serving_cost` |
| `vllm_encoded/` | — | Llama-3.1-8B in vLLM with weights and KV cache as codec words, receipted by PoUB | `stack.sh` |
| `inference_only/` | training | the inference-only policy's least training slowdown under the min-cut bound | `verity bench inference_only` |
| `slowdown/` | training | a lower bound on a training run's slowdown under the regime | `verity bench slowdown` |
| `nci/` | — | the sources, parameters and policies the inference-only bounds rest on | its modules |
| `../calibration/training/` | training | a training step's matrix products and circuit size beside its slowdown trace | `verity calibrate training` |
| `kernels/` | kernels | the served path's exact kernels and the ship frame every build sources | each script's header |
| `pouw/` | — | Pearl-C's kernels on the RTX PRO 6000, their harness and audits | `verity bench pouw NAME` |
| `../calibration/tc_probe/` | circuits | tensor-core instructions bit for bit against their silicon models | `verity calibrate tc_probe` |
| `../calibration/tc_probe_fp4/` | — | the FP4 path on Blackwell: conversions and multipliers | `verity calibrate tc_probe_fp4` |
| `../calibration/native_peak/` | — | a card's measured dense-GEMM throughput for one datatype | `verity calibrate native_peak` |
| `../calibration/tensor_duty/` | compute-research | the H200 tensor clock's falsification test by duty cycle | `verity calibrate tensor_duty` |
| `one_stage/` | — | the one-stage audit end to end, to the audit record, with its negatives | its scripts' headers |
| `prove_path/` | — | a served run's proof blocks proved after the fact, to the Lean verifier | `verity bench prove_path` |
| `served_units/` | circuits | a served model's proof blocks proved a type at a time from synthetic values | `verity bench served_units` |
| `unit_walk/` | proofs | the device witness's host walk of a proof-block type's unit, by width | `verity bench unit_walk` |
| `spot_proved/` | — | a replayed served step's values proved and audited end to end | `stage.py` |
| `flock_s8_block/` | proofs | Flock --zk's price for a block of one int8 GEMM output row | `verity bench flock_s8_block` |
| `prove/` | benchmarks | a proof system's prover cost on a workload, from the store's labelled records | `verity bench prove` |
| `proof_throughput/` | proofs | proof blocks a second Flock proves and the Lean verifier accepts, by provers | `verity bench proof_throughput` |
| `commitments/` | proofs | the commitment schemes' cost: leaves, trees, hiding leaves, row tables | `tool.py` |
| `../calibration/sampling_plan/` | — | a served workload's sampling plan: ε at δ over verification overhead (a model) | `glm53.py` |
| `spot_figures/` | — | the SPOT post's figures from a model's cards and SPOT's planner (a model) | `figures.py` |
| `exfiltration/` | benchmarks | GLM-5.3's exfiltration capacity in GB over a monitoring period | `verity bench exfiltration` |
| `verifier_cost/` | proofs | a verifier's time against the size of the circuit it reads | `verity bench verifier_cost` |
| `recursion_e2e/` | benchmarks | refuses by name: the old recursion's benchmark is gone; its metrics keep their readings | `verity bench recursion_e2e` |
| `remote_error/` | soundness | SPOT's cryptographic error in the remote application, at its parameters | `verity bench remote_error` |
| `lean_records/` | — | a records run of security/Proofs under two versions of the Lean tools | `guard.py` |
| `compile_cost/` | circuits | what compiling a class-G window's Programs costs, by phase | `verity bench compile_cost` |
| `../calibration/circuit_properties/` | training | structural properties of a Boolean circuit's counted operations | `verity calibrate circuit_properties` |
| `route_capture/` | — | what a served model executes: launch-shape classes and uncovered (route, statics) pairs | `verity bench route_capture` |
| `compute/` | compute-accounting | the compute audit's number: the share of a served model's MACs Pearl+ credits | `verity bench compute` |
| `served_errors/` | — | error terms of the quoted guarantees evaluated at served sizes (a model) | `verity bench served_errors` |
| `../calibration/machine_model/` | compute-accounting | the ACCEL model of each measured card and a design's ε | `verity calibrate machine_model` |
| `network_traces/` | network-accounting | the network attester's runs in front of a served endpoint and their proofs | `verity bench network_traces` |
| `network_e2e/` | — | the network chain's steps after N2 on fakes and recorded endpoint runs | `chain.py` |
| `certifier_term/` | — | the network attester's term for GLM-5.3 at a node, a datacenter and the world | `verity bench certifier_term` |
| `network_sim/` | — | the workload's Programs: a Boolean MLP and vLLM's serving Program | its modules |
| `../calibration/attester_timing/` | network-accounting | the bits an interval moves through when the attester releases bytes | `verity calibrate attester_timing` |
| `attester_load/` | network-accounting | what the network attester costs the requests it carries | `verity bench attester_load` |
| `mempool/` | — | vLLM's page pool as memory-checked RAM, held to vLLM's own pool | `difftest.py` |
| `datacenter/` | — | a datacenter as Verity accounts for it, as a small Python API | its README |
| `pous/` | — | every PoUS implementation on one dataset, by effective weight bandwidth | `bench.py` |
| `../calibration/bandwidth/` | memory-accounting | the device ceilings PoUB's timing cites, and a node's interface report | `verity calibrate bandwidth` |
| `poub_session/` | — | PoUB's session harness: verifier and prover over TCP, end to end | `verity bench poub_session` |
| `memory/` | memory-accounting | the memory audit's number: the share of a node's read rate left uncertified | `verity bench memory` |
| `kv_filler/` | — | the filler top-up beside vLLM serving, and its cost to inference | `bench.py` |
| `word_codec/` | — | the store's Sloth codecs: reference, CUDA kernel and benchmark | `verity bench word_codec` |
| `codec_tensor/` | — | PoUB's receipts and tree on the GPU, and the tensor-core wide-word codec | `verity bench codec_tensor` |
| `property_redteam/` | — | attacks on the stacking property and the gate model's counts | `demo.py` |
| `../calibration/rate_squaring/` | memory-accounting | squaring rates modulo the codec's primes on the GPU and host IFMA | `verity calibrate rate_squaring` |
| `ppl/` | compute-accounting | the perplexity an intervention on served matmuls or attention costs | `verity bench ppl` |
| `../calibration/f32_exhaustive/` | — | correctly rounded f32 primitives against MPFR on all 2^32 words | `verity calibrate f32_exhaustive` |
| `definitions/` | — | GPU probes behind the numerics Definitions: approximate instructions on every input | its scripts |
| `numerical/` | — | the bench spine Flock and circuit-check import, and the per-unit-type cost inventory | `verity bench numerical` |
| `landing/` | ci | mark to main over the last 7 days, by queue class | `verity bench landing` |
| `lean_unread_gate/` | — | the unread gate's time by stage on one change, and its guard | `verity bench lean_unread_gate` |
| `communication/` | top | minutes from a Linear comment to each rung lane's answer | `response.py` |
| `../calibration/census/` | — | data: the fixed quantities benchmarks are computed against | none |
| `tree/` | top | the not-yet list, unbuilt rows and lint sites of the tree, by lead | `not_yet.py` |
| `proof_status/` | lean | the deployed guarantees and their trust gaps, from the Lean locks | `verity bench proof_status` |
| `security_findings/` | infra | the security review's serious findings still open, each closed by its check | `open_findings.py` |
