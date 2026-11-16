# Benchmarks

A benchmark answers one question with a number: what a protocol costs, or how much it establishes. Each is one
directory and one command, `verity bench NAME`, and it prints its number and how the number was measured.

Benchmarks only measure. None is part of the system: a benchmark decides no verdict, and no guarantee reads a number
computed here.

## The core benchmarks

The numbers the core protocols are judged by:

| Benchmark | Its number | Command |
| --- | --- | --- |
| Compute | The share of a served model's multiply-accumulates that PoUW's Pearl+ verifier credits. | `verity bench compute` |
| Memory | The share of a card's memory read rate that a PoUB audit leaves uncertified. | `verity bench memory --local` |
| Network | Exfiltration capacity: the most gigabytes about the weights that can leave over a monitoring period. | `verity bench exfiltration` |
| Inference-only | The least slowdown of training run inside the inference-only policy's inference blocks. | `verity bench inference_only` |
| Proofs | The time and bytes of a proof, layer by layer: Flock alone, with zero knowledge, with sampling, with recursion. | `verity bench prove layers` prints the table; [`one_stage/`](one_stage/README.md) and `verity bench recursion_e2e` run its layers |

Each one's README says what it needs and how to run it. Most need a GPU node. The network benchmark reads its inputs
from our evidence store, so it does not run from a clone yet.

## Try one

~~~sh
uv sync --all-packages --extra torch-cpu
uv run verity bench compute --from tests/benchmarks/compute/fixtures/record
~~~

This prints the compute benchmark's number from a sample record written by hand, so it needs no GPU. Its first lines
are the form every core benchmark prints:

~~~text
compute: 95.66% of the served model's MACs ...
ran with: coins in-process; parameters development ...
a measurement, not an audit
a made-up record, not a run's
~~~

The number comes first, then the coins and parameters it ran with. A result is an audit only when the deployed
challenger flipped its coins at deployed parameters, and anything else says it is a measurement. A run the verifier
did not accept prints no number. [`DESIGN.md`](DESIGN.md) has the interface in full.

## All benchmarks

Every directory has one line here, in the group of what it measures. Its status is one of:

- **maintained**: developed against the current tree, and its commands run from it;
- **frozen**: kept so that what it measured can be measured again, and changed only to keep it running;
- **data**: no measuring code, only numbers that other code takes as parameters.

### Protocol overhead

What the protocols cost: measured on served inference, and bounded on training.

| Directory | What it measures | Status |
| --- | --- | --- |
| `harness/` | the shared harness: what an arm adds to served inference against stock vLLM, and the kit runners share | maintained |
| `overhead/` | the one front door for overhead: a scheme's γ over stock at a card's operating points | maintained |
| `served_commit/` | what the Commit, every value of every step committed, adds to served vLLM | maintained |
| `demo_minutes/` | a demo's minutes from its first command to its verdict, read off its runs' records | maintained |
| `replay_determinism/` | whether a served MoE window's recorded schedule, replayed, gives its served bytes | maintained |
| `dsa_topk/` | the DSA indexer's top-k made a function of its logits by a vLLM plugin: its outputs run to run, and its cost | maintained |
| `replay_ladder/` | whether replaying a saturated window under its recorded schedule gives the served tokens, across a ladder of models | maintained |
| `serve_tax/` | what each change to serving costs vLLM's own server against stock compiled vLLM, at three loads | maintained |
| `prod_repro/` | vLLM served at its published serving settings with and without our changes, and the served window replayed | maintained |
| `vllm_encoded/` | Llama-3.1-8B in vLLM with its weights and KV cache held as codec words and receipted by PoUB | maintained |
| `inference_only/` | the inference-only policy of Llama 3.2 1B as served, and the least slowdown of training run inside its inference blocks | maintained |
| `slowdown/` | a lower bound on the slowdown a training run pays under the regime, and the time to its target | maintained |
| `nci/` | the sources, parameters and policies the inference-only policy's bounds rest on | maintained |
| `training_slowdown/` | the grader, which reads a training step's size off its Boolean circuit; and the jobs of an attack config | maintained |

### Kernels

GPU code held bit for bit to its reference or Definition before it is timed.

| Directory | What it measures | Status |
| --- | --- | --- |
| `hash/` | GPU kernels for PoUW's Dither prover, and the served path's SHA-512 commitment library | maintained |
| `flock_kernels/` | Flock's CUDA prover kernels on one GPU, each printing a digest of its output beside its time | maintained |
| `torch_gpu/` | the H200 demo's GPU kernels, each bit-exact to its Definition | maintained |
| `kernels/` | the served path's exact kernels, and what a kernel build ships with | maintained |
| `pouw/` | Pearl-C's kernels on the RTX PRO 6000, their harness, and the served vLLM path and its audits | maintained |
| `tc_probe/` | tensor-core instructions bit for bit against their silicon models | maintained |
| `tc_probe_fp4/` | the FP4 path on Blackwell: its conversions bit for bit, and each multiplier priced against dense NVFP4 | maintained |
| `native_peak/` | a card's measured dense-GEMM throughput for one datatype | maintained |
| `int8_class_rate/` | the H200's sustained int8 rate per operand class at the declared clock | maintained |
| `tensor_duty/` | the falsification test of the H200 card's tensor clock | maintained |

### Proving and recursion

| Directory | What it measures | Status |
| --- | --- | --- |
| `one_stage/` | the one-stage audit end to end, with its negatives; and a served window's two-stage audit | maintained |
| `two_stage/` | the two-stage audit end to end on a toy replay block, with its negatives | maintained |
| `prove_path/` | a served run's proof blocks proved after the fact, behind the Lean ZK server and through the Lean verifier | maintained |
| `served_units/` | a served model's proof blocks proved one type at a time from synthetic values | maintained |
| `unit_walk/` | the device witness's host walk of a served proof-block type's unit | maintained |
| `spot_proved/` | a replayed served step's values proved, and audited one stage end to end by the Lean verifier | maintained |
| `multi_unit/` | an FP8 GEMM's proof blocks staged as Flock statements in several layouts | maintained |
| `flock_s8_block/` | Flock's price, with zero knowledge, for a proof block of adjacent cells of one int8 GEMM output row | maintained |
| `unit_block/` | a word primitive's Boolean version built as one Flock proof block | maintained |
| `prove/` | the one command for what a proof system's prover costs on a workload, by block type | maintained |
| `proof_throughput/` | the proof blocks a second Flock proves and the Lean verifier accepts, over the batch size | maintained |
| `commitments/` | the commitment schemes' cost | maintained |
| `sampling_plan/` | the sampling plan for one served workload (a model, not a measurement) | maintained |
| `spot_figures/` | the SPOT post's figures for a model from its cards, and SPOT's planner (a model, not a measurement) | maintained |
| `exfiltration/` | GLM-5.3's exfiltration capacity in GB over a monitoring period at 10% of total compute | maintained |
| `verifier_cost/` | a verifier's time against the size of the circuit description it reads and of the circuit that description unrolls to | maintained |
| `recursion_e2e/` | recursive witness zero knowledge, timed step by step | maintained |
| `recursion_figures/` | a recursion bench result's session scaled over the census's fleets and the monitoring periods | maintained |
| `private_circuit/` | the universal unit, which runs any private circuit of a size class, proved with zero knowledge | frozen |

### Checks

| Directory | What it measures | Status |
| --- | --- | --- |
| `lean_records/` | a records run of the Lean proofs under a base version of the Lean tools and under this tree's | maintained |

### Circuits

| Directory | What it measures | Status |
| --- | --- | --- |
| `bool_build_cost/` | a workload's Boolean Build against its word Build | maintained |
| `compile_cost/` | what compiling a class-G window's Programs costs: wall time and peak memory, by phase | maintained |
| `circuit_properties/` | structural properties of a Boolean circuit's counted operations, measured exactly | maintained |
| `route_capture/` | what a model executes when served, step by step, and the routes class G's Definitions don't cover | maintained |
| `unit_lowering/` | writing a PoUW H1 circuit file: peak memory and wall time by K | maintained |

### Compute

| Directory | What it measures | Status |
| --- | --- | --- |
| `compute/` | the front door for the compute audit's number: the share of a served model's MACs that Pearl+ credits | maintained |
| `served_errors/` | error terms of the quoted guarantees that had no value, evaluated at the served sizes (a model, not a measurement) | maintained |
| `machine_model/` | the ACCEL model of each measured card, as data | maintained |

### Network

| Directory | What it measures | Status |
| --- | --- | --- |
| `network_traces/` | the network attester's runs in front of a served endpoint, and their zero-knowledge proofs | maintained |
| `network_e2e/` | the network chain's steps after N2, held on fakes and recorded endpoint runs | maintained |
| `certifier_term/` | the network attester's term for GLM-5.3 at the load of a node, a datacenter and the world | maintained |
| `network_sim/` | the workload's Programs that `verifier_cost` measures a verifier on | maintained |
| `certifier_timing/` | the bits an interval a server moves through when the network attester releases its bytes | maintained |
| `scan_cost/` | what the network attester's scan costs for each second of execution phase it covers | maintained |
| `mempool/` | the network attester's prefix-cache pool as memory-checked RAM | maintained |
| `datacenter/` | a datacenter as Verity accounts for it, in a small Python API | maintained |

### Memory

| Directory | What it measures | Status |
| --- | --- | --- |
| `pous/` | every PoUS implementation on one dataset, by its effective weight bandwidth | maintained |
| `bandwidth/` | the device ceilings PoUB's timing cites | maintained |
| `remote_session/` | PoUB's session harness: a verifier's and a prover's process over TCP. Not the remote demo | maintained |
| `memory/` | the front door for the memory audit's number: the share of a node's memory read rate one accepted PoUB session leaves uncertified | maintained |
| `kv_filler/` | the filler top-up beside vLLM serving, and what it costs inference | maintained |
| `stacked_share/` | the share of a node's live KV a prover could hold as encoded carriers inside the audit's compute budget | maintained |
| `word_codec/` | the store's Sloth codecs: each a byte-exact reference, a CUDA kernel and its benchmark | maintained |
| `codec_tensor/` | PoUB's receipts, leaves and receipt tree on the GPU, and the tensor-core wide-word codec | maintained |
| `property_redteam/` | attacks on the stacking property and on the gate model's counts behind the encode bound | maintained |
| `rate_squaring/` | squaring rates modulo the codec's primes, the evidence for PoUB's stated squaring ceiling | maintained |

### Numerics

| Directory | What it measures | Status |
| --- | --- | --- |
| `ppl/` | the perplexity an intervention on a served model's matmuls or attention costs | maintained |
| `f32_exhaustive/` | a reference primitive against MPFR, or a Boolean version against its word, on all 2^32 binary32 words | maintained |
| `definitions/` | the GPU probes behind the numerics Definitions, measured on every 32-bit input | maintained |
| `numerical/` | the bench spine that Flock and circuit-check import, and the per-unit-type cost inventory | frozen |

### Landing

How long the team's own changes take to reach `main`: measured on the verified queue's records, not on a protocol.

| Directory | What it measures | Status |
| --- | --- | --- |
| `landing/` | mark to `main`: the hours from a PR's ready mark to the move of `main` that holds it, over the last 7 days | maintained |
| `lean_unread_gate/` | the unread gate a ready mark's preflight runs, on one change: its time by stage | maintained |

### Communication

How fast the team's leads answer each other: measured on Linear's comments, not on a protocol.

| Directory | What it measures | Status |
| --- | --- | --- |
| `communication/` | the minutes from a Linear comment that rings a lane to that lane's answer, over the last 24 hours | maintained |

### Census data

| Directory | What it measures | Status |
| --- | --- | --- |
| `census/` | the fixed quantities benchmarks are computed against: hardware, datatypes, networks, deployments, subcircuits and workloads | data |

### The repository

| Directory | What it measures | Status |
| --- | --- | --- |
| `tree/` | how far `main` is from the layout of record: the not-yet list, the code still to write, and the ratchet lints' sites | maintained |
| `proof_status/` | the proofs' status as the Lean locks record it | maintained |

### Security

| Directory | What it measures | Status |
| --- | --- | --- |
| `security_findings/` | the security review's serious findings still open | maintained |

## Rules

- A benchmark imports like any other caller: `verity`, an adapter's public modules,
  [the harness](harness/README.md), and other benchmarks.
- Nothing outside `benchmarks/` imports a benchmark.
- It writes measurements, under its run's output directory. Results, tables and plots are not committed.
- It writes no verdict. Whether an audit passed is the verifier's to say.

## Metrics

A metric is one named number a benchmark records. A benchmark declares each metric it is held to in a `metrics.toml`
beside its runner, and the dashboard's Metrics tab shows them. [`DESIGN.md`](DESIGN.md) lists a metric's keys and says
how a value reaches the tab.
