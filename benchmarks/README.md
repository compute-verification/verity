# Benchmarks

`benchmarks/` holds measurement only: each directory measures something, or holds the data measurements are computed
against. End-to-end examples, what a person runs to serve and audit a workload, are in `demos/`. Results are not
committed: a run publishes its outputs to the evidence store, and tables are rendered from the store on demand.
`verity-bench` (`harness/`) is the shared harness for what an arm adds to served inference against stock vLLM, with its
specs in `specs/`; `verity bench NAME` runs a directory's own runner.

Each directory's README, or its main module's docstring where it has none, says how to run it. Its status is one of:

- **maintained**: developed against the current tree, and its commands run from it;
- **frozen**: kept so that what it measured can be measured again, and changed only to keep it running;
- **data**: no measuring code, only numbers that other code takes as parameters.

`tests/test_benchmarks_index.py` holds this index to the tree: every directory under `benchmarks/` has one row, and every
row names a directory. `judge.py` and `latency_compare.py`, beside the directories, compare recorded runs' results.

## Protocol overhead

What the protocols cost: measured on served inference, and bounded on training.

| Directory | What it measures | Status |
| --- | --- | --- |
| `harness/` | `verity-bench`: specs, recorded runs, sweeps and results for an arm against stock vLLM, and the served table of each protocol's overhead | maintained |
| `specs/` | the harness's specs and sweeps, one TOML file each | maintained |
| `served_commit/` | what the Commit, every value of every step committed, adds to served vLLM: throughput, time to first token and per-token latency | maintained |
| `g2/` | the served Commit's overhead curve over stock, rendered from the evidence store; it measures nothing itself | maintained |
| `vllm_encoded/` | Llama-3.1-8B in vLLM with its weights and KV cache held as codec words and receipted by PoUB, as `verity-bench` arms beside the plain stack | maintained |
| `resource_accounting/` | a chain of dependent matmuls at Llama-3.1-8B's shapes on one RTX PRO 6000, natively and under each protocol, with what each run certifies | maintained |
| `load_time/` | the time from spawn to engine-ready for a large model under vLLM at TP 8, by page-cache, compile-cache and loader state | frozen |
| `rho/` | a lower bound on the overhead a training run pays under the regime, and the time to its target, per scenario and monitoring period | maintained |
| `nci/` | the sources, parameters and policies NCI's bounds rest on, and a board of training methods with their slowdown brackets | maintained |

## Kernels

GPU code held bit for bit to its reference or Definition before it is timed.

| Directory | What it measures | Status |
| --- | --- | --- |
| `hash/` | GPU kernels for PoUW's Dither prover (`dither/`: the no-fold and J10-R units, their gates and harnesses), the served path's SHA-512 commitment library (`lib/`), and vLLM's int8-exact linear kernels (`exactlin/`) | maintained |
| `torch_gpu/` | the H200 demo's GPU kernels, each bit-exact to its Definition: the serve-time commit (`commit_slots`, `sha512_rows`), the staging decode with bandwidth receipts (`p2_stage`), the decoder's non-matmul ops (`units`), attention (`attn_bf16`), the exact W8A8 linear (`s8_linear`) and J10-R's units on the H200 (`j10r_sm90`) | maintained |
| `pouw/` | Pearl-C's kernels on the RTX PRO 6000, their harness, the served vLLM path and its audits, and the ACCEL model (`machine_model/`); `contents.py` lists the rest | maintained |
| `tc_probe/` | tensor-core instructions bit for bit against their silicon models, and the trust each instruction's evidence earns | maintained |
| `tc_probe_fp4/` | the FP4 path on Blackwell: its conversions bit for bit, and cheaper multipliers (sparse, Strassen, lookup tables) priced against dense NVFP4 | maintained |
| `native_peak/` | a card's measured dense-GEMM throughput for one datatype (`native-peak/v1`) | maintained |

## Proving and recursion

| Directory | What it measures | Status |
| --- | --- | --- |
| `one_stage/` | the one-stage audit end to end: committed values, registration, the draw, C-Flock's circuit prover and the Lean verifier, to the audit record and its integrity profile, with its negatives | maintained |
| `prove_path/` | a served run's proof blocks proved after the fact: the compiler's Programs, a draw on live coins, C-Flock `--zk` behind the Lean firewall, and the Lean verifier | maintained |
| `served_units/` | a served model's proof blocks proved one type at a time from synthetic values, the Lean verifier accepting the honest run and refusing forgeries | maintained |
| `multi_unit/` | an FP8 GEMM's proof blocks staged as C-Flock statements in several layouts: prover and verifier time, peak memory and proof bytes | maintained |
| `proof_throughput/` | the proof blocks a second C-Flock proves and the Lean verifier accepts on GLM-5.3's dot products, over the batch size | maintained |
| `commitments/` | the commitment schemes' cost: hashing leaves and building the tree, hiding leaves per row, and serving's per-step row tables | maintained |
| `sampling_plan/` | the sampling plan for one served workload: a cost model fitted to proving measurements, and ε at δ over the verification overhead (a model, not a measurement) | maintained |
| `verifier_cost/` | a verifier's time against the size of the circuit description it reads and of the circuit that description unrolls to | maintained |
| `recursion_e2e/` | recursive witness zero knowledge, timed step by step: the inner proof, V* proved with `--zk`, and the Lean verifier | maintained |
| `private_circuit/` | the universal unit, which runs any private circuit of a size class, proved under `--zk` and verified by the Lean verifier | frozen |
| `live_challenger/` | the proof system's challenger on an auditor host: each session's live coins, recorded in a signed ledger before the first coin | maintained |

## Circuits

| Directory | What it measures | Status |
| --- | --- | --- |
| `bool_build_cost/` | a workload's Boolean Build against its word Build: the descriptor's size and each Definition's proof-block rule | maintained |
| `circuit_properties/` | structural properties of a Boolean circuit's counted operations, measured exactly: fan-out, fan-in, their count and s_× | maintained |

## Network

| Directory | What it measures | Status |
| --- | --- | --- |
| `network_traces/` | the network certifier's runs in front of a served endpoint, and their zero-knowledge proofs | maintained |
| `network_e2e/` | network accounting end to end on a served model, as one command and one report, with its negatives | maintained |
| `network_sim/` | the device graph, and a mock datacenter of Linux network namespaces that runs the real network certifier | maintained |
| `datacenter/` | a datacenter as Verity accounts for it, in a small Python API: the Program, hardware, binding, plan, run, claims and account | maintained |

## Memory

| Directory | What it measures | Status |
| --- | --- | --- |
| `pous/` | every PoUS implementation on one dataset, by its effective weight bandwidth | maintained |
| `bandwidth/` | the bandwidth audit's GPU tagger: each chunk's tag over the store in device memory, their leaves and root | maintained |
| `remote_session/` | PoUB's session harness (proof of useful bandwidth): a verifier's and a prover's process over TCP, end to end. The remote protocol's demo is the vLLM demo, not this | maintained |
| `live_verifier/` | PoUB's verifier as a service on an auditor host | maintained |
| `kv_filler/` | the filler top-up beside vLLM serving, and what it costs inference | maintained |
| `stacked_share/` | the share of a node's live KV a prover could hold as encoded carriers inside the audit's compute budget | maintained |
| `serve_bandwidth/` | where a served mix's DRAM read bandwidth goes on the RTX PRO 6000, by phase and kernel class | maintained |
| `word_codec/` | the store's Sloth codecs (the word codec, the KV codec that PoUB's `codec.py` mirrors, the wide codecs): each a byte-exact reference, a CUDA kernel and its benchmark | maintained |
| `codec_tensor/` | PoUB's receipts, leaves and receipt tree on the GPU, and the tensor-core wide-word codec | maintained |
| `property_redteam/` | attacks on the stacking property and on the gate model's counts behind the encode bound | maintained |
| `rate_squaring/` | squaring rates modulo the codec's primes on the GPU and on the host's AVX-512 IFMA, the evidence for PoUB's stated squaring ceiling | maintained |

## Numerics

| Directory | What it measures | Status |
| --- | --- | --- |
| `ppl/` | the perplexity an intervention on a served model's matmuls or attention costs, against bf16 and plain int8 | maintained |
| `numerical/` | the bench spine (`verity_numerical.bench`) that C-Flock and `circuit-check` import, and the per-unit-type cost inventory; the archived proof systems' tables are in the evidence store | frozen |

## Census data

| Directory | What it measures | Status |
| --- | --- | --- |
| `census/` | the fixed quantities benchmarks are computed against: hardware, datatypes, networks, deployments, subcircuits and workloads | data |
