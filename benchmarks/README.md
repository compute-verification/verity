# Benchmarks

`benchmarks/` holds measurement only: each directory measures something, or holds the data measurements are computed
against. A benchmark is never the verifier, writes no verdict and computes no number a guarantee reads, and
nothing outside `benchmarks/` reads it. End-to-end demos, what a person runs to serve and audit a workload, are in
`demos/`. This file is the directory's interface: its entry points, the shared harness, what a benchmark may import
and write, and the index of every directory.

## Entry points

`verity bench NAME [ARGS]` runs the directory NAME's own runner, its `run.py` or else its `run.sh`, and hands it ARGS (a
`-` in NAME is a `_`). A runner that needs a GPU or a whole node is the command of a `research run`, as its header
says. These are the directories that have one:

| Command | What it runs |
| --- | --- |
| `verity bench bandwidth` | the device ceilings PoUB's timing cites, one probe a run: `chip` (what a card holds on chip) and `write_gpu` on one GPU, `write_host` on the host's CPUs |
| `verity bench bool_build_cost` | a workload's Boolean Build against its word Build: the descriptor's size and each Definition's proof-block rule, timed |
| `verity bench certifier_term` | the network certifier's term for GLM-5.3 at a scope's load, in bits and GB |
| `verity bench circuit_properties` | its circuits' fan-out, fan-in and s_×, counted exactly, beside `rho`'s shape-derived ones |
| `verity bench compile_cost` | wall time and peak memory, by phase, of compiling a class-G window's Programs |
| `verity bench compute` | the compute audit's number: the share of a served model's MACs that Pearl+ credits over the tokens a served run served, from the verifier's side of a recorded run's record |
| `verity bench dsa_topk` | the canonical top-k on one GPU node: its outputs run to run and in two processes at once, the plugin as vLLM loads it, and its cost |
| `verity bench exfiltration` | GLM-5.3's exfiltration capacity in GB over a monitoring period at 10% of total compute: the compute audit's part beside the network certifier's, each input labelled |
| `verity bench f32_exhaustive` | a reference primitive against MPFR, or a Boolean version against its word, on all 2^32 binary32 words |
| `verity bench flock_s8_block` | Flock `--zk`'s price for proof blocks of adjacent cells of one int8 GEMM output row, on one GPU node |
| `verity bench int8_class_rate` | one card's sustained int8 rate per operand class at the declared clock, alone and beside SHA-512 |
| `verity bench moe_align` | the canonical `moe_align_block_size` on one GPU: its outputs run to run, the plugin as vLLM loads it, and its cost |
| `verity bench network_e2e` | network certification end to end on a served model, as one report with its negatives |
| `verity bench network_sim` | the device graph's demo, and the mock datacenter of network namespaces running the real network certifier |
| `verity bench network_traces` | one chunk of per-token timing traces of real vLLM serving on one GPU, for the network certifier's calibration |
| `verity bench overhead` | the front door for a scheme's γ over stock at a card's operating points: `points`, `run` and `table` |
| `verity bench ppl` | the perplexity an intervention on a served model's matmuls or attention costs: `eval`, `submit`, `report`, `baseline` |
| `verity bench prod_repro` | vLLM served at its published serving settings with and without our changes and the served window replayed, as the settings and three lines: the replayed tokens against the served, the cost over stock, the advice a request |
| `verity bench proof_throughput` | the proof blocks a second Flock proves and the Lean verifier accepts, over the batch size |
| `verity bench property_redteam` | a stacking-property check graded against the red team's cases |
| `verity bench prove_path` | a served run's proof blocks proved after the fact, behind the Lean ZK server and through the Lean verifier, with one forged run |
| `verity bench rate_squaring` | squarings a second modulo the store codec's primes, each form gated before it is timed: `sq` on one GPU, `ifma` on the host's CPUs |
| `verity bench recursion_e2e` | recursive witness zero knowledge, timed step by step: `plan`, `run`, `result`, `table` and `store` |
| `verity bench remote_session` | PoUB's sessions: a verifier's and a prover's process over TCP, end to end |
| `verity bench replay_determinism` | whether a served MoE window's recorded schedule, replayed, gives its served bytes, op by op |
| `verity bench replay_ladder` | the same replay across a ladder of dense and MoE models, with the per-model status table |
| `verity bench route_capture` | what a model executes when served, step by step, and the routes class G's Definitions don't cover |
| `verity bench serve_tax` | what each change to serving costs vLLM's own server against stock compiled vLLM, at three loads |
| `verity bench served_commit` | served vLLM against served vLLM with every value committed, interleaved, through `verity-bench vllm` |
| `verity bench tensor_duty` | the tensor-duty sweep on one H200: dense tensor MACs a second by duty cycle, against the card's tensor clock |
| `verity bench unit_block` | a word primitive's Boolean version built as one Flock proof block, against Flock's proved range |
| `verity bench unit_lowering` | writing a PoUW H1 circuit file: peak memory and wall time by K, its SHA-512 held to a reference |
| `verity bench unit_walk` | the device witness's host walk of a served proof-block type's unit: CPU and wall seconds at each width |
| `verity bench verifier_cost` | a verifier's time against the size of the description it reads and of the circuit that description unrolls to |
| `verity bench word_codec` | the KV cache's Sloth codec on one GPU: its tests and its benchmark at each width |

A directory without a runner is run by the commands its README names, or its main module's docstring where it has no
README. `verity-bench` is the harness's own command. `judge.py`, beside the directories, compares recorded runs'
results.

## The harness

`benchmarks/harness/verity_bench` (the distribution `verity-bench`) is the shared harness. It measures what an arm, a
protocol option or an experimental install, adds to served inference against stock vLLM, through the vLLM
integration's public interfaces. A spec in `benchmarks/specs/` names a preset, its arms and its sweep, and expands to
points, each with an identity. `verity-bench run SPEC` records one run per point without a passing run, `check` passes
only when every point has one, `compare` and `table` accept results only when their identities agree, and
`served-table` and `breakdown` render from the evidence store. It also holds the code behind `verity bench overhead`.
The harness's kit (`harness/verity_bench/kit.py`) holds what runners share, each written once: the GPU lease, the host
stamp, the paired ratio and the result's writer, which records measurements and refuses a verdict of record.
`benchmarks/harness/README.md` is its contract: how a point is timed and what a result says.

## What a benchmark may import and write

- **It imports like any other caller.** `verity` from the owning module, an integration through its public modules
  (`tests/test_vllm_public_api.py`; only the integration imports vLLM, `tests/test_vllm_boundary.py`), the harness,
  and other benchmarks.
- **Nothing outside imports it,** by package name, by a path put on `sys.path` or loaded by file. A number a protocol
  uses reaches it through a card file citing an `art:` id, and code that a protocol, a device or a verifier runs is not
  a benchmark: it moves out to the package that uses it. `BENCHMARKS_IMPORTERS` in `tests/test_boundaries.py` names the
  files that still import a benchmark, and only shrinks.
- **It writes measurements, under its run's output directory** (`$OUT` under `research run`, else its `--out`). The
  run publishes them to the evidence store, and tables are rendered from the store on demand. Results, tables and
  plots are not committed (`tests/test_repository.py`); the tree keeps code, the parameters code reads and small
  fixtures beside their generators.
- **It writes no verdict.** Whether an audit passed is the verifier's to say. A benchmark that runs a
  verifier records what the verifier returned, and its own gates say only whether its measurement stands.

Some files here still break these rules: a kernel or a store the integration loads, a probe whose rate a card file
holds, the exporter of Lean data a guarantee reads. Each moves out to the package that uses it, and none is a pattern
for a new benchmark.

## Metrics

`metrics.toml` declares the numbers a lead is held to, one `[[metric]]` each, and the console's Metrics tab shows each
one's history from the evidence store. A metric's value is a measurement in a run's `result.json` named by its `id`,
`OWNER/NAME`, which the benchmark records it under (`serve_tax`'s `Model.metric`), so a value without a run doesn't
exist. Its keys:

- `id`, `title`, `owner` (a handle in `.agents/skills/using-slack/registry.json`, the id's prefix) and `unit`, the
  unit its measurements carry.
- `better`: `lower` or `higher`.
- `command`: the one command that measures it, whose script's directory records it under `id`.
- `every`: how often it is measured (`7d`, `12h`); a value older than that is stale.
- Optional: `floor`, the number it must not cross; `goal`, the plan's goal it serves; `tags`, a list of words.

`tests/test_metrics_registry.py` refuses a repeated id, an unknown owner, an id outside its owner's prefix, a `better`
other than `lower` or `higher`, and a command whose script is missing or whose directory never names the id.

## Index

Every directory has one row, in the group of what it measures; `tests/test_benchmarks_index.py` holds the rows to the
tree. A directory's status is one of:

- **maintained**: developed against the current tree, and its commands run from it;
- **frozen**: kept so that what it measured can be measured again, and changed only to keep it running;
- **data**: no measuring code, only numbers that other code takes as parameters.

### Protocol overhead

What the protocols cost: measured on served inference, and bounded on training.

| Directory | What it measures | Status |
| --- | --- | --- |
| `harness/` | `verity-bench`: specs, recorded runs, sweeps and results for an arm against stock vLLM, the served table of each protocol's overhead, and the kit runners share (`verity_bench/kit.py`) | maintained |
| `specs/` | the harness's specs and sweeps, one TOML file each | maintained |
| `overhead/` | `verity bench overhead`, the one front door for overhead: a card's operating points C and L, a scheme's γ over stock at them in busy SM-time, and the table of γ from labelled runs of record | maintained |
| `served_commit/` | what the Commit, every value of every step committed, adds to served vLLM: throughput, time to first token and per-token latency | maintained |
| `replay_determinism/` | whether a served MoE window's recorded schedule, replayed without batch invariance, gives its served bytes, and whether a request alone gives the bytes it got in its batch, op by op | maintained |
| `dsa_topk/` | the DSA indexer's top-k at the integration's vLLM pin made a function of its logits by a vLLM plugin: outputs run to run, its set against the registered relation's, and its cost against the pin's kernels | maintained |
| `moe_align/` | the Marlin MoE path's `moe_align_block_size` at the integration's vLLM pin made a function of its inputs by the integration's `verity_vllm.engine.moe_align` (also its vLLM plugin), each expert's pairs in pair-index order: outputs run to run, each expert's pairs against the pin's op, and its cost against it | maintained |
| `replay_ladder/` | whether replaying a saturated window under its recorded schedule gives the served tokens across a ladder of dense and MoE models, run after run, against each request served alone; and the per-model status table | maintained |
| `serve_tax/` | what each change to serving (step trace, release order, the certified tick, committing tokens and the KV cache) costs vLLM's own server against stock compiled vLLM at three loads, the serving cost (one number with a target: the served engine's best goodput against stock's, by vLLM's own benchmark client), and replaying a recorded window under its committed schedule | maintained |
| `prod_repro/` | one command over the serving cost benchmark's serve, load and replay: what the serve turned on (prefix caching, speculative decoding, the context limit), whether the replayed window's tokens are the served ones, what our changes cost over stock with its interval, and the advice a request, each measured or said not to be | maintained |
| `vllm_encoded/` | Llama-3.1-8B in vLLM with its weights and KV cache held as codec words and receipted by PoUB, as `verity-bench` arms beside the plain stack | maintained |
| `inference_only/` | the inference-only policy of Llama 3.2 1B as served, stated, and of the class this tree builds, with the Lean count's verdict on it, and each one's number: the least slowdown of training run inside the policy's inference blocks, under the min-cut bound | maintained |
| `training_slowdown/` | the grader (`grader/`): a training step's matrix products, their multiply-adds and output wires, and the circuit's size, read off the Boolean circuit a lane hands over, beside training's slowdown code's trace of the same step; `verity bench training-slowdown CONFIG`, an attack config's jobs, each found by its label and run where it is missing | maintained |

### Kernels

GPU code held bit for bit to its reference or Definition before it is timed.

| Directory | What it measures | Status |
| --- | --- | --- |
| `hash/` | GPU kernels for PoUW's Dither prover (`dither/`: the no-fold and J10-R units, their gates and harnesses), the served path's SHA-512 commitment library (`lib/`), and vLLM's int8-exact linear kernels (`exactlin/`) | maintained |
| `flock_kernels/` | Flock's CUDA prover kernels on one GPU, each printing a digest of its output beside its time so that builds against two trees' kernels compare word for word: hm96's leaves (`hm96_leaves.cu`) | maintained |
| `torch_gpu/` | the H200 demo's GPU kernels, each bit-exact to its Definition: the serve-time commit (`commit_slots`, `sha512_rows`), the store's load with bandwidth receipts (`p2_stage`), the decoder's non-matmul ops (`units`), attention (`attn_bf16`), the exact W8A8 linear (`s8_linear`) and J10-R's units on the H200 (`j10r_sm90`) | maintained |
| `pouw/` | Pearl-C's kernels on the RTX PRO 6000, their harness, and the served vLLM path and its audits; `contents.py` lists the rest | maintained |
| `tc_probe/` | tensor-core instructions bit for bit against their silicon models, and the trust each instruction's evidence earns | maintained |
| `tc_probe_fp4/` | the FP4 path on Blackwell: its conversions bit for bit, and each sm_120a multiplier priced against dense NVFP4 | maintained |
| `native_peak/` | a card's measured dense-GEMM throughput for one datatype (`native-peak/v1`) | maintained |
| `int8_class_rate/` | the H200's sustained int8 rate per operand class at the declared clock, alone and beside SHA-512 at a ratio of compressions a T MAC (timings only) | maintained |
| `tensor_duty/` | the falsification test of the H200 card's tensor clock: dense tensor MACs a second over the host's wall clock by tensor duty cycle, operand class and fill, at maximum power, against 132 x 4096 x `clock.tensor` (timings only) | maintained |

### Proving and recursion

| Directory | What it measures | Status |
| --- | --- | --- |
| `one_stage/` | the one-stage audit end to end: committed values, registration, the sample, Flock's circuit prover and the Lean verifier, to the audit record and its integrity profile, with its negatives; and a served window's two-stage audit (`two_stage.py`: each class's first stage, and a class's second stage through that audit, round 2 and the stream's outcome; `two_stage_small.sh` runs both at small statics) | maintained |
| `two_stage/` | the two-stage audit end to end on a toy replay block: the boundary registered, a first-stage sample of replay blocks, their interiors replayed and registered, a second-stage sample per (replay block, template) cell, one Flock `--zk` session per template with link ports read at the boundary record, the Lean verifier, and the stream's outcome, with its negatives | maintained |
| `prove_path/` | a served run's proof blocks proved after the fact: the compiler's Programs, a sample on live coins, Flock `--zk` behind the Lean ZK server, and the Lean verifier | maintained |
| `served_units/` | a served model's proof blocks proved one type at a time from synthetic values, the Lean verifier accepting the honest run and refusing forgeries | maintained |
| `unit_walk/` | the device witness's host walk of a served proof-block type's unit: CPU and wall seconds a statement at each width (lane groups a pass), distance asked ahead and pool size, with and without its row vectors kept, every walk's packed words held to the first's | maintained |
| `spot_proved/` | a replayed served step's values proved: one replayed step of a recorded replay block captured per TP rank under CUDA graphs, staged as each sampled type's proof blocks, and audited one stage end to end by the Lean verifier | maintained |
| `multi_unit/` | an FP8 GEMM's proof blocks staged as Flock statements in several layouts: prover and verifier time, peak memory and proof bytes | maintained |
| `flock_s8_block/` | Flock `--zk`'s price for a proof block of adjacent cells of one int8 GEMM output row, on one GPU node and on the CPU | maintained |
| `unit_block/` | a word primitive's Boolean version built as one Flock proof block: the select's slot alone and with its rows' openings, against the 2^27 block | maintained |
| `prove/` | `verity bench prove`, the one command for a proof system's prover on a workload: per block type the blocks proved per token, the ms per block and that figure's source, and the prover GPUs per serving GPU with every block proved or at a sampling plan, from the store's labelled records | maintained |
| `proof_throughput/` | the proof blocks a second Flock proves and the Lean verifier accepts on GLM-5.3's dot products, over the batch size | maintained |
| `commitments/` | the commitment schemes' cost: hashing leaves and building the tree, hiding leaves per row, and serving's per-step row tables | maintained |
| `sampling_plan/` | the sampling plan for one served workload: a cost model fitted to proving measurements, and ε at δ over the verification overhead (a model, not a measurement) | maintained |
| `spot_figures/` | the SPOT post's Figures 1-3 for a model from its cards (`spot-card/v1`) and SPOT's planner: what an adversary serving it on all of a scope's compute can exfiltrate, infiltrate or skip against total cost (each server configuration's serve tax plus the share of compute for verification) and against proof overhead, proof overhead per proof-block type, and the serve-time commitment's cost over fast serving (a model, not a measurement) | maintained |
| `exfiltration/` | `verity bench exfiltration`: GLM-5.3's exfiltration capacity in GB over a monitoring period at 10% of total compute, the compute audit's part (the break-even planner's rows of record, from the store) beside the network certifier's part (`certifier_term/`'s rows), every input labelled measured, modeled or assumed | maintained |
| `verifier_cost/` | a verifier's time against the size of the circuit description it reads and of the circuit that description unrolls to | maintained |
| `recursion_e2e/` | recursive witness zero knowledge, timed step by step: the inner proof, V* proved with `--zk`, and the Lean verifier | maintained |
| `recursion_figures/` | a recursion bench result's session scaled over the census's fleets and the monitoring periods: its cost, knowledge-soundness error and stop-leak bound per window and pooled | maintained |
| `private_circuit/` | the universal unit, which runs any private circuit of a size class, proved under `--zk` and verified by the Lean verifier | frozen |

### Circuits

| Directory | What it measures | Status |
| --- | --- | --- |
| `bool_build_cost/` | a workload's Boolean Build against its word Build: the descriptor's size and each Definition's proof-block rule | maintained |
| `compile_cost/` | what compiling a class-G window's Programs costs (`predict.steps.compile_steps`, GLM-5.3 at TP8 on one rank): wall time and peak RSS, by phase, its digests held to their pins | maintained |
| `circuit_properties/` | structural properties of a Boolean circuit's counted operations, measured exactly: fan-out, fan-in, their count and s_× | maintained |
| `route_capture/` | what a model executes when served: every engine step's launch-shape class under CUDA-graph serving, and the kernels of the first steps of each class; and the gap: every (route, statics) pair the captured steps need that class G's registered Definitions don't cover, with its owner | maintained |
| `unit_lowering/` | writing a PoUW H1 circuit file (a leaf's lane and combine, or the whole tile unit, at any tile R × C): peak memory and wall time by K, the file's AND rows, bytes and block (k_log), its SHA-512 held to a reference, and where the memory goes | maintained |

### Compute

| Directory | What it measures | Status |
| --- | --- | --- |
| `compute/` | `verity bench compute`, the front door for the compute audit's number: the share of a served model's MACs (its linears', attention's and LM head's) that Pearl+ credits over the tokens a served run served, from the verifier's side of a recorded run's record | maintained |
| `machine_model/` | the ACCEL model of each measured card: its rows, census, attack families and designs as data, and a design's ε from them (`machine-model`) | maintained |

### Network

| Directory | What it measures | Status |
| --- | --- | --- |
| `network_traces/` | the network certifier's runs in front of a served endpoint, and their zero-knowledge proofs | maintained |
| `network_e2e/` | network certification end to end on a served model, as one command and one report, with its negatives | maintained |
| `certifier_term/` | the network certifier's term for GLM-5.3 at the load of a node, a datacenter and the world, in bits and GB, each row by its check's own counting rule: the scan and N2 as the monitoring-period check charges them, the scan under each change to its check, and the design with no run behind it, without its stop block and with it (`network_traces/leak_bounds.py`'s runner) | maintained |
| `network_sim/` | the device graph, and a mock datacenter of Linux network namespaces that runs the real network certifier | maintained |
| `certifier_timing/` | the bits an interval a server moves through when the network certifier releases its bytes: one number from a server that times its writes by a bit, with a guard, a target and the candidate fixes as flags | maintained |
| `mempool/` | the certifier's prefix-cache pool as memory-checked RAM (`pool_ram`, held to vLLM's pool on the encodings' traces; `stream.py` holds `CacheStep_v3`'s stream to it after every call): a window's cost at its busiest stretch, and `MemCheck_v1`'s and `CacheStep_v3`'s vectors | maintained |
| `datacenter/` | a datacenter as Verity accounts for it, in a small Python API: the Program, hardware, binding, plan, run, claims and account | maintained |

### Memory

| Directory | What it measures | Status |
| --- | --- | --- |
| `pous/` | every PoUS implementation on one dataset, by its effective weight bandwidth | maintained |
| `bandwidth/` | the device ceilings PoUB's timing cites: how fast a card and a host write labels, and the chip's on-chip stores; and a node's own report of its interfaces' line rates and its cards' usable memory | maintained |
| `remote_session/` | PoUB's session harness: a verifier's and a prover's process over TCP, end to end. The remote protocol's demo is the vLLM demo, not this | maintained |
| `memory/` | `verity bench memory`, the front door for the memory audit's number: the share of a node's memory read rate one accepted PoUB session leaves uncertified, at the data sheet's read bandwidth, from a gate run on the whole node, a recorded run or a loopback | maintained |
| `kv_filler/` | the filler top-up beside vLLM serving, and what it costs inference | maintained |
| `stacked_share/` | the share of a node's live KV a prover could hold as encoded carriers inside the audit's compute budget | maintained |
| `word_codec/` | the store's Sloth codecs (the word codec, the KV codec that PoUB's `codec.py` mirrors, the wide codecs): each a byte-exact reference, a CUDA kernel and its benchmark | maintained |
| `codec_tensor/` | PoUB's receipts, leaves and receipt tree on the GPU, and the tensor-core wide-word codec | maintained |
| `property_redteam/` | attacks on the stacking property and on the gate model's counts behind the encode bound | maintained |
| `rate_squaring/` | squaring rates modulo the codec's primes on the GPU and on the host's AVX-512 IFMA, the evidence for PoUB's stated squaring ceiling | maintained |

### Numerics

| Directory | What it measures | Status |
| --- | --- | --- |
| `ppl/` | the perplexity an intervention on a served model's matmuls or attention costs, against bf16 and plain int8 | maintained |
| `f32_exhaustive/` | a correctly rounded reference primitive (`F32LogRn_v1`, `F32ExpRn_v1`) against MPFR, or a Boolean version (`F32Sqrt_v2`) against its word, on all 2^32 binary32 words | maintained |
| `definitions/` | the GPU probes behind the numerics Definitions: the approximate instructions FlashAttention-2 and the RMSNorm kernel run (`MUFU` exp2, reciprocal, square root and tanh), measured on every 32-bit input | maintained |
| `numerical/` | the bench spine (`verity_numerical.bench`) that Flock and `circuit-check` import, and the per-unit-type cost inventory; the archived proof systems' tables are in the evidence store | frozen |

### Census data

| Directory | What it measures | Status |
| --- | --- | --- |
| `census/` | the fixed quantities benchmarks are computed against: hardware, datatypes, networks, deployments, subcircuits and workloads | data |
