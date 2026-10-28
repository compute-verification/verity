# One-stage audit runs

The one-stage audit (`verity.core.protocols.sampled_proofs.one_stage`) end to end: serving's committed values through registration, the verifier's
draw, C-Flock's circuit prover (M0) and the Lean verifier, to the audit record with its integrity profile, then the
negatives.

- `a<N>.py`: the driver of run A<N>; its docstring says what the run adds and how to call it.
- `compose_circuit.py`, `stage_subset.py`, `stage_tampered.py`: helpers that run inside M0's tree (its `verity_flock`).
- `r2_population.py`: the other population every driver's and `replay.py`'s registration-partition-mismatch negative (R2)
  registers, its first template one instance short, or one over where it has a single instance.
- `pod.sh`: the entry point on a CPU pod.
- `a5.sh`: run A5's entry point on a CPU machine. A5 is a deferred Commit's window, a stand-in linked to its run root, the
  commitment its record was committed in and its replay bundle: the CPU RoPE d64 Commit fixture, committed under rows-sha512
  (`COMMIT_SCHEME` names no other: vllm-v1 is kept only to read recorded runs), its window from `verity-vllm serving-rows`
  (with `WINDOW=served`, the in-Commit hook's), and M0 and Lean built from the run's tree. With `A5_ROW_DIR` it takes a
  single-GPU deferred Commit's row dir and sealed bundle in place of the fixture, and with `WINDOW=served` the window its C2
  site committed (`A5_WINDOW`).
- `a5_llama1b_h200.sh`: A5 over a GPU-served window, in one job on an H200 node, from a clone of the tree
  (`bash benchmarks/one_stage/a5_llama1b_h200.sh`). Llama-3.2-1B int8's B4 row on one H200 under `gpu-lease`: its Build and
  Match, then a deferred config-run Commit under rows-sha512 whose `--serving-rows` window
  (`llama32_1b_rope_d64_l0.json`: layer 0's RoPE d64 heads at steps 1-7) its C2 site commits before the bundle is sealed,
  a `served` window under the 64-byte run root. Then the Commit's CPU replay, and `a5.sh` with `WINDOW=served` on the bundle
  and that window, whose private bodies stay on the node.
- `a5_tp.sh`: A5 over a TP Commit's windows, one `served` window per rank (`a5.py --tp-commit`), taken as served only under a
  SHA-512 TP run root (`tp.run_root`'s v2, over the ranks' rows-sha512 roots; an old record's SHA-256 v1 root is refused before
  the draw): GLM-5.3's TP8 Commit with
  `SERVING_ROWS=benchmarks/one_stage/glm53_silu_mul_v2_tp8.json` (its SiluMul_v2 partition: every unit of `SiluMul_v2{I=1536}`,
  the dense layers' 861 per rank, by `subset:512`), or by default the CPU TP stand-in
  (`integrations/vllm/tests/pipeline/served_tp.py`). A member's whole population is one M0 statement, whose shape takes at most
  `2^(35 - k_log)` instances: 2,048 at I=1536 (`k_log` 24) and 8,192 at I=256 (`k_log` 22), so a rank's `SiluMul_v2{I=256}`
  (172,200 routed-expert and 21,525 shared-expert units) is not a member. Each session's `circuit.txt` files and lookup tables
  go to the store by SHA-512 (`sha512/<hex>`, the Lean verifier's `--archive` layout) through
  `integrations/vllm/verity_vllm/ops/store_trees.sh`, epoch_store.sh's put, and the run is labelled `circuits` with each tree's
  art id, so the run can be verified again from the store.
- `a5_row_tables.py` (`a5_tp.sh` with `STANDIN=row-tables`): A5 on a window with row tables, which no TP window of record has
  yet: `integrations/vllm/tests/pipeline/test_row_tables.py`'s ScaledMmS8Coordinate chunks, whose `x` rows link four row
  tables over two engine steps, each chunk one table's, as rank 0 of a one-rank step on a local challenger's coins. Lean
  verifies each drawn chunk with its table's registered reads (`--registered`, `tables.registered_reads`).
- `a5_warden.py` (`a5_tp.sh` with `STANDIN=warden`, `warden:altered`, `warden:pad`, `warden:pad-named` or `warden:swapped`):
  A5 on a window with transfer gates, which no served window has yet: two chunks of the warden's toy unit whose ports are the
  bucket rows of one connection's ingress and egress cells (a row a bucket, registration v2), committed under the warden's
  salts, proved whole (the work law's selection All) beside a drawn rest, on a local challenger's coins or the verifier's
  own. The units' wiring is the Program's (`warden_rows.Units`, from public structure alone: ingress cells 0 … n − 1, egress
  cells D·r … D·r + n − 1). Lean verifies each chunk with the verifier's own registered reads at the root the warden signed,
  named by the units' value and each unit's read (connection, cell) placed by the wiring the warden signed
  (`warden_rows.port`, after `check_registration` under the warden's key), which refuses a foreign, a duplicate and a
  padding-slot read by name; the altered run, and a statement made for another units' wiring (a pad in a data cell's place,
  its rows under the Program's value, two units swapped: `--made-for`), are refused by Lean's registered reads. Its warden run
  is rehearsed against a fake endpoint, or (`A5_WARDEN_RUN`) is `warden_replay.py`'s: a served window's requests through the
  warden, each answered with the tokens the server streamed it.
- `a5_tp_gates.py` (`a5_tp.sh` with `A5_COINS=verifier` and `STANDIN=tp-gates`, `tp-gates:altered`, `tp-gates:swapped`,
  `tp-gates:other-collective`, `tp-gates:other-request` or `tp-gates:own`): A5 on both ranks of a synthetic Llama TP2 step
  of two requests whose cross-rank transfers hold transfer gates, which no served TP2 window has yet. Each rank's partial of
  each reduction (layer 0's embedding, o_proj and down_proj, H = 2048) for each request is a registered value its
  registration names; each rank's chunk of a request's reduction, the registered `AllReduce2_v1{H}` with its input's two
  rows staged as two ports, is
  proved whole beside a drawn rest, and Lean verifies it with the verifier's own registered reads of both ranks' partials of
  that request at the roots each owner registered, every registration read before any rank's coin. A received row moved one
  bit (`--altered`), two tokens' rows swapped, another collective's partial, another request's, and the receiving rank's own
  partial read as its peer's (`--made-for`) are each refused by Lean's registered reads on rank 1, rank 0 accepted.
- `a5_glm53_commit.sh`, `a5_glm53_cpu.sh`: A5 on GLM-5.3 in its two halves. The first is the GPU job, from the source tree
  (`bash benchmarks/one_stage/a5_glm53_commit.sh`): the TP8 Commit with served windows, their
  private bodies kept on the GPU machine off custody. The second, `a5_glm53_cpu.sh GPU_RUN` from a VM, copies them to the CPU
  machine (`A5_ON`) and runs `a5_tp.sh` there in one research run.
- `examples/vllm/commit.sh` (the vLLM example's): the single-GPU counterpart of `a5_glm53_commit.sh`'s deferred Commit, on Llama-3.2-1B bf16
  bi-eager TP1 (one RTX PRO 6000 of vy-nebius-1): the workload's Build and Match of record, a config-run Commit that seals
  `commit/replay_bundle_p0/` (a single-GPU workload defers only as a config run), its replay on CPU, and the row dir in the evidence
  store. A deferred single-GPU Commit's window comes from its bundle through `verity-vllm serving-rows`, not `tp-serving-rows`.
- `served_verdicts.sh` (with `served_verdicts.py`): one served window through every subprotocol, in one job on an H200 node
  from a clone of the tree (`bash benchmarks/one_stage/served_verdicts.sh`). It serves
  Llama-3.2-1B int8 on one H200 under `gpu-lease` (`serve-openai --scheme rows-sha512 --seal --admit release-order`), sends a
  few greedy requests from the certifier's traffic generator, and seals each window. Before the first request, the committer
  registers its weights under its own key (`prove_path.py weights-registration`). Without a registration, `NO WEIGHTS
  REGISTRATION` is printed and every weight read counts free. On the window holding the most requests, none carried, it runs
  sampled proofs: the window's advice under that registration (`--weights-of-record`) and its request Programs, V3b's window
  circuit at each class of `CLASSES` (V2's 4,32,4 by default), with its weight reads counted registered at the registration's
  pins. The run needs every one of those counts. The class of record's count follows them as a measurement under its own
  `WC_RECORD_TIMEOUT` (default 20m), after the record: a timeout or refusal there is reported and never fails the run. Then comes the records audit (`prove_path/run.sh`, `PP_RECORDS=1`), which is C-Flock `--zk` on the drawn proof blocks
  and the Lean verifier. Then come PoUW, the bandwidth audit, PoUS and the network certifier. Each runs its piece where
  `POUW_CMD`, `BANDWIDTH_CMD`, `POUS_CMD` or `CERTIFIER_CMD` names one, within its own limit; otherwise its `MISSING` line
  names the command it lacks and its owner, and it gets no verdict. The script sets each owner's command by default where
  the tree has its piece and the node holds what the piece reads (its auditor key, the bandwidth audit's registered store).
  Finally the all-protocols window report and the block record, which accepts only when every subprotocol accepted on this
  node and names each one missing, rejecting or about another server, with its reason (PoUW's development piece: development
  run, credits nothing). An accept's clause says when its weights were held and their share of the checkpoint, where its
  profile says so (PoUS's), and each δ of the block record's at which its quantity is not established. V3c runs last, within `V3C_TIMEOUT` and `V3C_AUDIT_TIMEOUT`, since nothing before it reads it. Each
  step's verdict line is in `out/verdicts.txt`.
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
  host job, or a preemptible guest job on a shared node). A6's W8A8 coordinates at K = 4096 and 14336 are
  staged in statements of 1 to 512 units, and sessions of 1 to 512 drawn units are proved by `flock-circuit prove --zk
  --gpu`, with the firewall's send check on and off. The Lean verifier checks them, and the job fits the per-session,
  per-statement and per-unit costs with their standard errors, writing `costfit` observations. `ZK_DRAWS=c` proves every
  staged unit: the cost of c drawn units in one statement, up to a table's 2^(35 − k_log) (2048 at K = 4096). Each point
  takes its own GPU lease, and a node without CUDA 13.3 gets its packages unpacked under `FLOCK_WORK`.
- `prove_bundle.sh`: a drawn-unit replay bundle (`examples/h200/interfaces`, `verity/drawn-unit-bundle/v2`) proved by
  `prove_bundle.py`: `PB_BUNDLE` of the window `PB_WINDOW`, else the store's `BUNDLE_WINDOW` (a W8A8 MLP's window under a
  test key, its tampered twin and the interfaces' bundle of four drawn units under the window's registered `subset:4`).
  `tests/test_prove_bundle.py` holds that bundle to the window's public part (`tests/fixtures/bundle-window-public`). `PB_LIVE=1` requires the bundle's draw to be live. The bundle is held to the prover's window: its own check (the draw's
  law the registered one among it), its run, registration and partition, each drawn member's public body and a live
  draw's receipt and record. Each drawn member is one `prove --zk` session behind the Lean firewall, one at a time, served the bundle's
  draw, with Lean `verify --zk` as the verdict, overlapping the next session. The negatives are `zk_mutants.py`'s one-bit
  proof edits (`PB_FLIPS=N`: the first N), refused by Lean, and a tampered window's member, stopped by the firewall and
  refused by Lean without it. `PB_BUILD_ONLY=1` only builds the prover, Lean verifier and firewall into its `out/bin`, stamped
  with the trees they come from. A later run with `PB_BINARIES=` that directory uses them while the stamp matches its own
  source, so M3's reserved window doesn't spend time on the build.
- `zk_cost_cpu.sh`: the same sessions on a CPU node (`zk_cost.py prove --cpu`), proved by the CPU prover that
  `prove_path/run.sh` builds, with the firewall's send check. It runs in a proofs slot, one point at a time. Each session's
  record carries its wall time and its summed peak memory (prover, live verifier and firewall), which is what a CPU node
  spends on one drawn statement of a block's draw.
- `forged_proofs.sh`: the fixture of C-Flock's forged-proof negatives (`verity/ml/flock/tests/test_lean_forged_proofs.py`),
  on hidden_zk's cached builds (`forged_proofs.py`). A small A8 unit (`LlBf16DotprodDot_v1` at K = 128) staged bound to its partition,
  with one honest drawn `--zk` session and two whose committed output word is wrong, proved with the prover's self-check
  skipped (`FC_ZK_SELF_CHECK=skip`), so the Lean verifier gets a well-formed proof of a false statement.
- `a11.sh`: run A11's entry point, on A6's builds. A11 is GLM-5.3 TP8 B1's served step, rank 0: `a11_kinds.py` censuses
  the request Program's root Calls by Definition kind and tries one unit of each through M0 (a kind whose word units M0
  refuses, through its Boolean version's); `a11_slice.py` cuts the
  capture's values and the checkpoint's tensors for a few layers off the serving node; `a11.py` stages each lowering kind's
  units from those values, one template each, under the stratified law, proved under `--zk` and held by the Lean verifier;
  `a11_table.py` renders the per-kind table and its gap lines from the audits' summaries.
- `a9.sh`: run A9's entry point, on the same builds. A9 is GLM-5.3's routed-expert FP8 GEMMs in the registry's Boolean
  static pair form (`boolean_fp8_moe`): `a9.py` holds one coordinate of each launch's first routed pair to the capture in
  its word and Boolean forms, cuts an up/gate coordinate whose rows overflow M0's block into its gather and
  block-coordinate units, and saves the units for `a11.py --load-values`, whose audit proves them. `A9_SAVED` audits a
  saved directory without the values stage.
- `a10.sh`: run A10's entry point, on the same builds. A10 is A11's kinds and A9's launches on rank 0's committed values
  in place of the capture's: `a10.py values` opens rank 0's replay bundle from the Commit (`--replay-deferred`,
  `retain=host`) against its run root of record, reads each Call's operands and returns through its Program row as the
  deferred replay does, and runs `a11.py`'s and `a9.py`'s staging on them, writing every word where the committed value
  and the capture's differ to `compare.json`; `a10.py audit` is `a11.py --load-values` on a saved directory, with the
  registration window naming the committed values.
- `m1_families.py` and `m1_unit.py`: M1's full-step coverage. `m1_families.py` reads a rank's Programs of record from a
  deferred TP Commit's Build and lists the unit families of the given steps: each Definition the serving rows' partition
  draws from, its rows, op paths, ports and whether `serving_rows.template` takes it; `--estimate` adds each family's
  ANDs, table reads and assertions from its leaves lowered once (an upper bound on the flat count), against 2^26, and
  the log2 of M0's block for its unit against `K_MAX` (the hashing of its rows alone bounds that block from below), the
  same for each of its split units, and with `--also`/`--also-file` for Definitions a split main lacks would use.
  `m1_unit.py` builds one family's unit through M0's `UnitStager`, flat or `--structural`, and with `--bin`
  and `--lean` proves one round of it on stand-in bf16 values under `--zk` and verifies it with the Lean verifier.
- `served_stock.sh` (with `served_overhead.py`): `served_verdicts.sh`'s stock control, the same endpoint, engine and load
  served with `serve-openai --no-commit` and no protocol, once per fresh endpoint for `REPS` endpoints. `served_overhead.py
  table` compares the composed run with it on two clocks, each the same code in both runs: the client's
  (`serve_load.py --varied`'s timings files) and the engine's step trace (`advice_trace.jsonl`). It reports time per output
  token, time to first token and tokens per second as composed over stock, the composed run's window open and close on
  the engine loop, and `verity-bench served-table`'s single-protocol rows beside them.
- `served_util.sh` (with `served_util.py`): the whole-step tensor utilization of serve-openai on the int8 Llama-3.2-1B
  prefill mix (prompts of 1,024 ids, `max_tokens` 1, all queued at once, T tokens a step), one arm a run (stock, the
  Commit unsealed or sealed, the Commit with PoUW H1 on in-process coins), under Nsight Systems' GPU metrics.
  `served_util.py` takes Tensor Active's mean over the longest run of full steps, each step from the previous step's end
  to its own on the step trace's clock, and splits the rest into losses by what the GPU was running at each sample.
  With `PYSPY=1`, `pyspy_steps.py` splits the same steps' host time by Python function and thread, from py-spy's trace.

C-Flock's circuit prover (M0, `verity_flock.circuit`) is on `main`, but a run does not take it from its checkout:
`pod.sh` takes `git archive`s of M0's tree and the Lean verifier's tree at pinned commits, and a captured input set, as
inputs sent with the run; it builds `flock-circuit` and `flock-verify` once per pod, then runs the driver, whose
`--m0-pythonpath` names that M0 tree. The drivers' checks go through the verifier of record
(`one_stage.verifier`), which builds the sampled-proofs Lean in the run's tree on its first check, hence `--cwd clone`:

~~~sh
bash benchmarks/one_stage/pod.sh    # a job run from a clone of the tree, with m0.tar, lean.tar and set.tar sent
~~~

`check` runs nothing here.
