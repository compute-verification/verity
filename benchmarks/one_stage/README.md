# One-stage audit runs

The one-stage audit (`verity.protocols.proofs.spot.verifier.one_stage`) end to end: serving's committed values through docket, the verifier's
sample, Flock's circuit prover and the Lean verifier, to the audit record with its integrity profile, then the
negatives.

Its sampled sessions are SPOT[Flock]'s on its own, in its three roles: serving commits its values (the inference server);
each sampled session's prover runs `flock-circuit prove --zk` itself (the proof server: `a0.DIRECT`, `zk_cost.py`'s `dev` arm, which
Figure 1 prices); and the challenger's live coins, its signed log and the Lean verifier decide (the verifier). There is no ZK
server and no recursion on that path. The core protocols' path, every prover behind the Lean ZK server's send check, is
`--zk-server` on SPOT's window audit (`a5.py`), its variants here and `prove_bundle.py`, and `ZK_SERVER=1` on their shell entry points. The network
attester, where a run has one (`a5_network_attester.py`), is a layer of its own, whose signed stamp SPOT[Flock] reads as a certified
input. What the guarantee rests on: `verity/protocols/proofs/spot/verifier/README.md`.

- `a<N>.py`: the driver of run A<N>; its docstring says what the run adds and how to call it. A0's and A5's are SPOT's
  template and window audits, here because they load the adapter, Flock's prover and the benchmarks' scripts
  (`verity/protocols/proofs/spot/prover/audit/README.md`).
- `stage_subset.py`, `stage_tampered.py`: helpers that run inside the circuit prover's tree (its `verity_flock`), as
  `compose_circuit.py` does.
- `pod.sh`: the entry point on a CPU pod.
- `a5_llama1b_h200.sh`: A5 over a GPU-served window, in one job on an H200 node, from a clone of the tree
  (`bash benchmarks/one_stage/a5_llama1b_h200.sh`). Llama-3.2-1B int8's B4 workload on one H200 under `gpu-lease`: its Build and
  Match, then a deferred config-run Commit under rows-sha512 whose `--serving-rows` window
  (`llama32_1b_rope_d64_l0.json`: layer 0's RoPE d64 heads at steps 1-7) its C2 site commits before the bundle is sealed,
  a `served` window under the 64-byte run root. Then the Commit's CPU replay (the prover's pre-flight check, on the node), and
  `a5.sh` with `WINDOW=served` on the bundle and that window, whose private bodies stay on the node with the row dir: of the
  workload's files the run's record holds only the ones `adapters/vllm/tools/served_run/published.sh` lists, each with the reason it holds nothing
  of a request, and no file of the workload's `commit/` or `match/`, whose records, bundle, slim keep and request spec hold values.
  Its work directory is made once on the node by `(cd adapters/vllm && WORK=DIR bash verity_vllm/ops/pod_bootstrap.sh --cases
  LLAMA32_1B_INT8)`, and the script then runs with the same `WORK` (default `/workspace`). The bootstrap makes all the script reads
  under `WORK`: vLLM's environment (`venv312`), the int8 checkpoint (`hf`) and the sm_90 taps (`cp`: the guarded FA3 build and the
  norm tap). Before any build, the script refuses each one it doesn't find, naming that command. Under `gpu-lease` it leases one card
  (`A5_CARD=N` names it, else any free one) for its Match capture and its Commit's engine only, and every lease names the
  cores `A5_CORES=CPUS` gives (`--cores`; CPUS the cores the machine's map gives the lane, e.g. `40-59`), since a lease
  without them holds every other lane out; without `A5_CORES` the script refuses before any build. Its builds, replay and
  audit run on the cores the job is given (`nproc`). With no `gpu-lease` it takes card `A5_CARD` (default 0) directly. The row dir,
  the window and the audit's private files are under `WORK/a5-llama1b/RUN` (`A5_ROOT`). They are removed at exit, success or
  failure, unless `A5_KEEP=1`. M0's Flock build (`A5_FLOCK_WORK`, default `WORK/a5-llama1b/flock-work`) is kept for the next run.
- `a5_tp.sh`: A5 over a TP Commit's windows, one `served` window per rank (`a5.py --tp-commit`), taken as served only under a
  SHA-512 TP run root (`tp.run_root`'s v2, over the ranks' rows-sha512 roots; an old record's SHA-256 v1 root is refused before
  the sample): GLM-5.3's TP8 Commit with
  `SERVING_ROWS=benchmarks/one_stage/glm53_silu_mul_v2_tp8.json` (its SiluMul_v2 partition: every proof block of `SiluMul_v2{I=1536}`,
  the dense layers' 861 per rank, by `rate:512/861`), or by default the CPU TP stand-in
  (`tests/adapters/vllm/tests/pipeline/served_tp.py`). A member's whole population is one `verity/flock-circuit` statement, whose shape takes at most
  `2^(35 - k_log)` instances: 2,048 at I=1536 (`k_log` 24) and 8,192 at I=256 (`k_log` 22), so a rank's `SiluMul_v2{I=256}`
  (172,200 routed-expert and 21,525 shared-expert proof blocks) is not a member. Each session's `circuit.txt` files and lookup tables
  go to the store by SHA-512 (`sha512/<hex>`, the Lean verifier's `--archive` layout) through
  `adapters/vllm/verity_vllm/ops/store_trees.sh`, epoch_store.sh's put, and the run is labelled `circuits` with each tree's
  art id, so the run can be verified again from the store.
- `expert_tables_tp.sh`: the committer's per-expert tables registration of a TP model, one file a rank (`verity-vllm
  tp-register-tables --served`): each rank's expert stacks composed from the checkpoint of record, held to a capture's served
  SHA-512s and committed under the committer's key, which never leaves the node. A window's per-expert table reads count
  registered against these files (`records_audit.py audit --registered`).
- `a5_row_tables.py` (`a5_tp.sh` with `STANDIN=row-tables`): A5 on a window with row tables, which no TP window of record has
  yet: `tests/adapters/vllm/tests/pipeline/test_row_tables.py`'s ScaledMmS8Coordinate chunks, whose `x` rows link four row
  tables over two engine steps, each chunk one table's, as rank 0 of a one-rank step on a local challenger's coins. Lean
  verifies each sampled chunk with its table's registered reads (`--registered`, `tables.registered_reads`).
- `a5_topp.py` (`a5_tp.sh` with `STANDIN=topp` or `topp:slots`): A5 on a window of top-p select rows cut into their stages,
  which no served window has yet: `tests/adapters/vllm/tests/pipeline/test_serving_rows_stages.py`'s rows, each carried row wired
  to the proof block that wrote it, as rank 0 of a one-rank step on a local challenger's coins, run as `a5_sparse_mla.py` runs its
  window. `topp` names each stage on bits, the form Lean's `verify --zk` takes; `topp:slots` names the stages as words, whose
  table reads go in lookup slots, and Lean refuses each sampled proof block's statement. The window holds the stages whose
  proof blocks are composed: a stats block from 56 lanes a slice, so by default V = 64 and split count 1's stats, probs and
  step blocks, and the select chunks only where every split count's are (`A5_TOPP`, the script's `--v`, `--splits` and
  `--rows`). `--select-only` holds the select chunks alone, composed at any V, each wired to the chunk before it, with the
  rows they read from the step blocks committed as plain input rows that no proof block of the window writes.
- `a5_network_attester.py` (`a5_tp.sh` with `STANDIN=network_attester`, `network_attester:altered`, `network_attester:pad`, `network_attester:pad-named` or `network_attester:swapped`):
  A5 on a window with transfer gates, which no served window has yet: two chunks of the certifier's toy Definition whose ports are the
  bucket rows of one connection's ingress and egress cells (a row a bucket, docket v2), committed under the certifier's
  salts, proved whole (the work sampling design's selection All) beside a sampled rest, on a local challenger's coins or the verifier's
  own. The proof blocks' wiring is the Program's (`network_attester_rows.Units`, from public structure alone: ingress cells 0 … n − 1, egress
  cells D·r … D·r + n − 1). Lean verifies each chunk with the verifier's own registered reads at the root the certifier signed,
  named by the proof blocks' value and each proof block's read (connection, cell) placed by the wiring the certifier signed
  (`network_attester_rows.port`, after `check_registration` under the certifier's key), which refuses a foreign, a duplicate and a
  padding-slot read by name; the altered run, and a statement made for another wiring of the proof blocks (a pad in a data cell's place,
  its rows under the Program's value, two proof blocks swapped: `--made-for`), are refused by Lean's registered reads. Its certifier run
  is rehearsed against a fake endpoint, or (`A5_WARDEN_RUN`) is `network_attester_replay.py`'s: a served window's requests through the
  network attester, each answered with the tokens the server streamed it.
- `a5_tp_gates.py` (`a5_tp.sh` with `A5_CHALLENGER=local` and `STANDIN=tp-gates`, `tp-gates:altered`, `tp-gates:swapped`,
  `tp-gates:other-collective`, `tp-gates:other-request` or `tp-gates:own`): A5 on both ranks of a synthetic Llama TP2 step
  of two requests whose cross-rank transfers hold transfer gates, which no served TP2 window has yet. Each rank's partial of
  each reduction (layer 0's embedding, o_proj and down_proj, H = 2048) for each request is a registered value its
  docket names; each rank's chunk of a request's reduction, the registered `AllReduce2_v1{H}` with its input's two
  rows staged as two ports, is
  proved whole beside a sampled rest, and Lean verifies it with the verifier's own registered reads of both ranks' partials of
  that request at the roots each owner registered, every docket read before any rank's coin. A received row moved one
  bit (`--altered`), two tokens' rows swapped, another collective's partial, another request's, and the receiving rank's own
  partial read as its peer's (`--made-for`) are each refused by Lean's registered reads on rank 1, rank 0 accepted.
- `member_seconds.py` (`a5_tp.sh` with `A5_MEMBER_SECONDS=1`): an audit's seconds a member outside its prover's clock and
  in Lean's verify before its first check, split by what is read. The audit runs on stamping wrappers of its three
  programs (the prover, Lean's verifier and the ZK server's Lean loop); each member's seconds are its own processes'
  starts and ends and its logs' lines, and each reader of its files is then run alone (the verifier's and the prover's
  load, the ZK server's Lean loop's start with and without the window's partition and program, `archive-put` of each
  file). Behind the ZK server (`ZK_SERVER=1`) one kept process of Lean's verifier (`verify-members`) verifies an audit's
  members in turn (`a0.VerifyMembers`), and a member's seconds there are from its line to its answer. Among the readers
  run alone is `verify` on each member's session, a process of its own: it must print the audit's verdict line for that
  member, byte for byte, or the measurement fails, and its seconds are beside the audit's in one record.
- `cold_sessions.sh` (`cold_sessions.py`): one cold session a member of a kept audit window behind the ZK server, on one
  card (`prove --zk --gpu`, a lease a session, after a check that no other run's lease or compute process is on the card)
  then on the CPU prover (every member on one arm before the next, so the card is held only while the gpu arm runs, until
  its own end hour `COLD_END_gpu`), each on a proof server of its own and the member's own sample from the audit, then
  Lean's verdict on it beside the next session. Each session's seconds by part (each side's load, the Lean loop's start, the
  clock's witness, proving and wait, and the ZK server's rest of it), its host and card peak memory, or its refusal's
  text. `COLD_STAGE=probe` checks the window's files and the tools, `build` builds alone and `run` builds and proves.
  `a5.py --gpus` hands an audit's proof server the same devices.
- `a5_glm53_commit.sh`, `a5_glm53_cpu.sh`: A5 on GLM-5.3 in its two halves. The first is the GPU job, from the source tree
  (`bash benchmarks/one_stage/a5_glm53_commit.sh`): the TP8 Commit with served windows, their
  private bodies kept on the GPU machine off custody. The second, `a5_glm53_cpu.sh GPU_RUN` from a VM, copies them to the CPU
  machine (`A5_ON`) and runs `a5_tp.sh` there in one research run.
- `adapters/vllm/tools/served_run/commit.sh` (the vLLM served run's): the single-GPU counterpart of `a5_glm53_commit.sh`'s deferred Commit, on Llama-3.2-1B bf16
  bi-eager TP1 (one RTX PRO 6000 of vy-nebius-1): the workload's Build and Match of record, a config-run Commit that seals
  `commit/replay_bundle_p0/` (a single-GPU workload defers only as a config run) and its replay on CPU. The row dir stays on the
  node, the only copy of its bundle. A deferred single-GPU Commit's window comes from its bundle through `verity-vllm serving-rows`, not `tp-serving-rows`.
- `served_verdicts.sh` (with `served_verdicts.py`): one served window through every subprotocol, in one job on an H200 node
  from a clone of the tree (`bash benchmarks/one_stage/served_verdicts.sh`). It serves
  Llama-3.2-1B int8 on one H200 under `gpu-lease` (`serve-openai --scheme rows-sha512 --seal --admit release-order`), sends a
  few greedy requests from the certifier's traffic generator, and seals each window. Before the first request, the committer
  registers its weights under its own key (`prove_path.py weights-registration`). Without a docket, `NO WEIGHTS
  REGISTRATION` is printed and every weight read counts free. On the window holding the most requests, none carried, it runs
  SPOT: the window's advice under that docket (`--weights-of-record`) and its request Programs, V3b's window
  circuit at each class of `CLASSES` (V2's 4,32,4 by default), with its weight reads counted registered at the docket's
  pins. The run needs every one of those counts; the class of record is counted only where `CLASSES` names `record`.
  Then comes the records audit (`benchmarks/prove_path/run.sh`, `PP_RECORDS=1`), which is Flock `--zk` on the sampled proof blocks
  and the Lean verifier. Then come PoUW, PoUS, the bandwidth audit and the network attester. Each runs its piece where
  `POUW_CMD`, `POUS_CMD`, `BANDWIDTH_CMD` or `CERTIFIER_CMD` names one, within its own limit; otherwise its `MISSING` line
  names the command it lacks and its owner, and it gets no verdict. The script sets each owner's command by default where
  the tree has its piece and the node holds what the piece reads (its auditor key, the bandwidth audit's registered store).
  The bandwidth audit's session runs inside PoUS's piece, over the store that piece holds
  (`benchmarks/pous/served_piece.py`), and `BANDWIDTH_CMD` collects its answer (`served_bandwidth.py collect`).
  Finally the all-protocols window report and the block record, which accepts only when every subprotocol accepted on this
  node and names each one missing, rejecting or about another server, with its reason (PoUW's development piece: development
  run, credits nothing). `PRODUCT=remote` composes SPOT and PoUW alone: the onsite pieces are dropped, not run
  and not composed, and the preflight refuses any onsite piece's command. An accept's clause says when its weights were held and their share of the checkpoint, where its
  profile says so (PoUS's), each δ of the block record's at which its quantity is not established, and the bandwidth audit's error with the term it leaves out beside it (the block record's `named_terms`). V3c runs last, within `V3C_TIMEOUT` and `V3C_AUDIT_TIMEOUT`, since nothing before it reads it, and writes its audit as the network
  attester's period reads it (`period.json`). With `V3C_WARM_JOBS` its warm step (`records_audit.py warm`) composes the
  window's templates before the audit's coins, into the window's own directory or, with `V3C_COMPOSE_CACHE`, into the
  directory that names: a kept compose cache, from which a later window under the same tree takes the circuits it
  shares and composes only the rest. No step removes a kept cache, and it holds every template's circuit composed into
  it, one of which can be past 100 GB.
  With the certifier's piece split in two (`CERTIFIER_CHAIN_CMD` and
  `CERTIFIER_PERIOD_CMD`), V3c runs before the pieces instead: the chain after its window commit, beside its audit
  (within `CERTIFIER_TIMEOUT`, default 4h), and the period check on its `period.json` once both are done, whose answer
  the certifier's piece reads. Each
  step's verdict line is in `out/verdicts.txt`. The network attester's piece, the longest, starts as soon as the window is
  picked. No line counts a window's requests: the serve dir's request lists, serve-openai's log and the load's records,
  timings and log stay on the node, and leave it only as `served_overhead.py metrics`' aggregate, less the load's counts,
  tokens and span, which `publish` holds in the serve dir's private/out. It holds there too what Pearl+'s documents say
  of its exclusion tiles, their count and every replayed tile's id, which the retained values decide, and the exact
  credit of a run held to its class's fixed load (`exact_work`): such a run publishes its work in the grid form, the
  class's structural credit less the give-up padded up to H1's window grid (`verify_run.grid_work`). And the give-up
  before the coin itself: the piece's give-up document (`development.json`'s and its line's `give_up`: the A and B rows
  given up, the tiles over the inert-work bound before and left after, those replayed on the CPU) and the same counts in
  `served_passes.py volunteer`'s and `retain`'s lines in the piece's log.
- `a6.sh`: run A6's entry point on a CPU machine, on hidden_zk's cached builds. A6 is the W8A8 coordinates of an int8
  decoder layer (`ScaledMmS8Coordinate_v1` at each K, Llama-3.2-1B's by default) under the stratified sampling design, stand-in values
  staged by Flock's `UnitStager` bound to the partition in statements the spec's shape admits (m <= 35), each stratum's
  share proved under `--zk` across its template's statements and held by the Lean verifier with the proof blocks it derives itself.
  `--gpu` proves each session with `prove --gpu` under `gpu-lease 1`; `--plain` proves every proof block with Flock's plain protocol,
  each statement a window of its own (the Lean verifier's plain scope: one table, sampled whole by its stratified sample).
- `layers.sh ROW`: the proof system by layer on A6's stand-in values (never served) and one H200, one benchmark result a run: `flock`
  (every proof block, plain), `zk` (every proof block, `--zk`, one docket), `sampled` (the one-stage audit, `LAYERS_K` proof blocks) or `all`,
  each on a CUDA build of `flock-circuit`, its scratch in `/dev/shm`. `layers.txt` holds its `ran with:` line, coins
  and parameters: a development challenger local to each session, never an auditor's device, so a measurement and not
  an audit. The served table's benchmark results are `benchmarks/prove_path/run.sh`'s and `a5_llama1b_h200.sh`'s (`verity bench prove
  layers`). Its rules are the three parts at the end of this file.
- `proof_layers.sh`: the served table's first three benchmark results from a clone, as one command, from the bootstrap to
  the printer's table, removing its work directory at the end ("The three benchmarks of the proof-layers table").
- `composite.py`: onsite's or remote's case from its subprotocols' cases, and the Lean verdict on it. Each `--cases DIR`
  is a private directory (0700, files 0600) holding one subprotocol's `case.json` (its `serve` line, with `blobs` the
  paths of the files it reads) and its own Lean answer on it, `answer.json`, whose `commitment` is its digest (or a
  directory of such directories); for onsite, the inference-only policy's case is the docket. The composed params, commitment and
  evidence (the composition's formats, `verity/protocols/compose/verifier/lean/README.md`) and every answer go to
  `--private-out` (0700, files 0600), since a case's evidence and params are witness, and so are an answer's query,
  digests and profile. `--public-out` gets only the verdict lines (each piece's and Lean's id, verdict and refusal's
  name), the composite's commitment, and each case's params digest and commitment, and is refused when it shares a
  byte run with any case's evidence. With `--exe`, it runs `onsite check` or `remote check` and exits with
  Lean's code. It decides nothing: a subprotocol with no case is left out, Lean's verdict on what remains stands (a
  rejection naming the missing case), and the run is labelled `NOT_ESTABLISHED` with what is missing.
- `pouw_piece.py`: PoUW's `--cases` directory for `composite.py`, remote's `verity/pouw/v1` (`v1`: the verifier's params,
  the window's one-stage docket and the proof system's answer to it) or onsite's `verity/pouw/period/v1`
  (`period`: the key, main's first-parent history, the period's name, the auditor's ledger, the pins log and each
  window's evidence), and its answer, `pouw-check`'s or `pouw-period`'s with `--exe`, written private as `composite.py`
  reads it. From a served run (`--serve-run DIR --window K`) it names the piece MISSING with why and writes no case, so
  `composite.py` leaves it out: an H1 window has no `v1` case, and no `period` case outside a period or without its sampled
  proof blocks' proof outcomes.
- `v2_case.py`: SPOT v2's `--cases` directory from a served window's kept files (`case_v2` of its v1 case,
  partition, public file, sessions and circuit file), with its own Lean answer with `--exe`. A circuit file the run didn't
  keep is rebuilt by the command its `m0/compose.log` records and used only at the log's pin. The case names the public
  file by the digest the audit staged, where the window keeps it (`case-v2.json`), and refuses a kept file with another,
  leaving only `missing.json`; a window keeping none gets the kept file's own digest, so its case shows only that the
  kept files verify. `case.json` says which under `files`, and whether the audit staged the file before the sample or
  after one its caller handed it (the window's `summary.json`), or that the window cannot tell.
- `kept_composite.sh SRC [K]`: remote's and onsite's Lean verdicts on a served run's kept records (`served_verdicts.sh`'s
  `out/`, or a tarball of it) on a CPU node: it builds the products and the pieces' checks, writes PoUW's pieces
  (`pouw_piece.py`; the period from the kept `pieces/pouw/`, `main.txt`, `H1_KEY_SHA512` and the sampled and live runs'
  statements, `POUW_STATEMENTS` and `POUW_LIVE`), v2's (`v2_case.py`) and every kept piece holding a case beside its
  answer, all private in scratch, and runs `composite.py remote` and `onsite`; `$OUT` gets only their public output.
- `zk_cost.sh`: Flock's zero-knowledge session cost on sm_120, for the sampling planner's cost model (`zk_cost.py`, a GPU
  host job, or a preemptible guest job on a shared node). A6's W8A8 coordinates at K = 4096 and 14336 are
  staged in statements of 1 to 512 proof blocks, and sessions of 1 to 512 sampled proof blocks are proved by `flock-circuit prove --zk
  --gpu` in three arms: `dev`, as SPOT[Flock]'s proof server proves (no ZK server, the witness built on the device), and the
  ZK server's send check on (`fw`) and off (`nofw`) over the host's witness. The Lean verifier checks them, and the job fits the per-session,
  per-statement and per-proof-block costs with their standard errors, writing `costfit` observations. `ZK_SAMPLES=c` proves every
  staged proof block: the cost of c sampled proof blocks in one statement, up to a table's 2^(35 − k_log) (2048 at K = 4096). Each point
  takes its own GPU lease, and a node without CUDA 13.3 gets its packages unpacked under `FLOCK_WORK`.
- `server_rate.py`: SPOT's proof server's steady rate on one GPU, run inside a one-GPU `gpu-lease` with the GPU build of
  `flock-circuit`. Each staged type (`served_units/stage.py TYPE full`, every proof block sampled) is proved by
  `ProofServer(provers=K)` at each K of `--provers` (a templated one, a table read, with its proof block placed by
  `--tmpl-unit-bin`'s `flock-tmpl-unit`, and verified under `verity/flock-circuit/tmpl`), K streaming provers on the one device, each with a K-th of the job's
  threads (the proof server splits them). Every prover finishes its
  warm sessions before any timed one starts. Each K's benchmark result (`rate.json`) gives the statements a second over the steady
  window, timed by each session's filing by its verifier, since a prover reports a session only once it collects its
  verdict, `verify_ahead` sessions later. It also gives the device's busy share (nvidia-smi's utilization over that window) and memory
  peak, each prover's and verifier's peak RSS, the provers' CPU seconds a statement, the median phase times (the front is
  `e2e_s − prove_total_s`), the load, and the throttling of the cgroup whose `cpu.max` is gpu-lease's CPU quota
  (`throttled`: its counts, null where it has no `cpu.stat` that counts them, the cgroup and its `cpu.max`). A benchmark result also says what prices it: `gpus` (the devices its
  provers held, so a statement costs `gpus / statements_per_s` GPU-seconds), `stage.blocks` and `stage.full` (the proved
  statement's proof blocks, and the most a statement of its circuit holds), `verify_s` (the verifier's mean seconds a
  statement over the window), `flat`: whether each prover's statement cache had stopped filling, its hit rate over the
  last quarter of its timed sessions within 0.01 of the third's, with the quarters and the rule in `cache`; and `cores`: the
  CPUs its processes could run on (the script's own affinity, which every process of the run runs within: unplaced, each
  inherits it; with `FC_VERIFIER_CPUS`, the verifiers run on `verifier_cpus` and the provers on the rest), as their
  `count`, their `list` and the `numa_nodes` that hold them. With `--lean`, the last sessions at
  K = 1 also get the Lean verdict inside the server's verdict pool, with its CPU seconds and peak RSS. Without
  `--lean-program` a proof block's verdict is the `NO_PROGRAM` refusal. With it, the
  Program's descriptor is handed to `verify` (`--program`) and the benchmark result keeps what the verdict says of it:
  `program_sha512` only when the Program is held to the circuit, else `program_handed_sha512` and `program_note`, which
  on a statement under the provisional unit cover says the Program was handed and not held to the circuit. With
  `--wide-leaf` every prover and verifier of the run carries `flock-circuit --wide-leaf`, the wide level-0 leaf's
  statement version, and its benchmark results say so (`wide_leaf`); the script's docstring has the rest: the two statement names
  their Lean verdict is asked under, what the Lean verifier takes and refuses under them, and how a benchmark result is held to the
  version its sessions ran. A benchmark result with a refused session, a server process that exited nonzero, a prover out of device
  memory, or a session that ran another statement version than its own is void: no rate, its reasons in `void`, a
  `VOID` line, and the run exits 1.
- `zk_cost_cpu.sh`: the same sessions on a CPU node (`zk_cost.py prove --cpu`), proved by the CPU prover that
  `benchmarks/prove_path/run.sh` builds, with the ZK server's send check. It runs in a proofs slot, one point at a time. Each session's
  record carries its wall time and its summed peak memory (prover, live verifier and ZK server), which is what a CPU node
  spends on one sampled statement of a block's sample.
- `phase_profile.sh` (with `phase_profile.py`): one GPU prover phase (`PHASE`, default `t.ligerito`) on a pod of its own: the
  source's build (`AS=base` or `new`) on a served type at full (`UNITS`, default the census's decode MoE gate-up block, `census:MoeExpertFp8BlockGemm_v3{K=6144,N=1,G=128,KB=128,SWAP=false,DOT={"fn":"HopperE4m3MmaV2Dot32_v1"}}`), its transcript
  pinned to main's under Flock ZK and Flock's circuit prover, one warm `zk_cost.py` dev process under Nsight Systems with
  CUDA and OS calls traced, and an A/B of base and new.
  `phase_profile.py` cuts each rep's phase at the prover's synchronizes and splits it into GPU time by kernel family and the
  host's time with the GPU idle: its OS calls (a live coin's round trip), its CUDA calls and its own compute.
- `forged_proofs.sh`: the fixture of Flock's forged-proof negatives (`tests/verity/protocols/proofs/flock/prover/test_lean_forged_proofs.py`),
  on hidden_zk's cached builds (`forged_proofs.py`). A small A8 proof block (`LlBf16DotprodDot_v1` at K = 128) staged bound to its partition,
  with one honest sampled `--zk` session and two whose committed output word is wrong, proved with the prover's self-check
  skipped (`FC_ZK_SELF_CHECK=skip`), so the Lean verifier gets a well-formed proof of a false statement.
- `form_refusals.sh`: the fixture of `verify --zk`'s form refusals
  (`tests/verity/protocols/proofs/spot/verifier/test_form_refusals.py`), on the same builds (`form_refusals.py`): honest
  `--zk` sessions of that proof block flat, of it as its template, and of a toy whose last stage returns the outputs, with the
  flat session's verdicts under four statements `verify --zk` must refuse.
- `a11.sh`: run A11's entry point, on A6's builds. A11 is GLM-5.3 TP8 B1's served step, rank 0: `a11_kinds.py` censuses
  the request Program's root Calls by Definition kind and tries one proof block of each through the circuit prover (a kind whose word proof blocks it
  refuses, through its Boolean version's); `a11_slice.py` cuts the
  capture's values and the checkpoint's tensors for a few layers off the serving node; `a11.py` stages each lowering kind's
  proof blocks from those values, one template each, under the stratified sampling design, proved under `--zk` and held by the Lean verifier;
  `a11_table.py` renders the per-kind table and its gap lines from the audits' summaries.
- `a9.sh`: run A9's entry point, on the same builds. A9 is GLM-5.3's routed-expert FP8 GEMMs in the registry's Boolean
  static pair form (`boolean_fp8_moe`): `a9.py` holds one coordinate of each launch's first routed pair to the capture in
  its word and Boolean forms, cuts an up/gate coordinate whose rows overflow the circuit prover's block into its gather and
  block-coordinate proof blocks, and saves the proof blocks for `a11.py --load-values`, whose audit proves them. `A9_SAVED` audits a
  saved directory without the values stage.
- `a10.sh`: run A10's entry point, on the same builds. A10 is A11's kinds and A9's launches on rank 0's committed values
  in place of the capture's: `a10.py values` opens rank 0's replay bundle from the Commit (`--replay-deferred`,
  `retain=host`) against its run root of record, reads each Call's operands and returns through its Program row as the
  deferred replay does, and runs `a11.py`'s and `a9.py`'s staging on them, writing every word where the committed value
  and the capture's differ to `compare.json`; `a10.py audit` is `a11.py --load-values` on a saved directory, with the
  docket window naming the committed values.
- `m1_families.py` and `m1_unit.py`: M1's full-step coverage. `m1_families.py` reads a rank's Programs of record from a
  deferred TP Commit's Build and lists the proof-block families of the given steps: each Definition the serving rows' partition
  samples from, its rows, op paths, ports and whether `serving_rows.template` takes it; `--estimate` adds each family's
  ANDs, table reads and assertions from its leaves lowered once (table reads counted apart, so not comparable with a
  statement's `ands` with its reads inline: `m1_families.py`'s docstring), against 2^26, and
  the log2 of Flock's circuit prover's block for its proof block against `K_MAX` (the hashing of its rows alone bounds that block from below), the
  same for each of its split proof blocks, and with `--also`/`--also-file` for Definitions a split main lacks would use, with
  each proof block's served MACs and the bits of each of its ports; `--steps all` takes every request step.
- `block_census.py` (run by `block_census.sh ROW_DIR` on the machine holding the row's Build): the proof-block census of
  a TP Commit's Build. Every rank's `m1_families.py --steps all --estimate`, each Program row cut into proof blocks (the
  row itself when it fits Flock's block and a size class and is no longer than its family's `WHOLE_UP_TO`, else each slot
  of its split at its coarsest fitting proof block, else
  a gap with the reason; a `COLS` row or slot, and the block-scaled FP8 GEMM's coordinates (`WIDE_COORDS`), at its fitting
  width of least leakage-weighted proving cost, `width_pick`),
  and per proof-block Definition its count in the prefill step and the decode steps, output bits
  and size class, estimated ANDs, served MACs and opened row bytes, with the totals per token and per request, the gaps
  and the rows that are no proof block (`census.json`, `table.md`), and the planner's card (`card.json`,
  `verity/protocols/proofs/spot/prover/cost/glm53.py`). `block_recut.sh ART` makes a stored census again from its families (its
  `census-rank<r>.json`) with this tree's proof-block templates, sizing and `block_census.py`, for a change to the cut or to a
  family's proof blocks that needs no Commit Build, and writes `compare.json` against the planner's card.
  `m1_unit.py` builds one family's proof block through Flock's circuit prover's `UnitStager`, flat or `--structural`, and with `--bin`
  and `--lean` proves one round of it on stand-in bf16 values under `--zk` and verifies it with the Lean verifier.
  `m1_forms.sh` runs it on one Definition per form, side by side, with the Lean call's seconds and peak memory beside each
  circuit file's bytes: by default an attention head's first key block and its end as their word Definitions, whose table
  reads are inline, and as their Boolean versions, flat and structural.
- `served_stock.sh` (with `served_overhead.py`): `served_verdicts.sh`'s stock control, the same endpoint, engine and load
  served with `serve-openai --no-commit` and no protocol, once per fresh endpoint for `REPS` endpoints. `served_overhead.py
  table` compares the composed run with it on two clocks, each the same code in both runs: the client's
  (`serve_load.py`'s timings files: `--varied`'s, or with `FIXED` set `--fixed`'s, the composed run's `LOAD=fixed`) and the
  engine's step trace (`advice_trace.jsonl`). The trace and the load's
  records stay in the serve dir on the node, where `served_overhead.py metrics` reads them. Its aggregate (`overhead.json`)
  holds the load's request and token counts, so a stock run's stays there too, and the composed run publishes its own less
  its load's counts, tokens and span. `table` reads the full ones on the node, the stock runs' and the composed run's
  private/out/serve/overhead.json, and `table --public` publishes ratios alone. It reports time per output
  token, time to first token and tokens per second as composed over stock, the composed run's window open and close on
  the engine loop, and `verity-bench served-table`'s single-protocol rows beside them.
- `served_composite.sh`: `served_verdicts.sh`'s composite, the step before the final record, whose verdict is its
  product's compiled Lean answer when the composite names the record's window (`served_verdicts.py record
  --composite`): on the served node after V3c (`COMPOSE_OFF_NODE=0`, the default), in the compose follow-on off it
  (`COMPOSE_OFF_NODE=1`), or in a deferral's follow-on on a prover node. It
  names PoUS's and the wipe's composite cases as their pieces wrote them (PoUS's `onsite_pieces.sh`), then gets remote's
  and onsite's Lean verdicts on the run's cases (soundness's composite reader). Only the verdict lines reach the run's
  `verdicts.txt`.
- `onsite_periods.py`: the onsite record (`verity/onsite-network-record/v1`) per monitoring period, a week, a month and a
  year by the node, a datacenter and the world, each cell its K, share of the weights and δ_net beside K over one
  execution phase, marked proved or scaled by the record's rules. It refuses a record whose labels its rules contradict:
  K/m proved past one phase, a node cell proved over a period the run doesn't cover, a datacenter or world cell proved,
  a proved statement outside the lock, a rule without its sources. The tight K is shown conditional on what is open.
  Its test reads stand-ins cut from the writer's own record (`tests/benchmarks/one_stage/fixtures/onsite_periods/make.py`).
- `served_util.sh` (with `served_util.py`): the whole-step tensor utilization of serve-openai on the int8 Llama-3.2-1B
  prefill mix (prompts of 1,024 ids, `max_tokens` 1, all queued at once, T tokens a step), one arm a run (stock, the
  Commit unsealed or sealed, the Commit with PoUW H1 on in-process coins), under Nsight Systems' GPU metrics.
  `served_util.py` takes Tensor Active's mean over the longest run of full steps, each step from the previous step's end
  to its own on the step trace's clock, and splits the rest into losses by what the GPU was running at each sample.
  With `PYSPY=1`, `pyspy_steps.py` splits the same steps' host time by Python function and thread, from py-spy's trace.
- `served_shares.py SERVE_DIR [--credit ANSWER.json]`: a served run's utilization (the work its step trace's
  steps declare, the linears at θ and attention and the LM head at θ/2, over the loads' busy wall), term by term, and its
  credited share (the MACs Pearl+'s verdict, `served_passes.py verify`'s answer, credited at θ, over the same wall), at
  each reading of the tensor's clock the card states (`verity.device.readings`: the tensor clock's and the SM clock's).
- `served_pearlplus.sh SERVE_DIR WINDOW PIECE_DIR`: PoUW's piece with Pearl+ (`served_verdicts.sh`'s `POUW=pearlplus`).
  After the window's serve, it serves Llama-3.2-1B's BF16 checkpoint on the same card through `pouw:pearl-c-h100-v1-h3`
  (`benchmarks/pouw/pearl_c_vllm/served_passes.py`, a prefill-heavy load), retains every pass, and while Pearl+'s
  verdict rejects a run on a sampled row over the inert-work bound (the rule for a tile over it, `reject`, which `served_passes.py
  volunteer-needed` reports before anything is served) gives up the passes' rows over the inert-work bound before the verifier's
  coin (`served_passes.py volunteer --device`: every word's counts from the card's inert-work cubin, every
  `volunteer_pass.CROSS_CHECK`-th tile replayed on the CPU too; `PEARLPLUS_VOLUNTEER=cpu` replays every tile on the CPU,
  far too slow for the default load), refusing to serve with the give-up turned off then; under `uncredited`, where the
  verdict credits such a row nothing, the give-up is off unless `PEARLPLUS_VOLUNTEER` is set. It takes Pearl+'s verdict
  over all of them (`served_passes.py verify`, `--over-cap` from `PEARLPLUS_OVER_CAP`; its coin, where the lane's gate is
  set, a sample session's of its own there, reserved as the lane's first reserve after its Pearl-C period closed, right
  before the sample, and opened on the run's commitment, `bench_arm.LiveDraw`), and writes the piece through
  `served_verdicts.py pearlplus`: the verdict, the linears' declared and credited shares
  (`served_shares.pearlplus_shares`), and why it credits nothing.
- `served_compare.py`: the served compare, `served_verdicts.sh`'s step under `PEARLPLUS_TIE` once Pearl+'s piece wrote W
  and before the record, on the CPU. It holds the serve to the workload of record: serve.json's φ is the row's declared φ,
  taken at the run's `REV` (`--revision`; another revision the manifests pin of the row's model, its PHI line and last
  line naming both), and lowers the same window circuit (PHI), predict row's Build is the row's Build of record, `BUILD_OF_RECORD` (BUILD), and
  W registers exactly that circuit's Pearl+ values (CUT). Its last line, `SERVED-COMPARE: EQUAL` or `NOT EQUAL`, is the
  one the record names (`served_verdicts.py record --served-compare`); `served_verdicts.sh served-compare K` reruns it.
- `rehearsal.sh check|serve|audit|compose|probe`: the remote demo's served path as one command a node, each verb
  `served_verdicts.sh` at fixed settings (Pearl+ tied to the window, product remote) with its output through a filter
  that withholds a session secret. `check` runs on any CPU node what the other verbs read that needs no card and no
  gate: the settings, the class's circuit from the serve's φ, a stand-in window of the class through what a served
  window's close reads of it (`rehearsal_window.py`), the harness's policy step and its limits, the policy's Lean check
  on the class's own Program with the settings' `ALPHA`, `GAMMA` and `DAY_PROGRAM` held to what it answers and
  `HELD_ALPHA` and `HELD_GAMMA` to the record form's pair of the class (`rehearsal_limits.py`), the day's declaration,
  the Lean the verbs run, and Pearl+ against this tree's own gate code.

Flock's circuit prover (`verity_flock.circuit`) is on `main`, but a run does not take it from its checkout:
`pod.sh` takes `git archive`s of the circuit prover's tree and the Lean verifier's tree at pinned commits, and a captured input set, as
inputs sent with the run; it builds `flock-circuit` and `verity-verify-proof` once per pod, then runs the driver, whose
`--m0-pythonpath` names that tree. The drivers' checks go through the verifier
(`one_stage.verifier`), which builds the SPOT Lean in the run's tree on its first check, hence `--cwd clone`:

~~~sh
bash benchmarks/one_stage/pod.sh    # a job run from a clone of the tree, with m0.tar, lean.tar and set.tar sent
~~~

`check` runs nothing here.

## The three benchmarks of the proof-layers table

The proof-layers table's first three benchmarks are three runs of `a5_llama1b_h200.sh` on one work directory: Flock (a
plain audit), Flock with zero knowledge, and with sampling. They differ in their settings and in their partition file's
sampling design (its `law` key): `stratified-rate:880/880` and `subset:880` take all 880 proof blocks of the window, and
`subset:256` samples 256 of them. From a clone of `main` on an H200 node with at least 96 GB of host memory available
and 25 GB of disk free under the work directory, at the tree's root:

~~~sh
A5_CARD=0 A5_CORES=40-59 WORK=/workspace/a5-layers GPU_LEASE_WHO=$USER bash benchmarks/one_stage/proof_layers.sh
~~~

`A5_CARD` is the jobs' card, `A5_CORES` their cores, `WORK` a new or empty work directory and `GPU_LEASE_WHO` who their
leases are recorded for; `--out DIR` names where the results go (default `./out`, or the run's `out/` in a
`research run`), and `--dry-run` makes every check and prints each step's command without running it. What it does:
before anything is downloaded or built, it refuses a missing setting, a card that is not an H200, under 96 GB of host
memory available, cores this job may not run on, under 25 GB free under `WORK`, a `WORK` that is not empty and results
already in the results directory. Then it runs the eight steps below, stopping at the first that fails: the bootstrap,
the three partition files, the three runs and the printer, whose table it keeps as `proof-layers.txt`. Its `steps.log`
records each step's start, command, exit code and seconds; the peaks of the job's own memory (PSS over its session), of
the machine's memory in use (other jobs' included) and of `WORK`'s size; and the largest process. At its end, success or
failure, it copies the three runs' records and the bootstrap's to the results directory and removes `WORK`. It does not
run the recursion benchmark. By hand, the same steps (the script's bootstrap writes its record under `WORK`):

~~~sh
export WORK=/workspace/a5-layers A5_CARD=0 A5_CORES=40-59 GPU_LEASE_WHO=$USER
(cd adapters/vllm && taskset -c $A5_CORES bash verity_vllm/ops/pod_bootstrap.sh --cases LLAMA32_1B_INT8)
for b in flock=stratified-rate:880/880 zk=subset:880 sampling=subset:256; do python3 -c 'import json, sys; p = json.load(open(sys.argv[1])); p["law"] = sys.argv[2]; json.dump(p, open(sys.argv[3], "w"))' benchmarks/one_stage/llama32_1b_rope_d64_l0.json "${b#*=}" "$WORK/partition-${b%%=*}.json"; done
A5_COINS=challenger A5_PLAIN=1 A5_PARTITION=$WORK/partition-flock.json OUT=$WORK/out-flock taskset -c $A5_CORES bash benchmarks/one_stage/a5_llama1b_h200.sh
A5_COINS=challenger A5_PARTITION=$WORK/partition-zk.json OUT=$WORK/out-zk taskset -c $A5_CORES bash benchmarks/one_stage/a5_llama1b_h200.sh
A5_COINS=challenger A5_PARTITION=$WORK/partition-sampling.json OUT=$WORK/out-sampling taskset -c $A5_CORES bash benchmarks/one_stage/a5_llama1b_h200.sh
uv run verity bench prove layers --flock $WORK/out-flock --zk $WORK/out-zk --sampling $WORK/out-sampling
cp -r $WORK/out-flock $WORK/out-zk $WORK/out-sampling . && rm -rf "$WORK"
~~~

`taskset -c` gives each job its cores, and its builds, replay and audit run on them (`nproc`). On a node shared through
`gpu-lease`, the bootstrap runs under a lease of the same card and cores instead, with
`gpu-lease 1 --wait --on $A5_CARD --cores $A5_CORES --` in place of `taskset -c $A5_CORES`. The script leases `A5_CARD`
for its Match capture and its Commit, and every lease it takes names `A5_CORES`, without which it refuses before any
build; with no `gpu-lease` it takes that card directly. `gpu-lease status` shows each lease under `GPU_LEASE_WHO`;
unset, the script records its leases as `vllm-sha512`. Unless `CUDA_HOME` is set, the script takes the first of
`/usr/local/cuda-12.9`, `/usr/local/cuda-12`, `/usr/local/cuda-13.0` and `/usr/local/cuda` that has `nvcc`. Where that
toolkit lacks the CUDA math libraries' headers, it takes them from the venv's NVIDIA wheels, as the bootstrap's norm and
router tap builds do. Each run's record is its `OUT`, which the printer reads as a directory, as it reads a
`research run`'s id. The three `OUT` directories are inside `WORK`, so the last line copies them to the current
directory before it removes `WORK` (the venv, the checkpoint, the taps and the Flock build).

Development-only: the coins come from challengers run on the node, stand-ins for the auditor's device, so the printer
calls the table a measurement, not an audit, and says its parameters are not stated.

## What you may change

For `layers.sh`'s benchmark results: the prover and how a run drives it, `flock-circuit`'s build, kernels and flags, the chunking of
proof blocks into statements (`--chunk`), the lease and the parallelism of sessions.

## What is fixed

For `layers.sh`'s benchmark results: the workload (the members, their template ids and the stand-in rows' seed), the Lean verifier and
its flags, the coins (each session's own development challenger, Lean's stratified sample) and, in `sampled`, `k` and δ.

## How a cheating change is refused

For `layers.sh`'s benchmark results: by the Lean verifier. A benchmark result counts only where `verity-verify-proof verify` accepted every session
against the program and partition the driver derives itself, never the prover's own report; a6.py exits nonzero
otherwise.
