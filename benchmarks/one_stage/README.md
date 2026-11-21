# One-stage audit runs

The one-stage audit (`verity.protocols.proof.spot.verifier.one_stage`) end to end: serving's committed values through docket, the verifier's
sample, Flock's circuit prover and the Lean verifier, to the audit record with its integrity profile, then the
negatives.

Its sampled sessions are SPOT[Flock]'s on its own, in its three roles: serving commits its values (the inference server);
each sampled session's prover runs `flock-circuit prove --zk` itself (the proof server: `a0.DIRECT`, `zk_cost.py`'s `dev` arm, which
Figure 1 prices); and the challenger's live coins, its signed log and the Lean verifier decide (the verifier). There is no ZK
server and no recursion on that path. The core protocols' path, every prover behind the Lean ZK server's send check, is
`--zk-server` on SPOT's window audit (`a5.py`), its variants here, and `ZK_SERVER=1` on their shell entry points. The network
attester, where a run has one (`a5_network_attester.py`), is a layer of its own, whose signed stamp SPOT[Flock] reads as a certified
input. What the guarantee rests on: `verity/protocols/proof/spot/verifier/README.md`.

- `a<N>.py`: the driver of run A<N>; its docstring says what the run adds and how to call it. A5's is SPOT's window audit,
  here because it loads the adapter, Flock's prover and the benchmarks' scripts
  (`verity/protocols/proof/spot/prover/audit/README.md`). `a0.py` drives no run: it holds the steps A5 and the other runs
  share.
- `stage_subset.py`: a helper that runs inside the circuit prover's tree (its `verity_flock`), as `compose_circuit.py` does.
- `a5_tp.sh` (`STANDIN=ENTRY ...`, required): A5 over the stand-in windows below, one entry a stand-in script, on a CPU machine.
  Each session's `circuit.txt` files and lookup tables go to the store by SHA-512 (`sha512/<hex>`, the Lean verifier's `--archive`
  layout) through `adapters/vllm/verity_vllm/ops/store_trees.sh`, epoch_store.sh's put, and the run is labelled `circuits` with each
  tree's art id, so the run can be verified again from the store.
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
  A5 on a window with transfer gates, which no served window has yet: two chunks of network attestation's toy Definition whose ports are the
  bucket rows of one connection's ingress and egress cells (a row a bucket, docket v2), committed under the network attester's
  salts, proved whole (the work sampling design's selection All) beside a sampled rest, on a local challenger's coins or the verifier's
  own. The proof blocks' wiring is the Program's (`network_attester_rows.Units`, from public structure alone: ingress cells 0 … n − 1, egress
  cells D·r … D·r + n − 1). Lean verifies each chunk with the verifier's own registered reads at the root the network attester signed,
  named by the proof blocks' value and each proof block's read (connection, cell) placed by the wiring the network attester signed
  (`network_attester_rows.port`, after `check_registration` under the network attester's key), which refuses a foreign, a duplicate and a
  padding-slot read by name; the altered run, and a statement made for another wiring of the proof blocks (a pad in a data cell's place,
  its rows under the Program's value, two proof blocks swapped: `--made-for`), are refused by Lean's registered reads. Its attester run
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
- `served_verdicts.sh` (with `served_verdicts.py`): one served window through every subprotocol, in one job on an H200 node
  from a clone of the tree (`bash benchmarks/one_stage/served_verdicts.sh`). It serves
  Llama-3.2-1B int8 on one H200 under `gpu-lease` (`serve-openai --scheme rows-sha512 --seal`, the network attester scheme's order `release-order`), sends a
  few greedy requests from the network attester's traffic generator, and seals each window. Before the first request, the committer
  registers its weights under its own key (`prove_path.py weights-registration`). Without a docket, `NO WEIGHTS
  REGISTRATION` is printed and every weight read counts free. On the window holding the most requests, none carried, it runs
  SPOT: the window's advice under that docket (`--weights-of-record`) and its request Programs, V3b's window
  circuit at each class of `CLASSES` (V2's 4,32,4 by default), with its weight reads counted registered at the docket's
  pins. The run needs every one of those counts; the class of record is counted only where `CLASSES` names `record`.
  Then comes the records audit (`benchmarks/prove_path/records_audit.py audit`), which is Flock `--zk` on the sampled proof blocks
  and the Lean verifier. Then come PoUW, PoUS, the bandwidth audit and the network attester. Each runs its piece where
  `POUW_CMD`, `POUS_CMD`, `BANDWIDTH_CMD` or `CERTIFIER_CMD` names one, within its own limit; otherwise its `MISSING` line
  names the command it lacks and its owner, and it gets no verdict. The script sets each owner's command by default where
  the tree has its piece and the node holds what the piece reads (its auditor key, the bandwidth audit's registered store).
  The bandwidth audit's session runs inside PoUS's piece, over the store that piece holds
  (`benchmarks/pous/served_piece.py`), and `BANDWIDTH_CMD` collects its answer (`served_piece.py collect-bandwidth`).
  Finally the all-protocols window report and the block record, which accepts only when every subprotocol accepted on this
  node and names each one missing, rejecting or about another server, with its reason (PoUW's development piece: development
  run, credits nothing). `POUW` names PoUW's scheme and the one left is `pearlplus`: `POUW=h1`, an unset `POUW` and H1's settings are refused by name. `PRODUCT=remote` composes SPOT and PoUW alone: the onsite pieces are dropped, not run
  and not composed, and the preflight refuses any onsite piece's command. An accept's clause says when its weights were held and their share of the checkpoint, where its
  profile says so (PoUS's), each δ of the block record's at which its quantity is not established, and the bandwidth audit's error with the term it leaves out beside it (the block record's `named_terms`). V3c runs last, within `V3C_TIMEOUT` and `V3C_AUDIT_TIMEOUT`, since nothing before it reads it, and writes its audit as the network
  attester's period reads it (`period.json`). With `V3C_WARM_JOBS` its warm step (`records_audit.py warm`) composes the
  window's templates before the audit's coins, into the window's own directory or, with `V3C_COMPOSE_CACHE`, into the
  directory that names: a kept compose cache, from which a later window under the same tree takes the circuits it
  shares and composes only the rest. No step removes a kept cache, and it holds every template's circuit composed into
  it, one of which can be past 100 GB.
  With network attestation's piece split in two (`CERTIFIER_CHAIN_CMD` and
  `CERTIFIER_PERIOD_CMD`), V3c runs before the pieces instead: the chain after its window commit, beside its audit
  (within `CERTIFIER_TIMEOUT`, default 4h), and the period check on its `period.json` once both are done, whose answer
  network attestation's piece reads. Each
  step's verdict line is in `out/verdicts.txt`. The network attester's piece, the longest, starts as soon as the window is
  picked. No line counts a window's requests: the serve dir's request lists, serve-openai's log and the load's records,
  timings and log stay on the node, and leave it only as `served_overhead.py metrics`' aggregate, less the load's counts,
  tokens and span, which `publish` holds in the serve dir's private/out. It holds there too what Pearl+'s documents say
  of its exclusion tiles, their count and every replayed tile's id, which the retained values decide, and the exact
  credit of a run held to its class's fixed load (`exact_work`): such a run publishes its work in the grid form, the
  class's structural credit less the give-up padded up to the window grid (`verify_run.grid_work`). And the give-up
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
  an audit. Its rules are the three parts at the end of this file.
- `composite.py`: onsite's or remote's case from its subprotocols' cases, and the Lean verdict on it. Each `--cases DIR`
  is a private directory (0700, files 0600) holding one subprotocol's `case.json` (its `serve` line, with `blobs` the
  paths of the files it reads) and its own Lean answer on it, `answer.json`, whose `commitment` is its digest (or a
  directory of such directories); for onsite, the inference-only policy's case is the docket. The composed params, commitment and
  evidence (the composition's formats, `verity/protocols/run/verifier/lean/README.md`) and every answer go to
  `--private-out` (0700, files 0600), since a case's evidence and params are witness, and so are an answer's query,
  digests and profile. `--public-out` gets only the verdict lines (each piece's and Lean's id, verdict and refusal's
  name), the composite's commitment, and each case's params digest and commitment, and is refused when it shares a
  byte run with any case's evidence. With `--exe`, it runs `onsite check` or `remote check` and exits with
  Lean's code. It decides nothing: a subprotocol with no case is left out, Lean's verdict on what remains stands (a
  rejection naming the missing case), and the run is labelled `NOT_ESTABLISHED` with what is missing.
- `v2_case.py`: SPOT v2's `--cases` directory from a served window's kept files (`case_v2` of its v1 case,
  partition, public file, sessions and circuit file), with its own Lean answer with `--exe`. A circuit file the run didn't
  keep is rebuilt by the command its `m0/compose.log` records and used only at the log's pin. The case names the public
  file by the digest the audit staged, where the window keeps it (`case-v2.json`), and refuses a kept file with another,
  leaving only `missing.json`; a window keeping none gets the kept file's own digest, so its case shows only that the
  kept files verify. `case.json` says which under `files`, and whether the audit staged the file before the sample or
  after one its caller handed it (the window's `summary.json`), or that the window cannot tell.
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
- `forged_proofs.sh`: the fixture of Flock's forged-proof negatives (`tests/verity/protocols/proof/flock/prover/test_lean_forged_proofs.py`),
  on hidden_zk's cached builds (`forged_proofs.py`). A small A8 proof block (`LlBf16DotprodDot_v1` at K = 128) staged bound to its partition,
  with one honest sampled `--zk` session and two whose committed output word is wrong, proved with the prover's self-check
  skipped (`FC_ZK_SELF_CHECK=skip`), so the Lean verifier gets a well-formed proof of a false statement.
- A11 is GLM-5.3 TP8 B1's served step, rank 0: `a11_kinds.py` censuses
  the request Program's root Calls by Definition kind and tries one proof block of each through the circuit prover (a kind whose word proof blocks it
  refuses, through its Boolean version's); `a11_slice.py` cuts the
  capture's values and the checkpoint's tensors for a few layers off the serving node; `a11.py` stages each lowering kind's
  proof blocks from those values, one template each, under the stratified sampling design, proved under `--zk` and held by the Lean verifier;
  `a11_table.py` renders the per-kind table and its gap lines from the audits' summaries.
- `a9.sh`: run A9's entry point, on the same builds. A9 is GLM-5.3's routed-expert FP8 GEMMs in the registry's Boolean
  static pair form (`boolean_fp8_moe`): `a9.py` holds one coordinate of each launch's first routed pair to the capture in
  its word and Boolean forms, cuts an up/gate coordinate whose rows overflow the circuit prover's block into its gather and
  block-coordinate proof blocks, and saves the proof blocks for `a11.py --load-values`, whose audit proves them. `A9_SAVED` audits a
  saved directory (an input tar made outside any run's output) without the values stage. The saved units' values are witness: a9.sh
  keeps them in the job's scratch directory only.
- `public_files.py`: the one rule for what a launcher copies under its run's output (`filter`): a served run's witness
  (`served_verdicts.witness`) and private, salt, key and saved-value names are never copied; every copy loop of
  `benchmarks/one_stage` and `benchmarks/prove_path` pipes its `find` through it (`tests/benchmarks/one_stage/test_public_files.py`).
- `m1_families.py` and `m1_unit.py`: M1's full-step coverage. `m1_families.py --estimate CENSUS` adds each proof-block family's
  ANDs, table reads and assertions from its leaves lowered once (table reads counted apart, so not comparable with a
  statement's `ands` with its reads inline: `m1_families.py`'s docstring), against 2^26, and
  the log2 of Flock's circuit prover's block for its proof block against `K_MAX` (the hashing of its rows alone bounds that block from below), the
  same for each of its split proof blocks, and with `--also`/`--also-file` for Definitions a split main lacks would use,
  each proof block's served MACs and the bits of each of its ports.
- `block_census.py`: the proof-block census of a stored census's sized families (`block_recut.sh`'s), one census a request, under `<request>/`, when the requests are of several shapes (`--request`). Each rank's `m1_families.py --estimate`, each Program row cut into proof blocks (the
  row itself when it fits Flock's block and a size class and is no longer than its family's `WHOLE_UP_TO`, else each slot
  of its split at its coarsest fitting proof block, else
  a gap with the reason; a `COLS` row or slot, and the block-scaled FP8 GEMM's coordinates (`WIDE_COORDS`), at its fitting
  width of least leakage-weighted proving cost, `width_pick`),
  and per proof-block Definition its count in the prefill step and the decode steps, output bits
  and size class, estimated ANDs, served MACs and opened row bytes, the Definition on bits its proof is (`proved_as`, and
  `bounded_by` its kind's largest block in the census where that bounds it), with the totals per token and per request, the gaps
  and the rows that are no proof block (`census.json`, `table.md`), and the planner's card (`card.json`,
  `calibration/sampling_plan/glm53.py`). `block_recut.sh ART` makes a stored census again from its families (its
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
  (the load's timings files: `serve_load.py --varied`'s, or with `DATASET` and `TIMING` set the sender's
  (`verity_vllm.inputs.sender`), the composed run's `LOAD=dataset`) and the
  engine's step trace (`advice_trace.jsonl`). The trace and the load's
  records stay in the serve dir on the node, where `served_overhead.py metrics` reads them. Its aggregate (`overhead.json`)
  holds the load's request and token counts, so a stock run's stays there too, and the composed run publishes its own less
  its load's counts, tokens and span. `table` reads the full ones on the node, the stock runs' and the composed run's
  private/out/serve/overhead.json, and `table --public` publishes ratios alone. It reports time per output
  token, time to first token and tokens per second as composed over stock, the composed run's window open and close on
  the engine loop, and `verity-bench served-table`'s single-protocol rows beside them.
- `served_composite.sh`: `served_verdicts.sh`'s composite: in the compose follow-on, on an auditor device off the served
  node, the step before the final record, whose verdict is its product's compiled Lean answer there when the
  composite names the record's window (`served_verdicts.py record --composite`); in a deferral's follow-on, on a
  prover node, and on the served node, a development check that leaves the record pending (`COMPOSE_DEVELOPMENT`). It
  names PoUS's and the wipe's composite cases as their pieces wrote them (PoUS's `onsite_pieces.sh`), then gets remote's
  and onsite's Lean verdicts on the run's cases (soundness's composite reader). Only the verdict lines reach the run's
  `verdicts.txt`.
- `onsite_periods.py`: the onsite record (`verity/onsite-network-record/v1`) per monitoring period, a week, a month and a
  year by the node, a datacenter and the world, each cell its K, share of the weights and δ_net beside K over one
  execution phase, marked proved or scaled by the record's rules. It refuses a record whose labels its rules contradict:
  K/m proved past one phase, a node cell proved over a period the run doesn't cover, a datacenter or world cell proved,
  a proved statement outside the lock, a rule without its sources. The tight K is shown conditional on what is open.
  Its test reads stand-ins cut from the writer's own record (`tests/benchmarks/one_stage/fixtures/onsite_periods/make.py`).
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
- `rehearsal.sh check|serve|audit|compose|probe`: the remote application's served path as one command a node, each verb
  `served_verdicts.sh` at fixed settings (Pearl+ tied to the window, product remote) with its output through a filter
  that withholds a session secret. `check` runs on any CPU node what the other verbs read that needs no card and no
  gate: the settings, the class's circuit from the serve's φ, a stand-in window of the class through what a served
  window's close reads of it (`rehearsal_window.py`), the harness's policy step and its limits, the policy's Lean check
  on the class's own Program with the settings' `ALPHA`, `GAMMA` and `DAY_PROGRAM` held to what it answers and
  `HELD_ALPHA` and `HELD_GAMMA` to the record form's pair of the class (`rehearsal_limits.py`), the day's declaration,
  the Lean the verbs run, and Pearl+ against this tree's own gate code.

`check` runs nothing here.

## Two levels

Two levels are refused by name (`levels.TWO_LEVELS_REFUSED`): `a5.py` under `SPOT_LEVELS=2`, the stream at registration and
the Lean verifier at the parameters.

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
