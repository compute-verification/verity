# One-stage audit runs

The one-stage audit (`verity.protocols.spot.verifier.one_stage`) end to end: serving's committed values through registration, the verifier's
sample, Flock's circuit prover and the Lean verifier, to the audit record with its integrity profile, then the
negatives.

Its sampled sessions are SPOT[Flock]'s on its own, in its three roles: serving commits its values (the inference server);
each sampled session's prover runs `flock-circuit prove --zk` itself (the proof server: `a0.DIRECT`, `zk_cost.py`'s `dev` arm, which
Figure 1 prices); and the challenger's live coins, its signed log and the Lean verifier decide (the verifier). There is no ZK
server and no recursion on that path. The core protocols' path, every prover behind the Lean ZK server's send check, is
`--firewall` on `a5.py`, its variants and `prove_bundle.py`, and `ZK_SERVER=1` on their shell entry points. The network
certifier, where a run has one (`a5_warden.py`), is a layer of its own, whose signed stamp SPOT[Flock] reads as a certified
input. What the guarantee rests on: `verity/protocols/spot/verifier/README.md`.

- `a<N>.py`: the driver of run A<N>; its docstring says what the run adds and how to call it.
- `compose_circuit.py`, `stage_subset.py`, `stage_tampered.py`: helpers that run inside the circuit prover's tree (its `verity_flock`).
- `r2_population.py`: the other population every driver's and `replay.py`'s registration-partition-mismatch negative (R2)
  registers, its first template one instance short, or one over where it has a single instance.
- `pod.sh`: the entry point on a CPU pod.
- `a5.sh`: run A5's entry point on a CPU machine. A5 is a deferred Commit's window, a stand-in linked to its run root, the
  commitment its record was committed in and its replay bundle: the CPU RoPE d64 Commit fixture, committed under rows-sha512
  (`COMMIT_SCHEME` names no other: vllm-v1 is kept only to read recorded runs), its window from `verity-vllm serving-rows`
  (with `WINDOW=served`, the in-Commit hook's), and the circuit prover and Lean built from the run's tree. With `A5_ROW_DIR` it takes a
  single-GPU deferred Commit's row dir and sealed bundle in place of the fixture, and with `WINDOW=served` the window its C2
  site committed (`A5_WINDOW`).
- `a5_llama1b_h200.sh`: A5 over a GPU-served window, in one job on an H200 node, from a clone of the tree
  (`bash benchmarks/one_stage/a5_llama1b_h200.sh`). Llama-3.2-1B int8's B4 row on one H200 under `gpu-lease`: its Build and
  Match, then a deferred config-run Commit under rows-sha512 whose `--serving-rows` window
  (`llama32_1b_rope_d64_l0.json`: layer 0's RoPE d64 heads at steps 1-7) its C2 site commits before the bundle is sealed,
  a `served` window under the 64-byte run root. Then the Commit's CPU replay, and `a5.sh` with `WINDOW=served` on the bundle
  and that window, whose private bodies stay on the node with the row dir: of the row's files the run's record holds only
  the ones `demos/vllm/published.sh` lists, each with the reason it holds nothing of a request, and no file of the row's
  `commit/` or `match/`, whose records, bundle, slim keep and request spec hold values.
- `a5_tp.sh`: A5 over a TP Commit's windows, one `served` window per rank (`a5.py --tp-commit`), taken as served only under a
  SHA-512 TP run root (`tp.run_root`'s v2, over the ranks' rows-sha512 roots; an old record's SHA-256 v1 root is refused before
  the sample): GLM-5.3's TP8 Commit with
  `SERVING_ROWS=benchmarks/one_stage/glm53_silu_mul_v2_tp8.json` (its SiluMul_v2 partition: every unit of `SiluMul_v2{I=1536}`,
  the dense layers' 861 per rank, by `subset:512`), or by default the CPU TP stand-in
  (`adapters/vllm/tests/pipeline/served_tp.py`). A member's whole population is one `verity/flock-circuit` statement, whose shape takes at most
  `2^(35 - k_log)` instances: 2,048 at I=1536 (`k_log` 24) and 8,192 at I=256 (`k_log` 22), so a rank's `SiluMul_v2{I=256}`
  (172,200 routed-expert and 21,525 shared-expert units) is not a member. Each session's `circuit.txt` files and lookup tables
  go to the store by SHA-512 (`sha512/<hex>`, the Lean verifier's `--archive` layout) through
  `adapters/vllm/verity_vllm/ops/store_trees.sh`, epoch_store.sh's put, and the run is labelled `circuits` with each tree's
  art id, so the run can be verified again from the store.
- `expert_tables_tp.sh`: the committer's per-expert tables registration of a TP model, one file a rank (`verity-vllm
  tp-register-tables --served`): each rank's expert stacks composed from the checkpoint of record, held to a capture's served
  SHA-512s and committed under the committer's key, which never leaves the node. A window's per-expert table reads count
  registered against these files (`records_audit.py audit --registered`).
- `a5_row_tables.py` (`a5_tp.sh` with `STANDIN=row-tables`): A5 on a window with row tables, which no TP window of record has
  yet: `adapters/vllm/tests/pipeline/test_row_tables.py`'s ScaledMmS8Coordinate chunks, whose `x` rows link four row
  tables over two engine steps, each chunk one table's, as rank 0 of a one-rank step on a local challenger's coins. Lean
  verifies each sampled chunk with its table's registered reads (`--registered`, `tables.registered_reads`).
- `a5_topp.py` (`a5_tp.sh` with `STANDIN=topp` or `topp:slots`): A5 on a window of top-p select rows cut into their stages,
  which no served window has yet: `adapters/vllm/tests/pipeline/test_serving_rows_stages.py`'s rows, each carried row wired
  to the unit that wrote it, as rank 0 of a one-rank step on a local challenger's coins, run as `a5_sparse_mla.py` runs its
  window. `topp` names each stage on bits, the form Lean's `verify --zk` takes; `topp:slots` names the stages as words, whose
  table reads go in lookup slots, and Lean refuses each sampled proof block's statement. The window holds the stages whose
  proof blocks are composed: a stats block from 56 lanes a slice, so by default V = 64 and split count 1's stats, probs and
  step blocks, and the select chunks only where every split count's are (`A5_TOPP`, the script's `--v`, `--splits` and
  `--rows`). `--select-only` holds the select chunks alone, composed at any V, each wired to the chunk before it, with the
  rows they read from the step blocks committed as plain input rows that no unit of the window writes.
- `a5_warden.py` (`a5_tp.sh` with `STANDIN=warden`, `warden:altered`, `warden:pad`, `warden:pad-named` or `warden:swapped`):
  A5 on a window with transfer gates, which no served window has yet: two chunks of the certifier's toy unit whose ports are the
  bucket rows of one connection's ingress and egress cells (a row a bucket, registration v2), committed under the certifier's
  salts, proved whole (the work sampling design's selection All) beside a sampled rest, on a local challenger's coins or the verifier's
  own. The units' wiring is the Program's (`warden_rows.Units`, from public structure alone: ingress cells 0 … n − 1, egress
  cells D·r … D·r + n − 1). Lean verifies each chunk with the verifier's own registered reads at the root the certifier signed,
  named by the units' value and each unit's read (connection, cell) placed by the wiring the certifier signed
  (`warden_rows.port`, after `check_registration` under the certifier's key), which refuses a foreign, a duplicate and a
  padding-slot read by name; the altered run, and a statement made for another units' wiring (a pad in a data cell's place,
  its rows under the Program's value, two units swapped: `--made-for`), are refused by Lean's registered reads. Its certifier run
  is rehearsed against a fake endpoint, or (`A5_WARDEN_RUN`) is `warden_replay.py`'s: a served window's requests through the
  network certifier, each answered with the tokens the server streamed it.
- `a5_tp_gates.py` (`a5_tp.sh` with `A5_CHALLENGER=local` and `STANDIN=tp-gates`, `tp-gates:altered`, `tp-gates:swapped`,
  `tp-gates:other-collective`, `tp-gates:other-request` or `tp-gates:own`): A5 on both ranks of a synthetic Llama TP2 step
  of two requests whose cross-rank transfers hold transfer gates, which no served TP2 window has yet. Each rank's partial of
  each reduction (layer 0's embedding, o_proj and down_proj, H = 2048) for each request is a registered value its
  registration names; each rank's chunk of a request's reduction, the registered `AllReduce2_v1{H}` with its input's two
  rows staged as two ports, is
  proved whole beside a sampled rest, and Lean verifies it with the verifier's own registered reads of both ranks' partials of
  that request at the roots each owner registered, every registration read before any rank's coin. A received row moved one
  bit (`--altered`), two tokens' rows swapped, another collective's partial, another request's, and the receiving rank's own
  partial read as its peer's (`--made-for`) are each refused by Lean's registered reads on rank 1, rank 0 accepted.
- `a5_glm53_commit.sh`, `a5_glm53_cpu.sh`: A5 on GLM-5.3 in its two halves. The first is the GPU job, from the source tree
  (`bash benchmarks/one_stage/a5_glm53_commit.sh`): the TP8 Commit with served windows, their
  private bodies kept on the GPU machine off custody. The second, `a5_glm53_cpu.sh GPU_RUN` from a VM, copies them to the CPU
  machine (`A5_ON`) and runs `a5_tp.sh` there in one research run.
- `demos/vllm/commit.sh` (the vLLM demo's): the single-GPU counterpart of `a5_glm53_commit.sh`'s deferred Commit, on Llama-3.2-1B bf16
  bi-eager TP1 (one RTX PRO 6000 of vy-nebius-1): the workload's Build and Match of record, a config-run Commit that seals
  `commit/replay_bundle_p0/` (a single-GPU workload defers only as a config run) and its replay on CPU. The row dir stays on the
  node, the only copy of its bundle. A deferred single-GPU Commit's window comes from its bundle through `verity-vllm serving-rows`, not `tp-serving-rows`.
- `served_verdicts.sh` (with `served_verdicts.py`): one served window through every subprotocol, in one job on an H200 node
  from a clone of the tree (`bash benchmarks/one_stage/served_verdicts.sh`). It serves
  Llama-3.2-1B int8 on one H200 under `gpu-lease` (`serve-openai --scheme rows-sha512 --seal --admit release-order`), sends a
  few greedy requests from the certifier's traffic generator, and seals each window. Before the first request, the committer
  registers its weights under its own key (`prove_path.py weights-registration`). Without a registration, `NO WEIGHTS
  REGISTRATION` is printed and every weight read counts free. On the window holding the most requests, none carried, it runs
  sampled proofs: the window's advice under that registration (`--weights-of-record`) and its request Programs, V3b's window
  circuit at each class of `CLASSES` (V2's 4,32,4 by default), with its weight reads counted registered at the registration's
  pins. The run needs every one of those counts; the class of record is counted only where `CLASSES` names `record`.
  Then comes the records audit (`prove_path/run.sh`, `PP_RECORDS=1`), which is Flock `--zk` on the sampled proof blocks
  and the Lean verifier. Then come PoUW, PoUS, the bandwidth audit and the network certifier. Each runs its piece where
  `POUW_CMD`, `POUS_CMD`, `BANDWIDTH_CMD` or `CERTIFIER_CMD` names one, within its own limit; otherwise its `MISSING` line
  names the command it lacks and its owner, and it gets no verdict. The script sets each owner's command by default where
  the tree has its piece and the node holds what the piece reads (its auditor key, the bandwidth audit's registered store).
  The bandwidth audit's session runs inside PoUS's piece, over the store that piece holds
  (`benchmarks/pous/served_piece.py`), and `BANDWIDTH_CMD` collects its answer (`served_bandwidth.py collect`).
  Finally the all-protocols window report and the block record, which accepts only when every subprotocol accepted on this
  node and names each one missing, rejecting or about another server, with its reason (PoUW's development piece: development
  run, credits nothing). `PRODUCT=remote` composes sampled proofs and PoUW alone: the onsite pieces are dropped, not run
  and not composed, and the preflight refuses any onsite piece's command. An accept's clause says when its weights were held and their share of the checkpoint, where its
  profile says so (PoUS's), and each δ of the block record's at which its quantity is not established. V3c runs last, within `V3C_TIMEOUT` and `V3C_AUDIT_TIMEOUT`, since nothing before it reads it, and writes its audit as the network
  certifier's period reads it (`period.json`). With `V3C_WARM_JOBS` its warm step (`records_audit.py warm`) composes the
  window's templates before the audit's coins, into the window's own directory or, with `V3C_COMPOSE_CACHE`, into the
  directory that names: a kept compose cache, from which a later window under the same tree takes the circuits it
  shares and composes only the rest. No step removes a kept cache, and it holds every template's circuit composed into
  it, one of which can be past 100 GB.
  With the certifier's piece split in two (`CERTIFIER_CHAIN_CMD` and
  `CERTIFIER_PERIOD_CMD`), V3c runs before the pieces instead: the chain after its window commit, beside its audit
  (within `CERTIFIER_TIMEOUT`, default 4h), and the period check on its `period.json` once both are done, whose answer
  the certifier's piece reads. Each
  step's verdict line is in `out/verdicts.txt`. The network certifier's piece, the longest, starts as soon as the window is
  picked. No line counts a window's requests: the serve dir's request lists, serve-openai's log and the load's records,
  timings and log stay on the node, and leave it only as `served_overhead.py metrics`' aggregate, less the load's counts,
  tokens and span, which `publish` holds in the serve dir's private/out.
- `a6.sh`: run A6's entry point on a CPU machine, on hidden_zk's cached builds. A6 is the W8A8 coordinates of an int8
  decoder layer (`ScaledMmS8Coordinate_v1` at each K, Llama-3.2-1B's by default) under the stratified sampling design, stand-in values
  staged by Flock's `UnitStager` bound to the partition in statements the spec's shape admits (m <= 35), each stratum's
  share proved under `--zk` across its template's statements and held by the Lean verifier with the units it derives itself.
- `composite.py`: onsite's or remote's case from its subprotocols' cases, and the Lean verdict on it. Each `--cases DIR`
  is a private directory (0700, files 0600) holding one subprotocol's `case.json` (its `serve` line, with `blobs` the
  paths of the files it reads) and its own Lean answer on it, `answer.json`, whose `commitment` is its digest (or a
  directory of such directories); for onsite, the inference-only policy's case is the registration. The composed params, commitment and
  evidence (the composition's formats, `verity/protocols/remote/verifier/composition/README.md`) and every answer go to
  `--private-out` (0700, files 0600), since a case's evidence and params are witness, and so are an answer's query,
  digests and profile. `--public-out` gets only the verdict lines (each piece's and Lean's id, verdict and refusal's
  name), the composite's commitment, and each case's params digest and commitment, and is refused when it shares a
  byte run with any case's evidence. With `--exe`, it runs `onsite check` or `remote check` and exits with
  Lean's code. It decides nothing: a subprotocol with no case is left out, Lean's verdict on what remains stands (a
  rejection naming the missing case), and the run is labelled `NOT_ESTABLISHED` with what is missing.
- `pouw_piece.py`: PoUW's `--cases` directory for `composite.py`, remote's `verity/pouw/v1` (`v1`: the verifier's params,
  the window's one-stage registration and the proof system's answer to it) or onsite's `verity/pouw/period/v1`
  (`period`: the key, main's first-parent history, the period's name, the auditor's ledger, the pins log and each
  window's evidence), and its answer, `pouw-check`'s or `pouw-period`'s with `--exe`, written private as `composite.py`
  reads it. From a served run (`--serve-run DIR --window K`) it names the piece MISSING with why and writes no case, so
  `composite.py` leaves it out: an H1 window has no `v1` case, and no `period` case outside a period or without its sampled
  proof blocks' proof outcomes.
- `v2_case.py`: sampled proofs v2's `--cases` directory from a served window's kept files (`case_v2` of its v1 case,
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
  staged in statements of 1 to 512 units, and sessions of 1 to 512 sampled proof blocks are proved by `flock-circuit prove --zk
  --gpu` in three arms: `dev`, as SPOT[Flock]'s proof server proves (no ZK server, the witness built on the device), and the
  ZK server's send check on (`fw`) and off (`nofw`) over the host's witness. The Lean verifier checks them, and the job fits the per-session,
  per-statement and per-unit costs with their standard errors, writing `costfit` observations. `ZK_DRAWS=c` proves every
  staged unit: the cost of c sampled proof blocks in one statement, up to a table's 2^(35 − k_log) (2048 at K = 4096). Each point
  takes its own GPU lease, and a node without CUDA 13.3 gets its packages unpacked under `FLOCK_WORK`.
- `prove_bundle.sh`: a sampled-proof-block replay bundle (`benchmarks/one_stage/interfaces`, `verity/drawn-unit-bundle/v2`) proved by
  `prove_bundle.py`: `PB_BUNDLE` of the window `PB_WINDOW`, else `BUNDLE_WINDOW`, the registered fixture `tests/fixtures/bundle-window` (a W8A8 MLP's window under a
  test key, its tampered twin and the interfaces' bundle of four sampled proof blocks under the window's registered `subset:4`).
  `tests/test_prove_bundle.py` holds that bundle to the window's public part (`tests/fixtures/bundle-window-public`). `PB_LIVE=1` requires the bundle's sample to be live. The bundle is held to the prover's window: its own check (the sample's
  sampling design the registered one among it), its run, registration and partition, each sampled member's public body and a live
  sample's receipt and record. Each sampled member is one `prove --zk` session (behind the Lean ZK server with `ZK_SERVER=1`), one at a time, served the bundle's
  sample, with Lean `verify --zk` as the verdict, overlapping the next session. The negatives are `zk_mutants.py`'s one-bit
  proof edits (`PB_FLIPS=N`: the first N), refused by Lean, and a tampered window's member, refused by Lean (and with
  `ZK_SERVER=1` stopped by the ZK server as well). `PB_BUILD_ONLY=1` only builds the prover, Lean verifier and ZK server into its `out/bin`, stamped
  with the trees they come from. A later run with `PB_BINARIES=` that directory uses them while the stamp matches its own
  source, so M3's reserved window doesn't spend time on the build.
- `server_rate.py`: SPOT's proof server's steady rate on one GPU, run inside a one-GPU `gpu-lease` with the GPU build of
  `flock-circuit`. Each staged type (`served_units/stage.py TYPE full`, every unit drawn) is proved by
  `ProofServer(provers=K)` at each K of `--provers` (a templated one, a table read, with its unit placed by
  `--tmpl-unit-bin`'s `flock-tmpl-unit`, and verified under `verity/flock-circuit/tmpl`), K streaming provers on the one device, each with a K-th of the job's
  threads (the proof server splits them). Every prover finishes its
  warm sessions before any timed one starts. Each K's row (`rate.json`) gives the statements a second over the steady
  window, timed by each session's filing by its verifier, since a prover reports a session only once it collects its
  verdict, `verify_ahead` sessions later. It also gives the device's busy share (nvidia-smi's utilization over that window) and memory
  peak, each prover's and verifier's peak RSS, the provers' CPU seconds a statement, the median phase times (the front is
  `e2e_s − prove_total_s`), and the cgroup's throttling and load. A row also says what prices it: `gpus` (the devices its
  provers held, so a statement costs `gpus / statements_per_s` GPU-seconds), `stage.blocks` and `stage.full` (the proved
  statement's proof blocks, and the most a statement of its circuit holds), `verify_s` (the verifier's mean seconds a
  statement over the window), `flat`: whether each prover's statement cache had stopped filling, its hit rate over the
  last quarter of its timed sessions within 0.01 of the third's, with the quarters and the rule in `cache`; and `cores`: the
  CPUs its processes could run on (the script's own affinity, which every process of the row runs within: unplaced, each
  inherits it; with `FC_VERIFIER_CPUS`, the verifiers run on `verifier_cpus` and the provers on the rest), as their
  `count`, their `list` and the `numa_nodes` that hold them. With `--lean`, the last sessions at
  K = 1 also get the Lean verdict inside the server's verdict pool, with its CPU seconds and peak RSS. A proof block's
  verdict is the `NO_PROGRAM` refusal, as in `served_units`. With `--wide-leaf` every prover and verifier of the run
  carries `flock-circuit --wide-leaf`, the wide level-0 leaf's statement version, and its rows say so (`wide_leaf`); the
  script's docstring has the rest: the two statement names their Lean verdict is asked under, what `flock-verify`
  takes and refuses under them, and how a row is held to the version its sessions ran. A row with a refused session, a
  server process that exited nonzero, a prover out of device memory, or a session that ran another statement version
  than the row's is void: no rate, its reasons in `void`, a `VOID` line, and the run exits 1.
- `zk_cost_cpu.sh`: the same sessions on a CPU node (`zk_cost.py prove --cpu`), proved by the CPU prover that
  `prove_path/run.sh` builds, with the ZK server's send check. It runs in a proofs slot, one point at a time. Each session's
  record carries its wall time and its summed peak memory (prover, live verifier and ZK server), which is what a CPU node
  spends on one sampled statement of a block's sample.
- `phase_profile.sh` (with `phase_profile.py`): one GPU prover phase (`PHASE`, default `t.ligerito`) on a pod of its own: the
  source's build (`AS=base` or `new`) on a served type at full (`UNITS`, default `moe-w13:N=1,SWAP=False`), its transcript
  pinned to main's under Flock ZK and Flock's circuit prover, one warm `zk_cost.py` dev process under Nsight Systems with
  CUDA and OS calls traced, and an A/B of base and new.
  `phase_profile.py` cuts each rep's phase at the prover's synchronizes and splits it into GPU time by kernel family and the
  host's time with the GPU idle: its OS calls (a live coin's round trip), its CUDA calls and its own compute.
- `forged_proofs.sh`: the fixture of Flock's forged-proof negatives (`verity/protocols/flock/prover/tests/test_lean_forged_proofs.py`),
  on hidden_zk's cached builds (`forged_proofs.py`). A small A8 unit (`LlBf16DotprodDot_v1` at K = 128) staged bound to its partition,
  with one honest sampled `--zk` session and two whose committed output word is wrong, proved with the prover's self-check
  skipped (`FC_ZK_SELF_CHECK=skip`), so the Lean verifier gets a well-formed proof of a false statement.
- `form_refusals.sh`: the fixture of `verify --zk`'s form refusals
  (`verity/protocols/spot/verifier/tests/test_form_refusals.py`), on the same builds (`form_refusals.py`): honest
  `--zk` sessions of that unit flat, of it as its template, and of a toy whose last stage returns the outputs, with the
  flat session's verdicts under four statements `verify --zk` must refuse.
- `a11.sh`: run A11's entry point, on A6's builds. A11 is GLM-5.3 TP8 B1's served step, rank 0: `a11_kinds.py` censuses
  the request Program's root Calls by Definition kind and tries one unit of each through the circuit prover (a kind whose word units it
  refuses, through its Boolean version's); `a11_slice.py` cuts the
  capture's values and the checkpoint's tensors for a few layers off the serving node; `a11.py` stages each lowering kind's
  units from those values, one template each, under the stratified sampling design, proved under `--zk` and held by the Lean verifier;
  `a11_table.py` renders the per-kind table and its gap lines from the audits' summaries.
- `a9.sh`: run A9's entry point, on the same builds. A9 is GLM-5.3's routed-expert FP8 GEMMs in the registry's Boolean
  static pair form (`boolean_fp8_moe`): `a9.py` holds one coordinate of each launch's first routed pair to the capture in
  its word and Boolean forms, cuts an up/gate coordinate whose rows overflow the circuit prover's block into its gather and
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
  samples from, its rows, op paths, ports and whether `serving_rows.template` takes it; `--estimate` adds each family's
  ANDs, table reads and assertions from its leaves lowered once (table reads counted apart, so not comparable with a
  statement's `ands` with its reads inline: `m1_families.py`'s docstring), against 2^26, and
  the log2 of Flock's circuit prover's block for its unit against `K_MAX` (the hashing of its rows alone bounds that block from below), the
  same for each of its split units, and with `--also`/`--also-file` for Definitions a split main lacks would use, with
  each unit's served MACs and the bits of each of its ports; `--steps all` takes every request step.
- `block_census.py` (run by `block_census.sh ROW_DIR` on the machine holding the row's Build): the proof-block census of
  a TP Commit's Build. Every rank's `m1_families.py --steps all --estimate`, each Program row cut into proof blocks (the
  row itself when it fits Flock's block and a size class, else each slot of its split at its coarsest fitting unit, else
  a gap with the reason; a `COLS` row or slot, and the block-scaled FP8 GEMM's coordinates (`WIDE_COORDS`), at its fitting
  width of least leakage-weighted proving cost, `width_pick`),
  and per proof-block Definition its count in the prefill step and the decode steps, output bits
  and size class, estimated ANDs, served MACs and opened row bytes, with the totals per token and per request, the gaps
  and the rows that are no proof block (`census.json`, `table.md`), and the planner's card (`card.json`,
  `benchmarks/sampling_plan/glm53.py`).
  `m1_unit.py` builds one family's unit through Flock's circuit prover's `UnitStager`, flat or `--structural`, and with `--bin`
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
- `served_composite.sh`: `served_verdicts.sh`'s step after its record. It names PoUS's and the wipe's composite cases as
  their pieces wrote them (PoUS's `onsite_pieces.sh`), then gets remote's and onsite's Lean verdicts on the run's cases
  (soundness's composite reader), all on the node. Only the verdict lines reach the run's `verdicts.txt`.
- `onsite_periods.py`: the onsite record (`verity/onsite-network-record/v1`) per monitoring period, a week, a month and a
  year by the node, a datacenter and the world, each cell its K, share of the weights and δ_net beside K over one
  execution phase, marked proved or scaled by the record's rules. It refuses a record whose labels its rules contradict:
  K/m proved past one phase, a node cell proved over a period the run doesn't cover, a datacenter or world cell proved,
  a proved statement outside the lock, a rule without its sources. The tight K is shown conditional on what is open.
  Its test reads stand-ins cut from the writer's own record (`tests/fixtures/onsite_periods/make.py`).
- `served_util.sh` (with `served_util.py`): the whole-step tensor utilization of serve-openai on the int8 Llama-3.2-1B
  prefill mix (prompts of 1,024 ids, `max_tokens` 1, all queued at once, T tokens a step), one arm a run (stock, the
  Commit unsealed or sealed, the Commit with PoUW H1 on in-process coins), under Nsight Systems' GPU metrics.
  `served_util.py` takes Tensor Active's mean over the longest run of full steps, each step from the previous step's end
  to its own on the step trace's clock, and splits the rest into losses by what the GPU was running at each sample.
  With `PYSPY=1`, `pyspy_steps.py` splits the same steps' host time by Python function and thread, from py-spy's trace.
- `served_shares.py SERVE_DIR [--credit ANSWER.json]`: a served run's utilization (the work its step trace's
  steps declare, the linears at θ and attention and the LM head at θ/2, over the loads' busy wall) and its accounted share
  (the period credit's work at J10-R's 9/64 floor at θ, over the same wall), at the tensor and boost clock readings.

Flock's circuit prover (`verity_flock.circuit`) is on `main`, but a run does not take it from its checkout:
`pod.sh` takes `git archive`s of the circuit prover's tree and the Lean verifier's tree at pinned commits, and a captured input set, as
inputs sent with the run; it builds `flock-circuit` and `flock-verify` once per pod, then runs the driver, whose
`--m0-pythonpath` names that tree. The drivers' checks go through the verifier
(`one_stage.verifier`), which builds the sampled-proofs Lean in the run's tree on its first check, hence `--cwd clone`:

~~~sh
bash benchmarks/one_stage/pod.sh    # a job run from a clone of the tree, with m0.tar, lean.tar and set.tar sent
~~~

`check` runs nothing here.
