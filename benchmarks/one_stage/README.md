# One-stage audit runs

The one-stage audit (`verity.core.protocols.sampled_proofs.one_stage`) end to end: serving's committed values through registration, the verifier's
sample, C-Flock's circuit prover (M0) and the Lean verifier, to the audit record with its integrity profile, then the
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
  a `served` window under the 64-byte run root. Then the Commit's CPU replay (the prover's pre-flight check, on the node), and
  `a5.sh` with `WINDOW=served` on the bundle and that window, whose private bodies stay on the node.
- `a5_tp.sh`: A5 over a TP Commit's windows, one `served` window per rank (`a5.py --tp-commit`), taken as served only under a
  SHA-512 TP run root (`tp.run_root`'s v2, over the ranks' rows-sha512 roots; an old record's SHA-256 v1 root is refused before
  the sample): GLM-5.3's TP8 Commit with
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
  verifies each sampled chunk with its table's registered reads (`--registered`, `tables.registered_reads`).
- `a5_warden.py` (`a5_tp.sh` with `STANDIN=warden`, `warden:altered`, `warden:pad`, `warden:pad-named` or `warden:swapped`):
  A5 on a window with transfer gates, which no served window has yet: two chunks of the warden's toy unit whose ports are the
  bucket rows of one connection's ingress and egress cells (a row a bucket, registration v2), committed under the warden's
  salts, proved whole (the work law's selection All) beside a sampled rest, on a local challenger's coins or the verifier's
  own. The units' wiring is the Program's (`warden_rows.Units`, from public structure alone: ingress cells 0 … n − 1, egress
  cells D·r … D·r + n − 1). Lean verifies each chunk with the verifier's own registered reads at the root the warden signed,
  named by the units' value and each unit's read (connection, cell) placed by the wiring the warden signed
  (`warden_rows.port`, after `check_registration` under the warden's key), which refuses a foreign, a duplicate and a
  padding-slot read by name; the altered run, and a statement made for another units' wiring (a pad in a data cell's place,
  its rows under the Program's value, two units swapped: `--made-for`), are refused by Lean's registered reads. Its warden run
  is rehearsed against a fake endpoint, or (`A5_WARDEN_RUN`) is `warden_replay.py`'s: a served window's requests through the
  warden, each answered with the tokens the server streamed it.
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
- `examples/vllm/commit.sh` (the vLLM example's): the single-GPU counterpart of `a5_glm53_commit.sh`'s deferred Commit, on Llama-3.2-1B bf16
  bi-eager TP1 (one RTX PRO 6000 of vy-nebius-1): the workload's Build and Match of record, a config-run Commit that seals
  `commit/replay_bundle_p0/` (a single-GPU workload defers only as a config run), and its replay on CPU. The workload dir,
  bundle included, stays on the node; the run publishes the stages' records (`research-outputs write`). A deferred
  single-GPU Commit's window comes from its bundle through `verity-vllm serving-rows`, not `tp-serving-rows`.
- `served_verdicts.sh` (with `served_verdicts.py`): one served window through every subprotocol, in one job on an H200 node
  from a clone of the tree (`bash benchmarks/one_stage/served_verdicts.sh`). It serves
  Llama-3.2-1B int8 on one H200 under `gpu-lease` (`serve-openai --scheme rows-sha512 --seal --admit release-order`), sends a
  few greedy requests from the certifier's traffic generator, and seals each window. Before the first request, the committer
  registers its weights under its own key (`prove_path.py weights-registration`). Without a registration, `NO WEIGHTS
  REGISTRATION` is printed and every weight read counts free. On the window holding the most requests, none carried, it runs
  sampled proofs: the window's advice under that registration (`--weights-of-record`) and its request Programs, V3b's window
  circuit at each class of `CLASSES` (V2's 4,32,4 by default), with its weight reads counted registered at the registration's
  pins. The run needs every one of those counts. The class of record's count follows them as a measurement under its own
  `WC_RECORD_TIMEOUT` (default 20m), after the record: a timeout or refusal there is reported and never fails the run. Then
  comes V3c, sampled proofs' audit of record, within `V3C_TIMEOUT` and `V3C_AUDIT_TIMEOUT`: the window committed as the
  class circuit's units (`prove_path.py window-commit`), then its records audit (`records_audit.py audit --circuit`, on a
  local development challenger session's coins unless `PP_CHALLENGER` names the live challenger, as the run's COINS line
  says, a deferral's follow-on its own), which is C-Flock `--zk` on the drawn proof blocks and the Lean verifier. The class circuit is the
  class's, whatever the window's requests, and reads no advice, so no advice opening reaches the verifier. V3c writes its audit as the network
  certifier's period reads it (`period.json`). Then come PoUW, PoUS, the bandwidth audit and the network certifier. Each runs its piece where
  `POUW_CMD`, `POUS_CMD`, `BANDWIDTH_CMD` or `CERTIFIER_CMD` names one, within its own limit; otherwise its `MISSING` line
  names the command it lacks and its owner, and it gets no verdict. The script sets each owner's command by default where
  the tree has its piece and the node holds what the piece reads (its auditor key, the bandwidth audit's registered store).
  The bandwidth audit's session runs inside PoUS's piece, over the store that piece holds
  (`benchmarks/pous/served_piece.py`), and `BANDWIDTH_CMD` collects its answer (`served_bandwidth.py collect`).
  Finally the all-protocols window report and the block record, which accepts only when every subprotocol accepted on this
  node and names each one missing, rejecting or about another server, with its reason (PoUW's development piece: development
  run, credits nothing). `PRODUCT=remote` composes sampled proofs and PoUW alone: the onsite pieces are dropped, not run
  and not composed, and the preflight refuses any onsite piece's command. `CERTIFIER_FRONT` is not a piece but how the
  requests are served, so a remote run may set it (with `LOAD=front`) to serve through the network certifier in front
  of its endpoint, its verdicts unchanged. `PRODUCT=remote-network` composes sampled proofs, PoUW and the network
  certifier, dropping the wipe, PoUS and the bandwidth audit as remote does, and its composites are remote's and
  `verity/remote-network/v0`'s, which refuses a network certifier's period whose compute audits are not all on the run's
  registration record (its `egress` tie). So a remote-network run takes the split certifier below, whose one compute
  audit is V3c's of the served window, and its preflight refuses `CERTIFIER_CMD`, whose audits are its scan's own
  registrations. `PRODUCT=remote-network-no-pouw` is the same run without PoUW's piece, its one composite
  `verity/remote-network-no-pouw/v0`, over the network certifier's, the policy's and sampled proofs' cases and never
  PoUW's period case, so its record credits no verified compute. An accept's clause says when its weights were held and
  their share of the checkpoint, where its profile says so (PoUS's), and each δ of the block record's at which its quantity is not established. With the
  certifier's piece split in two (`CERTIFIER_CHAIN_CMD` and `CERTIFIER_PERIOD_CMD`), V3c's step also runs the chain after
  its window commit, beside its audit (within `CERTIFIER_TIMEOUT`, default 4h), and the period check on its
  `period.json` once both are done, whose answer the certifier's piece reads. Each
  step's verdict line is in `out/verdicts.txt`. The network certifier's piece, the longest, starts as soon as the window is
  picked. No line counts a window's requests: the serve dir's request lists, serve-openai's log and the load's records,
  timings and log stay on the node, and leave it only as `served_overhead.py metrics`' aggregate.
- `a6.sh`: run A6's entry point on a CPU machine, on hidden_zk's cached builds. A6 is the W8A8 coordinates of an int8
  decoder layer (`ScaledMmS8Coordinate_v1` at each K, Llama-3.2-1B's by default) under the stratified law, stand-in values
  staged by C-Flock's `UnitStager` bound to the partition in statements the spec's shape admits (m <= 35), each stratum's
  share proved under `--zk` across its template's statements and held by the Lean verifier with the units it derives itself.
- `composite.py`: onsite's or remote's case from its subprotocols' cases, and the Lean verdict on it. Each `--cases DIR`
  is a private directory (0700, files 0600) holding one subprotocol's `case.json` (its `serve` line, with `blobs` the
  paths of the files it reads) and its own Lean answer on it, `answer.json`, whose `commitment` is its digest (or a
  directory of such directories); for onsite, NCI's case is the registration. The composed params, commitment and
  evidence (the composition's formats, `verity/core/protocols/composition/README.md`) and every answer go to
  `--private-out` (0700, files 0600), since a case's evidence and params are witness, and so are an answer's query,
  digests and profile. `--public-out` gets only the verdict lines (each piece's and Lean's id, verdict and refusal's
  name), the composite's commitment, and each case's params digest and commitment, and is refused when it shares a
  byte run with any case's evidence. With `--exe`, it runs `onsite check` or `remote check` and exits with
  Lean's code. It decides nothing: a subprotocol with no case is left out, Lean's verdict on what remains stands (a
  rejection naming the missing case), and the run is labelled `NOT_ESTABLISHED` with what is missing.
- `pouw_piece.py`: PoUW's `--cases` directory for `composite.py`, remote's `verity/pouw/v1` (`v1`: the verifier's params,
  the window's one-stage registration and the proof service's answer to it) or onsite's `verity/pouw/period/v1`
  (`period`: the key, main's first-parent history, the period's name, the auditor's ledger, the pins log and each
  window's evidence), and its answer, `pouw-check`'s or `pouw-period`'s with `--exe`, written private as `composite.py`
  reads it. From a served run (`--serve-run DIR --window K`) it names the piece MISSING with why and writes no case, so
  `composite.py` leaves it out: an H1 window has no `v1` case, and no `period` case outside a period or without its drawn
  units' proof outcomes.
- `v2_case.py`: sampled proofs v2's `--cases` directory from a served window's kept files (`case_v2` of its v1 case,
  partition, public file, sessions and circuit file, and the challenger's release of the draw's coins, `release.json`),
  naming the challenger v2 requires by the auditor's own key record (`--challenger-key`), never by the key the release
  names: a release naming another key is refused. Its params name the public file by the SHA-512 a5 pinned
  (`public-pin.json`): as completed once the tie held, the file the drawn member's session proved, else as staged
  before the draw; never one read after it: a window with no pin, or whose kept public file isn't the pinned one, is
  refused. Its params' layout (`proofs.members`) is the one the pin fixed before the draw, each member's key and `n`
  (`a5.pin_layout`); a pin that states none is refused. It reads the drawn member's files, the one member whose sessions the window kept (none
  or several refused), its `compose.log` its own or its template's first member's. With its own Lean answer with
  `--exe`. A circuit file the run didn't keep is rebuilt by the command its `compose.log` records and used only at the
  log's pin.
- `kept_composite.sh SRC [K]`: remote's and onsite's Lean verdicts on a served run's kept records (`served_verdicts.sh`'s
  `out/`, or a tarball of it) on a CPU node: it builds the products and the pieces' checks, writes PoUW's pieces
  (`pouw_piece.py`; the period from the kept `pieces/pouw/`, `main.txt`, `H1_KEY_SHA512` and the drawn and live legs'
  statements, `POUW_STATEMENTS` and `POUW_LIVE`), v2's (`v2_case.py`, under the auditor's key record
  `PP_CHALLENGER_KEY`) and every kept piece holding a case beside its answer, all private in scratch, and runs
  `composite.py remote` and `onsite`; `$OUT` gets only their public output.
- `a8.sh`: run A8's entry point, on the same builds. A8 is one window of the remote bandwidth audit over a registered store of
  a few thousand blocks with every opening proved by the opening unit (`remote_bandwidth.opening`) under `--zk` and held by
  Lean, honest and with a tampered block, a scan of everything the verifier received for the sampled proof blocks' bytes, and the
  prover's cost per opening. `a8_gpu.sh` takes its cost points on sm_120 (`flock-circuit prove --zk --gpu`), as a GPU host
  job, and `a8_cost.py` reads a run's sessions into the one cost model's observations (`benchmarks/sampling_plan`).
- `a7.sh`: run A7's entry point, on the same builds; the remote session's checks 1-3 and 5 on each sampled receipt are
  proved, not opened, by the opening unit (`remote_bandwidth.opening`, `verity.ml.commitments.gates.block_check`) in a
  second statement. A7 is the requirement's plan and memory-map units
  (`verity.ml.circuits.requirement_units`), one of each per step of a stand-in decoder's requirement, under the stratified
  law, their witness rows bound to canonical commitments, then wired to the remote bandwidth audit's ledger as its
  requirement link (`audit.RequirementLink`), its ε the most share of the linked credit the units its sample leaves
  unproved can carry and its δ the sample's.
- `plan_shards.py`: the sharded plan unit (`PlanShard_v1`) on a recorded remote session (vllm's `remote_session.json` and the
  remote verifier's `verifier.json`): each pass's plan cut into the store's shards, every instance held to the unit's
  reference, each distinct one to the circuit, and one staged by `UnitStager` for its fit in M0's block. A CPU job; it
  builds nothing.
- `zk_cost.sh`: C-Flock's zero-knowledge session cost on sm_120, for the sampling planner's cost model (`zk_cost.py`, a GPU
  host job, or a preemptible guest job on a shared node). A6's W8A8 coordinates at K = 4096 and 14336 are
  staged in statements of 1 to 512 units, and sessions of 1 to 512 drawn units are proved by `flock-circuit prove --zk
  --gpu`, with the firewall's send check on and off. The Lean verifier checks them, and the job fits the per-session,
  per-statement and per-unit costs with their standard errors, writing `costfit` observations. `ZK_DRAWS=c` proves every
  staged unit: the cost of c sampled proof blocks in one statement, up to a table's 2^(35 − k_log) (2048 at K = 4096). Each point
  takes its own GPU lease, and a node without CUDA 13.3 gets its packages unpacked under `FLOCK_WORK`.
- `prove_bundle.sh`: a sampled-proof-block replay bundle (`examples/h200/interfaces`, `verity/drawn-unit-bundle/v2`) proved by
  `prove_bundle.py`: `PB_BUNDLE` of the window `PB_WINDOW`, else the store's `BUNDLE_WINDOW` (a W8A8 MLP's window under a
  test key, its tampered twin and the interfaces' bundle of four sampled proof blocks under the window's registered `subset:4`).
  `tests/test_prove_bundle.py` holds that bundle to the window's public part (`tests/fixtures/bundle-window-public`). `PB_LIVE=1` requires the bundle's sample to be live. The bundle is held to the prover's window: its own check (the sample's
  law the registered one among it), its run, registration and partition, each sampled member's public body and a live
  sample's receipt and record. Each sampled member is one `prove --zk` session behind the Lean ZK server, one at a time, served the bundle's
  sample, with Lean `verify --zk` as the verdict, overlapping the next session. The negatives are `zk_mutants.py`'s one-bit
  proof edits (`PB_FLIPS=N`: the first N), refused by Lean, and a tampered window's member, stopped by the ZK server and
  refused by Lean without it. `PB_BUILD_ONLY=1` only builds the prover, Lean verifier and ZK server into its `out/bin`, stamped
  with the trees they come from. A later run with `PB_BINARIES=` that directory uses them while the stamp matches its own
  source, so M3's reserved window doesn't spend time on the build.
- `zk_cost_cpu.sh`: the same sessions on a CPU node (`zk_cost.py prove --cpu`), proved by the CPU prover that
  `prove_path/run.sh` builds, with the ZK server's send check. It runs in a proofs slot, one point at a time. Each session's
  record carries its wall time and its summed peak memory (prover, live verifier and ZK server), which is what a CPU node
  spends on one sampled statement of a block's sample.
- `forged_proofs.sh`: the fixture of C-Flock's forged-proof negatives (`verity/ml/flock/tests/test_lean_forged_proofs.py`),
  on hidden_zk's cached builds (`forged_proofs.py`). A small A8 unit (`LlBf16DotprodDot_v1` at K = 128) staged bound to its partition,
  with one honest sampled `--zk` session and two whose committed output word is wrong, proved with the prover's self-check
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
  samples from, its rows, op paths, ports and whether `serving_rows.template` takes it; `--estimate` adds each family's
  ANDs, table reads and assertions from its leaves lowered once (an upper bound on the flat count), against 2^26, and
  the log2 of M0's block for its unit against `K_MAX` (the hashing of its rows alone bounds that block from below), the
  same for each of its split units, and with `--also`/`--also-file` for Definitions a split main lacks would use.
  `m1_unit.py` builds one family's unit through M0's `UnitStager`, flat or `--structural`, and with `--bin`
  and `--lean` proves one round of it on stand-in bf16 values under `--zk` and verifies it with the Lean verifier.
- `served_stock.sh` (with `served_overhead.py`): `served_verdicts.sh`'s stock control, the same endpoint, engine and load
  served with `serve-openai --no-commit` and no protocol, once per fresh endpoint for `REPS` endpoints. `served_overhead.py
  table` compares the composed run with it on two clocks, each the same code in both runs: the client's
  (`serve_load.py --varied`'s timings files) and the engine's step trace (`advice_trace.jsonl`). The trace and the load's
  records stay in the serve dir on the node, where `served_overhead.py metrics` reads them, and each run publishes only
  that aggregate (`overhead.json`), which `table` reads. It reports time per output
  token, time to first token and tokens per second as composed over stock, the composed run's window open and close on
  the engine loop, and `verity-bench served-table`'s single-protocol rows beside them.
- `served_composite.sh`: `served_verdicts.sh`'s step after its record. It writes PoUS's and the wipe's composite cases
  (PoUS's `served_case.sh`), then gets remote's, remote-network's and onsite's Lean verdicts, or remote-network-no-pouw's
  alone, on the run's cases (soundness's composite
  reader), all on the node. Only the verdict lines reach the run's `verdicts.txt`, and a tree without those two scripts skips the step.
- `onsite_record.py OUT [--advice-charge FILE] [--certifier-key RECORD --site-registration FILE]
  [--inference-only-case-policy FILE]`: the onsite demo's record of a served run (`record/onsite.json`), which
  `served_verdicts.sh` writes after its composite step when the network certifier is in the run: its piece
  (`CERTIFIER_CMD`, or the split's `CERTIFIER_CHAIN_CMD` and `CERTIFIER_PERIOD_CMD`) or its front (`CERTIFIER_FRONT`). It holds every verdict
  the run of record needs (sampled proofs, PoUW, inference-only, remote, the network certifier, onsite), each missing
  with its why where the run has no answer. Beside them it lists memory's three (PoUS, the bandwidth audit and the
  wipe: the window report's `space`, `bandwidth` and `wipe` rows) as their rows state them, missing where the run has
  no such row; whether the run is of record reads none of them. A verdict read from a row is `accept` only where a
  server is named for the row's answer (the report's `servers`, or its piece's records for the period check's answer
  a remote run takes) and it is the report's node: an answer about another server, an accept naming none, or an answer
  about the node whose source the report's `sources` or that piece says isn't `real` (an answer brought from a recorded
  run, whatever it says), is missing, says which, and gives the record no figure. A composite is read by the product
  its `composite.json` states, one a product: where two state one product, or one states none, the record reads none
  of them, and each verdict it would read from them is missing, naming the files.
  Sampled proofs' verdict is its leg in the composite of record, the outcome
  of remote's sampled proofs case there (one-stage v2), never the window report's integrity row, the records audit's
  one-stage v1 answer, which is no verdict: where the composite of record has no such case, it is not established, the
  row's answer beside it, and credits no member. The inference-only verdict says where its case's α and Γ came from, never
  a registered policy: off the run's own division, so the verdict confirms the division against itself, where the
  case's private `policy.json` says so (its `read`, the one field the record reads), else the run's `ALPHA` and `GAMMA`.
  It checks three obligations on each run (``checked``), reading only the auditor's inputs and outputs: `auditor-record`, that the period's key, LMS parameters and
  session, in parameters whose SHA-512 is the one the composite of record states for the network certifier's case, are
  the auditor's record of its network certifier (`CERTIFIER_KEY`, `certifier-key/v1`) and that the
  auditor's registration of the deployment (`SITE_REGISTRATION`: the registration log's params and the log, which
  checks) holds that key for a network certifier its log places, reading none of its link figures (beta and latency,
  stated parameters, not measured; that the node has no egress but the uplink the network certifier splits, which the
  registration asserts, is the auditor's placement under trusted-host, which no file establishes); and
  `auditor-listing`, that each circuit file the
  period lists is, by SHA-512, the one its audit's sessions were verified against, and the remote theorem's
  registered-files condition: each file remote's sampled proofs case registers, the circuit file and the public file, is
  by SHA-512 the one its params θ name (`proofs.circuit`, `proofs.public`), θ held to the params each composite that
  reads the case states, the files hashed and never read off the evidence, and the public file one a5 pinned
  (`--public-pin`): as staged before the draw, the staged bodies' leaf layer the one the registration the coins
  committed to holds, or as a5 completed it from the staged file once the tie held, its body that member's staged
  body, or an instance-major member's (the pin's `relaid`) re-laid port-major (`completed_from`), and θ's layout
  (`proofs.members`) the one the pin fixed before the draw (`layout_pinned`), a check of the record and not yet of Lean,
  which the record and its ONSITE line state as open (`MEMBER_OPEN`); and `scan-state-pinned`, that
  where an audit's window commits the scan's state in its activations (the entry `NetworkCheck.Period` holds to the
  scan's), V3c's audit opened its registered reads of it at that entry's
  value, root and domain (its published summary's `registered.own`), accepted, and reported each sampled member's
  registered reads: one-stage's registration checks read nothing of the window, so the audit's driver pins it there,
  and this checks that it did. `certifier-key` closes only where the
  lock holds PeriodLeakRegistered at its pinned statement taking hA, hr and hkey as `auditor-record` and that check
  holds; while it is open, an accepted network verdict is not established, its Lean answer's accept beside it, and its
  bound conditional on `certifier-key` by name. Where `auditor-listing` doesn't hold, the verdict is not established
  and there is no bound; where `scan-state-pinned` doesn't, its evidence missing included, neither is it nor the
  composite of record's, and there is no bound; where its registered-files condition doesn't, neither is a composite's that reads the case,
  nor sampled proofs', its leg there; and where θ names no public file, the public file isn't bound and sampled proofs'
  verdict is not established.
  Sampled proofs credits only the drawn members whose `--registered` file its audit completed at the window's
  committed roots and whose Lean verdict of record reported exactly that file's reads, and lists every other drawn
  member as not credited, by name; PoUW's freed compute is credited nowhere and not claimed, by name, beside the PoUW
  piece the run's COINS line says it served, and whether the run is of record reads every verdict but PoUW's, which
  stays listed as its row states it (missing and development for a development piece). Its composite of record, named
  by its id, is the network certifier's composite without PoUW where the run has one (registered under its own id in
  `Core.Composites`), and then whether the run is of record reads that composite and its subprotocols, and not remote's
  composite, which composes PoUW's case; else it is remote-network's. Either way the record credits no verified compute,
  and its `ONSITE` line, like each run's in `--beside`'s `ONSITE-BESIDE` line, names the composite by its id and says so
  in words: "no verified compute is credited: <id>'s verdict is no compute claim".
  Sampled proofs' guarantee is what its draw establishes, ε at δ, per window (its answer's bound over the window's n
  proof blocks) and pooled, never the window's alone: at most ⌊ln(1/δ)/q⌋ incorrect proof blocks over any number of
  windows each drawing k of n (q = k/n, `Flock.Guarantees.IntegrityHorizon`), stated for each monitoring period for the
  node, a datacenter's and the world's links (each link's at δ / links, summed), with its share of their proof blocks at
  a window an execution phase; a law other than a k-subset has no pooled figure, and a draw of several members
  (stratified:1, H1_TIE's default) is named refused, since it writes no sampled-proofs case. Beside them are the network
  certifier's K term by term in the notation table's symbols, on one-stage answers the driver verified; the check read
  them as given, so the record calls K no bound and states "answers read as given" beside it wherever it or one of its
  lines states K (the record's `bound` and its figure of record, the `ONSITE` and `ONSITE-BESIDE` lines),
  with its overcount, the seconds it covers, the
  connections it counts and δ_net term by term with each term's source (the check's `eps`, only its δ part, and each
  phase's knowledge, binding and collision errors, which no check computes), `eps` split into the scans' δ and the
  charged tied audits' with what sets each and what a smaller one costs, k_{skip,j}'s slot entries (every one of a
  scan sampled in part to which a tied audit sampled in part is tied; a scan with every proof block drawn is charged
  its slot entries once in ℓ_{a,j} instead, whatever its tied audits' draws), the composite of record's
  error as the sum of its subprotocols' errors (four, or three without PoUW's), each δ with where the
  run states it and the terms no check computes named. While the period's check reads its answers as given the record
  makes no period's figure from K (`periods` is missing, with why: nothing scaled, pooled or summed, no cell called
  proved, no share of weights or egress; `ANSWERS_VERIFIED` in `onsite_record.py` is the one switch for what a record
  is written as). A written record is read by what it says: one written on verified answers says so in its own
  `answers`, and any other, one written before the label included, is read as given whatever the reading code's
  switch, so `--beside` prints its K and its cells under "answers read as given". Once the check
  verifies them, a record states that K over the
  run's own certification period and each monitoring period for the node (scaled where the run's period doesn't
  cover it), the census's reference datacenter and the world (scaled), each scaled cell naming its rule, whose sources
  the record states, and each other cell reading proved only where nothing is open under the figure of record, and
  otherwise open, naming each premise, check or condition that is; beside its figure over one execution phase, and
  its share of one copy of the
  weights a deployment protects (the census adversary model's as published, a stated parameter with its census line),
  beside it of the demo's served model's (Llama-3.2-1B), and of the schedule's egress capacity (the served egress is in no published file, and the record says so). The cells are
  the figure of record's: K as the period's check states it, with nothing added to it. It is proved only where none of
  its premises is open: each is discharged by theorems `Security/lean-audit.json` or
  `Security/Properties/lean-audit.json` lists, taking no named assumption but a standing one (trusted-host and
  certifier-coverage), and no premise about our own Python stands; each guarantee the figure rests on is in the lock
  at the statement whose hypotheses and antecedents the record names by hand, each by the premise that covers it, since
  the lock records a guarantee's named assumptions and never its antecedents; the run's COINS line names the live
  challenger (PP_CHALLENGER gate:USER@HOST at a named peer); and, where the check charged a phase its tied audits (its
  scan sampled in part), its Lean verdict held both checks that charge rests on (each tied audit's free inputs computed
  from its descriptor, and its circuit held fixed whatever its advice). The record names a theorem as a guarantee or a
  discharge only where one of those locks lists it (`onsite_record.unlisted`, which a test holds empty), so a premise
  whose discharge isn't there is open. While anything is open the figure is the same K, open, each by name, and so is
  each cell that would read proved: no bits are added for what is open, the advice's cap included, since the link
  figure charges the advice through the entries, a scan's slot entries, which the check charges once where every proof
  block of the scan is drawn. A phase charged so rests on that cut (`entries-cut`, open till its statement is in the
  lock) and on none of the tied charge's premises or checks. The record says how the figure tightens, in its place and
  never beside it. For a phase charged its tied audits, the figure says which charge K applied each (`state_charge`):
  its free inputs with its reads of
  the scan's state held (`stateFree`) only where the lock holds RowsStateHeld at its pinned statement reading
  `stateHeldIn` and `auditStateEntries` at their pinned digests, so that its proof stands over the entry a window's
  activations commit, else `free`; where it is `free` and a tied audit's parameters state its circuit record, whose
  `stateFree` may be less than its `free`, there is no figure. While the coins are a local
  development session's, or no line names them, or the network certifier's period draw or its scan's draw takes K of
  what it draws from (or the run published no chain report), the condition live-coins is named, with each such draw,
  on every figure that rests on the draw (the sampled-proofs verdict where its draw is the one open, the figure of
  record and its cells), never as a live draw's.
  The realized advice charge is private: the record states ℓ_{a,j} with its source,
  the scan's slot entries or the tied compute audits' free inputs (no advice length and no cap is in them), beside it
  ℓ_a, the cap of the advice the run registered (its published `registration.json`'s compiler `cap_bits`), as a cap,
  in no term of K and in no figure, and
  `--advice-charge` states a test
  run's measurement beside K as that run's. It lists the re-proof inputs, what a later re-proof of the sampled
  units with each wire read inside its session needs (the committed values and openings of every sampled unit and of
  the units that write what it reads, the registration, the draw and its receipt, and the public files), by
  where each is and whether the run wrote each public one, and how each must be kept: past the week after the run,
  never only on the served node, the public ones in custody and the serve dir's, witness, in a private copy off the
  node, never in `$OUT` or custody. `--reproof-copy NODE:/PATH` with `--reproof-listing` (the copy's listing, kept
  off `$OUT`; `REPROOF_COPY` and `REPROOF_LISTING` in `served_verdicts.sh`) names the copy and checks each serve-dir
  input against the listing by its path alone: each file it names, and at least one file under each directory it names.
  Under `H1_TIE` (`--h1-tie`, which `served_verdicts.sh` passes) the circuit it names is the record reading H1's tables,
  `window-circuit-w<K>-<class>-tie.json`.
- `site_registration.py CERTIFIER_KEY OUT --node NODE --certifier ID --beta BITS --latency TICKS [--site SITE]`: the
  auditor's registration of its network certifier's deployment, `SITE_REGISTRATION` (`onsite_record.py`'s
  `--site-registration`), written on the auditor's side from its own `certifier-key/v1` record, `CERTIFIER_KEY`, and the
  ids and uplink figures it is given, never from the prover's node or its deployment.json: remote-network's params,
  their auditor holding the record's public key under ID, and a log of one site entry placing ID, the auditor's network
  certifier with that key, on NODE's uplink, a link from NODE to ID and one from ID to the outside, each of capacity
  BITS a tick each way and latency TICKS ticks. It writes OUT only where `registration.params_of` reads the params back
  and `registration.check_log` accepts the log under them, and refuses a record of a run in front of another node.
- `advice_charge.py ADVICE --run RUN --to FILE`: honest serving's advice charge from `verity-vllm advice`'s output
  over a test run with public inputs: each Commit period's advice bits, unpadded, beside its requests' prompt and
  output lengths and finishes, labelled as that run's. `served_verdicts.sh advice-charge` serves such a run
  (`PRODUCT=remote`, `H1_GATE` empty) and writes it.
- `served_util.sh` (with `served_util.py`): the whole-step tensor utilization of serve-openai on the int8 Llama-3.2-1B
  prefill mix (prompts of 1,024 ids, `max_tokens` 1, all queued at once, T tokens a step), one arm a run (stock, the
  Commit unsealed or sealed, the Commit with PoUW H1 on in-process coins), under Nsight Systems' GPU metrics.
  `served_util.py` takes Tensor Active's mean over the longest run of full steps, each step from the previous step's end
  to its own on the step trace's clock, and splits the rest into losses by what the GPU was running at each sample.
  With `PYSPY=1`, `pyspy_steps.py` splits the same steps' host time by Python function and thread, from py-spy's trace.
- `served_shares.py SERVE_DIR [--credit ANSWER.json]`: a served run's declared utilization (the work its step trace's
  steps declare, the linears at θ and attention and the LM head at θ/2, over the loads' busy wall), term by term, and its
  credited share (the MACs Pearl+'s verdict, `served_passes.py verify`'s answer, credited at θ, over the same wall), at
  the tensor and boost clock readings.
- `served_pearlplus.sh SERVE_DIR WINDOW PIECE_DIR`: PoUW's piece with Pearl+ (`served_verdicts.sh`'s `POUW=pearlplus`).
  After the window's serve, it serves Llama-3.2-1B's BF16 checkpoint on the same card through `pouw:pearl-c-h100-v1-h3`
  (`benchmarks/pouw/pearl_c_vllm/served_passes.py`, a prefill-heavy load), retains every pass, gives up the passes' rows
  over a debit cap before the verifier's coin (`served_passes.py volunteer --device`: every word's counts from the card's
  debit cubin, every `volunteer_pass.CROSS_CHECK`-th tile replayed on the CPU too; `PEARLPLUS_VOLUNTEER=cpu` replays
  every tile on the CPU, far too slow for the default load), takes Pearl+'s verdict over all of them
  (`served_passes.py verify`), and writes the piece through `served_verdicts.py pearlplus`: the verdict, the
  linears' declared and credited shares (`served_shares.pearlplus_shares`), and why it credits nothing.

C-Flock's circuit prover (M0, `verity_flock.circuit`) is on `main`, but a run does not take it from its checkout:
`pod.sh` takes `git archive`s of M0's tree and the Lean verifier's tree at pinned commits, and a captured input set, as
inputs sent with the run; it builds `flock-circuit` and `flock-verify` once per pod, then runs the driver, whose
`--m0-pythonpath` names that M0 tree. The drivers' checks go through the verifier of record
(`one_stage.verifier`), which builds the sampled-proofs Lean in the run's tree on its first check, hence `--cwd clone`:

~~~sh
bash benchmarks/one_stage/pod.sh    # a job run from a clone of the tree, with m0.tar, lean.tar and set.tar sent
~~~

`check` runs nothing here.
