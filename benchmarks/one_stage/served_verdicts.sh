#!/usr/bin/env bash
# One window of Llama-3.2-1B int8 served on an H200 through every subprotocol, each step's own
# verdict line in $OUT/verdicts.txt, and one record of every verdict. A window's steps and their order are `verity run`'s
# stages' (`steps STEP...`); what each verb and follow-on below runs:
#  0. B: `verity-vllm predict row` writes the H200 int8 row's Build of record (ROW_ID, default ROW's name) from its declared
#     inputs alone into PRED ($D/predicted), before the lease and on the CPU: its request Programs, GP-01's workload and the
#     manifest, and beside them a copy of the checkpoint config it was predicted from (checkpoint_config.json, which the
#     served compare reads). `predict compare ROW --predicted PRED` holds it to the traced Build ROW, the cross-check: PREDICT-COMPARE,
#     every Program digest and the manifest's digest, Program digest and identities, gates the serve.
#  1. serve: `verity-vllm serve-openai` of LLAMA32_1B_INT8 (REPO at REV, --precision PRECISION: verity-int8/Llama-3.2-1B-INT8, verity_int8) on one GPU
#     under `gpu-lease`, `--scheme rows-sha512 --seal --admit release-order --window-requests SLOTS`, its committer built from
#     B's manifest (PRED/manifest.json) and tap (HIDDEN_SO), T tokens a step (--max-num-batched-tokens; T defaults
#     to 512); B-OF-RECORD says each sealed sampling block's record names B's manifest, and its
#     Program digest and manifest digest predict's (a record sealed before the seal recorded the manifest digest: the Program
#     digest alone). Each closed window is sealed as a
#     rows-sha512 row dir, and the seal opens each one's replay bundle (`deferred_rows.of_record`). Once it listens, before
#     the first request, the committer registers its weights (`prove_path.py weights-registration`: each weight parameter of
#     the checkpoint CKPT, REPO's snapshot at REV under HF_HOME, and the rotary table at the words the engine serves,
#     $D/private/cos_sin_cache.npy, once within the analytic table's bound, committed under fresh OS salts in the salt file
#     $D/weights_registration.salts/ (0700, `salt_file.beside`), neither of which a record copies); WEIGHTS says how many it roots and which it leaves unrooted, and
#     why, or NO WEIGHTS REGISTRATION why not, and then every weight read counts free. Under REGISTRATION_LANE the serve
#     then holds (`registration_hold`): it prints the deployment's registration record's digest and the period-open that
#     registers it, and sends no request until the auditor's ledger REGISTRATION_LEDGER carries that period-open for the
#     lane (a path, or gate:USER@HOST read through the gate's ledger verb under PP_GATE_KEY), checked under the
#     challenger's key record REGISTRATION_KEY (default PP_CHALLENGER_KEY), held to the SHA-512 REGISTRATION_KEY_SHA512
#     the operator posted where given, within
#     REGISTRATION_HOLD_S seconds, else stops refused; the audit is then held to that head (`wreg_args`). With no lane it
#     prints the development line and does not hold.
#  2. requests: REQUESTS greedy requests (default SLOTS) from the network attester's traffic generator (`adapters/vllm/tools/served_run/serve_load.py --varied`:
#     prompts of PROMPTS tokens, at most MAX_TOKENS outputs), numbered under one release run and sent together (RATE a second),
#     so the engine's idle closes them into one window. With LOAD=dataset (default varied) they are the input dataset
#     DATASET at TIMING instead, both of which LOAD=dataset needs, sent by the adapter's one sender
#     (`verity_vllm.inputs.sender`, by `adapters/vllm/tools/served_run/inputs.py send`), which reads it through the adapter's door (`verity_vllm.inputs.serves`) before it sends any. With
#     LOAD=front there are none: CERTIFIER_FRONT's group, sent through the network attester, is the run's one load and its execution
#     block the one the run audits (a period that admits a single block, say). A front that failed before a window
#     closed stops the endpoint at once, with no window to seal; one that failed after (its leak guard holding a file,
#     say) leaves its exit to the network attester's verdict, and the run goes on with the window it sealed.
#     With LOAD=session the load is the caller's own, through a relay session of the network attester's, and the run
#     sends nothing itself: CERTIFIER_FRONT, passing `--session` to `attester_piece.py front`, opens the session and
#     returns once its relay listens; LOAD_CMD, the caller's command, run as `bash -c LOAD_CMD _ RELAY D` in TREE within
#     LOAD_TIMEOUT (default 170m, the gate's longest session), sends the load to the client address RELAY names (the
#     session's relay.json, in the serve dir's private/), its output the serve dir's load.log; and CERTIFIER_CLOSE
#     (`attester_piece.py close "$1" "$2"`, given D and the front's directory), within CERTIFIER_CLOSE_TIMEOUT
#     (default 180m), waits for the session's end and keeps and publishes its fetch as a rehearsal's front does. The
#     CERTIFIER-FRONT line is then the close's, and the step's exit the load's; a front that opened no session runs no
#     load, and the close says why. LOAD=session without all three, and LOAD_CMD, CERTIFIER_CLOSE or a front passing
#     `--session` under any other LOAD, are refused before the lease.
#     CERTIFIER_FRONT runs within CERTIFIER_FRONT_TIMEOUT (default 20m), and a flag it passes to `attester_piece.py front`
#     that this tree's front doesn't take, or a front whose endpoint isn't on the network attester's id form (SERVE_ARGS without
#     `--protocol ids`), is refused before the lease. The
#     lease ends once every closed window is sealed. What names a window's requests (the serve dir's windows.jsonl,
#     seals.jsonl, requests.jsonl and serve.json, each window's PoUW calls record, the load's records, timings file and
#     log, and the network attester front's session, bundle and log, in the serve dir's private/, of which the CERTIFIER-FRONT
#     line says only `served_verdicts.py front-line`'s fields) stays in the serve dir on the node; what leaves it is their
#     aggregate ($OUT/serve/overhead.json, except with LOAD=front, whose network attester-paced load no stock
#     control compares to, and LOAD=session, whose relay-paced one none does either) less the load's counts and span, and verdict lines that name no window and count none of a
#     window's own: no window's number, no count of the windows closed, sealed or refused, no engine step, request count or
#     user of one. Each line goes out through `served_verdicts.py public-lines`, in full to the serve dir's
#     private/verdicts.txt, and once the run is done, however it ends, `served_verdicts.py publish` holds in full in the
#     serve dir's private/out each file of $OUT that names or counts a window or holds a load's tokens or span (the
#     aggregate's, which `served_overhead.py table` reads there) or Pearl+'s exclusion tiles' count or its replayed tiles'
#     ids (its verdict's profile and replay by tile, the reference's and the Lean's), its exact credit where the work is
#     in the grid form, or its prover's give-up before the coin (the piece's give-up document and the counts in its
#     log's volunteer and retain lines), its public form in its place. The advice registration's published copies
#     ($OUT/registration.json, the records audit's served-records) carry its compiler's cap in place of the advice's
#     length (`served_verdicts.py public-registration`), which hides nothing: the advice is committed at its own row
#     count. The audit's log and run.sh's served path (its summary, registrations, sessions and audit log, which carry
#     the advice's length) stay in the serve dir, where the record reads them.
#  3. SPOT, on the window `served_verdicts.py window` picks (the most requests, none carried across windows; under
#     PEARLPLUS_TIE the fixed load's, the one window holding its requests and no other, with CERTIFIER_FRONT's group in a
#     window of its own, the one whose requests are all the network attester's, which the network attester's piece takes
#     and the record's `not_covered` names, its case left out of the composite; the
#     pick stays in the serve dir's private/window.json, the WINDOW line saying only the class's SLOTS slots, and CLASSES
#     the window circuit's classes beside network attestation's scan class, CERTIFIER_SCAN, default W63, the one the gate serves): V3b's window circuit (`prove_path.py window-circuit
#     --pins` the docket, and PIN's NAME=ROOT if set), its registered and free input bits at those
#     pins: at each class of CLASSES (default 4,32,4, V2's), each within WC_TIMEOUT (default 30m), all of which the run needs
#     (the class of record is counted only where CLASSES names `record`, and gates like any other); then the window
#     audited from its records alone
#     (`records_audit.py audit`): Flock `--zk` on the sampled proof blocks behind the Lean
#     ZK server, and the Lean verifier's verdict. Those proof blocks are the sealed window's step units (PP_PARTITION, default
#     layer 0's SiluMul_v2 at step 1, or at step 0 with MAX_TOKENS 1, whose window has no step 1), not the window circuit's.
#     Every unit of a template the served model's compiler of record lists as a transfer gate (prove_path.GATES_OF_RECORD)
#     is proved beside the sampled rest, which only a work rate (work-rate:NUM/DEN) samples, so the preflight refuses a partition of another sampling design
#     or without them; PP_SETS names their input sets.
#  4. PoUW (Pearl+'s own served run, POUW=pearlplus, below), PoUS, the bandwidth audit and the network
#     attester: `served_verdicts.py pieces` runs each one's piece
#     where POUW_CMD, POUS_CMD, BANDWIDTH_CMD or CERTIFIER_CMD names it, one at a time (the network attester's alone, the longest,
#     from the moment the window is picked, beside step 3, its lines said after the others'; split in two, the answer step
#     6's period check wrote, V3c having run before this step), each within <COMPONENT>_TIMEOUT
#     (else PIECE_TIMEOUT, default 60m), else names it MISSING with the command it lacks and its owner. No verdict is made
#     for a piece that didn't run or refused to; one that ran and wrote no answer errs. A piece that needs a GPU takes its
#     own `gpu-lease`, the serve's being over. Under PRODUCT=onsite, WIPE_CMD's, POUS_CMD's and BANDWIDTH_CMD's defaults
#     come from `onsite_piece_commands` in the file ONSITE_PIECES names (a path from TREE, or absolute), sourced before
#     the preflight, each kept where its variable is set; with ONSITE_PIECES unset there are none, and a file that isn't
#     there is refused. The file also gives `served_composite.sh` onsite's own cases. Set empty, a command is MISSING.
#     Where POUS_CMD is the tree's piece, the preflight asks the piece itself (the file's `onsite_pous_pins`) whether it
#     pins REPO at REV, and refuses a checkpoint it doesn't pin, which the piece would refuse only after the serve.
#     A subprotocol that runs on the serving engine (`--protocols ...`) goes in SERVE_ARGS, serve-openai's
#     further arguments, and its piece reads what it wrote in the serve dir. The wipe's piece runs in step 1 instead: with
#     WIPE_CMD set (empty, none), serve() calls it in TREE, under the serve's lease and before serve-openai starts, within WIPE_TIMEOUT
#     (default 20m), with (the serve dir D, `all`, $OUT/pieces/wipe), since the window isn't picked yet, and says its last
#     line as WIPE. The record reads its pieces/wipe/answer.json and composes the wipe only when that answer accepts and
#     pieces/wipe/served.json names this node as the server it ran on. A wipe that wrote no answer (refused, or failed
#     before its judge) leaves the wipe missing, one served elsewhere stays missing, and one whose answer rejects makes the
#     record reject: a failed wipe is never accepted.
#  5. the record: `served_verdicts.py record`: the all-protocols window report ($OUT/record/verdicts.json) and the block
#     record ($OUT/record/block-record.json), `reject` where a piece about this node rejects or errs, and otherwise the
#     verdict of the product's compiled Lean check (onsite's, or remote's) on the run's cases: the answer `composite.py`
#     wrote to the composite's public output, which the record reads (--composite) where the composite ran before it,
#     and takes only where it is of record (under VERITY_HOLD_PLACEMENT=1 composed on an auditor device off the served
#     node, `compose`; the deferral's follow-on composes on a prover node, so its composite is then a development
#     check), and names the window's
#     docket R that the record's SPOT audit sampled against: one of another window, or bound to
#     no record, is refused, exit 1; any other server's answer is carried as elsewhere. Network attestation's chain files
#     that list or count the window's requests (CERTIFIER_PRIVATE) then move to the serve dir's private/pieces/certifier. Then, unless COMPOSITE is set empty, remote's and onsite's Lean
#     verdicts on the run's cases (`served_composite.sh`'s docstring): a composite that didn't run (a tree without
#     composite.py) or whose Lean didn't answer fails the run. On the served node its composite is of record, and the
#     record is composed again after it, reading it (--composite), its composed_on saying where it was made (the served
#     node, served_node true); under VERITY_HOLD_PLACEMENT=1 none is (under COMPOSE_OFF_NODE=0 it is a
#     development check), so this record is `pending`; under COMPOSE_OFF_NODE=1, then the default,
#     it says (its `deferred`) that the final record and the composite are composed off it, and in the composite's place
#     the run's last step, after V3c, writes the compose hand-off they are composed from (below, at `handoff`).
#     The run's exit is the record's verdict: 0 only where it is the compiled Lean check's accept, 2 where it is
#     `pending`, no composite of record yet (on the served node and in a deferral's follow-on under
#     VERITY_HOLD_PLACEMENT=1), and otherwise
#     nonzero: 1 where it is `reject`, a composite is refused or a step around it failed, the step's own code where one
#     stopped the run before it.
#     PRODUCT (default onsite) is the product the run composes: onsite every subprotocol; remote SPOT and PoUW
#     alone, its other pieces dropped (not run, no default command, the wipe's and the network attester's included; the
#     preflight refuses any onsite command set), its record composing their verdicts alone and its composite remote's.
#     Beside them, the inference-only verdict (`inference_only.py`), at IO_CLASS (default the first class of CLASSES but
#     `record`): its policy, (c, k, n) and every input to them, is written before the lease from the run's settings alone
#     (the class, REPO at REV, LEASE_MIN; under PEARLPLUS_TIE in serve(), under the lease before serve-openai starts, from
#     PEARLPLUS-CIRCUIT's statement of φ on the card, serve-phi.json, its circuit cut at Pearl+'s linears, as a BF16
#     checkpoint's needs), $OUT/inference-only-policy.json and its INFERENCE-ONLY POLICY line; once the
#     build is done, step 3's window circuit record at the class is judged beside SPOT by the Lean verifier,
#     `inference-only-check --count`, where the build built it (and refused as undecided where it didn't), into
#     $OUT/record/inference-only.json (the judgment against n and no count, n judged on the exact count, the record's
#     count beside it) and its INFERENCE-ONLY line, in full with the exact counts in the serve dir's
#     private/inference-only.json. The record carries it as its inference-only row, which the combined verdict doesn't read.
#  6. V3c, after the record since V2 doesn't read it: it proves the window circuit's own units at V3C_CLASS (default 4,32,4,
#     which CLASSES must hold). `prove_path.py window-commit`, within V3C_TIMEOUT (default 15m), commits the window as that
#     class circuit's units, the weights it serves under the committer's key, so each served as registered has the
#     docket's root; its circuit.json less its run fields is held to the verifier's own window-circuit record less its
#     run fields at the class; then `records_audit.py audit --circuit --registered`, within V3C_AUDIT_TIMEOUT (default 30m), samples the
#     docket's members by its rule V3C_LAW, a rate (records_audit.py refuses a fixed count): by default the rate that
#     samples one proof block of class 4,32,4's window, under PEARLPLUS_TIE one of each template's, on V3C_CHALLENGER's coins
#     (below, Coins; pinned on main's first-parent line: MAIN_FILE's, kept as $OUT/v3c-audit/main.txt, else MAIN_SHA's)
#     and proves them, its composes made before it with V3C_WARM_JOBS (at `v3c_audit`), its sessions one at a time and
#     Lean's checks of finished members beside them, V3C_LEAN_WORKERS at a time, default 1 on every path, each started
#     only while its memory estimate fits: on a node it shares, with the running session's estimate
#     (`a5.SESSION_BYTES_PER_CIRCUIT_BYTE`) within V3C_AUDIT_MEM_GB; in a follow-on, which passes the flag only where
#     V3C_LEAN_WORKERS is set and else leaves a5's own default (`a5.LEAN_WORKERS`, 1), with that estimate within the
#     V3C_AUDIT_MEM_GB its command sets, else what its node has free (`a5.LeanPool`). The estimates admit a start
#     and limit nothing that runs: no limit is set on a check, and one over the budget runs alone. What the audit reads as free (`a5.available_memory`) is the least of two: its
#     node's MemAvailable, and under each cgroup memory limit over it (the lesser of memory.high and memory.max) the
#     limit less what the cgroup holds that the kernel can't give back at once. A cgroup's file cache (memory.stat's
#     inactive_file and active_file, less file_dirty, file_writeback and file_mapped) is counted free, as MemAvailable
#     counts the node's; its anonymous memory and its shared memory are not. So a follow-on inside a memory scope is
#     held to the scope's room, read low rather than high. A set V3C_AUDIT_MEM_GB is the audit's memory in place of
#     that reading; unset, the reading is taken as the sessions start and again as the negatives' workers and the
#     registered reads' pool are made. Each check and worker still starts only while its estimate is free by it.
#     Each pool prints the two terms and which binds as it starts, and once what held a start back. Its negatives
#     after the audit record run one after another, or with
#     V3C_NEGATIVE_WORKERS that many at once, each in a worker process with its own Lean verifier process, started only
#     while its memory estimate, read then, fits that memory (`a5.NegativeWorkers`), and with V3C_ARCHIVE_LINK=1 each member's circuit file linked into the audit's
#     archive rather than copied (records_audit.py's --archive-link); then
#     `records_audit.py period` writes the audit as the network attester's period reads it ($D/v3c-w<K>-<class>/period.json,
#     and a copy in $OUT/v3c-audit: the docket, the Program, its answer and its free inputs), and V3c fails where it
#     writes none. V3c then runs
#     after step 4 under PEARLPLUS_TIE, else after step 5's record, unless this node has less
#     memory than the audit needs (below: the audit and the composite are then deferred to a follow-on).
#     `served_verdicts.sh v3c` runs only that step on an
#     existing serve dir (D and its window K set), the build first, its limits 120m and 180m; it serves nothing, so it needs
#     no row dir, taps or gpu-lease (VENV the tree's .venv on a node without vLLM's). Untied, window-commit registers the
#     window at V3C_LAW's sampling design, the one its audit samples by (R1 holds the sample to it). V3C_CLASS set empty leaves
#     V3c out of the run, for a `served_verdicts.sh v3c` started once the window's advice is written.
#     With the network attester's piece split in two, CERTIFIER_CHAIN_CMD and CERTIFIER_PERIOD_CMD (CERTIFIER_CMD then
#     unset, and V3C_CLASS network attestation's scan class, which CLASSES holds), V3c runs before step 4, since the network attester's
#     answer reads its period.json: the chain after the window commit, beside V3c's audit and period.json, within
#     CERTIFIER_TIMEOUT (default 4h), and the period check once both are done and V3c wrote period.json, within
#     CERTIFIER_PERIOD_TIMEOUT (default 60m), each as `bash -c CMD _ D K DIR V3C` in TREE with DIR $OUT/pieces/certifier
#     and V3C V3c's $D/v3c-w<K>-<class>. V3c then takes the window commit, the longer of the chain and V3c's audit,
#     and the period check, which the run's --timeout holds beside the other steps. The period check writes the
#     network attester's answer.json in DIR, which step 4's attester piece reads; `served_verdicts.sh v3c`, which writes no
#     record, moves the steps' logs (CERTIFIER_PRIVATE) to the serve dir's private/pieces/certifier before it exits.
# Composed off the served node: with COMPOSE_OFF_NODE=1 (the default under VERITY_HOLD_PLACEMENT=1, else 0, which
# composes both on the served node, the composite a development check that claims no record only under
# VERITY_HOLD_PLACEMENT=1, `served_composite.sh`, which then refuses one there under 1) the run's
# last step (`handoff`, after V3c in every run, so the composite reads V3c's audit's cases where it ran, the class window
# of record, and step 3's audit's only in a run without V3c) writes the serve dir's private/compose: the
# record's pieces (only the files the record reads of each, Pearl+'s verdict of record's among them), the inference-only
# verdict, the settings the follow-on composes at (compose.env: the window and its engine steps, the product, POUW, COMPOSITE,
# ALPHA, GAMMA, the served node, the serve dir's path, COMPOSE_NODE, the tree's commit and the host V3c's audit ran on)
# and its every input relative to the serve dir
# (`handoff_inputs`: verifier-side files alone, the records audits' cases and the blobs they name,
# the serve's first registration record and each window's Pearl+ record W and audit's record R, PoUW's period case,
# SPOT's summary and docket), listed (inputs.txt) and hashed (inputs.sha256). A list
# holding a served run's witness (`served_verdicts.py witness`: a value file, a prover's instance, private or salts file,
# the advice, a key, a row dir, the requests) or a blob outside the serve dir is refused, and the run fails. Those files alone
# are copied to the same path on COMPOSE_NODE, an auditor device the prover doesn't control and where no witness goes
# (default vy-auditor-2; never a prover node such as a deferral's follow-on), outside any run dir (`served_verdicts.sh
# copy compose`), and there `served_verdicts.sh compose` with D alone reads compose.env as data (`read_env`, which refuses
# a line of any other name or character and runs nothing of it; `handoff` holds what it writes to the same), holds the
# copy to its sums, rebases the cases'
# blobs onto a copy at another path (`served_verdicts.py rebase`), runs at the served node's tree (another only with FOLLOW_ANY_TREE=1), refuses the served node itself and a serve dir
# holding any witness, needs no row dir, vLLM venv, card or gpu-lease, and composes the composite (`served_composite.sh`,
# SERVED_NODE the served node, COMPOSE_HANDOFF set) and then the final record, whose verdict is that composite's Lean
# answer (`record --off-node --steps --sampled-lean --composite`: composed_on the host it ran on, served_node false, from
# the hand-off, SPOT's row this host's own run of its Lean verifier on the case's proofs, each other row's answer
# as the served node wrote it, and each row naming which in its answer_from); its exit is the record's verdict (step 5),
# and its $OUT is published as every run's is. PRODUCT=onsite's composite
# reads PoUS's and the wipe's cases, which only the served node's card writes, so under COMPOSE_OFF_NODE=1 the preflight
# refuses it unless COMPOSITE is set empty. `served_verdicts.sh handoff [HOST]`, in TREE with D, writes the hand-off
# alone: on a deferral's serve dir from its pieces and inference-only verdict at its settings (deferred.env), HOST the
# follow-on's, where V3c's audit ran, which a serve dir holding that audit must name; else, with K and OUT, from OUT's
# pieces and its record's inference-only verdict.
# `served_verdicts.sh copy deferred|compose HOST`, with D, on the node that holds the list, sends each file a deferral's
# or a hand-off's inputs.txt lists, and nothing else, to the same path on HOST (RSYNC, default rsync over ssh), its log
# in the serve dir's private/: the deferral's (prover-side, its window's witness) to a prover node, never COMPOSE_NODE;
# the hand-off's only once `served_verdicts.py witness` passes its list again. `HOST:DIR` (DIR may be empty) sends them
# under DIR/<the serve dir's name>/ in the directory HOST's rsync writes to, a write-only rsync's (`rrsync -wo`, its key
# in RSYNC_RSH), which makes no directory the path needs; HOST's node then moves that directory to the serve dir's path.
# `served_verdicts.sh gate-compose PIECES [VERDICT [HOST]]`, with D and OUT, on the node that holds the serve dir, writes
# the hand-off from PIECES and the inference-only VERDICT (on a deferral's serve dir, at its settings, HOST as
# `handoff`'s) and sends it to the compose gate on COMPOSE_NODE
# (`verity/protocols/accounting/memory/poub/attester/live_verifier/host.py`'s `compose`: COMPOSE_GATE, default lane vllm's gate key as research@
# the address VY_AUDITOR_2_HOST holds, vy-auditor-2's public one, where the gate admits senders (set it for any other COMPOSE_NODE;
# with neither, the preflight refuses), the one word `compose`), a tar of the files its inputs.txt lists on the gate's stdin; the gate composes there at its
# pinned tree (COMPOSE_BUILT: built at its pin; SAMPLED_EXE its prebuilt SPOT Lean; COMPOSE_TREE_ON the branch
# infra pinned it on, which the record's composed_on names beside the commit) and answers its $OUT, written to
# $OUT/compose with the gate's pin in gate.json, one GATE-COMPOSE line saying what it composed.
# Coins: V3c's audit, the SPOT audit of record, samples on V3C_CHALLENGER's coins, by default PP_CHALLENGER's: the
# live challenger with PP_CHALLENGER=gate:USER@HOST (records_audit's, with its peer V3C_PEER, default PP_PEER, the node
# it runs on, PP_CHALLENGER_KEY and PP_GATE_KEY), where it runs, on the served node or a deferral's follow-on. There
# `served_verdicts.py gate --root` first finds no session on the window's run root (the gate opens a root once, ever) and
# no period of lane H1_LANE open (a reserve would join it) but, under REGISTRATION_LANE, the registering one with $DREG's
# digest, which the epoch's reserves join, its V3C-GATE line saying so and when. A local or loopback
# session is a development session, which the preflight refuses unless DEVELOPMENT=1 (a development run), as it refuses a
# V3C_PEER other than the node's peer Pearl+'s session names, and in a follow-on the served node's. SPOT's step 3 samples on a local development challenger session's coins on the served node (run.sh's PP_CHALLENGER
# default, local, pinned on PP_MAIN's line, else MAIN_FILE's, else origin's main's) unless PP_CHALLENGER names the live
# challenger (run.sh's gate:USER@HOST with PP_PEER, PP_CHALLENGER_KEY and PP_GATE_KEY), and then too while V3c's audit is
# live, since both open on the window's run root: step 3 is then a development check. With V3c off (V3C_CLASS empty) step
# 3's audit is the one of record, and the preflight refuses its local or loopback session unless DEVELOPMENT=1. Step 3's
# SAMPLED-PROOFS line says whose coins they were, the gate's or a development session's, and the audit's coin_trees; on
# the gate's coins, an audit whose summary says its sessions took their coin trees from anywhere but the session that
# sampled them (coin_trees other than window, `served_verdicts.py coin-trees`), which the auditor's Lean verifier refuses,
# stops the run there (`sampled_coins`, exit 20; `served_verdicts.sh sampled-coins` runs that check alone on D). V3c's
# audit is held to the same on V3C_CHALLENGER's coins: its V3C-AUDIT line says whose coins and its coin_trees, and on the
# gate's it fails, exit 20 with the same refusal, as a V3c audit that fails does (`v3c_trees`; `served_verdicts.sh
# v3c-coins K` alone on D at V3C_CLASS), and the compose hand-off of a record whose SPOT audit it is refuses the same
# way (`handoff`), so no compose is asked of it. A record
# composed after V3c (a follow-on's, or the compose's from its hand-off) reads V3c's audit where it wrote its summary,
# else step 3's (`sampled_of_record`).
# DAY=N (off by default: unset, serve-openai writes what it wrote before) declares day N's one circuit in DAY_LOG, the
# deployment's registration log, which holds one compiler for its life, before the model loads (serve-openai's --day,
# `serve_day`): DAY_PROGRAM, the SHA-512 of the program the day's windows register, DAY_BLOCKS its block Definitions, and
# DAY_PARAMS the log's {auditor, composite} (needed for a new log). A second declaration of the day, to another circuit, is
# refused by the log's check (duplicate-declaration) and the serve doesn't start; the same one again (a rerun within the
# day) is the day's.
# Each window's W names the declaration, an R of another program is refused at its close, and V3c's audit runs R2 against the
# log's declarations (records_audit's --day). A deferral carries the day: DAY
# in its settings, and DAY_LOG and DAY_PARAMS as the serve left them, copied to private/deferred/day-log.json and
# day-params.json and hashed with the serve's declaration.json among its inputs. Its follow-on audits against that copy
# alone, once the sums hold and the copy holds the declaration declaration.json names by digest (DAY-HELD).
# On a node with less memory than V3c's audit needs (its MemTotal, NODE_MEM_GB, under V3C_AUDIT_MEM_GB, default
# 290), step 6 holds the tie (WINDOW-COMMIT, WINDOW-CIRCUIT-HELD) and defers the audit (and, under COMPOSE_OFF_NODE=0, the
# composite; `defer`), for PRODUCT=remote alone (onsite's composite reads PoUS's case, which needs this node's card: the
# preflight refuses it).
# Step 5's record says what was deferred and why (its `deferred`), a DEFERRED line says so, and the serve dir's
# private/deferred gets the pieces the record read, the inference-only verdict (which the follow-on's record carries as
# the served node wrote it), the settings the follow-on runs at (deferred.env: the window, the
# product, V3c's class, sampling design and coins (V3C_CHALLENGER, DEVELOPMENT), COMPOSITE, ALPHA, GAMMA, the served node, its
# peer, the tree's commit, COMPOSE_OFF_NODE and COMPOSE_NODE; a deferral that names no COMPOSE_OFF_NODE follows on at 0),
# and the
# follow-on's every input relative to the serve dir (`deferred_inputs`), listed (inputs.txt, which names inputs.sha256
# too) and hashed (inputs.sha256). The copy is the prover's: those files alone, to the same path on a node with the
# memory, outside any run dir. There `served_verdicts.sh v3c` with D alone (deferred.env names the window, and an
# environment that sets one of its settings to another value is refused) holds the copy to its sums, runs at the served
# node's tree (another only with FOLLOW_ANY_TREE=1, which its WINDOW-CIRCUIT line names, since it reads that tree's
# window-circuit record), needs no row dir, vLLM venv, card or gpu-lease, and after V3c's audit writes the compose hand-off
# there (above; a compose follow-on refuses a serve dir holding the copy's value files, so COMPOSE_NODE is then another
# node), its exit then 2 (pending: the record is composed off it), or under COMPOSE_OFF_NODE=0 composes the composite
# (`served_composite.sh`, SERVED_NODE the served node) and then the final record (`record --node` the served node
# `--composite`, so its composed_on names the host it composed on, served_node false, and that it read a prover-side
# copy). That node is a prover node, which the prover controls, which its composite's COMPOSITE ON-NODE line says;
# under VERITY_HOLD_PLACEMENT=1 its composite is a development check (COMPOSE_DEVELOPMENT), never the record's verdict:
# the record stays pending and the exit is 2 at best (step 5); only the compose follow-on's composite, on an auditor
# device, is of record. Its $OUT is published as every run's is.
# POUW=pearlplus (required: PoUW H1 is deleted, so POUW=h1 and the empty POUW are refused, as are H1's settings) makes Pearl+
# PoUW's piece: POUW_CMD defaults to 'bash benchmarks/one_stage/served_pearlplus.sh "$1" "$2" "$3"' within POUW_TIMEOUT 120m, which
# once the window is audited serves Pearl+ (pouw:pearl-c-h100-v1-h3, Llama 3.2 1B's BF16 checkpoint) on this card under
# its own lease, every pass retained in the serve dir's private/, and takes Pearl+'s verdict over them (verify.json in
# the piece's dir; first, by default, it gives up the passes' rows over a debit cap from the card's counts, V-EX): a development
# piece, since its passes are not the window SPOT audited (its header has the rest). Its epoch's salt coin is
# a live session's of its own on vy-challenger-1: PEARLC_CHALLENGER_GATE defaults to the lane's gate command,
# PEARLC_CHALLENGER_PEER to H1_PEER where there is a gate and PEARLC_CHALLENGER_LANE to H1_LANE's, `served_verdicts.py gate`
# must find no period of that lane open, and the piece holds each pass's salt to the session under
# H1_KEY_SHA512 and MAIN_FILE's line. The piece opens the lane's Pearl-C period right before its
# retained serve and closes it right before its verifier's sample, a sample session's of its own on that gate, reserved as
# the lane's first reserve after the close and opened on the run's commitment (served_pearlplus.sh's steps 2 and 4);
# while the period is open its cap refuses every other reserve of the lane. PEARLC_CHALLENGER_GATE set
# empty leaves Pearl+'s coins an
# in-process challenger's: the COINS line, the run's first, leads with that, and the piece claims no verdict. A peer
# set with the gate empty is a pair the preflight refuses.
# With PEARLPLUS_TIE=1 too, the window itself is served under Pearl+: serve-openai serves
# pearl-plus by its name on its one engine (`serve_pearl`, --protocols with --protocols-config naming the cubin
# PEARLPLUS-CUBIN built before the lease), one Pearl+ epoch a Commit window beside SPOT cut at Pearl+'s linears
# (--pearl-tables, the kernel of PEARLPLUS_ARM: default Pearl+'s declared option, as records hold it), so the window's linears are the Pearl+ matmuls whose passes are committed, each window's retained in
# the serve dir's pearl/w<k>/. Each epoch's salt coin is a live salt-only session of its window's own on vy-challenger-1
# (`serve_pearl.live`: serve-openai's --challenger-gate PEARLC_CHALLENGER_GATE, --challenger-peer PEARLC_CHALLENGER_PEER
# and --challenger-lane PEARLC_CHALLENGER_LANE, defaulted as above), reserved under the lane as the window opens and
# joining its open Pearl-C period (with CERTIFIER_FRONT, on the fixed load's windows alone: Pearl+'s `users`, so the
# front's windows reserve nothing in the period's cap of 1); PEARLC_CHALLENGER_GATE set empty leaves them in-process, and the piece claims no
# verdict. Its defaults are then Llama 3.2 1B's
# BF16 checkpoint (REPO unsloth/Llama-3.2-1B, PRECISION bfloat16, the arm's), the BF16 B4 row's traced Build under
# /workspace/research/vllm-pearlc-h200bf16 as ROW, LOAD=dataset at V3C_CLASS (SLOTS its slots, WINDOW_STEPS its
# outputs: round s of W is the window's s-th step, the class's s-th output, and the cut circuit
# reads Pearl+'s tables at those rows; the preflight refuses another) and PEARLC_BF16_ROWS=0; V3C_LAW has none (the launch
# states it, a stratified rate over the cut window's templates, and the preflight refuses a launch without one), and
# step 3's partition layer 0's RoPE heads, the BF16 1B's transfer gates of record, every one proved, beside its
# SiluMul_v2 under work-rate:2/164 at step 1 (llama32_1b_tg_l0_step1.json, PP_SETS rope-head/d64/bf16). The preflight refuses
# another PRECISION or row (its id names __bf16__), a V3C_CLASS CLASSES doesn't count, vLLM's prefix cache (SERVE_ARGS'
# --prefix-caching: a hit leaves rows of a prompt uncomputed where the cut circuit reads Pearl+'s tables), a
# network attester front (its group closes into a window of its own), and a serve dir whose filesystem has fewer than
# PEARLPLUS_INODES inodes free (default 772388, two commits of the tied window at the class of record, three files a
# member; 0 checks none), naming the shortfall, the tied windows the serve root's other serve dirs keep that hold it
# (the largest first, with their entries) and the command that removes one (`research retention rm`). Step
# 3's window circuit at each class is the one cut at Pearl+'s linears (`window-circuit --pearl-tables`), the verifier's
# own record of the circuit the window is committed as. The same record at V3C_CLASS is built before serve-openai starts,
# under the lease, from what its serve.json will record of φ on this card (`prove_path.py serve-phi`, then `class-circuit`,
# no advice: the circuit is the class's whatever the traffic), into the serve dir's pearlplus/circuit-record.json
# (PEARLPLUS-CIRCUIT, within PEARLPLUS_CIRCUIT_TIMEOUT, default 30m), and step 3 says whether its record is that one.
# With a gate, serve() then opens the lane's Pearl-C period PEARLPLUS_PERIOD (default pearlplus-RUN_ID) on that record
# (`served_verdicts.py pearlplus-period open --circuit-record`, cap 1: the window's one salt session joins it; the
# PEARLPLUS-PERIOD lines) before serve-openai starts, and closes it once the windows are sealed and the endpoint stopped
# (or when it never listens, or is stopped), so before Pearl+'s piece samples: open, serve, seal, close, sample. The close
# first reads the ledger, waiting up to PEARLPLUS_CLOSE_WAIT seconds (default 300) for it to end each window's salt
# session, since the gate closes a period only once its windows have ended: the lane's reserves since the open must be
# the windows' salt sessions alone (pearl/w<k>/served.json's), its line saying held or REFUSED, and the piece samples only
# after a period that closed with its reserves held; once the coin is flipped, its RULE line says whether the sample session
# was the lane's first reserve after the close. The record
# reads the period's circuit against the window-circuit record V3c held R's circuit to (`window_tie` keeps it beside R).
# Pearl+'s piece then serves
# nothing of its own (served_pearlplus.sh's PEARLPLUS_TIE): it takes the window's epoch pearl/w<k>/, writes W
# (`retain --window-record`) chained after the serve's first registration record (`--previous-record`: the serve dir's
# deployment-registration.json, the weights registration as a `verity/registration/v2` record, which weights() writes
# before the first request and the WEIGHTS and DEPLOYMENT-REGISTRATION lines name: the developer's own record, whose
# digest no auditor holds), so the log before R is that record and W; then `verify --tie W --class
# V3C_CLASS` (PEARLPLUS_CLASS), which refuses passes that aren't the class's fixed load and publishes the work in its
# grid form (the class's structural credit less the give-up on the window grid, the exact credit held). V3c then runs
# after step 4 and before the record (on a node short of its audit's memory, deferred as under any tie): it audits that
# commit, held to W (WINDOW-COMMIT), with records_audit's --tie W (V3C-TIE), and the record reads R and that audit's
# summary from pearlplus/w<k>/tie (`record --window-tie`), where `pearlplus_of_record` holds them to W.
# Once Pearl+'s piece wrote W, before the record, the served compare holds the serve to the workload of record on the CPU
# (`served_compare.py`, within SERVED_COMPARE_TIMEOUT, default 30m): PHI, serve.json's φ is ROW_ID's declared φ at REV
# (`--revision`, which the manifests must pin of the row's model when it isn't the row's own) and lowers
# the same window circuit at V3C's class cut at Pearl+'s linears; BUILD, PRED is the row's Build of record BUILD_OF_RECORD
# (PROGRAM_DIGEST:MANIFEST_DIGEST); CUT, W registers exactly that circuit's Pearl+ values. Its last line, SERVED-COMPARE:
# EQUAL or NOT EQUAL, is the one the record names (`record --served-compare`, its served-compare/served_compare.json
# carried to a follow-on and the compose hand-off), and NOT EQUAL fails the run.
# A step that can't run says why, and the chain stops there. Flock, Lean's verity-verify-proof and spot-verifier and the
# tree's venv are built while the endpoint serves, Flock in FLOCK_WORK (kept on the node, so a rerun reuses its build).
#   research run --on vy-dc-h200 --project verity --source TREE --cwd clone --timeout 6h -- \
#     bash benchmarks/one_stage/served_verdicts.sh
# SETTINGS=PATH names a file of NAME=VALUE lines on the node, read before every default below: the run's job record
# publishes its launch command, so a load's counts (REQUESTS, PROMPTS, SLOTS, CERTIFIER_FRONT's) go in the file, and the
# command names only its path.
# PREFLIGHT_ONLY=1 runs the preflight alone (the settings, the row dir and taps, gpu-lease, the free inodes under
# PEARLPLUS_TIE, the front's flags, the checkpoint's pin in PoUS's piece and a follow-on's checks), says what the run would do (PREFLIGHT passed: its lease, or its deferral) and exits 0 before the
# serve dir is made, kept or anything built.
# `served_verdicts.sh circuits K`, in TREE with D and OUT, counts serve dir D's window K's circuits alone (its advice written).
# `served_verdicts.sh inference-policy`, in TREE with D and OUT, writes the inference-only policy at IO_CLASS alone, to
# IO_POLICY (default $OUT/inference-only-policy.json), before a serve, and with it the limits of its rate rule, to
# IO_LIMITS (default $OUT/inference-only-limits.json: `limits`); `served_verdicts.sh inference-only K` judges window
# K's circuit record at IO_CLASS alone against it.
# `served_verdicts.sh block-counts K`, in TREE with D and OUT, judges window K's circuit records at CLASSES against the
# policy alone (`block_counts`): the judgment public, the exact counts in D's private/.
# `served_verdicts.sh defer WHY`, in TREE with D, K and OUT, makes step 6's deferral alone.
# `served_verdicts.sh served-compare K`, in TREE with D and OUT, runs the served compare alone on serve dir D's window K.
# Defaults are vy-dc-h200's: the int8 B4 row's traced Build (ROW, traced after the required manifest stopped
# naming the FlashAttention M1 stream, as `predict row` doesn't) and taps under $H, vLLM's venv /workspace/cp/venv312, CUDA 13.0.
# The serve dir D (row dirs, advice, the window circuits' records) stays on the node, under KEEP_CMD where it is set (run as
# `bash -c KEEP_CMD _ ROOT` in TREE, before anything is built: the node's retention of ROOT, SERVE_ROOT's default
# /workspace/research/vllm-served-verdicts); $OUT keeps the public records and every log, each in its public form.
# V3C_COMPOSE_CACHE=DIR, with V3C_WARM_JOBS, keeps step 6's composed circuits in DIR in place of the window's own
# directory: a kept compose cache, which outlives the window and its serve dir, so that a later window's warm and audit,
# under the same tree and FLOCK_* environment, take each circuit they would compose the same and compose only the rest
# (at `v3c_audit`). It holds every such template's circuit, one of which can be past 100 GB, for as long as it is kept:
# no step here removes it or checks its room, and the run that names it finds the disk for it.
set -uo pipefail
if [ "${1:-}" = tie ]; then
  echo "served_verdicts.sh tie: refused: PoUW H1 is deleted, so its window commit at its close is" >&2
  exit 2
fi
if [ -n "${SETTINGS:-}" ]; then
  [ -f "$SETTINGS" ] && [ -r "$SETTINGS" ] || { echo "PREFLIGHT: SETTINGS is not a readable file" >&2; exit 3; }
  set -a
  # shellcheck source=/dev/null
  . "$SETTINGS"
  set +a
fi
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
TREE=$PWD
DEPLOY_MANIFEST=$TREE/tools/research/src/research/pods/nebius/deploy.toml  # infra's, where research reads the auditor devices
ROOT=${SERVE_ROOT:-/workspace/research/vllm-served-verdicts}
D=${D:-$ROOT/${RESEARCH_RUN_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)}}
# the serve dir holds the salt files and private/, which no run record or upload may carry
for under in "$OUT" ${RESEARCH_RUN_DIR:+"$RESEARCH_RUN_DIR"}; do
  case "$(realpath -m "$D")/" in "$(realpath -m "$under")"/*)
    echo "PREFLIGHT: D is under OUT or the run's directory, which a run uploads: set SERVE_ROOT or D outside it" >&2; exit 3;; esac
done
# the run's tool logs, builds and the node's facts: paths, hosts and free text, which no run record or upload may carry
LOGS=$D/private/logs
# step 3's records audit, relative to D: whole in the path prover's keep (PP_KEEP), since it holds the window's
# registration, its case, the advice's length and what is counted from it; never under OUT
SERVED_ARM=private/logs/sampled-proofs/keep/served
logs_dir() { (umask 077; mkdir -p "$LOGS/v3c-audit"); }
[ ! -d "$D" ] || logs_dir  # the serve dir itself is made after the preflight
export VERITY_PROTOCOLS_CONFIG=${VERITY_PROTOCOLS_CONFIG:-"{\"private\": \"$D/private\"}"}  # where a replay bundle's keys and salts go
# step 6's deferral (`defer`): the settings the follow-on runs at, its inputs' list and sums, and the pieces the record read
DEFER_DIR=$D/private/deferred
FOLLOWS="K PRODUCT V3C_CLASS V3C_LAW V3C_CHALLENGER DEVELOPMENT PEARLPLUS_TIE PEARLPLUS_TIE_UNIFORM COMPOSITE ALPHA GAMMA SERVED_NODE SERVED_PEER DEFERRED_TREE COMPOSE_OFF_NODE COMPOSE_NODE DAY"
# the compose hand-off (`handoff`): the settings its follow-on composes at, its inputs' list and sums, and the pieces
COMPOSE_DIR=$D/private/compose
COMPOSES="K PRODUCT POUW PEARLPLUS_TIE COMPOSITE ALPHA GAMMA SERVED_NODE SERVED_DIR COMPOSE_NODE COMPOSE_TREE AUDITED_ON SAMPLED_AUDIT STEPS"
FOLLOW= COMPOSE= CLASH= ENV_REFUSED=
# A compose hand-off's settings are the prover's bytes on an auditor device, so they are read as data and never run: each
# line NAME=VALUE, NAME one of the hand-off's and set once, VALUE as handoff's printf %q writes these characters (each as
# itself but a comma, which it writes \, and an empty value, which it writes '').
ENV_VALUE='^([A-Za-z0-9._/:+@=,-]|\\,)*$'
read_env() {  # $1's settings, those named in $2, set and exported; else none is, and ENV_REFUSED names the line and why
  local _l _n _v _i=0
  local -A _got=()
  ENV_REFUSED=
  while IFS= read -r _l || [ -n "$_l" ]; do
    _i=$((_i + 1)) _n=${_l%%=*} _v=${_l#*=}
    if [[ $_l != *=* ]]; then ENV_REFUSED="its line $_i is not NAME=VALUE"
    elif [[ ! $_n =~ ^[A-Z][A-Z0-9_]*$ ]]; then ENV_REFUSED="its line $_i names no setting"
    elif [[ " $2 " != *" $_n "* ]]; then ENV_REFUSED="its line $_i sets $_n, which is no setting of a hand-off"
    elif [ -n "${_got[$_n]+x}" ]; then ENV_REFUSED="its line $_i sets $_n a second time"
    elif [ "$_v" != "''" ] && [[ ! $_v =~ $ENV_VALUE ]]; then
      ENV_REFUSED="its line $_i sets $_n to a value of other characters than A-Za-z0-9._/:+@=,-"
    fi
    [ -z "$ENV_REFUSED" ] || return 1
    [ "$_v" != "''" ] || _v=
    _got[$_n]=${_v//\\,/,}
  done < "$1"
  for _n in "${!_got[@]}"; do printf -v "$_n" %s "${_got[$_n]}" && export "${_n?}"; done
}
follow_env() {  # $1's settings, those named in $2, read over the environment's (with $3 `data`, by read_env, else sourced),
  # CLASH naming each the environment set otherwise
  declare -A given=()
  for v in $2; do [ -z "${!v+x}" ] || given[$v]=${!v}; done
  if [ "${3:-}" = data ]; then
    read_env "$1" "$2"
  else
    set -a
    # shellcheck source=/dev/null
    . "$1"
    set +a
  fi
  for v in $2; do [ -z "${given[$v]+x}" ] || [ "${given[$v]}" = "${!v-}" ] || CLASH+=" $v"; done
  VENV=${VENV:-$TREE/.venv}
}
if [[ ${1:-} =~ ^(v3c|handoff|gate-compose)$ ]] && [ -f "$DEFER_DIR/deferred.env" ]; then  # the follow-on of a deferral,
  # or its compose hand-off written again (and sent to the compose gate), at the settings it recorded
  FOLLOW=1
  follow_env "$DEFER_DIR/deferred.env" "$FOLLOWS"
  COMPOSE_OFF_NODE=${COMPOSE_OFF_NODE-0}  # a deferral that names no COMPOSE_OFF_NODE predates it: it composes as it did
  if [ -n "${DAY:-}" ]; then  # the day's log and parameters as the deferral copied them, never this node's
    DAY_LOG=$DEFER_DIR/day-log.json DAY_PARAMS=
    [ ! -f "$DEFER_DIR/day-params.json" ] || DAY_PARAMS=$DEFER_DIR/day-params.json
  fi
elif [ "${1:-}" = compose ]; then  # the compose follow-on of a hand-off, at the settings it recorded
  COMPOSE=1
  [ ! -f "$COMPOSE_DIR/compose.env" ] || follow_env "$COMPOSE_DIR/compose.env" "$COMPOSES" data
fi
PEARLPLUS_TIE=${PEARLPLUS_TIE:-}
if [ -n "$PEARLPLUS_TIE" ]; then  # the window served under Pearl+: Llama 3.2 1B's BF16 checkpoint, whose linears Pearl-C reads
  HB=/workspace/research/vllm-pearlc-h200bf16
  ROW=${ROW:-$HB/r20261104-084130-0f59/llama32-1b__bf16__h200__tp1__b4__i256__o32__mixed__greedy__bi-eager}
  REPO=${REPO:-unsloth/Llama-3.2-1B} REV=${REV:-9535bd9b1d1dea6acafbdc4813b728796aeb28da} PRECISION=${PRECISION:-bfloat16}
  LOAD=${LOAD:-dataset} PEARLPLUS_ARM=${PEARLPLUS_ARM:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && python3 -m verity.protocols.run.schemes pearl-plus option)} || { echo "PREFLIGHT: PEARLPLUS_ARM unset and the declared schemes name no Pearl+ option (python3 -m verity.protocols.run.schemes pearl-plus option, above)" >&2; exit 3; }
  # the class's fixed load, at whose rows the circuit cut at Pearl+'s linears reads its tables
  IFS=, read -r slots _ <<< "${V3C_CLASS-4,32,4}"
  SLOTS=${SLOTS:-$slots}
  # sm_90's BF16-rows path off unless PEARLC_BF16_ROWS says otherwise (e2e.BF16_RACE); PEARLPLUS_CLASS is the class
  # Pearl+'s verdict holds the window's passes to (served_pearlplus.sh's `verify --class`)
  export PEARLPLUS_TIE PEARLPLUS_CLASS=${V3C_CLASS-4,32,4} PEARLC_BF16_ROWS=${PEARLC_BF16_ROWS:-0}
  # the BF16 1B's compiler of record lists its RoPE heads as transfer gates, every unit of which step 3 proves (work-rate:NUM/DEN)
  PP_PARTITION=${PP_PARTITION:-benchmarks/prove_path/llama32_1b_tg_l0_step1.json} PP_SETS=${PP_SETS:-rope-head/d64/bf16}
  # each tied row's Build of record, PROGRAM_DIGEST:MANIFEST_DIGEST, which the served compare holds PRED to (BUILD)
  case ${ROW_ID:-${ROW##*/}} in
    llama32-1b__bf16__h200__tp1__b4__i256__o32__mixed__greedy__bi-eager)
      BUILD_OF_RECORD=${BUILD_OF_RECORD:-95ba0fab376a62f5421ed21ced4e650afa516531508ecb802d065f44fd4f8ab451ba422c75312d8052b9087ff28826aa43edfc945435528856b97c8952903eb7:65ce5a6ccc1fd943c23a6d78b4eb80936d32c1043306ec3aab640b279eee419ef778205bd83b81721ad2c301114bd0e338e04b95acc959f8ba1188dcfa1ceac3} ;;
    *) BUILD_OF_RECORD=${BUILD_OF_RECORD:-} ;;
  esac
fi
H=/workspace/research/vllm-composed-h200int8
ROW=${ROW:-$H/r20261030-210701-a20f/llama32-1b__int8__h200__tp1__b4__i256__o32__mixed__greedy__bi-eager} ROW_ID=${ROW_ID:-${ROW##*/}}
HIDDEN_SO=${HIDDEN_SO:-$H/fa2/build/fa3_matReqG/verity_fa3_matReqG.so}
VENV=${VENV:-/workspace/cp/venv312}
REPO=${REPO:-verity-int8/Llama-3.2-1B-INT8} REV=${REV:-15b0976d400d796ee589e7737d03a919526bf800} PRECISION=${PRECISION:-verity_int8}
PORT=${PORT:-18190} SLOTS=${SLOTS:-4} PROMPTS=${PROMPTS:-16:24} MAX_TOKENS=${MAX_TOKENS:-4} RATE=${RATE:-1000} SEED=${SEED:-1}
REQUESTS=${REQUESTS:-$SLOTS} CLASSES=${CLASSES:-4,32,4} CAP=${CAP:-8388608} MAX_MODEL_LEN=${MAX_MODEL_LEN:-512}
PP_PARTITION=${PP_PARTITION:-benchmarks/prove_path/llama32_1b_silu_mul_v2_i8192_l0_step$([ "$MAX_TOKENS" = 1 ] && echo 0 || echo 1).json}
T=${T:-512}
DISK_FLOOR_GB=${DISK_FLOOR_GB:-100}
PEARLPLUS_INODES=${PEARLPLUS_INODES:-772388}  # two commits of the tied window at the class of record, three files a member
H1_KEY_OF_RECORD=27271eaf917d1a817198cbcfe7684debd9bdb11b9e386460948bb981642fab3ae065c0a877330a0da68ae76969550eb786dc172319876aaca21fd116e5a193bc  # vy-challenger-1's
H1_KEY_SHA512=${H1_KEY_SHA512:-$H1_KEY_OF_RECORD}
POUW=${POUW:-}
H1_PEER=${H1_PEER:-vy-dc-h200}
LANE_GATE="ssh -i ${PP_GATE_KEY:-} -o IdentitiesOnly=yes -o BatchMode=yes research@${VY_CHALLENGER_1_HOST:-}"
if [ "$POUW" = pearlplus ]; then
  PEARLC_CHALLENGER_GATE=${PEARLC_CHALLENGER_GATE-$LANE_GATE}
  # the peer is the gate's: with the gate empty it takes no default, since what reads the pair (bench_arm.py's live
  # coins, at the serve and at the verify) refuses a peer with no gate
  export PEARLC_CHALLENGER_GATE PEARLC_CHALLENGER_PEER=${PEARLC_CHALLENGER_PEER:-${PEARLC_CHALLENGER_GATE:+$H1_PEER}} \
    PEARLC_CHALLENGER_LANE=${PEARLC_CHALLENGER_LANE:-${H1_LANE:-vllm}}
  # shellcheck disable=SC2016
  export POUW_CMD=${POUW_CMD-'bash benchmarks/one_stage/served_pearlplus.sh "$1" "$2" "$3"'} POUW_TIMEOUT=${POUW_TIMEOUT:-120m}
fi
if [ -n "$PEARLPLUS_TIE" ]; then WINDOW_STEPS=${WINDOW_STEPS:-$(cut -d, -f3 <<< "${V3C_CLASS-4,32,4}")}  # round s the window's s-th step
else WINDOW_STEPS=${WINDOW_STEPS:-256}; fi
MAIN_SHA=${MAIN_SHA-$(git -C "$TREE" rev-parse HEAD 2>/dev/null)}
PRODUCT=${PRODUCT:-onsite}
. "$(dirname "${BASH_SOURCE[0]}")/placement.sh" || exit 2  # hold_placement
development_under_hold
# off the node only where the caller runs a composer there (COMPOSE_OFF_NODE=1), or under VERITY_HOLD_PLACEMENT=1
COMPOSE_OFF_NODE=${COMPOSE_OFF_NODE:-$(hold_placement && echo 1 || echo 0)} COMPOSE_NODE=${COMPOSE_NODE:-vy-auditor-2}
COMPOSE_GATE=${COMPOSE_GATE:-ssh -i $HOME/.ssh/vllm_gate_ed25519 -o BatchMode=yes -o IdentitiesOnly=yes research@${VY_AUDITOR_2_HOST:-}}
export COMPOSE_OFF_NODE
if [ -n "${ONSITE_PIECES:-}" ]; then
  case $ONSITE_PIECES in /*) ;; *) ONSITE_PIECES=$TREE/$ONSITE_PIECES ;; esac
  [ -f "$ONSITE_PIECES" ] || { echo "PREFLIGHT: ONSITE_PIECES is not a file" >&2; exit 3; }
  # shellcheck source=/dev/null
  . "$ONSITE_PIECES"
  [ "$PRODUCT" != onsite ] || onsite_piece_commands
fi
export H1_KEY_SHA512 H1_PEER WINDOW_STEPS MAIN_SHA WIPE_CMD POUS_CMD BANDWIDTH_CMD PRODUCT ONSITE_PIECES POUW
PRED=${PRED:-$D/predicted}
SERVED_COMPARE=$D/served-compare/served_compare.json
FW=${FLOCK_WORK:-$ROOT/flock-work}
export PY=$VENV/bin/python HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1
export CUDA_HOME=${CUDA_HOME:-$(python3 "$TREE/benchmarks/kernels/machine.py" toolkit 13.0)}
export PYTHONPATH=$TREE:$TREE/library/datasets:$TREE/adapters/vllm:$TREE/adapters/torch:$TREE/tools/config:$TREE/verity/protocols/proof/flock/prover/python:$TREE/tools/research/src:$TREE/benchmarks/harness
export TORCH_EXTENSIONS_DIR=$ROOT/cache/torch-extensions TRITON_CACHE_DIR=$ROOT/cache/triton VLLM_CACHE_ROOT=$ROOT/cache/vllm
export VLLM_BATCH_INVARIANT=1 VERITY_RETAIN=host NATIVE_COLLECT_BUILD=$ROOT/cache/jit/nc
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
# Every verdict line goes out in its published form, naming no window and counting none of a window's own, and in full to
# the serve dir's private/verdicts.txt (`served_verdicts.py public-lines`), under the node's python3, the venv's absent
SV=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/served_verdicts.py SAY_PY=$(command -v python3 || echo "$PY")
publish_lines() { "$SAY_PY" "$SV" public-lines --private "$D/private/verdicts.txt" | tee -a "$OUT/verdicts.txt"; }
say() { printf '%s\n' "$*" | publish_lines; }
failed() { "$SAY_PY" "$SV" failed "$@"; }  # RC [LOG|- ...]: a tool's failure by its exception's class and exit alone
said() { "$SAY_PY" "$SV" said "$@"; }  # RC [LOG|- ...]: our tool's last line where it passed, else its class alone
# No file holds a host's address: a gate left to its default reaches the host whose address the environment holds, the
# challenger host's in VY_CHALLENGER_1_HOST (LANE_GATE) and the compose node's in VY_AUDITOR_2_HOST (COMPOSE_GATE), and
# with that unset its command ends at research@. host_set VAR GATE...: the preflight's refusal of such a GATE, by VAR's
# name, for the verbs that ask a gate something; the others never reach one and run without the address
host_set() {
  local var=$1 gate
  shift
  for gate; do
    case $gate in
      *@) say "PREFLIGHT: $var is unset, the address of the host the default gate command reaches ($gate): set it, or give the gate's whole command"
        return 3 ;;
    esac
  done
}
# key_set GATE: the preflight's refusal of the lane's default gate (LANE_GATE) where PP_GATE_KEY, the lane's own gate key,
# is unset or no readable file: no gate call falls back to another key
key_set() {
  [ "$1" != "$LANE_GATE" ] || { [ -n "${PP_GATE_KEY:-}" ] && [ -r "$PP_GATE_KEY" ]; } && return 0
  say "PREFLIGHT: PP_GATE_KEY (${PP_GATE_KEY:-unset}) is no readable file: the lane's own gate key, made on this machine with its public half registered under the lane, which the lane's default gate command reaches the challenger with; set it, or give the gate's whole command"
  return 3
}
keep() { [ -d "$D" ] && (umask 077; mkdir -p "$D/private" && printf '%s\n' "$*" >> "$D/private/verdicts.txt"); }
count() { cat "$1" 2>/dev/null | grep -c .; }
vv() { "$PY" -m verity_vllm.pipeline.cli "$@"; }
case ${LOAD:=varied} in varied | dataset | front | session) ;; *) say "PREFLIGHT: LOAD is varied, dataset, front or session, not $LOAD"; exit 3 ;; esac
case ${1:-} in '' | serve) [ "$LOAD" != dataset ] || { [ -n "${DATASET:-}" ] && [ -n "${TIMING:-}" ]; } \
  || { say "PREFLIGHT: LOAD=dataset plays the input dataset DATASET at TIMING, and one of them is unset"; exit 3; } ;; esac
case $POUW in
  pearlplus) ;;
  h1) say "PREFLIGHT: POUW=h1 is refused: PoUW H1 is deleted, and PoUW's piece is Pearl+'s: POUW=pearlplus"; exit 3 ;;
  '') ;;
  *) say "PREFLIGHT: POUW is pearlplus, not $POUW"; exit 3 ;;
esac
if [ -n "${APPLICATION_PROTOCOLS+x}" ]; then  # the application's list, as verity run hands it the runner
  case " $APPLICATION_PROTOCOLS " in
    *" pearl-plus "*) [ -n "$POUW" ] || { say "PREFLIGHT: the application lists compute accounting (pearl-plus), and POUW names no scheme: run it under its application's hooks (verity run), which set POUW=pearlplus"; exit 3; } ;;
    *) [ -z "$POUW" ] || { say "PREFLIGHT: POUW=$POUW names a compute-accounting scheme, and the application lists no compute accounting: unset POUW, or run an application that lists pearl-plus"; exit 3; } ;;
  esac
fi
for v in H1_GATE H1_TIE H1_CIRCUITS H1_STATEMENTS H1_LIVE_EVIDENCE H1_RECEIPTS H1_PROTOCOLS_CONFIG; do
  [ -z "${!v:-}" ] || { say "PREFLIGHT: $v is PoUW H1's setting, and PoUW H1 is deleted"; exit 3; }
done
[ "$POUW" != pearlplus ] || [ -n "$PEARLC_CHALLENGER_GATE" ] || [ -z "$PEARLC_CHALLENGER_PEER" ] \
  || { say "PREFLIGHT: PEARLC_CHALLENGER_PEER is $PEARLC_CHALLENGER_PEER and PEARLC_CHALLENGER_GATE is empty, which leaves Pearl+'s coins in-process, and Pearl+'s serve and verify refuse a peer with no gate (bench_arm.py's live coins): unset PEARLC_CHALLENGER_PEER, or set PEARLC_CHALLENGER_GATE (unset, it is the lane's gate)"; exit 3; }
case $PRODUCT in
  onsite) ;;
  remote)
    for v in WIPE_CMD POUS_CMD BANDWIDTH_CMD CERTIFIER_CMD CERTIFIER_CHAIN_CMD CERTIFIER_PERIOD_CMD CERTIFIER_FRONT; do
      [ -z "${!v:-}" ] || { say "PREFLIGHT: PRODUCT=remote runs no onsite piece, and $v is set: unset it"; exit 3; }
    done ;;
  *) say "PREFLIGHT: PRODUCT is onsite or remote, not $PRODUCT"; exit 3 ;;
esac
case $COMPOSE_OFF_NODE in 0 | 1) ;; *) say "PREFLIGHT: COMPOSE_OFF_NODE is 1 (the default under VERITY_HOLD_PLACEMENT=1) or 0, not $COMPOSE_OFF_NODE"; exit 3 ;; esac
[[ -z ${V3C_GPUS:-} || $V3C_GPUS =~ ^[0-9]+(,[0-9]+)*$ ]] \
  || { say "PREFLIGHT: V3C_GPUS is the cards V3c's audit proves on, their indices comma-separated (0,1), not $V3C_GPUS"; exit 3; }
[ -z "${V3C_GPUS:-}" ] || command -v gpu-lease > /dev/null \
  || { say "PREFLIGHT: V3C_GPUS's cards are held under gpu-lease while V3c's audit proves on them, and this node has no gpu-lease"; exit 3; }
CKPT=${CKPT:-$HF_HOME/hub/models--${REPO//\//--}/snapshots/$REV} WREG=$D/weights_registration.json
NO_DREG="the deployment's registration record (deployment-registration.json) is absent: the serve writes it before its first request"
# the deployment's registration record, the registration log's first record, written once before the first request:
# the audits hand it to the verifier on the development path, and Pearl+'s W is chained after it
DREG=$D/deployment-registration.json
NO_WEIGHTS="NO WEIGHTS REGISTRATION: no window registers a weight's root, so every weight read counts free"
weights() {  # under the lease (its commit hashes on the leased GPU), before the first request: the committer's registration
  (cd "$TREE" && "$PY" benchmarks/prove_path/prove_path.py weights-registration --serve-run "$D" --checkpoint-dir "$CKPT" \
    --out "$WREG" --deployment-record "$DREG") > "$LOGS/weights_registration.txt" 2>&1
  local rc=$? log=$LOGS/weights_registration.txt
  grep -o '"deployment": "[0-9a-f]\{128\}"' "$log" | tail -1 > "$OUT/weights_registration.txt"
  if [ "$rc" != 0 ] || [ ! -f "$WREG" ]; then
    rm -f "$DREG"
    say "$NO_WEIGHTS (weights-registration $(failed "$rc" "$log"))"
    return
  fi
  cp "$WREG" "$OUT/"
  say "WEIGHTS $("$PY" - "$WREG" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
v, u = r["values"], r["unrooted"]
print(f"registered before the first request: {len(v)} weight parameters rooted ({sum(e['words'] * e['leaves'] for e in v.values()) * 16} "
      f"bits) of {r['model']['repo']}@{r['model']['revision'][:10]} under the committer's key (kept off every record); "
      f"{len(u)} unrooted" + "".join(f"; {p}: {why}" for p, why in sorted(u.items())))
EOF
)"
  [ -s "$DREG" ] || return 0
  cp "$DREG" "$OUT/"
  say "DEPLOYMENT-REGISTRATION the serve's first registration record, which registers those roots: written before the first" \
    "request by the developer (no auditor holds its digest), digest" \
    "$(grep -o '"deployment": "[0-9a-f]\{128\}"' "$log" | tail -1 | cut -d'"' -f4)"
}
V3C=${V3C_CLASS-4,32,4} v3c_ok=0 v3c_rc= v3c_why=
[ "${1:-}" = v3c ] && V3C=${V3C:-4,32,4}
# V3c's rule, a rate: window-commit commits the window under the sampling design it names over the class circuit, and the
# audit states it to its stream. The default samples one proof block of class 4,32,4's window; a window of more proof
# blocks is sampled in proportion. Under PEARLPLUS_TIE the window is
# another, cut at Pearl+'s linears, and the rule has no default: the launch states V3C_LAW, a stratified rate (the preflight)
if [ -z "$PEARLPLUS_TIE" ]; then V3C_LAW=${V3C_LAW:-rate:1/3481648}
else V3C_LAW=${V3C_LAW:-}; fi
vtag=${V3C//,/-}
PEARLPLUS_CIRCUIT=$D/pearlplus/circuit-record.json
if [ "${1:-}" = v3c ]; then V3C_TIMEOUT=${V3C_TIMEOUT:-120m} V3C_AUDIT_TIMEOUT=${V3C_AUDIT_TIMEOUT:-180m}
elif [ -n "$PEARLPLUS_TIE" ]; then V3C_TIMEOUT=${V3C_TIMEOUT:-60m} V3C_AUDIT_TIMEOUT=${V3C_AUDIT_TIMEOUT:-30m}
else V3C_TIMEOUT=${V3C_TIMEOUT:-15m} V3C_AUDIT_TIMEOUT=${V3C_AUDIT_TIMEOUT:-30m}; fi
V3C_WARM_TIMEOUT=${V3C_WARM_TIMEOUT:-90m}
V3C_WARM_GIVEN=${V3C_WARM_JOBS+1} V3C_WARM_JOBS=${V3C_WARM_JOBS-$(($(nproc) < 8 ? $(nproc) : 8))}
V3C_AUDIT_MEM_GIVEN=${V3C_AUDIT_MEM_GB:+1} V3C_AUDIT_MEM_GB=${V3C_AUDIT_MEM_GB:-290}
V3C_MIN_FREE_GB=${V3C_MIN_FREE_GB:-100}
NODE_MEM_GB=${NODE_MEM_GB:-$(awk '/^MemTotal:/ {print int($2 / 1048576)}' /proc/meminfo)}
short_of_memory() {  # 0 iff this node has less memory than V3c's audit needs, saying both
  [ "$NODE_MEM_GB" -lt "$V3C_AUDIT_MEM_GB" ] || return 1
  echo "this node has $NODE_MEM_GB GB, under the $V3C_AUDIT_MEM_GB GB V3c's audit needs (V3C_AUDIT_MEM_GB)"
}
# V3c's audit's coins (`v3c_audit`, records_audit's --challenger): V3C_CHALLENGER, by default SPOT's PP_CHALLENGER,
# on the gate with its peer V3C_PEER (default PP_PEER: the node the audit runs on, as the gate's /etc/vy/peers names it,
# in a follow-on not the served node) under PP_CHALLENGER_KEY and PP_GATE_KEY. A local or loopback session's coins are a
# development session's, simulated, which the COINS line says; under VERITY_HOLD_PLACEMENT=1 only a development run
# (DEVELOPMENT=1) takes them.
V3C_CHALLENGER=${V3C_CHALLENGER:-${PP_CHALLENGER:-local}} V3C_PEER=${V3C_PEER:-${PP_PEER:-}} DEVELOPMENT=${DEVELOPMENT:-}
v3c_live() { [[ $V3C_CHALLENGER == gate:* ]]; }
# Step 3's: PP_CHALLENGER's, but local while V3c's audit is live, since both open on the window's run root, which the
# gate opens once, ever, and V3c's is the SPOT audit of record
SP_CHALLENGER=${PP_CHALLENGER:-local}
[ -n "$V3C" ] && v3c_live && SP_CHALLENGER=local
v3c_coins() {  # where V3c's audit's coins come from, as its published COINS line says: a session's form, no host
  if v3c_live; then
    echo "the challenger ${V3C_CHALLENGER%%@*} (peer $([ -n "${DEFER:-}" ] && echo "the follow-on's node's" || echo "$V3C_PEER")), the window's run root's one session, the sampled-proofs audit of record"
  else
    echo "a development challenger session where it runs (V3C_CHALLENGER ${V3C_CHALLENGER%%:*}), $([ "$DEVELOPMENT" = 1 ] \
      && echo "in a development run (DEVELOPMENT=1)" || echo "simulated: not the auditor's challenger")"
  fi
}
coin_trees_held() {  # CHALLENGER SUMMARY STEP: an audit on the gate's coins (CHALLENGER) took its sessions' coin trees
  # from the session that sampled them, by its SUMMARY (`served_verdicts.py coin-trees`), else 20 with STEP's refusal
  # line; a development session's coins pass as before
  local why
  why=$("$SAY_PY" "$SV" coin-trees --challenger "$1" --summary "$2" 2>&1) && return 0
  say "$3 refused: $(tail -1 <<< "$why" | cut -c1-1200)"
  return 20
}
sampled_coins() { coin_trees_held "$SP_CHALLENGER" "$D/$SERVED_ARM/summary.json" SAMPLED-PROOFS; }  # step 3's
v3c_trees() { coin_trees_held "$V3C_CHALLENGER" "$VW/audit/summary.json" V3C-AUDIT; }  # V3c's audit's, VW its directory
v3c_gate() {  # V3C_CHALLENGER's gate as the command `served_verdicts.py gate` runs, the ssh records_audit's session reaches
  local g=(ssh -o BatchMode=yes)
  [ -z "${PP_GATE_KEY:-}" ] || g+=(-o IdentitiesOnly=yes -i "$PP_GATE_KEY")
  printf '%q ' "${g[@]}" "${V3C_CHALLENGER#gate:}"
}
tree_sha() { git -C "$TREE" rev-parse HEAD 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-}"; }
window_commit() {  # window K of $D committed as V3C's class circuit's units in $VW/window, its answer in $1 and its errors
  # in $2, the rest window-commit's further arguments: 0, else why on stdout
  local json=$1 err=$2 rc
  shift 2
  # off the GPU, which none leases here or the engine holds under the tie: the weights hashed by the NumPy path, the
  # reference
  (cd "$TREE" && CUDA_VISIBLE_DEVICES='' timeout -k 1m "$V3C_TIMEOUT" "$PY" benchmarks/one_stage/window_commit.py window-commit --serve-run "$D" \
    --window "$K" --row-dir "$D/windows/w$K" --advice "$D/advice/w$K/private/advice.bin" --class "$V3C" \
    --registration "$WREG" --checkpoint-dir "$CKPT" --sampling-design "$V3C_LAW" --out "$VW/window" "$@") > "$json" 2> "$err"
  rc=$?
  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then echo "timed out after V3C_TIMEOUT=$V3C_TIMEOUT (exit $rc)"
  elif [ "$rc" != 0 ]; then echo "refused $(failed "$rc" "$err")"; fi
  return "$rc"
}
# under PEARLPLUS_TIE: Pearl+'s window record W, in the serve dir's pearlplus/w<k>/, and where the record reads the tie
# (`window_tie`: R and V3c's audit's summary, record --window-tie)
pearl_dir() { echo "$D/pearlplus/w$K"; }
window_tie() {  # under PEARLPLUS_TIE: R, V3c's audit's summary and, once V3c held R's circuit to it, the window-circuit
  # record, in the layout `served_verdicts.py record --window-tie` reads
  local t
  t=$(pearl_dir)/tie
  mkdir -p "$t/window" && cp "$VW/window/registration.json" "$t/window/" 2>/dev/null \
    && { cp "$VW/audit/summary.json" "$t/" 2>/dev/null || true; } \
    && { [ "$v3c_ok" != 1 ] || cp "$D/$(v3c_record)" "$t/circuit-record.json" 2>/dev/null || true; }
}
served_compare() {  # under PEARLPLUS_TIE, once Pearl+'s piece wrote W and before the record, on the CPU: the serve held to
  # the workload of record (served_compare.py), its SERVED-COMPARE line the one the record names (--served-compare)
  local dir=${SERVED_COMPARE%/*} rc
  [ -n "$PEARLPLUS_TIE" ] || { say "SERVED-COMPARE: not run: PEARLPLUS_TIE is empty, so no W ties the serve to the row"; return 3; }
  # an earlier compare's verdict never stands for this one's, which can die before it writes its own
  mkdir -p "$dir" "$OUT/served-compare" && rm -f "$SERVED_COMPARE" "$dir/circuit_facts.json"
  (cd "$TREE" && CUDA_VISIBLE_DEVICES='' timeout -k 1m "${SERVED_COMPARE_TIMEOUT:-30m}" "$PY" benchmarks/one_stage/served_compare.py \
    --serve "$D" --row "$ROW_ID" --class "$V3C" --pearl-tables "${PEARLPLUS_ARM#pouw:}" --pred "$PRED" \
    --of-record "$BUILD_OF_RECORD" --window-record "$(pearl_dir)/registration.json" --revision "$REV" --out "$dir") 2> "$dir/compare.err" \
    | publish_lines
  rc=${PIPESTATUS[0]}
  # its answer is 0 EQUAL or 1 NOT EQUAL with its verdicts written; a traceback exits 1 too, so any other end is an ERROR
  # and leaves no verdict for the record to read
  if ! { [ "$rc" -le 1 ] && grep -q "\"ok\": $([ "$rc" = 0 ] && echo true || echo false)," "$SERVED_COMPARE" 2>/dev/null \
         && ! grep -q '^Traceback (most recent call last):' "$dir/compare.err"; }; then
    say "SERVED-COMPARE: ERROR served_compare.py ended with no verdict $(failed "$rc" "$dir/compare.err")"
    rm -f "$SERVED_COMPARE"
    [ "$rc" != 0 ] || rc=1
  fi
  cp "$dir/served_compare.json" "$dir/circuit_facts.json" "$OUT/served-compare/" 2>/dev/null
  return "$rc"
}
pearl_period() {  # under PEARLPLUS_TIE with a gate, in serve(): $1 open (on PEARLPLUS-CIRCUIT's record) or close the lane's
  # Pearl-C period the window's salt session joins, its reserves first held to the windows' salt sessions
  local out
  if [ "$1" = open ]; then
    [ -n "$PEARLPLUS_TIE" ] && [ -n "$PEARLC_CHALLENGER_GATE" ] || return 0
    [ -f "$PEARLPLUS_CIRCUIT" ] || { say "PEARLPLUS-PERIOD not opened: PEARLPLUS-CIRCUIT built no record, so the window's salt session joins no period"; return 1; }
    out=$("$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" pearlplus-period open "$D/pearlplus" \
      --gate "$PEARLC_CHALLENGER_GATE" --name "${PEARLPLUS_PERIOD:-pearlplus-${RESEARCH_RUN_ID:-local}}" \
      --circuit-record "$PEARLPLUS_CIRCUIT" 2>&1) && PEARL_PERIOD=1
  else
    [ "${PEARL_PERIOD:-}" = 1 ] || return 0
    out=$("$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" pearlplus-period close "$D/pearlplus" \
      --gate "$PEARLC_CHALLENGER_GATE" --lane "$PEARLC_CHALLENGER_LANE" --windows "$D/pearl" \
      --wait "${PEARLPLUS_CLOSE_WAIT:-300}" 2>&1) && PEARL_PERIOD=
  fi
  say "PEARLPLUS-$(tail -1 <<< "$out" | cut -c1-600)"
}
v3c_record() {  # the verifier's window-circuit record V3c holds window K's commit to, relative to D: at V3C's class
  echo "window-circuit-w$K-$vtag.json"
}
v3c_circuit() {  # in TREE: that record (`v3c_record`), within WC_TIMEOUT
  CUDA_VISIBLE_DEVICES='' circuit "$V3C" "WC_TIMEOUT=${WC_TIMEOUT:-30m}" "WINDOW-CIRCUIT class $V3C"
}
v3c_commit() {  # window K of $D committed as V3C's class circuit's units, its circuit held to the verifier's window-circuit record
  VW=$D/v3c-w$K-$vtag
  local json=$D/window-commit-$vtag.json why rec none=" (CLASSES '$CLASSES')"
  rec=$D/$(v3c_record)
  [ -f "$rec" ] || { say "WINDOW-COMMIT class $V3C: no window-circuit record at it$none"; return 1; }
  [ -f "$WREG" ] || { say "WINDOW-COMMIT class $V3C: $NO_WEIGHTS"; return 1; }
  why=$(window_commit "$json" "$LOGS/window-commit-$vtag.err") \
    || { v3c_rc=$? v3c_why=$why; say "WINDOW-COMMIT class $V3C $why"; return 1; }
  # the commit's population and sources count the window's own values: they stay in the serve dir
  say "WINDOW-COMMIT class $V3C: $("$PY" -c 'import json, sys; c = json.load(open(sys.argv[1])); print(json.dumps({k: c[k] for k in ("run", "circuit", "registration") if k in c}, sort_keys=True))' \
    "$json" 2>&1 | cut -c1-1500)"
  "$PY" - "$rec" "$VW/window/circuit.json" "$V3C" <<'EOF' 2>&1 | publish_lines
import json, sys
from verity.commitments.periods import interface as IF
mine, theirs = (json.load(open(f)) for f in sys.argv[1:3])
ok = IF.circuit_of(theirs) == IF.circuit_of(mine)
print(f"WINDOW-CIRCUIT-HELD class {sys.argv[3]}: the committed circuit.json less its run fields is the verifier's own "
      f"window-circuit record less its run fields: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
EOF
  [ "${PIPESTATUS[0]}" = 0 ] && v3c_ok=1
}
commit_unread() {  # why a failed v3c_commit left the split network attester's piece nothing to read: a timeout or the failure
  local unread="so the split network attester's chain and period check, which read it, did not start"
  case $v3c_rc in
    124|137) echo "V3c's window commit at class $V3C did not finish within V3C_TIMEOUT=$V3C_TIMEOUT (exit $v3c_rc), $unread: the commit is processor work, off the GPU, that grows with the class's units; set V3C_TIMEOUT to cover it (\`served_verdicts.sh v3c\` defaults to 120m)" ;;
    *) echo "V3c's window commit at class $V3C ${v3c_why:-failed (its WINDOW-COMMIT and WINDOW-CIRCUIT-HELD lines)}, $unread" ;;
  esac
}
piece_unread() {  # $1 as the split network attester's piece's refusal, which leaves its row missing, unless it answered or refused
  mkdir -p "$OUT/pieces/certifier"
  [ -e "$OUT/pieces/certifier/answer.json" ] || [ -e "$OUT/pieces/certifier/refused.txt" ] \
    || echo "$1" > "$OUT/pieces/certifier/refused.txt"
}
# The audit's sessions listen on V3C_PORTS ports (default 16, a5's RANK_PORTS), a sample needing more refused before any
# listens: a sample of K proof blocks opens one for each of at most K members, and records_audit's own default, a port for every
# member, finds no free span in `records_audit.PORTS` at a class window. Under PEARLPLUS_TIE the audit is tied to Pearl+'s W
# (records_audit's --tie W, `tie_of`): the tables the circuit reads are W's, R registers them as its outputs, and each sampled
# reader proves its rows at W's roots. Its rule is then a stratified rate (V3C_LAW), which samples a stratum of each reader
# template, and V3C_PORTS defaults to the size of R's sample, its strata's k summed: at most a session for each sampled proof
# block. The preflight refuses PEARLPLUS_TIE on a tree whose records_audit.py takes no --tie, or at another rule.
# With V3C_WARM_JOBS=N (default the cores this run is given, at most 8; set empty, no warm, and the audit composes each
# template itself, one at a time), `records_audit.py warm` first composes every template of the committed window into
# $VW/compose-cache, N at a time, each in a process of its own, within V3C_WARM_TIMEOUT (default 90m), printing each
# template's line as it finishes into warm.log and starting none past V3C_AUDIT_MEM_GB by earlier peaks: the
# templates and their binding are public, so this precedes the audit's coins, and the audit, taking the cache, composes
# none, its limit then its sessions'. A compose holds up to what the audit needs (V3C_AUDIT_MEM_GB, a 4,32,4 window's
# Int8RowQuant at K 8,192), so N is what the run's memory takes. A warm that fails says so, and the audit composes what it
# didn't keep.
# V3C_COMPOSE_CACHE=DIR names that cache instead of $VW/compose-cache: a directory outside the window's, absolute or
# refused (the warm and the audit run in TREE), which no step here removes. An entry is one template's circuit under
# one key (`a5.compose_key`): the template, its Definition's program, its statement in the circuit record, the composing code
# (compose_circuit.py, records_audit.py, the packages `compose_circuit.COMPOSERS`), the FLOCK_* lowering flags and
# Python's and NumPy's versions, and not the window, whose partition and program are two values of the circuit's
# META. So another window's warm and audit take the entries of the templates they share with the windows
# composed into DIR before, at this tree and environment, and compose and keep the rest: the audit gets each circuit as
# a link where its partition and program are the entry's (a copy on another file system than DIR's), else as a copy
# bound to its own, the entry held to its pin as it is copied (`compose_circuit.cached`). A kept entry is trusted as the
# tree is: staging takes a circuit's META, pin and class off the line its compose kept beside it, a sampled member's
# circuit is held to that line before its prover reads it (`a5.held`), and nothing else says what a template's circuit is.
# On a tree whose records_audit.py takes them, an entry is the compose's record alone (its META, pin, class and size, the
# text hashed as it is rendered) unless V3C_KEEP_CIRCUITS=1, and the audit composes a sampled member's template after its
# sample, held to the record's pin; the warm and the audit start no compose, and the warm stops, where the cache's file
# system would have less than V3C_MIN_FREE_GB (default 100) free.
v3c_compose_args() {  # the warm's and the audit's disk floor and text keeping, on a tree whose records_audit.py takes them
  grep -q -- '--min-free-gb' "$TREE/benchmarks/prove_path/records_audit.py" || return 0
  printf '%s\n' --min-free-gb "$V3C_MIN_FREE_GB" ${V3C_KEEP_CIRCUITS:+--keep-circuits}
}
wreg_args() {  # the deployment's registration record: under REGISTRATION_LANE, held to the head the auditor's challenger
  # signed for that lane in its ledger REGISTRATION_LEDGER under the key record REGISTRATION_KEY (default
  # PP_CHALLENGER_KEY); with no lane, the development path, named
  if [ -n "${REGISTRATION_LANE:-}" ]; then
    printf '%s\n' --registration "$DREG" --registration-ledger "${REGISTRATION_LEDGER:-}" \
      --challenger-key "${REGISTRATION_KEY:-${PP_CHALLENGER_KEY:-}}" --registration-lane "$REGISTRATION_LANE" \
      ${REGISTRATION_KEY_SHA512:+--challenger-key-sha512 "$REGISTRATION_KEY_SHA512"}
  else
    printf '%s\n' --registration "$DREG" --registration-head development
  fi
}
registration_hold() {  # after weights wrote $DREG, before the epoch's first reserve: under REGISTRATION_LANE, held until
  # REGISTRATION_LEDGER carries the lane's registering period-open with $DREG's digest, at most REGISTRATION_HOLD_S
  # seconds; with no lane, the development line and no hold
  local rc
  if [ -n "${REGISTRATION_LANE:-}" ]; then
    [ -s "$DREG" ] || { say "REGISTRATION-HOLD refused: $NO_DREG"; return 1; }
    timeout -k 10s "$REGISTRATION_HOLD_S" "$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" registration-hold "$DREG" \
      --lane "$REGISTRATION_LANE" --ledger "$REGISTRATION_LEDGER" --key "${REGISTRATION_KEY:-${PP_CHALLENGER_KEY:-}}" \
      ${REGISTRATION_KEY_SHA512:+--key-sha512 "$REGISTRATION_KEY_SHA512"} ${PP_GATE_KEY:+--gate-key "$PP_GATE_KEY"} | publish_lines
  else
    "$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" registration-hold "$DREG" | publish_lines
  fi
  rc=$?
  [ "$rc" != 124 ] || say "REGISTRATION-HOLD refused: $REGISTRATION_LEDGER carried no registering period-open of lane $REGISTRATION_LANE within REGISTRATION_HOLD_S=${REGISTRATION_HOLD_S}s"
  return "$rc"
}
v3c_warm() {  # every template of the committed window composed into $VW/compose-cache, V3C_WARM_JOBS at a time
  local ca; mapfile -t ca < <(v3c_compose_args)
  (cd "$TREE" && env -u PYTHONPATH RAYON_NUM_THREADS="$(nproc)" RUST_MIN_STACK=$((64 << 20)) timeout -k 1m "$V3C_WARM_TIMEOUT" \
    "$TREE/.venv/bin/python" benchmarks/prove_path/records_audit.py warm --window "$VW/window" --sampling-design "$V3C_LAW" \
    --compose-cache "${V3C_COMPOSE_CACHE:-$VW/compose-cache}" --jobs "$V3C_WARM_JOBS" --mem-gb "$V3C_AUDIT_MEM_GB" "${ca[@]}") > "$LOGS/v3c-audit/warm.log" 2> "$LOGS/v3c-audit/warm.err"
  local rc=$? why
  if [ "$rc" = 0 ]; then why=$(grep -m1 '^WARM {' "$LOGS/v3c-audit/warm.log" | cut -c1-600)
  else
    if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then why="timed out after V3C_WARM_TIMEOUT=$V3C_WARM_TIMEOUT with $(grep -c '^COMPOSED ' "$LOGS/v3c-audit/warm.log") of $(grep -m1 '^WARMING' "$LOGS/v3c-audit/warm.log" | cut -d' ' -f2) templates composed (each one's seconds in warm.log)"
    else why=$(failed "$rc" "$LOGS/v3c-audit/warm.err"); fi
    why+="; the audit composes the templates it didn't keep"
  fi
  say "V3C-WARM class $V3C (records_audit.py warm exit $rc, $V3C_WARM_JOBS at a time): $why"
}
v3c_cuda() { FLOCK_WORK=$FW . "$TREE/verity/protocols/proof/flock/prover/pod/cuda-13.3.sh" >&2 && { [ -n "${CUDA_TK:-}" ] && echo "$CUDA_TK" || python3 "$TREE/benchmarks/kernels/machine.py" toolkit 13.3; }; }
v3c_audit() {  # the committed window audited from its records alone: its circuit and run root the verifier's, the weights pinned
  local run leanb m0 rc gl tk tie=() day=() cache=() ca=() ch=(--challenger "$V3C_CHALLENGER") ports=${V3C_PORTS:-16} main=$OUT/v3c-audit/main.txt
  [ -n "${DAY:-}" ] && day=(--day "$DAY" --day-log "$DAY_LOG" ${DAY_PARAMS:+--day-params "$DAY_PARAMS"})
  run=$("$PY" -c 'import json, sys; print(next(json.loads(l)["run_root"] for l in open(sys.argv[1]) if l.strip() and json.loads(l)["window"] == int(sys.argv[2])))' \
    "$D/windows.jsonl" "$K") || { say "V3C-AUDIT: no run root for the window in windows.jsonl"; return 1; }
  [ -f "$DREG" ] || { say "V3C-AUDIT: $NO_DREG"; return 1; }
  mkdir -p "$OUT/v3c-audit"
  if [ -n "${MAIN_FILE:-}" ]; then  # main's first-parent line, which the audit's challenger session pins on
    [ -s "$MAIN_FILE" ] || { say "V3C-AUDIT: MAIN_FILE $MAIN_FILE, main's first-parent line its challenger session pins on, is missing or empty"; return 1; }
    [ "$MAIN_FILE" -ef "$main" ] || cp "$MAIN_FILE" "$main"
  else
    git -C "$TREE" rev-list --first-parent "${MAIN_SHA:-HEAD}" > "$main" 2>/dev/null && [ -s "$main" ] \
      || { say "V3C-AUDIT: no first-parent line of main for its challenger session to pin on (MAIN_FILE unset, and ${MAIN_SHA:-HEAD} has none in this tree): set MAIN_FILE"; return 1; }
  fi
  if [ -n "$PEARLPLUS_TIE" ]; then
    tie=(--tie "$(pearl_dir)/registration.json")
    [ -n "${V3C_PORTS:-}" ] || [[ $V3C_LAW == rate:* ]] || ports=$("$PY" -c 'import json, sys; print(sum(int(s["k"]) for s in json.load(open(sys.argv[1]))["law"]["strata"]))' \
      "$VW/window/registration.json") || { say "V3C-AUDIT: R, the window's registration, has no strata to size the audit's ports by: set V3C_PORTS"; return 1; }
  fi
  leanb=$TREE/verity/protocols/proof/flock/verifier/.lake/build/bin
  m0=$TREE:$TREE/benchmarks/numerical/python:$TREE/verity/protocols/proof/flock/prover/python:$TREE/adapters/vllm
  if [ -n "${V3C_WARM_JOBS:-}" ]; then
    v3c_warm
    cache=(--compose-cache "${V3C_COMPOSE_CACHE:-$VW/compose-cache}")
  fi
  local pool=(${V3C_LEAN_WORKERS:+--lean-workers "$V3C_LEAN_WORKERS"} ${V3C_AUDIT_MEM_GIVEN:+--audit-mem-gb "$V3C_AUDIT_MEM_GB"})
  [ -n "$FOLLOW" ] || pool=(--lean-workers "${V3C_LEAN_WORKERS:-1}" --audit-mem-gb "$V3C_AUDIT_MEM_GB")
  pool+=(${V3C_NEGATIVE_WORKERS:+--negative-workers "$V3C_NEGATIVE_WORKERS"})
  if v3c_live; then  # the gate opens the run root once, ever, and a reserve joins the lane's open period: neither, first
    gl=$("$PY" "$SV" gate --gate "$(v3c_gate)" --lane "${H1_LANE:-vllm}" --main "$main" --root "$run" \
      ${REGISTRATION_LANE:+--registration-record "$DREG"} 2>&1); rc=$?
    say "V3C-$(tail -1 <<< "$gl" | cut -c1-1200) (checked $(date -u +%Y-%m-%dT%H:%M:%SZ))"
    [ "$rc" = 0 ] || return 1
    ch+=(--peer "$V3C_PEER" --challenger-key "$PP_CHALLENGER_KEY" ${PP_GATE_KEY:+--gate-key "$PP_GATE_KEY"})
  fi
  V3C_AUDITED=$(hostname -s)
  local link=() gpus=() ld=() lease=() wr=() bin=$FW/flock-gpu-link/flock/target/release/flock-circuit n
  [ "${V3C_ARCHIVE_LINK:-}" != 1 ] || link=(--archive-link)
  if [ -n "${V3C_GPUS:-}" ]; then
    tk=$(v3c_cuda) || { say "V3C-AUDIT: V3C_GPUS=$V3C_GPUS, and this node has no CUDA 13.3 toolkit for its prover"; return 1; }
    gpus=(--gpus "$V3C_GPUS") ld=(LD_LIBRARY_PATH="$tk/compat:$tk/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
    bin=$FW/gpu/flock-gpu-link/flock/target/release/flock-circuit
    n=${V3C_GPUS//[^,]/}
    lease=(gpu-lease "$((${#n} + 1))" --wait --on "$V3C_GPUS" --mem-gb 0 -- env -u CUDA_VISIBLE_DEVICES)
  fi
  mapfile -t ca < <(v3c_compose_args); mapfile -t wr < <(wreg_args)
  (cd "$TREE" && "${lease[@]}" env -u PYTHONPATH "${ld[@]}" RAYON_NUM_THREADS="$(nproc)" RUST_MIN_STACK=$((64 << 20)) timeout -k 1m "$V3C_AUDIT_TIMEOUT" \
    "$TREE/.venv/bin/python" benchmarks/prove_path/records_audit.py audit --window "$VW/window" --sampling-design "$V3C_LAW" \
    --circuit "$(sha512sum "$VW/window/circuit.json" | cut -c1-128)" --run "$run" "${wr[@]}" "${tie[@]}" "${day[@]}" "${cache[@]}" "${ca[@]}" \
    "${pool[@]}" "${gpus[@]}" \
    "${ch[@]}" --main "$main" --m0-pythonpath "$m0" --bin "$bin" --lean "$leanb/verity-verify-proof" \
    --zk-server "$leanb/flock-zk-server" --ports "$ports" "${link[@]}" --out "$VW/audit") > "$LOGS/v3c-audit/audit.log" 2> "$LOGS/v3c-audit/audit.err"
  rc=$?
  cp "$VW/audit/summary.json" "$OUT/v3c-audit/" 2>/dev/null
  [ -z "$PEARLPLUS_TIE" ] || say "V3C-TIE: $(grep -m1 '^TIE ' "$LOGS/v3c-audit/audit.log" | "$PY" -c 'import json, sys
line = sys.stdin.read()
if not line.startswith("TIE "):
    sys.exit("no TIE line: the audit stopped before it read its records tied to W (v3c-audit/audit.log)")
t = json.loads(line[4:])
w, ok = t["record"][:16], "PASS" if t["chained"] else "FAIL"
print(f"the audit read its records tied to W {w}…, R chained right after it: {ok}")' 2>&1 \
    | tail -1 | cut -c1-400)"
  ! grep -q '^WEIGHTS ' "$LOGS/v3c-audit/audit.log" || say "V3C-AUDIT $(grep -m1 '^WEIGHTS ' "$LOGS/v3c-audit/audit.log" | cut -c9-400)"
  say "V3C-AUDIT class $V3C (records_audit.py exit $rc), $("$PY" "$SV" coins --challenger "$V3C_CHALLENGER" \
    --summary "$VW/audit/summary.json" 2>&1 | tail -1 | cut -c1-300): $(grep -m1 '^RECORDS' "$LOGS/v3c-audit/audit.log" | cut -c1-3000)"
  [ -n "$(grep -m1 '^RECORDS' "$LOGS/v3c-audit/audit.log")" ] \
    || say "V3C-AUDIT: no RECORDS line $(failed "$rc" "$LOGS/v3c-audit/audit.err")"
  v3c_trees || return 20
  return $rc
}
v3c_period() {  # V3c's audit as one compute audit of the network certifier's period: $VW/period.json
  (cd "$TREE" && env -u PYTHONPATH "$TREE/.venv/bin/python" benchmarks/prove_path/records_audit.py period --window "$VW/window" \
    --audit "$VW/audit" --pins "$WREG" --out "$VW/period.json" --tied "$("$TREE/.venv/bin/python" -c \
    'from verity.protocols.accounting.network.verifier.blocks import SCAN_STATE; print(SCAN_STATE)')") > "$LOGS/v3c-audit/period.log" 2>&1
  local rc=$?
  [ "$rc" = 0 ] && cp "$VW/period.json" "$OUT/v3c-audit/"
  say "V3C-PERIOD (records_audit.py period exit $rc): $(said "$rc" "$LOGS/v3c-audit/period.log" | cut -c1-800)"
  return $rc
}
attester_step() {  # $1 chain|period: network attestation's split step on V3c's window (step 6), if its command is set
  local cmd lim dir=$OUT/pieces/certifier rc
  if [ "$1" = chain ]; then cmd=${CERTIFIER_CHAIN_CMD:-} lim=${CERTIFIER_TIMEOUT:-4h}
  else cmd=${CERTIFIER_PERIOD_CMD:-} lim=${CERTIFIER_PERIOD_TIMEOUT:-60m}; fi
  [ -n "$cmd" ] || return 0
  mkdir -p "$dir"
  (cd "$TREE" && timeout -k 1m "$lim" bash -c "$cmd" _ "$D" "$K" "$dir" "$VW") > "$dir/$1.log" 2>&1
  rc=$?
  say "CERTIFIER-${1^^} (exit $rc): $(said "$rc" "$dir/$1.log" | cut -c1-600)"
  return $rc
}
v3c_proved() {  # V3c's window committed, the build done: network attestation's chain beside V3c's audit, then its period check
  local rc prc chain
  attester_step chain & chain=$!
  v3c_audit; rc=$?
  v3c_period; prc=$?
  wait "$chain"
  if [ "$prc" = 0 ]; then attester_step period
  elif [ -n "${CERTIFIER_PERIOD_CMD:-}" ]; then say "CERTIFIER-PERIOD: not run: V3c wrote no period.json"; fi
  return $((rc ? rc : prc))
}
pearl_circuit() {  # under PEARLPLUS_TIE, in serve() before serve-openai: the circuit at V3C's class cut at Pearl+'s linears,
  # from what serve.json will record of φ on this device (prove_path.py serve-phi, class-circuit), with no advice: the
  # record step 3's window-circuit writes of the window, which a period can open on before the serve
  local dir=$D/pearlplus asy=() rc
  [[ " ${SERVE_ARGS:-} " == *" --async-scheduling "* ]] && asy=(--async-scheduling)
  mkdir -p "$dir"
  "$PY" "$TREE/benchmarks/prove_path/prove_path.py" serve-phi --model "$REPO" --revision "$REV" --out "$dir/serve-phi.json" \
    "${asy[@]}" > "$dir/serve-phi.log" 2>&1 \
    && timeout "${PEARLPLUS_CIRCUIT_TIMEOUT:-30m}" "$PY" "$TREE/benchmarks/prove_path/prove_path.py" class-circuit \
      --serve "$dir/serve-phi.json" --class "$V3C" --pearl-tables "${PEARLPLUS_ARM#pouw:}" --out "$PEARLPLUS_CIRCUIT" \
      > "$dir/circuit-record.count.json" 2> "$dir/circuit-record.err"
  rc=$?
  if [ "$rc" = 0 ]; then
    say "PEARLPLUS-CIRCUIT class $V3C, cut at Pearl+'s linears, from φ before the serve ($(tail -1 "$dir/serve-phi.log" | cut -c1-200)): record $(sha256sum "$PEARLPLUS_CIRCUIT" | cut -c1-16)… $(stat -c %s "$PEARLPLUS_CIRCUIT") bytes"
  else
    rm -f "$PEARLPLUS_CIRCUIT"
    say "PEARLPLUS-CIRCUIT refused $(failed "$rc" "$dir/circuit-record.err" "$dir/serve-phi.log")"
  fi
  return "$rc"
}

serve() {  # under the lease: the endpoint, the requests, every closed window sealed, the endpoint stopped
  export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
  mkdir -p "$D" "$OUT/serve" && logs_dir
  cd "$TREE/adapters/vllm" || return 9
  local pw=() day=()
  [ -n "${DAY:-}" ] && day=(--day "$DAY" --day-log "$DAY_LOG" --day-program "$DAY_PROGRAM"
                            ${DAY_BLOCKS:+--day-blocks "$DAY_BLOCKS"} ${DAY_PARAMS:+--day-params "$DAY_PARAMS"})
  if [ -n "$PEARLPLUS_TIE" ]; then  # the window's linears Pearl+'s matmuls on this one engine (serve_pearl), one epoch a window
    pw=(--protocols pearl-plus --protocols-config "$("$SAY_PY" -c 'import json, sys; print(json.dumps({"pearl-plus": {"device": "cuda", "kernel": sys.argv[1], "cubin": sys.argv[2], **({"users": sys.argv[3]} if sys.argv[3] else {})}}))' \
      "$PEARLC_KERNEL" "$PEARLC_CUBIN" "${CERTIFIER_FRONT:+fixed\.[0-9]+}")")
    [ -n "$PEARLC_CHALLENGER_GATE" ] && pw+=(--challenger-gate "$PEARLC_CHALLENGER_GATE"
                                            --challenger-peer "$PEARLC_CHALLENGER_PEER" --challenger-lane "$PEARLC_CHALLENGER_LANE")
  fi
  local cfg="" rest=() i=0  # release order is the network-attester scheme's setting, in the serve's one --protocols-config
  while [ "$i" -lt "${#pw[@]}" ]; do
    if [ "${pw[$i]}" = --protocols-config ]; then cfg=${pw[$((i + 1))]}; i=$((i + 2)); else rest+=("${pw[$i]}"); i=$((i + 1)); fi
  done
  "$PY" -m verity.protocols.run.schemes network-attester settings > /dev/null || return 9
  pw=("${rest[@]}" --protocols-config "$("$PY" -c 'import json, sys; v, n = sys.argv[1:]; d = (json.loads(v) if v.lstrip().startswith("{") else json.load(open(v))) if v else {}; print(json.dumps({**d, n: {**d.get(n, {}), "order": "release-order"}}))' "$cfg" network-attester)")
  local polls=450
  if [ -n "${WIPE_CMD:-}" ]; then  # the wipe of this card, before its engine loads
    mkdir -p "$OUT/pieces/wipe"
    (cd "$TREE" && timeout -k 1m "${WIPE_TIMEOUT:-20m}" bash -c "$WIPE_CMD" _ "$D" all "$OUT/pieces/wipe") \
      > "$LOGS/wipe-piece.log" 2>&1
    rc=$?
    printf '{"limit": "%s", "rc": %d}\n' "${WIPE_TIMEOUT:-20m}" "$rc" > "$OUT/pieces/wipe/piece.json"
    say "WIPE (exit $rc, within WIPE_TIMEOUT=${WIPE_TIMEOUT:-20m}): $(said "$rc" "$LOGS/wipe-piece.log" | cut -c1-600)"
  fi
  # the policy is written before serve-openai's start, which the verdict reads as the serve's beginning
  [ -z "$PEARLPLUS_TIE" ] || { pearl_circuit; (cd "$TREE" && inference_policy); pearl_period open; }
  # shellcheck disable=SC2086
  "$PY" -m verity_vllm.pipeline.cli serve-openai --model "$REPO" --revision "$REV" --out "$D" --port "$PORT" \
    --precision "$PRECISION" --seal \
    --max-num-batched-tokens "$T" --window-steps "$WINDOW_STEPS" "${pw[@]}" "${day[@]}" \
    --row-build "$PRED" --hidden-so "$HIDDEN_SO" --tap-cap-mb 2048 \
    --window-requests "$SLOTS" --window-class "${V3C_CLASS:-4,32,4}" --max-num-seqs "$SLOTS" \
    --max-model-len "$(IFS=, read -r _ p o _ <<< "${V3C_CLASS:-4,32,4}"; echo $((p + o)))" \
    --disk-floor-gb "$DISK_FLOOR_GB" --seal-hold-gb 32 ${SERVE_ARGS:-} \
    > "$D/serve.log" 2>&1 &
  local srv=$! up=0 c=0 s=0 rc=0 release
  trap 'kill -TERM "$srv" 2>/dev/null; wait "$srv"; pearl_period close; exit 143' TERM INT
  for _ in $(seq 1 900); do
    grep -q "SERVE-OPENAI listening" "$D/serve.log" && { up=1; break; }
    kill -0 "$srv" 2>/dev/null || break
    sleep 2
  done
  if [ "$up" != 1 ]; then
    kill -TERM "$srv" 2>/dev/null; wait "$srv"; rc=$?
    say "SERVE: $("$SAY_PY" "$SV" not-listening "$D/serve.log" "$rc")"
    pearl_period close
    return 10
  fi
  say "SERVE $(grep -m1 'SERVE-OPENAI listening' "$D/serve.log" | cut -c1-300)"
  weights
  if ! registration_hold; then
    kill -TERM "$srv" 2>/dev/null; wait "$srv"
    pearl_period close
    return 11
  fi
  if [ -n "${CERTIFIER_FRONT:-}" ]; then  # the network certifier in front of the endpoint, before the requests
    # below: its group closes into a window of its own, the run's earliest: never the window of record under
    # PEARLPLUS_TIE (that is the fixed load's), and otherwise the window of record only where it has the most requests.
    # Its session, bundle and log hold the group's prompts, request and cells, so they stay in the serve dir's private/
    (umask 077; mkdir -p "$D/private")
    (cd "$TREE" && timeout -k 1m "${CERTIFIER_FRONT_TIMEOUT:-20m}" bash -c "$CERTIFIER_FRONT" _ "$PORT" "$D" "$D/private/certifier-front") \
      > "$D/private/certifier-front.log" 2>&1
    rc=$?
    [ "$LOAD" = session ] \
      || say "CERTIFIER-FRONT (exit $rc): $("$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" front-line "$D/private/certifier-front/session.json" 2>&1 | cut -c1-600)"
  fi
  if [ "$LOAD" = session ]; then  # the caller's load through the relay session the front opened, then the session's close
    local relay= f sent_rc closed
    for f in "$D"/private/certifier/*/relay.json; do  # the open session's: no fetch kept beside it yet
      [ -f "$f" ] && [ ! -e "${f%/*}/fetch.tar.gz" ] && { relay=$f; break; }
    done
    if [ "$rc" = 0 ] && [ -n "$relay" ]; then
      (cd "$TREE" && timeout -k 1m "${LOAD_TIMEOUT:-170m}" bash -c "$LOAD_CMD" _ "$relay" "$D") > "$D/load.log" 2>&1
      sent_rc=$?
      say "REQUESTS (exit $sent_rc): the caller's load, sent through the network certifier's relay session (the serve dir's load.log, on the node)"
    else
      sent_rc=$((rc ? rc : 1))
      say "REQUESTS: none sent: the front opened no relay session (exit $rc)"
    fi
    (cd "$TREE" && timeout -k 1m "${CERTIFIER_CLOSE_TIMEOUT:-180m}" bash -c "$CERTIFIER_CLOSE" _ "$D" "$D/private/certifier-front") \
      >> "$D/private/certifier-front.log" 2>&1
    closed=$?
    say "CERTIFIER-FRONT (exit $closed): $("$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" front-line "$D/private/certifier-front/session.json" 2>&1 | cut -c1-600)"
    rc=$sent_rc  # the step's exit is the load's, as under the run's own loads: the close's is the network attester's verdict
  elif [ "$LOAD" != front ]; then
    release=$("$PY" -c 'import uuid; print(uuid.uuid4().hex)')
    local load=("$TREE/adapters/vllm/tools/served_run/serve_load.py" --model "$REPO" --varied "$REQUESTS" --rate "$RATE"
      --max-tokens "$MAX_TOKENS" --prompt-range "$PROMPTS" --seed "$SEED")
    [ "$LOAD" = dataset ] && load=("$TREE/adapters/vllm/tools/served_run/inputs.py" send --dataset "$DATASET" --timing "$TIMING")
    "$PY" "${load[@]}" --port "$PORT" --release "$release" --out "$D/load.jsonl" > "$D/load.log" 2>&1
    rc=$?
    local sent all
    sent=$(tail -1 "$D/load.log" | grep -oE '[0-9]+/[0-9]+ ok' | head -1) all=${sent#*/}
    keep "REQUESTS: $sent"
    say "REQUESTS (exit $rc, release run $release): $([ -n "$sent" ] && [ "${sent%%/*}" = "${all% ok}" ] && echo "every request answered" \
      || echo "not every request answered (the serve dir's load.log, on the node)")"
    [ -f "$D/load.timings.json" ] || say "TIMINGS: no $D/load.timings.json"
  fi
  c=$(count "$D/windows.jsonl"); s=$(count "$D/seals.jsonl")
  # a front that failed before a window closed sent the run's only requests, so none will; a session's load likewise
  if { [ "$LOAD" != front ] && [ "$LOAD" != session ]; } || [ "$rc" = 0 ] || [ "$c" -gt 0 ]; then
    for _ in $(seq 1 "$polls"); do     # every closed window's seal line
      c=$(count "$D/windows.jsonl"); s=$(count "$D/seals.jsonl")
      [ "$c" -gt 0 ] && [ "$s" -ge "$c" ] && break
      sleep 2
    done
  fi
  trap - TERM INT
  kill -TERM "$srv" 2>/dev/null; wait "$srv"
  local stopped=$?
  keep "SERVE stopped (exit $stopped): $c window(s) closed, $s sealed"
  if [ "$c" -gt 0 ] && [ "$s" -ge "$c" ]; then say "SERVE stopped (exit $stopped): every closed window sealed"
  elif [ "$c" = 0 ]; then say "SERVE stopped (exit $stopped): no window closed"
  else say "SERVE stopped (exit $stopped): not every closed window sealed (the serve dir's private/verdicts.txt)"; fi
  pearl_period close
  # a front's exit is the network attester's verdict (CERTIFIER-FRONT, and its piece reads the session), not the load's
  { [ "$LOAD" = front ] || [ "$rc" = 0 ]; } && [ "$c" -gt 0 ] && [ "$s" -ge "$c" ]
}

circuit() {  # window K's circuit at class $1 within $2 (NAME=LIMIT), its line headed $3: 0 iff it counted
  local c=$1 head=$3 cls=() tag=record pins=() wp=$NO_WEIGHTS f rc
  [ "$c" = record ] || { cls=(--class "$c"); tag=${c//,/-}; }
  # under PEARLPLUS_TIE the verifier's own record of the circuit the window is committed as, cut at Pearl+'s linears
  [ -z "$PEARLPLUS_TIE" ] || [ "$c" = record ] || { cls+=(--pearl-tables "${PEARLPLUS_ARM#pouw:}"); head+=", cut at Pearl+'s linears"; }
  [ -f "$WREG" ] && pins=(--pins "$WREG") wp="at the weights registration's pins"
  f=$D/window-circuit-w$K-$tag.json
  timeout "${2#*=}" "$PY" benchmarks/prove_path/prove_path.py window-circuit --serve-run "$D" --window "$K" \
    --row-dir "$D/windows/w$K" --advice "$D/advice/w$K/private/advice.bin" --out "$f" "${pins[@]}" ${PIN:+--pin "$PIN"} \
    "${cls[@]}" > "$D/window-circuit-$tag.count.json" 2> "$D/window-circuit-$tag.err"
  rc=$?
  if [ "$rc" = 0 ]; then  # the count's `bits` is the advice's length, which stays on the node
    "$PY" -c 'import json, sys; c = json.load(open(sys.argv[1])); c.pop("bits", None); print(json.dumps(c, sort_keys=True))' \
      "$D/window-circuit-$tag.count.json" > "$OUT/window-circuit-$tag.json"
    if [ -n "$PEARLPLUS_TIE" ] && [ "$c" = "$V3C" ] && [ -f "$PEARLPLUS_CIRCUIT" ]; then
      head+=" (the record PEARLPLUS-CIRCUIT built before the serve: $(cmp -s "$f" "$PEARLPLUS_CIRCUIT" && echo the same || echo ANOTHER))"
    fi
    say "$head: $("$PY" -c 'import json, sys; c = json.load(open(sys.argv[1]))["count"]; print("registered", c["registered"], "bits, free", c["free"], "bits of", c["inputs"], "input bits")' \
      "$OUT/window-circuit-$tag.json" 2>&1 | cut -c1-200) $wp"
    keep "$head: record $(sha256sum "$f" | cut -c1-64), $(stat -c %s "$f") bytes"
  elif [ "$rc" = 124 ]; then
    say "$head timed out after $2 (exit 124): no count"
  else
    say "$head refused $(failed "$rc" "$D/window-circuit-$tag.err")"
  fi
  return "$rc"
}
circuits() {  # in TREE: every class of CLASSES gates (0 iff each counted)
  local bad=0 c
  grep -q '"window-circuit"' benchmarks/prove_path/prove_path.py \
    || { say "MISSING V3b's window circuit: this tree's prove_path.py has no window-circuit"; return 1; }
  for c in $CLASSES; do circuit "$c" "WC_TIMEOUT=${WC_TIMEOUT:-30m}" "WINDOW-CIRCUIT class $c" || bad=1; done
  return "$bad"
}
# block_counts.py rebuilds each template from its id through the registry and compares no digest with the record's
BASIS_129="its basis the registry at the names the record carries, not the declared circuit (candidate 129.2)"
counted_pair() {  # class $1's circuit record and the registration counted beside it: the window's record and its
  # registration, which binds it (interface.tied_record), where D holds one; else the circuits step's record at the class
  # beside the advice registration, which binds none and is refused
  local r=$D/windows/w$K/record
  if [ -f "$r/registration.json" ] && [ -f "$r/circuit.json" ]; then echo "$r/circuit.json" "$r/registration.json"
  else echo "$D/window-circuit-w$K-${1//,/-}.json" "$D/advice/w$K/registration.json"; fi
}
block_counts() {  # in TREE: the picked window's (c, k)-inference blocks at each class of CLASSES it counted, judged against
  # the inference-only policy (IO_POLICY) where the run has one: the judgment public, its exact counts in the serve dir's private/
  # its exit is 1 where a class refused (ROAD_WHY the first such line), and ROAD_NOT_RUN says why where nothing was counted
  local c tag f reg pins=() day decl=${IO_POLICY:-$OUT/inference-only-policy.json} rc counted=0 bad=0
  [ -f "$decl" ] || { ROAD_NOT_RUN="the run declares no c, k and n"; say "BLOCK-COUNTS: none, since $ROAD_NOT_RUN"; return 0; }
  [ -f "$WREG" ] && pins=(--pins "$WREG")
  day=$("$PY" -c 'import datetime, json, sys; t = json.load(open(sys.argv[1]))["clocks"]["time"]; print(datetime.datetime.fromtimestamp(t, datetime.timezone.utc).date())' "$D/serve.json") \
    || { ROAD_WHY="BLOCK-COUNTS refused: D/serve.json gives no clocks.time, the day the declaration is read at"; say "$ROAD_WHY"; return 1; }
  (umask 077; mkdir -p "$D/private")
  for c in $CLASSES; do
    tag=${c//,/-}; [ "$c" = record ] && tag=record
    read -r f reg < <(counted_pair "$c")
    [ -f "$f" ] || continue
    counted=1
    "$PY" -m verity_vllm.pipeline.cli block-counts --day "$day" --declaration "$decl" \
      --window "$f" "$reg" "${pins[@]}" --private "$D/private/block-counts-$tag.json" \
      --out "$OUT/block-counts-$tag.json" > /dev/null 2> "$D/block-counts-$tag.err"
    rc=$?
    if [ "$rc" = 0 ]; then
      say "BLOCK-COUNTS class $c, the picked window alone: $(said 0 "$D/block-counts-$tag.err" | cut -c1-400); $BASIS_129"
    else
      bad=1
      [ -n "${ROAD_WHY:-}" ] || ROAD_WHY="BLOCK-COUNTS class $c refused $(failed "$rc" "$D/block-counts-$tag.err")"
      say "BLOCK-COUNTS class $c refused $(failed "$rc" "$D/block-counts-$tag.err")"
    fi
  done
  [ "$counted" = 1 ] || ROAD_NOT_RUN="no class of CLASSES ($CLASSES) has a counted window circuit"
  return "$bad"
}
IO_POLICY=${IO_POLICY:-$OUT/inference-only-policy.json}
IO_LEAN=$TREE/verity/protocols/policy/training_detection/inference_only/verifier/lean/.lake/build/bin/inference-only-verifier
io_class() {  # the inference-only verdict's class: IO_CLASS, else the first of CLASSES but `record`
  local c
  [ -z "${IO_CLASS:-}" ] || { echo "$IO_CLASS"; return 0; }
  for c in $CLASSES; do [ "$c" = record ] || { echo "$c"; return 0; }; done
}
inference_policy() {  # in TREE, before the lease: the policy's (c, k, n) at the class, from the run's settings alone; under
  # PEARLPLUS_TIE in serve(), before serve-openai starts, from serve-phi's φ (pearl_circuit's) cut at Pearl+'s linears
  local c out rc phi=()
  c=$(io_class)
  [ -n "$c" ] || { ROAD_NOT_RUN="CLASSES names no class but record"; say "INFERENCE-ONLY POLICY: none, since $ROAD_NOT_RUN"; return 0; }
  [ -z "$PEARLPLUS_TIE" ] || phi=(--serve "$D/pearlplus/serve-phi.json" --pearl-tables "${PEARLPLUS_ARM#pouw:}")
  [ "${1:-}" != window ] || phi=(--serve "$D/serve.json")  # the window's serve's own φ, after it served
  out=$(CUDA_VISIBLE_DEVICES='' "$PY" benchmarks/one_stage/inference_only.py policy --class "$c" --model "$REPO" \
    --revision "$REV" --lease-min "${LEASE_MIN:-60}" --tree "$(tree_sha)" "${phi[@]}" --out "$IO_POLICY" \
    2> "$D/inference-only-policy.err")
  rc=$?
  say "${out:-INFERENCE-ONLY POLICY failed $(failed "$rc" "$D/inference-only-policy.err")}"
  [ "$rc" != 0 ] || limits
  return "$rc"
}
# The limits of the policy's rate rule (verity/inference-only/v0's α and Γ, which the composite's inference-only case is
# checked at), fixed with the policy, so before serve-openai starts: the run's ALPHA and GAMMA where both are set, else
# α = k/c and Γ = c of the policy's own c and k, the class's public numbers, so in the unit the policy counts c in:
# Boolean gates by its gate table (the count rule's unit; the run's own are in LIMITS_UNIT, which it states, with what
# they prove where LIMITS_PROVE says it: the file's `proves` and the line, in the run's own words). They go to
# IO_LIMITS beside the policy, with the count rule's (c, k, n) and the policy file's SHA-512, and the deferral, the
# compose hand-off and the composite take them as ALPHA and GAMMA (`fixed_limits`), so no later step reads them off the
# served window's division. The developer writes them, and no auditor's device signs them.
# Beside them, where the run sets both HELD_ALPHA and HELD_GAMMA, the file's `held` and one clause of the line state a
# second pair, the record form's: a block a slot, θ the weights the serve's first registration record registers, so a
# registered weight's bit is not fresh (the class's numbers alone: slots × k over Γ, and Γ a slot; `rehearsal.sh check`
# holds the two settings to them). No verdict of the run reads that pair, and `fixed_limits` hands on v0's alone. A held
# pair that is no limits (gamma under 1, alpha under 0, or text that is no number) is stated nowhere in the file, and the
# line says so with both values; v0's file is written all the same.
IO_LIMITS=${IO_LIMITS:-$OUT/inference-only-limits.json}
limits() {
  local out
  [ -s "$IO_POLICY" ] || return 0
  out=$("$SAY_PY" - "$IO_POLICY" "$IO_LIMITS" "${ALPHA:-}" "${GAMMA:-}" "${LIMITS_UNIT:-}" "${LIMITS_PROVE:-}" \
    "${HELD_ALPHA:-}" "${HELD_GAMMA:-}" <<'EOF' 2>&1
import hashlib, json, sys
from fractions import Fraction
policy, out, alpha, gamma, unit, proves, held_alpha, held_gamma = sys.argv[1:9]
body = open(policy, "rb").read()
p = json.loads(body)
given = bool(alpha and gamma)
a, g = (Fraction(alpha), int(gamma)) if given else (Fraction(p["k"], p["c"]), int(p["c"]))
if a < 0 or g < 1:
    sys.exit(f"alpha {a} and gamma {g} are no limits")
doc = {"format": "verity-bench/inference-only-limits/v1", "alpha": [a.numerator, a.denominator], "gamma": g,
       "unit": (unit or "not stated") if given else "Boolean gates by the policy's gate table, the unit of its c",
       "count": {x: p.get(x) for x in ("c", "k", "n")},
       "from": "the run's ALPHA and GAMMA" if given else "the policy's c and k (alpha = k/c, gamma = c)",
       "policy_sha512": hashlib.sha512(body).hexdigest(), "written_by": "the developer"}
if given and proves:
    doc["proves"] = proves
held = ""
if held_alpha and held_gamma:
    try:
        ha, hg = Fraction(held_alpha), int(held_gamma)
    except (ValueError, ZeroDivisionError):
        ha = hg = None
    if ha is None or ha < 0 or hg < 1:
        held = (f"; the held alpha {held_alpha} and gamma {held_gamma} are no limits, so the file states no held "
                "pair")
    else:
        doc["held"] = {"alpha": [ha.numerator, ha.denominator], "gamma": hg,
                       "unit": "the record form's, a block a slot: alpha in fresh input bits a gate of a block, gamma "
                               "in a block's gates",
                       "theta": "the weights the serve's first registration record registers "
                                "(deployment-registration.json)"}
        held = (f"; held for the record form, a block a slot and theta the serve's first registration record's "
                f"weights, alpha = {ha.numerator}/{ha.denominator} and gamma = {hg}, which no verdict of the run reads")
with open(out, "w") as f:
    f.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
print(f"alpha = {a.numerator}/{a.denominator} and gamma = {g}, {doc['from']}" + (f" ({proves})" if "proves" in doc else "")
      + held)
EOF
) || { say "INFERENCE-ONLY LIMITS: none written, so the composite's inference-only case is at the run's ALPHA and" \
           "GAMMA, or at what it reads off the served window's division:" \
           "$(echo "$out" | tail -1 | cut -c1-300)"; return 0; }
  say "INFERENCE-ONLY LIMITS $out: fixed with the policy before the serve, written by the developer and signed by no" \
      "auditor's device (inference-only-limits.json)"
}
fixed_limits() {  # ALPHA and GAMMA as the run fixed them before its serve (IO_LIMITS), where it did
  local got
  [ -s "$IO_LIMITS" ] || return 0
  got=$("$SAY_PY" -c 'import json, sys
d = json.load(open(sys.argv[1]))
print("%d/%d %d" % (d["alpha"][0], d["alpha"][1], d["gamma"]))' "$IO_LIMITS") || return 1
  export ALPHA=${got% *} GAMMA=${got#* }
}
inference_only() {  # in TREE, once step 3 counted window K's circuit at the class: its verdict against the policy
  local c out rc f reg lean=()
  c=$(io_class)
  [ -n "$c" ] || { ROAD_NOT_RUN="CLASSES names no class but record"; return 0; }
  read -r f reg < <(counted_pair "$c")
  [ -x "$IO_LEAN" ] && lean=(--lean "$IO_LEAN" --lean-timeout "${IO_LEAN_TIMEOUT:-1800}")
  out=$(CUDA_VISIBLE_DEVICES='' "$PY" benchmarks/one_stage/inference_only.py verdict --policy "$IO_POLICY" --serve-run "$D" \
    --circuit "$f" --registration "$reg" --pins "$WREG" \
    "${lean[@]}" --out "$OUT/record/inference-only.json" --private "$D/private/inference-only.json" 2> "$D/inference-only.err")
  rc=$?
  if [ -n "$out" ]; then say "$out; $BASIS_129"; else say "INFERENCE-ONLY failed (exit $rc): its log in the serve dir, on the node"; fi
  # a verdict that refuses is decided; one the count could not reach is undecided, though its exit is 0
  [[ "$out" != "INFERENCE-ONLY refuse: undecided: "* ]] || ROAD_WHY=$out
  return "$rc"
}
pous_pins_ok() {  # preflight: this tree's PoUS piece pins the checkpoint the run serves, asked of the piece itself
  # (ONSITE_PIECES' onsite_pous_pins), which else refuses once the window is served, tied and its Leans built
  case ${POUS_CMD:-} in *benchmarks/pous/served_piece.sh*) ;; *) return 0 ;; esac
  declare -F onsite_pous_pins > /dev/null || return 0
  local why rc
  why=$(cd "$TREE" && onsite_pous_pins "$REPO" "$REV" 2>&1)
  rc=$?
  case $rc in
    0) return 0 ;;
    3) say "PREFLIGHT: PoUS's piece answers no period of $REPO@$REV, the checkpoint this run serves: $(tail -n 1 <<< "$why" | cut -c1-300). Serve a row whose checkpoint the piece pins (ROW, REPO and REV), run a tree whose piece pins this one, or set POUS_CMD empty, which runs no PoUS piece and leaves its row missing" ;;
    *) say "PREFLIGHT: PoUS's piece can't be asked whether it pins $REPO@$REV, the checkpoint this run serves (onsite_pous_pins $(failed "$rc" - <<< "$why")): the piece would not start on this node either. Mend what it names, or set POUS_CMD empty, which runs no PoUS piece and leaves its row missing" ;;
  esac
  return 3
}
front_ok() {  # preflight: LOAD=front has a front, LOAD=session its three commands and no other LOAD any of a session's,
  # and each flag CERTIFIER_FRONT passes to the tree's front, it takes
  [ "$LOAD" != front ] || [ -n "${CERTIFIER_FRONT:-}" ] \
    || { say "PREFLIGHT: LOAD=front sends the front's group alone, and CERTIFIER_FRONT is unset"; return 3; }
  local v session=
  case " ${CERTIFIER_FRONT:-} " in *" --session "*) session=1 ;; esac
  if [ "$LOAD" = session ]; then
    for v in CERTIFIER_FRONT LOAD_CMD CERTIFIER_CLOSE; do
      [ -n "${!v:-}" ] \
        || { say "PREFLIGHT: LOAD=session is the caller's load through a relay session, which CERTIFIER_FRONT opens (attester_piece.py front with --session), LOAD_CMD sends to and CERTIFIER_CLOSE closes (attester_piece.py close), and $v is unset"; return 3; }
    done
    [ -n "$session" ] \
      || { say "PREFLIGHT: LOAD=session sends the caller's load through a relay session, and CERTIFIER_FRONT passes no --session, so its front would send a group of its own"; return 3; }
  else
    for v in LOAD_CMD CERTIFIER_CLOSE; do
      [ -z "${!v:-}" ] || { say "PREFLIGHT: $v is LOAD=session's, and LOAD is $LOAD: unset it"; return 3; }
    done
    [ -z "$session" ] \
      || { say "PREFLIGHT: CERTIFIER_FRONT passes --session, which opens a relay session for a caller's load, and LOAD is $LOAD, which sends none through it: LOAD=session, or a front without --session"; return 3; }
  fi
  case ${CERTIFIER_FRONT:-} in *"attester_piece.py front"*) ;; *) return 0 ;; esac
  case " ${SERVE_ARGS:-} " in *" --protocol ids "*) ;; *)
    say "PREFLIGHT: the network attester front sends id-form requests, which serve-openai answers only under" \
        "SERVE_ARGS='--protocol ids --id-payload 503'"; return 3 ;; esac
  local help f script
  script=$(grep -oE "[^[:space:]\"']*attester_piece\.py front" <<< "$CERTIFIER_FRONT" | head -1)
  help=$(cd "$TREE" && "$PY" "${script% front}" front --help 2>&1) \
    || { rc=$?; say "PREFLIGHT: this tree's attester front doesn't start $(failed "$rc" - <<< "$help")"; return 3; }
  for f in $(grep -oE -- '--[a-z][a-z0-9-]*' <<< "$CERTIFIER_FRONT" | sort -u); do
    grep -qE "(^|[^a-z0-9-])$f([^a-z0-9-]|$)" <<< "$help" \
      || { say "PREFLIGHT: CERTIFIER_FRONT passes $f, which this tree's attester front doesn't take"; return 3; }
  done
}
# network attestation's chain files list and count the window's requests (its certs, its tie's prompt lengths, its N2 network attester's
# counts, its report and the piece's log, or the split's two logs); once the record has read them, or
# `served_verdicts.sh v3c` is done, they move to the serve dir's private/
CERTIFIER_PRIVATE="chain/certs.json chain/report chain/tie/summary.json chain/n2/summary.json piece.log chain.log period.log"
private_pieces() {
  local f
  for f in $CERTIFIER_PRIVATE; do
    [ -e "$OUT/pieces/certifier/$f" ] || continue
    (umask 077; mkdir -p "$D/private/pieces/certifier/$(dirname "$f")") \
      && mv "$OUT/pieces/certifier/$f" "$D/private/pieces/certifier/$f"
  done
}
# What the follow-on of step 6's deferral reads, relative to the serve dir. V3c's: the weights registration, the verifier's
# window-circuit record at the class (step 3's, `v3c_record`), windows.jsonl (the run root), the tie's window-commit answer, and the commit's window dir (circuit.json, registration.json,
# every value file). The record's: seals.jsonl, windows.jsonl and requests.jsonl (the pick's steps), step 3's records
# audit's summary, audit record, registration and runs (which it reads where V3c's audit writes no summary:
# `sampled_of_record`), the pieces it read, and the inference-only verdict it carries. The
# composite's: PoUW's period case, Pearl+'s case's parts and blobs (served_pearlplus.sh's step 4c), and the serve's first
# registration record (deployment-registration.json), which Pearl+'s W is chained after. Under DAY, the
# serve's declaration.json and the deferral's copy of the day's log and parameters. And deferred.env, the settings the
# follow-on runs at. Each is listed where it exists.
deferred_inputs() {
  {
    printf '%s\n' weights_registration.json deployment-registration.json "$(v3c_record)" windows.jsonl seals.jsonl \
      requests.jsonl declaration.json \
      "$SERVED_ARM/summary.json" "$SERVED_ARM/window/audit.json" \
      "$SERVED_ARM/window/registration.json" "$SERVED_ARM/runs.json" \
      private/deferred/inference-only.json "pearlplus/w$K/registration.json" \
      "pearlplus/w$K/window-commit.json" served-compare/served_compare.json
    (cd "$D" && find "v3c-w$K-$vtag/window" private/deferred/pieces private/deferred/deferred.env private/deferred/day-log.json \
      private/deferred/day-params.json private/pearlplus/case -type f 2>/dev/null)
  } | (cd "$D" && while IFS= read -r f; do [ ! -f "$f" ] || echo "$f"; done) | LC_ALL=C sort -u
}
HANDED_WHAT="the final record$([ -z "${COMPOSITE-1}" ] || echo " and the composite")"
if [ "$COMPOSE_OFF_NODE" = 1 ]; then DEFERRED_WHAT="V3c's audit"
else DEFERRED_WHAT="V3c's audit$([ -z "${COMPOSITE-1}" ] || echo " and the composite")"; fi
defer() {  # step 6 on a node short of V3c's audit's memory ($1, why), left to a follow-on: the pieces the record read
  # copied to the serve dir's private/deferred, the settings it runs at (deferred.env), and the follow-on's every input
  # listed (inputs.txt, what a prover-side copy takes) and hashed (inputs.sha256, what the follow-on holds the copy to)
  (umask 077; rm -rf "$DEFER_DIR" && mkdir -p "$DEFER_DIR/pieces") || return 1
  [ ! -d "$OUT/pieces" ] || cp -a "$OUT/pieces/." "$DEFER_DIR/pieces/" || return 1
  [ ! -f "$OUT/record/inference-only.json" ] || cp "$OUT/record/inference-only.json" "$DEFER_DIR/inference-only.json" || return 1
  if [ -n "${DAY:-}" ]; then
    cp "$DAY_LOG" "$DEFER_DIR/day-log.json" || return 1
    [ -z "${DAY_PARAMS:-}" ] || cp "$DAY_PARAMS" "$DEFER_DIR/day-params.json" || return 1
  fi
  fixed_limits || return 1
  printf '%s=%q\n' K "$K" PRODUCT "$PRODUCT" V3C_CLASS "$V3C" V3C_LAW "$V3C_LAW" V3C_CHALLENGER "$V3C_CHALLENGER" \
    DEVELOPMENT "$DEVELOPMENT" PEARLPLUS_TIE "$PEARLPLUS_TIE" COMPOSITE "${COMPOSITE-1}" ALPHA "${ALPHA:-}" GAMMA "${GAMMA:-}" \
    SERVED_NODE "$(hostname)" SERVED_PEER "$V3C_PEER" DEFERRED_TREE "$(tree_sha)" COMPOSE_OFF_NODE "$COMPOSE_OFF_NODE" \
    COMPOSE_NODE "$COMPOSE_NODE" DAY "${DAY:-}" PEARLPLUS_TIE_UNIFORM "${PEARLPLUS_TIE_UNIFORM:-}" > "$DEFER_DIR/deferred.env" || return 1
  deferred_inputs > "$DEFER_DIR/inputs.txt"
  (cd "$D" && tr '\n' '\0' < "$DEFER_DIR/inputs.txt" | xargs -0 -r sha256sum) > "$DEFER_DIR/inputs.sha256" || return 1
  echo private/deferred/inputs.sha256 >> "$DEFER_DIR/inputs.txt"
  if [ "$COMPOSE_OFF_NODE" = 1 ]; then
    say "DEFERRED $DEFERRED_WHAT: $1. A follow-on on a node with the memory runs it there and writes the compose hand-off," \
        "from the serve dir's audit inputs copied prover-side, outside any run dir, each file its private/deferred/inputs.txt" \
        "lists: D=<that path> bash benchmarks/one_stage/served_verdicts.sh v3c; $HANDED_WHAT are composed off both nodes, on" \
        "COMPOSE_NODE, from that hand-off"
  else
    say "DEFERRED $DEFERRED_WHAT: $1. A follow-on on a node with the memory runs them and composes the final record there," \
        "from the serve dir's audit inputs copied prover-side, outside any run dir, each file its private/deferred/inputs.txt" \
        "lists: D=<that path> bash benchmarks/one_stage/served_verdicts.sh v3c"
  fi
}
day_held() {  # a follow-on under DAY: the deferral's copy of the day's log, which core's check accepts, holds the
  # declaration the serve's declaration.json names by digest, of day DAY, so V3c's audit runs R2 against the day the serve declared
  local got
  got=$("$TREE/.venv/bin/python" -c 'import json, sys
from verity import device as S
from verity.protocols.registration.verifier import registration as G
decl, day = json.load(open(sys.argv[1])), int(sys.argv[3])
composite, log = G.log_of(json.load(open(sys.argv[2])))
p = G.params_of(json.load(open(sys.argv[4]))) if len(sys.argv) > 4 else (composite, S.Auditor())
got = G.check_log(p, (composite, log))
if got[0] != "accept":
    sys.exit(f"core check refuses it, {got[1]}")
want = decl.get("declaration")
if decl.get("day") != day or not any(isinstance(e, G.Declaration) and e.day == day
                                     and G.Declaration(e.day, e.program, e.blocks).digest() == want for e in log):
    sys.exit(f"it holds no declaration of day {day} with digest {str(want)[:16]}…")
program = str(decl.get("program"))
print(f"day {day}, program {program[:16]}…, declaration {want[:16]}…")' \
    "$D/declaration.json" "$DEFER_DIR/day-log.json" "$DAY" ${DAY_PARAMS:+"$DAY_PARAMS"} 2>&1)
  local rc=$?
  say "DAY-HELD: the serve dir's private/deferred/day-log.json, the day's registration log as the deferral copied it," \
      "holds the declaration the serve's declaration.json names: $([ "$rc" = 0 ] && echo "PASS ($got)" || echo "FAIL ($(echo "$got" | tail -1 | cut -c1-300))")"
  return "$rc"
}
follow() {  # `served_verdicts.sh v3c` on a deferral's copy of the serve dir: the copy held to the served node's sums, then
  # step 6's audit, and the composite, a development check on this prover node, and the final record here rather than on
  # the served node, the record pending, or under COMPOSE_OFF_NODE=1 the compose hand-off for a node of neither
  local rc vrc rec comp=0 TPY=$TREE/.venv/bin/python
  (cd "$D" && sha256sum --check --strict --quiet private/deferred/inputs.sha256) > "$DEFER_DIR/check.log" 2>&1
  rc=$?
  say "FOLLOW-ON on a prover node, $NODE_MEM_GB GB: $DEFERRED_WHAT, which the served node (SERVED_NODE) deferred, from its serve" \
      "dir's audit inputs copied here prover-side, each as it hashed them (the serve dir's private/deferred/inputs.sha256):" \
      "$([ "$rc" = 0 ] && echo PASS || echo "FAIL (the serve dir's private/deferred/check.log)")"
  [ "$rc" = 0 ] || return 3
  wait "$BUILD"; rc=$?
  [ "$rc" = 0 ] || { echo "BUILD $(failed "$rc" "$LOGS/build.log")"; stop build "$rc"; }
  [ -z "${DAY:-}" ] || day_held || return 3
  local sha
  sha=$(tree_sha)
  say "WINDOW-CIRCUIT class $V3C: the served node's record at the class (step 3's), from its tree's commit $DEFERRED_TREE" \
      "$([ "$sha" = "$DEFERRED_TREE" ] && echo "(this tree's)" || echo "(this tree is $sha: FOLLOW_ANY_TREE)");" \
      "this node holds no row dir to count it from"
  v3c_commit && v3c_proved; vrc=$?
  local wt=()
  [ -z "$PEARLPLUS_TIE" ] || { window_tie; wt=(--window-tie "$(pearl_dir)/tie" --served-compare "$SERVED_COMPARE"); }
  if [ "$COMPOSE_OFF_NODE" = 1 ]; then
    handoff "$DEFER_DIR/pieces" "$DEFER_DIR/inference-only.json"; rc=$?
    echo "---- verdicts"; cat "$OUT/verdicts.txt"
    [ "$vrc" = 0 ] && [ "$rc" = 0 ] || return 1
    return 2  # no record here: it is composed off this node
  fi
  if [ -f "$DEFER_DIR/inference-only.json" ]; then
    mkdir -p "$OUT/record" && cp "$DEFER_DIR/inference-only.json" "$OUT/record/inference-only.json" \
      && say "INFERENCE-ONLY: the served node's verdict, carried into this record as it wrote it (the serve dir's private/deferred/inference-only.json)"
  fi
  local lean=()
  if [ -n "${COMPOSITE-1}" ]; then
    local why="composed by a deferral's follow-on on a prover node, which the prover controls" how=COMPOSE_ON_NODE
    if hold_placement; then
      why+=": a development check, no claim of record (only the compose follow-on's composite, off the served node on an"
      why+=" auditor device, is the record's)" how=COMPOSE_DEVELOPMENT
    fi
    TPY=$TPY SERVED_NODE=$SERVED_NODE env "$how=$why" bash benchmarks/one_stage/served_composite.sh "$D" "$OUT" \
      || comp=1
    lean=(--composite "$OUT/composite")
  fi
  "$TPY" benchmarks/one_stage/served_verdicts.py record --serve-run "$D" --window "$K" --sampled-proofs "$D/$(sampled_of_record)" \
    --pieces "$DEFER_DIR/pieces" --out "$OUT/record" --product "$PRODUCT" --node "$SERVED_NODE" ${wt[@]+"${wt[@]}"} \
    --key-sha512 "$H1_KEY_SHA512" --lane "${PEARLC_CHALLENGER_LANE:-${H1_LANE:-vllm}}" ${lean[@]+"${lean[@]}"} \
    --deploy-manifest "$DEPLOY_MANIFEST" 2>&1 | publish_lines
  rec=${PIPESTATUS[0]}
  echo "---- verdicts"; cat "$OUT/verdicts.txt"
  [ "$vrc" = 0 ] && [ "$comp" = 0 ] || return 1
  return "$rec"
}
# The record's SPOT audit, relative to the serve dir: V3c's of window K where it wrote its summary, the
# SPOT audit of record, on V3C_CHALLENGER's coins (step 3's are local while those are live) and, under a tie,
# tied to Pearl+'s W; else step 3's records audit
sampled_of_record() {
  if [ -f "$D/v3c-w$K-$vtag/audit/summary.json" ]; then echo "v3c-w$K-$vtag/audit"; else echo "$SERVED_ARM"; fi
}
# The compose hand-off (COMPOSE_OFF_NODE=1, the default): what the final record and the composite read, relative to the
# serve dir, and nothing else. The record's: its SPOT audit's (`sampled_of_record`) summary, audit record,
# registration and runs, and of each piece the files `piece_row` and `piece_server` read (copied to
# private/compose/pieces), Pearl+'s verdict of record's among them (`pearlplus_of_record`'s: the auditor's ledger, the
# coin record, Pearl+'s answer, the Lean verdict and each challenger session's public key, out/key.json at its path, and
# no other file of a session). The composite's: each records audit's cases ($SERVED_ARM's and V3c's
# window/case.json, case-v2.json and their error files), the files a case names as its blobs (the circuit, the public
# file, each session's record and proofs, the lookup tables), the serve's first
# registration record where it wrote one, each window's Pearl+ record W and audit's record R that the serve dir holds
# (`served_cases.py`'s RECORDS: every record a window's log can name, so the inference-only case's log is walked from
# the copy alone, with no part of Pearl+'s case), and PoUW's period case.
# And compose.env, its settings: the window, its engine steps (the record's, picked here from the requests file, which
# stays), the product, the composite, α and Γ, the served node, the follow-on node COMPOSE_NODE, the tree's commit, the
# host V3c's audit ran on (AUDITED_ON, empty where the hand-off carries none) and the record's audit (SAMPLED_AUDIT).
# Every one is a verifier's input: `public_files.py handoff` refuses a list holding any file its ``D/`` entries don't
# name, and a blob outside the serve dir is refused too.
COMPOSE_PIECE_FILES="answer.json development.json piece.json refused.txt served.json runs.json ledger.jsonl coin.json
pearlplus-answer.json pearlplus-lean.json"
handoff_inputs() {  # the hand-off's every input but its own sums, each existing file once, the record's audit $1's among them
  local a blobs cases=()
  for a in "$SERVED_ARM" "v3c-w$K-$vtag/audit"; do
    [ -f "$D/$a/window/case-v2.json" ] && cases+=("$D/$a/window/case-v2.json")
  done
  blobs=$([ "${#cases[@]}" = 0 ] || "$SAY_PY" - "$D" "${cases[@]}" <<'EOF'
import json, os, sys
d = os.path.abspath(sys.argv[1])
for case in sys.argv[2:]:
    for b in json.load(open(case)).get("blobs") or ():
        r = os.path.relpath(os.path.abspath(b), d)
        if r.startswith(".."):
            sys.exit(f"{case} names a blob outside the serve dir: {b}")
        print(r)
EOF
) || return 1
  {
    printf '%s\n' "$1/summary.json" "$1/window/audit.json" "$1/window/registration.json" "$1/runs.json" \
      deployment-registration.json \
      "pearlplus/w$K/tie/window/registration.json" \
      "pearlplus/w$K/tie/summary.json" "pearlplus/w$K/tie/circuit-record.json" served-compare/served_compare.json \
      private/front-window
    for a in "$SERVED_ARM" "v3c-w$K-$vtag/audit"; do
      printf "$a/window/%s\n" case.json case-v2.json case-error.txt case-v2-error.txt
    done
    [ -z "$blobs" ] || printf '%s\n' "$blobs"
    (cd "$D" && printf '%s\n' pearlplus/w*/registration.json v3c-w*/audit/window/registration.json)
    for f in pieces compose.env inference-only.json; do
      [ ! -e "$D/private/compose/$f" ] || (cd "$D" && find "private/compose/$f" -type f)
    done
  } | (cd "$D" && while IFS= read -r f; do [ ! -f "$f" ] || echo "$f"; done) | LC_ALL=C sort -u
}
handoff() {  # the compose hand-off in private/compose, from the pieces in $1 and the inference-only verdict $2
  local d f steps why rc sampled
  sampled=$(sampled_of_record)
  [ "$sampled" = "$SERVED_ARM" ] || coin_trees_held "$V3C_CHALLENGER" "$D/$sampled/summary.json" HANDOFF || return 20
  (umask 077; rm -rf "$COMPOSE_DIR" && mkdir -p "$COMPOSE_DIR/pieces") || return 1
  for d in "$1"/*/; do
    [ -d "$d" ] || continue
    for f in $COMPOSE_PIECE_FILES $(cd "$d" && find session -mindepth 3 -maxdepth 3 -path 'session/*/out/key.json' -type f 2>/dev/null | LC_ALL=C sort); do
      [ ! -f "$d$f" ] || { mkdir -p "$COMPOSE_DIR/pieces/$(basename "$d")/$(dirname "$f")" \
        && cp "$d$f" "$COMPOSE_DIR/pieces/$(basename "$d")/$f"; } || return 1
    done
  done
  [ ! -f "$2" ] || cp "$2" "$COMPOSE_DIR/inference-only.json" || return 1
  steps=$("$SAY_PY" "$SV" window "$D" --slots 1073741824 --window "$K" 2> "$COMPOSE_DIR/window.err" \
    | "$SAY_PY" -c 'import json, sys; print(",".join(map(str, json.load(sys.stdin)["steps"])))') \
    || { say "HANDOFF refused: the window has no engine steps to hand the record (the serve dir's private/compose/window.err)"; return 1; }
  fixed_limits || return 1
  printf '%s=%q\n' K "$K" PRODUCT "$PRODUCT" POUW "$POUW" PEARLPLUS_TIE "$PEARLPLUS_TIE" COMPOSITE "${COMPOSITE-1}" ALPHA "${ALPHA:-}" GAMMA "${GAMMA:-}" \
    SERVED_NODE "${SERVED_NODE:-$(hostname)}" SERVED_DIR "$D" COMPOSE_NODE "$COMPOSE_NODE" COMPOSE_TREE "$(tree_sha)" \
    AUDITED_ON "${V3C_AUDITED:-}" SAMPLED_AUDIT "$sampled" STEPS "$steps" > "$COMPOSE_DIR/compose.env" || return 1
  why=$(read_env "$COMPOSE_DIR/compose.env" "$COMPOSES" || echo "$ENV_REFUSED")
  [ -z "$why" ] || { say "HANDOFF refused: COMPOSE_NODE reads its settings (private/compose/compose.env) as data, and $why"; return 1; }
  handoff_inputs "$sampled" > "$COMPOSE_DIR/inputs.txt" 2> "$COMPOSE_DIR/inputs.err"
  rc=$?
  [ "$rc" = 0 ] || { say "HANDOFF refused $(failed "$rc" "$COMPOSE_DIR/inputs.err")"; return 1; }
  why=$("$SAY_PY" "${SV%/*}/public_files.py" handoff < "$COMPOSE_DIR/inputs.txt") \
    || { say "HANDOFF refused: it would carry a file the list doesn't name ($(echo "$why" | head -1 | cut -c1-400))"; return 1; }
  (cd "$D" && tr '\n' '\0' < "$COMPOSE_DIR/inputs.txt" | xargs -0 -r sha256sum) > "$COMPOSE_DIR/inputs.sha256" || return 1
  echo private/compose/inputs.sha256 >> "$COMPOSE_DIR/inputs.txt"
  say "HANDOFF $HANDED_WHAT, composed off this node: on COMPOSE_NODE, from the serve dir's verifier-side" \
      "files and no witness, each file its private/compose/inputs.txt lists, copied to the same path there:" \
      "D=<that path> bash benchmarks/one_stage/served_verdicts.sh compose"
}
compose() {  # `served_verdicts.sh compose` on a copy of a compose hand-off: the copy held to the served node's sums, then
  # the composite and the final record from it, here, off the served node
  local rc rec comp=0 TPY=$TREE/.venv/bin/python
  (cd "$D" && sha256sum --check --strict --quiet private/compose/inputs.sha256) > "$COMPOSE_DIR/check.log" 2>&1
  rc=$?
  say "COMPOSE on COMPOSE_NODE: $HANDED_WHAT of the served node, from its compose" \
      "hand-off (verifier-side files, no witness; $([ -n "${AUDITED_ON:-}" ] && echo "V3c's audit from its follow-on node" || echo "no V3c audit");" \
      "the record's sampled proofs ${SAMPLED_AUDIT:-$SERVED_ARM}), each file as it" \
      "hashed it (the serve dir's private/compose/inputs.sha256):" \
      "$([ "$rc" = 0 ] && echo PASS || echo "FAIL (the serve dir's private/compose/check.log)")"
  [ "$rc" = 0 ] || return 3
  # the cases' blobs, each a file the copy was held to, at its path here (the same where the copy is at the served path)
  "$SAY_PY" "$SV" rebase --held "$D" "$SERVED_DIR" < <(cut -d' ' -f3- "$COMPOSE_DIR/inputs.sha256") > "$COMPOSE_DIR/rebase.log" 2>&1 \
    || { rc=$?; say "COMPOSE refused: the cases' blobs don't rebase from the served node's serve dir to this copy $(failed "$rc" "$COMPOSE_DIR/rebase.log")"; return 3; }
  wait "$BUILD"; rc=$?
  [ "$rc" = 0 ] || { echo "BUILD $(failed "$rc" "$LOGS/build.log")"; stop build "$rc"; }
  if [ -f "$COMPOSE_DIR/inference-only.json" ]; then
    mkdir -p "$OUT/record" && cp "$COMPOSE_DIR/inference-only.json" "$OUT/record/inference-only.json" \
      && say "INFERENCE-ONLY: the served node's verdict, carried into this record as it wrote it (the serve dir's private/compose/inference-only.json)"
  fi
  local lean=()
  if [ -n "${COMPOSITE-1}" ]; then
    TPY=$TPY SERVED_NODE=$SERVED_NODE COMPOSE_HANDOFF=1 bash benchmarks/one_stage/served_composite.sh "$D" "$OUT" || comp=1
    lean=(--composite "$OUT/composite")
  fi
  "$TPY" benchmarks/one_stage/served_verdicts.py record --serve-run "$D" --window "$K" \
    --sampled-proofs "$D/${SAMPLED_AUDIT:-$SERVED_ARM}" \
    --pieces "$COMPOSE_DIR/pieces" --out "$OUT/record" --product "$PRODUCT" --node "$SERVED_NODE" --steps "$STEPS" \
    ${PEARLPLUS_TIE:+--window-tie "$(pearl_dir)/tie" --served-compare "$SERVED_COMPARE"} \
    --key-sha512 "$H1_KEY_SHA512" --lane "${PEARLC_CHALLENGER_LANE:-${H1_LANE:-vllm}}" --off-node --sampled-lean \
    --deploy-manifest "$DEPLOY_MANIFEST" \
    ${SAMPLED_EXE:+--sampled-exe "$SAMPLED_EXE"} --tree "$(tree_sha)" ${COMPOSE_TREE_ON:+--tree-on "$COMPOSE_TREE_ON"} \
    ${lean[@]+"${lean[@]}"} 2>&1 | publish_lines
  rec=${PIPESTATUS[0]}
  echo "---- verdicts"; cat "$OUT/verdicts.txt"
  [ "$comp" = 0 ] || return 1
  return "$rec"
}
compose_build() {  # what the compose follow-on runs: the tree's venv, and lake for the cases' and the product's Lean
  if [ -n "${COMPOSE_BUILT:-}" ]; then  # the compose gate's tree, built at its pin: nothing fetched or built here
    [ -x "$TREE/.venv/bin/python" ] || { echo "COMPOSE_BUILT, and the tree has no venv ($TREE/.venv)"; return 5; }
    return 0
  fi
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
  uv sync --all-packages --extra torch-cpu || return 5
}
copy() {  # `served_verdicts.sh copy deferred|compose HOST[:DIR]`: each file the list names, and nothing else, to the same path
  # on HOST, or under DIR/<the serve dir's name>/ in the directory HOST's rsync writes to (a write-only rsync's, no remote mkdir)
  local kind=$1 host=${2%%:*} name sub where dir served composer n rc why log=$D/private/copy-$1.log
  case $kind in
    deferred) dir=$DEFER_DIR ;;
    compose) dir=$COMPOSE_DIR ;;
    *) say "COPY refused: served_verdicts.sh copy deferred|compose HOST[:DIR], not $kind"; return 3 ;;
  esac
  [ -f "$dir/inputs.txt" ] && [ -f "$dir/$kind.env" ] \
    || { say "COPY refused: the serve dir D holds no $kind list (private/$kind/inputs.txt and $kind.env)"; return 3; }
  # shellcheck source=/dev/null
  IFS='|' read -r served composer < <(set +u; . "$dir/$kind.env"; echo "${SERVED_NODE%%.*}|${COMPOSE_NODE%%.*}")
  host=${host#*@}
  if [ "$kind" = deferred ]; then
    for n in "$composer" "${COMPOSE_NODE%%.*}"; do
      [ -z "$n" ] || [ "${host%%.*}" != "$n" ] \
        || { say "COPY refused: the deferral's list holds the window's witness, and its node is COMPOSE_NODE, an auditor device where no witness goes: copy it to a prover node"; return 3; }
    done
  else
    [ "${host%%.*}" != "$served" ] \
      || { say "COPY refused: its node is the served node, which the prover controls: copy the hand-off to COMPOSE_NODE"; return 3; }
    why=$("$SAY_PY" "${SV%/*}/public_files.py" handoff < "$dir/inputs.txt") \
      || { say "COPY refused: the hand-off's list names a file the list doesn't ($(echo "$why" | head -1 | cut -c1-400))"; return 3; }
  fi
  n=$(count "$dir/inputs.txt")
  if [[ $2 == *:* ]]; then
    name=${D%/} && name=${name##*/} && sub=${2#*:} && sub=${sub%/} && sub=${sub:+$sub/}$name
    where="under its rsync's directory, which its node moves to the serve dir's path"
    # shellcheck disable=SC2086
    ${RSYNC:-rsync} -a --files-from="$dir/inputs.txt" "$D/" "${2%%:*}:$sub/" > "$log" 2>&1
  else
    where="each to the same path there"
    # shellcheck disable=SC2086
    ${RSYNC:-rsync} -a --files-from="$dir/inputs.txt" --rsync-path="mkdir -p $(printf %q "$D") && rsync" "$D/" "$2:$D/" > "$log" 2>&1
  fi
  rc=$?
  say "COPY the $([ "$kind" = deferred ] && echo "deferral's prover-side" || echo "hand-off's verifier-side") list's $n files to its node," \
      "$where: $([ "$rc" = 0 ] && echo PASS || echo "FAIL (exit $rc, the serve dir's private/copy-$kind.log)")"
  return "$rc"
}
gate_compose() {  # `served_verdicts.sh gate-compose PIECES [VERDICT [HOST]]`: the hand-off from PIECES (and the
  # inference-only VERDICT), sent to the compose gate as a tar of the files its inputs.txt lists and nothing else; its
  # answer in $OUT/compose
  local rc why answer=$OUT/compose/answer.json
  handoff "$1" "${2:-}" || return 1
  why=$("$SAY_PY" "${SV%/*}/public_files.py" handoff < "$COMPOSE_DIR/inputs.txt") \
    || { say "GATE-COMPOSE refused: the hand-off's list names a file the list doesn't ($(echo "$why" | head -1 | cut -c1-400))"; return 3; }
  mkdir -p "$OUT/compose" && rm -f "$answer"
  # shellcheck disable=SC2086
  (cd "$D" && tar --create --dereference --no-recursion --file=- --files-from="$COMPOSE_DIR/inputs.txt") \
    | $COMPOSE_GATE compose > "$answer" 2> "$D/private/gate-compose.err"
  rc=${PIPESTATUS[1]}
  "$SAY_PY" "$SV" composed "$answer" "$OUT/compose" --gate-exit "$rc" --err "$D/private/gate-compose.err" \
    | publish_lines
  return "${PIPESTATUS[0]}"
}
follow_handoff() {  # whether $1 (handoff, gate-compose) may write a deferral's hand-off again: at its settings alone,
  # naming the host V3c's audit ran on (V3C_AUDITED) where the serve dir holds that audit, as $2, its usage, says
  [ -z "$CLASH" ] || { say "${1^^} refused: the hand-off is written at the deferral's settings (the serve dir's private/deferred/deferred.env), and the environment sets another$CLASH: unset it"; return 3; }
  [ -n "$V3C_AUDITED" ] || [ ! -d "$D/v3c-w$K-$vtag/audit" ] \
    || { say "${1^^} refused: the serve dir holds V3c's audit, so the hand-off names the host it ran on: $2"; return 3; }
}

stop() { say "STOP at $1 (exit $2)"; exit "$2"; }
publish() { "$SAY_PY" "$SV" publish "$OUT" --private "$D/private/out" || { echo "PUBLISH failed (exit $?): $OUT may name a window"; return 1; }; }
predict() {
vv predict row --row "$ROW_ID" --out "$PRED" > "$OUT/predict_row.json" 2> "$LOGS/predict_row.err"
rc=$?
[ "$rc" = 0 ] && [ -f "$PRED/predicted.json" ] \
  || { say "PREDICT-ROW $ROW_ID refused $(failed "$rc" "$LOGS/predict_row.err")"; stop predict 17; }
cp "$PRED/predicted.json" "$OUT/"
say "PREDICT-ROW $ROW_ID: $(cut -c1-600 "$OUT/predict_row.json")"
CONFIG=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["config"])' "$PRED/predicted.json")
# the config the Build was predicted from, kept beside it for the served compare (served_compare.py's CONFIG_COPY): the
# path predicted.json names is in this run's tree, which a run's own clone takes with it, and a kept serve dir is
# compared again after that
cp "$CONFIG" "$PRED/checkpoint_config.json" \
  || { say "PREDICT-ROW $ROW_ID: its checkpoint config $CONFIG can't be kept beside predicted.json, where the served compare reads it"; stop predict 17; }
vv predict compare "$ROW" --row "$ROW_ID" --config "$CONFIG" --predicted "$PRED" --out "$OUT/predict_compare.json" \
  > "$LOGS/predict_compare.log" 2>&1
rc=$?
say "PREDICT-COMPARE against the traced Build $ROW (exit $rc): $(grep -E '^(rank|predicted|\[predict\])' "$LOGS/predict_compare.log" | tr '\n' ' ' | cut -c1-1200)"
[ "$rc" = 0 ] || stop predict-compare 18
}
pearl_cubin() {
  ( machine=${RESEARCH_RUN_DIR:+$RESEARCH_RUN_DIR/inputs/${RESEARCH_MACHINE:-}.toml}
    export RESEARCH_RUN_DIR=$D/private/pearlplus-cubin VERITY_BENCH_VENV=${VERITY_BENCH_VENV:-$VENV}
    mkdir -p "$RESEARCH_RUN_DIR/inputs" && cd "$TREE" || exit 1
    [ ! -f "${machine:-}" ] || cp "$machine" "$RESEARCH_RUN_DIR/inputs/" || exit 1
    # shellcheck source=/dev/null
    . benchmarks/harness/env.sh && . benchmarks/harness/setup/pearl_c_sm90.sh \
      && printf '%s\n' "$PEARLC_KERNEL" "$PEARLC_CUBIN" > "$RESEARCH_RUN_DIR/setup.paths" ) > "$LOGS/pearlplus-cubin.log" 2>&1
  rc=$?
  { read -r PEARLC_KERNEL && read -r PEARLC_CUBIN; } < "$D/private/pearlplus-cubin/setup.paths" 2>/dev/null
  [ "$rc" = 0 ] && [ -s "${PEARLC_CUBIN:-}" ] \
    || { say "PEARLPLUS-CUBIN: pearl_c_sm90.sh failed $(failed "$rc" "$LOGS/pearlplus-cubin.log")"; stop pearlplus-cubin 19; }
  export PEARLC_KERNEL PEARLC_CUBIN
  say "PEARLPLUS-CUBIN: pearl_c_sm90.sh built it (sha256 $(sha256sum "$PEARLC_CUBIN" | cut -c1-16)); the window serves $PEARLPLUS_ARM on it, sm_90's BF16 rows $([ "$PEARLC_BF16_ROWS" = 0 ] && echo off || echo on) (PEARLC_BF16_ROWS=$PEARLC_BF16_ROWS)"
}
seal() {
# each sealed window's lines (its number, its bundle's files) go to the serve dir's private/seal.txt; the verdict says only
# whether every one passed, and why not
"$PY" - "$D" "$PRED" "$D/private/seal.txt" <<'EOF' 2>&1 | publish_lines
import json, os, sys
from pathlib import Path
from verity_vllm.pipeline import deferred_rows as DR
d, pred, held = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
predicted = json.loads((pred / "predicted.json").read_text())
b_of_record = (str((pred / "manifest.json").resolve()), predicted["program_digest"])
serve = json.loads((d / "serve.json").read_text())
seals = [json.loads(x) for x in (d / "seals.jsonl").read_text().splitlines() if x.strip()] if (d / "seals.jsonl").exists() else []
sealed_ok = bound_ok = bool(seals)
lines, why, committers, unbound = [], set(), set(), []
if not seals:
    why.add("no window sealed")
if serve.get("scheme") != "rows-sha512":
    why.add("the serve's scheme isn't rows-sha512")
for s in seals:
    k = s["window"]
    if "error" in s:
        lines.append(f"WINDOW w{k}: SEAL FAILED {s['error']}"[:600])
        why.add("a seal failed")
        sealed_ok = bound_ok = False
        continue
    w = d / "windows" / f"w{k}"
    v = json.loads((w / "commit" / "verdict.json").read_text())
    c = json.loads((w / "commit" / "runs.jsonl").read_text().splitlines()[0])["commit"]
    try:
        r = DR.of_record(v["replay_deferred"]["bundles"][0]["path"], str(w))
        opened, store, good = f"VERIFIED (manifest sha256 {r.bundle['manifest_sha256'][:16]}, {len(r.bundle['manifest']['files'])} files)", r.commitment, True
    except Exception as e:  # noqa: BLE001
        opened, store, good = f"REFUSED ({type(e).__name__}: {e})"[:400], None, False
    rr = v.get("run_root") or ""
    good = good and v.get("commit_scheme") == c.get("commit_scheme") == store == "rows-sha512" and len(rr) == 128
    bound = r.ctx["served_window"].get("required_manifest", {}) if good else {}
    b = (bound.get("path"), bound.get("program_digest")) == b_of_record
    # a seal before serve_seal.manifest_refusal recorded the manifest's digest as "digest", always None: its path and
    # Program digest bind it alone
    if "manifest_digest" in bound:
        b &= bound["manifest_digest"] == predicted["manifest_digest"]
        by = f", manifest digest {str(bound['manifest_digest'])[:16]}…"
    elif "digest" in bound and bound["digest"] is None:
        by = " (sealed before the seal recorded the manifest's digest: its path and Program digest bind it)"
        unbound.append(f"w{k}")
    else:
        b, by = False, ", and no manifest digest"
    sealed_ok &= good
    bound_ok &= b
    committers.add(str(c.get("committer")))
    if not good:
        why.add("a sealed window's bundle didn't open as a rows-sha512 commitment with a 64-byte run root")
    if not b:
        why.add("a sealed record binds another Build than predict row's")
    lines.append(f"WINDOW w{k}: commit_scheme {v.get('commit_scheme')}, committer {c.get('committer')}, run root {rr[:16]}… "
                 f"({len(rr) // 2} bytes); bundle {opened}: {'PASS' if good else 'FAIL'}")
    whose = "predict row's Build of record (its predicted.json's)" if b else f"not predict row's {b_of_record[0]}"
    lines.append(f"B-OF-RECORD w{k}: the sealed record binds {bound.get('path')}, program digest "
                 f"{str(bound.get('program_digest'))[:16]}…{by}: {whose}: {'PASS' if b else 'FAIL'}")
ok = sealed_ok and bound_ok and serve.get("scheme") == "rows-sha512"
os.umask(0o077)
held.write_text("".join(x + "\n" for x in lines))
print(f"SEALED: every closed window's row dir committed under rows-sha512 (committer {', '.join(sorted(committers)) or 'none'}) "
      f"and its replay bundle opened: {'PASS' if sealed_ok else 'FAIL'}")
print(f"B-OF-RECORD: every sealed record binds predict row's Build of record ({b_of_record[0]}, program digest "
      f"{b_of_record[1][:16]}…, manifest digest {predicted['manifest_digest'][:16]}…"
      + (f"; {', '.join(unbound)} sealed before the seal recorded the manifest's digest, by path and Program digest alone"
         if unbound else "") + f"): {'PASS' if bound_ok else 'FAIL'}")
print(f"SERVED-SEAL VERDICT: {'PASS' if ok else 'FAIL'} (scheme {serve.get('scheme')}, precision {serve.get('precision')}"
      + (f"; {'; '.join(sorted(why))}" if why else "") + "; each window's lines in the serve dir's private/seal.txt)")
sys.exit(0 if ok else 1)
EOF
[ "${PIPESTATUS[0]}" = 0 ] || stop seal 11
# serve.json (the window's request count and prefill tokens) and the windows' PoUW calls records (their requests) stay
# in the serve dir, where PoUW's piece and the record read them; the period and the window Program are public
# the serve dir's step trace and the load's records stay on the node; what the stock control's table reads is their aggregate,
# in full in the serve dir's private/out/serve once `publish` holds the load's counts and span.
# A front's load is the network certifier's, paced by its buckets, which no stock run's is, and one window's aggregate is its timing.
if [ "$LOAD" = front ]; then
  say "OVERHEAD: none with LOAD=front (the network certifier paces its requests, so no stock control compares)"
elif [ "$LOAD" = session ]; then
  say "OVERHEAD: none with LOAD=session (the relay paces the caller's requests, so no stock control compares)"
else
  "$PY" "$TREE/benchmarks/one_stage/served_overhead.py" metrics "$D" > "$OUT/serve/overhead.json" 2> "$LOGS/overhead.err" \
    || { rc=$?; say "OVERHEAD: no aggregate $(failed "$rc" "$LOGS/overhead.err")"; rm -f "$OUT/serve/overhead.json"; }
fi
}
window() {
# the pick (its number, steps, requests and users, and each window refused and why) stays in the serve dir's private/
TIED_BY=()
[ -z "$PEARLPLUS_TIE" ] || TIED_BY=(--of-load "$D/load.jsonl")
PICK=$("$PY" benchmarks/one_stage/served_verdicts.py window "$D" --slots "$SLOTS" ${WINDOW:+--window "$WINDOW"} \
  ${TIED_BY[@]+"${TIED_BY[@]}"} 2> "$D/private/window.err")
[ -n "$PICK" ] || { say "WINDOW refused: no sealed window qualifies (why, by window, in the serve dir's private/window.err)"; stop window 12; }
(umask 077; echo "$PICK" > "$D/private/window.json")
K=$(echo "$PICK" | "$PY" -c 'import json, sys; print(json.load(sys.stdin)["window"])')
# the execution phase the record is of, for the auditor's step: its seal's SHA-512 alone (the line holds its run root)
mkdir -p "$OUT/serve" && "$PY" -c 'import hashlib, json, sys; s = next(x.rstrip("\n") for x in open(sys.argv[1]) if x.strip() and json.loads(x)["window"] == int(sys.argv[2])); print(json.dumps({"seal_sha512": hashlib.sha512(s.encode()).hexdigest()}))' "$D/seals.jsonl" "$K" > "$OUT/serve/phase.json" \
  || { rm -f "$OUT/serve/phase.json"; say "PHASE refused: window $K's seal isn't published (serve/phase.json not written), so the onsite check has no execution phase to read"; stop window 21; }
say "WINDOW: $(if [ -n "$PEARLPLUS_TIE" ]; then echo "the fixed load's, the one sealed window holding its requests (load.jsonl's) and no other"
  elif [ -n "${WINDOW:-}" ]; then echo "the one WINDOW names"; else echo "the sealed one with the most requests"; fi), none carried across windows, at the class's $SLOTS slots (the pick in the serve dir's private/window.json)"
KN=$K
rm -f "$D/private/front-window"
if [ -n "$PEARLPLUS_TIE" ] && [ -n "${CERTIFIER_FRONT:-}" ]; then  # the network attester's row from its front's own window
  KN=$("$PY" benchmarks/one_stage/served_verdicts.py window "$D" --slots "$SLOTS" --attested 2>> "$D/private/window.err" \
    | "$PY" -c 'import json, sys; print(json.load(sys.stdin)["window"])' 2> /dev/null)
  [ -n "$KN" ] && [ "$KN" != "$K" ] \
    || { say "WINDOW refused: the network attester's front's group is in no one sealed window of its own apart from the fixed load's (why in the serve dir's private/window.err)"; stop window 12; }
  (umask 077; echo "$KN" > "$D/private/front-window")
  say "NETWORK-WINDOW: the network attester's row is over its front's group, the one sealed window of its requests alone, not the tied load's: it stands as its own verdict, its case left out of the composite"
fi
say "CLASSES: the window circuit's ${CLASSES}; network attestation's scan ${CERTIFIER_SCAN:-W63}, the one class the gate serves"
}
pieces() {
TPY=$TREE/.venv/bin/python
"$TPY" benchmarks/one_stage/served_verdicts.py pieces --serve-run "$D" --window "$K" --out "$OUT/pieces" "$@" \
  --product "$PRODUCT" 2>&1 | publish_lines
}
# The window's steps, a plain run's, in the order verity run's stages name them: each on what a window's serve left in
# D, after the lease, serving nothing and reading no ROW: the close's plan (plans.jsonl), its handoff and record (windows/wK/handoff, windows/wK/record), the
# seal and the advice. Each step is in one class: ran (its command ended 0 and decided), refused or undecided (by its own
# first line), or not run (by why: a step the record holds nothing for names what it lacks and whose it is).
RAN=() REFUSED=() NOT_RUN=()
declare -A WHY=()
step_not_run() { NOT_RUN+=("$1"); WHY[$1]=$2; say "$1: not run: $2"; }
step_refused() { REFUSED+=("$1"); WHY[$1]=$2; say "$1: refused: $2"; }
road_step() {  # NAME FN [ARGS]: FN ran where it ended 0 and set neither ROAD_WHY (its refusal or undecided line) nor
  # ROAD_NOT_RUN (why it decided nothing); else refused by ROAD_WHY, or by the first line it said
  local name=$1 n=0 rc first
  shift
  [ ! -f "$OUT/verdicts.txt" ] || n=$(wc -l < "$OUT/verdicts.txt")
  ROAD_WHY='' ROAD_NOT_RUN=''
  "$@"
  rc=$?
  first=$(tail -n +"$((n + 1))" "$OUT/verdicts.txt" 2> /dev/null | head -n 1)
  if [ "$rc" = 0 ] && [ -z "$ROAD_WHY" ] && [ -n "$ROAD_NOT_RUN" ]; then step_not_run "$name" "$ROAD_NOT_RUN"
  elif [ "$rc" = 0 ] && [ -z "$ROAD_WHY" ]; then RAN+=("$name")
  else step_refused "$name" "${ROAD_WHY:-${first:-it said nothing}} (exit $rc)"; fi
}
road_end() {  # the road's exit is 0 only where every step ran
  local sampled="no proof is made, and Lean verifies nothing; no sampled verdict, so no δ or ε: the sampled proofs did not run" s
  [[ " ${RAN[*]:-} " != *" sampled-proofs "* ]] || sampled="this line states no δ or ε of the sampled proofs that ran"
  say "ROAD END (the window's record): ran ${#RAN[@]} ($(IFS=,; echo "${RAN[*]:-}")); refused or undecided ${#REFUSED[@]} ($(IFS=,; echo "${REFUSED[*]:-}")); not run ${#NOT_RUN[@]} ($(IFS=,; echo "${NOT_RUN[*]:-}")); no verdict stands for a step that did not run; $sampled; the circuit digest is kept private and NO public value fixes it before the coins"
  for s in "${REFUSED[@]}"; do say "  REFUSED $s: ${WHY[$s]}"; done
  for s in "${NOT_RUN[@]}"; do say "  NOT RUN $s: ${WHY[$s]}"; done
  [ ${#REFUSED[@]} = 0 ] && [ ${#NOT_RUN[@]} = 0 ] && exit 0
  exit "$1"
}
road_stop() { step_refused "$1" "the road stops here (exit $2)"; road_end "$2"; }
tie_refused() {  # the window commit tied to Pearl+'s W is deleted, so a setting naming it is refused before anything runs
  local v
  for v in PEARLPLUS_TIE PEARLPLUS_TIE_UNIFORM; do
    [ -z "${!v:-}" ] || { say "PREFLIGHT: $v is refused: the tied leg is deleted: no window is tied to a W"; return 1; }
  done
}
window_preflight() {  # K the close's window with a record (WINDOW's, else plans.jsonl's last); each thing absent refused by name
  local miss=() f why
  [ -e "$PY" ] || miss+=("PY, python under VENV")
  [ -n "${WINDOW_CLASS:-}" ] || miss+=("WINDOW_CLASS, the class the serve took (its --window-class), which serve.json doesn't record")
  if [ ! -f "$D/plans.jsonl" ]; then miss+=("D/plans.jsonl, the close's plan of each window")
  elif ! K=$("$PY" - "$D/plans.jsonl" "${WINDOW:-}" 2>&1 <<'EOF'
import json, sys
lines = [json.loads(x) for x in open(sys.argv[1]) if x.strip()]
p = next((x for x in reversed(lines) if sys.argv[2] in ("", str(x["window"]))), None)
if p is None:
    sys.exit(f"D/plans.jsonl has no line for {'WINDOW' if sys.argv[2] else 'any window'}")
why = p.get("refused") or p.get("error") or p.get("pending") or (p.get("handoff") or {}).get("error") \
    or (p.get("record") or {}).get("error")
if why or "handoff" not in p:
    sys.exit(f"D/plans.jsonl's close of its window made no record: {why or 'no handoff (the serve ran without --record)'}"[:600])
print(p["window"])
EOF
  ); then miss+=("$K"); K=
  fi
  if [ -n "${K:-}" ]; then
    for f in serve.json seals.jsonl "windows/w$K/commit/verdict.json" "windows/w$K/commit/runs.jsonl" \
        "windows/w$K/handoff/handoff.json" "windows/w$K/record/circuit.json" "windows/w$K/record/registration.json" \
        "windows/w$K/record/index.json" "advice/w$K/registration.json" "advice/w$K/private/advice.bin" weights_registration.json; do
      [ -e "$D/$f" ] || miss+=("D/${f//w$K/wK}")
    done
  fi
  [ -f "$PRED/predicted.json" ] || miss+=("PRED/predicted.json, predict row's Build before the lease")
  [ ${#miss[@]} = 0 ] && return 0
  why=$(printf '%s; ' "${miss[@]}")
  say "PREFLIGHT: the window's record lacks ${why%; }"
  return 3
}
window_seal() {  # the window's seal held to its record (close_record's finish opened its bundle to make it): the seal's
  # run root, the record's circuit and its compiler, each at its path in the registration (`rows.registration`'s
  # ``window``, as `interface.of_records` reads it), the compiler held to predict row's; the commit's bound Program digest
  # beside the predicted workload digest, which no check here compares
"$PY" - "$D" "$K" "$PRED" <<'EOF' 2>&1 | publish_lines
import hashlib, json, sys
from pathlib import Path
d, k, pred = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
w, why = d / "windows" / f"w{k}", []


predicted = json.loads((pred / "predicted.json").read_text())
seal = next((json.loads(x) for x in (d / "seals.jsonl").read_text().splitlines() if x.strip() and json.loads(x)["window"] == k), None)
v = json.loads((w / "commit" / "verdict.json").read_text())
c = json.loads((w / "commit" / "runs.jsonl").read_text().splitlines()[0])["commit"]
reg = json.loads((w / "record" / "registration.json").read_text())
if seal is None or "error" in seal:
    why.append(f"seals.jsonl has {'no seal' if seal is None else 'a failed seal'} of the window")
win = reg.get("window") if isinstance(reg.get("window"), dict) else {}
if lacks := [f"window.{f}" for f in ("run", "circuit", "compiler") if f not in win]:
    why.append(f"the record's registration has no {', '.join(lacks)}")
run = win.get("run")
if not v.get("run_root") or v.get("run_root") != run:
    why.append(f"the seal's run root {str(v.get('run_root'))[:16]}… is not the record's {str(run)[:16]}…")
if "circuit" in win and win["circuit"] != hashlib.sha512((w / "record" / "circuit.json").read_bytes()).hexdigest():
    why.append("the record's circuit.json is not the circuit its registration binds")
compiler, p = win.get("compiler") if isinstance(win.get("compiler"), dict) else {}, predicted.get("compiler")
same = bool(compiler) and compiler.get("digest") == (p.get("digest") if isinstance(p, dict) else p)
print(f"SEAL: the window's seal, run root {str(v.get('run_root'))[:16]}…, and its record: {'PASS' if not why else 'FAIL (' + '; '.join(why) + ')'}")
print(f"COMPILER: the record's compiler digest and predict row's are equal strings: {'PASS' if same else 'FAIL'}; it is a "
      "digest of code files and the serve's configuration, which no verifier checks (candidate 135)")
print("BOUND-PROGRAM: the commit's bound Program digest and the predicted workload digest are kept side by side and "
      "compared by nothing, so the seal does not tie the window's circuit to the served one")
(d / "private").mkdir(mode=0o700, exist_ok=True)
with open(d / "private" / "verdicts.txt", "a") as kept:
    kept.write(f"COMPILER: the record's {compiler.get('name')} {compiler.get('digest')}, predict row's {p}\n"
               f"BOUND-PROGRAM: the commit's bound Program digest {(c.get('binding') or {}).get('program_digest')}, "
               f"the predicted workload digest {predicted.get('workload_digest')}\n")
sys.exit(0 if same and not why else 1)
EOF
return "${PIPESTATUS[0]}"
}
ROAD_PRIVATE='["plans.jsonl", "handoff/", "card.npz", "key.bin", "circuit.json", "predicted.json", "window-circuit-w*.json", "verdicts.txt"]'
road_private() {  # the plan, the handoff, the circuit and anything naming a template or the description's digest: names
  # no file of the run may have, in its private-paths.json as prove_path/run.sh writes it
  [ -d "${RESEARCH_RUN_DIR:-}" ] || return 0
  python3 - "$RESEARCH_RUN_DIR/private-paths.json" "$ROAD_PRIVATE" <<'EOF'
import json, pathlib, sys
f, names = pathlib.Path(sys.argv[1]), json.loads(sys.argv[2])
doc = json.loads(f.read_text()) if f.is_file() else {}
f.write_text(json.dumps({**doc, "schema": "private-paths/v1", "private": list(doc.get("private", [])),
                         "names": sorted({*doc.get("names", []), *names})}, indent=1) + "\n")
EOF
}
road_pieces() {  # Pearl+, PoUS and the bandwidth audit, where the application lists them: each through its owner's
  # launcher as step 4 runs it, each taking its own gpu-lease; on a node with no card nvidia-smi lists, each refused as
  # needing one. A piece ran only where it exited 0 and its own answer has a verdict: a piece that answers nothing ran nothing
  local only=() c step owner card=
  [ "$POUW" != pearlplus ] || only+=(pouw)
  [ -z "${POUS_CMD:-}" ] || only+=(pous)
  [ -z "${BANDWIDTH_CMD:-}" ] || only+=(bandwidth)
  [ ${#only[@]} -gt 0 ] || return 0
  if ! command -v gpu-lease > /dev/null; then card="this node has no gpu-lease"
  elif ! nvidia-smi -L 2> /dev/null | grep -q '^GPU '; then card="nvidia-smi lists no card on this node"; fi
  for c in "${only[@]}"; do
    step=${c/#pouw/pearl-plus} owner="memory accounting's"; [ "$c" != pouw ] || owner="compute accounting's"
    if [ -n "$card" ]; then step_not_run "$step" "needs a card: its piece takes its own gpu-lease, and $card: $owner"; continue; fi
    pieces --only "$c"
    if "$PY" -c 'import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
doc = next((json.loads((d / n).read_text()) for n in ("answer.json", "development.json") if (d / n).is_file()), {})
sys.exit(0 if json.loads((d / "piece.json").read_text())["rc"] == 0 and doc.get("verdict") else 1)' "$OUT/pieces/$c" 2> /dev/null
    then RAN+=("$step")
    else step_refused "$step" "$(head -n 1 "$OUT/pieces/$c/refused.txt" 2> /dev/null || echo "its piece answered no verdict ($owner)")"; fi
  done
}
road_wipe() {  # the wipe runs under the serve's lease before its engine loads, so the road only reads what it answered
  [ -n "${WIPE_CMD:-}" ] || return 0
  if [ -f "$D/pieces/wipe/answer.json" ]; then
    mkdir -p "$OUT/pieces/wipe" && cp "$D/pieces/wipe/answer.json" "$OUT/pieces/wipe/"
    [ ! -f "$D/pieces/wipe/summary.json" ] || cp "$D/pieces/wipe/summary.json" "$OUT/pieces/wipe/"
    say "WIPE: the card launcher's, under the serve's lease (D/pieces/wipe/answer.json)"; RAN+=(wipe)
  else
    step_not_run wipe "D/pieces/wipe/answer.json is absent: the card launcher did not run the wipe under the serve's lease (vllm's h100_smoke.sh)"
  fi
}
road_network() {  # the network attester's own piece, called as step 4 or 6 calls it; its refusal is its own text
  local cmd=${CERTIFIER_CMD:-${CERTIFIER_CHAIN_CMD:-}} dir=$OUT/pieces/certifier
  [ -n "$cmd" ] || return 0
  mkdir -p "$dir"
  local rc=0
  (cd "$TREE" && bash -c "$cmd" _ "$D" "$K" "$dir" "${VW:-}") > "$dir/road.log" 2>&1 || rc=$?
  if [ -f "$dir/refused.txt" ]; then REFUSED+=(network-attester); WHY[network-attester]=$(head -n 1 "$dir/refused.txt" | head -c 600)
    say "network-attester: $(head -c 600 "$dir/refused.txt")"
  elif [ "$rc" = 0 ]; then RAN+=(network-attester)
  else step_refused network-attester "$(failed "$rc" "$dir/road.log")"; fi
}
road_sampled() {  # step 3 on the window's road: the window's record, its Lean checks run from its own files (records_audit.py audit), its circuit the prover's
  local rec=$D/windows/w$K/record leanb=$TREE/verity/protocols/proof/flock/verifier/.lake/build/bin bin=$FW/flock-gpu-link/flock/target/release/flock-circuit
  local s=$D/private/sampled-proofs main=$OUT/sampled-proofs/main.txt unit=${TMPL_UNIT_BIN:-$FW/lean-bin/flock-tmpl-unit} how="bash $TREE/benchmarks/one_stage/build_sampled_step.sh" m0 law run rc b line regs
  [ "$DEVELOPMENT" = 1 ] || { step_not_run sampled-proofs "its circuit is the one the prover registered, and its coins the local challenger's: a development run's alone (DEVELOPMENT=1) until proofs' compile check runs and the auditor's challenger is wired"; return; }
  for b in "$bin" "$leanb/verity-verify-proof" "$leanb/flock-zk-server" "$leanb/flock-lower-check"; do
    [ -x "$b" ] || { step_not_run sampled-proofs "proofs' $(basename "$b") is not built at $b: build it, with the other programs this step runs, by: $how"; return; }
  done
  [ -x "$unit" ] || { step_not_run sampled-proofs "proofs' flock-tmpl-unit, which places a templated member's unit rows, is not built at $unit (TMPL_UNIT_BIN): build it, with the other programs this step runs, by: $how"; return; }
  grep -q -- '"--prover-circuit"' "$TREE/benchmarks/prove_path/records_audit.py" \
    || { step_not_run sampled-proofs "proofs' records_audit.py takes no --prover-circuit, the audit of a circuit the prover registered with its compile check not run"; return; }
  law=$("$PY" -c 'import json, sys; r = json.load(open(sys.argv[1])); print(r.get("rule") or (r.get("law") or {}).get("rule") or "")' "$rec/registration.json") || law=
  law=${law:-$V3C_LAW}
  [ -n "$law" ] || { step_not_run sampled-proofs "the window's record registers no rule of its sampling design, and the configuration states none (V3C_LAW): proofs' audit holds the record to one"; return; }
  # the commit verdict's run root, which the seal step held to the record's
  run=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["run_root"])' "$D/windows/w$K/commit/verdict.json") \
    || { step_not_run sampled-proofs "the window's commit verdict has no run root"; return; }
  # the deployment's registration record pins the roots a window's registered weight reads open against
  [ -f "$DREG" ] || { step_not_run sampled-proofs "$NO_DREG"; return; }
  mapfile -t regs < <(wreg_args)
  (umask 077; mkdir -p "$s") && mkdir -p "$OUT/sampled-proofs" "$LOGS/sampled-proofs"
  git -C "$TREE" rev-list --first-parent "${MAIN_SHA:-HEAD}" > "$main" 2>/dev/null && [ -s "$main" ] \
    || { step_not_run sampled-proofs "no first-parent line of main for its challenger session to pin on (${MAIN_SHA:-HEAD} has none in this tree)"; return; }
  m0=$TREE:$TREE/benchmarks/numerical/python:$TREE/verity/protocols/proof/flock/prover/python:$TREE/adapters/vllm
  (cd "$TREE" && env PYTHONPATH="$m0${PYTHONPATH:+:$PYTHONPATH}" DEVELOPMENT=1 RAYON_NUM_THREADS="$(nproc)" RUST_MIN_STACK=$((64 << 20)) timeout -k 1m "$V3C_AUDIT_TIMEOUT" \
    "$TREE/.venv/bin/python" benchmarks/prove_path/records_audit.py audit --window "$rec" --sampling-design "$law" \
    --prover-circuit --development --run "$run" ${regs[@]+"${regs[@]}"} \
    --challenger local --main "$main" --m0-pythonpath "$m0" --bin "$bin" --lean "$leanb/verity-verify-proof" \
    --zk-server "$leanb/flock-zk-server" --lower "$leanb/flock-lower-check" --tmpl-unit-bin "$unit" \
    --lean-workers "${V3C_LEAN_WORKERS:-1}" --stage-workers "${V3C_STAGE_WORKERS:-1}" --stage-worker-gb "${V3C_STAGE_WORKER_GB:-20}" \
    --audit-mem-gb "$V3C_AUDIT_MEM_GB" --ports "${V3C_PORTS:-16}" --out "$s/audit") \
    > "$LOGS/sampled-proofs/audit.log" 2> "$LOGS/sampled-proofs/audit.err"
  rc=$?
  grep -m1 '^RECORDS' "$LOGS/sampled-proofs/audit.log" > "$s/records.txt"
  cp "$s/audit/summary.json" "$s/" 2>/dev/null
  [ -s "$s/records.txt" ] && [ -s "$s/summary.json" ] \
    || { step_refused sampled-proofs "ERROR: the records audit stopped with no RECORDS row $(failed "$rc" "$LOGS/sampled-proofs/audit.err")"; return; }
  # $OUT/summary.json is the audit's public summary, which the record step reads; the full one names templates and stays in $s
  line=$("$PY" - "$s/summary.json" "$s/audit/window/summary.json" "$law" "$OUT/summary.json" <<'EOF'
import json, re, sys
from pathlib import Path
row, inner = json.load(open(sys.argv[1])), (json.load(open(sys.argv[2])) if Path(sys.argv[2]).is_file() else {})
draw = row.get("draw") if isinstance(row.get("draw"), list) else []
audit = row.get("audit") or {}
lean = audit.get("members_lean") or {}
accepted = audit.get("accepted") is True and row.get("ok") is True
bad = sorted(u for u in draw for x in (inner.get("unlowered") or {}).values() if x["base"] <= u < x["base"] + x["n"])
verdict = ((lean.get("detail") or {}).get("lean") or {}) if isinstance(lean.get("detail"), dict) else {}
wrong = audit.get("wrong_units") if isinstance(audit.get("wrong_units"), dict) else {}
public = {"ok": row.get("ok"), "audit": {"accepted": audit.get("accepted"),
                                         "members_lean": {**{k: lean[k] for k in ("ok", "ran", "codes") if k in lean},
                                                          "detail": {"lean": {k: verdict[k] for k in ("verdict", "output_sha512")
                                                                              if k in verdict}}} if lean else None,
                                         **({"wrong_units": {k: wrong[k] for k in ("bound", "delta") if k in wrong}} if wrong else {})},
          "sampled": len(draw), "draw": draw, "unprovable_sampled": bad, "compiled": row.get("compiled", "not stated by the audit"),
          "circuit": "registered by the prover", "coins": "the local challenger's (DEVELOPMENT=1)"}
Path(sys.argv[4]).write_text(json.dumps(public, indent=1, sort_keys=True))


def reasons(checks):
    """The window's failing checks before the draw: each member check's code with its count, its lowest member and that
    member's reason, then the registration's codes with their detail."""
    ms = sorted((int(k[1:]), v) for k, v in checks.items() if k[:1] == "m" and k[1:].isdigit() and isinstance(v, dict))
    codes = {}
    for k, v in ms:
        if v.get("ok") is False:
            for c in v.get("codes") or ["no code"]:
                codes.setdefault(c, [k, 0, (v.get("detail") or {}).get(c)])[1] += 1
    out = [f"{c}: {n} of {len(ms)} members, first m{k}: {why}" for c, (k, n, why) in codes.items()]
    reg = checks.get("registration")
    if isinstance(reg, dict) and reg.get("ok") is False:
        d = reg.get("detail") or {}
        out += [f"registration {c}: {json.dumps(d.get(c), sort_keys=True)}" for c in reg.get("codes") or ["no code"]]
    return "; ".join(out)


if row.get("refused") or not isinstance(row.get("draw"), list):
    why = row.get("refused") or reasons(inner.get("checks") or {}) or "the audit's summary names no failing check"
    text = why if isinstance(why, str) else json.dumps(why, sort_keys=True)
    text = re.sub(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*_v[0-9]+(?:\{[^}]*\})?", "a template", text)
    print(f"REFUSED before the sample: {text}"[:2000])
    sys.exit(2)
if not draw:
    print("the law samples no block of this window: nothing would be proved")
    sys.exit(3)
said = (f"under the DEVELOPMENT law {sys.argv[3]}, {len(draw)} sampled block(s) {draw}, on the local challenger's coins (DEVELOPMENT=1); "
        "its circuit registered by the prover; the compile check did not run, so compilation is not verified")
if bad:
    print(f"REJECTED: {said}; sampled unit(s) {bad} of a block the door refuses to lower: a failed proof, not resampled "
          "(its Definition in the serve dir's private/sampled-proofs/summary.json)")
    sys.exit(1)
print(f"the Lean checks pass on the sampled blocks' proofs (run on the prover's node): {said}" if accepted else f"REJECTED: {said}")
sys.exit(0 if accepted else 1)
EOF
)
  case $? in
    0) RAN+=(sampled-proofs)
       say "sampled-proofs: $line"
       if [ -f "$TREE/benchmarks/one_stage/accept_line.py" ]; then
         (cd "$TREE/benchmarks/one_stage" && env -u PYTHONPATH PYTHONPATH="$TREE" "$TREE/.venv/bin/python" accept_line.py "$s/summary.json" - -) 2>&1 \
           | while IFS= read -r b; do say "sampled-proofs: $b"; done
       else say "sampled-proofs: the bound: not stated (accept_line.py absent)"; fi ;;
    1 | 2) step_refused sampled-proofs "$line" ;;
    3) step_not_run sampled-proofs "$line" ;;
    *) step_refused sampled-proofs "ERROR: the records audit's summary (records_audit.py exit $rc) has no public form" ;;
  esac
}
road_step_named() {  # in TREE: the window's step $1, from its record or refused by name; the steps' order is the caller's
  case $1 in
    preflight)
      [ -z "${PEARLPLUS_TIE:-}" ] || { say "PREFLIGHT: PEARLPLUS_TIE ties a window to a ROW's serve, and the window's road serves none: it ties nothing"; exit 3; }
      say "ROAD: the window's record in D (the close's plan, handoff and record, the seal and the advice), after the lease; nothing served, no ROW read"
      road_private || road_stop private 2
      RAN+=(preflight) ;;
    seal)
      say "PREDICT, SERVE: the window's serve's, before and under its lease (PRED/predicted.json, D)"
      if window_seal; then RAN+=(seal); else road_stop seal 11; fi ;;
    window)
      mkdir -p "$OUT/serve" && "$PY" -c 'import hashlib, json, sys; s = next(x.rstrip("\n") for x in open(sys.argv[1]) if x.strip() and json.loads(x)["window"] == int(sys.argv[2])); print(json.dumps({"seal_sha512": hashlib.sha512(s.encode()).hexdigest()}))' "$D/seals.jsonl" "$K" > "$OUT/serve/phase.json" \
        || { rm -f "$OUT/serve/phase.json"; road_stop window 21; }
      say "WINDOW: the close's, with its record (WINDOW's, else plans.jsonl's last), at the class the serve took; its circuit registered by the prover, which no challenger holds before any coin on this road"
      RAN+=(window)
      say "$("$PY" - "$D/serve.json" "$WINDOW_CLASS" <<'EOF'
import json, math, sys
c = sys.argv[2]
t = ((json.load(open(sys.argv[1])).get("engine") or {}).get("engine_kwargs") or {}).get("max_num_batched_tokens")
s = int(c.split(",")[0])
print(f"CLASS {c}: its bits are not counted: serve.json records no engine.engine_kwargs.max_num_batched_tokens" if not t else
      f"CLASS {c}: {math.log2(int(t) // s):.2f} bits leave with it (its prompt length, one of the {int(t) // s} that fit {s} "
      f"slots in {t} tokens a step; its slots and outputs are the application's)")
EOF
)" ;;
    advice)
      "$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" public-registration "$D/advice/w$K/registration.json" > "$OUT/registration.json" \
        || { say "ADVICE: its registration has no public form"; road_stop advice 14; }
      say "ADVICE: committed (its root a hiding commitment); its content and length are held to no registered compiler (candidate 135)"
      RAN+=(advice) ;;
    inference-policy) road_step inference-policy inference_policy window ;;
    circuits)  # a count an earlier call left in OUT is no count of this one's, which inference-only reads
      local c
      for c in $CLASSES; do rm -f "$OUT/window-circuit-${c//,/-}.json"; done
      road_step circuits circuits ;;
    block-counts) road_step block-counts block_counts ;;
    sampled-proofs) road_sampled ;;
    pieces) road_pieces ;;
    wipe) road_wipe ;;
    network-attester) road_network ;;
    inference-only)  # the circuits step may have run in another call: its public count is what it leaves in OUT
      if circuits_counted && [[ " ${RAN[*]} " = *" inference-policy "* ]]; then road_step inference-only inference_only
      else step_not_run inference-only "its verdict reads the circuits step's window circuit and the inference-only policy, and one of them did not run"; fi ;;
    record)  # the sampled-proofs step may have run in another call: what it leaves in OUT is its public summary
      if [ ! -f "$OUT/summary.json" ]; then
        step_not_run record "its verdict of record is Lean's members check in the sampled proofs' summary ($OUT/summary.json), which the sampled-proofs step did not write"
      elif rec=$("$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" window-record "$OUT/summary.json" 2>&1); then RAN+=(record); say "$rec"
      else step_refused record "$rec"; fi ;;
    composite) step_not_run composite "served_composite.sh reads the audit's cases under D/private/logs/sampled-proofs/keep/served or D/v3c-w*/audit and, where compute accounting is listed, Pearl+'s run case, and the window road writes neither: no input of the composite's stands, so its verdict is not made" ;;
  esac
}
steps_named() {  # every name in "$@" is a step road_step_named knows, and none is named twice: refused by name otherwise
  local step seen=" " known
  known=" $(declare -f road_step_named | sed -n 's/^ *\([a-z][a-z-]*\))$/\1/p' | tr '\n' ' ')"
  for step in "$@"; do
    [[ $known = *" $step "* ]] || { say "PREFLIGHT: the window has no step $step: nothing run"; return 3; }
    [[ $seen = *" $step "* ]] && { say "PREFLIGHT: the step $step is named twice: nothing run"; return 3; }
    seen+="$step "
  done
}
circuits_counted() {  # every class of CLASSES counted (`circuit` writes its public count only then)
  local c
  for c in $CLASSES; do [ -f "$OUT/window-circuit-${c//,/-}.json" ] || return 1; done
}
road_steps() {  # in TREE: the steps named, in the caller's order, then the road's end
  local step
  for step in "$@"; do road_step_named "$step"; done
  road_end 20
}

if [ "${1:-}" = serve ]; then
  host_set VY_CHALLENGER_1_HOST "${PEARLC_CHALLENGER_GATE:-}" && key_set "${PEARLC_CHALLENGER_GATE:-}" || exit 3; serve; exit $?
fi
if [ "${1:-}" = circuits ]; then K=${2:?served_verdicts.sh circuits WINDOW}; circuits; exit $?; fi
if [ "${1:-}" = inference-policy ]; then mkdir -p "$D" && logs_dir && inference_policy; exit $?; fi
if [ "${1:-}" = weights ]; then weights; [ -f "$WREG" ]; exit $?; fi  # D a serve dir with its serve.json: the registration alone
if [ "${1:-}" = inference-only ]; then K=${2:?served_verdicts.sh inference-only WINDOW}; inference_only; exit $?; fi
if [ "${1:-}" = block-counts ]; then K=${2:?served_verdicts.sh block-counts WINDOW}; block_counts; exit $?; fi
if [ "${1:-}" = sampled-coins ]; then sampled_coins; exit $?; fi
if [ "${1:-}" = v3c-coins ]; then K=${2:?served_verdicts.sh v3c-coins WINDOW}; VW=$D/v3c-w$K-$vtag; v3c_trees; exit $?; fi
if [ "${1:-}" = front-flags ]; then front_ok; exit $?; fi
if [ "${1:-}" = pous-pins ]; then pous_pins_ok; exit $?; fi
if [ "${1:-}" = private-pieces ]; then private_pieces; exit $?; fi
if [ "${1:-}" = defer ]; then defer "${2:?served_verdicts.sh defer WHY}"; exit $?; fi
if [ "${1:-}" = served-compare ]; then K=${2:?served_verdicts.sh served-compare WINDOW}; served_compare; exit $?; fi
if [ "${1:-}" = pearl-cubin ]; then pearl_cubin; exit $?; fi
if [ "${1:-}" = seal ]; then seal; exit $?; fi
if [ "${1:-}" = window-seal ]; then K=${2:?served_verdicts.sh window-seal WINDOW}; window_seal; exit $?; fi
if [ "${1:-}" = window ]; then window; exit $?; fi
if [ "${1:-}" = pearlplus ]; then K=${2:?served_verdicts.sh pearlplus WINDOW}; pieces --only pouw; exit $?; fi
if [ "${1:-}" = publish ]; then publish; exit $?; fi
if [ "${1:-}" = handoff ]; then  # `served_verdicts.sh handoff [HOST]`: the compose hand-off written again, from a deferral's
  # pieces and inference-only verdict at its settings where the serve dir holds one, else from $OUT's; HOST is where V3c's
  # audit ran (a deferral's follow-on), which a serve dir holding that audit must name
  V3C_AUDITED=${2:-${V3C_AUDITED:-}}
  if [ -n "$FOLLOW" ]; then
    follow_handoff handoff "served_verdicts.sh handoff HOST" || exit 3
    handoff "$DEFER_DIR/pieces" "$DEFER_DIR/inference-only.json"
  else
    handoff "$OUT/pieces" "$OUT/record/inference-only.json"
  fi
  exit $?
fi
if [ "${1:-}" = copy ]; then copy "${2:?served_verdicts.sh copy deferred|compose HOST[:DIR]}" "${3:?served_verdicts.sh copy deferred|compose HOST[:DIR]}"; exit $?; fi
if [ "${1:-}" = gate-compose ]; then  # `served_verdicts.sh gate-compose PIECES [VERDICT [HOST]]`: on a deferral's
  # serve dir at its settings, HOST where V3c's audit ran, as `handoff`'s
  V3C_AUDITED=${4:-${V3C_AUDITED:-}}
  host_set VY_AUDITOR_2_HOST "$COMPOSE_GATE" || exit 3
  [ -z "$FOLLOW" ] || follow_handoff gate-compose "served_verdicts.sh gate-compose PIECES VERDICT HOST" || exit 3
  gate_compose "${2:?served_verdicts.sh gate-compose PIECES [VERDICT [HOST]]}" "${3:-}"; exit $?
fi

# $OUT leaves the node: once the run is done, however it ends, each of its files that names a window or counts its own is
# held in full in the serve dir's private/out, its public form in its place (`served_verdicts.py publish`)
trap 'rc=$?; publish; exit $rc' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
tree_sha | tee "$OUT/machine.txt"
if [ -n "$FOLLOW" ]; then  # a deferral's follow-on: no serve, its venv the tree's, on a node with the audit's memory
  [ -z "$CLASH" ] || { say "PREFLIGHT: the follow-on runs at the deferral's settings (the serve dir's private/deferred/deferred.env), and the environment sets another$CLASH: unset it"; exit 3; }
  why=$(short_of_memory) && { say "PREFLIGHT: the follow-on runs the V3c audit the served node deferred, and $why"; exit 3; }
  sha=$(tree_sha)
  [ -n "$DEFERRED_TREE" ] && { [ "$sha" = "$DEFERRED_TREE" ] || [ -n "${FOLLOW_ANY_TREE:-}" ]; } \
    || { say "PREFLIGHT: the follow-on reads the served node's window-circuit record, which its tree ${DEFERRED_TREE:-(unknown)} wrote, and this tree is ${sha:-(unknown)}: run it at that commit, or set FOLLOW_ANY_TREE=1"; exit 3; }
  [ -z "${DAY:-}" ] || grep -q -- '"--day"' "$TREE/benchmarks/prove_path/records_audit.py" \
    || { say "PREFLIGHT: the deferral declared day $DAY, whose audit runs R2 with records_audit's --day, which this tree's records_audit.py doesn't take"; exit 3; }
elif [ -n "$COMPOSE" ]; then  # a compose follow-on: no serve, row dir, card or gpu-lease, and never the served node
  [ -f "$COMPOSE_DIR/compose.env" ] \
    || { say "PREFLIGHT: the compose follow-on composes from the served node's hand-off, and the serve dir D holds none (private/compose/compose.env): copy each file its private/compose/inputs.txt lists to the same path here first"; exit 3; }
  [ -z "$ENV_REFUSED" ] || { say "PREFLIGHT: the compose follow-on reads the hand-off's settings (the serve dir's private/compose/compose.env) as data, and refuses them, none set: $ENV_REFUSED"; exit 3; }
  # every path the follow-on reads is a file of the copy: the window and the record's audit name one inside it, the
  # served node's serve dir is what the cases' blobs are rebased from, and the copy holds files, no link out of it
  [[ ${K:-} =~ ^[0-9]+$ ]] \
    || { say "PREFLIGHT: the compose follow-on reads the window's files in the copy, and the hand-off's K, ${K:-(none)}, is not a window's number"; exit 3; }
  [[ -z ${SAMPLED_AUDIT:-} || $SAMPLED_AUDIT =~ ^[A-Za-z0-9_-][A-Za-z0-9._-]*(/[A-Za-z0-9_-][A-Za-z0-9._-]*)*$ ]] \
    || { say "PREFLIGHT: the compose follow-on reads the record's audit in the copy, and the hand-off's SAMPLED_AUDIT, $SAMPLED_AUDIT, is no path inside it (a relative path, no part of it starting with a dot)"; exit 3; }
  [ -n "${SERVED_DIR:-}" ] \
    || { say "PREFLIGHT: the compose follow-on rebases the cases' blobs from the served node's serve dir, which the hand-off doesn't name (SERVED_DIR)"; exit 3; }
  link=$(find "$D" -type l -printf '%P\n' -quit)
  [ -z "$link" ] \
    || { say "PREFLIGHT: the compose follow-on reads the copy's own files, and its serve dir holds a link ($(printf %q "$link" | cut -c1-200)): copy the hand-off's files, never a link"; exit 3; }
  [ -z "$CLASH" ] || { say "PREFLIGHT: the compose follow-on runs at the hand-off's settings (the serve dir's private/compose/compose.env), and the environment sets another$CLASH: unset it"; exit 3; }
  { [ "$(hostname)" != "$SERVED_NODE" ] && [ "$(hostname -s)" != "${SERVED_NODE%%.*}" ]; } \
    || { say "PREFLIGHT: this is the served node (SERVED_NODE), which the prover controls, so nothing composed here claims a record: compose on COMPOSE_NODE"; exit 3; }
  sha=$(tree_sha)
  [ -n "$COMPOSE_TREE" ] && { [ "$sha" = "$COMPOSE_TREE" ] || [ -n "${FOLLOW_ANY_TREE:-}" ]; } \
    || { say "PREFLIGHT: the compose follow-on composes with the served node's tree ${COMPOSE_TREE:-(unknown)}, and this tree is ${sha:-(unknown)}: run it at that commit, or set FOLLOW_ANY_TREE=1"; exit 3; }
  held=$(cd "$D" && find . -type f -printf '%P\n' | "$SAY_PY" "$SV" witness) \
    || { say "PREFLIGHT: the compose follow-on holds no witness of the served run, and this node's serve dir does ($(echo "$held" | head -1 | cut -c1-400)): compose on a node that holds the hand-off alone"; exit 3; }
elif [ "${1:-}" = v3c ]; then  # V3c alone on an existing serve dir serves nothing: no row dir, taps or gpu-lease
  [ -e "$PY" ] || { say "PREFLIGHT: no python under VENV (VENV, which a node without vLLM's venv sets to the tree's .venv)"; exit 3; }
else
  tie_refused || exit 3
  [ -n "${APPLICATION_PROTOCOLS+x}" ] \
    || { say "PREFLIGHT: the runner is verity run's: no application was named (APPLICATION_PROTOCOLS), so what it lists cannot be checked"; exit 3; }
  [ "${1:-}" = steps ] && [ $# -gt 1 ] \
    || { say "PREFLIGHT: the runner's steps and their order are verity run's stages': none was named (served_verdicts.sh steps STEP...)"; exit 3; }
  steps_named "${@:2}" || exit 3
  window_preflight || exit 3
  KN=$K CLASSES=$WINDOW_CLASS
  logs_dir
  shift
  road_steps "$@"
fi
front_ok || exit 3
[ -n "$FOLLOW$COMPOSE" ] || [ "${1:-}" = v3c ] || pous_pins_ok || exit 3  # the serve's run, which starts PoUS's piece
SPLIT= DEFER=
if [ -n "${CERTIFIER_CHAIN_CMD:-}${CERTIFIER_PERIOD_CMD:-}" ]; then  # the split attester: its period check writes its answer
  [ -z "${CERTIFIER_CMD:-}" ] || { say "PREFLIGHT: CERTIFIER_CMD or the split's CERTIFIER_CHAIN_CMD and CERTIFIER_PERIOD_CMD, not both"; exit 3; }
  [ -n "$V3C" ] || { say "PREFLIGHT: the split attestation period check reads V3c's period.json, and V3C_CLASS is empty"; exit 3; }
  SPLIT=1
fi
case ${V3C_COMPOSE_CACHE:-/} in  # a kept compose cache: outside TREE, where the warm and the audit would put a relative path
  /*) ;;
  *) say "PREFLIGHT: V3C_COMPOSE_CACHE is the kept compose cache's directory, an absolute path (the warm and the audit run in the tree, where a relative one would put it), not $V3C_COMPOSE_CACHE: name it from /"; exit 3 ;;
esac
if [ -n "${V3C_WARM_JOBS:-}" ]; then  # V3c's composes made before its audit
  [[ $V3C_WARM_JOBS =~ ^[1-9][0-9]*$ ]] \
    || { say "PREFLIGHT: V3C_WARM_JOBS is how many of the window's templates warm composes at once, a positive integer, not $V3C_WARM_JOBS"; exit 3; }
  grep -q -- '"warm"' "$TREE/benchmarks/prove_path/records_audit.py" || [ -n "$V3C_WARM_GIVEN" ] || V3C_WARM_JOBS=
  [ -z "$V3C_WARM_JOBS" ] || grep -q -- '"warm"' "$TREE/benchmarks/prove_path/records_audit.py" \
    || { say "PREFLIGHT: V3C_WARM_JOBS composes the window's templates before its audit with records_audit.py warm, which this tree's records_audit.py doesn't have"; exit 3; }
fi
if [ -n "${V3C_NEGATIVE_WORKERS:-}" ]; then  # V3c's audit's negatives side by side
  [[ $V3C_NEGATIVE_WORKERS =~ ^[1-9][0-9]*$ ]] \
    || { say "PREFLIGHT: V3C_NEGATIVE_WORKERS is how many of the audit's negatives run at once, a positive integer, not $V3C_NEGATIVE_WORKERS"; exit 3; }
fi
case $DEVELOPMENT in '' | 1) ;; *) say "PREFLIGHT: DEVELOPMENT is 1, a development run, or unset, not $DEVELOPMENT"; exit 3 ;; esac
case ${VERITY_HOLD_PLACEMENT:-} in '' | 0 | 1) ;; *) say "PREFLIGHT: VERITY_HOLD_PLACEMENT is 1, which restores the refusals that rest on where coins are flipped or composites made, or 0 or unset, not $VERITY_HOLD_PLACEMENT"; exit 3 ;; esac
[ "$H1_KEY_SHA512" = "$H1_KEY_OF_RECORD" ] || [ "$DEVELOPMENT" = 1 ] || ! hold_placement \
  || { say "PREFLIGHT: H1_KEY_SHA512 replaces the challenger key of record (vy-challenger-1's), which under VERITY_HOLD_PLACEMENT=1 only a development run takes: unset H1_KEY_SHA512, DEVELOPMENT=1, or VERITY_HOLD_PLACEMENT unset"; exit 3; }
if [ -n "${REGISTRATION_LANE:-}" ]; then
  [ -n "${REGISTRATION_LEDGER:-}" ] \
    || { say "PREFLIGHT: REGISTRATION_LANE holds the audit to the registration the auditor's challenger signed for that lane, read from its ledger REGISTRATION_LEDGER, which is unset"; exit 3; }
  [[ "${REGISTRATION_HOLD_S:-}" =~ ^[1-9][0-9]*$ ]] \
    || { say "PREFLIGHT: REGISTRATION_LANE holds the serve after its weights step until REGISTRATION_LEDGER carries the lane's registration, at most REGISTRATION_HOLD_S seconds, a positive integer with no default, not '${REGISTRATION_HOLD_S:-}'"; exit 3; }
  [ -r "${REGISTRATION_KEY:-${PP_CHALLENGER_KEY:-}}" ] \
    || { say "PREFLIGHT: REGISTRATION_LANE's ledger is checked under the challenger's key record REGISTRATION_KEY (default PP_CHALLENGER_KEY), which is unset or not a readable file"; exit 3; }
  [ -z "${H1_LANE:-}" ] || [ "$H1_LANE" = "$REGISTRATION_LANE" ] \
    || { say "PREFLIGHT: REGISTRATION_LANE $REGISTRATION_LANE is not H1_LANE $H1_LANE, the lane the epoch's sessions reserve on, and the verifier refuses a session on another lane than the registration's"; exit 3; }
fi
if [ -n "$V3C" ] && [ -z "$COMPOSE" ]; then  # V3c's audit's coins: the live challenger's, or a development run's
  case $V3C_CHALLENGER in
    local | loopback:*)
      [ "$DEVELOPMENT" = 1 ] || ! hold_placement \
        || { say "PREFLIGHT: V3c's audit is the sampled-proofs audit of record, and V3C_CHALLENGER (default PP_CHALLENGER) is a development session, which under VERITY_HOLD_PLACEMENT=1 only a development run takes: set PP_CHALLENGER=gate:USER@HOST with PP_PEER, PP_CHALLENGER_KEY and PP_GATE_KEY, DEVELOPMENT=1, or VERITY_HOLD_PLACEMENT unset"; exit 3; } ;;
    gate:?*@?*)
      [ -n "$V3C_PEER" ] \
        || { say "PREFLIGHT: V3c's audit's session on its gate listens on its peer's 127.0.0.1, and V3C_PEER (default PP_PEER), this node's name in the gate's /etc/vy/peers, is unset"; exit 3; }
      [ -n "${PP_CHALLENGER_KEY:-}" ] && [ -r "$PP_CHALLENGER_KEY" ] \
        || { say "PREFLIGHT: V3c's audit's session on its gate is held to the challenger key record PP_CHALLENGER_KEY, which is unset or not a readable file"; exit 3; }
      if [ -n "$FOLLOW" ]; then here=${SERVED_PEER:-} what=SERVED_PEER
      elif [ "$POUW" = pearlplus ] && [ -n "$PEARLC_CHALLENGER_GATE" ]; then here=$PEARLC_CHALLENGER_PEER what=PEARLC_CHALLENGER_PEER
      else here= what=; fi
      if [ -n "$FOLLOW" ] && [ "$V3C_PEER" = "$here" ]; then
        say "PREFLIGHT: V3C_PEER (default PP_PEER) is the served node's (SERVED_PEER), which deferred the audit: its session would listen on that node's 127.0.0.1, not this one's; name this node as the gate's /etc/vy/peers does"; exit 3
      elif [ -z "$FOLLOW" ] && [ -n "$here" ] && [ "$V3C_PEER" != "$here" ]; then
        say "PREFLIGHT: V3C_PEER (default PP_PEER) and $what differ: each names the node the run serves on, where V3c's audit's session listens as the other one does"; exit 3
      fi ;;
    *) say "PREFLIGHT: V3C_CHALLENGER (default PP_CHALLENGER) is local, loopback:HOST:PORT or gate:USER@HOST, and it is none of them"; exit 3 ;;
  esac
fi
tie_refused || exit 3
# V3c's rule is a rate: window-commit and the audit's stream take no fixed count, and the class circuit has no work table
[ -z "$V3C" ] || [ -n "$COMPOSE" ] || [[ $V3C_LAW == rate:* || $V3C_LAW == stratified-rate:* ]] \
  || { say "PREFLIGHT: V3C_LAW is the rate V3c's window is committed under and its audit states to its stream, rate:NUM/DEN or stratified-rate:NUM/DEN, not $V3C_LAW: in place of a fixed count K, state K over the proof blocks the class's window registers (rate:K/N or stratified-rate:K/N)"; exit 3; }
if [ -n "${DAY:-}" ] && [ -z "$FOLLOW$COMPOSE" ]; then  # the day's one circuit declared before the first window, R2 holding V3c's to it
  [[ $DAY =~ ^[0-9]+$ ]] && [ -n "${DAY_LOG:-}" ] && [[ ${DAY_PROGRAM:-} =~ ^[0-9a-f]{128}$ ]] \
    || { say "PREFLIGHT: DAY=N declares day N's circuit: it needs DAY_LOG (the deployment's registration log) and DAY_PROGRAM (the SHA-512, lowercase hex, of the program its windows register)"; exit 3; }
  [ -f "$TREE/adapters/vllm/verity_vllm/pipeline/serve_day.py" ] && grep -q -- '"--day"' "$TREE/benchmarks/prove_path/records_audit.py" \
    || { say "PREFLIGHT: DAY needs serve-openai's --day and records_audit's --day, which this tree hasn't"; exit 3; }
fi
if [ -n "${PREFLIGHT_ONLY:-}" ]; then  # the checks above alone: nothing kept, built or served
  if [ -n "$FOLLOW" ]; then plan="the follow-on of the served node's (SERVED_NODE) deferral"
    [ "$COMPOSE_OFF_NODE" = 0 ] || plan+=", its compose hand-off for COMPOSE_NODE"
  elif [ -n "$COMPOSE" ]; then plan="the compose follow-on of the served node's (SERVED_NODE) hand-off: $HANDED_WHAT"
  else
    plan="the serve's lease ${LEASE_MIN:-60} min, capped at ${LEASE_MEM_GB:-128} GB"
    [ -z "$DEFER" ] || plan+="; $DEFER, so step 6 defers $DEFERRED_WHAT to a follow-on"
    [ "$COMPOSE_OFF_NODE" = 0 ] || plan+="; $HANDED_WHAT composed off this node, on COMPOSE_NODE, from its compose hand-off"
  fi
  say "PREFLIGHT passed, $NODE_MEM_GB GB: $plan. PREFLIGHT_ONLY: nothing kept, built or served"
  exit 0
fi
mkdir -p "$D" "$ROOT/cache" "$FW"
(umask 077; mkdir -p "$D/private") && logs_dir
{ id -un; hostname; git -C "$TREE" rev-parse HEAD 2>/dev/null; nproc; free -g | head -2; nvidia-smi -L; echo "serve dir $D"; } \
  > "$LOGS/machine.txt" 2>&1
[ -n "$COMPOSE" ] || [ -z "${KEEP_CMD:-}" ] || (cd "$TREE" && bash -c "$KEEP_CMD" _ "$ROOT") 2>&1 | tail -2
if [ "${1:-}" = predict ]; then predict; exit $?; fi

build() {  # what prove_path.sh builds, here while the endpoint serves; prove_path.sh's own builds then find them done
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
  uv sync --all-packages --extra torch-cpu || return 5
  (cd verity/protocols/proof/flock/verifier && lake build verity-verify-proof flock-zk-server flock-read-tmpl) || return 7
  (cd verity/protocols/proof/spot/verifier/lean && lake build spot-verifier) || return 7
  mkdir -p "$LOGS/m0-build"
  # FW is the node's, every run's: two builds in it at once break each other's crates
  RESEARCH_RUN_DIR=$LOGS/m0-build FLOCK_WORK=$FW MODE=build GPU=0 flock "$FW/.build.lock" bash verity/protocols/proof/flock/prover/pod/60-circuit.sh || return 6
  # in a work dir of its own: step 3's run.sh builds $FW's again with GPU=0 before V3c's audit
  if [ -n "${V3C_GPUS:-}" ]; then
    local tk
    tk=$(v3c_cuda) || return 6
    RESEARCH_RUN_DIR=$LOGS/m0-build-gpu FLOCK_WORK=$FW/gpu CUDA_TK=$tk MODE=build GPU=1 flock "$FW/.build.lock" \
      bash verity/protocols/proof/flock/prover/pod/60-circuit.sh || return 6
  fi
  # after verity-verify-proof's build, whose package it requires by path: the inference-only verdict's `inference-only-verifier --count`,
  # without which the verdict is the Python count's alone
  (cd verity/protocols/policy/training_detection/inference_only/verifier/lean && lake build inference-only-verifier) || echo "inference-only-verifier didn't build"
}
if [ -n "$COMPOSE" ]; then ( unset PYTHONPATH; compose_build ) > "$LOGS/build.log" 2>&1 &
else ( unset PYTHONPATH; build ) > "$LOGS/build.log" 2>&1 &
fi
BUILD=$!
if [ -n "$COMPOSE" ]; then compose; exit $?; fi

if [ "${1:-}" = v3c ]; then  # an existing serve dir's window: D and K set, its advice written (step 3); V3c alone
  say "COINS V3c's audit: $(v3c_coins)"
  if [ -n "$FOLLOW" ]; then follow; exit $?; fi
  [ -n "${K:-}" ] && [ -f "$D/advice/w$K/private/advice.bin" ] || { say "V3C: set D and K, a serve dir and its window with advice"; exit 3; }
  [ -f "$WREG" ] || CUDA_VISIBLE_DEVICES='' weights
  [ -f "$WREG" ] || exit 13
  # the verifier's record at the class from this tree, whatever an earlier run left in D
  v3c_circuit || exit 12
  v3c_commit || [ -z "$SPLIT" ] || piece_unread "$(commit_unread)"
  wait "$BUILD"; rc=$?
  [ "$rc" = 0 ] || { echo "BUILD $(failed "$rc" "$LOGS/build.log")"; stop build "$rc"; }
  [ "$v3c_ok" = 1 ] || exit 16
  v3c_proved; rc=$?
  [ -z "$PEARLPLUS_TIE" ] || window_tie
  private_pieces
  exit "$rc"
fi
