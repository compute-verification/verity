#!/usr/bin/env bash
# One window of Llama-3.2-1B int8 served on an H200 through every subprotocol, each step's own
# verdict line in $OUT/verdicts.txt, and one record of every verdict. The chain, in order:
#  0. B: `verity-vllm predict row` writes the H200 int8 row's Build of record (ROW_ID, default ROW's name) from its declared
#     inputs alone into PRED ($D/predicted), before the lease and on the CPU: its request Programs, GP-01's workload and the
#     manifest. `predict compare ROW --predicted PRED` holds it to the traced Build ROW, the cross-check: PREDICT-COMPARE,
#     every Program digest and the manifest's digest, Program digest and identities, gates the serve.
#  1. serve: `verity-vllm serve-openai` of LLAMA32_1B_INT8 (REPO at REV, --precision PRECISION: verity-int8/Llama-3.2-1B-INT8, verity_int8) on one GPU
#     under `gpu-lease`, `--scheme rows-sha512 --seal --admit release-order --window-requests SLOTS`, its committer built from
#     B's manifest (PRED/manifest.json) and taps (HIDDEN_SO, NORM_TAP_SO), T tokens a step (--max-num-batched-tokens; T defaults
#     to H1_T, else 512, with or without H1); B-OF-RECORD says each sealed sampling block's record names B's manifest, and its
#     Program digest and manifest digest predict's (a record sealed before the seal recorded the manifest digest: the Program
#     digest alone). Each closed window is sealed as a
#     rows-sha512 row dir, and the seal opens each one's replay bundle (`deferred_rows.of_record`). Once it listens, before
#     the first request, the committer registers its weights (`prove_path.py weights-registration`: each weight parameter of
#     the checkpoint CKPT, REPO's snapshot at REV under HF_HOME, and the rotary table at the words the engine serves,
#     $D/private/cos_sin_cache.npy, once within the analytic table's bound, committed under salts sampled from the committer's
#     key $D/private/weights.key, 0600, neither of which a record copies); WEIGHTS says how many it roots and which it leaves unrooted, and
#     why, or NO WEIGHTS REGISTRATION why not, and then every weight read counts free.
#  2. requests: REQUESTS greedy requests (default SLOTS) from the certifier's traffic generator (`demos/vllm/serve_load.py --varied`:
#     prompts of PROMPTS tokens, at most MAX_TOKENS outputs), numbered under one release run and sent together (RATE a second),
#     so the engine's idle closes them into one window. With LOAD=fixed (default varied) they are one full window of fixed
#     length instead (`serve_load.py --fixed SLOTS --window-steps WINDOW_STEPS`): SLOTS requests of PROMPTS tokens, each
#     with ignore_eos and max_tokens WINDOW_STEPS, all prefilled at the window's first step and run through its last. With
#     LOAD=front there are none: CERTIFIER_FRONT's group, sent through the warden, is the run's one load and its execution
#     block the one the run audits (a period that admits a single block, say). A front that failed before a window
#     closed stops the endpoint at once, with no window to seal; one that failed after (its leak guard holding a file,
#     say) leaves its exit to the certifier's verdict, and the run goes on with the window it sealed.
#     CERTIFIER_FRONT runs within CERTIFIER_FRONT_TIMEOUT (default 20m), and a flag it passes to `certifier_piece.py front`
#     that this tree's front doesn't take, or a front whose endpoint isn't on the warden's id form (SERVE_ARGS without
#     `--protocol ids`), is refused before the lease. The
#     lease ends once every closed window is sealed. What names a window's requests (the serve dir's windows.jsonl,
#     seals.jsonl, requests.jsonl and serve.json, each window's PoUW calls record, the load's records, timings file and
#     log, and the certifier front's session, bundle and log, in the serve dir's private/, of which the CERTIFIER-FRONT
#     line says only `served_verdicts.py front-line`'s fields) stays in the serve dir on the node; what leaves it is their
#     aggregate ($OUT/serve/overhead.json, except with LOAD=front, whose warden-paced load no stock
#     control compares to) less the load's counts and span, and verdict lines that name no window and count none of a
#     window's own: no window's number, no count of the windows closed, sealed or refused, no engine step, request count or
#     user of one. Each line goes out through `served_verdicts.py public-lines`, in full to the serve dir's
#     private/verdicts.txt, and once the run is done, however it ends, `served_verdicts.py publish` holds in full in the
#     serve dir's private/out each file of $OUT that names or counts a window or holds a load's tokens or span (the
#     aggregate's, which `served_overhead.py table` reads there), its public form in its place. The advice registration's published copies
#     ($OUT/registration.json, the records audit's served-records) carry the advice's length padded to its compiler's
#     cap (`served_verdicts.py public-registration`), and the audit's log and run.sh's served path (its summary,
#     registrations, sessions and audit log, which carry the advice's length) stay in the serve dir, where the record reads
#     them.
#  3. sampled proofs, on the window `served_verdicts.py window` picks (the most requests, none carried across windows; the
#     pick stays in the serve dir's private/window.json, the WINDOW line saying only the class's SLOTS slots, and CLASSES
#     the window circuit's classes beside the certifier's scan class, CERTIFIER_SCAN, default W63, the one the gate serves): every
#     closed window's advice under the prove path's registered compiler and the weights registration (`--weights-of-record`),
#     then this window's request Programs (`tools/advice_compile.py`); V3b's window circuit (`prove_path.py window-circuit
#     --pins` the registration, and PIN's NAME=ROOT if set), its registered and free input bits at those
#     pins: at each class of CLASSES (default 4,32,4, V2's), each within WC_TIMEOUT (default 30m), all of which the run needs
#     (the class of record is counted only where CLASSES names `record`, and gates like any other); then the window
#     audited from its records alone
#     (`prove_path/run.sh` with PP_RECORDS=1, `records_audit.py`): Flock `--zk` on the sampled proof blocks behind the Lean
#     firewall, and the Lean verifier's verdict. Those proof blocks are the sealed window's step units (PP_PARTITION, default
#     layer 0's SiluMul_v2 at step 1, or at step 0 with MAX_TOKENS 1, whose window has no step 1), not the window circuit's.
#     Every unit of a template the served model's compiler of record lists as a transfer gate (prove_path.GATES_OF_RECORD)
#     is proved beside the drawn rest, which only a work:K law draws, so the preflight refuses a partition of another law
#     or without them; PP_SETS names their input sets.
#  4. PoUW (H1 credit, or with POUW=pearlplus Pearl+'s own served run, below), PoUS, the bandwidth audit and the network
#     certifier: `served_verdicts.py pieces` runs each one's piece
#     where POUW_CMD, POUS_CMD, BANDWIDTH_CMD or CERTIFIER_CMD names it, one at a time (the certifier's alone, the longest,
#     from the moment the window is picked, beside step 3, its lines said after the others'; split in two, the answer step
#     6's period check wrote, V3c having run before this step), each within <COMPONENT>_TIMEOUT
#     (else PIECE_TIMEOUT, default 60m), else names it MISSING with the command it lacks and its owner. No verdict is made
#     for a piece that didn't run or refused to; one that ran and wrote no answer errs. A piece that needs a GPU takes its
#     own `gpu-lease`, the serve's being over. Under PRODUCT=onsite, WIPE_CMD's, POUS_CMD's and BANDWIDTH_CMD's defaults
#     come from `onsite_piece_commands` in the file ONSITE_PIECES names (a path from TREE, or absolute), sourced before
#     the preflight, each kept where its variable is set; with ONSITE_PIECES unset there are none, and a file that isn't
#     there is refused. The file also gives `served_composite.sh` onsite's own cases. Set empty, a command is MISSING.
#     A subprotocol that runs on the serving engine (H1's `--protocols pouw:...`) goes in SERVE_ARGS, serve-openai's
#     further arguments, and its piece reads what it wrote in the serve dir. The wipe's piece runs in step 1 instead: with
#     WIPE_CMD set (empty, none), serve() calls it in TREE, under the serve's lease and before serve-openai starts, within WIPE_TIMEOUT
#     (default 20m), with (the serve dir D, `all`, $OUT/pieces/wipe), since the window isn't picked yet, and says its last
#     line as WIPE. The record reads its pieces/wipe/answer.json and composes the wipe only when that answer accepts and
#     pieces/wipe/served.json names this node as the server it ran on. A wipe that wrote no answer (refused, or failed
#     before its judge) leaves the wipe missing, one served elsewhere stays missing, and one whose answer rejects makes the
#     record reject: a failed wipe is never accepted.
#  5. the record: `served_verdicts.py record`: the all-protocols window report ($OUT/record/verdicts.json) and the block
#     record ($OUT/record/block-record.json), composed `pending` while any subprotocol is missing, `accept` only when every
#     one accepted on this node; any other server's answer is carried as elsewhere. The certifier's chain files that list
#     or count the window's requests (CERTIFIER_PRIVATE) then move to the serve dir's private/pieces/certifier. Then,
#     unless COMPOSITE is set empty,
#     remote's and onsite's Lean verdicts on the run's cases (`served_composite.sh`'s docstring). Their verdicts gate
#     nothing, but a composite that didn't run (a tree without composite.py) or whose Lean didn't answer fails the run.
#     Under COMPOSE_OFF_NODE=1, the default, neither is of record here: the served node is the prover's, so this record
#     says (its `deferred`) that the final record and the composite are composed off it, and in the composite's place
#     the run's last step, after V3c, writes the compose hand-off they are composed from (below, at `handoff`).
#     PRODUCT (default onsite) is the product the run composes: onsite every subprotocol; remote sampled proofs and PoUW
#     alone, its other pieces dropped (not run, no default command, the wipe's and the certifier's included; the
#     preflight refuses any onsite command set), its record composing their verdicts alone and its composite remote's.
#     Under PEARLPLUS_TIE PoUW's verdict is Pearl+'s, and no composite holds onsite's pieces beside it (onsite's work
#     subprotocol is PoUW's period case, which that run doesn't write; remote's composite of record, whose is Pearl+'s
#     run, holds no onsite piece): the preflight refuses PRODUCT=onsite there unless COMPOSITE is set empty, and the run
#     then composes the record alone.
#     Beside them, the inference-only verdict (`inference_only.py`), at IO_CLASS (default the first class of CLASSES but
#     `record`): its policy, (c, k, n) and every input to them, is written before the lease from the run's settings alone
#     (the class, REPO at REV, LEASE_MIN; under PEARLPLUS_TIE in serve(), under the lease before serve-openai starts, from
#     PEARLPLUS-CIRCUIT's statement of φ on the card, serve-phi.json, its circuit cut at Pearl+'s linears, as a BF16
#     checkpoint's needs), $OUT/inference-only-policy.json and its INFERENCE-ONLY POLICY line; once the
#     build is done, step 3's window circuit record at the class is judged by `NciCheck.Count`'s rule beside sampled proofs,
#     with `nci-check --count` where the build built it, into $OUT/record/inference-only.json (the judgment against n and
#     no count, n judged on the exact count) and its INFERENCE-ONLY line, in full with the exact counts in the serve dir's
#     private/inference-only.json. It gates nothing.
#  6. V3c, after the record since V2 doesn't read it: it proves the window circuit's own units at V3C_CLASS (default 4,32,4,
#     which CLASSES must hold). `prove_path.py window-commit`, within V3C_TIMEOUT (default 15m), commits the window as that
#     class circuit's units, the weights it serves under the committer's key, so each served as registered has the
#     registration's root; its circuit.json less its run fields is held to the verifier's own window-circuit record at the
#     class; then `records_audit.py audit --circuit --registered`, within V3C_AUDIT_TIMEOUT (default 30m), samples the
#     registration's members (V3C_LAW, default subset:1, stratified:1 under H1_TIE) on V3C_CHALLENGER's coins (below,
#     Coins; pinned on main's first-parent line: MAIN_FILE's, kept as $OUT/v3c-audit/main.txt, else MAIN_SHA's) and
#     proves them, its composes made before it with V3C_WARM_JOBS (at `v3c_audit`); then
#     `records_audit.py period` writes the audit as the network certifier's period reads it ($D/v3c-w<K>-<class>/period.json,
#     and a copy in $OUT/v3c-audit: the registration, the Program, its answer and its free inputs), and V3c fails where it
#     writes none. Under H1_TIE (below and at `v3c_audit`) the window commit is the one made at H1's close, its
#     registration held to the record H1 bound and its circuit to the verifier's record reading H1's W's tables (step 3
#     writes it after the classes, `window-circuit --window-registration W`), none is made here, and the audit is tied to
#     H1's W;
#     V3c then runs before step 5's composite, whose one-stage and policy cases are its audit's, unless this node has less
#     memory than the audit needs (below: the audit and the composite are then deferred to a follow-on).
#     `served_verdicts.sh v3c` runs only that step on an
#     existing serve dir (D and its window K set), the build first, its limits 120m and 180m; it serves nothing, so it needs
#     no row dir, taps or gpu-lease (VENV the tree's .venv on a node without vLLM's). Untied, window-commit registers the
#     window at V3C_LAW's sampling design, the one its audit samples by (R1 holds the sample to it). V3C_CLASS set empty leaves
#     V3c out of the run, for a `served_verdicts.sh v3c` started once the window's advice is written.
#     With the network certifier's piece split in two, CERTIFIER_CHAIN_CMD and CERTIFIER_PERIOD_CMD (CERTIFIER_CMD then
#     unset, and V3C_CLASS the certifier's scan class, which CLASSES holds), V3c runs before step 4, since the certifier's
#     answer reads its period.json: the chain after the window commit, beside V3c's audit and period.json, within
#     CERTIFIER_TIMEOUT (default 4h), and the period check once both are done and V3c wrote period.json, within
#     CERTIFIER_PERIOD_TIMEOUT (default 60m), each as `bash -c CMD _ D K DIR V3C` in TREE with DIR $OUT/pieces/certifier
#     and V3C V3c's $D/v3c-w<K>-<class>. V3c then takes the window commit, the longer of the chain and V3c's audit,
#     and the period check, which the run's --timeout holds beside the other steps. The period check writes the
#     certifier's answer.json in DIR, which step 4's certifier piece reads; `served_verdicts.sh v3c`, which writes no
#     record, moves the steps' logs (CERTIFIER_PRIVATE) to the serve dir's private/pieces/certifier before it exits.
# Composed off the served node: with COMPOSE_OFF_NODE=1 (the default; 0 composes both on the served node, the composite
# then a development check that claims no record, `served_composite.sh`, which refuses one there under 1) the run's
# last step (`handoff`, after V3c in every run, so the composite reads V3c's audit's cases where it ran, the class window
# of record, and step 3's audit's only in a run without V3c) writes the serve dir's private/compose: the
# record's pieces (only the files the record reads of each, Pearl+'s verdict of record's among them), the inference-only
# verdict, the settings the follow-on composes at (compose.env: the window and its engine steps, the product, COMPOSITE,
# ALPHA, GAMMA, the served node, the serve dir's path, COMPOSE_NODE, the tree's commit and the host V3c's audit ran on)
# and its every input relative to the serve dir
# (`handoff_inputs`: verifier-side files alone, the records audits' cases and the blobs they name, H1's log and records,
# PoUW's period case, sampled proofs' summary and registration), listed (inputs.txt) and hashed (inputs.sha256). A list
# holding a served run's witness (`served_verdicts.py witness`: a value file, a prover's instance, private or salts file,
# the advice, a key, a row dir, the requests) or a blob outside the serve dir is refused, and the run fails. Those files alone
# are copied to the same path on COMPOSE_NODE, an auditor host the prover doesn't control and where no witness goes
# (default vy-auditor-2; never a prover node such as a deferral's follow-on), outside any run dir (`served_verdicts.sh
# copy compose`), and there `served_verdicts.sh compose` with D alone holds the copy to its sums, rebases the cases'
# blobs onto a copy at another path (`served_verdicts.py rebase`), runs at the served node's tree (another only with FOLLOW_ANY_TREE=1), refuses the served node itself and a serve dir
# holding any witness, needs no row dir, vLLM venv, card or gpu-lease, and composes the final record (`record --off-node
# --steps --sampled-lean`: composed_on the host it ran on, served_node false, from the hand-off, sampled proofs' row this
# host's own run of its Lean verifier on the case's proofs, each other row's answer as the served node wrote it, and each
# row naming which in its answer_from) and the composite (`served_composite.sh`,
# SERVED_NODE the served node, COMPOSE_HANDOFF set); its $OUT is published as every run's is. PRODUCT=onsite's composite
# reads its own pieces' cases (PoUS's, the wipe's), which their pieces write in the served node's serve dir and which
# stay on the prover's side (`served_case.py`), so no hand-off carries them: under COMPOSE_OFF_NODE=1 the preflight
# refuses it unless COMPOSITE is set empty, and the final record alone is composed there.
# `served_verdicts.sh handoff [HOST]`, in TREE with D, writes the hand-off
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
# (`verity/protocols/poub/device/live_verifier/host.py`'s `compose`: COMPOSE_GATE, default lane vllm's gate key as research@ on
# vy-auditor-2's public address, where the gate admits senders (set it for any other COMPOSE_NODE), the one word `compose`), a tar of the files its inputs.txt lists on the gate's stdin; the gate composes there at its
# pinned tree (COMPOSE_BUILT: built at its pin; SAMPLED_EXE its prebuilt sampled proofs Lean; COMPOSE_TREE_ON the branch
# infra pinned it on, which the record's composed_on names beside the commit) and answers its $OUT, written to
# $OUT/compose with the gate's pin in gate.json, one GATE-COMPOSE line saying what it composed.
# Coins: V3c's audit, the sampled-proofs audit of record, samples on V3C_CHALLENGER's coins, by default PP_CHALLENGER's: the
# live challenger with PP_CHALLENGER=gate:USER@HOST (records_audit's, with its peer V3C_PEER, default PP_PEER, the node
# it runs on, PP_CHALLENGER_KEY and PP_GATE_KEY), where it runs, on the served node or a deferral's follow-on. There
# `served_verdicts.py gate --root` first finds no session on the window's run root (the gate opens a root once, ever) and
# no period of lane H1_LANE open (a reserve would join it), its V3C-GATE line saying so and when. A local or loopback
# session is a development session, which the preflight refuses unless DEVELOPMENT=1 (a development run), as it refuses a
# V3C_PEER other than the node's peer H1's or Pearl+'s session names, and in a follow-on the served node's. Sampled
# proofs' step 3 samples on a local development challenger session's coins on the served node (run.sh's PP_CHALLENGER
# default, local, pinned on PP_MAIN's line, else MAIN_FILE's, else origin's main's) unless PP_CHALLENGER names the live
# challenger (run.sh's gate:USER@HOST with PP_PEER, PP_CHALLENGER_KEY and PP_GATE_KEY), and then too while V3c's audit is
# live, since both open on the window's run root: step 3 is then a development check. With V3c off (V3C_CLASS empty) step
# 3's audit is the one of record, and the preflight refuses its local or loopback session unless DEVELOPMENT=1. A record
# composed after V3c (a follow-on's, or the compose's from its hand-off) reads V3c's audit where it wrote its summary,
# else step 3's (`sampled_of_record`). With H1_GATE (the lane's gate command, by default LANE_GATE below: ssh as
# research on the gate's host under the lane's own key $HOME/.ssh/vllm_lane_ed25519, which the gate registers
# under lane vllm, so each reserve reads by vllm; COMPOSE_GATE keeps vllm_gate_ed25519; empty, no
# H1) the endpoint serves PoUW H1 (--protocols pouw:eps1-sm120-h1fx1 --challenger-gate H1_GATE --challenger-peer H1_PEER,
# default vy-dc-h200, at the engine's T and S WINDOW_STEPS 16, the Program build H1 affords), each window's coins from a
# session of its own on vy-challenger-1, once `served_verdicts.py gate` finds no period of lane H1_LANE (default vllm,
# the lane on H1_GATE's key's line) open (a reserve would join it). H1_PROTOCOLS_CONFIG (a JSON object or file; unset,
# none) is serve-openai's --protocols-config, given to it and to `served_verdicts.py program` alike, so the offline
# Program is the one it declares: {"pouw": {"unit": "j10r", "ks": [2048]}} declares J10-R's tile unit over the
# linears at K 2,048 alone, the form a period's guarantee covers. SERVE_ARGS carries no --protocols-config of its own.
# PoUW's piece (POUW_CMD's default then, `served_verdicts.py pouw`) checks the window's coins against the gate's session,
# ledger and pins under H1_KEY_SHA512 (vy-challenger-1's key) and main's first-parent line: MAIN_FILE's (a file, kept as
# $OUT/main.txt: `git rev-list --first-parent origin/main` of a fresh fetch, which PEARLPLUS_TIE's preflight requires
# under a live Pearl+ gate, since the gate's pin need not be on this tree's own line), else MAIN_SHA's
# (default the tree's HEAD). That key is a development key, so the run is a development run and credits nothing. The COINS line says which.
# With H1_CIRCUITS too (a directory), before serve-openai starts (its startup reserves window 0's session, which joins
# only a period already open), `served_verdicts.py program D/h1` writes the Program serve-openai will declare for REPO at
# T and WINDOW_STEPS, `pouw_h1_circuits --program D/h1 --out H1_CIRCUITS --live T,WINDOW_STEPS --strata H1_STRATA
# [--lanes H1_LANES]` builds its circuit files (the live file, only the strata H1_STRATA lists, default none, and each K
# of H1_LANES, default none, in credit's lanes form: lane<K> and combine<K>; a file already there under the same bind is
# reused), then `served_verdicts.py period open` opens period H1_PERIOD (default served-RUN_ID) on its
# digest, its sample K H1_DRAW_K (default `pouw_h1_challenger.draw_k`'s), its weights H1_MODEL's registration (default
# none, which credit refuses), and its fixed length SLOTS requests of MAX_TOKENS (`--cap 2`: one window and its sample session);
# `period served`
# holds the Program serve-openai then declared to it, and `period close` closes it once the windows are sealed, or
# when the endpoint never listens. The PERIOD lines say what it committed; $OUT/serve keeps period.json, the Program
# (h1/window-program.json.gz and its meta) and H1_CIRCUITS' circuit-files.json. H1_STATEMENTS and
# H1_LIVE_EVIDENCE (files) go to PoUW's piece as its --statements and --live-evidence, the evidence core's credit reads.
# H1_RECEIPTS=1 serves with --h1-receipts: each window registers each call's receipt table receipt@<c>, which a drawn
# J10-R call statement's o0 and a lanes-form K's combine read (drawn refuses to stage either without it).
# POUW_CMD='bash benchmarks/one_stage/served_pouw.sh "$1" "$2" "$3"' makes both on the node (the window's live rounds and
# sampled proof blocks proved) and writes the period's case beside the piece, the one remote's composite reads.
# With H1_TIE set too (and H1_CIRCUITS, V3C_CLASS not empty), V3c's window commit moves into the serve's close: at each H1
# window's close, once it is sealed and before its sample coin, serve-openai's `--h1-window-commit` runs
# `served_verdicts.sh tie` on the served node, under the serve's lease (LEASE_MIN then defaults to 150 and LEASE_MEM_GB to
# 160, the serve's 128 and window-commit's peak): the window's advice, the one step 3 then reads, and `prove_path.py window-commit --window-registration W --h1-key
# PROVER/key --law V3C_LAW` within V3C_TIMEOUT (then default 60m, at most 90m), whose registration R, chained right after
# H1's W, H1 names as the window's Commit record (`serve_tie`). Its output stays in the serve dir's h1/w<k>/. H1's sessions
# are reserved for the gate's default 120 minutes, the window's sample session at the serve's start, and the sample coin
# waits for R. window-commit makes each Program's Build itself (`own_builds`), since step 3's compile writes the row dir's
# only once the serve is done. Step 6 then audits that commit and makes none of its own: a window H1 bound no R for has no
# V3c, and its circuit is held to step 3's record at V3C_CLASS reading W's tables, which gates nothing. It runs
# before the composite, which composes the cases on R: V3c's audit's, the policy's (its log the serve's h1/log.jsonl)
# and PoUW's period case once each of its windows names R (`served_cases.py`). The
# preflight refuses H1_TIE on a tree whose window-commit takes no --window-registration.
# DAY=N (off by default: unset, serve-openai writes what it wrote before) declares day N's one circuit in DAY_LOG, the
# deployment's registration log kept across runs, before the model loads (serve-openai's --day, `serve_day`): DAY_PROGRAM,
# the SHA-512 of the program the day's windows register, DAY_BLOCKS its block Definitions, and DAY_PARAMS the log's
# {auditor, composite} (needed for a new log). A second declaration of the day, to another circuit, is refused by the
# log's check (duplicate-declaration) and the serve doesn't start; the same one again (a rerun within the day) is the day's.
# Each H1 W names the declaration, an R of another program is refused at its close, and V3c's audit runs R2 against the
# log's declarations (records_audit's --day). Under H1_TIE DAY_PROGRAM is V3C_CLASS's window circuit's at H1's tables, R's
# `program` (h1/w<k>/record.json), which the model, the class and H1's layout alone fix. A deferral carries the day: DAY
# in its settings, and DAY_LOG and DAY_PARAMS as the serve left them, copied to private/deferred/day-log.json and
# day-params.json and hashed with the serve's declaration.json among its inputs. Its follow-on audits against that copy
# alone, once the sums hold and the copy holds the declaration declaration.json names by digest (DAY-HELD).
# Under H1_TIE on a node with less memory than V3c's audit needs (its MemTotal, NODE_MEM_GB, under V3C_AUDIT_MEM_GB, default
# 290), step 6 holds the tie (WINDOW-COMMIT, WINDOW-CIRCUIT-HELD) and defers the audit (and, under COMPOSE_OFF_NODE=0, the
# composite; `defer`), under either product. Onsite's own pieces have each written its case in the serve dir by then (the
# wipe's in step 1, the others' in step 4), and before an onsite run's deferral an ONSITE-CASES line gives, for each of
# onsite's own subprotocols, the verdict of the answer its piece wrote beside its case (`served_verdicts.py onsite-cases`).
# The deferral's list holds none of the cases onsite's composite reads of them (PoUS's, the wipe's), so the preflight
# refuses a deferral of onsite's composite: PRODUCT=onsite defers with COMPOSITE set empty.
# Step 5's record says what was deferred and why (its `deferred`), a DEFERRED line says so, and the serve dir's
# private/deferred gets the pieces the record read, the inference-only verdict (which the follow-on's record carries as
# the served node wrote it), the settings the follow-on runs at (deferred.env: the window, the
# product, V3c's class, law and coins (V3C_CHALLENGER, DEVELOPMENT), H1_TIE, COMPOSITE, ALPHA, GAMMA, the served node, its
# peer, the tree's commit, COMPOSE_OFF_NODE and COMPOSE_NODE; a deferral that names no COMPOSE_OFF_NODE follows on at 0),
# and the
# follow-on's every input relative to the serve dir (`deferred_inputs`), listed (inputs.txt, which names inputs.sha256
# too) and hashed (inputs.sha256). The copy is the prover's: those files alone, to the same path on a node with the
# memory, outside any run dir. There `served_verdicts.sh v3c` with D alone (deferred.env names the window, and an
# environment that sets one of its settings to another value is refused) holds the copy to its sums, runs at the served
# node's tree (another only with FOLLOW_ANY_TREE=1, which its WINDOW-CIRCUIT line names, since it reads that tree's
# window-circuit record), needs no row dir, vLLM venv, card or gpu-lease, and after V3c's audit writes the compose hand-off
# there (above; a compose follow-on refuses a serve dir holding the copy's value files, so COMPOSE_NODE is then another
# node), or under COMPOSE_OFF_NODE=0 composes the final record (`record --node` the served node, so its composed_on names
# the host it composed on, served_node false, and that it read a prover-side copy) and the composite
# (`served_composite.sh`, SERVED_NODE the served node); its $OUT is published as every run's is.
# POUW=pearlplus (default h1) makes Pearl+ PoUW's piece instead: H1_GATE defaults empty (set, the preflight refuses
# it), and POUW_CMD to 'bash benchmarks/one_stage/served_pearlplus.sh "$1" "$2" "$3"' within POUW_TIMEOUT 120m, which
# once the window is audited serves Pearl+ (pouw:pearl-c-h100-v1-h3, Llama 3.2 1B's BF16 checkpoint) on this card under
# its own lease, every pass retained in the serve dir's private/, and takes Pearl+'s verdict over them (verify.json in
# the piece's dir; first, by default, it gives up the passes' rows over a debit cap from the card's counts, V-EX): a development
# piece, since its passes are not the window sampled proofs audited (its header has the rest). Its epoch's salt coin is
# a live session's of its own on vy-challenger-1: PEARLC_CHALLENGER_GATE defaults to the lane's gate command (H1_GATE's
# default under h1), PEARLC_CHALLENGER_PEER to H1_PEER and PEARLC_CHALLENGER_LANE to H1_LANE's, `served_verdicts.py gate`
# must find no period of that lane open, as for H1, and the piece holds each pass's salt to the session under
# H1_KEY_SHA512 and MAIN_FILE's line, as H1's piece does. The piece opens the lane's Pearl-C period right before its
# retained serve and closes it right before its verifier's draw, a draw session's of its own on that gate, reserved as
# the lane's first reserve after the close and opened on the run's commitment (served_pearlplus.sh's steps 2 and 4);
# while the period is open its cap refuses every other reserve of the lane. PEARLC_CHALLENGER_GATE set
# empty leaves Pearl+'s coins an
# in-process challenger's: the COINS line, the run's first, leads with that, and the piece claims no verdict.
# With PEARLPLUS_TIE=1 too, the window itself is served under Pearl+, the way H1_TIE ties H1's: serve-openai serves
# PEARLPLUS_ARM (default pouw:pearl-c-h100-v1-h3) on its one engine (`serve_pearl`, --protocols with --protocols-config
# naming the cubin PEARLPLUS-CUBIN built before the lease), one Pearl+ epoch a Commit window beside sampled proofs cut at
# Pearl+'s linears, so the window's linears are the Pearl+ matmuls whose passes are committed, each window's retained in
# the serve dir's pearl/w<k>/. Each epoch's salt coin is a live salt-only session of its window's own on vy-challenger-1
# (`serve_pearl.live`: serve-openai's --challenger-gate PEARLC_CHALLENGER_GATE, --challenger-peer PEARLC_CHALLENGER_PEER
# and --challenger-lane PEARLC_CHALLENGER_LANE, defaulted as above), reserved under the lane as the window opens and
# joining its open Pearl-C period; PEARLC_CHALLENGER_GATE set empty leaves them in-process, and the piece claims no
# verdict. Its defaults are then Llama 3.2 1B's
# BF16 checkpoint (REPO unsloth/Llama-3.2-1B, PRECISION bfloat16, the arm's), the BF16 B4 row's traced Build under
# /workspace/research/vllm-pearlc-h200bf16 as ROW, LOAD=fixed at V3C_CLASS's load (SLOTS its slots, PROMPTS P:P at its
# prompt P, WINDOW_STEPS its outputs: round s of W is the window's s-th step, the class's s-th output, and the cut circuit
# reads Pearl+'s tables at those rows; the preflight refuses another), PEARLC_BF16_ROWS=0 and V3C_LAW stratified:1, and
# step 3's partition layer 0's RoPE heads, the BF16 1B's transfer gates of record, every one proved, beside its
# SiluMul_v2 under work:2 at step 1 (llama32_1b_tg_l0_step1.json, PP_SETS rope-head/d64/bf16). The preflight refuses
# another PRECISION or row (its id names __bf16__), a V3C_CLASS CLASSES doesn't count, vLLM's prefix cache (SERVE_ARGS'
# --prefix-caching: a hit leaves rows of a prompt uncomputed where the cut circuit reads Pearl+'s tables), a
# certifier front (its group closes into a window of its own), and a serve dir whose filesystem has fewer than
# PEARLPLUS_INODES inodes free (default 772388, two commits of the tied window at the class of record, three files a
# member; 0 checks none), naming the shortfall; earlier runs' Pearl+ windows are deleted with `research retention rm`. Step
# 3's window circuit at each class is the one cut at Pearl+'s linears (`window-circuit --pearl-tables`), the verifier's
# own record of the circuit the window is committed as. The same record at V3C_CLASS is built before serve-openai starts,
# under the lease, from what its serve.json will record of φ on this card (`prove_path.py serve-phi`, then `class-circuit`,
# no advice: the circuit is the class's whatever the traffic), into the serve dir's pearlplus/circuit-record.json
# (PEARLPLUS-CIRCUIT, within PEARLPLUS_CIRCUIT_TIMEOUT, default 30m), and step 3 says whether its record is that one.
# With a gate, serve() then opens the lane's Pearl-C period PEARLPLUS_PERIOD (default pearlplus-RUN_ID) on that record
# (`served_verdicts.py pearlplus-period open --circuit-record`, cap 1: the window's one salt session joins it; the
# PEARLPLUS-PERIOD lines) before serve-openai starts, and closes it once the windows are sealed and the endpoint stopped
# (or when it never listens, or is stopped), so before Pearl+'s piece draws: open, serve, seal, close, draw. The close
# first reads the ledger, waiting up to PEARLPLUS_CLOSE_WAIT seconds (default 300) for it to end each window's salt
# session, since the gate closes a period only once its windows have ended: the lane's reserves since the open must be
# the windows' salt sessions alone (pearl/w<k>/served.json's), its line saying held or REFUSED, and the piece draws only
# after a period that closed with its reserves held; once the coin is drawn, its RULE line says whether the draw session
# was the lane's first reserve after the close. The record
# reads the period's circuit against the window-circuit record V3c held R's circuit to (`window_tie` keeps it beside R).
# Pearl+'s piece then serves
# nothing of its own (served_pearlplus.sh's PEARLPLUS_TIE): it takes the window's epoch pearl/w<k>/, writes W
# (`retain --window-record`), and runs `served_verdicts.sh pearl-tie` (PEARLPLUS_TIE_CMD) before its coin: the window's
# advice and `prove_path.py window-commit --window-registration W --pearl-passes DIR --law V3C_LAW`, the circuit cut at
# Pearl+'s linears and R chained right after W, in the serve dir's pearlplus/w<k>/; then `verify --tie W`. V3c then runs
# after step 4 and before the record (on a node short of its audit's memory, deferred as under H1_TIE): it audits that
# commit, held to W (WINDOW-COMMIT), with records_audit's --tie W (V3C-TIE), and the record reads R and that audit's
# summary from pearlplus/w<k>/tie (`record --window-tie`), where `pearlplus_of_record` holds them to W.
# Once Pearl+'s piece wrote W, before the record, the served compare holds the serve to the row of record on the CPU
# (`served_compare.py`, within SERVED_COMPARE_TIMEOUT, default 30m): PHI, serve.json's φ is ROW_ID's declared φ and lowers
# the same window circuit at V3C's class cut at Pearl+'s linears; BUILD, PRED is the row's Build of record BUILD_OF_RECORD
# (PROGRAM_DIGEST:MANIFEST_DIGEST); CUT, W registers exactly that circuit's Pearl+ values. Its last line, SERVED-COMPARE:
# EQUAL or NOT EQUAL, is the one the record names (`record --served-compare`, its served-compare/served_compare.json
# carried to a follow-on and the compose hand-off), and NOT EQUAL fails the run.
# A step that can't run says why, and the chain stops there. Flock, Lean's flock-verify and sampled-proofs-verifier and the
# tree's venv are built while the endpoint serves, Flock in FLOCK_WORK (kept on the node, so a rerun reuses its build).
#   research run --on vy-dc-h200 --project verity --source TREE --cwd clone --timeout 6h -- \
#     bash benchmarks/one_stage/served_verdicts.sh
# SETTINGS=PATH names a file of NAME=VALUE lines on the node, read before every default below: the run's job record
# publishes its launch command, so a load's counts (REQUESTS, PROMPTS, SLOTS, CERTIFIER_FRONT's) go in the file, and the
# command names only its path.
# PREFLIGHT_ONLY=1 runs the preflight alone (the settings, the row dir and taps, gpu-lease, the free inodes under
# PEARLPLUS_TIE, the front's flags, H1_TIE's and a follow-on's checks), says what the run would do (PREFLIGHT passed: its lease, or its deferral) and exits 0 before the
# serve dir is made, kept or anything built.
# `served_verdicts.sh circuits K`, in TREE with D and OUT, counts serve dir D's window K's circuits alone (its advice written).
# `served_verdicts.sh inference-policy`, in TREE with D and OUT, writes the inference-only policy at IO_CLASS alone, to
# IO_POLICY (default $OUT/inference-only-policy.json), before a serve; `served_verdicts.sh inference-only K` judges window
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
if [ -n "${SETTINGS:-}" ]; then
  [ -f "$SETTINGS" ] && [ -r "$SETTINGS" ] || { echo "PREFLIGHT: SETTINGS=$SETTINGS is not a readable file" >&2; exit 3; }
  set -a
  # shellcheck source=/dev/null
  . "$SETTINGS"
  set +a
fi
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
TREE=$PWD
ROOT=${SERVE_ROOT:-/workspace/research/vllm-served-verdicts}
D=${D:-$ROOT/${RESEARCH_RUN_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)}}
# step 6's deferral (`defer`): the settings the follow-on runs at, its inputs' list and sums, and the pieces the record read
DEFER_DIR=$D/private/deferred
FOLLOWS="K PRODUCT V3C_CLASS V3C_LAW V3C_CHALLENGER DEVELOPMENT H1_TIE PEARLPLUS_TIE COMPOSITE ALPHA GAMMA SERVED_NODE SERVED_PEER DEFERRED_TREE COMPOSE_OFF_NODE COMPOSE_NODE DAY"
# the compose hand-off (`handoff`): the settings its follow-on composes at, its inputs' list and sums, and the pieces
COMPOSE_DIR=$D/private/compose
COMPOSES="K PRODUCT PEARLPLUS_TIE COMPOSITE ALPHA GAMMA SERVED_NODE SERVED_DIR COMPOSE_NODE COMPOSE_TREE AUDITED_ON SAMPLED_AUDIT STEPS"
FOLLOW= COMPOSE= CLASH=
follow_env() {  # $1's settings, those named in $2, read over the environment's, CLASH naming each the environment set otherwise
  declare -A given=()
  for v in $2; do [ -z "${!v+x}" ] || given[$v]=${!v}; done
  set -a
  # shellcheck source=/dev/null
  . "$1"
  set +a
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
  [ ! -f "$COMPOSE_DIR/compose.env" ] || follow_env "$COMPOSE_DIR/compose.env" "$COMPOSES"
fi
PEARLPLUS_TIE=${PEARLPLUS_TIE:-}
if [ -n "$PEARLPLUS_TIE" ]; then  # the window served under Pearl+: Llama 3.2 1B's BF16 checkpoint, whose linears Pearl-C reads
  HB=/workspace/research/vllm-pearlc-h200bf16
  ROW=${ROW:-$HB/r20261104-084130-0f59/llama32-1b__bf16__h200__tp1__b4__i256__o32__mixed__greedy__bi-eager}
  REPO=${REPO:-unsloth/Llama-3.2-1B} REV=${REV:-9535bd9b1d1dea6acafbdc4813b728796aeb28da} PRECISION=${PRECISION:-bfloat16}
  LOAD=${LOAD:-fixed} PEARLPLUS_ARM=${PEARLPLUS_ARM:-pouw:pearl-c-h100-v1-h3}
  # the class's fixed load, at whose rows the circuit cut at Pearl+'s linears reads its tables
  IFS=, read -r slots prompt _ <<< "${V3C_CLASS-4,32,4}"
  SLOTS=${SLOTS:-$slots} PROMPTS=${PROMPTS:-$prompt:$prompt}
  # sm_90's BF16-rows path off unless PEARLC_BF16_ROWS says otherwise (e2e.BF16_RACE)
  export PEARLPLUS_TIE PEARLC_BF16_ROWS=${PEARLC_BF16_ROWS:-0}
  # the BF16 1B's compiler of record lists its RoPE heads as transfer gates, every unit of which step 3 proves (work:K)
  PP_PARTITION=${PP_PARTITION:-benchmarks/prove_path/llama32_1b_tg_l0_step1.json} PP_SETS=${PP_SETS:-rope-head/d64/bf16}
  # the row's Build of record, PROGRAM_DIGEST:MANIFEST_DIGEST, which the served compare holds PRED to (BUILD)
  BUILD_OF_RECORD=${BUILD_OF_RECORD:-95ba0fab376a62f5421ed21ced4e650afa516531508ecb802d065f44fd4f8ab451ba422c75312d8052b9087ff28826aa43edfc945435528856b97c8952903eb7:65ce5a6ccc1fd943c23a6d78b4eb80936d32c1043306ec3aab640b279eee419ef778205bd83b81721ad2c301114bd0e338e04b95acc959f8ba1188dcfa1ceac3}
fi
H=/workspace/research/vllm-composed-h200int8
ROW=${ROW:-$H/r20261030-210701-a20f/llama32-1b__int8__h200__tp1__b4__i256__o32__mixed__greedy__bi-eager} ROW_ID=${ROW_ID:-${ROW##*/}}
HIDDEN_SO=${HIDDEN_SO:-$H/fa2/build/fa3_matReqG/verity_fa3_matReqG.so} NORM_TAP_SO=${NORM_TAP_SO:-$H/norm_tap/build/verity_norm_tap.so}
VENV=${VENV:-/workspace/cp/venv312}
REPO=${REPO:-verity-int8/Llama-3.2-1B-INT8} REV=${REV:-15b0976d400d796ee589e7737d03a919526bf800} PRECISION=${PRECISION:-verity_int8}
PORT=${PORT:-18190} SLOTS=${SLOTS:-4} PROMPTS=${PROMPTS:-16:24} MAX_TOKENS=${MAX_TOKENS:-4} RATE=${RATE:-1000} SEED=${SEED:-1}
REQUESTS=${REQUESTS:-$SLOTS} CLASSES=${CLASSES:-4,32,4} CAP=${CAP:-8388608} MAX_MODEL_LEN=${MAX_MODEL_LEN:-512}
PP_PARTITION=${PP_PARTITION:-benchmarks/prove_path/llama32_1b_silu_mul_v2_i8192_l0_step$([ "$MAX_TOKENS" = 1 ] && echo 0 || echo 1).json}
T=${T:-${H1_T:-512}}
DISK_FLOOR_GB=${DISK_FLOOR_GB:-100}
PEARLPLUS_INODES=${PEARLPLUS_INODES:-772388}  # two commits of the tied window at the class of record, three files a member
H1_KEY_SHA512=${H1_KEY_SHA512:-27271eaf917d1a817198cbcfe7684debd9bdb11b9e386460948bb981642fab3ae065c0a877330a0da68ae76969550eb786dc172319876aaca21fd116e5a193bc}
POUW=${POUW:-h1}
H1_PEER=${H1_PEER:-vy-dc-h200}
LANE_GATE="ssh -i $HOME/.ssh/vllm_lane_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes research@204.12.178.84"
if [ "$POUW" = pearlplus ]; then
  H1_GATE=${H1_GATE-}
  export PEARLC_CHALLENGER_GATE=${PEARLC_CHALLENGER_GATE-$LANE_GATE} PEARLC_CHALLENGER_PEER=${PEARLC_CHALLENGER_PEER:-$H1_PEER} \
    PEARLC_CHALLENGER_LANE=${PEARLC_CHALLENGER_LANE:-${H1_LANE:-vllm}}
  # shellcheck disable=SC2016
  export POUW_CMD=${POUW_CMD-'bash benchmarks/one_stage/served_pearlplus.sh "$1" "$2" "$3"'} POUW_TIMEOUT=${POUW_TIMEOUT:-120m}
  [ -z "$PEARLPLUS_TIE" ] \
    || export PEARLPLUS_TIE_CMD="cd $(printf %q "$TREE") && bash benchmarks/one_stage/served_verdicts.sh pearl-tie \"\$@\""
else
  H1_GATE=${H1_GATE-$LANE_GATE}
fi
if [ -n "$H1_GATE" ]; then WINDOW_STEPS=${WINDOW_STEPS:-16}
elif [ -n "$PEARLPLUS_TIE" ]; then WINDOW_STEPS=${WINDOW_STEPS:-$(cut -d, -f3 <<< "${V3C_CLASS-4,32,4}")}  # round s the window's s-th step
else WINDOW_STEPS=${WINDOW_STEPS:-256}; fi
MAIN_SHA=${MAIN_SHA-$(git -C "$TREE" rev-parse HEAD 2>/dev/null)}
PRODUCT=${PRODUCT:-onsite}
COMPOSE_OFF_NODE=${COMPOSE_OFF_NODE:-1} COMPOSE_NODE=${COMPOSE_NODE:-vy-auditor-2}
COMPOSE_GATE=${COMPOSE_GATE:-ssh -i $HOME/.ssh/vllm_gate_ed25519 -o BatchMode=yes -o IdentitiesOnly=yes research@204.12.191.199}
export COMPOSE_OFF_NODE
if [ -n "${ONSITE_PIECES:-}" ]; then
  case $ONSITE_PIECES in /*) ;; *) ONSITE_PIECES=$TREE/$ONSITE_PIECES ;; esac
  [ -f "$ONSITE_PIECES" ] || { echo "PREFLIGHT: ONSITE_PIECES=$ONSITE_PIECES is not a file" >&2; exit 3; }
  # shellcheck source=/dev/null
  . "$ONSITE_PIECES"
  [ "$PRODUCT" != onsite ] || onsite_piece_commands
fi
export H1_KEY_SHA512 H1_GATE H1_PEER WINDOW_STEPS MAIN_SHA WIPE_CMD POUS_CMD BANDWIDTH_CMD PRODUCT ONSITE_PIECES POUW
PRED=${PRED:-$D/predicted}
SERVED_COMPARE=$D/served-compare/served_compare.json
FW=${FLOCK_WORK:-$ROOT/flock-work}
export PY=$VENV/bin/python HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export PYTHONPATH=$TREE:$TREE/adapters/vllm:$TREE/tools/research/src
export TORCH_EXTENSIONS_DIR=$ROOT/cache/torch-extensions TRITON_CACHE_DIR=$ROOT/cache/triton VLLM_CACHE_ROOT=$ROOT/cache/vllm
export VLLM_BATCH_INVARIANT=1 VERITY_RETAIN=host NATIVE_COLLECT_BUILD=$ROOT/cache/jit/nc
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
# Every verdict line goes out in its published form, naming no window and counting none of a window's own, and in full to
# the serve dir's private/verdicts.txt (`served_verdicts.py public-lines`), under the node's python3, the venv's absent
SV=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/served_verdicts.py SAY_PY=$(command -v python3 || echo "$PY")
publish_lines() { "$SAY_PY" "$SV" public-lines --private "$D/private/verdicts.txt" | tee -a "$OUT/verdicts.txt"; }
say() { printf '%s\n' "$*" | publish_lines; }
keep() { [ -d "$D" ] && (umask 077; mkdir -p "$D/private" && printf '%s\n' "$*" >> "$D/private/verdicts.txt"); }
count() { cat "$1" 2>/dev/null | grep -c .; }
vv() { "$PY" -m verity_vllm.pipeline.cli "$@"; }
case ${LOAD:=varied} in varied | fixed | front) ;; *) say "PREFLIGHT: LOAD is varied, fixed or front, not $LOAD"; exit 3 ;; esac
case $POUW in h1 | pearlplus) ;; *) say "PREFLIGHT: POUW is h1 or pearlplus, not $POUW"; exit 3 ;; esac
[ "$POUW" = h1 ] || [ -z "$H1_GATE" ] \
  || { say "PREFLIGHT: POUW=pearlplus is PoUW's piece, so the endpoint serves no H1, and H1_GATE is set: set it empty"; exit 3; }
case $PRODUCT in
  onsite) ;;
  remote)
    for v in WIPE_CMD POUS_CMD BANDWIDTH_CMD CERTIFIER_CMD CERTIFIER_CHAIN_CMD CERTIFIER_PERIOD_CMD CERTIFIER_FRONT; do
      [ -z "${!v:-}" ] || { say "PREFLIGHT: PRODUCT=remote runs no onsite piece, and $v is set: unset it"; exit 3; }
    done ;;
  *) say "PREFLIGHT: PRODUCT is onsite or remote, not $PRODUCT"; exit 3 ;;
esac
case $COMPOSE_OFF_NODE in 0 | 1) ;; *) say "PREFLIGHT: COMPOSE_OFF_NODE is 1 (the default) or 0, not $COMPOSE_OFF_NODE"; exit 3 ;; esac
CKPT=${CKPT:-$HF_HOME/hub/models--${REPO//\//--}/snapshots/$REV} WREG=$D/weights_registration.json
NO_WEIGHTS="NO WEIGHTS REGISTRATION: no window registers a weight's root, so every weight read counts free"
weights() {  # under the lease (its commit hashes on the leased GPU), before the first request: the committer's registration
  (cd "$TREE" && "$PY" benchmarks/prove_path/prove_path.py weights-registration --serve-run "$D" --checkpoint-dir "$CKPT" \
    --key "$D/private/weights.key" --out "$WREG") > "$OUT/weights_registration.txt" 2>&1
  local rc=$? log=$OUT/weights_registration.txt
  if [ "$rc" != 0 ] || [ ! -f "$WREG" ]; then
    say "$NO_WEIGHTS (weights-registration exit $rc: $({ grep -m1 'prove_path.py weights-registration:' "$log" || tail -1 "$log"; } | cut -c1-400))"
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
}
V3C=${V3C_CLASS-4,32,4} v3c_ok=0
[ "${1:-}" = v3c ] && V3C=${V3C:-4,32,4}
[ -z "${H1_TIE:-}$PEARLPLUS_TIE" ] || V3C_LAW=${V3C_LAW:-stratified:1}
vtag=${V3C//,/-}
PEARLPLUS_CIRCUIT=$D/pearlplus/circuit-record.json
if [ "${1:-}" = v3c ]; then V3C_TIMEOUT=${V3C_TIMEOUT:-120m} V3C_AUDIT_TIMEOUT=${V3C_AUDIT_TIMEOUT:-180m}
elif [ -n "${H1_TIE:-}$PEARLPLUS_TIE" ]; then V3C_TIMEOUT=${V3C_TIMEOUT:-60m} V3C_AUDIT_TIMEOUT=${V3C_AUDIT_TIMEOUT:-30m}
else V3C_TIMEOUT=${V3C_TIMEOUT:-15m} V3C_AUDIT_TIMEOUT=${V3C_AUDIT_TIMEOUT:-30m}; fi
V3C_WARM_TIMEOUT=${V3C_WARM_TIMEOUT:-$V3C_AUDIT_TIMEOUT}
V3C_AUDIT_MEM_GB=${V3C_AUDIT_MEM_GB:-290}
NODE_MEM_GB=${NODE_MEM_GB:-$(awk '/^MemTotal:/ {print int($2 / 1048576)}' /proc/meminfo)}
short_of_memory() {  # 0 iff this node has less memory than V3c's audit needs, saying both
  [ "$NODE_MEM_GB" -lt "$V3C_AUDIT_MEM_GB" ] || return 1
  echo "this node has $NODE_MEM_GB GB, under the $V3C_AUDIT_MEM_GB GB V3c's audit needs (V3C_AUDIT_MEM_GB)"
}
# V3c's audit's coins (`v3c_audit`, records_audit's --challenger): V3C_CHALLENGER, by default sampled proofs' PP_CHALLENGER,
# on the gate with its peer V3C_PEER (default PP_PEER: the node the audit runs on, as the gate's /etc/vy/peers names it,
# in a follow-on not the served node) under PP_CHALLENGER_KEY and PP_GATE_KEY. A local or loopback session's coins are a
# development session's, which only a development run (DEVELOPMENT=1) takes.
V3C_CHALLENGER=${V3C_CHALLENGER:-${PP_CHALLENGER:-local}} V3C_PEER=${V3C_PEER:-${PP_PEER:-}} DEVELOPMENT=${DEVELOPMENT:-}
v3c_live() { [[ $V3C_CHALLENGER == gate:* ]]; }
# Step 3's: PP_CHALLENGER's, but local while V3c's audit is live, since both open on the window's run root, which the
# gate opens once, ever, and V3c's is the sampled-proofs audit of record
SP_CHALLENGER=${PP_CHALLENGER:-local}
[ -n "$V3C" ] && v3c_live && SP_CHALLENGER=local
v3c_coins() {  # where V3c's audit's coins come from, as its COINS line says
  if v3c_live; then
    echo "the challenger $V3C_CHALLENGER (peer $([ -n "${DEFER:-}" ] && echo "the follow-on's node's" || echo "$V3C_PEER")), the window's run root's one session, the sampled-proofs audit of record"
  else
    echo "a development challenger session where it runs (V3C_CHALLENGER $V3C_CHALLENGER), in a development run (DEVELOPMENT=1)"
  fi
}
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
  # off the GPU, which none leases here or the engine holds under the tie: the weights hashed by h1.registered's NumPy
  # path, the reference
  (cd "$TREE" && CUDA_VISIBLE_DEVICES='' timeout -k 1m "$V3C_TIMEOUT" "$PY" benchmarks/prove_path/prove_path.py window-commit --serve-run "$D" \
    --window "$K" --row-dir "$D/windows/w$K" --advice "$D/advice/w$K/private/advice.bin" --class "$V3C" \
    --key "$D/private/weights.key" --registration "$WREG" --checkpoint-dir "$CKPT" --out "$VW/window" "$@") > "$json" 2> "$err"
  rc=$?
  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then echo "timed out after V3C_TIMEOUT=$V3C_TIMEOUT (exit $rc)"
  elif [ "$rc" != 0 ]; then echo "refused (exit $rc): $(grep -v '^ ' "$err" | tail -2 | cut -c1-800)"; fi
  return "$rc"
}
tied() {  # under H1_TIE: 0 iff H1 bound window K's R at its close (the serve dir's h1/log.jsonl) and R is $VW's registration
  "$PY" - "$D/h1/log.jsonl" "$K" "$VW/window/registration.json" <<'EOF' 2>&1
import json, sys
from verity.protocols.spot.verifier.one_stage import registration as REG
log, k, mine = sys.argv[1], int(sys.argv[2]), sys.argv[3]
try:
    bound = [x["digest"] for x in (json.loads(s) for s in open(log) if s.strip()) if x["window"] == k and x["kind"] == "record"]
except OSError:
    bound = []
if not bound:
    sys.exit("H1 bound no record R at the window's close, so V3c has no commit of it (the serve dir's h1/w<k>/window-commit.log)")
try:
    r = REG.digest(json.load(open(mine)))
except (OSError, ValueError):
    sys.exit("H1 bound a record R, and window-commit left no registration beside it")
if r != bound[-1]:
    sys.exit("H1 bound a record R other than window-commit's registration")
print(f"R {r[:16]}…")
EOF
}
# under PEARLPLUS_TIE: Pearl+'s window record W and the window commit tied to it (`pearl_tie`), in the serve dir's
# pearlplus/w<k>/, and where the record reads the tie (`window_tie`: R and V3c's audit's summary, record --window-tie)
pearl_dir() { echo "$D/pearlplus/w$K"; }
pearl_tied() {  # under PEARLPLUS_TIE: 0 iff Pearl+'s piece wrote W for window K and $VW's registration R is chained right after it
  "$PY" - "$(pearl_dir)/registration.json" "$VW/window/registration.json" <<'EOF' 2>&1
import json, sys
from verity.protocols.spot.verifier.one_stage import registration as REG
try:
    w = REG.digest(json.load(open(sys.argv[1])))
except (OSError, ValueError):
    sys.exit("Pearl+'s piece wrote no window record W (served_pearlplus.sh's retain --window-record), so V3c has no commit tied to it")
try:
    r = json.load(open(sys.argv[2]))
except (OSError, ValueError):
    sys.exit("Pearl+'s piece wrote W, and window-commit left no registration tied to it (the serve dir's pearlplus/w<k>/window-commit.err)")
if r.get("previous") != w:
    sys.exit(f"window-commit's registration R is chained after {str(r.get('previous'))[:16]}…, not right after Pearl+'s W {w[:16]}…")
print(f"W {w[:16]}…, R {REG.digest(r)[:16]}… chained right after it")
EOF
}
window_tie() {  # under PEARLPLUS_TIE: R, V3c's audit's summary and, once V3c held R's circuit to it, the window-circuit
  # record, in the layout `served_verdicts.py record --window-tie` reads
  local t
  t=$(pearl_dir)/tie
  mkdir -p "$t/window" && cp "$VW/window/registration.json" "$t/window/" 2>/dev/null \
    && { cp "$VW/audit/summary.json" "$t/" 2>/dev/null || true; } \
    && { [ "$v3c_ok" != 1 ] || cp "$D/$(v3c_record)" "$t/circuit-record.json" 2>/dev/null || true; }
}
served_compare() {  # under PEARLPLUS_TIE, once Pearl+'s piece wrote W and before the record, on the CPU: the serve held to
  # the row of record (served_compare.py), its SERVED-COMPARE line the one the record names (--served-compare)
  local dir=${SERVED_COMPARE%/*} rc
  [ -n "$PEARLPLUS_TIE" ] || { say "SERVED-COMPARE: not run: PEARLPLUS_TIE is empty, so no W ties the serve to the row"; return 3; }
  # an earlier compare's verdict never stands for this one's, which can die before it writes its own
  mkdir -p "$dir" "$OUT/served-compare" && rm -f "$SERVED_COMPARE" "$dir/circuit_facts.json"
  (cd "$TREE" && CUDA_VISIBLE_DEVICES='' timeout -k 1m "${SERVED_COMPARE_TIMEOUT:-30m}" "$PY" benchmarks/one_stage/served_compare.py \
    --serve "$D" --row "$ROW_ID" --class "$V3C" --pearl-tables "${PEARLPLUS_ARM#pouw:}" --pred "$PRED" \
    --of-record "$BUILD_OF_RECORD" --window-record "$(pearl_dir)/registration.json" --out "$dir") 2> "$dir/compare.err" \
    | publish_lines
  rc=${PIPESTATUS[0]}
  [ "$rc" -le 1 ] || say "SERVED-COMPARE: NOT EQUAL (served_compare.py exit $rc: $(tail -1 "$dir/compare.err" | cut -c1-400))"
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
v3c_record() {  # the verifier's window-circuit record V3c holds window K's commit to, relative to D: at V3C's class, and under
  # H1_TIE reading H1's W's tables of the window, as the commit H1 bound does (`v3c_circuit`)
  echo "window-circuit-w$K-$vtag${H1_TIE:+-tie}.json"
}
v3c_circuit() {  # in TREE: that record (`v3c_record`), within WC_TIMEOUT
  local head="WINDOW-CIRCUIT class $V3C" tie=()
  [ -z "${H1_TIE:-}" ] || { head+=" reading H1's tables"; tie=(tie); }
  CUDA_VISIBLE_DEVICES='' circuit "$V3C" "WC_TIMEOUT=${WC_TIMEOUT:-30m}" "$head" "${tie[@]}"
}
v3c_commit() {  # window K of $D committed as V3C's class circuit's units, its circuit held to the verifier's window-circuit record
  VW=$D/v3c-w$K-$vtag
  local json=$D/window-commit-$vtag.json why at= rec none=" (CLASSES '$CLASSES')"
  rec=$D/$(v3c_record)
  [ -z "${H1_TIE:-}" ] || none=" reading H1's tables"
  [ -f "$rec" ] || { say "WINDOW-COMMIT class $V3C: no window-circuit record at it$none"; return 1; }
  [ -f "$WREG" ] || { say "WINDOW-COMMIT class $V3C: $NO_WEIGHTS"; return 1; }
  if [ -n "${H1_TIE:-}" ]; then  # committed at H1's close (`tie`) and never again: the window's one commit, the one H1 bound
    json=$D/h1/w$K/window-commit.json
    why=$(tied) || { say "WINDOW-COMMIT class $V3C: $why"; return 1; }
    at=" at H1's close, its registration the Commit record H1 bound before the sample coin ($why)"
  elif [ -n "$PEARLPLUS_TIE" ]; then  # committed in Pearl+'s piece once W was written, before Pearl+'s coin
    json=$(pearl_dir)/window-commit.json
    why=$(pearl_tied) || { say "WINDOW-COMMIT class $V3C: $why"; return 1; }
    at=" in Pearl+'s piece, its registration chained right after Pearl+'s window record W before Pearl+'s coin ($why)"
  else
    why=$(window_commit "$json" "$OUT/window-commit-$vtag.err" --law "${V3C_LAW:-subset:1}") \
      || { say "WINDOW-COMMIT class $V3C $why"; return 1; }
  fi
  # the commit's population and sources count the window's own values: they stay in the serve dir
  say "WINDOW-COMMIT class $V3C$at: $("$PY" -c 'import json, sys; c = json.load(open(sys.argv[1])); print(json.dumps({k: c[k] for k in ("run", "circuit", "registration") if k in c}, sort_keys=True))' \
    "$json" 2>&1 | cut -c1-1500)"
  "$PY" - "$rec" "$VW/window/circuit.json" "$V3C" <<'EOF' 2>&1 | publish_lines
import json, sys
from verity.commitments.windows import interface as IF
mine, theirs = (json.load(open(f)) for f in sys.argv[1:3])
ok = IF.circuit_of(theirs) == mine
print(f"WINDOW-CIRCUIT-HELD class {sys.argv[3]}: the committed circuit.json less its run fields is the verifier's own "
      f"window-circuit record: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
EOF
  [ "${PIPESTATUS[0]}" = 0 ] && v3c_ok=1
}
# The audit's sessions listen on V3C_PORTS ports (default 16, a5's RANK_PORTS), a sample needing more refused before any
# listens: a subset:K sample opens one for each of at most K members, and records_audit's own default, a port for every
# member, finds no free span in `records_audit.PORTS` at a class window. Under H1_TIE the audit is tied to H1's W
# (records_audit's --tie W --tie-prefix h1/, `tie_of`): the tables the circuit reads are W's, R registers them as its
# outputs, and each sampled reader proves its rows at W's roots. Its sampling design is then stratified:K (V3C_LAW,
# default stratified:1), a stratum of each table's readers, so every table W registers has a sampled reader
# (Record.checks), and V3C_PORTS defaults to the size of R's sample, its strata's k summed: at most a session for each
# sampled proof block. The preflight refuses H1_TIE on a tree whose records_audit.py takes no --tie, or at another design.
# With V3C_WARM_JOBS=N, `records_audit.py warm` first composes every template of the committed window into
# $VW/compose-cache, N at a time, each in a process of its own, within V3C_WARM_TIMEOUT (default V3C_AUDIT_TIMEOUT): the
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
# tree is: staging takes a circuit's META, pin and class off the line its compose kept beside it, a drawn member's
# circuit is held to that line before its prover reads it (`a5.held`), and nothing else says what a template's circuit is.
v3c_warm() {  # every template of the committed window composed into $VW/compose-cache, V3C_WARM_JOBS at a time
  (cd "$TREE" && env -u PYTHONPATH RAYON_NUM_THREADS="$(nproc)" RUST_MIN_STACK=$((64 << 20)) timeout -k 1m "$V3C_WARM_TIMEOUT" \
    "$TREE/.venv/bin/python" benchmarks/prove_path/records_audit.py warm --window "$VW/window" --law "${V3C_LAW:-subset:1}" \
    --compose-cache "${V3C_COMPOSE_CACHE:-$VW/compose-cache}" --jobs "$V3C_WARM_JOBS") > "$OUT/v3c-audit/warm.log" 2> "$OUT/v3c-audit/warm.err"
  local rc=$? why
  if [ "$rc" = 0 ]; then why=$(grep -m1 '^WARM' "$OUT/v3c-audit/warm.log" | cut -c1-600)
  else
    if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then why="timed out after V3C_WARM_TIMEOUT=$V3C_WARM_TIMEOUT"
    else why=$(tail -3 "$OUT/v3c-audit/warm.err" | tr '\n' ' ' | cut -c1-800); why=${why% }; fi
    why+="; the audit composes the templates it didn't keep"
  fi
  say "V3C-WARM class $V3C (records_audit.py warm exit $rc, $V3C_WARM_JOBS at a time): $why"
}
v3c_audit() {  # the committed window audited from its records alone: its circuit and run root the verifier's, the weights pinned
  local run leanb m0 rc gl tie=() day=() cache=() ch=(--challenger "$V3C_CHALLENGER") ports=${V3C_PORTS:-16} main=$OUT/v3c-audit/main.txt
  [ -n "${DAY:-}" ] && day=(--day "$DAY" --day-log "$DAY_LOG" ${DAY_PARAMS:+--day-params "$DAY_PARAMS"})
  run=$("$PY" -c 'import json, sys; print(next(json.loads(l)["run_root"] for l in open(sys.argv[1]) if l.strip() and json.loads(l)["window"] == int(sys.argv[2])))' \
    "$D/windows.jsonl" "$K") || { say "V3C-AUDIT: no run root for the window in windows.jsonl"; return 1; }
  mkdir -p "$OUT/v3c-audit"
  if [ -n "${MAIN_FILE:-}" ]; then  # main's first-parent line, which the audit's challenger session pins on
    [ -s "$MAIN_FILE" ] || { say "V3C-AUDIT: MAIN_FILE $MAIN_FILE, main's first-parent line its challenger session pins on, is missing or empty"; return 1; }
    [ "$MAIN_FILE" -ef "$main" ] || cp "$MAIN_FILE" "$main"
  else
    git -C "$TREE" rev-list --first-parent "${MAIN_SHA:-HEAD}" > "$main" 2>/dev/null && [ -s "$main" ] \
      || { say "V3C-AUDIT: no first-parent line of main for its challenger session to pin on (MAIN_FILE unset, and ${MAIN_SHA:-HEAD} has none in this tree): set MAIN_FILE"; return 1; }
  fi
  if [ -n "${H1_TIE:-}$PEARLPLUS_TIE" ]; then
    if [ -n "${H1_TIE:-}" ]; then tie=(--tie "$D/h1/w$K/registration.json" --tie-prefix h1/)
    else tie=(--tie "$(pearl_dir)/registration.json"); fi
    [ -n "${V3C_PORTS:-}" ] || ports=$("$PY" -c 'import json, sys; print(sum(int(s["k"]) for s in json.load(open(sys.argv[1]))["law"]["strata"]))' \
      "$VW/window/registration.json") || { say "V3C-AUDIT: R, the window's registration, has no strata to size the audit's ports by: set V3C_PORTS"; return 1; }
  fi
  leanb=$TREE/verity/protocols/flock/verifier/.lake/build/bin
  m0=$TREE:$TREE/benchmarks/numerical/python:$TREE/verity/protocols/flock/prover/python:$TREE/adapters/vllm
  if [ -n "${V3C_WARM_JOBS:-}" ]; then
    v3c_warm
    cache=(--compose-cache "${V3C_COMPOSE_CACHE:-$VW/compose-cache}")
  fi
  if v3c_live; then  # the gate opens the run root once, ever, and a reserve joins the lane's open period: neither, first
    gl=$("$PY" "$SV" gate --gate "$(v3c_gate)" --lane "${H1_LANE:-vllm}" --main "$main" --root "$run" 2>&1); rc=$?
    say "V3C-$(tail -1 <<< "$gl" | cut -c1-1200) (checked $(date -u +%Y-%m-%dT%H:%M:%SZ))"
    [ "$rc" = 0 ] || return 1
    ch+=(--peer "$V3C_PEER" --challenger-key "$PP_CHALLENGER_KEY" ${PP_GATE_KEY:+--gate-key "$PP_GATE_KEY"})
  fi
  V3C_AUDITED=$(hostname -s)
  (cd "$TREE" && env -u PYTHONPATH RAYON_NUM_THREADS="$(nproc)" RUST_MIN_STACK=$((64 << 20)) timeout -k 1m "$V3C_AUDIT_TIMEOUT" \
    "$TREE/.venv/bin/python" benchmarks/prove_path/records_audit.py audit --window "$VW/window" --law "${V3C_LAW:-subset:1}" \
    --circuit "$(sha512sum "$VW/window/circuit.json" | cut -c1-128)" --run "$run" --registered "$WREG" "${tie[@]}" "${day[@]}" "${cache[@]}" \
    "${ch[@]}" --main "$main" --m0-pythonpath "$m0" --bin "$FW/flock-gpu-link/flock/target/release/flock-circuit" --lean "$leanb/flock-verify" \
    --firewall "$leanb/flock-firewall" --ports "$ports" --out "$VW/audit") > "$OUT/v3c-audit/audit.log" 2> "$OUT/v3c-audit/audit.err"
  rc=$?
  cp "$VW/audit/summary.json" "$OUT/v3c-audit/" 2>/dev/null
  [ -z "$PEARLPLUS_TIE" ] || say "V3C-TIE: $(grep -m1 '^TIE ' "$OUT/v3c-audit/audit.log" | "$PY" -c 'import json, sys
line = sys.stdin.read()
if not line.startswith("TIE "):
    sys.exit("no TIE line: the audit stopped before it read its records tied to W (v3c-audit/audit.log)")
t = json.loads(line[4:])
w, ok = t["record"][:16], "PASS" if t["chained"] else "FAIL"
print(f"the audit read its records tied to W {w}…, R chained right after it: {ok}")' 2>&1 \
    | tail -1 | cut -c1-400)"
  say "V3C-AUDIT class $V3C (records_audit.py exit $rc): $(grep -m1 '^RECORDS' "$OUT/v3c-audit/audit.log" | cut -c1-3000)"
  [ -n "$(grep -m1 '^RECORDS' "$OUT/v3c-audit/audit.log")" ] \
    || say "V3C-AUDIT: no RECORDS line; $(tail -3 "$OUT/v3c-audit/audit.err" | tr '\n' ' ' | cut -c1-800)"
  return $rc
}
v3c_period() {  # V3c's audit as one compute audit of the network certifier's period: $VW/period.json
  (cd "$TREE" && env -u PYTHONPATH "$TREE/.venv/bin/python" benchmarks/prove_path/records_audit.py period --window "$VW/window" \
    --audit "$VW/audit" --pins "$WREG" --out "$VW/period.json") > "$OUT/v3c-audit/period.log" 2>&1
  local rc=$?
  [ "$rc" = 0 ] && cp "$VW/period.json" "$OUT/v3c-audit/"
  say "V3C-PERIOD (records_audit.py period exit $rc): $(tail -1 "$OUT/v3c-audit/period.log" | cut -c1-800)"
  return $rc
}
certifier_step() {  # $1 chain|period: the split network certifier's step on V3c's window (step 6), if its command is set
  local cmd lim dir=$OUT/pieces/certifier rc
  if [ "$1" = chain ]; then cmd=${CERTIFIER_CHAIN_CMD:-} lim=${CERTIFIER_TIMEOUT:-4h}
  else cmd=${CERTIFIER_PERIOD_CMD:-} lim=${CERTIFIER_PERIOD_TIMEOUT:-60m}; fi
  [ -n "$cmd" ] || return 0
  mkdir -p "$dir"
  (cd "$TREE" && timeout -k 1m "$lim" bash -c "$cmd" _ "$D" "$K" "$dir" "$VW") > "$dir/$1.log" 2>&1
  rc=$?
  say "CERTIFIER-${1^^} (exit $rc): $(tail -1 "$dir/$1.log" | cut -c1-600)"
  return $rc
}
v3c_proved() {  # V3c's window committed, the build done: the certifier's chain beside V3c's audit, then its period check
  local rc prc chain
  certifier_step chain & chain=$!
  v3c_audit; rc=$?
  v3c_period; prc=$?
  wait "$chain"
  if [ "$prc" = 0 ]; then certifier_step period
  elif [ -n "${CERTIFIER_PERIOD_CMD:-}" ]; then say "CERTIFIER-PERIOD: not run: V3c wrote no period.json"; fi
  return $((rc ? rc : prc))
}
compiler_args() {  # the prove path's registered compiler, as verity-vllm advice takes it
  (cd "$TREE" && "$PY" benchmarks/prove_path/prove_path.py compiler --serve-run "$D" --cap-bits "$CAP") \
    | "$PY" -c "import json, sys; c = json.load(sys.stdin)['inner']; print('--compiler-name', c['name'], '--compiler-digest', c['digest'])"
}
advice_of() {  # window $1's advice under the compiler $2 in $D/advice/w$1, unless it is there already (the tie wrote it):
  # verity-vllm advice's exit, 3 while the window waits on a Commit record the running server hasn't written
  [ -f "$D/advice/w$1/registration.json" ] && return 0
  local wa=() rc
  [ -f "$WREG" ] && wa=(--weights-of-record "$WREG")
  # shellcheck disable=SC2086
  (cd "$TREE/adapters/vllm" && vv advice "$D" --window "$1" --cap "$CAP" $2 "${wa[@]}" --out "$D/advice/.w$1") \
    >> "$D/advice.log" 2>&1
  rc=$?
  [ -d "$D/advice/.w$1/w$1" ] && mv "$D/advice/.w$1/w$1" "$D/advice/w$1"
  return "$rc"
}
tie() {  # serve-openai's --h1-window-commit (`serve_tie`) at window $2's close, before its sample coin: $1 the serve dir, $3
  # the window's registration W, $4 where its record R goes, $5 its H1 prover dir. It says nothing in $OUT, whose lines
  # name no window: what it says goes to the serve dir's h1/w<k>/window-commit.log, and step 6 says the picked window's
  K=$2 VW=$D/v3c-w$2-$vtag
  local ca rc why
  [ "$1" -ef "$D" ] || { echo "TIE: serve-openai's serve dir $1 isn't D, $D"; return 3; }
  [ -n "$V3C" ] && [ -n "$5" ] || { echo "TIE: V3C_CLASS is empty or the window has no H1 prover dir"; return 3; }
  [ -f "$WREG" ] || { echo "TIE: $NO_WEIGHTS"; return 13; }
  ca=$(compiler_args) && [ -n "$ca" ] || { echo "TIE: no registered compiler (prove_path.py compiler)"; return 13; }
  advice_of "$K" "$ca"; rc=$?
  if [ "$rc" = 3 ]; then
    echo "TIE: the window's advice is pending: a request of it runs on past its close, with no Commit record yet"; return 14
  fi
  [ -f "$D/advice/w$K/private/advice.bin" ] || { echo "TIE: no advice (verity-vllm advice exit $rc; the serve dir's advice.log)"; return 14; }
  why=$(window_commit "$D/h1/w$K/window-commit.json" "$D/h1/w$K/window-commit.err" --window-registration "$3" \
    --h1-key "$5/key" --law "${V3C_LAW:-subset:1}") || { echo "TIE: window-commit class $V3C $why"; return 16; }
  cp "$VW/window/registration.json" "$4" || { echo "TIE: window-commit class $V3C wrote no registration"; return 16; }
  echo "TIE: window-commit class $V3C: its registration is the window's record R"
}
pearl_tie() {  # served_pearlplus.sh's PEARLPLUS_TIE_CMD once Pearl+'s W is written and before Pearl+'s coin: $1 the serve
  # dir, $2 the window, $3 W (retain --window-record's), $4 the window's passes (serve_pearl's pearl/w<k>/). W and the
  # commit go to the serve dir's pearlplus/w<k>/; like `tie`, it says nothing in $OUT
  K=$2 VW=$D/v3c-w$2-$vtag
  local ca rc why dir
  dir=$(pearl_dir)
  [ "$1" -ef "$D" ] || { echo "PEARL-TIE: the piece's serve dir $1 isn't D, $D"; return 3; }
  [ -n "$PEARLPLUS_TIE" ] && [ -n "$V3C" ] || { echo "PEARL-TIE: PEARLPLUS_TIE or V3C_CLASS is empty"; return 3; }
  [ -f "$WREG" ] || { echo "PEARL-TIE: $NO_WEIGHTS"; return 13; }
  ca=$(compiler_args) && [ -n "$ca" ] || { echo "PEARL-TIE: no registered compiler (prove_path.py compiler)"; return 13; }
  advice_of "$K" "$ca"; rc=$?
  [ -f "$D/advice/w$K/private/advice.bin" ] || { echo "PEARL-TIE: no advice (verity-vllm advice exit $rc; the serve dir's advice.log)"; return 14; }
  mkdir -p "$dir" && cp "$3" "$dir/registration.json" || { echo "PEARL-TIE: W $3 can't be kept in $dir"; return 3; }
  why=$(window_commit "$dir/window-commit.json" "$dir/window-commit.err" --window-registration "$dir/registration.json" \
    --pearl-passes "$4" --law "${V3C_LAW:-subset:1}") || { echo "PEARL-TIE: window-commit class $V3C $why"; return 16; }
  why=$(pearl_tied) || { echo "PEARL-TIE: $why"; return 16; }
  echo "PEARL-TIE: window-commit class $V3C, its circuit cut at Pearl+'s linears: $why"
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
    say "PEARLPLUS-CIRCUIT refused (exit $rc): $({ tail -1 "$dir/circuit-record.err" 2>/dev/null; tail -1 "$dir/serve-phi.log"; } | grep -m1 . | cut -c1-600)"
  fi
  return "$rc"
}

serve() {  # under the lease: the endpoint, the requests, every closed window sealed, the endpoint stopped
  export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
  mkdir -p "$D" "$OUT/serve"
  cd "$TREE/adapters/vllm" || return 9
  local h1=() pc=() day=()
  [ -n "${H1_PROTOCOLS_CONFIG:-}" ] && pc=(--protocols-config "$H1_PROTOCOLS_CONFIG")
  [ -n "${DAY:-}" ] && day=(--day "$DAY" --day-log "$DAY_LOG" --day-program "$DAY_PROGRAM"
                            ${DAY_BLOCKS:+--day-blocks "$DAY_BLOCKS"} ${DAY_PARAMS:+--day-params "$DAY_PARAMS"})
  [ -n "$H1_GATE" ] && h1=(--protocols pouw:eps1-sm120-h1fx1 --challenger-gate "$H1_GATE"
                           --challenger-peer "$H1_PEER" "${pc[@]}")
  [ -n "$H1_GATE" ] && [ -n "${H1_CIRCUITS:-}" ] && h1+=(--h1-prover "$D/private/h1")
  [ -n "$H1_GATE" ] && [ "${H1_RECEIPTS:-0}" = 1 ] && h1+=(--h1-receipts)
  if [ -n "$PEARLPLUS_TIE" ]; then  # the window's linears Pearl+'s matmuls on this one engine (serve_pearl), one epoch a window
    h1=(--protocols "$PEARLPLUS_ARM" --protocols-config "$("$SAY_PY" -c 'import json, sys; print(json.dumps({"pouw": {"device": "cuda", "kernel": sys.argv[1], "cubin": sys.argv[2]}}))' \
      "$PEARLC_KERNEL" "$PEARLC_CUBIN")")
    [ -n "$PEARLC_CHALLENGER_GATE" ] && h1+=(--challenger-gate "$PEARLC_CHALLENGER_GATE"
                                            --challenger-peer "$PEARLC_CHALLENGER_PEER" --challenger-lane "$PEARLC_CHALLENGER_LANE")
  fi
  local polls=450
  if [ -n "${H1_TIE:-}" ]; then  # a window's line waits for its release, which waits for its window commit
    h1+=(--h1-window-commit "cd $(printf %q "$TREE") && bash benchmarks/one_stage/served_verdicts.sh tie \"\$@\"")
    polls=$((polls + 30 * (${V3C_TIMEOUT%m} + 5)))
  fi
  if [ -n "${WIPE_CMD:-}" ]; then  # the wipe of this card, before its engine loads
    mkdir -p "$OUT/pieces/wipe"
    (cd "$TREE" && timeout -k 1m "${WIPE_TIMEOUT:-20m}" bash -c "$WIPE_CMD" _ "$D" all "$OUT/pieces/wipe") \
      > "$OUT/pieces/wipe/piece.log" 2>&1
    say "WIPE (exit $?, within WIPE_TIMEOUT=${WIPE_TIMEOUT:-20m}): $(tail -1 "$OUT/pieces/wipe/piece.log" | cut -c1-600)"
  fi
  local period=0
  if [ -n "${H1_GATE:-}" ] && [ -n "${H1_CIRCUITS:-}" ]; then  # before serve-openai, whose startup reserves window 0's
    # session (`serve_h1.arm_of`): a reserve joins only a period already open, so it's opened on the offline Program
    say "$("$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" program "$D/h1" --model "$REPO" --rows "$T" \
      --steps "$WINDOW_STEPS" "${pc[@]}" 2>&1 | tail -1)"
    (cd "$TREE/adapters/vllm" && PYTHONPATH=$PYTHONPATH:$TREE/verity/protocols/flock/prover/python "$PY" -m tests.protocols.pouw_h1_circuits \
      --program "$D/h1" --out "$H1_CIRCUITS" --live "$T,$WINDOW_STEPS" --strata "${H1_STRATA:-none}" \
      ${H1_LANES:+--lanes "$H1_LANES"}) \
      > "$OUT/h1-circuits.log" 2>&1 || say "H1-CIRCUITS refused (exit $?): $(tail -1 "$OUT/h1-circuits.log" | cut -c1-400)"
    # shellcheck disable=SC2086
    say "$("$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" period open "$D" --gate "$H1_GATE" \
      --name "${H1_PERIOD:-served-${RESEARCH_RUN_ID:-local}}" --circuits "$H1_CIRCUITS" --slots "$SLOTS" \
      --max-tokens "$MAX_TOKENS" ${H1_DRAW_K:+--draw-k "$H1_DRAW_K"} ${H1_MODEL:+--model "$H1_MODEL"} 2>&1 | tail -1)"
    [ -f "$D/period.json" ] && period=1
  fi
  # the policy is written before serve-openai's start, which the verdict reads as the serve's beginning
  [ -z "$PEARLPLUS_TIE" ] || { pearl_circuit; (cd "$TREE" && inference_policy); pearl_period open; }
  # shellcheck disable=SC2086
  "$PY" -m verity_vllm.pipeline.cli serve-openai --model "$REPO" --revision "$REV" --out "$D" --port "$PORT" \
    --precision "$PRECISION" --run-id "${RESEARCH_RUN_ID:-local}" --scheme rows-sha512 --seal \
    --max-num-batched-tokens "$T" --window-steps "$WINDOW_STEPS" "${h1[@]}" "${day[@]}" \
    --required-manifest "$PRED/manifest.json" --hidden-so "$HIDDEN_SO" --norm-tap-so "$NORM_TAP_SO" --tap-cap-mb 2048 \
    --admit release-order --window-requests "$SLOTS" --max-model-len "$MAX_MODEL_LEN" --window-gb 32 \
    --disk-floor-gb "$DISK_FLOOR_GB" --seal-hold-gb 32 --seal-warm-gb 0 ${SERVE_ARGS:-} \
    > "$D/serve.log" 2>&1 &
  local srv=$! up=0 c=0 s=0 rc=0 release
  trap 'kill -TERM "$srv" 2>/dev/null; wait "$srv"; pearl_period close; exit 143' TERM INT
  for _ in $(seq 1 900); do
    grep -q "SERVE-OPENAI listening" "$D/serve.log" && { up=1; break; }
    kill -0 "$srv" 2>/dev/null || break
    sleep 2
  done
  if [ "$up" != 1 ]; then
    tail -40 "$D/serve.log"
    say "SERVE: not listening: $(grep -m1 -E 'UsageError|Error|refus' "$D/serve.log" | cut -c1-400)"
    kill -TERM "$srv" 2>/dev/null; wait "$srv"
    [ "$period" = 1 ] && say "$("$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" period close "$D" --gate "$H1_GATE" 2>&1 | tail -1)"
    pearl_period close
    return 10
  fi
  say "SERVE $(grep -m1 'SERVE-OPENAI listening' "$D/serve.log" | cut -c1-300)"
  weights
  [ "$period" = 1 ] && say "$("$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" period served "$D" 2>&1 | tail -1)"
  if [ -n "${CERTIFIER_FRONT:-}" ]; then  # the network certifier's warden in front of the endpoint, before the requests
    # below: its group closes into the run's earliest window, which `served_verdicts.py window` picks on a tie in requests.
    # Its session, bundle and log hold the group's prompts, request and cells, so they stay in the serve dir's private/
    (umask 077; mkdir -p "$D/private")
    (cd "$TREE" && timeout -k 1m "${CERTIFIER_FRONT_TIMEOUT:-20m}" bash -c "$CERTIFIER_FRONT" _ "$PORT" "$D" "$D/private/certifier-front") \
      > "$D/private/certifier-front.log" 2>&1
    rc=$?
    say "CERTIFIER-FRONT (exit $rc): $("$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" front-line "$D/private/certifier-front/session.json" 2>&1 | cut -c1-600)"
  fi
  if [ "$LOAD" != front ]; then
    release=$("$PY" -c 'import uuid; print(uuid.uuid4().hex)')
    local load=(--varied "$REQUESTS" --rate "$RATE" --max-tokens "$MAX_TOKENS")
    [ "$LOAD" = fixed ] && load=(--fixed "$SLOTS" --window-steps "$WINDOW_STEPS")
    "$PY" "$TREE/demos/vllm/serve_load.py" --port "$PORT" --model "$REPO" "${load[@]}" --prompt-range "$PROMPTS" \
      --release "$release" --seed "$SEED" --out "$D/load.jsonl" > "$D/load.log" 2>&1
    rc=$?
    local sent all
    sent=$(tail -1 "$D/load.log" | grep -oE '[0-9]+/[0-9]+ ok' | head -1) all=${sent#*/}
    keep "REQUESTS: $sent"
    say "REQUESTS (exit $rc, release run $release): $([ -n "$sent" ] && [ "${sent%%/*}" = "${all% ok}" ] && echo "every request answered" \
      || echo "not every request answered (the serve dir's load.log, on the node)")"
    [ -f "$D/load.timings.json" ] || say "TIMINGS: no $D/load.timings.json"
  fi
  c=$(count "$D/windows.jsonl"); s=$(count "$D/seals.jsonl")
  # a front that failed before a window closed sent the run's only requests, so none will
  if [ "$LOAD" != front ] || [ "$rc" = 0 ] || [ "$c" -gt 0 ]; then
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
  [ "$period" = 1 ] && say "$("$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" period close "$D" --gate "$H1_GATE" 2>&1 | tail -1)"
  pearl_period close
  # a front's exit is the certifier's verdict (CERTIFIER-FRONT, and its piece reads the session), not the load's
  { [ "$LOAD" = front ] || [ "$rc" = 0 ]; } && [ "$c" -gt 0 ] && [ "$s" -ge "$c" ]
}

circuit() {  # window K's circuit at class $1 within $2 (NAME=LIMIT), its line headed $3, with $4 `tie` reading H1's W's
  # tables of the window (window-circuit --window-registration, its tag's suffix -tie): 0 iff it counted
  local c=$1 head=$3 cls=() tag=record pins=() wp=$NO_WEIGHTS f rc tie=()
  [ "$c" = record ] || { cls=(--class "$c"); tag=${c//,/-}; }
  # under PEARLPLUS_TIE the verifier's own record of the circuit the window is committed as, cut at Pearl+'s linears
  [ -z "$PEARLPLUS_TIE" ] || [ "$c" = record ] || { cls+=(--pearl-tables "${PEARLPLUS_ARM#pouw:}"); head+=", cut at Pearl+'s linears"; }
  [ "${4:-}" = tie ] && tie=(--window-registration "$D/h1/w$K/registration.json") tag+=-tie
  [ -f "$WREG" ] && pins=(--pins "$WREG") wp="at the weights registration's pins"
  f=$D/window-circuit-w$K-$tag.json
  timeout "${2#*=}" "$PY" benchmarks/prove_path/prove_path.py window-circuit --serve-run "$D" --window "$K" \
    --row-dir "$D/windows/w$K" --advice "$D/advice/w$K/private/advice.bin" --out "$f" "${pins[@]}" ${PIN:+--pin "$PIN"} \
    "${cls[@]}" "${tie[@]}" > "$D/window-circuit-$tag.count.json" 2> "$D/window-circuit-$tag.err"
  rc=$?
  if [ "$rc" = 0 ]; then  # the count's `bits` is the advice's length, which stays on the node
    "$PY" -c 'import json, sys; c = json.load(open(sys.argv[1])); c.pop("bits", None); print(json.dumps(c, sort_keys=True))' \
      "$D/window-circuit-$tag.count.json" > "$OUT/window-circuit-$tag.json"
    if [ -n "$PEARLPLUS_TIE" ] && [ "$c" = "$V3C" ] && [ -f "$PEARLPLUS_CIRCUIT" ]; then
      head+=" (the record PEARLPLUS-CIRCUIT built before the serve: $(cmp -s "$f" "$PEARLPLUS_CIRCUIT" && echo the same || echo ANOTHER))"
    fi
    say "$head: $("$PY" -c 'import json, sys; c = json.load(open(sys.argv[1]))["count"]; print("registered", c["registered"], "bits, free", c["free"], "bits of", c["inputs"], "input bits")' \
      "$OUT/window-circuit-$tag.json" 2>&1 | cut -c1-200) $wp; $(cat "$OUT/window-circuit-$tag.json"); record $(sha256sum "$f" | cut -c1-16)… $(stat -c %s "$f") bytes"
  elif [ "$rc" = 124 ]; then
    say "$head timed out after $2 (exit 124): no count"
  else
    say "$head refused (exit $rc): $({ grep -m1 'window-circuit' "$D/window-circuit-$tag.err" \
      || tail -1 "$D/window-circuit-$tag.err"; } | cut -c1-600)"
  fi
  return "$rc"
}
circuits() {  # in TREE: every class of CLASSES gates (0 iff each counted); under H1_TIE, V3c's record after them (`v3c_record`,
  # reading H1's tables), which gates nothing
  local bad=0 c
  grep -q '"window-circuit"' benchmarks/prove_path/prove_path.py \
    || { say "MISSING V3b's window circuit: this tree's prove_path.py has no window-circuit"; return 1; }
  for c in $CLASSES; do circuit "$c" "WC_TIMEOUT=${WC_TIMEOUT:-30m}" "WINDOW-CIRCUIT class $c" || bad=1; done
  [ -z "${H1_TIE:-}" ] || [ -z "$V3C" ] || v3c_circuit
  return "$bad"
}
block_counts() {  # in TREE: the picked window's (c, k)-inference blocks at each class of CLASSES it counted, judged against
  # the inference-only policy (IO_POLICY) where the run has one: the judgment public, its exact counts in the serve dir's private/
  local c tag f pins=() day decl=${IO_POLICY:-$OUT/inference-only-policy.json}
  [ -f "$decl" ] || { say "BLOCK-COUNTS: none, since the run declares no c, k and n"; return 0; }
  [ -f "$WREG" ] && pins=(--pins "$WREG")
  day=$("$PY" -c 'import datetime, json, sys; t = json.load(open(sys.argv[1]))["clocks"]["time"]; print(datetime.datetime.fromtimestamp(t, datetime.timezone.utc).date())' "$D/serve.json") || return 0
  (umask 077; mkdir -p "$D/private")
  for c in $CLASSES; do
    tag=${c//,/-}; [ "$c" = record ] && tag=record
    f=$D/window-circuit-w$K-$tag.json
    [ -f "$f" ] || continue
    if "$PY" -m verity_vllm.pipeline.cli block-counts --day "$day" --declaration "$decl" \
         --window "$f" "$D/advice/w$K/registration.json" "${pins[@]}" --private "$D/private/block-counts-$tag.json" \
         --out "$OUT/block-counts-$tag.json" > /dev/null 2> "$D/block-counts-$tag.err"; then
      say "BLOCK-COUNTS class $c, the picked window alone: $(tail -1 "$D/block-counts-$tag.err" | cut -c1-400)"
    else say "BLOCK-COUNTS class $c refused: $(tail -1 "$D/block-counts-$tag.err" | cut -c1-600)"; fi
  done
}
IO_POLICY=${IO_POLICY:-$OUT/inference-only-policy.json}
IO_LEAN=$TREE/verity/protocols/inference_only/verifier/nci/lean/.lake/build/bin/nci-check
io_class() {  # the inference-only verdict's class: IO_CLASS, else the first of CLASSES but `record`
  local c
  [ -z "${IO_CLASS:-}" ] || { echo "$IO_CLASS"; return 0; }
  for c in $CLASSES; do [ "$c" = record ] || { echo "$c"; return 0; }; done
}
inference_policy() {  # in TREE, before the lease: the policy's (c, k, n) at the class, from the run's settings alone; under
  # PEARLPLUS_TIE in serve(), before serve-openai starts, from serve-phi's φ (pearl_circuit's) cut at Pearl+'s linears
  local c out rc phi=()
  c=$(io_class)
  [ -n "$c" ] || { say "INFERENCE-ONLY POLICY: none, since CLASSES names no class but record"; return 0; }
  [ -z "$PEARLPLUS_TIE" ] || phi=(--serve "$D/pearlplus/serve-phi.json" --pearl-tables "${PEARLPLUS_ARM#pouw:}")
  out=$(CUDA_VISIBLE_DEVICES='' "$PY" benchmarks/one_stage/inference_only.py policy --class "$c" --model "$REPO" \
    --revision "$REV" --lease-min "${LEASE_MIN:-60}" --tree "$(tree_sha)" "${phi[@]}" --out "$IO_POLICY" \
    2> "$D/inference-only-policy.err")
  rc=$?
  say "${out:-INFERENCE-ONLY POLICY failed (exit $rc): $(tail -1 "$D/inference-only-policy.err" | cut -c1-300)}"
  return "$rc"
}
inference_only() {  # in TREE, once step 3 counted window K's circuit at the class: its verdict against the policy
  local c out rc lean=()
  c=$(io_class)
  [ -n "$c" ] || return 0
  [ -x "$IO_LEAN" ] && lean=(--lean "$IO_LEAN" --lean-timeout "${IO_LEAN_TIMEOUT:-1800}")
  out=$(CUDA_VISIBLE_DEVICES='' "$PY" benchmarks/one_stage/inference_only.py verdict --policy "$IO_POLICY" --serve-run "$D" \
    --circuit "$D/window-circuit-w$K-${c//,/-}.json" --registration "$D/advice/w$K/registration.json" --pins "$WREG" \
    "${lean[@]}" --out "$OUT/record/inference-only.json" --private "$D/private/inference-only.json" 2> "$D/inference-only.err")
  rc=$?
  say "${out:-INFERENCE-ONLY failed (exit $rc): its log in the serve dir, on the node}"
  return "$rc"
}
front_ok() {  # preflight: LOAD=front has a front, and each flag CERTIFIER_FRONT passes to the tree's front, it takes
  [ "$LOAD" != front ] || [ -n "${CERTIFIER_FRONT:-}" ] \
    || { say "PREFLIGHT: LOAD=front sends the front's group alone, and CERTIFIER_FRONT is unset"; return 3; }
  case ${CERTIFIER_FRONT:-} in *"certifier_piece.py front"*) ;; *) return 0 ;; esac
  case " ${SERVE_ARGS:-} " in *" --protocol ids "*) ;; *)
    say "PREFLIGHT: the certifier front's warden sends id-form requests, which serve-openai answers only under" \
        "SERVE_ARGS='--protocol ids --id-payload 503'"; return 3 ;; esac
  local help f script
  script=$(grep -oE "[^[:space:]\"']*certifier_piece\.py front" <<< "$CERTIFIER_FRONT" | head -1)
  help=$(cd "$TREE" && "$PY" "${script% front}" front --help 2>&1) \
    || { say "PREFLIGHT: this tree's certifier front doesn't start: $(echo "$help" | tail -1 | cut -c1-300)"; return 3; }
  for f in $(grep -oE -- '--[a-z][a-z0-9-]*' <<< "$CERTIFIER_FRONT" | sort -u); do
    grep -qE "(^|[^a-z0-9-])$f([^a-z0-9-]|$)" <<< "$help" \
      || { say "PREFLIGHT: CERTIFIER_FRONT passes $f, which this tree's certifier front doesn't take"; return 3; }
  done
}
# the certifier's chain files list and count the window's requests (its certs, its tie's prompt lengths, its N2 warden's
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
# window-circuit record at the class reading H1's tables (step 3's, `v3c_record`), windows.jsonl (the run root), H1's log
# and each record it names, the tie's window-commit answer, and the commit's window dir (circuit.json, registration.json,
# every value file). The record's: seals.jsonl, windows.jsonl and requests.jsonl (the pick's steps), step 3's records
# audit's summary, audit record, registration and runs (which it reads where V3c's audit writes no summary:
# `sampled_of_record`), the pieces it read, and the inference-only verdict it carries. The
# composite's: PoUW's period case, and Pearl+'s case's parts and blobs (served_pearlplus.sh's step 4c). Under DAY, the
# serve's declaration.json and the deferral's copy of the day's log and parameters. And deferred.env, the settings the
# follow-on runs at. Each is listed where it exists.
deferred_inputs() {
  {
    printf '%s\n' weights_registration.json "$(v3c_record)" windows.jsonl seals.jsonl requests.jsonl declaration.json \
      h1/log.jsonl "h1/w$K/window-commit.json" sampled-proofs-served/summary.json sampled-proofs-served/window/audit.json \
      sampled-proofs-served/window/registration.json sampled-proofs-served/runs.json private/pouw/period/case.json \
      private/pouw/period/answer.json private/deferred/inference-only.json "pearlplus/w$K/registration.json" \
      "pearlplus/w$K/window-commit.json" served-compare/served_compare.json
    [ ! -f "$D/h1/log.jsonl" ] || "$SAY_PY" -c 'import json, sys
for x in open(sys.argv[1]):
    if x.strip():
        print("h1/" + str(json.loads(x).get("file")))' "$D/h1/log.jsonl"
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
  printf '%s=%q\n' K "$K" PRODUCT "$PRODUCT" V3C_CLASS "$V3C" V3C_LAW "${V3C_LAW:-subset:1}" V3C_CHALLENGER "$V3C_CHALLENGER" \
    DEVELOPMENT "$DEVELOPMENT" H1_TIE "${H1_TIE:-}" PEARLPLUS_TIE "$PEARLPLUS_TIE" COMPOSITE "${COMPOSITE-1}" ALPHA "${ALPHA:-}" GAMMA "${GAMMA:-}" \
    SERVED_NODE "$(hostname)" SERVED_PEER "$V3C_PEER" DEFERRED_TREE "$(tree_sha)" COMPOSE_OFF_NODE "$COMPOSE_OFF_NODE" \
    COMPOSE_NODE "$COMPOSE_NODE" DAY "${DAY:-}" > "$DEFER_DIR/deferred.env" || return 1
  deferred_inputs > "$DEFER_DIR/inputs.txt"
  (cd "$D" && tr '\n' '\0' < "$DEFER_DIR/inputs.txt" | xargs -0 -r sha256sum) > "$DEFER_DIR/inputs.sha256" || return 1
  echo private/deferred/inputs.sha256 >> "$DEFER_DIR/inputs.txt"
  [ "$PRODUCT" != onsite ] \
    || say "ONSITE-CASES: before the deferral, each of onsite's own subprotocols with the verdict of the answer its piece" \
        "wrote beside its case on this node (none: no piece wrote one):" \
        "$("$SAY_PY" "$SV" onsite-cases --serve-run "$D" --pieces "$OUT/pieces" 2>&1 | tail -1 | cut -c1-600)"
  if [ "$COMPOSE_OFF_NODE" = 1 ]; then
    say "DEFERRED $DEFERRED_WHAT: $1. A follow-on on a node with the memory runs it there and writes the compose hand-off," \
        "from the serve dir's audit inputs copied prover-side, outside any run dir, each file its private/deferred/inputs.txt" \
        "lists: D=$D bash benchmarks/one_stage/served_verdicts.sh v3c; $HANDED_WHAT are composed off both nodes, on" \
        "COMPOSE_NODE ($COMPOSE_NODE), from that hand-off"
  else
    say "DEFERRED $DEFERRED_WHAT: $1. A follow-on on a node with the memory runs them and composes the final record there," \
        "from the serve dir's audit inputs copied prover-side, outside any run dir, each file its private/deferred/inputs.txt" \
        "lists: D=$D bash benchmarks/one_stage/served_verdicts.sh v3c"
  fi
}
day_held() {  # a follow-on under DAY: the deferral's copy of the day's log, which core's check accepts, holds the
  # declaration the serve's declaration.json names by digest, of day DAY, so V3c's audit runs R2 against the day the serve declared
  local got
  got=$("$TREE/.venv/bin/python" -c 'import json, sys
from verity.hardware import site as S
from verity.hardware import registration as G
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
  # step 6's audit, and the final record and the composite here rather than on the served node, or under
  # COMPOSE_OFF_NODE=1 the compose hand-off for a node of neither
  local rc vrc rec comp=0 TPY=$TREE/.venv/bin/python
  (cd "$D" && sha256sum --check --strict --quiet private/deferred/inputs.sha256) > "$DEFER_DIR/check.log" 2>&1
  rc=$?
  say "FOLLOW-ON on $(hostname -s), $NODE_MEM_GB GB: $DEFERRED_WHAT, which the served node $SERVED_NODE deferred, from its serve" \
      "dir's audit inputs copied here prover-side, each as it hashed them (the serve dir's private/deferred/inputs.sha256):" \
      "$([ "$rc" = 0 ] && echo PASS || echo "FAIL (the serve dir's private/deferred/check.log)")"
  [ "$rc" = 0 ] || return 3
  wait "$BUILD"; rc=$?
  [ "$rc" = 0 ] || { tail -30 "$OUT/build.log"; stop build "$rc"; }
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
    [ "$vrc" = 0 ] && [ "$rc" = 0 ]
    return
  fi
  if [ -f "$DEFER_DIR/inference-only.json" ]; then
    mkdir -p "$OUT/record" && cp "$DEFER_DIR/inference-only.json" "$OUT/record/inference-only.json" \
      && say "INFERENCE-ONLY: the served node's verdict, carried into this record as it wrote it (the serve dir's private/deferred/inference-only.json)"
  fi
  "$TPY" benchmarks/one_stage/served_verdicts.py record --serve-run "$D" --window "$K" --sampled-proofs "$D/$(sampled_of_record)" \
    --pieces "$DEFER_DIR/pieces" --out "$OUT/record" --product "$PRODUCT" --node "$SERVED_NODE" ${wt[@]+"${wt[@]}"} \
    --key-sha512 "$H1_KEY_SHA512" --lane "${PEARLC_CHALLENGER_LANE:-${H1_LANE:-vllm}}" 2>&1 | publish_lines
  rec=${PIPESTATUS[0]}
  [ -z "${COMPOSITE-1}" ] || TPY=$TPY SERVED_NODE=$SERVED_NODE bash benchmarks/one_stage/served_composite.sh "$D" "$OUT" || comp=1
  echo "---- verdicts"; cat "$OUT/verdicts.txt"
  [ "$vrc" = 0 ] && [ "$rec" = 0 ] && [ "$comp" = 0 ]
}
# The record's sampled-proofs audit, relative to the serve dir: V3c's of window K where it wrote its summary, the
# sampled-proofs audit of record, on V3C_CHALLENGER's coins (step 3's are local while those are live) and, under a tie,
# tied to H1's or Pearl+'s W; else step 3's records audit
sampled_of_record() {
  if [ -f "$D/v3c-w$K-$vtag/audit/summary.json" ]; then echo "v3c-w$K-$vtag/audit"; else echo sampled-proofs-served; fi
}
# The compose hand-off (COMPOSE_OFF_NODE=1, the default): what the final record and the composite read, relative to the
# serve dir, and nothing else. The record's: its sampled-proofs audit's (`sampled_of_record`) summary, audit record,
# registration and runs, and of each piece the files `piece_row` and `piece_server` read (copied to
# private/compose/pieces), Pearl+'s verdict of record's among them (`pearlplus_of_record`'s: the auditor's ledger, the
# coin record, Pearl+'s answer, the Lean verdict and each challenger session's public key, out/key.json at its path, and
# no other file of a session). The composite's: each records audit's cases (sampled-proofs-served's and V3c's
# window/case.json, case-v2.json and their error files), the files a case names as its blobs (the circuit, the public
# file, each session's record and proofs, the lookup tables), H1's log and each record it names, and PoUW's period case.
# And compose.env, its settings: the window, its engine steps (the record's, picked here from the requests file, which
# stays), the product, the composite, α and Γ, the served node, the follow-on node COMPOSE_NODE, the tree's commit, the
# host V3c's audit ran on (AUDITED_ON, empty where the hand-off carries none) and the record's audit (SAMPLED_AUDIT).
# Every one is a verifier's input: `served_verdicts.py witness` refuses a list holding a value file, a prover's instance or
# private file, the advice, a key, a row dir or the requests, and a blob outside the serve dir is refused too.
COMPOSE_PIECE_FILES="answer.json development.json piece.json refused.txt served.json runs.json ledger.jsonl coin.json
pearlplus-answer.json pearlplus-lean.json"
handoff_inputs() {  # the hand-off's every input but its own sums, each existing file once, the record's audit $1's among them
  local a blobs cases=()
  for a in sampled-proofs-served "v3c-w$K-$vtag/audit"; do
    [ -f "$D/$a/window/case-v2.json" ] && cases+=("$D/$a/window/case-v2.json")
  done
  [ -f "$D/private/pouw/period/case.json" ] && cases+=("$D/private/pouw/period/case.json")
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
    printf '%s\n' "$1/summary.json" "$1/window/audit.json" "$1/window/registration.json" "$1/runs.json" h1/log.jsonl \
      private/pouw/period/case.json private/pouw/period/answer.json "pearlplus/w$K/tie/window/registration.json" \
      "pearlplus/w$K/tie/summary.json" "pearlplus/w$K/tie/circuit-record.json" served-compare/served_compare.json
    for a in sampled-proofs-served "v3c-w$K-$vtag/audit"; do
      printf "$a/window/%s\n" case.json case-v2.json case-error.txt case-v2-error.txt
    done
    [ ! -f "$D/h1/log.jsonl" ] || "$SAY_PY" -c 'import json, sys
for x in open(sys.argv[1]):
    if x.strip():
        print("h1/" + str(json.loads(x).get("file")))' "$D/h1/log.jsonl"
    [ -z "$blobs" ] || printf '%s\n' "$blobs"
    for f in pieces compose.env inference-only.json; do
      [ ! -e "$D/private/compose/$f" ] || (cd "$D" && find "private/compose/$f" -type f)
    done
  } | (cd "$D" && while IFS= read -r f; do [ ! -f "$f" ] || echo "$f"; done) | LC_ALL=C sort -u
}
handoff() {  # the compose hand-off in private/compose, from the pieces in $1 and the inference-only verdict $2
  local d f steps why rc sampled
  sampled=$(sampled_of_record)
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
  printf '%s=%q\n' K "$K" PRODUCT "$PRODUCT" PEARLPLUS_TIE "$PEARLPLUS_TIE" COMPOSITE "${COMPOSITE-1}" ALPHA "${ALPHA:-}" GAMMA "${GAMMA:-}" \
    SERVED_NODE "${SERVED_NODE:-$(hostname)}" SERVED_DIR "$D" COMPOSE_NODE "$COMPOSE_NODE" COMPOSE_TREE "$(tree_sha)" \
    AUDITED_ON "${V3C_AUDITED:-}" SAMPLED_AUDIT "$sampled" STEPS "$steps" > "$COMPOSE_DIR/compose.env" || return 1
  handoff_inputs "$sampled" > "$COMPOSE_DIR/inputs.txt" 2> "$COMPOSE_DIR/inputs.err"
  rc=$?
  [ "$rc" = 0 ] || { say "HANDOFF refused: $(tail -1 "$COMPOSE_DIR/inputs.err" | cut -c1-600)"; return 1; }
  why=$("$SAY_PY" "$SV" witness < "$COMPOSE_DIR/inputs.txt") \
    || { say "HANDOFF refused: it would carry the served run's witness ($(echo "$why" | head -1 | cut -c1-400))"; return 1; }
  (cd "$D" && tr '\n' '\0' < "$COMPOSE_DIR/inputs.txt" | xargs -0 -r sha256sum) > "$COMPOSE_DIR/inputs.sha256" || return 1
  echo private/compose/inputs.sha256 >> "$COMPOSE_DIR/inputs.txt"
  say "HANDOFF $HANDED_WHAT, composed off this node: on COMPOSE_NODE ($COMPOSE_NODE), from the serve dir's verifier-side" \
      "files and no witness, each file its private/compose/inputs.txt lists, copied to the same path there:" \
      "D=$D bash benchmarks/one_stage/served_verdicts.sh compose"
}
compose() {  # `served_verdicts.sh compose` on a copy of a compose hand-off: the copy held to the served node's sums, then
  # the final record and the composite, here, off the served node
  local rc rec comp=0 TPY=$TREE/.venv/bin/python
  (cd "$D" && sha256sum --check --strict --quiet private/compose/inputs.sha256) > "$COMPOSE_DIR/check.log" 2>&1
  rc=$?
  say "COMPOSE on $(hostname -s) (COMPOSE_NODE $COMPOSE_NODE): $HANDED_WHAT of the served node $SERVED_NODE, from its compose" \
      "hand-off (verifier-side files, no witness; $([ -n "${AUDITED_ON:-}" ] && echo "V3c's audit from $AUDITED_ON" || echo "no V3c audit");" \
      "the record's sampled proofs ${SAMPLED_AUDIT:-sampled-proofs-served}), each file as it" \
      "hashed it (the serve dir's private/compose/inputs.sha256):" \
      "$([ "$rc" = 0 ] && echo PASS || echo "FAIL (the serve dir's private/compose/check.log)")"
  [ "$rc" = 0 ] || return 3
  if [ -n "${SERVED_DIR:-}" ] && [ "$SERVED_DIR" != "$D" ]; then  # a copy at another path: its cases' blobs, here
    "$SAY_PY" "$SV" rebase "$D" "$SERVED_DIR" < <(cut -d' ' -f3- "$COMPOSE_DIR/inputs.sha256") > "$COMPOSE_DIR/rebase.log" 2>&1 \
      || { say "COMPOSE refused: the cases' blobs don't rebase from the served node's serve dir to this copy ($(tail -1 "$COMPOSE_DIR/rebase.log" | cut -c1-400))"; return 3; }
  fi
  wait "$BUILD"; rc=$?
  [ "$rc" = 0 ] || { tail -30 "$OUT/build.log"; stop build "$rc"; }
  if [ -f "$COMPOSE_DIR/inference-only.json" ]; then
    mkdir -p "$OUT/record" && cp "$COMPOSE_DIR/inference-only.json" "$OUT/record/inference-only.json" \
      && say "INFERENCE-ONLY: the served node's verdict, carried into this record as it wrote it (the serve dir's private/compose/inference-only.json)"
  fi
  "$TPY" benchmarks/one_stage/served_verdicts.py record --serve-run "$D" --window "$K" \
    --sampled-proofs "$D/${SAMPLED_AUDIT:-sampled-proofs-served}" \
    --pieces "$COMPOSE_DIR/pieces" --out "$OUT/record" --product "$PRODUCT" --node "$SERVED_NODE" --steps "$STEPS" \
    ${PEARLPLUS_TIE:+--window-tie "$(pearl_dir)/tie" --served-compare "$SERVED_COMPARE"} \
    --key-sha512 "$H1_KEY_SHA512" --lane "${PEARLC_CHALLENGER_LANE:-${H1_LANE:-vllm}}" --off-node --sampled-lean \
    ${SAMPLED_EXE:+--sampled-exe "$SAMPLED_EXE"} --tree "$(tree_sha)" ${COMPOSE_TREE_ON:+--tree-on "$COMPOSE_TREE_ON"} \
    2>&1 | publish_lines
  rec=${PIPESTATUS[0]}
  [ -z "${COMPOSITE-1}" ] || TPY=$TPY SERVED_NODE=$SERVED_NODE COMPOSE_HANDOFF=1 bash benchmarks/one_stage/served_composite.sh "$D" "$OUT" \
    || comp=1
  echo "---- verdicts"; cat "$OUT/verdicts.txt"
  [ "$rec" = 0 ] && [ "$comp" = 0 ]
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
    || { say "COPY refused: the serve dir $D holds no $kind list (private/$kind/inputs.txt and $kind.env)"; return 3; }
  # shellcheck source=/dev/null
  IFS='|' read -r served composer < <(set +u; . "$dir/$kind.env"; echo "${SERVED_NODE%%.*}|${COMPOSE_NODE%%.*}")
  host=${host#*@}
  if [ "$kind" = deferred ]; then
    for n in "$composer" "${COMPOSE_NODE%%.*}"; do
      [ -z "$n" ] || [ "${host%%.*}" != "$n" ] \
        || { say "COPY refused: the deferral's list holds the window's witness, and $host is COMPOSE_NODE, an auditor host where no witness goes: copy it to a prover node"; return 3; }
    done
  else
    [ "${host%%.*}" != "$served" ] \
      || { say "COPY refused: $host is the served node, which the prover controls: copy the hand-off to COMPOSE_NODE ($composer)"; return 3; }
    why=$("$SAY_PY" "$SV" witness < "$dir/inputs.txt") \
      || { say "COPY refused: the hand-off's list names the served run's witness ($(echo "$why" | head -1 | cut -c1-400))"; return 3; }
  fi
  n=$(count "$dir/inputs.txt")
  if [[ $2 == *:* ]]; then
    name=${D%/} && name=${name##*/} && sub=${2#*:} && sub=${sub%/} && sub=${sub:+$sub/}$name
    where="under $sub/ in its rsync's directory, which its node moves to the serve dir's path"
    # shellcheck disable=SC2086
    ${RSYNC:-rsync} -a --files-from="$dir/inputs.txt" "$D/" "${2%%:*}:$sub/" > "$log" 2>&1
  else
    where="each to the same path there"
    # shellcheck disable=SC2086
    ${RSYNC:-rsync} -a --files-from="$dir/inputs.txt" --rsync-path="mkdir -p $(printf %q "$D") && rsync" "$D/" "$2:$D/" > "$log" 2>&1
  fi
  rc=$?
  say "COPY the $([ "$kind" = deferred ] && echo "deferral's prover-side" || echo "hand-off's verifier-side") list's $n files to $host," \
      "$where: $([ "$rc" = 0 ] && echo PASS || echo "FAIL (exit $rc, the serve dir's private/copy-$kind.log)")"
  return "$rc"
}
gate_compose() {  # `served_verdicts.sh gate-compose PIECES [VERDICT [HOST]]`: the hand-off from PIECES (and the
  # inference-only VERDICT), sent to the compose gate as a tar of the files its inputs.txt lists and nothing else; its
  # answer in $OUT/compose
  local rc why answer=$OUT/compose/answer.json
  handoff "$1" "${2:-}" || return 1
  why=$("$SAY_PY" "$SV" witness < "$COMPOSE_DIR/inputs.txt") \
    || { say "GATE-COMPOSE refused: the hand-off's list names the served run's witness ($(echo "$why" | head -1 | cut -c1-400))"; return 3; }
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

if [ "${1:-}" = serve ]; then serve; exit $?; fi
if [ "${1:-}" = tie ]; then shift; tie "$@"; exit $?; fi
if [ "${1:-}" = pearl-tie ]; then shift; pearl_tie "$@"; exit $?; fi
if [ "${1:-}" = circuits ]; then K=${2:?served_verdicts.sh circuits WINDOW}; circuits; exit $?; fi
if [ "${1:-}" = inference-policy ]; then mkdir -p "$D" && inference_policy; exit $?; fi
if [ "${1:-}" = inference-only ]; then K=${2:?served_verdicts.sh inference-only WINDOW}; inference_only; exit $?; fi
if [ "${1:-}" = block-counts ]; then K=${2:?served_verdicts.sh block-counts WINDOW}; block_counts; exit $?; fi
if [ "${1:-}" = front-flags ]; then front_ok; exit $?; fi
if [ "${1:-}" = private-pieces ]; then private_pieces; exit $?; fi
if [ "${1:-}" = defer ]; then defer "${2:?served_verdicts.sh defer WHY}"; exit $?; fi
if [ "${1:-}" = served-compare ]; then K=${2:?served_verdicts.sh served-compare WINDOW}; served_compare; exit $?; fi
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
  [ -z "$FOLLOW" ] || follow_handoff gate-compose "served_verdicts.sh gate-compose PIECES VERDICT HOST" || exit 3
  gate_compose "${2:?served_verdicts.sh gate-compose PIECES [VERDICT [HOST]]}" "${3:-}"; exit $?
fi

# $OUT leaves the node: once the run is done, however it ends, each of its files that names a window or counts its own is
# held in full in the serve dir's private/out, its public form in its place (`served_verdicts.py publish`)
trap 'rc=$?; "$SAY_PY" "$SV" publish "$OUT" --private "$D/private/out" || echo "PUBLISH failed (exit $?): $OUT may name a window"; exit $rc' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
stop() { say "STOP at $1 (exit $2)"; exit "$2"; }
{ id -un; hostname; git -C "$TREE" rev-parse HEAD 2>/dev/null; nproc; free -g | head -2; nvidia-smi -L; echo "serve dir $D"; } \
  | tee "$OUT/machine.txt"
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
    || { say "PREFLIGHT: the compose follow-on composes from the served node's hand-off, and the serve dir $D holds none (private/compose/compose.env): copy each file its private/compose/inputs.txt lists to the same path here first"; exit 3; }
  [ -z "$CLASH" ] || { say "PREFLIGHT: the compose follow-on runs at the hand-off's settings (the serve dir's private/compose/compose.env), and the environment sets another$CLASH: unset it"; exit 3; }
  { [ "$(hostname)" != "$SERVED_NODE" ] && [ "$(hostname -s)" != "${SERVED_NODE%%.*}" ]; } \
    || { say "PREFLIGHT: this is the served node $SERVED_NODE, which the prover controls, so nothing composed here claims a record: compose on COMPOSE_NODE ($COMPOSE_NODE)"; exit 3; }
  sha=$(tree_sha)
  [ -n "$COMPOSE_TREE" ] && { [ "$sha" = "$COMPOSE_TREE" ] || [ -n "${FOLLOW_ANY_TREE:-}" ]; } \
    || { say "PREFLIGHT: the compose follow-on composes with the served node's tree ${COMPOSE_TREE:-(unknown)}, and this tree is ${sha:-(unknown)}: run it at that commit, or set FOLLOW_ANY_TREE=1"; exit 3; }
  held=$(cd "$D" && find . -type f -printf '%P\n' | "$SAY_PY" "$SV" witness) \
    || { say "PREFLIGHT: the compose follow-on holds no witness of the served run, and this node's serve dir does ($(echo "$held" | head -1 | cut -c1-400)): compose on a node that holds the hand-off alone"; exit 3; }
elif [ "${1:-}" = v3c ]; then  # V3c alone on an existing serve dir serves nothing: no row dir, taps or gpu-lease
  [ -e "$PY" ] || { say "PREFLIGHT: no $PY (VENV, which a node without vLLM's venv sets to the tree's .venv)"; exit 3; }
else
  for f in "$ROW/manifest.json" "$ROW/build_summary.json" "$HIDDEN_SO" "$NORM_TAP_SO" "$PY"; do [ -e "$f" ] || { say "PREFLIGHT: no $f"; exit 3; }; done
  command -v gpu-lease >/dev/null || { say "PREFLIGHT: no gpu-lease on this node"; exit 3; }
  if [ -n "$PEARLPLUS_TIE" ] && [ "$PEARLPLUS_INODES" != 0 ]; then  # the tied window's commit goes to the serve dir's filesystem
    at=$D; until [ -e "$at" ]; do at=$(dirname "$at"); done
    read -r itotal ifree fs < <(df --output=itotal,iavail,target "$at" 2>/dev/null | tail -1)
    [[ ${itotal:-} =~ ^[0-9]+$ && ${ifree:-} =~ ^[0-9]+$ && $PEARLPLUS_INODES =~ ^[0-9]+$ ]] \
      || { say "PREFLIGHT: the serve dir's free inodes can't be held to PEARLPLUS_INODES ($PEARLPLUS_INODES): df says ${ifree:-nothing} free of ${itotal:-nothing} on $at"; exit 3; }
    # a filesystem with no inode table (none in all) makes its inodes as it needs them
    [ "$itotal" = 0 ] || [ "$ifree" -ge "$PEARLPLUS_INODES" ] \
      || { say "PREFLIGHT: the serve dir's filesystem $fs has $ifree inodes free, $((PEARLPLUS_INODES - ifree)) short of PEARLPLUS_INODES ($PEARLPLUS_INODES), two commits of the tied window at three files a member: delete earlier runs' Pearl+ windows with \`research retention rm\`, or set SERVE_ROOT on another filesystem"; exit 3; }
  fi
  if [ "$PRODUCT" = onsite ] && [ -n "${COMPOSITE-1}" ]; then  # onsite's composite, which reads its own pieces' cases
    [ -z "$PEARLPLUS_TIE" ] \
      || { say "PREFLIGHT: under PEARLPLUS_TIE PoUW's verdict is Pearl+'s, and no composite holds onsite's pieces beside it (onsite's work subprotocol is PoUW's period case, which this run doesn't write, and remote's composite of record, whose is Pearl+'s run, holds no onsite piece): COMPOSITE set empty, for the record alone, or PRODUCT=remote"; exit 3; }
    [ "$COMPOSE_OFF_NODE" = 0 ] \
      || { say "PREFLIGHT: COMPOSE_OFF_NODE=1 composes the composite on COMPOSE_NODE ($COMPOSE_NODE), an auditor host, from the compose hand-off, and onsite's reads its own pieces' cases (PoUS's, the wipe's), which stay on the prover's side and which no hand-off carries: COMPOSITE set empty, for the final record alone there, PRODUCT=remote, or COMPOSE_OFF_NODE=0 for a development composite here"; exit 3; }
  fi
fi
front_ok || exit 3
SPLIT= DEFER=
if [ -n "${CERTIFIER_CHAIN_CMD:-}${CERTIFIER_PERIOD_CMD:-}" ]; then  # the split certifier: its period check writes its answer
  [ -z "${CERTIFIER_CMD:-}" ] || { say "PREFLIGHT: CERTIFIER_CMD or the split's CERTIFIER_CHAIN_CMD and CERTIFIER_PERIOD_CMD, not both"; exit 3; }
  [ -n "$V3C" ] || { say "PREFLIGHT: the split certifier's period check reads V3c's period.json, and V3C_CLASS is empty"; exit 3; }
  SPLIT=1
fi
case ${V3C_COMPOSE_CACHE:-/} in  # a kept compose cache: outside TREE, where the warm and the audit would put a relative path
  /*) ;;
  *) say "PREFLIGHT: V3C_COMPOSE_CACHE is the kept compose cache's directory, an absolute path (the warm and the audit run in the tree, where a relative one would put it), not $V3C_COMPOSE_CACHE: name it from /"; exit 3 ;;
esac
if [ -n "${V3C_WARM_JOBS:-}" ]; then  # V3c's composes made before its audit
  [[ $V3C_WARM_JOBS =~ ^[1-9][0-9]*$ ]] \
    || { say "PREFLIGHT: V3C_WARM_JOBS is how many of the window's templates warm composes at once, a positive integer, not $V3C_WARM_JOBS"; exit 3; }
  grep -q -- '"warm"' "$TREE/benchmarks/prove_path/records_audit.py" \
    || { say "PREFLIGHT: V3C_WARM_JOBS composes the window's templates before its audit with records_audit.py warm, which this tree's records_audit.py doesn't have"; exit 3; }
fi
case $DEVELOPMENT in '' | 1) ;; *) say "PREFLIGHT: DEVELOPMENT is 1, a development run, or unset, not $DEVELOPMENT"; exit 3 ;; esac
if [ -n "$V3C" ] && [ -z "$COMPOSE" ]; then  # V3c's audit's coins: the live challenger's, or a development run's
  case $V3C_CHALLENGER in
    local | loopback:*)
      [ "$DEVELOPMENT" = 1 ] \
        || { say "PREFLIGHT: V3c's audit is the sampled-proofs audit of record, and V3C_CHALLENGER (default PP_CHALLENGER) is $V3C_CHALLENGER, a development session, which only a development run takes: set PP_CHALLENGER=gate:USER@HOST with PP_PEER, PP_CHALLENGER_KEY and PP_GATE_KEY, or DEVELOPMENT=1"; exit 3; } ;;
    gate:?*@?*)
      [ -n "$V3C_PEER" ] \
        || { say "PREFLIGHT: V3c's audit's session on $V3C_CHALLENGER listens on its peer's 127.0.0.1, and V3C_PEER (default PP_PEER), this node's name in the gate's /etc/vy/peers, is unset"; exit 3; }
      [ -n "${PP_CHALLENGER_KEY:-}" ] && [ -r "$PP_CHALLENGER_KEY" ] \
        || { say "PREFLIGHT: V3c's audit's session on $V3C_CHALLENGER is held to the challenger key record PP_CHALLENGER_KEY, which is ${PP_CHALLENGER_KEY:-unset} and not a readable file"; exit 3; }
      if [ -n "$FOLLOW" ]; then here=${SERVED_PEER:-} what=SERVED_PEER
      elif [ -n "$H1_GATE" ]; then here=$H1_PEER what=H1_PEER
      elif [ "$POUW" = pearlplus ] && [ -n "$PEARLC_CHALLENGER_GATE" ]; then here=$PEARLC_CHALLENGER_PEER what=PEARLC_CHALLENGER_PEER
      else here= what=; fi
      if [ -n "$FOLLOW" ] && [ "$V3C_PEER" = "$here" ]; then
        say "PREFLIGHT: V3C_PEER (default PP_PEER) is $V3C_PEER, the served node's (SERVED_PEER), which deferred the audit: its session would listen on that node's 127.0.0.1, not this one's; name this node as the gate's /etc/vy/peers does"; exit 3
      elif [ -z "$FOLLOW" ] && [ -n "$here" ] && [ "$V3C_PEER" != "$here" ]; then
        say "PREFLIGHT: V3C_PEER (default PP_PEER) is $V3C_PEER, and $what is $here: each names the node the run serves on, where V3c's audit's session listens as the other one does"; exit 3
      fi ;;
    *) say "PREFLIGHT: V3C_CHALLENGER (default PP_CHALLENGER) is local, loopback:HOST:PORT or gate:USER@HOST, not $V3C_CHALLENGER"; exit 3 ;;
  esac
fi
if [ -z "$V3C" ] && [ -z "$FOLLOW$COMPOSE" ] && [ "$DEVELOPMENT" != 1 ]; then  # step 3's audit is then the one of record
  case $SP_CHALLENGER in local | loopback:*)
    say "PREFLIGHT: with V3c off (V3C_CLASS empty) step 3's audit is the sampled-proofs audit of record, and PP_CHALLENGER is $SP_CHALLENGER, a development session, which only a development run takes: set PP_CHALLENGER=gate:USER@HOST with PP_PEER, PP_CHALLENGER_KEY and PP_GATE_KEY, or DEVELOPMENT=1"; exit 3 ;;
  esac
fi
if [ -n "$PEARLPLUS_TIE" ] && [ -z "$COMPOSE" ]; then  # the window served under Pearl+, its commit and audit tied to Pearl+'s W
  [ "$POUW" = pearlplus ] || { say "PREFLIGHT: PEARLPLUS_TIE ties the window to Pearl+'s W, which POUW=pearlplus's piece writes: POUW is $POUW"; exit 3; }
  [ -z "${H1_TIE:-}" ] || { say "PREFLIGHT: PEARLPLUS_TIE or H1_TIE, not both: the window is tied to one W"; exit 3; }
  [ -n "$V3C" ] || { say "PREFLIGHT: PEARLPLUS_TIE commits the window at V3C_CLASS, and it is empty"; exit 3; }
  [ -z "$SPLIT" ] || { say "PREFLIGHT: under PEARLPLUS_TIE V3c runs once Pearl+'s piece wrote W, after step 4, and the split certifier's period check needs it before: set CERTIFIER_CMD instead"; exit 3; }
  if [ -z "$FOLLOW" ] && [ "${1:-}" != v3c ]; then  # the serve's run, whose piece commits the window tied to W
    outs=$(cut -d, -f3 <<< "$V3C")
    [ "$LOAD" = fixed ] && [ "$WINDOW_STEPS" = "$outs" ] \
      || { say "PREFLIGHT: under PEARLPLUS_TIE round s of Pearl+'s W is the window's s-th step, the class's s-th output, so the window is one full window of fixed length: LOAD=fixed (not $LOAD) and WINDOW_STEPS the class's outputs, $outs (not $WINDOW_STEPS)"; exit 3; }
    IFS=, read -r slots prompt _ <<< "$V3C"
    [ "$SLOTS" = "$slots" ] && [ "$PROMPTS" = "$prompt:$prompt" ] \
      || { say "PREFLIGHT: under PEARLPLUS_TIE the circuit cut at Pearl+'s linears reads its tables at the class's fixed load, its slots' requests of its prompt's ids each: SLOTS $slots (not $SLOTS) and PROMPTS $prompt:$prompt (not $PROMPTS)"; exit 3; }
    [ "$PRECISION" = bfloat16 ] \
      || { say "PREFLIGHT: PEARLPLUS_TIE serves the BF16 checkpoint's linears as Pearl+'s FP8 product, and PRECISION is $PRECISION: set it bfloat16"; exit 3; }
    [[ $ROW_ID == *__bf16__* ]] \
      || { say "PREFLIGHT: PEARLPLUS_TIE serves the BF16 checkpoint, and ROW_ID $ROW_ID is no BF16 row's (its id names __bf16__): set ROW to the BF16 row's dir"; exit 3; }
    case " $CLASSES " in *" $V3C "*) ;; *)
      say "PREFLIGHT: PEARLPLUS_TIE commits the window at V3C_CLASS $V3C, whose window circuit CLASSES ($CLASSES) doesn't count: add it"; exit 3 ;; esac
    [[ " ${SERVE_ARGS:-} " != *" --prefix-caching "* ]] \
      || { say "PREFLIGHT: under PEARLPLUS_TIE each request's prefill computes every row of its prompt, at which the cut circuit reads Pearl+'s tables, and SERVE_ARGS turns vLLM's prefix cache on (--prefix-caching): drop it"; exit 3; }
    [ -z "${CERTIFIER_FRONT:-}" ] \
      || { say "PREFLIGHT: under PEARLPLUS_TIE the window is V3C_CLASS's fixed load, and CERTIFIER_FRONT's group closes into a window of its own, which the run's pick takes on a tie in requests: unset CERTIFIER_FRONT"; exit 3; }
    grep -q -- '"--pearl-passes"' "$TREE/benchmarks/prove_path/prove_path.py" \
      || { say "PREFLIGHT: PEARLPLUS_TIE needs window-commit's --pearl-passes, its circuit cut at Pearl+'s linears, which this tree's prove_path.py doesn't take"; exit 3; }
    [ -z "$PEARLC_CHALLENGER_GATE" ] || [ -n "${MAIN_FILE:-}" ] \
      || { say "PREFLIGHT: under PEARLPLUS_TIE the gate's pin is held to main's first-parent line, which MAIN_SHA's (default this tree's HEAD) need not hold: set MAIN_FILE to \`git rev-list --first-parent origin/main\` of a fresh fetch, sent with the run"; exit 3; }
    DEFER=$(short_of_memory) || DEFER=  # step 6's audit goes to a follow-on, under either product
  fi
fi
if [ -n "${H1_TIE:-}$PEARLPLUS_TIE" ] && [ -z "$COMPOSE" ]; then  # V3c's audit tied to H1's or Pearl+'s W, in the serve's run and in `served_verdicts.sh v3c` alike
  if [ -n "${H1_TIE:-}" ]; then TIED=H1_TIE WHOSE="H1's"; else TIED=PEARLPLUS_TIE WHOSE="Pearl+'s"; fi
  grep -q -- '"--tie"' "$TREE/benchmarks/prove_path/records_audit.py" \
    || { say "PREFLIGHT: under $TIED V3c's audit is tied to $WHOSE W, whose tables the circuit reads, with records_audit's --tie, which this tree's records_audit.py doesn't take"; exit 3; }
  [[ ${V3C_LAW:-} == stratified:* ]] \
    || { say "PREFLIGHT: under $TIED every table $WHOSE W registers needs a sampled reader, which only stratified:K, a stratum of each table's readers, gives: V3C_LAW is ${V3C_LAW:-unset}"; exit 3; }
fi
if [ -n "${H1_TIE:-}" ] && [ "${1:-}" != v3c ] && [ -z "$COMPOSE" ]; then  # V3c's window commit at each H1 window's close, before its sample coin
  [ -n "$H1_GATE" ] && [ -n "${H1_CIRCUITS:-}" ] \
    || { say "PREFLIGHT: H1_TIE ties H1's windows, and H1_GATE or H1_CIRCUITS (whose prover state window-commit reads) is unset"; exit 3; }
  [ -n "$V3C" ] || { say "PREFLIGHT: H1_TIE commits each H1 window at V3C_CLASS, and it is empty"; exit 3; }
  grep -q -- '"--window-registration"' "$TREE/benchmarks/prove_path/prove_path.py" \
    || { say "PREFLIGHT: H1_TIE needs window-commit's --window-registration, its registration chained right after H1's, which this tree's prove_path.py doesn't take"; exit 3; }
  [[ $V3C_TIMEOUT =~ ^[0-9]+m$ ]] && [ "${V3C_TIMEOUT%m}" -le 90 ] \
    || { say "PREFLIGHT: under H1_TIE the window's sample coin waits for window-commit, and its session's reserve lasts the gate's 120 minutes from the serve's start: V3C_TIMEOUT is at most 90m, not $V3C_TIMEOUT"; exit 3; }
  LEASE_MIN=${LEASE_MIN:-150} LEASE_MEM_GB=${LEASE_MEM_GB:-160}
  if DEFER=$(short_of_memory); then  # step 6's audit and the composite go to a follow-on
    [ "$PRODUCT" != onsite ] || [ -z "${COMPOSITE-1}" ] \
      || { say "PREFLIGHT: under H1_TIE, where $DEFER, its audit and the composite are deferred to a follow-on, which composes from the deferral's list, and that list holds none of the cases onsite's composite reads of its own pieces (PoUS's, the wipe's): COMPOSITE set empty, PRODUCT=remote, or a node with the memory"; exit 3; }
  fi
fi
if [ -n "${DAY:-}" ] && [ -z "$FOLLOW$COMPOSE" ]; then  # the day's one circuit declared before the first window, R2 holding V3c's to it
  [[ $DAY =~ ^[0-9]+$ ]] && [ -n "${DAY_LOG:-}" ] && [[ ${DAY_PROGRAM:-} =~ ^[0-9a-f]{128}$ ]] \
    || { say "PREFLIGHT: DAY=N declares day N's circuit: it needs DAY_LOG (the deployment's registration log) and DAY_PROGRAM (the SHA-512, lowercase hex, of the program its windows register)"; exit 3; }
  [ -f "$TREE/adapters/vllm/verity_vllm/pipeline/serve_day.py" ] && grep -q -- '"--day"' "$TREE/benchmarks/prove_path/records_audit.py" \
    || { say "PREFLIGHT: DAY needs serve-openai's --day and records_audit's --day, which this tree hasn't"; exit 3; }
fi
if [ -z "$FOLLOW$COMPOSE" ] && [ "${1:-}" != v3c ]; then  # step 3's law draws every unit of the served compiler's transfer gates
  why=$(cd "$TREE" && "$SAY_PY" - benchmarks/prove_path/prove_path.py "$REPO" "$PP_PARTITION" 2>&1 <<'EOF'
import ast, json, sys
src, model, part = sys.argv[1:]
record = next((ast.literal_eval(n.value) for n in ast.parse(open(src).read()).body if isinstance(n, (ast.Assign, ast.AnnAssign))
               and "GATES_OF_RECORD" in [t.id for t in getattr(n, "targets", [getattr(n, "target", None)]) if isinstance(t, ast.Name)]), {})
gates = list(record.get(model, ()))
p = json.load(open(part)) if gates else None
if p is not None and not (str(p["law"]).startswith("work:") and set(gates) <= set(p["templates"])):
    print(f"{model}'s compiler of record lists {gates} as transfer gates (prove_path.GATES_OF_RECORD), every unit of which "
          f"step 3 proves beside work:K of the rest, and PP_PARTITION {part} is {p['law']} over {p['templates']}: give a "
          "work:K partition that holds them, and in PP_SETS their input sets")
EOF
) || { say "PREFLIGHT: step 3's partition PP_PARTITION $PP_PARTITION can't be read: $(tail -1 <<< "$why" | cut -c1-400)"; exit 3; }
  [ -z "$why" ] || { say "PREFLIGHT: $why"; exit 3; }
fi
if [ -n "${PREFLIGHT_ONLY:-}" ]; then  # the checks above alone: nothing kept, built or served
  if [ -n "$FOLLOW" ]; then plan="the follow-on of $SERVED_NODE's deferral"
    [ "$COMPOSE_OFF_NODE" = 0 ] || plan+=", its compose hand-off for COMPOSE_NODE ($COMPOSE_NODE)"
  elif [ -n "$COMPOSE" ]; then plan="the compose follow-on of $SERVED_NODE's hand-off: $HANDED_WHAT"
  else
    plan="the serve's lease ${LEASE_MIN:-60} min, capped at ${LEASE_MEM_GB:-128} GB"
    [ -z "$DEFER" ] || plan+="; $DEFER, so step 6 defers $DEFERRED_WHAT to a follow-on"
    [ "$COMPOSE_OFF_NODE" = 0 ] || plan+="; $HANDED_WHAT composed off this node, on COMPOSE_NODE ($COMPOSE_NODE), from its compose hand-off"
  fi
  say "PREFLIGHT passed on $(hostname -s), $NODE_MEM_GB GB: $plan. PREFLIGHT_ONLY: nothing kept, built or served"
  exit 0
fi
mkdir -p "$D" "$ROOT/cache" "$FW"
(umask 077; mkdir -p "$D/private")
[ -n "$COMPOSE" ] || [ -z "${KEEP_CMD:-}" ] || (cd "$TREE" && bash -c "$KEEP_CMD" _ "$ROOT") 2>&1 | tail -2
# H1's every window open runs its Lean verifier of record; without lake each is refused and serving goes on uncredited
if [ -z "$FOLLOW$COMPOSE" ] && { [ -n "$H1_GATE" ] || [[ " ${SERVE_ARGS:-} " == *" --protocols "* ]]; }; then
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --default-toolchain none > "$OUT/elan.log" 2>&1
  (cd "$TREE/verity/protocols/pouw/verifier/lean/check" && lake build pouw-check-difftest) > "$OUT/h1-lean.log" 2>&1 \
    || { tail -20 "$OUT/h1-lean.log"; say "PREFLIGHT: H1's Lean verifier (pouw-check-difftest) didn't build"; exit 3; }
fi

build() {  # what prove_path/run.sh builds, here while the endpoint serves; run.sh's own builds then find them done
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
  uv sync --all-packages --extra torch-cpu || return 5
  (cd verity/protocols/flock/verifier && lake build flock-verify flock-firewall) || return 7
  (cd verity/protocols/spot/verifier/lean && lake build sampled-proofs-verifier) || return 7
  mkdir -p "$OUT/m0-build"
  # FW is the node's, every run's: two builds in it at once break each other's crates
  RESEARCH_RUN_DIR=$OUT/m0-build FLOCK_WORK=$FW MODE=build GPU=0 flock "$FW/.build.lock" bash verity/protocols/flock/pod/60-circuit.sh || return 6
  # after flock-verify's build, whose package it requires by path: the inference-only verdict's `nci-check --count`,
  # without which the verdict is the Python count's alone
  (cd verity/protocols/inference_only/verifier/nci/lean && lake build nci-check) || echo "nci-check didn't build"
}
if [ -n "$COMPOSE" ]; then ( unset PYTHONPATH; compose_build ) > "$OUT/build.log" 2>&1 &
else ( unset PYTHONPATH; build ) > "$OUT/build.log" 2>&1 &
fi
BUILD=$!
if [ -n "$COMPOSE" ]; then compose; exit $?; fi

if [ "${1:-}" = v3c ]; then  # an existing serve dir's window: D and K set, its advice written (step 3); V3c alone
  say "COINS V3c's audit: $(v3c_coins)"
  if [ -n "$FOLLOW" ]; then follow; exit $?; fi
  [ -n "${K:-}" ] && [ -f "$D/advice/w$K/private/advice.bin" ] || { say "V3C: set D and K, a serve dir and its window with advice"; exit 3; }
  [ -f "$WREG" ] || CUDA_VISIBLE_DEVICES='' weights
  [ -f "$WREG" ] || exit 13
  # the verifier's record at the class from this tree, whatever an earlier run left in D (under H1_TIE reading H1's tables)
  v3c_circuit || exit 12
  v3c_commit
  wait "$BUILD"; rc=$?
  [ "$rc" = 0 ] || { tail -30 "$OUT/build.log"; stop build "$rc"; }
  [ "$v3c_ok" = 1 ] || exit 16
  v3c_proved; rc=$?
  [ -z "$PEARLPLUS_TIE" ] || window_tie
  private_pieces
  exit "$rc"
fi

H1_COINS="PoUW H1: not served (H1_GATE empty)"
PEARLPLUS="PoUW: Pearl+'s own served run on this card after the window's (POUW=pearlplus)"
[ -z "$PEARLPLUS_TIE" ] || PEARLPLUS="PoUW: Pearl+ on the window's own engine, its passes the window's linears (POUW=pearlplus, PEARLPLUS_TIE)"
LIVE_GATE=$H1_GATE LIVE_LANE=${H1_LANE:-vllm} LEAD=
if [ "$POUW" = pearlplus ] && [ -n "$PEARLC_CHALLENGER_GATE" ]; then
  LIVE_GATE=$PEARLC_CHALLENGER_GATE LIVE_LANE=$PEARLC_CHALLENGER_LANE
  PP_CREDIT="it credits nothing"
  [ -z "$PEARLPLUS_TIE" ] || PP_CREDIT="which the record takes as PoUW's verdict of record, the prover side's, only where pearlplus_of_record holds it"
  H1_COINS="$PEARLPLUS, its epoch's salt coin a live session's of its own on vy-challenger-1's gate (peer $PEARLC_CHALLENGER_PEER, lane $PEARLC_CHALLENGER_LANE), each pass's salt checked by core_check.live_salt in its piece, its serve inside a Pearl-C period of that lane's own, and its replay's draws a draw session's coin on that gate, reserved withheld once the period closed and opened on the run's commitment, taken once before the replay, checked against the signed ledger in its piece and cited by store id in the POUW line's coin; a development piece, $PP_CREDIT"
elif [ "$POUW" = pearlplus ]; then
  LEAD="Pearl+'s epoch coins are in-process and the record claims no Pearl+ verdict; "
  H1_COINS="$PEARLPLUS, its epoch's coins an in-process challenger's (PEARLC_CHALLENGER_GATE empty), which the serving process holds, and its replay's draws the Lean verifier's IO.getRandomBytes on this node, taken once before the replay and cited by store id in the POUW line's coin; a development piece, it credits nothing"
fi
if [ -n "$LIVE_GATE" ]; then
  if [ -n "${MAIN_FILE:-}" ]; then
    [ -s "$MAIN_FILE" ] || { say "PREFLIGHT: MAIN_FILE $MAIN_FILE is missing or empty"; exit 3; }
    [ "$MAIN_FILE" -ef "$OUT/main.txt" ] || cp "$MAIN_FILE" "$OUT/main.txt"
    export MAIN_FILE=$OUT/main.txt
  elif [ -n "$MAIN_SHA" ]; then
    git -C "$TREE" rev-list --first-parent "$MAIN_SHA" > "$OUT/main.txt" || { say "PREFLIGHT: MAIN_SHA $MAIN_SHA not in this tree"; exit 3; }
    export MAIN_FILE=$OUT/main.txt
  fi
  "$PY" benchmarks/one_stage/served_verdicts.py gate --gate "$LIVE_GATE" --lane "$LIVE_LANE" ${MAIN_FILE:+--main "$MAIN_FILE"} 2>&1 \
    | publish_lines
  [ "${PIPESTATUS[0]}" = 0 ] || stop gate 3
fi
if [ -n "$H1_GATE" ]; then
  H1_COINS="PoUW H1: each window's coins from a session of its own on vy-challenger-1's gate (peer $H1_PEER, T $T, S $WINDOW_STEPS), a development run: it credits nothing"
  export POUW_CMD=${POUW_CMD:-"$TREE/.venv/bin/python benchmarks/one_stage/served_verdicts.py pouw \$1 \$2 \$3${H1_STATEMENTS:+ --statements $H1_STATEMENTS}${H1_LIVE_EVIDENCE:+ --live-evidence $H1_LIVE_EVIDENCE}"}
fi
if [ "$SP_CHALLENGER" != local ]; then
  COINS="the challenger $SP_CHALLENGER (peer ${PP_PEER:-?})"
elif [ "${PP_CHALLENGER:-local}" != local ]; then
  COINS="a local development challenger session on $(hostname -s), the served node, for step 3, a development check, since V3c's audit's live session takes the window's run root, which the gate opens once, ever"
else
  COINS="a local development challenger session on $(hostname -s), the served node, not an auditor's (PP_CHALLENGER local)"
fi
COINS+="; $H1_COINS;"
[ -z "$V3C" ] || COINS+=" V3c's audit: $(v3c_coins);"
say "COINS ${LEAD}sampled proofs: $COINS the bandwidth audit's: vy-auditor-2's bw_party, inside PoUS's piece"

# 0. B: the row's Build of record from its declared inputs, before the lease; the traced Build only its cross-check
vv predict row --row "$ROW_ID" --out "$PRED" > "$OUT/predict_row.json" 2> "$OUT/predict_row.err"
rc=$?
[ "$rc" = 0 ] && [ -f "$PRED/predicted.json" ] \
  || { say "PREDICT-ROW $ROW_ID refused (exit $rc): $(grep -v '^\[predict\]' "$OUT/predict_row.err" | tail -2 | cut -c1-600)"; stop predict 17; }
cp "$PRED/predicted.json" "$OUT/"
say "PREDICT-ROW $ROW_ID: $(cut -c1-600 "$OUT/predict_row.json")"
CONFIG=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["config"])' "$PRED/predicted.json")
vv predict compare "$ROW" --row "$ROW_ID" --config "$CONFIG" --predicted "$PRED" --out "$OUT/predict_compare.json" \
  > "$OUT/predict_compare.log" 2>&1
rc=$?
say "PREDICT-COMPARE against the traced Build $ROW (exit $rc): $(grep -E '^(rank|predicted|\[predict\])' "$OUT/predict_compare.log" | tr '\n' ' ' | cut -c1-1200)"
[ "$rc" = 0 ] || stop predict-compare 18
[ -n "$PEARLPLUS_TIE" ] || inference_policy  # under PEARLPLUS_TIE in serve(), once serve-phi states φ on the card
if [ -n "$PEARLPLUS_TIE" ]; then  # Pearl-C's sm_90 cubin, which the serve's engine and Pearl+'s piece read
  ( export RESEARCH_RUN_DIR=$D/private/pearlplus-cubin VERITY_BENCH_VENV=${VERITY_BENCH_VENV:-$VENV}
    mkdir -p "$RESEARCH_RUN_DIR" && cd "$TREE" || exit 1
    # shellcheck source=/dev/null
    . benchmarks/harness/env.sh && . benchmarks/harness/setup/pearl_c_sm90.sh \
      && printf '%s\n' "$PEARLC_KERNEL" "$PEARLC_CUBIN" > "$RESEARCH_RUN_DIR/setup.paths" ) > "$OUT/pearlplus-cubin.log" 2>&1
  rc=$?
  { read -r PEARLC_KERNEL && read -r PEARLC_CUBIN; } < "$D/private/pearlplus-cubin/setup.paths" 2>/dev/null
  [ "$rc" = 0 ] && [ -s "${PEARLC_CUBIN:-}" ] \
    || { say "PEARLPLUS-CUBIN: pearl_c_sm90.sh failed (exit $rc): $(tail -1 "$OUT/pearlplus-cubin.log" | cut -c1-400)"; stop pearlplus-cubin 19; }
  export PEARLC_KERNEL PEARLC_CUBIN
  say "PEARLPLUS-CUBIN: pearl_c_sm90.sh built it (sha256 $(sha256sum "$PEARLC_CUBIN" | cut -c1-16)); the window serves $PEARLPLUS_ARM on it, sm_90's BF16 rows $([ "$PEARLC_BF16_ROWS" = 0 ] && echo off || echo on) (PEARLC_BF16_ROWS=$PEARLC_BF16_ROWS)"
fi

# 1-2. the endpoint and its requests, under the GPU lease
export D OUT PRED
GPU_LEASE_WHO=${GPU_LEASE_WHO:-vllm-composed} gpu-lease 1 --wait --max-min "${LEASE_MIN:-60}" --mem-gb "${LEASE_MEM_GB:-128}" \
  -- bash "$TREE/benchmarks/one_stage/served_verdicts.sh" serve
rc=$?
[ "$rc" = 0 ] || stop serve "$rc"
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
cp "$D/period.json" "$OUT/serve/" 2>/dev/null
[ -d "$D/h1" ] && (cd "$D" && cp --parents h1/window-program.json.gz h1/window-program.meta.json "$OUT/serve/" 2>/dev/null)
[ -n "${H1_CIRCUITS:-}" ] && [ -f "$H1_CIRCUITS/circuit-files.json" ] && cp "$H1_CIRCUITS/circuit-files.json" "$OUT/serve/"
# the serve dir's step trace and the load's records stay on the node; what the stock control's table reads is their aggregate,
# in full in the serve dir's private/out/serve once `publish` holds the load's counts and span.
# A front's load is the warden's, paced by its buckets, which no stock run's is, and one window's aggregate is its timing.
if [ "$LOAD" = front ]; then
  say "OVERHEAD: none with LOAD=front (the warden paces its requests, so no stock control compares)"
else
  "$PY" "$TREE/benchmarks/one_stage/served_overhead.py" metrics "$D" > "$OUT/serve/overhead.json" 2> "$D/overhead.err" \
    || { say "OVERHEAD: no aggregate: $(cut -c1-300 "$D/overhead.err")"; rm -f "$OUT/serve/overhead.json"; }
fi

# 3. sampled proofs: the window, its advice and Programs, V3b's window circuit, then C-Flock --zk and the Lean verifier
# the pick (its number, steps, requests and users, and each window refused and why) stays in the serve dir's private/
PICK=$("$PY" benchmarks/one_stage/served_verdicts.py window "$D" --slots "$SLOTS" ${WINDOW:+--window "$WINDOW"} 2> "$D/private/window.err")
[ -n "$PICK" ] || { say "WINDOW refused: no sealed window qualifies (why, by window, in the serve dir's private/window.err)"; stop window 12; }
(umask 077; echo "$PICK" > "$D/private/window.json")
K=$(echo "$PICK" | "$PY" -c 'import json, sys; print(json.load(sys.stdin)["window"])')
say "WINDOW: $([ -n "${WINDOW:-}" ] && echo "the one WINDOW names" || echo "the sealed one with the most requests"), none carried across windows, at the class's $SLOTS slots (the pick in the serve dir's private/window.json)"
say "CLASSES: the window circuit's ${CLASSES}; the certifier's scan ${CERTIFIER_SCAN:-W63}, the one class the gate serves"
CERT=
if [ -n "${CERTIFIER_CMD:-}" ]; then  # the certifier's chain is the longest piece, so it starts once the window is picked
  # kept off CPUs 0 and 1 (CERT_CPUS), which PoUS's answers need idle: the chain's load stretches a round past Δ + RTT
  CERT_TASKSET=(); NCPU=$(nproc)
  command -v taskset > /dev/null && [ "$NCPU" -gt 2 ] && CERT_TASKSET=(taskset -c "${CERT_CPUS:-2-$((NCPU - 1))}")
  ${CERT_TASKSET[@]+"${CERT_TASKSET[@]}"} "$PY" benchmarks/one_stage/served_verdicts.py pieces --serve-run "$D" --window "$K" --out "$OUT/pieces" \
    --product "$PRODUCT" --only certifier > "$OUT/pieces-certifier.txt" 2>&1 &
  CERT=$!
fi
cd "$TREE/adapters/vllm" || exit 9
CA=$(compiler_args)
[ -n "$CA" ] || stop compiler 13
echo "registered compiler: $CA" > "$OUT/compiler.txt"
WA=()
[ -f "$WREG" ] && WA=(--weights-of-record "$WREG")
for k in $("$PY" -c "import json,sys; [print(json.loads(l)['window']) for l in open(sys.argv[1]) if l.strip()]" "$D/windows.jsonl"); do
  advice_of "$k" "$CA"
  [ "$k" = "$K" ] && break
done
# the advice and compile logs name the request Programs by their prompt lengths and count the window's requests, which
# are values: they stay in the serve dir
[ -f "$D/advice/w$K/registration.json" ] || { say "ADVICE: none written (the serve dir's advice.log, on the node)"; stop advice 14; }
"$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" public-registration "$D/advice/w$K/registration.json" > "$OUT/registration.json" \
  || { say "ADVICE: its registration has no public form"; stop advice 14; }
"$PY" tools/advice_compile.py "$D" --window "$K" > "$D/compile_w$K.log" 2>&1
rc=$?
say "ADVICE: $("$PY" -c 'import json, sys; w = json.load(open(sys.argv[1]))["windows"][sys.argv[2]]; print("root", w["root"][:16] + "…", "bits padded to the cap")' \
  "$D/advice/.w$K/windows.json" "$K" 2>&1 | cut -c1-200); $("$PY" -c 'import json, sys; print(len(json.load(open(sys.argv[1])).get("registered_roots") or {}))' \
  "$D/advice/w$K/registration.json" 2>&1 | cut -c1-200) weight roots recorded (${WA[*]:-$NO_WEIGHTS}); compile (advice_compile.py) exit $rc$([ "$rc" = 0 ] || echo ", its log in the serve dir, on the node")"
[ "$rc" = 0 ] || stop compile 15
cd "$TREE" || exit 9
wc_ok=0
circuits && wc_ok=1
block_counts
wait "$BUILD"; rc=$?
inference_only &  # beside sampled proofs, which it doesn't read; the record waits for it
IOV=$!
[ "$rc" = 0 ] || { tail -30 "$OUT/build.log"; wait "$IOV"; stop build "$rc"; }
export PP_PARTITION PP_SETS=${PP_SETS:-}
env -u PYTHONPATH PP_SERVE_RUN="$D" PP_WINDOW="$K" PP_SERVE_ROWS="$D/windows/w$K" PP_RECORDS=1 PP_ONLY="${PP_ONLY:-honest}" \
  PP_CHALLENGER="$SP_CHALLENGER" PP_MAIN="${PP_MAIN:-${MAIN_FILE:-}}" PP_VLLM_PYTHON="$PY" PP_FLOCK_WORK="$FW" \
  OUT="$OUT/sampled-proofs" bash benchmarks/prove_path/run.sh > "$D/sampled-proofs.log" 2>&1
rc=$?
SP=$D/sampled-proofs-served
rm -rf "$SP"; [ ! -d "$OUT/sampled-proofs/served" ] || { mv "$OUT/sampled-proofs/served" "$SP" \
  && (cd "$SP" && find . -name '*.json' | sed 's|^\./||') | "$SAY_PY" "$SV" rebase "$SP" "$OUT/sampled-proofs/served"; }
# run.sh's records copy holds the window's request lists (windows.jsonl), its request count and prefill tokens
# (serve.json) and the advice's length (registration.json): the published copy is the registration's public form alone
SR=$OUT/sampled-proofs/served-records
rm -f "$SR/windows.jsonl" "$SR/serve.json"
if [ -f "$SR/registration.json" ]; then
  "$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" public-registration "$SR/registration.json" > "$SR/registration.public.json" \
    && mv "$SR/registration.public.json" "$SR/registration.json" || rm -f "$SR/registration.json" "$SR/registration.public.json"
fi
say "SAMPLED-PROOFS (run.sh exit $rc): $("$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" sampled-line "$D/sampled-proofs.log" \
  --partition "$PP_PARTITION" 2>&1 | cut -c1-3000)$(
  grep -q '^RECORDS' "$D/sampled-proofs.log" || echo " (run.sh's log in the serve dir, on the node)")"

# 6. here with the split certifier, whose period check reads V3c's period.json: its piece then reads the answer it wrote
if [ -n "$SPLIT" ]; then
  v3c_commit && v3c_proved
  export CERTIFIER_CMD=true
fi

# 4. the other subprotocols, each its piece's verdict or MISSING with the command it lacks
TPY=$TREE/.venv/bin/python
"$TPY" benchmarks/one_stage/served_verdicts.py pieces --serve-run "$D" --window "$K" --out "$OUT/pieces" ${CERT:+--skip certifier} \
  --product "$PRODUCT" 2>&1 | publish_lines
[ -z "$CERT" ] || { wait "$CERT"; publish_lines < "$OUT/pieces-certifier.txt"; }
# under PEARLPLUS_TIE, once Pearl+'s piece wrote W: the served compare, whose line the record names
SC=() sc=0
if [ -n "$PEARLPLUS_TIE" ]; then served_compare || sc=1; SC=(--served-compare "$SERVED_COMPARE"); fi

# 6. here under PEARLPLUS_TIE, the commit Pearl+'s piece tied to W: the record reads the audit's tie to W
WT=() P3C=
if [ -n "$PEARLPLUS_TIE" ]; then  # on a node short of the audit's memory, step 6's deferral below holds the tie
  WT=(--window-tie "$(pearl_dir)/tie")
  [ -n "$DEFER" ] || { v3c_commit && v3c_proved; window_tie; P3C=1; }
fi

# 5. one record of every verdict
wait "$IOV"
LEFT=
if [ "$COMPOSE_OFF_NODE" = 1 ]; then
  LEFT="$HANDED_WHAT, composed off this node, on COMPOSE_NODE ($COMPOSE_NODE), from its compose hand-off"
  [ -z "$DEFER" ] || LEFT="$DEFERRED_WHAT, to a follow-on on a node with the memory, which writes the compose hand-off: $DEFER; and $LEFT"
elif [ -n "$DEFER" ]; then LEFT="$DEFERRED_WHAT, to a follow-on on a node with the memory, which composes the final record: $DEFER"; fi
"$TPY" benchmarks/one_stage/served_verdicts.py record --serve-run "$D" --window "$K" --sampled-proofs "$SP" \
  --pieces "$OUT/pieces" --out "$OUT/record" --product "$PRODUCT" ${WT[@]+"${WT[@]}"} ${SC[@]+"${SC[@]}"} \
  --key-sha512 "$H1_KEY_SHA512" --lane "${PEARLC_CHALLENGER_LANE:-${H1_LANE:-vllm}}" \
  ${LEFT:+--deferred "$LEFT"} 2>&1 | publish_lines
rec=${PIPESTATUS[0]}
private_pieces
# 6. V3c after the record, each step within its limit: it gates nothing. Under H1_TIE it comes before the composite,
# whose one-stage and policy cases are its audit's of the record H1 bound; on a node short of the audit's memory, the
# tie held here, the audit and the composite are deferred to a follow-on. Under COMPOSE_OFF_NODE=1 the compose hand-off
# comes last, after V3c in every run, so the composite composed from it reads V3c's audit's cases.
v3c_done=$SPLIT$P3C left=0
if [ -n "$DEFER" ]; then v3c_commit; defer "$DEFER" || left=1; v3c_done=1
elif [ -z "$v3c_done" ] && [ -n "${H1_TIE:-}" ]; then v3c_commit && v3c_proved; v3c_done=1; fi
comp=0
[ -n "$DEFER" ] || [ "$COMPOSE_OFF_NODE" = 1 ] || [ -z "${COMPOSITE-1}" ] \
  || TPY=$TPY bash benchmarks/one_stage/served_composite.sh "$D" "$OUT" || comp=1
if [ -n "$v3c_done" ]; then :
elif [ -n "$V3C" ]; then v3c_commit && v3c_proved
else say "V3C: not in this run (V3C_CLASS empty): \`served_verdicts.sh v3c\` runs it on the serve dir"; fi
[ -n "$DEFER" ] || [ "$COMPOSE_OFF_NODE" = 0 ] || handoff "$OUT/pieces" "$OUT/record/inference-only.json" || left=1
echo "---- verdicts"; cat "$OUT/verdicts.txt"
[ "$wc_ok" = 1 ] && [ "$rec" = 0 ] && [ "$comp" = 0 ] && [ "$left" = 0 ] && [ "$sc" = 0 ] \
  && grep -q '"composed": "\(pending\|accept\)"' "$OUT/verdicts.txt"
