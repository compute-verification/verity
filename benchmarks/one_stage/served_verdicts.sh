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
#     Program digest predict's. Each closed window is sealed as a
#     rows-sha512 row dir, and the seal opens each one's replay bundle (`deferred_rows.of_record`). Once it listens, before
#     the first request, the committer registers its weights (`prove_path.py weights-registration`: each weight parameter of
#     the checkpoint CKPT, REPO's snapshot at REV under HF_HOME, and the rotary table at the words the engine serves,
#     $D/private/cos_sin_cache.npy, once within the analytic table's bound, committed under salts sampled from the committer's
#     key $D/private/weights.key, 0600, neither of which a record copies); WEIGHTS says how many it roots and which it leaves unrooted, and
#     why, or NO WEIGHTS REGISTRATION why not, and then every weight read counts free. Under `--advice --advice-cap CAP`
#     (`serve_advice`) the server registers the served compiler B, capped at CAP bits (ℓ_a; by default the least that pads
#     every Commit period within the run's limits, `advice_cap`), before its engine loop runs
#     ($D/advice_registration.json), with the day's declaration of B as its one program (SERVE_ARGS' `--advice-day`, default
#     0; `--advice-log` extends an earlier day's log), and commits each closed window's advice as it serves under the weights registration
#     (`--advice-weights`), each window's line ($D/advice.jsonl: its root and ℓ_a) out before any coin of the window.
#  2. requests: REQUESTS greedy requests (default SLOTS) from the certifier's traffic generator (`examples/vllm/serve_load.py --varied`:
#     prompts of PROMPTS tokens, at most MAX_TOKENS outputs), numbered under one release run and sent together (RATE a second),
#     so the engine's idle closes them into one window. With LOAD=fixed (default varied) they are one full window of fixed
#     length instead (`serve_load.py --fixed SLOTS --window-steps WINDOW_STEPS`): SLOTS requests of PROMPTS tokens, each
#     with ignore_eos and max_tokens WINDOW_STEPS, all prefilled at the window's first step and run through its last. With
#     LOAD=front there are none: CERTIFIER_FRONT's group, sent through the network certifier, is the run's one load and
#     its execution block the one the run audits (a period that admits a single block, say). A front that failed before a
#     window closed stops the endpoint at once, with no window to seal; one that failed after (its leak guard holding a
#     file, say) leaves its exit to the certifier's verdict, and the run goes on with the window it sealed.
#     CERTIFIER_FRONT runs within CERTIFIER_FRONT_TIMEOUT (default 20m), and a flag it passes to `certifier_piece.py front`
#     that this tree's front doesn't take, a front whose endpoint isn't on the id form the network certifier sends
#     (SERVE_ARGS without `--protocol ids`), or a V3C_CLASS whose window isn't one link-window of the network certifier
#     the gate runs behind this tree's front (`prove_path.py window-class`: 8,32,4,4,16 at the front's scan class w63k4,
#     the class of record at w63) is refused before the lease. The
#     lease ends once every closed window is sealed. What names a window's requests (the serve dir's windows.jsonl,
#     seals.jsonl, requests.jsonl and serve.json, each window's PoUW calls record, the load's records, timings file and
#     log, and the certifier front's session, bundle and log, in the serve dir's private/, of which the CERTIFIER-FRONT
#     line says only `served_verdicts.py front-line`'s fields) stays in the serve dir on the node; what leaves it is their
#     aggregate ($OUT/serve/overhead.json, except with LOAD=front, whose load the network certifier paces and no
#     stock control compares to) and verdict lines that name no window and count none of a window's own: no window's number, no
#     count of the windows closed, sealed or refused, no engine step, request count or user of one. Each line goes out
#     through `served_verdicts.py public-lines`, in full to the serve dir's private/verdicts.txt, and once the run is
#     done, however it ends, `served_verdicts.py publish` holds in full in the serve dir's private/out each file of $OUT
#     that names or counts a window, its public form in its place. Each window's advice is padded to its compiler's cap
#     ℓ_a before it is committed and compiled (`advice_windows.pad`), so its registration and its rows say ℓ_a, never the
#     advice's length; the registration's published copy ($OUT/registration.json) is refused unless it is so
#     (`served_verdicts.py public-registration`).
#  3. sampled proofs, on the window `served_verdicts.py window` picks (the most requests, none carried across windows; the
#     pick stays in the serve dir's private/window.json, the WINDOW line saying only the class's SLOTS slots, and CLASSES
#     the window circuit's classes beside the certifier's scan class, CERTIFIER_SCAN, default W63, the one the gate serves):
#     ADVICE-LINE holds the server's B to the compiler `prove_path.py compiler` recomputes from serve.json (published as
#     $OUT/compiler.json) and every closed window's advice line to the time before the run's first coin
#     (`served_verdicts.py advice-line`), then the prover's check that this window's request Programs compile
#     (`tools/advice_compile.py`, honest serving's completeness, which nothing on the verifier's side reads); V3b's window
#     circuit (`prove_path.py window-circuit --pins` the weights registration's published copy, and PIN's NAME=ROOT if
#     set), the verifier's record of the class circuit, built from the class and $OUT's registered compiler (held to the
#     advice registration's binding) alone, never the serve dir or the advice, and its registered and free
#     input bits at those
#     pins: at each class of CLASSES (default 4,32,4, V2's), each within WC_TIMEOUT (default 30m), all of which the run needs;
#     and at the class of record, a measurement within WC_RECORD_TIMEOUT (default 45m), run after the record (step 5),
#     since nothing reads it, whose timeout or refusal is said and never fails the run (unless CLASSES names `record`, which
#     then gates here instead); then the window audited from its records alone, by V3c (step 6, run here): C-Flock `--zk`
#     on the class circuit's sampled proof blocks behind the Lean firewall, and the Lean verifier's verdict. That circuit
#     is the class's, whatever the window's requests, and reads no advice: neither its proofs nor the verifier read one,
#     so no advice opening reaches the verifier.
#  4. PoUW (H1 credit, or with POUW=pearlplus Pearl+'s own served run, below), PoUS, the bandwidth audit and the network
#     certifier: `served_verdicts.py pieces` runs each one's piece
#     where POUW_CMD, POUS_CMD, BANDWIDTH_CMD or CERTIFIER_CMD names it, one at a time (the certifier's alone, the longest,
#     from the moment the window is picked, beside step 3, its lines said after the others'; split in two, the answer step
#     6's period check wrote, V3c having run before this step, and none where step 6 defers V3c's audit, which that
#     period check reads: the piece refused, naming the deferral, `pieces --deferred`), each within <COMPONENT>_TIMEOUT
#     (else PIECE_TIMEOUT, default 60m), else names it MISSING with the command it lacks and its owner. No verdict is made
#     for a piece that didn't run or refused to; one that ran and wrote no answer errs. A piece that needs a GPU takes its
#     own `gpu-lease`, the serve's being over. POUS_CMD defaults to 'bash benchmarks/pous/served_piece.sh "$@"' where the
#     tree has it and this node holds the memory challenger's key (~/.ssh/vy-auditor2-pous). That piece runs the bandwidth
#     session over the store it holds and writes its answer where the bandwidth piece's goes, which BANDWIDTH_CMD collects:
#     by default, once POUS_CMD is set, '"$PY" benchmarks/one_stage/served_bandwidth.py collect --serve-dir "$1" --window
#     "$2" --out "$3"'; with no PoUS piece, 'bash benchmarks/one_stage/served_bandwidth.sh "$1" "$2" "$3"' where the tree has
#     it and this node holds the bandwidth verifier's key (~/.ssh/vy-auditor2-bw) and its registered store (BW_STORE's
#     setup.json). Set empty, a command is MISSING.
#     A subprotocol that runs on the serving engine (H1's `--protocols pouw:...`) goes in SERVE_ARGS, serve-openai's
#     further arguments, and its piece reads what it wrote in the serve dir. The wipe's piece runs in step 1 instead: with
#     WIPE_CMD set (by default 'bash benchmarks/pous/streamed_wipe/served_wipe.sh "$1" "$2" "$3"' where the tree has it;
#     empty, none), serve() calls it in TREE, under the serve's lease and before serve-openai starts, within WIPE_TIMEOUT
#     (default 20m), with (the serve dir D, `all`, $OUT/pieces/wipe), since the window isn't picked yet, and says its last
#     line as WIPE. The record reads its pieces/wipe/answer.json and composes the wipe only when that answer accepts and
#     pieces/wipe/served.json names this node as the server it ran on. A wipe that wrote no answer (refused, or failed
#     before its judge) leaves the wipe missing, one served elsewhere stays missing, and one whose answer rejects makes the
#     record reject: a failed wipe is never accepted.
#  5. the record: `served_verdicts.py record`: the all-protocols window report ($OUT/record/verdicts.json) and the block
#     record ($OUT/record/block-record.json), composed `pending` while any subprotocol is missing, `accept` only when every
#     one accepted on this node; any other server's answer is carried as elsewhere. The certifier's chain files that list
#     or count the window's requests (CERTIFIER_PRIVATE) then move to the serve dir's private/pieces/certifier (and, where
#     the run ends before the record, as it exits). Then,
#     unless COMPOSITE is set empty,
#     the product's composites' Lean verdicts on the run's cases (`served_composite.sh`'s docstring). Their verdicts gate
#     nothing, but a composite that didn't run (a tree without composite.py) or whose Lean didn't answer fails the run.
#     With the network certifier in the run, its piece (CERTIFIER_CMD) or its front (CERTIFIER_FRONT), `onsite_record.py`
#     then writes the onsite demo's record
#     ($OUT/record/onsite.json: every verdict its run of record needs, the network bound term by term and over each
#     monitoring period, labelled with the run's WINDOW_STEPS and MAX_TOKENS, beside it ADVICE_CHARGE's measured
#     advice charge, a test run's advice-charge.json, below, and
#     the re-proof inputs, the serve dir's and $OUT's files a later re-proof of the sampled units needs, with
#     REPROOF_COPY's copy of the serve dir's, NODE:/PATH, checked against REPROOF_LISTING, its listing kept off $OUT;
#     and its checks of the auditor's inputs, CERTIFIER_KEY's record of the network certifier and SITE_REGISTRATION's
#     registration of the deployment, both the auditor's and never the prover's node's, without which the network
#     certifier's verdict is not established; its check that the files remote's sampled proofs case registers are the
#     ones its params name by SHA-512 (the serve dir's private/sampled-proofs/case.json, where the composite wrote it),
#     its public file one V3c's audit pinned (its window/public-pin.json), at staging before the draw or completed
#     once the tie held, and its params' layout the one that pin fixed before the draw, without which neither that
#     verdict nor a composite's that reads the case is; and, where the composite wrote the
#     inference-only case, where its α and Γ
#     came from, its private policy.json's read: off the run's own division, or ALPHA and GAMMA, never a registered
#     policy); its ONSITE line gates nothing.
#     PRODUCT (default onsite) is the product the run composes: onsite every subprotocol; remote sampled proofs and PoUW
#     alone, its other pieces dropped (not run, no default command, the wipe's and the certifier's included; the
#     preflight refuses any onsite command set), its record composing their verdicts alone and its composite remote's;
#     remote-network remote's pieces and the network certifier's, the wipe's, PoUS's and bandwidth's dropped as remote
#     drops them, its composites remote's and remote-network's. Remote-network's egress tie refuses a certifier's period
#     with a compute audit on any registration but the run's, so the preflight refuses CERTIFIER_CMD there: the whole
#     piece's audits are its scan's own, and only the split's one audit, V3c's of the served window, is the run's.
#     remote-network-no-pouw is remote-network's but PoUW's piece, dropped too (`block_record.PRODUCTS`), its composite
#     remote-network-no-pouw's alone: it credits no verified compute, and the same preflight holds.
#     CERTIFIER_FRONT is remote's too: the network certifier in front of the endpoint is how its requests are served, not
#     a piece, so its session stays in the serve dir's private/ and its exit is the CERTIFIER-FRONT line's, which remote's
#     record doesn't read. Behind the front remote also takes the split certifier's CERTIFIER_CHAIN_CMD and
#     CERTIFIER_PERIOD_CMD (step 6), whose period case and answer the onsite record reads; remote's record drops the
#     certifier.
#     Beside them, the inference-only verdict (`inference_only.py`), at IO_CLASS (default the first class of CLASSES but
#     `record`): its policy, (c, k, n) and every input to them, is written before the lease from the run's settings alone
#     (the class, REPO at REV, LEASE_MIN), $OUT/inference-only-policy.json and its INFERENCE-ONLY POLICY line; once the
#     build is done, step 3's window circuit record at the class is judged by `NciCheck.Count`'s rule beside sampled proofs,
#     with `nci-check --count` where the build built it, into $OUT/record/inference-only.json (the judgment against n and
#     no count, n judged on the exact count) and its INFERENCE-ONLY line, in full with the exact counts in the serve dir's
#     private/inference-only.json. It gates nothing.
#  6. V3c, sampled proofs' audit of record, run before step 4 (V3C_CLASS set empty is refused): it proves the window
#     circuit's own units at V3C_CLASS (default 4,32,4; behind this tree's certifier front the one whose window is the
#     network certifier's link-window, 8,32,4,4,16 at the front's w63k4, which the preflight holds it to). Untied,
#     CLASSES must hold it, since step 3 writes the record its commit is held to only at those, and the preflight refuses
#     CLASSES that don't. `prove_path.py window-commit`, within V3C_TIMEOUT (default 15m), commits the window as that
#     class circuit's units, the weights it serves under the committer's key, so each served as registered has the
#     registration's root; its circuit.json less its run fields is held to the verifier's own window-circuit record at the
#     class; then `records_audit.py audit --circuit --registered`, within V3C_AUDIT_TIMEOUT (default 30m), samples the
#     registration's members (V3C_LAW, default subset:1, stratified:1 under H1_TIE) on the run's coins (Coins, below) and
#     proves them, its composes made before it with V3C_WARM_JOBS (at `v3c_audit`), its sessions one at a time and Lean's
#     checks of finished members beside them, V3C_LEAN_WORKERS at a time: on a node it shares default 1, within
#     V3C_AUDIT_MEM_GB beside the running session's estimate (`a5.SESSION_BYTES_PER_CIRCUIT_BYTE`), a follow-on's default
#     a5's LEAN_WORKERS, 3, within what its node has free beside that estimate (`a5.LeanPool`); its negatives after the
#     audit record one after another, or with V3C_NEGATIVE_WORKERS that many at once, each in a worker process with its
#     own Lean verifier process, started only while its memory estimate, read then, fits (`a5.NegativeWorkers`); then
#     `records_audit.py period` writes the audit as the network certifier's period reads it ($D/v3c-w<K>-<class>/period.json,
#     and a copy in $OUT/v3c-audit: the registration, the Program, its answer and its free inputs), and V3c fails where it
#     writes none. The record reads that audit ($D/v3c-w<K>-<class>/audit, on the window commit's registration). No
#     answer accepts beside an audit that exited nonzero: the period's answer and the record's row are one function of
#     the audit (`served_integrity.answer`), a reject saying so where its record accepts and its summary doesn't say it
#     passed, and `v3c_proved` returns the audit's exit. The audit's directory is one attempt's: where an attempt
#     reached its draw (the registration's receipt, which a5 writes before it draws) or another audit is running (its
#     lock on the directory's audit.lock), `records_audit.py audit` starts no other, and an attempt that died after
#     its receipt is a reject that says the audit did not finish;
#     $OUT/v3c-audit holds what it reads of it (`served_integrity.py public-copy`), the registration it audited and that
#     period.json, nothing else (`kind_scan.V3C_DOCS`): the audit's logs and main's first-parent line stay in the serve
#     dir's private/v3c-audit. Under H1_TIE (below and at `v3c_audit`) the window commit is the one made at H1's close,
#     its registration held to the record H1 bound and its circuit to the verifier's record reading H1's W's tables (step
#     3 writes it after the classes, `window-circuit --window-registration W`), none is made here, and the audit is tied
#     to H1's W. On a node with less memory than the audit needs, tied or not, the window commit is made (or the tie's
#     held) here and the audit and the composite are deferred to a follow-on (below), and the record here has sampled
#     proofs missing, and the split certifier missing too, its chain and period check going with the audit, so the
#     record composes pending. `served_verdicts.sh v3c` runs only that step on an
#     existing serve dir (D and its window K set), the build first, its limits 120m and 180m.
#     With the network certifier's piece split in two, CERTIFIER_CHAIN_CMD and CERTIFIER_PERIOD_CMD (both set, CERTIFIER_CMD
#     unset, and V3C_CLASS the window class of the certifier's scan class, 8,32,4,4,16 at w63k4), the certifier's answer reads V3c's
#     period.json: the chain after the window commit, beside V3c's audit and period.json on a node that holds both
#     (NODE_MEM_GB at least V3C_AUDIT_MEM_GB plus CERTIFIER_CHAIN_MEM_GB, default 96), else after them, which a
#     CERTIFIER-CHAIN line says, and always after them in a deferral's follow-on, whose node is held to the audit's
#     memory alone (after them, only once V3c wrote period.json), within
#     CERTIFIER_TIMEOUT (default 4h), and the period check once both are done and V3c wrote period.json, within
#     CERTIFIER_PERIOD_TIMEOUT (default 12h, a period's listed circuit files each read whole), each as
#     `bash -c CMD _ D K DIR V3C` in TREE with DIR $OUT/pieces/certifier
#     and V3C V3c's $D/v3c-w<K>-<class>. V3c then takes the window commit, the longer of the chain and V3c's audit
#     (their sum on a node that holds only one), and the period check, which the run's --timeout holds beside the other
#     steps. The period check writes the
#     certifier's answer.json in DIR, which step 4's certifier piece reads; the steps' logs (CERTIFIER_PRIVATE) move to the
#     serve dir's private/pieces/certifier before the run exits, `served_verdicts.sh v3c`, which writes no record, included.
#     The chain's scan proves at CR_SCAN_DELTA (chain_record.sh's), 2^-64 by default, written as the double that is
#     2^-64 exactly, which scan_zk.py keeps exact: every scan block is drawn, so no δ costs proving or leaves a wrong unit.
#     The tied V3c audit's δ stays sampled proofs' own; the network period reads that draw at its own δ_i.
# Coins: sampled proofs' sample, V3c's audit's, is on a local development challenger session's coins where the audit runs
# (`--challenger local`: the served node, or a deferral's follow-on) unless PP_CHALLENGER names the live challenger
# (gate:USER@HOST with PP_PEER, PP_CHALLENGER_KEY and PP_GATE_KEY); the verifier draws none of its own. The COINS line
# names them (`pp_coins`), and a follow-on, whose $OUT holds the record of record, says its own before its audit draws,
# from its own PP_CHALLENGER. Its session is
# pinned on the verifier's copy of main's first-parent line: PP_MAIN's, else MAIN_FILE's, else MAIN_SHA's in the tree,
# kept as the serve dir's private/v3c-audit/main.txt. The challenger's release of the draw's coins (its key, the window
# root and the coins) is kept beside the audit's case where the audit runs, $VW/audit/window/release.json (`a5.RELEASE`),
# which a5's v2 case carries and a re-proof copy carries (`onsite_record.REPROOF`); `v2_case.py` reads its coins and
# window root under the auditor's own key record, never the key it names. With H1_GATE (the
# lane's gate command, by default "ssh -i $HOME/.ssh/vllm_gate_ed25519 -o BatchMode=yes research@204.12.178.84"; empty, no
# H1) the endpoint serves PoUW H1 (--protocols pouw:eps1-sm120-h1fx1 --challenger-gate H1_GATE --challenger-peer H1_PEER,
# default vy-dc-h200, at the engine's T and S WINDOW_STEPS 16, the Program build H1 affords), each window's coins from a
# session of its own on vy-challenger-1, once `served_verdicts.py gate` finds no period of lane H1_LANE (default vllm,
# the lane on H1_GATE's key's line) open (a reserve would join it). H1_PROTOCOLS_CONFIG (a JSON object or file; unset,
# none) is serve-openai's --protocols-config, given to it and to `served_verdicts.py program` alike, so the offline
# Program is the one it declares: {"pouw": {"unit": "j10r", "ks": [2048]}} declares J10-R's tile unit over the
# linears at K 2,048 alone, the form a period's guarantee covers. SERVE_ARGS carries no --protocols-config of its own.
# PoUW's piece (POUW_CMD's default then, `served_verdicts.py pouw`) checks the window's coins against the gate's session,
# ledger and pins under H1_KEY_SHA512 (vy-challenger-1's key) and main's first-parent line: MAIN_FILE's (a file, kept as
# $OUT/main.txt; the gate's pin is a commit of `next`, so give `git rev-list --first-parent` of it), else MAIN_SHA's
# (default the tree's HEAD). That key is a development key, so the run is a development run and credits nothing. The COINS line says which.
# With H1_CIRCUITS too (a directory), before serve-openai starts (its startup reserves window 0's session, which joins
# only a period already open), `served_verdicts.py program D/h1` writes the Program serve-openai will declare for REPO at
# T and WINDOW_STEPS, `pouw_h1_circuits --program D/h1 --out H1_CIRCUITS --live T,WINDOW_STEPS --strata H1_STRATA
# [--lanes H1_LANES]` builds its circuit files (the live file, only the strata H1_STRATA lists, default none, and each K
# of H1_LANES, default none, in credit's lanes form: lane<K> and combine<K>; a file already there under the same bind is
# reused), then `served_verdicts.py period open` opens period H1_PERIOD (default served-RUN_ID) on its
# digest, its draw K H1_DRAW_K (default `pouw_h1_challenger.draw_k`'s), its weights H1_MODEL's registration (default
# none, which credit refuses), and its fixed length SLOTS requests of MAX_TOKENS (`--cap 2`: one window and its draw session);
# `period served`
# holds the Program serve-openai then declared to it, and `period close` closes it once the windows are sealed, or
# when the endpoint never listens. The PERIOD lines say what it committed; $OUT/serve keeps period.json, the Program
# (h1/window-program.json.gz and its meta) and H1_CIRCUITS' circuit-files.json. H1_STATEMENTS and
# H1_LIVE_EVIDENCE (files) go to PoUW's piece as its --statements and --live-evidence, the evidence core's credit reads.
# POUW_CMD='bash benchmarks/one_stage/served_pouw.sh "$1" "$2" "$3"' makes both on the node (the window's live rounds and
# drawn units proved) and writes the period's case beside the piece, the one remote's composite reads.
# With H1_TIE set too (and H1_CIRCUITS, V3C_CLASS not empty), V3c's window commit moves into the serve's close: at each H1
# window's close, once it is sealed and before its sample coin, serve-openai's `--h1-window-commit` runs
# `served_verdicts.sh tie` on the served node, under the serve's lease (LEASE_MIN then defaults to 150 and LEASE_MEM_GB to
# 160, the serve's 128 and window-commit's peak): on the window's advice as the server committed it (serve-openai commits
# every whole window before the tie runs, and the tie commits none: a window it finds none for is refused), `prove_path.py window-commit --window-registration W --h1-key
# PROVER/key --law V3C_LAW`, the advice read at the ℓ_a it was padded to (`--cap-bits CAP`, the run's, which the tie
# inherits), within V3C_TIMEOUT (then default 60m, at most 90m), whose registration R, chained right after
# H1's W, H1 names as the window's Commit record (`serve_tie`). Its output stays in the serve dir's h1/w<k>/. H1's sessions
# are reserved for the gate's default 120 minutes, the window's sample session at the serve's start, and the sample coin
# waits for R. window-commit makes each Program's Build itself (`own_builds`), since step 3's compile writes the row dir's
# only once the serve is done. Step 6 then audits that commit and makes none of its own: a window H1 bound no R for has no
# V3c, and so no sampled proofs' audit, and its circuit is held to step 3's record at V3C_CLASS reading W's tables, which
# gates nothing. The composite composes the cases on R: V3c's audit's, the policy's (its log the serve's
# h1/log.jsonl) and PoUW's period case once each of its windows names R (`served_cases.py`). The
# preflight refuses H1_TIE on a tree whose window-commit takes no --window-registration.
# On a node with less memory than V3c's audit needs (its MemTotal, NODE_MEM_GB, under V3C_AUDIT_MEM_GB, default 290),
# tied or not, step 6 makes the window commit (under H1_TIE, holds the tie's) (WINDOW-COMMIT, WINDOW-CIRCUIT-HELD) and
# defers the audit and the composite (`defer`), for
# PRODUCT=remote and remote-network, with PoUW or without, whose composites read neither PoUS's case nor the wipe's, and
# for onsite with COMPOSITE set empty (onsite's composite reads both cases, which only this node's card writes: the preflight refuses it).
# Step 5's record has sampled proofs missing and says what was deferred and why (its `deferred`), a DEFERRED line says
# so, and the serve dir's
# private/deferred gets the pieces the record read, the inference-only verdict (which the follow-on's record carries as
# the served node wrote it), the advice registration's public form ($OUT/registration.json, which the follow-on's
# $OUT carries for the onsite record), the settings the follow-on runs at (deferred.env: the window, the
# product, V3c's class and law, H1_TIE, COMPOSITE, ALPHA, GAMMA, the served node, the tree's commit, ONSITE, whether
# the run writes the onsite record, the split certifier's two commands, WINDOW_STEPS and MAX_TOKENS, which label
# the onsite record's figures, and OF_RECORD), and the
# follow-on's every input relative to the serve dir (`deferred_inputs`, with the split's chain the network certifier's
# kept run), listed (inputs.txt, which names inputs.sha256
# too) and hashed (inputs.sha256). The copy is the prover's: those files alone, to the same path on a node with the
# memory, outside any run dir. There `served_verdicts.sh v3c` with D alone (deferred.env names the window, and an
# environment that sets one of its settings to another value is refused) holds the copy to its sums, runs at the served
# node's tree (another only with FOLLOW_ANY_TREE=1, which its WINDOW-CIRCUIT line names, since it reads that tree's
# window-circuit record), needs no row dir, vLLM venv, card or gpu-lease, syncs the tree's venv before its preflight's
# checks run its python (`uv sync --all-packages --extra torch-cpu`, its log sync.log), makes no window commit (it holds the served
# node's, the tie's or untied step 6's, in the copy, as there), runs the split certifier's chain and period
# check after V3c's audit (their answer and case in its own $OUT/pieces/certifier), then removes the window's circuit
# files, which nothing after reads but the composite and the onsite record a v2 case's (`drop_circuits`, its
# V3C-CIRCUITS line; that one after the onsite record), then runs
# REPROOF_CMD where it is set (`bash -c CMD _ D K VW` in TREE, its log the serve dir's private/reproof.log and its last
# line the REPROOF line: a command that copies the serve dir's re-proof inputs off the node and writes REPROOF_LISTING, and fails the
# follow-on where it fails; it runs whatever V3c's audit gave, a failed audit's inputs kept as well), composes the
# final record (`record --node` the served node, so its composed_on names the
# host it composed on, served_node false, and that it read a prover-side copy; its pieces the served node's, but for the
# split certifier's, whose piece runs here as step 4 runs it, about the served node (`pieces --served`), so its answer
# here, which remote-network's record reads, is composed, and its failure here rejects) and the
# composite (`served_composite.sh`, SERVED_NODE the served node), and under ONSITE the onsite record, as the served
# node's run would (with REPROOF_COPY and REPROOF_LISTING); its $OUT is published as every run's is.
# OF_RECORD set makes the run one of record, and the preflight refuses it lacking any input its onsite record's of_record
# reads, naming each: PRODUCT=remote-network-no-pouw untied (H1_TIE empty) at V3C_LAW subset:1 (V3c's audit is C-Flock
# --zk whatever the run), since untied PoUW's period case can't name R, which step 6 makes after the serve, so a
# composite with PoUW rejects on it; the split certifier behind its front (CERTIFIER_FRONT, or on a deferral's follow-on
# its ONSITE, and CERTIFIER_CHAIN_CMD and CERTIFIER_PERIOD_CMD), whose network case that composite reads; COMPOSITE
# not set empty; and the chain's draws of record, every link-window and every proof block at the scan's δ of 2^-64:
# CR_SCAN_DELTA its default, and CERTIFIER_CHAIN_CMD naming no CR_N2_DRAW, CR_N2_CMD, CR_SCAN_DELTA, CR_SCAN_ARGS,
# CR_SCAN_CMD, --draw or --chain-cmd, since of_record reads each draw's law off the chain's report and holds one whose
# law isn't all open. It doesn't need ALPHA and GAMMA: unset, the inference-only case reads α and Γ off the division it
# checks (`--from-division`), and its policy.json says so. On the node that runs V3c's audit, the chain and the onsite
# record (the served node where nothing is deferred, else the follow-on, which runs at the deferral's OF_RECORD), also
# PP_CHALLENGER=gate:USER@HOST with PP_PEER and PP_CHALLENGER_KEY, without which V3c's coins are a local session's (or
# its audit refuses the gate) and of_record is conditional on live coins; CERTIFIER_KEY and SITE_REGISTRATION, each
# a non-empty file, without which auditor-record doesn't hold and certifier-key stays open; and the chain's own
# environment there, which deferred.env doesn't carry, drawing everything: CR_N2_DRAW, CR_N2_CMD and CR_SCAN_CMD
# unset, and no --draw in CR_SCAN_ARGS.
# POUW=pearlplus (default h1) makes Pearl+ PoUW's piece instead: H1_GATE defaults empty (set, the preflight refuses
# it), and POUW_CMD to 'bash benchmarks/one_stage/served_pearlplus.sh "$1" "$2" "$3"' within POUW_TIMEOUT 120m, which
# once the window is audited serves Pearl+ (pouw:pearl-c-h100-v1-h3, Llama 3.2 1B's BF16 checkpoint) on this card under
# its own lease, every pass retained in the serve dir's private/, and takes Pearl+'s verdict over them (verify.json in
# the piece's dir; first, by default, it gives up the passes' rows over a debit cap from the card's counts, V-EX): a development
# piece, since its passes are not the window sampled proofs audited (its header has the rest). Its epoch's salt coin is
# a live session's of its own on vy-challenger-1: PEARLC_CHALLENGER_GATE defaults to the lane's gate command (H1_GATE's
# default under h1), PEARLC_CHALLENGER_PEER to H1_PEER and PEARLC_CHALLENGER_LANE to H1_LANE's, `served_verdicts.py gate`
# must find no period of that lane open, as for H1, and the piece holds each pass's salt to the session under
# H1_KEY_SHA512 and MAIN_FILE's line, as H1's piece does. PEARLC_CHALLENGER_GATE set empty leaves Pearl+'s coins an
# in-process challenger's: the COINS line, the run's first, leads with that, and the piece claims no verdict.
# A step that can't run says why, and the chain stops there. C-Flock, Lean's flock-verify and sampled-proofs-verifier and the
# tree's venv are built while the endpoint serves, C-Flock in FLOCK_WORK (kept on the node, so a rerun reuses its build).
#   research run --on vy-dc-h200 --project verity --source TREE --cwd clone --timeout 6h -- \
#     bash benchmarks/one_stage/served_verdicts.sh
# SETTINGS=PATH names a file of NAME=VALUE lines on the node, read before every default below: the run's job record
# publishes its launch command, so a load's counts (REQUESTS, PROMPTS, SLOTS, CERTIFIER_FRONT's) go in the file, and the
# command names only its path.
# PREFLIGHT_ONLY=1 runs the preflight alone (the settings, the row dir and taps, gpu-lease, the front's flags, H1_TIE's and
# a follow-on's checks), says what the run would do (PREFLIGHT passed: its lease, or its deferral) and exits 0 before the
# serve dir is made, kept or anything built but a follow-on's venv, which its checks run.
# A run that serves is refused where the serve dir's filesystem has fewer free inodes than MIN_FREE_INODES (default
# 500000): step 6 commits the class window there, an inode a file.
# `served_verdicts.sh circuits K`, in TREE with D and OUT, counts serve dir D's window K's circuits alone (its advice written).
# `served_verdicts.sh advice-charge`, with PRODUCT=remote, measures honest serving's advice charge on a test run whose
# inputs and counts are public, never on a served run: steps 0 to 2 at the run of record's serving (T, WINDOW_STEPS, the
# front's tick), H1_GATE empty and so no H1 option in SERVE_ARGS (serve-openai refuses one without H1's --protocols),
# then every closed window's advice: the server's own where it committed each as it served (`--advice`, whose
# `--advice-tau` must be ADVICE_TAU, a PREFLIGHT refusal otherwise), else `verity-vllm advice`'s under the prove path's registered compiler at ADVICE_T
# buckets of ADVICE_TAU seconds (default 600 and 0.1; an execution phase is ADVICE_T buckets of the network certifier's
# tick) in the serve dir; and `advice_charge.py`'s record of it, each window's unpadded length and their sum over each
# execution phase, $OUT/advice-charge.json,
# which names each window and counts its requests: its inputs being public, the run publishes $OUT as it is, with no
# `publish`. Its record is what a served run's ADVICE_CHARGE names. `served_verdicts.sh charge`, in TREE with D and OUT,
# is its last step alone, on a serve dir D already served.
# `served_verdicts.sh inference-policy`, in TREE with D and OUT, writes the inference-only policy at IO_CLASS alone, to
# IO_POLICY (default $OUT/inference-only-policy.json), before a serve; `served_verdicts.sh inference-only K` judges window
# K's circuit record at IO_CLASS alone against it.
# `served_verdicts.sh block-counts K`, in TREE with D and OUT, judges window K's circuit records at CLASSES against the
# policy alone (`block_counts`): the judgment public, the exact counts in D's private/.
# `served_verdicts.sh defer WHY`, in TREE with D, K and OUT, makes step 6's deferral alone.
# Defaults are vy-dc-h200's: the int8 B4 row's traced Build (r20261030-210701-a20f's, traced after the required manifest stopped
# naming the FlashAttention M1 stream, as `predict row` doesn't) and taps under $H, vLLM's venv /workspace/cp/venv312, CUDA 13.0.
# The serve dir D (row dirs, advice, the window circuits' records) stays on the node under `research keep`; $OUT keeps the
# public records and the run's other logs, each in its public form. What a verifier reads of $OUT (`served_verdicts.RECEIVED`:
# the certifier's piece, V3c's audit, the registered compiler, the advice and weights registrations and the window circuit's
# counts) holds no log, and leaves the node, every field as written, only where `kind_scan.py --received` finds each one
# of the public kinds: else it stays whole in the serve dir's private/out and the run fails (RECEIVED-SCAN).
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
FOLLOWS="K PRODUCT V3C_CLASS V3C_LAW H1_TIE COMPOSITE ALPHA GAMMA SERVED_NODE DEFERRED_TREE CR_SCAN_DELTA ONSITE"
FOLLOWS+=" CERTIFIER_CHAIN_CMD CERTIFIER_PERIOD_CMD WINDOW_STEPS MAX_TOKENS OF_RECORD" FOLLOW= CLASH=
if [ "${1:-}" = v3c ] && [ -f "$DEFER_DIR/deferred.env" ]; then  # the follow-on of a deferral, at the settings it recorded
  FOLLOW=1
  declare -A given=()
  for v in $FOLLOWS; do [ -z "${!v+x}" ] || given[$v]=${!v}; done
  set -a
  # shellcheck source=/dev/null
  . "$DEFER_DIR/deferred.env"
  set +a
  for v in $FOLLOWS; do [ -z "${given[$v]+x}" ] || [ "${given[$v]}" = "${!v-}" ] || CLASH+=" $v"; done
  VENV=${VENV:-$TREE/.venv}
fi
H=/workspace/research/vllm-composed-h200int8
ROW=${ROW:-$H/r20261030-210701-a20f/llama32-1b__int8__h200__tp1__b4__i256__o32__mixed__greedy__bi-eager} ROW_ID=${ROW_ID:-${ROW##*/}}
HIDDEN_SO=${HIDDEN_SO:-$H/fa2/build/fa3_matReqG/verity_fa3_matReqG.so} NORM_TAP_SO=${NORM_TAP_SO:-$H/norm_tap/build/verity_norm_tap.so}
VENV=${VENV:-/workspace/cp/venv312}
REPO=${REPO:-verity-int8/Llama-3.2-1B-INT8} REV=${REV:-15b0976d400d796ee589e7737d03a919526bf800} PRECISION=${PRECISION:-verity_int8}
PORT=${PORT:-18190} SLOTS=${SLOTS:-4} PROMPTS=${PROMPTS:-16:24} MAX_TOKENS=${MAX_TOKENS:-4} RATE=${RATE:-1000} SEED=${SEED:-1}
REQUESTS=${REQUESTS:-$SLOTS} CLASSES=${CLASSES:-4,32,4} MAX_MODEL_LEN=${MAX_MODEL_LEN:-512} ID_BYTES=${ID_BYTES:-24}
T=${T:-${H1_T:-512}}
DISK_FLOOR_GB=${DISK_FLOOR_GB:-100}
H1_KEY_SHA512=${H1_KEY_SHA512:-27271eaf917d1a817198cbcfe7684debd9bdb11b9e386460948bb981642fab3ae065c0a877330a0da68ae76969550eb786dc172319876aaca21fd116e5a193bc}
POUW=${POUW:-h1}
H1_PEER=${H1_PEER:-vy-dc-h200}
LANE_GATE="ssh -i $HOME/.ssh/vllm_gate_ed25519 -o BatchMode=yes research@204.12.178.84"
if [ "$POUW" = pearlplus ]; then
  H1_GATE=${H1_GATE-}
  export PEARLC_CHALLENGER_GATE=${PEARLC_CHALLENGER_GATE-$LANE_GATE} PEARLC_CHALLENGER_PEER=${PEARLC_CHALLENGER_PEER:-$H1_PEER} \
    PEARLC_CHALLENGER_LANE=${PEARLC_CHALLENGER_LANE:-${H1_LANE:-vllm}}
  # shellcheck disable=SC2016
  export POUW_CMD=${POUW_CMD-'bash benchmarks/one_stage/served_pearlplus.sh "$1" "$2" "$3"'} POUW_TIMEOUT=${POUW_TIMEOUT:-120m}
else
  H1_GATE=${H1_GATE-$LANE_GATE}
fi
if [ -n "$H1_GATE" ]; then WINDOW_STEPS=${WINDOW_STEPS:-16}; else WINDOW_STEPS=${WINDOW_STEPS:-256}; fi
MAIN_SHA=${MAIN_SHA-$(git -C "$TREE" rev-parse HEAD 2>/dev/null)}
PRODUCT=${PRODUCT:-onsite}
if [ "$PRODUCT" = onsite ]; then
  # shellcheck disable=SC2016
  [ -f "$TREE/benchmarks/pous/streamed_wipe/served_wipe.sh" ] \
    && WIPE_CMD=${WIPE_CMD-'bash benchmarks/pous/streamed_wipe/served_wipe.sh "$1" "$2" "$3"'}
  # shellcheck disable=SC2016
  [ -f "$TREE/benchmarks/pous/served_piece.sh" ] && [ -f "$HOME/.ssh/vy-auditor2-pous" ] \
    && POUS_CMD=${POUS_CMD-'bash benchmarks/pous/served_piece.sh "$@"'}
  # shellcheck disable=SC2016
  if [ -n "${POUS_CMD:-}" ]; then
    BANDWIDTH_CMD=${BANDWIDTH_CMD-'"$PY" benchmarks/one_stage/served_bandwidth.py collect --serve-dir "$1" --window "$2" --out "$3"'}
  elif [ -f "$TREE/benchmarks/one_stage/served_bandwidth.sh" ] && [ -f "$HOME/.ssh/vy-auditor2-bw" ] \
    && [ -f "${BW_STORE:-/workspace/research/pous-store/mem69-v2}/setup.json" ]; then
    BANDWIDTH_CMD=${BANDWIDTH_CMD-'bash benchmarks/one_stage/served_bandwidth.sh "$1" "$2" "$3"'}
  fi
fi
export H1_KEY_SHA512 H1_GATE H1_PEER WINDOW_STEPS MAIN_SHA WIPE_CMD POUS_CMD BANDWIDTH_CMD PRODUCT
export CR_SCAN_DELTA=${CR_SCAN_DELTA:-5.421010862427522e-20}
PRED=${PRED:-$D/predicted}
FW=${FLOCK_WORK:-$ROOT/flock-work}
export PY=$VENV/bin/python HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export PYTHONPATH=$TREE:$TREE/integrations/vllm:$TREE/tools/research/src
export TORCH_EXTENSIONS_DIR=$ROOT/cache/torch-extensions TRITON_CACHE_DIR=$ROOT/cache/triton VLLM_CACHE_ROOT=$ROOT/cache/vllm
export VLLM_BATCH_INVARIANT=1 VERITY_RETAIN=host HIDDEN_GPU_BUILD=$ROOT/cache/jit/hidden_gpu NATIVE_COLLECT_BUILD=$ROOT/cache/jit/nc
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
    for v in WIPE_CMD POUS_CMD BANDWIDTH_CMD CERTIFIER_CMD; do
      [ -z "${!v:-}" ] || { say "PREFLIGHT: PRODUCT=remote runs no onsite piece, and $v is set: unset it"; exit 3; }
    done
    # the split certifier's period case is the onsite record's, which remote's record doesn't read: it runs behind the
    # front, here or in the follow-on of a deferral whose served node ran it
    if [ -z "${CERTIFIER_FRONT:-}" ] && { [ -z "$FOLLOW" ] || [ -z "${ONSITE:-}" ]; }; then
      for v in CERTIFIER_CHAIN_CMD CERTIFIER_PERIOD_CMD; do
        [ -z "${!v:-}" ] || { say "PREFLIGHT: PRODUCT=remote runs the split certifier's $v only with the network" \
          "certifier in front, whose period case it writes: set CERTIFIER_FRONT, or unset it"; exit 3; }
      done
    fi ;;
  remote-network | remote-network-no-pouw)
    for v in WIPE_CMD POUS_CMD BANDWIDTH_CMD; do
      [ -z "${!v:-}" ] || { say "PREFLIGHT: PRODUCT=$PRODUCT runs no onsite piece but the network certifier's, and $v is set: unset it"; exit 3; }
    done
    [ -z "${CERTIFIER_CMD:-}" ] || { say "PREFLIGHT: PRODUCT=$PRODUCT holds every compute audit of the network certifier's period to the run's registration, which only the split certifier's (V3c's) is: set CERTIFIER_CHAIN_CMD and CERTIFIER_PERIOD_CMD in place of CERTIFIER_CMD"; exit 3; } ;;
  *) say "PREFLIGHT: PRODUCT is onsite, remote, remote-network or remote-network-no-pouw, not $PRODUCT"; exit 3 ;;
esac
# ℓ_a, unless CAP gives it: the least width that pads every Commit period's advice this server commits (advice_windows.least_cap
# of window_advice.max_bits), at most SLOTS requests (--window-requests) and WINDOW_STEPS engine steps a period, positions
# within MAX_MODEL_LEN, no prefix cache, and ids of at most ID_BYTES bytes, the loads' longest (varied's varied.<j:05d>, the
# front's user-<i>, a warm-up's warmup-<8 hex>-<i>). A period past it is refused by name (advice.within_cap).
advice_cap() {
  [ -n "${CAP:-}" ] && return 0
  case " ${SERVE_ARGS:-} " in
    *" --prefix-caching "*) say "PREFLIGHT: SERVE_ARGS serves with the prefix cache, whose KV sources ℓ_a's default leaves out: set CAP"; return 3 ;;
  esac
  CAP=$("$PY" -c 'import sys; from verity_vllm.pipeline import advice_windows as AW, window_advice as WA; print(AW.least_cap(WA.max_bits(*map(int, sys.argv[1:]))))' \
    "$SLOTS" "$WINDOW_STEPS" "$MAX_MODEL_LEN" "$ID_BYTES") && [ -n "$CAP" ] \
    || { CAP=; say "PREFLIGHT: no ℓ_a from SLOTS $SLOTS, WINDOW_STEPS $WINDOW_STEPS, MAX_MODEL_LEN $MAX_MODEL_LEN and ID_BYTES $ID_BYTES (window_advice.max_bits): set CAP"; return 3; }
  export CAP
}
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
V3C=${V3C_CLASS-4,32,4} v3c_ok=0 VLOG=$D/private/v3c-audit
[ "${1:-}" = v3c ] && V3C=${V3C:-4,32,4}
[ -z "${H1_TIE:-}" ] || V3C_LAW=${V3C_LAW:-stratified:1}
vtag=${V3C//,/-}
if [ "${1:-}" = v3c ]; then V3C_TIMEOUT=${V3C_TIMEOUT:-120m} V3C_AUDIT_TIMEOUT=${V3C_AUDIT_TIMEOUT:-180m}
elif [ -n "${H1_TIE:-}" ]; then V3C_TIMEOUT=${V3C_TIMEOUT:-60m} V3C_AUDIT_TIMEOUT=${V3C_AUDIT_TIMEOUT:-30m}
else V3C_TIMEOUT=${V3C_TIMEOUT:-15m} V3C_AUDIT_TIMEOUT=${V3C_AUDIT_TIMEOUT:-30m}; fi
V3C_WARM_TIMEOUT=${V3C_WARM_TIMEOUT:-$V3C_AUDIT_TIMEOUT}
V3C_AUDIT_MEM_GB=${V3C_AUDIT_MEM_GB:-290} CERTIFIER_CHAIN_MEM_GB=${CERTIFIER_CHAIN_MEM_GB:-96}
NODE_MEM_GB=${NODE_MEM_GB:-$(awk '/^MemTotal:/ {print int($2 / 1048576)}' /proc/meminfo)}
short_of_memory() {  # 0 iff this node has less memory than V3c's audit needs, saying both
  [ "$NODE_MEM_GB" -lt "$V3C_AUDIT_MEM_GB" ] || return 1
  echo "this node has $NODE_MEM_GB GB, under the $V3C_AUDIT_MEM_GB GB V3c's audit needs (V3C_AUDIT_MEM_GB)"
}
MIN_FREE_INODES=${MIN_FREE_INODES:-500000}
short_of_inodes() {  # 0 iff the serve dir's filesystem has fewer free inodes than MIN_FREE_INODES, saying both
  local at=$D total free
  until [ -e "$at" ]; do at=$(dirname "$at"); done
  # a filesystem with no fixed inode table (btrfs, some network ones) says a total of 0 or -, and has no floor
  read -r total free < <(df --output=itotal,iavail "$at" 2>/dev/null | tail -1)
  [[ ${total:-} =~ ^[1-9][0-9]*$ && ${free:-} =~ ^[0-9]+$ ]] && [ "$free" -lt "$MIN_FREE_INODES" ] || return 1
  echo "$at's filesystem has $free free inodes, under MIN_FREE_INODES $MIN_FREE_INODES"
}
tree_sha() { git -C "$TREE" rev-parse HEAD 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-}"; }
window_commit() {  # window K of $D committed as V3C's class circuit's units in $VW/window, its answer in $1 and its errors
  # in $2, the rest window-commit's further arguments: 0, else why on stdout
  local json=$1 err=$2 rc
  shift 2
  # the advice as the server padded it, to ℓ_a (CAP, which the run exports and a tie inherits), as `compiler` reads it
  advice_cap || { echo "refused: no ℓ_a, the cap the server padded the advice to (advice_cap)"; return 3; }
  # off the GPU, which none leases here or the engine holds under the tie: the weights hashed by h1.registered's NumPy
  # path, the reference
  (cd "$TREE" && CUDA_VISIBLE_DEVICES='' timeout -k 1m "$V3C_TIMEOUT" "$PY" benchmarks/prove_path/prove_path.py window-commit --serve-run "$D" \
    --window "$K" --row-dir "$D/windows/w$K" --advice "$D/advice/w$K/private/advice.bin" --class "$V3C" \
    --key "$D/private/weights.key" --registration "$WREG" --checkpoint-dir "$CKPT" --cap-bits "$CAP" --out "$VW/window" "$@") \
    > "$json" 2> "$err"
  rc=$?
  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then echo "timed out after V3C_TIMEOUT=$V3C_TIMEOUT (exit $rc)"
  elif [ "$rc" != 0 ]; then echo "refused (exit $rc): $(grep -v '^ ' "$err" | tail -2 | cut -c1-800)"; fi
  return "$rc"
}
tied() {  # under H1_TIE: 0 iff H1 bound window K's R at its close (the serve dir's h1/log.jsonl) and R is $VW's registration
  "$PY" - "$D/h1/log.jsonl" "$K" "$VW/window/registration.json" <<'EOF' 2>&1
import json, sys
from verity.core.protocols.sampled_proofs.one_stage import registration as REG
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
  elif [ -n "$FOLLOW" ]; then  # the served node's step-6 commit, in the copy: this node has no row dir or advice
    [ -f "$json" ] || { say "WINDOW-COMMIT class $V3C: the copy holds no commit of the served node's ($(basename "$json"))"; return 1; }
    at=" on the served node $SERVED_NODE at step 6, its copy held to that node's sums"
  else
    why=$(window_commit "$json" "$OUT/window-commit-$vtag.err") || { say "WINDOW-COMMIT class $V3C $why"; return 1; }
  fi
  # the commit's population and sources count the window's own values: they stay in the serve dir
  say "WINDOW-COMMIT class $V3C$at: $("$PY" -c 'import json, sys; c = json.load(open(sys.argv[1])); print(json.dumps({k: c[k] for k in ("run", "circuit", "registration") if k in c}, sort_keys=True))' \
    "$json" 2>&1 | cut -c1-1500)"
  "$PY" - "$rec" "$VW/window/circuit.json" "$V3C" <<'EOF' 2>&1 | publish_lines
import json, sys
from verity.ml.windows import interface as IF
mine, theirs = (json.load(open(f)) for f in sys.argv[1:3])
ok = IF.circuit_of(theirs) == mine
print(f"WINDOW-CIRCUIT-HELD class {sys.argv[3]}: the committed circuit.json less its run fields is the verifier's own "
      f"window-circuit record: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
EOF
  [ "${PIPESTATUS[0]}" = 0 ] && v3c_ok=1
}
v3c_coins() {  # V3C_CH: the challenger the audit draws on (Coins, above), its session pinned on main's first-parent line
  local main=$VLOG/main.txt given=${PP_MAIN:-${MAIN_FILE:-}} name=MAIN_FILE
  [ -z "${PP_MAIN:-}" ] || name=PP_MAIN
  if [ -n "$given" ]; then
    [ -s "$given" ] || { say "V3C-AUDIT: $name $given, main's first-parent line its challenger session pins on, is missing or empty"; return 1; }
    [ "$given" -ef "$main" ] || cp "$given" "$main"
  else
    git -C "$TREE" rev-list --first-parent "${MAIN_SHA:-HEAD}" > "$main" 2>/dev/null && [ -s "$main" ] \
      || { say "V3C-AUDIT: no first-parent line of main for its challenger session to pin on (MAIN_FILE unset, and ${MAIN_SHA:-HEAD} has none in this tree): set MAIN_FILE"; return 1; }
  fi
  V3C_CH=(--challenger "${PP_CHALLENGER:-local}" --main "$main")
  case ${PP_CHALLENGER:-local} in gate:*)
    V3C_CH+=(--peer "${PP_PEER:-}" --challenger-key "${PP_CHALLENGER_KEY:-}" ${PP_GATE_KEY:+--gate-key "$PP_GATE_KEY"});;
  esac
}
# The audit's sessions listen on V3C_PORTS ports (default 16, a5's RANK_PORTS), a draw needing more refused before any
# listens: a subset:K draw opens one for each of at most K members, and records_audit's own default, a port for every
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
v3c_warm() {  # every template of the committed window composed into $VW/compose-cache, V3C_WARM_JOBS at a time
  (cd "$TREE" && env -u PYTHONPATH RAYON_NUM_THREADS="$(nproc)" RUST_MIN_STACK=$((64 << 20)) timeout -k 1m "$V3C_WARM_TIMEOUT" \
    "$TREE/.venv/bin/python" benchmarks/prove_path/records_audit.py warm --window "$VW/window" --law "${V3C_LAW:-subset:1}" \
    --compose-cache "$VW/compose-cache" --jobs "$V3C_WARM_JOBS") > "$VLOG/warm.log" 2> "$VLOG/warm.err"
  local rc=$? why
  if [ "$rc" = 0 ]; then why=$(grep -m1 '^WARM' "$VLOG/warm.log" | cut -c1-600)
  else
    if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then why="timed out after V3C_WARM_TIMEOUT=$V3C_WARM_TIMEOUT"
    else why=$(tail -3 "$VLOG/warm.err" | tr '\n' ' ' | cut -c1-800); why=${why% }; fi
    why+="; the audit composes the templates it didn't keep"
  fi
  say "V3C-WARM class $V3C (records_audit.py warm exit $rc, $V3C_WARM_JOBS at a time): $why"
}
v3c_audit() {  # the committed window audited from its records alone: its circuit and run root the verifier's, the weights
               # pinned, and no advice opening: the class window's circuit is the class's and reads no advice
  local run leanb m0 rc V3C_CH=() tie=() cache=() ports=${V3C_PORTS:-16}
  run=$("$PY" -c 'import json, sys; print(next(json.loads(l)["run_root"] for l in open(sys.argv[1]) if l.strip() and json.loads(l)["window"] == int(sys.argv[2])))' \
    "$D/windows.jsonl" "$K") || { say "V3C-AUDIT: no run root for the window in windows.jsonl"; return 1; }
  # its logs and main's line stay on the node: $OUT/v3c-audit holds only what the record reads (`kind_scan.V3C_DOCS`)
  (umask 077; mkdir -p "$VLOG")
  mkdir -p "$OUT/v3c-audit"
  v3c_coins || return 1
  if [ -n "${H1_TIE:-}" ]; then
    tie=(--tie "$D/h1/w$K/registration.json" --tie-prefix h1/)
    [ -n "${V3C_PORTS:-}" ] || ports=$("$PY" -c 'import json, sys; print(sum(int(s["k"]) for s in json.load(open(sys.argv[1]))["law"]["strata"]))' \
      "$VW/window/registration.json") || { say "V3C-AUDIT: R, the window's registration, has no strata to size the audit's ports by: set V3C_PORTS"; return 1; }
  fi
  leanb=$TREE/verity/core/service/flock/.lake/build/bin
  m0=$TREE:$TREE/benchmarks/numerical/python:$TREE/verity/ml/flock/python:$TREE/integrations/vllm
  if [ -n "${V3C_WARM_JOBS:-}" ]; then
    v3c_warm
    cache=(--compose-cache "$VW/compose-cache")
  fi
  local pool=(${V3C_LEAN_WORKERS:+--lean-workers "$V3C_LEAN_WORKERS"})
  [ -n "$FOLLOW" ] || pool=(--lean-workers "${V3C_LEAN_WORKERS:-1}" --audit-mem-gb "$V3C_AUDIT_MEM_GB")
  pool+=(${V3C_NEGATIVE_WORKERS:+--negative-workers "$V3C_NEGATIVE_WORKERS"})
  (cd "$TREE" && env -u PYTHONPATH RAYON_NUM_THREADS="$(nproc)" RUST_MIN_STACK=$((64 << 20)) timeout -k 1m "$V3C_AUDIT_TIMEOUT" \
    "$TREE/.venv/bin/python" benchmarks/prove_path/records_audit.py audit --window "$VW/window" --law "${V3C_LAW:-subset:1}" \
    --circuit "$(sha512sum "$VW/window/circuit.json" | cut -c1-128)" --run "$run" --registered "$WREG" "${tie[@]}" "${cache[@]}" \
    "${pool[@]}" \
    "${V3C_CH[@]}" --m0-pythonpath "$m0" --bin "$FW/flock-gpu-link/flock/target/release/flock-circuit" --lean "$leanb/flock-verify" \
    --firewall "$leanb/flock-firewall" --ports "$ports" --out "$VW/audit") > "$VLOG/audit.log" 2> "$VLOG/audit.err"
  rc=$?
  # what the record's row reads, which the onsite check reads from the public record: what `served_integrity.answer`
  # reads of the audit (`served_integrity.public_copy`) and the registration it audited
  "$PY" "$(dirname "$SV")/served_integrity.py" public-copy "$VW/audit" "$OUT/v3c-audit" \
    || say "V3C-AUDIT: no public copy of the audit (served_integrity.py public-copy exit $?)"
  cp "$VW/window/registration.json" "$OUT/v3c-audit/" 2>/dev/null
  say "V3C-AUDIT class $V3C (records_audit.py exit $rc): $(grep -m1 '^RECORDS' "$VLOG/audit.log" | cut -c1-3000)"
  [ -n "$(grep -m1 '^RECORDS' "$VLOG/audit.log")" ] \
    || say "V3C-AUDIT: no RECORDS line; $(tail -3 "$VLOG/audit.err" | tr '\n' ' ' | cut -c1-800)"
  return $rc
}
v3c_period() {  # V3c's audit as one compute audit of the network certifier's period: $VW/period.json
  (cd "$TREE" && env -u PYTHONPATH "$TREE/.venv/bin/python" benchmarks/prove_path/records_audit.py period --window "$VW/window" \
    --audit "$VW/audit" --pins "$WREG" --out "$VW/period.json") > "$VLOG/period.log" 2>&1
  local rc=$?
  [ "$rc" = 0 ] && cp "$VW/period.json" "$OUT/v3c-audit/"
  say "V3C-PERIOD (records_audit.py period exit $rc): $(tail -1 "$VLOG/period.log" | cut -c1-800)"
  return $rc
}
certifier_step() {  # $1 chain|period: the split network certifier's step on V3c's window (step 6), if its command is set
  local cmd lim dir=$OUT/pieces/certifier rc
  if [ "$1" = chain ]; then cmd=${CERTIFIER_CHAIN_CMD:-} lim=${CERTIFIER_TIMEOUT:-4h}
  else cmd=${CERTIFIER_PERIOD_CMD:-} lim=${CERTIFIER_PERIOD_TIMEOUT:-12h}; fi
  [ -n "$cmd" ] || return 0
  mkdir -p "$dir"
  (cd "$TREE" && timeout -k 1m "$lim" bash -c "$cmd" _ "$D" "$K" "$dir" "$VW") > "$dir/$1.log" 2>&1
  rc=$?
  say "CERTIFIER-${1^^} (exit $rc): $(tail -1 "$dir/$1.log" | cut -c1-600)"
  return $rc
}
v3c_proved() {  # V3c's window committed, the build done: the certifier's chain beside V3c's audit on a node that holds
  # both, else after it (always in a deferral's follow-on, whose node is held to the audit's memory alone), once V3c wrote
  # the period.json its period check reads; then its period check
  local rc prc chain=
  if [ -z "$FOLLOW" ] && [ "$NODE_MEM_GB" -ge $((V3C_AUDIT_MEM_GB + CERTIFIER_CHAIN_MEM_GB)) ]; then
    certifier_step chain & chain=$!
  elif [ -z "$FOLLOW" ] && [ -n "${CERTIFIER_CHAIN_CMD:-}" ]; then
    say "CERTIFIER-CHAIN after V3c's audit: this node has $NODE_MEM_GB GB, under the $V3C_AUDIT_MEM_GB GB V3c's audit needs" \
        "(V3C_AUDIT_MEM_GB) plus the $CERTIFIER_CHAIN_MEM_GB GB of the chain beside it (CERTIFIER_CHAIN_MEM_GB)"
  fi
  v3c_audit; rc=$?
  v3c_period; prc=$?
  if [ -n "$chain" ]; then wait "$chain"; elif [ "$prc" = 0 ]; then certifier_step chain; fi
  if [ "$prc" = 0 ]; then certifier_step period
  elif [ -n "${CERTIFIER_PERIOD_CMD:-}" ]; then say "CERTIFIER-PERIOD: not run: V3c wrote no period.json"; fi
  return $((rc ? rc : prc))
}
compiler() {  # the prove path's registered compiler (`prove_path.py compiler`), the one $OUT/compiler.json publishes, at ℓ_a
  advice_cap || return 3
  (cd "$TREE" && "$PY" benchmarks/prove_path/prove_path.py compiler --serve-run "$D" --cap-bits "$CAP")
}
compiler_args() {  # the registered compiler on stdin, as verity-vllm advice takes it
  "$PY" -c "import json, sys; c = json.load(sys.stdin)['inner']; print('--compiler-name', c['name'], '--compiler-digest', c['digest'])"
}
tie() {  # serve-openai's --h1-window-commit (`serve_tie`) at window $2's close, before its sample coin: $1 the serve dir, $3
  # the window's registration W, $4 where its record R goes, $5 its H1 prover dir. It says nothing in $OUT, whose lines
  # name no window: what it says goes to the serve dir's h1/w<k>/window-commit.log, and step 6 says the picked window's
  K=$2 VW=$D/v3c-w$2-$vtag
  local why
  [ "$1" -ef "$D" ] || { echo "TIE: serve-openai's serve dir $1 isn't D, $D"; return 3; }
  [ -n "$V3C" ] && [ -n "$5" ] || { echo "TIE: V3C_CLASS is empty or the window has no H1 prover dir"; return 3; }
  [ -f "$WREG" ] || { echo "TIE: $NO_WEIGHTS"; return 13; }
  # the server's committer is the window's one: serve_tie's flush committed every whole window before this ran
  [ -f "$D/advice/w$K/registration.json" ] && [ -f "$D/advice/w$K/private/advice.bin" ] || {
    echo "TIE: the server committed no advice for the window (advice.jsonl): it refused it, or a request of it runs on past its close"
    return 14; }
  why=$(window_commit "$D/h1/w$K/window-commit.json" "$D/h1/w$K/window-commit.err" --window-registration "$3" \
    --h1-key "$5/key" --law "${V3C_LAW:-subset:1}") || { echo "TIE: window-commit class $V3C $why"; return 16; }
  cp "$VW/window/registration.json" "$4" || { echo "TIE: window-commit class $V3C wrote no registration"; return 16; }
  echo "TIE: window-commit class $V3C: its registration is the window's record R"
}

serve() {  # under the lease: the endpoint, the requests, every closed window sealed, the endpoint stopped
  export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
  mkdir -p "$D" "$OUT/serve"
  cd "$TREE/integrations/vllm" || return 9
  advice_cap || return 3
  local h1=() pc=()
  [ -n "${H1_PROTOCOLS_CONFIG:-}" ] && pc=(--protocols-config "$H1_PROTOCOLS_CONFIG")
  [ -n "$H1_GATE" ] && h1=(--protocols pouw:eps1-sm120-h1fx1 --challenger-gate "$H1_GATE"
                           --challenger-peer "$H1_PEER" "${pc[@]}")
  [ -n "$H1_GATE" ] && [ -n "${H1_CIRCUITS:-}" ] && h1+=(--h1-prover "$D/private/h1")
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
    (cd "$TREE/integrations/vllm" && PYTHONPATH=$PYTHONPATH:$TREE/verity/ml/flock/python "$PY" -m tests.protocols.pouw_h1_circuits \
      --program "$D/h1" --out "$H1_CIRCUITS" --live "$T,$WINDOW_STEPS" --strata "${H1_STRATA:-none}" \
      ${H1_LANES:+--lanes "$H1_LANES"}) \
      > "$OUT/h1-circuits.log" 2>&1 || say "H1-CIRCUITS refused (exit $?): $(tail -1 "$OUT/h1-circuits.log" | cut -c1-400)"
    # shellcheck disable=SC2086
    say "$("$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" period open "$D" --gate "$H1_GATE" \
      --name "${H1_PERIOD:-served-${RESEARCH_RUN_ID:-local}}" --circuits "$H1_CIRCUITS" --slots "$SLOTS" \
      --max-tokens "$MAX_TOKENS" ${H1_DRAW_K:+--draw-k "$H1_DRAW_K"} ${H1_MODEL:+--model "$H1_MODEL"} 2>&1 | tail -1)"
    [ -f "$D/period.json" ] && period=1
  fi
  # shellcheck disable=SC2086
  "$PY" -m verity_vllm.pipeline.cli serve-openai --model "$REPO" --revision "$REV" --out "$D" --port "$PORT" \
    --precision "$PRECISION" --run-id "${RESEARCH_RUN_ID:-local}" --scheme rows-sha512 --seal \
    --max-num-batched-tokens "$T" --window-steps "$WINDOW_STEPS" "${h1[@]}" \
    --required-manifest "$PRED/manifest.json" --hidden-so "$HIDDEN_SO" --norm-tap-so "$NORM_TAP_SO" --tap-cap-mb 2048 \
    --admit release-order --window-requests "$SLOTS" --max-model-len "$MAX_MODEL_LEN" --window-gb 32 \
    --disk-floor-gb "$DISK_FLOOR_GB" --seal-hold-gb 32 --seal-warm-gb 0 \
    --advice --advice-cap "$CAP" --advice-composite "$PRODUCT" --advice-weights "$WREG" ${SERVE_ARGS:-} \
    > "$D/serve.log" 2>&1 &
  local srv=$! up=0 c=0 s=0 rc=0 release
  trap 'kill -TERM "$srv" 2>/dev/null; wait "$srv"; exit 143' TERM INT
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
    return 10
  fi
  say "SERVE $(grep -m1 'SERVE-OPENAI listening' "$D/serve.log" | cut -c1-300)"
  weights
  [ "$period" = 1 ] && say "$("$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" period served "$D" 2>&1 | tail -1)"
  if [ -n "${CERTIFIER_FRONT:-}" ]; then  # the network certifier in front of the endpoint, before the requests
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
    "$PY" "$TREE/examples/vllm/serve_load.py" --port "$PORT" --model "$REPO" "${load[@]}" --prompt-range "$PROMPTS" \
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
  # a front's exit is the certifier's verdict (CERTIFIER-FRONT, and its piece reads the session), not the load's
  { [ "$LOAD" = front ] || [ "$rc" = 0 ]; } && [ "$c" -gt 0 ] && [ "$s" -ge "$c" ]
}

circuit() {  # window K's circuit at class $1 within $2 (NAME=LIMIT), its line headed $3, with $4 `tie` reading H1's W's
  # tables of the window (window-circuit --window-registration, its tag's suffix -tie): 0 iff it counted
  local c=$1 head=$3 cls=() tag=record pins=() wp=$NO_WEIGHTS f rc tie=() meta=$OUT/serve/h1/window-program.meta.json
  [ "$c" = record ] || { cls=(--class "$c"); tag=${c//,/-}; }
  if [ "${4:-}" = tie ]; then  # W as V3c's audit reads it (`v3c_audit`'s --tie), and its Program's meta as the run publishes it
    [ -s "$meta" ] || (mkdir -p "$OUT/serve" && cd "$D" && cp --parents h1/window-program.meta.json "$OUT/serve/") \
      || { say "$head: no H1 window Program's meta (the serve dir's h1/window-program.meta.json)"; return 1; }
    tie=(--window-registration "$D/h1/w$K/registration.json" --h1-meta "$meta") tag+=-tie
  fi
  f=$D/window-circuit-w$K-$tag.json
  # the verifier reads what the run publishes alone (`kind_scan.RECEIVED`): the registered compiler, the window's advice
  # registration (the advice's length at the cap) and the weights registration, never the serve dir
  [ -s "$OUT/compiler.json" ] || compiler > "$OUT/compiler.json" 2> "$D/compiler.err" \
    || { rm -f "$OUT/compiler.json"; say "$head: no registered compiler (prove_path.py compiler)"; return 1; }
  [ -s "$OUT/registration.json" ] || "$PY" "$SV" public-registration "$D/advice/w$K/registration.json" \
    > "$OUT/registration.json" \
    || { rm -f "$OUT/registration.json"; say "$head: the advice registration has no public form"; return 1; }
  [ -f "$WREG" ] && [ ! -s "$OUT/weights_registration.json" ] && cp "$WREG" "$OUT/"
  [ -f "$WREG" ] && pins=(--pins "$OUT/weights_registration.json") wp="at the weights registration's pins"
  timeout "${2#*=}" "$PY" benchmarks/prove_path/prove_path.py window-circuit --compiler "$OUT/compiler.json" \
    --registration "$OUT/registration.json" --out "$f" "${pins[@]}" ${PIN:+--pin "$PIN"} "${cls[@]}" "${tie[@]}" \
    > "$D/window-circuit-$tag.count.json" 2> "$D/window-circuit-$tag.err"
  rc=$?
  if [ "$rc" = 0 ]; then
    cp "$D/window-circuit-$tag.count.json" "$OUT/window-circuit-$tag.json"
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
measure() {  # in TREE: the class of record, unless CLASSES gates on it; it gates nothing, and nothing after it reads it
  grep -q '"window-circuit"' benchmarks/prove_path/prove_path.py || return 0
  case " $CLASSES " in
    *" record "*) ;;
    *) circuit record "WC_RECORD_TIMEOUT=${WC_RECORD_TIMEOUT:-45m}" "WINDOW-CIRCUIT class record (measurement)";;
  esac
  return 0
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
IO_LEAN=$TREE/verity/core/protocols/network/nci/lean/.lake/build/bin/nci-check
io_class() {  # the inference-only verdict's class: IO_CLASS, else the first of CLASSES but `record`
  local c
  [ -z "${IO_CLASS:-}" ] || { echo "$IO_CLASS"; return 0; }
  for c in $CLASSES; do [ "$c" = record ] || { echo "$c"; return 0; }; done
}
inference_policy() {  # in TREE, before the lease: the policy's (c, k, n) at the class, from the run's settings alone
  local c out rc
  c=$(io_class)
  [ -n "$c" ] || { say "INFERENCE-ONLY POLICY: none, since CLASSES names no class but record"; return 0; }
  out=$(CUDA_VISIBLE_DEVICES='' "$PY" benchmarks/one_stage/inference_only.py policy --class "$c" --model "$REPO" \
    --revision "$REV" --lease-min "${LEASE_MIN:-60}" --tree "$(tree_sha)" --out "$IO_POLICY" 2> "$D/inference-only-policy.err")
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
    say "PREFLIGHT: the network certifier in front of the endpoint sends id-form requests, which serve-openai answers" \
        "only under SERVE_ARGS='--protocol ids --id-payload 503'"; return 3 ;; esac
  local help f
  help=$(cd "$TREE" && "$PY" benchmarks/network_traces/certifier_piece.py front --help 2>&1) \
    || { say "PREFLIGHT: this tree's certifier front doesn't start: $(echo "$help" | tail -1 | cut -c1-300)"; return 3; }
  for f in $(grep -oE -- '--[a-z][a-z0-9-]*' <<< "$CERTIFIER_FRONT" | sort -u); do
    grep -qE "(^|[^a-z0-9-])$f([^a-z0-9-]|$)" <<< "$help" \
      || { say "PREFLIGHT: CERTIFIER_FRONT passes $f, which this tree's certifier front doesn't take"; return 3; }
  done
}
v3c_link() {  # preflight: behind this tree's certifier front, V3C_CLASS is the one class whose window the chain's scan
  # reads at the front's scan class (`prove_path.py window-class`), the only one window-commit's scan takes after the serve
  case ${CERTIFIER_FRONT:-} in *"certifier_piece.py front"*) ;; *) return 0 ;; esac
  local scan why
  scan=$(grep -oE -- "--scan-class[= ]+['\"]?[a-z0-9]+" <<< "$CERTIFIER_FRONT" | grep -oE '[a-z0-9]+$')
  [[ $CERTIFIER_FRONT != *--scan-class* ]] || [ -n "$scan" ] \
    || { say "PREFLIGHT: CERTIFIER_FRONT passes --scan-class with no class this preflight reads: name it"; return 3; }
  why=$(cd "$TREE" && "$PY" benchmarks/prove_path/prove_path.py window-class --class "$V3C" ${scan:+--scan-class "$scan"} \
    2>&1 >/dev/null) \
    || { say "PREFLIGHT: V3C_CLASS $V3C behind the certifier front: $({ grep -m1 'prove_path.py window-class' <<< "$why" \
      || tail -1 <<< "$why"; } | cut -c1-600)"; return 3; }
}
# the certifier's chain files list and count the window's requests (its certs, its N2 warden's counts, its report and the
# piece's log, or the split's two logs); once the record has read them, and however the run ends, they move to the serve
# dir's private/
CERTIFIER_PRIVATE="chain/certs.json chain/report chain/n2/summary.json piece.log chain.log period.log"
private_pieces() {
  local f
  for f in $CERTIFIER_PRIVATE; do
    [ -e "$OUT/pieces/certifier/$f" ] || continue
    (umask 077; mkdir -p "$D/private/pieces/certifier/$(dirname "$f")") \
      && mv "$OUT/pieces/certifier/$f" "$D/private/pieces/certifier/$f"
  done
}
# What the follow-on of step 6's deferral reads, relative to the serve dir. V3c's: the weights registration, the verifier's
# window-circuit record at the class (step 3's, `v3c_record`; under H1_TIE reading H1's tables), windows.jsonl (the run
# root), H1's log and each record it names, the window commit's answer (the tie's, or untied step 6's), and the commit's
# window dir (circuit.json, registration.json, every value file). The
# record's, besides V3c's audit, which the follow-on makes: seals.jsonl, windows.jsonl and requests.jsonl (the pick's
# steps), the pieces it read, and the inference-only verdict it carries. The composite's: PoUW's period case. The
# onsite record's: the advice registration's public form. The split certifier's, where its chain is set: the network
# certifier's kept run (`certifier_piece.kept_run`: the gate's fetch, and the front's session.json, which names the
# gate), beside requests.jsonl, which its bind reads. And deferred.env, the settings the follow-on runs at. Each is
# listed where it exists.
deferred_inputs() {
  {
    printf '%s\n' weights_registration.json "$(v3c_record)" windows.jsonl seals.jsonl requests.jsonl \
      h1/log.jsonl "h1/w$K/window-commit.json" "window-commit-$vtag.json" private/pouw/period/case.json \
      private/pouw/period/answer.json \
      private/deferred/inference-only.json private/deferred/registration.json
    [ ! -f "$D/h1/log.jsonl" ] || "$SAY_PY" -c 'import json, sys
for x in open(sys.argv[1]):
    if x.strip():
        print("h1/" + str(json.loads(x).get("file")))' "$D/h1/log.jsonl"
    [ -z "${CERTIFIER_CHAIN_CMD:-}" ] || (cd "$D" && find private/certifier -mindepth 2 -maxdepth 2 -type f \
      \( -name fetch.tar.gz -o -name session.json \) 2>/dev/null)
    (cd "$D" && find "v3c-w$K-$vtag/window" private/deferred/pieces private/deferred/deferred.env -type f 2>/dev/null)
  } | (cd "$D" && while IFS= read -r f; do [ ! -f "$f" ] || echo "$f"; done) | LC_ALL=C sort -u
}
DEFERRED_WHAT="V3c's audit$([ -z "${COMPOSITE-1}" ] || echo " and the composite")"
defer() {  # step 6 on a node short of V3c's audit's memory ($1, why), left to a follow-on: the pieces the record read
  # copied to the serve dir's private/deferred, the settings it runs at (deferred.env), and the follow-on's every input
  # listed (inputs.txt, what a prover-side copy takes) and hashed (inputs.sha256, what the follow-on holds the copy to)
  (umask 077; rm -rf "$DEFER_DIR" && mkdir -p "$DEFER_DIR/pieces") || return 1
  [ ! -d "$OUT/pieces" ] || cp -a "$OUT/pieces/." "$DEFER_DIR/pieces/" || return 1
  [ ! -f "$OUT/record/inference-only.json" ] || cp "$OUT/record/inference-only.json" "$DEFER_DIR/inference-only.json" || return 1
  [ ! -f "$OUT/registration.json" ] || cp "$OUT/registration.json" "$DEFER_DIR/registration.json" || return 1
  printf '%s=%q\n' K "$K" PRODUCT "$PRODUCT" V3C_CLASS "$V3C" V3C_LAW "${V3C_LAW:-subset:1}" H1_TIE "${H1_TIE:-}" \
    COMPOSITE "${COMPOSITE-1}" ALPHA "${ALPHA:-}" GAMMA "${GAMMA:-}" SERVED_NODE "$(hostname)" DEFERRED_TREE "$(tree_sha)" \
    CR_SCAN_DELTA "$CR_SCAN_DELTA" ONSITE "$([ -z "${CERTIFIER_CMD:-}${CERTIFIER_FRONT:-}" ] || echo 1)" \
    CERTIFIER_CHAIN_CMD "${CERTIFIER_CHAIN_CMD:-}" CERTIFIER_PERIOD_CMD "${CERTIFIER_PERIOD_CMD:-}" \
    WINDOW_STEPS "$WINDOW_STEPS" MAX_TOKENS "$MAX_TOKENS" OF_RECORD "${OF_RECORD:-}" > "$DEFER_DIR/deferred.env" || return 1
  deferred_inputs > "$DEFER_DIR/inputs.txt"
  (cd "$D" && tr '\n' '\0' < "$DEFER_DIR/inputs.txt" | xargs -0 -r sha256sum) > "$DEFER_DIR/inputs.sha256" || return 1
  echo private/deferred/inputs.sha256 >> "$DEFER_DIR/inputs.txt"
  say "DEFERRED $DEFERRED_WHAT: $1. A follow-on on a node with the memory runs them and composes the final record there," \
      "from the serve dir's audit inputs copied prover-side, outside any run dir, each file its private/deferred/inputs.txt" \
      "lists: D=$D bash benchmarks/one_stage/served_verdicts.sh v3c"
}
pp_coins() {  # sampled proofs' coins, V3c's audit's, as a COINS line names them: PP_CHALLENGER's, else local on $1
  if [ "${PP_CHALLENGER:-local}" != local ]; then echo "the challenger $PP_CHALLENGER (peer ${PP_PEER:-?})"
  else echo "a local development challenger session on $1, not an auditor's (PP_CHALLENGER local)"; fi
}
follow() {  # `served_verdicts.sh v3c` on a deferral's copy of the serve dir: the copy held to the served node's sums, then
  # step 6's audit, REPROOF_CMD, the final record, the composite and the onsite record, here rather than on the served node
  local rc vrc rec comp=0 rp=0 dc=0 TPY=$TREE/.venv/bin/python pieces=$DEFER_DIR/pieces p
  (cd "$D" && sha256sum --check --strict --quiet private/deferred/inputs.sha256) > "$DEFER_DIR/check.log" 2>&1
  rc=$?
  say "FOLLOW-ON on $(hostname -s), $NODE_MEM_GB GB: $DEFERRED_WHAT, which the served node $SERVED_NODE deferred, from its serve" \
      "dir's audit inputs copied here prover-side, each as it hashed them (the serve dir's private/deferred/inputs.sha256):" \
      "$([ "$rc" = 0 ] && echo PASS || echo "FAIL (the serve dir's private/deferred/check.log)")"
  [ "$rc" = 0 ] || return 3
  [ ! -f "$DEFER_DIR/registration.json" ] || cp "$DEFER_DIR/registration.json" "$OUT/registration.json" || return 1
  wait "$BUILD"; rc=$?
  [ "$rc" = 0 ] || { tail -30 "$OUT/build.log"; stop build "$rc"; }
  local sha
  sha=$(tree_sha)
  say "WINDOW-CIRCUIT class $V3C: the served node's record at the class (step 3's), from its tree's commit $DEFERRED_TREE" \
      "$([ "$sha" = "$DEFERRED_TREE" ] && echo "(this tree's)" || echo "(this tree is $sha: FOLLOW_ANY_TREE)");" \
      "this node holds no row dir to count it from"
  say "COINS sampled proofs: $(pp_coins "$(hostname -s), the follow-on's node, where it runs V3c's audit"); the other" \
      "protocols' as the served node's COINS line names them"
  v3c_commit && v3c_proved; vrc=$?
  drop_circuits || dc=1
  if [ -n "${REPROOF_CMD:-}" ]; then
    (umask 077; cd "$TREE" && bash -c "$REPROOF_CMD" _ "$D" "$K" "$VW" > "$D/private/reproof.log" 2>&1)
    rp=$?
    say "REPROOF (exit $rp): $(tail -1 "$D/private/reproof.log" | cut -c1-600)"
  fi
  if [ -f "$DEFER_DIR/inference-only.json" ]; then
    mkdir -p "$OUT/record" && cp "$DEFER_DIR/inference-only.json" "$OUT/record/inference-only.json" \
      && say "INFERENCE-ONLY: the served node's verdict, carried into this record as it wrote it (the serve dir's private/deferred/inference-only.json)"
  fi
  if [ -n "$SPLIT" ]; then  # the split certifier ran here: the record reads its piece, run as step 4 runs it, about the
    # served node (its served.json), beside the served node's other pieces
    mkdir -p "$OUT/pieces" || return 1
    for p in "$DEFER_DIR/pieces"/*; do [ "${p##*/}" = certifier ] || cp -a "$p" "$OUT/pieces/" || return 1; done
    CERTIFIER_CMD=true "$TPY" benchmarks/one_stage/served_verdicts.py pieces --serve-run "$D" --window "$K" \
      --out "$OUT/pieces" --only certifier --served "$SERVED_NODE" --product "$PRODUCT" 2>&1 | publish_lines
    pieces=$OUT/pieces
  fi
  "$TPY" benchmarks/one_stage/served_verdicts.py record --serve-run "$D" --window "$K" --sampled-proofs "$VW/audit" \
    --registration "$VW/window/registration.json" --pieces "$pieces" --out "$OUT/record" --product "$PRODUCT" \
    --node "$SERVED_NODE" 2>&1 | publish_lines
  rec=${PIPESTATUS[0]}
  [ -z "${COMPOSITE-1}" ] || TPY=$TPY SERVED_NODE=$SERVED_NODE bash benchmarks/one_stage/served_composite.sh "$D" "$OUT" || comp=1
  [ -z "${ONSITE:-}" ] || onsite_record
  drop_circuits all || dc=1
  echo "---- verdicts"; cat "$OUT/verdicts.txt"
  [ "$vrc" = 0 ] && [ "$rec" = 0 ] && [ "$comp" = 0 ] && [ "$rp" = 0 ] && [ "$dc" = 0 ]
}
# A follow-on's window's circuit files, each composed C-Flock circuit under V3c's dir (the audit's at each template's
# first member and linked at each drawn member of it, the warm's compose cache, and where the period writes them its
# copies under circuits/), recomposable from the public registration and the templates, the audit's window/summary.json
# naming each by the SHA-512 its sessions were held to: removed once the period check has read them, before the re-proof
# copy (whose inputs name the circuit by that pin alone) and the node's next window; with $1 all, also a file the audit's
# v2 case names (window/case-v2.json's blobs), which the composite and the onsite record read first.
drop_circuits() {
  local said
  said=$("$SAY_PY" - "$VW" "${1:-}" <<'EOF'
import json, os, sys
from pathlib import Path
vw, keep = Path(sys.argv[1]), set()
if sys.argv[2] != "all" and (v2 := vw / "audit" / "window" / "case-v2.json").is_file():
    keep = {os.path.realpath(b) for b in json.loads(v2.read_text())["blobs"]}
gone = held = freed = 0
for dp, _, fs in os.walk(vw):
    for f in fs:
        p = Path(dp, f)
        if (f != "circuit.txt" and Path(dp) != vw / "circuits") or p.is_symlink() or not p.is_file():
            continue
        if os.path.realpath(p) in keep:
            held += 1
            continue
        st = p.stat()
        p.unlink()
        gone, freed = gone + 1, freed + (st.st_size if st.st_nlink == 1 else 0)
if gone or held:
    print(f"{gone} file{'s' * (gone != 1)}, {freed} bytes freed" + (f"; {held} the v2 case names, after the composite" if held else ""))
EOF
)
  local rc=$?
  [ "$rc" = 0 ] || { say "V3C-CIRCUITS: not removed (exit $rc)"; return 1; }
  [ -z "$said" ] || say "V3C-CIRCUITS removed, the window's circuit files, recomposable and named by their pins: $said"
}
onsite_record() {  # the onsite record ($OUT/record/onsite.json, the header's), which gates nothing
  local io=() sp=() c=$D/private/inference-only s=$D/private/sampled-proofs/case.json
  local pin=${VW:+$VW/audit/window/public-pin.json}
  [ ! -f "$c/case.json" ] || io=(--inference-only-case-policy "$c/policy.json")
  [ ! -f "$s" ] || sp=(--sampled-proofs-case "$s")
  [ -z "$pin" ] || [ ! -f "$pin" ] || sp+=(--public-pin "$pin")
  "$TPY" benchmarks/one_stage/onsite_record.py "$OUT" --window-steps "$WINDOW_STEPS" --max-tokens "$MAX_TOKENS" \
    ${ADVICE_CHARGE:+--advice-charge "$ADVICE_CHARGE"} \
    ${REPROOF_COPY:+--reproof-copy "$REPROOF_COPY"} ${REPROOF_LISTING:+--reproof-listing "$REPROOF_LISTING"} \
    ${CERTIFIER_KEY:+--certifier-key "$CERTIFIER_KEY"} ${SITE_REGISTRATION:+--site-registration "$SITE_REGISTRATION"} \
    "${io[@]}" "${sp[@]}" ${H1_TIE:+--h1-tie} 2>&1 | publish_lines
}

advice_charge() {  # the test run's every closed window's advice and advice_charge.py's record of it (the header's)
  local ca rc=0 adv="$D/advice-charge"
  if [ -f "$D/advice.jsonl" ]; then  # serve-openai --advice committed each window's advice as it served, and only once
    adv="$D/advice"
    say "ADVICE: the server's own, each window's committed as it served"
  else
    advice_cap || return 3
    ca=$(compiler | compiler_args)
    [ -n "$ca" ] || { say "ADVICE-CHARGE: no registered compiler (prove_path.py compiler)"; return 13; }
    # shellcheck disable=SC2086
    (cd "$TREE/integrations/vllm" && vv advice "$D" --cap "$CAP" $ca --T "${ADVICE_T:-600}" --tau "${ADVICE_TAU:-0.1}" \
      --out "$adv") > "$OUT/advice.log" 2>&1
    rc=$?
    say "ADVICE (verity-vllm advice exit $rc): $(tail -1 "$OUT/advice.log" | cut -c1-400)"
  fi
  "$PY" "$TREE/benchmarks/one_stage/advice_charge.py" "$adv" --run "${RESEARCH_RUN_ID:-local}" \
    --tau "${ADVICE_TAU:-0.1}" --T "${ADVICE_T:-600}" --to "$OUT/advice-charge.json" 2>&1 | publish_lines
  [ "${PIPESTATUS[0]}" = 0 ] && [ "$rc" = 0 ]
}

if [ "${1:-}" = serve ]; then serve; exit $?; fi
if [ "${1:-}" = tie ]; then shift; tie "$@"; exit $?; fi
if [ "${1:-}" = circuits ]; then K=${2:?served_verdicts.sh circuits WINDOW}; circuits; rc=$?; measure; exit "$rc"; fi
if [ "${1:-}" = inference-policy ]; then mkdir -p "$D" && inference_policy; exit $?; fi
if [ "${1:-}" = inference-only ]; then K=${2:?served_verdicts.sh inference-only WINDOW}; inference_only; exit $?; fi
if [ "${1:-}" = block-counts ]; then K=${2:?served_verdicts.sh block-counts WINDOW}; block_counts; exit $?; fi
if [ "${1:-}" = front-flags ]; then front_ok; exit $?; fi
if [ "${1:-}" = private-pieces ]; then private_pieces; exit $?; fi
if [ "${1:-}" = defer ]; then defer "${2:?served_verdicts.sh defer WHY}"; exit $?; fi
if [ "${1:-}" = charge ]; then advice_charge; exit $?; fi
if [ "${1:-}" = advice-charge ]; then
  [ "$PRODUCT" = remote ] && [ -z "$H1_GATE" ] \
    || { say "PREFLIGHT: advice-charge serves a test run alone: PRODUCT=remote and H1_GATE empty"; exit 3; }
  case " ${SERVE_ARGS:-} " in *" --h1-receipts "* | *" --h1-prover "*)
    say "PREFLIGHT: advice-charge serves with H1_GATE empty, and serve-openai refuses SERVE_ARGS' --h1-receipts or" \
      "--h1-prover without H1's --protocols: drop them"; exit 3 ;;
  esac
  tau=0.1 next=  # serve-openai's --advice-tau, the buckets the server commits each window's advice in (its default 0.1)
  for w in ${SERVE_ARGS:-}; do
    [ -n "$next" ] && tau=$w next=
    case $w in --advice-tau) next=1 ;; --advice-tau=*) tau=${w#*=} ;; esac
  done
  awk -v a="$tau" -v b="${ADVICE_TAU:-0.1}" 'BEGIN { exit a != b }' \
    || { say "PREFLIGHT: the server commits the advice in buckets of SERVE_ARGS' --advice-tau ($tau s), and advice_charge.py" \
      "charges it in buckets of ADVICE_TAU (${ADVICE_TAU:-0.1} s): make them equal"; exit 3; }
fi

KS=$(dirname "$SV")/../network_traces/kind_scan.py
received_scan() {  # what a verifier reads of $OUT (`served_verdicts.RECEIVED`) held to its kinds (`kind_scan.py --received`):
                   # 0 iff every field of it is one of the public kinds, its findings in the serve dir's private/
  local log=$D/private/received-scan.txt rc
  (umask 077; mkdir -p "$D/private") && "$PY" "$KS" --received "$OUT" > "$log" 2>&1
  rc=$?
  if [ "$rc" = 0 ]; then
    echo "RECEIVED-SCAN $(tail -1 "$log" | cut -c1-400)"
  else
    say "RECEIVED-SCAN failed (exit $rc, $(grep -c '^KIND-SCAN finding' "$log") findings, in the serve dir's" \
        "private/received-scan.txt): what a verifier reads of \$OUT stays whole in its private/out, and the run fails"
  fi
  return "$rc"
}
# $OUT leaves the node: once the run is done, however it ends, the certifier's chain files that count the window's
# requests move to the serve dir's private/ (CERTIFIER_PRIVATE), what a verifier reads of $OUT is held to its kinds
# (`received_scan`, which fails the run on a finding), and each of its files that names a window or counts its own is
# held in full in the serve dir's private/out, its public form in its place (`served_verdicts.py publish`; what a
# verifier reads keeps every field once the scan is clean, only its text in public form, else held whole); an
# advice-charge run's inputs are public, and its $OUT leaves as it is
[ "${1:-}" = advice-charge ] \
  || trap 'rc=$?; private_pieces; scanned=--received-scanned; received_scan || { scanned=; [ "$rc" = 0 ] && rc=1; }
           "$SAY_PY" "$SV" publish "$OUT" --private "$D/private/out" $scanned \
             || echo "PUBLISH failed (exit $?): $OUT may name a window"; exit $rc' EXIT
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
  # the checks below run the venv's python, which a fresh clone of the tree doesn't have until it syncs
  ( unset PYTHONPATH; command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv sync --all-packages --extra torch-cpu ) > "$OUT/sync.log" 2>&1 \
    || { tail -20 "$OUT/sync.log"; say "PREFLIGHT: the follow-on's venv, whose python its checks run, didn't sync (uv sync --all-packages --extra torch-cpu): its log is sync.log"; exit 3; }
else
  for f in "$ROW/manifest.json" "$ROW/build_summary.json" "$HIDDEN_SO" "$NORM_TAP_SO" "$PY"; do [ -e "$f" ] || { say "PREFLIGHT: no $f"; exit 3; }; done
  command -v gpu-lease >/dev/null || { say "PREFLIGHT: no gpu-lease on this node"; exit 3; }
fi
front_ok || exit 3
advice_cap || exit 3
[ -n "$V3C" ] \
  || { say "PREFLIGHT: sampled proofs' audit of record is V3c's (and the split certifier reads its period.json), and V3C_CLASS is empty"; exit 3; }
[ "${1:-}" = advice-charge ] || v3c_link || exit 3
case ${1:-} in v3c | advice-charge) ;; *)  # a run that serves: untied, step 6 holds V3c's commit to step 3's record
  [ -n "${H1_TIE:-}" ] || [[ " $CLASSES " == *" $V3C "* ]] \
    || { say "PREFLIGHT: untied, V3c's window commit is held to the window-circuit record step 3 writes at each class of CLASSES, and CLASSES '$CLASSES' doesn't hold V3C_CLASS $V3C: set CLASSES to hold it"; exit 3; } ;;
esac
SPLIT= DEFER=
if [ -n "${CERTIFIER_CHAIN_CMD:-}${CERTIFIER_PERIOD_CMD:-}" ]; then  # the split certifier: its period check writes its answer
  [ -z "${CERTIFIER_CMD:-}" ] || { say "PREFLIGHT: CERTIFIER_CMD or the split's CERTIFIER_CHAIN_CMD and CERTIFIER_PERIOD_CMD, not both"; exit 3; }
  for v in CERTIFIER_CHAIN_CMD CERTIFIER_PERIOD_CMD; do
    [ -n "${!v:-}" ] || { say "PREFLIGHT: the split certifier runs its chain and its period check, and $v is unset"; exit 3; }
  done
  SPLIT=1
fi
if [ -n "${V3C_WARM_JOBS:-}" ]; then  # V3c's composes made before its audit
  [[ $V3C_WARM_JOBS =~ ^[1-9][0-9]*$ ]] \
    || { say "PREFLIGHT: V3C_WARM_JOBS is how many of the window's templates warm composes at once, a positive integer, not $V3C_WARM_JOBS"; exit 3; }
  grep -q -- '"warm"' "$TREE/benchmarks/prove_path/records_audit.py" \
    || { say "PREFLIGHT: V3C_WARM_JOBS composes the window's templates before its audit with records_audit.py warm, which this tree's records_audit.py doesn't have"; exit 3; }
fi
if [ -n "${V3C_NEGATIVE_WORKERS:-}" ]; then  # V3c's audit's negatives side by side
  [[ $V3C_NEGATIVE_WORKERS =~ ^[1-9][0-9]*$ ]] \
    || { say "PREFLIGHT: V3C_NEGATIVE_WORKERS is how many of the audit's negatives run at once, a positive integer, not $V3C_NEGATIVE_WORKERS"; exit 3; }
fi
if [ -n "${H1_TIE:-}" ]; then  # V3c's audit tied to H1's W, in the serve's run and in `served_verdicts.sh v3c` alike
  grep -q -- '"--tie"' "$TREE/benchmarks/prove_path/records_audit.py" \
    || { say "PREFLIGHT: under H1_TIE V3c's audit is tied to H1's W, whose tables the circuit reads, with records_audit's --tie, which this tree's records_audit.py doesn't take"; exit 3; }
  [[ ${V3C_LAW:-} == stratified:* ]] \
    || { say "PREFLIGHT: under H1_TIE every table H1's W registers needs a sampled reader, which only stratified:K, a stratum of each table's readers, gives: V3C_LAW is ${V3C_LAW:-unset}"; exit 3; }
fi
if [ -n "${H1_TIE:-}" ] && [ "${1:-}" != v3c ]; then  # V3c's window commit at each H1 window's close, before its sample coin
  [ -n "$H1_GATE" ] && [ -n "${H1_CIRCUITS:-}" ] \
    || { say "PREFLIGHT: H1_TIE ties H1's windows, and H1_GATE or H1_CIRCUITS (whose prover state window-commit reads) is unset"; exit 3; }
  grep -q -- '"--window-registration"' "$TREE/benchmarks/prove_path/prove_path.py" \
    || { say "PREFLIGHT: H1_TIE needs window-commit's --window-registration, its registration chained right after H1's, which this tree's prove_path.py doesn't take"; exit 3; }
  [[ $V3C_TIMEOUT =~ ^[0-9]+m$ ]] && [ "${V3C_TIMEOUT%m}" -le 90 ] \
    || { say "PREFLIGHT: under H1_TIE the window's sample coin waits for window-commit, and its session's reserve lasts the gate's 120 minutes from the serve's start: V3C_TIMEOUT is at most 90m, not $V3C_TIMEOUT"; exit 3; }
  LEASE_MIN=${LEASE_MIN:-150} LEASE_MEM_GB=${LEASE_MEM_GB:-160}
fi
case ${1:-} in v3c | advice-charge) ;; *)  # a run that serves and reaches step 6, tied or not
  if DEFER=$(short_of_memory); then  # step 6's audit and the composite go to a follow-on
    [ "$PRODUCT" != onsite ] || [ -z "${COMPOSITE-1}" ] \
      || { say "PREFLIGHT: where $DEFER, its audit and the composite are deferred to a follow-on, and onsite's composite reads PoUS's and the wipe's cases, which only this node's card writes: PRODUCT=remote, remote-network or remote-network-no-pouw, COMPOSITE set empty, or a node with the memory"; exit 3; }
  fi
  # step 6 commits the class window after the whole serve, so out of inodes the run would fail only there
  why=$(short_of_inodes) \
    && { say "PREFLIGHT: the serve commits its class window into the serve dir, an inode a file, and $why: remove old serve dirs' class windows (their v3c-w* dirs, once each serve dir has its copy and its run its custody), or serve on a node with the inodes"; exit 3; } ;;
esac
OF=
if [ -n "${OF_RECORD:-}" ] && [ "${1:-}" != advice-charge ] && { [ "${1:-}" != v3c ] || [ -n "$FOLLOW" ]; }; then
  of=()  # a run of record (the header's OF_RECORD): each input its onsite record's of_record reads that it lacks
  [ "$PRODUCT" = remote-network-no-pouw ] && [ -z "${H1_TIE:-}" ] && [ "${V3C_LAW:-subset:1}" = subset:1 ] \
    || of+=("PRODUCT=remote-network-no-pouw untied (H1_TIE empty) at V3C_LAW subset:1, not PRODUCT=$PRODUCT${H1_TIE:+ under H1_TIE} at ${V3C_LAW:-subset:1}")
  { [ -n "${CERTIFIER_FRONT:-}" ] || { [ -n "$FOLLOW" ] && [ -n "${ONSITE:-}" ]; }; } && [ -n "$SPLIT" ] \
    || of+=("the split certifier behind its front (CERTIFIER_FRONT, CERTIFIER_CHAIN_CMD and CERTIFIER_PERIOD_CMD), whose network case the composite reads")
  [ -n "${COMPOSITE-1}" ] || of+=("COMPOSITE unset, not empty: the composite is the record's verdict")
  draws='CR_N2_DRAW|CR_N2_CMD|CR_SCAN_DELTA|CR_SCAN_ARGS|CR_SCAN_CMD|--draw|--chain-cmd'
  [ "${CR_SCAN_DELTA:-}" = 5.421010862427522e-20 ] && ! [[ ${CERTIFIER_CHAIN_CMD:-} =~ $draws ]] \
    || of+=("CR_SCAN_DELTA its default 2^-64, and CERTIFIER_CHAIN_CMD naming none of ${draws//|/, }, the chain's draws of record, without which a chain draw can be open (of_record then conditional on live coins) or the scan's bound at another δ")
  if [ -z "$DEFER" ]; then  # this node runs V3c's audit, the certifier's chain and the onsite record
    [[ ${PP_CHALLENGER:-} == gate:?*@?* ]] && [ -n "${PP_PEER:-}" ] && [ -s "${PP_CHALLENGER_KEY:-}" ] \
      || of+=("PP_CHALLENGER=gate:USER@HOST with PP_PEER and PP_CHALLENGER_KEY (a non-empty file), the live challenger V3c's audit draws on, without which of_record is conditional on live coins")
    [ -s "${CERTIFIER_KEY:-}" ] && [ -s "${SITE_REGISTRATION:-}" ] \
      || of+=("CERTIFIER_KEY and SITE_REGISTRATION, each a non-empty file (the auditor's record of its network certifier and its registration), without which certifier-key stays open")
    [ -z "${CR_N2_DRAW:-}${CR_N2_CMD:-}${CR_SCAN_CMD:-}" ] && [[ ${CR_SCAN_ARGS:-} != *--draw* ]] \
      || of+=("CR_N2_DRAW, CR_N2_CMD and CR_SCAN_CMD unset and no --draw in CR_SCAN_ARGS where the chain runs, without which a chain draw can be open and of_record conditional on live coins")
  fi
  if [ ${#of[@]} -gt 0 ]; then
    printf -v why '%s; ' "${of[@]}"
    say "PREFLIGHT: OF_RECORD makes this run one of record, and it lacks what its onsite record's of_record reads: ${why%; }"
    exit 3
  fi
  OF=1
fi
if [ -n "${PREFLIGHT_ONLY:-}" ]; then  # the checks above alone: nothing kept, built or served
  if [ -n "$FOLLOW" ]; then plan="the follow-on of $SERVED_NODE's deferral, its venv synced for the checks"
  else
    plan="the serve's lease ${LEASE_MIN:-60} min, capped at ${LEASE_MEM_GB:-128} GB"
    [ -z "$DEFER" ] || plan+="; $DEFER, so step 6 defers $DEFERRED_WHAT to a follow-on"
  fi
  [ -z "$OF" ] || plan+="; of record (OF_RECORD): every input its of_record reads is set$([ -z "$DEFER" ] || echo ", but the follow-on's, which its own preflight checks")"
  say "PREFLIGHT passed on $(hostname -s), $NODE_MEM_GB GB: $plan. PREFLIGHT_ONLY: nothing kept, built or served"
  exit 0
fi
mkdir -p "$D" "$ROOT/cache" "$FW"
(umask 077; mkdir -p "$D/private")
KEEP_WHY="served_verdicts.sh's serve dirs (sealed row dirs, advice, window circuits) and C-Flock's build"
{ "$PY" -m research keep "$ROOT" --owner @vllm --expires 7d --priority high --why "$KEEP_WHY" \
  || research keep "$ROOT" --owner @vllm --expires 7d --priority high --why "$KEEP_WHY"; } 2>&1 | tail -2
# H1's every window open runs its Lean verifier of record; without lake each is refused and serving goes on uncredited
if [ -z "$FOLLOW" ] && { [ -n "$H1_GATE" ] || [[ " ${SERVE_ARGS:-} " == *" --protocols "* ]]; }; then
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --default-toolchain none > "$OUT/elan.log" 2>&1
  (cd "$TREE/verity/core/protocols/pouw/lean/check" && lake build pouw-check-difftest) > "$OUT/h1-lean.log" 2>&1 \
    || { tail -20 "$OUT/h1-lean.log"; say "PREFLIGHT: H1's Lean verifier (pouw-check-difftest) didn't build"; exit 3; }
fi

build() {  # what V3c's audit runs (prove_path/run.sh's builds), here while the endpoint serves
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
  uv sync --all-packages --extra torch-cpu || return 5
  (cd verity/core/service/flock && lake build flock-verify flock-firewall) || return 7
  (cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) || return 7
  mkdir -p "$OUT/m0-build"
  # FW is the node's, every run's: two builds in it at once break each other's crates
  RESEARCH_RUN_DIR=$OUT/m0-build FLOCK_WORK=$FW MODE=build GPU=0 flock "$FW/.build.lock" bash verity/ml/flock/pod/60-circuit.sh || return 6
  # after flock-verify's build, whose package it requires by path: the inference-only verdict's `nci-check --count`,
  # without which the verdict is the Python count's alone
  (cd verity/core/protocols/network/nci/lean && lake build nci-check) || echo "nci-check didn't build"
}
if [ "${1:-}" != advice-charge ]; then
  ( unset PYTHONPATH; build ) > "$OUT/build.log" 2>&1 &
  BUILD=$!
fi

if [ "${1:-}" = v3c ]; then  # an existing serve dir's window: D and K set, its advice written (step 3); V3c alone
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
  private_pieces
  exit "$rc"
fi

H1_COINS="PoUW H1: not served (H1_GATE empty)"
PEARLPLUS="PoUW: Pearl+'s own served run on this card after the window's (POUW=pearlplus)"
LIVE_GATE=$H1_GATE LIVE_LANE=${H1_LANE:-vllm} LEAD=
if [ "$POUW" = pearlplus ] && [ -n "$PEARLC_CHALLENGER_GATE" ]; then
  LIVE_GATE=$PEARLC_CHALLENGER_GATE LIVE_LANE=$PEARLC_CHALLENGER_LANE
  H1_COINS="$PEARLPLUS, its epoch's salt coin a live session's of its own on vy-challenger-1's gate (peer $PEARLC_CHALLENGER_PEER, lane $PEARLC_CHALLENGER_LANE), each pass's salt checked by core_check.live_salt in its piece, and its replay's draws the Lean verifier's IO.getRandomBytes on this node, taken once before the replay and cited by store id in the POUW line's coin; a development piece under the lane's development key, it credits nothing"
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
  H1_COINS="PoUW H1: each window's coins from a session of its own on vy-challenger-1's gate (peer $H1_PEER, T $T, S $WINDOW_STEPS), a development run under the lane's development key: it credits nothing"
  export POUW_CMD=${POUW_CMD:-"$TREE/.venv/bin/python benchmarks/one_stage/served_verdicts.py pouw \$1 \$2 \$3${H1_STATEMENTS:+ --statements $H1_STATEMENTS}${H1_LIVE_EVIDENCE:+ --live-evidence $H1_LIVE_EVIDENCE}"}
fi
if [ -n "$DEFER" ]; then COINS=$(pp_coins "the follow-on's node, where step 6 defers V3c's audit")
else COINS=$(pp_coins "$(hostname -s), the served node"); fi
say "COINS ${LEAD}sampled proofs: $COINS; $H1_COINS; the bandwidth audit's: vy-auditor-2's bw_party, inside PoUS's piece"

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
inference_policy

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
b_of_record = (str((pred / "manifest.json").resolve()), json.loads((pred / "predicted.json").read_text())["program_digest"])
serve = json.loads((d / "serve.json").read_text())
seals = [json.loads(x) for x in (d / "seals.jsonl").read_text().splitlines() if x.strip()] if (d / "seals.jsonl").exists() else []
sealed_ok = bound_ok = bool(seals)
lines, why, committers = [], set(), set()
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
                 f"{str(bound.get('program_digest'))[:16]}…: {whose}: {'PASS' if b else 'FAIL'}")
ok = sealed_ok and bound_ok and serve.get("scheme") == "rows-sha512"
os.umask(0o077)
held.write_text("".join(x + "\n" for x in lines))
print(f"SEALED: every closed window's row dir committed under rows-sha512 (committer {', '.join(sorted(committers)) or 'none'}) "
      f"and its replay bundle opened: {'PASS' if sealed_ok else 'FAIL'}")
print(f"B-OF-RECORD: every sealed record binds predict row's Build of record ({b_of_record[0]}, program digest "
      f"{b_of_record[1][:16]}…): {'PASS' if bound_ok else 'FAIL'}")
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
# the serve dir's step trace and the load's records stay on the node; what the stock control's table reads is their aggregate.
# A front's load is the network certifier's, paced by its buckets, which no stock run's is, and one window's aggregate is
# its timing.
if [ "$LOAD" = front ]; then
  say "OVERHEAD: none with LOAD=front (the network certifier paces its requests, so no stock control compares)"
else
  "$PY" "$TREE/benchmarks/one_stage/served_overhead.py" metrics "$D" > "$OUT/serve/overhead.json" 2> "$D/overhead.err" \
    || { say "OVERHEAD: no aggregate: $(cut -c1-300 "$D/overhead.err")"; rm -f "$OUT/serve/overhead.json"; }
fi
if [ "${1:-}" = advice-charge ]; then advice_charge; exit $?; fi

# 3. sampled proofs: the window, its advice and Programs, V3b's window circuit, then C-Flock --zk and the Lean verifier
# the pick (its number, steps, requests and users, and each window refused and why) stays in the serve dir's private/
PICK=$("$PY" benchmarks/one_stage/served_verdicts.py window "$D" --slots "$SLOTS" ${WINDOW:+--window "$WINDOW"} 2> "$D/private/window.err")
[ -n "$PICK" ] || { say "WINDOW refused: no sealed window qualifies (why, by window, in the serve dir's private/window.err)"; stop window 12; }
(umask 077; echo "$PICK" > "$D/private/window.json")
K=$(echo "$PICK" | "$PY" -c 'import json, sys; print(json.load(sys.stdin)["window"])')
say "WINDOW: $([ -n "${WINDOW:-}" ] && echo "the one WINDOW names" || echo "the sealed one with the most requests"), none carried across windows, at the class's $SLOTS slots (the pick in the serve dir's private/window.json)"
say "CLASSES: the window circuit's ${CLASSES}; the certifier's scan ${CERTIFIER_SCAN:-W63}, the one class the gate serves"
COINS_AFTER=$(date +%s.%N)  # sampled proofs' and the network certifier's coins are all drawn after this
CERT=
if [ -n "${CERTIFIER_CMD:-}" ]; then  # the certifier's chain is the longest piece, so it starts once the window is picked
  # kept off CPUs 0 and 1 (CERT_CPUS), which PoUS's answers need idle: the chain's load stretches a round past Δ + RTT
  CERT_TASKSET=(); NCPU=$(nproc)
  command -v taskset > /dev/null && [ "$NCPU" -gt 2 ] && CERT_TASKSET=(taskset -c "${CERT_CPUS:-2-$((NCPU - 1))}")
  ${CERT_TASKSET[@]+"${CERT_TASKSET[@]}"} "$PY" benchmarks/one_stage/served_verdicts.py pieces --serve-run "$D" --window "$K" --out "$OUT/pieces" \
    --product "$PRODUCT" --only certifier > "$OUT/pieces-certifier.txt" 2>&1 &
  CERT=$!
fi
compiler > "$OUT/compiler.json" || { rm -f "$OUT/compiler.json"; stop compiler 13; }
(cd "$TREE" && "$PY" benchmarks/one_stage/served_verdicts.py advice-line "$D" --window "$K" --compiler "$OUT/compiler.json" \
  --coins-after "$COINS_AFTER") 2>&1 | publish_lines
[ "${PIPESTATUS[0]}" = 0 ] || stop advice 14
cd "$TREE/integrations/vllm" || exit 9
# the compile log names the request Programs by their prompt lengths and counts the window's requests, which are values:
# it stays in the serve dir
"$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" public-registration "$D/advice/w$K/registration.json" > "$OUT/registration.json" \
  || { say "ADVICE: its registration has no public form"; stop advice 14; }
"$PY" tools/advice_compile.py "$D" --window "$K" > "$D/compile_w$K.log" 2>&1
rc=$?
say "ADVICE: $("$PY" -c 'import json, sys; w = json.load(open(sys.argv[1]))["windows"][sys.argv[2]]; print("root", w["root"][:16] + "…", "bits padded to the cap")' \
  "$D/advice/windows.json" "$K" 2>&1 | cut -c1-200); $("$PY" -c 'import json, sys; print(len(json.load(open(sys.argv[1])).get("registered_roots") or {}))' \
  "$D/advice/w$K/registration.json" 2>&1 | cut -c1-200) weight roots recorded ($([ -f "$WREG" ] && echo "--advice-weights $WREG" || echo "$NO_WEIGHTS")); compile (advice_compile.py) exit $rc$([ "$rc" = 0 ] || echo ", its log in the serve dir, on the node")"
[ "$rc" = 0 ] || stop compile 15
cd "$TREE" || exit 9
wc_ok=0
circuits && wc_ok=1
block_counts
wait "$BUILD"; rc=$?
inference_only &  # beside sampled proofs, which it doesn't read; the record waits for it
IOV=$!
[ "$rc" = 0 ] || { tail -30 "$OUT/build.log"; wait "$IOV"; stop build "$rc"; }

# 6. here: V3c's audit is sampled proofs' of record, and the split certifier's period check reads its period.json (its
# piece then reads the answer that check wrote). On a node short of the audit's memory (DEFER) the window commit is made
# (under H1_TIE, the tie's held) here, and the audit and the composite are deferred to a follow-on (after the record)
if [ -n "$DEFER" ]; then v3c_commit
else v3c_commit && v3c_proved; fi
[ -z "$SPLIT" ] || export CERTIFIER_CMD=true

# 4. the other subprotocols, each its piece's verdict or MISSING with the command it lacks; the split certifier's, where
# step 6 defers V3c's audit, missing, its chain and period check the follow-on's
TPY=$TREE/.venv/bin/python
"$TPY" benchmarks/one_stage/served_verdicts.py pieces --serve-run "$D" --window "$K" --out "$OUT/pieces" ${CERT:+--skip certifier} \
  ${SPLIT:+${DEFER:+--deferred "its chain and period check, which read V3c's period.json, go with V3c's audit to a follow-on on a node with the memory, which composes the final record: $DEFER"}} \
  --product "$PRODUCT" 2>&1 | publish_lines
[ -z "$CERT" ] || { wait "$CERT"; publish_lines < "$OUT/pieces-certifier.txt"; }

# 5. one record of every verdict, sampled proofs' V3c's audit on the window commit's registration (missing, said why,
# where the audit is deferred)
wait "$IOV"
"$TPY" benchmarks/one_stage/served_verdicts.py record --serve-run "$D" --window "$K" --sampled-proofs "$VW/audit" \
  --registration "$VW/window/registration.json" --pieces "$OUT/pieces" --out "$OUT/record" --product "$PRODUCT" \
  ${DEFER:+--deferred "$DEFERRED_WHAT, to a follow-on on a node with the memory, which composes the final record: $DEFER"} \
  2>&1 | publish_lines
rec=${PIPESTATUS[0]}
private_pieces
left=0
[ -z "$DEFER" ] || defer "$DEFER" || left=1
comp=0
[ -n "$DEFER" ] || [ -z "${COMPOSITE-1}" ] || TPY=$TPY bash benchmarks/one_stage/served_composite.sh "$D" "$OUT" || comp=1
[ -z "${CERTIFIER_CMD:-}${CERTIFIER_FRONT:-}" ] || onsite_record
measure
echo "---- verdicts"; cat "$OUT/verdicts.txt"
[ "$wc_ok" = 1 ] && [ "$rec" = 0 ] && [ "$comp" = 0 ] && [ "$left" = 0 ] \
  && grep -q '"composed": "\(pending\|accept\)"' "$OUT/verdicts.txt"
