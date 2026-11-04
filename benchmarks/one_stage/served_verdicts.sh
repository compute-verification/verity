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
#  4. PoUW (H1 credit), PoUS, the bandwidth audit and the network certifier: `served_verdicts.py pieces` runs each one's piece
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
#     PRODUCT (default onsite) is the product the run composes: onsite every subprotocol; remote sampled proofs and PoUW
#     alone, its other pieces dropped (not run, no default command, the wipe's and the certifier's included; the
#     preflight refuses any onsite command set), its record composing their verdicts alone and its composite remote's.
#  6. V3c, after the record since V2 doesn't read it: it proves the window circuit's own units at V3C_CLASS (default 4,32,4,
#     which CLASSES must hold). `prove_path.py window-commit`, within V3C_TIMEOUT (default 15m), commits the window as that
#     class circuit's units, the weights it serves under the committer's key, so each served as registered has the
#     registration's root; its circuit.json less its run fields is held to the verifier's own window-circuit record at the
#     class; then `records_audit.py audit --circuit --registered`, within V3C_AUDIT_TIMEOUT (default 30m), samples the
#     registration's members (V3C_LAW, default subset:1, stratified:1 under H1_TIE) on a local development challenger
#     session's coins (--challenger local, pinned on main's first-parent line: MAIN_FILE's, kept as
#     $OUT/v3c-audit/main.txt, else MAIN_SHA's) and proves them, its composes made before it with V3C_WARM_JOBS (at
#     `v3c_audit`), its sessions one at a time and Lean's checks of finished members beside them, V3C_LEAN_WORKERS at a
#     time: on a node it shares default 1, within V3C_AUDIT_MEM_GB less the prover's session's reserve
#     (`a5.PROVER_RESERVE_GB`), a follow-on's default a5's LEAN_WORKERS, 3, within what its node has free less that
#     reserve (`a5.LeanPool`); then
#     `records_audit.py period` writes the audit as the network certifier's period reads it ($D/v3c-w<K>-<class>/period.json,
#     and a copy in $OUT/v3c-audit: the registration, the Program, its answer and its free inputs), and V3c fails where it
#     writes none. Under H1_TIE (below and at `v3c_audit`) the window commit is the one made at H1's close, its
#     registration held to the record H1 bound, none is made here, and the audit is tied to H1's W;
#     V3c then runs before step 5's composite, whose one-stage and policy cases are its audit's, unless this node has less
#     memory than the audit needs (below: the audit and the composite are then deferred to a follow-on).
#     `served_verdicts.sh v3c` runs only that step on an
#     existing serve dir (D and its window K set), the build first, its limits 120m and 180m. V3C_CLASS set empty leaves
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
# Coins: sampled proofs' sample is on a local development challenger session's coins on the served node (run.sh's
# PP_CHALLENGER default, local, pinned on PP_MAIN's line, else MAIN_FILE's, else origin's main's) unless PP_CHALLENGER
# names the live challenger (run.sh's gate:USER@HOST with PP_PEER, PP_CHALLENGER_KEY and PP_GATE_KEY); V3c's audit's is on
# a local development session of its own where it runs (step 6). With H1_GATE (the
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
# digest, its sample K H1_DRAW_K (default `pouw_h1_challenger.draw_k`'s), its weights H1_MODEL's registration (default
# none, which credit refuses), and its fixed length SLOTS requests of MAX_TOKENS (`--cap 2`: one window and its sample session);
# `period served`
# holds the Program serve-openai then declared to it, and `period close` closes it once the windows are sealed, or
# when the endpoint never listens. The PERIOD lines say what it committed; $OUT/serve keeps period.json, the Program
# (h1/window-program.json.gz and its meta) and H1_CIRCUITS' circuit-files.json. H1_STATEMENTS and
# H1_LIVE_EVIDENCE (files) go to PoUW's piece as its --statements and --live-evidence, the evidence core's credit reads.
# POUW_CMD='bash benchmarks/one_stage/served_pouw.sh "$1" "$2" "$3"' makes both on the node (the window's live rounds and
# sampled proof blocks proved) and writes the period's case beside the piece, the one remote's composite reads.
# With H1_TIE set too (and H1_CIRCUITS, V3C_CLASS not empty), V3c's window commit moves into the serve's close: at each H1
# window's close, once it is sealed and before its sample coin, serve-openai's `--h1-window-commit` runs
# `served_verdicts.sh tie` on the served node, under the serve's lease (LEASE_MIN then defaults to 150 and LEASE_MEM_GB to
# 160, the serve's 128 and window-commit's peak): the window's advice, the one step 3 then reads, and `prove_path.py window-commit --window-registration W --h1-key
# PROVER/key --law V3C_LAW` within V3C_TIMEOUT (then default 60m, at most 90m), whose registration R, chained right after
# H1's W, H1 names as the window's Commit record (`serve_tie`). Its output stays in the serve dir's h1/w<k>/. H1's sessions
# are reserved for the gate's default 120 minutes, the window's sample session at the serve's start, and the sample coin
# waits for R. Step 6 then audits that commit and makes none of its own: a window H1 bound no R for has no V3c. It runs
# before the composite, which hands no product SERVE_DIR/private/pouw's period case, untied to R (`served_composite.sh`).
# The preflight refuses H1_TIE on a tree whose window-commit takes no --window-registration.
# Under H1_TIE on a node with less memory than V3c's audit needs (its MemTotal, NODE_MEM_GB, under V3C_AUDIT_MEM_GB, default
# 290), step 6 holds the tie (WINDOW-COMMIT, WINDOW-CIRCUIT-HELD) and defers the audit and the composite (`defer`), for
# PRODUCT=remote alone (onsite's composite reads PoUS's case, which needs this node's card: the preflight refuses it).
# Step 5's record says what was deferred and why (its `deferred`), a DEFERRED line says so, and the serve dir's
# private/deferred gets the pieces the record read, the settings the follow-on runs at (deferred.env: the window, the
# product, V3c's class and law, H1_TIE, COMPOSITE, ALPHA, GAMMA, the served node and the tree's commit), and the
# follow-on's every input relative to the serve dir (`deferred_inputs`), listed (inputs.txt, which names inputs.sha256
# too) and hashed (inputs.sha256). The copy is the prover's: those files alone, to the same path on a node with the
# memory, outside any run dir. There `served_verdicts.sh v3c` with D alone (deferred.env names the window, and an
# environment that sets one of its settings to another value is refused) holds the copy to its sums, runs at the served
# node's tree (another only with FOLLOW_ANY_TREE=1, which its WINDOW-CIRCUIT line names, since it reads that tree's
# window-circuit record), needs no row dir, vLLM venv, card or gpu-lease, and after V3c's audit composes the final record
# (`record --node` the served node, so its composed_on names the host it composed on, served_node false, and that it read
# a prover-side copy) and the
# composite (`served_composite.sh`, SERVED_NODE the served node); its $OUT is published as every run's is.
# A step that can't run says why, and the chain stops there. Flock, Lean's flock-verify and sampled-proofs-verifier and the
# tree's venv are built while the endpoint serves, Flock in FLOCK_WORK (kept on the node, so a rerun reuses its build).
#   research run --on vy-dc-h200 --project verity --source TREE --cwd clone --timeout 6h -- \
#     bash benchmarks/one_stage/served_verdicts.sh
# SETTINGS=PATH names a file of NAME=VALUE lines on the node, read before every default below: the run's job record
# publishes its launch command, so a load's counts (REQUESTS, PROMPTS, SLOTS, CERTIFIER_FRONT's) go in the file, and the
# command names only its path.
# PREFLIGHT_ONLY=1 runs the preflight alone (the settings, the row dir and taps, gpu-lease, the front's flags, H1_TIE's and
# a follow-on's checks), says what the run would do (PREFLIGHT passed: its lease, or its deferral) and exits 0 before the
# serve dir is made, kept or anything built.
# `served_verdicts.sh circuits K`, in TREE with D and OUT, counts serve dir D's window K's circuits alone (its advice written).
# `served_verdicts.sh defer WHY`, in TREE with D, K and OUT, makes step 6's deferral alone.
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
FOLLOWS="K PRODUCT V3C_CLASS V3C_LAW H1_TIE COMPOSITE ALPHA GAMMA SERVED_NODE DEFERRED_TREE" FOLLOW= CLASH=
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
REQUESTS=${REQUESTS:-$SLOTS} CLASSES=${CLASSES:-4,32,4} CAP=${CAP:-8388608} MAX_MODEL_LEN=${MAX_MODEL_LEN:-512}
T=${T:-${H1_T:-512}}
DISK_FLOOR_GB=${DISK_FLOOR_GB:-100}
H1_KEY_SHA512=${H1_KEY_SHA512:-27271eaf917d1a817198cbcfe7684debd9bdb11b9e386460948bb981642fab3ae065c0a877330a0da68ae76969550eb786dc172319876aaca21fd116e5a193bc}
H1_GATE=${H1_GATE-ssh -i $HOME/.ssh/vllm_gate_ed25519 -o BatchMode=yes research@204.12.178.84} H1_PEER=${H1_PEER:-vy-dc-h200}
if [ -n "$H1_GATE" ]; then WINDOW_STEPS=${WINDOW_STEPS:-16}; else WINDOW_STEPS=${WINDOW_STEPS:-256}; fi
MAIN_SHA=${MAIN_SHA-$(git -C "$TREE" rev-parse HEAD 2>/dev/null)}
PRODUCT=${PRODUCT:-onsite}
if [ -n "${ONSITE_PIECES:-}" ]; then
  case $ONSITE_PIECES in /*) ;; *) ONSITE_PIECES=$TREE/$ONSITE_PIECES ;; esac
  [ -f "$ONSITE_PIECES" ] || { echo "PREFLIGHT: ONSITE_PIECES=$ONSITE_PIECES is not a file" >&2; exit 3; }
  # shellcheck source=/dev/null
  . "$ONSITE_PIECES"
  [ "$PRODUCT" != onsite ] || onsite_piece_commands
fi
export H1_KEY_SHA512 H1_GATE H1_PEER WINDOW_STEPS MAIN_SHA WIPE_CMD POUS_CMD BANDWIDTH_CMD PRODUCT ONSITE_PIECES
PRED=${PRED:-$D/predicted}
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
case $PRODUCT in
  onsite) ;;
  remote)
    for v in WIPE_CMD POUS_CMD BANDWIDTH_CMD CERTIFIER_CMD CERTIFIER_CHAIN_CMD CERTIFIER_PERIOD_CMD CERTIFIER_FRONT; do
      [ -z "${!v:-}" ] || { say "PREFLIGHT: PRODUCT=remote runs no onsite piece, and $v is set: unset it"; exit 3; }
    done ;;
  *) say "PREFLIGHT: PRODUCT is onsite or remote, not $PRODUCT"; exit 3 ;;
esac
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
[ -z "${H1_TIE:-}" ] || V3C_LAW=${V3C_LAW:-stratified:1}
vtag=${V3C//,/-}
if [ "${1:-}" = v3c ]; then V3C_TIMEOUT=${V3C_TIMEOUT:-120m} V3C_AUDIT_TIMEOUT=${V3C_AUDIT_TIMEOUT:-180m}
elif [ -n "${H1_TIE:-}" ]; then V3C_TIMEOUT=${V3C_TIMEOUT:-60m} V3C_AUDIT_TIMEOUT=${V3C_AUDIT_TIMEOUT:-30m}
else V3C_TIMEOUT=${V3C_TIMEOUT:-15m} V3C_AUDIT_TIMEOUT=${V3C_AUDIT_TIMEOUT:-30m}; fi
V3C_WARM_TIMEOUT=${V3C_WARM_TIMEOUT:-$V3C_AUDIT_TIMEOUT}
V3C_AUDIT_MEM_GB=${V3C_AUDIT_MEM_GB:-290}
NODE_MEM_GB=${NODE_MEM_GB:-$(awk '/^MemTotal:/ {print int($2 / 1048576)}' /proc/meminfo)}
short_of_memory() {  # 0 iff this node has less memory than V3c's audit needs, saying both
  [ "$NODE_MEM_GB" -lt "$V3C_AUDIT_MEM_GB" ] || return 1
  echo "this node has $NODE_MEM_GB GB, under the $V3C_AUDIT_MEM_GB GB V3c's audit needs (V3C_AUDIT_MEM_GB)"
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
v3c_commit() {  # window K of $D committed as V3C's class circuit's units, its circuit held to the verifier's window-circuit record
  VW=$D/v3c-w$K-$vtag
  local json=$D/window-commit-$vtag.json why at=
  [ -f "$D/window-circuit-w$K-$vtag.json" ] || { say "WINDOW-COMMIT class $V3C: no window-circuit record at it (CLASSES '$CLASSES')"; return 1; }
  [ -f "$WREG" ] || { say "WINDOW-COMMIT class $V3C: $NO_WEIGHTS"; return 1; }
  if [ -n "${H1_TIE:-}" ]; then  # committed at H1's close (`tie`) and never again: the window's one commit, the one H1 bound
    json=$D/h1/w$K/window-commit.json
    why=$(tied) || { say "WINDOW-COMMIT class $V3C: $why"; return 1; }
    at=" at H1's close, its registration the Commit record H1 bound before the sample coin ($why)"
  else
    why=$(window_commit "$json" "$OUT/window-commit-$vtag.err") || { say "WINDOW-COMMIT class $V3C $why"; return 1; }
  fi
  # the commit's population and sources count the window's own values: they stay in the serve dir
  say "WINDOW-COMMIT class $V3C$at: $("$PY" -c 'import json, sys; c = json.load(open(sys.argv[1])); print(json.dumps({k: c[k] for k in ("run", "circuit", "registration") if k in c}, sort_keys=True))' \
    "$json" 2>&1 | cut -c1-1500)"
  "$PY" - "$D/window-circuit-w$K-$vtag.json" "$VW/window/circuit.json" "$V3C" <<'EOF' 2>&1 | publish_lines
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
  local run leanb m0 rc tie=() cache=() ports=${V3C_PORTS:-16} main=$OUT/v3c-audit/main.txt
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
  if [ -n "${H1_TIE:-}" ]; then
    tie=(--tie "$D/h1/w$K/registration.json" --tie-prefix h1/)
    [ -n "${V3C_PORTS:-}" ] || ports=$("$PY" -c 'import json, sys; print(sum(int(s["k"]) for s in json.load(open(sys.argv[1]))["law"]["strata"]))' \
      "$VW/window/registration.json") || { say "V3C-AUDIT: R, the window's registration, has no strata to size the audit's ports by: set V3C_PORTS"; return 1; }
  fi
  leanb=$TREE/verity/protocols/flock/verifier/.lake/build/bin
  m0=$TREE:$TREE/benchmarks/numerical/python:$TREE/verity/protocols/flock/prover/python:$TREE/adapters/vllm
  if [ -n "${V3C_WARM_JOBS:-}" ]; then
    v3c_warm
    cache=(--compose-cache "${V3C_COMPOSE_CACHE:-$VW/compose-cache}")
  fi
  local pool=(${V3C_LEAN_WORKERS:+--lean-workers "$V3C_LEAN_WORKERS"})
  [ -n "$FOLLOW" ] || pool=(--lean-workers "${V3C_LEAN_WORKERS:-1}" --audit-mem-gb "$V3C_AUDIT_MEM_GB")
  (cd "$TREE" && env -u PYTHONPATH RAYON_NUM_THREADS="$(nproc)" RUST_MIN_STACK=$((64 << 20)) timeout -k 1m "$V3C_AUDIT_TIMEOUT" \
    "$TREE/.venv/bin/python" benchmarks/prove_path/records_audit.py audit --window "$VW/window" --law "${V3C_LAW:-subset:1}" \
    --circuit "$(sha512sum "$VW/window/circuit.json" | cut -c1-128)" --run "$run" --registered "$WREG" "${tie[@]}" "${cache[@]}" \
    "${pool[@]}" \
    --challenger local --main "$main" --m0-pythonpath "$m0" --bin "$FW/flock-gpu-link/flock/target/release/flock-circuit" --lean "$leanb/flock-verify" \
    --firewall "$leanb/flock-firewall" --ports "$ports" --out "$VW/audit") > "$OUT/v3c-audit/audit.log" 2> "$OUT/v3c-audit/audit.err"
  rc=$?
  cp "$VW/audit/summary.json" "$OUT/v3c-audit/" 2>/dev/null
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

serve() {  # under the lease: the endpoint, the requests, every closed window sealed, the endpoint stopped
  export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
  mkdir -p "$D" "$OUT/serve"
  cd "$TREE/adapters/vllm" || return 9
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
  # shellcheck disable=SC2086
  "$PY" -m verity_vllm.pipeline.cli serve-openai --model "$REPO" --revision "$REV" --out "$D" --port "$PORT" \
    --precision "$PRECISION" --run-id "${RESEARCH_RUN_ID:-local}" --scheme rows-sha512 --seal \
    --max-num-batched-tokens "$T" --window-steps "$WINDOW_STEPS" "${h1[@]}" \
    --required-manifest "$PRED/manifest.json" --hidden-so "$HIDDEN_SO" --norm-tap-so "$NORM_TAP_SO" --tap-cap-mb 2048 \
    --admit release-order --window-requests "$SLOTS" --max-model-len "$MAX_MODEL_LEN" --window-gb 32 \
    --disk-floor-gb "$DISK_FLOOR_GB" --seal-hold-gb 32 --seal-warm-gb 0 ${SERVE_ARGS:-} \
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
  # a front's exit is the certifier's verdict (CERTIFIER-FRONT, and its piece reads the session), not the load's
  { [ "$LOAD" = front ] || [ "$rc" = 0 ]; } && [ "$c" -gt 0 ] && [ "$s" -ge "$c" ]
}

circuit() {  # window K's circuit at class $1 within $2 (NAME=LIMIT), its line headed $3: 0 iff it counted
  local c=$1 head=$3 cls=() tag=record pins=() wp=$NO_WEIGHTS f rc
  [ "$c" = record ] || { cls=(--class "$c"); tag=${c//,/-}; }
  [ -f "$WREG" ] && pins=(--pins "$WREG") wp="at the weights registration's pins"
  f=$D/window-circuit-w$K-$tag.json
  timeout "${2#*=}" "$PY" benchmarks/prove_path/prove_path.py window-circuit --serve-run "$D" --window "$K" \
    --row-dir "$D/windows/w$K" --advice "$D/advice/w$K/private/advice.bin" --out "$f" "${pins[@]}" ${PIN:+--pin "$PIN"} \
    "${cls[@]}" > "$D/window-circuit-$tag.count.json" 2> "$D/window-circuit-$tag.err"
  rc=$?
  if [ "$rc" = 0 ]; then  # the count's `bits` is the advice's length, which stays on the node
    "$PY" -c 'import json, sys; c = json.load(open(sys.argv[1])); c.pop("bits", None); print(json.dumps(c, sort_keys=True))' \
      "$D/window-circuit-$tag.count.json" > "$OUT/window-circuit-$tag.json"
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
circuits() {  # in TREE: every class of CLASSES gates (0 iff each counted)
  local bad=0 c
  grep -q '"window-circuit"' benchmarks/prove_path/prove_path.py \
    || { say "MISSING V3b's window circuit: this tree's prove_path.py has no window-circuit"; return 1; }
  for c in $CLASSES; do circuit "$c" "WC_TIMEOUT=${WC_TIMEOUT:-30m}" "WINDOW-CIRCUIT class $c" || bad=1; done
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
# window-circuit record at the class (step 3's), windows.jsonl (the run root), H1's log and each record it names, the
# tie's window-commit answer, and the commit's window dir (circuit.json, registration.json, every value file). The
# record's: seals.jsonl, windows.jsonl and requests.jsonl (the pick's steps), step 3's records audit's summary, audit
# record, registration and runs, and the pieces it read. The composite's: PoUW's period case. And deferred.env, the
# settings the follow-on runs at. Each is listed where it exists.
deferred_inputs() {
  {
    printf '%s\n' weights_registration.json "window-circuit-w$K-$vtag.json" windows.jsonl seals.jsonl requests.jsonl \
      h1/log.jsonl "h1/w$K/window-commit.json" sampled-proofs-served/summary.json sampled-proofs-served/window/audit.json \
      sampled-proofs-served/windows/rank0/registration.json sampled-proofs-served/runs.json private/pouw/period/case.json \
      private/pouw/period/answer.json
    [ ! -f "$D/h1/log.jsonl" ] || "$SAY_PY" -c 'import json, sys
for x in open(sys.argv[1]):
    if x.strip():
        print("h1/" + str(json.loads(x).get("file")))' "$D/h1/log.jsonl"
    (cd "$D" && find "v3c-w$K-$vtag/window" private/deferred/pieces private/deferred/deferred.env -type f 2>/dev/null)
  } | (cd "$D" && while IFS= read -r f; do [ ! -f "$f" ] || echo "$f"; done) | LC_ALL=C sort -u
}
DEFERRED_WHAT="V3c's audit$([ -z "${COMPOSITE-1}" ] || echo " and the composite")"
defer() {  # step 6 on a node short of V3c's audit's memory ($1, why), left to a follow-on: the pieces the record read
  # copied to the serve dir's private/deferred, the settings it runs at (deferred.env), and the follow-on's every input
  # listed (inputs.txt, what a prover-side copy takes) and hashed (inputs.sha256, what the follow-on holds the copy to)
  (umask 077; rm -rf "$DEFER_DIR" && mkdir -p "$DEFER_DIR/pieces") || return 1
  [ ! -d "$OUT/pieces" ] || cp -a "$OUT/pieces/." "$DEFER_DIR/pieces/" || return 1
  printf '%s=%q\n' K "$K" PRODUCT "$PRODUCT" V3C_CLASS "$V3C" V3C_LAW "${V3C_LAW:-subset:1}" H1_TIE "${H1_TIE:-}" \
    COMPOSITE "${COMPOSITE-1}" ALPHA "${ALPHA:-}" GAMMA "${GAMMA:-}" SERVED_NODE "$(hostname)" DEFERRED_TREE "$(tree_sha)" \
    > "$DEFER_DIR/deferred.env" || return 1
  deferred_inputs > "$DEFER_DIR/inputs.txt"
  (cd "$D" && tr '\n' '\0' < "$DEFER_DIR/inputs.txt" | xargs -0 -r sha256sum) > "$DEFER_DIR/inputs.sha256" || return 1
  echo private/deferred/inputs.sha256 >> "$DEFER_DIR/inputs.txt"
  say "DEFERRED $DEFERRED_WHAT: $1. A follow-on on a node with the memory runs them and composes the final record there," \
      "from the serve dir's audit inputs copied prover-side, outside any run dir, each file its private/deferred/inputs.txt" \
      "lists: D=$D bash benchmarks/one_stage/served_verdicts.sh v3c"
}
follow() {  # `served_verdicts.sh v3c` on a deferral's copy of the serve dir: the copy held to the served node's sums, then
  # step 6's audit, the final record and the composite, here rather than on the served node
  local rc vrc rec comp=0 TPY=$TREE/.venv/bin/python
  (cd "$D" && sha256sum --check --strict --quiet private/deferred/inputs.sha256) > "$DEFER_DIR/check.log" 2>&1
  rc=$?
  say "FOLLOW-ON on $(hostname -s), $NODE_MEM_GB GB: $DEFERRED_WHAT, which the served node $SERVED_NODE deferred, from its serve" \
      "dir's audit inputs copied here prover-side, each as it hashed them (the serve dir's private/deferred/inputs.sha256):" \
      "$([ "$rc" = 0 ] && echo PASS || echo "FAIL (the serve dir's private/deferred/check.log)")"
  [ "$rc" = 0 ] || return 3
  wait "$BUILD"; rc=$?
  [ "$rc" = 0 ] || { tail -30 "$OUT/build.log"; stop build "$rc"; }
  local sha
  sha=$(tree_sha)
  say "WINDOW-CIRCUIT class $V3C: the served node's record at the class (step 3's), from its tree's commit $DEFERRED_TREE" \
      "$([ "$sha" = "$DEFERRED_TREE" ] && echo "(this tree's)" || echo "(this tree is $sha: FOLLOW_ANY_TREE)");" \
      "this node holds no row dir to count it from"
  v3c_commit && v3c_proved; vrc=$?
  "$TPY" benchmarks/one_stage/served_verdicts.py record --serve-run "$D" --window "$K" --sampled-proofs "$D/sampled-proofs-served" \
    --pieces "$DEFER_DIR/pieces" --out "$OUT/record" --product "$PRODUCT" --node "$SERVED_NODE" 2>&1 | publish_lines
  rec=${PIPESTATUS[0]}
  [ -z "${COMPOSITE-1}" ] || TPY=$TPY SERVED_NODE=$SERVED_NODE bash benchmarks/one_stage/served_composite.sh "$D" "$OUT" || comp=1
  echo "---- verdicts"; cat "$OUT/verdicts.txt"
  [ "$vrc" = 0 ] && [ "$rec" = 0 ] && [ "$comp" = 0 ]
}

if [ "${1:-}" = serve ]; then serve; exit $?; fi
if [ "${1:-}" = tie ]; then shift; tie "$@"; exit $?; fi
if [ "${1:-}" = circuits ]; then K=${2:?served_verdicts.sh circuits WINDOW}; circuits; exit $?; fi
if [ "${1:-}" = front-flags ]; then front_ok; exit $?; fi
if [ "${1:-}" = private-pieces ]; then private_pieces; exit $?; fi
if [ "${1:-}" = defer ]; then defer "${2:?served_verdicts.sh defer WHY}"; exit $?; fi

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
else
  for f in "$ROW/manifest.json" "$ROW/build_summary.json" "$HIDDEN_SO" "$NORM_TAP_SO" "$PY"; do [ -e "$f" ] || { say "PREFLIGHT: no $f"; exit 3; }; done
  command -v gpu-lease >/dev/null || { say "PREFLIGHT: no gpu-lease on this node"; exit 3; }
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
if [ -n "${H1_TIE:-}" ]; then  # V3c's audit tied to H1's W, in the serve's run and in `served_verdicts.sh v3c` alike
  grep -q -- '"--tie"' "$TREE/benchmarks/prove_path/records_audit.py" \
    || { say "PREFLIGHT: under H1_TIE V3c's audit is tied to H1's W, whose tables the circuit reads, with records_audit's --tie, which this tree's records_audit.py doesn't take"; exit 3; }
  [[ ${V3C_LAW:-} == stratified:* ]] \
    || { say "PREFLIGHT: under H1_TIE every table H1's W registers needs a sampled reader, which only stratified:K, a stratum of each table's readers, gives: V3C_LAW is ${V3C_LAW:-unset}"; exit 3; }
fi
if [ -n "${H1_TIE:-}" ] && [ "${1:-}" != v3c ]; then  # V3c's window commit at each H1 window's close, before its sample coin
  [ -n "$H1_GATE" ] && [ -n "${H1_CIRCUITS:-}" ] \
    || { say "PREFLIGHT: H1_TIE ties H1's windows, and H1_GATE or H1_CIRCUITS (whose prover state window-commit reads) is unset"; exit 3; }
  [ -n "$V3C" ] || { say "PREFLIGHT: H1_TIE commits each H1 window at V3C_CLASS, and it is empty"; exit 3; }
  grep -q -- '"--window-registration"' "$TREE/benchmarks/prove_path/prove_path.py" \
    || { say "PREFLIGHT: H1_TIE needs window-commit's --window-registration, its registration chained right after H1's, which this tree's prove_path.py doesn't take"; exit 3; }
  [[ $V3C_TIMEOUT =~ ^[0-9]+m$ ]] && [ "${V3C_TIMEOUT%m}" -le 90 ] \
    || { say "PREFLIGHT: under H1_TIE the window's sample coin waits for window-commit, and its session's reserve lasts the gate's 120 minutes from the serve's start: V3C_TIMEOUT is at most 90m, not $V3C_TIMEOUT"; exit 3; }
  LEASE_MIN=${LEASE_MIN:-150} LEASE_MEM_GB=${LEASE_MEM_GB:-160}
  if DEFER=$(short_of_memory); then  # step 6's audit and the composite go to a follow-on
    [ "$PRODUCT" = remote ] || { say "PREFLIGHT: under H1_TIE, where $DEFER, its audit and the composite are deferred to a follow-on, and onsite's composite reads PoUS's case, which needs this node's card: PRODUCT=remote, or a node with the memory"; exit 3; }
  fi
fi
if [ -n "${PREFLIGHT_ONLY:-}" ]; then  # the checks above alone: nothing kept, built or served
  if [ -n "$FOLLOW" ]; then plan="the follow-on of $SERVED_NODE's deferral"
  else
    plan="the serve's lease ${LEASE_MIN:-60} min, capped at ${LEASE_MEM_GB:-128} GB"
    [ -z "$DEFER" ] || plan+="; $DEFER, so step 6 defers V3c's audit and the composite to a follow-on"
  fi
  say "PREFLIGHT passed on $(hostname -s), $NODE_MEM_GB GB: $plan. PREFLIGHT_ONLY: nothing kept, built or served"
  exit 0
fi
mkdir -p "$D" "$ROOT/cache" "$FW"
(umask 077; mkdir -p "$D/private")
[ -z "${KEEP_CMD:-}" ] || (cd "$TREE" && bash -c "$KEEP_CMD" _ "$ROOT") 2>&1 | tail -2
# H1's every window open runs its Lean verifier of record; without lake each is refused and serving goes on uncredited
if [ -z "$FOLLOW" ] && { [ -n "$H1_GATE" ] || [[ " ${SERVE_ARGS:-} " == *" --protocols "* ]]; }; then
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
}
( unset PYTHONPATH; build ) > "$OUT/build.log" 2>&1 &
BUILD=$!

if [ "${1:-}" = v3c ]; then  # an existing serve dir's window: D and K set, its advice written (step 3); V3c alone
  if [ -n "$FOLLOW" ]; then follow; exit $?; fi
  [ -n "${K:-}" ] && [ -f "$D/advice/w$K/private/advice.bin" ] || { say "V3C: set D and K, a serve dir and its window with advice"; exit 3; }
  [ -f "$WREG" ] || CUDA_VISIBLE_DEVICES='' weights
  [ -f "$WREG" ] || exit 13
  # the verifier's record at the class from this tree, whatever an earlier run left in D
  CUDA_VISIBLE_DEVICES='' circuit "$V3C" "WC_TIMEOUT=${WC_TIMEOUT:-30m}" "WINDOW-CIRCUIT class $V3C" || exit 12
  v3c_commit
  wait "$BUILD"; rc=$?
  [ "$rc" = 0 ] || { tail -30 "$OUT/build.log"; stop build "$rc"; }
  [ "$v3c_ok" = 1 ] || exit 16
  v3c_proved; rc=$?
  private_pieces
  exit "$rc"
fi

H1_COINS="PoUW H1: not served (H1_GATE empty)"
if [ -n "$H1_GATE" ]; then
  if [ -n "${MAIN_FILE:-}" ]; then
    [ -s "$MAIN_FILE" ] || { say "PREFLIGHT: MAIN_FILE $MAIN_FILE is missing or empty"; exit 3; }
    [ "$MAIN_FILE" -ef "$OUT/main.txt" ] || cp "$MAIN_FILE" "$OUT/main.txt"
    export MAIN_FILE=$OUT/main.txt
  elif [ -n "$MAIN_SHA" ]; then
    git -C "$TREE" rev-list --first-parent "$MAIN_SHA" > "$OUT/main.txt" || { say "PREFLIGHT: MAIN_SHA $MAIN_SHA not in this tree"; exit 3; }
    export MAIN_FILE=$OUT/main.txt
  fi
  "$PY" benchmarks/one_stage/served_verdicts.py gate --gate "$H1_GATE" --lane "${H1_LANE:-vllm}" ${MAIN_FILE:+--main "$MAIN_FILE"} 2>&1 \
    | publish_lines
  [ "${PIPESTATUS[0]}" = 0 ] || stop gate 3
  H1_COINS="PoUW H1: each window's coins from a session of its own on vy-challenger-1's gate (peer $H1_PEER, T $T, S $WINDOW_STEPS), a development run under the lane's development key: it credits nothing"
  export POUW_CMD=${POUW_CMD:-"$TREE/.venv/bin/python benchmarks/one_stage/served_verdicts.py pouw \$1 \$2 \$3${H1_STATEMENTS:+ --statements $H1_STATEMENTS}${H1_LIVE_EVIDENCE:+ --live-evidence $H1_LIVE_EVIDENCE}"}
fi
if [ "${PP_CHALLENGER:-local}" != local ]; then
  COINS="the challenger $PP_CHALLENGER (peer ${PP_PEER:-?})"
else
  COINS="a local development challenger session on $(hostname -s), the served node, not an auditor's (PP_CHALLENGER local)"
fi
COINS+="; $H1_COINS;"
[ -z "$V3C" ] || COINS+=" V3c's audit: a local development challenger session where it runs;"
say "COINS sampled proofs: $COINS the bandwidth audit's: vy-auditor-2's bw_party, inside PoUS's piece"

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
[ "$rc" = 0 ] || { tail -30 "$OUT/build.log"; stop build "$rc"; }
[ "$MAX_TOKENS" = 1 ] && export PP_PARTITION=${PP_PARTITION:-benchmarks/prove_path/llama32_1b_silu_mul_v2_i8192_l0_step0.json}
env -u PYTHONPATH PP_SERVE_RUN="$D" PP_WINDOW="$K" PP_SERVE_ROWS="$D/windows/w$K" PP_RECORDS=1 PP_ONLY="${PP_ONLY:-honest}" \
  PP_MAIN="${PP_MAIN:-${MAIN_FILE:-}}" PP_VLLM_PYTHON="$PY" PP_FLOCK_WORK="$FW" OUT="$OUT/sampled-proofs" bash benchmarks/prove_path/run.sh > "$D/sampled-proofs.log" 2>&1
rc=$?
SP=$D/sampled-proofs-served
rm -rf "$SP"; [ ! -d "$OUT/sampled-proofs/served" ] || mv "$OUT/sampled-proofs/served" "$SP"
# run.sh's records copy holds the window's request lists (windows.jsonl), its request count and prefill tokens
# (serve.json) and the advice's length (registration.json): the published copy is the registration's public form alone
SR=$OUT/sampled-proofs/served-records
rm -f "$SR/windows.jsonl" "$SR/serve.json"
if [ -f "$SR/registration.json" ]; then
  "$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" public-registration "$SR/registration.json" > "$SR/registration.public.json" \
    && mv "$SR/registration.public.json" "$SR/registration.json" || rm -f "$SR/registration.json" "$SR/registration.public.json"
fi
say "SAMPLED-PROOFS (run.sh exit $rc): $("$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" sampled-line "$D/sampled-proofs.log" \
  --partition "${PP_PARTITION:-benchmarks/prove_path/llama32_1b_silu_mul_v2_i8192_l0_step1.json}" 2>&1 | cut -c1-3000)$(
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

# 5. one record of every verdict
"$TPY" benchmarks/one_stage/served_verdicts.py record --serve-run "$D" --window "$K" --sampled-proofs "$SP" \
  --pieces "$OUT/pieces" --out "$OUT/record" --product "$PRODUCT" \
  ${DEFER:+--deferred "$DEFERRED_WHAT, to a follow-on on a node with the memory, which composes the final record: $DEFER"} \
  2>&1 | publish_lines
rec=${PIPESTATUS[0]}
private_pieces
# 6. V3c after the record, each step within its limit: it gates nothing. Under H1_TIE it comes before the composite,
# whose one-stage and policy cases are its audit's of the record H1 bound; on a node short of the audit's memory, the
# tie held here, the audit and the composite are deferred to a follow-on.
v3c_done=$SPLIT left=0
if [ -n "$DEFER" ]; then v3c_commit; defer "$DEFER" || left=1; v3c_done=1
elif [ -z "$v3c_done" ] && [ -n "${H1_TIE:-}" ]; then v3c_commit && v3c_proved; v3c_done=1; fi
comp=0
[ -n "$DEFER" ] || [ -z "${COMPOSITE-1}" ] || TPY=$TPY bash benchmarks/one_stage/served_composite.sh "$D" "$OUT" || comp=1
if [ -n "$v3c_done" ]; then :
elif [ -n "$V3C" ]; then v3c_commit && v3c_proved
else say "V3C: not in this run (V3C_CLASS empty): \`served_verdicts.sh v3c\` runs it on the serve dir"; fi
echo "---- verdicts"; cat "$OUT/verdicts.txt"
[ "$wc_ok" = 1 ] && [ "$rec" = 0 ] && [ "$comp" = 0 ] && [ "$left" = 0 ] \
  && grep -q '"composed": "\(pending\|accept\)"' "$OUT/verdicts.txt"
