#!/usr/bin/env bash
# The proof system's challenger on its auditor host (cwd = the source tree): `serve.py`, one session a run, its record,
# key, release log, its signed close and address in $RESEARCH_RUN_DIR/out, which the run publishes from this machine. It runs on a
# machine the prover doesn't control (AGENTS.md's auditor-trusted devices), never on PEER: once it listens it opens
# `ssh -R PORT` to PEER (a name in /etc/vy/peers), as benchmarks/poub_session/run.sh's verifier does, so the prover's
# client (`remote.RemoteChallenger.connect("127.0.0.1:PORT")`) connects to PEER's own 127.0.0.1:PORT. The auditor host
# runs no such script: there a session is `vy-challenger-gate reserve ...` then `open --session S --root R` (auditor.py), as the auditor user
# from the pinned tree, recorded in its ledger. This script runs a session elsewhere, for development, from the source tree:
#
#   RESEARCH_RUN_DIR=DIR PEER=NODE PORT=7811 SCHEME=vllm/commit SESSION=s-20261020-1 \
#     FAMILIES='[["opening-positions",8],["opening-identities",8],["replay",8],["kernel-check",8]]' \
#     bash benchmarks/live_challenger/run.sh
#
# Run it, as poub_session's verifier, on a node that lets a run hold no check slot: a session waits on its client for the
# whole Commit, and a node under slot_only.py stops a run that holds none. SESSION is the session's id, the one the auditor's signed ledger opens
# for its window (the key signs each id once). FAMILIES is the session's coin layout, [[label, count], ...] (the vLLM
# Commit's is `challenge.layout(runs)`); EPOCH its epoch (default 0); KEY the challenger's LMS key directory on this
# machine (default ~/.local/state/live-challenger/key, made on first use over KEYGEN_WORKERS cores, never under out/);
# PARAMS=W,H an LMS width and height other than lms.DEPLOYED's; HOST_NAME the host the record names (default this
# machine's name in /etc/vy/peers, by its address); PY the Python (default python3: the challenger reads only the
# standard library and this tree). Its coins are the Lean verifier of record's (`one_stage.verifier.coins`), built with
# this machine's lake (~/.elan/bin) on first use. The verifier reads out/record.json, out/releases.jsonl and
# out/challenger.json (the release log's signed close) from this run's out/ (then `remote.trusted_session` with the
# auditor's signed ledger, its windows.jsonl), never from the prover's.
set -euo pipefail
O=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$O"
command -v lake >/dev/null || PATH="$HOME/.elan/bin:$PATH"
for ip in $(hostname -I); do
  HOST_NAME=${HOST_NAME:-$(awk -v ip="$ip" '{n = split($NF, a, "@")} a[n] == ip {print $1; exit}' /etc/vy/peers || true)}
done
exec "${PY:-python3}" benchmarks/live_challenger/serve.py --out "$O" --port "${PORT:?}" --scheme "${SCHEME:?}" \
  --families "${FAMILIES:?}" --session "${SESSION:?}" --epoch "${EPOCH:-0}" --peer "${PEER:?}" --host "${HOST_NAME:-$(hostname)}" \
  ${KEY:+--key "$KEY"} ${KEYGEN_WORKERS:+--keygen-workers "$KEYGEN_WORKERS"} ${PARAMS:+--params "$PARAMS"} \
  2> >(tee "$O/serve.log" >&2)
