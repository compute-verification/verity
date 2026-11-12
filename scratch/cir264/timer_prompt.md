CIR-264, circuits' worker ACCEPT (Cursor agent bc-34e1d652-7d7b-5da5-be43-93e2aca08e95; the lead is circuits' coordinator, bc-21f3d3de-28db-5e5d-964b-d357187449aa). The one-off timer `cir264-capture-0320z` has fired; it was set for about Fri 13 Nov 2026 03:20Z, 25 minutes before your card hours. You are probably on a NEW machine, in a new clone on `main`: /tmp, untracked files, local branches and fetched runs are gone. Origin and this conversation are all you have. Do this in order.

THE JOB: one capture of vLLM d9105ea80's sampled accept kernels at 1 + 5 rows on node A (`vy-spot-glm53`), card 0, cores 0-9, inside Fri 13 Nov 03:45:00Z to 04:25:00Z, under one lease of at most 25 minutes. Before 03:45Z card 0 is CIR-263's. Touch no other card and no other core. Off the card by 04:25Z.

0. `date -u` in its own call. If it is past 04:14:00Z, launch nothing on a card and return to the lead (ask for the Fri 17:45Z to 19:45Z row, card 0, cores 0-9).

1. Set up from origin:
R=/agent/repos/verity
git -C $R fetch origin cursor/circuits-cir264-sampled-accept-6f04 cursor/circuits-cir264-scratch-6f04
git -C $R rev-parse origin/cursor/circuits-cir264-sampled-accept-6f04
(the pull request's branch, 903c545eb3d34a50eccbf66352a19a805efa0aeb when this timer was set; the capture runs from exactly that commit whatever the head is now)
mkdir -p /tmp/cir264 && git -C $R archive origin/cursor/circuits-cir264-scratch-6f04 scratch/cir264 | tar -x -C /tmp/cir264
(the scratch branch, 7b006a0ea7d703b1fe283d246b4c3bf7853b5ea5 when this timer was set; a later head descends from it)
sha256sum /tmp/cir264/scratch/cir264/accept_capture.sh
(7469e283f83020f1eaeaae378f7935a6de9b7ca6557cb3377aafb40bb5c0c27d: the card script as rehearsed off the card in r20261112-232227-fd47)
git -C $R worktree add --detach /tmp/cir264/tree 903c545eb3d34a50eccbf66352a19a805efa0aeb && git -C /tmp/cir264/tree rev-parse HEAD
Then read /tmp/cir264/scratch/cir264/wake_plan.md: it has every step below in full. The checkout stays on `main`; `research run --source /tmp/cir264/tree` ships the worktree's commit.

2. Re-read the calendar's row first (a row can move): `cd /agent/repos/verity && uv run -q --project tools/research research notes sync`, then `rg -n "2026-11-13T03:00Z" ~/.research/notes/calendars/node-a.md`. The row `| 2026-11-13T03:00Z | 2026-11-13T04:25Z | circuits | 0, 1, 2, 3 | 0-39 |` must still give "card 0 with cores 0-9 CIR-263 and from 03:45Z CIR-264's accept kernels". If it moved, was released or is gone: launch nothing and return to the lead with the row's text.

3. At 03:45:20Z or later by `date -u` (wait with the waiting tool, AwaitShell with block_until_ms and no shell id, in slices of at most ten minutes, never a shell sleep; read `date -u` after each wait), launch ONCE:
cd /agent/repos/verity && uv run -q --project tools/research research run --on vy-spot-glm53 --project verity --source /tmp/cir264/tree --cwd source --timeout 30m --mem-gb 16 --send /tmp/cir264/scratch/cir264/accept_capture.sh --env CARD=0 --env CORES=0-9 --env NOT_BEFORE=2026-11-13T03:45:00Z --env END_UTC=2026-11-13T04:25:00Z --env MAX_MIN=25 --env "ARGS=--random 400" -- bash -c 'bash $RESEARCH_RUN_DIR/inputs/accept_capture.sh'
The script checks that card 0 is idle before it touches it, and its one lease declares its cores: `GPU_LEASE_WHO=circuits-accept bash "$LEASE" 1 --on "$CARD" --cores "$CORES" --max-min "$MAX_MIN" --mem-gb 16 -- taskset -c "$CORES" "$PY" ...`.
Read the run with `research fetch --all RUN` (then ~/.research/runs/RUN/stdout.log, stderr.log, status.json).
- Exit 0 or 1: the capture ran (GO or NO-GO). Do not launch again.
- Exit 76 (card 0 not idle: CIR-263's job may still hold it), or 75 without a quota refusal: look again two minutes later.
- LAUNCH CUTOFFS: `MAX_MIN=25` until 04:00:00Z; then `--env MAX_MIN=15` until 04:10:00Z; then `--env MAX_MIN=10 --env "ARGS=--random 100"` until 04:14:00Z; after that the hours are lost: return to the lead.
- Exit 75 with `gpu-lease: refused, holding nothing` (the node's CPU quota): do NOT retry in a loop. Write the lead one checkpoint line with the refusal's text (`bash /tmp/cir264/scratch/cir264/ckpt.sh "TEXT"`); at most one more launch, after the named lease's `until=` and inside the cutoffs; else return to the lead.
- Exit 2, 3 or 124: read why (wake_plan.md says what each means); return to the lead unless the cause is plain and one more launch fits the cutoffs.

4. After the capture, read it: cases and words compared and how many differ (never "passes" alone), `halves-split` apart (it is outside the Definition's domain), each kernel's PTX `ok` and machine code `code_ok`, the batch, the minutes under the lease (gpu-lease-usage.jsonl), the run's wait and wall time, its `.custody` run_record. Then the replay off the card:
tar -czf /tmp/cir264/capture.tgz -C ~/.research/runs/RUN/out capture/sampled_accept_capture.json capture/cases
cd /agent/repos/verity && uv run -q --project tools/research research run --on auto --project verity --source /tmp/cir264/tree --cwd source --timeout 40m --mem-gb 16 --send /tmp/cir264/scratch/cir264/accept_replay.sh --send /tmp/cir264/capture.tgz --env N=12 -- bash -c 'bash $RESEARCH_RUN_DIR/inputs/accept_replay.sh'

5. After a GO, in the worktree, for the pull request's branch: `verity/definitions/evidence.json` gets `sampled_accept_capture` (the capture's run_record) and `_CONF` in `verity/definitions/registry/sampled_accept.py` names it. Change nothing else there unless a kernel's word differs from the reference (then say which lines), and do not cut the accept's units again (that is CIR-273's). Commit and `git -C /tmp/cir264/tree push origin HEAD:cursor/circuits-cir264-sampled-accept-6f04` (never --force). Then through `research run --on auto`, one run at a time: circuit-check on the module's targets, the suites (`suites.py --changed 3498d4ad65 --quick --json $OUT/suites.json --logs $OUT/logs`, then `repository` whole), the retired-terms lint, the ratchet gate. Then the pull request, open and unmarked, by Fri 13 Nov noon PST: `research queue open --head cursor/circuits-cir264-sampled-accept-6f04 --sha SHA --title T --body-file F --issue CIR-264 --by circuits`, its body with the lines `Lane circuits`, `Refs CIR-264`, `Goal vllm/production-vllm-reproducible`, and saying that the `logits` port's writer is CIR-273, that vLLM's split top-p pipeline takes at most 64 rows a call under a draft [vllm's], and that how leakage counts the masked row is asked of proofs on CIR-264. After a NO-GO: keep the run, change nothing on a guess, and return with the first differing words.

6. Before the turn ends: commit and push everything (the pull request's branch if it changed; a checkpoint line and any new scratch file with ckpt.sh). Read list_subscriptions: no timer may be left; say how many were held (2: `cir264-card-hours`, replaced before it fired, and this one). Return to the lead: the capture's run id and what you read in it, the replay's, which lines of sampled_accept.py changed or none, the pull request, and what was not run.

Rules that still bind: only node A, card 0, cores 0-9, inside the hours, `GPU_LEASE_WHO=circuits-accept`; one run in a check slot at a time; no shell sleep; kill only by PID; never print a secret's value; on Linear post only on CIR-264 (`research msg linear comment CIR-264 --file F.md --worker ACCEPT`), nothing on Slack; write a time, count, SHA or run id only from output read in that session.
