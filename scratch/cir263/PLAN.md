# CIR-263 (DRAFTKERNELS): what a fresh machine does next

A turn starts on a new machine with a new clone on `main`; only what is pushed survives. Commit and push to
`cursor/circuits-cir263-scratch-6f04` before every turn ends. Nothing goes to the PR branch except reviewed code changes.

- PR branch: `cursor/circuits-cir263-draft-layer-kernels-6f04`, head of record `920597d5ca7f9d1ac9330eab4a6e0944a5c4215e`
  (ratchet gate `failures: []`; `repository` r20261112-210847-6529 352 passed; `verity-vllm` r20261112-215601-1b52 14,983
  passed, 0 failed, 485 skipped). No PR open yet: it opens after the captures, unmarked, by Mon 16 Nov 16:00Z.
- Drafts here: `comment-draft.md` (CIR-263 comment, card results as `<...>` placeholders), `pr-body.md` (PR body), `jobs/`
  (suite.sh, suites.sh, cc_pins.sh, ehnorm_ptx.py), `dry_*.py` (CPU dry runs of each capture's plumbing).

## 1. Checkout (before anything is launched)

    git -C /agent/repos/verity fetch origin cursor/circuits-cir263-draft-layer-kernels-6f04 cursor/circuits-cir263-scratch-6f04
    git -C /agent/repos/verity worktree add --detach /tmp/cir263-pr 920597d5ca7f9d1ac9330eab4a6e0944a5c4215e
    git -C /tmp/cir263-pr rev-parse HEAD        # must print 920597d5ca7f9d1ac9330eab4a6e0944a5c4215e
    git -C /agent/repos/verity worktree add -B cursor/circuits-cir263-scratch-6f04 /tmp/cir263-scratch origin/cursor/circuits-cir263-scratch-6f04

Push the scratch branch from the worktree (`git -C /tmp/cir263-scratch push origin cursor/circuits-cir263-scratch-6f04`);
a separate clone has no push credentials.

## 2. Calendar (read again right before the first launch)

    cd /agent/repos/verity && uv run -q --project tools/research research notes sync
    rg -n "CIR-263" ~/.research/notes/calendars/node-a.md

Row: Fri 13 Nov, circuits; ours is card 0 with cores 0-9 from 03:15Z to 03:45Z, 30 minutes. Cards 0-3 and cores 0-39 are
vllm's from 03:00Z to 03:15Z (VLLM-339, a serve on all eight cards): no lease on card 0 before 03:15Z, or that serve stops.
ACCEPT (CIR-264) has card 0 from 03:45Z: be off the card by then. If the row moved again or does not fit, launch nothing
and return to the coordinator.

## 3. Launch: ONE lease tonight, on node A (from /agent/repos/verity, shipping the worktree)

Tonight is one lease, the first kernel first. The card script takes one lease per capture, so tonight runs `eh_norm` only;
`eh_proj` and `indexer_logits` go to Friday's row (15:45Z, card 0), which holds the capture of record.

- Launch at or after 03:15:00Z by `date -u` read in its own command (read it twice: the first read after a wait is stale).
  Every launch carries `NOT_BEFORE=2026-11-13T03:15:00Z`; the guard in front of the card script refuses on the node before
  it (exit 75, nothing touched).
- MAX_MIN = the whole minutes left before 03:43Z at launch, at most 28; launch nothing with under 5 left.
- The card script checks card 0 holds no compute process and under 1 GiB (else exits 75, touching nothing), then takes ONE
  `gpu-lease` on card 0 with `--cores 0-9` and `--max-min $MAX_MIN`, the job pinned with `taskset -c 0-9`.
- Card 0 busy at 03:15Z (exit 75 from the start check) means vllm's serve overran: wait (the waiting tool), launch again,
  last start 03:25Z; say in the return how many minutes it cost. A lease refused at the quota: no retry; read who holds
  what (`gpu-lease` prints it) and write one checkpoint line with the refusal's text.

    cd /agent/repos/verity && uv run -q --project tools/research research run --on vy-spot-glm53 --project verity --source /tmp/cir263-pr --cwd source --timeout 1800 --wait --env NOT_BEFORE=2026-11-13T03:15:00Z --env CARD=0 --env CORES=0-9 --env GPU_LEASE_WHO=circuits-draftkernels --env MAX_MIN=28 -- bash -c 'nb=$(date -u -d "$NOT_BEFORE" +%s); if [ "$(date -u +%s)" -lt "$nb" ]; then echo "before NOT_BEFORE=$NOT_BEFORE: starting nothing"; exit 75; fi; exec bash adapters/vllm/tests/properties/glm53_mtp_capture_card.sh "$0"' eh_norm

Friday's row (15:45Z-17:45Z, card 0, cores 0-9; CIR-264 from 17:45Z), one capture a lease, each after the previous ended,
NOT_BEFORE=2026-11-13T15:45:00Z and MAX_MIN the whole minutes left before 17:43Z (at most 30): `eh_proj`, then
`indexer_logits`, then `eh_norm` again if tonight's did not finish; the same line with the name changed.

`--wait` stops after 20 minutes; then read `research status RUN --refresh` between waits (the waiting tool, never a shell
sleep), and `research fetch RUN --all` when done.

Each record: `<OUT>/<name>/` holds the capture's JSON record (`ok`), words and PTX; the last stdout line is
`GLM53-MTP-<NAME> GO|NO-GO ...` with words compared and words that differ. Say rows/words compared and words that differ,
never "passes" alone. A phase that raises writes no record: the traceback is in stdout.log.

## 4. After the captures

- Fill `comment-draft.md`'s placeholders (run ids, rows, words compared, words that differ, routes, splits), lead with the
  result in bold, and post by Sat 14 Nov 02:00Z:
  `research msg linear comment CIR-263 --file /tmp/cir263-scratch/scratch/cir263/comment-draft.md --worker DRAFTKERNELS`
  (from /agent/repos/verity with `uv run -q --project tools/research`). Then return to the coordinator.
- eh_proj's routes give the splits: add `Gemm_v2{K=12288,N=6144,DOT=HopperBF16WgmmaDot16_v1}` and `CublasSplitKGemm_v1` at each
  split seen to `mtp_draft.circuit-check.py`, run `jobs/cc_pins.sh` with `--cwd clone`, apply its pins diff, rerun
  `repository` and `verity-vllm` (`jobs/suite.sh NAME`), then the ratchet gate; any change of a Definition's name, statics
  or ports since the first comment goes to the coordinator first, old and new.
- Card row 2: Fri 15:45Z-17:45Z, card 0 cores 0-9 (CIR-264 from 17:45Z): any capture to repeat.
- Commit and push the scratch branch before the turn ends.
