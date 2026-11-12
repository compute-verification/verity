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

Row: Fri 13 Nov 03:00Z-04:25Z, circuits; ours is card 0 with cores 0-9 from 03:00Z to 03:45Z (ACCEPT, CIR-264, has card 0
from 03:45Z). It starts only once proofs' row on cores 0-47 has ended (03:00Z, or when proofs releases it). Nothing of ours
runs past 03:45Z: set MAX_MIN to the whole minutes left before 03:43Z, at most 30, and launch nothing with under 5 left.
If the row moved or the hours don't fit, launch nothing and return to the coordinator.

## 3. Launch, one capture at a time on node A (from /agent/repos/verity, shipping the worktree)

The card script checks card 0 holds no compute process and under 1 GiB (else exits 75 touching nothing), then takes ONE
`gpu-lease` on card 0 with `--cores 0-9` and `--max-min $MAX_MIN`, the job pinned with `taskset -c 0-9`.

    cd /agent/repos/verity && uv run -q --project tools/research research run --on vy-spot-glm53 --project verity --source /tmp/cir263-pr --cwd source --timeout 2400 --wait --env CARD=0 --env CORES=0-9 --env GPU_LEASE_WHO=circuits-draftkernels --env MAX_MIN=30 -- bash adapters/vllm/tests/properties/glm53_mtp_capture_card.sh eh_norm
    cd /agent/repos/verity && uv run -q --project tools/research research run --on vy-spot-glm53 --project verity --source /tmp/cir263-pr --cwd source --timeout 2400 --wait --env CARD=0 --env CORES=0-9 --env GPU_LEASE_WHO=circuits-draftkernels --env MAX_MIN=30 -- bash adapters/vllm/tests/properties/glm53_mtp_capture_card.sh eh_proj
    cd /agent/repos/verity && uv run -q --project tools/research research run --on vy-spot-glm53 --project verity --source /tmp/cir263-pr --cwd source --timeout 2400 --wait --env CARD=0 --env CORES=0-9 --env GPU_LEASE_WHO=circuits-draftkernels --env MAX_MIN=30 -- bash adapters/vllm/tests/properties/glm53_mtp_capture_card.sh indexer_logits

(Lower MAX_MIN per section 2.) `--wait` stops after 20 minutes; then read `research status RUN --refresh` between waits
(the waiting tool, never a shell sleep), and `research fetch RUN --all` when done. Start the next only after the previous
ended. A lease refused at the quota: no retry loop; read who holds what (`gpu-lease` prints it) and write one checkpoint
line with the refusal's text. Exit 75 (card not idle): same, no retry.

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
