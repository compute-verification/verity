# CIR-263 (DRAFTKERNELS): what a fresh machine does next

A turn starts on a new machine with a new clone on `main`; only what is pushed survives. Commit and push to
`cursor/circuits-cir263-scratch-6f04` before every turn ends. Nothing goes to the PR branch except reviewed code changes.

- PR branch: `cursor/circuits-cir263-draft-layer-kernels-6f04`, head `ce09de03c318f40d29efbec7cca2536a8b7cf735` (four commits
  over `920597d5ca7f`, the head the first row's lease ran at): the three captures are registered properties (schema ids in
  `verity_vllm/records/mtp_captures.py`); the split-K probe separates strided from contiguous at K 12288, 2 splits, bf16;
  `tests/properties/test_glm53_mtp_captures.py` drives each capture's `main` on a CPU to its sealed record; and the eh_proj
  capture puts its shape in the sibling's `MODELS` only while a run or a scan lasts (at `78884a1b8a` the import did it, and
  `verity-vllm` r20261113-040826-9fec failed 18 of the sibling's tests on the third model: 14,979 passed, 18 failed). At
  `ce09de03c3`: the ratchet gate `failures: []`; `repository` r20261113-052134-678c 352 passed, 0 failed; `verity-vllm`
  r20261113-052846-14e2 (its counts are in `checkpoint.md`; if the last lines there do not give them, read `research status`
  and fetch it). No PR open yet: it opens after the captures, unmarked, by Mon 16 Nov 16:00Z.
- Tonight's lease (Fri 03:15Z, r20261113-031522-215e at `920597d5ca7f`, `jobs/card_row.sh`): every comparison that ran found
  0 differing words (eh_norm 25,165,824 words; indexer_logits 290,933 words), but no record was sealed (the schemas were not
  registered) and eh_proj stopped at its first split launch (the probe). So nothing of tonight is a capture of record; the
  numbers are in `checkpoint.md` and `tonight.md`.
- Drafts here: `comment-draft.md` (CIR-263 comment, card results as `<...>` placeholders), `pr-body.md` (PR body), `tonight.md`
  (tonight's run, read from its logs), `jobs/` (card_one.sh, card_row.sh, suite.sh, cc_pins.sh, ehproj_words.*, ehnorm_ptx.py),
  `dry_*.py`.

## 1. Checkout (before anything is launched)

    git -C /agent/repos/verity fetch origin cursor/circuits-cir263-draft-layer-kernels-6f04 cursor/circuits-cir263-scratch-6f04
    git -C /agent/repos/verity worktree add -B cursor/circuits-cir263-draft-layer-kernels-6f04 /tmp/cir263-pr origin/cursor/circuits-cir263-draft-layer-kernels-6f04
    git -C /tmp/cir263-pr rev-parse HEAD        # must print ce09de03c318f40d29efbec7cca2536a8b7cf735 (or what checkpoint.md's last line names)
    git -C /agent/repos/verity worktree add -B cursor/circuits-cir263-scratch-6f04 /tmp/cir263-scratch origin/cursor/circuits-cir263-scratch-6f04

Push from the worktrees only (`git -C /tmp/cir263-scratch push origin cursor/circuits-cir263-scratch-6f04`,
`git -C /tmp/cir263-pr push -u origin cursor/circuits-cir263-draft-layer-kernels-6f04`); a separate clone has no push
credentials. One named test file locally:
`cd /tmp/cir263-pr/adapters/vllm && PYTHONPATH=/tmp/cir263-pr/adapters/vllm:/tmp/cir263-pr timeout 170 /agent/repos/verity/.venv/bin/python -m pytest -q -p no:cacheprovider tests/properties/FILE.py`
(after `uv sync --all-packages --extra torch-cpu` in /agent/repos/verity if the venv is missing).

## 2. Calendar (read again right before the first launch)

    cd /agent/repos/verity && uv run -q --project tools/research research notes sync
    rg -n "CIR-263" ~/.research/notes/calendars/node-a.md

Friday's row as read at Fri 04:30Z: 2026-11-13T15:45Z to 19:45Z, circuits, PROVISIONAL; CIR-263 has card 0 with cores 0-9
until 17:45Z, then CIR-264 has card 0. "One lease a job, none over 30 minutes, the same start check; untimed." Circuits says
in the row by Fri 14:00Z which cards it still needs, so the row may have changed: if it moved, shrank or no longer names
CIR-263 on card 0, launch nothing and return to the coordinator.

## 3. Launch: one capture a lease, on node A, with the pull request's own card script

`jobs/card_one.sh` (shipped with `--send`) adds only the row's clock to the PR's `glm53_mtp_capture_card.sh`: nothing before
NOT_BEFORE (exit 75, the card untouched), MAX_MIN = whole minutes left before END less 2 minutes, at most 30, nothing with
under 5. The card script checks card 0 is idle (exit 75 if not), takes ONE lease with `--cores 0-9`, pins every process, and
runs the capture, whose record, PTX and words go to `$OUT/<NAME>`.

- Read `date -u` in its own command (twice: the first read after a wait is stale) and launch only at or after 15:45:00Z.
- One run at a time, each after the previous has ended (`research status RUN --refresh`); the last start is 17:10Z, so that a
  30-minute lease is over by 17:43Z. Order: `eh_proj` first (the one with a launch tonight did not reach: split-K 2), then
  `eh_norm`, then `indexer_logits`.
- Card 0 busy (exit 75 from the start check): wait (the waiting tool, never a shell sleep) and launch again; say in the
  return how many minutes it cost. A lease refused at the quota: no retry; read who holds what (`gpu-lease` prints it) and
  write one checkpoint line with the refusal's text.

    cd /agent/repos/verity && uv run -q --project tools/research research run --on vy-spot-glm53 --project verity --source /tmp/cir263-pr --cwd source --timeout 2100 --wait --send /tmp/cir263-scratch/scratch/cir263/jobs/card_one.sh --env NOT_BEFORE=2026-11-13T15:45:00Z --env END=2026-11-13T17:45:00Z --env CARD=0 --env CORES=0-9 --env GPU_LEASE_WHO=circuits-draftkernels -- bash -c 'exec bash "$RESEARCH_RUN_DIR/inputs/card_one.sh" eh_proj'

and the same line with `eh_norm`, then with `indexer_logits`. `--wait` stops after 20 minutes; then
`research status RUN --refresh` between waits and `research fetch RUN --all` when done (into `~/.research/runs/RUN/`).

What to read in each run: stdout's `start ... card 0: N compute processes, M MiB`, the lease's lines, the capture's last line
`GLM53-MTP-<NAME> GO|NO-GO ...` and `end ... exit RC`; `out/<name>/glm53_mtp_<name>_capture.json` (`ok`, `digest`, the
summary's rows and words compared and words that differ, the reference counts, the routes and splits for eh_proj, the PTX
hash and counts for eh_norm, the alternatives' refutation counts, and `environment.vllm`: say whether it names the pin
`d9105ea80`, since nothing in the capture checks it). Say rows and words compared and words that differ, never
"passes" alone. A phase that raises writes no record: the traceback is in stdout.

Expected from tonight (to compare, not to copy): eh_norm 4 x 6,291,456 words; indexer_logits 100,664 + 190,269 words;
eh_proj routes at K 12288, N 6144 unsplit except split-K 2 at M 105-111, 129-159, 161-223, 272, 288, 336, so 24 launches
(M 1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 105, 128, 192, 256, 384, 512, 768, 1024, 1025, 8192), split at M 105 and
192, 4,458 rows compared (computed here from the capture's plan, about 4 to 5 minutes of words on 8 workers by tonight's
2.2 rows per worker-second).

If eh_proj's split launches do not fit `contiguous-f32-forward` (its `hypothesis` block names which model fits): that is a
finding, not a failure to hide. Keep the words (they are under `out/eh_proj/words/`), write what fits and what does not in the
checkpoint, and return to the coordinator before changing any Definition or binding.

## 4. After the captures

- Fill `comment-draft.md`'s placeholders (run ids, rows, words compared, words that differ, routes, splits), lead with the
  result in bold, and post by Sat 14 Nov 02:00Z:
  `research msg linear comment CIR-263 --file /tmp/cir263-scratch/scratch/cir263/comment-draft.md --worker DRAFTKERNELS`
  (from /agent/repos/verity with `uv run -q --project tools/research`). Then return to the coordinator.
- eh_proj's routes give the splits: add `Gemm_v2{K=12288,N=6144,DOT=HopperBF16WgmmaDot16_v1}` and `CublasSplitKGemm_v1` at each
  split seen to `mtp_draft.circuit-check.py`, run `jobs/cc_pins.sh` with `--cwd clone`, apply its pins diff, rerun
  `repository` and then `verity-vllm` (`jobs/suite.sh NAME`, one run at a time, `--on auto --cwd source --wait`), then the
  ratchet gate (`git fetch origin main && python3 adapters/vllm/tools/ratchet_gate.py origin/main HEAD --json`); any change
  of a Definition's name, statics or ports since the first comment goes to the coordinator first, old and new. A capture of
  record that ran at an earlier head stays of record only with the PR body saying at which head it ran and what differs
  since.
- The PR: open, unmarked, no `### Mark` section, by Mon 16 Nov 16:00Z, from `pr-body.md`.
- Commit and push the scratch branch before the turn ends; a checkpoint line at least every 60 minutes.
