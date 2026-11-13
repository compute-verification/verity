# CIR-262, worker FP8READ: state a later turn needs

Everything here stands alone on a fresh machine. Paths are in this directory (`scratch/cir262/` of branch
`cursor/circuits-cir262-scratch-6f04`) unless they start with `/` or `~`. The scripts expect to sit in `/tmp/cir262/`:

~~~text
git fetch origin cursor/circuits-cir262-scratch-6f04 cursor/circuits-cir262-flashmla-fp8-read-6f04 main
git checkout cursor/circuits-cir262-flashmla-fp8-read-6f04          # the PR branch; never push scratch files to it
mkdir -p /tmp/cir262 && git archive origin/cursor/circuits-cir262-scratch-6f04 scratch/cir262 | tar -x --strip-components=2 -C /tmp/cir262
~~~

`research` is `uv run -q --project tools/research research ...`, run from the checkout.

## Where things stand (updated at each checkpoint; `checkpoint.md` has the one-line history)

See the last lines of `checkpoint.md` and the section "Now" at the end of this file.

## The branch

- PR branch `cursor/circuits-cir262-flashmla-fp8-read-6f04`, cut from `main` at `3498d4ad6546bed9289878c1b4cf19f218d7478e`.
- `1f6f0969a9b5b7a50585dcc8ba89c5af7548eb4b`: the head both captures of record and the all-token evaluator ran at.
- `ce254bd223c6e83f0a0cc1cd79ad9531977d302f`: drops the script's seven catch-all handlers (the adapter's lint ratchet;
  `python3 adapters/vllm/tools/ratchet_gate.py origin/main HEAD --json` prints `failures: []`).
- The nine changed files: `changed.txt`.

## Runs read so far (ids are in the run store: `research fetch RUN --all`)

- Card, node A card 1, head `1f6f0969a9`: `r20261112-181549-6fce`, `r20261112-183549-43fa`. Each: 30 cases, 1,472 tokens,
  48,234,496 words compared, 0 differ (reference function 6,782,976 / 0; other 55 pad heads against pad head 8
  41,451,520 / 0); evaluator 73 tokens, 299,008 words, 0 differ. Record digests `1002e7be1d2d530a9055cac2a44a0009912c6126c3acc19c445bfa871bb9a96e`
  and `9c288715b7903d9779e90cb177f219f4c78bb5d39ddfaa3e3a2b5a7a27257f54`. Leases: 18:16:34Z to 18:16:45Z and 18:36:17Z to
  18:36:28Z (11 s each; a third, quick, run `r20261112-180535-23a8`, 18:06:12Z to 18:06:23Z). None declared `--cores`; the
  wrapper does now.
- All-token evaluator off the card, `r20261112-185922-3b8c` (vy-nebius-cpu-10, head `1f6f0969a9`, from the first card
  run's saved capture): 1,472 tokens, 6,029,312 words, 0 differ, 12 bindings of `FlashMlaV32Decode_v1`; rc 0; waited
  41 min 17 s (19:00:04Z to 19:41:21Z), held slot r2 to 23:05:48Z (3 h 24 min 27 s). It replaces `r20261112-183708-9a9a`,
  cancelled after 21 min in the queue without a slot (the second CIR-262 comment names the cancelled one).
- circuit-check on `1b61bd883e` (the Definitions' files are unchanged since): `r20261112-173637-e88d`; the reports are in
  `pr_body_runs.md`. Pins update pass `r20261112-152809-d14e` wrote nothing.
- Suites: `r20261112-153801-6d0c` (on `55d7a918bb`), `r20261112-170240-ac18` (on `1b61bd883e`). Waits and holds of every
  run: the list at the end of this file.
- `nvidia-smi -q -i 1` around the second capture (for the return only; quote no card-1 time in the PR or on CIR-262):
  every Clocks Event Reason "Not Active" before and after; counters unchanged (SW Power Capping 74,718,974,742 us, Sync
  Boost 0, SW Thermal Slowdown 110,729,508,549 us, HW Thermal Slowdown 604,834,645 us, HW Power Braking 0); temperature
  53 C before, 58 C after; P0.

## The saved capture

`capture.tar` (not on the branch: 255 MB) is `sass/` and `capture/` of the first card run's output directory:

~~~text
research fetch r20261112-181549-6fce --all
tar -cf /tmp/cir262/capture.tar -C ~/.research/runs/r20261112-181549-6fce/out sass capture
cp ~/.research/runs/r20261112-181549-6fce/out/glm53_flashmla_fp8_capture.json /tmp/cir262/card-record.json
~~~

## The jobs

- `final.sh` (sends `cc2.sh`, `suites.sh`, `capture.tar`, the card run's record): the saved capture compared again at the
  head with its record held byte for byte to the card run's; circuit-check's pins update pass on the sidecar; the suites
  the change touches (quick tier) and `repository` whole.

  ~~~text
  research run --on auto --project verity --source /agent/repos/verity --cwd clone --timeout 200m --mem-gb 48 \
    --send /tmp/cir262/final.sh --send /tmp/cir262/cc2.sh --send /tmp/cir262/suites.sh --send /tmp/cir262/capture.tar \
    --send /tmp/cir262/glm53_flashmla_fp8_capture.json --env BASE=<merge base with main> \
    -- bash -c 'bash $RESEARCH_RUN_DIR/inputs/final.sh'
  ~~~

- `fp8read_card.sh` through `launch_card.sh END_UTC CORES [ARGS] [DEF_TOKENS]` (env `NOT_BEFORE`): the capture on node A's
  card 1. Only inside my row of `~/.research/notes/calendars/node-a.md` (`research notes sync`, then
  `rg -n FP8READ ~/.research/notes/calendars/node-a.md`), on its cores, one lease of at most 30 minutes with `--cores`,
  `GPU_LEASE_WHO=circuits-fp8read`.
- `cc2.sh` (circuit-check, env `CC_ARGS`, `SERVED`), `suites.sh` (env `BASE`), `suites_named.sh` (env `SUITES`),
  `defs_all.sh` (the all-token evaluator from `capture.tar`), `rss_one.sh`, `retired.py FILES...` (the retired-terms lint).

## The PR

Open, unmarked: https://github.com/compute-verification/verity/pull/9567 at head
`70a27d97ec6086b5fcd45b3bf936efa1bd6702e1` (my commits to `ce254bd223`, then `main` `417526501974` merged in). Its body is
`pr_body.md`, assembled from `pr_body_static.md`, `pr_body_runs.md`, `pr_body_capture.md`, `pr_body_suites.md` and
`pr_body_notrun.md`; edit it with `research queue edit 9567 --body-file F`. `comment1.md`, `comment2.md` and `comment3.md`
are the three comments as posted on CIR-262. The coordinator marks it; I do not.

Final job `r20261112-231418-ada0` at that head, rc 0: the first card run's saved capture compared again gives the card
run's record byte for byte; the pins update pass changed no file; circuit-check sidecar 12 targets and served-size heads
5 targets, 0 failures, 0 unchecked, the whole token 0 failures, 1 unchecked (edge vectors only); `--changed 417526501974
--quick` 57 suites passed (8 cached), `repository` whole 356 passed. Waited 0 s, held slot r0 from 23:14:51Z to 00:36:46Z.

#9567 is on `main` (merged at `70a27d97ec6086b5fcd45b3bf936efa1bd6702e1`); its branch is gone from the remote, and the
commit is an ancestor of `main`: `git checkout --detach 70a27d97ec6086b5fcd45b3bf936efa1bd6702e1` gives its tree. `main`
has since renamed these paths, so the scripts here fit that commit, not `main`.

Closed on Fri 13 Nov: the launch stage ran on a card at that head. Run `r20261113-163614-e29b` (node A, card 1, cores
10-19, source exactly that commit, `fp8read_card.sh` as it is here): exit 0; one lease with `--cores`, 16:36:59Z to
16:37:11Z; 30 cases, 1,472 tokens, 48,234,496 words compared, 0 differ; evaluator 73 tokens, 299,008 words, 0 differ;
record ok, digest `14d0bd3a6419a5b399df5a9207adee13c8756e4a59f738810ffed19c19f594d2` (recomputed by `redigest.py`).
Against Thursday's `r20261112-181549-6fce` (`research fetch RUN --on vy-spot-glm53 --all`): 246 of 247 capture files the
same bytes, `launch.json` differing in its two paths and the `errors` key; the 20 SASS listings equal once unzipped. Card
1 read 0 MiB, no process, lease free at the run's end. `comment4.md` is the comment as posted on CIR-262.

## Waits and holds (read in each run's `slot.json` and logs)

- `r20261112-150200-2bd9` circuit-check: waited 84 s, held 17 min 33 s.
- `r20261112-152809-d14e` pins pass: waited 4 min 45 s, held 59 s.
- `r20261112-153801-6d0c` suites: waited 91 s, held 80 min 20 s.
- `r20261112-170240-ac18` suites: waited 3 s, held 27 min 26 s.
- `r20261112-173637-e88d` circuit-check: waited 2 min 15 s, held 18 min 48 s.
- `r20261112-183708-9a9a`: waited 21 min, cancelled, never held a slot.
- `r20261112-185922-3b8c` evaluator: waited 41 min 17 s, held 3 h 24 min 27 s.
- `r20261112-191237-3c6f` (node A cores 12-23, no card, no lease): one token's evaluation peaks at 919 MiB.
- Timers held: 0.

## Now

- 2026-11-12T23:12:46Z: the evaluator job is read. Next: merge main into the PR branch, launch final.sh on that head, open the PR, and (if my row of Fri 03:00Z to 04:25Z stands) one more capture at the PR's head.
- 2026-11-12T23:15:00Z: the PR branch is at 70a27d97ec6086b5fcd45b3bf936efa1bd6702e1 (main 417526501974 merged in; the diff against main is the nine files). The final job r20261112-231418-ada0 (final.sh, BASE=417526501974912c8776a6ea920fb51024038eeb) is running on vy-nebius-cpu-4. When it ends: read out/rc.txt, out/recompare-verdict.txt, out/pins, out/cc, out/suites.json and out/suites-repository.json; fill the PR body; open the PR.
- 2026-11-13T00:07:51Z: final job r20261112-231418-ada0 so far, read: the saved capture compared again at 70a27d97ec gives the card run's record byte for byte (digest 1002e7be...); pins pass changed no file; circuit-check sidecar 12 targets 0 failures, heads 5 targets 0 failures, whole token 0 failures 1 unchecked (edge vectors only); suites running (verity-vllm passed: 14,366 passed, 401 skipped).
- 2026-11-13T00:50:42Z: PR #9567 is open at 70a27d97ec, unmarked; third comment posted on CIR-262; returning to the coordinator.
- 2026-11-13T17:01:14Z: the capture at #9567's merged head is read and the fourth comment is posted; nothing is open on CIR-262 from me.
