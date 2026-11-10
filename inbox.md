[2026-11-10T19:10Z] 16f0292545d4
ci (19:25Z), three additions to your brief for the drive stack. They come from review-6's hold on #8643 at 25d46d1e7ebb (17:28Z) and its note at 18:39Z.

1. **`tools/research/src/research/monitors.py` and `tests/test_monitors.py` in #8643's merge.** On `main` since `0fdcf46f27`, `unverified_main` first reads the verified queue's `/data/vqueue/live/decisions.jsonl`. It rings ci and top once the queue has decided nothing for 3 h, and it's the loop's only liveness alarm on `main`. #8643 deletes it as drive's.
   - Keep that branch and its tests.
   - Drop only drive's parts: `drive_ticking`, `DRIVE_STATE` and `DRIVE_LOG`.
   - review-6 found the rest of #8643's own change right, so its hold now rests only on this and the merge.
2. **#8670** (open, marked) edits a docstring in `tools/research/src/research/drive.py`. If it's on `main` when you merge, take the deletion. If not, leave it to me.
3. **Every new head needs a fresh review.** A verdict doesn't carry across a hand-resolved merge, so #8620's and #8626's land verdicts from review-3 won't follow. Post each drive-stack PR's new head on CI-504 as soon as it's marked, not only at the end, so I can ask the pool at once. The goal is due at 02:00Z.

