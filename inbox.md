[2026-11-11T12:11Z] 03ed5de0c13f
CROSS, 4:11 AM PST (12:11Z). To training. FINAL MESSAGE of TRN-82 CROSS: DONE. Nothing of this job is open with me. All run or read by me.

LAST: #8606 marked at 86d435632e3764ca2cc65f4662865cfda880e358, 4:01 AM PST (12:01:56Z), rc 0; the queue says ready, admitted. Suites r20261111-110943-0cbb, 9 of 9 (none cached), source.tree f685216b74f3, the head's. One comment is on TRN-82 with the head, the run and the mark (92 words). GitHub shows #8606 as a draft, as ci opened it; the queue ignores draft state (queue.py: "Draft state is ignored: agents open PRs as drafts").

SHARPER THAN I SAID AT 12:04Z, on #8731: it conflicts with #8606's head in 20 files, and in the same 20 with main 86c03d1840cb alone, none of them one of the four READMEs. #8606 adds nothing to its conflict.

THE QUEUE at 12:06Z: #8550, #8551, #8577, #8606, #8680, #8712, #8753, #8775, #8778, #8809, #8840, #8854 each "ready ... admitted" at the head in the table's "head now"; #8541 and #8591 merged.

PROVISO: YES. Every input of #8541's crossed list (7b7ae0802eab) is in #8551's list at 0bb3841abce9, the package's and the grader's pyproject.toml both, and so is every input of main's (read again at 12:09Z).

THE HAND-OFF NOTE, synced: training-20261111T1208Z-handoff-trn82-cross-crossing-done. Its three sections follow whole. Evidence for #8606: /opt/cursor/artifacts/trn82-8606-readmes.log (the seven commits, the claims in the code, the trial merges, the five runs' records, the mark).

## Every head

Times are Pacific, the night of Tue 10 to Wed 11 Nov. "Crossed" is the head after the layout's move; "now" is the head
on origin when this was written.

| PR | old head | crossed head | head now | crossed by | suites at the head's tree | marked | the mark rests on |
|---|---|---|---|---|---|---|---|
| #8541 | `2758e3d11201` | `7b7ae0802eab` | on `main` (`507903f4dbe9`) | me (`restack.py`, clean) | `r20261111-041544-e6b6` 4 of 4 | 8:49 PM | that run and the quick tier `r20261111-043022-6b9b` |
| #8591 | `3bd2ed9a166a` | `e83685c5feaa` | on `main` (`d21813f29c`) | ci (`r20261111-035048-0337`) | TIE's `r20261111-014053-109d` 6 of 6, before the crossing | 9:46 PM, by me | ci's crossing run and TIE's run |
| #8550 | `64d4b2dea887` | `27f04254356f` | `54805be6d6b5` | me, clean; then #8591's crossed head merged in (`73dd43836579`); then the tests' tolerance | `r20261111-041545-8b9b` 8 of 8 at `27f04254356f`; `r20261111-044403-adc0` 50 of 50 at `73dd43836579`'s tree; `r20261111-072858-c576` 3 of 3 at `54805be6d6b5` | 10:10 PM at `73dd43836579`; held by review-14 at 10:47 PM; again 11:44 PM at `54805be6d6b5` | `r20261111-072858-c576` and the VM's counts |
| #8551 | `b1f563aefacb` | `72ee373de033` | `0bb3841abce9` | me, one hand fix; then `main` merged in, one hunk by hand | `r20261111-035308-1c55` 27 of 27 and `r20261111-044402-0d17` 50 of 50 at `72ee373de033` | 10:49 PM at `72ee373de033`; 1:24 AM at `0bb3841abce9` | at the new head: the merge of `main`, the one README hunk, the two tests, and the run at the old head |
| #8680 | `f84a02f0a1cc` | `6991ea8e32f9` | `09e3ec29ee74` | me; then #8551's new head merged in, no conflict | `r20261111-035707-6547` 27 of 27 and `r20261111-044403-9338` 50 of 50 at `6991ea8e32f9` | 10:51 PM; 1:26 AM at `09e3ec29ee74` | the same, on `r20261111-044403-9338` |
| #8712 | `ae52eca38092` | `14d09e7837a3` | `0d75e798b9f0` | me; then #8680's merged in, no conflict | `r20261111-040044-7bfb` 27 of 27 and `r20261111-044402-ec45` 50 of 50 at `14d09e7837a3` | 10:54 PM; 1:28 AM at `0d75e798b9f0` | the same, on `r20261111-044402-ec45` |
| #8778 | `6d567e320d04` | `e44842663fd0` | `777bc49fda80` | me; then #8712's merged in, no conflict | `r20261111-040341-cb09` 27 of 27 and `r20261111-044403-382d` 50 of 50 at `e44842663fd0` | 10:57 PM; 1:30 AM at `777bc49fda80` | the same, on `r20261111-044403-382d` |
| #8809 | (not crossed by me) | `95239b187833` | `f6dbfd9134bc` | its own worker; I merged #8778's new head in, no conflict | none of mine | 1:32 AM at `f6dbfd9134bc`, by me | the merge and the two tests; marked at its old head (the coordinator's word) |
| #8840 | (not crossed by me) | `189ccfb740ce` | `062c072e6e7b` | its own worker; I merged #8809's new head in, no conflict | none of mine | 1:34 AM at `062c072e6e7b`, by me | the same |
| #8854 | (not crossed by me) | `a4ed47cb711c` | `356070ed9c76` | its own worker; I merged #8809's new head in, no conflict | none of mine | 1:35 AM at `356070ed9c76`, by me | the same |
| #8775 | (not crossed by me) | `b754621d606d` | `7d6a71567da5` | the resume worker; I merged #8680's new head in, no conflict | none of mine | not by me: the coordinator's | |
| #8577 | `0a7ba617e6e0` | `993844f9421b` | `993844f9421b` | me, clean, one commit for a dead path | `r20261111-040341-f67d` 3 of 3 | not by me; the queue shows it ready | |
| #8753 | `4dafee69a48f` | `5fc53c6ca4b1` | `3e436cc09acc` | ci; then my mend of the Definitions' modules | `r20261111-073300-d91f` at `5fc53c6ca4b1`: 3 of 4, 2 tests failed; `r20261111-074912-8545` at `3e436cc09acc`: 4 of 4 | 11:59 PM | `r20261111-074912-8545` |
| #8606 | `6e75357abd59` | `c7619832c307` | `86d435632e37` | ci; then seven commits of mine (three README mends, a merge of `main`, the lists, the lines naming archived files, and main's row restored) | `r20261111-075240-ec16` 9 of 9 at `1433dd58511f`; `r20261111-084241-f3c3` 9 of 9 at `68292c6c93e5`; `r20261111-101610-28e4` 9 of 9 at `83e4fc973b34`; `r20261111-110943-0cbb` 9 of 9 at `86d435632e37` | 4:01 AM at `86d435632e37` | `r20261111-110943-0cbb` and the trial merges before the mark |

Each run named for a head ran at a commit whose recorded `source.tree` equals that head's tree (the rehearsal commits'
trees are in the 06:39Z note). Every push was a fast-forward by literal commit id; nothing forced, nothing amended.
The queue's lines at 4:06 AM (12:06Z), after #8606's mark: every open PR above, #8606 and #8775 with them, is "ready ... admitted" at the head in "head now"; #8541 and #8591 are merged. GitHub shows #8606, #8778 and #8809 as drafts, which the queue ignores (`queue.py`'s own words: "Draft state is ignored: agents open PRs as drafts").

## Hand resolutions, as commands

~~~text
# #8551 at the move's base: benchmarks/README.md, kept the branch's training_slowdown/ row and the layout's deletion of
#   the rho/ and nci/ rows.
# the crossed #8541 into #8551: two files conflict; kept #8551's side, which already holds #8541's change
git merge --no-commit 7b7ae0802eab35359715348a5f3ca4ee780b376e
git checkout --ours -- benchmarks/training_slowdown/README.md benchmarks/training_slowdown/pyproject.toml
git add benchmarks/training_slowdown/README.md benchmarks/training_slowdown/pyproject.toml && git commit   # 9fe94d23f5c9
# #8591's crossed head into #8550: one line
git checkout --detach 27f04254356f971a37def547f15c67bfd7e5a370
git merge --no-commit e83685c5feaaafd4a6e42ef341b891fc8bd03af9
#   benchmarks/training_slowdown/training_slowdown/__init__.py: VERSION "4.14.0" (ours) against "4.13.5"; kept "4.14.0"
git add benchmarks/training_slowdown/training_slowdown/__init__.py && git commit                            # 73dd43836579
# main, after it moved to 0a722c6ec984, into #8551: one hunk
git checkout cursor/attack-harness-config-17f2                                                              # 72ee373de033
git merge --no-commit --no-ff 0a722c6ec984ad93e9f32cfb82dbf601ec9f0ae8
#   benchmarks/README.md: ours is the branch's training_slowdown/ row; theirs is main's new inference_only/ row and the
#   training_slowdown/ row as at the merge base. Kept main's inference_only/ row, then ours, byte for byte.
git add benchmarks/README.md && git commit                                                                  # 0bb3841abce9
# every other merge above (each branch with the new head below it, #8606 with main): `git merge --no-ff`, no conflict
~~~

At each of the eight merged trees: the `training_slowdown/` row of `benchmarks/README.md` is the old head's byte for
byte, main's `inference_only/` row is on the line above it, and
`pytest tests/test_benchmarks_index.py tests/test_doc_paths.py` gives 5 passed. The coordinator checked each head's
tree against `git merge-tree`'s (his word at 1:29 AM).

## What is left, and for whom

- **Nothing of this job is open.** Every head of training's that the brief names is crossed, its suites ran at its
  tree, and it is marked, or is another worker's by the coordinator's word. The goal
  `training/other-changes-cross-move` is the coordinator's to close.
- **Trial merges of #8606 not done:** with the 22 open pull requests of over 100 files, whose file lists GitHub
  cuts (#6856, #7200, #7850, #7952, #8177, #8179, #8221, #8327, #8330, #8363, #8401, #8403, #8415, #8435, #8543,
  #8563, #8569, #8696, #8789, #8795, #8814, #8843). #8731 (`3cb71ba0758c`, not across the move) conflicts with it in
  20 files, the same 20 it conflicts in with `main` `86c03d1840cb` alone, none of them one of the four READMEs: this
  PR adds nothing to its conflict.
- **For the coordinator:** seven of #8550's children (#8551, #8680, #8712, #8778, #8809, #8840, #8854, at the heads in
  the table) hold #8550's old head `73dd43836579` and not `54805be6d6b5` (`git merge-base --is-ancestor`, each), by
  his word. If one lands before #8550 it brings the old `tests/test_evaluate.py`. #8775 (`7d6a71567da5`) holds
  `54805be6d6b5`; its mark is his, and the queue shows it ready at that head (read at 4:06 AM).
- **Stale on `main`, placed by the coordinator and not mine:** the Modules row of
  `benchmarks/training_slowdown/README.md` names `capture.lower`, and `benchmarks/training_slowdown/pyproject.toml`'s
  comment names it too, with `adapters/vllm/verity_vllm` among its `inputs`, which nothing in the package or its tests
  imports since the move archived `capture/lower.py` (the grader's suite does read it): both to the one-calculator
  change. `run.py`'s docstring says a checkpoint's weights are written with `--keep-checkpoints` only, and
  `Harness.checkpoint_state` also writes the last planned checkpoint's; and from #8680 on the run has no
  `--keep-checkpoints` and writes no `checkpoints/`, so #8606's Writes clause on them goes stale when #8680 lands:
  both to the harness's next change, above #8680.
- **For the bounds lane:** #8490, crossed by ci, is bounds' (the coordinator's word). #8502, #8504, #8593, #8595, #8810
  and #8811 were never mine.
- **#8577** (`993844f9421b`): crossed by me, suites `r20261111-040341-f67d` 3 of 3; the queue shows it ready, by a
  mark that is not mine.
- **Not known:** why a forward pass under bf16 autocast on a CPU is not bit-reproducible even on one thread under
  load. The coordinator ruled not to spend on it; the sweeps and their logs are named in the 07:37Z note.
- **Local only, never pushed:** branch `cursor/evaluate-one-thread-dropped-d1d6` at `1ee1b6158e39` in `~/wt/h8550`
  (the one-thread pin that failed 1 run of 190); nothing reads it.
- **Proviso of the brief: YES.** Every input of #8541's crossed list (`7b7ae0802eab`) is in #8551's list at
  `0bb3841abce9`, the package's and the grader's `pyproject.toml` both, and so is every input of `main`'s
  (`86c03d1840cb`); read again after the mark.

