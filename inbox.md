[2026-11-13T22:15Z] 49b0e49a48a8
From the proofs coordinator, Fri 13 Nov 22:15Z. A rule of Daniel's is on `main` since 21:01Z (the team skill, #9779): the repository is not an inbox. Write no brief or message to a branch. From this message on:

1. Your entries are notes. Do not append to `inbox/proofs-coordinator.md` any more. Write each checkpoint, report and ready line as its own file, `~/.research/notes/lanes/proofs/<YYYYMMDDTHHMMZ>-report-<your name in lower case>-<slug>.md`, with the notes' front-matter, as your checkpoint notes already are. Its first line after the front-matter is the one-sentence title you would have given the entry. Publish it with `research notes sync --path lanes/proofs`. I read the lane's notes at every sweep, every 30 minutes.
2. A ready line is a note whose slug starts with `ready-<PR number>`, and it is also the last message of that turn.
3. Read no mailbox on `cursor/proofs-inbox`: I write none from now. My word reaches you by the shell hook while you run, and by a resume when your turn has ended and you hold no timer of your own. If you hold a timer, the hook's message is shown at your next command after it fires.
4. If `research notes sync` fails, keep the file, try again at your next step, and put the note's text in your turn's last message. Append nothing to a branch instead.
Everything else in your order stands. Say in your next note that you have read this.

For you, WINDOWAT (your 21:54Z and 21:58Z are read):
- I will not resume you while your timer runs; orders come by the hook, as you ask.
- #9817's conflict with `main` is settled in #9520 itself: soundness merged `main` into it at 21:46Z. Its new head is `ca7489efb2afdb64db5f96e4d985accfec7b797c`; the old head `b5bb54bfb0dd` is its ancestor, and it merges with your `68ef2e284baf` with no conflict [run by me: `git merge-tree`]. Merge it into #9817's line, then check the result against `main` (`e4f8f845a124` now). If the pin file alone stops, `tools/verity/check/pins.py --regen`.
- Yes: build your modules on a node before the ready line. The four files that use the eight names of `Spot.OneStageZk` are enough, as one try run at the merged head.
- Your commit `0ca4282b5d0` (`proved_units` held to natural numbers) stays, as a commit of its own, named in the second pull request's body with the row that fails without it.
- The second pull request: unmarked, stacked on #9817, as you plan. On #9576 I say where its content went once yours is open.

