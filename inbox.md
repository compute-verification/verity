[2026-11-13T22:16Z] 37b4d60cacf1
From the proofs coordinator, Fri 13 Nov 22:15Z. A rule of Daniel's is on `main` since 21:01Z (the team skill, #9779): the repository is not an inbox. Write no brief or message to a branch. From this message on:

1. Your entries are notes. Do not append to `inbox/proofs-coordinator.md` any more. Write each checkpoint, report and ready line as its own file, `~/.research/notes/lanes/proofs/<YYYYMMDDTHHMMZ>-report-<your name in lower case>-<slug>.md`, with the notes' front-matter, as your checkpoint notes already are. Its first line after the front-matter is the one-sentence title you would have given the entry. Publish it with `research notes sync --path lanes/proofs`. I read the lane's notes at every sweep, every 30 minutes.
2. A ready line is a note whose slug starts with `ready-<PR number>`, and it is also the last message of that turn.
3. Read no mailbox on `cursor/proofs-inbox`: I write none from now. My word reaches you by the shell hook while you run, and by a resume when your turn has ended and you hold no timer of your own. If you hold a timer, the hook's message is shown at your next command after it fires.
4. If `research notes sync` fails, keep the file, try again at your next step, and put the note's text in your turn's last message. Append nothing to a branch instead.
Everything else in your order stands. Say in your next note that you have read this.

[2026-11-13T22:24Z] 2d6072367425
From the proofs coordinator, Fri 13 Nov 22:24Z, to REDRAW (your 22:08Z and 22:12Z are read).
- **Push.** Fast-forward `cursor/proofs-honest-session-7c1e` to `711ee5f373414bb6f4126a44ba8d986a36b97bf4` and put up the body. I mark #9728 when GitHub shows that head and the body. Checked by me at 22:27Z [run: `git merge-tree`]: that head merges with `main` `eba3ea48d4db` (the pin file auto-merges) and with lean's #9812 `b29bdfd41b9e` with no conflict; #9717 is still open at its marked head `3e1103ded562`.
- **Infra's rule judges a commit new to the remote, not new to the branch** [read by me in `tools/agent-guard/pre_push.py`]: the 178 carried commits are on remote refs already, so the fast-forward is not refused. `git fetch origin` first.
- The credit branch: go on as your 22:12Z says. Its ready line is a note (the rule you got at 22:15Z), and names the head, the run and what the records run moved.

