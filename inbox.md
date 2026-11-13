[2026-11-13T22:15Z] 71854eed8a2e
From the proofs coordinator, Fri 13 Nov 22:15Z. A rule of Daniel's is on `main` since 21:01Z (the team skill, #9779): the repository is not an inbox. Write no brief or message to a branch. From this message on:

1. Your entries are notes. Do not append to `inbox/proofs-coordinator.md` any more. Write each checkpoint, report and ready line as its own file, `~/.research/notes/lanes/proofs/<YYYYMMDDTHHMMZ>-report-<your name in lower case>-<slug>.md`, with the notes' front-matter, as your checkpoint notes already are. Its first line after the front-matter is the one-sentence title you would have given the entry. Publish it with `research notes sync --path lanes/proofs`. I read the lane's notes at every sweep, every 30 minutes.
2. A ready line is a note whose slug starts with `ready-<PR number>`, and it is also the last message of that turn.
3. Read no mailbox on `cursor/proofs-inbox`: I write none from now. My word reaches you by the shell hook while you run, and by a resume when your turn has ended and you hold no timer of your own. If you hold a timer, the hook's message is shown at your next command after it fires.
4. If `research notes sync` fails, keep the file, try again at your next step, and put the note's text in your turn's last message. Append nothing to a branch instead.
Everything else in your order stands. Say in your next note that you have read this.

For you, ADDRROW2 (your 21:51Z is read; I do not resume you, since you hold timers):
- Mend 1 stays in #9824, as soundness asked. Go on as your section 3 ends: when the try passes, the records run with `--update`, and commit the lock it writes.
- Add one lemma beside the new `up`: that it equals the old text, `if even then H (a ++ s) else H (s ++ a)`, stated with the old right-hand side written out. It lets each owner see that the function did not change.
- The body lists what moves: `up`'s record, `upIn` new, the three modules' digests, the 83 readers by owner and the 4 deployed ones, with the run's own list in place of each [guess].
- The change crosses owners (lean 6, network-accounting 2, soundness 1, compute-accounting 1). I tell them on PRF-490 today. #9824 is marked when its ready line is in and either all four have said no objection or Mon 16 Nov noon PST has passed.

