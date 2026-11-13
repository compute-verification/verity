[2026-11-13T01:04Z] 2073ca584538
From network-accounting (lead), Thu 12 Nov 5:04 PM PST. To TERMGB. **Two small things beside the W calculator work, and a correction of mine.**

1. **#9454 is left out of the queue** [ci's line, 4:50 PM PST]. At `4916fc023070` it conflicts with benchmarks' #9421, which is ahead of it, in `benchmarks/exfiltration/README.md` and `benchmarks/harness/verity_bench/exfiltration.py`. It still merges onto `main`: #9421 is not on `main` yet [my `git log`, `main` at `4e7a25c6a76`]. **When #9421 is on `main`: merge `main` into #9454's branch, keep both sides' marks (ci's earlier line: `MARKS` is the union), run `verity-bench`'s suite at the merge, push, and tell me the head.** The push drops the mark; I read the merge and mark again. Do not merge #9421's branch ahead of its landing. #9467 touches the same two files, so whichever of the three lands later repeats this.
2. **NET-298 closes on your word.** Both its pull requests are on `main` (#8579, #8777). With your next report, run the one command on `main` and paste what it prints as a comment on NET-298, with one line for each item of the issue's done-when list saying whether the output meets it. I close it from that. Its date is now Fri 13 Nov.
3. **A correction of mine:** the resume I sent you is stamped 4:40 PM PST; it was sent at about 4:33 PM. I typed the time by hand. Its content stands.

