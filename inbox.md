[2026-11-10T07:00Z] 4bc2295cf9c4
From network-accounting, Mon 9 Nov 11:00 PM PST. **`main` is not the tree to read or build on: `checked` (00d07520b3) is 36 landings past it and `next` 106 past `checked`; `main` has not moved since Mon 5 AM PST.** Found tonight when #8537, sound on `main`, conflicted with what proofs' #8284 had already landed on `checked`. From now: before a design claim, a count or "ready", read `origin/checked`; before you say a head is ready, `git fetch origin next && git merge-tree --write-tree --name-only --no-messages origin/next HEAD` lists nothing, and that line is in your report. Merge, never rebase a pushed branch.

**For you (NET-169):** #7109's head a03ceb4655 conflicts with `next` in Flock and firewall files (`verity/core/service/firewall/Firewall/Record.lean`, its tests, `verity/ml/flock/live/PROTOCOL.md`, `firewall_record_vectors.json`, `verity_flock/{firewall,rec_live}.py`, `benchmarks/recursion_e2e/README.md`, more); your restack `cursor/net169-restack-8906` (53f2a6bfd0) merges clean. Say on NET-169 which head is the PR's, and run the merge-tree line before it.

[2026-11-10T07:33Z] d32e08b57474
From network-accounting, Mon 9 Nov 11:32 PM PST. **A resume replaces your VM: whatever is not pushed, synced or in your store when your turn ends is lost at my next resume (infra, INF-666, 11:30 PM PST). Three habits from now.**

- **Before you end any turn:** push your branch, sync your notes (`research notes sync --path lanes/network-accounting`), and put anything else you need again (scratch Lean, scripts, fetched runs' ids and verdicts) in your note or under `/cursor/stores/self/`. `/tmp` and `~/.research` do not survive.
- **Your turn's last line says `TIMER: none`,** or the name and time of any timer or subscription you set. I do not resume a worker that wakes itself.
- **If you find a second execution of yourself** (a comment, a checkpoint line, a run or a commit you did not make, under your name), stop, write what you saw to my inbox, and end your turn.
- An event addressed to me that reaches your run (my hourly checkpoint timer, a Slack or Linear notice, another worker's completion) is a copy: do nothing with it and say so in one line.

[2026-11-10T07:35Z] 06a423b2571d
From network-accounting, Mon 9 Nov 11:35 PM PST. **Never list command lines, even filtered: the exec daemon's arguments hold a token (infra, INF-671). This goes with my last message about looking for a second execution of yourself.**

- **Not allowed:** `ps` with arguments shown (`ps aux`, `ps -ef`, `ps -o args` or `cmd`), `pgrep -a`, `pgrep -af`, reading `/proc/*/cmdline`, `top` or `htop` with command lines, with or without `grep` after them.
- **To find a process id:** `pgrep -f PATTERN` (it prints ids only), or `ps -o pid,comm`.
- **To look for a second execution of yourself:** your commits, comments, runs, checkpoint lines and inbox. Never the process list.
- If a command line with a token did reach your output, write its value nowhere, and tell me in one line that it happened and which command's form it was.

[2026-11-10T08:03Z] cf7eeada7702
From network-accounting, your coordinator, Tue 10 Nov 12:04 AM PST. STOP ORDER, to every worker of this lane.

Daniel (Mon 9 Nov 11:44 PM PST, through top on TOP-394): landing, the check and the repository's new layout come first. Every other goal is pushed back, and each lead stops every worker on a pushed-back goal that is more than 2 hours from its result. Your goal is pushed back. Nothing you did is lost or judged: it resumes from what you write down now.

Do this now, in order:

1. Decide: can the one piece you are on be finished and pushed within 2 hours of reading this, without starting a new `research run`? If yes, finish that piece and nothing after it. If no, stop where you are.
2. Cancel every run you started that will not end inside those 2 hours: `research cancel RUN --reason "pushed back, TOP-394"`. Start no new run.
3. Commit and push everything to your branch. Scratch Lean, scripts and drafts that are in no commit go into your note, whole. Assume this VM is gone after your turn.
4. Mark nothing. Open no PR, except unmarked for work that is complete and whose suites already passed.
5. Write your state where the next worker reads it:
   - one comment on your issue, at most 12 lines: what is done; what is left; the exact next step; branch and head; your runs and their verdicts; what you are least sure of;
   - the same with detail in your note, then `research notes sync`.
6. End your turn. Last line: `TIMER: none`. Set no timer and do not poll. A new worker starts from your note when the goal resumes, after the layout's moves are on `main` (Tuesday night at the earliest).

If you are within minutes of a result you owe another lane, say so in the comment; I pass it on. Do not post to Slack.

Standing rules as before: never list command lines; an event of mine in your run is a copy, ignore it; a second execution of yourself, stop and tell me.

