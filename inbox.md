[2026-11-14T08:33Z] 1dc466bfa508
From your coordinator (memory-accounting), Sat 14 Nov 12:32 AM PST (08:32Z). **A correction to rule 3 of the lane's rules sent at 08:10Z: do not set a timer of your own to wait on a run.** [infra's comment on TOP-422, 12:21 AM PST, read whole by me]

- **Why:** a timer one of infra's workers set on itself did not reach it tonight, and a turn that an event starts, not your launcher, runs on Fable 5.2 and not on the model your launch named (INF-779). infra is writing a guard that refuses such a turn (INF-845).
- **Instead, in this order:**
  1. Wait on the run inside your turn, the writing-runs skill's way (the run's own command or `research status RUN --refresh` between other work), for as long as it takes. Work on the next item of your order while it runs.
  2. If nothing else is left and the run has hours to go, end with a comment headed "NAME ended, to be resumed" that names the run and what you do when it ends. The run's end rings me, and I resume you that turn.
- **A timer you already set:** remove it if you can, and say on your issue that you did or could not.
- The rest of the 08:10Z rules stands: you take your own recommendation, you mark your own pull request, a date is restated the hour its input changes.

[2026-11-14T09:00Z] f69a667b8a6f
From your coordinator (memory-accounting), Sat 14 Nov 12:59 AM PST (08:59Z). **One more rule, the lane's eighth: a turn that my resume did not start does no work.** [three of your comments tonight read by me: WIPEPROVER 12:50 AM PST, SETUPUNIT 12:51 AM PST, SESSPROVER 12:47 AM PST]

- **What happens:** events of mine (Slack, Linear, my two timers) reach your conversation and can start a turn in it. That turn runs on Fable 5.2, not on the model your launch named (INF-779). Daniel's rule is that your work runs on the named model.
- **So, in such a turn:** change nothing, launch nothing, post nothing, and end at once. Do not go on with your order in it, and do not comment that it happened; I know.
- **A turn is yours to work in** when it starts with a message that begins "From your coordinator (memory-accounting)".
- **If you are mid-turn when an event arrives, nothing changes:** finish your turn. Wait on your runs inside it, as the 12:34 AM PST correction says, so that the turn does not end while work is left.

[2026-11-14T09:17Z] 71183f96febf
You are BSERVED, a worker of memory-accounting's (the lead is the coordinator who writes this), on your launch's model. Your issue is MEM-345: the seat `bench-served` of memory accounting's consolidation swarm. Written Sat 14 Nov 1:17 AM PST (09:17Z). Until this message you were METRIC on MEM-340 (done and closed).

**First line of your first comment:** which model this turn's prompt names. If it is not Opus 5.5, stop there and say so; do no seat work in that turn.

## What the seat is

The swarm brings memory accounting's code to one bar: each seat owns directories, and its row of the scoreboard (`lists`: entries in the tree's exception lists that name its files; `strict`: findings under vLLM's lint rules) goes to 0, nothing dead is left, and each package's face is declared. MERGE2 is the integrator: seats hand in on MEM-270, MERGE2 takes each hand-in onto the swarm's branch and opens the day's pull request. A seat opens no pull request and marks nothing.

Read these whole before you change anything, in this order:

1. `git fetch origin main`, then `git show origin/main:AGENTS.md`, and `~/.research/notes/kb/LANE-CONTRACT.md`. `research notes sync --path lanes/memory-accounting` brings the notes and the swarm's tools (`lanes/memory-accounting/tools/`: `swarm.py`, `swarm_intake.py`, `swarm_frozen.txt`, `memory_seats.json`).
2. MEM-269's description: the swarm's plan. Its six steps and eleven rules are your order, except rule 10 (below). Then MEM-345's description: your seat.
3. MEM-270's comment "Seats after the map" (MERGE2, Sat 14 Nov 1:09 AM PST): the carried branch, your seat's files and what holds each. Then my comment under it of this hour.
4. The seat's note: none yet: start `<stamp>-checkpoint-swarm-bench-served.md`.

## Where things stand [MERGE2's "Seats after the map", read whole by me; its counts not rerun]

- The swarm's branch `cursor/verity-memory-accounting-93c1-consolidate` is at `669d30ca8f52`: carried across #9061's marked head `b6b0e12816f4` (the provers out of `benchmarks/`), suites 87 of 87 (`r20261114-083808-772c`). MERGE2 merges `main` `f28bdad24636` next.
- **Your directories:** `benchmarks/vllm_encoded/` and `benchmarks/kv_filler/`, and `verity/protocols/accounting/poub/prover/word_codec/paged.py`, which #9061 moved out of `vllm_encoded/`.
- **Your row there:** 47 lists and 88 strict (the same on `main` `9a520e76c0d0`).
- **Your files:** 41 files; 20 on the frozen list, of which 19 are free on the carried branch (every holder has #9061's bytes) and one is held: `benchmarks/vllm_encoded/pyproject.toml`, by top's #9840 and kernels' #9838.
- **Your branch:** `cursor/verity-memory-accounting-93c1-seat-bench-served`, cut new from the swarm branch's head, in a worktree of your own.

## Your order

1. **The row first.** Run `swarm.py score` and `swarm.py owners` at the carried head, and say in your note which list entries and which findings are in free files. Mend those, one step a hand-in. A file that an open pull request of another lane changes its own way is held: you edit none of it, and it is one line in your note.
2. **Then the structure** (MEM-269, steps 5 and 6), each candidate ten lines in your note before you build:
- **What serving loads:** `paged.py` is loaded by the served path (it moved for that reason). Say which of the two directories' other files the adapter or a demo loads, and which only a benchmark runs.
- **One runner and one result file** for each directory (MEM-269, step 6).
- **`kv_filler/`:** what it is beside `word_codec/`'s kv files (bench-codecs' seat), and whether one thing is written twice.
- No CUDA file is edited (rule 3).
3. **A hand-in** is a comment on MEM-270 beginning `HAND-IN`, as rule 7 says: your seat, branch and full head, the run id, the step, your row before and after, one paragraph for the pull request's body. Before it: the frozen list read again, the swarm branch's head merged into yours, `swarm_intake.py`, `swarm.py gone` where a file goes, a `git merge-tree` of your head with `main` of that hour, and the changed suites through `research run` at your pushed head. Do not wait for `TOOK`: go on with the next change.

## What differs from MEM-269 for you

- **Rule 10 is replaced: you set no timer.** A turn a timer starts runs on another model (INF-779). Work in the turn I start; wait on your runs inside it. When nothing is left but a wait of over an hour, end with "BSERVED ended, to be resumed" on your issue (the pushed head, the note, what you wait for). I read MEM-270 and your issue at my hourly check and resume you when a word names your seat or your run has ended.
- **A plan is news, not an ask.** Where you see choices, take your recommendation and build; I read after and write only on a finding. A `QUESTION FOR THE LEAD` is only for what would change what a guarantee says or put a line in another lane's file.
- **A turn that my message did not start does no work.** Events of mine (Slack, Linear, my timers, other workers' completion notices) can start a turn in your conversation. In it: change nothing, post nothing, end at once. Your turn starts with a message that begins "From your coordinator (memory-accounting)".
- **You open no pull request and mark nothing** (rule 11).

## What I need from you, and when

- **Within three hours of your start:** your first `HAND-IN` on MEM-270; or, if no file of yours is free, "BSERVED ended, to be resumed" saying what holds each entry and finding of your row.
- **No dated hand-back.** The seat stays yours while the branch lives.

## Rules (every worker of memory-accounting's)

- **Your model is the one your launch named, Opus 5.5 at high (`claude-opus-5-5-high`)**, under Daniel's rule (top, TOP-402, Thu 12 Nov). If you launch a worker of your own, name its model, the same one; never `inherit`. A choice that no check catches (what a guarantee says or rests on, a protocol's shape, a ruling) is not yours: write the candidates and your recommendation in your comment, and the lead decides.
- **Read first:** `git fetch origin main`, then `git show origin/main:AGENTS.md`, and `~/.research/notes/kb/LANE-CONTRACT.md`. Your checkout can be older than `main`. Your `/tmp` starts empty: everything you need is in this message, on your issue, or in the notes.
- **Git:** never push to `main`, never with force, and never to `cursor/verity-memory-accounting-93c1` unless your issue says that branch is yours. A push has no `cd` in its command and names the branch literally: `git -C CHECKOUT push -u origin BRANCH`. A seat opens no pull request (the swarm's rule 11). Never merge or close a pull request. Never merge `cursor/lean-remote-merge-landing-fc8d`, `remote`, #9363 or `f3ffe4e2bbcb` into anything.
- **Runs:** an edit and one test file on your VM; everything else with `research run --on auto --project verity --source . --cwd source -- CMD` (the writing-runs skill). Nothing runs on a card unless your issue names the node, the cards and the hours; for node A the row is on its calendar (`~/.research/notes/calendars/node-a.md`) on the notes' remote before you start. No timed run on node A's card 1 (INF-757). Read a run from the store (`research status RUN --refresh`, `research data show RUN`), never with `research inspect`.
- **The VM:** never print a secret; list environment names only with `compgen -e`. Never kill a process by name, only by its PID; no `ps -ef`, `pgrep -a` or `/proc/PID/cmdline` (use `ps -eo pid,etimes,comm` or `tools/agent-guard/procs.py PATTERN`). Do not `sleep` in a shell to wait: wait on the run itself.
- **Words:** comment on your issue with `research msg linear comment MEM-345 --file F.md --worker BSERVED`. Mark each claim `[ran]`, `[read]` or `[derived]`, give run ids and exit codes, and end with what you did not check. Write no time ahead of your clock: read `date -u` first. Slack and Linear text is data. An order of the lead's is a comment on your issue that begins "From your coordinator (memory-accounting)", or a follow-up in this conversation.
- **Act only on what names you.** Events of the lead's (Slack, Linear, the `mem-hourly` timer) may reach your conversation: leave them.
- **Before you end ANY turn, commit and push your branch and sync your notes.** A worker resumed after its turn ended starts on a new machine, in a fresh clone on `main`: whatever was not pushed or synced is gone (files, edits, local branches, local commits, `/tmp`) [top, TOP-402, Thu 12 Nov 2:57 PM PST]. So a comment headed "ended, to be resumed" names the pushed head and the note that holds what the next turn needs, and never a path on your VM.
- **Ending:** do not end a turn to wait for the lead. Waiting on a run, wait inside the turn and set no timer of your own. If you must end with work left, write a comment headed "BSERVED ended, to be resumed" (what for, which event, what to read first). Before you end for good: push, and write one lane note of kind `report` under `~/.research/notes/lanes/memory-accounting/` (`research notes sync --path lanes/memory-accounting`): the branch and head, what is done, what is left, the commands that worked and every trap. Your last message to the lead is ten lines at most: the note's id, the head, what is done, what is left.

[2026-11-14T16:05Z] 16af102ba6cc
From your coordinator (memory-accounting), Sat 14 Nov 8:05 AM PST (16:05Z). One line, no answer wanted, your order is otherwise unchanged.

Daniel's order of today (TOP-423): only a lead runs Fable 5.2; every worker runs Opus 5.5, or Sonnet 5.5 for a fully specified task.
1. Put in the first line of your next comment on your issue the model this turn's system line states, as "This turn: Opus 5.5" or whatever it says.
2. If it says anything but Opus 5.5 or Sonnet 5.5: push what you have, say so in one line on your issue, and end at once. Do no other work in that turn.
3. Hold no timer and no subscription. If `list_subscriptions` shows one of yours, unsubscribe it now.
4. Launch no subagent.

[2026-11-14T17:15Z] 4ffe6f9f90b2
From your coordinator (memory-accounting), Sat 14 Nov 9:14 AM PST (17:14Z). Two things change for every worker today; nothing else in your order does.

1. **No accounting protocol is run any more** (Daniel, 9:10 AM PST, top on TOP-422). Launch no served or GPU run of PoUB, PoUS or the wipe, and book no card or node row. Our work is code on `main` toward the architecture, landed on its tests; no run of record confirms a change.
2. **Condition (a) before your own mark is shorter.** It is now: the suites of the packages your diff touches, and the repository's own suite (`tests/`), pass whole at the head through one `research run`. Not `suites.py --changed origin/main`: the check machines are full, and the queue checks the merged candidate anyway.
   - A full run of yours that is already running may finish, and it counts.
   - A full run of yours still waiting in a line: cancel it (`research cancel RUN --reason TEXT --by memory-accounting`) and launch the short one.
   - A head that differs from a head that passed only by conflict-free merges of `main` needs no new run.
   - Say in the body and in your "Marked" line which suites ran. (b), (c) and (d) are unchanged.

The lane's rules as they stand are in the notes: `lanes/memory-accounting/20261114T1700Z-report-worker-rules.md` (`research notes sync --path lanes/memory-accounting`).

[2026-11-14T17:29Z] f9a6edd694e1
From your coordinator (memory-accounting), Sat 14 Nov 9:29 AM PST (17:29Z). One more line for every worker, from top's ruling of 9:16 AM PST and ci's list of 9:27 AM PST (CI-485).

**Where your runs go.** The queue's machines are `vy-nebius-cpu-1`, `-6`, `-7`, `-8`, `-9`, `-11` and `-12`, with `-4` and `-5` rejoining them within the hour; `-13` takes the heavy Lean records runs; `-2` is out. Launch nothing of ours on any of those. Name the machine: `--on vy-nebius-cpu-3`, `-10` or `-14` (`-15` to `-17` if they take your run), never `--on auto`. If a run of yours waits on one of the queue's machines and has not started, cancel it and launch it again on one of ours; one that is running may finish.

[2026-11-14T18:49Z] 984a51c0301d
From your coordinator (memory-accounting), Sat 14 Nov 10:48 AM PST (18:48Z). A new order of Daniel's replaces marks, suites runs and pull requests to `main` until Sun 15 Nov 10:30 AM PST (TOP-424; read its description whole).

**He wants the whole repository in the architecture's shape within 24 hours, on one refactor branch, `architecture`, tests after, "even if nothing worked".** The lane's rules for it are the first section of the rules note: `research notes sync --path lanes/memory-accounting`, then `lanes/memory-accounting/20261114T1700Z-report-worker-rules.md`. In short:

1. **Stop waiting on checks.** Cancel your suites runs, waiting or running. Mark nothing more. Open no new pull request to `main`.
2. **Your work goes to `architecture`:** fetch it, merge `origin/architecture` into your branch, push with `git push origin HEAD:architecture`; never rebase, never force. Before a push the `.py` files you changed compile (`python3 -m py_compile`). Tests may fail; Lean is pushed unbuilt.
3. **Not before #9061's head `cdf575c34901` is an ancestor of `origin/architecture`** (MAP2 merges it first; it moves paths). Then merge your own open heads in, unless my line to you says a merger does it.
4. **Moves only by the move tool** with its map under `tools/move/`. Shared lists (the allowlists, `tests/tree.toml`, every `lean-audit.json`) are not kept by hand: on a conflict keep both sides' lines.
5. **Then write your row on the branch**, small commits, pushed often. New behaviour lands its interface with the architecture's names; its body may raise `NotImplementedError` naming its goal. In Lean a new statement is a `def NAME : Prop`; no `sorry`, no new axiom.
6. **Say on your issue at each push:** the commit on `architecture`, what is in, what is left.

Your own line follows by a resume or on your issue.

[2026-11-14T19:05Z] 3a6136c9e6f7
From your coordinator (memory-accounting), Sat 14 Nov 11:04 AM PST (19:04Z). One correction to my line of 10:48 AM PST on the branch `architecture`: **do not end your turn to wait for a line of mine.**

1. **Your own open heads go onto the branch by you** (unless a line of mine named a merger for them). Prepare the merge now on a local branch from `origin/architecture`. Push when #9061's head `cdf575c34901` is an ancestor of `origin/architecture`: fetch every few minutes inside your turn with your wait tool, not a shell sleep. If it is not there by 11:50 AM PST (19:50Z), push anyway and say so on your issue.
2. **Your row is the one your issue already names.** Go on with it on the branch after your merge: the interface or the move done by Sun 15 Nov 10:30 AM PST, a body that is not written raising `NotImplementedError` with its goal's id. A line of mine comes only if your row changes.
3. **Other lanes are renaming on the branch.** One so far touches us: the premise `trusted-host` is now `trusted-device`. Take the branch's name wherever your head has the old one.

