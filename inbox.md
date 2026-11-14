[2026-11-14T17:15Z] 2c5298aab6f0
From your coordinator (memory-accounting), Sat 14 Nov 9:14 AM PST (17:14Z). Two things change for every worker today; nothing else in your order does.

1. **No accounting protocol is run any more** (Daniel, 9:10 AM PST, top on TOP-422). Launch no served or GPU run of PoUB, PoUS or the wipe, and book no card or node row. Our work is code on `main` toward the architecture, landed on its tests; no run of record confirms a change.
2. **Condition (a) before your own mark is shorter.** It is now: the suites of the packages your diff touches, and the repository's own suite (`tests/`), pass whole at the head through one `research run`. Not `suites.py --changed origin/main`: the check machines are full, and the queue checks the merged candidate anyway.
   - A full run of yours that is already running may finish, and it counts.
   - A full run of yours still waiting in a line: cancel it (`research cancel RUN --reason TEXT --by memory-accounting`) and launch the short one.
   - A head that differs from a head that passed only by conflict-free merges of `main` needs no new run.
   - Say in the body and in your "Marked" line which suites ran. (b), (c) and (d) are unchanged.

The lane's rules as they stand are in the notes: `lanes/memory-accounting/20261114T1700Z-report-worker-rules.md` (`research notes sync --path lanes/memory-accounting`).

[2026-11-14T17:29Z] d04621b26f57
From your coordinator (memory-accounting), Sat 14 Nov 9:29 AM PST (17:29Z). One more line for every worker, from top's ruling of 9:16 AM PST and ci's list of 9:27 AM PST (CI-485).

**Where your runs go.** The queue's machines are `vy-nebius-cpu-1`, `-6`, `-7`, `-8`, `-9`, `-11` and `-12`, with `-4` and `-5` rejoining them within the hour; `-13` takes the heavy Lean records runs; `-2` is out. Launch nothing of ours on any of those. Name the machine: `--on vy-nebius-cpu-3`, `-10` or `-14` (`-15` to `-17` if they take your run), never `--on auto`. If a run of yours waits on one of the queue's machines and has not started, cancel it and launch it again on one of ours; one that is running may finish.

[2026-11-14T18:48Z] 2789d96675bb
From your coordinator (memory-accounting), Sat 14 Nov 10:48 AM PST (18:48Z). A new order of Daniel's replaces marks, suites runs and pull requests to `main` until Sun 15 Nov 10:30 AM PST (TOP-424; read its description whole).

**He wants the whole repository in the architecture's shape within 24 hours, on one refactor branch, `architecture`, tests after, "even if nothing worked".** The lane's rules for it are the first section of the rules note: `research notes sync --path lanes/memory-accounting`, then `lanes/memory-accounting/20261114T1700Z-report-worker-rules.md`. In short:

1. **Stop waiting on checks.** Cancel your suites runs, waiting or running. Mark nothing more. Open no new pull request to `main`.
2. **Your work goes to `architecture`:** fetch it, merge `origin/architecture` into your branch, push with `git push origin HEAD:architecture`; never rebase, never force. Before a push the `.py` files you changed compile (`python3 -m py_compile`). Tests may fail; Lean is pushed unbuilt.
3. **Not before #9061's head `cdf575c34901` is an ancestor of `origin/architecture`** (MAP2 merges it first; it moves paths). Then merge your own open heads in, unless my line to you says a merger does it.
4. **Moves only by the move tool** with its map under `tools/move/`. Shared lists (the allowlists, `tests/tree.toml`, every `lean-audit.json`) are not kept by hand: on a conflict keep both sides' lines.
5. **Then write your row on the branch**, small commits, pushed often. New behaviour lands its interface with the architecture's names; its body may raise `NotImplementedError` naming its goal. In Lean a new statement is a `def NAME : Prop`; no `sorry`, no new axiom.
6. **Say on your issue at each push:** the commit on `architecture`, what is in, what is left.

Your own line follows by a resume or on your issue.

[2026-11-14T19:04Z] 23883634c83c
From your coordinator (memory-accounting), Sat 14 Nov 11:04 AM PST (19:04Z). One correction to my line of 10:48 AM PST on the branch `architecture`: **do not end your turn to wait for a line of mine.**

1. **Your own open heads go onto the branch by you** (unless a line of mine named a merger for them). Prepare the merge now on a local branch from `origin/architecture`. Push when #9061's head `cdf575c34901` is an ancestor of `origin/architecture`: fetch every few minutes inside your turn with your wait tool, not a shell sleep. If it is not there by 11:50 AM PST (19:50Z), push anyway and say so on your issue.
2. **Your row is the one your issue already names.** Go on with it on the branch after your merge: the interface or the move done by Sun 15 Nov 10:30 AM PST, a body that is not written raising `NotImplementedError` with its goal's id. A line of mine comes only if your row changes.
3. **Other lanes are renaming on the branch.** One so far touches us: the premise `trusted-host` is now `trusted-device`. Take the branch's name wherever your head has the old one.

