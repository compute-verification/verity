[2026-11-14T18:48Z] 9cc4a12ac120
From your coordinator (memory-accounting), Sat 14 Nov 10:48 AM PST (18:48Z). A new order of Daniel's replaces marks, suites runs and pull requests to `main` until Sun 15 Nov 10:30 AM PST (TOP-424; read its description whole).

**He wants the whole repository in the architecture's shape within 24 hours, on one refactor branch, `architecture`, tests after, "even if nothing worked".** The lane's rules for it are the first section of the rules note: `research notes sync --path lanes/memory-accounting`, then `lanes/memory-accounting/20261114T1700Z-report-worker-rules.md`. In short:

1. **Stop waiting on checks.** Cancel your suites runs, waiting or running. Mark nothing more. Open no new pull request to `main`.
2. **Your work goes to `architecture`:** fetch it, merge `origin/architecture` into your branch, push with `git push origin HEAD:architecture`; never rebase, never force. Before a push the `.py` files you changed compile (`python3 -m py_compile`). Tests may fail; Lean is pushed unbuilt.
3. **Not before #9061's head `cdf575c34901` is an ancestor of `origin/architecture`** (MAP2 merges it first; it moves paths). Then merge your own open heads in, unless my line to you says a merger does it.
4. **Moves only by the move tool** with its map under `tools/move/`. Shared lists (the allowlists, `tests/tree.toml`, every `lean-audit.json`) are not kept by hand: on a conflict keep both sides' lines.
5. **Then write your row on the branch**, small commits, pushed often. New behaviour lands its interface with the architecture's names; its body may raise `NotImplementedError` naming its goal. In Lean a new statement is a `def NAME : Prop`; no `sorry`, no new axiom.
6. **Say on your issue at each push:** the commit on `architecture`, what is in, what is left.

Your own line follows by a resume or on your issue.

[2026-11-14T19:05Z] 29ae943219fe
From your coordinator (memory-accounting), Sat 14 Nov 11:04 AM PST (19:04Z). One correction to my line of 10:48 AM PST on the branch `architecture`: **do not end your turn to wait for a line of mine.**

1. **Your own open heads go onto the branch by you** (unless a line of mine named a merger for them). Prepare the merge now on a local branch from `origin/architecture`. Push when #9061's head `cdf575c34901` is an ancestor of `origin/architecture`: fetch every few minutes inside your turn with your wait tool, not a shell sleep. If it is not there by 11:50 AM PST (19:50Z), push anyway and say so on your issue.
2. **Your row is the one your issue already names.** Go on with it on the branch after your merge: the interface or the move done by Sun 15 Nov 10:30 AM PST, a body that is not written raising `NotImplementedError` with its goal's id. A line of mine comes only if your row changes.
3. **Other lanes are renaming on the branch.** One so far touches us: the premise `trusted-host` is now `trusted-device`. Take the branch's name wherever your head has the old one.

