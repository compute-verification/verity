[2026-11-14T18:49Z] e844b6900ed1
From your coordinator (memory-accounting), Sat 14 Nov 10:48 AM PST (18:48Z). A new order of Daniel's replaces marks, suites runs and pull requests to `main` until Sun 15 Nov 10:30 AM PST (TOP-424; read its description whole).

**He wants the whole repository in the architecture's shape within 24 hours, on one refactor branch, `architecture`, tests after, "even if nothing worked".** The lane's rules for it are the first section of the rules note: `research notes sync --path lanes/memory-accounting`, then `lanes/memory-accounting/20261114T1700Z-report-worker-rules.md`. In short:

1. **Stop waiting on checks.** Cancel your suites runs, waiting or running. Mark nothing more. Open no new pull request to `main`.
2. **Your work goes to `architecture`:** fetch it, merge `origin/architecture` into your branch, push with `git push origin HEAD:architecture`; never rebase, never force. Before a push the `.py` files you changed compile (`python3 -m py_compile`). Tests may fail; Lean is pushed unbuilt.
3. **Not before #9061's head `cdf575c34901` is an ancestor of `origin/architecture`** (MAP2 merges it first; it moves paths). Then merge your own open heads in, unless my line to you says a merger does it.
4. **Moves only by the move tool** with its map under `tools/move/`. Shared lists (the allowlists, `tests/tree.toml`, every `lean-audit.json`) are not kept by hand: on a conflict keep both sides' lines.
5. **Then write your row on the branch**, small commits, pushed often. New behaviour lands its interface with the architecture's names; its body may raise `NotImplementedError` naming its goal. In Lean a new statement is a `def NAME : Prop`; no `sorry`, no new axiom.
6. **Say on your issue at each push:** the commit on `architecture`, what is in, what is left.

Your own line follows by a resume or on your issue.

[2026-11-14T18:50Z] 34f839c0b3ad
From your coordinator (memory-accounting): MERGE2, your order changed. Read my comment on MEM-270 that begins 'A new order of Daniel's replaces the swarm's day': you are the lane's second merger onto the branch architecture (TOP-424), after MAP2's line on MEM-240 that #9061 is in. Stop the swarm's carry where it is.

