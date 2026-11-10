[2026-11-10T02:07Z] 4505726a56df
From network-accounting, for the NET-289 job. The standing items still hold, in case the new brief read as replacing them:

- Each hour and before each commit: `python3 tools/agent-guard/inbox.py pull bc-07111ee9-6238-5314-a1fc-1e98e3ac1511 --repo <your worktree>` and read `/tmp/agent-inbox/bc-07111ee9-6238-5314-a1fc-1e98e3ac1511.md`.
- Your handoff file is `/opt/cursor/artifacts/code-tour-author-handoff-NET-289-CARRYEDGE.md`, never the shared `code-tour-author-handoff.md`.
- Delete no fetched run copy under `~/.research/runs` unless `research data preserved RUN` says preserved. Quote nothing of Monday's case (its window's number and counts) in any text.
- The whole brief is also at `/cursor/stores/self/briefs/carryedge-net289.md`.
- One addition to it: if a README or a docstring outside the declarations you change states that a window's operator reads only that window's records, rewrite it in the same PR, and list the places in the PR body.

[2026-11-10T02:18Z] 5c9808e2bd45
From network-accounting, on NET-289. Soundness answered (PRF-488): the change is right, go ahead. NET-289's description now carries it; reread it. Two conditions to keep, in the statement and said in the docstrings:

1. **The operator is fixed before the period.** `op` is quantified before the secret (before `Zs`, `schedOf`, `srvOf`), as your draft has it. It is a function of records, never of the secret itself.
2. **It reads only closed windows.** `op i` reads the earlier windows through your `hop` (windows `j < i` only), and its own window only through the records before step `t`, as on main. Nothing lets it read a window still open or a later one.

If the Lean check wants `hop` as a named `Prop`, name it and list it as `domain`, with a one-line why (the lean-proofs skill). Soundness's words for why the count holds: the operator's choice may depend on what earlier windows already put on the wire, which the count already charged, so the product bound holds by induction and the old statement is the special case.

Out of scope stays out: the rule's own Lean statement (your section 1) is NET-291, undated; the horizon is NET-292, undated. #8459's follow-on (the stamps in the signed records) is NET-290, not yours for now.

