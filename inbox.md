[2026-11-10T02:07Z] 4505726a56df
From network-accounting, for the NET-289 job. The standing items still hold, in case the new brief read as replacing them:

- Each hour and before each commit: `python3 tools/agent-guard/inbox.py pull bc-07111ee9-6238-5314-a1fc-1e98e3ac1511 --repo <your worktree>` and read `/tmp/agent-inbox/bc-07111ee9-6238-5314-a1fc-1e98e3ac1511.md`.
- Your handoff file is `/opt/cursor/artifacts/code-tour-author-handoff-NET-289-CARRYEDGE.md`, never the shared `code-tour-author-handoff.md`.
- Delete no fetched run copy under `~/.research/runs` unless `research data preserved RUN` says preserved. Quote nothing of Monday's case (its window's number and counts) in any text.
- The whole brief is also at `/cursor/stores/self/briefs/carryedge-net289.md`.
- One addition to it: if a README or a docstring outside the declarations you change states that a window's operator reads only that window's records, rewrite it in the same PR, and list the places in the PR body.

