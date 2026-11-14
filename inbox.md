[2026-11-14T18:12Z] a9253a65bfb7
circuits lead, Sat 14 Nov 18:13Z (10:13 AM PST). #10075 landed in `main` at 18:11Z (ci's notice names d3b88c2da5a7). #10160 is based on it: merge today's `main` into `cursor/circuits-cir233-silent-excepts-6f04`, after checking `git log HEAD..origin/main` for a `Move:` or `Layout-move:` line your branch lacks (if there is one, carry against it first with `python3 tools/move/restack.py BRANCH --main MOVE`). Then your suites and merge checks at the merged head, and your mark.

