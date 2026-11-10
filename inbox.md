[2026-11-10T17:51Z] 765ae893451e
From the coordinator, Tue 10 Nov 9:51 AM PST. **CROSS: your job on `main` as it is.** The layout is not today and the map will change, so the rehearsal stops. What still pays: every open pull request of ours that lands before the map is final never crosses. Marked [read] where I read it at 9:50 AM PST.

**Part 1, one comment on NET-311 by 10:30 AM PST (the hour stands).** For each of our 12 open PRs, one row:
- its state in the queue (`uv run -q research queue status --json`, the entry whose `pr` is the number). [read] admitted: #8433, #8459, #8547, #8579. Refused: #8040 ("conflicts with `main` in `verity/core/protocols/certifier/PROTOCOL.md`: merge `main` into the branch and re-mark it") and #8415 (ejected: it waits for #8303 and #8040 to land). Not marked: #7109, #8405, #8477, #8499, #8537, #8557.
- `git merge-tree --write-tree --name-only --no-messages origin/main HEAD` after `git fetch origin main`: exit 0 clean, 1 with the files that stop, 124 under `timeout` is NO RESULT (say so; remove any `.merge_file_*` it leaves).
- Under the table, at most eight lines: what the rehearsal found on the staging that was dropped (clean, stopped, each mend branch and its head), headed "against a staging that was dropped; kept for the final map".

**Part 2, by 1 PM PST: #8040 to a head that merges `main`.** On a new branch `cursor/net311-main-8040-<4 hex>` cut from the PR's head 069df3e007: `git merge origin/main`, close `PROTOCOL.md` and anything else that stops (reuse your mend's resolution where the conflict is the same one), then ONE run of the suites the change touches at that head (`suites.py --changed origin/main --quick` by `research run`; not on cpu-10 before 11:35 AM PST). Send me the head and the run id with its verdict. I fast-forward the PR's branch to it and mark it again.

**Part 3, then, in this order: #8405, #8499 (it holds #8405), #8557, #8477, #7109.** For each: if it merges `main` clean AND its issue (NET-281, NET-283, NET-100, NET-169) holds a suites verdict at its present head, say "markable as it stands" with the run id; else the same as part 2 on `cursor/net311-main-<PR>-<4 hex>`, one run a head. #7109 also needs ci's and lean's word on its `public_boundary.toml` entry: send me the entry's lines and I ask.

**Not yours:** #8537 (CARRYLOCK's), #8547 (SCANLEAN's), the four admitted (a marked PR takes no push), #8415.

**As before:** never push over a PR's branch, never mark, eject or close. One inbox message to me at the end of each part. You end when part 3's list is done or at 4 PM PST, whichever comes first.

