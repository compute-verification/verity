[2026-11-10T10:52Z] e8a2b81d401c
PATHJOINS (NET-306), Tue 10 Nov 2:53 AM PST. **PR #8636 is open, unmarked, head 7e8fffb3ba. Not ready to mark: the suites run has no verdict yet.**

- 42 of 42 fixed, 42 of 42 name the same path on main [run]; table in the body.
- Evaluator on the one map at 539f956d62 (`tools/move/layout_one_map.toml`, holds row 25): ours 22 on unchanged main, 0 with the branch; all owners 132 to 110 [run]. Mover without `--report-splits` on the branch: rc 0 [run].
- Suites: one run, `r20261110-095624-6a25`, 55 suites (`--changed a337d86e48`, slow tests included), still running. I add its verdict to the body, then send the comment on NET-306 and my report.
- `queue open` warned: NET-306 has no goal label, so the body's `Goal network/slice-clean` matches no label of the issue. Yours to label.

