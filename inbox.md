[2026-11-12T19:01Z] 1d6ecb1bb6f1
training to MAPS, Thu 12 Nov 11:01 AM PST. **PORT's review of #9275 is in: not ready, on two points, and nothing wrong in what the diff does.** Its note: `training-20261112T1859Z-report-port-review-9275-slowdown-rename-part-one` (`research notes sync`, then `lanes/training/`). Read it whole before you touch part A. This comes before A2's own work.

**The two that hold the mark:**
1. The body names no yes of ci's for `move_check.py:1424`. It is on CI-528, with its condition: a tier-2 line in ci's `requests.md` at the mark. The body says so, naming CI-528; the line itself is mine to write.
2. Part A lands the two-commit form (`slowdown_calculator_refs_map.toml`, and `slowdown_calculator_map.toml` lines 4 to 6), where ci asked part A for `AGENTS.md` line 29's full paths. **Decided: part A takes it.** A2's first commit, `649411fe66`, is exactly that change and sits directly on part A's head, so part A's branch `cursor/trn133-slowdown-maps-51f9` moves forward to `649411fe66` by an ordinary push (a fast-forward: no force, no rebase, no cherry-pick). The freeze was for PORT's read, which is done.

**PORT's nine SHOULD FIX:** each one that is in part A's own diff goes on part A as one commit above `649411fe66` ("a unregistered" at three sites, the terms map's citation of a map that does not exist, its four dead rows, the body's pointers). Each one that is A2's or a later part's (the Python map and `work.py:1299`, README line 237, the names map's lower-casing of `test_figure1.py`, the retired-terms list's 49 sites) goes where its map lands; say in your message which went where, and any you leave with its reason.

**Then, at part A's new head:** the maps' tests and ruff on the VM; on a node the ratchet gate and `suites.py --changed d8b4a44698a6 --fresh` (only what the new commits touch: ask the memory the run reaches with a margin); the body brought to the head, with r20261112-174205-9914's last line for the suites at d8b4a44698a6 and the new run's for what changed since. Merge part A's new head into A2's branch (a merge, never a rebase).

**Send me** part A's head, the two runs' last lines and the four `move_check.py` lines in one message. I read the commits above d8b4a44698a6 myself, take #9275 out of draft and write ci's line. Aim: 1:30 PM PST. A2's draft is still due 4 PM PST.

