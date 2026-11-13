[2026-11-13T15:39Z] b19bfd3d1ce3
ONSITEMAIN, from the lead, Fri 13 Nov 7:39 AM PST. **Correction to my line of this morning: the second move chain IS on `main` now, at `4ae17f7eafd7dd42d56e166f04d060e96174d384` (#9734, landed 7:31 AM PST). The carry is open. Launch it now, and do the read I gave you while it runs.**

**The carry (TOP-415), the same for every branch of ours:**
- `git fetch origin main`; `main` must contain `4ae17f7eafd7` [rev-parse].
- One command a branch, from a checkout of the new `main`: `tools/move/restack.py BRANCH`, with `--dry-run` first. No rebase and no plain merge of `main`: a plain merge can be clean and wrong, since only the move's own script renames a moved import in a file the branch adds.
- Where the tool's first step conflicts (the tree before the moves), merge the base it names by hand, push, then run the tool.
- The outcome must be `clean`. On `clean-skipped`: look for the skipped map's old names in what the branch adds, then run the confirming command the tool prints. On `locks-stale`: regenerate with `check.py --update`. On anything else (`conflicted`, `no-row`, `script-failed`, `pins-stale`, `based-after`): stop on that branch and send me the tool's words.
- The tool's last step builds Lean, 8 to 9 GB of scratch a branch. Take scratch from `mktemp -d -p /workspace/research/scratch`, removed by a trap, and run at most three branches abreast on one node. Or use `--no-pins` and one `tools/verity/check/pins.py --regen` after.
- Push the result to the pull request's own branch as a fast-forward, never a force push. Then one run of the suites changed since the merge base at the carried head, and for a Lean change its records run.
- Then the body: the carried head, the run, and no path or name the moves retired.
- **You mark nothing.** I read each carried head and mark it. Push nothing to a pull request once I say it is marked.
- Nobody moves a protocol's directory by hand; new files go where the protocol is on `main` today.

**Yours, in this order:**
1. #9670 (A, on `main`) first. Then #9677 (B), #9678 (L) and #9679 (I1), which are stacked on A: each on A's carried head.
2. #9736 (H) holds #9664. SCANLEAN carries #9664 now (it is marked, and ci refuses it until it is carried). Carry H after #9664's carried head is pushed, and say how H takes it.
3. #9319: say whether anything in it is in none of the five cuts. If nothing is, do not carry it; I close it.
4. My 7:11 AM order stands with its hours: a and b by 1 PM PST, and NET-411's pull request open, unmarked, by 3 PM PST, cut from the new `main`. Say at once if the carry moves either.

Send me one line by inbox when A's carried head is pushed, and the tool's words at once for any branch that is not `clean`. The standing lines of my 7:11 AM order hold.

