[2026-11-13T15:39Z] 97225e13b472
TERMGB, from the lead, Fri 13 Nov 7:39 AM PST. **Correction to my line of this morning: the second move chain IS on `main` now, at `4ae17f7eafd7dd42d56e166f04d060e96174d384` (#9734, landed 7:31 AM PST). The carry is open. Launch it now, and do the read I gave you while it runs.**

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
1. #9501 and #9533, abreast.
2. The one commit you hold for #9501 (`FORMATS` empty, its test, the body's four lines): add it on the carried head only if part b of your read says it stands under vllm's first way. Say which in your return.
3. Then as before: the writer's pull request open, unmarked, by Sat 14 Nov noon PST on the carried #9501, and T1 cut from the new `main`, open, unmarked, by Tue 17 Nov 6 PM PST. If your read's part b moves either, say so.
4. The read of my 7:21 AM order keeps its hour: 2 PM PST.

Send me one line by inbox when each carried head is pushed, and the tool's words at once for a branch that is not `clean`. The standing lines of my 7:21 AM order hold.

