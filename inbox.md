[2026-11-13T15:39Z] 76c32878e05c
NETSEATS, from the lead, Fri 13 Nov 7:39 AM PST. **Correction to my line of this morning: the second move chain IS on `main` now, at `4ae17f7eafd7dd42d56e166f04d060e96174d384` (#9734, landed 7:31 AM PST). The carry is open. Launch it now, and do the read I gave you while it runs.**

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
1. #9656 first, then #9741 on #9656's carried head.
2. Both #9656 and #9369 add a bare `import warden_serve` at `tests/test_session_start.py:24`. Write that import by the root's name in the carry's own commit, so the two pull requests change the line the same way.
3. #9741's body has no owner line and the queue says so: add `Lane network-accounting`.
4. **A change to my 7:20 AM order:** the branch that takes the guard's third entry out is cut from #9741's CARRIED head, not from `e13350660c0a`. The read of it (every reader of the two keys, the pins, the mend) can run while the carry does. Its hour stands: 1 PM PST.

Send me one line by inbox when #9656's carried head is pushed, and the tool's words at once for any branch that is not `clean`. The standing lines of my 7:20 AM order hold.

