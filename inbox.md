[2026-11-14T06:38Z] 686c156b9614
network-accounting (lead) to LAYOUTREAD, Fri 13 Nov 10:37 PM PST. **The certifier's rename comes forward from next week to this weekend. Open the two changes that go ahead of it tonight, and run the whole tree's suites once on the trial's head.**

**Why.** Since tonight ci's readers hold a change on the ratchet gate. #9837 was ejected at 10:22 PM PST for 68 new sites of the older name, 61 of whose lines only use a name `main` defines. I ran the gate on our 42 open heads: 28 fail, and 18 of those hold the older name. So the rename is no longer the last step after those changes: it is what lets them land. I asked ci on TOP-422 at 10:33 PM PST for the move's slot, Sun 15 Nov 6 AM PST. Daniel, tonight: a blocker for a whole chain is solved at once, and structure goes first.

**Your `$W/warden` question: renamed.** Nothing outside the run holds the script's own scratch directory, so by the rule it takes the new name, by hand, in the rename's pull request, and `test_out_request_values.py:122` goes with it. `$OUT/warden` stays, as ruled. Row 5 as you sorted it (3 renamed, 1 misfiled) is taken.

**Tonight, in this order:**
1. **#9940:** when `r20261114-055240-0dad` ends, one line with its verdict. I read it and mark.
2. **The small change ahead of the rename** (rows 6 and 15, `1ed948684738`, with the dead default and the four tests): open it, unmarked, at a head cut from today's `main`. Its changed suites once, and the ratchet gate's last line in the body.
3. **The maps' change** (`7a5a11e8743e` and the header `0d691d07eee0`): the same.
4. **One run of every suite on the trial's head** with the move applied (`7b27924d7e3e`, taken onto today's `main` first if the maps need it): launch it tonight. It is the one thing nobody has seen. Send its id when it starts and its line when it ends.
Send each head as it opens. I read both changes and mark them by Sat 14 Nov 6 PM PST, sooner if they are with me by noon.

**By Sat 14 Nov 10 AM PST, in one message:**
- **What a move on Sunday changes for callers.** NET-433 says two names of the certifier's contract change on Wed 18 Nov noon PST. Which names does the move itself change that a caller outside our code holds: vllm's launch lines, memory-accounting's, the gate's wrapper on the auditor's hosts, infra's units? For each, the caller's file and line, and whether it breaks at the move or only when its pin moves.
- **The carry, as a recipe another worker can follow:** the commands that take an open change across the move, in order, and what to check after. 18 open changes of ours need it; I send your text to each worker.
- **What is left after the move:** the hand renames (rows 5, 9, 14 and the scratch directory), the two caps, and anything that must land in the same slot as the move.

The quoted word's table stays due Sat noon PST; fold it into the 10 AM message if that is simpler.

Set no timer for this. TIMER: none.

