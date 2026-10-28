
# 00:00Z (coordinator): Daniel's deadlines (4:38 PM PDT)

From now: a plan or answer is due within 1 hour of the ask; code is a PR marked within 3 hours of starting, and longer work splits into pieces each marked within 3 hours. Mark a PR as soon as its touched tests pass locally (drive needs no quick-tier pass; put the remote verdict in the body when it ends). Your dates are on your Linear issue; if one slips, say so in your next checkpoint with the new time.

**Your plan is late** (due within 1 h of your 22:30Z start): post the checkpoint note and the date on VLLM-110 now.

# 00:06Z (coordinator): correction

Your plan is in note 20261027T2240Z-report-b-of-record (22:40Z), so ignore my "late" line. #5715 and #5716 stay unmarked until 01:30Z as briefed, then mark them. Your run r20261027-225718-27f8 ended rc 143 after 34 min on vy-nebius-cpu-1; say in your next checkpoint whether it was your timeout or the node.

# 00:20Z (coordinator): Daniel's 5:09 PM PDT rulings, words and privacy

1. **Blocks, not periods or windows.** "Monitoring period" is the one word for a span of time. Every other span is a block's (execution block, sampling block, server block), said as the block's own span. "Execution period", "sampling period", "certification period", "Commit period", "link-window" and "window" in that sense all retire. New prose (docs, docstrings, PR bodies, Linear comments) uses this now. Identifiers, file names, flags, schema ids and wire strings stay until network-accounting's time cluster renames them, so don't rename them yourself.
2. **Privacy.** The site is public. Where each part of a workload runs (loc_j), the inference blocks and the declared advice are private, with only padded counts public. Publish none of them.
3. **The verdict.** Python may only read out the Lean verifier's result; it decides nothing.

# 00:37Z (coordinator): CORRECTION, vy-dc-h200 stays V2's until its run ends

V2's run of record holds vy-dc-h200 from 00:45Z until it ends (ETA about 03:30Z, at most 05:00Z), not just until 02:00Z: its PoUS and wipe pieces take the card's GPU lease after the serve, and a run of yours taking the lease between them could time PoUS out. **Launch nothing on vy-dc-h200 until V2's run has ended** (the coordinator posts it; or check that no r2026102[78] run of vllm-v2-compose is running there). Use vy-dc-h200-2 only if compute-accounting's energy runs aren't waiting there.
