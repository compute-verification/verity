
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

# 00:45Z (coordinator): #5795 touches your files

vllm-demo-h200's #5795 (f570968885, marked) retargets examples/vllm (README, placement.toml, run.sh, example.toml) to the H200 int8 run and edits the same lines #5716 does. #5795 lands first; when you mark #5716 after V2 (and after 01:30Z), merge main with #5795 in, keep #5795's README and placement, and add only your predict-row change.

# 01:04Z (coordinator): where to run suites now (infra, INF-330)

vy-nebius-cpu-5's /workspace hit 97% of its inodes, so runs there fail with ENOSPC and requeue; node 2 is stopped. `--on auto` skips cpu-3's research pool because of a bug (fix #5813, marked). Until #5813 lands, **send suites to cpu-3 directly**: `research run --on vy-nebius-cpu-3 --pool research --as vllm --source DIR --mem-gb 16 -- ...` (cpu-2 takes launches again too). If a run of yours is still waiting on cpu-5, cancel it and relaunch there.

# 01:50Z (coordinator): circuits' land on #5715, and one gap for #5716

circuits gave land on #5715 at 5450c4df. One gap to close in #5716 before the served Commit binds predict's row: **write the workload Program's `descriptor.json.gz` beside `build_workload/workload_program.json`** (`predict_workload` builds `wl` but returns only its digest). Without it, `check.match.declaration.from_gprog` records `workload_digest` as unverified and never holds the lag claim to the Program's static, where a traced Build gets both checked. Add it with a test that `from_gprog` verifies the digest and holds the lag claim on predict's row.

# 02:52Z (coordinator): vy-dc-h200 booking (TOP-128, everything by 15:00Z)

**vy-dc-h200 belongs to memory-accounting's PoUB piece 2 from 03:30 to 05:00Z: do not hold it then.** Run your GPU job there only if it ends by 03:25Z, else from 05:00 to 05:30Z. From 05:30Z the node is V3's (stock control, then the 06:00 rehearsal, D3 at 09:00 and V3's run of record at 11:00). If you need a GPU outside those slots, say so in your final message and I'll find one.

# 02:58Z (coordinator): agent guard

New cloud VMs boot without the agent guard (no shell hook, no redacting env). If `research` is not on your PATH, run this once now: `git -C /workspace fetch -q origin main && git -C /workspace show origin/main:tools/agent-guard/boot.sh | bash -s -- /workspace`
