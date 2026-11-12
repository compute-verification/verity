[2026-11-12T16:09Z] 4aa0953a250a
CERTNEG, Thu 12 Nov 8:09 AM PST. **The G = 8 run is launched, once: `r20261112-160747-765d` (NET-385). My estimate is 156 to 170 minutes of its 180, so it fits by 10 to 24 minutes and can still end at the limit with no result.**

**The launch** [ran]
- `research run --on NODE --project verity --source . --cwd clone --pool network --as network-accounting --mem-gb 128 --timeout 10800 -- env SC_GROUP=8 bash benchmarks/scan_cost/run.sh`, from the clean checkout at `f919f768d435`; the run's record names that tree.
- The baseline's own machine, slot `n0`, taken at once. It started at 8:08:10 AM PST; its limit ends it at 11:08 AM PST.
- Nothing is pushed to #9290's branch.

**The estimate (item 1), written before the launch.** Baseline figures are [read] from `r20261112-135528-7404`'s `stdout.log`, `out/sczk/summary.json` and `out/rss.jsonl`; the two local probes are [ran] on my VM (no proving); the G = 8 column is [argued] from them.

| Part of the scan | G = 1, s [read] | G = 8, s [argued] | Grows with the group? |
| --- | --- | --- | --- |
| start, builds, the host's read | 19 | 70 (up to 520 if the node's builds are cold for this tree) | the host's read does, about ×8 [ran: `--dry` here, 4 s and 33 s] |
| Programs and lowering | 277 | 300 | no: lowering is a template's; the Program is 81.02 MB at G = 1 and 81.04 MB at G = 8, the partition 494 bytes at both [ran] |
| `block` round | 31 | 60 to 120 | yes: one session, 8 units become 64 |
| `honest` round | 983 | 3,990 to 4,100 | yes: 4 sessions become 18 |
| `chain` round | 1,016 | 4,090 to 4,200 | yes: 4 sessions become 18 |
| `tampered` round | 69 | 510 to 560 | yes: 1 session becomes 8 |
| `unread` round | 281 | 280 to 290 | no: one unit at any group |
| copy, removal, result | 13 | 55 | with the kept bytes |
| **the run** | **2,687 (44.8 min)** | **9,350 to 9,700 (156 to 162 min); to 170 min with cold builds** | |

- **Inside `honest` at G = 8:** staging 75; the registration and the draw before the first session 265 to 375 (202 at G = 1, paid once a round); the 18 sessions 2,376 (first 70.9, last 96.1, eight ingress at 214 to 218, eight egress at 59.0) and their servers' starts 51; the Lean verifier 443; the scan of what the verifier received 745 (213 at G = 1: it reads every file under `verifier/` and the round then holds ten copies of the big circuits, not three).
- **Sessions in the scan:** 11 at G = 1, 46 at G = 8 (1 + 18 + 18 + 8 + 1) [ran: `--dry`, which the host's read passes at G = 8].
- **Peak memory:** the largest prover process about 23 to 24 GB (22.61 GB at G = 1, for 30 blocks at m = 26 + 5; a member at G = 8 has 31 or 32 blocks at the same m), about 31 GB with its server (6.97 GB), under 50 GB for the job with the driver. The slot gives 128 GB.
- **It fits three hours by 10 to 24 minutes** if a session in a group costs what it costs alone. Sessions and their Lean verifier are 59% of the estimate, so a session that costs a fifth more in a group ends the run at the limit with no result. That is the same assumption as your 93 and 17.
- **Not known before the run:** whether the node's builds are warm for this tree (`main` is merged in since the baseline); what the Lean registration check costs at 512 units (its Python twin is flat: 18.1 s at 64 units, 16.7 s at 512 [ran]).

**If it ends at the limit**
- The scan's log, `lean.jsonl` and `rss.jsonl` are written to `out/` as it goes, so the honest round's session seconds and the Lean verifier's CPU seconds survive. The stage split, the answers and the guard do not: no result.
- A limit of 240 minutes would cover the top of the estimate with a quarter to spare. A group of 6 comes to 126 to 135 minutes by the same arithmetic, a group of 4 to 93 to 99.

**One thing I found on the way** [read]: the baseline run carries no label in the store. The benchmarks' rule asks each measured run for one with its numbers. I will label this run when it ends and the baseline the same way, unless you say not to.

I report when it ends, or at the limit.

