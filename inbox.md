[2026-11-13T08:41Z] 4d6e2b003e53
GROWN, Fri 13 Nov, 12:41 AM PST.

**The draft is open, [#9711](https://github.com/compute-verification/verity/pull/9711), at 473edd3bba onto the grow branch, with all nine runs in its table. On one machine, `ema` did not move and the held entry's record differs only by `trained_sha256`. The VERSION bullet says 4.29.0 by today's order, not yet written.**

**#9711**, opened with `research queue open --draft --issue TRN-109 --by training`: a draft, unmarked, base `cursor/training-grow-depth-48ba`. The body starts "Lane training" / "Goal training/search-frontier-in-library" and ends "Refs TRN-109". It has:
- the table (entry, run, points, what binds, verdict);
- the RAN and READ lists;
- the VERSION bullet as you worded it;
- why the two rate-0.006 `-ema` entries are held by `AveragedModel` at `CLOSE`;
- ca11's point as INFERRED from your brief, checkpoint not read;
- the grown model's parameters and MACs a token beside each record's last checkpoint. Parameters agree exactly; the counted MACs a token are 0.04 to 0.07% over the record's on all ten, the held entry included.

**RAN, r20261113-071701-649f (vy-nebius-cpu-11, rc 0), the before and after part** (the base `ema.py` and `grow.py` swapped in on the same machine):
- The seven `ema` entries' `library.json` rows: "equal", each (`l16-t20-1.1x-fullmuon-fe-wsd-5383-ema`, `l16-t20-1.1x-fullmuon-fe-wsd-ema`, `m-t20-1.1x-fullmuon-wsd-4611-ema`, `m-t20-1.1x-fullmuon-wsd-ema`, `s-t20-1.1x-ema`, `s-t20-1.1x-fullmuon-ema`, `s-t20-1.1x-fullmuon-fe-wsd-ema`).
- Their digests from `config.load`: "equal", each (for example `s-t20-1.1x-ema` a98aab796283…).
- The held entry's counted record, timings left out: one line, "only after: checkpoints.0.trained_sha256 = 4ad76bfce108…", equal to that checkpoint's `output_sha256`. The before side reads "1 entry: 1 equal, 0 differ, 0 refused."
- Last lines: "of_record rc=0", "tests rc=0", "of_record before rc=0".

**Left, in order:**
1. The VERSION line (4.29.0 by today's order), with the one merge after the grow branch and #9643 hold theirs, when you say.
2. Carrying the branch across the move chains with `tools/move/restack.py` when you say they are on `main`, then the suites on the result and the head to you.
3. Anything your read of the draft finds.

No run in flight. No `main` merged.

