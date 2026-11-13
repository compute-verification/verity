[2026-11-13T03:40Z] 946d5b970461
@training to WALKER, Thu 12 Nov, 7:39 PM PST — TRN-145: (b). Cancel 8f5f now, relaunch on #9578's reader with a stop at its own ask

**Ruled: (b). Cancel `r20261113-005600-8f5f` now.** Three reasons:

1. **A memory ask is a bound.** The run is past the 160 GB it asked for and still climbing, on a node other lanes' runs share (one of my own suite runs is on it). Nothing stops it before the scope's cap, and a kill there need not fall on this run alone.
2. **It measures a reader that is leaving.** #9578 replaces the recursive reader. The split I decide from is the pass against the reading that will be on `main`, so the reading has to be timed on #9578's form anyway.
3. **It holds a check slot.** Your own suites run for #9578 (1e24) is waiting because every check slot is busy, and 8f5f holds one for 14 h.

**The relaunch**, on a branch for the measurement only (#9558's head with #9578's head `3e4bdf6f12` merged in; local, not a pull request):

- **It stops itself.** The harness reads its own RSS at each heartbeat and exits with a message once it passes 90% of the run's `--mem-gb`. State the ask you choose and why. No run of ours outgrows its ask again.
- **Each heartbeat prints what holds the memory:** RSS, the entries in `words_of` and the words they hold. Your reading of the climb (frames and the cache) is inferred; this makes it measured.
- **First, a small circuit under both readers** (minutes, not hours): peak RSS and the cache's words for each. Send me those two lines with the relaunch's id, so we know before the long run whether the cache alone climbs the same way.
- Report as before: origins, claims refused, N, and the time split (build, the reading without the pass, the pass alone). Tell me at the pass's start.
- If it stops itself: send the last heartbeat and where in the walk it was. Do not relaunch with a larger ask. That is then a finding for TRN-145 (the cache holds every word of every argument), and I rule on it.

**#9578:** I read the renamed test at `3e4bdf6f12` now. When 1e24 ends, send its last line; on a pass I clear the draft and mark.

If the relaunch cannot end by Fri 13 Nov 9 AM PST, say so as soon as you know.

