[2026-11-13T00:58Z] ddf484702310
WALKER, Thu 12 Nov, 4:58 PM PST. TRN-145.
**s-t20 at 2^-3 is relaunched as r20261113-005600-8f5f with a 14 h lease and 160 GB, but main's reader hits Python's recursion limit on s-t20, so the run raises the limit in its harness, with no change to the code.**

1. The run. RAN: r20261113-005600-8f5f, at 102ad1d7021c, `--on auto --timeout 14h --mem-gb 160 --cwd clone`. It went to vy-nebius-cpu-14's research pool and took slot r3 at 00:56:35Z. The 14 h (50,400 s) count from taking the slot, so the slot's run-max ends 14:56Z (6:56 AM PST Fri).

2. What 84cf shows, and the finding. READ, by pattern in out/st20.log and st20.json:
- r20261112-222055-84cf took its slot at 23:00:39Z and built in 1,315 s, which puts its build's end at about 23:22Z. It has printed nothing since. My harness prints only once the pass starts, so it is still in the reading before the pass: more than 1.5 h so far.
- The earlier run r20261112-185304-7c4e (at 759d5750ae, the same reader) built in 1,114 s, then ended its reading after 6,422 s with `RecursionError: maximum recursion depth exceeded` in work.py's reader. The frames are `value` → `member_words` → `words` (`[Previous line repeated 983 more times]`), before the pass ever ran.
- That `value` is on main, from 7fd737067e "rows become specific at a softmax over a mask that makes them differ". It is not this PR's code, and it is unchanged at 102ad1d7021c.
- INFERRED: 84cf will die the same way after about 1.8 h of reading, around 01:10Z, unless its depth falls under 1,000 at this commit.

3. What the new run does about it. It sets `sys.setrecursionlimit(1_000_000)` and runs the build and the read in a thread with a 1 GiB stack. It also times the build, the reading before the pass and the pass alone, and counts the pass's words, the claims, a refusal, N, origins, nonfinite nodes and NaN nodes. It prints a heartbeat every 10 minutes. RAN: the same harness at lf tiny on my VM took 5.7 s to build, 5.7 s to read before the pass and 1.0 s for the pass, with 22 claims, 0 origins and 8 nonfinite nodes.
That a reading of s-t20 needs more than 1,000 frames is a limit of main's reader, not of the refusal. Fixing it (making `value` iterative) is a code change: tell me if you want it on this branch or as its own PR. I make neither without your word.

4. Memory asked: 160 GB. READ: 84cf's session scope holds 50.6 GiB now and peaked at 57.7 GiB, growing 0.3 GiB/min, in its reading. The pass's arrays are unknown at this size, hence the margin.

5. Estimate. INFERRED:
- The build ends about 01:18Z.
- The reading before the pass takes at least 6,422 s (where 7c4e died) and some unknown remainder, so it ends at 03:00Z at the earliest.
- The pass is unknown. Its words grow about as the step's flat words: 3.3e11 here against lf tiny's, where it took 1 s for 2.7e7 words. Read as words, that is hours.
- I put the end between 05:00Z and 10:00Z (9 PM to 2 AM PST), inside the lease and before Fri 9 AM PST. I refine this at the pass's start, which the run prints with the time before it.

84cf runs on, and I report it when it ends. I look every 20 to 30 minutes.

