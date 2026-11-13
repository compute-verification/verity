[2026-11-13T17:33Z] 2507fcf8acda
WALKER, Fri 13 Nov, 9:33 AM PST.

**The fix is in #9558 (cfc639f925): `_strided` builds the index at each take and `Walk.index` is gone. Its test fails on the old code, the four shapes' `Work`s are equal byte for byte before and after, and the pass's peak falls by the cache's size at each. At D the new pass took 1.31 times the old one's, under your 1.5, on a node where untouched code ran 1.2 to 1.36 times slower in the same process. s-t20's full step is launched and is waiting for memory on its node.**

## 1. The fix

- READ, cfc639f925, `points.py` +11 −10: `_strided` is a module function that returns `base + (j // inner) * outer_stride + (j % inner) * inner_stride` and keeps nothing. `self.index` is deleted, and `Walk.take` calls the function.

## 2. The test

- READ: `test_what_the_walk_holds_does_not_grow_with_its_strided_takes`, in `tests/test_work.py` (+58 lines). A loop with one `Strided` take of 8,192 words a token position, each with its own base, beside a claimed product. A pass runs at 4 positions and at 16, and what it leaves allocated (tracemalloc, no clock) may differ by less than one take's index, 65,536 bytes.
- RAN on the VM, the named test: with ea11a40e04's `points.py` it fails, `[265541, 1054952]` bytes left, one index a position. With the fix it passes: 2,485 and 2,752 bytes left, and 4,368 at 64 positions.
- It measures allocation, not `Walk`'s attributes, so a cache that comes back at module level fails it too (INFERRED).

## 3. The readings, before and after

RAN r20261113-165019-3d55 (A), r20261113-165050-3f1b (B), r20261113-165120-d018 (C), r20261113-165149-90f3 (D), all rc 0. The tree is 30a0a4935c: cfc639f925 with #9676 and #9730 merged, a local commit, since without #9676 the reading does not finish at these shapes. Each run reads its shape twice on one node, first with ea11a40e04's `points.py` put into the tree, then with the fix's; `diff -rq` shows no other file differs. The two `Work`s, every figure as JSON, are compared with `cmp`.

| shape | `cmp` of the two `Work`s | pass, old | pass, new | new / old | pass's peak, old | new | at the pass's start | N |
|---|---|---|---|---|---|---|---|---|
| A s(2,64,512) | equal, 228,700 bytes | 266.5 s | 243.5 s | 0.91 | 5.07 GiB | 4.45 GiB | 2.23 GiB | 4,462,739,456 |
| B s(2,128,512) | equal, 234,302 bytes | 440.4 s | 457.9 s | 1.04 | 7.28 GiB | 6.12 GiB | 3.91 GiB | 8,925,478,912 |
| C s(2,64,2048) | equal, 231,166 bytes | 229.0 s | 271.2 s | 1.18 | 6.80 GiB | 4.60 GiB | 2.26 GiB | 4,563,402,752 |
| D s(2,128,2048) | equal, 236,767 bytes | 563.6 s | 738.5 s | 1.31 | 10.54 GiB | 6.25 GiB | 3.99, 3.95 GiB | 9,126,805,504 |

- RAN: tiny(2,16,64) is read first in each run, old and new: equal in all four, 51,636 bytes. Every shape: 70 claims (22 at tiny), 1 point, none refused, the same pass words old and new.
- RAN: A's `Work` has the same SHA-256 (45958f2a...) as the trace run's at #9730's head, which has no pass.
- **Memory.** RAN: the peak falls by 0.62, 1.16, 2.20 and 4.29 GiB; the cache held 0.52 and 2.04 GiB at A and C. With the fix the pass's peak is 2.21 to 2.34 GiB above its start at all four shapes.
- **Time at D: 738.5 s against 563.6 s, 1.31 times.** RAN, the same two processes: the build took 102.2 s against 74.9 s (1.36 times) and the reading before the pass 237.5 s against 198.2 s (1.20 times), and the fix touches neither. INFERRED: most of the 1.31 is the node. RAN on the VM, numpy alone: one index of 1,048,576 entries builds in 6.9 ms and one of 16,384,000 in 143 ms; D has under 1,600 wide takes a pass, about 11 s.
- A run that times the index builds inside the pass at D is going (r20261113-172733-ec35); its figure comes with the next message. I did not stop, since 1.31 is under 1.5.

## 4. Checks and the body

- RAN r20261113-165920-4307 at cfc639f925: `suites.py --changed b843a10af4`, `4 suites: 4 passed (0 cached), 0 failed` (repository 392, circuit-properties 132, training_slowdown 646, verity-graph 246). Ratchet gate `1 change(s), 0 failed`; ruff clean on the two files.
- RAN `research queue edit 9558`: the body now has "The walk's memory": the cause, the three options and why the first, the test, this table, and the whole-state copies (`Walk.inputs`, `Walk.take`'s concatenation, `Walk.scan`) at 0.797 GiB a copy at s-t20, stated and not fixed. The edit printed a 502; the live body equals my file (`cmp`). #9558 is unmarked, and nothing crossed `main`.

## 5. s-t20's full step

- RAN: r20261113-172658-0505, launched 9:27 AM PST on 30a0a4935c, `--on auto`, ask 92 GB, the reading's own stop at 90% of it, limit 10 h.
- **The ask.** INFERRED peak 69 GiB: 56.8 GiB at the pass's start and 8.1 GiB more through an iteration's products (RAN, b24e), then about 4 GiB for the state's copies and the gradient's takes. A fifth above that, over 0.9 for the stop, is 92.
- **It has not started.** RAN, the slot's own reason at 9:31 AM: vy-nebius-cpu-8 has 40 GB free and 306 GB reserved by live runs, under the 86 GB this run may peak at. Every other pool had no free slot at launch. It waits up to 240 min.
- INFERRED: the run is about 5 hours (b24e: 913 s build, 5,387 s reading, 4,985 s for one iteration's products). If it starts after about 12:30 PM it is not done by 6 PM, and the 6 PM message says how far it is.
- The harness now records resident memory after the build, before and after one collection, and the seconds spent building indices. RAN at D: 3.9 GiB resident after the build, 1.86 GiB after the collection.

One slip: I ran `research status --balance` once, and its table printed the first 48 characters of my runs' command column (a lock's path, no secret). I do not run it again.

