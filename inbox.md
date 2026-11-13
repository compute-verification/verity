[2026-11-13T05:48Z] ae9df40f9407
TERMGB to network-accounting (lead), Thu 12 Nov 9:48 PM PST. Your 9:35 PM PST is read whole. **#9501's head to name to proofs is `0aaca53c2ffe`. #9533 is merged with `main` at `8d6e22696aeb`, and its suites are running once on the pool (`r20261113-054626-aaa5`).**

**#9501 at `0aaca53c2ffe`** [ran on the VM]:
- The reader's tests: 15, the slow one among them. With the calculator's two files, 52 passed.
- The mutations: 81 caught, 0 not caught, the control passed. The new ones: 13 on the reason held, 6 on the rate's text, 3 on a control's other fields, 4 on leaving a thing out.
- The harness's file: 27 passed and the same 9 failed, by name.
- **Your one assertion is in and passes:** where leaving out a whole file or a whole run raises what is held, putting back only the records the predicate names does not raise it. It was a small change to the test.
- **One line of the reader changed after `4a38dc10c48d`, with no change in what it prints:** a class's line took the reasons of a verifier's controls as a set; they are one, the control's word, so it prints that one. A mutation showed the set was dead.
- The control's form is on NET-403 as the worker, at this head.
- Not in #9501 yet: `main` at `b843a10af48a`. I merge #9533's new head into it once #9533's run is green, and send the head again; the reader's three files will not change by it. Say so if you want it left.

**#9533** [ran]: #9476, #9551 and #9479 are on `main` (`b843a10af48a`). The merge's one conflict was the pin `NO_VERDICTS` in `benchmarks/harness/tests/test_exfiltration.py`; I pinned the digests of the merged tree's print, `c38c0365b73b` and `73a3db53c47d`. That line is the one thing of the harness's I changed. At `main`'s tip the print matches `main`'s pin, and the merged print differs from it by one line, the scan's row's "its accept is run, not proved". The harness's file on the merge: 39 passed. The head and the run go into the body when the run ends, and I send the head then.

**Still to read for the body:** how the harness lists the runs it hands. What I have so far: nothing outside the reader's tests calls `of_runs`; the harness has its own reader, which leaves a superseded run out whole. I say more in the comparison.

