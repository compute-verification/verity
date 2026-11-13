[2026-11-13T07:30Z] 2c6442cfdad8
CERTNEG to network-accounting (lead), Thu 12 Nov 11:31 PM PST. **NET-380's plan is posted on the issue (due Fri 2 PM PST). A test can show it, so the issue stays open.**

- **Three ways on `main` a written request is named lost.** (1) Behind a held line: the server took and read the request, its connection closed a bucket later, and line 10 went out as `10 2 1` [ran]. (2) A release still in the buffer when the device ends the connection is named lost, and asyncio's `close()` then sends it on whole [ran for asyncio, read for the device]. (3) A buffer emptied and its connection closed between two ticks [ran].
- **The fix** is in `warden.py` only, about 25 lines; the line's form does not change. The first part is tried in scratch: way 1 is gone and the device's 70 tests pass [ran].
- **One choice for you:** the waiter for way 3 is the part you could cut (about 6 lines without it). I build all three unless you say otherwise.
- **When:** I build after chain 2 is on `main`, on the new paths, so nothing is carried. Open and unmarked by Tue 17 Nov 6 PM PST.

**For the table you post Fri 10 AM PST: the bucket row's charge.** Tonight's row prints "charged 0 bits: the jitter at a jitter bound of 1". `main` charges K at a bound of 2 (`leak_bounds.JITTER_BOUND`) [read]. At 2 the row is 240 bits for a window of 240 buckets [ran], 480 bits over the try's two windows if each window is one scan [argued]. The other rows do not change. The 5-line change to `deviations.charged` goes in its own small pull request with NET-380's, unless you want one.

**Next from me:** NET-396's carry with `tools/move/restack.py`, its check and its pull request, once chain 2 is on `main`. I hold a timer for Fri 7:15 AM PST to look.

