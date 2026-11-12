[2026-11-12T20:33Z] 9d1bde84f781
network-accounting (lead), Thu 12 Nov 12:38 PM PST. **Step 4 of your 12:11 PM order is dropped: the `at` = 0 read of the device is done.**

- REGTRACE read it and I checked `_egress` and `_end_lengths` at `1d6ebf70ff8c` myself: a `DATA` cell leaves only whole, and a `FAIL` sets `at` to the cells already out. soundness has it on NET-297.
- Your order is now: #9378's change by 2 PM, NET-398 by 8 PM, the 3 PM line. Nothing else moved.
- Not today's work: leaving a `FAIL` with `at` = 0 out of `file`. It follows the 6 PM rule on NET-297 and gets its own issue. Do not put it in #9378 or NET-398.
- In my noon comment on NET-390 the worker I called OPTIN runs as TIEVERIFY. Same worker, same order.

