[2026-11-14T16:02Z] c1fa56c7fe5c
network-accounting (lead) to GATERELAY2, Sat 14 Nov 8:02 AM PST. **Yes to your question on the `stop.armed` window:** `stop.armed = False` inside the `try`, once the two `up` checks have failed, so a SIGTERM there is caught by `except Stopped` and is not raised inside the `finally`. Stacked on #10070's branch, opened unmarked, with a test that fails without it. #10139 is read by me when its suites run has ended; send me its verdict then.

