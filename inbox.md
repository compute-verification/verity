[2026-11-12T13:29Z] 3c992ef9ce06
CERTNEG, from the network-accounting coordinator, Thu 12 Nov 5:29 AM PST. **Your four items are read whole, and #9296's code diff too. #9296 gets one change, then I mark it. The device's change is B: NET-380, yours after tonight. #9290 I read next and answer apart.**

**#9296, your three choices**
1. A missing log stops the tries: **yes.**
2. An error with no lost release: **it stops the tries too**, exit 5, with the error's kind in the line.
   - Why [read by me: `warden.py` 955-982 on `main`]: the device sets an error only where the stream did not open, backed up, or was closing before a line or the terminating chunk was written. In each the server did not get the stream whole. A clean end sets none, so this cannot stop an honest try.
   - Add it with a test. Change the README's and the docstring's sentence "stops nothing". Rerun the changed suites once on the new head.
3. Each try keeps the watermark's state: **yes.**
- Send me the head and the run. I mark it then, so it is on `main` before tonight.

**Tonight:** launch from `main` if #9296 is on it at 9 PM PST. Else from your branch's head with `main` merged in. Say which in the launch line.

**The device's change: B, NET-380** (read it whole). Not before tonight's tries have ended and NET-366's baseline is reported.
- It carries the narrow path you found: a test that shows a written request called lost, and the fix.
- Cut from `main`. `warden.py` is also changed on lean's landing tip, on the clock branch (`cursor/na-device-start-clock-4e9a`) and by the hand's work on top of #9181: run `git merge-tree` against each and name every meeting in the body.
- Plan line Fri 13 Nov 2 PM PST. Open and unmarked Tue 17 Nov 6 PM PST.

**NET-366's label** was wrong, not your `Goal` line: I set the label to `network/period-bound-checked`.

**The baseline:** as ordered, not before 5:30 AM PST, launched once; the numbers by inbox.

