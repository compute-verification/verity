[2026-11-12T16:25Z] 297bacd7fa95
BOUNDTIE, from the network-accounting coordinator, Thu 12 Nov 8:25 AM PST. **NET-384 is decided: the turn-away, in the stricter form. It is NET-388, yours, and it starts when NET-379's pull request is open. Nothing changes in NET-379.**

- soundness ruled at 8:18 AM PST that the cell charges nothing and needs no record, under three conditions held by the device's tests:
  1. the cell's bytes are the same constant at every turn-away;
  2. it leaves on the next tick through the pacer, never off the tick;
  3. the decision reads only the device's tick count and the schedule, and a test varies the queue and the lanes at the stretch's start and shows the turned-away set does not move.
- NET-388's text has the rest: your part 2's rule and tests, with the cell on the tick. Its own pull request, beside NET-379's. Open and unmarked by Tue 17 Nov 6 PM PST.
- In NET-379's report, add one line: whether you start NET-388 at once.

