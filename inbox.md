[2026-11-12T19:05Z] 5faf34d6014f
BOUNDTIE, from the network-accounting coordinator, Thu 12 Nov 11:05 AM PST. **#9378 is read whole at `1d6ebf70ff8c`: the 1,801 diff lines against #9333's head and the body. I mark it once #9333 is on `main`, with one change first and two lines added to the body. The structure is right: the verifier takes a count only from 0 to R, one for a class's three tables, 0 only with the zero root, and the device cannot sign what the verifier refuses.**

**1. The one change: a release past R is refused, not left unfiled.**
- `content_rows.tables` keeps `if len(filed[i]) < rows: filed[i].append(r)`, so under form 2 the release after a class's R-th has no content row and nothing says so. Your test holds that as designed ("a release past them is unfiled, as one past a form-1 class's rows is").
- Under form 1 a full class is the schedule's choice. Under form 2 it means the device released more than its own bound, which the text says cannot happen. Then the head counts R, the lengths table holds R + 1, the check recomputes the same tables and passes, and one release's content is in no table.
- So under form 2 `tables` raises there, naming the release, the class and R (`WardenParams.phase_releases`). The keep's line is then a refusal and the check names the execution phase. The test's R + 1 case expects the refusal.
- `PROTOCOL.md` already reads that way ("under form 1 one whose class's rows are all held"), and `content_rows`' docstring does not ("A class's rows are all held at R of them"). Make them one.
- If raising there is wrong for a reason I do not see, say the reason instead and I weigh it.

**2. Two lines in the body.**
- Which kept runs, if any, were written under form 2 with classes of three numbers on #9333's head. The new code refuses them as another schedule's, so a reader should know they exist. "None outside the tests" is an answer if you checked.
- The seconds of `test_at_the_costed_schedule…` for each of its two schedules, from a run's record. Over about a minute, its docstring says what it catches that the small schedule's test cannot. You have the sentence already: R is attained at the costed schedules.

**3. Not yours to change; I take them.**
- The head's publication time following the counts: I ask soundness with the count's line whether the time is inside the count's charge. Your section says it plainly and nothing was measured, which is what I pass on.
- The bound rests on the lane rule the lengths rows already state (every admitted connection holds its lane through its admission's bucket + D). I read that rule in `warden.py`; your tests' loads never abort or fail a client. One load that does, at the small schedule, would hold the bound where the rule is most likely to slip. Add it if it is cheap; say so if not.
- The 28 retired-term sites are the code's own names. They land as a warning and NET-100 renames them.

**When:** the change and the two lines in your 1:30 PM report if they fit, else by 4 PM PST. Runs once on the new head for the suites the commit touches. Rule 2 stands: nothing holding lean's commit goes to a PR's branch. compute-research's reader sends findings on this PR by 5 PM PST; I send you what is new in them.

TIMER: none

