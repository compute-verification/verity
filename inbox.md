[2026-11-14T08:04Z] 9dc0df1622c7
From the infra lead, to you by name for your current job: act on it. A rule from top tonight changes one line of your brief: a worker opens and marks its own pull request the minute its node run exits 0, with the body it wrote; the lead reads after the mark.

When the node run at your final head exits 0 and your body is synced:
1. From your checkout: research queue open --head <your branch> --title '<the change's title, one sentence>' --body-file <your body file> --issue <your INF issue>
2. research queue ready <the PR's number> --mark --by infra --sha <the full head sha>
3. Put the PR's number and the marked head first in your final message, and comment both on your issue with --worker <your name>.

If the open or the mark is refused, do not work around it: put the refusal's text in your report. A job in several changes (TERMS, STYLE) does this for each change as it passes, and goes on to the next without waiting for me. If you are in a round of a reader's findings, the same holds once the fixes pass. Everything else in your brief stands: change no machine, act on no event that does not name you.

[2026-11-14T08:22Z] 27b516e5967d
From the infra lead, to INF-822 SENDTWO by name: act on it. Three facts about the metrics door from its owner (dashboard-metrics, 08:12Z), to hold your change to before you mark it:
1. The body is {"readings": [...]}: yes.
2. The door takes NO `loop` key and refuses any key it does not know. A loop's reading names `command` (the script's path) beside `commit`.
3. The door is not live yet: it is in compute-verification/infra pull request 47 (the website), and a migration (019) must be applied to the site's database after its merge. That pull request also reads rows under `infra/` and takes `every` in minutes.
Read that pull request's handler for the exact keys it accepts (gh pr view 47 -R compute-verification/infra; gh pr diff 47 -R compute-verification/infra) and make every reading you send hold only those keys; a test that fails when a reading carries an unknown key. Say in the body that the door is not live and what the loops do until it answers (they must not fail or log an error each pass for a door that refuses or is absent: say what they do). Then open and mark as the earlier message says.

