[2026-11-14T08:04Z] e55d09ece477
From the infra lead, to you by name for your current job: act on it. A rule from top tonight changes one line of your brief: a worker opens and marks its own pull request the minute its node run exits 0, with the body it wrote; the lead reads after the mark.

When the node run at your final head exits 0 and your body is synced:
1. From your checkout: research queue open --head <your branch> --title '<the change's title, one sentence>' --body-file <your body file> --issue <your INF issue>
2. research queue ready <the PR's number> --mark --by infra --sha <the full head sha>
3. Put the PR's number and the marked head first in your final message, and comment both on your issue with --worker <your name>.

If the open or the mark is refused, do not work around it: put the refusal's text in your report. A job in several changes (TERMS, STYLE) does this for each change as it passes, and goes on to the next without waiting for me. If you are in a round of a reader's findings, the same holds once the fixes pass. Everything else in your brief stands: change no machine, act on no event that does not name you.

