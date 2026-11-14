[2026-11-14T09:06Z] 178f77be9ba1
To INF-824 RUNID, from the infra lead: this message names you, so act on it. It replaces one line of your brief ("do not open a pull request and do not mark anything").

1. The minute the node run at your final head exits 0 and your body is synced, open and mark your own pull request, from your checkout:
   research queue open --head <your branch> --title '<the change's title, one sentence>' --body-file <your body file> --issue INF-824
   research queue ready <the PR's number> --mark --by infra --sha <the full head sha>
   If the open or the mark is refused, do not work around it: put the refusal's text in your report and on the issue. A job in several changes does this for each change as it passes and goes on to the next.
2. The lead's conversation is replaced by a successor's at about 15:00Z today (Sat 14 Nov). The successor cannot resume you and reads only the issue, the notes and your branch. So comment the PR's number and marked head, each run's id and verdict, and anything left undone on INF-824 (research msg linear comment INF-824 --file F --worker RUNID), and never end on a state only your final message holds.
Everything else in your brief stands: change no machine; act on no event that does not name INF-824 RUNID.

