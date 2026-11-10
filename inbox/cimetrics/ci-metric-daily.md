# ci-metric-daily (CI-485), as run from Tue 10 Nov. Supersedes the timer's own text where they differ.

1. In /tmp/rchk: `git fetch -q origin main checked`; `python3 -m research queue status > /tmp/qstatus.txt` (the scorer
   reads it for marked PRs still waiting).
2. `python3 /cursor/stores/self/bin/ci_metric.py <yesterday 00:00 PST as UTC ISO, minus 2 days> --json /tmp/cimetric/daily.json`
   (scorer v2: landings are `Queue: #N` commits on main's own chain; v1, which followed `checked`, is ci_metric_v1.py).
3. `python3 /cursor/stores/self/bin/admitted_verdicts.py > /tmp/admitted.txt`: how many waiting PRs hold a land verdict.
4. Report yesterday's row: PRs marked, share on main within 2 h, landed (median, fastest), still waiting (with a land
   verdict or not), reverted; and the rate: landings an hour against marks an hour, and the backlog now.
   The lag row behind `checked` ended with Mon 9 Nov: do not report it.
5. Floor, from Tue 10 Nov: nine in ten PRs on main within two hours of the ready mark, a revert a miss
   (goal ci/landing-two-hours). A missed day: ask ci for the cause and the fix on CI-485 by noon PST. Two missed days in
   a row: bring Daniel options.
6. `research data put --kind landing-metric/v1 --file /tmp/cimetric/daily.json --preserve` (the "unknown kind" line is a
   warning); cite the art id. One short line (60 words) in ci's queue thread 1790957906.278529 with `--kind fyi`,
   the detail as a comment on CI-485, and the numbers to Daniel in my chat's final message.
