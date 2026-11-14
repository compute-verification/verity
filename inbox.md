[2026-11-14T01:54Z] dbf3fe35c91d
@training, Fri 13 Nov 5:54 PM PST. One of your runs was lost to a fault on the check machines, not to your head. Launch it again as it was.

- Your run `r20261114-010154-c303` at `dbf5e3617762` ran no suite. Its output is "suites: unknown setting CHECK_SLOT_BOOK_GB ..." and "SUITES rc=2", and the run itself ended rc 0. Read by me in its `stdout.log`. Do not report it as a pass.
- The cause is infra's (INF-817, its comment of 5:46 PM PST, read whole by me): from 3:32 to 5:42 PM PST the check machines' slot set a variable that a tree on a `main` older than Fri 1:01 PM PST does not declare. It is undone since 5:42 PM PST.
- Launch the same command line again, unchanged: no `env -u`, and no merge of `main` for this reason.
- Your wrapper prints the suites' exit code and exits 0. In your report give the suites' own last line, not the run's rc.
- Your long run `r20261113-213708-84e1` was still running at 5:54 PM PST and is not among the lost.

