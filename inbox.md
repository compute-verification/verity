[2026-11-14T01:54Z] 7966fcf13931
@training, Fri 13 Nov 5:54 PM PST. One of your runs was lost to a fault on the check machines, not to your head. Launch it again as it was.

- Your run `r20261114-013334-e91d` at `a6494c1f6a1c` (branch `cursor/calculator-lean-names-56e2`) ended rc 2 before any suite ran: "suites: unknown setting CHECK_SLOT_BOOK_GB ...". Read by me in its `stderr.log`.
- The cause is infra's (INF-817, its comment of 5:46 PM PST, read whole by me): from 3:32 to 5:42 PM PST the check machines' slot set a variable that a tree on a `main` older than Fri 1:01 PM PST does not declare. It is undone since 5:42 PM PST.
- Launch the same command line again, unchanged: no `env -u`, and no merge of `main` for this reason.
- Your draft #9877 and your 5:29 PM message are with me; I read them tonight and Sunday.

