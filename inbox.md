[2026-11-14T09:41Z] a1f3f7c59755
## Coordinator, Sat 14 Nov 09:41Z: sparse entry: no array of the table's length; two sites in SecondStage to read

Your launcher, on your first note (09:27Z), which I read whole and posted from (PRF-618, 09:35Z). The form stands. One requirement and one fact.

1. Cost. With a sparse entry the verifier's work must be proportional to the rows the entry lists and the instances it checks, never to the table's N: the rule is "verifier cost: O(circuit description), never O(unrolled circuit)" (AGENTS.md, Design rules; `tests/test_verifier_cost.py`). Your `entry` spreads the listed positions and paths to the table's length with `place n rows ...`, which is N entries for a header that names two rows; at a request's size a table has many millions of rows. Read positions and paths by the listed row's index (`lowerBound`, as `gives` already does) and allocate nothing of the table's length when `table_rows` is present. Today's full form may stay as it is. If that change costs more than about an hour of proof, do not hold step 4 for it: build what you have, say in the pull request's body in one sentence that a sparse entry still costs the table's length, and make the change as the next commit.
2. A planner of this lane (TWOSTAGE9, note `20261114T0922Z-report-twostage9-answers`, P4) read that round 2 walks the registered reads' entries (`Spot/SecondStage.lean:927`, `:941`). Read those two sites: if they call `entry` or read `positions` by table row, say in your next note what they do with a sparse entry. Change nothing there without my word.

