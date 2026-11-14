[2026-11-14T02:50Z] 54b87ac6d2dc
NETSEATS to network-accounting (lead), Fri 13 Nov 6:50 PM PST.

**Item 6 is in: the gate's branch holds all eleven items and ruling 1, pushed, head `ac4869241619d5cd596de47be606b2c95b8ad212`. I expect the pull request open and unmarked Sat 14 Nov by 8 AM PST: it waits only on its changed-suites run, whose budget ends 5:53 AM PST at the latest.**

Commits since my 6:08 PM PST message, on `cursor/na-gate-end-fixed-words-4d2a` [ran: each read from `git log`]:
- item 8 `3fb33a2229da4e975a8e6fdb6ca6c6e161832539`, item 9 `207aacfce214a371c22ea338fa4fcf1b651a90bf`, item 10 `ebe6c2f5b83ae561436d3e9465598cd2f08d565c`
- your 6:21 PM PST ruling 2 (`signer_refused` out of the end and out of the chain's `COUNTS`) `7b5b5d14581f3865c2c538572bbb953583c165cf`
- item 11 `63fa0eee83fc29c5a0837ae7b3122614589d6e7e` (the store's record in form; the two refusals as codes)
- item 6 `bb5d4eedcee1469e4f4e104cfe39f06b340a68d7`, last of the items as ordered
- ruling 1 `70728d33c3ede820a1ad4d11916413de5eaebac4` and its merge `ac4869241619d5cd596de47be606b2c95b8ad212`

Item 6 as built [ran locally: the gate's three rehearsal files 415 passed; the runner's ten files that reach the gate or the lists 218 passed at item 6; after ruling 1's merge the five that read the lists 111 passed]:
- Every published key of a certificate's line and of a registration's line names a form: a constant, a value the run's own deployment gives (its link, the schedule's digest, a cell's bytes, an execution phase's buckets), hex of a fixed length (a signature's length from the deployment's key), a direction, a count below 2^63, and for a registration's `value` the name its own link, direction and execution phase give it.
- `certificates.jsonl` goes through the same forms by its own list. Before, a certified line's certificate was copied whole.
- A line out of form ends the run nonzero as `unruled`; its file and every file after it are not written. A deployment that doesn't fix the forms publishes no line.
- The forms are the gate's own. One test in the runner's suite holds the gate's three constants to `warden_rows`', its links and directions to the runner's, and a real run's lines to the forms (all published, none unruled).

Two things I did that you did not order, each in one line:
1. **Ruling 1's commit is on a scratch branch, not yet on #9867's.** #9867's suites run `r20261113-231315-f4ad` is still running [ran: 6:44 PM PST], so its head has not moved. The commit sits on `cursor/na-record-keys-refused-undeclared-7c1e`, one commit over `da1f3ee64436dd920aad9707e25a0b09ed17fdfc`, and is merged into the gate's branch from there, so the gate's runs are at its last head. When the suites run ends I push the same commit to #9867's branch as a fast-forward and replace its body. The merge conflicted in one file, the lists' test: the gate's forms stand and a refused line is undeclared under both lists.
2. **The control's clean file lost its refused line**, since that line now fails. The one new test: a record of a gate run with one refused line in each of the four files fails on those four, by the key and its count.

Runs launched once each at `ac4869241619d5cd596de47be606b2c95b8ad212`, ids from the launches' own lines:
- `r20261114-024714-f559`: the gate's suite and the runner's whole.
- `r20261114-024813-9266`: `suites.py --changed fa3e619ee1460da60580f0f9d462e0aa7c747c35` (the merge base with `main`), 86 suites.

Two finds for you, neither changed:
- **The gate's `out/` still writes a refused line** (`{"refused": [link, direction, execution phase]}`) in its four line files, in form. After ruling 1 a record with one fails. A key with no name that fetches the run still reads it. Leaving the lines out would say the same by the certificates that are missing, so I left it and the body's list names it. Say if the lines go.
- **On an end that is not accepted, `certified_windows` and `certificates` beside `windows` give the count the signer refused, by subtraction.** On an accepted end all three are the schedule's. You ruled `signer_refused` only, so both stay. Default if you say nothing: they stay, and the body's list names them.

Least sure of: whether the named-suites run's 60 minute limit, which the launch gave it, is enough for the runner's whole suite. If it ends at the limit I launch it again once with `--max-min` and say so.
Not looked at: the slow tests of any suite locally; the other 84 suites the change touches; #9867's suites result.

Held running: `r20261113-231315-f4ad` (#9867's suites), `r20261114-024714-f559`, `r20261114-024813-9266`.
TIMER: one, NETSEATS's own, Fri 13 Nov 8:00 PM PST, to read the three runs.

