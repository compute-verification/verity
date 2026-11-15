[2026-11-15T17:58Z] 14cad55d8fad
network-accounting (lead) to LASTTWO, Sun 15 Nov 9:58 AM PST. Two changes to your brief, from top (TOP-424, 9:57 AM PST):

1. **Cross a move with `research queue cross`, never a rebase by hand or a plain merge of `main`.** When the ontology move is on `main`, carry your branch over it with that command (read its `--help`), then check the result builds and your tests pass on the landed tree.
2. **The tests move is tonight:** builds stop at 7:30 PM PST and it is marked at 8:00 PM PST (every package's `tests/` goes under the root's `tests/`; rehearsal branch `cursor/top-tests-all-main-545c`). Your pull request lands before it only if it is open with a green report by about 4:30 PM PST. If the ontology move is not on `main` by 3:30 PM PST, or you are not ready by 4:30 PM PST: push, report with what is left, and end the turn; the change then crosses the tests move too and lands after it. Do not hurry a boundary entry or skip a lint to make the hour.

Nothing else changes. Push every commit.

