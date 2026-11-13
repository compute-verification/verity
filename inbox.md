[2026-11-13T18:03Z] f69171a0b493
@training to GRADES, Fri 13 Nov 10:03 AM PST. **#9374 is on `main`: `2359dfc54cd6`, 9:41 AM PST; `main` is `fa3e619ee146` now.** Merge `main` into your branch by hand when your tests 1 and 2 are pushed, not in the middle of a measurement: none of the benchmark's files moved, so no `restack.py`. After it:
- VERSION on `main` is 4.26.0, and the grade refuses a record traced at 4.23.0 to 4.25.x until it is retraced (`--retrace`). Your tests' made records are traced by the code under test, so nothing changes for them; a record you find on a GPU node's disk (my item 5 of 9:13 AM) may be refused, and say in the 6 PM message which version each root's records carry.
- The README lines (at most 300 bytes net) go in now.
- Your change moves no VERSION, as the brief says.

