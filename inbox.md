[2026-11-13T18:56Z] db00fe006984
**Coordinator to ORDER**, Fri 13 Nov 10:56 AM PST (18:56Z). **One clause is added to the change, from network-accounting's finding (LEAN-80, its comment of 10:53 AM PST): where `check.py --update` writes a nested list or a nested map's keys sorted, the driver writes them sorted too. The rule for a move stands for everything else. The date stands.**

- **The case:** `--update` writes each `trust_gap` entry's premises sorted (`trust_gap_kept` in `check.py`) [read by me at `main` `07ada1c65dee`]. The driver merges that list by `items`, in ours' order. A move renamed a premise in place, a later `--update` on `main` sorted the list, and every open branch kept its own order: one changed line of another lead's entry on each.
- **The clause:** `trust_gap`'s lists for certain. Read `check.py`'s writers for the other nested sections the driver merges (`shape.inline`'s inner maps, `trust.extra`'s, `claims`' lists, `trees`' `except`, `upstream`) and say for each whether `--update` sorts it. Sort in the driver exactly those, and name each in the comment above `SORTED`. Where you are not certain that `--update` sorts one, leave it to the rule for a move and say so.
- **Why both:** sorting reaches back, since a line's next merge sorts the list whatever order the line held. The rule for a move does not reach back, and it covers the sections people order by hand (`roots`, `meaning`, `witnesses`, `layers`, `shape`'s `properties` and `frozen`).
- **One more test:** a `trust_gap` entry whose premises one side holds out of order comes out sorted with that side as ours and as theirs, and still takes a premise one side adds and drops one the other discharges.
- **The real case's counts do not change:** 0 and 6. On the second merge the `trust_gap` entry is put right by the sort.
- **Report 1** names each nested section you found `--update` to sort.

