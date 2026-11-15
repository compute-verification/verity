[2026-11-15T17:35Z] 2d29133699fa
From top, 17:35Z Sun 15 Nov. Three facts and one rule for opening the pull request; no change to what you build.

1. main is 809eb268e4b, 45 files past the 24fd3ecd6349 your rebuild r20261115-172250-407e stands on. Among them: tools/move/layout.py and tools/move/suite_check.py (from #10296, the tests' tools), 15 files under tools/verity (#9823, the Lean tool), tools/check (7), benchmarks/tree (5), security/README.md, benchmarks/README.md.
2. So before you open: merge today's main into the rebuilt branch ONLY IF rerunning the move's three scripts on the merged tree changes 0 paths (the layout step with today's layout.py). If any path changes, rebuild on today's main instead. Put at the top of the pull request's body: the main sha it is built or merged on, and that count, with the command, so ci can repeat it when it pins the change.
3. Open the pull request UNMARKED by 18:15Z and write its number, branch and head as the first line of your note. network-accounting reads the branch at 19:00Z: the head it reads must be the one you open, so push before then and do not push after without a line in the note saying which strings changed.
4. The tests move now runs Mon 16 Nov 17:00Z and stands on yours being on main tonight. If something holds you past 18:15Z, write what and the run id in the note's first line; do not wait silently on a run.

