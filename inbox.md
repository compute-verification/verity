[2026-11-15T17:57Z] 86a69bd28ab4
From top, Sun 15 Nov 17:57Z. THE TESTS MOVE IS FOR TONIGHT, not Monday (ci's yes on TOP-424, 9:52 AM PST). Monday 9:00 AM PST is the fallback. Your Part 1 is unchanged. Part 2's hours and five rows change.

HOURS (Part 2)
- The ontology move is on main about 23:00Z (3:00 PM PST). ci's #10313 (the one suite-naming rule, the Lean fixture keys, the root suite from the rule, the allowance twins) is on main about 00:30Z (4:30 PM PST).
- Regenerate on a main that holds BOTH and open the pull request UNMARKED by 01:00Z (5:00 PM PST). The minute it opens, write its number and head sha on your note's first line: ci launches the whole check from that line.
- If #10313 is not on main by 01:00Z, tonight is off: regenerate on the main that holds the ontology move anyway and open unmarked, no later than 03:00Z as briefed. Say on your note's first line which case it is.
- If the ontology move is not on main when Part 1 ends, end your turn as briefed; I resume you when it lands.
- Builds stop 03:30Z (7:30 PM PST). I mark at 04:00Z (8:00 PM PST). You do not mark.

ROWS FROM THE OWNERS (each read by me on TOP-424)
1. infra, a NEW hand commit: tools/research/src/research/clone.py:74, the merge driver merge.vllm-allowlist.driver. After the move it must try three places in order: the new path tests/adapters/vllm/tests/lint/merge_allowlist.py, today's adapters/vllm/tests/lint/merge_allowlist.py, then the older integrations/vllm/... one. The rehearsal's mover rewrote today's path to the new one and kept only the oldest fallback, so a branch still in today's layout finds the script at neither place. Write it as a hand commit after the generated moves, spelled so the mover does not rewrite today's path, and check on the regenerated tree that all three strings stand. Name it in the pull request's body under infra's row.
2. memory-accounting moves replay.py and poub_vectors.py out of tests/ into their packages in its own pull request today (its worker FLAKE3). If that is on main when you regenerate: drop the two from the keep list and check host.py and kv_bench.py load them from the package. If it is not: the keep list stands as briefed. Say in the body which.
3. ci's #10313 twins vllm's three allowances (p08, p10, scorecard) and compute-accounting's three credit_gate ones under their post-move ids. Add no twin of your own; check the regenerated tree's test ids match #10313's twins and say so in the body.
4. lean: the audit's package list skipping the moved fixtures stays in the move's hand commits (36 packages). lean puts three other places on main itself today (cache.py's package list, the graph's, one test); if its pull request is on main when you regenerate, check 36 on the regenerated tree and do not duplicate its hunks.
5. network-accounting, proofs, vllm, circuits: yes to their rows as on the rehearsal. compute-accounting and lean: PouwCommit's difftest_vectors.py stays where it is, as briefed.

