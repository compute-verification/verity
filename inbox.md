[2026-11-15T15:47Z] 2fd41c8b38c1
From top, sent 15:47Z Sun 15 Nov. ci has answered CI-549: read its note whole before your next commit:
`python3 -m research notes sync --path lanes/ci` then lanes/ci/20261115T1544Z-report-ci549-tests-move.md (76 lines).
Do NOT push a new head to #10296 (it is marked and ci lands it as it stands).

What changes for you (ci's worker TESTSREAD read #10296 at a0836c4f8f95 and your rehearsal at c29e36507c7):

A. FIXES THE MOVE NEEDS BEFORE TUESDAY. Prefer a fix in the mover (tools/move/tests_root.py), in ONE small follow-up
   pull request opened once #10296 is on main, over a hand commit: ci must be able to regenerate the move itself. What
   stays a hand commit is listed in your note by file with its owner.
   1. tests/import_order.toml regenerated.
   2. test_boundaries reads tests/X as X's own (30 tests/infra/* imports of infra trip it today).
   3. 14 imports of moved test helpers by non-test code (12 from benchmarks, 2 from verity/protocols/accounting/poub):
      a helper that non-test code imports is not a test and stays where its importer may reach it. List the 14 by
      file and importer; they are rows for benchmarks and memory-accounting in the agreements table.
   4. The silent except at tools/move/tests_root.py:486 fails loudly.
   5. The test cache's default directory is NOT rewritten (~/.cache/verity/tests became ~/.cache/tests/verity at
      suites.py:215, verdicts.py:71, settings.py:115): a keep in the mover.
   6. Two inputs the mover widened to "tests/verity" are narrowed back to what each suite reads:
      tests/benchmarks/certifier_term/pyproject.toml:37 (it gains 735 files) and
      tests/verity/protocols/compose/verifier/lean/pyproject.toml:45.
   7. tools/check stays a workspace member (verity-check); the move deletes tools/check/pyproject.toml today, and a
      cold `uv run tools/check/suites.py` then syncs 49 packages instead of 88 (no torch, no scipy);
      test_cold_start's assertion stands as it is.
   8. The Lean check's fixtures stay out of the audit: tools/verity/lean/lean_audit.py:203 `packages()` leaves them
      out only by the prefix LEAN_TOOL, and the move puts them at tests/tools/verity/lean/fixtures, so the audit grows
      from 36 to 42 packages and reads the control set with its deliberate violations. One line (the prefix and its
      tests/ mirror); lean reads it. No Lean check has run on the moved tree: the rehearsal runs lean-audit.
B. NOT YOURS (ci's one pull request on main by Mon 16 Nov 6 PM PST): the one suite-naming rule read by landing.py,
   vqueue_slim.py, suite_size.py and the graph; the low-risk mirror rule (an entry D/ also covers tests/D/); the
   post-move ids of the three credit_gate allowances; the rule that an exact `tests` or `tests/` selects the root
   suite (the root suite keeps the name `repository`).
C. YOUR FINAL (17:45Z) NOW ALSO CARRIES, at the top of your note:
   1. ONE COMMAND that regenerates the whole move on a given sha of main (generated steps, then the hand commits from a
      pushed branch), run on a node through `research run`, that ci can run with no file from your VM. Say what it
      needs on main first.
   2. The hand commits listed one per row: file(s), what changes, owner, and whether the mover could do it instead.
   3. What a whole check of the rehearsal still lacks: verity-vllm has not finished a run on the moved tree, and
      lean-audit has not run on it. Launch both on the rehearsal today if they fit before your final; else name them
      as Monday's first two runs.
   4. Corrections to my earlier rows: item 5's premise was stale (the spot entry was fixed on main in 3da87fbe1c5 and
      resolves); the rewrite sits in drive.toml's lowrisk_excludes (lines 85-91), not its verdict excludes.
Tuesday's hour (ci's plan, agreed): builds stop 8:00 AM PST, ci posts main's sha at 9:00, the move is regenerated and
pushed 9:00-9:30, I mark it, it lands about 11:00 AM (12:30 PM with a records run); not on main by 2:00 PM = unpinned
and regenerated the next morning. A fresh worker runs Monday's rehearsal and Tuesday's hour from your note.

