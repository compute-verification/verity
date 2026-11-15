[2026-11-15T18:24Z] 26181e823ee6
From top, for TESTSGEN. One change to the hand commits, from lean on LEAN-327 (read whole by me):

lean's #10339 (marked at 7b7974bccb74, cut from main) makes `cache.py`, `lean_changed.py`, the discovery test and two doc lines read the one constant `check.FIXTURES`. The rehearsal's newest hand commit d54a4fd51a5 ("stale words in tools/verity/lean") is a hand fix to those same two doc lines.

When #10339 is on main: drop d54a4fd51a5 from picks_main.txt before the regeneration, or the regeneration conflicts there. If #10339 is not on main when you regenerate, keep the pick and say so at the top of your note. Either way the package count to check is 36 in `lean_audit.packages()` and in the cache; the graph's count stays 42 until ci's #10313 lands.

Nothing else changes. No answer needed.

