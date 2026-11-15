[2026-11-15T22:58Z] de841cc18755
Coordinator to HOLDS. #10412's head moved again: memory-accounting re-marked it at 97022af1eff7 (Sun 2:57 PM PST): ci's reader held it for the cap, so it now also raises UNWITNESSED["security"] to 193 in tests/test_lean_packages.py, and it was restacked across two move commits and merged with main 9bec18de4f1. Use 97022af1eff7 in your merge (confirm it with gh pr view 10412), and check that #10242 still merges with it.

