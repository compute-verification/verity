[2026-11-18T15:52Z] 799ff90312ba
From coordinator: one more small commit, on `cursor/compute-accounting-coins-record-strings-1de3` (no force), after the mapping work: soundness (TOP-442 7:51 AM PST) says yes to the hunk with one phrase changed in all three strings: where they open "a salt-only session on {challenger}" / "a draw session of its own on {challenger}" as a fact, write "a salt-only session the record names as {challenger}`s" and "a draw session ... the record names as {challenger}`s". Re-run the four tests by name on a node, push, and include that head and hunk in your return.

