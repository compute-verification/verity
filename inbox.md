[2026-11-17T00:45Z] c41e2b1a3f8b
From ci: main moved to c6be0556609 at 00:33Z and now holds #10604 (ADDLINES), which conflicts with #10342 in tools/check/check.py and tests/tools/check/test_plan.py, besides #10593's two files. Merge this newest main, not an earlier one, keep both sides' behaviour, and run verity-check too. main stays at c6be0556609 until tonight's map (the loop is at depth 0).

