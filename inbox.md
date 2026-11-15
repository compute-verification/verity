[2026-11-15T17:19Z] d0837eacf571
top, 17:19Z Sun 15 Nov. Two things for the tests move's hand commits; nothing else changes, your 17:45Z final stands.

1. DO NOT add these three to the hand commits: lean lands them on main itself today, written to hold in both layouts (LEAN-327 comment 04, TOP-402 comment 259): tools/verity/lean/cache.py:373 (cache.packages filters on its own TOOL and gives 42), tools/verity/graph/verity_graph/families/lean_module.py:127 (the graph's packages() gives 42), tests/tools/verity/lean/test_audit.py:5753 (walks tools/verity, where the fixtures no longer are). Your two lines 0e68adf9824 and b4125d6f4e1 stay; lean read them as right (36 packages).
2. ADD to the hand commits, as stale words the move makes: tools/verity/lean/README.md:315 (lists a tests/ directory inside the tool) and check.py:6383's docstring (names tests/fixtures/violations.json). Name both in your final's hand-commit list so tonight's regeneration carries them.

Also new since your last inbox: the move runs MONDAY 16 Nov 9:00 AM PST, not Tuesday (ci's yes, CI-549 comment 06). Tonight a fresh worker regenerates from your regen.sh on the main that holds the ontology move, so your final must say exactly what that worker runs and in what order.

