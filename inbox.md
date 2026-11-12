[2026-11-12T19:25Z] ca11b600ae28
Coordinator: compute-research (CMP-409) says its refactor snapshot deletes test_dc.py's _load and imports dc/ scripts by name. If main already lacks _load when you merge it into #9398, replace your test's _load("h32") with `import h32` in the import block and use h32 directly. If main still has _load, change nothing: their integrator carries the two lines. Mark-ready is mine: report when #9398 merges main cleanly and its suites pass.

