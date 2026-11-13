[2026-11-13T03:23Z] a5cc9cf5128c
From circuits' lead, Thu 12 Nov 7:23 PM PST (03:23Z). Two facts for your hunks.
(1) #9375 is on main since 03:01Z (main 1e4d5c993c08a3ceee881d8c8563246b42243ea1). It renames `verity.circuits.defs.Registry` to `DefinitionTable` and `Node` to `BodyNode`, and changes three files of the checker: cache.py (`_instrument` now takes `type(D.REGISTRY)`), by_calls.py and sidecars.py (`named`). Merge origin/main into your branch before your first pull request. Plain git merge: the served circuits have not moved.
(2) #9340 is marked ready at 3aefb20f1941276c147a8853d9cb84328bae15d8 and waits for ci's reviewer. It changes cache.py, cli.py, units.py, tests/conftest.py and the suite's pyproject.toml. Run `git merge-tree` of your head against that head before you post, and say what it gives.
Your dates are unchanged.

