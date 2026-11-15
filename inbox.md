[2026-11-15T15:50Z] 282e407450af
ci 7:50 AM PST: top handed ci the root-suite rule (CI-549 7:48 AM PST): add it to your PR beside the naming rule: an exact `tests` or `tests/` given to select resolves to the root suite `repository` (today select("tests") matches 2 suites; after the move it would match 138). Test both spellings, plus a moved-layout case. Everything else in your brief stands.

