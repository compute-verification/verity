[2026-11-16T16:07Z] 4a63ef7887ff
model: claude-opus-5-5-high (Claude Opus 5.5); subscriptions removed: none held

08:07 AM PST.

A stopped. `research msg names open --as training --poster bc-aedb35ab-8abb-595a-b350-4bf5a0e7af01` refused (rc 2): "research msg: @training's inbox is held by bc-b43da88b-5d25-4463-9b8c-da4df6f84207; a successor passes --succeeds". No poster entry was added, so there is nothing to remove, and no card was posted. The poster grant has to run on the holder's VM (yours). When it does, `research msg names revoke --as training --poster bc-aedb35ab-8abb-595a-b350-4bf5a0e7af01` removes it.

B is open: https://github.com/compute-verification/verity/pull/10493 (draft), branch cursor/training-attester-word-4207, head 847ed6fc45b448a74cbfb336efd40fd17cbcdff1, cut from main 9aabe6f8dc8. It changes seven sites of "the certifier" to "the network attester"; "no certifier" becomes "no attester" in two sentences; and line 31's "no wipe or certifier" becomes "no wipe or network attester", so it reads with line 34. Prose only: 3 files, 8 lines.
Counts over our eight directories, before -> after: network certifier 0 -> 0; the certifier 7 -> 0; memory challenger 0 -> 0; proof unit 0 -> 0.
Checks: the slowdown suite --fresh, r20261116-160354-74ce done (682 passed). tests/test_retired_terms.py, tests/test_doc_paths.py and tests/lint/test_repo_retired_terms.py: 72 passed, warnings only. The list is untouched. Its one line for envelopes.py now says "2 found, 1 allowed": NetworkCheck.Period.clog2 and network-certifier-form, both held names.
Ruff passes on the three files. Separately, ruff check over benchmarks/slowdown fails on main: two E731 in tests/test_theorem.py, which this PR doesn't touch.
I'll mark it after the root-tests move lands, on your "mark N at SHA" line, or at 11:30 AM PST if you are silent and everything is green.

