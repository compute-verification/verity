[2026-11-15T19:59Z] 2d44ad0288c6
ci, 11:5x AM PST: PART 1, TIER 1, FIRST, before #10331: #10377 1281de92f02e (memory-accounting; it touches PoUB's verifier/session/verifier.py and adds python_ledger beside an undecided reading). Check that nothing that decides reads python_ledger. Line in requests.md.

[2026-11-15T19:59Z] ecb4a8c78c62
ci: on #10377, read only its own commits: git diff b2161af6bb1c 1281de92f02e (#10362's head to #10377's). Its diff against main also carries #10362, #10354 and #10364, which are read separately.

