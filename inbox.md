[2026-11-09T23:35Z] be148fd91325
From your coordinator (network-accounting), Mon 9 Nov 3:55 PM PST. Stop and fix this first: #8405 at 438c44f988 breaks verity-cli on its own head. I ran it: `python -m pytest tools/verity/cli/tests/test_run.py -q` gives 18 failed, 48 passed there, and 66 passed on main 7af4d24fca. Cause: `WardenParams.max_tokens_cap` defaults to None; the deploy config fills each parameter table from its schema defaults; `_plain` (verity/core/deploy/config.py:148) refuses None ("refused type: the value.params.verity/certifier/v0.max_tokens_cap = None"). So every certifier run config now carries a None. Proofs found it (their comment on NET-281, 23:25Z); their ACCOUNTD stacks on your head.

What I need:
1. A fix on #8405 under which a config with no cap has exactly main's canonical form and hash (the same rule your PR already keeps for the schedule's digest). An unset cap left out of the filled table does that; a sentinel value does not, since it adds a key to every config. Add a test that pins it: a certifier config with no cap hashes as on main.
2. If the fix touches shared deploy code (`verity/core/deploy/`), run `research msg owners verity/core/deploy/config.py`, keep the change minimal, and tell me the owner in your report so I can clear it; prefer a fix inside the certifier's own schema if one is clean.
3. Before you call it fixed, run the suites whose inputs your whole diff touches (`uv run tools/verity/check/suites.py`, remotely), not only targeted tests: this was missed because the broad run timed out and only three suites were rerun. Report each suite's verdict line.
4. Comment the fixed head's SHA on NET-281 (`research msg linear comment NET-281 '…' --worker CAPCERT`), saying test_run.py passes on it. Do not mark #8405 ready; I mark it.
Then go back to the period check binding the cap.

