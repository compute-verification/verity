# CIR-278 LINTFIX checkpoint, 2026-11-13T21:50:04Z

LINTFIX on CIR-278. Branch `cursor/circuits-cir278-lint-self-fixes-6f04`, head `9cdadd330996` (pushed 18:26Z; nothing new to push since).

Now: opening the pull request. Its body is drafted.

Count of the files open changes touch (of the 392 files that hold any circuits entry): 117 touched by an open PR under their current path; 52 more only under an older path (a rename on main since 1 Sep); 223 untouched. Of circuits' six-rule files: 48 touched (45 current path, 3 older only), 133 entries.

Runs:
- r20261113-182828-14b7 (vy-nebius-cpu-10, suites.py --changed fa3e619ee146 at 9cdadd33): ended STAGE_TIMEOUT (rc 143) at ~21:47Z while the Flock prover suite was running, so no suites.json. Before that, read in its stdout: 20 suites passed; verity-vllm 17865 passed, 3 failed.
- The 3 verity-vllm failures follow the length of the temporary directory, not this branch. My scratch dir sat under pytest's basetemp, so tmp_path was long: two handoff tests got "AF_UNIX path too long"; test_gate_plan cuts a message that begins with tmp_path at 200 characters.
- r20261113-204955-a048 (vy-nebius-cpu-3, done rc 0) ran the 3 tests in four cases. Base fa3e619 and head 9cdadd33 both pass with a short basetemp (44 chars) and both fail all 3 with a long one (83/85 chars). The long-path processes hung after failing and I stopped them by PID.
- No run in flight.

Left: open the PR (unmarked), the CIR-278 comment, the return.

Catalog/checker: no entry of the six rules in verity/circuits/ or tools/verity/circuit_check/ (run).

Stuck: nothing. The suites run left the Flock prover suite and whatever came after it without a verdict. The branch changes only adapters/vllm (30 files) and tests/lint/allowlists/repo_ruff.json (run: git diff --name-only); that the Flock suite ran because the cache's last pass predates other changes on main is inferred.
