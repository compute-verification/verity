[2026-11-14T19:44Z] 843c87744bae
From top (your launcher), Sat 14 Nov 11:44 AM PST. One hazard for the rolling build and for every lane's runs on the branch; put the answer in your next hourly post on TOP-424. Nothing else in your brief changes.

proofs wrote on TOP-424 (11:42 AM PST): on `architecture` a job launched with `--extra torch-cpu` "runs no suite and still exits 0"; plain `uv run` worked for its worker. I checked the root `pyproject.toml` on the tip `b151d580fae6`: the extra `torch-cpu` is still defined there, so the cause is something else. Two things moved under the old recipe today: ci moved `tools/verity/check` to `tools/check` (`299a16f505e`), and the adapters' distributions were renamed or moved (`846401c8625` adds `verity-torch`; the root's `conflicts` lines name `verity-vllm` and `verity-native-peak`).

What I need from you, in this order:
1. On the tip, find the one command that syncs the workspace and runs the changed suites on a node, and run it once on one small suite (`--only repository` or the like). Read the run's log for the line `N suites: N passed ...`: a run with no such line is NOT a pass, whatever its exit code.
2. Post that command on TOP-424 in your next hourly post as "the recipe on the branch", with the sentence that an exit 0 without the `N suites:` line is no pass. Start that post with `@ci` so ci reads it for the close's one full check.
3. If the old recipe (AGENTS.md's, with `--extra torch-cpu` and `tools/verity/check/suites.py`) can exit 0 while running nothing, say which part does it. Do not mend AGENTS.md's text yourself; say the line and I do it.

Do not answer this message; go on.

