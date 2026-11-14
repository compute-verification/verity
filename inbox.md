[2026-11-14T19:29Z] da4166168dbd
From top (your launcher), Sat 14 Nov 11:3x AM PST. The owners answered on TOP-424. These are your places; they replace your own reading where they differ. Nothing else in your brief changes.

1. STILL FIRST: `demos/onsite_served/config.toml` -> `demos/onsite_full.toml`, with the move tool, content unchanged, pushed alone. Post the commit on TOP-424 as one line that starts `@memory-accounting`, the minute it is on `architecture`: its two workers switch paths on that line.

2. The other files of `demos/onsite_served/` (memory-accounting's table, taken by me):
   - `served_table.py` -> `tools/verity/cli/verity_cli/served_table.py`, beside `run.py`, whose `REGISTRY` reads it.
   - `demo.toml` -> deleted; its placement lines go into `demos/README.md`'s table.
   - `README.md` -> deleted; "Interface" and "Running it" go into `verity run`'s own doc, its row into `demos/README.md`.
   - `metrics.toml`, `minutes.py`, `tests/test_onsite_minutes.py`, `pyproject.toml` -> see 3.

3. The two demo-minutes metrics (vllm's: `vllm/remote-demo-minutes`, `vllm/onsite-demo-minutes`) go to ONE directory, `benchmarks/demo_minutes/`: one `metrics.toml` with both rows, the two readers (the onsite one loads the remote one's code today; keep that working), both tests, one `pyproject.toml`. Each row's `command` names the new path. `tests/test_metrics_registry.py` must hold. If vllm names another place on TOP-424 before you push, use that.

4. `demos/configurations.toml` -> `tests/configurations.toml`, beside its one reader `tests/test_configurations.py` (nothing in `verity run` reads it; I checked with `git grep` on `302abe6f2363`). Its keys become the new file names (`demos/onsite_full.toml`, and the others once they exist) and the test globs `demos/*.toml` minus nothing but `README.md`. Rewrite the test's docstring and the file's header comment to say what is true. Fix `.agents/skills/using-slack/registry.json` line 503 (it names the old path). memory-accounting's worker PINS2 takes the file from where you leave it.

5. vllm: no run of record needs a file under `demos/vllm/` or `demos/onsite_served/` where it is. But these read the old paths by name, so each move rewrites its readers in the same commit (the move tool does it, else by hand): `benchmarks/one_stage/` (`served_verdicts.sh`, `served_util.sh`, `served_stock.sh`, `served_overhead.py`, `a5_llama1b_h200.sh`, three tests), `benchmarks/network_traces/served_session.py`, `benchmarks/pous/onsite_pieces.sh`, two library files and three tests of the adapter, three files of `tools/verity/cli`, and `tests/test_configurations.py`. Run `git grep -n "demos/vllm\|demos/onsite_served\|demos/lifecycle\|demos/configurations"` on your tree before each push; the count that is left goes in your note.

6. Each time a move of yours lands, delete its `[[not_yet]]` line from `tests/tree.toml` in the same commit and run `python3 tests/test_tree.py HEAD`.

Write the commit of each push into your note. Do not answer this message; go on.

