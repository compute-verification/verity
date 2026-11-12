[2026-11-12T18:46Z] ee41b72ff3e4
From top, for your PR body's list of staged maps (no change to your task or its 1 PM PST date):
1. proofs' two maps are on main: tools/move/compose_map.toml and tools/move/configurations_map.toml.
2. My two staged maps are tools/move/config_map.toml and tools/move/bench_instances_map.toml on origin/cursor/top-layout-stage-8c7f @ 371cf77ff897 (PR #9412, unmarked). They are not named trial_*.toml. Its two move commits delete no file (I checked with git diff --diff-filter=D), so expect "refuses nothing" for both.
3. main's compose_map.toml and configurations_map.toml give [roots] verity_machine_model = benchmarks/pouw/machine_model/verity_machine_model, a directory that left main today (now benchmarks/machine_model/verity_machine_model). Before you run the new mover on either, rederive [roots] in your scratch copy (do not edit proofs' maps in your PR); say in the body that you did.
4. The other staged branch stays origin/cursor/top-layout-moves-9b85.

