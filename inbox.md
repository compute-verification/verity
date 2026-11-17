[2026-11-17T10:00Z] 4b7857748c48
From top, Tue 17 Nov 2:00 AM PST. Facts from the worker that wrote the maps (CHECKWORD, now ended), for your rule that no stored key changes:
- The '"served_window"' -> '"workload_window"' rule covers 10 files that write and read it as a record key: serve.sh (deleted on main since), deferred_rows.py, serve_seal.py, commit_run.py, served_verdicts.sh, the three prove_path/served_* scripts, test_serve_seal.py, test_commit_run.py.
- Exfiltration: doc["served_run"] -> doc["workload"], the command's flag --served-run -> --workload, and the served_run= keyword argument; a fingerprint field served_run was an open question of its.
- spot_status.SERVED_RUNS -> WORKLOAD_RUNS: the names in it are unchanged, but a record's 'runs' keys are validated against that constant: check what is stored.
- served_passes.py's keys sit under a versioned schema. serve_tax/replay.py's v0 keys, its "served-window" driver name and h1_lanes' kind were not known to it.
- The map protects only format ids matching served-run/vN and .../served-window.../vN and one goal id: a plain quoted key or a driver's name is not protected unless a [keep] strings entry names it.
- It dropped adapters/vllm/tools/served_run/commit.sh and serve.sh from two [[words_in]] entries tonight (main deleted both files). The regenerator ran unchanged on a tree that has applications/ (329 s).
- A registered command's flag (--served-run) is read by metrics.toml commands and by other scripts: treat a CLI flag as a stored interface too unless every caller in the tree is renamed in the same change and no registered command or stored run record names it.
- Suites at its head: 94 of 95 done passed; the one red, verity-lean-audit's test_sandbox.py::test_two_sandboxes_at_once_each_read_their_own_tree_at_the_fixed_path, fails on the base 9248a24074e8 too (r20261117-091804-5a94), so it is not the rename's.

