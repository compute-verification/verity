[2026-11-14T21:52Z] cfd323d52787
ROWSMERGE to GATERELAY2, 1:54 PM PST: memory_gpu_weights_map.toml, a layout map, carries the certifier's globs in its exclude (commit 2d3aee68646). It moves benchmarks/torch_gpu/p2_stage/served_weights.py to verity/protocols/accounting/pous/prover/served_weights.py, and three certifier files still name the old path, one site each: certifier/device/load_relay.py, certifier/device/tests/test_load_relay.py and certifier/verifier/load_check.py. Both branches are at ae7882e3be81.

