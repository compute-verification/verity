[2026-11-15T17:10Z] 26e0ae111e7e
ci 9:1x AM PST, for #9224: the queue declined to carry the old land from 08ba223ef8ad because commit 1da59137b36e, a restack rerun of tools/move/rename.py --map verify_proof_map.toml, changes benchmarks/one_stage/rehearsal.sh (+2 -2). That file is not in the diff against main, so the rerun should equal main's text. Please confirm in your read that rehearsal.sh at 70e61abc34c4 matches main byte for byte.

