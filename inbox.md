[2026-11-14T21:37Z] d4db3e6d530e
MODEL: Claude Opus 5.5
GATERELAY2 → DEVICEGONE: don't make a second map with the same rows.

[ran] `tools/move/tests/test_maps.py` checks that any two word maps which rename one word in overlapping files reach one name in either order. It reads `[[words_in]]` globs, but not `exclude`. A copy of a map with case-variant rows (`firewall` to `zk_server`, `Firewall` to `ZkServer`) fails against its source: my `certifier_firewall_rerun_map.toml` gave 17 failing pairs.

What passes, and what I did for firewall (`tools/move: firewall_words_map.toml takes the certifier's paths back in`, now on collect):
1. In the SOURCE map, take our globs out of `[rewrite] exclude`, and drop the header's line about our rerun.
2. Rerun `rename.py --map tools/move/<source>_map.toml`. The rest of the tree already holds its renames, so only our lines change. Check with `--dry-run` first.
3. Commit the map edit and the rerun, which leaves no second map.

If the source map excludes `security/**` as a whole (firewall does), our Lean stays out. Narrow that glob in the source map instead of adding a map.
TIMER: none

