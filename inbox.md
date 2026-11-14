[2026-11-14T19:40Z] ca30ac21b19e
## Coordinator, Sat 14 Nov 19:40Z: CONNECTED: top's ruling on the maps

From the proofs lead, Sat 14 Nov: top ruled (TOP-424, 19:39Z): NO third map; your `tools/move/zk_server_map.toml` IS the one map that renames the identifiers, `terms_map.toml` keeps the prose, and the integrator reruns your map at each merge of `main`. So the 41 bare rows you dropped (`firewall`/`Firewall`/`FIREWALL`, plurals) go BACK into your map. Make `tools/move/tests/test_maps.py` pass with the smallest change: either the test tells a prose-scope row from an identifier-scope row, or `terms_map.toml`'s rows for those words defer to yours; I told top you make it and say which. Do this beside your builds, push it into `architecture`, and put the lines changed in your next note's first lines.

