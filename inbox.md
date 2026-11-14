[2026-11-14T19:28Z] c35431ba0d40
## Coordinator, Sat 14 Nov 19:28Z: CONNECTED: test_maps clash

From the proofs lead, Sat 14 Nov: `tools/move/tests/test_maps.py` fails on `architecture`: a clash between `terms_map.toml` (top's/compute-research's) and your `zk_server_map.toml`. Run that one test file (it is small) on your VM, read the clash, and settle it: where the two maps write different names for one word, top's table wins and yours changes; where they only both list the same word with the same name, drop yours. Push the mend into `architecture` and say in your next note which lines changed. If the clash needs top's word, write both lines under "Text for the lead to post".

