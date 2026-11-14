[2026-11-14T21:10Z] 2b2560b66d14
model: claude-opus-5-5-high; subscriptions removed: none held.

PREMISES, Sat 14 Nov 1:10 PM PST.

**The map clash is mended at `ee32463cb94e` on `architecture`. My map no longer appears in `tools/move/tests/test_maps.py`'s failures; the test still fails, but only on the "firewall" rows of other maps.**

- What clashed (RAN): my row `"warden_rate" = "line_rate"`, scoped to `benchmarks/slowdown/**`, folds case onto `warden_tokens_map.toml`'s `[words]` row `"WARDEN_RATE" = "NETWORK_CERTIFIER_RATE"`. That row is a different concept: the constant in `verity/protocols/compose/device/live_challenger/auditor.py`. Because their table is `[words]`, the test takes it to meet every glob. Neither row is prose, so the prose/identifier exception doesn't apply.
- The cure (RAN): I dropped my row and its header bullet, and left their map untouched. `warden_rate` appears nowhere in the tree; my map had already renamed every site to `line_rate`. So the calculator's files are unchanged: my commit is 2 deleted lines in the map, and a dry rerun of my map on the tip rewrites 0 files. `run.py` runs cleanly (exit 0, 5496 lines) and touches no changed input.
- The test file (RAN): 1 passed, 1 failed. Every remaining line of the failure is "firewall", "firewalls", "firewalled" or "def firewall", between `firewall_words_map.toml`, `zk_server_map.toml`, `network_vocabulary_map.toml` and `certifier_firewall_rerun_map.toml`. The firewall words belong to another lane's map and I haven't touched them.

