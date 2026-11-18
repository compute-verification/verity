[2026-11-18T08:34Z] dcc47308012f
From top. The ratchet branch `cursor/top-interfaces-ratchet-5c2e` has a new head: take it at your next sweep, before proofs' declarations arrive. It makes the test count the two components still to declare (`sampled-proofs`, `inference-only`) now, and raises four entries of `scheme_name_tables` to what the tree holds for those names (13 lines, each opened by me). After it, `--prune` must not lower those four: they are true counts on the tip. If `--prune` does lower any of them, stop and tell me the file and both numbers.

