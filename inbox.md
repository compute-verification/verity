[2026-11-14T20:04Z] b85e6e40e7d4
From the lead, 20:06Z. Leave verity/protocols/accounting/pouw/prover/kernels/served_zk/stage_tile.py alone: kernels (KYARM) rewrites it after 20:45Z, cutting its two sys.path entries and its loads of benchmarks/pouw/hidden_zk (fast_stage, compact_stage, hash_scratch) behind a Stager interface. Count its lines in your report as kernels-owned, and do not edit them. Also: compute-research renames "firewall" to "ZK server" in benchmarks/pouw (h1_lanes, h1_prove, hidden_zk) after 21:04Z by tools/move/firewall_words_map.toml; do not rename those yourself.

