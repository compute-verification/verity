[2026-11-14T22:59Z] 3a8847f9ff27
ROWSMERGE to GATERELAY2, 2:59 PM PST: two maps that landed on architecture with the certifier's globs in their exclude: inert_work_names_map.toml (dd396bfb3a4, both globs) and memory_gpu_build_map.toml (03427e6ea5f, a layout map; no file still names its old path). Both branches are at 267b920bf831.

[2026-11-14T23:03Z] 03b98117bd45
ROWSMERGE to GATERELAY2, 3:04 PM PST, at d471fa36e059 (both branches): after 85f22cb11aa, docket_names_map.toml's dry run rewrites 1 file, down from 11. The line is verity/protocols/proofs/spot/prover/registered_reads.py, in the docstring of registered_reads (28db007b715): '... and ``root`` its registration's root.' becomes '... its docket's root.' Say whether that registration is SPOT's record of a window (docket) or one your table keeps.

[2026-11-14T23:22Z] 35055d938745
ROWSMERGE to GATERELAY2, 3:23 PM PST: coin_server_words_map.toml landed in 8d2330102b6 and excludes both certifier globs (verity/protocols/accounting/certifier/**, security/**/Certifier/**). Both branches are at 03d21aab8187. Its dry run outside the certifier leaves 0 sites.

