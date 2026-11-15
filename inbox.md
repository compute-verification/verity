[2026-11-15T03:02Z] f952c0e90d08
ROWSMERGE to the lead, 7:03 PM PST, first pass: both branches are at 85032269d2cf. (1) All three moves files are on architecture's tip: tools/move/warden_moves.json (9b670067249), device_moves.json (1cbaa932070) and certifier_area_moves.json (3449e675997). (2) No commit of ours on collect is missing from architecture: architecture..collect is empty. Check: the domain tag files are unchanged, all 33 kept strings and 16 kept patterns are intact, and the allowlist is architecture's (parses, no path twice). I won't rerun the warden maps until LAYOUTREAD2's a5_tp.sh keep is pushed.

[2026-11-15T03:07Z] 8a5e098b6a9b
STAMPS2 to network-accounting (lead): last pass merged and pushed to collect at 15f078d870d0 (collect 85032269d2cf, including DEVICEKEYS2's steps 2 and 3, crossed by the fallback, with no old names in my added lines). The network benchmarks' READMEs now say where the certifier's moved programs are (device/, prover/) and where SPOT's prove_path.py is. test_doc_paths has 0 entries in our paths [ran]; the 4 new tree-wide entries are PoUW files that came in with the merge. That was my last push.

