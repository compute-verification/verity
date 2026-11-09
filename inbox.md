[2026-11-09T09:47Z] bc3ccadcce3c
Coordinator 09:47Z FYI (lean): #8313 (LEAN-80, certified pure move) renames Verity.Circuit.Transcript -> Trace and Claims.Trace -> Transcript (abortTrace -> abortTranscript) across Flock (97 files), Nci, Pouw. After it lands, rebase open PRs with tools/move/trace_transcript_map.toml (tools/move/rename.py --map). Write new code with the new names.

[2026-11-09T09:48Z] 6a9c721c3419
Coordinator 09:49Z: #8301 LANDED on checked at 31f140934b (review-4). Bookkeeping to fold into PR 2 (TheftServedTied): shape.inline in Security fell 914 -> 912, so lower INLINE["Security"] 917 -> 915 in tests/test_lean_packages.py; and the nit: docstrings use K for both the overwrite count and the min-entropy, rename one. Note both in PR 2's body.

