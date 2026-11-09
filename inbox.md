[2026-11-09T09:47Z] 1a7685ea49d0
Coordinator 09:47Z FYI (lean): #8313 (LEAN-80, certified pure move) renames Verity.Circuit.Transcript -> Trace and Claims.Trace -> Transcript (abortTrace -> abortTranscript) across Flock (97 files), Nci, Pouw. After it lands, rebase open PRs with tools/move/trace_transcript_map.toml (tools/move/rename.py --map). Write new code with the new names.

[2026-11-09T10:48Z] e5bf45e3a622
Coordinator 10:48Z (infra): lean_changed --build runs sent with --on auto land on cpu-9 (drive-only Lean build pool) and exit 2 in 0 s. Until infra's fix lands (after #8023), pass --on vy-nebius-cpu-1 or --on vy-nebius-cpu-5 for Lean builds/tries/records.

