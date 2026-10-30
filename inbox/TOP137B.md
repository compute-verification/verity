# TOP137B inbox

- 11:48Z Fri, proofs coordinator: #6498 is the proof_system layout move. It lands ahead of the queue only as a pure
  move that `tools/verity/check/move_check.py BASE HEAD` (`verity check move BASE HEAD`) certifies; ci certified it
  at 549886a063. Your new head d3206d75c2 needs that certification rerun before the re-mark. If move_check refuses
  because the main merge added non-move edits, regenerate the move on current main with `tools/move/layout.py`, as
  TOP137-MOVE did, instead of merging. Post the certification (renames, path-only edits) on TOP-137, and ask ci on
  thread 1790957906.278529, through your final message, to re-place it in --first with --no-preflight.
- 17:17Z Fri, proofs coordinator: vy-nebius-2 STOPS at 18:00Z (infra). Launch nothing new there; use cpu-1/3/6/11, never cpu-5. If a run of yours on vy-nebius-2 won't finish by 17:55Z, cancel it and relaunch on a cpu node now (its run copies are kept).
