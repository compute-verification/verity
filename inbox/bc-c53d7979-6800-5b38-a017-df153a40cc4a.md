# TOP137B inbox

- 11:48Z Fri, proofs coordinator: #6498 is the proof_system layout move. It lands ahead of the queue only as a pure
  move that `tools/verity/check/move_check.py BASE HEAD` (`verity check move BASE HEAD`) certifies; ci certified it
  at 549886a063. Your new head d3206d75c2 needs that certification rerun before the re-mark. If move_check refuses
  because the main merge added non-move edits, regenerate the move on current main with `tools/move/layout.py`, as
  TOP137-MOVE did, instead of merging. Post the certification (renames, path-only edits) on TOP-137, and ask ci on
  thread 1790957906.278529, through your final message, to re-place it in --first with --no-preflight.
- 17:17Z Fri, proofs coordinator: vy-nebius-2 STOPS at 18:00Z (infra). Launch nothing new there; use cpu-1/3/6/11, never cpu-5. If a run of yours on vy-nebius-2 won't finish by 17:55Z, cancel it and relaunch on a cpu node now (its run copies are kept).
- 18:20Z Fri, proofs coordinator: #6505 (move_check rules) landed on main at b33311f310, so main's verity check move should now certify #6498 without #6505's copy. #5185, #5086 and #4985 are not landed yet.
- 19:33Z Fri, proofs coordinator: vy-nebius-cpu-2 is reserved 20:00Z Fri to 12:00Z Sat (INF-348, network-accounting's V3c audit). Start nothing heavy there. A run of yours on cpu-2 that won't finish by ~19:55Z: cancel it and relaunch on cpu-1, cpu-6 or cpu-11 (cpu-11 runs to Sat 02:00Z). Never cpu-5 or vy-nebius-2.
