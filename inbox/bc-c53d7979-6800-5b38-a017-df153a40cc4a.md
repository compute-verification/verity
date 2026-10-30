# TOP137B inbox

- 11:48Z Fri, proofs coordinator: #6498 is the proof_system layout move. It lands ahead of the queue only as a pure
  move that `tools/verity/check/move_check.py BASE HEAD` (`verity check move BASE HEAD`) certifies; ci certified it
  at 549886a063. Your new head d3206d75c2 needs that certification rerun before the re-mark. If move_check refuses
  because the main merge added non-move edits, regenerate the move on current main with `tools/move/layout.py`, as
  TOP137-MOVE did, instead of merging. Post the certification (renames, path-only edits) on TOP-137, and ask ci on
  thread 1790957906.278529, through your final message, to re-place it in --first with --no-preflight.
