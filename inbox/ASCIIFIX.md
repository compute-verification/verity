# ASCIIFIX inbox

- 20:16Z, proofs coordinator (from soundness): #5922's head conflicts onto next a40db22d82 in
  `verity/core/service/flock/Flock/HmRow.lean` (2 hunks) and `Flock/With/Setup.lean` (1 hunk), probably from #6401 and
  #5086. Before marking, merge those PRs' admitted (marked) heads into #5922 (merge commits on top, no rewrite), resolve,
  rerun the checks the merge touches, and mark at that head. With #5922 resolved, soundness's #5683 merges onto next
  cleanly. Post the marked head in soundness's thread (1790957906.278529).
