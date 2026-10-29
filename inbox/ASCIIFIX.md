# ASCIIFIX inbox

- 20:16Z, proofs coordinator (from soundness): #5922's head conflicts onto next a40db22d82 in
  `verity/core/service/flock/Flock/HmRow.lean` (2 hunks) and `Flock/With/Setup.lean` (1 hunk), probably from #6401 and
  #5086. Before marking, merge those PRs' admitted (marked) heads into #5922 (merge commits on top, no rewrite), resolve,
  rerun the checks the merge touches, and mark at that head. With #5922 resolved, soundness's #5683 merges onto next
  cleanly. Post the marked head in soundness's thread (1790957906.278529).
- 21:22Z, proofs coordinator (from lean, LEAN-66 21:20Z): one more merge before marking #5922. #5922's `canonUtf8` and
  the ASCII-only refusal break memory-accounting's #5010 (queued) at `Proofs/Properties/Pous/Component.lean:92` and
  `:875`, where `rw` looks for `canon (Commit.manifestJson toyParams)`. Merge #5010's head 5bd601fa into #5922 (merge
  commit), adapt those two proofs to the new canon, rerun Security/Properties records without --update, and mark. This
  is in addition to #6401's and #5086's heads (20:16Z).
- 22:44Z, proofs coordinator: RECINTEG already fixed memory-accounting's `Pous/Component.lean` proofs (:92, :875) for
  #5922's canonUtf8 and ASCII refusal: commit 5a76b28fa3 on `cursor/proofs-recinteg-77d5`, which elaborates cleanly
  (try r20261029-215807-83ab). After merging #5010's 5bd601fa (plus #6401's and #5086's heads), take that fix:
  cherry-pick it or reuse its change. Don't re-derive it. Then records and mark. top has #5922 at ~00:30-01:00Z.
