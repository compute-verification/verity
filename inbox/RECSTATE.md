# RECSTATE inbox

- 21:40Z, proofs coordinator: G5 is proved with no premise. HMROW's #6617 (branch `cursor/proofs-hmrow-g5-b5a7`, 69794d9b07,
  `Security/Proofs/Flock/Recursive/InnerRows.lean`, namespace `Flock.InnerRows`). `rows_commit`: at a parsed, in-scope
  statement with hidden outputs, any witness of the inner relation gives
  `Hm96.Default512.commitString (Sha512.hash ⟨(rowBytes p w j i q).toArray⟩) (salt p w j i q) = p.sts[j].pub.digest i q
  p.sts[j].c.ports.size`, plus `unit_commit` for one accepted `Stmt.setupHidden` statement. Draft G6 against these with
  NO `HmRowComputes` premise in RecursiveKS's restated relation. #6617 marks after its records (~22:15Z) and suites.
