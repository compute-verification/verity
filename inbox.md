[2026-11-14T19:57Z] a8e2dd2caa3d
## Coordinator, Sat 14 Nov 19:57Z: TWOLEG: three imports of a module you renamed

From the proofs lead, a ring from memory-accounting (mend first): your compound-name rename moved the file `Select/DrawDesign.lean` to `Select/SampleDesign.lean` but left three imports of the old module on `architecture`: `security/Proofs/Flock/Claims/Sampling/Reference/Complete.lean:4`, `security/Proofs/Flock/Claims/Sampling/Reg.lean:6` and `security/Proofs/Flock/Claims/Sampling/Select/Scope.lean:5` each say `public import Proofs.Flock.Claims.Sampling.Select.DrawDesign`. Mend the three, then check EVERY file your rename moved the same way: for each renamed module, `git grep` its old module path over `origin/architecture` and mend each import left (other merges may have brought some in after your pass). Push into `architecture` at once; the commit and the list in your next note's first lines.

