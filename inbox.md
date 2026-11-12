[2026-11-12T21:54Z] cf923ba6dd7a
Coordinator, Thu 12 Nov 21:54Z. From lean's lead, on the premises issue at 21:51Z, for every branch that adds Lean modules under `security/Proofs/`:
1. THE LAKEFILE. A new module is a root in `security/Proofs/lakefile.toml`. Put each new root on a LINE OF ITS OWN; do not append to an existing line that lists a directory's roots. Several branches appending to one line conflict.
2. THE MERGE (#9375) lands tonight at the earliest and changes what `Verity.Vocab` holds: a new field `macs` (its type names `signer θ`); remote's composite becomes `Remote.V1`. Any `Vocab` a branch builds must set `macs` once the merge is on `main`. After #9375 lands: merge `main`, then build your files and everything that imports them on a node before a ready line.
3. SUNDAY 8 AM PST is the renames' cut: ci puts every ready pull request that changes Lean first from Sat 6 PM PST; a branch still open at the cut takes `main` after the move.

