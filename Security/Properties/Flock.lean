import Properties.Flock.Level3.Bits
import Properties.Flock.Level3.GF128
import Properties.Flock.Level3.GF128Ring
import Properties.Flock.Level3.GF128Field
import Properties.Flock.Level3.GF256
import Properties.Flock.Level3.Seam
import Properties.Flock.Level3.ArithFacts
import Properties.Flock.Level3.Nodes
import Properties.Flock.Level3.NovelBasis
import Properties.Flock.Level3.FoldStmt
import Properties.Flock.Level3.Placed
import Properties.Flock.Instance

/-!
C-Flock's definitions that read its compiled verifier (`flock_verifier`'s `Flock.*`), extracted from `Proofs/Flock` so
that a guarantee stated over them reads no proof: the fields `GF128` (the code's `F128`) and `GF256` with their `Field`
instances (`Level3.*`, whose proof fields come with them), the executable's arithmetic `execArith` (`Instance`), and the
level-0 matrices the lincheck fold builds (`Level3.Placed`). The lemmas about them stay in `Proofs/Flock`.
-/
