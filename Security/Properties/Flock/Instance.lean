import Definitions.Flock.Arith
import Properties.Flock.Level3.ArithFacts
import Properties.Flock.Level3.Nodes
import Properties.Flock.Level3.NovelBasis

/-!
`execArith`: the arithmetic the executable verifier computes with, as `Arith F K` for `F = GF(2^128)` in GHASH form and
`K = GF(2^256)`, read from `FlockLevel3`'s functions. That it satisfies `Arith.Correct` is
`Proofs.Flock.Soundness.Instance`'s `execArith_correct`.
-/

namespace FlockSoundness

open Model

/-- **The executable's arithmetic.** -/
noncomputable def execArith : Arith FlockLevel3.GF128 FlockLevel3.GF256 where
  lagS := FlockLevel3.lagS
  lagΛ := FlockLevel3.lagΛ
  combW := FlockLevel3.combW
  pinned := FlockLevel3.pinnedF
  pack := FlockLevel3.pack
  unpack := FlockLevel3.unpack
  embed := FlockLevel3.embed
  u := FlockLevel3.u
  omega := FlockLevel3.omega
  What := FlockLevel3.What
  lo := FlockLevel3.lo

end FlockSoundness
