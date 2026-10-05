import FlockSoundness.Binding.Layout
import FlockLevel3.RowV2
import FlockSoundness.Protocol.Binding.Rows

/-!
# `hm96-sha512/row/v1` and `/row/v2` rows: the registration fact from `HmRowComputes`

A layout's rows are the commit strings under the serving roots that hold `hm96-sha512/row/v1` or `/row/v2` leaves
(`HmRows`): the circuit hashes each one's row in `sha512x3` slots and outputs its commit string `b ‖ c` from an `hm96` slot,
the verifier hashes the public `b ‖ c` into the frame-v3-sha512 roots it checks against the registration
(`HmRow.checkPublic`), and the statement pins the `hm96` slots' outputs to the public `b ‖ c` (`HmRow.regions`). The other
commit strings (the returned outputs, u16 words hashed in the clear) have leaf schemes of their own (`other`). A
`/row/v2` layout's rows are bit rows (`HmRows.v2`), hashed as `sha512/row/v2` hashes them (`FlockLevel3.RowV2.enc`), whose
injectivity on rows of fewer than `2^64` bits is `FlockLevel3.RowV2.enc_inj`.

The registration fact (`Layout.Registered`) for the scheme that puts `ExplicitLeaf.hm96` at the rows (`HmRows.scheme`)
then follows from `Assumptions.HmRowComputes` (`HmRows.registered`): the `b ‖ c` a satisfying witness holds is the commit
string of its row and salt, and its leaf is the registered one. So the value binding (`HmRows.binding`) takes
`HmRowComputes` and the layout's facts, and nothing cryptographic.
-/

namespace FlockSoundness.Binding

open Game Model Accounting Refine Audit

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}

/-- **A layout's `hm96-sha512/row/v2` rows**: `HmRows` whose rows are bit rows of fewer than `2^64` bits (`bits`,
injective on the layout's values), hashed as `sha512/row/v2` hashes them (`FlockLevel3.RowV2.enc`), so that `enc_inj` is
`FlockLevel3.RowV2.enc_inj` after `bits`. -/
def HmRows.v2 {lay : Layout P plan tab decode} (isRow : lay.Pos → Prop) [DecidablePred isRow]
    (bits : lay.Val → {bs : List Bool // bs.length < 2 ^ 64}) (hbits : Function.Injective bits)
    (other : lay.Pos → ExplicitLeaf lay.Val)
    (bcAt : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → lay.Pos → Fin 128 → Fin 8 →
      Fin (2 ^ (Audit.Partition.tabOf plan tab S R u).S.m))
    (bc_registered : ∀ S R u, u ∈ S → ∀ M, Sat plan tab S R u M → ∀ g ∈ P.io u, isRow (lay.pos g) →
      H512 (K512.leafPrefix ++ bytesAt (wit plan tab S R u M) (bcAt S R u (lay.pos g))) = lay.regLeaf R (lay.pos g))
    (other_registered : ∀ S R u, u ∈ S → ∀ M, Sat plan tab S R u M → ∀ g ∈ P.io u, ¬ isRow (lay.pos g) →
      (other (lay.pos g)).leaf (lay.row S R u M (lay.pos g)) (lay.salt S R u M (lay.pos g)) = lay.regLeaf R (lay.pos g)) :
    HmRows lay where
  isRow := isRow
  enc v := FlockLevel3.RowV2.enc (bits v).1
  enc_inj := FlockLevel3.RowV2.enc_inj.comp hbits
  other := other
  bcAt := bcAt
  bc_registered := bc_registered
  other_registered := other_registered

namespace HmRows

variable {lay : Layout P plan tab decode} (rs : HmRows lay)

end HmRows

end FlockSoundness.Binding
