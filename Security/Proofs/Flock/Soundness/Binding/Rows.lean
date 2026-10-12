import Proofs.Flock.Soundness.Binding.Layout
import Proofs.Flock.Level3.RowV2

/-!
# `hm96-sha512/row/v1` and `/row/v2` rows: the registration fact from `HmRowComputes`

A layout's rows are the commit strings under the serving roots that hold `hm96-sha512/row/v1` or `/row/v2` leaves
(`HmRows`): the circuit hashes each one's row in `sha512x3` slots and outputs its commit string `b ‖ c` from an `hm96` slot,
the verifier hashes the public `b ‖ c` into frame-v3-sha512 roots and holds them to the public file's header
(`HmRow.checkPublic`, stated by `FailClosed.flock_verify_public_ties`; that the header's roots are the registration's is
the auditor's check, `matches_public`), and the statement pins the `hm96` slots' outputs to the public `b ‖ c`
(`HmRow.regions`). The other
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

/-- **A layout's `hm96-sha512/row/v1` or `/row/v2` rows**: which commit strings are rows (`isRow`), the bytes the row
digest hashes for a row (`enc`, injective: `sha512/row/v1`'s row prefix then its words, or `sha512/row/v2`'s `P(ℓ)` then
its bits, `v2`), the leaf scheme of the other commit strings (`other`), and
where a drawn unit's table's witness holds a row's commit string `b ‖ c`: the addresses of its `hm96` slot's 128 output
bytes, bit by bit (`bcAt`). Two facts about the layout:
* `bc_registered`: in a satisfying witness, at each of the unit's wires held by a row, `SHA-512(leaf_prefix ‖ b ‖ c)` of
  the `b ‖ c` the witness holds there is the registered leaf: the statement's regions pin the `hm96` slots' outputs to
  the public `b ‖ c`, whose leaves the verifier hashes into the registered frame-v3-sha512 roots;
* `other_registered`: at the other commit strings, the leaf of what the witness holds is the registered one. -/
structure HmRows (lay : Layout P plan tab decode) where
  isRow : lay.Pos → Prop
  [decRow : DecidablePred isRow]
  enc : lay.Val → List UInt8
  enc_inj : Function.Injective enc
  other : lay.Pos → ExplicitLeaf lay.Val
  bcAt : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → lay.Pos → Fin 128 → Fin 8 →
    Fin (2 ^ (Audit.Partition.tabOf plan tab S R u).S.m)
  bc_registered : ∀ S R u, u ∈ S → ∀ M, Sat plan tab S R u M → ∀ g ∈ P.io u, isRow (lay.pos g) →
    H512 (K512.leafPrefix ++ bytesAt (wit plan tab S R u M) (bcAt S R u (lay.pos g))) = lay.regLeaf R (lay.pos g)
  other_registered : ∀ S R u, u ∈ S → ∀ M, Sat plan tab S R u M → ∀ g ∈ P.io u, ¬ isRow (lay.pos g) →
    (other (lay.pos g)).leaf (lay.row S R u M (lay.pos g)) (lay.salt S R u M (lay.pos g)) = lay.regLeaf R (lay.pos g)

attribute [instance] HmRows.decRow

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

/-- A place the circuit hashes a row: a drawn unit `u` of the draw `S` under the registration `R`, a message of its
table, and one of the unit's wires. -/
structure RowSite (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length) (N : ℕ) where
  S : Finset (Fin n)
  R : Reg
  u : Fin n
  M : Msg plan tab S R u
  g : Fin N

namespace HmRows

variable {lay : Layout P plan tab decode} (rs : HmRows lay)

/-- Each commit string's leaf scheme: `hm96-sha512/row/v1` at a row, `other` elsewhere. -/
def scheme (p : lay.Pos) : ExplicitLeaf lay.Val :=
  if rs.isRow p then ExplicitLeaf.hm96 rs.enc rs.enc_inj else rs.other p

/-- **`HmRowComputes` at the drawn units' rows**: at each wire of a drawn unit held by a row, in a satisfying witness of
the unit's table, the `b ‖ c` the row's `hm96` slot outputs (the witness's bytes at `bcAt`) is the commit string of the
row's bytes and the salt the witness holds, under SHA-512 and the executable's key (`K512`). -/
def Computes : Prop :=
  Assumptions.HmRowComputes (fun l => digBytes (H512 l)) K512.saltPrefix K512.mask
    (fun w : RowSite plan tab C.N =>
      w.u ∈ w.S ∧ Sat plan tab w.S w.R w.u w.M ∧ w.g ∈ P.io w.u ∧ rs.isRow (lay.pos w.g))
    (fun w => rs.enc (lay.row w.S w.R w.u w.M (lay.pos w.g)))
    (fun w => lay.salt w.S w.R w.u w.M (lay.pos w.g))
    (fun w => bytesAt (wit plan tab w.S w.R w.u w.M) (rs.bcAt w.S w.R w.u (lay.pos w.g)))

/-- **The registration fact from `HmRowComputes`**: at a row, the witness's `b ‖ c` is the commit string of its row and
salt, so hm96's leaf of them is `SHA-512(leaf_prefix ‖ b ‖ c)`, the registered leaf; elsewhere it is `other_registered`. -/
theorem registered (hHm : rs.Computes) : lay.Registered rs.scheme := by
  intro S R u hu M hM g hg
  unfold scheme
  split_ifs with hrow
  · have hc : bytesAt (wit plan tab S R u M) (rs.bcAt S R u (lay.pos g)) =
        commitString (rs.enc (lay.row S R u M (lay.pos g))) (lay.salt S R u M (lay.pos g)) :=
      hHm ⟨S, R, u, M, g⟩ ⟨hu, hM, hg, hrow⟩
    have hb := rs.bc_registered S R u hu M hM g hg hrow
    rw [hc] at hb
    exact hb
  · exact rs.other_registered S R u hu M hM g hg hrow

/-- **The value binding of `hm96-sha512/row/v1` or `/row/v2` rows**: `Layout.binding` at the rows' scheme, from `HmRowComputes`. -/
noncomputable def binding (hHm : rs.Computes) : P.ValueBinding plan tab decode H512 :=
  lay.binding rs.scheme (rs.registered hHm)

end HmRows

end FlockSoundness.Binding
