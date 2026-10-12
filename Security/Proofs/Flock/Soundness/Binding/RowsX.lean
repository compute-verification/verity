import Proofs.Flock.Soundness.Binding.Rows
import Proofs.Flock.Soundness.Binding.RowSeg

/-!
# Rows whose bytes need not be injective: `hm96-sha512/row/v1`, `/row/v2` and `/row-seg/v1` rows together

`HmRows` hashes every row's value under one injective encoding, so its leaf scheme is `ExplicitLeaf.hm96`. A
`sha512/row-seg/v1` row's outer message (`RowSeg.enc`) is not injective in the row: its leaf scheme is
`ExplicitLeaf.hm96X`, with row-seg's extractor (`RowSeg.extPair`). `HmRowsX` is `HmRows` with the bytes a row's digest
hashes chosen per commit string (`enc p`), and an extractor for two different values with the same bytes there
(`ext`, `ext_spec`: a SHA-512 collision). Its scheme puts `hm96X` at the rows (`HmRowsX.scheme`), and the registration
fact follows from `Assumptions.HmRowComputes` as for `HmRows` (`HmRowsX.registered`). `HmRows.toX` is an `HmRows` as one,
its extractor never needed.
-/

namespace FlockSoundness.Binding

open Game Model Accounting Refine Audit

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}

/-- **A layout's rows, each hashed under its own encoding**: `HmRows` with the bytes the row digest hashes at commit
string `p` (`enc p`), and for two different values with the same bytes there, two byte strings that collide (`ext`,
`ext_spec`). -/
structure HmRowsX (lay : Layout P plan tab decode) where
  isRow : lay.Pos → Prop
  [decRow : DecidablePred isRow]
  enc : lay.Pos → lay.Val → List UInt8
  ext : lay.Pos → lay.Val → lay.Val → List UInt8 × List UInt8
  ext_spec : ∀ p v v', v ≠ v' → enc p v = enc p v' → Collides (ext p v v')
  other : lay.Pos → ExplicitLeaf lay.Val
  bcAt : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → lay.Pos → Fin 128 → Fin 8 →
    Fin (2 ^ (Audit.Partition.tabOf plan tab S R u).S.m)
  bc_registered : ∀ S R u, u ∈ S → ∀ M, Sat plan tab S R u M → ∀ g ∈ P.io u, isRow (lay.pos g) →
    H512 (K512.leafPrefix ++ bytesAt (wit plan tab S R u M) (bcAt S R u (lay.pos g))) = lay.regLeaf R (lay.pos g)
  other_registered : ∀ S R u, u ∈ S → ∀ M, Sat plan tab S R u M → ∀ g ∈ P.io u, ¬ isRow (lay.pos g) →
    (other (lay.pos g)).leaf (lay.row S R u M (lay.pos g)) (lay.salt S R u M (lay.pos g)) = lay.regLeaf R (lay.pos g)

attribute [instance] HmRowsX.decRow

namespace HmRowsX

variable {lay : Layout P plan tab decode} (rs : HmRowsX lay)

/-- Each commit string's leaf scheme: `hm96X` over its encoding at a row, `other` elsewhere. -/
def scheme (p : lay.Pos) : ExplicitLeaf lay.Val :=
  if rs.isRow p then ExplicitLeaf.hm96X (rs.enc p) (rs.ext p) (rs.ext_spec p) else rs.other p

/-- **`HmRowComputes` at the drawn units' rows**, each under its own encoding: at each wire of a drawn unit held by a
row, in a satisfying witness of the unit's table, the `b ‖ c` the row's `hm96` slot outputs is the commit string of the
row's bytes there and the salt the witness holds. -/
def Computes : Prop :=
  Assumptions.HmRowComputes (fun l => digBytes (H512 l)) K512.saltPrefix K512.mask
    (fun w : RowSite plan tab C.N =>
      w.u ∈ w.S ∧ Sat plan tab w.S w.R w.u w.M ∧ w.g ∈ P.io w.u ∧ rs.isRow (lay.pos w.g))
    (fun w => rs.enc (lay.pos w.g) (lay.row w.S w.R w.u w.M (lay.pos w.g)))
    (fun w => lay.salt w.S w.R w.u w.M (lay.pos w.g))
    (fun w => bytesAt (wit plan tab w.S w.R w.u w.M) (rs.bcAt w.S w.R w.u (lay.pos w.g)))

/-- **The registration fact from `HmRowComputes`**, each row under its own encoding. -/
theorem registered (hHm : rs.Computes) : lay.Registered rs.scheme := by
  intro S R u hu M hM g hg
  unfold scheme
  split_ifs with hrow
  · have hc : bytesAt (wit plan tab S R u M) (rs.bcAt S R u (lay.pos g)) =
        commitString (rs.enc (lay.pos g) (lay.row S R u M (lay.pos g))) (lay.salt S R u M (lay.pos g)) :=
      hHm ⟨S, R, u, M, g⟩ ⟨hu, hM, hg, hrow⟩
    have hb := rs.bc_registered S R u hu M hM g hg hrow
    rw [hc] at hb
    exact hb
  · exact rs.other_registered S R u hu M hM g hg hrow

/-- **The value binding of rows each under its own encoding**: `Layout.binding` at the rows' scheme. -/
noncomputable def binding (hHm : rs.Computes) : P.ValueBinding plan tab decode H512 :=
  lay.binding rs.scheme (rs.registered hHm)

end HmRowsX

/-- **`HmRows` as `HmRowsX`**: one injective encoding at every row, so two different values never share bytes. -/
def HmRows.toX {lay : Layout P plan tab decode} (rs : HmRows lay) : HmRowsX lay where
  isRow := rs.isRow
  enc _ := rs.enc
  ext _ v v' := (rs.enc v, rs.enc v')
  ext_spec _ _ _ hne he := absurd (rs.enc_inj he) hne
  other := rs.other
  bcAt := rs.bcAt
  bc_registered := rs.bc_registered
  other_registered := rs.other_registered

end FlockSoundness.Binding
