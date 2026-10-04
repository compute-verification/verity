import FlockSoundness.Binding.Rows
import FlockSoundness.Discharge.Layout.Pins

set_option Elab.async false

/-!
# Rows whose leaf scheme is each row's own

`Binding.HmRows` hashes every row with one injective encoding (`enc`). Under the widened scope a row's encoding is its
schema's, and a segmented row's is not injective (`ScopeW.rowEncSeg`). `HmRowsP` is `HmRows` with an encoding per
commit string (`enc q`) and, at equal encodings of different values, an extractor (`pair q`, `pair_spec`):

* `hm96X`: hm96's leaf `SHA-512(leaf_prefix ‖ b ‖ c)` of an encoding with an extractor: `hm96Pair` when the encodings
  differ, the encoding's own extractor when they agree;
* `HmRowsP.scheme`, `Computes`, `registered`, `binding`: `HmRows`'s, at the per-row encoding;
* `HmRowsP.toHmRows`: the same binding as an `HmRows` whose every commit string is an `other` one, its leaf scheme
  `HmRowsP.scheme` (`toHmRows_binding`). The headlines take `HmRows` and read it only through its binding, so they
  hold at `HmRowsP` unchanged;
* `rowsP`: `Layout.Readers.rows` at a per-row encoding.
-/

namespace FlockSoundness.Discharge.ScopeW

open Game Model Accounting Refine Audit Binding

variable {Val : Type}

/-- **hm96's leaf for an encoding with an extractor**: the leaf of `v` under salt `y` is `SHA-512(leaf_prefix ‖ b ‖ c)`
of the commit string of `enc v`; two openings with one leaf collide in `hm96Pair` when their encodings differ, and in
`pairE` when they agree. -/
def hm96X (enc : Val → List UInt8) (pairE : Val → Val → List UInt8 × List UInt8)
    (hpair : ∀ v v', v ≠ v' → enc v = enc v' → Collides (pairE v v')) : ExplicitLeaf Val where
  leaf v y := H512 (hm96Input (enc v) y)
  pair v y v' y' := if enc v = enc v' then pairE v v' else hm96Pair (enc v) y (enc v') y'
  pair_spec v y v' y' hv h := by
    split_ifs with he
    · exact hpair v v' hv he
    · exact hm96Pair_spec (enc v) y (enc v') y' he h

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}

/-- **A layout's rows, each hashed under its own schema**: `HmRows` with an encoding per commit string (`enc q`) and its
extractor at equal encodings (`pair q`, `pair_spec`). -/
structure HmRowsP (lay : Layout P plan tab decode) where
  isRow : lay.Pos → Prop
  [decRow : DecidablePred isRow]
  enc : lay.Pos → lay.Val → List UInt8
  pair : lay.Pos → lay.Val → lay.Val → List UInt8 × List UInt8
  pair_spec : ∀ q v v', v ≠ v' → enc q v = enc q v' → Collides (pair q v v')
  other : lay.Pos → ExplicitLeaf lay.Val
  bcAt : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → lay.Pos → Fin 128 → Fin 8 →
    Fin (2 ^ (Audit.Partition.tabOf plan tab S R u).S.m)
  bc_registered : ∀ S R u, u ∈ S → ∀ M, Binding.Sat plan tab S R u M → ∀ g ∈ P.io u, isRow (lay.pos g) →
    H512 (K512.leafPrefix ++ bytesAt (Binding.wit plan tab S R u M) (bcAt S R u (lay.pos g))) =
      lay.regLeaf R (lay.pos g)
  other_registered : ∀ S R u, u ∈ S → ∀ M, Binding.Sat plan tab S R u M → ∀ g ∈ P.io u, ¬ isRow (lay.pos g) →
    (other (lay.pos g)).leaf (lay.row S R u M (lay.pos g)) (lay.salt S R u M (lay.pos g)) = lay.regLeaf R (lay.pos g)

attribute [instance] HmRowsP.decRow

namespace HmRowsP

variable {lay : Layout P plan tab decode} (rs : HmRowsP lay)

/-- Each commit string's leaf scheme: hm96 of its own encoding at a row (`hm96X`), `other` elsewhere. -/
def scheme (q : lay.Pos) : ExplicitLeaf lay.Val :=
  if rs.isRow q then hm96X (rs.enc q) (rs.pair q) (rs.pair_spec q) else rs.other q

/-- **`HmRowComputes` at the drawn units' rows**, `HmRows.Computes` at each row's own encoding. -/
def Computes : Prop :=
  Assumptions.HmRowComputes (fun l => digBytes (H512 l)) K512.saltPrefix K512.mask
    (fun w : Binding.RowSite plan tab C.N =>
      w.u ∈ w.S ∧ Binding.Sat plan tab w.S w.R w.u w.M ∧ w.g ∈ P.io w.u ∧ rs.isRow (lay.pos w.g))
    (fun w => rs.enc (lay.pos w.g) (lay.row w.S w.R w.u w.M (lay.pos w.g)))
    (fun w => lay.salt w.S w.R w.u w.M (lay.pos w.g))
    (fun w => bytesAt (Binding.wit plan tab w.S w.R w.u w.M) (rs.bcAt w.S w.R w.u (lay.pos w.g)))

/-- **The registration fact from `HmRowComputes`**, as `HmRows.registered`. -/
theorem registered (hHm : rs.Computes) : lay.Registered rs.scheme := by
  intro S R u hu M hM g hg
  unfold scheme
  split_ifs with hrow
  · have hc : bytesAt (Binding.wit plan tab S R u M) (rs.bcAt S R u (lay.pos g)) =
        commitString (rs.enc (lay.pos g) (lay.row S R u M (lay.pos g))) (lay.salt S R u M (lay.pos g)) :=
      hHm ⟨S, R, u, M, g⟩ ⟨hu, hM, hg, hrow⟩
    have hb := rs.bc_registered S R u hu M hM g hg hrow
    rw [hc] at hb
    exact hb
  · exact rs.other_registered S R u hu M hM g hg hrow

/-- **The value binding of rows under their own schemas**: `Layout.binding` at `scheme`, from `HmRowComputes`. -/
noncomputable def binding (hHm : rs.Computes) : P.ValueBinding plan tab decode H512 :=
  lay.binding rs.scheme (rs.registered hHm)

/-- **The rows as `HmRows`**: no commit string a row of one encoding, every one an `other` one with leaf scheme
`scheme`, registered by `registered`. -/
noncomputable def toHmRows (hHm : rs.Computes) : HmRows lay where
  isRow _ := False
  enc v := List.replicate (Fintype.equivFin lay.Val v).val 0
  enc_inj a b h := (Fintype.equivFin lay.Val).injective (Fin.ext (by simpa using congrArg List.length h))
  other := rs.scheme
  bcAt := rs.bcAt
  bc_registered _ _ _ _ _ _ _ _ h := h.elim
  other_registered S R u hu M hM g hg _ := rs.registered hHm S R u hu M hM g hg

/-- `toHmRows`'s `HmRowComputes` holds: it has no row. -/
theorem toHmRows_computes (hHm : rs.Computes) : (rs.toHmRows hHm).Computes := fun _ h => h.2.2.2.elim

/-- **`toHmRows` binds as the rows do.** -/
theorem toHmRows_binding (hHm : rs.Computes) :
    (rs.toHmRows hHm).binding (rs.toHmRows_computes hHm) = rs.binding hHm := rfl

end HmRowsP

open FlockSoundness.Discharge.Layout (BitVal widthOf)

variable (rd : Layout.Readers C plan tab) (place : ∀ S R u, u ∈ S → UnitPlace P (Partition.tabOf plan tab S R u).S u)

/-- **The readers' rows under their own schemas**: `Layout.Readers.rows` with the encoding `enc q` and extractor
`pair q` of each commit string. -/
noncomputable def rowsP (enc : rd.Pos → BitVal rd.W → List UInt8)
    (pair : rd.Pos → BitVal rd.W → BitVal rd.W → List UInt8 × List UInt8)
    (hpair : ∀ q v v', v ≠ v' → enc q v = enc q v' → Collides (pair q v v'))
    (encO : BitVal rd.W → List UInt8) (encO_inj : Function.Injective encO) (hcopy : rd.Copies place)
    (hbc : rd.BcPinned (P := P)) (hval : rd.ValPinned (P := P)) : HmRowsP (rd.layout place encO hcopy) where
  isRow := rd.isRow
  decRow := rd.decRow
  enc := enc
  pair := pair
  pair_spec := hpair
  other _ := ExplicitLeaf.plain encO encO_inj
  bcAt := rd.bcAt
  bc_registered := by
    intro S R u hu M hM g hg hrow
    have hb : bytesAt (Binding.wit plan tab S R u M) (rd.bcAt S R u (rd.pos g)) = rd.regBc R (rd.pos g) :=
      hbc S R u hu M hM g hg hrow
    show Refine.H512 (Refine.K512.leafPrefix ++ bytesAt (Binding.wit plan tab S R u M) (rd.bcAt S R u (rd.pos g))) =
      rd.regLeaf encO R (rd.pos g)
    rw [hb]
    simp only [Layout.Readers.regLeaf, show rd.isRow (rd.pos g) from hrow, ↓reduceIte]
  other_registered := by
    intro S R u hu M hM g hg hrow
    have hv : rd.valAt S R u M (rd.pos g) = rd.regVal R (rd.pos g) := hval S R u hu M hM g hg hrow
    show Refine.H512 (encO (widthOf rd.W (rd.rowBits (rd.pos g)), rd.valAt S R u M (rd.pos g))) =
      rd.regLeaf encO R (rd.pos g)
    rw [hv]
    simp only [Layout.Readers.regLeaf, show ¬ rd.isRow (rd.pos g) from hrow, ↓reduceIte]

end FlockSoundness.Discharge.ScopeW
