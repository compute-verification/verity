import Proofs.Flock.Soundness.Binding.Rows
import Proofs.Flock.Soundness.Discharge.Layout.Readers

/-!
# A layout and its rows from three pinning facts

`Prog.flock_headline` takes the layout's facts as `lay : Binding.Layout` and `rs : Binding.HmRows lay`. This file builds
both from fixed readers (`Readers`: each commit string's row, salt and `b ‖ c` addresses in a drawn unit's table, which
commit strings are `hm96-sha512/row/v1` rows, and what the registration holds at each: the public `b ‖ c` of a row, the
value of any other commit string) and three facts about every satisfying witness of a drawn unit's table:

* `Readers.Copies`: each column of the unit on a committed wire holds the bit the witness has at the wire's row address
  (Δ's copy into the row's message bit for an input; the column itself for an output);
* `Readers.BcPinned`: at a row, the 128 bytes at `bcAt` are the registration's `b ‖ c` (the statement's regions pin the
  `hm96` slots' outputs to the public `b ‖ c`);
* `Readers.ValPinned`: at another commit string, the bits at `rowAt` are the registration's value.

The registered leaf (`Readers.regLeaf`) is `SHA-512(leaf_prefix ‖ b ‖ c)` of the registration's `b ‖ c` at a row, the
leaf the verifier computes from the public `b ‖ c` and holds, in its frame-v3-sha512 tree, to the public file's header
root (`HmRow.checkPublic`, stated by `FailClosed.flock_verify_public_ties`; that root is the registration's by the
auditor's `matches_public`, not by anything here), and `SHA-512(enc v)` of the registration's
value elsewhere. That the serving roots commit to these leaves is the serving roots' rewinding (`StrictCR.TreeRewind`),
paid once in the headline as `δ_tree`; nothing here is cryptographic.
-/

namespace FlockSoundness.Discharge.Layout

open Game Model Audit Binding

variable {C : Circuit} {n : ℕ} {Reg : Type} {mPts : ℕ}

/-- **Fixed readers and the registration's values.** The commit strings (`Pos`), the commit string holding each wire
(`pos`) and its bit there (`idx`); where a drawn unit's table's witness holds bit `k < rowBits q` of each commit string
`q` (`rowAt`), its salt (`saltAt`) and, at a row (`isRow`), its `hm96` slot's 128 output bytes (`bcAt`); and what the
registration holds: a row's `b ‖ c` (`regBc`), another commit string's value (`regVal`). All are fixed by the draw, the
registration, the unit and the commit string, before any witness. -/
structure Readers (C : Circuit) {n : ℕ} {Reg : Type} {mPts : ℕ} (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length) where
  Pos : Type
  [finPos : Fintype Pos]
  [decPos : DecidableEq Pos]
  [inhPos : Inhabited Pos]
  W : ℕ
  pos : Fin C.N → Pos
  idx : Fin C.N → Fin W
  rowBits : Pos → ℕ
  hidx : ∀ g, (idx g).val < rowBits (pos g)
  rowAt : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → Pos → ℕ → Fin (2 ^ (Partition.tabOf plan tab S R u).S.m)
  saltAt : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → Pos → Fin 192 → Fin 8 →
    Fin (2 ^ (Partition.tabOf plan tab S R u).S.m)
  bcAt : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → Pos → Fin 128 → Fin 8 →
    Fin (2 ^ (Partition.tabOf plan tab S R u).S.m)
  isRow : Pos → Prop
  [decRow : DecidablePred isRow]
  regBc : Reg → Pos → List UInt8
  regVal : Reg → Pos → Fin W → Bool

attribute [instance] Readers.finPos Readers.decPos Readers.inhPos Readers.decRow

namespace Readers

variable {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  (rd : Readers C plan tab) {P : Audit.Partition C n}
  (place : ∀ S R u, u ∈ S → UnitPlace P (Partition.tabOf plan tab S R u).S u)

/-- The value a witness holds at commit string `q`: its bits at `rowAt`, `false` past them. -/
noncomputable def valAt (S : Finset (Fin n)) (R : Reg) (u : Fin n) (M : Binding.Msg plan tab S R u) (q : rd.Pos) :
    Fin rd.W → Bool :=
  padBits (bitsAt (Binding.wit plan tab S R u M) fun k : Fin (rd.rowBits q) => rd.rowAt S R u q k.val)

/-- **Δ's copies**: in a satisfying witness, each column of a drawn unit on a committed wire holds the witness's bit at
the wire's row address. -/
def Copies : Prop :=
  ∀ S R u (hu : u ∈ S) M, Binding.Sat plan tab S R u M → ∀ c, (place S R u hu).inst.wire c ∈ P.io u →
    (Partition.tabOf plan tab S R u).S.block (Binding.wit plan tab S R u M) (place S R u hu).o ((place S R u hu).col c) =
      Binding.wit plan tab S R u M (rd.rowAt S R u (rd.pos ((place S R u hu).inst.wire c))
        (rd.idx ((place S R u hu).inst.wire c)).val)

/-- **The rows' `b ‖ c` is pinned**: in a satisfying witness, at each of a drawn unit's committed wires held by a row,
the bytes at `bcAt` are the registration's `b ‖ c`. -/
def BcPinned : Prop :=
  ∀ S R u, u ∈ S → ∀ M, Binding.Sat plan tab S R u M → ∀ g ∈ P.io u, rd.isRow (rd.pos g) →
    bytesAt (Binding.wit plan tab S R u M) (rd.bcAt S R u (rd.pos g)) = rd.regBc R (rd.pos g)

/-- **The other values are pinned**: in a satisfying witness, at each of a drawn unit's committed wires held by another
commit string, the value at `rowAt` is the registration's. -/
def ValPinned : Prop :=
  ∀ S R u, u ∈ S → ∀ M, Binding.Sat plan tab S R u M → ∀ g ∈ P.io u, ¬ rd.isRow (rd.pos g) →
    rd.valAt S R u M (rd.pos g) = rd.regVal R (rd.pos g)

/-- **The registered leaves**: `SHA-512(leaf_prefix ‖ b ‖ c)` of the registration's `b ‖ c` at a row, the leaf the
verifier computes from the public `b ‖ c`; `SHA-512(encO v)` of the registration's value `v`, at its width, elsewhere. -/
def regLeaf (encO : BitVal rd.W → List UInt8) (R : Reg) (q : rd.Pos) : Refine.Dig :=
  if rd.isRow q then Refine.H512 (Refine.K512.leafPrefix ++ rd.regBc R q)
  else Refine.H512 (encO (widthOf rd.W (rd.rowBits q), rd.regVal R q))

/-- **The layout** of the readers' bit-string values, for the decoder that reads each drawn unit from its place, from
Δ's copies. -/
noncomputable def layout (encO : BitVal rd.W → List UInt8) (hcopy : rd.Copies place) :
    Binding.Layout P plan tab (P.placeDecoder plan tab place) :=
  ofBits place rd.Pos rd.W rd.pos rd.idx (rd.regLeaf encO) rd.rowBits rd.hidx
    (fun S R u q k => rd.rowAt S R u q k.val) rd.saltAt hcopy

/-- **The layout's rows**: `hm96-sha512/row/v1` at the readers' rows, with `enc` the bytes `sha512/row/v1` hashes, and a
plain leaf `SHA-512(encO v)` elsewhere. `bc_registered` is `BcPinned` and `other_registered` is `ValPinned`. -/
noncomputable def rows (enc : BitVal rd.W → List UInt8) (enc_inj : Function.Injective enc)
    (encO : BitVal rd.W → List UInt8) (encO_inj : Function.Injective encO) (hcopy : rd.Copies place)
    (hbc : rd.BcPinned (P := P)) (hval : rd.ValPinned (P := P)) : Binding.HmRows (rd.layout place encO hcopy) where
  isRow := rd.isRow
  decRow := rd.decRow
  enc := enc
  enc_inj := enc_inj
  other _ := ExplicitLeaf.plain encO encO_inj
  bcAt := rd.bcAt
  bc_registered := by
    intro S R u hu M hM g hg hrow
    have hb : bytesAt (Binding.wit plan tab S R u M) (rd.bcAt S R u (rd.pos g)) = rd.regBc R (rd.pos g) :=
      hbc S R u hu M hM g hg hrow
    show Refine.H512 (Refine.K512.leafPrefix ++ bytesAt (Binding.wit plan tab S R u M) (rd.bcAt S R u (rd.pos g))) =
      rd.regLeaf encO R (rd.pos g)
    rw [hb]
    simp only [regLeaf, show rd.isRow (rd.pos g) from hrow, ↓reduceIte]
  other_registered := by
    intro S R u hu M hM g hg hrow
    have hv : rd.valAt S R u M (rd.pos g) = rd.regVal R (rd.pos g) := hval S R u hu M hM g hg hrow
    show Refine.H512 (encO (widthOf rd.W (rd.rowBits (rd.pos g)), rd.valAt S R u M (rd.pos g))) =
      rd.regLeaf encO R (rd.pos g)
    rw [hv]
    simp only [regLeaf, show ¬ rd.isRow (rd.pos g) from hrow, ↓reduceIte]

/-- The layout's commit string of a wire is the readers'. -/
theorem layout_pos (encO : BitVal rd.W → List UInt8) (hcopy : rd.Copies place) (g : Fin C.N) :
    (rd.layout place encO hcopy).pos g = rd.pos g := rfl

/-- The layout's registered leaf is the readers'. -/
theorem layout_regLeaf (encO : BitVal rd.W → List UInt8) (hcopy : rd.Copies place) (R : Reg) (q : rd.Pos) :
    (rd.layout place encO hcopy).regLeaf R q = rd.regLeaf encO R q := rfl

/-- **The value a message holds at a commit string** is the readers' `valAt`, at the commit string's width. -/
theorem layout_row (encO : BitVal rd.W → List UInt8) (hcopy : rd.Copies place) (S : Finset (Fin n)) (R : Reg)
    (u : Fin n) (M : Binding.Msg plan tab S R u) (q : rd.Pos) :
    (rd.layout place encO hcopy).row S R u M q = (widthOf rd.W (rd.rowBits q), rd.valAt S R u M q) := rfl

end Readers

end FlockSoundness.Discharge.Layout
