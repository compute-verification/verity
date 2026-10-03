import FlockSoundness.Discharge.ZkBind.Bind
import FlockSoundness.Discharge.Layout.Keyed

/-!
# A `--zk` layout from a public one at the same statements

`Binding.Layout` and `Binding.HmRows` are about plans of `TabSpec` tables, `LayoutZ` and `HmRowsZ` about plans of
`TabZK` tables. Their facts read a drawn unit's table only through its statement and the witness a message packs. So a
public layout at plan `planS` gives a `--zk` one at plan `planZ` whenever every drawn unit's `--zk` table has the
statement of its public table (`e`): the addresses move along `e` (`adr`), and every witness of the public table's
statement is one a message packs (`wit_surj`, the `fast100` schedule's `m = 7 + k₀ + cols`), so a fact about every
satisfying public message is one about every satisfying `--zk` message.

* `LayoutZ.ofShadow`: the `--zk` layout, given that the two decoders agree on messages packing one witness (`hdec`);
* `HmRowsZ.ofShadow`: its rows, and `HmRowsZ.ofShadow_computes`: `HmRowComputes` moves along.
-/

namespace FlockSoundness.Discharge.ZkBind

open Game Model Accounting Refine Audit ZkSession ZkLink Binding

local notation "𝔽" => FlockLevel3.GF128

/-! ## Witnesses along an equality of statements -/

/-- An address of `T`'s witness as one of `T'`'s, along `T' = T`. -/
def adrOf {T' T : Statement} (e : T' = T) (a : Fin (2 ^ T.m)) : Fin (2 ^ T'.m) :=
  Fin.cast (congrArg (fun S : Statement => 2 ^ S.m) e).symm a

/-- A witness of `T'` as one of `T`, along `T' = T`. -/
def witOf {T' T : Statement} (e : T' = T) (z : Witness T'.m) : Witness T.m := fun a => z (adrOf e a)

theorem sat_witOf {T' T : Statement} (e : T' = T) {z : Witness T'.m} (hz : T'.Satisfies z) :
    T.Satisfies (witOf e z) := by
  sorry

theorem block_witOf {T' T : Statement} (e : T' = T) (z : Witness T'.m) (o : Fin (2 ^ T'.nbl)) (c : Fin (2 ^ T'.kLog))
    (o' : Fin (2 ^ T.nbl)) (c' : Fin (2 ^ T.kLog)) (ho : o.val = o'.val) (hc : c.val = c'.val) :
    T'.block z o c = T.block (witOf e z) o' c' := by
  sorry

/-- **Every witness of a `fast100` table's statement is one a message packs.** -/
theorem wit_surj {mPts : ℕ} (T : TabSpec mPts) (z : Witness T.S.m) :
    ∃ M : Message 𝔽 T.sch.k0 T.sch.level₀.logCols, witnessOf execArith T.S.m T.sch.k0 T.sch.level₀.logCols M = z := by
  sorry

/-! ## The `--zk` layout of a public one -/

section Shadow

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length}
  {planS : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tabS : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planS S R).length}
  (e : ∀ S R u, (tabOfZ planZ tab S R u).S = (Partition.tabOf planS tabS S R u).S)
  {decZ : DecoderZK (C := C) planZ tab} {decS : Partition.DecoderB (C := C) (Reg := Reg) planS tabS}

/-- The decoders agree on messages that pack one witness. -/
def DecAgree : Prop :=
  ∀ S R u, u ∈ S → ∀ (M' : MsgZ planZ tab S R u) (M : Binding.Msg planS tabS S R u),
    Binding.wit planS tabS S R u M = witOf (e S R u) (witZ execArith planZ tab S R u M') →
    SatZ execArith planZ tab S R u M' → ∀ g ∈ P.io u, decZ S R u M' g = decS S R u M g

/-- A satisfying `--zk` message's witness is a satisfying public message's. -/
theorem shadow_msg (S : Finset (Fin n)) (R : Reg) (u : Fin n) (M' : MsgZ planZ tab S R u)
    (hM' : SatZ execArith planZ tab S R u M') :
    ∃ M : Binding.Msg planS tabS S R u,
      Binding.wit planS tabS S R u M = witOf (e S R u) (witZ execArith planZ tab S R u M') ∧
        Binding.Sat planS tabS S R u M := by
  sorry

variable (lay : Binding.Layout P planS tabS decS) (hdec : DecAgree e (decZ := decZ) (decS := decS) (P := P))

include hdec in
theorem ofShadow_decode_row :
    ∀ S R u, u ∈ S → ∀ M, SatZ execArith planZ tab S R u M → ∀ g ∈ P.io u,
      decZ S R u M g = lay.bit (lay.rowOf (lay.pos g) (bitsAt (witZ execArith planZ tab S R u M)
        (fun i => adrOf (e S R u) (lay.rowAt S R u (lay.pos g) i)))) g := by
  sorry

/-- **The `--zk` layout of a public layout at the same statements.** -/
noncomputable def LayoutZ.ofShadow : LayoutZ execArith P planZ tab decZ where
  Pos := lay.Pos
  Val := lay.Val
  pos := lay.pos
  bit := lay.bit
  regLeaf := lay.regLeaf
  rowBits := lay.rowBits
  rowOf := lay.rowOf
  rowAt S R u p i := adrOf (e S R u) (lay.rowAt S R u p i)
  saltAt S R u p j t := adrOf (e S R u) (lay.saltAt S R u p j t)
  decode_row := ofShadow_decode_row e lay hdec

theorem row_ofShadow (S : Finset (Fin n)) (R : Reg) (u : Fin n) (M' : MsgZ planZ tab S R u)
    (M : Binding.Msg planS tabS S R u)
    (hM : Binding.wit planS tabS S R u M = witOf (e S R u) (witZ execArith planZ tab S R u M')) (p : lay.Pos) :
    (LayoutZ.ofShadow e lay hdec).row S R u M' p = lay.row S R u M p ∧
      (LayoutZ.ofShadow e lay hdec).salt S R u M' p = lay.salt S R u M p := by
  sorry

variable {lay} (rs : Binding.HmRows lay)

theorem ofShadow_bc_registered :
    ∀ S R u, u ∈ S → ∀ M, SatZ execArith planZ tab S R u M → ∀ g ∈ P.io u, rs.isRow (lay.pos g) →
      H512 (K512.leafPrefix ++ bytesAt (witZ execArith planZ tab S R u M)
        (fun j t => adrOf (e S R u) (rs.bcAt S R u (lay.pos g) j t))) = lay.regLeaf R (lay.pos g) := by
  sorry

theorem ofShadow_other_registered :
    ∀ S R u, u ∈ S → ∀ M, SatZ execArith planZ tab S R u M → ∀ g ∈ P.io u, ¬ rs.isRow (lay.pos g) →
      (rs.other (lay.pos g)).leaf ((LayoutZ.ofShadow e lay hdec).row S R u M (lay.pos g))
        ((LayoutZ.ofShadow e lay hdec).salt S R u M (lay.pos g)) = lay.regLeaf R (lay.pos g) := by
  sorry

/-- **The `--zk` rows of a public layout's rows.** -/
noncomputable def HmRowsZ.ofShadow : HmRowsZ (LayoutZ.ofShadow e lay hdec) where
  isRow := rs.isRow
  decRow := rs.decRow
  enc := rs.enc
  enc_inj := rs.enc_inj
  other := rs.other
  bcAt S R u p j t := adrOf (e S R u) (rs.bcAt S R u p j t)
  bc_registered := ofShadow_bc_registered e rs
  other_registered := ofShadow_other_registered e hdec rs

/-- **`HmRowComputes` moves along**: the public rows' `HmRowComputes` gives the `--zk` rows'. -/
theorem HmRowsZ.ofShadow_computes (hHm : rs.Computes) : (HmRowsZ.ofShadow e hdec rs).Computes := by
  sorry

end Shadow

end FlockSoundness.Discharge.ZkBind
