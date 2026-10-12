import Proofs.Flock.Soundness.Discharge.ZkBind.Bind
import Proofs.Flock.Soundness.Discharge.Layout.Keyed

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
  subst e
  exact hz

theorem block_witOf {T' T : Statement} (e : T' = T) (z : Witness T'.m) (o : Fin (2 ^ T'.nbl)) (c : Fin (2 ^ T'.kLog))
    (o' : Fin (2 ^ T.nbl)) (c' : Fin (2 ^ T.kLog)) (ho : o.val = o'.val) (hc : c.val = c'.val) :
    T'.block z o c = T.block (witOf e z) o' c' := by
  subst e
  obtain rfl : o = o' := Fin.ext ho
  obtain rfl : c = c' := Fin.ext hc
  rfl

/-- **Every witness of a `fast100` table's statement is one a message packs.** -/
theorem wit_surj {mPts : ℕ} (T : TabSpec mPts) (z : Witness T.S.m) :
    ∃ M : Message 𝔽 T.sch.k0 T.sch.level₀.logCols, witnessOf execArith T.S.m T.sch.k0 T.sch.level₀.logCols M = z := by
  obtain ⟨-, -, hcols, -⟩ := fast100_shape _ _ T.hsch
  have hmS := fast100_m _ _ T.hsch
  have hm := T.hm
  simp only [TabSpec.sch, Schedule.level₀] at hcols hmS ⊢
  set k := T.k0
  set L := ((T.l₀ :: T.l₁ :: T.rest).headD default).logCols
  have hpow : 2 ^ T.S.m = 2 ^ L * 2 ^ k * 128 := by
    rw [show (128 : ℕ) = 2 ^ 7 by norm_num, ← pow_add, ← pow_add]; congr 1; omega
  refine ⟨fun ℓ c => execArith.pack fun v => z ⟨(c.val * 2 ^ k + ℓ.val) * 128 + v.val, ?_⟩, ?_⟩
  · have := c.isLt; have := ℓ.isLt; have := v.isLt
    rw [hpow]
    calc (c.val * 2 ^ k + ℓ.val) * 128 + v.val < (c.val * 2 ^ k + ℓ.val + 1) * 128 := by nlinarith
      _ ≤ (c.val * 2 ^ k + 2 ^ k) * 128 := Nat.mul_le_mul_right 128 (by have := ℓ.isLt; omega)
      _ = (c.val + 1) * 2 ^ k * 128 := by ring
      _ ≤ 2 ^ L * 2 ^ k * 128 :=
        Nat.mul_le_mul_right 128 (Nat.mul_le_mul_right _ (by have := c.isLt; omega))
  · funext a
    have ha : a.val < 2 ^ L * 2 ^ k * 128 := hpow ▸ a.isLt
    have hc : a.val / 128 / 2 ^ k < 2 ^ L := by
      rw [Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _), Nat.div_lt_iff_lt_mul (by norm_num)]
      linarith
    simp only [witnessOf]
    rw [dite_eq_left_of_eq_true (eq_true hc), execArith_correct.unpack_pack]
    congr 1
    apply Fin.ext
    simp only
    rw [Nat.div_add_mod' (a.val / 128) (2 ^ k), Nat.div_add_mod']

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
  obtain ⟨M, hM⟩ := wit_surj (Partition.tabOf planS tabS S R u) (witOf (e S R u) (witZ execArith planZ tab S R u M'))
  refine ⟨M, hM, ?_⟩
  unfold Binding.Sat
  simp only [Binding.wit]
  rw [hM]
  exact sat_witOf (e S R u) hM'

variable (lay : Binding.Layout P planS tabS decS) (hdec : DecAgree e (decZ := decZ) (decS := decS) (P := P))

include hdec in
theorem ofShadow_decode_row :
    ∀ S R u, u ∈ S → ∀ M, SatZ execArith planZ tab S R u M → ∀ g ∈ P.io u,
      decZ S R u M g = lay.bit (lay.rowOf (lay.pos g) (bitsAt (witZ execArith planZ tab S R u M)
        (fun i => adrOf (e S R u) (lay.rowAt S R u (lay.pos g) i)))) g := by
  intro S R u hu M' hM' g hg
  obtain ⟨M, hM, hS⟩ := shadow_msg e S R u M' hM'
  rw [hdec S R u hu M' M hM hM' g hg, lay.decode_row S R u hu M hS g hg, hM]
  rfl

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
  simp only [LayoutZ.row, LayoutZ.salt, Binding.Layout.row, Binding.Layout.salt, hM]
  exact ⟨rfl, rfl⟩

variable {lay} (rs : Binding.HmRows lay)

theorem ofShadow_bc_registered :
    ∀ S R u, u ∈ S → ∀ M, SatZ execArith planZ tab S R u M → ∀ g ∈ P.io u, rs.isRow (lay.pos g) →
      H512 (K512.leafPrefix ++ bytesAt (witZ execArith planZ tab S R u M)
        (fun j t => adrOf (e S R u) (rs.bcAt S R u (lay.pos g) j t))) = lay.regLeaf R (lay.pos g) := by
  intro S R u hu M' hM' g hg hrow
  obtain ⟨M, hM, hS⟩ := shadow_msg e S R u M' hM'
  have h := rs.bc_registered S R u hu M hS g hg hrow
  rw [hM] at h
  exact h

theorem ofShadow_other_registered :
    ∀ S R u, u ∈ S → ∀ M, SatZ execArith planZ tab S R u M → ∀ g ∈ P.io u, ¬ rs.isRow (lay.pos g) →
      (rs.other (lay.pos g)).leaf ((LayoutZ.ofShadow e lay hdec).row S R u M (lay.pos g))
        ((LayoutZ.ofShadow e lay hdec).salt S R u M (lay.pos g)) = lay.regLeaf R (lay.pos g) := by
  intro S R u hu M' hM' g hg hrow
  obtain ⟨M, hM, hS⟩ := shadow_msg e S R u M' hM'
  obtain ⟨hr, hs⟩ := row_ofShadow e lay hdec S R u M' M hM (lay.pos g)
  rw [hr, hs]
  exact rs.other_registered S R u hu M hS g hg hrow

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
  rintro ⟨S, R, u, M', g⟩ ⟨hu, hM', hg, hrow⟩
  obtain ⟨M, hM, hS⟩ := shadow_msg e S R u M' hM'
  obtain ⟨hr, hs⟩ := row_ofShadow e lay hdec S R u M' M hM (lay.pos g)
  have h := hHm ⟨S, R, u, M, g⟩ ⟨hu, hS, hg, hrow⟩
  simp only at h
  rw [← hr, ← hs, hM] at h
  exact h

end Shadow

end FlockSoundness.Discharge.ZkBind
