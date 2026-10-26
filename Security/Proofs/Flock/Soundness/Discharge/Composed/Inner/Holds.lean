import Proofs.Flock.Soundness.Discharge.Composed.Coined
import Proofs.Flock.Soundness.Discharge.Composed.Inner.Lane
import Proofs.Flock.Soundness.Discharge.Composed.Inner.ZerocheckGlue
import Proofs.Flock.Soundness.Discharge.Composed.Inner.RingSwitch
import Proofs.Flock.Soundness.ExecSetup

/-!
# `innerOf_holds`: the honest prover's pads satisfy the inner proof's constraint at the draw

`InnerHolds` is `⟨λ, h⟩ = θ` for `verifyInner`'s batch of rep `r`'s ten rows (`lamOf`, `thetaOf`), at the honest pads
`hFull` (`h_ab = h_fa·h_fb`, `h_tw = h_tx·h_ty`). By `batch_eval` it holds once every row vanishes at the pads by layout
index (`batch_of_rows`). The triple's three rows vanish at the honest `(ρ_in, σ_in)` (`triple_rows_zero`).
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock Model ZkSession ZkLink ZkExec DecodeZL FlockLevel3 ZkView

/-- **The batch holds where its rows vanish**: `⟨λ, h⟩ = θ` over `PadIdx` once each row vanishes at `h` by layout index
(`padEquiv`). -/
theorem batch_of_rows {F : Type} [Field F] {nE n : ℕ} (γ : Fin n → F) (R : Fin n → ZkSession.Lin F (nE + 4))
    (hF : ZK.PadIdx nE → F) (h0 : ∀ k, (R k).eval (hF ∘ (padEquiv nE).symm) = 0) :
    ∑ i, batchCoef γ R (padEquiv nE i) * hF i = batchT γ R := by
  have hb := batch_eval γ R (hF ∘ (padEquiv nE).symm)
  have hz : ∑ k, γ k * (R k).eval (hF ∘ (padEquiv nE).symm) = 0 :=
    Finset.sum_eq_zero fun k _ => by rw [h0 k, mul_zero]
  rw [hz, sub_eq_zero] at hb
  rw [← hb, ← Equiv.sum_comp (padEquiv nE)]
  simp

/-- **The triple's rows vanish** at the honest `ρ_in = ε·h_fa + h_tx`, `σ_in = h_fb + h_ty`, where `h_ab = h_fa·h_fb`
and `h_tw = h_tx·h_ty` (characteristic 2). -/
theorem triple_rows_zero {F : Type} [Field F] [CharP F 2] {s : ℕ} (tr : Triple s) (ε : F) (h : Fin s → F)
    (hab : h tr.ab = h tr.fa * h tr.fb) (htw : h tr.tw = h tr.tx * h tr.ty) (k : Fin 3) :
    (tripleRows tr ε (ε * h tr.fa + h tr.tx) (h tr.fb + h tr.ty) k).eval h = 0 := by
  have h2 : (2 : F) = 0 := CharTwo.two_eq_zero
  fin_cases k
  · rw [show ((⟨0, by omega⟩ : Fin 3)) = 0 from rfl, tripleRows_eval_zero]
    linear_combination (ε * h tr.fa + h tr.tx) * h2
  · rw [show ((⟨1, by omega⟩ : Fin 3)) = 1 from rfl, tripleRows_eval_one]
    linear_combination (h tr.fb + h tr.ty) * h2
  · rw [show ((⟨2, by omega⟩ : Fin 3)) = 2 from rfl, tripleRows_eval_two, hab, htw]
    linear_combination (ε * h tr.fa * h tr.fb + h tr.tx * h tr.ty + h tr.fb * h tr.tx + ε * h tr.fa * h tr.ty +
      h tr.tx * h tr.ty) * h2

section Setup

open FlockSoundness.Refine

theorem avoidsMask_delta {st : Stmt} (h : avoidsMask st = true) : deltaAvoidsMask st = true := by
  simp only [avoidsMask, regionsAvoidMask, Bool.and_eq_true] at h; exact h.1

theorem avoidsMask_cells {st : Stmt} (h : avoidsMask st = true) : regionCellsOff st = true := by
  simp only [avoidsMask, regionsAvoidMask, Bool.and_eq_true] at h; exact h.2.1

theorem avoidsMask_words {st : Stmt} (h : avoidsMask st = true) : regionWordsOff st = true := by
  simp only [avoidsMask, regionsAvoidMask, Bool.and_eq_true] at h; exact h.2.2

theorem regionPoint_fixed (st : Stmt) (reg : Region) (r : Array F128) {j : ℕ} (hj : j < st.c.kLog)
    (hjm : j < st.m) (hf : j ∉ reg.free.toList) :
    (st.regionPoint reg r)[j]? =
      some (if (reg.mask >>> j) % 2 == 1 && (reg.fixed >>> j) % 2 == 1 then 1 else 0) := by
  unfold Stmt.regionPoint
  simp only [Id.run]
  refine forIn_id_inv_range (fun p : Array F128 => p[j]? = some _) (fun i _ hi p hp => ⟨_, rfl, ?_⟩)
    (forIn_id_inv_range (fun p : Array F128 => p[j]? = some _) (fun t _ ht p hp => ⟨_, rfl, ?_⟩) ?_)
  · rw [Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds_ne (by omega)]; exact hp
  · have hm : reg.free[t]! ∈ reg.free.toList := by
      rw [getElem!_pos reg.free t (by omega)]; exact Array.getElem_mem_toList _
    rw [Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds_ne (by intro h; apply hf; rw [← h]; exact hm)]; exact hp
  · simp [hjm, hj]

theorem mul_zero_F128 (a : F128) : a * 0 = 0 := fOf_injective (by rw [fOf_mul, fOf_zero, mul_zero])
theorem zero_mul_F128 (a : F128) : 0 * a = 0 := fOf_injective (by rw [fOf_mul, fOf_zero, zero_mul])
theorem one_add_one' : (1 : F128) + 1 = 0 :=
  fOf_injective (by rw [fOf_add, fOf_one, fOf_zero]; exact CharTwo.add_self_eq_zero 1)

theorem foldl_zero (f : ℕ → F128) : ∀ L : List ℕ, L.foldl (fun acc i => acc * f i) 0 = 0
  | [] => rfl
  | i :: L => by rw [List.foldl_cons, zero_mul_F128]; exact foldl_zero f L

theorem foldl_mul_eq_zero (f : ℕ → F128) : ∀ (L : List ℕ) (a : F128), (∃ i ∈ L, f i = 0) →
    L.foldl (fun acc i => acc * f i) a = 0
  | [], _, ⟨_, h, _⟩ => absurd h List.not_mem_nil
  | i :: L, a, ⟨k, hk, hf⟩ => by
      rw [List.foldl_cons]
      rcases List.mem_cons.1 hk with rfl | hk
      · rw [hf, mul_zero_F128]; exact foldl_zero f L
      · exact foldl_mul_eq_zero f L _ ⟨k, hk, hf⟩

theorem regions_off_of_words {st : Stmt} (hm : st.m = st.c.kLog + st.nbl) (hw : regionWordsOff st = true)
    (link : Fin 2 → Fin st.mPts → F128) (j : Fin (2 * st.regions.size)) (x : ℕ) (hx : x ∈ maskWordsOf st) :
    eqWord (regionPtOf st link j) x = 0 := by
  have hj : j.val / 2 < st.regions.size := by omega
  have hreg : st.regions[j.val / 2]! ∈ st.regions.toList := by
    rw [getElem!_pos st.regions _ hj]; exact Array.getElem_mem_toList _
  obtain ⟨i, hi, hb⟩ := List.any_eq_true.1 (List.all_eq_true.1 (List.all_eq_true.1 hw _ hreg) x hx)
  rw [List.mem_range] at hi
  simp only [Bool.and_eq_true, Bool.not_eq_true', bne_iff_ne, ne_eq] at hb
  obtain ⟨hc, hne⟩ := hb
  have hnot : i + 7 ∉ (st.regions[j.val / 2]!).free.toList := by
    intro h
    have h' : (st.regions[j.val / 2]!).free.contains (i + 7) = true := by simpa using Array.mem_toList_iff.1 h
    rw [h'] at hc; cases hc
  have hfix := regionPoint_fixed st (st.regions[j.val / 2]!)
    (Array.ofFn (link (if j.val % 2 = 0 then 0 else 1))) (by omega) (by omega) hnot
  generalize (((st.regions[j.val / 2]!).mask >>> (i + 7)) % 2 == 1 &&
    ((st.regions[j.val / 2]!).fixed >>> (i + 7)) % 2 == 1) = bv at hfix hne
  unfold eqWord regionPtOf
  generalize hq : (st.regionPoint st.regions[j.val / 2]! (Array.ofFn (link (if j.val % 2 = 0 then 0 else 1)))).toList.drop 7 = q
  have hqi : q[i]? = some (if bv = true then 1 else 0) := by rw [← hq, List.getElem?_drop, Array.getElem?_toList, Nat.add_comm]; exact hfix
  have hlen : i < q.length := (List.getElem?_eq_some_iff.1 hqi).1
  have hv : q[i]! = (if bv = true then 1 else 0) := by rw [getElem!_pos q i hlen]; exact Option.some.inj ((List.getElem?_eq_getElem hlen).symm.trans hqi)
  refine foldl_mul_eq_zero _ _ _ ⟨i, List.mem_range.2 hlen, ?_⟩
  try dsimp only
  rw [hv]
  revert hne
  cases bv <;> cases x.testBit i <;> simp [one_add_one', add_zero]

/-- **The regions avoid the mask slot**: at every link point, a region claim's `eq` weight vanishes on every mask-slot
word, since the region fixes a word bit where the word differs (`regionWordsOff`). -/
theorem setupH_regions_off_mask {tags : Tags} {circuitFile publicFile : ByteArray} {coins : Coins}
    {tables : List (String × ByteArray)} {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupH tags circuitFile publicFile coins tables draw partition program = .ok st)
    (hdm : avoidsMask st = true)
    (link : Fin 2 → Fin st.mPts → F128) (j : Fin (2 * st.regions.size)) (x : ℕ) (hx : x ∈ maskWordsOf st) :
    eqWord (regionPtOf st link j) x = 0 :=
  regions_off_of_words (Refine.setupH_mask h).2 (avoidsMask_words hdm) link j x hx

end Setup

/-- **The mask slot's words are distinct and inside the witness**, its slot at least a word and inside the block. -/
theorem maskWordsOf_ok (st : Stmt) (hm : st.m = st.c.kLog + st.nbl)
    (hmask : ∀ r ∈ st.c.ranges.toList, r.kind = Kind.mask → 7 ≤ r.slotLog ∧ r.start + 2 ^ r.slotLog ≤ 2 ^ st.c.kLog) :
    (maskWordsOf st).Nodup ∧ ∀ x ∈ maskWordsOf st, x < 2 ^ (st.m - 7) := by
  unfold maskWordsOf
  split
  · exact ⟨List.nodup_nil, fun _ h => absurd h List.not_mem_nil⟩
  · rename_i r hr
    have hmem := List.mem_of_find?_eq_some hr
    have hk := List.find?_some hr
    have hk : r.kind = Kind.mask := by
      cases hkd : r.kind <;> rw [hkd] at hk
      · exact absurd hk (by decide)
      · exact absurd hk (by intro h; cases h)
    obtain ⟨h7, hb⟩ := hmask r hmem hk
    have hsk : r.slotLog ≤ st.c.kLog := by
      by_contra hc
      have : 2 ^ st.c.kLog < 2 ^ r.slotLog := Nat.pow_lt_pow_right (by norm_num) (by omega)
      omega
    have hJ : 2 ^ r.slotLog = 128 * 2 ^ (r.slotLog - 7) := by
      rw [show (128 : ℕ) = 2 ^ 7 by norm_num, ← pow_add, Nat.add_sub_cancel' h7]
    have hK : 2 ^ st.c.kLog = 128 * 2 ^ (st.c.kLog - 7) := by
      rw [show (128 : ℕ) = 2 ^ 7 by norm_num, ← pow_add, Nat.add_sub_cancel' (by omega)]
    have hoJ : r.start / 128 + 2 ^ (r.slotLog - 7) ≤ 2 ^ (st.c.kLog - 7) := by omega
    have hM : 2 ^ (st.m - 7) = 2 ^ st.nbl * 2 ^ (st.c.kLog - 7) := by
      rw [← pow_add]; congr 1; omega
    set K := 2 ^ (st.c.kLog - 7)
    set J := 2 ^ (r.slotLog - 7)
    set o := r.start / 128
    have hlt : ∀ b < 2 ^ st.nbl, ∀ j < J, b * K + o + j < (b + 1) * K := fun b _ j hj => by
      rw [Nat.add_mul, one_mul]; omega
    refine ⟨?_, ?_⟩
    · rw [List.nodup_flatMap]
      refine ⟨fun b _ => (List.nodup_range).map fun j j' h => by simpa using h, ?_⟩
      refine List.Pairwise.imp_of_mem ?_ (List.pairwise_lt_range (n := 2 ^ st.nbl))
      intro a b ha hb' hab
      show List.Disjoint _ _
      rw [List.disjoint_left]
      intro x hxa hxb
      simp only [List.mem_map, List.mem_range] at hxa hxb
      obtain ⟨j, hj, rfl⟩ := hxa
      obtain ⟨j', hj', hx⟩ := hxb
      have h1 := hlt a (List.mem_range.1 ha) j hj
      have h2 : (a + 1) * K ≤ b * K := Nat.mul_le_mul_right _ hab
      have h3 : b * K ≤ b * K + o + j' := by omega
      omega
    · intro x hx
      simp only [List.mem_flatMap, List.mem_map, List.mem_range] at hx
      obtain ⟨b, hb2, j, hj, rfl⟩ := hx
      have h1 := hlt b hb2 j hj
      have h2 : (b + 1) * K ≤ 2 ^ st.nbl * K := Nat.mul_le_mul_right _ hb2
      rw [hM]; omega

section Glue

open Flock Model ZkSession ZkView ZK Refine

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-- Pads as the executable's array. -/
def arrOf {s : ℕ} (h : Fin s → 𝔽) : Array F128 := Array.ofFn fun i => (h i).val

theorem arrOf_get {s : ℕ} (h : Fin s → 𝔽) (k : ℕ) (hk : k < s) : fOf (arrOf h)[k]! = h ⟨k, hk⟩ := by
  rw [arrOf, getElem!_pos _ k (by simpa using hk), Array.getElem_ofFn]
  rfl

theorem linVal_eq_evalL {s : ℕ} (h : Fin s → 𝔽) (l : Zk.Lin) (hl : l.a.size = s) :
    Lincheck.linVal h l = fOf (ZerocheckMasked.evalL l (arrOf h)) := by
  refine Eq.trans ?_ (ZerocheckMasked.of_evalL l _ s hl).symm
  exact congrArg (fOf l.c + ·) (Finset.sum_congr rfl fun i _ => congrArg (fOf l.a[i.val]! * ·) (arrOf_get h i i.isLt).symm)

/-- The pads on the layout's `len` slots. -/
def padsAt {nE s : ℕ} (hs : s = nE + 4) (hp : Fin (nE + 4) → 𝔽) : Fin s → 𝔽 := fun i => hp (Fin.cast hs i)

theorem ofZk_eval {nE s : ℕ} (hs : s = nE + 4) (hp : Fin (nE + 4) → 𝔽) (l : Zk.Lin) :
    (ofZk l : ZkSession.Lin 𝔽 (nE + 4)).eval hp = Lincheck.linVal (padsAt hs hp) l := by
  subst hs
  simp only [ZkSession.Lin.eval, ofZk, Lincheck.linVal, padsAt]
  rw [add_comm]
  rfl


theorem val_mul_cast (x y : ZMod 2) : (((x * y).val : ℕ) : 𝔽) = ((x.val : ℕ) : 𝔽) * ((y.val : ℕ) : 𝔽) := by
  simp only [ZMod.natCast_val]
  exact ZMod.cast_mul (dvd_refl 2) x y

/-- `modelFold` realizes the statement by construction. -/
theorem modelFold_realizes (S : Statement) : FoldRealizes (modelFold S) S where
  kLog := rfl
  pin := rfl
  fold α lw ρ v _ _ hv col := by
    simp only [modelFold, Except.ok.injEq] at hv
    subst hv
    rw [getElem!_pos _ col.val (by simp), Array.getElem_map, Array.getElem_range, dif_pos col.isLt]
    rfl
  folded α β lw ρ ts w hw := by
    refine ⟨_, rfl, by simp, ?_⟩
    simp only [modelFold, Except.ok.injEq] at hw
    exact hw.symm

section Rows

variable {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2)
  (zw : Witness Tz.S.m) {nE L : ℕ} (R : Fin 2 → Fin L → 𝔽) (hp : Fin (nE + 4) → 𝔽)

theorem ix_eq (k : ℕ) (hk : k < nE + 4) : ix nE k = ⟨k, hk⟩ := Fin.ext (Nat.mod_eq_of_lt hk)

/-- A masked value is its clear value plus its pad (characteristic 2). -/
theorem msgAt_honest (k : ℕ) (hk : k < nE) :
    msgAt (honestMasked Tz st c r zw R hp) k = honestAt Tz st c r zw R k + hp ⟨k, by omega⟩ := by
  rw [← msgAt_add_pad Tz st c r zw R hp k hk, add_assoc, ix_eq k (by omega), CharTwo.add_self_eq_zero, add_zero]

/-! ### The honest values at the zerocheck's and the lincheck's indices -/

theorem honestAt_r1ab (i : ℕ) (hi : i < 64) :
    honestAt Tz st c r zw R ((layOf Tz st).r1ab i) = fOf (zcH Tz st c r zw).round1ab[i]! := by
  unfold honestAt; dsimp only
  rw [ite_eq_left (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega))]
  rfl

theorem honestAt_r1c (i : ℕ) (hi : i < 64) :
    honestAt Tz st c r zw R ((layOf Tz st).r1c i) = fOf (zcH Tz st c r zw).round1c[i]! := by
  unfold honestAt; dsimp only
  rw [ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_left (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), show (layOf Tz st).r1c i - 64 = i by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)]

theorem honestAt_ml (j t : ℕ) (hj : j < st.m - 6) (ht : t < 2) :
    honestAt Tz st c r zw R ((layOf Tz st).ml j t) =
      if t = 0 then fOf (zcH Tz st c r zw).rounds[j]!.1 else fOf (zcH Tz st c r zw).rounds[j]!.2 := by
  unfold honestAt; dsimp only
  rw [ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_left (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)),
    show ((layOf Tz st).ml j t - 128) / 2 = j by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega), show ((layOf Tz st).ml j t - 128) % 2 = t by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)]

theorem honestAt_fa : honestAt Tz st c r zw R (layOf Tz st).fa = fOf (zcH Tz st c r zw).finalA := by
  unfold honestAt; dsimp only
  rw [ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_left rfl]

theorem honestAt_fb : honestAt Tz st c r zw R (layOf Tz st).fb = fOf (zcH Tz st c r zw).finalB := by
  unfold honestAt; dsimp only
  rw [ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)),
    ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_left rfl]

theorem honestAt_fc : honestAt Tz st c r zw R (layOf Tz st).fc = fOf (zcH Tz st c r zw).finalC := by
  unfold honestAt; dsimp only
  rw [ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)),
    ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_left rfl]

theorem honestAt_lc (j t : ℕ) (hj : j < st.c.kLog - 6) (ht : t < 2) :
    honestAt Tz st c r zw R ((layOf Tz st).lc j t) =
      if t = 0 then (lcH Tz st c r zw j).1 else (lcH Tz st c r zw j).2 := by
  unfold honestAt; dsimp only
  rw [ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)),
    ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_left (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)),
    show ((layOf Tz st).lc j t - (layOf Tz st).fc - 1) / 2 = j by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega),
    show ((layOf Tz st).lc j t - (layOf Tz st).fc - 1) % 2 = t by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)]

theorem honestAt_zp (i : ℕ) (hi : i < 64) :
    honestAt Tz st c r zw R ((layOf Tz st).zp i) = zpH Tz st c r zw i := by
  unfold honestAt; dsimp only
  rw [ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)),
    ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)),
    ite_eq_left (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)), show (layOf Tz st).zp i - (layOf Tz st).zp 0 = i by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)]

/-- The pads on the layout's slots, `len = nE + 4`. -/
theorem len_eq (hE : (layOf Tz st).hab = nE) : (layOf Tz st).len = nE + 4 := by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)

/-- **The masked zerocheck is the honest one under the pads** (`Masks`), on its first `n ≤ m − 6` rounds. -/
theorem masks_honest (hE : (layOf Tz st).hab = nE) (n : ℕ) (hn : n ≤ st.m - 6) :
    ZerocheckMasked.Masks (layOf Tz st) (arrOf (padsAt (len_eq Tz st hE) hp)) (zcH Tz st c r zw)
      (zcOf Tz st (honestMasked Tz st c r zw R hp)) n := by
  have hpad : ∀ k (hk : k < nE), FlockLevel3.GF128.of (arrOf (padsAt (len_eq Tz st hE) hp))[k]! = hp ⟨k, by omega⟩ :=
    fun k hk => by rw [show FlockLevel3.GF128.of _ = fOf _ from rfl, arrOf_get _ k (by rw [len_eq Tz st hE]; omega)]; rfl
  have hv : ∀ k (hk : k < nE) (x : F128), honestAt Tz st c r zw R k = fOf x →
      (msgAt (honestMasked Tz st c r zw R hp) k).val = x + (arrOf (padsAt (len_eq Tz st hE) hp))[k]! := by
    intro k hk x hx
    apply Zerocheck.eq_of_of
    rw [Zerocheck.of_val, FlockLevel3.of_add, hpad k hk, msgAt_honest Tz st c r zw R hp k hk, hx]
  refine ⟨fun i hi => ?_, fun i hi => ?_, fun j hj => ?_, ?_, ?_, ?_⟩
  · rw [zcOf, Zerocheck.get_map_range _ hi]
    exact hv _ (by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)) _ (honestAt_r1ab Tz st c r zw R i hi)
  · rw [zcOf, Zerocheck.get_map_range _ hi]
    exact hv _ (by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)) _ (honestAt_r1c Tz st c r zw R i hi)
  · rw [zcOf, Zerocheck.get_map_range _ (by omega : j < st.m - 6)]
    exact Prod.ext (hv _ (by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)) _ (honestAt_ml Tz st c r zw R j 0 (by omega) (by omega)))
      (hv _ (by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)) _ (honestAt_ml Tz st c r zw R j 1 (by omega) (by omega)))
  · exact hv _ (by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)) _ (honestAt_fa Tz st c r zw R)
  · exact hv _ (by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)) _ (honestAt_fb Tz st c r zw R)
  · exact hv _ (by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)) _ (honestAt_fc Tz st c r zw R)

theorem size_rowCEval_form (lay : Zk.PadLayout) (zc : ZerocheckProof) (z : F128) :
    ((ZerocheckMasked.computedC lay zc z).add (ZerocheckMasked.fcL lay zc)).a.size = lay.len := by
  simp [Zk.Lin.add, ZerocheckMasked.size_computedC, ZerocheckMasked.fcL, Zk.Lin.masked]

theorem size_rowAB_form (lay : Zk.PadLayout) (zc : ZerocheckProof) (rr rho : Array F128) (z : F128) (n : ℕ) :
    ((ZerocheckMasked.running lay zc rr rho z n).add (ZerocheckMasked.prodL lay zc)).a.size = lay.len := by
  simp [Zk.Lin.add, ZerocheckMasked.size_running, ZerocheckMasked.prodL]

/-- **Row 0 (zerocheck c_eval) vanishes** at the honest pads, at every coin (`c_eval_zero`). -/
theorem row0_honest (hE : (layOf Tz st).hab = nE) (hm : st.m = Tz.S.m) :
    (rowCEval Tz st c r (honestMasked Tz st c r zw R hp)).eval hp = 0 := by
  rw [rowCEval, ofZk_eval (len_eq Tz st hE), linVal_eq_evalL _ _ (size_rowCEval_form _ _ _),
    ZerocheckMasked.c_eval_zero (layOf Tz st) _ (Tz.S.m - 6) (aT Tz zw) (bT Tz zw) (cT Tz zw) (rF Tz st c r)
      (rhoF Tz st c r) (zcF Tz st c r) _ (masks_honest Tz st c r zw R hp hE _ (by omega))]
  exact fOf_zero

/-- **Row 1 (zerocheck final a·b) vanishes** at the honest pads with `h_ab = h_fa·h_fb`, for valid tables, where no
outer coin is 1 (`final_zero`). -/
theorem row1_honest (hE : (layOf Tz st).hab = nE) (hm : st.m = Tz.S.m)
    (hvalid : ∀ i < 2 ^ (Tz.S.m - 6 + 6), aT Tz zw i * bT Tz zw i = cT Tz zw i)
    (hr : ∀ k < Tz.S.m - 6, (rF Tz st c r)[k]! ≠ 1)
    (hab : hp ⟨(layOf Tz st).hab, by omega⟩ =
      hp ⟨(layOf Tz st).fa, by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)⟩ * hp ⟨(layOf Tz st).fb, by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)⟩) :
    (rowAB Tz st c r (honestMasked Tz st c r zw R hp)).eval hp = 0 := by
  have hs := len_eq Tz st hE
  have hab' : (arrOf (padsAt hs hp))[(layOf Tz st).hab]! =
      (arrOf (padsAt hs hp))[(layOf Tz st).fa]! * (arrOf (padsAt hs hp))[(layOf Tz st).fb]! := by
    apply Zerocheck.eq_of_of
    rw [FlockLevel3.of_mul]
    rw [show FlockLevel3.GF128.of (arrOf (padsAt hs hp))[(layOf Tz st).hab]! = fOf _ from rfl,
      show FlockLevel3.GF128.of (arrOf (padsAt hs hp))[(layOf Tz st).fa]! = fOf _ from rfl,
      show FlockLevel3.GF128.of (arrOf (padsAt hs hp))[(layOf Tz st).fb]! = fOf _ from rfl,
      arrOf_get _ _ (by rw [hs]; omega), arrOf_get _ _ (by rw [hs, ← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)),
      arrOf_get _ _ (by rw [hs, ← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega))]
    exact hab
  rw [rowAB, ofZk_eval hs, linVal_eq_evalL _ _ (size_rowAB_form _ _ _ _ _ _), show st.m - 6 = Tz.S.m - 6 by rw [hm],
    ZerocheckMasked.final_zero (layOf Tz st) _ (Tz.S.m - 6) (by simp [layOf, Zk.PadLayout.new, hm]) (aT Tz zw)
      (bT Tz zw) (cT Tz zw) hvalid (rF Tz st c r) (rhoF Tz st c r) hr (zcF Tz st c r) _
      (masks_honest Tz st c r zw R hp hE _ (by omega)) hab']
  exact fOf_zero

/-- A masked value plus its pad is its clear value (characteristic 2), as `Lincheck.pad` reads the pads. -/
theorem masked_add_pad (hE : (layOf Tz st).hab = nE) (k : ℕ) (hk : k < nE) :
    fOf (msgAt (honestMasked Tz st c r zw R hp) k).val + Lincheck.pad (padsAt (len_eq Tz st hE) hp) k =
      honestAt Tz st c r zw R k := by
  have hk' : k < (layOf Tz st).len := by rw [len_eq Tz st hE]; omega
  rw [Lincheck.pad, dite_eq_left hk']
  show msgAt (honestMasked Tz st c r zw R hp) k + hp ⟨k, by omega⟩ = _
  rw [msgAt_honest Tz st c r zw R hp k hk]
  exact (add_assoc _ _ _).trans ((congrArg (_ + ·) (CharTwo.add_self_eq_zero _)).trans (add_zero _))

/-- **Row 2 (lincheck final) vanishes** at the honest pads, and the `ab` claim, valued at the pads' `linVal` of the
replay's `ab` value, holds for the table (`lincheck_complete` on `modelFold`), given that the clear `final_a`, `final_b`
are the true `a`, `b` claims at the zerocheck's coins. -/
theorem row2_honest (hE : (layOf Tz st).hab = nE) (hm : st.m = Tz.S.m) (hk : st.c.kLog = Tz.S.kLog)
    (hw : Tz.S.Satisfies zw)
    (hA : (⟨fOf (zcH Tz st c r zw).finalA, execArith.lagS (fOf (zcF Tz st c r)), vecOf (Tz.S.m - 6) (rhoF Tz st c r)⟩ :
      Model.Claim 𝔽 Tz.S.m).Holds (aVec Tz.S zw))
    (hB : (⟨fOf (zcH Tz st c r zw).finalB, execArith.lagS (fOf (zcF Tz st c r)), vecOf (Tz.S.m - 6) (rhoF Tz st c r)⟩ :
      Model.Claim 𝔽 Tz.S.m).Holds (bVec Tz.S zw)) :
    (rowLC Tz st c r (honestMasked Tz st c r zw R hp)).eval hp = 0 ∧
      (Lincheck.claimOf Tz.S.m (Lincheck.lcAbClaim Tz.S.m Tz.S.kLog (rhoF Tz st c r) (tsF Tz st c r) (c.rest.rPrime r))
        (Lincheck.linVal (padsAt (len_eq Tz st hE) hp) (Lincheck.lcAbValue (layOf Tz st)
          (zPartialOf Tz st (honestMasked Tz st c r zw R hp)) (c.rest.rPrime r)))).Holds zw := by
  have hs := len_eq Tz st hE
  have hrho : (rhoF Tz st c r).size = Tz.S.m - 6 := by simp [rhoF, hm]
  have hts : (tsF Tz st c r).size = Tz.S.kLog - 6 := by simp [tsF, hk]
  have hcomb : (modelFold Tz.S).folded (c.rest.lcAlpha r) (c.rest.lcBeta r) (skipWeights (zcF Tz st c r))
      ((rhoF Tz st c r).extract 0 (Tz.S.kLog - 6)) (tsF Tz st c r) = .ok (combvOf Tz st c r) := by
    rw [← hk]; rfl
  have hfa : (layOf Tz st).fa < nE := by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)
  have hfb : (layOf Tz st).fb < nE := by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)
  have hA' := masked_add_pad Tz st c r zw R hp hE _ hfa
  have hB' := masked_add_pad Tz st c r zw R hp hE _ hfb
  rw [honestAt_fa] at hA'
  rw [honestAt_fb] at hB'
  have hround : ∀ j < Tz.S.kLog - 6,
      (fOf (lcRoundsOf Tz st (honestMasked Tz st c r zw R hp))[j]!.1 +
          Lincheck.pad (padsAt hs hp) ((layOf Tz st).lc j 0),
        fOf (lcRoundsOf Tz st (honestMasked Tz st c r zw R hp))[j]!.2 +
          Lincheck.pad (padsAt hs hp) ((layOf Tz st).lc j 1)) =
        Lincheck.lcMsg (2 ^ Tz.S.kLog) (Lincheck.lcU Tz.S (zcF Tz st c r) (c.rest.lcAlpha r) (c.rest.lcBeta r)
          (rhoF Tz st c r)) (Lincheck.lcV Tz.S (rhoF Tz st c r) zw) ((tsF Tz st c r).toList.map fOf) j := by
    intro j hj
    have hj' : j < st.c.kLog - 6 := by rw [hk]; exact hj
    rw [lcRoundsOf, Zerocheck.get_map_range _ hj']
    refine Prod.ext ?_ ?_
    · exact (masked_add_pad Tz st c r zw R hp hE _ (by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega))).trans
        (honestAt_lc Tz st c r zw R j 0 hj' (by omega))
    · exact (masked_add_pad Tz st c r zw R hp hE _ (by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega))).trans
        (honestAt_lc Tz st c r zw R j 1 hj' (by omega))
  have hzp : ∀ i < 64, fOf (zPartialOf Tz st (honestMasked Tz st c r zw R hp))[i]! +
      Lincheck.pad (padsAt hs hp) ((layOf Tz st).zp i) =
        foldAll ((tsF Tz st c r).toList.map fOf) (2 ^ Tz.S.kLog) (Lincheck.lcV Tz.S (rhoF Tz st c r) zw) i := by
    intro i hi
    rw [zPartialOf, Zerocheck.get_map_range _ hi]
    exact (masked_add_pad Tz st c r zw R hp hE _ (by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega))).trans (honestAt_zp Tz st c r zw R i hi)
  have hL := Lincheck.lincheck_complete Tz.S (modelFold Tz.S) (modelFold_realizes Tz.S) zw hw (layOf Tz st)
    (padsAt hs hp) (zcF Tz st c r) (c.rest.lcAlpha r) (c.rest.lcBeta r) (c.rest.rPrime r) (rhoF Tz st c r)
    (tsF Tz st c r) hrho hts (zcOf Tz st (honestMasked Tz st c r zw R hp)).finalA
    (zcOf Tz st (honestMasked Tz st c r zw R hp)).finalB (lcRoundsOf Tz st (honestMasked Tz st c r zw R hp))
    (zPartialOf Tz st (honestMasked Tz st c r zw R hp)) (combvOf Tz st c r)
    (by rw [show fOf (zcOf Tz st (honestMasked Tz st c r zw R hp)).finalA + _ = _ from hA']; exact hA)
    (by rw [show fOf (zcOf Tz st (honestMasked Tz st c r zw R hp)).finalB + _ = _ from hB']; exact hB)
    hround hzp hcomb
  refine ⟨?_, hL.2⟩
  have h1 := hL.1
  rw [← hk] at h1
  rw [rowLC, ofZk_eval hs]
  exact h1

/-- **Row 3 (ring switch ab) vanishes** at the pads where the `ab` claim holds at its value and `ŝ_{2r}` is the
honest `shat` at the claim (characteristic 2). -/
theorem row3_honest (hE : (layOf Tz st).hab = nE) (hm : st.m = Tz.S.m) (hk : st.c.kLog = Tz.S.kLog)
    {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽)
    (hab : (Lincheck.claimOf Tz.S.m (Lincheck.lcAbClaim Tz.S.m Tz.S.kLog (rhoF Tz st c r) (tsF Tz st c r)
        (c.rest.rPrime r)) (Lincheck.linVal (padsAt (len_eq Tz st hE) hp) (Lincheck.lcAbValue (layOf Tz st)
          (zPartialOf Tz st (honestMasked Tz st c r zw R hp)) (c.rest.rPrime r)))).Holds zw)
    (hs : ∀ v : Fin 128, fOf (shArr sh (2 * r.val))[v.val]! = shat (by have := Tz.hm; omega)
      (RingSwitch.claimOf Tz.S.m (Lincheck.lcAbClaim Tz.S.m Tz.S.kLog (rhoF Tz st c r) (tsF Tz st c r)
        (c.rest.rPrime r))) zw v) :
    (rowRsAb Tz st c r (honestMasked Tz st c r zw R hp) sh).eval hp = 0 := by
  have h7 : 7 ≤ Tz.S.m := by have := Tz.hm; omega
  have hcl : abClaim Tz st c r =
      Lincheck.lcAbClaim Tz.S.m Tz.S.kLog (rhoF Tz st c r) (tsF Tz st c r) (c.rest.rPrime r) := by
    simp only [abClaim, hm, hk]
  have hcc : fOf (Zk.claimCheck (Lincheck.lcAbClaim Tz.S.m Tz.S.kLog (rhoF Tz st c r) (tsF Tz st c r)
      (c.rest.rPrime r)) (shArr sh (2 * r.val))) = (RingSwitch.claimOf Tz.S.m (Lincheck.lcAbClaim Tz.S.m Tz.S.kLog
        (rhoF Tz st c r) (tsF Tz st c r) (c.rest.rPrime r))).eval zw := by
    rw [RingSwitch.fOf_claimCheck h7, claim_eval_eq h7]
    exact Finset.sum_congr rfl fun v _ => by rw [show vecOf 128 _ v = _ from hs v]
  rw [rowRsAb, hcl, ofZk_eval (len_eq Tz st hE), Lincheck.linVal_addConst, hcc]
  refine (congrArg (· + _) hab.symm).trans ?_
  exact CharTwo.add_self_eq_zero _

/-- **Row 4 (ring switch c) vanishes** at the honest pads where `ŝ_{2r+1}` is the honest `shat` at the `c` claim: the
clear `final_c` is the claim's value (`finalC_eq_eval`). -/
theorem row4_honest (hE : (layOf Tz st).hab = nE) {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽)
    (hs : ∀ v : Fin 128, fOf (shArr sh (2 * r.val + 1))[v.val]! = shat (by have := Tz.hm; omega)
      (RingSwitch.claimOf Tz.S.m (cClaim Tz st c r)) zw v) :
    (rowRsC Tz st c r (honestMasked Tz st c r zw R hp) sh).eval hp = 0 := by
  have h7 : 7 ≤ Tz.S.m := by have := Tz.hm; omega
  have hfc : (layOf Tz st).fc < nE := by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)
  have hC := ZerocheckMasked.finalC_eq_eval Tz.S.m (six_le Tz) (aT Tz zw) (bT Tz zw) (cT Tz zw) zw
    (fun i => by rw [cT, dite_eq_left i.isLt]) (rF Tz st c r) (rhoF Tz st c r) (zcF Tz st c r)
  have hcc : fOf (Zk.claimCheck (cClaim Tz st c r) (shArr sh (2 * r.val + 1))) =
      (RingSwitch.claimOf Tz.S.m (cClaim Tz st c r)).eval zw := by
    rw [RingSwitch.fOf_claimCheck h7, claim_eval_eq h7]
    exact Finset.sum_congr rfl fun v _ => by rw [show vecOf 128 _ v = _ from hs v]
  rw [rowRsC, ofZk_eval (len_eq Tz st hE), Lincheck.linVal_addConst, Lincheck.linVal_masked,
    show fOf (zcOf Tz st (honestMasked Tz st c r zw R hp)).finalC + _ = _ from
      (masked_add_pad Tz st c r zw R hp hE _ hfc), honestAt_fc, hcc]
  exact (congrArg (· + _) hC).trans (CharTwo.add_self_eq_zero _)

/-- The tables are valid where the witness satisfies the statement (the R1CS in every block). -/
theorem hvalid_of (hw : Tz.S.Satisfies zw) : ∀ i < 2 ^ (Tz.S.m - 6 + 6), aT Tz zw i * bT Tz zw i = cT Tz zw i := by
  intro i hi
  rw [Nat.sub_add_cancel (six_le Tz)] at hi
  rw [aT, bT, cT, dite_eq_left hi, dite_eq_left hi, dite_eq_left hi, ← val_mul_cast]
  refine congrArg (fun x : ZMod 2 => ((x.val : ℕ) : 𝔽)) ?_
  exact (hw.1 ⟨i / 2 ^ Tz.S.kLog, Tz.S.block_index_lt ⟨i, hi⟩⟩ ⟨i % 2 ^ Tz.S.kLog, Nat.mod_lt _ (Nat.two_pow_pos _)⟩).trans
    (congrArg zw (Fin.ext (Nat.div_add_mod' i _)))

/-- No zerocheck coin is 1 where no outer coin is (`Admits`' last clause). -/
theorem rF_ne_one (h : ∀ x ∈ List.ofFn (c.rest.rOuter r), x ≠ 1) : ∀ k : ℕ, (rF Tz st c r)[k]! ≠ 1 :=
  Zerocheck.pinned_append_ne_one (Array.ofFn (c.rest.rOuter r)) fun k hk => by
    rw [getElem!_pos _ k hk, Array.getElem_ofFn]
    exact h _ (List.mem_ofFn.2 ⟨_, rfl⟩)

/-- **Rep `r`'s ten rows vanish** at the honest pads, given the bridges each row reads. -/
theorem rows_honest (hE : (layOf Tz st).hab = nE) (hm : st.m = Tz.S.m) (hk : st.c.kLog = Tz.S.kLog)
    (hw : Tz.S.Satisfies zw) (hr : ∀ k : ℕ, (rF Tz st c r)[k]! ≠ 1)
    (hA : (⟨fOf (zcH Tz st c r zw).finalA, execArith.lagS (fOf (zcF Tz st c r)), vecOf (Tz.S.m - 6) (rhoF Tz st c r)⟩ :
      Model.Claim 𝔽 Tz.S.m).Holds (aVec Tz.S zw))
    (hB : (⟨fOf (zcH Tz st c r zw).finalB, execArith.lagS (fOf (zcF Tz st c r)), vecOf (Tz.S.m - 6) (rhoF Tz st c r)⟩ :
      Model.Claim 𝔽 Tz.S.m).Holds (bVec Tz.S zw))
    {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽)
    (hs0 : ∀ v : Fin 128, fOf (shArr sh (2 * r.val))[v.val]! = shat (by have := Tz.hm; omega)
      (RingSwitch.claimOf Tz.S.m (Lincheck.lcAbClaim Tz.S.m Tz.S.kLog (rhoF Tz st c r) (tsF Tz st c r)
        (c.rest.rPrime r))) zw v)
    (hs1 : ∀ v : Fin 128, fOf (shArr sh (2 * r.val + 1))[v.val]! = shat (by have := Tz.hm; omega)
      (RingSwitch.claimOf Tz.S.m (cClaim Tz st c r)) zw v)
    (hk0 : Tz.sch.k0 ≤ Tz.S.m - 7) (hL : L = 2 ^ (Tz.S.m - 7 - Tz.sch.k0))
    (hT : openT Tz st c r sh = ∑ y ∈ Finset.range (2 ^ (Tz.S.m - 7)), wordOf zw y * openB Tz st c r y)
    (y' : Fin L → 𝕂)
    (hy : ∀ col : Fin L, y' col = ∑ l : Fin (2 ^ Tz.sch.k0),
      (∏ i : Fin Tz.sch.k0, if l.val.testBit i.val then FlockLevel3.GF256.of (c.rest.lane r i)
        else 1 + FlockLevel3.GF256.of (c.rest.lane r i)) *
      execArith.embed (wordOf zw (l.val + 2 ^ Tz.sch.k0 * col.val)) +
        (rhoK Tz st c r 0 * execArith.embed (R 0 col) + rhoK Tz st c r 1 * execArith.embed (R 1 col)))
    (hab : hp (ix nE (layOf Tz st).hab) = hp (ix nE (layOf Tz st).fa) * hp (ix nE (layOf Tz st).fb))
    (htw : hp (ix nE (layOf Tz st).tw) = hp (ix nE (layOf Tz st).tx) * hp (ix nE (layOf Tz st).ty))
    (k : Fin (7 + 3)) :
    (rowsOf Tz st c r (honestMasked Tz st c r zw R hp) sh (∑ col : Fin L, y' col * bFold Tz st c r col.val)
      (innerEps Tz st c r * hp (ix nE (layOf Tz st).fa) + hp (ix nE (layOf Tz st).tx),
        hp (ix nE (layOf Tz st).fb) + hp (ix nE (layOf Tz st).ty)) k).eval hp = 0 := by
  have h2 := row2_honest Tz st c r zw R hp hE hm hk hw hA hB
  refine Fin.addCases (fun k => ?_) (fun k => ?_) k
  · rw [rowsOf, Fin.append_left]
    fin_cases k
    · exact row0_honest Tz st c r zw R hp hE hm
    · have hab' := hab
      rw [ix_eq (nE := nE) _ (by omega), ix_eq (nE := nE) _ (by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)),
        ix_eq (nE := nE) _ (by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega))] at hab'
      exact row1_honest Tz st c r zw R hp hE hm (hvalid_of Tz zw hw) (fun k _ => hr k) hab'
    · exact h2.1
    · exact row3_honest Tz st c r zw R hp hE hm hk sh h2.2 hs0
    · exact row4_honest Tz st c r zw R hp hE sh hs1
    · exact laneRows_honest Tz st c r zw R hp hE hk0 hL sh hT y' hy 0
    · exact laneRows_honest Tz st c r zw R hp hE hk0 hL sh hT y' hy 1
  · rw [rowsOf, Fin.append_right]
    exact triple_rows_zero (trOf nE (layOf Tz st)) _ hp hab htw k

end Rows

/-- The pads at a layout index below `nE` are the exposed ones. -/
theorem comp_symm_lt {nE : ℕ} (hf : PadIdx nE → 𝔽) (k : ℕ) (hk : k < nE) :
    (hf ∘ (padEquiv nE).symm) (ix nE k) = hf (.inl ⟨k, hk⟩) := by
  rw [Function.comp_apply]
  refine congrArg hf ((Equiv.symm_apply_eq _).2 (Fin.ext ?_))
  simp [padEquiv, ix, Nat.mod_eq_of_lt (by omega : k < nE + 4)]

/-- The pads at the layout's last four indices are `PadIdx`'s tail, in `tailPerm`'s order. -/
theorem comp_symm_tail {nE : ℕ} (hf : PadIdx nE → 𝔽) (j : ℕ) (hj : j < 4) :
    (hf ∘ (padEquiv nE).symm) (ix nE (nE + j)) = hf (.inr (tailPerm.symm ⟨j, hj⟩)) := by
  rw [Function.comp_apply]
  refine congrArg hf ((Equiv.symm_apply_eq _).2 (Fin.ext ?_))
  simp [padEquiv, ix, Nat.mod_eq_of_lt (by omega : nE + j < nE + 4)]

/-- The layout's triple indices at `hab = nE`. -/
theorem lay_triple {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) {nE : ℕ} (hE : (layOf Tz st).hab = nE) :
    (layOf Tz st).fa < nE ∧ (layOf Tz st).fb < nE ∧ (layOf Tz st).tx = nE + 1 ∧ (layOf Tz st).ty = nE + 2 ∧
      (layOf Tz st).tw = nE + 3 := by
  subst hE
  exact ⟨by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega), by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega), rfl, rfl, rfl⟩

/-- The residual fold is `foldLoL` at the coins, lowest first. -/
theorem foldN_eq_foldLoL (ρ : ℕ → 𝔽) : ∀ (n : ℕ) (u : ℕ → 𝔽),
    Zerocheck.foldN ρ n u = foldLoL (List.ofFn fun j : Fin n => ρ j.val) u
  | 0, _ => rfl
  | n + 1, u => by
      rw [List.ofFn_succ_last, foldLoL_append_single]
      exact congrArg (foldLo (ρ n)) (foldN_eq_foldLoL ρ n u)

/-- **The residual fold at `0` is the `eq`-weighted sum** over the folded variables. -/
theorem foldN_eq_sum (ρ : ℕ → 𝔽) (n : ℕ) (u : ℕ → 𝔽) :
    Zerocheck.foldN ρ n u 0 = ∑ x ∈ Finset.range (2 ^ n), Zerocheck.eqR ρ 0 n x * u x := by
  rw [foldN_eq_foldLoL, foldLoL_eq_sum, List.length_ofFn]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [eqList_ofFn, mul_zero, add_zero]
  congr 1
  unfold Zerocheck.eqR Model.eqAt
  rw [← Fin.prod_univ_eq_prod_range (fun j => if x.testBit j then ρ (0 + j) else 1 + ρ (0 + j)) n]
  simp only [zero_add]

/-- **The honest `final_a` (any of the honest zerocheck's folded tables) is the claim's value** at the skip coin and
the residual coins: `Σ_x eq(ρ, x)·â(z, x)` is the model's evaluation of the claim `(·, L_s(z), ρ)` (`finalC_eq_eval` at
the table `a` and the point `ρ`). -/
theorem finalA_holds (m : ℕ) (hm : 6 ≤ m) (a b c : ℕ → 𝔽) (w : Model.Witness m)
    (ha : ∀ i : Fin (2 ^ m), a i.val = ((w i).val : 𝔽)) (r rho : Array F128) (z : F128) :
    (⟨fOf (Zerocheck.honest (m - 6) a b c r rho z).finalA, execArith.lagS (fOf z), vecOf (m - 6) rho⟩ :
      Model.Claim 𝔽 m).Holds w := by
  have h := ZerocheckMasked.finalC_eq_eval m hm a a a w ha rho rho z
  show Model.Claim.eval ⟨fOf 0, vecOf 64 (skipWeights z), vecOf (m - 6) rho⟩ w = _
  rw [← h]
  exact (foldN_eq_sum _ _ _).symm

theorem finalB_holds (m : ℕ) (hm : 6 ≤ m) (a b c : ℕ → 𝔽) (w : Model.Witness m)
    (hb : ∀ i : Fin (2 ^ m), b i.val = ((w i).val : 𝔽)) (r rho : Array F128) (z : F128) :
    (⟨fOf (Zerocheck.honest (m - 6) a b c r rho z).finalB, execArith.lagS (fOf z), vecOf (m - 6) rho⟩ :
      Model.Claim 𝔽 m).Holds w := by
  have h := ZerocheckMasked.finalC_eq_eval m hm b b b w hb rho rho z
  show Model.Claim.eval ⟨fOf 0, vecOf 64 (skipWeights z), vecOf (m - 6) rho⟩ w = _
  rw [← h]
  exact (foldN_eq_sum _ _ _).symm

theorem addr128_div {M : ℕ} (hm7 : 7 ≤ M) (y : Fin (2 ^ (M - 7))) (v : Fin 128) :
    (addr128 hm7 y v).val / 128 = y.val := by
  simp only [addr128]; omega

theorem getD_mem_of_lt (l : List ℕ) {x : ℕ} (hx : x < l.length) : l.getD x 0 ∈ l := by
  rw [List.getD_eq_getElem _ _ hx]; exact List.getElem_mem hx

/-- Off the mask slot, `z = (w, u)` is the witness. -/
theorem zWith_off (st : Stmt) {M nMask : ℕ} (hm7 : 7 ≤ M) (w : Witness M) (hn : nMask = (maskWordsOf st).length)
    (u : Fin nMask → Fin 128 → ZMod 2) (y : Fin (2 ^ (M - 7))) (v : Fin 128) (hy : y.val ∉ maskWordsOf st) :
    zWith st w u (addr128 hm7 y v) = w (addr128 hm7 y v) := by
  unfold zWith
  split
  · rename_i x heq
    have h := List.find?_some heq
    rw [addr128_div, decide_eq_true_eq] at h
    have hmem := getD_mem_of_lt (maskWordsOf st) (lt_of_lt_of_eq x.isLt hn)
    rw [h] at hmem
    exact absurd hmem hy
  · rfl

/-- On mask word `x`, `z = (w, u)` is `u x`, the mask slot's words distinct. -/
theorem zWith_on (st : Stmt) {M nMask : ℕ} (hm7 : 7 ≤ M) (w : Witness M) (hn : nMask = (maskWordsOf st).length)
    (u : Fin nMask → Fin 128 → ZMod 2) (hnd : (maskWordsOf st).Nodup) (x : Fin nMask) (y : Fin (2 ^ (M - 7)))
    (v : Fin 128) (hy : (maskWordsOf st).getD x.val 0 = y.val) :
    zWith st w u (addr128 hm7 y v) = u x v := by
  unfold zWith
  split
  · rename_i x' heq
    have h := List.find?_some heq
    rw [addr128_div, decide_eq_true_eq, ← hy] at h
    have hx' := lt_of_lt_of_eq x'.isLt hn
    have hx := lt_of_lt_of_eq x.isLt hn
    rw [List.getD_eq_getElem _ _ hx', List.getD_eq_getElem _ _ hx] at h
    have he : x' = x := Fin.ext ((List.Nodup.getElem_inj_iff hnd).1 h)
    subst he
    exact congrArg (u x') (Fin.ext (by simp only [addr128]; omega))
  · rename_i hnone
    have := List.find?_eq_none.1 hnone x (List.mem_finRange x)
    rw [addr128_div, ← hy] at this
    exact absurd (decide_eq_true rfl) this

theorem extract_one_toList {α : Type} (a : Array α) : (a.extract 1).toList = a.toList.drop 1 := by
  simp only [Array.toList_extract, List.extract_eq_take_drop, List.drop_one, List.take_eq_self_iff,
    List.length_tail, Array.length_toList, Std.le_refl]

theorem rep_of_lt (r : Fin 2) (k : ℕ) (hk : k / 2 = r.val) : (if k < 2 then (0 : Fin 2) else 1) = r := by
  have := r.isLt
  split <;> (apply Fin.ext; simp only [Fin.val_zero, Fin.val_one]; omega)

theorem castF_eq (b : ZMod 2) : ZK.Table.castF (F := 𝔽) b = ((b.val : ℕ) : 𝔽) := (ZMod.natCast_val b).symm

/-- **The masked claims' `ŝ` is `shat` at `z = (w, u)`**: the non-mask words' part `ν` plus the mask words' part
(`maskHom`) is the sum over every word of `z = (w, u)`, the mask slot's words distinct and inside the witness. -/
theorem sHat_shat (st : Stmt) {M nMask : ℕ} (hm7 : 7 ≤ M) (w : Witness M) (hn : nMask = (maskWordsOf st).length)
    (u : Fin nMask → Fin 128 → ZMod 2) (hnd : (maskWordsOf st).Nodup)
    (hlt : ∀ x ∈ maskWordsOf st, x < 2 ^ (M - 7)) (e : ℕ → 𝔽) (v : Fin 128) :
    (∑ x : Fin (2 ^ (M - 7)), if x.val ∈ maskWordsOf st then 0
        else e x.val * ZK.Table.castF (F := 𝔽) (wordBit w x.val v.val)) +
      ∑ x : Fin nMask, e ((maskWordsOf st).getD x.val 0) * ZK.Table.castF (F := 𝔽) (u x v) =
    ∑ y : Fin (2 ^ (M - 7)), e y.val * ((zWith st w u (addr128 hm7 y v)).val : 𝔽) := by
  have hsplit : ∀ y : Fin (2 ^ (M - 7)), e y.val * ((zWith st w u (addr128 hm7 y v)).val : 𝔽) =
      (if y.val ∈ maskWordsOf st then 0 else e y.val * ZK.Table.castF (F := 𝔽) (wordBit w y.val v.val)) +
      (if y.val ∈ maskWordsOf st then e y.val * ((zWith st w u (addr128 hm7 y v)).val : 𝔽) else 0) := by
    intro y
    by_cases hy : y.val ∈ maskWordsOf st
    · rw [ite_eq_left hy, ite_eq_left hy, zero_add]
    · rw [ite_eq_right hy, ite_eq_right hy, add_zero, zWith_off st hm7 w hn u y v hy, castF_eq, wordBit,
        dite_eq_left (show 128 * y.val + v.val < 2 ^ M from (addr128 hm7 y v).isLt)]
      rfl
  rw [Finset.sum_congr rfl fun y _ => hsplit y, Finset.sum_add_distrib, ← Finset.sum_filter]
  congr 1
  refine Finset.sum_bij (fun x _ => ⟨(maskWordsOf st).getD x.val 0, hlt _ (getD_mem_of_lt _ (lt_of_lt_of_eq x.isLt hn))⟩)
    (fun x _ => Finset.mem_filter.2 ⟨Finset.mem_univ _, getD_mem_of_lt _ (lt_of_lt_of_eq x.isLt hn)⟩) ?_ ?_ ?_
  · intro a _ b _ hab
    have h := congrArg Fin.val hab
    simp only at h
    rw [List.getD_eq_getElem _ _ (lt_of_lt_of_eq a.isLt hn), List.getD_eq_getElem _ _ (lt_of_lt_of_eq b.isLt hn)] at h
    exact Fin.ext ((List.Nodup.getElem_inj_iff hnd).1 h)
  · intro y hy
    obtain ⟨i, hi, hiy⟩ := List.mem_iff_getElem.1 (Finset.mem_filter.1 hy).2
    exact ⟨⟨i, hn ▸ hi⟩, Finset.mem_univ _, Fin.ext (by simp only; rw [List.getD_eq_getElem _ _ hi, hiy])⟩
  · intro x _
    rw [zWith_on st hm7 w hn u hnd x _ v rfl, castF_eq]

theorem wordOf_bits {M : ℕ} (hm7 : 7 ≤ M) (zw : Witness M) (y : Fin (2 ^ (M - 7))) (v : Fin 128) :
    zw (addr128 hm7 y v) = execArith.unpack (wordOf zw y.val) v := by
  rw [wordOf, execArith_correct.unpack_pack]
  show _ = dite _ _ _
  rw [dite_eq_left (show 128 * y.val + v.val < 2 ^ M from (addr128 hm7 y v).isLt)]
  rfl

theorem recombine_sum {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
    (A : Arith F K) (hA : A.Correct) {M : ℕ} (hm7 : 7 ≤ M) (z : Witness M) (f : Fin (2 ^ (M - 7)) → F)
    (hz : ∀ y v, z (addr128 hm7 y v) = A.unpack (f y) v) (e : ℕ → F) (s : Fin 128 → F)
    (hs : ∀ v, s v = ∑ y : Fin (2 ^ (M - 7)), e y.val * ((z (addr128 hm7 y v)).val : F)) (r : Fin 7 → F) :
    recombine A s r = ∑ y : Fin (2 ^ (M - 7)), f y * Model.phi A r (e y.val) := by
  classical
  have hbits : ∀ u : Fin 128, (fun v => A.unpack (s v) u) =
      ∑ y : Fin (2 ^ (M - 7)), A.unpack (e y.val) u • A.unpack (f y) := by
    intro u
    funext v
    rw [hs v, ← unpackHom_apply A hA, map_sum]
    simp only [unpackHom_apply, unpack_mul_bit A hA, hz, Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
    exact Finset.sum_congr rfl fun y _ => mul_comm _ _
  have hpack : ∀ u : Fin 128, A.pack (fun v => A.unpack (s v) u) =
      ∑ y : Fin (2 ^ (M - 7)), ((A.unpack (e y.val) u).val : F) * f y := by
    intro u
    rw [hbits u, ← packHom_apply A hA, map_sum]
    simp only [packHom_apply, pack_smul_bit A hA, hA.pack_unpack]
  unfold recombine Model.phi
  simp only [hpack, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun y _ => Finset.sum_congr rfl fun u _ => by ring

theorem sum_swap_mul {R : Type} [CommRing R] {n N : ℕ} (γ : Fin n → R) (f : ℕ → R) (φ : Fin n → ℕ → R) :
    ∑ k, γ k * ∑ y : Fin N, f y.val * φ k y.val = ∑ y ∈ Finset.range N, f y * ∑ k, γ k * φ k y := by
  rw [← Fin.sum_univ_eq_sum_range (fun y => f y * ∑ k, γ k * φ k y)]
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun y _ => Finset.sum_congr rfl fun k _ => by ring

section Tails

variable {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2)

theorem abTail (hm : st.m = Tz.S.m) (hk : st.c.kLog = Tz.S.kLog) (i : ℕ) (hi : i < Tz.S.m - 6) :
    fOf (Lincheck.lcAbClaim Tz.S.m Tz.S.kLog (rhoF Tz st c r) (tsF Tz st c r) (c.rest.rPrime r)).x[i]! =
      abX Tz st c r i := by
  rw [← hm] at hi
  rw [← hm, ← hk]
  have hkm := Tz.S.hk
  rw [← hm, ← hk] at hkm
  unfold Lincheck.lcAbClaim abX lcT zcRho seqOf rhoF tsF
  by_cases hlt : i < st.c.kLog - 6
  · rw [ite_eq_left hlt, getElem!_pos _ i (by simp; omega), Array.getElem_append_left (by simp; omega),
      Array.getElem_reverse, Array.getElem_ofFn, dif_pos (by omega)]
    simp only [Array.size_ofFn]
  · rw [ite_eq_right hlt, getElem!_pos _ i (by simp; omega), Array.getElem_append_right (by simp; omega),
      Array.getElem_extract, Array.getElem_ofFn, dif_pos hi]
    simp only [Array.size_reverse, Array.size_ofFn]
    congr 2
    apply Fin.ext
    simp only
    omega

theorem cTail (hm : st.m = Tz.S.m) (i : ℕ) (hi : i < Tz.S.m - 6) :
    fOf (rF Tz st c r)[i]! = zcReq Tz st c r i := by
  rw [← hm] at hi
  have h13 := Tz.hm
  rw [← hm] at h13
  unfold rF zcReq seqOf
  by_cases hlt : i < 7
  · rw [dite_eq_left hlt, getElem!_pos _ i (by simp [FlockLevel3.pinned_size]; omega),
      Array.getElem_append_left (by simp [FlockLevel3.pinned_size]; omega)]
    show _ = FlockLevel3.pinnedF ⟨i, hlt⟩
    rw [FlockLevel3.pinnedF, getElem!_pos _ i (by simp [FlockLevel3.pinned_size]; omega)]
  · rw [dite_eq_right hlt, getElem!_pos _ i (by simp [FlockLevel3.pinned_size]; omega),
      Array.getElem_append_right (by simp [FlockLevel3.pinned_size]; omega), Array.getElem_ofFn, dif_pos (by omega)]
    simp only [FlockLevel3.pinned_size]

theorem claimX_region (k : ℕ) (hk2 : 2 ≤ k) (h : k - 2 < 2 * st.regions.size) (hm : st.m = Tz.S.m) (y : ℕ) :
    Model.eqAt (claimX Tz st c r k) y = fOf (eqWord (regionPtOf st c.rest.link ⟨k - 2, h⟩) y) := by
  have hlen : (regionPtOf st c.rest.link ⟨k - 2, h⟩).length = Tz.S.m - 7 := by
    simp only [regionPtOf, List.length_drop, Array.length_toList, Refine.regionPoint_size, hm]
  rw [RingSwitch.fOf_eqWord, hlen]
  refine RingSwitch.eqAt_range _ (fun i => fOf (regionPtOf st c.rest.link ⟨k - 2, h⟩)[i]!) (fun j => ?_) y
  rw [claimX, if_neg (by omega), if_neg (by omega), dif_pos h]
  simp only [List.getD_eq_getElem?_getD, getElem!_def]
  rfl

end Tails

end Glue

section Mask

open Model Refine

/-- In-block column `i` is in the mask slot: its word is one of the slot's words in block 0. -/
def MaskCol (st : Stmt) (i : ℕ) : Prop :=
  match st.c.ranges.toList.find? (fun r => r.kind == Kind.mask) with
  | none => False
  | some r => r.start / 128 ≤ i / 128 ∧ i / 128 < r.start / 128 + 2 ^ (r.slotLog - 7)

/-- **The mask slot is isolated** in a statement: its rows of `A₀`, `B₀` are identity rows, no other row reads a mask-slot
column, and neither the pin nor a region bit is in it. -/
structure MaskIsolated (st : Stmt) (S : Statement) : Prop where
  row : ∀ i c : Fin (2 ^ S.kLog), MaskCol st i.val →
    S.A₀ i c = (if i = c then 1 else 0) ∧ S.B₀ i c = (if i = c then 1 else 0)
  col : ∀ i c : Fin (2 ^ S.kLog), ¬ MaskCol st i.val → MaskCol st c.val → S.A₀ i c = 0 ∧ S.B₀ i c = 0
  pin : ¬ MaskCol st S.pin.val
  regions : ∀ R ∈ S.regions, ∀ i < 2 ^ R.free.length, ¬ MaskCol st (R.inBlock i)

theorem div_unique {K o b a c : ℕ} (ha : a < K) (hc : c < K) (h : o * K + a = b * K + c) : o = b ∧ a = c := by
  have hK : 0 < K := by omega
  have h1 := congrArg (· / K) h
  have h2 := congrArg (· % K) h
  simp only [Nat.mul_comm _ K, Nat.mul_add_div hK, Nat.div_eq_of_lt ha, Nat.div_eq_of_lt hc, add_zero,
    Nat.mul_add_mod, Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hc] at h1 h2
  exact ⟨h1, h2⟩

/-- Block `o`'s word of in-block column `i` is a mask-slot word exactly when `i` is a mask-slot column. -/
theorem mem_maskWordsOf (st : Stmt)
    (hmask : ∀ r ∈ st.c.ranges.toList, r.kind = Kind.mask → 7 ≤ r.slotLog ∧ r.start + 2 ^ r.slotLog ≤ 2 ^ st.c.kLog)
    (o : ℕ) (ho : o < 2 ^ st.nbl) (i : ℕ) (hi : i < 2 ^ st.c.kLog) :
    (o * 2 ^ st.c.kLog + i) / 128 ∈ maskWordsOf st ↔ MaskCol st i := by
  unfold maskWordsOf MaskCol
  cases hr : st.c.ranges.toList.find? (fun r => r.kind == Kind.mask) with
  | none => simp
  | some r =>
    dsimp only
    have hmem := List.mem_of_find?_eq_some hr
    have hk := List.find?_some hr
    have hk : r.kind = Kind.mask := by
      cases hkd : r.kind <;> rw [hkd] at hk
      · exact absurd hk (by decide)
      · exact absurd hk (by intro h; cases h)
    obtain ⟨h7, hb⟩ := hmask r hmem hk
    have hsk : r.slotLog ≤ st.c.kLog := by
      by_contra hc
      have : 2 ^ st.c.kLog < 2 ^ r.slotLog := Nat.pow_lt_pow_right (by norm_num) (by omega)
      omega
    have hJ : 2 ^ r.slotLog = 128 * 2 ^ (r.slotLog - 7) := by
      rw [show (128 : ℕ) = 2 ^ 7 by norm_num, ← pow_add, Nat.add_sub_cancel' h7]
    have hK : 2 ^ st.c.kLog = 128 * 2 ^ (st.c.kLog - 7) := by
      rw [show (128 : ℕ) = 2 ^ 7 by norm_num, ← pow_add, Nat.add_sub_cancel' (by omega)]
    have hoJ : r.start / 128 + 2 ^ (r.slotLog - 7) ≤ 2 ^ (st.c.kLog - 7) := by omega
    set K := 2 ^ (st.c.kLog - 7)
    set J := 2 ^ (r.slotLog - 7)
    set s := r.start / 128
    have hdiv : (o * 2 ^ st.c.kLog + i) / 128 = o * K + i / 128 := by
      rw [hK, show o * (128 * K) + i = 128 * (o * K) + i by ring, Nat.mul_add_div (by norm_num)]
    have hiK : i / 128 < K := by omega
    rw [hdiv]
    simp only [List.mem_flatMap, List.mem_map, List.mem_range]
    constructor
    · rintro ⟨b, hb2, j, hj, he⟩
      have := div_unique (o := o) (b := b) hiK (by omega : s + j < K) (by rw [← he]; ring)
      omega
    · rintro ⟨h1, h2⟩
      exact ⟨o, ho, i / 128 - s, by omega, by omega⟩

theorem zmod2_mul_self (x : ZMod 2) : x * x = x := by
  fin_cases x <;> rfl

/-- **`z = (w, u)` satisfies the statement where `w` does**, the mask slot isolated: its rows hold for any bits
(`z_i·z_i = z_i`), and every other row, the pin and the regions read only columns where `z` is `w`. -/
theorem satisfies_zWith (st : Stmt) (hwf : StmtWF st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (hm : st.m = st.c.kLog + st.nbl)
    (hmask : ∀ r ∈ st.c.ranges.toList, r.kind = Kind.mask → 7 ≤ r.slotLog ∧ r.start + 2 ^ r.slotLog ≤ 2 ^ st.c.kLog)
    (hI : MaskIsolated st (stmtOf st hwf regs)) (w : Witness st.m) (hw : (stmtOf st hwf regs).Satisfies w)
    {nMask : ℕ} (u : Fin nMask → Fin 128 → ZMod 2) (hn : nMask = (maskWordsOf st).length) :
    (stmtOf st hwf regs).Satisfies (zWith st w u) := by
  set S := stmtOf st hwf regs with hS
  have hnbl : S.nbl = st.nbl := by show st.m - st.c.kLog = st.nbl; omega
  have hblk : ∀ o i, ¬ MaskCol st i.val → S.block (zWith st w u) o i = S.block w o i := by
    intro o i hi
    show zWith st w u (S.addr o i) = w (S.addr o i)
    unfold zWith
    split
    · rename_i x heq
      have h := List.find?_some heq
      rw [decide_eq_true_eq] at h
      have hmem := getD_mem_of_lt (maskWordsOf st) (lt_of_lt_of_eq x.isLt hn)
      rw [h] at hmem
      exact absurd ((mem_maskWordsOf st hmask o.val (hnbl ▸ o.isLt) i.val i.isLt).1 hmem) hi
    · rfl
  obtain ⟨hr1, hpin, hreg⟩ := hw
  refine ⟨fun o i => ?_, fun o => ?_, fun R hR o i hi h => ?_⟩
  · by_cases hi : MaskCol st i.val
    · have hA : (S.A₀.mulVec (S.block (zWith st w u) o)) i = S.block (zWith st w u) o i := by
        simp only [Matrix.mulVec, dotProduct, fun c => (hI.row i c hi).1, ite_mul, one_mul, zero_mul,
          Finset.sum_ite_eq, Finset.mem_univ, ite_true]
      have hB : (S.B₀.mulVec (S.block (zWith st w u) o)) i = S.block (zWith st w u) o i := by
        simp only [Matrix.mulVec, dotProduct, fun c => (hI.row i c hi).2, ite_mul, one_mul, zero_mul,
          Finset.sum_ite_eq, Finset.mem_univ, ite_true]
      rw [hA, hB, zmod2_mul_self]
    · have hv : ∀ M : Matrix (Fin (2 ^ S.kLog)) (Fin (2 ^ S.kLog)) (ZMod 2),
          (∀ c, MaskCol st c.val → M i c = 0) →
          (M.mulVec (S.block (zWith st w u) o)) i = (M.mulVec (S.block w o)) i := fun M hM => by
        simp only [Matrix.mulVec, dotProduct]
        refine Finset.sum_congr rfl fun c _ => ?_
        by_cases hc : MaskCol st c.val
        · rw [hM c hc, zero_mul, zero_mul]
        · rw [hblk o c hc]
      rw [hv _ fun c hc => (hI.col i c hi hc).1, hv _ fun c hc => (hI.col i c hi hc).2, hblk o i hi]
      exact hr1 o i
  · rw [hblk o _ hI.pin]; exact hpin o
  · rw [hblk o _ (hI.regions R hR i hi)]; exact hreg R hR o i hi h

theorem lastCover_mem : ∀ (rs : List Range) (c : ℕ) {r : Range}, Refine.lastCover rs c = some r →
    r ∈ rs ∧ FlockLevel3.covers r c
  | [], _, _, h => by simp [Refine.lastCover] at h
  | r' :: rs, c, r, h => by
      rw [Refine.lastCover] at h
      cases h' : Refine.lastCover rs c with
      | some r'' =>
          rw [h', Option.some_or] at h
          cases h
          exact ⟨List.mem_cons_of_mem _ (lastCover_mem rs c h').1, (lastCover_mem rs c h').2⟩
      | none =>
          rw [h', Option.none_or] at h
          split at h
          · cases h; exact ⟨List.mem_cons_self, by assumption⟩
          · cases h

theorem lastCover_isSome : ∀ (rs : List Range) (c : ℕ) {r : Range}, r ∈ rs → FlockLevel3.covers r c →
    (Refine.lastCover rs c).isSome
  | [], _, _, h, _ => absurd h List.not_mem_nil
  | r' :: rs, c, r, h, hc => by
      rw [Refine.lastCover]
      rcases List.mem_cons.1 h with rfl | h
      · cases Refine.lastCover rs c <;> simp [hc]
      · have := lastCover_isSome rs c h hc
        cases h' : Refine.lastCover rs c with
        | some _ => simp
        | none => rw [h'] at this; cases this

/-- **The layout facts mask isolation reads**: the mask range is one slot of at least a word, no other range covers a
column it covers, Δ's pairs avoid its columns, and so do the pin and the regions' cells. -/
structure MaskLayout (st : Stmt) : Prop where
  slot : ∀ m, st.c.ranges.toList.find? (fun r => r.kind == Kind.mask) = some m → m.count = 1 ∧ 7 ≤ m.slotLog
  apart : ∀ m, st.c.ranges.toList.find? (fun r => r.kind == Kind.mask) = some m →
    ∀ r ∈ st.c.ranges.toList, ∀ x, FlockLevel3.covers r x → FlockLevel3.covers m x → r = m
  delta : ∀ p ∈ st.da.toList ++ st.db.toList, ¬ MaskCol st p.1 ∧ ¬ MaskCol st p.2
  pin : ¬ MaskCol st st.pin
  regions : ∀ R ∈ regionsOf st, ∀ i < 2 ^ R.free.length, ¬ MaskCol st (R.inBlock i)

theorem maskCol_iff {st : Stmt} {m : Range} (hf : st.c.ranges.toList.find? (fun r => r.kind == Kind.mask) = some m)
    (h1 : m.count = 1) (h7 : 7 ≤ m.slotLog) (x : ℕ) : MaskCol st x ↔ FlockLevel3.covers m x := by
  unfold MaskCol FlockLevel3.covers
  rw [hf]
  dsimp only
  have hJ : 2 ^ m.slotLog = 128 * 2 ^ (m.slotLog - 7) := by
    rw [show (128 : ℕ) = 2 ^ 7 by norm_num, ← pow_add, Nat.add_sub_cancel' h7]
  have hK : 0 < 2 ^ (m.slotLog - 7) := Nat.two_pow_pos _
  have hs : m.start / 128 = m.pos0 * 2 ^ (m.slotLog - 7) := by
    rw [Range.start, hJ, ← Nat.mul_assoc, Nat.mul_comm m.pos0 128, Nat.mul_assoc,
      Nat.mul_div_cancel_left _ (by norm_num)]
  rw [hs, h1, hJ, ← Nat.div_div_eq_div_mul, Nat.le_div_iff_mul_le hK, Nat.div_lt_iff_lt_mul hK, Nat.add_mul, one_mul]

theorem dCnt_zero {st : Stmt} {d : Array (ℕ × ℕ)} (hd : ∀ p ∈ d.toList, ¬ MaskCol st p.1 ∧ ¬ MaskCol st p.2) (i c : ℕ)
    (h : MaskCol st i ∨ MaskCol st c) : dCnt d i c = 0 :=
  List.count_eq_zero.2 fun hm => by rcases h with h | h; exacts [(hd _ hm).1 h, (hd _ hm).2 h]

theorem rangeCnt_mask {st : Stmt} (hL : MaskLayout st) (side : Flock.Kind → ℕ → ℕ → ℕ)
    (hside : ∀ i c, side Flock.Kind.mask i c = if i = c then 1 else 0) (i c : ℕ) (h : MaskCol st i ∨ MaskCol st c) :
    rangeCnt side st.c.ranges.toList i c = if i = c then 1 else 0 := by
  have hf : ∃ m, st.c.ranges.toList.find? (fun r => r.kind == Kind.mask) = some m := by
    rcases h with h | h <;> (unfold MaskCol at h; split at h; · exact h.elim) <;> exact ⟨_, by assumption⟩
  obtain ⟨m, hf⟩ := hf
  obtain ⟨h1, h7⟩ := hL.slot m hf
  have hmk : m.kind = Flock.Kind.mask := by
    have hk := List.find?_some hf
    cases hkd : m.kind <;> rw [hkd] at hk
    · exact absurd hk (by decide)
    · exact absurd hk (by intro h; cases h)
  have hmem : m ∈ st.c.ranges.toList := List.mem_of_find?_eq_some hf
  have hiff := maskCol_iff hf h1 h7
  have hsame : ∀ (r : Range) (a b : ℕ), FlockLevel3.covers r b → a / 2 ^ r.slotLog = b / 2 ^ r.slotLog →
      FlockLevel3.covers r a := fun r a b hb he => by unfold FlockLevel3.covers at *; rw [he]; exact hb
  unfold rangeCnt
  cases hl : Refine.lastCover st.c.ranges.toList c with
  | none =>
      have hic : i ≠ c := fun he => by
        subst he
        have hc : FlockLevel3.covers m i := (hiff i).1 (h.elim id id)
        have := lastCover_isSome _ i hmem hc
        rw [hl] at this; cases this
      simp [hic]
  | some r =>
      obtain ⟨hr, hrc⟩ := lastCover_mem _ c hl
      dsimp only
      by_cases hq : i / 2 ^ r.slotLog = c / 2 ^ r.slotLog
      · rw [if_pos hq]
        have hri := hsame r i c hrc hq
        have hrm : r = m := by
          rcases h with h | h
          · exact hL.apart m hf r hr i hri ((hiff i).1 h)
          · exact hL.apart m hf r hr c hrc ((hiff c).1 h)
        subst hrm
        rw [hmk, hside]
        by_cases he : i % 2 ^ r.slotLog = c % 2 ^ r.slotLog
        · have hic : i = c := by
            rw [← Nat.div_add_mod i (2 ^ r.slotLog), ← Nat.div_add_mod c (2 ^ r.slotLog), hq, he]
          rw [if_pos he, if_pos hic]
        · rw [if_neg he, if_neg (by intro h'; subst h'; exact he rfl)]
      · rw [if_neg hq, if_neg (by intro h'; subst h'; exact hq rfl)]

/-- **Mask isolation from the layout** (`stmtOf`'s `matA`, `matB`): the mask range's identity slot is the only range that
reads its columns or rows, and Δ adds nothing there. -/
theorem maskIsolated_of_layout (st : Stmt) (hwf : StmtWF st) (hL : MaskLayout st) :
    MaskIsolated st (stmtOf st hwf (regionsOf st)) := by
  have hda : ∀ p ∈ st.da.toList, ¬ MaskCol st p.1 ∧ ¬ MaskCol st p.2 := fun p hp => hL.delta p (List.mem_append_left _ hp)
  have hdb : ∀ p ∈ st.db.toList, ¬ MaskCol st p.1 ∧ ¬ MaskCol st p.2 := fun p hp => hL.delta p (List.mem_append_right _ hp)
  have hA : ∀ i c : ℕ, MaskCol st i ∨ MaskCol st c → matA st i c = if i = c then 1 else 0 := fun i c h => by
    rw [matA, dCnt_zero hda i c h, Nat.add_zero, rangeCnt_mask hL (slotA st) (fun _ _ => rfl) i c h]
    split_ifs <;> simp
  have hB : ∀ i c : ℕ, MaskCol st i ∨ MaskCol st c → matB st i c = if i = c then 1 else 0 := fun i c h => by
    rw [matB, dCnt_zero hdb i c h, Nat.add_zero, rangeCnt_mask hL (slotB st) (fun _ _ => rfl) i c h]
    split_ifs <;> simp
  refine ⟨fun i c hi => ?_, fun i c hi hc => ?_, hL.pin, hL.regions⟩
  · show matA st i.val c.val = _ ∧ matB st i.val c.val = _
    rw [hA _ _ (Or.inl hi), hB _ _ (Or.inl hi)]
    exact ⟨by simp [Fin.ext_iff], by simp [Fin.ext_iff]⟩
  · show matA st i.val c.val = 0 ∧ matB st i.val c.val = 0
    have hne : i.val ≠ c.val := fun he => hi (he ▸ hc)
    rw [hA _ _ (Or.inr hc), hB _ _ (Or.inr hc), if_neg hne]; exact ⟨rfl, rfl⟩

theorem maskCol_iff_B (st : Stmt) (i : ℕ) : MaskCol st i ↔ maskColB st i = true := by
  unfold MaskCol maskColB
  cases st.c.ranges.toList.find? (fun r => r.kind == Kind.mask) <;> simp

theorem delta_of_avoids {st : Stmt} (h : deltaAvoidsMask st = true) :
    ∀ p ∈ st.da.toList ++ st.db.toList, ¬ MaskCol st p.1 ∧ ¬ MaskCol st p.2 := by
  intro p hp
  have := List.all_eq_true.1 h p hp
  simp only [Bool.and_eq_true, Bool.not_eq_true'] at this
  simp [maskCol_iff_B, this.1, this.2]

theorem kind_of_find {st : Stmt} {m : Range} (hf : st.c.ranges.toList.find? (fun r => r.kind == Kind.mask) = some m) :
    m.kind = Flock.Kind.mask := by
  have hk := List.find?_some hf
  cases hkd : m.kind <;> rw [hkd] at hk
  · exact absurd hk (by decide)
  · exact absurd hk (by intro h; cases h)

/-- `HmRow.check`'s mask range is one slot. -/
theorem hm_check_maskCount {c : Flock.Circuit} {tags : Tags} {ls : Lean.Json} (h : HmRow.check c tags ls = .ok ()) :
    ∀ r ∈ c.ranges.toList, r.kind = Flock.Kind.mask → r.count = 1 := by
  unfold HmRow.check at h
  dsimp only at h
  obtain ⟨_, hloop, -⟩ := ebind_ok h
  clear h
  rw [← Array.forIn_toList] at hloop
  refine Refine.list_forIn_each _ (fun r => r.kind = Flock.Kind.mask → r.count = 1) ?_ _ _ _ hloop
  clear hloop
  intro r s y hy
  repeat' (first | (obtain ⟨_, hx, hy⟩ := ebind_ok hy; try (cases hx)) | (split at hy))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    refine ⟨fun hk => ?_, _, rfl⟩
    simp_all

/-- **The regions' cells avoid the mask slot** (`regionCellsOff`). -/
theorem regions_layout_of {st : Stmt} (hdm : avoidsMask st = true) :
    ∀ R ∈ regionsOf st, ∀ i < 2 ^ R.free.length, ¬ MaskCol st (R.inBlock i) := by
  intro R hR i hi hmc
  have := List.all_eq_true.1 (List.all_eq_true.1 (avoidsMask_cells hdm) R hR) i (List.mem_range.2 hi)
  rw [maskCol_iff_B] at hmc
  simp [hmc] at this

/-- **What the mask layout reads of a setup**: `Layout` of it at some Δ (it reads the ranges and the pin), `HmRow.check`
on its circuit, and `m`. Both `Stmt.setupH` and `Stmt.setupHidden` (untyped; its flat twin's Δ) give it. -/
structure MaskSetup (st : Stmt) : Prop where
  layout : ∃ da db, Layout { st with da := da, db := db }
  check : ∃ tags ls, HmRow.check st.c tags ls = .ok ()
  m : st.m = st.c.kLog + st.nbl

theorem MaskSetup.mask {st : Stmt} (M : MaskSetup st) :
    ∀ r ∈ st.c.ranges.toList, r.kind = Kind.mask → 7 ≤ r.slotLog ∧ r.start + 2 ^ r.slotLog ≤ 2 ^ st.c.kLog :=
  let ⟨_, _, hc⟩ := M.check; Refine.hm_check_mask hc

theorem maskSetup_of_setupH {tags : Tags} {circuitFile publicFile : ByteArray} {coins : Coins}
    {tables : List (String × ByteArray)} {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (ht : tags.typed = false)
    (h : Stmt.setupH tags circuitFile publicFile coins tables draw partition program = .ok st) : MaskSetup st :=
  ⟨⟨st.da, st.db, setupH_layout ht h⟩, let ⟨ls, hc⟩ := Refine.hm_parse_check (setupH_spec ht h).1; ⟨_, ls, hc⟩,
    (Refine.setupH_mask h).2⟩

/-- **A setup's layout facts**: `HmRow.check`'s mask range is one slot of at least a word, apart from every other range
(`Layout`); the pin is a net slot's constant; Δ and the regions avoid the mask (`avoidsMask`). -/
theorem maskLayout_of {st : Stmt} (M : MaskSetup st) (hdm : avoidsMask st = true) : MaskLayout st := by
  obtain ⟨da, db, L⟩ := M.layout
  obtain ⟨_, _, hc⟩ := M.check
  have hmask := M.mask
  have hslot : ∀ m, st.c.ranges.toList.find? (fun r => r.kind == Kind.mask) = some m →
      m.count = 1 ∧ 7 ≤ m.slotLog := fun m hf =>
    ⟨hm_check_maskCount hc m (List.mem_of_find?_eq_some hf) (kind_of_find hf),
      (hmask m (List.mem_of_find?_eq_some hf) (kind_of_find hf)).1⟩
  have hapart : ∀ m, st.c.ranges.toList.find? (fun r => r.kind == Kind.mask) = some m →
      ∀ r ∈ st.c.ranges.toList, ∀ x, FlockLevel3.covers r x → FlockLevel3.covers m x → r = m :=
    fun m hf r hr _ h1 h2 => eq_of_inSpan L.disj hr (List.mem_of_find?_eq_some hf) (inSpan_of_covers h1)
      (inSpan_of_covers h2)
  refine ⟨hslot, hapart, delta_of_avoids (avoidsMask_delta hdm), ?_, regions_layout_of hdm⟩
  intro hpin
  cases hf : st.c.ranges.toList.find? (fun r => r.kind == Kind.mask) with
  | none => unfold MaskCol at hpin; rw [hf] at hpin; exact hpin
  | some m =>
      obtain ⟨h1, h7⟩ := hslot m hf
      have hcm := (maskCol_iff hf h1 h7 st.pin).1 hpin
      obtain ⟨rp, hrp, ip, hkp, hcp, hpv, -⟩ := L.pin
      have hcr : FlockLevel3.covers rp st.pin := by
        rw [hpv, start_eq_slot]
        exact covers_slot hcp (by rw [L.slotLog rp hrp ip hkp]; exact (L.fit ip).1)
      have hk := kind_of_find hf
      rw [← hapart m hf rp hrp _ hcr hcm, hkp] at hk
      cases hk

/-- **The setup's layout facts** (`maskLayout_of` at `Stmt.setupH`). -/
theorem setupH_maskLayout {tags : Tags} {circuitFile publicFile : ByteArray} {coins : Coins}
    {tables : List (String × ByteArray)} {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (ht : tags.typed = false)
    (h : Stmt.setupH tags circuitFile publicFile coins tables draw partition program = .ok st)
    (hdm : avoidsMask st = true) : MaskLayout st :=
  maskLayout_of (maskSetup_of_setupH ht h) hdm

/-- **The setup's statements isolate the mask slot**: `HmRow.check`'s ranges are aligned and pairwise
apart, so the mask range's rows of `A₀`, `B₀` are its own identity rows and no other range's row reads its columns; Δ's
pairs lie in net slots; the pin is a net slot's constant row; the regions lie in the hm96 and output ranges. -/
theorem setupH_mask_isolated {tags : Tags} {circuitFile publicFile : ByteArray} {coins : Coins}
    {tables : List (String × ByteArray)} {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (ht : tags.typed = false)
    (h : Stmt.setupH tags circuitFile publicFile coins tables draw partition program = .ok st)
    (hdm : avoidsMask st = true) :
    MaskIsolated st (stmtOf st (setupH_wf h).1 (regionsOf st)) :=
  maskIsolated_of_layout st _ (setupH_maskLayout ht h hdm)

end Mask

variable {mPts n : ℕ} {Reg : Type} (P : LigProver)

/-- The `--zk` tables at a draw `verify --zk` sets up are its table. -/
theorem tablesAtZK_eq {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) (S : Finset (Fin n)) (R : Reg) (y : DrawSetupZK I mPts S (dj S)) :
    tablesAtZK (Reg := Reg) dj y₀ S R = y.table := by
  show tableAtZK dj y₀ S = _
  rw [tableAtZK, dif_pos ⟨y⟩]
  exact tableZK_eq _ _

/-- **The inner relation at one rep of a `--zk` table of a setup** (`innerOf_holds_of` at rep `r`): at every coin vector
whose rep-`r` outer coins aren't `1` (`Admits`' last clause at `r`, the only part of `Admits` rep `r`'s constraint
reads), every valid witness whose registered values are the public ones satisfies rep `r`'s batched constraint at every
draw. -/
theorem innerOf_holds_rep_of (Tz : TabZK FlockLevel3.GF128 mPts) (st : Stmt) (hwf : StmtWF st)
    (hS : Tz.S = stmtOf st hwf (regionsOf st)) (M : MaskSetup st) (hdm : avoidsMask st = true) :
    let C := coinedOf P Tz st
    ∀ c (r : Fin 2), (∀ x ∈ C.rOuter c r, x ≠ 1) → ∀ w, C.Valid w → (C.T c).tab.sReg w = (C.T c).tab.pubReg →
      ∀ ω, ∑ i, (C.T c).tab.lam r ((C.T c).tab.view w ω).pre i * (C.T c).tab.hFull (ω.2.2.1 r) i =
        (C.T c).tab.theta r ((C.T c).tab.view w ω).pre := by
  intro C c r hr w hw hreg ω
  apply batch_of_rows
  intro k
  have hE : (layOf (Tz) st).hab = (C.T c).tab.nE := by
    show _ = padLenOf _ _ - 4
    simp only [padLenOf, layOf, Zk.PadLayout.len, Zk.PadLayout.tw]
    omega
  obtain ⟨hfa, hfb, htx, hty, htw⟩ := lay_triple _ _ hE
  have hv : (ZK.Table.View.pre (C.T c).tab ((C.T c).tab.view w ω)).2.1 r =
      honestMasked (Tz) st c r (zWith st w ω.1) (ω.2.1.2.1 r)
        ((C.T c).tab.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.nE).symm) := funext fun i => by
    show _ + _ = _ + _
    refine congrArg (_ + ·) ?_
    rw [Function.comp_apply, show (padEquiv (C.T c).tab.nE).symm (Fin.castAdd 4 i) = .inl i from
      (Equiv.symm_apply_eq _).2 (Fin.ext (by simp [padEquiv]))]
    rfl
  have hinner : (ZK.Table.View.pre (C.T c).tab ((C.T c).tab.view w ω)).2.2.2.2.1 r =
      (innerEps (Tz) st c r *
          ((C.T c).tab.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.nE).symm)
            (ix (C.T c).tab.nE (layOf (Tz) st).fa) +
        ((C.T c).tab.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.nE).symm)
          (ix (C.T c).tab.nE (layOf (Tz) st).tx),
       ((C.T c).tab.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.nE).symm)
          (ix (C.T c).tab.nE (layOf (Tz) st).fb) +
        ((C.T c).tab.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.nE).symm)
          (ix (C.T c).tab.nE (layOf (Tz) st).ty)) := by
    rw [comp_symm_lt _ _ hfa, comp_symm_lt _ _ hfb, htx, hty, comp_symm_tail _ 1 (by omega),
      comp_symm_tail _ 2 (by omega)]
    rfl
  have hab : ((C.T c).tab.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.nE).symm)
        (ix (C.T c).tab.nE (layOf (Tz) st).hab) =
      ((C.T c).tab.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.nE).symm)
          (ix (C.T c).tab.nE (layOf (Tz) st).fa) *
        ((C.T c).tab.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.nE).symm)
          (ix (C.T c).tab.nE (layOf (Tz) st).fb) := by
    rw [comp_symm_lt _ _ hfa, comp_symm_lt _ _ hfb,
      show ix (C.T c).tab.nE (layOf (Tz) st).hab = ix (C.T c).tab.nE ((C.T c).tab.nE + 0) from
        congrArg _ hE, comp_symm_tail _ 0 (by omega)]
    rfl
  have htw' : ((C.T c).tab.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.nE).symm)
        (ix (C.T c).tab.nE (layOf (Tz) st).tw) =
      ((C.T c).tab.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.nE).symm)
          (ix (C.T c).tab.nE (layOf (Tz) st).tx) *
        ((C.T c).tab.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.nE).symm)
          (ix (C.T c).tab.nE (layOf (Tz) st).ty) := by
    rw [htw, htx, hty, comp_symm_tail _ 1 (by omega), comp_symm_tail _ 2 (by omega), comp_symm_tail _ 3 (by omega)]
    rfl
  have hmS : st.m = (Tz).S.m := by rw [hS]; rfl
  have hkS : st.c.kLog = (Tz).S.kLog := by rw [hS]; rfl
  have hm7 : 7 ≤ (Tz).S.m := by have := (Tz).hm; omega
  obtain ⟨hnd, hlt⟩ := maskWordsOf_ok st M.m M.mask
  rw [hmS] at hlt
  have hs0 : ∀ v : Fin 128, fOf (shArr (ZK.Table.View.pre (C.T c).tab ((C.T c).tab.view w ω)).1 (2 * r.val))[v.val]! = shat hm7
      (RingSwitch.claimOf (Tz).S.m (Lincheck.lcAbClaim (Tz).S.m (Tz).S.kLog
        (rhoF (Tz) st c r) (tsF (Tz) st c r) (c.rest.rPrime r))) (zWith st w ω.1) v := by
    intro v
    have h2r : 2 * r.val < Zk.RANK_POINTS := by have := r.isLt; simp only [Zk.RANK_POINTS]; omega
    have hcl : Lincheck.lcAbClaim (Tz).S.m (Tz).S.kLog
        (rhoF (Tz) st c r) (tsF (Tz) st c r) (c.rest.rPrime r) =
        (claimsOf c.rest r)[0]! := by
      rw [← hmS, ← hkS]; rfl
    have hpt : pointsOf c.rest ⟨2 * r.val, h2r⟩ = ((claimsOf c.rest r)[0]!).x.toList.drop 1 := by
      show ((claimsOf c.rest (if 2 * r.val < 2 then 0 else 1))[2 * r.val % 2]!).x.toList.drop 1 = _
      rw [show 2 * r.val % 2 = 0 by omega, rep_of_lt r (2 * r.val) (by omega)]
    have hx : (Lincheck.lcAbClaim (Tz).S.m (Tz).S.kLog
        (rhoF (Tz) st c r) (tsF (Tz) st c r) (c.rest.rPrime r)).x.size =
        (Tz).S.m - 6 := by
      have := (Tz).S.hk
      simp only [Lincheck.lcAbClaim, rhoF, tsF, Array.size_append, Array.size_reverse, Array.size_ofFn,
        Array.size_extract]
      omega
    rw [RingSwitch.shat_eqWord _ _ hx, hcl, extract_one_toList, ← hpt]
    refine Eq.trans ?_ (sHat_shat st hm7 w rfl ω.1 hnd hlt
      (fun x => fOf (eqWord (pointsOf c.rest ⟨2 * r.val, h2r⟩) x)) v)
    rw [shArr, getElem!_pos _ v.val (by simp), Array.getElem_ofFn]
    unfold sHatAt
    split
    · rfl
    · exact absurd h2r ‹_›
  have hs1 : ∀ v : Fin 128, fOf (shArr (ZK.Table.View.pre (C.T c).tab ((C.T c).tab.view w ω)).1 (2 * r.val + 1))[v.val]! = shat hm7
      (RingSwitch.claimOf (Tz).S.m (cClaim (Tz) st c r)) (zWith st w ω.1) v := by
    intro v
    have h2r : 2 * r.val + 1 < Zk.RANK_POINTS := by have := r.isLt; simp only [Zk.RANK_POINTS]; omega
    have hcl : cClaim (Tz) st c r = (claimsOf c.rest r)[1]! := rfl
    have hpt : pointsOf c.rest ⟨2 * r.val + 1, h2r⟩ = ((claimsOf c.rest r)[1]!).x.toList.drop 1 := by
      show ((claimsOf c.rest (if 2 * r.val + 1 < 2 then 0 else 1))[(2 * r.val + 1) % 2]!).x.toList.drop 1 = _
      rw [show (2 * r.val + 1) % 2 = 1 by omega, rep_of_lt r (2 * r.val + 1) (by omega)]
    have hx : (cClaim (Tz) st c r).x.size = (Tz).S.m - 6 := by
      have := (Tz).hm
      simp only [cClaim, rF, Array.size_append, FlockLevel3.pinned_size, Array.size_ofFn]
      omega
    rw [RingSwitch.shat_eqWord _ _ hx, hcl, extract_one_toList, ← hpt]
    refine Eq.trans ?_ (sHat_shat st hm7 w rfl ω.1 hnd hlt
      (fun x => fOf (eqWord (pointsOf c.rest ⟨2 * r.val + 1, h2r⟩) x)) v)
    rw [shArr, getElem!_pos _ v.val (by simp), Array.getElem_ofFn]
    unfold sHatAt
    split
    · rfl
    · exact absurd h2r ‹_›
  rw [hv, hinner]
  refine rows_honest (Tz) st c r (zWith st w ω.1) (ω.2.1.2.1 r) _ hE hmS hkS ?hw
    (rF_ne_one _ _ c r hr) ?hA ?hB _ hs0 hs1 ?hk0 ?hL ?hT ((C.T c).tab.y1' w ω.1 ω.2.1.2.1 r)
    (fun _ => rfl) hab htw' k
  -- the executable statement's sizes are the table's, and the schedule's lane split
  case hk0 =>
    have := (Tz).hshape.2.2.1
    have := (Tz).hmsch
    omega
  case hL =>
    show 2 ^ (Tz).sch.level₀.logCols = _
    have := (Tz).hshape.2.2.1
    have := (Tz).hmsch
    rw [Accounting.Schedule.level₀]
    congr 1
    omega
  -- the prover's table `z = (w, u)` satisfies the statement
  case hw =>
    have key : ∀ (S' : Statement) (_ : S' = stmtOf st hwf (regionsOf st)) (w : Witness S'.m), S'.Satisfies w →
        S'.Satisfies (zWith st w ω.1) := by
      intro S' hS' w hw
      subst hS'
      exact satisfies_zWith st hwf (regionsOf st) M.m M.mask (maskIsolated_of_layout st hwf (maskLayout_of M hdm))
        w hw ω.1 rfl
    exact key _ hS w hw
  -- the clear `final_a`, `final_b` are the `a`, `b` claims at the zerocheck's coins
  case hA =>
    exact finalA_holds _ (six_le _) _ _ _ _ (fun i => by rw [aT, dite_eq_left i.isLt]) _ _ _
  case hB =>
    exact finalB_holds _ (six_le _) _ _ _ _ (fun i => by rw [bT, dite_eq_left i.isLt]) _ _ _
  -- the masked claims' `ŝ` is the honest `shat` at the replay's claims, and the opening's target is `Σ_y f(y)·b(y)`
  -- the opening's target is `Σ_y f(y)·b(y)`: ring switching claim by claim (`recombine_sum`)
  case hT =>
    have hrec : ∀ k : Fin (2 + 2 * st.regions.size),
        recombine execArith (swAt (Tz) st c r (ZK.Table.View.pre (C.T c).tab ((C.T c).tab.view w ω)).1 k.val) (fun i => fOf (c.rest.rpp r k i)) =
        ∑ y' : Fin (2 ^ ((Tz).S.m - 7)), wordOf (zWith st w ω.1) y'.val *
          Model.phi execArith (fun i => fOf (c.rest.rpp r k i)) (Model.eqAt (claimX (Tz) st c r k.val) y'.val) := by
      intro k
      refine recombine_sum execArith execArith_correct hm7 _ _ (wordOf_bits hm7 (zWith st w ω.1)) _ _
        (fun v => ?_) _
      rcases k with ⟨k, hk⟩
      by_cases h0 : k = 0
      · subst h0
        have h := hs0 v
        rw [shArr, getElem!_pos _ v.val (by simp), Array.getElem_ofFn] at h
        unfold swAt
        rw [ite_eq_left rfl]
        refine h.trans (Finset.sum_congr rfl fun y' _ => ?_)
        congr 2
        funext j
        unfold claimX
        rw [ite_eq_left rfl]
        exact abTail _ _ c r hmS hkS (j.val + 1) (by omega)
      by_cases h1 : k = 1
      · subst h1
        have h := hs1 v
        rw [shArr, getElem!_pos _ v.val (by simp), Array.getElem_ofFn] at h
        unfold swAt
        rw [ite_eq_right h0, ite_eq_left rfl]
        refine h.trans (Finset.sum_congr rfl fun y' _ => ?_)
        congr 2
        funext j
        unfold claimX
        rw [ite_eq_right h0, ite_eq_left rfl]
        exact cTail _ _ c r hmS (j.val + 1) (by omega)
      have h2 : k - 2 < 2 * st.regions.size := by omega
      unfold swAt
      rw [ite_eq_right h0, ite_eq_right h1]
      show dite _ _ _ = _
      rw [dite_eq_left h2]
      refine (congrFun (congrFun hreg ⟨k - 2, h2⟩) v).symm.trans ?_
      show (∑ x : Fin (2 ^ ((Tz).S.m - 7)), _) = _
      refine Finset.sum_congr rfl fun x _ => ?_
      rw [claimX_region _ _ c r k (by omega) h2 hmS]
      by_cases hx : x.val ∈ maskWordsOf st
      · have h0 := regions_off_of_words M.m (avoidsMask_words hdm) c.rest.link ⟨k - 2, h2⟩ x.val hx
        rw [h0, fOf_zero, zero_mul]
        show fOf (eqWord (regionPtOf st c.rest.link ⟨k - 2, h2⟩) x.val) * _ = 0
        rw [h0, fOf_zero, zero_mul]
      · refine congrArg (fOf (eqWord (regionPtOf st c.rest.link ⟨k - 2, h2⟩) x.val) * ·) ?_
        have hz := zWith_off st hm7 w rfl ω.1 x v hx
        rw [castF_eq]
        unfold wordBit
        rw [dite_eq_left (show 128 * x.val + v.val < 2 ^ _ from (addr128 hm7 x v).isLt)]
        exact congrArg (fun b : ZMod 2 => ((b.val : ℕ) : FlockLevel3.GF128)) hz.symm
    unfold openT
    refine (Finset.sum_congr rfl fun k _ => congrArg (fOf (c.rest.rsGamma r k) * ·) (hrec k)).trans ?_
    exact sum_swap_mul (fun k => fOf (c.rest.rsGamma r k)) (wordOf (zWith st w ω.1))
      (fun k y' => Model.phi execArith (fun i => fOf (c.rest.rpp r k i))
        (Model.eqAt (claimX (Tz) st c r k.val) y'))

/-- **The inner relation at a `--zk` table of a setup** (`innerOf_holds` at any table `Tz` over `st`'s statement at its
regions, `st` a `MaskSetup` that avoids the mask slot): at every coin vector the honest prover doesn't refuse, every valid
witness whose registered values are the public ones satisfies `Table.InnerHolds`. -/
theorem innerOf_holds_of (Tz : TabZK FlockLevel3.GF128 mPts) (st : Stmt) (hwf : StmtWF st) (hS : Tz.S = stmtOf st hwf (regionsOf st))
    (M : MaskSetup st) (hdm : avoidsMask st = true) :
    let C := coinedOf P Tz st
    ∀ c, C.Admits c → ∀ w, C.Valid w → (C.T c).tab.sReg w = (C.T c).tab.pubReg → (C.T c).tab.InnerHolds w := by
  intro C c hc w hw hreg ω r
  exact innerOf_holds_rep_of P Tz st hwf hS M hdm c r (hc.2.2.2 r) w hw hreg ω

/-- **The inner relation at the draw** (open, inner lane): at every coin vector the honest prover doesn't refuse, every
valid witness whose registered values are the public ones satisfies `Table.InnerHolds` (§2.4 line 16's ten
constraints). -/
theorem innerOf_holds (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) (S : Finset (Fin n)) (R : Reg) (y : DrawSetupZK I mPts S (dj S))
    (ht : I.tags.typed = false) (hdm : avoidsMask y.x.st = true) :
    let C := coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st
    ∀ c, C.Admits c → ∀ w, C.Valid w → (C.T c).tab.sReg w = (C.T c).tab.pubReg → (C.T c).tab.InnerHolds w :=
  innerOf_holds_of P _ y.x.st y.x.wf.1 (by rw [tablesAtZK_eq dj y₀ S R y]; rfl)
    (maskSetup_of_setupH ((ZkExec.zkTags_typed I).trans ht) y.x.h) hdm

end FlockSoundness.Discharge.Composed
