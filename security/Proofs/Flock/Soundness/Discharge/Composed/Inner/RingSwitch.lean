module

public import Proofs.Flock.Soundness.Discharge.Composed.Base
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.Polynomial.Basic
import Mathlib.Tactic.ReduceModChar
import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictInner
import Proofs.Flock.Soundness.Refine.Arith
import Proofs.Flock.Soundness.Refine.Final
import Proofs.Flock.Soundness.Refine.Opening

/-! The ring switch is complete for the honest prover at the replay's two claims and at every coin, because it
takes no division or inverse. -/

@[expose] public section

namespace FlockSoundness.Discharge.Composed.RingSwitch

open Flock Model Refine Exec Transcript ZkExec

local notation "𝔽" => FlockLevel3.GF128

def LinHolds (l : Zk.Lin) (h : Array F128) : Prop := dot l.a h = l.c

noncomputable def claimOf (m : ℕ) (c : Flock.Claim) : Model.Claim 𝔽 m := ⟨fOf c.value, vecOf 64 c.skip, vecOf (m - 6) c.x⟩

theorem two_pow_eq_mul64 {m : ℕ} (hm : 6 ≤ m) : 2 ^ m = 2 ^ (m - 6) * 64 := by
  rw [show (64 : ℕ) = 2 ^ 6 by norm_num, ← pow_add, Nat.sub_add_cancel hm]

def addr64 {m : ℕ} (hm : 6 ≤ m) (u : Fin (2 ^ (m - 6))) (i : Fin 64) : Fin (2 ^ m) :=
  ⟨64 * u.val + i.val, by rw [two_pow_eq_mul64 hm]; have := u.isLt; have := i.isLt; omega⟩

noncomputable def zPartialOf {m : ℕ} (hm : 6 ≤ m) (c : Model.Claim 𝔽 m) (z : Witness m) (i : Fin 64) : 𝔽 :=
  ∑ u : Fin (2 ^ (m - 6)), eqAt c.x u.val * ((z (addr64 hm u i)).val : 𝔽)

theorem fOf_foldl_mul (g : ℕ → F128) : ∀ (n : ℕ) (acc : F128),
    fOf ((List.range n).foldl (fun acc i => acc * g i) acc) = fOf acc * ∏ i ∈ Finset.range n, fOf (g i) := by
  intro n
  induction n with
  | zero => intro acc; simp
  | succ n ih =>
      intro acc
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, fOf_mul, ih, Finset.prod_range_succ,
        mul_assoc]

theorem fOf_eqWord (p : List F128) (y : ℕ) :
    fOf (eqWord p y) = ∏ i ∈ Finset.range p.length, if y.testBit i then fOf p[i]! else 1 + fOf p[i]! := by
  unfold eqWord
  rw [fOf_foldl_mul (fun i => if y.testBit i then p[i]! else 1 + p[i]!) p.length 1, fOf_one, one_mul]
  refine Finset.prod_congr rfl fun i _ => ?_
  split_ifs <;> simp only [fOf_add, fOf_one]

theorem eqAt_range {n : ℕ} (r : Fin n → 𝔽) (g : ℕ → 𝔽) (hg : ∀ j : Fin n, r j = g j.val) (b : ℕ) :
    eqAt r b = ∏ i ∈ Finset.range n, if b.testBit i then g i else 1 + g i := by
  unfold eqAt
  rw [← Fin.prod_univ_eq_prod_range (fun i => if b.testBit i then g i else 1 + g i) n]
  refine Finset.prod_congr rfl fun j _ => ?_
  rw [hg j]

theorem fOf_claimCheck {m : ℕ} (hm : 7 ≤ m) (c : Flock.Claim) (s : Array F128) :
    fOf (Zk.claimCheck c s) = ∑ v : Fin 128, packedWeight (claimOf m c) hm v * vecOf 128 s v := by
  show fOf (dot (wOf c) s) = _
  rw [fOf_dot _ _ 128 (by simp [wOf])]
  refine Finset.sum_congr rfl fun v _ => ?_
  simp only [wOf, Model.packedWeight]
  rw [Refine.getElem!_range_map _ v.isLt, fOf_mul]
  show fOf c.skip[v.val % 64]! * _ * _ = fOf c.skip[v.val % 64]! * _ * fOf s[v.val]!
  congr 2
  simp only [beq_iff_eq]
  split_ifs
  · rfl
  · rw [fOf_add, fOf_one]; rfl

theorem sum_zPartialOf {m : ℕ} (hm : 6 ≤ m) (c : Model.Claim 𝔽 m) (z : Witness m) :
    ∑ i : Fin 64, c.skip i * zPartialOf hm c z i = c.eval z := by
  classical
  let e : Fin (2 ^ (m - 6)) × Fin 64 ≃ Fin (2 ^ m) :=
    finProdFinEquiv.trans (finCongr (two_pow_eq_mul64 hm).symm)
  have he : ∀ q, e q = addr64 hm q.1 q.2 := fun q => by
    ext; simp [e, addr64, finProdFinEquiv]; ring
  unfold Claim.eval
  rw [← e.sum_comp, Fintype.sum_prod_type, Finset.sum_comm]
  refine Finset.sum_congr rfl fun i _ => ?_
  unfold zPartialOf
  refine (Finset.mul_sum _ _ _).trans ?_
  refine Finset.sum_congr rfl fun u _ => ?_
  simp only [he]
  have hmod : (addr64 hm u i).val % 64 = i.val := by simp only [addr64]; omega
  have hdiv : (addr64 hm u i).val / 64 = u.val := by simp only [addr64]; omega
  have hskip : c.skip ⟨(addr64 hm u i).val % 64, Nat.mod_lt _ (by norm_num)⟩ = c.skip i :=
    congrArg c.skip (Fin.ext hmod)
  rw [hskip, hdiv]
  ring

def zpLins (lay : Zk.PadLayout) (p : Proof) : Array Zk.Lin :=
  (Array.range 64).map fun i => Zk.Lin.masked p.lincheck.zPartial[i]! (lay.zp i) lay.len

theorem replay_values (st : Setup) (lay : Zk.PadLayout) (p : Proof) (root : ByteArray) {t t' : Flock.Transcript}
    {rp : Zk.Replayed} (h : Zk.replay st lay p root t = .ok (rp, t')) :
    rp.claims[0]!.skip.size = 64 ∧ rp.abValue = Zk.Lin.combo rp.claims[0]!.skip (zpLins lay p) lay.len ∧
      rp.cValue = Zk.Lin.masked p.zerocheck.finalC lay.fc lay.len := by
  unfold Zk.replay at h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨zz, _, _, h⟩ := bind_ok h
  obtain ⟨s, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨α₀, _, _, h⟩ := bind_ok h
  obtain ⟨β₀, _, _, h⟩ := bind_ok h
  obtain ⟨s₂, _, _, h⟩ := bind_ok h
  dsimp only at h
  cases hf : st.fold.folded α₀ β₀ (skipWeights zz) (s.2.extract 0 (st.kLog - 6)) s₂.2 with
  | ok v =>
    rw [hf] at h
    try dsimp only at h
    obtain ⟨combv, _, h₁₇, h'⟩ := bind_ok h
    clear h
    obtain ⟨rfl, rfl⟩ := pure_ok h₁₇
    obtain ⟨-, h⟩ := check_bind h'
    obtain ⟨_, _, _, h⟩ := bind_ok h
    obtain ⟨rP, _, _, h⟩ := bind_ok h
    obtain ⟨rfl, rfl⟩ := pure_ok h
    exact ⟨skipWeights_size rP, rfl, rfl⟩
  | error e =>
    rw [hf] at h
    try dsimp only at h
    obtain ⟨_, _, h₁₇, _⟩ := bind_ok h
    cases h₁₇

theorem zp_lt (lay : Zk.PadLayout) {i : ℕ} (hi : i < 64) : lay.zp i < lay.len := by
  simp only [Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.hab, Zk.PadLayout.zp]
  omega

theorem fc_lt (lay : Zk.PadLayout) : lay.fc < lay.len := by
  simp only [Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.hab, Zk.PadLayout.zp, Zk.PadLayout.lc]
  omega

theorem masked_a_size (v : F128) (i s : ℕ) : (Zk.Lin.masked v i s).a.size = s := by
  simp [Zk.Lin.masked]

theorem masked_a_get (v : F128) {i s j : ℕ} (hj : j < s) :
    (Zk.Lin.masked v i s).a[j]! = if j = i then 1 else 0 := by
  show ((Array.replicate s 0).set! i 1)[j]! = _
  rw [Array.set!_eq_setIfInBounds, getElem!_pos _ j (by simp; exact hj),
    Array.getElem_setIfInBounds (by simp; exact hj)]
  by_cases h : j = i
  · subst h; simp
  · simp [h, Ne.symm h]

theorem fOf_masked_a (v : F128) {i s j : ℕ} (hj : j < s) :
    fOf (Zk.Lin.masked v i s).a[j]! = if j = i then 1 else 0 := by
  rw [masked_a_get v hj]
  split_ifs
  · exact fOf_one
  · exact fOf_zero

theorem fOf_dot_masked (v : F128) {i s : ℕ} (hi : i < s) (h : Array F128) :
    fOf (dot (Zk.Lin.masked v i s).a h) = fOf h[i]! := by
  rw [fOf_dot _ _ s (masked_a_size v i s)]
  rw [Finset.sum_eq_single ⟨i, hi⟩ (fun j _ hj => by
    rw [fOf_masked_a v j.isLt, ite_eq_right (fun e => hj (Fin.ext e)), zero_mul]) (by simp)]
  rw [fOf_masked_a v hi, ite_eq_left rfl, one_mul]

theorem zpLins_get (lay : Zk.PadLayout) (p : Proof) {k : ℕ} (hk : k < 64) :
    (zpLins lay p)[k]! = Zk.Lin.masked p.lincheck.zPartial[k]! (lay.zp k) lay.len :=
  Refine.getElem!_range_map _ hk

theorem fOf_dot_combo (ws : Array F128) (hw : ws.size = 64) (lay : Zk.PadLayout) (p : Proof) (h : Array F128) :
    fOf (dot (Zk.Lin.combo ws (zpLins lay p) lay.len).a h) = ∑ k : Fin 64, fOf ws[k.val]! * fOf h[lay.zp k.val]! := by
  have hls : ∀ k < 64, ((zpLins lay p)[k]!).a.size = lay.len := fun k hk => by
    rw [zpLins_get lay p hk, masked_a_size]
  rw [fOf_dot _ _ lay.len (combo_size _ _ _)]
  have e : ∀ j : Fin lay.len, fOf (Zk.Lin.combo ws (zpLins lay p) lay.len).a[j.val]! * fOf h[j.val]! =
      ∑ k : Fin 64, if j.val = lay.zp k.val then fOf ws[k.val]! * fOf h[lay.zp k.val]! else 0 := fun j => by
    rw [combo_a ws (zpLins lay p) lay.len 64 hw (by simp [zpLins]) hls j.val j.isLt]
    refine (Finset.sum_mul _ _ _).trans ?_
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [zpLins_get lay p k.isLt, fOf_masked_a _ j.isLt]
    split_ifs with hj
    · rw [mul_one, hj]
    · rw [mul_zero, zero_mul]
  rw [Finset.sum_congr rfl fun j _ => e j, Finset.sum_comm]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [Finset.sum_eq_single ⟨lay.zp k.val, zp_lt lay k.isLt⟩ (fun j _ hj => by
    rw [ite_eq_right (fun e => hj (Fin.ext e))]) (by simp), ite_eq_left rfl]

theorem ringSwitch_holds (st : Setup) (lay : Zk.PadLayout) (p : Proof) (root : ByteArray) {t t' : Flock.Transcript}
    {rp : Zk.Replayed} (hrun : Zk.replay st lay p root t = .ok (rp, t')) (hm : 7 ≤ st.m) (z : Witness st.m)
    (h : Array F128)
    (hzp : ∀ i : Fin 64, fOf p.lincheck.zPartial[i.val]! =
      zPartialOf (by omega) (claimOf st.m rp.claims[0]!) z i + fOf h[lay.zp i.val]!)
    (hfc : fOf p.zerocheck.finalC = (claimOf st.m rp.claims[1]!).eval z + fOf h[lay.fc]!)
    (hs : ∀ (k : Fin 2) (v : Fin 128),
      fOf (p.ringSwitches[k.val]!)[v.val]! = shat hm (claimOf st.m rp.claims[k.val]!) z v) :
    LinHolds (rp.abValue.addConst (Zk.claimCheck rp.claims[0]! p.ringSwitches[0]!)) h ∧
      LinHolds (rp.cValue.addConst (Zk.claimCheck rp.claims[1]! p.ringSwitches[1]!)) h := by
  obtain ⟨hsk, hab, hc⟩ := replay_values st lay p root hrun
  have hcc : ∀ k : Fin 2, fOf (Zk.claimCheck rp.claims[k.val]! p.ringSwitches[k.val]!) =
      (claimOf st.m rp.claims[k.val]!).eval z := fun k => by
    rw [fOf_claimCheck hm, claim_eval_eq hm]
    exact Finset.sum_congr rfl fun v _ => by rw [show vecOf 128 _ v = _ from hs k v]
  have h0 : fOf (Zk.claimCheck rp.claims[0]! p.ringSwitches[0]!) = (claimOf st.m rp.claims[0]!).eval z := hcc 0
  have h1 : fOf (Zk.claimCheck rp.claims[1]! p.ringSwitches[1]!) = (claimOf st.m rp.claims[1]!).eval z := hcc 1
  refine ⟨fOf_injective ?_, fOf_injective ?_⟩
  · show fOf (dot rp.abValue.a h) = fOf (rp.abValue.c + Zk.claimCheck rp.claims[0]! p.ringSwitches[0]!)
    rw [fOf_add, h0, hab, fOf_dot_combo _ hsk, combo_c _ _ _ 64 hsk (by simp [zpLins])]
    have hterm : ∀ k : Fin 64, fOf rp.claims[0]!.skip[k.val]! * fOf ((zpLins lay p)[k.val]!).c =
        (claimOf st.m rp.claims[0]!).skip k * zPartialOf (by omega) (claimOf st.m rp.claims[0]!) z k +
          fOf rp.claims[0]!.skip[k.val]! * fOf h[lay.zp k.val]! := fun k => by
      rw [zpLins_get lay p k.isLt]
      show _ * fOf p.lincheck.zPartial[k.val]! = _
      rw [hzp k, mul_add]
      rfl
    rw [Finset.sum_congr rfl fun k _ => hterm k, Finset.sum_add_distrib, sum_zPartialOf, add_comm, ← add_assoc,
      CharTwo.add_self_eq_zero, zero_add]
  · show fOf (dot rp.cValue.a h) = fOf (rp.cValue.c + Zk.claimCheck rp.claims[1]! p.ringSwitches[1]!)
    rw [fOf_add, h1, hc, fOf_dot_masked _ (fc_lt lay)]
    show fOf h[lay.fc]! = fOf p.zerocheck.finalC + _
    rw [hfc, add_comm, ← add_assoc, CharTwo.add_self_eq_zero, zero_add]

theorem fOf_dot_right (a b : Array F128) (n : ℕ) (hb : b.size = n) :
    fOf (dot a b) = ∑ i : Fin n, fOf a[i.val]! * fOf b[i.val]! := by
  rw [dot_eq_foldl, fOf_foldl_dot, fOf_zero, zero_add, Fin.sum_univ_eq_sum_range (fun i => fOf a[i]! * fOf b[i]!) n,
    hb]
  by_cases ha : n ≤ a.size
  · rw [min_eq_right ha]
  · rw [min_eq_left (by omega)]
    refine Finset.sum_subset (Finset.range_mono (by omega)) fun i hi hni => ?_
    simp only [Finset.mem_range, not_lt] at hi hni
    rw [getElem!_neg a i (by omega)]
    show fOf 0 * fOf b[i]! = 0
    rw [fOf_zero, zero_mul]

theorem ringSwitch_target (m : ℕ) (hm : 7 ≤ m) (claims : Array Flock.Claim) (rs : Array (Array F128))
    {t t' : Flock.Transcript} {T : F128} {sws : Array Switched} (hwf : RoundsWF t.rounds)
    (hrun : ringSwitch m claims rs t = .ok ((T, sws), t')) (hx : ∀ k < claims.size, claims[k]!.x.size = m - 6)
    (z : Witness m) (f : Fin (2 ^ (m - 7)) → 𝔽) (hz : ∀ y v, z (addr128 hm y v) = execArith.unpack (f y) v)
    (hs : ∀ k < claims.size, ∀ v : Fin 128, fOf (rs[k]!)[v.val]! = shat hm (claimOf m claims[k]!) z v) :
    sws.size = claims.size ∧
      (∀ k < claims.size, (sws[k]!).z = claims[k]!.x.extract 1 ∧ (sws[k]!).z.size = m - 7 ∧
        (sws[k]!).rpp.size = 7) ∧
      fOf T = ∑ y : Fin (2 ^ (m - 7)), f y * ∑ k : Fin claims.size, fOf (sws[k.val]!).gamma *
        phi execArith (vecOf 7 (sws[k.val]!).rpp) (eqAt (vecOf (m - 7) (sws[k.val]!).z) y.val) := by
  unfold ringSwitch at hrun
  obtain ⟨t₁, k₁, h⟩ := label_bind hrun
  obtain ⟨s, t₂, hloop, h⟩ := bind_ok h
  obtain ⟨cγ, t₃, -, h⟩ := bind_ok h
  obtain ⟨hTs, -⟩ := pure_ok h
  rw [forIn_range_eq] at hloop
  obtain ⟨-, -, hQ, hs'⟩ := forIn_roundsQ _ (oStep claims rs) 7 (RsCheck claims rs) (by
      intro k st tt b tt' hb
      obtain ⟨tt₁, kk₁, hb⟩ := label_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbSlice_bind hb
      obtain ⟨hval, hb⟩ := check_bind hb
      obtain ⟨-, nn, tt₃, rr₃, kk₃, hb⟩ := sampleVec_bind (by omega) hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      have e1 : tt₂.rounds = tt.rounds := by rw [kk₂.2, kk₁.2]
      have e2 : tt₂.k = tt.k := by rw [kk₂.1, kk₁.1]
      refine ⟨by rw [← e1, ← e2]; exact nn, by rw [rr₃, e1], by rw [kk₃, e2], ?_, hval⟩
      rw [← e1, ← e2]
      rfl) claims.size 0 t₁.k _ t₁ t₂ s (by omega) hloop
  obtain ⟨hT, hsws⟩ := Prod.mk.inj hTs
  set C : ℕ → Array F128 := fun i => (t₁.rounds[t₁.k + i]!).coins with hC
  have hwf₁ : RoundsWF t₁.rounds := k₁.2 ▸ hwf
  have hC7 : ∀ k < claims.size, (C k).size = 7 := fun k hk => by
    rw [hC]; simp only; rw [RoundsWF.get hwf₁, (hQ k (by simp; omega)).1]
  obtain ⟨hs1, hs2⟩ := oStep_fold claims rs C (List.range' 0 claims.size) (#[], #[])
  rw [← hs'] at hs1 hs2
  have hs1size : s.1.size = claims.size := by rw [hs1]; simp
  have hs1k : ∀ k < claims.size, s.1[k]! = dot (eqTable (C k)) (tOf rs[k]!) := fun k hk => by
    rw [hs1, getElem!_pos _ k (by simp [hk])]; simp
  have hs2k : ∀ k < claims.size, s.2[k]! = (C k, claims[k]!.x.extract 1) := fun k hk => by
    rw [hs2, getElem!_pos _ k (by simp [hk])]; simp
  have hsw : ∀ k < claims.size, sws[k]! = { gamma := cγ[k]!, rpp := C k, z := claims[k]!.x.extract 1 } := by
    intro k hk
    rw [hsws, getElem!_pos _ k (by simp [hk]), Array.getElem_map, Array.getElem_range, hs2k k hk]
  refine ⟨by rw [hsws]; simp, fun k hk => ?_, ?_⟩
  · rw [hsw k hk]
    refine ⟨rfl, ?_, hC7 k hk⟩
    have := hx k hk
    simp; omega
  · have hk : ∀ k : Fin claims.size, fOf cγ[k.val]! * fOf s.1[k.val]! = ∑ y : Fin (2 ^ (m - 7)),
        f y * (fOf (sws[k.val]!).gamma *
          phi execArith (vecOf 7 (sws[k.val]!).rpp) (eqAt (vecOf (m - 7) (sws[k.val]!).z) y.val)) := fun k => by
      rw [hsw k.val k.isLt, hs1k k.val k.isLt, recombine_eq _ _ (hC7 k.val k.isLt)]
      have hsv : vecOf 128 rs[k.val]! = shat hm (claimOf m claims[k.val]!) z :=
        funext fun v => hs k.val k.isLt v
      have htail : (claimOf m claims[k.val]!).tail hm = vecOf (m - 7) (claims[k.val]!.x.extract 1) := by
        funext j
        have := hx k.val k.isLt
        simp only [Model.Claim.tail, claimOf, vecOf]
        rw [FlockLevel3.extract_get! _ 1 _ j (by omega) le_rfl, Nat.add_comm]
      rw [hsv, recombine_shat execArith execArith_correct hm _ z f hz, htail]
      refine (Finset.mul_sum _ _ _).trans (Finset.sum_congr rfl fun y _ => ?_)
      ring
    rw [hT, fOf_dot_right cγ s.1 claims.size hs1size, Finset.sum_congr rfl fun k _ => hk k, Finset.sum_comm]
    exact Finset.sum_congr rfl fun y _ => (Finset.mul_sum _ _ _).symm

theorem eqWord_eq_eqAt (p : List F128) (y : ℕ) :
    fOf (eqWord p y) = eqAt (fun j : Fin p.length => fOf p[j.val]) y := by
  rw [fOf_eqWord, eqAt_range _ (fun i => fOf p[i]!) fun j => by rw [getElem!_pos p j.val j.isLt]]

theorem shat_eqWord {m : ℕ} (hm : 7 ≤ m) (c : Flock.Claim) (hx : c.x.size = m - 6) (z : Witness m) (v : Fin 128) :
    shat hm (claimOf m c) z v =
      ∑ y : Fin (2 ^ (m - 7)), fOf (eqWord (c.x.extract 1).toList y.val) * ((z (addr128 hm y v)).val : 𝔽) := by
  unfold shat
  refine Finset.sum_congr rfl fun y _ => ?_
  congr 1
  have hlen : (c.x.extract 1).toList.length = m - 7 := by simp [hx]; omega
  rw [fOf_eqWord, hlen, eqAt_range ((claimOf m c).tail hm) (fun i => fOf c.x[i + 1]!) fun j => rfl]
  refine Finset.prod_congr rfl fun i hi => ?_
  have hi' : i < (c.x.extract 1).toList.length := by rw [hlen]; exact Finset.mem_range.1 hi
  have hx1 : (c.x.extract 1).toList[i]! = c.x[i + 1]! := by
    rw [getElem!_pos _ i hi', getElem!_pos c.x (i + 1) (by simp at hi'; omega)]
    simp
  rw [hx1]

end FlockSoundness.Discharge.Composed.RingSwitch
