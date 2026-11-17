module

public import Proofs.Flock.Claims.Complete.Zk.Inner
public import Proofs.Flock.Soundness.Discharge.Composed.Inner.ZerocheckMasked

/-! The honest pads' algebra: at a rep's pads the batched constraint `innerP` checks holds at every coin once the rep's
own constraints vanish (`innerP_batched`). -/

@[expose] public section

namespace FlockComplete

open Flock FlockLevel3 FlockSoundness.Discharge.Composed.ZerocheckMasked
open FlockSoundness.Discharge.Composed.Zerocheck (eq_of_of of_one' mul_sum')

def ConsHold (cons : Array (String × Zk.Lin)) (h : Array F128) : Prop :=
  ∀ q ∈ cons, q.2.a.size = h.size ∧ evalL q.2 h = 0

theorem ConsHold.append {c₁ c₂ : Array (String × Zk.Lin)} {h : Array F128} (h₁ : ConsHold c₁ h)
    (h₂ : ConsHold c₂ h) : ConsHold (c₁ ++ c₂) h := by
  intro q hq
  rcases Array.mem_append.1 hq with hq | hq
  · exact h₁ q hq
  · exact h₂ q hq

theorem default_F128 : (default : F128) = 0 := rfl

theorem batched_of_holds {cons : Array (String × Zk.Lin)} {h : Array F128} (hc : ConsHold cons h)
    (ws : Array F128) : evalL (Zk.Lin.combo ws (cons.map (·.2)) h.size) h = 0 := by
  rw [evalL_combo ws _ h.size h (fun l hl => by
    obtain ⟨q, hq, rfl⟩ := Array.mem_map.1 hl
    exact (hc q hq).1)]
  have hz : ∀ k, ((cons.map (·.2)).map fun l => evalL l h)[k]! = (0 : F128) := fun k => by
    by_cases hk : k < cons.size
    · have hk' : k < ((cons.map (·.2)).map fun l => evalL l h).size := by simpa using hk
      rw [getElem!_pos _ k hk']
      simp only [Array.getElem_map]
      exact (hc _ (Array.getElem_mem _)).2
    · have hk' : ¬k < ((cons.map (·.2)).map fun l => evalL l h).size := by simpa using hk
      rw [getElem!_neg _ k hk']
      exact default_F128
  apply eq_of_of
  rw [of_dot_range ws _ ws.size rfl, ← GF128.zero_eq]
  refine Finset.sum_eq_zero fun k _ => ?_
  rw [hz k, ← GF128.zero_eq, mul_zero]

theorem of_dot_replicate (s : ℕ) (h : Array F128) : GF128.of (dot (Array.replicate s 0) h) = 0 := by
  rw [of_dot_range _ h s (by simp)]
  refine Finset.sum_eq_zero fun k hk => ?_
  rw [get_replicate s k (Finset.mem_range.1 hk), ← GF128.zero_eq, zero_mul]

theorem of_dot_set (a h : Array F128) (i : ℕ) (v : F128) (hi : i < a.size) (h0 : a[i]! = 0) :
    GF128.of (dot (a.set! i v) h) = GF128.of (dot a h) + GF128.of v * GF128.of h[i]! := by
  rw [of_dot_range _ h a.size (by simp), of_dot_range a h a.size rfl]
  have hent : ∀ k ∈ Finset.range a.size, GF128.of (a.set! i v)[k]! * GF128.of h[k]! =
      GF128.of a[k]! * GF128.of h[k]! + if k = i then GF128.of v * GF128.of h[i]! else 0 := fun k hk => by
    rw [get_set a i k v (Finset.mem_range.1 hk)]
    by_cases hki : k = i
    · subst hki
      rw [ite_eq_left rfl, ite_eq_left rfl, h0, ← GF128.zero_eq, zero_mul, zero_add]
    · rw [ite_eq_right hki, ite_eq_right hki, add_zero]
  rw [Finset.sum_congr rfl hent, Finset.sum_add_distrib, Finset.sum_ite_eq' (Finset.range a.size) i,
    ite_eq_left (Finset.mem_range.2 hi)]

theorem get_set_ne (a : Array F128) (i k : ℕ) (v : F128) (hk : k < a.size) (hki : k ≠ i) :
    (a.set! i v)[k]! = a[k]! := by
  rw [get_set a i k v hk, ite_eq_right hki]

theorem hab_lt_tx (lay : Zk.PadLayout) : lay.hab < lay.tx := by simp only [Zk.PadLayout.tx]; omega
theorem tx_lt_ty (lay : Zk.PadLayout) : lay.tx < lay.ty := by
  simp only [Zk.PadLayout.tx, Zk.PadLayout.ty]; omega
theorem ty_lt_tw (lay : Zk.PadLayout) : lay.ty < lay.tw := by
  simp only [Zk.PadLayout.ty, Zk.PadLayout.tw]; omega
theorem tw_lt_len (lay : Zk.PadLayout) : lay.tw < lay.len := by simp only [Zk.PadLayout.len]; omega

theorem padsOf_size (tape : TapeZk) (i r : ℕ) (lay : Zk.PadLayout) : (padsOf tape i r lay).h.size = lay.len := by
  simp [padsOf]

theorem padsOf_get (tape : TapeZk) (i r : ℕ) (lay : Zk.PadLayout) (k : ℕ) (hk : k < lay.len) (h1 : k ≠ lay.hab)
    (h2 : k ≠ lay.tw) : (padsOf tape i r lay).h[k]! = tape.h i r k := by
  simp only [padsOf]
  rw [get_set_ne _ _ _ _ (by simpa using hk) h2, get_set_ne _ _ _ _ (by simpa using hk) h1,
    getElem!_pos _ k (by simpa using hk)]
  simp

theorem padsOf_hab (tape : TapeZk) (i r : ℕ) (lay : Zk.PadLayout) :
    (padsOf tape i r lay).h[lay.hab]! = (padsOf tape i r lay).h[lay.fa]! * (padsOf tape i r lay).h[lay.fb]! := by
  have := fa_lt_fb lay; have := fb_lt_hab lay; have := hab_lt_tx lay; have := tx_lt_ty lay
  have := ty_lt_tw lay; have := tw_lt_len lay
  rw [padsOf_get tape i r lay lay.fa (by omega) (by omega) (by omega),
    padsOf_get tape i r lay lay.fb (by omega) (by omega) (by omega)]
  simp only [padsOf]
  rw [get_set_ne _ _ _ _ (by simp; omega) (by omega), get_set _ _ _ _ (by simp; omega), ite_eq_left rfl,
    getElem!_pos _ lay.fa (by simp; omega), getElem!_pos _ lay.fb (by simp; omega)]
  simp

theorem padsOf_tw (tape : TapeZk) (i r : ℕ) (lay : Zk.PadLayout) :
    (padsOf tape i r lay).h[lay.tw]! = (padsOf tape i r lay).h[lay.tx]! * (padsOf tape i r lay).h[lay.ty]! := by
  have := fa_lt_fb lay; have := fb_lt_hab lay; have := hab_lt_tx lay; have := tx_lt_ty lay
  have := ty_lt_tw lay; have := tw_lt_len lay
  rw [padsOf_get tape i r lay lay.tx (by omega) (by omega) (by omega),
    padsOf_get tape i r lay lay.ty (by omega) (by omega) (by omega)]
  simp only [padsOf]
  rw [get_set _ _ _ _ (by simp; omega), ite_eq_left rfl, get_set_ne _ _ _ _ (by simp; omega) (by omega),
    get_set_ne _ _ _ _ (by simp; omega) (by omega), getElem!_pos _ lay.tx (by simp; omega),
    getElem!_pos _ lay.ty (by simp; omega)]
  simp

theorem triple_holds (lay : Zk.PadLayout) (h : Array F128) (hs : h.size = lay.len)
    (hab : h[lay.hab]! = h[lay.fa]! * h[lay.fb]!) (htw : h[lay.tw]! = h[lay.tx]! * h[lay.ty]!) (eps : F128) :
    ConsHold (Zk.triple lay eps (eps * h[lay.fa]! + h[lay.tx]!) (h[lay.fb]! + h[lay.ty]!)) h := by
  have := fa_lt_fb lay; have := fb_lt_hab lay; have := hab_lt_tx lay; have := tx_lt_ty lay
  have := ty_lt_tw lay; have := tw_lt_len lay
  have two : ∀ x : GF128, x + x = 0 := fun x => CharTwo.add_self_eq_zero x
  have z0 : ∀ k < lay.len, (Array.replicate lay.len (0 : F128))[k]! = 0 := fun k hk => get_replicate _ k hk
  have zs : (Array.replicate lay.len (0 : F128)).size = lay.len := by simp
  intro q hq
  simp only [Zk.triple, List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hq
  rcases hq with rfl | rfl | rfl
  · refine ⟨by simp [hs], eq_of_of ?_⟩
    dsimp only [evalL]
    rw [of_add, of_dot_set _ h _ _ (by simp; omega)
        (by rw [get_set_ne _ _ _ _ (by simp; omega) (by omega)]; exact z0 _ (by omega)),
      of_dot_set _ h _ _ (by simp; omega) (z0 _ (by omega)), of_dot_replicate, of_add, of_mul, of_one',
      ← GF128.zero_eq]
    linear_combination two (GF128.of eps * GF128.of h[lay.fa]! + GF128.of h[lay.tx]!)
  · refine ⟨by simp [hs], eq_of_of ?_⟩
    dsimp only [evalL]
    rw [of_add, of_dot_set _ h _ _ (by simp; omega)
        (by rw [get_set_ne _ _ _ _ (by simp; omega) (by omega)]; exact z0 _ (by omega)),
      of_dot_set _ h _ _ (by simp; omega) (z0 _ (by omega)), of_dot_replicate, of_add, of_one',
      ← GF128.zero_eq]
    linear_combination two (GF128.of h[lay.fb]! + GF128.of h[lay.ty]!)
  · refine ⟨by simp [hs], eq_of_of ?_⟩
    dsimp only [evalL]
    rw [of_add, of_dot_set _ h _ _ (by simp; omega) (by
        rw [get_set_ne _ _ _ _ (by simp; omega) (by omega), get_set_ne _ _ _ _ (by simp; omega) (by omega),
          get_set_ne _ _ _ _ (by simp; omega) (by omega)]
        exact z0 _ (by omega)),
      of_dot_set _ h _ _ (by simp; omega) (by
        rw [get_set_ne _ _ _ _ (by simp; omega) (by omega), get_set_ne _ _ _ _ (by simp; omega) (by omega)]
        exact z0 _ (by omega)),
      of_dot_set _ h _ _ (by simp; omega) (by
        rw [get_set_ne _ _ _ _ (by simp; omega) (by omega)]
        exact z0 _ (by omega)),
      of_dot_set _ h _ _ (by simp; omega) (z0 _ (by omega)), of_dot_replicate, hab, htw]
    simp only [of_add, of_mul, of_one']
    rw [← GF128.zero_eq]
    linear_combination two (GF128.of eps * GF128.of h[lay.fa]! * GF128.of h[lay.fb]! +
      GF128.of eps * GF128.of h[lay.fa]! * GF128.of h[lay.ty]! + GF128.of h[lay.tx]! * GF128.of h[lay.fb]! +
      2 * GF128.of h[lay.tx]! * GF128.of h[lay.ty]!)

theorem innerP_batched (tape : TapeZk) (i r : ℕ) (lay : Zk.PadLayout) {cons : Array (String × Zk.Lin)}
    (hc : ConsHold cons (padsOf tape i r lay).h) (eps : F128) (gammas : Array F128) :
    let p := padsOf tape i r lay
    let l := Zk.Lin.combo gammas ((cons ++ Zk.triple lay eps (eps * p.h[lay.fa]! + p.h[lay.tx]!)
      (p.h[lay.fb]! + p.h[lay.ty]!)).map (·.2)) lay.len
    (l.c + dot l.a p.h).isZero = true := by
  intro p l
  have hs : p.h.size = lay.len := padsOf_size tape i r lay
  have := batched_of_holds (hc.append (triple_holds lay p.h hs (padsOf_hab tape i r lay)
    (padsOf_tw tape i r lay) eps)) gammas
  rw [hs] at this
  have e : evalL l p.h = 0 := this
  show (evalL l p.h).isZero = true
  rw [e]
  rfl

def innerY (hv mv : Array F128) (beta : F128) : Array F128 :=
  (Array.range hv.size).map fun i => hv[i]! + beta * mv[i]!

theorem innerY_size (hv mv : Array F128) (beta : F128) : (innerY hv mv beta).size = hv.size := by simp [innerY]

theorem innerY_get (hv mv : Array F128) (beta : F128) {i : ℕ} (hi : i < hv.size) :
    (innerY hv mv beta)[i]! = hv[i]! + beta * mv[i]! := by
  rw [getElem!_pos _ i (by simpa [innerY] using hi)]
  simp [innerY]

theorem get_append_left (a b : Array F128) {i : ℕ} (hi : i < a.size) : (a ++ b)[i]! = a[i]! := by
  rw [getElem!_pos _ i (by simp; omega), getElem!_pos a i hi, Array.getElem_append_left hi]

theorem f128_beq_self (x : F128) : (x == x) = true := by cases x; simp [BEq.beq, instBEqF128.beq]

theorem isZero_eq {a : F128} (h : a.isZero = true) : a = 0 := FlockLevel3.F128.isZero_iff a h

theorem dot_innerY (a h mu padH padMu : Array F128) (beta : F128) (ha : a.size = h.size) (hm : mu.size = h.size) :
    dot a (innerY (h ++ padH) (mu ++ padMu) beta) = dot a h + beta * dot a mu := by
  apply eq_of_of
  rw [of_dot_range a _ h.size ha, of_add, of_mul, of_dot_range a h h.size ha, of_dot_range a mu h.size ha,
    mul_sum', ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun k hk => ?_
  have hk' := Finset.mem_range.1 hk
  rw [innerY_get _ _ _ (by simp; omega), get_append_left _ _ hk', get_append_left _ _ (by omega), of_add, of_mul]
  ring

theorem innerY_batched (l : Zk.Lin) (h mu padH padMu : Array F128) (beta : F128) (ha : l.a.size = h.size)
    (hm : mu.size = h.size) (hz : (l.c + dot l.a h).isZero = true) :
    (dot l.a (innerY (h ++ padH) (mu ++ padMu) beta) == l.c + beta * dot l.a mu) = true := by
  rw [dot_innerY _ _ _ _ _ _ ha hm]
  have e : dot l.a h = l.c := by
    have := isZero_eq hz
    apply eq_of_of
    have h2 := congrArg GF128.of this
    rw [of_add, ← GF128.zero_eq] at h2
    linear_combination h2 - CharTwo.add_self_eq_zero (GF128.of l.c)
  rw [e]
  exact f128_beq_self _

theorem of_horner (x : F128) : ∀ l : List F128,
    GF128.of (l.foldr (fun c acc => acc * x + c) 0) =
      ∑ i ∈ Finset.range l.length, GF128.of (l.getD i 0) * GF128.of x ^ i
  | [] => by simp [← GF128.zero_eq]
  | c :: l => by
      rw [List.foldr_cons, of_add, of_mul, of_horner x l, List.length_cons, Finset.sum_range_succ']
      refine congrArg₂ (· + ·) ((Finset.sum_mul _ _ _).trans (Finset.sum_congr rfl fun i _ => ?_)) ?_
      · rw [List.getD_cons_succ, pow_succ, mul_assoc]
      · simp

theorem of_codeAt (y : Array F128) (j : ℕ) :
    GF128.of (Zk.codeAt y j) = ∑ i ∈ Finset.range y.size, GF128.of y[i]! * GF128.of ⟨(j + 1).toUInt64, 0⟩ ^ i := by
  show GF128.of (y.foldr (fun c acc => acc * (⟨(j + 1).toUInt64, 0⟩ : F128) + c) 0) = _
  rw [← Array.foldr_toList, of_horner, Array.length_toList]
  refine Finset.sum_congr rfl fun i hi => ?_
  have hi' := Finset.mem_range.1 hi
  have e : y.toList.getD i 0 = y[i]! := by simp [List.getD_eq_getElem?_getD, hi']
  rw [e]

theorem codeAt_innerY (hv mv : Array F128) (beta : F128) (hs : mv.size = hv.size) (j : ℕ) :
    Zk.codeAt (innerY hv mv beta) j = Zk.codeAt hv j + beta * Zk.codeAt mv j := by
  apply eq_of_of
  rw [of_add, of_mul, of_codeAt, of_codeAt, of_codeAt, innerY_size, hs, mul_sum', ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i hi => ?_
  rw [innerY_get _ _ _ (Finset.mem_range.1 hi), of_add, of_mul]
  ring

theorem innerY_column (hv mv : Array F128) (beta : F128) (hs : mv.size = hv.size) (n j : ℕ) (hj : j < n) :
    (Zk.codeAt (innerY hv mv beta) j ==
      ((Array.range n).map (Zk.codeAt hv))[j]! + beta * ((Array.range n).map (Zk.codeAt mv))[j]!) = true := by
  rw [codeAt_innerY hv mv beta hs, getElem!_pos _ j (by simpa using hj), getElem!_pos _ j (by simpa using hj)]
  simp only [Array.getElem_map, Array.getElem_range]
  exact f128_beq_self _

theorem padsOf_mu_size (tape : TapeZk) (i r : ℕ) (lay : Zk.PadLayout) :
    (padsOf tape i r lay).mu.size = (padsOf tape i r lay).h.size := by
  simp [padsOf]

end FlockComplete
