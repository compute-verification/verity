import Proofs.Core.TC.Proofs.SemGroup

/-!
# One group, both sides: `group_sound`

The relation's special-value flags are the semantics' (`ind_sum`, `contains_infSign`), its finite path is `group_sum`
(`groupSum_eq`), and its output state is the canonical state of `group_step_total`'s value.
-/

namespace Verity.TC.Proofs

open Verity.TC Verity.TC.Spec Verity.TC.Relation

/-- The 0/1 indicator of a test on a value. -/
def ind (p : Value → Bool) (v : Value) : Int := if p v then 1 else 0

theorem ind_sum (p : Value → Bool) : ∀ l : List Value,
    0 ≤ (l.map (ind p)).sum ∧ (l.map (ind p)).sum ≤ l.length ∧ ((l.map (ind p)).sum ≠ 0 ↔ l.any p = true)
  | [] => by simp
  | x :: l => by
    have ih := ind_sum p l
    simp only [List.map_cons, List.sum_cons, List.length_cons, List.any_cons, Bool.or_eq_true]
    have hx : ind p x = if p x then 1 else 0 := rfl
    rw [hx]
    cases p x
    · simp only [Bool.false_eq_true, ite_false, false_or, Int.zero_add]
      exact ⟨ih.1, by omega, ih.2.2⟩
    · simp only [ite_true, true_or, iff_true]
      exact ⟨by omega, by omega, by omega⟩

theorem tAny_eq (n : Nat) (s a : Int) (h : LOOKUP (tAny n) s a) : a = if s ≠ 0 then 1 else 0 := by
  unfold LOOKUP tAny at h
  split at h
  · simp only [Option.some.injEq] at h; exact h.symm
  · cases h

theorem contains_infSign (s : Bool) : ∀ l : List Value,
    (l.filterMap Value.infSign?).contains s = l.any (Value.isInf s)
  | [] => rfl
  | x :: l => by
    have ih := contains_infSign s l
    cases x with
    | nan => simpa [Value.infSign?, Value.isInf] using ih
    | inf n => cases n <;> cases s <;> simp_all [Value.infSign?, Value.isInf]
    | fin t => simpa [Value.infSign?, Value.isInf] using ih

theorem match_signs (l : List Bool) (hno : ¬ (l.contains true = true ∧ l.contains false = true))
    (f : Bool → Value) (x : Value) :
    (match l with
      | s :: _ => f s
      | [] => x) = if l.contains true then f true else if l.contains false then f false else x := by
  cases l with
  | nil => rfl
  | cons s rest =>
    cases s
    · by_cases h : true ∈ rest
      · exact absurd ⟨by simp [h], by simp⟩ hno
      · simp [h]
    · simp

theorem filterMap_term_fin : ∀ l : List Value, l.any Value.isNaN = false → l.any (Value.isInf true) = false →
    l.any (Value.isInf false) = false → ∀ x ∈ l, ∃ t, x = .fin t
  | [], _, _, _ => by simp
  | x :: l, h1, h2, h3 => by
    simp only [List.any_cons, Bool.or_eq_false_iff] at h1 h2 h3
    have ih := filterMap_term_fin l h1.2 h2.2 h3.2
    intro y hy
    rcases List.mem_cons.mp hy with rfl | hy
    · cases y with
      | nan => simp [Value.isNaN] at h1
      | inf n => cases n <;> simp [Value.isInf] at h2 h3
      | fin t => exact ⟨t, rfl⟩
    · exact ih y hy

/-! ## The accumulator's row -/

theorem stateOf_facts (v : Value) (hv : Admissible v) :
    let st := stateOf (-133) v
    st.nan = ind Value.isNaN v ∧ st.inf * (1 - st.neg) = ind (Value.isInf false) v ∧
      st.inf * st.neg = ind (Value.isInf true) v ∧ 0 ≤ st.M ∧ -133 ≤ st.e ∧ st.e ≤ 127 ∧
      (st.neg = 0 ∨ st.neg = 1) ∧ (∀ t, v = .fin t → TermCorr (-133) t ⟨st.neg, st.e, st.M * 2 ^ HP.rho⟩) ∧
      (v.term? = none → st.M = 0) := by
  cases v with
  | nan => simp [stateOf, ind, Value.isNaN, Value.isInf, Value.term?]
  | inf n => cases n <;> simp [stateOf, ind, Value.isNaN, Value.isInf, Value.term?]
  | fin t =>
    simp only [Admissible] at hv
    obtain ⟨hlt, hnz⟩ := hv
    by_cases h0 : t.magnitude = 0
    · simp only [stateOf, h0, ite_true, ind, Value.isNaN, Value.isInf, Value.term?]
      refine ⟨rfl, by simp, by simp, by simp, by simp, by simp, by simp, ?_, by simp⟩
      intro t' ht'
      cases ht'
      exact ⟨by simp [h0, HP], by omega, fun h => absurd h0 h, fun h => absurd h0 h, fun _ => rfl, Or.inl rfl⟩
    · obtain ⟨he1, he2, _⟩ := hnz h0
      simp only [stateOf, h0, ite_false, ind, Value.isNaN, Value.isInf, Value.term?]
      have hn : ((if t.negative then 1 else 0 : Int) = 0 ∨ (if t.negative then 1 else 0 : Int) = 1) := by
        split <;> simp
      refine ⟨rfl, by simp, by simp, by omega, by omega, he2, hn, ?_, by simp⟩
      intro t' ht'
      cases ht'
      refine ⟨by simp [HP], by omega, fun _ => rfl, fun _ => ?_, fun h => absurd h h0, hn⟩
      split <;> simp_all

/-! ## The product rows -/

/-- A product row as `_terms` lists it. -/
def prodIn (q : ProdCols) : TermIn := ⟨q.sp, q.em, q.P * 2 ^ (9 + HP.rho)⟩

theorem prodOK_mem (ab : List (Nat × Nat)) (hw : ∀ x ∈ ab, x.1 < 2 ^ 16 ∧ x.2 < 2 ^ 16) :
    ∀ x ∈ ab, ProdOK (-133) (bf16ProductTotal x.1 x.2) (prodCols (-133) x.1 x.2) :=
  fun x hx => prodOK (-133) x.1 x.2 (hw x hx).1 (hw x hx).2

theorem prods_sum (ab : List (Nat × Nat)) (hok : ∀ x ∈ ab, ProdOK (-133) (bf16ProductTotal x.1 x.2) (prodCols (-133) x.1 x.2)) :
    let prods := ab.map (fun x => prodCols (-133) x.1 x.2)
    let pvs := ab.map (fun x => bf16ProductTotal x.1 x.2)
    (prods.map (·.nan)).sum = (pvs.map (ind Value.isNaN)).sum ∧
      (prods.map (·.pinf)).sum = (pvs.map (ind (Value.isInf false))).sum ∧
      (prods.map (·.ninf)).sum = (pvs.map (ind (Value.isInf true))).sum := by
  simp only [List.map_map]
  refine ⟨?_, ?_, ?_⟩ <;> congr 1 <;> apply List.map_congr_left <;> intro x hx <;>
    simp only [Function.comp, ind] <;> have := hok x hx
  · exact this.nan
  · exact this.pinf
  · exact this.ninf

theorem prods_corr : ∀ (ab : List (Nat × Nat)),
    (∀ x ∈ ab, ProdOK (-133) (bf16ProductTotal x.1 x.2) (prodCols (-133) x.1 x.2)) →
    (∀ x ∈ ab, ∃ t, bf16ProductTotal x.1 x.2 = .fin t) →
    Corr (-133) ((ab.map (fun x => bf16ProductTotal x.1 x.2)).filterMap Value.term?)
      ((ab.map (fun x => prodCols (-133) x.1 x.2)).map prodIn)
  | [], _, _ => trivial
  | x :: ab, hok, hfin => by
    obtain ⟨t, ht⟩ := hfin x (List.mem_cons_self ..)
    have ih := prods_corr ab (fun y hy => hok y (List.mem_cons_of_mem _ hy))
      (fun y hy => hfin y (List.mem_cons_of_mem _ hy))
    simp only [List.map_cons, List.filterMap_cons, ht, Value.term?]
    refine ⟨?_, ih⟩
    have hq := hok x (List.mem_cons_self ..)
    have tk := hq.fin t ht
    have hP := hq.P_lt
    have hP0 := hq.P_nonneg
    refine ⟨?_, ?_, tk.exp, tk.sign, tk.zero, tk.neg01⟩
    · simp only [prodIn, HP]; push_cast; rw [tk.mag]; omega
    · have := tk.mag; omega

theorem mag24_normal (T E : Int) (hT0 : T ≠ 0) (hT : T.natAbs < 2 ^ 32) (hn : ¬ eOut T E < -126) :
    2 ^ 23 ≤ mag24Of T E := by
  obtain ⟨_, hlo, _⟩ := window T.natAbs (by omega) hT
  unfold mag24Of magWOf
  have : (min (eFin T E - eOut T E + 2) 36).toNat = 2 := by unfold eFin; omega
  rw [this]
  exact (Nat.le_div_iff_mul_le (by decide)).mpr (by omega)

theorem mag24_zero (E : Int) : mag24Of 0 E = 0 := by simp [mag24Of, magWOf]

theorem tab_fact (ab : List (Nat × Nat)) (a0 : TermIn) (j : Nat) (hj : j < ab.length)
    (hP : 0 ≤ (prodCols (-133) ab[j].1 ab[j].2).P)
    (h : j + 1 < (a0 :: (ab.map (fun x => prodCols (-133) x.1 x.2)).map prodIn).length) :
    (a0 :: (ab.map (fun x => prodCols (-133) x.1 x.2)).map prodIn)[j + 1].scaled =
        (if j + 1 = 0 then 0 else (ab.map (fun x => prodCols (-133) x.1 x.2))[j + 1 - 1]!.P) * 2 ^ 11 ∧
      0 ≤ (if j + 1 = 0 then 0 else (ab.map (fun x => prodCols (-133) x.1 x.2))[j + 1 - 1]!.P) := by
  rw [if_neg (Nat.succ_ne_zero j), Nat.add_sub_cancel]
  have e : (ab.map (fun x => prodCols (-133) x.1 x.2))[j]!.P = (prodCols (-133) ab[j].1 ab[j].2).P := by
    simp [hj]
  rw [e]
  refine ⟨?_, hP⟩
  simp only [List.getElem_cons_succ, List.getElem_map, prodIn]
  rfl

theorem nan_case (Pf Ng O zt zo nt ef m : Int) : outPoly (-133) 1 Pf Ng O zt zo nt ef m = stateOf (-133) .nan := by
  simp [outPoly, stateOf]

theorem both_case (O zt zo nt ef m : Int) : outPoly (-133) 0 1 1 O zt zo nt ef m = stateOf (-133) .nan := by
  simp [outPoly, stateOf]

theorem inf_case (s : Bool) (O zt zo nt ef m : Int) :
    outPoly (-133) 0 (if s then 0 else 1) (if s then 1 else 0) O zt zo nt ef m = stateOf (-133) (.inf s) := by
  cases s <;> simp [outPoly, stateOf]

theorem fin_case (T E O zt zo nt ef m : Int) (hT : T.natAbs < 2 ^ 32)
    (hO : O = if eFin T E > 127 then 1 else 0) (hzt : zt = if T = 0 then 1 else 0)
    (hzo : zo = if mag24Of T E = 0 then 1 else 0) (hnt : T ≠ 0 → nt = if T < 0 then 1 else 0)
    (hnt01 : nt = 0 ∨ nt = 1) (hef : ef = eFin T E) (hm : m = (mag24Of T E : Int)) :
    outPoly (-133) 0 0 0 O zt zo nt ef m =
        stateOf (-133) (if (finResult T E).magnitude ≠ 0 ∧ (finResult T E).exponent > FP32_MAX_EXPONENT
          then .inf (finResult T E).negative else .fin (finResult T E)) ∧
      Admissible (if (finResult T E).magnitude ≠ 0 ∧ (finResult T E).exponent > FP32_MAX_EXPONENT
          then .inf (finResult T E).negative else .fin (finResult T E)) := by
  subst hO hzt hzo hef hm
  by_cases hT0 : T = 0
  · subst hT0
    have hz : mag24Of 0 E = 0 := mag24_zero E
    simp only [finResult, ite_true, ZERO, ne_eq, not_true_eq_false, false_and, ite_false, hz, Int.natCast_zero]
    refine ⟨?_, by simp [Admissible]⟩
    simp only [outPoly, stateOf, ite_true, State.mk.injEq]
    omega
  · have hnt' := hnt hT0
    subst hnt'
    by_cases hm0 : mag24Of T E = 0
    · have hov : ¬ eFin T E > 127 := by
        intro hov
        have := mag24_normal T E hT0 hT (by unfold eFin at hov; omega)
        omega
      simp only [finResult, hT0, hm0, ite_false, ite_true, ZERO, ne_eq, not_true_eq_false, false_and, hov,
        Int.natCast_zero]
      refine ⟨?_, by simp [Admissible]⟩
      simp only [outPoly, stateOf, ite_true, State.mk.injEq]
      omega
    · by_cases hov : eFin T E > 127
      · simp only [finResult, hT0, hm0, ite_false, ne_eq, not_false_eq_true, true_and, FP32_MAX_EXPONENT, hov,
          ite_true]
        refine ⟨?_, trivial⟩
        simp only [outPoly, stateOf, State.mk.injEq]
        by_cases hneg : T < 0 <;> simp [hneg]
      · have hadm := finResult_admissible T E hT hov
        simp only [finResult, hT0, hm0, ite_false] at hadm
        simp only [finResult, hT0, hm0, ite_false, ne_eq, not_false_eq_true, true_and, FP32_MAX_EXPONENT, hov]
        refine ⟨?_, hadm⟩
        simp only [outPoly, stateOf, hm0, ite_false, State.mk.injEq]
        by_cases hneg : T < 0 <;> simp [hneg] <;> omega

/-- One group of the Hopper step: the relation's output state is the canonical state of `group_step_total`'s value,
and that value is again an admissible accumulator. -/
theorem group_sound (align : Align) (v : Value) (hv : Admissible v) (ab : List (Nat × Nat)) (hlen : ab.length = 16)
    (hw : ∀ x ∈ ab, x.1 < 2 ^ 16 ∧ x.2 < 2 ^ 16) (g : GroupCols) (hg : g.terms.length = 17)
    (h : groupHolds HP align 16 (stateOf (-133) v) (ab.map (fun x => prodCols (-133) x.1 x.2)) g) :
    g.out = stateOf (-133) (groupStepTotal v (ab.map (fun x => bf16ProductTotal x.1 x.2)) 26 (-133)) ∧
      Admissible (groupStepTotal v (ab.map (fun x => bf16ProductTotal x.1 x.2)) 26 (-133)) := by
  have hok := prodOK_mem ab hw
  obtain ⟨fnan, fpinf, fninf, fM, fe1, fe2, fneg, fcorr, fspec⟩ := stateOf_facts v hv
  generalize hst : stateOf (-133) v = st at *
  have hplen : (ab.map (fun x => prodCols (-133) x.1 x.2)).length = 16 := by rw [List.length_map, hlen]
  have hins : terms HP st (ab.map (fun x => prodCols (-133) x.1 x.2)) =
      ⟨st.neg, st.e, st.M * 2 ^ HP.rho⟩ :: (ab.map (fun x => prodCols (-133) x.1 x.2)).map prodIn := rfl
  unfold groupHolds at h
  simp only [] at h
  rw [hins] at h
  generalize hI : (⟨st.neg, st.e, st.M * 2 ^ HP.rho⟩ :: (ab.map (fun x => prodCols (-133) x.1 x.2)).map prodIn :
    List TermIn) = ins at h
  have hilen : ins.length = 17 := by rw [← hI]; simp [hlen]
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22,
    h23, h24, h25, h26, h27, h28, h29, h30, h31, h32, h33, h34, h35, h36, h37, h38, h39, h40, h41, h42, h43, h44⟩ := h
  have hpj : ∀ j (hj : j < (ab.map (fun x => prodCols (-133) x.1 x.2)).length),
      0 ≤ (ab.map (fun x => prodCols (-133) x.1 x.2))[j].P := by
    intro j hj
    simp only [List.getElem_map]
    exact (hok _ (List.getElem_mem _)).P_nonneg
  -- each term: its alignment
  have hterm : ∀ i (hi : i < ins.length) (ht : i < g.terms.length),
      0 ≤ g.Estar - ins[i].e ∧ g.Estar - ins[i].e < 512 ∧
        g.terms[i].aligned - 2 * g.terms[i].signed = contrib g.Estar ins[i] := by
    intro i hi ht
    apply term_facts align g.Estar i _ _ _ (h1 i hi ht)
    · subst hI
      cases i with
      | zero => simp only [List.getElem_cons_zero]; exact Int.mul_nonneg fM (by decide)
      | succ j =>
        simp only [List.getElem_cons_succ, List.getElem_map, prodIn]
        have := hpj j (by simp at hi; omega)
        simp only [List.getElem_map] at this
        exact Int.mul_nonneg this (by decide)
    · intro hi0
      subst hI
      cases i with
      | zero => exact absurd rfl hi0
      | succ j =>
        have hj : j < ab.length := by simp at hi; omega
        have := hpj j (by simpa using hj)
        simp only [List.getElem_map] at this
        exact tab_fact ab _ j hj this hi
  have hge : ∀ x ∈ ins, x.e ≤ g.Estar := by
    intro x hx
    obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.mp hx
    have := (hterm i hi (by omega)).1
    omega
  have hZ : ∃ x ∈ ins, -133 ≤ x.e := ⟨⟨st.neg, st.e, st.M * 2 ^ HP.rho⟩, by rw [← hI]; exact List.mem_cons_self .., fe1⟩
  have hE : g.Estar = maxE (-133) ins := estar_eq _ _ ins hge hZ h3
  have htot : g.total = (ins.map (contrib g.Estar)).sum := by
    unfold LIN at h4
    have : g.terms.map (fun t => t.aligned - 2 * t.signed) = ins.map (contrib g.Estar) := by
      apply List.ext_getElem (by simp [hg, hilen])
      intro i h1' h2'
      simp only [List.getElem_map]
      exact (hterm i (by simpa using h2') (by simpa using h1')).2.2
    rw [← this]; omega
  have hfp := finite_path g h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17 h18 h19
  have hout := out_poly HP.Z g h28 h30 h31 h33 h35 h36 h37 h38 h40 h41 h42 h43 h44
  -- the special-value flags are the semantics'
  obtain ⟨eN, eP, eNg⟩ := prods_sum ab hok
  unfold LIN at h22 h23 h24
  unfold MUL at h20 h21
  have hflag : ∀ (p : Value → Bool) (S any0 x0 : Int) (hS : S - x0 - ((ab.map (fun x => bf16ProductTotal x.1 x.2)).map (ind p)).sum = 0)
      (hx0 : x0 = ind p v) (hA : LOOKUP (tAny (16 + 1)) S any0),
      any0 = if (v :: ab.map (fun x => bf16ProductTotal x.1 x.2)).any p then 1 else 0 := by
    intro p S any0 x0 hS hx0 hA
    have hS' : S = ((v :: ab.map (fun x => bf16ProductTotal x.1 x.2)).map (ind p)).sum := by
      simp only [List.map_cons, List.sum_cons]; omega
    rw [tAny_eq _ _ _ hA, hS']
    have := (ind_sum p (v :: ab.map (fun x => bf16ProductTotal x.1 x.2))).2.2
    by_cases hp : (v :: ab.map (fun x => bf16ProductTotal x.1 x.2)).any p = true
    · simp only [hp, ite_true]; rw [if_pos (this.mpr hp)]
    · simp only [hp, ite_false, Bool.false_eq_true]; rw [if_neg (fun h => hp (this.mp h))]
  have han := hflag Value.isNaN _ _ _ (by rw [← eN]; exact h22) fnan h25
  have hap := hflag (Value.isInf false) _ _ _ (by rw [← eP]; exact h23) (by rw [← h20]; exact fpinf) h26
  have hang := hflag (Value.isInf true) _ _ _ (by rw [← eNg]; exact h24) (by rw [← h21]; exact fninf) h27
  rw [hout, han, hap, hang]
  -- the semantics, by cases
  unfold groupStepTotal
  simp only []
  rw [contains_infSign, contains_infSign]
  generalize hL : (v :: ab.map (fun x => bf16ProductTotal x.1 x.2)) = L at *
  by_cases bN : L.any Value.isNaN = true
  · simp only [bN, ite_true]
    exact ⟨nan_case _ _ _ _ _ _ _ _, trivial⟩
  · simp only [bN, ite_false, Bool.false_eq_true]
    by_cases bboth : L.any (Value.isInf true) = true ∧ L.any (Value.isInf false) = true
    · rw [if_pos bboth, if_pos bboth.1, if_pos bboth.2]
      exact ⟨both_case _ _ _ _ _ _, trivial⟩
    · rw [if_neg bboth]
      rcases hsg : L.filterMap Value.infSign? with _ | ⟨sg, tail⟩
      · have bNg : ¬ L.any (Value.isInf true) = true := by rw [← contains_infSign, hsg]; simp
        have bP : ¬ L.any (Value.isInf false) = true := by rw [← contains_infSign, hsg]; simp
        simp only [bNg, bP, ite_false, Bool.false_eq_true]
        -- every term is finite: the finite path is `group_sum`
        have hall := filterMap_term_fin L (by simpa using bN) (by simpa using bNg) (by simpa using bP)
        subst hL
        obtain ⟨t0, rfl⟩ := hall v (List.mem_cons_self ..)
        have hpf : ∀ x ∈ ab, ∃ t, bf16ProductTotal x.1 x.2 = .fin t := fun x hx =>
          hall _ (List.mem_cons_of_mem _ (List.mem_map.mpr ⟨x, hx, rfl⟩))
        have hc : Corr (-133) ((Value.fin t0 :: ab.map (fun x => bf16ProductTotal x.1 x.2)).filterMap Value.term?)
            ins := by
          rw [← hI]
          simp only [List.filterMap_cons, Value.term?]
          exact ⟨fcorr t0 rfl, prods_corr ab hok hpf⟩
        have hT : ((ins.map (contrib (maxE (-133) ins))).sum).natAbs < 2 ^ 32 := by
          rw [← hE, ← htot]; exact hfp.T_lt
        rw [groupSum_eq _ ins hc hT, ← hE, ← htot]
        exact fin_case g.total g.Estar _ _ _ _ _ _ hfp.T_lt hfp.ovf hfp.z_t hfp.z_out hfp.neg_t hfp.neg01
          hfp.e_fin hfp.mag24
      · have hs : L.any (Value.isInf sg) = true := by rw [← contains_infSign, hsg]; simp
        cases sg
        · have bNg : ¬ L.any (Value.isInf true) = true := fun h => bboth ⟨h, hs⟩
          simp only [hs, bNg, ite_true, ite_false, Bool.false_eq_true]
          exact ⟨inf_case false _ _ _ _ _ _, trivial⟩
        · have bP : ¬ L.any (Value.isInf false) = true := fun h => bboth ⟨hs, h⟩
          simp only [hs, bP, ite_true, ite_false, Bool.false_eq_true]
          exact ⟨inf_case true _ _ _ _ _ _, trivial⟩

end Verity.TC.Proofs
