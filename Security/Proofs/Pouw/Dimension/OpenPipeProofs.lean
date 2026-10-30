import Definitions.Pouw.Assumptions.Dimension.OpenPipe
import Proofs.Pouw.Dimension.ForestProofs

/-!
# The open quadratic pipe: proofs

* **The index form** (`openOK_get`): instruction `i` of an open pipe is priced, and it is big or an FP32 step that is
  a `QuadStep` over the values of the instructions before it (`vals`).
* **The count** (`theorem1Open`). On Q-parts, the FP32 steps are `forestBound`'s merge-free steps `F` and the big
  values are the others. The family's words are slots, not inputs (an input is free, so its Q-part is `0`), so they
  are values; independence gives `|U|·nonFinal ≤ dim`. The cost splits over `F` and its complement:
  `pb·(N − |F|) + cs·|F| ≤ cost`, and with `μ = min(pb, 2·cs)`, `2·μ·dim ≤ μ·(2·(N − |F|) + |F|) ≤ 2·cost`.
* **The closed pipe is open** (`quadToOpen`): a closed pipe's FP32 step reads earlier FP32 outputs only, a sublist
  of all earlier priced values (`quadStep_mono`).
-/

namespace Pouw.Dimension.OpenPipe

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension Pouw.Dimension.Assumptions Pouw.Dimension.Forest

variable {N m n k : ℕ}

/-- A priced instruction's value (`0` for a free one, which an open pipe has none of). -/
def val {G : Type} : Instr G → (G → ℚ)
  | .free _ _ => 0
  | .priced _ f => f

/-- The values of a program's instructions, in order. -/
def vals {G : Type} (P : List (Instr G)) : List (G → ℚ) := P.map val

theorem openOK_get (pb cs : ℕ) :
    ∀ (P : List (Instr (Noise N m n k))) (Vs : List (Noise N m n k → ℚ)), OpenOK pb cs Vs P →
      ∀ i (h : i < P.length), ∃ p f, P[i] = Instr.priced p f ∧
        (pb ≤ p ∨ (cs ≤ p ∧ QuadStep (Vs ++ (vals P).take i) f)) := by
  intro P
  induction P with
  | nil => intro Vs _ i h; simp at h
  | cons ins P ih =>
    intro Vs hP i hi
    cases ins with
    | free c w => exact hP.elim
    | priced p f =>
      obtain ⟨h1, h2⟩ := hP
      cases i with
      | zero => exact ⟨p, f, rfl, by simpa using h1⟩
      | succ i =>
        obtain ⟨p', f', hget, h⟩ := ih (Vs ++ [f]) h2 i (by simpa using hi)
        refine ⟨p', f', by simpa using hget, ?_⟩
        simpa [vals, val] using h

theorem openOK_nofree (pb cs : ℕ) :
    ∀ (P : List (Instr (Noise N m n k))) (Vs : List (Noise N m n k → ℚ)), OpenOK pb cs Vs P →
      ∀ c w, Instr.free c w ∉ P := by
  intro P
  induction P with
  | nil => intro _ _ c w h; simp at h
  | cons ins P ih =>
    intro Vs hP c w h
    cases ins with
    | free c' w' => exact hP.elim
    | priced p f =>
      rcases List.mem_cons.1 h with h | h
      · cases h
      · exact ih _ hP.2 c w h

theorem mem_slots_vals {G : Type} (P : List (Instr G)) :
    ∀ L : List (G → ℚ), (∀ c w, Instr.free c w ∉ P) → ∀ f ∈ slots P L, f ∈ L ∨ f ∈ vals P := by
  induction P with
  | nil => intro L _ f hf; exact Or.inl hf
  | cons ins P ih =>
    intro L hP f hf
    rw [slots_cons] at hf
    have hP' : ∀ c w, Instr.free c w ∉ P := fun c w h => hP c w (List.mem_cons_of_mem _ h)
    cases ins with
    | free c w => exact (hP c w (by simp)).elim
    | priced p g =>
      rcases ih _ hP' f hf with h | h
      · rcases List.mem_append.1 h with h | h
        · exact Or.inl h
        · have hg : f = g := by simpa [Instr.out] using h
          exact Or.inr (hg ▸ List.mem_cons_self (a := val (Instr.priced p g)) (l := vals P))
      · exact Or.inr (List.mem_cons_of_mem (val (Instr.priced p g)) h)

theorem cost_eq_sum {G : Type} :
    ∀ P : List (Instr G), cost P = ∑ i ∈ range P.length, (P.getD i (.free 0 0)).price := by
  intro P
  induction P with
  | nil => simp [cost]
  | cons ins P ih =>
    rw [List.length_cons, Finset.sum_range_succ']
    simp only [List.getD_cons_succ, List.getD_cons_zero]
    rw [← ih]
    simp [cost, add_comm]

theorem pipeOpnd_mono {Rs Rs' : List (Noise N m n k → ℚ)} (h : ∀ r ∈ Rs, r ∈ Rs') {x : Noise N m n k → ℚ}
    (hx : PipeOpnd Rs x) : PipeOpnd Rs' x := by
  rcases hx with hx | ⟨r, hr, β, rfl⟩
  · exact Or.inl hx
  · exact Or.inr ⟨r, h r hr, β, rfl⟩

theorem quadStep_mono {Rs Rs' : List (Noise N m n k → ℚ)} (h : ∀ r ∈ Rs, r ∈ Rs') {f : Noise N m n k → ℚ}
    (hf : QuadStep Rs f) : QuadStep Rs' f := by
  obtain ⟨a, b, c, hc, rfl, hab⟩ := hf
  refine ⟨a, b, c, pipeOpnd_mono h hc, rfl, ?_⟩
  rcases hab with hab | ⟨hα, hb⟩
  · exact Or.inl hab
  · exact Or.inr ⟨hα, pipeOpnd_mono h hb⟩

theorem quad_open (cs : ℕ) :
    ∀ (P : List (Instr (Noise N m n k))) (Vs Rs : List (Noise N m n k → ℚ)), (∀ r ∈ Rs, r ∈ Vs) →
      (∀ c w, Instr.free c w ∉ P) → (∀ p f, Instr.priced p f ∈ P → 16 ≤ p ∨ cs ≤ p) →
      (∀ i (h : i < (prodOuts P).length), QuadStep (Rs ++ (prodOuts P).take i) ((prodOuts P)[i]'h)) →
      OpenOK 16 cs Vs P := by
  intro P
  induction P with
  | nil => intros; trivial
  | cons ins P ih =>
    intro Vs Rs hRs hfree hpr hq
    have hfree' : ∀ c w, Instr.free c w ∉ P := fun c w h => hfree c w (List.mem_cons_of_mem _ h)
    have hpr' : ∀ p f, Instr.priced p f ∈ P → 16 ≤ p ∨ cs ≤ p := fun p f h => hpr p f (List.mem_cons_of_mem _ h)
    cases ins with
    | free c w => exact hfree c w (by simp)
    | priced p f =>
      by_cases h16 : 16 ≤ p
      · have hpo : prodOuts (Instr.priced p f :: P) = prodOuts P := by simp [prodOuts, h16]
        refine ⟨Or.inl h16, ih (Vs ++ [f]) Rs (fun r hr => List.mem_append_left _ (hRs r hr)) hfree' hpr' ?_⟩
        intro i hi
        have := hq i (by rw [hpo]; exact hi)
        simpa [hpo] using this
      · have hpo : prodOuts (Instr.priced p f :: P) = f :: prodOuts P := by simp [prodOuts, h16]
        have hcs : cs ≤ p := (hpr p f (by simp)).resolve_left h16
        have h0 := hq 0 (by simp [hpo])
        simp only [hpo, List.take_zero, List.append_nil, List.getElem_cons_zero] at h0
        refine ⟨Or.inr ⟨hcs, quadStep_mono hRs h0⟩, ih (Vs ++ [f]) (Rs ++ [f]) ?_ hfree' hpr' ?_⟩
        · intro r hr
          rcases List.mem_append.1 hr with hr | hr
          · exact List.mem_append_left _ (hRs r hr)
          · exact List.mem_append_right _ hr
        · intro i hi
          have := hq (i + 1) (by simp [hpo]; exact hi)
          simpa [hpo] using this

theorem quadToOpen : QuadToOpen := by
  intro N m n k cs P hP
  exact quad_open cs P [] [] (fun _ h => h) hP.1.1 hP.1.2.1 (fun i h => by simpa using hP.2 i h)

/-- **Theorem 1 in the open quadratic pipe, for any word family.** -/
theorem theorem1Open : Theorem1Open := by
  intro N m n k w U h2 hli pb cs L P hL hP S hS hout
  classical
  have hlen : (vals P).length = P.length := by simp [vals]
  let y : ℕ → QPart N m n k := fun j => Submodule.Quotient.mk ((vals P).getD j 0)
  let pr : ℕ → ℕ := fun i => (P.getD i (.free 0 0)).price
  let F := (range P.length).filter (fun i => pr i < pb)
  let G := (range P.length).filter (fun i => ¬ pr i < pb)
  have hget : ∀ i (h : i < P.length), ∃ p f, P[i] = Instr.priced p f ∧ (vals P).getD i 0 = f ∧ pr i = p ∧
      (pb ≤ p ∨ (cs ≤ p ∧ QuadStep ((vals P).take i) f)) := by
    intro i h
    obtain ⟨p, f, hPi, hq⟩ := openOK_get pb cs P [] hP i h
    refine ⟨p, f, hPi, ?_, ?_, by simpa using hq⟩
    · rw [getD_of_lt (by rw [hlen]; exact h)]
      simp [vals, hPi, val]
    · show (P.getD i (.free 0 0)).price = p
      rw [getD_of_lt h, hPi]
      rfl
  have hF : ∀ i ∈ F, FStep (atoms N m n k) y i := by
    intro i hi
    obtain ⟨hi, hlt⟩ := mem_filter.1 hi
    obtain ⟨p, f, -, hyf, hpr, hq⟩ := hget i (mem_range.1 hi)
    have hq : QuadStep ((vals P).take i) f := (hq.resolve_left (by omega)).2
    obtain ⟨a, b, c, hc, hf, hab⟩ := hq
    have hyi : y i = Submodule.Quotient.mk (a * b + c) := by simp only [y, hyf, hf]
    rcases hab with ⟨ha, hb⟩ | ⟨⟨α, rfl⟩, hb⟩
    · exact Or.inl ⟨_, ⟨a, ha, b, hb, rfl⟩, _, opnd_of_pipeOpnd hc, by rw [hyi, Submodule.Quotient.mk_add]⟩
    · refine Or.inr ⟨α, _, _, opnd_of_pipeOpnd hb, opnd_of_pipeOpnd hc, ?_⟩
      rw [hyi, Submodule.Quotient.mk_add, ← Submodule.Quotient.mk_smul]
      congr 2
  have hB := forestBound (QPart N m n k) (familyQ w U) (atoms N m n k) atoms_smul h2 y P.length F
    (filter_subset _ _) hF
  have hFG : F.card + G.card = P.length := by
    rw [card_filter_add_card_filter_not, card_range]
  let wS : ↥S → QPart N m n k := fun x => Submodule.Quotient.mk (w x.1.1 x.1.2)
  have hliS : LinearIndependent ℚ wS :=
    hli.comp (fun x : ↥S => (⟨x.1, hS x.2⟩ : ↥(U ×ˢ range (nonFinal k m n))))
      (fun x x' h => Subtype.ext (Subtype.mk.inj h))
  have hle : Submodule.span ℚ (Set.range wS) ≤
      Submodule.span ℚ (y '' {i | i < P.length ∧ y i ∈ familyQ w U}) := by
    refine Submodule.span_le.2 ?_
    rintro _ ⟨x, rfl⟩
    rcases mem_slots_vals P L (openOK_nofree pb cs P [] hP) _ (hout x.1 x.2) with hin | hin
    · exact absurd ((Submodule.Quotient.mk_eq_zero _).2 (hL _ hin)) (hliS.ne_zero x)
    · obtain ⟨i, hi, hix⟩ := List.getElem_of_mem hin
      have hyi : y i = wS x := by simp only [y, wS, getD_of_lt hi, hix]
      refine Submodule.subset_span ⟨i, ⟨by rw [← hlen]; exact hi, ?_⟩, hyi⟩
      rw [hyi]
      exact Submodule.subset_span ⟨⟨x.1, hS x.2⟩, rfl⟩
  have hd := Submodule.finrank_mono hle
  rw [finrank_span_eq_card hliS, Fintype.card_coe] at hd
  have hcost : pb * G.card + cs * F.card ≤ cost P := by
    rw [cost_eq_sum, ← sum_filter_add_sum_filter_not (range P.length) (fun i => pr i < pb)]
    have h1 : F.card • cs ≤ ∑ i ∈ F, pr i := by
      refine card_nsmul_le_sum _ _ _ fun i hi => ?_
      obtain ⟨hi, hlt⟩ := mem_filter.1 hi
      obtain ⟨p, f, -, -, hpr, hq⟩ := hget i (mem_range.1 hi)
      exact hpr ▸ (hq.resolve_left (by omega)).1
    have h2 : G.card • pb ≤ ∑ i ∈ G, pr i := by
      refine card_nsmul_le_sum _ _ _ fun i hi => ?_
      obtain ⟨-, hlt⟩ := mem_filter.1 hi
      omega
    rw [smul_eq_mul] at h1 h2
    change pb * G.card + cs * F.card ≤ ∑ i ∈ F, pr i + ∑ i ∈ G, pr i
    nlinarith
  set μ := min pb (2 * cs)
  have hμ1 : μ ≤ pb := min_le_left _ _
  have hμ2 : μ ≤ 2 * cs := min_le_right _ _
  set r := Module.finrank ℚ (Submodule.span ℚ (y '' {i | i < P.length ∧ y i ∈ familyQ w U}))
  have hB' : 2 * r ≤ 2 * G.card + F.card := by omega
  have e1 : μ * S.card ≤ μ * r := Nat.mul_le_mul_left _ hd
  have e2 : μ * (2 * r) ≤ μ * (2 * G.card + F.card) := Nat.mul_le_mul_left _ hB'
  have e3 : μ * G.card ≤ pb * G.card := Nat.mul_le_mul_right _ hμ1
  have e4 : μ * F.card ≤ 2 * cs * F.card := Nat.mul_le_mul_right _ hμ2
  nlinarith

/-- **Theorem D in the open quadratic pipe.** -/
theorem algBoundOpenQuad : AlgBoundOpenQuad := by
  intro k m n N R A B U L P pb cs hind hMR hU hL hP hout
  obtain ⟨h2, hli⟩ := gridNear k m n N R A B U (word k m n N R A B) hind hMR hU
    (fun u _ p _ => by rw [sub_self]; exact Submodule.zero_mem _)
  have h := theorem1Open N m n k (word k m n N R A B) U h2 hli pb cs L P hL hP _ subset_rfl
    fun x hx => hout _ (mem_product.1 hx).1 _ (mem_range.1 (mem_product.1 hx).2)
  have hD : (U ×ˢ range (nonFinal k m n)).card = U.card * (m * n) * (3 * k / 16 - 1) := by
    rw [card_product, card_range, nonFinal, Nat.mul_comm (3 * k / 16 - 1), ← Nat.mul_assoc]
  rwa [hD] at h

theorem algBoundOpenQuad8 : AlgBoundOpenQuad8 := by
  intro k m n N R A B U L P hind hMR hU hL hP hout
  exact algBoundOpenQuad k m n N R A B U L P 16 8 hind hMR hU hL hP hout

theorem openOK_append (pb cs : ℕ) :
    ∀ (P Q : List (Instr (Noise N m n k))) (Vs : List (Noise N m n k → ℚ)), OpenOK pb cs Vs P →
      OpenOK pb cs (Vs ++ vals P) Q → OpenOK pb cs Vs (P ++ Q) := by
  intro P
  induction P with
  | nil => intro Q Vs _ hQ; simpa [vals] using hQ
  | cons ins P ih =>
    intro Q Vs hP hQ
    cases ins with
    | free c w => exact hP.elim
    | priced p f => exact ⟨hP.1, ih Q (Vs ++ [f]) hP.2 (by simpa [vals, val] using hQ)⟩

theorem openOK_big (pb cs : ℕ) :
    ∀ (P : List (Instr (Noise N m n k))) (Vs : List (Noise N m n k → ℚ)),
      (∀ ins ∈ P, ∃ p f, ins = Instr.priced p f ∧ pb ≤ p) → OpenOK pb cs Vs P := by
  intro P
  induction P with
  | nil => intros; trivial
  | cons ins P ih =>
    intro Vs h
    obtain ⟨p, f, rfl, hp⟩ := h _ (List.mem_cons_self ..)
    exact ⟨Or.inl hp, ih _ fun ins hm => h ins (List.mem_cons_of_mem _ hm)⟩

/-- **An open-pipe program that no closed pipe is.** -/
theorem openPipeWitness : OpenPipeWitness := by
  intro k hk h16 m n N A B U u₀ hU hu₀ hm hn
  have hu₀N : u₀ < N := Finset.mem_range.1 (hU hu₀)
  obtain ⟨R, -, hadm, hind, hMR, -⟩ := pipeWordWitness k hk h16 m n N A B U u₀ hU hu₀ hm hn
  let a : Noise N m n k → ℚ := fun ω => (e1Of ω u₀ 0 16 : ℚ)
  let b : Noise N m n k → ℚ := fun ω => (f1Of ω 16 0 : ℚ)
  have ha : a ∈ V0 N m n k := e1_mem_V0 hu₀N hm (by omega)
  have hb : b ∈ V0 N m n k := f1_mem_V0 (by omega) hn
  set w₀ := word k m n N R A B u₀ 0
  have hpos : 0 < nonFinal k m n := Nat.mul_pos (by omega) (Nat.mul_pos hm hn)
  have hx : (u₀, 0) ∈ U ×ˢ range (nonFinal k m n) := Finset.mem_product.2 ⟨hu₀, Finset.mem_range.2 hpos⟩
  have hw₀ : Instr.priced 16 w₀ ∈ honest k m n N R A B U 16 := mem_honest.2 ⟨u₀, hu₀, 0, hpos, rfl⟩
  have hbig : ∀ p f, Instr.priced p f ∈ honest k m n N R A B U 16 → 16 ≤ p := fun p f hm => by
    obtain ⟨u, _, q, _, heq⟩ := mem_honest.1 hm
    exact (Instr.priced.inj heq).1.le
  refine ⟨R, honest k m n N R A B U 16 ++ [Instr.priced 8 (a * b + w₀)], a, b, hadm, hind, hMR, ?_, ?_, ?_, ha, hb,
    List.mem_append_left _ hw₀, List.mem_append_right _ (List.mem_singleton_self _), ?_⟩
  · refine openOK_append 16 8 _ _ [] (openOK_big 16 8 _ _ fun ins hm => ?_) ?_
    · obtain ⟨u, _, q, _, rfl⟩ := mem_honest.1 hm
      exact ⟨16, _, rfl, le_rfl⟩
    · refine ⟨Or.inr ⟨le_rfl, a, b, w₀, Or.inr ⟨w₀, ?_, 1, (one_smul _ _).symm⟩, rfl, Or.inl ⟨ha, hb⟩⟩, trivial⟩
      exact List.mem_append_right _ (List.mem_map.2 ⟨_, hw₀, rfl⟩)
  · intro hC
    have hpo : prodOuts (honest k m n N R A B U 16 ++ [Instr.priced 8 (a * b + w₀)]) = [a * b + w₀] := by
      rw [prodOuts_append, prodOuts_eq_nil hbig, List.nil_append]
      simp [prodOuts]
    have hstep := hC.2.2 0 (by rw [hpo]; simp)
    simp only [hpo, List.take_zero, List.getElem_cons_zero] at hstep
    obtain ⟨a', b', c', ha', hb', hc', heq⟩ := hstep
    have hV : ∀ x, PipeOpnd ([] : List (Noise N m n k → ℚ)) x → x ∈ V0 N m n k := fun x h =>
      h.resolve_right (by simp)
    have hsum : (Submodule.Quotient.mk (a' * b') : QPart N m n k) + Submodule.Quotient.mk ((-a) * b) =
        Submodule.Quotient.mk w₀ := by
      have h1 : (Submodule.Quotient.mk (a * b + w₀) : QPart N m n k) = Submodule.Quotient.mk (a' * b' + c') :=
        congrArg _ heq
      rw [Submodule.Quotient.mk_add, Submodule.Quotient.mk_add, (Submodule.Quotient.mk_eq_zero _).2 (hV _ hc'),
        add_zero] at h1
      rw [neg_mul, Submodule.Quotient.mk_neg, ← h1]
      abel
    have h0 := gridTwoAtom k m n N R A B U hind (minRankGe_mono (by norm_num) hMR) hU _
      ⟨a', hV _ ha', b', hV _ hb', rfl⟩ _ ⟨-a, Submodule.neg_mem _ ha, b, hb, rfl⟩
      (by rw [hsum]; exact Submodule.subset_span ⟨⟨(u₀, 0), hx⟩, rfl⟩)
    rw [hsum] at h0
    exact pipe_word_nmem A B hind hU hx ((Submodule.Quotient.mk_eq_zero _).1 h0)
  · intro u hu p hp
    exact mem_slots_of_priced (List.mem_append_left _ (mem_honest.2 ⟨u, hu, p, hp, rfl⟩)) _
  · rw [cost_append, honest_cost]
    simp [cost, Instr.price]

end Pouw.Dimension.OpenPipe
