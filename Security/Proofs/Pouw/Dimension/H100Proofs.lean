import Definitions.Pouw.Assumptions.Dimension.H100
import Proofs.Pouw.Dimension.EmbedProofs

/-!
# `MH100` and its per-word bound: proofs

* **The translation** (`tr100`): a `wgmma` or other tensor-core instruction to one priced instruction per accumulator
  register, at `regPrice` or `tc`; `fp32`, `int`, `fp64` and `other` to one priced instruction at their price; `move`
  and `operand` to nothing. Under `Floor c` every priced instruction costs at least `c` (`GOK` with `c ≤ p`).
* **The invariant** (`tr100_spec`): every register is in `V₀` or a value of the translation, as in `tr_spec`.
* **The count** (`perWordMH100`): a target independent modulo `V₀` is not in `V₀`, so it is a slot of the
  translation; `count_le` gives one priced instruction per target, and `hc_cost` `c` per priced instruction
  (`perWordMH100Floor`; `perWordMH100` is `c = 32`).
-/

namespace Pouw.Dimension.H100

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension Pouw.Dimension.Assumptions Pouw.Dimension.Embed
  Pouw.Dimension.OpenPipe

variable {N m n k : ℕ}

/-! ## Registers and the priced model -/

theorem mem_regs100_of_mem {G : Type} (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) :
    ∀ (P : List (Op100 G)) (L : List (G → ℚ)), ∀ r ∈ L, r ∈ regs100 fpSem P L := by
  intro P
  induction P with
  | nil => intro L r hr; exact hr
  | cons o P ih => intro L r hr; exact ih _ r (List.mem_append_left _ hr)

theorem hc_cost {G : Type} {c : ℕ} {P : List (Instr G)} (h : Hc c P) : c * nPriced P ≤ cost P := by
  induction P with
  | nil => simp [nPriced, cost]
  | cons ins P ih =>
    rw [nPriced_cons, cost_cons]
    have h1 := ih fun x hx => h x (List.mem_cons_of_mem _ hx)
    have h2 := h ins (by simp)
    cases ins with
    | free c' w =>
      simp only [Instr.isPriced, Instr.price, Bool.false_eq_true, ite_false, add_zero, zero_add]
      exact h1
    | priced p f =>
      simp only [Instr.isPriced, Instr.price, ite_true, forall_const] at h2 ⊢
      rw [Nat.mul_add, Nat.mul_one]
      omega

/-- The admission rule of AlgM_c as a pipe: every instruction priced at `c` or more. -/
abbrev Cc {G : Type} (c : ℕ) : List (G → ℚ) → ℕ → (G → ℚ) → Prop := fun _ p _ => c ≤ p

theorem hc_of_gok {G : Type} {c : ℕ} {P : List (Instr G)} {Vs : List (G → ℚ)} (h : GOK (Cc c) Vs P) : Hc c P := by
  intro ins hins _
  cases ins with
  | free c w => exact absurd hins (gok_nofree _ _ _ h c w)
  | priced p f =>
    obtain ⟨_, hp⟩ := gok_mem _ _ _ h p f hins
    exact hp

theorem gok_map {G : Type} (c p : ℕ) (hp : c ≤ p) :
    ∀ (fs Ws : List (G → ℚ)), GOK (Cc c) Ws (fs.map (Instr.priced p))
  | [], _ => trivial
  | _ :: fs, _ => ⟨hp, gok_map c p hp fs _⟩

theorem vals_map {G : Type} (p : ℕ) (fs : List (G → ℚ)) : vals (fs.map (Instr.priced p)) = fs := by
  induction fs with
  | nil => rfl
  | cons f fs ih =>
    simp only [vals, List.map_cons, List.map_map] at ih ⊢
    exact congrArg (List.cons f) ih

theorem cost_map {G : Type} (p : ℕ) (fs : List (G → ℚ)) : cost (fs.map (Instr.priced p)) = p * fs.length := by
  induction fs with
  | nil => simp [cost]
  | cons f fs ih =>
    simp only [cost, List.map_cons, List.sum_cons, List.length_cons, Instr.price] at ih ⊢
    rw [ih, Nat.mul_succ]
    ring

/-! ## The translation -/

/-- One instruction's translation into the priced model. -/
def Op100.tr {G : Type} (pr : Prices100) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) :
    Op100 G → List (G → ℚ) → List (Instr G)
  | .wgmma s outs, _ => outs.map (Instr.priced (s.regPrice pr))
  | .mma outs _, _ => outs.map (Instr.priced pr.tc)
  | .fp32 κ i j l, Rs => [Instr.priced pr.ffma (fpVal fpSem κ i j l Rs)]
  | .int f, _ => [Instr.priced pr.int32 f]
  | .fp64 f, _ => [Instr.priced pr.fp64 f]
  | .other f, _ => [Instr.priced pr.other f]
  | .move _, _ => []
  | .operand _, _ => []

/-- A program's translation. -/
def tr100 {G : Type} (pr : Prices100) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) :
    List (Op100 G) → List (G → ℚ) → List (Instr G)
  | [], _ => []
  | o :: P, Rs => o.tr pr fpSem Rs ++ tr100 pr fpSem P (Rs ++ o.outs fpSem Rs)

theorem getD_invV {G : Type} {V : Submodule ℚ (G → ℚ)} {Rs Vs : List (G → ℚ)} (h : ∀ r ∈ Rs, r ∈ V ∨ r ∈ Vs)
    (i : ℕ) : Rs.getD i 0 ∈ V ∨ Rs.getD i 0 ∈ Vs := by
  rcases getD_mem_or Rs i with hi | hi
  · exact h _ hi
  · rw [hi]
    exact Or.inl (Submodule.zero_mem _)

section Spec

variable {G : Type} (pr : Prices100) (c : ℕ) (hpr : Floor c pr) (V : Submodule ℚ (G → ℚ)) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ)
include hpr

theorem op100_spec (o : Op100 G) (Rs Vs : List (G → ℚ)) (hinv : ∀ r ∈ Rs, r ∈ V ∨ r ∈ Vs) (hwf : o.WF pr V) :
    GOK (Cc c) Vs (o.tr pr fpSem Rs) ∧
      (∀ r ∈ o.outs fpSem Rs, r ∈ V ∨ r ∈ Vs ++ vals (o.tr pr fpSem Rs)) ∧
      cost (o.tr pr fpSem Rs) ≤ o.price pr := by
  obtain ⟨hs, htc, hffma, hint, hfp64, hother⟩ := hpr
  have one : ∀ (p : ℕ) (f : G → ℚ), c ≤ p →
      GOK (Cc c) Vs [Instr.priced p f] ∧
        (∀ r ∈ [f], r ∈ V ∨ r ∈ Vs ++ vals [Instr.priced p f]) ∧ cost [Instr.priced p f] ≤ p :=
    fun p f hp => ⟨⟨hp, trivial⟩, fun r hr => Or.inr (by simp_all [vals, OpenPipe.val]), by simp [cost, Instr.price]⟩
  cases o with
  | wgmma s outs =>
    refine ⟨gok_map c _ (hs s) outs Vs, fun r hr => Or.inr ?_, ?_⟩
    · simp only [Op100.tr, vals_map]
      exact List.mem_append_right _ hr
    · simp only [Op100.tr, cost_map, Op100.price, le_refl]
  | mma outs p =>
    refine ⟨gok_map c _ htc outs Vs, fun r hr => Or.inr ?_, ?_⟩
    · simp only [Op100.tr, vals_map]
      exact List.mem_append_right _ hr
    · simp only [Op100.tr, cost_map, Op100.price]
      exact hwf
  | fp32 κ i j l => exact one _ _ hffma
  | int f => exact one _ _ hint
  | fp64 f => exact one _ _ hfp64
  | other f => exact one _ _ hother
  | move i =>
    refine ⟨trivial, fun r hr => ?_, by simp [Op100.tr, cost]⟩
    simp only [Op100.outs, List.mem_singleton] at hr
    subst hr
    simpa [Op100.tr, vals] using getD_invV hinv i
  | operand f =>
    refine ⟨trivial, fun r hr => ?_, by simp [Op100.tr, cost]⟩
    simp only [Op100.outs, List.mem_singleton] at hr
    exact Or.inl (hr ▸ hwf)

theorem tr100_spec : ∀ (P : List (Op100 G)) (Rs Vs : List (G → ℚ)),
    (∀ r ∈ Rs, r ∈ V ∨ r ∈ Vs) → (∀ o ∈ P, o.WF pr V) →
    GOK (Cc c) Vs (tr100 pr fpSem P Rs) ∧
      (∀ r ∈ regs100 fpSem P Rs, r ∈ V ∨ r ∈ Vs ++ vals (tr100 pr fpSem P Rs)) ∧
      cost (tr100 pr fpSem P Rs) ≤ cost100 pr P := by
  intro P
  induction P with
  | nil =>
    intro Rs Vs hinv _
    refine ⟨trivial, fun r hr => ?_, by simp [tr100, cost]⟩
    simpa [tr100, vals] using hinv r hr
  | cons o P ih =>
    intro Rs Vs hinv hwf
    obtain ⟨h1, h2, h3⟩ := op100_spec pr c hpr V fpSem o Rs Vs hinv (hwf o List.mem_cons_self)
    have hinv' : ∀ r ∈ Rs ++ o.outs fpSem Rs, r ∈ V ∨ r ∈ Vs ++ vals (o.tr pr fpSem Rs) := by
      intro r hr
      rcases List.mem_append.1 hr with hr | hr
      · rcases hinv r hr with h | h
        · exact Or.inl h
        · exact Or.inr (List.mem_append_left _ h)
      · exact h2 r hr
    obtain ⟨i1, i2, i3⟩ := ih _ _ hinv' fun o' ho' => hwf o' (List.mem_cons_of_mem _ ho')
    refine ⟨gok_append (Cc c) _ _ _ h1 i1, fun r hr => ?_, ?_⟩
    · have := i2 r hr
      simpa [tr100, vals, List.map_append, List.append_assoc] using this
    · have e1 : cost100 pr (o :: P) = o.price pr + cost100 pr P := by simp [cost100]
      have e3 : tr100 pr fpSem (o :: P) Rs = o.tr pr fpSem Rs ++ tr100 pr fpSem P (Rs ++ o.outs fpSem Rs) := rfl
      rw [e1, e3, cost_append]
      omega

end Spec

/-! ## The pinned theorems -/

/-- The embedding at any floor `c`: into AlgM_c. -/
theorem embedFloor (c : ℕ) (pr : Prices100) (hpr : Floor c pr) (G : Type) (V : Submodule ℚ (G → ℚ))
    (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (hL : ∀ f ∈ L, f ∈ V)
    (hwf : ∀ o ∈ P, o.WF pr V) :
    ∃ P' : List (Instr G), Hc c P' ∧ (∀ r ∈ regs100 fpSem P L, r ∈ V ∨ r ∈ slots P' L) ∧ cost P' ≤ cost100 pr P := by
  obtain ⟨h1, h2, h3⟩ := tr100_spec pr c hpr V fpSem P L [] (fun r hr => Or.inl (hL r hr)) hwf
  refine ⟨tr100 pr fpSem P L, hc_of_gok h1, fun r hr => ?_, h3⟩
  rcases h2 r hr with h | h
  · exact Or.inl h
  · exact Or.inr (mem_slots_of_vals (gok_nofree (Cc c) _ _ h1) (by simpa using h) L)

theorem span1_le {G : Type} {V : Submodule ℚ (G → ℚ)} {L : List (G → ℚ)} (h1 : (1 : G → ℚ) ∈ V)
    (hL : ∀ f ∈ L, f ∈ V) : span1 L ≤ V := by
  refine Submodule.span_le.2 fun f hf => ?_
  rcases hf with rfl | hf
  · exact h1
  · exact hL f hf

theorem nmem_of_indep {G : Type} {V : Submodule ℚ (G → ℚ)} {ι : Type} [Fintype ι] {t : ι → G → ℚ}
    (hind : IndepMod V t) (i : ι) : t i ∉ V := by
  classical
  intro h
  have := hind (fun j => if j = i then 1 else 0) (by simpa [ite_smul] using h) i
  simp at this

/-- **The per-word cost in `MH100` at any floor.** -/
theorem perWordMH100Floor : PerWordMH100Floor := by
  intro c pr hpr G _ V h1 fpSem ι _ t L P hind hL hwf hout
  obtain ⟨P', hP', hreg, hcost⟩ := embedFloor c pr hpr G V fpSem L P hL hwf
  have hcount := count_le P' L t (fun a ha => hind a (span1_le h1 hL ha))
    (fun i => (hreg _ (hout i)).resolve_left (nmem_of_indep hind i))
  exact le_trans (Nat.mul_le_mul_left c hcount) (le_trans (hc_cost hP') hcost)

/-- **The per-word cost in `MH100`.** -/
theorem perWordMH100 : PerWordMH100 := fun pr hpr => perWordMH100Floor 32 pr hpr

/-! ## The honest split-K schedule -/

/-- The registers `splitK` appends: `R 0`, then `I τ` and `R (τ + 1)` for each step. -/
def hist {G : Type} (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (κ : ℕ) (I : ℕ → G → ℚ) : ℕ → List (G → ℚ)
  | 0 => [0]
  | τ + 1 => hist fpSem κ I τ ++ [I τ, runWords fpSem κ I (τ + 1)]

theorem mem_hist {G : Type} (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (κ : ℕ) (I : ℕ → G → ℚ) :
    ∀ T, ∀ τ < T, I τ ∈ hist fpSem κ I T ∧ runWords fpSem κ I (τ + 1) ∈ hist fpSem κ I T
  | 0 => fun τ h => absurd h (Nat.not_lt_zero τ)
  | T + 1 => fun τ h => by
    rw [hist]
    rcases Nat.lt_succ_iff_lt_or_eq.1 h with h | rfl
    · obtain ⟨h1, h2⟩ := mem_hist fpSem κ I T τ h
      exact ⟨List.mem_append_left _ h1, List.mem_append_left _ h2⟩
    · exact ⟨List.mem_append_right _ (by simp), List.mem_append_right _ (by simp)⟩

end Pouw.Dimension.H100
