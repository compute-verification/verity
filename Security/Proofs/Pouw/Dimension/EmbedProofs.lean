import Definitions.Pouw.Assumptions.Dimension.Embed
import Proofs.Pouw.Dimension.OpenPipeProofs

/-!
# M4090 and the embedding theorem: proofs

* **One translation, two targets.** `tr fpSem hi` maps an `M4090` program instruction by instruction: `mma outs p`
  to `len(outs)` instructions at 16, a quadratic `fp32` to one at 8 and any other `fp32` to one at `hi`, `int`, `dd`
  and `subword` to one at 16, and `move`, `presalt` and `operand` to nothing. `GOK C` is a pipe whose priced
  instructions satisfy `C`; `OpenOK 16 8` and `FpOK fpSem` are instances (`openOK_iff`, `fpOK_iff`).
* **The invariant** (`tr_spec`): every register is in `V₀` or an earlier value of the translation. A `QuadStep` over
  the registers is then one over the translation's values (`quadStep_transfer`: a multiple of a `V₀` value is in `V₀`),
  and an `fp32` value is an `FpStep` over them (`fpVal_step`).
* **The count** (`perWordM4090`): a checked word is not in `V₀`, so it is a slot of the translation, and `theorem1Open`
  pays 16 per word.
* **The range gap** (`a1fpOnRangeGap`): a checked word that takes a value `fpSem` never outputs
  cannot be an FP32 step, so each is a distinct instruction at 16 (`count16`); `perWordM4090Gap` composes it with
  `embedFp`.
-/

namespace Pouw.Dimension.Embed

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension Pouw.Dimension.Assumptions Pouw.Dimension.OpenPipe
  Pouw.Dimension.Forest

variable {N m n k : ℕ}

/-! ## Registers -/

theorem getD_mem_or {G : Type} (Rs : List (G → ℚ)) (i : ℕ) : Rs.getD i 0 ∈ Rs ∨ Rs.getD i 0 = 0 := by
  by_cases h : i < Rs.length
  · left
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h, Option.getD_some]
    exact List.getElem_mem h
  · right
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none (not_lt.1 h), Option.getD_none]

theorem getD_inv {Rs Vs : List (Noise N m n k → ℚ)} (h : ∀ r ∈ Rs, r ∈ V0 N m n k ∨ r ∈ Vs) (i : ℕ) :
    Rs.getD i 0 ∈ V0 N m n k ∨ Rs.getD i 0 ∈ Vs := by
  rcases getD_mem_or Rs i with hi | hi
  · exact h _ hi
  · rw [hi]
    exact Or.inl (Submodule.zero_mem _)

/-! ## Transfer of steps from registers to values -/

theorem pipeOpnd_transfer {Rs Vs : List (Noise N m n k → ℚ)} (h : ∀ r ∈ Rs, r ∈ V0 N m n k ∨ r ∈ Vs)
    {x : Noise N m n k → ℚ} (hx : PipeOpnd Rs x) : PipeOpnd Vs x := by
  rcases hx with hx | ⟨r, hr, β, rfl⟩
  · exact Or.inl hx
  · rcases h r hr with h0 | h1
    · exact Or.inl (Submodule.smul_mem _ β h0)
    · exact Or.inr ⟨r, h1, β, rfl⟩

theorem quadStep_transfer {Rs Vs : List (Noise N m n k → ℚ)} (h : ∀ r ∈ Rs, r ∈ V0 N m n k ∨ r ∈ Vs)
    {f : Noise N m n k → ℚ} (hf : QuadStep Rs f) : QuadStep Vs f := by
  obtain ⟨a, b, c, hc, rfl, hab⟩ := hf
  refine ⟨a, b, c, pipeOpnd_transfer h hc, rfl, ?_⟩
  rcases hab with hab | ⟨hα, hb⟩
  · exact Or.inl hab
  · exact Or.inr ⟨hα, pipeOpnd_transfer h hb⟩

theorem fpVal_step (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) {Rs Vs : List (Noise N m n k → ℚ)}
    (h : ∀ r ∈ Rs, r ∈ V0 N m n k ∨ r ∈ Vs) (κ i j l : ℕ) : FpStep fpSem Vs (fpVal fpSem κ i j l Rs) :=
  ⟨κ, Rs.getD i 0, Rs.getD j 0, Rs.getD l 0, getD_inv h i, getD_inv h j, getD_inv h l, rfl⟩

/-! ## A pipe with an arbitrary admission rule -/

/-- A pipe whose priced instructions satisfy `C` (given the earlier values), with no free instruction. -/
def GOK {G : Type} (C : List (G → ℚ) → ℕ → (G → ℚ) → Prop) : List (G → ℚ) → List (Instr G) → Prop
  | _, [] => True
  | _, .free _ _ :: _ => False
  | Vs, .priced p f :: P => C Vs p f ∧ GOK C (Vs ++ [f]) P

theorem gok_append {G : Type} (C : List (G → ℚ) → ℕ → (G → ℚ) → Prop) :
    ∀ (P Q : List (Instr G)) (Vs : List (G → ℚ)), GOK C Vs P → GOK C (Vs ++ vals P) Q → GOK C Vs (P ++ Q) := by
  intro P
  induction P with
  | nil => intro Q Vs _ hQ; simpa [vals] using hQ
  | cons ins P ih =>
    intro Q Vs hP hQ
    cases ins with
    | free c w => exact hP.elim
    | priced p f => exact ⟨hP.1, ih Q (Vs ++ [f]) hP.2 (by simpa [vals, OpenPipe.val] using hQ)⟩

theorem gok_nofree {G : Type} (C : List (G → ℚ) → ℕ → (G → ℚ) → Prop) :
    ∀ (P : List (Instr G)) (Vs : List (G → ℚ)), GOK C Vs P → ∀ c w, Instr.free c w ∉ P := by
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

theorem gok_price {G : Type} (C : List (G → ℚ) → ℕ → (G → ℚ) → Prop) (c : ℕ) (hC : ∀ Vs p f, C Vs p f → c ≤ p) :
    ∀ (P : List (Instr G)) (Vs : List (G → ℚ)), GOK C Vs P → c * P.length ≤ cost P := by
  intro P
  induction P with
  | nil => intro _ _; simp [cost]
  | cons ins P ih =>
    intro Vs hP
    cases ins with
    | free c' w' => exact hP.elim
    | priced p f =>
      have h1 := hC _ _ _ hP.1
      have h2 := ih _ hP.2
      simp only [cost, List.map_cons, List.sum_cons, Instr.price, List.length_cons] at h2 ⊢
      rw [Nat.mul_succ]
      omega

theorem openOK_iff : ∀ (P : List (Instr (Noise N m n k))) (Vs : List (Noise N m n k → ℚ)),
    OpenOK 16 8 Vs P ↔ GOK (fun Vs p f => 16 ≤ p ∨ (8 ≤ p ∧ QuadStep Vs f)) Vs P := by
  intro P
  induction P with
  | nil => intro Vs; exact Iff.rfl
  | cons ins P ih =>
    intro Vs
    cases ins with
    | free c w => exact Iff.rfl
    | priced p f => exact and_congr Iff.rfl (ih _)

theorem fpOK_iff (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) : ∀ (P : List (Instr (Noise N m n k))) (Vs : List (Noise N m n k → ℚ)),
    FpOK fpSem Vs P ↔ GOK (fun Vs p f => 16 ≤ p ∨ (8 ≤ p ∧ FpStep fpSem Vs f)) Vs P := by
  intro P
  induction P with
  | nil => intro Vs; exact Iff.rfl
  | cons ins P ih =>
    intro Vs
    cases ins with
    | free c w => exact Iff.rfl
    | priced p f => exact and_congr Iff.rfl (ih _)

theorem mem_slots_of_vals {G : Type} {P : List (Instr G)} (hP : ∀ c w, Instr.free c w ∉ P) {r : G → ℚ}
    (hr : r ∈ vals P) (L : List (G → ℚ)) : r ∈ slots P L := by
  obtain ⟨ins, hins, rfl⟩ := List.mem_map.1 hr
  cases ins with
  | free c w => exact absurd hins (hP c w)
  | priced c f => exact mem_slots_of_priced hins L

theorem cost_append {G : Type} (P Q : List (Instr G)) : cost (P ++ Q) = cost P + cost Q := by
  simp [cost]

/-! ## The translation -/

open Classical in
/-- One instruction's translation; a non-quadratic FP32 instruction is priced `hi`. -/
noncomputable def Op.tr (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (hi : ℕ) :
    Op (Noise N m n k) → List (Noise N m n k → ℚ) → List (Instr (Noise N m n k))
  | .mma outs _, _ => outs.map (Instr.priced 16)
  | .fp32 κ i j l, Rs =>
      [Instr.priced (if QuadStep Rs (fpVal fpSem κ i j l Rs) then 8 else hi) (fpVal fpSem κ i j l Rs)]
  | .int f, _ => [Instr.priced 16 f]
  | .dd f, _ => [Instr.priced 16 f]
  | .subword f, _ => [Instr.priced 16 f]
  | .move _, _ => []
  | .presalt _, _ => []
  | .operand _, _ => []

/-- A program's translation. -/
noncomputable def tr (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (hi : ℕ) :
    List (Op (Noise N m n k)) → List (Noise N m n k → ℚ) → List (Instr (Noise N m n k))
  | [], _ => []
  | o :: P, Rs => o.tr fpSem hi Rs ++ tr fpSem hi P (Rs ++ o.outs fpSem Rs)

section Spec

variable (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (hi : ℕ)
  (C : List (Noise N m n k → ℚ) → ℕ → (Noise N m n k → ℚ) → Prop)
  (hbig : ∀ Vs p f, 16 ≤ p → C Vs p f)
  (hq8 : ∀ κ i j l (Rs Vs : List (Noise N m n k → ℚ)), (∀ r ∈ Rs, r ∈ V0 N m n k ∨ r ∈ Vs) →
    QuadStep Rs (fpVal fpSem κ i j l Rs) → C Vs 8 (fpVal fpSem κ i j l Rs))
  (hfp : ∀ κ i j l (Rs Vs : List (Noise N m n k → ℚ)), (∀ r ∈ Rs, r ∈ V0 N m n k ∨ r ∈ Vs) →
    C Vs hi (fpVal fpSem κ i j l Rs))
include hbig hq8 hfp

theorem op_spec (o : Op (Noise N m n k)) (Rs Vs : List (Noise N m n k → ℚ))
    (hinv : ∀ r ∈ Rs, r ∈ V0 N m n k ∨ r ∈ Vs) (hwf : o.WF) :
    GOK C Vs (o.tr fpSem hi Rs) ∧
      (∀ r ∈ o.outs fpSem Rs, r ∈ V0 N m n k ∨ r ∈ Vs ++ vals (o.tr fpSem hi Rs)) ∧
      cost (o.tr fpSem hi Rs) ≤ o.price + (hi - 8) * o.nq fpSem Rs := by
  have one : ∀ (p : ℕ) (f : Noise N m n k → ℚ), C Vs p f → GOK C Vs [Instr.priced p f] := fun p f h => ⟨h, trivial⟩
  have self : ∀ (p : ℕ) (f : Noise N m n k → ℚ), f ∈ Vs ++ vals [Instr.priced p f] := fun p f => by simp [vals, OpenPipe.val]
  cases o with
  | mma outs p =>
    have hvals : vals (outs.map (Instr.priced 16)) = outs := by
      simp only [vals, List.map_map]
      exact List.map_id' _ ▸ rfl
    refine ⟨?_, fun r hr => Or.inr (List.mem_append_right _ (by simpa [Op.tr, Op.outs, hvals] using hr)), ?_⟩
    · have : ∀ (fs : List (Noise N m n k → ℚ)) (Ws : List (Noise N m n k → ℚ)), GOK C Ws (fs.map (Instr.priced 16)) := by
        intro fs
        induction fs with
        | nil => intro _; trivial
        | cons f fs ih => intro Ws; exact ⟨hbig _ _ _ le_rfl, ih _⟩
      exact this outs Vs
    · have hc : cost (outs.map (Instr.priced 16) : List (Instr (Noise N m n k))) = 16 * outs.length := by
        simp only [cost, List.map_map]
        rw [show (Instr.price ∘ Instr.priced 16 : (Noise N m n k → ℚ) → ℕ) = fun _ => 16 from rfl, List.map_const',
          List.sum_replicate, smul_eq_mul, Nat.mul_comm]
      simp only [Op.tr, Op.price, Op.nq, Nat.mul_zero, Nat.add_zero, hc]
      exact hwf
  | fp32 κ i j l =>
    by_cases hq : QuadStep Rs (fpVal fpSem κ i j l Rs)
    · have htr : Op.tr fpSem hi (Op.fp32 κ i j l) Rs = [Instr.priced 8 (fpVal fpSem κ i j l Rs)] := by
        simp [Op.tr, hq]
      have hnq : Op.nq fpSem (Op.fp32 κ i j l) Rs = 0 := by simp [Op.nq, hq]
      rw [htr, hnq]
      refine ⟨one _ _ (hq8 κ i j l Rs Vs hinv hq), fun r hr => ?_,
        by simp [cost, Op.price, Instr.price]⟩
      simp only [Op.outs, List.mem_singleton] at hr
      exact Or.inr (hr ▸ self _ _)
    · have htr : Op.tr fpSem hi (Op.fp32 κ i j l) Rs = [Instr.priced hi (fpVal fpSem κ i j l Rs)] := by
        simp [Op.tr, hq]
      have hnq : Op.nq fpSem (Op.fp32 κ i j l) Rs = 1 := by simp [Op.nq, hq]
      rw [htr, hnq]
      refine ⟨one _ _ (hfp κ i j l Rs Vs hinv), fun r hr => ?_, ?_⟩
      · simp only [Op.outs, List.mem_singleton] at hr
        exact Or.inr (hr ▸ self _ _)
      · simp only [cost, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Instr.price, Op.price]
        omega
  | int f =>
    refine ⟨one _ _ (hbig _ _ _ le_rfl), fun r hr => ?_, by simp [Op.tr, cost, Op.price, Op.nq, Instr.price]⟩
    simp only [Op.outs, List.mem_singleton] at hr
    exact Or.inr (hr ▸ self _ _)
  | dd f =>
    refine ⟨one _ _ (hbig _ _ _ le_rfl), fun r hr => ?_, by simp [Op.tr, cost, Op.price, Op.nq, Instr.price]⟩
    simp only [Op.outs, List.mem_singleton] at hr
    exact Or.inr (hr ▸ self _ _)
  | subword f =>
    refine ⟨one _ _ (hbig _ _ _ le_rfl), fun r hr => ?_, by simp [Op.tr, cost, Op.price, Op.nq, Instr.price]⟩
    simp only [Op.outs, List.mem_singleton] at hr
    exact Or.inr (hr ▸ self _ _)
  | move i =>
    refine ⟨trivial, fun r hr => ?_, by simp [Op.tr, cost, Op.nq]⟩
    simp only [Op.outs, List.mem_singleton] at hr
    subst hr
    simpa [Op.tr, vals] using getD_inv hinv i
  | presalt f =>
    refine ⟨trivial, fun r hr => ?_, by simp [Op.tr, cost, Op.nq]⟩
    simp only [Op.outs, List.mem_singleton] at hr
    exact Or.inl (hr ▸ hwf)
  | operand f =>
    refine ⟨trivial, fun r hr => ?_, by simp [Op.tr, cost, Op.nq]⟩
    simp only [Op.outs, List.mem_singleton] at hr
    exact Or.inl (hr ▸ hwf)

theorem tr_spec : ∀ (P : List (Op (Noise N m n k))) (Rs Vs : List (Noise N m n k → ℚ)),
    (∀ r ∈ Rs, r ∈ V0 N m n k ∨ r ∈ Vs) → (∀ o ∈ P, o.WF) →
    GOK C Vs (tr fpSem hi P Rs) ∧
      (∀ r ∈ regs fpSem P Rs, r ∈ V0 N m n k ∨ r ∈ Vs ++ vals (tr fpSem hi P Rs)) ∧
      cost (tr fpSem hi P Rs) ≤ cost4090 P + (hi - 8) * nonQuad fpSem P Rs := by
  intro P
  induction P with
  | nil =>
    intro Rs Vs hinv _
    refine ⟨trivial, fun r hr => ?_, by simp [tr, cost]⟩
    simpa [tr, vals] using hinv r hr
  | cons o P ih =>
    intro Rs Vs hinv hwf
    obtain ⟨h1, h2, h3⟩ := op_spec fpSem hi C hbig hq8 hfp o Rs Vs hinv (hwf o List.mem_cons_self)
    have hinv' : ∀ r ∈ Rs ++ o.outs fpSem Rs, r ∈ V0 N m n k ∨ r ∈ Vs ++ vals (o.tr fpSem hi Rs) := by
      intro r hr
      rcases List.mem_append.1 hr with hr | hr
      · rcases hinv r hr with h | h
        · exact Or.inl h
        · exact Or.inr (List.mem_append_left _ h)
      · exact h2 r hr
    obtain ⟨i1, i2, i3⟩ := ih _ _ hinv' fun o' ho' => hwf o' (List.mem_cons_of_mem _ ho')
    refine ⟨gok_append C _ _ _ h1 i1, fun r hr => ?_, ?_⟩
    · have := i2 r hr
      simpa [tr, vals, List.map_append, List.append_assoc] using this
    · have e1 : cost4090 (o :: P) = o.price + cost4090 P := by simp [cost4090]
      have e2 : nonQuad fpSem (o :: P) Rs = o.nq fpSem Rs + nonQuad fpSem P (Rs ++ o.outs fpSem Rs) := rfl
      have e3 : tr fpSem hi (o :: P) Rs = o.tr fpSem hi Rs ++ tr fpSem hi P (Rs ++ o.outs fpSem Rs) := rfl
      rw [e1, e2, e3, cost_append, Nat.mul_add]
      omega

theorem embed_gen (L : List (Noise N m n k → ℚ)) (P : List (Op (Noise N m n k))) (hL : ∀ f ∈ L, f ∈ V0 N m n k)
    (hwf : ∀ o ∈ P, o.WF) :
    GOK C [] (tr fpSem hi P L) ∧ (∀ r ∈ regs fpSem P L, r ∈ V0 N m n k ∨ r ∈ slots (tr fpSem hi P L) L) ∧
      cost (tr fpSem hi P L) ≤ cost4090 P + (hi - 8) * nonQuad fpSem P L := by
  obtain ⟨h1, h2, h3⟩ := tr_spec fpSem hi C hbig hq8 hfp P L [] (fun r hr => Or.inl (hL r hr)) hwf
  refine ⟨h1, fun r hr => ?_, h3⟩
  rcases h2 r hr with h | h
  · exact Or.inl h
  · exact Or.inr (mem_slots_of_vals (gok_nofree C _ _ h1) (by simpa using h) L)

end Spec

/-! ## The pinned theorems -/

/-- **The embedding theorem.** -/
theorem embedM4090 : EmbedM4090 := by
  intro N m n k fpSem L P hL hwf
  obtain ⟨h1, h2, h3⟩ := embed_gen fpSem 16 (fun Vs p f => 16 ≤ p ∨ (8 ≤ p ∧ QuadStep Vs f))
    (fun _ _ _ h => Or.inl h) (fun _ _ _ _ _ _ hinv hq => Or.inr ⟨le_rfl, quadStep_transfer hinv hq⟩)
    (fun _ _ _ _ _ _ _ => Or.inl le_rfl) L P hL hwf
  refine ⟨_, (openOK_iff _ _).2 h1, h2, ?_⟩
  have e : (16 - 8 : ℕ) = 8 := rfl
  rw [e] at h3
  exact h3

/-- **The exact embedding into the FP32 pipe.** -/
theorem embedFp : EmbedFp := by
  intro N m n k fpSem L P hL hwf
  obtain ⟨h1, h2, h3⟩ := embed_gen fpSem 8 (fun Vs p f => 16 ≤ p ∨ (8 ≤ p ∧ FpStep fpSem Vs f))
    (fun _ _ _ h => Or.inl h) (fun κ i j l _ _ hinv _ => Or.inr ⟨le_rfl, fpVal_step fpSem hinv κ i j l⟩)
    (fun κ i j l _ _ hinv => Or.inr ⟨le_rfl, fpVal_step fpSem hinv κ i j l⟩) L P hL hwf
  exact ⟨_, (fpOK_iff fpSem _ _).2 h1, h2, by simpa using h3⟩

theorem word_slot {k m n N : ℕ} {R : Mat} (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) {U : Finset ℕ}
    (hind : CheckpointIndependent k 16 R) (hU : U ⊆ range N) {x : ℕ × ℕ} (hx : x ∈ U ×ˢ range (nonFinal k m n))
    {P : List (Instr (Noise N m n k))} {L : List (Noise N m n k → ℚ)}
    (h : word k m n N R A B x.1 x.2 ∈ V0 N m n k ∨ word k m n N R A B x.1 x.2 ∈ slots P L) :
    word k m n N R A B x.1 x.2 ∈ slots P L :=
  h.resolve_left (pipe_word_nmem A B hind hU hx)

/-- **The per-word cost in `M4090`, less the non-quadratic FP32 instructions.** -/
theorem perWordM4090 : PerWordM4090 := by
  intro k m n N R A B U fpSem L P hind hMR hU hL hwf S hS hSw
  obtain ⟨P', hP', hreg, hcost⟩ := embedM4090 N m n k fpSem L P hL hwf
  obtain ⟨h2, hli⟩ := gridNear k m n N R A B U (word k m n N R A B) hind hMR hU
    (fun u _ p _ => by rw [sub_self]; exact Submodule.zero_mem _)
  have h := theorem1Open N m n k (word k m n N R A B) U h2 hli 16 8 L P' hL hP' S hS fun x hx =>
    word_slot A B hind hU (hS hx) (hreg _ (hSw x hx))
  have hmin : min 16 (2 * 8) = 16 := rfl
  rw [hmin] at h
  omega

/-! ## The converse: the FP32 pipe inside `M4090` -/

/-- **`A1_fp` at `γ = 1/2` for every FP32 semantics.** -/
theorem a1fpHalf : A1fpHalf := by
  intro fpSem k m n N R A B U hind hMR hU L P hL hP S hS hSw
  classical
  have hg := (fpOK_iff fpSem P []).1 hP
  have hnf := gok_nofree _ _ _ hg
  have hmem : ∀ x ∈ S, word k m n N R A B x.1 x.2 ∈ vals P := fun x hx => by
    rcases mem_slots_vals P L hnf _ (hSw x hx) with h | h
    · exact absurd (hL _ h) (pipe_word_nmem A B hind hU (hS hx))
    · exact h
  have hinj : Set.InjOn (fun x : ℕ × ℕ => word k m n N R A B x.1 x.2) ↑S :=
    (pipe_word_injOn A B hind hU).mono (Finset.coe_subset.2 hS)
  have hcard : S.card ≤ P.length :=
    calc S.card = (S.image fun x => word k m n N R A B x.1 x.2).card := (card_image_of_injOn hinj).symm
      _ ≤ (vals P).toFinset.card := card_le_card fun φ hφ => by
          obtain ⟨x, hx, rfl⟩ := mem_image.1 hφ
          exact List.mem_toFinset.2 (hmem x hx)
      _ ≤ (vals P).length := List.toFinset_card_le _
      _ = P.length := List.length_map _
  have hcost := gok_price _ 8 (fun _ _ _ h => by rcases h with h | ⟨h, -⟩ <;> omega) P [] hg
  have h8 : (8 * S.card : ℝ) ≤ cost P := by exact_mod_cast (Nat.mul_le_mul_left 8 hcard).trans hcost
  linarith

theorem gok_mono {G : Type} (C C' : List (G → ℚ) → ℕ → (G → ℚ) → Prop) (h : ∀ Vs p f, C Vs p f → C' Vs p f) :
    ∀ (P : List (Instr G)) (Vs : List (G → ℚ)), GOK C Vs P → GOK C' Vs P := by
  intro P
  induction P with
  | nil => intro _ _; trivial
  | cons ins P ih =>
    intro Vs hP
    cases ins with
    | free c w => exact hP.elim
    | priced p f => exact ⟨h _ _ _ hP.1, ih _ hP.2⟩

/-- An FADD step `x + y` is a `QuadStep`: `1·x + y`. -/
theorem fadd_quad {Vs : List (Noise N m n k → ℚ)} {f : Noise N m n k → ℚ} (h : FpStep faddSem Vs f) :
    QuadStep Vs f := by
  obtain ⟨κ, x, y, z, hx, hy, -, rfl⟩ := h
  have op : ∀ {v : Noise N m n k → ℚ}, FpOpnd Vs v → PipeOpnd Vs v := fun hv =>
    hv.imp id fun hv => ⟨_, hv, 1, (one_smul ℚ _).symm⟩
  refine ⟨fun _ => 1, x, y, op hy, ?_, Or.inr ⟨⟨1, rfl⟩, op hx⟩⟩
  funext ω
  simp [faddSem]

/-- **`A1_fp` at `γ = 0` for the FADD-only semantics.** -/
theorem a1fpFadd : A1fpFadd := by
  intro k m n N R A B U hind hMR hU L P hL hP S hS hSw
  have hP' : OpenQuadPipe 16 8 P := (openOK_iff P []).2
    (gok_mono _ _ (fun _ _ _ h => h.imp id fun h => ⟨h.1, fadd_quad h.2⟩) P [] ((fpOK_iff faddSem P []).1 hP))
  obtain ⟨h2, hli⟩ := gridNear k m n N R A B U (word k m n N R A B) hind hMR hU
    (fun u _ p _ => by rw [sub_self]; exact Submodule.zero_mem _)
  have h := theorem1Open N m n k (word k m n N R A B) U h2 hli 16 8 L P hL hP' S hS hSw
  have hmin : min 16 (2 * 8) = 16 := rfl
  rw [hmin] at h
  have h' : (16 * S.card : ℝ) ≤ cost P := by exact_mod_cast h
  rw [sub_zero, one_mul]
  exact h'

/-! ## The range gap -/

/-- **`CanonicalNaN` is satisfiable.** -/
theorem canonicalNaNSat : CanonicalNaNSat := by
  intro fpSem κ a b c h
  unfold canonNaN at h
  split_ifs at h with h0
  · unfold NaNGap at h
    rcases h with ⟨-, h⟩ | ⟨-, h⟩ <;> norm_num at h
  · exact h0 h

theorem gok_mem {G : Type} (C : List (G → ℚ) → ℕ → (G → ℚ) → Prop) :
    ∀ (P : List (Instr G)) (Vs : List (G → ℚ)), GOK C Vs P → ∀ p f, Instr.priced p f ∈ P → ∃ Ws, C Ws p f := by
  intro P
  induction P with
  | nil => intro _ _ p f h; simp at h
  | cons ins P ih =>
    intro Vs hP p f h
    cases ins with
    | free c w => exact hP.elim
    | priced q g =>
      rcases List.mem_cons.1 h with h | h
      · cases h
        exact ⟨Vs, hP.1⟩
      · exact ih _ hP.2 p f h

open Classical in
/-- If every `w ∈ W` is the value of a priced instruction of `P`, and every priced instruction whose value is in `W`
costs at least 16, then `16·|W| ≤ cost P`. -/
theorem count16 {G : Type} : ∀ (P : List (Instr G)) (W : Finset (G → ℚ)),
    (∀ w ∈ W, ∃ p, Instr.priced p w ∈ P) → (∀ p f, Instr.priced p f ∈ P → f ∈ W → 16 ≤ p) →
    16 * W.card ≤ cost P := by
  intro P
  induction P with
  | nil =>
    intro W hW _
    have hW0 : W = ∅ := by
      ext w
      refine ⟨fun hw => ?_, fun h => by simp at h⟩
      obtain ⟨p, hp⟩ := hW w hw
      simp at hp
    simp [hW0, cost]
  | cons ins P ih =>
    intro W hW hp
    cases ins with
    | free c w =>
      have h := ih W (fun w' hw' => by
          obtain ⟨p, hp'⟩ := hW w' hw'
          rcases List.mem_cons.1 hp' with h | h
          · cases h
          · exact ⟨p, h⟩)
        (fun p f h hf => hp p f (List.mem_cons_of_mem _ h) hf)
      simp only [cost, List.map_cons, List.sum_cons] at h ⊢
      omega
    | priced q f =>
      by_cases hf : f ∈ W
      · have h16 := hp q f List.mem_cons_self hf
        have h := ih (W.erase f) (fun w' hw' => by
            obtain ⟨hne, hw⟩ := Finset.mem_erase.1 hw'
            obtain ⟨p, hp'⟩ := hW w' hw
            rcases List.mem_cons.1 hp' with h | h
            · cases h
              exact absurd rfl hne
            · exact ⟨p, h⟩)
          (fun p g h hg => hp p g (List.mem_cons_of_mem _ h) (Finset.mem_of_mem_erase hg))
        have hc := Finset.card_erase_add_one hf
        simp only [cost, List.map_cons, List.sum_cons, Instr.price] at h ⊢
        rw [← hc]
        omega
      · have h := ih W (fun w' hw' => by
            obtain ⟨p, hp'⟩ := hW w' hw'
            rcases List.mem_cons.1 hp' with h | h
            · cases h
              exact absurd hw' hf
            · exact ⟨p, h⟩)
          (fun p g h hg => hp p g (List.mem_cons_of_mem _ h) hg)
        simp only [cost, List.map_cons, List.sum_cons] at h ⊢
        omega

/-- **`A1_fp` at `γ = 0` from a range gap** (`a1fpOn_of_rangeGap`). -/
theorem a1fpOnRangeGap : A1fpOnRangeGap := by
  intro fpSem k m n N R A B U hind hU hgap L P hL hP S hS hSw
  classical
  have hg := (fpOK_iff fpSem P []).1 hP
  have hnf := gok_nofree _ _ _ hg
  have hmem : ∀ x ∈ S, ∃ p, Instr.priced p (word k m n N R A B x.1 x.2) ∈ P := fun x hx => by
    rcases mem_slots_vals P L hnf _ (hSw x hx) with h | h
    · exact absurd (hL _ h) (pipe_word_nmem A B hind hU (hS hx))
    · obtain ⟨ins, hins, he⟩ := List.mem_map.1 h
      cases ins with
      | free c w => exact absurd hins (hnf c w)
      | priced p f =>
        have he' : f = word k m n N R A B x.1 x.2 := he
        subst he'
        exact ⟨p, hins⟩
  have hinj : Set.InjOn (fun x : ℕ × ℕ => word k m n N R A B x.1 x.2) ↑S :=
    (pipe_word_injOn A B hind hU).mono (Finset.coe_subset.2 hS)
  have hW : ∀ w ∈ S.image (fun x => word k m n N R A B x.1 x.2), ∃ p, Instr.priced p w ∈ P := fun w hw => by
    obtain ⟨x, hx, rfl⟩ := Finset.mem_image.1 hw
    exact hmem x hx
  have hp : ∀ p f, Instr.priced p f ∈ P → f ∈ S.image (fun x => word k m n N R A B x.1 x.2) → 16 ≤ p :=
    fun p f h hf => by
      obtain ⟨x, hx, rfl⟩ := Finset.mem_image.1 hf
      obtain ⟨Ws, hC⟩ := gok_mem _ _ [] hg p _ h
      rcases hC with h16 | ⟨-, κ, a, b, c, -, -, -, he⟩
      · exact h16
      · obtain ⟨ω, hω⟩ := hgap x (hS hx)
        have e : word k m n N R A B x.1 x.2 ω = fpSem κ (a ω) (b ω) (c ω) := congrFun he ω
        exact absurd e.symm (hω κ (a ω) (b ω) (c ω))
  have h := count16 P _ hW hp
  rw [card_image_of_injOn hinj] at h
  have h' : (16 * S.card : ℝ) ≤ cost P := by exact_mod_cast h
  rw [sub_zero, one_mul]
  exact h'

/-- **The per-word cost of every `M4090` program from a range gap.** -/
theorem perWordM4090Gap : PerWordM4090Gap := by
  intro fpSem k m n N R A B U L P hind hU hL hwf hgap S hS hSw
  obtain ⟨P', hP', hreg, hcost⟩ := embedFp N m n k fpSem L P hL hwf
  have h := a1fpOnRangeGap fpSem k m n N R A B U hind hU hgap L P' hL hP' S hS fun x hx =>
    word_slot A B hind hU (hS hx) (hreg _ (hSw x hx))
  rw [sub_zero, one_mul] at h
  have h' : 16 * S.card ≤ cost P' := by exact_mod_cast h
  omega

end Pouw.Dimension.Embed
