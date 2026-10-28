import Definitions.Pouw.Assumptions.Dimension.H100W
import Proofs.Pouw.Dimension.H100Proofs

/-!
# `MH100w` and the distinct-writes bound: proofs

* **The cover** (`regs_cover`): every register of a well-formed `MH100w` program on free inputs is free, or written by
  an instruction at `a` or more per register (`hiP`), or by a tensor-core instruction below `a` (`loP`).
* **The cost** (`cost_hi_lo`): under `FloorTc a c`, the program costs at least `a` per `hiP` register and `c` per
  `loP` register.
* **The count** (`distinctWritesLegacy`): each target matches on `S` a register that is not free, and distinct
  targets match distinct registers; targets outside `K` match `hiP` registers. Every other bound is an instance.
-/

namespace Pouw.Dimension.H100

open Finset Pouw.Dimension Pouw.Dimension.Embed

variable {G : Type}

/-- The registers an instruction writes at `a` or more per register. -/
def Op100.hi (a : ℕ) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) : Op100 G → List (G → ℚ) → List (G → ℚ)
  | .wgmma _ outs, _ => outs
  | .mma outs p, _ => if a * outs.length ≤ p then outs else []
  | .fp32 κ i j l, Rs => [fpVal fpSem κ i j l Rs]
  | .int f, _ => [f]
  | .fp64 f, _ => [f]
  | .other f, _ => [f]
  | .move _, _ => []
  | .operand _, _ => []

/-- The registers a tensor-core instruction writes below `a` per register. -/
def Op100.lo (a : ℕ) : Op100 G → List (G → ℚ)
  | .wgmma _ _ => []
  | .mma outs p => if a * outs.length ≤ p then [] else outs
  | .fp32 .. => []
  | .int _ => []
  | .fp64 _ => []
  | .other _ => []
  | .move _ => []
  | .operand _ => []

/-- A program's registers written at `a` or more per register. -/
def hiP (a : ℕ) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) : List (Op100 G) → List (G → ℚ) → List (G → ℚ)
  | [], _ => []
  | o :: P, Rs => o.hi a fpSem Rs ++ hiP a fpSem P (Rs ++ o.outs fpSem Rs)

/-- A program's registers written below `a` per register. -/
def loP (a : ℕ) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) : List (Op100 G) → List (G → ℚ) → List (G → ℚ)
  | [], _ => []
  | o :: P, Rs => o.lo a ++ loP a fpSem P (Rs ++ o.outs fpSem Rs)

theorem regs_cover (a : ℕ) (pr : Prices100) (F : Set (G → ℚ)) (h0 : (0 : G → ℚ) ∈ F)
    (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) :
    ∀ (P : List (Op100 G)) (Rs H Lo : List (G → ℚ)),
      (∀ r ∈ Rs, r ∈ F ∨ r ∈ H ∨ r ∈ Lo) → (∀ o ∈ P, o.WFw pr F) →
      ∀ r ∈ regs100 fpSem P Rs, r ∈ F ∨ r ∈ H ++ hiP a fpSem P Rs ∨ r ∈ Lo ++ loP a fpSem P Rs := by
  intro P
  induction P with
  | nil =>
    intro Rs H Lo hinv _ r hr
    simpa [hiP, loP, regs100] using hinv r hr
  | cons o P ih =>
    intro Rs H Lo hinv hwf r hr
    have hwo := hwf o List.mem_cons_self
    have hstep : ∀ r ∈ Rs ++ o.outs fpSem Rs,
        r ∈ F ∨ r ∈ H ++ o.hi a fpSem Rs ∨ r ∈ Lo ++ o.lo a := by
      intro r hr
      have hi : r ∈ o.hi a fpSem Rs → r ∈ F ∨ r ∈ H ++ o.hi a fpSem Rs ∨ r ∈ Lo ++ o.lo a :=
        fun h => Or.inr (Or.inl (List.mem_append_right _ h))
      have old : ∀ r ∈ Rs, r ∈ F ∨ r ∈ H ++ o.hi a fpSem Rs ∨ r ∈ Lo ++ o.lo a := by
        intro r hr
        rcases hinv r hr with h | h | h
        · exact Or.inl h
        · exact Or.inr (Or.inl (List.mem_append_left _ h))
        · exact Or.inr (Or.inr (List.mem_append_left _ h))
      rcases List.mem_append.1 hr with hr | hr
      · exact old r hr
      · cases o with
        | wgmma s outs => exact hi (by simpa [Op100.outs, Op100.hi] using hr)
        | mma outs p =>
          simp only [Op100.outs] at hr
          by_cases hp : a * outs.length ≤ p
          · exact hi (by simpa [Op100.hi, hp] using hr)
          · exact Or.inr (Or.inr (List.mem_append_right _ (by simpa [Op100.lo, hp] using hr)))
        | fp32 κ i j l => exact hi (by simpa [Op100.outs, Op100.hi] using hr)
        | int f => exact hi (by simpa [Op100.outs, Op100.hi] using hr)
        | fp64 f => exact hi (by simpa [Op100.outs, Op100.hi] using hr)
        | other f => exact hi (by simpa [Op100.outs, Op100.hi] using hr)
        | move i =>
          simp only [Op100.outs, List.mem_singleton] at hr
          subst hr
          rcases getD_mem_or Rs i with h | h
          · exact old _ h
          · rw [h]
            exact Or.inl h0
        | operand f =>
          simp only [Op100.outs, List.mem_singleton] at hr
          subst hr
          exact Or.inl hwo
    have := ih (Rs ++ o.outs fpSem Rs) (H ++ o.hi a fpSem Rs) (Lo ++ o.lo a) hstep
      (fun o' ho' => hwf o' (List.mem_cons_of_mem _ ho')) r hr
    simpa [hiP, loP, List.append_assoc] using this

theorem loP_sub (a : ℕ) (K : Set (G → ℚ)) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) :
    ∀ (P : List (Op100 G)) (Rs : List (G → ℚ)), (∀ o ∈ P, o.CheapIn a K) → ∀ r ∈ loP a fpSem P Rs, r ∈ K := by
  intro P
  induction P with
  | nil => intro Rs _ r hr; simp [loP] at hr
  | cons o P ih =>
    intro Rs hK r hr
    simp only [loP, List.mem_append] at hr
    rcases hr with hr | hr
    · have hko := hK o List.mem_cons_self
      cases o with
      | mma outs p =>
        by_cases hp : a * outs.length ≤ p
        · simp [Op100.lo, hp] at hr
        · simp only [Op100.lo, hp, ite_false] at hr
          exact hko (by omega) r hr
      | _ => simp [Op100.lo] at hr
    · exact ih _ (fun o' ho' => hK o' (List.mem_cons_of_mem _ ho')) r hr

theorem cost_hi_lo (a c : ℕ) (pr : Prices100) (hpr : FloorTc a c pr) (F : Set (G → ℚ))
    (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) :
    ∀ (P : List (Op100 G)) (Rs : List (G → ℚ)), (∀ o ∈ P, o.WFw pr F) →
      a * (hiP a fpSem P Rs).length + c * (loP a fpSem P Rs).length ≤ cost100 pr P := by
  obtain ⟨hs, htc, hffma, hint, hfp64, hother⟩ := hpr
  intro P
  induction P with
  | nil => intro Rs _; simp [hiP, loP, cost100]
  | cons o P ih =>
    intro Rs hwf
    have h1 := ih (Rs ++ o.outs fpSem Rs) (fun o' ho' => hwf o' (List.mem_cons_of_mem _ ho'))
    have hwo := hwf o List.mem_cons_self
    have h2 : a * (o.hi a fpSem Rs).length + c * (o.lo a).length ≤ o.price pr := by
      cases o with
      | wgmma s outs => simpa [Op100.hi, Op100.lo, Op100.price] using Nat.mul_le_mul_right outs.length (hs s)
      | mma outs p =>
        simp only [Op100.WFw] at hwo
        by_cases hp : a * outs.length ≤ p
        · simp [Op100.hi, Op100.lo, Op100.price, hp]
        · simp only [Op100.hi, Op100.lo, Op100.price, hp, ite_false, List.length_nil, Nat.mul_zero, Nat.zero_add]
          exact le_trans (Nat.mul_le_mul_right _ htc) hwo
      | fp32 κ i j l => simpa [Op100.hi, Op100.lo, Op100.price] using hffma
      | int f => simpa [Op100.hi, Op100.lo, Op100.price] using hint
      | fp64 f => simpa [Op100.hi, Op100.lo, Op100.price] using hfp64
      | other f => simpa [Op100.hi, Op100.lo, Op100.price] using hother
      | move i => simp [Op100.hi, Op100.lo, Op100.price]
      | operand f => simp [Op100.hi, Op100.lo, Op100.price]
    have e : cost100 pr (o :: P) = o.price pr + cost100 pr P := by simp [cost100]
    simp only [hiP, loP, List.length_append, e, Nat.mul_add]
    omega

theorem const_zero {F : Set (G → ℚ)} (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ F) : (0 : G → ℚ) ∈ F :=
  hconst 0

/-- **The distinct-writes bound with a cheap legacy class.** -/
theorem distinctWritesLegacy : DistinctWritesLegacy := by
  intro a c pr hpr hca G F K hconst S fpSem L P T₁ T₂ hL hwf hK hd₁ hd₂ hcross hT₁ ho₁ ho₂
  classical
  have hcov := regs_cover a pr F (const_zero hconst) fpSem P L [] [] (fun r hr => Or.inl (hL r hr)) hwf
  have hloK := loP_sub a K fpSem P L hK
  have hcost := cost_hi_lo a c pr hpr F fpSem P L hwf
  have hex : ∀ t ∈ T₁ ∪ T₂, ∃ r ∈ regs100 fpSem P L, ∀ g ∈ S, r g = t g := by
    intro t ht
    rcases Finset.mem_union.1 ht with h | h
    · exact ho₁ t h
    · exact ho₂ t h
  choose! ρ hρR hρS using hex
  have hdist : ∀ t ∈ T₁ ∪ T₂, ∀ t' ∈ T₁ ∪ T₂, (∀ g ∈ S, t g = t' g) → t = t' := by
    intro t ht t' ht' h
    rcases Finset.mem_union.1 ht with ht | ht <;> rcases Finset.mem_union.1 ht' with ht' | ht'
    · exact hd₁.1 t ht t' ht' h
    · exact absurd h (hcross t ht t' ht')
    · exact absurd (fun g hg => (h g hg).symm) (hcross t' ht' t ht)
    · exact hd₂.1 t ht t' ht' h
  have hfree : ∀ t ∈ T₁ ∪ T₂, ρ t ∉ F := by
    intro t ht hF
    have hnf : ∀ f ∈ F, ¬ ∀ g ∈ S, t g = f g := by
      rcases Finset.mem_union.1 ht with h | h
      · exact hd₁.2 t h
      · exact hd₂.2 t h
    exact hnf _ hF fun g hg => (hρS t ht g hg).symm
  have hinj : Set.InjOn ρ ↑(T₁ ∪ T₂) := by
    intro t ht t' ht' h
    refine hdist t ht t' ht' fun g hg => ?_
    rw [← hρS t ht g hg, ← hρS t' ht' g hg, h]
  have hall : ∀ t ∈ T₁ ∪ T₂, ρ t ∈ (hiP a fpSem P L ++ loP a fpSem P L).toFinset := by
    intro t ht
    rcases hcov _ (hρR t ht) with h | h | h
    · exact absurd h (hfree t ht)
    · simpa using Or.inl h
    · simpa using Or.inr h
  have hone : ∀ t ∈ T₁, ρ t ∈ (hiP a fpSem P L).toFinset := by
    intro t ht
    have ht' : t ∈ T₁ ∪ T₂ := Finset.mem_union_left _ ht
    rcases hcov _ (hρR t ht') with h | h | h
    · exact absurd h (hfree t ht')
    · simpa using h
    · exact absurd (fun g hg => (hρS t ht' g hg).symm) (hT₁ t ht _ (hloK _ (by simpa using h)))
  have hc1 : (T₁ ∪ T₂).card ≤ (hiP a fpSem P L).length + (loP a fpSem P L).length := by
    refine le_trans (Finset.card_le_card_of_injOn ρ (fun t ht => hall t ht) hinj) ?_
    exact le_trans (List.toFinset_card_le _) (by simp)
  have hc2 : T₁.card ≤ (hiP a fpSem P L).length := by
    refine le_trans (Finset.card_le_card_of_injOn ρ (fun t ht => hone t ht)
      (hinj.mono (by simp [Finset.coe_union]))) ?_
    exact List.toFinset_card_le _
  have hdisj : Disjoint T₁ T₂ := by
    rw [Finset.disjoint_left]
    intro t h1 h2
    exact hcross t h1 t h2 fun _ _ => rfl
  rw [Finset.card_union_of_disjoint hdisj] at hc1
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le hca
  have e1 := Nat.mul_le_mul_left c hc1
  have e2 := Nat.mul_le_mul_left d hc2
  nlinarith [e1, e2, hcost]

/-- **`FloorTc` below `H32` is satisfiable.** -/
theorem floorTcSat : FloorTcSat := by
  refine ⟨⟨fun s => by cases s <;> decide, by decide, by decide, by decide, by decide, by decide⟩, fun h => ?_⟩
  have : 32 ≤ h100Legacy.tc := h.2.1
  exact absurd this (by decide)

theorem cheapIn_empty {c : ℕ} {pr : Prices100} (hpr : Floor c pr) {F : Set (G → ℚ)} :
    ∀ o : Op100 G, o.WFw pr F → o.CheapIn c ∅ := by
  intro o hwo
  cases o with
  | mma outs p =>
    intro hp
    exact absurd (le_trans (Nat.mul_le_mul_right _ hpr.2.1) hwo) (by omega)
  | _ => trivial

/-- **The distinct-writes bound at any floor.** -/
theorem distinctWritesMH100Floor : DistinctWritesMH100Floor := by
  intro c pr hpr G F hconst S fpSem L P T hL hwf hd ho
  have h := distinctWritesLegacy c c pr hpr le_rfl G F ∅ hconst S fpSem L P T ∅ hL hwf
    (fun o ho => cheapIn_empty hpr o (hwf o ho)) hd ⟨by simp, by simp⟩ (by simp) (by simp) ho (by simp [OutputsOn])
  simpa using h

/-- **The distinct-writes bound on a success set.** -/
theorem distinctWritesMH100On : DistinctWritesMH100On := fun pr hpr => distinctWritesMH100Floor 32 pr hpr

/-- **The distinct-writes bound for `MH100w`.** -/
theorem distinctWritesMH100 : DistinctWritesMH100 := by
  intro pr hpr G F hconst fpSem L P T hL hwf hT ho
  refine distinctWritesMH100On pr hpr G F hconst Set.univ fpSem L P T hL hwf ⟨fun t _ t' _ h => ?_, ?_⟩ ?_
  · exact funext fun g => h g (Set.mem_univ g)
  · intro t ht f hf h
    exact hT t ht (by rwa [show t = f from funext fun g => h g (Set.mem_univ g)])
  · exact fun t ht => ⟨t, ho t ht, fun _ _ => rfl⟩

theorem mem_regs100_of_operand (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) :
    ∀ (P : List (Op100 G)) (L : List (G → ℚ)) (f : G → ℚ), Op100.operand f ∈ P → f ∈ regs100 fpSem P L := by
  intro P
  induction P with
  | nil => intro L f h; simp at h
  | cons o P ih =>
    intro L f h
    rcases List.mem_cons.1 h with rfl | h
    · exact mem_regs100_of_mem fpSem P _ f (List.mem_append_right _ (by simp [Op100.outs]))
    · exact ih _ f h

/-- **The distinct-writes bound is attained.** -/
theorem distinctWritesAttained : DistinctWritesAttained := by
  intro G F fpSem T Fr L hFr
  let P₁ : List (Op100 G) := T.toList.map fun t => Op100.wgmma Shape100.e4m3F32 [t]
  let P₂ : List (Op100 G) := Fr.map Op100.operand
  have hcost₁ : ∀ l : List (G → ℚ),
      cost100 h100Nominal (l.map fun t => Op100.wgmma Shape100.e4m3F32 [t]) = 32 * l.length := by
    intro l
    induction l with
    | nil => rfl
    | cons t l ih =>
      simp only [cost100, List.map_cons, List.sum_cons, List.length_cons] at ih ⊢
      rw [ih]
      simp only [Op100.price, Shape100.regPrice, h100Nominal, nominalMac, Shape100.k, Shape100.elems,
        List.length_singleton]
      ring
  have hcost₂ : ∀ l : List (G → ℚ), cost100 h100Nominal (l.map Op100.operand) = 0 := by
    intro l
    induction l with
    | nil => rfl
    | cons t l ih =>
      simp only [cost100, List.map_cons, List.sum_cons] at ih ⊢
      rw [ih]
      rfl
  refine ⟨P₁ ++ P₂, fun o ho => ?_, fun t ht => ?_, fun f hf => ?_, ?_⟩
  · rcases List.mem_append.1 ho with ho | ho
    · obtain ⟨t, -, rfl⟩ := List.mem_map.1 ho
      trivial
    · obtain ⟨f, hf, rfl⟩ := List.mem_map.1 ho
      exact hFr f hf
  · exact mem_regs100_of_wgmma fpSem (P₁ ++ P₂) _ _ _ _
      (List.mem_append_left _ (List.mem_map.2 ⟨t, mem_toList.2 ht, rfl⟩)) (List.mem_singleton_self _)
  · exact mem_regs100_of_operand fpSem (P₁ ++ P₂) L f (List.mem_append_right _ (List.mem_map.2 ⟨f, hf, rfl⟩))
  · rw [cost100_append, hcost₁, hcost₂, length_toList, Nat.add_zero]

theorem wfw_splitK (F : Set (G → ℚ)) (h0 : (0 : G → ℚ) ∈ F) (κ : ℕ) (I : ℕ → G → ℚ) (b : ℕ) :
    ∀ τ, ∀ o ∈ splitK κ I b τ, o.WFw h100Nominal F
  | 0 => fun o ho => by
    rw [splitK, List.mem_singleton] at ho
    subst ho
    exact h0
  | τ + 1 => fun o ho => by
    rw [splitK] at ho
    rcases List.mem_append.1 ho with h | h
    · exact wfw_splitK F h0 κ I b τ o h
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at h
      rcases h with rfl | rfl <;> trivial

/-- **The C-2 corollary.** -/
theorem c2DistinctWords : C2DistinctWords := by
  intro G F hconst fpSem κ I T L
  refine ⟨wfw_splitK F (const_zero hconst) κ I L.length T, ?_⟩
  intro pr P hpr hL hwf hout D hD
  refine distinctWritesMH100 pr hpr G F hconst fpSem L P D hL hwf (fun w hw => (hD w hw).2) fun w hw => ?_
  obtain ⟨τ, hτ, h | h⟩ := (hD w hw).1
  · exact h ▸ (hout τ hτ).1
  · exact h ▸ (hout τ hτ).2

/-- **The first running word is a copy.** -/
theorem c2FirstRunCopy : C2FirstRunCopy := by
  intro G fpSem κ I h
  funext g
  simp only [runWords, Pi.zero_apply, h]

/-- **`DistinctLive` gives the per-word cost.** -/
theorem distinctLiveMH100 : DistinctLiveMH100 := by
  intro pr hpr G F hconst ε S ts fpSem L P hlive hL hwf hout
  obtain ⟨D, hDts, hD, hcard⟩ := hlive
  have h := distinctWritesMH100On pr hpr G F hconst S fpSem L P D hL hwf hD fun t ht => hout t (hDts t ht)
  have h' : (32 * D.card : ℚ) ≤ cost100 pr P := by exact_mod_cast h
  nlinarith [hcard, h']

end Pouw.Dimension.H100
