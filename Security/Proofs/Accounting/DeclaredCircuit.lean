import Mathlib
import Models.Accounting.Guarantees.DeclaredCircuit
import Proofs.Accounting.Stacking

/-! The proof of `Accounting.Guarantees.DeclaredCircuitRequirement`, from its pieces: a carrier of a declared
circuit's is within the stacking property's budgets (`PassProvenance.OfCircuit.budget`), a uniform function restricted
along an injection is uniform (`pr_comp_embedding`), key binding moves a store whose layout is steered after the coins
to uniform blocks at `2^a` times their event (`KeyBinding.pr_steered_le`), and the stacked relay session on a steered
layout (`stackedRelay_steered`). -/

namespace Accounting.Rate.PassProvenance

/-- **A carrier of a declared circuit's is within the property's budgets**: over at most `life` of the audit's windows,
its inputs are at most `8·R_in·life` bits and its encodes at most `c·R̂·life` budgeted units, `R̂` the most bytes the
audit certified in a window. -/
theorem OfCircuit.budget {N b h ρ : ℕ} {α ι : Type} {K : PassProvenance N b h ρ α} {cost : ℕ → ℕ∞}
    {W : Gates.Dag ι} {caps : Gates.Caps} {A : Gates.Audit} {I : Finset ℕ} {life : ℕ} (hK : K.OfCircuit cost W A I)
    (hP : Gates.Property caps A W) (hI : I ⊆ Finset.range A.windows) (hcard : I.card ≤ life) (E : Store N b)
    (R : Pous.Bits ρ) (O : Oracle N b h ρ) :
    K.inputs E R O ≤ 8 * caps.runBytes * life ∧
      K.gates cost E R O ≤ (caps.total * (Finset.range A.windows).sup A.reads * life : ℕ) := by
  have hw : ∀ v ∈ I, (Gates.census W fun t => A.win t = v).runBits ≤ 8 * caps.runBytes ∧
      (Gates.census W fun t => A.win t = v).budgeted ≤ caps.total * (Finset.range A.windows).sup A.reads := by
    intro v hv
    have h := hP.2.1 v (Finset.mem_range.mp (hI hv))
    dsimp only at h
    obtain ⟨h1, h2, h3, h4, h5, h6⟩ := h
    have hr : A.reads v ≤ (Finset.range A.windows).sup A.reads := Finset.le_sup (hI hv)
    refine ⟨h6, ?_⟩
    calc (Gates.census W fun t => A.win t = v).budgeted
        = (Gates.census W fun t => A.win t = v).rounding + (Gates.census W fun t => A.win t = v).compareSelect +
            (Gates.census W fun t => A.win t = v).gather + (Gates.census W fun t => A.win t = v).fold +
            (Gates.census W fun t => A.win t = v).neutral := rfl
      _ ≤ caps.rounding * A.reads v + caps.compareSelect * A.reads v + caps.gather * A.reads v +
            caps.fold * A.reads v + 0 := by rw [h5]; gcongr
      _ = caps.total * A.reads v := by simp only [Gates.Caps.total]; ring
      _ ≤ _ := Nat.mul_le_mul le_rfl hr
  refine ⟨(hK E R O).1.trans ?_, (hK E R O).2.trans ?_⟩
  · calc _ ≤ ∑ _v ∈ I, 8 * caps.runBytes := Finset.sum_le_sum fun v hv => (hw v hv).1
      _ = I.card * (8 * caps.runBytes) := by rw [Finset.sum_const, smul_eq_mul]
      _ ≤ life * (8 * caps.runBytes) := Nat.mul_le_mul_right _ hcard
      _ = _ := by ring
  · have : ∑ v ∈ I, (Gates.census W fun t => A.win t = v).budgeted ≤
        caps.total * (Finset.range A.windows).sup A.reads * life := by
      calc _ ≤ ∑ _v ∈ I, caps.total * (Finset.range A.windows).sup A.reads := Finset.sum_le_sum fun v hv => (hw v hv).2
        _ = I.card * (caps.total * (Finset.range A.windows).sup A.reads) := by rw [Finset.sum_const, smul_eq_mul]
        _ ≤ life * (caps.total * (Finset.range A.windows).sup A.reads) := Nat.mul_le_mul_right _ hcard
        _ = _ := by ring
    exact_mod_cast this

end Accounting.Rate.PassProvenance

namespace Accounting

open Pous (pr)
open scoped ENNReal

/-- **A uniform function restricted along an injection is uniform**: an event of `F ∘ π`, for a uniform `F` on `π`'s
codomain, is as likely as the event of a uniform function on its domain. -/
theorem pr_comp_embedding {α β B : Type} [Fintype α] [DecidableEq α] [Fintype β] [DecidableEq β] [Fintype B] [Nonempty B]
    (π : α ↪ β) (Q : (α → B) → Prop) : pr (fun F : β → B => Q (F ∘ π)) = pr Q := by
  classical
  let p : β → Prop := fun x => x ∈ Set.range π
  let e : (β → B) ≃ (α → B) × ({x // ¬ p x} → B) :=
    (Equiv.piEquivPiSubtypeProd p fun _ => B).trans
      (Equiv.prodCongr (Equiv.arrowCongr (Equiv.ofInjective π π.injective).symm (Equiv.refl B)) (Equiv.refl _))
  have he : ∀ F, (e F).1 = F ∘ π := fun F => by
    funext x
    rfl
  have hs : ∀ y, e.symm y ∘ π = y.1 := fun y => by rw [← he, Equiv.apply_symm_apply]
  rw [Rate.pr_equiv e (fun F => Q (F ∘ π))]
  simp only [hs]
  rw [Assumptions.pr_prod_eq_sum_div (fun y : (α → B) × ({x // ¬ p x} → B) => Q y.1)]
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  rw [mul_comm, ENNReal.mul_div_cancel_right (by exact_mod_cast Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)]

/-- `pr_comp_embedding` beside independent coins. -/
theorem pr_comp_embedding_prod {α β B C : Type} [Fintype α] [DecidableEq α] [Fintype β] [DecidableEq β] [Fintype B]
    [Nonempty B] [Fintype C] (π : α ↪ β) (Q : (α → B) → C → Prop) :
    pr (fun y : (β → B) × C => Q (y.1 ∘ π) y.2) = pr (fun y : (α → B) × C => Q y.1 y.2) := by
  rw [Assumptions.pr_prod_eq_sum_div (fun y : (β → B) × C => Q (y.1 ∘ π) y.2),
    Assumptions.pr_prod_eq_sum_div (fun y : (α → B) × C => Q y.1 y.2)]
  congr 1
  exact Finset.sum_congr rfl fun c _ => pr_comp_embedding π (fun E => Q E c)

/-- **Key binding with a layout steered after the coins**: when the layout is one of `2^a`, chosen by any function
`steer` of the codec and the coins, an event of the blocks and of independent coins is at most `2^a` times as likely
as on uniform blocks, plus `δ`. Key binding applies once, at the whole layout, and the union over the `2^a` layouts is
taken inside the uniform store, so `δ` is not multiplied. -/
theorem Assumptions.KeyBinding.pr_steered_le {Ω Ca Ch B : Type} [Fintype Ω] [Fintype Ca] [Fintype B] [Nonempty B]
    {T M N a : ℕ} {δ : ℝ≥0∞} {enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → B}
    (h : Assumptions.KeyBinding T M δ enc) (choose : Ω → (a : Fin T) → ({b : Fin T // b < a} → Ca) → Ch)
    (slot : Pous.Bits a → Fin N ↪ Fin T × Fin M) (steer : Ω × (Fin T → Ca) → Pous.Bits a) {C : Type} [Fintype C]
    (Q : (Fin N → B) → C → Prop) :
    pr (fun x : (Ω × (Fin T → Ca)) × C => Q (appendCoinBlocks enc choose (slot (steer x.1)) x.1) x.2)
      ≤ 2 ^ a * pr (fun y : (Fin N → B) × C => Q y.1 y.2) + δ := by
  classical
  let full : Fin (T * M) ↪ Fin T × Fin M := finProdFinEquiv.symm.toEmbedding
  let π : Pous.Bits a → Fin N ↪ Fin (T * M) := fun s => (slot s).trans finProdFinEquiv.toEmbedding
  have hfull : ∀ p, full (finProdFinEquiv p) = p := fun p => finProdFinEquiv.symm_apply_apply p
  have hblk : ∀ s y, appendCoinBlocks enc choose (slot s) y = appendCoinBlocks enc choose full y ∘ π s :=
    fun s y => by
      funext u
      show _ = appendCoinBlocks enc choose full y (finProdFinEquiv (slot s u))
      unfold appendCoinBlocks
      rw [hfull]
  have h1 := (h.uniformBut_blocks choose full).pr_le () (fun F c => ∃ s, Q (F ∘ π s) c)
  calc pr (fun x : (Ω × (Fin T → Ca)) × C => Q (appendCoinBlocks enc choose (slot (steer x.1)) x.1) x.2)
      ≤ pr (fun x : (Ω × (Fin T → Ca)) × C => ∃ s, Q (appendCoinBlocks enc choose full x.1 ∘ π s) x.2) :=
        Bandwidth.pr_mono fun x hx => ⟨steer x.1, by rw [← hblk]; exact hx⟩
    _ ≤ pr (fun y : (Fin (T * M) → B) × C => ∃ s, Q (y.1 ∘ π s) y.2) + δ := h1
    _ ≤ (∑ s, pr (fun y : (Fin (T * M) → B) × C => Q (y.1 ∘ π s) y.2)) + δ :=
        add_le_add (Bandwidth.pr_exists_le _) le_rfl
    _ = (∑ _s : Pous.Bits a, pr (fun y : (Fin N → B) × C => Q y.1 y.2)) + δ := by
        congr 1
        exact Finset.sum_congr rfl fun s _ => pr_comp_embedding_prod (π s) Q
    _ = 2 ^ a * pr (fun y : (Fin N → B) × C => Q y.1 y.2) + δ := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
        simp [Pous.Bits]

end Accounting

namespace Accounting.SecurityProofs

open Pous (pr)
open scoped ENNReal

/-- **The stacked relay session on a layout steered after the coins**: `StackedSessionConservationRelay`'s bound, with
the carriers' bits `Cc` given directly, for a store whose layout is one of `2^a` chosen by any function of the codec
and the coins. The steering is charged as `a` bits of side capacity: the unpaid event is taken at slack `λ + a`, which
`2^a` turns back into `λ` in its query term and adds to the exponent of its relay term. -/
theorem stackedRelay_steered {n P N b h ρ S Lall L q r s d G lam m τ rounds cRoot F Cc a T M : ℕ} {cost : ℕ → ℕ}
    (tier : ℕ → Fin τ) {δ : ℝ≥0∞} {Ω Ca Ch : Type} [Fintype Ω] [Fintype Ca]
    {enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → Pous.Bits b}
    (choose : Ω → (a : Fin T) → ({c : Fin T // c < a} → Ca) → Ch) (slot : Pous.Bits a → Fin N ↪ Fin T × Fin M)
    (steer : Ω × (Fin T → Ca) → Pous.Bits a) (A : Rate.DeployedSession n P N b h ρ S Lall q r s d G) (dbar dw : ℝ)
    (hn : 0 < n) (hP : 0 < P) (hN : 0 < N) (hb : 0 < b) (hrc : 0 < rounds * cRoot) (hdw : dw ≤ 1)
    (hKB : Assumptions.KeyBinding T M δ enc) (hW : Assumptions.WorkHardness b rounds cRoot cost)
    (hF : ∀ w η j ι c, Assumptions.SquaringCeiling cost F ((A.window w η).device j ι c))
    (hK : ∀ w η j ι c, ∃ K : Rate.PassProvenance N b h ρ (ℕ → Bool),
      (∀ Es R O v i, ((A.window w η).device j ι c).side Es R O v (2 * i + 1) = K.run Es R O i) ∧
      (∀ st wd, (((A.window w η).device j ι c).chip st).linkBitsEven wd [] ≤ L) ∧
      ∀ Es R O, K.inputs Es R O + K.encoded Es R O ≤ Cc) :
    pr (fun x : (Ω × (Fin T → Ca)) × (Fin n → Rate.WindowCoins P N b h ρ r s d) =>
        (∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll (appendCoinBlocks enc choose (slot (steer x.1)) x.1) (x.2 w).1) ∧
          ((((∑ w, (A.win w x.2).reads tier (appendCoinBlocks enc choose (slot (steer x.1)) x.1) (x.2 w).1 : ℕ)) : ℝ) <
              (1 - (dbar + Rate.Deployed.gammaPass N b ρ S (L + Cc) (lam + a + m * b) rounds cRoot F)) *
                (n * P * N * b) ∨
            ∃ w, (((A.win w x.2).reads tier (appendCoinBlocks enc choose (slot (steer x.1)) x.1) (x.2 w).1 : ℕ) : ℝ) <
              (1 - (dw + Rate.Deployed.gammaPass N b ρ S (L + Cc) (lam + a + m * b) rounds cRoot F)) * (P * N * b)))
      ≤ ENNReal.ofReal (1 - dbar) ^ (d * n) + ENNReal.ofReal (1 - dw / n) ^ (d * n) +
        (n * P * G : ℝ≥0∞) * ((N.choose m : ℝ≥0∞) * 2 ^ (L + Cc + a) / 2 ^ (h * m) + (2 ^ q : ℝ≥0∞) / 2 ^ lam) +
          δ := by
  choose K hside hL hC using hK
  have hu := hKB.pr_steered_le choose slot steer
    fun (Es : Rate.Store N b) (ω : Fin n → Rate.WindowCoins P N b h ρ r s d) =>
      ∃ w, ((A.stack L Cc K hL).win w ω).reads tier Es (ω w).1 +
          P * (S + (L + Cc - ρ) + Rate.Deployed.rebuildBits b rounds cRoot F + (lam + a + m * b)) <
        (((A.stack L Cc K hL).win w ω).correctAll Es (ω w).1).card * b
  have hunif := Rate.DeployedSession.pr_unpaid_le (A.stack L Cc K hL) tier _ _
    (Remote.window_relay_le (lam := lam + a) (m := m) (A.stack L Cc K hL) tier hrc hW
      (fun w η j ι c => Rate.PIDevice.stack_ceiling _ L Cc (K w η j ι c) (hL w η j ι c) (hF w η j ι c)))
  have base := Remote.session_relay_of_unpaid (fun y => appendCoinBlocks enc choose (slot (steer y)) y)
    (A.stack L Cc K hL) tier (lam + a) m rounds cRoot F dbar dw hn hP hN hb hdw _
    (hu.trans (add_le_add (mul_le_mul_right hunif _) le_rfl))
  refine ((Bandwidth.pr_mono fun y hy => ?_).trans base).trans (le_of_eq ?_)
  · have e := fun w => Rate.DeployedSession.stack_win A L Cc K hL hside hC tier w y.2
      (appendCoinBlocks enc choose (slot (steer y.1)) y.1) (y.2 w).1
    simp only [fun w => (e w).1, fun w => (e w).2]
    exact hy
  · have h2 : (2 : ℝ≥0∞) ^ a * ((2 : ℝ≥0∞) ^ (lam + a))⁻¹ = ((2 : ℝ≥0∞) ^ lam)⁻¹ := by
      rw [pow_add, ENNReal.mul_inv (Or.inl (pow_ne_zero _ two_ne_zero)) (Or.inl (ENNReal.pow_ne_top ENNReal.ofNat_ne_top)),
        mul_comm ((2 : ℝ≥0∞) ^ lam)⁻¹, ← mul_assoc,
        ENNReal.mul_inv_cancel (pow_ne_zero _ two_ne_zero) (ENNReal.pow_ne_top ENNReal.ofNat_ne_top), one_mul]
    simp only [div_eq_mul_inv]
    rw [← h2, pow_add (2 : ℝ≥0∞) (L + Cc) a]
    ring

/-- **A declared circuit's live and read bits from an accepted stacked audit**,
`Accounting.Guarantees.DeclaredCircuitRequirement`. -/
theorem DeclaredCircuitRequirement : Accounting.Guarantees.DeclaredCircuitRequirement := by
  sorry

end Accounting.SecurityProofs
