import Proofs.Pouw.PearlC.RowSeedGamma

/-!
# The code-form seed (`-h3`): a relabelling of the one-row-unit form (staging, nothing pinned)

The code seeds row `i` of unit `u` at the index `u·2³² + i`, with
`leaf_i` framed inside unit `u`; C1 (`RowSeedNoise`) seeds it as the one-row unit `rowIdx L u i` of the split workload.
A fixed permutation of the oracle's inputs, one per layout, carries one to the other (C1-h3, `CodeSeedNoise`).

* **A uniform oracle doesn't see a fixed relabelling:** `pr_comp_equiv`, the probability of an event of `(H ∘ π, s)`
  is that of `(H, s)`.
* **Nor does a bounded adversary:** its preprocessing relabels query by query (`relabelTree`, `eval_relabelTree`,
  `calls_relabelTree`), and C5 (`CostModel.RelabelClosed`) relabels its online program, so it runs against `H` as the
  original runs against `H ∘ π⁻¹`, within the same budget (`exists_relabel_adversary`).
* **The bridge:** `ttOutRowSeed_code` and `ttOutTileRowSeed_code` carry the row-seeded TT_OUT from `sem'` to the
  code's `semC`, at the same error `εPearlC q R`, so `(q + R)/2^128`.
* **From rev1's TT_OUT:** `ttOutRowSeedCode_of_ttOut` and `ttOutTileRowSeedCode_of_ttOutTile` (C1, C1-h3, C3, C5). The
  γ theorems `pearlCGammaDevRev1RowSeedAt` and `pearlCSampledDevRev1RowSeedAt` apply at `semC` unchanged.
-/

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

variable {Q R S : Type}

/-! ## Relabelling -/

/-- A query tree with every query relabelled by `π`. -/
def relabelTree {α : Type} (π : Q → Q) : QTree Q R α → QTree Q R α
  | .done a => .done a
  | .query q k => .query (π q) fun r => relabelTree π (k r)

theorem eval_relabelTree {α : Type} (π : Q → Q) (H : Q → R) (t : QTree Q R α) :
    (relabelTree π t).eval H = t.eval (H ∘ π) := by
  induction t with
  | done a => rfl
  | query q k ih => exact ih (H (π q))

theorem calls_relabelTree {α : Type} (π : Q → Q) (H : Q → R) (t : QTree Q R α) :
    (relabelTree π t).calls H = t.calls (H ∘ π) := by
  induction t with
  | done a => rfl
  | query q k ih => exact congrArg (· + 1) (ih (H (π q)))

/-- The noise relabelled: `sem` at `H ∘ π L`. -/
def PearlCSem.relabel (sem : PearlCSem Q R S) (π : UnitLayout → Q ≃ Q) : PearlCSem Q R S :=
  { sem with noise := fun H s U u act => sem.noise (H ∘ π U.layout) s U u act }

theorem eq_relabel {sem' semC : PearlCSem Q R S} {π : UnitLayout → Q ≃ Q} (h : CodeSeedNoise sem' semC π) :
    semC = sem'.relabel π := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  cases semC
  cases sem'
  simp only [PearlCSem.relabel, PearlCSem.mk.injEq] at h1 h2 h3 h4 ⊢
  exact ⟨funext fun H => funext fun s => funext fun U => funext fun u => funext fun act => h5 H s U u act,
    h1, h2, h3, h4⟩

/-- **A uniform oracle's probabilities are unchanged under a fixed permutation of its inputs.** -/
theorem pr_comp_equiv [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (π : Q ≃ Q)
    (P : (Q → R) × S → Prop) : pr (fun x : (Q → R) × S => P (x.1 ∘ π, x.2)) = pr P := by
  classical
  let e : (Q → R) × S ≃ (Q → R) × S :=
    { toFun := fun x => (x.1 ∘ π, x.2)
      invFun := fun x => (x.1 ∘ π.symm, x.2)
      left_inv := fun x => by ext q <;> simp
      right_inv := fun x => by ext q <;> simp }
  unfold pr
  congr 2
  exact Finset.card_equiv e fun x => by simp [e]

/-! ## The bridge -/

/-- The adversary relabelled by `π⁻¹`: its preprocessing's queries relabelled, and its online program replaced by the
one C5 gives, so that against `H` it commits what the original commits against `H ∘ π⁻¹`. -/
theorem exists_relabel_adversary {CM : CostModel Q R S} (hCM : CM.RelabelClosed) (π : Q ≃ Q) (A : Adversary CM)
    (U : Workload) {T : ℝ} {q : ℕ} (hA : A.Bounded U T q) :
    ∃ A' : Adversary CM, A'.Bounded U T q ∧
      ∀ (H : Q → R) (s : S), A'.transcript U H s = A.transcript U (H ∘ π.symm) s := by
  obtain ⟨prog', hrun⟩ := hCM π.symm A.online U A.act
  refine ⟨⟨relabelTree π.symm A.pre, prog', A.act⟩, fun H s => ?_, fun H s => ?_⟩
  · have hb := hA (H ∘ π.symm) s
    have hr := hrun H ((relabelTree π.symm A.pre).eval H) s
    rw [eval_relabelTree] at hr
    refine ⟨?_, ?_⟩
    · show CM.cost prog' H U A.act ((relabelTree π.symm A.pre).eval H) s ≤ T
      rw [eval_relabelTree]; exact hr.2.1.trans hb.1
    · show (relabelTree π.symm A.pre).calls H + CM.calls prog' H U A.act ((relabelTree π.symm A.pre).eval H) s ≤ q
      rw [eval_relabelTree, calls_relabelTree]
      exact le_trans (Nat.add_le_add_left hr.2.2 _) hb.2
  · show CM.run prog' H U A.act ((relabelTree π.symm A.pre).eval H) s =
      CM.run A.online (H ∘ π.symm) U A.act (A.pre.eval (H ∘ π.symm)) s
    rw [eval_relabelTree]; exact (hrun H _ s).1

theorem comp_symm_comp (π : Q ≃ Q) (H : Q → R) : (H ∘ π) ∘ π.symm = H := by
  funext q; simp

section Bridge

variable [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]

/-- **The code-form seed bridge** (`-h3`, §14): under C1-h3 and C5, the row-seeded TT_OUT at the one-row-unit form
`sem'` gives it at the code's form `semC`, at the same error `εPearlC q R`. A bounded adversary against `semC` is,
relabelled by `π⁻¹`, one against `sem'` that wins on `H ∘ π` exactly when it wins on `H`, and `H ∘ π` is uniform. -/
theorem ttOutRowSeed_code {CM : CostModel Q R S} {d : PearlCDevice} {sem' semC : PearlCSem Q R S}
    {π : UnitLayout → Q ≃ Q} {ρ : ℚ} {D : UnitLayout → Prop} {γ₀ : ℝ}
    (h : TTOutRowSeed CM (pearlCProtocolDevRev1RowCap d sem' ρ) D γ₀) (hC : CodeSeedNoise sem' semC π)
    (hCM : CM.RelabelClosed) : TTOutRowSeed CM (pearlCProtocolDevRev1RowCap d semC ρ) D γ₀ := by
  obtain rfl := eq_relabel hC
  refine ⟨h.1, fun U hU hR T hT q hq A hA => ?_⟩
  obtain ⟨A', hA', htr⟩ := exists_relabel_adversary hCM (π U.layout) A U hA
  have key := h.2 U hU hR T hT q hq A' hA'
  rw [← pr_comp_equiv (π U.layout)] at key
  refine le_of_eq_of_le ?_ key
  congr 1
  funext x
  rw [htr, comp_symm_comp]
  rfl

/-- **The code-form seed bridge per audit tile**, as `ttOutRowSeed_code`. -/
theorem ttOutTileRowSeed_code {CM : CostModel Q R S} {d : PearlCDevice} {sem' semC : PearlCSem Q R S}
    {π : UnitLayout → Q ≃ Q} {ρ : ℚ} {D : UnitLayout → Prop} {γ₀ : ℝ}
    (h : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowCap d sem' ρ) (pearlCTilesDevRev1RowCap d sem' ρ) D γ₀)
    (hC : CodeSeedNoise sem' semC π) (hCM : CM.RelabelClosed) :
    TTOutTileRowSeed CM (pearlCProtocolDevRev1RowCap d semC ρ) (pearlCTilesDevRev1RowCap d semC ρ) D γ₀ := by
  obtain rfl := eq_relabel hC
  refine ⟨h.1, fun U hU hR T hT q hq A hA => ?_⟩
  obtain ⟨A', hA', htr⟩ := exists_relabel_adversary hCM (π U.layout) A U hA
  have key := h.2 U hU hR T hT q hq A' hA'
  rw [← pr_comp_equiv (π U.layout)] at key
  refine le_of_eq_of_le ?_ key
  congr 1
  funext x
  rw [htr, comp_symm_comp]
  rfl

/-- **The code-form seed from rev1's TT_OUT:** C1, C1-h3, C3 and C5 give the row-seeded TT_OUT at the code's form,
at the row count's error. -/
theorem ttOutRowSeedCode_of_ttOut {CM : CostModel Q R S} {d : PearlCDevice} {sem sem' semC : PearlCSem Q R S}
    {π : UnitLayout → Q ≃ Q} {ρ : ℚ} (hTT : TTOutPearlCDevRev1 CM d sem ρ) (hN : RowSeedNoise sem sem')
    (hC : CodeSeedNoise sem' semC π) (hCM : CM.SplitClosed) (hCM' : CM.RelabelClosed) (hd : d.RowLocal) (hρ : ρ ≤ 1)
    (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) :
    TTOutRowSeed CM (pearlCProtocolDevRev1RowCap d semC ρ) (pearlCDomainDev d) (1 / 400) :=
  ttOutRowSeed_code (ttOutRowSeed_of_ttOut hTT hN hCM hd hρ hadd hbf) hC hCM'

/-- **The code-form seed from rev1's TT_OUT per audit tile**, as `ttOutRowSeedCode_of_ttOut`. -/
theorem ttOutTileRowSeedCode_of_ttOutTile {CM : CostModel Q R S} {d : PearlCDevice} {sem sem' semC : PearlCSem Q R S}
    {π : UnitLayout → Q ≃ Q} {ρ : ℚ} (hTT : TTOutTilePearlCDevRev1 CM d sem ρ) (hN : RowSeedNoise sem sem')
    (hC : CodeSeedNoise sem' semC π) (hCM : CM.SplitClosed) (hCM' : CM.RelabelClosed) (hd : d.RowLocal) (hρ : ρ ≤ 1)
    (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) :
    TTOutTileRowSeed CM (pearlCProtocolDevRev1RowCap d semC ρ) (pearlCTilesDevRev1RowCap d semC ρ) (pearlCDomainDev d)
      (1 / 400) :=
  ttOutTileRowSeed_code (ttOutTileRowSeed_of_ttOutTile hTT hN hCM hd hρ hadd hbf) hC hCM'

end Bridge

end Pouw.PearlC
