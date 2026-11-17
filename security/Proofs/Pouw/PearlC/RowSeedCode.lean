module

public import Definitions.Pouw.Assumptions.PearlC.TTOutRev1
public import Definitions.Pouw.Assumptions.PearlC.TTOutRowSeed
public import Mathlib.Tactic.Continuity
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset
public import Proofs.Pouw.PearlC.RowSeedGap

/-! The code-form seed (`-h3`), staged: a relabelling of the one-row-unit form. -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

variable {Q R S : Type}

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

def PearlCSem.relabel (sem : PearlCSem Q R S) (π : Layout → Q ≃ Q) : PearlCSem Q R S :=
  { sem with noise := fun H s U u act => sem.noise (H ∘ π U.layout) s U u act }

theorem eq_relabel {sem' semC : PearlCSem Q R S} {π : Layout → Q ≃ Q} (h : CodeSeedNoise sem' semC π) :
    semC = sem'.relabel π := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  cases semC
  cases sem'
  simp only [PearlCSem.relabel, PearlCSem.mk.injEq] at h1 h2 h3 h4 ⊢
  exact ⟨funext fun H => funext fun s => funext fun U => funext fun u => funext fun act => h5 H s U u act,
    h1, h2, h3, h4⟩

theorem pr_comp_equiv [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (π : Q ≃ Q)
    (P : (Q → R) × S → Prop) : pr (fun x : (Q → R) × S => P (x.1 ∘ π, x.2)) = pr P := by
  classical
  let e : (Q → R) × S ≃ (Q → R) × S :=
    { toFun := fun x => (x.1 ∘ π, x.2)
      invFun := fun x => (x.1 ∘ π.symm, x.2)
      left_inv := fun x => by ext q <;> simp
      right_inv := fun x => by ext q <;> simp }
  unfold pr Claims.Pr Claims.prUniform
  congr 2
  exact Finset.card_equiv e fun x => by simp [e]

theorem exists_relabel_adversary {CM : CostModel Q R S} (hCM : CM.RelabelClosed) (π : Q ≃ Q) (A : Adversary CM)
    (U : Workload) {T : ℝ} {q : ℕ} (hA : A.Bounded U T q) :
    ∃ A' : Adversary CM, A'.Bounded U T q ∧
      ∀ (H : Q → R) (s : S), A'.trace U H s = A.trace U (H ∘ π.symm) s := by
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

theorem ttOutRowSeed_code {CM : CostModel Q R S} {d : PearlCDevice} {sem' semC : PearlCSem Q R S}
    {π : Layout → Q ≃ Q} {ρ : ℚ} {D : Layout → Prop} {g₀ : ℝ}
    (h : TTOutRowSeed CM (pearlCProtocolDevRev1RowInertBound d sem' ρ) D g₀) (hC : CodeSeedNoise sem' semC π)
    (hCM : CM.RelabelClosed) : TTOutRowSeed CM (pearlCProtocolDevRev1RowInertBound d semC ρ) D g₀ := by
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

theorem ttOutTileRowSeed_code {CM : CostModel Q R S} {d : PearlCDevice} {sem' semC : PearlCSem Q R S}
    {π : Layout → Q ≃ Q} {ρ : ℚ} {D : Layout → Prop} {g₀ : ℝ}
    (h : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowInertBound d sem' ρ) (pearlCTilesDevRev1RowInertBound d sem' ρ) D g₀)
    (hC : CodeSeedNoise sem' semC π) (hCM : CM.RelabelClosed) :
    TTOutTileRowSeed CM (pearlCProtocolDevRev1RowInertBound d semC ρ) (pearlCTilesDevRev1RowInertBound d semC ρ) D g₀ := by
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

theorem ttOutRowSeedCode_of_ttOut {CM : CostModel Q R S} {d : PearlCDevice} {sem sem' semC : PearlCSem Q R S}
    {π : Layout → Q ≃ Q} {ρ : ℚ} (hTT : TTOutPearlCDevRev1 CM d sem ρ) (hN : RowSeedNoise sem sem')
    (hC : CodeSeedNoise sem' semC π) (hCM : CM.SplitClosed) (hCM' : CM.RelabelClosed) (hd : d.RowLocal) (hρ : ρ ≤ 1)
    (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) :
    TTOutRowSeed CM (pearlCProtocolDevRev1RowInertBound d semC ρ) (pearlCDomainDev d) (1 / 400) :=
  ttOutRowSeed_code (ttOutRowSeed_of_ttOut hTT hN hCM hd hρ hadd hbf) hC hCM'

theorem ttOutTileRowSeedCode_of_ttOutTile {CM : CostModel Q R S} {d : PearlCDevice} {sem sem' semC : PearlCSem Q R S}
    {π : Layout → Q ≃ Q} {ρ : ℚ} (hTT : TTOutTilePearlCDevRev1 CM d sem ρ) (hN : RowSeedNoise sem sem')
    (hC : CodeSeedNoise sem' semC π) (hCM : CM.SplitClosed) (hCM' : CM.RelabelClosed) (hd : d.RowLocal) (hρ : ρ ≤ 1)
    (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) :
    TTOutTileRowSeed CM (pearlCProtocolDevRev1RowInertBound d semC ρ) (pearlCTilesDevRev1RowInertBound d semC ρ) (pearlCDomainDev d)
      (1 / 400) :=
  ttOutTileRowSeed_code (ttOutTileRowSeed_of_ttOutTile hTT hN hCM hd hρ hadd hbf) hC hCM'

end Bridge

end Pouw.PearlC
