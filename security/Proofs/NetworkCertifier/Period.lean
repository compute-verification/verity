module

public import Definitions.NetworkCertifier.Guarantees.Period
public import Proofs.NetworkCertifier.PeriodChecked
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.Order.CompletePartialOrder
import Mathlib.RingTheory.Radical.NatInt
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
import Mathlib.Topology.Sheaves.Init

/-!
# The leak through the network certifier over a monitoring period of sampled audits

`Guarantees.PeriodLeakAccepted` is the instance of `Guarantees.PeriodLeakChecked` at no checked input, at a check every
run passes (`periodLeak_of_checked`); `periodLeak` is that at `periodLeakChecked`.

**Each window's bound** is the checked statement's at the period of no windows, whose first part reads none of the
period: the check misses with probability `0` (`checkedWitness_true`), and the leak relation at the network
certifier's rows with a valid run gives the leak relation there (`rLeakCertifier_of_rLeakCertifierChecked`).

**The count.** At a check every run passes, a window that meets `WindowGood` meets `WindowGoodChecked`
(`windowGoodChecked_of_windowGood`), and with no checked input every check pins the rows where the layout fits
(`checkedUnique_empty`), which it does at each window once one secret is good (`fits_of_windowGood`). The record
format reads back under its own schedule as the parameter (`RecordFormat.readsBy_read`), so the registered parameters
are the schedules `Ss`, and each window's operator is the checked statement's, at the windows before it
(`runsBefore`). The checked count is then `#Ss` times the windows' count at `reg`; the statement's `∏_c (cap_c + 1)`
is room above it.

At operators that read no earlier window the statement is the one with each window's operator fixed before the
period (`periodLeak_fixed`).
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling Finset

section Compiled

open Claims.SamplingGeneric (Arg Compiled)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **A window that meets its good event is registered at a layout that fits.** -/
theorem fits_of_windowGood {H : Msg → Msg} {D : Compiled Coin A} {L K S : ℕ}
    {lay : AuditStatement → BoundaryGates L S} {k : AuditStatement → ℕ} {δ : ℝ} {reg : AuditStatement → Finset ℕ}
    {Dj : ℕ} {s₀ : AuditStatement} {ρ : ℕ → Bool} {tops : AuditStatement → List (Opening Bool)} {x : Msg} {w : Wit}
    {o : List (Obs L K S)} (h : WindowGood H D L K S lay k δ reg Dj s₀ ρ tops x w o) :
    (lay s₀).Fits s₀.prog (k s₀) := by
  obtain ⟨-, ⟨-, s₂, hs₂, hfits⟩, s, hs, ⟨hprog, -, hlay, hk, -, -⟩, -⟩ := h
  rw [Option.some_injective _ (hs₂.symm.trans hs), hprog, hlay, hk] at hfits
  exact hfits

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **At a check every run passes, a window's good event is its checked form.** -/
theorem windowGoodChecked_of_windowGood {H : Msg → Msg} {D : Compiled Coin A} {L K S : ℕ}
    {lay : AuditStatement → BoundaryGates L S} {k : AuditStatement → ℕ} {δ : ℝ} {reg : AuditStatement → Finset ℕ}
    {valid : AuditStatement → (ℕ → Bool) → Prop} (hv : ∀ s r, valid s r) {Dj : ℕ} {s₀ : AuditStatement}
    {ρ : ℕ → Bool} {tops : AuditStatement → List (Opening Bool)} {x : Msg} {w : Wit} {o : List (Obs L K S)}
    (h : WindowGood H D L K S lay k δ reg Dj s₀ ρ tops x w o) :
    WindowGoodChecked H D L K S lay k δ reg valid Dj s₀ ρ tops x w o := by
  obtain ⟨hR, hsc, s, hs, hreg, hρ, hcov, hl, hj⟩ := h
  exact ⟨rLeakCertifierChecked_of_rLeakCertifier hv hR, hsc, s, hs, hreg,
    funext fun r => propext ⟨fun _ => hv s₀ r, fun _ => hv s r⟩, hρ, hcov, hl, hj⟩

end Compiled

/-- **`PeriodLeakAccepted` is `PeriodLeakChecked` at no checked input**, at the check every run passes, the record
format's own read-back with the schedule as the parameter, and the same operators. Each window's bound is the checked
statement's first part at the period of no windows, which reads none of the period. -/
theorem periodLeak_of_checked (hsrc : Guarantees.PeriodLeakChecked) : Guarantees.PeriodLeakAccepted := by
  intro Coin _ _ _ A H D hks L K S lay k δ hδ reg Dj n sw Pw hPw ρ op β hβ Conn Cell Srv Dev Zt _
    arr f tap run htrust cap Ss hcap hfit Zs schedOf srvOf hs dec hdz x wit tops hb
  refine ⟨fun m hm Ω _ _ P => ?_, fun htap => ?_⟩
  · let f₀ : RecordFormat Empty Unit (Fin 0 → List (Obs L K S)) :=
      ⟨fun _ _ _ => Fin.elim0, fun _ _ c => c.elim, fun _ => True, fun _ _ _ => funext fun c => c.elim⟩
    have h := (hsrc (n := 0) (Conn := Empty) (Cell := Unit) (Srv := Unit) (Dev := Unit) (Z := Unit) (Par := Unit)
      H D hks L K S lay k δ hδ reg (fun _ => ∅) (fun _ _ => True) (fun _ _ => True) (fun _ => 0) (fun _ => 0) Dj
      Fin.elim0 Fin.elim0 Fin.elim0 ⟨fun i => i.elim0, fun i => i.elim0⟩ (fun i => i.elim0) Fin.elim0 Fin.elim0
      (fun _ c => c.elim) f₀ () (fun _ p => recordingCode f₀ p.1 (fun c => c.elim)) (fun _ => rfl)
      (fun _ _ c => c.elim) (fun _ _ => True) (fun _ _ _ _ _ => funext fun c => c.elim) ∅ ∅
      (fun _ => ⟨0, 0, Empty.elim, Empty.elim⟩) (fun _ => ()) (fun _ => ())
      ⟨fun _ h => h.elim, fun _ h => h.elim⟩ (fun _ => Fin.elim0) (fun _ => Fin.elim0) tops hb).1 m hm
      (checkedWitness_true H D _ m) (checkedWitness_true H D _ m) Ω P
    have hmono : PrBad H D.V (RLeakCertifier H D.commit D.parse D.partition L K S lay k δ
          (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj tops) D.E m P ≤
        PrBad H D.V (RLeakCertifierChecked H D.commit D.parse D.partition L K S lay k δ
          (fun s => AsReg s.prog (lay s) (k s) (reg s)) (fun _ _ => True ∧ True) Dj tops) D.E m P := by
      unfold PrBad
      refine Pr_mono fun s hs => ⟨hs.1, fun hc => hs.2 ?_⟩
      rcases hw : (D.E.eval H P m s.1 s.2.1 s.2.2).1 with _ | w
      · rw [hw] at hc
        simp [Rel.ext] at hc
      · rw [hw] at hc
        exact rLeakCertifier_of_rLeakCertifierChecked hc
    exact (hmono.trans h).trans (le_of_eq (by simp))
  · set G := {z | z ∈ Zs ∧ ∀ i, WindowGood H D L K S lay k δ reg Dj (sw i) (ρ i) tops (x z i) (wit z i)
      ((run tap (schedOf z, srvOf z)).2 i)} with hG
    rcases G.eq_empty_or_nonempty with he | ⟨z₀, hz₀⟩
    · rw [he, Set.encard_empty]
      exact zero_le
    have hfits : ∀ i, (lay (sw i)).Fits (sw i).prog (k (sw i)) := fun i => fits_of_windowGood (hz₀.2 i)
    have h := ((hsrc H D hks L K S lay k δ hδ reg (fun _ => ∅) (fun _ _ => True) (fun _ _ => True) (fun _ => 0)
      (fun _ => 0) Dj sw Pw β ⟨hPw, hβ⟩ (fun i => checkedUnique_empty (Pw i) (sw i).design (hfits i) _) ρ
      op arr f tap run htrust f.read (fun θ s => θ = s) (RecordFormat.readsBy_read f) Ss Zs schedOf
      srvOf dec ⟨fun z hz => ⟨hfit _ (hs z hz), schedOf z, hs z hz, rfl⟩, hdz⟩ x wit tops hb).2 htap).2
    have hcap1 : 0 < ∏ c, (cap c + 1) := Finset.prod_pos fun c _ => Nat.succ_pos _
    refine (Set.encard_le_encard fun z hz => ?_).trans (h.trans (mul_le_mul' le_rfl ?_))
    · exact ⟨hz.1, fun i => windowGoodChecked_of_windowGood (fun _ _ => ⟨trivial, trivial⟩) (hz.2 i)⟩
    · simp only [windowsCount, Finset.union_empty]
      exact_mod_cast Nat.le_mul_of_pos_left _ hcap1

/-- **The leak through the network certifier over a monitoring period**: `Guarantees.PeriodLeakAccepted`, the instance
of `periodLeakChecked` at no checked input (`periodLeak_of_checked`). -/
theorem periodLeak : Guarantees.PeriodLeakAccepted := periodLeak_of_checked periodLeakChecked

open Claims.SamplingGeneric (Arg Compiled) in
set_option linter.unusedVariables false in
/-- **`PeriodLeakAccepted` at operators fixed before the period**: its statement with window `i`'s operator `op i` one
that reads no earlier window, which is `PeriodLeakAccepted` at the operator `fun i _ => op i`, one that ignores the
windows before it. -/
theorem periodLeak_fixed (h : Guarantees.PeriodLeakAccepted) :
    ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
      (H : Msg → Msg) (D : Compiled Coin A) (hks : D.KS H) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S)
      (k : AuditStatement → ℕ) (δ : ℝ) (hδ : 0 ≤ δ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ)
      {n : ℕ} (sw : Fin n → AuditStatement) (Pw : Fin n → Partition) (hPw : ∀ i, D.partition (sw i) = some (Pw i))
      (ρ : Fin n → ℕ → Bool) (op : Fin n → ℕ → List (Obs L K S) → List Bool) (β : Fin n → ℕ)
      (hβ : ∀ i, ∀ F ∈ faultySets (sw i).design (Pw i) δ, faultyBits (sw i).prog (Pw i) F ≤ β i)
      {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell)
      (f : RecordFormat Conn Cell (Fin n → List (Obs L K S))) (tap : Dev)
      (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
      (htrust : Verity.Assumptions.TrustedHost tap run fun p => recordingCode f p.1 (arr p.2))
      (cap : Conn → ℕ) (Ss : Set (CertifierSchedule Conn)) (hcap : ∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c)
      (hfit : ∀ s ∈ Ss, f.fits s)
      (Zs : Set Z) (schedOf : Z → CertifierSchedule Conn) (srvOf : Z → Srv) (hs : ∀ z ∈ Zs, schedOf z ∈ Ss)
      (dec : (Conn → List (ℕ × CellView Cell)) → Z) (hdz : ∀ z ∈ Zs, dec (run tap (schedOf z, srvOf z)).1 = z)
      (x : Z → Fin n → Msg) (wit : Z → Fin n → Wit) (tops : AuditStatement → List (Opening Bool))
      (hb : D.commit.Binds H),
      (∀ m, ScopeAcc D lay k m → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
        PrBad H D.V
            (RLeakCertifier H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s))
              Dj tops)
            D.E m P ≤
          D.κ δ m + Real.sqrt (D.γ m * PrB H D.V P m) + (Bext H D.V D.E Ω P m).Pr H +
            (Btap H D.V D.E D.commit D.parse tops Ω P m).Pr H) ∧
        ((∀ z ∈ Zs, ∀ i s, D.parse (x z i) = some s → CertifierRegisters H D.commit (lay s) (k s) (op i) s (tops s)
            ((run tap (schedOf z, srvOf z)).2 i)) →
          {z | z ∈ Zs ∧ ∀ i, WindowGood H D L K S lay k δ reg Dj (sw i) (ρ i) tops (x z i) (wit z i)
              ((run tap (schedOf z, srvOf z)).2 i)}.encard ≤
            Ss.encard * (((∏ c, (cap c + 1)) * ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
              reg (sw i)).card * (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i)) : ℕ) : ℕ∞)) := by
  intro Coin _ _ _ A H D hks L K S lay k δ hδ reg Dj n sw Pw hPw ρ op β hβ Conn Cell Srv Dev Z _
    arr f tap run htrust cap Ss hcap hfit Zs schedOf srvOf hs dec hdz x wit tops hb
  exact h H D hks L K S lay k δ hδ reg Dj sw Pw hPw ρ (fun i _ => op i) β hβ arr f tap run htrust
    cap Ss hcap hfit Zs schedOf srvOf hs dec hdz x wit tops hb

end NetTiming
