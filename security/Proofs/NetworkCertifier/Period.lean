module

public import Definitions.NetworkCertifier.Guarantees.Period
public import Proofs.NetworkCertifier.SampledAuditAccepted
public import Proofs.NetworkCertifier.SampledLeak
public import Proofs.NetworkCertifier.Tap
public import Proofs.NetworkCertifier.TapRows
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
import Proofs.NetworkCertifier.SampledCharge

/-!
# The leak through the network certifier over a monitoring period of sampled audits

A window that meets `WindowGood`, whose observations the network certifier registers at the transfer gates
(`CertifierRegisters`), has them among the sequences `AccSeqsSampled` counts at its registration, with the registration's values
at `reg` (`mem_accSeqsSampled_of_windowGood`). The observations are the trusted network certifier's record of the cells
it sends, so the client's view reads back from them under its schedule, one its record format holds
(`certifier_cells_read`): the views, and the secrets
they decode to, number at most `#Ss` times the number of the periods whose every window's observations are in its
set (`encard_periodSeqs_le`, over any sets of accepted sequences; `encard_periodGood_le` at `AccSeqsSampled`). A
window's set is the one at its operator, which the earlier windows' observations fix
(`runsBefore`), so those periods number at most the product of the sets' bounds, by induction over the windows
(`encard_chain_le`). The record
holds the fail cells and the padding too, so the count needs no separate charge for them: the bound's
`∏_c (cap_c + 1)`, one fail choice per connection, is room here. `periodLeak` adds the per-window bound at
the network certifier's rows (`sampledLeak_certifier`). At operators that read no earlier window the statement is
the one with each window's operator fixed before the period (`periodLeak_fixed`).
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling Finset

theorem asReg_congr {L S : ℕ} (C : Prog) (lay : BoundaryGates L S) (k : ℕ) (R : Finset ℕ) {ρ ρ' : ℕ → Bool}
    (h : ∀ g ∈ R, ρ g = ρ' g) : AsReg C lay k R ρ = AsReg C lay k R ρ' := by
  ext a
  constructor <;> rintro ⟨h1, h2⟩ <;> refine ⟨fun g hg hf => ?_, h2⟩
  · rw [h1 g hg hf, h g hg]
  · rw [h1 g hg hf, ← h g hg]

/-- **The secrets of a monitoring period, counted over the windows' accepted sequences.** Under `trusted-host`, on
schedules in `Ss` that the network certifier's record format holds, the secrets a decoder recovers from the cells the
client receives, whose every window's observations lie in that window's set `A i (runsBefore o i)` at the run's
observations `o`, number at most `#Ss · ∏_c (cap c + 1) · ∏_i c i`, where each set `A i l`, which reads only the
windows before `i`, holds at most `c i` sequences: the periods so observed number at most `∏_i c i`
(`encard_chain_le`), and the cells read back from the record under the schedule (`certifier_cells_read`);
`∏_c (cap c + 1)` is room. -/
theorem encard_periodSeqs_le {L K S n : ℕ} {Conn Cell Srv Dev Z : Type} [Fintype Conn]
    (arr : Srv → Conn → Arrivals Cell) (f : RecordFormat Conn Cell (Fin n → List (Obs L K S))) (tap : Dev)
    (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => recordingCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (CertifierSchedule Conn)) (hfit : ∀ s ∈ Ss, f.fits s)
    (Zs : Set Z) (schedOf : Z → CertifierSchedule Conn) (srvOf : Z → Srv) (hs : ∀ z ∈ Zs, schedOf z ∈ Ss)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z) (hdz : ∀ z ∈ Zs, dec (run tap (schedOf z, srvOf z)).1 = z)
    (A : Fin n → List (List (Obs L K S)) → Set (List (Obs L K S))) (c : Fin n → ℕ)
    (hc : ∀ i l, (A i l).encard ≤ (c i : ℕ∞)) :
    {z | z ∈ Zs ∧ ∀ i, (run tap (schedOf z, srvOf z)).2 i ∈
        A i (runsBefore (run tap (schedOf z, srvOf z)).2 i)}.encard ≤
      Ss.encard * (((∏ c', (cap c' + 1)) * ∏ i, c i : ℕ) : ℕ∞) := by
  let Y : Set (Fin n → List (Obs L K S)) := {o | ∀ i, o i ∈ A i (runsBefore o i)}
  let V : Set (Conn → List (ℕ × CellView Cell)) :=
    (fun p : CertifierSchedule Conn × (Fin n → List (Obs L K S)) => f.read p.1 p.2) '' (Ss ×ˢ Y)
  have hsub : {z | z ∈ Zs ∧ ∀ i, (run tap (schedOf z, srvOf z)).2 i ∈
      A i (runsBefore (run tap (schedOf z, srvOf z)).2 i)} ⊆ dec '' V := fun z hz =>
    ⟨(run tap (schedOf z, srvOf z)).1, ⟨(schedOf z, (run tap (schedOf z, srvOf z)).2),
      ⟨hs z hz.1, hz.2⟩, (certifier_cells_read f arr tap run htrust _ (hfit _ (hs z hz.1)) _).symm⟩,
      hdz z hz.1⟩
  have hY : Y.encard ≤ ((∏ i, c i : ℕ) : ℕ∞) := encard_chain_le A c hc
  have hcap1 : 0 < ∏ c', (cap c' + 1) := Finset.prod_pos fun c' _ => Nat.succ_pos _
  calc {z | z ∈ Zs ∧ ∀ i, (run tap (schedOf z, srvOf z)).2 i ∈
        A i (runsBefore (run tap (schedOf z, srvOf z)).2 i)}.encard
      ≤ (dec '' V).encard := Set.encard_le_encard hsub
    _ ≤ V.encard := Set.encard_image_le _ _
    _ ≤ (Ss ×ˢ Y).encard := Set.encard_image_le _ _
    _ = Ss.encard * Y.encard := Set.encard_prod
    _ ≤ _ := by
        gcongr
        refine hY.trans ?_
        exact_mod_cast Nat.le_mul_of_pos_left _ hcap1

section Compiled

open Claims.SamplingGeneric (Arg Compiled)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem mem_accSeqsSampled_of_windowGood {H : Msg → Msg} {D : Compiled Coin A} {L K S : ℕ}
    {lay : AuditStatement → BoundaryGates L S} {k : AuditStatement → ℕ} {δ : ℝ} {reg : AuditStatement → Finset ℕ}
    {Dj : ℕ} {s₀ : AuditStatement} {P₀ : Partition}
    (hP₀ : D.partition s₀ = some P₀) {ρ : ℕ → Bool} {tops : AuditStatement → List (Opening Bool)}
    {op : ℕ → List (Obs L K S) → List Bool} {x : Msg} {w : Wit} {o : List (Obs L K S)}
    (htap : ∀ s, D.parse x = some s → CertifierRegisters H D.commit (lay s) (k s) op s (tops s) o)
    (h : WindowGood H D L K S lay k δ reg Dj s₀ ρ tops x w o) :
    (lay s₀).Fits s₀.prog (k s₀) ∧
      o ∈ AccSeqsSampled s₀.prog P₀ s₀.design (lay s₀) (k s₀) (AsReg s₀.prog (lay s₀) (k s₀) (reg s₀) ρ) δ op Dj := by
  obtain ⟨⟨s₁, P₁, hs₁, hP₁, hl⟩, ⟨-, s₂, hs₂, hfits⟩, s, hs, ⟨hprog, hdes, hlay, hk, hreg, hpart⟩, hρ, hcov,
    hlen, hj⟩ := h
  rw [Option.some_injective _ (hs₁.symm.trans hs)] at hP₁ hl
  rw [Option.some_injective _ (hs₂.symm.trans hs)] at hfits
  rw [Option.some_injective _ (hP₁.symm.trans (hpart.trans hP₀))] at hl
  have hmem := hl op o (htap s hs) hcov hlen hj
  beta_reduce at hmem
  rw [asReg_congr _ _ _ _ hρ] at hmem
  rw [hprog, hdes, hlay, hk, hreg] at hmem
  rw [hprog, hlay, hk] at hfits
  exact ⟨hfits, hmem⟩

theorem encard_periodGood_le (H : Msg → Msg) (D : Compiled Coin A) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (δ : ℝ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ) {n : ℕ} (sw : Fin n → AuditStatement) (Pw : Fin n → Partition)
    (hPw : ∀ i, D.partition (sw i) = some (Pw i)) (ρ : Fin n → ℕ → Bool)
    (op : Fin n → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool) (β : Fin n → ℕ)
    (hβ : ∀ i, ∀ F ∈ faultySets (sw i).design (Pw i) δ, faultyBits (sw i).prog (Pw i) F ≤ β i)
    {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell)
    (f : RecordFormat Conn Cell (Fin n → List (Obs L K S))) (tap : Dev)
    (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => recordingCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (CertifierSchedule Conn)) (hfit : ∀ s ∈ Ss, f.fits s)
    (Zs : Set Z) (schedOf : Z → CertifierSchedule Conn) (srvOf : Z → Srv) (hs : ∀ z ∈ Zs, schedOf z ∈ Ss)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z) (hdz : ∀ z ∈ Zs, dec (run tap (schedOf z, srvOf z)).1 = z)
    (x : Z → Fin n → Msg) (wit : Z → Fin n → Wit) (tops : AuditStatement → List (Opening Bool))
    (htap : ∀ z ∈ Zs, ∀ i s, D.parse (x z i) = some s → CertifierRegisters H D.commit (lay s) (k s)
      (op i (runsBefore (run tap (schedOf z, srvOf z)).2 i)) s (tops s) ((run tap (schedOf z, srvOf z)).2 i)) :
    {z | z ∈ Zs ∧ ∀ i, WindowGood H D L K S lay k δ reg Dj (sw i) (ρ i) tops (x z i) (wit z i)
        ((run tap (schedOf z, srvOf z)).2 i)}.encard ≤
      Ss.encard * (((∏ c, (cap c + 1)) * ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card *
        (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i)) : ℕ) : ℕ∞) := by
  set G := {z | z ∈ Zs ∧ ∀ i, WindowGood H D L K S lay k δ reg Dj (sw i) (ρ i) tops (x z i) (wit z i)
    ((run tap (schedOf z, srvOf z)).2 i)} with hG
  rcases G.eq_empty_or_nonempty with he | ⟨z₀, hz₀⟩
  · rw [he, Set.encard_empty]
    exact zero_le
  let A : Fin n → List (List (Obs L K S)) → Set (List (Obs L K S)) := fun i l =>
    AccSeqsSampled (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i))
      (AsReg (sw i).prog (lay (sw i)) (k (sw i)) (reg (sw i)) (ρ i)) δ (op i l) Dj
  have hsub : G ⊆ {z | z ∈ Zs ∧ ∀ i, (run tap (schedOf z, srvOf z)).2 i ∈
      A i (runsBefore (run tap (schedOf z, srvOf z)).2 i)} :=
    fun z hz => ⟨hz.1, fun i => (mem_accSeqsSampled_of_windowGood (hPw i) (htap z hz.1 i) (hz.2 i)).2⟩
  have hfits : ∀ i, (lay (sw i)).Fits (sw i).prog (k (sw i)) := fun i =>
    (mem_accSeqsSampled_of_windowGood (hPw i) (htap z₀ hz₀.1 i) (hz₀.2 i)).1
  exact (Set.encard_le_encard hsub).trans
    (encard_periodSeqs_le arr f tap run htrust cap Ss hfit Zs schedOf srvOf hs dec hdz A
      (fun i => 2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card *
        (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i))
      fun i l =>
        encard_accSeqsSampled_le _ _ _ _ _ (hfits i) δ _ _ (encard_asReg_le _ _ _ _ _) (β i) (hβ i) (op i l) Dj)

/-- **The leak through the network certifier over a monitoring period**: `Guarantees.PeriodLeakAccepted`, from
`sampledLeak_certifier` with the registered inputs pinned (`adviceIn_asReg`), and `encard_periodGood_le`. -/
theorem periodLeak : Guarantees.PeriodLeakAccepted := by
  intro Coin _ _ _ A H D hks L K S lay k δ hδ reg Dj n sw Pw hPw ρ op β hβ Conn Cell Srv Dev Z _
    arr f tap run htrust cap Ss hcap hfit Zs schedOf srvOf hs dec hdz x wit tops hb
  exact ⟨fun m hm Ω _ _ P => TapProof.sampledLeak_certifier H D hks hb L K S lay k δ hδ _ Dj
    (fun _ _ _ _ => adviceIn_asReg K _ _ _ _ _ _ _ _) tops m hm Ω P,
    encard_periodGood_le H D L K S lay k δ reg Dj sw Pw hPw ρ op β hβ arr f tap run htrust cap Ss hfit Zs
      schedOf srvOf hs dec hdz x wit tops⟩

end Compiled

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
