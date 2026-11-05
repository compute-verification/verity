import Definitions.NetworkCertifier.Guarantees.Period
import Proofs.NetworkCertifier.KSWired

/-!
# The guarantees restated over the wiring, at a layout with no wires

Each guarantee restated over the layout's wiring (`BoundaryGates.wires`) takes `Claims.Sampling.KSBy` at the wired check
(`wiredOk`) where its statement before the wiring took `Claims.Sampling.KS`. At a layout with no wires,
`Claims.Sampling.KS` gives it (`ksWired_none`), and
the definitions the conclusions read say what they said before the wiring: the incorrect set is the faulty set
(`incorrect_none`), the sampling relation at the incorrect set is `Claims.Sampling.R` (`RWired_none`), and the circuit's
Program reads its inputs as given (`resolveWires_nil`). So each guarantee gives, at a layout with no wires, its statement
before the wiring, by one application of `ksWired_none`:

* `sampledAuditAccepted_noWires`: `SampledAuditAccepted`;
* `sampledLeakAccepted_noWires`: `SampledLeakAccepted`;
* `sampledLeakRegistered_noWires`: `SampledLeakRegistered`;
* `periodLeakAccepted_noWires`: `PeriodLeakAccepted`.

`Accounting.SecurityProofs.Bridge.commOfPeriodLeakAccepted_noWires` is `CommOfPeriodLeakAccepted`'s.
-/

namespace NetTiming

open Claims Claims.Sampling Finset

-- A corollary keeps its guarantee's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **`SampledAuditAccepted` at a layout with no wires** is its statement before the wiring, at `Claims.Sampling.KS`. -/
theorem sampledAuditAccepted_noWires (h : Guarantees.SampledAuditAccepted) :
    ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
      (H : Msg → Msg) (D : Deploy Coin Col T) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S)
      (hw : ∀ s, (lay s).wires = fun _ => none) (k : AuditStatement → ℕ),
      KS H D → KSAccepted H D L K S lay k := by
  intro Coin Col T _ _ _ _ _ H D L K S lay hw k hks
  exact h H D L K S lay k (ksWired_none hw hks)

/-- **`SampledLeakAccepted` at a layout with no wires** is its statement before the wiring, at `Claims.Sampling.KS`. -/
theorem sampledLeakAccepted_noWires (h : Guarantees.SampledLeakAccepted) :
    Guarantees.SampledAuditAccepted →
      ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
        (H : Claims.Msg → Claims.Msg) (D : Deploy Coin Col T) (hks : KS H D) (L K S : ℕ)
        (lay : AuditStatement → BoundaryGates L S) (hw : ∀ s, (lay s).wires = fun _ => none) (k : AuditStatement → ℕ)
        (δ : ℝ) (decode : List (ℕ × Col) → List T) (hδ : 0 ≤ δ)
        (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (reg : AuditStatement → Finset ℕ)
        (β : AuditStatement → ℕ) (Dj : ℕ)
        (hβ : ∀ s P, D.partition s = some P → ∀ F ∈ faultySets s.design P δ, faultyBits s.prog P F ≤ β s),
        KSLinOn H D.V (ScopeAcc D lay k)
            (RLeak H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj)
            (E D decode) (κ D δ) (γ D) (fun _ => 1) (Bext H D.V (E D decode)) ∧
          ∀ x, ScopeAcc D lay k x → ∀ s P, D.parse x = some s → D.partition s = some P →
            ∀ (op : ℕ → List (Obs L K S) → List Bool) (ρ : ℕ → Bool),
              (AccSeqsSampled s.prog P s.design (lay s) (k s) (AsReg s.prog (lay s) (k s) (reg s) ρ) δ op Dj).encard ≤
                ((2 ^ (freeInputs s.prog (lay s) (k s) \ reg s).card * (faultySets s.design P δ).card * 2 ^ β s *
                  Dj ^ k s : ℕ) : ℕ∞) := by
  intro hSA Coin Col T _ _ _ _ _ H D hks L K S lay hw k
  exact h hSA H D L K S lay (ksWired_none hw hks) k

/-- **`SampledLeakRegistered` at a layout with no wires** is its statement before the wiring, at
`Claims.Sampling.KS`. -/
theorem sampledLeakRegistered_noWires (h : Guarantees.SampledLeakRegistered) :
    Guarantees.SampledAuditAccepted →
      ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
        (H : Claims.Msg → Claims.Msg) (D : Deploy Coin Col T) (hks : KS H D) (hb : D.commit.Binds H) (L K S : ℕ)
        (lay : AuditStatement → BoundaryGates L S) (hw : ∀ s, (lay s).wires = fun _ => none) (k : AuditStatement → ℕ)
        (δ : ℝ) (decode : List (ℕ × Col) → List T) (hδ : 0 ≤ δ)
        (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (reg : AuditStatement → Finset ℕ)
        (ops : AuditStatement → List (Opening Bool)) (β : AuditStatement → ℕ) (Dj : ℕ)
        (hβ : ∀ s P, D.partition s = some P → ∀ F ∈ faultySets s.design P δ, faultyBits s.prog P F ≤ β s),
        (∀ x, ScopeAcc D lay k x → (∀ s, D.parse x = some s → RegOpens H D.commit s (reg s) (ops s)) →
          ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
            Claims.PrBad H D.V
                (RLeak H D.commit D.parse D.partition L K S lay k δ
                  (fun s _ => AsReg s.prog (lay s) (k s) (reg s) (regVal (ops s))) Dj) (E D decode) x P ≤
              κ D δ x + Real.sqrt (γ D x * Claims.PrB H D.V P x) + (Bext H D.V (E D decode) Ω P x).Pr H +
                (Breg H D.V (E D decode) D.commit D.parse ops Ω P x).Pr H) ∧
          ∀ x, ScopeAcc D lay k x → ∀ s P, D.parse x = some s → D.partition s = some P →
            ∀ op : ℕ → List (Obs L K S) → List Bool,
              (AccSeqsSampled s.prog P s.design (lay s) (k s) (AsReg s.prog (lay s) (k s) (reg s) (regVal (ops s))) δ
                  op Dj).encard ≤
                ((2 ^ (freeInputs s.prog (lay s) (k s) \ reg s).card * (faultySets s.design P δ).card * 2 ^ β s *
                  Dj ^ k s : ℕ) : ℕ∞) := by
  intro hSA Coin Col T _ _ _ _ _ H D hks hb L K S lay hw k
  exact h hSA H D hb L K S lay (ksWired_none hw hks) k

/-- **`PeriodLeakAccepted` at a layout with no wires** is its statement before the wiring, at `Claims.Sampling.KS`. -/
theorem periodLeakAccepted_noWires (h : Guarantees.PeriodLeakAccepted) :
    ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
      (H : Claims.Msg → Claims.Msg) (D : Deploy Coin Col T) (hks : KS H D) (L K S : ℕ)
      (lay : AuditStatement → BoundaryGates L S) (hw : ∀ s, (lay s).wires = fun _ => none)
      (k : AuditStatement → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T) (hδ : 0 ≤ δ)
      (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (reg : AuditStatement → Finset ℕ)
      (Dj : ℕ) {n : ℕ} (sw : Fin n → AuditStatement) (Pw : Fin n → Partition)
      (hPw : ∀ i, D.partition (sw i) = some (Pw i))
      (ρ : Fin n → ℕ → Bool) (op : Fin n → ℕ → List (Obs L K S) → List Bool) (β : Fin n → ℕ)
      (hβ : ∀ i, ∀ F ∈ faultySets (sw i).design (Pw i) δ, faultyBits (sw i).prog (Pw i) F ≤ β i)
      {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell)
      (f : TapRecord Conn Cell (Fin n → List (Obs L K S))) (tap : Dev)
      (run : Dev → WardenSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
      (htrust : Verity.Assumptions.TrustedHost tap run fun p => tapCode f p.1 (arr p.2))
      (cap : Conn → ℕ) (Ss : Set (WardenSchedule Conn)) (hcap : ∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c)
      (hfit : ∀ s ∈ Ss, f.fits s)
      (Zs : Set Z) (schedOf : Z → WardenSchedule Conn) (srvOf : Z → Srv) (hs : ∀ z ∈ Zs, schedOf z ∈ Ss)
      (dec : (Conn → List (ℕ × CellView Cell)) → Z) (hdz : ∀ z ∈ Zs, dec (run tap (schedOf z, srvOf z)).1 = z)
      (x : Z → Fin n → Claims.Msg) (wit : Z → Fin n → Wit) (tops : AuditStatement → List (Opening Bool))
      (hb : D.commit.Binds H),
      (∀ m, ScopeAcc D lay k m → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
        Claims.PrBad H D.V
            (RLeakTap H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj
              tops) (E D decode) m P ≤
          κ D δ m + Real.sqrt (γ D m * Claims.PrB H D.V P m) + (Bext H D.V (E D decode) Ω P m).Pr H +
            (Btap H D.V (E D decode) D.commit D.parse tops Ω P m).Pr H) ∧
        ((∀ z ∈ Zs, ∀ i s, D.parse (x z i) = some s → TapOpens H D.commit (lay s) (k s) (op i) s (tops s)
            ((run tap (schedOf z, srvOf z)).2 i)) →
          {z | z ∈ Zs ∧ ∀ i, WindowGood H D L K S lay k δ reg Dj (sw i) (ρ i) tops (x z i) (wit z i)
              ((run tap (schedOf z, srvOf z)).2 i)}.encard ≤
            Ss.encard * (((∏ c, (cap c + 1)) * ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
              reg (sw i)).card * (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i)) : ℕ) : ℕ∞)) := by
  intro Coin Col T _ _ _ _ _ H D hks L K S lay hw k
  exact h H D L K S lay (ksWired_none hw hks) k

end NetTiming
