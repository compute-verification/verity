module

public import Definitions.Claims.Encodings
public import Proofs.Flock.Encodings.Binding
public import Proofs.Flock.Encodings.Relink

/-! The registered encodings' binding statements, proved with no assumption, and the composed statements, which are not
proved. -/

@[expose] public section

namespace Claims.Encodings

open Claims.Sampling (CommitmentScheme Bext)
open Claims.SamplingGeneric (Arg Compiled)

def CompilerKSLinked : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : Compiled Coin A)
    (Lk : RegLink Coin) (H : Msg → Msg) (enc : List Msg → ℕ → Bool) (Ce Cm : CommitmentScheme Bool),
    A.KS H → Lk.A.KS H → L.commit = byPos enc Ce Cm → Ce.Binds H → Cm.Binds H → L.Sessions H →
    Lk.Opens H enc Cm Ce → ∀ δ : ℝ, 0 ≤ δ → ∀ x, L.Scope x → (∀ s, L.parse x = some s → Lk.names s) →
    ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (Pℓ : Prover Ω) (Pa : Prover (Ω × (Fin (Lk.A.V.k Lk.x) → Coin))),
      PrBadLinked H L Lk enc Cm δ x Pℓ Pa ≤
        L.κ δ x + Real.sqrt (L.γ x * PrB H L.V Pa x) + (Bext H L.V L.E _ Pa x).Pr H + Lk.ε H Pℓ +
          (Blink H L Lk Cm Ce x Pℓ Pa).Pr H

def CompilerKSRegLinked : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : Compiled Coin A)
    (Lk : RegLink Coin) (H : Msg → Msg) (enc : List Msg → ℕ → Bool) (Ce Cm : CommitmentScheme Bool),
    A.KS H → Lk.A.KS H → L.commit = byPos enc Ce Cm → Ce.Binds H → Cm.Binds H → L.Sessions H →
    Lk.Opens H enc Cm Ce → ∀ x, L.Scope x → (∀ s, L.parse x = some s → Lk.names s) →
    ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (Pℓ : Prover Ω) (Pa : Prover (Ω × (Fin (Lk.A.V.k Lk.x) → Coin))),
      Pr (fun q : Ω × (Fin (Lk.A.V.k Lk.x) → Coin) × Lk.A.E.C Lk.x × (Fin (L.V.k x) → Coin) =>
          (run H Pℓ Lk.A.V Lk.x q.1 q.2.1).out = .acc ∧ (run H Pa L.V x (q.1, q.2.1) q.2.2.2).out = .acc ∧
            ∃ u ∈ L.checked x q.2.2.2, u ∈ FregL H L Lk enc Cm Ce x Pℓ Pa q.1 q.2.1 q.2.2.1) ≤
          εregLinked H L Lk Cm Ce x Pℓ Pa ∧
        ∀ s, L.parse x = some s → ∀ δ : ℝ, 0 ≤ δ →
          Pr (fun q : Ω × (Fin (Lk.A.V.k Lk.x) → Coin) × Lk.A.E.C Lk.x × (Fin (L.V.k x) → Coin) =>
              (run H Pℓ Lk.A.V Lk.x q.1 q.2.1).out = .acc ∧ (run H Pa L.V x (q.1, q.2.1) q.2.2.2).out = .acc ∧
                FregL H L Lk enc Cm Ce x Pℓ Pa q.1 q.2.1 q.2.2.1 ∉ s.design.M (L.part s) δ) ≤
            δ + εregLinked H L Lk Cm Ce x Pℓ Pa

end Claims.Encodings
