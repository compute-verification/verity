import Definitions.Pouw.Guarantees.Lead
import Definitions.Pouw.Lead.Game

/-!
# The lead in PoUW's game (trusted statements)

The lead's per-call tail (`Pouw.Guarantees.Lead.G_tail`) gives PoUW's break-form γ-gap property for the lead's calls
(`Pouw.Lead.Game`), and with it the game's end-to-end bound. Both take `G_tail` as a hypothesis.

* `GammaFromTail`: `G_γ` at `γ = 1 − 32 · (honestRegs − gapFixed) / ⌊32 · honestRegs⌋` and `η(q, N) = N · etaLead`,
  for every unit's data meeting `G_tail`'s premises and a fixed gap below the honest work: for every workload of at most
  `N₀` calls, budget `T` and adversary within it, `Pr[the honest work of the accepted calls exceeds T / (1 − γ) ∧ the
  run hashed no SHA-512 collision] ≤ N · etaLead`. The break is over every unit's executions and honest commitments
  together, so it is at least as large as each call's `Break`.
* `EndToEndFromTail`: the audit with `t` work-weighted samples on top, as `Pouw.Guarantees.EndToEnd`:
  `Pr[accept ∧ T < (1 − γ)(1 − ε_s) · W ∧ no collision] ≤ (1 − ε_s)^t + N · etaLead`.
-/

namespace Pouw.Guarantees.Lead

open Pouw.Lead Pouw.Assumptions.Lead Pouw.Dimension.H100

/-- **`G_γ` for the lead's calls, from `G_tail`.** For every oracle, price table `prc`, hash `sha`, call
`m × K × n` at `T` tokens, bound `N₀` on the units, coin stream, commitment format `C`, form classifier `isA`, sets
`can` and `real` of strategies and data `κ U u` of each unit meeting `G_tail`'s premises, with the fixed gap below the
honest work, PoUW's γ-gap property holds for the lead's game at
`γ = 1 − 32 · (honestRegs − gapFixed) / ⌊32 · honestRegs⌋` with error `N · etaLead`. -/
def GammaFromTail : Prop :=
  ∀ (Q R : Type) [Fintype Q] [DecidableEq Q] [Fintype R]
    (prc : Prices100) (sha : Hash.Msg → Hash.Digest) (T m n K N₀ : ℕ)
    (stream : CoinStream (lead T m n K)) (C : Pouw.Lead.Commit (lead T m n K))
    (isA : Hash.Msg → Bool) (can real : Strategy (lead T m n K) → Prop)
    (κ : Pouw.Workload → ℕ → Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ),
    G_tail →
    (∀ U u, CommitL1w sha C isA (κ U u) ∧ TileGrain sha C (κ U u) stream real ∧ HonestHashRuled sha C (κ U u) real ∧
      OutsourceWithinEgressLead prc (betaLead (lead T m n K) C) sha C (κ U u) real) →
    H32 prc → W1 prc real → OneInstructionMinimal real → CardWithinAccelW1 can real → ShaWrites isA real →
    gapFixed (lead T m n K) C < honestRegs (lead T m n K) C →
    Gγ (Game.costModel (lead T m n K) N₀ Q R prc sha C κ can) (Game.protocol (lead T m n K) N₀ Q R C)
      (Game.domain N₀) (Game.gamma (lead T m n K) C) (Game.eta (lead T m n K))

/-- **The lead's calls end to end, from `G_tail`.** Under `GammaFromTail`'s hypotheses, for every workload of at most
`N₀` calls, budget `B`, oracle bound `q` and adversary within them, the audit with `t` work-weighted samples has
`Pr[accept ∧ B < (1 − γ)(1 − ε_s) · W ∧ the run hashed no SHA-512 collision] ≤ (1 − ε_s)^t + N · etaLead`, with `γ`
as there and `W = N · ⌊32 · honestRegs⌋`. -/
def EndToEndFromTail : Prop :=
  ∀ (Q R : Type) [Fintype Q] [DecidableEq Q] [Fintype R]
    (prc : Prices100) (sha : Hash.Msg → Hash.Digest) (T m n K N₀ : ℕ)
    (stream : CoinStream (lead T m n K)) (C : Pouw.Lead.Commit (lead T m n K))
    (isA : Hash.Msg → Bool) (can real : Strategy (lead T m n K) → Prop)
    (κ : Pouw.Workload → ℕ → Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ),
    G_tail →
    (∀ U u, CommitL1w sha C isA (κ U u) ∧ TileGrain sha C (κ U u) stream real ∧ HonestHashRuled sha C (κ U u) real ∧
      OutsourceWithinEgressLead prc (betaLead (lead T m n K) C) sha C (κ U u) real) →
    H32 prc → W1 prc real → OneInstructionMinimal real → CardWithinAccelW1 can real → ShaWrites isA real →
    gapFixed (lead T m n K) C < honestRegs (lead T m n K) C →
    ∀ U : Pouw.Workload, U.InDomain (Game.domain N₀) → ∀ B : ℝ, 0 ≤ B → ∀ q : ℕ, q ≤ 2 ^ 64 →
    ∀ A : Adversary (Game.costModel (lead T m n K) N₀ Q R prc sha C κ can), A.Bounded U B q →
    ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      Pouw.pr (fun x : (Q → R) × Game.Salt (lead T m n K) N₀ ×
            (Fin t → Ticket ((Game.protocol (lead T m n K) N₀ Q R C).Wref U.layout) U.layout.N) =>
          AuditAccepts (correctSet (Game.protocol (lead T m n K) N₀ Q R C) U x.1 x.2.1 (A.transcript U x.1 x.2.1))
            x.2.2 ∧
          B < (1 - Game.gamma (lead T m n K) C) * (1 - εs) *
            totalWref (Game.protocol (lead T m n K) N₀ Q R C) U.layout ∧
          ¬ A.Breaks U x.1 x.2.1)
        ≤ (1 - εs) ^ t + U.layout.N * etaLead (lead T m n K)

end Pouw.Guarantees.Lead
