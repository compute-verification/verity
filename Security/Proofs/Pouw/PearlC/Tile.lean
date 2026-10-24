import Definitions.Pouw.PearlC.Tile
import Proofs.Pouw.PearlC.GameDefs

/-! The proof-only part of `Definitions.Pouw.PearlC.Tile`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

variable {Q R S : Type}

/-- `AdmitsRef` with the reference's cost bounded tile by tile. -/
def AdmitsRefTile (CM : CostModel Q R S) (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop)
    (qref : Workload → ℕ) : Prop :=
  ∃ prog : CM.Prog, ∀ U : Workload, U.InDomain P D → U.WeightsOK P → ∀ act : ℕ → Codes,
    (∀ u < U.layout.N, P.actOK (U.layout.shape u) (act u)) → ∀ (H : Q → R) (s : S),
      CM.run prog H U act [] s = honestTranscript P H s U act ∧
      CM.cost prog H U act [] s ≤ tileWrefTotal TR U.layout ∧
      CM.calls prog H U act [] s ≤ qref U

/-- **The γ-gap property under the sampled audit with U-only binding**: `GγSampled` with the audit accepting a drawn tile
when it is good under U-only binding (`goodTilesU`) or of a blank unit. -/
def GγSampledU [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (TR : TileRules Q R S) (D : Layout → Prop) (γ : ℝ) (η : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    ∀ (w : ℕ → ℕ) (κ : ℝ), 0 < κ → (∀ g < (TR.tiling U.layout).NT, (w g : ℝ) = κ * TR.Wref U.layout g) →
    ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
          AuditAccepts (goodTilesU P TR U x.1 x.2.1 (A.transcript U x.1 x.2.1) ∪
            blankTiles TR U (A.transcript U x.1 x.2.1)) x.2.2 ∧
          T < (1 - γ) * (claimedWref TR U (A.transcript U x.1 x.2.1) - εs * tileWrefTotal TR U.layout))
        ≤ (1 - εs) ^ t + η q U.layout.N

end Pouw.PearlC
