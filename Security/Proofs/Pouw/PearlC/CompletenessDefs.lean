import Definitions.Pouw.PearlC.Tile

/-!
# The sampled audit's completeness on a population of activations (trusted)

The honest prover commits the honest transcript (`honestTranscript`). The sampled audit (`GγSampled`'s `AuditAccepts`)
accepts it when every drawn tile is good or of a blank unit. A drawn tile of a committed unit is good when its weight
rows pass the rules, which the workload's admission gives (`Workload.WeightsOK`), its words are the reference's, which
an honest prover's are once every reference value is an FP32 word's, and its replayed debit is within the cap, which
depends on the activations and the noise.

The cap is what can fail. Activations aligned against it pass Π's rules and exceed it, so no completeness holds for
every activation input. `CompleteSampled` therefore names its population `Served`, the activation inputs it speaks of,
and bounds the audit's rejection by `t·ε` over `t` draws: `ε` per draw, by the union bound.
-/

namespace Pouw.PearlC

variable {Q R S : Type}

/-- **The sampled audit is complete on `Served` at `ε` per draw.** For every in-domain workload `U` whose weights pass
the rules, every activation input `act` in `Served`, every ticket count `w` proportional to the tiles' `W_ref`
(`verity.protocols.accounting.work.pouw.audit`'s `tickets`) and every number of draws `t`: over the oracle, the salt and `t` independent
work-weighted draws, the audit rejects the honest transcript (some drawn tile is neither good nor of a blank unit) with
probability at most `t·ε`. -/
def CompleteSampled [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) (TR : TileRules Q R S)
    (D : Layout → Prop) (Served : Workload → (ℕ → Codes) → Prop) (ε : ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → U.WeightsOK P → ∀ act : ℕ → Codes, Served U act →
    ∀ (w : ℕ → ℕ) (κ : ℝ), 0 < κ → (∀ g < (TR.tiling U.layout).NT, (w g : ℝ) = κ * TR.Wref U.layout g) →
    ∀ t : ℕ,
      pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
          ¬ AuditAccepts (goodTiles P TR U x.1 x.2.1 (honestTranscript P x.1 x.2.1 U act) ∪
            blankTiles TR U (honestTranscript P x.1 x.2.1 U act)) x.2.2)
        ≤ t * ε

end Pouw.PearlC
