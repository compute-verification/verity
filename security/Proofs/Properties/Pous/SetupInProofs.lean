module

public import Properties.Pous.SetupInProofs
public import Proofs.Properties.Accounting.InProofs

/-!
# Proof of PoUS's setup inside proofs (`Properties.Pous.SetupInProofs`)

When the run's proof blocks hold the registered `W`'s slices (`CarriedW` with no collision), the blocks that don't decode
to their slice are exactly the proof blocks whose values fail `BlockOk`, so the event's count is `failing`'s and `Gives` at
`right` gives it at `BlockOk` (`Gives.mono`).
-/

@[expose] public section

namespace PousCheck.Properties.InProofs.SecurityProofs

open PousCheck.Properties.InProofs
open Verity.InProofs Properties.Accounting.InProofs Verity.Game

theorem setupGives : SetupGives := by
  intro q salt W Ω μ r right ε g h hc
  refine le_trans (prPMF_mono μ fun ω hω => ⟨hω.1, ?_⟩) (Gives.mono h g)
  obtain ⟨ha, hk, hn⟩ := hω
  have hs := (hc ω ha).resolve_right hn
  have hw : wrong q salt W r ω = r.failing (BlockOk q salt) ω := by
    ext u
    have hp : r.ports u ω = ((r.ports u ω).1, slice q W u.val) := Prod.ext rfl (hs u)
    simp only [wrong, Run.failing, Finset.mem_filter, Finset.mem_univ, true_and]
    rw [← hp]
  rw [← hw]
  exact hk

end PousCheck.Properties.InProofs.SecurityProofs
