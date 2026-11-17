module

public import Proofs.Slowdown.CutCharged
public import Proofs.Slowdown.Layered
public import Proofs.Slowdown.FitKeep
public import Proofs.Slowdown.WeightsSlotsWitness
public import Proofs.Slowdown.FitRate
public import Proofs.Slowdown.FitWitness
public import Proofs.Slowdown.CoveringTwoCostsWitness
public import Proofs.Slowdown.Witnesses
public import Proofs.Slowdown.Cover
public import Proofs.Slowdown.Lemmas
public import Proofs.Slowdown.ReadsBy
public import Proofs.Core
/-!
# Proofs of the slowdown bound's guarantees

The cuts and the min-cut bound (`Cut`, `CutCharged`), the work and product counts of threads, fit and layered
references, the cover theorem (`Cover.Training`), covering at one and two costs, the
slowdown over execution phases (`Slowdown`), the remote bound over monitoring
periods (`RemoteMonitoringPeriod`), and their witnesses. The policy they take as a hypothesis is
`Proofs.InferenceOnly`'s. Inference divided into sessions complies and is slowed by at most 1 (`Inference`).
-/

@[expose] public section

open InferenceOnly

namespace Slowdown.SecurityProofs

open Verity Finset

theorem Inference : Guarantees.Inference := by
  intro c k d B P prompt hB hS
  obtain ⟨hcard, hreads, hprompt⟩ := hS
  have hc : Complies c k d := by
    refine ⟨B, P, ⟨hB, fun u => (hcard u).le, fun u => ?_⟩⟩
    have hsub : d.unregistered (P.gates u) ⊆ prompt u := fun a ha => by
      rw [RegisteredCircuit.unregistered, Finset.mem_sdiff] at ha
      exact (Finset.mem_union.1 (hreads u ha.1)).resolve_left ha.2
    have hw : d.D.width (prompt u) = (prompt u).card := by
      simp [Circuit.width, hB.one_bit]
    exact (Finset.card_le_card hsub).trans (hw.symm.le.trans (hprompt u))
  refine ⟨hc, le_trans (iInf₂_le d ⟨Verity.SecurityProofs.ContainsImplements d d (Containment.refl d.D)
    fun _ _ h => h, hc⟩) ENNReal.div_self_le_one⟩

end Slowdown.SecurityProofs
