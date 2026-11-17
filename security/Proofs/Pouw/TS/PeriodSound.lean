module

public import Properties.Certifier.JointBridge.ServiceSoundOfSessions
public import Proofs.Pouw.TS.SessionSound
public import Proofs.Pouw.FillerLuck
public import Proofs.Pouw.MonitoringPeriod.Honest

/-! PoUW's proof system soundness from its proof layer; no theorem here binds `X`, because
`Proofs.Pouw.TS.SessionSound`'s imports make it a token. -/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.Bridge

open Pouw.RelativeSecurity Properties.Certifier.JointBridge Properties.Certifier.JointBridge.Sessions

theorem prob_drawn_le {Ω U V : Type} [Fintype Ω] {n nπ : ℕ} (acc : Paths Ω n → Fin n → Prop)
    (drawn : Paths Ω n → Fin n → U → Prop) (key : U → ℕ)
    (opened : Paths Ω n → Fin n → ℕ → ℕ → V) (rel : Paths Ω n → Fin n → U → (ℕ → V) → Prop)
    (sessOf : Paths Ω n → Fin n → U → Fin nπ) {δ : ℝ}
    (h : (layer acc drawn key opened rel sessOf).SoundAt δ) (w : Fin n) :
    prob (fun x => acc x w ∧ ∃ u, drawn x w u ∧ ¬ rel x w u (opened x w (key u))) ≤ nπ * δ := by
  calc prob (fun x => acc x w ∧ ∃ u, drawn x w u ∧ ¬ rel x w u (opened x w (key u)))
      ≤ prob (fun x => ∃ i ∈ (Finset.univ : Finset (Fin nπ)),
          (layer acc drawn key opened rel sessOf).Unsound x w i) :=
        prob_mono fun x ⟨ha, u, hd, hr⟩ =>
          ⟨sessOf x w u, Finset.mem_univ _, ha, fun hall => hr (hall u hd rfl)⟩
    _ ≤ ∑ i : Fin nπ, prob (fun x => (layer acc drawn key opened rel sessOf).Unsound x w i) :=
        prob_exists_le Finset.univ (fun i x => (layer acc drawn key opened rel sessOf).Unsound x w i)
    _ ≤ ∑ _i : Fin nπ, δ := Finset.sum_le_sum fun i _ => h w i
    _ = nπ * δ := by simp

/-- **PoUW's proof system is sound against its check, from its sessions' soundness**
(`Properties.Certifier.JointBridge.CheckProofSystemSound`): `prob_drawn_le` at the check's layer, whose verdict is the
check's acceptance and whose sampled proof blocks are the units it publishes. -/
theorem CheckProofSystemSound : Properties.Certifier.JointBridge.CheckProofSystemSound := by
  intro C Ξ V _ _ m Np n nπ p ws opened rel sessOf δ h w
  refine (prob_mono fun x hx => ?_).trans (prob_drawn_le _ _ Fin.val opened rel sessOf h w)
  obtain ⟨pub, prof, hc, u, hu, hr⟩ := hx
  exact ⟨⟨pub, prof, hc⟩, u, ⟨pub, prof, hc, hu⟩, hr⟩

open Flock.SoundAtBridge
open scoped ENNReal

theorem serviceSound_of_sessions {C Ξ V : Type} [Fintype C] [Fintype Ξ] {m Np n nπ : ℕ} (p : PouwCheck.Params)
    (ws : Properties.Pouw.MonitoringPeriod.Executions (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop)
    (sessOf : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → Fin nπ)
    (sess : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → ℕ → ℕ → V)
    (δs δ : ℝ)
    (hCR : ∀ x w i, (sess x w i).CR)
    (hAcc : ResourceAccounting.Assumptions.TS.AcceptedLive (checkLayer p ws opened rel sessOf) sess out)
    (hSamplingDesign : ResourceAccounting.Assumptions.TS.ExecutionPhaseDesign sess out)
    (hChk : ResourceAccounting.Assumptions.TS.ServedChecksHold (checkLayer p ws opened rel sessOf) sess out rows)
    (hLink : ResourceAccounting.Assumptions.TS.ServedRootsLinked (checkLayer p ws opened rel sessOf) sess out rows δs)
    (hδ : ∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) (hδ0 : 0 ≤ δ) :
    Properties.Pouw.MonitoringPeriod.ServiceSound p ws opened rel (nπ * (δs + δ)) :=
  ResourceAccounting.SecurityProofs.Bridge.CheckProofSystemSound p ws opened rel sessOf (δs + δ)
    (ResourceAccounting.SecurityProofs.TS.SessionSoundAt (checkLayer p ws opened rel sessOf) sess out rows δs δ hCR hAcc hSamplingDesign
      hChk hLink hδ hδ0)

open Properties.Pouw.MonitoringPeriod Properties.Pouw.MonitoringPeriod.SecurityProofs.HonestH1

/-- **The honest witness of `CheckProofSystemSound`** (`CheckProofSystemSoundHonest`): at H1's honest live-salt period
of one window, one session a window, the deployed check accepts the window on every path, and its layer is sound at
`0`. -/
theorem CheckProofSystemSound_honest : CheckProofSystemSoundHonest :=
  ⟨circuit.rounds, prm, executionPhases 1, fun x w => accepted_all 1 x w,
    fun _ _ => le_of_eq
      (ResourceAccounting.SecurityProofs.ComposeMonitoringPeriod.Honest.prob_none fun _ h => h.2 fun _ _ _ => trivial)⟩

/-- **The nontrivial witness of `CheckProofSystemSound`** (`CheckProofSystemSoundNontrivial`): at H1's honest period of
one window the deployed check accepts and publishes a sampled proof block on every path, so with relations that every opening
fails `ServiceSound` fails at `0`. -/
theorem CheckProofSystemSound_nontrivial : CheckProofSystemSoundNontrivial := by
  refine ⟨circuit.rounds, prm, executionPhases 1, fun h => ?_⟩
  have : Nonempty (Paths Ω1 1) := ⟨((Paths.const true _, (Equiv.refl _, ())), ())⟩
  refine absurd (h 0) (not_le.2 (lt_of_lt_of_le one_pos
    ((ResourceAccounting.SecurityProofs.FillerLuck.Cell.prob_all (Ω := Paths Ω1 1)).symm.le.trans
      (prob_mono fun x _ => ?_))))
  obtain ⟨pub, prof, hc⟩ := accepted_all 1 x 0
  obtain ⟨-, -, hd⟩ := sampleIsCoin 1 x 0 pub prof hc
  exact ⟨pub, prof, hc, ResourceAccounting.Compose.drawOf hK24 (ResourceAccounting.TS.coinAt x 0).2 ⟨0, by decide⟩,
    (hd _).2 ⟨⟨0, by decide⟩, rfl⟩, fun h => h⟩

end ResourceAccounting.SecurityProofs.Bridge
