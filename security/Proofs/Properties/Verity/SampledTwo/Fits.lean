module

public import Proofs.Properties.Verity.SampledTwo.ReadsOnly

/-! `SourcesFit` at every held copy and case map, and `acceptsRun_of'` with it discharged (`acceptsRun_fits`). -/

@[expose] public section

namespace Verity.SampledTwo

open Claims

theorem stageBytes_eq : VStar.Recursive.SampledTwo.STAGE_BYTES = SpotFlock.L := rfl

theorem sourcesFit (A : Live₂) (C : CaseOf) : SourcesFit A C := by
  intro o i it hA hi hit
  obtain ⟨k, hk, p, cm, e, pub, prof, ⟨h1, h2, h3⟩, hacc⟩ := hA
  obtain ⟨-, c₂, ξ₂, r, i', run, hin, -⟩ := VStar.Recursive.SampledTwo.accepted_steps hacc
  obtain ⟨hf1, hf2⟩ := VStar.Recursive.SampledTwo.accepted_fits hacc hin
  obtain ⟨-, -, -, -, vs, rc, j₁, j₂, -, hieq, -, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
  have hik : inputOfCase k = some i' := by
    unfold inputOfCase
    rw [h1, h2, h3]
    simp only [bind, Except.bind, hin]
    rfl
  have hi' : inputOn C o = some i' := by
    unfold inputOn
    rw [hk]
    exact hik
  have hii : i = i' := Option.some.inj (hi.symm.trans hi')
  rw [hii, hieq] at hit
  have hs1 : i.source = e.source := by rw [hii, hieq]; rfl
  have hs2 : it.source = ξ₂ := (congrArg Spot.Run.Interior.source (Option.some.inj hit)).symm
  rw [stageBytes_eq] at hf1 hf2
  have hst : A.h.stage = A.h.more + 1 := rfl
  have hpos : 0 < A.h.stage := by omega
  rw [hs1, hs2]
  exact ⟨hf1, Nat.le_trans hf2 (Nat.le_mul_of_pos_left _ hpos)⟩

theorem acceptsRun_fits (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (hplay : PlayAccepts A C S P') (hsess : SessionMsgs A C S P') :
    AcceptsRun A x (formReading A C S dev) (FormAccepts C) :=
  acceptsRun_of' A x C S dev P' (sourcesFit A C) hplay hsess

end Verity.SampledTwo
