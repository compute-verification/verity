module

public import Proofs.Properties.Verity.SampledTwo.ReadsOnly

/-!
# An accepted run read at most the coins it has

`sourcesFit`: `SourcesFit` (`Accepts.lean`) at every held copy and every case map. The form's check refuses a stage
whose bytes run past one coin's (`stagesFit`, read by `Steps.accepted_fits`), the run of record's first-stage bytes are
the evidence's and its interior's are the bytes the interior audit's sample read (`inputOf`), and the proved
verifier's second stage holds at least one coin (`stage = more + 1`). So where the form's check accepts, the first
stage's bytes are within one coin's and the second's within `stage` coins'.

`acceptsRun_fits` is `acceptsRun_of'` with `SourcesFit` discharged.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims

/-- **The form's bound on a stage's bytes is one coin's bytes.** -/
theorem stageBytes_eq : VStar.Recursive.SampledTwo.STAGE_BYTES = SpotFlock.L := rfl

/-- **An accepted run read at most the coins it has** (`SourcesFit`): at a play the form's check accepts, the run of
record's first-stage bytes are at most one coin's and its interior's at most `stage` coins'. -/
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

/-- **The form's accept is the run's accept**, `acceptsRun_of'` with `SourcesFit` discharged: where the run of record
is a play SPOT[Flock] accepts (`PlayAccepts`) whose sessions' messages are those the check read (`SessionMsgs`), every
run whose coins agree with what the form's check read and whose messages are what it read accepts. -/
theorem acceptsRun_fits (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (hplay : PlayAccepts A C S P') (hsess : SessionMsgs A C S P') :
    AcceptsRun A x (formReading A C S dev) (FormAccepts C) :=
  acceptsRun_of' A x C S dev P' (sourcesFit A C) hplay hsess

end Verity.SampledTwo
