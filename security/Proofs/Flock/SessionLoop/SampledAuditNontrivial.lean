module

public import Proofs.Flock.SessionLoop.SampledAudit
public import Proofs.Flock.SessionLoop.ZkAnyHonest
public import Proofs.Flock.SessionLoop.Leak
public import Proofs.Core.ValueSalts

/-!
# `SampledAuditAnyDeployed`'s conclusion can fail

`Flock.Guarantees.SampledAuditAnyDeployed` concludes `Loop.AuditAnyAt`: the audit's view is within the registration's
bound of the simulators' and the sessions' errors. That conclusion fails at a stack that reads the key, as
`Verity.ValueSalts.RegisteredLeakage`'s does: at the witnesses' value service, with no hm96 leaf, the stack that hands a
record of its own as registration 0, its public commitment `true` and its record tag the key, is always admitted with a
`true` commitment, while no genuine or ideal record has one and, run at the key `false`, it forges only when the key is
`false`.

* `Loop.auditSv`: the witness's value service, two keys, one session and tag, no hm96 leaf, every public commitment
  `false` and the record tag the key; `Loop.auditSt`: its stack, which hands a record of its own as registration 0,
  public commitment `true` and record tag the key.
* `Loop.auditQ`: sessions at the concrete statement and its trace, the honest values and the honest worker of the
  session loop's release on the tape's samples, against `Loop.helloRej`; `Loop.auditCB` and `Loop.auditEC`, their
  simulators' coins as they are.
* `Flock.SecurityProofs.SampledAuditAnyDeployed_nontrivial`: there, at no sampled session, `AuditAnyAt` fails:
  `1 ≤ 1 · (0 + 0 + 0) + 1/2 + 0` is false.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer.Loop

open Claims Claims.ZKMal Verity.ValueSalts FlockSoundness.ZK

/-- **The witness's value service**: two keys, one session and tag, no hm96 leaf; every public commitment is `false`,
and the record tag is the key. -/
abbrev auditSv : Service Bool Unit (Fin 0 → FlockSoundness.Discharge.ZkView.Salt) Unit Bool Unit Bool :=
  Service.mk (fun _ _ i => Fin.elim0 i) (fun _ _ => false) (fun _ _ _ => ()) (fun k _ _ _ => k)

/-- **The witness's stack**: it hands a record of its own as registration 0, public commitment `true` and record tag the
key. -/
abbrev auditSt : Stack Bool Unit Unit Bool Unit Bool 1 := fun k _ _ => .inr (0, 1, ((), true, ()), k)

/-- **Sessions at the concrete statement**: each at its trace, the honest values and the honest worker of the session
loop's release on the tape's samples (`tapeL₀`, `tapeSrcHi`), against `helloRej`. -/
noncomputable def auditQ : Sessions (RegistrationRecord Unit Bool Unit Bool) Unit (Bool × (Fin 1 → Unit))
    ((CoinLayer.session loopNr Masked.dataE).SwVals ×
      (LeanRelease.ofLoopLink tapeL₀ tapeSrcHi (realOf tapeSrcHi)).toSrv.Worker) Flock.Wit
    (((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand) ×
      (LeanRelease.ofLoopLink tapeL₀ tapeSrcHi (realOf tapeSrcHi)).toSrv.Chk) Unit (List (Msg × Out)) where
  x _ _ _ _ := FlockDeploy.concreteStmt
  Vs _ _ _ _ := helloRej
  θ _ _ _ _ := ()
  i _ _ _ _ _ := (finS.honestVals, tapeWk)
  w _ _ _ _ _ := FlockDeploy.wC

/-- The simulators' coins of `auditQ`'s sessions, at the session loop's release at one repetition. -/
noncomputable abbrev auditCB (_ : ℕ) : Type :=
  (anySim Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink tapeL₀ tapeSrcHi) tapeSrcHi) (realOf tapeSrcHi) 1).C
    FlockDeploy.concreteStmt

/-- `auditQ`'s simulators' coins, as they are. -/
noncomputable def auditEC : ∀ j ρ χ b, auditCB j ≃
    (anySim Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink tapeL₀ tapeSrcHi) tapeSrcHi) (realOf tapeSrcHi)
      1).C (auditQ.x j ρ χ b) :=
  fun _ _ _ _ => Equiv.refl _

end Claims.ZKServer.Loop

namespace Flock.SecurityProofs

open Claims Claims.ZKMal Claims.ZKServer
open Verity Verity.Game Verity.ValueSalts FlockSoundness.ZK
open scoped ENNReal

/-- **The bound fails at a stack that reads the key.** At the witness's value service (`Loop.auditSv`), with no hm96
leaf, and no sampled session (of `Loop.auditQ`, at one repetition), the stack that hands a record of its own as
registration 0, public commitment `true` and record tag the key (`Loop.auditSt`), is always admitted with a `true`
commitment, while no genuine or ideal record has one and, run at the key `false`, it forges only when the key is
`false`: `1 ≤ 1 · (0 + 0 + 0) + 1/2 + 0` is false. -/
theorem SampledAuditAnyDeployed_nontrivial :
    ¬ Loop.AuditAnyAt Loop.auditSv 0 1 () () Loop.auditSt false 0 1 Loop.tapeL₀ Loop.tapeSrcHi Loop.auditQ
      Loop.auditEC (fun _ _ => 0) () (fun ρ _ _ => ρ.2.2.1.2.1 = true) := by
  have hideal : Service.ideal Loop.auditSv 0 1 () (fun ρ => ρ.2.2.1.2.1 = true) = 0 :=
    Verity.ValueSalts.prCoin_of_forall_not fun _ h => Bool.false_ne_true h
  have hgen : Service.genuine Loop.auditSv 0 1 () (fun ρ => ρ.2.2.1.2.1 = true) = 0 :=
    Verity.ValueSalts.prCoin_of_forall_not fun _ h => Bool.false_ne_true h
  have hforge : Service.forgeAdvantage Loop.auditSv 0 1 () (Loop.auditSt false ()) = 2⁻¹ := by
    have : Service.forgeAdvantage Loop.auditSv 0 1 () (Loop.auditSt false ()) =
        Game.prCoin fun p : Bool × (Fin 1 → Unit) => p.1 = false := by
      unfold Service.forgeAdvantage
      congr 1
      funext p
      apply propext
      constructor
      · rintro ⟨ρ, hρ, had, -⟩
        cases hρ
        exact had.symm
      · intro hk
        refine ⟨_, rfl, hk.symm, fun i h => ?_⟩
        exact Bool.false_ne_true (congrArg (fun ρ : RegistrationRecord Unit Bool Unit Bool => ρ.2.2.1.2.1) h).symm
    rw [this]
    unfold Game.prCoin
    rw [Verity.ValueSalts.avg_prod]
    simp [Game.avg]
  haveI : ∀ F : ℕ → Type, Nonempty (Tup F 0) := fun _ => ⟨()⟩
  intro h
  unfold Loop.AuditAnyAt at h
  refine absurd (le_trans (le_of_eq (Verity.ValueSalts.prCoin_of_forall ?_).symm) h) ?_
  · intro q
    exact ⟨rfl, rfl, rfl⟩
  · simp only [Service.prfAdvantage, hideal, hgen, hforge, Tup.errs, Nat.cast_one, Nat.cast_zero, zero_mul, add_zero,
      one_mul, tsub_zero]
    simp [Game.avg]

end Flock.SecurityProofs
