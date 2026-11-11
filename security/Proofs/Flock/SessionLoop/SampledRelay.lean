module

public import Proofs.Flock.SessionLoop.SampledAudit

/-!
# A sampled audit's view at any values, at the deployed held relay's frames

What leaves a session for the auditor is the deployed held relay's frames: `Flock.Firewall.Session.relay` sends each
request's frame, the loop's proofs and `Finish` on the auditor's connection, which `Relay.out` models. Those frames
are not the session loop's messages. `Flock.Guarantees.SampledAuditAnyRelayDeployed` is `SampledAuditAnyDeployed`
at those frames. The `ℓ` sampled sessions each go through the held relay at their setup, and the audit's view is
their frames. Each session's outcome is its `Finish` verdict with how many frames left (`Relay.relayLen`, at most the
setup's `Relay.relayB`).

* `perOutcome_of_zkOutRelay`: `ZKOutRelay` read per outcome at one session, with the frame count in `ℕ` and the
  simulator's coins read through a bijection.
* `RelaySessions`: `Sessions` releasing the relay's frames. It adds each session's setup, a function of what the
  auditor has seen, and the requests its relay reads, a function of anything.
* `RelaySessions.run`, `rel`, `outc`, `sim`: session `j`'s run of the model, its frames, its outcome and its
  simulator. The simulator is the model's, given the verdict. Its messages are encoded at the setup, woven with the
  setup's control frames, and cut where the frames ended.
* `sampledAudit_zkOutRelay`: the view at `ℓ` sessions, in sequence, of a protocol that meets `ZKOutRelay`.
* **`Flock.SecurityProofs.SampledAuditAnyRelayDeployed`**: `sampledAudit_zkOutRelay` at the second conjunct of
  `Loop.ZkAnyAt` (`SessionLoopZKAnyDeployed`), and `ideal_hm96` for `δ`.

Each session's outcome takes up to `relayB + 1` values for each verdict, at the same error (`εS j` at the verdict).
Scope: the released bytes and their order, not their times.
-/

@[expose] public section

namespace FlockSoundness.ZK

open Claims Claims.ZKServer Verity Verity.Game Verity.ValueSalts Verity.Assumptions
open scoped ENNReal

/-- **`ZKOutRelay` read per outcome at one session**. At the setup `(spec, sts)`, `(x, w) ∈ R`, a verifier `V*` of
the protocol's rounds and its tape, and requests `ins` where the relay's host gives it the setup and what it read at
each model slot is the model's message there, the relay's frames are simulated per outcome. The outcome is the model's
outcome with how many frames left, in `ℕ`. The simulator's coins are read through `e`. -/
theorem perOutcome_of_zkOutRelay {W St Ωs : Type} {N Ω : Msg → Type} [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)]
    [∀ x, Fintype (Ω x)] [∀ x, Nonempty (Ω x)] [Fintype Ωs] {H : Msg → Msg} {k : Msg → ℕ}
    {P : W → (x : Msg) → Prover (N x × Ω x)} {R : Rel W} {out : Msg → List (Msg × Out) → St} {S : SimOut St}
    {ε : (x : Msg) → W → (Θ : Type) → VStar Θ → Θ → St → ℝ} (h : ZKOutRelay N Ω H k P R out S ε)
    (spec : _root_.Flock.SessionSpec) (sts : Array _root_.Flock.Stmt) {x : Msg} {w : W} (hR : R x w) {Θ : Type}
    (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = k x) (ins : N x × Ω x → List (ByteArray × Relay.Env))
    (hE : RelayEnvAt spec sts ins) (hRd : RelayReads spec ins fun ω => (runStar H (P w x) Vs x ω θ).msgs.map (·.1))
    (e : Ωs ≃ S.C x) :
    PerOutcome (fun ω => (out x (runStar H (P w x) Vs x ω θ).msgs,
        (Relay.relayLen true (Relay.relayB spec sts) (ins ω)).val))
      (fun ω => Relay.out true {} {} (ins ω))
      (fun o c => (Relay.weave (Relay.ctlOf (.ok (spec, sts))) (Relay.tmplOf (.ok (spec, sts)))
        (Relay.encOf spec sts (((S.given o.1).out H Vs θ x (e c)).map (·.1)))).take o.2)
      (fun o => ENNReal.ofReal (ε x w Θ Vs θ o.1)) := by
  have hf : Function.Injective fun o : St × Fin (Relay.relayB spec sts + 1) => (o.1, o.2.val) := by
    intro a b hab
    have h₁ := congrArg Prod.fst hab
    have h₂ := congrArg Prod.snd hab
    exact Prod.ext h₁ (Fin.ext h₂)
  have hP := (h spec sts x w hR Θ Vs θ hk ins hE hRd).map_out hf
    (S' := fun o (c : S.C x) => (Relay.weave (Relay.ctlOf (.ok (spec, sts))) (Relay.tmplOf (.ok (spec, sts)))
      (Relay.encOf spec sts (((S.given o.1).out H Vs θ x c).map (·.1)))).take o.2)
    (fun _ => rfl) (ε' := fun o => ENNReal.ofReal (ε x w Θ Vs θ o.1)) (fun _ => le_rfl)
  exact hP.comap_sim e

/-- **An audit's sampled sessions of one protocol through the deployed held relay**, oldest first: `Sessions` releasing
the relay's frames. Session `j`'s setup `(spec, sts)` is a function of the record, the sample and the earlier sessions'
frames. The requests `ins` its relay reads at each of its randomness are a function of the registration's randomness,
the sample, the earlier sessions' randomness and their frames. -/
structure RelaySessions (Rec Xs P ι W A Θ : Type) extends Sessions Rec Xs P ι W A Θ (List ByteArray) where
  spec : (j : ℕ) → Rec → Xs → Tup (fun _ => List ByteArray) j → _root_.Flock.SessionSpec
  sts : (j : ℕ) → Rec → Xs → Tup (fun _ => List ByteArray) j → Array _root_.Flock.Stmt
  ins : (j : ℕ) → P → Xs → Tup (fun _ => A) j → Tup (fun _ => List ByteArray) j → A → List (ByteArray × Relay.Env)

namespace RelaySessions

variable {Rec Xs P ι W A Θ St : Type} (Q : RelaySessions Rec Xs P ι W A Θ)

/-- **Session `j`'s run of the model** at its own randomness `ω`: the prover `Pv` at the session's parameters and
witness, against its verifier, at its statement, the record being `ρ p`. -/
def run (H : Msg → Msg) (Pv : ι → W → Msg → Prover A) (ρ : P → Rec) (p : P) (χ : Xs) (j : ℕ)
    (π : Tup (fun _ => A) j) (b : Tup (fun _ => List ByteArray) j) (ω : A) : Transcript :=
  runStar H (Pv (Q.i j p χ π b) (Q.w j p χ π b) (Q.x j (ρ p) χ b)) (Q.Vs j (ρ p) χ b) (Q.x j (ρ p) χ b) ω
    (Q.θ j (ρ p) χ b)

/-- **Session `j`'s frames** at its own randomness: what the held relay lets out for the auditor over the requests it
reads. -/
def rel (p : P) (χ : Xs) (j : ℕ) (π : Tup (fun _ => A) j) (b : Tup (fun _ => List ByteArray) j) (ω : A) :
    List ByteArray :=
  Relay.out true {} {} (Q.ins j p χ π b ω)

/-- **Session `j`'s outcome**: the model's, read off its messages by `out`, and how many frames left. -/
def outc (out : Msg → List (Msg × Out) → St) (H : Msg → Msg) (Pv : ι → W → Msg → Prover A) (ρ : P → Rec) (p : P)
    (χ : Xs) (j : ℕ) (π : Tup (fun _ => A) j) (b : Tup (fun _ => List ByteArray) j) (ω : A) : St × ℕ :=
  (out (Q.x j (ρ p) χ b) (Q.run H Pv ρ p χ j π b ω).msgs,
    (Relay.relayLen true (Relay.relayB (Q.spec j (ρ p) χ b) (Q.sts j (ρ p) χ b)) (Q.ins j p χ π b ω)).val)

/-- **Session `j`'s simulator** at the record `ρ`, the sample and the earlier simulated frames `b`, given its outcome
`o`: the model's simulator `Sm` given `o.1`, against the session's verifier, at its statement, its coins read through
`eC`; its messages encoded at the session's setup, woven with the setup's control frames and cut after `o.2` frames. -/
def sim (Sm : SimOut St) (H : Msg → Msg) {CB : ℕ → Type} (eC : ∀ j ρ χ b, CB j ≃ Sm.C (Q.x j ρ χ b)) (ρ : Rec)
    (χ : Xs) (j : ℕ) (_s : Tup (fun _ => St × ℕ) j) (b : Tup (fun _ => List ByteArray) j) (o : St × ℕ) (c : CB j) :
    List ByteArray :=
  (Relay.weave (Relay.ctlOf (.ok (Q.spec j ρ χ b, Q.sts j ρ χ b))) (Relay.tmplOf (.ok (Q.spec j ρ χ b, Q.sts j ρ χ b)))
    (Relay.encOf (Q.spec j ρ χ b) (Q.sts j ρ χ b)
      (((Sm.given o.1).out H (Q.Vs j ρ χ b) (Q.θ j ρ χ b) (Q.x j ρ χ b) (eC j ρ χ b c)).map (·.1)))).take o.2

end RelaySessions

/-- **A sampled audit's view at any values, its sampled sessions `ℓ` sessions of one protocol through the deployed held
relay, in sequence.** As `sampledAudit_zkOut`, at a protocol that meets `ZKOutRelay`. Each session's relay must have
its setup from its host (`RelayEnvAt`) and read the model's messages at the model's slots (`RelayReads`). Then the
audit's view of the sessions' frames, at their outcomes `s` (each a verdict and how many frames left), is
`SampledAuditAny`'s bound. The sessions' simulators are run in turn (`Tup.sims`), and their errors at the verdicts
are added. -/
theorem sampledAudit_zkOutRelay {K S Ys V O T R Xs ι W St N₀ Ω₀ Θ : Type} [Fintype K] [Fintype S] [Fintype Ys]
    [Fintype T] [Fintype R] [Fintype Xs] [Fintype N₀] [Nonempty N₀] [Fintype Ω₀] [Nonempty Ω₀] {CB : ℕ → Type}
    [∀ j, Fintype (CB j)] [∀ j, Nonempty (CB j)] {H : Msg → Msg} {k : Msg → ℕ}
    {Pv : ι → W → Msg → Prover (N₀ × Ω₀)} {Rl : Rel W} {out : Msg → List (Msg × Out) → St} {Sm : SimOut St}
    {ε : ι → (x : Msg) → W → (Θ' : Type) → VStar Θ' → Θ' → St → ℝ} {ok : ι → Prop}
    (hZ : ∀ i, ok i → ZKOutRelay (fun _ => N₀) (fun _ => Ω₀) H k (Pv i) Rl out Sm (ε i))
    (sv : Service K S Ys V O T R) (n m : ℕ) (v v₀ : V) (st : Stack K V S O T R m) (k₀ : K) (ℓ : ℕ)
    (Q : RelaySessions (RegistrationRecord S O T R) Xs (K × (Fin m → S)) ι W (N₀ × Ω₀) Θ)
    (eC : ∀ j ρ χ b, CB j ≃ Sm.C (Q.x j ρ χ b)) (εS : ℕ → St → ℝ≥0∞) (δ : ℝ≥0∞) (hown : ZkServerOwnUser st)
    (hok : ∀ j p χ π b, ok (Q.i j p χ π b))
    (hR : ∀ j p χ π b, Rl (Q.x j (recOf sv n v st p) χ b) (Q.w j p χ π b))
    (hk : ∀ j ρ χ b, (Q.Vs j ρ χ b).k (Q.x j ρ χ b) = k (Q.x j ρ χ b))
    (hE : ∀ j p χ π b, RelayEnvAt (Q.spec j (recOf sv n v st p) χ b) (Q.sts j (recOf sv n v st p) χ b)
      (Q.ins j p χ π b))
    (hRd : ∀ j p χ π b, RelayReads (Q.spec j (recOf sv n v st p) χ b) (Q.ins j p χ π b)
      fun ω => (Q.run H Pv (recOf sv n v st) p χ j π b ω).msgs.map (·.1))
    (hε : ∀ j p χ π b o, ENNReal.ofReal (ε (Q.i j p χ π b) (Q.x j (recOf sv n v st p) χ b) (Q.w j p χ π b) Θ
      (Q.Vs j (recOf sv n v st p) χ b) (Q.θ j (recOf sv n v st p) χ b) o) ≤ εS j o)
    (hδ : ∀ A, sv.ideal n m v A ≤ sv.ideal n m v₀ A + δ) (s : Tup (fun _ => St × ℕ) ℓ)
    (E : RegistrationRecord S O T R → Xs → Tup (fun _ => List ByteArray) ℓ → Prop) :
    prCoin (fun q : (K × (Fin m → S)) × Xs × Tup (fun _ => N₀ × Ω₀) ℓ =>
        sv.admits q.1.1 (recOf sv n v st q.1) ∧
          Tup.outs (Q.outc out H Pv (recOf sv n v st) q.1 q.2.1) (Q.rel q.1 q.2.1) ℓ q.2.2 = s ∧
            E (recOf sv n v st q.1) q.2.1 (Tup.rels (Q.rel q.1 q.2.1) ℓ q.2.2)) ≤
      m * (avg (fun q : Xs × Tup CB ℓ =>
          sv.ideal n m v₀ (fun ρ => E ρ q.1 (Tup.sims (Q.sim Sm H eC ρ q.1) ℓ s q.2)) +
            sv.prfAdvantage n m v (fun ρ => E ρ q.1 (Tup.sims (Q.sim Sm H eC ρ q.1) ℓ s q.2))) + δ) +
        sv.forgeAdvantage n m v (st k₀ v) + Tup.errs (fun j o => εS j o.1) ℓ s := by
  have hS : ∀ p χ, PerOutcome (Tup.outs (Q.outc out H Pv (recOf sv n v st) p χ) (Q.rel p χ) ℓ)
      (Tup.rels (Q.rel p χ) ℓ) (Tup.sims (Q.sim Sm H eC (recOf sv n v st p) χ) ℓ)
      (Tup.errs (fun j o => εS j o.1) ℓ) := by
    intro p χ
    refine PerOutcome.stream (A := fun _ => N₀ × Ω₀) (C := CB) (Q.outc out H Pv (recOf sv n v st) p χ)
      (Q.rel p χ) (Q.sim Sm H eC (recOf sv n v st p) χ) (fun j o => εS j o.1) ℓ fun j _ π => ?_
    exact (perOutcome_of_zkOutRelay (hZ _ (hok j p χ π (Tup.rels (Q.rel p χ) j π)))
      (Q.spec j (recOf sv n v st p) χ (Tup.rels (Q.rel p χ) j π))
      (Q.sts j (recOf sv n v st p) χ (Tup.rels (Q.rel p χ) j π))
      (hR j p χ π (Tup.rels (Q.rel p χ) j π))
      (Q.Vs j (recOf sv n v st p) χ (Tup.rels (Q.rel p χ) j π))
      (Q.θ j (recOf sv n v st p) χ (Tup.rels (Q.rel p χ) j π))
      (hk j (recOf sv n v st p) χ (Tup.rels (Q.rel p χ) j π))
      (Q.ins j p χ π (Tup.rels (Q.rel p χ) j π))
      (hE j p χ π (Tup.rels (Q.rel p χ) j π)) (hRd j p χ π (Tup.rels (Q.rel p χ) j π))
      (eC j (recOf sv n v st p) χ (Tup.rels (Q.rel p χ) j π))).mono
      fun o => hε j p χ π (Tup.rels (Q.rel p χ) j π) o.1
  exact Flock.SecurityProofs.SampledAuditAny sv n m v v₀ st k₀
    (fun p χ => Tup.outs (Q.outc out H Pv (recOf sv n v st) p χ) (Q.rel p χ) ℓ)
    (fun p χ => Tup.rels (Q.rel p χ) ℓ) (fun ρ χ => Tup.sims (Q.sim Sm H eC ρ χ) ℓ)
    (Tup.errs (fun j (o : St × ℕ) => εS j o.1) ℓ) δ hown hS hδ s E

end FlockSoundness.ZK

namespace Flock.SecurityProofs

open Claims Claims.ZKMal Claims.ZKServer
open Verity Verity.Game Verity.ValueSalts Verity.Assumptions FlockSoundness.ZK
open FlockSoundness.Discharge
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **A sampled audit's view at any values, at the deployed held relay's frames.** As `SampledAuditAnyDeployed`, with
each sampled session through the deployed held relay. The sessions `Q` give each session's setup `(spec, sts)` and
the requests `ins` its relay reads. The premises are those of `SampledAuditAnyDeployed`, plus two at each session: the
relay's host gives it the setup (`RelayEnvAt`), and what it read at each model slot is the session loop's message there
(`RelayReads`).

Then at the sessions' outcomes `s` (each its verdict and how many frames left) and every event `E` on the record, the
sample and the sessions' frames, the probability that the loop admits the record and the audit shows `s` and `E` is
at most the following: `m` times (the average over the sample and the simulators' coins of `E` on an ideal record of
`v₀` with the simulated frames, plus the PRF gap there, plus `N·2δ₁`), plus the forge term, plus the sessions' errors
at their verdicts. -/
def _root_.Flock.Guarantees.SampledAuditAnyRelayDeployed : Prop :=
  ∀ {K S V O T R Xs Θ D : Type} [Fintype K] [Fintype S] [Fintype T] [Fintype R] [Nonempty S] [Nonempty T] [Nonempty R]
    [Fintype Xs] {N : ℕ} {CB : ℕ → Type} [∀ j, Fintype (CB j)] [∀ j, Nonempty (CB j)]
    (sv : Service K S (Fin N → ZkView.Salt) V O T R) (Hc : ZkView.Bits × ByteArray → D)
    (dig : V → Fin N → ZkView.Bits) (g : (Fin N → D) → O),
    (∀ v y, sv.out v y = g fun i => Hc (dig v i + ZkView.My (y i), ZkView.saltHash (y i))) →
    ∀ (n m : ℕ) (v v₀ : V) (st : Stack K V S O T R m) (k₀ : K) (ℓ t : ℕ), 0 < t →
    ∀ (Rl : Rel Claims.Flock.Wit), (CoinLayer.session Loop.loopNr Masked.dataE).Complete Rl →
    ∀ (L₀ : LeanRelease Claims.Flock.Wit
        ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
      (src : Msg → Claims.Flock.Wit →
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
        ℕ → _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
      (Q : RelaySessions (RegistrationRecord S O T R) Xs (K × (Fin m → S))
        ((CoinLayer.session Loop.loopNr Masked.dataE).SwVals ×
          (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Worker) Claims.Flock.Wit
        (((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
          (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand) ×
          (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Chk) Θ)
      (eC : ∀ j ρ χ b, CB j ≃
        (Loop.anySim Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t).C
          (Q.x j ρ χ b))
      (εS : ℕ → Bool → ℝ≥0∞),
    ZkServerOwnUser st →
    (∀ j p χ π b, (CoinLayer.session Loop.loopNr Masked.dataE).UnCausal (Q.i j p χ π b).1 ∧
      LeanRelease.TrustsStop (CoinLayer.session Loop.loopNr Masked.dataE) (Q.i j p χ π b).1
        (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)) CoinOpen.shaH (Q.i j p χ π b).2 Rl) →
    (∀ j p χ π b, Rl (Q.x j (recOf sv n v st p) χ b) (Q.w j p χ π b)) →
    (∀ j ρ χ b, (Q.Vs j ρ χ b).k (Q.x j ρ χ b) = refK (Q.x j ρ χ b) + 2) →
    (∀ j p χ π b, RelayEnvAt (Q.spec j (recOf sv n v st p) χ b) (Q.sts j (recOf sv n v st p) χ b)
      (Q.ins j p χ π b)) →
    (∀ j p χ π b, RelayReads (Q.spec j (recOf sv n v st p) χ b) (Q.ins j p χ π b)
      fun ω => (Q.run CoinOpen.shaH (fun i w _ => (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.proverD
          (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).fillOf (fun x => refK x + 2) i.2 w)
        (recOf sv n v st) p χ j π b ω).msgs.map (·.1)) →
    (∀ j p χ π b o, ENNReal.ofReal (Loop.anyErr Masked.dataE (Q.i j p χ π b).1
        (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t (Q.x j (recOf sv n v st p) χ b)
        (Q.w j p χ π b) Θ (Q.Vs j (recOf sv n v st p) χ b) (Q.θ j (recOf sv n v st p) χ b) o) ≤ εS j o) →
    ∀ (s : Tup (fun _ => Bool × ℕ) ℓ) (E : RegistrationRecord S O T R → Xs → Tup (fun _ => List ByteArray) ℓ → Prop),
      prCoin (fun q : (K × (Fin m → S)) × Xs × Tup (fun _ =>
          ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
            (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand) ×
            (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Chk) ℓ =>
          sv.admits q.1.1 (recOf sv n v st q.1) ∧
            Tup.outs (Q.outc (fun x τ => goesOnAt (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).stop (refK x + 1) τ)
                CoinOpen.shaH (fun i w _ => (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.proverD
                  (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).fillOf (fun x => refK x + 2) i.2 w)
                (recOf sv n v st) q.1 q.2.1)
              (Q.rel q.1 q.2.1) ℓ q.2.2 = s ∧
              E (recOf sv n v st q.1) q.2.1 (Tup.rels (Q.rel q.1 q.2.1) ℓ q.2.2)) ≤
        m * (avg (fun q : Xs × Tup CB ℓ =>
            sv.ideal n m v₀ (fun ρ => E ρ q.1 (Tup.sims (Q.sim (Loop.anySim Masked.dataE
              (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t) CoinOpen.shaH eC ρ q.1) ℓ s
                q.2)) +
              sv.prfAdvantage n m v (fun ρ => E ρ q.1 (Tup.sims (Q.sim (Loop.anySim Masked.dataE
                (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t) CoinOpen.shaH eC ρ q.1) ℓ
                  s q.2))) +
            N * (2 * ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash))) +
          sv.forgeAdvantage n m v (st k₀ v) + Tup.errs (fun j o => εS j o.1) ℓ s

theorem SampledAuditAnyRelayDeployed : Flock.Guarantees.SampledAuditAnyRelayDeployed := by
  intro K S V O T R Xs Θ D _ _ _ _ _ _ _ _ N CB _ _ sv Hc dig g hout n m v v₀ st k₀ ℓ t ht Rl hV L₀ src Q eC εS hown
    hok hR hk hE hRd hε s E
  exact sampledAudit_zkOutRelay
    (ok := fun i : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals ×
        (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Worker =>
      (CoinLayer.session Loop.loopNr Masked.dataE).UnCausal i.1 ∧
        LeanRelease.TrustsStop (CoinLayer.session Loop.loopNr Masked.dataE) i.1
          (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)) CoinOpen.shaH i.2 Rl)
    (fun i hi => (SessionLoopZKAnyDeployed i.1 t ht Rl hV hi.1 L₀ src i.2 hi.2).2) sv n m v v₀ st k₀ ℓ Q eC εS _ hown
    hok hR hk hE hRd hε (fun A => ideal_hm96 sv ZkView.My ZkView.saltHash Hc dig g hout n m v v₀ A) s E

end Flock.SecurityProofs
