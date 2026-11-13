module

public import Proofs.Flock.SessionLoop.SampledRelay
public import Proofs.Flock.ZkAny.SampledChosenTwo

/-!
# A two-stage sampled audit's view at any values, the auditor choosing the first-stage sample, at the relay's frames

`Flock.Guarantees.SampledAuditTwoChosenAnyRelayDeployed` is `Flock.Guarantees.SampledAuditTwoChosenAny` at the deployed
session loop: its sampled sessions are `ℓ` sessions of the deployed session loop through the deployed held relay, as
`Flock.Guarantees.SampledAuditAnyRelayDeployed` has them for one stage, and each stream commits its values as hm96
leaves at the pinned key. The boundary is the registrations on the ZK server's record before the first-stage sample.
The auditor reads its admitted record and chooses that sample with coins of its own. The interior is the value
service's interior commits (`Flock.Firewall.Salts.interiorAnswer`): a stream of its own under a key `k₁` uniform and
independent of the boundary's randomness and of the auditor's coins, whose values, layouts and stack are any functions
of the boundary's record and those coins. What the audit sees of each session is the held relay's frames
(`ZKOutRelay`).

* `recChosenOf`: what the auditor holds before the second stage: the boundary's record, its coins and the interior's
  record.
* `sampledAuditTwoChosen_zkOutRelay`: `SampledAuditTwoChosenAny` at `ℓ` sessions, in sequence, of a protocol that
  meets `ZKOutRelay`, each through the held relay.
* **`Flock.SecurityProofs.SampledAuditTwoChosenAnyRelayDeployed`**: `sampledAuditTwoChosen_zkOutRelay` at the second
  conjunct of `Loop.ZkAnyAt` (`SessionLoopZKAnyDeployed`), and `ideal_hm96_stream` for each stream's hiding.

The interior's key here is uniform. The deployed one is sampled from the operating system with the interior session's
salts (`Values.answer`'s `interior`), no function of the boundary's key.
-/

@[expose] public section

namespace FlockSoundness.ZK

open Claims Claims.ZKServer Verity Verity.Game Verity.ValueSalts Verity.Assumptions
open scoped ENNReal

/-- **What the auditor holds before the second stage, having chosen the first-stage sample**: at both streams'
randomness `q.1` and the auditor's coins `q.2`, the boundary's record, the coins and the interior's record. -/
def recChosenOf {K S Ys V O T R K₁ S₁ Ys₁ W O₁ T₁ R₁ X₁ : Type} (sv : Service K S Ys V O T R) {r : ℕ} {m : Fin r → ℕ}
    (v : Fin r → V) (st : StreamStack K V S O T R r m) (sv₁ : Service K₁ S₁ Ys₁ W O₁ T₁ R₁) {r₁ : ℕ}
    {m₁ : Fin r₁ → ℕ} (w : StreamRecord r S O T R → X₁ → Fin r₁ → W)
    (st₁ : StreamRecord r S O T R → X₁ → StreamStack K₁ W S₁ O₁ T₁ R₁ r₁ m₁)
    (q : ((K × ((i : Fin r) → Fin (m i) → S)) × (K₁ × ((i : Fin r₁) → Fin (m₁ i) → S₁))) × X₁) :
    StreamRecord r S O T R × X₁ × StreamRecord r₁ S₁ O₁ T₁ R₁ :=
  (streamRecOf sv v st q.1.1, q.2, interiorRecOf sv v st sv₁ w st₁ q.1 q.2)

/-- **A two-stage sampled audit's view at any values, the auditor choosing the first-stage sample, its sampled sessions
`ℓ` sessions of one protocol through the held relay, in sequence.** At every boundary and interior stream, stacks and
keys of `SampledAuditTwoChosenAny`, and every `ℓ` sessions `Q` of a protocol that meets `ZKOutRelay` at the prover
`Pv i` within `ε i` wherever its parameters `i` are `ok`, each session's statement, `V*`, `V*`'s tape and setup a
function of what the auditor holds before the second stage (`recChosenOf`), the second-stage coins and the earlier
sessions' frames: where neither stack reads a key, every session's parameters are `ok`, its statement and witness are
in `Rl`, its verifier runs the protocol's `k` rounds, its relay's host gives it its setup (`RelayEnvAt`) and what it
read at each model slot is the model's message there (`RelayReads`), its error is at most `εS j` at each verdict, its
simulator's coins are those of `CB j` through `eC`, and each stream's ideal values are within `δ` and `δ₁` of its
dummy values, the audit's view of the sessions' frames at their outcomes `s` (each a verdict and how many frames left)
is `SampledAuditTwoChosenAny`'s bound, with the sessions' simulators run in turn (`Tup.sims`) and their errors at the
verdicts added. -/
theorem sampledAuditTwoChosen_zkOutRelay {K S Ys V O T R K₁ S₁ Ys₁ W O₁ T₁ R₁ X₁ X₂ ι Wt St N₀ Ω₀ Θ : Type}
    [Fintype K] [Fintype S] [Fintype Ys] [Fintype T] [Fintype R] [Fintype K₁] [Fintype S₁] [Fintype Ys₁] [Fintype T₁]
    [Fintype R₁] [Fintype X₁] [Fintype X₂] [Fintype N₀] [Nonempty N₀] [Fintype Ω₀] [Nonempty Ω₀] {CB : ℕ → Type}
    [∀ j, Fintype (CB j)] [∀ j, Nonempty (CB j)] {H : Msg → Msg} {k : Msg → ℕ}
    {Pv : ι → Wt → Msg → Prover (N₀ × Ω₀)} {Rl : Rel Wt} {out : Msg → List (Msg × Out) → St} {Sm : SimOut St}
    {ε : ι → (x : Msg) → Wt → (Θ' : Type) → VStar Θ' → Θ' → St → ℝ} {ok : ι → Prop}
    (hZ : ∀ i, ok i → ZKOutRelay (fun _ => N₀) (fun _ => Ω₀) H k (Pv i) Rl out Sm (ε i))
    (sv : Service K S Ys V O T R) (r : ℕ) (m : Fin r → ℕ) (b : ℕ) (v v₀ : Fin r → V)
    (st : StreamStack K V S O T R r m) (k₀ : K) (sv₁ : Service K₁ S₁ Ys₁ W O₁ T₁ R₁) (r₁ : ℕ) (m₁ : Fin r₁ → ℕ)
    (b₁ : ℕ) (w w₀ : StreamRecord r S O T R → X₁ → Fin r₁ → W)
    (st₁ : StreamRecord r S O T R → X₁ → StreamStack K₁ W S₁ O₁ T₁ R₁ r₁ m₁) (k₁ : K₁) (ℓ : ℕ)
    (Q : RelaySessions (StreamRecord r S O T R × X₁ × StreamRecord r₁ S₁ O₁ T₁ R₁) X₂
      (((K × ((i : Fin r) → Fin (m i) → S)) × (K₁ × ((i : Fin r₁) → Fin (m₁ i) → S₁))) × X₁) ι Wt (N₀ × Ω₀) Θ)
    (eC : ∀ j ρ χ b, CB j ≃ Sm.C (Q.x j ρ χ b)) (εS : ℕ → St → ℝ≥0∞) (δ δ₁ : ℝ≥0∞)
    (hown : ZkServerOwnUser st) (hown₁ : ∀ ρ α, ZkServerOwnUser (st₁ ρ α))
    (hok : ∀ j p χ π b, ok (Q.i j p χ π b))
    (hR : ∀ j p χ π b, Rl (Q.x j (recChosenOf sv v st sv₁ w st₁ p) χ b) (Q.w j p χ π b))
    (hk : ∀ j ρ χ b, (Q.Vs j ρ χ b).k (Q.x j ρ χ b) = k (Q.x j ρ χ b))
    (hE : ∀ j p χ π b, RelayEnvAt (Q.spec j (recChosenOf sv v st sv₁ w st₁ p) χ b)
      (Q.sts j (recChosenOf sv v st sv₁ w st₁ p) χ b) (Q.ins j p χ π b))
    (hRd : ∀ j p χ π b, RelayReads (Q.spec j (recChosenOf sv v st sv₁ w st₁ p) χ b) (Q.ins j p χ π b)
      fun ω => (Q.run H Pv (recChosenOf sv v st sv₁ w st₁) p χ j π b ω).msgs.map (·.1))
    (hε : ∀ j p χ π b o, ENNReal.ofReal (ε (Q.i j p χ π b) (Q.x j (recChosenOf sv v st sv₁ w st₁ p) χ b)
      (Q.w j p χ π b) Θ (Q.Vs j (recChosenOf sv v st sv₁ w st₁ p) χ b) (Q.θ j (recChosenOf sv v st sv₁ w st₁ p) χ b)
      o) ≤ εS j o)
    (hδ : ∀ P A, sv.streamIdeal m v P A ≤ sv.streamIdeal m v₀ P A + δ)
    (hδ₁ : ∀ ρ α P A, sv₁.streamIdeal m₁ (w ρ α) P A ≤ sv₁.streamIdeal m₁ (w₀ ρ α) P A + δ₁)
    (s : Tup (fun _ => St × ℕ) ℓ)
    (E : StreamRecord r S O T R → X₁ → StreamRecord r₁ S₁ O₁ T₁ R₁ → X₂ → Tup (fun _ => List ByteArray) ℓ → Prop) :
    prCoin (fun q : ((K × ((i : Fin r) → Fin (m i) → S)) × (K₁ × ((i : Fin r₁) → Fin (m₁ i) → S₁))) ×
          X₁ × X₂ × Tup (fun _ => N₀ × Ω₀) ℓ =>
        (sv.streamAdmits q.1.1.1 b (streamRecOf sv v st q.1.1) ∧
          sv₁.streamAdmits q.1.2.1 b₁ (interiorRecOf sv v st sv₁ w st₁ q.1 q.2.1)) ∧
          Tup.outs (Q.outc out H Pv (recChosenOf sv v st sv₁ w st₁) (q.1, q.2.1) q.2.2.1)
              (Q.rel (q.1, q.2.1) q.2.2.1) ℓ q.2.2.2 = s ∧
            E (streamRecOf sv v st q.1.1) q.2.1 (interiorRecOf sv v st sv₁ w st₁ q.1 q.2.1) q.2.2.1
              (Tup.rels (Q.rel (q.1, q.2.1) q.2.2.1) ℓ q.2.2.2)) ≤
      avg (fun q : X₁ × X₂ × Tup CB ℓ =>
          (∑ d₁ ∈ streamChoices r₁ m₁ b₁, avg fun y : Fin r₁ → S₁ × Ys₁ × T₁ × R₁ =>
            (∑ d ∈ streamChoices r m b,
              (sv.streamIdeal m v₀ (passed d) (fun ρ =>
                  E ρ q.1 (idealStreamRec sv₁ m₁ (w₀ ρ q.1) (passed d₁) y) q.2.1
                    (Tup.sims (Q.sim Sm H eC (ρ, q.1, idealStreamRec sv₁ m₁ (w₀ ρ q.1) (passed d₁) y) q.2.1) ℓ s
                      q.2.2)) + δ +
                sv.streamPrfAdvantage v d (fun ρ =>
                  E ρ q.1 (idealStreamRec sv₁ m₁ (w₀ ρ q.1) (passed d₁) y) q.2.1
                    (Tup.sims (Q.sim Sm H eC (ρ, q.1, idealStreamRec sv₁ m₁ (w₀ ρ q.1) (passed d₁) y) q.2.1) ℓ s
                      q.2.2)))) +
              sv.streamForgeAdvantage v (st k₀ v)) +
          avg fun p : K × ((i : Fin r) → Fin (m i) → S) =>
            (∑ d₁ ∈ streamChoices r₁ m₁ b₁, (δ₁ + sv₁.streamPrfAdvantage (w (streamRecOf sv v st p) q.1) d₁
              fun ρ₁ => E (streamRecOf sv v st p) q.1 ρ₁ q.2.1
                (Tup.sims (Q.sim Sm H eC (streamRecOf sv v st p, q.1, ρ₁) q.2.1) ℓ s q.2.2))) +
              sv₁.streamForgeAdvantage (w (streamRecOf sv v st p) q.1)
                (st₁ (streamRecOf sv v st p) q.1 k₁ (w (streamRecOf sv v st p) q.1))) +
        Tup.errs (fun j o => εS j o.1) ℓ s := by
  have hS : ∀ p α χ₂, PerOutcome
      (Tup.outs (Q.outc out H Pv (recChosenOf sv v st sv₁ w st₁) (p, α) χ₂) (Q.rel (p, α) χ₂) ℓ)
      (Tup.rels (Q.rel (p, α) χ₂) ℓ) (Tup.sims (Q.sim Sm H eC (recChosenOf sv v st sv₁ w st₁ (p, α)) χ₂) ℓ)
      (Tup.errs (fun j o => εS j o.1) ℓ) := by
    intro p α χ₂
    refine PerOutcome.stream (A := fun _ => N₀ × Ω₀) (C := CB)
      (Q.outc out H Pv (recChosenOf sv v st sv₁ w st₁) (p, α) χ₂) (Q.rel (p, α) χ₂)
      (Q.sim Sm H eC (recChosenOf sv v st sv₁ w st₁ (p, α)) χ₂) (fun j o => εS j o.1) ℓ fun j _ π => ?_
    exact (perOutcome_of_zkOutRelay (hZ _ (hok j (p, α) χ₂ π (Tup.rels (Q.rel (p, α) χ₂) j π)))
      (Q.spec j (recChosenOf sv v st sv₁ w st₁ (p, α)) χ₂ (Tup.rels (Q.rel (p, α) χ₂) j π))
      (Q.sts j (recChosenOf sv v st sv₁ w st₁ (p, α)) χ₂ (Tup.rels (Q.rel (p, α) χ₂) j π))
      (hR j (p, α) χ₂ π (Tup.rels (Q.rel (p, α) χ₂) j π))
      (Q.Vs j (recChosenOf sv v st sv₁ w st₁ (p, α)) χ₂ (Tup.rels (Q.rel (p, α) χ₂) j π))
      (Q.θ j (recChosenOf sv v st sv₁ w st₁ (p, α)) χ₂ (Tup.rels (Q.rel (p, α) χ₂) j π))
      (hk j (recChosenOf sv v st sv₁ w st₁ (p, α)) χ₂ (Tup.rels (Q.rel (p, α) χ₂) j π))
      (Q.ins j (p, α) χ₂ π (Tup.rels (Q.rel (p, α) χ₂) j π))
      (hE j (p, α) χ₂ π (Tup.rels (Q.rel (p, α) χ₂) j π)) (hRd j (p, α) χ₂ π (Tup.rels (Q.rel (p, α) χ₂) j π))
      (eC j (recChosenOf sv v st sv₁ w st₁ (p, α)) χ₂ (Tup.rels (Q.rel (p, α) χ₂) j π))).mono
      fun o => hε j (p, α) χ₂ π (Tup.rels (Q.rel (p, α) χ₂) j π) o.1
  exact Flock.SecurityProofs.SampledAuditTwoChosenAny sv r m b v v₀ st k₀ sv₁ r₁ m₁ b₁ w w₀ st₁ k₁
    (fun p α χ₂ => Tup.outs (Q.outc out H Pv (recChosenOf sv v st sv₁ w st₁) (p, α) χ₂) (Q.rel (p, α) χ₂) ℓ)
    (fun p α χ₂ => Tup.rels (Q.rel (p, α) χ₂) ℓ) (fun ρ α ρ₁ χ₂ => Tup.sims (Q.sim Sm H eC (ρ, α, ρ₁) χ₂) ℓ)
    (Tup.errs (fun j (o : St × ℕ) => εS j o.1) ℓ) δ δ₁ hown hown₁ hS hδ hδ₁ s E

end FlockSoundness.ZK

namespace Flock.SecurityProofs

open Claims Claims.ZKMal Claims.ZKServer
open Verity Verity.Game Verity.ValueSalts Verity.Assumptions FlockSoundness.ZK
open FlockSoundness.Discharge
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **A two-stage sampled audit's view at any values, the auditor choosing the first-stage sample, at the deployed held
relay's frames.** As `SampledAuditTwoChosenAny`, at the deployed session loop. The boundary's value service `sv` and the
interior's `sv₁` each commit their values as hm96 leaves at the pinned key (`N` and `N₁` of them, put together by the
values' public layout, `sh` and `sh₁`), and each stream's dummy values have its values' layouts. The sampled sessions
`Q` are `ℓ` sessions of the deployed session loop at `0 < t` repetitions through the deployed held relay, at `L₀` and
`src`, with a relation `Rl` at whose statements the session's table witnesses are valid (`Complete`). They give each
session's setup `(spec, sts)` and the requests `ins` its relay reads, and the simulators' coins of session `j` are those
of `CB j` through `eC`.

If neither stack reads a key, every session's values are causal and the transport's trusts hold at its worker, its
statement and witness are in `Rl`, its `V*` runs `refK x + 2` rounds, its relay's host gives it its setup
(`RelayEnvAt`), what it read at each model slot is the session loop's message there (`RelayReads`), and its error
(`Loop.anyErr`) is at most `εS j` at each verdict, then at the sessions' outcomes `s` (each its verdict and how many
frames left) and every event `E` on both records, the auditor's coins, the second-stage coins and the frames,
`SampledAuditTwoChosenAny`'s bound holds with the simulated frames, the boundary's hiding `r·N·2γ` and the interior's
`r₁·N₁·2γ`, `γ` the hiding gap at the pinned key, and the sessions' errors at their verdicts. -/
def _root_.Flock.Guarantees.SampledAuditTwoChosenAnyRelayDeployed : Prop :=
  ∀ {K S V O T R K₁ S₁ W O₁ T₁ R₁ X₁ X₂ Θ D Sh Sh₁ : Type} [Fintype K] [Fintype S] [Fintype T] [Fintype R]
    [Nonempty S] [Nonempty T] [Nonempty R] [Fintype K₁] [Fintype S₁] [Fintype T₁] [Fintype R₁] [Nonempty S₁]
    [Nonempty T₁] [Nonempty R₁] [Fintype X₁] [Fintype X₂] {N N₁ : ℕ} {CB : ℕ → Type} [∀ j, Fintype (CB j)]
    [∀ j, Nonempty (CB j)]
    (sv : Service K S (Fin N → ZkView.Salt) V O T R) (sv₁ : Service K₁ S₁ (Fin N₁ → ZkView.Salt) W O₁ T₁ R₁)
    (Hc : ZkView.Bits × ByteArray → D) (dig : V → Fin N → ZkView.Bits) (sh : V → Sh) (g : Sh → (Fin N → D) → O)
    (dig₁ : W → Fin N₁ → ZkView.Bits) (sh₁ : W → Sh₁) (g₁ : Sh₁ → (Fin N₁ → D) → O₁),
    (∀ v y, sv.out v y = g (sh v) fun i => Hc (dig v i + ZkView.My (y i), ZkView.saltHash (y i))) →
    (∀ u y, sv₁.out u y = g₁ (sh₁ u) fun i => Hc (dig₁ u i + ZkView.My (y i), ZkView.saltHash (y i))) →
    ∀ (r : ℕ) (m : Fin r → ℕ) (b : ℕ) (v v₀ : Fin r → V), (∀ i, sh (v i) = sh (v₀ i)) →
    ∀ (st : StreamStack K V S O T R r m) (k₀ : K) (r₁ : ℕ) (m₁ : Fin r₁ → ℕ) (b₁ : ℕ)
      (w w₀ : StreamRecord r S O T R → X₁ → Fin r₁ → W), (∀ ρ α i, sh₁ (w ρ α i) = sh₁ (w₀ ρ α i)) →
    ∀ (st₁ : StreamRecord r S O T R → X₁ → StreamStack K₁ W S₁ O₁ T₁ R₁ r₁ m₁) (k₁ : K₁) (ℓ t : ℕ), 0 < t →
    ∀ (Rl : Rel Claims.Flock.Wit), (CoinLayer.session Loop.loopNr Masked.dataE).Complete Rl →
    ∀ (L₀ : LeanRelease Claims.Flock.Wit
        ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
          (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
      (src : Msg → Claims.Flock.Wit →
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
          (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
        ℕ → _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
      (Q : RelaySessions (StreamRecord r S O T R × X₁ × StreamRecord r₁ S₁ O₁ T₁ R₁) X₂
        (((K × ((i : Fin r) → Fin (m i) → S)) × (K₁ × ((i : Fin r₁) → Fin (m₁ i) → S₁))) × X₁)
        ((CoinLayer.session Loop.loopNr Masked.dataE).SwVals ×
          (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Worker) Claims.Flock.Wit
        (((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
          (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand) ×
          (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Chk) Θ)
      (eC : ∀ j ρ χ b, CB j ≃
        (Loop.anySim Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t).C
          (Q.x j ρ χ b))
      (εS : ℕ → Bool → ℝ≥0∞),
    ZkServerOwnUser st → (∀ ρ α, ZkServerOwnUser (st₁ ρ α)) →
    (∀ j p χ π b, (CoinLayer.session Loop.loopNr Masked.dataE).UnCausal (Q.i j p χ π b).1 ∧
      LeanRelease.TrustsStop (CoinLayer.session Loop.loopNr Masked.dataE) (Q.i j p χ π b).1
        (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)) CoinOpen.shaH (Q.i j p χ π b).2 Rl) →
    (∀ j p χ π b, Rl (Q.x j (recChosenOf sv v st sv₁ w st₁ p) χ b) (Q.w j p χ π b)) →
    (∀ j ρ χ b, (Q.Vs j ρ χ b).k (Q.x j ρ χ b) = refK (Q.x j ρ χ b) + 2) →
    (∀ j p χ π b, RelayEnvAt (Q.spec j (recChosenOf sv v st sv₁ w st₁ p) χ b)
      (Q.sts j (recChosenOf sv v st sv₁ w st₁ p) χ b) (Q.ins j p χ π b)) →
    (∀ j p χ π b, RelayReads (Q.spec j (recChosenOf sv v st sv₁ w st₁ p) χ b) (Q.ins j p χ π b)
      fun ω => (Q.run CoinOpen.shaH (fun i u _ => (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.proverD
          (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).fillOf (fun x => refK x + 2) i.2 u)
        (recChosenOf sv v st sv₁ w st₁) p χ j π b ω).msgs.map (·.1)) →
    (∀ j p χ π b o, ENNReal.ofReal (Loop.anyErr Masked.dataE (Q.i j p χ π b).1
        (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t
        (Q.x j (recChosenOf sv v st sv₁ w st₁ p) χ b) (Q.w j p χ π b) Θ
        (Q.Vs j (recChosenOf sv v st sv₁ w st₁ p) χ b) (Q.θ j (recChosenOf sv v st sv₁ w st₁ p) χ b) o) ≤ εS j o) →
    ∀ (s : Tup (fun _ => Bool × ℕ) ℓ)
      (E : StreamRecord r S O T R → X₁ → StreamRecord r₁ S₁ O₁ T₁ R₁ → X₂ → Tup (fun _ => List ByteArray) ℓ → Prop),
      prCoin (fun q : ((K × ((i : Fin r) → Fin (m i) → S)) × (K₁ × ((i : Fin r₁) → Fin (m₁ i) → S₁))) × X₁ × X₂ ×
          Tup (fun _ => ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
            (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand) ×
            (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Chk) ℓ =>
          (sv.streamAdmits q.1.1.1 b (streamRecOf sv v st q.1.1) ∧
            sv₁.streamAdmits q.1.2.1 b₁ (interiorRecOf sv v st sv₁ w st₁ q.1 q.2.1)) ∧
            Tup.outs (Q.outc (fun x τ => goesOnAt (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).stop (refK x + 1) τ)
                CoinOpen.shaH (fun i u _ => (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.proverD
                  (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).fillOf (fun x => refK x + 2) i.2 u)
                (recChosenOf sv v st sv₁ w st₁) (q.1, q.2.1) q.2.2.1)
              (Q.rel (q.1, q.2.1) q.2.2.1) ℓ q.2.2.2 = s ∧
              E (streamRecOf sv v st q.1.1) q.2.1 (interiorRecOf sv v st sv₁ w st₁ q.1 q.2.1) q.2.2.1
                (Tup.rels (Q.rel (q.1, q.2.1) q.2.2.1) ℓ q.2.2.2)) ≤
        avg (fun q : X₁ × X₂ × Tup CB ℓ =>
            (∑ d₁ ∈ streamChoices r₁ m₁ b₁, avg fun y : Fin r₁ → S₁ × (Fin N₁ → ZkView.Salt) × T₁ × R₁ =>
              (∑ d ∈ streamChoices r m b,
                (sv.streamIdeal m v₀ (passed d) (fun ρ =>
                    E ρ q.1 (idealStreamRec sv₁ m₁ (w₀ ρ q.1) (passed d₁) y) q.2.1
                      (Tup.sims (Q.sim (Loop.anySim Masked.dataE
                        (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t) CoinOpen.shaH eC
                          (ρ, q.1, idealStreamRec sv₁ m₁ (w₀ ρ q.1) (passed d₁) y) q.2.1) ℓ s q.2.2)) +
                  r * (N * (2 * ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash))) +
                  sv.streamPrfAdvantage v d (fun ρ =>
                    E ρ q.1 (idealStreamRec sv₁ m₁ (w₀ ρ q.1) (passed d₁) y) q.2.1
                      (Tup.sims (Q.sim (Loop.anySim Masked.dataE
                        (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t) CoinOpen.shaH eC
                          (ρ, q.1, idealStreamRec sv₁ m₁ (w₀ ρ q.1) (passed d₁) y) q.2.1) ℓ s q.2.2)))) +
                sv.streamForgeAdvantage v (st k₀ v)) +
            avg fun p : K × ((i : Fin r) → Fin (m i) → S) =>
              (∑ d₁ ∈ streamChoices r₁ m₁ b₁,
                (r₁ * (N₁ * (2 * ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash))) +
                  sv₁.streamPrfAdvantage (w (streamRecOf sv v st p) q.1) d₁ fun ρ₁ =>
                    E (streamRecOf sv v st p) q.1 ρ₁ q.2.1
                      (Tup.sims (Q.sim (Loop.anySim Masked.dataE
                        (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t) CoinOpen.shaH eC
                          (streamRecOf sv v st p, q.1, ρ₁) q.2.1) ℓ s q.2.2))) +
                sv₁.streamForgeAdvantage (w (streamRecOf sv v st p) q.1)
                  (st₁ (streamRecOf sv v st p) q.1 k₁ (w (streamRecOf sv v st p) q.1))) +
          Tup.errs (fun j o => εS j o.1) ℓ s

theorem SampledAuditTwoChosenAnyRelayDeployed : Flock.Guarantees.SampledAuditTwoChosenAnyRelayDeployed := by
  intro K S V O T R K₁ S₁ W O₁ T₁ R₁ X₁ X₂ Θ D Sh Sh₁ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ N N₁ CB _ _ sv sv₁ Hc dig sh g
    dig₁ sh₁ g₁ hout hout₁ r m b v v₀ hsh st k₀ r₁ m₁ b₁ w w₀ hsh₁ st₁ k₁ ℓ t ht Rl hV L₀ src Q eC εS hown hown₁ hok hR
    hk hE hRd hε s E
  exact sampledAuditTwoChosen_zkOutRelay
    (ok := fun i : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals ×
        (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Worker =>
      (CoinLayer.session Loop.loopNr Masked.dataE).UnCausal i.1 ∧
        LeanRelease.TrustsStop (CoinLayer.session Loop.loopNr Masked.dataE) i.1
          (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)) CoinOpen.shaH i.2 Rl)
    (fun i hi => (SessionLoopZKAnyDeployed i.1 t ht Rl hV hi.1 L₀ src i.2 hi.2).2) sv r m b v v₀ st k₀ sv₁ r₁ m₁ b₁
    w w₀ st₁ k₁ ℓ Q eC εS _ _ hown hown₁ hok hR hk hE hRd hε
    (fun P A => ideal_hm96_stream sv ZkView.My ZkView.saltHash Hc dig sh g hout m v v₀ hsh P A)
    (fun ρ α P A => ideal_hm96_stream sv₁ ZkView.My ZkView.saltHash Hc dig₁ sh₁ g₁ hout₁ m₁ (w ρ α) (w₀ ρ α)
      (hsh₁ ρ α) P A) s E

end Flock.SecurityProofs
