module

public import Proofs.Pous.Obligation
public import Proofs.Pous.PeriodAt
public import Proofs.Pous.PeriodDeployed
public import Proofs.Pous.Pooled
public import Properties.Pous.Period.Witnesses
public import Proofs.Properties.Pous.SetupAccepts

/-!
# PoUS's per-period guarantees: their honest lemmas at one instance

`periodHeld_honest`, `obligationHeld_honest`, `periodHeldAt_honest` and `periodHeldDemo_honest` hold at every run and
audit meeting their hypotheses; here each is taken at one built instance (`Properties.Pous.Period.Witnesses`), so its
premises are shown to hold together, the certificate `HONEST_S` fitting the state bound beside the device's on-chip
state and side channel (`Fits`):

* `periodHeld`, `periodHeldDeployed`: the honest period (`honestRun` on `honestDevice`, one tick a round) and the honest
  strategy against `sidAudit`, a 32-byte session id with nothing else in it, at the deployed budget
  `(RING_D, RING_Q, RING_L)`; `periodHeld_accepted`, `periodHeldDeployed_accepted`: the same against an audit that
  accepts (`acceptedAudit`: one trusted key, the honest `C*` and a run on what setup published), under `PrimeP16448`,
  or SHA-512 has a collision.
* `periodCapacity`: the same, on the honest period whose capacity verdict is the verifier's whole decision outside a
  break (`verdictRun`), as `PeriodCapacity` takes the joint run's wiring.
* `obligationHeld`: the same, with the store's tags the snapshot's indices (`tag = id`) and the other tags' keys among
  the other audits' coins (`obligationRun`, `obligationAudit`), some strategy meeting both premises.
* `periodHeldAt`: the honest period at the deployed point `Run.DEPLOYED` and `N = 2^19`, whose `FULL` bits exceed
  `SAt (2^19)`; `periodHeldDemo`: a period at Demo B's `N = 1038·2^16` whose account holds its whole store
  (`demoRun`), above `SAt N`.

`periodCapacity` is nontrivial at the empty period (`emptyRun`), whose capacity verdict accepts every window: its
conclusion fails. The others' nontrivial lemmas are the accept-all events of `periodHeld_nontrivial` (at `2^19`
blocks) and of `periodHeldDemo_nontrivialDemo` (at Demo B's).
-/

@[expose] public section

namespace Pous.Period.SecurityProofs

open Pous.EndToEnd PousCheck PousCheck.Properties
open scoped ENNReal

set_option exponentiation.threshold 20000

theorem zero_mem_ticks : (0 : ℕ) ∈ (honestRun.win .capacity).ticks := by
  simp [honestRun, periodRun, Accounting.Window.ticks, K]

theorem honest_fits (r : Accounting.Run honestDevice Unit Unit Unit
    ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)) (hr : r.win .capacity = ⟨0, K⟩) :
    Fits honestDevice (r.win .capacity).len 1 HONEST_S Pous.EndToEnd.S := by
  rw [hr, S_eq]
  show (HONEST_S : ℝ) * ((K : ℕ) : ℝ) + ((2 ^ 30 : ℕ) : ℝ) + ((∑ _i : Unit, 437248000 * 1 : ℕ) : ℝ) ≤
    ((8169621180 : ℕ) : ℝ)
  norm_num [HONEST_S, K]

/-- **`periodHeld`'s honest lemma at one instance**: the honest period and strategy against `sidAudit` meet the device
model at the deployed budget and the instantiation, and the certificate fits. -/
theorem periodHeld_honestInstance :
    DeviceModel honestRun sidAudit honestStrategy RING_D RING_Q RING_L 1 ∧ Instantiates sidAudit honestStrategy ∧
      Fits honestDevice (honestRun.win .capacity).len 1 HONEST_S Pous.EndToEnd.S := by
  have h := periodHeld_honest honestRun sidAudit RING_D RING_Q RING_L 1 sidAudit_sid
    fun _ => ⟨0, zero_mem_ticks, le_rfl⟩
  exact ⟨h.1, h.2, honest_fits honestRun rfl⟩

/-- **`periodHeldDeployed`'s honest lemma**: `periodHeld_honestInstance`, its budget being the deployed one. -/
theorem periodHeldDeployed_honest :
    DeviceModel honestRun sidAudit honestStrategy RING_D RING_Q RING_L 1 ∧ Instantiates sidAudit honestStrategy ∧
      Fits honestDevice (honestRun.win .capacity).len 1 HONEST_S Pous.EndToEnd.S :=
  periodHeld_honestInstance

/-- **An audit of one accepted play**: the verifier's one trusted key `key`, the 32-byte session id `honestSid`, the
commitment `c`, the store `C`, setup's public output `pub`, the row digests `digests` and the session `r`, the same on
every coin. -/
def acceptedAudit (key : ByteArray) (c : PousCheck.SetupProtocol.Commitment) (C : Array (Array ByteArray))
    (pub : PousCheck.SetupProtocol.Public) (digests : ByteArray) (r : PousCheck.Run.ChallengePhaseRecord) :
    Audits Unit Unit where
  trusted := [key]
  sid := Pous.EndToEnd.SecurityProofs.Honest.honestSid
  commitment _ := c
  store _ := C
  pub _ := pub
  digests _ := digests
  W _ _ := false
  session _ := r
  device _ := ⟨fun _ => []⟩

/-- **`periodHeld`'s honest lemma at an accepted play**: under `PrimeP16448`, at the salt `zeros 24`, there is an audit
whose trusted list is one key and whose store is the honest `C*` of eight segments, against which the honest period
and strategy meet the device model at the deployed budget and the instantiation, the certificate fits, and the audit
accepts (setup accepts `C*` and `Run.check` a run on what it published, `setup_run_accept_sid_or_collision`), or
SHA-512 has a collision. -/
theorem periodHeld_accepted (hP : Pous.P2.Assumptions.PrimeP16448) :
    (∃ P : Audits Unit Unit, P.trusted ≠ [] ∧ (∀ o, (P.store o).size = 8) ∧
        DeviceModel honestRun P honestStrategy RING_D RING_Q RING_L 1 ∧ Instantiates P honestStrategy ∧
        Fits honestDevice (honestRun.win .capacity).len 1 HONEST_S Pous.EndToEnd.S ∧ ∃ ω, P.Accepts ω) ∨
      ∃ u v : ByteArray, u ≠ v ∧ Flock.Sha512.hash u = Flock.Sha512.hash v := by
  rcases PousCheck.Properties.SecurityProofs.Accepts.setup_run_accept_sid_or_collision hP
      (Pous.EndToEnd.SecurityProofs.Honest.zeros_size 24) with ⟨c, C, pub, -, hC, hs, key, r, h⟩ | hcol
  · have h' := periodHeld_honest honestRun
      (acceptedAudit key c C pub
        (PousCheck.Properties.SecurityProofs.Accepts.setupDigests (PousCheck.Prim.zeros 24)) r)
      RING_D RING_Q RING_L 1 (Pous.EndToEnd.SecurityProofs.Honest.zeros_size 32)
      fun _ => ⟨0, zero_mem_ticks, le_rfl⟩
    obtain ⟨x⟩ := (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).nonemptyΩ
    exact Or.inl ⟨_, List.cons_ne_nil _ _, fun _ => hC, h'.1, h'.2, honest_fits honestRun rfl,
      ((), (), x, default, default), hs, h⟩
  · exact Or.inr hcol

/-- **`periodHeldDeployed`'s honest lemma at an accepted play**: `periodHeld_accepted`, its budget being the deployed
one. -/
theorem periodHeldDeployed_accepted (hP : Pous.P2.Assumptions.PrimeP16448) :
    (∃ P : Audits Unit Unit, P.trusted ≠ [] ∧ (∀ o, (P.store o).size = 8) ∧
        DeviceModel honestRun P honestStrategy RING_D RING_Q RING_L 1 ∧ Instantiates P honestStrategy ∧
        Fits honestDevice (honestRun.win .capacity).len 1 HONEST_S Pous.EndToEnd.S ∧ ∃ ω, P.Accepts ω) ∨
      ∃ u v : ByteArray, u ≠ v ∧ Flock.Sha512.hash u = Flock.Sha512.hash v :=
  periodHeld_accepted hP

/-- **`periodCapacity`'s honest lemma at one instance**: on the honest period whose capacity verdict is the verifier's
whole decision outside a break, the device model, the instantiation, the certificate's fit and the verdict's wiring
hold. -/
theorem periodCapacity_honestInstance :
    DeviceModel verdictRun sidAudit honestStrategy RING_D RING_Q RING_L 1 ∧ Instantiates sidAudit honestStrategy ∧
      Fits honestDevice (verdictRun.win .capacity).len 1 HONEST_S Pous.EndToEnd.S ∧
        ∀ ω, verdictRun.accepts .capacity ω → sidAudit.Accepts ω ∧ ¬ sidAudit.Break ω := by
  have h := periodHeld_honest verdictRun sidAudit RING_D RING_Q RING_L 1 sidAudit_sid
    fun _ => ⟨0, zero_mem_ticks, le_rfl⟩
  exact ⟨h.1, h.2, honest_fits verdictRun rfl, fun _ hω => hω⟩

/-- **`periodCapacity` is nontrivial**: at the empty period (`emptyRun`), whose capacity verdict accepts every window
and whose account holds nothing, the certificate `HONEST_S` is not sound per run at `δ + ε_max`: every window is
accepted short. -/
theorem periodCapacity_nontrivialEmpty :
    ¬ (emptyRun.CapacitySoundPerRun HONEST_S ((Params.δ : ℝ) + Params.εMax.toReal) ∧
      emptyRun.CapacitySound HONEST_S ((Params.δ : ℝ) + Params.εMax.toReal)) := by
  rintro ⟨h, -⟩
  have h1 : Pouw.pr (fun x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams =>
      emptyRun.accepts .capacity ((), (), x) ∧
        (emptyRun.held ((), (), x) : ℝ) < HONEST_S * (emptyRun.win .capacity).len) = 1 := by
    refine Pouw.pr_eq_one fun x => ⟨trivial, ?_⟩
    have he : emptyRun.held ((), (), x) = K * 0 := held_periodRun 0 _ _
    rw [he]
    show ((K * 0 : ℕ) : ℝ) < (HONEST_S : ℝ) * ((K : ℕ) : ℝ)
    norm_num [HONEST_S, K]
  have h2 := h1.symm.le.trans (h () ())
  rw [εMax_toReal] at h2
  norm_num [Params.δ] at h2

theorem obligationRun_mem_ticks : (0 : ℕ) ∈ (obligationRun.win .capacity).ticks := by
  simp [obligationRun, Accounting.Window.ticks, K]

/-- **`obligationHeld`'s honest lemma at one instance**: at the snapshot `tag = id`, the honest period and a strategy
(the lookup strategy) against `obligationAudit` meet the device model at the deployed budget and the instantiation, and
the certificate fits. -/
theorem obligationHeld_honestInstance :
    ∃ τ : Strategy Unit (Unit × IdKeys),
      DeviceModel obligationRun obligationAudit τ RING_D RING_Q RING_L 1 ∧ Instantiates obligationAudit τ ∧
        Fits honestDevice (obligationRun.win .capacity).len 1 HONEST_S Pous.EndToEnd.S := by
  have h := periodHeld_honest obligationRun obligationAudit RING_D RING_Q RING_L 1 rfl
    fun _ => ⟨0, obligationRun_mem_ticks, le_rfl⟩
  refine ⟨lookup Unit (Unit × IdKeys), h.1, h.2, ?_⟩
  rw [S_eq]
  show (HONEST_S : ℝ) * ((K : ℕ) : ℝ) + ((2 ^ 30 : ℕ) : ℝ) + ((∑ _i : Unit, 437248000 * 1 : ℕ) : ℝ) ≤
    ((8169621180 : ℕ) : ℝ)
  norm_num [HONEST_S, K]

/-- **`periodHeldAt`'s honest lemma at one instance**: at the deployed point and `2^19` blocks, the honest period, whose
account holds `FULL` bits above `SAt (2^19)`, meets the ring-model device premise at the deployed budget, and the
certificate fits `SAt (2^19)`. -/
theorem periodHeldAt_honestInstance :
    RingDeviceAt (2 ^ 19) PousCheck.Run.DEPLOYED honestRun (sidAuditAt (2 ^ 19)) RING_D RING_Q RING_L 1 ∧
      Fits honestDevice (honestRun.win .capacity).len 1 HONEST_S (SAt (2 ^ 19)) := by
  have hS : SAt (2 ^ 19) = 8169621180 := PousP2Ring.stateBound_p2ring_b19 _ _
  refine ⟨ringDeviceAt_honest (2 ^ 19) PousCheck.Run.DEPLOYED honestRun (sidAuditAt (2 ^ 19)) RING_D RING_Q RING_L 1
    fun _ => ⟨0, zero_mem_ticks, ?_⟩, ?_⟩
  · rw [hS]
    show 8169621180 < FULL
    norm_num [FULL]
  · rw [hS]
    show (HONEST_S : ℝ) * ((K : ℕ) : ℝ) + ((2 ^ 30 : ℕ) : ℝ) + ((∑ _i : Unit, 437248000 * 1 : ℕ) : ℝ) ≤
      ((8169621180 : ℕ) : ℝ)
    norm_num [HONEST_S, K]

theorem demoRun_mem_ticks : (0 : ℕ) ∈ (demoRun.win .capacity).ticks := by
  simp [demoRun, Accounting.Window.ticks, K]

/-- **`periodHeldDemo`'s honest lemma at one instance**: at Demo B's point and `N = 1038·2^16`, a period whose account
holds the whole store, above `SAt N`, meets the ring-model device premise at the deployed budget, and the certificate
fits Demo B's state bound. -/
theorem periodHeldDemo_honestInstance :
    RingDeviceAt (1038 * 2 ^ 16) demoPoint demoRun (sidAuditAt (1038 * 2 ^ 16)) RING_D RING_Q RING_L 1 ∧
      Fits demoDevice (demoRun.win .capacity).len 1 HONEST_S 1060008348186 := by
  have hS : SAt (1038 * 2 ^ 16) = 1060008348186 := Pous.SecurityProofs.stateBound_p2ring_demo _ _ _ rfl
  refine ⟨ringDeviceAt_honest (1038 * 2 ^ 16) demoPoint demoRun (sidAuditAt (1038 * 2 ^ 16)) RING_D RING_Q RING_L 1
    fun _ => ⟨0, demoRun_mem_ticks, ?_⟩, ?_⟩
  · rw [hS]
    show 1060008348186 < DEMO_FULL
    norm_num [DEMO_FULL]
  · show (HONEST_S : ℝ) * ((K : ℕ) : ℝ) + ((2 ^ 30 : ℕ) : ℝ) + ((∑ _i : Unit, 437248000 * 1 : ℕ) : ℝ) ≤
      ((1060008348186 : ℕ) : ℝ)
    norm_num [HONEST_S, K]

/-- **`periodHeldDemo` is nontrivial at Demo B's store**: `periodHeldAt_nontrivial` at `N = 1038·2^16`. -/
theorem periodHeldDemo_nontrivialDemo :
    ¬ Pouw.pr (fun _ : (ringRO (1038 * 2 ^ 16) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams => True) ≤
      (Params.δ : ℝ) + Params.εMax.toReal :=
  periodHeldAt_nontrivial (1038 * 2 ^ 16)

end Pous.Period.SecurityProofs
