module

public import Proofs.Pous.Obligation
public import Proofs.Pous.PeriodAt
public import Proofs.Pous.PeriodDeployed
public import Proofs.Pous.Pooled

/-!
# PoUS's per-period guarantees: their honest lemmas at one instance

`periodHeld_honest`, `obligationHeld_honest`, `periodHeldAt_honest` and `periodHeldDemo_honest` hold at every run and
audit meeting their hypotheses; here each is taken at one built instance, so its premises are shown to hold together:

* `periodHeld`, `periodHeldDeployed`: the honest period (`honestRun` on `honestDevice`, one tick a round) and the honest
  strategy against `sidAudit`, a 32-byte session id with nothing else in it, at the deployed budget
  `(RING_D, RING_Q, RING_L)`.
* `periodCapacity`: the same, on the honest period whose capacity verdict is the verifier's whole decision outside a
  break (`verdictRun`), as `PeriodCapacity` takes the joint run's wiring.
* `obligationHeld`: the same, with the store's tags the snapshot's indices (`tag = id`) and the other tags' keys among
  the other audits' coins (`obligationRun`, `obligationAudit`).
* `periodHeldAt`: the honest period at the deployed point `Run.DEPLOYED` and `N = 2^19`, whose `FULL` bits exceed
  `SAt (2^19)`; `periodHeldDemo`: a period at Demo B's `N = 1038·2^16` whose account holds its whole store
  (`demoRun`), above `SAt N`.

Their nontrivial lemmas are the accept-all events of `periodHeld_nontrivial` (at `2^19` blocks) and of
`periodHeldDemo_nontrivialDemo` (at Demo B's).
-/

@[expose] public section

namespace Pous.Period

open Pous.EndToEnd PousCheck PousCheck.Properties

set_option exponentiation.threshold 20000

instance : NeZero (1038 * 2 ^ 16) := ⟨by norm_num⟩

/-- The honest period whose capacity verdict is the verifier's whole decision on `sidAudit` outside a break. -/
def verdictRun : Accounting.Run honestDevice Unit Unit Unit
    ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) :=
  { honestRun with accepts := fun _ ω => sidAudit.Accepts ω ∧ ¬ sidAudit.Break ω }

/-- The other tags' keys when the store's tags are the snapshot's `2^19` indices. -/
abbrev IdKeys : Type := Mutable.RestKeys (id : Fin (2 ^ 19) → Fin (2 ^ 19)) 16448 (2 ^ 16448 - 21065)

/-- The honest period with the other tags' keys among the other audits' coins. -/
def obligationRun : Accounting.Run honestDevice Unit Unit (Unit × IdKeys)
    ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) where
  acct _ := ()
  win _ := ⟨0, K⟩
  schedule _ _ := ⟨fun _ _ => 0, fun _ => 0, fun _ _ => 0, fun _ => 0, fun _ => FULL⟩
  fits := by
    intro _ _
    simp [Accounting.Step.Fits, honestDevice]
  accepts _ _ := True

/-- `sidAudit` with the other tags' keys among the other audits' coins. -/
def obligationAudit : Audits Unit (Unit × IdKeys) where
  trusted := []
  sid := ⟨Array.replicate 32 0⟩
  commitment _ := ⟨ByteArray.empty, ByteArray.empty⟩
  store _ := #[]
  pub _ := ⟨0, ByteArray.empty, ByteArray.empty, ByteArray.empty⟩
  digests _ := ByteArray.empty
  W _ _ := false
  session _ := ⟨default, default, #[]⟩
  device _ := ⟨fun _ => []⟩

/-- `sidAudit` at `N` blocks. -/
def sidAuditAt (N : ℕ) [NeZero N] : AuditsAt N Unit Unit where
  trusted := []
  sid := ⟨Array.replicate 32 0⟩
  commitment _ := ⟨ByteArray.empty, ByteArray.empty⟩
  store _ := #[]
  pub _ := ⟨0, ByteArray.empty, ByteArray.empty, ByteArray.empty⟩
  digests _ := ByteArray.empty
  W _ _ := false
  session _ := ⟨default, default, #[]⟩
  device _ := ⟨fun _ => []⟩

/-- Demo B's whole store, in bits. -/
def DEMO_FULL : ℕ := 1038 * 2 ^ 16 * 16448

/-- `honestDevice` with Demo B's store as its device memory. -/
def demoDevice : Accounting.Device Unit := ⟨2 ^ 30, DEMO_FULL, fun _ => 0, 0, fun _ => 437248000⟩

/-- A period at Demo B's store whose capacity account holds the whole store at every tick. -/
def demoRun : Accounting.Run demoDevice Unit Unit Unit
    ((ringRO (1038 * 2 ^ 16) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) where
  acct _ := ()
  win _ := ⟨0, K⟩
  schedule _ _ := ⟨fun _ _ => 0, fun _ => 0, fun _ _ => 0, fun _ => 0, fun _ => DEMO_FULL⟩
  fits := by
    intro _ _
    simp [Accounting.Step.Fits, demoDevice]
  accepts _ _ := True

/-- Demo B's point: its codec's parameters at `1038` segments, the deployed rounds and limit. -/
def demoPoint : PousCheck.Run.Point :=
  ⟨⟨PousCheck.P2.Params.deployed.w, 21065, 2 ^ 16, 1038⟩, PousCheck.Run.DEPLOYED.k, PousCheck.Run.DEPLOYED.limitNs⟩

end Pous.Period

namespace Pous.Period.SecurityProofs

open Pous.EndToEnd PousCheck PousCheck.Properties
open scoped ENNReal

set_option exponentiation.threshold 20000

theorem zero_mem_ticks : (0 : ℕ) ∈ (honestRun.win .capacity).ticks := by
  simp [honestRun, periodRun, Accounting.Window.ticks, K]

/-- **`periodHeld`'s honest lemma at one instance**: the honest period and strategy against `sidAudit` meet the device
model at the deployed budget and the instantiation. -/
theorem periodHeld_honestInstance :
    DeviceModel honestRun sidAudit honestStrategy RING_D RING_Q RING_L 1 ∧ Instantiates sidAudit honestStrategy :=
  periodHeld_honest honestRun sidAudit RING_D RING_Q RING_L 1 sidAudit_sid fun _ => ⟨0, zero_mem_ticks, le_rfl⟩

/-- **`periodHeldDeployed`'s honest lemma**: `periodHeld_honestInstance`, its budget being the deployed one. -/
theorem periodHeldDeployed_honest :
    DeviceModel honestRun sidAudit honestStrategy RING_D RING_Q RING_L 1 ∧ Instantiates sidAudit honestStrategy :=
  periodHeld_honestInstance

/-- **`periodCapacity`'s honest lemma at one instance**: on the honest period whose capacity verdict is the verifier's
whole decision outside a break, the device model, the instantiation and the verdict's wiring hold. -/
theorem periodCapacity_honestInstance :
    DeviceModel verdictRun sidAudit honestStrategy RING_D RING_Q RING_L 1 ∧ Instantiates sidAudit honestStrategy ∧
      ∀ ω, verdictRun.accepts .capacity ω → sidAudit.Accepts ω ∧ ¬ sidAudit.Break ω := by
  have h := periodHeld_honest verdictRun sidAudit RING_D RING_Q RING_L 1 sidAudit_sid
    fun _ => ⟨0, zero_mem_ticks, le_rfl⟩
  exact ⟨h.1, h.2, fun _ hω => hω⟩

theorem obligationRun_mem_ticks : (0 : ℕ) ∈ (obligationRun.win .capacity).ticks := by
  simp [obligationRun, Accounting.Window.ticks, K]

/-- **`obligationHeld`'s honest lemma at one instance**: at the snapshot `tag = id`, the honest period and the lookup
strategy against `obligationAudit` meet the device model at the deployed budget and the instantiation. -/
theorem obligationHeld_honestInstance :
    DeviceModel obligationRun obligationAudit (lookup Unit (Unit × IdKeys)) RING_D RING_Q RING_L 1 ∧
      Instantiates obligationAudit (lookup Unit (Unit × IdKeys)) :=
  periodHeld_honest obligationRun obligationAudit RING_D RING_Q RING_L 1 rfl
    fun _ => ⟨0, obligationRun_mem_ticks, le_rfl⟩

/-- **`periodHeldAt`'s honest lemma at one instance**: at the deployed point and `2^19` blocks, the honest period, whose
account holds `FULL` bits above `SAt (2^19)`, meets the ring-model device premise at the deployed budget. -/
theorem periodHeldAt_honestInstance :
    RingDeviceAt (2 ^ 19) PousCheck.Run.DEPLOYED honestRun (sidAuditAt (2 ^ 19)) RING_D RING_Q RING_L 1 := by
  have hS : SAt (2 ^ 19) = 8169621180 := PousP2Ring.stateBound_p2ring_b19 _ _
  refine ringDeviceAt_honest (2 ^ 19) PousCheck.Run.DEPLOYED honestRun (sidAuditAt (2 ^ 19)) RING_D RING_Q RING_L 1
    fun _ => ⟨0, zero_mem_ticks, ?_⟩
  rw [hS]
  show 8169621180 < FULL
  norm_num [FULL]

theorem demoRun_mem_ticks : (0 : ℕ) ∈ (demoRun.win .capacity).ticks := by
  simp [demoRun, Accounting.Window.ticks, K]

/-- **`periodHeldDemo`'s honest lemma at one instance**: at Demo B's point and `N = 1038·2^16`, a period whose account
holds the whole store, above `SAt N`, meets the ring-model device premise at the deployed budget. -/
theorem periodHeldDemo_honestInstance :
    RingDeviceAt (1038 * 2 ^ 16) demoPoint demoRun (sidAuditAt (1038 * 2 ^ 16)) RING_D RING_Q RING_L 1 := by
  have hS : SAt (1038 * 2 ^ 16) = 1060008348186 := Pous.SecurityProofs.stateBound_p2ring_demo _ _ _ rfl
  refine ringDeviceAt_honest (1038 * 2 ^ 16) demoPoint demoRun (sidAuditAt (1038 * 2 ^ 16)) RING_D RING_Q RING_L 1
    fun _ => ⟨0, demoRun_mem_ticks, ?_⟩
  rw [hS]
  show 1060008348186 < DEMO_FULL
  norm_num [DEMO_FULL]

/-- **`periodHeldDemo` is nontrivial at Demo B's store**: `periodHeldAt_nontrivial` at `N = 1038·2^16`. -/
theorem periodHeldDemo_nontrivialDemo :
    ¬ Pouw.pr (fun _ : (ringRO (1038 * 2 ^ 16) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams => True) ≤
      (Params.δ : ℝ) + Params.εMax.toReal :=
  periodHeldAt_nontrivial (1038 * 2 ^ 16)

end Pous.Period.SecurityProofs
