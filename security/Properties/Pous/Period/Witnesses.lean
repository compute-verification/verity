module

public import Properties.Pous.PeriodAt
public import Properties.Pous.Pooled

/-!
# The instances PoUS's per-period guarantees' honest lemmas are stated at

Beside `honestRun`, `honestStrategy` and `sidAudit` (`Properties.Pous.PeriodDeployed`, `Properties.Pous.Pooled`):

* `verdictRun`: the honest period whose capacity verdict is the verifier's whole decision on `sidAudit` outside a break,
  as `PeriodCapacity` takes the joint run's wiring.
* `obligationRun` and `obligationAudit`: the honest period and `sidAudit` with the other tags' keys among the other
  audits' coins, at the snapshot `tag = id` (`IdKeys`), as `ObligationHeld` takes them.
* `sidAuditAt N`: `sidAudit` at `N` blocks, as `PeriodHeldAt` and `PeriodHeldDemo` take an audit.
* `demoDevice`, `demoRun` and `demoPoint`: `honestDevice` with Demo B's store, `1038·2^16` blocks, as its device
  memory, a period whose capacity account holds the whole store at every tick, and Demo B's point.

Each run's capacity verdict is its audit's decision outside a break.
-/

@[expose] public section

namespace Pous.Period

open Pous.EndToEnd PousCheck PousCheck.Properties

set_option exponentiation.threshold 20000

instance instNeZeroDemoBlocks : NeZero (1038 * 2 ^ 16) := ⟨by norm_num⟩

/-- The honest period whose capacity verdict is the verifier's whole decision on `sidAudit` outside a break. -/
def verdictRun : Accounting.Run honestDevice Unit Unit Unit
    ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) :=
  { honestRun with accepts := fun _ ω => sidAudit.Accepts ω ∧ ¬ sidAudit.Break ω }

/-- The other tags' keys when the store's tags are the snapshot's `2^19` indices. -/
abbrev IdKeys : Type := Mutable.RestKeys (id : Fin (2 ^ 19) → Fin (2 ^ 19)) 16448 (2 ^ 16448 - 21065)

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

/-- The honest period with the other tags' keys among the other audits' coins, its capacity verdict
`obligationAudit`'s decision outside a break. -/
def obligationRun : Accounting.Run honestDevice Unit Unit (Unit × IdKeys)
    ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) where
  acct _ := ()
  win _ := ⟨0, K⟩
  schedule _ _ := ⟨fun _ _ => 0, fun _ => 0, fun _ _ => 0, fun _ => 0, fun _ => FULL⟩
  fits := by
    intro _ _
    simp [Accounting.Step.Fits, honestDevice]
  accepts _ ω := obligationAudit.Accepts ω ∧ ¬ obligationAudit.Break ω

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

/-- Demo B's point: its codec's parameters at `1038` segments, the deployed rounds and limit. -/
def demoPoint : PousCheck.Run.Point :=
  ⟨⟨PousCheck.P2.Params.deployed.w, 21065, 2 ^ 16, 1038⟩, PousCheck.Run.DEPLOYED.k, PousCheck.Run.DEPLOYED.limitNs⟩

/-- A period at Demo B's store whose capacity account holds the whole store at every tick, its capacity verdict
`sidAuditAt`'s decision at `demoPoint` outside a break. -/
def demoRun : Accounting.Run demoDevice Unit Unit Unit
    ((ringRO (1038 * 2 ^ 16) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) where
  acct _ := ()
  win _ := ⟨0, K⟩
  schedule _ _ := ⟨fun _ _ => 0, fun _ => 0, fun _ _ => 0, fun _ => 0, fun _ => DEMO_FULL⟩
  fits := by
    intro _ _
    simp [Accounting.Step.Fits, demoDevice]
  accepts _ ω := (sidAuditAt (1038 * 2 ^ 16)).Accepts demoPoint ω ∧ ¬ (sidAuditAt (1038 * 2 ^ 16)).Break demoPoint ω

end Pous.Period
