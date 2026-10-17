import Mathlib
import Proofs.Accounting.Bandwidth.Interactive
import Proofs.Accounting.Actual.Source

/-!
# The bandwidth audit on the actual circuit (B3), given a simulation

B3 of `note:console/20261012T0645Z-draft-actual-circuit-model` (v2.1): in the audit's window `[t₀, t₁)`, the cores
`cores` read the store from HBM `hbm` and anything else from `rest` over links. The bound is `AuditInteractive`'s with
`c` the capacity of the links from `rest` across the cores' boundary over the window, under `Links`
(`Bandwidth.SecurityProofs.Actual.audit`), and `AuditInteractiveRO`'s when what `rest` answers ignores the oracle
(`audit_ro`). No `2^e` term: the rest of the prover is addressed by what the cores sent, and compression counts the bits
read.

What the circuits don't give yet is isolated in `Simulates`: for each store and oracle, a chip program that commits the
run's tags and reads no more than the run's cores read from `hbm` and from `rest` in the window, on state laid out from
the store alone. Supplying one at `S = M_cores` is the strategy layer (oracle gates, a strategy's circuit depending on
the coins, causality to lay the state out before the oracle, and a store no gate regenerates at `G = 0`). -/

namespace Bandwidth

open Pous (Bits pr)
open Accounting
open scoped ENNReal

namespace Actual

/-- **A run of the bandwidth audit simulated by a chip program**: `run E O` is the actual circuit of the prover's run on
store `E` and oracle `O`, and `tags E O` the tags it commits. The simulation is a chip program on `S` bits of state laid
out from the store, device memory and side sources answering what the chip sent, whose run in `(E, O)`'s world commits
`tags E O`, reads at most as many bits from device memory as `run E O`'s cores read from `hbm` in `[t₀, t₁)`, and from
side sources as they read from `rest`, regenerates at most `G` blocks and makes at most `q` oracle queries. -/
structure Simulates (N b h S G q : ℕ) {L : Type*} (cores hbm rest : Finset L) (t₀ t₁ : ℕ)
    (run : Store N b → Oracle N b h → Placed L) (tags : Store N b → Oracle N b h → Fin N → Bits h) where
  state : Store N b → Bits S
  mem : Store N b → ℕ → Bool
  side : Store N b → Oracle N b h → List Bool → ℕ → Bool
  chip : Bits S → IChip N b h (Fin N → Bits h)
  tags_eq : ∀ E O, (chip (state E)).run ⟨O, mem E, side E O, E⟩ [] = tags E O
  ingress_le : ∀ E O,
    (chip (state E)).ingressBits ⟨O, mem E, side E O, E⟩ [] ≤ (run E O).ingressFrom cores hbm (Set.Ico t₀ t₁)
  side_le : ∀ E O,
    (chip (state E)).sideBits ⟨O, mem E, side E O, E⟩ [] ≤ (run E O).ingressFrom cores rest (Set.Ico t₀ t₁)
  regens_le : ∀ E O, (chip (state E)).regens ⟨O, mem E, side E O, E⟩ [] ≤ G
  queries_le : ∀ E O, ((chip (state E)).queries ⟨O, mem E, side E O, E⟩ []).length ≤ q

variable {N b h S G q : ℕ} {L : Type*} {cores hbm rest : Finset L} {t₀ t₁ : ℕ}
  {run : Store N b → Oracle N b h → Placed L} {tags : Store N b → Oracle N b h → Fin N → Bits h}

/-- The simulation as an interactive device with side budget `c`: its chip capped at `c` side bits. -/
noncomputable def Simulates.device (sim : Simulates N b h S G q cores hbm rest t₀ t₁ run tags) (c : ℕ) :
    InteractiveDevice N b h S c G q :=
  InteractiveDevice.ofRuns sim.state sim.mem sim.side sim.chip

/-- Under `Links`, the device at `c = linkCapacityFrom ends β cores rest · (t₁ − t₀)` commits the run's tags and reads
at most the run's ingress from `hbm`. -/
theorem Simulates.device_run {E' : Type*} [Fintype E'] {ends : E' → L × L} {β : E' → ℕ}
    (sim : Simulates N b h S G q cores hbm rest t₀ t₁ run tags)
    (link : ∀ E O, Fin (run E O).N → Fin (run E O).N → E') (hL : ∀ E O, (run E O).Links ends β (link E O))
    (E : Store N b) (O : Oracle N b h) :
    (sim.device (linkCapacityFrom ends β cores rest * (t₁ - t₀))).tags E O = tags E O ∧
      (sim.device (linkCapacityFrom ends β cores rest * (t₁ - t₀))).ingress E O ≤
        (run E O).ingressFrom cores hbm (Set.Ico t₀ t₁) := by
  have hc := (sim.side_le E O).trans (Accounting.SecurityProofs.Actual.ingressFrom_le _ (hL E O) cores rest t₀ t₁)
  obtain ⟨ht, hi⟩ := InteractiveDevice.ofRuns_run (c := linkCapacityFrom ends β cores rest * (t₁ - t₀)) (G := G)
    (q := q) sim.state sim.mem sim.side sim.chip E O ⟨hc, sim.regens_le E O, sim.queries_le E O⟩
  exact ⟨ht.trans (sim.tags_eq E O), hi ▸ sim.ingress_le E O⟩

end Actual

namespace SecurityProofs.Actual

open Bandwidth.Actual

variable {N b h S G q : ℕ} {L : Type*} {cores hbm rest : Finset L} {t₀ t₁ : ℕ}
  {run : Store N b → Oracle N b h → Placed L} {tags : Store N b → Oracle N b h → Fin N → Bits h}

/-- **B3 given a simulation**: with `c` the capacity of the links from `rest` across the cores' boundary over the
window, the probability that the verifier accepts the run's tags while the cores read fewer than
`k·b − (S + c + m·b + G·b + λ)` bits from HBM in the window is at most `AuditInteractive`'s bound. -/
theorem audit {E' : Type*} [Fintype E'] {ends : E' → L × L} {β : E' → ℕ}
    (sim : Simulates N b h S G q cores hbm rest t₀ t₁ run tags)
    (link : ∀ E O, Fin (run E O).N → Fin (run E O).N → E') (hL : ∀ E O, (run E O).Links ends β (link E O))
    (s k m lam : ℕ) :
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, tags x.1 x.2.1 (x.2.2 i) = x.2.1 (x.2.2 i, x.1 (x.2.2 i))) ∧
          (run x.1 x.2.1).ingressFrom cores hbm (Set.Ico t₀ t₁) + S + linkCapacityFrom ends β cores rest * (t₁ - t₀)
            + m * b + G * b + lam < k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s
        + (N.choose m : ℝ≥0∞) * 2 ^ (linkCapacityFrom ends β cores rest * (t₁ - t₀)) / 2 ^ (h * m)
        + ((q.choose (k - m) : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  refine (pr_mono fun x hx => ?_).trans
    (AuditInteractive N b h S _ G q s k m lam (sim.device (linkCapacityFrom ends β cores rest * (t₁ - t₀))))
  obtain ⟨ht, hi⟩ := sim.device_run link hL x.1 x.2.1
  refine ⟨fun i => ?_, by omega⟩
  rw [ht]
  exact hx.1 i

/-- **B3 given a simulation whose side sources ignore the oracle**: `AuditInteractiveRO`'s bound, with no relayed-tag
term. -/
theorem audit_ro {E' : Type*} [Fintype E'] {ends : E' → L × L} {β : E' → ℕ}
    (sim : Simulates N b h S G q cores hbm rest t₀ t₁ run tags) (hside : ∀ E O O', sim.side E O = sim.side E O')
    (link : ∀ E O, Fin (run E O).N → Fin (run E O).N → E') (hL : ∀ E O, (run E O).Links ends β (link E O))
    (s k lam : ℕ) :
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, tags x.1 x.2.1 (x.2.2 i) = x.2.1 (x.2.2 i, x.1 (x.2.2 i))) ∧
          (run x.1 x.2.1).ingressFrom cores hbm (Set.Ico t₀ t₁) + S + linkCapacityFrom ends β cores rest * (t₁ - t₀)
            + G * b + lam < k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (q.choose k : ℝ≥0∞) / 2 ^ lam := by
  refine (pr_mono fun x hx => ?_).trans
    (AuditInteractiveRO N b h S _ G q s k lam (sim.device (linkCapacityFrom ends β cores rest * (t₁ - t₀))) hside)
  obtain ⟨ht, hi⟩ := sim.device_run link hL x.1 x.2.1
  refine ⟨fun i => ?_, by omega⟩
  rw [ht]
  exact hx.1 i

end SecurityProofs.Actual

end Bandwidth
