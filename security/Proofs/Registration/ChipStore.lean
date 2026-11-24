module

public import Definitions.Registration.ChipStore
public import Proofs.Poub.Bandwidth.Causal
public import Proofs.ResourceAccounting.Actual.Device

/-!
# The chip-and-store model gives the bandwidth audit's certificate and the device's schedule

`reads_certificate`: under the model's structure (`ChipStore.Certifies`), a store that hears only the chip, and the
random oracle as a location among the outside ports, the bandwidth audit's capacity form at `e = 0`
(`Bandwidth.Guarantees.B3CapacityRO`) holds at the chip. `schedule_fits`: under the model's limits
(`ChipStore.Limits`), the schedule a run induces fits the device with those limits, which is what the joint audit's
covert residual reads.
-/

@[expose] public section

namespace ResourceAccounting.ChipStore.SecurityProofs

open Bandwidth Bandwidth.Causal
open Pous (Bits pr)
open scoped ENNReal

/-- **The bandwidth audit's certificate under the chip-and-store model.** Beyond the model it takes the oracle's
ports among the outside ports (`hout`, `hor`, at most `qt` queries a tick) and the other outside ports' signals not
depending on the oracle (`hside`): neither is a property of the hardware. -/
theorem reads_certificate {L K : Type*} [LinearOrder L] {N b h : ℕ} (S : Store N b × Oracle N b h → Actual.Run L)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K) (P : Ports L K) (t₀ T s qt : ℕ) (β : K → ℕ)
    (H : Certifies S link P t₀ s β) (sidePorts orPorts : List (K × L))
    (hout : P.outsidePorts = sidePorts ++ orPorts) (hstore : HearsChip S link P.chip t₀ T P.storePorts)
    (Q : ℕ → Acts L → List (Fin N × Bits b)) (hQ : ∀ u a, (Q u a).length ≤ qt)
    (hor : OracleLocation S link P.chip t₀ T orPorts Q)
    (hside : ∀ E O O' u, portSignals S link sidePorts (E, O) u = portSignals S link sidePorts (E, O') u)
    (τ : Acts L → Fin N → Bits h) (n k lam : ℕ) :
    pr (fun x : Store N b × Oracle N b h × (Fin n → Fin N) =>
        (∀ i, τ ((S (x.1, x.2.1)).acts ↑P.chip (Set.Ico t₀ (t₀ + T))) (x.2.2 i) = x.2.1 (x.2.2 i, x.1 (x.2.2 i))) ∧
          (P.storePorts.map fun p => β p.1).sum * T + (s + 1) + (sidePorts.map fun p => β p.1).sum * T + lam < k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ n + (n : ℝ≥0∞) / 2 ^ h + (((T * qt).choose k : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  refine Bandwidth.SecurityProofs.B3CapacityRO S H.causal β link H.alphabet P.chip t₀ T s qt (fun r => H.chip r t₀)
    H.fixed P.storePorts sidePorts orPorts ?_ hstore Q hQ hor hside τ n k lam
  intro r g a ha hx hg
  have hw := H.wired r g a ha hx hg
  rwa [hout, ← List.append_assoc] at hw

open Classical in
/-- **The induced schedule fits the device under the chip-and-store model's limits**, in each run: the device has
the chip's and the store's bounds, the work rates, the store link's bits as its bandwidth and each side link's bits as
that side channel's capacity. -/
theorem schedule_fits {Ω L : Type*} {K ι α : Type} [Fintype α] (S : Ω → Actual.Run L)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K) (P : Ports L K) (cls : ∀ r, Fin (S r).A.N → Option OpClass)
    (ends : K → L × L) (s m : ℕ) (β : K → ℕ) (w : OpClass → ℕ) (H : Limits S link P cls ends s m β w)
    (hne : P.chip ≠ P.store) (r : Ω) (acct : Fin (S r).A.N → α) (hbm : K × L) (side : ι → K × L) :
    ((S r).schedule P.chip P.store (cls r) acct (link r) hbm side).Fits
      { chip := s, mem := m, rate := w, bw := β hbm.1, cap := fun i => β (side i).1 } := by
  classical
  refine ResourceAccounting.SecurityProofs.Actual.fits (S r) _ P.chip P.store (cls r) acct
    (𝒳 := {P.chip, P.store}) (M := fun X => if X = P.chip then s else m) hbm side (H.work r) ?_ (by simp) (by simp)
    (by simp) (by simp [hne.symm]) (H.bits r) le_rfl (fun _ => le_rfl)
  intro X hX t
  rcases hX with rfl | rfl
  · simpa using H.chip r t
  · simpa [hne.symm] using H.store r t

end ResourceAccounting.ChipStore.SecurityProofs
