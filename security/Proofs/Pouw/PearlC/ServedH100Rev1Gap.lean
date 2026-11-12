module

public import Proofs.Pouw.PearlC.ServedMixedRev1Gap
public import Proofs.Pouw.PearlC.Rev1Witness

/-!
# rev1 on the H100 record at Llama 3.2 1B's linears: the served point's gap (compute accounting's file)

Lemmas; nothing here is in the lock. What is served is Llama 3.2 1B on one H200, credited under the H100 record
(`pearl-c-h100-v1-h3`, `devH100`) at the inert-work share 1/400. The lock's H100 guarantees
(`Pouw.Guarantees.PearlC.GapH100Rev1_8192`, `_16384` and their per-tile forms) are stated at the cubes 8192³ and 16384³.
These are the same statements at the linears the served model credits, `(n, k)` = (3072, 2048) qkv, (2048, 2048) o,
(16384, 2048) gate_up and (2048, 8192) down, each at any batch size `m`.

* **The record's form on a linear or a set of linears at any `m`:** `pearlCGapDevRev1NK`, `pearlCSampledDevRev1NK`,
  `pearlCGapDevRev1Shapes` and `pearlCSampledDevRev1Shapes` are `ServedRev1Gap`'s and `ServedMixedRev1Gap`'s theorems at
  the record's own `W_ref`, the kernel's at `c = 0` (`pearlCProtocolDevRev1K_zero`, `pearlCTilesDevRev1K_zero`).
* **The gap per linear,** per unit from `TTOutPearlCH100Rev1` at 1/400 and per audit tile from
  `TTOutTilePearlCH100Rev1` at 1/400, `g = 1 − (399/400)²·credit/W_ref`:

  | linear | `(n, k)` | `W_ref/credit` | g | ε_a = g/(1 − g) |
  | --- | --- | --- | --- | --- |
  | qkv | (3072, 2048) | 1021/1020 | 48749/8168000 (0.59683%) | 48749/8119251 (0.60041%) |
  | o | (2048, 2048) | 689/688 | 44357/6890000 (0.64379%) | 44357/6845643 (0.64796%) |
  | gate_up | (16384, 2048) | 5337/5336 | 61437/11860000 (0.51802%) | 61437/11798563 (0.52072%) |
  | down | (2048, 8192) | 671/670 | 69533/10736000 (0.64766%) | 69533/10666467 (0.65188%) |

* **The gap on the served layout,** the four linears in one layout: the down projection's, the max of the four
  (`ServedMixedRev1Gap` says why it is the max and not a mean), `g = 69533/10736000` (0.64766%), ε_a = 69533/10666467
  (0.65188%): `pearlCGapH100Rev1_llama32_1b` and `pearlCSampledH100Rev1_llama32_1b`, and at W1 on `W1Device.h100`, the
  form the lock's H100 guarantees take, `pearlCGapH100Rev1_llama32_1b_w1` and `pearlCSampledH100Rev1_llama32_1b_w1`.
  `epsA_h100Rev1_llama32_1b` is the arithmetic from g to ε_a.

Each is conditional on its TT_OUT form, an open conjecture, and carries its `δPearlC`, which is stated for at most
2^64 oracle calls. The sampled forms bound the audit's miss by `(1 − ε_s)^t` beside it: ε_a is the assumption's gap
alone, not the run's ε.
-/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- Llama 3.2 1B's linears `(n, k)`: qkv, o, gate_up, down. -/
def llama32_1bLinears : Finset (ℕ × ℕ) := {(3072, 2048), (2048, 2048), (16384, 2048), (2048, 8192)}

section
variable {Q R S : Type}

/-- The kernel's `W_ref` at `c = 0` is the record's. -/
theorem wrefDevRev1K_zero (d : PearlCDevice) (s : Shape) : wrefDevRev1K d 0 s = wrefDevRev1 d s := by
  rw [wrefDevRev1K]
  ring

/-- `pearlCProtocolDevRev1K` at `c = 0` is the record's protocol. -/
theorem pearlCProtocolDevRev1K_zero (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) :
    pearlCProtocolDevRev1K d sem ρ 0 = pearlCProtocolDevRev1 d sem ρ := by
  have key : ∀ w : Shape → ℚ, w = wrefDevRev1 d →
      ({ pearlCProtocolDevRev1 d sem ρ with Wref := fun L u => (w (L.shape u) : ℝ) } : Protocol Q R S) =
        pearlCProtocolDevRev1 d sem ρ := by
    rintro w rfl
    rfl
  exact key (wrefDevRev1K d 0) (funext (wrefDevRev1K_zero d))

/-- `pearlCTilesDevRev1K` at `c = 0` is the record's audit tiles. -/
theorem pearlCTilesDevRev1K_zero (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) :
    pearlCTilesDevRev1K d sem ρ 0 = pearlCTilesDevRev1 d sem ρ := by
  have key : ∀ w : Shape → ℚ, w = wrefDevRev1 d →
      ({ pearlCTilesDevRev1 d sem ρ with
          Wref := fun L g =>
            let tl := auditTiling 64 64 L
            let sh := L.shape (tl.unit g)
            ((w sh * tileShare sh (tl.rows g) (tl.cols g) : ℚ) : ℝ)
          Wcred := fun L g act =>
            let tl := auditTiling 64 64 L
            let sh := L.shape (tl.unit g)
            ((w sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ) } : TileRules Q R S) =
        pearlCTilesDevRev1 d sem ρ := by
    rintro w rfl
    rfl
  exact key (wrefDevRev1K d 0) (funext (wrefDevRev1K_zero d))

/-- `pearlCGapDevRev1KNK` at the record's own `W_ref`: TT_OUT rev1 at a device record gives `G_g` on one linear at any
`m` at `g = 1 − (399/400)/ω`, where `wrefDevRev1 = ω·(1 − ρ)·creditDevRev1` at every `m × n × k`. -/
theorem pearlCGapDevRev1NK [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (n k : ℕ) (ω g : ℚ) (hω0 : 0 < ω)
    (hω : ∀ m : ℕ, wrefDevRev1 d ⟨m, n, k⟩ = ω * ((1 - ρ) * creditDevRev1 d ⟨m, n, k⟩))
    (hg : g = 1 - (1 - 1 / 400) / ω) (hTT : TTOutPearlCDevRev1 CM d sem ρ) :
    GGap CM (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDevNK d n k) (g : ℝ) δPearlC := by
  have h := pearlCGapDevRev1KNK CM d sem ρ 0 n k ω g hω0 (fun m => by rw [wrefDevRev1K_zero]; exact hω m) hg hTT
  rw [pearlCProtocolDevRev1K_zero] at h
  exact h

/-- `pearlCSampledDevRev1KNK` at the record's own `W_ref`: TT_OUT rev1 per tile at a device record gives `GGapSampled`
on one linear at any `m` at `g = 1 − (399/400)/ω`, where `wrefDevRev1 = ω·(1 − ρ)·creditDevRev1 ≥ 0` at every
`m × n × k`. -/
theorem pearlCSampledDevRev1NK [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (n k : ℕ) (ω g : ℚ) (hω0 : 0 < ω)
    (hw0 : ∀ m : ℕ, 0 ≤ wrefDevRev1 d ⟨m, n, k⟩)
    (hω : ∀ m : ℕ, wrefDevRev1 d ⟨m, n, k⟩ = ω * ((1 - ρ) * creditDevRev1 d ⟨m, n, k⟩))
    (hg : g = 1 - (1 - 1 / 400) / ω) (hTT : TTOutTilePearlCDevRev1 CM d sem ρ) :
    GGapSampled CM (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) (pearlCDomainDevNK d n k) (g : ℝ)
      δPearlC := by
  have h := pearlCSampledDevRev1KNK CM d sem ρ 0 n k ω g hω0 (fun m => by rw [wrefDevRev1K_zero]; exact hw0 m)
    (fun m => by rw [wrefDevRev1K_zero]; exact hω m) hg hTT
  rw [pearlCProtocolDevRev1K_zero, pearlCTilesDevRev1K_zero] at h
  exact h

/-- `pearlCGapDevRev1KShapes` at the record's own `W_ref`: TT_OUT rev1 at a device record gives `G_g` on a set of
linears at any `m` at `g = 1 − (399/400)/ω`, where `wrefDevRev1 ≤ ω·(1 − ρ)·creditDevRev1` at every `m × n × k` with
`(n, k) ∈ nks`. -/
theorem pearlCGapDevRev1Shapes [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (nks : Finset (ℕ × ℕ)) (ω g : ℚ)
    (hω0 : 0 < ω)
    (hω : ∀ nk ∈ nks, ∀ m : ℕ, wrefDevRev1 d ⟨m, nk.1, nk.2⟩ ≤ ω * ((1 - ρ) * creditDevRev1 d ⟨m, nk.1, nk.2⟩))
    (hg : g = 1 - (1 - 1 / 400) / ω) (hTT : TTOutPearlCDevRev1 CM d sem ρ) :
    GGap CM (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDevShapes d nks) (g : ℝ) δPearlC := by
  have h := pearlCGapDevRev1KShapes CM d sem ρ 0 nks ω g hω0
    (fun nk hnk m => by rw [wrefDevRev1K_zero]; exact hω nk hnk m) hg hTT
  rw [pearlCProtocolDevRev1K_zero] at h
  exact h

/-- `pearlCSampledDevRev1KShapes` at the record's own `W_ref`: TT_OUT rev1 per tile at a device record gives
`GGapSampled` on a set of linears at any `m` at `g = 1 − (399/400)/ω`, where
`0 ≤ wrefDevRev1 ≤ ω·(1 − ρ)·creditDevRev1` at every `m × n × k` with `(n, k) ∈ nks`. -/
theorem pearlCSampledDevRev1Shapes [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (nks : Finset (ℕ × ℕ)) (ω g : ℚ)
    (hω0 : 0 < ω) (hw0 : ∀ nk ∈ nks, ∀ m : ℕ, 0 ≤ wrefDevRev1 d ⟨m, nk.1, nk.2⟩)
    (hω : ∀ nk ∈ nks, ∀ m : ℕ, wrefDevRev1 d ⟨m, nk.1, nk.2⟩ ≤ ω * ((1 - ρ) * creditDevRev1 d ⟨m, nk.1, nk.2⟩))
    (hg : g = 1 - (1 - 1 / 400) / ω) (hTT : TTOutTilePearlCDevRev1 CM d sem ρ) :
    GGapSampled CM (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) (pearlCDomainDevShapes d nks)
      (g : ℝ) δPearlC := by
  have h := pearlCSampledDevRev1KShapes CM d sem ρ 0 nks ω g hω0
    (fun nk hnk m => by rw [wrefDevRev1K_zero]; exact hw0 nk hnk m)
    (fun nk hnk m => by rw [wrefDevRev1K_zero]; exact hω nk hnk m) hg hTT
  rw [pearlCProtocolDevRev1K_zero, pearlCTilesDevRev1K_zero] at h
  exact h

end

/-! ## The H100 record at Llama 3.2 1B's linears -/

theorem wrefDevRev1_h100_nonneg (s : Shape) : 0 ≤ wrefDevRev1 devH100 s :=
  le_trans (creditDevRev1_h100_nonneg s) (creditDevRev1_le_wref_h100 s)

theorem creditDevRev1_h100_n3072k2048 (m : ℕ) : creditDevRev1 devH100 ⟨m, 3072, 2048⟩ = 8355840 * m := by
  norm_num [creditDevRev1, firstAddDev, creditDev, devH100, Prices.h100, Params.pi, Costs.adopted]
  ring

theorem wrefDevRev1_h100_n3072k2048 (m : ℕ) : wrefDevRev1 devH100 ⟨m, 3072, 2048⟩ = 8364032 * m := by
  rw [wrefDevRev1, creditDevRev1_h100_n3072k2048]
  norm_num [devH100, Prices.h100, Costs.adopted]
  ring

theorem creditDevRev1_h100_n2048k2048 (m : ℕ) : creditDevRev1 devH100 ⟨m, 2048, 2048⟩ = 5636096 * m := by
  norm_num [creditDevRev1, firstAddDev, creditDev, devH100, Prices.h100, Params.pi, Costs.adopted]
  ring

theorem wrefDevRev1_h100_n2048k2048 (m : ℕ) : wrefDevRev1 devH100 ⟨m, 2048, 2048⟩ = 5644288 * m := by
  rw [wrefDevRev1, creditDevRev1_h100_n2048k2048]
  norm_num [devH100, Prices.h100, Costs.adopted]
  ring

theorem creditDevRev1_h100_n16384k2048 (m : ℕ) : creditDevRev1 devH100 ⟨m, 16384, 2048⟩ = 43712512 * m := by
  norm_num [creditDevRev1, firstAddDev, creditDev, devH100, Prices.h100, Params.pi, Costs.adopted]
  ring

theorem wrefDevRev1_h100_n16384k2048 (m : ℕ) : wrefDevRev1 devH100 ⟨m, 16384, 2048⟩ = 43720704 * m := by
  rw [wrefDevRev1, creditDevRev1_h100_n16384k2048]
  norm_num [devH100, Prices.h100, Costs.adopted]
  ring

theorem creditDevRev1_h100_n2048k8192 (m : ℕ) : creditDevRev1 devH100 ⟨m, 2048, 8192⟩ = 21954560 * m := by
  norm_num [creditDevRev1, firstAddDev, creditDev, devH100, Prices.h100, Params.pi, Costs.adopted]
  ring

theorem wrefDevRev1_h100_n2048k8192 (m : ℕ) : wrefDevRev1 devH100 ⟨m, 2048, 8192⟩ = 21987328 * m := by
  rw [wrefDevRev1, creditDevRev1_h100_n2048k8192]
  norm_num [devH100, Prices.h100, Costs.adopted]
  ring

/-- Llama 3.2 1B's largest ratio `wrefDevRev1/creditDevRev1` on the H100 record is its down projection's, `671/670`:
`ω = (671/670)·(400/399) = 26840/26733` bounds every linear's `W_ref` against `(1 − 1/400)` of its credit. -/
theorem wrefDevRev1_h100_llama32_1b_le :
    ∀ nk ∈ llama32_1bLinears, ∀ m : ℕ,
      wrefDevRev1 devH100 ⟨m, nk.1, nk.2⟩ ≤
        26840 / 26733 * ((1 - 1 / 400) * creditDevRev1 devH100 ⟨m, nk.1, nk.2⟩) := by
  intro nk hnk m
  have hm : (0 : ℚ) ≤ m := Nat.cast_nonneg m
  simp only [llama32_1bLinears, mem_insert, mem_singleton] at hnk
  rcases hnk with rfl | rfl | rfl | rfl
  · rw [wrefDevRev1_h100_n3072k2048, creditDevRev1_h100_n3072k2048]; linarith
  · rw [wrefDevRev1_h100_n2048k2048, creditDevRev1_h100_n2048k2048]; linarith
  · rw [wrefDevRev1_h100_n16384k2048, creditDevRev1_h100_n16384k2048]; linarith
  · rw [wrefDevRev1_h100_n2048k8192, creditDevRev1_h100_n2048k8192]; linarith

section Served

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
  (sem : PearlCSem Q R S)

/-- **The H100 record under rev1, `m × 3072 × 2048` at any `m` (Llama 3.2 1B's qkv)**, worst case at the inert-work
share `1/400`: `g = 48749/8168000` (0.59683%). -/
theorem pearlCGapH100Rev1_n3072k2048 (hTT : TTOutPearlCH100Rev1 CM sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCDomainDevNK devH100 3072 2048)
      ((48749 / 8168000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1NK CM devH100 sem _ _ _ (20420 / 20349) _ (by norm_num)
    (fun m => by rw [wrefDevRev1_h100_n3072k2048, creditDevRev1_h100_n3072k2048]; ring) (by norm_num) hTT

/-- **The H100 record under rev1 per audit tile, `m × 3072 × 2048` at any `m` (Llama 3.2 1B's qkv)**:
`g = 48749/8168000` (0.59683%). -/
theorem pearlCSampledH100Rev1_n3072k2048 (hTT : TTOutTilePearlCH100Rev1 CM sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCTilesDevRev1 devH100 sem (1 / 400))
      (pearlCDomainDevNK devH100 3072 2048) ((48749 / 8168000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1NK CM devH100 sem _ _ _ (20420 / 20349) _ (by norm_num)
    (fun _ => wrefDevRev1_h100_nonneg _)
    (fun m => by rw [wrefDevRev1_h100_n3072k2048, creditDevRev1_h100_n3072k2048]; ring) (by norm_num) hTT

/-- **The H100 record under rev1, `m × 2048 × 2048` at any `m` (Llama 3.2 1B's o)**, worst case at the inert-work
share `1/400`: `g = 44357/6890000` (0.64379%). -/
theorem pearlCGapH100Rev1_n2048k2048 (hTT : TTOutPearlCH100Rev1 CM sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCDomainDevNK devH100 2048 2048)
      ((44357 / 6890000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1NK CM devH100 sem _ _ _ (17225 / 17157) _ (by norm_num)
    (fun m => by rw [wrefDevRev1_h100_n2048k2048, creditDevRev1_h100_n2048k2048]; ring) (by norm_num) hTT

/-- **The H100 record under rev1 per audit tile, `m × 2048 × 2048` at any `m` (Llama 3.2 1B's o)**:
`g = 44357/6890000` (0.64379%). -/
theorem pearlCSampledH100Rev1_n2048k2048 (hTT : TTOutTilePearlCH100Rev1 CM sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCTilesDevRev1 devH100 sem (1 / 400))
      (pearlCDomainDevNK devH100 2048 2048) ((44357 / 6890000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1NK CM devH100 sem _ _ _ (17225 / 17157) _ (by norm_num)
    (fun _ => wrefDevRev1_h100_nonneg _)
    (fun m => by rw [wrefDevRev1_h100_n2048k2048, creditDevRev1_h100_n2048k2048]; ring) (by norm_num) hTT

/-- **The H100 record under rev1, `m × 16384 × 2048` at any `m` (Llama 3.2 1B's gate_up)**, worst case at the
inert-work share `1/400`: `g = 61437/11860000` (0.51802%). -/
theorem pearlCGapH100Rev1_n16384k2048 (hTT : TTOutPearlCH100Rev1 CM sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCDomainDevNK devH100 16384 2048)
      ((61437 / 11860000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1NK CM devH100 sem _ _ _ (88950 / 88711) _ (by norm_num)
    (fun m => by rw [wrefDevRev1_h100_n16384k2048, creditDevRev1_h100_n16384k2048]; ring) (by norm_num) hTT

/-- **The H100 record under rev1 per audit tile, `m × 16384 × 2048` at any `m` (Llama 3.2 1B's gate_up)**:
`g = 61437/11860000` (0.51802%). -/
theorem pearlCSampledH100Rev1_n16384k2048 (hTT : TTOutTilePearlCH100Rev1 CM sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCTilesDevRev1 devH100 sem (1 / 400))
      (pearlCDomainDevNK devH100 16384 2048) ((61437 / 11860000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1NK CM devH100 sem _ _ _ (88950 / 88711) _ (by norm_num)
    (fun _ => wrefDevRev1_h100_nonneg _)
    (fun m => by rw [wrefDevRev1_h100_n16384k2048, creditDevRev1_h100_n16384k2048]; ring) (by norm_num) hTT

/-- **The H100 record under rev1, `m × 2048 × 8192` at any `m` (Llama 3.2 1B's down)**, worst case at the inert-work
share `1/400`: `g = 69533/10736000` (0.64766%). -/
theorem pearlCGapH100Rev1_n2048k8192 (hTT : TTOutPearlCH100Rev1 CM sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCDomainDevNK devH100 2048 8192)
      ((69533 / 10736000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1NK CM devH100 sem _ _ _ (26840 / 26733) _ (by norm_num)
    (fun m => by rw [wrefDevRev1_h100_n2048k8192, creditDevRev1_h100_n2048k8192]; ring) (by norm_num) hTT

/-- **The H100 record under rev1 per audit tile, `m × 2048 × 8192` at any `m` (Llama 3.2 1B's down)**:
`g = 69533/10736000` (0.64766%). -/
theorem pearlCSampledH100Rev1_n2048k8192 (hTT : TTOutTilePearlCH100Rev1 CM sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCTilesDevRev1 devH100 sem (1 / 400))
      (pearlCDomainDevNK devH100 2048 8192) ((69533 / 10736000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1NK CM devH100 sem _ _ _ (26840 / 26733) _ (by norm_num)
    (fun _ => wrefDevRev1_h100_nonneg _)
    (fun m => by rw [wrefDevRev1_h100_n2048k8192, creditDevRev1_h100_n2048k8192]; ring) (by norm_num) hTT

/-- **The H100 record under rev1 on Llama 3.2 1B's served layout**, its four linears at any `m` in one layout, worst
case at the inert-work share `1/400`: `g = 69533/10736000` (0.64766%), the down projection's, the max of the four. -/
theorem pearlCGapH100Rev1_llama32_1b (hTT : TTOutPearlCH100Rev1 CM sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCDomainDevShapes devH100 llama32_1bLinears)
      ((69533 / 10736000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1Shapes CM devH100 sem _ _ (26840 / 26733) _ (by norm_num) wrefDevRev1_h100_llama32_1b_le
    (by norm_num) hTT

/-- **The H100 record under rev1 per audit tile on Llama 3.2 1B's served layout**, the run of record's point:
`g = 69533/10736000` (0.64766%), the down projection's, the max of the four. -/
theorem pearlCSampledH100Rev1_llama32_1b (hTT : TTOutTilePearlCH100Rev1 CM sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCTilesDevRev1 devH100 sem (1 / 400))
      (pearlCDomainDevShapes devH100 llama32_1bLinears) ((69533 / 10736000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1Shapes CM devH100 sem _ _ (26840 / 26733) _ (by norm_num)
    (fun _ _ _ => wrefDevRev1_h100_nonneg _) wrefDevRev1_h100_llama32_1b_le (by norm_num) hTT

/-- `pearlCGapH100Rev1_llama32_1b` at W1 on `W1Device.h100`, the form of `Pouw.Guarantees.PearlC.GapH100Rev1_8192`:
`g = 69533/10736000` (0.64766%). -/
theorem pearlCGapH100Rev1_llama32_1b_w1 (sha fetch : ℚ) (salt : S → List ℕ)
    (hTT : TTOutPearlCH100Rev1W1 sha fetch salt sem (1 / 400)) :
    GGap (W1.w1Priced (W1.W1Device.h100 sha fetch) salt) (pearlCProtocolDevRev1 devH100 sem (1 / 400))
      (pearlCDomainDevShapes devH100 llama32_1bLinears) ((69533 / 10736000 : ℚ) : ℝ) δPearlC :=
  pearlCGapH100Rev1_llama32_1b (W1.w1Priced (W1.W1Device.h100 sha fetch) salt) sem
    (Pouw.PearlC.TTOutW1.ttOutPearlCH100Rev1_of_w1 hTT)

/-- `pearlCSampledH100Rev1_llama32_1b` at W1 on `W1Device.h100`, the form of
`Pouw.Guarantees.PearlC.SampledH100Rev1_8192`: `g = 69533/10736000` (0.64766%). -/
theorem pearlCSampledH100Rev1_llama32_1b_w1 (sha fetch : ℚ) (salt : S → List ℕ)
    (hTT : TTOutTilePearlCH100Rev1W1 sha fetch salt sem (1 / 400)) :
    GGapSampled (W1.w1Priced (W1.W1Device.h100 sha fetch) salt) (pearlCProtocolDevRev1 devH100 sem (1 / 400))
      (pearlCTilesDevRev1 devH100 sem (1 / 400)) (pearlCDomainDevShapes devH100 llama32_1bLinears)
      ((69533 / 10736000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledH100Rev1_llama32_1b (W1.w1Priced (W1.W1Device.h100 sha fetch) salt) sem
    (Pouw.PearlC.TTOutW1.ttOutTilePearlCH100Rev1_of_w1 hTT)

end Served

/-- **ε_a = g/(1 − g) at the served linears**, in the order qkv, o, gate_up, down: 0.60041%, 0.64796%, 0.52072% and
0.65188%. The last is the served layout's. -/
theorem epsA_h100Rev1_llama32_1b :
    (48749 / 8168000 : ℚ) / (1 - 48749 / 8168000) = 48749 / 8119251 ∧
      (44357 / 6890000 : ℚ) / (1 - 44357 / 6890000) = 44357 / 6845643 ∧
      (61437 / 11860000 : ℚ) / (1 - 61437 / 11860000) = 61437 / 11798563 ∧
      (69533 / 10736000 : ℚ) / (1 - 69533 / 10736000) = 69533 / 10666467 := by
  norm_num

end Pouw.PearlC
