import Proofs.Pouw.Lead.Compose.Inventory

/-!
# The composition, over its pieces as hypotheses

The lead's guarantee bounds a prover strategy's priced W1 work over a call below by the honest work less a gap. This
module composes that bound from four pieces, each a hypothesis here:

* **the matmul floor** (`hmm` in `call_floor`): on each path's share of the coins with at least `m` points, the path's
  program outputs a set `D` of at least `N` target words, pairwise distinct and non-free on that share, each in the
  hash's excluded set `X`, so that no SHA-512 register is one of them;
* **hash-compose under `ShaWrites`** (`hsha`): `ShaWrites` at `R` registers on every share of at least `m` points;
* **the network term** (`hE` in `call_priced`): what the strategy runs outside the counted devices `I` costs at most
  `E`;
* **the period inventory with the hosts counted**: `I` holds every card and host inside the network certifier's
  boundary (`PeriodInventory`, `PeriodInventoryT1`, `hostsCounted_call`).

`call_floor` adds the first two on each large share (`hashCompose`) and bounds the outcomes where they don't reach by
the paths times `m` (`parts_le`). `call_priced` charges the network term, `lead_gap` restates the result as the
honest work less `gap`, and `lead_bound` puts the three together over a strategy's runs on the devices.
-/

namespace Pouw.SecurityProofs.Lead.Compose

open Finset Pouw.Dimension Pouw.Dimension.H100

section Sketch

variable {G : Type}

/-- **The small shares**: if every point of `S` whose path's share has at least `m` points is good, the bad points
number at most `m` a path. -/
theorem parts_le {ι : Type} [DecidableEq ι] (S : Finset G) (path : G → ι) (m : ℕ) (bad : G → Prop)
    [DecidablePred bad] (h : ∀ g ∈ S, m ≤ (S.filter fun y => path y = path g).card → ¬ bad g) :
    (S.filter bad).card ≤ (S.image path).card * m := by
  rw [Finset.card_eq_sum_card_fiberwise (f := path) (t := S.image path) fun g hg =>
    Finset.mem_coe.2 (Finset.mem_image_of_mem path (Finset.mem_filter.1 (Finset.mem_coe.1 hg)).1)]
  refine (Finset.sum_le_card_nsmul _ _ m fun i _ => ?_).trans (smul_eq_mul _ _).le
  by_cases hbig : m ≤ (S.filter fun g => path g = i).card
  · have hempty : ((S.filter bad).filter fun g => path g = i) = ∅ :=
      Finset.filter_eq_empty_iff.2 fun g hg hgi => by
        obtain ⟨hgS, hgb⟩ := Finset.mem_filter.1 hg
        exact h g hgS (by rw [hgi]; exact hbig) hgb
    rw [hempty, Finset.card_empty]
    exact Nat.zero_le _
  · refine (Finset.card_le_card fun g hg => ?_).trans (Nat.le_of_lt (not_le.1 hbig))
    obtain ⟨hgb, hgi⟩ := Finset.mem_filter.1 hg
    exact Finset.mem_filter.2 ⟨(Finset.mem_filter.1 hgb).1, hgi⟩

/-- **The call's floor from the matmul floor and hash-compose**: a strategy that runs the program `P i` on path `i`'s
share of the coins `S`, outputting the digests there, pays `c·(N + R)` at every outcome but at most `m` a path, given
the matmul floor's `N` distinct non-free words on each share of at least `m` points (`hmm`) and `ShaWrites` at `R` on
every such share (`hsha`). -/
theorem call_floor {c : ℕ} {pr : Prices100} (hpr : Floor c pr) {Adm : List (Op100 G) → List (G → ℚ) → Prop}
    {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {F X : Set (G → ℚ)} (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ F) (hFX : F ⊆ X)
    {Dig : Finset (G → ℚ)} {N R m : ℕ} {ι : Type} [DecidableEq ι] (S : Finset G) (path : G → ι)
    (L : ι → List (G → ℚ)) (P : ι → List (Op100 G))
    (hL : ∀ i, ∀ f ∈ L i, f ∈ F) (hP : ∀ i, ∀ o ∈ P i, o.WFw pr F) (hA : ∀ i, Adm (P i) (L i))
    (hODig : ∀ i, OutputsOn fpSem (↑(S.filter fun g => path g = i) : Set G) (P i) (L i) Dig)
    (hmm : ∀ i, m ≤ (S.filter fun g => path g = i).card → ∃ D : Finset (G → ℚ), N ≤ D.card ∧
      (∀ t ∈ D, t ∈ X) ∧ DistinctOn (↑(S.filter fun g => path g = i) : Set G) F D ∧
      OutputsOn fpSem (↑(S.filter fun g => path g = i) : Set G) (P i) (L i) D)
    (hsha : ∀ S' ⊆ S, m ≤ S'.card → ShaWrites pr Adm fpSem (↑S' : Set G) X Dig R) :
    (S.filter fun g => cost100 pr (P (path g)) < c * (N + R)).card ≤ (S.image path).card * m := by
  refine parts_le S path m _ fun g _ hbig => ?_
  obtain ⟨D, hN, hDX, hD, hOD⟩ := hmm (path g) hbig
  have h := hashCompose hpr hconst hFX hDX (hsha _ (Finset.filter_subset _ _) hbig) (hL _) (hP _) (hA _) hD hOD
    (hODig _)
  exact not_lt.2 ((Nat.mul_le_mul (le_refl c) (Nat.add_le_add_right hN R)).trans h)

/-- **The network term**: where the call's program pays `c·(N + R)`, the counted devices `I` pay all of it but what
ran outside them, which the network term `E` bounds. -/
theorem call_priced {Dev : Type} {c : ℕ} {pr : Prices100} {N R E : ℕ} {I : Set Dev} {Pa : Run Dev G}
    (hfloor : c * (N + R) ≤ cost100 pr (ops Pa)) (hE : costOn pr Iᶜ Pa ≤ E) :
    c * (N + R) ≤ costOn pr I Pa + E :=
  calc c * (N + R) ≤ cost100 pr (ops Pa) := hfloor
    _ = costOn pr I Pa + costOn pr Iᶜ Pa := (costOn_split pr I Pa).symm
    _ ≤ costOn pr I Pa + E := Nat.add_le_add_left hE _

/-- The gap: the matmul words honest writes beyond the floor's `N`, the hash registers honest writes beyond `R`, and
the network term. -/
def gap (c N R Nh Rh E : ℕ) : ℕ := c * (Nh - N) + c * (Rh - R) + E

/-- **Honest less the gap**: with honest's priced work `c·(Nh + Rh)`, a strategy whose counted work `W` and network
term `E` reach `c·(N + R)` has `W` at least the honest work less `gap`. -/
theorem lead_gap {c N R Nh Rh E W : ℕ} (hN : N ≤ Nh) (hR : R ≤ Rh) (hW : c * (N + R) ≤ W + E) :
    c * (Nh + Rh) ≤ W + gap c N R Nh Rh E := by
  obtain ⟨a, rfl⟩ := Nat.exists_eq_add_of_le hN
  obtain ⟨b, rfl⟩ := Nat.exists_eq_add_of_le hR
  have e : c * (N + a + (R + b)) = c * (N + R) + c * a + c * b := by ring
  simp only [gap, Nat.add_sub_cancel_left]
  rw [e]
  calc c * (N + R) + c * a + c * b ≤ W + E + c * a + c * b := by
        exact Nat.add_le_add_right (Nat.add_le_add_right hW _) _
    _ = W + (c * a + c * b + E) := by ring

/-- **The lead's bound from its pieces**: a strategy that runs `Pa g` on the devices at the outcome `g`, its
instructions path `path g`'s program, has its work on the counted devices `I` at least the honest work less `gap`
at every outcome of `S` but at most `m` a path, given the matmul floor (`hmm`), `ShaWrites` (`hsha`) and the network
term (`hE`). -/
theorem lead_bound {Dev : Type} {c : ℕ} {pr : Prices100} (hpr : Floor c pr)
    {Adm : List (Op100 G) → List (G → ℚ) → Prop} {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {F X : Set (G → ℚ)}
    (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ F) (hFX : F ⊆ X) {Dig : Finset (G → ℚ)} {N R m Nh Rh E : ℕ} {ι : Type}
    [DecidableEq ι] (S : Finset G) (path : G → ι) (L : ι → List (G → ℚ)) (P : ι → List (Op100 G))
    (hL : ∀ i, ∀ f ∈ L i, f ∈ F) (hP : ∀ i, ∀ o ∈ P i, o.WFw pr F) (hA : ∀ i, Adm (P i) (L i))
    (hODig : ∀ i, OutputsOn fpSem (↑(S.filter fun g => path g = i) : Set G) (P i) (L i) Dig)
    (hmm : ∀ i, m ≤ (S.filter fun g => path g = i).card → ∃ D : Finset (G → ℚ), N ≤ D.card ∧
      (∀ t ∈ D, t ∈ X) ∧ DistinctOn (↑(S.filter fun g => path g = i) : Set G) F D ∧
      OutputsOn fpSem (↑(S.filter fun g => path g = i) : Set G) (P i) (L i) D)
    (hsha : ∀ S' ⊆ S, m ≤ S'.card → ShaWrites pr Adm fpSem (↑S' : Set G) X Dig R)
    (Pa : G → Run Dev G) (hops : ∀ g, ops (Pa g) = P (path g)) {I : Set Dev} (hE : ∀ g, costOn pr Iᶜ (Pa g) ≤ E)
    (hN : N ≤ Nh) (hR : R ≤ Rh) :
    (S.filter fun g => costOn pr I (Pa g) + gap c N R Nh Rh E < c * (Nh + Rh)).card ≤ (S.image path).card * m := by
  refine (Finset.card_le_card fun g hg => ?_).trans (call_floor hpr hconst hFX S path L P hL hP hA hODig hmm hsha)
  obtain ⟨hgS, hlt⟩ := Finset.mem_filter.1 hg
  refine Finset.mem_filter.2 ⟨hgS, not_le.1 fun hle => ?_⟩
  have h1 : c * (N + R) ≤ costOn pr I (Pa g) + E :=
    call_priced (N := N) (R := R) (by rw [hops]; exact hle) (hE g)
  exact absurd (lead_gap hN hR h1) (not_le.2 hlt)

end Sketch

end Pouw.SecurityProofs.Lead.Compose
