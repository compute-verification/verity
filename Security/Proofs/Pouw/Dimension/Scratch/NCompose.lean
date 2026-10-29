import Proofs.Pouw.Dimension.H100FloorProofs
import Definitions.Pouw.DitherInstance

/-!
# `hash-compose` in `MH100w`: SHA-512's written registers beside the distinct-writes targets (scratch; not in a lock)

**Draft, pending review**, on `MH100w`'s draft semantics (`Op100.WFw`, `DistinctWritesMH100Floor`). Nothing here is
in a lock, and nothing imports it.

* **`ShaWrites`** (a named premise, algorithmic knowledge about SHA-512): ShaSpeed restated as a count. Every
  admissible program on inputs from an excluded set `X` that outputs the digest words `Dig` on the grid `S` writes `R`
  registers, pairwise distinct on `S`, none equal on `S` to a member of `X`. `X` holds the free values, the message
  words and whatever stands for a message word (its carriers, its encodings), so none of the `R` registers is a
  distinct-writes target. `shaWrites_floor`: under `Floor c` it gives ShaSpeed's form, `c·R ≤ cost`.
* **The counting theorem** (`hashCompose`, and `hashCompose_by` for any target–register relation `Rel` that reads
  only values on `S`): a well-formed program on free inputs that outputs `DistinctOn` targets `T` (or, for each target,
  a register in `Rel` to it) and the digests costs at least `c·(|T| + R)`. No instruction counts toward both.
* **Why `ShaWrites` is a premise over a class `Adm`, and why its exclusion is needed**
  (`shaWrites_unrestricted_false`, `shaWrites_false_of_free`, `compose_needs_exclusion`): over all of `MH100w`, which
  lets an instruction write any function, it is false for `R > |Dig|`; it is false for `R > 0` once the digests are
  free; and a cost floor for hashing does not add to `DistinctWritesFloor` when the hash's registers may be targets.
* **Carried and encoded words** (`Carries`, `Encodes`, `charged_by`): a register carries a target when it equals the
  target plus a constant on `S` (a `wgmma` with a constant accumulator), and encodes it when each is a function of the
  other on `S` (a byte swap, a rotation, a float conversion). The count holds with either, under the matching
  separation (`SepBy`). `CarriedWords` (a named premise) is that every admissible program writes a carrier of each
  committed word; with it, only carriers leave the hash's count (`hashCompose_of_carriedWords`). The pair gives the
  joint count, `ShaWrites` with only the free values excluded at `|T| + R` (`jointWrites_of_carriedWords`), which
  doesn't say which registers are the words, and the joint count holds where `CarriedWords` fails
  (`fold_joint_not_carried`, the fold on a toy grid). Over every program of `MH100w` `CarriedWords` is false
  (`carriedWords_unrestricted_false`), so, like `ShaWrites`, it is a premise about the card's programs.
  `sepShift_needed`: targets that differ by a constant share one carrier. The byte swap
  (`bswap32`, little-endian rows read by SHA-512's big-endian words): a carrier `w + s` with `s ≠ 0` does not survive
  the swap up to a constant (`bswap32_shift_not_const`), and the swapped word is no carrier of the word
  (`bswap32_not_carrier`) but encodes it (`bswap32_bswap32`).
* **The atomic call, T1's layer** (`AtomicCall`; `atomicCall_closes`): T1's adversary is a program on the card alone.
  `AtomicCall` makes the call one program from the call's free words, so `root(A)` is not free. With `ShaWrites` over
  every digest of the call, `root(A)`'s included, the card's cost is at least `c·(|T| + R_A + R_W)`: the pre-noise
  hashing of `A` is charged to the card, beside the targets. Without it the hash's credit is zero
  (`shaWrites_false_of_free`, under a free set holding `root(A)`).
* **The cards' inventory** (`Run`, `costOn`, `PeriodInventory`; `costOn_split`, `egress_charged`,
  `preNoiseHash_credited`): instructions on devices, cost counted on the cards `I`. The cards are charged the bound
  less what devices outside `I` ran (`egress_charged`); for devices outside the network certifier's boundary, bounding
  that is the egress layer's. `PeriodInventory`, every device of both runs in `I`, makes it zero. Without it,
  outsourced hashing goes uncounted on `I` (`costOn_outside`, `outsourced_uncredited`); a toy instance meets every
  hypothesis and attains the bound (`toy_attained`).
* **The hosts inside T1** (`PeriodInventoryT1`, `HostsWithinRow`, `hostsW1`; `hostsCounted_closes`,
  `hostsCounted_call`, `hostsCounted_capacity`, `hostRow_residual`, `routeH_toy`): inside the network certifier's
  boundary every device is a card or one of its hosts, and the hosts are counted at θ's rows (`HostRow`, as
  `HostsCounted` counts them), so a host's hashing is credited as hashing. The credited W1 is at most the cards' cost,
  the hosts' counted W1 and what came from outside the boundary, and with the other work on the cards it fits the
  period's capacity with the hosts at their rows (`hostsCounted_capacity`). A row below the hosts' best leaves
  `Σ_r Tp·n·w_r·(best_r − rate_r)⁺`, resource by resource. Both premises read the headline's own terms: the hosts'
  counted W1 is within `Period.cap` at the host resources of the `G` that `OnCard` describes (`hostsW1_le_hostsCounted`,
  `hostsW1_le_cap`), and `Period.Fits` at hosts counted at the row gives `HostsWithinRow` (`hostsWithinRow_of_fits`,
  `hostsCounted_closes_of_fits`; `hostsCounted_others`: the row's own hosts are counted).
* **Which grid** (`shaWrites_of_slice`, `shaWrites_grid_not_slice`): `ShaWrites` on the slice where the prover's chosen
  values are fixed and only the window's coins vary gives it on the data grid, not conversely; the data grid credits
  the digest of a value the prover chose, which the slice can't. So `R` is counted on the slice, and a chosen block's
  constant work drops out of it.
* **Lifting a data-case result to the call grid** (`distinctOn_lift`, `sepBy_lift`): a target set distinct at one
  data case, with that case's free values, is distinct on the call grid with the call's free values, when each call
  free value restricts to a data-case free value.
* **The ledger's cost forms** (`ShaSpeedW1`, `HashComposeW1`): ShaSpeed in `MH100w` and `hash-compose` as named
  Props. Both follow from `ShaWrites` (`shaSpeedW1_of_shaWrites`, `hashComposeW1_of_shaWrites`); the second doesn't
  follow from the first and distinct targets (`hashComposeW1_not_of_shaSpeedW1`), so the ledger needs `ShaWrites`,
  not ShaSpeed plus `hash-compose`.
-/

namespace Pouw.Dimension.NCompose

open Finset Pouw.Dimension Pouw.Dimension.H100

/-! ## 1. Generic counting -/

section Count

variable {G : Type}

theorem wfw_mono {pr : Prices100} {F F' : Set (G → ℚ)} (h : F ⊆ F') (o : Op100 G) (ho : o.WFw pr F) :
    o.WFw pr F' := by
  cases o with
  | operand f => exact h ho
  | _ => exact ho

theorem distinctOn_mono {S : Set G} {F F' : Set (G → ℚ)} {T : Finset (G → ℚ)} (h : F ⊆ F')
    (hT : DistinctOn S F' T) : DistinctOn S F T :=
  ⟨hT.1, fun t ht f hf => hT.2 t ht f (h hf)⟩

/-- Two target sets that are each distinct and non-free, and pairwise different across, are so together (the
sibling scratch file `SwLean.lean` proves the same lemma). -/
theorem distinctOn_union [DecidableEq (G → ℚ)] {S : Set G} {F : Set (G → ℚ)} {T₁ T₂ : Finset (G → ℚ)}
    (h₁ : DistinctOn S F T₁) (h₂ : DistinctOn S F T₂) (hx : ∀ t ∈ T₁, ∀ t' ∈ T₂, ¬ ∀ g ∈ S, t g = t' g) :
    DistinctOn S F (T₁ ∪ T₂) := by
  refine ⟨fun t ht t' ht' h => ?_, fun t ht => ?_⟩
  · rcases Finset.mem_union.1 ht with ht | ht <;> rcases Finset.mem_union.1 ht' with ht' | ht'
    · exact h₁.1 t ht t' ht' h
    · exact absurd h (hx t ht t' ht')
    · exact absurd (fun g hg => (h g hg).symm) (hx t' ht' t ht)
    · exact h₂.1 t ht t' ht' h
  · rcases Finset.mem_union.1 ht with ht | ht
    · exact h₁.2 t ht
    · exact h₂.2 t ht

theorem outputsOn_union [DecidableEq (G → ℚ)] {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {S : Set G} {P : List (Op100 G)}
    {L : List (G → ℚ)} {T₁ T₂ : Finset (G → ℚ)} (h₁ : OutputsOn fpSem S P L T₁) (h₂ : OutputsOn fpSem S P L T₂) :
    OutputsOn fpSem S P L (T₁ ∪ T₂) := by
  intro t ht
  rcases Finset.mem_union.1 ht with ht | ht
  · exact h₁ t ht
  · exact h₂ t ht

/-- A relation between a target and a register that reads the register only through its values on `S`. -/
def OnS (S : Set G) (Rel : (G → ℚ) → (G → ℚ) → Prop) : Prop :=
  ∀ t r r', (∀ g ∈ S, r g = r' g) → Rel t r → Rel t r'

/-- **Targets separated by `Rel`**: no register stands in `Rel` to two targets, and no free value to a target. -/
def SepBy (Rel : (G → ℚ) → (G → ℚ) → Prop) (F : Set (G → ℚ)) (T : Finset (G → ℚ)) : Prop :=
  (∀ t ∈ T, ∀ t' ∈ T, ∀ r, Rel t r → Rel t' r → t = t') ∧ ∀ t ∈ T, ∀ f ∈ F, ¬ Rel t f

/-- The program writes, for every target, a register in `Rel` to it. -/
def WritesBy (Rel : (G → ℚ) → (G → ℚ) → Prop) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (P : List (Op100 G))
    (L : List (G → ℚ)) (T : Finset (G → ℚ)) : Prop :=
  ∀ t ∈ T, ∃ r ∈ regs100 fpSem P L, Rel t r

/-- The registers that stand in `Rel` to some target of `T`. -/
def RelOf (Rel : (G → ℚ) → (G → ℚ) → Prop) (T : Finset (G → ℚ)) : Set (G → ℚ) := {r | ∃ t ∈ T, Rel t r}

/-- **A register per target, chosen**: registers in `Rel` to separated targets are distinct and non-free on `S`, one
per target. -/
theorem distinctOn_image [DecidableEq (G → ℚ)] {S : Set G} {Rel : (G → ℚ) → (G → ℚ) → Prop} (hS : OnS S Rel)
    {F : Set (G → ℚ)} {T : Finset (G → ℚ)} (hsep : SepBy Rel F T) (ρ : (G → ℚ) → G → ℚ)
    (hρ : ∀ t ∈ T, Rel t (ρ t)) : DistinctOn S F (T.image ρ) ∧ (T.image ρ).card = T.card := by
  refine ⟨⟨fun r hr r' hr' h => ?_, fun r hr f hf h => ?_⟩, ?_⟩
  · obtain ⟨t, ht, rfl⟩ := Finset.mem_image.1 hr
    obtain ⟨t', ht', rfl⟩ := Finset.mem_image.1 hr'
    have h' : Rel t' (ρ t) := hS t' (ρ t') (ρ t) (fun g hg => (h g hg).symm) (hρ t' ht')
    rw [hsep.1 t ht t' ht' (ρ t) (hρ t ht) h']
  · obtain ⟨t, ht, rfl⟩ := Finset.mem_image.1 hr
    exact hsep.2 t ht f hf (hS t (ρ t) f h (hρ t ht))
  · refine Finset.card_image_of_injOn fun t ht t' ht' h => ?_
    have h' : Rel t' (ρ t) := by
      have := hρ t' ht'
      rwa [← h] at this
    exact hsep.1 t ht t' ht' (ρ t) (hρ t ht) h'

/-- **The count with any relation** (rank 0 in `MH100w`): under `Floor c`, a well-formed program on free inputs that
writes, for each separated target, a register in `Rel` to it costs at least `c` a target. -/
theorem charged_by {c : ℕ} {pr : Prices100} (hpr : Floor c pr) {F : Set (G → ℚ)}
    (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ F) {S : Set G} {Rel : (G → ℚ) → (G → ℚ) → Prop} (hS : OnS S Rel)
    {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {L : List (G → ℚ)} {P : List (Op100 G)} {T : Finset (G → ℚ)}
    (hL : ∀ f ∈ L, f ∈ F) (hP : ∀ o ∈ P, o.WFw pr F) (hsep : SepBy Rel F T) (hw : WritesBy Rel fpSem P L T) :
    c * T.card ≤ cost100 pr P := by
  classical
  choose! ρ hρR hρ using hw
  obtain ⟨hD, hcard⟩ := distinctOn_image hS hsep ρ hρ
  have hO : OutputsOn fpSem S P L (T.image ρ) := by
    intro r hr
    obtain ⟨t, ht, rfl⟩ := Finset.mem_image.1 hr
    exact ⟨ρ t, hρR t ht, fun _ _ => rfl⟩
  have := distinctWritesMH100Floor c pr hpr G F hconst S fpSem L P (T.image ρ) hL hP hD hO
  rwa [hcard] at this

end Count

/-! ## 2. `ShaWrites` and the counting theorem -/

section Compose

variable {G : Type}

/-- **`ShaWrites pr Adm fpSem S X Dig R`** (named premise; rank 4a, algorithmic knowledge about SHA-512): ShaSpeed
restated as a count of written registers. Every admissible program (`Adm`) on inputs from `X`, well-formed over `X`,
that matches on `S` every digest word of `Dig`, writes at least `R` registers, pairwise distinct on `S` and none equal
on `S` to a member of `X`.

`X` holds the free values and the hash's message words, with whatever stands for one (its carriers or encodings), so no
counted register is a distinct-writes target. The digests are jointly those of every hash the period computes, so
nothing is shared across instances. Over all of `MH100w` it is false (`shaWrites_unrestricted_false`): `Adm` is the
class of programs the hardware runs. -/
def ShaWrites (pr : Prices100) (Adm : List (Op100 G) → List (G → ℚ) → Prop) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ)
    (S : Set G) (X : Set (G → ℚ)) (Dig : Finset (G → ℚ)) (R : ℕ) : Prop :=
  ∀ (L : List (G → ℚ)) (P : List (Op100 G)), (∀ f ∈ L, f ∈ X) → (∀ o ∈ P, o.WFw pr X) → Adm P L →
    OutputsOn fpSem S P L Dig →
    ∃ H : Finset (G → ℚ), R ≤ H.card ∧ DistinctOn S X H ∧ OutputsOn fpSem S P L H

/-- **`ShaWrites` gives ShaSpeed's form** in `MH100w`: under `Floor c`, every admissible program that outputs the
digests costs at least `c·R`. -/
theorem shaWrites_floor {c : ℕ} {pr : Prices100} (hpr : Floor c pr) {Adm : List (Op100 G) → List (G → ℚ) → Prop}
    {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {S : Set G} {X : Set (G → ℚ)} (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ X)
    {Dig : Finset (G → ℚ)} {R : ℕ} (hsha : ShaWrites pr Adm fpSem S X Dig R) {L : List (G → ℚ)}
    {P : List (Op100 G)} (hL : ∀ f ∈ L, f ∈ X) (hP : ∀ o ∈ P, o.WFw pr X) (hA : Adm P L)
    (hOD : OutputsOn fpSem S P L Dig) : c * R ≤ cost100 pr P := by
  obtain ⟨H, hR, hD, hO⟩ := hsha L P hL hP hA hOD
  exact le_trans (Nat.mul_le_mul_left c hR) (distinctWritesMH100Floor c pr hpr G X hconst S fpSem L P H hL hP hD hO)

/-- **The composed set**: under the hypotheses of `hashCompose_by` the program writes at least `|T| + R` registers,
pairwise distinct on `S`, none free: one related register a target and the hash's `R`. -/
theorem composed_set {Adm : List (Op100 G) → List (G → ℚ) → Prop} {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {pr : Prices100}
    {S : Set G} {F X : Set (G → ℚ)} (hFX : F ⊆ X) {Rel : (G → ℚ) → (G → ℚ) → Prop} (hS : OnS S Rel)
    {T Dig : Finset (G → ℚ)} {R : ℕ} (hRX : RelOf Rel T ⊆ X) (hsha : ShaWrites pr Adm fpSem S X Dig R)
    {L : List (G → ℚ)} {P : List (Op100 G)} (hL : ∀ f ∈ L, f ∈ F) (hP : ∀ o ∈ P, o.WFw pr F) (hA : Adm P L)
    (hsep : SepBy Rel F T) (hw : WritesBy Rel fpSem P L T) (hOD : OutputsOn fpSem S P L Dig) :
    ∃ H' : Finset (G → ℚ), T.card + R ≤ H'.card ∧ DistinctOn S F H' ∧ OutputsOn fpSem S P L H' := by
  classical
  obtain ⟨H, hR, hDX, hOH⟩ :=
    hsha L P (fun f hf => hFX (hL f hf)) (fun o ho => wfw_mono hFX o (hP o ho)) hA hOD
  choose! ρ hρR hρ using hw
  obtain ⟨hDC, hcard⟩ := distinctOn_image hS hsep ρ hρ
  have hDH : DistinctOn S F H := distinctOn_mono hFX hDX
  have hx : ∀ r ∈ T.image ρ, ∀ h ∈ H, ¬ ∀ g ∈ S, r g = h g := by
    intro r hr h hh heq
    obtain ⟨t, ht, rfl⟩ := Finset.mem_image.1 hr
    have hhX : h ∈ X := hRX ⟨t, ht, hS t (ρ t) h heq (hρ t ht)⟩
    exact hDX.2 h hh h hhX fun _ _ => rfl
  have hO : OutputsOn fpSem S P L (T.image ρ) := by
    intro r hr
    obtain ⟨t, ht, rfl⟩ := Finset.mem_image.1 hr
    exact ⟨ρ t, hρR t ht, fun _ _ => rfl⟩
  have hdisj : Disjoint (T.image ρ) H :=
    Finset.disjoint_left.2 fun r h₁ h₂ => hx r h₁ r h₂ fun _ _ => rfl
  refine ⟨T.image ρ ∪ H, ?_, distinctOn_union hDC hDH hx, outputsOn_union hO hOH⟩
  rw [Finset.card_union_of_disjoint hdisj, hcard]
  exact Nat.add_le_add_left hR _

/-- **hash-compose with any relation** (rank 0 given `ShaWrites`): under `Floor c`, an admissible well-formed
program on free inputs that writes a register in `Rel` to each separated target, and outputs the digests, costs at
least `c·(|T| + R)`, when every register in `Rel` to a target is excluded from the hash's count (`hRX`). -/
theorem hashCompose_by {c : ℕ} {pr : Prices100} (hpr : Floor c pr) {Adm : List (Op100 G) → List (G → ℚ) → Prop}
    {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {S : Set G} {F X : Set (G → ℚ)} (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ F)
    (hFX : F ⊆ X) {Rel : (G → ℚ) → (G → ℚ) → Prop} (hS : OnS S Rel) {T Dig : Finset (G → ℚ)} {R : ℕ}
    (hRX : RelOf Rel T ⊆ X) (hsha : ShaWrites pr Adm fpSem S X Dig R) {L : List (G → ℚ)} {P : List (Op100 G)}
    (hL : ∀ f ∈ L, f ∈ F) (hP : ∀ o ∈ P, o.WFw pr F) (hA : Adm P L) (hsep : SepBy Rel F T)
    (hw : WritesBy Rel fpSem P L T) (hOD : OutputsOn fpSem S P L Dig) :
    c * (T.card + R) ≤ cost100 pr P := by
  obtain ⟨H', hcard, hD, hO⟩ := composed_set hFX hS hRX hsha hL hP hA hsep hw hOD
  exact le_trans (Nat.mul_le_mul_left c hcard)
    (distinctWritesMH100Floor c pr hpr G F hconst S fpSem L P H' hL hP hD hO)

/-- Equality on `S`: the register is the target there. -/
def EqOn' (S : Set G) (t r : G → ℚ) : Prop := ∀ g ∈ S, r g = t g

theorem onS_eqOn (S : Set G) : OnS S (EqOn' S) :=
  fun _ _ _ h h' g hg => by rw [← h g hg, h' g hg]

theorem sepBy_eqOn {S : Set G} {F : Set (G → ℚ)} {T : Finset (G → ℚ)} (h : DistinctOn S F T) :
    SepBy (EqOn' S) F T :=
  ⟨fun t ht t' ht' r h₁ h₂ => h.1 t ht t' ht' fun g hg => by rw [← h₁ g hg, h₂ g hg],
    fun t ht f hf h' => h.2 t ht f hf fun g hg => (h' g hg).symm⟩

/-- **hash-compose** (rank 0 given `ShaWrites`): a program that outputs `DistinctOn` targets, each a message word (in
`X`), and the digests, costs at least `c·(|T| + R)`. -/
theorem hashCompose {c : ℕ} {pr : Prices100} (hpr : Floor c pr) {Adm : List (Op100 G) → List (G → ℚ) → Prop}
    {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {S : Set G} {F X : Set (G → ℚ)} (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ F)
    (hFX : F ⊆ X) {T Dig : Finset (G → ℚ)} {R : ℕ} (hTX : ∀ t ∈ T, t ∈ X) (hsha : ShaWrites pr Adm fpSem S X Dig R)
    {L : List (G → ℚ)} {P : List (Op100 G)} (hL : ∀ f ∈ L, f ∈ F) (hP : ∀ o ∈ P, o.WFw pr F) (hA : Adm P L)
    (hT : DistinctOn S F T) (hOT : OutputsOn fpSem S P L T) (hOD : OutputsOn fpSem S P L Dig) :
    c * (T.card + R) ≤ cost100 pr P := by
  classical
  obtain ⟨H, hR, hDX, hOH⟩ :=
    hsha L P (fun f hf => hFX (hL f hf)) (fun o ho => wfw_mono hFX o (hP o ho)) hA hOD
  have hx : ∀ t ∈ T, ∀ h ∈ H, ¬ ∀ g ∈ S, t g = h g :=
    fun t ht h hh heq => hDX.2 h hh t (hTX t ht) fun g hg => (heq g hg).symm
  have hb := distinctWritesMH100Floor c pr hpr G F hconst S fpSem L P (T ∪ H) hL hP
    (distinctOn_union hT (distinctOn_mono hFX hDX) hx) (outputsOn_union hOT hOH)
  have hdisj : Disjoint T H := Finset.disjoint_left.2 fun t h₁ h₂ => hx t h₁ t h₂ fun _ _ => rfl
  rw [Finset.card_union_of_disjoint hdisj] at hb
  exact le_trans (Nat.mul_le_mul_left c (Nat.add_le_add_left hR _)) hb

/-- **At the H200's floor table**: 32 a target and 32 a counted hash register. -/
theorem hashCompose_floor {Adm : List (Op100 G) → List (G → ℚ) → Prop} {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ}
    {S : Set G} {F X : Set (G → ℚ)} (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ F) (hFX : F ⊆ X)
    {T Dig : Finset (G → ℚ)} {R : ℕ} (hTX : ∀ t ∈ T, t ∈ X) (hsha : ShaWrites h100Floor Adm fpSem S X Dig R)
    {L : List (G → ℚ)} {P : List (Op100 G)} (hL : ∀ f ∈ L, f ∈ F) (hP : ∀ o ∈ P, o.WFw h100Floor F)
    (hA : Adm P L) (hT : DistinctOn S F T) (hOT : OutputsOn fpSem S P L T) (hOD : OutputsOn fpSem S P L Dig) :
    32 * (T.card + R) ≤ cost100 h100Floor P :=
  hashCompose h32Floor hconst hFX hTX hsha hL hP hA hT hOT hOD

end Compose

/-! ## 3. Why `ShaWrites` needs a class and an exclusion -/

section Refute

variable {G : Type}

theorem regs100_map_int (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) :
    ∀ (l Rs : List (G → ℚ)), regs100 fpSem (l.map Op100.int) Rs = Rs ++ l
  | [], Rs => by simp [regs100]
  | a :: l, Rs => by
    have ih := regs100_map_int fpSem l (Rs ++ [a])
    simp only [regs100, List.map_cons, List.foldl_cons, Op100.outs] at ih ⊢
    rw [ih]
    simp

theorem regs100_map_operand (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) :
    ∀ (l Rs : List (G → ℚ)), regs100 fpSem (l.map Op100.operand) Rs = Rs ++ l
  | [], Rs => by simp [regs100]
  | a :: l, Rs => by
    have ih := regs100_map_operand fpSem l (Rs ++ [a])
    simp only [regs100, List.map_cons, List.foldl_cons, Op100.outs] at ih ⊢
    rw [ih]
    simp

theorem cost100_map_operand (pr : Prices100) (l : List (G → ℚ)) : cost100 pr (l.map Op100.operand) = 0 := by
  induction l with
  | nil => rfl
  | cons a l ih => simpa [cost100, Op100.price] using ih

/-- Registers matching on `S`, one per counted register, among a list: at most the list's length. -/
theorem card_le_of_match {S : Set G} {X : Set (G → ℚ)} {H Dig : Finset (G → ℚ)} (hD : DistinctOn S X H)
    (hm : ∀ h ∈ H, ∃ d ∈ Dig, ∀ g ∈ S, d g = h g) : H.card ≤ Dig.card := by
  classical
  choose! σ hσ hσS using hm
  refine Finset.card_le_card_of_injOn σ (fun h hh => hσ h hh) fun h hh h' hh' heq => ?_
  refine hD.1 h hh h' hh' fun g hg => ?_
  rw [← hσS h hh g hg, ← hσS h' hh' g hg, heq]

/-- **Over all of `MH100w`, `ShaWrites` is false** for `R > |Dig|`: one `int` instruction a digest word writes the
digests and nothing else. So `hash-compose` is no theorem of `MH100w`; it needs the class of programs the hardware
runs. -/
theorem shaWrites_unrestricted_false (pr : Prices100) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (S : Set G)
    (X : Set (G → ℚ)) (Dig : Finset (G → ℚ)) (R : ℕ) (hR : Dig.card < R) :
    ¬ ShaWrites pr (fun _ _ => True) fpSem S X Dig R := by
  intro h
  have hregs : regs100 fpSem (Dig.toList.map Op100.int) [] = Dig.toList := by
    simpa using regs100_map_int fpSem Dig.toList []
  have hwf : ∀ o ∈ Dig.toList.map Op100.int, o.WFw pr X := by
    intro o ho
    obtain ⟨d, _, rfl⟩ := List.mem_map.1 ho
    trivial
  obtain ⟨H, hRH, hD, hO⟩ := h [] (Dig.toList.map Op100.int) (by simp) hwf trivial
    (fun t ht => ⟨t, by rw [hregs]; exact Finset.mem_toList.2 ht, fun _ _ => rfl⟩)
  have : H.card ≤ Dig.card := card_le_of_match hD fun h hh => by
    obtain ⟨r, hr, hrS⟩ := hO h hh
    rw [hregs] at hr
    exact ⟨r, Finset.mem_toList.1 hr, hrS⟩
  omega

/-- **Free digests get no credit**: if the digest words are free (in `X`), as `root(A)` is in the free set that
holds every salt-independent function, then operand words output them at cost 0, and `ShaWrites` fails for `R > 0`
whenever that program is admissible. -/
theorem shaWrites_false_of_free (pr : Prices100) {Adm : List (Op100 G) → List (G → ℚ) → Prop}
    (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (S : Set G) (X : Set (G → ℚ)) (Dig : Finset (G → ℚ)) (R : ℕ)
    (hX : ∀ d ∈ Dig, d ∈ X) (hA : Adm (Dig.toList.map Op100.operand) []) (hR : 0 < R) :
    ¬ ShaWrites pr Adm fpSem S X Dig R := by
  intro h
  have hregs : regs100 fpSem (Dig.toList.map Op100.operand) [] = Dig.toList := by
    simpa using regs100_map_operand fpSem Dig.toList []
  have hwf : ∀ o ∈ Dig.toList.map Op100.operand, o.WFw pr X := by
    intro o ho
    obtain ⟨d, hd, rfl⟩ := List.mem_map.1 ho
    exact hX d (Finset.mem_toList.1 hd)
  obtain ⟨H, hRH, hD, hO⟩ := h [] _ (by simp) hwf hA
    (fun t ht => ⟨t, by rw [hregs]; exact Finset.mem_toList.2 ht, fun _ _ => rfl⟩)
  have hH : H = ∅ := by
    refine Finset.eq_empty_of_forall_notMem fun h hh => ?_
    obtain ⟨r, hr, hrS⟩ := hO h hh
    rw [hregs] at hr
    exact hD.2 h hh r (hX r (Finset.mem_toList.1 hr)) fun g hg => (hrS g hg).symm
  rw [hH] at hRH
  simp at hRH
  omega

/-- The toy grid: two points. -/
def tw : Bool → ℚ := fun b => if b then 1 else 0

/-- Constants on a grid. -/
def Consts (G : Type) : Set (G → ℚ) := {f | ∃ q : ℚ, f = fun _ => q}

theorem consts_mem (q : ℚ) : (fun _ : G => q) ∈ Consts G := ⟨q, rfl⟩

theorem tw_not_const (f : Bool → ℚ) (hf : f ∈ Consts Bool) : ¬ ∀ g ∈ (Set.univ : Set Bool), tw g = f g := by
  obtain ⟨q, rfl⟩ := hf
  intro h
  have h1 := h true trivial
  have h0 := h false trivial
  simp [tw] at h1 h0
  rw [← h1] at h0
  norm_num at h0

/-- **A cost floor for hashing does not add to `DistinctWritesFloor` without the exclusion**: when the hash's
counted registers may be targets (here `X` is only the constants and the digest is the target), a program pays
`32·|T|`, below `32·(|T| + R)`. -/
theorem compose_needs_exclusion :
    ∃ (Adm : List (Op100 Bool) → List (Bool → ℚ) → Prop) (P : List (Op100 Bool)),
      ShaWrites h100Floor Adm (fun _ _ _ _ => 0) Set.univ (Consts Bool) {tw} 1 ∧ Adm P [] ∧
      (∀ o ∈ P, o.WFw h100Floor (Consts Bool)) ∧ DistinctOn Set.univ (Consts Bool) {tw} ∧
      OutputsOn (fun _ _ _ _ => 0) Set.univ P [] {tw} ∧ cost100 h100Floor P < 32 * (({tw} : Finset _).card + 1) := by
  refine ⟨fun P L => tw ∈ regs100 (fun _ _ _ _ => 0) P L, [Op100.int tw], ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro L P _ _ hA _
    refine ⟨{tw}, by simp, ⟨fun t ht t' ht' _ => by simp at ht ht'; rw [ht, ht'], fun t ht f hf => ?_⟩,
      fun t ht => ⟨tw, hA, fun g _ => by simp at ht; rw [ht]⟩⟩
    simp at ht
    rw [ht]
    exact tw_not_const f hf
  · simp [regs100, Op100.outs]
  · intro o ho
    simp at ho
    rw [ho]
    trivial
  · exact ⟨fun t ht t' ht' _ => by simp at ht ht'; rw [ht, ht'], fun t ht f hf => by
      simp at ht; rw [ht]; exact tw_not_const f hf⟩
  · intro t ht
    simp at ht
    exact ⟨tw, by simp [regs100, Op100.outs], fun g _ => by rw [ht]⟩
  · decide

end Refute

/-! ## 4. Carried and encoded words -/

section Carry

variable {G : Type}

/-- **`r` carries `t` on `S`**: they differ by a constant there, as a `wgmma` with a constant accumulator writes it. -/
def Carries (S : Set G) (t r : G → ℚ) : Prop := ∃ q : ℚ, ∀ g ∈ S, r g = t g + q

/-- **`r` encodes `t` on `S`**: each is a function of the other there (a carrier, a byte swap, a rotation, a float
conversion). -/
def Encodes (S : Set G) (t r : G → ℚ) : Prop := ∃ φ ψ : ℚ → ℚ, ∀ g ∈ S, t g = φ (r g) ∧ r g = ψ (t g)

theorem onS_carries (S : Set G) : OnS S (Carries S) := by
  rintro t r r' h ⟨q, hq⟩
  exact ⟨q, fun g hg => by rw [← h g hg, hq g hg]⟩

theorem onS_encodes (S : Set G) : OnS S (Encodes S) := by
  rintro t r r' h ⟨φ, ψ, hφ⟩
  exact ⟨φ, ψ, fun g hg => by rw [← h g hg]; exact hφ g hg⟩

theorem carries_of_eqOn {S : Set G} {t r : G → ℚ} (h : EqOn' S t r) : Carries S t r :=
  ⟨0, fun g hg => by rw [h g hg, add_zero]⟩

theorem encodes_of_carries {S : Set G} {t r : G → ℚ} (h : Carries S t r) : Encodes S t r := by
  obtain ⟨q, hq⟩ := h
  exact ⟨fun x => x - q, fun x => x + q, fun g hg => ⟨by rw [hq g hg]; ring, hq g hg⟩⟩

theorem sepBy_mono {R₁ R₂ : (G → ℚ) → (G → ℚ) → Prop} (h : ∀ t r, R₁ t r → R₂ t r) {F : Set (G → ℚ)}
    {T : Finset (G → ℚ)} (hs : SepBy R₂ F T) : SepBy R₁ F T :=
  ⟨fun t ht t' ht' r h₁ h₂ => hs.1 t ht t' ht' r (h _ _ h₁) (h _ _ h₂), fun t ht f hf h₁ => hs.2 t ht f hf (h _ _ h₁)⟩

theorem writesBy_mono {R₁ R₂ : (G → ℚ) → (G → ℚ) → Prop} (h : ∀ t r, R₁ t r → R₂ t r)
    {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {P : List (Op100 G)} {L : List (G → ℚ)} {T : Finset (G → ℚ)}
    (hw : WritesBy R₁ fpSem P L T) : WritesBy R₂ fpSem P L T := fun t ht => by
  obtain ⟨r, hr, h₁⟩ := hw t ht
  exact ⟨r, hr, h _ _ h₁⟩

theorem writesBy_of_outputsOn {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {S : Set G} {P : List (Op100 G)} {L : List (G → ℚ)}
    {T : Finset (G → ℚ)} (h : OutputsOn fpSem S P L T) : WritesBy (EqOn' S) fpSem P L T := h

/-- **Shift-separated targets** are carrier-separated: no two differ by a constant on `S`, and none is a free value
plus a constant. -/
theorem sepBy_carries {S : Set G} {F : Set (G → ℚ)} {T : Finset (G → ℚ)}
    (h₁ : ∀ t ∈ T, ∀ t' ∈ T, (∃ q : ℚ, ∀ g ∈ S, t g = t' g + q) → t = t')
    (h₂ : ∀ t ∈ T, ∀ f ∈ F, ¬ ∃ q : ℚ, ∀ g ∈ S, f g = t g + q) : SepBy (Carries S) F T := by
  refine ⟨fun t ht t' ht' r ⟨q, hq⟩ ⟨q', hq'⟩ => h₁ t ht t' ht' ⟨q' - q, fun g hg => ?_⟩, fun t ht f hf => h₂ t ht f hf⟩
  have := hq g hg
  rw [hq' g hg] at this
  linarith

/-- **Targets that differ by a constant share one carrier**: the separation `SepBy (Carries S)` is needed. On the toy
grid, `tw` and `tw + 1` are distinct and non-constant, and one `int` register carries both, at 32 against
`32·2`. -/
theorem sepShift_needed :
    DistinctOn Set.univ (Consts Bool) {tw, fun b => tw b + 1} ∧
      WritesBy (Carries Set.univ) (fun _ _ _ _ => 0) [Op100.int tw] [] {tw, fun b => tw b + 1} ∧
      cost100 h100Floor [Op100.int tw] < 32 * ({tw, fun b => tw b + 1} : Finset (Bool → ℚ)).card := by
  have hne : tw ≠ fun b => tw b + 1 := by
    intro h
    have := congrFun h true
    simp [tw] at this
  have hcard : ({tw, fun b => tw b + 1} : Finset (Bool → ℚ)).card = 2 := Finset.card_pair hne
  refine ⟨⟨fun t ht t' ht' h => funext fun g => h g trivial, fun t ht f hf => ?_⟩, ?_, ?_⟩
  · simp only [Finset.mem_insert, Finset.mem_singleton] at ht
    rcases ht with rfl | rfl
    · exact tw_not_const f hf
    · obtain ⟨q, rfl⟩ := hf
      intro h
      have h1 := h true trivial
      have h0 := h false trivial
      simp [tw] at h1 h0
      rw [← h0] at h1
      norm_num at h1
  · intro t ht
    refine ⟨tw, by simp [regs100, Op100.outs], ?_⟩
    simp only [Finset.mem_insert, Finset.mem_singleton] at ht
    rcases ht with rfl | rfl
    · exact ⟨0, fun g _ => by ring⟩
    · exact ⟨-1, fun g _ => by ring⟩
  · rw [hcard]
    decide

/-- **hash-compose, carried** (rank 0 given `ShaWrites`): targets written only through carriers, with the carriers
excluded from the hash's count. -/
theorem hashCompose_carried {c : ℕ} {pr : Prices100} (hpr : Floor c pr)
    {Adm : List (Op100 G) → List (G → ℚ) → Prop} {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {S : Set G} {F X : Set (G → ℚ)}
    (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ F) (hFX : F ⊆ X) {T Dig : Finset (G → ℚ)} {R : ℕ}
    (hRX : RelOf (Carries S) T ⊆ X) (hsha : ShaWrites pr Adm fpSem S X Dig R) {L : List (G → ℚ)}
    {P : List (Op100 G)} (hL : ∀ f ∈ L, f ∈ F) (hP : ∀ o ∈ P, o.WFw pr F) (hA : Adm P L)
    (hsep : SepBy (Carries S) F T) (hw : WritesBy (Carries S) fpSem P L T) (hOD : OutputsOn fpSem S P L Dig) :
    c * (T.card + R) ≤ cost100 pr P :=
  hashCompose_by hpr hconst hFX (onS_carries S) hRX hsha hL hP hA hsep hw hOD

/-- **hash-compose, encoded** (rank 0 given `ShaWrites`): the same with encodings. -/
theorem hashCompose_encoded {c : ℕ} {pr : Prices100} (hpr : Floor c pr)
    {Adm : List (Op100 G) → List (G → ℚ) → Prop} {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {S : Set G} {F X : Set (G → ℚ)}
    (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ F) (hFX : F ⊆ X) {T Dig : Finset (G → ℚ)} {R : ℕ}
    (hRX : RelOf (Encodes S) T ⊆ X) (hsha : ShaWrites pr Adm fpSem S X Dig R) {L : List (G → ℚ)}
    {P : List (Op100 G)} (hL : ∀ f ∈ L, f ∈ F) (hP : ∀ o ∈ P, o.WFw pr F) (hA : Adm P L)
    (hsep : SepBy (Encodes S) F T) (hw : WritesBy (Encodes S) fpSem P L T) (hOD : OutputsOn fpSem S P L Dig) :
    c * (T.card + R) ≤ cost100 pr P :=
  hashCompose_by hpr hconst hFX (onS_encodes S) hRX hsha hL hP hA hsep hw hOD

/-- **`CarriedWords Adm fpSem S T Dig`** (named premise; rank 3, the hardware's instruction set): every admissible
program that outputs the digests writes, for each committed word, a register that carries it. `MH100w` alone doesn't
give it (`words_not_forced`). On little-endian rows SHA-512 reads `bswap32 w`, which is no carrier of `w`
(`bswap32_not_carrier`), so a `wgmma` whose accumulator absorbs a round's sum doesn't stand in for it; on big-endian
rows that fold drops the carriers of each later block's `M_0`. -/
def CarriedWords (Adm : List (Op100 G) → List (G → ℚ) → Prop) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (S : Set G)
    (T Dig : Finset (G → ℚ)) : Prop :=
  ∀ (L : List (G → ℚ)) (P : List (Op100 G)), Adm P L → OutputsOn fpSem S P L Dig → WritesBy (Carries S) fpSem P L T

/-- **hash-compose from the premises alone** (rank 0 given `ShaWrites` and `CarriedWords`): an admissible program on
free inputs that outputs the digests costs at least `c·(|T| + R)`, with only the carriers excluded from the hash's
count, so every register of the hash that encodes a word without carrying it counts. -/
theorem hashCompose_of_carriedWords {c : ℕ} {pr : Prices100} (hpr : Floor c pr)
    {Adm : List (Op100 G) → List (G → ℚ) → Prop} {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {S : Set G} {F X : Set (G → ℚ)}
    (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ F) (hFX : F ⊆ X) {T Dig : Finset (G → ℚ)} {R : ℕ}
    (hRX : RelOf (Carries S) T ⊆ X) (hsha : ShaWrites pr Adm fpSem S X Dig R)
    (hcw : CarriedWords Adm fpSem S T Dig) (hsep : SepBy (Carries S) F T) {L : List (G → ℚ)}
    {P : List (Op100 G)} (hL : ∀ f ∈ L, f ∈ F) (hP : ∀ o ∈ P, o.WFw pr F) (hA : Adm P L)
    (hOD : OutputsOn fpSem S P L Dig) : c * (T.card + R) ≤ cost100 pr P :=
  hashCompose_carried hpr hconst hFX hRX hsha hL hP hA hsep (hcw L P hA hOD) hOD

/-- **The joint count follows from the pair** (rank 0 given `ShaWrites` and `CarriedWords`): `ShaWrites` with only
the free values excluded, at `|T| + R` registers. The joint form counts the registers that hashing the words costs, not
which of them are the words, so a program that folds a word into the hash's adds, which `CarriedWords` excludes, may
still meet it; under `Floor c` it gives `c·(|T| + R)` by `shaWrites_floor`. -/
theorem jointWrites_of_carriedWords {pr : Prices100} {Adm : List (Op100 G) → List (G → ℚ) → Prop}
    {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {S : Set G} {F X : Set (G → ℚ)} (hFX : F ⊆ X) {T Dig : Finset (G → ℚ)} {R : ℕ}
    (hRX : RelOf (Carries S) T ⊆ X) (hsha : ShaWrites pr Adm fpSem S X Dig R) (hcw : CarriedWords Adm fpSem S T Dig)
    (hsep : SepBy (Carries S) F T) : ShaWrites pr Adm fpSem S F Dig (T.card + R) := fun L P hL hP hA hOD =>
  composed_set hFX (onS_carries S) hRX hsha hL hP hA hsep (hcw L P hA hOD) hOD

/-- **Committed words need not be materialized in `MH100w`**: one `int` a digest word outputs the digests and no
register that carries, or even encodes, a target, when no digest word encodes one. -/
theorem words_not_forced (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (S : Set G) (T Dig : Finset (G → ℚ))
    (hD : ∀ d ∈ Dig, ∀ t ∈ T, ¬ Encodes S t d) :
    OutputsOn fpSem S (Dig.toList.map Op100.int) [] Dig ∧
      ∀ t ∈ T, ∀ r ∈ regs100 fpSem (Dig.toList.map Op100.int) [], ¬ Encodes S t r := by
  have hregs : regs100 fpSem (Dig.toList.map Op100.int) [] = Dig.toList := by
    simpa using regs100_map_int fpSem Dig.toList []
  refine ⟨fun t ht => ⟨t, by rw [hregs]; exact Finset.mem_toList.2 ht, fun _ _ => rfl⟩, fun t ht r hr => ?_⟩
  rw [hregs] at hr
  exact hD r (Finset.mem_toList.1 hr) t ht

/-- **`CarriedWords` needs its class**, as `ShaWrites` does: over every program of `MH100w` it fails for a nonempty
target set no digest word encodes, since one `int` a digest word outputs the digests and writes no carrier
(`words_not_forced`). So it is a premise about the card's programs (no instruction fuses a byte permute into an add or
a dot product), not a fact of `MH100w`. -/
theorem carriedWords_unrestricted_false (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (S : Set G) (T Dig : Finset (G → ℚ))
    (hT : T.Nonempty) (hD : ∀ d ∈ Dig, ∀ t ∈ T, ¬ Encodes S t d) :
    ¬ CarriedWords (fun _ _ => True) fpSem S T Dig := by
  intro h
  obtain ⟨hO, hno⟩ := words_not_forced fpSem S T Dig hD
  obtain ⟨t, ht⟩ := hT
  obtain ⟨r, hr, hc⟩ := h [] _ trivial hO t ht
  exact hno t ht r hr (encodes_of_carries hc)

end Carry

/-! ## 5. The byte swap: little-endian rows under SHA-512's big-endian words -/

section Bswap

/-- The byte swap of a 32-bit word: SHA-512 reads a little-endian committed word `w` as `bswap32 w`. -/
def bswap32 (x : ℕ) : ℕ :=
  (x % 256) * 16777216 + (x / 256 % 256) * 65536 + (x / 65536 % 256) * 256 + x / 16777216 % 256

theorem bswap32_lt (x : ℕ) : bswap32 x < 4294967296 := by
  unfold bswap32; omega

theorem bswap32_bytes (b0 b1 b2 b3 : ℕ) (h0 : b0 < 256) (h1 : b1 < 256) (h2 : b2 < 256) (h3 : b3 < 256) :
    bswap32 (b0 + 256 * b1 + 65536 * b2 + 16777216 * b3) = b3 + 256 * b2 + 65536 * b1 + 16777216 * b0 := by
  unfold bswap32; omega

/-- **The swap is an involution on 32-bit words**: the swapped word encodes the word. -/
theorem bswap32_bswap32 (x : ℕ) (hx : x < 4294967296) : bswap32 (bswap32 x) = x := by
  obtain ⟨b0, b1, b2, b3, h0, h1, h2, h3, rfl⟩ : ∃ b0 b1 b2 b3 : ℕ, b0 < 256 ∧ b1 < 256 ∧ b2 < 256 ∧ b3 < 256 ∧
      x = b0 + 256 * b1 + 65536 * b2 + 16777216 * b3 :=
    ⟨x % 256, x / 256 % 256, x / 65536 % 256, x / 16777216, by omega, by omega, by omega, by omega, by omega⟩
  rw [bswap32_bytes b0 b1 b2 b3 h0 h1 h2 h3, bswap32_bytes b3 b2 b1 b0 h3 h2 h1 h0]

theorem bswap32_succ_ne (y : ℕ) : bswap32 (y + 1) ≠ bswap32 y + 1 := by
  unfold bswap32; omega

theorem bswap32_eq_zero (x : ℕ) (hx : x < 4294967296) (h : bswap32 x = 0) : x = 0 := by
  unfold bswap32 at h; omega

theorem bswap32_shift (s : ℕ) (h0 : 0 < s) (h1 : s < 4294967296) :
    bswap32 s ≠ (bswap32 (s - 1) + 1) % 4294967296 := by
  obtain ⟨y, rfl⟩ : ∃ y, s = y + 1 := ⟨s - 1, by omega⟩
  simp only [Nat.add_sub_cancel]
  intro h
  have hl := bswap32_lt y
  by_cases hb : bswap32 y + 1 < 4294967296
  · rw [Nat.mod_eq_of_lt hb] at h
    exact bswap32_succ_ne y h
  · have : (bswap32 y + 1) % 4294967296 = 0 := by omega
    rw [this] at h
    have := bswap32_eq_zero (y + 1) h1 h
    omega

/-- **A carrier does not survive the swap up to a constant**: for every shift `s ≢ 0 (mod 2³²)` and every constant
`k`, `bswap32 (w + s) ≡ bswap32 w + k (mod 2³²)` fails at `w = 0` or at `w = 2³² − 1`. So a register `w + s` cannot
stand for `w` past the byte swap with a constant correction. -/
theorem bswap32_shift_not_const (s k : ℕ) (h0 : 0 < s) (h1 : s < 4294967296) :
    ¬ (bswap32 ((0 + s) % 4294967296) % 4294967296 = (bswap32 0 + k) % 4294967296 ∧
      bswap32 ((4294967295 + s) % 4294967296) % 4294967296 = (bswap32 4294967295 + k) % 4294967296) := by
  rintro ⟨ha, hb⟩
  have e0 : bswap32 0 = 0 := by decide
  have e1 : bswap32 4294967295 = 4294967295 := by decide
  have m0 : (0 + s) % 4294967296 = s := by omega
  have m1 : (4294967295 + s) % 4294967296 = s - 1 := by omega
  rw [e0, m0] at ha
  rw [e1, m1] at hb
  have l0 := bswap32_lt s
  have l1 := bswap32_lt (s - 1)
  apply bswap32_shift s h0 h1
  omega

/-- **The swapped word is no carrier of the word**: `bswap32 w − w` differs at `w = 0` and `w = 1`. -/
theorem bswap32_not_carrier : (bswap32 1 : ℤ) - 1 ≠ (bswap32 0 : ℤ) - 0 := by decide

end Bswap

/-! ## 6. The call model: `AtomicCall` and the cards' `PeriodInventory` (T1's) -/

section Call

variable {G Dev : Type}

/-- **`AtomicCall pr Fcall L P`** (named premise; rank 3): the card's call is one program. Its inputs `L` and its
operand words are the call's free words `Fcall` (the constants, the call's input words and its raw coin words), so no
function of the call's input computed before the noise (such as `root(A)`) enters free; the grid it is judged on
ranges over the call's input. -/
def AtomicCall (pr : Prices100) (Fcall : Set (G → ℚ)) (L : List (G → ℚ)) (P : List (Op100 G)) : Prop :=
  (∀ f ∈ L, f ∈ Fcall) ∧ ∀ o ∈ P, o.WFw pr Fcall

/-- **What the atomic call alone closes** (rank 0 given `AtomicCall` and `ShaWrites`; T1's layer, the card alone): on
the call grid `S`, with `ShaWrites` over every digest of the call (`root(A)`'s `DigA` and the committed rows' `DigW`,
`R_A + R_W` registers jointly), the card's program, admissible, writing a register in `Rel` to each separated target
and outputting every digest, costs at least `c·(|T| + R_A + R_W)`. -/
theorem atomicCall_closes {c : ℕ} {pr : Prices100} (hpr : Floor c pr)
    {Adm : List (Op100 G) → List (G → ℚ) → Prop} {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {S : Set G}
    {Fcall X : Set (G → ℚ)} (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ Fcall) (hFX : Fcall ⊆ X)
    {Rel : (G → ℚ) → (G → ℚ) → Prop} (hS : OnS S Rel) {T DigA DigW : Finset (G → ℚ)} {RA RW : ℕ}
    [DecidableEq (G → ℚ)] (hRX : RelOf Rel T ⊆ X) (hsha : ShaWrites pr Adm fpSem S X (DigA ∪ DigW) (RA + RW))
    (hsep : SepBy Rel Fcall T) {L : List (G → ℚ)} {P : List (Op100 G)} (hatom : AtomicCall pr Fcall L P)
    (hadm : Adm P L) (hw : WritesBy Rel fpSem P L T) (hOD : OutputsOn fpSem S P L (DigA ∪ DigW)) :
    c * (T.card + RA + RW) ≤ cost100 pr P := by
  rw [Nat.add_assoc]
  exact hashCompose_by hpr hconst hFX hS hRX hsha hatom.1 hatom.2 hadm hsep hw hOD

/-- A run of the call model: its instructions, each on a device. -/
abbrev Run (Dev G : Type) := List (Dev × Op100 G)

/-- The run's instructions. -/
def ops (P : Run Dev G) : List (Op100 G) := P.map Prod.snd

/-- The devices the run uses. -/
def devs (P : Run Dev G) : Set Dev := {d | ∃ o, (d, o) ∈ P}

open Classical in
/-- The run's cost counted on the devices `I`. -/
noncomputable def costOn (pr : Prices100) (I : Set Dev) (P : Run Dev G) : ℕ :=
  cost100 pr (ops (P.filter fun p => p.1 ∈ I))

/-- **`PeriodInventory I Ph Pa`** (named premise; rank 3; T1's), the call model's form of the Accel premise of that
name for the devices that run the card's instructions: the honest run `Ph` and the adversary's run `Pa` use only
devices of `I`, so what either computes in the period is counted on `I`. The cards' hosts run no card instruction and
are counted at their rows (`PeriodInventoryT1`, §9); a device outside the network certifier's boundary is the egress
layer's (`egress_charged`). -/
def PeriodInventory (I : Set Dev) (Ph Pa : Run Dev G) : Prop := devs Ph ⊆ I ∧ devs Pa ⊆ I

theorem costOn_eq (pr : Prices100) {I : Set Dev} {P : Run Dev G} (h : devs P ⊆ I) :
    costOn pr I P = cost100 pr (ops P) := by
  classical
  unfold costOn
  congr 2
  refine List.filter_eq_self.2 fun p hp => ?_
  simpa using h ⟨p.2, hp⟩

/-- **A device outside the inventory is not counted**: without `PeriodInventory`, a run on such a device costs
nothing on `I`, whatever it computes. -/
theorem costOn_outside (pr : Prices100) {I : Set Dev} {d : Dev} (hd : d ∉ I) (Q : List (Op100 G)) :
    costOn pr I (Q.map fun o => (d, o)) = 0 := by
  classical
  unfold costOn
  have : (Q.map fun o => (d, o)).filter (fun p => decide (p.1 ∈ I)) = [] := by
    refine List.filter_eq_nil_iff.2 fun p hp => ?_
    obtain ⟨o, _, rfl⟩ := List.mem_map.1 hp
    simpa using hd
  rw [this]
  rfl

/-- The run's cost is its cost on `I` plus its cost off `I`. -/
theorem costOn_split (pr : Prices100) (I : Set Dev) (P : Run Dev G) :
    costOn pr I P + costOn pr Iᶜ P = cost100 pr (ops P) := by
  classical
  unfold costOn
  induction P with
  | nil => rfl
  | cons p P ih =>
    by_cases h : p.1 ∈ I
    · have h' : p.1 ∉ Iᶜ := fun h' => h' h
      simp only [List.filter_cons, h, h', decide_true, decide_false, Bool.false_eq_true, ↓reduceIte, ops,
        List.map_cons, cost100, List.sum_cons] at ih ⊢
      omega
    · have h' : p.1 ∈ Iᶜ := h
      simp only [List.filter_cons, h, h', decide_true, decide_false, Bool.false_eq_true, ↓reduceIte, ops,
        List.map_cons, cost100, List.sum_cons] at ih ⊢
      omega

/-- **The card is charged the bound less what ran off it** (rank 0 given `AtomicCall` over the joint run and
`ShaWrites`; the egress layer's interface): if the adversary's joint run, on the card's devices `I` and outside, meets
`atomicCall_closes`'s hypotheses, and the egress layer bounds the outside devices' cost by `E`, the card's cost is at
least `c·(|T| + R_A + R_W) − E`. -/
theorem egress_charged {c : ℕ} {pr : Prices100} (hpr : Floor c pr)
    {Adm : List (Op100 G) → List (G → ℚ) → Prop} {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {S : Set G}
    {Fcall X : Set (G → ℚ)} (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ Fcall) (hFX : Fcall ⊆ X)
    {Rel : (G → ℚ) → (G → ℚ) → Prop} (hS : OnS S Rel) {T DigA DigW : Finset (G → ℚ)} {RA RW : ℕ}
    [DecidableEq (G → ℚ)] (hRX : RelOf Rel T ⊆ X) (hsha : ShaWrites pr Adm fpSem S X (DigA ∪ DigW) (RA + RW))
    (hsep : SepBy Rel Fcall T) {I : Set Dev} {Pa : Run Dev G} {L : List (G → ℚ)} {E : ℕ}
    (hatom : AtomicCall pr Fcall L (ops Pa)) (hadm : Adm (ops Pa) L) (hw : WritesBy Rel fpSem (ops Pa) L T)
    (hOD : OutputsOn fpSem S (ops Pa) L (DigA ∪ DigW)) (hE : costOn pr Iᶜ Pa ≤ E) :
    c * (T.card + RA + RW) ≤ costOn pr I Pa + E := by
  have h := atomicCall_closes hpr hconst hFX hS hRX hsha hsep hatom hadm hw hOD
  rw [← costOn_split pr I Pa] at h
  omega

/-- **The pre-noise hashing of `A` is credited on `I`** (rank 0 given `PeriodInventory`, `AtomicCall` and
`ShaWrites`; T1's): with every device of both runs in `I`, the adversary's cost on `I` is at least
`c·(|T| + R_A + R_W)`. -/
theorem preNoiseHash_credited {c : ℕ} {pr : Prices100} (hpr : Floor c pr)
    {Adm : List (Op100 G) → List (G → ℚ) → Prop} {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {S : Set G}
    {Fcall X : Set (G → ℚ)} (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ Fcall) (hFX : Fcall ⊆ X)
    {Rel : (G → ℚ) → (G → ℚ) → Prop} (hS : OnS S Rel) {T DigA DigW : Finset (G → ℚ)} {RA RW : ℕ}
    [DecidableEq (G → ℚ)] (hRX : RelOf Rel T ⊆ X) (hsha : ShaWrites pr Adm fpSem S X (DigA ∪ DigW) (RA + RW))
    (hsep : SepBy Rel Fcall T) {I : Set Dev} {Ph Pa : Run Dev G} {L : List (G → ℚ)}
    (hinv : PeriodInventory I Ph Pa) (hatom : AtomicCall pr Fcall L (ops Pa)) (hadm : Adm (ops Pa) L)
    (hw : WritesBy Rel fpSem (ops Pa) L T) (hOD : OutputsOn fpSem S (ops Pa) L (DigA ∪ DigW)) :
    c * (T.card + RA + RW) ≤ costOn pr I Pa := by
  rw [costOn_eq pr hinv.2]
  exact atomicCall_closes hpr hconst hFX hS hRX hsha hsep hatom hadm hw hOD

/-- **Both sides on one inventory** (T1's): the gap on `I` is at most honest's whole cost less the
adversary's bound. -/
theorem gap_on_inventory {c : ℕ} {pr : Prices100} {I : Set Dev} {Ph Pa : Run Dev G} {B : ℕ}
    (hinv : PeriodInventory I Ph Pa) (hB : c * B ≤ costOn pr I Pa) :
    (costOn pr I Ph : ℤ) - costOn pr I Pa ≤ (cost100 pr (ops Ph) : ℤ) - c * B := by
  rw [costOn_eq pr hinv.1]
  have : ((c * B : ℕ) : ℤ) ≤ (costOn pr I Pa : ℤ) := by exact_mod_cast hB
  push_cast at this
  linarith

/-- **The hashing term closes to `δ` of honest's hashing**: if `ShaWrites` counts at least `(1 − δ)·R_h` of the
`R_h` hash registers honest writes, and honest pays at most `c·(|T| + R_h) + E`, the gap is at most
`c·δ·R_h + E`. -/
theorem hashTerm_closes {c Th Rh R E costH costA : ℕ} {δ : ℚ} (hR : (1 - δ) * Rh ≤ R)
    (hH : costH ≤ c * (Th + Rh) + E) (hA : c * (Th + R) ≤ costA) :
    (costH : ℚ) - costA ≤ c * δ * Rh + E := by
  have h1 : (costH : ℚ) ≤ c * (Th + Rh) + E := by exact_mod_cast hH
  have h2 : (c : ℚ) * (Th + R) ≤ costA := by exact_mod_cast hA
  have hc : (0 : ℚ) ≤ c := by positivity
  have h3 : (c : ℚ) * ((1 - δ) * Rh) ≤ c * R := mul_le_mul_of_nonneg_left hR hc
  nlinarith

/-- The toy digest word. -/
def td : Bool → ℚ := fun b => if b then 2 else 7

/-- **A toy instance meets every hypothesis of `preNoiseHash_credited` and attains its bound**: on two grid points,
one target `tw`, one digest word `td` (the toy's `root(A)`), the constants free and the target's carriers excluded,
with `Adm` the programs that write `td`; the honest run on one inventoried device costs exactly `32·(1 + 1)`. -/
theorem toy_attained :
    let S : Set Bool := Set.univ
    let X : Set (Bool → ℚ) := Consts Bool ∪ RelOf (Carries S) {tw}
    let Adm : List (Op100 Bool) → List (Bool → ℚ) → Prop := fun P L => td ∈ regs100 (fun _ _ _ _ => 0) P L
    let Pa : Run Unit Bool := [((), Op100.int tw), ((), Op100.int td)]
    ShaWrites h100Floor Adm (fun _ _ _ _ => 0) S X ({td} ∪ ∅) (1 + 0) ∧ SepBy (Carries S) (Consts Bool) {tw} ∧
      PeriodInventory Set.univ Pa Pa ∧ AtomicCall h100Floor (Consts Bool) [] (ops Pa) ∧ Adm (ops Pa) [] ∧
      WritesBy (Carries S) (fun _ _ _ _ => 0) (ops Pa) [] {tw} ∧
      OutputsOn (fun _ _ _ _ => 0) S (ops Pa) [] ({td} ∪ ∅) ∧
      costOn h100Floor Set.univ Pa = 32 * (({tw} : Finset (Bool → ℚ)).card + 1 + 0) := by
  intro S X Adm Pa
  have htdX : ∀ x ∈ X, ¬ ∀ g ∈ S, td g = x g := by
    rintro x (⟨q, rfl⟩ | ⟨t, ht, q, hq⟩) h
    · have h1 := h true trivial
      have h0 := h false trivial
      simp [td] at h1 h0
      rw [← h1] at h0
      norm_num at h0
    · simp at ht
      subst ht
      have h1 := h true trivial
      have h0 := h false trivial
      rw [hq true trivial] at h1
      rw [hq false trivial] at h0
      simp [td, tw] at h1 h0
      rw [← h0] at h1
      norm_num at h1
  have hregs : regs100 (fun _ _ _ _ => 0) (ops Pa) [] = [tw, td] := by
    simp [regs100, ops, Pa, Op100.outs]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro L P _ _ hA _
    refine ⟨{td}, by simp, ⟨fun t ht t' ht' _ => by simp at ht ht'; rw [ht, ht'], fun t ht x hx => ?_⟩,
      fun t ht => ⟨td, hA, fun g _ => by simp at ht; rw [ht]⟩⟩
    simp at ht
    rw [ht]
    exact htdX x hx
  · refine sepBy_carries (fun t ht t' ht' _ => by simp at ht ht'; rw [ht, ht']) fun t ht f hf => ?_
    simp at ht
    subst ht
    obtain ⟨q', rfl⟩ := hf
    rintro ⟨q, hq⟩
    have h1 := hq true trivial
    have h0 := hq false trivial
    simp [tw] at h1 h0
    rw [h0] at h1
    norm_num at h1
  · exact ⟨fun _ _ => trivial, fun _ _ => trivial⟩
  · refine ⟨by simp, fun o ho => ?_⟩
    simp [ops, Pa] at ho
    rcases ho with rfl | rfl <;> trivial
  · show td ∈ regs100 (fun _ _ _ _ => 0) (ops Pa) []
    rw [hregs]
    simp
  · intro t ht
    simp at ht
    exact ⟨tw, by rw [hregs]; simp, ⟨0, fun g _ => by rw [ht]; ring⟩⟩
  · intro t ht
    simp at ht
    exact ⟨td, by rw [hregs]; simp, fun g _ => by rw [ht]⟩
  · rw [costOn_eq (I := Set.univ) h100Floor (fun _ _ => Set.mem_univ _), Finset.card_singleton]
    decide

/-- **Outsourced hashing goes uncounted on the card** without `PeriodInventory`: the toy's run with its digest
computed on a device outside the card (`false ∉ I = {true}`) meets `ShaWrites`, `AtomicCall` over the joint run,
admissibility and the digest's output, yet costs the card `32`, below `32·(1 + 1)`. -/
theorem outsourced_uncredited :
    let S : Set Bool := Set.univ
    let X : Set (Bool → ℚ) := Consts Bool ∪ RelOf (Carries S) {tw}
    let Adm : List (Op100 Bool) → List (Bool → ℚ) → Prop := fun P L => td ∈ regs100 (fun _ _ _ _ => 0) P L
    let Pa : Run Bool Bool := [(true, Op100.int tw), (false, Op100.int td)]
    ShaWrites h100Floor Adm (fun _ _ _ _ => 0) S X ({td} ∪ ∅) (1 + 0) ∧
      AtomicCall h100Floor (Consts Bool) [] (ops Pa) ∧ Adm (ops Pa) [] ∧
      OutputsOn (fun _ _ _ _ => 0) S (ops Pa) [] ({td} ∪ ∅) ∧ ¬ PeriodInventory {true} Pa Pa ∧
      costOn h100Floor {true} Pa < 32 * (({tw} : Finset (Bool → ℚ)).card + 1 + 0) := by
  intro S X Adm Pa
  obtain ⟨hsha, -, -, hatom, hadm, -, hOD, -⟩ := toy_attained
  refine ⟨hsha, hatom, hadm, hOD, fun h => ?_, ?_⟩
  · have hm : false ∈ devs Pa := ⟨Op100.int td, by simp [Pa]⟩
    simpa using h.1 hm
  · simp [costOn, Pa, ops]
    decide

end Call

/-! ## 7. Lifting a data-case result to the call grid -/

section Lift

variable {G : Type}

/-- **Distinct at a data case, distinct on the call grid**: targets distinct on a sub-grid `S'` with its free
values `F'` are distinct on `S ⊇ S'` with free values `F`, when each of `F` restricts on `S'` to one of `F'` (the call's
input words are constants at a data case). -/
theorem distinctOn_lift {S S' : Set G} {F F' : Set (G → ℚ)} {T : Finset (G → ℚ)} (hS : S' ⊆ S)
    (hF : ∀ f ∈ F, ∃ f' ∈ F', ∀ g ∈ S', f g = f' g) (h : DistinctOn S' F' T) : DistinctOn S F T := by
  refine ⟨fun t ht t' ht' heq => h.1 t ht t' ht' fun g hg => heq g (hS hg), fun t ht f hf heq => ?_⟩
  obtain ⟨f', hf', hff'⟩ := hF f hf
  exact h.2 t ht f' hf' fun g hg => by rw [heq g (hS hg), hff' g hg]

/-- The same for carrier separation. -/
theorem sepBy_carries_lift {S S' : Set G} {F F' : Set (G → ℚ)} {T : Finset (G → ℚ)} (hS : S' ⊆ S)
    (hF : ∀ f ∈ F, ∃ f' ∈ F', ∀ g ∈ S', f g = f' g) (h : SepBy (Carries S') F' T) : SepBy (Carries S) F T := by
  refine ⟨fun t ht t' ht' r ⟨q, hq⟩ ⟨q', hq'⟩ =>
    h.1 t ht t' ht' r ⟨q, fun g hg => hq g (hS hg)⟩ ⟨q', fun g hg => hq' g (hS hg)⟩, fun t ht f hf ⟨q, hq⟩ => ?_⟩
  obtain ⟨f', hf', hff'⟩ := hF f hf
  exact h.2 t ht f' hf' ⟨q, fun g hg => by rw [← hff' g hg]; exact hq g (hS hg)⟩

end Lift

/-! ## 8. The ledger's cost forms: ShaSpeed in `MH100w` and `hash-compose` -/

section Ledger

variable {G : Type}

/-- **`ShaSpeedW1 pr Adm fpSem S F Dig B`** (named premise; rank 4a at `δ`, with `B = (1 − δ)` of honest's priced
hashing): every admissible well-formed program on free inputs that outputs the digests costs at least `B` in
`MH100w`. -/
def ShaSpeedW1 (pr : Prices100) (Adm : List (Op100 G) → List (G → ℚ) → Prop) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ)
    (S : Set G) (F : Set (G → ℚ)) (Dig : Finset (G → ℚ)) (B : ℕ) : Prop :=
  ∀ (L : List (G → ℚ)) (P : List (Op100 G)), (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) → Adm P L →
    OutputsOn fpSem S P L Dig → B ≤ cost100 pr P

/-- **`HashComposeW1 pr Adm fpSem S F T Dig c B`**, the ledger's `hash-compose`: every admissible well-formed program
on free inputs that outputs the committed words `T` and the digests costs at least `c·|T| + B`, so hashing shares no
work with the words. -/
def HashComposeW1 (pr : Prices100) (Adm : List (Op100 G) → List (G → ℚ) → Prop) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ)
    (S : Set G) (F : Set (G → ℚ)) (T Dig : Finset (G → ℚ)) (c B : ℕ) : Prop :=
  ∀ (L : List (G → ℚ)) (P : List (Op100 G)), (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) → Adm P L →
    OutputsOn fpSem S P L T → OutputsOn fpSem S P L Dig → c * T.card + B ≤ cost100 pr P

/-- **`ShaWrites` gives ShaSpeed's `MH100w` form** at `B = c·R`. -/
theorem shaSpeedW1_of_shaWrites {c : ℕ} {pr : Prices100} (hpr : Floor c pr)
    {Adm : List (Op100 G) → List (G → ℚ) → Prop} {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {S : Set G} {F X : Set (G → ℚ)}
    (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ X) (hFX : F ⊆ X) {Dig : Finset (G → ℚ)} {R : ℕ}
    (hsha : ShaWrites pr Adm fpSem S X Dig R) : ShaSpeedW1 pr Adm fpSem S F Dig (c * R) :=
  fun _ _ hL hP hA hO =>
    shaWrites_floor hpr hconst hsha (fun f hf => hFX (hL f hf)) (fun o ho => wfw_mono hFX o (hP o ho)) hA hO

/-- **`ShaWrites` gives the ledger's `hash-compose`** at `B = c·R`, for words that are message words (in `X`) and
`DistinctOn` targets. -/
theorem hashComposeW1_of_shaWrites {c : ℕ} {pr : Prices100} (hpr : Floor c pr)
    {Adm : List (Op100 G) → List (G → ℚ) → Prop} {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {S : Set G} {F X : Set (G → ℚ)}
    (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ F) (hFX : F ⊆ X) {T Dig : Finset (G → ℚ)} {R : ℕ}
    (hTX : ∀ t ∈ T, t ∈ X) (hT : DistinctOn S F T) (hsha : ShaWrites pr Adm fpSem S X Dig R) :
    HashComposeW1 pr Adm fpSem S F T Dig c (c * R) := fun _ _ hL hP hA hOT hOD => by
  have h := hashCompose hpr hconst hFX hTX hsha hL hP hA hT hOT hOD
  rwa [Nat.mul_add] at h

/-- **ShaSpeed's cost form and `DistinctOn` targets don't give `hash-compose`**: on the toy where the digest is the
target, ShaSpeed's form holds at `32` and the target is distinct, but a program outputting both costs `32`, below
`32·1 + 32`. -/
theorem hashComposeW1_not_of_shaSpeedW1 :
    ∃ Adm : List (Op100 Bool) → List (Bool → ℚ) → Prop,
      ShaSpeedW1 h100Floor Adm (fun _ _ _ _ => 0) Set.univ (Consts Bool) {tw} 32 ∧
        DistinctOn Set.univ (Consts Bool) {tw} ∧
        ¬ HashComposeW1 h100Floor Adm (fun _ _ _ _ => 0) Set.univ (Consts Bool) {tw} {tw} 32 32 := by
  obtain ⟨Adm, P, hsha, hA, hP, hT, hO, hc⟩ := compose_needs_exclusion
  refine ⟨Adm, fun L P' hL hP' hA' hO' => ?_, hT, fun h => ?_⟩
  · simpa using shaWrites_floor h32Floor (fun q => consts_mem q) hsha hL hP' hA' hO'
  · have h' := h [] P (by simp) hP hA hO hO
    rw [Finset.card_singleton] at hc h'
    omega

/-- `tw + td` on the toy grid: the fold's register, a `wgmma` on the word started from the round sum `td`. -/
def tf : Bool → ℚ := fun b => if b then 3 else 7

/-- **The joint count holds where `CarriedWords` fails** (the fold, on the toy grid): the hash writes its round sum
`td` and `tf = tw + td`, so no register carries the word `tw`, yet it writes `|T| + R = 2` registers, distinct and
non-constant. The pair implies the joint form (`jointWrites_of_carriedWords`), so the joint form is strictly weaker. -/
theorem fold_joint_not_carried :
    ∃ (Adm : List (Op100 Bool) → List (Bool → ℚ) → Prop) (P : List (Op100 Bool)),
      ShaWrites h100Floor Adm (fun _ _ _ _ => 0) Set.univ (Consts Bool) {tf} (({tw} : Finset _).card + 1) ∧
        Adm P [] ∧ (∀ o ∈ P, o.WFw h100Floor (Consts Bool)) ∧ OutputsOn (fun _ _ _ _ => 0) Set.univ P [] {tf} ∧
        DistinctOn Set.univ (Consts Bool) {tw} ∧ ¬ CarriedWords Adm (fun _ _ _ _ => 0) Set.univ {tw} {tf} := by
  have hregs : regs100 (fun _ _ _ _ => (0 : ℚ)) ([td, tf].map Op100.int) [] = [td, tf] := by
    simpa using regs100_map_int (fun _ _ _ _ => (0 : ℚ)) [td, tf] []
  have hne : td ≠ tf := by
    intro h
    have := congrFun h true
    norm_num [td, tf] at this
  have hnc : ∀ r ∈ ({td, tf} : Finset (Bool → ℚ)), ∀ f ∈ Consts Bool, ¬ ∀ g ∈ (Set.univ : Set Bool), r g = f g := by
    intro r hr f hf h
    obtain ⟨q, rfl⟩ := hf
    have h1 := h true trivial
    have h0 := h false trivial
    simp only [Finset.mem_insert, Finset.mem_singleton] at hr
    rcases hr with rfl | rfl <;> norm_num [td, tf] at h1 h0 <;> linarith
  have hcar : ∀ r ∈ [td, tf], ¬ Carries Set.univ tw r := by
    intro r hr ⟨q, hq⟩
    have h1 := hq true trivial
    have h0 := hq false trivial
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
    rcases hr with rfl | rfl <;> norm_num [td, tf, tw] at h1 h0 <;> linarith
  have hO : OutputsOn (fun _ _ _ _ => (0 : ℚ)) Set.univ ([td, tf].map Op100.int) [] {tf} := by
    intro t ht
    rw [Finset.mem_singleton] at ht
    exact ⟨tf, by rw [hregs]; simp, fun g _ => by rw [ht]⟩
  refine ⟨fun P L => P = [td, tf].map Op100.int ∧ L = [], [td, tf].map Op100.int, ?_, ⟨rfl, rfl⟩, ?_, hO, ?_, ?_⟩
  · rintro L P - - ⟨rfl, rfl⟩ -
    refine ⟨{td, tf}, by rw [Finset.card_pair hne, Finset.card_singleton],
      ⟨fun t _ t' _ h => funext fun g => h g trivial, hnc⟩, fun t ht => ⟨t, ?_, fun _ _ => rfl⟩⟩
    rw [hregs]
    simp only [Finset.mem_insert, Finset.mem_singleton] at ht
    rcases ht with rfl | rfl <;> simp
  · intro o ho
    simp only [List.map_cons, List.map_nil, List.mem_cons, List.not_mem_nil, or_false] at ho
    rcases ho with rfl | rfl <;> trivial
  · exact ⟨fun t ht t' ht' _ => by simp at ht ht'; rw [ht, ht'], fun t ht f hf => by
      simp at ht; rw [ht]; exact tw_not_const f hf⟩
  · intro h
    obtain ⟨r, hr, hc⟩ := h [] _ ⟨rfl, rfl⟩ hO tw (Finset.mem_singleton_self _)
    rw [hregs] at hr
    exact hcar r hr hc

end Ledger

/-! ## 9. The hosts inside T1: `PeriodInventoryT1` with the hosts counted at θ's rows -/

section Hosts

open Pouw.Dither (HostRes HostRow)

/-- A sum over the host resources. -/
theorem sum_hostRes {β : Type} [AddCommMonoid β] (f : HostRes → β) : ∑ r, f r = f .amx + f .sha := by
  show ∑ r ∈ ({HostRes.amx, HostRes.sha} : Finset HostRes), f r = _
  rw [Finset.sum_insert (by simp), Finset.sum_singleton]

/-- A host row's rate at a host resource: AMX in s8 MACs a second, SHA-512 in compressions a second. -/
def rateAt (hr : HostRow) : HostRes → ℝ
  | .amx => hr.amx
  | .sha => hr.sha

/-- **The hosts' counted W1 for a period** (`HostsCounted` in the call model): the `n` cards' hosts at `n` times the
row `hr`'s rate at each resource for the period's `Tp` seconds, a unit at its credited W1 `w r` (a compression at the
W1 the verifier credits for one, a MAC at a word's credit over its MACs). -/
noncomputable def hostsW1 (n : ℕ) (hr : HostRow) (w : HostRes → ℝ) (Tp : ℝ) : ℝ :=
  ∑ r, Tp * n * w r * rateAt hr r

theorem hostsW1_eq (n : ℕ) (hr : HostRow) (w : HostRes → ℝ) (Tp : ℝ) :
    hostsW1 n hr w Tp = Tp * n * w .amx * hr.amx + Tp * n * w .sha * hr.sha := by
  simp only [hostsW1, sum_hostRes, rateAt]

/-- **The headline's `HostsCounted` covers the hosts' counted W1**: when the period's devices count the `n` cards'
hosts at the row `hr` (`Pouw.Dither.HostsCounted`), some injection of the host resources into the other devices'
gives those devices, at the credited W1 a unit, at least `hostsW1` for the period. So the call model's count reads
the same rows as the headline's capacity. -/
theorem hostsW1_le_hostsCounted {c : Verity.Accel.Card} {D : Pouw.Dither.Devices c} {hr : HostRow}
    (h : Pouw.Dither.HostsCounted D hr) {w : HostRes → ℝ} {Tp : ℝ} (hTp : 0 ≤ Tp) (hw : ∀ r, 0 ≤ w r) :
    ∃ j : HostRes → D.oth.Res, Function.Injective j ∧ hostsW1 D.n hr w Tp ≤ ∑ r, Tp * w r * D.oth.rate (j r) := by
  obtain ⟨j, hj, ha, hs⟩ := h
  refine ⟨j, hj, ?_⟩
  have h1 : Tp * w .amx * (D.n * hr.amx) ≤ Tp * w .amx * D.oth.rate (j .amx) :=
    mul_le_mul_of_nonneg_left ha (mul_nonneg hTp (hw _))
  have h2 : Tp * w .sha * (D.n * hr.sha) ≤ Tp * w .sha * D.oth.rate (j .sha) :=
    mul_le_mul_of_nonneg_left hs (mul_nonneg hTp (hw _))
  rw [hostsW1_eq, sum_hostRes]
  linarith

/-- **`PeriodInventoryT1 Wc Wcard hc Wout`** (named premise; rank 3; T1's): inside the network certifier's boundary
every device is a card of the inventory or one of those cards' hosts. So the period's credited W1 `Wc` is at most what
the cards computed (`Wcard`), what the hosts computed through each host resource (`hc r`, at its credited W1), and what
devices outside the boundary computed (`Wout`), which reaches the devices only across the certifier and which the
egress layer bounds. -/
def PeriodInventoryT1 (Wc Wcard : ℝ) (hc : HostRes → ℝ) (Wout : ℝ) : Prop := Wc ≤ Wcard + ∑ r, hc r + Wout

/-- **`HostsWithinRow n hr w Tp hc`** (named premise: θ's host row is at least the hosts' best rate; rank 3 for the
row's cores a card and all-core clock, 4a for SHA-512's rate a core): through each host resource `r`, the `n` cards'
hosts compute in `Tp` seconds at most `Tp·n·rate_r` units' credited W1. It is resource by resource, so AMX's idle
capacity doesn't absorb SHA-512 beyond its row. -/
def HostsWithinRow (n : ℕ) (hr : HostRow) (w : HostRes → ℝ) (Tp : ℝ) (hc : HostRes → ℝ) : Prop :=
  ∀ r, hc r ≤ Tp * n * w r * rateAt hr r

/-- **With the hosts counted, the host's hashing is hashing** (rank 0 given `PeriodInventoryT1` and
`HostsWithinRow`): the period's credited W1 is at most the cards' share, the hosts' counted W1 and what came from
outside the boundary. -/
theorem hostsCounted_closes {Wc Wcard Wout Tp : ℝ} {n : ℕ} {hr : HostRow} {w hc : HostRes → ℝ}
    (hinv : PeriodInventoryT1 Wc Wcard hc Wout) (hrow : HostsWithinRow n hr w Tp hc) :
    Wc ≤ Wcard + hostsW1 n hr w Tp + Wout := by
  have : ∑ r, hc r ≤ hostsW1 n hr w Tp := Finset.sum_le_sum fun r _ => hrow r
  unfold PeriodInventoryT1 at hinv
  linarith

/-- **What a row below the hosts' best leaves** (rank 0 given `PeriodInventoryT1`): if the hosts compute through each
resource at most at the rates `best`, the credited W1 exceeds the cards' share, the hosts' counted W1 at the row `hr`
and the outside's by at most `Σ_r Tp·n·w_r·(best_r − rate_r)⁺`, resource by resource. -/
theorem hostRow_residual {Wc Wcard Wout Tp : ℝ} {n : ℕ} {hr best : HostRow} {w hc : HostRes → ℝ}
    (hTp : 0 ≤ Tp) (hw : ∀ r, 0 ≤ w r) (hinv : PeriodInventoryT1 Wc Wcard hc Wout)
    (hbest : HostsWithinRow n best w Tp hc) :
    Wc ≤ Wcard + hostsW1 n hr w Tp + ∑ r, Tp * n * w r * max 0 (rateAt best r - rateAt hr r) + Wout := by
  have key : ∀ r, hc r ≤ Tp * n * w r * rateAt hr r + Tp * n * w r * max 0 (rateAt best r - rateAt hr r) := by
    intro r
    have hk : 0 ≤ Tp * n * w r := mul_nonneg (mul_nonneg hTp (Nat.cast_nonneg n)) (hw r)
    have hm : rateAt best r ≤ rateAt hr r + max 0 (rateAt best r - rateAt hr r) := by
      have := le_max_right 0 (rateAt best r - rateAt hr r)
      linarith
    calc hc r ≤ Tp * n * w r * rateAt best r := hbest r
      _ ≤ Tp * n * w r * (rateAt hr r + max 0 (rateAt best r - rateAt hr r)) := mul_le_mul_of_nonneg_left hm hk
      _ = _ := by ring
  have hs : ∑ r, hc r ≤ ∑ r, (Tp * n * w r * rateAt hr r + Tp * n * w r * max 0 (rateAt best r - rateAt hr r)) :=
    Finset.sum_le_sum fun r _ => key r
  rw [Finset.sum_add_distrib] at hs
  unfold PeriodInventoryT1 at hinv
  unfold hostsW1
  linarith

/-- **T1 with the hosts, from the card's premises** (rank 0 given `ShaWrites`, `AtomicCall`, the cards'
`PeriodInventory`, `PeriodInventoryT1` and `HostsWithinRow`): the cards' share is the targets and digests their
program writes, `c·(|T| + R_A + R_W)`, which `preNoiseHash_credited` charges to the cards; the hosts' share is their
row's. So the period's credited W1 is at most the cards' cost on `I`, the hosts' counted W1 and the outside's. -/
theorem hostsCounted_call {G Dev : Type} {c : ℕ} {pr : Prices100} (hpr : Floor c pr)
    {Adm : List (Op100 G) → List (G → ℚ) → Prop} {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {S : Set G}
    {Fcall X : Set (G → ℚ)} (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ Fcall) (hFX : Fcall ⊆ X)
    {Rel : (G → ℚ) → (G → ℚ) → Prop} (hS : OnS S Rel) {T DigA DigW : Finset (G → ℚ)} {RA RW : ℕ}
    [DecidableEq (G → ℚ)] (hRX : RelOf Rel T ⊆ X) (hsha : ShaWrites pr Adm fpSem S X (DigA ∪ DigW) (RA + RW))
    (hsep : SepBy Rel Fcall T) {I : Set Dev} {Ph Pa : Run Dev G} {L : List (G → ℚ)}
    (hinvC : PeriodInventory I Ph Pa) (hatom : AtomicCall pr Fcall L (ops Pa)) (hadm : Adm (ops Pa) L)
    (hw : WritesBy Rel fpSem (ops Pa) L T) (hOD : OutputsOn fpSem S (ops Pa) L (DigA ∪ DigW))
    {Wc Wout Tp : ℝ} {n : ℕ} {hr : HostRow} {wh hc : HostRes → ℝ}
    (hinv : PeriodInventoryT1 Wc ((c * (T.card + RA + RW) : ℕ) : ℝ) hc Wout) (hrow : HostsWithinRow n hr wh Tp hc) :
    Wc ≤ (costOn pr I Pa : ℝ) + hostsW1 n hr wh Tp + Wout := by
  have hcard : ((c * (T.card + RA + RW) : ℕ) : ℝ) ≤ (costOn pr I Pa : ℝ) := by
    exact_mod_cast preNoiseHash_credited hpr hconst hFX hS hRX hsha hsep hinvC hatom hadm hw hOD
  have := hostsCounted_closes hinv hrow
  linarith

/-- **Route H gains no capacity** (rank 0 given `PeriodInventoryT1` and `HostsWithinRow`): if the cards' share is
at most the adversary's card cost and that cost, with the other work `W_other` it runs on the cards, fits the cards'
capacity `C_card` for the period, then the credited W1 and the other work together fit the period's capacity with the
hosts counted at their rows, plus what came from outside the boundary. So hashing moved to the hosts frees no card
capacity the period doesn't count. -/
theorem hostsCounted_capacity {Wc Wcard Wout Tp costA Wother Ccard : ℝ} {n : ℕ} {hr : HostRow} {w hc : HostRes → ℝ}
    (hinv : PeriodInventoryT1 Wc Wcard hc Wout) (hrow : HostsWithinRow n hr w Tp hc) (hcard : Wcard ≤ costA)
    (hcap : costA + Wother ≤ Ccard) : Wc + Wother ≤ Ccard + hostsW1 n hr w Tp + Wout := by
  have := hostsCounted_closes hinv hrow
  linarith

/-- **The row's own hosts are counted**: devices whose other devices are the row's hosts for their `n` cards
(`HostRow.others`) meet `HostsCounted` at that row. -/
theorem hostsCounted_others {c : Verity.Accel.Card} (n : ℕ) (hn : 0 < n) (f mem : ℝ) (hr : HostRow) :
    Pouw.Dither.HostsCounted (c := c) ⟨n, hn, f, hr.others n mem⟩ hr :=
  ⟨fun r => r, fun _ _ h => h, le_rfl, le_rfl⟩

/-- **The hosts' counted W1 is within the headline's capacity** (rank 0 given `OnCard`'s rates and `HostsCounted`):
at the `G` the headline is read at, the period's capacity (`Period.cap`) at the host resources, a unit at its credited
W1, is at least `hostsW1` for the period. So the term `hostsCounted_capacity` adds is capacity the headline's
`OPT_honest` and `C` already count. -/
theorem hostsW1_le_cap {GA : Pouw.Dither.Accel} {c : Verity.Accel.Card} {x} {D : Pouw.Dither.Devices c} {P} {L}
    {e : GA.Res ≃ Verity.Accel.Res ⊕ D.oth.Res} (hon : Pouw.Dither.OnCard GA c x D P L e) (Pd : Pouw.Dither.Period GA)
    {hr : HostRow} (h : Pouw.Dither.HostsCounted D hr) {w : HostRes → ℝ} (hT : 0 ≤ Pd.T) (hw : ∀ r, 0 ≤ w r) :
    ∃ j : HostRes → D.oth.Res, Function.Injective j ∧
      hostsW1 D.n hr w Pd.T ≤ ∑ r, w r * Pd.cap (e.symm (.inr (j r))) := by
  obtain ⟨j, hj, hle⟩ := hostsW1_le_hostsCounted h hT hw
  refine ⟨j, hj, hle.trans (le_of_eq (Finset.sum_congr rfl fun r _ => ?_))⟩
  rw [Pouw.Dither.Period.cap, hon.2.1]
  ring

/-- **`HostsWithinRow` is the headline's capacity at a row-counted host** (rank 0 given `OnCard`'s rates): if a
strategy fits the period (`Period.Fits`, which `FreedSound` reads at every other work `z` and `TimedWorkFits` gives at
`z = 0` for an accepted period's sampling blocks' work) and the host resources `j` are counted at no more than the
row, then its use of them, a unit at its credited W1, meets `HostsWithinRow` at that row. So for the hosts' timed work
the premise is the headline's own (G2) rate at θ's row. -/
theorem hostsWithinRow_of_fits {GA : Pouw.Dither.Accel} {c : Verity.Accel.Card} {x} {D : Pouw.Dither.Devices c} {P}
    {L} {e : GA.Res ≃ Verity.Accel.Res ⊕ D.oth.Res} (hon : Pouw.Dither.OnCard GA c x D P L e)
    {Pd : Pouw.Dither.Period GA} {X : GA.Res → ℝ} {z : Pd.J → ℝ} (hfit : Pd.Fits X z)
    (hneeds : ∀ k r, 0 ≤ Pd.needs k r) (hT : 0 ≤ Pd.T) {hr : HostRow} {j : HostRes → D.oth.Res}
    (hj : ∀ r, D.oth.rate (j r) ≤ D.n * rateAt hr r) {w : HostRes → ℝ} (hw : ∀ r, 0 ≤ w r) :
    HostsWithinRow D.n hr w Pd.T (fun r => w r * X (e.symm (.inr (j r)))) := by
  intro r
  have hz : 0 ≤ ∑ k, z k * Pd.needs k (e.symm (.inr (j r))) :=
    Finset.sum_nonneg fun k _ => mul_nonneg (hfit.1 k) (hneeds k _)
  have h1 := hfit.2 (e.symm (.inr (j r)))
  have h2 : Pd.cap (e.symm (.inr (j r))) ≤ D.n * rateAt hr r * Pd.T := by
    rw [Pouw.Dither.Period.cap, hon.2.1]
    exact mul_le_mul_of_nonneg_right (hj r) hT
  have hX : X (e.symm (.inr (j r))) ≤ Pd.T * D.n * rateAt hr r := by linarith
  calc w r * X (e.symm (.inr (j r))) ≤ w r * (Pd.T * D.n * rateAt hr r) := mul_le_mul_of_nonneg_left hX (hw r)
    _ = Pd.T * D.n * w r * rateAt hr r := by ring

/-- **The hosts inside T1 from the headline's terms** (rank 0 given `OnCard`'s rates and `PeriodInventoryT1`): when
the strategy fits the period and the host resources are counted at no more than the row, the credited W1 is at most
the cards' share, the hosts' counted W1 at the row and the outside's. `HostsWithinRow` is then no premise of its own. -/
theorem hostsCounted_closes_of_fits {GA : Pouw.Dither.Accel} {c : Verity.Accel.Card} {x} {D : Pouw.Dither.Devices c}
    {P} {L} {e : GA.Res ≃ Verity.Accel.Res ⊕ D.oth.Res} (hon : Pouw.Dither.OnCard GA c x D P L e)
    {Pd : Pouw.Dither.Period GA} {X : GA.Res → ℝ} {z : Pd.J → ℝ} (hfit : Pd.Fits X z)
    (hneeds : ∀ k r, 0 ≤ Pd.needs k r) (hT : 0 ≤ Pd.T) {hr : HostRow} {j : HostRes → D.oth.Res}
    (hj : ∀ r, D.oth.rate (j r) ≤ D.n * rateAt hr r) {w : HostRes → ℝ} (hw : ∀ r, 0 ≤ w r) {Wc Wcard Wout : ℝ}
    (hinv : PeriodInventoryT1 Wc Wcard (fun r => w r * X (e.symm (.inr (j r)))) Wout) :
    Wc ≤ Wcard + hostsW1 D.n hr w Pd.T + Wout :=
  hostsCounted_closes hinv (hostsWithinRow_of_fits hon hfit hneeds hT hj hw)

/-- **Route H on a toy, and the host row that closes it**: the card writes the word `tw` (32) and a host hashes the
digest, credited 32 through its SHA-512. The cards' `PeriodInventory` holds, and the credited 64 is over the card's
cost. Counted at a row of one compression a second (credited 32), the hosts meet `HostsWithinRow` and the bound of
`hostsCounted_closes` is attained; at a zero SHA-512 row, `HostsWithinRow` fails. -/
theorem routeH_toy :
    let Pa : Run Unit Bool := [((), Op100.int tw)]
    let hc : HostRes → ℝ := fun r => match r with
      | .amx => 0
      | .sha => 32
    PeriodInventory Set.univ Pa Pa ∧ PeriodInventoryT1 64 32 hc 0 ∧
      (costOn h100Floor Set.univ Pa : ℝ) + 0 < 64 ∧
      HostsWithinRow 1 ⟨0, 1⟩ (fun _ => 32) 1 hc ∧
      (64 : ℝ) = (costOn h100Floor Set.univ Pa : ℝ) + hostsW1 1 ⟨0, 1⟩ (fun _ => 32) 1 + 0 ∧
      ¬ HostsWithinRow 1 ⟨0, 0⟩ (fun _ => 32) 1 hc := by
  intro Pa hc
  have hcost : costOn h100Floor Set.univ Pa = 32 := by
    rw [costOn_eq (I := Set.univ) h100Floor (fun _ _ => Set.mem_univ _)]
    decide
  refine ⟨⟨fun _ _ => Set.mem_univ _, fun _ _ => Set.mem_univ _⟩, ?_, ?_, ?_, ?_, ?_⟩
  · show (64 : ℝ) ≤ 32 + ∑ r, hc r + 0
    rw [sum_hostRes]
    norm_num [hc]
  · rw [hcost]
    norm_num
  · intro r
    cases r <;> norm_num [hc, rateAt]
  · rw [hcost, hostsW1_eq]
    norm_num
  · intro h
    have := h .sha
    norm_num [hc, rateAt] at this

end Hosts

/-! ## 10. Which grid: the window's coins at the prover's values -/

section Slice

variable {G : Type}

/-- **The slice's statement is the stronger** (rank 0): `ShaWrites` on a part `S'` of the grid `S` gives it on `S`.
On the slice where the prover's chosen values are fixed and only the window's coins vary, a program specialized to
those values must match the digests only there; so the premise that binds it is the slice's, and the data grid's
follows from it. -/
theorem shaWrites_of_slice {pr : Prices100} {Adm : List (Op100 G) → List (G → ℚ) → Prop}
    {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {S S' : Set G} (hS : S' ⊆ S) {X : Set (G → ℚ)} {Dig : Finset (G → ℚ)} {R : ℕ}
    (h : ShaWrites pr Adm fpSem S' X Dig R) : ShaWrites pr Adm fpSem S X Dig R := by
  classical
  intro L P hL hP hA hOD
  obtain ⟨H, hR, hD, hO⟩ := h L P hL hP hA fun t ht => by
    obtain ⟨r, hr, hrS⟩ := hOD t ht
    exact ⟨r, hr, fun g hg => hrS g (hS hg)⟩
  choose! ρ hρR hρS using hO
  have hinj : Set.InjOn ρ H := fun t ht t' ht' heq => hD.1 t ht t' ht' fun g hg => by
    rw [← hρS t ht g hg, ← hρS t' ht' g hg, heq]
  refine ⟨H.image ρ, ?_, ⟨fun r hr r' hr' heq => ?_, fun r hr f hf heq => ?_⟩, fun r hr => ?_⟩
  · rw [Finset.card_image_of_injOn hinj]
    exact hR
  · obtain ⟨t, ht, rfl⟩ := Finset.mem_image.1 hr
    obtain ⟨t', ht', rfl⟩ := Finset.mem_image.1 hr'
    rw [hD.1 t ht t' ht' fun g hg => by rw [← hρS t ht g hg, ← hρS t' ht' g hg]; exact heq g (hS hg)]
  · obtain ⟨t, ht, rfl⟩ := Finset.mem_image.1 hr
    exact hD.2 t ht f hf fun g hg => by rw [← hρS t ht g hg]; exact heq g (hS hg)
  · obtain ⟨t, _, rfl⟩ := Finset.mem_image.1 hr
    exact ⟨ρ t, hρR t (by assumption), fun _ _ => rfl⟩

/-- **The data grid credits what the slice can't** (rank 0): on the two-point grid of a value the prover may choose,
the digest `tw` of that value is credited one register over every program. On the slice where the prover chose
`false`, the digest is the constant 0 there, an operand outputs it, and `ShaWrites` fails at the same count. So a
value block the prover chose (constant, or a repeat) drops out of `R` on the slice; `AtomicCall`, which reads the
call's input words, doesn't remove it. -/
theorem shaWrites_grid_not_slice :
    ShaWrites h100Floor (fun _ _ => True) (fun _ _ _ _ => 0) Set.univ (Consts Bool) {tw} 1 ∧
      ¬ ShaWrites h100Floor (fun _ _ => True) (fun _ _ _ _ => 0) {false} (Consts Bool) {tw} 1 := by
  refine ⟨fun L P _ _ _ hOD => ⟨{tw}, by simp, ⟨fun t ht t' ht' _ => by simp at ht ht'; rw [ht, ht'],
    fun t ht f hf => by simp at ht; rw [ht]; exact tw_not_const f hf⟩, hOD⟩, fun h => ?_⟩
  have hregs : regs100 (fun _ _ _ _ => 0) [Op100.operand (fun _ : Bool => (0 : ℚ))] [] = [fun _ => 0] := by
    simpa using regs100_map_operand (fun _ _ _ _ => 0) [fun _ : Bool => (0 : ℚ)] []
  obtain ⟨H, hR, hD, hO⟩ := h [] [Op100.operand (fun _ => 0)] (by simp)
    (fun o ho => by simp at ho; rw [ho]; exact consts_mem 0) trivial fun t ht => by
      simp at ht
      rw [ht, hregs]
      exact ⟨fun _ => 0, by simp, fun g hg => by simp at hg; rw [hg]; simp [tw]⟩
  obtain ⟨t, ht⟩ : H.Nonempty := Finset.card_pos.1 (by omega)
  obtain ⟨r, hr, hrS⟩ := hO t ht
  rw [hregs] at hr
  simp at hr
  rw [hr] at hrS
  exact hD.2 t ht _ (consts_mem 0) fun g hg => (hrS g hg).symm

end Slice

end Pouw.Dimension.NCompose

#print axioms Pouw.Dimension.NCompose.charged_by
#print axioms Pouw.Dimension.NCompose.shaWrites_floor
#print axioms Pouw.Dimension.NCompose.hashCompose_by
#print axioms Pouw.Dimension.NCompose.hashCompose
#print axioms Pouw.Dimension.NCompose.hashCompose_floor
#print axioms Pouw.Dimension.NCompose.shaWrites_unrestricted_false
#print axioms Pouw.Dimension.NCompose.shaWrites_false_of_free
#print axioms Pouw.Dimension.NCompose.compose_needs_exclusion
#print axioms Pouw.Dimension.NCompose.sepShift_needed
#print axioms Pouw.Dimension.NCompose.hashCompose_carried
#print axioms Pouw.Dimension.NCompose.hashCompose_encoded
#print axioms Pouw.Dimension.NCompose.hashCompose_of_carriedWords
#print axioms Pouw.Dimension.NCompose.jointWrites_of_carriedWords
#print axioms Pouw.Dimension.NCompose.composed_set
#print axioms Pouw.Dimension.NCompose.words_not_forced
#print axioms Pouw.Dimension.NCompose.bswap32_bswap32
#print axioms Pouw.Dimension.NCompose.bswap32_shift_not_const
#print axioms Pouw.Dimension.NCompose.bswap32_not_carrier
#print axioms Pouw.Dimension.NCompose.atomicCall_closes
#print axioms Pouw.Dimension.NCompose.costOn_outside
#print axioms Pouw.Dimension.NCompose.costOn_split
#print axioms Pouw.Dimension.NCompose.egress_charged
#print axioms Pouw.Dimension.NCompose.preNoiseHash_credited
#print axioms Pouw.Dimension.NCompose.gap_on_inventory
#print axioms Pouw.Dimension.NCompose.hashTerm_closes
#print axioms Pouw.Dimension.NCompose.toy_attained
#print axioms Pouw.Dimension.NCompose.outsourced_uncredited
#print axioms Pouw.Dimension.NCompose.distinctOn_lift
#print axioms Pouw.Dimension.NCompose.sepBy_carries_lift
#print axioms Pouw.Dimension.NCompose.shaSpeedW1_of_shaWrites
#print axioms Pouw.Dimension.NCompose.hashComposeW1_of_shaWrites
#print axioms Pouw.Dimension.NCompose.hashComposeW1_not_of_shaSpeedW1
#print axioms Pouw.Dimension.NCompose.fold_joint_not_carried
#print axioms Pouw.Dimension.NCompose.carriedWords_unrestricted_false
#print axioms Pouw.Dimension.NCompose.sum_hostRes
#print axioms Pouw.Dimension.NCompose.hostsW1_eq
#print axioms Pouw.Dimension.NCompose.hostsW1_le_hostsCounted
#print axioms Pouw.Dimension.NCompose.hostsCounted_closes
#print axioms Pouw.Dimension.NCompose.hostRow_residual
#print axioms Pouw.Dimension.NCompose.hostsCounted_call
#print axioms Pouw.Dimension.NCompose.hostsCounted_capacity
#print axioms Pouw.Dimension.NCompose.hostsCounted_others
#print axioms Pouw.Dimension.NCompose.hostsW1_le_cap
#print axioms Pouw.Dimension.NCompose.hostsWithinRow_of_fits
#print axioms Pouw.Dimension.NCompose.hostsCounted_closes_of_fits
#print axioms Pouw.Dimension.NCompose.routeH_toy
#print axioms Pouw.Dimension.NCompose.shaWrites_of_slice
#print axioms Pouw.Dimension.NCompose.shaWrites_grid_not_slice
