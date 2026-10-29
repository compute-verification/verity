import Proofs.Pouw.Dimension.H100FloorProofs

/-!
# `hash-compose` in `MH100w` under `ShaWrites`

On `MH100w`'s semantics (`Op100.WFw`, `DistinctWritesMH100Floor`).

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
* **Which grid** (`shaWrites_of_slice`, `shaWrites_grid_not_slice`): `ShaWrites` on the slice where the prover's chosen
  values are fixed and only the window's coins vary gives it on the data grid, not conversely; the data grid credits
  the digest of a value the prover chose, which the slice can't. So `R` is counted on the slice, and a chosen block's
  constant work drops out of it.
* **A keyed mix on the slice** (`mix_const_card_le`, `mix_nonconst`, `mix_nonconst_strict_needed`): a value word
  mixed with a key word (XOR or addition) is one constant on the slice only if the value takes at least as many values
  there as the key. So the mixed blocks' schedule stays in `R` unless the prover's choices after the coins reach as
  many values as the key has, and that count is tight.
* **Branching on the coins** (`shaWrites_branch`, `distinctOn_subslice`): `ShaWrites` on the slice says nothing
  about a path's program, which matches only on its part. Stated on every part of at least `m` points, it leaves at most
  `m` cheap coins a path. The honest witness holds on such parts once its registers' level sets have fewer than `m`
  points.
* **Lifting a data-case result to the call grid** (`distinctOn_lift`, `sepBy_lift`): a target set distinct at one
  data case, with that case's free values, is distinct on the call grid with the call's free values, when each call
  free value restricts to a data-case free value.
* **The ledger's cost forms** (`ShaSpeedW1`, `HashComposeW1`): ShaSpeed in `MH100w` and `hash-compose` as named
  Props. Both follow from `ShaWrites` (`shaSpeedW1_of_shaWrites`, `hashComposeW1_of_shaWrites`); the second doesn't
  follow from the first and distinct targets (`hashComposeW1_not_of_shaSpeedW1`), so the ledger needs `ShaWrites`,
  not ShaSpeed plus `hash-compose`.
-/

namespace Pouw.SecurityProofs.Lead.Compose

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

/-- Two target sets that are each distinct and non-free, and pairwise different across, are so together. -/
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

/-- **The count with any relation** (in `MH100w`): under `Floor c`, a well-formed program on free inputs that
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

/-- **`ShaWrites pr Adm fpSem S X Dig R`** (named premise, algorithmic knowledge about SHA-512): ShaSpeed
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

/-- **hash-compose with any relation** (given `ShaWrites`): under `Floor c`, an admissible well-formed
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

/-- **hash-compose** (given `ShaWrites`): a program that outputs `DistinctOn` targets, each a message word (in
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

/-- **hash-compose, carried** (given `ShaWrites`): targets written only through carriers, with the carriers
excluded from the hash's count. -/
theorem hashCompose_carried {c : ℕ} {pr : Prices100} (hpr : Floor c pr)
    {Adm : List (Op100 G) → List (G → ℚ) → Prop} {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {S : Set G} {F X : Set (G → ℚ)}
    (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ F) (hFX : F ⊆ X) {T Dig : Finset (G → ℚ)} {R : ℕ}
    (hRX : RelOf (Carries S) T ⊆ X) (hsha : ShaWrites pr Adm fpSem S X Dig R) {L : List (G → ℚ)}
    {P : List (Op100 G)} (hL : ∀ f ∈ L, f ∈ F) (hP : ∀ o ∈ P, o.WFw pr F) (hA : Adm P L)
    (hsep : SepBy (Carries S) F T) (hw : WritesBy (Carries S) fpSem P L T) (hOD : OutputsOn fpSem S P L Dig) :
    c * (T.card + R) ≤ cost100 pr P :=
  hashCompose_by hpr hconst hFX (onS_carries S) hRX hsha hL hP hA hsep hw hOD

/-- **hash-compose, encoded** (given `ShaWrites`): the same with encodings. -/
theorem hashCompose_encoded {c : ℕ} {pr : Prices100} (hpr : Floor c pr)
    {Adm : List (Op100 G) → List (G → ℚ) → Prop} {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {S : Set G} {F X : Set (G → ℚ)}
    (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ F) (hFX : F ⊆ X) {T Dig : Finset (G → ℚ)} {R : ℕ}
    (hRX : RelOf (Encodes S) T ⊆ X) (hsha : ShaWrites pr Adm fpSem S X Dig R) {L : List (G → ℚ)}
    {P : List (Op100 G)} (hL : ∀ f ∈ L, f ∈ F) (hP : ∀ o ∈ P, o.WFw pr F) (hA : Adm P L)
    (hsep : SepBy (Encodes S) F T) (hw : WritesBy (Encodes S) fpSem P L T) (hOD : OutputsOn fpSem S P L Dig) :
    c * (T.card + R) ≤ cost100 pr P :=
  hashCompose_by hpr hconst hFX (onS_encodes S) hRX hsha hL hP hA hsep hw hOD

/-- **`CarriedWords Adm fpSem S T Dig`** (named premise, about the hardware's instruction set): every admissible
program that outputs the digests writes, for each committed word, a register that carries it. `MH100w` alone doesn't
give it (`words_not_forced`). On little-endian rows SHA-512 reads `bswap32 w`, which is no carrier of `w`
(`bswap32_not_carrier`), so a `wgmma` whose accumulator absorbs a round's sum doesn't stand in for it; on big-endian
rows that fold drops the carriers of each later block's `M_0`. -/
def CarriedWords (Adm : List (Op100 G) → List (G → ℚ) → Prop) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (S : Set G)
    (T Dig : Finset (G → ℚ)) : Prop :=
  ∀ (L : List (G → ℚ)) (P : List (Op100 G)), Adm P L → OutputsOn fpSem S P L Dig → WritesBy (Carries S) fpSem P L T

/-- **hash-compose from the premises alone** (given `ShaWrites` and `CarriedWords`): an admissible program on
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

/-- **The joint count follows from the pair** (given `ShaWrites` and `CarriedWords`): `ShaWrites` with only
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

/-- A misaligned message word: the four bytes from byte `o` of the little-endian pair `w0 ‖ w1`, read big-endian. A row
whose length is no multiple of 4 puts its later words at such an offset in SHA-512's block. -/
def asm32 (o w0 w1 : ℕ) : ℕ :=
  bswap32 ((w0 / 256 ^ o + w1 * 256 ^ (4 - o)) % 4294967296)

/-- **A misaligned word is no free value**: at each offset `o ∈ {1, 2, 3}` it reads both committed words, and
on the grid where each is `0` or `2³² − 1` it is no constant and equals neither word. So a register holding it counts in
`ShaWrites`' `R` beside the hash's own; that a program must write it is the premise at the larger `R`. -/
theorem asm32_not_free : ∀ o ∈ ({1, 2, 3} : Finset ℕ),
    asm32 o 0 4294967295 ≠ asm32 o 0 0 ∧ asm32 o 4294967295 0 ≠ asm32 o 0 0 ∧
      asm32 o 0 4294967295 ≠ 0 ∧ asm32 o 0 4294967295 ≠ 4294967295 ∧
      asm32 o 4294967295 0 ≠ 0 ∧ asm32 o 4294967295 0 ≠ 4294967295 := by
  decide

end Bswap

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

/-- The toy digest word. -/
def td : Bool → ℚ := fun b => if b then 2 else 7

/-! ## 8. The ledger's cost forms: ShaSpeed in `MH100w` and `hash-compose` -/

section Ledger

variable {G : Type}

/-- **`ShaSpeedW1 pr Adm fpSem S F Dig B`** (named premise at `δ`, with `B = (1 − δ)` of honest's priced
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

/-! ## 10. Which grid: the window's coins at the prover's values -/

section Slice

variable {G : Type}

/-- **The slice's statement is the stronger**: `ShaWrites` on a part `S'` of the grid `S` gives it on `S`.
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

/-- **The data grid credits what the slice can't**: on the two-point grid of a value the prover may choose,
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

/-- **A keyed mix is one constant on the slice only if the value has as many values as the key**. A value word
`V` is mixed with a key word `Z` by `mix`, cancellative in the key (XOR, or addition). If the mixed word is one constant
on the finite slice `S`, `Z` is a function of `V` there, so `V` takes at least as many values on `S` as `Z`. -/
theorem mix_const_card_le {W : Type} [DecidableEq W] (mix : W → W → W)
    (hmix : ∀ v z z', mix v z = mix v z' → z = z') (S : Finset G) (V Z : G → W) {c : W}
    (h : ∀ g ∈ S, mix (V g) (Z g) = c) : (S.image Z).card ≤ (S.image V).card := by
  have hinj : Set.InjOn Prod.fst (↑(S.image fun g => (V g, Z g)) : Set (W × W)) := by
    rintro ⟨v, z⟩ hp ⟨v', z'⟩ hp' hvv
    change v = v' at hvv
    obtain ⟨g, hg, hgp⟩ := Finset.mem_image.1 (Finset.mem_coe.1 hp)
    obtain ⟨g', hg', hgp'⟩ := Finset.mem_image.1 (Finset.mem_coe.1 hp')
    simp only [Prod.mk.injEq] at hgp hgp'
    have e1 : mix v z = c := by rw [← hgp.1, ← hgp.2]; exact h g hg
    have e2 : mix v z' = c := by rw [hvv, ← hgp'.1, ← hgp'.2]; exact h g' hg'
    exact Prod.ext hvv (hmix v z z' (e1.trans e2.symm))
  calc (S.image Z).card = ((S.image fun g => (V g, Z g)).image Prod.snd).card := by
        rw [Finset.image_image]; rfl
    _ ≤ (S.image fun g => (V g, Z g)).card := Finset.card_image_le
    _ = ((S.image fun g => (V g, Z g)).image Prod.fst).card := (Finset.card_image_of_injOn hinj).symm
    _ = (S.image V).card := by rw [Finset.image_image]; rfl

/-- **Fewer values than the key leave the mix live**: if `V` takes fewer values on the slice than `Z`, no
constant equals the mixed word on all of it, so its schedule isn't free there. -/
theorem mix_nonconst {W : Type} [DecidableEq W] (mix : W → W → W)
    (hmix : ∀ v z z', mix v z = mix v z' → z = z') (S : Finset G) (V Z : G → W)
    (hlt : (S.image V).card < (S.image Z).card) : ¬ ∃ c, ∀ g ∈ S, mix (V g) (Z g) = c :=
  fun ⟨_, h⟩ => absurd (mix_const_card_le mix hmix S V Z h) (not_le.mpr hlt)

/-- **The count is tight**: on the two-point grid, a value that takes the key's two values cancels the XOR
mix everywhere, so `mix_nonconst` needs fewer values, not at most as many. -/
theorem mix_nonconst_strict_needed :
    ∃ V Z : Bool → Bool, (Finset.univ.image V).card = (Finset.univ.image Z).card ∧
      ∀ g ∈ (Finset.univ : Finset Bool), Bool.xor (V g) (Z g) = false :=
  ⟨id, id, rfl, fun g _ => by cases g <;> rfl⟩

/-- **The honest witness survives on a large part of the slice**. If no two counted registers agree, and no
register agrees with a free value, on `m` or more points of the slice `S`, they are pairwise distinct and non-free on
every part `S'` of at least `m` points. So a `DistinctOn` witness checked on the slice holds on each part a branching
program can take, once its registers' level sets are small. -/
theorem distinctOn_subslice {S S' : Finset G} (hS : S' ⊆ S) {m : ℕ} (hm : m ≤ S'.card) {X : Set (G → ℚ)}
    {H : Finset (G → ℚ)} (hpair : ∀ t ∈ H, ∀ t' ∈ H, t ≠ t' → (S.filter fun g => t g = t' g).card < m)
    (hfree : ∀ t ∈ H, ∀ f ∈ X, (S.filter fun g => t g = f g).card < m) : DistinctOn (↑S' : Set G) X H := by
  refine ⟨fun t ht t' ht' heq => ?_, fun t ht f hf heq => ?_⟩
  · by_contra hne
    have hsub : S' ⊆ S.filter fun g => t g = t' g := fun g hg =>
      Finset.mem_filter.2 ⟨hS hg, heq g (Finset.mem_coe.2 hg)⟩
    exact absurd (hm.trans (Finset.card_le_card hsub)) (not_le.2 (hpair t ht t' ht' hne))
  · have hsub : S' ⊆ S.filter fun g => t g = f g := fun g hg =>
      Finset.mem_filter.2 ⟨hS hg, heq g (Finset.mem_coe.2 hg)⟩
    exact absurd (hm.trans (Finset.card_le_card hsub)) (not_le.2 (hfree t ht f hf))

/-- **A program that branches on the coins is cheap on few of them** (given `ShaWrites` on every part of at
least `m` points). Each path `i` runs a straight-line program `P i` that outputs the digests only on its part of the
slice, the coins `path` sends to it. If `ShaWrites` holds on every part of the slice of at least `m` points, the coins on
which the program costs less than `c·R` number at most `m` a path. `ShaWrites` on the whole slice doesn't give this:
a path's program needn't match there (`shaWrites_grid_not_slice`). -/
theorem shaWrites_branch {c : ℕ} {pr : Prices100} (hpr : Floor c pr) {Adm : List (Op100 G) → List (G → ℚ) → Prop}
    {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {X : Set (G → ℚ)} (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ X)
    {Dig : Finset (G → ℚ)} {R m : ℕ} {ι : Type} [DecidableEq ι] (S : Finset G) (path : G → ι)
    (L : ι → List (G → ℚ)) (P : ι → List (Op100 G))
    (hsha : ∀ S' ⊆ S, m ≤ S'.card → ShaWrites pr Adm fpSem (↑S' : Set G) X Dig R)
    (hL : ∀ i, ∀ f ∈ L i, f ∈ X) (hP : ∀ i, ∀ o ∈ P i, o.WFw pr X) (hA : ∀ i, Adm (P i) (L i))
    (hOD : ∀ i, OutputsOn fpSem (↑(S.filter fun g => path g = i) : Set G) (P i) (L i) Dig) :
    (S.filter fun g => cost100 pr (P (path g)) < c * R).card ≤ (S.image path).card * m := by
  rw [Finset.card_eq_sum_card_fiberwise (f := path) (t := S.image path) fun g hg =>
    Finset.mem_coe.2 (Finset.mem_image_of_mem path (Finset.mem_filter.1 (Finset.mem_coe.1 hg)).1)]
  refine (Finset.sum_le_card_nsmul _ _ m fun i _ => ?_).trans (smul_eq_mul _ _).le
  by_cases hbig : m ≤ (S.filter fun g => path g = i).card
  · have hcost : c * R ≤ cost100 pr (P i) :=
      shaWrites_floor hpr hconst (hsha _ (Finset.filter_subset _ _) hbig) (hL i) (hP i) (hA i) (hOD i)
    have hempty : ((S.filter fun g => cost100 pr (P (path g)) < c * R).filter fun g => path g = i) = ∅ :=
      Finset.filter_eq_empty_iff.2 fun g hg hgi => by
        have hlt := (Finset.mem_filter.1 hg).2
        rw [hgi] at hlt
        exact absurd hcost (not_le.2 hlt)
    rw [hempty, Finset.card_empty]
    exact Nat.zero_le _
  · refine (Finset.card_le_card fun g hg => ?_).trans (Nat.le_of_lt (not_le.1 hbig))
    obtain ⟨hgb, hgi⟩ := Finset.mem_filter.1 hg
    exact Finset.mem_filter.2 ⟨(Finset.mem_filter.1 hgb).1, hgi⟩

end Slice

end Pouw.SecurityProofs.Lead.Compose
