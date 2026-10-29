import Proofs.Pouw.Dimension.H100FloorProofs
import Proofs.Pouw.Dimension.H100Proofs
import Proofs.Pouw.NCP.Proofs

/-!
# Committed formed operands and the final word under `DistinctWritesFloor` (scratch; not in a lock)

**Draft, pending review.** Two questions against `MH100w`'s `DistinctWritesFloor` (`Pouw.Dimension.H100Floor`) at the
floor table `h100Floor`, which is the H200's measured table (`h200Measured = h100Floor`).

* **Committed formed operands** (`formed_charged`, `formed_free_of_saltIndep`, `rawDither_exact`). Commit the formed
  noised operands `X = A + D` and `Y = B + E`, int8 packed four to a 32-bit register (`pack4`). On the grid `D × Ω`
  (data fixed before the noise, then the noise), with the free set `FreeC1 C`: every salt-independent value and the
  raw coin words `C`. A formed register is charged 32 iff it moves with the salt at some fixed datum on `S` and equals
  no coin word on `S`; distinct registers count once each. Otherwise a zero-cost program outputs it
  (`formed_free_of_saltIndep`, `free_costs_nothing`), so no bound charges it. For an i.i.d. raw-byte dither the charged
  registers are exactly those whose data register is not `0` mod 256 (`rawDither_exact`).
* **The final word** (`finalWordTarget`, `finalWord_charged`). On a call's grid `Call m n k` (its int7 input `A` and
  its noise, the weights `B` fixed) with the free set the constants and the call's own words (its input words
  `inWords`, and noise-side words `W'`), NCP's final words `A_i·B_j` and every non-final checked word are pairwise
  distinct and none is free. The non-final words rest on the proved `indep_words_sub` at the input-`0` slice; the
  final words on `B`'s columns being nonzero, pairwise distinct and int8. The free set is the words themselves, not
  their span: on a weights slice that span holds every final word. The grid ranges over every int7 input: with `A`
  fixed, `A·B` is a constant, and constants are free.
* **Both** (`allTarget`, `all_charged`; concretely `block1_charged` and `ncp_charged`). With the free values the
  constants, the input words and the coin words (`coinWords`), the committed `X` registers (all three blocks, biased
  packing: `xAll`, under `xAll_props`' conditions on `R`) and block 1 of `Y` (`yWord`) are targets beside the checked
  and final words.
-/

namespace Pouw.Dimension.SwLean

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension Pouw.Dimension.H100

/-! ## Separation on a grid -/

section Generic

variable {G : Type}

/-- `f` and `h` differ at a point of `S`. -/
theorem differ_of_point {S : Set G} {f h : G → ℚ} {g : G} (hg : g ∈ S) (hne : f g ≠ h g) :
    ¬ ∀ x ∈ S, f x = h x :=
  fun heq => hne (heq g hg)

/-- `f` is unchanged along every step `St` inside `S`. -/
def Ignores (S : Set G) (St : G → G → Prop) (f : G → ℚ) : Prop :=
  ∀ g ∈ S, ∀ g' ∈ S, St g g' → f g = f g'

/-- `f` changes along some step `St` inside `S`. -/
def Moves (S : Set G) (St : G → G → Prop) (f : G → ℚ) : Prop :=
  ∃ g ∈ S, ∃ g' ∈ S, St g g' ∧ f g ≠ f g'

/-- **Separation**: a function that moves along a step differs on `S` from one that ignores it. -/
theorem differ_of_moves {S : Set G} {St : G → G → Prop} {f h : G → ℚ} (hm : Moves S St f) (hi : Ignores S St h) :
    ¬ ∀ x ∈ S, f x = h x := by
  intro heq
  obtain ⟨g, hg, g', hg', hs, hne⟩ := hm
  exact hne (by rw [heq g hg, heq g' hg', hi g hg g' hg' hs])

theorem ignores_const (S : Set G) (St : G → G → Prop) (q : ℚ) : Ignores S St (fun _ => q) :=
  fun _ _ _ _ _ => rfl

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

/-- Cross-different target sets are disjoint, so their counts add. -/
theorem card_union_of_cross [DecidableEq (G → ℚ)] {S : Set G} {T₁ T₂ : Finset (G → ℚ)}
    (hx : ∀ t ∈ T₁, ∀ t' ∈ T₂, ¬ ∀ g ∈ S, t g = t' g) : (T₁ ∪ T₂).card = T₁.card + T₂.card := by
  refine Finset.card_union_of_disjoint (Finset.disjoint_left.2 fun t h₁ h₂ => hx t h₁ t h₂ fun _ _ => rfl)

/-- The image of a family that differs pairwise on `S` has one element per index. -/
theorem card_image_of_differ {ι : Type} [DecidableEq (G → ℚ)] {S : Set G} (s : Finset ι) (t : ι → G → ℚ)
    (h : ∀ i ∈ s, ∀ j ∈ s, i ≠ j → ¬ ∀ g ∈ S, t i g = t j g) : (s.image t).card = s.card := by
  classical
  refine Finset.card_image_of_injOn fun i hi j hj hij => ?_
  by_contra hne
  exact h i hi j hj hne fun g _ => by rw [hij]

/-- **The floor charges every target**: at `h100Floor` (the H200's measured table), a well-formed `MH100w` program on
free inputs that matches on `S` a target set distinct and non-free on `S` costs at least 32 a target. -/
theorem charged (F : Set (G → ℚ)) (hF : ∀ q : ℚ, (fun _ : G => q) ∈ F) (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ)
    (L : List (G → ℚ)) (P : List (Op100 G)) (T : Finset (G → ℚ)) (hL : ∀ f ∈ L, f ∈ F)
    (hP : ∀ o ∈ P, o.WFw h100Floor F) (hT : DistinctOn S F T) (hO : OutputsOn fpSem S P L T) :
    32 * T.card ≤ cost100 h100Floor P :=
  distinctWritesFloor G F hF S fpSem L P T hL hP hT hO

/-- **A free value costs nothing**: one `operand` word outputs it, at cost 0. So no bound charges a target that
equals a free value on `S`. -/
theorem free_costs_nothing (F : Set (G → ℚ)) (pr : Prices100) (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ)
    (L : List (G → ℚ)) (t f : G → ℚ) (hf : f ∈ F) (ht : ∀ g ∈ S, t g = f g) :
    (∀ o ∈ [Op100.operand f], o.WFw pr F) ∧ OutputsOn fpSem S [Op100.operand f] L {t} ∧
      cost100 pr [Op100.operand f] = 0 := by
  refine ⟨fun o ho => ?_, fun t' ht' => ?_, rfl⟩
  · rw [List.mem_singleton] at ho
    subst ho
    exact hf
  · rw [Finset.mem_singleton] at ht'
    subst ht'
    refine ⟨f, ?_, fun g hg => (ht g hg).symm⟩
    simp [regs100, Op100.outs]

end Generic

/-! ## Packing four int8 entries into one register -/

/-- The byte an int8 entry is stored as (two's complement). -/
def byte (x : ℤ) : ℤ := x % 256

/-- Four int8 entries packed little-endian into one 32-bit register, read as its unsigned value. -/
def pack4 (v : Fin 4 → ℤ) : ℤ := byte (v 0) + 256 * byte (v 1) + 65536 * byte (v 2) + 16777216 * byte (v 3)

theorem pack4_eq_iff (v w : Fin 4 → ℤ) : pack4 v = pack4 w ↔ ∀ t, byte (v t) = byte (w t) := by
  constructor
  · intro h
    simp only [pack4, byte] at h
    have h0 : byte (v 0) = byte (w 0) := by simp only [byte]; omega
    have h1 : byte (v 1) = byte (w 1) := by simp only [byte]; omega
    have h2 : byte (v 2) = byte (w 2) := by simp only [byte]; omega
    have h3 : byte (v 3) = byte (w 3) := by simp only [byte]; omega
    intro t
    fin_cases t
    exacts [h0, h1, h2, h3]
  · intro h
    simp only [pack4, h 0, h 1, h 2, h 3]

/-- A register as a function on the grid. -/
def reg {G : Type} (X : G → Fin 4 → ℤ) : G → ℚ := fun g => ((pack4 (X g) : ℤ) : ℚ)

/-- Two registers, at two grid points, are equal iff their four bytes are. -/
theorem reg_eq_iff {G G' : Type} (X : G → Fin 4 → ℤ) (Y : G' → Fin 4 → ℤ) (g : G) (g' : G') :
    reg X g = reg Y g' ↔ ∀ t, byte (X g t) = byte (Y g' t) := by
  simp only [reg, Int.cast_inj]
  exact pack4_eq_iff _ _

/-! ## Candidate 1: committed formed operands -/

section Candidate1

variable {D Ω : Type}

/-- **Salt-independent values** on `D × Ω`: functions of the data (fixed before the noise) alone. -/
def SaltIndep (D Ω : Type) : Set (D × Ω → ℚ) := {f | ∀ d ω ω', f (d, ω) = f (d, ω')}

/-- **Candidate 1's free set**: every salt-independent value, and the raw coin words `C`. -/
def FreeC1 (C : Set (D × Ω → ℚ)) : Set (D × Ω → ℚ) := SaltIndep D Ω ∪ C

theorem const_mem_freeC1 (C : Set (D × Ω → ℚ)) (q : ℚ) : (fun _ : D × Ω => q) ∈ FreeC1 C :=
  Or.inl fun _ _ _ => rfl

/-- The salt step: the same datum, any two salts. -/
def SaltStep : D × Ω → D × Ω → Prop := fun g g' => g.1 = g'.1

theorem saltIndep_ignores (S : Set (D × Ω)) {f : D × Ω → ℚ} (hf : f ∈ SaltIndep D Ω) : Ignores S SaltStep f := by
  rintro ⟨d, ω⟩ _ ⟨d', ω'⟩ _ (h : d = d')
  subst h
  exact hf d ω ω'

/-- **Candidate 1, the bound's side**: formed registers `X r`, each moving with the salt at a fixed datum on `S`
(some byte changes), none equal on `S` to a coin word, and pairwise different on `S` (some byte differs), are
distinct and non-free, one target each. -/
theorem formed_charged {ι : Type} [DecidableEq (D × Ω → ℚ)] (S : Set (D × Ω)) (C : Set (D × Ω → ℚ)) (ρ : Finset ι)
    (X : ι → D × Ω → Fin 4 → ℤ)
    (hmove : ∀ r ∈ ρ, ∃ g ∈ S, ∃ g' ∈ S, g.1 = g'.1 ∧ ∃ t, byte (X r g t) ≠ byte (X r g' t))
    (hcoin : ∀ r ∈ ρ, ∀ c ∈ C, ¬ ∀ g ∈ S, reg (X r) g = c g)
    (hdist : ∀ r ∈ ρ, ∀ r' ∈ ρ, r ≠ r' → ∃ g ∈ S, ∃ t, byte (X r g t) ≠ byte (X r' g t)) :
    DistinctOn S (FreeC1 C) (ρ.image fun r => reg (X r)) ∧ (ρ.image fun r => reg (X r)).card = ρ.card := by
  have hdiff : ∀ r ∈ ρ, ∀ r' ∈ ρ, r ≠ r' → ¬ ∀ g ∈ S, reg (X r) g = reg (X r') g := by
    intro r hr r' hr' hne heq
    obtain ⟨g, hg, t, ht⟩ := hdist r hr r' hr' hne
    exact ht (((reg_eq_iff _ _ g g).1 (heq g hg)) t)
  refine ⟨⟨?_, ?_⟩, card_image_of_differ ρ _ hdiff⟩
  · intro t ht t' ht' h
    obtain ⟨r, hr, rfl⟩ := Finset.mem_image.1 ht
    obtain ⟨r', hr', rfl⟩ := Finset.mem_image.1 ht'
    by_cases hrr : r = r'
    · rw [hrr]
    · exact absurd h (hdiff r hr r' hr' hrr)
  · intro t ht f hf
    obtain ⟨r, hr, rfl⟩ := Finset.mem_image.1 ht
    rcases hf with hf | hf
    · refine differ_of_moves (St := SaltStep) ?_ (saltIndep_ignores S hf)
      obtain ⟨g, hg, g', hg', hgg, t, htt⟩ := hmove r hr
      exact ⟨g, hg, g', hg', hgg, fun h => htt ((reg_eq_iff _ _ g g').1 h t)⟩
    · exact hcoin r hr f hf

/-- **Candidate 1, the converse**: a formed register that does not move with the salt on `S` (every byte the same at
the same datum) equals a salt-independent value on `S`, so it is free (`free_costs_nothing`). This covers every
salt-independent dither, `D = −A` (`X = 0`) and `D = 0` (`X = A`) among them. -/
theorem formed_free_of_saltIndep (S : Set (D × Ω)) (X : D × Ω → Fin 4 → ℤ)
    (h : ∀ g ∈ S, ∀ g' ∈ S, g.1 = g'.1 → ∀ t, byte (X g t) = byte (X g' t)) :
    ∃ f ∈ SaltIndep D Ω, ∀ g ∈ S, reg X g = f g := by
  classical
  let f : D × Ω → ℚ := fun g => if hd : ∃ ω, (g.1, ω) ∈ S then reg X (g.1, hd.choose) else 0
  refine ⟨f, fun d ω ω' => rfl, fun g hg => ?_⟩
  have hd : ∃ ω, (g.1, ω) ∈ S := ⟨g.2, hg⟩
  simp only [f]
  rw [dif_pos hd]
  exact (reg_eq_iff X X g (g.1, hd.choose)).2 fun t => h g hg (g.1, hd.choose) hd.choose_spec rfl t

/-- **`D = −A`**: the formed register is `0` everywhere, a constant, so free. -/
theorem negDither_free (A : D → Fin 4 → ℤ) (S : Set (D × Ω)) :
    ∃ f ∈ SaltIndep D Ω, ∀ g ∈ S, reg (fun g => A g.1 + -A g.1) g = f g :=
  ⟨fun _ => 0, fun _ _ _ => rfl, fun g _ => by simp [reg, pack4, byte]⟩

end Candidate1

/-! ### An i.i.d. raw-byte dither on a data slice -/

section RawDither

variable {ι : Type} [DecidableEq ι]

/-- The raw coin bytes: one 4-byte coin word a register. -/
abbrev Coins (ι : Type) : Type := ι → Fin 4 → ℤ

/-- The grid points whose coin bytes are bytes. -/
def rawS (ι : Type) : Set (Unit × Coins ι) := {g | ∀ r t, 0 ≤ g.2 r t ∧ g.2 r t < 256}

/-- The raw coin words: register `r`'s coin bytes, packed. -/
def rawCoins (ι : Type) : Set (Unit × Coins ι → ℚ) := {c | ∃ r, c = reg fun g => g.2 r}

/-- The formed register `A r + D r`, with `D r` register `r`'s raw coin bytes. -/
def rawX (A : ι → Fin 4 → ℤ) (r : ι) : Unit × Coins ι → Fin 4 → ℤ := fun g => A r + g.2 r

/-- **The raw-byte dither, exactly**: on the data slice with the noise every byte value, the formed registers whose data
register is not `0` mod 256 are distinct and non-free (one target each), and each one whose data register is `0` mod
256 equals its coin word on the grid, so it is free. -/
theorem rawDither_exact [DecidableEq (Unit × Coins ι → ℚ)] (A : ι → Fin 4 → ℤ) (ρ : Finset ι) :
    (DistinctOn (rawS ι) (FreeC1 (rawCoins ι)) ((ρ.filter fun r => ∃ t, byte (A r t) ≠ 0).image fun r => reg (rawX A r)) ∧
      ((ρ.filter fun r => ∃ t, byte (A r t) ≠ 0).image fun r => reg (rawX A r)).card =
        (ρ.filter fun r => ∃ t, byte (A r t) ≠ 0).card) ∧
    ∀ r, (∀ t, byte (A r t) = 0) → ∀ g ∈ rawS ι, reg (rawX A r) g = (reg fun g => g.2 r) g := by
  have hzero : ((), fun _ _ => (0 : ℤ)) ∈ rawS ι := fun _ _ => by
    show (0 : ℤ) ≤ 0 ∧ (0 : ℤ) < 256
    omega
  refine ⟨formed_charged _ _ _ _ ?_ ?_ ?_, ?_⟩
  · intro r _
    refine ⟨((), fun _ _ => 0), hzero, ((), fun s _ => if s = r then 1 else 0), fun s t => ?_, rfl, 0, ?_⟩
    · show 0 ≤ (if s = r then (1 : ℤ) else 0) ∧ (if s = r then (1 : ℤ) else 0) < 256
      split_ifs <;> omega
    · show byte (A r 0 + 0) ≠ byte (A r 0 + if r = r then 1 else 0)
      rw [if_pos rfl]
      simp only [byte]
      omega
  · intro r hr c hc heq
    obtain ⟨r', rfl⟩ := hc
    obtain ⟨t, ht⟩ := (Finset.mem_filter.1 hr).2
    have h0 : byte (A r t + 0) = byte 0 := (reg_eq_iff _ _ _ _).1 (heq _ hzero) t
    simp only [byte] at h0 ht
    omega
  · intro r _ r' _ hne
    refine ⟨((), fun s t => if s = r then byte (-A r t) else byte (1 - A s t)), fun s t => ?_, 0, ?_⟩
    · show 0 ≤ (if s = r then byte (-A r t) else byte (1 - A s t)) ∧
        (if s = r then byte (-A r t) else byte (1 - A s t)) < 256
      split_ifs <;> simp only [byte] <;> omega
    · show byte (A r 0 + if r = r then byte (-A r 0) else byte (1 - A r 0)) ≠
        byte (A r' 0 + if r' = r then byte (-A r 0) else byte (1 - A r' 0))
      rw [if_pos rfl, if_neg (Ne.symm hne)]
      simp only [byte]
      omega
  · intro r hr g hg
    refine (reg_eq_iff _ _ _ _).2 fun t => ?_
    show byte (A r t + g.2 r t) = byte (g.2 r t)
    have := hr t
    have hb := hg r t
    simp only [byte] at this ⊢
    omega

end RawDither

/-! ## The final word: a call's grid, with the call's own words free -/

section FinalWord

variable {m n k : ℕ}

/-- A call's grid: its input `A` (data) and its noise; the weights `B` are fixed (a weights slice). -/
abbrev Call (m n k : ℕ) : Type := Mat × Noise 1 m n k

/-- The int7 inputs, with every noise. -/
def int7 (m n k : ℕ) : Set (Call m n k) := {g | ∀ i l, -63 ≤ g.1 i l ∧ g.1 i l ≤ 63}

/-- The noise at `0`: every coin `31`, so every noise entry `0`. -/
def ω₀ : Noise 1 m n k := (fun _ _ _ => 31, fun _ _ => 31)

/-- The input `A` with all entries `0`. -/
def zeroA : Mat := fun _ _ => 0

/-- The data step: the same noise, any two inputs. -/
def DataStep : Call m n k → Call m n k → Prop := fun g g' => g.2 = g'.2

/-- The noise step: the same input, any two noises. -/
def NoiseStep : Call m n k → Call m n k → Prop := fun g g' => g.1 = g'.1

/-- **The call's own words**: the constants and the words `W`. -/
def FreeCall (W : Set (Call m n k → ℚ)) : Set (Call m n k → ℚ) := {f | (∃ q : ℚ, f = fun _ => q) ∨ f ∈ W}

theorem freeCall_const (W : Set (Call m n k → ℚ)) (q : ℚ) : (fun _ : Call m n k => q) ∈ FreeCall W :=
  Or.inl ⟨q, rfl⟩

/-- The call's input words: four consecutive entries of a row of `A`, packed. -/
def inWord (i c : ℕ) : Call m n k → ℚ := reg fun g t => g.1 i (4 * c + t)

/-- The set of the call's input words. -/
def inWords (m n k : ℕ) : Set (Call m n k → ℚ) := {f | ∃ i c, f = inWord i c}

/-- A non-final checked word on the call's grid. -/
def nfWord (R B : Mat) (p : ℕ) : Call m n k → ℚ := fun g => (checked k m n 1 R (fun _ => g.1) (fun _ => B) g.2 0 p : ℚ)

/-- The final word of dot product `(i, j)`: `Σ_{l < k} A_il·B_lj`, a function of the input alone. -/
def finalWord (B : Mat) (i j : ℕ) : Call m n k → ℚ := fun g => (mm k g.1 B i j : ℚ)

/-- **The final word is honest's last checked value**: at depth `3k` the stacked product is `A·B` (NCP's
cancellation, `Pouw.NCP.Proofs.stacked`). -/
theorem finalWord_eq_ck (R B : Mat) (i j : ℕ) (g : Call m n k) :
    finalWord (m := m) (n := n) (k := k) B i j g =
      (ck k g.1 B (e1Of g.2 0) (f1Of g.2) R i j (3 * k) : ℚ) := by
  have e := congrFun (congrFun (Pouw.NCP.Proofs.stacked k g.1 B (e1Of g.2 0) (f1Of g.2) R).2 i) j
  simp only [finalWord, ck]
  rw [← e]
  rfl

/-- The slice at input `0`: a function of the noise alone. -/
def slice0 (f : Call m n k → ℚ) : Noise 1 m n k → ℚ := fun ω => f (zeroA, ω)

theorem zero_int7 (ω : Noise 1 m n k) : (zeroA, ω) ∈ int7 m n k := fun _ _ => by simp [zeroA]

theorem unitAt_int7 (r c : ℕ) (ω : Noise 1 m n k) : (unitAt r c, ω) ∈ int7 m n k := by
  intro i l
  simp only [unitAt]
  split_ifs <;> omega

theorem mm_unitAt (B : Mat) (r c i j : ℕ) : mm k (unitAt r c) B i j = if i = r ∧ c < k then B c j else 0 := by
  simp only [mm, unitAt]
  by_cases hi : i = r
  · subst hi
    by_cases hc : c < k
    · rw [Finset.sum_eq_single c (fun l _ hl => by simp [hl]) (fun h => absurd (Finset.mem_range.2 hc) h)]
      simp [hc]
    · rw [Finset.sum_eq_zero fun l hl => ?_]
      · simp [hc]
      · have : l ≠ c := fun h => hc (h ▸ Finset.mem_range.1 hl)
        simp [this]
  · rw [Finset.sum_eq_zero fun l _ => by simp [hi]]
    simp [hi]

/-- The non-final words at the input-`0` slice are Theorem D's words with `A = 0`. -/
theorem slice0_nfWord (R B : Mat) (p : ℕ) :
    slice0 (nfWord (m := m) (n := n) (k := k) R B p) = word k m n 1 R (fun _ _ => zeroA) (fun _ => B) 0 p := rfl

variable {R : Mat}

/-- The non-final words at positions `P` of the one unit, at the input-`0` slice, are independent modulo `V₀`. -/
theorem indep_slice (hind : CheckpointIndependent k 16 R) (B : Mat) (P : Finset ℕ)
    (hP : P ⊆ range (nonFinal k m n)) :
    IndepMod (V0 1 m n k) (fun x : ↥(({0} : Finset ℕ) ×ˢ P) =>
      word k m n 1 R (fun _ _ => zeroA) (fun _ => B) x.1.1 x.1.2) :=
  indep_words_sub (fun _ _ => zeroA) (fun _ => B) hind (by simp)
    (Finset.product_subset_product (subset_refl _) hP)

theorem mem_V0_of_const (q : ℚ) : (fun _ : Noise 1 m n k => q) ∈ V0 1 m n k := by
  have : (fun _ : Noise 1 m n k => q) = q • (1 : Noise 1 m n k → ℚ) := by
    funext ω
    simp
  rw [this]
  exact Submodule.smul_mem _ q one_V0

/-- Independent modulo `V` makes the family injective. -/
theorem indep_ne {G ι : Type} [Fintype ι] [DecidableEq ι] {V : Submodule ℚ (G → ℚ)} {t : ι → G → ℚ}
    (h : IndepMod V t) {i j : ι} (hij : i ≠ j) : t i ≠ t j := by
  intro heq
  have hs : ∑ x, ((if x = i then (1 : ℚ) else 0) - (if x = j then 1 else 0)) • t x = 0 := by
    simp [sub_smul, ite_smul, Finset.sum_sub_distrib, heq]
  have := h _ (by rw [hs]; exact Submodule.zero_mem _) i
  simp [hij] at this

/-- The non-final words of `P`, at the input-`0` slice, are not in `V₀`. -/
theorem nf_slice_nmem (hind : CheckpointIndependent k 16 R) (B : Mat) {P : Finset ℕ}
    (hP : P ⊆ range (nonFinal k m n)) {p : ℕ} (hp : p ∈ P) : slice0 (nfWord (m := m) (n := n) (k := k) R B p) ∉ V0 1 m n k := by
  have := nmem_of_indep (indep_slice (m := m) (n := n) hind B P hP) ⟨(0, p), by simp [hp]⟩
  simpa [slice0_nfWord] using this

/-- A non-final word differs on `int7` from every function whose input-`0` slice is in `V₀`. -/
theorem nf_ne_of_slice_mem (hind : CheckpointIndependent k 16 R) (B : Mat) {P : Finset ℕ}
    (hP : P ⊆ range (nonFinal k m n)) {p : ℕ} (hp : p ∈ P) {f : Call m n k → ℚ} (hf : slice0 f ∈ V0 1 m n k) :
    ¬ ∀ g ∈ int7 m n k, nfWord R B p g = f g := by
  intro heq
  refine nf_slice_nmem hind B hP hp ?_
  have : slice0 (nfWord (m := m) (n := n) (k := k) R B p) = slice0 f := by
    funext ω
    exact heq (zeroA, ω) (zero_int7 ω)
  rwa [this]

/-- Two non-final words at different positions differ on `int7`. -/
theorem nf_ne_nf (hind : CheckpointIndependent k 16 R) (B : Mat) {P : Finset ℕ} (hP : P ⊆ range (nonFinal k m n))
    {p p' : ℕ} (hp : p ∈ P) (hp' : p' ∈ P) (hne : p ≠ p') :
    ¬ ∀ g ∈ int7 m n k, nfWord R B p g = nfWord R B p' g := by
  classical
  intro heq
  have hI := indep_slice (m := m) (n := n) hind B P hP
  have hij : (⟨(0, p), by simp [hp]⟩ : ↥(({0} : Finset ℕ) ×ˢ P)) ≠ ⟨(0, p'), by simp [hp']⟩ := by
    intro h
    exact hne (congrArg (fun x : ↥(({0} : Finset ℕ) ×ˢ P) => x.1.2) h)
  refine indep_ne hI hij ?_
  funext ω
  exact heq (zeroA, ω) (zero_int7 ω)

/-- A non-final word moves with the noise at input `0`. -/
theorem nf_moves (hind : CheckpointIndependent k 16 R) (B : Mat) {P : Finset ℕ} (hP : P ⊆ range (nonFinal k m n))
    {p : ℕ} (hp : p ∈ P) : Moves (int7 m n k) NoiseStep (nfWord (m := m) (n := n) (k := k) R B p) := by
  by_contra hno
  refine nf_slice_nmem hind B hP hp ?_
  have : slice0 (nfWord (m := m) (n := n) (k := k) R B p) =
      fun _ : Noise 1 m n k => nfWord (m := m) (n := n) (k := k) R B p (zeroA, ω₀) := by
    funext ω
    by_contra hne
    exact hno ⟨(zeroA, ω), zero_int7 ω, (zeroA, ω₀), zero_int7 ω₀, rfl, hne⟩
  rw [this]
  exact mem_V0_of_const _

/-! ### The final words -/

theorem final_ignores_noise (B : Mat) (i j : ℕ) : Ignores (int7 m n k) NoiseStep (finalWord (m := m) (n := n) (k := k) B i j) := by
  rintro ⟨a, ω⟩ _ ⟨a', ω'⟩ _ (h : a = a')
  subst h
  rfl

theorem final_at_zero (B : Mat) (i j : ℕ) (ω : Noise 1 m n k) : finalWord (m := m) (n := n) (k := k) B i j (zeroA, ω) = 0 := by
  simp [finalWord, mm, zeroA]

theorem final_at_unit (B : Mat) (i j r c : ℕ) (ω : Noise 1 m n k) :
    finalWord (m := m) (n := n) (k := k) B i j (unitAt r c, ω) = if i = r ∧ c < k then (B c j : ℚ) else 0 := by
  simp only [finalWord, mm_unitAt]
  split_ifs <;> simp

/-- A final word with a nonzero column moves with the input. -/
theorem final_moves_data (B : Mat) (i j : ℕ) {l : ℕ} (hl : l < k) (hB : B l j ≠ 0) :
    Moves (int7 m n k) DataStep (finalWord (m := m) (n := n) (k := k) B i j) := by
  refine ⟨(zeroA, ω₀), zero_int7 _, (unitAt i l, ω₀), unitAt_int7 _ _ _, rfl, ?_⟩
  rw [final_at_zero, final_at_unit, if_pos ⟨rfl, hl⟩]
  exact fun h => hB (by exact_mod_cast h.symm)

theorem not_eq_symm {G : Type} {S : Set G} {f h : G → ℚ} (hne : ¬ ∀ g ∈ S, f g = h g) : ¬ ∀ g ∈ S, h g = f g :=
  fun heq => hne fun g hg => (heq g hg).symm

/-- Two final words of different dot products differ on `int7` when `B`'s columns are nonzero and pairwise distinct. -/
theorem final_ne_final (B : Mat) {i j i' j' : ℕ} (hj0 : ∃ l < k, B l j ≠ 0)
    (hjj : j ≠ j' → ∃ l < k, B l j ≠ B l j') (hne : (i, j) ≠ (i', j')) :
    ¬ ∀ g ∈ int7 m n k, finalWord (m := m) (n := n) (k := k) B i j g = finalWord B i' j' g := by
  by_cases hi : i = i'
  · subst hi
    have hj : j ≠ j' := fun h => hne (by rw [h])
    obtain ⟨l, hl, hB⟩ := hjj hj
    refine differ_of_point (unitAt_int7 (m := m) (n := n) (k := k) i l ω₀) ?_
    rw [final_at_unit, final_at_unit, if_pos ⟨rfl, hl⟩, if_pos ⟨rfl, hl⟩]
    exact fun h => hB (by exact_mod_cast h)
  · obtain ⟨l, hl, hB⟩ := hj0
    refine differ_of_point (unitAt_int7 (m := m) (n := n) (k := k) i l ω₀) ?_
    rw [final_at_unit, final_at_unit, if_pos ⟨rfl, hl⟩, if_neg (fun h => hi h.1.symm)]
    exact fun h => hB (by exact_mod_cast h)

theorem inWord_at_unit (i c : ℕ) (ω : Noise 1 m n k) : inWord (m := m) (n := n) (k := k) i c (unitAt i (4 * c + 1), ω) = 256 := by
  simp [inWord, reg, pack4, byte, unitAt]

theorem inWord_at_zero (i c : ℕ) (ω : Noise 1 m n k) : inWord (m := m) (n := n) (k := k) i c (zeroA, ω) = 0 := by
  simp [inWord, reg, pack4, byte, zeroA]

/-- A final word with a nonzero int8 column is no input word: at the input with one `1`, in byte 1 of the word's
four entries, the word reads 256 and the final word an entry of `B`. -/
theorem final_ne_inWord (B : Mat) (hB : ∀ l j, -255 ≤ B l j ∧ B l j ≤ 255) (i j i' c : ℕ) :
    ¬ ∀ g ∈ int7 m n k, finalWord (m := m) (n := n) (k := k) B i j g = inWord i' c g := by
  refine differ_of_point (unitAt_int7 (m := m) (n := n) (k := k) i' (4 * c + 1) ω₀) ?_
  rw [final_at_unit, inWord_at_unit]
  split_ifs
  · have := hB (4 * c + 1) j
    intro h
    have h' : B (4 * c + 1) j = 256 := by exact_mod_cast h
    omega
  · norm_num

/-- A final word with a nonzero column is no constant. -/
theorem final_ne_const (B : Mat) (i j : ℕ) (hj0 : ∃ l < k, B l j ≠ 0) (q : ℚ) :
    ¬ ∀ g ∈ int7 m n k, finalWord (m := m) (n := n) (k := k) B i j g = q := by
  obtain ⟨l, hl, hB⟩ := hj0
  exact differ_of_moves (final_moves_data B i j hl hB) (ignores_const _ _ q)

/-- A final word differs from every function that ignores the input or moves with the noise. -/
theorem final_ne_of_dichotomy (B : Mat) (i j : ℕ) (hj0 : ∃ l < k, B l j ≠ 0) {w : Call m n k → ℚ}
    (hw : Ignores (int7 m n k) DataStep w ∨ Moves (int7 m n k) NoiseStep w) :
    ¬ ∀ g ∈ int7 m n k, finalWord (m := m) (n := n) (k := k) B i j g = w g := by
  obtain ⟨l, hl, hB⟩ := hj0
  rcases hw with hw | hw
  · exact differ_of_moves (final_moves_data B i j hl hB) hw
  · exact not_eq_symm (differ_of_moves hw (final_ignores_noise B i j))

/-! ### The lemma -/

/-- **The final-word lemma for NCP**: on a call's grid (int7 input, every noise, the weights `B` fixed), with the
free values the constants, the call's input words and noise-side words `W'` (each in `V₀` at the input-`0` slice, and
each ignoring the input or moving with the noise: coin words qualify, and so do the formed words in the biased
packing), the non-final checked words at positions `P` and the final words `A_i·B_j` for `i ∈ Ix`,
`j ∈ Jx` are distinct and none is free, when `B` is int8 and its columns in `Jx` are nonzero and pairwise distinct.
There are `|P| + |Ix|·|Jx|` of them. -/
theorem finalWordTarget [DecidableEq (Call m n k → ℚ)] (hind : CheckpointIndependent k 16 R) (B : Mat)
    (hB : ∀ l j, -255 ≤ B l j ∧ B l j ≤ 255) (W' : Set (Call m n k → ℚ))
    (hW0 : ∀ w ∈ W', slice0 w ∈ V0 1 m n k)
    (hW1 : ∀ w ∈ W', Ignores (int7 m n k) DataStep w ∨ Moves (int7 m n k) NoiseStep w)
    (P : Finset ℕ) (hP : P ⊆ range (nonFinal k m n)) (Ix Jx : Finset ℕ)
    (hJ0 : ∀ j ∈ Jx, ∃ l < k, B l j ≠ 0) (hJ : ∀ j ∈ Jx, ∀ j' ∈ Jx, j ≠ j' → ∃ l < k, B l j ≠ B l j') :
    DistinctOn (int7 m n k) (FreeCall (inWords m n k ∪ W'))
        (P.image (nfWord R B) ∪ (Ix ×ˢ Jx).image fun x => finalWord B x.1 x.2) ∧
      (P.image (nfWord (m := m) (n := n) (k := k) R B) ∪
        (Ix ×ˢ Jx).image fun x => finalWord (m := m) (n := n) (k := k) B x.1 x.2).card =
        P.card + Ix.card * Jx.card := by
  have hnfnf : ∀ p ∈ P, ∀ p' ∈ P, p ≠ p' → ¬ ∀ g ∈ int7 m n k, nfWord R B p g = nfWord R B p' g :=
    fun p hp p' hp' hne => nf_ne_nf hind B hP hp hp' hne
  have hff : ∀ x ∈ Ix ×ˢ Jx, ∀ y ∈ Ix ×ˢ Jx, x ≠ y →
      ¬ ∀ g ∈ int7 m n k, finalWord (m := m) (n := n) (k := k) B x.1 x.2 g = finalWord B y.1 y.2 g := by
    rintro ⟨i, j⟩ hx ⟨i', j'⟩ hy hne
    obtain ⟨-, hj⟩ := Finset.mem_product.1 hx
    obtain ⟨-, hj'⟩ := Finset.mem_product.1 hy
    exact final_ne_final B (hJ0 j hj) (hJ j hj j' hj') hne
  have hcross : ∀ t ∈ P.image (nfWord (m := m) (n := n) (k := k) R B),
      ∀ t' ∈ (Ix ×ˢ Jx).image fun x => finalWord (m := m) (n := n) (k := k) B x.1 x.2,
      ¬ ∀ g ∈ int7 m n k, t g = t' g := by
    intro t ht t' ht'
    obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 ht
    obtain ⟨x, -, rfl⟩ := Finset.mem_image.1 ht'
    exact differ_of_moves (nf_moves hind B hP hp) (final_ignores_noise B x.1 x.2)
  have hnf : DistinctOn (int7 m n k) (FreeCall (inWords m n k ∪ W'))
      (P.image (nfWord (m := m) (n := n) (k := k) R B)) := by
    refine ⟨fun t ht t' ht' h => ?_, fun t ht f hf => ?_⟩
    · obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 ht
      obtain ⟨p', hp', rfl⟩ := Finset.mem_image.1 ht'
      by_cases hpp : p = p'
      · rw [hpp]
      · exact absurd h (hnfnf p hp p' hp' hpp)
    · obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 ht
      refine nf_ne_of_slice_mem hind B hP hp ?_
      rcases hf with ⟨q, rfl⟩ | hf | hf
      · exact mem_V0_of_const q
      · obtain ⟨i, c, rfl⟩ := hf
        have : slice0 (inWord (m := m) (n := n) (k := k) i c) = fun _ => 0 := by
          funext ω
          exact inWord_at_zero i c ω
        rw [this]
        exact mem_V0_of_const 0
      · exact hW0 f hf
  have hfin : DistinctOn (int7 m n k) (FreeCall (inWords m n k ∪ W'))
      ((Ix ×ˢ Jx).image fun x => finalWord (m := m) (n := n) (k := k) B x.1 x.2) := by
    refine ⟨fun t ht t' ht' h => ?_, fun t ht f hf => ?_⟩
    · obtain ⟨x, hx, rfl⟩ := Finset.mem_image.1 ht
      obtain ⟨y, hy, rfl⟩ := Finset.mem_image.1 ht'
      by_cases hxy : x = y
      · rw [hxy]
      · exact absurd h (hff x hx y hy hxy)
    · obtain ⟨⟨i, j⟩, hx, rfl⟩ := Finset.mem_image.1 ht
      have hj0 := hJ0 j (Finset.mem_product.1 hx).2
      rcases hf with ⟨q, rfl⟩ | hf | hf
      · exact final_ne_const B i j hj0 q
      · obtain ⟨i', c, rfl⟩ := hf
        exact final_ne_inWord B hB i j i' c
      · exact final_ne_of_dichotomy B i j hj0 (hW1 f hf)
  refine ⟨distinctOn_union hnf hfin hcross, ?_⟩
  rw [card_union_of_cross hcross, card_image_of_differ P _ hnfnf,
    card_image_of_differ (Ix ×ˢ Jx) (fun x => finalWord (m := m) (n := n) (k := k) B x.1 x.2) hff,
    Finset.card_product]

/-- **The final word is charged** (NCP, at the H200's measured table): every well-formed `MH100w` program on the
call's free words that outputs, on every int7 input and every noise, the non-final words at `P` and the final words of
`Ix × Jx`, costs at least `32·(|P| + |Ix|·|Jx|)`. With `P` all of honest's non-final k32 words, that is 32 for every
one of honest's words, the final word included, so the final word's `32/K` term closes. -/
theorem finalWord_charged [DecidableEq (Call m n k → ℚ)] (hind : CheckpointIndependent k 16 R) (B : Mat)
    (hB : ∀ l j, -255 ≤ B l j ∧ B l j ≤ 255) (W' : Set (Call m n k → ℚ))
    (hW0 : ∀ w ∈ W', slice0 w ∈ V0 1 m n k)
    (hW1 : ∀ w ∈ W', Ignores (int7 m n k) DataStep w ∨ Moves (int7 m n k) NoiseStep w)
    (P : Finset ℕ) (hP : P ⊆ range (nonFinal k m n)) (Ix Jx : Finset ℕ)
    (hJ0 : ∀ j ∈ Jx, ∃ l < k, B l j ≠ 0) (hJ : ∀ j ∈ Jx, ∀ j' ∈ Jx, j ≠ j' → ∃ l < k, B l j ≠ B l j')
    (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (Call m n k → ℚ)) (Pg : List (Op100 (Call m n k)))
    (hL : ∀ f ∈ L, f ∈ FreeCall (inWords m n k ∪ W')) (hPg : ∀ o ∈ Pg, o.WFw h100Floor (FreeCall (inWords m n k ∪ W')))
    (hO : OutputsOn fpSem (int7 m n k) Pg L (P.image (nfWord R B) ∪ (Ix ×ˢ Jx).image fun x => finalWord B x.1 x.2)) :
    32 * (P.card + Ix.card * Jx.card) ≤ cost100 h100Floor Pg := by
  obtain ⟨hD, hc⟩ := finalWordTarget hind B hB W' hW0 hW1 P hP Ix Jx hJ0 hJ
  rw [← hc]
  exact charged _ (freeCall_const _) _ fpSem L Pg _ hL hPg hD hO

/-! ### Candidate 1 and the final word together -/

theorem inWord_ignores_noise (i c : ℕ) : Ignores (int7 m n k) NoiseStep (inWord (m := m) (n := n) (k := k) i c) := by
  rintro ⟨a, ω⟩ _ ⟨a', ω'⟩ _ (h : a = a')
  subst h
  rfl

/-- An operand register that moves with the noise is neither a constant nor an input word. -/
theorem op_ne_const_inWord {w : Call m n k → ℚ} (hw : Moves (int7 m n k) NoiseStep w) (f : Call m n k → ℚ)
    (hf : (∃ q : ℚ, f = fun _ => q) ∨ f ∈ inWords m n k) : ¬ ∀ g ∈ int7 m n k, w g = f g := by
  rcases hf with ⟨q, rfl⟩ | ⟨i, c, rfl⟩
  · exact differ_of_moves hw (ignores_const _ _ q)
  · exact differ_of_moves hw (inWord_ignores_noise i c)

/-- **Candidate 1 and the final-word lemma together**: with the free values only the constants, the call's input words
and the coin words `W'`, the committed operand registers `Op` are targets beside the checked words: each in `V₀` at
the input-`0` slice, moving with the noise, equal to no coin word, and pairwise different on `int7`. Then the
non-final words at `P`, the final words of `Ix × Jx` and the registers of `Op` are distinct and none is free, so
there are `|P| + |Ix|·|Jx| + |Op|` of them. -/
theorem allTarget [DecidableEq (Call m n k → ℚ)] (hind : CheckpointIndependent k 16 R) (B : Mat)
    (hB : ∀ l j, -255 ≤ B l j ∧ B l j ≤ 255) (W' : Set (Call m n k → ℚ))
    (hW0 : ∀ w ∈ W', slice0 w ∈ V0 1 m n k)
    (hW1 : ∀ w ∈ W', Ignores (int7 m n k) DataStep w ∨ Moves (int7 m n k) NoiseStep w)
    (P : Finset ℕ) (hP : P ⊆ range (nonFinal k m n)) (Ix Jx : Finset ℕ)
    (hJ0 : ∀ j ∈ Jx, ∃ l < k, B l j ≠ 0) (hJ : ∀ j ∈ Jx, ∀ j' ∈ Jx, j ≠ j' → ∃ l < k, B l j ≠ B l j')
    (Op : Finset (Call m n k → ℚ)) (hop0 : ∀ w ∈ Op, slice0 w ∈ V0 1 m n k)
    (hopN : ∀ w ∈ Op, Moves (int7 m n k) NoiseStep w) (hopW : ∀ w ∈ Op, ∀ c ∈ W', ¬ ∀ g ∈ int7 m n k, w g = c g)
    (hopd : ∀ w ∈ Op, ∀ w' ∈ Op, (∀ g ∈ int7 m n k, w g = w' g) → w = w') :
    DistinctOn (int7 m n k) (FreeCall (inWords m n k ∪ W'))
        ((P.image (nfWord R B) ∪ (Ix ×ˢ Jx).image fun x => finalWord B x.1 x.2) ∪ Op) ∧
      ((P.image (nfWord (m := m) (n := n) (k := k) R B) ∪
        (Ix ×ˢ Jx).image fun x => finalWord (m := m) (n := n) (k := k) B x.1 x.2) ∪ Op).card =
        P.card + Ix.card * Jx.card + Op.card := by
  obtain ⟨hD, hc⟩ := finalWordTarget hind B hB W' hW0 hW1 P hP Ix Jx hJ0 hJ
  have hcross : ∀ t ∈ P.image (nfWord (m := m) (n := n) (k := k) R B) ∪
      (Ix ×ˢ Jx).image (fun x => finalWord (m := m) (n := n) (k := k) B x.1 x.2),
      ∀ t' ∈ Op, ¬ ∀ g ∈ int7 m n k, t g = t' g := by
    intro t ht w hw
    rcases Finset.mem_union.1 ht with ht | ht
    · obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 ht
      exact nf_ne_of_slice_mem hind B hP hp (hop0 w hw)
    · obtain ⟨⟨i, j⟩, hx, rfl⟩ := Finset.mem_image.1 ht
      exact final_ne_of_dichotomy B i j (hJ0 j (Finset.mem_product.1 hx).2) (Or.inr (hopN w hw))
  have hop : DistinctOn (int7 m n k) (FreeCall (inWords m n k ∪ W')) Op := by
    refine ⟨hopd, fun w hw f hf => ?_⟩
    rcases hf with hf | hf | hf
    · exact op_ne_const_inWord (hopN w hw) f (Or.inl hf)
    · exact op_ne_const_inWord (hopN w hw) f (Or.inr hf)
    · exact hopW w hw f hf
  exact ⟨distinctOn_union hD hop hcross, by rw [card_union_of_cross hcross, hc]⟩

/-- **Everything charged** (at the H200's measured table): a well-formed `MH100w` program on the constants, the call's
input words and coin words that outputs the non-final words at `P`, the final words of `Ix × Jx` and the committed
operand registers `Op` costs at least `32·(|P| + |Ix|·|Jx| + |Op|)`. -/
theorem all_charged [DecidableEq (Call m n k → ℚ)] (hind : CheckpointIndependent k 16 R) (B : Mat)
    (hB : ∀ l j, -255 ≤ B l j ∧ B l j ≤ 255) (W' : Set (Call m n k → ℚ))
    (hW0 : ∀ w ∈ W', slice0 w ∈ V0 1 m n k)
    (hW1 : ∀ w ∈ W', Ignores (int7 m n k) DataStep w ∨ Moves (int7 m n k) NoiseStep w)
    (P : Finset ℕ) (hP : P ⊆ range (nonFinal k m n)) (Ix Jx : Finset ℕ)
    (hJ0 : ∀ j ∈ Jx, ∃ l < k, B l j ≠ 0) (hJ : ∀ j ∈ Jx, ∀ j' ∈ Jx, j ≠ j' → ∃ l < k, B l j ≠ B l j')
    (Op : Finset (Call m n k → ℚ)) (hop0 : ∀ w ∈ Op, slice0 w ∈ V0 1 m n k)
    (hopN : ∀ w ∈ Op, Moves (int7 m n k) NoiseStep w) (hopW : ∀ w ∈ Op, ∀ c ∈ W', ¬ ∀ g ∈ int7 m n k, w g = c g)
    (hopd : ∀ w ∈ Op, ∀ w' ∈ Op, (∀ g ∈ int7 m n k, w g = w' g) → w = w')
    (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (Call m n k → ℚ)) (Pg : List (Op100 (Call m n k)))
    (hL : ∀ f ∈ L, f ∈ FreeCall (inWords m n k ∪ W')) (hPg : ∀ o ∈ Pg, o.WFw h100Floor (FreeCall (inWords m n k ∪ W')))
    (hO : OutputsOn fpSem (int7 m n k) Pg L
      ((P.image (nfWord R B) ∪ (Ix ×ˢ Jx).image fun x => finalWord B x.1 x.2) ∪ Op)) :
    32 * (P.card + Ix.card * Jx.card + Op.card) ≤ cost100 h100Floor Pg := by
  obtain ⟨hD, hc⟩ := allTarget hind B hB W' hW0 hW1 P hP Ix Jx hJ0 hJ Op hop0 hopN hopW hopd
  rw [← hc]
  exact charged _ (freeCall_const _) _ fpSem L Pg _ hL hPg hD hO

/-! ### Concrete registers: the coin words, and block 1 of `X` and `Y` -/

/-- Every coin `33`, so every noise entry `1`. -/
def ω₁ : Noise 1 m n k := (fun _ _ _ => 32, fun _ _ => 32)

/-- `F₁` is `1` at `(l, j)` and `0` elsewhere. -/
def ωF (l j : ℕ) : Noise 1 m n k := (fun _ _ _ => 31, fun l' j' => if (l' : ℕ) = l ∧ (j' : ℕ) = j then 32 else 31)

theorem e1_ω₀ (u i l : ℕ) : e1Of (ω₀ (m := m) (n := n) (k := k)) u i l = 0 := by
  simp only [e1Of, ω₀]
  split_ifs <;> rfl

theorem f1_ω₀ (l j : ℕ) : f1Of (ω₀ (m := m) (n := n) (k := k)) l j = 0 := by
  simp only [f1Of, ω₀]
  split_ifs <;> rfl

theorem e1_ω₁ {i l : ℕ} (hi : i < m) (hl : l < k) : e1Of (ω₁ (m := m) (n := n) (k := k)) 0 i l = 1 := by
  simp only [e1Of, ω₁, dif_pos (show 0 < 1 ∧ i < m ∧ l < k from ⟨Nat.one_pos, hi, hl⟩)]
  rfl

theorem f1_ωF (l j l' j' : ℕ) :
    f1Of (ωF (m := m) (n := n) (k := k) l j) l' j' = if l' < k ∧ j' < n ∧ l' = l ∧ j' = j then 1 else 0 := by
  simp only [f1Of, ωF]
  by_cases h : l' < k ∧ j' < n
  · rw [dif_pos h]
    by_cases h' : l' = l ∧ j' = j
    · rw [if_pos h', if_pos ⟨h.1, h.2, h'⟩]
      rfl
    · rw [if_neg h', if_neg fun h'' => h' ⟨h''.2.2.1, h''.2.2.2⟩]
      rfl
  · rw [dif_neg h, if_neg fun h'' => h ⟨h''.1, h''.2.1⟩]

theorem e1_ωF (l j u i l' : ℕ) : e1Of (ωF (m := m) (n := n) (k := k) l j) u i l' = 0 := by
  simp only [e1Of, ωF]
  split_ifs <;> rfl

theorem e1_range (ω : Noise 1 m n k) (u i l : ℕ) : -31 ≤ e1Of ω u i l ∧ e1Of ω u i l ≤ 32 := by
  simp only [e1Of]
  split_ifs with h
  · have := (ω.1 ⟨u, h.1⟩ ⟨i, h.2.1⟩ ⟨l, h.2.2⟩).isLt
    omega
  · omega

theorem byte_of_range {x : ℤ} (h0 : 0 ≤ x) (h1 : x < 256) : byte x = x := Int.emod_eq_of_lt h0 h1

/-- A register whose bytes are in `[0, 255]` is the plain base-256 sum. -/
theorem reg_of_range {G : Type} (X : G → Fin 4 → ℤ) (g : G) (h : ∀ t, 0 ≤ X g t ∧ X g t < 256) :
    reg X g = (X g 0 : ℚ) + 256 * X g 1 + 65536 * X g 2 + 16777216 * X g 3 := by
  simp only [reg, pack4, byte_of_range (h 0).1 (h 0).2, byte_of_range (h 1).1 (h 1).2, byte_of_range (h 2).1 (h 2).2,
    byte_of_range (h 3).1 (h 3).2]
  push_cast
  ring

/-- `E₁[i, l]` as a function of the noise is in `V₀`. -/
theorem e1_mem_V0 (i l : ℕ) : (fun ω : Noise 1 m n k => (e1Of ω 0 i l : ℚ)) ∈ V0 1 m n k := by
  by_cases h : 0 < 1 ∧ i < m ∧ l < k
  · refine Submodule.mem_sup_left (Submodule.subset_span (Set.mem_insert_of_mem _ ?_))
    show _ ∈ coords 1 m n k
    refine List.mem_append_left _ (List.mem_flatMap.2 ⟨0, List.mem_finRange _, List.mem_flatMap.2 ⟨⟨i, h.2.1⟩,
      List.mem_finRange _, List.mem_map.2 ⟨⟨l, h.2.2⟩, List.mem_finRange _, ?_⟩⟩⟩)
    funext ω
    simp only [e1Of, dif_pos h]
    push_cast
    rfl
  · have : (fun ω : Noise 1 m n k => (e1Of ω 0 i l : ℚ)) = 0 := by
      funext ω
      simp only [e1Of, dif_neg h]
      rfl
    rw [this]
    exact Submodule.zero_mem _

/-- A register whose input-`0` slice has bytes in `[0, 255]`, each `E₁[i, l_t]` plus a constant, is in `V₀` there. -/
theorem reg_e1_mem_V0 (X : Call m n k → Fin 4 → ℤ) (i : ℕ) (l : Fin 4 → ℕ) (q : Fin 4 → ℤ)
    (hX : ∀ ω t, X (zeroA, ω) t = e1Of ω 0 i (l t) + q t) (hr : ∀ ω t, 0 ≤ X (zeroA, ω) t ∧ X (zeroA, ω) t < 256) :
    slice0 (reg X) ∈ V0 1 m n k := by
  have hf : slice0 (reg X) =
      (fun ω : Noise 1 m n k => (e1Of ω 0 i (l 0) : ℚ)) + (256 : ℚ) • (fun ω => (e1Of ω 0 i (l 1) : ℚ)) +
        (65536 : ℚ) • (fun ω => (e1Of ω 0 i (l 2) : ℚ)) + (16777216 : ℚ) • (fun ω => (e1Of ω 0 i (l 3) : ℚ)) +
        fun _ => ((q 0 + 256 * q 1 + 65536 * q 2 + 16777216 * q 3 : ℤ) : ℚ) := by
    funext ω
    simp only [slice0, reg_of_range X _ (hr ω), hX ω, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    push_cast
    ring
  rw [hf]
  refine Submodule.add_mem _ (Submodule.add_mem _ (Submodule.add_mem _ (Submodule.add_mem _ (e1_mem_V0 _ _)
    (Submodule.smul_mem _ _ (e1_mem_V0 _ _))) (Submodule.smul_mem _ _ (e1_mem_V0 _ _)))
    (Submodule.smul_mem _ _ (e1_mem_V0 _ _))) (mem_V0_of_const _)

/-- A function of the call that reads the noise only through `F₁` has its input-`0` slice in `fnF ⊆ V₀`. -/
theorem slice0_mem_V0_of_f1 (w : Call m n k → ℚ) (s : Mat → ℚ) (hw : ∀ g, w g = s (f1Of g.2)) :
    slice0 w ∈ V0 1 m n k := by
  refine Submodule.mem_sup_right (Submodule.subset_span fun ω ω' h => ?_)
  have : f1Of ω = f1Of ω' := by
    funext l j
    simp only [f1Of, h]
  simp only [slice0, hw, this]

/-- The `E₁` coin word `(i, c)`: the coins of `E₁[i, 4c..4c+3]`, each `e + 31 ∈ [0, 63]`, packed. -/
def coinE (i c : ℕ) : Call m n k → ℚ := reg fun g t => e1Of g.2 0 i (4 * c + t) + 31

/-- The `F₁` coin word `(j, c)`: the coins of `F₁[4c..4c+3, j]`, packed. -/
def coinF (j c : ℕ) : Call m n k → ℚ := reg fun g t => f1Of g.2 (4 * c + t) j + 31

/-- The call's coin words. -/
def coinWords (m n k : ℕ) : Set (Call m n k → ℚ) :=
  {f | (∃ i c, f = coinE (m := m) (n := n) (k := k) i c) ∨ ∃ j c, f = coinF (m := m) (n := n) (k := k) j c}

/-- Block 1 of `X` in the biased packing: bytes `A[i, 4c+t] + E₁[i, 4c+t] + 128`. -/
def xWord (i c : ℕ) : Call m n k → ℚ := reg fun g t => g.1 i (4 * c + t) + e1Of g.2 0 i (4 * c + t) + 128

/-- Block 1 of `Y`: bytes `B[4c+t, j] + F₁[4c+t, j]`, in two's complement. -/
def yWord (B : Mat) (j c : ℕ) : Call m n k → ℚ := reg fun g t => B (4 * c + t) j + f1Of g.2 (4 * c + t) j

theorem coinWords_V0 : ∀ w ∈ coinWords m n k, slice0 w ∈ V0 1 m n k := by
  rintro w (⟨i, c, rfl⟩ | ⟨j, c, rfl⟩)
  · refine reg_e1_mem_V0 _ i (fun t => 4 * c + t) (fun _ => 31) (fun _ _ => rfl) fun ω t => ?_
    have := e1_range ω 0 i (4 * c + t)
    show (0 : ℤ) ≤ e1Of ω 0 i (4 * c + t) + 31 ∧ e1Of ω 0 i (4 * c + t) + 31 < 256
    omega
  · exact slice0_mem_V0_of_f1 _ (fun F => reg (fun _ : Unit => fun t => F (4 * c + t) j + 31) ()) fun _ => rfl

theorem coinWords_ignore : ∀ w ∈ coinWords m n k, Ignores (int7 m n k) DataStep w ∨ Moves (int7 m n k) NoiseStep w := by
  rintro w (⟨i, c, rfl⟩ | ⟨j, c, rfl⟩) <;> left <;> rintro ⟨a, ω⟩ _ ⟨a', ω'⟩ _ (h : ω = ω') <;> subst h <;> rfl

/-- At `(0, ω₀)` every coin word reads the bytes `31`. -/
theorem coin_at_ω₀ (w : Call m n k → ℚ) (hw : w ∈ coinWords m n k) :
    w (zeroA, ω₀) = reg (fun (_ : Unit) (_ : Fin 4) => (31 : ℤ)) () := by
  rcases hw with ⟨i, c, rfl⟩ | ⟨j, c, rfl⟩
  · unfold coinE
    refine (reg_eq_iff _ _ _ _).2 fun t => ?_
    show byte (e1Of ω₀ 0 i (4 * c + t) + 31) = byte 31
    rw [e1_ω₀, zero_add]
  · unfold coinF
    refine (reg_eq_iff _ _ _ _).2 fun t => ?_
    show byte (f1Of ω₀ (4 * c + t) j + 31) = byte 31
    rw [f1_ω₀, zero_add]

theorem xWord_moves {i c : ℕ} (hi : i < m) (hc : 4 * c < k) :
    Moves (int7 m n k) NoiseStep (xWord (m := m) (n := n) (k := k) i c) := by
  refine ⟨(zeroA, ω₀), zero_int7 _, (zeroA, ω₁), zero_int7 _, rfl, fun h => ?_⟩
  have h0 := (reg_eq_iff _ _ _ _).1 h 0
  have e : byte (zeroA i (4 * c + 0) + e1Of ω₀ 0 i (4 * c + 0) + 128) =
      byte (zeroA i (4 * c + 0) + e1Of ω₁ 0 i (4 * c + 0) + 128) := h0
  rw [e1_ω₀, e1_ω₁ hi (by omega)] at e
  simp only [zeroA, byte] at e
  omega

theorem xWord_ne_coin (i c : ℕ) (w : Call m n k → ℚ) (hw : w ∈ coinWords m n k) :
    ¬ ∀ g ∈ int7 m n k, xWord (m := m) (n := n) (k := k) i c g = w g := by
  refine differ_of_point (zero_int7 ω₀) ?_
  rw [coin_at_ω₀ w hw]
  intro h
  unfold xWord at h
  have e : byte (zeroA i (4 * c + 0) + e1Of ω₀ 0 i (4 * c + 0) + 128) = byte 31 := (reg_eq_iff _ _ _ _).1 h 0
  rw [e1_ω₀] at e
  simp only [zeroA, byte] at e
  omega

theorem xWord_slice0 (i c : ℕ) : slice0 (xWord (m := m) (n := n) (k := k) i c) ∈ V0 1 m n k := by
  refine reg_e1_mem_V0 _ i (fun t => 4 * c + t) (fun _ => 128) (fun ω t => ?_) fun ω t => ?_
  · show zeroA i (4 * c + t) + e1Of ω 0 i (4 * c + t) + 128 = _
    simp only [zeroA, zero_add]
  · have := e1_range ω 0 i (4 * c + t)
    show 0 ≤ zeroA i (4 * c + t) + e1Of ω 0 i (4 * c + t) + 128 ∧ zeroA i (4 * c + t) + e1Of ω 0 i (4 * c + t) + 128 < 256
    simp only [zeroA]
    omega

/-- Two block-1 `X` registers at different indices differ on `int7`: at `A = e_{i, 4c}` byte 0 of `(i, c)` reads
`129`, and of any other in-range index `128`. -/
theorem xWord_ne_xWord {i c i' c' : ℕ} (hne : (i, c) ≠ (i', c')) :
    ¬ ∀ g ∈ int7 m n k, xWord (m := m) (n := n) (k := k) i c g = xWord i' c' g := by
  refine differ_of_point (unitAt_int7 (m := m) (n := n) (k := k) i (4 * c) ω₀) fun h => ?_
  unfold xWord at h
  have e : byte (unitAt i (4 * c) i (4 * c + 0) + e1Of ω₀ 0 i (4 * c + 0) + 128) =
      byte (unitAt i (4 * c) i' (4 * c' + 0) + e1Of ω₀ 0 i' (4 * c' + 0) + 128) := (reg_eq_iff _ _ _ _).1 h 0
  have hu : unitAt i (4 * c) i' (4 * c' + 0) = 0 := by
    simp only [unitAt]
    rw [if_neg]
    rintro ⟨rfl, h'⟩
    exact hne (by rw [show c = c' by omega])
  have hu1 : unitAt i (4 * c) i (4 * c + 0) = 1 := by simp [unitAt]
  simp only [e1_ω₀, hu, hu1, byte] at e
  omega

theorem xWord_moves_data (i c : ℕ) : Moves (int7 m n k) DataStep (xWord (m := m) (n := n) (k := k) i c) := by
  refine ⟨(zeroA, ω₀), zero_int7 _, (unitAt i (4 * c), ω₀), unitAt_int7 _ _ _, rfl, fun h => ?_⟩
  unfold xWord at h
  have e : byte (zeroA i (4 * c + 0) + e1Of ω₀ 0 i (4 * c + 0) + 128) =
      byte (unitAt i (4 * c) i (4 * c + 0) + e1Of ω₀ 0 i (4 * c + 0) + 128) := (reg_eq_iff _ _ _ _).1 h 0
  have hu1 : unitAt i (4 * c) i (4 * c + 0) = 1 := by simp [unitAt]
  simp only [e1_ω₀, hu1, zeroA, byte] at e
  omega

theorem yWord_ignores_data (B : Mat) (j c : ℕ) : Ignores (int7 m n k) DataStep (yWord (m := m) (n := n) (k := k) B j c) := by
  rintro ⟨a, ω⟩ _ ⟨a', ω'⟩ _ (h : ω = ω')
  subst h
  rfl

theorem yWord_slice0 (B : Mat) (j c : ℕ) : slice0 (yWord (m := m) (n := n) (k := k) B j c) ∈ V0 1 m n k :=
  slice0_mem_V0_of_f1 _ (fun F => reg (fun _ : Unit => fun t => B (4 * c + t) j + F (4 * c + t) j) ()) fun _ => rfl

theorem yWord_step (B : Mat) {j c : ℕ} (hj : j < n) (hc : 4 * c < k) :
    yWord (m := m) (n := n) (k := k) B j c (zeroA, ω₀) ≠
      yWord (m := m) (n := n) (k := k) B j c (zeroA, ωF (4 * c) j) := by
  intro h
  unfold yWord at h
  have e : byte (B (4 * c + 0) j + f1Of ω₀ (4 * c + 0) j) = byte (B (4 * c + 0) j + f1Of (ωF (4 * c) j) (4 * c + 0) j) :=
    (reg_eq_iff _ _ _ _).1 h 0
  rw [f1_ω₀, f1_ωF, if_pos ⟨by omega, hj, by omega, rfl⟩] at e
  simp only [byte] at e
  omega

theorem yWord_moves (B : Mat) {j c : ℕ} (hj : j < n) (hc : 4 * c < k) :
    Moves (int7 m n k) NoiseStep (yWord (m := m) (n := n) (k := k) B j c) :=
  ⟨(zeroA, ω₀), zero_int7 _, (zeroA, ωF (4 * c) j), zero_int7 _, rfl, yWord_step B hj hc⟩

/-- A block-1 `Y` register is no coin word when its four weights are not all the coin offset `31` mod 256. -/
theorem yWord_ne_coin (B : Mat) (j c : ℕ) (hB31 : ∃ t : Fin 4, byte (B (4 * c + t) j) ≠ 31) (w : Call m n k → ℚ)
    (hw : w ∈ coinWords m n k) : ¬ ∀ g ∈ int7 m n k, yWord (m := m) (n := n) (k := k) B j c g = w g := by
  refine differ_of_point (zero_int7 ω₀) ?_
  rw [coin_at_ω₀ w hw]
  intro h
  unfold yWord at h
  obtain ⟨t, ht⟩ := hB31
  have e : byte (B (4 * c + t) j + f1Of ω₀ (4 * c + t) j) = byte 31 := (reg_eq_iff _ _ _ _).1 h t
  rw [f1_ω₀, add_zero] at e
  exact ht (e.trans rfl)

/-- Two block-1 `Y` registers at different in-range indices differ on `int7`: the noise step `ωF (4c) j` moves
`(j, c)` and leaves `(j', c')`. -/
theorem yWord_ne_yWord (B : Mat) {j c j' c' : ℕ} (hj : j < n) (hc : 4 * c < k) (hne : (j, c) ≠ (j', c')) :
    ¬ ∀ g ∈ int7 m n k, yWord (m := m) (n := n) (k := k) B j c g = yWord B j' c' g := by
  intro heq
  have h0 := heq _ (zero_int7 ω₀)
  have h1 := heq _ (zero_int7 (ωF (4 * c) j))
  have hfix : yWord (m := m) (n := n) (k := k) B j' c' (zeroA, ω₀) =
      yWord (m := m) (n := n) (k := k) B j' c' (zeroA, ωF (4 * c) j) := by
    unfold yWord
    refine (reg_eq_iff _ _ _ _).2 fun t => ?_
    show byte (B (4 * c' + t) j' + f1Of ω₀ (4 * c' + t) j') =
      byte (B (4 * c' + t) j' + f1Of (ωF (4 * c) j) (4 * c' + t) j')
    rw [f1_ω₀, f1_ωF, if_neg]
    rintro ⟨-, -, h', rfl⟩
    have : c' = c := by omega
    exact hne (by rw [this])
  exact yWord_step B hj hc (by rw [h0, hfix, ← h1])

/-- **Candidate 1 and the final word on NCP's block 1, concretely** (at the H200's measured table): with the free
values the constants, the call's input words and its coin words, a well-formed `MH100w` program that outputs the
non-final words at `P`, the final words of `Ix × Jx`, the committed block-1 `X` registers of `Xi` (biased packing) and
the committed block-1 `Y` registers of `Yi`, on every int7 input and every noise, costs at least
`32·(|P| + |Ix|·|Jx| + |Xi| + |Yi|)`. The `Y` registers need weights not all `31` mod 256. -/
theorem block1_charged [DecidableEq (Call m n k → ℚ)] (hind : CheckpointIndependent k 16 R) (B : Mat)
    (hB : ∀ l j, -255 ≤ B l j ∧ B l j ≤ 255)
    (P : Finset ℕ) (hP : P ⊆ range (nonFinal k m n)) (Ix Jx : Finset ℕ)
    (hJ0 : ∀ j ∈ Jx, ∃ l < k, B l j ≠ 0) (hJ : ∀ j ∈ Jx, ∀ j' ∈ Jx, j ≠ j' → ∃ l < k, B l j ≠ B l j')
    (Xi Yi : Finset (ℕ × ℕ)) (hXi : ∀ x ∈ Xi, x.1 < m ∧ 4 * x.2 < k)
    (hYi : ∀ y ∈ Yi, y.1 < n ∧ 4 * y.2 < k ∧ ∃ t : Fin 4, byte (B (4 * y.2 + t) y.1) ≠ 31)
    (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (Call m n k → ℚ)) (Pg : List (Op100 (Call m n k)))
    (hL : ∀ f ∈ L, f ∈ FreeCall (inWords m n k ∪ coinWords m n k))
    (hPg : ∀ o ∈ Pg, o.WFw h100Floor (FreeCall (inWords m n k ∪ coinWords m n k)))
    (hO : OutputsOn fpSem (int7 m n k) Pg L
      ((P.image (nfWord R B) ∪ (Ix ×ˢ Jx).image fun x => finalWord B x.1 x.2) ∪
        (Xi.image (fun x => xWord x.1 x.2) ∪ Yi.image fun y => yWord B y.1 y.2))) :
    32 * (P.card + Ix.card * Jx.card + (Xi.card + Yi.card)) ≤ cost100 h100Floor Pg := by
  have hxx : ∀ x ∈ Xi, ∀ x' ∈ Xi, x ≠ x' →
      ¬ ∀ g ∈ int7 m n k, xWord (m := m) (n := n) (k := k) x.1 x.2 g = xWord x'.1 x'.2 g :=
    fun x _ x' _ hne => xWord_ne_xWord hne
  have hyy : ∀ y ∈ Yi, ∀ y' ∈ Yi, y ≠ y' →
      ¬ ∀ g ∈ int7 m n k, yWord (m := m) (n := n) (k := k) B y.1 y.2 g = yWord B y'.1 y'.2 g :=
    fun y hy y' _ hne => yWord_ne_yWord B (hYi y hy).1 (hYi y hy).2.1 hne
  have hxy : ∀ t ∈ Xi.image (fun x => xWord (m := m) (n := n) (k := k) x.1 x.2),
      ∀ t' ∈ Yi.image (fun y => yWord (m := m) (n := n) (k := k) B y.1 y.2), ¬ ∀ g ∈ int7 m n k, t g = t' g := by
    intro t ht t' ht'
    obtain ⟨x, -, rfl⟩ := Finset.mem_image.1 ht
    obtain ⟨y, -, rfl⟩ := Finset.mem_image.1 ht'
    exact differ_of_moves (xWord_moves_data x.1 x.2) (yWord_ignores_data B y.1 y.2)
  have hcard : (Xi.image (fun x => xWord (m := m) (n := n) (k := k) x.1 x.2) ∪
      Yi.image fun y => yWord (m := m) (n := n) (k := k) B y.1 y.2).card = Xi.card + Yi.card := by
    rw [card_union_of_cross hxy, card_image_of_differ Xi _ hxx, card_image_of_differ Yi _ hyy]
  have hd : ∀ w ∈ Xi.image (fun x => xWord (m := m) (n := n) (k := k) x.1 x.2) ∪
      Yi.image (fun y => yWord (m := m) (n := n) (k := k) B y.1 y.2),
      ∀ w' ∈ Xi.image (fun x => xWord (m := m) (n := n) (k := k) x.1 x.2) ∪
      Yi.image (fun y => yWord (m := m) (n := n) (k := k) B y.1 y.2), (∀ g ∈ int7 m n k, w g = w' g) → w = w' := by
    intro w hw w' hw' h
    rcases Finset.mem_union.1 hw with hw | hw <;> rcases Finset.mem_union.1 hw' with hw' | hw'
    · obtain ⟨x, hx, rfl⟩ := Finset.mem_image.1 hw
      obtain ⟨x', hx', rfl⟩ := Finset.mem_image.1 hw'
      by_cases hxe : x = x'
      · rw [hxe]
      · exact absurd h (hxx x hx x' hx' hxe)
    · exact absurd h (hxy w hw w' hw')
    · exact absurd (fun g hg => (h g hg).symm) (hxy w' hw' w hw)
    · obtain ⟨y, hy, rfl⟩ := Finset.mem_image.1 hw
      obtain ⟨y', hy', rfl⟩ := Finset.mem_image.1 hw'
      by_cases hye : y = y'
      · rw [hye]
      · exact absurd h (hyy y hy y' hy' hye)
  have h := all_charged hind B hB (coinWords m n k) coinWords_V0 coinWords_ignore P hP Ix Jx hJ0 hJ _
    (fun w hw => by
      rcases Finset.mem_union.1 hw with hw | hw
      · obtain ⟨x, -, rfl⟩ := Finset.mem_image.1 hw
        exact xWord_slice0 x.1 x.2
      · obtain ⟨y, -, rfl⟩ := Finset.mem_image.1 hw
        exact yWord_slice0 B y.1 y.2)
    (fun w hw => by
      rcases Finset.mem_union.1 hw with hw | hw
      · obtain ⟨x, hx, rfl⟩ := Finset.mem_image.1 hw
        exact xWord_moves (hXi x hx).1 (hXi x hx).2
      · obtain ⟨y, hy, rfl⟩ := Finset.mem_image.1 hw
        exact yWord_moves B (hYi y hy).1 (hYi y hy).2.1)
    (fun w hw c hc => by
      rcases Finset.mem_union.1 hw with hw | hw
      · obtain ⟨x, -, rfl⟩ := Finset.mem_image.1 hw
        exact xWord_ne_coin x.1 x.2 c hc
      · obtain ⟨y, hy, rfl⟩ := Finset.mem_image.1 hw
        exact yWord_ne_coin B y.1 y.2 (hYi y hy).2.2 c hc)
    hd fpSem L Pg hL hPg hO
  rwa [hcard] at h

/-! ### All three blocks of `X` -/

/-- A register whose input-`0` slice has bytes in `[0, 255]`, each a function in `V₀` plus a constant, is in `V₀`
there. -/
theorem reg_mem_V0 (X : Call m n k → Fin 4 → ℤ) (ψ : Fin 4 → Noise 1 m n k → ℤ) (q : Fin 4 → ℤ)
    (hψ : ∀ t, (fun ω => (ψ t ω : ℚ)) ∈ V0 1 m n k) (hX : ∀ ω t, X (zeroA, ω) t = ψ t ω + q t)
    (hr : ∀ ω t, 0 ≤ X (zeroA, ω) t ∧ X (zeroA, ω) t < 256) : slice0 (reg X) ∈ V0 1 m n k := by
  have hf : slice0 (reg X) =
      (fun ω : Noise 1 m n k => (ψ 0 ω : ℚ)) + (256 : ℚ) • (fun ω => (ψ 1 ω : ℚ)) +
        (65536 : ℚ) • (fun ω => (ψ 2 ω : ℚ)) + (16777216 : ℚ) • (fun ω => (ψ 3 ω : ℚ)) +
        fun _ => ((q 0 + 256 * q 1 + 65536 * q 2 + 16777216 * q 3 : ℤ) : ℚ) := by
    funext ω
    simp only [slice0, reg_of_range X _ (hr ω), hX ω, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    push_cast
    ring
  rw [hf]
  exact Submodule.add_mem _ (Submodule.add_mem _ (Submodule.add_mem _ (Submodule.add_mem _ (hψ 0)
    (Submodule.smul_mem _ _ (hψ 1))) (Submodule.smul_mem _ _ (hψ 2))) (Submodule.smul_mem _ _ (hψ 3)))
    (mem_V0_of_const _)

/-- A formed `X` register in the biased packing: bytes `s·A[i, 4c+t] + φ t ω + 128`, with `s` the block's sign and `φ`
its noise. -/
def xReg (s : ℤ) (φ : Fin 4 → Noise 1 m n k → ℤ) (i c : ℕ) : Call m n k → ℚ :=
  reg fun g t => s * g.1 i (4 * c + t) + φ t g.2 + 128

/-- A block's noise: `0` at `ω₀`, within `±64`, and in `V₀`. -/
structure NoiseForm (φ : Fin 4 → Noise 1 m n k → ℤ) : Prop where
  zero : ∀ t, φ t ω₀ = 0
  range : ∀ t ω, -64 ≤ φ t ω ∧ φ t ω ≤ 64
  mem : ∀ t, (fun ω => (φ t ω : ℚ)) ∈ V0 1 m n k

theorem xReg_byte (s : ℤ) (φ : Fin 4 → Noise 1 m n k → ℤ) (i c : ℕ) (g g' : Call m n k)
    (h : xReg s φ i c g = xReg s φ i c g') (t : Fin 4) :
    byte (s * g.1 i (4 * c + t) + φ t g.2 + 128) = byte (s * g'.1 i (4 * c + t) + φ t g'.2 + 128) :=
  (reg_eq_iff _ _ _ _).1 h t

theorem xReg_slice0 {s : ℤ} {φ : Fin 4 → Noise 1 m n k → ℤ} (hφ : NoiseForm φ) (i c : ℕ) :
    slice0 (xReg s φ i c) ∈ V0 1 m n k := by
  refine reg_mem_V0 _ φ (fun _ => 128) hφ.mem (fun ω t => ?_) fun ω t => ?_
  · show s * zeroA i (4 * c + t) + φ t ω + 128 = φ t ω + 128
    simp [zeroA]
  · have := hφ.range t ω
    show 0 ≤ s * zeroA i (4 * c + t) + φ t ω + 128 ∧ s * zeroA i (4 * c + t) + φ t ω + 128 < 256
    simp only [zeroA, mul_zero, zero_add]
    omega

theorem xReg_moves {s : ℤ} {φ : Fin 4 → Noise 1 m n k → ℤ} (hφ : NoiseForm φ) (i c : ℕ)
    (hm : ∃ t ω, φ t ω ≠ 0) : Moves (int7 m n k) NoiseStep (xReg s φ i c) := by
  obtain ⟨t, ω, ht⟩ := hm
  refine ⟨(zeroA, ω₀), zero_int7 _, (zeroA, ω), zero_int7 _, rfl, fun h => ?_⟩
  have e := xReg_byte s φ i c _ _ h t
  have hr := hφ.range t ω
  simp only [zeroA, mul_zero, zero_add, hφ.zero, byte] at e
  omega

theorem xReg_ne_coin {s : ℤ} {φ : Fin 4 → Noise 1 m n k → ℤ} (hφ : NoiseForm φ) (i c : ℕ) (w : Call m n k → ℚ)
    (hw : w ∈ coinWords m n k) : ¬ ∀ g ∈ int7 m n k, xReg s φ i c g = w g := by
  refine differ_of_point (zero_int7 ω₀) ?_
  rw [coin_at_ω₀ w hw]
  intro h
  unfold xReg at h
  have e : byte (s * zeroA i (4 * c + 0) + φ 0 ω₀ + 128) = byte 31 := (reg_eq_iff _ _ _ _).1 h 0
  simp only [zeroA, mul_zero, zero_add, hφ.zero, byte] at e
  omega

/-- **Data separates**: two registers of signs `±1` differ on `int7` when their indices or their signs differ. At
`A = e_{i, 4c}` byte 0 of `(s, i, c)` reads `128 + s`, and of `(s', i', c')` reads `128 + s'` if the index is the
same and `128` if not. -/
theorem xReg_ne_data {s s' : ℤ} (hs : s = 1 ∨ s = -1) (hs' : s' = 1 ∨ s' = -1) {φ φ' : Fin 4 → Noise 1 m n k → ℤ}
    (hφ : NoiseForm φ) (hφ' : NoiseForm φ') {i c i' c' : ℕ} (hne : (i, c) ≠ (i', c') ∨ s ≠ s') :
    ¬ ∀ g ∈ int7 m n k, xReg s φ i c g = xReg s' φ' i' c' g := by
  refine differ_of_point (unitAt_int7 (m := m) (n := n) (k := k) i (4 * c) ω₀) fun h => ?_
  unfold xReg at h
  have e : byte (s * unitAt i (4 * c) i (4 * c + 0) + φ 0 ω₀ + 128) =
      byte (s' * unitAt i (4 * c) i' (4 * c' + 0) + φ' 0 ω₀ + 128) := (reg_eq_iff _ _ _ _).1 h 0
  have hu1 : unitAt i (4 * c) i (4 * c + 0) = 1 := by simp [unitAt]
  rw [hu1, hφ.zero, hφ'.zero] at e
  by_cases hii : i' = i ∧ c' = c
  · obtain ⟨rfl, rfl⟩ := hii
    rw [hu1] at e
    have hss : s ≠ s' := hne.resolve_left (fun h => h rfl)
    simp only [byte] at e
    rcases hs with rfl | rfl <;> rcases hs' with rfl | rfl <;> simp_all <;> omega
  · have hu : unitAt i (4 * c) i' (4 * c' + 0) = 0 := by
      simp only [unitAt]
      rw [if_neg]
      rintro ⟨rfl, h'⟩
      exact hii ⟨rfl, by omega⟩
    rw [hu] at e
    simp only [byte] at e
    rcases hs with rfl | rfl <;> omega

/-- **Noise separates**: registers with the same sign and index differ on `int7` when their noise differs somewhere. -/
theorem xReg_ne_noise {s : ℤ} {φ φ' : Fin 4 → Noise 1 m n k → ℤ} (hφ : NoiseForm φ) (hφ' : NoiseForm φ') (i c : ℕ)
    (hd : ∃ t ω, φ t ω ≠ φ' t ω) : ¬ ∀ g ∈ int7 m n k, xReg s φ i c g = xReg s φ' i c g := by
  obtain ⟨t, ω, ht⟩ := hd
  refine differ_of_point (zero_int7 ω) fun h => ?_
  unfold xReg at h
  have e : byte (s * zeroA i (4 * c + t) + φ t ω + 128) = byte (s * zeroA i (4 * c + t) + φ' t ω + 128) :=
    (reg_eq_iff _ _ _ _).1 h t
  have h1 := hφ.range t ω
  have h2 := hφ'.range t ω
  simp only [zeroA, mul_zero, zero_add, byte] at e
  omega

theorem xReg_moves_data {s : ℤ} (hs : s = 1 ∨ s = -1) {φ : Fin 4 → Noise 1 m n k → ℤ} (hφ : NoiseForm φ) (i c : ℕ) :
    Moves (int7 m n k) DataStep (xReg s φ i c) := by
  refine ⟨(zeroA, ω₀), zero_int7 _, (unitAt i (4 * c), ω₀), unitAt_int7 _ _ _, rfl, fun h => ?_⟩
  unfold xReg at h
  have e : byte (s * zeroA i (4 * c + 0) + φ 0 ω₀ + 128) = byte (s * unitAt i (4 * c) i (4 * c + 0) + φ 0 ω₀ + 128) :=
    (reg_eq_iff _ _ _ _).1 h 0
  have hu1 : unitAt i (4 * c) i (4 * c + 0) = 1 := by simp [unitAt]
  simp only [hu1, zeroA, hφ.zero, mul_zero, mul_one, byte] at e
  rcases hs with rfl | rfl <;> omega

/-- `E₁` is `1` at `(i, j)` and `0` elsewhere. -/
def ωE (i j : ℕ) : Noise 1 m n k := (fun _ i' j' => if (i' : ℕ) = i ∧ (j' : ℕ) = j then 32 else 31, fun _ _ => 31)

theorem e1_ωE (i j i' l : ℕ) :
    e1Of (ωE (m := m) (n := n) (k := k) i j) 0 i' l = if i' < m ∧ l < k ∧ i' = i ∧ l = j then 1 else 0 := by
  simp only [e1Of, ωE]
  by_cases h : 0 < 1 ∧ i' < m ∧ l < k
  · rw [dif_pos h]
    by_cases h' : i' = i ∧ l = j
    · rw [if_pos h', if_pos ⟨h.2.1, h.2.2, h'⟩]
      rfl
    · rw [if_neg h', if_neg fun h'' => h' ⟨h''.2.2.1, h''.2.2.2⟩]
      rfl
  · rw [dif_neg h, if_neg fun h'' => h ⟨Nat.one_pos, h''.1, h''.2.1⟩]

theorem mm_e1_ωE (R : Mat) {i j : ℕ} (hi : i < m) (hj : j < k) (l : ℕ) :
    mm k (e1Of (ωE (m := m) (n := n) (k := k) i j) 0) R i l = R j l := by
  simp only [mm, e1_ωE]
  rw [Finset.sum_eq_single j (fun j' hj' hne => by simp [hne]) (fun h => absurd (Finset.mem_range.2 hj) h)]
  simp [hi, hj]

/-- Block 1's noise: `E₁[i, 4c+t]`. -/
def φ₁ (i c : ℕ) : Fin 4 → Noise 1 m n k → ℤ := fun t ω => e1Of ω 0 i (4 * c + t)

/-- Block 2's noise: `E₂[i, 4c+t]`, `E₂ = E₁·R`. -/
def φ₂ (R : Mat) (i c : ℕ) : Fin 4 → Noise 1 m n k → ℤ := fun t ω => E2 k (e1Of ω 0) R i (4 * c + t)

/-- Block 3's noise: `−E₃[i, 4c+t]`, `E₃ = E₁ + E₂`; block 3's sign is `−1`. -/
def φ₃ (R : Mat) (i c : ℕ) : Fin 4 → Noise 1 m n k → ℤ := fun t ω => -(E3 k (e1Of ω 0) R i (4 * c + t))

theorem abs_e1 (ω : Noise 1 m n k) (i l : ℕ) : |e1Of ω 0 i l| ≤ 32 := by
  have := e1_range ω 0 i l
  rw [abs_le]
  omega

/-- `|E₂| ≤ 32` when every column of `R` has absolute sum at most 1 (a signed permutation's columns have sum 1). -/
theorem abs_e2 (R : Mat) (hR : ∀ l, ∑ j ∈ range k, |R j l| ≤ 1) (ω : Noise 1 m n k) (i l : ℕ) :
    |E2 k (e1Of ω 0) R i l| ≤ 32 := by
  simp only [E2, mm]
  calc |∑ j ∈ range k, e1Of ω 0 i j * R j l| ≤ ∑ j ∈ range k, |e1Of ω 0 i j * R j l| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j ∈ range k, 32 * |R j l| := Finset.sum_le_sum fun j _ => by
        rw [abs_mul]
        exact mul_le_mul_of_nonneg_right (abs_e1 ω i j) (abs_nonneg _)
    _ = 32 * ∑ j ∈ range k, |R j l| := by rw [Finset.mul_sum]
    _ ≤ 32 := by linarith [hR l]

theorem e2_mem_V0 (R : Mat) (i l : ℕ) : (fun ω : Noise 1 m n k => (E2 k (e1Of ω 0) R i l : ℚ)) ∈ V0 1 m n k := by
  have : (fun ω : Noise 1 m n k => (E2 k (e1Of ω 0) R i l : ℚ)) =
      ∑ j ∈ range k, (R j l : ℚ) • fun ω : Noise 1 m n k => (e1Of ω 0 i j : ℚ) := by
    funext ω
    simp only [E2, mm, Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
    push_cast
    exact Finset.sum_congr rfl fun j _ => mul_comm _ _
  rw [this]
  exact Submodule.sum_mem _ fun j _ => Submodule.smul_mem _ _ (e1_mem_V0 i j)

theorem e2_ω₀ (R : Mat) (i l : ℕ) : E2 k (e1Of (ω₀ (m := m) (n := n) (k := k)) 0) R i l = 0 := by
  simp [E2, mm, e1_ω₀]

theorem φ₁_form (i c : ℕ) : NoiseForm (φ₁ (m := m) (n := n) (k := k) i c) where
  zero t := e1_ω₀ 0 i _
  range t ω := by
    have := e1_range ω 0 i (4 * c + t)
    exact ⟨by simp only [φ₁]; omega, by simp only [φ₁]; omega⟩
  mem t := e1_mem_V0 i _

theorem φ₂_form (R : Mat) (hR : ∀ l, ∑ j ∈ range k, |R j l| ≤ 1) (i c : ℕ) :
    NoiseForm (φ₂ (m := m) (n := n) (k := k) R i c) where
  zero t := e2_ω₀ R i _
  range t ω := by
    have := abs_le.1 (abs_e2 R hR ω i (4 * c + t))
    exact ⟨by simp only [φ₂]; linarith, by simp only [φ₂]; linarith⟩
  mem t := e2_mem_V0 R i _

theorem φ₃_form (R : Mat) (hR : ∀ l, ∑ j ∈ range k, |R j l| ≤ 1) (i c : ℕ) :
    NoiseForm (φ₃ (m := m) (n := n) (k := k) R i c) where
  zero t := by simp [φ₃, E3, e1_ω₀, e2_ω₀]
  range t ω := by
    have h1 := abs_le.1 (abs_e1 ω i (4 * c + t))
    have h2 := abs_le.1 (abs_e2 R hR ω i (4 * c + t))
    simp only [φ₃, E3, Pi.add_apply]
    exact ⟨by linarith, by linarith⟩
  mem t := by
    have : (fun ω : Noise 1 m n k => (φ₃ R i c t ω : ℚ)) =
        -((fun ω : Noise 1 m n k => (e1Of ω 0 i (4 * c + t) : ℚ)) + fun ω => (E2 k (e1Of ω 0) R i (4 * c + t) : ℚ)) := by
      funext ω
      simp only [φ₃, E3, Pi.add_apply, Pi.neg_apply]
      push_cast
      ring
    rw [this]
    exact Submodule.neg_mem _ (Submodule.add_mem _ (e1_mem_V0 i _) (e2_mem_V0 R i _))

/-- `X`'s committed register `(b, i, c)`: block `b ∈ {0, 1, 2}` (blocks 1, 2, 3), row `i`, entries `4c..4c+3` of
the block. -/
def xAll (R : Mat) (x : ℕ × ℕ × ℕ) : Call m n k → ℚ :=
  if x.1 = 0 then xReg 1 (φ₁ x.2.1 x.2.2) x.2.1 x.2.2
  else if x.1 = 1 then xReg 1 (φ₂ R x.2.1 x.2.2) x.2.1 x.2.2
  else xReg (-1) (φ₃ R x.2.1 x.2.2) x.2.1 x.2.2

theorem xAll_0 (R : Mat) (i c : ℕ) : xAll (m := m) (n := n) (k := k) R (0, i, c) = xReg 1 (φ₁ i c) i c := by
  simp [xAll]

theorem xAll_1 (R : Mat) (i c : ℕ) : xAll (m := m) (n := n) (k := k) R (1, i, c) = xReg 1 (φ₂ R i c) i c := by
  simp [xAll]

theorem xAll_2 (R : Mat) (i c : ℕ) : xAll (m := m) (n := n) (k := k) R (2, i, c) = xReg (-1) (φ₃ R i c) i c := by
  simp [xAll]

/-- Each `X` register as `xReg` with its sign and a noise form. -/
theorem xAll_eq (R : Mat) (hR : ∀ l, ∑ j ∈ range k, |R j l| ≤ 1) (x : ℕ × ℕ × ℕ) :
    ∃ s φ, (s = 1 ∨ s = -1) ∧ NoiseForm φ ∧ xAll (m := m) (n := n) (k := k) R x = xReg s φ x.2.1 x.2.2 := by
  by_cases h0 : x.1 = 0
  · exact ⟨1, φ₁ x.2.1 x.2.2, Or.inl rfl, φ₁_form _ _, by simp [xAll, h0]⟩
  · by_cases h1 : x.1 = 1
    · exact ⟨1, φ₂ R x.2.1 x.2.2, Or.inl rfl, φ₂_form R hR _ _, by simp [xAll, h1]⟩
    · exact ⟨-1, φ₃ R x.2.1 x.2.2, Or.inr rfl, φ₃_form R hR _ _, by simp [xAll, h0, h1]⟩

/-- **The conditions on `X`'s registers** (all three blocks, biased packing), for index triples `(b, i, c)` with
`b < 3` and `i < m`:
* block 1 moves with the noise when `4c < k`;
* block 2 moves when some column `4c + t` of `R` has a nonzero entry above `k`;
* block 3 moves when some column `4c + t` of `I + R` does (else `E₃ = 0` there, and the register is `−A`: free);
* blocks 1 and 2 of one index differ when some column `4c + t` of `R` is not the unit vector `e_{4c+t}` (else
  `E₂ = E₁` there, and they are one register).
Under these, the registers are in `V₀` at the input-`0` slice, move with the noise, equal no coin word, and differ
pairwise on `int7`. -/
theorem xAll_props (R : Mat) (hR : ∀ l, ∑ j ∈ range k, |R j l| ≤ 1) (XI : Finset (ℕ × ℕ × ℕ))
    (hXI : ∀ x ∈ XI, x.1 < 3 ∧ x.2.1 < m ∧
      (x.1 = 0 → 4 * x.2.2 < k) ∧
      (x.1 = 1 → ∃ t : Fin 4, ∃ j < k, R j (4 * x.2.2 + t) ≠ 0) ∧
      (x.1 = 2 → ∃ t : Fin 4, ∃ j < k, (if j = 4 * x.2.2 + t then 1 else 0) + R j (4 * x.2.2 + t) ≠ 0))
    (h12 : ∀ i c, (0, i, c) ∈ XI → (1, i, c) ∈ XI →
      ∃ t : Fin 4, ∃ j < k, R j (4 * c + t) ≠ if j = 4 * c + t then 1 else 0) :
    (∀ x ∈ XI, slice0 (xAll (m := m) (n := n) (k := k) R x) ∈ V0 1 m n k) ∧
    (∀ x ∈ XI, Moves (int7 m n k) NoiseStep (xAll (m := m) (n := n) (k := k) R x)) ∧
    (∀ x ∈ XI, ∀ w ∈ coinWords m n k, ¬ ∀ g ∈ int7 m n k, xAll (m := m) (n := n) (k := k) R x g = w g) ∧
    (∀ x ∈ XI, ∀ x' ∈ XI, x ≠ x' →
      ¬ ∀ g ∈ int7 m n k, xAll (m := m) (n := n) (k := k) R x g = xAll R x' g) ∧
    (∀ x, Moves (int7 m n k) DataStep (xAll (m := m) (n := n) (k := k) R x)) := by
  refine ⟨fun x _ => ?_, fun x hx => ?_, fun x _ w hw => ?_, fun x hx x' hx' hne => ?_, fun x => ?_⟩
  · obtain ⟨s, φ, _hs, hφ, he⟩ := xAll_eq (m := m) (n := n) (k := k) R hR x
    rw [he]
    exact xReg_slice0 hφ _ _
  · obtain ⟨b, i, c⟩ := x
    obtain ⟨hb, hi, h0, h1, h2⟩ := hXI _ hx
    simp only at hb hi h0 h1 h2
    by_cases hb0 : b = 0
    · subst hb0
      rw [xAll_0]
      refine xReg_moves (φ₁_form i c) i c ⟨0, ω₁, ?_⟩
      show e1Of ω₁ 0 i (4 * c + 0) ≠ 0
      rw [add_zero, e1_ω₁ hi (h0 rfl)]
      norm_num
    · by_cases hb1 : b = 1
      · subst hb1
        rw [xAll_1]
        obtain ⟨t, j, hj, hR0⟩ := h1 rfl
        refine xReg_moves (φ₂_form R hR i c) i c ⟨t, ωE i j, ?_⟩
        show mm k (e1Of (ωE i j) 0) R i (4 * c + t) ≠ 0
        rw [mm_e1_ωE R hi hj]
        exact hR0
      · have hb2 : b = 2 := by omega
        subst hb2
        rw [xAll_2]
        obtain ⟨t, j, hj, hR0⟩ := h2 rfl
        refine xReg_moves (φ₃_form R hR i c) i c ⟨t, ωE i j, ?_⟩
        show -(e1Of (ωE i j) 0 i (4 * c + t) + mm k (e1Of (ωE i j) 0) R i (4 * c + t)) ≠ 0
        rw [mm_e1_ωE R hi hj, e1_ωE]
        intro h
        apply hR0
        by_cases hjt : j = 4 * c + t
        · subst hjt
          rw [if_pos ⟨hi, hj, rfl, rfl⟩] at h
          rw [if_pos rfl]
          omega
        · rw [if_neg fun h' => hjt h'.2.2.2.symm] at h
          rw [if_neg hjt]
          omega
  · obtain ⟨s, φ, _hs, hφ, he⟩ := xAll_eq (m := m) (n := n) (k := k) R hR x
    rw [he]
    exact xReg_ne_coin hφ _ _ w hw
  · obtain ⟨b, i, c⟩ := x
    obtain ⟨b', i', c'⟩ := x'
    by_cases hic : (i, c) = (i', c')
    · obtain ⟨rfl, rfl⟩ := Prod.mk.inj hic
      have hbb : b ≠ b' := fun h => hne (by rw [h])
      have hb3 := (hXI _ hx).1
      have hb3' := (hXI _ hx').1
      have hi := (hXI _ hx).2.1
      simp only at hb3 hb3' hi
      have key : (0, i, c) ∈ XI → (1, i, c) ∈ XI →
          ¬ ∀ g ∈ int7 m n k, xReg 1 (φ₁ (m := m) (n := n) (k := k) i c) i c g = xReg 1 (φ₂ R i c) i c g := by
        intro h0 h1
        obtain ⟨t, j, hj, hRj⟩ := h12 i c h0 h1
        refine xReg_ne_noise (φ₁_form i c) (φ₂_form R hR i c) i c ⟨t, ωE i j, ?_⟩
        show e1Of (ωE i j) 0 i (4 * c + t) ≠ mm k (e1Of (ωE i j) 0) R i (4 * c + t)
        rw [mm_e1_ωE R hi hj, e1_ωE]
        intro h
        apply hRj
        rw [← h]
        by_cases hjt : j = 4 * c + t
        · subst hjt
          rw [if_pos ⟨hi, hj, rfl, rfl⟩, if_pos rfl]
        · rw [if_neg hjt, if_neg fun h' => hjt h'.2.2.2.symm]
      have hne13 : (1 : ℤ) ≠ -1 := by norm_num
      rcases (show b = 0 ∨ b = 1 ∨ b = 2 by omega) with rfl | rfl | rfl <;>
        rcases (show b' = 0 ∨ b' = 1 ∨ b' = 2 by omega) with rfl | rfl | rfl
      · exact absurd rfl hbb
      · rw [xAll_0, xAll_1]
        exact key hx hx'
      · rw [xAll_0, xAll_2]
        exact xReg_ne_data (Or.inl rfl) (Or.inr rfl) (φ₁_form i c) (φ₃_form R hR i c) (Or.inr hne13)
      · rw [xAll_1, xAll_0]
        exact not_eq_symm (key hx' hx)
      · exact absurd rfl hbb
      · rw [xAll_1, xAll_2]
        exact xReg_ne_data (Or.inl rfl) (Or.inr rfl) (φ₂_form R hR i c) (φ₃_form R hR i c) (Or.inr hne13)
      · rw [xAll_2, xAll_0]
        exact xReg_ne_data (Or.inr rfl) (Or.inl rfl) (φ₃_form R hR i c) (φ₁_form i c) (Or.inr hne13.symm)
      · rw [xAll_2, xAll_1]
        exact xReg_ne_data (Or.inr rfl) (Or.inl rfl) (φ₃_form R hR i c) (φ₂_form R hR i c) (Or.inr hne13.symm)
      · exact absurd rfl hbb
    · obtain ⟨s, φ, hs, hφ, he⟩ := xAll_eq (m := m) (n := n) (k := k) R hR (b, i, c)
      obtain ⟨s', φ', hs', hφ', he'⟩ := xAll_eq (m := m) (n := n) (k := k) R hR (b', i', c')
      rw [he, he']
      exact xReg_ne_data hs hs' hφ hφ' (Or.inl hic)
  · obtain ⟨s, φ, hs, hφ, he⟩ := xAll_eq (m := m) (n := n) (k := k) R hR x
    rw [he]
    exact xReg_moves_data hs hφ _ _

/-- **Candidate 1 and the final word on NCP, concretely** (at the H200's measured table): with the free values the
constants, the call's input words and its coin words, a well-formed `MH100w` program that outputs, on every int7 input
and every noise, the non-final words at `P`, the final words of `Ix × Jx`, the committed `X` registers `XI` (all three
blocks, biased packing; `xAll_props`' conditions on `R`) and the committed block-1 `Y` registers `Yi` (weights not all
`31` mod 256), costs at least `32·(|P| + |Ix|·|Jx| + |XI| + |Yi|)`. -/
theorem ncp_charged [DecidableEq (Call m n k → ℚ)] (hind : CheckpointIndependent k 16 R) (B : Mat)
    (hB : ∀ l j, -255 ≤ B l j ∧ B l j ≤ 255) (hR : ∀ l, ∑ j ∈ range k, |R j l| ≤ 1)
    (P : Finset ℕ) (hP : P ⊆ range (nonFinal k m n)) (Ix Jx : Finset ℕ)
    (hJ0 : ∀ j ∈ Jx, ∃ l < k, B l j ≠ 0) (hJ : ∀ j ∈ Jx, ∀ j' ∈ Jx, j ≠ j' → ∃ l < k, B l j ≠ B l j')
    (XI : Finset (ℕ × ℕ × ℕ))
    (hXI : ∀ x ∈ XI, x.1 < 3 ∧ x.2.1 < m ∧
      (x.1 = 0 → 4 * x.2.2 < k) ∧
      (x.1 = 1 → ∃ t : Fin 4, ∃ j < k, R j (4 * x.2.2 + t) ≠ 0) ∧
      (x.1 = 2 → ∃ t : Fin 4, ∃ j < k, (if j = 4 * x.2.2 + t then 1 else 0) + R j (4 * x.2.2 + t) ≠ 0))
    (h12 : ∀ i c, (0, i, c) ∈ XI → (1, i, c) ∈ XI →
      ∃ t : Fin 4, ∃ j < k, R j (4 * c + t) ≠ if j = 4 * c + t then 1 else 0)
    (Yi : Finset (ℕ × ℕ)) (hYi : ∀ y ∈ Yi, y.1 < n ∧ 4 * y.2 < k ∧ ∃ t : Fin 4, byte (B (4 * y.2 + t) y.1) ≠ 31)
    (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (Call m n k → ℚ)) (Pg : List (Op100 (Call m n k)))
    (hL : ∀ f ∈ L, f ∈ FreeCall (inWords m n k ∪ coinWords m n k))
    (hPg : ∀ o ∈ Pg, o.WFw h100Floor (FreeCall (inWords m n k ∪ coinWords m n k)))
    (hO : OutputsOn fpSem (int7 m n k) Pg L
      ((P.image (nfWord R B) ∪ (Ix ×ˢ Jx).image fun x => finalWord B x.1 x.2) ∪
        (XI.image (xAll R) ∪ Yi.image fun y => yWord B y.1 y.2))) :
    32 * (P.card + Ix.card * Jx.card + (XI.card + Yi.card)) ≤ cost100 h100Floor Pg := by
  obtain ⟨hx0, hxN, hxW, hxx, hxD⟩ := xAll_props (m := m) (n := n) (k := k) R hR XI hXI h12
  have hyy : ∀ y ∈ Yi, ∀ y' ∈ Yi, y ≠ y' →
      ¬ ∀ g ∈ int7 m n k, yWord (m := m) (n := n) (k := k) B y.1 y.2 g = yWord B y'.1 y'.2 g :=
    fun y hy y' _ hne => yWord_ne_yWord B (hYi y hy).1 (hYi y hy).2.1 hne
  have hxy : ∀ t ∈ XI.image (xAll (m := m) (n := n) (k := k) R),
      ∀ t' ∈ Yi.image (fun y => yWord (m := m) (n := n) (k := k) B y.1 y.2), ¬ ∀ g ∈ int7 m n k, t g = t' g := by
    intro t ht t' ht'
    obtain ⟨x, -, rfl⟩ := Finset.mem_image.1 ht
    obtain ⟨y, -, rfl⟩ := Finset.mem_image.1 ht'
    exact differ_of_moves (hxD x) (yWord_ignores_data B y.1 y.2)
  have hcard : (XI.image (xAll (m := m) (n := n) (k := k) R) ∪
      Yi.image fun y => yWord (m := m) (n := n) (k := k) B y.1 y.2).card = XI.card + Yi.card := by
    rw [card_union_of_cross hxy, card_image_of_differ XI _ hxx, card_image_of_differ Yi _ hyy]
  have hd : ∀ w ∈ XI.image (xAll (m := m) (n := n) (k := k) R) ∪
      Yi.image (fun y => yWord (m := m) (n := n) (k := k) B y.1 y.2),
      ∀ w' ∈ XI.image (xAll (m := m) (n := n) (k := k) R) ∪
      Yi.image (fun y => yWord (m := m) (n := n) (k := k) B y.1 y.2), (∀ g ∈ int7 m n k, w g = w' g) → w = w' := by
    intro w hw w' hw' h
    rcases Finset.mem_union.1 hw with hw | hw <;> rcases Finset.mem_union.1 hw' with hw' | hw'
    · obtain ⟨x, hx, rfl⟩ := Finset.mem_image.1 hw
      obtain ⟨x', hx', rfl⟩ := Finset.mem_image.1 hw'
      by_cases hxe : x = x'
      · rw [hxe]
      · exact absurd h (hxx x hx x' hx' hxe)
    · exact absurd h (hxy w hw w' hw')
    · exact absurd (fun g hg => (h g hg).symm) (hxy w' hw' w hw)
    · obtain ⟨y, hy, rfl⟩ := Finset.mem_image.1 hw
      obtain ⟨y', hy', rfl⟩ := Finset.mem_image.1 hw'
      by_cases hye : y = y'
      · rw [hye]
      · exact absurd h (hyy y hy y' hy' hye)
  have h := all_charged hind B hB (coinWords m n k) coinWords_V0 coinWords_ignore P hP Ix Jx hJ0 hJ _
    (fun w hw => by
      rcases Finset.mem_union.1 hw with hw | hw
      · obtain ⟨x, hx, rfl⟩ := Finset.mem_image.1 hw
        exact hx0 x hx
      · obtain ⟨y, -, rfl⟩ := Finset.mem_image.1 hw
        exact yWord_slice0 B y.1 y.2)
    (fun w hw => by
      rcases Finset.mem_union.1 hw with hw | hw
      · obtain ⟨x, hx, rfl⟩ := Finset.mem_image.1 hw
        exact hxN x hx
      · obtain ⟨y, hy, rfl⟩ := Finset.mem_image.1 hw
        exact yWord_moves B (hYi y hy).1 (hYi y hy).2.1)
    (fun w hw c hc => by
      rcases Finset.mem_union.1 hw with hw | hw
      · obtain ⟨x, hx, rfl⟩ := Finset.mem_image.1 hw
        exact hxW x hx c hc
      · obtain ⟨y, hy, rfl⟩ := Finset.mem_image.1 hw
        exact yWord_ne_coin B y.1 y.2 (hYi y hy).2.2 c hc)
    hd fpSem L Pg hL hPg hO
  rwa [hcard] at h

end FinalWord

end Pouw.Dimension.SwLean
