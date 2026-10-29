import Proofs.Pouw.Dimension.H100FloorProofs
import Proofs.Pouw.Dimension.H100Proofs
import Proofs.Pouw.NCP.Proofs

/-!
# Committed formed operands and the final word under `DistinctWritesFloor` (scratch; not in a lock)

**Draft, pending review.** Two questions against `MH100w`'s `DistinctWritesFloor` (`Pouw.Dimension.H100Floor`) at the
floor table `h100Floor`, which is the H200's measured table (`h200Measured = h100Floor`).

* **Candidate 1** (`formed_charged`, `formed_free_of_saltIndep`, `rawDither_exact`). Commit the formed noised operands
  `X = A + D` and `Y = B + E`, int8 packed four to a 32-bit register (`pack4`). On the grid `D × Ω` (data fixed before the
  noise, then the noise), with the free set `FreeC1 C`: every salt-independent value and the raw coin words `C`. A formed
  register is charged 32 iff it moves with the salt at some fixed datum on `S` and equals no coin word on `S`; distinct
  registers count once each. Otherwise a zero-cost program outputs it (`formed_free_of_saltIndep`, `coin_free`), so no
  bound charges it. For an i.i.d. raw-byte dither the charged registers are exactly those whose data register is not
  `0` mod 256 (`rawDither_exact`).
* **The final-word lemma** (`finalWordTarget`, `finalWord_charged`, `all_charged`). On a call's grid `Call m n k`
  (its int7 input `A` and its noise, the weights `B` fixed) with the free set the constants and the call's own words
  (its input words `inWords`, and noise-side words `W'`: coin words and, in the variant CORE-TIME stated, the formed
  operand words), NCP's final words `A_i·B_j` and every non-final checked word are pairwise distinct and none is free.
  The non-final words rest on the proved `indep_words_sub` at the zero-data slice; the final words on `B`'s columns
  being nonzero and pairwise distinct and int8. The free set is the words themselves, not their span: the span on a
  weights slice holds every final word (CORE-TIME's `span_frees_final_word`).
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
theorem reg_eq_iff {G : Type} (X Y : G → Fin 4 → ℤ) (g g' : G) :
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

/-- **The call's own words** (`FreeCall` as CORE-TIME stated it): the constants and the words `W`. -/
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

/-- **The final-word lemma for NCP** (CORE-TIME's `FinalWordTarget`, proved): on a call's grid (int7 input, every
noise, the weights `B` fixed), with the free values the constants, the call's input words and noise-side words `W'`
(each in `V₀` at the input-`0` slice, and each ignoring the input or moving with the noise: coin words and the formed
words both qualify), the non-final checked words at positions `P` and the final words `A_i·B_j` for `i ∈ Ix`,
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

end FinalWord

end Pouw.Dimension.SwLean
