import Proofs.Pouw.Eps1.WordLocal

/-!
# Theorem C's count in M3: class-P programs over issue slots

`Pouw.Guarantees.Eps1.ClassPSlots`. Walking back from each output's writer through arguments that hold an incidence
ends at an instruction that creates it (`exists_creates`), so the instructions create all `32·L·m·n` incidences
(`total_le`). An instruction that writes no output is a free word, which creates nothing, or a tuple, which creates
only incidences of its fields at their own segments (a field's prefix comes from an argument that holds it), at most
`Σ_k |E_k|` of them. A field's range has at least `128|E| + 1` values (`range_ncard`), and `Π_k (128 e_k + 1) ≤ 2^32`
with `e_k ≤ 32` caps `Σ_k e_k` at 65 (`sum_le_65`: two 32-entry fields and a one-entry field). Writers create at most
`w` each by hypothesis, and are `L·m·n` distinct instructions (`out_injective`), each alone in its slot.
-/

namespace Pouw.SecurityProofs.Eps1.ClassP

open Pouw.Eps1 Pouw.SecurityProofs.Eps1 Finset

section Capacity

/-- `g s = 4097^⌊s/32⌋·(128·(s mod 32) + 1)`, the least `Π_k (128 e_k + 1)` over entries `e_k ≤ 32` summing to `s`. -/
def g (s : ℕ) : ℕ := 4097 ^ (s / 32) * (128 * (s % 32) + 1)

theorem g_step (a e : ℕ) (he : e ≤ 32) : g (a + e) ≤ g a * (128 * e + 1) := by
  unfold g
  have hr := Nat.mod_lt a (by norm_num : 32 > 0)
  rcases Nat.lt_or_ge (a % 32 + e) 32 with h | h
  · have h1 : (a + e) / 32 = a / 32 := by omega
    have h2 : (a + e) % 32 = a % 32 + e := by omega
    rw [h1, h2, mul_assoc]
    apply Nat.mul_le_mul_left
    have hx : (128 * (a % 32) + 1) * (128 * e + 1) = 128 * (a % 32 + e) + 1 + 16384 * (a % 32 * e) := by ring
    rw [hx]
    exact Nat.le_add_right _ _
  · have h1 : (a + e) / 32 = a / 32 + 1 := by omega
    obtain ⟨k, hk⟩ : ∃ k, a % 32 + e = 32 + k := ⟨a % 32 + e - 32, by omega⟩
    have h2 : (a + e) % 32 = k := by omega
    rw [h1, h2, pow_succ, mul_assoc, mul_assoc]
    apply Nat.mul_le_mul_left
    have hk' : ((a % 32 : ℕ) : ℤ) + e = 32 + k := by exact_mod_cast hk
    have h3 : (0 : ℤ) ≤ (4096 - 128 * ((a % 32 : ℕ) : ℤ)) * (4096 - 128 * (e : ℤ)) :=
      mul_nonneg (by omega) (by omega)
    zify
    nlinarith [h3, hk']

theorem g_list (l : List ℕ) (h : ∀ e ∈ l, e ≤ 32) : g l.sum ≤ (l.map fun e => 128 * e + 1).prod := by
  induction l with
  | nil => simp [g]
  | cons e l ih =>
    rw [List.sum_cons, List.map_cons, List.prod_cons, add_comm, mul_comm]
    exact (g_step _ _ (h e List.mem_cons_self)).trans
      (Nat.mul_le_mul_right _ (ih fun x hx => h x (List.mem_cons_of_mem _ hx)))

theorem g_big (s : ℕ) (hs : 66 ≤ s) : 2 ^ 32 < g s := by
  unfold g
  rcases Nat.lt_or_ge (s / 32) 3 with h | h
  · have hq : s / 32 = 2 := by omega
    have hr : 2 ≤ s % 32 := by omega
    rw [hq]
    calc 2 ^ 32 < 4097 ^ 2 * (128 * 2 + 1) := by norm_num
      _ ≤ 4097 ^ 2 * (128 * (s % 32) + 1) := Nat.mul_le_mul_left _ (by omega)
  · calc 2 ^ 32 < 4097 ^ 3 * 1 := by norm_num
      _ ≤ 4097 ^ (s / 32) * (128 * (s % 32) + 1) :=
        Nat.mul_le_mul (Nat.pow_le_pow_right (by norm_num) h) (by omega)

/-- Entries `e_k ≤ 32` with `Π_k (128 e_k + 1) ≤ 2^32` sum to at most 65. -/
theorem sum_le_65 (l : List ℕ) (h : ∀ e ∈ l, e ≤ 32) (hp : (l.map fun e => 128 * e + 1).prod ≤ 2 ^ 32) :
    l.sum ≤ 65 := by
  by_contra hc
  have h1 := g_list l h
  have h2 := g_big l.sum (by omega)
  omega

/-- Every integer in `[−64|E|, 64|E|]` is a sum over `E` of entries in `[−64, 64]`. -/
theorem sum_decomp (E : Finset (Fin 32)) : ∀ z : ℤ, |z| ≤ 64 * E.card →
    ∃ c : Fin 32 → ℤ, (∀ p, |c p| ≤ 64) ∧ ∑ p ∈ E, c p = z := by
  induction E using Finset.induction_on with
  | empty =>
    intro z hz
    refine ⟨fun _ => 0, fun _ => by simp, ?_⟩
    simp only [card_empty, Nat.cast_zero, mul_zero] at hz
    simp only [sum_empty]
    exact (abs_nonpos_iff.mp hz).symm
  | insert a E ha ih =>
    intro z hz
    rw [card_insert_of_notMem ha] at hz
    have hz' := abs_le.mp hz
    let c0 : ℤ := if 64 ≤ z then 64 else if z ≤ -64 then -64 else z
    have hc0 : |c0| ≤ 64 := by
      apply abs_le.mpr
      simp only [c0]
      split_ifs <;> constructor <;> omega
    have hrest : |z - c0| ≤ 64 * E.card := by
      apply abs_le.mpr
      simp only [c0]
      push_cast at hz'
      split_ifs <;> constructor <;> omega
    obtain ⟨c, hc, hs⟩ := ih (z - c0) hrest
    refine ⟨Function.update c a c0, fun p => ?_, ?_⟩
    · by_cases hp : p = a
      · subst hp
        simpa using hc0
      · simpa [hp] using hc p
    · rw [sum_insert ha, Function.update_self,
        sum_congr rfl fun p hp => Function.update_of_ne (ne_of_mem_of_not_mem hp ha) c0 c, hs]
      ring

variable {m n Lseg : ℕ}

/-- A field's range holds every integer in `[−64|E|, 64|E|]`: dither bits off outside its segment, and on at its
segment exactly where the entry is nonzero. -/
theorem Icc_sub_range (φ : Field m n Lseg) :
    Set.Icc (-(64 * φ.E.card : ℤ)) (64 * φ.E.card) ⊆ φ.range := by
  intro z hz
  obtain ⟨c, hc, hs⟩ := sum_decomp φ.E z (abs_le.mpr ⟨by linarith [hz.1], hz.2⟩)
  refine ⟨fun i l p => decide (i = φ.τ.2.1 ∧ l = φ.τ.1 ∧ c p ≠ 0), fun _ p => if c p = 0 then 1 else c p,
    fun j p => ?_, ?_⟩
  · by_cases h : c p = 0
    · simp [h]
    · simp only [h, ite_false]
      exact ⟨Int.one_le_abs h, hc p⟩
  · have hpre : zeroPrefix (fun i l p => decide (i = φ.τ.2.1 ∧ l = φ.τ.1 ∧ c p ≠ 0))
        (fun _ p => if c p = 0 then 1 else c p) φ.τ = 0 := by
      apply sum_eq_zero
      intro l hl
      apply sum_eq_zero
      intro p _
      have : l ≠ φ.τ.1 := ne_of_lt (mem_filter.mp hl).2
      simp [this]
    simp only [Field.val, hpre, ite_self, zero_add]
    rw [← hs]
    apply sum_congr rfl
    intro p _
    by_cases h : c p = 0 <;> simp [h]

/-- A field's range is finite: its values lie in `[−64·32·(L + 1), 64·32·(L + 1)]`. -/
theorem range_finite (φ : Field m n Lseg) : φ.range.Finite := by
  apply (Set.finite_Icc (-(64 * 32 * ((Lseg : ℤ) + 1))) (64 * 32 * ((Lseg : ℤ) + 1))).subset
  rintro z ⟨R, U, hU, rfl⟩
  have hterm : ∀ (i : Fin m) (j : Fin n) (l : Fin Lseg) (p : Fin 32),
      |(if R i l p then U j p else 0)| ≤ 64 := by
    intro i j l p
    split_ifs
    · exact (hU _ _).2
    · simp
  have hseg : ∀ l, |zeroSeg R U l φ.τ.2.1 φ.τ.2.2| ≤ 64 * 32 := by
    intro l
    calc |zeroSeg R U l φ.τ.2.1 φ.τ.2.2|
        ≤ ∑ p, |(if R φ.τ.2.1 l p then U φ.τ.2.2 p else 0)| := abs_sum_le_sum_abs _ _
      _ ≤ ∑ _p : Fin 32, (64 : ℤ) := sum_le_sum fun p _ => hterm _ _ l p
      _ = 64 * 32 := by simp
  have hpre : |zeroPrefix R U φ.τ| ≤ 64 * 32 * Lseg := by
    calc |zeroPrefix R U φ.τ| ≤ ∑ l ∈ univ.filter (· < φ.τ.1), |zeroSeg R U l φ.τ.2.1 φ.τ.2.2| :=
          abs_sum_le_sum_abs _ _
      _ ≤ ∑ _l ∈ univ.filter (· < φ.τ.1), (64 * 32 : ℤ) := sum_le_sum fun l _ => hseg l
      _ ≤ ∑ _l : Fin Lseg, (64 * 32 : ℤ) :=
          sum_le_sum_of_subset_of_nonneg (filter_subset _ _) fun _ _ _ => by norm_num
      _ = 64 * 32 * Lseg := by simp; ring
  have hE : |∑ p ∈ φ.E, (if R φ.τ.2.1 φ.τ.1 p then U φ.τ.2.2 p else 0)| ≤ 64 * 32 := by
    calc _ ≤ ∑ p ∈ φ.E, |(if R φ.τ.2.1 φ.τ.1 p then U φ.τ.2.2 p else 0)| := abs_sum_le_sum_abs _ _
      _ ≤ ∑ p : Fin 32, |(if R φ.τ.2.1 φ.τ.1 p then U φ.τ.2.2 p else 0)| :=
          sum_le_sum_of_subset_of_nonneg (subset_univ _) fun _ _ _ => abs_nonneg _
      _ ≤ ∑ _p : Fin 32, (64 : ℤ) := sum_le_sum fun p _ => hterm _ _ _ p
      _ = 64 * 32 := by simp
  have hb : |(if φ.β then zeroPrefix R U φ.τ else 0)| ≤ 64 * 32 * Lseg := by
    split_ifs
    · exact hpre
    · simp
  have hsum := abs_add_le (if φ.β then zeroPrefix R U φ.τ else 0)
    (∑ p ∈ φ.E, (if R φ.τ.2.1 φ.τ.1 p then U φ.τ.2.2 p else 0))
  have hv := abs_le.mp (hsum.trans (add_le_add hb hE))
  simp only [Field.val]
  constructor <;> linarith [hv.1, hv.2]

/-- A field's range has at least `128|E| + 1` values. -/
theorem range_ncard (φ : Field m n Lseg) : 128 * φ.E.card + 1 ≤ φ.range.ncard := by
  have h := Set.ncard_le_ncard (Icc_sub_range φ) (range_finite φ)
  rw [← Finset.coe_Icc, Set.ncard_coe_finset, Int.card_Icc] at h
  omega

/-- A mixed-radix tuple's fields have at most 65 entries in all. -/
theorem fields_card_le (π : Packing m n Lseg) (h : π.MixedRadix) :
    (π.fields.map fun f => f.1.E.card).sum ≤ 65 := by
  obtain ⟨-, -, -, hprod, -⟩ := h
  apply sum_le_65
  · intro e he
    obtain ⟨f, -, rfl⟩ := List.mem_map.mp he
    exact (card_le_univ _).trans (by simp)
  · have h1 : ((π.fields.map fun f => f.1.E.card).map fun e => 128 * e + 1).prod ≤
        (π.fields.map fun f => f.1.range.ncard).prod := by
      rw [List.map_map]
      exact List.prod_le_prod fun f _ => range_ncard f.1
    have h2 : (((π.fields.map fun f => f.1.range.ncard).prod : ℕ) : ℝ) ≤ 2 ^ 32 := by
      rw [Nat.cast_list_prod, List.map_map]
      exact hprod
    have h3 : (π.fields.map fun f => f.1.range.ncard).prod ≤ 2 ^ 32 := by exact_mod_cast h2
    exact h1.trans h3

end Capacity

section Count

variable {m n Lseg : ℕ} {FR FU : Type} {G : ℕ}

/-- Every incidence has a creator: walk back from its writer through arguments that hold it. -/
theorem exists_creates (P : M3Algorithm (Dither m Lseg) (Cols n) m n Lseg FR FU G)
    (pk : Fin G → Option (Packing m n Lseg)) (τ : Tau m n Lseg) (p : Fin 32) : ∃ t, Creates P pk t τ p := by
  have key : ∀ k, ∀ t : Fin G, t.val < k → Holds P pk t τ p → ∃ t', Creates P pk t' τ p := by
    intro k
    induction k with
    | zero => intro t h _; exact absurd h (Nat.not_lt_zero _)
    | succ k ih =>
      intro t hk hh
      by_cases hc : ∀ j s, P.prog.arg t j = .inr s → ¬ Holds P pk (M3.earlier t s) τ p
      · exact ⟨t, hh, hc⟩
      · push_neg at hc
        obtain ⟨j, s, -, hs⟩ := hc
        exact ih _ (by have := s.isLt; simp only [M3.earlier]; omega) hs
  exact key _ (P.out τ) (Nat.lt_succ_self _) (Or.inl ⟨τ, rfl, rfl, le_rfl⟩)

open Classical in
/-- The incidences instruction `t` creates. -/
noncomputable def creSet (P : M3Algorithm (Dither m Lseg) (Cols n) m n Lseg FR FU G)
    (pk : Fin G → Option (Packing m n Lseg)) (t : Fin G) : Finset (Tau m n Lseg × Fin 32) :=
  univ.filter fun x => Creates P pk t x.1 x.2

theorem mem_creSet {P : M3Algorithm (Dither m Lseg) (Cols n) m n Lseg FR FU G} {pk : Fin G → Option (Packing m n Lseg)}
    {t : Fin G} {x : Tau m n Lseg × Fin 32} : x ∈ creSet P pk t ↔ Creates P pk t x.1 x.2 := by
  simp [creSet]

theorem created_eq (P : M3Algorithm (Dither m Lseg) (Cols n) m n Lseg FR FU G)
    (pk : Fin G → Option (Packing m n Lseg)) (t : Fin G) : created P pk t = (creSet P pk t).card := by
  rw [created, ← Set.ncard_coe_finset]
  congr 1
  ext x
  simp [mem_creSet]

/-- The instructions create all `32·L·m·n` incidences. -/
theorem total_le (P : M3Algorithm (Dither m Lseg) (Cols n) m n Lseg FR FU G) (pk : Fin G → Option (Packing m n Lseg)) :
    32 * (Lseg * m * n) ≤ ∑ t, created P pk t := by
  classical
  have hcard : (univ : Finset (Tau m n Lseg × Fin 32)).card = 32 * (Lseg * m * n) := by
    simp only [card_univ, Fintype.card_prod, Fintype.card_fin]
    ring
  rw [← hcard]
  calc (univ : Finset (Tau m n Lseg × Fin 32)).card ≤ (univ.biUnion fun t => creSet P pk t).card := by
        apply card_le_card
        intro x _
        obtain ⟨t, ht⟩ := exists_creates P pk x.1 x.2
        exact mem_biUnion.mpr ⟨t, mem_univ _, mem_creSet.mpr ht⟩
    _ ≤ ∑ t, (creSet P pk t).card := card_biUnion_le
    _ = ∑ t, created P pk t := sum_congr rfl fun t _ => (created_eq P pk t).symm

/-- An instruction that writes no output creates at most 65 incidences: none as a free word, and as a tuple only
incidences of its fields at their own segments. -/
theorem created_le_65 (P : M3Algorithm (Dither m Lseg) (Cols n) m n Lseg FR FU G)
    (pk : Fin G → Option (Packing m n Lseg)) (hP : ClassP P pk) (t : Fin G) (ht : ∀ τ, P.out τ ≠ t) :
    created P pk t ≤ 65 := by
  classical
  rcases hP t with ⟨⟨τ, hτ⟩, -⟩ | ⟨hn, -⟩ | ⟨π, hπ, hmr, -, hpre⟩
  · exact absurd hτ (ht τ)
  · have : creSet P pk t = ∅ := by
      apply eq_empty_of_forall_notMem
      intro x hx
      obtain ⟨hh, -⟩ := mem_creSet.mp hx
      rcases hh with ⟨τ', hτ', -⟩ | ⟨π, hπ, -⟩
      · exact ht τ' hτ'
      · rw [hn] at hπ
        cases hπ
    rw [created_eq, this, card_empty]
    norm_num
  · let L := π.fields.flatMap fun f => f.1.E.toList.map fun p => (f.1.τ, p)
    have hsub : creSet P pk t ⊆ L.toFinset := by
      intro x hx
      obtain ⟨hh, hc⟩ := mem_creSet.mp hx
      rcases hh with ⟨τ', hτ', -⟩ | ⟨π', hπ', f, hf, hcase⟩
      · exact absurd hτ' (ht τ')
      · rw [hπ] at hπ'
        cases hπ'
        rcases hcase with ⟨hτf, hp⟩ | ⟨hβ, h2, hlt⟩
        · rw [List.mem_toFinset, List.mem_flatMap]
          exact ⟨f, hf, List.mem_map.mpr ⟨x.2, Finset.mem_toList.mpr hp, by rw [hτf]⟩⟩
        · obtain ⟨j, s, harg, hs⟩ := hpre f hf hβ (by have := Fin.lt_def.mp hlt; omega)
          have hh := hs x.1.1 hlt x.2
          rw [h2] at hh
          exact absurd hh (hc j s harg)
    rw [created_eq]
    calc (creSet P pk t).card ≤ L.toFinset.card := card_le_card hsub
      _ ≤ L.length := List.toFinset_card_le _
      _ = (π.fields.map fun f => f.1.E.card).sum := by
          simp only [L, List.length_flatMap, List.length_map, Finset.length_toList]
      _ ≤ 65 := fields_card_le π hmr

/-- Distinct outputs have distinct writers: on the fiber of the later one (`rowF`, `colF` with entries `2`), its
word is `2` and the other's is `1` or `0`. -/
theorem out_injective (P : M3Algorithm (Dither m Lseg) (Cols n) m n Lseg FR FU G)
    (hC : P.Computes (fun _ U => ColsValid U) zeroOut) : Function.Injective P.out := by
  have key : ∀ τa τb : Tau m n Lseg, τb.1 ≤ τa.1 → τa ≠ τb → P.out τa ≠ P.out τb := by
    intro τa τb hle hne heq
    let r : Fin 32 → Bool := fun q => decide (q = 0)
    have hU : ColsValid (WordLocal.colF τa fun _ => (2 : ℤ)) :=
      WordLocal.colF_valid τa _ fun _ => by norm_num
    obtain ⟨v, hv⟩ := run_exists P.prog (P.inputs (WordLocal.rowF τa r) (WordLocal.colF τa fun _ => (2 : ℤ)))
    have ha := hC _ _ hU v hv τa
    have hb := hC _ _ hU v hv τb
    rw [heq, hb, WordLocal.zeroOut_fiber] at ha
    have hl : (∑ p : Fin 32, if r p then (2 : ℤ) else 0) = 2 := by simp [r]
    rw [hl] at ha
    have hpre := WordLocal.zeroPrefix_rowF τa τb r (WordLocal.colF τa fun _ => (2 : ℤ)) hle
    rcases lt_or_eq_of_le hle with hlt | heq1
    · rw [zeroOut, hpre, WordLocal.zeroSeg_rowF_ne τa r _ _ _ _ (ne_of_lt hlt)] at ha
      exact absurd ha (by decide)
    · by_cases hi : τb.2.1 = τa.2.1
      · have hj : τb.2.2 ≠ τa.2.2 := fun hj => hne (Prod.ext heq1.symm (Prod.ext hi.symm hj.symm))
        have : zeroSeg (WordLocal.rowF τa r) (WordLocal.colF τa fun _ => (2 : ℤ)) τb.1 τb.2.1 τb.2.2 = 1 := by
          simp [zeroSeg, WordLocal.rowF, WordLocal.colF, hi, heq1, hj, r]
        rw [zeroOut, hpre, this] at ha
        exact absurd ha (by decide)
      · have : zeroSeg (WordLocal.rowF τa r) (WordLocal.colF τa fun _ => (2 : ℤ)) τb.1 τb.2.1 τb.2.2 = 0 := by
          simp [zeroSeg, WordLocal.rowF, hi]
        rw [zeroOut, hpre, this] at ha
        exact absurd ha (by decide)
  intro τ1 τ2 h
  by_contra hne
  rcases le_total τ2.1 τ1.1 with h12 | h21
  · exact key τ1 τ2 h12 hne h
  · exact key τ2 τ1 h21 (Ne.symm hne) h.symm

/-- **Theorem C's count over issue slots** (`Pouw.Guarantees.Eps1.ClassPSlots`). -/
theorem classPSlots : Pouw.Guarantees.Eps1.ClassPSlots := by
  intro m n Lseg FR FU G P pk w d S slot hC hP hw halone hd
  classical
  have hinj := out_injective P hC
  set W : Finset (Fin G) := univ.image P.out with hWdef
  have hmemW : ∀ t, t ∈ W ↔ ∃ τ, P.out τ = t := fun t => by simp [W]
  have hWc : W.card = Lseg * m * n := by
    rw [card_image_of_injective _ hinj, card_univ]
    simp only [Fintype.card_prod, Fintype.card_fin]
    ring
  set NW := univ.filter fun t => t ∉ W with hNWdef
  have hsum : ∑ t, created P pk t ≤ w * W.card + 65 * NW.card := by
    rw [← sum_filter_add_sum_filter_not univ (fun t => t ∈ W)]
    have hfW : univ.filter (fun t => t ∈ W) = W := by ext t; simp
    have h1 : ∑ t ∈ univ.filter (fun t => t ∈ W), created P pk t ≤ w * W.card := by
      rw [hfW]
      calc ∑ t ∈ W, created P pk t ≤ ∑ _t ∈ W, w := sum_le_sum fun t ht => by
              obtain ⟨τ, rfl⟩ := (hmemW t).mp ht
              exact hw τ
        _ = w * W.card := by rw [sum_const, smul_eq_mul, mul_comm]
    have h2 : ∑ t ∈ NW, created P pk t ≤ 65 * NW.card := by
      calc ∑ t ∈ NW, created P pk t ≤ ∑ _t ∈ NW, 65 := sum_le_sum fun t ht =>
              created_le_65 P pk hP t fun τ hτ => (mem_filter.mp ht).2 ((hmemW t).mpr ⟨τ, hτ⟩)
        _ = 65 * NW.card := by rw [sum_const, smul_eq_mul, mul_comm]
    exact add_le_add h1 h2
  set WS := W.image slot with hWSdef
  have hWS : WS.card = W.card := card_image_of_injOn fun a ha b hb hab => by
    obtain ⟨τa, rfl⟩ := (hmemW a).mp ha
    obtain ⟨τb, rfl⟩ := (hmemW b).mp hb
    exact halone τb (P.out τa) hab
  have hS : WS.card ≤ S := (card_le_univ WS).trans (by simp)
  have hNW : NW.card ≤ d * (univ \ WS).card := by
    apply card_le_mul_card_image_of_maps_to (f := slot)
    · intro t ht
      simp only [mem_sdiff, mem_univ, true_and]
      intro hts
      obtain ⟨t', ht', hst⟩ := mem_image.mp hts
      obtain ⟨τ, rfl⟩ := (hmemW t').mp ht'
      exact (mem_filter.mp ht).2 ((hmemW t).mpr ⟨τ, (halone τ t hst.symm).symm⟩)
    · intro s _
      exact (card_le_card (filter_subset_filter _ (subset_univ NW))).trans (hd s)
  have hcomp : (univ \ WS).card = S - WS.card := by
    rw [card_sdiff_of_subset (subset_univ _), card_univ, Fintype.card_fin]
  have htot := total_le P pk
  have hle : Lseg * m * n ≤ S := hWc ▸ hWS ▸ hS
  have hnat : 32 * (Lseg * m * n) ≤ w * (Lseg * m * n) + 65 * (d * (S - Lseg * m * n)) := by
    rw [hcomp, hWS, hWc] at hNW
    rw [hWc] at hsum
    calc 32 * (Lseg * m * n) ≤ ∑ t, created P pk t := htot
      _ ≤ w * (Lseg * m * n) + 65 * NW.card := hsum
      _ ≤ w * (Lseg * m * n) + 65 * (d * (S - Lseg * m * n)) := by
          apply Nat.add_le_add_left
          exact Nat.mul_le_mul_left _ hNW
  have hreal : ((32 * (Lseg * m * n) : ℕ) : ℝ) ≤ ((w * (Lseg * m * n) + 65 * (d * (S - Lseg * m * n)) : ℕ) : ℝ) :=
    Nat.cast_le.mpr hnat
  push_cast [Nat.cast_sub hle] at hreal
  linarith

end Count

end Pouw.SecurityProofs.Eps1.ClassP
