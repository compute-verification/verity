import Models.Accounting.Guarantees.Isa

/-!
# A6 and A7 from an instruction table

`Models.Accounting.Guarantees.Isa`'s statements (note:20261019T0520Z-draft-mm-isa):

* `stray_eq_zero`, `fits_of_ne`: every head is on a route its row allows: a traced row passes the discipline only as a
  copy, whose instances head nothing, and the labelling places every other class's heads.
* `noSum_of_table`: NoSum is the class list's completeness: the fixed words are at most the heads, every head is on a
  route, and a route's words are its heads.
* `sideUnits_of_table`: a side unit's words are its class's heads, at most `words` an instruction.
* `sheetWords_of_table`: A6 from the rows' words: a writer kind's class at most `ω`, a vector class at most 64.
* `Sm120Sheet`, `Sm90Sheet`, `Sm100Sheet`: each device's discipline decided row by row, over the shared classes
  (`sync_ok`, `base_ok`) and its own; its absent families are vacuous.
-/

namespace Accounting.SecurityProofs.Isa

open Accounting.TS Accounting.Isa Accounting.Assumptions.TS Accounting.Assumptions.Isa Finset

/-- A traced row passes the discipline only as a mover. -/
theorem mover_of_ok {r : Row} (h : r.ok = true) (hr : r.route = .traced) : r.mover = true := by
  simpa only [Row.ok, hr] using h

/-- A vector row passes the discipline only at 64 words or fewer. -/
theorem words_le_of_ok {r : Row} (h : r.ok = true) (hr : r.route = .lane) : r.words ≤ 64 := by
  simp only [Row.ok, hr, Bool.and_eq_true, decide_eq_true_eq] at h
  exact h.2

section Rows

variable {K T L U : Type} {Tb : K → Row} {A : Anatomy K T L U} {H : Heads K T L U}

/-- The ten families' rows are every class's. -/
theorem rowAt_of_families (hmma : IsaRowsMma Tb A H) (hwgmma : IsaRowsWgmma Tb A H)
    (htc : IsaRowsTcgen05 Tb A H) (htmem : IsaRowsTmem Tb A H) (hlane : IsaRowsLane Tb A H)
    (huni : IsaRowsUniform Tb A H) (hmem : IsaRowsMemory Tb A H) (hbulk : IsaRowsBulk Tb A H)
    (htex : IsaRowsTexture Tb A H) (hwarp : IsaRowsWarp Tb A H) (k : K) : RowAt Tb A H k := by
  cases hf : (Tb k).fam
  exacts [hmma k hf, hwgmma k hf, htc k hf, htmem k hf, hlane k hf, huni k hf, hmem k hf, hbulk k hf, htex k hf,
    hwarp k hf]

end Rows

section Table

variable {K R T L U F X : Type} [Fintype K] {Tb : K → Row} {σ : U → K} {Z : Model K R T L U F X}
  {A : Anatomy K T L U} {H : Heads K T L U}

/-- No head is off the routes: a traced class heads nothing, and the labelling places every other class's heads. -/
theorem stray_eq_zero (hok : ∀ k, (Tb k).ok = true) (hlab : IsaLabelled Tb σ Z A H) (hrows : ∀ k, RowAt Tb A H k)
    (k : K) : H.stray k = 0 := by
  by_cases hr : (Tb k).route = .traced
  · exact ((hrows k).2 (mover_of_ok (hok k) hr)).1
  · exact hlab.2.1 k hr

/-- Class `k`'s heads on route `h` are on a route its row allows. -/
theorem fits_of_ne (hok : ∀ k, (Tb k).ok = true) (hlab : IsaLabelled Tb σ Z A H) (hrows : ∀ k, RowAt Tb A H k)
    {k : K} {h : Head T L U} (hne : H.on k h ≠ 0) : Fits Tb σ Z k h := by
  by_cases hr : (Tb k).route = .traced
  · exact absurd (((hrows k).2 (mover_of_ok (hok k) hr)).2 h) hne
  · exact hlab.2.2 k h hr hne

/-- **NoSum from the table**: the fixed words are at most the heads, every head is on a route, and a route's words
are its heads. -/
theorem noSum_of_table [Fintype T] [Fintype L] [Fintype U] (hok : ∀ k, (Tb k).ok = true)
    (hcomp : IsaComplete A H) (hlab : IsaLabelled Tb σ Z A H) (hrows : ∀ k, RowAt Tb A H k) : NoSum A := by
  have hcomp' : A.fixed ≤ ∑ k, (H.stray k + ∑ h, H.on k h) := hcomp
  show A.fixed ≤ ∑ h, A.words h
  calc A.fixed ≤ ∑ k, (H.stray k + ∑ h, H.on k h) := hcomp'
    _ = ∑ k, ∑ h, H.on k h := sum_congr rfl fun k _ => by rw [stray_eq_zero hok hlab hrows k, zero_add]
    _ = ∑ h, ∑ k, H.on k h := sum_comm
    _ = ∑ h, A.words h := sum_congr rfl fun h _ => (hlab.1 h).symm

/-- Side unit `u`'s words are its class's heads on it. -/
theorem words_side (hok : ∀ k, (Tb k).ok = true) (hlab : IsaLabelled Tb σ Z A H) (hrows : ∀ k, RowAt Tb A H k)
    (u : U) : A.words (.side u) = H.on (σ u) (.side u) := by
  rw [hlab.1]
  refine sum_eq_single (σ u) (fun k _ hk => ?_) (fun h => absurd (mem_univ _) h)
  by_contra hne
  have hf : (Tb k).route = .side ∧ σ u = k := fits_of_ne hok hlab hrows hne
  exact hk hf.2.symm

/-- Writer kind `t`'s words are its class's heads on it. -/
theorem words_writer (hok : ∀ k, (Tb k).ok = true) (hlab : IsaLabelled Tb σ Z A H) (hrows : ∀ k, RowAt Tb A H k)
    (t : T) : A.words (.writer t) = H.on (Z.cls t) (.writer t) := by
  rw [hlab.1]
  refine sum_eq_single (Z.cls t) (fun k _ hk => ?_) (fun h => absurd (mem_univ _) h)
  by_contra hne
  have hf : (Tb k).route = .writer ∧ Z.cls t = k := fits_of_ne hok hlab hrows hne
  exact hk hf.2.symm

/-- **SideUnits from the table**: a side unit's words are its class's heads, at most `words` an instruction of the
class, and its per-word vector is at most `1/words` on that class and nothing elsewhere. -/
theorem sideUnits_of_table [DecidableEq K] [Fintype T] [Fintype L] [Fintype U] (hok : ∀ k, (Tb k).ok = true)
    (hread : ReadsTable Tb σ Z) (hsplit : A.Split) (hlab : IsaLabelled Tb σ Z A H) (hrows : ∀ k, RowAt Tb A H k) :
    SideUnits Z A := by
  intro u k
  have hw0 : 0 ≤ A.words (.side u) := hsplit.2.2.2.1 _
  have hn0 : 0 ≤ A.nword (.side u) k := hsplit.1 _ _
  refine (mul_le_mul_of_nonneg_left (hread.1 u k) hw0).trans ?_
  split_ifs with hk
  · subst hk
    have hon : H.on (σ u) (.side u) ≤ ((Tb (σ u)).words : ℝ) * A.nword (.side u) (σ u) := (hrows (σ u)).1 _
    rw [words_side hok hlab hrows u]
    rcases Nat.eq_zero_or_pos (Tb (σ u)).words with h0 | hpos
    · rw [h0, Nat.cast_zero, div_zero, mul_zero]
      exact hn0
    · have hne : ((Tb (σ u)).words : ℝ) ≠ 0 := (Nat.cast_pos.mpr hpos).ne'
      calc H.on (σ u) (.side u) * (1 / ((Tb (σ u)).words : ℝ))
          ≤ ((Tb (σ u)).words : ℝ) * A.nword (.side u) (σ u) * (1 / ((Tb (σ u)).words : ℝ)) :=
            mul_le_mul_of_nonneg_right hon (by positivity)
        _ = A.nword (.side u) (σ u) := by
            rw [mul_comm ((Tb (σ u)).words : ℝ), mul_assoc, mul_one_div_cancel hne, mul_one]
  · rw [mul_zero]
    exact hn0

/-- **A6 from the table**: a writer kind's words are its class's heads, at most `words ≤ ω` an instruction, and a
vector or table route's are vector classes' heads, at most 64 an instruction, every vector class in `V`. -/
theorem sheetWords_of_table [DecidableEq K] [Fintype T] [Fintype L] [Fintype U] (hok : ∀ k, (Tb k).ok = true)
    (hread : ReadsTable Tb σ Z) (hsplit : A.Split) (hlab : IsaLabelled Tb σ Z A H) (hrows : ∀ k, RowAt Tb A H k) :
    SheetWords Z A := by
  have lane_le : ∀ h, (∀ k, Fits Tb σ Z k h → (Tb k).route = .lane) →
      A.words h ≤ 2 * (32 * ∑ k ∈ Z.vec, A.nword h k) := by
    intro h hlane
    have hout : ∀ k ∈ (univ : Finset K), k ∉ Z.vec → H.on k h = 0 := by
      intro k _ hk
      by_contra hne
      exact hk (hread.2.2 k (hlane k (fits_of_ne hok hlab hrows hne)))
    have hle : ∀ k ∈ Z.vec, H.on k h ≤ 64 * A.nword h k := by
      intro k _
      by_cases hz : H.on k h = 0
      · rw [hz]
        exact mul_nonneg (by norm_num) (hsplit.1 _ _)
      · have hw : ((Tb k).words : ℝ) ≤ 64 := by
          exact_mod_cast words_le_of_ok (hok k) (hlane k (fits_of_ne hok hlab hrows hz))
        exact ((hrows k).1 h).trans (mul_le_mul_of_nonneg_right hw (hsplit.1 _ _))
    calc A.words h = ∑ k, H.on k h := hlab.1 h
      _ = ∑ k ∈ Z.vec, H.on k h := (sum_subset (subset_univ _) hout).symm
      _ ≤ ∑ k ∈ Z.vec, 64 * A.nword h k := sum_le_sum hle
      _ = 2 * (32 * ∑ k ∈ Z.vec, A.nword h k) := by rw [← mul_sum]; ring
  refine ⟨fun t => ?_, lane_le .vector fun k hf => hf, fun l => lane_le (.table l) fun k hf => hf⟩
  rw [words_writer hok hlab hrows t]
  exact ((hrows (Z.cls t)).1 _).trans (mul_le_mul_of_nonneg_right (hread.2.1 t) (hsplit.1 _ _))

end Table

/-- **A6 and A7 from an instruction table.** -/
theorem IsaSheet : Accounting.Guarantees.Isa.IsaSheet := by
  intro K R T L U F X _ _ _ _ _ Tb σ Z A H hok hread hsplit hcomp hlab hmma hwgmma htc htmem hlane huni hmem hbulk
    htex hwarp
  have hrows := rowAt_of_families hmma hwgmma htc htmem hlane huni hmem hbulk htex hwarp
  exact ⟨sheetWords_of_table hok hread hsplit hlab hrows, noSum_of_table hok hcomp hlab hrows,
    sideUnits_of_table hok hread hsplit hlab hrows⟩

section Classes

theorem sync_ok (s : Sync) : s.row.ok = true := by cases s <;> decide +kernel
theorem base_ok (c : Base) : c.row.ok = true := by cases c <;> decide +kernel
theorem gmma_ok (g : Gmma) : g.row.ok = true := by cases g <;> decide +kernel
theorem tc05_ok (t : Tc05) : t.row.ok = true := by cases t <;> decide +kernel
theorem tmem_ok (m : Tmem) : m.row.ok = true := by cases m <;> decide +kernel

theorem sync_fam (s : Sync) : s.row.fam = .mma := by cases s <;> rfl
theorem gmma_fam (g : Gmma) : g.row.fam = .wgmma := by cases g <;> rfl
theorem tc05_fam (t : Tc05) : t.row.fam = .tcgen05 := by cases t <;> rfl
theorem tmem_fam (m : Tmem) : m.row.fam = .tmem := by cases m <;> rfl

/-- No class outside the tensor cores is in a tensor family. -/
theorem base_fam (c : Base) :
    c.row.fam ≠ .mma ∧ c.row.fam ≠ .wgmma ∧ c.row.fam ≠ .tcgen05 ∧ c.row.fam ≠ .tmem := by
  cases c <;> decide +kernel

theorem ne_of_fam {r : Row} {f g : Family} (h : r.fam = f) (hfg : f ≠ g) : r.fam ≠ g := by
  rw [h]
  exact hfg

end Classes

/-- Every sm_120 row passes the discipline. -/
theorem sm120_ok : ∀ k, (Sm120.row k).ok = true
  | .sync s => sync_ok s
  | .base c => base_ok c

/-- sm_120 has no wgmma, tcgen05 or tensor-memory class. -/
theorem sm120_fam :
    ∀ k : Sm120, (Sm120.row k).fam ≠ .wgmma ∧ (Sm120.row k).fam ≠ .tcgen05 ∧ (Sm120.row k).fam ≠ .tmem
  | .sync s => ⟨ne_of_fam (sync_fam s) (by decide), ne_of_fam (sync_fam s) (by decide),
      ne_of_fam (sync_fam s) (by decide)⟩
  | .base c => (base_fam c).2

/-- **A6 and A7 on sm_120.** -/
theorem Sm120Sheet : Accounting.Guarantees.Isa.Sm120Sheet := by
  intro R T L U F X _ _ _ σ Z A H hread hsplit hcomp hlab hmma hlane huni hmem hbulk htex hwarp
  exact IsaSheet Sm120.row σ Z A H sm120_ok hread hsplit hcomp hlab hmma
    (fun k hk => absurd hk (sm120_fam k).1) (fun k hk => absurd hk (sm120_fam k).2.1)
    (fun k hk => absurd hk (sm120_fam k).2.2) hlane huni hmem hbulk htex hwarp

/-- Every sm_90a row passes the discipline. -/
theorem sm90_ok : ∀ k, (Sm90.row k).ok = true
  | .sync s => sync_ok s
  | .gmma g => gmma_ok g
  | .base c => base_ok c

/-- sm_90a has no tcgen05 or tensor-memory class. -/
theorem sm90_fam : ∀ k : Sm90, (Sm90.row k).fam ≠ .tcgen05 ∧ (Sm90.row k).fam ≠ .tmem
  | .sync s => ⟨ne_of_fam (sync_fam s) (by decide), ne_of_fam (sync_fam s) (by decide)⟩
  | .gmma g => ⟨ne_of_fam (gmma_fam g) (by decide), ne_of_fam (gmma_fam g) (by decide)⟩
  | .base c => (base_fam c).2.2

/-- **A6 and A7 on sm_90a.** -/
theorem Sm90Sheet : Accounting.Guarantees.Isa.Sm90Sheet := by
  intro R T L U F X _ _ _ σ Z A H hread hsplit hcomp hlab hmma hwgmma hlane huni hmem hbulk htex hwarp
  exact IsaSheet Sm90.row σ Z A H sm90_ok hread hsplit hcomp hlab hmma hwgmma
    (fun k hk => absurd hk (sm90_fam k).1) (fun k hk => absurd hk (sm90_fam k).2) hlane huni hmem hbulk htex hwarp

/-- Every sm_100a row passes the discipline. -/
theorem sm100_ok : ∀ k, (Sm100.row k).ok = true
  | .sync s => sync_ok s
  | .tc05 t => tc05_ok t
  | .tmem m => tmem_ok m
  | .base c => base_ok c

/-- sm_100a has no wgmma class. -/
theorem sm100_fam : ∀ k : Sm100, (Sm100.row k).fam ≠ .wgmma
  | .sync s => ne_of_fam (sync_fam s) (by decide)
  | .tc05 t => ne_of_fam (tc05_fam t) (by decide)
  | .tmem m => ne_of_fam (tmem_fam m) (by decide)
  | .base c => (base_fam c).2.1

/-- **A6 and A7 on sm_100a.** -/
theorem Sm100Sheet : Accounting.Guarantees.Isa.Sm100Sheet := by
  intro R T L U F X _ _ _ σ Z A H hread hsplit hcomp hlab hmma htc htmem hlane huni hmem hbulk htex hwarp
  exact IsaSheet Sm100.row σ Z A H sm100_ok hread hsplit hcomp hlab hmma (fun k hk => absurd hk (sm100_fam k)) htc
    htmem hlane huni hmem hbulk htex hwarp

end Accounting.SecurityProofs.Isa
