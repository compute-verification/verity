module

public import NciCheck.Record
public import Mathlib.Tactic
public import Proofs.Properties.Nci.RecordGrid

/-!
# The slots the record check's summary tries reach every slot's tally

`summarize` takes the greatest of each per-slot count over the slots of `slotsToTry`, not over every slot. That is
enough:

* `inSlot_const`: between two marks in order (`marks`), a template's units in a slot do not change with the slot.
  `below S q x s` is `q` for a slot under `x % (q · S) / q`, the rest at it and 0 past it, so it changes only at
  that slot and the next, which are the marks of each end of each grid (`below_const`).
* `slotsToTry_reaches`: every slot's tally is the tally of a slot `slotsToTry` lists: the slot itself where it lists
  every slot, else the greatest listed slot at most it, with no mark of any declaration between the two.
* `fresh_ge`: the summary's fresh bits are at least every slot's input bits less its `inside` and `theta` bits.
-/

@[expose] public section

namespace NciCheck.Record.SecurityProofs

open RecordText

/-- `min q (r - s · q)` as a function of the slot `s` changes only at `r / q` and at the next slot. -/
private theorem min_sub_const (q r s s' : ℕ) (hss : s ≤ s') (h1 : ¬ (s < r / q ∧ r / q ≤ s'))
    (h2 : ¬ (s < r / q + 1 ∧ r / q + 1 ≤ s')) : min q (r - s * q) = min q (r - s' * q) := by
  rcases Nat.eq_zero_or_pos q with hq | hq
  · subst hq
    rw [Nat.zero_min, Nat.zero_min]
  · have hdm : r / q * q + r % q = r := by rw [Nat.mul_comm]; exact Nat.div_add_mod r q
    have hlt := Nat.mod_lt r hq
    have hmono : s * q ≤ s' * q := Nat.mul_le_mul_right q hss
    generalize r / q = a at h1 h2 hdm
    by_cases hsa : s < a
    · have e := Nat.mul_le_mul_right q (show s' + 1 ≤ a by omega)
      rw [Nat.add_mul, Nat.one_mul] at e
      omega
    · by_cases hsa' : s = a
      · have hs' : s' = s := by omega
        rw [hs']
      · have e := Nat.mul_le_mul_right q (show a + 1 ≤ s by omega)
        rw [Nat.add_mul, Nat.one_mul] at e
        omega

/-- `below S q x` as a function of the slot changes only at `x % (q · S) / q` and at the next slot. -/
theorem below_const (S q x s s' : ℕ) (hss : s ≤ s')
    (h1 : ¬ (s < x % (q * S) / q ∧ x % (q * S) / q ≤ s'))
    (h2 : ¬ (s < x % (q * S) / q + 1 ∧ x % (q * S) / q + 1 ≤ s')) : below S q x s = below S q x s' := by
  unfold below
  rw [min_sub_const q (x % (q * S)) s s' hss h1 h2]

/-- Between two marks `inSlot` does not change with the slot. -/
theorem inSlot_const (S : ℕ) (gs : List Grid) (lo hi s s' : ℕ) (hss : s ≤ s')
    (hm : ∀ m ∈ marks S gs lo hi, ¬ (s < m ∧ m ≤ s')) :
    inSlot S gs lo hi s = inSlot S gs lo hi s' := by
  unfold inSlot
  congr 1
  apply List.map_congr_left
  intro g hg
  have key : ∀ m, m ∈ [(max lo g.start - g.start) % (g.q * S) / g.q,
      (max lo g.start - g.start) % (g.q * S) / g.q + 1,
      (min hi (g.start + g.units) - g.start) % (g.q * S) / g.q,
      (min hi (g.start + g.units) - g.start) % (g.q * S) / g.q + 1] → ¬ (s < m ∧ m ≤ s') :=
    fun m hmem => hm m (List.mem_flatMap.2 ⟨g, hg, hmem⟩)
  rw [below_const S g.q (min hi (g.start + g.units) - g.start) s s' hss
      (key _ (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ (List.mem_cons_self ..))))
      (key _ (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ (List.mem_cons_self ..))))),
    below_const S g.q (max lo g.start - g.start) s s' hss
      (key _ (List.mem_cons_self ..)) (key _ (List.mem_cons_of_mem _ (List.mem_cons_self ..)))]

/-- A list with a member at most `s` has a greatest such member. -/
private theorem exists_greatest_le (l : List ℕ) : ∀ s : ℕ, (∃ m ∈ l, m ≤ s) →
    ∃ m ∈ l, m ≤ s ∧ ∀ m' ∈ l, m' ≤ s → m' ≤ m
  | 0, ⟨m, hm, hms⟩ => ⟨m, hm, hms, fun m' _ h => by omega⟩
  | s + 1, ⟨m, hm, hms⟩ => by
    by_cases h : s + 1 ∈ l
    · exact ⟨s + 1, h, Nat.le_refl _, fun m' _ h' => h'⟩
    · have hne : ∀ m' ∈ l, m' ≤ s + 1 → m' ≤ s := fun m' hm' h' => by
        have : m' ≠ s + 1 := fun e => h (e ▸ hm')
        omega
      obtain ⟨m₀, hm₀, hm₀s, hmax⟩ := exists_greatest_le l s ⟨m, hm, hne m hm hms⟩
      exact ⟨m₀, hm₀, by omega, fun m' hm' h' => hmax m' hm' (hne m' hm' h')⟩

/-- Two slots in which every declaration has as many reader units have one tally. -/
theorem tally_congr (S : ℕ) (ds : List Counted) (s s' : ℕ)
    (h : ∀ x ∈ ds, inSlot S x.2.1 x.2.2.w.start (x.2.2.w.start + x.2.2.w.readerUnits) s =
      inSlot S x.2.1 x.2.2.w.start (x.2.2.w.start + x.2.2.w.readerUnits) s') :
    tally S ds s = tally S ds s' := by
  unfold tally
  generalize Array.replicate 7 0 = acc
  induction ds generalizing acc with
  | nil => rfl
  | cons x ds ih =>
    obtain ⟨k, gs, d⟩ := x
    have hx : inSlot S gs d.w.start (d.w.start + d.w.readerUnits) s =
        inSlot S gs d.w.start (d.w.start + d.w.readerUnits) s' := h (k, gs, d) (List.mem_cons_self ..)
    simp only [List.foldl_cons]
    rw [hx]
    exact ih (fun y hy => h y (List.mem_cons_of_mem _ hy)) _

/-- Every slot's tally is the tally of a slot `summarize` tries. -/
theorem slotsToTry_reaches (S : ℕ) (ds : List Counted) (s : ℕ) (hs : s < S) :
    ∃ s' ∈ slotsToTry S ds, tally S ds s' = tally S ds s := by
  unfold slotsToTry
  dsimp only
  split
  · exact ⟨s, List.mem_range.2 hs, rfl⟩
  · obtain ⟨m, hm, hms, hmax⟩ := exists_greatest_le
      (0 :: (ds.flatMap fun (_, gs, d) => marks S gs d.w.start (d.w.start + d.w.readerUnits)).filter (· < S)) s
      ⟨0, List.mem_cons_self .., Nat.zero_le _⟩
    refine ⟨m, hm, tally_congr S ds m s fun x hx => inSlot_const S _ _ _ m s hms fun k hk hk' => ?_⟩
    obtain ⟨k₀, gs, d⟩ := x
    have hmem : k ∈ 0 :: (ds.flatMap fun (_, gs, d) =>
        marks S gs d.w.start (d.w.start + d.w.readerUnits)).filter (· < S) :=
      List.mem_cons_of_mem _ (List.mem_filter.2 ⟨List.mem_flatMap.2 ⟨(k₀, gs, d), hx, hk⟩,
        decide_eq_true (show k < S by omega)⟩)
    have := hmax k hmem hk'.2
    omega

/-- A fold of `max` is at least its start and at least each member. -/
private theorem le_foldl_max (l : List ℕ) : ∀ a : ℕ, a ≤ l.foldl max a ∧ ∀ x ∈ l, x ≤ l.foldl max a := by
  induction l with
  | nil => exact fun a => ⟨Nat.le_refl _, fun x hx => by cases hx⟩
  | cons b l ih =>
    intro a
    have h := ih (max a b)
    rw [List.foldl_cons]
    refine ⟨Nat.le_trans (Nat.le_max_left a b) h.1, fun x hx => ?_⟩
    rcases List.mem_cons.1 hx with rfl | hx
    · exact Nat.le_trans (Nat.le_max_right a x) h.1
    · exact h.2 x hx

/-- The summary's fresh bits are at least every slot's. -/
theorem fresh_ge (c : Case) (s : ℕ) (hs : s < c.S) :
    (summarize c).inputs
        - (tally c.S (c.x.decls.map fun d => (c.kind (some d.fill), c.gridsAt d.w.reader, d)) s)[Kind.inside.index]!
        - (tally c.S (c.x.decls.map fun d => (c.kind (some d.fill), c.gridsAt d.w.reader, d)) s)[Kind.theta.index]!
      ≤ (summarize c).fresh := by
  obtain ⟨s', hs', he⟩ := slotsToTry_reaches c.S
    (c.x.decls.map fun d => (c.kind (some d.fill), c.gridsAt d.w.reader, d)) s hs
  rw [← he]
  exact (le_foldl_max _ 0).2 _ (List.mem_map.2 ⟨_, List.mem_map.2 ⟨s', hs', rfl⟩, rfl⟩)

end NciCheck.Record.SecurityProofs
