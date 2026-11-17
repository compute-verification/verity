module

public import InferenceOnlyCheck.Record
public import Mathlib.Tactic
public import Spot.WiresSpan
public import Proofs.Properties.InferenceOnly.RecordGrid

/-! `sameSlot` decides without visiting a wire's rows that every row it covers is read from the reader's own slot, and
is sound (`sameSlot_sound`). -/

@[expose] public section

namespace InferenceOnlyCheck.Record.SecurityProofs

open Spot.RecordText
open Spot.Wires (Wire accum prodR)

def BoxRow : ℕ → ℤ → List Digit → ℕ → ℤ → Prop
  | start, base, [], u, y => u = start ∧ y = base
  | start, base, d :: ds, u, y => ∃ v, v < d.r ∧ BoxRow (start + v * d.p) (base + (v : ℤ) * d.s) ds u y

private theorem foldl_add (l : List ℤ) : ∀ a : ℤ, l.foldl (· + ·) a = a + l.foldl (· + ·) 0 := by
  induction l with
  | nil => intro a; exact (Int.add_zero a).symm
  | cons x l ih =>
    intro a
    rw [List.foldl_cons, List.foldl_cons, ih (a + x), ih (0 + x)]
    omega

private theorem foldl_add_cons (x : ℤ) (l : List ℤ) : (x :: l).foldl (· + ·) 0 = x + l.foldl (· + ·) 0 := by
  rw [List.foldl_cons, foldl_add l (0 + x)]
  omega

private theorem foldl_add_nonneg (l : List ℤ) (h : ∀ x ∈ l, 0 ≤ x) : 0 ≤ l.foldl (· + ·) 0 := by
  induction l with
  | nil => exact Int.le_refl 0
  | cons x l ih =>
    have h1 := h x (List.mem_cons_self ..)
    have h2 := ih fun y hy => h y (List.mem_cons_of_mem _ hy)
    rw [foldl_add_cons]
    omega

private theorem foldl_add_nonpos (l : List ℤ) (h : ∀ x ∈ l, x ≤ 0) : l.foldl (· + ·) 0 ≤ 0 := by
  induction l with
  | nil => exact Int.le_refl 0
  | cons x l ih =>
    have h1 := h x (List.mem_cons_self ..)
    have h2 := ih fun y hy => h y (List.mem_cons_of_mem _ hy)
    rw [foldl_add_cons]
    omega

private theorem reach_nonneg (q : ℤ) (ds : List Digit) (hdig : ∀ d ∈ ds, 1 ≤ d.r) :
    0 ≤ (ds.map fun d => if (d.p : ℤ) % q ≠ 0 then (d.p : ℤ) * ((d.r : ℤ) - 1) else 0).foldl (· + ·) 0 :=
  foldl_add_nonneg _ fun z hz => by
    obtain ⟨d, hd, rfl⟩ := List.mem_map.1 hz
    have := hdig d hd
    split
    · exact Int.mul_nonneg (by omega) (by omega)
    · exact Int.le_refl 0

private theorem low_nonpos (q : ℤ) (ds : List Digit) :
    (ds.map fun d => if d.s % q ≠ 0 then min 0 (d.s * ((d.r : ℤ) - 1)) else 0).foldl (· + ·) 0 ≤ 0 :=
  foldl_add_nonpos _ fun z hz => by
    obtain ⟨d, _, rfl⟩ := List.mem_map.1 hz
    split
    · exact Int.min_le_left 0 _
    · exact Int.le_refl 0

private theorem high_nonneg (q : ℤ) (ds : List Digit) :
    0 ≤ (ds.map fun d => if d.s % q ≠ 0 then max 0 (d.s * ((d.r : ℤ) - 1)) else 0).foldl (· + ·) 0 :=
  foldl_add_nonneg _ fun z hz => by
    obtain ⟨d, _, rfl⟩ := List.mem_map.1 hz
    split
    · exact Int.le_max_left 0 _
    · exact Int.le_refl 0

private theorem step_reader (q x p H : ℤ) (v r : ℕ) (hq : 0 < q) (hp : 0 ≤ p) (hv : v < r) (hH : 0 ≤ H)
    (h : x % q + ((if p % q ≠ 0 then p * ((r : ℤ) - 1) else 0) + H) < q) :
    (x + (v : ℤ) * p) / q = x / q + (v : ℤ) * (if p % q = 0 then p / q else 0) ∧
      (x + (v : ℤ) * p) % q + H < q := by
  have hq0 : q ≠ 0 := by omega
  by_cases hpq : p % q = 0
  · have hp' : q * (p / q) = p := Int.mul_ediv_cancel' (Int.dvd_of_emod_eq_zero hpq)
    rw [ite_eq_left hpq]
    rw [ite_eq_right (fun hne : p % q ≠ 0 => hne hpq)] at h
    have e : x + (v : ℤ) * p = x + q * ((v : ℤ) * (p / q)) := by
      rw [← Int.mul_assoc, Int.mul_comm q (v : ℤ), Int.mul_assoc, hp']
    rw [e, Int.add_mul_ediv_left _ _ hq0, Int.add_mul_emod_self_left]
    exact ⟨rfl, by omega⟩
  · rw [ite_eq_right hpq]
    rw [ite_eq_left hpq] at h
    have h0 : 0 ≤ (v : ℤ) * p := Int.mul_nonneg (by omega) hp
    have h1 : (v : ℤ) * p ≤ p * ((r : ℤ) - 1) := by
      rw [Int.mul_comm p]
      exact Int.mul_le_mul_of_nonneg_right (by omega) hp
    have hx0 := Int.emod_nonneg x hq0
    have hx := Int.emod_add_mul_ediv x q
    obtain ⟨e1, e2⟩ := (Int.ediv_emod_unique (a := x + (v : ℤ) * p) (q := x / q) (r := x % q + (v : ℤ) * p) hq).2
      ⟨by omega, by omega, by omega⟩
    rw [e1, e2]
    exact ⟨by omega, by omega⟩

private theorem step_writer (q y s L H : ℤ) (v r : ℕ) (hq : 0 < q) (hv : v < r) (hL : L ≤ 0) (hH : 0 ≤ H)
    (hlo : 0 ≤ y % q + ((if s % q ≠ 0 then min 0 (s * ((r : ℤ) - 1)) else 0) + L))
    (hhi : y % q + ((if s % q ≠ 0 then max 0 (s * ((r : ℤ) - 1)) else 0) + H) < q) :
    (y + (v : ℤ) * s) / q = y / q + (v : ℤ) * (if s % q = 0 then s / q else 0) ∧
      0 ≤ (y + (v : ℤ) * s) % q + L ∧ (y + (v : ℤ) * s) % q + H < q := by
  have hq0 : q ≠ 0 := by omega
  by_cases hsq : s % q = 0
  · have hs' : q * (s / q) = s := Int.mul_ediv_cancel' (Int.dvd_of_emod_eq_zero hsq)
    rw [ite_eq_left hsq]
    rw [ite_eq_right (fun hne : s % q ≠ 0 => hne hsq)] at hlo hhi
    have e : y + (v : ℤ) * s = y + q * ((v : ℤ) * (s / q)) := by
      rw [← Int.mul_assoc, Int.mul_comm q (v : ℤ), Int.mul_assoc, hs']
    rw [e, Int.add_mul_ediv_left _ _ hq0, Int.add_mul_emod_self_left]
    exact ⟨rfl, by omega, by omega⟩
  · rw [ite_eq_right hsq]
    rw [ite_eq_left hsq] at hlo hhi
    have hb := Spot.Wires.digit_bounds v r s hv
    have hy := Int.emod_add_mul_ediv y q
    obtain ⟨e1, e2⟩ := (Int.ediv_emod_unique (a := y + (v : ℤ) * s) (q := y / q) (r := y % q + (v : ℤ) * s) hq).2
      ⟨by omega, by omega, by omega⟩
    rw [e1, e2]
    exact ⟨by omega, by omega, by omega⟩

private theorem leaf_rows (S : ℕ) (qr qw : ℤ) (hqr : 0 < qr) (hqw : 0 < qw) :
    ∀ (ds : List Digit) (start : ℕ) (base x y : ℤ) (u : ℕ) (yw : ℤ),
      (∀ d ∈ ds, 1 ≤ d.r) →
      x % qr + (ds.map fun d => if (d.p : ℤ) % qr ≠ 0 then (d.p : ℤ) * ((d.r : ℤ) - 1) else 0).foldl (· + ·) 0
        < qr →
      0 ≤ y % qw + (ds.map fun d => if d.s % qw ≠ 0 then min 0 (d.s * ((d.r : ℤ) - 1)) else 0).foldl (· + ·) 0 →
      y % qw + (ds.map fun d => if d.s % qw ≠ 0 then max 0 (d.s * ((d.r : ℤ) - 1)) else 0).foldl (· + ·) 0 < qw →
      (x / qr - y / qw) % (S : ℤ) = 0 →
      (∀ d ∈ ds, ((if (d.p : ℤ) % qr = 0 then (d.p : ℤ) / qr else 0) -
        if d.s % qw = 0 then d.s / qw else 0) % (S : ℤ) = 0) →
      BoxRow start base ds u yw →
      ((x + ((u : ℤ) - start)) / qr - (y + (yw - base)) / qw) % (S : ℤ) = 0 := by
  intro ds
  induction ds with
  | nil =>
    intro start base x y u yw _ _ _ _ h4 _ hrow
    obtain ⟨rfl, rfl⟩ := hrow
    rw [Int.sub_self, Int.sub_self, Int.add_zero, Int.add_zero]
    exact h4
  | cons d ds ih =>
    intro start base x y u yw hdig h1 h2 h3 h4 h5 hrow
    obtain ⟨v, hv, hrow⟩ := hrow
    rw [List.map_cons, foldl_add_cons] at h1 h2 h3
    have hdig' : ∀ d' ∈ ds, 1 ≤ d'.r := fun d' hd' => hdig d' (List.mem_cons_of_mem _ hd')
    have hH := reach_nonneg qr ds hdig'
    have hL := low_nonpos qw ds
    have hH' := high_nonneg qw ds
    obtain ⟨er, hr'⟩ := step_reader qr x d.p _ v d.r hqr (by omega) hv hH h1
    obtain ⟨ew, hw1, hw2⟩ := step_writer qw y d.s _ _ v d.r hqw hv hL hH' h2 h3
    have h4' : ((x + (v : ℤ) * (d.p : ℤ)) / qr - (y + (v : ℤ) * d.s) / qw) % (S : ℤ) = 0 := by
      have a := Int.dvd_of_emod_eq_zero h4
      have b := Int.dvd_of_emod_eq_zero (h5 d (List.mem_cons_self ..))
      rw [er, ew]
      generalize (if (d.p : ℤ) % qr = 0 then (d.p : ℤ) / qr else 0) = dr at b ⊢
      generalize (if d.s % qw = 0 then d.s / qw else 0) = dw at b ⊢
      have e : x / qr + (v : ℤ) * dr - (y / qw + (v : ℤ) * dw) = x / qr - y / qw + (v : ℤ) * (dr - dw) := by
        rw [Int.mul_sub]; omega
      rw [e]
      exact Int.emod_eq_zero_of_dvd (Int.dvd_add a (Int.dvd_trans b (Int.dvd_mul_left _ _)))
    have key := ih (start + v * d.p) (base + (v : ℤ) * d.s) (x + (v : ℤ) * (d.p : ℤ)) (y + (v : ℤ) * d.s) u yw
      hdig' hr' hw1 hw2 h4' (fun d' hd' => h5 d' (List.mem_cons_of_mem _ hd')) hrow
    have e1 : x + (v : ℤ) * (d.p : ℤ) + ((u : ℤ) - ((start + v * d.p : ℕ) : ℤ)) = x + ((u : ℤ) - start) := by
      rw [Int.natCast_add, Int.natCast_mul]; omega
    have e2 : y + (v : ℤ) * d.s + (yw - (base + (v : ℤ) * d.s)) = y + (yw - base) := by omega
    rw [e1, e2] at key
    exact key

theorem leaf_sound (S : ℕ) (gr gw : Grid) (b : Box) (hdig : ∀ d ∈ b.digs, 1 ≤ d.r)
    (h : leaf S gr gw b = true) (hgr : gr.start ≤ b.start) (u : ℕ) (y : ℤ)
    (hrow : BoxRow b.start b.base b.digs u y) (hu : gr.start ≤ u) (hy : (gw.start : ℤ) ≤ y) :
    (u - gr.start) / gr.q % S = (y.toNat - gw.start) / gw.q % S := by
  simp only [leaf, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, bne_iff_ne, List.all_eq_true] at h
  obtain ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩ := h
  have hqr : 0 < (gr.q : ℤ) := by
    rcases Nat.eq_zero_or_pos gr.q with h0 | h0
    · have e : (gr.q : ℤ) = 0 := by omega
      have hH := reach_nonneg 0 b.digs hdig
      rw [e, Int.emod_zero] at h1
      omega
    · omega
  have hqw : 0 < (gw.q : ℤ) := by
    rcases Nat.eq_zero_or_pos gw.q with h0 | h0
    · have e : (gw.q : ℤ) = 0 := by omega
      have hL := low_nonpos 0 b.digs
      have hH := high_nonneg 0 b.digs
      rw [e, Int.emod_zero] at h2 h3
      omega
    · omega
  have key := leaf_rows S gr.q gw.q hqr hqw b.digs b.start b.base ((b.start : ℤ) - gr.start)
    (b.base - gw.start) u y hdig h1 h2 h3 h4 h5 hrow
  have e1 : (b.start : ℤ) - gr.start + ((u : ℤ) - b.start) = ((u - gr.start : ℕ) : ℤ) := by omega
  have e2 : b.base - (gw.start : ℤ) + (y - b.base) = ((y.toNat - gw.start : ℕ) : ℤ) := by omega
  rw [e1, e2] at key
  have key' := Int.emod_eq_emod_iff_emod_sub_eq_zero.2 key
  rw [← Int.natCast_ediv, ← Int.natCast_ediv, ← Int.natCast_emod, ← Int.natCast_emod] at key'
  exact Int.ofNat.inj key'

private theorem row_bounds : ∀ (ds : List Digit) (start : ℕ) (base : ℤ) (u : ℕ) (y : ℤ), (∀ d ∈ ds, 1 ≤ d.r) →
    BoxRow start base ds u y →
    (start : ℤ) ≤ u ∧ (u : ℤ) ≤ start + (ds.map fun d => (d.p : ℤ) * ((d.r : ℤ) - 1)).foldl (· + ·) 0 ∧
      base + (ds.map fun d => min 0 (d.s * ((d.r : ℤ) - 1))).foldl (· + ·) 0 ≤ y ∧
      y ≤ base + (ds.map fun d => max 0 (d.s * ((d.r : ℤ) - 1))).foldl (· + ·) 0 := by
  intro ds
  induction ds with
  | nil =>
    intro start base u y _ hrow
    obtain ⟨rfl, rfl⟩ := hrow
    simp only [List.map_nil, List.foldl_nil]
    omega
  | cons d ds ih =>
    intro start base u y hdig hrow
    obtain ⟨v, hv, hrow⟩ := hrow
    have h := ih (start + v * d.p) (base + (v : ℤ) * d.s) u y
      (fun d' hd' => hdig d' (List.mem_cons_of_mem _ hd')) hrow
    rw [Int.natCast_add, Int.natCast_mul] at h
    have h0 : 0 ≤ (v : ℤ) * (d.p : ℤ) := Int.mul_nonneg (by omega) (by omega)
    have h1 : (v : ℤ) * (d.p : ℤ) ≤ (d.p : ℤ) * ((d.r : ℤ) - 1) := by
      rw [Int.mul_comm (d.p : ℤ)]
      exact Int.mul_le_mul_of_nonneg_right (by omega) (by omega)
    have hb := Spot.Wires.digit_bounds v d.r d.s hv
    rw [List.map_cons, foldl_add_cons, List.map_cons, foldl_add_cons, List.map_cons, foldl_add_cons]
    omega

def BoxInside (S : ℕ) (gr gw : List Grid) (b : Box) : Prop :=
  ∀ u y, BoxRow b.start b.base b.digs u y →
    ∀ s s', slotOf S gr u = some s → slotOf S gw y.toNat = some s' → s = s'

private theorem shown_inside (S : ℕ) (gr gw : List Grid) (hr : OneGrid gr) (hw : OneGrid gw) (b : Box)
    (hdig : ∀ d ∈ b.digs, 1 ≤ d.r) (r0 w0 : Grid)
    (hR : gridFor gr b.start
      (b.start + (1 + (b.digs.map fun d => (d.p : ℤ) * ((d.r : ℤ) - 1)).foldl (· + ·) 0)) = some r0)
    (hW : gridFor gw (b.base + (b.digs.map fun d => min 0 (d.s * ((d.r : ℤ) - 1))).foldl (· + ·) 0)
      (b.base + (b.digs.map fun d => max 0 (d.s * ((d.r : ℤ) - 1))).foldl (· + ·) 0 + 1) = some w0)
    (hleaf : leaf S r0 w0 b = true) : BoxInside S gr gw b := by
  intro u y hrow s s' hs hs'
  obtain ⟨hb1, hb2, hb3, hb4⟩ := row_bounds b.digs b.start b.base u y hdig hrow
  unfold gridFor at hR hW
  have hRm := List.mem_of_find?_eq_some hR
  have hRp := List.find?_some hR
  have hWm := List.mem_of_find?_eq_some hW
  have hWp := List.find?_some hW
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hRp hWp
  obtain ⟨hR1, hR2⟩ := hRp
  obtain ⟨hW1, hW2⟩ := hWp
  rw [Int.natCast_add] at hR2 hW2
  rw [slotOf_of_mem S gr u hr r0 hRm (by omega) (by omega)] at hs
  rw [slotOf_of_mem S gw y.toNat hw w0 hWm (by omega) (by omega)] at hs'
  have e := leaf_sound S r0 w0 b hdig hleaf (by omega) u y hrow (by omega) (by omega)
  have e1 := Option.some.inj hs
  have e2 := Option.some.inj hs'
  rw [← e1, ← e2]
  exact e

theorem row_split : ∀ (k : ℕ) (ds : List Digit) (hk : k < ds.length) (start : ℕ) (base : ℤ) (u : ℕ) (y : ℤ),
    BoxRow start base ds u y →
    ∃ v, v < ds[k].r ∧
      BoxRow (start + v * ds[k].p) (base + (v : ℤ) * ds[k].s) (ds.take k ++ ds.drop (k + 1)) u y := by
  intro k
  induction k with
  | zero =>
    intro ds hk start base u y hrow
    cases ds with
    | nil => cases hk
    | cons d ds =>
      obtain ⟨v, hv, hrow⟩ := hrow
      exact ⟨v, hv, hrow⟩
  | succ k ih =>
    intro ds hk start base u y hrow
    cases ds with
    | nil => cases hk
    | cons d ds =>
      obtain ⟨v0, hv0, hrow⟩ := hrow
      have hk' : k < ds.length := Nat.lt_of_succ_lt_succ hk
      obtain ⟨v, hv, hrow'⟩ := ih ds hk' (start + v0 * d.p) (base + (v0 : ℤ) * d.s) u y hrow
      have e1 : start + v0 * d.p + v * ds[k].p = start + v * ds[k].p + v0 * d.p := by omega
      have e2 : base + (v0 : ℤ) * d.s + (v : ℤ) * ds[k].s = base + (v : ℤ) * ds[k].s + (v0 : ℤ) * d.s := by omega
      rw [e1, e2] at hrow'
      exact ⟨v, hv, v0, hv0, hrow'⟩

private theorem search_step (S : ℕ) (gr gw : List Grid) (fuel : ℕ) (b : Box) (todo : List Box)
    (h : search S gr gw (fuel + 1) (b :: todo) = true) :
    (∃ r0 w0,
      gridFor gr b.start
        (b.start + (1 + (b.digs.map fun d => (d.p : ℤ) * ((d.r : ℤ) - 1)).foldl (· + ·) 0)) = some r0 ∧
      gridFor gw (b.base + (b.digs.map fun d => min 0 (d.s * ((d.r : ℤ) - 1))).foldl (· + ·) 0)
        (b.base + (b.digs.map fun d => max 0 (d.s * ((d.r : ℤ) - 1))).foldl (· + ·) 0 + 1) = some w0 ∧
      leaf S r0 w0 b = true ∧ search S gr gw fuel todo = true) ∨
    ∃ k, k < b.digs.length ∧
      search S gr gw fuel (((List.range (b.digs[k]!).r).map fun v =>
        ({ start := b.start + v * (b.digs[k]!).p, base := b.base + (v : ℤ) * (b.digs[k]!).s,
           digs := b.digs.take k ++ b.digs.drop (k + 1) } : Box)) ++ todo) = true := by
  simp only [search] at h
  generalize hR : gridFor gr b.start
    (b.start + (1 + (b.digs.map fun d => (d.p : ℤ) * ((d.r : ℤ) - 1)).foldl (· + ·) 0)) = R at h ⊢
  generalize hW : gridFor gw (b.base + (b.digs.map fun d => min 0 (d.s * ((d.r : ℤ) - 1))).foldl (· + ·) 0)
    (b.base + (b.digs.map fun d => max 0 (d.s * ((d.r : ℤ) - 1))).foldl (· + ·) 0 + 1) = Wg at h ⊢
  split at h
  · rename_i r0 w0
    split at h
    · rename_i hleaf
      exact Or.inl ⟨r0, w0, rfl, rfl, hleaf, h⟩
    · split at h
      · exact Bool.noConfusion h
      · rename_i k heq
        split at h
        · exact Bool.noConfusion h
        · have hk := List.mem_range.1 (List.mem_of_find?_eq_some heq)
          rw [List.length_map] at hk
          exact Or.inr ⟨k, hk, h⟩
  · split at h
    · rename_i hc
      exact Bool.noConfusion hc
    · split at h
      · exact Bool.noConfusion h
      · rename_i k heq
        split at h
        · exact Bool.noConfusion h
        · have hk := List.mem_range.1 (List.mem_of_find?_eq_some heq)
          rw [List.length_map] at hk
          exact Or.inr ⟨k, hk, h⟩

theorem search_sound (S : ℕ) (gr gw : List Grid) (hr : OneGrid gr) (hw : OneGrid gw) :
    ∀ (fuel : ℕ) (stack : List Box), search S gr gw fuel stack = true →
      (∀ b ∈ stack, ∀ d ∈ b.digs, 1 ≤ d.r) → ∀ b ∈ stack, BoxInside S gr gw b := by
  intro fuel
  induction fuel with
  | zero =>
    intro stack h _ b hb
    cases stack with
    | nil => cases hb
    | cons b0 todo => exact Bool.noConfusion (show false = true from h)
  | succ fuel ih =>
    intro stack h hdig b hb
    cases stack with
    | nil => cases hb
    | cons b0 todo =>
      have hdig0 := hdig b0 (List.mem_cons_self ..)
      rcases search_step S gr gw fuel b0 todo h with ⟨r0, w0, hR, hW, hleaf, hrest⟩ | ⟨k, hk, hrest⟩
      · rcases List.mem_cons.1 hb with rfl | hb
        · exact shown_inside S gr gw hr hw b hdig0 r0 w0 hR hW hleaf
        · exact ih todo hrest (fun b' hb' => hdig b' (List.mem_cons_of_mem _ hb')) b hb
      · have hall := ih _ hrest (by
          intro b' hb' d hd
          rcases List.mem_append.1 hb' with hb' | hb'
          · obtain ⟨v, _, rfl⟩ := List.mem_map.1 hb'
            rcases List.mem_append.1 hd with hd | hd
            · exact hdig0 d (List.mem_of_mem_take hd)
            · exact hdig0 d (List.mem_of_mem_drop hd)
          · exact hdig b' (List.mem_cons_of_mem _ hb') d hd)
        rcases List.mem_cons.1 hb with rfl | hb
        · intro u y hrow s s' hs hs'
          obtain ⟨v, hv, hrow'⟩ := row_split k b.digs hk b.start b.base u y hrow
          rw [← getElem!_pos b.digs k hk] at hv hrow'
          exact hall _ (List.mem_append_left _ (List.mem_map.2 ⟨v, List.mem_range.2 hv, rfl⟩)) u y hrow' s s' hs hs'
        · exact hall b (List.mem_append_right _ hb)

private theorem foldl_mul (l : List ℕ) : ∀ a : ℕ, l.foldl (· * ·) a = a * l.foldl (· * ·) 1 := by
  induction l with
  | nil => intro a; exact (Nat.mul_one a).symm
  | cons x l ih =>
    intro a
    rw [List.foldl_cons, List.foldl_cons, ih (a * x), ih (1 * x), Nat.one_mul, Nat.mul_assoc]

private theorem foldl_mul_cons (x : ℕ) (l : List ℕ) : (x :: l).foldl (· * ·) 1 = x * l.foldl (· * ·) 1 := by
  rw [List.foldl_cons, foldl_mul, Nat.one_mul]

private theorem prodR_concat (r : ℕ) (s : ℤ) (ps : List (ℕ × ℤ)) : prodR (ps ++ [(r, s)]) = prodR ps * r := by
  induction ps with
  | nil =>
    show r * 1 = 1 * r
    rw [Nat.mul_one, Nat.one_mul]
  | cons q ps ih =>
    show q.1 * prodR (ps ++ [(r, s)]) = q.1 * prodR ps * r
    rw [ih, Nat.mul_assoc]

private theorem accum_concat (r : ℕ) (s : ℤ) : ∀ (ps : List (ℕ × ℤ)) (i : ℕ),
    accum (ps ++ [(r, s)]) i = accum ps i + ((i / prodR ps % r : ℕ) : ℤ) * s := by
  intro ps
  induction ps with
  | nil =>
    intro i
    show ((i % r : ℕ) : ℤ) * s + 0 = 0 + ((i / 1 % r : ℕ) : ℤ) * s
    rw [Nat.div_one, Int.add_zero, Int.zero_add]
  | cons q ps ih =>
    intro i
    show ((i % q.1 : ℕ) : ℤ) * q.2 + accum (ps ++ [(r, s)]) (i / q.1) =
      ((i % q.1 : ℕ) : ℤ) * q.2 + accum ps (i / q.1) + ((i / (q.1 * prodR ps) % r : ℕ) : ℤ) * s
    rw [ih, Nat.div_div_eq_div_mul, Int.add_assoc]

private theorem prodR_zip : ∀ (rs : List ℕ) (ss : List ℤ), ss.length = rs.length →
    prodR (rs.zip ss).reverse = rs.foldl (· * ·) 1
  | [], _, _ => by rw [List.zip_nil_left]; rfl
  | r :: rs, [], h => by
    rw [List.length_nil, List.length_cons] at h
    omega
  | r :: rs, s :: ss, h => by
    rw [List.length_cons, List.length_cons] at h
    rw [List.zip_cons_cons, List.reverse_cons, prodR_concat, prodR_zip rs ss (by omega), foldl_mul_cons,
      Nat.mul_comm]

private theorem prodR_drop (radix : List ℕ) (step : List ℤ) (hlen : step.length = radix.length) (b : ℕ) :
    prodR ((radix.zip step).drop b).reverse = (radix.drop b).foldl (· * ·) 1 := by
  have e : (radix.zip step).drop b = (radix.drop b).zip (step.drop b) := by
    unfold List.zip
    exact List.drop_zipWith
  rw [e, prodR_zip _ _ (by rw [List.length_drop, List.length_drop, hlen])]

private def strideAt (radix : List ℕ) (c k : ℕ) : ℕ :=
  if k < radix.length - (if c > 1 then 1 else 0) then
    ((radix.take (radix.length - (if c > 1 then 1 else 0))).drop (k + 1)).foldl (· * ·) 1
  else 0

private def digitAt (w : Wire) (k : ℕ) : Option Digit :=
  if w.radix[k]! > 1 then some { r := w.radix[k]!, p := strideAt w.radix w.cuts k, s := w.step[k]! } else none

private theorem digitsOf_eq (w : Wire) : digitsOf w = (List.range w.radix.length).filterMap (digitAt w) := rfl

private theorem stride_spec (radix : List ℕ) (c : ℕ) (hc : 1 ≤ c) (hlast : c = 1 ∨ radix.getLast? = some c)
    (k : ℕ) (hk : k < radix.length) :
    (radix.drop (k + 1)).foldl (· * ·) 1 = c * strideAt radix c k ∨
      ((radix.drop (k + 1)).foldl (· * ·) 1 = 1 ∧ strideAt radix c k = 0 ∧ radix[k] = c) := by
  unfold strideAt
  by_cases h1 : c > 1
  · rcases hlast with hlast | hlast
    · omega
    · obtain ⟨ys, rfl⟩ := List.getLast?_eq_some_iff.1 hlast
      rw [List.length_append, List.length_singleton] at hk
      rw [ite_eq_left h1, List.length_append, List.length_singleton, Nat.add_sub_cancel,
        List.take_left' rfl]
      by_cases h2 : k < ys.length
      · left
        rw [ite_eq_left h2, List.drop_append_of_le_length (by omega), List.foldl_append, List.foldl_cons,
          List.foldl_nil, Nat.mul_comm]
      · right
        have e : k = ys.length := by omega
        subst e
        refine ⟨?_, ite_eq_right h2, List.getElem_concat_length rfl _⟩
        rw [List.drop_of_length_le (Nat.le_of_eq (by rw [List.length_append, List.length_singleton]))]
        rfl
  · left
    have e : c = 1 := by omega
    subst e
    rw [ite_eq_right h1, Nat.sub_zero, ite_eq_left hk, List.take_length, Nat.one_mul]

private theorem reader_step (c M p r d x : ℕ) (hc : 1 ≤ c) (hd : d < r)
    (h : M = c * p ∨ (M = 1 ∧ p = 0 ∧ r = c)) : (x % M + M * d) / c = d * p + x % M / c := by
  rcases h with h | ⟨h1, h2, h3⟩
  · rw [h, Nat.mul_assoc, Nat.add_mul_div_left _ _ (by omega : 0 < c), Nat.mul_comm p d]
    exact Nat.add_comm _ _
  · subst h3
    rw [h1, h2, Nat.mod_one, Nat.one_mul, Nat.zero_add, Nat.mul_zero, Nat.zero_div, Nat.div_eq_of_lt hd]

private theorem rows_from (w : Wire) (hlen : w.step.length = w.radix.length) (hrad : ∀ r ∈ w.radix, 1 ≤ r)
    (hcuts : 1 ≤ w.cuts) (hlast : w.cuts = 1 ∨ w.radix.getLast? = some w.cuts) (i : ℕ) :
    ∀ (k a : ℕ), a + k = w.radix.length → ∀ (start : ℕ) (base : ℤ) (u : ℕ) (y : ℤ),
      u = start + i % prodR ((w.radix.zip w.step).drop a).reverse / w.cuts →
      y = base + accum ((w.radix.zip w.step).drop a).reverse i →
      BoxRow start base ((List.range' a k).filterMap (digitAt w)) u y := by
  intro k
  induction k with
  | zero =>
    intro a ha start base u y hu hy
    have e : (w.radix.zip w.step).drop a = [] :=
      List.drop_of_length_le (by rw [List.length_zip]; omega)
    rw [e] at hu hy
    have e1 : prodR ([] : List (ℕ × ℤ)).reverse = 1 := rfl
    have e2 : accum ([] : List (ℕ × ℤ)).reverse i = 0 := rfl
    rw [e1, Nat.mod_one, Nat.zero_div] at hu
    rw [e2] at hy
    exact ⟨by omega, by omega⟩
  | succ k ih =>
    intro a ha start base u y hu hy
    have han : a < w.radix.length := by omega
    have has : a < w.step.length := by omega
    have haz : a < (w.radix.zip w.step).length := by rw [List.length_zip]; omega
    have e : ((w.radix.zip w.step).drop a).reverse
        = ((w.radix.zip w.step).drop (a + 1)).reverse ++ [(w.radix[a], w.step[a])] := by
      rw [List.drop_eq_getElem_cons haz, List.reverse_cons, List.getElem_zip]
    have hM := prodR_drop w.radix w.step hlen (a + 1)
    have hstride := stride_spec w.radix w.cuts hcuts hlast a han
    rw [← hM] at hstride
    rw [e, prodR_concat] at hu
    rw [e, accum_concat] at hy
    rw [List.range'_succ]
    by_cases hr : w.radix[a] > 1
    · have hf : digitAt w a
          = some { r := w.radix[a], p := strideAt w.radix w.cuts a, s := w.step[a] } := by
        unfold digitAt
        rw [getElem!_pos w.radix a han, getElem!_pos w.step a has, ite_eq_left hr]
      rw [List.filterMap_cons_some hf]
      have hv : i / prodR ((w.radix.zip w.step).drop (a + 1)).reverse % w.radix[a] < w.radix[a] :=
        Nat.mod_lt _ (by omega)
      have hy' : y = base
          + ((i / prodR ((w.radix.zip w.step).drop (a + 1)).reverse % w.radix[a] : ℕ) : ℤ) * w.step[a]
          + accum ((w.radix.zip w.step).drop (a + 1)).reverse i := by
        rw [hy]
        omega
      refine ⟨_, hv, ?_⟩
      refine ih (a + 1) (by omega) _ _ u y ?_ hy'
      rw [hu, Nat.mod_mul, reader_step w.cuts _ _ _ _ i hcuts hv hstride]
      exact (Nat.add_assoc _ _ _).symm
    · have h1 : w.radix[a] = 1 := by
        have := hrad _ (List.getElem_mem han)
        omega
      have hf : digitAt w a = none := by
        unfold digitAt
        rw [getElem!_pos w.radix a han, ite_eq_right hr]
      rw [List.filterMap_cons_none hf]
      refine ih (a + 1) (by omega) _ _ u y ?_ ?_
      · rw [hu, h1, Nat.mul_one]
      · rw [hy, h1, Nat.mod_one]
        omega

theorem row_digits (w : Wire) (hlen : w.step.length = w.radix.length) (hrad : ∀ r ∈ w.radix, 1 ≤ r)
    (hcuts : 1 ≤ w.cuts) (hlast : w.cuts = 1 ∨ w.radix.getLast? = some w.cuts) (i : ℕ) (hi : i < w.units) :
    BoxRow w.start w.base (digitsOf w) (w.readerAt i).1 (w.writerAt i) := by
  have hi' : i < prodR (w.radix.zip w.step).reverse := hi
  rw [digitsOf_eq, List.range_eq_range']
  refine rows_from w hlen hrad hcuts hlast i w.radix.length 0 (by omega) _ _ _ _ ?_ ?_
  · show w.start + i / w.cuts = w.start + i % prodR ((w.radix.zip w.step).drop 0).reverse / w.cuts
    rw [List.drop_zero, Nat.mod_eq_of_lt hi']
  · show w.base + accum (w.radix.zip w.step).reverse i = w.base + accum ((w.radix.zip w.step).drop 0).reverse i
    rw [List.drop_zero]

theorem digitsOf_radix (w : Wire) : ∀ d ∈ digitsOf w, 1 ≤ d.r := by
  intro d hd
  rw [digitsOf_eq] at hd
  obtain ⟨k, _, hk⟩ := List.mem_filterMap.1 hd
  unfold digitAt at hk
  split at hk
  · rename_i h
    rw [← Option.some.inj hk]
    exact Nat.le_of_lt h
  · cases hk

theorem sameSlot_sound (S : ℕ) (gr gw : List Grid) (w : Spot.Wires.Wire) (hr : OneGrid gr) (hw : OneGrid gw)
    (hlen : w.step.length = w.radix.length) (hrad : ∀ r ∈ w.radix, 1 ≤ r) (hcuts : 1 ≤ w.cuts)
    (hlast : w.cuts = 1 ∨ w.radix.getLast? = some w.cuts)
    (h : sameSlot S gr gw w = true) (i : ℕ) (hi : i < w.units) (s s' : ℕ)
    (hs : slotOf S gr (w.readerAt i).1 = some s)
    (hs' : slotOf S gw (w.writerAt i).toNat = some s') : s = s' := by
  unfold sameSlot at h
  simp only [Bool.or_eq_true] at h
  rcases h with h1 | h2
  · have hS : S = 1 := eq_of_beq h1
    subst hS
    unfold slotOf at hs hs'
    obtain ⟨_, _, e1⟩ := Option.map_eq_some_iff.1 hs
    obtain ⟨_, _, e2⟩ := Option.map_eq_some_iff.1 hs'
    rw [← e1, ← e2, Nat.mod_one, Nat.mod_one]
  · exact search_sound S gr gw hr hw FUEL _ h2
      (fun b hb => by rw [List.mem_singleton.1 hb]; exact digitsOf_radix w) _ (List.mem_singleton.2 rfl)
      _ _ (row_digits w hlen hrad hcuts hlast i hi) s s' hs hs'

end InferenceOnlyCheck.Record.SecurityProofs
