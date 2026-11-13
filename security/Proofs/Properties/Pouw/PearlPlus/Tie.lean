module

public import Properties.Pouw.PearlPlusTie
public import Proofs.Pouw.PearlPlus.Draw
public import Proofs.Properties.Pouw.PearlPlus.Draw

/-!
# The wrong share counts the untied cells (`Properties.Pouw.PearlPlus.UntiedCellsAtMost`)

A run whose wrong share is at most ε has at most n(ε) (`cellsAt`) untied credited cells. No premise: the statement is
about the run's passes and table alone.

* **Every untied credited cell is in an untied tile.** A credited cell is a credited cell of tile `x` of its record,
  whose id is the record's offset plus `x` (`tile?_offsets`). Were that tile tied (`TileTiedAtW`), the table would tie
  every credited cell of it, the cell included. So the untied credited cells lie in the untied tiles' credited cells,
  and number at most their sum (`card_untiedCredited_le`).
* **A tile's integer ticket is its cells at its record's cell ticket, scaled by the lcm.** `tickets` scales each tile's
  weight `wref(1, |K|, n) |cr| |cc| / n` by the lcm `L` of the denominators, so its integer ticket is
  `L |cr| |cc| cellTicket r` (`tileTickets_eq`). Every tile has a width `n > 0` (`Scheme.tiles` cuts no block of a width
  of 0, `n_pos_of_tile`), so `L > 0`, and the wrong share is the untied tiles' cells weighed by their cell tickets over
  all tiles' (`wrong_eq`, `total_eq`).
* **The fill is the most cells such a share holds.** Grouped by record, a unit of ticket weight of record `r` holds
  1 / `cellTicket r` cells (`cellUnits`, whose pairs are `unitsBy`'s). For any `l ≥ 0`,
  Σ ρ a ≤ l Σ a + Σ w (ρ − l)⁺ over units of weight `w`, count `ρ` a unit of weight and `a ≤ w` of it taken. Filling
  the units in `unitsBy`'s order, most cells a unit of weight first, reaches that bound at the last unit it fills
  (`fill_dual`), so the cells of any choice of weight at most `b` are at most `fill` at `b` (`sum_le_fill`).
-/

@[expose] public section

open Pouw.PearlPlus.Draw (cells tickets_eq mem_cells dvd_lcmOf lcmOf tile?_offsets grid tiles_getElem? nblocks
  ticketWeight_eq tileWeight wref_one_cast length_flatMap_const)

namespace Properties.Pouw.PearlPlus

open PouwCheck.PearlPlus

open Classical

/-! ## Sums over lists -/

theorem sum_map_le' {α : Type} (f g : α → ℝ) : ∀ L : List α, (∀ x ∈ L, f x ≤ g x) → (L.map f).sum ≤ (L.map g).sum
  | [], _ => by simp
  | x :: L, h => by
    simp only [List.map_cons, List.sum_cons]
    exact add_le_add (h x List.mem_cons_self) (sum_map_le' f g L fun y hy => h y (List.mem_cons_of_mem _ hy))

theorem sum_map_nonneg' {α : Type} (f : α → ℝ) : ∀ L : List α, (∀ x ∈ L, 0 ≤ f x) → 0 ≤ (L.map f).sum
  | [], _ => by simp
  | x :: L, h => by
    simp only [List.map_cons, List.sum_cons]
    exact add_nonneg (h x List.mem_cons_self) (sum_map_nonneg' f L fun y hy => h y (List.mem_cons_of_mem _ hy))

theorem sum_map_eq_zero' {α : Type} (f : α → ℝ) : ∀ L : List α, (∀ x ∈ L, f x = 0) → (L.map f).sum = 0
  | [], _ => by simp
  | x :: L, h => by
    simp only [List.map_cons, List.sum_cons]
    rw [h x List.mem_cons_self, sum_map_eq_zero' f L fun y hy => h y (List.mem_cons_of_mem _ hy), add_zero]

theorem sum_map_mul_left' {α : Type} (c : ℝ) (f : α → ℝ) : ∀ L : List α,
    (L.map fun a => c * f a).sum = c * (L.map f).sum
  | [] => by simp
  | a :: L => by
    simp only [List.map_cons, List.sum_cons, sum_map_mul_left' c f L]
    ring

theorem sum_map_lin {α : Type} (c : ℝ) (g h : α → ℝ) : ∀ L : List α,
    (L.map fun t => c * g t + h t).sum = c * (L.map g).sum + (L.map h).sum
  | [] => by simp
  | t :: L => by
    simp only [List.map_cons, List.sum_cons, sum_map_lin c g h L]
    ring

theorem sum_map_flatMap' {α β : Type} (f : α → List β) (F : β → ℝ) : ∀ l : List α,
    ((l.flatMap f).map F).sum = (l.map fun a => ((f a).map F).sum).sum
  | [] => by simp
  | a :: l => by
    rw [List.flatMap_cons, List.map_append, List.sum_append, sum_map_flatMap' f F l, List.map_cons, List.sum_cons]

theorem sum_map_filterMap' {α β : Type} (g : α → Option β) (G : β → ℝ) : ∀ l : List α,
    ((l.filterMap g).map G).sum = (l.map fun a => ((g a).map G).getD 0).sum
  | [] => by simp
  | a :: l => by
    have ih := sum_map_filterMap' g G l
    rw [List.filterMap_cons, List.map_cons, List.sum_cons, ← ih]
    rcases g a with _ | b <;> simp

theorem cast_sum_nat {α : Type} (f : α → ℕ) : ∀ l : List α,
    (((l.map f).sum : ℕ) : ℝ) = (l.map fun a => (f a : ℝ)).sum
  | [] => by simp
  | a :: l => by simp only [List.map_cons, List.sum_cons, Nat.cast_add, cast_sum_nat f l]

theorem cast_sum_rat {α : Type} (f : α → ℚ) : ∀ l : List α,
    (((l.map f).sum : ℚ) : ℝ) = (l.map fun a => (f a : ℝ)).sum
  | [] => by simp
  | a :: l => by simp only [List.map_cons, List.sum_cons, Rat.cast_add, cast_sum_rat f l]

theorem flatMap_zipIdx_fst {α β : Type} (f : α → List β) : ∀ (l : List α) (k : ℕ),
    (l.zipIdx k).flatMap (fun q => f q.1) = l.flatMap f
  | [], _ => rfl
  | a :: l, k => by
    have ih := flatMap_zipIdx_fst f l (k + 1)
    simp only [List.zipIdx_cons, List.flatMap_cons, ih]

theorem filterMap_zipIdx_fst {α β : Type} (f : α → Option β) : ∀ (l : List α) (k : ℕ),
    (l.zipIdx k).filterMap (fun q => f q.1) = l.filterMap f
  | [], _ => rfl
  | a :: l, k => by
    have ih := filterMap_zipIdx_fst f l (k + 1)
    rw [List.zipIdx_cons, List.filterMap_cons, List.filterMap_cons]
    rcases f a with _ | b <;> simp [ih]

theorem map_zipIdx_fst {α β : Type} (f : α → β) : ∀ (l : List α) (k : ℕ),
    (l.zipIdx k).map (fun q => f q.1) = l.map f
  | [], _ => rfl
  | a :: l, k => by
    have ih := map_zipIdx_fst f l (k + 1)
    simp only [List.zipIdx_cons, List.map_cons, ih]

/-! ## The fill -/

theorem fill_nonneg : ∀ (S : List (ℚ × ℚ)) (b : ℝ), (∀ x ∈ S, 0 ≤ x.1 ∧ 0 ≤ x.2) → 0 ≤ b → 0 ≤ fill S b
  | [], _, _, _ => by simp [fill]
  | (ρ, w) :: S, b, hnn, hb => by
    have h0 : 0 ≤ ρ ∧ 0 ≤ w := hnn (ρ, w) List.mem_cons_self
    have hρ : (0 : ℝ) ≤ ρ := by exact_mod_cast h0.1
    have hw : (0 : ℝ) ≤ w := by exact_mod_cast h0.2
    have h1 : 0 ≤ min b (w : ℝ) := le_min hb hw
    have h2 := fill_nonneg S (b - min b (w : ℝ)) (fun x hx => hnn x (List.mem_cons_of_mem _ hx))
      (sub_nonneg.2 (min_le_left _ _))
    simp only [fill]
    exact add_nonneg (mul_nonneg h1 hρ) h2

/-- **The fill reaches the dual bound**: on units in `unitsBy`'s order, some `l ≥ 0`, 0 or one of the units' counts a
unit of weight, has `l b + Σ w (ρ − l)⁺` at most the fill at `b`. -/
theorem fill_dual : ∀ (S : List (ℚ × ℚ)) (b : ℝ), S.Pairwise (fun a c => decide (c.1 ≤ a.1) = true) →
    (∀ x ∈ S, 0 ≤ x.1 ∧ 0 ≤ x.2) → 0 ≤ b →
    ∃ l : ℝ, 0 ≤ l ∧ (l = 0 ∨ ∃ x ∈ S, l = (x.1 : ℝ)) ∧
      l * b + (S.map fun x => (x.2 : ℝ) * max ((x.1 : ℝ) - l) 0).sum ≤ fill S b
  | [], b, _, _, _ => ⟨0, le_refl 0, Or.inl rfl, by simp [fill]⟩
  | (ρ, w) :: S, b, hs, hnn, hb => by
    rw [List.pairwise_cons] at hs
    obtain ⟨hhead, hs⟩ := hs
    have h0 : 0 ≤ ρ ∧ 0 ≤ w := hnn (ρ, w) List.mem_cons_self
    have hρ : (0 : ℝ) ≤ ρ := by exact_mod_cast h0.1
    have hw : (0 : ℝ) ≤ w := by exact_mod_cast h0.2
    have hnn' : ∀ x ∈ S, 0 ≤ x.1 ∧ 0 ≤ x.2 := fun x hx => hnn x (List.mem_cons_of_mem _ hx)
    have hle : ∀ x ∈ S, (x.1 : ℝ) ≤ ρ := fun x hx => by
      have h := hhead x hx
      simp only [decide_eq_true_eq] at h
      exact_mod_cast h
    simp only [fill, List.map_cons, List.sum_cons]
    by_cases hbw : b ≤ (w : ℝ)
    · refine ⟨(ρ : ℝ), hρ, Or.inr ⟨(ρ, w), List.mem_cons_self, rfl⟩, ?_⟩
      have hz : (S.map fun x => (x.2 : ℝ) * max ((x.1 : ℝ) - ρ) 0).sum = 0 :=
        sum_map_eq_zero' _ S fun x hx => by
          rw [max_eq_right (by linarith [hle x hx]), mul_zero]
      have hf := fill_nonneg S 0 hnn' le_rfl
      simp only [min_eq_left hbw, sub_self, max_self, mul_zero, zero_add]
      linarith
    · have hbw : (w : ℝ) < b := not_le.1 hbw
      obtain ⟨l, hl0, hlS, hl⟩ := fill_dual S (b - w) hs hnn' (by linarith)
      have hlρ : l ≤ ρ := by
        rcases hlS with h | ⟨x, hx, h⟩
        · rw [h]; exact hρ
        · rw [h]; exact hle x hx
      refine ⟨l, hl0, ?_, ?_⟩
      · rcases hlS with h | ⟨x, hx, h⟩
        · exact Or.inl h
        · exact Or.inr ⟨x, List.mem_cons_of_mem _ hx, h⟩
      · rw [min_eq_right hbw.le, max_eq_left (by linarith : (0 : ℝ) ≤ ρ - l)]
        linarith

/-- **The fill bounds every choice of weight**: units (count a unit of weight `ρ`, weight `w`) with `a ≤ w` of each
taken, `a` at most `b` in all, hold at most the fill at `b` of the units in `unitsBy`'s order. -/
theorem sum_le_fill (L : List (ℚ × ℚ × ℝ)) (b : ℝ) (hnn : ∀ t ∈ L, 0 ≤ t.1 ∧ 0 ≤ t.2.1)
    (ha : ∀ t ∈ L, 0 ≤ t.2.2 ∧ t.2.2 ≤ (t.2.1 : ℝ)) (hb : (L.map fun t => t.2.2).sum ≤ b) :
    (L.map fun t => (t.1 : ℝ) * t.2.2).sum ≤
      fill ((L.map fun t => (t.1, t.2.1)).mergeSort fun a c => decide (c.1 ≤ a.1)) b := by
  have hperm : ((L.map fun t => (t.1, t.2.1)).mergeSort fun a c => decide (c.1 ≤ a.1)).Perm
      (L.map fun t => (t.1, t.2.1)) := List.mergeSort_perm _ _
  have hsorted : ((L.map fun t => (t.1, t.2.1)).mergeSort fun a c => decide (c.1 ≤ a.1)).Pairwise
      (fun a c => decide (c.1 ≤ a.1) = true) :=
    List.pairwise_mergeSort (fun a b c h1 h2 => by simp only [decide_eq_true_eq] at *; exact le_trans h2 h1)
      (fun a b => by simpa only [Bool.or_eq_true, decide_eq_true_eq] using le_total b.1 a.1) _
  have hSnn : ∀ x ∈ ((L.map fun t => (t.1, t.2.1)).mergeSort fun a c => decide (c.1 ≤ a.1)),
      0 ≤ x.1 ∧ 0 ≤ x.2 := by
    intro x hx
    obtain ⟨t, ht, rfl⟩ := List.mem_map.1 (hperm.subset hx)
    exact hnn t ht
  have hb0 : 0 ≤ b := le_trans (sum_map_nonneg' _ L fun t ht => (ha t ht).1) hb
  obtain ⟨l, hl0, -, hl⟩ := fill_dual _ b hsorted hSnn hb0
  have hpos : (((L.map fun t => (t.1, t.2.1)).mergeSort fun a c => decide (c.1 ≤ a.1)).map
      fun x => (x.2 : ℝ) * max ((x.1 : ℝ) - l) 0).sum =
      (L.map fun t => (t.2.1 : ℝ) * max ((t.1 : ℝ) - l) 0).sum :=
    ((hperm.map fun x : ℚ × ℚ => (x.2 : ℝ) * max ((x.1 : ℝ) - l) 0).sum_eq).trans
      (congrArg List.sum List.map_map)
  calc (L.map fun t => (t.1 : ℝ) * t.2.2).sum
      ≤ (L.map fun t => l * t.2.2 + (t.2.1 : ℝ) * max ((t.1 : ℝ) - l) 0).sum :=
        sum_map_le' _ _ L fun t ht => by
          obtain ⟨ha0, haw⟩ := ha t ht
          have h1 : ((t.1 : ℝ) - l) * t.2.2 ≤ max ((t.1 : ℝ) - l) 0 * t.2.2 :=
            mul_le_mul_of_nonneg_right (le_max_left _ _) ha0
          have h2 : max ((t.1 : ℝ) - l) 0 * t.2.2 ≤ max ((t.1 : ℝ) - l) 0 * t.2.1 :=
            mul_le_mul_of_nonneg_left haw (le_max_right _ _)
          linarith
    _ = l * (L.map fun t => t.2.2).sum + (L.map fun t => (t.2.1 : ℝ) * max ((t.1 : ℝ) - l) 0).sum :=
        sum_map_lin l (fun t => t.2.2) (fun t => (t.2.1 : ℝ) * max ((t.1 : ℝ) - l) 0) L
    _ ≤ l * b + (L.map fun t => (t.2.1 : ℝ) * max ((t.1 : ℝ) - l) 0).sum := by
        have := mul_le_mul_of_nonneg_left hb hl0
        linarith
    _ ≤ _ := by rw [← hpos]; exact hl

/-! ## The run's tiles and records -/

/-- A tile's credited cells: |credited rows| · |credited columns|. -/
def tileCells (c : _root_.Pouw.PearlPlus.Draw.Cell) : ℕ :=
  (c.r.credited c.rows c.cols).1.length * (c.r.credited c.rows c.cols).2.length

/-- A record's credited cells, as `unitsBy` counts them. -/
def recordCells (p : Pass) (r : UnitRecord) : ℕ :=
  ((p.scheme.tiles r.matmul.m r.matmul.n).map fun (rows, cols) =>
    (r.credited rows cols).1.length * (r.credited rows cols).2.length).sum

/-- A record's credited cells in ℚ, as `unitsBy`'s `cells` sums them. -/
def recordCellsQ (p : Pass) (r : UnitRecord) : ℚ :=
  ((p.scheme.tiles r.matmul.m r.matmul.n).map fun (rows, cols) =>
    ((r.credited rows cols).1.length : ℚ) * (r.credited rows cols).2.length).sum

theorem recordCellsQ_eq (p : Pass) (r : UnitRecord) : recordCellsQ p r = (recordCells p r : ℚ) := by
  unfold recordCellsQ recordCells
  rw [Nat.cast_list_sum, List.map_map]
  congr 1
  apply List.map_congr_left
  intro ⟨rows, cols⟩ _
  simp

/-- The run's records, (pass, the pass, position, record), in `tickets`' order. -/
def runRecords (passes : List Pass) : List (ℕ × Pass × ℕ × UnitRecord) :=
  passes.zipIdx.flatMap fun q => q.1.records.zipIdx.map fun s => (q.2, q.1, s.2, s.1)

/-- A record's tiles, as `cells` lists them. -/
def recordTiles (q : ℕ × Pass × ℕ × UnitRecord) : List _root_.Pouw.PearlPlus.Draw.Cell :=
  (q.2.1.scheme.tiles q.2.2.2.matmul.m q.2.2.2.matmul.n).zipIdx.map fun e =>
    { i := q.1, p := q.2.1, pos := q.2.2.1, r := q.2.2.2, x := e.2, rows := e.1.1, cols := e.1.2 }

theorem cells_eq (passes : List Pass) : cells passes = (runRecords passes).flatMap recordTiles := by
  simp only [cells, runRecords, List.flatMap_assoc, List.flatMap_map]
  rfl

theorem sum_cells_eq (passes : List Pass) (F : _root_.Pouw.PearlPlus.Draw.Cell → ℝ) :
    ((cells passes).map F).sum = ((runRecords passes).map fun q => ((recordTiles q).map F).sum).sum := by
  rw [cells_eq, sum_map_flatMap']

theorem mem_recordTiles_r {q : ℕ × Pass × ℕ × UnitRecord} {c : _root_.Pouw.PearlPlus.Draw.Cell}
    (h : c ∈ recordTiles q) : c.r = q.2.2.2 := by
  unfold recordTiles at h
  obtain ⟨e, -, rfl⟩ := List.mem_map.1 h
  rfl

theorem sum_cnt_recordTiles (q : ℕ × Pass × ℕ × UnitRecord) :
    ((recordTiles q).map fun c => (tileCells c : ℝ)).sum = (recordCells q.2.1 q.2.2.2 : ℝ) := by
  unfold recordTiles recordCells
  rw [List.map_map, cast_sum_nat]
  exact congrArg List.sum (map_zipIdx_fst (fun rc : List ℕ × List ℕ =>
    (((q.2.2.2.credited rc.1 rc.2).1.length * (q.2.2.2.credited rc.1 rc.2).2.length : ℕ) : ℝ)) _ 0)

theorem sum_ticket_left (q : ℕ × Pass × ℕ × UnitRecord) (f : _root_.Pouw.PearlPlus.Draw.Cell → ℝ) :
    ((recordTiles q).map fun c => (cellTicket c.r : ℝ) * f c).sum =
      (cellTicket q.2.2.2 : ℝ) * ((recordTiles q).map f).sum := by
  rw [← sum_map_mul_left']
  congr 1
  apply List.map_congr_left
  intro c hc
  rw [mem_recordTiles_r hc]

/-- A tile cuts a width `n > 0`: `Scheme.tiles` has no block of a width of 0. -/
theorem n_pos_of_tile {s : Scheme} {m n x : ℕ} {rc : List ℕ × List ℕ} (h : (s.tiles m n)[x]? = some rc) : 0 < n := by
  have hx : x < nblocks m s.tileM * nblocks n s.tileN := by
    rw [tiles_getElem?] at h
    split_ifs at h with hx
    exact hx
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · exfalso
    have h0 : nblocks 0 s.tileN = 0 := by
      unfold nblocks
      rcases Nat.eq_zero_or_pos s.tileN with h0 | h0
      · simp [h0]
      · exact Nat.div_eq_of_lt (by omega)
    rw [h0, Nat.mul_zero] at hx
    exact Nat.not_lt_zero _ hx
  · exact hn

theorem cellTicket_nonneg (r : UnitRecord) : 0 ≤ cellTicket r := by
  unfold cellTicket
  exact div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)

theorem cellTicket_pos {r : UnitRecord} (hn : 0 < r.matmul.n) : 0 < cellTicket r := by
  unfold cellTicket
  refine div_pos ?_ (by exact_mod_cast hn)
  rw [wref_one_cast]
  exact add_pos_of_pos_of_nonneg (add_pos_of_nonneg_of_pos (add_nonneg (le_max_right _ _) (by positivity))
    (mul_pos (by norm_num) (by exact_mod_cast hn))) (by positivity)

/-! ## The integer tickets -/

/-- The lcm of the tiles' denominators, by which `tickets` scales every tile's weight. -/
def ticketDen (passes : List Pass) : ℕ := lcmOf ((cells passes).map _root_.Pouw.PearlPlus.Draw.Cell.weight)

/-- A tile's integer ticket, as `tickets` scales it. -/
def tileTickets (passes : List Pass) (c : _root_.Pouw.PearlPlus.Draw.Cell) : ℕ :=
  (ticketWeight c.r c.rows c.cols).1 * (ticketDen passes / (ticketWeight c.r c.rows c.cols).2)

theorem tickets_eq' (passes : List Pass) :
    tickets passes = (cells passes).map fun c => ((c.i, c.p.offsets[c.pos]! + c.x), tileTickets passes c) := by
  rw [tickets_eq, List.map_map]
  rfl

theorem foldl_lcm_pos : ∀ (ws : List ((ℕ × ℕ) × ℕ × ℕ)) (a : ℕ), 0 < a → (∀ e ∈ ws, 0 < e.2.2) →
    0 < ws.foldl (fun l (_, (_, d)) => Nat.lcm l d) a
  | [], _, ha, _ => ha
  | e :: ws, a, ha, h => by
    rw [List.foldl_cons]
    exact foldl_lcm_pos ws _ (Nat.lcm_pos ha (h e List.mem_cons_self)) fun e' he' => h e' (List.mem_cons_of_mem _ he')

theorem ticketDen_pos (passes : List Pass) : 0 < ticketDen passes := by
  unfold ticketDen lcmOf
  refine foldl_lcm_pos _ 1 Nat.one_pos fun e he => ?_
  obtain ⟨c, hc, rfl⟩ := List.mem_map.1 he
  have hn := n_pos_of_tile (mem_cells.1 hc).2.2
  show 0 < (ticketWeight c.r c.rows c.cols).2
  rw [ticketWeight_eq _ _ _ hn]
  exact Nat.pos_of_ne_zero (tileWeight c.r c.rows c.cols).den_nz

/-- **A tile's integer ticket** is its credited cells at its record's cell ticket, scaled by `ticketDen`. -/
theorem tileTickets_eq {passes : List Pass} {c : _root_.Pouw.PearlPlus.Draw.Cell} (hc : c ∈ cells passes) :
    (tileTickets passes c : ℚ) = (ticketDen passes : ℚ) * tileCells c * cellTicket c.r := by
  have hn := n_pos_of_tile (mem_cells.1 hc).2.2
  have hd := dvd_lcmOf (List.mem_map_of_mem (f := _root_.Pouw.PearlPlus.Draw.Cell.weight) hc)
  change (ticketWeight c.r c.rows c.cols).2 ∣ ticketDen passes at hd
  unfold tileTickets
  rw [ticketWeight_eq _ _ _ hn] at hd ⊢
  dsimp only at hd ⊢
  have hq0 : 0 ≤ tileWeight c.r c.rows c.cols := div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
  have hden : ((tileWeight c.r c.rows c.cols).den : ℚ) ≠ 0 := by
    exact_mod_cast (tileWeight c.r c.rows c.cols).den_nz
  rw [Nat.cast_mul, Nat.cast_div hd hden]
  have hnum : (((tileWeight c.r c.rows c.cols).num.toNat : ℕ) : ℚ) = ((tileWeight c.r c.rows c.cols).num : ℚ) := by
    rw [← Int.cast_natCast, Int.toNat_of_nonneg (Rat.num_nonneg.2 hq0)]
  rw [hnum]
  have hqq : tileWeight c.r c.rows c.cols = (tileCells c : ℚ) * cellTicket c.r := by
    simp only [tileWeight, cellTicket, tileCells]
    push_cast
    ring
  calc ((tileWeight c.r c.rows c.cols).num : ℚ) * ((ticketDen passes : ℚ) / (tileWeight c.r c.rows c.cols).den)
      = (ticketDen passes : ℚ) * (((tileWeight c.r c.rows c.cols).num : ℚ) / (tileWeight c.r c.rows c.cols).den) := by
        ring
    _ = (ticketDen passes : ℚ) * tileWeight c.r c.rows c.cols := by rw [Rat.num_div_den]
    _ = _ := by rw [hqq]; ring

theorem tileTickets_real {passes : List Pass} {c : _root_.Pouw.PearlPlus.Draw.Cell} (hc : c ∈ cells passes) :
    (tileTickets passes c : ℝ) = (ticketDen passes : ℝ) * ((cellTicket c.r : ℝ) * tileCells c) := by
  have h := congrArg (fun x : ℚ => (x : ℝ)) (tileTickets_eq hc)
  simp only [Rat.cast_mul, Rat.cast_natCast] at h
  rw [h]
  ring

/-! ## The wrong share as cells -/

/-- A tile's credited cells where its table does not tie it, else 0. -/
noncomputable def untiedTileCells (R : Run) (c : _root_.Pouw.PearlPlus.Draw.Cell) : ℕ :=
  if TileTiedAtW R.w R.rounds R.passes R.table (c.i, c.p.offsets[c.pos]! + c.x) then 0 else tileCells c

/-- The run's tiles' credited cells, each at its record's cell ticket. -/
noncomputable def ticketSum (R : Run) : ℝ :=
  ((cells R.passes).map fun c => (cellTicket c.r : ℝ) * tileCells c).sum

/-- The run's untied tiles' credited cells, each at its record's cell ticket. -/
noncomputable def untiedSum (R : Run) : ℝ :=
  ((cells R.passes).map fun c => (cellTicket c.r : ℝ) * untiedTileCells R c).sum

theorem untiedTileCells_le (R : Run) (c : _root_.Pouw.PearlPlus.Draw.Cell) : untiedTileCells R c ≤ tileCells c := by
  unfold untiedTileCells
  split_ifs <;> simp

theorem sum_ite_fin (tk : List ((ℕ × ℕ) × ℕ)) (Q : ℕ × ℕ → Prop) [DecidablePred Q] :
    (∑ u : Fin tk.length, if Q (tk.get u).1 then 0 else weightsOf tk u) =
      (tk.map fun e => if Q e.1 then 0 else e.2).sum := by
  have h : ∀ u : Fin tk.length, (if Q (tk.get u).1 then 0 else weightsOf tk u) =
      (tk.map fun e => if Q e.1 then 0 else e.2).getD u 0 := by
    intro u
    have h1 : weightsOf tk u = (tk.get u).2 := by
      unfold weightsOf
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem u.isLt]
      rfl
    have h2 : (tk.map fun e => if Q e.1 then 0 else e.2).getD u 0 =
        if Q (tk.get u).1 then 0 else (tk.get u).2 := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_getElem u.isLt]
      rfl
    rw [h1, h2]
  rw [Finset.sum_congr rfl fun u _ => h u,
    Fin.sum_univ_eq_sum_range (fun i => (tk.map fun e => if Q e.1 then 0 else e.2).getD i 0)]
  have hl : tk.length = (tk.map fun e => if Q e.1 then 0 else e.2).length := (List.length_map _).symm
  rw [hl, sum_range_getD]

theorem total_eq (R : Run) : (ticketTotal (tickets R.passes) : ℝ) = (ticketDen R.passes : ℝ) * ticketSum R := by
  unfold ticketTotal ticketSum
  rw [tickets_eq', List.map_map, cast_sum_nat, ← sum_map_mul_left']
  congr 1
  apply List.map_congr_left
  intro c hc
  show (tileTickets R.passes c : ℝ) = _
  rw [tileTickets_real hc]

theorem wrong_eq (R : Run) : (R.wrongWeight : ℝ) = (ticketDen R.passes : ℝ) * untiedSum R := by
  unfold Run.wrongWeight untiedSum
  rw [sum_ite_fin (tickets R.passes) (TileTiedAtW R.w R.rounds R.passes R.table), tickets_eq', List.map_map,
    cast_sum_nat, ← sum_map_mul_left']
  congr 1
  apply List.map_congr_left
  intro c hc
  show ((if TileTiedAtW R.w R.rounds R.passes R.table (c.i, c.p.offsets[c.pos]! + c.x) then 0
    else tileTickets R.passes c : ℕ) : ℝ) = _
  unfold untiedTileCells
  split_ifs
  · simp
  · rw [tileTickets_real hc]

/-- **The untied credited cells are at most an untied share's**: at most ε of the ticket weight is untied, so the untied
tiles' cells at their cell tickets are at most ε of all tiles'. -/
theorem untiedSum_le (R : Run) (ε : ℝ) (hε : R.wrongShare ≤ ε) : untiedSum R ≤ ε * ticketSum R := by
  have hAW : untiedSum R ≤ ticketSum R := by
    unfold untiedSum ticketSum
    exact sum_map_le' _ _ _ fun c _ => mul_le_mul_of_nonneg_left (Nat.cast_le.2 (untiedTileCells_le R c))
      (Rat.cast_nonneg.2 (cellTicket_nonneg c.r))
  have hW0 : 0 ≤ ticketSum R := by
    unfold ticketSum
    exact sum_map_nonneg' _ _ fun c _ => mul_nonneg (Rat.cast_nonneg.2 (cellTicket_nonneg c.r)) (Nat.cast_nonneg _)
  have hD : (0 : ℝ) < ticketDen R.passes := by exact_mod_cast ticketDen_pos R.passes
  have hs : R.wrongShare = ((ticketDen R.passes : ℝ) * untiedSum R) / ((ticketDen R.passes : ℝ) * ticketSum R) := by
    unfold Run.wrongShare
    rw [wrong_eq, total_eq]
  rcases hW0.lt_or_eq with hWp | hWz
  · rw [hs, div_le_iff₀ (mul_pos hD hWp)] at hε
    have h : (ticketDen R.passes : ℝ) * untiedSum R ≤ (ticketDen R.passes : ℝ) * (ε * ticketSum R) := by
      calc (ticketDen R.passes : ℝ) * untiedSum R ≤ ε * ((ticketDen R.passes : ℝ) * ticketSum R) := hε
        _ = (ticketDen R.passes : ℝ) * (ε * ticketSum R) := by ring
    exact le_of_mul_le_mul_left h hD
  · rw [← hWz] at hAW ⊢
    simpa using hAW

/-! ## The untied credited cells lie in untied tiles -/

/-- A tile's credited cells, as cells (pass, record, row, column). -/
def boxOf (c : _root_.Pouw.PearlPlus.Draw.Cell) : List Cell :=
  (c.r.credited c.rows c.cols).1.flatMap fun i => (c.r.credited c.rows c.cols).2.map fun j => (c.i, c.pos, i, j)

theorem boxOf_length (c : _root_.Pouw.PearlPlus.Draw.Cell) : (boxOf c).length = tileCells c := by
  unfold boxOf tileCells
  rw [length_flatMap_const (fun i => (c.r.credited c.rows c.cols).2.map fun j => (c.i, c.pos, i, j))
    (c.r.credited c.rows c.cols).2.length (fun _ => by simp)]

/-- The untied tiles' credited cells. -/
noncomputable def untiedBoxes (R : Run) : List Cell :=
  (cells R.passes).flatMap fun c =>
    if TileTiedAtW R.w R.rounds R.passes R.table (c.i, c.p.offsets[c.pos]! + c.x) then [] else boxOf c

theorem untiedCredited_sub (R : Run) : R.untiedCredited ⊆ (untiedBoxes R).toFinset := by
  intro z hz
  rw [Run.untiedCredited, givenUpCells, Finset.mem_sdiff, Run.untiedCells, Finset.mem_filter] at hz
  obtain ⟨⟨hzr, hzc⟩, hzg⟩ := hz
  have hcred : z ∈ creditedCells R.passes := by
    by_contra h
    exact hzg (Finset.mem_sdiff.2 ⟨hzr, h⟩)
  rw [creditedCells, List.mem_toFinset] at hcred
  obtain ⟨⟨p, pi⟩, hp, hz1⟩ := List.mem_flatMap.1 hcred
  obtain ⟨⟨r, pos⟩, hr, hz2⟩ := List.mem_flatMap.1 hz1
  obtain ⟨⟨rows, cols⟩, hrc, hz3⟩ := List.mem_flatMap.1 hz2
  obtain ⟨i, hi, hz4⟩ := List.mem_flatMap.1 hz3
  obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hz4
  simp only [List.mem_zipIdx_iff_getElem?] at hp hr
  have hi' : i ∈ (r.credited rows cols).1 := hi
  have hj' : j ∈ (r.credited rows cols).2 := hj
  simp only [hp] at hzc
  obtain ⟨x, hx⟩ : ∃ x, (p.scheme.tiles r.matmul.m r.matmul.n)[x]? = some (rows, cols) :=
    List.mem_iff_getElem?.1 hrc
  obtain ⟨hpl, hrr⟩ := List.getElem?_eq_some_iff.1 hr
  have hr! : p.records[pos]! = r := by rw [getElem!_pos p.records pos hpl, hrr]
  have hnt : ¬ TileTiedAtW R.w R.rounds R.passes R.table (pi, p.offsets[pos]! + x) := by
    rintro ⟨p', pos', rows', cols', hp', ht', hall⟩
    simp only [hp, Option.some.injEq] at hp'
    subst hp'
    have hxl : x < (grid p.scheme p.records[pos]).length := by
      rw [hrr]
      exact (List.getElem?_eq_some_iff.1 hx).1
    have hto := tile?_offsets p hpl hxl
    rw [hrr] at hto
    have hg : (grid p.scheme r)[x]! = (rows, cols) := by
      unfold grid
      rw [List.getElem!_eq_getElem?_getD, hx, Option.getD_some]
    rw [hg] at hto
    have ht2 : p.tile? (p.offsets[pos]! + x) = some (pos', rows', cols') := ht'
    rw [hto] at ht2
    simp only [Option.some.injEq, Prod.mk.injEq] at ht2
    obtain ⟨rfl, rfl, rfl⟩ := ht2
    rw [hr!] at hall
    exact hzc (by rw [hr!]; exact hall i hi' j hj')
  rw [List.mem_toFinset]
  unfold untiedBoxes
  refine List.mem_flatMap.2 ⟨⟨pi, p, pos, r, x, rows, cols⟩, mem_cells.2 ⟨hp, hr, hx⟩, ?_⟩
  show (pi, pos, i, j) ∈ (if TileTiedAtW R.w R.rounds R.passes R.table (pi, p.offsets[pos]! + x) then []
    else boxOf ⟨pi, p, pos, r, x, rows, cols⟩)
  simp only [hnt, ↓reduceIte]
  unfold boxOf
  exact List.mem_flatMap.2 ⟨i, hi', List.mem_map.2 ⟨j, hj', rfl⟩⟩

theorem card_untiedCredited_le (R : Run) :
    R.untiedCredited.card ≤ ((cells R.passes).map (untiedTileCells R)).sum := by
  calc R.untiedCredited.card ≤ (untiedBoxes R).toFinset.card := Finset.card_le_card (untiedCredited_sub R)
    _ ≤ (untiedBoxes R).length := List.toFinset_card_le _
    _ = ((cells R.passes).map (untiedTileCells R)).sum := by
      unfold untiedBoxes
      rw [List.length_flatMap]
      congr 1
      apply List.map_congr_left
      intro c _
      unfold untiedTileCells
      split_ifs
      · rfl
      · exact boxOf_length c

/-! ## The run's units -/

/-- A record's untied tiles' credited cells, at its cell ticket. -/
noncomputable def untiedWeight (R : Run) (q : ℕ × Pass × ℕ × UnitRecord) : ℝ :=
  (cellTicket q.2.2.2 : ℝ) * ((recordTiles q).map fun c => (untiedTileCells R c : ℝ)).sum

/-- The run's units (`unitsBy` at one a cell), each with its untied tiles' weight. -/
noncomputable def cellUnits (R : Run) : List (ℚ × ℚ × ℝ) :=
  (runRecords R.passes).filterMap fun q =>
    if recordCells q.2.1 q.2.2.2 = 0 then none
    else some (1 / cellTicket q.2.2.2, (recordCells q.2.1 q.2.2.2 : ℚ) * cellTicket q.2.2.2, untiedWeight R q)

theorem cellUnits_map (R : Run) : (cellUnits R).map (fun t => (t.1, t.2.1)) =
    R.passes.flatMap fun p => p.records.filterMap fun r =>
      if recordCells p r = 0 then none else some (1 / cellTicket r, (recordCells p r : ℚ) * cellTicket r) := by
  unfold cellUnits runRecords
  rw [List.map_filterMap, List.filterMap_flatMap]
  refine Eq.trans ?_ (flatMap_zipIdx_fst (fun p : Pass => p.records.filterMap fun r =>
    if recordCells p r = 0 then none else some (1 / cellTicket r, (recordCells p r : ℚ) * cellTicket r)) R.passes 0)
  congr 1
  funext q
  rw [List.filterMap_map]
  refine Eq.trans ?_ (filterMap_zipIdx_fst (fun r : UnitRecord =>
    if recordCells q.1 r = 0 then none else some (1 / cellTicket r, (recordCells q.1 r : ℚ) * cellTicket r))
    q.1.records 0)
  congr 1
  funext s
  simp only [Function.comp_apply]
  split_ifs <;> rfl

theorem unitsBy_one (R : Run) : unitsBy (fun _ => 1) R.passes =
    ((cellUnits R).map fun t => (t.1, t.2.1)).mergeSort fun a b => decide (b.1 ≤ a.1) := by
  rw [cellUnits_map]
  unfold unitsBy
  congr 1
  refine congrArg (List.flatMap · R.passes) (funext fun p => ?_)
  refine congrArg (List.filterMap · p.records) (funext fun r => ?_)
  show (if recordCellsQ p r = 0 then none else some ((1 : ℚ) / cellTicket r, recordCellsQ p r * cellTicket r)) = _
  rw [recordCellsQ_eq]
  by_cases h : recordCells p r = 0 <;> simp [h]

theorem sum_units (R : Run) (G : ℚ × ℚ × ℝ → ℝ) :
    ((cellUnits R).map G).sum = ((runRecords R.passes).map fun q => if recordCells q.2.1 q.2.2.2 = 0 then 0 else
      G (1 / cellTicket q.2.2.2, (recordCells q.2.1 q.2.2.2 : ℚ) * cellTicket q.2.2.2, untiedWeight R q)).sum := by
  unfold cellUnits
  rw [sum_map_filterMap']
  congr 1
  apply List.map_congr_left
  intro q _
  split_ifs <;> rfl

theorem sum_ut_le_cells (R : Run) (q : ℕ × Pass × ℕ × UnitRecord) :
    ((recordTiles q).map fun c => (untiedTileCells R c : ℝ)).sum ≤ (recordCells q.2.1 q.2.2.2 : ℝ) := by
  rw [← sum_cnt_recordTiles q]
  exact sum_map_le' _ _ _ fun c _ => by exact_mod_cast untiedTileCells_le R c

theorem sum_ut_eq (R : Run) (q : ℕ × Pass × ℕ × UnitRecord) :
    ((recordTiles q).map fun c => (untiedTileCells R c : ℝ)).sum = if recordCells q.2.1 q.2.2.2 = 0 then 0 else
      ((1 / cellTicket q.2.2.2 : ℚ) : ℝ) * untiedWeight R q := by
  split_ifs with h0
  · apply le_antisymm
    · have h := sum_ut_le_cells R q
      rw [h0, Nat.cast_zero] at h
      exact h
    · exact sum_map_nonneg' _ _ fun c _ => Nat.cast_nonneg _
  · have hne : q.2.1.scheme.tiles q.2.2.2.matmul.m q.2.2.2.matmul.n ≠ [] := by
      intro h
      apply h0
      simp [recordCells, h]
    obtain ⟨rc, hrc⟩ : ∃ rc, (q.2.1.scheme.tiles q.2.2.2.matmul.m q.2.2.2.matmul.n)[0]? = some rc := by
      cases h : q.2.1.scheme.tiles q.2.2.2.matmul.m q.2.2.2.matmul.n with
      | nil => exact absurd h hne
      | cons a l => exact ⟨a, by simp⟩
    have ht : (cellTicket q.2.2.2 : ℝ) ≠ 0 := by exact_mod_cast (cellTicket_pos (n_pos_of_tile hrc)).ne'
    unfold untiedWeight
    rw [Rat.cast_div, Rat.cast_one, one_div, ← mul_assoc, inv_mul_cancel₀ ht, one_mul]

theorem units_nonneg (R : Run) : ∀ t ∈ cellUnits R, 0 ≤ t.1 ∧ 0 ≤ t.2.1 := by
  intro t ht
  unfold cellUnits at ht
  obtain ⟨q, -, hq⟩ := List.mem_filterMap.1 ht
  split_ifs at hq with h0
  cases hq
  exact ⟨div_nonneg zero_le_one (cellTicket_nonneg _), mul_nonneg (Nat.cast_nonneg _) (cellTicket_nonneg _)⟩

theorem units_taken (R : Run) : ∀ t ∈ cellUnits R, 0 ≤ t.2.2 ∧ t.2.2 ≤ (t.2.1 : ℝ) := by
  intro t ht
  unfold cellUnits at ht
  obtain ⟨q, -, hq⟩ := List.mem_filterMap.1 ht
  split_ifs at hq with h0
  cases hq
  have h1 := sum_ut_le_cells R q
  have h2 : (0 : ℝ) ≤ cellTicket q.2.2.2 := by exact_mod_cast cellTicket_nonneg _
  have h3 := sum_map_nonneg' (fun c => (untiedTileCells R c : ℝ)) (recordTiles q) fun c _ => Nat.cast_nonneg _
  refine ⟨mul_nonneg h2 h3, ?_⟩
  show (cellTicket q.2.2.2 : ℝ) * ((recordTiles q).map fun c => (untiedTileCells R c : ℝ)).sum ≤
    (((recordCells q.2.1 q.2.2.2 : ℚ) * cellTicket q.2.2.2 : ℚ) : ℝ)
  rw [Rat.cast_mul, Rat.cast_natCast]
  nlinarith

theorem untiedSum_units (R : Run) : untiedSum R = ((cellUnits R).map fun t => t.2.2).sum := by
  unfold untiedSum
  rw [sum_cells_eq, sum_units]
  congr 1
  apply List.map_congr_left
  intro q _
  rw [sum_ticket_left q (fun c => (untiedTileCells R c : ℝ))]
  split_ifs with h0
  · have h1 := sum_ut_le_cells R q
    have h2 := sum_map_nonneg' (fun c => (untiedTileCells R c : ℝ)) (recordTiles q) fun c _ => Nat.cast_nonneg _
    rw [h0, Nat.cast_zero] at h1
    rw [le_antisymm h1 h2, mul_zero]
  · rfl

theorem ticketSum_units (R : Run) : ticketSum R = ((cellUnits R).map fun t => (t.2.1 : ℝ)).sum := by
  unfold ticketSum
  rw [sum_cells_eq, sum_units]
  congr 1
  apply List.map_congr_left
  intro q _
  rw [sum_ticket_left q (fun c => (tileCells c : ℝ)), sum_cnt_recordTiles]
  split_ifs with h0
  · simp [h0]
  · push_cast
    ring

theorem units_w (R : Run) :
    ((((cellUnits R).map fun t => (t.1, t.2.1)).map (·.2)).sum : ℝ) = ticketSum R := by
  rw [List.map_map]
  exact (ticketSum_units R).symm

theorem card_le_units (R : Run) :
    (R.untiedCredited.card : ℝ) ≤ ((cellUnits R).map fun t => (t.1 : ℝ) * t.2.2).sum := by
  have h : (R.untiedCredited.card : ℝ) ≤ (((cells R.passes).map (untiedTileCells R)).sum : ℝ) :=
    Nat.cast_le.2 (card_untiedCredited_le R)
  rw [cast_sum_nat, sum_cells_eq] at h
  rw [sum_units]
  refine h.trans (le_of_eq ?_)
  congr 1
  apply List.map_congr_left
  intro q _
  exact sum_ut_eq R q

/-! ## The statement -/

theorem untiedCellsAtMost : UntiedCellsAtMost := by
  intro R ε hε
  have hb : ((cellUnits R).map fun t => t.2.2).sum ≤
      ε * ((((cellUnits R).map fun t => (t.1, t.2.1)).map (·.2)).sum : ℝ) := by
    rw [units_w, ← untiedSum_units]
    exact untiedSum_le R ε hε
  have hfill := sum_le_fill (cellUnits R) _ (units_nonneg R) (units_taken R) hb
  unfold cellsAt atBy
  rw [unitsBy_one R]
  have hperm : ((((cellUnits R).map fun t => (t.1, t.2.1)).mergeSort fun a b => decide (b.1 ≤ a.1)).map
      fun x => (x.2 : ℝ)).sum = (((cellUnits R).map fun t => (t.1, t.2.1)).map fun x => (x.2 : ℝ)).sum :=
    ((List.mergeSort_perm _ _).map _).sum_eq
  rw [hperm]
  exact (card_le_units R).trans hfill

end Properties.Pouw.PearlPlus
