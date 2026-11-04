import Proofs.Pouw.PearlPlus.Assembly
import Proofs.Pouw.PearlPlus.Rules
import Definitions.Pouw.PearlC.TileDraw

/-!
# Pearl+'s bridge, the draw: the run's tickets and acceptance as the game's draw reads them

`Definitions/Pouw/PearlC/TileDraw.lean` states γ for the draw the run makes: each tile's tickets from the declarations
its unit's record makes before the coin (`drawTickets`), and an audit that accepts a drawn tile that is good and whose
declaration is consistent with it (`goodDeclTiles`). This file reads the code's draw and acceptance in those terms.

* (c) A passed tile's declaration is consistent: a tile that `checkTile` passes has `declOK` under its record's
  declaration (`declOf`). Its kept width is within `k` (`LoadedGivesUp`), and each credited A row it declares passes
  `RowOK`: `checkOpened` holds the row to `row_passes`, on the words its opening parses to and their `s5_stats`, which
  `RowsAreTheRules` reads as `RowOK` at the committed activations.
* (d) The code's tickets are `drawTickets`: on a run laid out as the game's workload, a bijection from `tickets`'
  entries onto the tiles with tickets maps each entry to its `TileOf` tile, its ticket that tile's `drawTickets` of
  `pearlCDrawRev1 devH100` under the records' declarations (`decOf`). Its parts: `wref 1 k n` is
  `wrefRowRev1 devH100 k n`, `ticketWeight` is the weight's numerator and denominator, and the run's lcm fold is
  `Finset.lcm` over the tiles.
* (d) Acceptance is the tolerant audit's: on an accepted run, every sample of the game on a drawn tile is in
  `goodDeclTiles ∪ blankTiles` or over its cap, or SHA-512 collides.

The run's layout is what the game reads of the run: each pass tiles 64 × 64 (a manifest's `instance` may set another
tile), each record is a committed unit of the workload with its shape and weight, and no two records share a call index.
The game's transcript has one salt, so the run is one salt's; `passesDistinct` keys call indices per salt.
-/

namespace Pouw.PearlPlus.Draw

open PouwCheck.PearlPlus

/-! ## Lists -/

theorem getElem?_flatMap_const {α β : Type} (f : α → List β) (c : ℕ) (hc : ∀ a, (f a).length = c) :
    ∀ (L : List α) (x : ℕ), (L.flatMap f)[x]? =
      if x < L.length * c then L[x / c]?.bind fun a => (f a)[x % c]? else none
  | [], x => by simp
  | a :: L, x => by
    rw [List.flatMap_cons, List.length_cons]
    by_cases hx : x < c
    · rw [List.getElem?_append_left (by rw [hc]; exact hx), ite_eq_left (by rw [Nat.succ_mul]; omega),
        Nat.div_eq_of_lt hx, Nat.mod_eq_of_lt hx]
      rfl
    · rcases Nat.eq_zero_or_pos c with hc0 | hc0
      · subst hc0
        have h0 : ∀ L' : List α, (L'.flatMap f).length = 0 := fun L' => by
          induction L' with
          | nil => rfl
          | cons b L' ih => rw [List.flatMap_cons, List.length_append, hc, ih]
        rw [ite_eq_right (by omega), List.getElem?_eq_none (by rw [List.length_append, hc, h0]; omega)]
      rw [List.getElem?_append_right (by rw [hc]; omega), hc, getElem?_flatMap_const f c hc L (x - c)]
      obtain ⟨y, rfl⟩ : ∃ y, x = c + y := ⟨x - c, by omega⟩
      rw [Nat.add_sub_cancel_left, Nat.add_div_left _ hc0, Nat.add_mod_left]
      by_cases hy : y < L.length * c
      · rw [ite_eq_left hy, ite_eq_left (by rw [Nat.succ_mul]; omega)]
        rfl
      · rw [ite_eq_right hy, ite_eq_right (by rw [Nat.succ_mul]; omega)]

theorem length_flatMap_range {β : Type} (f : ℕ → List β) (N : ℕ) :
    ((List.range N).flatMap f).length = ((List.range N).map fun v => (f v).length).sum := by
  induction N with
  | zero => rfl
  | succ N ih => rw [List.range_succ, List.flatMap_append, List.map_append, List.length_append, ih]; simp

theorem getElem?_flatMap_range {β : Type} (f : ℕ → List β) :
    ∀ (N u x : ℕ), u < N → x < (f u).length →
      ((List.range N).flatMap f)[((List.range u).map fun v => (f v).length).sum + x]? = (f u)[x]?
  | 0, _, _, hu, _ => absurd hu (Nat.not_lt_zero _)
  | N + 1, u, x, hu, hx => by
    rw [List.range_succ, List.flatMap_append]
    rcases Nat.lt_succ_iff_lt_or_eq.1 hu with hu | rfl
    · have hlt : ((List.range u).map fun v => (f v).length).sum + x < ((List.range N).flatMap f).length := by
        rw [length_flatMap_range]
        obtain ⟨d, rfl⟩ : ∃ d, N = u + 1 + d := ⟨N - (u + 1), by omega⟩
        rw [List.range_add, List.map_append, List.sum_append, List.range_succ, List.map_append, List.sum_append]
        simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
        omega
      rw [List.getElem?_append_left hlt]
      exact getElem?_flatMap_range f N u x hu hx
    · rw [List.getElem?_append_right (by rw [length_flatMap_range]; omega), length_flatMap_range,
        Nat.add_sub_cancel_left]
      simp

theorem getElem?_flatMap_range_inv {β : Type} (f : ℕ → List β) :
    ∀ (N g : ℕ) (b : β), ((List.range N).flatMap f)[g]? = some b →
      ∃ u < N, ∃ x < (f u).length, g = ((List.range u).map fun v => (f v).length).sum + x ∧ (f u)[x]? = some b
  | 0, _, _, h => by simp at h
  | N + 1, g, b, h => by
    rw [List.range_succ, List.flatMap_append] at h
    by_cases hg : g < ((List.range N).flatMap f).length
    · rw [List.getElem?_append_left hg] at h
      obtain ⟨u, hu, x, hx, rfl, hb⟩ := getElem?_flatMap_range_inv f N g b h
      exact ⟨u, by omega, x, hx, rfl, hb⟩
    · rw [List.getElem?_append_right (by omega)] at h
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil] at h
      have hx : g - ((List.range N).flatMap f).length < (f N).length := by
        rcases Nat.lt_or_ge (g - ((List.range N).flatMap f).length) (f N).length with h' | h'
        · exact h'
        · rw [List.getElem?_eq_none h'] at h; cases h
      refine ⟨N, by omega, _, hx, ?_, h⟩
      rw [← length_flatMap_range]; omega

theorem nodup_flatMap_zipIdx {α β : Type} (key : β → ℕ) (F : α × ℕ → List β)
    (hk : ∀ q, ∀ b ∈ F q, key b = q.2) (hF : ∀ q, (F q).Nodup) :
    ∀ (l : List α) (n : ℕ), ((l.zipIdx n).flatMap F).Nodup ∧ ∀ b ∈ (l.zipIdx n).flatMap F, n ≤ key b
  | [], n => ⟨List.nodup_nil, fun _ h => absurd h List.not_mem_nil⟩
  | a :: l, n => by
    obtain ⟨h1, h2⟩ := nodup_flatMap_zipIdx key F hk hF l (n + 1)
    rw [List.zipIdx_cons, List.flatMap_cons]
    refine ⟨List.nodup_append.2 ⟨hF (a, n), h1, fun b hb b' hb' e => ?_⟩, fun b hb => ?_⟩
    · subst e
      have := hk (a, n) b hb
      have := h2 b hb'
      simp only at *
      omega
    · rcases List.mem_append.1 hb with hb | hb
      · exact Nat.le_of_eq (hk (a, n) b hb).symm
      · have := h2 b hb
        omega

theorem nodup_map_zipIdx {α β : Type} (key : β → ℕ) (f : α × ℕ → β) (hk : ∀ q, key (f q) = q.2) :
    ∀ (l : List α) (n : ℕ), ((l.zipIdx n).map f).Nodup ∧ ∀ b ∈ (l.zipIdx n).map f, n ≤ key b
  | [], n => ⟨List.nodup_nil, fun _ h => absurd h List.not_mem_nil⟩
  | a :: l, n => by
    obtain ⟨h1, h2⟩ := nodup_map_zipIdx key f hk l (n + 1)
    rw [List.zipIdx_cons, List.map_cons]
    refine ⟨List.nodup_cons.2 ⟨fun hb => ?_, h1⟩, fun b hb => ?_⟩
    · have := h2 _ hb
      rw [hk] at this
      simp only at this
      omega
    · rcases List.mem_cons.1 hb with rfl | hb
      · exact Nat.le_of_eq (hk (a, n)).symm
      · have := h2 b hb
        omega

/-! ## A pass's tiles -/

def tileLoop (s : Scheme) (tid : ℕ) (L : List (UnitRecord × ℕ)) (base : ℕ) : Option (ℕ × List ℕ × List ℕ) :=
  Id.run do
  let mut base := base
  for (r, pos) in L do
    let grid := s.tiles r.matmul.m r.matmul.n
    if tid < base + grid.length then
      let (rows, cols) := grid[tid - base]!
      return some (pos, rows, cols)
    base := base + grid.length
  return none

theorem tile?_eq (p : Pass) (tid : ℕ) : p.tile? tid = tileLoop p.scheme tid p.records.zipIdx 0 := rfl

theorem tileLoop_cons (s : Scheme) (tid base : ℕ) (r : UnitRecord) (pos : ℕ) (L : List (UnitRecord × ℕ)) :
    tileLoop s tid ((r, pos) :: L) base =
      if tid < base + (s.tiles r.matmul.m r.matmul.n).length then
        some (pos, (s.tiles r.matmul.m r.matmul.n)[tid - base]!)
      else tileLoop s tid L (base + (s.tiles r.matmul.m r.matmul.n).length) := by
  unfold tileLoop
  simp only [List.forIn_cons, Id.run]
  split <;> rfl

/-- Record `r`'s tiles in scheme `s`. -/
def grid (s : Scheme) (r : UnitRecord) : List (List ℕ × List ℕ) := s.tiles r.matmul.m r.matmul.n

/-- The number of tiles before record `pos` of `rs`. -/
def before (s : Scheme) (rs : List UnitRecord) (pos : ℕ) : ℕ := ((rs.take pos).map fun r => (grid s r).length).sum

theorem before_zero (s : Scheme) (r : UnitRecord) (rs : List UnitRecord) : before s (r :: rs) 0 = 0 := rfl

theorem before_succ (s : Scheme) (r : UnitRecord) (rs : List UnitRecord) (j : ℕ) :
    before s (r :: rs) (j + 1) = (grid s r).length + before s rs j := by
  simp only [before, List.take_succ_cons, List.map_cons, List.sum_cons]

theorem tileLoop_at (s : Scheme) : ∀ (rs : List UnitRecord) (n b pos x : ℕ) (hp : pos < rs.length),
    x < (grid s rs[pos]).length →
      tileLoop s (b + before s rs pos + x) (rs.zipIdx n) b = some (n + pos, (grid s rs[pos])[x]!)
  | [], _, _, _, _, hp, _ => absurd hp (Nat.not_lt_zero _)
  | r :: rs, n, b, 0, x, _, hx => by
    rw [List.zipIdx_cons, tileLoop_cons, before_zero]
    have hx' : b + 0 + x < b + (s.tiles r.matmul.m r.matmul.n).length := by
      simp only [grid, List.getElem_cons_zero] at hx; omega
    rw [ite_eq_left hx']
    simp only [grid, List.getElem_cons_zero, Nat.add_zero]
    congr 3; omega
  | r :: rs, n, b, pos + 1, x, hp, hx => by
    rw [List.zipIdx_cons, tileLoop_cons, before_succ]
    have hx' : ¬ (b + ((grid s r).length + before s rs pos) + x < b + (s.tiles r.matmul.m r.matmul.n).length) := by
      simp only [grid]; omega
    rw [ite_eq_right hx']
    have h := tileLoop_at s rs (n + 1) (b + (s.tiles r.matmul.m r.matmul.n).length) pos x
      (by simpa using hp) (by simpa using hx)
    have e : b + (s.tiles r.matmul.m r.matmul.n).length + before s rs pos + x =
        b + ((grid s r).length + before s rs pos) + x := by simp only [grid]; omega
    rw [e] at h
    rw [h]
    simp only [List.getElem_cons_succ]
    congr 2; omega

theorem tileLoop_some (s : Scheme) : ∀ (rs : List UnitRecord) (n b tid q : ℕ) (rows cols : List ℕ), b ≤ tid →
    tileLoop s tid (rs.zipIdx n) b = some (q, rows, cols) →
      ∃ pos, ∃ hp : pos < rs.length, ∃ x < (grid s rs[pos]).length, q = n + pos ∧ tid = b + before s rs pos + x ∧
        (grid s rs[pos])[x]! = (rows, cols)
  | [], _, _, _, _, _, _, _, h => by simp [tileLoop] at h
  | r :: rs, n, b, tid, q, rows, cols, hb, h => by
    rw [List.zipIdx_cons, tileLoop_cons] at h
    by_cases ht : tid < b + (s.tiles r.matmul.m r.matmul.n).length
    · rw [ite_eq_left ht] at h
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, h⟩ := h
      refine ⟨0, by simp, tid - b, by simp only [grid, List.getElem_cons_zero]; omega, by omega,
        by rw [before_zero]; omega, ?_⟩
      simp only [grid, List.getElem_cons_zero]
      exact h
    · rw [ite_eq_right ht] at h
      obtain ⟨pos, hp, x, hx, hq, htid, hg⟩ := tileLoop_some s rs (n + 1) _ tid q rows cols (by omega) h
      refine ⟨pos + 1, by simpa using hp, x, by simpa using hx, by omega, ?_, by simpa using hg⟩
      rw [before_succ]; simp only [grid] at htid ⊢; omega


theorem offsets_fold (s : Scheme) : ∀ (rs : List UnitRecord) (A : List ℕ) (b : ℕ),
    (rs.foldl (fun (acc : List ℕ × ℕ) r => (acc.1 ++ [acc.2], acc.2 + (s.tiles r.matmul.m r.matmul.n).length))
      (A, b)).1 = A ++ (List.range rs.length).map fun j => b + before s rs j
  | [], A, b => by simp
  | r :: rs, A, b => by
    rw [List.foldl_cons, offsets_fold s rs, List.length_cons, List.range_succ_eq_map, List.map_cons, List.map_map]
    simp only [before_zero, Nat.add_zero, List.append_assoc, List.cons_append, List.nil_append]
    congr 2
    apply List.map_congr_left
    intro j _
    simp only [Function.comp, before_succ, grid]
    omega

theorem offsets_get (p : Pass) {pos : ℕ} (hp : pos < p.records.length) :
    p.offsets[pos]! = before p.scheme p.records pos := by
  have e : p.offsets = (List.range p.records.length).map fun j => 0 + before p.scheme p.records j := by
    have := offsets_fold p.scheme p.records [] 0
    rw [List.nil_append] at this
    exact this
  rw [e, getElem!_pos _ pos (by simpa using hp), List.getElem_map, List.getElem_range, Nat.zero_add]

/-- Tile `x` of record `pos` is the tile whose id is the record's offset plus `x`. -/
theorem tile?_offsets (p : Pass) {pos x : ℕ} (hp : pos < p.records.length)
    (hx : x < (grid p.scheme p.records[pos]).length) :
    p.tile? (p.offsets[pos]! + x) = some (pos, (grid p.scheme p.records[pos])[x]!) := by
  have h := tileLoop_at p.scheme p.records 0 0 pos x hp hx
  rw [Nat.zero_add, Nat.zero_add] at h
  rw [tile?_eq, offsets_get p hp, h]

/-- A tile the pass numbers is tile `x` of a record `pos`, its id the record's offset plus `x`. -/
theorem tile?_some {p : Pass} {tid pos : ℕ} {rows cols : List ℕ} (h : p.tile? tid = some (pos, rows, cols)) :
    ∃ hp : pos < p.records.length, ∃ x < (grid p.scheme p.records[pos]).length, tid = p.offsets[pos]! + x ∧
      (grid p.scheme p.records[pos])[x]! = (rows, cols) := by
  rw [tile?_eq] at h
  obtain ⟨pos', hp, x, hx, hq, htid, hg⟩ := tileLoop_some p.scheme p.records 0 0 tid pos rows cols (Nat.zero_le _) h
  rw [Nat.zero_add] at hq
  subst hq
  exact ⟨hp, x, hx, by rw [offsets_get p hp]; omega, hg⟩


/-- The number of blocks of `w` that cover `len` positions, as `Scheme.tiles` counts them. -/
def nblocks (len w : ℕ) : ℕ := (len + w - 1) / w

/-- Block `b` of `len` positions in blocks of `w`, clipped at the end, as `Scheme.tiles` cuts it. -/
def block (len w b : ℕ) : List ℕ := List.range' (b * w) (min w (len - b * w))

theorem tiles_getElem? (s : Scheme) (m n x : ℕ) :
    (s.tiles m n)[x]? = if x < nblocks m s.tileM * nblocks n s.tileN then
      some (block m s.tileM (x / nblocks n s.tileN), block n s.tileN (x % nblocks n s.tileN)) else none := by
  unfold Scheme.tiles
  dsimp only
  rw [getElem?_flatMap_const _ (nblocks n s.tileN) (fun _ => by simp [nblocks]), List.length_map, List.length_range]
  by_cases hx : x < nblocks m s.tileM * nblocks n s.tileN
  · rw [ite_eq_left (by simpa [nblocks] using hx), ite_eq_left hx]
    have hn : 0 < nblocks n s.tileN := Nat.pos_of_ne_zero fun h => by rw [h, Nat.mul_zero] at hx; omega
    have hq : x / nblocks n s.tileN < nblocks m s.tileM := (Nat.div_lt_iff_lt_mul hn).2 hx
    have h1 : (List.range ((m + s.tileM - 1) / s.tileM))[x / nblocks n s.tileN]? = some (x / nblocks n s.tileN) :=
      List.getElem?_range hq
    have h2 : (List.range ((n + s.tileN - 1) / s.tileN))[x % nblocks n s.tileN]? = some (x % nblocks n s.tileN) :=
      List.getElem?_range (Nat.mod_lt x hn)
    rw [List.getElem?_map, h1]
    simp only [Option.map_some, Option.bind_some, List.getElem?_map, h2]
    rfl
  · rw [ite_eq_right (by simpa [nblocks] using hx), ite_eq_right hx]

/-! ## The run's tiles -/

/-- A tile of the run as `tickets` enumerates it: tile `x` (rows `rows`, columns `cols`) of record `pos` (`r`) of pass
`i` (`p`). -/
structure Cell where
  i : ℕ
  p : Pass
  pos : ℕ
  r : UnitRecord
  x : ℕ
  rows : List ℕ
  cols : List ℕ

/-- The run's tiles in `tickets`' order. -/
def cells (passes : List Pass) : List Cell :=
  passes.zipIdx.flatMap fun (p, i) =>
    p.records.zipIdx.flatMap fun (r, pos) =>
      (p.scheme.tiles r.matmul.m r.matmul.n).zipIdx.map fun ((rows, cols), x) => ⟨i, p, pos, r, x, rows, cols⟩

/-- A tile's id and ticket weight, as `tickets` lists them before scaling. -/
def Cell.weight (c : Cell) : (ℕ × ℕ) × ℕ × ℕ :=
  ((c.i, c.p.offsets[c.pos]! + c.x), ticketWeight c.r c.rows c.cols)

/-- The lcm of the weights' denominators, as `tickets` folds it. -/
def lcmOf (ws : List ((ℕ × ℕ) × ℕ × ℕ)) : ℕ := ws.foldl (fun l (_, (_, d)) => Nat.lcm l d) 1

theorem tickets_eq (passes : List Pass) : tickets passes = ((cells passes).map Cell.weight).map
    fun e => (e.1, e.2.1 * (lcmOf ((cells passes).map Cell.weight) / e.2.2)) := by
  have hw : ((cells passes).map Cell.weight) = passes.zipIdx.flatMap fun (p, i) =>
      p.records.zipIdx.flatMap fun (r, pos) => (p.scheme.tiles r.matmul.m r.matmul.n).zipIdx.map
        fun ((rows, cols), x) => ((i, p.offsets[pos]! + x), ticketWeight r rows cols) := by
    simp only [cells, List.map_flatMap, List.map_map]
    rfl
  rw [hw]
  rfl

theorem mem_cells {passes : List Pass} {c : Cell} :
    c ∈ cells passes ↔ passes[c.i]? = some c.p ∧ c.p.records[c.pos]? = some c.r ∧
      (c.p.scheme.tiles c.r.matmul.m c.r.matmul.n)[c.x]? = some (c.rows, c.cols) := by
  obtain ⟨i, p, pos, r, x, rows, cols⟩ := c
  simp only [cells, List.mem_flatMap, List.mem_map, List.mem_zipIdx_iff_getElem?, Prod.exists]
  constructor
  · rintro ⟨p', i', hp, r', pos', hr, rows', cols', x', hx, h⟩
    simp only [Cell.mk.injEq] at h
    obtain ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ := h
    exact ⟨hp, hr, hx⟩
  · rintro ⟨hp, hr, hx⟩
    exact ⟨p, i, hp, r, pos, hr, rows, cols, x, hx, rfl⟩

theorem cells_nodup (passes : List Pass) : (cells passes).Nodup := by
  refine (nodup_flatMap_zipIdx Cell.i _ ?_ (fun q => ?_) passes 0).1
  · rintro ⟨p, i⟩ b hb
    simp only [List.mem_flatMap, List.mem_map, Prod.exists] at hb
    obtain ⟨_, _, _, _, _, _, _, rfl⟩ := hb
    rfl
  · obtain ⟨p, i⟩ := q
    refine (nodup_flatMap_zipIdx Cell.pos _ ?_ (fun q => ?_) p.records 0).1
    · rintro ⟨r, pos⟩ b hb
      simp only [List.mem_map, Prod.exists] at hb
      obtain ⟨_, _, _, _, rfl⟩ := hb
      rfl
    · obtain ⟨r, pos⟩ := q
      exact (nodup_map_zipIdx Cell.x _ (fun _ => rfl) _ 0).1

theorem foldl_lcm_dvd : ∀ (ws : List ((ℕ × ℕ) × ℕ × ℕ)) (a : ℕ),
    a ∣ ws.foldl (fun l (_, (_, d)) => Nat.lcm l d) a ∧ ∀ e ∈ ws, e.2.2 ∣ ws.foldl (fun l (_, (_, d)) => Nat.lcm l d) a
  | [], a => ⟨Nat.dvd_refl a, fun _ h => absurd h (List.not_mem_nil)⟩
  | e :: ws, a => by
    obtain ⟨h1, h2⟩ := foldl_lcm_dvd ws (Nat.lcm a e.2.2)
    rw [List.foldl_cons]
    refine ⟨Nat.dvd_trans (Nat.dvd_lcm_left a e.2.2) h1, fun e' he' => ?_⟩
    rcases List.mem_cons.1 he' with rfl | he'
    · exact Nat.dvd_trans (Nat.dvd_lcm_right a _) h1
    · exact h2 e' he'

theorem foldl_lcm_le : ∀ (ws : List ((ℕ × ℕ) × ℕ × ℕ)) (a M : ℕ), a ∣ M → (∀ e ∈ ws, e.2.2 ∣ M) →
    ws.foldl (fun l (_, (_, d)) => Nat.lcm l d) a ∣ M
  | [], _, _, ha, _ => ha
  | e :: ws, a, M, ha, h => by
    rw [List.foldl_cons]
    exact foldl_lcm_le ws _ M (Nat.lcm_dvd ha (h e List.mem_cons_self)) fun e' he' => h e' (List.mem_cons_of_mem _ he')

theorem dvd_lcmOf {ws : List ((ℕ × ℕ) × ℕ × ℕ)} {e : (ℕ × ℕ) × ℕ × ℕ} (he : e ∈ ws) :
    e.2.2 ∣ lcmOf ws := (foldl_lcm_dvd ws 1).2 e he

theorem lcmOf_dvd {ws : List ((ℕ × ℕ) × ℕ × ℕ)} {M : ℕ} (h : ∀ e ∈ ws, e.2.2 ∣ M) : lcmOf ws ∣ M :=
  foldl_lcm_le ws 1 M (Nat.one_dvd M) h


/-! ## A passed tile's rows -/

theorem forIn_none {α : Type} : ∀ (L : List α)
    (f : α → Option (Option String) × Unit → Id (ForInStep (Option (Option String) × Unit))),
    (∀ a, f a (none, ()) = .yield (none, ()) ∨ ∃ m, f a (none, ()) = .done (some (some m), ())) →
    (forIn L ((none : Option (Option String)), ()) f).fst = none → ∀ a ∈ L, f a (none, ()) = .yield (none, ())
  | [], _, _, _, a, ha => absurd ha List.not_mem_nil
  | b :: L, f, hf, h, a, ha => by
    rw [List.forIn_cons] at h
    rcases hf b with hb | ⟨m, hb⟩
    · rw [hb] at h
      rcases List.mem_cons.1 ha with rfl | ha
      · exact hb
      · exact forIn_none L f hf h a ha
    · rw [hb] at h
      simp at h

theorem forIn_fst {α : Type} : ∀ (L : List α)
    (f : α → Option (Option String) × Unit → Id (ForInStep (Option (Option String) × Unit))),
    (∀ a, f a (none, ()) = .yield (none, ()) ∨ ∃ m, f a (none, ()) = .done (some (some m), ())) →
    (forIn L ((none : Option (Option String)), ()) f).fst = none ∨
      ∃ m, (forIn L ((none : Option (Option String)), ()) f).fst = some (some m)
  | [], _, _ => Or.inl rfl
  | b :: L, f, hf => by
    rw [List.forIn_cons]
    rcases hf b with hb | ⟨m, hb⟩
    · rw [hb]; exact forIn_fst L f hf
    · rw [hb]; exact Or.inr ⟨m, rfl⟩

theorem forIn_some {α : Type} {L : List α}
    {f : α → Option (Option String) × Unit → Id (ForInStep (Option (Option String) × Unit))} {r : Option String}
    (hf : ∀ a, f a (none, ()) = .yield (none, ()) ∨ ∃ m, f a (none, ()) = .done (some (some m), ()))
    (h : (forIn L ((none : Option (Option String)), ()) f).fst = some r) : ∃ m, r = some m := by
  rcases forIn_fst L f hf with hn | ⟨m, hm⟩
  · rw [hn] at h; cases h
  · rw [hm] at h; exact ⟨m, (Option.some.inj h).symm⟩

theorem checkOpened_credited {s : Scheme} {r : UnitRecord} {rows cols : List ℕ} {ops : Words.Ops}
    (h : Rule.checkOpened s r rows cols ops = none) {x : ℕ} (hx : x < rows.length)
    (hc : (decide (rows[x] < r.realRows) && !r.excludedA.contains rows[x] && !r.voluntaryA.contains rows[x]) = true) :
    Rule.rowPasses ops.a.rho[x]! ops.a.alpha[x]! ops.a.x[x]! = true := by
  unfold Rule.checkOpened at h
  dsimp only [Id.run, bind, pure] at h
  split at h
  · rename_i r' heq
    have hs := forIn_some ?_ heq
    · obtain ⟨m, rfl⟩ := hs
      cases h
    · intro a
      try dsimp only
      repeat' split
      all_goals first | exact Or.inl rfl | exact Or.inr ⟨_, rfl⟩
  · rename_i heq
    have hall := forIn_none _ _ ?_ heq
    rotate_left
    · intro a
      try dsimp only
      repeat' split
      all_goals first | exact Or.inl rfl | exact Or.inr ⟨_, rfl⟩
    simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_true'] at hc
    obtain ⟨⟨hlt, hex⟩, hvol⟩ := hc
    have hmem := hall ("A", rows[x], ops.a, x, Rule.Kind.credited) (List.mem_append_left _ (List.mem_map.2
      ⟨(rows[x], x), List.mem_zipIdx_iff_getElem?.2 (by simp [hx]), by
        dsimp only
        rw [ite_eq_right (Nat.not_le.2 hlt), ite_eq_right (by rw [hex]; decide),
          ite_eq_right (by rw [hvol]; decide)]⟩))
    dsimp only at hmem
    simp only [show (Rule.Kind.credited == Rule.Kind.filler) = false from rfl,
      show (Rule.Kind.credited == Rule.Kind.voluntary) = false from rfl,
      show (Rule.Kind.credited == Rule.Kind.credited) = true from rfl,
      show (Rule.Kind.credited == Rule.Kind.excluded) = false from rfl] at hmem
    simp only [Bool.false_eq_true, ↓reduceIte, Bool.true_and, Bool.false_and, Bool.or_false] at hmem
    split at hmem
    · cases hmem
    · split at hmem
      · cases hmem
      · rename_i hrp
        simpa using hrp

/-- A `for` loop that went through every element of `L` from index `n` keeps an invariant each step keeps, a step
reading its element by its index. -/
theorem forIn_inv_idx {α σ ε : Type} (I : ℕ → σ → Prop) (body : α → σ → Except ε (ForInStep σ)) :
    ∀ (L : List α) (n : ℕ) (s out : σ), I n s →
      (∀ j a s r, L[j]? = some a → I (n + j) s → body a s = .ok r → ∃ t, r = .yield t ∧ I (n + j + 1) t) →
      forIn L s body = .ok out → I (n + L.length) out
  | [], n, s, out, hs, _, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    simpa using hs
  | a :: L, n, s, out, hs, hb, h => by
    rw [List.forIn_cons] at h
    obtain ⟨r, hr, h⟩ := Assembly.bind_ok h
    obtain ⟨t, rfl, ht⟩ := hb 0 a s r rfl hs hr
    have := forIn_inv_idx I body L (n + 1) t out ht (fun j a' s' r' hj hs' hr' => by
      have := hb (j + 1) a' s' r' (by simpa using hj) (by simpa [Nat.add_assoc, Nat.add_comm 1] using hs') hr'
      simpa [Nat.add_assoc, Nat.add_comm 1] using this) h
    simpa [List.length_cons, Nat.add_assoc, Nat.add_comm 1] using this

theorem push_get!_lt {β : Type} [Inhabited β] (a : Array β) (v : β) {j : ℕ} (hj : j < a.size) :
    (a.push v)[j]! = a[j]! := by
  rw [getElem!_pos (a.push v) j (by simp; omega), getElem!_pos a j hj, Array.getElem_push_lt]

theorem push_get!_eq {β : Type} [Inhabited β] (a : Array β) (v : β) : (a.push v)[a.size]! = v := by
  rw [getElem!_pos (a.push v) a.size (by simp), Array.getElem_push_eq]

/-- Each row `formV1` forms keeps its words, and its ρ and α are `s5_stats`' of them. -/
theorem formV1_rows {rows eRows f : Array (Array ℕ)} {out : Words.Formed} (h : Words.formV1 rows eRows f = .ok out) :
    ∀ j < (rows.zip eRows).size, out.x[j]! = (rows.zip eRows)[j]!.1 ∧
      ∃ s β, Words.s5Stats out.x[j]! = .ok (s, out.rho[j]!, out.alpha[j]!, β) := by
  unfold Words.formV1 at h
  obtain ⟨r, hr, h⟩ := Assembly.bind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  rw [← Array.forIn_toList] at hr
  have := forIn_inv_idx (fun n (o : Words.Formed) => o.x.size = n ∧ o.rho.size = n ∧ o.alpha.size = n ∧
      ∀ j < n, o.x[j]! = (rows.zip eRows).toList[j]!.1 ∧
        ∃ s β, Words.s5Stats o.x[j]! = .ok (s, o.rho[j]!, o.alpha[j]!, β)) _ _ 0 _ _
    ⟨rfl, rfl, rfl, fun j hj => absurd hj (Nat.not_lt_zero _)⟩ ?_ hr
  · intro j hj
    obtain ⟨-, -, -, hall⟩ := this
    have hj' : j < (rows.zip eRows).toList.length := by simpa using hj
    obtain ⟨h1, h2⟩ := hall j (by omega)
    refine ⟨?_, h2⟩
    rw [h1, getElem!_pos _ j hj', getElem!_pos _ j hj]
    simp
  · intro n a s r ha hs hr
    obtain ⟨x, e⟩ := a
    obtain ⟨st, hst, hr⟩ := Assembly.bind_ok hr
    obtain ⟨fst, rho, alpha, beta⟩ := st
    obtain ⟨ep, -, hr⟩ := Assembly.bind_ok hr
    obtain ⟨codes, -, hr⟩ := Assembly.bind_ok hr
    simp only [pure, Except.pure, Except.ok.injEq] at hr
    subst hr
    obtain ⟨hx, hrho, halpha, hall⟩ := hs
    refine ⟨_, rfl, by simp [hx], by simp [hrho], by simp [halpha], fun j hj => ?_⟩
    simp only [Nat.zero_add] at hx hrho halpha hall hj ha ⊢
    rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | hj
    · rw [push_get!_lt _ _ (by omega), push_get!_lt _ _ (by omega), push_get!_lt _ _ (by omega)]
      exact hall j hj
    · have g1 : (s.x.push x)[j]! = x := by rw [hj, ← hx]; exact push_get!_eq _ _
      have g2 : (s.rho.push rho)[j]! = rho := by rw [hj, ← hrho]; exact push_get!_eq _ _
      have g3 : (s.alpha.push alpha)[j]! = alpha := by rw [hj, ← halpha]; exact push_get!_eq _ _
      rw [g1, g2, g3]
      refine ⟨?_, fst, beta, hst⟩
      rw [List.getElem!_eq_getElem?_getD, hj, ha]
      rfl


/-- The A rows `encode` forms are the parsed openings, and their ρ and α are `s5_stats`' of them. -/
theorem encode_rowA {salt rootB : ByteArray} {wn : Words.FixedNoise} {index k : ℕ} {rows : List ℕ}
    {aRows : List ByteArray} {cols : List ℕ} {bRows : List ByteArray} {ops : Words.Ops}
    (ha : aRows.length = rows.length) (h : Words.encode salt rootB wn index k rows aRows cols bRows = .ok ops) :
    ∀ x < rows.length, Words.parseRow aRows[x]! k = .ok ops.a.x[x]! ∧
      ∃ s β, Words.s5Stats ops.a.x[x]! = .ok (s, ops.a.rho[x]!, ops.a.alpha[x]!, β) := by
  unfold Words.encode at h
  obtain ⟨ea, hea, h⟩ := Assembly.bind_ok h
  obtain ⟨xa, hxa, h⟩ := Assembly.bind_ok h
  obtain ⟨a, hfa, h⟩ := Assembly.bind_ok h
  obtain ⟨eb, -, h⟩ := Assembly.bind_ok h
  obtain ⟨xb, -, h⟩ := Assembly.bind_ok h
  obtain ⟨b, -, h⟩ := Assembly.bind_ok h
  obtain ⟨pa, -, h⟩ := Assembly.bind_ok h
  obtain ⟨pb, -, h⟩ := Assembly.bind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  intro x hx
  have sea := (Assembly.array_mapM_ok _ _ _ hea).1
  obtain ⟨sxa, gxa⟩ := Assembly.array_mapM_ok _ _ _ hxa
  simp only [List.size_toArray, List.length_zip] at sea sxa
  have hz : x < (xa.zip ea).size := by simp only [Array.size_zip]; omega
  obtain ⟨h1, h2⟩ := formV1_rows hfa x hz
  refine ⟨?_, h2⟩
  rw [h1, getElem!_pos (xa.zip ea) x hz, Array.getElem_zip]
  have := gxa x (by simp; omega)
  rw [getElem!_pos _ x (by simp; omega), List.getElem_toArray, getElem!_pos xa x (by omega)] at this
  rw [getElem!_pos aRows x (by omega)]
  exact this

/-- The bytes of `k` words, each little-endian in four bytes. -/
def wordBytes (w : ℕ → ℕ) (k : ℕ) : ByteArray :=
  ⟨(((List.range k).flatMap fun l => (List.range 4).map fun b => w l / 256 ^ b % 256).map Nat.toUInt8).toArray⟩

theorem flatMap4 {β : Type} (f : ℕ → ℕ → β) : ∀ k : ℕ,
    ((List.range k).flatMap fun l => (List.range 4).map (f l)).length = 4 * k ∧
      ∀ l < k, ∀ b < 4, ((List.range k).flatMap fun l => (List.range 4).map (f l))[4 * l + b]? = some (f l b)
  | 0 => ⟨rfl, fun _ h => absurd h (Nat.not_lt_zero _)⟩
  | k + 1 => by
    obtain ⟨hlen, hget⟩ := flatMap4 f k
    rw [List.range_succ (n := k), List.flatMap_append, List.flatMap_singleton]
    refine ⟨by simp [hlen]; omega, fun l hl b hb => ?_⟩
    rcases Nat.lt_succ_iff_lt_or_eq.1 hl with hl | rfl
    · rw [List.getElem?_append_left (by rw [hlen]; omega)]
      exact hget l hl b hb
    · rw [List.getElem?_append_right (by rw [hlen]; omega), hlen, show 4 * l + b - 4 * l = b by omega]
      simp [hb]

theorem bytes_or (w : ℕ) : (w % 256 ||| (w / 256 % 256) <<< 8 ||| (w / 65536 % 256) <<< 16 |||
    (w / 16777216 % 256) <<< 24) = w % 2 ^ 32 := by
  have h1 : (w % 256 ||| (w / 256 % 256) <<< 8) = (w / 256 % 256) * 256 + w % 256 := by
    rw [Nat.or_comm, ← Nat.shiftLeft_add_eq_or_of_lt (by omega), Nat.shiftLeft_eq]
  have h2 : ((w / 256 % 256) * 256 + w % 256 ||| (w / 65536 % 256) <<< 16) =
      (w / 65536 % 256) * 65536 + ((w / 256 % 256) * 256 + w % 256) := by
    rw [Nat.or_comm, ← Nat.shiftLeft_add_eq_or_of_lt (by omega), Nat.shiftLeft_eq]
  have h3 : ((w / 65536 % 256) * 65536 + ((w / 256 % 256) * 256 + w % 256) ||| (w / 16777216 % 256) <<< 24) =
      (w / 16777216 % 256) * 16777216 + ((w / 65536 % 256) * 65536 + ((w / 256 % 256) * 256 + w % 256)) := by
    rw [Nat.or_comm, ← Nat.shiftLeft_add_eq_or_of_lt (by omega), Nat.shiftLeft_eq]
  rw [h1, h2, h3]
  omega

theorem parseRow_wordBytes {w : ℕ → ℕ} {k : ℕ} {x : Array ℕ} (h : Words.parseRow (wordBytes w k) k = .ok x) :
    x.size = k ∧ ∀ l < k, x[l]! = w l % 2 ^ 32 ∧ ((x[l]! >>> 23) &&& 0xFF == 0xFF) = false := by
  obtain ⟨hlen, hget⟩ := flatMap4 (fun l b => w l / 256 ^ b % 256) k
  have hsz : (wordBytes w k).size = 4 * k := by
    simp only [wordBytes, ByteArray.size, List.size_toArray, List.length_map]
    exact hlen
  unfold Words.parseRow at h
  simp only [hsz, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte] at h
  obtain ⟨hs, hg⟩ := Assembly.array_mapM_ok _ _ _ h
  refine ⟨by simpa using hs, fun l hl => ?_⟩
  have := hg l (by simpa using hl)
  have hb : ∀ b < 4, ((wordBytes w k)[4 * l + b]!).toNat = w l / 256 ^ b % 256 := by
    intro b hb
    have hi : 4 * l + b < (wordBytes w k).size := by rw [hsz]; omega
    rw [getElem!_pos (wordBytes w k) (4 * l + b) hi]
    have hg' := hget l hl b hb
    simp only [wordBytes, ByteArray.getElem_eq_getElem_data, List.getElem_toArray, List.getElem_map]
    rw [(List.getElem_eq_iff _).2 hg', Nat.toUInt8_eq, UInt8.toNat_ofNat', Nat.mod_mod]
  have h0 := hb 0 (by omega)
  have h1 := hb 1 (by omega)
  have h2 := hb 2 (by omega)
  have h3 := hb 3 (by omega)
  rw [getElem!_pos (Array.range k) l (by simpa using hl), Array.getElem_range] at this
  simp only [Nat.add_zero, Nat.pow_zero, Nat.div_one, Nat.pow_one] at h0 h1
  simp only [Nat.reducePow] at h2 h3
  rw [h0, h1, h2, h3, bytes_or] at this
  unfold Words.finite at this
  split at this
  · cases this
  · rename_i hf
    have hx : x[l]! = w l % 2 ^ 32 := (Except.ok.inj this).symm
    refine ⟨hx, ?_⟩
    rw [hx]
    simpa using hf

end Pouw.PearlPlus.Draw
