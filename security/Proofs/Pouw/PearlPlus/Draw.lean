module

public import Proofs.Pouw.PearlPlus.Assembly
public import Proofs.Pouw.PearlPlus.Rules
public import Proofs.Properties.Pouw.PearlPlus.Verdict
public import Definitions.Pouw.PearlC.TileDraw
public import Proofs.Pouw.PearlC.Records

/-!
# Pearl+'s bridge, the draw: the run's tickets and acceptance as the game's draw reads them

`Definitions/Pouw/PearlC/TileDraw.lean` states γ for the draw the run makes: each tile's tickets from the declarations
its unit's record makes before the coin (`drawTickets`), and an audit that accepts a drawn tile that is good and whose
declaration is consistent with it (`goodDeclTiles`). This file reads the code's draw and acceptance in those terms.

* (c) A passed tile's declaration is consistent: a tile that `checkTile` passes, or finds only over its inert-work
  bound, has `declOK` under its record's declaration (`declOf`). Its kept width is within `k` (a loaded pass's records
  keep `UnitRecord`'s rule, `load_facts`), and each credited A row it declares passes `RowOK`: `rowsFailure` holds the row to `row_passes`, on the words its opening parses to and their `s5_stats`, which
  `RowsAreTheRules` reads as `RowOK` at the committed activations.
* (d) The code's tickets are `drawTickets` (`ticketsAreTheDraw`): on a run laid out as the game's workload, an
  injective map from `tickets`' entries to the game's tiles, onto every tile with a ticket, takes each entry to its
  `TileOf` tile, its tickets that tile's `drawTickets` of `pearlCDrawRev1 devH100` under the records' declarations
  (`decOf`). Its parts: `wref 1 k n` is `wrefRowRev1 devH100 k n`, `ticketWeight` is the weight's numerator and
  denominator, and the run's lcm fold is `Finset.lcm` over the tiles.
* (d) Acceptance is the tolerant audit's (`acceptGivesTheAudit`): when the check accepts, every sample of the game on
  tiles the verdict opened is in `goodDeclTiles ∪ blankTiles` or over the inert-work bound (`overCapTiles`), or SHA-512
  collides on an opened tile. A sampled tile's check gives its words (`TilePassGivesTheWords`) and its declaration
  (`TilePassGivesTheDecl`), so it is good within the game's inert-work bound (`capOK`) and over the bound otherwise.
  The tile's weight rows passing the rules is a hypothesis (`DrawnOn.weights`), as in `TilePassGivesTheWords`: the
  check holds only credited B rows to the rules (an excluded row must fail them and a voluntary one is unread), where
  `TileGood` reads every column.

The run's layout (`RunLayout`) is what the game reads of the run: each pass tiles 64 × 64 (a manifest's `instance` may
set another tile), each record is a unit of the workload that the transcript commits, with the record's shape and
weight, and no two records share a call index. `passesDistinct` keys call indices per salt, so a run whose passes of
different salts share an index is not one game's run.
-/

@[expose] public section

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

theorem rowsFailure_credited {r : UnitRecord} {rows cols : List ℕ} {ops : Words.Ops}
    (h : Rule.rowsFailure r rows cols ops = none) {x : ℕ} (hx : x < rows.length)
    (hc : (decide (rows[x] < r.realRows) && !r.excludedA.contains rows[x] && !r.voluntaryA.contains rows[x]) = true) :
    Rule.rowPasses ops.a.rho[x]! ops.a.alpha[x]! ops.a.x[x]! = true := by
  unfold Rule.rowsFailure at h
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

open Pouw.PearlC Finset

/-! ## A record -/

/-- A record that loads keeps its real rows within `m` and its kept width within `k`, neither empty. -/
theorem reason_none {r : UnitRecord} (h : r.reason = none) :
    0 < r.realRows ∧ r.realRows ≤ r.matmul.m ∧ 0 < r.kKept ∧ r.kKept ≤ r.matmul.k := by
  unfold UnitRecord.reason at h
  dsimp only at h
  split at h
  · cases h
  · rename_i hc
    simp only [Bool.not_eq_true', Bool.not_eq_false, Bool.and_eq_true, decide_eq_true_eq] at hc
    omega

/-- A row's opening bytes are the row's words, four little-endian bytes each. -/
theorem rowBytesOf_eq (W : Codes) (k i : ℕ) : Bridge.rowBytesOf W k i = wordBytes (W i) k := rfl

/-! ## A tile's weight -/

theorem cast_sub_max (a b : ℕ) : ((a - b : ℕ) : ℚ) = max ((a : ℚ) - b) 0 := by
  rcases le_total b a with h | h
  · rw [Nat.cast_sub h, max_eq_left (by linarith [(Nat.cast_le (α := ℚ)).2 h])]
  · rw [Nat.sub_eq_zero_of_le h, max_eq_right (by linarith [(Nat.cast_le (α := ℚ)).2 h]), Nat.cast_zero]

theorem wref_one_cast (k n : ℕ) : (wref 1 k n : ℚ) =
    max ((n : ℚ) * k + 32 * n * ((k / 128 : ℕ) : ℚ) - 32 * n) 0 + (k : ℚ) * (32 : ℕ) + 2 * 2 * (32 : ℕ) * n +
      (((64 : ℕ) : ℚ) + ((4 : ℕ) : ℚ)) * k := by
  unfold wref
  simp only [FADD, WINDOW, PouwCheck.PearlPlus.R, BF16_MAC, PEEL_K, FORMING]
  push_cast [cast_sub_max]
  ring_nf

theorem wrefRowRev1_h100 (k n : ℕ) : wrefRowRev1 devH100 k n =
    max ((n : ℚ) * k + 32 * n * ((k / 128 : ℕ) : ℚ) - 32 * n) 0 + (k : ℚ) * (32 : ℕ) + 2 * 2 * (32 : ℕ) * n +
      (((64 : ℕ) : ℚ) + ((4 : ℕ) : ℚ)) * k := by
  have hG : devH100.G = 4 := rfl
  have hadd : devH100.prices.add = 32 := rfl
  have hbf : devH100.prices.bf16 = 2 := rfl
  have hfs : devH100.prices.costs.fs = 64 := rfl
  have hqa : devH100.prices.costs.qa = 4 := rfl
  have hr : Params.pi.r = 32 := rfl
  unfold wrefRowRev1
  rw [ite_eq_right (by rw [hG]; decide), hadd, hbf, hfs, hqa, hr, hG]

/-- The code's `wref(1, k, n)` is the draw's `wrefRowRev1 devH100 k n`, the natural subtraction's cut at zero
included. -/
theorem wref_one_eq (k n : ℕ) : (wref 1 k n : ℚ) = wrefRowRev1 devH100 k n := by
  rw [wref_one_cast, wrefRowRev1_h100]

theorem num_den_div (x n : ℕ) (hn : 0 < n) :
    ((x : ℚ) / n).num = ((x / Nat.gcd x n : ℕ) : ℤ) ∧ ((x : ℚ) / n).den = n / Nat.gcd x n := by
  have hg : 0 < Nat.gcd x n := Nat.gcd_pos_of_pos_right x hn
  have hcop : Nat.Coprime (x / Nat.gcd x n) (n / Nat.gcd x n) := Nat.coprime_div_gcd_div_gcd hg
  have hb0 : 0 < n / Nat.gcd x n := Nat.div_pos (Nat.gcd_le_right x hn) hg
  have hb : (0 : ℤ) < ((n / Nat.gcd x n : ℕ) : ℤ) := by exact_mod_cast hb0
  have he : (x : ℚ) / n = (((x / Nat.gcd x n : ℕ) : ℤ) : ℚ) / (((n / Nat.gcd x n : ℕ) : ℤ) : ℚ) := by
    rw [Int.cast_natCast, Int.cast_natCast, Nat.cast_div (Nat.gcd_dvd_left x n) (by exact_mod_cast hg.ne'),
      Nat.cast_div (Nat.gcd_dvd_right x n) (by exact_mod_cast hg.ne'),
      div_div_div_cancel_right₀ (by exact_mod_cast hg.ne')]
  rw [he]
  have hc : Nat.Coprime ((x / Nat.gcd x n : ℕ) : ℤ).natAbs ((n / Nat.gcd x n : ℕ) : ℤ).natAbs := by
    rw [Int.natAbs_natCast, Int.natAbs_natCast]; exact hcop
  refine ⟨Rat.num_div_eq_of_coprime hb hc, ?_⟩
  have := Rat.den_div_eq_of_coprime hb hc
  exact_mod_cast this

/-- The weight of a tile with `cr` credited rows and `cc` credited columns, as the draw weighs it. -/
def tileWeight (r : UnitRecord) (rows cols : List ℕ) : ℚ :=
  ((wref 1 r.kKept r.matmul.n * (r.credited rows cols).1.length * (r.credited rows cols).2.length : ℕ) : ℚ) /
    r.matmul.n

/-- `ticketWeight` is `tileWeight` in lowest terms. -/
theorem ticketWeight_eq (r : UnitRecord) (rows cols : List ℕ) (hn : 0 < r.matmul.n) :
    ticketWeight r rows cols = ((tileWeight r rows cols).num.toNat, (tileWeight r rows cols).den) := by
  obtain ⟨h1, h2⟩ := num_den_div
    (wref 1 r.kKept r.matmul.n * (r.credited rows cols).1.length * (r.credited rows cols).2.length) r.matmul.n hn
  rw [tileWeight, h1, h2, Int.toNat_natCast]
  rfl

theorem card_inter_filter {L : List ℕ} (hL : L.Nodup) (M : ℕ) (P : ℕ → Bool) (hM : ∀ i ∈ L, P i = true → i < M) :
    (L.toFinset ∩ (range M).filter fun i => P i = true).card = (L.filter P).length := by
  rw [← List.toFinset_card_of_nodup (hL.filter P)]
  congr 1
  ext i
  simp only [Finset.mem_inter, List.mem_toFinset, Finset.mem_filter, Finset.mem_range, List.mem_filter]
  constructor
  · rintro ⟨h, -, hp⟩
    exact ⟨h, hp⟩
  · rintro ⟨h, hp⟩
    exact ⟨h, hM i h hp, hp⟩

/-! ## The game's tiles -/

theorem block_nodup (len w b : ℕ) : (block len w b).Nodup := List.nodup_range' 1 Nat.one_pos

theorem mem_block {len w b i : ℕ} : i ∈ block len w b ↔ b * w ≤ i ∧ i < b * w + min w (len - b * w) := by
  rw [block, List.mem_range'_1]

theorem block_lt {len w b i : ℕ} (h : i ∈ block len w b) : i < len := by
  rw [mem_block] at h
  generalize b * w = a at h
  omega

theorem block_toFinset (len w b : ℕ) : (block len w b).toFinset = Ico (b * w) (min ((b + 1) * w) len) := by
  ext i
  rw [List.mem_toFinset, mem_block, Finset.mem_Ico, Nat.succ_mul]
  generalize b * w = a
  omega

theorem length_flatMap_const {α β : Type} (f : α → List β) (c : ℕ) (hc : ∀ a, (f a).length = c) :
    ∀ L : List α, (L.flatMap f).length = L.length * c
  | [] => by simp
  | a :: L => by
    rw [List.flatMap_cons, List.length_append, hc, length_flatMap_const f c hc L, List.length_cons, Nat.succ_mul,
      Nat.add_comm]

/-- Unit `u`'s tiles in `gridTiles 64 64`, in its order. -/
def unitGrid (L : Pouw.PearlC.Layout) (u : ℕ) : List (ℕ × ℕ × ℕ) :=
  (List.range (((L.shape u).m + 64 - 1) / 64)).flatMap fun bi =>
    (List.range (((L.shape u).n + 64 - 1) / 64)).map fun bj => (u, bi, bj)

theorem gridTiles_eq (L : Pouw.PearlC.Layout) : gridTiles 64 64 L = (List.range L.N).flatMap (unitGrid L) := rfl

theorem unitGrid_length (L : Pouw.PearlC.Layout) (u : ℕ) :
    (unitGrid L u).length = nblocks (L.shape u).m 64 * nblocks (L.shape u).n 64 := by
  rw [unitGrid, length_flatMap_const _ (nblocks (L.shape u).n 64) (fun _ => by simp [nblocks]), List.length_range]
  rfl

theorem unitGrid_getElem? (L : Pouw.PearlC.Layout) (u x : ℕ) : (unitGrid L u)[x]? =
    if x < nblocks (L.shape u).m 64 * nblocks (L.shape u).n 64 then
      some (u, x / nblocks (L.shape u).n 64, x % nblocks (L.shape u).n 64) else none := by
  unfold unitGrid
  rw [getElem?_flatMap_const _ (nblocks (L.shape u).n 64) (fun _ => by simp [nblocks]), List.length_range]
  by_cases hx : x < nblocks (L.shape u).m 64 * nblocks (L.shape u).n 64
  · rw [ite_eq_left (by simpa [nblocks] using hx), ite_eq_left hx]
    have hn : 0 < nblocks (L.shape u).n 64 := Nat.pos_of_ne_zero fun h => by rw [h, Nat.mul_zero] at hx; omega
    have hq : x / nblocks (L.shape u).n 64 < nblocks (L.shape u).m 64 := (Nat.div_lt_iff_lt_mul hn).2 hx
    have h1 : (List.range (((L.shape u).m + 64 - 1) / 64))[x / nblocks (L.shape u).n 64]? =
        some (x / nblocks (L.shape u).n 64) := List.getElem?_range hq
    have h2 : (List.range (((L.shape u).n + 64 - 1) / 64))[x % nblocks (L.shape u).n 64]? =
        some (x % nblocks (L.shape u).n 64) := List.getElem?_range (Nat.mod_lt x hn)
    rw [h1]
    simp only [Option.bind_some, List.getElem?_map, h2, Option.map_some]
  · rw [ite_eq_right (by simpa [nblocks] using hx), ite_eq_right hx]

/-- The index in `gridTiles 64 64 L` of unit `u`'s first tile. -/
def gstart (L : Pouw.PearlC.Layout) (u : ℕ) : ℕ := ((List.range u).map fun v => (unitGrid L v).length).sum

theorem gridTiles_getElem? {L : Pouw.PearlC.Layout} {u x : ℕ} (hu : u < L.N)
    (hx : x < nblocks (L.shape u).m 64 * nblocks (L.shape u).n 64) :
    (gridTiles 64 64 L)[gstart L u + x]? =
      some (u, x / nblocks (L.shape u).n 64, x % nblocks (L.shape u).n 64) := by
  rw [gridTiles_eq, gstart, getElem?_flatMap_range (unitGrid L) L.N u x hu (by rw [unitGrid_length]; exact hx),
    unitGrid_getElem?, ite_eq_left hx]

theorem gridTiles_inv {L : Pouw.PearlC.Layout} {g : ℕ} {b : ℕ × ℕ × ℕ} (h : (gridTiles 64 64 L)[g]? = some b) :
    ∃ u < L.N, ∃ x < nblocks (L.shape u).m 64 * nblocks (L.shape u).n 64, g = gstart L u + x ∧
      b = (u, x / nblocks (L.shape u).n 64, x % nblocks (L.shape u).n 64) := by
  rw [gridTiles_eq] at h
  obtain ⟨u, hu, x, hx, rfl, hb⟩ := getElem?_flatMap_range_inv (unitGrid L) L.N g b h
  rw [unitGrid_length] at hx
  rw [unitGrid_getElem?, ite_eq_left hx, Option.some.injEq] at hb
  exact ⟨u, hu, x, hx, rfl, hb.symm⟩

/-- The audit's tile `x` of unit `u`: its index is below `NT`, and its unit, rows and columns are the code's. -/
theorem audit_at {L : Pouw.PearlC.Layout} {u x : ℕ} (hu : u < L.N)
    (hx : x < nblocks (L.shape u).m 64 * nblocks (L.shape u).n 64) :
    gstart L u + x < (auditTiling 64 64 L).NT ∧ (auditTiling 64 64 L).unit (gstart L u + x) = u ∧
      (auditTiling 64 64 L).rows (gstart L u + x) =
        (block (L.shape u).m 64 (x / nblocks (L.shape u).n 64)).toFinset ∧
      (auditTiling 64 64 L).cols (gstart L u + x) =
        (block (L.shape u).n 64 (x % nblocks (L.shape u).n 64)).toFinset := by
  have h := gridTiles_getElem? hu hx
  have hd : (gridTiles 64 64 L).getD (gstart L u + x) (0, 0, 0) =
      (u, x / nblocks (L.shape u).n 64, x % nblocks (L.shape u).n 64) := by
    rw [List.getD_eq_getElem?_getD, h, Option.getD_some]
  refine ⟨(List.getElem?_eq_some_iff.1 h).1, ?_, ?_, ?_⟩
  · show ((gridTiles 64 64 L).getD _ (0, 0, 0)).1 = u
    rw [hd]
  · rw [block_toFinset]
    show Ico (((gridTiles 64 64 L).getD _ (0, 0, 0)).2.1 * 64) _ = _
    rw [hd]
  · rw [block_toFinset]
    show Ico (((gridTiles 64 64 L).getD _ (0, 0, 0)).2.2 * 64) _ = _
    rw [hd]

end Pouw.PearlPlus.Draw

namespace Pouw.PearlPlus.Bridge

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus Finset Draw

/-- **The declaration a unit's record makes** before the coin: its kept width, and the rows and columns it credits, as
`UnitRecord.credited`'s band rule filters a tile's (a real, unexcluded A row whose band holds no voluntary A row; a
column whose block holds no excluded or voluntary B row). -/
def declOf (r : UnitRecord) : Decl where
  kept := r.kKept
  rows := (Finset.range r.realRows).filter fun i =>
    (decide (i < r.realRows) && !r.excludedA.contains i &&
      !r.voluntaryA.any (· / UnitRecord.BAND == i / UnitRecord.BAND)) = true
  cols := (Finset.range r.matmul.n).filter fun j =>
    (!(r.excludedB ++ r.voluntaryB).any (· / UnitRecord.BAND == j / UnitRecord.BAND)) = true

/-- A row the band rule credits is neither excluded nor voluntary, so the check holds it to the per-row rules. -/
theorem banded_row {r : UnitRecord} {i : ℕ}
    (h : (decide (i < r.realRows) && !r.excludedA.contains i &&
      !r.voluntaryA.any (· / UnitRecord.BAND == i / UnitRecord.BAND)) = true) :
    (decide (i < r.realRows) && !r.excludedA.contains i && !r.voluntaryA.contains i) = true := by
  rw [Bool.and_eq_true] at h ⊢
  refine ⟨h.1, ?_⟩
  have h2 := h.2
  cases hv : r.voluntaryA.contains i with
  | false => rfl
  | true =>
    have ha : r.voluntaryA.any (· / UnitRecord.BAND == i / UnitRecord.BAND) = true :=
      List.any_eq_true.2 ⟨i, List.contains_iff_mem.1 hv, by simp⟩
    rw [ha] at h2
    cases h2

/-- **(c) A passed tile's declaration is consistent**: a tile of a pass `load` gives, which is audit tile `g` of `U`,
whose unit `τ` commits as `c` and declares by its record (`declOf`), whose openings are `c`, and that `checkTile` passes
on its weight's noise or finds only over its inert-work bound, has `declOK` at the deployed rules: its kept width is within `k`, and every credited row it
declares in the tile passes `RowOK` at `c`'s activations. -/
def TilePassGivesTheDecl : Prop :=
  ∀ {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (ρ : ℚ) (coins : PouwCheck.PearlPlus.EpochCoins)
    (man : Lean.Json) (p : Pass) (tid : ℕ) (o : TileOpening) (U : Pouw.PearlC.Workload) (g : ℕ)
    (τ : Pouw.PearlC.Transcript) (c : Pouw.PearlC.Commit) (dec : ℕ → Decl),
    PouwCheck.PearlPlus.load coins man = .ok p → TileOf p tid U g →
    τ ((auditTiling 64 64 U.layout).unit g) = some c → (∀ r ∈ p.records, dec r.matmul.index = declOf r) →
    OpensCommitted p tid o U ((auditTiling 64 64 U.layout).unit g) c →
    (checkTile p (Properties.Pouw.PearlPlus.noiseOf p tid) tid o = .pass ∨
      ∃ why, checkTile p (Properties.Pouw.PearlPlus.noiseOf p tid) tid o = .overCap why) →
      declOK (tilesH3 saltOf rootOf ρ : TileRules Q R S) U τ dec g

/-- **(c)** from what a passed tile establishes (`TileAcceptMeans`), what a loaded pass's records keep (`load_facts`)
and the per-row rules (`RowsAreTheRules`). -/
theorem tilePassGivesTheDecl (hT : Properties.Pouw.PearlPlus.TileAcceptMeans) (hR : RowsAreTheRules) :
    TilePassGivesTheDecl := by
  intro Q R S saltOf rootOf ρ coins man p tid o U g τ c dec hload hof hτ hdec hopen hpass
  obtain ⟨pos, rows, cols, htile, -, hunit, -, hk, -, -, hrows, -⟩ := hof
  obtain ⟨bound, hb⟩ := Assembly.passOrOver_bound hpass
  obtain ⟨pos', rows', cols', ops, ch, htile', hrest⟩ :=
    hT p (Properties.Pouw.PearlPlus.noiseOf p tid) tid o bound (by cases bound <;> exact hb)
  simp only [htile, Option.some.injEq, Prod.mk.injEq] at htile'
  obtain ⟨rfl, rfl, rfl⟩ := htile'
  try dsimp only at hrest
  obtain ⟨-, -, -, -, -, -, hform, -, hchk, -, -⟩ := hrest
  obtain ⟨hp, -⟩ := Draw.tile?_some htile
  have hrec : p.records[pos]! ∈ p.records := by rw [getElem!_pos p.records pos hp]; exact List.getElem_mem hp
  obtain ⟨-, -, hkept0, hkept⟩ := Draw.reason_none ((Properties.Pouw.PearlPlus.load_facts hload).1 _ hrec)
  have hd : dec ((auditTiling 64 64 U.layout).unit g) = declOf p.records[pos]! := by rw [hunit]; exact hdec _ hrec
  have hK := Assembly.record_domain htile hpass
  rw [← hk] at hK hkept
  refine ⟨?_, fun i hi hdi => ?_⟩
  · change (dec ((auditTiling 64 64 U.layout).unit g)).kept ≤ (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k
    rw [hd]
    exact hkept
  · change i ∈ (auditTiling 64 64 U.layout).rows g at hi
    change i ∈ (dec ((auditTiling 64 64 U.layout).unit g)).rows at hdi
    have him := Assembly.rows_lt hi
    rw [hrows, List.mem_toFinset] at hi
    obtain ⟨x, hx, rfl⟩ := List.getElem_of_mem hi
    rw [hd] at hdi
    simp only [declOf, Finset.mem_filter, Finset.mem_range] at hdi
    have hrp := Draw.rowsFailure_credited hchk hx (banded_row hdi.2)
    obtain ⟨wn', -, henc, -⟩ := Assembly.formed_ok hform
    obtain ⟨hoa, -, -, hact, -, -⟩ := hopen pos rows cols htile
    rw [hoa] at henc
    obtain ⟨hparse, s, β, hst⟩ := Draw.encode_rowA (by simp) henc x hx
    rw [getElem!_pos _ x (by simpa using hx), List.getElem_map, Draw.rowBytesOf_eq, ← hk] at hparse
    obtain ⟨hsz, hwords⟩ := Draw.parseRow_wordBytes hparse
    have hW : ∀ l < (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k,
        (ops.a.x[x]!)[l]! = c.act rows[x] l ∧ c.act rows[x] l < 2 ^ 32 ∧ (wordQ (c.act rows[x] l)).isSome := by
      intro l hl
      have h32 := hact rows[x] (List.getElem_mem hx) l hl
      obtain ⟨hxl, hfin⟩ := hwords l hl
      rw [Nat.mod_eq_of_lt h32] at hxl
      refine ⟨hxl, h32, ?_⟩
      rw [hxl, word_and_ff] at hfin
      have hne : (c.act rows[x] l >>> 23) % 256 ≠ 255 := by simpa using hfin
      rw [Pouw.PearlC.PeelExactProofs.wordQ_fields hne]
      rfl
    have hiff := @hR Q R S saltOf rootOf _ _ c.act rows[x] ops.a.x[x]! s _ _ β him (by omega) (by omega) (by omega) hsz
      hW hst
    have hA : actOf τ ((auditTiling 64 64 U.layout).unit g) = c.act := by unfold actOf; rw [hτ]
    show (tilesH3 saltOf rootOf ρ : TileRules Q R S).rowOK (U.layout.shape ((auditTiling 64 64 U.layout).unit g))
      (actOf τ ((auditTiling 64 64 U.layout).unit g)) rows[x]
    rw [hA]
    exact hiff.1 hrp

/-- The declaration the run's records make for unit `u`: its record's (`declOf`), or the empty one for a unit no record
calls. -/
def decOf (passes : List Pass) (u : ℕ) : Decl :=
  match (passes.flatMap (·.records)).find? (·.matmul.index == u) with
  | some r => declOf r
  | none => ⟨0, ∅, ∅⟩

/-- How the run's passes lie over the game's workload `U` and transcript `τ`: 64 × 64 tiles, each record's call a unit
of `U` that `τ` commits, with the record's shape and weight, and no call in two records. -/
structure RunLayout (passes : List Pass) (U : Pouw.PearlC.Workload) (τ : Pouw.PearlC.Transcript) : Prop where
  tile64 : ∀ p ∈ passes, p.scheme.tileM = 64 ∧ p.scheme.tileN = 64
  unit : ∀ p ∈ passes, ∀ r ∈ p.records, r.matmul.index < U.layout.N ∧
    (U.layout.shape r.matmul.index).m = r.matmul.m ∧ (U.layout.shape r.matmul.index).k = r.matmul.k ∧
    (U.layout.shape r.matmul.index).n = r.matmul.n ∧ U.layout.fixedId r.matmul.index = r.matmul.weight ∧
    (τ r.matmul.index).isSome
  index_inj : ∀ {i i' pos pos' : ℕ} {p p' : Pass} {r r' : UnitRecord}, passes[i]? = some p →
    p.records[pos]? = some r → passes[i']? = some p' → p'.records[pos']? = some r' →
    r.matmul.index = r'.matmul.index → i = i' ∧ pos = pos'

variable {passes : List Pass} {U : Pouw.PearlC.Workload} {τ : Pouw.PearlC.Transcript}

theorem decOf_cases (passes : List Pass) (u : ℕ) : decOf passes u = ⟨0, ∅, ∅⟩ ∨
    ∃ (i : ℕ) (p : Pass) (pos : ℕ) (r : UnitRecord), passes[i]? = some p ∧ p.records[pos]? = some r ∧
      r.matmul.index = u ∧ decOf passes u = declOf r := by
  unfold decOf
  split
  · rename_i r h
    right
    obtain ⟨p, hp, hr⟩ := List.mem_flatMap.1 (List.mem_of_find?_eq_some h)
    obtain ⟨i, hi⟩ := List.getElem?_of_mem hp
    obtain ⟨pos, hpos⟩ := List.getElem?_of_mem hr
    exact ⟨i, p, pos, r, hi, hpos, by simpa using List.find?_some h, rfl⟩
  · left
    rfl

theorem decOf_eq (hL : RunLayout passes U τ) {i pos : ℕ} {p : Pass} {r : UnitRecord} (hp : passes[i]? = some p)
    (hr : p.records[pos]? = some r) : decOf passes r.matmul.index = declOf r := by
  unfold decOf
  split
  · rename_i r' h
    obtain ⟨p', hp', hr'⟩ := List.mem_flatMap.1 (List.mem_of_find?_eq_some h)
    obtain ⟨i', hi'⟩ := List.getElem?_of_mem hp'
    obtain ⟨pos', hpos'⟩ := List.getElem?_of_mem hr'
    have hu : r'.matmul.index = r.matmul.index := by simpa using List.find?_some h
    obtain ⟨rfl, rfl⟩ := hL.index_inj hp hr hi' hpos' hu.symm
    rw [hi', Option.some.injEq] at hp
    subst hp
    rw [hpos', Option.some.injEq] at hr
    subst hr
    rfl
  · rename_i hn
    have := List.find?_eq_none.1 hn r (List.mem_flatMap.2 ⟨p, List.mem_of_getElem? hp, List.mem_of_getElem? hr⟩)
    simp at this

/-- The game tile of a run's tile: tile `x` of its call's unit in `gridTiles 64 64`. -/
def gOf (L : Pouw.PearlC.Layout) (c : Cell) : ℕ := gstart L c.r.matmul.index + c.x

theorem cell_facts (hL : RunLayout passes U τ) {c : Cell} (hc : c ∈ cells passes) :
    c.p ∈ passes ∧ c.r ∈ c.p.records ∧ c.x < nblocks c.r.matmul.m 64 * nblocks c.r.matmul.n 64 ∧
      c.rows = block c.r.matmul.m 64 (c.x / nblocks c.r.matmul.n 64) ∧
      c.cols = block c.r.matmul.n 64 (c.x % nblocks c.r.matmul.n 64) := by
  obtain ⟨hp, hr, ht⟩ := mem_cells.1 hc
  have hpm := List.mem_of_getElem? hp
  obtain ⟨hM, hN⟩ := hL.tile64 _ hpm
  rw [tiles_getElem?, hM, hN] at ht
  by_cases hx : c.x < nblocks c.r.matmul.m 64 * nblocks c.r.matmul.n 64
  · rw [ite_eq_left hx, Option.some.injEq, Prod.mk.injEq] at ht
    exact ⟨hpm, List.mem_of_getElem? hr, hx, ht.1.symm, ht.2.symm⟩
  · rw [ite_eq_right hx] at ht
    cases ht

theorem cell_n_pos (hL : RunLayout passes U τ) {c : Cell} (hc : c ∈ cells passes) : 0 < c.r.matmul.n := by
  obtain ⟨-, -, hx, -⟩ := cell_facts hL hc
  refine Nat.pos_of_ne_zero fun h => ?_
  rw [h] at hx
  simp [nblocks] at hx

/-- A run's tile is its game tile's: the audit's tile `gOf` has the record's unit, shape, weight, rows and columns. -/
theorem cell_audit (hL : RunLayout passes U τ) {c : Cell} (hc : c ∈ cells passes) :
    gOf U.layout c < (auditTiling 64 64 U.layout).NT ∧
      (auditTiling 64 64 U.layout).unit (gOf U.layout c) = c.r.matmul.index ∧
      (auditTiling 64 64 U.layout).rows (gOf U.layout c) = c.rows.toFinset ∧
      (auditTiling 64 64 U.layout).cols (gOf U.layout c) = c.cols.toFinset := by
  obtain ⟨hp, hr, hx, hrows, hcols⟩ := cell_facts hL hc
  obtain ⟨hu, hm, -, hn, -⟩ := hL.unit _ hp _ hr
  rw [← hm, ← hn] at hx
  obtain ⟨h1, h2, h3, h4⟩ := audit_at hu hx
  refine ⟨h1, h2, ?_, ?_⟩
  · rw [gOf, h3, hrows, hm, hn]
  · rw [gOf, h4, hcols, hn]

theorem cell_tileOf (hL : RunLayout passes U τ) {c : Cell} (hc : c ∈ cells passes) :
    TileOf c.p (c.p.offsets[c.pos]! + c.x) U (gOf U.layout c) := by
  obtain ⟨-, hrec, htl⟩ := mem_cells.1 hc
  obtain ⟨hpm, hrm, -⟩ := cell_facts hL hc
  obtain ⟨hp, hpr⟩ := List.getElem?_eq_some_iff.1 hrec
  obtain ⟨hx, hxr⟩ := List.getElem?_eq_some_iff.1 htl
  have hx' : c.x < (grid c.p.scheme c.p.records[c.pos]).length := by rw [hpr]; exact hx
  have htile := tile?_offsets c.p hp hx'
  have hg : (grid c.p.scheme c.p.records[c.pos])[c.x]! = (c.rows, c.cols) := by
    rw [getElem!_pos _ c.x hx']
    simp only [grid, hpr, hxr]
  rw [hg] at htile
  have hr! : c.p.records[c.pos]! = c.r := by rw [getElem!_pos c.p.records c.pos hp, hpr]
  obtain ⟨hNT, hunit, hrows, hcols⟩ := cell_audit hL hc
  obtain ⟨-, hm, hk, hn, hw, -⟩ := hL.unit _ hpm _ hrm
  refine ⟨c.pos, c.rows, c.cols, htile, hNT, ?_, ?_, ?_, ?_, ?_, hrows, hcols⟩ <;> rw [hr!] <;> try rw [hunit]
  · exact hm
  · exact hk
  · exact hn
  · exact hw

theorem cell_weight {Q R S : Type} {TR : TileRules Q R S} (hTR : ∀ L, TR.tiling L = auditTiling 64 64 L)
    (hL : RunLayout passes U τ) {c : Cell} (hc : c ∈ cells passes) :
    drawWeight TR (pearlCDrawRev1 devH100) U τ (decOf passes) (gOf U.layout c) = tileWeight c.r c.rows c.cols := by
  obtain ⟨hpi, hrec, -⟩ := mem_cells.1 hc
  obtain ⟨hpm, hrm, -, hrows, hcols⟩ := cell_facts hL hc
  obtain ⟨-, -, -, hn, -, hτ⟩ := hL.unit _ hpm _ hrm
  obtain ⟨-, hunit, hrows', hcols'⟩ := cell_audit hL hc
  unfold drawWeight
  rw [hTR, hunit, ite_eq_right (by rw [Option.isNone_iff_eq_none]; exact Option.isSome_iff_ne_none.1 hτ),
    decOf_eq hL hpi hrec]
  unfold pearlCDrawRev1
  rw [hunit, hrows', hcols', hn]
  have hr : (c.rows.toFinset ∩ (declOf c.r).rows).card = (c.r.credited c.rows c.cols).1.length := by
    refine card_inter_filter (by rw [hrows]; exact block_nodup _ _ _) c.r.realRows _ fun i _ hi => ?_
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hi
    exact hi.1.1
  have hcc : (c.cols.toFinset ∩ (declOf c.r).cols).card = (c.r.credited c.rows c.cols).2.length :=
    card_inter_filter (by rw [hcols]; exact block_nodup _ _ _) c.r.matmul.n _ fun j hj _ => by
      rw [hcols] at hj; exact block_lt hj
  rw [hr, hcc, tileWeight, declOf, ← wref_one_eq]
  push_cast
  ring

theorem lcm_eq {Q R S : Type} {TR : TileRules Q R S} (hTR : ∀ L, TR.tiling L = auditTiling 64 64 L)
    (hL : RunLayout passes U τ) (hsurj : ∀ g < (TR.tiling U.layout).NT,
      drawWeight TR (pearlCDrawRev1 devH100) U τ (decOf passes) g ≠ 0 → ∃ c ∈ cells passes, gOf U.layout c = g) :
    (range (TR.tiling U.layout).NT).lcm (fun g => (drawWeight TR (pearlCDrawRev1 devH100) U τ (decOf passes) g).den) =
      lcmOf ((cells passes).map Cell.weight) := by
  have hden : ∀ c ∈ cells passes, (drawWeight TR (pearlCDrawRev1 devH100) U τ (decOf passes) (gOf U.layout c)).den =
      c.weight.2.2 := fun c hc => by
    rw [cell_weight hTR hL hc, Cell.weight, ticketWeight_eq _ _ _ (cell_n_pos hL hc)]
  refine Nat.dvd_antisymm (Finset.lcm_dvd fun g hg => ?_) (lcmOf_dvd fun e he => ?_)
  · rw [Finset.mem_range] at hg
    by_cases hv : drawWeight TR (pearlCDrawRev1 devH100) U τ (decOf passes) g = 0
    · rw [hv, Rat.den_zero]
      exact Nat.one_dvd _
    · obtain ⟨c, hc, rfl⟩ := hsurj g hg hv
      rw [hden c hc]
      exact dvd_lcmOf (List.mem_map_of_mem hc)
  · obtain ⟨c, hc, rfl⟩ := List.mem_map.1 he
    rw [← hden c hc]
    refine Finset.dvd_lcm (Finset.mem_range.2 ?_)
    rw [hTR]
    exact (cell_audit hL hc).1

theorem cell_of_weight {Q R S : Type} {TR : TileRules Q R S} (hTR : ∀ L, TR.tiling L = auditTiling 64 64 L)
    (hL : RunLayout passes U τ) {g : ℕ} (hg : g < (TR.tiling U.layout).NT)
    (hv : drawWeight TR (pearlCDrawRev1 devH100) U τ (decOf passes) g ≠ 0) :
    ∃ c ∈ cells passes, gOf U.layout c = g := by
  rw [hTR] at hg
  obtain ⟨u, hu, x, hx, rfl, -⟩ := gridTiles_inv (List.getElem?_eq_getElem hg)
  obtain ⟨-, hunit, -, -⟩ := audit_at hu hx
  unfold drawWeight at hv
  rw [hTR, hunit] at hv
  rcases decOf_cases passes u with h | ⟨i, p, pos, r, hp, hr, rfl, -⟩
  · rw [h] at hv
    unfold pearlCDrawRev1 at hv
    simp at hv
  · obtain ⟨hM, hN⟩ := hL.tile64 _ (List.mem_of_getElem? hp)
    obtain ⟨-, hm, -, hn, -⟩ := hL.unit _ (List.mem_of_getElem? hp) _ (List.mem_of_getElem? hr)
    rw [hm, hn] at hx
    refine ⟨⟨i, p, pos, r, x, block r.matmul.m 64 (x / nblocks r.matmul.n 64),
      block r.matmul.n 64 (x % nblocks r.matmul.n 64)⟩, mem_cells.2 ⟨hp, hr, ?_⟩, rfl⟩
    rw [tiles_getElem?, hM, hN, ite_eq_left hx]

theorem gOf_inj (hL : RunLayout passes U τ) {c c' : Cell} (hc : c ∈ cells passes) (hc' : c' ∈ cells passes)
    (h : gOf U.layout c = gOf U.layout c') : c = c' := by
  have hu := (cell_audit hL hc).2.1
  rw [h, (cell_audit hL hc').2.1] at hu
  obtain ⟨hp, hr, ht⟩ := mem_cells.1 hc
  obtain ⟨hp', hr', ht'⟩ := mem_cells.1 hc'
  obtain ⟨hi, hpos⟩ := hL.index_inj hp hr hp' hr' hu.symm
  obtain ⟨i, p, pos, r, x, rows, cols⟩ := c
  obtain ⟨i', p', pos', r', x', rows', cols'⟩ := c'
  simp only at hi hpos hp hr ht hp' hr' ht' h hu
  subst hi hpos
  rw [hp', Option.some.injEq] at hp
  subst hp
  rw [hr', Option.some.injEq] at hr
  subst hr
  simp only [gOf, Nat.add_left_cancel_iff] at h
  subst h
  rw [ht', Option.some.injEq, Prod.mk.injEq] at ht
  obtain ⟨rfl, rfl⟩ := ht
  rfl

/-- **The code's tickets are the draw's.** Over a run laid out on the game's workload (`RunLayout`), with each unit's
declaration read from its record (`decOf`), each of `Run.tickets`' entries is a tile of its pass on a game tile, whose
`drawTickets` under `pearlCDrawRev1 devH100` are the entry's tickets; distinct entries are on distinct game tiles, and
every game tile with a ticket has an entry. -/
def TicketsAreTheDraw : Prop :=
  ∀ {Q R S : Type} (TR : TileRules Q R S) (passes : List Pass) (U : Pouw.PearlC.Workload) (τ : Pouw.PearlC.Transcript),
    (∀ L, TR.tiling L = auditTiling 64 64 L) → RunLayout passes U τ →
    ∃ φ : ℕ → ℕ,
      (∀ j id t, (tickets passes)[j]? = some (id, t) →
        TileOf passes[id.1]! id.2 U (φ j) ∧ t = drawTickets TR (pearlCDrawRev1 devH100) U τ (decOf passes) (φ j)) ∧
      (∀ j j', j < (tickets passes).length → j' < (tickets passes).length → φ j = φ j' → j = j') ∧
      ∀ g, drawTickets TR (pearlCDrawRev1 devH100) U τ (decOf passes) g ≠ 0 →
        ∃ j < (tickets passes).length, φ j = g

theorem ticketsAreTheDraw : TicketsAreTheDraw := by
  intro Q R S TR passes U τ hTR hL
  have hsurj : ∀ g < (TR.tiling U.layout).NT,
      drawWeight TR (pearlCDrawRev1 devH100) U τ (decOf passes) g ≠ 0 → ∃ c ∈ cells passes, gOf U.layout c = g :=
    fun g hg hv => cell_of_weight hTR hL hg hv
  have hlen : (tickets passes).length = (cells passes).length := by rw [tickets_eq, List.length_map, List.length_map]
  refine ⟨fun j => if h : j < (cells passes).length then gOf U.layout (cells passes)[j] else 0, ?_, ?_, ?_⟩
  · intro j id t h
    rw [tickets_eq, List.getElem?_map, List.getElem?_map] at h
    cases hj : (cells passes)[j]? with
    | none => rw [hj] at h; cases h
    | some c =>
      rw [hj] at h
      simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      obtain ⟨hjl, hjc⟩ := List.getElem?_eq_some_iff.1 hj
      have hc : c ∈ cells passes := List.mem_of_getElem? hj
      dsimp only
      rw [dite_eq_left hjl, hjc]
      obtain ⟨hpi, -, -⟩ := mem_cells.1 hc
      have hp! : passes[c.i]! = c.p := by
        obtain ⟨hil, hip⟩ := List.getElem?_eq_some_iff.1 hpi
        rw [getElem!_pos passes c.i hil, hip]
      refine ⟨by rw [Cell.weight, hp!]; exact cell_tileOf hL hc, ?_⟩
      unfold drawTickets ticketsOf
      rw [lcm_eq hTR hL hsurj, hTR, ite_eq_left (cell_audit hL hc).1, cell_weight hTR hL hc, Cell.weight,
        ticketWeight_eq _ _ _ (cell_n_pos hL hc)]
  · intro j j' hj hj' h
    rw [hlen] at hj hj'
    dsimp only at h
    rw [dite_eq_left hj, dite_eq_left hj'] at h
    exact (List.Nodup.getElem_inj_iff (cells_nodup passes)).1
      (gOf_inj hL (List.getElem_mem hj) (List.getElem_mem hj') h)
  · intro g hg
    have hNT : g < (TR.tiling U.layout).NT := by
      by_contra hn
      unfold drawTickets ticketsOf at hg
      rw [ite_eq_right hn] at hg
      exact hg rfl
    have hv : drawWeight TR (pearlCDrawRev1 devH100) U τ (decOf passes) g ≠ 0 := fun h0 => by
      unfold drawTickets ticketsOf at hg
      rw [h0] at hg
      simp at hg
    obtain ⟨c, hc, rfl⟩ := hsurj g hNT hv
    obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hc
    exact ⟨j, by rw [hlen]; exact hj, by dsimp only; rw [dite_eq_left hj]⟩

/-! ## Acceptance -/

/-- What the game reads of the plan's tile `(pi, tid)` on audit tile `g`: the verdict opens it, it is `g` (`TileOf`),
its pass's salt is the game's, its weight's root is the semantics', each weight row of its columns passes `P`'s rules,
and its openings are its unit's commitment. -/
structure DrawnOn {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (P : Pouw.PearlC.Protocol Q R S) (s : S)
    (pl : Plan) (opener : ℕ → ℕ → Except String TileOpening) (U : Pouw.PearlC.Workload) (τ : Pouw.PearlC.Transcript)
    (pi tid g : ℕ) : Prop where
  todo : (pi, tid) ∈ pl.todo
  tile : TileOf pl.passes[pi]! tid U g
  salt : bytesNat pl.passes[pi]!.salt = saltOf s
  root : bytesNat ((pl.passes[pi]!.weightRoots.lookup (U.layout.fixedId ((auditTiling 64 64 U.layout).unit g))).getD
    ByteArray.empty) = rootOf (U.layout.fixedId ((auditTiling 64 64 U.layout).unit g))
  weights : ∀ j ∈ (auditTiling 64 64 U.layout).cols g,
    P.fixedOK (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) (U.fixed (U.layout.fixedId
      ((auditTiling 64 64 U.layout).unit g))) j
  opens : ∀ o c, opener pi tid = .ok o → τ ((auditTiling 64 64 U.layout).unit g) = some c →
    OpensCommitted pl.passes[pi]! tid o U ((auditTiling 64 64 U.layout).unit g) c

/-- **(d) Acceptance is the tolerant audit's**: at rules `P`, `TR` that read the deployed tiles, rules and words, when
the check accepts a run laid out as the game's workload, every sample `c` of the game whose tiles the verdict opened
(`DrawnOn`) is accepted by `AuditAcceptsTol` over `goodDeclTiles ∪ blankTiles`, with the tiles over the inert-work bound
(`overCapTiles`) tolerated, under the records' declarations; or SHA-512 collides on two different messages of an opened
tile, one hashed in digesting the words the check forms and one in digesting the commitment's words. -/
def AcceptGivesTheAudit : Prop :=
  ∀ {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (ρ : ℚ) (P : Pouw.PearlC.Protocol Q R S)
    (TR : TileRules Q R S) (H : Q → R) (s : S) (i : Input) (opener : ℕ → ℕ → Except String TileOpening) (pl : Plan)
    (U : Pouw.PearlC.Workload) (τ : Pouw.PearlC.Transcript) {w : ℕ → ℕ} {t : ℕ}
    (c : Fin t → Ticket w (TR.tiling U.layout).NT),
    TR.tiling = auditTiling 64 64 → TR.rowOK = (tilesH3 saltOf rootOf ρ : TileRules Q R S).rowOK →
    P.ticket = (protocolH3 saltOf rootOf ρ : Pouw.PearlC.Protocol Q R S).ticket →
    P.useful = (protocolH3 saltOf rootOf ρ : Pouw.PearlC.Protocol Q R S).useful →
    P.fixedOK = (protocolH3 saltOf rootOf ρ : Pouw.PearlC.Protocol Q R S).fixedOK →
    (check i opener).accept = true → plan i = .ok pl → RunLayout pl.passes U τ →
    (∀ j, ∃ pi tid, DrawnOn saltOf rootOf P s pl opener U τ pi tid (c j).1) →
    AuditAcceptsTol (goodDeclTiles P TR U H s τ (decOf pl.passes) ∪ blankTiles TR U τ) (overCapTiles TR U H s τ) c ∨
      ∃ pi tid g o cm, DrawnOn saltOf rootOf P s pl opener U τ pi tid g ∧ opener pi tid = .ok o ∧
        τ ((auditTiling 64 64 U.layout).unit g) = some cm ∧
        ∃ x ∈ leafHashed pl.passes[pi]! tid o, ∃ y ∈ commitHashed pl.passes[pi]! tid cm, x ≠ y ∧
          Flock.Sha512.hash x = Flock.Sha512.hash y

theorem acceptGivesTheAudit (hA : Properties.Pouw.PearlPlus.AcceptMeans)
    (hP : Properties.Pouw.PearlPlus.PlanMeans) (hD : TilePassGivesTheDecl) (hW : TilePassGivesTheWords) :
    AcceptGivesTheAudit := by
  intro Q R S saltOf rootOf ρ P TR H s i opener pl U τ w t c hT hrow hPt hPu hPw hacc hplan hL hc
  obtain ⟨pl', hpl', -, -, -, -, hopen⟩ := hA i opener hacc
  rw [hplan, Except.ok.injEq] at hpl'
  subst hpl'
  have hload := (hP i pl hplan).2.1
  by_cases hcol : ∃ pi tid g o cm, DrawnOn saltOf rootOf P s pl opener U τ pi tid g ∧ opener pi tid = .ok o ∧
      τ ((auditTiling 64 64 U.layout).unit g) = some cm ∧
      ∃ x ∈ leafHashed pl.passes[pi]! tid o, ∃ y ∈ commitHashed pl.passes[pi]! tid cm, x ≠ y ∧
        Flock.Sha512.hash x = Flock.Sha512.hash y
  · exact Or.inr hcol
  left
  intro j
  obtain ⟨pi, tid, hd⟩ := hc j
  generalize ((c j).1 : ℕ) = g at hd ⊢
  obtain ⟨p, o, hp, ho, hpass'⟩ := hopen pi tid hd.todo
  have hpass := hpass'.imp id fun h => h.2
  have hp! : pl.passes[pi]! = p := by
    obtain ⟨hl, he⟩ := List.getElem?_eq_some_iff.1 hp
    rw [getElem!_pos pl.passes pi hl, he]
  obtain ⟨man, -, hman⟩ := Assembly.mapM_ok_mem hload p (List.mem_of_getElem? hp)
  have htile := hd.tile
  rw [hp!] at htile
  obtain ⟨-, -, -, -, hNT, -⟩ := id htile
  simp only [goodDeclTiles, goodTiles, blankTiles, overCapTiles, Finset.mem_union, Finset.mem_filter,
    Finset.mem_range, hT]
  cases hτ : τ ((auditTiling 64 64 U.layout).unit g) with
  | none => exact Or.inl (Or.inr ⟨hNT, rfl⟩)
  | some cm =>
    have hop := hd.opens o cm ho hτ
    rw [hp!] at hop
    have hdec : ∀ r ∈ p.records, decOf pl.passes r.matmul.index = declOf r := fun r hr => by
      obtain ⟨pos, hpos⟩ := List.getElem?_of_mem hr
      exact decOf_eq hL hp hpos
    have hdecl := @hD Q R S saltOf rootOf ρ i.coins man p tid o U g τ cm (decOf pl.passes) hman htile hτ hdec hop
      hpass
    have hw := hW saltOf rootOf ρ H s i.coins man p tid o U g cm hman htile (by rw [← hp!]; exact hd.salt)
      (by rw [← hp!]; exact hd.root) (by rw [← hPw]; exact hd.weights) hop hpass
    rcases hw with hw | ⟨x, hx, y, hy, hne, hxy⟩
    · by_cases hcap : TR.capOK H s U g cm.act
      · refine Or.inl (Or.inl ⟨⟨hNT, cm, rfl, ?_⟩, ?_⟩)
        · unfold TileGood
          rw [hT, hrow, hPt, hPu]
          exact ⟨hcap, hd.weights, hw⟩
        · unfold declOK
          rw [hT, hrow]
          exact hdecl
      · exact Or.inr ⟨hNT, cm, rfl, hcap⟩
    · exact absurd ⟨pi, tid, g, o, cm, hd, ho, hτ, x, by rw [hp!]; exact hx, y, by rw [hp!]; exact hy, hne, hxy⟩
        hcol

end Pouw.PearlPlus.Bridge
