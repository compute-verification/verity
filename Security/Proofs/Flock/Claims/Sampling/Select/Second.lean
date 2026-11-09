module

public import Proofs.Flock.Claims.Sampling.Select.Design
public import Proofs.Flock.Claims.Sampling.Count
public import Proofs.Flock.Soundness.ExecDraw
public import Properties.Spot.Second
import all Spot.Draw
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String
import Mathlib.Data.List.GetD
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Select.Bounds
import Proofs.Flock.Claims.Sampling.Select.Take

/-!
# The second stage misses proof blocks no more often than the cells' escapes (`SecondMisses`)

* `flockLaw_second`: `secondLaw`'s Flock law is the work law at `k = 1`, its strata the cells in order, template by
  template and in each by sampled replay block (`cellPairs`), cell `(t, i)` exactly
  `[off_t + i·n_t, off_t + (i+1)·n_t)` sampling `k_t` (`cellOf`).
* `selectSecond_ok`: an accepted second stage passed `secondProblem`, drew its units from the coins under that sampling
  design, and proves each of them.
* `positions_second`: the cells, in order, are the units `[0, |sample|·Σ n_t)`; `sizes_second`: cell `(t, i)` is `n_t`
  units sampling `k_t`.
* `strataPa_cells`: `strataPa` of the cells is the product over the templates of their cells' (`strataPa_replicate`).
* `secondMisses`: the second stage is accepted and proves no proof block of `X` with probability at most
  `∏_t ∏_i escCell n_t k_t |X ∩ [off_t + i·n_t, off_t + (i+1)·n_t)|`.
-/

@[expose] public section

open private Spot.Draw.stratum? Spot.Draw.natRanges from Spot.Draw

open Spot

namespace Properties.Spot.SecurityProofs

open Lean (Json)
open Claims.Sampling Finset

/-! ## `secondLaw` parses to the cells -/

theorem getD_secondLaw_law (sample ns ks : Array ℕ) :
    getD (Select.secondLaw sample ns ks) "law" = Json.str "work" := (rfl)

theorem getD_secondLaw_k (sample ns ks : Array ℕ) :
    getD (Select.secondLaw sample ns ks) "k" = Lean.toJson (1 : ℕ) := (rfl)

theorem getD_secondLaw_strata (sample ns ks : Array ℕ) :
    getD (Select.secondLaw sample ns ks) "strata" = Json.arr ((List.range ns.size).flatMap fun t =>
      (List.range sample.size).map fun i =>
        Select.cellObj (cellOff sample.size ns t) (ns.getD t 0) (ks.getD t 0) i (sample.getD i 0) t).toArray := (rfl)

theorem getD_cellObj_template (o n k i r t : ℕ) :
    getD (Select.cellObj o n k i r t) "template" = Json.str s!"{r}/{t}" := (rfl)

theorem getD_cellObj_units (o n k i r t : ℕ) :
    getD (Select.cellObj o n k i r t) "units" = Json.arr #[natsJson #[o + i * n, o + (i + 1) * n]] := (rfl)

theorem getD_cellObj_k (o n k i r t : ℕ) : getD (Select.cellObj o n k i r t) "k" = Lean.toJson k := (rfl)

theorem getD_cellObj_work (o n k i r t : ℕ) : getD (Select.cellObj o n k i r t) "work" = Lean.toJson (1 : ℕ) := (rfl)

theorem getD_cellObj_floor (o n k i r t : ℕ) : getD (Select.cellObj o n k i r t) "floor" = Lean.toJson k := (rfl)

theorem int_toJson (a : ℕ) : int? (Lean.toJson a) = some (a : Int) := (rfl)

theorem ranges_pair {a b : ℕ} {rs : Array (Int × Int)} (h : Draw.ranges? (Json.arr #[natsJson #[a, b]]) = some rs) :
    rs = #[((a : Int), (b : Int))] := by
  unfold Draw.ranges? at h
  simp only [natsJson, Array.isEmpty, List.map_toArray, List.map_cons, List.map_nil, Array.mapM_eq_mapM_toList] at h
  by_cases hab : a < b <;> simp [hab, int_toJson] at h
  exact h.symm

private theorem natRanges_pair {a b : Int} {rs : Array (ℕ × ℕ)} (h : Spot.Draw.natRanges #[(a, b)] = some rs) :
    rs = #[(a.toNat, b.toNat)] := by
  unfold Spot.Draw.natRanges at h
  by_cases ha : 0 ≤ a <;> simp [ha, Array.mapM_eq_mapM_toList] at h
  exact h.symm

/-- A cell object of `secondLaw` that parses is `[o + i·n, o + (i+1)·n)` sampling `k`. -/
private theorem stratum_cell {o n k i r t : ℕ} {ws : Flock.Draw.WorkStratum}
    (h : Spot.Draw.stratum? true (Select.cellObj o n k i r t) = some ws) :
    (ws.units, ws.k) = (#[(o + i * n, o + (i + 1) * n)], k) := by
  unfold Spot.Draw.stratum? at h
  simp only [getD_cellObj_template, getD_cellObj_units, getD_cellObj_k, getD_cellObj_work, getD_cellObj_floor,
    nat_toJson, ↓reduceIte] at h
  split at h
  · simp at h
  · simp only [Option.bind_eq_some_iff, pure, bind, Option.some.injEq] at h
    obtain ⟨rs0, hr, rs, hn, _, rfl, _, rfl, _, rfl, rfl⟩ := h
    obtain rfl := ranges_pair hr
    obtain rfl := natRanges_pair hn
    simp only [Int.toNat_natCast]

theorem list_mapM_forall₂ {α β : Type} (f : α → Option β) :
    ∀ (xs : List α) (ys : List β), xs.mapM f = some ys → List.Forall₂ (fun a b => f a = some b) xs ys
  | [], ys, h => by
    simp only [List.mapM_nil, pure, Option.some.injEq] at h; subst h; exact .nil
  | x :: xs, ys, h => by
    simp only [List.mapM_cons, Option.bind_eq_some_iff, bind, pure, Option.some.injEq] at h
    obtain ⟨y, hy, ys', hys, rfl⟩ := h
    exact .cons hy (list_mapM_forall₂ f xs ys' hys)

theorem mapM_forall₂ {α β : Type} (f : α → Option β) (xs : Array α) (ys : Array β) (h : xs.mapM f = some ys) :
    List.Forall₂ (fun a b => f a = some b) xs.toList ys.toList := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | none => simp [hl] at h
  | some ys' =>
    rw [hl] at h
    obtain rfl : ys = ys'.toArray := (Option.some.inj h).symm
    exact list_mapM_forall₂ f _ _ hl

/-- Pointwise images of related lists: `ys` relates to `zs.map φ` by `f`, and `g` of what `f` makes of `φ z` is `ψ z`. -/
theorem forall₂_map_eq {α β γ δ : Type} {f : α → Option β} {g : β → δ} {φ : γ → α} {ψ : γ → δ}
    (hg : ∀ z b, f (φ z) = some b → g b = ψ z) :
    ∀ {zs : List γ} {ys : List β}, List.Forall₂ (fun a b => f a = some b) (zs.map φ) ys → ys.map g = zs.map ψ
  | [], _, h => by rw [List.map_nil] at h; cases h; rfl
  | z :: zs, _, h => by
    rw [List.map_cons] at h
    cases h with
    | cons hb hrest => rw [List.map_cons, List.map_cons, hg _ _ hb, forall₂_map_eq hg hrest]

/-- The cells' (template, sampled replay block) pairs over a replay sample of `q` and `T` templates, in `secondLaw`'s
order: template by template. -/
def cellPairs (q T : ℕ) : List (ℕ × ℕ) := (List.range T).flatMap fun t => (List.range q).map fun i => (t, i)

/-- Cell `(t, i)`'s units and count over a replay sample of `q`: `[off_t + i·n_t, off_t + (i+1)·n_t)` sampling `k_t`. -/
def cellOf (q : ℕ) (ns ks : Array ℕ) (z : ℕ × ℕ) : Array (ℕ × ℕ) × ℕ :=
  (#[(cellOff q ns z.1 + z.2 * ns.getD z.1 0, cellOff q ns z.1 + (z.2 + 1) * ns.getD z.1 0)], ks.getD z.1 0)

theorem strata_second (sample ns ks : Array ℕ) :
    ((List.range ns.size).flatMap fun t => (List.range sample.size).map fun i =>
        Select.cellObj (cellOff sample.size ns t) (ns.getD t 0) (ks.getD t 0) i (sample.getD i 0) t) =
      (cellPairs sample.size ns.size).map fun z =>
        Select.cellObj (cellOff sample.size ns z.1) (ns.getD z.1 0) (ks.getD z.1 0) z.2 (sample.getD z.2 0) z.1 := by
  simp only [cellPairs, List.map_flatMap, List.map_map, Function.comp_def]

/-- **`secondLaw`'s Flock law**: the work law at `k = 1`, its strata the cells in order (`cellOf` over `cellPairs`). -/
theorem flockLaw_second (sample ns ks : Array ℕ) (m : ℕ) (parsed : Flock.Draw.Law)
    (h : Draw.flockLaw (Select.secondLaw sample ns ks) m = .ok parsed) :
    ∃ ws : Array Flock.Draw.WorkStratum, parsed = .work 1 ws ∧
      ws.toList.map (fun s => (s.units, s.k)) = (cellPairs sample.size ns.size).map (cellOf sample.size ns ks) := by
  unfold Draw.flockLaw at h
  rw [getD_secondLaw_law] at h
  dsimp only at h
  split at h
  all_goals try (rename_i hs; exact absurd hs (by decide))
  · split at h
    · exact absurd h (throw_bind_ne _ _ _)
    rw [getD_secondLaw_k, getD_secondLaw_strata, nat_toJson] at h
    dsimp only at h
    split at h
    · exact absurd h (throw_bind_ne _ _ _)
    split at h
    · rename_i strata hss
      have hk : ("work" == "work") = true := by decide
      simp only [hk, ↓reduceIte, pure, Except.pure, Except.ok.injEq] at h hss
      subst h
      refine ⟨_, rfl, ?_⟩
      have hF := mapM_forall₂ _ _ _ hss
      rw [List.toList_toArray, strata_second] at hF
      exact forall₂_map_eq (fun _ _ hb => stratum_cell hb) hF
    · exact absurd h (throw_ne _ _)

theorem getD_draw_units (sample ns ks : Array ℕ) (m : ℕ) (units : Array ℕ) :
    getD (Json.mkObj ((entries (Select.secondLaw sample ns ks)).toList ++
      [("population", Lean.toJson m), ("units", natsJson units)])) "units" = natsJson units := (rfl)

/-- A sample object whose `units` are `units` proves each of them. -/
theorem proved_units (draw : Json) (units : Array ℕ) (hd : getD draw "units" = natsJson units)
    (u : ℕ) (hu : u ∈ units) : (u : Int) ∈ Draw.proved draw := by
  unfold Draw.proved
  rw [FlockSoundness.Audit.DrawExec.mem_qsort]
  refine Array.mem_append_left _ ?_
  dsimp only
  rw [hd]
  simp only [natsJson, Array.mem_filterMap, Array.mem_map]
  exact ⟨_, ⟨u, hu, rfl⟩, int_toJson u⟩

/-- **`Select.selectSecond`'s steps**: an accepted second stage passed `secondProblem`, drew its units from the coins
under `secondLaw`'s Flock law over the `|sample|·Σ n_t` proof blocks, and proves each of them. -/
theorem selectSecond_ok {sample ns ks : Array ℕ} {src : ByteArray} {draw : Json} {read : ℕ}
    (h : Select.selectSecond sample ns ks src = .ok (draw, read)) :
    Select.secondProblem sample ns ks = none ∧
      ∃ parsed units, Draw.flockLaw (Select.secondLaw sample ns ks) (sample.size * ns.toList.sum) = .ok parsed ∧
        Draw.drawUnits parsed (sample.size * ns.toList.sum) src = .ok (units, read) ∧
          ∀ u ∈ units, (u : Int) ∈ Draw.proved draw := by
  unfold Select.selectSecond at h
  cases hq : Select.secondProblem sample ns ks with
  | some e => rw [hq] at h; dsimp only at h; exact absurd h (throw_bind_ne _ _ _)
  | none =>
    rw [hq] at h
    dsimp only at h
    refine ⟨rfl, ?_⟩
    obtain ⟨⟨draw', read'⟩, hd, h⟩ := bind_eq_ok.1 h
    split at h
    · exact absurd h (throw_bind_ne _ _ _)
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    unfold Draw.drawFirstRead at hd
    obtain ⟨parsed, hp, hd⟩ := bind_eq_ok.1 hd
    obtain ⟨⟨units, read''⟩, hu, hd⟩ := bind_eq_ok.1 hd
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at hd
    obtain ⟨rfl, rfl⟩ := hd
    exact ⟨parsed, units, hp, hu, proved_units _ _ (getD_draw_units _ _ _ _ _)⟩

/-- A second stage `secondProblem` passes has a nonempty, ascending replay sample. -/
theorem secondProblem_sample {sample ns ks : Array ℕ} (h : Select.secondProblem sample ns ks = none) :
    sample.isEmpty = false ∧ (sample.toList.zip sample.toList.tail).all (fun (a, b) => decide (a < b)) = true := by
  unfold Select.secondProblem at h
  split at h
  · cases h
  · rename_i hs
    simpa [Bool.or_eq_true, not_or] using hs

/-! ## The cells' offsets -/

theorem getD_toList (a : Array ℕ) (t : ℕ) : a.getD t 0 = a.toList.getD t 0 := by
  simp [List.getD_eq_getElem?_getD]

theorem sum_take_succ (l : List ℕ) (t : ℕ) : (l.take (t + 1)).sum = (l.take t).sum + l.getD t 0 := by
  induction l generalizing t with
  | nil => simp
  | cons a l ih =>
    cases t with
    | zero => simp
    | succ t => simp [List.take_succ_cons, ih, Nat.add_assoc]

theorem cellOff_succ (q : ℕ) (ns : Array ℕ) (t : ℕ) : cellOff q ns (t + 1) = cellOff q ns t + q * ns.getD t 0 := by
  rw [cellOff, cellOff, sum_take_succ, getD_toList, Nat.mul_add]

theorem cellOff_mono (q : ℕ) (ns : Array ℕ) {t T : ℕ} (h : t ≤ T) : cellOff q ns t ≤ cellOff q ns T := by
  induction T, h using Nat.le_induction with
  | base => exact le_rfl
  | succ T _ ih => rw [cellOff_succ]; omega

theorem cellOff_size (q : ℕ) (ns : Array ℕ) : cellOff q ns ns.size = q * ns.toList.sum := by
  rw [cellOff, ← Array.length_toList, List.take_length]

/-- Cell `(t, i)` of a template `t` and `i < q` lies in the interior's `q · Σ n_t` proof blocks. -/
theorem cell_le_size (q : ℕ) (ns : Array ℕ) {t i : ℕ} (ht : t < ns.size) (hi : i < q) :
    cellOff q ns t + (i + 1) * ns.getD t 0 ≤ q * ns.toList.sum := by
  rw [← cellOff_size]
  refine le_trans ?_ (cellOff_mono q ns (show t + 1 ≤ ns.size by omega))
  rw [cellOff_succ]
  exact Nat.add_le_add_left (Nat.mul_le_mul_right _ hi) _

/-! ## The cells tile the interior, and their escapes multiply -/

theorem flatMap_range' (o n : ℕ) : ∀ q : ℕ,
    (List.range q).flatMap (fun i => List.range' (o + i * n) n) = List.range' o (q * n)
  | 0 => by simp
  | q + 1 => by
    rw [List.range_succ, List.flatMap_append, flatMap_range' o n q, List.flatMap_singleton, List.range'_append_1,
      Nat.succ_mul]

theorem positions_cells (q : ℕ) (ns : Array ℕ) : ∀ T : ℕ,
    ((List.range T).flatMap fun t => (List.range q).flatMap fun i =>
      List.range' (cellOff q ns t + i * ns.getD t 0) (ns.getD t 0)) = List.range' 0 (cellOff q ns T)
  | 0 => by simp [cellOff]
  | T + 1 => by
    rw [List.range_succ, List.flatMap_append, positions_cells q ns T, List.flatMap_singleton, flatMap_range',
      cellOff_succ, ← List.range'_append_1, Nat.zero_add]

theorem expand_cell (o n i : ℕ) : expand [(o + i * n, o + (i + 1) * n)] = List.range' (o + i * n) n := by
  simp only [expand, List.flatMap_cons, List.flatMap_nil, List.append_nil]
  congr 1
  rw [Nat.succ_mul]
  omega

/-- **`secondLaw`'s strata, in order, are the units `[0, |sample|·Σ n_t)`.** -/
theorem positions_second {ws : Array Flock.Draw.WorkStratum} {q : ℕ} {ns ks : Array ℕ}
    (hmap : ws.toList.map (fun s => (s.units, s.k)) = (cellPairs q ns.size).map (cellOf q ns ks)) :
    (ws.map (·.toStratum)).toList.flatMap (fun s => expand s.units.toList) = List.range (q * ns.toList.sum) := by
  have h : (ws.map (·.toStratum)).toList.flatMap (fun s => expand s.units.toList) =
      (ws.toList.map (fun s => (s.units, s.k))).flatMap (fun p => expand p.1.toList) := by
    rw [Array.toList_map, List.flatMap_map, List.flatMap_map]
  rw [h, hmap, List.flatMap_map, cellPairs, List.flatMap_assoc, ← cellOff_size,
    List.range_eq_range'.trans (positions_cells q ns ns.size).symm]
  congr 1
  funext t
  rw [List.flatMap_map]
  congr 1
  funext i
  simp only [cellOf]
  exact expand_cell _ _ _

/-- **`secondLaw`'s strata are the cells' sizes and counts**: template by template, `|sample|` cells of `n_t` sampling
`k_t`. -/
theorem sizes_second {ws : Array Flock.Draw.WorkStratum} {q : ℕ} {ns ks : Array ℕ}
    (hmap : ws.toList.map (fun s => (s.units, s.k)) = (cellPairs q ns.size).map (cellOf q ns ks)) :
    (ws.map (·.toStratum)).toList.map (fun s => (s.size, s.k)) =
      (List.range ns.size).flatMap fun t => List.replicate q (ns.getD t 0, ks.getD t 0) := by
  have h : (ws.map (·.toStratum)).toList.map (fun s => (s.size, s.k)) =
      (ws.toList.map (fun s => (s.units, s.k))).map
        (fun p => (p.1.foldl (fun acc (r : ℕ × ℕ) => acc + (r.2 - r.1)) 0, p.2)) := by
    simp only [Array.toList_map, List.map_map, Function.comp_def]
    rfl
  rw [h, hmap, List.map_map, cellPairs, List.map_flatMap]
  congr 1
  funext t
  rw [List.map_map, show List.replicate q (ns.getD t 0, ks.getD t 0) =
    (List.range q).map fun _ => (ns.getD t 0, ks.getD t 0) by rw [List.map_const', List.length_range]]
  congr 1
  funext i
  simp [cellOf, Nat.succ_mul]
  omega

theorem strataPa_append (W : Finset ℕ) : ∀ (l₁ l₂ : List (ℕ × ℕ)) (o : ℕ),
    strataPa (l₁ ++ l₂) o W = strataPa l₁ o W * strataPa l₂ (o + (l₁.map Prod.fst).sum) W
  | [], l₂, o => by simp [strataPa]
  | (n, k) :: l₁, l₂, o => by
    rw [List.cons_append, strataPa, strataPa, strataPa_append W l₁ l₂ (o + n), mul_assoc]
    simp [Nat.add_assoc]

theorem strataPa_replicate (n k : ℕ) (W : Finset ℕ) : ∀ (S o : ℕ),
    strataPa (List.replicate S (n, k)) o W =
      ∏ i ∈ range S, (((n - (window (o + i * n) n W).card).choose k : ℝ) / (n.choose k : ℝ))
  | 0, o => by simp [strataPa]
  | S + 1, o => by
    rw [List.replicate_succ, strataPa, strataPa_replicate n k W S (o + n), prod_range_succ', mul_comm]
    congr 1
    · refine prod_congr rfl fun i _ => ?_
      rw [show o + n + i * n = o + (i + 1) * n by ring]
    · simp

theorem fst_sum_cells (q : ℕ) (ns ks : Array ℕ) : ∀ T : ℕ,
    (((List.range T).flatMap fun t => List.replicate q (ns.getD t 0, ks.getD t 0)).map Prod.fst).sum = cellOff q ns T
  | 0 => by simp [cellOff]
  | T + 1 => by
    rw [List.range_succ, List.flatMap_append, List.map_append, List.sum_append, fst_sum_cells q ns ks T,
      List.flatMap_singleton, cellOff_succ, List.map_replicate, List.sum_replicate_nat]

theorem strataPa_cells (q : ℕ) (ns ks : Array ℕ) (W : Finset ℕ) : ∀ T : ℕ,
    strataPa ((List.range T).flatMap fun t => List.replicate q (ns.getD t 0, ks.getD t 0)) 0 W =
      ∏ t ∈ range T, strataPa (List.replicate q (ns.getD t 0, ks.getD t 0)) (cellOff q ns t) W
  | 0 => by simp [strataPa]
  | T + 1 => by
    rw [List.range_succ, List.flatMap_append, strataPa_append, strataPa_cells q ns ks W T, prod_range_succ,
      List.flatMap_singleton, Nat.zero_add, fst_sum_cells]

/-- A uniform `k`-subset's escape from `n` units holding `w` marked ones is at most the claims' `escCell`, equal to it
where `k ≤ n`. -/
theorem choose_div_le_escCell (n k w : ℕ) : ((n - w).choose k : ℝ) / (n.choose k : ℝ) ≤ escCell n k w := by
  unfold escCell
  rcases le_or_gt k n with h | h
  · rw [min_eq_left h]
  · rw [Nat.choose_eq_zero_of_lt h, Nat.cast_zero, div_zero]; positivity

/-- **The second stage misses `X` no more often than the cells' escapes**
(`Properties.Spot.SecondMisses`). -/
theorem secondMisses : SecondMisses := by
  intro sample ns ks L X
  have hev : ∀ src : Fin L → Fin 256, ((Select.selectSecond sample ns ks (bytesOf src)).isOk = true ∧
        ∀ j ∈ X, (j : Int) ∉ secondOf (Select.selectSecond sample ns ks (bytesOf src))) →
      ∃ draw read, Select.selectSecond sample ns ks (bytesOf src) = .ok (draw, read) ∧
        ∀ j ∈ X, (j : Int) ∉ Draw.proved draw := by
    rintro src ⟨hok, hm⟩
    cases h : Select.selectSecond sample ns ks (bytesOf src) with
    | error e => rw [h] at hok; exact absurd hok (by simp [Except.isOk, Except.toBool])
    | ok x => rw [h] at hm; exact ⟨x.1, x.2, rfl, hm⟩
  have hR0 : 0 ≤ ∏ t ∈ range ns.size, ∏ i ∈ range sample.size, escCell (ns.getD t 0) (ks.getD t 0)
      (X.filter fun j => cellOff sample.size ns t + i * ns.getD t 0 ≤ j ∧
        j < cellOff sample.size ns t + (i + 1) * ns.getD t 0).card :=
    prod_nonneg fun _ _ => prod_nonneg fun _ _ => by unfold escCell; positivity
  by_cases hex : ∃ src : Fin L → Fin 256, (Select.selectSecond sample ns ks (bytesOf src)).isOk = true ∧
      ∀ j ∈ X, (j : Int) ∉ secondOf (Select.selectSecond sample ns ks (bytesOf src))
  swap
  · rw [Claims.Pr_congr fun src => iff_false_intro fun hs => hex ⟨src, hs⟩, Claims.Pr_false]
    exact hR0
  obtain ⟨src0, hs0⟩ := hex
  obtain ⟨draw0, read0, h0, -⟩ := hev src0 hs0
  obtain ⟨-, parsed, -, hp, -, -⟩ := selectSecond_ok h0
  obtain ⟨ws, rfl, hmap⟩ := flockLaw_second _ _ _ _ _ hp
  refine le_trans (Claims.Pr_mono fun src hs => ?_)
    (le_trans (strata_bound (.work 1 ws) (ws.map (·.toStratum)) (Or.inr ⟨1, ws, rfl, rfl⟩)
      (sample.size * ns.toList.sum) (X.filter (· < sample.size * ns.toList.sum)) L) ?_)
  · obtain ⟨draw, read, h, hmiss⟩ := hev src hs
    obtain ⟨-, parsed', units, hp', hu, hpr⟩ := selectSecond_ok h
    rw [hp] at hp'
    obtain rfl := Except.ok.inj hp'
    refine ⟨units, read, hu, fun v hv hvu => ?_⟩
    rw [positions_second hmap] at hvu
    have hv' := mem_filter.1 hv
    rw [List.getD_eq_getElem _ _ (by simpa using hv'.2), List.getElem_range] at hvu
    exact hmiss v hv'.1 (hpr v hvu)
  · rw [sizes_second hmap, strataPa_cells]
    refine prod_le_prod₀ (fun t _ => ?_) fun t ht => ?_
    · rw [strataPa_replicate]
      exact prod_nonneg fun _ _ => by positivity
    rw [strataPa_replicate]
    refine prod_le_prod₀ (fun _ _ => by positivity) fun i hi => ?_
    have hle := cell_le_size sample.size ns (mem_range.1 ht) (mem_range.1 hi)
    rw [Count.card_window, filter_filter]
    refine le_of_eq_of_le ?_ (choose_div_le_escCell _ _ _)
    congr 5
    refine filter_congr fun v _ => ?_
    rw [Nat.succ_mul] at hle ⊢
    omega

end Properties.Spot.SecurityProofs
