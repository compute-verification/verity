module

public import Proofs.Properties.Spot.SecondSample
public import Proofs.Flock.Soundness.ExecStrataOf
import Spot.Draw
import Spot.Json
import Batteries.Tactic.OpenPrivate
import Proofs.Core.QSort
import Proofs.Flock.Claims.Sampling.Select.Take
import Proofs.Flock.Claims.Sampling.Select.Steps

/-!
# The second stage's sampling design is well formed wherever `secondProblem` passes

`Select.selectSecond` reads its sampling design (`secondLaw`) twice before it reads a coin: `Draw.flockLaw` parses it
(`Draw.strataProblem`, `Flock.Draw.workProblem`), and `Draw.checkDraw` holds the sample object's strata to
`Draw.strataMessage`. Both walk the strata in loops. This file runs those loops on `secondLaw`'s cells: where
`secondProblem` passes, the cells have distinct templates, each is one nonempty range sampling `1 ≤ k_t ≤ n_t` of it,
and their ranges follow each other from `0` to `|sample| · Σ_t n_t`, so neither finds a fault. The second stage then
refuses an honest run only where its coins sample no units, with probability at most `|sample| · Σ_t k_t · 2⁻⁶⁴`.

* `forIn_yield_foldl`, `forIn_chain`, `forIn_all_yield`: a loop that never stops early is a fold, a loop over ranges
  that follow each other carries the last range's end, and a loop whose steps keep the state returns it.
* `cellW`, `CellJson`: a list of cells, each one range `[lo, hi)` of unit work and floor `k`, and their objects.
* `workProblem_cells`, `strataProblem_cells`, `flockLaw_cells`, `strataMessage_cells`: such cells, with distinct
  templates and ranges that follow each other from `0` to the population, pass each check.
* `chain_cells`, `templates_nodup`: `secondLaw`'s cells are such a list.
* `flockLaw_second_ok`, `strataMessage_second`, `secondFormed`: the sampling design parses to its cells' strata, and
  its strata are well formed.
* `selectSecond_refused_le`: on a uniform stream of at least `secondRead` bytes, `selectSecond` refuses with
  probability at most `|sample| · Σ_t k_t · 2⁻⁶⁴`.
-/

@[expose] public section

set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open private Spot.Draw.stratum? Spot.Draw.natRanges Spot.Draw.WORK_STRATUM_KEYS Spot.Draw.STRATUM_KEYS
  Spot.Draw.sumSizes from Spot.Draw

open Spot

namespace Properties.Spot.SecurityProofs

open Lean (Json)
open FlockSoundness FlockSoundness.Game Finset
open FlockSoundness.Audit.StrataExec (Chain chain_facts)
open scoped ENNReal

/-! ## Loops that run to their end -/

/-- A loop whose every step goes on with `g` of the state is the fold of `g`. -/
theorem forIn_yield_foldl {α β : Type} (g : β → α → β) (f : α → β → Id (ForInStep β)) :
    ∀ (l : List α) (b : β),
      (∀ pre x suf, l = pre ++ x :: suf → f x (pre.foldl g b) = ForInStep.yield (g (pre.foldl g b) x)) →
      forIn (m := Id) l b f = pure (l.foldl g b)
  | [], _, _ => rfl
  | x :: xs, b, h => by
    rw [List.forIn_cons, show f x b = ForInStep.yield (g b x) from h [] x xs rfl]
    exact forIn_yield_foldl g f xs (g b x) fun pre y suf hl => h (x :: pre) y suf (by rw [hl]; rfl)

/-- A loop over ranges that follow each other from `a` to `c`, each step taking a state at its range's start to
the state at its end, takes the state at `a` to the state at `c`. -/
theorem forIn_chain {α σ : Type} (ι : ℕ × ℕ → α) (κ : ℕ → σ)
    (f : α → Option (Option String) × σ → Id (ForInStep (Option (Option String) × σ)))
    (hf : ∀ x : ℕ × ℕ, f (ι x) (none, κ x.1) = ForInStep.yield (none, κ x.2)) :
    ∀ (l : List (ℕ × ℕ)) (a c : ℕ), Chain a c l →
      forIn (m := Id) (l.map ι) (none, κ a) f = pure (none, κ c)
  | [], a, c, h => by
    obtain rfl : a = c := h
    rfl
  | r :: rs, a, c, h => by
    obtain ⟨h1, h2⟩ := h
    subst h1
    rw [List.map_cons, List.forIn_cons, hf r]
    exact forIn_chain ι κ f hf rs r.2 c h2

/-- A loop whose every step keeps the state returns it. -/
theorem forIn_all_yield {α β : Type} (f : α → β → Id (ForInStep β)) (b : β) :
    ∀ l : List α, (∀ i ∈ l, f i b = ForInStep.yield b) → forIn (m := Id) l b f = pure b
  | [], _ => rfl
  | i :: is, h => by
    rw [List.forIn_cons, h i List.mem_cons_self]
    exact forIn_all_yield f b is fun j hj => h j (List.mem_cons_of_mem _ hj)

/-! ## Chains -/

/-- Nonempty ranges that follow each other start in ascending order. -/
theorem chain_getElem_lt : ∀ {l : List (ℕ × ℕ)} {a c : ℕ}, Chain a c l → (∀ r ∈ l, r.1 < r.2) →
    ∀ (i j : ℕ) (_ : i < j) (hj : j < l.length), (l[i]'(by omega)).1 < l[j].1
  | [], _, _, _, _, _, _, _, hj => nomatch hj
  | x :: xs, a, c, h, hlt, i, j, hij, hj => by
    obtain ⟨h1, h2⟩ := h
    have hlt' : ∀ r ∈ xs, r.1 < r.2 := fun r hr => hlt r (List.mem_cons_of_mem _ hr)
    cases j with
    | zero => omega
    | succ j =>
      have hj' : j < xs.length := by simpa using hj
      cases i with
      | zero =>
        have h3 := ((chain_facts _ _ _ h2 hlt').2.1 xs[j] (List.getElem_mem hj')).1
        have h4 := hlt x List.mem_cons_self
        simp only [List.getElem_cons_zero, List.getElem_cons_succ]
        omega
      | succ i =>
        simp only [List.getElem_cons_succ]
        exact chain_getElem_lt h2 hlt' i j (by omega) hj'

/-- Nonempty ranges that follow each other from `a` to `c` have `c - a` units in all. -/
theorem chain_sum : ∀ {l : List (ℕ × ℕ)} {a c : ℕ}, Chain a c l → (∀ r ∈ l, r.1 < r.2) →
    a + (l.map fun r => r.2 - r.1).sum = c
  | [], a, c, h, _ => by
    obtain rfl : a = c := h
    simp
  | x :: xs, a, c, h, hlt => by
    obtain ⟨h1, h2⟩ := h
    have := chain_sum h2 fun r hr => hlt r (List.mem_cons_of_mem _ hr)
    have := hlt x List.mem_cons_self
    simp only [List.map_cons, List.sum_cons]
    omega

theorem le_sum_of_mem : ∀ {l : List ℕ} {x : ℕ}, x ∈ l → x ≤ l.sum
  | y :: ys, x, h => by
    rcases List.mem_cons.1 h with rfl | h
    · simp
    · have := le_sum_of_mem h
      simp only [List.sum_cons]
      omega

/-- A chain of nonempty ranges is its own sort by the ranges' starts. -/
theorem chain_sorted {l : List (ℕ × ℕ)} {a c : ℕ} (h : Chain a c l) (hlt : ∀ r ∈ l, r.1 < r.2)
    (i : ℕ) (hi : i < l.length) (j : ℕ) (hj : j < l.length) : decide (l[i].1 < l[j].1) = decide (i < j) := by
  rcases Nat.lt_trichotomy i j with hij | rfl | hij
  · have := chain_getElem_lt h hlt i j hij hj
    simp [this, hij]
  · simp
  · have := chain_getElem_lt h hlt j i hij hi
    have h1 : ¬ l[i].1 < l[j].1 := by omega
    have h2 : ¬ i < j := by omega
    simp [h1, h2]

/-! ## Cells: one range each, of unit work -/

section Cells

variable {γ : Type}

/-- Cell `z`'s work stratum: the one range `[lo z, hi z)` of unit work, sampling its floor `kk z`. -/
def cellW (tpl : γ → String) (lo hi kk : γ → ℕ) (z : γ) : Flock.Draw.WorkStratum :=
  { template := tpl z, units := #[(lo z, hi z)], k := kk z, work := 1, floor := kk z }

theorem cellW_size (tpl : γ → String) (lo hi kk : γ → ℕ) (z : γ) :
    (cellW tpl lo hi kk z).size = hi z - lo z := by
  simp [Flock.Draw.Stratum.size, cellW]

theorem foldl_cellW_units (tpl : γ → String) (lo hi kk : γ → ℕ) :
    ∀ (zs : List γ) (init : Array (ℕ × ℕ)),
    (zs.map (cellW tpl lo hi kk)).foldl (fun x1 x2 => x1 ++ x2.units) init =
      init ++ (zs.map fun z => (lo z, hi z)).toArray
  | [], init => by simp
  | z :: zs, init => by
    rw [List.map_cons, List.foldl_cons, foldl_cellW_units tpl lo hi kk zs]
    simp [cellW]

theorem foldl_cellW_work (tpl : γ → String) (lo hi kk : γ → ℕ) : ∀ (zs : List γ) (init : ℕ),
    (zs.map (cellW tpl lo hi kk)).foldl (fun acc s => acc + s.work * s.size) init =
      init + ((zs.map fun z => (lo z, hi z)).map fun r => r.2 - r.1).sum
  | [], init => by simp
  | z :: zs, init => by
    rw [List.map_cons, List.foldl_cons, foldl_cellW_work tpl lo hi kk zs, cellW_size]
    simp [cellW, Nat.add_assoc]

/-- The cells' total work is the population their ranges tile. -/
theorem totalWork_cells (tpl : γ → String) (lo hi kk : γ → ℕ) (zs : List γ) (m : ℕ)
    (hz : ∀ z ∈ zs, lo z < hi z) (hch : Chain 0 m (zs.map fun z => (lo z, hi z))) :
    Flock.Draw.totalWork (zs.map (cellW tpl lo hi kk)).toArray = m := by
  unfold Flock.Draw.totalWork
  rw [List.foldl_toArray' (h := rfl), foldl_cellW_work, Nat.zero_add]
  have := chain_sum hch (by
    intro r hr
    obtain ⟨z, hz', rfl⟩ := List.mem_map.1 hr
    exact hz z hz')
  omega

theorem cells_pos (lo hi : γ → ℕ) (zs : List γ) (m : ℕ) (hne : zs ≠ []) (hz : ∀ z ∈ zs, lo z < hi z)
    (hch : Chain 0 m (zs.map fun z => (lo z, hi z))) : 1 ≤ m := by
  obtain ⟨z, zs', rfl⟩ := List.exists_cons_of_ne_nil hne
  have h1 := chain_sum hch (by
    intro r hr
    obtain ⟨z, hz', rfl⟩ := List.mem_map.1 hr
    exact hz z hz')
  have h2 := hz z List.mem_cons_self
  simp only [List.map_cons, List.sum_cons] at h1
  omega

/-- **`Flock.Draw.workProblem` passes cells** with distinct templates, each a nonempty range sampling `1 ≤ k ≤ n` of
its `n` units, whose ranges follow each other from `0` to the population, at `k = 1`. -/
theorem workProblem_cells (tpl : γ → String) (lo hi kk : γ → ℕ) (zs : List γ) (m : ℕ)
    (hne : zs ≠ []) (hz : ∀ z ∈ zs, lo z < hi z ∧ 1 ≤ kk z ∧ kk z ≤ hi z - lo z)
    (hnd : (zs.map tpl).Nodup) (hch : Chain 0 m (zs.map fun z => (lo z, hi z))) :
    Flock.Draw.workProblem 1 (zs.map (cellW tpl lo hi kk)).toArray m = none := by
  have hlt : ∀ r ∈ zs.map fun z => (lo z, hi z), r.1 < r.2 := by
    intro r hr
    obtain ⟨z, hz', rfl⟩ := List.mem_map.1 hr
    exact (hz z hz').1
  have hsum := chain_sum hch hlt
  have htot := totalWork_cells tpl lo hi kk zs m (fun z h => (hz z h).1) hch
  have hsz : ∀ z ∈ zs, hi z - lo z ≤ m := by
    intro z hz'
    have := le_sum_of_mem (List.mem_map_of_mem (f := fun r : ℕ × ℕ => r.2 - r.1)
      (List.mem_map_of_mem (f := fun z => (lo z, hi z)) hz'))
    simp only at this
    omega
  have hm : 1 ≤ m := by
    obtain ⟨z, zs', rfl⟩ := List.exists_cons_of_ne_nil hne
    have h1 := hsz z List.mem_cons_self
    have h2 := (hz z List.mem_cons_self).1
    omega
  have hemp : (zs.map (cellW tpl lo hi kk)).toArray.isEmpty = false := by
    simpa using hne
  have h0 : ((zs.map (cellW tpl lo hi kk)).toArray.isEmpty || decide (1 < 1) || decide (1 > m)) = false := by
    rw [hemp, Bool.false_or, Bool.or_eq_false_iff]
    exact ⟨by decide, decide_eq_false (by omega)⟩
  have ht0 : (m == 0) = false := by
    rw [beq_eq_false_iff_ne]
    omega
  unfold Flock.Draw.workProblem
  rw [h0, htot]
  simp only [Bool.false_eq_true, ↓reduceIte, ht0, Std.Legacy.Range.forIn_eq_forIn_range']
  rw [forIn_all_yield]
  · have hfold : Array.foldl (fun x1 x2 => x1 ++ x2.units) #[] (List.map (cellW tpl lo hi kk) zs).toArray =
        (zs.map fun z => (lo z, hi z)).toArray := by
      rw [List.foldl_toArray' (h := rfl), foldl_cellW_units]
      simp
    have hsort : ((zs.map fun z => (lo z, hi z)).toArray.qsort fun a b => decide (a.fst < b.fst)) =
        (zs.map fun z => (lo z, hi z)).toArray :=
      Verity.QSort.qsort_sorted _ _ fun i hi' j hj => by
        simp only [List.getElem_toArray]
        exact chain_sorted hch hlt i (by simpa using hi') j (by simpa using hj)
    have hloop := fun f hf => forIn_chain id id f hf (zs.map fun z => (lo z, hi z)) 0 m hch
    simp only [List.map_id, id_eq] at hloop
    rw [hfold, hsort, List.forIn_toArray, hloop]
    · simp only [pure_bind, beq_self_eq_true, ↓reduceIte]
      rfl
    · intro x
      rw [ite_eq_right (by simp)]
      rfl
  · intro i hi'
    have hi2 : i < zs.length := by
      simp [Std.Legacy.Range.size] at hi'
      omega
    have hget : (List.map (cellW tpl lo hi kk) zs).toArray[i]! = cellW tpl lo hi kk zs[i] := by
      simp [hi2]
    simp only [hget]
    have hany : (((List.map (cellW tpl lo hi kk) zs).toArray.extract 0 i).any fun x =>
        x.template == (cellW tpl lo hi kk zs[i]).template) = false := by
      rw [Array.any_eq_false]
      intro k hk
      have hk2 : k < i := by
        simp at hk
        omega
      have hk3 : k < zs.length := by omega
      have hne' : tpl zs[k] ≠ tpl zs[i] := by
        have := List.pairwise_iff_getElem.1 hnd k i (by simpa using hk3) (by simpa using hi2) hk2
        simpa using this
      simp [cellW, hne']
    have hzi := hz zs[i] (List.getElem_mem hi2)
    have hfl : ¬ (cellW tpl lo hi kk zs[i]).floor < 1 := by
      simp only [cellW]
      omega
    have hk : ((cellW tpl lo hi kk zs[i]).k != Flock.Draw.workK 1 (cellW tpl lo hi kk zs[i]).work
        (cellW tpl lo hi kk zs[i]).size m (cellW tpl lo hi kk zs[i]).floor) = false := by
      rw [cellW_size]
      simp only [cellW]
      rw [workK_unit hzi.2.1 hzi.2.2 (hsz _ (List.getElem_mem hi2))]
      simp
    simp only [hany, hfl, hk, Bool.false_eq_true, ↓reduceIte]
    rfl

/-! ## Cells as JSON -/

/-- Cell `z`'s object `φ z` reads as its work stratum: a work stratum's keys, its template a string, its units the one
range, unit work, and its count and floor `kk z`. -/
structure CellJson (φ : γ → Json) (tpl : γ → String) (lo hi kk : γ → ℕ) (z : γ) : Prop where
  keys : keys? (φ z) = some Spot.Draw.WORK_STRATUM_KEYS
  template : getD (φ z) "template" = .str (tpl z)
  units : Spot.Draw.ranges? (getD (φ z) "units") = some #[((lo z : Int), (hi z : Int))]
  k : int? (getD (φ z) "k") = some (kk z : Int)
  work : int? (getD (φ z) "work") = some 1
  floor : int? (getD (φ z) "floor") = some (kk z : Int)

theorem nat_of_int {j : Json} {n : ℕ} (h : int? j = some (n : Int)) : nat? j = some n := by
  unfold nat?
  rw [h]
  simp

theorem natRanges_pair_ok (a b : ℕ) : Spot.Draw.natRanges #[((a : Int), (b : Int))] = some #[(a, b)] := by
  unfold Spot.Draw.natRanges
  simp

/-- A cell's object parses to its work stratum. -/
theorem stratum_ok {φ : γ → Json} {tpl : γ → String} {lo hi kk : γ → ℕ} {z : γ}
    (h : CellJson φ tpl lo hi kk z) :
    Spot.Draw.stratum? true (φ z) = some (cellW tpl lo hi kk z) := by
  unfold Spot.Draw.stratum?
  simp [h.keys, h.template, h.units, nat_of_int h.k, nat_of_int (n := 1) h.work, nat_of_int h.floor,
    natRanges_pair_ok, cellW]

theorem list_mapM_map_some {α β : Type} (f : α → Option β) (φ : γ → α) (ψ : γ → β) :
    ∀ zs : List γ, (∀ z ∈ zs, f (φ z) = some (ψ z)) → (zs.map φ).mapM f = some (zs.map ψ)
  | [], _ => rfl
  | z :: zs, h => by
    rw [List.map_cons, List.mapM_cons, h z List.mem_cons_self,
      list_mapM_map_some f φ ψ zs fun z hz => h z (List.mem_cons_of_mem _ hz)]
    rfl

theorem mapM_cells {φ : γ → Json} {tpl : γ → String} {lo hi kk : γ → ℕ} {zs : List γ}
    (hc : ∀ z ∈ zs, CellJson φ tpl lo hi kk z) :
    (zs.map φ).toArray.mapM (Spot.Draw.stratum? true) = some (zs.map (cellW tpl lo hi kk)).toArray := by
  rw [List.mapM_toArray, list_mapM_map_some _ φ (cellW tpl lo hi kk) zs fun z hz => stratum_ok (hc z hz)]
  rfl

/-- **`Draw.strataProblem` passes a work sampling design at `k = 1` over such cells.** -/
theorem strataProblem_cells (φ : γ → Json) (tpl : γ → String) (lo hi kk : γ → ℕ) (zs : List γ) (m : ℕ)
    (law : Json) (hw : jeq (getD law "law") (Json.str "work") = true)
    (hss : getD law "strata" = Json.arr (zs.map φ).toArray) (hk : int? (getD law "k") = some 1)
    (hc : ∀ z ∈ zs, CellJson φ tpl lo hi kk z)
    (hne : zs ≠ []) (hz : ∀ z ∈ zs, lo z < hi z ∧ 1 ≤ kk z ∧ kk z ≤ hi z - lo z)
    (hnd : (zs.map tpl).Nodup) (hch : Chain 0 m (zs.map fun z => (lo z, hi z))) :
    Spot.Draw.strataProblem law (m : Int) = false := by
  have hm := cells_pos lo hi zs m hne (fun z h => (hz z h).1) hch
  have hwp := workProblem_cells tpl lo hi kk zs m hne hz hnd hch
  have hany : ((zs.map (cellW tpl lo hi kk)).toArray.any fun s =>
      (List.range (s.units.size - 1)).any fun i => decide (s.units[i + 1]!.fst ≤ s.units[i]!.snd)) = false := by
    rw [Array.any_eq_false]
    intro i hi'
    simp [cellW]
  have h0 : ((zs.map φ).toArray.isEmpty || decide ((1 : Int) < 1) || decide ((m : Int) < 1)) = false := by
    have : (zs.map φ).toArray.isEmpty = false := by simpa using hne
    rw [this, Bool.false_or, Bool.or_eq_false_iff]
    exact ⟨by decide, decide_eq_false (by omega)⟩
  unfold Spot.Draw.strataProblem
  simp only [hss, hk, hw, h0, mapM_cells hc, hany]
  have h1 : Int.toNat 1 = 1 := rfl
  simp only [Bool.false_eq_true, ↓reduceIte, h1, Int.toNat_natCast, hwp]
  rfl

/-- **`Draw.flockLaw` reads a work sampling design at `k = 1` over such cells** as the one on their strata. -/
theorem flockLaw_cells (φ : γ → Json) (tpl : γ → String) (lo hi kk : γ → ℕ) (zs : List γ) (m : ℕ)
    (law : Json) (hl : getD law "law" = Json.str "work") (hw : jeq (getD law "law") (Json.str "work") = true)
    (hkeys : keys? law = some (sortStrings #["law", "k", "strata"]))
    (hss : getD law "strata" = Json.arr (zs.map φ).toArray) (hk : int? (getD law "k") = some 1)
    (hc : ∀ z ∈ zs, CellJson φ tpl lo hi kk z)
    (hne : zs ≠ []) (hz : ∀ z ∈ zs, lo z < hi z ∧ 1 ≤ kk z ∧ kk z ≤ hi z - lo z)
    (hnd : (zs.map tpl).Nodup) (hch : Chain 0 m (zs.map fun z => (lo z, hi z))) :
    Spot.Draw.flockLaw law m = .ok (.work 1 (zs.map (cellW tpl lo hi kk)).toArray) := by
  have hsp := strataProblem_cells φ tpl lo hi kk zs m law hw hss hk hc hne hz hnd hch
  unfold Spot.Draw.flockLaw
  rw [hl]
  dsimp only
  split
  all_goals try (rename_i hs; exact absurd hs (by decide))
  have hb : ("work" == "work") = true := by decide
  simp only [hkeys, nat_of_int (n := 1) hk, hss, hsp, hb, mapM_cells hc]
  simp only [bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]
  rfl

/-- `forIn_yield_foldl` over a mapped list. -/
theorem forIn_map_yield_foldl {α β δ : Type} (φ : δ → α) (g : β → δ → β)
    (f : α → β → Id (ForInStep β)) :
    ∀ (l : List δ) (b : β),
      (∀ pre x suf, l = pre ++ x :: suf → f (φ x) (pre.foldl g b) = ForInStep.yield (g (pre.foldl g b) x)) →
      forIn (m := Id) (l.map φ) b f = pure (l.foldl g b)
  | [], _, _ => rfl
  | x :: xs, b, h => by
    rw [List.map_cons, List.forIn_cons, show f (φ x) b = ForInStep.yield (g b x) from h [] x xs rfl]
    exact forIn_map_yield_foldl φ g f xs (g b x) fun pre y suf hl => h (x :: pre) y suf (by rw [hl]; rfl)

/-- `strataMessage`'s state after a cell: its template seen, its range kept. -/
def stepG (tpl : γ → String) (lo hi : γ → ℕ) (st : Option (Option String) × Array String × Array (Int × Int))
    (z : γ) : Option (Option String) × Array String × Array (Int × Int) :=
  (none, st.2.1.push (tpl z), st.2.2 ++ #[((lo z : Int), (hi z : Int))])

theorem foldl_stepG (tpl : γ → String) (lo hi : γ → ℕ) : ∀ (zs : List γ) (seen : Array String)
    (ranges : Array (Int × Int)),
    zs.foldl (stepG tpl lo hi) (none, seen, ranges) =
      (none, seen ++ (zs.map tpl).toArray, ranges ++ (zs.map fun z => ((lo z : Int), (hi z : Int))).toArray)
  | [], seen, ranges => by simp
  | z :: zs, seen, ranges => by
    rw [List.foldl_cons, stepG, foldl_stepG tpl lo hi zs]
    simp

theorem sumSizes_pair (a b : Int) : Spot.Draw.sumSizes #[(a, b)] = b - a := by
  simp [Spot.Draw.sumSizes]

theorem foldl_total {φ : γ → Json} {tpl : γ → String} {lo hi kk : γ → ℕ} : ∀ (zs : List γ) (init : Int),
    (∀ z ∈ zs, CellJson φ tpl lo hi kk z) → (∀ z ∈ zs, lo z < hi z) →
    (zs.map φ).foldl (fun a s => a + (int? (getD s "work")).getD 0 *
        Spot.Draw.sumSizes ((Spot.Draw.ranges? (getD s "units")).getD #[])) init =
      init + ((((zs.map fun z => (lo z, hi z)).map fun r => r.2 - r.1).sum : ℕ) : Int)
  | [], init, _, _ => by simp
  | z :: zs, init, hc, hz => by
    have h := hc z List.mem_cons_self
    have hlt := hz z List.mem_cons_self
    rw [List.map_cons, List.foldl_cons, foldl_total zs _ (fun z hz' => hc z (List.mem_cons_of_mem _ hz'))
      (fun z hz' => hz z (List.mem_cons_of_mem _ hz')), h.work, h.units]
    simp only [Option.getD_some, sumSizes_pair, List.map_cons, List.sum_cons]
    omega

theorem qsort_chainI {l : List (ℕ × ℕ)} {a c : ℕ} (h : Chain a c l) (hlt : ∀ r ∈ l, r.1 < r.2) :
    ((l.map fun r : ℕ × ℕ => ((r.1 : Int), (r.2 : Int))).toArray.qsort fun a b =>
        decide (a.fst < b.fst) || a.fst == b.fst && decide (a.snd < b.snd)) =
      (l.map fun r : ℕ × ℕ => ((r.1 : Int), (r.2 : Int))).toArray :=
  Verity.QSort.qsort_sorted _ _ fun i hi' j hj => by
    have hi2 : i < l.length := by simpa using hi'
    have hj2 : j < l.length := by simpa using hj
    simp only [List.getElem_toArray, List.getElem_map]
    rcases Nat.lt_trichotomy i j with hij | rfl | hij
    · have := chain_getElem_lt h hlt i j hij hj2
      have h1 : ((l[i].1 : ℕ) : Int) < (l[j].1 : ℕ) := by omega
      simp [h1, hij]
    · simp
    · have := chain_getElem_lt h hlt j i hij hi2
      have h1 : ¬ ((l[i].1 : ℕ) : Int) < (l[j].1 : ℕ) := by omega
      have h2 : ¬ ((l[i].1 : ℕ) : Int) = (l[j].1 : ℕ) := by omega
      have h3 : ¬ i < j := by omega
      simp [h1, h2, h3]

/-- **`Draw.strataMessage` finds nothing at fault in a work sampling design at `k = 1` over such cells.** -/
theorem strataMessage_cells (φ : γ → Json) (tpl : γ → String) (lo hi kk : γ → ℕ) (zs : List γ) (m : ℕ)
    (law : Json) (hw : jeq (getD law "law") (Json.str "work") = true)
    (hss : getD law "strata" = Json.arr (zs.map φ).toArray) (hk : int? (getD law "k") = some 1)
    (hc : ∀ z ∈ zs, CellJson φ tpl lo hi kk z)
    (hne : zs ≠ []) (hz : ∀ z ∈ zs, lo z < hi z ∧ 1 ≤ kk z ∧ kk z ≤ hi z - lo z)
    (hnd : (zs.map tpl).Nodup) (hch : Chain 0 m (zs.map fun z => (lo z, hi z))) :
    Spot.Draw.strataMessage law (m : Int) = none := by
  have hm := cells_pos lo hi zs m hne (fun z h => (hz z h).1) hch
  have hlt : ∀ r ∈ zs.map fun z => (lo z, hi z), r.1 < r.2 := by
    intro r hr
    obtain ⟨z, hz', rfl⟩ := List.mem_map.1 hr
    exact (hz z hz').1
  have hsum := chain_sum hch hlt
  have hsz : ∀ z ∈ zs, hi z - lo z ≤ m := by
    intro z hz'
    have := le_sum_of_mem (List.mem_map_of_mem (f := fun r : ℕ × ℕ => r.2 - r.1)
      (List.mem_map_of_mem (f := fun z => (lo z, hi z)) hz'))
    simp only at this
    omega
  have h0 : ((zs.map φ).toArray.isEmpty || decide ((1 : Int) < 1) || decide ((m : Int) < 1)) = false := by
    have : (zs.map φ).toArray.isEmpty = false := by simpa using hne
    rw [this, Bool.false_or, Bool.or_eq_false_iff]
    exact ⟨by decide, decide_eq_false (by omega)⟩
  have hall : ((zs.map φ).toArray.all fun a =>
      keys? a == some (if True then Spot.Draw.WORK_STRATUM_KEYS else Spot.Draw.STRATUM_KEYS) &&
        (nat? (getD a "work")).isSome && (nat? (getD a "floor")).isSome &&
          (Spot.Draw.ranges? (getD a "units")).isSome) = true := by
    simp only [List.all_toArray, List.all_map, List.all_eq_true]
    intro z hz'
    have h := hc z hz'
    simp [h.keys, nat_of_int (n := 1) h.work, nat_of_int h.floor, h.units]
  have htotal : Array.foldl (fun a s => a + (int? (getD s "work")).getD 0 *
      Spot.Draw.sumSizes ((Spot.Draw.ranges? (getD s "units")).getD #[])) 0 (zs.map φ).toArray = (m : Int) := by
    rw [List.foldl_toArray' (h := rfl), foldl_total zs 0 hc fun z h => (hz z h).1]
    omega
  have hm0 : ((m : Int) == 0) = false := by
    rw [beq_eq_false_iff_ne]
    omega
  unfold Spot.Draw.strataMessage
  simp only [hss, hk, hw, h0, hall, htotal, hm0]
  simp only [Bool.false_eq_true, ↓reduceIte, Bool.not_true, List.forIn_toArray]
  rw [forIn_map_yield_foldl φ (stepG tpl lo hi)]
  · have hmapI : (zs.map fun z => ((lo z : Int), (hi z : Int))) =
        (zs.map fun z => (lo z, hi z)).map (fun r : ℕ × ℕ => ((r.1 : Int), (r.2 : Int))) := by
      rw [List.map_map]
      rfl
    have hloop := fun f hf => forIn_chain (fun r : ℕ × ℕ => ((r.1 : Int), (r.2 : Int)))
      (fun n : ℕ => (n : Int)) f hf (zs.map fun z => (lo z, hi z)) 0 m hch
    rw [foldl_stepG, hmapI]
    simp only [Array.empty_append, pure_bind, qsort_chainI hch hlt, List.forIn_toArray]
    rw [show ((0 : Int)) = ((0 : ℕ) : Int) from rfl, hloop]
    · simp only [pure_bind, beq_self_eq_true, ↓reduceIte]
      rfl
    · intro x
      rw [ite_eq_right (by simp)]
      rfl
  · intro pre x suf hl
    have hx : x ∈ zs := by rw [hl]; simp
    have h := hc x hx
    have hzx := hz x hx
    rw [foldl_stepG]
    have hseen : (#[] ++ (pre.map tpl).toArray).contains (tpl x) = false := by
      rw [hl, List.map_append, List.map_cons] at hnd
      have hnot : tpl x ∉ pre.map tpl := fun hmem =>
        (List.nodup_append.1 hnd).2.2 _ hmem _ List.mem_cons_self rfl
      simpa using hnot
    have hfl : (true && decide ((kk x : Int) < 1)) = false := by
      rw [Bool.true_and]
      exact decide_eq_false (by omega)
    have hrule : Flock.Draw.workK (Int.toNat 1) (Int.toNat 1) (((hi x : ℕ) : Int) - (lo x : ℕ)).toNat
        (m : Int).toNat ((kk x : ℕ) : Int).toNat = kk x := by
      have e1 : (((hi x : ℕ) : Int) - (lo x : ℕ)).toNat = hi x - lo x := by omega
      rw [e1, Int.toNat_natCast, Int.toNat_natCast]
      exact workK_unit hzx.2.1 hzx.2.2 (hsz x hx)
    simp only [h.template, h.keys, h.units, h.floor, h.work, h.k, hseen, Option.getD_some, sumSizes_pair, hfl,
      hrule, stepG]
    have hr : ((List.range (#[((lo x : Int), (hi x : Int))].size - 1)).any fun i =>
        decide (#[((lo x : Int), (hi x : Int))][i + 1]!.fst ≤ #[((lo x : Int), (hi x : Int))][i]!.snd)) = false := by
      simp
    simp only [bne_self_eq_false, Bool.false_eq_true, ↓reduceIte, hr]
    rfl

end Cells

/-! ## The second stage's cells -/

theorem chain_append : ∀ {l₁ l₂ : List (ℕ × ℕ)} {a b c : ℕ}, Chain a b l₁ → Chain b c l₂ →
    Chain a c (l₁ ++ l₂)
  | [], _, a, b, c, h₁, h₂ => by
    obtain rfl : a = b := h₁
    exact h₂
  | r :: rs, _, a, b, c, h₁, h₂ => ⟨h₁.1, chain_append h₁.2 h₂⟩

/-- A template's cells over a replay sample of `q` follow each other from `o` to `o + q · n`. -/
theorem chain_row (o n : ℕ) : ∀ q : ℕ,
    Chain o (o + q * n) ((List.range q).map fun i => (o + i * n, o + (i + 1) * n))
  | 0 => by
    show o = o + 0 * n
    omega
  | q + 1 => by
    rw [List.range_succ, List.map_append]
    refine chain_append (chain_row o n q) ?_
    exact ⟨rfl, rfl⟩

/-- The cells of the first `T` templates follow each other from `0` to `cellOff q ns T`. -/
theorem chain_cells (q : ℕ) (ns : Array ℕ) : ∀ T : ℕ,
    Chain 0 (cellOff q ns T) ((cellPairs q T).map fun z : ℕ × ℕ =>
      (cellOff q ns z.1 + z.2 * ns.getD z.1 0, cellOff q ns z.1 + (z.2 + 1) * ns.getD z.1 0))
  | 0 => by
    show 0 = cellOff q ns 0
    simp [cellOff]
  | T + 1 => by
    rw [cellPairs, List.range_succ, List.flatMap_append, List.map_append]
    refine chain_append (chain_cells q ns T) ?_
    rw [List.flatMap_singleton, List.map_map, cellOff_succ]
    exact chain_row (cellOff q ns T) (ns.getD T 0) q

theorem cellPairs_nodup (q T : ℕ) : (cellPairs q T).Nodup := by
  unfold cellPairs
  induction T with
  | zero => simp
  | succ T ih =>
    rw [List.range_succ, List.flatMap_append, List.flatMap_singleton, List.nodup_append]
    refine ⟨ih, ?_, ?_⟩
    · exact (List.pairwise_map.2 (List.Pairwise.imp (fun h e => h (Prod.mk.inj e).2) List.nodup_range))
    · intro a ha b hb hab
      obtain ⟨t, ht, ha⟩ := List.mem_flatMap.1 ha
      obtain ⟨i, -, rfl⟩ := List.mem_map.1 ha
      obtain ⟨j, -, rfl⟩ := List.mem_map.1 hb
      have := (Prod.mk.inj hab).1
      have := List.mem_range.1 ht
      omega

/-- Two strings of digits, each followed by the same non-digit, agree where what they start agrees. -/
theorem digits_split {s : Char} (hs : s.isDigit = false) {a b c d : List Char} (ha : ∀ x ∈ a, x.isDigit)
    (hc : ∀ x ∈ c, x.isDigit) (h : a ++ s :: b = c ++ s :: d) : a = c ∧ b = d := by
  induction a generalizing c with
  | nil =>
    cases c with
    | nil => simpa using h
    | cons x c =>
      simp only [List.nil_append, List.cons_append, List.cons.injEq] at h
      have := hc x (by simp)
      rw [← h.1, hs] at this
      exact absurd this (by decide)
  | cons x a ih =>
    cases c with
    | nil =>
      simp only [List.nil_append, List.cons_append, List.cons.injEq] at h
      have := ha x (by simp)
      rw [h.1, hs] at this
      exact absurd this (by decide)
    | cons y c =>
      simp only [List.cons_append, List.cons.injEq] at h
      obtain ⟨rfl, h⟩ := h
      have := ih (fun z hz => ha z (by simp [hz])) (fun z hz => hc z (by simp [hz])) h
      exact ⟨by rw [this.1], this.2⟩

/-- A cell's template names its replay block and its template: `r/t` is injective. -/
theorem cellTemplate_inj {r t r' t' : ℕ} (h : (s!"{r}/{t}" : String) = s!"{r'}/{t'}") : r = r' ∧ t = t' := by
  have h := congrArg String.toList h
  simp only [String.toList_append, toString, Nat.toList_repr, List.append_assoc] at h
  have dig : ∀ n, ∀ x ∈ Nat.toDigits 10 n, x.isDigit := fun n x hx =>
    Nat.isDigit_of_mem_toDigits (by decide) (by decide) hx
  obtain ⟨h1, h2⟩ := digits_split (s := '/') (by decide) (dig r) (dig r') (by simpa using h)
  exact ⟨Nat.repr_injective (String.toList_injective (by rw [Nat.toList_repr, Nat.toList_repr, h1])),
    Nat.repr_injective (String.toList_injective (by rw [Nat.toList_repr, Nat.toList_repr, h2]))⟩

/-- A strictly ascending replay sample holds no replay block twice. -/
theorem ascending_getD_inj (sample : Array ℕ)
    (hs : (sample.toList.zip sample.toList.tail).all (fun (a, b) => decide (a < b)) = true) {i j : ℕ}
    (hi : i < sample.size) (hj : j < sample.size) (h : sample.getD i 0 = sample.getD j 0) : i = j := by
  have hpw : sample.toList.Pairwise (· < ·) := by
    generalize sample.toList = l at hs
    induction l with
    | nil => exact List.Pairwise.nil
    | cons a l ih =>
      cases l with
      | nil => exact List.pairwise_singleton _ _
      | cons b l =>
        simp only [List.tail_cons, List.zip_cons_cons, List.all_cons, Bool.and_eq_true, decide_eq_true_eq] at hs
        have ih' := ih (by simpa using hs.2)
        refine List.Pairwise.cons ?_ ih'
        intro c hc
        rcases List.mem_cons.1 hc with rfl | hc
        · exact hs.1
        · exact Nat.lt_trans hs.1 (List.rel_of_pairwise_cons ih' hc)
  have hget := List.pairwise_iff_getElem.1 hpw
  have hi' : sample.getD i 0 = sample.toList[i]'(by simpa using hi) := by simp [Array.getD, hi]
  have hj' : sample.getD j 0 = sample.toList[j]'(by simpa using hj) := by simp [Array.getD, hj]
  rw [hi', hj'] at h
  rcases Nat.lt_trichotomy i j with hij | rfl | hij
  · have := hget i j (by simpa using hi) (by simpa using hj) hij
    omega
  · rfl
  · have := hget j i (by simpa using hj) (by simpa using hi) hij
    omega

theorem cellPairs_mem {q T : ℕ} {z : ℕ × ℕ} (hz : z ∈ cellPairs q T) : z.1 < T ∧ z.2 < q := by
  obtain ⟨t, ht, hz⟩ := List.mem_flatMap.1 hz
  obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hz
  exact ⟨List.mem_range.1 ht, List.mem_range.1 hi⟩

/-- The cells' templates are distinct over a strictly ascending replay sample. -/
theorem templates_nodup (sample : Array ℕ) (T : ℕ)
    (hs : (sample.toList.zip sample.toList.tail).all (fun (a, b) => decide (a < b)) = true) :
    ((cellPairs sample.size T).map fun z : ℕ × ℕ => (s!"{sample.getD z.2 0}/{z.1}" : String)).Nodup := by
  refine List.Nodup.map_on ?_ (cellPairs_nodup _ _)
  intro x hx y hy h
  obtain ⟨h1, h2⟩ := cellTemplate_inj h
  have hi := ascending_getD_inj sample hs (cellPairs_mem hx).2 (cellPairs_mem hy).2 h1
  exact Prod.ext h2 hi

/-! ## The second stage's sampling design is well formed -/

/-- The sampling design's strata are its cells' objects, in the cells' order. -/
theorem getD_secondLaw_cells (sample ns ks : Array ℕ) :
    getD (Select.secondLaw sample ns ks) "strata" = Json.arr ((cellPairs sample.size ns.size).map fun z =>
      Select.cellObj (cellOff sample.size ns z.1) (ns.getD z.1 0) (ks.getD z.1 0) z.2 (sample.getD z.2 0)
        z.1).toArray :=
  (getD_secondLaw_strata sample ns ks).trans
    (congrArg (fun l : List Json => Json.arr l.toArray) (strata_second sample ns ks))

/-- A second stage `secondProblem` passes has a template. -/
theorem secondProblem_ns {sample ns ks : Array ℕ} (h : Select.secondProblem sample ns ks = none) : ns.size ≠ 0 := by
  unfold Select.secondProblem at h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i hns
      intro h0
      have hb : ns.isEmpty = true := by
        rw [Array.size_eq_zero_iff.1 h0]
        rfl
      exact hns (by rw [hb, Bool.true_or, Bool.true_or])

/-- A cell of a second stage `secondProblem` passes is a nonempty range, sampling `1 ≤ k_t ≤ n_t` of its units. -/
theorem cell_counts {sample ns ks : Array ℕ} (hp : Select.secondProblem sample ns ks = none) {z : ℕ × ℕ}
    (hz : z ∈ cellPairs sample.size ns.size) :
    cellOff sample.size ns z.1 + z.2 * ns.getD z.1 0 < cellOff sample.size ns z.1 + (z.2 + 1) * ns.getD z.1 0 ∧
      1 ≤ ks.getD z.1 0 ∧ ks.getD z.1 0 ≤ cellOff sample.size ns z.1 + (z.2 + 1) * ns.getD z.1 0 -
        (cellOff sample.size ns z.1 + z.2 * ns.getD z.1 0) := by
  obtain ⟨hk1, hkn⟩ := secondProblem_counts hp (cellPairs_mem hz).1
  have e : (z.2 + 1) * ns.getD z.1 0 = z.2 * ns.getD z.1 0 + ns.getD z.1 0 := Nat.succ_mul _ _
  omega

theorem keys_cellObj (o n k i r t : ℕ) : keys? (Select.cellObj o n k i r t) = some Spot.Draw.WORK_STRATUM_KEYS := by
  unfold Select.cellObj Spot.Draw.WORK_STRATUM_KEYS
  exact keys_mkObj_perm (ks := ["template", "units", "work", "floor", "k"]) (List.Perm.refl _) (by decide)

theorem keys_secondLaw (sample ns ks : Array ℕ) :
    keys? (Select.secondLaw sample ns ks) = some (sortStrings #["law", "k", "strata"]) := by
  unfold Select.secondLaw
  exact keys_mkObj_perm (ks := ["law", "k", "strata"]) (List.Perm.refl _) (by decide)

/-- **The second stage's sampling design parses, and its strata are well formed**, wherever `secondProblem` passes. -/
theorem second_parses {sample ns ks : Array ℕ} (hp : Select.secondProblem sample ns ks = none) :
    Draw.flockLaw (Select.secondLaw sample ns ks) (sample.size * ns.toList.sum) =
        .ok (.work 1 ((cellPairs sample.size ns.size).map (cellStratum sample ns ks)).toArray) ∧
      Spot.Draw.strataMessage (Select.secondLaw sample ns ks) ((sample.size * ns.toList.sum : ℕ) : Int) = none := by
  let φ : ℕ × ℕ → Json := fun z =>
    Select.cellObj (cellOff sample.size ns z.1) (ns.getD z.1 0) (ks.getD z.1 0) z.2 (sample.getD z.2 0) z.1
  let tpl : ℕ × ℕ → String := fun z => s!"{sample.getD z.2 0}/{z.1}"
  let lo : ℕ × ℕ → ℕ := fun z => cellOff sample.size ns z.1 + z.2 * ns.getD z.1 0
  let hi : ℕ × ℕ → ℕ := fun z => cellOff sample.size ns z.1 + (z.2 + 1) * ns.getD z.1 0
  let kk : ℕ × ℕ → ℕ := fun z => ks.getD z.1 0
  have hs := secondProblem_sample hp
  have hq : sample.size ≠ 0 := fun h0 => by
    have := hs.1
    rw [Array.size_eq_zero_iff.1 h0] at this
    exact absurd this (by decide)
  have hT := secondProblem_ns hp
  have hne : cellPairs sample.size ns.size ≠ [] := fun h => by
    have hm : (0, 0) ∈ cellPairs sample.size ns.size :=
      List.mem_flatMap.2 ⟨0, List.mem_range.2 (by omega), List.mem_map.2 ⟨0, List.mem_range.2 (by omega), rfl⟩⟩
    rw [h] at hm
    exact List.not_mem_nil hm
  have hw : jeq (getD (Select.secondLaw sample ns ks) "law") (Json.str "work") = true := by
    rw [getD_secondLaw_law]
    obtain ⟨t, ht⟩ := FlockComplete.JsonRT.canon_ok_some _ (Claims.Json.Good.str "work")
    exact jeq_self ht
  have hk : int? (getD (Select.secondLaw sample ns ks) "k") = some 1 :=
    (congrArg int? (getD_secondLaw_k sample ns ks)).trans (int_toJson 1)
  have hc : ∀ z ∈ cellPairs sample.size ns.size, CellJson φ tpl lo hi kk z := fun z hz =>
    ⟨keys_cellObj _ _ _ _ _ _, getD_cellObj_template _ _ _ _ _ _,
      (congrArg Draw.ranges? (getD_cellObj_units _ _ _ _ _ _)).trans (ranges_pair_ok (cell_counts hp hz).1),
      (congrArg int? (getD_cellObj_k _ _ _ _ _ _)).trans (int_toJson _),
      (congrArg int? (getD_cellObj_work _ _ _ _ _ _)).trans (int_toJson 1),
      (congrArg int? (getD_cellObj_floor _ _ _ _ _ _)).trans (int_toJson _)⟩
  have hz : ∀ z ∈ cellPairs sample.size ns.size, lo z < hi z ∧ 1 ≤ kk z ∧ kk z ≤ hi z - lo z :=
    fun z hz => cell_counts hp hz
  have hnd : ((cellPairs sample.size ns.size).map tpl).Nodup := templates_nodup sample ns.size hs.2
  have hch : Chain 0 (sample.size * ns.toList.sum) ((cellPairs sample.size ns.size).map fun z => (lo z, hi z)) := by
    have h := chain_cells sample.size ns ns.size
    rw [cellOff_size] at h
    exact h
  exact ⟨flockLaw_cells φ tpl lo hi kk _ _ _ (getD_secondLaw_law sample ns ks) hw (keys_secondLaw sample ns ks)
      (getD_secondLaw_cells sample ns ks) hk hc hne hz hnd hch,
    strataMessage_cells φ tpl lo hi kk _ _ _ hw (getD_secondLaw_cells sample ns ks) hk hc hne hz hnd hch⟩

/-- **The second stage's sampling design parses to its cells' strata** wherever `secondProblem` passes. -/
theorem flockLaw_second_ok {sample ns ks : Array ℕ} (hp : Select.secondProblem sample ns ks = none) :
    Draw.flockLaw (Select.secondLaw sample ns ks) (sample.size * ns.toList.sum) =
      .ok (.work 1 ((cellPairs sample.size ns.size).map (cellStratum sample ns ks)).toArray) :=
  (second_parses hp).1

/-- **The second stage's strata are well formed** wherever `secondProblem` passes: `Draw.strataMessage` finds nothing
at fault in them. -/
theorem strataMessage_second {sample ns ks : Array ℕ} (hp : Select.secondProblem sample ns ks = none) :
    Draw.strataMessage (Select.secondLaw sample ns ks) ((sample.size * ns.toList.sum : ℕ) : Int) = none :=
  (second_parses hp).2

/-- **The second stage's sampling design is well formed** (`SecondFormed`) wherever `secondProblem` passes. -/
theorem secondFormed {sample ns ks : Array ℕ} (hp : Select.secondProblem sample ns ks = none) :
    SecondFormed sample ns ks :=
  ⟨⟨_, flockLaw_second_ok hp⟩, strataMessage_second hp⟩

/-- **The second stage refuses an honest run with probability at most `|sample| · Σ_t k_t · 2⁻⁶⁴`**: where
`secondProblem` passes, on a uniform stream of at least `secondRead` bytes, `selectSecond` returns a sample unless some
cell's first read samples nothing. -/
theorem selectSecond_refused_le {sample ns ks : Array ℕ} (hp : Select.secondProblem sample ns ks = none) (len : ℕ)
    (hlen : secondRead sample.size ns ks ≤ len) :
    prCoin (fun f : Fin len → Fin 256 => ¬ ∃ draw read,
      Select.selectSecond sample ns ks (bytesOf f) = .ok (draw, read)) ≤
      ((sample.size * ks.toList.sum : ℕ) : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ :=
  selectSecond_refused_le_of hp (secondFormed hp) len hlen

end Properties.Spot.SecurityProofs
