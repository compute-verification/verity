module

public import Proofs.Flock.Claims.Sampling.Select.Design
public import Properties.SampledProofs.Second
public import Batteries.Tactic.OpenPrivate
public import Std.Data.TreeMap.Raw.Lemmas
public import Std.Data.TreeMap.Raw.WF
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic

/-!
# The second stage misses a set of units no more often than its strata's escape (`SecondMisses`)

* `flockLaw_second`: `Draw.flockLaw` reads `Select.secondLaw` as the stratified law over one stratum per sampled replay
  block, `[i·n, (i+1)·n)` with `k` drawn (`secondStrata`).
* `selectSecond_ok`: an accepted second stage has `1 ≤ k < n`, and its draw object holds the units `Draw.drawUnits`
  draws under that law (`mem_proved`: each of them is one `Draw.proved` reads).
* `secondMisses`: `strata_bound` on those strata, whose units in order are `[0, sample.size · n)`, its `strataPa` the
  product of `escCell n k |X ∩ [i·n, (i+1)·n)|`.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open SampledProofs

namespace Properties.SampledProofs.SecurityProofs

open Lean (Json)
open FlockSoundness.Audit Finset Claims.Sampling
open private SampledProofs.Draw.natRanges SampledProofs.Draw.stratum? from SampledProofs.Draw

theorem int_toJson (x : ℕ) : int? (Lean.toJson x) = some (x : Int) := rfl

/-- An object's field from its last entries, when they name it once. -/
theorem getD_mkObj_append (A l : List (String × Json)) (hd : l.Pairwise fun a b => ¬ compare a.1 b.1 = .eq)
    {k : String} {v : Json} (hm : (k, v) ∈ l) : getD (Json.mkObj (A ++ l)) k = v := by
  have h : (Std.TreeMap.Raw.ofList (A ++ l) compare)[k]? = some v := by
    rw [Std.TreeMap.Raw.ofList_eq_insertMany_empty, Std.TreeMap.Raw.insertMany_append]
    exact Std.TreeMap.Raw.getElem?_insertMany_list_of_mem
      (Std.TreeMap.Raw.WF.insertMany Std.TreeMap.Raw.WF.emptyc) Std.ReflCmp.compare_self hd hm
  simp only [getD, field?, Json.mkObj, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?, h]
  rfl

theorem getD_mkObj (l : List (String × Json)) (hd : l.Pairwise fun a b => ¬ compare a.1 b.1 = .eq)
    {k : String} {v : Json} (hm : (k, v) ∈ l) : getD (Json.mkObj l) k = v :=
  getD_mkObj_append [] l hd hm

section
variable (sample : Array ℕ) (n k : ℕ)

theorem secondLaw_law : getD (Select.secondLaw sample n k) "law" = Json.str "stratified" :=
  getD_mkObj _ (by simp) (by simp)

theorem secondLaw_k : getD (Select.secondLaw sample n k) "k" = Lean.toJson (sample.size * k) :=
  getD_mkObj _ (by simp) (by simp)

theorem secondLaw_strata : getD (Select.secondLaw sample n k) "strata" =
    Json.arr (sample.mapIdx fun i r => Json.mkObj [("template", Json.str (toString r)),
      ("units", Json.arr #[natsJson #[i * n, (i + 1) * n]]), ("k", Lean.toJson k)]) :=
  getD_mkObj _ (by simp) (by simp)

end

/-- The second stage's stratum of replay block `r`, the `i`-th sampled: its proof blocks `[i·n, (i+1)·n)`, `k` drawn. -/
def secondStratum (n k i r : ℕ) : Flock.Draw.Stratum := ⟨toString r, #[(i * n, (i + 1) * n)], k⟩

/-- The second stage's strata, one per sampled replay block. -/
def secondStrata (sample : Array ℕ) (n k : ℕ) : Array Flock.Draw.Stratum := sample.mapIdx (secondStratum n k)

/-- `secondLaw`'s stratum object of replay block `r`, the `i`-th sampled. -/
def stratumObj (n k i r : ℕ) : Json := Json.mkObj [("template", Json.str (toString r)),
  ("units", Json.arr #[natsJson #[i * n, (i + 1) * n]]), ("k", Lean.toJson k)]

theorem ranges_pair (a b : ℕ) (h : a < b) :
    Draw.ranges? (Json.arr #[natsJson #[a, b]]) = some #[((a : Int), (b : Int))] := by
  simp [Draw.ranges?, natsJson, int_toJson, h]

private theorem natRanges_pair (a b : ℕ) : SampledProofs.Draw.natRanges #[((a : Int), (b : Int))] = some #[(a, b)] := by
  simp [SampledProofs.Draw.natRanges]

private theorem stratum_obj (n k i r : ℕ) (hn : 0 < n) (ws : Flock.Draw.WorkStratum)
    (h : SampledProofs.Draw.stratum? false (stratumObj n k i r) = some ws) : ws.toStratum = secondStratum n k i r := by
  have ht : getD (stratumObj n k i r) "template" = Json.str (toString r) := getD_mkObj _ (by simp) (by simp)
  have hu : getD (stratumObj n k i r) "units" = Json.arr #[natsJson #[i * n, (i + 1) * n]] :=
    getD_mkObj _ (by simp) (by simp)
  have hk : getD (stratumObj n k i r) "k" = Lean.toJson k := getD_mkObj _ (by simp) (by simp)
  have hlt : i * n < (i + 1) * n := by rw [Nat.succ_mul]; omega
  unfold SampledProofs.Draw.stratum? at h
  simp only [ht, hu, hk, ranges_pair _ _ hlt, nat_toJson, Bool.false_eq_true, ↓reduceIte] at h
  split at h
  · exact absurd h (by simp)
  · simp only [bind, Option.bind, pure, natRanges_pair, Option.some.injEq] at h
    rw [← h]
    rfl

private theorem strata_mapM (n k : ℕ) (hn : 0 < n) : ∀ (l : List ℕ) (o : ℕ) (sl : List Flock.Draw.WorkStratum),
    (l.mapIdx fun i r => stratumObj n k (o + i) r).mapM (SampledProofs.Draw.stratum? false) = some sl →
      sl.map (·.toStratum) = l.mapIdx fun i r => secondStratum n k (o + i) r
  | [], _, sl, h => by simp at h; simp [h]
  | r :: l, o, sl, h => by
    simp only [List.mapIdx_cons, List.mapM_cons, bind, Option.bind, pure] at h
    split at h
    · exact absurd h (by simp)
    · rename_i ws hws
      dsimp only at h
      split at h
      · exact absurd h (by simp)
      · rename_i sl' hsl
        simp only [Option.some.injEq] at h
        subst h
        have ih := strata_mapM n k hn l (o + 1) sl' (by simpa [Nat.add_assoc, Nat.add_comm 1] using hsl)
        simp only [List.map_cons, List.mapIdx_cons, Nat.add_zero, stratum_obj n k o r hn ws hws]
        rw [ih]
        simp [Nat.add_assoc, Nat.add_comm 1]

/-- **`flockLaw` reads `secondLaw`** as the stratified law of `sample.size · k` over one stratum per sampled replay
block: the `i`-th's proof blocks `[i·n, (i+1)·n)`, `k` drawn. -/
theorem flockLaw_second (sample : Array ℕ) (n k m : ℕ) (hn : 0 < n) (fl : Flock.Draw.Law)
    (h : Draw.flockLaw (Select.secondLaw sample n k) m = .ok fl) :
    fl = .stratified (sample.size * k) (secondStrata sample n k) := by
  unfold Draw.flockLaw at h
  simp only [secondLaw_law, secondLaw_k, secondLaw_strata, nat_toJson] at h
  split at h
  · exact absurd h (throw_bind_ne _ _ _)
  split at h
  · exact absurd h (throw_bind_ne _ _ _)
  split at h
  · rename_i strata hm
    have hw : ("stratified" == "work") = false := by decide
    simp only [hw, Bool.false_eq_true, ↓reduceIte] at h hm
    obtain rfl := (Except.ok.inj h).symm
    have e : ∀ f : ℕ → ℕ → Json, sample.mapIdx f = (sample.toList.mapIdx f).toArray := fun f =>
      Array.toList_inj.1 (by simp)
    rw [e, List.mapM_toArray] at hm
    obtain ⟨sl, hsl, rfl⟩ := Option.map_eq_some_iff.1 hm
    have := strata_mapM n k hn sample.toList 0 sl (by simpa [stratumObj] using hsl)
    congr 1
    apply Array.toList_inj.1
    simp only [List.map_toArray, secondStrata, Array.toList_mapIdx, this, Nat.zero_add]
  · exact absurd h (throw_ne _ _)

/-- **An accepted second stage**: `1 ≤ k < n`, and its draw object is `secondLaw`'s entries with the population and the
units `drawUnits` draws under the law `flockLaw` reads, with the bytes it read. -/
theorem selectSecond_ok (sample : Array ℕ) (n k : ℕ) (src : ByteArray) (draw : Json) (read : ℕ)
    (h : Select.selectSecond sample n k src = .ok (draw, read)) :
    1 ≤ k ∧ k < n ∧ ∃ fl units, Draw.flockLaw (Select.secondLaw sample n k) (sample.size * n) = .ok fl ∧
      Draw.drawUnits fl (sample.size * n) src = .ok (units, read) ∧
      draw = Json.mkObj ((entries (Select.secondLaw sample n k)).toList ++
        [("population", Lean.toJson (sample.size * n)), ("units", natsJson units)]) := by
  unfold Select.selectSecond at h
  dsimp only at h
  split at h
  · exact absurd h (throw_bind_ne _ _ _)
  split at h
  · exact absurd h (throw_bind_ne _ _ _)
  split at h
  · exact absurd h (throw_bind_ne _ _ _)
  rename_i hk
  have hk' : 1 ≤ k ∧ k < n := by simp only [Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at hk; omega
  obtain ⟨⟨draw', read'⟩, hd, h⟩ := bind_eq_ok.1 h
  dsimp only at h
  split at h
  · exact absurd h (throw_bind_ne _ _ _)
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj h)
  unfold Draw.drawFirstRead at hd
  obtain ⟨fl, hfl, hd⟩ := bind_eq_ok.1 hd
  obtain ⟨⟨units, read''⟩, hu, hd⟩ := bind_eq_ok.1 hd
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hd)
  exact ⟨hk'.1, hk'.2, fl, units, hfl, hu, rfl⟩

/-- A unit of the draw object's `units` is one it proves. -/
theorem mem_proved (A : List (String × Json)) (pop : Json) (units : Array ℕ) (u : ℕ) (hu : u ∈ units) :
    (u : Int) ∈ Draw.proved (Json.mkObj (A ++ [("population", pop), ("units", natsJson units)])) := by
  have hg : getD (Json.mkObj (A ++ [("population", pop), ("units", natsJson units)])) "units" = natsJson units :=
    getD_mkObj_append A _ (by simp) (by simp)
  unfold Draw.proved
  dsimp only
  rw [hg, DrawExec.mem_qsort, Array.mem_append]
  simp only [natsJson]
  exact Or.inl (Array.mem_filterMap.2 ⟨_, Array.mem_map_of_mem hu, int_toJson u⟩)

theorem secondStratum_size (n k i r : ℕ) : (secondStratum n k i r).size = n := by
  simp [Flock.Draw.Stratum.size, secondStratum, Nat.succ_mul]

/-- The strata's sizes and draws: `(n, k)` for each sampled replay block. -/
theorem secondStrata_sizes (sample : Array ℕ) (n k : ℕ) :
    (secondStrata sample n k).toList.map (fun s => (s.size, s.k)) = List.replicate sample.size (n, k) := by
  apply List.ext_getElem (by simp [secondStrata])
  intro i h1 h2
  simp only [List.getElem_map, Array.getElem_toList, secondStrata, Array.getElem_mapIdx, secondStratum_size,
    List.getElem_replicate]
  rfl

theorem flatMap_second (n k : ℕ) : ∀ (l : List ℕ) (o : ℕ),
    (l.mapIdx fun i r => secondStratum n k (o + i) r).flatMap (fun s => expand s.units.toList) =
      List.range' (o * n) (l.length * n)
  | [], _ => by simp
  | r :: l, o => by
    have e : (fun i r => secondStratum n k (o + (i + 1)) r) = fun i r => secondStratum n k (o + 1 + i) r := by
      funext i r; congr 1; omega
    rw [List.mapIdx_cons, List.flatMap_cons, e, flatMap_second n k l (o + 1)]
    simp only [secondStratum, expand, Nat.add_zero, List.toList_toArray, List.flatMap_cons, List.flatMap_nil,
      List.append_nil, List.length_cons]
    rw [show (o + 1) * n - o * n = n by rw [Nat.succ_mul]; omega, show (o + 1) * n = o * n + n by rw [Nat.succ_mul],
      List.range'_append_1, Nat.succ_mul, Nat.add_comm n]

/-- The strata's units, in order, are the second stage's `[0, sample.size · n)`. -/
theorem secondStrata_units (sample : Array ℕ) (n k : ℕ) :
    (secondStrata sample n k).toList.flatMap (fun s => expand s.units.toList) = List.range (sample.size * n) := by
  have := flatMap_second n k sample.toList 0
  simp only [Nat.zero_add, Nat.zero_mul, Array.length_toList] at this
  rw [secondStrata, Array.toList_mapIdx, this, List.range_eq_range']

/-- `escCell` at `k < n` is the subset escape `C(n − m, k)/C(n, k)`. -/
theorem escCell_lt (n k m : ℕ) (hk : k < n) : escCell n k m = ((n - m).choose k : ℝ) / (n.choose k : ℝ) := by
  unfold escCell
  rw [min_eq_left hk.le]

/-- `strataPa` of `j` strata of `(n, k)` from slot `o`: the product of their subset escapes. -/
theorem strataPa_replicate (n k : ℕ) (W : Finset ℕ) : ∀ (j o : ℕ),
    strataPa (List.replicate j (n, k)) o W =
      ∏ i ∈ range j, ((n - (window (o + i * n) n W).card).choose k : ℝ) / (n.choose k : ℝ)
  | 0, o => by simp [strataPa]
  | j + 1, o => by
    rw [List.replicate_succ, strataPa, strataPa_replicate n k W j (o + n), prod_range_succ']
    have h : ∀ i, o + n + i * n = o + (i + 1) * n := fun i => by rw [Nat.succ_mul]; omega
    simp only [h, Nat.zero_mul, Nat.add_zero]
    exact mul_comm _ _

/-- The window of replay block `i`'s proof blocks in `X`'s units below `m`: `X ∩ [i·n, (i+1)·n)`, shifted. -/
theorem window_second (X : Finset ℕ) (m n i : ℕ) (hi : (i + 1) * n ≤ m) :
    (window (i * n) n (X.filter (· < m))).card = (X ∩ Ico (i * n) ((i + 1) * n)).card := by
  unfold window
  rw [card_image_of_injOn]
  · congr 1
    ext v
    simp only [mem_filter, mem_inter, mem_Ico]
    rw [Nat.succ_mul] at hi ⊢
    constructor
    · rintro ⟨⟨hx, -⟩, h1, h2⟩
      exact ⟨hx, h1, h2⟩
    · rintro ⟨hx, h1, h2⟩
      exact ⟨⟨hx, by omega⟩, h1, h2⟩
  · intro a ha b hb h
    have ha' := (mem_filter.1 (mem_coe.1 ha)).2.1
    have hb' := (mem_filter.1 (mem_coe.1 hb)).2.1
    have h' : a - i * n = b - i * n := h
    omega

/-- **The second stage misses `X` no more often than its strata's escape** (`SecondMisses`). -/
theorem secondMisses : SecondMisses := by
  intro sample n k L X
  have hprod : 0 ≤ ∏ i ∈ range sample.size, escCell n k (X ∩ Ico (i * n) ((i + 1) * n)).card :=
    prod_nonneg fun i _ => by unfold escCell; positivity
  by_cases hk : 1 ≤ k ∧ k < n
  · have hn : 0 < n := by omega
    have hb := strata_bound (.stratified (sample.size * k) (secondStrata sample n k)) (secondStrata sample n k)
      (Or.inl ⟨_, rfl⟩) (sample.size * n) (X.filter (· < sample.size * n)) L
    refine le_trans (Claims.Pr_mono fun src hsrc => ?_) (le_of_le_of_eq hb ?_)
    · obtain ⟨hok, hmiss⟩ := hsrc
      obtain ⟨⟨draw, read⟩, hsel⟩ : ∃ p, Select.selectSecond sample n k (bytesOf src) = .ok p := by
        cases h : Select.selectSecond sample n k (bytesOf src) with
        | error e => rw [h] at hok; exact absurd hok (by simp [Except.isOk, Except.toBool])
        | ok p => exact ⟨p, rfl⟩
      obtain ⟨-, -, fl, units, hfl, hu, hdraw⟩ := selectSecond_ok sample n k _ draw read hsel
      obtain rfl := flockLaw_second sample n k _ hn fl hfl
      refine ⟨units, read, hu, fun v hv hvu => ?_⟩
      obtain ⟨hvX, hvm⟩ := mem_filter.1 hv
      rw [secondStrata_units, List.getD_eq_getElem?_getD, List.getElem?_range hvm, Option.getD_some] at hvu
      refine hmiss v hvX ?_
      rw [hsel]
      show (v : Int) ∈ Draw.proved draw
      rw [hdraw]
      exact mem_proved _ _ units v hvu
    · rw [secondStrata_sizes, strataPa_replicate]
      refine prod_congr rfl fun i hi => ?_
      rw [Nat.zero_add, window_second X _ n i (Nat.mul_le_mul_right n (mem_range.1 hi)), escCell_lt n k _ hk.2]
  · refine le_trans (Claims.Pr_mono fun src hsrc => ?_) (Claims.Pr_false.trans_le hprod)
    obtain ⟨hok, -⟩ := hsrc
    cases h : Select.selectSecond sample n k (bytesOf src) with
    | error e => rw [h] at hok; exact absurd hok (by simp [Except.isOk, Except.toBool])
    | ok p =>
      obtain ⟨draw, read⟩ := p
      obtain ⟨h1, h2, -⟩ := selectSecond_ok sample n k _ draw read h
      exact hk ⟨h1, h2⟩

end Properties.SampledProofs.SecurityProofs
