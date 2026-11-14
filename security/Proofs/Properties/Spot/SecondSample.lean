module

public import Proofs.Flock.Claims.Sampling.Select.SecondLaw
public import Proofs.Flock.Claims.Sampling.Select.SecondUnits
public import Proofs.Properties.Spot
public import Proofs.Flock.Claims.Complete.Json.Canon
public import Proofs.Flock.Claims.Complete.Json.Record
public import Proofs.Pouw.Period.Sort
public import Proofs.Flock.Recursive.FirewallObj
import Spot.Draw
import Spot.Json
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String
import Std.Data.DTreeMap.Internal.WF.Lemmas
import Proofs.Flock.Claims.Sampling.Select.Take
import Proofs.Flock.Claims.Sampling.Select.Steps

/-!
# The second stage accepts the sample its coins give

`Select.selectSecond` refuses a replay sample and cells `secondProblem` refuses, a sampling design `Draw.flockLaw`
doesn't read, a sample the coins don't end (`Draw.drawUnits`), and a sample object `Draw.checkDraw` finds at fault. This
file shows the last never happens on its own: where the second stage's sampling design is well formed (`SecondFormed`:
`Draw.flockLaw` reads it and `Draw.strataMessage` finds nothing), every sample `Draw.drawUnits` returns under it passes
`Draw.checkDraw`, so `selectSecond` returns it. The second stage therefore refuses only where its units are not
sampled, which `drawUnits_second_error_le` bounds.

* `entries_mkObj_perm`, `keys_mkObj_perm`: an object built from members with distinct keys has those keys.
* `secondObj`: the sample object `selectSecond` builds from the units, with its fields, keys and sampling design.
* `countIn_cell`: `checkDraw`'s count of the units in a cell, as the list's.
* `SecondFormed`: the sampling design is well formed, which `secondFormed` shows wherever `secondProblem` passes.
* `drawn_secondObj`, `checkDraw_secondObj`: the sample object is `Properties.Spot.Drawn`, so `checkDraw` passes.
* `selectSecond_of_units`: `selectSecond` returns the sample `Draw.drawUnits` gives.
* `selectSecond_refused_le_of`: under a well-formed sampling design, on a uniform stream of at least `secondRead` bytes,
  `selectSecond` refuses with probability at most `|sample| · Σ_t k_t · 2⁻⁶⁴`.
-/

@[expose] public section

open Spot

namespace Properties.Spot.SecurityProofs

open Lean (Json)
open Std.DTreeMap.Internal (Impl)
open FlockSoundness FlockSoundness.Game Finset
open scoped ENNReal

/-! ## An object's keys -/

/-- An object built from members with distinct keys lists those keys. -/
theorem entries_mkObj_perm {L : List (String × Json)} (h : (L.map (·.1)).Nodup) :
    ((entries (Json.mkObj L)).toList.map (·.1)).Perm (L.map (·.1)) := by
  rw [FlockSoundness.Discharge.Recursion.FirewallObj.mkObj_eq]
  show (((⟨⟨Impl.Const.ofList L⟩⟩ : Std.TreeMap.Raw String Json compare).toArray).toList.map (·.1)).Perm _
  rw [Std.TreeMap.Raw.toList_toArray]
  change ((Impl.Const.toList (Impl.Const.ofList L : Impl String fun _ => Json)).map (·.1)).Perm (L.map (·.1))
  rw [Impl.Const.toList_eq_toListModel_map, List.map_map]
  have hp := (FlockSoundness.Discharge.Recursion.FirewallObj.perm_ofList (List.pairwise_map.1 h)).map
    fun e : (_ : String) × Json => e.1
  rw [List.map_map] at hp
  exact hp

/-- **An object's keys**: built from members whose keys are the distinct `ks` in some order, its keys are `ks`,
sorted. -/
theorem keys_mkObj_perm {L : List (String × Json)} {ks : List String} (hp : (L.map (·.1)).Perm ks)
    (hk : ks.Nodup) : keys? (Json.mkObj L) = some (sortStrings ks.toArray) := by
  have he := entries_mkObj_perm (hp.nodup_iff.2 hk)
  rw [FlockSoundness.Discharge.Recursion.FirewallObj.mkObj_eq] at he ⊢
  show some (sortStrings ((⟨⟨Impl.Const.ofList L⟩⟩ : Std.TreeMap.Raw String Json compare).toArray.map (·.1))) = _
  refine congrArg some (Array.toList_inj.1 (Properties.Pouw.Period.SecurityProofs.Sort.qsort_eq_of_perm ?_))
  rw [Array.toList_map]
  exact he.trans hp

/-! ## The sample object -/

/-- **The second stage's sample object** over the interior's `|sample| · Σ_t n_t` proof blocks with the units `units`,
as `Draw.drawFirstRead` builds it: `secondLaw`'s members, the population and the units. -/
def secondObj (sample ns ks units : Array ℕ) : Json :=
  Json.mkObj ((entries (Select.secondLaw sample ns ks)).toList ++
    [("population", Lean.toJson (sample.size * ns.toList.sum)), ("units", natsJson units)])

theorem getD_obj_law (sample ns ks units : Array ℕ) :
    getD (secondObj sample ns ks units) "law" = Json.str "work" := (rfl)

theorem getD_obj_k (sample ns ks units : Array ℕ) :
    getD (secondObj sample ns ks units) "k" = Lean.toJson (1 : ℕ) := (rfl)

theorem getD_obj_strata (sample ns ks units : Array ℕ) :
    getD (secondObj sample ns ks units) "strata" = getD (Select.secondLaw sample ns ks) "strata" := (rfl)

theorem getD_obj_population (sample ns ks units : Array ℕ) :
    getD (secondObj sample ns ks units) "population" = Lean.toJson (sample.size * ns.toList.sum) := (rfl)

theorem getD_obj_units (sample ns ks units : Array ℕ) :
    getD (secondObj sample ns ks units) "units" = Json.arr (units.map fun x => Lean.toJson x) := (rfl)

theorem field_obj_closure (sample ns ks units : Array ℕ) :
    (field? (secondObj sample ns ks units) "closure").isSome = false := (rfl)

theorem kindOf_obj (sample ns ks units : Array ℕ) :
    Properties.Spot.kindOf (secondObj sample ns ks units) = "work" := (rfl)

theorem drawKeys_work : drawKeys "work" false = sortStrings (#["law", "population", "units"] ++ #["k", "strata"]) := by
  unfold drawKeys
  have h1 : ("work" == "subset") = false := by decide
  have h2 : ("work" == "bernoulli") = false := by decide
  simp only [h1, h2, Bool.false_eq_true, ↓reduceIte]

/-- The sample object has exactly a work sample's keys. -/
theorem keys_secondObj (sample ns ks units : Array ℕ) :
    keys? (secondObj sample ns ks units) = some (drawKeys "work" false) := by
  have hl : ((entries (Select.secondLaw sample ns ks)).toList.map (·.1)).Perm ["law", "k", "strata"] := by
    unfold Select.secondLaw
    exact entries_mkObj_perm (by show ["law", "k", "strata"].Nodup; decide)
  have h5 : (#["law", "population", "units"] ++ #["k", "strata"] : Array String) =
      ["law", "population", "units", "k", "strata"].toArray := by simp
  rw [drawKeys_work, h5]
  unfold secondObj
  refine keys_mkObj_perm ?_ (by decide)
  rw [List.map_append]
  exact (hl.append_right _).trans
    (List.Perm.cons "law" (List.perm_append_comm (l₁ := ["k", "strata"]) (l₂ := ["population", "units"])))

/-- A value `Flock.canon` prints equals itself. -/
theorem jeq_self {a : Json} {t : String} (h : Flock.canon a = .ok t) : jeq a a = true := by
  unfold jeq
  rw [h]
  simp

/-- The sampling design the sample object names is `secondLaw`. -/
theorem lawOf_secondObj (sample ns ks units : Array ℕ) :
    Draw.lawOf (secondObj sample ns ks units) = Select.secondLaw sample ns ks := by
  unfold Draw.lawOf
  have hw : jeq (getD (secondObj sample ns ks units) "law") "work" = true := by
    rw [getD_obj_law]
    obtain ⟨t, ht⟩ := FlockComplete.JsonRT.canon_ok_some _ (Claims.Json.Good.str "work")
    exact jeq_self ht
  simp only [hw, Bool.or_true, ↓reduceIte]
  rfl

/-- `Draw.strataMessage` reads the sample object's `law`, `k` and strata, which are `secondLaw`'s. -/
theorem strataMessage_secondObj (sample ns ks units : Array ℕ) (n : Int) :
    Draw.strataMessage (secondObj sample ns ks units) n = Draw.strataMessage (Select.secondLaw sample ns ks) n := by
  unfold Draw.strataMessage
  simp only [getD_obj_law, getD_obj_strata, getD_obj_k, getD_secondLaw_law, getD_secondLaw_k]

/-! ## `Flock.canon` prints `secondLaw` -/

/-- A cell's object is printable: distinct keys, a string, naturals and one range of naturals. -/
theorem good_cellObj (o n k i r t : ℕ) : Claims.Json.Good (Select.cellObj o n k i r t) := by
  unfold Select.cellObj
  refine FlockComplete.JsonRT.good_mkObj (by simp) fun p hp => ?_
  simp only [List.mem_cons, List.mem_nil_iff, or_false] at hp
  rcases hp with rfl | rfl | rfl | rfl | rfl
  · exact .str _
  · refine .arr fun x hx => ?_
    obtain rfl : x = natsJson #[o + i * n, o + (i + 1) * n] := List.mem_singleton.1 (Array.mem_toList_iff.2 hx)
    refine .arr fun y hy => ?_
    obtain ⟨a, -, rfl⟩ := Array.mem_map.1 hy
    exact FlockComplete.JsonRT.good_toJson a
  · exact FlockComplete.JsonRT.good_toJson 1
  · exact FlockComplete.JsonRT.good_toJson k
  · exact FlockComplete.JsonRT.good_toJson k

/-- `secondLaw` is printable: distinct keys, and each of its cells is. -/
theorem good_secondLaw (sample ns ks : Array ℕ) : Claims.Json.Good (Select.secondLaw sample ns ks) := by
  unfold Select.secondLaw
  refine FlockComplete.JsonRT.good_mkObj (by simp) fun p hp => ?_
  simp only [List.mem_cons, List.mem_nil_iff, or_false] at hp
  rcases hp with rfl | rfl | rfl
  · exact .str "work"
  · exact FlockComplete.JsonRT.good_toJson 1
  · refine .arr fun x hx => ?_
    obtain ⟨t, -, hx⟩ := List.mem_flatMap.1 (Array.mem_toList_iff.2 hx)
    obtain ⟨i, -, rfl⟩ := List.mem_map.1 hx
    exact good_cellObj _ _ _ _ _ _

theorem canon_secondLaw (sample ns ks : Array ℕ) : ∃ s, Flock.canon (Select.secondLaw sample ns ks) = .ok s :=
  FlockComplete.JsonRT.canon_ok_some _ (good_secondLaw sample ns ks)

/-! ## The units, as `checkDraw` reads them -/

theorem list_mapM_int (l : List ℕ) :
    (l.map fun x => Lean.toJson x).mapM int? = some (l.map fun a : ℕ => (a : Int)) := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    rw [List.map_cons, List.mapM_cons, int_toJson, ih]
    rfl

/-- The units' JSON reads back as the units. -/
theorem mapM_int_toJson (units : Array ℕ) :
    (units.map fun x => Lean.toJson x).mapM int? = some (units.map fun a : ℕ => (a : Int)) := by
  rw [Array.mapM_eq_mapM_toList, Array.toList_map, list_mapM_int, ← Array.toList_map]
  rfl

/-- A fold that adds one for each entry meeting `p` counts them. -/
theorem foldl_count {g : Int → ℕ → Int} (p : ℕ → Bool)
    (hg : ∀ acc u, g acc u = if p u = true then acc + 1 else acc) (l : List ℕ) (c : Int) :
    l.foldl g c = c + ((l.filter p).length : Int) := by
  induction l generalizing c with
  | nil => simp
  | cons u l ih =>
    rw [List.foldl_cons, ih, hg]
    by_cases hu : p u = true
    · rw [ite_eq_left hu, List.filter_cons_of_pos hu, List.length_cons]
      push_cast
      ring
    · rw [ite_eq_right hu, List.filter_cons_of_neg hu]

/-- A cell's two bounds, as `Draw.ranges?` reads them. -/
theorem ranges_pair_ok {a b : ℕ} (hab : a < b) :
    Draw.ranges? (Json.arr #[natsJson #[a, b]]) = some #[((a : Int), (b : Int))] := by
  unfold Draw.ranges?
  simp only [natsJson, Array.isEmpty, List.map_toArray, List.map_cons, List.map_nil, Array.mapM_eq_mapM_toList]
  simp [hab, int_toJson]

/-- **`checkDraw`'s count of the units in a cell** `[lo, hi)` is the number of the units' list in it. -/
theorem countIn_cell (units : Array ℕ) (lo hi : ℕ) (s : Json)
    (hr : Draw.ranges? (getD s "units") = some #[((lo : Int), (hi : Int))]) :
    countIn (units.map fun a : ℕ => (a : Int)) s =
      ((units.toList.filter fun u => decide (u ∈ expand [(lo, hi)])).length : Int) := by
  unfold countIn
  rw [hr, Array.foldl_map, ← Array.foldl_toList]
  refine (foldl_count (fun u => decide (u ∈ expand [(lo, hi)])) (fun acc u => ?_) _ _).trans (Int.zero_add _)
  have hiff : (((some #[((lo : Int), (hi : Int))]).getD #[]).any fun x =>
      match x with
      | (lo, hi) => decide (lo ≤ (u : Int)) && decide ((u : Int) < hi)) = decide (u ∈ expand [(lo, hi)]) := by
    rw [Bool.eq_iff_iff]
    simp only [Option.getD_some, expand, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.mem_range'_1,
      List.any_toArray, List.any_cons, List.any_nil, Bool.or_false, Bool.and_eq_true, decide_eq_true_eq]
    omega
  rw [hiff]

/-! ## A well-formed sampling design, and the sample object under it -/

/-- **The second stage's sampling design is well formed** over the interior's `|sample| · Σ_t n_t` proof blocks:
`Draw.flockLaw` reads it, and `Draw.strataMessage` finds nothing at fault. Neither reads the coins. -/
def SecondFormed (sample ns ks : Array ℕ) : Prop :=
  (∃ parsed, Draw.flockLaw (Select.secondLaw sample ns ks) (sample.size * ns.toList.sum) = .ok parsed) ∧
    Draw.strataMessage (Select.secondLaw sample ns ks) ((sample.size * ns.toList.sum : ℕ) : Int) = none

/-- A cell `(t, i)` of a second stage `secondProblem` passes has `n_t > 0` proof blocks. -/
theorem cell_lt {sample ns ks : Array ℕ} (hp : Select.secondProblem sample ns ks = none) {z : ℕ × ℕ}
    (hz : z ∈ cellPairs sample.size ns.size) :
    cellOff sample.size ns z.1 + z.2 * ns.getD z.1 0 < cellOff sample.size ns z.1 + (z.2 + 1) * ns.getD z.1 0 := by
  obtain ⟨hk1, hkn⟩ := secondProblem_counts hp (cellPairs_fst hz)
  rw [Nat.succ_mul]
  omega

/-- **The sample object is a sample of `secondLaw` over the interior** (`Properties.Spot.Drawn`), where the sampling
design is well formed and the units are strictly ascending, in the interior, and `k_t` of them in each cell of template
`t`. -/
theorem drawn_secondObj {sample ns ks : Array ℕ} (hp : Select.secondProblem sample ns ks = none)
    (hf : SecondFormed sample ns ks) {units : Array ℕ} (hasc : units.toList.Pairwise (· < ·))
    (hlt : ∀ u ∈ units.toList, u < sample.size * ns.toList.sum)
    (hcnt : ∀ z ∈ cellPairs sample.size ns.size,
      (units.toList.filter fun u => decide (u ∈ expand (cellStratum sample ns ks z).units.toList)).length =
        ks.getD z.1 0) :
    Drawn (secondObj sample ns ks units) (Lean.toJson (sample.size * ns.toList.sum))
      (Select.secondLaw sample ns ks) := by
  have hclo : ("work" == "work" && (field? (secondObj sample ns ks units) "closure").isSome) = false := by
    rw [field_obj_closure]
    rfl
  have hpop : int? (getD (secondObj sample ns ks units) "population") =
      some ((sample.size * ns.toList.sum : ℕ) : Int) := by
    rw [getD_obj_population]
    exact int_toJson _
  unfold Drawn
  dsimp only
  rw [kindOf_obj, hclo]
  refine ⟨rfl, by decide, keys_secondObj sample ns ks units, ?_, ?_, units.map fun x => Lean.toJson x,
    units.map fun a : ℕ => (a : Int), getD_obj_units sample ns ks units, mapM_int_toJson units, ?_, ?_,
    fun h => absurd h (by decide), fun h => absurd h (by decide), fun _ => ?_, fun h => absurd h (by decide)⟩
  · rw [lawOf_secondObj]
    obtain ⟨t, ht⟩ := canon_secondLaw sample ns ks
    exact jeq_self ht
  · rw [getD_obj_population]
    obtain ⟨t, ht⟩ := FlockComplete.JsonRT.canon_ok_some _
      (FlockComplete.JsonRT.good_toJson (sample.size * ns.toList.sum))
    exact jeq_self ht
  · intro i h
    have h' : i + 1 < units.size := by simpa using h
    have hlt' := List.pairwise_iff_getElem.1 hasc i (i + 1) (by simp; omega) (by simp; omega) (by omega)
    simp only [Array.getElem_map, Array.getElem_toList] at hlt' ⊢
    omega
  · intro h
    have h' : 0 < units.size := by simpa using h
    refine ⟨?_, _, hpop, ?_⟩
    · simp only [Array.getElem_map]
      omega
    · simp only [Array.getElem_map, Array.size_map]
      have hu := hlt units[units.size - 1] (Array.mem_toList_iff.2 (Array.getElem_mem (by omega)))
      omega
  · refine ⟨_, hpop, ?_, fun ss hss s hs t _ => ?_⟩
    · rw [strataMessage_secondObj]
      exact hf.2
    · rw [getD_obj_strata, getD_secondLaw_strata, strata_second] at hss
      obtain rfl := Json.arr.inj hss
      obtain ⟨z, hz, rfl⟩ := List.mem_map.1 (Array.mem_toList_iff.2 hs)
      rw [getD_cellObj_k, int_toJson, countIn_cell units _ _ _
        (by rw [getD_cellObj_units]; exact ranges_pair_ok (cell_lt hp hz))]
      exact congrArg some (congrArg (fun n : ℕ => (n : Int)) (hcnt z hz).symm)

/-- **`checkDraw` passes the sample object** under a well-formed sampling design, at units as `drawn_secondObj` asks. -/
theorem checkDraw_secondObj {sample ns ks : Array ℕ} (hp : Select.secondProblem sample ns ks = none)
    (hf : SecondFormed sample ns ks) {units : Array ℕ} (hasc : units.toList.Pairwise (· < ·))
    (hlt : ∀ u ∈ units.toList, u < sample.size * ns.toList.sum)
    (hcnt : ∀ z ∈ cellPairs sample.size ns.size,
      (units.toList.filter fun u => decide (u ∈ expand (cellStratum sample ns ks z).units.toList)).length =
        ks.getD z.1 0) :
    Draw.checkDraw (secondObj sample ns ks units) (Lean.toJson (sample.size * ns.toList.sum))
      (Select.secondLaw sample ns ks) = #[] :=
  (checkDraw_nil _ _ _).2 (drawn_secondObj hp hf hasc hlt hcnt)

/-- **`selectSecond` returns the sample its coins give**: where `secondProblem` passes and the sampling design is well
formed, a sample `Draw.drawUnits` returns under the sampling design `Draw.flockLaw` reads is the one `selectSecond`
returns, with the bytes it read. -/
theorem selectSecond_of_units {sample ns ks : Array ℕ} (hp : Select.secondProblem sample ns ks = none)
    (hf : SecondFormed sample ns ks) {parsed : Flock.Draw.Law}
    (hl : Draw.flockLaw (Select.secondLaw sample ns ks) (sample.size * ns.toList.sum) = .ok parsed)
    {src : ByteArray} {units : Array ℕ} {read : ℕ}
    (hu : Draw.drawUnits parsed (sample.size * ns.toList.sum) src = .ok (units, read)) :
    Select.selectSecond sample ns ks src = .ok (secondObj sample ns ks units, read) := by
  obtain ⟨ws, rfl, hws⟩ := flockLaw_second_strata sample ns ks _ parsed hl
  have hmap : ws.toList.map (fun s => (s.units, s.k)) =
      (cellPairs sample.size ns.size).map (cellOf sample.size ns ks) := by
    rw [hws, List.map_map]
    exact List.map_congr_left fun _ _ => rfl
  obtain ⟨hasc, hlt, hcnt⟩ := drawUnits_work_units hu (positions_second hmap)
  have hcheck := checkDraw_secondObj hp hf hasc hlt fun z hz =>
    hcnt (cellStratum sample ns ks z) (hws ▸ List.mem_map_of_mem hz)
  have hdfr : Draw.drawFirstRead (Select.secondLaw sample ns ks) (sample.size * ns.toList.sum) src =
      .ok (secondObj sample ns ks units, read) := by
    unfold Draw.drawFirstRead
    rw [hl, bind_ok, hu, bind_ok]
    rfl
  unfold Select.selectSecond
  rw [hp]
  dsimp only
  rw [hdfr, bind_ok]
  dsimp only
  rw [hcheck]
  rfl

/-- **The second stage refuses with probability at most `|sample| · Σ_t k_t · 2⁻⁶⁴`**, where `secondProblem` passes and
the sampling design is well formed, on a uniform stream of at least `secondRead` bytes: it refuses only where some
cell's first read samples nothing. -/
theorem selectSecond_refused_le_of {sample ns ks : Array ℕ} (hp : Select.secondProblem sample ns ks = none)
    (hf : SecondFormed sample ns ks) (len : ℕ) (hlen : secondRead sample.size ns ks ≤ len) :
    prCoin (fun f : Fin len → Fin 256 => ¬ ∃ draw read,
      Select.selectSecond sample ns ks (bytesOf f) = .ok (draw, read)) ≤
      ((sample.size * ks.toList.sum : ℕ) : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ := by
  obtain ⟨parsed, hl⟩ := hf.1
  refine (prCoin_mono fun f hno => ?_).trans (drawUnits_second_error_le hp hl len hlen)
  rintro ⟨units, read, hu⟩
  exact hno ⟨_, _, selectSecond_of_units hp hf hl hu⟩

end Properties.Spot.SecurityProofs
