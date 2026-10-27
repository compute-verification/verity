import Proofs.Properties.Verity.SampledCanonDrawAt
import Std.Data.TreeMap.Raw.WF

/-!
# The canonical subset draw file decodes to its draw

`DrawFileZK` says a draw file that `UnitDraw.ofJson` decodes decodes to the verifier's draw. Its other direction is that
the canonical file `UnitDraw.toJson` prints for the verifier's subset draw of `S`, `drawOf (.subset kd) n S` with
`kd = S.card`, is one `ofJson` accepts, and that it gives the draw back (`Canon.ofJson_drawOf_subset`): its keys are
exactly the subset law's, its units ascending, distinct units of the population, `kd` of them.
-/

namespace Verity.Canon

open Lean (Json)

theorem objVal_mkObj {l : List (String × Json)} (hd : l.Pairwise (fun a b => ¬ compare a.1 b.1 = .eq))
    {k : String} {v : Json} (hm : (k, v) ∈ l) : (Json.mkObj l).getObjVal? k = .ok v := by
  simp only [Json.mkObj, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?]
  rw [Std.TreeMap.Raw.getElem?_ofList_of_mem (Std.ReflCmp.compare_self) hd hm]
  rfl

theorem objVal_mkObj_not {l : List (String × Json)} {k : String} (hk : (l.map Prod.fst).contains k = false) :
    ∃ e, (Json.mkObj l).getObjVal? k = .error e := by
  simp only [Json.mkObj, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?]
  rw [Std.TreeMap.Raw.getElem?_ofList_of_contains_eq_false hk]
  exact ⟨_, rfl⟩

theorem asNat_toJson (x : Nat) (c : String) : _root_.Flock.J.asNat (Lean.toJson x) c = .ok x := rfl

theorem asStr_str (t c : String) : _root_.Flock.J.asStr (Json.str t) c = .ok t := rfl

theorem asArr_arr (a : Array Json) (c : String) : _root_.Flock.J.asArr (Json.arr a) c = .ok a := rfl

theorem ok_bind {ε α β : Type} (a : α) (f : α → Except ε β) : (Except.ok a >>= f) = f a := rfl

theorem getObj?_mkObj (l : List (String × Json)) :
    (Json.mkObj l).getObj? = .ok (Std.TreeMap.Raw.ofList l compare) := rfl

theorem mapM_asNat_toJson (c : String) : ∀ l : List Nat,
    (l.map fun x => Lean.toJson x).mapM (_root_.Flock.J.asNat · c) = .ok l
  | [] => rfl
  | a :: l => by
    simp only [List.map_cons, List.mapM_cons, mapM_asNat_toJson c l]
    rfl

theorem arr_mapM_asNat_toJson (c : String) (us : Array Nat) :
    (us.map fun x => Lean.toJson x).mapM (_root_.Flock.J.asNat · c) = .ok us := by
  obtain ⟨l⟩ := us
  rw [List.map_toArray, List.mapM_toArray, mapM_asNat_toJson c l]
  rfl

/-- **A subset draw file `UnitDraw.toJson` prints decodes to its draw**: `UnitDraw.ofJson` accepts the canonical file of
a subset draw of `k ≤ pop` ascending, distinct units of the population, and gives that draw back. -/
theorem ofJson_toJson_subset_ok (k pop : Nat) (us : Array Nat) (hsz : us.size = k) (hk : k ≤ pop)
    (hasc : (List.range us.size).all (fun i => us[i]! < pop && (i == 0 || us[i - 1]! < us[i]!)) = true) :
    _root_.Flock.Draw.UnitDraw.ofJson
        ({ law := .subset k, population := pop, units := us } : _root_.Flock.Draw.UnitDraw).toJson =
      .ok { law := .subset k, population := pop, units := us } := by
  have hd : [("law", Json.str "subset"), ("population", Lean.toJson pop), ("k", Lean.toJson k),
      ("units", Json.arr (us.map fun x => Lean.toJson x))].Pairwise (fun a b => ¬ compare a.1 b.1 = .eq) := by
    simp
  obtain ⟨e, he⟩ := objVal_mkObj_not (k := "closure") (l := [("law", Json.str "subset"), ("population", Lean.toJson pop),
    ("k", Lean.toJson k), ("units", Json.arr (us.map fun x => Lean.toJson x))]) (by simp)
  have hsize : (Std.TreeMap.Raw.ofList [("law", Json.str "subset"), ("population", Lean.toJson pop),
      ("k", Lean.toJson k), ("units", Json.arr (us.map fun x => Lean.toJson x))] compare).toArray.size = 4 := by
    rw [Std.TreeMap.Raw.size_toArray Std.TreeMap.Raw.WF.ofList, Std.TreeMap.Raw.size_ofList hd]
    rfl
  have hlaw := objVal_mkObj hd (k := "law") (v := Json.str "subset") (by simp)
  have hpop := objVal_mkObj hd (k := "population") (v := Lean.toJson pop) (by simp)
  have hkk := objVal_mkObj hd (k := "k") (v := Lean.toJson k) (by simp)
  have hun := objVal_mkObj hd (k := "units") (v := Json.arr (us.map fun x => Lean.toJson x)) (by simp)
  have htj : ({ law := .subset k, population := pop, units := us } : _root_.Flock.Draw.UnitDraw).toJson =
      Json.mkObj [("law", Json.str "subset"), ("population", Lean.toJson pop), ("k", Lean.toJson k),
        ("units", Json.arr (us.map fun x => Lean.toJson x))] := rfl
  rw [htj]
  subst hsz
  have hgt : decide (us.size > pop) = false := decide_eq_false (by omega)
  unfold _root_.Flock.Draw.UnitDraw.ofJson
  simp only [_root_.Flock.J.nat, _root_.Flock.J.arr, _root_.Flock.J.str, _root_.Flock.J.get, hlaw, hpop, hkk, hun, he,
    getObj?_mkObj, ok_bind, asNat_toJson, asStr_str, asArr_arr, arr_mapM_asNat_toJson, hsize, hasc, bne_self_eq_false,
    Bool.false_eq_true, ↓reduceIte, Bool.not_true, hgt, Bool.false_or, pure_bind]
  rfl

/-- **`drawOf`'s units are ascending units of the population**, as `UnitDraw.ofJson` checks them. -/
theorem drawOf_units_asc {n : ℕ} (S : Finset (Fin n)) :
    (List.range ((S.image Fin.val).sort (· ≤ ·)).toArray.size).all (fun i =>
      ((S.image Fin.val).sort (· ≤ ·)).toArray[i]! < n &&
        (i == 0 || ((S.image Fin.val).sort (· ≤ ·)).toArray[i - 1]! < ((S.image Fin.val).sort (· ≤ ·)).toArray[i]!)) =
      true := by
  have hs : ((S.image Fin.val).sort (· ≤ ·)).Pairwise (· < ·) :=
    ((Finset.pairwise_sort _ _).and (Finset.sort_nodup _ _)).imp fun h => lt_of_le_of_ne h.1 h.2
  rw [List.all_eq_true]
  intro i hi
  rw [List.mem_range, List.size_toArray] at hi
  rw [getElem!_pos _ i (by simpa using hi), List.getElem_toArray]
  have hm : ((S.image Fin.val).sort (· ≤ ·))[i] ∈ S.image Fin.val :=
    (Finset.mem_sort _).1 (List.getElem_mem hi)
  obtain ⟨u, -, hu⟩ := Finset.mem_image.1 hm
  have hlt : ((S.image Fin.val).sort (· ≤ ·))[i] < n := hu ▸ u.isLt
  by_cases h0 : i = 0
  · subst h0; simp [hlt]
  · rw [getElem!_pos _ (i - 1) (by rw [List.size_toArray]; omega), List.getElem_toArray]
    have := List.pairwise_iff_getElem.1 hs (i - 1) i (by omega) hi (by omega)
    simp [hlt, this]

/-- **The canonical subset draw file decodes to its draw**: `UnitDraw.ofJson` of what `UnitDraw.toJson` prints for the
verifier's draw of `S` under `subset kd`, `kd = S.card`, is that draw. -/
theorem ofJson_drawOf_subset {n kd : ℕ} (S : Finset (Fin n)) (hk : S.card = kd) :
    _root_.Flock.Draw.UnitDraw.ofJson (FlockSoundness.Discharge.ZkLink.drawOf (.subset kd) n S).toJson =
      .ok (FlockSoundness.Discharge.ZkLink.drawOf (.subset kd) n S) := by
  subst hk
  unfold FlockSoundness.Discharge.ZkLink.drawOf
  exact ofJson_toJson_subset_ok S.card n ((S.image Fin.val).sort (· ≤ ·)).toArray
    (by rw [List.size_toArray, Finset.length_sort, Finset.card_image_of_injective _ Fin.val_injective])
    ((Finset.card_le_univ S).trans_eq (Fintype.card_fin n)) (drawOf_units_asc S)

end Verity.Canon
