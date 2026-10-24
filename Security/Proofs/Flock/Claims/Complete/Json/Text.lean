import Flock.RecordText
import Std.Data.TreeMap.Raw.Lemmas
import Definitions.Claims.Json
import Proofs.Flock.Claims.Complete.Prover.Print

/-!
# The honest record is in its canonical text (S0)

S0 (`Flock.RecordText.check`) accepts a record that is `printT` of what it parses to. The honest prover prints its
record with `printJ`, which is `printT` (`printJ_eq`), and alike values print alike (`printJ_same`: an object's fields
print in its tree's order, which for a well-formed tree is its keys' sorted order, `Std.TreeMap.Raw.ordered_keys`). So
under `JsonSpec`, whose `parse_print` reads a printable value's text back as an alike value, S0 accepts the honest
record (`check_printJ`). No assumption beyond `JsonSpec`.
-/

namespace FlockComplete

open Lean (Json)
open Std.DTreeMap.Internal (Impl)

theorem strJ_eq (s : String) : strJ s = Flock.RecordText.strT s := rfl

mutual
theorem printJ_eq : ∀ j, printJ j = Flock.RecordText.printT j
  | .null => rfl
  | .bool _ => rfl
  | .num _ => rfl
  | .str _ => rfl
  | .arr ⟨l⟩ => by rw [printJ, Flock.RecordText.printT, itemsJ_eq l]
  | .obj ⟨⟨t⟩⟩ => by rw [printJ, Flock.RecordText.printT, fieldsJ_eq t]
theorem itemsJ_eq : ∀ l, itemsJ l = Flock.RecordText.itemsT l
  | [] => rfl
  | j :: js => by rw [itemsJ, Flock.RecordText.itemsT, printJ_eq j, itemsJ_eq js]
theorem fieldsJ_eq : ∀ t, fieldsJ t = Flock.RecordText.fieldsT t
  | .leaf => rfl
  | .inner _ k v l r => by
    rw [fieldsJ, Flock.RecordText.fieldsT, fieldsJ_eq l, printJ_eq v, fieldsJ_eq r, strJ_eq]
end

theorem itemsJ_map : ∀ l, itemsJ l = l.map printJ
  | [] => rfl
  | j :: js => by rw [itemsJ, itemsJ_map js, List.map_cons]

theorem fieldsJ_model : ∀ t : Impl String (fun _ => Json),
    fieldsJ t = t.toListModel.map fun e => strJ e.1 ++ ":" ++ printJ e.2
  | .leaf => rfl
  | .inner _ k v l r => by
    rw [fieldsJ, fieldsJ_model l, fieldsJ_model r, Impl.toListModel_inner, List.map_append, List.map_cons]

theorem fieldsJ_toList (t : Impl String (fun _ => Json)) :
    fieldsJ t = (Std.TreeMap.Raw.toList (⟨⟨t⟩⟩ : Std.TreeMap.Raw String Json)).map
      fun p => strJ p.1 ++ ":" ++ printJ p.2 := by
  rw [fieldsJ_model]
  show _ = (Impl.Const.toList t).map _
  rw [Impl.Const.toList_eq_toListModel_map, List.map_map]
  rfl

/-- Two lists of strings in strictly increasing order with the same members are one list. -/
theorem strings_eq_of_sorted : ∀ {l₁ l₂ : List String}, l₁.Pairwise (fun a b => compare a b = .lt) →
    l₂.Pairwise (fun a b => compare a b = .lt) → (∀ k, k ∈ l₁ ↔ k ∈ l₂) → l₁ = l₂
  | [], [], _, _, _ => rfl
  | [], b :: _, _, _, hm => absurd ((hm b).2 List.mem_cons_self) List.not_mem_nil
  | a :: _, [], _, _, hm => absurd ((hm a).1 List.mem_cons_self) List.not_mem_nil
  | a :: t, b :: u, h₁, h₂, hm => by
    rw [List.pairwise_cons] at h₁ h₂
    have irr : ∀ x : String, compare x x ≠ .lt := fun x h => by
      rw [(Std.ReflCmp.compare_self (cmp := compare) : compare x x = .eq)] at h
      cases h
    have hab : a = b := by
      apply Classical.byContradiction
      intro hne
      have ha : a ∈ u := by
        rcases List.mem_cons.1 ((hm a).1 List.mem_cons_self) with h | h
        · exact absurd h hne
        · exact h
      have hb : b ∈ t := by
        rcases List.mem_cons.1 ((hm b).2 List.mem_cons_self) with h | h
        · exact absurd h.symm hne
        · exact h
      have h1 := h₁.1 b hb
      have h2 := h₂.1 a ha
      rw [Std.OrientedCmp.eq_swap (cmp := compare) (a := a) (b := b), h2] at h1
      cases h1
    subst hab
    rw [strings_eq_of_sorted h₁.2 h₂.2 fun k => ?_]
    constructor
    · intro hk
      rcases List.mem_cons.1 ((hm k).1 (List.mem_cons_of_mem _ hk)) with h | h
      · subst h
        exact absurd (h₁.1 _ hk) (irr _)
      · exact h
    · intro hk
      rcases List.mem_cons.1 ((hm k).2 (List.mem_cons_of_mem _ hk)) with h | h
      · subst h
        exact absurd (h₂.1 _ hk) (irr _)
      · exact h

theorem keys_eq_of_same {m n : Std.TreeMap.Raw String Json} (hm : m.WF) (hn : n.WF)
    (hk : ∀ k, (m.get? k).isSome = (n.get? k).isSome) : m.keys = n.keys := by
  apply strings_eq_of_sorted (Std.TreeMap.Raw.ordered_keys hm) (Std.TreeMap.Raw.ordered_keys hn)
  intro k
  rw [Std.TreeMap.Raw.mem_keys hm, Std.TreeMap.Raw.mem_keys hn, Std.TreeMap.Raw.mem_iff_contains,
    Std.TreeMap.Raw.mem_iff_contains, Std.TreeMap.Raw.contains_eq_isSome_getElem? hm,
    Std.TreeMap.Raw.contains_eq_isSome_getElem? hn]
  exact iff_of_eq (congrArg (· = true) (hk k))

/-- **Alike values print alike**: `printJ` is one text on `Same` values, objects' keys printed in their sorted order. -/
theorem printJ_same {a b : Json} (h : Claims.Json.Same a b) : printJ a = printJ b := by
  induction h with
  | null => rfl
  | bool => rfl
  | num => rfl
  | str => rfl
  | @arr a b hs he ih =>
    rcases a with ⟨la⟩
    rcases b with ⟨lb⟩
    simp only [printJ, itemsJ_map]
    have : la.map printJ = lb.map printJ := by
      apply List.ext_getElem (by simpa using hs)
      intro i h1 h2
      simp only [List.getElem_map]
      exact ih i (by simpa using h1) (by simpa using h2)
    rw [this]
  | @obj m n hm hn hk hv ih =>
    rcases m with ⟨⟨tm⟩⟩
    rcases n with ⟨⟨tn⟩⟩
    simp only [printJ, fieldsJ_toList]
    have hkeys := keys_eq_of_same hm hn hk
    rw [← Std.TreeMap.Raw.map_fst_toList_eq_keys, ← Std.TreeMap.Raw.map_fst_toList_eq_keys] at hkeys
    have hlen := congrArg List.length hkeys
    simp only [List.length_map] at hlen
    refine congrArg (fun l => "{" ++ ",".intercalate l ++ "}") (List.ext_getElem (by simpa using hlen) ?_)
    intro i h1 h2
    simp only [List.length_map] at h1 h2
    simp only [List.getElem_map]
    have hfst : (Std.TreeMap.Raw.toList (⟨⟨tm⟩⟩ : Std.TreeMap.Raw String Json))[i].1 =
        (Std.TreeMap.Raw.toList (⟨⟨tn⟩⟩ : Std.TreeMap.Raw String Json))[i].1 := by
      have := congrArg (·[i]?) hkeys
      simp only [List.getElem?_map, List.getElem?_eq_getElem h1, List.getElem?_eq_getElem h2, Option.map_some,
        Option.some.injEq] at this
      exact this
    have key : ∀ pa pb : String × Json, pa ∈ Std.TreeMap.Raw.toList (⟨⟨tm⟩⟩ : Std.TreeMap.Raw String Json) →
        pb ∈ Std.TreeMap.Raw.toList (⟨⟨tn⟩⟩ : Std.TreeMap.Raw String Json) → pa.1 = pb.1 →
        strJ pa.1 ++ ":" ++ printJ pa.2 = strJ pb.1 ++ ":" ++ printJ pb.2 := by
      rintro ⟨ka, va⟩ ⟨kb, vb⟩ ha hb (rfl : ka = kb)
      rw [ih _ _ _ ((Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm).1 ha)
        ((Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hn).1 hb)]
    exact key _ _ (List.getElem_mem h1) (List.getElem_mem h2) hfst

/-- **S0 accepts the honest record**: the text `printJ` prints of a printable value. -/
theorem check_printJ {canon : Json → Except String String} (S : Claims.JsonSpec printJ canon) {j : Json}
    (hg : Claims.Json.Good j) : Flock.RecordText.check (printJ j) = .ok () := by
  obtain ⟨j', hp, hs⟩ := S.parse_print j hg
  exact Flock.RecordText.check_of hp (by rw [← printJ_eq, printJ_same hs])

end FlockComplete
