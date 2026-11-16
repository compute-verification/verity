module

public import Proofs.Flock.Claims.CanonRead
public import Proofs.Flock.Claims.JsonNorm
import all Init.Data.Ord.String

/-!
# A decided check that a JSON value's objects are well-formed trees

The reference reads a held run carries have to be well-formed trees (`RefReadsWf`, `HeldOf.ReadsWf`): the check that
a reference's reads are its own (`refReadsOwn`) blanks fields by inserting into the reads' tree, and an insert is only
specified on a balanced search tree. The reads come from the auditor's parameters, which Lean's `Json.parse` reads, and
no proof can look inside that parser. This file gives the fact a check instead of an assumption:

* `balancedKeys j`: every object in `j` is a tree whose nodes record their sizes and are balanced, decided by one walk
  over `j`;
* `wf_of_checks`: a value whose objects hold their keys in increasing order (`Spot.sortedKeys`, the check the
  challenger session's reader already runs) and are balanced (`balancedKeys`) is well-formed (`Claims.Json.Wf`);
* `readsChecked cs` and `wf_of_checked`: the two checks on each template's registered reads give `HeldOf.ReadsWf cs`,
  which is by definition `∀ c ∈ cs, Wf c.2.2.2`.

## What this does not show

That anything runs `readsChecked`: neither `heldOf`'s decided conditions nor the form calls it. That the values
`Json.parse` builds pass it: they do by reading the parser, which builds every object by inserts from the empty tree,
but that is not proved, for the reason above.
-/

@[expose] public section

namespace SpotFlock.ReadsJson

open Lean (Json JsonNumber)
open Claims.Json (Wf)
open Std.DTreeMap.Internal (Impl)
open Properties.Challenger.CanonRead (sortedKeys_arr sortedKeys_obj sorted_toList get?_lookup lookup_toList)

/-- Whether a tree's nodes record their sizes and are balanced at every node (`Impl.Balanced`, decided). -/
def balancedTree : Impl String (fun _ => Json) → Bool
  | .leaf => true
  | .inner sz _ _ l r =>
    balancedTree l && balancedTree r &&
      (decide (l.size + r.size ≤ 1) || (decide (l.size ≤ 3 * r.size) && decide (r.size ≤ 3 * l.size))) &&
      decide (sz = l.size + 1 + r.size)

theorem balanced_of : ∀ t : Impl String (fun _ => Json), balancedTree t = true → t.Balanced
  | .leaf, _ => .leaf
  | .inner sz k v l r, h => by
    simp only [balancedTree, Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq] at h
    obtain ⟨⟨⟨hl, hr⟩, hb⟩, hs⟩ := h
    exact .inner (balanced_of l hl) (balanced_of r hr) hb hs

/-- Every object in `j` is a balanced tree (`balancedTree`). -/
def balancedKeys : Json → Bool
  | .arr a => a.attach.all fun x => balancedKeys x.1
  | .obj kvs => balancedTree kvs.inner.inner && kvs.toArray.attach.all fun p => balancedKeys p.1.2
  | _ => true
termination_by j => sizeOf j
decreasing_by
  · exact FlockDeploy.sizeOf_lt_of_mem_arr x.2
  · exact Flock.sizeOf_lt_of_mem_toArray p.2

theorem balancedKeys_arr {a : Array Json} : balancedKeys (.arr a) = true ↔ ∀ x ∈ a, balancedKeys x = true := by
  rw [balancedKeys, Array.all_eq_true']
  exact ⟨fun h x hx => h ⟨x, hx⟩ (Array.mem_attach _ _), fun h x _ => h x.1 x.2⟩

theorem balancedKeys_obj {kvs : Std.TreeMap.Raw String Json} :
    balancedKeys (.obj kvs) = true ↔
      balancedTree kvs.inner.inner = true ∧ ∀ p ∈ kvs.toArray, balancedKeys p.2 = true := by
  rw [balancedKeys, Bool.and_eq_true, Array.all_eq_true']
  exact and_congr_right fun _ => ⟨fun h p hp => h ⟨p, hp⟩ (Array.mem_attach _ _), fun h p _ => h p.1 p.2⟩

/-- A tree whose keys increase and whose nodes are balanced is well-formed. -/
theorem wf_tree {m : Std.TreeMap.Raw String Json} (h : (m.toList.map (·.1)).Pairwise (· < ·))
    (hb : m.inner.inner.Balanced) : m.WF := by
  obtain ⟨⟨t⟩⟩ := m
  change ((Impl.Const.toList t).map (·.1)).Pairwise (· < ·) at h
  rw [Impl.Const.toList_eq_toListModel_map, List.map_map, List.pairwise_map] at h
  have hto : t.Ordered := by
    unfold Impl.Ordered
    exact h.imp fun hab => compareOfLessAndEq_eq_lt.2 hab
  exact ⟨⟨Impl.WF.wf hb hto⟩⟩

/-- **A value whose objects hold their keys in increasing order and are balanced trees is well-formed.** -/
theorem wf_of_checks : ∀ j : Json, Spot.sortedKeys j = true → balancedKeys j = true → Wf j := by
  refine FlockDeploy.json_ind (fun _ _ => .null) (fun b _ _ => .bool b) (fun n _ _ => .num n)
    (fun s _ _ => .str s) ?_ ?_
  · intro a ih hs hb
    exact .arr fun x hx => ih x hx (sortedKeys_arr.1 hs x hx) (balancedKeys_arr.1 hb x hx)
  · intro m ih hs hb
    obtain ⟨-, hsc⟩ := sortedKeys_obj.1 hs
    obtain ⟨hbt, hbc⟩ := balancedKeys_obj.1 hb
    have hsorted := sorted_toList hs
    refine .obj (wf_tree hsorted (balanced_of _ hbt)) fun k v hkv => ?_
    rw [get?_lookup hsorted] at hkv
    have hmem : (k, v) ∈ m.toList := lookup_toList hkv
    have hmem' : (k, v) ∈ m.toArray := by
      rw [← Array.mem_toList_iff, Std.TreeMap.Raw.toList_toArray]
      exact hmem
    exact ih (k, v) hmem (hsc (k, v) hmem') (hbc (k, v) hmem')

/-- **The decided check on the parameters' templates**: each template's registered reads hold their keys in
increasing order and are balanced trees. -/
def readsChecked (cs : Array (String × String × Json × Json)) : Bool :=
  cs.all fun c => Spot.sortedKeys c.2.2.2 && balancedKeys c.2.2.2

/-- **Templates that pass the check have well-formed registered reads** (`HeldOf.ReadsWf`, unfolded). -/
theorem wf_of_checked {cs : Array (String × String × Json × Json)} (h : readsChecked cs = true) :
    ∀ c ∈ cs, Wf c.2.2.2 := by
  intro c hc
  have h' := Array.all_eq_true_iff_forall_mem.1 h c hc
  simp only [Bool.and_eq_true] at h'
  exact wf_of_checks _ h'.1 h'.2

end SpotFlock.ReadsJson
