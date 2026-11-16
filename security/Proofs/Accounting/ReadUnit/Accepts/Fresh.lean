module

public import Flock.ReadNet.Template
public import Proofs.Flock.Soundness.Discharge.Hm.Fresh

/-!
# Each part's run writes fresh variables

Every generator the read's parts run adds rows only as `fresh` does (`FlockSoundness.Discharge.Hm.Fresh`): SHA-512's
compressions under `shaW`, the row segment and its outer hash, the hm96 leaf, the frame's leaf and node hashes, the
digest comparison and the select. So each part's run, from its inputs on, writes each row's variable past its inputs and
before its next (`runOk_parts`), at every `V` and `H`.
-/

@[expose] public section

namespace Accounting.ReadUnit.Accepts

open FlockSoundness.Discharge.Hm Flock.ReadNet
open Flock.HmNets hiding and

theorem fresh_shaW : ∀ (n : ℕ) (cv msg : List Word), Fresh (shaW n cv msg)
  | 0, _, _ => Fresh.pure' _
  | n + 1, _, _ => Fresh.bind' (fresh_compress _ _) fun st => fresh_shaW n st _

theorem fresh_segment (m : ℕ) (cv : List Word) (bits : List Lin) : Fresh (segment m cv bits) := fresh_shaW _ _ _

theorem fresh_outer (l : ℕ) (d : List Word) : Fresh (outer l d) := fresh_shaW _ _ _

theorem fresh_leaf (x y : List Word) : Fresh (leaf x y) :=
  Fresh.bind' (Fresh.mapM' _ (fun _ => fresh_commitW _) _) fun _ => Fresh.bind' (fresh_shaW _ _ _) fun _ =>
    Fresh.bind' (fresh_shaW _ _ _) fun _ => Fresh.pure' _

namespace Frame

open Flock.ReadNet.Frame

theorem fresh_orL (a b : Lin) : Fresh (orL a b) := Fresh.bind' (fresh_and a b) fun _ => Fresh.pure' _

theorem fresh_flagsGo (bits : List Lin) : ∀ (t : ℕ) (acc : Option Lin) (out : List Lin), Fresh (flagsGo bits t acc out)
  | 0, _, _ => Fresh.pure' _
  | t + 1, none, _ => Fresh.bind' (Fresh.pure' _) fun _ => fresh_flagsGo bits t _ _
  | t + 1, some _, _ => Fresh.bind' (fresh_orL _ _) fun _ => fresh_flagsGo bits t _ _

theorem fresh_flags (bits : List Lin) (n : ℕ) : Fresh (flags bits n) := fresh_flagsGo _ _ _ _

theorem fresh_selectGo : ∀ (m : Lin) (zs : List Lin) (ps : List (Lin × Lin)), Fresh (selectGo m zs ps)
  | m, z :: zs, (a, b) :: ps => by
      unfold selectGo
      dsimp only
      split
      · exact Fresh.bind' (Fresh.pure' _) fun _ => fresh_selectGo _ zs ps
      · exact Fresh.bind' (fresh_and _ _) fun _ => Fresh.bind' (Fresh.pure' _) fun _ => fresh_selectGo _ zs ps
  | m, [], _ => by unfold selectGo; exact Fresh.pure' _
  | m, _ :: _, [] => by unfold selectGo; exact Fresh.pure' _

theorem fresh_selectBit (fl col : List Lin) : Fresh (selectBit fl col) := fresh_selectGo _ _ _

theorem fresh_select (fl : List Lin) (layouts : List (List Word)) : Fresh (select fl layouts) :=
  Fresh.mapM' _ (fun _ => Fresh.mapM' _ (fun _ => fresh_selectBit _ _) _) _

theorem fresh_hashUint (tag : List UInt8) (head : List (List Byte)) (x : List Lin) (n : ℕ)
    (tail : List Byte → List (List Byte)) : Fresh (hashUint tag head x n tail) :=
  Fresh.bind' (fresh_flags _ _) fun _ => Fresh.bind' (fresh_select _ _) fun _ => fresh_shaW _ _ _

theorem fresh_leafHash (schema : String) (n : ℕ) (domain value : List Word) (x : List Lin) :
    Fresh (leafHash schema n domain value x) := fresh_hashUint _ _ _ _ _

theorem fresh_nodeHash (n : ℕ) (domain : List Word) (level : Byte) (index : List Lin) (d : Lin) (cur sib : List Word) :
    Fresh (nodeHash n domain level index d cur sib) :=
  Fresh.bind' (Fresh.mapM' _ (fun _ => Fresh.mapM' _ (fun _ => fresh_and _ _) _) _) fun _ => fresh_hashUint _ _ _ _ _

theorem fresh_andPairs : ∀ l : List Lin, Fresh (andPairs l)
  | a :: b :: rest => by
      unfold andPairs
      exact Fresh.bind' (fresh_and a b) fun _ => Fresh.bind' (fresh_andPairs rest) fun _ => Fresh.pure' _
  | [] => by unfold andPairs; exact Fresh.pure' _
  | [_] => by unfold andPairs; exact Fresh.pure' _

theorem fresh_andTree : ∀ (k : ℕ) (l : List Lin), Fresh (andTree k l)
  | _, [] => by unfold andTree; exact Fresh.pure' _
  | _, [_] => by unfold andTree; exact Fresh.pure' _
  | 0, _ :: _ :: _ => by unfold andTree; exact Fresh.pure' _
  | k + 1, a :: b :: l => by
      unfold andTree
      exact Fresh.bind' (fresh_andPairs _) fun _ => fresh_andTree k _

theorem fresh_eqDigest (x y : List Word) : Fresh (eqDigest x y) := fresh_andTree _ _

end Frame

theorem fresh_belowGo (bound : ℕ) : ∀ (bs : List Lin) (k : ℕ) (lt : Lin), Fresh (belowGo bound bs k lt)
  | [], _, _ => by
      unfold belowGo
      exact Fresh.pure' _
  | _ :: bs, _, _ => by
      unfold belowGo
      dsimp only
      split
      · exact Fresh.bind' (fresh_and _ _) fun _ => Fresh.bind' (Fresh.pure' _) fun _ => fresh_belowGo bound bs _ _
      · exact Fresh.bind' (fresh_and _ _) fun _ => fresh_belowGo bound bs _ _

theorem fresh_select (V : ℕ) (tok : List Lin) (opened : Lin) (row : List Lin) : Fresh (select V tok opened row) :=
  Fresh.bind' (fresh_belowGo _ _ _ _) fun _ =>
    Fresh.bind' (Fresh.mapM' _ (fun _ => Fresh.bind' (fresh_and _ _) fun _ => Fresh.pure' _) _) fun _ =>
      Fresh.bind' (fresh_and _ _) fun _ => Fresh.pure' _

/-- Each of the read's parts runs a program that adds rows only as `fresh` does. -/
theorem fresh_parts (V H : ℕ) : ∀ p ∈ parts V H, Fresh p.prog := by
  intro p hp
  simp only [parts, List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact Fresh.bind' (fresh_segment _ _ _) fun _ => Fresh.pure' _
  · exact Fresh.bind' (fresh_segment _ _ _) fun _ => Fresh.pure' _
  · exact Fresh.bind' (fresh_outer _ _) fun _ => Fresh.pure' _
  · exact Fresh.bind' (fresh_leaf _ _) fun _ => Fresh.pure' _
  · exact Fresh.bind' (Frame.fresh_leafHash _ _ _ _ _) fun _ => Fresh.pure' _
  · exact Fresh.bind' (Frame.fresh_nodeHash _ _ _ _ _ _ _) fun _ => Fresh.pure' _
  · exact Fresh.bind' (Frame.fresh_nodeHash _ _ _ _ _ _ _) fun _ => Fresh.pure' _
  · exact Fresh.bind' (Frame.fresh_nodeHash _ _ _ _ _ _ _) fun _ => Fresh.pure' _
  · exact Fresh.bind' (Frame.fresh_nodeHash _ _ _ _ _ _ _) fun _ => Fresh.pure' _
  · exact Fresh.bind' (Frame.fresh_eqDigest _ _) fun _ => Fresh.pure' _
  · exact Fresh.bind' (fresh_select _ _ _ _) fun _ => Fresh.pure' _

/-- **Each part's run writes fresh variables**, at every `V` and `H`. -/
theorem runOk_parts (V H : ℕ) : (parts V H).all (fun p => runOk p.ins p.run) = true := by
  rw [Array.all_eq_true_iff_forall_mem]
  intro p hp
  obtain ⟨hn, hr⟩ := fresh_parts V H p hp p.ins ⟨p.ins, []⟩ ⟨rfl, fun k r h => by simp at h⟩
  unfold runOk Part.run
  rw [List.all_eq_true]
  intro w hw
  obtain ⟨k, hk, hkw⟩ := List.getElem_of_mem (List.mem_reverse.2 hw)
  have hout := hr k w (by rw [List.getElem?_eq_getElem hk, hkw])
  rw [List.length_reverse] at hk
  have hout' : (w.out : ℕ) = p.ins + k := hout
  have hn' : (p.prog.run ⟨p.ins, []⟩).2.next = p.ins + (p.prog.run ⟨p.ins, []⟩).2.rows.length := hn
  simp only [Bool.and_eq_true, decide_eq_true_eq]
  show p.ins ≤ (w.out : ℕ) ∧ (w.out : ℕ) < (p.prog.run ⟨p.ins, []⟩).2.next
  rw [hout', hn']
  exact ⟨Nat.le_add_right _ _, Nat.add_lt_add_left hk _⟩

end Accounting.ReadUnit.Accepts
