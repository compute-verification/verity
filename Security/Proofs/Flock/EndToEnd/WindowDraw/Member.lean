import Proofs.Flock.EndToEnd.WindowDraw.Statement

/-!
# A member's share of a window's draw, and the service's split

The member's units embed in the window (`Member.emb`), so its share of a draw is the draw's units in its range, in its
own indices (`map_share`), and the share of a draw outside the member with the member's own units `S` placed in it is
`S` (`share_union_unshift`). The service's split of a draw file's unit list (`Member.localUnits`) is the share of the
draw the list names (`mem_share_filter`), with as many units as the share for a list without repeats
(`length_localUnits`), so the session file is the share's (`sessionFile_eq`).
-/

namespace FlockSoundness.Audit.Window

open Finset

namespace Member

variable {N : ℕ} (mb : Member)

/-- The member's own units, embedded in the window. -/
def emb (h : mb.base + mb.size ≤ N) : Fin mb.size ↪ Fin N where
  toFun v := ⟨mb.base + v.val, by have := v.isLt; omega⟩
  inj' a b hab := Fin.ext (by simpa using congrArg Fin.val hab)

variable {mb}

theorem emb_val (h : mb.base + mb.size ≤ N) (v : Fin mb.size) : (mb.emb h v).val = mb.base + v.val := rfl

theorem mem_share {D : Finset (Fin N)} {v : Fin mb.size} :
    v ∈ mb.share D ↔ ∃ u ∈ D, u.val = mb.base + v.val := by
  simp [share]

theorem mem_share_emb (h : mb.base + mb.size ≤ N) {D : Finset (Fin N)} {v : Fin mb.size} :
    v ∈ mb.share D ↔ mb.emb h v ∈ D := by
  rw [mem_share]
  constructor
  · rintro ⟨u, hu, huv⟩
    rwa [show u = mb.emb h v from Fin.ext huv] at hu
  · intro hv
    exact ⟨_, hv, rfl⟩

theorem unshift_eq_map (h : mb.base + mb.size ≤ N) (S : Finset (Fin mb.size)) :
    (mb.unshift S : Finset (Fin N)) = S.map (mb.emb h) := by
  ext u
  simp only [unshift, mem_filter, mem_univ, true_and, mem_map]
  constructor
  · rintro ⟨v, hv, huv⟩
    exact ⟨v, hv, Fin.ext huv.symm⟩
  · rintro ⟨v, hv, rfl⟩
    exact ⟨v, hv, rfl⟩

theorem range_eq_map (h : mb.base + mb.size ≤ N) : (mb.range : Finset (Fin N)) = univ.map (mb.emb h) := by
  ext u
  simp only [range, mem_filter, mem_univ, true_and, mem_map]
  constructor
  · rintro ⟨h1, h2⟩
    exact ⟨⟨u.val - mb.base, by omega⟩, Fin.ext (show mb.base + (u.val - mb.base) = u.val by omega)⟩
  · rintro ⟨v, rfl⟩
    rw [emb_val]
    have := v.isLt
    omega

theorem card_range (h : mb.base + mb.size ≤ N) : (mb.range : Finset (Fin N)).card = mb.size := by
  rw [range_eq_map h, card_map, card_univ, Fintype.card_fin]

theorem map_share (h : mb.base + mb.size ≤ N) (D : Finset (Fin N)) :
    (mb.share D).map (mb.emb h) = D ∩ mb.range := by
  ext u
  simp only [mem_map, mem_inter, range_eq_map h, mem_univ, true_and]
  constructor
  · rintro ⟨v, hv, rfl⟩
    exact ⟨(mem_share_emb h).1 hv, v, rfl⟩
  · rintro ⟨hu, v, rfl⟩
    exact ⟨v, (mem_share_emb h).2 hu, rfl⟩

theorem card_share (h : mb.base + mb.size ≤ N) (D : Finset (Fin N)) :
    (mb.share D).card = (D ∩ mb.range).card := by
  rw [← map_share h, card_map]

theorem unshift_subset_range (h : mb.base + mb.size ≤ N) (S : Finset (Fin mb.size)) :
    (mb.unshift S : Finset (Fin N)) ⊆ mb.range := by
  rw [unshift_eq_map h, range_eq_map h]
  exact map_subset_map.2 (subset_univ S)

theorem card_unshift (h : mb.base + mb.size ≤ N) (S : Finset (Fin mb.size)) :
    (mb.unshift S : Finset (Fin N)).card = S.card := by
  rw [unshift_eq_map h, card_map]

/-- The member's share of a draw outside it with its own units `S` placed in it is `S`. -/
theorem share_union_unshift (h : mb.base + mb.size ≤ N) (ctx : Finset (Fin N)) (S : Finset (Fin mb.size))
    (hctx : Disjoint ctx mb.range) : mb.share (ctx ∪ mb.unshift S) = S := by
  ext v
  have hr : mb.emb h v ∈ (mb.range : Finset (Fin N)) := by
    rw [range_eq_map h]
    exact mem_map_of_mem _ (mem_univ v)
  rw [mem_share_emb h, mem_union, unshift_eq_map h, mem_map']
  exact ⟨fun h' => h'.resolve_left (disjoint_right.1 hctx hr), Or.inr⟩

theorem mem_localUnits (units : List ℕ) (x : ℕ) :
    x ∈ mb.localUnits units ↔ mb.base + x ∈ units ∧ x < mb.size := by
  simp only [localUnits, List.mem_insertionSort, List.mem_map, List.mem_filter, Bool.and_eq_true,
    decide_eq_true_eq]
  constructor
  · rintro ⟨u, ⟨hu, h1, h2⟩, rfl⟩
    refine ⟨?_, by omega⟩
    rwa [Nat.add_sub_cancel' h1]
  · rintro ⟨hx, hlt⟩
    exact ⟨mb.base + x, ⟨hx, by omega, by omega⟩, by omega⟩

/-- **The service's split is the share.** A unit is in the member's local units of a draw file's unit list exactly
when the member's share of the draw that list names holds it. -/
theorem mem_share_filter (h : mb.base + mb.size ≤ N) (units : List ℕ) (v : Fin mb.size) :
    v ∈ mb.share (univ.filter fun u : Fin N => u.val ∈ units) ↔ v.val ∈ mb.localUnits units := by
  rw [mem_share_emb h, mem_filter, mem_localUnits, emb_val]
  exact ⟨fun h' => ⟨h'.2, v.isLt⟩, fun h' => ⟨mem_univ _, h'.1⟩⟩

theorem nodup_localUnits (units : List ℕ) (hnd : units.Nodup) : (mb.localUnits units).Nodup := by
  unfold localUnits
  refine (List.perm_insertionSort _ _).nodup_iff.2 ((hnd.filter _).map_on fun x hx y hy hxy => ?_)
  simp only [List.mem_filter, Bool.and_eq_true, decide_eq_true_eq] at hx hy
  omega

/-- **The session file's `k` is the share's size**: for a unit list without repeats, the member's local units number
its share's units. -/
theorem length_localUnits (h : mb.base + mb.size ≤ N) (units : List ℕ) (hnd : units.Nodup) :
    (mb.localUnits units).length = (mb.share (univ.filter fun u : Fin N => u.val ∈ units)).card := by
  rw [← List.toFinset_card_of_nodup (nodup_localUnits units hnd),
    ← card_map ⟨Fin.val, Fin.val_injective⟩]
  congr 1
  ext x
  simp only [List.mem_toFinset, mem_map, Function.Embedding.coeFn_mk]
  constructor
  · intro hx
    exact ⟨⟨x, ((mem_localUnits units x).1 hx).2⟩, (mem_share_filter h units _).2 hx, rfl⟩
  · rintro ⟨v, hv, rfl⟩
    exact (mem_share_filter h units v).1 hv

theorem sessionFile_eq (h : mb.base + mb.size ≤ N) (W : DrawFile) (hnd : W.units.Nodup) :
    (mb.sessionFile W = none ↔ mb.share (univ.filter fun u : Fin N => u.val ∈ W.units) = ∅) ∧
    ∀ F, mb.sessionFile W = some F → F.law = "subset" ∧ F.population = mb.size ∧
      F.k = (mb.share (univ.filter fun u : Fin N => u.val ∈ W.units)).card ∧ F.units = mb.localUnits W.units := by
  have hnil : mb.localUnits W.units = [] ↔ mb.share (univ.filter fun u : Fin N => u.val ∈ W.units) = ∅ := by
    rw [← List.length_eq_zero_iff, length_localUnits h W.units hnd, card_eq_zero]
  refine ⟨?_, fun F hF => ?_⟩
  · unfold sessionFile
    split_ifs with hl
    · exact ⟨fun _ => hnil.1 hl, fun _ => rfl⟩
    · exact ⟨fun h' => absurd h' (by simp), fun h' => absurd (hnil.2 h') hl⟩
  · unfold sessionFile at hF
    split_ifs at hF with hl
    cases hF
    exact ⟨rfl, rfl, length_localUnits h W.units hnd, rfl⟩

end Member

end FlockSoundness.Audit.Window
