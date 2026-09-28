import FlockSoundness.ExecRows
import Flock.HmRow

/-!
# The rows Δ writes (`HmRow.delta`)

Δ is the list of `(row, column)` pairs the statement XORs into the level-0 matrices (PROTOCOL.md
§16.1). `delta_rows`: every row it writes has one of six forms, one per loop of `HmRow.delta`:
- a slot's constant, other than the pin (rewired to read the pin);
- a row slot's first `cv` bit;
- an hm96 slot's `pad` bit;
- a unit's leaf bit;
- a unit's leaf-cut bit;
- a wire's destination bit.

The first form is a constant row, and the others are input-port rows of net slots. That no form is a unit slot's computed
row, and that a slot constant gets exactly its two pairs, follow from the layout checks, which are next (the row-placement
plan's W3).
-/

open Flock
namespace FlockSoundness

/-- A loop in `Id` whose body always continues and preserves `P` keeps `P`. -/
theorem forIn_id_inv {α β : Type} (P : β → Prop) {l : List α} {f : α → β → Id (ForInStep β)}
    (hf : ∀ a ∈ l, ∀ b, P b → ∃ b', f a b = ForInStep.yield b' ∧ P b') {b : β} (hb : P b) :
    P (forIn l b f) := by
  induction l generalizing b with
  | nil => exact hb
  | cons a l ih =>
    rw [List.forIn_cons]
    obtain ⟨b', hab, hb'⟩ := hf a (List.mem_cons_self) b hb
    rw [hab]
    exact ih (fun a ha b hb => hf a (List.mem_cons_of_mem _ ha) b hb) hb'

theorem forIn_id_inv_range {β : Type} (P : β → Prop) {lo hi : ℕ} {f : ℕ → β → Id (ForInStep β)}
    (hf : ∀ a, lo ≤ a → a < hi → ∀ b, P b → ∃ b', f a b = ForInStep.yield b' ∧ P b') {b : β} (hb : P b) :
    P (forIn [lo:hi] b f) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  refine forIn_id_inv P (fun a ha b hb => hf a ?_ ?_ b hb) hb <;>
  · simp only [Std.Legacy.Range.size, List.mem_range'_1] at ha; omega

theorem forIn_id_inv_array {α β : Type} (P : β → Prop) {xs : Array α} {f : α → β → Id (ForInStep β)}
    (hf : ∀ a ∈ xs.toList, ∀ b, P b → ∃ b', f a b = ForInStep.yield b' ∧ P b') {b : β} (hb : P b) :
    P (forIn xs b f) := by
  rw [← Array.forIn_toList]
  exact forIn_id_inv P hf hb

/-- Every pair's row satisfies `Q`. -/
def RowsIn (Q : ℕ → Prop) (d : Array (ℕ × ℕ)) : Prop := ∀ p ∈ d.toList, Q p.1

theorem RowsIn.push2 {Q : ℕ → Prop} {d : Array (ℕ × ℕ)} (h : RowsIn Q d) {i a b : ℕ} (hi : Q i) :
    RowsIn Q ((d.push (i, a)).push (i, b)) := by
  intro p hp
  simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hp
  rcases hp with (hp | rfl) | rfl
  · exact h p hp
  · exact hi
  · exact hi

theorem RowsIn.mono {Q Q' : ℕ → Prop} {d : Array (ℕ × ℕ)} (h : RowsIn Q d) (hQ : ∀ x, Q x → Q' x) :
    RowsIn Q' d := fun p hp => hQ _ (h p hp)

/-- **The rows Δ writes**, one form per loop of `HmRow.delta`: a slot's constant (not the pin), a
row slot's first `cv` bit, an hm96 slot's `pad` bit, a unit's leaf bit, a unit's leaf-cut bit, or a
wire's destination bit. -/
def DeltaRow (c : Circuit) (pin x : ℕ) : Prop :=
  (∃ r ∈ c.ranges.toList, ∃ i, r.kind = Kind.net i ∧ ∃ s < r.count,
      x = r.slot s + (c.nets.getD i default).2.constPos ∧ x ≠ pin) ∨
  (∃ g < c.g, ∃ p < (HmRow.rowSlots c.ports).2.size, ∃ t < 512,
      x = c.slotCol c.shaNet g ((HmRow.rowSlots c.ports).2[p]!)[0]!.1 +
        (portCols (c.nets.getD c.shaNet default).2.inGroups)[2 * ((HmRow.rowSlots c.ports).2[p]!)[0]!.2]!.1 + t) ∨
  (∃ g < c.g, ∃ p < (HmRow.rowSlots c.ports).2.size, ∃ t < 1024,
      x = c.slotCol c.hmNet g p + (portCols (c.nets.getD c.hmNet default).2.inGroups)[1]!.1 + t) ∨
  (∃ k < c.unitRange.count, ∃ j < (c.leavesIn.getD (k % c.upv) #[]).size, ∃ t < 16,
      x = c.unitRange.slot k + ((portCols #[c.unit.inGroups.getD 0 default]).map (·.1)).getD j 0 + t) ∨
  (∃ g < c.g, ∃ e ∈ c.leafCuts.toList, ∃ t < (portCols c.unit.inGroups)[e.2.1]!.2,
      x = c.slotCol c.unitNet g e.1 + (portCols c.unit.inGroups)[e.2.1]!.1 + t) ∨
  (∃ g < c.g, ∃ w ∈ c.wires.toList,
      ∃ t < ((portCols (c.nets.getD w.dst.1 default).2.inGroups).getD w.dst.2.2 default).2,
      x = c.slotCol w.dst.1 g w.dst.2.1 +
        ((portCols (c.nets.getD w.dst.1 default).2.inGroups).getD w.dst.2.2 default).1 + t)

theorem RowsIn_run_bind {Q : ℕ → Prop} {m : Id (Array (ℕ × ℕ))}
    {k : Array (ℕ × ℕ) → Id (Array (ℕ × ℕ) × Array (ℕ × ℕ))} (hm : RowsIn Q (Id.run m))
    (hk : ∀ d, RowsIn Q d → RowsIn Q (Id.run (k d)).1) : RowsIn Q (Id.run (m >>= k)).1 := hk _ hm

/-- **Every row Δ writes has one of its six forms** (`HmRow.delta`, both halves the same array). -/
theorem delta_rows (c : Circuit) (pin : ℕ) : RowsIn (DeltaRow c pin) (HmRow.delta c pin).1 := by
  unfold HmRow.delta
  simp only
  -- slot constants
  refine RowsIn_run_bind (forIn_id_inv_array _ (fun r hr d hd => ?_) (fun p hp => by simp at hp)) ?_
  · split
    · rename_i i hk
      refine ⟨_, rfl, forIn_id_inv_range _ (fun s _ hs d hd => ?_) hd⟩
      split
      · rename_i hne
        exact ⟨_, rfl, hd.push2 (Or.inl ⟨r, hr, i, hk, s, hs, rfl, by simpa using hne⟩)⟩
      · exact ⟨_, rfl, hd⟩
    · exact ⟨_, rfl, hd⟩
  intro d hd
  -- the rows' cv and pad bits
  refine RowsIn_run_bind (forIn_id_inv_range _ (fun g _ hg d hd => ⟨_, rfl,
    forIn_id_inv_range _ (fun p _ hp d hd => ⟨_, rfl, ?_⟩) hd⟩) hd) ?_
  · refine forIn_id_inv_range _ (fun t _ ht d hd => ⟨_, rfl, hd.push2 ?_⟩)
      (forIn_id_inv_range _ (fun t _ ht d hd => ⟨_, rfl, hd.push2 ?_⟩) hd)
    · exact Or.inr (Or.inr (Or.inl ⟨g, hg, p, hp, t, ht, rfl⟩))
    · exact Or.inr (Or.inl ⟨g, hg, p, hp, t, ht, rfl⟩)
  intro d hd
  -- the units' leaf bits
  refine RowsIn_run_bind (forIn_id_inv_range _ (fun k _ hk d hd => ⟨_, rfl,
    forIn_id_inv_range _ (fun j _ hj d hd => ⟨_, rfl,
      forIn_id_inv_range _ (fun t _ ht d hd => ⟨_, rfl, hd.push2
        (Or.inr (Or.inr (Or.inr (Or.inl ⟨k, hk, j, hj, t, ht, rfl⟩))))⟩) hd⟩) hd⟩) hd) ?_
  intro d hd
  -- the units' leaf cuts
  refine RowsIn_run_bind (forIn_id_inv_range _ (fun g _ hg d hd => ⟨_, rfl,
    forIn_id_inv_array _ (fun e he d hd => ⟨_, rfl,
      forIn_id_inv_range _ (fun t _ ht d hd => ⟨_, rfl, hd.push2
        (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨g, hg, e, he, t, ht, rfl⟩)))))⟩) hd⟩) hd⟩) hd) ?_
  intro d hd
  -- the wires
  exact forIn_id_inv_range _ (fun g _ hg d hd => ⟨_, rfl,
    forIn_id_inv_array _ (fun w hw d hd => ⟨_, rfl,
      forIn_id_inv_range _ (fun t _ ht d hd => ⟨_, rfl, hd.push2
        (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨g, hg, w, hw, t, ht, rfl⟩)))))⟩) hd⟩) hd⟩) hd

end FlockSoundness

