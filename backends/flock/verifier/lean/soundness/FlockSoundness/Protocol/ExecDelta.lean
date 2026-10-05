import Flock.HmRow
import FlockSoundness.Protocol.ExecRows

/-!
# The rows Δ writes (`HmRow.delta`)

Δ is the list of `(row, column)` pairs the statement XORs into the level-0 matrices (PROTOCOL.md
§16.1). `delta_rows`: every row it writes has one of six forms, one per loop of `HmRow.delta`:
- a slot's constant, other than the pin (rewired to read the pin);
- a bit of a row's chain (`ShaRowBit`): a `cv` or message bit of one of its `sha512x3` slots (its first `cv`, and its
  message's constant bits past the row's length);
- a bit of a row's hm96 slot (`HmRowBit`): its `cv` (a row of no compression's midstate) or `pad`;
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

/-- **A bit of a row's chain**: in VU `g`, input `i` (`0` its `cv`, 512 bits; `1` its message, 1024) of row `p`'s `k`-th
`sha512x3` slot. -/
def ShaRowBit (c : Circuit) (x : ℕ) : Prop :=
  ∃ g < c.g, ∃ p < (HmRow.rowSlots c.ports).2.size, ∃ k < ((HmRow.rowSlots c.ports).2[p]!).size, ∃ i < 2,
    ∃ t < 512 * (i + 1), x = c.slotCol c.shaNet g ((HmRow.rowSlots c.ports).2[p]!)[k]!.1 +
      (portCols (c.nets.getD c.shaNet default).2.inGroups)[2 * ((HmRow.rowSlots c.ports).2[p]!)[k]!.2 + i]!.1 + t

/-- **A bit of a row's hm96 slot**: in VU `g`, input `i` (`0` its `cv`, 512 bits; `1` its `pad`, 1024) of row `p`'s
`hm96` slot. -/
def HmRowBit (c : Circuit) (x : ℕ) : Prop :=
  ∃ g < c.g, ∃ p < (HmRow.rowSlots c.ports).2.size, ∃ i < 2, ∃ t < 512 * (i + 1),
    x = c.slotCol c.hmNet g p + (portCols (c.nets.getD c.hmNet default).2.inGroups)[i]!.1 + t

end FlockSoundness
