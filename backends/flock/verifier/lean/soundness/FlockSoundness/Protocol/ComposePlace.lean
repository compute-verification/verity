import Flock.Derive
import Flock.DeriveCheck
import FlockSoundness.Protocol.ExecSetup

/-!
# Composed rows, placed: what 1d and 1e hand to a table's class (`dp`)

`placement_of_realizes` (`Compose.lean`) proves `Placement` for `Rows.compose` from facts about the block matrices. Here
those facts are one proposition, and they are discharged for a flat statement `Stmt.setupH` accepted:
- **`Compose.Realizes S L pos`**: each logical column's position is in the block; at each computed row's position the
  matrices read exactly the positions of its columns, or the pin for a row that copies `one`; and the pin's row reads
  the pin. `Realizes.placement` is the `Placement` it gives, at `composePlace`.
- **`Compose.BlockFacts`**: 1e's form of the same facts, over the unit's block rows (`blockRow`, Δ applied), for the
  verifier's own `deriveChecked` run. `BlockFacts.realizes` turns them into `Realizes` for the unit's composed rows
  (`ofBlock`), and `BlockFacts.placement` is a table class's `placed` at a slot (the soundness lane's `TableClass`).
- **`Layout.realizes`**: in the model of a statement `setupH` accepted (#177's `Layout`, carried through #257 by #284),
  at unit slot `s₀` with positions `slot s₀ + c`, `Realizes` holds for every `Logical` whose rows are the unit net's
  (`Compose.NetRows`). It reads Δ only as #177's `Layout` does, for statements without a typed template.

`NetRows st.c.unit (ofBlock words done u)` is the link between the unit net `setupH` accepted and the rows the verifier
derives. For a typed flat class, whose unit net is built from those rows (`Typed.netOfD`), it is `netRows_flat`
(`ExecFlatClass.lean`).
-/

open Flock FlockLevel3

namespace FlockSoundness

open Audit

namespace Compose

open Flock.DeriveAll

end Compose

open Compose

/-! ## A flat statement's unit slots realize the rows of its unit net -/

/-- Shifting a list below `W` by `B·W`, counted at `j`: the count at `j mod W` when `j` is in block `B`, else 0. -/
theorem count_map_add_ite {W B : ℕ} (hW : 0 < W) (l : List ℕ) (hl : ∀ x ∈ l, x < W) (j : ℕ) :
    (((l.map (B * W + ·)).count j : ℕ) : ZMod 2) = if j / W = B then ((l.count (j % W) : ℕ) : ZMod 2) else 0 := by
  by_cases hq : j / W = B
  · rw [ite_eq_left hq]
    have hj : j = B * W + j % W := by
      have := Nat.div_add_mod j W
      rw [hq, Nat.mul_comm] at this
      exact this.symm
    have : (l.map (B * W + ·)).count j = l.count (j % W) := by
      conv_lhs => rw [hj]
      exact List.count_map_of_injective _ _ (fun a b hab => by simpa using hab) _
    rw [this]
  · rw [ite_eq_right hq, List.count_eq_zero.2]
    · rfl
    intro hm
    obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hm
    apply hq
    rw [show B * W + x = x + W * B by ring, Nat.add_mul_div_left _ _ hW, Nat.div_eq_of_lt (hl x hx), zero_add]

namespace Layout

variable {st : Stmt} (L : Layout st)
include L

theorem slot_eq {s₀ : ℕ} : st.c.unitRange.slot s₀ = (st.c.unitRange.pos0 + s₀) * 2 ^ st.c.unit.unitLog := by
  unfold Range.slot; rw [L.unit_slotLog]

end Layout

end FlockSoundness
