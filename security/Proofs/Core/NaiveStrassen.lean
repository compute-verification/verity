module

public import Definitions.Core.NaiveStrassen
public import Definitions.Core.Guarantees.NaiveStrassen
public import Proofs.Core.NaiveStrassen.Eval

/-!
# Naive tiled against Strassen: the two statements' proofs, their witnesses and the four card tables

Theorem 1 at `L = 12` and its size form at `c = 4096` are proved on every atom table that the RTX PRO 6000 data
sheet dominates (`Packed.Dom`), from one table, that sheet's. The table is computed and checked by
`decide +kernel` (`NaiveStrassen/Eval.lean`): a pure kernel check, with no `native_decide` and no compiled code. Its
soundness is the induction on `Sched` in `NaiveStrassen/Sound.lean`.

Each statement has two witnesses. Its honest one: the sheet dominates the H200's data sheet, and one Winograd at
depth 1 (`winogradOnce`) is a schedule there that the statement is about: it costs Theorem 1's bound exactly, and in
size it loses to naive tiled at `l = b = n = 1`. Its nontrivial one: a table of one free atom is not dominated, and
on it the bound fails.

The four card tables, each card at its data sheet's and at its measured instruction `k`, are dominated, so both
statements hold on each (`naive_beats_strassen`, `naive_beats_strassen_up_to`). The two measured tables are written
here, beside those corollaries: nothing outside this file reads them.

Theorem 2 (root CRT, `NaiveBeatsCrtStrassen`) is open: its statement stays in `Definitions/Core/NaiveStrassen.lean`,
with its hypotheses' witness `crt_hypotheses` here, and nothing cites it.
-/

@[expose] public section

namespace Verity.Accel.NaiveStrassen

/-- One Winograd at depth 1 whose seven products are on a free atom: it costs nothing. -/
def freeOnce : Sched := .scheme winograd fun _ => .atom ⟨0, 4097, none⟩

/-- The RTX PRO 6000's data sheet does not dominate the table of one free atom: at `(0, 0)` its lightest atom weighs
`1`. -/
theorem free_not_dom : ¬ Packed.Dom [(⟨0, 4097, none⟩ : Atom)] := by
  intro hD
  have h := hD ⟨0, 4097, none⟩ (by simp) 0 0 (by decide)
  exact absurd h (by decide)

end Verity.Accel.NaiveStrassen

namespace Verity.SecurityProofs

open Verity.Accel.NaiveStrassen

theorem NaiveBeatsStrassenAny : Verity.Guarantees.NaiveBeatsStrassenAny :=
  fun _ hD => Packed.naiveBeatsStrassen_of_dom hD

theorem NaiveBeatsStrassenAnySize : Verity.Guarantees.NaiveBeatsStrassenAnySize :=
  fun _ hD => Packed.naiveBeatsStrassenUpTo_of_dom hD

/-- **`NaiveBeatsStrassenAny`'s honest witness**: the RTX PRO 6000's data sheet dominates the H200's, and on the
H200's one Winograd at depth 1 is a schedule the statement is about, whose cost is the bound, `8^2 + 32`. -/
theorem NaiveBeatsStrassenAny_honest :
    Packed.Dom h200Sheet ∧ winogradOnce.valid h200Sheet schemes 0 1 256 256 = true ∧ winogradOnce.fast = true ∧
      winogradOnce.cost 1 = 8 ^ (1 + 1) + 32 :=
  ⟨Packed.dom_h200Sheet, by decide, by decide, by decide⟩

/-- **`NaiveBeatsStrassenAnySize`'s honest witness**: the RTX PRO 6000's data sheet dominates the H200's, and on the
H200's one Winograd at depth 1 is a schedule the statement is about, which loses to naive tiled at `l = b = n = 1`. -/
theorem NaiveBeatsStrassenAnySize_honest :
    Packed.Dom h200Sheet ∧ winogradOnce.valid h200Sheet schemes 0 1 256 256 = true ∧ winogradOnce.fast = true ∧
      8 * 1 ^ 3 < 1 ^ 3 * winogradOnce.cost 1 :=
  ⟨Packed.dom_h200Sheet, by decide, by decide, by decide⟩

/-- **`NaiveBeatsStrassenAny` says something**: the table of one free atom is not one the RTX PRO 6000's data sheet
dominates, and on it Theorem 1 fails at depth 1, where `freeOnce` costs `0`. -/
theorem NaiveBeatsStrassenAny_nontrivial :
    ¬ Packed.Dom [(⟨0, 4097, none⟩ : Atom)] ∧ ¬ NaiveBeatsStrassen [(⟨0, 4097, none⟩ : Atom)] 12 := by
  refine ⟨free_not_dom, fun h => ?_⟩
  have h1 := h 1 (by decide) freeOnce (by decide) (by decide)
  exact absurd h1 (by decide)

/-- **`NaiveBeatsStrassenAnySize` says something**: the table of one free atom is not one the RTX PRO 6000's data
sheet dominates, and on it the size form fails at `l = b = n = 1`, where `freeOnce` costs `0`. -/
theorem NaiveBeatsStrassenAnySize_nontrivial :
    ¬ Packed.Dom [(⟨0, 4097, none⟩ : Atom)] ∧ ¬ NaiveBeatsStrassenUpTo [(⟨0, 4097, none⟩ : Atom)] 4096 := by
  refine ⟨free_not_dom, fun h => ?_⟩
  have h1 := h 1 1 1 freeOnce (by decide) (by decide) (by decide) (by decide) (by decide)
  exact absurd h1 (by decide)

end Verity.SecurityProofs

namespace Verity.Accel.NaiveStrassen

/-! ## The four card tables -/

/-- The H200 SXM at the measured instruction `k` (f16 and bf16 at 16, tf32 at 8). -/
def h200Measured : List Atom :=
  [s8, ⟨8, 33, some (32, 14)⟩, ⟨8, 17, some (32, 14)⟩, ⟨16, 4097, some (16, 24)⟩, ⟨16, 513, some (16, 24)⟩,
    ⟨32, 4097, some (8, 24)⟩, b1]

/-- The RTX PRO 6000 at the measured `k` (f16 and bf16 at 16, f16 at 8 at twice the weight, tf32 at 4), its `b1`
`rtx6000Sheet`'s, at the H200's weight 1/8. -/
def rtx6000Measured : List Atom :=
  [s8, ⟨8, 33, some (32, 24)⟩, ⟨8, 17, some (32, 24)⟩, ⟨8, 9, some (32, 24)⟩, ⟨4, 9, some (64, 24)⟩,
    ⟨16, 4097, some (16, 24)⟩, ⟨32, 4097, some (8, 24)⟩, ⟨16, 513, some (16, 24)⟩, ⟨32, 4097, some (4, 24)⟩, b1]

/-- The four card tables. -/
def instances : List (List Atom) := [h200Sheet, h200Measured, rtx6000Sheet, rtx6000Measured]

theorem Packed.dom_h200Measured : Packed.Dom h200Measured := Packed.dom_of_covers (by decide)

theorem Packed.dom_rtx6000Measured : Packed.Dom rtx6000Measured := Packed.dom_of_covers (by decide)

/-- Theorem 1 at the served depth on the four card tables. -/
theorem naive_beats_strassen :
    NaiveBeatsStrassen h200Sheet 12 ∧ NaiveBeatsStrassen h200Measured 12 ∧ NaiveBeatsStrassen rtx6000Sheet 12 ∧
      NaiveBeatsStrassen rtx6000Measured 12 :=
  ⟨Verity.SecurityProofs.NaiveBeatsStrassenAny _ Packed.dom_h200Sheet,
    Verity.SecurityProofs.NaiveBeatsStrassenAny _ Packed.dom_h200Measured,
    Verity.SecurityProofs.NaiveBeatsStrassenAny _ Packed.dom_rtx6000Sheet,
    Verity.SecurityProofs.NaiveBeatsStrassenAny _ Packed.dom_rtx6000Measured⟩

/-- Theorem 1 in size on the four card tables. -/
theorem naive_beats_strassen_up_to :
    NaiveBeatsStrassenUpTo h200Sheet 4096 ∧ NaiveBeatsStrassenUpTo h200Measured 4096 ∧
      NaiveBeatsStrassenUpTo rtx6000Sheet 4096 ∧ NaiveBeatsStrassenUpTo rtx6000Measured 4096 :=
  ⟨Verity.SecurityProofs.NaiveBeatsStrassenAnySize _ Packed.dom_h200Sheet,
    Verity.SecurityProofs.NaiveBeatsStrassenAnySize _ Packed.dom_h200Measured,
    Verity.SecurityProofs.NaiveBeatsStrassenAnySize _ Packed.dom_rtx6000Sheet,
    Verity.SecurityProofs.NaiveBeatsStrassenAnySize _ Packed.dom_rtx6000Measured⟩

/-- Two s8 residues, mod 256 and mod 255, meet `crtValid` at `n = 2` on every card table. -/
theorem crt_hypotheses :
    (instances.all fun fs => crtValid fs 2 0 [(256, .atom s8), (255, .atom s8)]) = true ∧
      8 * 2 ^ 3 < 2 ^ 3 * (([(256, Sched.atom s8), (255, Sched.atom s8)].map fun r => r.2.cost 0).foldl (· + ·) 0) := by
  decide

end Verity.Accel.NaiveStrassen
