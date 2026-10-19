import Models.Accounting.Guarantees.FpG1
import Models.Accounting.Guarantees.Headline

/-!
# RT2-FP-13: sm_120's g1 headlines are the record's (a red-team probe on #2770, not for landing)

`G1Sm120SheetHeadline`'s grids hypothesis reads `ops`, which the statement binds itself, so every window meets it: at an
empty `ι` (`sheet_of_g1`, `measured_of_g1`), and at any `ι` by the constant reading `(.act, .int8 0 0)` (`grids_any`).
Each g1 headline therefore implies the record's, and the record implies it (`Proofs.Accounting.Headline.FpG1`): the two
are equivalent, and the hypothesis restricts no window.
-/

namespace Accounting.SecurityProofs.TS.RT2FP

open Accounting.Guarantees.TS

theorem sheet_of_g1 (h : G1Sm120SheetHeadline) : Sm120SheetHeadline := by
  intro Ω _ Pl G Uf V nπ pf δbind δzk
  exact h Ω Pl G Uf pf (fun (_ : V) (j : Empty) => j.elim) δbind δzk (fun _ _ _ _ j => j.elim)

theorem measured_of_g1 (h : G1Sm120MeasuredHeadline) : Sm120MeasuredHeadline := by
  intro Ω _ Pl G Uf V nπ pf δbind δzk
  exact h Ω Pl G Uf pf (fun (_ : V) (j : Empty) => j.elim) δbind δzk (fun _ _ _ _ j => j.elim)

theorem grids_any {V ι : Type} (rel : V → Prop) :
    ∃ ops : V → ι → Pouw.FpG1.Side × Pouw.FpG1.Entry,
      ∀ v, rel v → ∀ j, Pouw.FpG1.Grids.sm120.On (ops v j).1 (ops v j).2 :=
  ⟨fun _ _ => (.act, .int8 0 0), fun _ _ _ => show (-126 : ℤ) ≤ 0 ∧ (0 : ℤ) ≤ 126 from ⟨by omega, by omega⟩⟩

end Accounting.SecurityProofs.TS.RT2FP
