import FlockSoundness.Lowering
import Flock.Net
-- level3 already realizes the lemmas this file's `simp` calls need (`Std.Legacy.Range.mk.congr_simp`), so they are
-- reused rather than declared a second time, which the audit's kernel replay refuses
import FlockLevel3.LookupRows
import FlockSoundness.Protocol.ExecRows

/-!
# The rows the verifier parses, as the audit's `Rows`

The executable's parser (`Flock.Net.parse`) holds every expanded circuit's rows to #144's order
(`Net.checkOrder`, PR #147): a computed row, `inCols ≤ r < constPos`, reads in `A` and in `B` only
the constant or an earlier column that is not input padding (`RowOrder`). `checkOrder_ok` turns an
accepted check into that property, and `Rows.ofOrdered` builds the audit's `Rows` from rows that
have it, with `topo` proved: what #144 named, the parser now checks.
-/

namespace FlockSoundness

open Flock

/-! ## Loops that only continue or throw -/

/-! ## The order `Net.checkOrder` checks -/

/-! ## The audit's `Rows` from ordered rows -/

section ofOrdered

variable {inCols constPos : ℕ} {ra rb : Array (Array ℕ)} {h : RowOrder inCols constPos ra rb}

theorem Rows.ofOrdered_nIn (hle : inCols ≤ constPos) : (Rows.ofOrdered inCols constPos ra rb h).nIn = inCols := by
  show min inCols constPos = inCols
  exact Nat.min_eq_left hle

theorem Rows.ofOrdered_comp (i : Fin (Rows.ofOrdered inCols constPos ra rb h).nComp) :
    ((Rows.ofOrdered inCols constPos ra rb h).comp i).val =
      (Rows.ofOrdered inCols constPos ra rb h).nIn + i.val := rfl

end ofOrdered

/-! ## What `Net.parse` accepted -/

theorem parse_tail {α : Type} {x y : Except String Sparse} {mk : Sparse → Sparse → α} {v : α}
    (h : (do let a ← x; let b ← y; pure (mk a b)) = .ok v) : ∃ a b, x = .ok a ∧ y = .ok b ∧ v = mk a b := by
  obtain ⟨a, ha, h⟩ := except_bind_ok h
  obtain ⟨b, hb, h⟩ := except_bind_ok h
  cases h
  exact ⟨a, b, ha, hb, rfl⟩

/-- **The parser orders every net it accepts**: the rows it stored as `A` and `B` have `RowOrder`,
so #144's `Rows.topo` for a parsed net is an executable check, not a hypothesis. -/
theorem parse_rowOrder {lines : Array String} {net : Net} (h : Net.parse lines = .ok net) :
    ∃ ra rb, RowOrder (net.inWords * WORD) net.constPos ra rb ∧
      Sparse.ofRows ra (2 ^ net.unitLog) = .ok net.a ∧ Sparse.ofRows rb (2 ^ net.unitLog) = .ok net.b :=
  built_rowOrder (parse_built h)

/-! ## `Sparse`: the bytes `ofRows` writes, and `row` reads them back -/

/-! ## `Sparse.ofRows`, read back -/

/-! ## The audit's `Rows` for a built net -/

section ofNet

variable {net : Net} (h : Built net)

theorem Rows.ofNet_comp (i : Fin (Rows.ofNet h).nComp) :
    ((Rows.ofNet h).comp i).val = (Rows.ofNet h).nIn + i.val := rfl

end ofNet

/-! ## An instance's rows: `k` copies of the template's, stacked -/

namespace Rows

variable (R : Rows) (k : ℕ)

end Rows

/-! ## An instance placed in a statement -/

section Place

variable (S : Model.Statement) (R : Rows) (k : ℕ) (base : ℕ → ℕ)
  (hb : ∀ u < k, base u + (R.nIn + R.nComp) < 2 ^ S.kLog)

end Place

end FlockSoundness