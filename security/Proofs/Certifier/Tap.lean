module

public import Aesop.BuiltinRules
public import Definitions.Core.Assumptions
public import Definitions.Certifier.Tap
public import Mathlib.Tactic.ToDual
public import Proofs.Certifier.Cells
import Mathlib.Data.Finset.Attr
import Mathlib.Tactic.Finiteness.Attr
import Mathlib.Tactic.SetLike

/-! What the tap's certifier saw is what its relay sent; the record does not fix the server's bytes after the fail cell or past the lane's end. -/

@[expose] public section

namespace NetTiming

variable {κ Cell Ob Srv Dev : Type}

/-- The cells the trusted tap sends are the relay's code's. -/
theorem certifier_cells (f : RecordFormat κ Cell Ob) (arr : Srv → κ → Arrivals Cell) (tap : Dev)
    (run : Dev → CertifierSchedule κ × Srv → (κ → List (ℕ × CellView Cell)) × Ob)
    (htrust : Verity.Assumptions.TrustedDevice tap run fun p => recordingCode f p.1 (arr p.2)) (s : CertifierSchedule κ)
    (σ : Srv) : (run tap (s, σ)).1 = certifierCode s (arr σ) := by
  rw [htrust (s, σ)]
  rfl

/-- **What the certifier saw is what the relay sent**: on a schedule the record format holds, the cells the trusted tap
sends read back from the record its same run writes. -/
theorem certifier_cells_read (f : RecordFormat κ Cell Ob) (arr : Srv → κ → Arrivals Cell) (tap : Dev)
    (run : Dev → CertifierSchedule κ × Srv → (κ → List (ℕ × CellView Cell)) × Ob)
    (htrust : Verity.Assumptions.TrustedDevice tap run fun p => recordingCode f p.1 (arr p.2)) (s : CertifierSchedule κ)
    (hfit : f.fits s) (σ : Srv) : (run tap (s, σ)).1 = f.read s (run tap (s, σ)).2 := by
  rw [htrust (s, σ)]
  exact (f.read_record s hfit (arr σ)).symm

/-- **The data reaching the tap is in its record, up to the fail cell**: on a schedule the record format holds, cell `j`
of connection `c`'s stream, when its lane has cell `j` and no fail cell at or before it, is in the record as that
cell's data, in its bucket. -/
theorem certifier_data_read (f : RecordFormat κ Cell Ob) (arr : Srv → κ → Arrivals Cell) (tap : Dev)
    (run : Dev → CertifierSchedule κ × Srv → (κ → List (ℕ × CellView Cell)) × Ob)
    (htrust : Verity.Assumptions.TrustedDevice tap run fun p => recordingCode f p.1 (arr p.2)) (s : CertifierSchedule κ)
    (hfit : f.fits s) (σ : Srv) (c : κ) {j : ℕ} (hj : j < s.cells c) (hd : j < (arr σ c).stream.length)
    (hf : ∀ i, failCell s (arr σ) c = some i → j < i) :
    (f.read s (run tap (s, σ)).2 c)[j]? = some (s.bucket c j, .data (arr σ c).stream[j]) := by
  rw [← certifier_cells_read f arr tap run htrust s hfit σ, certifier_cells f arr tap run htrust s σ, certifierCode_eq]
  simp only [clientView, List.getElem?_map, List.getElem?_range hj, Option.map_some, cellView_of_lt _ hf,
    streamCell, hd, ↓reduceDIte]

end NetTiming
