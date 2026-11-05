import Definitions.NetworkCertifier.Tap
import Definitions.Core.Assumptions
import Proofs.NetworkCertifier.Warden

/-!
# What the tap's certifier saw is what crossed the wire

`Definitions.NetworkCertifier.Tap`. Under `trusted-host` the tap's run is its code, so the frames the client receives
are those of the warden's code's cells, padding included (`tap_cells`), and, on a schedule the record format holds
(`TapRecord.fits`), read back from the record the same run writes (`tap_cells_read`): what the certifier saw is what
crossed the wire, by the one code run. Of the data reaching the tap on a connection, each cell its lane sends before
its fail cell is in the record, as that cell's data, the `(j mod r)`-th of the connection's frames in its bucket
(`tap_data_read`). The rest of the server's bytes, after the fail cell or past the lane's end, reach neither the client
nor the record, so the record does not fix the whole stream the server sent.
-/

namespace NetTiming

variable {κ Cell Ob Srv Dev : Type}

/-- The frames the trusted tap sends are those of the warden's code's cells, padding included. -/
theorem tap_cells (f : TapRecord κ Cell Ob) (arr : Srv → κ → Arrivals Cell) (tap : Dev)
    (run : Dev → WardenSchedule κ × Srv → (κ → List (ℕ × CellView Cell)) × Ob)
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => tapCode f p.1 (arr p.2)) (s : WardenSchedule κ)
    (σ : Srv) : (run tap (s, σ)).1 = laneWire s (wardenCode s (arr σ)) := by
  rw [htrust (s, σ)]
  rfl

/-- **What the certifier saw is what crossed the wire**: on a schedule the record format holds, the frames the trusted
tap sends read back from the record its same run writes. -/
theorem tap_cells_read (f : TapRecord κ Cell Ob) (arr : Srv → κ → Arrivals Cell) (tap : Dev)
    (run : Dev → WardenSchedule κ × Srv → (κ → List (ℕ × CellView Cell)) × Ob)
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => tapCode f p.1 (arr p.2)) (s : WardenSchedule κ)
    (hfit : f.fits s) (σ : Srv) : (run tap (s, σ)).1 = f.read s (run tap (s, σ)).2 := by
  rw [htrust (s, σ)]
  exact (f.read_record s hfit (arr σ)).symm

/-- **The data reaching the tap is in its record, up to the fail cell**: on a schedule the record format holds, cell `j`
of connection `c`'s stream, when its lane has cell `j` and no fail cell at or before it, is in the record as that
cell's data, the `(j mod r)`-th of the connection's frames in its bucket. -/
theorem tap_data_read (f : TapRecord κ Cell Ob) (arr : Srv → κ → Arrivals Cell) (tap : Dev)
    (run : Dev → WardenSchedule κ × Srv → (κ → List (ℕ × CellView Cell)) × Ob)
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => tapCode f p.1 (arr p.2)) (s : WardenSchedule κ)
    (hfit : f.fits s) (σ : Srv) (c : κ) {j : ℕ} (hj : j < s.cells c) (hd : j < (arr σ c).stream.length)
    (hf : ∀ i, failCell s (arr σ) c = some i → j < i) :
    ((f.read s (run tap (s, σ)).2 c).filter fun x => x.1 == s.bucket c j)[j % s.r]? =
      some (s.bucket c j, .data (arr σ c).stream[j]) := by
  rw [← tap_cells_read f arr tap run htrust s hfit σ, tap_cells f arr tap run htrust s σ,
    filter_laneWire s _ c (bucket_mem_span s c hj), List.getElem?_map, laneFrames_cell s _ c hj, wardenCode_eq]
  simp only [clientView, List.getElem?_map, List.getElem?_range hj, Option.map_some, cellView_of_lt _ hf,
    streamCell, hd, ↓reduceDIte]

end NetTiming
