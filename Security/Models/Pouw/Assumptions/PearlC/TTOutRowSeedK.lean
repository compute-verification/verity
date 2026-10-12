import Models.Pouw.Assumptions.PearlC.TTOutRowSeed
import Definitions.Pouw.PearlC.DeviceKernelWref

/-!
# Per-row seeds against the honest kernel's `W_ref` (trusted definitions; nothing assumed)

The served line's record is rev1 with `W_ref` at `wrefDevRev1K d c`, the record's `W_ref` plus `c` per activation
element (`DeviceKernelWref.lean`). These are (C2)'s per-row cap protocol and tiles (`TTOutRowSeed.lean`) with that
`W_ref`, and change nothing else: the credit, the debit, the per-row cap and the checks are (C2)'s, so the row-seeded
TT_OUT reads the same game, and rev1's TT_OUT on the split workload bounds it as it does (C2)'s
(`RowSeedKGamma.lean`).
-/

namespace Pouw.PearlC

variable {Q R S : Type}

/-- **(C2) with the honest kernel's `W_ref`:** `pearlCProtocolDevRev1RowCap` with `W_ref` at `wrefDevRev1K d c`. -/
noncomputable def pearlCProtocolDevRev1KRowCap (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) :
    Protocol Q R S :=
  { pearlCProtocolDevRev1RowCap d sem ρ with Wref := fun L u => (wrefDevRev1K d c (L.shape u) : ℝ) }

/-- **(C2)'s tiles with the honest kernel's `W_ref`:** `pearlCTilesDevRev1RowCap` with `W_ref` and the credited rows'
`W_ref` at `wrefDevRev1K d c`. -/
noncomputable def pearlCTilesDevRev1KRowCap (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) :
    TileRules Q R S :=
  { pearlCTilesDevRev1RowCap d sem ρ with
    Wref := fun L g =>
      let tl := auditTiling 64 64 L
      let sh := L.shape (tl.unit g)
      ((wrefDevRev1K d c sh * tileShare sh (tl.rows g) (tl.cols g) : ℚ) : ℝ)
    Wcred := fun L g act =>
      let tl := auditTiling 64 64 L
      let sh := L.shape (tl.unit g)
      ((wrefDevRev1K d c sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ) }

end Pouw.PearlC
