import Definitions.Pouw.Guarantees.Writes
import Proofs.Pouw.Dimension.H100WProofs
import Proofs.Pouw.Lead.CoinBranch.PointwiseProofs
import Proofs.Pouw.Lead.Tags.SsnProofs

/-!
# Proofs of what writing distinct words costs (draft semantics, pending review)

One theorem `Pouw.SecurityProofs.Writes.G : Pouw.Guarantees.Writes.G` per guarantee, each the theorem of
`H100WProofs`, `SsnProofs` or `PointwiseProofs` that proves it under its own name.
-/

namespace Pouw.SecurityProofs.Writes

open Pouw.Dimension.Ssn Pouw.Dimension.Pointwise

theorem DistinctWritesMH100Floor : Pouw.Guarantees.Writes.DistinctWritesMH100Floor :=
  Pouw.Dimension.H100.distinctWritesMH100Floor

theorem SsnServedDistinct : Pouw.Guarantees.Writes.SsnServedDistinct := fun _ _ _ κ n hS =>
  ⟨fun hC hn => ssn1lServed_full hC κ hS n hn, fun hC hn => ssn1l5Served_full hC κ hS n hn⟩

theorem SsnServedCharged : Pouw.Guarantees.Writes.SsnServedCharged := fun _ _ _ κ n fpSem L P hS hL hP =>
  ⟨fun hC hn hO => ssn1lServedB_charged hC κ hS n hn fpSem L P hL hP hO,
    fun hC hn hO => ssn1l5Served_charged hC κ hS n hn fpSem L P hL hP hO⟩

theorem SsnServedShift : Pouw.Guarantees.Writes.SsnServedShift := fun _ _ _ κ n hS =>
  ⟨fun hC hn => ssn1lServed_shift hC κ hS n hn, fun hC hn => ssn1l5Served_shift hC κ hS n hn⟩

theorem RunPairs : Pouw.Guarantees.Writes.RunPairs := fun _ _ _ _ _ _ m₀ P Y hY => run_pairs_le m₀ P Y hY

theorem RunBall : Pouw.Guarantees.Writes.RunBall := fun _ _ _ _ _ _ m₀ P Y r hY hI hYY =>
  run_ball_le m₀ P Y hY r hI hYY

end Pouw.SecurityProofs.Writes
