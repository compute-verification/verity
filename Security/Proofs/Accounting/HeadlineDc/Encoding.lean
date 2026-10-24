import Definitions.Accounting.HeadlineDc.Encoding

/-! The proof-only part of `Definitions.Accounting.HeadlineDc.Encoding`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Accounting.TS.Headline

open Accounting.TS

/-- **The design's TS model on machine `G` with a filler**: `DesignData.model`, its filler menu one route whose demand
is the load on `G` of `fill`, the filler's counts a unit of honest work. -/
noncomputable def DesignData.modelFill {R : Type} (D : DesignData) (n nt nf : ℕ) [NeZero n] (G : Machine (Fin n) R)
    (fill : Sparse) : Model (Fin n) R (Fin nt) Unit Empty (Fin nf) Unit :=
  { D.model n nt nf G with fill := fun _ => G.load (Sparse.counts fill) }

end Accounting.TS.Headline
