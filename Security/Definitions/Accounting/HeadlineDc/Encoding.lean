import Definitions.Accounting.Headline.Encoding

/-!
# Exact data for a data-centre headline instance (trusted)

`Definitions.Accounting.Headline.Encoding`'s encoding, with the two things the H200's and B200's instances
(`benchmarks/pouw/lean_headline_dc.py`) need beside it:

* `DesignData.modelFill`: `DesignData.model` with a filler, its menu the one route the floor program's filler runs
  on, whose demand on each facet is the load of its counts a unit of honest work (the H200's ffill, which FILL-OPT
  prices at `χ*`);
* `cert`: a certificate over `Fin m` from sparse weights, for a certificate on more than one facet.
-/

namespace Accounting.TS.Headline

open Accounting.TS

/-- **The design's TS model on machine `G` with a filler**: `DesignData.model`, its filler menu one route whose demand
is the load on `G` of `fill`, the filler's counts a unit of honest work. -/
noncomputable def DesignData.modelFill {R : Type} (D : DesignData) (n nt nf : ℕ) [NeZero n] (G : Machine (Fin n) R)
    (fill : Sparse) : Model (Fin n) R (Fin nt) Unit Empty (Fin nf) Unit :=
  { D.model n nt nf G with fill := fun _ => G.load (Sparse.counts fill) }

/-- A certificate over `Fin m` from `(facet index, numerator, denominator)` weights. -/
noncomputable def cert (m : ℕ) (v : Sparse) : Fin m → ℝ := Sparse.counts v

end Accounting.TS.Headline
