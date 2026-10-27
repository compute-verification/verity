import Mathlib

/-!
# The tensor–scalar machine (trusted)

EPS1's tensor–scalar (TS) model (§0.1–§0.3, §1.1): instruction classes `K`,
facets `R`, each facet a row `a r : K → ℝ` (the cycles of facet `r` one instruction of class `k` occupies, per SMSP),
SM-clocked or fixed-clock, and the highest SM clock `fmax` any program runs at (P1: 2,430 MHz on sm_120).

* A program's counts `n : K → ℝ` are its device totals of each class divided by the SMSPs, per unit of honest work.
* Its load on facet `r` is `a_r · n` (`Machine.load`), and its bound `B(n) = max_r a_r · n` (`Machine.bound`).
* A run (`Run`) is a program's counts with its cycles per SMSP and its SM clock; its time is cycles over clock.
* A certificate (`IsCert`) is a probability vector `λ` on the facets. Its price of counts is
  `π · n = Σ_r λ_r a_r · n` (`Machine.price`), and of a route's per-facet demand `d` it is `Σ_r λ_r d_r`
  (`demandPrice`).
-/

namespace Accounting.TS

open Finset

/-- **A TS machine** over classes `K` and facets `R`: per facet its row `a r` (cycles of that facet one class-`k`
instruction occupies, per SMSP), whether the facet is SM-clocked (`true`: pipes, issue, joint issue frontiers, shared
memory, TEX, LDSM) or fixed-clock (`false`: DRAM, PCIe, anything off the SM clock, its row
`a_{r,k} = fmax · N_SMSP / R_{r,k}` priced at the fastest SM clock), and `fmax`, the highest SM clock. -/
structure Machine (K R : Type) where
  a : R → K → ℝ
  smClocked : R → Bool
  fmax : ℝ

/-- A run of a program on the device, per unit of honest work: its instruction counts `n` (normalized per SMSP), its
cycles `cycles` per SMSP and its SM clock `clock`. -/
structure Run (K : Type) where
  n : K → ℝ
  cycles : ℝ
  clock : ℝ

variable {K R : Type} [Fintype K] [Fintype R]

/-- A program's load on facet `r`: `a_r · n`. -/
def Machine.load (G : Machine K R) (n : K → ℝ) (r : R) : ℝ := ∑ k, G.a r k * n k

/-- The model's bound `B(n) = max_r a_r · n`, in cycles. -/
noncomputable def Machine.bound (G : Machine K R) (n : K → ℝ) : ℝ := ⨆ r, G.load n r

/-- A run's time: its cycles over its clock. -/
noncomputable def Run.time (P : Run K) : ℝ := P.cycles / P.clock

/-- A certificate: facet weights `λ ≥ 0` with `Σ_r λ_r = 1`. -/
def IsCert (lam : R → ℝ) : Prop := (∀ r, 0 ≤ lam r) ∧ ∑ r, lam r = 1

/-- The certificate `π = Σ_r λ_r a_r`'s price of counts: `π · n = Σ_r λ_r a_r · n`. -/
def Machine.price (G : Machine K R) (lam : R → ℝ) (n : K → ℝ) : ℝ := ∑ r, lam r * G.load n r

/-- The certificate's price of a route's per-facet demand `d`: `Σ_r λ_r d_r`. -/
def demandPrice (lam : R → ℝ) (d : R → ℝ) : ℝ := ∑ r, lam r * d r

/-- The certificate that sits on one facet `r` (`π = a_r`). -/
def onFacet [DecidableEq R] (r : R) : R → ℝ := fun r' => if r' = r then 1 else 0

end Accounting.TS
