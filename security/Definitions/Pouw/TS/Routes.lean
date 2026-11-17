module

public import Definitions.Pouw.TS.Machine
public import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-! EPS1-SPEC's routes, labelled programs and terms over the TS machine, priced per unit of honest work throughout. -/

@[expose] public section

namespace ResourceAccounting.TS

open Finset

/-- **What heads an owned right fixed committed word** (§1.3), and the word route that prices it (§0.5): a writer kind
`t` (an MMA kind, priced `T_κ`, or a uniform class), a gather-free vector instruction (`S_V`), a vector instruction
whose production reads a load at a data-dependent address served from memory level `l` (a table word), or side unit
`u`. -/
inductive Head (T L U : Type)
  | writer (t : T)
  | vector
  | table (l : L)
  | side (u : U)

/-- `Head T L U ≃ T ⊕ Unit ⊕ L ⊕ U`, for its `Fintype`. -/
def Head.equiv (T L U : Type) : Head T L U ≃ T ⊕ (Unit ⊕ (L ⊕ U)) where
  toFun
    | .writer t => .inl t
    | .vector => .inr (.inl ())
    | .table l => .inr (.inr (.inl l))
    | .side u => .inr (.inr (.inr u))
  invFun
    | .inl t => .writer t
    | .inr (.inl ()) => .vector
    | .inr (.inr (.inl l)) => .table l
    | .inr (.inr (.inr u)) => .side u
  left_inv h := by cases h <;> rfl
  right_inv s := by rcases s with t | ⟨⟨⟩⟩ | l | u <;> rfl

noncomputable instance instFintypeHead (T L U : Type) [Fintype T] [Fintype L] [Fintype U] : Fintype (Head T L U) :=
  Fintype.ofEquiv _ (Head.equiv T L U).symm

/-- **The TS model of a design** over classes `K`, facets `R`, writer kinds `T`, memory levels `L`, side units `U`,
fold routes `F` and filler routes `X`. -/
structure Model (K R T L U F X : Type) where
  G : Machine K R
  cls : T → K
  /-- The words one instruction of writer kind `t` writes, at most. -/
  omega : T → ℝ
  vec : Finset K
  bytes : L → K
  /-- SCALAR-ROUTE's floor for a gather-free vector-headed word, in lane-instructions. -/
  q : ℝ
  /-- SCALAR-ROUTE's floors for a table word, in lane-instructions and in bytes on its level. -/
  qT : ℝ
  bT : ℝ
  /-- Side unit `u`'s instruction vector per word. -/
  side : U → K → ℝ
  /-- Each fold route's demand per committed word on every facet. -/
  fold : F → R → ℝ
  /-- Each filler route's demand per certified window on every facet. -/
  fill : X → R → ℝ
  /-- `N_w`, the committed words per unit of honest work. -/
  Nw : ℝ
  /-- The fold's and the filler's optimality slack, 0 under FOLD-OPT and FILL-OPT. -/
  muF : ℝ
  muX : ℝ

/-- **A labelled program** (§0.4), per unit of honest work: its word production `nw` (the MMAs and the operand loading
and forming the checked words need, and everything else it executes that isn't below), its fold `nf`, its filler
`nx` and its ε_skip work `ns` (work Verify neither checks nor needs). -/
structure Labelled (K : Type) where
  nw : K → ℝ
  nf : K → ℝ
  nx : K → ℝ
  ns : K → ℝ

/-- A labelled program's counts: `n = n^w + n^f + n^x + n^s`. -/
def Labelled.n {K : Type} (P : Labelled K) : K → ℝ := fun k => P.nw k + P.nf k + P.nx k + P.ns k

/-- **The honest kernel** `H`: its labelled split, its measured cycles per unit at saturation `C_H` and its measured
sustained SM clock `f_H`. -/
structure Honest (K : Type) where
  prog : Labelled K
  CH : ℝ
  fH : ℝ

/-- Honest's time per unit: `T(H) = C_H / f_H`. -/
noncomputable def Honest.time {K : Type} (H : Honest K) : ℝ := H.CH / H.fH

/-- **A cheater's labelled run** (§2.1's RD reads it): the run, its word production split by the head of the words it
produces (`nword h`), its fold `nfold` and filler `nfill`, its owned right fixed committed words per head
(`words h`), and `fixed`, its owned right committed words less the adaptive ones (`N_right − N_adapt`), all per unit
of honest work. The labelling is §0.4's and §1.3's (heads, ownership, copies); the named assumptions over an
`Anatomy` are claimed at that labelling. -/
structure Anatomy (K T L U : Type) where
  run : Run K
  nword : Head T L U → K → ℝ
  nfold : K → ℝ
  nfill : K → ℝ
  words : Head T L U → ℝ
  fixed : ℝ

variable {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
  [Fintype X]

/-- The labelling splits the run: every part is `≥ 0` and the parts sum to at most the run's counts, class by class. -/
def Anatomy.Split (A : Anatomy K T L U) : Prop :=
  (∀ h k, 0 ≤ A.nword h k) ∧ (∀ k, 0 ≤ A.nfold k) ∧ (∀ k, 0 ≤ A.nfill k) ∧ (∀ h, 0 ≤ A.words h) ∧
    ∀ k, ∑ h, A.nword h k + A.nfold k + A.nfill k ≤ A.run.n k

namespace Model

variable (Z : Model K R T L U F X)

/-- The cheapest vector class's coefficient on facet `r`: `min_{k ∈ V} a_{r,k}`. -/
noncomputable def vecMin (r : R) : ℝ := ⨅ k : Z.vec, Z.G.a r k

/-- **The word routes' demands** (§0.5): `T_κ` is `1/ω_t` of an instruction of `t`'s class, operand delivery priced 0
and the FP extraction 0 (conservative); `S_V` is `(q/32)·min_{k∈V} a_{r,k}`; a table route at level `l` is at least
`(q_T/32)·min_{k∈V} a_{r,k}` on issue and `b_T` bytes on `l`'s facet (the larger of the two, conservative); a side
unit's route is its per-word vector. -/
noncomputable def demand : Head T L U → R → ℝ
  | .writer t, r => Z.G.a r (Z.cls t) / Z.omega t
  | .vector, r => Z.q / 32 * Z.vecMin r
  | .table l, r => max (Z.qT / 32 * Z.vecMin r) (Z.bT * Z.G.a r (Z.bytes l))
  | .side u, r => Z.G.load (Z.side u) r

noncomputable def tensorDemandDel (t : T) (phiDel : ℝ) (r : R) : ℝ :=
  (Z.G.a r (Z.cls t) + phiDel * Z.vecMin r) / Z.omega t

/-- `w*(π) = min_{ρ∈𝒲} π·v_ρ`, per word. -/
noncomputable def wStar (lam : R → ℝ) : ℝ := ⨅ h : Head T L U, demandPrice lam (Z.demand h)

/-- `φ*(π) = min_{φ∈ℱ} π·u_φ`, per word. -/
noncomputable def phiStar (lam : R → ℝ) : ℝ := ⨅ φ : F, demandPrice lam (Z.fold φ)

/-- `χ*(π) = min_{ψ∈𝒳} π·x_ψ`, per window. -/
noncomputable def chiStar (lam : R → ℝ) : ℝ := ⨅ ψ : X, demandPrice lam (Z.fill ψ)

/-- The labelled program's bound, `B(L)`, every term's denominator (`B(H)`, or `B_♭` in the ♭ form). -/
noncomputable def bnd (P : Labelled K) : ℝ := Z.G.bound P.n

/-- `ε_cert(π) = 1 − π·n(L)/B(L)`. -/
noncomputable def epsCert (P : Labelled K) (lam : R → ℝ) : ℝ := 1 - Z.G.price lam P.n / Z.bnd P

/-- `ε_skip(π) = π·n^s(L)/B(L)`. -/
noncomputable def epsSkip (P : Labelled K) (lam : R → ℝ) : ℝ := Z.G.price lam P.ns / Z.bnd P

/-- `ε_route(π) = [N_w(w_L − w*) + N_w(φ_L − φ*) + (χ_L − χ*)]/B(L)`, with `w_L = π·n^w(L)/N_w`,
`φ_L = π·n^f(L)/N_w` and `χ_L = π·n^x(L)`. -/
noncomputable def epsRoute (P : Labelled K) (lam : R → ℝ) : ℝ :=
  (Z.Nw * (Z.G.price lam P.nw / Z.Nw - Z.wStar lam) + Z.Nw * (Z.G.price lam P.nf / Z.Nw - Z.phiStar lam) +
    (Z.G.price lam P.nx - Z.chiStar lam)) / Z.bnd P

/-- `ε_machine(π) = ε_cert(π) + ε_route(π)`. -/
noncomputable def epsMachine (P : Labelled K) (lam : R → ℝ) : ℝ := Z.epsCert P lam + Z.epsRoute P lam

/-- `ε_mand(π) = [μ_f N_w φ* + μ_x χ*]/B(L)`. -/
noncomputable def epsMand (P : Labelled K) (lam : R → ℝ) : ℝ :=
  (Z.muF * Z.Nw * Z.phiStar lam + Z.muX * Z.chiStar lam) / Z.bnd P

/-- `κ(π) = N_w w*(π)/B(L)`. -/
noncomputable def kappa (P : Labelled K) (lam : R → ℝ) : ℝ := Z.Nw * Z.wStar lam / Z.bnd P

noncomputable def epsLuck (P : Labelled K) (lam : R → ℝ) (dbar : ℝ) : ℝ := dbar * Z.kappa P lam

noncomputable def epsAdapt (P : Labelled K) (lam : R → ℝ) (abar : ℝ) : ℝ := abar * Z.kappa P lam

/-- `ε_jl(π) = κ(π)·ln(1/δ′)/(ln(1/p)·S)`: the luck at 100% checking, one bet a round. -/
noncomputable def epsJl (P : Labelled K) (lam : R → ℝ) (p δ' : ℝ) (S : ℕ) : ℝ :=
  Z.kappa P lam * Real.log (1 / δ') / (Real.log (1 / p) * S)

/-- `ε_sample = (Λ − Λ_full)·(ln(1/δ′)/S)·κ(π)`, `Λ_full = 1/ln(1/p)` (2.8). -/
noncomputable def epsSample (P : Labelled K) (lam : R → ℝ) (Λ p δ' : ℝ) (S : ℕ) : ℝ :=
  (Λ - 1 / Real.log (1 / p)) * (Real.log (1 / δ') / S) * Z.kappa P lam

/-- `ε_sched = 1 − B(H)/C_H`. -/
noncomputable def epsSched (H : Honest K) : ℝ := 1 - Z.bnd H.prog / H.CH

/-- `ε_clock = 1 − f_H/f_max`. -/
noncomputable def epsClock (H : Honest K) : ℝ := 1 - H.fH / Z.G.fmax

/-- The credit at the floor: `T_cred = B_♭ / f_max` (§0.7a). -/
noncomputable def credit (P : Labelled K) : ℝ := Z.bnd P / Z.G.fmax

/-- `γ_♭ = (C_H/B_♭)(f_max/f_H)`, so that `T(H) = γ_♭ · T_cred`. -/
noncomputable def gammaFloor (H : Honest K) (P : Labelled K) : ℝ := H.CH / Z.bnd P * (Z.G.fmax / H.fH)

def RouteDominated (n : K → ℝ) (fixed : ℝ) : Prop :=
  ∃ (N : Head T L U → ℝ) (Mf : F → ℝ) (Mx : X → ℝ),
    (∀ h, 0 ≤ N h) ∧ (∀ φ, 0 ≤ Mf φ) ∧ (∀ ψ, 0 ≤ Mx ψ) ∧
    fixed ≤ ∑ h, N h ∧ (1 - Z.muF) * Z.Nw ≤ ∑ φ, Mf φ ∧ (1 - Z.muX) ≤ ∑ ψ, Mx ψ ∧
    ∀ r, ∑ h, N h * Z.demand h r + ∑ φ, Mf φ * Z.fold φ r + ∑ ψ, Mx ψ * Z.fill ψ r ≤ Z.G.load n r

def FloorKnown (P : Labelled K) : Prop :=
  (∃ N : Head T L U → ℝ, (∀ h, 0 ≤ N h) ∧ ∑ h, N h = Z.Nw ∧ ∀ r, Z.G.load P.nw r = ∑ h, N h * Z.demand h r) ∧
    (∃ φ : F, ∀ r, Z.G.load P.nf r = Z.Nw * Z.fold φ r) ∧ (∃ ψ : X, ∀ r, Z.G.load P.nx r = Z.fill ψ r)

/-- **The model is well formed** for the labelled program `L` the terms divide by: every facet row is `≥ 0`
(Definition 1), `f_max`, `N_w` and `B(L)` are positive, every `ω_t` is positive, SCALAR-ROUTE's floors, the side
units' vectors and the fold and filler demands are `≥ 0`, and `μ_f, μ_x ∈ [0, 1]`. -/
def Wf (P : Labelled K) : Prop :=
  (∀ r k, 0 ≤ Z.G.a r k) ∧ 0 < Z.G.fmax ∧ 0 < Z.Nw ∧ 0 < Z.bnd P ∧ (∀ t, 0 < Z.omega t) ∧
    0 ≤ Z.q ∧ 0 ≤ Z.qT ∧ 0 ≤ Z.bT ∧ (∀ u k, 0 ≤ Z.side u k) ∧ (∀ φ r, 0 ≤ Z.fold φ r) ∧
    (∀ ψ r, 0 ≤ Z.fill ψ r) ∧ 0 ≤ Z.muF ∧ Z.muF ≤ 1 ∧ 0 ≤ Z.muX ∧ Z.muX ≤ 1

end Model

end ResourceAccounting.TS
