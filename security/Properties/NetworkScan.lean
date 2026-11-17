module

public import Properties.NetworkScan.Assumptions
public import NetworkCheck.Period
public import Definitions.Certifier.SampledAudit
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

/-! The rows form's scan fits its layout: what `NetworkCheck.Period.scanOf`'s acceptance gives the sampled audit's scope. -/

@[expose] public section

namespace Properties.NetworkScan

open NetworkCheck.Period

/-- The gate the reference `r` resolves to from the activation `sc` (`Flock.Program.resolve`), `0` where it doesn't
resolve. -/
def gateAt (P : Flock.Program.Prog) (sc : Flock.Program.Scope) (r : Flock.Program.Ref) : ℕ :=
  (Flock.Program.resolve P sc r).toOption.getD 0

/-- **The scan's layout** in its Program `P` at the schedule `s`, over one egress link: step `t`'s slot `σ` is bit 0 of
egress member `t`'s verdict and its frame's `8B` bits, and step `t`'s ingress is bucket `t`'s `row` parameter. -/
def scanLayout (P : Flock.Program.Prog) (s : Schedule) : NetTiming.BoundaryGates 1 s.allocation :=
  let nodes := (P.defs[P.root]!).body.nodes
  let ts := NetworkCheck.Period.templateNodes P
  let ke := ts.getLast?.getD 0
  let le := (P.defs[nodes[ke]!.fn]!).outLeaves
  let bits := 8 * s.cellBytes
  { egress := fun t _ σ =>
      let leaf := fun j => gateAt P P.rootScope ⟨false, ke, t * le + j⟩
      (leaf 0, (List.range bits).map fun b => leaf (outBits + σ.val * bits + b))
    ingress := fun t =>
      (List.range (s.allocation * bits)).map fun j => gateAt P (P.child P.rootScope (ts.getD 1 0) t) ⟨true, 0, j⟩ }

/-- A statement's layout at the schedule `s`: its descriptor's scan layout, none where the descriptor doesn't decode. -/
def layOf (s : Schedule) (x : Claims.Sampling.AuditStatement) : NetTiming.BoundaryGates 1 s.allocation :=
  match Flock.Program.decode ⟨x.desc.toArray⟩ with
  | .ok P => scanLayout P s
  | .error _ => { egress := fun _ _ _ => (0, []), ingress := fun _ => [] }

/-- **An accepted scan's layout fits its Program flattened.** If `scanOf` accepts the scan `j` as the registered
Program `program` at the schedule `s`, its canonical text is `c` and decodes to `P`, then `P`'s scan layout fits every
flattening of `P` over the window's `T` steps: every egress gate is returned and not an input, every ingress gate is an
input, and no egress gate of a step depends on an ingress gate of a later step. -/
def ScanFits : Prop :=
  ∀ (j : Lean.Json) (program : String) (s : Schedule) (sc : Scan) (c : String) (P : Flock.Program.Prog)
    (C : Claims.Sampling.Prog),
    scanOf j program s = .ok sc → Flock.canon j = .ok c → Flock.Program.decode c.toUTF8 = .ok P → Flattens P C →
      (scanLayout P s).Fits C s.window

/-- **An accepted scan's statements are in the sampled audit's scope.** If `scanOf` accepts the scan `j` at the
schedule `s` and its canonical text is `c`, then at every compiled sampled verifier `D` whose statements are their
descriptors flattened (`StmtFlattens`), a statement `x` in its sampling scope (`Claims.SamplingGeneric.Compiled.Scope`),
every parse of which has `c`'s bytes as its descriptor, is in `ScopeAcc` at the layout `layOf s` and the horizon `T`. -/
def ScanScopeAcc : Prop :=
  ∀ (j : Lean.Json) (program : String) (s : Schedule) (sc : Scan) (c : String),
    scanOf j program s = .ok sc → Flock.canon j = .ok c →
    ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Claims.SamplingGeneric.Arg Coin}
      (D : Claims.SamplingGeneric.Compiled Coin A) (x : Claims.Msg),
      StmtFlattens D → D.Scope x →
      (∀ st, D.parse x = some st → (⟨st.desc.toArray⟩ : ByteArray) = c.toUTF8) →
      NetTiming.ScopeAcc D (layOf s) (fun _ => s.window) x

end Properties.NetworkScan
