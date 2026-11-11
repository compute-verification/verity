module

public import Properties.NetworkScan
public import Proofs.Properties.NetworkScan.Fits
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
import Mathlib.Tactic.ENatToNat
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
import Std.Tactic.BVDecide.Normalize.Prop

/-!
# An accepted scan's statements are in the sampled audit's scope, from its layout fitting

`scanScopeAcc_of_scanFits`: the sampling half is the scope check's (`ScopeMatches`), and the layout half is `ScanFits`
at the statement's descriptor, the scan's canonical bytes, flattened (`StmtFlattens`).

`scanFits_nontrivial` and `scanScopeAcc_nontrivial_of`: neither conclusion is trivial, at the empty Program and its
empty flattening (`empty_not_fits`).
-/

@[expose] public section

namespace Properties.NetworkScan

theorem scanScopeAcc_of_scanFits (h : ScanFits) : ScanScopeAcc := by
  intro Coin _ _ _ A D j program s sc c x hflat hmatch hsc hscan hc hdesc
  have hS : D.Scope x := (hmatch x).2 hsc
  refine ⟨hS, ?_⟩
  obtain ⟨st, _, hx, _⟩ := hS
  obtain ⟨P, hP, hF⟩ := hflat x st hx
  refine ⟨st, hx, ?_⟩
  unfold layOf
  rw [hP]
  rw [hdesc st hx] at hP
  exact h j program s sc c P st.prog hscan hc hP hF

/-- The empty Program's scan layout, at one slot and a window of one step, doesn't fit the empty flattening: step 0's
egress presence gate isn't returned. -/
theorem empty_not_fits :
    ¬ (scanLayout default ⟨0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1⟩).Fits ⟨[], []⟩ 1 := by
  rintro ⟨h, -, -⟩
  have := h 0 (by decide) 0 ⟨0, by decide⟩ _ (List.mem_cons.2 (Or.inl rfl))
  simp at this

/-- **`ScanFits`' conclusion is not trivial** (the two-lemma rule's `_nontrivial`): at a Program and a flattening of
it, built by hand, which a scan check accepting everything would pass, the layout doesn't fit. The Program has no
definitions, so its root is the default definition, of no gates and no return, and its flattening returns nothing
(`empty_not_fits`). The instance is the model's: no file decoding to it can be exhibited, since
`Flock.Program.decode` reads through `Lean.Json.parse`, a `partial def` the kernel can't unfold. -/
theorem scanFits_nontrivial :
    ∃ (P : Flock.Program.Prog) (C : Claims.Sampling.Prog) (s : NetworkCheck.Period.Schedule),
      Flattens P C ∧ ¬ (scanLayout P s).Fits C s.window := by
  refine ⟨default, ⟨[], []⟩, ⟨0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1⟩,
    ⟨rfl, fun g hg => absurd hg (Nat.not_lt_zero g), fun g => ⟨fun h => by simp at h, ?_⟩⟩, empty_not_fits⟩
  rintro ⟨j, hj, -⟩
  exact absurd hj (Nat.not_lt_zero j)

/-- **`ScanScopeAcc`'s conclusion is not trivial**, transferred from `scanFits_nontrivial`'s instance (the transfer
form of the two-lemma rule's `_nontrivial`): at any compiled sampled verifier, a statement that parses
to a descriptor decoding to the empty Program, with the empty circuit, is not in `ScopeAcc` at that Program's
schedule, though a scan check and a scope check accepting everything would pass it. -/
theorem scanScopeAcc_nontrivial_of {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    {A : Claims.SamplingGeneric.Arg Coin} {D : Claims.SamplingGeneric.Compiled Coin A} {x : Claims.Msg}
    {st : Claims.Sampling.AuditStatement} (hx : D.parse x = some st)
    (hP : Flock.Program.decode ⟨st.desc.toArray⟩ = .ok default) (hC : st.prog = ⟨[], []⟩) :
    ¬ NetTiming.ScopeAcc D (layOf ⟨0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1⟩) (fun _ => 1) x := by
  rintro ⟨-, st', hx', hF⟩
  obtain rfl := Option.some.inj (hx.symm.trans hx')
  unfold layOf at hF
  rw [hP, hC] at hF
  exact empty_not_fits hF

end Properties.NetworkScan
