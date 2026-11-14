module

public import Properties.NetworkScan
public import Proofs.Properties.NetworkScan.Fits
public import Proofs.Properties.NetworkScan.Causal
public import Proofs.Flock.Claims.Witnesses.SamplingGeneric
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

`scanFits`: `NetTiming.BoundaryGates.Fits`' parts (1) and (2) from what `scanOf`'s acceptance gives the layout
(`Fits.scanFits_egress_ingress`), and part (3), causality, from the walk lemma (`CausalProof.scanCausal`) at the
Program `scanOf` checks topological (`Fits.Accepted.topo`).

`scanScopeAcc_of_scanFits`, and so `scanScopeAcc`: the sampling half is the scope check's (`ScopeMatches`), and the
layout half is `ScanFits` at the statement's descriptor, the scan's canonical bytes, flattened (`StmtFlattens`).

The witnesses (the two-lemma rule): `scan_honest` exhibits what both take that no named assumption supplies, and
`scanFits_nontrivial` and `scanScopeAcc_nontrivial` show neither conclusion is trivial, at a circuit that returns
nothing (`not_fits_of_no_outs`). `scanScopeAcc_nontrivial_of` is the second's transfer form, at any compiled sampled
verifier.
-/

@[expose] public section

namespace Properties.NetworkScan

theorem scanScopeAcc_of_scanFits (h : ScanFits) : ScanScopeAcc := by
  intro j program s sc c hscan hc Coin _ _ _ A D x hflat hmatch hsc hdesc
  have hm : Claims.ScopeMatches D.Scope D.scopeCheck := hmatch
  have hS : D.Scope x := (hm x).2 hsc
  refine ⟨hS, ?_⟩
  obtain ⟨st, _, hx, _⟩ := hS
  obtain ⟨P, hP, hF⟩ := hflat x st hx
  refine ⟨st, hx, ?_⟩
  unfold layOf
  rw [hP]
  rw [hdesc st hx] at hP
  exact h j program s sc c P st.prog hscan hc hP hF

/-- **An accepted scan's layout fits its Program flattened.** -/
theorem scanFits : ScanFits := by
  intro j program s sc c P C h hc hP hF
  obtain ⟨he, hi⟩ := Fits.scanFits_egress_ingress h hc hP hF
  exact ⟨he, hi, CausalProof.scanCausal j program s sc c P C h hc hP (Fits.scanOf_accepted h hc hP).topo hF⟩

/-- **An accepted scan's statements are in the sampled audit's scope.** -/
theorem scanScopeAcc : ScanScopeAcc := scanScopeAcc_of_scanFits scanFits

/-- One slot of one lane and two ticks (T = 2): the schedule of the nontrivial instances. -/
def oneSlot : NetworkCheck.Period.Schedule := ⟨0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 2, none⟩

/-- A circuit that returns nothing fits no layout over a horizon of at least one step: step 0's egress presence gate
isn't returned. -/
theorem not_fits_of_no_outs {L S : ℕ} (lay : NetTiming.BoundaryGates L S) (gs : List Claims.Sampling.BGate) {k : ℕ}
    (hk : 0 < k) (ℓ : Fin L) (σ : Fin S) : ¬ lay.Fits ⟨gs, []⟩ k := by
  rintro ⟨h, -, -⟩
  have := h 0 hk ℓ σ _ (List.mem_cons.2 (Or.inl rfl))
  simp at this

/-- The Program with no definitions, whose root is the default definition of no gates and no return, flattens to the
empty circuit. -/
theorem empty_flattens : Flattens default ⟨[], []⟩ :=
  ⟨rfl, fun g hg => absurd hg (Nat.not_lt_zero g),
    fun g => ⟨fun h => by simp at h, fun ⟨j, hj, _⟩ => absurd hj (Nat.not_lt_zero j)⟩⟩

theorem canon_null : Flock.canon .null = .ok "null" := by
  simp only [Flock.canon]
  rfl

/-- **`ScanFits`' and `ScanScopeAcc`'s honest witness** (the two-lemma rule's `_honest`): of what the two take,
Flock's canonical form accepts a scan (`Flock.canon` at `null`), and a Program has a flattening (`Flattens`, at the
Program with no definitions). The compiled sampled verifier's premises are named assumptions
(`Properties.NetworkScan.Assumptions`).
`scanOf`'s and `Flock.Program.decode`'s acceptances are not exhibited: both read `Lean.Json.parse`, a `partial def`
the kernel can't unfold. -/
theorem scan_honest : Flock.canon .null = .ok "null" ∧ Flattens default ⟨[], []⟩ := ⟨canon_null, empty_flattens⟩

/-- **`ScanFits`' conclusion is not trivial** (the two-lemma rule's `_nontrivial`): at a Program and a flattening of
it, built by hand, which a scan check accepting everything would pass, the layout doesn't fit. The Program has no
definitions, so its flattening returns nothing (`empty_flattens`, `not_fits_of_no_outs`). The instance is the model's:
no file decoding to it can be exhibited, since `Flock.Program.decode` reads through `Lean.Json.parse`, a `partial def`
the kernel can't unfold. -/
theorem scanFits_nontrivial :
    ∃ (P : Flock.Program.Prog) (C : Claims.Sampling.Prog) (s : NetworkCheck.Period.Schedule) (_ : Flattens P C),
      ¬ (scanLayout P s).Fits C s.window :=
  ⟨default, ⟨[], []⟩, oneSlot, empty_flattens, not_fits_of_no_outs _ [] (by decide) 0 ⟨0, by decide⟩⟩

theorem bytes_of_data_toList (b : ByteArray) : (⟨b.data.toList.toArray⟩ : ByteArray) = b := rfl

/-- The statement whose descriptor is the bytes of `null` and whose circuit is empty. -/
def nullStmt : Claims.Sampling.AuditStatement :=
  { Claims.Witness.Sampling.toyStmt with desc := "null".toUTF8.data.toList, prog := ⟨[], []⟩ }

/-- A compiled sampled verifier whose scope check takes everything and which parses every message to `nullStmt`: the
toy verifier `Claims.Witness.SamplingGeneric.sksL` with that parser. -/
noncomputable def acceptAllCompiled :
    Claims.SamplingGeneric.Compiled Bool (Claims.Witness.SamplingGeneric.toyA Bool) :=
  { Claims.Witness.SamplingGeneric.sksL Bool with parse := fun _ => some nullStmt }

/-- **`ScanScopeAcc`'s conclusion is not trivial** (the two-lemma rule's `_nontrivial`): at a scan Flock
canonicalizes as `null` and a compiled sampled verifier whose scope check takes everything and whose statements have
that text as their descriptor and the empty circuit, a statement is not in `ScopeAcc` at a schedule (`oneSlot`): its
circuit returns nothing (`not_fits_of_no_outs`). -/
theorem scanScopeAcc_nontrivial :
    ∃ (s : NetworkCheck.Period.Schedule) (j : Lean.Json) (c : String) (_ : Flock.canon j = .ok c)
      (A : Claims.SamplingGeneric.Arg Bool) (D : Claims.SamplingGeneric.Compiled Bool A) (x : Claims.Msg)
      (_ : D.scopeCheck x = true)
      (_ : ∀ st, D.parse x = some st → (⟨st.desc.toArray⟩ : ByteArray) = c.toUTF8),
      ¬ NetTiming.ScopeAcc D (layOf s) (fun _ => s.window) x := by
  refine ⟨oneSlot, .null, "null", canon_null, _, acceptAllCompiled, [], rfl, fun st hx => ?_, ?_⟩
  · cases (Option.some.inj hx : nullStmt = st)
    exact bytes_of_data_toList "null".toUTF8
  · rintro ⟨-, st, hx, hF⟩
    cases (Option.some.inj hx : nullStmt = st)
    exact not_fits_of_no_outs _ [] (by decide) 0 ⟨0, by decide⟩ hF

/-- **`ScanScopeAcc`'s conclusion is not trivial**, transferred from `scanFits_nontrivial`'s instance (the transfer
form of the two-lemma rule's `_nontrivial`): at any compiled sampled verifier, a statement that parses
to a descriptor decoding to the empty Program, with the empty circuit, is not in `ScopeAcc` at that Program's
schedule, though a scan check and a scope check accepting everything would pass it. -/
theorem scanScopeAcc_nontrivial_of {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    {A : Claims.SamplingGeneric.Arg Coin} {D : Claims.SamplingGeneric.Compiled Coin A} {x : Claims.Msg}
    {st : Claims.Sampling.AuditStatement} (hx : D.parse x = some st)
    (hP : Flock.Program.decode ⟨st.desc.toArray⟩ = .ok default) (hC : st.prog = ⟨[], []⟩) :
    ¬ NetTiming.ScopeAcc D (layOf oneSlot) (fun _ => oneSlot.window) x := by
  rintro ⟨-, st', hx', hF⟩
  obtain rfl := Option.some.inj (hx.symm.trans hx')
  unfold layOf at hF
  rw [hP, hC] at hF
  exact not_fits_of_no_outs _ [] (show 0 < oneSlot.window by decide) 0 ⟨0, by decide⟩ hF

end Properties.NetworkScan
